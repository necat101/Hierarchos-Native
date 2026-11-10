//! Phase XVIII mixed-WMMA drift diagnostic harness.
//!
//! This binary exists to separate three things that the Phase XVII final
//! comparison could not: forward inference error at identical weights,
//! one-step backward/AdamW update divergence, and accumulated trajectory
//! divergence. It is observation-only: it calls the same public
//! `VulkanTransformer` training and evaluation entry points the qualified
//! harnesses use, records every step, and writes exact parameter snapshots and
//! weights-only HF checkpoints for offline comparison.
//!
//! Nothing here changes a kernel, shader, optimizer rule or tolerance. The
//! FP32 master weights stay authoritative; the only mode switch is the
//! existing process-wide `HIERARCHOS_WMMA_MODE` environment variable that
//! `WmmaMode::from_env` already implements.
//!
//! A parameter snapshot is raw FP32 master bytes plus an index. Exporting a
//! weights-only package is explicitly *not* an exact training-resume
//! checkpoint: AdamW moments, the optimizer step and RNG state are not part of
//! an HF package. The report labels it as such.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    wmma::WmmaMode, wmma_linear, AdamWHyperParams, VulkanDevice, VulkanTransformer,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
struct Window {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    valid_tokens: usize,
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    train: PathBuf,
    validation: PathBuf,
    test: Option<PathBuf>,
    report: PathBuf,
    dump_dir: Option<PathBuf>,
    export_root: Option<PathBuf>,
    batch: usize,
    seq: usize,
    steps: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    seed: u64,
    order_offset: usize,
    eval_steps: BTreeSet<usize>,
    dump_steps: BTreeSet<usize>,
    export_steps: BTreeSet<usize>,
    final_test: bool,
    compare_forward: bool,
    dump_logits: Option<PathBuf>,
    partition: String,
}

fn parse_steps(value: &str) -> Result<BTreeSet<usize>> {
    let mut steps = BTreeSet::new();
    for part in value.split(',').map(str::trim).filter(|part| !part.is_empty()) {
        steps.insert(part.parse().with_context(|| format!("invalid step {part:?}"))?);
    }
    Ok(steps)
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            model: PathBuf::new(),
            train: PathBuf::new(),
            validation: PathBuf::new(),
            test: None,
            report: PathBuf::new(),
            dump_dir: None,
            export_root: None,
            batch: 8,
            seq: 64,
            steps: 100,
            lr: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
            seed: 7,
            order_offset: 0,
            eval_steps: BTreeSet::from([0, 1, 2, 5, 10, 20, 40, 60, 80, 100]),
            dump_steps: BTreeSet::new(),
            export_steps: BTreeSet::new(),
            final_test: true,
            compare_forward: false,
            dump_logits: None,
            partition: "validation".to_owned(),
        };
        let mut rest = std::env::args().skip(1);
        while let Some(arg) = rest.next() {
            let mut value = || {
                rest.next()
                    .with_context(|| format!("missing value for {arg}"))
            };
            match arg.as_str() {
                "--model" => args.model = PathBuf::from(value()?),
                "--train" => args.train = PathBuf::from(value()?),
                "--validation" => args.validation = PathBuf::from(value()?),
                "--test" => args.test = Some(PathBuf::from(value()?)),
                "--report" => args.report = PathBuf::from(value()?),
                "--dump-dir" => args.dump_dir = Some(PathBuf::from(value()?)),
                "--export-root" => args.export_root = Some(PathBuf::from(value()?)),
                "--batch" => args.batch = value()?.parse()?,
                "--seq" => args.seq = value()?.parse()?,
                "--steps" => args.steps = value()?.parse()?,
                "--lr" => args.lr = value()?.parse()?,
                "--beta1" => args.beta1 = value()?.parse()?,
                "--beta2" => args.beta2 = value()?.parse()?,
                "--eps" => args.eps = value()?.parse()?,
                "--weight-decay" => args.weight_decay = value()?.parse()?,
                "--seed" => args.seed = value()?.parse()?,
                "--order-offset" => args.order_offset = value()?.parse()?,
                "--eval-at" => args.eval_steps = parse_steps(&value()?)?,
                "--dump-at" => args.dump_steps = parse_steps(&value()?)?,
                "--export-at" => args.export_steps = parse_steps(&value()?)?,
                "--no-final-test" => args.final_test = false,
                "--compare-forward" => args.compare_forward = true,
                "--dump-logits" => args.dump_logits = Some(PathBuf::from(value()?)),
                "--partition" => args.partition = value()?,
                other => bail!("unknown argument {other}"),
            }
        }
        if args.model.as_os_str().is_empty() || args.report.as_os_str().is_empty() {
            bail!("--model and --report are required");
        }
        if !args.compare_forward
            && args.dump_logits.is_none()
            && (args.train.as_os_str().is_empty() || args.validation.as_os_str().is_empty())
        {
            bail!("training mode requires --train and --validation");
        }
        if args.dump_logits.is_some() && args.partition == "validation" && args.validation.as_os_str().is_empty() {
            bail!("--dump-logits requires --validation for the validation partition");
        }
        if args.compare_forward && args.validation.as_os_str().is_empty() {
            bail!("--compare-forward requires --validation");
        }
        if args.dump_logits.is_some() {
            if !matches!(args.partition.as_str(), "validation" | "test") {
                bail!("--partition must be validation or test");
            }
            if args.partition == "test" && args.test.is_none() {
                bail!("--partition test requires --test");
            }
        }
        if args.batch == 0 || args.seq == 0 || args.steps == 0 {
            bail!("--batch, --seq and --steps must be positive");
        }
        if !args.dump_steps.is_empty() && args.dump_dir.is_none() {
            bail!("--dump-at requires --dump-dir");
        }
        if !args.export_steps.is_empty() && args.export_root.is_none() {
            bail!("--export-at requires --export-root");
        }
        Ok(args)
    }
}

fn load_windows(path: &Path, seq_len: usize) -> Result<Vec<Window>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut documents = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading {} line {}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)?;
        let ids = value
            .get("input_ids")
            .and_then(serde_json::Value::as_array)
            .context("partition rows require input_ids")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|number| u32::try_from(number).ok())
                    .context("input_ids must be u32")
            })
            .collect::<Result<Vec<u32>>>()?;
        documents.push(ids);
    }
    let mut windows = Vec::new();
    for ids in documents {
        if ids.len() < 2 {
            continue;
        }
        let predict = ids.len() - 1;
        for start in (0..predict).step_by(seq_len) {
            let count = (predict - start).min(seq_len);
            windows.push(Window {
                input_ids: ids[start..start + count].to_vec(),
                targets: ids[start + 1..start + 1 + count].to_vec(),
                valid_tokens: count,
            });
        }
    }
    if windows.is_empty() {
        bail!("partition {} produced no windows", path.display());
    }
    Ok(windows)
}

fn batch_arrays(windows: &[Window], seq_len: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut input_ids = Vec::with_capacity(windows.len() * seq_len);
    let mut targets = Vec::with_capacity(windows.len() * seq_len);
    let mut attention_mask = Vec::with_capacity(windows.len() * seq_len);
    let mut loss_weights = Vec::with_capacity(windows.len() * seq_len);
    for window in windows {
        let mut ids = window.input_ids.clone();
        ids.resize(seq_len, 0);
        let mut row_targets = window.targets.clone();
        row_targets.resize(seq_len, u32::MAX);
        input_ids.extend_from_slice(&ids);
        targets.extend_from_slice(&row_targets);
        attention_mask.extend(std::iter::repeat(1.0).take(window.valid_tokens));
        attention_mask.extend(std::iter::repeat(0.0).take(seq_len - window.valid_tokens));
        loss_weights.extend(std::iter::repeat(1.0).take(window.valid_tokens));
        loss_weights.extend(std::iter::repeat(0.0).take(seq_len - window.valid_tokens));
    }
    let normalization = loss_weights.iter().copied().sum::<f32>();
    if normalization > 0.0 {
        for weight in &mut loss_weights {
            *weight /= normalization;
        }
    }
    (input_ids, targets, attention_mask, loss_weights)
}

fn digest_parameters(values: &std::collections::BTreeMap<String, Vec<f32>>) -> (String, usize, usize) {
    let mut hash = Sha256::new();
    let mut elements = 0usize;
    let mut nonfinite = 0usize;
    for (name, tensor) in values {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((tensor.len() as u64).to_le_bytes());
        for value in tensor {
            hash.update(value.to_bits().to_le_bytes());
            elements += 1;
            if !value.is_finite() {
                nonfinite += 1;
            }
        }
    }
    (format!("{:x}", hash.finalize()), elements, nonfinite)
}

#[derive(Serialize)]
struct Evaluation {
    after_step: usize,
    partition: String,
    examples: usize,
    valid_target_tokens: usize,
    cross_entropy: f64,
    perplexity: f64,
    nonfinite_logits: usize,
    eval_ms: f64,
    parameter_digest_before: String,
    parameter_digest_after: String,
    wmma_forward_delta: u64,
    wmma_weight_grad_delta: u64,
    wmma_input_grad_delta: u64,
    /// Per-window summed cross-entropy in window order. A checker re-adds
    /// these and divides by `valid_target_tokens`, so the aggregate is not a
    /// trusted scalar.
    loss_sums: Vec<f64>,
    window_valid_tokens: Vec<u32>,
    stream_sha256: String,
}

fn evaluate(
    graph: &VulkanTransformer,
    windows: &[Window],
    batch: usize,
    seq: usize,
    partition: &str,
    after_step: usize,
) -> Result<Evaluation> {
    let (digest_before, _, _) = digest_parameters(&graph.parameter_values()?);
    let phases_before = wmma_linear::linear_wmma_phase_counts();
    let started = Instant::now();
    let vocab = graph.config().vocab_size;
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    let mut nonfinite_logits = 0usize;
    let mut loss_sums = Vec::with_capacity(windows.len());
    let mut window_valid_tokens = Vec::with_capacity(windows.len());
    for chunk in windows.chunks(batch) {
        let (mut input_ids, _, mut attention_mask, _) = batch_arrays(chunk, seq);
        while input_ids.len() < batch * seq {
            let tail = input_ids[input_ids.len() - seq..].to_vec();
            input_ids.extend_from_slice(&tail);
            attention_mask.extend(std::iter::repeat(0.0).take(seq));
        }
        let logits = graph.forward_logits(&input_ids, &attention_mask)?;
        if logits.len() != batch * seq * vocab {
            bail!("forward_logits returned {} values, expected {}", logits.len(), batch * seq * vocab);
        }
        for (index, window) in chunk.iter().enumerate() {
            let mut loss = 0.0f64;
            for position in 0..window.valid_tokens {
                let target = window.targets[position] as usize;
                let base = (index * seq + position) * vocab;
                let row = &logits[base..base + vocab];
                let mut max = f64::NEG_INFINITY;
                for value in row {
                    if value.is_finite() {
                        max = max.max(f64::from(*value));
                    } else {
                        nonfinite_logits += 1;
                    }
                }
                if !max.is_finite() {
                    loss = f64::INFINITY;
                    break;
                }
                let mut sum = 0.0f64;
                for value in row {
                    sum += (f64::from(*value) - max).exp();
                }
                if sum <= 0.0 || !sum.is_finite() {
                    loss = f64::INFINITY;
                    break;
                }
                loss += (max + sum.ln()) - f64::from(row[target]);
            }
            total_tokens += window.valid_tokens;
            total_loss += loss;
            loss_sums.push(loss);
            window_valid_tokens.push(window.valid_tokens as u32);
        }
    }
    let eval_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (digest_after, _, _) = digest_parameters(&graph.parameter_values()?);
    let phases_after = wmma_linear::linear_wmma_phase_counts();
    let cross_entropy = total_loss / total_tokens as f64;
    Ok(Evaluation {
        after_step,
        partition: partition.to_owned(),
        examples: windows.len(),
        valid_target_tokens: total_tokens,
        cross_entropy,
        perplexity: cross_entropy.exp(),
        nonfinite_logits,
        eval_ms,
        parameter_digest_before: digest_before,
        parameter_digest_after: digest_after,
        wmma_forward_delta: (phases_after[0] - phases_before[0]) as u64,
        wmma_weight_grad_delta: (phases_after[1] - phases_before[1]) as u64,
        wmma_input_grad_delta: (phases_after[2] - phases_before[2]) as u64,
        loss_sums,
        window_valid_tokens,
        stream_sha256: window_stream_digest(windows),
    })
}

/// Bind the exact window stream (ids, causal targets, valid counts) an
/// evaluation consumed. The checker re-derives it from the committed partition
/// files, so a re-windowed corpus or shifted causal alignment cannot hide
/// behind a plausible loss.
fn window_stream_digest(windows: &[Window]) -> String {
    let mut hash = Sha256::new();
    for window in windows {
        hash.update((window.valid_tokens as u64).to_le_bytes());
        for value in &window.input_ids {
            hash.update(value.to_le_bytes());
        }
        for value in &window.targets {
            hash.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

#[derive(Serialize)]
struct StepRecord {
    global_step: usize,
    optimizer_step: u32,
    loss: f32,
    loss_bits: u32,
    supervised_tokens: usize,
    param_digest: String,
    step_ms: f64,
    wmma_forward: u64,
    wmma_weight_grad: u64,
    wmma_input_grad: u64,
    wmma_cooperative: u64,
    wmma_shape_fallbacks: u64,
    weight_conversions: u64,
    mirror_reuses: u64,
}

#[derive(Serialize)]
struct Dump {
    step: usize,
    binary: String,
    index: String,
    parameter_digest: String,
}

#[derive(Serialize)]
struct Export {
    step: usize,
    directory: String,
    parameter_digest: String,
    exact_training_resume: bool,
}

#[derive(Serialize)]
struct Report {
    device: String,
    mode: &'static str,
    two_pass: bool,
    model: String,
    batch: usize,
    seq: usize,
    steps: usize,
    seed: u64,
    order_offset: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    parameter_elements: usize,
    parameter_start_digest: String,
    parameter_end_digest: String,
    parameter_nonfinite: usize,
    optimizer_step: u32,
    dropout_seed: u32,
    train_windows: usize,
    validation_windows: usize,
    test_windows: Option<usize>,
    partition_stream_sha256: BTreeMap<String, String>,
    records: Vec<StepRecord>,
    evaluations: Vec<Evaluation>,
    dumps: Vec<Dump>,
    exports: Vec<Export>,
    total_wall_ms: f64,
}

fn dump_parameters(
    graph: &VulkanTransformer,
    dir: &Path,
    step: usize,
) -> Result<(Dump, String)> {
    let values = graph.parameter_values()?;
    let (digest, _, _) = digest_parameters(&values);
    let mut blob = Vec::new();
    let mut index = Vec::new();
    for (name, tensor) in &values {
        let offset = blob.len() / 4;
        for value in tensor {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        index.push(serde_json::json!({"name": name, "offset": offset, "len": tensor.len()}));
    }
    fs::create_dir_all(dir)?;
    let binary = dir.join(format!("params-{step:04}.bin"));
    let index_path = dir.join(format!("params-{step:04}.json"));
    fs::write(&binary, &blob)?;
    fs::write(&index_path, serde_json::to_vec(&index)?)?;
    Ok((
        Dump {
            step,
            binary: binary.display().to_string(),
            index: index_path.display().to_string(),
            parameter_digest: digest.clone(),
        },
        digest,
    ))
}

#[derive(Serialize)]
struct ForwardComparison {
    model: String,
    parameter_digest_off: String,
    parameter_digest_mixed: String,
    parameter_digest_equal: bool,
    windows: usize,
    valid_target_tokens: usize,
    logit_elements: usize,
    max_abs: f64,
    rms: f64,
    mean_abs: f64,
    abs_over_1e_3: usize,
    abs_over_1e_2: usize,
    nonfinite_off: usize,
    nonfinite_mixed: usize,
    cross_entropy_off: f64,
    cross_entropy_mixed: f64,
    cross_entropy_delta: f64,
    wmma_forward_dispatches: u64,
    wmma_shape_fallbacks: u64,
    elapsed_ms: f64,
}

fn row_cross_entropy(row: &[f32], target: usize) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for value in row {
        if value.is_finite() {
            max = max.max(f64::from(*value));
        }
    }
    if !max.is_finite() {
        return f64::INFINITY;
    }
    let mut sum = 0.0f64;
    for value in row {
        sum += (f64::from(*value) - max).exp();
    }
    if sum <= 0.0 || !sum.is_finite() {
        return f64::INFINITY;
    }
    (max + sum.ln()) - f64::from(row[target])
}

/// Execute the same weights and inputs through an FP32 graph and a mixed-WMMA
/// graph in one process, so the reported difference is inference arithmetic
/// only. Both graphs load the same package and their parameter digests must be
/// byte-identical before any comparison is trusted.
fn run_compare_forward(args: &Args) -> Result<()> {
    let started = Instant::now();
    let off = VulkanTransformer::from_hf_package(
        VulkanDevice::new_for_wmma(WmmaMode::Off)?,
        &args.model,
        args.batch,
        args.seq,
    )?;
    let mixed = VulkanTransformer::from_hf_package(
        VulkanDevice::new_for_wmma(WmmaMode::Mixed)?,
        &args.model,
        args.batch,
        args.seq,
    )?;
    let (digest_off, _, _) = digest_parameters(&off.parameter_values()?);
    let (digest_mixed, _, _) = digest_parameters(&mixed.parameter_values()?);
    if digest_off != digest_mixed {
        bail!("the two comparison graphs do not hold identical weights");
    }
    let windows = load_windows(&args.validation, args.seq)?;
    let vocab = off.config().vocab_size;
    let phases_before = wmma_linear::linear_wmma_phase_counts();
    let counts_before = wmma_linear::linear_wmma_dispatch_counts();
    let mut logit_elements = 0usize;
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut abs_over_1e_3 = 0usize;
    let mut abs_over_1e_2 = 0usize;
    let mut nonfinite_off = 0usize;
    let mut nonfinite_mixed = 0usize;
    let mut loss_off = 0.0f64;
    let mut loss_mixed = 0.0f64;
    let mut total_tokens = 0usize;
    for chunk in windows.chunks(args.batch) {
        let (mut input_ids, _, mut attention_mask, _) = batch_arrays(chunk, args.seq);
        while input_ids.len() < args.batch * args.seq {
            let tail = input_ids[input_ids.len() - args.seq..].to_vec();
            input_ids.extend_from_slice(&tail);
            attention_mask.extend(std::iter::repeat(0.0).take(args.seq));
        }
        let logits_off = off.forward_logits(&input_ids, &attention_mask)?;
        let logits_mixed = mixed.forward_logits(&input_ids, &attention_mask)?;
        if logits_off.len() != logits_mixed.len() || logits_off.len() != args.batch * args.seq * vocab {
            bail!("logit geometry mismatch between the two graphs");
        }
        for (a, b) in logits_off.iter().zip(logits_mixed.iter()) {
            if !a.is_finite() {
                nonfinite_off += 1;
            }
            if !b.is_finite() {
                nonfinite_mixed += 1;
            }
            let delta = f64::from(*a) - f64::from(*b);
            let abs = delta.abs();
            if abs.is_finite() {
                max_abs = max_abs.max(abs);
                sum_sq += delta * delta;
                sum_abs += abs;
                if abs > 1.0e-3 {
                    abs_over_1e_3 += 1;
                }
                if abs > 1.0e-2 {
                    abs_over_1e_2 += 1;
                }
            }
            logit_elements += 1;
        }
        for (index, window) in chunk.iter().enumerate() {
            for position in 0..window.valid_tokens {
                let target = window.targets[position] as usize;
                let base = (index * args.seq + position) * vocab;
                loss_off += row_cross_entropy(&logits_off[base..base + vocab], target);
                loss_mixed += row_cross_entropy(&logits_mixed[base..base + vocab], target);
                total_tokens += 1;
            }
        }
    }
    let phases_after = wmma_linear::linear_wmma_phase_counts();
    let counts_after = wmma_linear::linear_wmma_dispatch_counts();
    let cross_entropy_off = loss_off / total_tokens as f64;
    let cross_entropy_mixed = loss_mixed / total_tokens as f64;
    let comparison = ForwardComparison {
        model: args.model.display().to_string(),
        parameter_digest_off: digest_off,
        parameter_digest_mixed: digest_mixed,
        parameter_digest_equal: true,
        windows: windows.len(),
        valid_target_tokens: total_tokens,
        logit_elements,
        max_abs,
        rms: (sum_sq / logit_elements as f64).sqrt(),
        mean_abs: sum_abs / logit_elements as f64,
        abs_over_1e_3,
        abs_over_1e_2,
        nonfinite_off,
        nonfinite_mixed,
        cross_entropy_off,
        cross_entropy_mixed,
        cross_entropy_delta: cross_entropy_mixed - cross_entropy_off,
        wmma_forward_dispatches: (phases_after[0] - phases_before[0]) as u64,
        wmma_shape_fallbacks: counts_after
            .shape_fallbacks
            .saturating_sub(counts_before.shape_fallbacks),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.report, serde_json::to_vec_pretty(&comparison)?)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "max_abs": comparison.max_abs,
            "rms": comparison.rms,
            "cross_entropy_off": comparison.cross_entropy_off,
            "cross_entropy_mixed": comparison.cross_entropy_mixed,
            "cross_entropy_delta": comparison.cross_entropy_delta,
        }))?
    );
    Ok(())
}

#[derive(Serialize)]
struct LogitDump {
    model: String,
    mode: &'static str,
    partition: String,
    windows: usize,
    seq: usize,
    vocab: usize,
    logit_elements: usize,
    valid_target_tokens: usize,
    cross_entropy: f64,
    parameter_digest: String,
    wmma_forward_dispatches: u64,
    wmma_shape_fallbacks: u64,
    elapsed_ms: f64,
}

/// Run one precision mode over one partition and write its exact FP32 logits.
///
/// The execution mode comes from `HIERARCHOS_WMMA_MODE`, exactly as in the
/// qualified harnesses. Two separate processes (one per mode) are required
/// because the graph captures the mode at construction, which is why an
/// in-process dual-graph comparison would silently run FP32 twice.
fn run_dump_logits(args: &Args, path: &Path) -> Result<()> {
    let started = Instant::now();
    let mode = WmmaMode::from_env()?;
    let graph = VulkanTransformer::from_hf_package(
        VulkanDevice::new_for_wmma(mode)?,
        &args.model,
        args.batch,
        args.seq,
    )?;
    let windows = if args.partition == "test" {
        load_windows(args.test.as_deref().context("--partition test requires --test")?, args.seq)?
    } else {
        load_windows(&args.validation, args.seq)?
    };
    let (digest, _, _) = digest_parameters(&graph.parameter_values()?);
    let phases_before = wmma_linear::linear_wmma_phase_counts();
    let counts_before = wmma_linear::linear_wmma_dispatch_counts();
    let vocab = graph.config().vocab_size;
    let mut bytes = Vec::new();
    let mut loss = 0.0f64;
    let mut tokens = 0usize;
    for chunk in windows.chunks(args.batch) {
        let (mut input_ids, _, mut attention_mask, _) = batch_arrays(chunk, args.seq);
        while input_ids.len() < args.batch * args.seq {
            let tail = input_ids[input_ids.len() - args.seq..].to_vec();
            input_ids.extend_from_slice(&tail);
            attention_mask.extend(std::iter::repeat(0.0).take(args.seq));
        }
        let logits = graph.forward_logits(&input_ids, &attention_mask)?;
        if logits.len() != args.batch * args.seq * vocab {
            bail!("logit geometry mismatch in --dump-logits");
        }
        for value in &logits {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for (index, window) in chunk.iter().enumerate() {
            for position in 0..window.valid_tokens {
                let target = window.targets[position] as usize;
                let base = (index * args.seq + position) * vocab;
                loss += row_cross_entropy(&logits[base..base + vocab], target);
                tokens += 1;
            }
        }
    }
    let phases_after = wmma_linear::linear_wmma_phase_counts();
    let counts_after = wmma_linear::linear_wmma_dispatch_counts();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, &bytes)?;
    let dump = LogitDump {
        model: args.model.display().to_string(),
        mode: mode.label(),
        partition: args.partition.clone(),
        windows: windows.len(),
        seq: args.seq,
        vocab,
        logit_elements: bytes.len() / 4,
        valid_target_tokens: tokens,
        cross_entropy: loss / tokens as f64,
        parameter_digest: digest,
        wmma_forward_dispatches: (phases_after[0] - phases_before[0]) as u64,
        wmma_shape_fallbacks: counts_after
            .shape_fallbacks
            .saturating_sub(counts_before.shape_fallbacks),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    let mut sidecar = path.to_path_buf();
    sidecar.set_extension("json");
    fs::write(&sidecar, serde_json::to_vec_pretty(&dump)?)?;
    println!("{}", serde_json::to_string(&serde_json::json!({
        "mode": dump.mode,
        "partition": dump.partition,
        "cross_entropy": dump.cross_entropy,
        "wmma_forward_dispatches": dump.wmma_forward_dispatches,
        "wmma_shape_fallbacks": dump.wmma_shape_fallbacks,
    }))?);
    Ok(())
}

fn run(args: Args) -> Result<()> {
    if let Some(path) = args.dump_logits.clone() {
        return run_dump_logits(&args, &path);
    }
    if args.compare_forward {
        return run_compare_forward(&args);
    }
    let mode = WmmaMode::from_env()?;
    let two_pass = std::env::var("HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS")
        .map(|value| value == "1")
        .unwrap_or(false);
    let device = VulkanDevice::new_for_wmma(mode)?;
    let mut graph = VulkanTransformer::from_hf_package(device, &args.model, args.batch, args.seq)?;
    graph.set_dropout_seed(args.seed as u32 ^ (args.seed >> 32) as u32);
    let hyper = AdamWHyperParams {
        lr: args.lr,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.eps,
        weight_decay: args.weight_decay,
    };

    let mut train_windows = load_windows(&args.train, args.seq)?;
    if args.order_offset > 0 {
        let shift = args.order_offset % train_windows.len();
        train_windows.rotate_left(shift);
    }
    let validation_windows = load_windows(&args.validation, args.seq)?;
    let test_windows = args
        .test
        .as_ref()
        .map(|path| load_windows(path, args.seq))
        .transpose()?;
    let batches_per_epoch = train_windows.len() / args.batch;
    if batches_per_epoch == 0 {
        bail!("training partition has fewer windows than one batch");
    }

    let start_values = graph.parameter_values()?;
    let (start_digest, elements, nonfinite) = digest_parameters(&start_values);
    drop(start_values);
    let mut records = Vec::new();
    let mut evaluations = Vec::new();
    let mut dumps = Vec::new();
    let mut exports = Vec::new();
    let mut total_wall_ms = 0.0f64;

    if args.dump_steps.contains(&0) {
        let (dump, _) = dump_parameters(&graph, args.dump_dir.as_deref().unwrap(), 0)?;
        dumps.push(dump);
    }
    if args.eval_steps.contains(&0) {
        evaluations.push(evaluate(&graph, &validation_windows, args.batch, args.seq, "validation", 0)?);
    }
    if args.final_test && args.eval_steps.contains(&0) {
        if let Some(windows) = test_windows.as_ref() {
            evaluations.push(evaluate(&graph, windows, args.batch, args.seq, "test", 0)?);
        }
    }    for index in 0..args.steps {
        let batch_index = index % batches_per_epoch;
        let begin = batch_index * args.batch;
        let (input_ids, targets, attention_mask, loss_weights) =
            batch_arrays(&train_windows[begin..begin + args.batch], args.seq);
        // Capture the counters immediately before the step. Evaluations run
        // between steps, so a delta taken from the previous record would
        // attribute the evaluation's forward dispatches to this training step.
        let step_phases = wmma_linear::linear_wmma_phase_counts();
        let step_counts = wmma_linear::linear_wmma_dispatch_counts();
        let step_conversions = wmma_linear::linear_wmma_conversion_stats();
        let step_started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
        total_wall_ms += step_ms;
        if result.step as usize != index + 1 {
            bail!("optimizer step divergence at step {}: got {}", index + 1, result.step);
        }
        if !result.loss.is_finite() {
            bail!("non-finite training loss at step {}", result.step);
        }
        let phases = wmma_linear::linear_wmma_phase_counts();
        let counts = wmma_linear::linear_wmma_dispatch_counts();
        let conversions = wmma_linear::linear_wmma_conversion_stats();
        let (param_digest, _, _) = digest_parameters(&graph.parameter_values()?);
        records.push(StepRecord {
            global_step: index,
            optimizer_step: result.step,
            loss: result.loss,
            loss_bits: result.loss.to_bits(),
            supervised_tokens: result.supervised_tokens,
            param_digest,
            step_ms,
            wmma_forward: (phases[0] - step_phases[0]) as u64,
            wmma_weight_grad: (phases[1] - step_phases[1]) as u64,
            wmma_input_grad: (phases[2] - step_phases[2]) as u64,
            wmma_cooperative: counts.cooperative.saturating_sub(step_counts.cooperative),
            wmma_shape_fallbacks: counts.shape_fallbacks.saturating_sub(step_counts.shape_fallbacks),
            weight_conversions: conversions
                .weight_conversions
                .saturating_sub(step_conversions.weight_conversions),
            mirror_reuses: conversions.mirror_reuses.saturating_sub(step_conversions.mirror_reuses),
        });
        let completed = index + 1;
        if args.dump_steps.contains(&completed) {
            let (dump, _) = dump_parameters(&graph, args.dump_dir.as_deref().unwrap(), completed)?;
            dumps.push(dump);
        }
        if args.export_steps.contains(&completed) {
            let root = args.export_root.as_deref().unwrap();
            let directory = root.join(format!("step-{completed:04}-{}", mode.label()));
            graph.export_hf_package(&args.model, &directory)?;
            let (param_digest, _, _) = digest_parameters(&graph.parameter_values()?);
            exports.push(Export {
                step: completed,
                directory: directory.display().to_string(),
                parameter_digest: param_digest,
                exact_training_resume: false,
            });
        }
        if args.eval_steps.contains(&completed) {
            evaluations.push(evaluate(
                &graph,
                &validation_windows,
                args.batch,
                args.seq,
                "validation",
                completed,
            )?);
            if args.final_test && completed == args.steps {
                if let Some(windows) = test_windows.as_ref() {
                    evaluations.push(evaluate(&graph, windows, args.batch, args.seq, "test", completed)?);
                }
            }
        }
    }

    let end_values = graph.parameter_values()?;
    let (end_digest, _, end_nonfinite) = digest_parameters(&end_values);
    let report = Report {
        device: graph.device_name().to_owned(),
        mode: mode.label(),
        two_pass,
        model: args.model.display().to_string(),
        batch: args.batch,
        seq: args.seq,
        steps: args.steps,
        seed: args.seed,
        order_offset: args.order_offset,
        lr: hyper.lr,
        beta1: hyper.beta1,
        beta2: hyper.beta2,
        eps: hyper.eps,
        weight_decay: hyper.weight_decay,
        parameter_elements: elements,
        parameter_start_digest: start_digest,
        parameter_end_digest: end_digest,
        parameter_nonfinite: nonfinite.max(end_nonfinite),
        optimizer_step: graph.step(),
        dropout_seed: graph.dropout_seed(),
        train_windows: train_windows.len(),
        validation_windows: validation_windows.len(),
        test_windows: test_windows.as_ref().map(Vec::len),
        partition_stream_sha256: {
            let mut streams = BTreeMap::new();
            streams.insert("train".to_owned(), window_stream_digest(&train_windows));
            streams.insert("validation".to_owned(), window_stream_digest(&validation_windows));
            if let Some(windows) = test_windows.as_ref() {
                streams.insert("test".to_owned(), window_stream_digest(windows));
            }
            streams
        },
        records,
        evaluations,
        dumps,
        exports,
        total_wall_ms,
    };
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.report, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "mode": report.mode,
            "steps": report.steps,
            "order_offset": report.order_offset,
            "parameter_elements": report.parameter_elements,
            "final_validation": report.evaluations.iter().rev().find(|e| e.partition == "validation").map(|e| e.cross_entropy),
            "final_test": report.evaluations.iter().rev().find(|e| e.partition == "test").map(|e| e.cross_entropy),
            "total_wall_ms": report.total_wall_ms,
        }))?
    );
    Ok(())
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| run(Args::parse()?))
        .context("spawning drift-diagnostic worker")?
        .join()
        .map_err(|_| anyhow::anyhow!("drift-diagnostic worker panicked"))?
}
