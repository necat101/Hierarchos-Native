//! Phase XIX activation probe.
//!
//! Two observation-only modes, both against the resident production graph:
//!
//! * `--trace`: one forward over the first `--limit` windows with
//!   `HIERARCHOS_ACTIVATION_TRACE=1` set by the caller, writing the captured
//!   tensors, their digests and statistics.
//! * `--eval`: whole-partition cross-entropy and logit digest for one
//!   weights/arithmetic cell of the first-step cross-evaluation matrix.
//!
//! It changes no kernel, shader, optimizer rule or tolerance.

use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{wmma::WmmaMode, wmma_linear, VulkanDevice, VulkanTransformer};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
struct Window {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    valid_tokens: usize,
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    partition: PathBuf,
    out: PathBuf,
    batch: usize,
    seq: usize,
    limit: usize,
    trace: bool,
    eval: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            model: PathBuf::new(),
            partition: PathBuf::new(),
            out: PathBuf::new(),
            batch: 8,
            seq: 64,
            limit: 1,
            trace: false,
            eval: false,
        };
        let mut rest = std::env::args().skip(1);
        while let Some(arg) = rest.next() {
            let mut value = || {
                rest.next()
                    .with_context(|| format!("missing value for {arg}"))
            };
            match arg.as_str() {
                "--model" => args.model = PathBuf::from(value()?),
                "--partition" => args.partition = PathBuf::from(value()?),
                "--out" => args.out = PathBuf::from(value()?),
                "--batch" => args.batch = value()?.parse()?,
                "--seq" => args.seq = value()?.parse()?,
                "--limit" => args.limit = value()?.parse()?,
                "--trace" => args.trace = true,
                "--eval" => args.eval = true,
                other => bail!("unknown argument {other}"),
            }
        }
        if args.model.as_os_str().is_empty()
            || args.partition.as_os_str().is_empty()
            || args.out.as_os_str().is_empty()
        {
            bail!("--model, --partition and --out are required");
        }
        if args.trace == args.eval {
            bail!("select exactly one of --trace or --eval");
        }
        if args.batch == 0 || args.seq == 0 || args.limit == 0 {
            bail!("--batch, --seq and --limit must be positive");
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

fn batch_arrays(windows: &[Window], seq_len: usize) -> (Vec<u32>, Vec<f32>) {
    let mut input_ids = Vec::with_capacity(windows.len() * seq_len);
    let mut attention_mask = Vec::with_capacity(windows.len() * seq_len);
    for window in windows {
        let mut ids = window.input_ids.clone();
        ids.resize(seq_len, 0);
        input_ids.extend_from_slice(&ids);
        attention_mask.extend(std::iter::repeat(1.0).take(window.valid_tokens));
        attention_mask.extend(std::iter::repeat(0.0).take(seq_len - window.valid_tokens));
    }
    (input_ids, attention_mask)
}

#[derive(Serialize)]
struct TensorSummary {
    layer: usize,
    operation: String,
    shape: [usize; 3],
    elements: usize,
    sha256: String,
    max_abs: f32,
    rms: f32,
    mean_abs: f32,
    nonfinite: usize,
}

#[derive(Serialize)]
struct TraceReport {
    precision: String,
    effective_mode: String,
    dispatches: [usize; 3],
    shape_fallbacks: u64,
    tensors: Vec<TensorSummary>,
    bundle_sha256: String,
    device: String,
}

#[derive(Serialize)]
struct EvalReport {
    precision: String,
    effective_mode: String,
    dispatches: [usize; 3],
    shape_fallbacks: u64,
    windows: usize,
    valid_target_tokens: usize,
    cross_entropy: f64,
    perplexity: f64,
    nonfinite_logits: usize,
    logits_sha256: String,
    logits_elements: usize,
    logits_max_abs: f32,
    device: String,
}

fn digest_values(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn mode_label() -> String {
    match WmmaMode::from_env() {
        Ok(mode) => format!("{mode:?}"),
        Err(_) => "unavailable".to_owned(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out)?;
    let mode = WmmaMode::from_env()?;
    let device = VulkanDevice::new_for_wmma(mode)?;
    let device_name = device.name().to_owned();
    let graph = VulkanTransformer::from_hf_package(device, &args.model, args.batch, args.seq)
        .with_context(|| format!("loading {}", args.model.display()))?;
    let windows = load_windows(&args.partition, args.seq)?;
    let phases_before = wmma_linear::linear_wmma_phase_counts();
    let counts_before = wmma_linear::linear_wmma_dispatch_counts();
    if args.trace {
        if std::env::var("HIERARCHOS_ACTIVATION_TRACE").ok().as_deref() != Some("1") {
            bail!("--trace requires HIERARCHOS_ACTIVATION_TRACE=1");
        }
        let count = args.limit.min(windows.len());
        let (mut input_ids, mut attention_mask) = batch_arrays(&windows[..count], args.seq);
        while input_ids.len() < args.batch * args.seq {
            let tail = input_ids[input_ids.len() - args.seq..].to_vec();
            input_ids.extend_from_slice(&tail);
            attention_mask.extend(std::iter::repeat(0.0).take(args.seq));
        }
        let logits = graph.forward_logits(&input_ids, &attention_mask)?;
        let bundle = hierarchos_vulkan::activation_trace::take()
            .context("the traced forward produced no activation bundle")?;
        let dir = args.out.join("tensors");
        fs::create_dir_all(&dir)?;
        let mut tensors = Vec::with_capacity(bundle.tensors.len());
        for tensor in &bundle.tensors {
            let mut max_abs = 0.0f32;
            let mut mean_abs = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut nonfinite = 0usize;
            for value in &tensor.values {
                if !value.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                max_abs = max_abs.max(value.abs());
                mean_abs += f64::from(value.abs());
                sum_sq += f64::from(*value) * f64::from(*value);
            }
            let finite = tensor.values.len().saturating_sub(nonfinite);
            let name = format!("{:02}_{}", tensor.layer, tensor.operation);
            let bytes = tensor
                .values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<u8>>();
            fs::write(dir.join(format!("{name}.f32")), &bytes)?;
            tensors.push(TensorSummary {
                layer: tensor.layer,
                operation: tensor.operation.clone(),
                shape: tensor.shape,
                elements: tensor.values.len(),
                sha256: digest_values(&tensor.values),
                max_abs,
                rms: (sum_sq / finite.max(1) as f64).sqrt() as f32,
                mean_abs: (mean_abs / finite.max(1) as f64) as f32,
                nonfinite,
            });
        }
        let phases_after = wmma_linear::linear_wmma_phase_counts();
        let counts_after = wmma_linear::linear_wmma_dispatch_counts();
        let report = TraceReport {
            precision: bundle.precision.clone(),
            effective_mode: mode_label(),
            dispatches: [
                phases_after[0].saturating_sub(phases_before[0]),
                phases_after[1].saturating_sub(phases_before[1]),
                phases_after[2].saturating_sub(phases_before[2]),
            ],
            shape_fallbacks: counts_after
                .shape_fallbacks
                .saturating_sub(counts_before.shape_fallbacks),
            tensors,
            bundle_sha256: bundle.digest(),
            device: device_name,
        };
        fs::write(args.out.join("trace.json"), serde_json::to_vec_pretty(&report)?)?;
        // Keep the logit sidecar next to the trace for same-batch inspection.
        fs::write(
            args.out.join("logits.f32"),
            logits
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        println!(
            "trace precision={} tensors={} bundle={}",
            report.precision,
            report.tensors.len(),
            report.bundle_sha256
        );
        return Ok(());
    }

    let vocab = graph.config().vocab_size;
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    let mut nonfinite = 0usize;
    let mut all_logits = Vec::new();
    for chunk in windows.chunks(args.batch) {
        let (mut input_ids, mut attention_mask) = batch_arrays(chunk, args.seq);
        while input_ids.len() < args.batch * args.seq {
            let tail = input_ids[input_ids.len() - args.seq..].to_vec();
            input_ids.extend_from_slice(&tail);
            attention_mask.extend(std::iter::repeat(0.0).take(args.seq));
        }
        let logits = graph.forward_logits(&input_ids, &attention_mask)?;
        if logits.len() != args.batch * args.seq * vocab {
            bail!(
                "forward_logits returned {} values, expected {}",
                logits.len(),
                args.batch * args.seq * vocab
            );
        }
        for (index, window) in chunk.iter().enumerate() {
            for position in 0..window.valid_tokens {
                let target = window.targets[position] as usize;
                let base = (index * args.seq + position) * vocab;
                let row = &logits[base..base + vocab];
                let mut max = f64::NEG_INFINITY;
                for value in row {
                    if value.is_finite() {
                        max = max.max(f64::from(*value));
                    } else {
                        nonfinite += 1;
                    }
                }
                if !max.is_finite() {
                    total_loss = f64::INFINITY;
                    continue;
                }
                let mut sum = 0.0f64;
                for value in row {
                    sum += (f64::from(*value) - max).exp();
                }
                let log_sum = max + sum.ln();
                total_loss += log_sum - f64::from(row[target]);
                total_tokens += 1;
            }
        }
        all_logits.extend_from_slice(&logits[..chunk.len() * args.seq * vocab]);
    }
    let phases_after = wmma_linear::linear_wmma_phase_counts();
    let counts_after = wmma_linear::linear_wmma_dispatch_counts();
    let mut max_abs = 0.0f32;
    for value in &all_logits {
        if value.is_finite() {
            max_abs = max_abs.max(value.abs());
        }
    }
    let cross_entropy = total_loss / total_tokens as f64;
    let report = EvalReport {
        precision: mode_label(),
        effective_mode: mode_label(),
        dispatches: [
            phases_after[0].saturating_sub(phases_before[0]),
            phases_after[1].saturating_sub(phases_before[1]),
            phases_after[2].saturating_sub(phases_before[2]),
        ],
        shape_fallbacks: counts_after
            .shape_fallbacks
            .saturating_sub(counts_before.shape_fallbacks),
        windows: windows.len(),
        valid_target_tokens: total_tokens,
        cross_entropy,
        perplexity: cross_entropy.exp(),
        nonfinite_logits: nonfinite,
        logits_sha256: digest_values(&all_logits),
        logits_elements: all_logits.len(),
        logits_max_abs: max_abs,
        device: device_name,
    };
    fs::write(args.out.join("eval.json"), serde_json::to_vec_pretty(&report)?)?;
    fs::write(
        args.out.join("logits.f32"),
        all_logits
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    println!(
        "eval precision={} ce={:.6} tokens={}",
        report.precision, report.cross_entropy, report.valid_target_tokens
    );
    Ok(())
}
