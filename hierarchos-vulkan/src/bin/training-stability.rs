//! Stability qualification harness for the native Transformer trainer.
//!
//! Owns the optimizer loop so each step records its loss bits, the exact AdamW
//! hyperparameters and per-step learning rate, gradient norm/digest where the
//! public surface exposes gradients (native LoRA), full-parameter digest and
//! update norm, WMMA dispatch deltas, and wall time. The full-parameter path
//! does not expose raw gradient tensors; that limitation is recorded explicitly
//! rather than inferred away.
//!
//! This is a separate binary so the sealed Phase XVI training receipt and its
//! bound harness remain untouched.
//!
//! This binary uses only the public library surface. It does not change any
//! kernel, shader or numerical policy.

use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    wmma::WmmaMode, wmma_linear, AdamWHyperParams, VulkanDevice, VulkanTransformer,
    VulkanTransformerLoraConfig,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    dataset: PathBuf,
    output: PathBuf,
    batch: usize,
    seq: usize,
    steps: usize,
    start_step: usize,
    warmup: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    seed: u64,
    log_every: usize,
    digest_every: usize,
    lora_rank: Option<usize>,
    lora_alpha: Option<usize>,
    save_state: Option<PathBuf>,
    resume_state: Option<PathBuf>,
    adapter_out: Option<PathBuf>,
    report: Option<PathBuf>,
    no_export: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            model: PathBuf::new(),
            dataset: PathBuf::new(),
            output: PathBuf::new(),
            batch: 8,
            seq: 64,
            steps: 20,
            start_step: 0,
            warmup: 0,
            lr: 1.0e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
            seed: 7,
            log_every: 1,
            digest_every: 1,
            lora_rank: None,
            lora_alpha: None,
            save_state: None,
            resume_state: None,
            adapter_out: None,
            report: None,
            no_export: false,
        };
        let mut rest = std::env::args().skip(1);
        while let Some(arg) = rest.next() {
            let mut value = || {
                rest.next()
                    .with_context(|| format!("missing value for {arg}"))
            };
            match arg.as_str() {
                "--model" => args.model = PathBuf::from(value()?),
                "--dataset" => args.dataset = PathBuf::from(value()?),
                "--output" => args.output = PathBuf::from(value()?),
                "--batch" => args.batch = value()?.parse()?,
                "--seq" => args.seq = value()?.parse()?,
                "--steps" => args.steps = value()?.parse()?,
                "--start-step" => args.start_step = value()?.parse()?,
                "--warmup" => args.warmup = value()?.parse()?,
                "--lr" => args.lr = value()?.parse()?,
                "--beta1" => args.beta1 = value()?.parse()?,
                "--beta2" => args.beta2 = value()?.parse()?,
                "--eps" => args.eps = value()?.parse()?,
                "--weight-decay" => args.weight_decay = value()?.parse()?,
                "--seed" => args.seed = value()?.parse()?,
                "--log-every" => args.log_every = value()?.parse()?,
                "--digest-every" => args.digest_every = value()?.parse()?,
                "--lora-rank" => {
                    args.lora_rank = Some(value()?.parse()?)
                }
                "--lora-alpha" => {
                    args.lora_alpha = Some(value()?.parse()?)
                }
                "--save-state" => args.save_state = Some(PathBuf::from(value()?)),
                "--resume-state" => args.resume_state = Some(PathBuf::from(value()?)),
                "--adapter-out" => args.adapter_out = Some(PathBuf::from(value()?)),
                "--report" => args.report = Some(PathBuf::from(value()?)),
                "--no-export" => args.no_export = true,
                other => bail!("unknown argument {other}"),
            }
        }
        if args.model.as_os_str().is_empty()
            || args.dataset.as_os_str().is_empty()
            || args.output.as_os_str().is_empty()
        {
            bail!("--model, --dataset and --output are required");
        }
        if args.batch == 0 || args.seq == 0 || args.steps == 0 || args.log_every == 0 {
            bail!("batch, seq, steps and log-every must be positive");
        }
        if args.warmup > args.steps {
            bail!("--warmup cannot exceed --steps");
        }
        if args.adapter_out.is_some() && args.lora_rank.is_none() && args.resume_state.is_none()
        {
            bail!("--adapter-out requires a LoRA run");
        }
        if args.save_state.is_some() && args.lora_rank.is_none() && args.resume_state.is_none()
        {
            bail!(
                "--save-state requires a LoRA run; full-parameter exact state is not implemented"
            );
        }
        Ok(args)
    }
}

#[derive(Debug, Clone)]
struct Sequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

fn load_sequences(path: &Path, seq_len: usize) -> Result<Vec<Sequence>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading {} line {}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("decoding {} line {}", path.display(), index + 1))?;
        let ids = value
            .get("input_ids")
            .and_then(serde_json::Value::as_array)
            .context("qualification dataset rows require input_ids")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .context("input_ids must be u32")
            })
            .collect::<Result<Vec<_>>>()?;
        rows.push(ids);
    }
    let mut sequences = Vec::new();
    for tokens in rows {
        if tokens.len() < 2 {
            continue;
        }
        let predict = tokens.len() - 1;
        for start in (0..predict).step_by(seq_len) {
            let count = (predict - start).min(seq_len);
            let mut sequence = Sequence {
                input_ids: vec![0; seq_len],
                targets: vec![u32::MAX; seq_len],
                attention_mask: vec![0.0; seq_len],
                loss_weights: vec![0.0; seq_len],
            };
            sequence.input_ids[..count].copy_from_slice(&tokens[start..start + count]);
            sequence.targets[..count]
                .copy_from_slice(&tokens[start + 1..start + 1 + count]);
            sequence.attention_mask[..count].fill(1.0);
            sequence.loss_weights[..count].fill(1.0);
            sequences.push(sequence);
        }
    }
    if sequences.is_empty() {
        bail!("dataset produced no trainable sequences");
    }
    Ok(sequences)
}

fn batch_arrays(sequences: &[Sequence], seq_len: usize) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>)> {
    let mut input_ids = Vec::with_capacity(sequences.len() * seq_len);
    let mut targets = Vec::with_capacity(sequences.len() * seq_len);
    let mut attention_mask = Vec::with_capacity(sequences.len() * seq_len);
    let mut loss_weights = Vec::with_capacity(sequences.len() * seq_len);
    for sequence in sequences {
        input_ids.extend_from_slice(&sequence.input_ids);
        targets.extend_from_slice(&sequence.targets);
        attention_mask.extend_from_slice(&sequence.attention_mask);
        loss_weights.extend_from_slice(&sequence.loss_weights);
    }
    let normalization = loss_weights.iter().copied().sum::<f32>();
    if !normalization.is_finite() || normalization <= 0.0 {
        bail!("batch has no finite positive supervised loss weight");
    }
    for weight in &mut loss_weights {
        *weight /= normalization;
    }
    Ok((input_ids, targets, attention_mask, loss_weights))
}

fn digest_map(values: &BTreeMap<String, Vec<f32>>) -> (String, usize, usize, f64) {
    let mut hash = Sha256::new();
    let mut nonfinite = 0usize;
    let mut elements = 0usize;
    let mut sumsq = 0.0f64;
    for (name, tensor) in values {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((tensor.len() as u64).to_le_bytes());
        for value in tensor {
            hash.update(value.to_bits().to_le_bytes());
            elements += 1;
            if value.is_finite() {
                sumsq += f64::from(*value) * f64::from(*value);
            } else {
                nonfinite += 1;
            }
        }
    }
    (format!("{:x}", hash.finalize()), elements, nonfinite, sumsq.sqrt())
}

fn update_stats(
    previous: &BTreeMap<String, Vec<f32>>,
    current: &BTreeMap<String, Vec<f32>>,
) -> Result<(f64, f64, usize)> {
    let mut sumsq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut count = 0usize;
    for (name, values) in current {
        let before = previous
            .get(name)
            .with_context(|| format!("parameter set changed between steps: {name}"))?;
        if before.len() != values.len() {
            bail!("parameter length changed between steps: {name}");
        }
        for (old, new) in before.iter().zip(values) {
            let delta = f64::from(*new) - f64::from(*old);
            sumsq += delta * delta;
            max_abs = max_abs.max(delta.abs());
            count += 1;
        }
    }
    Ok((sumsq.sqrt(), max_abs, count))
}

#[derive(Serialize)]
struct StepRecord {
    event: &'static str,
    global_step: usize,
    optimizer_step: u32,
    data_epoch: usize,
    data_batch: usize,
    loss: f32,
    loss_bits: u32,
    loss_finite: bool,
    step_ms: f64,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    optimizer_status: &'static str,
    grad_norm: Option<f64>,
    grad_digest: Option<String>,
    grad_nonfinite: Option<usize>,
    param_digest: Option<String>,
    param_nonfinite: Option<usize>,
    update_norm: Option<f64>,
    update_max_abs: Option<f64>,
    wmma_forward: u64,
    wmma_weight_grad: u64,
    wmma_input_grad: u64,
    wmma_cooperative: u64,
    wmma_shape_fallbacks: u64,
    wmma_weight_conversions: u64,
    wmma_mirror_reuses: u64,
}

#[derive(Serialize)]
struct Report {
    mode: &'static str,
    precision_requested: String,
    two_pass_requested: bool,
    device: String,
    model: String,
    dataset: String,
    batch: usize,
    seq: usize,
    steps: usize,
    start_step: usize,
    warmup: usize,
    lora: bool,
    lora_rank: Option<usize>,
    lora_alpha: Option<usize>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    lr_schedule: &'static str,
    gradient_norm_source: &'static str,
    seed: u64,
    optimizer_steps: usize,
    first_loss: f32,
    final_loss: f32,
    loss_min: f32,
    loss_max: f32,
    all_losses_finite: bool,
    parameter_start_digest: Option<String>,
    parameter_end_digest: Option<String>,
    parameter_elements: Option<usize>,
    parameter_nonfinite: Option<usize>,
    parameter_frozen: Option<bool>,
    grad_norm_min: Option<f64>,
    grad_norm_max: Option<f64>,
    grad_nonfinite_total: usize,
    wmma_forward: u64,
    wmma_weight_grad: u64,
    wmma_input_grad: u64,
    wmma_cooperative: u64,
    wmma_shape_fallbacks: u64,
    wmma_weight_conversions: u64,
    wmma_mirror_reuses: u64,
    total_wall_ms: f64,
    measured_step_ms_mean: f64,
    reference_started_unix_ms: u128,
    completed_unix_ms: u128,
    records: Vec<StepRecord>,
}

fn run(args: Args) -> Result<()> {
    let sequences = load_sequences(&args.dataset, args.seq)?;
    let batches_per_epoch = sequences.len() / args.batch;
    if batches_per_epoch == 0 {
        bail!(
            "dataset produced {} sequences, fewer than batch size {}",
            sequences.len(),
            args.batch
        );
    }
    let precision = WmmaMode::from_env()?;
    let two_pass = std::env::var("HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS")
        .map(|value| value == "1")
        .unwrap_or(false);
    let device = VulkanDevice::new()?;
    let mut graph = VulkanTransformer::from_hf_package(device, &args.model, args.batch, args.seq)?;
    let mut hyper = AdamWHyperParams {
        lr: args.lr,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.eps,
        weight_decay: args.weight_decay,
    };
    let mut lora = false;
    if let Some(state) = args.resume_state.as_deref() {
        let restored = graph
            .load_lora_adapter_for_training(state)
            .with_context(|| format!("loading adapter from {}", state.display()))?;
        hyper = graph
            .load_lora_optimizer_state(state)
            .with_context(|| format!("restoring optimizer state from {}", state.display()))?;
        eprintln!("resume: {restored} adapter tensors and AdamW state restored");
        lora = true;
    } else if let Some(rank) = args.lora_rank {
        let alpha = args.lora_alpha.unwrap_or(rank.saturating_mul(2));
        let target = graph
            .config()
            .architecture
            .default_lora_target_module()
            .to_owned();
        graph.enable_lora(
            VulkanTransformerLoraConfig::causal_lm(rank, alpha, vec![target], false)?,
            args.seed,
        )?;
        lora = true;
    }
    graph.set_dropout_seed(args.seed as u32 ^ (args.seed >> 32) as u32);
    let mode = if lora { "lora" } else { "full-finetune" };
    eprintln!(
        "QUAL_CONFIG mode={} precision={} two_pass={} batch={} seq={} steps={} start_step={} \
         lr={} seed={} device={}",
        mode,
        precision.label(),
        two_pass,
        args.batch,
        args.seq,
        args.steps,
        args.start_step,
        hyper.lr,
        args.seed,
        graph.device_name()
    );

    let mut records = Vec::with_capacity(args.steps);
    let mut parameter_previous = if lora {
        None
    } else {
        Some(graph.parameter_values()?)
    };
    let parameter_start_digest = parameter_previous
        .as_ref()
        .map(digest_map)
        .map(|(digest, _, _, _)| digest);
    let parameter_elements = parameter_previous.as_ref().map(|values| {
        values.values().map(Vec::len).sum::<usize>()
    });
    let initial_base_digest = if lora {
        Some(digest_map(&graph.parameter_values()?).0)
    } else {
        None
    };

    let wmma_start_counts = wmma_linear::linear_wmma_dispatch_counts();
    let wmma_start_phases = wmma_linear::linear_wmma_phase_counts();
    let wmma_start_conversions = wmma_linear::linear_wmma_conversion_stats();
    let mut previous_counts = wmma_start_counts;
    let mut previous_phases = wmma_start_phases;
    let mut previous_conversions = wmma_start_conversions;

    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let mut total_wall_ms = 0.0f64;
    let mut measured_ms = Vec::new();
    let mut grad_norms = Vec::new();
    let mut grad_nonfinite_total = 0usize;

    for index in 0..args.steps {
        let global_step = args.start_step + index;
        let epoch = global_step / batches_per_epoch;
        let batch_index = global_step % batches_per_epoch;
        let begin = batch_index * args.batch;
        let (input_ids, targets, attention_mask, loss_weights) =
            batch_arrays(&sequences[begin..begin + args.batch], args.seq)?;
        let step_started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
        total_wall_ms += step_ms;
        if index >= args.warmup {
            measured_ms.push(step_ms);
        }
        if result.step as usize != global_step + 1 {
            bail!(
                "optimizer step divergence: expected {}, graph reported {}",
                global_step + 1,
                result.step
            );
        }
        if !result.loss.is_finite() {
            bail!("non-finite loss {} at step {}", result.loss, result.step);
        }

        let counts = wmma_linear::linear_wmma_dispatch_counts();
        let phases = wmma_linear::linear_wmma_phase_counts();
        let conversions = wmma_linear::linear_wmma_conversion_stats();
        let delta = |now: u64, before: u64| now.saturating_sub(before);
        let conversions_delta = conversions.weight_conversions - previous_conversions.weight_conversions;
        let reuses_delta = conversions.mirror_reuses - previous_conversions.mirror_reuses;

        let (grad_norm, grad_digest, grad_nonfinite) = if lora {
            let gradients = graph.lora_gradient_values()?;
            if gradients.is_empty() {
                bail!("LoRA run produced no native gradient tensors");
            }
            let (digest, _, nonfinite, norm) = digest_map(&gradients);
            grad_nonfinite_total += nonfinite;
            grad_norms.push(norm);
            (Some(norm), Some(digest), Some(nonfinite))
        } else {
            (None, None, None)
        };

        let digest_due = args.digest_every > 0
            && (index % args.digest_every == 0 || index + 1 == args.steps);
        let (param_digest, param_nonfinite, update_norm, update_max_abs) = if digest_due {
            if lora {
                let values = graph.parameter_values()?;
                let (digest, _, nonfinite, _) = digest_map(&values);
                (Some(digest), Some(nonfinite), None, None)
            } else {
                let values = graph.parameter_values()?;
                let (digest, _, nonfinite, _) = digest_map(&values);
                let (norm, max_abs, _) = update_stats(
                    parameter_previous
                        .as_ref()
                        .context("missing previous parameter snapshot")?,
                    &values,
                )?;
                parameter_previous = Some(values);
                (Some(digest), Some(nonfinite), Some(norm), Some(max_abs))
            }
        } else {
            (None, None, None, None)
        };

        let record = StepRecord {
            event: "qual_step",
            global_step,
            optimizer_step: result.step,
            data_epoch: epoch,
            data_batch: batch_index,
            loss: result.loss,
            loss_bits: result.loss.to_bits(),
            loss_finite: result.loss.is_finite(),
            step_ms,
            learning_rate: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay: hyper.weight_decay,
            optimizer_status: if result.step as usize == global_step + 1 { "applied" } else { "mismatch" },
            grad_norm,
            grad_digest,
            grad_nonfinite,
            param_digest,
            param_nonfinite,
            update_norm,
            update_max_abs,
            wmma_forward: delta(phases[0] as u64, previous_phases[0] as u64),
            wmma_weight_grad: delta(phases[1] as u64, previous_phases[1] as u64),
            wmma_input_grad: delta(phases[2] as u64, previous_phases[2] as u64),
            wmma_cooperative: delta(counts.cooperative, previous_counts.cooperative),
            wmma_shape_fallbacks: delta(counts.shape_fallbacks, previous_counts.shape_fallbacks),
            wmma_weight_conversions: conversions_delta,
            wmma_mirror_reuses: reuses_delta,
        };
        if index % args.log_every == 0 || index + 1 == args.steps {
            println!("{}", serde_json::to_string(&record)?);
        }
        previous_counts = counts;
        previous_phases = phases;
        previous_conversions = conversions;
        records.push(record);
    }

    if let Some(state) = args.save_state.as_deref() {
        graph
            .save_lora_training_checkpoint(state, hyper)
            .with_context(|| format!("saving exact state to {}", state.display()))?;
    }
    if let Some(adapter) = args.adapter_out.as_deref() {
        graph.export_lora_adapter(adapter)?;
    } else if !lora && !args.no_export {
        graph.export_hf_package(&args.model, &args.output)?;
    }

    let final_parameter_digest = if lora {
        None
    } else {
        let values = graph.parameter_values()?;
        let (digest, _, nonfinite, _) = digest_map(&values);
        if nonfinite > 0 {
            bail!("{nonfinite} non-finite parameters after training");
        }
        Some(digest)
    };
    let (parameter_end_digest, parameter_nonfinite, parameter_frozen) = if lora {
        let values = graph.parameter_values()?;
        let (digest, _, nonfinite, _) = digest_map(&values);
        let frozen = initial_base_digest.as_deref() == Some(digest.as_str());
        (Some(digest), Some(nonfinite), Some(frozen))
    } else {
        let nonfinite = records
            .last()
            .and_then(|record| record.param_nonfinite)
            .unwrap_or(0);
        (final_parameter_digest, Some(nonfinite), None)
    };
    let losses = records.iter().map(|record| record.loss).collect::<Vec<_>>();
    let loss_min = losses.iter().copied().fold(f32::INFINITY, f32::min);
    let loss_max = losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wmma_end_counts = wmma_linear::linear_wmma_dispatch_counts();
    let wmma_end_phases = wmma_linear::linear_wmma_phase_counts();
    let wmma_end_conversions = wmma_linear::linear_wmma_conversion_stats();
    let mean = if measured_ms.is_empty() {
        0.0
    } else {
        measured_ms.iter().sum::<f64>() / measured_ms.len() as f64
    };
    let report = Report {
        mode,
        precision_requested: precision.label().to_owned(),
        two_pass_requested: two_pass,
        device: graph.device_name().to_owned(),
        model: args.model.display().to_string(),
        dataset: args.dataset.display().to_string(),
        batch: args.batch,
        seq: args.seq,
        steps: args.steps,
        start_step: args.start_step,
        warmup: args.warmup,
        lora,
        lora_rank: args.lora_rank,
        lora_alpha: args.lora_alpha,
        learning_rate: hyper.lr,
        beta1: hyper.beta1,
        beta2: hyper.beta2,
        eps: hyper.eps,
        weight_decay: hyper.weight_decay,
        lr_schedule: "constant",
        gradient_norm_source: if lora { "native-lora-gradients" } else { "unavailable-full-parameter" },
        seed: args.seed,
        optimizer_steps: records.len(),
        first_loss: losses.first().copied().unwrap_or(f32::NAN),
        final_loss: losses.last().copied().unwrap_or(f32::NAN),
        loss_min,
        loss_max,
        all_losses_finite: records.iter().all(|record| record.loss_finite),
        parameter_start_digest,
        parameter_end_digest,
        parameter_elements,
        parameter_nonfinite,
        parameter_frozen,
        grad_norm_min: grad_norms.iter().copied().reduce(f64::min),
        grad_norm_max: grad_norms.iter().copied().reduce(f64::max),
        grad_nonfinite_total,
        wmma_forward: phases_delta(wmma_end_phases[0], wmma_start_phases[0]),
        wmma_weight_grad: phases_delta(wmma_end_phases[1], wmma_start_phases[1]),
        wmma_input_grad: phases_delta(wmma_end_phases[2], wmma_start_phases[2]),
        wmma_cooperative: wmma_end_counts.cooperative - wmma_start_counts.cooperative,
        wmma_shape_fallbacks: wmma_end_counts.shape_fallbacks - wmma_start_counts.shape_fallbacks,
        wmma_weight_conversions: wmma_end_conversions.weight_conversions
            - wmma_start_conversions.weight_conversions,
        wmma_mirror_reuses: wmma_end_conversions.mirror_reuses
            - wmma_start_conversions.mirror_reuses,
        total_wall_ms,
        measured_step_ms_mean: mean,
        reference_started_unix_ms: started,
        completed_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis(),
        records,
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.report.as_deref() {
        fs::write(path, &encoded).with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{encoded}");
    Ok(())
}

fn phases_delta(now: usize, before: usize) -> u64 {
    now.saturating_sub(before) as u64
}

fn main() -> Result<()> {
    // The resident training graph is deep; the trainer uses the same explicit
    // stack so qualification measures the graph rather than a stack overflow.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| run(Args::parse()?))
        .context("spawning qualification worker")?
        .join()
        .map_err(|_| anyhow::anyhow!("qualification worker panicked"))?
}
