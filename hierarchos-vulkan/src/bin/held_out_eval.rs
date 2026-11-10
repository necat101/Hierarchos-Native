//! Phase XVII held-out evaluation harness.
//!
//! This binary consumes the document-level fixture in
//! `validation/fixtures/phase17` and evaluates the *actual* Vulkan Transformer
//! forward graph. Evaluation calls only `VulkanTransformer::forward_logits`,
//! which takes `&self`: Rust's borrow checker therefore forbids it from
//! mutating parameters, AdamW moments, the optimizer step or the dropout seed.
//! No backward kernel and no optimizer dispatch is reachable from that path.
//!
//! Cross-entropy is computed on the host from the returned logits with a
//! stable log-sum-exp in f64, causal alignment `logit[t] -> token[t+1]`, and
//! token weighting over valid (non-padding) targets only.
//!
//! Training mode owns the optimizer loop exactly like `training-stability.rs`
//! so the two are comparable, and evaluates the validation partition at the
//! requested steps. The final test partition is only touched where the caller
//! asks for it.
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
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
struct Document {
    document_id: String,
    family: String,
    input_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
struct Window {
    document_id: String,
    family: String,
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    valid_tokens: usize,
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    train: Option<PathBuf>,
    validation: Option<PathBuf>,
    test: Option<PathBuf>,
    report: PathBuf,
    batch: usize,
    seq: usize,
    steps: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    seed: u64,
    schedule: Vec<usize>,
    eval_only: Option<String>,
    eval_at: Option<usize>,
    no_eval: bool,
    test_at_final: bool,
    order_offset: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            model: PathBuf::new(),
            train: None,
            validation: None,
            test: None,
            report: PathBuf::new(),
            batch: 8,
            seq: 64,
            steps: 100,
            lr: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
            seed: 7,
            schedule: vec![0, 10, 20, 40, 60, 80, 100],
            eval_only: None,
            eval_at: None,
            no_eval: false,
            test_at_final: true,
            order_offset: 0,
        };
        let mut rest = std::env::args().skip(1);
        while let Some(arg) = rest.next() {
            let mut value = || {
                rest.next()
                    .with_context(|| format!("missing value for {arg}"))
            };
            match arg.as_str() {
                "--model" => args.model = PathBuf::from(value()?),
                "--train" => args.train = Some(PathBuf::from(value()?)),
                "--validation" => args.validation = Some(PathBuf::from(value()?)),
                "--test" => args.test = Some(PathBuf::from(value()?)),
                "--report" => args.report = PathBuf::from(value()?),
                "--batch" => args.batch = value()?.parse()?,
                "--seq" => args.seq = value()?.parse()?,
                "--steps" => args.steps = value()?.parse()?,
                "--lr" => args.lr = value()?.parse()?,
                "--beta1" => args.beta1 = value()?.parse()?,
                "--beta2" => args.beta2 = value()?.parse()?,
                "--eps" => args.eps = value()?.parse()?,
                "--weight-decay" => args.weight_decay = value()?.parse()?,
                "--seed" => args.seed = value()?.parse()?,
                "--schedule" => {
                    args.schedule = value()?
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| part.trim().parse())
                        .collect::<std::result::Result<Vec<usize>, _>>()?
                }
                "--eval-only" => args.eval_only = Some(value()?),
                "--eval-at" => args.eval_at = Some(value()?.parse()?),
                "--no-eval" => args.no_eval = true,
                "--no-final-test" => args.test_at_final = false,
                "--order-offset" => args.order_offset = value()?.parse()?,
                other => bail!("unknown argument {other}"),
            }
        }
        if args.model.as_os_str().is_empty() || args.report.as_os_str().is_empty() {
            bail!("--model and --report are required");
        }
        if args.batch == 0 || args.seq == 0 {
            bail!("--batch and --seq must be positive");
        }
        if let Some(partition) = args.eval_only.as_deref() {
            if !matches!(partition, "train" | "validation" | "test") {
                bail!("--eval-only must be train, validation or test");
            }
        } else if args.train.is_none() || args.validation.is_none() {
            bail!("training mode requires --train and --validation");
        }
        if args.steps == 0 && args.eval_only.is_none() {
            bail!("--steps must be positive");
        }
        if let Some(step) = args.eval_at {
            if step > args.steps {
                bail!("--eval-at cannot exceed --steps");
            }
        }
        Ok(args)
    }
}

fn load_documents(path: &Path) -> Result<Vec<Document>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut documents = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading {} line {}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("decoding {} line {}", path.display(), index + 1))?;
        let document_id = value
            .get("document_id")
            .and_then(serde_json::Value::as_str)
            .context("partition rows require document_id")?
            .to_owned();
        let family = value
            .get("family")
            .and_then(serde_json::Value::as_str)
            .context("partition rows require family")?
            .to_owned();
        let input_ids = value
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
            .collect::<Result<Vec<_>>>()?;
        documents.push(Document {
            document_id,
            family,
            input_ids,
        });
    }
    if documents.is_empty() {
        bail!("partition {} has no documents", path.display());
    }
    Ok(documents)
}

/// Document-scoped windows, identical to the generator's own rule: the final
/// token has no target, a short tail window is kept with an explicit valid
/// count, and no window ever crosses a document boundary.
fn build_windows(documents: &[Document], seq_len: usize) -> Vec<Window> {
    let mut windows = Vec::new();
    for document in documents {
        if document.input_ids.len() < 2 {
            continue;
        }
        let predict = document.input_ids.len() - 1;
        for start in (0..predict).step_by(seq_len) {
            let count = (predict - start).min(seq_len);
            windows.push(Window {
                document_id: document.document_id.clone(),
                family: document.family.clone(),
                input_ids: document.input_ids[start..start + count].to_vec(),
                targets: document.input_ids[start + 1..start + 1 + count].to_vec(),
                valid_tokens: count,
            });
        }
    }
    windows
}

fn digest_parameters(values: &BTreeMap<String, Vec<f32>>) -> (String, usize, usize) {
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

/// Digest of the optimizer state this build can observe through the public
/// API. Full-parameter AdamW moment tensors are private, so the observable
/// state is the optimizer step, the amortised learning rate, the coefficients
/// and the dropout seed. The Phase XVII receipt states this limitation and
/// pairs it with a bitwise interleaved-evaluation control that covers the
/// unobservable moments behaviourally.
fn optimizer_state_digest(
    step: u32,
    parameter_digest: &str,
    hyper: &AdamWHyperParams,
    dropout_seed: u32,
) -> String {
    let mut hash = Sha256::new();
    hash.update(step.to_le_bytes());
    hash.update(parameter_digest.as_bytes());
    hash.update(hyper.lr.to_bits().to_le_bytes());
    hash.update(hyper.beta1.to_bits().to_le_bytes());
    hash.update(hyper.beta2.to_bits().to_le_bytes());
    hash.update(hyper.eps.to_bits().to_le_bytes());
    hash.update(hyper.weight_decay.to_bits().to_le_bytes());
    hash.update(dropout_seed.to_le_bytes());
    format!("{:x}", hash.finalize())
}

#[derive(Serialize, Clone)]
struct ExampleLoss {
    document_id: String,
    family: String,
    valid_tokens: usize,
    loss_sum: f64,
    cross_entropy: f32,
}

#[derive(Serialize)]
struct Evaluation {
    tag: String,
    partition: String,
    after_step: u32,
    examples: usize,
    valid_target_tokens: usize,
    cross_entropy: f64,
    perplexity: f64,
    per_example: Vec<ExampleLoss>,
    nonfinite_logits: usize,
    nonfinite_examples: usize,
    eval_ms: f64,
    parameter_digest_before: String,
    parameter_digest_after: String,
    optimizer_state_digest_before: String,
    optimizer_state_digest_after: String,
    optimizer_step_before: u32,
    optimizer_step_after: u32,
    dropout_seed_before: u32,
    dropout_seed_after: u32,
    parameter_elements: usize,
    wmma_forward_delta: u64,
    wmma_weight_grad_delta: u64,
    wmma_input_grad_delta: u64,
}

#[derive(Serialize)]
struct StepRecord {
    global_step: usize,
    optimizer_step: u32,
    loss: f32,
    loss_bits: u32,
    loss_finite: bool,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    param_digest: String,
    step_ms: f64,
    wmma_forward: u64,
    wmma_weight_grad: u64,
    wmma_input_grad: u64,
    wmma_cooperative: u64,
    wmma_shape_fallbacks: u64,
}

#[derive(Serialize)]
struct Report {
    mode: &'static str,
    device: String,
    model: String,
    train_partition: Option<String>,
    validation_partition: Option<String>,
    test_partition: Option<String>,
    batch: usize,
    seq: usize,
    steps: usize,
    seed: u64,
    precision_requested: String,
    two_pass_requested: bool,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    schedule: Vec<usize>,
    parameter_elements: usize,
    train_windows: Option<usize>,
    validation_windows: Option<usize>,
    test_windows: Option<usize>,
    partition_stream_sha256: BTreeMap<String, String>,
    parameter_start_digest: String,
    parameter_end_digest: String,
    parameter_nonfinite: usize,
    optimizer_step: u32,
    dropout_seed: u32,
    all_losses_finite: bool,
    records: Vec<StepRecord>,
    evaluations: Vec<Evaluation>,
    total_wall_ms: f64,
}

/// Binds the exact window stream (ids, causal targets, valid counts) the run
/// consumed. The checker re-derives the same digest from the committed
/// partitions, so a changed causal alignment or re-windowed corpus cannot be
/// hidden behind a plausible-looking loss.
fn window_stream_digest(windows: &[Window]) -> String {
    let mut hash = Sha256::new();
    for window in windows {
        hash.update((window.document_id.len() as u64).to_le_bytes());
        hash.update(window.document_id.as_bytes());
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

fn batch_arrays(
    windows: &[Window],
    seq_len: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
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

/// Forward-only evaluation over an arbitrary window count.
///
/// The final batch is padded by repeating the last window's ids with a zero
/// attention mask; padded rows are excluded from every reported total.
fn evaluate(
    graph: &VulkanTransformer,
    windows: &[Window],
    batch: usize,
    seq: usize,
    tag: &str,
    partition: &str,
    hyper: &AdamWHyperParams,
) -> Result<Evaluation> {
    let parameters_before = graph.parameter_values()?;
    let (digest_before, elements, _) = digest_parameters(&parameters_before);
    let step_before = graph.step();
    let seed_before = graph.dropout_seed();
    let optimizer_before =
        optimizer_state_digest(step_before, &digest_before, hyper, seed_before);
    let phases_before = wmma_linear::linear_wmma_phase_counts();
    let started = Instant::now();

    let mut per_example = Vec::with_capacity(windows.len());
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    let mut nonfinite_logits = 0usize;
    let mut nonfinite_examples = 0usize;

    for chunk in windows.chunks(batch) {
        let (mut input_ids, _, mut attention_mask, _) = batch_arrays(chunk, seq);
        // Padding rows: repeat the last real row, then mask it out.
        while input_ids.len() < batch * seq {
            let tail = input_ids[input_ids.len() - seq..].to_vec();
            input_ids.extend_from_slice(&tail);
            attention_mask.extend(std::iter::repeat(0.0).take(seq));
        }
        let logits = graph.forward_logits(&input_ids, &attention_mask)?;
        if logits.len() != batch * seq * graph.config().vocab_size {
            bail!(
                "{tag}: forward_logits returned {} values, expected {}",
                logits.len(),
                batch * seq * graph.config().vocab_size
            );
        }
        let vocab = graph.config().vocab_size;
        for (index, window) in chunk.iter().enumerate() {
            let mut loss = 0.0f64;
            let mut nonfinite = 0usize;
            for position in 0..window.valid_tokens {
                let target = window.targets[position] as usize;
                let base = (index * seq + position) * vocab;
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
                let logit = f64::from(row[target]);
                loss += (max + sum.ln()) - logit;
            }
            total_tokens += window.valid_tokens;
            if nonfinite > 0 || !loss.is_finite() {
                nonfinite_examples += 1;
                nonfinite_logits += nonfinite;
                total_loss = f64::INFINITY;
            } else {
                total_loss += loss;
            }
            let per_token = if window.valid_tokens > 0 {
                (loss / window.valid_tokens as f64) as f32
            } else {
                f32::NAN
            };
            per_example.push(ExampleLoss {
                document_id: window.document_id.clone(),
                family: window.family.clone(),
                valid_tokens: window.valid_tokens,
                loss_sum: loss,
                cross_entropy: per_token,
            });
        }
    }

    let eval_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cross_entropy = total_loss / total_tokens as f64;
    let parameters_after = graph.parameter_values()?;
    let (digest_after, _, _) = digest_parameters(&parameters_after);
    let step_after = graph.step();
    let seed_after = graph.dropout_seed();
    let optimizer_after =
        optimizer_state_digest(step_after, &digest_after, hyper, seed_after);
    let phases_after = wmma_linear::linear_wmma_phase_counts();

    Ok(Evaluation {
        tag: tag.to_owned(),
        partition: partition.to_owned(),
        after_step: step_after,
        examples: windows.len(),
        valid_target_tokens: total_tokens,
        cross_entropy,
        perplexity: cross_entropy.exp(),
        per_example,
        nonfinite_logits,
        nonfinite_examples,
        eval_ms,
        parameter_digest_before: digest_before,
        parameter_digest_after: digest_after,
        optimizer_state_digest_before: optimizer_before,
        optimizer_state_digest_after: optimizer_after,
        optimizer_step_before: step_before,
        optimizer_step_after: step_after,
        dropout_seed_before: seed_before,
        dropout_seed_after: seed_after,
        parameter_elements: elements,
        wmma_forward_delta: (phases_after[0] - phases_before[0]) as u64,
        wmma_weight_grad_delta: (phases_after[1] - phases_before[1]) as u64,
        wmma_input_grad_delta: (phases_after[2] - phases_before[2]) as u64,
    })
}

fn run(args: Args) -> Result<()> {
    let precision = WmmaMode::from_env()?;
    let two_pass = std::env::var("HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS")
        .map(|value| value == "1")
        .unwrap_or(false);
    let device = VulkanDevice::new()?;
    let mut graph =
        VulkanTransformer::from_hf_package(device, &args.model, args.batch, args.seq)?;
    graph.set_dropout_seed(args.seed as u32 ^ (args.seed >> 32) as u32);
    let hyper = AdamWHyperParams {
        lr: args.lr,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.eps,
        weight_decay: args.weight_decay,
    };

    let train_windows = match args.train.as_deref() {
        Some(path) => Some(build_windows(&load_documents(path)?, args.seq)),
        None => None,
    };
    let validation_windows = match args.validation.as_deref() {
        Some(path) => Some(build_windows(&load_documents(path)?, args.seq)),
        None => None,
    };
    let test_windows = match args.test.as_deref() {
        Some(path) => Some(build_windows(&load_documents(path)?, args.seq)),
        None => None,
    };
    // The stream binding describes the corpus the run consumed, not the order
    // it was consumed in, so it is captured before any order variation.
    let train_stream_digest = train_windows
        .as_ref()
        .map(|windows| window_stream_digest(windows));

    let start_parameters = graph.parameter_values()?;
    let (start_digest, elements, parameter_nonfinite) = digest_parameters(&start_parameters);
    let mut records = Vec::new();
    let mut evaluations = Vec::new();
    let started = Instant::now();
    let mut total_wall_ms = 0.0f64;

    if let Some(partition) = args.eval_only.as_deref() {
        let windows = match partition {
            "train" => train_windows.as_ref(),
            "validation" => validation_windows.as_ref(),
            "test" => test_windows.as_ref(),
            _ => unreachable!(),
        }
        .with_context(|| format!("--eval-only {partition} requires that partition path"))?;
        evaluations.push(evaluate(
            &graph, windows, args.batch, args.seq,
            &format!("eval-{partition}"), partition, &hyper,
        )?);
    } else {
        let mut train_windows = train_windows
            .as_ref()
            .context("training mode requires --train")?
            .clone();
        // A controlled, genuinely active variation: rotate the deterministic
        // window order. Unlike the dropout seed (dropout is 0.0 here), this
        // changes the optimizer trajectory, so repeated runs under different
        // offsets are real replications rather than relabelled identical runs.
        if args.order_offset > 0 && !train_windows.is_empty() {
            let shift = args.order_offset % train_windows.len();
            train_windows.rotate_left(shift);
        }
        let train_windows = train_windows;
        let validation_windows = validation_windows
            .as_ref()
            .context("training mode requires --validation")?;
        let batches_per_epoch = train_windows.len() / args.batch;
        if batches_per_epoch == 0 {
            bail!(
                "training partition produced {} windows, fewer than batch size {}",
                train_windows.len(),
                args.batch
            );
        }
        if args.schedule.contains(&0) && !args.no_eval {
            evaluations.push(evaluate(
                &graph, validation_windows, args.batch, args.seq,
                "validation@0", "validation", &hyper,
            )?);
        }
        if args.test_at_final {
            if let Some(test_windows) = test_windows.as_ref() {
                evaluations.push(evaluate(
                    &graph, test_windows, args.batch, args.seq,
                    "test@0", "test", &hyper,
                )?);
            }
        }

        let mut previous_phases = wmma_linear::linear_wmma_phase_counts();
        let mut previous_counts = wmma_linear::linear_wmma_dispatch_counts();
        for index in 0..args.steps {
            let epoch = index / batches_per_epoch;
            let batch_index = index % batches_per_epoch;
            let begin = batch_index * args.batch;
            let (input_ids, targets, attention_mask, loss_weights) =
                batch_arrays(&train_windows[begin..begin + args.batch], args.seq);
            let step_started = Instant::now();
            let result =
                graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
            let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            total_wall_ms += step_ms;
            if result.step as usize != index + 1 {
                bail!(
                    "optimizer step divergence at step {}: expected {}, got {}",
                    index,
                    index + 1,
                    result.step
                );
            }
            if !result.loss.is_finite() {
                bail!("non-finite training loss {} at step {}", result.loss, result.step);
            }
            let phases = wmma_linear::linear_wmma_phase_counts();
            let counts = wmma_linear::linear_wmma_dispatch_counts();
            let (param_digest, _, _) = digest_parameters(&graph.parameter_values()?);
            records.push(StepRecord {
                global_step: index,
                optimizer_step: result.step,
                loss: result.loss,
                loss_bits: result.loss.to_bits(),
                loss_finite: result.loss.is_finite(),
                learning_rate: hyper.lr,
                beta1: hyper.beta1,
                beta2: hyper.beta2,
                eps: hyper.eps,
                weight_decay: hyper.weight_decay,
                param_digest,
                step_ms,
                wmma_forward: (phases[0] - previous_phases[0]) as u64,
                wmma_weight_grad: (phases[1] - previous_phases[1]) as u64,
                wmma_input_grad: (phases[2] - previous_phases[2]) as u64,
                wmma_cooperative: counts
                    .cooperative
                    .saturating_sub(previous_counts.cooperative),
                wmma_shape_fallbacks: counts
                    .shape_fallbacks
                    .saturating_sub(previous_counts.shape_fallbacks),
            });
            previous_phases = phases;
            previous_counts = counts;
            let completed = index + 1;
            let due = args.schedule.contains(&completed)
                || args.eval_at == Some(completed);
            if due && !args.no_eval {
                evaluations.push(evaluate(
                    &graph, validation_windows, args.batch, args.seq,
                    &format!("validation@{completed}"), "validation", &hyper,
                )?);
            }
        }
        if args.test_at_final && !args.no_eval {
            if let Some(test_windows) = test_windows.as_ref() {
                evaluations.push(evaluate(
                    &graph, test_windows, args.batch, args.seq,
                    "test@final", "test", &hyper,
                )?);
            }
        }
    }

    let end_parameters = graph.parameter_values()?;
    let (end_digest, _, end_nonfinite) = digest_parameters(&end_parameters);
    let report = Report {
        mode: if args.eval_only.is_some() { "evaluation" } else { "training" },
        device: graph.device_name().to_owned(),
        model: args.model.display().to_string(),
        train_partition: args.train.as_ref().map(|path| path.display().to_string()),
        validation_partition: args
            .validation
            .as_ref()
            .map(|path| path.display().to_string()),
        test_partition: args.test.as_ref().map(|path| path.display().to_string()),
        batch: args.batch,
        seq: args.seq,
        steps: args.steps,
        seed: args.seed,
        precision_requested: precision.label().to_owned(),
        two_pass_requested: two_pass,
        lr: hyper.lr,
        beta1: hyper.beta1,
        beta2: hyper.beta2,
        eps: hyper.eps,
        weight_decay: hyper.weight_decay,
        schedule: args.schedule.clone(),
        train_windows: train_windows.as_ref().map(Vec::len),
        validation_windows: validation_windows.as_ref().map(Vec::len),
        test_windows: test_windows.as_ref().map(Vec::len),
        partition_stream_sha256: {
            let mut streams = BTreeMap::new();
            if let Some(digest) = train_stream_digest.clone() {
                streams.insert("train".to_owned(), digest);
            }
            if let Some(windows) = validation_windows.as_ref() {
                streams.insert("validation".to_owned(), window_stream_digest(windows));
            }
            if let Some(windows) = test_windows.as_ref() {
                streams.insert("test".to_owned(), window_stream_digest(windows));
            }
            streams
        },
        parameter_elements: elements,
        parameter_start_digest: start_digest,
        parameter_end_digest: end_digest,
        parameter_nonfinite: parameter_nonfinite.max(end_nonfinite),
        optimizer_step: graph.step(),
        dropout_seed: graph.dropout_seed(),
        all_losses_finite: records.iter().all(|record| record.loss_finite),
        records,
        evaluations,
        total_wall_ms,
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.report, &encoded)
        .with_context(|| format!("writing {}", args.report.display()))?;
    println!("{}", serde_json::to_string(&serde_json::json!({
        "mode": report.mode,
        "device": report.device,
        "steps": report.steps,
        "parameter_elements": report.parameter_elements,
        "evaluations": report.evaluations.len(),
        "total_wall_ms": report.total_wall_ms,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    }))?);
    Ok(())
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| run(Args::parse()?))
        .context("spawning held-out evaluation worker")?
        .join()
        .map_err(|_| anyhow::anyhow!("held-out evaluation worker panicked"))?
}
