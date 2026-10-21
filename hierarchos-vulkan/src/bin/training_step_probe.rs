//! End-to-end single-process training-step probe.
//!
//! Reports wall-clock steps/second, tokens/second, and the command-batch
//! dispatch/barrier accounting for the canonical raw-token microbatch training
//! path on a fixed small model package. Kernel-level harnesses cannot show
//! whether scheduler-level barrier elision or kernel changes actually move a
//! whole training step, so this probe measures the real composition:
//! forward, reverse, and the canonical full-model AdamW update in one call.
//!
//! The model package is either supplied with `--model-dir` or created once in a
//! temporary directory from the fixed bootstrap config below, so the same
//! fixture can be built on an older artifact to compare like with like.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use hierarchos_inference::{initialize_model_package, NativeBootstrapConfig};
use hierarchos_vulkan::{
    AdamWHyperParams, HierarchosRawTokenSequenceContextInput,
    HierarchosRawTokenTapeMicrobatchInput, HierarchosRawTokenTapeStepInput,
    HierarchosSequenceGradientNormalization, HierarchosTokenTapeReadbackPolicy,
    HierarchosTrainingGraph, VulkanDevice,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Full FP32 trajectory checks are opt-in and run outside the timed step.
/// Include lengths and raw little-endian bits so signed zero is significant.
fn vector_digest(values: &[f32]) -> Result<String> {
    if values.iter().any(|value| !value.is_finite()) {
        bail!("non-finite value in training state snapshot");
    }
    let mut hash = Sha256::new();
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Serialize)]
struct StateSnapshot {
    step: usize,
    optimizer_step: u32,
    tensors: BTreeMap<String, String>,
    optimizer_slot_steps: BTreeMap<String, u32>,
}

/// Opt-in exact-value companion to [`StateSnapshot`] for diagnosing a
/// trajectory divergence: digests say that two runs differ, raw FP32 bits show
/// how large the difference is and in which direction. Off unless
/// `HIERARCHOS_PROBE_STATE_VALUES` names an output path.
#[derive(Serialize)]
struct ValueSnapshot {
    step: usize,
    optimizer_step: u32,
    values: BTreeMap<String, Vec<u32>>,
}

const STATE_VALUES_ENV: &str = "HIERARCHOS_PROBE_STATE_VALUES";
const ROUTE_LABELS_ENV: &str = "HIERARCHOS_PROBE_ROUTE_LABELS";

#[derive(Serialize)]
struct Sample {
    step: usize,
    elapsed_ms: f64,
    tokens_per_second: f64,
}

#[derive(Serialize)]
struct Output {
    device: String,
    decision_device: String,
    model_dir: String,
    batch: usize,
    tokens_per_sequence: usize,
    sequences: usize,
    batch_tokens_per_step: usize,
    warmup_steps: usize,
    measured_steps: usize,
    readback: &'static str,
    median_step_ms: f64,
    median_tokens_per_second: f64,
    median_steps_per_second: f64,
    dispatch_count: usize,
    shader_barrier_count: usize,
    queue_submissions: u32,
    mean_loss: f32,
    /// Host-side recording counters, per step. A step costs ~10 us of host time per
    /// recorded dispatch, and these are what that time is spent on: descriptor sets
    /// are allocated on a binding-set miss, binds are emitted when the bound buffers
    /// differ from the previous dispatch, and a pipeline bind happens per kernel
    /// switch. They are reported so the host cost of a dispatch can be judged
    /// against the dispatch itself rather than guessed at.
    descriptor_set_count: usize,
    descriptor_bind_count: usize,
    pipeline_bind_count: usize,
    samples: Vec<Sample>,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

struct Args {
    model_dir: Option<PathBuf>,
    batch: usize,
    tokens: usize,
    sequences: usize,
    warmup: usize,
    steps: usize,
    readback: HierarchosTokenTapeReadbackPolicy,
    device_index: usize,
    context_dim: usize,
    hidden: usize,
    vocab: usize,
    state_output: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            model_dir: None,
            batch: 2,
            tokens: 8,
            sequences: 1,
            warmup: 1,
            steps: 5,
            readback: HierarchosTokenTapeReadbackPolicy::Full,
            device_index: 0,
            context_dim: 128,
            hidden: 128,
            vocab: 256,
            state_output: None,
        };
        let mut rest = std::env::args().skip(1);
        while let Some(arg) = rest.next() {
            let mut value = || {
                rest.next()
                    .with_context(|| format!("missing value for {arg}"))
            };
            match arg.as_str() {
                "--model-dir" => args.model_dir = Some(PathBuf::from(value()?)),
                "--batch" => args.batch = value()?.parse()?,
                "--tokens" => args.tokens = value()?.parse()?,
                "--sequences" => args.sequences = value()?.parse()?,
                "--warmup" => args.warmup = value()?.parse()?,
                "--steps" => args.steps = value()?.parse()?,
                "--device-index" => args.device_index = value()?.parse()?,
                "--context-dim" => args.context_dim = value()?.parse()?,
                "--hidden" => args.hidden = value()?.parse()?,
                "--vocab" => args.vocab = value()?.parse()?,
                "--state-output" => args.state_output = Some(PathBuf::from(value()?)),
                "--loss-only-readback" => {
                    args.readback = HierarchosTokenTapeReadbackPolicy::LossOnly
                }
                other => bail!("unknown argument {other}"),
            }
        }
        if args.batch == 0 || args.tokens == 0 || args.sequences == 0 || args.steps == 0 {
            bail!("batch, tokens, sequences, and steps must all be positive");
        }
        Ok(args)
    }
}

fn bootstrap_config(args: &Args) -> NativeBootstrapConfig {
    let mut config = NativeBootstrapConfig::for_vocab(args.vocab);
    config.context_dim = args.context_dim;
    config.h_hidden = args.hidden;
    config.l_hidden = args.hidden;
    config.persistent_dim = 32;
    config.ltm_slots = 16;
    config.ltm_key_dim = 32;
    config.ltm_val_dim = 32;
    config.ltm_topk = 2;
    config.h_stride = 2;
    config.max_h_steps = 3;
    config.max_l_steps = 2;
    config.rwkv_head_size = 32;
    config.token_adapter_rank = 16;
    config.rosa_max_context = 16;
    config.memory_gate_warmup_steps = 0;
    config
}

/// Deterministic fixture path: the same process id-independent directory gives
/// every run — including runs of an older artifact — the same random
/// initialization, so a before/after comparison is not confounded by weights.
///
/// `--model-dir` overrides that path, and is initialized when it does not
/// already hold a package. That lets a caller point the probe at a scratch
/// directory it deletes first, which is how a repeated run gets the same
/// starting weights: an existing package keeps advancing through AdamW, so
/// reusing one measures a different point in training every time.
fn fixture(args: &Args) -> Result<PathBuf> {
    let path = args.model_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "hierarchos-training-step-probe-v{}-{}",
            args.context_dim, args.hidden
        ))
    });
    if path.join("hierarchos_config.json").exists() {
        return Ok(path);
    }
    initialize_model_package(&path, &bootstrap_config(args))
        .with_context(|| format!("initializing model package at {}", path.display()))?;
    Ok(path)
}

fn main() -> Result<()> {
    // The canonical training graph is deep; match the trainer's explicit host
    // stack so the probe measures the graph instead of overflowing on it.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(probe)
        .context("spawning training-step probe worker")?
        .join()
        .map_err(|_| anyhow::anyhow!("training-step probe worker panicked"))?
}

fn probe() -> Result<()> {
    let args = Args::parse()?;
    let device = VulkanDevice::new_with_index(args.device_index)?;
    let model_dir = fixture(&args)?;
    let mut graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        args.batch,
        3,
        2,
        args.batch,
        args.batch,
    )
    .context("building Hierarchos training graph")?;
    let config = graph.config();
    let h_state_len = args
        .batch
        .checked_mul(config.h_hidden)
        .and_then(|value| value.checked_mul(graph.h_recurrent().state_size()))
        .context("H packed-state length overflow")?;
    let l_state_len = args
        .batch
        .checked_mul(config.l_hidden)
        .and_then(|value| value.checked_mul(graph.l_recurrent().state_size()))
        .context("L packed-state length overflow")?;
    let h_states = vec![vec![0.0f32; h_state_len]; args.sequences];
    let l_states = vec![vec![0.0f32; l_state_len]; args.sequences];
    let h_refs = h_states.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let l_refs = l_states.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut arena = graph.create_token_tape_arena(args.batch, args.tokens, &h_refs, &l_refs)?;

    // Fixed synthetic corpus: a deterministic token cycle keeps every step
    // identical in shape and cost while still producing real cross-entropy
    // gradients and a real AdamW update.
    let token_at = |step: usize, row: usize| -> u32 {
        (((step * 7 + row * 3 + 3) % (config.vocab_size - 1)) + 1) as u32
    };
    let step_token_ids = (0..args.tokens)
        .map(|step| {
            (0..args.batch)
                .map(|row| token_at(step, row))
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<_>>();
    let step_targets = (0..args.tokens)
        .map(|step| {
            (0..args.batch)
                .map(|row| token_at(step + 1, row))
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<_>>();
    let reset_all = vec![1u32; args.batch];
    let reset_none = vec![0u32; args.batch];
    let context = vec![0.0f32; args.batch * config.context_dim];
    let h_to_context_grad = vec![0.0f32; args.batch * config.context_dim];
    let h_depth_grad = vec![0.0f32; args.batch];
    let final_drift_grad = vec![0.0f32; args.batch * config.context_dim];
    let commitment_cost_grad = vec![0.0f32; args.batch];

    let mut snapshots = Vec::new();
    let mut value_snapshots = Vec::new();
    let capture_values = std::env::var_os(STATE_VALUES_ENV).is_some();
    let mut run = |graph: &mut HierarchosTrainingGraph,
                   arena: &mut hierarchos_vulkan::HierarchosTokenTapeArena,
                   steps: usize|
     -> Result<(Vec<f64>, usize, usize, u32, f32, (usize, usize, usize))> {
        let mut elapsed = Vec::with_capacity(steps);
        let mut dispatch = 0usize;
        let mut barriers = 0usize;
        let mut submissions = 0u32;
        let mut mean_loss = 0.0f32;
        let mut recording = (0usize, 0usize, 0usize);
        for _ in 0..steps {
            let tapes = (0..args.tokens)
                .map(|step| HierarchosRawTokenTapeStepInput {
                    h_steps: 3,
                    shadow_steps: 2,
                    token_ids: &step_token_ids[step],
                    rosa_reset_lanes: if step == 0 { &reset_all } else { &reset_none },
                    previous_context: &context,
                    target_context: &context,
                    context_alpha: 0.5,
                    h_token_ids: &step_token_ids[step],
                    l_token_ids: &step_token_ids[step],
                    h_to_context_grad: &h_to_context_grad,
                    h_depth_grad: &h_depth_grad,
                    final_drift_grad: &final_drift_grad,
                    commitment_cost_grad: &commitment_cost_grad,
                    ltm_value_alignment_position: step as u64,
                    ltm_value_alignment_mask: None,
                    ltm_value_alignment_grad: 0.0,
                    targets: &step_targets[step],
                    supervision_weights: None,
                    pytorch_tbptt_token_mask: None,
                })
                .collect::<Vec<_>>();
            let sequences = (0..args.sequences)
                .map(|_| HierarchosRawTokenTapeMicrobatchInput {
                    steps: &tapes,
                    final_h_packed_state_adjoint: None,
                    final_l_packed_state_adjoint: None,
                    sequence_context: Some(HierarchosRawTokenSequenceContextInput {
                        initial_previous_context: &context,
                        initial_target_context: &context,
                        z_loss_weight: 0.0,
                    }),
                    pytorch_tbptt_real_token_count: None,
                    pytorch_tbptt_chunk_size: None,
                    preweighted_ponder_and_commitment: false,
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            let result = graph.train_raw_token_tape_microbatch_with_readback_policy(
                arena,
                &sequences,
                AdamWHyperParams::default(),
                HierarchosSequenceGradientNormalization::MeanByToken,
                args.readback,
            )?;
            elapsed.push(started.elapsed().as_secs_f64() * 1_000.0);
            if std::env::var_os(ROUTE_LABELS_ENV).is_some() {
                eprintln!(
                    "route step={} batch={} h_geometry={:?} l_geometry={:?} h_segment={:?} l_segment={:?}",
                    elapsed.len() - 1,
                    args.batch,
                    graph.h_recurrent().backward_kernel_geometry_label(args.batch),
                    graph.l_recurrent().backward_kernel_geometry_label(args.batch),
                    graph.h_recurrent().backward_segment_schedule_label(args.batch),
                    graph.l_recurrent().backward_segment_schedule_label(args.batch),
                );
            }
            dispatch += result.dispatch_count;
            barriers += result.shader_barrier_count;
            submissions = result.queue_submissions;
            recording = (
                result.descriptor_set_count,
                result.descriptor_bind_count,
                result.pipeline_bind_count,
            );
            let loss_sum = result
                .sequences
                .iter()
                .flat_map(|sequence| sequence.losses.iter().copied())
                .sum::<f32>();
            mean_loss = loss_sum / (result.total_tokens.max(1) as f32);
            if args.state_output.is_some() || capture_values {
                let mut tensors = BTreeMap::new();
                let mut values = BTreeMap::new();
                for parameter in graph.full_model_parameter_snapshots()? {
                    tensors.insert(
                        format!("parameter/{}", parameter.name),
                        vector_digest(&parameter.values)?,
                    );
                    if capture_values {
                        values.insert(
                            format!("parameter/{}", parameter.name),
                            parameter
                                .values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect(),
                        );
                    }
                }
                let optimizer = graph.full_model_optimizer_state()?;
                let mut optimizer_slot_steps = BTreeMap::new();
                for slot in optimizer.slots {
                    tensors.insert(
                        format!("exp_avg/{}", slot.name),
                        vector_digest(&slot.exp_avg)?,
                    );
                    tensors.insert(
                        format!("exp_avg_sq/{}", slot.name),
                        vector_digest(&slot.exp_avg_sq)?,
                    );
                    if capture_values {
                        values.insert(
                            format!("exp_avg/{}", slot.name),
                            slot.exp_avg.iter().map(|value| value.to_bits()).collect(),
                        );
                        values.insert(
                            format!("exp_avg_sq/{}", slot.name),
                            slot.exp_avg_sq
                                .iter()
                                .map(|value| value.to_bits())
                                .collect(),
                        );
                    }
                    optimizer_slot_steps.insert(slot.name, slot.step);
                }
                if capture_values {
                    value_snapshots.push(ValueSnapshot {
                        step: snapshots.len(),
                        optimizer_step: optimizer.step,
                        values,
                    });
                }
                for (index, sequence) in result.sequences.iter().enumerate() {
                    for (name, values) in [
                        ("losses", &sequence.losses),
                        ("h_state", &sequence.final_h_packed_state),
                        ("l_state", &sequence.final_l_packed_state),
                        ("grad_initial_h", &sequence.grad_initial_h_packed_state),
                        ("grad_initial_l", &sequence.grad_initial_l_packed_state),
                    ] {
                        tensors.insert(format!("sequence/{index}/{name}"), vector_digest(values)?);
                    }
                    for (name, values) in [
                        ("previous_context", &sequence.final_previous_context),
                        ("target_context", &sequence.final_target_context),
                    ] {
                        if let Some(values) = values {
                            tensors
                                .insert(format!("sequence/{index}/{name}"), vector_digest(values)?);
                        }
                    }
                }
                snapshots.push(StateSnapshot {
                    step: snapshots.len(),
                    optimizer_step: optimizer.step,
                    tensors,
                    optimizer_slot_steps,
                });
            }
        }
        Ok((
            elapsed,
            dispatch,
            barriers,
            submissions,
            mean_loss,
            recording,
        ))
    };

    let _ = run(&mut graph, &mut arena, args.warmup)?;
    let (elapsed, dispatch, barriers, submissions, mean_loss, recording) =
        run(&mut graph, &mut arena, args.steps)?;
    let step_ms = median(elapsed.clone());
    let tokens_per_step = args.batch * args.tokens * args.sequences;
    let per_step_dispatch = dispatch / args.steps;
    let per_step_barriers = barriers / args.steps;
    let output = Output {
        device: device.name().to_string(),
        decision_device: format!(
            "{} [pci={:04x}:{:04x};driver={:08x}]",
            device.name(),
            device.vendor_id(),
            device.device_id(),
            device.driver_version()
        ),
        model_dir: model_dir.display().to_string(),
        batch: args.batch,
        tokens_per_sequence: args.tokens,
        sequences: args.sequences,
        batch_tokens_per_step: tokens_per_step,
        warmup_steps: args.warmup,
        measured_steps: args.steps,
        readback: match args.readback {
            HierarchosTokenTapeReadbackPolicy::Full => "full",
            HierarchosTokenTapeReadbackPolicy::LossOnly => "loss-only",
        },
        median_step_ms: step_ms,
        median_tokens_per_second: tokens_per_step as f64 / (step_ms / 1_000.0),
        median_steps_per_second: 1_000.0 / step_ms,
        dispatch_count: per_step_dispatch,
        shader_barrier_count: per_step_barriers,
        queue_submissions: submissions,
        mean_loss,
        descriptor_set_count: recording.0 / args.steps,
        descriptor_bind_count: recording.1 / args.steps,
        pipeline_bind_count: recording.2 / args.steps,
        samples: elapsed
            .into_iter()
            .enumerate()
            .map(|(index, ms)| Sample {
                step: index,
                elapsed_ms: ms,
                tokens_per_second: tokens_per_step as f64 / (ms / 1_000.0),
            })
            .collect(),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    if let Some(path) = args.state_output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(&snapshots)?)?;
    }
    if let Some(path) = std::env::var_os(STATE_VALUES_ENV) {
        let path = PathBuf::from(path);
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec(&value_snapshots)?)?;
    }
    Ok(())
}
