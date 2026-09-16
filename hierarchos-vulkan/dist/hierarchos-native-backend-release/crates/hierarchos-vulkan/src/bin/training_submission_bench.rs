use std::{fs, path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, HierarchosRawTokenTapeMicrobatchInput, HierarchosRawTokenTapeStepInput,
    HierarchosSequenceGradientNormalization, HierarchosTapeMemoryPolicy,
    HierarchosTokenTapeReadbackPolicy, HierarchosTrainingGraph, RwkvNumericsPolicy, VulkanDevice,
    BACKWARD_KERNEL_GEOMETRY_POLICY_REVISION, HIERARCHOS_VULKAN_TRAINING_PRECISION_ENV,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct OptimizerCase {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

#[derive(Deserialize)]
struct StepCase {
    token_ids: Vec<u32>,
    rosa_reset_lanes: Vec<u32>,
    previous_context: Vec<f32>,
    target_context: Vec<f32>,
    context_alpha: f32,
    h_token_ids: Vec<u32>,
    l_token_ids: Vec<u32>,
    h_to_context_grad: Vec<f32>,
    h_depth_grad: Vec<f32>,
    final_drift_grad: Vec<f32>,
    commitment_cost_grad: Vec<f32>,
    targets: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    h_initial_packed_state: Vec<f32>,
    l_initial_packed_state: Vec<f32>,
    steps: Vec<StepCase>,
    optimizer: OptimizerCase,
}

#[derive(Clone, Copy)]
enum NormalizationArg {
    Sum,
    MeanByToken,
}

impl NormalizationArg {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "sum" => Ok(Self::Sum),
            "mean" | "mean-by-token" => Ok(Self::MeanByToken),
            _ => bail!("--normalization must be sum or mean; got {raw:?}"),
        }
    }

    fn value(self) -> HierarchosSequenceGradientNormalization {
        match self {
            Self::Sum => HierarchosSequenceGradientNormalization::Sum,
            Self::MeanByToken => HierarchosSequenceGradientNormalization::MeanByToken,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::MeanByToken => "mean-by-token",
        }
    }
}

#[derive(Clone, Copy)]
enum ReadbackArg {
    Full,
    LossOnly,
}

impl ReadbackArg {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "full" => Ok(Self::Full),
            "loss-only" | "loss" => Ok(Self::LossOnly),
            _ => bail!("--readback must be full or loss-only; got {raw:?}"),
        }
    }

    fn value(self) -> HierarchosTokenTapeReadbackPolicy {
        match self {
            Self::Full => HierarchosTokenTapeReadbackPolicy::Full,
            Self::LossOnly => HierarchosTokenTapeReadbackPolicy::LossOnly,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::LossOnly => "loss-only",
        }
    }
}

struct Args {
    model_dir: PathBuf,
    case_path: PathBuf,
    tokens_per_sequence: usize,
    sequences: usize,
    warmup_iterations: usize,
    measured_iterations: usize,
    training_step: u64,
    normalization: NormalizationArg,
    readback: ReadbackArg,
    budget_fraction: f32,
    reserve_mib: u64,
    sequence_microbatch_size: Option<usize>,
    state_checkpoint_stride: Option<usize>,
    h_backward_kernel_geometry: Option<String>,
    l_backward_kernel_geometry: Option<String>,
    numerics_policy: RwkvNumericsPolicy,
    training_precision_policy: String,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args_os().skip(1);
        let mut model_dir = None;
        let mut case_path = None;
        let mut tokens_per_sequence = 8usize;
        let mut sequences = 1usize;
        let mut warmup_iterations = 1usize;
        let mut measured_iterations = 5usize;
        let mut training_step = 0u64;
        let mut normalization = NormalizationArg::MeanByToken;
        let mut readback = ReadbackArg::Full;
        let mut budget_fraction = 0.85f32;
        let mut reserve_mib = 512u64;
        let mut sequence_microbatch_size = None;
        let mut state_checkpoint_stride = None;
        let mut h_backward_kernel_geometry = None;
        let mut l_backward_kernel_geometry = None;
        let mut numerics_policy = RwkvNumericsPolicy::StrictParity;
        let mut training_precision_policy = "fp32".to_string();

        while let Some(arg) = args.next() {
            let flag = arg.to_string_lossy();
            let value = |args: &mut std::iter::Skip<std::env::ArgsOs>, name: &str| {
                args.next()
                    .with_context(|| format!("missing value after {name}"))
            };
            match flag.as_ref() {
                "--model" => model_dir = Some(PathBuf::from(value(&mut args, "--model")?)),
                "--case" => case_path = Some(PathBuf::from(value(&mut args, "--case")?)),
                "--tokens" => {
                    tokens_per_sequence = parse_usize(value(&mut args, "--tokens")?, "--tokens")?
                }
                "--sequences" => {
                    sequences = parse_usize(value(&mut args, "--sequences")?, "--sequences")?
                }
                "--warmup" => {
                    warmup_iterations = parse_usize(value(&mut args, "--warmup")?, "--warmup")?
                }
                "--iterations" => {
                    measured_iterations =
                        parse_usize(value(&mut args, "--iterations")?, "--iterations")?
                }
                "--training-step" => {
                    training_step =
                        parse_u64(value(&mut args, "--training-step")?, "--training-step")?
                }
                "--normalization" => {
                    let raw = value(&mut args, "--normalization")?;
                    normalization = NormalizationArg::parse(&raw.to_string_lossy())?;
                }
                "--readback" => {
                    let raw = value(&mut args, "--readback")?;
                    readback = ReadbackArg::parse(&raw.to_string_lossy())?;
                }
                "--budget-fraction" => {
                    budget_fraction =
                        parse_f32(value(&mut args, "--budget-fraction")?, "--budget-fraction")?
                }
                "--reserve-mib" => {
                    reserve_mib = parse_u64(value(&mut args, "--reserve-mib")?, "--reserve-mib")?
                }
                "--microbatch-size" => {
                    sequence_microbatch_size = Some(parse_usize(
                        value(&mut args, "--microbatch-size")?,
                        "--microbatch-size",
                    )?)
                }
                "--checkpoint-stride" => {
                    state_checkpoint_stride = Some(parse_usize(
                        value(&mut args, "--checkpoint-stride")?,
                        "--checkpoint-stride",
                    )?)
                }
                "--h-kernel-geometry" => {
                    h_backward_kernel_geometry = Some(
                        value(&mut args, "--h-kernel-geometry")?
                            .to_string_lossy()
                            .into_owned(),
                    )
                }
                "--l-kernel-geometry" => {
                    l_backward_kernel_geometry = Some(
                        value(&mut args, "--l-kernel-geometry")?
                            .to_string_lossy()
                            .into_owned(),
                    )
                }
                "--numerics" => {
                    let raw = value(&mut args, "--numerics")?;
                    numerics_policy = match raw.to_string_lossy().as_ref() {
                        "strict" | "strict-parity" => RwkvNumericsPolicy::StrictParity,
                        "fast-subgroup" => RwkvNumericsPolicy::FastSubgroup,
                        "fast-recurrent-tree" | "tree" => {
                            RwkvNumericsPolicy::FastRecurrentTree
                        }
                        "fast-recurrent-tiled" | "tiled" => {
                            RwkvNumericsPolicy::FastRecurrentTiled
                        }
                        "fast-recurrent-subgroup" | "recurrent-subgroup" | "subgroup-recurrent" => {
                            RwkvNumericsPolicy::FastRecurrentSubgroup
                        }
                        other => bail!(
                            "--numerics must be strict, fast-subgroup, fast-recurrent-tree, fast-recurrent-tiled, or fast-recurrent-subgroup; got {other:?}"
                        ),
                    };
                }
                "--precision" => {
                    let raw = value(&mut args, "--precision")?;
                    training_precision_policy = match raw.to_string_lossy().as_ref() {
                        "fp32" => "fp32".to_string(),
                        "fp16" | "fp16-storage-fp32-compute" => {
                            "fp16-storage-fp32-compute".to_string()
                        }
                        "fp16-parity" | "fp16-storage-parity" => {
                            "fp16-storage-parity".to_string()
                        }
                        "fp16-lm-backward" | "fp16-storage-fp16-lm-backward" => {
                            "fp16-storage-fp16-lm-backward".to_string()
                        }
                        other => bail!(
                            "--precision must be fp32, fp16-storage-fp32-compute, fp16-storage-parity, or fp16-storage-fp16-lm-backward; got {other:?}"
                        ),
                    };
                }
                "--help" | "-h" => {
                    println!(
                        "usage: hierarchos-vulkan-training-submission-bench --model MODEL_DIR --case CASE.json [--tokens N] [--sequences N] [--warmup N] [--iterations N] [--training-step N] [--normalization mean|sum] [--readback full|loss-only] [--budget-fraction F] [--reserve-mib N] [--microbatch-size N --checkpoint-stride N] [--h-kernel-geometry LABEL --l-kernel-geometry LABEL] [--numerics strict|fast-subgroup|fast-recurrent-tree|fast-recurrent-tiled|fast-recurrent-subgroup] [--precision fp32|fp16-storage-fp32-compute|fp16-storage-parity|fp16-storage-fp16-lm-backward]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other:?}"),
            }
        }

        if tokens_per_sequence == 0 || sequences == 0 || measured_iterations == 0 {
            bail!("--tokens, --sequences, and --iterations must all be positive");
        }
        if !budget_fraction.is_finite() || budget_fraction <= 0.0 || budget_fraction > 1.0 {
            bail!("--budget-fraction must be finite and in (0, 1]");
        }
        if sequence_microbatch_size.is_some() != state_checkpoint_stride.is_some() {
            bail!("--microbatch-size and --checkpoint-stride must be supplied together");
        }
        if sequence_microbatch_size == Some(0) || state_checkpoint_stride == Some(0) {
            bail!("--microbatch-size and --checkpoint-stride must be positive");
        }
        if h_backward_kernel_geometry.is_some() != l_backward_kernel_geometry.is_some() {
            bail!("--h-kernel-geometry and --l-kernel-geometry must be supplied together");
        }
        if h_backward_kernel_geometry.is_some() && sequence_microbatch_size.is_none() {
            bail!(
                "forced --h-kernel-geometry/--l-kernel-geometry requires explicit --microbatch-size/--checkpoint-stride so the automatic policy cannot replace the requested arm"
            );
        }

        Ok(Self {
            model_dir: model_dir.context("missing --model MODEL_DIR")?,
            case_path: case_path.context("missing --case CASE.json")?,
            tokens_per_sequence,
            sequences,
            warmup_iterations,
            measured_iterations,
            training_step,
            normalization,
            readback,
            budget_fraction,
            reserve_mib,
            sequence_microbatch_size,
            state_checkpoint_stride,
            h_backward_kernel_geometry,
            l_backward_kernel_geometry,
            numerics_policy,
            training_precision_policy,
        })
    }
}

fn parse_usize(raw: std::ffi::OsString, name: &str) -> Result<usize> {
    raw.to_string_lossy()
        .parse::<usize>()
        .with_context(|| format!("invalid {name}"))
}

fn parse_u64(raw: std::ffi::OsString, name: &str) -> Result<u64> {
    raw.to_string_lossy()
        .parse::<u64>()
        .with_context(|| format!("invalid {name}"))
}

fn parse_f32(raw: std::ffi::OsString, name: &str) -> Result<f32> {
    raw.to_string_lossy()
        .parse::<f32>()
        .with_context(|| format!("invalid {name}"))
}

#[derive(Serialize)]
struct Sample {
    optimizer_step: u32,
    elapsed_ms: f64,
    tokens_per_second: f64,
    outer_token_positions_per_second: f64,
    queue_submissions: u32,
    sequence_microbatch_size: usize,
    sequence_microbatch_count: usize,
    state_checkpoint_stride: usize,
    device_local_pressure_bucket: Option<u8>,
    h_backward_segment_schedule: Option<String>,
    l_backward_segment_schedule: Option<String>,
    h_backward_kernel_geometry: Option<String>,
    l_backward_kernel_geometry: Option<String>,
    rwkv_numerics_policy: String,
    sparse_state_replay: bool,
    available_for_tape_bytes: u64,
    planned_peak_bytes: u64,
    profiled_tokens_per_second: Option<f64>,
    profile_adaptive_tokens_per_second: Option<f64>,
    profile_confidence_adjusted_tokens_per_second: Option<f64>,
    profile_exploration_score_tokens_per_second: Option<f64>,
    profile_relative_uncertainty: Option<f64>,
    profile_effective_measured_iterations: Option<f64>,
    profile_observations_since_last_measurement: Option<usize>,
    profile_records: usize,
    profile_measured_iterations: usize,
    online_exploration: bool,
    mean_loss: f32,
}

#[derive(Serialize)]
struct Output {
    device: String,
    subgroup_size: u32,
    training_precision_policy: String,
    h_low_rank_fp16_parameter_storage_active: bool,
    l_low_rank_fp16_parameter_storage_active: bool,
    h_low_rank_native_fp16_backward_compute_active: bool,
    l_low_rank_native_fp16_backward_compute_active: bool,
    h_low_rank_native_fp16_parameter_grad_compute_active: bool,
    l_low_rank_native_fp16_parameter_grad_compute_active: bool,
    h_low_rank_parameter_grad_arithmetic: String,
    l_low_rank_parameter_grad_arithmetic: String,
    h_low_rank_fp16_full_forward_first_stage_arm: Option<String>,
    l_low_rank_fp16_full_forward_first_stage_arm: Option<String>,
    projection_fp16_parameter_storage_active: bool,
    projection_native_fp16_backward_compute_active: bool,
    lm_head_fp16_parameter_storage_active: bool,
    lm_head_execution_arm: String,
    lm_head_weight_grad_topology: Option<String>,
    lm_head_fused_adjoint_topology: Option<String>,
    lm_head_native_fp16_backward_compute_active: bool,
    out_norm_native_fp16_backward_compute_active: bool,
    storage_buffer_16_bit_access_enabled: bool,
    shader_float16_enabled: bool,
    native_fp16_storage_compute_ready: bool,
    shader_bfloat16_extension_exposed: bool,
    backward_kernel_geometry_revision: u32,
    architecture_revision: String,
    batch: usize,
    context_dim: usize,
    persistent_dim: usize,
    ltm_slots: usize,
    ltm_key_dim: usize,
    ltm_val_dim: usize,
    ltm_topk: usize,
    vocab_size: usize,
    h_hidden: usize,
    l_hidden: usize,
    h_width: usize,
    l_width: usize,
    h_state_size: usize,
    l_state_size: usize,
    h_rwkv_head_size: usize,
    l_rwkv_head_size: usize,
    h_low_rank_ranks: Option<(usize, usize, usize)>,
    l_low_rank_ranks: Option<(usize, usize, usize)>,
    token_adapter_rank: usize,
    max_h_steps: usize,
    max_l_steps: usize,
    sequences: usize,
    tokens_per_sequence: usize,
    outer_token_positions_per_optimizer_step: usize,
    batch_tokens_per_optimizer_step: usize,
    normalization: &'static str,
    readback: &'static str,
    plan_mode: &'static str,
    requested_sequence_microbatch_size: Option<usize>,
    requested_state_checkpoint_stride: Option<usize>,
    budget_fraction: f32,
    reserve_bytes: u64,
    warmup_iterations: usize,
    measured_iterations: usize,
    h_backward_schedule: Option<String>,
    l_backward_schedule: Option<String>,
    h_backward_kernel_geometry: Option<String>,
    l_backward_kernel_geometry: Option<String>,
    numerics_policy: String,
    h_backward_schedule_autotuned: bool,
    l_backward_schedule_autotuned: bool,
    median_optimizer_step_ms: f64,
    median_tokens_per_second: f64,
    median_outer_token_positions_per_second: f64,
    samples: Vec<Sample>,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn mean_loss(result: &hierarchos_vulkan::HierarchosBudgetedTokenTapeTrainResult) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for sequence in &result.sequences {
        for &loss in &sequence.losses {
            total += loss;
            count += 1;
        }
    }
    total / count.max(1) as f32
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    std::env::set_var(
        HIERARCHOS_VULKAN_TRAINING_PRECISION_ENV,
        &args.training_precision_policy,
    );
    let case: Case = serde_json::from_slice(&fs::read(&args.case_path)?)?;
    let batch = case
        .steps
        .first()
        .context("benchmark case must contain at least one tape step")?
        .token_ids
        .len();
    if batch == 0 {
        bail!("benchmark case must contain at least one token lane");
    }
    let reserve_bytes = args
        .reserve_mib
        .checked_mul(1024 * 1024)
        .context("--reserve-mib overflows bytes")?;
    let policy = HierarchosTapeMemoryPolicy {
        budget_fraction: args.budget_fraction,
        reserve_bytes,
    };
    let hyper = AdamWHyperParams {
        lr: case.optimizer.lr,
        beta1: case.optimizer.beta1,
        beta2: case.optimizer.beta2,
        eps: case.optimizer.eps,
        weight_decay: case.optimizer.weight_decay,
    };

    let mut steps = Vec::with_capacity(args.tokens_per_sequence);
    for token_index in 0..args.tokens_per_sequence {
        let source = &case.steps[token_index % case.steps.len()];
        steps.push(HierarchosRawTokenTapeStepInput {
            h_steps: 1,
            shadow_steps: 1,
            token_ids: &source.token_ids,
            rosa_reset_lanes: &source.rosa_reset_lanes,
            previous_context: &source.previous_context,
            target_context: &source.target_context,
            context_alpha: source.context_alpha,
            h_token_ids: &source.h_token_ids,
            l_token_ids: &source.l_token_ids,
            h_to_context_grad: &source.h_to_context_grad,
            h_depth_grad: &source.h_depth_grad,
            final_drift_grad: &source.final_drift_grad,
            commitment_cost_grad: &source.commitment_cost_grad,
            ltm_value_alignment_position: token_index as u64,
            ltm_value_alignment_mask: None,
            ltm_value_alignment_grad: 0.0,
            targets: &source.targets,
            supervision_weights: None,
            pytorch_tbptt_token_mask: None,
        });
    }
    let sequences = (0..args.sequences)
        .map(|_| HierarchosRawTokenTapeMicrobatchInput {
            steps: &steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            sequence_context: None,
            pytorch_tbptt_real_token_count: None,
            pytorch_tbptt_chunk_size: None,
            preweighted_ponder_and_commitment: false,
        })
        .collect::<Vec<_>>();
    let h_initial_states = vec![case.h_initial_packed_state.as_slice(); args.sequences];
    let l_initial_states = vec![case.l_initial_packed_state.as_slice(); args.sequences];
    let outer_token_positions = args
        .tokens_per_sequence
        .checked_mul(args.sequences)
        .context("outer token-position count overflow")?;
    let batch_tokens = outer_token_positions
        .checked_mul(batch)
        .context("batch token count overflow")?;

    let device = VulkanDevice::new()?;
    let subgroup_size = device.subgroup_capabilities().subgroup_size;
    let mut graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device,
        &args.model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    graph.set_training_step(args.training_step)?;
    graph.set_rwkv_numerics_policy(args.numerics_policy)?;
    if let (Some(h_geometry), Some(l_geometry)) = (
        args.h_backward_kernel_geometry.as_deref(),
        args.l_backward_kernel_geometry.as_deref(),
    ) {
        graph.set_backward_kernel_geometry_labels(batch, h_geometry, l_geometry)?;
    }

    for _ in 0..args.warmup_iterations {
        let result = if let (Some(microbatch_size), Some(checkpoint_stride)) =
            (args.sequence_microbatch_size, args.state_checkpoint_stride)
        {
            graph.train_raw_token_tape_sequences_with_plan_and_readback_policy(
                batch,
                &h_initial_states,
                &l_initial_states,
                &sequences,
                hyper,
                args.normalization.value(),
                microbatch_size,
                checkpoint_stride,
                policy,
                args.readback.value(),
            )?
        } else {
            graph.train_raw_token_tape_sequences_budgeted_with_readback_policy(
                batch,
                &h_initial_states,
                &l_initial_states,
                &sequences,
                hyper,
                args.normalization.value(),
                policy,
                args.readback.value(),
            )?
        };
        if result.total_tokens != outer_token_positions {
            bail!(
                "warmup processed {} outer tokens; expected {outer_token_positions}",
                result.total_tokens
            );
        }
    }

    let mut samples = Vec::with_capacity(args.measured_iterations);
    for _ in 0..args.measured_iterations {
        let started = Instant::now();
        let result = if let (Some(microbatch_size), Some(checkpoint_stride)) =
            (args.sequence_microbatch_size, args.state_checkpoint_stride)
        {
            graph.train_raw_token_tape_sequences_with_plan_and_readback_policy(
                batch,
                &h_initial_states,
                &l_initial_states,
                &sequences,
                hyper,
                args.normalization.value(),
                microbatch_size,
                checkpoint_stride,
                policy,
                args.readback.value(),
            )?
        } else {
            graph.train_raw_token_tape_sequences_budgeted_with_readback_policy(
                batch,
                &h_initial_states,
                &l_initial_states,
                &sequences,
                hyper,
                args.normalization.value(),
                policy,
                args.readback.value(),
            )?
        };
        let elapsed = started.elapsed().as_secs_f64();
        if result.total_tokens != outer_token_positions {
            bail!(
                "timed step processed {} outer tokens; expected {outer_token_positions}",
                result.total_tokens
            );
        }
        let loss = mean_loss(&result);
        if !loss.is_finite() {
            bail!("timed training submission produced non-finite mean loss");
        }
        samples.push(Sample {
            optimizer_step: result.full_model_optimizer.step,
            elapsed_ms: elapsed * 1_000.0,
            tokens_per_second: batch_tokens as f64 / elapsed,
            outer_token_positions_per_second: outer_token_positions as f64 / elapsed,
            queue_submissions: result.queue_submissions,
            sequence_microbatch_size: result.plan.sequence_microbatch_size,
            sequence_microbatch_count: result.plan.sequence_microbatch_count,
            state_checkpoint_stride: result.plan.state_checkpoint_stride,
            device_local_pressure_bucket: result.plan.device_local_pressure_bucket,
            h_backward_segment_schedule: result.plan.h_backward_segment_schedule.clone(),
            l_backward_segment_schedule: result.plan.l_backward_segment_schedule.clone(),
            h_backward_kernel_geometry: result.plan.h_backward_kernel_geometry.clone(),
            l_backward_kernel_geometry: result.plan.l_backward_kernel_geometry.clone(),
            rwkv_numerics_policy: result.plan.rwkv_numerics_policy.label().to_string(),
            sparse_state_replay: result.plan.requires_sparse_state_replay,
            available_for_tape_bytes: result.plan.available_for_tape_bytes,
            planned_peak_bytes: result.plan.planned_peak_bytes,
            profiled_tokens_per_second: result.plan.profiled_tokens_per_second,
            profile_adaptive_tokens_per_second: result.plan.profile_adaptive_tokens_per_second,
            profile_confidence_adjusted_tokens_per_second: result
                .plan
                .profile_confidence_adjusted_tokens_per_second,
            profile_exploration_score_tokens_per_second: result
                .plan
                .profile_exploration_score_tokens_per_second,
            profile_relative_uncertainty: result.plan.profile_relative_uncertainty,
            profile_effective_measured_iterations: result
                .plan
                .profile_effective_measured_iterations,
            profile_observations_since_last_measurement: result
                .plan
                .profile_observations_since_last_measurement,
            profile_records: result.plan.profile_records,
            profile_measured_iterations: result.plan.profile_measured_iterations,
            online_exploration: result.plan.online_exploration,
            mean_loss: loss,
        });
    }

    let h_backward_schedule = graph.h_recurrent().backward_segment_schedule_label(batch);
    let l_backward_schedule = graph.l_recurrent().backward_segment_schedule_label(batch);
    let h_backward_kernel_geometry = graph
        .h_recurrent()
        .backward_kernel_geometry_label(batch)
        .map(str::to_owned);
    let l_backward_kernel_geometry = graph
        .l_recurrent()
        .backward_kernel_geometry_label(batch)
        .map(str::to_owned);
    let config = graph.config();
    let graph_summary = graph.summary();
    let mixed_precision = graph_summary.mixed_precision_capabilities;
    let expects_fp16_parameter_storage = args.training_precision_policy != "fp32";
    if graph_summary.h_low_rank_fp16_parameter_storage_active != expects_fp16_parameter_storage
        || graph_summary.l_low_rank_fp16_parameter_storage_active != expects_fp16_parameter_storage
        || graph_summary.projection_fp16_parameter_storage_active != expects_fp16_parameter_storage
        || graph_summary.lm_head_fp16_parameter_storage_active != expects_fp16_parameter_storage
    {
        bail!(
            "training precision policy {:?} produced inconsistent packed-FP16 consumers: H-low-rank={} L-low-rank={} projections={} LM-head={}",
            graph_summary.training_precision_policy.label(),
            graph_summary.h_low_rank_fp16_parameter_storage_active,
            graph_summary.l_low_rank_fp16_parameter_storage_active,
            graph_summary.projection_fp16_parameter_storage_active,
            graph_summary.lm_head_fp16_parameter_storage_active,
        );
    }
    let output = Output {
        device: graph_summary.device,
        subgroup_size,
        training_precision_policy: graph_summary.training_precision_policy.label().to_string(),
        h_low_rank_fp16_parameter_storage_active: graph_summary
            .h_low_rank_fp16_parameter_storage_active,
        l_low_rank_fp16_parameter_storage_active: graph_summary
            .l_low_rank_fp16_parameter_storage_active,
        h_low_rank_native_fp16_backward_compute_active: graph_summary
            .h_low_rank_native_fp16_backward_compute_active,
        l_low_rank_native_fp16_backward_compute_active: graph_summary
            .l_low_rank_native_fp16_backward_compute_active,
        h_low_rank_native_fp16_parameter_grad_compute_active: graph_summary
            .h_low_rank_native_fp16_parameter_grad_compute_active,
        l_low_rank_native_fp16_parameter_grad_compute_active: graph_summary
            .l_low_rank_native_fp16_parameter_grad_compute_active,
        h_low_rank_parameter_grad_arithmetic: graph_summary
            .h_low_rank_parameter_grad_arithmetic
            .label()
            .to_string(),
        l_low_rank_parameter_grad_arithmetic: graph_summary
            .l_low_rank_parameter_grad_arithmetic
            .label()
            .to_string(),
        h_low_rank_fp16_full_forward_first_stage_arm: graph_summary
            .h_low_rank_fp16_full_forward_first_stage_arm
            .map(str::to_string),
        l_low_rank_fp16_full_forward_first_stage_arm: graph_summary
            .l_low_rank_fp16_full_forward_first_stage_arm
            .map(str::to_string),
        projection_fp16_parameter_storage_active: graph_summary
            .projection_fp16_parameter_storage_active,
        projection_native_fp16_backward_compute_active: graph_summary
            .projection_native_fp16_backward_compute_active,
        lm_head_fp16_parameter_storage_active: graph_summary.lm_head_fp16_parameter_storage_active,
        lm_head_execution_arm: graph_summary.lm_head_execution_arm.label().to_string(),
        lm_head_weight_grad_topology: graph_summary
            .lm_head_weight_grad_topology
            .map(|topology| topology.label().to_string()),
        lm_head_fused_adjoint_topology: graph_summary
            .lm_head_fused_adjoint_topology
            .map(str::to_string),
        lm_head_native_fp16_backward_compute_active: graph_summary
            .lm_head_native_fp16_backward_compute_active,
        out_norm_native_fp16_backward_compute_active: graph_summary
            .out_norm_native_fp16_backward_compute_active,
        storage_buffer_16_bit_access_enabled: mixed_precision.storage_buffer_16_bit_access_enabled,
        shader_float16_enabled: mixed_precision.shader_float16_enabled,
        native_fp16_storage_compute_ready: mixed_precision.native_fp16_storage_compute_ready(),
        shader_bfloat16_extension_exposed: mixed_precision.shader_bfloat16_extension_exposed,
        backward_kernel_geometry_revision: BACKWARD_KERNEL_GEOMETRY_POLICY_REVISION,
        architecture_revision: config.architecture_revision.clone(),
        batch,
        context_dim: config.context_dim,
        persistent_dim: config.persistent_dim,
        ltm_slots: config.ltm_slots,
        ltm_key_dim: config.ltm_key_dim,
        ltm_val_dim: config.ltm_val_dim,
        ltm_topk: config.ltm_topk,
        vocab_size: config.vocab_size,
        h_hidden: config.h_hidden,
        l_hidden: config.l_hidden,
        h_width: graph.h_recurrent().width(),
        l_width: graph.l_recurrent().width(),
        h_state_size: graph.h_recurrent().state_size(),
        l_state_size: graph.l_recurrent().state_size(),
        h_rwkv_head_size: config.h_rwkv_head_size,
        l_rwkv_head_size: config.l_rwkv_head_size,
        h_low_rank_ranks: graph.h_recurrent().low_rank_ranks(),
        l_low_rank_ranks: graph.l_recurrent().low_rank_ranks(),
        token_adapter_rank: config.token_adapter_rank,
        max_h_steps: config.max_h_steps,
        max_l_steps: config.max_l_steps,
        sequences: args.sequences,
        tokens_per_sequence: args.tokens_per_sequence,
        outer_token_positions_per_optimizer_step: outer_token_positions,
        batch_tokens_per_optimizer_step: batch_tokens,
        normalization: args.normalization.label(),
        readback: args.readback.label(),
        plan_mode: if args.sequence_microbatch_size.is_some() {
            "explicit"
        } else {
            "automatic"
        },
        requested_sequence_microbatch_size: args.sequence_microbatch_size,
        requested_state_checkpoint_stride: args.state_checkpoint_stride,
        budget_fraction: args.budget_fraction,
        reserve_bytes,
        warmup_iterations: args.warmup_iterations,
        measured_iterations: args.measured_iterations,
        h_backward_schedule,
        l_backward_schedule,
        h_backward_kernel_geometry,
        l_backward_kernel_geometry,
        numerics_policy: graph.h_recurrent().numerics_policy().label().to_string(),
        h_backward_schedule_autotuned: graph
            .h_recurrent()
            .backward_segment_schedule_was_autotuned(batch),
        l_backward_schedule_autotuned: graph
            .l_recurrent()
            .backward_segment_schedule_was_autotuned(batch),
        median_optimizer_step_ms: median(samples.iter().map(|sample| sample.elapsed_ms).collect()),
        median_tokens_per_second: median(
            samples
                .iter()
                .map(|sample| sample.tokens_per_second)
                .collect(),
        ),
        median_outer_token_positions_per_second: median(
            samples
                .iter()
                .map(|sample| sample.outer_token_positions_per_second)
                .collect(),
        ),
        samples,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
