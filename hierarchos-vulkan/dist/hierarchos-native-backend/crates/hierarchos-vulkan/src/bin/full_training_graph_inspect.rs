use std::path::PathBuf;

use anyhow::{Context, Result};
use hierarchos_vulkan::{HierarchosTapeMemoryPolicy, HierarchosTrainingGraph, VulkanDevice};
use serde::Serialize;

#[derive(Serialize)]
struct Output {
    device: String,
    context_dim: usize,
    h_hidden: usize,
    l_hidden: usize,
    vocab_size: usize,
    projection_tensor_count: usize,
    shared_lm_head_identity: bool,
    h_low_rank_parameter_grad_arithmetic: String,
    l_low_rank_parameter_grad_arithmetic: String,
    live_buffer_count: usize,
    live_buffer_bytes: usize,
    driver_allocation_count: usize,
    reserved_bytes: usize,
    max_driver_allocation_count: u32,
    memory_budget_extension_supported: bool,
    device_local_heap_size_bytes: u64,
    device_local_budget_bytes: u64,
    device_local_usage_bytes: u64,
    device_local_available_bytes: u64,
    hierarchos_reserved_bytes: u64,
    scratch_slab_capacity_bytes: usize,
    scratch_slab_live_bytes: usize,
    training_working_set_logical_bytes: usize,
    training_working_set_planned_bytes: usize,
    training_working_set_reused_bytes: usize,
    training_working_set_slot_count: usize,
    training_working_set_bindings: Vec<TrainingWorkingSetBindingOutput>,
    estimated_vulkan_training_peak_bytes: Option<u64>,
    pytorch_reference_peak_bytes: Option<u64>,
    vulkan_to_pytorch_peak_ratio: Option<f64>,
    token_tape_plan: Option<TokenTapePlanOutput>,
}

#[derive(Serialize)]
struct TrainingWorkingSetBindingOutput {
    name: &'static str,
    f32_len: usize,
    bytes: usize,
    begin: &'static str,
    end: &'static str,
    intervals: Vec<TrainingWorkingSetIntervalOutput>,
    slot: usize,
}

#[derive(Serialize)]
struct TrainingWorkingSetIntervalOutput {
    epoch: &'static str,
    begin: &'static str,
    end: &'static str,
    repeated: bool,
}

#[derive(Serialize)]
struct TokenTapePlanOutput {
    requested_sequences: usize,
    requested_tokens_per_sequence: usize,
    sequence_microbatch_size: usize,
    sequence_microbatch_count: usize,
    state_checkpoint_stride: usize,
    requires_sparse_state_replay: bool,
    current_backend_executable: bool,
    working_set_limit_bytes: u64,
    available_for_tape_bytes: u64,
    dense_requested_peak_bytes: u64,
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
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut max_batch = 1usize;
    let mut max_h_steps = 2usize;
    let mut max_l_steps = 2usize;
    let mut max_loss_rows = 2usize;
    let mut plan_sequences = None;
    let mut plan_tokens = None;
    let mut pytorch_peak_bytes = None;
    let mut tape_policy = HierarchosTapeMemoryPolicy::default();
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--max-batch" => max_batch = parse_usize(args.next(), "--max-batch")?,
            "--max-h-steps" => max_h_steps = parse_usize(args.next(), "--max-h-steps")?,
            "--max-l-steps" => max_l_steps = parse_usize(args.next(), "--max-l-steps")?,
            "--max-loss-rows" => max_loss_rows = parse_usize(args.next(), "--max-loss-rows")?,
            "--plan-sequences" => {
                plan_sequences = Some(parse_usize(args.next(), "--plan-sequences")?)
            }
            "--plan-tokens" => plan_tokens = Some(parse_usize(args.next(), "--plan-tokens")?),
            "--pytorch-peak-bytes" => {
                pytorch_peak_bytes = Some(parse_u64(args.next(), "--pytorch-peak-bytes")?)
            }
            "--tape-budget-fraction" => {
                tape_policy.budget_fraction = parse_f32(args.next(), "--tape-budget-fraction")?
            }
            "--tape-reserve-mib" => {
                let mib = parse_usize(args.next(), "--tape-reserve-mib")?;
                tape_policy.reserve_bytes = u64::try_from(mib)
                    .context("--tape-reserve-mib exceeds u64 range")?
                    .checked_mul(1024 * 1024)
                    .context("--tape-reserve-mib byte conversion overflow")?;
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let model_dir = model_dir.context(
        "usage: --model MODEL_DIR [--max-batch N --max-h-steps N --max-l-steps N --max-loss-rows N] [--plan-sequences N --plan-tokens N --tape-budget-fraction F --tape-reserve-mib N] [--pytorch-peak-bytes N]",
    )?;
    let device = VulkanDevice::new()?;
    let graph = HierarchosTrainingGraph::from_model_package(
        device,
        model_dir,
        max_batch,
        max_h_steps,
        max_l_steps,
        max_loss_rows,
    )?;
    let summary = graph.summary();
    let memory = graph.memory_stats()?;
    let budget = graph.memory_budget()?;
    let submission = graph.submission_arena_stats()?;
    let training_working_set_bindings = graph
        .training_working_set_plan()
        .entries
        .iter()
        .map(|entry| TrainingWorkingSetBindingOutput {
            name: entry.name,
            f32_len: entry.f32_len,
            bytes: entry.bytes,
            begin: entry.lifetime.begin.label(),
            end: entry.lifetime.end.label(),
            intervals: entry
                .lifetime
                .intervals
                .iter()
                .map(|interval| TrainingWorkingSetIntervalOutput {
                    epoch: interval.epoch.label(),
                    begin: interval.begin.label(),
                    end: interval.end.label(),
                    repeated: interval.repeated,
                })
                .collect(),
            slot: entry.slot,
        })
        .collect::<Vec<_>>();
    let token_tape_plan = match (plan_sequences, plan_tokens) {
        (None, None) => None,
        (Some(sequences), Some(tokens)) => {
            let plan = graph.plan_token_tape_memory(max_batch, sequences, tokens, tape_policy)?;
            Some(TokenTapePlanOutput {
                requested_sequences: plan.requested_sequences,
                requested_tokens_per_sequence: plan.requested_tokens_per_sequence,
                sequence_microbatch_size: plan.sequence_microbatch_size,
                sequence_microbatch_count: plan.sequence_microbatch_count,
                state_checkpoint_stride: plan.state_checkpoint_stride,
                requires_sparse_state_replay: plan.requires_sparse_state_replay,
                current_backend_executable: plan.current_backend_executable,
                working_set_limit_bytes: plan.working_set_limit_bytes,
                available_for_tape_bytes: plan.available_for_tape_bytes,
                dense_requested_peak_bytes: plan.dense_requested_peak_bytes,
                planned_peak_bytes: plan.planned_peak_bytes,
                profiled_tokens_per_second: plan.profiled_tokens_per_second,
                profile_adaptive_tokens_per_second: plan.profile_adaptive_tokens_per_second,
                profile_confidence_adjusted_tokens_per_second: plan
                    .profile_confidence_adjusted_tokens_per_second,
                profile_exploration_score_tokens_per_second: plan
                    .profile_exploration_score_tokens_per_second,
                profile_relative_uncertainty: plan.profile_relative_uncertainty,
                profile_effective_measured_iterations: plan.profile_effective_measured_iterations,
                profile_observations_since_last_measurement: plan
                    .profile_observations_since_last_measurement,
                profile_records: plan.profile_records,
                profile_measured_iterations: plan.profile_measured_iterations,
            })
        }
        _ => anyhow::bail!("--plan-sequences and --plan-tokens must be supplied together"),
    };
    let scratch_slab_live_bytes = submission
        .scratch_slab_capacity_bytes
        .saturating_sub(submission.scratch_slab_free_bytes);
    let estimated_vulkan_training_peak_bytes = token_tape_plan
        .as_ref()
        .map(|plan| {
            u64::try_from(memory.live_buffer_bytes)
                .context("live Vulkan buffer bytes exceed u64 range")?
                .checked_add(
                    u64::try_from(scratch_slab_live_bytes)
                        .context("live Vulkan scratch slab bytes exceed u64 range")?,
                )
                .context("Vulkan graph working-set byte estimate overflow")?
                .checked_add(plan.planned_peak_bytes)
                .context("Vulkan training peak byte estimate overflow")
        })
        .transpose()?;
    let vulkan_to_pytorch_peak_ratio = estimated_vulkan_training_peak_bytes
        .zip(pytorch_peak_bytes)
        .map(|(vulkan, pytorch)| vulkan as f64 / pytorch as f64);
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: summary.device,
            context_dim: summary.context_dim,
            h_hidden: summary.h_hidden,
            l_hidden: summary.l_hidden,
            vocab_size: summary.vocab_size,
            projection_tensor_count: summary.projection_tensor_count,
            shared_lm_head_identity: summary.shared_lm_head_identity,
            h_low_rank_parameter_grad_arithmetic: summary
                .h_low_rank_parameter_grad_arithmetic
                .label()
                .to_string(),
            l_low_rank_parameter_grad_arithmetic: summary
                .l_low_rank_parameter_grad_arithmetic
                .label()
                .to_string(),
            live_buffer_count: memory.live_buffer_count,
            live_buffer_bytes: memory.live_buffer_bytes,
            driver_allocation_count: memory.driver_allocation_count,
            reserved_bytes: memory.reserved_bytes,
            max_driver_allocation_count: memory.max_driver_allocation_count,
            memory_budget_extension_supported: budget.budget_extension_supported,
            device_local_heap_size_bytes: budget.device_local_heap_size_bytes,
            device_local_budget_bytes: budget.device_local_budget_bytes,
            device_local_usage_bytes: budget.device_local_usage_bytes,
            device_local_available_bytes: budget.device_local_available_bytes,
            hierarchos_reserved_bytes: budget.hierarchos_reserved_bytes,
            scratch_slab_capacity_bytes: submission.scratch_slab_capacity_bytes,
            scratch_slab_live_bytes,
            training_working_set_logical_bytes: summary.training_working_set_logical_bytes,
            training_working_set_planned_bytes: summary.training_working_set_planned_bytes,
            training_working_set_reused_bytes: summary.training_working_set_reused_bytes,
            training_working_set_slot_count: summary.training_working_set_slot_count,
            training_working_set_bindings,
            estimated_vulkan_training_peak_bytes,
            pytorch_reference_peak_bytes: pytorch_peak_bytes,
            vulkan_to_pytorch_peak_ratio,
            token_tape_plan,
        })?
    );
    Ok(())
}

fn parse_usize(value: Option<std::ffi::OsString>, name: &str) -> Result<usize> {
    let value = value.with_context(|| format!("{name} requires a value"))?;
    let parsed = value
        .to_string_lossy()
        .parse::<usize>()
        .with_context(|| format!("{name} must be a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{name} must be positive");
    }
    Ok(parsed)
}

fn parse_f32(value: Option<std::ffi::OsString>, name: &str) -> Result<f32> {
    let value = value.with_context(|| format!("{name} requires a value"))?;
    let parsed = value
        .to_string_lossy()
        .parse::<f32>()
        .with_context(|| format!("{name} must be a finite decimal value"))?;
    if !parsed.is_finite() {
        anyhow::bail!("{name} must be finite");
    }
    Ok(parsed)
}

fn parse_u64(value: Option<std::ffi::OsString>, name: &str) -> Result<u64> {
    let value = value.with_context(|| format!("{name} requires a value"))?;
    let parsed = value
        .to_string_lossy()
        .parse::<u64>()
        .with_context(|| format!("{name} must be a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{name} must be positive");
    }
    Ok(parsed)
}
