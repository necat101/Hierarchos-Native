use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, HierarchosExecutionPolicyState, HierarchosLabeledSequenceObjective,
    HierarchosLossScalingState, HierarchosPortableReplayTensor, HierarchosPortableTrainingReplay,
    HierarchosRawTokenLabeledSequenceInput, HierarchosStochasticRngPolicyState,
    HierarchosTapeMemoryPolicy, HierarchosTokenTapeMemoryPlan, HierarchosTokenTapeReadbackPolicy,
    HierarchosTokenTapeUpdateMode, HierarchosTrainingCheckpointManifest, HierarchosTrainingGraph,
    HierarchosTrainingPrecisionPolicy, HierarchosTrainingSessionState, VulkanDevice,
    HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT, HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT,
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
struct ObjectiveCase {
    z_loss_weight: f32,
    ponder_loss_weight: f32,
    commitment_loss_weight: f32,
    #[serde(default)]
    max_ce_loss_for_backward: f32,
    #[serde(default)]
    max_ponder_cost_for_backward: f32,
    #[serde(default = "default_max_commitment_cost_for_backward")]
    max_commitment_cost_for_backward: f32,
}

fn default_max_commitment_cost_for_backward() -> f32 {
    2.0
}

#[derive(Deserialize)]
struct UpdateCase {
    input_ids: Vec<u32>,
    labels: Vec<i64>,
    attention_mask: Option<Vec<f32>>,
    loss_weights: Option<Vec<f32>>,
    initial_previous_context: Vec<f32>,
    initial_target_context: Vec<f32>,
    h_initial_packed_state: Vec<f32>,
    l_initial_packed_state: Vec<f32>,
    global_pos_offset: u64,
    reset_rosa_at_start: bool,
    pytorch_tbptt_chunk_size: Option<usize>,
}

#[derive(Deserialize)]
struct Case {
    batch: usize,
    tokens: usize,
    max_h_steps: usize,
    max_l_steps: usize,
    #[serde(default = "default_gradient_accumulation_steps")]
    gradient_accumulation_steps: usize,
    #[serde(default)]
    leave_final_accumulation_open: bool,
    #[serde(default)]
    resume_open_accumulation: bool,
    #[serde(default)]
    dynamic_loss_scale: Option<f64>,
    #[serde(default)]
    dynamic_loss_scale_growth_tracker: u64,
    #[serde(default)]
    expected_dynamic_loss_scale_overflows: usize,
    #[serde(default = "default_grad_clip")]
    grad_clip: f32,
    #[serde(default)]
    capture_pending_gradients: bool,
    #[serde(flatten)]
    first_update: UpdateCase,
    #[serde(default)]
    additional_updates: Vec<UpdateCase>,
    objective: ObjectiveCase,
    optimizer: OptimizerCase,
}

#[derive(Serialize)]
struct NamedValues {
    name: String,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device_index: usize,
    device_name: String,
    batch: usize,
    tokens: usize,
    updates: usize,
    gradient_accumulation_steps: usize,
    requested_sequence_microbatch_size: Option<usize>,
    requested_state_checkpoint_stride: Option<usize>,
    dynamic_loss_scale: Option<f64>,
    dynamic_loss_scale_after: Option<f64>,
    dynamic_loss_scale_growth_tracker: Option<u64>,
    dynamic_loss_scale_overflow_count: usize,
    dynamic_loss_scale_window_overflowed: Vec<bool>,
    dynamic_loss_scale_window_stepped: Vec<bool>,
    dynamic_loss_scale_window_scale_before: Vec<f64>,
    dynamic_loss_scale_window_scale_after: Vec<f64>,
    grad_clip: f32,
    training_precision_policy: String,
    h_low_rank_native_fp16_parameter_grad_compute_active: bool,
    l_low_rank_native_fp16_parameter_grad_compute_active: bool,
    h_low_rank_parameter_grad_arithmetic: String,
    l_low_rank_parameter_grad_arithmetic: String,
    queue_submissions: u32,
    memory_live_buffer_count: usize,
    memory_live_buffer_bytes: usize,
    memory_driver_allocation_count: usize,
    memory_reserved_bytes: usize,
    memory_max_driver_allocation_count: u32,
    memory_budget_extension_supported: bool,
    device_local_heap_size_bytes: u64,
    device_local_budget_bytes: u64,
    device_local_usage_bytes: u64,
    device_local_available_bytes: u64,
    optimizer_step: u32,
    training_elapsed_ms: f64,
    optimizer_window_ms: Vec<f64>,
    optimizer_window_gradient_norms: Vec<f64>,
    optimizer_window_clip_coefficients: Vec<f32>,
    accumulation_open: bool,
    pending_gradient_tensor_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pending_gradients_before_step: Vec<Vec<NamedValues>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    low_rank_g2_per_use: Vec<G2TraceOutput>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    budgeted_plans: Vec<BudgetedPlanOutput>,
    losses: Vec<f32>,
    losses_finite: bool,
    final_h_packed_states: Vec<Vec<f32>>,
    final_l_packed_states: Vec<Vec<f32>>,
    parameter_count: usize,
}

/// Scheduler evidence for the exact budgeted plan that executed one optimizer
/// window. These are execution-policy coordinates only; checkpoint tensor
/// names, layouts, dtypes, and optimizer semantics remain canonical.
#[derive(Serialize)]
struct BudgetedPlanOutput {
    update_start: usize,
    update_end: usize,
    sequence_microbatch_size: usize,
    state_checkpoint_stride: usize,
    h_backward_segment_schedule: Option<String>,
    l_backward_segment_schedule: Option<String>,
    h_backward_kernel_geometry: Option<String>,
    l_backward_kernel_geometry: Option<String>,
    rwkv_numerics_policy: String,
    requires_sparse_state_replay: bool,
    device_local_pressure_bucket: Option<u8>,
    profiled_tokens_per_second: Option<f64>,
    profile_confidence_adjusted_tokens_per_second: Option<f64>,
    profile_exploration_score_tokens_per_second: Option<f64>,
    profile_relative_uncertainty: Option<f64>,
    profile_records: usize,
    profile_measured_iterations: usize,
    online_exploration: bool,
}

impl BudgetedPlanOutput {
    fn from_plan(
        update_start: usize,
        update_end: usize,
        plan: &HierarchosTokenTapeMemoryPlan,
    ) -> Self {
        Self {
            update_start,
            update_end,
            sequence_microbatch_size: plan.sequence_microbatch_size,
            state_checkpoint_stride: plan.state_checkpoint_stride,
            h_backward_segment_schedule: plan.h_backward_segment_schedule.clone(),
            l_backward_segment_schedule: plan.l_backward_segment_schedule.clone(),
            h_backward_kernel_geometry: plan.h_backward_kernel_geometry.clone(),
            l_backward_kernel_geometry: plan.l_backward_kernel_geometry.clone(),
            rwkv_numerics_policy: plan.rwkv_numerics_policy.label().to_string(),
            requires_sparse_state_replay: plan.requires_sparse_state_replay,
            device_local_pressure_bucket: plan.device_local_pressure_bucket,
            profiled_tokens_per_second: plan.profiled_tokens_per_second,
            profile_confidence_adjusted_tokens_per_second: plan
                .profile_confidence_adjusted_tokens_per_second,
            profile_exploration_score_tokens_per_second: plan
                .profile_exploration_score_tokens_per_second,
            profile_relative_uncertainty: plan.profile_relative_uncertainty,
            profile_records: plan.profile_records,
            profile_measured_iterations: plan.profile_measured_iterations,
            online_exploration: plan.online_exploration,
        }
    }
}

#[derive(Serialize)]
struct G2TraceOutput {
    update_index: usize,
    h_uses: Vec<Vec<f32>>,
    l_uses: Vec<Vec<f32>>,
}

fn default_gradient_accumulation_steps() -> usize {
    1
}

fn default_grad_clip() -> f32 {
    1.0
}

fn restored_accumulation_update_count(
    manifest: Option<&HierarchosTrainingCheckpointManifest>,
    tokens_per_update: usize,
    accumulation_steps: usize,
) -> Result<usize> {
    let Some(manifest) = manifest.filter(|manifest| manifest.accumulation_open) else {
        return Ok(0);
    };
    anyhow::ensure!(
        tokens_per_update > 0,
        "restored accumulation requires tokens > 0"
    );
    anyhow::ensure!(
        accumulation_steps > 1,
        "an open restored accumulation requires gradient_accumulation_steps > 1"
    );
    let tokens_per_update = u64::try_from(tokens_per_update)
        .context("labeled parity token width exceeds portable u64 accounting")?;
    let consumed_tokens = manifest.accumulation_consumed_token_count;
    anyhow::ensure!(
        consumed_tokens > 0 && consumed_tokens % tokens_per_update == 0,
        "open restored accumulation consumed {consumed_tokens} tokens, which is not a positive multiple of the labeled parity update width {tokens_per_update}"
    );
    let consumed_updates = usize::try_from(consumed_tokens / tokens_per_update)
        .context("restored accumulation update count exceeds host usize")?;
    anyhow::ensure!(
        consumed_updates < accumulation_steps,
        "open restored accumulation contains {consumed_updates} microbatches for a {accumulation_steps}-way window; a complete window must not remain open"
    );
    Ok(consumed_updates)
}

#[cfg(test)]
mod tests {
    use super::restored_accumulation_update_count;
    use hierarchos_vulkan::HierarchosTrainingCheckpointManifest;

    fn open_manifest(consumed_tokens: u64) -> HierarchosTrainingCheckpointManifest {
        serde_json::from_value(serde_json::json!({
            "format": "hierarchos-vulkan-training-v6",
            "architecture_revision": "test",
            "model_file": "model.safetensors",
            "optimizer_file": "optimizer.safetensors",
            "optimizer_step": 0,
            "optimizer_tensor_count": 0,
            "training_step": null,
            "training_precision_policy": "fp32",
            "accumulation_open": true,
            "accumulation_consumed_token_count": consumed_tokens
        }))
        .expect("minimal checkpoint manifest should deserialize")
    }

    #[test]
    fn restored_accumulation_recovers_multi_microbatch_prefix() {
        let manifest = open_manifest(18);
        assert_eq!(
            restored_accumulation_update_count(Some(&manifest), 6, 4).unwrap(),
            3
        );
    }

    #[test]
    fn restored_accumulation_rejects_ambiguous_or_complete_prefix() {
        let misaligned = open_manifest(17);
        assert!(restored_accumulation_update_count(Some(&misaligned), 6, 4).is_err());

        let complete = open_manifest(24);
        assert!(restored_accumulation_update_count(Some(&complete), 6, 4).is_err());
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut case_path = None;
    let mut output_package = None;
    let mut device_index = None;
    let mut budgeted_windows = false;
    let mut sequence_microbatch_size = None;
    let mut state_checkpoint_stride = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--case" => case_path = args.next().map(PathBuf::from),
            "--output-package" => output_package = args.next().map(PathBuf::from),
            "--device-index" => {
                let raw = args
                    .next()
                    .context("--device-index requires a non-negative integer")?;
                device_index = Some(
                    raw.to_string_lossy()
                        .parse::<usize>()
                        .context("parsing --device-index")?,
                );
            }
            "--budgeted-windows" => budgeted_windows = true,
            "--sequence-microbatch-size" => {
                let raw = args
                    .next()
                    .context("--sequence-microbatch-size requires a positive integer")?;
                sequence_microbatch_size = Some(
                    raw.to_string_lossy()
                        .parse::<usize>()
                        .context("parsing --sequence-microbatch-size")?,
                );
            }
            "--state-checkpoint-stride" => {
                let raw = args
                    .next()
                    .context("--state-checkpoint-stride requires a positive integer")?;
                state_checkpoint_stride = Some(
                    raw.to_string_lossy()
                        .parse::<usize>()
                        .context("parsing --state-checkpoint-stride")?,
                );
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let model_dir = model_dir.context("missing --model MODEL_DIR")?;
    let case_path = case_path.context("missing --case CASE.json")?;
    let output_package = output_package.context("missing --output-package DIR")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    anyhow::ensure!(case.batch > 0, "labeled parity case requires batch > 0");
    anyhow::ensure!(case.tokens > 0, "labeled parity case requires tokens > 0");
    anyhow::ensure!(
        case.gradient_accumulation_steps > 0,
        "gradient_accumulation_steps must be positive"
    );
    anyhow::ensure!(
        case.grad_clip.is_finite() && case.grad_clip >= 0.0,
        "grad_clip must be finite and non-negative"
    );
    anyhow::ensure!(
        sequence_microbatch_size.is_some() == state_checkpoint_stride.is_some(),
        "--sequence-microbatch-size and --state-checkpoint-stride must be supplied together"
    );
    if let (Some(sequence_microbatch_size), Some(state_checkpoint_stride)) =
        (sequence_microbatch_size, state_checkpoint_stride)
    {
        anyhow::ensure!(
            budgeted_windows,
            "an explicit tape plan requires --budgeted-windows"
        );
        anyhow::ensure!(
            sequence_microbatch_size > 0 && state_checkpoint_stride > 0,
            "explicit tape geometry must be positive"
        );
    }
    let update_count = 1 + case.additional_updates.len();
    if case.leave_final_accumulation_open {
        anyhow::ensure!(
            case.gradient_accumulation_steps > 1,
            "leave_final_accumulation_open requires gradient_accumulation_steps > 1"
        );
    }
    if let Some(scale) = case.dynamic_loss_scale {
        anyhow::ensure!(
            scale.is_finite() && scale > 0.0,
            "dynamic_loss_scale must be finite and positive"
        );
    }
    if budgeted_windows {
        anyhow::ensure!(
            !case.leave_final_accumulation_open
                && !case.resume_open_accumulation
                && !case.capture_pending_gradients,
            "--budgeted-windows requires complete fresh accumulation windows without pending-gradient capture"
        );
    }

    let device = match device_index {
        Some(index) => VulkanDevice::new_with_index(index)?,
        None => VulkanDevice::new()?,
    };
    let selected_device_index = device.physical_device_index();
    let selected_device_name = device.name().to_owned();
    let mut graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device,
        &model_dir,
        case.batch,
        case.max_h_steps,
        case.max_l_steps,
        case.batch,
        case.batch,
    )?;
    let restored_manifest = if model_dir.join("training_state.json").is_file() {
        Some(
            graph
                .load_training_checkpoint_package_state(&model_dir)
                .context("restoring portable Vulkan training state before labeled continuation")?,
        )
    } else {
        None
    };
    let restored_accumulation_open = restored_manifest
        .as_ref()
        .is_some_and(|manifest| manifest.accumulation_open);
    anyhow::ensure!(
        case.resume_open_accumulation == restored_accumulation_open,
        "resume_open_accumulation={} but restored package accumulation_open={restored_accumulation_open}",
        case.resume_open_accumulation,
    );
    let restored_update_count = restored_accumulation_update_count(
        restored_manifest.as_ref(),
        case.tokens,
        case.gradient_accumulation_steps,
    )?;
    let scheduled_update_count = update_count
        .checked_add(restored_update_count)
        .context("labeled parity scheduled update count overflow")?;
    if case.leave_final_accumulation_open {
        anyhow::ensure!(
            scheduled_update_count % case.gradient_accumulation_steps != 0,
            "leave_final_accumulation_open requires an incomplete final accumulation group after the restored prefix"
        );
    }
    if let Some(session) = restored_manifest
        .as_ref()
        .and_then(|manifest| manifest.training_session.as_ref())
    {
        if let Some(saved) = session
            .effective_training_config
            .get("gradient_accumulation_steps")
            .and_then(serde_json::Value::as_u64)
        {
            anyhow::ensure!(
                saved == case.gradient_accumulation_steps as u64,
                "restored gradient_accumulation_steps={saved} does not match continuation {}",
                case.gradient_accumulation_steps
            );
        }
        if let Some(saved) = session
            .effective_training_config
            .get("grad_clip")
            .and_then(serde_json::Value::as_f64)
        {
            anyhow::ensure!(
                (saved - case.grad_clip as f64).abs() <= f64::from(f32::EPSILON),
                "restored grad_clip={saved} does not match continuation {}",
                case.grad_clip
            );
        }
    }
    let hyper = AdamWHyperParams {
        lr: case.optimizer.lr,
        beta1: case.optimizer.beta1,
        beta2: case.optimizer.beta2,
        eps: case.optimizer.eps,
        weight_decay: case.optimizer.weight_decay,
    };
    let objective = HierarchosLabeledSequenceObjective {
        z_loss_weight: case.objective.z_loss_weight,
        ponder_loss_weight: case.objective.ponder_loss_weight,
        commitment_loss_weight: case.objective.commitment_loss_weight,
        max_ce_loss_for_backward: case.objective.max_ce_loss_for_backward,
        max_ponder_cost_for_backward: case.objective.max_ponder_cost_for_backward,
        max_commitment_cost_for_backward: case.objective.max_commitment_cost_for_backward,
    };
    let mut updates = Vec::with_capacity(1 + case.additional_updates.len());
    updates.push(&case.first_update);
    updates.extend(case.additional_updates.iter());
    let mut losses: Vec<f32> = Vec::with_capacity(case.tokens * updates.len());
    let mut final_h_packed_states: Vec<Vec<f32>> = Vec::with_capacity(updates.len());
    let mut final_l_packed_states: Vec<Vec<f32>> = Vec::with_capacity(updates.len());
    let mut queue_submissions = 0u32;
    let mut optimizer_step = 0u32;
    let mut pending_gradients_before_step = Vec::new();
    let trace_low_rank_g2 = std::env::var_os("HIERARCHOS_VULKAN_DIAGNOSTIC_G2_PER_USE").is_some();
    anyhow::ensure!(
        !budgeted_windows || !trace_low_rank_g2,
        "--budgeted-windows does not support the per-use low-rank g2 diagnostic"
    );
    let mut low_rank_g2_per_use = Vec::new();
    let mut budgeted_plans = Vec::new();
    let training_started = Instant::now();
    let mut optimizer_window_started = None;
    let mut optimizer_window_ms = Vec::new();
    let mut optimizer_window_gradient_norms = Vec::new();
    let mut optimizer_window_clip_coefficients = Vec::new();
    let mut dynamic_loss_scale_overflow_count = 0usize;
    let mut dynamic_loss_scale_window_overflowed = Vec::new();
    let mut dynamic_loss_scale_window_stepped = Vec::new();
    let mut dynamic_loss_scale_window_scale_before = Vec::new();
    let mut dynamic_loss_scale_window_scale_after = Vec::new();
    let restored_loss_scaling = restored_manifest
        .as_ref()
        .and_then(|manifest| manifest.training_session.as_ref())
        .and_then(|session| session.execution_policy.as_ref())
        .map(|policy| policy.loss_scaling.clone());
    let mut dynamic_loss_scaling = match restored_loss_scaling {
        Some(loss_scaling) if loss_scaling.mode == "dynamic" => {
            if let Some(requested_scale) = case.dynamic_loss_scale {
                let restored_scale = loss_scaling
                    .scale
                    .context("restored dynamic loss scaler is missing scale")?;
                anyhow::ensure!(
                    requested_scale == restored_scale,
                    "requested dynamic loss scale {requested_scale} does not match restored scale {restored_scale}"
                );
            }
            Some(loss_scaling)
        }
        Some(loss_scaling) if loss_scaling.mode == "none" => {
            anyhow::ensure!(
                !restored_accumulation_open || case.dynamic_loss_scale.is_none(),
                "cannot introduce dynamic loss scaling while resuming an unscaled open window"
            );
            case.dynamic_loss_scale
                .map(|scale| HierarchosLossScalingState {
                    mode: "dynamic".to_string(),
                    scale: Some(scale),
                    growth_factor: Some(1.0),
                    backoff_factor: Some(0.5),
                    growth_interval: Some(u64::MAX),
                    growth_tracker: Some(case.dynamic_loss_scale_growth_tracker),
                    pending_gradients_scaled: false,
                })
        }
        Some(loss_scaling) => anyhow::bail!(
            "labeled parity continuation does not support restored loss-scaling mode {:?}",
            loss_scaling.mode
        ),
        None => case
            .dynamic_loss_scale
            .map(|scale| HierarchosLossScalingState {
                mode: "dynamic".to_string(),
                scale: Some(scale),
                growth_factor: Some(1.0),
                backoff_factor: Some(0.5),
                growth_interval: Some(u64::MAX),
                growth_tracker: Some(case.dynamic_loss_scale_growth_tracker),
                pending_gradients_scaled: false,
            }),
    };
    if restored_accumulation_open {
        if let Some(loss_scaling) = dynamic_loss_scaling.as_mut() {
            queue_submissions = queue_submissions
                .checked_add(
                    graph
                        .rehydrate_full_model_accumulation_for_dynamic_loss_scaling(loss_scaling)?,
                )
                .context("resume dynamic loss-scale rehydration submission count overflow")?;
        }
    }
    for (update_index, update) in updates.iter().copied().enumerate() {
        if trace_low_rank_g2 {
            graph
                .begin_low_rank_g2_gradient_trace()
                .with_context(|| format!("arming g2 per-use trace for update {update_index}"))?;
        }
        // A restored open package contributes one already-accumulated virtual
        // microbatch before the first update supplied by this continuation
        // case. Offset the schedule by that prior microbatch instead of
        // subtracting the first live update from the schedule: for a 2-way
        // window, one restored microbatch plus updates[0] must close the window.
        let schedule_index = update_index + restored_update_count;
        let group_offset = schedule_index % case.gradient_accumulation_steps;
        let group_start = group_offset == 0;
        let final_scheduled_update = schedule_index + 1 == scheduled_update_count;
        let group_end = group_offset + 1 == case.gradient_accumulation_steps
            || (final_scheduled_update && !case.leave_final_accumulation_open);
        if group_start {
            optimizer_window_started = Some(Instant::now());
        }
        // Every certificate path leaves AdamW open until the same global
        // gradient-safety boundary used by the production trainer has measured
        // and clipped the complete optimizer window. This prevents a parity
        // certificate from silently qualifying an unclipped RWKV trajectory.
        let accumulation_mode = if case.resume_open_accumulation && update_index == 0 {
            HierarchosTokenTapeUpdateMode::Accumulate
        } else if group_start {
            HierarchosTokenTapeUpdateMode::BeginAccumulation
        } else {
            HierarchosTokenTapeUpdateMode::Accumulate
        };
        if budgeted_windows {
            if !group_start {
                continue;
            }
            let group_end_index = updates
                .len()
                .min(update_index + case.gradient_accumulation_steps);
            let group_updates = &updates[update_index..group_end_index];
            let labeled_inputs = group_updates
                .iter()
                .copied()
                .map(|update| HierarchosRawTokenLabeledSequenceInput {
                    tokens: case.tokens,
                    input_ids: &update.input_ids,
                    labels: &update.labels,
                    attention_mask: update.attention_mask.as_deref(),
                    loss_weights: update.loss_weights.as_deref(),
                    initial_previous_context: &update.initial_previous_context,
                    initial_target_context: &update.initial_target_context,
                    global_pos_offset: update.global_pos_offset,
                    reset_rosa_at_start: update.reset_rosa_at_start,
                    pytorch_tbptt_chunk_size: update.pytorch_tbptt_chunk_size,
                })
                .collect::<Vec<_>>();
            let h_initial_packed_states = group_updates
                .iter()
                .copied()
                .map(|update| update.h_initial_packed_state.as_slice())
                .collect::<Vec<_>>();
            let l_initial_packed_states = group_updates
                .iter()
                .copied()
                .map(|update| update.l_initial_packed_state.as_slice())
                .collect::<Vec<_>>();
            let result = if let (Some(requested_microbatch), Some(requested_stride)) =
                (sequence_microbatch_size, state_checkpoint_stride)
            {
                // Match the production joint-runtime scheduler exactly: a
                // globally learned geometry may be replayed on a smaller local
                // shard, but replay can only reduce tape residency/checkpoint
                // spacing. The exact-plan executor revalidates the resulting
                // geometry against the live Vulkan memory budget before use.
                let effective_microbatch = requested_microbatch.min(labeled_inputs.len()).max(1);
                let effective_stride = requested_stride.min(case.tokens).max(1);
                if let Some(loss_scaling) = dynamic_loss_scaling.as_mut() {
                    graph
                        .train_raw_token_labeled_sequences_with_plan_and_dynamic_loss_scaling_and_readback_policy(
                            case.batch,
                            &h_initial_packed_states,
                            &l_initial_packed_states,
                            &labeled_inputs,
                            objective,
                            hyper,
                            HierarchosTokenTapeUpdateMode::BeginAccumulation,
                            effective_microbatch,
                            effective_stride,
                            HierarchosTapeMemoryPolicy::default(),
                            HierarchosTokenTapeReadbackPolicy::Full,
                            loss_scaling,
                        )
                } else {
                    graph
                        .train_raw_token_labeled_sequences_with_plan_and_update_mode_and_readback_policy(
                            case.batch,
                            &h_initial_packed_states,
                            &l_initial_packed_states,
                            &labeled_inputs,
                            objective,
                            hyper,
                            HierarchosTokenTapeUpdateMode::BeginAccumulation,
                            effective_microbatch,
                            effective_stride,
                            HierarchosTapeMemoryPolicy::default(),
                            HierarchosTokenTapeReadbackPolicy::Full,
                        )
                }
            } else {
                if let Some(loss_scaling) = dynamic_loss_scaling.as_mut() {
                    graph.train_raw_token_labeled_sequences_budgeted_with_dynamic_loss_scaling_and_readback_policy(
                        case.batch,
                        &h_initial_packed_states,
                        &l_initial_packed_states,
                        &labeled_inputs,
                        objective,
                        hyper,
                        HierarchosTokenTapeUpdateMode::BeginAccumulation,
                        HierarchosTapeMemoryPolicy::default(),
                        HierarchosTokenTapeReadbackPolicy::Full,
                        loss_scaling,
                    )
                } else {
                    graph.train_raw_token_labeled_sequences_budgeted_with_update_mode_and_readback_policy(
                        case.batch,
                        &h_initial_packed_states,
                        &l_initial_packed_states,
                        &labeled_inputs,
                        objective,
                        hyper,
                        HierarchosTokenTapeUpdateMode::BeginAccumulation,
                        HierarchosTapeMemoryPolicy::default(),
                        HierarchosTokenTapeReadbackPolicy::Full,
                    )
                }
            }
            .with_context(|| {
                format!(
                    "running budgeted labeled parity window {update_index}..{group_end_index}"
                )
            })?;
            queue_submissions = queue_submissions
                .checked_add(result.queue_submissions)
                .context("budgeted labeled parity queue-submission count overflow")?;
            budgeted_plans.push(BudgetedPlanOutput::from_plan(
                update_index,
                group_end_index,
                &result.plan,
            ));
            for sequence in result.sequences {
                losses.extend(sequence.losses);
                final_h_packed_states.push(sequence.final_h_packed_state);
                final_l_packed_states.push(sequence.final_l_packed_state);
            }
            if let Some(loss_scaling) = dynamic_loss_scaling.as_mut() {
                let finish = graph
                    .finish_full_model_accumulation_with_dynamic_loss_scaling_and_gradient_clipping(
                        hyper,
                        loss_scaling,
                        case.grad_clip,
                    )
                    .with_context(|| {
                        format!(
                            "finishing clipped dynamic budgeted labeled parity window {update_index}..{group_end_index}"
                        )
                    })?;
                anyhow::ensure!(
                    finish.decision.should_step == !finish.decision.overflowed
                        && finish.stepped == finish.decision.should_step,
                    "budgeted dynamic labeled parity window {update_index}..{group_end_index} returned an inconsistent loss-scale decision: overflowed={} should_step={} stepped={}",
                    finish.decision.overflowed,
                    finish.decision.should_step,
                    finish.stepped,
                );
                dynamic_loss_scale_window_overflowed.push(finish.decision.overflowed);
                dynamic_loss_scale_window_stepped.push(finish.stepped);
                dynamic_loss_scale_window_scale_before.push(finish.decision.scale_before);
                dynamic_loss_scale_window_scale_after.push(finish.decision.scale_after);
                if finish.decision.overflowed {
                    dynamic_loss_scale_overflow_count = dynamic_loss_scale_overflow_count
                        .checked_add(1)
                        .context("dynamic loss-scale overflow counter overflow")?;
                } else {
                    optimizer_window_gradient_norms.push(finish.total_norm);
                    optimizer_window_clip_coefficients.push(finish.clip_coefficient);
                }
                queue_submissions = queue_submissions
                    .checked_add(finish.queue_submissions)
                    .context("budgeted dynamic clipped-finish queue-submission count overflow")?;
                optimizer_step = finish.full_model_optimizer.step;
            } else {
                let finish = graph
                    .finish_full_model_accumulation_with_gradient_clipping(hyper, case.grad_clip)
                    .with_context(|| {
                        format!(
                            "finishing clipped budgeted labeled parity window {update_index}..{group_end_index}"
                        )
                    })?;
                anyhow::ensure!(
                    finish.stepped,
                    "budgeted labeled parity window {update_index}..{group_end_index} was rejected by the global gradient safety boundary; total_norm={}",
                    finish.total_norm
                );
                queue_submissions = queue_submissions
                    .checked_add(finish.queue_submissions)
                    .context("budgeted clipped-finish queue-submission count overflow")?;
                optimizer_step = finish.full_model_optimizer.step;
                optimizer_window_gradient_norms.push(finish.total_norm);
                optimizer_window_clip_coefficients.push(finish.clip_coefficient);
            }
            if let Some(started) = optimizer_window_started.take() {
                optimizer_window_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            }
            continue;
        }
        let mut tape = graph
            .create_token_tape(
                case.batch,
                case.tokens,
                &update.h_initial_packed_state,
                &update.l_initial_packed_state,
            )
            .with_context(|| format!("creating labeled parity tape for update {update_index}"))?;
        let labeled_input = HierarchosRawTokenLabeledSequenceInput {
            tokens: case.tokens,
            input_ids: &update.input_ids,
            labels: &update.labels,
            attention_mask: update.attention_mask.as_deref(),
            loss_weights: update.loss_weights.as_deref(),
            initial_previous_context: &update.initial_previous_context,
            initial_target_context: &update.initial_target_context,
            global_pos_offset: update.global_pos_offset,
            reset_rosa_at_start: update.reset_rosa_at_start,
            pytorch_tbptt_chunk_size: update.pytorch_tbptt_chunk_size,
        };
        let result = if let Some(loss_scaling) = dynamic_loss_scaling.as_mut() {
            graph.train_raw_token_labeled_sequence_with_dynamic_loss_scaling(
                &mut tape,
                &labeled_input,
                objective,
                hyper,
                accumulation_mode,
                loss_scaling,
            )
        } else {
            graph.train_raw_token_labeled_sequence_with_update_mode(
                &mut tape,
                &labeled_input,
                objective,
                hyper,
                accumulation_mode,
            )
        }
        .with_context(|| format!("running labeled parity update {update_index}"))?;
        if trace_low_rank_g2 {
            let trace = graph
                .take_low_rank_g2_gradient_trace()
                .with_context(|| format!("reading g2 per-use trace for update {update_index}"))?;
            low_rank_g2_per_use.push(G2TraceOutput {
                update_index,
                h_uses: trace.h_uses,
                l_uses: trace.l_uses,
            });
        }
        queue_submissions = queue_submissions
            .checked_add(result.queue_submissions)
            .context("labeled parity queue-submission count overflow")?;
        optimizer_step = result.full_model_optimizer.step;
        losses.extend(result.losses);
        final_h_packed_states.push(result.final_h_packed_state);
        final_l_packed_states.push(result.final_l_packed_state);
        if case.capture_pending_gradients {
            anyhow::ensure!(
                dynamic_loss_scaling.is_some(),
                "capture_pending_gradients currently requires dynamic_loss_scale so the optimizer window remains open until finish"
            );
            pending_gradients_before_step.push(
                graph
                    .full_model_pending_gradient_snapshots()
                    .with_context(|| {
                        format!("capturing pending gradients before update {update_index} finish")
                    })?
                    .into_iter()
                    .map(|snapshot| NamedValues {
                        name: snapshot.name,
                        values: snapshot.values,
                    })
                    .collect(),
            );
        }
        if group_end {
            if let Some(loss_scaling) = dynamic_loss_scaling.as_mut() {
                let finish = graph
                    .finish_full_model_accumulation_with_dynamic_loss_scaling_and_gradient_clipping(
                        hyper,
                        loss_scaling,
                        case.grad_clip,
                    )
                    .with_context(|| {
                        format!(
                            "finishing clipped dynamic loss-scale window at update {update_index}"
                        )
                    })?;
                anyhow::ensure!(
                    finish.decision.should_step == !finish.decision.overflowed
                        && finish.stepped == finish.decision.should_step,
                    "dynamic labeled parity window at update {update_index} returned an inconsistent loss-scale decision: overflowed={} should_step={} stepped={}",
                    finish.decision.overflowed,
                    finish.decision.should_step,
                    finish.stepped,
                );
                dynamic_loss_scale_window_overflowed.push(finish.decision.overflowed);
                dynamic_loss_scale_window_stepped.push(finish.stepped);
                dynamic_loss_scale_window_scale_before.push(finish.decision.scale_before);
                dynamic_loss_scale_window_scale_after.push(finish.decision.scale_after);
                if finish.decision.overflowed {
                    dynamic_loss_scale_overflow_count = dynamic_loss_scale_overflow_count
                        .checked_add(1)
                        .context("dynamic loss-scale overflow counter overflow")?;
                } else {
                    optimizer_window_gradient_norms.push(finish.total_norm);
                    optimizer_window_clip_coefficients.push(finish.clip_coefficient);
                }
                queue_submissions = queue_submissions
                    .checked_add(finish.queue_submissions)
                    .context("dynamic loss-scale queue-submission count overflow")?;
                optimizer_step = finish.full_model_optimizer.step;
            } else {
                let finish = graph
                    .finish_full_model_accumulation_with_gradient_clipping(hyper, case.grad_clip)
                    .with_context(|| {
                        format!("finishing clipped optimizer window at update {update_index}")
                    })?;
                anyhow::ensure!(
                    finish.stepped,
                    "labeled parity fixture was rejected by the global gradient safety boundary; total_norm={}",
                    finish.total_norm
                );
                queue_submissions = queue_submissions
                    .checked_add(finish.queue_submissions)
                    .context("clipped optimizer-finish queue-submission count overflow")?;
                optimizer_step = finish.full_model_optimizer.step;
                optimizer_window_gradient_norms.push(finish.total_norm);
                optimizer_window_clip_coefficients.push(finish.clip_coefficient);
            }
            if let Some(started) = optimizer_window_started.take() {
                optimizer_window_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            }
        }
    }

    anyhow::ensure!(
        dynamic_loss_scale_overflow_count == case.expected_dynamic_loss_scale_overflows,
        "labeled parity observed {dynamic_loss_scale_overflow_count} dynamic loss-scale overflows, expected {}",
        case.expected_dynamic_loss_scale_overflows,
    );

    let training_elapsed_ms = training_started.elapsed().as_secs_f64() * 1_000.0;
    let graph_summary = graph.summary();
    let precision_policy = graph.training_precision_policy();
    let (compute_dtype, autocast_enabled) = match precision_policy {
        HierarchosTrainingPrecisionPolicy::Fp16StorageFp16LmBackward => ("float16", true),
        _ => ("float32", false),
    };
    let session_loss_scaling = dynamic_loss_scaling
        .clone()
        .unwrap_or(HierarchosLossScalingState {
            mode: "none".to_string(),
            scale: None,
            growth_factor: None,
            backoff_factor: None,
            growth_interval: None,
            growth_tracker: None,
            pending_gradients_scaled: false,
        });
    let replay_tensors = final_h_packed_states
        .last()
        .zip(final_l_packed_states.last())
        .map(|(h, l)| {
            vec![
                HierarchosPortableReplayTensor::f32("state_000000", vec![h.len()], h.clone()),
                HierarchosPortableReplayTensor::f32("state_000001", vec![l.len()], l.clone()),
            ]
        })
        .unwrap_or_default();
    let replay_state = if replay_tensors.is_empty() {
        serde_json::json!({"__kind__": "dict", "items": []})
    } else {
        serde_json::json!({
            "__kind__": "dict",
            "items": [[
                "token_tape_replay",
                {
                    "__kind__": "dict",
                    "items": [
                        ["final_h_packed_state", {"__kind__": "tensor", "name": "state_000000"}],
                        ["final_l_packed_state", {"__kind__": "tensor", "name": "state_000001"}],
                        ["tokens", case.tokens],
                        ["batch", case.batch]
                    ]
                }
            ]]
        })
    };
    let replay = HierarchosPortableTrainingReplay::new_with_training_session(
        0,
        0,
        HierarchosTrainingSessionState {
            format: HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT.to_string(),
            completed_epoch: 0,
            mid_epoch_step: 0,
            optimizer_grouping_version: 2,
            main_lr_scheduler: None,
            ltm_lr_scheduler: None,
            effective_training_config: serde_json::json!({
                "gradient_accumulation_steps": case.gradient_accumulation_steps,
                "grad_clip": case.grad_clip,
            }),
            skipped_train_batches: 0,
            data_stream_cursor: None,
            execution_policy: Some(HierarchosExecutionPolicyState {
                format: HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT.to_string(),
                source_backend: "vulkan".to_string(),
                compute_dtype: compute_dtype.to_string(),
                autocast_enabled,
                stochastic_rng: HierarchosStochasticRngPolicyState {
                    mode: "none".to_string(),
                    state_required: false,
                    canonical_counter: None,
                },
                loss_scaling: session_loss_scaling,
            }),
        },
        replay_state,
        replay_tensors,
    )?;
    // Checkpoint serialization is an explicit host-observation boundary. The
    // training loop leaves LTM readiness Vulkan-authoritative, so materialize
    // the tiny controller mirror only here, immediately before export.
    graph.synchronize_ltm_alignment_controller_metadata()?;
    let manifest = graph.export_training_checkpoint_package_with_replay(
        &model_dir,
        &output_package,
        &replay,
    )?;
    let parameter_count = graph.full_model_parameter_snapshots()?.len();
    let memory = graph.memory_stats()?;
    let memory_budget = graph.memory_budget()?;
    let dynamic_loss_scale_after = dynamic_loss_scaling.as_ref().and_then(|state| state.scale);
    let dynamic_loss_scale_growth_tracker = dynamic_loss_scaling
        .as_ref()
        .and_then(|state| state.growth_tracker);
    let output = Output {
        device_index: selected_device_index,
        device_name: selected_device_name,
        batch: case.batch,
        tokens: case.tokens,
        updates: updates.len(),
        gradient_accumulation_steps: case.gradient_accumulation_steps,
        requested_sequence_microbatch_size: sequence_microbatch_size,
        requested_state_checkpoint_stride: state_checkpoint_stride,
        dynamic_loss_scale: case.dynamic_loss_scale,
        dynamic_loss_scale_after,
        dynamic_loss_scale_growth_tracker,
        dynamic_loss_scale_overflow_count,
        dynamic_loss_scale_window_overflowed,
        dynamic_loss_scale_window_stepped,
        dynamic_loss_scale_window_scale_before,
        dynamic_loss_scale_window_scale_after,
        grad_clip: case.grad_clip,
        training_precision_policy: precision_policy.label().to_string(),
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
        queue_submissions,
        memory_live_buffer_count: memory.live_buffer_count,
        memory_live_buffer_bytes: memory.live_buffer_bytes,
        memory_driver_allocation_count: memory.driver_allocation_count,
        memory_reserved_bytes: memory.reserved_bytes,
        memory_max_driver_allocation_count: memory.max_driver_allocation_count,
        memory_budget_extension_supported: memory_budget.budget_extension_supported,
        device_local_heap_size_bytes: memory_budget.device_local_heap_size_bytes,
        device_local_budget_bytes: memory_budget.device_local_budget_bytes,
        device_local_usage_bytes: memory_budget.device_local_usage_bytes,
        device_local_available_bytes: memory_budget.device_local_available_bytes,
        optimizer_step,
        training_elapsed_ms,
        optimizer_window_ms,
        optimizer_window_gradient_norms,
        optimizer_window_clip_coefficients,
        accumulation_open: manifest.accumulation_open,
        pending_gradient_tensor_count: manifest.gradient_tensor_count,
        pending_gradients_before_step,
        low_rank_g2_per_use,
        budgeted_plans,
        losses_finite: losses.iter().all(|value| value.is_finite()),
        losses,
        final_h_packed_states,
        final_l_packed_states,
        parameter_count,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
