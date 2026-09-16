use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, AdamWOptimizerState, HierarchosFullModelUpdateMode,
    HierarchosLossScalingState, HierarchosSequenceGradientNormalization,
    HierarchosTapeMemoryPolicy, HierarchosTokenTapeControlSnapshot,
    HierarchosTokenTapeMicrobatchInput, HierarchosTokenTapeStepInput, HierarchosTrainingGraph,
    HierarchosWorkerRefinementLossInput, RwkvParameterSnapshot, VulkanDevice,
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
struct Case {
    batch: usize,
    h_steps: usize,
    shadow_steps: usize,
    enc: Vec<f32>,
    previous_context: Vec<f32>,
    target_context: Vec<f32>,
    context_alpha: f32,
    h_token_ids: Vec<u32>,
    l_token_ids: Vec<u32>,
    h_initial_packed_state: Vec<f32>,
    l_initial_packed_state: Vec<f32>,
    l_final_packed_state_grad: Option<Vec<f32>>,
    h_to_context_grad: Vec<f32>,
    h_depth_grad: Vec<f32>,
    h_selected_packed_state_grad: Option<Vec<f32>>,
    final_drift_grad: Vec<f32>,
    commitment_cost_grad: Vec<f32>,
    targets: Vec<u32>,
    optimizer: OptimizerCase,
    #[serde(default = "default_accumulation_repeats")]
    accumulation_repeats: usize,
}

fn default_accumulation_repeats() -> usize {
    1
}

#[derive(Serialize)]
struct ParameterOutput {
    name: String,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    training_precision_policy: String,
    h_low_rank_fp16_parameter_storage_active: bool,
    l_low_rank_fp16_parameter_storage_active: bool,
    h_low_rank_native_fp16_backward_compute_active: bool,
    l_low_rank_native_fp16_backward_compute_active: bool,
    h_low_rank_native_fp16_parameter_grad_compute_active: bool,
    l_low_rank_native_fp16_parameter_grad_compute_active: bool,
    h_low_rank_parameter_grad_arithmetic: String,
    l_low_rank_parameter_grad_arithmetic: String,
    projection_fp16_parameter_storage_active: bool,
    lm_head_fp16_parameter_storage_active: bool,
    lm_head_execution_arm: String,
    lm_head_weight_grad_topology: Option<String>,
    lm_head_fused_adjoint_topology: Option<String>,
    lm_head_native_fp16_backward_compute_active: bool,
    out_norm_native_fp16_backward_compute_active: bool,
    projection_native_fp16_backward_compute_active: bool,
    activation_clamp: f32,
    queue_submissions: u32,
    microbatches: usize,
    loss: f32,
    h_optimizer_step: u32,
    l_optimizer_step: u32,
    projection_optimizer_step: u32,
    lm_optimizer_step: u32,
    full_model_optimizer_step: u32,
    full_model_optimizer_tensor_count: usize,
    full_model_optimizer_names: Vec<String>,
    full_model_optimizer_checkpoint_roundtrip: bool,
    h_outputs: Vec<f32>,
    h_final_packed_state: Vec<f32>,
    h_grad_initial_packed_state: Vec<f32>,
    l_outputs: Vec<f32>,
    l_final_packed_state: Vec<f32>,
    l_grad_initial_packed_state: Vec<f32>,
    final_drift: Vec<f32>,
    commitment_cost: Vec<f32>,
    effective_l_steps: Vec<f32>,
    grad_enc: Vec<f32>,
    grad_previous_context: Vec<f32>,
    grad_target_context: Vec<f32>,
    manager_halt_probabilities: Vec<f32>,
    manager_selected_index: Vec<u32>,
    manager_executed_steps: Vec<f32>,
    manager_selected_output: Vec<f32>,
    manager_selected_packed_state: Vec<f32>,
    sequence_state_h_packed_state: Vec<f32>,
    sequence_state_l_packed_state: Vec<f32>,
    sequence_state_h_packed_state_adjoint: Vec<f32>,
    sequence_state_l_packed_state_adjoint: Vec<f32>,
    h_parameters: Vec<ParameterOutput>,
    l_parameters: Vec<ParameterOutput>,
    projection_parameters: Vec<ParameterOutput>,
    lm_head_weight: Vec<f32>,
    lm_head_fp16_execution_weight: Option<Vec<f32>>,
    out_norm_weight: Vec<f32>,
    out_norm_bias: Vec<f32>,
    token_tape_tokens: usize,
    token_tape_queue_submissions: u32,
    token_tape_optimizer_step: u32,
    token_tape_max_state_diff: f32,
    token_tape_max_adjoint_diff: f32,
    token_tape_max_loss_diff: f32,
    token_tape_max_optimizer_diff: f32,
    token_tape_mean_normalization_max_moment_diff: f32,
    token_tape_control_match: bool,
    token_tape_parity: bool,
    token_tape_microbatch_sequences: usize,
    token_tape_microbatch_total_tokens: usize,
    token_tape_microbatch_queue_submissions: u32,
    token_tape_microbatch_descriptor_pool_count: usize,
    token_tape_microbatch_descriptor_set_count: usize,
    token_tape_microbatch_dispatch_count: usize,
    token_tape_microbatch_shader_barrier_count: usize,
    token_tape_microbatch_pipeline_bind_count: usize,
    token_tape_microbatch_descriptor_bind_count: usize,
    token_tape_microbatch_push_constant_write_count: usize,
    token_tape_microbatch_upload_count: usize,
    token_tape_microbatch_uploaded_bytes: usize,
    token_tape_microbatch_upload_arena_buffer_count: usize,
    token_tape_microbatch_optimizer_step: u32,
    token_tape_microbatch_max_state_diff: f32,
    token_tape_microbatch_max_adjoint_diff: f32,
    token_tape_microbatch_max_loss_diff: f32,
    token_tape_microbatch_max_optimizer_diff: f32,
    token_tape_microbatch_mean_normalization_max_moment_diff: f32,
    token_tape_microbatch_control_match: bool,
    token_tape_microbatch_parity: bool,
    token_tape_sparse_replay_tokens: usize,
    token_tape_sparse_replay_checkpoint_stride: usize,
    token_tape_sparse_replay_queue_submissions: u32,
    token_tape_sparse_replay_max_state_diff: f32,
    token_tape_sparse_replay_max_adjoint_diff: f32,
    token_tape_sparse_replay_max_loss_diff: f32,
    token_tape_sparse_replay_max_optimizer_diff: f32,
    token_tape_sparse_replay_control_match: bool,
    token_tape_sparse_replay_parity: bool,
    dynamic_loss_scale_optimizer_step: u32,
    dynamic_loss_scale_queue_submissions: u32,
    dynamic_loss_scale_scale_after: f64,
    dynamic_loss_scale_growth_tracker: u64,
    dynamic_loss_scale_max_parameter_diff: f32,
    dynamic_loss_scale_max_moment_diff: f32,
    dynamic_loss_scale_parity: bool,
}

struct TokenTapeCheck {
    tokens: usize,
    queue_submissions: u32,
    optimizer_step: u32,
    max_state_diff: f32,
    max_adjoint_diff: f32,
    max_loss_diff: f32,
    max_optimizer_diff: f32,
    mean_normalization_max_moment_diff: f32,
    control_match: bool,
}

struct TokenTapeMicrobatchCheck {
    sequences: usize,
    total_tokens: usize,
    queue_submissions: u32,
    descriptor_pool_count: usize,
    descriptor_set_count: usize,
    dispatch_count: usize,
    shader_barrier_count: usize,
    pipeline_bind_count: usize,
    descriptor_bind_count: usize,
    push_constant_write_count: usize,
    upload_count: usize,
    uploaded_bytes: usize,
    upload_arena_buffer_count: usize,
    optimizer_step: u32,
    max_state_diff: f32,
    max_adjoint_diff: f32,
    max_loss_diff: f32,
    max_optimizer_diff: f32,
    mean_normalization_max_moment_diff: f32,
    control_match: bool,
    sparse_replay_tokens: usize,
    sparse_replay_checkpoint_stride: usize,
    sparse_replay_queue_submissions: u32,
    sparse_replay_max_state_diff: f32,
    sparse_replay_max_adjoint_diff: f32,
    sparse_replay_max_loss_diff: f32,
    sparse_replay_max_optimizer_diff: f32,
    sparse_replay_control_match: bool,
}

struct DynamicLossScaleCheck {
    optimizer_step: u32,
    queue_submissions: u32,
    scale_after: f64,
    growth_tracker: u64,
    max_parameter_diff: f32,
    max_moment_diff: f32,
}

fn main() -> Result<()> {
    // This long parity runner intentionally holds several full training graphs
    // and host reference trajectories alive together. Give the Windows harness
    // more stack headroom without changing production graph allocation.
    std::thread::Builder::new()
        .name("hierarchos-worker-refinement-parity".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("worker-refinement parity thread panicked"))?
}

fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context("worker refinement parity runner requires --case")?;
    let model_dir = model_dir.context("worker refinement parity runner requires --model-dir")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    if case.shadow_steps == 0 {
        anyhow::bail!("worker refinement parity case requires positive shadow_steps");
    }
    if case.accumulation_repeats == 0 {
        anyhow::bail!("worker refinement parity case requires positive accumulation_repeats");
    }

    let device = VulkanDevice::new()?;
    let mut graph = HierarchosTrainingGraph::from_model_package(
        device,
        &model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let hyper = AdamWHyperParams {
        lr: case.optimizer.lr,
        beta1: case.optimizer.beta1,
        beta2: case.optimizer.beta2,
        eps: case.optimizer.eps,
        weight_decay: case.optimizer.weight_decay,
    };
    let token_tape_check = verify_token_tape_boundary(&case, &model_dir, hyper)?;
    let token_tape_microbatch_check =
        verify_token_tape_microbatch_boundary(&case, &model_dir, hyper)?;
    let sequence_state_arena = graph.create_sequence_state_arena(
        case.batch,
        &case.h_initial_packed_state,
        &case.l_initial_packed_state,
    )?;
    let mut result = None;
    let mut queue_submissions = 0u32;
    for microbatch in 0..case.accumulation_repeats {
        let update_mode = if case.accumulation_repeats == 1 {
            HierarchosFullModelUpdateMode::Step
        } else if microbatch == 0 {
            HierarchosFullModelUpdateMode::BeginAccumulation
        } else if microbatch + 1 == case.accumulation_repeats {
            HierarchosFullModelUpdateMode::FinishAccumulation
        } else {
            HierarchosFullModelUpdateMode::Accumulate
        };
        let current = graph.train_worker_refinement_loss_one_submit_with_update_mode(
            HierarchosWorkerRefinementLossInput {
                batch: case.batch,
                h_steps: case.h_steps,
                shadow_steps: case.shadow_steps,
                enc: &case.enc,
                previous_context: &case.previous_context,
                target_context: &case.target_context,
                context_alpha: case.context_alpha,
                h_token_ids: &case.h_token_ids,
                l_token_ids: &case.l_token_ids,
                h_initial_packed_state: &case.h_initial_packed_state,
                l_initial_packed_state: &case.l_initial_packed_state,
                l_final_packed_state_grad: case.l_final_packed_state_grad.as_deref(),
                h_to_context_grad: &case.h_to_context_grad,
                h_depth_grad: &case.h_depth_grad,
                h_selected_packed_state_grad: case.h_selected_packed_state_grad.as_deref(),
                final_drift_grad: &case.final_drift_grad,
                commitment_cost_grad: &case.commitment_cost_grad,
                targets: &case.targets,
                supervision_weights: None,
            },
            hyper,
            update_mode,
        )?;
        queue_submissions = queue_submissions
            .checked_add(current.queue_submissions)
            .context("worker refinement queue-submission count overflow")?;
        result = Some(current);
    }
    let result = result.context("worker refinement accumulation produced no microbatches")?;
    graph.capture_last_worker_step_into_sequence_state(&sequence_state_arena)?;
    let sequence_state = graph.sequence_state_snapshot(&sequence_state_arena)?;
    let (out_norm_weight, out_norm_bias) = graph.out_norm_parameters()?;
    let lm_head_weight = graph.shared_lm_head().weights()?;
    let lm_head_fp16_execution_weight = graph.shared_lm_head().fp16_parameter_storage_values()?;
    let h_outputs = result.h.sequence.outputs.clone();
    let h_final_packed_state = result.h.sequence.final_packed_state.clone();
    let h_grad_initial_packed_state = result.h.sequence.grad_initial_packed_state.clone();
    let l_outputs = result.l.sequence.outputs.clone();
    let l_final_packed_state = result.l.sequence.final_packed_state.clone();
    let l_grad_initial_packed_state = result.l.sequence.grad_initial_packed_state.clone();
    let full_model_optimizer_state = graph.full_model_optimizer_state()?;
    let full_model_parameters = graph.full_model_parameter_snapshots()?;
    let dynamic_loss_scale_check = verify_dynamic_loss_scale_finish(
        &case,
        &model_dir,
        hyper,
        &full_model_parameters,
        &full_model_optimizer_state,
    )?;
    let optimizer_checkpoint = model_dir.join("worker_refinement_full_model_optimizer.safetensors");
    graph.save_full_model_optimizer_checkpoint(&optimizer_checkpoint)?;
    graph.load_full_model_optimizer_checkpoint(&optimizer_checkpoint)?;
    let reloaded_full_model_optimizer_state = graph.full_model_optimizer_state()?;
    let full_model_optimizer_checkpoint_roundtrip = full_model_optimizer_state.step
        == reloaded_full_model_optimizer_state.step
        && full_model_optimizer_state.slots.len()
            == reloaded_full_model_optimizer_state.slots.len()
        && full_model_optimizer_state
            .slots
            .iter()
            .zip(&reloaded_full_model_optimizer_state.slots)
            .all(|(before, after)| {
                before.name == after.name
                    && before.exp_avg == after.exp_avg
                    && before.exp_avg_sq == after.exp_avg_sq
            });
    let graph_summary = graph.summary();

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: graph_summary.device,
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
            projection_fp16_parameter_storage_active: graph_summary
                .projection_fp16_parameter_storage_active,
            lm_head_fp16_parameter_storage_active: graph_summary
                .lm_head_fp16_parameter_storage_active,
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
            projection_native_fp16_backward_compute_active: graph_summary
                .projection_native_fp16_backward_compute_active,
            activation_clamp: graph.config().activation_clamp,
            queue_submissions,
            microbatches: case.accumulation_repeats,
            loss: result.output.loss,
            h_optimizer_step: result.h.optimizer.step,
            l_optimizer_step: result.l.optimizer.step,
            projection_optimizer_step: result.projection_optimizer.step,
            lm_optimizer_step: result.output.step,
            full_model_optimizer_step: result.full_model_optimizer.step,
            full_model_optimizer_tensor_count: result.full_model_optimizer.tensor_count,
            full_model_optimizer_names: full_model_optimizer_state
                .slots
                .into_iter()
                .map(|slot| slot.name)
                .collect(),
            full_model_optimizer_checkpoint_roundtrip,
            h_outputs,
            h_final_packed_state,
            h_grad_initial_packed_state,
            l_outputs,
            l_final_packed_state,
            l_grad_initial_packed_state,
            final_drift: result.final_drift,
            commitment_cost: result.commitment_cost,
            effective_l_steps: result.effective_l_steps,
            grad_enc: result.grad_enc,
            grad_previous_context: result.grad_previous_context,
            grad_target_context: result.grad_target_context,
            manager_halt_probabilities: result.manager_halt_probabilities,
            manager_selected_index: result.manager_selected_index,
            manager_executed_steps: result.manager_executed_steps,
            manager_selected_output: result.manager_selected_output,
            manager_selected_packed_state: result.manager_selected_packed_state,
            sequence_state_h_packed_state: sequence_state.h_packed_state,
            sequence_state_l_packed_state: sequence_state.l_packed_state,
            sequence_state_h_packed_state_adjoint: sequence_state.h_packed_state_adjoint,
            sequence_state_l_packed_state_adjoint: sequence_state.l_packed_state_adjoint,
            h_parameters: snapshots(result.h.parameters),
            l_parameters: snapshots(result.l.parameters),
            projection_parameters: snapshots(result.projection_parameters),
            lm_head_weight,
            lm_head_fp16_execution_weight,
            out_norm_weight,
            out_norm_bias,
            token_tape_tokens: token_tape_check.tokens,
            token_tape_queue_submissions: token_tape_check.queue_submissions,
            token_tape_optimizer_step: token_tape_check.optimizer_step,
            token_tape_max_state_diff: token_tape_check.max_state_diff,
            token_tape_max_adjoint_diff: token_tape_check.max_adjoint_diff,
            token_tape_max_loss_diff: token_tape_check.max_loss_diff,
            token_tape_max_optimizer_diff: token_tape_check.max_optimizer_diff,
            token_tape_mean_normalization_max_moment_diff: token_tape_check
                .mean_normalization_max_moment_diff,
            token_tape_control_match: token_tape_check.control_match,
            token_tape_parity: token_tape_check.queue_submissions == 1
                && token_tape_check.control_match
                && token_tape_check.max_state_diff <= 5.0e-6
                && token_tape_check.max_adjoint_diff <= 5.0e-6
                && token_tape_check.max_loss_diff <= 5.0e-6
                && token_tape_check.max_optimizer_diff <= 5.0e-6,
            token_tape_microbatch_sequences: token_tape_microbatch_check.sequences,
            token_tape_microbatch_total_tokens: token_tape_microbatch_check.total_tokens,
            token_tape_microbatch_queue_submissions: token_tape_microbatch_check.queue_submissions,
            token_tape_microbatch_descriptor_pool_count: token_tape_microbatch_check
                .descriptor_pool_count,
            token_tape_microbatch_descriptor_set_count: token_tape_microbatch_check
                .descriptor_set_count,
            token_tape_microbatch_dispatch_count: token_tape_microbatch_check.dispatch_count,
            token_tape_microbatch_shader_barrier_count: token_tape_microbatch_check
                .shader_barrier_count,
            token_tape_microbatch_pipeline_bind_count: token_tape_microbatch_check
                .pipeline_bind_count,
            token_tape_microbatch_descriptor_bind_count: token_tape_microbatch_check
                .descriptor_bind_count,
            token_tape_microbatch_push_constant_write_count: token_tape_microbatch_check
                .push_constant_write_count,
            token_tape_microbatch_upload_count: token_tape_microbatch_check.upload_count,
            token_tape_microbatch_uploaded_bytes: token_tape_microbatch_check.uploaded_bytes,
            token_tape_microbatch_upload_arena_buffer_count: token_tape_microbatch_check
                .upload_arena_buffer_count,
            token_tape_microbatch_optimizer_step: token_tape_microbatch_check.optimizer_step,
            token_tape_microbatch_max_state_diff: token_tape_microbatch_check.max_state_diff,
            token_tape_microbatch_max_adjoint_diff: token_tape_microbatch_check.max_adjoint_diff,
            token_tape_microbatch_max_loss_diff: token_tape_microbatch_check.max_loss_diff,
            token_tape_microbatch_max_optimizer_diff: token_tape_microbatch_check
                .max_optimizer_diff,
            token_tape_microbatch_mean_normalization_max_moment_diff: token_tape_microbatch_check
                .mean_normalization_max_moment_diff,
            token_tape_microbatch_control_match: token_tape_microbatch_check.control_match,
            token_tape_microbatch_parity: token_tape_microbatch_check.queue_submissions == 1
                && token_tape_microbatch_check.control_match
                && token_tape_microbatch_check.max_state_diff <= 5.0e-6
                && token_tape_microbatch_check.max_adjoint_diff <= 5.0e-6
                && token_tape_microbatch_check.max_loss_diff <= 5.0e-6
                && token_tape_microbatch_check.max_optimizer_diff <= 5.0e-6,
            token_tape_sparse_replay_tokens: token_tape_microbatch_check.sparse_replay_tokens,
            token_tape_sparse_replay_checkpoint_stride: token_tape_microbatch_check
                .sparse_replay_checkpoint_stride,
            token_tape_sparse_replay_queue_submissions: token_tape_microbatch_check
                .sparse_replay_queue_submissions,
            token_tape_sparse_replay_max_state_diff: token_tape_microbatch_check
                .sparse_replay_max_state_diff,
            token_tape_sparse_replay_max_adjoint_diff: token_tape_microbatch_check
                .sparse_replay_max_adjoint_diff,
            token_tape_sparse_replay_max_loss_diff: token_tape_microbatch_check
                .sparse_replay_max_loss_diff,
            token_tape_sparse_replay_max_optimizer_diff: token_tape_microbatch_check
                .sparse_replay_max_optimizer_diff,
            token_tape_sparse_replay_control_match: token_tape_microbatch_check
                .sparse_replay_control_match,
            token_tape_sparse_replay_parity: token_tape_microbatch_check
                .sparse_replay_queue_submissions
                == 1
                && token_tape_microbatch_check.sparse_replay_control_match
                && token_tape_microbatch_check.sparse_replay_max_state_diff <= 5.0e-6
                && token_tape_microbatch_check.sparse_replay_max_adjoint_diff <= 5.0e-6
                && token_tape_microbatch_check.sparse_replay_max_loss_diff <= 5.0e-6
                && token_tape_microbatch_check.sparse_replay_max_optimizer_diff <= 5.0e-6,
            dynamic_loss_scale_optimizer_step: dynamic_loss_scale_check.optimizer_step,
            dynamic_loss_scale_queue_submissions: dynamic_loss_scale_check.queue_submissions,
            dynamic_loss_scale_scale_after: dynamic_loss_scale_check.scale_after,
            dynamic_loss_scale_growth_tracker: dynamic_loss_scale_check.growth_tracker,
            dynamic_loss_scale_max_parameter_diff: dynamic_loss_scale_check.max_parameter_diff,
            dynamic_loss_scale_max_moment_diff: dynamic_loss_scale_check.max_moment_diff,
            dynamic_loss_scale_parity: dynamic_loss_scale_check.optimizer_step == 1
                && dynamic_loss_scale_check.queue_submissions
                    == if graph_summary
                        .training_precision_policy
                        .uses_fp16_parameter_storage()
                    {
                        // prepare + coalesced AdamW wavefront + FP16 mirror
                        // refresh + scaler/LTM readback
                        4
                    } else {
                        // prepare + coalesced AdamW wavefront + scaler/LTM
                        // readback
                        3
                    }
                && dynamic_loss_scale_check.scale_after.to_bits() == 1.0f64.to_bits()
                && dynamic_loss_scale_check.growth_tracker == 1
                && dynamic_loss_scale_check.max_parameter_diff <= 5.0e-6
                && dynamic_loss_scale_check.max_moment_diff <= 5.0e-6,
        })?
    );
    Ok(())
}

fn verify_dynamic_loss_scale_finish(
    case: &Case,
    model_dir: &PathBuf,
    hyper: AdamWHyperParams,
    baseline_parameters: &[RwkvParameterSnapshot],
    baseline_optimizer: &AdamWOptimizerState,
) -> Result<DynamicLossScaleCheck> {
    let mut graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    for microbatch in 0..case.accumulation_repeats {
        let update_mode = if microbatch == 0 {
            HierarchosFullModelUpdateMode::BeginAccumulation
        } else {
            HierarchosFullModelUpdateMode::Accumulate
        };
        graph.train_worker_refinement_loss_one_submit_with_update_mode(
            HierarchosWorkerRefinementLossInput {
                batch: case.batch,
                h_steps: case.h_steps,
                shadow_steps: case.shadow_steps,
                enc: &case.enc,
                previous_context: &case.previous_context,
                target_context: &case.target_context,
                context_alpha: case.context_alpha,
                h_token_ids: &case.h_token_ids,
                l_token_ids: &case.l_token_ids,
                h_initial_packed_state: &case.h_initial_packed_state,
                l_initial_packed_state: &case.l_initial_packed_state,
                l_final_packed_state_grad: case.l_final_packed_state_grad.as_deref(),
                h_to_context_grad: &case.h_to_context_grad,
                h_depth_grad: &case.h_depth_grad,
                h_selected_packed_state_grad: case.h_selected_packed_state_grad.as_deref(),
                final_drift_grad: &case.final_drift_grad,
                commitment_cost_grad: &case.commitment_cost_grad,
                targets: &case.targets,
                supervision_weights: None,
            },
            hyper,
            update_mode,
        )?;
    }

    // Scale 1.0 makes the deferred path directly comparable with the ordinary
    // close while still exercising the dynamic finite scan, scaler transition,
    // coalesced range-addressed AdamW, and canonical full-model registry. The
    // optimizer unit test separately proves a nontrivial 8x device unscale.
    let mut loss_scaling = HierarchosLossScalingState {
        mode: "dynamic".to_string(),
        scale: Some(1.0),
        growth_factor: Some(2.0),
        backoff_factor: Some(0.5),
        growth_interval: Some(2),
        growth_tracker: Some(0),
        pending_gradients_scaled: true,
    };
    let (finish, state_ranges) = graph
        .finish_full_model_accumulation_with_dynamic_loss_scaling_wavefront(
            hyper,
            &mut loss_scaling,
            256 * 1024,
        )?;
    if finish.decision.overflowed || !finish.decision.should_step {
        anyhow::bail!("finite dynamic loss-scale parity window unexpectedly skipped AdamW");
    }
    if state_ranges == 0 {
        anyhow::bail!("finite dynamic loss-scale wavefront traversed zero optimizer ranges");
    }
    let dynamic_parameters = graph.full_model_parameter_snapshots()?;
    let dynamic_optimizer = graph.full_model_optimizer_state()?;
    let max_parameter_diff = max_parameter_snapshot_diff(baseline_parameters, &dynamic_parameters)?;
    let max_moment_diff = max_optimizer_moment_diff(baseline_optimizer, &dynamic_optimizer)?;
    Ok(DynamicLossScaleCheck {
        optimizer_step: finish.full_model_optimizer.step,
        queue_submissions: finish.queue_submissions,
        scale_after: finish.decision.scale_after,
        growth_tracker: finish.decision.growth_tracker,
        max_parameter_diff,
        max_moment_diff,
    })
}

fn max_parameter_snapshot_diff(
    baseline: &[RwkvParameterSnapshot],
    actual: &[RwkvParameterSnapshot],
) -> Result<f32> {
    if baseline.len() != actual.len() {
        anyhow::bail!(
            "dynamic loss-scale parameter registry size mismatch: {} vs {}",
            baseline.len(),
            actual.len()
        );
    }
    let mut max_diff = 0.0f32;
    for (baseline, actual) in baseline.iter().zip(actual) {
        if baseline.name != actual.name || baseline.values.len() != actual.values.len() {
            anyhow::bail!(
                "dynamic loss-scale parameter registry mismatch: {:?} vs {:?}",
                baseline.name,
                actual.name
            );
        }
        for (&expected, &observed) in baseline.values.iter().zip(&actual.values) {
            max_diff = max_diff.max((expected - observed).abs());
        }
    }
    Ok(max_diff)
}

fn max_optimizer_moment_diff(
    baseline: &AdamWOptimizerState,
    actual: &AdamWOptimizerState,
) -> Result<f32> {
    if baseline.step != actual.step || baseline.slots.len() != actual.slots.len() {
        anyhow::bail!(
            "dynamic loss-scale optimizer summary mismatch: step/tensors={}/{} vs {}/{}",
            baseline.step,
            baseline.slots.len(),
            actual.step,
            actual.slots.len()
        );
    }
    let mut max_diff = 0.0f32;
    for (baseline, actual) in baseline.slots.iter().zip(&actual.slots) {
        if baseline.name != actual.name
            || baseline.step != actual.step
            || baseline.exp_avg.len() != actual.exp_avg.len()
            || baseline.exp_avg_sq.len() != actual.exp_avg_sq.len()
        {
            anyhow::bail!(
                "dynamic loss-scale optimizer slot mismatch: {:?} vs {:?}",
                baseline.name,
                actual.name
            );
        }
        for (&expected, &observed) in baseline.exp_avg.iter().zip(&actual.exp_avg) {
            max_diff = max_diff.max((expected - observed).abs());
        }
        for (&expected, &observed) in baseline.exp_avg_sq.iter().zip(&actual.exp_avg_sq) {
            max_diff = max_diff.max((expected - observed).abs());
        }
    }
    Ok(max_diff)
}

fn verify_token_tape_boundary(
    case: &Case,
    model_dir: &PathBuf,
    hyper: AdamWHyperParams,
) -> Result<TokenTapeCheck> {
    const TOKENS: usize = 2;

    let mut tape_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut tape = tape_graph.create_token_tape(
        case.batch,
        TOKENS,
        &case.h_initial_packed_state,
        &case.l_initial_packed_state,
    )?;
    let tape_steps = token_tape_steps(case, TOKENS);
    let tape_result = tape_graph.train_token_tape(
        &mut tape,
        &tape_steps,
        case.h_selected_packed_state_grad.as_deref(),
        case.l_final_packed_state_grad.as_deref(),
        hyper,
    )?;
    let tape_optimizer_state = tape_graph.full_model_optimizer_state()?;

    let mut mean_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut mean_tape = mean_graph.create_token_tape(
        case.batch,
        TOKENS,
        &case.h_initial_packed_state,
        &case.l_initial_packed_state,
    )?;
    let mean_result = mean_graph.train_token_tape_with_normalization(
        &mut mean_tape,
        &tape_steps,
        case.h_selected_packed_state_grad.as_deref(),
        case.l_final_packed_state_grad.as_deref(),
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
    )?;
    let mean_optimizer_state = mean_graph.full_model_optimizer_state()?;
    let mean_normalization_max_moment_diff = optimizer_state_scaled_moment_diff(
        &tape_optimizer_state,
        &mean_optimizer_state,
        1.0 / TOKENS as f32,
    )?;
    let mean_observable_diff = max_abs_diff(
        &tape_result.final_h_packed_state,
        &mean_result.final_h_packed_state,
    )?
    .max(max_abs_diff(
        &tape_result.final_l_packed_state,
        &mean_result.final_l_packed_state,
    )?)
    .max(max_abs_diff(
        &tape_result.grad_initial_h_packed_state,
        &mean_result.grad_initial_h_packed_state,
    )?)
    .max(max_abs_diff(
        &tape_result.grad_initial_l_packed_state,
        &mean_result.grad_initial_l_packed_state,
    )?)
    .max(max_abs_diff(&tape_result.losses, &mean_result.losses)?);
    if mean_observable_diff > 5.0e-6 || mean_normalization_max_moment_diff > 5.0e-6 {
        anyhow::bail!(
            "mean-by-token normalization changed tape observables or scaled AdamW moments incorrectly: observable_diff={mean_observable_diff:.9e} moment_diff={mean_normalization_max_moment_diff:.9e}"
        );
    }

    // Reference path: materialize every committed H/L state on the host, then
    // explicitly feed the state adjoints returned by token t+1 into token t.
    // This is intentionally the old seam the Vulkan token tape removes.
    let mut host_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut host_h_states = vec![case.h_initial_packed_state.clone()];
    let mut host_l_states = vec![case.l_initial_packed_state.clone()];
    let mut host_controls = Vec::with_capacity(TOKENS);
    for token_index in 0..TOKENS {
        let current = host_graph.train_worker_refinement_loss_one_submit_with_update_mode(
            worker_input(
                case,
                &host_h_states[token_index],
                &host_l_states[token_index],
                None,
                None,
            ),
            hyper,
            if token_index == 0 {
                HierarchosFullModelUpdateMode::BeginAccumulation
            } else {
                HierarchosFullModelUpdateMode::Accumulate
            },
        )?;
        host_controls.push((
            current.manager_selected_index.clone(),
            current.manager_executed_steps.clone(),
            current.effective_l_steps.clone(),
        ));
        host_h_states.push(current.manager_selected_packed_state);
        host_l_states.push(current.l.sequence.final_packed_state);
    }

    // State materialization deliberately leaves that scratch accumulation
    // window unfinished so parameters remain frozen. Start reverse parity on
    // a pristine graph instead of opening a second window on the scratch one.
    host_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;

    let mut future_h = case.h_selected_packed_state_grad.clone();
    let mut future_l = case.l_final_packed_state_grad.clone();
    let mut host_losses = vec![0.0f32; TOKENS];
    let mut host_optimizer_step = 0u32;
    for reverse_rank in 0..TOKENS {
        let token_index = TOKENS - 1 - reverse_rank;
        let current = host_graph.train_worker_refinement_loss_one_submit_with_update_mode(
            worker_input(
                case,
                &host_h_states[token_index],
                &host_l_states[token_index],
                future_l.as_deref(),
                future_h.as_deref(),
            ),
            hyper,
            if reverse_rank == 0 {
                HierarchosFullModelUpdateMode::BeginAccumulation
            } else {
                HierarchosFullModelUpdateMode::FinishAccumulation
            },
        )?;
        host_losses[token_index] = current.output.loss;
        future_h = Some(current.h.sequence.grad_initial_packed_state.clone());
        future_l = Some(current.l.sequence.grad_initial_packed_state.clone());
        host_optimizer_step = current.full_model_optimizer.step;
    }
    let host_optimizer_state = host_graph.full_model_optimizer_state()?;

    let max_state_diff = max_abs_diff(&tape_result.final_h_packed_state, &host_h_states[TOKENS])?
        .max(max_abs_diff(
            &tape_result.final_l_packed_state,
            &host_l_states[TOKENS],
        )?);
    let max_adjoint_diff = max_abs_diff(
        &tape_result.grad_initial_h_packed_state,
        future_h
            .as_deref()
            .context("host token-tape reference produced no H adjoint")?,
    )?
    .max(max_abs_diff(
        &tape_result.grad_initial_l_packed_state,
        future_l
            .as_deref()
            .context("host token-tape reference produced no L adjoint")?,
    )?);
    let max_loss_diff = max_abs_diff(&tape_result.losses, &host_losses)?;
    let max_optimizer_diff =
        optimizer_state_max_diff(&tape_optimizer_state, &host_optimizer_state)?;

    let mut control_match = tape_result.controls.len() == TOKENS;
    for (actual, (selected_index, executed_steps, effective_l_steps)) in
        tape_result.controls.iter().zip(&host_controls)
    {
        control_match &= actual.manager_selected_index == *selected_index;
        control_match &= max_abs_diff(&actual.manager_executed_steps, executed_steps)? <= 1.0e-6;
        control_match &=
            max_abs_diff(&actual.worker_effective_l_steps, effective_l_steps)? <= 1.0e-6;
        control_match &= actual.worker_active_history.len() == case.shadow_steps + 1;
        if let Some(first) = actual.worker_active_history.first() {
            control_match &= first.iter().all(|value| (*value - 1.0).abs() <= 1.0e-6);
        }
        for pair in actual.worker_active_history.windows(2) {
            control_match &= pair[0]
                .iter()
                .zip(&pair[1])
                .all(|(before, after)| *after <= *before + 1.0e-6);
        }
    }
    control_match &= tape_result.full_model_optimizer.step == host_optimizer_step;

    Ok(TokenTapeCheck {
        tokens: TOKENS,
        queue_submissions: tape_result.queue_submissions,
        optimizer_step: tape_result.full_model_optimizer.step,
        max_state_diff,
        max_adjoint_diff,
        max_loss_diff,
        max_optimizer_diff,
        mean_normalization_max_moment_diff,
        control_match,
    })
}

fn verify_token_tape_microbatch_boundary(
    case: &Case,
    model_dir: &PathBuf,
    hyper: AdamWHyperParams,
) -> Result<TokenTapeMicrobatchCheck> {
    const SEQUENCES: usize = 2;
    const TOKENS: usize = 2;
    let total_tokens = SEQUENCES * TOKENS;

    let h_initials = [
        case.h_initial_packed_state.as_slice(),
        case.h_initial_packed_state.as_slice(),
    ];
    let l_initials = [
        case.l_initial_packed_state.as_slice(),
        case.l_initial_packed_state.as_slice(),
    ];

    let mut arena_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut arena =
        arena_graph.create_token_tape_arena(case.batch, TOKENS, &h_initials, &l_initials)?;
    let steps_a = token_tape_steps(case, TOKENS);
    let steps_b = token_tape_steps(case, TOKENS);
    let arena_inputs = [
        HierarchosTokenTapeMicrobatchInput {
            steps: &steps_a,
            final_h_packed_state_adjoint: case.h_selected_packed_state_grad.as_deref(),
            final_l_packed_state_adjoint: case.l_final_packed_state_grad.as_deref(),
            pytorch_tbptt_real_token_count: None,
        },
        HierarchosTokenTapeMicrobatchInput {
            steps: &steps_b,
            final_h_packed_state_adjoint: case.h_selected_packed_state_grad.as_deref(),
            final_l_packed_state_adjoint: case.l_final_packed_state_grad.as_deref(),
            pytorch_tbptt_real_token_count: None,
        },
    ];
    let arena_result = arena_graph.train_token_tape_microbatch_with_normalization(
        &mut arena,
        &arena_inputs,
        hyper,
        HierarchosSequenceGradientNormalization::Sum,
    )?;
    let arena_optimizer_state = arena_graph.full_model_optimizer_state()?;

    let mut mean_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut mean_arena =
        mean_graph.create_token_tape_arena(case.batch, TOKENS, &h_initials, &l_initials)?;
    let mean_steps_a = token_tape_steps(case, TOKENS);
    let mean_steps_b = token_tape_steps(case, TOKENS);
    let mean_inputs = [
        HierarchosTokenTapeMicrobatchInput {
            steps: &mean_steps_a,
            final_h_packed_state_adjoint: case.h_selected_packed_state_grad.as_deref(),
            final_l_packed_state_adjoint: case.l_final_packed_state_grad.as_deref(),
            pytorch_tbptt_real_token_count: None,
        },
        HierarchosTokenTapeMicrobatchInput {
            steps: &mean_steps_b,
            final_h_packed_state_adjoint: case.h_selected_packed_state_grad.as_deref(),
            final_l_packed_state_adjoint: case.l_final_packed_state_grad.as_deref(),
            pytorch_tbptt_real_token_count: None,
        },
    ];
    let mean_result = mean_graph.train_token_tape_microbatch_with_normalization(
        &mut mean_arena,
        &mean_inputs,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
    )?;
    let mean_optimizer_state = mean_graph.full_model_optimizer_state()?;
    let mean_normalization_max_moment_diff = optimizer_state_scaled_moment_diff(
        &arena_optimizer_state,
        &mean_optimizer_state,
        1.0 / total_tokens as f32,
    )?;
    let mut mean_observable_diff = 0.0f32;
    if arena_result.sequences.len() != mean_result.sequences.len() {
        anyhow::bail!(
            "microbatch mean normalization changed sequence count: sum={} mean={}",
            arena_result.sequences.len(),
            mean_result.sequences.len()
        );
    }
    for (summed, mean) in arena_result.sequences.iter().zip(&mean_result.sequences) {
        mean_observable_diff = mean_observable_diff
            .max(max_abs_diff(
                &summed.final_h_packed_state,
                &mean.final_h_packed_state,
            )?)
            .max(max_abs_diff(
                &summed.final_l_packed_state,
                &mean.final_l_packed_state,
            )?)
            .max(max_abs_diff(
                &summed.grad_initial_h_packed_state,
                &mean.grad_initial_h_packed_state,
            )?)
            .max(max_abs_diff(
                &summed.grad_initial_l_packed_state,
                &mean.grad_initial_l_packed_state,
            )?)
            .max(max_abs_diff(&summed.losses, &mean.losses)?);
    }
    if mean_observable_diff > 5.0e-6 || mean_normalization_max_moment_diff > 5.0e-6 {
        anyhow::bail!(
            "microbatch mean-by-token normalization changed observables or scaled AdamW moments incorrectly: observable_diff={mean_observable_diff:.9e} moment_diff={mean_normalization_max_moment_diff:.9e}"
        );
    }

    // Force the budget planner to keep exactly one sequence tape resident at a
    // time, then prove its automatic two-submit accumulation is identical to
    // the one-submit MeanByToken path.
    let mut split_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let one_sequence_footprint =
        split_graph.estimate_token_tape_footprint(case.batch, TOKENS, 1)?;
    let split_budget = split_graph.memory_budget()?;
    let forced_reserve = split_budget
        .device_local_budget_bytes
        .checked_sub(split_budget.device_local_usage_bytes)
        .and_then(|available| available.checked_sub(one_sequence_footprint.estimated_peak_bytes))
        .context("test device has insufficient Vulkan budget for one forced token tape")?;
    let budgeted_result = split_graph.train_token_tape_sequences_budgeted(
        case.batch,
        &h_initials,
        &l_initials,
        &mean_inputs,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
        HierarchosTapeMemoryPolicy {
            budget_fraction: 1.0,
            reserve_bytes: forced_reserve,
        },
    )?;
    if budgeted_result.plan.sequence_microbatch_size != 1
        || budgeted_result.plan.sequence_microbatch_count != SEQUENCES
        || budgeted_result.queue_submissions != SEQUENCES as u32
    {
        anyhow::bail!(
            "forced budgeted tape plan did not split one sequence per submit: microbatch_size={} microbatch_count={} submissions={}",
            budgeted_result.plan.sequence_microbatch_size,
            budgeted_result.plan.sequence_microbatch_count,
            budgeted_result.queue_submissions,
        );
    }
    let split_optimizer_state = split_graph.full_model_optimizer_state()?;
    let split_optimizer_diff =
        optimizer_state_max_diff(&mean_optimizer_state, &split_optimizer_state)?;
    let mut split_observable_diff = 0.0f32;
    for (split, one_submit) in budgeted_result.sequences.iter().zip(&mean_result.sequences) {
        split_observable_diff = split_observable_diff
            .max(max_abs_diff(
                &split.final_h_packed_state,
                &one_submit.final_h_packed_state,
            )?)
            .max(max_abs_diff(
                &split.final_l_packed_state,
                &one_submit.final_l_packed_state,
            )?)
            .max(max_abs_diff(&split.losses, &one_submit.losses)?);
    }
    if split_optimizer_diff > 5.0e-6 || split_observable_diff > 5.0e-6 {
        anyhow::bail!(
            "budgeted cross-submit token-tape accumulation diverged from one-submit MeanByToken: observable_diff={split_observable_diff:.9e} optimizer_diff={split_optimizer_diff:.9e}"
        );
    }

    // Weighted-token parity must be invariant to the same memory split even
    // when logical sequence lengths and supervised row mass are uneven. This
    // models a right-padded / response-masked accumulation window: zero rows do
    // not source CE gradients, fractional rows preserve response-token weights,
    // and the optimizer divides by the full window's supervision mass once.
    let weighted_a_weights = vec![
        (0..case.batch)
            .map(|row| if row == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        (0..case.batch)
            .map(|row| if row % 2 == 0 { 0.25 } else { 0.5 })
            .collect::<Vec<_>>(),
    ];
    let weighted_b_weights = vec![(0..case.batch)
        .map(|row| if row % 2 == 0 { 0.0 } else { 0.75 })
        .collect::<Vec<_>>()];
    let weighted_steps_a = token_tape_steps_with_supervision_weights(case, &weighted_a_weights);
    let weighted_steps_b = token_tape_steps_with_supervision_weights(case, &weighted_b_weights);
    let weighted_inputs = [
        HierarchosTokenTapeMicrobatchInput {
            steps: &weighted_steps_a,
            final_h_packed_state_adjoint: case.h_selected_packed_state_grad.as_deref(),
            final_l_packed_state_adjoint: case.l_final_packed_state_grad.as_deref(),
            pytorch_tbptt_real_token_count: Some(case.batch * weighted_steps_a.len()),
        },
        HierarchosTokenTapeMicrobatchInput {
            steps: &weighted_steps_b,
            final_h_packed_state_adjoint: case.h_selected_packed_state_grad.as_deref(),
            final_l_packed_state_adjoint: case.l_final_packed_state_grad.as_deref(),
            pytorch_tbptt_real_token_count: Some(case.batch * weighted_steps_b.len()),
        },
    ];

    let mut weighted_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut weighted_arena =
        weighted_graph.create_token_tape_arena(case.batch, TOKENS, &h_initials, &l_initials)?;
    let weighted_one_submit = weighted_graph.train_token_tape_microbatch_with_normalization(
        &mut weighted_arena,
        &weighted_inputs,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
    )?;
    let weighted_one_submit_optimizer = weighted_graph.full_model_optimizer_state()?;

    let mut weighted_split_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let weighted_split_budget = weighted_split_graph.memory_budget()?;
    let weighted_split_footprint =
        weighted_split_graph.estimate_token_tape_footprint(case.batch, TOKENS, 1)?;
    let weighted_forced_reserve = weighted_split_budget
        .device_local_budget_bytes
        .checked_sub(weighted_split_budget.device_local_usage_bytes)
        .and_then(|available| available.checked_sub(weighted_split_footprint.estimated_peak_bytes))
        .context("test device has insufficient Vulkan budget for one weighted token tape")?;
    let weighted_split = weighted_split_graph.train_token_tape_sequences_budgeted(
        case.batch,
        &h_initials,
        &l_initials,
        &weighted_inputs,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
        HierarchosTapeMemoryPolicy {
            budget_fraction: 1.0,
            reserve_bytes: weighted_forced_reserve,
        },
    )?;
    if weighted_split.plan.sequence_microbatch_size != 1
        || weighted_split.plan.sequence_microbatch_count != SEQUENCES
        || weighted_split.queue_submissions != SEQUENCES as u32
    {
        anyhow::bail!(
            "forced weighted-token budget plan did not split one sequence per submit: microbatch_size={} microbatch_count={} submissions={}",
            weighted_split.plan.sequence_microbatch_size,
            weighted_split.plan.sequence_microbatch_count,
            weighted_split.queue_submissions,
        );
    }
    let weighted_split_optimizer = weighted_split_graph.full_model_optimizer_state()?;
    let weighted_optimizer_diff =
        optimizer_state_max_diff(&weighted_one_submit_optimizer, &weighted_split_optimizer)?;
    let mut weighted_observable_diff = 0.0f32;
    for (split, one_submit) in weighted_split
        .sequences
        .iter()
        .zip(&weighted_one_submit.sequences)
    {
        weighted_observable_diff = weighted_observable_diff
            .max(max_abs_diff(
                &split.final_h_packed_state,
                &one_submit.final_h_packed_state,
            )?)
            .max(max_abs_diff(
                &split.final_l_packed_state,
                &one_submit.final_l_packed_state,
            )?)
            .max(max_abs_diff(&split.losses, &one_submit.losses)?);
    }
    if weighted_optimizer_diff > 5.0e-6 || weighted_observable_diff > 5.0e-6 {
        anyhow::bail!(
            "budgeted cross-submit weighted-token accumulation diverged from one-submit parity: observable_diff={weighted_observable_diff:.9e} optimizer_diff={weighted_optimizer_diff:.9e}"
        );
    }

    // Force the memory planner itself into sparse mode and compare its exact
    // replay against today's dense tape. Eight tokens with a stride-two
    // footprint target leave enough pressure to require multiple reverse
    // segments. The planner may select a larger equally-safe sparse stride when
    // projected peaks tie, so validate the sparse/multi-segment property rather
    // than overfitting this parity check to one tie-break result.
    const SPARSE_TOKENS: usize = 8;
    const SPARSE_STRIDE: usize = 2;
    let sparse_steps = token_tape_steps(case, SPARSE_TOKENS);
    let sparse_h_initials = [case.h_initial_packed_state.as_slice()];
    let sparse_l_initials = [case.l_initial_packed_state.as_slice()];
    let sparse_inputs = [HierarchosTokenTapeMicrobatchInput {
        steps: &sparse_steps,
        final_h_packed_state_adjoint: case.h_selected_packed_state_grad.as_deref(),
        final_l_packed_state_adjoint: case.l_final_packed_state_grad.as_deref(),
        pytorch_tbptt_real_token_count: None,
    }];

    let mut sparse_dense_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut sparse_dense_tape = sparse_dense_graph.create_token_tape(
        case.batch,
        SPARSE_TOKENS,
        &case.h_initial_packed_state,
        &case.l_initial_packed_state,
    )?;
    let sparse_dense_result = sparse_dense_graph.train_token_tape_with_normalization(
        &mut sparse_dense_tape,
        &sparse_steps,
        case.h_selected_packed_state_grad.as_deref(),
        case.l_final_packed_state_grad.as_deref(),
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
    )?;
    let sparse_dense_optimizer_state = sparse_dense_graph.full_model_optimizer_state()?;

    let mut sparse_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let sparse_footprint = sparse_graph.estimate_token_tape_footprint_with_stride(
        case.batch,
        SPARSE_TOKENS,
        1,
        SPARSE_STRIDE,
    )?;
    let sparse_dense_footprint =
        sparse_graph.estimate_token_tape_footprint(case.batch, SPARSE_TOKENS, 1)?;
    if sparse_footprint.estimated_peak_bytes >= sparse_dense_footprint.estimated_peak_bytes {
        anyhow::bail!(
            "sparse replay parity geometry does not reduce projected tape memory: sparse={} dense={}",
            sparse_footprint.estimated_peak_bytes,
            sparse_dense_footprint.estimated_peak_bytes,
        );
    }
    let sparse_budget = sparse_graph.memory_budget()?;
    // Leave the forced budget safely between the stride-2 and dense tape
    // footprints. The driver-visible usage can move slightly between this
    // probe and the scheduler's own budget query as Vulkan allocations are
    // committed; pinning available bytes to the exact stride-2 estimate made
    // this parity-only check nondeterministically fall through to stride 3.
    let sparse_budget_headroom = sparse_dense_footprint
        .estimated_peak_bytes
        .checked_sub(sparse_footprint.estimated_peak_bytes)
        .context("sparse replay parity footprint gap underflow")?
        / 2;
    let sparse_forced_available = sparse_footprint
        .estimated_peak_bytes
        .checked_add(sparse_budget_headroom)
        .context("forced sparse replay budget headroom overflow")?;
    let sparse_forced_reserve = sparse_budget
        .device_local_budget_bytes
        .checked_sub(sparse_budget.device_local_usage_bytes)
        .and_then(|available| available.checked_sub(sparse_forced_available))
        .context("test device has insufficient Vulkan budget for forced sparse replay")?;
    let sparse_result = sparse_graph.train_token_tape_sequences_budgeted(
        case.batch,
        &sparse_h_initials,
        &sparse_l_initials,
        &sparse_inputs,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
        HierarchosTapeMemoryPolicy {
            budget_fraction: 1.0,
            reserve_bytes: sparse_forced_reserve,
        },
    )?;
    if !sparse_result.plan.requires_sparse_state_replay
        || sparse_result.plan.state_checkpoint_stride < SPARSE_STRIDE
        || sparse_result.plan.state_checkpoint_stride >= SPARSE_TOKENS
    {
        anyhow::bail!(
            "forced sparse replay plan did not select a multi-segment sparse stride in {SPARSE_STRIDE}..{SPARSE_TOKENS}: requires_sparse={} stride={} dense_peak={} planned_peak={} available={}",
            sparse_result.plan.requires_sparse_state_replay,
            sparse_result.plan.state_checkpoint_stride,
            sparse_result.plan.dense_requested_peak_bytes,
            sparse_result.plan.planned_peak_bytes,
            sparse_result.plan.available_for_tape_bytes,
        );
    }
    let sparse_optimizer_state = sparse_graph.full_model_optimizer_state()?;
    let sparse_actual = sparse_result
        .sequences
        .first()
        .context("sparse replay parity produced no sequence result")?;
    let sparse_replay_max_state_diff = max_abs_diff(
        &sparse_actual.final_h_packed_state,
        &sparse_dense_result.final_h_packed_state,
    )?
    .max(max_abs_diff(
        &sparse_actual.final_l_packed_state,
        &sparse_dense_result.final_l_packed_state,
    )?);
    let sparse_replay_max_adjoint_diff = max_abs_diff(
        &sparse_actual.grad_initial_h_packed_state,
        &sparse_dense_result.grad_initial_h_packed_state,
    )?
    .max(max_abs_diff(
        &sparse_actual.grad_initial_l_packed_state,
        &sparse_dense_result.grad_initial_l_packed_state,
    )?);
    let sparse_replay_max_loss_diff =
        max_abs_diff(&sparse_actual.losses, &sparse_dense_result.losses)?;
    let sparse_replay_max_optimizer_diff =
        optimizer_state_max_diff(&sparse_optimizer_state, &sparse_dense_optimizer_state)?;
    let sparse_replay_control_match =
        token_tape_controls_match(&sparse_actual.controls, &sparse_dense_result.controls)?
            && sparse_result.full_model_optimizer.step
                == sparse_dense_result.full_model_optimizer.step;
    if sparse_result.queue_submissions != 1
        || !sparse_replay_control_match
        || sparse_replay_max_state_diff > 5.0e-6
        || sparse_replay_max_adjoint_diff > 5.0e-6
        || sparse_replay_max_loss_diff > 5.0e-6
        || sparse_replay_max_optimizer_diff > 5.0e-6
    {
        anyhow::bail!(
            "sparse segment replay diverged from dense token tape: submissions={} state_diff={sparse_replay_max_state_diff:.9e} adjoint_diff={sparse_replay_max_adjoint_diff:.9e} loss_diff={sparse_replay_max_loss_diff:.9e} optimizer_diff={sparse_replay_max_optimizer_diff:.9e} control_match={sparse_replay_control_match}",
            sparse_result.queue_submissions,
        );
    }

    // Explicit reference: materialize each sequence boundary on the host, keep
    // parameters fixed while all forward trajectories are captured, then walk
    // each sequence backward with one shared Begin/Accumulate/Finish window.
    // This reproduces PyTorch-style gradient accumulation while retaining
    // independent recurrent adjoint chains.
    let mut host_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;
    let mut host_h_states = Vec::with_capacity(SEQUENCES);
    let mut host_l_states = Vec::with_capacity(SEQUENCES);
    let mut host_controls = Vec::with_capacity(SEQUENCES);
    let mut forward_ordinal = 0usize;
    for _ in 0..SEQUENCES {
        let mut h_states = vec![case.h_initial_packed_state.clone()];
        let mut l_states = vec![case.l_initial_packed_state.clone()];
        let mut controls = Vec::with_capacity(TOKENS);
        for token_index in 0..TOKENS {
            let current = host_graph.train_worker_refinement_loss_one_submit_with_update_mode(
                worker_input(
                    case,
                    &h_states[token_index],
                    &l_states[token_index],
                    None,
                    None,
                ),
                hyper,
                if forward_ordinal == 0 {
                    HierarchosFullModelUpdateMode::BeginAccumulation
                } else {
                    HierarchosFullModelUpdateMode::Accumulate
                },
            )?;
            controls.push((
                current.manager_selected_index.clone(),
                current.manager_executed_steps.clone(),
                current.effective_l_steps.clone(),
            ));
            h_states.push(current.manager_selected_packed_state);
            l_states.push(current.l.sequence.final_packed_state);
            forward_ordinal += 1;
        }
        host_h_states.push(h_states);
        host_l_states.push(l_states);
        host_controls.push(controls);
    }

    // As above, forward-only boundary materialization used an open scratch
    // accumulation window to avoid stepping parameters. Reverse parity needs a
    // fresh optimizer lifecycle while retaining only those host state snapshots.
    host_graph = HierarchosTrainingGraph::from_model_package(
        VulkanDevice::new()?,
        model_dir,
        case.batch,
        case.h_steps.max(1),
        case.shadow_steps.max(1),
        case.batch,
    )?;

    let mut host_losses = vec![vec![0.0f32; TOKENS]; SEQUENCES];
    let mut host_initial_h_adjoints = Vec::with_capacity(SEQUENCES);
    let mut host_initial_l_adjoints = Vec::with_capacity(SEQUENCES);
    let mut reverse_ordinal = 0usize;
    let mut host_optimizer_step = 0u32;
    for sequence_index in 0..SEQUENCES {
        let mut future_h = case.h_selected_packed_state_grad.clone();
        let mut future_l = case.l_final_packed_state_grad.clone();
        for reverse_rank in 0..TOKENS {
            let token_index = TOKENS - 1 - reverse_rank;
            let current = host_graph.train_worker_refinement_loss_one_submit_with_update_mode(
                worker_input(
                    case,
                    &host_h_states[sequence_index][token_index],
                    &host_l_states[sequence_index][token_index],
                    future_l.as_deref(),
                    future_h.as_deref(),
                ),
                hyper,
                if reverse_ordinal == 0 {
                    HierarchosFullModelUpdateMode::BeginAccumulation
                } else if reverse_ordinal + 1 == total_tokens {
                    HierarchosFullModelUpdateMode::FinishAccumulation
                } else {
                    HierarchosFullModelUpdateMode::Accumulate
                },
            )?;
            host_losses[sequence_index][token_index] = current.output.loss;
            future_h = Some(current.h.sequence.grad_initial_packed_state.clone());
            future_l = Some(current.l.sequence.grad_initial_packed_state.clone());
            host_optimizer_step = current.full_model_optimizer.step;
            reverse_ordinal += 1;
        }
        host_initial_h_adjoints
            .push(future_h.context("microbatch host reference produced no H initial adjoint")?);
        host_initial_l_adjoints
            .push(future_l.context("microbatch host reference produced no L initial adjoint")?);
    }
    let host_optimizer_state = host_graph.full_model_optimizer_state()?;

    if arena_result.sequences.len() != SEQUENCES {
        anyhow::bail!(
            "token-tape arena returned {} sequences; expected {SEQUENCES}",
            arena_result.sequences.len()
        );
    }
    let mut max_state_diff = 0.0f32;
    let mut max_adjoint_diff = 0.0f32;
    let mut max_loss_diff = 0.0f32;
    let mut control_match = true;
    for sequence_index in 0..SEQUENCES {
        let actual = &arena_result.sequences[sequence_index];
        max_state_diff = max_state_diff
            .max(max_abs_diff(
                &actual.final_h_packed_state,
                &host_h_states[sequence_index][TOKENS],
            )?)
            .max(max_abs_diff(
                &actual.final_l_packed_state,
                &host_l_states[sequence_index][TOKENS],
            )?);
        max_adjoint_diff = max_adjoint_diff
            .max(max_abs_diff(
                &actual.grad_initial_h_packed_state,
                &host_initial_h_adjoints[sequence_index],
            )?)
            .max(max_abs_diff(
                &actual.grad_initial_l_packed_state,
                &host_initial_l_adjoints[sequence_index],
            )?);
        max_loss_diff =
            max_loss_diff.max(max_abs_diff(&actual.losses, &host_losses[sequence_index])?);
        control_match &= actual.controls.len() == TOKENS;
        for (actual_control, (selected_index, executed_steps, effective_l_steps)) in
            actual.controls.iter().zip(&host_controls[sequence_index])
        {
            control_match &= actual_control.manager_selected_index == *selected_index;
            control_match &=
                max_abs_diff(&actual_control.manager_executed_steps, executed_steps)? <= 1.0e-6;
            control_match &=
                max_abs_diff(&actual_control.worker_effective_l_steps, effective_l_steps)?
                    <= 1.0e-6;
            control_match &= actual_control.worker_active_history.len() == case.shadow_steps + 1;
        }
    }
    control_match &= arena_result.full_model_optimizer.step == host_optimizer_step;
    let max_optimizer_diff =
        optimizer_state_max_diff(&arena_optimizer_state, &host_optimizer_state)?;

    Ok(TokenTapeMicrobatchCheck {
        sequences: SEQUENCES,
        total_tokens: arena_result.total_tokens,
        queue_submissions: arena_result.queue_submissions,
        descriptor_pool_count: arena_result.descriptor_pool_count,
        descriptor_set_count: arena_result.descriptor_set_count,
        dispatch_count: arena_result.dispatch_count,
        shader_barrier_count: arena_result.shader_barrier_count,
        pipeline_bind_count: arena_result.pipeline_bind_count,
        descriptor_bind_count: arena_result.descriptor_bind_count,
        push_constant_write_count: arena_result.push_constant_write_count,
        upload_count: arena_result.upload_count,
        uploaded_bytes: arena_result.uploaded_bytes,
        upload_arena_buffer_count: arena_result.upload_arena_buffer_count,
        optimizer_step: arena_result.full_model_optimizer.step,
        max_state_diff,
        max_adjoint_diff,
        max_loss_diff,
        max_optimizer_diff,
        mean_normalization_max_moment_diff,
        control_match,
        sparse_replay_tokens: SPARSE_TOKENS,
        sparse_replay_checkpoint_stride: sparse_result.plan.state_checkpoint_stride,
        sparse_replay_queue_submissions: sparse_result.queue_submissions,
        sparse_replay_max_state_diff,
        sparse_replay_max_adjoint_diff,
        sparse_replay_max_loss_diff,
        sparse_replay_max_optimizer_diff,
        sparse_replay_control_match,
    })
}

fn token_tape_steps<'a>(case: &'a Case, tokens: usize) -> Vec<HierarchosTokenTapeStepInput<'a>> {
    (0..tokens)
        .map(|_| HierarchosTokenTapeStepInput {
            h_steps: case.h_steps,
            shadow_steps: case.shadow_steps,
            enc: &case.enc,
            previous_context: &case.previous_context,
            target_context: &case.target_context,
            context_alpha: case.context_alpha,
            h_token_ids: &case.h_token_ids,
            l_token_ids: &case.l_token_ids,
            h_to_context_grad: &case.h_to_context_grad,
            h_depth_grad: &case.h_depth_grad,
            final_drift_grad: &case.final_drift_grad,
            commitment_cost_grad: &case.commitment_cost_grad,
            targets: &case.targets,
            supervision_weights: None,
        })
        .collect()
}

fn token_tape_steps_with_supervision_weights<'a>(
    case: &'a Case,
    supervision_weights: &'a [Vec<f32>],
) -> Vec<HierarchosTokenTapeStepInput<'a>> {
    supervision_weights
        .iter()
        .map(|weights| HierarchosTokenTapeStepInput {
            h_steps: case.h_steps,
            shadow_steps: case.shadow_steps,
            enc: &case.enc,
            previous_context: &case.previous_context,
            target_context: &case.target_context,
            context_alpha: case.context_alpha,
            h_token_ids: &case.h_token_ids,
            l_token_ids: &case.l_token_ids,
            h_to_context_grad: &case.h_to_context_grad,
            h_depth_grad: &case.h_depth_grad,
            final_drift_grad: &case.final_drift_grad,
            commitment_cost_grad: &case.commitment_cost_grad,
            targets: &case.targets,
            supervision_weights: Some(weights),
        })
        .collect()
}

fn worker_input<'a>(
    case: &'a Case,
    h_state: &'a [f32],
    l_state: &'a [f32],
    l_future_state_grad: Option<&'a [f32]>,
    h_future_state_grad: Option<&'a [f32]>,
) -> HierarchosWorkerRefinementLossInput<'a> {
    HierarchosWorkerRefinementLossInput {
        batch: case.batch,
        h_steps: case.h_steps,
        shadow_steps: case.shadow_steps,
        enc: &case.enc,
        previous_context: &case.previous_context,
        target_context: &case.target_context,
        context_alpha: case.context_alpha,
        h_token_ids: &case.h_token_ids,
        l_token_ids: &case.l_token_ids,
        h_initial_packed_state: h_state,
        l_initial_packed_state: l_state,
        l_final_packed_state_grad: l_future_state_grad,
        h_to_context_grad: &case.h_to_context_grad,
        h_depth_grad: &case.h_depth_grad,
        h_selected_packed_state_grad: h_future_state_grad,
        final_drift_grad: &case.final_drift_grad,
        commitment_cost_grad: &case.commitment_cost_grad,
        targets: &case.targets,
        supervision_weights: None,
    }
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        anyhow::bail!(
            "token-tape parity length mismatch: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        );
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max))
}

fn token_tape_controls_match(
    lhs: &[HierarchosTokenTapeControlSnapshot],
    rhs: &[HierarchosTokenTapeControlSnapshot],
) -> Result<bool> {
    if lhs.len() != rhs.len() {
        return Ok(false);
    }
    for (left, right) in lhs.iter().zip(rhs) {
        if left.manager_selected_index != right.manager_selected_index
            || max_abs_diff(&left.manager_executed_steps, &right.manager_executed_steps)? > 1.0e-6
            || max_abs_diff(
                &left.worker_effective_l_steps,
                &right.worker_effective_l_steps,
            )? > 1.0e-6
            || left.worker_active_history.len() != right.worker_active_history.len()
        {
            return Ok(false);
        }
        for (left_active, right_active) in left
            .worker_active_history
            .iter()
            .zip(&right.worker_active_history)
        {
            if max_abs_diff(left_active, right_active)? > 1.0e-6 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn optimizer_state_max_diff(lhs: &AdamWOptimizerState, rhs: &AdamWOptimizerState) -> Result<f32> {
    if lhs.step != rhs.step || lhs.slots.len() != rhs.slots.len() {
        anyhow::bail!(
            "token-tape optimizer metadata mismatch: lhs_step={} rhs_step={} lhs_slots={} rhs_slots={}",
            lhs.step,
            rhs.step,
            lhs.slots.len(),
            rhs.slots.len()
        );
    }
    let mut max_diff = 0.0f32;
    for (left, right) in lhs.slots.iter().zip(&rhs.slots) {
        if left.name != right.name {
            anyhow::bail!(
                "token-tape optimizer slot mismatch: {:?} vs {:?}",
                left.name,
                right.name
            );
        }
        max_diff = max_diff.max(max_abs_diff(&left.exp_avg, &right.exp_avg)?);
        max_diff = max_diff.max(max_abs_diff(&left.exp_avg_sq, &right.exp_avg_sq)?);
    }
    Ok(max_diff)
}

fn optimizer_state_scaled_moment_diff(
    summed: &AdamWOptimizerState,
    scaled: &AdamWOptimizerState,
    gradient_scale: f32,
) -> Result<f32> {
    if summed.step != scaled.step || summed.slots.len() != scaled.slots.len() {
        anyhow::bail!(
            "normalized token-tape optimizer metadata mismatch: sum_step={} scaled_step={} sum_slots={} scaled_slots={}",
            summed.step,
            scaled.step,
            summed.slots.len(),
            scaled.slots.len()
        );
    }
    let squared_scale = gradient_scale * gradient_scale;
    let mut max_diff = 0.0f32;
    for (sum_slot, scaled_slot) in summed.slots.iter().zip(&scaled.slots) {
        if sum_slot.name != scaled_slot.name {
            anyhow::bail!(
                "normalized token-tape optimizer slot mismatch: {:?} vs {:?}",
                sum_slot.name,
                scaled_slot.name
            );
        }
        for (sum_value, scaled_value) in sum_slot.exp_avg.iter().zip(&scaled_slot.exp_avg) {
            max_diff = max_diff.max((sum_value * gradient_scale - scaled_value).abs());
        }
        for (sum_value, scaled_value) in sum_slot.exp_avg_sq.iter().zip(&scaled_slot.exp_avg_sq) {
            max_diff = max_diff.max((sum_value * squared_scale - scaled_value).abs());
        }
    }
    Ok(max_diff)
}

fn snapshots(values: Vec<RwkvParameterSnapshot>) -> Vec<ParameterOutput> {
    values
        .into_iter()
        .map(|snapshot| ParameterOutput {
            name: snapshot.name,
            values: snapshot.values,
        })
        .collect()
}
