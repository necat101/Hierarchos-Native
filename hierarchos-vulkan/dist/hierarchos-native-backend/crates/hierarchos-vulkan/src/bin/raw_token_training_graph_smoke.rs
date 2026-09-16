use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    read_f32_tensor, AdamWHyperParams, HierarchosDataStreamCursorState,
    HierarchosExecutionPolicyState, HierarchosLearningRateScheduleState,
    HierarchosLossScalingState, HierarchosPortableReplayTensor, HierarchosPortableTrainingReplay,
    HierarchosRawTokenTapeStepInput, HierarchosRawTokenWorkerRefinementLossInput,
    HierarchosStochasticRngPolicyState, HierarchosTokenFrontendOp,
    HierarchosTokenMemoryFrontendInput, HierarchosTrainingGraph, HierarchosTrainingSessionState,
    HierarchosWorkerRefinementLossInput, VulkanDevice, HIERARCHOS_VULKAN_DATA_STREAM_CURSOR_FORMAT,
    HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT, HIERARCHOS_VULKAN_PORTABLE_SAMPLER_RNG_ALGORITHM,
    HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT,
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
    token_ids: Vec<u32>,
    previous_context: Vec<f32>,
    target_context: Vec<f32>,
    context_alpha: f32,
    h_token_ids: Vec<u32>,
    l_token_ids: Vec<u32>,
    h_initial_packed_state: Vec<f32>,
    l_initial_packed_state: Vec<f32>,
    h_to_context_grad: Vec<f32>,
    h_depth_grad: Vec<f32>,
    final_drift_grad: Vec<f32>,
    commitment_cost_grad: Vec<f32>,
    targets: Vec<u32>,
    optimizer: OptimizerCase,
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
    projection_native_fp16_backward_compute_active: bool,
    lm_head_fp16_parameter_storage_active: bool,
    lm_head_execution_arm: String,
    lm_head_weight_grad_topology: Option<String>,
    lm_head_fused_adjoint_topology: Option<String>,
    lm_head_native_fp16_backward_compute_active: bool,
    out_norm_native_fp16_backward_compute_active: bool,
    batch: usize,
    raw_queue_submissions: u32,
    reference_queue_submissions: u32,
    raw_loss: f32,
    reference_loss: f32,
    loss_abs_diff: f32,
    grad_enc_max_abs_diff: f32,
    grad_previous_context_max_abs_diff: f32,
    h_output_max_abs_diff: f32,
    l_output_max_abs_diff: f32,
    raw_tape_tokens: usize,
    raw_tape_queue_submissions: u32,
    raw_tape_optimizer_step: u32,
    raw_tape_losses_finite: bool,
    raw_tape_rosa_predictions: Vec<Vec<i64>>,
    raw_tape_frontend_moment_l1: f32,
    reference_optimizer_tensor_count: usize,
    raw_optimizer_tensor_count: usize,
    raw_optimizer_names: Vec<String>,
    frontend_moment_l1: f32,
    val_proj_moment_l1: f32,
    training_step: u64,
    checkpoint_warmup_step: f32,
    checkpoint_in_proj_max_abs_delta: f32,
    checkpoint_val_proj_max_abs_delta: f32,
    checkpoint_fast_vals_exact: bool,
    training_checkpoint_manifest_roundtrip: bool,
    optimizer_checkpoint_roundtrip: bool,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> Result<f32> {
    anyhow::ensure!(
        a.len() == b.len(),
        "vector length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    Ok(a.iter()
        .zip(b)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max))
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut case_path = None;
    let mut trained_model = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--case" => case_path = args.next().map(PathBuf::from),
            "--trained-model" => trained_model = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let model_dir = model_dir.context("missing --model MODEL_DIR")?;
    let case_path = case_path.context("missing --case CASE.json")?;
    let trained_model = trained_model.context("missing --trained-model OUTPUT.safetensors")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    let batch = case.token_ids.len();
    anyhow::ensure!(batch > 0, "smoke case requires at least one raw token");

    let hyper = AdamWHyperParams {
        lr: case.optimizer.lr,
        beta1: case.optimizer.beta1,
        beta2: case.optimizer.beta2,
        eps: case.optimizer.eps,
        weight_decay: case.optimizer.weight_decay,
    };
    let device = VulkanDevice::new()?;

    // Build an independent frontend only to generate the legacy host-enc oracle.
    // Each batch row is reset and evaluated separately, matching independent
    // ROSA lane semantics instead of treating rows as one contiguous sequence.
    let mut oracle_frontend =
        HierarchosTokenFrontendOp::from_model_package(device.clone(), &model_dir, 1)?;
    oracle_frontend.set_training_step(7);
    let context_dim = oracle_frontend.config().context_dim;
    anyhow::ensure!(
        case.previous_context.len() == batch * context_dim,
        "previous_context length does not match batch/context geometry"
    );
    let mut oracle_enc = Vec::with_capacity(batch * context_dim);
    for row in 0..batch {
        oracle_frontend.reset_rosa_state()?;
        let context_start = row * context_dim;
        oracle_enc.extend(
            oracle_frontend
                .forward_memory(HierarchosTokenMemoryFrontendInput {
                    token_ids: &case.token_ids[row..row + 1],
                    prev_context: &case.previous_context
                        [context_start..context_start + context_dim],
                })?
                .enc,
        );
    }

    let mut reference_graph = HierarchosTrainingGraph::from_model_package(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
    )?;
    let reference = reference_graph.train_worker_refinement_loss_one_submit(
        HierarchosWorkerRefinementLossInput {
            batch,
            h_steps: 1,
            shadow_steps: 1,
            enc: &oracle_enc,
            previous_context: &case.previous_context,
            target_context: &case.target_context,
            context_alpha: case.context_alpha,
            h_token_ids: &case.h_token_ids,
            l_token_ids: &case.l_token_ids,
            h_initial_packed_state: &case.h_initial_packed_state,
            l_initial_packed_state: &case.l_initial_packed_state,
            l_final_packed_state_grad: None,
            h_to_context_grad: &case.h_to_context_grad,
            h_depth_grad: &case.h_depth_grad,
            h_selected_packed_state_grad: None,
            final_drift_grad: &case.final_drift_grad,
            commitment_cost_grad: &case.commitment_cost_grad,
            targets: &case.targets,
            supervision_weights: None,
        },
        hyper,
    )?;

    let mut raw_graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    raw_graph.set_training_step(7)?;
    let rosa_reset_lanes = vec![1u32; batch];
    let raw = raw_graph.train_raw_token_worker_refinement_loss_one_submit(
        HierarchosRawTokenWorkerRefinementLossInput {
            batch,
            h_steps: 1,
            shadow_steps: 1,
            token_ids: &case.token_ids,
            rosa_reset_lanes: &rosa_reset_lanes,
            previous_context: &case.previous_context,
            target_context: &case.target_context,
            context_alpha: case.context_alpha,
            h_token_ids: &case.h_token_ids,
            l_token_ids: &case.l_token_ids,
            h_initial_packed_state: &case.h_initial_packed_state,
            l_initial_packed_state: &case.l_initial_packed_state,
            l_final_packed_state_grad: None,
            h_to_context_grad: &case.h_to_context_grad,
            h_depth_grad: &case.h_depth_grad,
            h_selected_packed_state_grad: None,
            final_drift_grad: &case.final_drift_grad,
            commitment_cost_grad: &case.commitment_cost_grad,
            ltm_value_alignment_position: 0,
            ltm_value_alignment_mask: None,
            ltm_value_alignment_grad: 0.25,
            targets: &case.targets,
            supervision_weights: None,
        },
        hyper,
    )?;

    let loss_abs_diff = (raw.output.loss - reference.output.loss).abs();
    let grad_enc_max_abs_diff = max_abs_diff(&raw.grad_enc, &reference.grad_enc)?;
    let mut frontend_prev_context_grad = Vec::with_capacity(batch * context_dim);
    let mut grad_oracle_frontend =
        HierarchosTokenFrontendOp::from_model_package(device.clone(), &model_dir, 1)?;
    grad_oracle_frontend.set_training_step(7);
    for row in 0..batch {
        grad_oracle_frontend.reset_rosa_state()?;
        let context_start = row * context_dim;
        frontend_prev_context_grad.extend(
            grad_oracle_frontend
                .forward_memory_backward(
                    HierarchosTokenMemoryFrontendInput {
                        token_ids: &case.token_ids[row..row + 1],
                        prev_context: &case.previous_context
                            [context_start..context_start + context_dim],
                    },
                    &raw.grad_enc[context_start..context_start + context_dim],
                )?
                .grad_prev_context,
        );
    }
    let expected_raw_prev_context_grad = reference
        .grad_previous_context
        .iter()
        .zip(&frontend_prev_context_grad)
        .map(|(worker, frontend)| worker + frontend)
        .collect::<Vec<_>>();
    let grad_previous_context_max_abs_diff =
        max_abs_diff(&raw.grad_previous_context, &expected_raw_prev_context_grad)?;
    let h_output_max_abs_diff =
        max_abs_diff(&raw.h.sequence.outputs, &reference.h.sequence.outputs)?;
    let l_output_max_abs_diff =
        max_abs_diff(&raw.l.sequence.outputs, &reference.l.sequence.outputs)?;
    anyhow::ensure!(
        raw.queue_submissions == 1,
        "raw graph must submit exactly once"
    );
    anyhow::ensure!(
        loss_abs_diff <= 2.0e-5,
        "raw/legacy loss drift: {loss_abs_diff}"
    );
    anyhow::ensure!(
        grad_enc_max_abs_diff <= 2.0e-5,
        "raw/legacy d(enc) drift: {grad_enc_max_abs_diff}"
    );
    anyhow::ensure!(
        grad_previous_context_max_abs_diff <= 2.0e-5,
        "raw full previous-context adjoint drift: {grad_previous_context_max_abs_diff}"
    );
    anyhow::ensure!(
        h_output_max_abs_diff <= 2.0e-5,
        "raw/legacy H output drift: {h_output_max_abs_diff}"
    );
    anyhow::ensure!(
        l_output_max_abs_diff <= 2.0e-5,
        "raw/legacy L output drift: {l_output_max_abs_diff}"
    );

    // Exercise the multi-token raw tape with a repeated pattern so the third
    // token in each lane has a non-empty ROSA prediction. Forward checkpointing
    // must retain those discrete decisions while reverse rematerialization runs
    // without advancing ROSA a second time.
    anyhow::ensure!(batch == 2, "raw tape smoke currently expects batch=2");
    let tape_token_0 = case.token_ids.clone();
    let tape_token_1 = vec![5u32, 3u32];
    let tape_token_2 = case.token_ids.clone();
    let reset_first = vec![1u32; batch];
    let reset_none = vec![0u32; batch];
    let mut raw_tape_graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    raw_tape_graph.set_training_step(7)?;
    let mut raw_tape = raw_tape_graph.create_token_tape(
        batch,
        3,
        &case.h_initial_packed_state,
        &case.l_initial_packed_state,
    )?;
    let raw_tape_steps = [
        HierarchosRawTokenTapeStepInput {
            h_steps: 1,
            shadow_steps: 1,
            token_ids: &tape_token_0,
            rosa_reset_lanes: &reset_first,
            previous_context: &case.previous_context,
            target_context: &case.target_context,
            context_alpha: case.context_alpha,
            h_token_ids: &tape_token_0,
            l_token_ids: &tape_token_0,
            h_to_context_grad: &case.h_to_context_grad,
            h_depth_grad: &case.h_depth_grad,
            final_drift_grad: &case.final_drift_grad,
            commitment_cost_grad: &case.commitment_cost_grad,
            ltm_value_alignment_position: 0,
            ltm_value_alignment_mask: None,
            ltm_value_alignment_grad: 0.0,
            targets: &case.targets,
            supervision_weights: None,
            pytorch_tbptt_token_mask: None,
        },
        HierarchosRawTokenTapeStepInput {
            h_steps: 1,
            shadow_steps: 1,
            token_ids: &tape_token_1,
            rosa_reset_lanes: &reset_none,
            previous_context: &case.previous_context,
            target_context: &case.target_context,
            context_alpha: case.context_alpha,
            h_token_ids: &tape_token_1,
            l_token_ids: &tape_token_1,
            h_to_context_grad: &case.h_to_context_grad,
            h_depth_grad: &case.h_depth_grad,
            final_drift_grad: &case.final_drift_grad,
            commitment_cost_grad: &case.commitment_cost_grad,
            ltm_value_alignment_position: 1,
            ltm_value_alignment_mask: None,
            ltm_value_alignment_grad: 0.0,
            targets: &case.targets,
            supervision_weights: None,
            pytorch_tbptt_token_mask: None,
        },
        HierarchosRawTokenTapeStepInput {
            h_steps: 1,
            shadow_steps: 1,
            token_ids: &tape_token_2,
            rosa_reset_lanes: &reset_none,
            previous_context: &case.previous_context,
            target_context: &case.target_context,
            context_alpha: case.context_alpha,
            h_token_ids: &tape_token_2,
            l_token_ids: &tape_token_2,
            h_to_context_grad: &case.h_to_context_grad,
            h_depth_grad: &case.h_depth_grad,
            final_drift_grad: &case.final_drift_grad,
            commitment_cost_grad: &case.commitment_cost_grad,
            ltm_value_alignment_position: 2,
            ltm_value_alignment_mask: None,
            ltm_value_alignment_grad: 0.0,
            targets: &case.targets,
            supervision_weights: None,
            pytorch_tbptt_token_mask: None,
        },
    ];
    let raw_tape_result =
        raw_tape_graph.train_raw_token_tape(&mut raw_tape, &raw_tape_steps, None, None, hyper)?;
    let raw_tape_rosa_predictions = raw_tape.rosa_prediction_checkpoints()?;
    let expected_tape_rosa_predictions = vec![vec![-1, -1], vec![-1, -1], vec![5, 3]];
    anyhow::ensure!(
        raw_tape_rosa_predictions == expected_tape_rosa_predictions,
        "raw tape ROSA checkpoints drifted: expected={expected_tape_rosa_predictions:?} actual={raw_tape_rosa_predictions:?}"
    );
    anyhow::ensure!(
        raw_tape_result.queue_submissions == 1 && raw_tape_result.full_model_optimizer.step == 1,
        "raw tape must run one Vulkan submission and one canonical AdamW step"
    );
    let raw_tape_losses_finite = raw_tape_result.losses.iter().all(|loss| loss.is_finite());
    anyhow::ensure!(raw_tape_losses_finite, "raw tape produced non-finite loss");
    let raw_tape_optimizer_state = raw_tape_graph.full_model_optimizer_state()?;
    let raw_tape_frontend_moment_l1 = raw_tape_optimizer_state
        .slots
        .iter()
        .filter(|slot| {
            matches!(
                slot.name.as_str(),
                "persistent"
                    | "rosa_adapter.down.weight"
                    | "rosa_adapter.up.weight"
                    | "rosa_adapter.bias"
                    | "rosa_gate_logit"
                    | "rosa_router.weight"
                    | "rosa_router.bias"
                    | "qproj.weight"
                    | "val_proj.weight"
                    | "ltm.keys"
                    | "ltm.vals"
                    | "ltm_gate_logit"
                    | "ltm_router.weight"
                    | "ltm_router.bias"
                    | "in_proj.weight"
                    | "in_proj.bias"
            )
        })
        .flat_map(|slot| slot.exp_avg.iter())
        .map(|value| value.abs())
        .sum::<f32>();
    anyhow::ensure!(
        raw_tape_frontend_moment_l1 > 0.0 && raw_tape_frontend_moment_l1.is_finite(),
        "raw tape frontend gradients did not reach canonical AdamW moments"
    );

    let reference_state = reference_graph.full_model_optimizer_state()?;
    let raw_state = raw_graph.full_model_optimizer_state()?;
    anyhow::ensure!(
        raw_state.slots.len() == reference_state.slots.len() + 16,
        "raw registry should add exactly 16 frontend tensors; reference={} raw={}",
        reference_state.slots.len(),
        raw_state.slots.len()
    );
    let raw_names = raw_state
        .slots
        .iter()
        .map(|slot| slot.name.clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        raw_names
            .iter()
            .filter(|name| name.as_str() == "lm_head.weight")
            .count()
            == 1,
        "raw registry must contain exactly one tied lm_head.weight slot"
    );
    for required in [
        "persistent",
        "rosa_adapter.down.weight",
        "rosa_adapter.up.weight",
        "rosa_adapter.bias",
        "rosa_gate_logit",
        "rosa_router.weight",
        "rosa_router.bias",
        "qproj.weight",
        "val_proj.weight",
        "ltm.keys",
        "ltm.vals",
        "ltm_gate_logit",
        "ltm_router.weight",
        "ltm_router.bias",
        "in_proj.weight",
        "in_proj.bias",
    ] {
        anyhow::ensure!(
            raw_names.iter().any(|name| name == required),
            "missing raw optimizer slot {required}"
        );
    }
    let frontend_moment_l1 = raw_state
        .slots
        .iter()
        .filter(|slot| {
            matches!(
                slot.name.as_str(),
                "persistent"
                    | "rosa_adapter.down.weight"
                    | "rosa_adapter.up.weight"
                    | "rosa_adapter.bias"
                    | "rosa_gate_logit"
                    | "rosa_router.weight"
                    | "rosa_router.bias"
                    | "qproj.weight"
                    | "val_proj.weight"
                    | "ltm.keys"
                    | "ltm.vals"
                    | "ltm_gate_logit"
                    | "ltm_router.weight"
                    | "ltm_router.bias"
                    | "in_proj.weight"
                    | "in_proj.bias"
            )
        })
        .flat_map(|slot| slot.exp_avg.iter())
        .map(|value| value.abs())
        .sum::<f32>();
    anyhow::ensure!(
        frontend_moment_l1 > 0.0 && frontend_moment_l1.is_finite(),
        "raw frontend optimizer moments stayed zero"
    );
    let val_proj_moment_l1 = raw_state
        .slots
        .iter()
        .find(|slot| slot.name == "val_proj.weight")
        .context("raw optimizer lost val_proj.weight")?
        .exp_avg
        .iter()
        .map(|value| value.abs())
        .sum::<f32>();
    anyhow::ensure!(
        val_proj_moment_l1 > 0.0 && val_proj_moment_l1.is_finite(),
        "LTM value-alignment gradient did not reach val_proj.weight AdamW moments"
    );

    let source_model = model_dir.join("model.safetensors");
    let (_, source_in_proj) = read_f32_tensor(&source_model, "in_proj.weight")?;
    let (_, source_val_proj) = read_f32_tensor(&source_model, "val_proj.weight")?;
    let (_, source_fast_vals) = read_f32_tensor(&source_model, "ltm.fast_vals")?;
    let trained_package = trained_model
        .parent()
        .context("--trained-model must have a parent package directory")?;
    let native_replay = HierarchosPortableTrainingReplay::new(
        1,
        0,
        serde_json::json!({
            "__kind__": "dict",
            "items": [
                ["native_replay_probe", {"__kind__": "tensor", "name": "state_000000"}],
                ["native_rng_probe", {"__kind__": "tensor", "name": "state_000001"}]
            ]
        }),
        vec![
            HierarchosPortableReplayTensor::f32("state_000000", vec![2], vec![0.125, -0.25]),
            HierarchosPortableReplayTensor::u8("state_000001", vec![4], vec![1, 2, 3, 255]),
        ],
    )?
    .with_training_session(HierarchosTrainingSessionState {
        format: HIERARCHOS_VULKAN_TRAINING_SESSION_FORMAT.to_string(),
        completed_epoch: 1,
        mid_epoch_step: 0,
        optimizer_grouping_version: 2,
        main_lr_scheduler: Some(HierarchosLearningRateScheduleState {
            enabled: true,
            step: Some(1),
            total_steps: Some(8),
            max_lr: Some(case.optimizer.lr as f64),
            min_lr: Some(case.optimizer.lr as f64 * 0.1),
            warmup_steps: Some(1),
            warmup_ratio: Some(0.125),
            resolved_warmup_steps: Some(1),
            base_lrs: vec![case.optimizer.lr as f64],
            last_lrs: vec![case.optimizer.lr as f64 * 0.75],
            step_count: Some(2),
        }),
        ltm_lr_scheduler: Some(HierarchosLearningRateScheduleState {
            enabled: true,
            step: Some(1),
            total_steps: Some(8),
            max_lr: Some(1.0e-3),
            min_lr: Some(1.0e-4),
            warmup_steps: None,
            warmup_ratio: None,
            resolved_warmup_steps: None,
            base_lrs: Vec::new(),
            last_lrs: Vec::new(),
            step_count: None,
        }),
        effective_training_config: serde_json::json!({"training_backend": "vulkan"}),
        skipped_train_batches: 0,
        data_stream_cursor: Some(HierarchosDataStreamCursorState {
            format: HIERARCHOS_VULKAN_DATA_STREAM_CURSOR_FORMAT.to_string(),
            sampler_kind: "epoch-shuffle".to_string(),
            rng_algorithm: HIERARCHOS_VULKAN_PORTABLE_SAMPLER_RNG_ALGORITHM.to_string(),
            seed: 123,
            epoch: 1,
            batch_cursor: 0,
            dataset_size: 1,
            batch_size: 1,
            shuffle: true,
            drop_last: false,
            bucket_size: None,
            preserve_order: false,
        }),
        execution_policy: Some(HierarchosExecutionPolicyState {
            format: HIERARCHOS_VULKAN_EXECUTION_POLICY_FORMAT.to_string(),
            source_backend: "vulkan".to_string(),
            compute_dtype: "float32".to_string(),
            autocast_enabled: false,
            stochastic_rng: HierarchosStochasticRngPolicyState {
                mode: "none".to_string(),
                state_required: false,
                canonical_counter: None,
            },
            loss_scaling: HierarchosLossScalingState {
                mode: "none".to_string(),
                scale: None,
                growth_factor: None,
                backoff_factor: None,
                growth_interval: None,
                growth_tracker: None,
                pending_gradients_scaled: false,
            },
        }),
    })?;
    raw_graph.synchronize_ltm_alignment_controller_metadata()?;
    let exported_manifest = raw_graph.export_training_checkpoint_package_with_replay(
        &model_dir,
        trained_package,
        &native_replay,
    )?;
    let (_, trained_in_proj) = read_f32_tensor(&trained_model, "in_proj.weight")?;
    let (_, trained_val_proj) = read_f32_tensor(&trained_model, "val_proj.weight")?;
    let (_, trained_fast_vals) = read_f32_tensor(&trained_model, "ltm.fast_vals")?;
    let (_, trained_warmup_step) = read_f32_tensor(&trained_model, "memory_gate_warmup_step")?;
    let checkpoint_in_proj_max_abs_delta = max_abs_diff(&source_in_proj, &trained_in_proj)?;
    let checkpoint_val_proj_max_abs_delta = max_abs_diff(&source_val_proj, &trained_val_proj)?;
    let checkpoint_fast_vals_exact = source_fast_vals == trained_fast_vals;
    let checkpoint_warmup_step = *trained_warmup_step
        .first()
        .context("trained memory_gate_warmup_step is empty")?;
    anyhow::ensure!(
        checkpoint_in_proj_max_abs_delta > 0.0,
        "trained checkpoint did not contain the Vulkan in_proj update"
    );
    anyhow::ensure!(
        checkpoint_val_proj_max_abs_delta > 0.0,
        "trained checkpoint did not contain the Vulkan val_proj value-alignment update"
    );
    anyhow::ensure!(
        checkpoint_fast_vals_exact,
        "untrained ltm.fast_vals changed during export"
    );
    anyhow::ensure!(
        checkpoint_warmup_step == 7.0,
        "trained checkpoint warmup step is {checkpoint_warmup_step}, expected 7"
    );

    let optimizer_before_reload = raw_graph.full_model_optimizer_state()?;
    let reloaded_manifest = raw_graph.load_training_checkpoint_package_state(trained_package)?;
    let optimizer_after_reload = raw_graph.full_model_optimizer_state()?;
    let training_session_roundtrip = match (
        exported_manifest.training_session.as_ref(),
        reloaded_manifest.training_session.as_ref(),
    ) {
        (Some(exported), Some(reloaded)) => {
            let lr_close = |left: f64, right: f64| {
                let scale = left.abs().max(right.abs()).max(1.0);
                (left - right).abs() <= 8.0 * f64::EPSILON * scale
            };
            let main_matches = match (
                exported.main_lr_scheduler.as_ref(),
                reloaded.main_lr_scheduler.as_ref(),
            ) {
                (Some(left), Some(right)) => {
                    left.enabled == right.enabled
                        && left.step == right.step
                        && left.total_steps == right.total_steps
                        && left.warmup_steps == right.warmup_steps
                        && left.resolved_warmup_steps == right.resolved_warmup_steps
                        && left.step_count == right.step_count
                        && left
                            .max_lr
                            .zip(right.max_lr)
                            .is_some_and(|(a, b)| lr_close(a, b))
                        && left
                            .min_lr
                            .zip(right.min_lr)
                            .is_some_and(|(a, b)| lr_close(a, b))
                        && left.base_lrs.len() == right.base_lrs.len()
                        && left
                            .base_lrs
                            .iter()
                            .zip(&right.base_lrs)
                            .all(|(a, b)| lr_close(*a, *b))
                        && left.last_lrs.len() == right.last_lrs.len()
                        && left
                            .last_lrs
                            .iter()
                            .zip(&right.last_lrs)
                            .all(|(a, b)| lr_close(*a, *b))
                }
                (None, None) => true,
                _ => false,
            };
            let ltm_matches = match (
                exported.ltm_lr_scheduler.as_ref(),
                reloaded.ltm_lr_scheduler.as_ref(),
            ) {
                (Some(left), Some(right)) => {
                    left.enabled == right.enabled
                        && left.step == right.step
                        && left.total_steps == right.total_steps
                        && left.warmup_steps == right.warmup_steps
                        && left.resolved_warmup_steps == right.resolved_warmup_steps
                        && left.step_count == right.step_count
                        && left
                            .max_lr
                            .zip(right.max_lr)
                            .is_some_and(|(a, b)| lr_close(a, b))
                        && left
                            .min_lr
                            .zip(right.min_lr)
                            .is_some_and(|(a, b)| lr_close(a, b))
                }
                (None, None) => true,
                _ => false,
            };
            exported.format == reloaded.format
                && exported.completed_epoch == reloaded.completed_epoch
                && exported.mid_epoch_step == reloaded.mid_epoch_step
                && exported.optimizer_grouping_version == reloaded.optimizer_grouping_version
                && exported.effective_training_config == reloaded.effective_training_config
                && exported.skipped_train_batches == reloaded.skipped_train_batches
                && main_matches
                && ltm_matches
        }
        (None, None) => true,
        _ => false,
    };
    anyhow::ensure!(
        training_session_roundtrip,
        "portable native training session did not round-trip semantically"
    );
    let training_checkpoint_manifest_roundtrip = exported_manifest.format
        == reloaded_manifest.format
        && exported_manifest.architecture_revision == reloaded_manifest.architecture_revision
        && exported_manifest.optimizer_file == reloaded_manifest.optimizer_file
        && exported_manifest.optimizer_step == reloaded_manifest.optimizer_step
        && exported_manifest.optimizer_tensor_count == reloaded_manifest.optimizer_tensor_count
        && exported_manifest.training_step == reloaded_manifest.training_step
        && exported_manifest.training_precision_policy
            == reloaded_manifest.training_precision_policy
        && exported_manifest.completed_epoch == reloaded_manifest.completed_epoch
        && exported_manifest.mid_epoch_step == reloaded_manifest.mid_epoch_step
        && exported_manifest.portable_replay_file == reloaded_manifest.portable_replay_file
        && exported_manifest.portable_replay_tensor_file
            == reloaded_manifest.portable_replay_tensor_file
        && exported_manifest.ltm_alignment_controller == reloaded_manifest.ltm_alignment_controller;
    anyhow::ensure!(
        training_checkpoint_manifest_roundtrip,
        "portable training checkpoint manifest did not round-trip"
    );
    let optimizer_checkpoint_roundtrip = optimizer_before_reload.step
        == optimizer_after_reload.step
        && optimizer_before_reload.slots.len() == optimizer_after_reload.slots.len()
        && optimizer_before_reload
            .slots
            .iter()
            .zip(&optimizer_after_reload.slots)
            .all(|(before, after)| {
                before.name == after.name
                    && before.step == after.step
                    && before.exp_avg == after.exp_avg
                    && before.exp_avg_sq == after.exp_avg_sq
            });
    anyhow::ensure!(
        optimizer_checkpoint_roundtrip,
        "full-model optimizer checkpoint did not round-trip"
    );
    let graph_summary = raw_graph.summary();

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
            projection_native_fp16_backward_compute_active: graph_summary
                .projection_native_fp16_backward_compute_active,
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
            batch,
            raw_queue_submissions: raw.queue_submissions,
            reference_queue_submissions: reference.queue_submissions,
            raw_loss: raw.output.loss,
            reference_loss: reference.output.loss,
            loss_abs_diff,
            grad_enc_max_abs_diff,
            grad_previous_context_max_abs_diff,
            h_output_max_abs_diff,
            l_output_max_abs_diff,
            raw_tape_tokens: raw_tape_result.tokens,
            raw_tape_queue_submissions: raw_tape_result.queue_submissions,
            raw_tape_optimizer_step: raw_tape_result.full_model_optimizer.step,
            raw_tape_losses_finite,
            raw_tape_rosa_predictions,
            raw_tape_frontend_moment_l1,
            reference_optimizer_tensor_count: reference_state.slots.len(),
            raw_optimizer_tensor_count: raw_state.slots.len(),
            raw_optimizer_names: raw_names,
            frontend_moment_l1,
            val_proj_moment_l1,
            training_step: raw_graph.training_step().unwrap_or_default(),
            checkpoint_warmup_step,
            checkpoint_in_proj_max_abs_delta,
            checkpoint_val_proj_max_abs_delta,
            checkpoint_fast_vals_exact,
            training_checkpoint_manifest_roundtrip,
            optimizer_checkpoint_roundtrip,
        })?
    );
    Ok(())
}
