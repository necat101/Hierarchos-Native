use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, AdamWOptimizerState, HierarchosRawTokenSequenceContextInput,
    HierarchosRawTokenTapeMicrobatchInput, HierarchosRawTokenTapeStepInput,
    HierarchosSequenceGradientNormalization, HierarchosTapeMemoryPolicy, HierarchosTokenFrontendOp,
    HierarchosTokenMemoryFrontendLaneInput, HierarchosTokenTapeControlSnapshot,
    HierarchosTokenTapeMicrobatchInput, HierarchosTokenTapeReadbackPolicy,
    HierarchosTokenTapeStepInput, HierarchosTokenTapeTrainResult, HierarchosTokenTapeUpdateMode,
    HierarchosTrainingGraph, RwkvParameterSnapshot, VulkanDevice,
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

#[derive(Serialize)]
struct Output {
    device: String,
    batch: usize,
    tokens: usize,
    raw_queue_submissions: u32,
    reference_queue_submissions: u32,
    loss_max_abs_diff: f32,
    final_state_max_abs_diff: f32,
    initial_adjoint_max_abs_diff: f32,
    control_match: bool,
    common_optimizer_max_abs_diff: f32,
    reference_optimizer_tensor_count: usize,
    raw_optimizer_tensor_count: usize,
    raw_frontend_moment_l1: f32,
    dense_microbatch_loss_max_abs_diff: f32,
    dense_microbatch_state_max_abs_diff: f32,
    dense_microbatch_initial_adjoint_max_abs_diff: f32,
    dense_microbatch_control_match: bool,
    sparse_raw_vs_dense_loss_max_abs_diff: f32,
    sparse_raw_vs_dense_state_max_abs_diff: f32,
    sparse_raw_vs_dense_initial_adjoint_max_abs_diff: f32,
    sparse_raw_vs_dense_control_match: bool,
    sparse_raw_vs_dense_optimizer_max_abs_diff: f32,
    dense_microbatch_queue_submissions: u32,
    sparse_microbatch_queue_submissions: u32,
    exact_tbptt_sparse_vs_dense_loss_max_abs_diff: f32,
    exact_tbptt_sparse_vs_dense_state_max_abs_diff: f32,
    exact_tbptt_sparse_vs_dense_initial_adjoint_max_abs_diff: f32,
    exact_tbptt_sparse_vs_dense_control_match: bool,
    exact_tbptt_sparse_vs_dense_optimizer_max_abs_diff: f32,
    exact_tbptt_controller_last_abs_diff: f32,
    exact_tbptt_controller_window_rows: u64,
    exact_tbptt_controller_closing_microbatch_rows: u64,
    multi_device_reduce_optimizer_max_abs_diff: f32,
    multi_device_reduce_parameter_max_abs_diff: f32,
    multi_device_reduce_controller_last_abs_diff: f32,
    multi_device_replica_optimizer_max_abs_diff: f32,
    multi_device_replica_parameter_max_abs_diff: f32,
    multi_device_shard_gradient_tensor_count: usize,
    multi_device_stream_chunk_count: usize,
    multi_device_stream_value_count: usize,
    multi_device_stream_pipeline_slots: usize,
    multi_device_stream_backend: String,
    multi_device_stream_peak_host_gradient_bytes: usize,
    multi_device_stream_peak_device_gradient_bytes: usize,
    multi_device_stream_peak_host_heap_gradient_bytes: usize,
    multi_device_stream_queue_submissions: usize,
    multi_device_stream_persistent_transport_reused: bool,
    multi_device_second_stream_persistent_transport_reused: bool,
    multi_device_replica_state_stream_backend: String,
    multi_device_replica_state_stream_chunk_count: usize,
    multi_device_replica_state_stream_value_count: usize,
    multi_device_replica_state_stream_pipeline_slots: usize,
    multi_device_replica_state_stream_persistent_transport_reused: bool,
    multi_device_replica_state_second_stream_persistent_transport_reused: bool,
    open_accumulation_checkpoint_roundtripped: bool,
    open_accumulation_checkpoint_pending_gradient_max_abs_diff: f32,
    open_accumulation_checkpoint_optimizer_max_abs_diff: f32,
    open_accumulation_checkpoint_parameter_max_abs_diff: f32,
    exact_tbptt_dense_queue_submissions: u32,
    exact_tbptt_sparse_queue_submissions: u32,
    context_sparse_vs_dense_result_max_abs_diff: f32,
    context_sparse_vs_dense_context_max_abs_diff: f32,
    context_sparse_vs_dense_optimizer_max_abs_diff: f32,
    context_budget_dense_vs_dense_result_max_abs_diff: f32,
    context_budget_dense_vs_dense_context_max_abs_diff: f32,
    context_budget_dense_vs_dense_optimizer_max_abs_diff: f32,
    context_budget_sparse_vs_dense_result_max_abs_diff: f32,
    context_budget_sparse_vs_dense_context_max_abs_diff: f32,
    context_budget_sparse_vs_dense_optimizer_max_abs_diff: f32,
    context_budget_dense_queue_submissions: u32,
    context_budget_sparse_queue_submissions: u32,
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    anyhow::ensure!(
        lhs.len() == rhs.len(),
        "vector length mismatch {} vs {}",
        lhs.len(),
        rhs.len()
    );
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max))
}

fn controls_match(
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

fn common_optimizer_max_diff(
    reference: &AdamWOptimizerState,
    raw: &AdamWOptimizerState,
) -> Result<f32> {
    anyhow::ensure!(
        reference.step == raw.step,
        "optimizer step mismatch {} vs {}",
        reference.step,
        raw.step
    );
    let mut worst = 0.0f32;
    for reference_slot in &reference.slots {
        // The raw frontend adds two tied-embedding gradient paths to the same
        // physical lm_head matrix, so this one shared slot is intentionally not
        // expected to equal the host-enc oracle. Every other legacy slot should.
        if reference_slot.name == "lm_head.weight" || reference_slot.name == "val_proj.weight" {
            continue;
        }
        let raw_slot = raw
            .slots
            .iter()
            .find(|slot| slot.name == reference_slot.name)
            .with_context(|| format!("raw optimizer is missing {:?}", reference_slot.name))?;
        worst = worst.max(max_abs_diff(&reference_slot.exp_avg, &raw_slot.exp_avg)?);
        worst = worst.max(max_abs_diff(
            &reference_slot.exp_avg_sq,
            &raw_slot.exp_avg_sq,
        )?);
    }
    Ok(worst)
}

fn optimizer_max_diff(lhs: &AdamWOptimizerState, rhs: &AdamWOptimizerState) -> Result<f32> {
    anyhow::ensure!(
        lhs.step == rhs.step,
        "optimizer step mismatch {} vs {}",
        lhs.step,
        rhs.step
    );
    anyhow::ensure!(
        lhs.slots.len() == rhs.slots.len(),
        "optimizer slot count mismatch {} vs {}",
        lhs.slots.len(),
        rhs.slots.len()
    );
    let mut worst = 0.0f32;
    for lhs_slot in &lhs.slots {
        let rhs_slot = rhs
            .slots
            .iter()
            .find(|slot| slot.name == lhs_slot.name)
            .with_context(|| format!("optimizer is missing {:?}", lhs_slot.name))?;
        worst = worst.max(max_abs_diff(&lhs_slot.exp_avg, &rhs_slot.exp_avg)?);
        worst = worst.max(max_abs_diff(&lhs_slot.exp_avg_sq, &rhs_slot.exp_avg_sq)?);
    }
    Ok(worst)
}

fn optimizer_worst_slot_diff(
    lhs: &AdamWOptimizerState,
    rhs: &AdamWOptimizerState,
) -> Result<(String, f32)> {
    anyhow::ensure!(lhs.step == rhs.step);
    let mut worst_name = String::new();
    let mut worst = 0.0f32;
    for lhs_slot in &lhs.slots {
        let rhs_slot = rhs
            .slots
            .iter()
            .find(|slot| slot.name == lhs_slot.name)
            .with_context(|| format!("optimizer is missing {:?}", lhs_slot.name))?;
        let diff = max_abs_diff(&lhs_slot.exp_avg, &rhs_slot.exp_avg)?
            .max(max_abs_diff(&lhs_slot.exp_avg_sq, &rhs_slot.exp_avg_sq)?);
        if diff > worst {
            worst = diff;
            worst_name = lhs_slot.name.clone();
        }
    }
    Ok((worst_name, worst))
}

fn parameter_snapshot_max_diff(
    lhs: &[RwkvParameterSnapshot],
    rhs: &[RwkvParameterSnapshot],
) -> Result<f32> {
    anyhow::ensure!(
        lhs.len() == rhs.len(),
        "parameter snapshot count mismatch {} vs {}",
        lhs.len(),
        rhs.len()
    );
    let mut worst = 0.0f32;
    for lhs_tensor in lhs {
        let rhs_tensor = rhs
            .iter()
            .find(|tensor| tensor.name == lhs_tensor.name)
            .with_context(|| format!("parameter snapshot is missing {:?}", lhs_tensor.name))?;
        worst = worst.max(max_abs_diff(&lhs_tensor.values, &rhs_tensor.values)?);
    }
    Ok(worst)
}

fn sequence_result_diffs(
    lhs: &[HierarchosTokenTapeTrainResult],
    rhs: &[HierarchosTokenTapeTrainResult],
) -> Result<(f32, f32, f32, bool)> {
    anyhow::ensure!(
        lhs.len() == rhs.len(),
        "sequence result count mismatch {} vs {}",
        lhs.len(),
        rhs.len()
    );
    let mut loss_worst = 0.0f32;
    let mut state_worst = 0.0f32;
    let mut adjoint_worst = 0.0f32;
    let mut controls_equal = true;
    for (left, right) in lhs.iter().zip(rhs) {
        anyhow::ensure!(left.tokens == right.tokens, "sequence token count mismatch");
        loss_worst = loss_worst.max(max_abs_diff(&left.losses, &right.losses)?);
        state_worst = state_worst.max(max_abs_diff(
            &left.final_h_packed_state,
            &right.final_h_packed_state,
        )?);
        state_worst = state_worst.max(max_abs_diff(
            &left.final_l_packed_state,
            &right.final_l_packed_state,
        )?);
        adjoint_worst = adjoint_worst.max(max_abs_diff(
            &left.grad_initial_h_packed_state,
            &right.grad_initial_h_packed_state,
        )?);
        adjoint_worst = adjoint_worst.max(max_abs_diff(
            &left.grad_initial_l_packed_state,
            &right.grad_initial_l_packed_state,
        )?);
        controls_equal &= controls_match(&left.controls, &right.controls)?;
    }
    Ok((loss_worst, state_worst, adjoint_worst, controls_equal))
}

fn sequence_context_result_max_diff(
    lhs: &[HierarchosTokenTapeTrainResult],
    rhs: &[HierarchosTokenTapeTrainResult],
) -> Result<f32> {
    anyhow::ensure!(
        lhs.len() == rhs.len(),
        "context sequence result count mismatch {} vs {}",
        lhs.len(),
        rhs.len()
    );
    let mut worst = 0.0f32;
    for (index, (left, right)) in lhs.iter().zip(rhs).enumerate() {
        let left_previous = left
            .final_previous_context
            .as_deref()
            .with_context(|| format!("left sequence {index} is missing final previous context"))?;
        let right_previous = right
            .final_previous_context
            .as_deref()
            .with_context(|| format!("right sequence {index} is missing final previous context"))?;
        let left_target = left
            .final_target_context
            .as_deref()
            .with_context(|| format!("left sequence {index} is missing final target context"))?;
        let right_target = right
            .final_target_context
            .as_deref()
            .with_context(|| format!("right sequence {index} is missing final target context"))?;
        worst = worst.max(max_abs_diff(left_previous, right_previous)?);
        worst = worst.max(max_abs_diff(left_target, right_target)?);
    }
    Ok(worst)
}

fn main() -> Result<()> {
    // This parity harness deliberately keeps several complete training graphs
    // alive at once so it can compare dense/sparse/TBPTT/checkpoint paths in a
    // single process. On Windows the default main-thread stack is small enough
    // that modest growth in the graph type can trip it before any Vulkan work
    // begins. Give the harness headroom without changing production graph
    // allocation semantics.
    std::thread::Builder::new()
        .name("hierarchos-raw-token-parity".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("raw-token parity worker thread panicked"))?
}

fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut case_path = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--case" => case_path = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let model_dir = model_dir.context("missing --model MODEL_DIR")?;
    let case_path = case_path.context("missing --case CASE.json")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    anyhow::ensure!(!case.steps.is_empty(), "raw-token tape case has no steps");
    let batch = case.steps[0].token_ids.len();
    anyhow::ensure!(batch > 0, "raw-token tape batch must be positive");
    anyhow::ensure!(
        case.steps.iter().all(|step| step.token_ids.len() == batch),
        "raw-token tape case changes batch width between steps"
    );

    let hyper = AdamWHyperParams {
        lr: case.optimizer.lr,
        beta1: case.optimizer.beta1,
        beta2: case.optimizer.beta2,
        eps: case.optimizer.eps,
        weight_decay: case.optimizer.weight_decay,
    };
    let device = VulkanDevice::new()?;

    // Materialize only the parity oracle's enc rows. The production raw tape
    // below never consumes these values.
    let mut oracle_frontend =
        HierarchosTokenFrontendOp::from_model_package(device.clone(), &model_dir, batch)?;
    oracle_frontend.set_training_step(7);
    let mut oracle_enc = Vec::with_capacity(case.steps.len());
    for step in &case.steps {
        oracle_enc.push(
            oracle_frontend
                .forward_memory_lanes(HierarchosTokenMemoryFrontendLaneInput {
                    token_ids: &step.token_ids,
                    prev_context: &step.previous_context,
                    reset_lanes: &step.rosa_reset_lanes,
                })?
                .enc,
        );
    }

    let legacy_steps = case
        .steps
        .iter()
        .zip(&oracle_enc)
        .map(|(step, enc)| HierarchosTokenTapeStepInput {
            h_steps: 1,
            shadow_steps: 1,
            enc,
            previous_context: &step.previous_context,
            target_context: &step.target_context,
            context_alpha: step.context_alpha,
            h_token_ids: &step.h_token_ids,
            l_token_ids: &step.l_token_ids,
            h_to_context_grad: &step.h_to_context_grad,
            h_depth_grad: &step.h_depth_grad,
            final_drift_grad: &step.final_drift_grad,
            commitment_cost_grad: &step.commitment_cost_grad,
            targets: &step.targets,
            supervision_weights: None,
        })
        .collect::<Vec<_>>();
    let raw_steps = case
        .steps
        .iter()
        .enumerate()
        .map(|(token_index, step)| HierarchosRawTokenTapeStepInput {
            h_steps: 1,
            shadow_steps: 1,
            token_ids: &step.token_ids,
            rosa_reset_lanes: &step.rosa_reset_lanes,
            previous_context: &step.previous_context,
            target_context: &step.target_context,
            context_alpha: step.context_alpha,
            h_token_ids: &step.h_token_ids,
            l_token_ids: &step.l_token_ids,
            h_to_context_grad: &step.h_to_context_grad,
            h_depth_grad: &step.h_depth_grad,
            final_drift_grad: &step.final_drift_grad,
            commitment_cost_grad: &step.commitment_cost_grad,
            ltm_value_alignment_position: token_index as u64,
            ltm_value_alignment_mask: None,
            ltm_value_alignment_grad: 0.0,
            targets: &step.targets,
            supervision_weights: None,
            pytorch_tbptt_token_mask: None,
        })
        .collect::<Vec<_>>();

    // Exercise exact historical-TBPTT auxiliary weighting with deliberately
    // uneven padding. Position 0 keeps every row; later positions retain only
    // row 0. With h_stride=2 and LTM stride=2 this gives different sampled
    // denominators in the two chunks, which catches a global sampled-mean
    // shortcut while still keeping dense and sparse replay directly comparable.
    let tbptt_masks = (0..case.steps.len())
        .map(|token_index| {
            (0..batch)
                .map(|row| {
                    if token_index == 0 || row == 0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tbptt_real_token_count = tbptt_masks
        .iter()
        .flat_map(|mask| mask.iter())
        .filter(|&&value| value != 0.0)
        .count();
    let tbptt_h_depth_grads = case
        .steps
        .iter()
        .zip(&tbptt_masks)
        .map(|(step, mask)| {
            step.h_depth_grad
                .iter()
                .zip(mask)
                .map(|(value, weight)| value * weight)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tbptt_commitment_grads = case
        .steps
        .iter()
        .zip(&tbptt_masks)
        .map(|(step, mask)| {
            step.commitment_cost_grad
                .iter()
                .zip(mask)
                .map(|(value, weight)| value * weight)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let exact_tbptt_steps = case
        .steps
        .iter()
        .enumerate()
        .map(|(token_index, step)| HierarchosRawTokenTapeStepInput {
            h_steps: 1,
            shadow_steps: 1,
            token_ids: &step.token_ids,
            rosa_reset_lanes: &step.rosa_reset_lanes,
            previous_context: &step.previous_context,
            target_context: &step.target_context,
            context_alpha: step.context_alpha,
            h_token_ids: &step.h_token_ids,
            l_token_ids: &step.l_token_ids,
            h_to_context_grad: &step.h_to_context_grad,
            h_depth_grad: &tbptt_h_depth_grads[token_index],
            final_drift_grad: &step.final_drift_grad,
            commitment_cost_grad: &tbptt_commitment_grads[token_index],
            ltm_value_alignment_position: token_index as u64,
            ltm_value_alignment_mask: Some(&tbptt_masks[token_index]),
            ltm_value_alignment_grad: 0.0,
            targets: &step.targets,
            supervision_weights: Some(&tbptt_masks[token_index]),
            pytorch_tbptt_token_mask: Some(&tbptt_masks[token_index]),
        })
        .collect::<Vec<_>>();

    let mut reference_graph = HierarchosTrainingGraph::from_model_package(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
    )?;
    let mut reference_tape = reference_graph.create_token_tape(
        batch,
        case.steps.len(),
        &case.h_initial_packed_state,
        &case.l_initial_packed_state,
    )?;
    let reference = reference_graph.train_token_tape_with_normalization(
        &mut reference_tape,
        &legacy_steps,
        None,
        None,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
    )?;
    let reference_optimizer = reference_graph.full_model_optimizer_state()?;

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
    let mut raw_tape = raw_graph.create_token_tape(
        batch,
        case.steps.len(),
        &case.h_initial_packed_state,
        &case.l_initial_packed_state,
    )?;
    let raw = raw_graph.train_raw_token_tape_with_normalization(
        &mut raw_tape,
        &raw_steps,
        None,
        None,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
    )?;
    let raw_optimizer = raw_graph.full_model_optimizer_state()?;

    let loss_max_abs_diff = max_abs_diff(&reference.losses, &raw.losses)?;
    let final_state_max_abs_diff =
        max_abs_diff(&reference.final_h_packed_state, &raw.final_h_packed_state)?.max(
            max_abs_diff(&reference.final_l_packed_state, &raw.final_l_packed_state)?,
        );
    let initial_adjoint_max_abs_diff = max_abs_diff(
        &reference.grad_initial_h_packed_state,
        &raw.grad_initial_h_packed_state,
    )?
    .max(max_abs_diff(
        &reference.grad_initial_l_packed_state,
        &raw.grad_initial_l_packed_state,
    )?);
    let control_match = controls_match(&reference.controls, &raw.controls)?;
    let common_optimizer_max_abs_diff =
        common_optimizer_max_diff(&reference_optimizer, &raw_optimizer)?;
    let raw_frontend_moment_l1 = raw_optimizer
        .slots
        .iter()
        .filter(|slot| {
            !reference_optimizer
                .slots
                .iter()
                .any(|reference_slot| reference_slot.name == slot.name)
        })
        .flat_map(|slot| slot.exp_avg.iter())
        .map(|value| value.abs())
        .sum::<f32>();

    // Exercise two recurrently distinct sequences in one arena. The first token
    // resets all ROSA lanes, so each positional sequence slot is an independent
    // suffix stream even though the graph owns one physical frontend scratch
    // state. Distinct recurrent initials are important for the exact-TBPTT
    // controller check below: the closing microbatch must not collapse into the
    // same score as the first one by construction.
    let h_initial_variant = case
        .h_initial_packed_state
        .iter()
        .map(|value| value + 0.075)
        .collect::<Vec<_>>();
    let l_initial_variant = case
        .l_initial_packed_state
        .iter()
        .map(|value| value - 0.05)
        .collect::<Vec<_>>();
    let h_initials = [
        case.h_initial_packed_state.as_slice(),
        h_initial_variant.as_slice(),
    ];
    let l_initials = [
        case.l_initial_packed_state.as_slice(),
        l_initial_variant.as_slice(),
    ];
    let legacy_microbatch = [
        HierarchosTokenTapeMicrobatchInput {
            steps: &legacy_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            pytorch_tbptt_real_token_count: None,
        },
        HierarchosTokenTapeMicrobatchInput {
            steps: &legacy_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            pytorch_tbptt_real_token_count: None,
        },
    ];
    let raw_microbatch = [
        HierarchosRawTokenTapeMicrobatchInput {
            steps: &raw_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            sequence_context: None,
            pytorch_tbptt_real_token_count: None,
            pytorch_tbptt_chunk_size: None,
            preweighted_ponder_and_commitment: false,
        },
        HierarchosRawTokenTapeMicrobatchInput {
            steps: &raw_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            sequence_context: None,
            pytorch_tbptt_real_token_count: None,
            pytorch_tbptt_chunk_size: None,
            preweighted_ponder_and_commitment: false,
        },
    ];
    let exact_tbptt_microbatch = [
        HierarchosRawTokenTapeMicrobatchInput {
            steps: &exact_tbptt_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            sequence_context: None,
            pytorch_tbptt_real_token_count: Some(tbptt_real_token_count),
            pytorch_tbptt_chunk_size: Some(2),
            preweighted_ponder_and_commitment: false,
        },
        HierarchosRawTokenTapeMicrobatchInput {
            steps: &exact_tbptt_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            sequence_context: None,
            pytorch_tbptt_real_token_count: Some(tbptt_real_token_count),
            pytorch_tbptt_chunk_size: Some(2),
            preweighted_ponder_and_commitment: false,
        },
    ];
    let initial_previous_context = case.steps[0].previous_context.as_slice();
    let initial_target_context = case.steps[0].target_context.as_slice();
    let initial_previous_context_variant = initial_previous_context
        .iter()
        .map(|value| value + 0.03125)
        .collect::<Vec<_>>();
    let initial_target_context_variant = initial_target_context
        .iter()
        .map(|value| value - 0.0275)
        .collect::<Vec<_>>();
    let context_tbptt_microbatch = [
        HierarchosRawTokenTapeMicrobatchInput {
            steps: &exact_tbptt_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            sequence_context: Some(HierarchosRawTokenSequenceContextInput {
                initial_previous_context,
                initial_target_context,
                z_loss_weight: 1.0e-4,
            }),
            pytorch_tbptt_real_token_count: Some(tbptt_real_token_count),
            pytorch_tbptt_chunk_size: Some(2),
            preweighted_ponder_and_commitment: false,
        },
        HierarchosRawTokenTapeMicrobatchInput {
            steps: &exact_tbptt_steps,
            final_h_packed_state_adjoint: None,
            final_l_packed_state_adjoint: None,
            sequence_context: Some(HierarchosRawTokenSequenceContextInput {
                initial_previous_context: &initial_previous_context_variant,
                initial_target_context: &initial_target_context_variant,
                z_loss_weight: 1.0e-4,
            }),
            pytorch_tbptt_real_token_count: Some(tbptt_real_token_count),
            pytorch_tbptt_chunk_size: Some(2),
            preweighted_ponder_and_commitment: false,
        },
    ];

    let mut dense_reference_graph = HierarchosTrainingGraph::from_model_package(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
    )?;
    let mut dense_reference_arena = dense_reference_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &h_initials,
        &l_initials,
    )?;
    let dense_reference = dense_reference_graph.train_token_tape_microbatch_with_normalization(
        &mut dense_reference_arena,
        &legacy_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
    )?;
    let dense_reference_optimizer = dense_reference_graph.full_model_optimizer_state()?;

    let mut dense_raw_graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    dense_raw_graph.set_training_step(7)?;
    let mut dense_raw_arena = dense_raw_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &h_initials,
        &l_initials,
    )?;
    let dense_raw = dense_raw_graph.train_raw_token_tape_microbatch_with_normalization(
        &mut dense_raw_arena,
        &raw_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
    )?;
    let dense_raw_optimizer = dense_raw_graph.full_model_optimizer_state()?;
    let (
        dense_microbatch_loss_max_abs_diff,
        dense_microbatch_state_max_abs_diff,
        dense_microbatch_initial_adjoint_max_abs_diff,
        dense_microbatch_control_match,
    ) = sequence_result_diffs(&dense_reference.sequences, &dense_raw.sequences)?;
    let dense_microbatch_common_optimizer_max_abs_diff =
        common_optimizer_max_diff(&dense_reference_optimizer, &dense_raw_optimizer)?;

    let mut sparse_raw_graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    sparse_raw_graph.set_training_step(7)?;
    let sparse_raw = sparse_raw_graph.train_raw_token_tape_microbatch_with_sparse_replay(
        batch,
        &h_initials,
        &l_initials,
        &raw_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanByToken,
        2,
    )?;
    let sparse_raw_optimizer = sparse_raw_graph.full_model_optimizer_state()?;
    let (
        sparse_raw_vs_dense_loss_max_abs_diff,
        sparse_raw_vs_dense_state_max_abs_diff,
        sparse_raw_vs_dense_initial_adjoint_max_abs_diff,
        sparse_raw_vs_dense_control_match,
    ) = sequence_result_diffs(&dense_raw.sequences, &sparse_raw.sequences)?;
    let sparse_raw_vs_dense_optimizer_max_abs_diff =
        optimizer_max_diff(&dense_raw_optimizer, &sparse_raw_optimizer)?;

    let mut exact_tbptt_dense_graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            device.clone(),
            &model_dir,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    exact_tbptt_dense_graph.set_training_step(7)?;
    let mut exact_tbptt_dense_arena = exact_tbptt_dense_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &h_initials,
        &l_initials,
    )?;
    let exact_tbptt_dense = exact_tbptt_dense_graph
        .train_raw_token_tape_microbatch_with_normalization(
            &mut exact_tbptt_dense_arena,
            &exact_tbptt_microbatch,
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
        )?;
    let exact_tbptt_dense_optimizer = exact_tbptt_dense_graph.full_model_optimizer_state()?;
    let exact_tbptt_dense_controller =
        exact_tbptt_dense_graph.synchronized_ltm_alignment_controller_state()?;

    let mut exact_tbptt_sparse_graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            device.clone(),
            &model_dir,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    exact_tbptt_sparse_graph.set_training_step(7)?;
    let exact_tbptt_sparse = exact_tbptt_sparse_graph
        .train_raw_token_tape_microbatch_with_sparse_replay(
            batch,
            &h_initials,
            &l_initials,
            &exact_tbptt_microbatch,
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
            2,
        )?;
    let exact_tbptt_sparse_optimizer = exact_tbptt_sparse_graph.full_model_optimizer_state()?;
    let exact_tbptt_sparse_controller =
        exact_tbptt_sparse_graph.synchronized_ltm_alignment_controller_state()?;
    let (
        exact_tbptt_sparse_vs_dense_loss_max_abs_diff,
        exact_tbptt_sparse_vs_dense_state_max_abs_diff,
        exact_tbptt_sparse_vs_dense_initial_adjoint_max_abs_diff,
        exact_tbptt_sparse_vs_dense_control_match,
    ) = sequence_result_diffs(&exact_tbptt_dense.sequences, &exact_tbptt_sparse.sequences)?;
    let exact_tbptt_sparse_vs_dense_optimizer_max_abs_diff =
        optimizer_max_diff(&exact_tbptt_dense_optimizer, &exact_tbptt_sparse_optimizer)?;

    // Manager-context recurrence must remain identical when the same logical
    // PyTorch TBPTT window is executed densely, through sparse replay, or
    // through the production budgeted scheduler with sequence chunking.
    let mut context_dense_graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    context_dense_graph.set_training_step(7)?;
    let mut context_dense_arena = context_dense_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &h_initials,
        &l_initials,
    )?;
    let context_dense = context_dense_graph.train_raw_token_tape_microbatch_with_normalization(
        &mut context_dense_arena,
        &context_tbptt_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
    )?;
    let context_dense_optimizer = context_dense_graph.full_model_optimizer_state()?;
    let context_dense_controller =
        context_dense_graph.synchronized_ltm_alignment_controller_state()?;

    let mut context_sparse_graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    context_sparse_graph.set_training_step(7)?;
    let context_sparse = context_sparse_graph.train_raw_token_tape_microbatch_with_sparse_replay(
        batch,
        &h_initials,
        &l_initials,
        &context_tbptt_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
        2,
    )?;
    let context_sparse_optimizer = context_sparse_graph.full_model_optimizer_state()?;
    let (
        context_sparse_loss_max_abs_diff,
        context_sparse_state_max_abs_diff,
        context_sparse_adjoint_max_abs_diff,
        context_sparse_control_match,
    ) = sequence_result_diffs(&context_dense.sequences, &context_sparse.sequences)?;
    let context_sparse_vs_dense_result_max_abs_diff = context_sparse_loss_max_abs_diff
        .max(context_sparse_state_max_abs_diff)
        .max(context_sparse_adjoint_max_abs_diff);
    let context_sparse_vs_dense_context_max_abs_diff =
        sequence_context_result_max_diff(&context_dense.sequences, &context_sparse.sequences)?;
    let context_sparse_vs_dense_optimizer_max_abs_diff =
        optimizer_max_diff(&context_dense_optimizer, &context_sparse_optimizer)?;

    let parity_memory_policy = HierarchosTapeMemoryPolicy {
        budget_fraction: 1.0,
        reserve_bytes: 0,
    };
    let mut context_budget_dense_graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            device.clone(),
            &model_dir,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    context_budget_dense_graph.set_training_step(7)?;
    let context_budget_dense = context_budget_dense_graph
        .train_raw_token_tape_sequences_with_plan(
            batch,
            &h_initials,
            &l_initials,
            &context_tbptt_microbatch,
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
            1,
            1,
            parity_memory_policy,
        )?;
    let context_budget_dense_optimizer = context_budget_dense_graph.full_model_optimizer_state()?;
    let (
        context_budget_dense_loss_max_abs_diff,
        context_budget_dense_state_max_abs_diff,
        context_budget_dense_adjoint_max_abs_diff,
        context_budget_dense_control_match,
    ) = sequence_result_diffs(&context_dense.sequences, &context_budget_dense.sequences)?;
    let context_budget_dense_vs_dense_result_max_abs_diff = context_budget_dense_loss_max_abs_diff
        .max(context_budget_dense_state_max_abs_diff)
        .max(context_budget_dense_adjoint_max_abs_diff);
    let context_budget_dense_vs_dense_context_max_abs_diff = sequence_context_result_max_diff(
        &context_dense.sequences,
        &context_budget_dense.sequences,
    )?;
    let context_budget_dense_vs_dense_optimizer_max_abs_diff =
        optimizer_max_diff(&context_dense_optimizer, &context_budget_dense_optimizer)?;

    let mut context_budget_sparse_graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            device.clone(),
            &model_dir,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    context_budget_sparse_graph.set_training_step(7)?;
    let context_budget_sparse = context_budget_sparse_graph
        .train_raw_token_tape_sequences_with_plan(
            batch,
            &h_initials,
            &l_initials,
            &context_tbptt_microbatch,
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
            1,
            2,
            parity_memory_policy,
        )?;
    let context_budget_sparse_optimizer =
        context_budget_sparse_graph.full_model_optimizer_state()?;
    let (
        context_budget_sparse_loss_max_abs_diff,
        context_budget_sparse_state_max_abs_diff,
        context_budget_sparse_adjoint_max_abs_diff,
        context_budget_sparse_control_match,
    ) = sequence_result_diffs(&context_dense.sequences, &context_budget_sparse.sequences)?;
    let context_budget_sparse_vs_dense_result_max_abs_diff =
        context_budget_sparse_loss_max_abs_diff
            .max(context_budget_sparse_state_max_abs_diff)
            .max(context_budget_sparse_adjoint_max_abs_diff);
    let context_budget_sparse_vs_dense_context_max_abs_diff = sequence_context_result_max_diff(
        &context_dense.sequences,
        &context_budget_sparse.sequences,
    )?;
    let context_budget_sparse_vs_dense_optimizer_max_abs_diff =
        optimizer_max_diff(&context_dense_optimizer, &context_budget_sparse_optimizer)?;

    // A one-sequence step using the second arena slot is the PyTorch controller
    // oracle for an optimizer-closing microbatch. The two-sequence accumulated
    // step must report this score exactly, while its gradient-window sampled-row
    // count still includes both sequences.
    let mut exact_tbptt_closing_graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            device.clone(),
            &model_dir,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    exact_tbptt_closing_graph.set_training_step(7)?;
    let closing_h_initials = [h_initial_variant.as_slice()];
    let closing_l_initials = [l_initial_variant.as_slice()];
    let mut exact_tbptt_closing_arena = exact_tbptt_closing_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &closing_h_initials,
        &closing_l_initials,
    )?;
    let exact_tbptt_closing_microbatch = [HierarchosRawTokenTapeMicrobatchInput {
        steps: &exact_tbptt_steps,
        final_h_packed_state_adjoint: None,
        final_l_packed_state_adjoint: None,
        sequence_context: None,
        pytorch_tbptt_real_token_count: Some(tbptt_real_token_count),
        pytorch_tbptt_chunk_size: Some(2),
        preweighted_ponder_and_commitment: false,
    }];
    exact_tbptt_closing_graph.train_raw_token_tape_microbatch_with_normalization(
        &mut exact_tbptt_closing_arena,
        &exact_tbptt_closing_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
    )?;
    let exact_tbptt_closing_controller =
        exact_tbptt_closing_graph.synchronized_ltm_alignment_controller_state()?;
    let exact_tbptt_controller_last_abs_diff = match (
        exact_tbptt_dense_controller.last,
        exact_tbptt_sparse_controller.last,
        exact_tbptt_closing_controller.last,
    ) {
        (Some(dense), Some(sparse), Some(closing)) => {
            (dense - closing).abs().max((sparse - closing).abs())
        }
        _ => f32::INFINITY,
    };

    // Simulate synchronous data parallelism with two independent Vulkan graph
    // replicas and two independent VkDevice objects. On drivers exposing opaque
    // external memory/semaphores this exercises the persistent zero-host-copy
    // two-slot route even on a single physical GPU; otherwise the same canonical
    // contract falls back to bounded host-visible staging.
    let mut multi_device_primary = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        &model_dir,
        batch,
        1,
        1,
        batch,
        batch,
    )?;
    let multi_device_secondary_device =
        VulkanDevice::new_with_index(device.physical_device_index())?;
    let mut multi_device_secondary =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            multi_device_secondary_device,
            &model_dir,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    multi_device_primary.set_training_step(7)?;
    multi_device_secondary.set_training_step(7)?;
    multi_device_primary
        .train_raw_token_tape_sequences_budgeted_with_update_mode_and_readback_policy(
            batch,
            &h_initials[..1],
            &l_initials[..1],
            &context_tbptt_microbatch[..1],
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            parity_memory_policy,
            HierarchosTokenTapeReadbackPolicy::Full,
        )?;
    multi_device_secondary
        .train_raw_token_tape_sequences_budgeted_with_update_mode_and_readback_policy(
            batch,
            &h_initials[1..2],
            &l_initials[1..2],
            &context_tbptt_microbatch[1..2],
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            parity_memory_policy,
            HierarchosTokenTapeReadbackPolicy::Full,
        )?;
    let multi_device_gradient_source =
        multi_device_secondary.full_model_pending_gradient_transport_source()?;
    let multi_device_stream = multi_device_primary
        .accumulate_full_model_pending_gradients_streamed_from_source(
            &multi_device_gradient_source,
            4096,
        )?;
    let multi_device_shard_gradient_tensor_count = multi_device_stream.tensor_count;
    multi_device_primary.finish_full_model_accumulation(hyper)?;
    let multi_device_primary_optimizer = multi_device_primary.full_model_optimizer_state()?;
    let multi_device_primary_parameters = multi_device_primary.full_model_parameter_snapshots()?;
    let multi_device_primary_controller =
        multi_device_primary.synchronized_ltm_alignment_controller_state()?;
    let multi_device_reduce_optimizer_max_abs_diff =
        optimizer_max_diff(&context_dense_optimizer, &multi_device_primary_optimizer)?;
    let multi_device_reduce_parameter_max_abs_diff = parameter_snapshot_max_diff(
        &context_dense_graph.full_model_parameter_snapshots()?,
        &multi_device_primary_parameters,
    )?;
    let multi_device_reduce_controller_last_abs_diff = match (
        context_dense_controller.last,
        multi_device_primary_controller.last,
    ) {
        (Some(reference), Some(distributed)) => (reference - distributed).abs(),
        (None, None) => 0.0,
        _ => f32::INFINITY,
    };
    anyhow::ensure!(
        multi_device_primary_controller.last_step_sampled_rows
            == context_dense_controller.last_step_sampled_rows
            && multi_device_primary_controller.last_step_controller_sampled_rows
                == context_dense_controller.last_step_controller_sampled_rows,
        "multi-device LTM sampled-row/controller geometry diverged from dense reference: distributed={}/{} dense={}/{}",
        multi_device_primary_controller.last_step_sampled_rows,
        multi_device_primary_controller.last_step_controller_sampled_rows,
        context_dense_controller.last_step_sampled_rows,
        context_dense_controller.last_step_controller_sampled_rows,
    );

    // Broadcast the completed primary state back onto the secondary replica.
    // Direct-compatible logical devices use the same bounded two-slot Vulkan
    // transport as reduction, but copy parameter + exp_avg + exp_avg_sq planes
    // instead of materializing a model-sized host snapshot. Unsupported pairs
    // retain the canonical portable snapshot fallback.
    multi_device_secondary.discard_full_model_accumulation_after_overflow()?;
    let multi_device_replica_transport_source =
        multi_device_primary.full_model_replica_transport_source()?;
    let multi_device_replica_state_stream = multi_device_secondary
        .stream_full_model_replica_state_from_source(
            &multi_device_replica_transport_source,
            4096,
        )?;
    if multi_device_replica_state_stream.is_none() {
        let replica_state = multi_device_primary.full_model_replica_state()?;
        multi_device_secondary.load_full_model_replica_state(&replica_state)?;
    }
    let multi_device_replica_state_second_stream = if multi_device_replica_state_stream.is_some() {
        multi_device_secondary.stream_full_model_replica_state_from_source(
            &multi_device_replica_transport_source,
            4096,
        )?
    } else {
        None
    };
    let multi_device_replica_optimizer_max_abs_diff = optimizer_max_diff(
        &multi_device_primary_optimizer,
        &multi_device_secondary.full_model_optimizer_state()?,
    )?;
    let multi_device_replica_parameter_max_abs_diff = parameter_snapshot_max_diff(
        &multi_device_primary_parameters,
        &multi_device_secondary.full_model_parameter_snapshots()?,
    )?;
    anyhow::ensure!(
        multi_device_secondary.synchronized_ltm_alignment_controller_state()?
            == multi_device_primary_controller,
        "multi-device replica broadcast did not reproduce the primary LTM controller state"
    );

    // Open one more throwaway window on the same logical-device pair. The
    // first external-memory reduction constructs/imports its two slots; this
    // second reduction must check the same transport back out of the primary
    // graph cache instead of re-exporting memory handles and rebuilding binary
    // semaphore pairs. Host-staged/device-group routes remain valid and simply
    // report no external-cache reuse.
    multi_device_primary
        .train_raw_token_tape_sequences_budgeted_with_update_mode_and_readback_policy(
            batch,
            &h_initials[..1],
            &l_initials[..1],
            &context_tbptt_microbatch[..1],
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            parity_memory_policy,
            HierarchosTokenTapeReadbackPolicy::Full,
        )?;
    multi_device_secondary
        .train_raw_token_tape_sequences_budgeted_with_update_mode_and_readback_policy(
            batch,
            &h_initials[1..2],
            &l_initials[1..2],
            &context_tbptt_microbatch[1..2],
            hyper,
            HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
            parity_memory_policy,
            HierarchosTokenTapeReadbackPolicy::Full,
        )?;
    let multi_device_second_gradient_source =
        multi_device_secondary.full_model_pending_gradient_transport_source()?;
    let multi_device_second_stream = multi_device_primary
        .accumulate_full_model_pending_gradients_streamed_from_source(
            &multi_device_second_gradient_source,
            4096,
        )?;
    if multi_device_stream.backend.label() == "opaque-external-memory" {
        anyhow::ensure!(
            !multi_device_stream.persistent_transport_reused
                && multi_device_second_stream.backend == multi_device_stream.backend
                && multi_device_second_stream.persistent_transport_reused,
            "opaque external gradient transport was not reused across windows: first={multi_device_stream:?} second={multi_device_second_stream:?}"
        );
    }
    multi_device_primary.discard_full_model_accumulation_after_overflow()?;
    multi_device_secondary.discard_full_model_accumulation_after_overflow()?;

    // Checkpoint after the first microbatch of a live optimizer window, restore
    // into a fresh graph, then close both windows with the same second
    // microbatch. Canonical gradient serialization must make the two paths
    // bit-identical even though lm_head.weight uses the shared tied buffer.
    let mut checkpoint_source_graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            device.clone(),
            &model_dir,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    checkpoint_source_graph.set_training_step(7)?;
    let mut checkpoint_begin_arena = checkpoint_source_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &closing_h_initials,
        &closing_l_initials,
    )?;
    checkpoint_source_graph.train_raw_token_tape_microbatch_with_update_mode(
        &mut checkpoint_begin_arena,
        &exact_tbptt_closing_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
        None,
    )?;
    let checkpoint_output = model_dir
        .parent()
        .context("model package has no parent for open-checkpoint parity test")?
        .join("open-accumulation-checkpoint");
    let checkpoint_manifest = checkpoint_source_graph
        .export_training_checkpoint_package(&model_dir, &checkpoint_output)?;
    anyhow::ensure!(
        checkpoint_manifest.accumulation_open,
        "mid-window checkpoint manifest did not preserve accumulation_open"
    );
    anyhow::ensure!(
        checkpoint_manifest.gradient_file.is_some()
            && checkpoint_manifest.gradient_tensor_count > 0,
        "mid-window checkpoint manifest did not publish pending gradients"
    );
    anyhow::ensure!(
        checkpoint_manifest.lm_head_gradient_topology.as_deref() == Some("shared-tied"),
        "mid-window checkpoint did not record the tied lm_head gradient topology"
    );

    let mut checkpoint_resumed_graph =
        HierarchosTrainingGraph::from_model_package_with_token_frontend(
            device,
            &checkpoint_output,
            batch,
            1,
            1,
            batch,
            batch,
        )?;
    let restored_manifest =
        checkpoint_resumed_graph.load_training_checkpoint_package_state(&checkpoint_output)?;
    anyhow::ensure!(restored_manifest.accumulation_open);
    let open_accumulation_checkpoint_pending_gradient_max_abs_diff = parameter_snapshot_max_diff(
        &checkpoint_source_graph.full_model_pending_gradient_snapshots()?,
        &checkpoint_resumed_graph.full_model_pending_gradient_snapshots()?,
    )?;
    // Open PyTorch-TBPTT checkpoints serialize val_proj.weight in canonical
    // objective units and divide its LTM weight back out on Vulkan restore.
    // That f32 multiply/divide round-trip is intentionally portable but cannot
    // be bit-exact for arbitrary values, so keep the resume oracle at a tight
    // one-ULP-scale numerical tolerance instead of requiring literal equality.
    let checkpoint_float_roundtrip_tolerance = 1.0e-7f32;
    anyhow::ensure!(
        open_accumulation_checkpoint_pending_gradient_max_abs_diff
            <= checkpoint_float_roundtrip_tolerance,
        "mid-window pending gradients exceeded portable f32 round-trip tolerance: {open_accumulation_checkpoint_pending_gradient_max_abs_diff}"
    );

    let mut source_finish_arena = checkpoint_source_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &closing_h_initials,
        &closing_l_initials,
    )?;
    let mut resumed_finish_arena = checkpoint_resumed_graph.create_token_tape_arena(
        batch,
        case.steps.len(),
        &closing_h_initials,
        &closing_l_initials,
    )?;
    checkpoint_source_graph.train_raw_token_tape_microbatch_with_update_mode(
        &mut source_finish_arena,
        &exact_tbptt_closing_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
        HierarchosTokenTapeUpdateMode::FinishAccumulation,
        None,
    )?;
    checkpoint_resumed_graph.train_raw_token_tape_microbatch_with_update_mode(
        &mut resumed_finish_arena,
        &exact_tbptt_closing_microbatch,
        hyper,
        HierarchosSequenceGradientNormalization::MeanBySupervisionWeight,
        HierarchosTokenTapeUpdateMode::FinishAccumulation,
        None,
    )?;
    let checkpoint_source_optimizer = checkpoint_source_graph.full_model_optimizer_state()?;
    let checkpoint_resumed_optimizer = checkpoint_resumed_graph.full_model_optimizer_state()?;
    let open_accumulation_checkpoint_optimizer_max_abs_diff =
        optimizer_max_diff(&checkpoint_source_optimizer, &checkpoint_resumed_optimizer)?;
    let (open_accumulation_checkpoint_optimizer_worst_slot, _) =
        optimizer_worst_slot_diff(&checkpoint_source_optimizer, &checkpoint_resumed_optimizer)?;
    let open_accumulation_checkpoint_parameter_max_abs_diff = parameter_snapshot_max_diff(
        &checkpoint_source_graph.full_model_parameter_snapshots()?,
        &checkpoint_resumed_graph.full_model_parameter_snapshots()?,
    )?;
    anyhow::ensure!(
        open_accumulation_checkpoint_optimizer_max_abs_diff <= checkpoint_float_roundtrip_tolerance,
        "mid-window checkpoint optimizer resume exceeded portable f32 round-trip tolerance: {open_accumulation_checkpoint_optimizer_max_abs_diff} in {open_accumulation_checkpoint_optimizer_worst_slot}"
    );
    anyhow::ensure!(
        open_accumulation_checkpoint_parameter_max_abs_diff <= checkpoint_float_roundtrip_tolerance,
        "mid-window checkpoint parameter resume exceeded portable f32 round-trip tolerance: {open_accumulation_checkpoint_parameter_max_abs_diff}"
    );
    let open_accumulation_checkpoint_roundtripped = true;

    anyhow::ensure!(reference.queue_submissions == 1 && raw.queue_submissions == 1);
    anyhow::ensure!(
        loss_max_abs_diff <= 2.0e-5,
        "loss drift {loss_max_abs_diff}"
    );
    anyhow::ensure!(
        final_state_max_abs_diff <= 3.0e-5,
        "final recurrent-state drift {final_state_max_abs_diff}"
    );
    anyhow::ensure!(
        initial_adjoint_max_abs_diff <= 3.0e-5,
        "initial recurrent-adjoint drift {initial_adjoint_max_abs_diff}"
    );
    anyhow::ensure!(control_match, "raw-token tape control decisions diverged");
    anyhow::ensure!(
        common_optimizer_max_abs_diff <= 3.0e-5,
        "legacy optimizer-slot drift {common_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        raw_optimizer.slots.len() == reference_optimizer.slots.len() + 16,
        "raw tape optimizer registry is {} -> {}, expected +16 frontend slots",
        reference_optimizer.slots.len(),
        raw_optimizer.slots.len()
    );
    anyhow::ensure!(
        raw_frontend_moment_l1.is_finite() && raw_frontend_moment_l1 > 0.0,
        "raw tape frontend optimizer moments stayed zero"
    );
    anyhow::ensure!(
        dense_reference.queue_submissions == 1 && dense_raw.queue_submissions == 1,
        "dense two-sequence microbatch must submit exactly once"
    );
    anyhow::ensure!(
        dense_reference.total_tokens == case.steps.len() * 2
            && dense_raw.total_tokens == case.steps.len() * 2,
        "dense two-sequence microbatch token accounting drifted"
    );
    anyhow::ensure!(
        dense_microbatch_loss_max_abs_diff <= 2.0e-5,
        "dense raw/legacy microbatch loss drift {dense_microbatch_loss_max_abs_diff}"
    );
    anyhow::ensure!(
        dense_microbatch_state_max_abs_diff <= 3.0e-5,
        "dense raw/legacy microbatch recurrent-state drift {dense_microbatch_state_max_abs_diff}"
    );
    anyhow::ensure!(
        dense_microbatch_initial_adjoint_max_abs_diff <= 3.0e-5,
        "dense raw/legacy microbatch recurrent-adjoint drift {dense_microbatch_initial_adjoint_max_abs_diff}"
    );
    anyhow::ensure!(
        dense_microbatch_control_match,
        "dense raw-token microbatch control decisions diverged"
    );
    anyhow::ensure!(
        dense_microbatch_common_optimizer_max_abs_diff <= 3.0e-5,
        "dense raw/legacy microbatch optimizer drift {dense_microbatch_common_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        sparse_raw.queue_submissions == 1 && sparse_raw.total_tokens == case.steps.len() * 2,
        "raw sparse two-sequence microbatch submission/token accounting drifted"
    );
    anyhow::ensure!(
        sparse_raw_vs_dense_loss_max_abs_diff <= 2.0e-5,
        "raw sparse/dense loss drift {sparse_raw_vs_dense_loss_max_abs_diff}"
    );
    anyhow::ensure!(
        sparse_raw_vs_dense_state_max_abs_diff <= 3.0e-5,
        "raw sparse/dense recurrent-state drift {sparse_raw_vs_dense_state_max_abs_diff}"
    );
    anyhow::ensure!(
        sparse_raw_vs_dense_initial_adjoint_max_abs_diff <= 3.0e-5,
        "raw sparse/dense recurrent-adjoint drift {sparse_raw_vs_dense_initial_adjoint_max_abs_diff}"
    );
    anyhow::ensure!(
        sparse_raw_vs_dense_control_match,
        "raw sparse/dense control decisions diverged"
    );
    anyhow::ensure!(
        sparse_raw_vs_dense_optimizer_max_abs_diff <= 3.0e-5,
        "raw sparse/dense optimizer drift {sparse_raw_vs_dense_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        exact_tbptt_dense.queue_submissions == 1 && exact_tbptt_sparse.queue_submissions == 1,
        "exact PyTorch-TBPTT dense/sparse paths must each submit exactly once"
    );
    anyhow::ensure!(
        exact_tbptt_sparse_vs_dense_loss_max_abs_diff <= 2.0e-5,
        "exact PyTorch-TBPTT sparse/dense loss drift {exact_tbptt_sparse_vs_dense_loss_max_abs_diff}"
    );
    anyhow::ensure!(
        exact_tbptt_sparse_vs_dense_state_max_abs_diff <= 3.0e-5,
        "exact PyTorch-TBPTT sparse/dense recurrent-state drift {exact_tbptt_sparse_vs_dense_state_max_abs_diff}"
    );
    anyhow::ensure!(
        exact_tbptt_sparse_vs_dense_initial_adjoint_max_abs_diff <= 3.0e-5,
        "exact PyTorch-TBPTT sparse/dense recurrent-adjoint drift {exact_tbptt_sparse_vs_dense_initial_adjoint_max_abs_diff}"
    );
    anyhow::ensure!(
        exact_tbptt_sparse_vs_dense_control_match,
        "exact PyTorch-TBPTT sparse/dense control decisions diverged"
    );
    anyhow::ensure!(
        exact_tbptt_sparse_vs_dense_optimizer_max_abs_diff <= 3.0e-5,
        "exact PyTorch-TBPTT sparse/dense optimizer drift {exact_tbptt_sparse_vs_dense_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        exact_tbptt_controller_last_abs_diff <= 3.0e-5,
        "exact PyTorch-TBPTT closing-microbatch controller drift {exact_tbptt_controller_last_abs_diff}"
    );
    anyhow::ensure!(
        exact_tbptt_dense_controller.last_step_sampled_rows
            == exact_tbptt_sparse_controller.last_step_sampled_rows
            && exact_tbptt_dense_controller.last_step_controller_sampled_rows
                == exact_tbptt_sparse_controller.last_step_controller_sampled_rows,
        "exact PyTorch-TBPTT dense/sparse controller row accounting diverged"
    );
    anyhow::ensure!(
        exact_tbptt_dense_controller.last_step_sampled_rows
            == exact_tbptt_closing_controller.last_step_sampled_rows * 2,
        "exact PyTorch-TBPTT gradient window did not retain both microbatches"
    );
    anyhow::ensure!(
        exact_tbptt_dense_controller.last_step_controller_sampled_rows
            == exact_tbptt_closing_controller.last_step_controller_sampled_rows,
        "exact PyTorch-TBPTT controller did not isolate the optimizer-closing microbatch"
    );
    anyhow::ensure!(
        multi_device_reduce_optimizer_max_abs_diff <= 3.0e-5,
        "multi-device Vulkan gradient reduction optimizer drift {multi_device_reduce_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        multi_device_reduce_parameter_max_abs_diff <= 3.0e-5,
        "multi-device Vulkan gradient reduction parameter drift {multi_device_reduce_parameter_max_abs_diff}"
    );
    anyhow::ensure!(
        multi_device_reduce_controller_last_abs_diff <= 3.0e-5,
        "multi-device Vulkan LTM controller drift {multi_device_reduce_controller_last_abs_diff}"
    );
    anyhow::ensure!(
        multi_device_replica_optimizer_max_abs_diff == 0.0
            && multi_device_replica_parameter_max_abs_diff == 0.0,
        "multi-device replica broadcast was not bit-exact: optimizer={multi_device_replica_optimizer_max_abs_diff} parameters={multi_device_replica_parameter_max_abs_diff}"
    );
    if let Some(stream) = multi_device_replica_state_stream {
        anyhow::ensure!(
            stream.tensor_count == multi_device_shard_gradient_tensor_count
                && stream.chunk_count >= stream.tensor_count * 3
                && stream.value_count > stream.max_chunk_values
                && stream.pipeline_slots == 2
                && stream.peak_host_state_bytes == 0
                && stream.peak_device_state_bytes
                    <= stream.pipeline_slots
                        * 2
                        * 4096
                        * std::mem::size_of::<f32>()
                && stream.queue_submissions == stream.chunk_count * 2 + 1,
            "multi-device replica broadcast did not exercise bounded direct Vulkan state streaming: {stream:?}"
        );
        let second = multi_device_replica_state_second_stream.context(
            "direct replica-state transport disappeared on an immediate repeated broadcast",
        )?;
        anyhow::ensure!(
            second.backend == stream.backend,
            "replica-state transport backend changed across immediate broadcasts: first={stream:?} second={second:?}"
        );
        if stream.backend.label() == "opaque-external-memory" {
            anyhow::ensure!(
                second.persistent_transport_reused,
                "opaque external replica-state transport was not reused across broadcasts: first={stream:?} second={second:?}"
            );
        }
    }
    anyhow::ensure!(
        multi_device_stream.chunk_count >= multi_device_stream.tensor_count
            && multi_device_stream.value_count > multi_device_stream.max_chunk_values
            && multi_device_stream.pipeline_slots == 2
            && multi_device_stream.peak_host_gradient_bytes
                <= multi_device_stream.pipeline_slots
                    * 2
                    * 4096
                    * std::mem::size_of::<f32>()
            && multi_device_stream.peak_device_gradient_bytes
                <= multi_device_stream.pipeline_slots
                    * 2
                    * 4096
                    * std::mem::size_of::<f32>()
            && multi_device_stream.peak_host_heap_gradient_bytes == 0
            && multi_device_stream.queue_submissions == multi_device_stream.chunk_count * 2 + 1,
        "multi-device reduction did not exercise bounded streamed transport: {multi_device_stream:?}"
    );
    anyhow::ensure!(
        context_sparse_control_match,
        "manager-context sparse replay changed hard-control decisions"
    );
    anyhow::ensure!(
        context_sparse_vs_dense_result_max_abs_diff <= 3.0e-5,
        "manager-context sparse/dense result drift {context_sparse_vs_dense_result_max_abs_diff}"
    );
    anyhow::ensure!(
        context_sparse_vs_dense_context_max_abs_diff <= 3.0e-5,
        "manager-context sparse/dense carrier drift {context_sparse_vs_dense_context_max_abs_diff}"
    );
    anyhow::ensure!(
        context_sparse_vs_dense_optimizer_max_abs_diff <= 3.0e-5,
        "manager-context sparse/dense optimizer drift {context_sparse_vs_dense_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        context_budget_dense_control_match,
        "budgeted dense manager-context execution changed hard-control decisions"
    );
    anyhow::ensure!(
        context_budget_dense_vs_dense_result_max_abs_diff <= 3.0e-5,
        "budgeted dense manager-context result drift {context_budget_dense_vs_dense_result_max_abs_diff}"
    );
    anyhow::ensure!(
        context_budget_dense_vs_dense_context_max_abs_diff <= 3.0e-5,
        "budgeted dense manager-context carrier drift {context_budget_dense_vs_dense_context_max_abs_diff}"
    );
    anyhow::ensure!(
        context_budget_dense_vs_dense_optimizer_max_abs_diff <= 3.0e-5,
        "budgeted dense manager-context optimizer drift {context_budget_dense_vs_dense_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        context_budget_sparse_control_match,
        "budgeted sparse manager-context execution changed hard-control decisions"
    );
    anyhow::ensure!(
        context_budget_sparse_vs_dense_result_max_abs_diff <= 3.0e-5,
        "budgeted sparse manager-context result drift {context_budget_sparse_vs_dense_result_max_abs_diff}"
    );
    anyhow::ensure!(
        context_budget_sparse_vs_dense_context_max_abs_diff <= 3.0e-5,
        "budgeted sparse manager-context carrier drift {context_budget_sparse_vs_dense_context_max_abs_diff}"
    );
    anyhow::ensure!(
        context_budget_sparse_vs_dense_optimizer_max_abs_diff <= 3.0e-5,
        "budgeted sparse manager-context optimizer drift {context_budget_sparse_vs_dense_optimizer_max_abs_diff}"
    );
    anyhow::ensure!(
        context_budget_dense.queue_submissions == 2
            && context_budget_dense.plan.sequence_microbatch_size == 1
            && context_budget_dense.plan.state_checkpoint_stride == 1,
        "budgeted dense manager-context parity did not exercise two sequence chunks"
    );
    anyhow::ensure!(
        context_budget_sparse.queue_submissions == 2
            && context_budget_sparse.plan.sequence_microbatch_size == 1
            && context_budget_sparse.plan.state_checkpoint_stride == 2,
        "budgeted sparse manager-context parity did not exercise sparse replay across two sequence chunks"
    );

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: raw_graph.summary().device,
            batch,
            tokens: case.steps.len(),
            raw_queue_submissions: raw.queue_submissions,
            reference_queue_submissions: reference.queue_submissions,
            loss_max_abs_diff,
            final_state_max_abs_diff,
            initial_adjoint_max_abs_diff,
            control_match,
            common_optimizer_max_abs_diff,
            reference_optimizer_tensor_count: reference_optimizer.slots.len(),
            raw_optimizer_tensor_count: raw_optimizer.slots.len(),
            raw_frontend_moment_l1,
            dense_microbatch_loss_max_abs_diff,
            dense_microbatch_state_max_abs_diff,
            dense_microbatch_initial_adjoint_max_abs_diff,
            dense_microbatch_control_match,
            sparse_raw_vs_dense_loss_max_abs_diff,
            sparse_raw_vs_dense_state_max_abs_diff,
            sparse_raw_vs_dense_initial_adjoint_max_abs_diff,
            sparse_raw_vs_dense_control_match,
            sparse_raw_vs_dense_optimizer_max_abs_diff,
            dense_microbatch_queue_submissions: dense_raw.queue_submissions,
            sparse_microbatch_queue_submissions: sparse_raw.queue_submissions,
            exact_tbptt_sparse_vs_dense_loss_max_abs_diff,
            exact_tbptt_sparse_vs_dense_state_max_abs_diff,
            exact_tbptt_sparse_vs_dense_initial_adjoint_max_abs_diff,
            exact_tbptt_sparse_vs_dense_control_match,
            exact_tbptt_sparse_vs_dense_optimizer_max_abs_diff,
            exact_tbptt_controller_last_abs_diff,
            exact_tbptt_controller_window_rows: exact_tbptt_dense_controller.last_step_sampled_rows,
            exact_tbptt_controller_closing_microbatch_rows: exact_tbptt_dense_controller
                .last_step_controller_sampled_rows,
            multi_device_reduce_optimizer_max_abs_diff,
            multi_device_reduce_parameter_max_abs_diff,
            multi_device_reduce_controller_last_abs_diff,
            multi_device_replica_optimizer_max_abs_diff,
            multi_device_replica_parameter_max_abs_diff,
            multi_device_shard_gradient_tensor_count,
            multi_device_stream_chunk_count: multi_device_stream.chunk_count,
            multi_device_stream_value_count: multi_device_stream.value_count,
            multi_device_stream_pipeline_slots: multi_device_stream.pipeline_slots,
            multi_device_stream_backend: multi_device_stream.backend.label().to_string(),
            multi_device_stream_peak_host_gradient_bytes: multi_device_stream
                .peak_host_gradient_bytes,
            multi_device_stream_peak_device_gradient_bytes: multi_device_stream
                .peak_device_gradient_bytes,
            multi_device_stream_peak_host_heap_gradient_bytes: multi_device_stream
                .peak_host_heap_gradient_bytes,
            multi_device_stream_queue_submissions: multi_device_stream.queue_submissions,
            multi_device_stream_persistent_transport_reused: multi_device_stream
                .persistent_transport_reused,
            multi_device_second_stream_persistent_transport_reused: multi_device_second_stream
                .persistent_transport_reused,
            multi_device_replica_state_stream_backend: multi_device_replica_state_stream
                .map(|stream| stream.backend.label().to_string())
                .unwrap_or_else(|| "portable-host-snapshot".to_string()),
            multi_device_replica_state_stream_chunk_count: multi_device_replica_state_stream
                .map(|stream| stream.chunk_count)
                .unwrap_or(0),
            multi_device_replica_state_stream_value_count: multi_device_replica_state_stream
                .map(|stream| stream.value_count)
                .unwrap_or(0),
            multi_device_replica_state_stream_pipeline_slots: multi_device_replica_state_stream
                .map(|stream| stream.pipeline_slots)
                .unwrap_or(0),
            multi_device_replica_state_stream_persistent_transport_reused:
                multi_device_replica_state_stream
                    .is_some_and(|stream| stream.persistent_transport_reused),
            multi_device_replica_state_second_stream_persistent_transport_reused:
                multi_device_replica_state_second_stream
                    .is_some_and(|stream| stream.persistent_transport_reused),
            open_accumulation_checkpoint_roundtripped,
            open_accumulation_checkpoint_pending_gradient_max_abs_diff,
            open_accumulation_checkpoint_optimizer_max_abs_diff,
            open_accumulation_checkpoint_parameter_max_abs_diff,
            exact_tbptt_dense_queue_submissions: exact_tbptt_dense.queue_submissions,
            exact_tbptt_sparse_queue_submissions: exact_tbptt_sparse.queue_submissions,
            context_sparse_vs_dense_result_max_abs_diff,
            context_sparse_vs_dense_context_max_abs_diff,
            context_sparse_vs_dense_optimizer_max_abs_diff,
            context_budget_dense_vs_dense_result_max_abs_diff,
            context_budget_dense_vs_dense_context_max_abs_diff,
            context_budget_dense_vs_dense_optimizer_max_abs_diff,
            context_budget_sparse_vs_dense_result_max_abs_diff,
            context_budget_sparse_vs_dense_context_max_abs_diff,
            context_budget_sparse_vs_dense_optimizer_max_abs_diff,
            context_budget_dense_queue_submissions: context_budget_dense.queue_submissions,
            context_budget_sparse_queue_submissions: context_budget_sparse.queue_submissions,
        })?
    );
    Ok(())
}
