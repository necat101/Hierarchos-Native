use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, HierarchosPortableTrainingReplay,
    HierarchosRawTokenWorkerRefinementLossInput, HierarchosTrainingGraph, VulkanDevice,
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
    lm_head_native_fp16_backward_compute_active: bool,
    optimizer_step_before: u32,
    optimizer_step_after: u32,
    loss: f32,
    output_package: String,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut case_path = None;
    let mut output_package = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--case" => case_path = args.next().map(PathBuf::from),
            "--output-package" => output_package = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let model_dir = model_dir.context("missing --model TRAINING_PACKAGE")?;
    let case_path = case_path.context("missing --case CASE.json")?;
    let output_package = output_package.context("missing --output-package DIR")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    let batch = case.token_ids.len();
    anyhow::ensure!(batch > 0, "resume step requires at least one raw token");

    let device = VulkanDevice::new()?;
    let mut graph = HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device, &model_dir, batch, 1, 1, batch, batch,
    )?;
    let restored = graph.load_training_checkpoint_package_state(&model_dir)?;
    anyhow::ensure!(
        !restored.accumulation_open,
        "trajectory resume probe currently requires a closed optimizer boundary"
    );
    let optimizer_step_before = graph.full_model_optimizer_state()?.step;
    anyhow::ensure!(
        optimizer_step_before == restored.optimizer_step,
        "restored optimizer step disagrees with training manifest"
    );

    let rosa_reset_lanes = vec![1u32; batch];
    let result = graph.train_raw_token_worker_refinement_loss_one_submit(
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
            ltm_value_alignment_grad: 0.0,
            targets: &case.targets,
            supervision_weights: None,
        },
        AdamWHyperParams {
            lr: case.optimizer.lr,
            beta1: case.optimizer.beta1,
            beta2: case.optimizer.beta2,
            eps: case.optimizer.eps,
            weight_decay: case.optimizer.weight_decay,
        },
    )?;
    let optimizer_step_after = graph.full_model_optimizer_state()?.step;
    anyhow::ensure!(
        optimizer_step_after == optimizer_step_before + 1,
        "Vulkan resume step did not advance the outer AdamW step exactly once"
    );

    let completed_epoch = restored.completed_epoch.unwrap_or(0);
    let mut replay = HierarchosPortableTrainingReplay::new(
        completed_epoch,
        0,
        serde_json::json!({
            "__kind__": "dict",
            "items": [
                ["trajectory_backend", "vulkan-return"],
                ["source_optimizer_step", optimizer_step_before]
            ]
        }),
        Vec::new(),
    )?;
    if let Some(mut training_session) = restored.training_session.clone() {
        training_session.completed_epoch = completed_epoch;
        training_session.mid_epoch_step = 0;
        replay = replay.with_training_session(training_session)?;
    }
    graph.synchronize_ltm_alignment_controller_metadata()?;
    graph.export_training_checkpoint_package_with_replay(&model_dir, &output_package, &replay)?;

    let summary = graph.summary();
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: summary.device,
            training_precision_policy: summary.training_precision_policy.label().to_string(),
            lm_head_native_fp16_backward_compute_active: summary
                .lm_head_native_fp16_backward_compute_active,
            optimizer_step_before,
            optimizer_step_after,
            loss: result.output.loss,
            output_package: output_package.display().to_string(),
        })?
    );
    Ok(())
}
