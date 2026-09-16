use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    read_f32_tensor, AdamWHyperParams, AdamWOptimizerState, HierarchosProjectionCoupledTokenInput,
    HierarchosTrainingGraph, VulkanDevice,
};
use serde::Serialize;

#[derive(Serialize)]
struct Output {
    device: String,
    queue_submissions: u32,
    h_optimizer_step: u32,
    l_optimizer_step: u32,
    projection_optimizer_step: u32,
    projection_optimizer_tensor_count: usize,
    shared_lm_step: u32,
    projection_changed_tensor_count: usize,
    projection_max_abs_delta: f32,
    h_grad_x_max_abs: f32,
    l_grad_x_max_abs: f32,
    l_initial_state_grad_max_abs: f32,
    optimizer_checkpoint_step: u32,
    optimizer_checkpoint_max_abs_diff: f32,
}

fn main() -> Result<()> {
    let model_dir = parse_model_arg()?;
    let device = VulkanDevice::new()?;
    let mut graph =
        HierarchosTrainingGraph::from_model_package(device.clone(), &model_dir, 1, 1, 1, 1)?;
    let summary = graph.summary();
    if summary.vocab_size < 2 {
        bail!("projection-coupling verifier requires vocabulary size >= 2");
    }

    let h_base = deterministic_values(summary.h_hidden, 0.019, 3);
    let l_input = deterministic_values(summary.context_dim * 2, 0.017, 5);
    let h_state = deterministic_values(
        summary.h_hidden * graph.h_recurrent().state_size(),
        0.003,
        7,
    );
    let l_state = deterministic_values(
        summary.l_hidden * graph.l_recurrent().state_size(),
        0.003,
        11,
    );
    let h_context_grad = deterministic_values(summary.context_dim, 0.013, 13);
    let h_halt_grad = vec![0.021];
    let l_drift_grad = deterministic_values(summary.context_dim, 0.011, 17);
    let l_out_grad = deterministic_values(summary.context_dim, 0.009, 19);
    let hyper = AdamWHyperParams {
        lr: 7.0e-4,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1.0e-8,
        // Keep this verifier gradient-pure: a changed projection tensor must
        // have received an actual graph gradient rather than decoupled decay.
        weight_decay: 0.0,
    };

    let result = graph.train_projection_coupled_token_one_submit(
        HierarchosProjectionCoupledTokenInput {
            batch: 1,
            h_base_residual: &h_base,
            l_input_source: &l_input,
            h_token_ids: &[0],
            l_token_ids: &[1],
            h_initial_packed_state: &h_state,
            l_initial_packed_state: &l_state,
            h_to_context_grad: &h_context_grad,
            h_halt_grad: &h_halt_grad,
            context_drift_grad: &l_drift_grad,
            l_to_out_grad: &l_out_grad,
        },
        hyper,
    )?;

    let checkpoint = model_dir.join("model.safetensors");
    let mut changed = 0usize;
    let mut projection_max_abs_delta = 0.0f32;
    for snapshot in &result.projection_parameters {
        let (_, original) = read_f32_tensor(&checkpoint, &snapshot.name)
            .with_context(|| format!("reading original projection {:?}", snapshot.name))?;
        let delta = max_abs_diff(&original, &snapshot.values)?;
        if delta > 0.0 {
            changed += 1;
        }
        projection_max_abs_delta = projection_max_abs_delta.max(delta);
    }

    let optimizer_checkpoint = model_dir.join("projection_optimizer.safetensors");
    graph.save_projection_optimizer_checkpoint(&optimizer_checkpoint)?;
    let saved_state = graph.projection_optimizer_state()?;
    let mut restored = HierarchosTrainingGraph::from_model_package(device, &model_dir, 1, 1, 1, 1)?;
    restored.load_projection_optimizer_checkpoint(&optimizer_checkpoint)?;
    let restored_state = restored.projection_optimizer_state()?;
    let optimizer_checkpoint_max_abs_diff = optimizer_state_diff(&saved_state, &restored_state)?;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: summary.device,
            queue_submissions: result.queue_submissions,
            h_optimizer_step: result.h.optimizer.step,
            l_optimizer_step: result.l.optimizer.step,
            projection_optimizer_step: result.projection_optimizer.step,
            projection_optimizer_tensor_count: result.projection_optimizer.tensor_count,
            shared_lm_step: result.shared_lm_step,
            projection_changed_tensor_count: changed,
            projection_max_abs_delta,
            h_grad_x_max_abs: max_abs(&result.h.sequence.grad_x),
            l_grad_x_max_abs: max_abs(&result.l.sequence.grad_x),
            l_initial_state_grad_max_abs: max_abs(&result.l.sequence.grad_initial_packed_state),
            optimizer_checkpoint_step: restored_state.step,
            optimizer_checkpoint_max_abs_diff,
        })?
    );
    Ok(())
}

fn parse_model_arg() -> Result<PathBuf> {
    let mut args = std::env::args_os().skip(1);
    let mut model = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model = args.next().map(PathBuf::from),
            other => bail!("unknown argument {other:?}"),
        }
    }
    model.context("usage: --model MODEL_DIR")
}

fn deterministic_values(len: usize, scale: f32, phase: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let centered = ((index * 37 + phase * 19) % 101) as f32 - 50.0;
            centered * scale / 50.0
        })
        .collect()
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        bail!("vector length mismatch: {} vs {}", lhs.len(), rhs.len());
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max))
}

fn optimizer_state_diff(lhs: &AdamWOptimizerState, rhs: &AdamWOptimizerState) -> Result<f32> {
    if lhs.step != rhs.step || lhs.slots.len() != rhs.slots.len() {
        bail!(
            "optimizer state identity mismatch: steps {}/{} slots {}/{}",
            lhs.step,
            rhs.step,
            lhs.slots.len(),
            rhs.slots.len()
        );
    }
    let mut max_diff = 0.0f32;
    for (left, right) in lhs.slots.iter().zip(&rhs.slots) {
        if left.name != right.name {
            bail!(
                "optimizer slot name mismatch: {:?} vs {:?}",
                left.name,
                right.name
            );
        }
        max_diff = max_diff.max(max_abs_diff(&left.exp_avg, &right.exp_avg)?);
        max_diff = max_diff.max(max_abs_diff(&left.exp_avg_sq, &right.exp_avg_sq)?);
    }
    Ok(max_diff)
}
