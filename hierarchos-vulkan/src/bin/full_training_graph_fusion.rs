use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, HierarchosRecurrentBranchInput, HierarchosTrainingGraph,
    RwkvParameterSnapshot, RwkvTbpttSchedule, VulkanDevice,
};
use serde::Serialize;

#[derive(Serialize)]
struct Output {
    device: String,
    queue_submissions: u32,
    h_sequence_max_abs_diff: f32,
    l_sequence_max_abs_diff: f32,
    h_parameter_max_abs_diff: f32,
    l_parameter_max_abs_diff: f32,
    lm_head_max_abs_diff: f32,
    out_norm_weight_max_abs_diff: f32,
    out_norm_bias_max_abs_diff: f32,
    loss_abs_diff: f32,
    h_optimizer_step_match: bool,
    l_optimizer_step_match: bool,
    lm_optimizer_step_match: bool,
}

fn main() -> Result<()> {
    let model_dir = parse_model_arg()?;
    let device = VulkanDevice::new()?;
    let mut sequential =
        HierarchosTrainingGraph::from_model_package(device.clone(), &model_dir, 1, 2, 2, 2)?;
    let mut fused =
        HierarchosTrainingGraph::from_model_package(device.clone(), &model_dir, 1, 2, 2, 2)?;

    let summary = sequential.summary();
    if summary.vocab_size < 2 {
        bail!("fusion verifier requires vocabulary size >= 2");
    }
    let h_steps = 2usize;
    let l_steps = 2usize;
    let h_x = deterministic_values(h_steps * summary.h_hidden, 0.017, 3);
    let l_x = deterministic_values(l_steps * summary.l_hidden, 0.013, 7);
    let h_grad = deterministic_values(h_steps * summary.h_hidden, 0.009, 11);
    let l_grad = deterministic_values(l_steps * summary.l_hidden, 0.008, 13);
    let h_state = vec![0.0; summary.h_hidden * sequential.h_recurrent().state_size()];
    let l_state = vec![0.0; summary.l_hidden * sequential.l_recurrent().state_size()];
    let h_tokens = vec![0u32, 1u32];
    let l_tokens = vec![1u32, 0u32];
    let hidden = deterministic_values(2 * summary.context_dim, 0.021, 17);
    let targets = vec![1u32, 0u32];
    let hyper = AdamWHyperParams {
        lr: 3.0e-4,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1.0e-8,
        weight_decay: 0.01,
    };

    let sequential_h = sequential.train_h_begin_shared_lm(
        1,
        h_steps,
        &h_x,
        &h_tokens,
        &h_state,
        &h_grad,
        None,
        RwkvTbpttSchedule::full_bptt(),
        hyper,
    )?;
    let sequential_l = sequential.train_l_accumulate_shared_lm(
        1,
        l_steps,
        &l_x,
        &l_tokens,
        &l_state,
        &l_grad,
        None,
        RwkvTbpttSchedule::full_bptt(),
        hyper,
    )?;
    let sequential_output = sequential.finalize_shared_lm_loss(&hidden, &targets, hyper)?;

    let fused_result = fused.train_recurrent_and_loss_one_submit(
        HierarchosRecurrentBranchInput {
            batch: 1,
            steps: h_steps,
            x_sequence: &h_x,
            token_id_sequence: &h_tokens,
            initial_packed_state: &h_state,
            grad_output_sequence: &h_grad,
            final_packed_state_grad: None,
            schedule: RwkvTbpttSchedule::full_bptt(),
        },
        HierarchosRecurrentBranchInput {
            batch: 1,
            steps: l_steps,
            x_sequence: &l_x,
            token_id_sequence: &l_tokens,
            initial_packed_state: &l_state,
            grad_output_sequence: &l_grad,
            final_packed_state_grad: None,
            schedule: RwkvTbpttSchedule::full_bptt(),
        },
        &hidden,
        &targets,
        hyper,
    )?;

    let sequential_lm = sequential.shared_lm_head().weights()?;
    let fused_lm = fused.shared_lm_head().weights()?;
    let (sequential_norm_weight, sequential_norm_bias) = sequential.out_norm_parameters()?;
    let (fused_norm_weight, fused_norm_bias) = fused.out_norm_parameters()?;

    let output = Output {
        device: summary.device,
        queue_submissions: fused_result.queue_submissions,
        h_sequence_max_abs_diff: sequence_diff(&sequential_h.sequence, &fused_result.h.sequence)?,
        l_sequence_max_abs_diff: sequence_diff(&sequential_l.sequence, &fused_result.l.sequence)?,
        h_parameter_max_abs_diff: snapshot_diff(
            &sequential_h.parameters,
            &fused_result.h.parameters,
        )?,
        l_parameter_max_abs_diff: snapshot_diff(
            &sequential_l.parameters,
            &fused_result.l.parameters,
        )?,
        lm_head_max_abs_diff: max_abs_diff(&sequential_lm, &fused_lm)?,
        out_norm_weight_max_abs_diff: max_abs_diff(&sequential_norm_weight, &fused_norm_weight)?,
        out_norm_bias_max_abs_diff: max_abs_diff(&sequential_norm_bias, &fused_norm_bias)?,
        loss_abs_diff: (sequential_output.loss - fused_result.output.loss).abs(),
        h_optimizer_step_match: sequential_h.optimizer.step == fused_result.h.optimizer.step,
        l_optimizer_step_match: sequential_l.optimizer.step == fused_result.l.optimizer.step,
        lm_optimizer_step_match: sequential_output.step == fused_result.output.step,
    };
    println!("{}", serde_json::to_string(&output)?);
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

fn sequence_diff(
    lhs: &hierarchos_vulkan::RwkvTbpttSequenceResult,
    rhs: &hierarchos_vulkan::RwkvTbpttSequenceResult,
) -> Result<f32> {
    let mut max_diff = 0.0f32;
    for (left, right) in [
        (&lhs.outputs, &rhs.outputs),
        (&lhs.final_packed_state, &rhs.final_packed_state),
        (&lhs.grad_x, &rhs.grad_x),
        (&lhs.token_feature_grad, &rhs.token_feature_grad),
        (
            &lhs.grad_initial_packed_state,
            &rhs.grad_initial_packed_state,
        ),
    ] {
        max_diff = max_diff.max(max_abs_diff(left, right)?);
    }
    Ok(max_diff)
}

fn snapshot_diff(lhs: &[RwkvParameterSnapshot], rhs: &[RwkvParameterSnapshot]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        bail!(
            "parameter snapshot count mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        );
    }
    let mut max_diff = 0.0f32;
    for left in lhs {
        let right = rhs
            .iter()
            .find(|candidate| candidate.name == left.name)
            .with_context(|| format!("missing fused snapshot {:?}", left.name))?;
        max_diff = max_diff.max(max_abs_diff(&left.values, &right.values)?);
    }
    Ok(max_diff)
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
