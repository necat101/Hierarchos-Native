use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    read_f32_tensor, AdamWHyperParams, HierarchosLossCoupledTokenInput, HierarchosTrainingGraph,
    VulkanDevice,
};
use serde::Serialize;

#[derive(Serialize)]
struct Output {
    device: String,
    queue_submissions: u32,
    loss: f32,
    h_optimizer_step: u32,
    l_optimizer_step: u32,
    projection_optimizer_step: u32,
    lm_optimizer_step: u32,
    projection_changed_tensor_count: usize,
    l_to_out_max_abs_delta: f32,
    lm_head_max_abs_delta: f32,
    out_norm_max_abs_delta: f32,
    l_grad_x_max_abs: f32,
}

fn main() -> Result<()> {
    let model_dir = parse_model_arg()?;
    let device = VulkanDevice::new()?;
    let mut graph = HierarchosTrainingGraph::from_model_package(device, &model_dir, 1, 1, 1, 1)?;
    let summary = graph.summary();
    if summary.vocab_size < 3 {
        bail!("loss-coupling verifier requires vocabulary size >= 3");
    }

    let checkpoint = model_dir.join("model.safetensors");
    let (_, original_lm) = read_f32_tensor(&checkpoint, "lm_head.weight")?;
    let (_, original_norm_weight) = read_f32_tensor(&checkpoint, "out_norm.weight")?;
    let (_, original_norm_bias) = read_f32_tensor(&checkpoint, "out_norm.bias")?;

    let enc = deterministic_values(summary.context_dim, 0.019, 3);
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
    let hyper = AdamWHyperParams {
        lr: 7.0e-4,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1.0e-8,
        // With decay disabled, a changed l_to_out tensor proves that the
        // cross-entropy backward path reached it through Vulkan buffers.
        weight_decay: 0.0,
    };

    let result = graph.train_loss_coupled_token_one_submit(
        HierarchosLossCoupledTokenInput {
            batch: 1,
            enc: &enc,
            l_input_source: &l_input,
            h_token_ids: &[0],
            l_token_ids: &[1],
            h_initial_packed_state: &h_state,
            l_initial_packed_state: &l_state,
            h_to_context_grad: &h_context_grad,
            h_halt_grad: &h_halt_grad,
            context_drift_grad: &l_drift_grad,
            targets: &[2],
        },
        hyper,
    )?;

    let mut changed = 0usize;
    let mut l_to_out_max_abs_delta = 0.0f32;
    for snapshot in &result.projection_parameters {
        let (_, original) = read_f32_tensor(&checkpoint, &snapshot.name)
            .with_context(|| format!("reading original projection {:?}", snapshot.name))?;
        let delta = max_abs_diff(&original, &snapshot.values)?;
        if delta > 0.0 {
            changed += 1;
        }
        if snapshot.name.starts_with("l_to_out.") {
            l_to_out_max_abs_delta = l_to_out_max_abs_delta.max(delta);
        }
    }

    let trained_lm = graph.shared_lm_head().weights()?;
    let lm_head_max_abs_delta = max_abs_diff(&original_lm, &trained_lm)?;
    let (norm_weight, norm_bias) = graph.out_norm_parameters()?;
    let out_norm_max_abs_delta = max_abs_diff(&original_norm_weight, &norm_weight)?
        .max(max_abs_diff(&original_norm_bias, &norm_bias)?);

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: summary.device,
            queue_submissions: result.queue_submissions,
            loss: result.output.loss,
            h_optimizer_step: result.h.optimizer.step,
            l_optimizer_step: result.l.optimizer.step,
            projection_optimizer_step: result.projection_optimizer.step,
            lm_optimizer_step: result.output.step,
            projection_changed_tensor_count: changed,
            l_to_out_max_abs_delta,
            lm_head_max_abs_delta,
            out_norm_max_abs_delta,
            l_grad_x_max_abs: max_abs(&result.l.sequence.grad_x),
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
