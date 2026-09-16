use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    HierarchosTokenFrontendOp, HierarchosTokenMemoryFrontendInput, VulkanDevice,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    token_ids: Vec<u32>,
    prev_context: Vec<f32>,
    grad_enc: Vec<f32>,
    alignment_target: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    rows: usize,
    queue_submissions: u32,
    rosa_prediction_ids: Vec<i64>,
    raw_token_features: Vec<f32>,
    token_features: Vec<f32>,
    query: Vec<f32>,
    topk_indices: Vec<u32>,
    gated_ltm_values: Vec<f32>,
    enc: Vec<f32>,
    grad_prev_context: Vec<f32>,
    grad_persistent: Vec<f32>,
    grad_lm_head_weight: Vec<f32>,
    grad_rosa_adapter_down_weight: Vec<f32>,
    grad_rosa_adapter_up_weight: Vec<f32>,
    grad_rosa_adapter_bias: Vec<f32>,
    grad_rosa_gate_logit: f32,
    grad_rosa_router_weight: Vec<f32>,
    grad_rosa_router_bias: Vec<f32>,
    grad_qproj_weight: Vec<f32>,
    grad_ltm_keys: Vec<f32>,
    grad_ltm_vals: Vec<f32>,
    grad_ltm_gate_logit: f32,
    grad_ltm_router_weight: Vec<f32>,
    grad_ltm_router_bias: Vec<f32>,
    grad_in_proj_weight: Vec<f32>,
    grad_in_proj_bias: Vec<f32>,
    ltm_value_alignment_row_cost: Vec<f32>,
    grad_val_proj_weight: Vec<f32>,
}

fn main() -> Result<()> {
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
    anyhow::ensure!(!case.token_ids.is_empty(), "case must contain token_ids");

    let device = VulkanDevice::new()?;
    let mut frontend =
        HierarchosTokenFrontendOp::from_model_package(device, &model_dir, case.token_ids.len())?;
    let result = frontend.forward_memory_backward(
        HierarchosTokenMemoryFrontendInput {
            token_ids: &case.token_ids,
            prev_context: &case.prev_context,
        },
        &case.grad_enc,
    )?;
    let alignment = frontend.ltm_value_alignment_backward(&case.alignment_target, 1.0)?;
    let forward = &result.forward;
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: frontend.device_name().to_string(),
            rows: forward.rows,
            queue_submissions: forward.queue_submissions,
            rosa_prediction_ids: forward.rosa_prediction_ids.clone(),
            raw_token_features: forward.raw_token_features.clone(),
            token_features: forward.token_features.clone(),
            query: forward.query.clone(),
            topk_indices: forward.topk_indices.clone(),
            gated_ltm_values: forward.gated_ltm_values.clone(),
            enc: forward.enc.clone(),
            grad_prev_context: result.grad_prev_context,
            grad_persistent: result.grad_persistent,
            grad_lm_head_weight: result.grad_lm_head_weight,
            grad_rosa_adapter_down_weight: result.grad_rosa_adapter_down_weight,
            grad_rosa_adapter_up_weight: result.grad_rosa_adapter_up_weight,
            grad_rosa_adapter_bias: result.grad_rosa_adapter_bias,
            grad_rosa_gate_logit: result.grad_rosa_gate_logit,
            grad_rosa_router_weight: result.grad_rosa_router_weight,
            grad_rosa_router_bias: result.grad_rosa_router_bias,
            grad_qproj_weight: result.grad_qproj_weight,
            grad_ltm_keys: result.grad_ltm_keys,
            grad_ltm_vals: result.grad_ltm_vals,
            grad_ltm_gate_logit: result.grad_ltm_gate_logit,
            grad_ltm_router_weight: result.grad_ltm_router_weight,
            grad_ltm_router_bias: result.grad_ltm_router_bias,
            grad_in_proj_weight: result.grad_in_proj_weight,
            grad_in_proj_bias: result.grad_in_proj_bias,
            ltm_value_alignment_row_cost: alignment.row_cost,
            grad_val_proj_weight: alignment.grad_val_proj_weight,
        })?
    );
    Ok(())
}
