use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{HierarchosTokenFrontendInput, HierarchosTokenFrontendOp, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    token_ids: Vec<u32>,
    token_residual: Option<Vec<f32>>,
    gated_ltm_values: Vec<f32>,
    grad_enc: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    rows: usize,
    queue_submissions: u32,
    token_features: Vec<f32>,
    enc: Vec<f32>,
    grad_token_features: Vec<f32>,
    grad_gated_ltm_values: Vec<f32>,
    grad_persistent: Vec<f32>,
    grad_in_proj_weight: Vec<f32>,
    grad_in_proj_bias: Vec<f32>,
    grad_lm_head_weight: Vec<f32>,
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
    let result = frontend.forward_backward(
        HierarchosTokenFrontendInput {
            token_ids: &case.token_ids,
            token_residual: case.token_residual.as_deref(),
            gated_ltm_values: &case.gated_ltm_values,
        },
        &case.grad_enc,
    )?;
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: frontend.device_name().to_string(),
            rows: result.rows,
            queue_submissions: result.queue_submissions,
            token_features: result.token_features,
            enc: result.enc,
            grad_token_features: result.grad_token_features,
            grad_gated_ltm_values: result.grad_gated_ltm_values,
            grad_persistent: result.grad_persistent,
            grad_in_proj_weight: result.grad_in_proj_weight,
            grad_in_proj_bias: result.grad_in_proj_bias,
            grad_lm_head_weight: result.grad_lm_head_weight,
        })?
    );
    Ok(())
}
