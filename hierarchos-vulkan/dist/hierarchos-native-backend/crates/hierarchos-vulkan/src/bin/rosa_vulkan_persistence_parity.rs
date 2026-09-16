use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    HierarchosTokenFrontendOp, HierarchosTokenMemoryFrontendInput,
    HierarchosTokenMemoryFrontendLaneInput, VulkanDevice,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    chunks: Vec<Vec<u32>>,
    reset_tokens: Vec<u32>,
    lane_tokens: Vec<Vec<u32>>,
    lane_resets: Vec<Vec<u32>>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    rosa_workgroup_size: u32,
    rosa_kernel_label: String,
    rosa_autotuned: bool,
    predictions: Vec<Vec<i64>>,
    after_reset: Vec<i64>,
    lane_predictions: Vec<Vec<i64>>,
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
    let max_rows = case
        .chunks
        .iter()
        .map(Vec::len)
        .chain(std::iter::once(case.reset_tokens.len()))
        .chain(case.lane_tokens.iter().map(Vec::len))
        .max()
        .unwrap_or(0);
    anyhow::ensure!(max_rows > 0, "case must contain at least one token");

    let device = VulkanDevice::new()?;
    let mut frontend = HierarchosTokenFrontendOp::from_model_package(device, &model_dir, max_rows)?;
    let context_dim = frontend.config().context_dim;
    anyhow::ensure!(
        frontend.config().enforce_rosa_max_context && frontend.config().rosa_max_context > 0,
        "persistence parity requires bounded ROSA"
    );

    let mut predictions = Vec::with_capacity(case.chunks.len());
    for chunk in &case.chunks {
        if chunk.is_empty() {
            predictions.push(Vec::new());
            continue;
        }
        let prev_context = vec![0.0f32; chunk.len() * context_dim];
        predictions.push(
            frontend
                .forward_memory(HierarchosTokenMemoryFrontendInput {
                    token_ids: chunk,
                    prev_context: &prev_context,
                })?
                .rosa_prediction_ids,
        );
    }

    frontend.reset_rosa_state()?;
    let after_reset = if case.reset_tokens.is_empty() {
        Vec::new()
    } else {
        let prev_context = vec![0.0f32; case.reset_tokens.len() * context_dim];
        frontend
            .forward_memory(HierarchosTokenMemoryFrontendInput {
                token_ids: &case.reset_tokens,
                prev_context: &prev_context,
            })?
            .rosa_prediction_ids
    };

    anyhow::ensure!(
        case.lane_tokens.len() == case.lane_resets.len(),
        "lane token/reset step counts differ"
    );
    frontend.reset_rosa_state()?;
    let mut lane_predictions = Vec::with_capacity(case.lane_tokens.len());
    for (tokens, resets) in case.lane_tokens.iter().zip(&case.lane_resets) {
        anyhow::ensure!(
            tokens.len() == resets.len(),
            "lane token/reset widths differ: {} vs {}",
            tokens.len(),
            resets.len()
        );
        if tokens.is_empty() {
            lane_predictions.push(Vec::new());
            continue;
        }
        let prev_context = vec![0.0f32; tokens.len() * context_dim];
        lane_predictions.push(
            frontend
                .forward_memory_lanes(HierarchosTokenMemoryFrontendLaneInput {
                    token_ids: tokens,
                    prev_context: &prev_context,
                    reset_lanes: resets,
                })?
                .rosa_prediction_ids,
        );
    }

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: frontend.device_name().to_string(),
            rosa_workgroup_size: frontend.rosa_workgroup_size(),
            rosa_kernel_label: frontend.rosa_kernel_label().to_string(),
            rosa_autotuned: frontend.rosa_was_autotuned(),
            predictions,
            after_reset,
            lane_predictions,
        })?
    );
    Ok(())
}
