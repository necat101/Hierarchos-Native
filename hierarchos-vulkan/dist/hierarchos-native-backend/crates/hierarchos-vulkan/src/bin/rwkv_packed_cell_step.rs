use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvPackedCellOp, RwkvStateReadoutMode, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    input_dim: usize,
    state_mode: String,
    state_clamp: f32,
    x: Vec<f32>,
    token_features: Vec<f32>,
    packed_state: Vec<f32>,
    grad_output: Vec<f32>,
    grad_packed_new_state: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    state_size: usize,
    output: Vec<f32>,
    packed_new_state: Vec<f32>,
    grad_x: Vec<f32>,
    grad_packed_state: Vec<f32>,
    token_feature_grad: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--cell-prefix" => {
                cell_prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--adapter-prefix" => {
                adapter_prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context(
        "usage: --case CASE.json --model-dir MODEL_DIR --cell-prefix h_rnn --adapter-prefix h_deepembed_adapter",
    )?;
    let model_dir = model_dir.context("packed cell runner requires --model-dir")?;
    let cell_prefix = cell_prefix.context("packed cell runner requires --cell-prefix")?;
    let adapter_prefix = adapter_prefix.context("packed cell runner requires --adapter-prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    let state_mode = match case.state_mode.as_str() {
        "legacy-input-cache" => RwkvStateReadoutMode::LegacyInputCache,
        "explicit-output" => RwkvStateReadoutMode::ExplicitOutput,
        other => anyhow::bail!("unknown packed state mode {other:?}"),
    };

    let device = VulkanDevice::new()?;
    let mut op = RwkvPackedCellOp::from_model_package(
        device,
        model_dir,
        &cell_prefix,
        &adapter_prefix,
        case.head_size,
        case.batch,
        12.0,
        4.0,
        state_mode,
        case.state_clamp,
    )?;
    if op.width() != case.width || op.token_feature_width() != case.input_dim {
        anyhow::bail!(
            "case geometry width/input_dim={}/{} does not match packed cell {}/{}",
            case.width,
            case.input_dim,
            op.width(),
            op.token_feature_width()
        );
    }
    let result = op.forward_backward(
        case.batch,
        &case.x,
        &case.token_features,
        &case.packed_state,
        &case.grad_output,
        &case.grad_packed_new_state,
    )?;
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            width: op.width(),
            state_size: op.state_size(),
            output: result.output,
            packed_new_state: result.packed_new_state,
            grad_x: result.grad_x,
            grad_packed_state: result.grad_packed_state,
            token_feature_grad: result.token_feature_grad,
        })?
    );
    Ok(())
}
