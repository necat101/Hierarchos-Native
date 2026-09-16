use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvCellSliceOp, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    input_dim: usize,
    x: Vec<f32>,
    previous_tm: Vec<f32>,
    previous_cm: Vec<f32>,
    matrix_state: Vec<f32>,
    token_features: Vec<f32>,
    grad_matrix_state_out: Vec<f32>,
    grad_output: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    head_size: usize,
    output: Vec<f32>,
    new_matrix_state: Vec<f32>,
    grad_x: Vec<f32>,
    grad_matrix_state: Vec<f32>,
    grad_previous_tm: Vec<f32>,
    grad_previous_cm: Vec<f32>,
    token_feature_grad: Vec<f32>,
    grad_ln1_weight: Vec<f32>,
    grad_ln1_bias: Vec<f32>,
    grad_channel_mix_k: Vec<f32>,
    grad_channel_key_weight: Vec<f32>,
    grad_channel_value_weight: Vec<f32>,
    grad_ln2_weight: Vec<f32>,
    grad_ln2_bias: Vec<f32>,
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
    let model_dir = model_dir.context("cell slice runner requires --model-dir")?;
    let cell_prefix = cell_prefix.context("cell slice runner requires --cell-prefix")?;
    let adapter_prefix = adapter_prefix.context("cell slice runner requires --adapter-prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;

    let device = VulkanDevice::new()?;
    let mut op = RwkvCellSliceOp::from_model_package(
        device,
        model_dir,
        &cell_prefix,
        &adapter_prefix,
        case.head_size,
        case.batch,
        12.0,
        4.0,
    )?;
    if op.width() != case.width || case.input_dim == 0 {
        anyhow::bail!(
            "case geometry width/input_dim={}/{} does not match loaded cell width {}",
            case.width,
            case.input_dim,
            op.width()
        );
    }
    let result = op.forward_backward(
        case.batch,
        &case.x,
        &case.previous_tm,
        &case.previous_cm,
        &case.matrix_state,
        &case.token_features,
        &case.grad_matrix_state_out,
        &case.grad_output,
    )?;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            width: op.width(),
            head_size: op.head_size(),
            output: result.output,
            new_matrix_state: result.new_matrix_state,
            grad_x: result.grad_x,
            grad_matrix_state: result.grad_matrix_state,
            grad_previous_tm: result.grad_previous_tm,
            grad_previous_cm: result.grad_previous_cm,
            token_feature_grad: result.token_feature_grad,
            grad_ln1_weight: result.grad_ln1_weight,
            grad_ln1_bias: result.grad_ln1_bias,
            grad_channel_mix_k: result.channel_mix.grad_mix_k,
            grad_channel_key_weight: result.channel_mix.grad_key_weight,
            grad_channel_value_weight: result.channel_mix.grad_value_weight,
            grad_ln2_weight: result.channel_mix.grad_layer_norm_weight,
            grad_ln2_bias: result.channel_mix.grad_layer_norm_bias,
        })?
    );
    Ok(())
}
