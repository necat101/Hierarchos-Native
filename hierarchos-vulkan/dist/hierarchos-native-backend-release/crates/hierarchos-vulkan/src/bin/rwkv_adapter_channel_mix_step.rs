use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    RwkvAdapterChannelMixOp, RwkvChannelMixOp, SharedTokenAdapterTrainer, VulkanDevice,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    input_dim: usize,
    width: usize,
    rank: usize,
    token_features: Vec<f32>,
    x: Vec<f32>,
    previous: Vec<f32>,
    grad_output: Vec<f32>,
    adapter_down_weight: Vec<f32>,
    adapter_up_weight: Vec<f32>,
    adapter_bias: Vec<f32>,
    layer_norm_weight: Vec<f32>,
    layer_norm_bias: Vec<f32>,
    mix_k: Vec<f32>,
    key_weight: Vec<f32>,
    value_weight: Vec<f32>,
    key_clamp: f32,
    deepembed_clamp: f32,
}

#[derive(Serialize)]
struct Output {
    device: String,
    output: Vec<f32>,
    grad_x: Vec<f32>,
    grad_previous: Vec<f32>,
    grad_deepembed: Vec<f32>,
    token_feature_grad: Vec<f32>,
    grad_mix_k: Vec<f32>,
    grad_key_weight: Vec<f32>,
    grad_value_weight: Vec<f32>,
    grad_layer_norm_weight: Vec<f32>,
    grad_layer_norm_bias: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context("usage: --case CASE.json")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    if case.batch == 0 || case.input_dim == 0 || case.width == 0 || case.rank == 0 {
        anyhow::bail!("case batch/input_dim/width/rank must be positive");
    }

    let device = VulkanDevice::new()?;
    let adapter = SharedTokenAdapterTrainer::new(
        device.clone(),
        case.input_dim,
        case.width * 4,
        case.rank,
        case.batch,
        &case.adapter_down_weight,
        &case.adapter_up_weight,
        &case.adapter_bias,
        0.0,
    )?;
    let channel_mix = RwkvChannelMixOp::new(
        device.clone(),
        case.width,
        case.batch,
        &case.layer_norm_weight,
        &case.layer_norm_bias,
        &case.mix_k,
        &case.key_weight,
        &case.value_weight,
        case.key_clamp,
        case.deepembed_clamp,
    )?;
    let mut op = RwkvAdapterChannelMixOp::new(device, adapter, channel_mix)?;
    let result = op.forward_backward(
        case.batch,
        &case.x,
        &case.previous,
        &case.token_features,
        &case.grad_output,
    )?;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            output: result.channel_mix.output,
            grad_x: result.channel_mix.grad_x,
            grad_previous: result.channel_mix.grad_previous,
            grad_deepembed: result.channel_mix.grad_deepembed,
            token_feature_grad: result.token_feature_grad,
            grad_mix_k: result.channel_mix.grad_mix_k,
            grad_key_weight: result.channel_mix.grad_key_weight,
            grad_value_weight: result.channel_mix.grad_value_weight,
            grad_layer_norm_weight: result.channel_mix.grad_layer_norm_weight,
            grad_layer_norm_bias: result.channel_mix.grad_layer_norm_bias,
        })?
    );
    Ok(())
}
