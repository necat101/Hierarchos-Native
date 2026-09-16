use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvChannelMixOp, VulkanDevice};
use serde::{Deserialize, Serialize};

const QUALIFICATION_DEVICE_INDEX_ENV: &str = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX";

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    x: Vec<f32>,
    previous: Vec<f32>,
    deepembed: Vec<f32>,
    grad_output: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    output: Vec<f32>,
    grad_x: Vec<f32>,
    grad_previous: Vec<f32>,
    grad_deepembed: Vec<f32>,
    grad_mix_k: Vec<f32>,
    grad_key_weight: Vec<f32>,
    grad_value_weight: Vec<f32>,
    grad_layer_norm_weight: Vec<f32>,
    grad_layer_norm_bias: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut prefix = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--prefix" => {
                prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path =
        case_path.context("usage: --case CASE.json --model-dir MODEL_DIR --prefix h_rnn|l_rnn")?;
    let model_dir = model_dir.context("channel-mix runner requires --model-dir")?;
    let prefix = prefix.context("channel-mix runner requires --prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    if case.width == 0 {
        anyhow::bail!("case width must be positive");
    }

    let device = qualification_device()?;
    let mut op =
        RwkvChannelMixOp::from_model_package(device, model_dir, &prefix, case.batch, 12.0, 4.0)?;
    let result = op.forward_backward(
        case.batch,
        &case.x,
        &case.previous,
        &case.deepembed,
        &case.grad_output,
    )?;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            output: result.output,
            grad_x: result.grad_x,
            grad_previous: result.grad_previous,
            grad_deepembed: result.grad_deepembed,
            grad_mix_k: result.grad_mix_k,
            grad_key_weight: result.grad_key_weight,
            grad_value_weight: result.grad_value_weight,
            grad_layer_norm_weight: result.grad_layer_norm_weight,
            grad_layer_norm_bias: result.grad_layer_norm_bias,
        })?
    );
    Ok(())
}

fn qualification_device() -> Result<VulkanDevice> {
    match std::env::var(QUALIFICATION_DEVICE_INDEX_ENV) {
        Ok(raw) => {
            let index = raw.parse::<usize>().with_context(|| {
                format!("{QUALIFICATION_DEVICE_INDEX_ENV} must be a non-negative device index")
            })?;
            VulkanDevice::new_with_index(index)
        }
        Err(std::env::VarError::NotPresent) => VulkanDevice::new(),
        Err(err) => Err(err).with_context(|| format!("reading {QUALIFICATION_DEVICE_INDEX_ENV}")),
    }
}
