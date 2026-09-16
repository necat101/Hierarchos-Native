use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvTimeMixCoreOp, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    state: Vec<f32>,
    x_norm: Vec<f32>,
    previous: Vec<f32>,
    mix_r: Vec<f32>,
    mix_k: Vec<f32>,
    mix_v: Vec<f32>,
    receptance_weight: Vec<f32>,
    key_weight: Vec<f32>,
    value_weight: Vec<f32>,
    k_k: Vec<f32>,
    k_a: Vec<f32>,
    a: Vec<f32>,
    w: Vec<f32>,
    grad_new_state: Vec<f32>,
    grad_tmix: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    heads: usize,
    projection_native_fp16_backward_compute_active: bool,
    new_state: Vec<f32>,
    tmix: Vec<f32>,
    scaled_k: Vec<f32>,
    kk: Vec<f32>,
    grad_state: Vec<f32>,
    grad_x_norm: Vec<f32>,
    grad_previous: Vec<f32>,
    grad_a: Vec<f32>,
    grad_w: Vec<f32>,
    grad_mix_r: Vec<f32>,
    grad_mix_k: Vec<f32>,
    grad_mix_v: Vec<f32>,
    grad_receptance_weight: Vec<f32>,
    grad_key_weight: Vec<f32>,
    grad_value_weight: Vec<f32>,
    grad_k_k: Vec<f32>,
    grad_k_a: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut prefix = None;
    let mut native_fp16_projection_backward = false;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--prefix" => {
                prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--native-fp16-projection-backward" => native_fp16_projection_backward = true,
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path
        .context("usage: --case CASE.json [--model-dir MODEL_DIR --prefix h_rnn|l_rnn]")?;
    let case: Case = serde_json::from_slice(&fs::read(&case_path)?)?;

    let device = VulkanDevice::new()?;
    let mut op = if let Some(model_dir) = model_dir {
        let prefix = prefix
            .as_deref()
            .context("--model-dir requires --prefix h_rnn or --prefix l_rnn")?;
        RwkvTimeMixCoreOp::from_model_package(
            device,
            model_dir,
            prefix,
            case.head_size,
            case.batch,
        )?
    } else {
        if prefix.is_some() {
            anyhow::bail!("--prefix is only meaningful together with --model-dir");
        }
        RwkvTimeMixCoreOp::new(
            device,
            case.width,
            case.head_size,
            case.batch,
            &case.mix_r,
            &case.mix_k,
            &case.mix_v,
            &case.receptance_weight,
            &case.key_weight,
            &case.value_weight,
            &case.k_k,
            &case.k_a,
        )?
    };
    if native_fp16_projection_backward {
        op.enable_projection_native_fp16_backward_compute()?;
    }
    let result = op.forward_backward(
        case.batch,
        &case.state,
        &case.x_norm,
        &case.previous,
        &case.a,
        &case.w,
        &case.grad_new_state,
        &case.grad_tmix,
    )?;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            heads: op.heads(),
            projection_native_fp16_backward_compute_active: op
                .projection_native_fp16_backward_compute_active(),
            new_state: result.new_state,
            tmix: result.tmix,
            scaled_k: result.scaled_k,
            kk: result.kk,
            grad_state: result.grad_state,
            grad_x_norm: result.grad_x_norm,
            grad_previous: result.grad_previous,
            grad_a: result.grad_a,
            grad_w: result.grad_w,
            grad_mix_r: result.grad_mix_r,
            grad_mix_k: result.grad_mix_k,
            grad_mix_v: result.grad_mix_v,
            grad_receptance_weight: result.grad_receptance_weight,
            grad_key_weight: result.grad_key_weight,
            grad_value_weight: result.grad_value_weight,
            grad_k_k: result.grad_k_k,
            grad_k_a: result.grad_k_a,
        })?
    );
    Ok(())
}
