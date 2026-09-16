use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvNumericsPolicy, RwkvTimeMixCoreOp, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    state: Vec<f32>,
    x_norm: Vec<f32>,
    previous: Vec<f32>,
    grad_new_state: Vec<f32>,
    grad_output: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    heads: usize,
    new_state: Vec<f32>,
    tmix: Vec<f32>,
    scaled_k: Vec<f32>,
    kk: Vec<f32>,
    a: Vec<f32>,
    w: Vec<f32>,
    g: Vec<f32>,
    post_output: Vec<f32>,
    group_normed: Vec<f32>,
    grad_state: Vec<f32>,
    grad_x_norm: Vec<f32>,
    grad_previous: Vec<f32>,
    grad_mix_r: Vec<f32>,
    grad_mix_k: Vec<f32>,
    grad_mix_v: Vec<f32>,
    grad_mix_w: Vec<f32>,
    grad_mix_a: Vec<f32>,
    grad_mix_g: Vec<f32>,
    grad_receptance_weight: Vec<f32>,
    grad_key_weight: Vec<f32>,
    grad_value_weight: Vec<f32>,
    grad_k_k: Vec<f32>,
    grad_k_a: Vec<f32>,
    grad_w0: Vec<f32>,
    grad_w1: Vec<f32>,
    grad_w2: Vec<f32>,
    grad_a0: Vec<f32>,
    grad_a1: Vec<f32>,
    grad_a2: Vec<f32>,
    grad_g1: Vec<f32>,
    grad_g2: Vec<f32>,
    grad_r_k: Vec<f32>,
    grad_output_weight: Vec<f32>,
    grad_group_norm_weight: Vec<f32>,
    grad_group_norm_bias: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut prefix = None;
    let mut kernel_geometry = None;
    let mut numerics_policy = RwkvNumericsPolicy::StrictParity;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--prefix" => {
                prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--kernel-geometry" => {
                kernel_geometry = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--numerics" => {
                let value = args
                    .next()
                    .context("--numerics requires strict|fast-subgroup")?;
                numerics_policy = match value.to_string_lossy().as_ref() {
                    "strict" | "strict-parity" => RwkvNumericsPolicy::StrictParity,
                    "fast-subgroup" => RwkvNumericsPolicy::FastSubgroup,
                    other => anyhow::bail!(
                        "unknown --numerics value {other:?}; expected strict|fast-subgroup; recurrent tree/tiled policies are exercised by the TBPTT/full-training parity path"
                    ),
                };
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path =
        case_path.context("usage: --case CASE.json --model-dir MODEL_DIR --prefix h_rnn|l_rnn")?;
    let model_dir = model_dir.context("full RWKV runner requires --model-dir")?;
    let prefix = prefix.context("full RWKV runner requires --prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    if case.width == 0 || case.head_size == 0 || case.width % case.head_size != 0 {
        anyhow::bail!(
            "case width {} must be positive and divisible by head_size {}",
            case.width,
            case.head_size
        );
    }

    let device = VulkanDevice::new()?;
    let mut op = RwkvTimeMixCoreOp::from_model_package_full(
        device,
        model_dir,
        &prefix,
        case.head_size,
        case.batch,
    )?;
    if let Some(kernel_geometry) = kernel_geometry.as_deref() {
        op.set_backward_kernel_geometry_label(case.batch, kernel_geometry)?;
    }
    op.set_numerics_policy(numerics_policy)?;
    let result = op.forward_backward_full(
        case.batch,
        &case.state,
        &case.x_norm,
        &case.previous,
        &case.grad_new_state,
        &case.grad_output,
    )?;
    let core = result.core;
    let low_rank = result.low_rank;
    let post = result.post_mix;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            heads: op.heads(),
            new_state: core.new_state,
            tmix: core.tmix,
            scaled_k: core.scaled_k,
            kk: core.kk,
            a: low_rank.a,
            w: low_rank.w,
            g: low_rank.g,
            post_output: post.output,
            group_normed: post.group_normed,
            grad_state: core.grad_state,
            grad_x_norm: result.grad_x_norm,
            grad_previous: result.grad_previous,
            grad_mix_r: core.grad_mix_r,
            grad_mix_k: core.grad_mix_k,
            grad_mix_v: core.grad_mix_v,
            grad_mix_w: low_rank.grad_mix_w,
            grad_mix_a: low_rank.grad_mix_a,
            grad_mix_g: low_rank.grad_mix_g,
            grad_receptance_weight: core.grad_receptance_weight,
            grad_key_weight: core.grad_key_weight,
            grad_value_weight: core.grad_value_weight,
            grad_k_k: core.grad_k_k,
            grad_k_a: core.grad_k_a,
            grad_w0: low_rank.grad_w0,
            grad_w1: low_rank.grad_w1,
            grad_w2: low_rank.grad_w2,
            grad_a0: low_rank.grad_a0,
            grad_a1: low_rank.grad_a1,
            grad_a2: low_rank.grad_a2,
            grad_g1: low_rank.grad_g1,
            grad_g2: low_rank.grad_g2,
            grad_r_k: post.grad_r_k,
            grad_output_weight: post.grad_output_weight,
            grad_group_norm_weight: post.grad_group_norm_weight,
            grad_group_norm_bias: post.grad_group_norm_bias,
        })?
    );
    Ok(())
}
