use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvLowRankOp, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    x_norm: Vec<f32>,
    previous: Vec<f32>,
    mix_w: Vec<f32>,
    mix_a: Vec<f32>,
    mix_g: Vec<f32>,
    w0: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    a0: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
    g1: Vec<f32>,
    g2: Vec<f32>,
    grad_a: Vec<f32>,
    grad_w: Vec<f32>,
    grad_g: Vec<f32>,
    #[serde(default)]
    g2_input: Option<Vec<f32>>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    a: Vec<f32>,
    w: Vec<f32>,
    g: Vec<f32>,
    grad_x_norm: Vec<f32>,
    grad_previous: Vec<f32>,
    grad_mix_w: Vec<f32>,
    grad_mix_a: Vec<f32>,
    grad_mix_g: Vec<f32>,
    grad_w0: Vec<f32>,
    grad_w1: Vec<f32>,
    grad_w2: Vec<f32>,
    grad_a0: Vec<f32>,
    grad_a1: Vec<f32>,
    grad_a2: Vec<f32>,
    grad_g1: Vec<f32>,
    grad_g2: Vec<f32>,
}

#[derive(Serialize)]
struct NativeFp16DwDiagnosticOutput {
    device: String,
    branch: &'static str,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    native_fp16_grad: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut prefix = None;
    let mut native_fp16_dw_diagnostic = false;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--prefix" => {
                prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--native-fp16-dw-diagnostic" => native_fp16_dw_diagnostic = true,
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path
        .context(
            "usage: --case CASE.json [--model-dir MODEL_DIR --prefix h_rnn|l_rnn] [--native-fp16-dw-diagnostic]",
        )?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;

    let device = VulkanDevice::new()?;
    let mut op = if let Some(model_dir) = model_dir {
        let prefix = prefix
            .as_deref()
            .context("--model-dir requires --prefix h_rnn or --prefix l_rnn")?;
        RwkvLowRankOp::from_model_package(device, model_dir, prefix, case.batch)?
    } else {
        if prefix.is_some() {
            anyhow::bail!("--prefix is only meaningful together with --model-dir");
        }
        RwkvLowRankOp::new(
            device,
            case.width,
            case.w_rank,
            case.a_rank,
            case.g_rank,
            case.batch,
            &case.mix_w,
            &case.mix_a,
            &case.mix_g,
            &case.w0,
            &case.w1,
            &case.w2,
            &case.a0,
            &case.a1,
            &case.a2,
            &case.g1,
            &case.g2,
        )?
    };

    if native_fp16_dw_diagnostic {
        let g2_input = case
            .g2_input
            .as_deref()
            .context("--native-fp16-dw-diagnostic requires g2_input in the case JSON")?;
        let native_fp16_grad = op.diagnose_native_fp16_weight_grad(
            case.batch,
            case.g_rank,
            case.width,
            g2_input,
            &case.grad_g,
        )?;
        println!(
            "{}",
            serde_json::to_string(&NativeFp16DwDiagnosticOutput {
                device: op.device_name().to_string(),
                branch: "g2",
                rows: case.batch,
                input_dim: case.g_rank,
                output_dim: case.width,
                native_fp16_grad,
            })?
        );
        return Ok(());
    }

    let result = op.forward_backward(
        case.batch,
        &case.x_norm,
        &case.previous,
        &case.grad_a,
        &case.grad_w,
        &case.grad_g,
    )?;
    let (w_rank, a_rank, g_rank) = op.ranks();

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            w_rank,
            a_rank,
            g_rank,
            a: result.a,
            w: result.w,
            g: result.g,
            grad_x_norm: result.grad_x_norm,
            grad_previous: result.grad_previous,
            grad_mix_w: result.grad_mix_w,
            grad_mix_a: result.grad_mix_a,
            grad_mix_g: result.grad_mix_g,
            grad_w0: result.grad_w0,
            grad_w1: result.grad_w1,
            grad_w2: result.grad_w2,
            grad_a0: result.grad_a0,
            grad_a1: result.grad_a1,
            grad_a2: result.grad_a2,
            grad_g1: result.grad_g1,
            grad_g2: result.grad_g2,
        })?
    );
    Ok(())
}
