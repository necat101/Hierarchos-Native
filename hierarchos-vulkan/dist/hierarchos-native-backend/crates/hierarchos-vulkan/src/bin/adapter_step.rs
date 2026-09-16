use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use hierarchos_vulkan::{AdamWHyperParams, SharedTokenAdapterTrainer, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    #[serde(default = "default_steps")]
    steps: u32,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    input: Vec<f32>,
    grad_output: Vec<f32>,
    down_weight: Vec<f32>,
    up_weight: Vec<f32>,
    bias: Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    matrix_weight_decay: f32,
}

fn default_steps() -> u32 {
    1
}

#[derive(Serialize)]
struct Output {
    device: String,
    step: u32,
    elapsed_ms: f64,
    dispatch_count: usize,
    shader_barrier_count: usize,
    output: Vec<f32>,
    input_grad: Vec<f32>,
    down_weight: Vec<f32>,
    up_weight: Vec<f32>,
    bias: Vec<f32>,
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
    let case: Case = serde_json::from_slice(&fs::read(&case_path)?)?;
    anyhow::ensure!(
        case.input.len() == case.rows * case.input_dim,
        "case input length does not match rows * input_dim"
    );
    anyhow::ensure!(
        case.grad_output.len() == case.rows * case.output_dim,
        "case grad_output length does not match rows * output_dim"
    );
    anyhow::ensure!(case.steps > 0, "case steps must be positive");

    let device = VulkanDevice::new()?;
    let mut trainer = SharedTokenAdapterTrainer::new(
        device,
        case.input_dim,
        case.output_dim,
        case.rank,
        case.rows,
        &case.down_weight,
        &case.up_weight,
        &case.bias,
        case.matrix_weight_decay,
    )?;
    let hyper = AdamWHyperParams {
        lr: case.lr,
        beta1: case.beta1,
        beta2: case.beta2,
        eps: case.eps,
        weight_decay: case.matrix_weight_decay,
    };
    let mut result = None;
    let started = Instant::now();
    for _ in 0..case.steps {
        result = Some(trainer.train_step(&case.input, &case.grad_output, hyper)?);
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let result = result.expect("positive step count was validated");
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: trainer.device_name().to_string(),
            step: result.step,
            elapsed_ms,
            dispatch_count: result.dispatch_count,
            shader_barrier_count: result.shader_barrier_count,
            output: result.output,
            input_grad: result.input_grad,
            down_weight: trainer.down_weights()?,
            up_weight: trainer.up_weights()?,
            bias: trainer.bias()?,
        })?
    );
    Ok(())
}
