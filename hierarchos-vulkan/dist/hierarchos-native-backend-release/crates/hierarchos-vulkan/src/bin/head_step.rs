use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{write_f32_tensor, AdamWHyperParams, HierarchosHeadTrainer, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    #[serde(default = "default_steps")]
    steps: u32,
    rows: usize,
    context_dim: usize,
    vocab_size: usize,
    hidden: Vec<f32>,
    targets: Vec<u32>,
    weight: Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

fn default_steps() -> u32 {
    1
}

#[derive(Serialize)]
struct Output {
    device: String,
    loss: f32,
    step: u32,
    weights: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut output_path = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--output-safetensors" => output_path = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context("usage: --case CASE.json [--output-safetensors FILE]")?;
    let case: Case = serde_json::from_slice(&fs::read(&case_path)?)?;
    anyhow::ensure!(
        case.hidden.len() == case.rows * case.context_dim,
        "case hidden length does not match rows * context_dim"
    );
    anyhow::ensure!(
        case.targets.len() == case.rows,
        "case target count does not match rows"
    );
    anyhow::ensure!(case.steps > 0, "case steps must be positive");

    let device = VulkanDevice::new()?;
    let mut trainer = HierarchosHeadTrainer::new(
        device,
        case.context_dim,
        case.vocab_size,
        case.rows,
        &case.weight,
    )?;
    let hyper = AdamWHyperParams {
        lr: case.lr,
        beta1: case.beta1,
        beta2: case.beta2,
        eps: case.eps,
        weight_decay: case.weight_decay,
    };
    let mut result = None;
    for _ in 0..case.steps {
        result = Some(trainer.train_step(&case.hidden, &case.targets, hyper)?);
    }
    let result = result.expect("positive step count was validated");
    let weights = trainer.weights()?;
    if let Some(path) = output_path {
        write_f32_tensor(
            &path,
            "lm_head.weight",
            &[case.vocab_size, case.context_dim],
            &weights,
        )?;
    }
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: trainer.device_name().to_string(),
            loss: result.loss,
            step: result.step,
            weights,
        })?
    );
    Ok(())
}
