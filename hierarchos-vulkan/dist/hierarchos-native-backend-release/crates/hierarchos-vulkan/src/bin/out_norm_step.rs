use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{AdamWHyperParams, HierarchosOutNormHeadTrainer, VulkanDevice};
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
    lm_weight: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    #[serde(default = "default_activation_clamp")]
    activation_clamp: f32,
    #[serde(default)]
    tied_token_ids: Vec<u32>,
    #[serde(default)]
    tied_embedding_grad: Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

fn default_steps() -> u32 {
    1
}

fn default_activation_clamp() -> f32 {
    100.0
}

#[derive(Serialize)]
struct Output {
    device: String,
    activation_clamp: f32,
    loss: f32,
    step: u32,
    lm_weight: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    input_grad: Vec<f32>,
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
        case.hidden.len() == case.rows * case.context_dim,
        "case hidden length does not match rows * context_dim"
    );
    anyhow::ensure!(
        case.targets.len() == case.rows,
        "case target count does not match rows"
    );
    anyhow::ensure!(case.steps > 0, "case steps must be positive");

    let device = VulkanDevice::new()?;
    let mut trainer = HierarchosOutNormHeadTrainer::new(
        device,
        case.context_dim,
        case.vocab_size,
        case.rows,
        &case.lm_weight,
        &case.norm_weight,
        &case.norm_bias,
    )?;
    trainer.set_activation_clamp(case.activation_clamp)?;
    let hyper = AdamWHyperParams {
        lr: case.lr,
        beta1: case.beta1,
        beta2: case.beta2,
        eps: case.eps,
        weight_decay: case.weight_decay,
    };
    let mut result = None;
    for _ in 0..case.steps {
        result = Some(if case.tied_token_ids.is_empty() {
            trainer.train_step(&case.hidden, &case.targets, hyper)?
        } else {
            trainer.train_step_with_tied_embedding_grad(
                &case.hidden,
                &case.targets,
                &case.tied_token_ids,
                &case.tied_embedding_grad,
                hyper,
            )?
        });
    }
    let result = result.expect("positive step count was validated");
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: trainer.device_name().to_string(),
            activation_clamp: trainer.activation_clamp(),
            loss: result.loss,
            step: result.step,
            lm_weight: trainer.lm_weights()?,
            norm_weight: trainer.norm_weights()?,
            norm_bias: trainer.norm_bias()?,
            input_grad: trainer.input_grad(case.rows)?,
        })?
    );
    Ok(())
}
