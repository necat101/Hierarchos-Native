use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{AdamWHyperParams, HierarchosHeadTrainer, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    hidden: Vec<f32>,
    targets: Vec<u32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

#[derive(Serialize)]
struct Output {
    device: String,
    loss: f32,
    step: u32,
    output_model: String,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut case_path = None;
    let mut output_model = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--case" => case_path = args.next().map(PathBuf::from),
            "--output-model" => output_model = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let model_dir = model_dir.context("missing --model MODEL_DIR")?;
    let case_path = case_path.context("missing --case CASE.json")?;
    let output_model = output_model.context("missing --output-model OUTPUT_DIR")?;
    let case: Case = serde_json::from_slice(&fs::read(&case_path)?)?;
    anyhow::ensure!(
        !case.targets.is_empty(),
        "case must contain at least one target"
    );

    let device = VulkanDevice::new()?;
    let mut trainer =
        HierarchosHeadTrainer::from_model_package(device, &model_dir, case.targets.len())?;
    let result = trainer.train_step(
        &case.hidden,
        &case.targets,
        AdamWHyperParams {
            lr: case.lr,
            beta1: case.beta1,
            beta2: case.beta2,
            eps: case.eps,
            weight_decay: case.weight_decay,
        },
    )?;
    trainer.export_model_package(&model_dir, &output_model)?;
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: trainer.device_name().to_string(),
            loss: result.loss,
            step: result.step,
            output_model: output_model.display().to_string(),
        })?
    );
    Ok(())
}
