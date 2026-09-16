use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, VulkanDevice, VulkanTransformer, VulkanTransformerArchitecture,
    VulkanTransformerLoraConfig,
};
use serde::{Deserialize, Serialize};

fn default_dropout_seed() -> u32 {
    0x4849_4552
}

#[derive(Debug, Deserialize)]
struct Fixture {
    batch_size: usize,
    seq_len: usize,
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    #[serde(default = "default_dropout_seed")]
    dropout_seed: u32,
}

#[derive(Debug, Serialize)]
struct Report {
    device: String,
    architecture: String,
    loss: f32,
    losses: Vec<f32>,
    step: u32,
    supervised_tokens: usize,
    peft: bool,
}

fn main() -> Result<()> {
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    let mut base_output = None;
    let mut device_index = None;
    let mut lora_rank = None;
    let mut lora_alpha = None;
    let mut lora_targets = Vec::new();
    let mut lora_adapter = None;
    let mut seed = 42u64;
    let mut steps = 1usize;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model = Some(PathBuf::from(args.next().context("missing --model value")?)),
            "--fixture" => {
                fixture = Some(PathBuf::from(
                    args.next().context("missing --fixture value")?,
                ))
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().context("missing --output value")?,
                ))
            }
            "--base-output" => {
                base_output = Some(PathBuf::from(
                    args.next().context("missing --base-output value")?,
                ))
            }
            "--device-index" => {
                let raw = args.next().context("missing --device-index value")?;
                device_index = Some(
                    raw.to_string_lossy()
                        .parse::<usize>()
                        .context("invalid --device-index")?,
                );
            }
            "--lora-rank" => {
                let raw = args.next().context("missing --lora-rank value")?;
                lora_rank = Some(
                    raw.to_string_lossy()
                        .parse::<usize>()
                        .context("invalid --lora-rank")?,
                );
            }
            "--lora-alpha" => {
                let raw = args.next().context("missing --lora-alpha value")?;
                lora_alpha = Some(
                    raw.to_string_lossy()
                        .parse::<usize>()
                        .context("invalid --lora-alpha")?,
                );
            }
            "--lora-target" => lora_targets.push(
                args.next()
                    .context("missing --lora-target value")?
                    .to_string_lossy()
                    .into_owned(),
            ),
            "--lora-adapter" => {
                lora_adapter = Some(PathBuf::from(
                    args.next().context("missing --lora-adapter value")?,
                ))
            }
            "--seed" => {
                let raw = args.next().context("missing --seed value")?;
                seed = raw
                    .to_string_lossy()
                    .parse::<u64>()
                    .context("invalid --seed")?;
            }
            "--steps" => {
                let raw = args.next().context("missing --steps value")?;
                steps = raw
                    .to_string_lossy()
                    .parse::<usize>()
                    .context("invalid --steps")?;
                if steps == 0 {
                    bail!("--steps must be positive");
                }
            }
            "-h" | "--help" => {
                eprintln!("usage: transformer_parity --model DIR --fixture fixture.json --output DIR [--device-index N] [--steps N] [--lora-rank N --lora-alpha N --lora-target NAME ... | --lora-adapter DIR] [--seed N]");
                return Ok(());
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    let model = model.context("--model is required")?;
    let fixture_path = fixture.context("--fixture is required")?;
    let output = output.context("--output is required")?;
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(&fixture_path).with_context(|| format!("reading {}", fixture_path.display()))?,
    )?;
    let device = match device_index {
        Some(index) => VulkanDevice::new_with_index(index)?,
        None => VulkanDevice::new()?,
    };
    let mut graph =
        VulkanTransformer::from_hf_package(device, &model, fixture.batch_size, fixture.seq_len)?;
    graph.set_dropout_seed(fixture.dropout_seed);
    if lora_adapter.is_some() && lora_rank.is_some() {
        bail!("--lora-adapter and --lora-rank are mutually exclusive");
    }
    if let Some(adapter) = lora_adapter.as_deref() {
        graph.load_lora_adapter(adapter)?;
    } else if let Some(rank) = lora_rank {
        let alpha = lora_alpha.unwrap_or(rank * 2);
        let targets = if lora_targets.is_empty() {
            vec![graph
                .config()
                .architecture
                .default_lora_target_module()
                .to_owned()]
        } else {
            lora_targets
        };
        let fan_in_fan_out = matches!(
            graph.config().architecture,
            VulkanTransformerArchitecture::OpenAiGpt
                | VulkanTransformerArchitecture::Gpt2
                | VulkanTransformerArchitecture::GptSw3
        );
        graph.enable_lora(
            VulkanTransformerLoraConfig::causal_lm(rank, alpha, targets, fan_in_fan_out)?,
            seed,
        )?;
    }
    let peft = graph.lora_config().is_some();
    let hyper = AdamWHyperParams {
        lr: fixture.learning_rate,
        beta1: fixture.beta1,
        beta2: fixture.beta2,
        eps: fixture.eps,
        weight_decay: fixture.weight_decay,
    };
    let mut losses = Vec::with_capacity(steps);
    let mut final_result = None;
    for _ in 0..steps {
        let result = graph.train_step(
            &fixture.input_ids,
            &fixture.targets,
            &fixture.attention_mask,
            &fixture.loss_weights,
            hyper,
        )?;
        losses.push(result.loss);
        final_result = Some(result);
    }
    let result = final_result.context("parity trajectory produced no optimizer steps")?;
    if peft {
        graph.export_lora_adapter(&output)?;
        if let Some(base_output) = base_output.as_deref() {
            graph.export_hf_package(&model, base_output)?;
        }
    } else {
        graph.export_hf_package(&model, &output)?;
    }
    println!(
        "{}",
        serde_json::to_string(&Report {
            device: graph.device_name().to_owned(),
            architecture: graph.config().architecture.model_type().to_owned(),
            loss: result.loss,
            losses,
            step: result.step,
            supervised_tokens: result.supervised_tokens,
            peft,
        })?
    );
    Ok(())
}
