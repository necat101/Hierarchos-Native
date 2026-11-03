//! Experimental projected-feature training gate, not a processor/tower CLI.
use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    multimodal::ModalityEmbeddingRows, CpuAdamWHyperParams, LinearProjectionTrainer, VulkanDevice,
    VulkanTransformer,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Fixture {
    batch_size: usize,
    seq_len: usize,
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
    modality_rows: Vec<usize>,
    features: Vec<f32>,
    feature_dim: usize,
    projector_weight: Vec<f32>,
    projector_bias: Option<Vec<f32>>,
    steps: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

#[derive(Serialize)]
struct Step {
    step: u32,
    loss: f32,
    logits: Vec<f32>,
    projected: Vec<f32>,
    modality_embedding_grad: Vec<f32>,
    feature_grad: Vec<f32>,
    projector_weight: Vec<f32>,
    projector_bias: Option<Vec<f32>>,
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .name("multimodal-training-probe".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("multimodal probe panicked"))?
}

fn run() -> Result<()> {
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .context("expected --model/--fixture/--output value")?;
        match arg.to_str() {
            Some("--model") => model = Some(PathBuf::from(value)),
            Some("--fixture") => fixture = Some(PathBuf::from(value)),
            Some("--output") => output = Some(PathBuf::from(value)),
            _ => bail!("unknown option {arg:?}"),
        }
    }
    let model = model.context("--model required")?;
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(fixture.context("--fixture required")?)?)?;
    let output = output.context("--output required")?;
    if fixture.steps < 2 || fixture.steps > 16 {
        bail!("qualification requires 2..=16 steps");
    }
    let device = VulkanDevice::new()?;
    let mut graph = VulkanTransformer::from_hf_package(
        device.clone(),
        &model,
        fixture.batch_size,
        fixture.seq_len,
    )?;
    let dim = if graph.config().embedding_size == 0 {
        graph.config().hidden_size
    } else {
        graph.config().embedding_size
    };
    let expected_features = fixture
        .modality_rows
        .len()
        .checked_mul(fixture.feature_dim)
        .context("feature size overflow")?;
    if fixture.features.len() != expected_features {
        bail!("features must contain modality_rows.len() * feature_dim values");
    }
    let mut projector = LinearProjectionTrainer::new(
        device.clone(),
        fixture.feature_dim,
        dim,
        fixture.modality_rows.len(),
        &fixture.projector_weight,
        fixture.projector_bias.as_deref(),
        fixture.weight_decay as f32,
    )?;
    let hyper = CpuAdamWHyperParams {
        lr: fixture.lr,
        beta1: fixture.beta1,
        beta2: fixture.beta2,
        eps: fixture.eps,
        weight_decay: fixture.weight_decay,
    };
    projector.set_cpu_reference_math(true)?;
    graph.set_cpu_adamw_policy(Some(hyper))?;
    fs::create_dir(&output).context("output must be a new directory with an existing parent")?;
    let mut steps = Vec::new();
    for _ in 0..fixture.steps {
        let projected = projector.forward(&fixture.features)?;
        let rows = ModalityEmbeddingRows {
            rows: &fixture.modality_rows,
            values: &projected,
        };
        let logits = graph.forward_logits_with_modality_embeddings(
            &fixture.input_ids,
            &fixture.attention_mask,
            rows,
        )?;
        let result = graph.train_step_with_modality_embeddings(
            &fixture.input_ids,
            &fixture.targets,
            &fixture.attention_mask,
            &fixture.loss_weights,
            hyper.as_f32(),
            rows,
        )?;
        let projection = projector.train_step_with_optimizer(
            &fixture.features,
            &result.modality_embedding_grad,
            hyper,
        )?;
        if projection.step != result.text.step || projection.output != projected {
            bail!("composed projector trajectory diverged from its evaluation forward");
        }
        graph.export_hf_package(&model, output.join(format!("step-{}", result.text.step)))?;
        steps.push(Step {
            step: result.text.step,
            loss: result.text.loss,
            logits,
            projected,
            modality_embedding_grad: result.modality_embedding_grad,
            feature_grad: projection.input_grad,
            projector_weight: projector.weights()?,
            projector_bias: projector.bias_values()?,
        });
    }
    let projected = projector.forward(&fixture.features)?;
    let final_logits = graph.forward_logits_with_modality_embeddings(
        &fixture.input_ids,
        &fixture.attention_mask,
        ModalityEmbeddingRows {
            rows: &fixture.modality_rows,
            values: &projected,
        },
    )?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": 1, "device": device.name(), "vendor_id": device.vendor_id(),
            "device_id": device.device_id(), "architecture": graph.config().architecture.model_type(),
            "scope": "projected_features_full_parameter_training", "steps": steps,
            "optimizer_policy": "cpu_single_tensor_f64_coefficients_vulkan_f32",
            "projection_math_policy": "cpu_avx2_small_row_gemm_vulkan_f32",
            "final_logits": final_logits,
        }))?,
    )?;
    Ok(())
}
