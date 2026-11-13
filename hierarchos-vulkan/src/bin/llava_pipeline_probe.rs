//! Raw RGB image -> CLIP -> projector -> text logits, full backward and export.
use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{CpuAdamWHyperParams, VulkanDevice, VulkanLlava};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    batch_size: usize,
    seq_len: usize,
    image_count: usize,
    input_ids: Vec<u32>,
    attention_mask: Vec<f32>,
    targets: Vec<u32>,
    loss_weights: Vec<f32>,
    rgb: Vec<u8>,
    alternate_rgb: Vec<u8>,
    steps: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}
fn main() -> Result<()> {
    std::thread::Builder::new()
        .name("llava-pipeline".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("LLaVA probe panicked"))?
}
fn run() -> Result<()> {
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let value = PathBuf::from(args.next().context("missing argument value")?);
        match arg.to_str() {
            Some("--model") => model = Some(value),
            Some("--fixture") => fixture = Some(value),
            Some("--output") => output = Some(value),
            _ => bail!("unknown argument {arg:?}"),
        }
    }
    let model = model.context("--model required")?;
    let output = output.context("--output required")?;
    let f: Fixture = serde_json::from_slice(&fs::read(fixture.context("--fixture required")?)?)?;
    if f.steps != 2 {
        bail!("qualification requires exactly two composed updates");
    }
    let device = VulkanDevice::new()?;
    let mut graph = VulkanLlava::from_hf_package(
        device.clone(),
        &model,
        f.batch_size,
        f.seq_len,
        f.image_count,
    )?;
    let pixels = graph.preprocess_rgb8(&f.rgb)?;
    let alternate_pixels = graph.preprocess_rgb8(&f.alternate_rgb)?;
    let alternate = graph.forward(&f.input_ids, &f.attention_mask, &alternate_pixels)?;
    fs::create_dir(&output).context("output must be a new directory")?;
    let hyper = CpuAdamWHyperParams {
        lr: f.lr,
        beta1: f.beta1,
        beta2: f.beta2,
        eps: f.eps,
        weight_decay: f.weight_decay,
    };
    let mut steps = Vec::new();
    for _ in 0..f.steps {
        let forward = graph.forward(&f.input_ids, &f.attention_mask, &pixels)?;
        let result = graph.train_step_with_optimizer(
            &f.input_ids,
            &f.attention_mask,
            &pixels,
            &f.targets,
            &f.loss_weights,
            hyper,
        )?;
        graph.export_hf_package(&model, output.join(format!("step-{}", result.text.step)))?;
        steps.push(serde_json::json!({"forward":forward,"training":result}));
    }
    let final_output = graph.forward(&f.input_ids, &f.attention_mask, &pixels)?;
    let reloaded = VulkanLlava::from_hf_package(
        device.clone(),
        output.join("step-2"),
        f.batch_size,
        f.seq_len,
        f.image_count,
    )?;
    let reload_pixels = reloaded.preprocess_rgb8(&f.rgb)?;
    let reload_output = reloaded.forward(&f.input_ids, &f.attention_mask, &reload_pixels)?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":1,"scope":"raw_rgb_clip_llava_full_parameter_training",
            "device":device.name(),"vendor_id":device.vendor_id(),"device_id":device.device_id(),
            "precision":{"requested":hierarchos_vulkan::wmma::WmmaMode::from_env()?.label(),
                "accumulation_dispatches":hierarchos_vulkan::wmma_linear::linear_wmma_precision_counts(),
                "phases":hierarchos_vulkan::wmma_linear::linear_wmma_phase_counts(),
                "master_weights":"fp32","optimizer_state":"fp32",
                "loss_scaler":graph.loss_scaler_state()?},
            "pixels":pixels,"alternate":alternate,"steps":steps,"final":final_output,"reload":reload_output,
        }))?,
    )?;
    Ok(())
}
