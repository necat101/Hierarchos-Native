//! Native V4.1 tower/aligner precision qualification with real optimizer steps.
use anyhow::{Context, Result};
use hierarchos_vulkan::{
    CpuAdamWHyperParams, DeepseekV41VisionConfig, DeepseekV41VisionTensor, GpuBuffer,
    VulkanDeepseekV41Vision, VulkanDevice,
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Deserialize)]
struct Case {
    config: DeepseekV41VisionConfig,
    image: Image,
    patches: DeepseekV41VisionTensor,
    upstream: DeepseekV41VisionTensor,
    parameters: BTreeMap<String, DeepseekV41VisionTensor>,
}
#[derive(Deserialize)]
struct Image {
    n_vit_h: usize,
    n_vit_w: usize,
}
fn main() -> Result<()> {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("native vision precision worker panicked"))?
}
fn run() -> Result<()> {
    let args = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        args.len() == 2,
        "usage: deepseek_v41_vision_precision_probe CASE.json OUTPUT.json"
    );
    let case: Case = serde_json::from_slice(&fs::read(&args[0])?)?;
    let device = VulkanDevice::new()?;
    let mut graph = VulkanDeepseekV41Vision::from_tensors(
        device.clone(),
        case.config.clone(),
        case.image.n_vit_h,
        case.image.n_vit_w,
        &case.parameters,
    )?;
    let upstream = GpuBuffer::from_f32(&device, &case.upstream.values)?;
    let hyper = CpuAdamWHyperParams {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 0.001,
        weight_decay: 0.0,
    };
    let mut steps = Vec::new();
    for step in 0..3 {
        graph.forward(&case.patches.values)?;
        let mut row = serde_json::json!({"projected":graph.projected().read_f32(case.upstream.values.len())?});
        if step < 2 {
            graph.backward(&upstream)?;
            row["patch_gradient"] =
                serde_json::to_value(graph.patch_gradient().read_f32(case.patches.values.len())?)?;
            row["gradients"] = serde_json::to_value(graph.gradients()?)?;
            graph.apply_gradients(hyper)?;
            row["parameters"] = serde_json::to_value(graph.parameters()?)?;
        }
        steps.push(row);
    }
    let parameters = graph.parameters()?;
    let reload = VulkanDeepseekV41Vision::from_tensors(
        device.clone(),
        case.config,
        case.image.n_vit_h,
        case.image.n_vit_w,
        &parameters,
    )?;
    reload.forward(&case.patches.values)?;
    let report = serde_json::json!({"device":device.name(),"steps":steps,
        "reload":reload.projected().read_f32(case.upstream.values.len())?,
        "precision":{"requested":hierarchos_vulkan::wmma::WmmaMode::from_env()?.label(),
            "phases":hierarchos_vulkan::wmma_linear::linear_wmma_phase_counts(),
            "accumulation_dispatches":hierarchos_vulkan::wmma_linear::linear_wmma_precision_counts(),
            "master_weights":"fp32","optimizer_state":"fp32","automatic_loss_scaling":false}});
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])
        .context("creating precision report")?;
    serde_json::to_writer(file, &report)?;
    Ok(())
}
