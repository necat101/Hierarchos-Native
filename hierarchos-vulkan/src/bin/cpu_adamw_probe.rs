use anyhow::{Context, Result};
use hierarchos_vulkan::{cpu_adamw::CpuAdamW, CpuAdamWHyperParams, GpuBuffer, VulkanDevice};
use serde::Deserialize;
#[derive(Deserialize)]
struct Case {
    initial: Vec<f32>,
    gradients: Vec<Vec<f32>>,
    hyper: CpuAdamWHyperParams,
    decay: bool,
}
fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .context("expected optimizer fixture")?;
    let case: Case = serde_json::from_slice(&std::fs::read(path)?)?;
    let device = VulkanDevice::new()?;
    let len = case.initial.len();
    let p = GpuBuffer::from_f32(&device, &case.initial)?;
    let g = GpuBuffer::zeros_f32(&device, len)?;
    let m = GpuBuffer::zeros_f32(&device, len)?;
    let v = GpuBuffer::zeros_f32(&device, len)?;
    let kernel = CpuAdamW::new(&device)?;
    let mut steps = Vec::new();
    for (index, gradient) in case.gradients.iter().enumerate() {
        anyhow::ensure!(gradient.len() == len, "gradient geometry differs");
        g.write_f32(gradient)?;
        kernel.step(
            &p,
            &g,
            &m,
            &v,
            len,
            index as u32 + 1,
            case.hyper,
            case.decay,
        )?;
        steps.push(serde_json::json!({"parameter":p.read_f32(len)?,"moment":m.read_f32(len)?,"square":v.read_f32(len)?}));
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({"device":device.name(),"steps":steps}))?
    );
    Ok(())
}
