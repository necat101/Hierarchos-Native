//! Explicit operand replay for candidate shaders; no model or oracle mutation.
#![allow(dead_code)]
#[path = "../src/kernel_golden.rs"]
mod kernel_golden;
#[path = "../src/vulkan.rs"]
mod vulkan;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Input {
    shaders: Vec<PathBuf>,
    bindings: Vec<Vec<f32>>,
    push: Vec<u32>,
    grid: [u32; 3],
    outputs: Vec<usize>,
}
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input: Input =
        serde_json::from_slice(&std::fs::read(args.next().context("input JSON required")?)?)?;
    let destination = args.next().context("output JSON required")?;
    let device = vulkan::VulkanDevice::new()?;
    anyhow::ensure!(
        device.vendor_id() == 0x1002 && device.device_id() == 0x15bf,
        "unexpected adapter"
    );
    let buffers = input
        .bindings
        .iter()
        .map(|v| vulkan::GpuBuffer::from_f32(&device, v))
        .collect::<Result<Vec<_>>>()?;
    let references = buffers.iter().collect::<Vec<_>>();
    let mut results = Vec::new();
    for path in input.shaders {
        for (buffer, values) in buffers.iter().zip(&input.bindings) {
            buffer.write_f32(values)?;
        }
        let shader = std::fs::read(&path)?;
        let kernel = vulkan::ComputeKernel::new(
            &device,
            &shader,
            buffers.len(),
            (input.push.len() * 4) as u32,
        )?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &references,
            bytemuck::cast_slice(&input.push),
            input.grid,
        )?;
        commands.submit()?;
        let output = input
            .outputs
            .iter()
            .map(|&index| {
                Ok(buffers[index]
                    .read_f32(input.bindings[index].len())?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>>>()?;
        results.push(serde_json::json!({"shader":path,"words":output}));
    }
    std::fs::write(
        destination,
        serde_json::to_vec(&serde_json::json!({"vendor_id":device.vendor_id(),
        "device_id":device.device_id(),"driver_version":device.driver_version(),"results":results}))?,
    )?;
    Ok(())
}
