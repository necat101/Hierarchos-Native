//! Replay a primitive with externally supplied buffers and frozen expected words.
#![allow(dead_code)]
#[path = "../vulkan.rs"]
mod vulkan;
// `vulkan.rs` names its cooperative-matrix capability types through
// `crate::wmma`, so a bin that compiles it standalone needs that module in the
// same crate root.
#[path = "../wmma.rs"]
mod wmma;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use vulkan::{ComputeBatch, ComputeKernel, GpuBuffer, VulkanDevice};

#[derive(Deserialize)]
struct Fixture {
    buffers: Vec<Vec<f32>>,
    output_binding: usize,
    expected_bits: Vec<u32>,
    push: Vec<u32>,
    dispatch: [u32; 3],
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(
        args.first().context("fixture path required")?,
    )?)?;
    let code = std::fs::read(args.get(1).context("SPIR-V path required")?)?;
    if fixture.output_binding >= fixture.buffers.len() || fixture.expected_bits.is_empty() {
        bail!("invalid primitive fixture");
    }
    let device = VulkanDevice::new()?;
    let buffers = fixture
        .buffers
        .iter()
        .map(|v| GpuBuffer::from_f32(&device, v))
        .collect::<Result<Vec<_>>>()?;
    let kernel = ComputeKernel::new(
        &device,
        &code,
        buffers.len(),
        (fixture.push.len() * 4) as u32,
    )?;
    let mut commands = ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &buffers.iter().collect::<Vec<_>>(),
        bytemuck::cast_slice(&fixture.push),
        fixture.dispatch,
    )?;
    commands.submit()?;
    let actual = buffers[fixture.output_binding].read_f32(fixture.expected_bits.len())?;
    if let Some(path) = args.get(2) {
        std::fs::write(path, serde_json::to_vec(&actual)?)?;
    }
    let failures = actual
        .iter()
        .zip(&fixture.expected_bits)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != **b)
        .collect::<Vec<_>>();
    let worst = actual
        .iter()
        .zip(&fixture.expected_bits)
        .map(|(a, b)| (*a - f32::from_bits(*b)).abs())
        .fold(0.0, f32::max);
    println!(
        "{}/{} different words, max {worst:e} on {}",
        failures.len(),
        actual.len(),
        device.name()
    );
    for (index, (value, bits)) in failures.iter().take(8) {
        println!(
            "  {index}: actual={:08x} expected={bits:08x}",
            value.to_bits()
        );
    }
    if !failures.is_empty() {
        bail!("GPU primitive differs");
    }
    Ok(())
}
