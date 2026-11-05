//! Independent frozen oneDNN GELU primitive replay; no model arithmetic on CPU.
#![allow(dead_code)]
#[path = "../vulkan.rs"]
mod vulkan;
use anyhow::{bail, Context, Result};
use vulkan::{ComputeBatch, ComputeKernel, GpuBuffer, VulkanDevice};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    input: Vec<f32>,
    gradient: Vec<f32>,
    expected_forward_bits: Vec<u32>,
    expected_backward_bits: Vec<u32>,
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .context("frozen GELU fixture required")?;
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    let n = fixture.input.len();
    if n == 0
        || fixture.gradient.len() != n
        || fixture.expected_forward_bits.len() != n
        || fixture.expected_backward_bits.len() != n
    {
        bail!("invalid primitive fixture shape");
    }
    let device = VulkanDevice::new()?;
    let input = GpuBuffer::from_f32(&device, &fixture.input)?;
    let gradient = GpuBuffer::from_f32(&device, &fixture.gradient)?;
    let output = GpuBuffer::zeros_f32(&device, n)?;
    for backward in [false, true] {
        let code: &[u8] = if backward {
            include_bytes!("../../shaders/vision_gelu_erf_cpu_backward.spv")
        } else {
            include_bytes!("../../shaders/vision_gelu_erf_cpu_forward.spv")
        };
        let kernel = ComputeKernel::new(&device, code, if backward { 3 } else { 2 }, 4)?;
        let mut commands = ComputeBatch::new(&device)?;
        let bindings = if backward {
            vec![&input, &gradient, &output]
        } else {
            vec![&input, &output]
        };
        kernel.record_dispatch(
            &mut commands,
            &bindings,
            bytemuck::cast_slice(&[n as u32]),
            [n.div_ceil(256) as u32, 1, 1],
        )?;
        commands.submit()?;
        let actual = output.read_f32(n)?;
        let expected = if backward {
            &fixture.expected_backward_bits
        } else {
            &fixture.expected_forward_bits
        };
        let failures = actual
            .iter()
            .zip(expected)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != **b)
            .collect::<Vec<_>>();
        println!(
            "{}: {}/{} different words on {}",
            if backward { "backward" } else { "forward" },
            failures.len(),
            n,
            device.name()
        );
        for (index, (value, bits)) in failures.iter().take(8) {
            println!(
                "  {index}: actual={:08x} expected={bits:08x}",
                value.to_bits()
            );
        }
        if !failures.is_empty() {
            bail!("oneDNN GELU primitive differs");
        }
    }
    Ok(())
}
