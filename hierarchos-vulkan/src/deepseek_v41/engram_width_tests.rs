use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Case {
    parent_fixture_sha256: String,
    rows: usize,
    dim: usize,
    hc: usize,
    stream_bits: Vec<u32>,
    kv_bits: Vec<u32>,
    weight_bits: Vec<u32>,
    mask: Vec<u32>,
    expected_bits: Vec<u32>,
}

#[test]
fn deepseek_v41_engram_gate_at_production_width_matches_cpu_bits() -> Result<()> {
    let source = include_bytes!("../../validation/fixtures/deepseek_v41_engram_width.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(source)),
        "c381e90883fcfdf24d7ef5d225e955d191b83a3827a92bf2f91c0c61138f68d1"
    );
    let case: Case = serde_json::from_slice(source)?;
    assert_eq!(
        case.parent_fixture_sha256,
        "70d2218429c2d4b8623a8ebbdd6c659eec22f7f64f9eeecd3d95c5d7877fdcf0"
    );
    assert_eq!((case.rows, case.dim, case.hc), (3, 5120, 4));
    assert_eq!(case.mask, [1, 0, 1]);
    let length = case.rows * case.hc * case.dim;
    assert_eq!(case.stream_bits.len(), length);
    assert_eq!(case.expected_bits.len(), length);
    assert_eq!(case.kv_bits.len(), case.rows * (case.hc + 1) * case.dim);
    assert_eq!(case.weight_bits.len(), case.hc * case.dim);
    let device = VulkanDevice::new()?;
    let gate =
        vulkan::ComputeKernel::new(&device, GATE, 5, std::mem::size_of::<GatePush>() as u32)?;
    let stream = GpuBuffer::from_u32(&device, &case.stream_bits)?;
    let kv = GpuBuffer::from_u32(&device, &case.kv_bits)?;
    let weight = GpuBuffer::from_u32(&device, &case.weight_bits)?;
    let mask = GpuBuffer::from_u32(&device, &case.mask)?;
    let output = GpuBuffer::zeros_f32(&device, length)?;
    let push = GatePush {
        rows: case.rows as u32,
        hc: case.hc as u32,
        dim: case.dim as u32,
        proj_width: ((case.hc + 1) * case.dim) as u32,
        eps: 1e-20,
        scale: (case.dim as f64).powf(-0.5) as f32,
        has_mask: 1,
    };
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    gate.record_dispatch(
        &mut commands,
        &[&stream, &kv, &weight, &mask, &output],
        bytemuck::bytes_of(&push),
        [(case.rows * case.hc) as u32, 1, 1],
    )?;
    commands.submit()?;
    let cast = DeepseekV41MixedLinearKernel::new(device)?;
    let result = cast
        .cast_buffer(&output, length, DeepseekV41WeightOutput::Bf16)?
        .read_u32(length)?;
    let mismatches = result
        .iter()
        .zip(&case.expected_bits)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches, 0,
        "production-width Engram BF16 residual mismatches"
    );
    Ok(())
}
