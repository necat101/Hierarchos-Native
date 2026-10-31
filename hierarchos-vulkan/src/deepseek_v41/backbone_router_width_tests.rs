//! Qualify the full 384-candidate router at the released hidden width.
use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct WidthCase {
    name: String,
    rows: usize,
    input_bits: Vec<u32>,
    dot_bits: Vec<u32>,
    indices: Vec<u32>,
    raw_bits: Vec<u32>,
    normalized_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct RouterWidth {
    seed: u64,
    source_sha256: String,
    dim: usize,
    experts: usize,
    topk: usize,
    weight_sha256: String,
    bias_bits: Vec<u32>,
    cases: Vec<WidthCase>,
}

#[test]
fn deepseek_v41_384_expert_top6_router_at_production_width_matches_cpu_bits() -> Result<()> {
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../validation/fixtures/deepseek_v41_router_width.json"
            ))
        ),
        "06ada7581a6e115542427e245ca6ce58c20b7ec30697a0ed2163ccd3f9526041"
    );
    let fixture: RouterWidth = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_router_width.json"
    ))?;
    assert_eq!(fixture.seed, 4141212);
    assert_eq!(
        fixture.source_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    assert_eq!((fixture.dim, fixture.experts, fixture.topk), (5120, 384, 6));
    assert_eq!(
        fixture.cases.iter().map(|c| c.rows).collect::<Vec<_>>(),
        [1, 3, 4]
    );
    let bytes = include_bytes!("../../validation/fixtures/deepseek_v41_router_width.bf16.bin");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        fixture.weight_sha256
    );
    assert_eq!(bytes.len(), fixture.dim * fixture.experts * 2);
    let weight_bits: Vec<u32> = bytes
        .chunks_exact(2)
        .map(|b| (u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
        .collect();
    let device = VulkanDevice::new()?;
    let gate = vulkan::ComputeKernel::new(
        &device,
        MOE_GATE,
        7,
        std::mem::size_of::<MoeGatePush>() as u32,
    )?;
    let linear = LinearKernel::new(&device)?;
    let w = GpuBuffer::from_u32(&device, &weight_bits)?;
    let bias = GpuBuffer::from_u32(&device, &fixture.bias_bits)?;
    let seeds = crate::deepseek_v41::cpu_sqrt_seeds(&device)?;
    for case in &fixture.cases {
        let slots = case.rows * fixture.topk;
        assert_eq!(case.input_bits.len(), case.rows * fixture.dim);
        assert_eq!(case.indices.len(), slots);
        assert_eq!(case.raw_bits.len(), slots);
        assert_eq!(case.normalized_bits.len(), slots);
        let x = GpuBuffer::from_u32(&device, &case.input_bits)?;
        let dot = linear.apply(&device, &x, &w, case.rows, fixture.dim, fixture.experts)?;
        assert_eq!(
            dot.read_u32(case.dot_bits.len())?,
            case.dot_bits,
            "{} dots",
            case.name
        );
        for normalize in [0, 1] {
            let output = GpuBuffer::zeros_u32(&device, slots)?;
            let indices = GpuBuffer::zeros_u32(&device, slots)?;
            let counts = GpuBuffer::zeros_u32(&device, fixture.experts)?;
            let push = MoeGatePush {
                rows: case.rows as u32,
                dim: fixture.dim as u32,
                experts: fixture.experts as u32,
                topk: fixture.topk as u32,
                score_func: 2,
                gate_temp: 1.0,
                norm_topk: normalize,
                route_scale: if normalize == 0 { 1.0 } else { 1.5 },
            };
            let mut commands = vulkan::ComputeBatch::new(&device)?;
            gate.record_dispatch(
                &mut commands,
                &[&x, &w, &bias, &output, &indices, &counts, &seeds],
                bytemuck::bytes_of(&push),
                [case.rows as u32, 1, 1],
            )?;
            commands.submit()?;
            let expected = if normalize == 0 {
                &case.raw_bits
            } else {
                &case.normalized_bits
            };
            assert_eq!(
                output.read_u32(slots)?,
                *expected,
                "{} normalize {normalize}",
                case.name
            );
            assert_eq!(
                indices.read_u32(slots)?,
                case.indices,
                "{} selected experts",
                case.name
            );
            let mut expected_counts = vec![0; fixture.experts];
            for &expert in &case.indices {
                expected_counts[expert as usize] += 1;
            }
            assert_eq!(
                counts.read_u32(fixture.experts)?,
                expected_counts,
                "{} counts",
                case.name
            );
        }
    }
    Ok(())
}
