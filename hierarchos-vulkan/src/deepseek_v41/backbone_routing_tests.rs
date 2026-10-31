use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    dim: usize,
    experts: usize,
    topk: usize,
    input_bits: Vec<u32>,
    weight_bits: Vec<u32>,
    bias_bits: Vec<u32>,
    dot_bits: Vec<u32>,
    indices: Vec<u32>,
    raw_bits: Vec<u32>,
    normalized_bits: Vec<u32>,
}

#[test]
fn deepseek_v41_top6_routing_reduction_matches_frozen_cpu() -> Result<()> {
    #[derive(Deserialize)]
    struct Fixture {
        parent_fixture_sha256: String,
        cases: Vec<Case>,
    }
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_routing_reduction.json"
    ))?;
    assert_eq!(
        fixture.parent_fixture_sha256,
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../validation/fixtures/deepseek_v41_structure_model.json"
            ))
        )
    );
    assert_eq!(fixture.cases.len(), 9);
    let device = VulkanDevice::new()?;
    let gate = vulkan::ComputeKernel::new(
        &device,
        MOE_GATE,
        7,
        std::mem::size_of::<MoeGatePush>() as u32,
    )?;
    let linear = LinearKernel::new(&device)?;
    let seeds = crate::deepseek_v41::cpu_sqrt_seeds(&device)?;
    let mut failures = vec![];
    for case in fixture.cases {
        let x = GpuBuffer::from_u32(&device, &case.input_bits)?;
        let w = GpuBuffer::from_u32(&device, &case.weight_bits)?;
        let b = GpuBuffer::from_u32(&device, &case.bias_bits)?;
        let dots = linear
            .apply(&device, &x, &w, case.rows, case.dim, case.experts)?
            .read_u32(case.dot_bits.len())?;
        let different = dots
            .iter()
            .zip(&case.dot_bits)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("{} raw linear mismatches {different}", case.name);
        if different > 0 {
            failures.push(format!("{} raw linear {different}", case.name));
        }
        for norm in [0, 1] {
            let output = GpuBuffer::zeros_u32(&device, case.indices.len())?;
            let indices = GpuBuffer::zeros_u32(&device, case.indices.len())?;
            let counts = GpuBuffer::zeros_u32(&device, case.experts)?;
            let push = MoeGatePush {
                rows: case.rows as u32,
                dim: case.dim as u32,
                experts: case.experts as u32,
                topk: case.topk as u32,
                score_func: 2,
                gate_temp: 1.0,
                norm_topk: norm,
                route_scale: if norm == 0 { 1.0 } else { 1.5 },
            };
            let mut commands = vulkan::ComputeBatch::new(&device)?;
            gate.record_dispatch(
                &mut commands,
                &[&x, &w, &b, &output, &indices, &counts, &seeds],
                bytemuck::bytes_of(&push),
                [case.rows as u32, 1, 1],
            )?;
            commands.submit()?;
            let actual = output.read_u32(case.indices.len())?;
            let expected = if norm == 0 {
                &case.raw_bits
            } else {
                &case.normalized_bits
            };
            let different = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
            eprintln!("{} normalize {norm}: {different} mismatches", case.name);
            if different > 0 {
                let first = actual
                    .iter()
                    .zip(expected)
                    .position(|(a, b)| a != b)
                    .unwrap();
                failures.push(format!(
                    "{} normalize {norm}: {different}, first {first} {:08x}/{:08x}",
                    case.name, actual[first], expected[first]
                ));
            }
            assert_eq!(
                indices.read_u32(case.indices.len())?,
                case.indices,
                "{} selections",
                case.name
            );
            assert_eq!(
                counts.read_u32(case.experts)?.iter().sum::<u32>() as usize,
                case.indices.len()
            );
        }
    }
    assert!(failures.is_empty(), "routing regression: {failures:?}");
    Ok(())
}

/// The real long-prompt boundary tie used to evict expert 351 in favor of 284.
/// Four copies retain the matrix gate path without loading the 40-layer graph.
#[test]
fn deepseek_v41_top6_boundary_tie_matches_frozen_cpu() -> Result<()> {
    #[derive(Deserialize)]
    struct TieFixture {
        parent_fixture_sha256: String,
        revision: String,
        source_sha256: String,
        layer: usize,
        row: usize,
        source_rows: usize,
        tied_experts: [u32; 2],
        tied_score_bits: u32,
        score_bits: Vec<u32>,
        biased_score_bits: Vec<u32>,
        legacy_indices: Vec<u32>,
        counts: Vec<u32>,
        case: Case,
    }
    let source = include_str!("../../validation/fixtures/deepseek_v41_gate_tie.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        "29d5a3d7bab55bef6106fd2cc005435b61938620f203fe7df9461e07c1dc624e",
        "boundary-tie expectations are frozen independently of the native gate"
    );
    let fixture: TieFixture = serde_json::from_str(source)?;
    assert_eq!(
        fixture.parent_fixture_sha256,
        "82dcb51beea92833fa8dc382daf30846cd195f5974a17e9eaf0f9c7dd314bbbc"
    );
    assert_eq!(fixture.revision, crate::deepseek_v41::REFERENCE_REVISION);
    assert_eq!(
        fixture.source_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    assert_eq!(
        (fixture.layer, fixture.row, fixture.source_rows),
        (27, 1019, 2048)
    );
    assert_eq!(fixture.tied_experts, [284, 351]);
    assert_eq!(fixture.tied_score_bits, 1.0706253051757812f32.to_bits());
    let c = fixture.case;
    assert_eq!((c.rows, c.dim, c.experts, c.topk), (4, 32, 384, 6));
    assert_eq!(c.input_bits.len(), c.rows * c.dim);
    assert_eq!(c.weight_bits.len(), c.experts * c.dim);
    assert_eq!(c.bias_bits.len(), c.experts);
    assert_eq!(c.dot_bits.len(), c.rows * c.experts);
    assert_eq!(fixture.score_bits.len(), c.rows * c.experts);
    assert_eq!(fixture.biased_score_bits.len(), c.rows * c.experts);
    assert_eq!(c.indices.len(), c.rows * c.topk);
    assert_eq!(c.raw_bits.len(), c.indices.len());
    assert_eq!(c.normalized_bits.len(), c.indices.len());
    assert_eq!(fixture.counts.len(), c.experts);
    assert_eq!(fixture.counts[284], 0);
    assert_eq!(fixture.counts[351] as usize, c.rows);
    for row in 0..c.rows {
        let tied = row * c.experts;
        assert_eq!(
            fixture.biased_score_bits[tied + 284],
            fixture.tied_score_bits
        );
        assert_eq!(
            fixture.biased_score_bits[tied + 351],
            fixture.tied_score_bits
        );
        // Correction bias ties the selection scores, while unbiased weights differ.
        assert_ne!(
            fixture.score_bits[tied + 284],
            fixture.score_bits[tied + 351]
        );
        let expected = &c.indices[row * c.topk..(row + 1) * c.topk];
        assert!(expected.contains(&351) && !expected.contains(&284));
        let mut legacy: Vec<_> = (0..c.experts as u32).collect();
        legacy.sort_by(|&a, &b| {
            f32::from_bits(fixture.biased_score_bits[tied + b as usize])
                .total_cmp(&f32::from_bits(
                    fixture.biased_score_bits[tied + a as usize],
                ))
                .then(a.cmp(&b))
        });
        let legacy = &legacy[..c.topk];
        assert_eq!(
            legacy,
            &fixture.legacy_indices[row * c.topk..(row + 1) * c.topk]
        );
        assert!(legacy.contains(&284) && !legacy.contains(&351));
        assert_ne!(
            legacy, expected,
            "the former tie rule must fail this fixture"
        );
    }
    let device = VulkanDevice::new()?;
    let x = GpuBuffer::from_u32(&device, &c.input_bits)?;
    let w = GpuBuffer::from_u32(&device, &c.weight_bits)?;
    let b = GpuBuffer::from_u32(&device, &c.bias_bits)?;
    let seeds = crate::deepseek_v41::cpu_sqrt_seeds(&device)?;
    assert_eq!(
        LinearKernel::new(&device)?
            .apply(&device, &x, &w, c.rows, c.dim, c.experts)?
            .read_u32(c.dot_bits.len())?,
        c.dot_bits,
        "original row's matrix gate products"
    );
    let gate = vulkan::ComputeKernel::new(
        &device,
        MOE_GATE,
        7,
        std::mem::size_of::<MoeGatePush>() as u32,
    )?;
    for (norm, expected) in [(0, &c.raw_bits), (1, &c.normalized_bits)] {
        let output = GpuBuffer::zeros_u32(&device, c.indices.len())?;
        let indices = GpuBuffer::zeros_u32(&device, c.indices.len())?;
        let counts = GpuBuffer::zeros_u32(&device, c.experts)?;
        let push = MoeGatePush {
            rows: c.rows as u32,
            dim: c.dim as u32,
            experts: c.experts as u32,
            topk: c.topk as u32,
            score_func: 2,
            gate_temp: 1.0,
            norm_topk: norm,
            route_scale: if norm == 0 { 1.0 } else { 1.5 },
        };
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        gate.record_dispatch(
            &mut commands,
            &[&x, &w, &b, &output, &indices, &counts, &seeds],
            bytemuck::bytes_of(&push),
            [c.rows as u32, 1, 1],
        )?;
        commands.submit()?;
        assert_eq!(
            indices.read_u32(c.indices.len())?,
            c.indices,
            "boundary expert, norm={norm}"
        );
        assert_eq!(
            output.read_u32(c.indices.len())?,
            *expected,
            "weight bits, norm={norm}"
        );
        assert_eq!(
            counts.read_u32(c.experts)?,
            fixture.counts,
            "all counts, norm={norm}"
        );
    }
    Ok(())
}
