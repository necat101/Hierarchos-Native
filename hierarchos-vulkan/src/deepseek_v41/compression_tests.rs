use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    revision: String,
    atol_exclusive: f64,
    rtol: f64,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    ratio: usize,
    hidden: usize,
    head: usize,
    batch: usize,
    total: usize,
    kv_weight: Vec<f32>,
    gate_weight: Option<Vec<f32>>,
    norm_weight: Vec<f32>,
    input: Vec<f32>,
    steps: Vec<Step>,
    full_expected: Vec<f32>,
}

#[derive(Deserialize)]
struct Step {
    start_pos: usize,
    seq_len: usize,
    input: Vec<f32>,
    groups: usize,
    expected: Vec<f32>,
    pre_norm: Vec<f32>,
    variance: Vec<f32>,
    rstd: Vec<f32>,
    projected_kv: Vec<f32>,
    projected_gate: Vec<f32>,
}

fn assert_strict_values(label: &str, actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label} shape mismatch");
    let mut worst = 0.0f64;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let drift = (a as f64 - b as f64).abs();
        assert!(a.is_finite() && b.is_finite() && drift < 2e-7,
            "{label}, value {index}: native={a:.10e}, reference={b:.10e}, absolute drift={drift:.10e}");
        worst = worst.max(drift);
    }
    worst
}

#[test]
fn deepseek_v41_compressor_prefill_and_decode_match_pinned_fp32_reference() -> Result<()> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_compressor.json"
    ))?;
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    assert_eq!(fixture.atol_exclusive, 2e-7);
    assert_eq!(fixture.rtol, 0.0);
    let device = crate::VulkanDevice::new()?;
    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for case in fixture.cases {
        let weights = DeepseekV41CompressorWeights {
            hidden_size: case.hidden,
            head_dim: case.head,
            compress_ratio: case.ratio,
            eps: 1e-20,
            kv_weight: case.kv_weight,
            gate_weight: case.gate_weight,
            norm_weight: case.norm_weight,
        };
        let mut module =
            DeepseekV41Compressor::new(device.clone(), weights, case.batch, case.total)?;
        assert!(module
            .decode_step(
                &case.steps[0].input[..case.batch * case.hidden],
                case.batch,
                0
            )
            .is_err());
        let mut emitted_groups = 0;
        for step in &case.steps {
            if step.start_pos > 0 {
                let too_small = crate::GpuBuffer::zeros_f32(&device, 1)?;
                assert!(module
                    .decode_step_from_buffer(&too_small, case.batch, step.start_pos)
                    .is_err());
                assert!(module
                    .decode_step(&step.input, case.batch, step.start_pos + 1)
                    .is_err());
                assert!(module
                    .decode_step(
                        &step.input[..step.input.len() - 1],
                        case.batch,
                        step.start_pos
                    )
                    .is_err());
            }
            let latent = if step.start_pos == 0 {
                let gpu_input = crate::GpuBuffer::from_f32(&device, &step.input)?;
                module.prefill_from_buffer(&gpu_input, case.batch, step.seq_len)?
            } else if step.start_pos.is_multiple_of(2) {
                let gpu_input = crate::GpuBuffer::from_f32(&device, &step.input)?;
                module.decode_step_from_buffer(&gpu_input, case.batch, step.start_pos)?
            } else {
                module.decode_step(&step.input, case.batch, step.start_pos)?
            };
            assert_eq!(
                latent.groups_per_batch, step.groups,
                "{} at {}",
                case.name, step.start_pos
            );
            assert_eq!(latent.first_group, step.start_pos / case.ratio);
            if step.groups > 0 {
                assert_eq!(
                    latent.first_group, emitted_groups,
                    "{} group continuity",
                    case.name
                );
                emitted_groups += step.groups;
            }
            let actual = latent.read_f32()?;
            if actual
                .iter()
                .zip(&step.expected)
                .any(|(&a, &b)| (a as f64 - b as f64).abs() >= 2e-7)
            {
                let pooled = if step.groups == 0 {
                    vec![]
                } else {
                    latent
                        .pre_norm
                        .read_f32(case.batch * step.groups * case.head)?
                };
                let variance = if step.groups == 0 {
                    vec![]
                } else {
                    latent.variance.read_f32(case.batch * step.groups)?
                };
                let rstd = if step.groups == 0 {
                    vec![]
                } else {
                    latent.rstd.read_f32(case.batch * step.groups)?
                };
                let drift = pooled
                    .iter()
                    .zip(&step.pre_norm)
                    .map(|(&a, &b)| (a as f64 - b as f64).abs())
                    .fold(0.0, f64::max);
                eprintln!("{} position {} pre_norm drift={drift:e}; variance native={variance:?} ref={:?}; rstd native={rstd:?} ref={:?}", case.name, step.start_pos, step.variance, step.rstd);
                for (label, buffer, reference) in [
                    ("kv", &latent.projected_kv, &step.projected_kv),
                    ("gate", &latent.projected_gate, &step.projected_gate),
                ] {
                    if reference.is_empty() {
                        continue;
                    }
                    let vals = buffer.read_f32(reference.len())?;
                    let diff = vals
                        .iter()
                        .zip(reference)
                        .map(|(&a, &b)| (a as f64 - b as f64).abs())
                        .fold(0.0, f64::max);
                    let count = vals.iter().zip(reference).filter(|(a, b)| a != b).count();
                    eprintln!("{label} projection: {count} differing values; worst={diff:e}");
                }
            }
            worst = worst.max(assert_strict_values(
                &format!("{} position {}", case.name, step.start_pos),
                &actual,
                &step.expected,
            ));
            compared += actual.len();
        }
        assert!(module
            .decode_step(
                &case.steps.last().unwrap().input,
                case.batch,
                case.total + 1
            )
            .is_err());
        // Reset a used module; dirty state from the first stream must not leak.
        let full = module
            .prefill(&case.input, case.batch, case.total)?
            .read_f32()?;
        worst = worst.max(assert_strict_values(
            &format!("{} full prefill", case.name),
            &full,
            &case.full_expected,
        ));
        compared += full.len();
        // Prefill/decode reference GEMM shapes can round differently. Both are
        // independently checked against the unchanged reference, at the same
        // strict threshold, rather than assuming reference bit identity.
        assert_eq!(
            emitted_groups,
            case.total / case.ratio,
            "{} emitted group count",
            case.name
        );
        let reset = &case.steps[0];
        let reset_values = module
            .prefill(&reset.input, case.batch, reset.seq_len)?
            .read_f32()?;
        assert_strict_values(
            &format!("{} reset", case.name),
            &reset_values,
            &reset.expected,
        );
    }
    eprintln!(
        "V4.1 compressor on {}: {compared} reference values, worst absolute drift {worst:.12e}",
        device.name()
    );
    Ok(())
}

#[test]
fn deepseek_v41_compressor_rejects_foreign_device_input_before_dispatch() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let foreign = crate::VulkanDevice::new()?;
    let weights = DeepseekV41CompressorWeights {
        hidden_size: 8,
        head_dim: 16,
        compress_ratio: 1,
        eps: 1e-20,
        kv_weight: vec![0.25; 128],
        gate_weight: None,
        norm_weight: vec![1.0; 16],
    };
    let mut module = DeepseekV41Compressor::new(device, weights, 1, 4)?;
    let input = crate::GpuBuffer::from_f32(&foreign, &[1.0; 8])?;
    let err = module
        .prefill_from_buffer(&input, 1, 1)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("another Vulkan device"));
    Ok(())
}
