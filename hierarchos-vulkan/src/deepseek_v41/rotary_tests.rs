use super::*;
use crate::GpuBuffer;
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
    dim: usize,
    max_positions: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    angles: Vec<f32>,
    cosine: Vec<f32>,
    sine: Vec<f32>,
    rows: usize,
    width: usize,
    position_base: usize,
    position_stride: usize,
    input: Vec<f32>,
    applications: Vec<Application>,
}

#[derive(Deserialize)]
struct Application {
    inverse: bool,
    expected: Vec<f32>,
}

fn strict(label: &str, actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label} shape mismatch");
    let mut worst = 0.0f64;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let drift = (a as f64 - b as f64).abs();
        assert!(
            a.is_finite() && b.is_finite() && drift < 2e-7,
            "{label}, value {index}: native={a:.10e}, reference={b:.10e}, drift={drift:.10e}"
        );
        worst = worst.max(drift);
    }
    worst
}

#[test]
fn deepseek_v41_rotary_matches_pinned_reference() -> Result<()> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_rotary.json"
    ))?;
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    assert_eq!(fixture.atol_exclusive, 2e-7);
    assert_eq!(fixture.rtol, 0.0);
    let device = crate::VulkanDevice::new()?;
    let mut worst_table = 0.0f64;
    let mut worst_apply = 0.0f64;
    let mut compared = 0usize;
    for case in fixture.cases {
        let config = DeepseekV41RotaryConfig {
            dim: case.dim,
            max_positions: case.max_positions,
            original_max_position_embeddings: case.original_seq_len,
            base: case.base,
            factor: case.factor,
            beta_fast: case.beta_fast,
            beta_slow: case.beta_slow,
        };
        let rotary = DeepseekV41Rotary::new(device.clone(), config)?;
        // Position angles are computed with the reference's FP32 order.
        for (index, (&actual, &expected)) in rotary.angles().iter().zip(&case.angles).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{} angle {index}: native={actual:.10e}, reference={expected:.10e}",
                case.name
            );
        }
        let (cosine, sine) = rotary.table();
        worst_table = worst_table.max(strict(
            &format!("{} cosine", case.name),
            cosine,
            &case.cosine,
        ));
        worst_table = worst_table.max(strict(&format!("{} sine", case.name), sine, &case.sine));

        // Native table end to end, then the same kernel with the reference's
        // exact table to isolate application arithmetic.
        let installed = DeepseekV41Rotary::from_table(
            device.clone(),
            config,
            case.cosine.clone(),
            case.sine.clone(),
        )?;
        for application in &case.applications {
            for (label, module) in [("generated", &rotary), ("installed", &installed)] {
                let output = module.apply(
                    &case.input,
                    case.rows,
                    case.width,
                    case.position_base,
                    case.position_stride,
                    application.inverse,
                )?;
                let values = output.read_f32(case.input.len())?;
                worst_apply = worst_apply.max(strict(
                    &format!(
                        "{} {label} inverse={} applied",
                        case.name, application.inverse
                    ),
                    &values,
                    &application.expected,
                ));
                compared += values.len();
            }
        }
        // The GPU-resident entry point must agree with the host path.
        let resident = crate::GpuBuffer::from_f32(&device, &case.input)?;
        let output = rotary.apply_buffer(
            &resident,
            case.rows,
            case.width,
            case.position_base,
            case.position_stride,
            false,
        )?;
        strict(
            &format!("{} resident", case.name),
            &output.read_f32(case.input.len())?,
            &case.applications[0].expected,
        );
    }
    assert!(compared > 0);
    eprintln!(
        "V4.1 rotary on {}: {compared} applied values, worst table drift {worst_table:.12e}, worst apply drift {worst_apply:.12e}",
        device.name()
    );
    Ok(())
}

#[test]
fn deepseek_v41_rotary_rejects_invalid_configuration_and_positions() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let invalid = DeepseekV41RotaryConfig {
        dim: 63,
        max_positions: 8,
        original_max_position_embeddings: 0,
        base: 10000.0,
        factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    assert!(DeepseekV41Rotary::new(device.clone(), invalid).is_err());
    let config = DeepseekV41RotaryConfig::sliding(64, 8, 10000.0);
    let rotary = DeepseekV41Rotary::new(device.clone(), config)?;
    let input = vec![0.5; 2 * 64];
    assert!(rotary.apply(&input, 2, 64, 7, 1, false).is_err());
    assert!(rotary.apply(&input, 2, 32, 0, 1, false).is_err());
    assert!(rotary.apply(&input[..127], 2, 64, 0, 1, false).is_err());
    assert!(rotary.apply(&[f32::NAN; 128], 2, 64, 0, 1, false).is_err());
    assert!(DeepseekV41Rotary::from_table(device, config, vec![1.0; 1], vec![0.0; 1]).is_err());
    Ok(())
}

/// Per-head rows past one dispatch axis tile on z: every sampled row must match
/// a single-row application at the same position bit for bit.
#[test]
fn deepseek_v41_rotary_tiles_rows_past_one_dispatch_axis() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let rotary = DeepseekV41Rotary::new(
        device.clone(),
        DeepseekV41RotaryConfig::compressed(16, 65_600, 10_000.0, 16, 1.0, 32.0, 1.0),
    )?;
    let (width, rows) = (24usize, 65_535 + 3);
    let mut values = Vec::with_capacity(rows * width);
    for i in 0..rows * width {
        values.push(((i * 37 % 101) as f32 - 50.0) / 50.0);
    }
    let input = GpuBuffer::from_f32(&device, &values)?;
    let tiled = rotary
        .apply_repeated_buffer(&input, rows, width, 1, rows, 0, 1, false)?
        .read_f32(rows * width)?;
    for row in [0usize, 65_534, 65_535, rows - 1] {
        let slice = GpuBuffer::from_f32(&device, &values[row * width..(row + 1) * width])?;
        let single = rotary
            .apply_repeated_buffer(&slice, 1, width, 1, 1, row, 1, false)?
            .read_f32(width)?;
        assert_eq!(
            tiled[row * width..(row + 1) * width]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "row {row}"
        );
    }
    Ok(())
}
