use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    revision: String,
    exact: bool,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    cols: usize,
    block_size: usize,
    mode: String,
    scale_mode: String,
    input: Vec<f32>,
    expected_values: Vec<f32>,
    expected_scale_bits: Vec<u32>,
}

fn parse_mode(value: &str) -> DeepseekV41QuantMode {
    match value {
        "fp8" => DeepseekV41QuantMode::Fp8E4M3,
        "fp4" => DeepseekV41QuantMode::Fp4E2M1,
        other => panic!("unknown fixture quantization mode {other}"),
    }
}

fn parse_scale(value: &str) -> DeepseekV41ScaleMode {
    match value {
        "ue8m0" => DeepseekV41ScaleMode::Ue8m0,
        "e4m3" => DeepseekV41ScaleMode::E4m3,
        other => panic!("unknown fixture scale mode {other}"),
    }
}

fn assert_exact_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} shape mismatch");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}, value {index}: native={a:.10e} ({:#010x}), reference={b:.10e} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn deepseek_v41_fp4_midpoints_match_cpu_for_all_positive_e4m3_scales() -> Result<()> {
    #[derive(Deserialize)]
    struct Midpoints {
        revision: String,
        rows: usize,
        cols: usize,
        input: Vec<f32>,
        expected_bits: Vec<u32>,
        expected_scale_bits: Vec<u32>,
    }
    let f: Midpoints = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_bf16_quantization.json"
    ))?;
    assert_eq!(f.revision, REFERENCE_REVISION);
    assert_eq!((f.rows, f.cols), (126, 16));
    let device = crate::VulkanDevice::new()?;
    let q = DeepseekV41ActivationQuantizer::new(device)?.quantize(
        &f.input,
        f.rows,
        f.cols,
        16,
        DeepseekV41QuantMode::Fp4E2M1,
        DeepseekV41ScaleMode::E4m3,
    )?;
    assert_eq!(q.read_scale_bits()?, f.expected_scale_bits);
    assert_eq!(q.values.read_u32(f.rows * f.cols)?, f.expected_bits);
    Ok(())
}

#[test]
fn deepseek_v41_quantization_matches_released_kernels_bit_exactly() -> Result<()> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_quantization.json"
    ))?;
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    assert!(fixture.exact);
    let device = crate::VulkanDevice::new()?;
    let quantizer = DeepseekV41ActivationQuantizer::new(device.clone())?;
    let mut compared = 0usize;
    let case_count = fixture.cases.len();
    for case in fixture.cases {
        let result = quantizer.quantize(
            &case.input,
            case.rows,
            case.cols,
            case.block_size,
            parse_mode(&case.mode),
            parse_scale(&case.scale_mode),
        )?;
        let values = result.read_values()?;
        assert_exact_bits(&case.name, &values, &case.expected_values);
        assert_eq!(
            result.read_scale_bits()?,
            case.expected_scale_bits,
            "{} scale bits",
            case.name
        );
        compared += values.len();

        // The same contract must hold through the GPU-resident entry point.
        let resident = crate::GpuBuffer::from_f32(&device, &case.input)?;
        let resident_result = quantizer.quantize_buffer(
            &resident,
            case.rows,
            case.cols,
            case.block_size,
            parse_mode(&case.mode),
            parse_scale(&case.scale_mode),
        )?;
        assert_exact_bits(
            &format!("{} resident", case.name),
            &resident_result.read_values()?,
            &case.expected_values,
        );
    }
    assert!(compared > 0);
    eprintln!(
        "V4.1 quantization on {}: {compared} exact values across {case_count} cases",
        device.name()
    );
    Ok(())
}

#[test]
fn deepseek_v41_quantization_rejects_invalid_shapes_and_modes() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let quantizer = DeepseekV41ActivationQuantizer::new(device.clone())?;
    assert!(quantizer
        .quantize(
            &[0.0; 7],
            1,
            7,
            16,
            DeepseekV41QuantMode::Fp4E2M1,
            DeepseekV41ScaleMode::Ue8m0
        )
        .is_err());
    assert!(quantizer
        .quantize(
            &[0.0; 16],
            0,
            16,
            16,
            DeepseekV41QuantMode::Fp4E2M1,
            DeepseekV41ScaleMode::Ue8m0
        )
        .is_err());
    assert!(quantizer
        .quantize(
            &[f32::NAN; 16],
            1,
            16,
            16,
            DeepseekV41QuantMode::Fp4E2M1,
            DeepseekV41ScaleMode::Ue8m0
        )
        .is_err());
    assert!(quantizer
        .quantize(
            &[0.0; 16],
            1,
            16,
            16,
            DeepseekV41QuantMode::Fp8E4M3,
            DeepseekV41ScaleMode::E4m3
        )
        .is_err());
    let foreign = crate::VulkanDevice::new()?;
    let input = crate::GpuBuffer::from_f32(&foreign, &[0.0; 16])?;
    let err = quantizer
        .quantize_buffer(
            &input,
            1,
            16,
            16,
            DeepseekV41QuantMode::Fp4E2M1,
            DeepseekV41ScaleMode::Ue8m0,
        )
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("another Vulkan device"), "{err}");
    Ok(())
}
