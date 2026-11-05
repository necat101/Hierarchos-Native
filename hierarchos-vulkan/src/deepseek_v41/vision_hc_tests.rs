use crate::vulkan::{ComputeBatch, ComputeKernel, GpuBuffer, VulkanDevice};
use anyhow::Result;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    rows: u32,
    hc: u32,
    dim: u32,
    gradient: Vec<f32>,
    expected_bits: Vec<u32>,
}

#[test]
fn deepseek_v41_modality_hc_repeat_adjoint_matches_frozen_words() -> Result<()> {
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_hc_adjoint.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "afe782a47ac9d781be69ea13349c2011af6e733f349fb97e111e1469a00298ac"
    );
    let corpus: Corpus = serde_json::from_slice(bytes)?;
    assert_eq!(corpus.cases.len(), 6);
    let device = VulkanDevice::new()?;
    let kernel = ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_hc_reduce.spv"),
        2,
        12,
    )?;
    let mut compared = 0;
    for case in corpus.cases {
        let input = GpuBuffer::from_f32(&device, &case.gradient)?;
        let n = (case.rows * case.dim) as usize;
        assert_eq!(case.gradient.len(), n * case.hc as usize);
        assert_eq!(case.expected_bits.len(), n);
        let output = GpuBuffer::zeros_f32(&device, n)?;
        let mut commands = ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &output],
            bytemuck::cast_slice(&[case.rows, case.hc, case.dim]),
            [(n as u32).div_ceil(256), 1, 1],
        )?;
        commands.submit()?;
        for (i, (actual, expected)) in output
            .read_f32(n)?
            .iter()
            .zip(&case.expected_bits)
            .enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                *expected,
                "HC={} D={} word {i}",
                case.hc,
                case.dim
            );
        }
        compared += n;
    }
    assert_eq!(compared, 1664);
    eprintln!("V4.1 released HC-repeat adjoint: {compared} frozen CPU words exact");
    Ok(())
}
