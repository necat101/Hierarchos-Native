use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

const UNARY_SPV: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_unary_test.spv");
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct V41UnaryPush {
    elements: u32,
    op: u32,
}
#[derive(Deserialize)]
struct Fixture {
    torch_revision: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    op: u32,
    inputs: Vec<f32>,
    expected_bits: Vec<u32>,
    #[serde(default)]
    max_ulps: u32,
}

#[test]
fn deepseek_v41_unary_matches_independent_scalar_and_vector_tail_references() -> Result<()> {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_unary.json"
    ))?;
    assert_eq!(f.torch_revision, "5c4886908584029761b579af026dcfb627c84070");
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        UNARY_SPV,
        3,
        std::mem::size_of::<V41UnaryPush>() as u32,
    )?;
    let mut count = 0;
    let seeds = crate::deepseek_v41::cpu_sqrt_seeds(&device)?;
    let mut exact = 0;
    let mut worst_ulps = 0;
    for c in f.cases {
        let input = GpuBuffer::from_f32(&device, &c.inputs)?;
        let output = GpuBuffer::zeros_f32(&device, c.inputs.len())?;
        let push = V41UnaryPush {
            elements: c.inputs.len() as u32,
            op: c.op,
        };
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &output, &seeds],
            bytemuck::bytes_of(&push),
            [c.inputs.len().div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        let actual = output.read_u32(c.inputs.len())?;
        assert_eq!(actual.len(), c.expected_bits.len());
        for (i, (&a, &b)) in actual.iter().zip(&c.expected_bits).enumerate() {
            let ulps = (a as i64 - b as i64).unsigned_abs();
            assert!(
                ulps <= u64::from(c.max_ulps),
                "{} index {i}, input {}: {} versus {}",
                c.name,
                c.inputs[i],
                f32::from_bits(a),
                f32::from_bits(b)
            );
            exact += usize::from(a == b);
            worst_ulps = worst_ulps.max(ulps);
        }
        count += actual.len();
    }
    eprintln!(
        "V4.1 scalar/vector unary: {exact}/{count} exact values, worst {worst_ulps} ULP on {}",
        device.name()
    );
    Ok(())
}

#[test]
fn deepseek_v41_cpu_sqrt_seed_and_staging_match_cpu() -> Result<()> {
    #[derive(Deserialize)]
    struct CpuMath {
        policy: String,
        seed_table_sha256: String,
        seed_count: usize,
        input_bits: Vec<u32>,
        expected_bits: Vec<u32>,
    }
    use sha2::{Digest, Sha256};
    let fixture: CpuMath = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_cpu_math.json"
    ))?;
    assert_eq!(fixture.policy, "windows_mkl_sse_rsqrt_refine_v1");
    assert_eq!(fixture.seed_count, 8192);
    assert_eq!(
        fixture.seed_table_sha256,
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../shaders/deepseek_v41_rsqrt_seed_amd_zen4.bin"
            ))
        )
    );
    assert_eq!(fixture.input_bits.len(), 4104);
    assert_eq!(fixture.expected_bits.len(), 4104);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        UNARY_SPV,
        3,
        std::mem::size_of::<V41UnaryPush>() as u32,
    )?;
    let input = GpuBuffer::from_u32(&device, &fixture.input_bits)?;
    let output = GpuBuffer::zeros_u32(&device, fixture.input_bits.len())?;
    let seeds = crate::deepseek_v41::cpu_sqrt_seeds(&device)?;
    let push = V41UnaryPush {
        elements: fixture.input_bits.len() as u32,
        op: 5,
    };
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    kernel.record_dispatch(
        &mut commands,
        &[&input, &output, &seeds],
        bytemuck::bytes_of(&push),
        [fixture.input_bits.len().div_ceil(64) as u32, 1, 1],
    )?;
    commands.submit()?;
    assert_eq!(
        output.read_u32(fixture.expected_bits.len())?,
        fixture.expected_bits
    );
    Ok(())
}

#[test]
fn deepseek_v41_cpu_scalar_log_staging_matches_cpu() -> Result<()> {
    #[derive(Deserialize)]
    struct CpuLog {
        policy: String,
        table_sha256: String,
        input_bits: Vec<u32>,
        logf_bits: Vec<u32>,
        log1pf_bits: Vec<u32>,
    }
    use sha2::{Digest, Sha256};
    let fixture: CpuLog = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_cpu_log.json"
    ))?;
    assert_eq!(fixture.policy, "windows_crt_scalar_log_fma_v1");
    assert_eq!(
        fixture.table_sha256,
        format!(
            "{:x}",
            Sha256::digest(
                include_str!("../../shaders/fp32_v41_cpu_log_table.glsl")
                    .replace("\r\n", "\n")
                    .as_bytes()
            )
        )
    );
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        UNARY_SPV,
        3,
        std::mem::size_of::<V41UnaryPush>() as u32,
    )?;
    let input = GpuBuffer::from_u32(&device, &fixture.input_bits)?;
    let seeds = crate::deepseek_v41::cpu_sqrt_seeds(&device)?;
    for (op, expected) in [(7, fixture.logf_bits), (8, fixture.log1pf_bits)] {
        assert_eq!(expected.len(), fixture.input_bits.len());
        assert!(!expected.is_empty());
        let output = GpuBuffer::zeros_u32(&device, expected.len())?;
        let push = V41UnaryPush {
            elements: expected.len() as u32,
            op,
        };
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &output, &seeds],
            bytemuck::bytes_of(&push),
            [expected.len().div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        let actual = output.read_u32(expected.len())?;
        let wrong = actual.iter().zip(&expected).position(|(a, b)| a != b);
        assert!(wrong.is_none(), "log op {op} first mismatch {wrong:?}");
    }
    Ok(())
}
