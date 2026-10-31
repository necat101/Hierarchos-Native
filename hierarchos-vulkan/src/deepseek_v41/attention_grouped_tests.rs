use super::*;
use crate::GpuBuffer;

#[test]
fn deepseek_v41_pinned_sparse_reduction_matches_cpu_production_shapes() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Step {
        name: String,
        q_shape: Vec<usize>,
        kv_shape: Vec<usize>,
        indices_shape: Vec<usize>,
        q_bits: Vec<u32>,
        kv_bits: Vec<u32>,
        sink_bits: Vec<u32>,
        indices: Vec<i32>,
        weight_bits: Vec<u32>,
        raw_bits: Vec<u32>,
        rounded_bits: Vec<u32>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        fixture_sha256: String,
        steps: Vec<Step>,
    }
    use sha2::{Digest, Sha256};
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_production_attention.json"
    ))?;
    assert_eq!(
        f.fixture_sha256,
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../validation/fixtures/deepseek_v41_production_model.json"
            ))
        )
    );
    assert_eq!(f.steps.len(), 2);
    let device = crate::VulkanDevice::new()?;
    for step in f.steps {
        assert_eq!(step.q_shape, vec![1, 1, 64, 512]);
        assert_eq!(step.indices_shape, vec![1, 1, step.kv_shape[1]]);
        let attention = DeepseekV41Attention::new(
            device.clone(),
            DeepseekV41AttentionConfig {
                heads: 64,
                head_dim: 512,
                hidden_size: 32,
                q_lora_rank: 32,
                o_groups: 8,
                o_lora_rank: 32,
                window_size: 128,
                compress_ratio: 0,
                norm_eps: 1e-20,
                rotary: DeepseekV41RotaryConfig::compressed(64, 8, 10000.0, 0, 1.0, 32.0, 1.0),
            },
            DeepseekV41AttentionWeights {
                wq_a: vec![0.0; 32 * 32],
                q_norm: vec![1.0; 32],
                wq_b: vec![0.0; 64 * 512 * 32],
                wkv: vec![0.0; 512 * 32],
                kv_norm: vec![1.0; 512],
                wo_a: vec![0.0; 8 * 32 * 4096],
                wo_b: vec![0.0; 32 * 8 * 32],
                attn_sink: step.sink_bits.iter().map(|&b| f32::from_bits(b)).collect(),
            },
            1,
        )?;
        let indices: Vec<u32> = step.indices.iter().map(|&i| i as u32).collect();
        let raw = attention.gathered_attention(
            &GpuBuffer::from_u32(&device, &step.q_bits)?,
            &GpuBuffer::from_u32(&device, &step.kv_bits)?,
            None,
            &GpuBuffer::from_u32(&device, &indices)?,
            &GpuBuffer::zeros_u32(&device, 1)?,
            DeepseekV41SparseAttentionParams {
                batch: 1,
                seq: 1,
                window_rows: step.kv_shape[1],
                window_topk: step.indices.len(),
                compressed_topk: 0,
                compressed_capacity: 0,
            },
        )?;
        assert_eq!(
            raw.read_u32(step.raw_bits.len())?,
            step.raw_bits,
            "{} FP32 raw",
            step.name
        );
        assert_eq!(
            attention
                .weights_trace
                .borrow()
                .as_ref()
                .unwrap()
                .read_u32(step.weight_bits.len())?,
            step.weight_bits,
            "{} weights",
            step.name
        );
        let cast = DeepseekV41MixedLinearKernel::new(device.clone())?;
        let rounded =
            cast.cast_buffer(&raw, step.rounded_bits.len(), DeepseekV41WeightOutput::Bf16)?;
        assert_eq!(
            rounded.read_u32(step.rounded_bits.len())?,
            step.rounded_bits,
            "{} BF16",
            step.name
        );
    }
    Ok(())
}

#[test]
fn deepseek_v41_bf16_grouped_decode_reduction_matches_frozen_cpu() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Row {
        group: usize,
        row: usize,
        bits: Vec<u32>,
    }
    #[derive(serde::Deserialize)]
    struct Step {
        name: String,
        input_bits: Vec<u32>,
        expected_bits: Vec<u32>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        policy: String,
        parent_fixture_sha256: String,
        groups: usize,
        group_width: usize,
        output_rank: usize,
        weight_rows: Vec<Row>,
        steps: Vec<Step>,
    }
    use sha2::{Digest, Sha256};
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_grouped_reduction.json"
    ))?;
    assert_eq!(f.policy, "bf16_short_grouped_fp32_k512_v1");
    assert_eq!(
        f.parent_fixture_sha256,
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../validation/fixtures/deepseek_v41_production_model.json"
            ))
        )
    );
    assert_eq!((f.groups, f.group_width, f.output_rank), (3, 4096, 1024));
    assert_eq!(f.weight_rows.len(), 9);
    assert_eq!(f.steps.len(), 2);
    let mut weights = vec![0.0f32; f.groups * f.output_rank * f.group_width];
    for row in &f.weight_rows {
        assert!(row.group < f.groups && row.row < f.output_rank);
        assert_eq!(row.bits.len(), f.group_width);
        for (i, &b) in row.bits.iter().enumerate() {
            assert_eq!(b & 0xffff, 0);
            weights[(row.group * f.output_rank + row.row) * f.group_width + i] = f32::from_bits(b);
        }
    }
    let device = crate::VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        GROUPED_OUTPUT,
        3,
        std::mem::size_of::<GroupedPush>() as u32,
    )?;
    let weights = GpuBuffer::from_f32(&device, &weights)?;
    let mixed = DeepseekV41MixedLinearKernel::new(device.clone())?;
    let mut legacy_mismatches = 0;
    for step in f.steps {
        assert_eq!(step.input_bits.len(), f.groups * f.group_width);
        assert_eq!(step.expected_bits.len(), f.groups * f.output_rank);
        let input = GpuBuffer::from_u32(&device, &step.input_bits)?;
        for bf16 in [1, 0] {
            let output = GpuBuffer::zeros_f32(&device, step.expected_bits.len())?;
            let push = GroupedPush {
                rows: 1,
                groups: f.groups as u32,
                group_width: f.group_width as u32,
                output_rank: f.output_rank as u32,
                bf16,
            };
            let mut commands = vulkan::ComputeBatch::new(&device)?;
            kernel.record_dispatch(
                &mut commands,
                &[&input, &weights, &output],
                bytemuck::bytes_of(&push),
                [f.output_rank.div_ceil(64) as u32, f.groups as u32, 1],
            )?;
            commands.submit()?;
            let output = mixed.cast_buffer(
                &output,
                step.expected_bits.len(),
                DeepseekV41WeightOutput::Bf16,
            )?;
            let actual = output.read_u32(step.expected_bits.len())?;
            if bf16 == 1 {
                assert_eq!(actual, step.expected_bits, "{}", step.name);
            } else {
                legacy_mismatches += actual
                    .iter()
                    .zip(&step.expected_bits)
                    .filter(|(a, b)| a != b)
                    .count();
            }
        }
    }
    assert!(
        legacy_mismatches > 0,
        "regression must detect the former four-lane BF16 path"
    );
    Ok(())
}
