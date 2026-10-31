use super::*;
use crate::deepseek_v41::{DeepseekV41Rotary, DeepseekV41RotaryConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    tolerance: f64,
    rtol: f64,
    comparison: String,
    sparse: Vec<Sparse>,
    grouped: Vec<Grouped>,
    rotary: Vec<Rotary>,
    cache: Vec<Cache>,
}
#[derive(Deserialize)]
struct Sparse {
    name: String,
    batch: usize,
    seq: usize,
    heads: usize,
    dim: usize,
    window: usize,
    topk: usize,
    compressed: usize,
    capacity: usize,
    wi: Vec<i32>,
    ci: Vec<i32>,
    q_bits: Vec<u32>,
    w_bits: Vec<u32>,
    c_bits: Vec<u32>,
    sink_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    output_bits: Vec<u32>,
    grad_q_bits: Vec<u32>,
    grad_w_bits: Vec<u32>,
    grad_c_bits: Vec<u32>,
    grad_dot_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Grouped {
    rows: usize,
    groups: usize,
    width: usize,
    rank: usize,
    weight_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Rotary {
    batch: usize,
    seq: usize,
    heads: usize,
    width: usize,
    rd: usize,
    inverse: bool,
    cosine_bits: Vec<u32>,
    sine_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
}
#[derive(Deserialize)]
struct Cache {
    slots: Vec<i32>,
    batch: usize,
    capacity: usize,
    dim: usize,
    upstream_bits: Vec<u32>,
    grad_source_bits: Vec<u32>,
    grad_previous_bits: Vec<u32>,
}

fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_attention_grad.json"
    ))
    .unwrap();
    assert_eq!(f.tolerance, 2e-7);
    assert_eq!(f.rtol, 0.0);
    assert_eq!(f.comparison, "strict_less_than");
    f
}
fn upload(device: &VulkanDevice, bits: &[u32]) -> Result<GpuBuffer> {
    if bits.is_empty() {
        GpuBuffer::zeros_u32(device, 1)
    } else {
        GpuBuffer::from_u32(device, bits)
    }
}
fn indices(device: &VulkanDevice, values: &[i32]) -> Result<GpuBuffer> {
    upload(
        device,
        &values.iter().map(|&v| v as u32).collect::<Vec<_>>(),
    )
}
fn gate(
    label: &str,
    buffer: &GpuBuffer,
    expected: &[u32],
    failures: &mut Vec<String>,
) -> Result<()> {
    let actual = buffer.read_f32(expected.len())?;
    let mut worst = 0.0f64;
    let mut at = 0;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let e = f32::from_bits(e);
        assert!(
            a.is_finite() && e.is_finite(),
            "{label} non-finite at {index}"
        );
        let drift = (a as f64 - e as f64).abs();
        if drift > worst {
            worst = drift;
            at = index;
        }
    }
    eprintln!(
        "V4.1 attention adjoint {label}: {} values, worst {worst:.17e} at {at}",
        expected.len()
    );
    if worst >= 2e-7 {
        failures.push(format!(
            "{label}: {worst:.17e} at {at}, native={} cpu={}",
            actual[at],
            f32::from_bits(expected[at])
        ));
    }
    Ok(())
}

#[test]
fn deepseek_v41_sparse_attention_adjoints_cpu_pytorch_strict() -> Result<()> {
    let f = fixture();
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41AttentionAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in f.sparse {
        let q = upload(&device, &c.q_bits)?;
        let w = upload(&device, &c.w_bits)?;
        let compressed = upload(&device, &c.c_bits)?;
        let wi = indices(&device, &c.wi)?;
        let ci = indices(&device, &c.ci)?;
        let sink = upload(&device, &c.sink_bits)?;
        let g = upload(&device, &c.upstream_bits)?;
        let params = DeepseekV41SparseAttentionParams {
            batch: c.batch,
            seq: c.seq,
            window_rows: c.window,
            window_topk: c.topk,
            compressed_topk: c.compressed,
            compressed_capacity: c.capacity,
        };
        let gradients = kernels.sparse(
            &q,
            &w,
            if c.capacity == 0 {
                None
            } else {
                Some(&compressed)
            },
            &wi,
            &ci,
            &sink,
            &g,
            c.heads,
            c.dim,
            params,
        )?;
        gate(
            &format!("{} Q", c.name),
            &gradients.grad_query,
            &c.grad_q_bits,
            &mut failures,
        )?;
        gate(
            &format!("{} window KV", c.name),
            &gradients.grad_window,
            &c.grad_w_bits,
            &mut failures,
        )?;
        gate(
            &format!("{} Q.K", c.name),
            &gradients.grad_dot,
            &c.grad_dot_bits,
            &mut failures,
        )?;
        if let Some(grad) = gradients.grad_compressed {
            gate(
                &format!("{} compressed KV", c.name),
                &grad,
                &c.grad_c_bits,
                &mut failures,
            )?;
        } else {
            assert!(c.grad_c_bits.is_empty());
        }
        // Independently check that the very same native weight/value operands
        // also produce the pinned forward result; no stale probability tape.
        let weights =
            GpuBuffer::zeros_f32(&device, c.batch * c.seq * c.heads * (c.topk + c.compressed))?;
        let output = GpuBuffer::zeros_f32(&device, c.q_bits.len())?;
        let push = WeightPush {
            rows: (c.batch * c.seq) as u32,
            batch: c.batch as u32,
            seq: c.seq as u32,
            heads: c.heads as u32,
            dim: c.dim as u32,
            window_capacity: c.window as u32,
            window_topk: c.topk as u32,
            compressed_topk: c.compressed as u32,
            compressed_capacity: c.capacity as u32,
            scale: (c.dim as f64).powf(-0.5) as f32,
        };
        let values = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_values.spv"),
            6,
            32,
        )?;
        let vp = [
            push.rows,
            push.seq,
            push.heads,
            push.dim,
            push.window_capacity,
            push.window_topk,
            push.compressed_topk,
            push.compressed_capacity,
        ];
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernels.weights.record_dispatch(
            &mut commands,
            &[&q, &w, &compressed, &wi, &ci, &sink, &weights],
            bytemuck::bytes_of(&push),
            [1, c.heads as u32, push.rows],
        )?;
        values.record_dispatch(
            &mut commands,
            &[&w, &compressed, &wi, &ci, &weights, &output],
            bytemuck::cast_slice(&vp),
            [c.dim.div_ceil(64) as u32, c.heads as u32, push.rows],
        )?;
        commands.submit()?;
        gate(
            &format!("{} forward", c.name),
            &output,
            &c.output_bits,
            &mut failures,
        )?;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn deepseek_v41_grouped_output_adjoints_cpu_pytorch_strict() -> Result<()> {
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41AttentionAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().grouped {
        let grad = kernels.grouped_output(
            &upload(&device, &c.upstream_bits)?,
            &upload(&device, &c.weight_bits)?,
            c.rows,
            c.groups,
            c.width,
            c.rank,
        )?;
        gate(
            &format!(
                "grouped {}x{}x{} rank {}",
                c.rows, c.groups, c.width, c.rank
            ),
            &grad,
            &c.grad_x_bits,
            &mut failures,
        )?;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn deepseek_v41_rotary_adjoints_cpu_pytorch_strict() -> Result<()> {
    let device = VulkanDevice::new()?;
    let mut failures = Vec::new();
    for c in fixture().rotary {
        let rotary = DeepseekV41Rotary::from_table(
            device.clone(),
            DeepseekV41RotaryConfig::sliding(c.rd, 32, 10000.0),
            c.cosine_bits.iter().map(|&b| f32::from_bits(b)).collect(),
            c.sine_bits.iter().map(|&b| f32::from_bits(b)).collect(),
        )?;
        let grad = rotary.input_gradient_repeated(
            &upload(&device, &c.upstream_bits)?,
            c.batch * c.seq * c.heads,
            c.width,
            c.heads,
            c.seq,
            1,
            2,
            c.inverse,
        )?;
        gate(
            &format!("RoPE inverse={}", c.inverse),
            &grad,
            &c.grad_x_bits,
            &mut failures,
        )?;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn deepseek_v41_cache_write_adjoints_cpu_pytorch_strict() -> Result<()> {
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41AttentionAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().cache {
        let grad = kernels.cache_write(
            &upload(&device, &c.upstream_bits)?,
            &c.slots,
            c.batch,
            c.capacity,
            c.dim,
        )?;
        gate(
            "cache source",
            &grad.grad_source,
            &c.grad_source_bits,
            &mut failures,
        )?;
        gate(
            "cache previous",
            &grad.grad_previous,
            &c.grad_previous_bits,
            &mut failures,
        )?;
    }
    let g = GpuBuffer::zeros_f32(&device, 8)?;
    assert!(kernels.cache_write(&g, &[4], 1, 4, 2).is_err());
    assert!(kernels.cache_write(&g, &[-2], 1, 4, 2).is_err());
    assert!(kernels.cache_write(&g, &[], 1, 4, 2).is_err());
    assert!(kernels.grouped_output(&g, &g, 1, 0, 2, 2).is_err());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn deepseek_v41_attention_adjoint_crosses_dispatch_axis_bound() -> Result<()> {
    let device = VulkanDevice::new()?;
    let kernels = DeepseekV41AttentionAdjoints::new(device.clone())?;
    let width = 65_535 * 64 + 17;
    let weights: Vec<_> = (0..width)
        .map(|i| 0.125 + (i % 257) as f32 / 256.0)
        .collect();
    let gradient = kernels.grouped_output(
        &GpuBuffer::from_f32(&device, &[1.0])?,
        &GpuBuffer::from_f32(&device, &weights)?,
        1,
        1,
        width,
        1,
    )?;
    assert_eq!(gradient.read_f32(width)?, weights);
    assert_eq!(grid(width), [65_535, 2, 1]);
    assert!(product(&[u32::MAX as usize, 2]).is_err());
    Ok(())
}
