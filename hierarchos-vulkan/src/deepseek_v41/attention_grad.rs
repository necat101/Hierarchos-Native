//! FP32 attention-stage adjoints. These operate on actual projection/cache
//! operands, with frozen sink and fixed discrete index selection. They do not
//! enable a model trainer or define gradients for packed activation quantizers.

use super::attention::DeepseekV41SparseAttentionParams;
use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

#[cfg(test)]
#[path = "attention_grad_tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupedGradPush {
    rows: u32,
    groups: u32,
    width: u32,
    rank: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CacheGradPush {
    batch: u32,
    rows: u32,
    capacity: u32,
    dim: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WeightPush {
    rows: u32,
    batch: u32,
    seq: u32,
    heads: u32,
    dim: u32,
    window_capacity: u32,
    window_topk: u32,
    compressed_topk: u32,
    compressed_capacity: u32,
    scale: f32,
}

pub struct DeepseekV41SparseAttentionGradients {
    pub grad_query: GpuBuffer,
    pub grad_window: GpuBuffer,
    pub grad_compressed: Option<GpuBuffer>,
    /// Gradient with respect to Q.K, including the forward softmax scale.
    pub grad_dot: GpuBuffer,
}
pub struct DeepseekV41CacheWriteGradients {
    pub grad_source: GpuBuffer,
    pub grad_previous: GpuBuffer,
}

pub struct DeepseekV41AttentionAdjoints {
    device: VulkanDevice,
    weights: vulkan::ComputeKernel,
    weight_grad: vulkan::ComputeKernel,
    score_grad: vulkan::ComputeKernel,
    query_grad: vulkan::ComputeKernel,
    gather_grad: vulkan::ComputeKernel,
    scatter_grad: vulkan::ComputeKernel,
    grouped_grad: vulkan::ComputeKernel,
    cache_grad: vulkan::ComputeKernel,
}

fn kernel(
    device: &VulkanDevice,
    bytes: &[u8],
    inputs: usize,
    outputs: usize,
    push: usize,
) -> Result<vulkan::ComputeKernel> {
    let mut access = vec![vulkan::BindingAccess::ReadOnly; inputs];
    access.extend(vec![vulkan::BindingAccess::WriteOnly; outputs]);
    vulkan::ComputeKernel::new_with_access(device, bytes, &access, push as u32)
}
fn product(shape: &[usize]) -> Result<usize> {
    let len = shape
        .iter()
        .try_fold(
            1usize,
            |n, &dim| {
                if dim == 0 {
                    None
                } else {
                    n.checked_mul(dim)
                }
            },
        )
        .context("V4.1 attention adjoint has zero/overflowing geometry")?;
    if len > u32::MAX as usize {
        bail!("V4.1 attention adjoint exceeds u32 addressing");
    }
    Ok(len)
}

impl DeepseekV41AttentionAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        let push = std::mem::size_of::<WeightPush>();
        Ok(Self {
            // The forward weight pass reads its own score scratch.
            weights: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_attention_weights.spv"),
                7,
                push as u32,
            )?,
            weight_grad: kernel(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_attention_weight_grad.spv"),
                5,
                1,
                push,
            )?,
            score_grad: kernel(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_attention_score_grad.spv"),
                2,
                1,
                push,
            )?,
            query_grad: kernel(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_attention_query_grad.spv"),
                5,
                1,
                push,
            )?,
            gather_grad: kernel(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_attention_gather_grad.spv"),
                4,
                1,
                push,
            )?,
            scatter_grad: kernel(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_attention_scatter_grad.spv"),
                3,
                2,
                push,
            )?,
            grouped_grad: kernel(
                &device,
                include_bytes!(
                    "../../shaders/transformer_deepseek_v41_grouped_output_input_grad.spv"
                ),
                2,
                1,
                std::mem::size_of::<GroupedGradPush>(),
            )?,
            cache_grad: kernel(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_cache_write_grad.spv"),
                2,
                2,
                std::mem::size_of::<CacheGradPush>(),
            )?,
            device,
        })
    }

    fn check(&self, buffer: &GpuBuffer, len: usize) -> Result<()> {
        if buffer.f32_capacity() < len || !buffer.belongs_to_device(&self.device) {
            bail!("V4.1 attention adjoint buffer geometry or device mismatch");
        }
        Ok(())
    }

    /// Gather/softmax/value VJP with deterministic duplicate-index scatter.
    /// Masked slots contribute no gradient. The frozen sink participates in
    /// the denominator; discrete top-k indices have no differentiation rule.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse(
        &self,
        query: &GpuBuffer,
        window: &GpuBuffer,
        compressed: Option<&GpuBuffer>,
        window_indices: &GpuBuffer,
        compressed_indices: &GpuBuffer,
        sink: &GpuBuffer,
        upstream: &GpuBuffer,
        heads: usize,
        dim: usize,
        params: DeepseekV41SparseAttentionParams,
    ) -> Result<DeepseekV41SparseAttentionGradients> {
        let p = params;
        if p.window_topk == 0
            || p.window_topk > p.window_rows
            || p.compressed_topk > p.compressed_capacity
            || (compressed.is_none() && (p.compressed_topk != 0 || p.compressed_capacity != 0))
            || (compressed.is_some() && p.compressed_capacity == 0)
            || p.window_rows
                .checked_add(p.compressed_capacity)
                .is_none_or(|n| n > i32::MAX as usize)
        {
            bail!("V4.1 sparse adjoint window/compressed geometry mismatch");
        }
        let rows = product(&[p.batch, p.seq])?;
        let query_len = product(&[rows, heads, dim])?;
        let window_len = product(&[p.batch, p.window_rows, dim])?;
        let total = p
            .window_topk
            .checked_add(p.compressed_topk)
            .context("V4.1 sparse slot overflow")?;
        let weights_len = product(&[rows, heads, total])?;
        let gathered_len = product(&[rows, total, dim])?;
        let cache_len = product(&[
            p.batch,
            p.window_rows
                .checked_add(p.compressed_capacity)
                .context("V4.1 cache size overflow")?,
            dim,
        ])?;
        if rows > 65_535 || heads > 65_535 {
            bail!("V4.1 sparse adjoint row/head dispatch overflow");
        }
        for (buffer, len) in [
            (query, query_len),
            (window, window_len),
            (upstream, query_len),
            (window_indices, product(&[rows, p.window_topk])?),
            (sink, heads),
        ] {
            self.check(buffer, len)?;
        }
        self.check(compressed_indices, rows * p.compressed_topk)?;
        let empty = GpuBuffer::zeros_f32(&self.device, 1)?;
        let compressed_len = if p.compressed_capacity == 0 {
            0
        } else {
            product(&[p.batch, p.compressed_capacity, dim])?
        };
        let compressed = compressed.unwrap_or(&empty);
        self.check(compressed, compressed_len)?;
        let weights = GpuBuffer::zeros_f32(&self.device, weights_len)?;
        let grad_weight = GpuBuffer::zeros_f32(&self.device, weights_len)?;
        let grad_dot = GpuBuffer::zeros_f32(&self.device, weights_len)?;
        let grad_gather = GpuBuffer::zeros_f32(&self.device, gathered_len)?;
        let grad_query = GpuBuffer::zeros_f32(&self.device, query_len)?;
        let grad_window = GpuBuffer::zeros_f32(&self.device, window_len)?;
        let grad_compressed = GpuBuffer::zeros_f32(&self.device, compressed_len.max(1))?;
        let push = WeightPush {
            rows: rows as u32,
            batch: p.batch as u32,
            seq: p.seq as u32,
            heads: heads as u32,
            dim: dim as u32,
            window_capacity: p.window_rows as u32,
            window_topk: p.window_topk as u32,
            compressed_topk: p.compressed_topk as u32,
            compressed_capacity: p.compressed_capacity as u32,
            scale: (dim as f64).powf(-0.5) as f32,
        };
        let bytes = bytemuck::bytes_of(&push);
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.weights.record_dispatch(
            &mut commands,
            &[
                query,
                window,
                compressed,
                window_indices,
                compressed_indices,
                sink,
                &weights,
            ],
            bytes,
            [1, heads as u32, rows as u32],
        )?;
        self.weight_grad.record_dispatch(
            &mut commands,
            &[
                window,
                compressed,
                window_indices,
                compressed_indices,
                upstream,
                &grad_weight,
            ],
            bytes,
            grid(weights_len),
        )?;
        self.score_grad.record_dispatch(
            &mut commands,
            &[&weights, &grad_weight, &grad_dot],
            bytes,
            [1, heads as u32, rows as u32],
        )?;
        self.query_grad.record_dispatch(
            &mut commands,
            &[
                window,
                compressed,
                window_indices,
                compressed_indices,
                &grad_dot,
                &grad_query,
            ],
            bytes,
            grid(query_len),
        )?;
        self.gather_grad.record_dispatch(
            &mut commands,
            &[query, upstream, &weights, &grad_dot, &grad_gather],
            bytes,
            grid(gathered_len),
        )?;
        self.scatter_grad.record_dispatch(
            &mut commands,
            &[
                window_indices,
                compressed_indices,
                &grad_gather,
                &grad_window,
                &grad_compressed,
            ],
            bytes,
            grid(cache_len),
        )?;
        commands.submit()?;
        Ok(DeepseekV41SparseAttentionGradients {
            grad_query,
            grad_window,
            grad_compressed: (compressed_len != 0).then_some(grad_compressed),
            grad_dot,
        })
    }

    /// Input adjoint of the frozen FP32 block-diagonal wo_a einsum. Packed
    /// activation quantization and BF16 training boundaries are separate gates.
    pub fn grouped_output(
        &self,
        upstream: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        groups: usize,
        width: usize,
        rank: usize,
    ) -> Result<GpuBuffer> {
        let len = product(&[rows, groups, width])?;
        self.check(upstream, product(&[rows, groups, rank])?)?;
        self.check(weight, product(&[groups, rank, width])?)?;
        let output = GpuBuffer::zeros_f32(&self.device, len)?;
        let push = GroupedGradPush {
            rows: rows as u32,
            groups: groups as u32,
            width: width as u32,
            rank: rank as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.grouped_grad.record_dispatch(
            &mut commands,
            &[upstream, weight, &output],
            bytemuck::bytes_of(&push),
            grid(len),
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Cache overwrite VJP. `slots` maps each source row to a destination row,
    /// repeated per batch; -1 skips a write. The last writer owns a repeated
    /// slot. Untouched cache rows retain gradients to the previous cache.
    pub fn cache_write(
        &self,
        upstream: &GpuBuffer,
        slots: &[i32],
        batch: usize,
        capacity: usize,
        dim: usize,
    ) -> Result<DeepseekV41CacheWriteGradients> {
        let source_len = product(&[batch, slots.len(), dim])?;
        let previous_len = product(&[batch, capacity, dim])?;
        if capacity > i32::MAX as usize
            || slots
                .iter()
                .any(|&slot| slot < -1 || slot >= capacity as i32)
        {
            bail!("V4.1 cache adjoint write slot is out of bounds");
        }
        self.check(upstream, previous_len)?;
        let slots_gpu = GpuBuffer::from_u32(
            &self.device,
            &slots.iter().map(|&slot| slot as u32).collect::<Vec<_>>(),
        )?;
        let grad_source = GpuBuffer::zeros_f32(&self.device, source_len)?;
        let grad_previous = GpuBuffer::zeros_f32(&self.device, previous_len)?;
        let push = CacheGradPush {
            batch: batch as u32,
            rows: slots.len() as u32,
            capacity: capacity as u32,
            dim: dim as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.cache_grad.record_dispatch(
            &mut commands,
            &[upstream, &slots_gpu, &grad_source, &grad_previous],
            bytemuck::bytes_of(&push),
            grid(source_len.max(previous_len)),
        )?;
        commands.submit()?;
        Ok(DeepseekV41CacheWriteGradients {
            grad_source,
            grad_previous,
        })
    }
}

fn grid(len: usize) -> [u32; 3] {
    let groups = len.div_ceil(64);
    [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1]
}
