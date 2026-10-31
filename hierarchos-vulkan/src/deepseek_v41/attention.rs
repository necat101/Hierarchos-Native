//! DeepSeek V4.1 attention. Combines the sliding-window KV ring with the
//! compressed-KV cache and the fine indexer's selected positions, then applies
//! the attention sink, inverse output rotation, and the grouped output
//! projection. Typed projections retain FP8/FP4 storage and explicit BF16
//! boundaries. CPU PyTorch qualifies the measured component/hybrid cases;
//! production geometry and the complete BF16 residual graph remain separate.
//! The retained sparse/cache tape qualifies shared-source and cross-step VJPs
//! on post-quantization operands. Retained input/output projection tapes compose
//! those with normalization, rotary and mixed-storage PEFT projection VJPs.
//! Compressor/finalizer and complete model training integration remain separate.

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

use super::checkpoint::DeepseekV41StreamedLayer;
use super::indexer::{CacheWriter, NormKernel};
use super::{
    DeepseekV41ActivationQuantizer, DeepseekV41Checkpoint, DeepseekV41IndexSelection,
    DeepseekV41LinearWeights, DeepseekV41MixedLinearKernel, DeepseekV41QuantMode,
    DeepseekV41QuantizedActivation, DeepseekV41ResidentLinear, DeepseekV41Rotary,
    DeepseekV41RotaryConfig, DeepseekV41ScaleMode, DeepseekV41WeightOutput,
};

const WINDOW_CACHE: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_window_cache.spv");
const ATTENTION_WEIGHTS: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_attention_weights.spv");
const ATTENTION_VALUES: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_attention_values.spv");
const GROUPED_OUTPUT: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output.spv");

const KV_QUANT_BLOCK: usize = 32;

#[path = "attention_projection_grad.rs"]
mod projection_grad;
pub use projection_grad::{
    DeepseekV41AttentionInputTape, DeepseekV41AttentionOutputTape,
    DeepseekV41AttentionProjectionAdjoints, DeepseekV41AttentionProjectionGradients,
};

#[path = "attention_tape.rs"]
mod tape;
pub use tape::{
    DeepseekV41AttentionCacheId, DeepseekV41AttentionRead, DeepseekV41AttentionReadId,
    DeepseekV41AttentionReverseTape, DeepseekV41AttentionTape, DeepseekV41AttentionTapeGradients,
    DeepseekV41AttentionWindowView, DeepseekV41AttentionWriteId,
};

#[cfg(test)]
#[path = "attention_grouped_tests.rs"]
mod grouped_tests;

#[cfg(test)]
#[path = "attention_schedule_tests.rs"]
mod schedule_tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WindowPush {
    batch: u32,
    seq: u32,
    dim: u32,
    window: u32,
    start_pos: u32,
    decode: u32,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ValuePush {
    rows: u32,
    seq: u32,
    heads: u32,
    dim: u32,
    window_capacity: u32,
    window_topk: u32,
    compressed_topk: u32,
    compressed_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupedPush {
    rows: u32,
    groups: u32,
    group_width: u32,
    output_rank: u32,
    bf16: u32,
}

/// Geometry of one gathered sparse-attention call. `window_rows` is the number
/// of KV rows the window indices address (the chunk length during prefill, the
/// ring size during decode) and is also the offset the compressed indices were
/// produced with.
#[derive(Clone, Copy, Debug)]
pub struct DeepseekV41SparseAttentionParams {
    pub batch: usize,
    pub seq: usize,
    pub window_rows: usize,
    pub window_topk: usize,
    pub compressed_topk: usize,
    pub compressed_capacity: usize,
}

pub struct DeepseekV41WindowKvCache {
    device: VulkanDevice,
    batch: usize,
    window: usize,
    head_dim: usize,
    values: GpuBuffer,
    kernel: vulkan::ComputeKernel,
}

impl DeepseekV41WindowKvCache {
    fn tape_alias(&self) -> Result<Self> {
        Ok(Self {
            device: self.device.clone(),
            batch: self.batch,
            window: self.window,
            head_dim: self.head_dim,
            values: self.values.clone(),
            kernel: vulkan::ComputeKernel::new(
                &self.device,
                WINDOW_CACHE,
                2,
                std::mem::size_of::<WindowPush>() as u32,
            )?,
        })
    }
    pub fn new(device: VulkanDevice, batch: usize, window: usize, head_dim: usize) -> Result<Self> {
        if batch == 0 || window == 0 || head_dim == 0 {
            bail!("V4.1 window KV cache dimensions must be positive");
        }
        let elements = batch
            .checked_mul(window)
            .and_then(|n| n.checked_mul(head_dim))
            .context("V4.1 window KV cache size overflow")?;
        if elements > u32::MAX as usize {
            bail!("V4.1 window KV cache exceeds u32 addressing");
        }
        Ok(Self {
            values: GpuBuffer::zeros_f32(&device, elements)?,
            kernel: vulkan::ComputeKernel::new(
                &device,
                WINDOW_CACHE,
                2,
                std::mem::size_of::<WindowPush>() as u32,
            )?,
            device,
            batch,
            window,
            head_dim,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn values(&self) -> &GpuBuffer {
        &self.values
    }

    /// Seeds the ring from a prefill chunk. `source` is `[batch, seq, head_dim]`.
    pub fn seed_prefill(&self, source: &GpuBuffer, seq: usize) -> Result<()> {
        if seq == 0
            || source.f32_capacity() < self.batch * seq * self.head_dim
            || !source.belongs_to_device(&self.device)
        {
            bail!("V4.1 window KV prefill source geometry or device mismatch");
        }
        self.dispatch(source, seq, 0, false)
    }

    /// Writes one decode token into slot `start_pos % window`.
    pub fn write_decode(&self, source: &GpuBuffer, start_pos: usize) -> Result<()> {
        if source.f32_capacity() < self.batch * self.head_dim
            || !source.belongs_to_device(&self.device)
            || start_pos > u32::MAX as usize
        {
            bail!("V4.1 window KV decode source geometry or position mismatch");
        }
        self.dispatch(source, 1, start_pos, true)
    }

    pub fn read_all(&self) -> Result<Vec<f32>> {
        self.values
            .read_f32(self.batch * self.window * self.head_dim)
    }

    fn dispatch(
        &self,
        source: &GpuBuffer,
        seq: usize,
        start_pos: usize,
        decode: bool,
    ) -> Result<()> {
        let push = WindowPush {
            batch: self.batch as u32,
            seq: seq as u32,
            dim: self.head_dim as u32,
            window: self.window as u32,
            start_pos: start_pos as u32,
            decode: u32::from(decode),
        };
        let mut commands = self.device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[source, &self.values],
            bytemuck::bytes_of(&push),
            [
                self.head_dim.div_ceil(64) as u32,
                (self.batch * self.window) as u32,
                1,
            ],
        )?;
        commands.submit()
    }
}

/// Shared compressed-KV cache written by the finalizer and read by every
/// attention layer that reuses its source.
pub struct DeepseekV41CompressedKvCache {
    device: VulkanDevice,
    batch: usize,
    capacity: usize,
    head_dim: usize,
    values: GpuBuffer,
    writer: CacheWriter,
}

impl DeepseekV41CompressedKvCache {
    fn tape_alias(&self) -> Result<Self> {
        Ok(Self {
            device: self.device.clone(),
            batch: self.batch,
            capacity: self.capacity,
            head_dim: self.head_dim,
            values: self.values.clone(),
            writer: CacheWriter::new(&self.device)?,
        })
    }
    pub fn new(
        device: VulkanDevice,
        batch: usize,
        capacity: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if batch == 0 || capacity == 0 || head_dim == 0 {
            bail!("V4.1 compressed KV cache dimensions must be positive");
        }
        let elements = batch
            .checked_mul(capacity)
            .and_then(|n| n.checked_mul(head_dim))
            .context("V4.1 compressed KV cache size overflow")?;
        if elements > u32::MAX as usize {
            bail!("V4.1 compressed KV cache exceeds u32 addressing");
        }
        Ok(Self {
            values: GpuBuffer::zeros_f32(&device, elements)?,
            writer: CacheWriter::new(&device)?,
            device,
            batch,
            capacity,
            head_dim,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn values(&self) -> &GpuBuffer {
        &self.values
    }

    /// Appends completed compressed groups. `activation` is
    /// `[batch * groups, head_dim]` already rotated and FP4-quantized.
    pub fn append_groups(
        &self,
        activation: &DeepseekV41QuantizedActivation,
        batch: usize,
        groups: usize,
        first_group: usize,
    ) -> Result<()> {
        if batch == 0 || batch > self.batch || groups == 0 {
            bail!("V4.1 compressed KV append dimensions are invalid");
        }
        if activation.rows != batch * groups
            || activation.cols != self.head_dim
            || activation.mode != DeepseekV41QuantMode::Fp4E2M1
            || activation.scale_mode != DeepseekV41ScaleMode::E4m3
        {
            bail!("V4.1 compressed KV append requires FP4/E4M3 finalized latents");
        }
        self.writer.append(
            &self.device,
            &activation.values,
            &self.values,
            activation.rows,
            self.head_dim,
            groups,
            first_group,
            self.capacity,
        )
    }

    pub fn read_batch(&self, batch_index: usize, groups: usize) -> Result<Vec<f32>> {
        if batch_index >= self.batch || groups > self.capacity {
            bail!("V4.1 compressed KV cache read exceeds its geometry");
        }
        let all = self
            .values
            .read_f32(self.batch * self.capacity * self.head_dim)?;
        let start = batch_index * self.capacity * self.head_dim;
        Ok(all[start..start + groups * self.head_dim].to_vec())
    }
}

/// Released `get_window_topk_idxs`, including -1 padding. `window_topk` is the
/// number of entries per query: `min(seq, window)` for a start-zero prefill and
/// `window` for decode.
pub fn deepseek_v41_window_indices(
    batch: usize,
    seq: usize,
    start_pos: usize,
    window: usize,
) -> Result<(Vec<i32>, usize)> {
    if batch == 0 || seq == 0 || window == 0 || start_pos > i32::MAX as usize {
        bail!("V4.1 window index dimensions are invalid");
    }
    let topk = if start_pos == 0 {
        seq.min(window)
    } else {
        window
    };
    let mut indices = Vec::with_capacity(batch * topk);
    for _ in 0..batch {
        if start_pos == 0 {
            for query in 0..seq {
                let start = (query as isize - window as isize + 1).max(0) as usize;
                for offset in 0..topk {
                    let position = start + offset;
                    indices.push(if position > query {
                        -1
                    } else {
                        position as i32
                    });
                }
            }
        } else {
            let oldest = start_pos % window + 1;
            for offset in 0..topk {
                let position = if offset < window - oldest {
                    oldest + offset
                } else {
                    offset - (window - oldest)
                };
                indices.push(if position > start_pos {
                    -1
                } else {
                    position as i32
                });
            }
        }
    }
    Ok((indices, topk))
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekV41AttentionConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    /// 0 = sliding-only; 1 or 2 = compressed attention with an index source.
    pub compress_ratio: usize,
    pub norm_eps: f32,
    pub rotary: DeepseekV41RotaryConfig,
}

pub struct DeepseekV41AttentionWeights {
    pub wq_a: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wkv: Vec<f32>,
    pub kv_norm: Vec<f32>,
    /// `[groups, o_lora_rank, group_width]` row-major.
    pub wo_a: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub attn_sink: Vec<f32>,
}

/// Checkpoint storage is retained for every projection. `wo_a` is the released
/// converter's dense grouped matrix; packed storage there is rejected.
pub struct DeepseekV41MixedAttentionWeights {
    pub wq_a: DeepseekV41LinearWeights,
    pub q_norm: Vec<f32>,
    pub wq_b: DeepseekV41LinearWeights,
    pub wkv: DeepseekV41LinearWeights,
    pub kv_norm: Vec<f32>,
    pub wo_a: DeepseekV41LinearWeights,
    pub wo_b: DeepseekV41LinearWeights,
    pub attn_sink: Vec<f32>,
}

impl DeepseekV41MixedAttentionWeights {
    pub(super) fn from_fp32(
        c: &DeepseekV41AttentionConfig,
        w: DeepseekV41AttentionWeights,
    ) -> Self {
        let dense = |rows, cols, values| DeepseekV41LinearWeights::Fp32 {
            rows,
            cols,
            values,
            lora: None,
        };
        Self {
            wq_a: dense(c.q_lora_rank, c.hidden_size, w.wq_a),
            q_norm: w.q_norm,
            wq_b: dense(c.heads * c.head_dim, c.q_lora_rank, w.wq_b),
            wkv: dense(c.head_dim, c.hidden_size, w.wkv),
            kv_norm: w.kv_norm,
            wo_a: dense(
                c.o_groups * c.o_lora_rank,
                c.heads / c.o_groups.max(1) * c.head_dim,
                w.wo_a,
            ),
            wo_b: dense(c.hidden_size, c.o_groups * c.o_lora_rank, w.wo_b),
            attn_sink: w.attn_sink,
        }
    }
}

/// Uploaded per-layer attention projections. A streamed attention object keeps
/// this slot empty between layer executions.
struct AttentionProjections {
    wq_a: DeepseekV41ResidentLinear,
    q_norm: GpuBuffer,
    wq_b: DeepseekV41ResidentLinear,
    wkv: DeepseekV41ResidentLinear,
    kv_norm: GpuBuffer,
    wo_a: GpuBuffer,
    wo_a_precision: DeepseekV41WeightOutput,
    wo_b: DeepseekV41ResidentLinear,
    attn_sink: GpuBuffer,
}

fn weight_output_precision(w: &DeepseekV41LinearWeights) -> DeepseekV41WeightOutput {
    match w {
        DeepseekV41LinearWeights::Fp32 { .. } => DeepseekV41WeightOutput::Fp32,
        DeepseekV41LinearWeights::Bf16 { .. } | DeepseekV41LinearWeights::Packed { .. } => {
            DeepseekV41WeightOutput::Bf16
        }
    }
}

impl AttentionProjections {
    fn build(
        device: &VulkanDevice,
        config: &DeepseekV41AttentionConfig,
        weights: DeepseekV41MixedAttentionWeights,
        stream_precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        let q_width = config
            .heads
            .checked_mul(config.head_dim)
            .context("V4.1 attention query width overflow")?;
        let output_width = config
            .o_groups
            .checked_mul(config.o_lora_rank)
            .context("V4.1 attention output width overflow")?;
        if weights.wq_a.shape() != [config.q_lora_rank, config.hidden_size]
            || weights.q_norm.len() != config.q_lora_rank
            || weights.wq_b.shape() != [q_width, config.q_lora_rank]
            || weights.wkv.shape() != [config.head_dim, config.hidden_size]
            || weights.kv_norm.len() != config.head_dim
            || weights.wo_a.shape()
                != [
                    output_width,
                    config.heads / config.o_groups * config.head_dim,
                ]
            || weights.wo_b.shape() != [config.hidden_size, output_width]
            || weights.attn_sink.len() != config.heads
        {
            bail!("V4.1 attention parameter geometry is invalid");
        }
        for w in [
            &weights.wq_a,
            &weights.wq_b,
            &weights.wkv,
            &weights.wo_a,
            &weights.wo_b,
        ] {
            w.validate()?;
            if stream_precision == DeepseekV41WeightOutput::Bf16
                && matches!(w, DeepseekV41LinearWeights::Fp32 { .. })
            {
                bail!("V4.1 BF16 attention requires BF16/packed projection storage");
            }
        }
        for values in [&weights.q_norm, &weights.kv_norm, &weights.attn_sink] {
            if !values.iter().all(|value| value.is_finite()) {
                bail!("V4.1 attention parameters must be finite");
            }
        }
        let (wo_a, wo_a_precision) = match weights.wo_a {
            DeepseekV41LinearWeights::Fp32 { values, .. } => {
                (values, DeepseekV41WeightOutput::Fp32)
            }
            DeepseekV41LinearWeights::Bf16 { values, .. } => {
                (values, DeepseekV41WeightOutput::Bf16)
            }
            DeepseekV41LinearWeights::Packed { .. } => {
                bail!("V4.1 grouped wo_a requires converted dense F32/BF16 storage")
            }
        };
        Ok(Self {
            wq_a: weights.wq_a.upload(device)?,
            q_norm: GpuBuffer::from_f32(device, &weights.q_norm)?,
            wq_b: weights.wq_b.upload(device)?,
            wkv: weights.wkv.upload(device)?,
            kv_norm: GpuBuffer::from_f32(device, &weights.kv_norm)?,
            wo_a: GpuBuffer::from_f32(device, &wo_a)?,
            wo_a_precision,
            wo_b: weights.wo_b.upload(device)?,
            attn_sink: GpuBuffer::from_f32(device, &weights.attn_sink)?,
        })
    }
}

pub struct DeepseekV41Attention {
    #[cfg(test)]
    pub(super) projection_trace: Option<(GpuBuffer, GpuBuffer)>,
    #[cfg(test)]
    pub(super) sparse_trace: Option<(GpuBuffer, GpuBuffer)>,
    #[cfg(test)]
    pub(super) weights_trace: std::cell::RefCell<Option<GpuBuffer>>,
    #[cfg(test)]
    pub(super) raw_trace: std::cell::RefCell<Option<GpuBuffer>>,
    device: VulkanDevice,
    config: DeepseekV41AttentionConfig,
    group_width: usize,
    projections: Option<AttentionProjections>,
    source: Option<DeepseekV41StreamedLayer>,
    query_precision: DeepseekV41WeightOutput,
    linear: DeepseekV41MixedLinearKernel,
    stream_precision: DeepseekV41WeightOutput,
    norm: NormKernel,
    rotary: DeepseekV41Rotary,
    quantizer: DeepseekV41ActivationQuantizer,
    window_cache: DeepseekV41WindowKvCache,
    weights_kernel: vulkan::ComputeKernel,
    values_kernel: vulkan::ComputeKernel,
    grouped_kernel: vulkan::ComputeKernel,
    empty_kv: GpuBuffer,
    empty_indices: GpuBuffer,
    active_batch: Option<usize>,
    next_position: usize,
}

impl DeepseekV41Attention {
    pub(super) fn mark_retained_step(&mut self, batch: usize, seq: usize, position: usize) {
        self.active_batch = Some(batch);
        self.next_position = position + seq;
    }
    pub fn new(
        device: VulkanDevice,
        config: DeepseekV41AttentionConfig,
        weights: DeepseekV41AttentionWeights,
        batch_capacity: usize,
    ) -> Result<Self> {
        let weights = DeepseekV41MixedAttentionWeights::from_fp32(&config, weights);
        Self::new_mixed(
            device,
            config,
            weights,
            batch_capacity,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    pub fn new_mixed(
        device: VulkanDevice,
        config: DeepseekV41AttentionConfig,
        weights: DeepseekV41MixedAttentionWeights,
        batch_capacity: usize,
        stream_precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        let query_precision = weight_output_precision(&weights.wq_a);
        Self::build(
            device,
            config,
            Some(weights),
            None,
            batch_capacity,
            stream_precision,
            query_precision,
        )
    }

    /// Projections are read from `checkpoint` when the orchestrator enters this
    /// layer instead of being uploaded at construction.
    #[allow(clippy::too_many_arguments)]
    pub fn new_mixed_streamed(
        device: VulkanDevice,
        config: DeepseekV41AttentionConfig,
        checkpoint: DeepseekV41Checkpoint,
        layer: usize,
        batch_capacity: usize,
        stream_precision: DeepseekV41WeightOutput,
        query_precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        Self::build(
            device,
            config,
            None,
            Some(DeepseekV41StreamedLayer::new(checkpoint, layer)),
            batch_capacity,
            stream_precision,
            query_precision,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        device: VulkanDevice,
        config: DeepseekV41AttentionConfig,
        weights: Option<DeepseekV41MixedAttentionWeights>,
        source: Option<DeepseekV41StreamedLayer>,
        batch_capacity: usize,
        stream_precision: DeepseekV41WeightOutput,
        query_precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        if config.heads == 0
            || config.head_dim == 0
            || config.hidden_size == 0
            || config.q_lora_rank == 0
            || config.o_groups == 0
            || !config.heads.is_multiple_of(config.o_groups)
            || config.o_lora_rank == 0
            || config.window_size == 0
            || batch_capacity == 0
        {
            bail!("V4.1 attention dimensions and capacities must be positive");
        }
        if !matches!(config.compress_ratio, 0 | 1 | 2) {
            bail!("V4.1 attention compress_ratio must be 0, 1, or 2");
        }
        if !config.head_dim.is_multiple_of(KV_QUANT_BLOCK) {
            bail!("V4.1 attention head width must be a multiple of the FP8 block");
        }
        if !config.norm_eps.is_finite() || config.norm_eps <= 0.0 {
            bail!("V4.1 attention norm epsilon must be finite and positive");
        }
        if config.rotary.dim == 0
            || config.rotary.dim > config.head_dim
            || !config.rotary.dim.is_multiple_of(2)
        {
            bail!("V4.1 attention rotary tail is invalid");
        }
        let group_width = (config.heads / config.o_groups) * config.head_dim;
        if group_width == 0 || !group_width.is_multiple_of(KV_QUANT_BLOCK) {
            bail!("V4.1 attention grouped output width must align with the FP8 block");
        }
        let projections = weights
            .map(|weights| AttentionProjections::build(&device, &config, weights, stream_precision))
            .transpose()?;
        Ok(Self {
            projections,
            source,
            query_precision,
            linear: DeepseekV41MixedLinearKernel::new(device.clone())?,
            stream_precision,
            norm: NormKernel::new(&device)?,
            rotary: DeepseekV41Rotary::new(device.clone(), config.rotary)?,
            quantizer: DeepseekV41ActivationQuantizer::new(device.clone())?,
            window_cache: DeepseekV41WindowKvCache::new(
                device.clone(),
                batch_capacity,
                config.window_size,
                config.head_dim,
            )?,
            weights_kernel: vulkan::ComputeKernel::new(
                &device,
                ATTENTION_WEIGHTS,
                7,
                std::mem::size_of::<WeightPush>() as u32,
            )?,
            values_kernel: vulkan::ComputeKernel::new(
                &device,
                ATTENTION_VALUES,
                6,
                std::mem::size_of::<ValuePush>() as u32,
            )?,
            grouped_kernel: vulkan::ComputeKernel::new(
                &device,
                GROUPED_OUTPUT,
                3,
                std::mem::size_of::<GroupedPush>() as u32,
            )?,
            empty_kv: GpuBuffer::zeros_f32(&device, 1)?,
            empty_indices: GpuBuffer::zeros_u32(&device, 1)?,
            group_width,
            active_batch: None,
            next_position: 0,
            config,
            device,
            #[cfg(test)]
            projection_trace: None,
            #[cfg(test)]
            sparse_trace: None,
            #[cfg(test)]
            weights_trace: std::cell::RefCell::new(None),
            #[cfg(test)]
            raw_trace: std::cell::RefCell::new(None),
        })
    }

    /// Reads and uploads this layer's projections for one layer execution.
    pub(super) fn load_projections(&mut self) -> Result<()> {
        if self.projections.is_some() {
            return Ok(());
        }
        let source = self
            .source
            .as_ref()
            .context("V4.1 streamed attention has no checkpoint source")?;
        let mut checkpoint = source.lock()?;
        let weights = checkpoint.load_attention_components(source.layer())?;
        let weights = match weights.attention {
            super::orchestrator::AttentionWeights::Fp32(w) => {
                DeepseekV41MixedAttentionWeights::from_fp32(&self.config, w)
            }
            super::orchestrator::AttentionWeights::Mixed(w) => w,
        };
        self.projections = Some(AttentionProjections::build(
            &self.device,
            &self.config,
            weights,
            self.stream_precision,
        )?);
        Ok(())
    }

    pub(super) fn projections_are_resident(&self) -> bool {
        self.projections.is_some()
    }

    /// Drops streamed projections after the layer finishes. Resident weights stay.
    pub(super) fn release_projections(&mut self) {
        if self.source.is_some() {
            self.projections = None;
        }
    }

    /// `qr = q_norm(wq_a(x))`, shared with the fine indexer exactly as in the
    /// released attention block.
    pub fn query_latent(&self, x: &GpuBuffer, rows: usize) -> Result<GpuBuffer> {
        if x.f32_capacity() < rows * self.config.hidden_size
            || !x.belongs_to_device(&self.device)
            || rows == 0
        {
            bail!("V4.1 attention query latent input geometry or device mismatch");
        }
        let p = self
            .projections
            .as_ref()
            .context("V4.1 attention projections are not loaded")?;
        let projected = self.linear.apply(
            x,
            rows,
            &p.wq_a,
            p.wq_a.input_precision(self.stream_precision),
        )?;
        let normalized = self.norm.apply(
            &self.device,
            &projected,
            &p.q_norm,
            rows,
            self.config.q_lora_rank,
            self.config.norm_eps,
        )?;
        self.linear.cast_buffer(
            &normalized,
            rows * self.config.q_lora_rank,
            p.wq_a.output_precision(),
        )
    }

    pub(super) fn query_precision(&self) -> DeepseekV41WeightOutput {
        self.query_precision
    }

    /// Full prefill from position zero. Seals the sliding-window ring for the
    /// following decode steps.
    pub fn prefill(
        &mut self,
        x: &GpuBuffer,
        compressed_kv: Option<&DeepseekV41CompressedKvCache>,
        selection: Option<&DeepseekV41IndexSelection>,
        batch: usize,
        seq: usize,
    ) -> Result<GpuBuffer> {
        self.forward_step(x, None, compressed_kv, selection, batch, seq, 0, true)
    }

    /// One token per batch in the current sequence.
    pub fn decode_step(
        &mut self,
        x: &GpuBuffer,
        compressed_kv: Option<&DeepseekV41CompressedKvCache>,
        selection: Option<&DeepseekV41IndexSelection>,
        batch: usize,
        start_pos: usize,
    ) -> Result<GpuBuffer> {
        self.forward_step(
            x,
            None,
            compressed_kv,
            selection,
            batch,
            1,
            start_pos,
            false,
        )
    }

    /// The orchestrator prepares `qr` once and shares that same GPU tensor with
    /// the indexer and attention. Existing standalone calls still prepare it here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_step(
        &mut self,
        x: &GpuBuffer,
        prepared_qr: Option<&GpuBuffer>,
        compressed_kv: Option<&DeepseekV41CompressedKvCache>,
        selection: Option<&DeepseekV41IndexSelection>,
        batch: usize,
        seq: usize,
        start_pos: usize,
        prefill: bool,
    ) -> Result<GpuBuffer> {
        if seq == 0 || (prefill && start_pos != 0) || (!prefill && seq != 1) {
            bail!("V4.1 attention requires a start-zero prefill or one-token decode");
        }
        if !prefill && (self.active_batch != Some(batch) || start_pos != self.next_position) {
            bail!(
                "V4.1 attention decode must continue the current batch at position {}",
                self.next_position
            );
        }
        let output = self.forward(
            x,
            prepared_qr,
            compressed_kv,
            selection,
            batch,
            seq,
            start_pos,
            prefill,
        )?;
        self.active_batch = Some(batch);
        self.next_position = start_pos + seq;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        x: &GpuBuffer,
        prepared_qr: Option<&GpuBuffer>,
        compressed_kv: Option<&DeepseekV41CompressedKvCache>,
        selection: Option<&DeepseekV41IndexSelection>,
        batch: usize,
        seq: usize,
        start_pos: usize,
        prefill: bool,
    ) -> Result<GpuBuffer> {
        if batch == 0 || batch > self.window_cache.batch() {
            bail!("V4.1 attention batch exceeds capacity");
        }
        let rows = batch
            .checked_mul(seq)
            .context("V4.1 attention row count overflow")?;
        if rows > 65_535 {
            bail!("V4.1 attention exceeds dispatch capacity");
        }
        if start_pos + seq > self.config.rotary.max_positions {
            bail!("V4.1 attention position exceeds the rotary table");
        }
        if self.config.compress_ratio == 0 && (compressed_kv.is_some() || selection.is_some()) {
            bail!("V4.1 sliding-only attention cannot consume compressed state");
        }
        if x.f32_capacity() < rows * self.config.hidden_size || !x.belongs_to_device(&self.device) {
            bail!("V4.1 attention input geometry or device mismatch");
        }
        if let Some(qr) = prepared_qr {
            if qr.f32_capacity() < rows * self.config.q_lora_rank
                || !qr.belongs_to_device(&self.device)
            {
                bail!("V4.1 prepared query latent geometry or device mismatch");
            }
        }
        if prefill && start_pos != 0 {
            bail!("V4.1 attention prefill must start at position zero");
        }
        let width = if self.config.compress_ratio == 0 {
            0
        } else {
            (start_pos + seq) / self.config.compress_ratio
        };
        let compressed_topk = match selection {
            Some(selection) => {
                if selection.rows != rows
                    || (width != 0 && selection.width != width)
                    || selection.topk > selection.width
                {
                    bail!("V4.1 attention index selection is incompatible with this step");
                }
                selection.topk
            }
            None => 0,
        };
        let compressed = match compressed_kv {
            Some(cache) => {
                if cache.batch() < batch || (width != 0 && cache.capacity() < width) {
                    bail!("V4.1 attention compressed KV cache is too small");
                }
                cache
            }
            None => {
                if compressed_topk != 0 {
                    bail!("V4.1 attention has compressed indices but no compressed KV cache");
                }
                if width != 0 {
                    bail!("V4.1 compressed attention step has no compressed KV cache");
                }
                return self.run_attention(
                    x,
                    prepared_qr,
                    None,
                    selection,
                    batch,
                    seq,
                    start_pos,
                    prefill,
                    rows,
                );
            }
        };
        self.run_attention(
            x,
            prepared_qr,
            Some(compressed),
            selection,
            batch,
            seq,
            start_pos,
            prefill,
            rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention(
        &mut self,
        x: &GpuBuffer,
        prepared_qr: Option<&GpuBuffer>,
        compressed: Option<&DeepseekV41CompressedKvCache>,
        selection: Option<&DeepseekV41IndexSelection>,
        batch: usize,
        seq: usize,
        start_pos: usize,
        prefill: bool,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let p = self
            .projections
            .as_ref()
            .context("V4.1 attention projections are not loaded")?;
        let compressed_topk = selection.map_or(0, |selection| selection.topk);
        let qr = match prepared_qr {
            Some(qr) => qr.clone(),
            None => self.query_latent(x, rows)?,
        };
        let q = self.linear.apply(
            &qr,
            rows,
            &p.wq_b,
            p.wq_b.input_precision(self.query_precision),
        )?;
        let q = self.rotary.apply_repeated_buffer(
            &q,
            rows * self.config.heads,
            self.config.head_dim,
            self.config.heads,
            seq,
            start_pos,
            1,
            false,
        )?;
        let q = self.linear.cast_buffer(
            &q,
            rows * self.config.heads * self.config.head_dim,
            p.wq_b.output_precision(),
        )?;

        let kv = self.linear.apply(
            x,
            rows,
            &p.wkv,
            p.wkv.input_precision(self.stream_precision),
        )?;
        let kv = self.norm.apply(
            &self.device,
            &kv,
            &p.kv_norm,
            rows,
            self.config.head_dim,
            self.config.norm_eps,
        )?;
        let kv =
            self.linear
                .cast_buffer(&kv, rows * self.config.head_dim, p.wkv.output_precision())?;
        let kv = self.rotary.apply_grouped_buffer(
            &kv,
            rows,
            self.config.head_dim,
            seq,
            start_pos,
            1,
            false,
        )?;
        let kv =
            self.linear
                .cast_buffer(&kv, rows * self.config.head_dim, p.wkv.output_precision())?;
        let kv = self.quantizer.quantize_buffer(
            &kv,
            rows,
            self.config.head_dim,
            KV_QUANT_BLOCK,
            DeepseekV41QuantMode::Fp8E4M3,
            DeepseekV41ScaleMode::Ue8m0,
        )?;
        let kv_values = self.linear.cast_buffer(
            &kv.values,
            rows * self.config.head_dim,
            p.wkv.output_precision(),
        )?;

        let (window_kv, window_capacity) = if prefill {
            self.window_cache.seed_prefill(&kv_values, seq)?;
            (kv_values, seq)
        } else {
            self.window_cache.write_decode(&kv_values, start_pos)?;
            (
                self.window_cache.values().clone(),
                self.window_cache.window(),
            )
        };
        let (window_indices, window_topk) =
            deepseek_v41_window_indices(batch, seq, start_pos, self.config.window_size)?;
        let window_bits: Vec<u32> = window_indices
            .into_iter()
            .map(|index| index as u32)
            .collect();
        let window_index_buffer = GpuBuffer::from_u32_scheduled(&self.device, &window_bits)?;

        let (compressed_kv, compressed_capacity, compressed_indices) = match (compressed, selection)
        {
            (Some(cache), Some(selection)) if compressed_topk > 0 => (
                Some(cache.values()),
                cache.capacity(),
                selection
                    .indices
                    .as_ref()
                    .context("V4.1 attention selection has no indices")?,
            ),
            (Some(cache), _) => (Some(cache.values()), cache.capacity(), &self.empty_indices),
            _ => (None, 0usize, &self.empty_indices),
        };
        let attention = self.gathered_attention(
            &q,
            &window_kv,
            compressed_kv,
            &window_index_buffer,
            compressed_indices,
            DeepseekV41SparseAttentionParams {
                batch,
                seq,
                window_rows: window_capacity,
                window_topk,
                compressed_topk,
                compressed_capacity,
            },
        )?;
        let attention = self.linear.cast_buffer(
            &attention,
            rows * self.config.heads * self.config.head_dim,
            p.wq_b.output_precision(),
        )?;
        #[cfg(test)]
        {
            self.sparse_trace = Some((q.clone(), attention.clone()));
        }

        // Inverse output rotation, then the block-diagonal grouped projection.
        let rotated = self.rotary.apply_repeated_buffer(
            &attention,
            rows * self.config.heads,
            self.config.head_dim,
            self.config.heads,
            seq,
            start_pos,
            1,
            true,
        )?;
        let rotated = self.linear.cast_buffer(
            &rotated,
            rows * self.config.heads * self.config.head_dim,
            p.wq_b.output_precision(),
        )?;
        // The reference quantizes the `[groups, group_width]` view, so a block
        // never crosses a group. The grouped width is validated to be a
        // multiple of the block, which makes this flat per-row pass produce the
        // same amax blocks.
        let quantized_output = self.quantizer.quantize_buffer(
            &rotated,
            rows,
            self.config.heads * self.config.head_dim,
            KV_QUANT_BLOCK,
            DeepseekV41QuantMode::Fp8E4M3,
            DeepseekV41ScaleMode::Ue8m0,
        )?;
        let grouped_input = self.linear.cast_buffer(
            &quantized_output.values,
            rows * self.config.heads * self.config.head_dim,
            p.wo_a_precision,
        )?;
        let output_width = self.config.o_groups * self.config.o_lora_rank;
        let grouped = GpuBuffer::zeros_f32(&self.device, rows * output_width)?;
        let grouped_push = GroupedPush {
            rows: rows as u32,
            groups: self.config.o_groups as u32,
            group_width: self.group_width as u32,
            output_rank: self.config.o_lora_rank as u32,
            bf16: u32::from(p.wo_a_precision == DeepseekV41WeightOutput::Bf16),
        };
        let mut commands = self.device.compute_batch()?;
        self.grouped_kernel.record_dispatch(
            &mut commands,
            &[&grouped_input, &p.wo_a, &grouped],
            bytemuck::bytes_of(&grouped_push),
            [
                self.config.o_lora_rank.div_ceil(64) as u32,
                self.config.o_groups as u32,
                rows as u32,
            ],
        )?;
        commands.submit()?;
        let grouped = self
            .linear
            .cast_buffer(&grouped, rows * output_width, p.wo_a_precision)?;
        #[cfg(test)]
        {
            self.projection_trace = Some((grouped_input, grouped.clone()));
        }
        self.linear.apply(
            &grouped,
            rows,
            &p.wo_b,
            p.wo_b.input_precision(p.wo_a_precision),
        )
    }

    /// Gathered sparse attention over the window rows followed by the selected
    /// compressed rows, matching `sparse_attn(q, cat([window_kv, compress_kv]),
    /// sink, cat([window_idxs, compress_idxs]))`. `-1` skips a slot; a compressed
    /// index is offset by `window_rows`. Returns the pre-rotation head output
    /// `[rows, heads, head_dim]`. Shared with `run_attention` and exposed so the
    /// gather, sink, and normalization can be qualified directly.
    #[allow(clippy::too_many_arguments)]
    pub fn gathered_attention(
        &self,
        queries: &GpuBuffer,
        window_kv: &GpuBuffer,
        compressed_kv: Option<&GpuBuffer>,
        window_indices: &GpuBuffer,
        compressed_indices: &GpuBuffer,
        params: DeepseekV41SparseAttentionParams,
    ) -> Result<GpuBuffer> {
        let p = self
            .projections
            .as_ref()
            .context("V4.1 attention projections are not loaded")?;
        let DeepseekV41SparseAttentionParams {
            batch,
            seq,
            window_rows,
            window_topk,
            compressed_topk,
            compressed_capacity,
        } = params;
        if batch == 0
            || seq == 0
            || window_rows == 0
            || window_topk == 0
            || window_topk > window_rows
        {
            bail!("V4.1 sparse attention window geometry is invalid");
        }
        let rows = batch
            .checked_mul(seq)
            .context("V4.1 sparse attention row count overflow")?;
        if rows > 65_535 {
            bail!("V4.1 sparse attention exceeds dispatch capacity");
        }
        match compressed_kv {
            Some(_) if compressed_capacity == 0 => {
                bail!("V4.1 sparse attention compressed cache has no capacity")
            }
            Some(_) if compressed_topk > compressed_capacity => {
                bail!("V4.1 sparse attention selects beyond the compressed cache")
            }
            None if compressed_topk != 0 => {
                bail!("V4.1 sparse attention selects compressed rows without a cache")
            }
            _ => {}
        }
        let query_elements = rows * self.config.heads * self.config.head_dim;
        let window_elements = batch * window_rows * self.config.head_dim;
        let heads = self.config.heads;
        if queries.f32_capacity() < query_elements
            || window_kv.f32_capacity() < window_elements
            || window_indices.f32_capacity() < rows * window_topk
            || compressed_indices.f32_capacity() < rows * compressed_topk
            || !queries.belongs_to_device(&self.device)
            || !window_kv.belongs_to_device(&self.device)
            || !window_indices.belongs_to_device(&self.device)
            || !compressed_indices.belongs_to_device(&self.device)
        {
            bail!("V4.1 sparse attention buffer geometry or device mismatch");
        }
        if let Some(cache) = compressed_kv {
            if cache.f32_capacity() < batch * compressed_capacity * self.config.head_dim
                || !cache.belongs_to_device(&self.device)
            {
                bail!("V4.1 sparse attention compressed cache geometry mismatch");
            }
        }
        let total_topk = window_topk + compressed_topk;
        let weights = GpuBuffer::zeros_f32(
            &self.device,
            rows.checked_mul(heads)
                .and_then(|n| n.checked_mul(total_topk))
                .context("V4.1 sparse attention weight buffer overflow")?,
        )?;
        let attention = GpuBuffer::zeros_f32(&self.device, query_elements)?;
        #[cfg(test)]
        {
            *self.weights_trace.borrow_mut() = Some(weights.clone());
            *self.raw_trace.borrow_mut() = Some(attention.clone());
        }
        let weight_push = WeightPush {
            rows: rows as u32,
            batch: batch as u32,
            seq: seq as u32,
            heads: heads as u32,
            dim: self.config.head_dim as u32,
            window_capacity: window_rows as u32,
            window_topk: window_topk as u32,
            compressed_topk: compressed_topk as u32,
            compressed_capacity: compressed_capacity as u32,
            scale: (self.config.head_dim as f64).powf(-0.5) as f32,
        };
        let value_push = ValuePush {
            rows: rows as u32,
            seq: seq as u32,
            heads: heads as u32,
            dim: self.config.head_dim as u32,
            window_capacity: window_rows as u32,
            window_topk: window_topk as u32,
            compressed_topk: compressed_topk as u32,
            compressed_capacity: compressed_capacity as u32,
        };
        let compressed = compressed_kv.unwrap_or(&self.empty_kv);
        let mut commands = self.device.compute_batch()?;
        self.weights_kernel.record_dispatch(
            &mut commands,
            &[
                queries,
                window_kv,
                compressed,
                window_indices,
                compressed_indices,
                &p.attn_sink,
                &weights,
            ],
            bytemuck::bytes_of(&weight_push),
            [1, heads as u32, rows as u32],
        )?;
        self.values_kernel.record_dispatch(
            &mut commands,
            &[
                window_kv,
                compressed,
                window_indices,
                compressed_indices,
                &weights,
                &attention,
            ],
            bytemuck::bytes_of(&value_push),
            [
                self.config.head_dim.div_ceil(64) as u32,
                heads as u32,
                rows as u32,
            ],
        )?;
        commands.submit()?;
        Ok(attention)
    }

    pub fn window_cache(&self) -> &DeepseekV41WindowKvCache {
        &self.window_cache
    }

    pub fn attention_sink(&self) -> Result<&GpuBuffer> {
        Ok(&self
            .projections
            .as_ref()
            .context("V4.1 attention projections are not loaded")?
            .attn_sink)
    }

    /// Width of the compressed cache this step can see, matching the indexer's
    /// `width`. Exposed for graph wiring.
    pub fn compressed_width(&self, start_pos: usize, seq: usize) -> usize {
        if self.config.compress_ratio == 0 {
            0
        } else {
            (start_pos + seq) / self.config.compress_ratio
        }
    }
}
