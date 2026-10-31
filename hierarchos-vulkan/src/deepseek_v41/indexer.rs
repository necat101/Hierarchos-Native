//! DeepSeek V4.1 fine indexer. Mirrors `Indexer` in the released
//! inference/model.py: an index-source layer turns the compressor's pre-RoPE
//! latent into FP4/UE8M0 keys in a shared cache, scores compressed positions
//! with FP4 queries and head weights, applies the level-two candidate mask and
//! the causal compressed-length rule, and returns sorted top-k positions
//! shifted past the sliding window.
//!
//! Typed projections preserve packed/BF16 storage and dtype boundaries through
//! normalization, rotary, score products and head reduction. The legacy FP32
//! score diagnostic retains its separate wide-GEMM reduction limitation.

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

use super::checkpoint::DeepseekV41StreamedLayer;
use super::{
    DeepseekV41ActivationQuantizer, DeepseekV41CandidateSelector, DeepseekV41Checkpoint,
    DeepseekV41CompressedLatents, DeepseekV41LinearWeights, DeepseekV41MixedLinearKernel,
    DeepseekV41QuantMode, DeepseekV41ResidentLinear, DeepseekV41Rotary, DeepseekV41RotaryConfig,
    DeepseekV41ScaleMode, DeepseekV41WeightOutput,
};

const LINEAR: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_linear.spv");
const LINEAR_HALF: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_linear_half.spv");
const NORM: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_rms_norm.spv");
const SCORES: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_index_scores.spv");
const TOPK: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_index_topk.spv");
const CACHE_WRITE: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_cache_write.spv");

const INDEX_QUANT_BLOCK: usize = 32;
const MAX_TOPK: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
    bf16: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormPush {
    rows: u32,
    dim: u32,
    parameter_groups: u32,
    eps: f32,
    weight_offset: f32,
    pow_rstd: u32,
    unfused_products: u32,
    extra_rsqrt_refinement: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScorePush {
    rows: u32,
    batch: u32,
    seq: u32,
    heads: u32,
    head_dim: u32,
    width: u32,
    key_capacity: u32,
    start_pos: u32,
    ratio: u32,
    use_candidates: u32,
    weight_scale: f32,
    dot_bf16: u32,
    weight_bf16: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TopkPush {
    rows: u32,
    seq: u32,
    width: u32,
    topk: u32,
    offset: u32,
    start_pos: u32,
    ratio: u32,
    raw: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CachePush {
    rows: u32,
    dim: u32,
    groups_per_batch: u32,
    first_group: u32,
    capacity: u32,
}

pub(crate) struct LinearKernel {
    kernel: vulkan::ComputeKernel,
}

impl LinearKernel {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new(
                device,
                LINEAR,
                3,
                std::mem::size_of::<LinearPush>() as u32,
            )?,
        })
    }

    pub(crate) fn apply(
        &self,
        device: &VulkanDevice,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<GpuBuffer> {
        self.apply_with_precision(device, input, weight, rows, input_dim, output_dim, false)
    }

    pub(crate) fn apply_with_precision(
        &self,
        device: &VulkanDevice,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        bf16: bool,
    ) -> Result<GpuBuffer> {
        let mut commands = device.compute_batch()?;
        let output = self.record_apply_with_precision(
            &mut commands,
            device,
            input,
            weight,
            rows,
            input_dim,
            output_dim,
            bf16,
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Record-only form of [`Self::apply_with_precision`]: identical buffers,
    /// push constants, grid and order, with the wait left to the caller's batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_apply_with_precision(
        &self,
        commands: &mut vulkan::ComputeBatch,
        device: &VulkanDevice,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        bf16: bool,
    ) -> Result<GpuBuffer> {
        if rows == 0
            || input_dim == 0
            || output_dim == 0
            || output_dim.div_ceil(16) > 65_535
            || rows.div_ceil(16) > 65_535
        {
            bail!("V4.1 indexer projection exceeds dispatch capacity");
        }
        let elements = rows
            .checked_mul(input_dim)
            .context("V4.1 indexer projection input overflow")?;
        let outputs = rows
            .checked_mul(output_dim)
            .context("V4.1 indexer projection output overflow")?;
        if elements > u32::MAX as usize
            || outputs > u32::MAX as usize
            || input.f32_capacity() < elements
            || weight.f32_capacity() < output_dim * input_dim
            || !input.belongs_to_device(device)
            || !weight.belongs_to_device(device)
        {
            bail!("V4.1 indexer projection buffer geometry or device mismatch");
        }
        let output = GpuBuffer::zeros_f32(device, outputs)?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            bf16: u32::from(bf16),
        };
        self.kernel.record_dispatch(
            commands,
            &[input, weight, &output],
            bytemuck::bytes_of(&push),
            [output_dim.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
        )?;
        Ok(output)
    }
}

/// Dense projection whose weights are BF16-exact but stored as 16-bit pairs.
/// The released LM head keeps an FP32 weight containing only BF16-rounded
/// values, so the accumulation is the FP32 SGEMM staging and not the dense-BF16
/// single chain. Decoding the pair in the shader is exact.
pub(crate) struct HalfLinearKernel {
    kernel: vulkan::ComputeKernel,
}

impl HalfLinearKernel {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new(
                device,
                LINEAR_HALF,
                3,
                std::mem::size_of::<HalfLinearPush>() as u32,
            )?,
        })
    }

    pub(crate) fn apply(
        &self,
        device: &VulkanDevice,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<GpuBuffer> {
        if rows == 0
            || input_dim == 0
            || output_dim == 0
            || output_dim.div_ceil(16) > 65_535
            || rows.div_ceil(16) > 65_535
        {
            bail!("V4.1 half projection exceeds dispatch capacity");
        }
        let elements = rows
            .checked_mul(input_dim)
            .context("V4.1 half projection input overflow")?;
        let outputs = rows
            .checked_mul(output_dim)
            .context("V4.1 half projection output overflow")?;
        if elements > u32::MAX as usize
            || outputs > u32::MAX as usize
            || input.f32_capacity() < elements
            || weight.f32_capacity() < (output_dim * input_dim).div_ceil(2)
            || !input.belongs_to_device(device)
            || !weight.belongs_to_device(device)
        {
            bail!("V4.1 half projection buffer geometry or device mismatch");
        }
        let output = GpuBuffer::zeros_f32(device, outputs)?;
        let push = HalfLinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        };
        let mut commands = device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[input, weight, &output],
            bytemuck::bytes_of(&push),
            [output_dim.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HalfLinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

pub(crate) struct NormKernel {
    kernel: vulkan::ComputeKernel,
}

impl NormKernel {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new(
                device,
                NORM,
                5,
                std::mem::size_of::<NormPush>() as u32,
            )?,
        })
    }

    pub(crate) fn apply(
        &self,
        device: &VulkanDevice,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<GpuBuffer> {
        if rows == 0 || dim == 0 || rows.div_ceil(64) > 65_535 {
            bail!("V4.1 indexer norm exceeds dispatch capacity");
        }
        let elements = rows
            .checked_mul(dim)
            .context("V4.1 indexer norm overflow")?;
        if elements > u32::MAX as usize
            || input.f32_capacity() < elements
            || weight.f32_capacity() < dim
            || !input.belongs_to_device(device)
            || !weight.belongs_to_device(device)
        {
            bail!("V4.1 indexer norm buffer geometry or device mismatch");
        }
        let output = GpuBuffer::zeros_f32(device, elements)?;
        let rstd = GpuBuffer::zeros_f32(device, rows)?;
        let variance = GpuBuffer::zeros_f32(device, rows)?;
        let push = NormPush {
            rows: rows as u32,
            dim: dim as u32,
            parameter_groups: 1,
            eps,
            weight_offset: 0.0,
            pow_rstd: 0,
            unfused_products: 1,
            extra_rsqrt_refinement: 3,
        };
        let mut commands = device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[input, weight, &output, &rstd, &variance],
            bytemuck::bytes_of(&push),
            [rows.div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }
}

/// Shared index-key cache owned by an index source. Consumers read the same
/// buffer, matching the released `shared_attn.index_k` hand-off.
#[derive(Clone)]
pub struct DeepseekV41IndexKeyCache {
    batch: usize,
    capacity: usize,
    head_dim: usize,
    values: GpuBuffer,
}

impl DeepseekV41IndexKeyCache {
    pub fn new(
        device: VulkanDevice,
        batch: usize,
        capacity: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if batch == 0 || capacity == 0 || head_dim == 0 {
            bail!("V4.1 index-key cache dimensions must be positive");
        }
        let elements = batch
            .checked_mul(capacity)
            .and_then(|n| n.checked_mul(head_dim))
            .context("V4.1 index-key cache size overflow")?;
        if elements > u32::MAX as usize {
            bail!("V4.1 index-key cache exceeds u32 addressing");
        }
        Ok(Self {
            values: GpuBuffer::zeros_f32(&device, elements)?,
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

    fn values(&self) -> &GpuBuffer {
        &self.values
    }

    /// Read `groups` groups for one batch item. Intended for qualification and
    /// cache-integrity checks; serving reads the buffer in place.
    pub fn read_batch(&self, batch_index: usize, groups: usize) -> Result<Vec<f32>> {
        if batch_index >= self.batch || groups > self.capacity {
            bail!("V4.1 index-key cache read exceeds its geometry");
        }
        let all = self
            .values
            .read_f32(self.batch * self.capacity * self.head_dim)?;
        let start = batch_index * self.capacity * self.head_dim;
        Ok(all[start..start + groups * self.head_dim].to_vec())
    }
}

/// Scores and sorted positions for one indexer call. Masked positions are
/// `-inf` in `scores`; `indices` uses the reference's `-1` for unreachable
/// selected positions and already includes the sliding-window `offset`.
#[derive(Clone)]
pub struct DeepseekV41IndexSelection {
    pub rows: usize,
    pub width: usize,
    pub topk: usize,
    pub scores: GpuBuffer,
    pub indices: Option<GpuBuffer>,
    pub candidates: Option<GpuBuffer>,
}

impl DeepseekV41IndexSelection {
    pub fn read_scores(&self) -> Result<Vec<f32>> {
        if self.width == 0 {
            return Ok(Vec::new());
        }
        self.scores.read_f32(self.rows * self.width)
    }

    pub fn read_indices(&self) -> Result<Vec<i32>> {
        match &self.indices {
            Some(buffer) => Ok(buffer
                .read_u32(self.rows * self.topk)?
                .into_iter()
                .map(|value| value as i32)
                .collect()),
            None => Ok(Vec::new()),
        }
    }

    pub fn read_candidates(&self) -> Result<Vec<u32>> {
        match &self.candidates {
            Some(buffer) => buffer.read_u32(self.rows * self.width),
            None => Ok(Vec::new()),
        }
    }
}

/// Standalone top-k stage, also used by qualification fixtures.
pub struct DeepseekV41IndexTopk {
    device: VulkanDevice,
    kernel: vulkan::ComputeKernel,
}

impl DeepseekV41IndexTopk {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new(
                &device,
                TOPK,
                3,
                std::mem::size_of::<TopkPush>() as u32,
            )?,
            device,
        })
    }

    /// `scores` is row-major `[rows, width]` and may contain `-inf`. The shader
    /// selects `topk` positions, sorts their indices ascending, and maps every
    /// position beyond `(start_pos + row_in_seq + 1) / ratio` to `-1`.
    pub fn rank(
        &self,
        scores: &GpuBuffer,
        rows: usize,
        width: usize,
        seq: usize,
        start_pos: usize,
        ratio: usize,
        topk: usize,
        offset: usize,
    ) -> Result<GpuBuffer> {
        if rows == 0 || width == 0 || topk == 0 || topk > width || topk > MAX_TOPK {
            bail!("V4.1 index top-k requires 0 < topk <= min(width, 512)");
        }
        if seq == 0 || !rows.is_multiple_of(seq) || !matches!(ratio, 1 | 2) {
            bail!("V4.1 index top-k requires a valid sequence shape and ratio");
        }
        if width > 65_535
            || rows > 65_535
            || start_pos > u32::MAX as usize
            || offset > u32::MAX as usize
        {
            bail!("V4.1 index top-k exceeds dispatch or position capacity");
        }
        if !scores.belongs_to_device(&self.device) || scores.f32_capacity() < rows * width {
            bail!("V4.1 index top-k scores buffer geometry or device mismatch");
        }
        let output = GpuBuffer::zeros_u32(
            &self.device,
            rows.checked_mul(topk)
                .context("V4.1 index top-k output overflow")?,
        )?;
        let push = TopkPush {
            rows: rows as u32,
            seq: seq as u32,
            width: width as u32,
            topk: topk as u32,
            offset: offset as u32,
            start_pos: start_pos as u32,
            ratio: ratio as u32,
            raw: 0,
        };
        let scratch = GpuBuffer::zeros_u32(&self.device, rows * width)?;
        let mut commands = self.device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[scores, &output, &scratch],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Ranks a row-major `[rows, blocks]` score table with the same CPU
    /// top-k staging the indexer uses, returning `[rows, min(topk, blocks)]`
    /// block ids without position mapping. The level-one candidate selector
    /// needs this: PyTorch's `topk` on tied block maxima does not break ties by
    /// block id, and the tie set changes with `k` (heap vs partition staging).
    pub fn select_blocks(
        &self,
        scores: &GpuBuffer,
        rows: usize,
        blocks: usize,
        topk_blocks: usize,
    ) -> Result<GpuBuffer> {
        if rows == 0 || blocks == 0 {
            bail!("V4.1 block top-k requires a non-empty score table");
        }
        let topk = topk_blocks.min(blocks);
        if topk == 0 || topk > MAX_TOPK || blocks > 65_535 || rows > 65_535 {
            bail!("V4.1 block top-k requires 0 < topk <= min(blocks, 512)");
        }
        if !scores.belongs_to_device(&self.device) || scores.f32_capacity() < rows * blocks {
            bail!("V4.1 block top-k scores buffer geometry or device mismatch");
        }
        let output = GpuBuffer::zeros_u32(
            &self.device,
            rows.checked_mul(topk)
                .context("V4.1 block top-k output overflow")?,
        )?;
        let push = TopkPush {
            rows: rows as u32,
            seq: 1,
            width: blocks as u32,
            topk: topk as u32,
            offset: 0,
            start_pos: 0,
            ratio: 1,
            raw: 1,
        };
        let scratch = GpuBuffer::zeros_u32(&self.device, rows * blocks)?;
        let mut commands = self.device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[scores, &output, &scratch],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }
}

/// Standalone score stage: `[rows, heads, head_dim]` queries against one shared
/// key per compressed position, rectified and combined with per-head weights.
pub struct DeepseekV41IndexScorer {
    device: VulkanDevice,
    kernel: vulkan::ComputeKernel,
    empty_candidates: GpuBuffer,
}

impl DeepseekV41IndexScorer {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new(
                &device,
                SCORES,
                5,
                std::mem::size_of::<ScorePush>() as u32,
            )?,
            empty_candidates: GpuBuffer::zeros_u32(&device, 1)?,
            device,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        queries: &GpuBuffer,
        keys: &GpuBuffer,
        weights: &GpuBuffer,
        candidates: Option<&GpuBuffer>,
        rows: usize,
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        width: usize,
        key_capacity: usize,
        start_pos: usize,
        ratio: usize,
        weight_scale: f32,
    ) -> Result<GpuBuffer> {
        self.score_typed(
            queries,
            keys,
            weights,
            candidates,
            rows,
            batch,
            seq,
            heads,
            head_dim,
            width,
            key_capacity,
            start_pos,
            ratio,
            weight_scale,
            DeepseekV41WeightOutput::Fp32,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_typed(
        &self,
        queries: &GpuBuffer,
        keys: &GpuBuffer,
        weights: &GpuBuffer,
        candidates: Option<&GpuBuffer>,
        rows: usize,
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        width: usize,
        key_capacity: usize,
        start_pos: usize,
        ratio: usize,
        weight_scale: f32,
        query_precision: DeepseekV41WeightOutput,
        weight_precision: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        if rows == 0
            || batch == 0
            || seq == 0
            || heads == 0
            || head_dim == 0
            || width == 0
            || !rows.is_multiple_of(seq)
            || rows / seq != batch
            || width > 65_535
            || rows > 65_535
            || width.div_ceil(64) > 65_535
        {
            bail!("V4.1 index scoring shape or dispatch capacity is invalid");
        }
        if !matches!(ratio, 1 | 2) {
            bail!("V4.1 index scoring requires ratio 1 or 2");
        }
        let query_elements = rows
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(head_dim))
            .context("V4.1 index query size overflow")?;
        let key_elements = batch
            .checked_mul(key_capacity)
            .and_then(|n| n.checked_mul(head_dim))
            .context("V4.1 index key size overflow")?;
        if queries.f32_capacity() < query_elements
            || keys.f32_capacity() < key_elements
            || weights.f32_capacity() < rows * heads
            || !queries.belongs_to_device(&self.device)
            || !keys.belongs_to_device(&self.device)
            || !weights.belongs_to_device(&self.device)
        {
            bail!("V4.1 index scoring buffer geometry or device mismatch");
        }
        let use_candidates = candidates.is_some();
        let candidate_buffer = candidates.unwrap_or(&self.empty_candidates);
        if use_candidates && candidate_buffer.f32_capacity() < rows * width {
            bail!("V4.1 index candidate mask is too small");
        }
        let scores = GpuBuffer::zeros_f32(&self.device, rows * width)?;
        let push = ScorePush {
            rows: rows as u32,
            batch: batch as u32,
            seq: seq as u32,
            heads: heads as u32,
            head_dim: head_dim as u32,
            width: width as u32,
            key_capacity: key_capacity as u32,
            start_pos: start_pos as u32,
            ratio: ratio as u32,
            use_candidates: u32::from(use_candidates),
            weight_scale,
            dot_bf16: u32::from(query_precision == DeepseekV41WeightOutput::Bf16),
            weight_bf16: u32::from(weight_precision == DeepseekV41WeightOutput::Bf16),
        };
        let mut commands = self.device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[queries, keys, weights, candidate_buffer, &scores],
            bytemuck::bytes_of(&push),
            [width.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(scores)
    }
}

pub(crate) struct CacheWriter {
    kernel: vulkan::ComputeKernel,
}

impl CacheWriter {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new(
                device,
                CACHE_WRITE,
                2,
                std::mem::size_of::<CachePush>() as u32,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn append(
        &self,
        device: &VulkanDevice,
        source: &GpuBuffer,
        destination: &GpuBuffer,
        rows: usize,
        dim: usize,
        groups_per_batch: usize,
        first_group: usize,
        capacity: usize,
    ) -> Result<()> {
        if rows == 0
            || dim == 0
            || groups_per_batch == 0
            || !rows.is_multiple_of(groups_per_batch)
            || first_group + groups_per_batch > capacity
            || dim.div_ceil(64) > 65_535
            || rows > 65_535
        {
            bail!("V4.1 index cache append exceeds its geometry");
        }
        if !source.belongs_to_device(device)
            || !destination.belongs_to_device(device)
            || source.f32_capacity() < rows * dim
            || destination.f32_capacity() < (rows / groups_per_batch) * capacity * dim
        {
            bail!("V4.1 index cache append buffer geometry or device mismatch");
        }
        let push = CachePush {
            rows: rows as u32,
            dim: dim as u32,
            groups_per_batch: groups_per_batch as u32,
            first_group: first_group as u32,
            capacity: capacity as u32,
        };
        let mut commands = device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[source, destination],
            bytemuck::bytes_of(&push),
            [dim.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV41IndexerConfig {
    pub heads: usize,
    pub index_head_dim: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub q_lora_rank: usize,
    pub compress_ratio: usize,
    pub index_topk: usize,
    pub owns_k: bool,
    pub is_candidate_source: bool,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub norm_eps: f32,
    pub rotary: DeepseekV41RotaryConfig,
}

pub struct DeepseekV41IndexerWeights {
    /// `[index_head_dim, head_dim]`, only used when this layer owns keys.
    pub wk: Vec<f32>,
    /// `[index_head_dim]`, only used when this layer owns keys.
    pub k_norm: Vec<f32>,
    /// `[heads * index_head_dim, q_lora_rank]`.
    pub wq_b: Vec<f32>,
    /// `[heads, hidden_size]`.
    pub weights_proj: Vec<f32>,
}

pub struct DeepseekV41MixedIndexerWeights {
    /// Absent at a consumer of shared keys.
    pub wk: Option<DeepseekV41LinearWeights>,
    pub k_norm: Vec<f32>,
    pub wq_b: DeepseekV41LinearWeights,
    pub weights_proj: DeepseekV41LinearWeights,
}

impl DeepseekV41MixedIndexerWeights {
    pub(super) fn from_fp32(
        c: &DeepseekV41IndexerConfig,
        w: DeepseekV41IndexerWeights,
    ) -> Result<Self> {
        if !c.owns_k && (!w.wk.is_empty() || !w.k_norm.is_empty()) {
            bail!("V4.1 indexer copy must not allocate its own key weights");
        }
        let dense = |rows, cols, values| DeepseekV41LinearWeights::Fp32 {
            rows,
            cols,
            values,
            lora: None,
        };
        Ok(Self {
            wk: c.owns_k.then(|| dense(c.index_head_dim, c.head_dim, w.wk)),
            k_norm: w.k_norm,
            wq_b: dense(c.heads * c.index_head_dim, c.q_lora_rank, w.wq_b),
            weights_proj: dense(c.heads, c.hidden_size, w.weights_proj),
        })
    }
}

struct IndexerProjection {
    wk: Option<DeepseekV41ResidentLinear>,
    k_norm: Option<GpuBuffer>,
    wq_b: DeepseekV41ResidentLinear,
    weights_proj: DeepseekV41ResidentLinear,
}

pub struct DeepseekV41Indexer {
    device: VulkanDevice,
    config: DeepseekV41IndexerConfig,
    projection: Option<IndexerProjection>,
    source: Option<DeepseekV41StreamedLayer>,
    linear: DeepseekV41MixedLinearKernel,
    stream_precision: DeepseekV41WeightOutput,
    query_precision: DeepseekV41WeightOutput,
    norm: NormKernel,
    rotary: DeepseekV41Rotary,
    quantizer: DeepseekV41ActivationQuantizer,
    scorer: DeepseekV41IndexScorer,
    topk: DeepseekV41IndexTopk,
    cache_writer: CacheWriter,
    key_cache: Option<DeepseekV41IndexKeyCache>,
    candidate_selector: Option<DeepseekV41CandidateSelector>,
}

impl DeepseekV41Indexer {
    pub fn new(
        device: VulkanDevice,
        config: DeepseekV41IndexerConfig,
        weights: DeepseekV41IndexerWeights,
        batch_capacity: usize,
        max_positions: usize,
    ) -> Result<Self> {
        let weights = DeepseekV41MixedIndexerWeights::from_fp32(&config, weights)?;
        Self::new_mixed(
            device,
            config,
            weights,
            batch_capacity,
            max_positions,
            DeepseekV41WeightOutput::Fp32,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_mixed(
        device: VulkanDevice,
        config: DeepseekV41IndexerConfig,
        weights: DeepseekV41MixedIndexerWeights,
        batch_capacity: usize,
        max_positions: usize,
        stream_precision: DeepseekV41WeightOutput,
        query_precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        if config.heads == 0
            || config.index_head_dim == 0
            || config.head_dim == 0
            || config.hidden_size == 0
            || config.q_lora_rank == 0
            || config.index_topk == 0
            || config.index_topk > MAX_TOPK
            || batch_capacity == 0
            || max_positions == 0
        {
            bail!("V4.1 indexer dimensions and capacities must be positive");
        }
        if !matches!(config.compress_ratio, 1 | 2) {
            bail!("V4.1 indexer requires compression ratio 1 or 2");
        }
        if !config.index_head_dim.is_multiple_of(INDEX_QUANT_BLOCK) {
            bail!("V4.1 indexer head width must be a multiple of the FP4 block");
        }
        if config.rotary.dim == 0
            || config.rotary.dim > config.index_head_dim
            || !config.rotary.dim.is_multiple_of(2)
            || config.rotary.max_positions < max_positions
        {
            bail!("V4.1 indexer rotary tail or position capacity is invalid");
        }
        if !config.norm_eps.is_finite() || config.norm_eps <= 0.0 {
            bail!("V4.1 indexer norm epsilon must be finite and positive");
        }
        if config.is_candidate_source
            && (config.candidate_topk_blocks == 0 || config.candidate_block_size == 0)
        {
            bail!("V4.1 index candidate source requires positive block settings");
        }
        if weights.wk.is_some() != config.owns_k || (!config.owns_k && !weights.k_norm.is_empty()) {
            bail!("V4.1 indexer copy must not allocate its own key weights");
        }
        let q_width = config
            .heads
            .checked_mul(config.index_head_dim)
            .context("V4.1 indexer query width overflow")?;
        if weights.wq_b.shape() != [q_width, config.q_lora_rank]
            || weights.weights_proj.shape() != [config.heads, config.hidden_size]
        {
            bail!("V4.1 indexer query or weights-projection parameter geometry is invalid");
        }
        let wk = if config.owns_k {
            let w = weights.wk.as_ref().expect("owner has wk");
            if w.shape() != [config.index_head_dim, config.head_dim]
                || weights.k_norm.len() != config.index_head_dim
            {
                bail!("V4.1 indexer key parameter geometry is invalid");
            }
            w.validate()?;
            if !weights.k_norm.iter().all(|v| v.is_finite()) {
                bail!("V4.1 indexer key parameters must be finite");
            }
            weights.wk.map(|w| w.upload(&device)).transpose()?
        } else {
            None
        };
        let k_norm = if config.owns_k {
            Some(GpuBuffer::from_f32(&device, &weights.k_norm)?)
        } else {
            None
        };
        weights.wq_b.validate()?;
        weights.weights_proj.validate()?;
        if stream_precision == DeepseekV41WeightOutput::Bf16
            && (matches!(weights.wq_b, DeepseekV41LinearWeights::Fp32 { .. })
                || matches!(weights.weights_proj, DeepseekV41LinearWeights::Fp32 { .. })
                || wk
                    .as_ref()
                    .is_some_and(|w| w.output_precision() != DeepseekV41WeightOutput::Bf16)
                || query_precision != DeepseekV41WeightOutput::Bf16)
        {
            bail!("V4.1 BF16 indexer requires BF16/packed projections and BF16 query latents");
        }
        let key_cache = if config.owns_k {
            Some(DeepseekV41IndexKeyCache::new(
                device.clone(),
                batch_capacity,
                max_positions.div_ceil(config.compress_ratio),
                config.index_head_dim,
            )?)
        } else {
            None
        };
        let candidate_selector = if config.is_candidate_source {
            Some(DeepseekV41CandidateSelector::new(device.clone())?)
        } else {
            None
        };
        Ok(Self {
            projection: Some(IndexerProjection {
                wk,
                k_norm,
                wq_b: weights.wq_b.upload(&device)?,
                weights_proj: weights.weights_proj.upload(&device)?,
            }),
            source: None,
            linear: DeepseekV41MixedLinearKernel::new(device.clone())?,
            stream_precision,
            query_precision,
            norm: NormKernel::new(&device)?,
            rotary: DeepseekV41Rotary::new(device.clone(), config.rotary)?,
            quantizer: DeepseekV41ActivationQuantizer::new(device.clone())?,
            scorer: DeepseekV41IndexScorer::new(device.clone())?,
            topk: DeepseekV41IndexTopk::new(device.clone())?,
            cache_writer: CacheWriter::new(&device)?,
            key_cache,
            candidate_selector,
            config,
            device,
        })
    }

    /// Key cache and candidate state stay resident; projections are read from the
    /// checkpoint when the orchestrator enters the layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_mixed_streamed(
        device: VulkanDevice,
        config: DeepseekV41IndexerConfig,
        batch_capacity: usize,
        max_positions: usize,
        stream_precision: DeepseekV41WeightOutput,
        query_precision: DeepseekV41WeightOutput,
        checkpoint: DeepseekV41Checkpoint,
        layer: usize,
    ) -> Result<Self> {
        if config.heads == 0
            || config.index_head_dim == 0
            || config.head_dim == 0
            || config.hidden_size == 0
            || config.q_lora_rank == 0
            || config.index_topk == 0
            || config.index_topk > MAX_TOPK
            || batch_capacity == 0
            || max_positions == 0
        {
            bail!("V4.1 indexer dimensions and capacities must be positive");
        }
        if !matches!(config.compress_ratio, 1 | 2) {
            bail!("V4.1 indexer requires compression ratio 1 or 2");
        }
        if !config.index_head_dim.is_multiple_of(INDEX_QUANT_BLOCK)
            || config.rotary.dim == 0
            || config.rotary.dim > config.index_head_dim
            || !config.rotary.dim.is_multiple_of(2)
            || config.rotary.max_positions < max_positions
        {
            bail!("V4.1 indexer head/rotary geometry is invalid");
        }
        if !config.norm_eps.is_finite() || config.norm_eps <= 0.0 {
            bail!("V4.1 indexer norm epsilon must be finite and positive");
        }
        if config.is_candidate_source
            && (config.candidate_topk_blocks == 0 || config.candidate_block_size == 0)
        {
            bail!("V4.1 index candidate source requires positive block settings");
        }
        let key_cache = if config.owns_k {
            Some(DeepseekV41IndexKeyCache::new(
                device.clone(),
                batch_capacity,
                max_positions.div_ceil(config.compress_ratio),
                config.index_head_dim,
            )?)
        } else {
            None
        };
        let candidate_selector = if config.is_candidate_source {
            Some(DeepseekV41CandidateSelector::new(device.clone())?)
        } else {
            None
        };
        Ok(Self {
            projection: None,
            source: Some(DeepseekV41StreamedLayer::new(checkpoint, layer)),
            linear: DeepseekV41MixedLinearKernel::new(device.clone())?,
            stream_precision,
            query_precision,
            norm: NormKernel::new(&device)?,
            rotary: DeepseekV41Rotary::new(device.clone(), config.rotary)?,
            quantizer: DeepseekV41ActivationQuantizer::new(device.clone())?,
            scorer: DeepseekV41IndexScorer::new(device.clone())?,
            topk: DeepseekV41IndexTopk::new(device.clone())?,
            cache_writer: CacheWriter::new(&device)?,
            key_cache,
            candidate_selector,
            config,
            device,
        })
    }

    /// Reads and uploads the indexer projections for one layer execution.
    pub(super) fn load_weights(&mut self) -> Result<()> {
        if self.projection.is_some() {
            return Ok(());
        }
        let Some(source) = self.source.as_ref() else {
            return Ok(());
        };
        let mut checkpoint = source.lock()?;
        let weights = checkpoint.load_attention_components(source.layer())?;
        let indexer = weights
            .indexer
            .context("V4.1 streamed indexer has no indexer weights")?;
        let weights = match indexer {
            super::orchestrator::IndexerWeights::Fp32(w) => {
                DeepseekV41MixedIndexerWeights::from_fp32(&self.config, w)?
            }
            super::orchestrator::IndexerWeights::Mixed(w) => w,
        };
        let q_width = self
            .config
            .heads
            .checked_mul(self.config.index_head_dim)
            .context("V4.1 indexer query width overflow")?;
        if weights.wq_b.shape() != [q_width, self.config.q_lora_rank]
            || weights.weights_proj.shape() != [self.config.heads, self.config.hidden_size]
            || weights.wk.is_some() != self.config.owns_k
            || (!self.config.owns_k && !weights.k_norm.is_empty())
        {
            bail!("V4.1 streamed indexer parameter geometry is invalid");
        }
        if let Some(wk) = &weights.wk {
            if wk.shape() != [self.config.index_head_dim, self.config.head_dim]
                || weights.k_norm.len() != self.config.index_head_dim
            {
                bail!("V4.1 streamed indexer key parameter geometry is invalid");
            }
        }
        weights.wq_b.validate()?;
        weights.weights_proj.validate()?;
        let wk = weights.wk.map(|w| w.upload(&self.device)).transpose()?;
        let k_norm = if self.config.owns_k {
            Some(GpuBuffer::from_f32(&self.device, &weights.k_norm)?)
        } else {
            None
        };
        if self.stream_precision == DeepseekV41WeightOutput::Bf16
            && (matches!(weights.wq_b, DeepseekV41LinearWeights::Fp32 { .. })
                || matches!(weights.weights_proj, DeepseekV41LinearWeights::Fp32 { .. })
                || wk
                    .as_ref()
                    .is_some_and(|w| w.output_precision() != DeepseekV41WeightOutput::Bf16))
        {
            bail!("V4.1 BF16 indexer requires BF16/packed projections");
        }
        self.projection = Some(IndexerProjection {
            wk,
            k_norm,
            wq_b: weights.wq_b.upload(&self.device)?,
            weights_proj: weights.weights_proj.upload(&self.device)?,
        });
        Ok(())
    }

    pub(super) fn release_weights(&mut self) {
        if self.source.is_some() {
            self.projection = None;
        }
    }

    pub fn owns_keys(&self) -> bool {
        self.config.owns_k
    }

    pub fn index_keys(&self) -> Option<&DeepseekV41IndexKeyCache> {
        self.key_cache.as_ref()
    }

    /// Turns a compressor latent into shared index keys. Returns the first
    /// group written, or `None` when this layer does not own keys or the
    /// compressor has no completed group yet. Keys read the pre-RoPE latent,
    /// before any compressed-KV finalization.
    pub fn write_index_keys(
        &self,
        latent: &DeepseekV41CompressedLatents,
        start_pos: usize,
    ) -> Result<Option<usize>> {
        let Some(cache) = &self.key_cache else {
            return Ok(None);
        };
        if latent.groups_per_batch == 0 {
            return Ok(None);
        }
        if latent.compress_ratio != self.config.compress_ratio {
            bail!("V4.1 index keys require the matching compression ratio");
        }
        if latent.head_dim != self.config.head_dim {
            bail!("V4.1 index keys require the compressor head dimension");
        }
        if latent.batch_size > cache.batch() {
            bail!("V4.1 index key cache batch capacity is too small");
        }
        if start_pos > 0 && latent.groups_per_batch != 1 {
            bail!("V4.1 decode index keys require one completed group");
        }
        let rows = latent
            .batch_size
            .checked_mul(latent.groups_per_batch)
            .context("V4.1 index key row count overflow")?;
        let first_group = latent.first_group;
        let quantized = self
            .produce_index_keys(
                &latent.values,
                latent.batch_size,
                latent.groups_per_batch,
                first_group,
                start_pos,
            )?
            .expect("owner produces keys");
        self.cache_writer.append(
            &self.device,
            &quantized,
            cache.values(),
            rows,
            self.config.index_head_dim,
            latent.groups_per_batch,
            first_group,
            cache.capacity(),
        )?;
        Ok(Some(first_group))
    }

    /// Numeric key production without the cache append, used by qualification
    /// and by `write_index_keys`. `latent` is `[batch * groups, head_dim]` in
    /// the compressor's pre-RoPE, pre-quantization form.
    pub fn produce_index_keys(
        &self,
        latent: &GpuBuffer,
        batch: usize,
        groups_per_batch: usize,
        first_group: usize,
        start_pos: usize,
    ) -> Result<Option<GpuBuffer>> {
        if !self.config.owns_k {
            return Ok(None);
        }
        let p = self
            .projection
            .as_ref()
            .context("V4.1 indexer projections are not loaded")?;
        if batch == 0 || groups_per_batch == 0 || batch > self.capacity_batch() {
            bail!("V4.1 index key production dimensions are invalid");
        }
        if start_pos > 0 && groups_per_batch != 1 {
            bail!("V4.1 decode index keys require one completed group");
        }
        let end_group = first_group
            .checked_add(groups_per_batch)
            .context("V4.1 index key group range overflow")?;
        if end_group > self.capacity_groups() {
            bail!("V4.1 index key cache does not fit the completed group");
        }
        let rows = batch
            .checked_mul(groups_per_batch)
            .context("V4.1 index key row count overflow")?;
        if latent.f32_capacity() < rows * self.config.head_dim
            || !latent.belongs_to_device(&self.device)
        {
            bail!("V4.1 index key latent geometry or device mismatch");
        }
        let projected = self.linear.apply(
            latent,
            rows,
            p.wk.as_ref().expect("owner has wk"),
            p.wk.as_ref()
                .expect("owner has wk")
                .input_precision(self.stream_precision),
        )?;
        let normalized = self.norm.apply(
            &self.device,
            &projected,
            p.k_norm.as_ref().expect("owner has k_norm"),
            rows,
            self.config.index_head_dim,
            self.config.norm_eps,
        )?;
        let precision = p.wk.as_ref().expect("owner has wk").output_precision();
        let normalized =
            self.linear
                .cast_buffer(&normalized, rows * self.config.index_head_dim, precision)?;
        // Group j rotates at position j * ratio.
        let rotated = self.rotary.apply_grouped_buffer(
            &normalized,
            rows,
            self.config.index_head_dim,
            groups_per_batch,
            first_group * self.config.compress_ratio,
            self.config.compress_ratio,
            false,
        )?;
        let rotated =
            self.linear
                .cast_buffer(&rotated, rows * self.config.index_head_dim, precision)?;
        let keys = self.quantizer.quantize_buffer(
            &rotated,
            rows,
            self.config.index_head_dim,
            INDEX_QUANT_BLOCK,
            DeepseekV41QuantMode::Fp4E2M1,
            DeepseekV41ScaleMode::Ue8m0,
        )?;
        Ok(Some(self.linear.cast_buffer(
            &keys.values,
            rows * self.config.index_head_dim,
            precision,
        )?))
    }

    fn capacity_batch(&self) -> usize {
        self.key_cache.as_ref().map_or(0, |cache| cache.batch())
    }

    fn capacity_groups(&self) -> usize {
        self.key_cache.as_ref().map_or(0, |cache| cache.capacity())
    }

    /// Scores compressed positions and returns the sorted top-k selection.
    /// `candidates` is the level-one mask from the candidate source, shaped
    /// `[batch * seq, width]`.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        x: &GpuBuffer,
        qr: &GpuBuffer,
        keys: &DeepseekV41IndexKeyCache,
        batch: usize,
        seq: usize,
        start_pos: usize,
        offset: usize,
        candidates: Option<&GpuBuffer>,
    ) -> Result<DeepseekV41IndexSelection> {
        let p = self
            .projection
            .as_ref()
            .context("V4.1 indexer projections are not loaded")?;
        if batch == 0 || seq == 0 || batch > keys.batch() {
            bail!("V4.1 index selection batch/sequence is invalid");
        }
        let rows = batch
            .checked_mul(seq)
            .context("V4.1 index selection row count overflow")?;
        if rows > 65_535 {
            bail!("V4.1 index selection exceeds dispatch capacity");
        }
        let width = (start_pos + seq) / self.config.compress_ratio;
        if width > keys.capacity() {
            bail!("V4.1 index selection exceeds the shared key cache");
        }
        let x_elements = rows
            .checked_mul(self.config.hidden_size)
            .context("V4.1 index selection x geometry overflow")?;
        let qr_elements = rows
            .checked_mul(self.config.q_lora_rank)
            .context("V4.1 index selection qr geometry overflow")?;
        if x.f32_capacity() < x_elements
            || qr.f32_capacity() < qr_elements
            || !x.belongs_to_device(&self.device)
            || !qr.belongs_to_device(&self.device)
        {
            bail!("V4.1 index selection input buffer geometry or device mismatch");
        }
        if width == 0 {
            return Ok(DeepseekV41IndexSelection {
                rows,
                width: 0,
                topk: 0,
                scores: GpuBuffer::zeros_f32(&self.device, 1)?,
                indices: None,
                candidates: None,
            });
        }

        let q_width = self.config.heads * self.config.index_head_dim;
        let projected = self.linear.apply(
            qr,
            rows,
            &p.wq_b,
            p.wq_b.input_precision(self.query_precision),
        )?;
        let repeated_rows = rows
            .checked_mul(self.config.heads)
            .context("V4.1 index query row overflow")?;
        let rotated = self.rotary.apply_repeated_buffer(
            &projected,
            repeated_rows,
            self.config.index_head_dim,
            self.config.heads,
            seq,
            start_pos,
            1,
            false,
        )?;
        let rotated =
            self.linear
                .cast_buffer(&rotated, rows * q_width, p.wq_b.output_precision())?;
        let query = self.quantizer.quantize_buffer(
            &rotated,
            repeated_rows,
            self.config.index_head_dim,
            INDEX_QUANT_BLOCK,
            DeepseekV41QuantMode::Fp4E2M1,
            DeepseekV41ScaleMode::Ue8m0,
        )?;
        let query_values =
            self.linear
                .cast_buffer(&query.values, rows * q_width, p.wq_b.output_precision())?;
        let weights = self.linear.apply(
            x,
            rows,
            &p.weights_proj,
            p.weights_proj.input_precision(self.stream_precision),
        )?;
        let weight_scale = ((self.config.index_head_dim as f64).powf(-0.5)
            * (self.config.heads as f64).powf(-0.5)) as f32;
        let scores = self.scorer.score_typed(
            &query_values,
            keys.values(),
            &weights,
            candidates,
            rows,
            batch,
            seq,
            self.config.heads,
            self.config.index_head_dim,
            width,
            keys.capacity(),
            start_pos,
            self.config.compress_ratio,
            weight_scale,
            p.wq_b.output_precision(),
            p.weights_proj.output_precision(),
        )?;

        let mut published_candidates = None;
        if let Some(selector) = &self.candidate_selector {
            let compress_lens: Vec<u32> = (0..rows)
                .map(|row| {
                    let visible = if start_pos == 0 {
                        (row % seq + 1) / self.config.compress_ratio
                    } else {
                        (start_pos + seq) / self.config.compress_ratio
                    };
                    u32::try_from(visible).expect("compressed length is bounded by width")
                })
                .collect();
            let mask = selector.select_buffer(
                &scores,
                &compress_lens,
                width,
                self.config.candidate_block_size,
                self.config.candidate_topk_blocks,
            )?;
            published_candidates = Some(mask);
        }

        let topk = self.config.index_topk.min(width);
        let indices = self.topk.rank(
            &scores,
            rows,
            width,
            seq,
            start_pos,
            self.config.compress_ratio,
            topk,
            offset,
        )?;
        Ok(DeepseekV41IndexSelection {
            rows,
            width,
            topk,
            scores,
            indices: Some(indices),
            candidates: published_candidates,
        })
    }
}
