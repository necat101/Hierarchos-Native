//! First-level block selection for the V4.1 two-level sparse indexer.
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

use super::indexer::DeepseekV41IndexTopk;

const BLOCK_SCORES: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_candidate_scores.spv");
const BLOCK_MASK: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_candidate_mask.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    rows: u32,
    width: u32,
    blocks: u32,
    block_size: u32,
    topk_blocks: u32,
}

/// Vulkan counterpart of `inference/model.py::select_candidate_blocks`.
///
/// The newest reachable block is always retained (when top-k > 0), including
/// positions beyond the causal boundary within that block. The second-level
/// indexer must still apply its causal position mask. Unreachable blocks are
/// dropped. Ties follow the pinned CPU `topk` staging, whose selected set
/// depends on `k` rather than a lower-block-id order.
///
/// Ranking is still one serial top-k per query, and the mask expansion is one
/// invocation per selected slot. This is not a million-token serving path.
pub struct DeepseekV41CandidateSelector {
    device: VulkanDevice,
    scores: vulkan::ComputeKernel,
    mask: vulkan::ComputeKernel,
    topk: DeepseekV41IndexTopk,
}

impl DeepseekV41CandidateSelector {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        let push_size = std::mem::size_of::<Push>() as u32;
        Ok(Self {
            scores: vulkan::ComputeKernel::new(&device, BLOCK_SCORES, 3, push_size)?,
            mask: vulkan::ComputeKernel::new(&device, BLOCK_MASK, 3, push_size)?,
            topk: DeepseekV41IndexTopk::new(device.clone())?,
            device,
        })
    }

    /// Host I/O wrapper for qualification. All score reduction, ranking, and
    /// mask expansion run on Vulkan; no Python or host tensor math is used.
    /// `logits` is row-major [visible_lengths.len(), width]. Each row's visible
    /// length is the number of complete compression groups for that query.
    pub fn select(
        &self,
        logits: &[f32],
        visible_lengths: &[u32],
        width: usize,
        block_size: usize,
        topk_blocks: usize,
    ) -> Result<Vec<u32>> {
        let (_, elements, _) = Self::shape(visible_lengths, width, block_size, topk_blocks)?;
        if logits.len() != elements {
            bail!("DeepSeek-V4.1 candidate logits shape mismatch");
        }
        if logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
            bail!("DeepSeek-V4.1 candidate logits must be finite or negative infinity");
        }
        if width == 0 {
            return Ok(Vec::new());
        }
        // -inf is an intentional sparse-mask sentinel. Upload its validated
        // IEEE payload through integer I/O rather than the finite-only guard.
        let bits: Vec<u32> = logits.iter().map(|value| value.to_bits()).collect();
        let logits = GpuBuffer::from_u32(&self.device, &bits)?;
        self.select_buffer(&logits, visible_lengths, width, block_size, topk_blocks)?
            .read_u32(elements)
    }

    fn shape(
        visible_lengths: &[u32],
        width: usize,
        block_size: usize,
        topk_blocks: usize,
    ) -> Result<(usize, usize, usize)> {
        let rows = visible_lengths.len();
        if rows == 0
            || block_size == 0
            || block_size > u32::MAX as usize
            || width > u32::MAX as usize
            || topk_blocks > u32::MAX as usize
        {
            bail!("DeepSeek-V4.1 candidate selection requires rows > 0 and valid u32 dimensions (block_size > 0)");
        }
        let elements = rows
            .checked_mul(width)
            .context("candidate logits size overflow")?;
        if elements > u32::MAX as usize {
            bail!("DeepSeek-V4.1 candidate logits shape mismatch or u32 address overflow");
        }
        if visible_lengths.iter().any(|&len| len as usize > width) {
            bail!("DeepSeek-V4.1 candidate visible length exceeds width");
        }
        let blocks = width.div_ceil(block_size);
        // Stay within Vulkan's minimum guaranteed workgroup-count limits.
        if rows > 65_535 || blocks.div_ceil(64) > 65_535 {
            bail!("DeepSeek-V4.1 candidate selection exceeds dispatch capacity");
        }
        Ok((rows, elements, blocks))
    }

    /// Select directly from resident row-major scores without downloading the
    /// logits or mask. Visible lengths are CPU shape/position metadata only.
    /// Scores must contain finite values or -inf, as produced by the indexer;
    /// this path validates geometry/device, without a host scan of GPU values.
    /// A zero-width result uses a one-word allocation with no logical elements.
    pub fn select_buffer(
        &self,
        logits: &GpuBuffer,
        visible_lengths: &[u32],
        width: usize,
        block_size: usize,
        topk_blocks: usize,
    ) -> Result<GpuBuffer> {
        let (rows, elements, blocks) =
            Self::shape(visible_lengths, width, block_size, topk_blocks)?;
        if !logits.belongs_to_device(&self.device) || logits.f32_capacity() < elements {
            bail!("DeepSeek-V4.1 candidate logits buffer geometry or device mismatch");
        }
        if width == 0 {
            return GpuBuffer::zeros_u32(&self.device, 1);
        }
        let mask = GpuBuffer::zeros_u32(&self.device, elements)?;
        if topk_blocks == 0 {
            return Ok(mask);
        }
        let push = Push {
            rows: rows as u32,
            width: width as u32,
            blocks: blocks as u32,
            block_size: block_size as u32,
            topk_blocks: topk_blocks.min(blocks) as u32,
        };
        let visible = GpuBuffer::from_u32_scheduled(&self.device, visible_lengths)?;
        let scores = GpuBuffer::zeros_f32(&self.device, rows * blocks)?;
        let mut commands = self.device.compute_batch()?;
        self.scores.record_dispatch(
            &mut commands,
            &[logits, &visible, &scores],
            bytemuck::bytes_of(&push),
            [blocks.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        let selected = self
            .topk
            .select_blocks(&scores, rows, blocks, topk_blocks)?;
        let mut commands = self.device.compute_batch()?;
        self.mask.record_dispatch(
            &mut commands,
            &[&selected, &scores, &mask],
            bytemuck::bytes_of(&push),
            [push.topk_blocks.div_ceil(64), rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(mask)
    }
}
