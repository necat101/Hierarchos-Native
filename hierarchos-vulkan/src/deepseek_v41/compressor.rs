//! Dedicated ratio-1/2 V4.1 compressor, including incomplete-group decode state.
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use super::checkpoint::DeepseekV41StreamedLayer;
use super::{DeepseekV41Checkpoint, DeepseekV41MixedLinearKernel, DeepseekV41WeightOutput};
use crate::{vulkan, GpuBuffer, VulkanDevice};

const LINEAR: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_linear.spv");
const POOL: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_compress.spv");
const NORM: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_rms_norm.spv");

#[path = "compressor_tape.rs"]
mod tape;
pub use tape::DeepseekV41CompressorTape;

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
struct PoolPush {
    batch: u32,
    seq: u32,
    dim: u32,
    ratio: u32,
    groups: u32,
    start_pos: u32,
    decode: u32,
    vector_width: u32,
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

pub struct DeepseekV41CompressorWeights {
    pub hidden_size: usize,
    pub head_dim: usize,
    pub compress_ratio: usize,
    pub eps: f32,
    /// Canonical row-major [head_dim, hidden_size].
    pub kv_weight: Vec<f32>,
    pub gate_weight: Option<Vec<f32>>,
    pub norm_weight: Vec<f32>,
}

/// Pre-RoPE, pre-quantization latents. Index-key projection must consume these
/// before a separate buffer is rotated and quantized for the attention cache.
pub struct DeepseekV41CompressedLatents {
    pub batch_size: usize,
    /// Compression ratio that produced these latents; consumers must match it.
    pub compress_ratio: usize,
    pub first_group: usize,
    pub groups_per_batch: usize,
    pub head_dim: usize,
    pub values: GpuBuffer,
    #[cfg(test)]
    pub(crate) pre_norm: GpuBuffer,
    #[cfg(test)]
    pub(crate) variance: GpuBuffer,
    #[cfg(test)]
    pub(crate) rstd: GpuBuffer,
    #[cfg(test)]
    pub(crate) projected_kv: GpuBuffer,
    #[cfg(test)]
    pub(crate) projected_gate: GpuBuffer,
}

impl DeepseekV41CompressedLatents {
    pub fn read_f32(&self) -> Result<Vec<f32>> {
        if self.groups_per_batch == 0 {
            return Ok(Vec::new());
        }
        self.values
            .read_f32(self.batch_size * self.groups_per_batch * self.head_dim)
    }
}

/// Stateful compressor with FP32 pooling and explicit F32/BF16 stream
/// boundaries. All projection/pooling/norm math and casts run on Vulkan.
struct CompressorProjection {
    kv_weight: GpuBuffer,
    gate_weight: Option<GpuBuffer>,
    norm_weight: GpuBuffer,
}

pub struct DeepseekV41Compressor {
    cast: DeepseekV41MixedLinearKernel,
    precision: DeepseekV41WeightOutput,
    device: VulkanDevice,
    hidden: usize,
    dim: usize,
    ratio: usize,
    eps: f32,
    max_batch: usize,
    max_prefill: usize,
    active_batch: Option<usize>,
    next_position: usize,
    projection: Option<CompressorProjection>,
    source: Option<DeepseekV41StreamedLayer>,
    kv_state: GpuBuffer,
    gate_state: GpuBuffer,
    linear: vulkan::ComputeKernel,
    pool: vulkan::ComputeKernel,
    norm: vulkan::ComputeKernel,
}

fn checked_size(label: &str, sizes: &[usize]) -> Result<usize> {
    let size = sizes
        .iter()
        .try_fold(1usize, |n, &s| n.checked_mul(s))
        .with_context(|| format!("V4.1 {label} size overflow"))?;
    if size == 0 || size > u32::MAX as usize {
        bail!("V4.1 {label} size must fit positive u32 addressing");
    }
    Ok(size)
}

impl DeepseekV41Compressor {
    pub fn new(
        device: VulkanDevice,
        weights: DeepseekV41CompressorWeights,
        max_batch: usize,
        max_prefill: usize,
    ) -> Result<Self> {
        Self::new_with_precision(
            device,
            weights,
            max_batch,
            max_prefill,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    pub fn new_with_precision(
        device: VulkanDevice,
        weights: DeepseekV41CompressorWeights,
        max_batch: usize,
        max_prefill: usize,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        let projection = checked_size(
            "compressor projection",
            &[weights.hidden_size, weights.head_dim],
        )?;
        checked_size(
            "compressor input",
            &[max_batch, max_prefill, weights.hidden_size],
        )?;
        checked_size(
            "compressor activations",
            &[max_batch, max_prefill, weights.head_dim],
        )?;
        let state = checked_size(
            "compressor state",
            &[max_batch, weights.compress_ratio, weights.head_dim],
        )?;
        if !matches!(weights.compress_ratio, 1 | 2)
            || !weights.eps.is_finite()
            || weights.eps <= 0.0
            || weights.kv_weight.len() != projection
            || weights.norm_weight.len() != weights.head_dim
            || (weights.compress_ratio == 2
                && weights
                    .gate_weight
                    .as_ref()
                    .is_none_or(|w| w.len() != projection))
            || (weights.compress_ratio == 1 && weights.gate_weight.is_some())
        {
            bail!("V4.1 compressor parameter geometry, ratio, or epsilon is invalid");
        }
        if precision == DeepseekV41WeightOutput::Bf16
            && weights.compress_ratio == 1
            && weights.kv_weight.iter().any(|v| v.to_bits() & 0xffff != 0)
        {
            bail!("V4.1 ratio-1 BF16 compressor requires BF16-exact weights");
        }
        Ok(Self {
            cast: DeepseekV41MixedLinearKernel::new(device.clone())?,
            precision,
            hidden: weights.hidden_size,
            dim: weights.head_dim,
            ratio: weights.compress_ratio,
            eps: weights.eps,
            max_batch,
            max_prefill,
            active_batch: None,
            next_position: 0,
            projection: Some(CompressorProjection {
                kv_weight: GpuBuffer::from_f32(&device, &weights.kv_weight)?,
                gate_weight: weights
                    .gate_weight
                    .as_ref()
                    .map(|w| GpuBuffer::from_f32(&device, w))
                    .transpose()?,
                norm_weight: GpuBuffer::from_f32(&device, &weights.norm_weight)?,
            }),
            source: None,
            kv_state: GpuBuffer::zeros_f32(&device, state)?,
            gate_state: GpuBuffer::zeros_f32(&device, state)?,
            linear: vulkan::ComputeKernel::new(
                &device,
                LINEAR,
                3,
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            pool: vulkan::ComputeKernel::new(
                &device,
                POOL,
                5,
                std::mem::size_of::<PoolPush>() as u32,
            )?,
            norm: vulkan::ComputeKernel::new(
                &device,
                NORM,
                5,
                std::mem::size_of::<NormPush>() as u32,
            )?,
            device,
        })
    }

    /// Compressor geometry without weight payloads. Weights are read from the
    /// checkpoint when the orchestrator enters the layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_streamed(
        device: VulkanDevice,
        hidden: usize,
        head_dim: usize,
        compress_ratio: usize,
        eps: f32,
        max_batch: usize,
        max_prefill: usize,
        precision: DeepseekV41WeightOutput,
        checkpoint: DeepseekV41Checkpoint,
        layer: usize,
    ) -> Result<Self> {
        checked_size("compressor projection", &[hidden, head_dim])?;
        checked_size("compressor input", &[max_batch, max_prefill, hidden])?;
        checked_size(
            "compressor activations",
            &[max_batch, max_prefill, head_dim],
        )?;
        let state = checked_size("compressor state", &[max_batch, compress_ratio, head_dim])?;
        if !matches!(compress_ratio, 1 | 2)
            || !eps.is_finite()
            || eps <= 0.0
            || max_batch == 0
            || max_prefill == 0
        {
            bail!("V4.1 compressor parameter geometry, ratio, or epsilon is invalid");
        }
        Ok(Self {
            cast: DeepseekV41MixedLinearKernel::new(device.clone())?,
            precision,
            hidden,
            dim: head_dim,
            ratio: compress_ratio,
            eps,
            max_batch,
            max_prefill,
            active_batch: None,
            next_position: 0,
            projection: None,
            source: Some(DeepseekV41StreamedLayer::new(checkpoint, layer)),
            kv_state: GpuBuffer::zeros_f32(&device, state)?,
            gate_state: GpuBuffer::zeros_f32(&device, state)?,
            linear: vulkan::ComputeKernel::new(
                &device,
                LINEAR,
                3,
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            pool: vulkan::ComputeKernel::new(
                &device,
                POOL,
                5,
                std::mem::size_of::<PoolPush>() as u32,
            )?,
            norm: vulkan::ComputeKernel::new(
                &device,
                NORM,
                5,
                std::mem::size_of::<NormPush>() as u32,
            )?,
            device,
        })
    }

    /// Reads and uploads this layer's compressor weights for one layer execution.
    pub(super) fn load_weights(&mut self) -> Result<()> {
        if self.projection.is_some() {
            return Ok(());
        }
        let Some(source) = self.source.as_ref() else {
            return Ok(());
        };
        let mut checkpoint = source.lock()?;
        let weights = checkpoint.load_compressor(source.layer())?;
        let projection = self.hidden * self.dim;
        if weights.hidden_size != self.hidden
            || weights.head_dim != self.dim
            || weights.compress_ratio != self.ratio
            || weights.eps != self.eps
            || weights.kv_weight.len() != projection
            || weights.norm_weight.len() != self.dim
            || weights.kv_weight.iter().any(|v| !v.is_finite())
            || weights.norm_weight.iter().any(|v| !v.is_finite())
            || weights
                .gate_weight
                .as_ref()
                .is_some_and(|w| w.len() != projection || w.iter().any(|v| !v.is_finite()))
            || (self.ratio == 1) != weights.gate_weight.is_none()
        {
            bail!("V4.1 streamed compressor geometry does not match its layer");
        }
        if self.precision == DeepseekV41WeightOutput::Bf16
            && self.ratio == 1
            && weights.kv_weight.iter().any(|v| v.to_bits() & 0xffff != 0)
        {
            bail!("V4.1 ratio-1 BF16 compressor requires BF16-exact weights");
        }
        self.projection = Some(CompressorProjection {
            kv_weight: GpuBuffer::from_f32(&self.device, &weights.kv_weight)?,
            gate_weight: weights
                .gate_weight
                .as_ref()
                .map(|w| GpuBuffer::from_f32(&self.device, w))
                .transpose()?,
            norm_weight: GpuBuffer::from_f32(&self.device, &weights.norm_weight)?,
        });
        Ok(())
    }

    /// The pool state a decode step left behind, as the forward holds it.
    /// Backward operands come from here, so a trainer reads the same buffers
    /// the forward wrote instead of reconstructing them.
    #[cfg(test)]
    pub(crate) fn pool_state(&self) -> (&GpuBuffer, &GpuBuffer) {
        (&self.kv_state, &self.gate_state)
    }

    pub(super) fn release_weights(&mut self) {
        if self.source.is_some() {
            self.projection = None;
        }
    }

    /// Starts a new sequence and replaces any unfinished group from a previous
    /// prefill. Only complete groups are returned, matching the released code.
    pub fn prefill(
        &mut self,
        input: &[f32],
        batch: usize,
        seq: usize,
    ) -> Result<DeepseekV41CompressedLatents> {
        let elements = self.input_elements(batch, seq, 0)?;
        if input.len() != elements {
            bail!("V4.1 compressor input length mismatch");
        }
        let input = GpuBuffer::from_f32(&self.device, input)?;
        self.prefill_from_buffer(&input, batch, seq)
    }

    /// Decode is one token per batch. Positions and batch shape must continue
    /// the current sequence exactly; an invalid call leaves cache metadata intact.
    pub fn decode_step(
        &mut self,
        input: &[f32],
        batch: usize,
        start_pos: usize,
    ) -> Result<DeepseekV41CompressedLatents> {
        let elements = self.input_elements(batch, 1, start_pos)?;
        if input.len() != elements {
            bail!("V4.1 compressor input length mismatch");
        }
        let input = GpuBuffer::from_f32(&self.device, input)?;
        self.decode_step_from_buffer(&input, batch, start_pos)
    }

    /// Consumes an upstream Vulkan tensor without a host readback/upload.
    pub fn prefill_from_buffer(
        &mut self,
        input: &GpuBuffer,
        batch: usize,
        seq: usize,
    ) -> Result<DeepseekV41CompressedLatents> {
        self.run(input, batch, seq, 0, false)
    }

    /// GPU-resident continuation of the current compressor sequence.
    pub fn decode_step_from_buffer(
        &mut self,
        input: &GpuBuffer,
        batch: usize,
        start_pos: usize,
    ) -> Result<DeepseekV41CompressedLatents> {
        if self.active_batch != Some(batch) || start_pos != self.next_position {
            bail!(
                "V4.1 compressor decode must continue the current batch at position {}",
                self.next_position
            );
        }
        self.run(input, batch, 1, start_pos, true)
    }

    fn input_elements(&self, batch: usize, seq: usize, start_pos: usize) -> Result<usize> {
        if batch == 0
            || batch > self.max_batch
            || seq == 0
            || seq > self.max_prefill
            || start_pos > u32::MAX as usize - seq
        {
            bail!("V4.1 compressor input dimensions or position exceed capacity");
        }
        Ok(batch * seq * self.hidden)
    }

    fn run(
        &mut self,
        input: &GpuBuffer,
        batch: usize,
        seq: usize,
        start_pos: usize,
        decode: bool,
    ) -> Result<DeepseekV41CompressedLatents> {
        self.run_recorded(input, batch, seq, start_pos, decode, false)
            .map(|(latents, _)| latents)
    }

    fn run_recorded(
        &mut self,
        input: &GpuBuffer,
        batch: usize,
        seq: usize,
        start_pos: usize,
        decode: bool,
        retain: bool,
    ) -> Result<(
        DeepseekV41CompressedLatents,
        Option<DeepseekV41CompressorTape>,
    )> {
        let elements = self.input_elements(batch, seq, start_pos)?;
        if input.f32_capacity() < elements || !input.belongs_to_device(&self.device) {
            bail!("V4.1 compressor input buffer is too small or belongs to another Vulkan device");
        }
        let projection = self
            .projection
            .as_ref()
            .context("V4.1 compressor projections are not loaded")?;
        let rounded = self.cast.cast_buffer(input, elements, self.precision)?;
        let input = &rounded;
        let rows = batch * seq;
        let groups = if decode {
            usize::from((start_pos + 1).is_multiple_of(self.ratio))
        } else {
            seq / self.ratio
        };
        let output_rows = batch * groups;
        if self.dim.div_ceil(16) > 65_535
            || rows.div_ceil(16) > 65_535
            || (batch * groups.max(1) * self.dim).div_ceil(64) > 65_535
        {
            bail!("V4.1 compressor exceeds dispatch capacity; use a smaller prefill");
        }
        let kv = GpuBuffer::zeros_f32(&self.device, rows * self.dim)?;
        let scores = GpuBuffer::zeros_f32(&self.device, rows * self.dim)?;
        let pooled = GpuBuffer::zeros_f32(&self.device, output_rows.max(1) * self.dim)?;
        let output = GpuBuffer::zeros_f32(&self.device, output_rows.max(1) * self.dim)?;
        let mean = GpuBuffer::zeros_f32(&self.device, output_rows.max(1))?;
        let rstd = GpuBuffer::zeros_f32(&self.device, output_rows.max(1))?;
        let mut commands = self.device.compute_batch()?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: self.hidden as u32,
            output_dim: self.dim as u32,
            bf16: u32::from(self.ratio == 1 && self.precision == DeepseekV41WeightOutput::Bf16),
        };
        let grid = [self.dim.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1];
        self.linear.record_dispatch(
            &mut commands,
            &[input, &projection.kv_weight, &kv],
            bytemuck::bytes_of(&push),
            grid,
        )?;
        if let Some(gate_weight) = &projection.gate_weight {
            self.linear.record_dispatch(
                &mut commands,
                &[input, gate_weight, &scores],
                bytemuck::bytes_of(&push),
                grid,
            )?;
        }
        let push = PoolPush {
            batch: batch as u32,
            seq: seq as u32,
            dim: self.dim as u32,
            ratio: self.ratio as u32,
            groups: groups as u32,
            start_pos: start_pos as u32,
            decode: decode as u32,
            vector_width: crate::transformer::aten_vector_width() as u32,
        };
        self.pool.record_dispatch(
            &mut commands,
            &[&kv, &scores, &self.kv_state, &self.gate_state, &pooled],
            bytemuck::bytes_of(&push),
            [(batch * groups.max(1) * self.dim).div_ceil(64) as u32, 1, 1],
        )?;
        // Ratio 1 rounds its dense projection before RMSNorm; ratio 2 pools
        // in FP32 then casts to the original stream dtype before RMSNorm.
        let pooled = if output_rows > 0 && self.precision == DeepseekV41WeightOutput::Bf16 {
            commands.submit()?;
            let rounded = self
                .cast
                .cast_buffer(&pooled, output_rows * self.dim, self.precision)?;
            commands = self.device.compute_batch()?;
            rounded
        } else {
            pooled
        };
        if output_rows > 0 {
            let push = NormPush {
                rows: output_rows as u32,
                dim: self.dim as u32,
                parameter_groups: 1,
                eps: self.eps,
                weight_offset: 0.0,
                pow_rstd: 0,
                unfused_products: 1,
                extra_rsqrt_refinement: 3,
            };
            self.norm.record_dispatch(
                &mut commands,
                &[&pooled, &projection.norm_weight, &output, &rstd, &mean],
                bytemuck::bytes_of(&push),
                [output_rows.div_ceil(64) as u32, 1, 1],
            )?;
        }
        commands.submit()?;
        let values = if output_rows > 0 {
            self.cast
                .cast_buffer(&output, output_rows * self.dim, self.precision)?
        } else {
            output
        };
        self.active_batch = Some(batch);
        self.next_position = start_pos + seq;
        let tape = if retain {
            Some(DeepseekV41CompressorTape::capture(
                self, batch, seq, start_pos, decode, &pooled, &kv, &scores,
            )?)
        } else {
            None
        };
        Ok((
            DeepseekV41CompressedLatents {
                batch_size: batch,
                compress_ratio: self.ratio,
                first_group: if decode { start_pos / self.ratio } else { 0 },
                groups_per_batch: groups,
                head_dim: self.dim,
                values,
                #[cfg(test)]
                pre_norm: pooled,
                #[cfg(test)]
                variance: mean,
                #[cfg(test)]
                rstd,
                #[cfg(test)]
                projected_kv: kv,
                #[cfg(test)]
                projected_gate: scores,
            },
            tape,
        ))
    }
}
