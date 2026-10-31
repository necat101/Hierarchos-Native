//! Shared compressed-KV publication. The released graph hands the compressor's
//! pre-RoPE latent to the indexer first, then rotates its tail at group
//! positions and FP4-quantizes it (16-element groups, E4M3 scales) before the
//! cache write. This module owns the second half of that order and never
//! mutates the latent, so an index-key producer can safely read it first.
use anyhow::{bail, Result};

use crate::{GpuBuffer, VulkanDevice};

use super::{
    DeepseekV41ActivationQuantizer, DeepseekV41CompressedLatents, DeepseekV41MixedLinearKernel,
    DeepseekV41QuantMode, DeepseekV41QuantizedActivation, DeepseekV41Rotary,
    DeepseekV41RotaryConfig, DeepseekV41ScaleMode, DeepseekV41WeightOutput,
};

pub const COMPRESSED_KV_QUANT_BLOCK: usize = 16;

/// Geometry/rounding contract of one finalized publication. Rounding is the
/// established FP32 STE edge; the remaining VJP is the RoPE transpose.
pub struct DeepseekV41CompressedKvTape {
    batch: usize,
    groups: usize,
    dim: usize,
    ratio: usize,
    position: usize,
    precision: DeepseekV41WeightOutput,
    rotary: DeepseekV41RotaryConfig,
}

pub struct DeepseekV41CompressedKvFinalizer {
    cast: DeepseekV41MixedLinearKernel,
    precision: DeepseekV41WeightOutput,
    rotary: DeepseekV41Rotary,
    quantizer: DeepseekV41ActivationQuantizer,
    ratio: usize,
    head_dim: usize,
    rope_dim: usize,
}

impl DeepseekV41CompressedKvFinalizer {
    pub fn prefill_retained(
        &self,
        latents: &DeepseekV41CompressedLatents,
    ) -> Result<(DeepseekV41QuantizedActivation, DeepseekV41CompressedKvTape)> {
        let output = self.finalize_prefill(latents)?;
        Ok((output, self.record(latents, 0)))
    }
    pub fn decode_retained(
        &self,
        latents: &DeepseekV41CompressedLatents,
        position: usize,
    ) -> Result<(DeepseekV41QuantizedActivation, DeepseekV41CompressedKvTape)> {
        let output = self.finalize_decode(latents, position)?;
        Ok((output, self.record(latents, position + 1 - self.ratio)))
    }
    fn record(
        &self,
        latents: &DeepseekV41CompressedLatents,
        position: usize,
    ) -> DeepseekV41CompressedKvTape {
        DeepseekV41CompressedKvTape {
            batch: latents.batch_size,
            groups: latents.groups_per_batch,
            dim: self.head_dim,
            ratio: self.ratio,
            position,
            precision: self.precision,
            rotary: self.rotary.config(),
        }
    }
    pub fn backward(
        &self,
        tape: &DeepseekV41CompressedKvTape,
        upstream: &GpuBuffer,
    ) -> Result<GpuBuffer> {
        if tape.dim != self.head_dim
            || tape.ratio != self.ratio
            || tape.precision != self.precision
            || tape.rotary != self.rotary.config()
        {
            bail!("V4.1 compressed finalizer backward configuration differs from its tape");
        }
        let rows = tape
            .batch
            .checked_mul(tape.groups)
            .ok_or_else(|| anyhow::anyhow!("V4.1 finalizer adjoint row overflow"))?;
        self.rotary.input_gradient_repeated(
            upstream,
            rows,
            tape.dim,
            1,
            tape.groups,
            tape.position,
            tape.ratio,
            false,
        )
    }
    pub fn new(
        device: VulkanDevice,
        rotary: DeepseekV41RotaryConfig,
        ratio: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new_with_precision(
            device,
            rotary,
            ratio,
            head_dim,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    pub fn new_with_precision(
        device: VulkanDevice,
        rotary: DeepseekV41RotaryConfig,
        ratio: usize,
        head_dim: usize,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        if !matches!(ratio, 1 | 2) {
            bail!("V4.1 compressed KV finalization requires ratio 1 or 2");
        }
        if head_dim == 0 || rotary.dim == 0 || rotary.dim > head_dim {
            bail!("V4.1 compressed KV rotary tail must fit a positive head dimension");
        }
        if !head_dim.is_multiple_of(COMPRESSED_KV_QUANT_BLOCK) {
            bail!("V4.1 compressed KV head dimension must be a multiple of 16");
        }
        let quantizer = DeepseekV41ActivationQuantizer::new(device.clone())?;
        let rope_dim = rotary.dim;
        let cast = DeepseekV41MixedLinearKernel::new(device.clone())?;
        let rotary = DeepseekV41Rotary::new(device, rotary)?;
        Ok(Self {
            cast,
            precision,
            rotary,
            quantizer,
            ratio,
            head_dim,
            rope_dim,
        })
    }

    pub fn ratio(&self) -> usize {
        self.ratio
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Rotate a prefill latent at group positions `0, ratio, 2*ratio, ...`.
    /// `latents` must begin at group zero; the compressor resets on prefill.
    pub fn finalize_prefill(
        &self,
        latents: &DeepseekV41CompressedLatents,
    ) -> Result<DeepseekV41QuantizedActivation> {
        if latents.first_group != 0 {
            bail!("V4.1 compressed KV prefill latent must begin at group zero");
        }
        self.finalize(latents, 0, latents.groups_per_batch)
    }

    /// Rotate one newly completed group per batch. Its group position is
    /// `start_pos + 1 - ratio`, matching the released `_compress_kv` rule.
    pub fn finalize_decode(
        &self,
        latents: &DeepseekV41CompressedLatents,
        start_pos: usize,
    ) -> Result<DeepseekV41QuantizedActivation> {
        if start_pos + 1 < self.ratio || !(start_pos + 1).is_multiple_of(self.ratio) {
            bail!("V4.1 compressed KV decode position does not complete a group");
        }
        if latents.first_group != start_pos / self.ratio || latents.groups_per_batch != 1 {
            bail!("V4.1 compressed KV decode latent does not match the completed group");
        }
        self.finalize(latents, start_pos + 1 - self.ratio, 1)
    }

    fn finalize(
        &self,
        latents: &DeepseekV41CompressedLatents,
        position_base: usize,
        groups_per_batch: usize,
    ) -> Result<DeepseekV41QuantizedActivation> {
        if latents.head_dim != self.head_dim {
            bail!("V4.1 compressed KV latent head dimension mismatch");
        }
        if latents.compress_ratio != self.ratio {
            bail!("V4.1 compressed KV latent compression ratio mismatch");
        }
        if latents.groups_per_batch != groups_per_batch || latents.batch_size == 0 {
            bail!("V4.1 compressed KV latent group shape mismatch");
        }
        let rows = latents
            .batch_size
            .checked_mul(groups_per_batch)
            .ok_or_else(|| anyhow::anyhow!("V4.1 compressed KV row count overflow"))?;
        // Read the latent through the public buffer; no host round trip.
        let rotated = self.rotary.apply_grouped_buffer(
            &latents.values,
            rows,
            self.head_dim,
            groups_per_batch,
            position_base,
            self.ratio,
            false,
        )?;
        let rotated = self
            .cast
            .cast_buffer(&rotated, rows * self.head_dim, self.precision)?;
        let mut quantized = self.quantizer.quantize_buffer(
            &rotated,
            rows,
            self.head_dim,
            COMPRESSED_KV_QUANT_BLOCK,
            DeepseekV41QuantMode::Fp4E2M1,
            DeepseekV41ScaleMode::E4m3,
        )?;
        quantized.values =
            self.cast
                .cast_buffer(&quantized.values, rows * self.head_dim, self.precision)?;
        Ok(quantized)
    }

    /// The released graph reads the index keys from the latent before the cache
    /// write. This helper exposes the same pre-RoPE byte range for a caller that
    /// is about to publish index keys; it does not copy or mutate.
    pub fn pre_rope_latent<'a>(&self, latents: &'a DeepseekV41CompressedLatents) -> &'a GpuBuffer {
        &latents.values
    }

    pub fn rope_dim(&self) -> usize {
        self.rope_dim
    }
}
