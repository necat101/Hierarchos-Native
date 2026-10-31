//! Forward operands and state snapshots from the actual compressor execution.
//! Frozen projection payloads are supplied again for backward, not retained.
use super::*;
use crate::deepseek_v41::{
    DeepseekV41CompressorAdjoints, DeepseekV41CompressorDecode,
    DeepseekV41CompressorDecodeOperands, DeepseekV41CompressorGradients,
    DeepseekV41CompressorOperands, DeepseekV41CompressorPrefill,
};

pub struct DeepseekV41CompressorTape {
    batch: usize,
    seq: usize,
    position: usize,
    decode: bool,
    hidden: usize,
    dim: usize,
    ratio: usize,
    eps: f32,
    precision: DeepseekV41WeightOutput,
    pooled: GpuBuffer,
    kv: GpuBuffer,
    gate: Option<GpuBuffer>,
    state_kv: Option<GpuBuffer>,
    state_gate: Option<GpuBuffer>,
    norm: GpuBuffer,
}
impl DeepseekV41CompressorTape {
    pub(super) fn capture(
        c: &DeepseekV41Compressor,
        batch: usize,
        seq: usize,
        position: usize,
        decode: bool,
        pooled: &GpuBuffer,
        kv: &GpuBuffer,
        gate: &GpuBuffer,
    ) -> Result<Self> {
        let p = c
            .projection
            .as_ref()
            .context("V4.1 retained compressor has no projections")?;
        let n = batch * seq * c.dim;
        // Projection/pool outputs are immutable local allocations. State is
        // overwritten by the next token and must be a physical GPU snapshot.
        Ok(Self {
            batch,
            seq,
            position,
            decode,
            hidden: c.hidden,
            dim: c.dim,
            ratio: c.ratio,
            eps: c.eps,
            precision: c.precision,
            pooled: pooled.clone(),
            kv: kv.clone(),
            gate: (c.ratio == 2).then(|| gate.clone()),
            state_kv: if decode && c.ratio == 2 {
                Some(c.kv_state.duplicate_f32(batch * c.ratio * c.dim)?)
            } else {
                None
            },
            state_gate: if decode && c.ratio == 2 {
                Some(c.gate_state.duplicate_f32(batch * c.ratio * c.dim)?)
            } else {
                None
            },
            norm: p.norm_weight.duplicate_f32(c.dim)?,
        })
        .and_then(|t| {
            if t.kv.f32_capacity() < n {
                bail!("V4.1 retained compressor projection is undersized");
            }
            Ok(t)
        })
    }
    pub fn groups_per_batch(&self) -> usize {
        if self.decode {
            usize::from((self.position + 1).is_multiple_of(self.ratio))
        } else {
            self.seq / self.ratio
        }
    }
    /// Use the same frozen compressor, with its weights loaded/restaged.
    /// `state` is the adjoint returned by the following cached step.
    pub fn backward(
        &self,
        c: &DeepseekV41Compressor,
        adjoints: &DeepseekV41CompressorAdjoints,
        upstream: Option<&GpuBuffer>,
        state: Option<(&GpuBuffer, &GpuBuffer)>,
    ) -> Result<DeepseekV41CompressorGradients> {
        if (c.hidden, c.dim, c.ratio, c.eps, c.precision)
            != (self.hidden, self.dim, self.ratio, self.eps, self.precision)
            || !adjoints.device().shares_logical_device_with(&c.device)
        {
            bail!("V4.1 retained compressor backward configuration/device mismatch");
        }
        let p = c
            .projection
            .as_ref()
            .context("V4.1 compressor backward requires restaged frozen projections")?;
        let zero;
        let upstream = match upstream {
            Some(g) => g,
            None => {
                zero = GpuBuffer::zeros_f32(
                    &c.device,
                    (self.batch * self.groups_per_batch() * self.dim).max(1),
                )?;
                &zero
            }
        };
        if self.decode {
            adjoints.decode(
                &DeepseekV41CompressorDecode {
                    batch: self.batch,
                    start_pos: self.position,
                    hidden: self.hidden,
                    dim: self.dim,
                    ratio: self.ratio,
                    eps: self.eps,
                },
                &DeepseekV41CompressorDecodeOperands {
                    pooled: &self.pooled,
                    projected_kv: &self.kv,
                    projected_gate: self.gate.as_ref(),
                    state_kv: self.state_kv.as_ref(),
                    state_score: self.state_gate.as_ref(),
                    kv_weight: &p.kv_weight,
                    gate_weight: p.gate_weight.as_ref(),
                    norm_weight: &self.norm,
                },
                upstream,
                state,
            )
        } else {
            adjoints.prefill(
                &DeepseekV41CompressorPrefill {
                    batch: self.batch,
                    seq: self.seq,
                    start_pos: 0,
                    hidden: self.hidden,
                    dim: self.dim,
                    ratio: self.ratio,
                    eps: self.eps,
                },
                &DeepseekV41CompressorOperands {
                    pooled: &self.pooled,
                    projected_kv: &self.kv,
                    projected_gate: self.gate.as_ref(),
                    kv_weight: &p.kv_weight,
                    gate_weight: p.gate_weight.as_ref(),
                    norm_weight: &self.norm,
                },
                upstream,
                state,
            )
        }
    }
}
impl DeepseekV41Compressor {
    pub fn prefill_retained(
        &mut self,
        input: &GpuBuffer,
        batch: usize,
        seq: usize,
    ) -> Result<(DeepseekV41CompressedLatents, DeepseekV41CompressorTape)> {
        let (latents, tape) = self.run_recorded(input, batch, seq, 0, false, true)?;
        Ok((latents, tape.unwrap()))
    }
    pub fn decode_retained(
        &mut self,
        input: &GpuBuffer,
        batch: usize,
        position: usize,
    ) -> Result<(DeepseekV41CompressedLatents, DeepseekV41CompressorTape)> {
        if self.active_batch != Some(batch) || self.next_position != position {
            bail!(
                "V4.1 retained compressor decode requires the next position of the current batch"
            );
        }
        let (latents, tape) = self.run_recorded(input, batch, 1, position, true, true)?;
        Ok((latents, tape.unwrap()))
    }
}
