//! FP32 V4.1 compressor prefill adjoints.
//!
//! The forward is `compressor.rs` over three qualified kernels: the frozen
//! projections (`transformer_deepseek_v41_linear`), the ratio-2 softmax pool
//! (`transformer_deepseek_v41_compress`) and RMSNorm
//! (`transformer_deepseek_v41_rms_norm`). This module differentiates that chain
//! for one prefill evaluation, including the trailing incomplete group and the
//! pool-state routing the write performs. The projections are frozen in this
//! graph: adapters attach to the attention and model projections, so the
//! compressor contributes only the input gradient the block stream receives.
//!
//! Scope: FP32 operands. `prefill` covers `start_pos == 0`; `decode` covers one
//! cached step, whose pool reads the state slots earlier evaluations wrote, so
//! its truthful gradient is a chain across steps: each step returns the
//! gradient of the state it inherited, which is the incoming state gradient of
//! the step before it. BF16/FP8 activation boundaries remain a separate
//! qualification and are rejected here rather than approximated.

use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

#[cfg(test)]
#[path = "compressor_grad_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "compressor_decode_grad_tests.rs"]
mod decode_tests;

const LINEAR_INPUT_GRAD: &[u8] = include_bytes!("../../shaders/linear_input_grad.spv");
// CPU `mm` switches reduction order with the row count; the decode shapes are
// one or two rows, which the serial kernel does not reproduce (see the decode
// phase's reduction note). Prefill keeps the serial kernel its fixtures gate.
const LINEAR_INPUT_GRAD_BLOCK8: &[u8] =
    include_bytes!("../../shaders/linear_input_grad_block8.spv");
const LINEAR_INPUT_GRAD_GROUP4: &[u8] =
    include_bytes!("../../shaders/linear_input_grad_group4_muladd.spv");
const NORM_INPUT_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_rms_norm_input_grad.spv");
const POOL_GRAD: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_compress_grad.spv");
const STATE_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_compress_state_grad.spv");
const DECODE_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_compress_decode_grad.spv");
const VECTOR: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_projection_grad_vector.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormPush {
    rows: u32,
    dim: u32,
    eps: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PoolPush {
    batch: u32,
    seq: u32,
    dim: u32,
    ratio: u32,
    groups: u32,
    tail: u32,
    written_slot: u32,
    has_slot_grad: u32,
    vector_width: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StatePush {
    batch: u32,
    ratio: u32,
    dim: u32,
    written_slot: u32,
    has_write: u32,
    has_in: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DecodePush {
    batch: u32,
    dim: u32,
    ratio: u32,
    written_slot: u32,
    groups: u32,
    has_state_grad: u32,
    vector_width: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VectorPush {
    len: u32,
    scale: f32,
    add: u32,
}

/// Geometry of one compressor prefill evaluation. `start_pos` names the
/// evaluation the operands came from and must be 0 for this adjoint.
#[derive(Clone, Copy, Debug)]
pub struct DeepseekV41CompressorPrefill {
    pub batch: usize,
    pub seq: usize,
    pub start_pos: usize,
    pub hidden: usize,
    pub dim: usize,
    pub ratio: usize,
    pub eps: f32,
}

/// Geometry of one cached decode step: `batch` sequences each consume one
/// token at `start_pos`, continuing the sequence the forward already saw.
#[derive(Clone, Copy, Debug)]
pub struct DeepseekV41CompressorDecode {
    pub batch: usize,
    pub start_pos: usize,
    pub hidden: usize,
    pub dim: usize,
    pub ratio: usize,
    pub eps: f32,
}

/// Retained operands of the evaluation being differentiated. Every buffer is
/// immutable and stays owned by the caller, so a compressor can release its
/// projections between the forward and the backward.
pub struct DeepseekV41CompressorOperands<'a> {
    /// The buffer the forward handed to RMSNorm: `[batch, groups, dim]`.
    pub pooled: &'a GpuBuffer,
    /// Frozen projection output before pooling: `[batch, seq, dim]`.
    pub projected_kv: &'a GpuBuffer,
    /// Gate projection output, ratio 2 only: `[batch, seq, dim]`.
    pub projected_gate: Option<&'a GpuBuffer>,
    /// Frozen projection weights, canonical row-major `[dim, hidden]`.
    pub kv_weight: &'a GpuBuffer,
    pub gate_weight: Option<&'a GpuBuffer>,
    /// Frozen RMSNorm gain: `[dim]`.
    pub norm_weight: &'a GpuBuffer,
}

pub struct DeepseekV41CompressorGradients {
    /// `[batch, seq, hidden]`: the gradient the block stream receives.
    pub grad_input: GpuBuffer,
    /// Gradient with respect to the pool state before the evaluation, as the
    /// forward's own `(kv_state, score_state)` pair. `Some` for ratio 2 only.
    pub grad_state_kv: Option<GpuBuffer>,
    pub grad_state_score: Option<GpuBuffer>,
}

/// Operands of one decode step. `state_kv`/`state_score` are the pool state the
/// step left behind (`[batch, ratio, dim]`), because the written slot holds the
/// step's own row verbatim; the untouched slot holds the value an earlier step
/// wrote and is the only path to it.
pub struct DeepseekV41CompressorDecodeOperands<'a> {
    /// The buffer the forward handed to RMSNorm: `[batch, 1, dim]` when the
    /// step completed a group, unused otherwise.
    pub pooled: &'a GpuBuffer,
    /// Frozen projection output of this step: `[batch, 1, dim]`.
    pub projected_kv: &'a GpuBuffer,
    pub projected_gate: Option<&'a GpuBuffer>,
    /// Pool state after the step wrote its slot: `[batch, ratio, dim]`.
    pub state_kv: Option<&'a GpuBuffer>,
    pub state_score: Option<&'a GpuBuffer>,
    /// Frozen projection weights, canonical row-major `[dim, hidden]`.
    pub kv_weight: &'a GpuBuffer,
    pub gate_weight: Option<&'a GpuBuffer>,
    /// Frozen RMSNorm gain: `[dim]`.
    pub norm_weight: &'a GpuBuffer,
}

pub struct DeepseekV41CompressorAdjoints {
    device: VulkanDevice,
    norm_input_grad: vulkan::ComputeKernel,
    pool_grad: vulkan::ComputeKernel,
    state_grad: vulkan::ComputeKernel,
    decode_grad: vulkan::ComputeKernel,
    input_grad: vulkan::ComputeKernel,
    input_grad_block8: vulkan::ComputeKernel,
    input_grad_group4: vulkan::ComputeKernel,
    vector: vulkan::ComputeKernel,
}

fn kernel(
    device: &VulkanDevice,
    bytes: &[u8],
    reads: usize,
    writes: usize,
    push: usize,
) -> Result<vulkan::ComputeKernel> {
    let mut access = vec![vulkan::BindingAccess::ReadOnly; reads];
    access.extend(vec![vulkan::BindingAccess::WriteOnly; writes]);
    vulkan::ComputeKernel::new_with_access(device, bytes, &access, push as u32)
}
fn count(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| a != 0 && b != 0 && n <= u32::MAX as usize)
        .context("V4.1 compressor adjoint has zero/overflowing geometry")
}
fn grid(len: usize) -> [u32; 3] {
    let groups = len.div_ceil(64);
    [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1]
}
fn input_grid(rows: usize, hidden: usize) -> [u32; 3] {
    [hidden.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1]
}

impl DeepseekV41CompressorAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            norm_input_grad: kernel(
                &device,
                NORM_INPUT_GRAD,
                3,
                1,
                std::mem::size_of::<NormPush>(),
            )?,
            // grad_pooled, kv, scores, slot kv, slot score -> grad kv, grad score
            pool_grad: kernel(&device, POOL_GRAD, 5, 2, std::mem::size_of::<PoolPush>())?,
            // slot kv, slot score -> grad state kv, grad state score
            state_grad: kernel(&device, STATE_GRAD, 2, 2, std::mem::size_of::<StatePush>())?,
            // grad_pooled, state kv/score, incoming state kv/score -> row and
            // previous-state gradients for one step
            decode_grad: kernel(
                &device,
                DECODE_GRAD,
                5,
                4,
                std::mem::size_of::<DecodePush>(),
            )?,
            input_grad: kernel(
                &device,
                LINEAR_INPUT_GRAD,
                2,
                1,
                std::mem::size_of::<LinearPush>(),
            )?,
            input_grad_block8: kernel(
                &device,
                LINEAR_INPUT_GRAD_BLOCK8,
                2,
                1,
                std::mem::size_of::<LinearPush>(),
            )?,
            input_grad_group4: kernel(
                &device,
                LINEAR_INPUT_GRAD_GROUP4,
                2,
                1,
                std::mem::size_of::<LinearPush>(),
            )?,
            vector: kernel(&device, VECTOR, 2, 1, std::mem::size_of::<VectorPush>())?,
            device,
        })
    }

    fn check(&self, label: &str, buffer: &GpuBuffer, len: usize) -> Result<()> {
        if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < len {
            bail!("V4.1 compressor adjoint {label} geometry or device mismatch");
        }
        Ok(())
    }

    /// Differentiate one FP32 prefill evaluation.
    ///
    /// `upstream` is `d loss / d latents` over the emitted groups
    /// (`[batch, groups, dim]`, empty when the evaluation emitted none).
    /// `state_gradient` is `d loss / d pool state` after the evaluation, as
    /// `(kv_state, score_state)`. Without it, a slot this evaluation wrote has
    /// no gradient (its only path to a loss is a later evaluation) and the
    /// untouched slots return zero; with it, untouched slots pass the incoming
    /// gradient through and the trailing partial row receives the gradient of
    /// the slot it was written to.
    pub fn prefill(
        &self,
        prefill: &DeepseekV41CompressorPrefill,
        operands: &DeepseekV41CompressorOperands<'_>,
        upstream: &GpuBuffer,
        state_gradient: Option<(&GpuBuffer, &GpuBuffer)>,
    ) -> Result<DeepseekV41CompressorGradients> {
        let p = *prefill;
        if p.start_pos != 0 {
            bail!(
                "V4.1 compressor adjoint covers prefill evaluations; a decode step pools from slots written by earlier steps"
            );
        }
        if p.batch == 0 || p.seq == 0 || p.hidden == 0 || p.dim == 0 {
            bail!("V4.1 compressor adjoint geometry must be positive");
        }
        if !matches!(p.ratio, 1 | 2) {
            bail!("V4.1 compressor adjoint ratio must be 1 or 2");
        }
        if !p.eps.is_finite() || p.eps <= 0.0 {
            bail!("V4.1 compressor adjoint epsilon is invalid");
        }
        if p.hidden.div_ceil(16) > 65_535 || p.dim.div_ceil(16) > 65_535 {
            bail!("V4.1 compressor adjoint exceeds dispatch capacity");
        }
        let rows = count(p.batch, p.seq)?;
        if rows.div_ceil(16) > 65_535 {
            bail!("V4.1 compressor adjoint row count exceeds dispatch capacity");
        }
        let groups = p.seq / p.ratio;
        let tail = usize::from(!p.seq.is_multiple_of(p.ratio));
        // An evaluation shorter than one group emits nothing, so these two
        // counts are legitimately zero rather than invalid geometry.
        let pooled_rows = p
            .batch
            .checked_mul(groups)
            .filter(|&n| n <= u32::MAX as usize)
            .context("V4.1 compressor adjoint pooled row count overflow")?;
        let output_len = pooled_rows
            .checked_mul(p.dim)
            .filter(|&n| n <= u32::MAX as usize)
            .context("V4.1 compressor adjoint pooled element count overflow")?;
        let state_len = count(count(p.batch, p.ratio)?, p.dim)?;
        let projection = count(p.dim, p.hidden)?;
        let activations = count(rows, p.dim)?;

        self.check("projected kv", operands.projected_kv, activations)?;
        self.check("kv weight", operands.kv_weight, projection)?;
        self.check("norm weight", operands.norm_weight, p.dim)?;
        self.check("pooled input", operands.pooled, output_len.max(p.dim))?;
        self.check("upstream", upstream, output_len)?;
        match p.ratio {
            1 => {
                if operands.projected_gate.is_some()
                    || operands.gate_weight.is_some()
                    || state_gradient.is_some()
                {
                    bail!("V4.1 ratio-1 compressor has no gate projection or pool state");
                }
            }
            _ => {
                let gate = operands
                    .projected_gate
                    .context("V4.1 ratio-2 compressor adjoint requires its gate projection")?;
                let gate_weight = operands
                    .gate_weight
                    .context("V4.1 ratio-2 compressor adjoint requires its gate weight")?;
                self.check("projected gate", gate, activations)?;
                self.check("gate weight", gate_weight, projection)?;
                if let Some((kv, score)) = state_gradient {
                    self.check("state kv gradient", kv, state_len)?;
                    self.check("state score gradient", score, state_len)?;
                }
            }
        }

        let grad_pooled = GpuBuffer::zeros_f32(&self.device, output_len.max(1))?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        if pooled_rows != 0 {
            let push = NormPush {
                rows: pooled_rows as u32,
                dim: p.dim as u32,
                eps: p.eps,
            };
            self.norm_input_grad.record_dispatch(
                &mut commands,
                &[
                    upstream,
                    operands.pooled,
                    operands.norm_weight,
                    &grad_pooled,
                ],
                bytemuck::bytes_of(&push),
                [pooled_rows as u32, 1, 1],
            )?;
        }

        // Ratio 1 pools nothing: the projection output is the norm input, so
        // its gradient is already laid out as the projection's gradient.
        let (grad_kv, grad_score, grad_state_kv, grad_state_score) = if p.ratio == 1 {
            commands.submit()?;
            (grad_pooled, None, None, None)
        } else {
            let grad_projection = GpuBuffer::zeros_f32(&self.device, activations)?;
            let grad_gate = GpuBuffer::zeros_f32(&self.device, activations)?;
            let empty = GpuBuffer::zeros_f32(&self.device, 1)?;
            let (slot_kv, slot_score) = match state_gradient {
                Some((kv, score)) => (kv, score),
                None => (&empty, &empty),
            };
            let work_rows = groups + tail;
            if work_rows != 0 {
                let push = PoolPush {
                    batch: p.batch as u32,
                    seq: p.seq as u32,
                    dim: p.dim as u32,
                    ratio: p.ratio as u32,
                    groups: groups as u32,
                    tail: tail as u32,
                    written_slot: 0,
                    has_slot_grad: u32::from(state_gradient.is_some()),
                    vector_width: crate::transformer::aten_vector_width() as u32,
                };
                self.pool_grad.record_dispatch(
                    &mut commands,
                    &[
                        &grad_pooled,
                        operands.projected_kv,
                        operands.projected_gate.expect("ratio 2 gate checked above"),
                        slot_kv,
                        slot_score,
                        &grad_projection,
                        &grad_gate,
                    ],
                    bytemuck::bytes_of(&push),
                    grid(count(count(p.batch, work_rows)?, p.dim)?),
                )?;
            }
            let state_kv = GpuBuffer::zeros_f32(&self.device, state_len)?;
            let state_score = GpuBuffer::zeros_f32(&self.device, state_len)?;
            let push = StatePush {
                batch: p.batch as u32,
                ratio: p.ratio as u32,
                dim: p.dim as u32,
                written_slot: 0,
                has_write: u32::from(tail != 0),
                has_in: u32::from(state_gradient.is_some()),
            };
            self.state_grad.record_dispatch(
                &mut commands,
                &[slot_kv, slot_score, &state_kv, &state_score],
                bytemuck::bytes_of(&push),
                grid(state_len),
            )?;
            commands.submit()?;
            (
                grad_projection,
                Some(grad_gate),
                Some(state_kv),
                Some(state_score),
            )
        };

        // The frozen projections contribute input gradients only. PyTorch sums
        // the two matmul contributions into `x.grad`; FP32 addition is exact
        // under reordering, so the native add matches either engine order.
        let grad_kv_input = GpuBuffer::zeros_f32(&self.device, count(rows, p.hidden)?)?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: p.hidden as u32,
            output_dim: p.dim as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.input_grad.record_dispatch(
            &mut commands,
            &[&grad_kv, operands.kv_weight, &grad_kv_input],
            bytemuck::bytes_of(&push),
            input_grid(rows, p.hidden),
        )?;
        let grad_input = match (grad_score, operands.gate_weight) {
            (Some(grad_gate), Some(gate_weight)) => {
                let grad_gate_input = GpuBuffer::zeros_f32(&self.device, count(rows, p.hidden)?)?;
                self.input_grad.record_dispatch(
                    &mut commands,
                    &[&grad_gate, gate_weight, &grad_gate_input],
                    bytemuck::bytes_of(&push),
                    input_grid(rows, p.hidden),
                )?;
                let combined = GpuBuffer::zeros_f32(&self.device, count(rows, p.hidden)?)?;
                let add = VectorPush {
                    len: count(rows, p.hidden)? as u32,
                    scale: 1.0,
                    add: 1,
                };
                self.vector.record_dispatch(
                    &mut commands,
                    &[&grad_kv_input, &grad_gate_input, &combined],
                    bytemuck::bytes_of(&add),
                    grid(add.len as usize),
                )?;
                combined
            }
            _ => grad_kv_input,
        };
        commands.submit()?;
        Ok(DeepseekV41CompressorGradients {
            grad_input,
            grad_state_kv,
            grad_state_score,
        })
    }

    /// Differentiate one cached decode step.
    ///
    /// `upstream` is `d loss / d latents` for the group this step completes
    /// (`[batch, 1, dim]`); a step that completes no group does not read it.
    /// `state_gradient` is `d loss / d pool state` after this step, and the
    /// returned `grad_state_*` is the gradient with respect to the state the
    /// step inherited -- the incoming gradient of the preceding step. Replaying
    /// a decode sequence backwards and feeding each result into the step before
    /// it is the truthful cross-step chain; the written slot returns zero to
    /// the previous state while the untouched slot passes the incoming gradient
    /// through plus its pool term.
    ///
    /// Ratio 1 has no gate and its state is never read by the forward, so there
    /// is nothing to chain: a state operand or gradient is rejected instead of
    /// pretending a path exists.
    pub fn decode(
        &self,
        decode: &DeepseekV41CompressorDecode,
        operands: &DeepseekV41CompressorDecodeOperands<'_>,
        upstream: &GpuBuffer,
        state_gradient: Option<(&GpuBuffer, &GpuBuffer)>,
    ) -> Result<DeepseekV41CompressorGradients> {
        let p = *decode;
        if p.start_pos == 0 {
            bail!(
                "V4.1 compressor decode adjoint covers cached steps; position zero rewrites the sequence as a prefill"
            );
        }
        if p.batch == 0 || p.hidden == 0 || p.dim == 0 {
            bail!("V4.1 compressor decode adjoint geometry must be positive");
        }
        if !matches!(p.ratio, 1 | 2) {
            bail!("V4.1 compressor decode adjoint ratio must be 1 or 2");
        }
        if !p.eps.is_finite() || p.eps <= 0.0 {
            bail!("V4.1 compressor decode adjoint epsilon is invalid");
        }
        if p.start_pos >= u32::MAX as usize {
            bail!("V4.1 compressor decode adjoint position exceeds u32 addressing");
        }
        if p.hidden.div_ceil(16) > 65_535 || p.dim.div_ceil(16) > 65_535 {
            bail!("V4.1 compressor decode adjoint exceeds dispatch capacity");
        }
        if p.batch.div_ceil(16) > 65_535 {
            bail!("V4.1 compressor decode adjoint row count exceeds dispatch capacity");
        }
        let groups = if p.ratio == 1 {
            1
        } else {
            usize::from((p.start_pos + 1).is_multiple_of(p.ratio))
        };
        let activations = count(p.batch, p.dim)?;
        // A step that completes no group legitimately emits nothing.
        let output_len = if groups == 0 {
            0
        } else {
            count(activations, groups)?
        };
        let state_len = count(count(p.batch, p.ratio)?, p.dim)?;
        let projection = count(p.dim, p.hidden)?;
        if groups != 0 {
            self.check("pooled input", operands.pooled, output_len)?;
            self.check("upstream", upstream, output_len)?;
        }
        self.check("projected kv", operands.projected_kv, activations)?;
        self.check("kv weight", operands.kv_weight, projection)?;
        self.check("norm weight", operands.norm_weight, p.dim)?;
        match p.ratio {
            1 => {
                if operands.projected_gate.is_some()
                    || operands.gate_weight.is_some()
                    || operands.state_kv.is_some()
                    || operands.state_score.is_some()
                    || state_gradient.is_some()
                {
                    bail!("V4.1 ratio-1 compressor has no gate projection and never reads its pool state");
                }
            }
            _ => {
                let gate = operands.projected_gate.context(
                    "V4.1 ratio-2 compressor decode adjoint requires its gate projection",
                )?;
                let gate_weight = operands
                    .gate_weight
                    .context("V4.1 ratio-2 compressor decode adjoint requires its gate weight")?;
                let state_kv = operands.state_kv.context(
                    "V4.1 ratio-2 compressor decode adjoint requires the state after the step",
                )?;
                let state_score = operands
                    .state_score
                    .context("V4.1 ratio-2 compressor decode adjoint requires the score state after the step")?;
                self.check("projected gate", gate, activations)?;
                self.check("gate weight", gate_weight, projection)?;
                self.check("state kv", state_kv, state_len)?;
                self.check("state score", state_score, state_len)?;
                if let Some((kv, score)) = state_gradient {
                    self.check("state kv gradient", kv, state_len)?;
                    self.check("state score gradient", score, state_len)?;
                }
            }
        }

        let grad_pooled = GpuBuffer::zeros_f32(&self.device, output_len.max(1))?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        if groups != 0 {
            let push = NormPush {
                rows: p.batch as u32,
                dim: p.dim as u32,
                eps: p.eps,
            };
            self.norm_input_grad.record_dispatch(
                &mut commands,
                &[
                    upstream,
                    operands.pooled,
                    operands.norm_weight,
                    &grad_pooled,
                ],
                bytemuck::bytes_of(&push),
                [p.batch as u32, 1, 1],
            )?;
        }

        let (grad_kv, grad_score, grad_state_kv, grad_state_score) = if p.ratio == 1 {
            (grad_pooled, None, None, None)
        } else {
            let grad_kv_row = GpuBuffer::zeros_f32(&self.device, activations)?;
            let grad_score_row = GpuBuffer::zeros_f32(&self.device, activations)?;
            let state_kv_grad = GpuBuffer::zeros_f32(&self.device, state_len)?;
            let state_score_grad = GpuBuffer::zeros_f32(&self.device, state_len)?;
            let empty = GpuBuffer::zeros_f32(&self.device, 1)?;
            let (in_kv, in_score) = match state_gradient {
                Some((kv, score)) => (kv, score),
                None => (&empty, &empty),
            };
            let push = DecodePush {
                batch: p.batch as u32,
                dim: p.dim as u32,
                ratio: p.ratio as u32,
                written_slot: (p.start_pos % p.ratio) as u32,
                groups: groups as u32,
                has_state_grad: u32::from(state_gradient.is_some()),
                vector_width: crate::transformer::aten_vector_width() as u32,
            };
            self.decode_grad.record_dispatch(
                &mut commands,
                &[
                    &grad_pooled,
                    operands.state_kv.expect("ratio-2 state checked above"),
                    operands
                        .state_score
                        .expect("ratio-2 score state checked above"),
                    in_kv,
                    in_score,
                    &grad_kv_row,
                    &grad_score_row,
                    &state_kv_grad,
                    &state_score_grad,
                ],
                bytemuck::bytes_of(&push),
                grid(activations),
            )?;
            (
                grad_kv_row,
                Some(grad_score_row),
                Some(state_kv_grad),
                Some(state_score_grad),
            )
        };

        let rows = p.batch;
        let grad_kv_input = GpuBuffer::zeros_f32(&self.device, count(rows, p.hidden)?)?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: p.hidden as u32,
            output_dim: p.dim as u32,
        };
        // ATen's CPU `mm` walks the reduction in eight-element blocks for one
        // row and in four-product groups for two; the serial kernel is the
        // order qualified above that, so select the path these rows take.
        let input_grad = if rows == 1 {
            &self.input_grad_block8
        } else if rows == 2 && p.dim % 4 == 0 {
            &self.input_grad_group4
        } else {
            &self.input_grad
        };
        input_grad.record_dispatch(
            &mut commands,
            &[&grad_kv, operands.kv_weight, &grad_kv_input],
            bytemuck::bytes_of(&push),
            input_grid(rows, p.hidden),
        )?;
        let grad_input = match (grad_score, operands.gate_weight) {
            (Some(grad_gate), Some(gate_weight)) => {
                let grad_gate_input = GpuBuffer::zeros_f32(&self.device, count(rows, p.hidden)?)?;
                input_grad.record_dispatch(
                    &mut commands,
                    &[&grad_gate, gate_weight, &grad_gate_input],
                    bytemuck::bytes_of(&push),
                    input_grid(rows, p.hidden),
                )?;
                let combined = GpuBuffer::zeros_f32(&self.device, count(rows, p.hidden)?)?;
                let add = VectorPush {
                    len: count(rows, p.hidden)? as u32,
                    scale: 1.0,
                    add: 1,
                };
                self.vector.record_dispatch(
                    &mut commands,
                    &[&grad_kv_input, &grad_gate_input, &combined],
                    bytemuck::bytes_of(&add),
                    grid(add.len as usize),
                )?;
                combined
            }
            _ => grad_kv_input,
        };
        commands.submit()?;
        Ok(DeepseekV41CompressorGradients {
            grad_input,
            grad_state_kv,
            grad_state_score,
        })
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
}
