//! Retained projection operands and native LoRA/rsLoRA VJPs.
//!
//! Forward reuses MixedLinearKernel::apply. A tape owns immutable input/factor
//! snapshots, never the frozen base matrix. Backward accepts that matrix again
//! so an offloaded caller can restage it only while its input adjoint runs.
//!
//! Storage coverage: FP32 dense, BF16 dense and packed (FP8 block32/block128,
//! FP4 row32). `forward_mixed` evaluates the released mixed boundary (stream
//! cast, block-scaled GEMM, one BF16 output rounding, PEFT merge) and
//! `backward` contracts the input gradient against the decoded, weight-scaled
//! matrix. The released activation quantizer is a frozen rounding boundary
//! whose straight-through edge divides by an exact UE8M0 power-of-two scale;
//! that division cancels in the input adjoint, which is why only the decoded
//! matrix is needed. This contract is pinned by
//! `validation/validate_deepseek_v41_packed_grad.py`. Dropout and
//! quantization-aware weight gradients remain out of scope.

use super::{
    DeepseekV41MixedLinearKernel, DeepseekV41ResidentLinear, DeepseekV41ResidentLora, LinearStorage,
};
use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

#[cfg(test)]
#[path = "linear_grad_tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VectorPush {
    len: u32,
    scale: f32,
    add: u32,
}

/// Immutable operands of one FP32 projection evaluation. Adapter factors are
/// snapshots of this evaluation, not references to later optimizer values.
pub struct DeepseekV41LinearTape {
    pub output: GpuBuffer,
    input: GpuBuffer,
    lora: Option<DeepseekV41ResidentLora>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
}
pub struct DeepseekV41LinearGradients {
    pub grad_input: GpuBuffer,
    pub grad_a: Option<GpuBuffer>,
    pub grad_b: Option<GpuBuffer>,
    a_len: usize,
    b_len: usize,
}
impl DeepseekV41LinearGradients {
    /// Read only the small adapter gradients for the native AdamW state.
    /// `target` is the canonical module path reported by that state's targets().
    pub fn read_adapter_gradient(
        &self,
        target: &str,
    ) -> Result<crate::deepseek_v41::DeepseekV41AdapterGradient> {
        if target.is_empty() {
            bail!("V4.1 projection gradient requires a target name");
        }
        let (Some(a), Some(b)) = (&self.grad_a, &self.grad_b) else {
            bail!("V4.1 frozen projection has no adapter gradients");
        };
        Ok(crate::deepseek_v41::DeepseekV41AdapterGradient {
            target: target.to_owned(),
            grad_a: a.read_f32(self.a_len)?,
            grad_b: b.read_f32(self.b_len)?,
        })
    }
}

pub struct DeepseekV41LinearAdjoints {
    device: VulkanDevice,
    forward: DeepseekV41MixedLinearKernel,
    input_grad: vulkan::ComputeKernel,
    input_grad_block8: vulkan::ComputeKernel,
    input_grad_group4: vulkan::ComputeKernel,
    input_grad_lane2: vulkan::ComputeKernel,
    weight_grad: vulkan::ComputeKernel,
    weight_grad_cpu_short: vulkan::ComputeKernel,
    vector: vulkan::ComputeKernel,
}
impl DeepseekV41LinearAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        let access = [
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::WriteOnly,
        ];
        let kernel = |bytes: &[u8], push| {
            vulkan::ComputeKernel::new_with_access(&device, bytes, &access, push)
        };
        Ok(Self {
            forward: DeepseekV41MixedLinearKernel::new(device.clone())?,
            input_grad: kernel(include_bytes!("../../shaders/linear_input_grad.spv"), 12)?,
            input_grad_block8: kernel(
                include_bytes!("../../shaders/linear_input_grad_block8.spv"),
                12,
            )?,
            input_grad_group4: kernel(
                include_bytes!("../../shaders/linear_input_grad_group4_muladd.spv"),
                12,
            )?,
            input_grad_lane2: kernel(
                include_bytes!("../../shaders/linear_input_grad_lane2.spv"),
                12,
            )?,
            weight_grad: kernel(include_bytes!("../../shaders/linear_weight_grad.spv"), 12)?,
            weight_grad_cpu_short: kernel(
                include_bytes!("../../shaders/linear_weight_grad_cpu_short.spv"),
                12,
            )?,
            vector: kernel(
                include_bytes!("../../shaders/transformer_deepseek_v41_projection_grad_vector.spv"),
                12,
            )?,
            device,
        })
    }

    fn check(&self, buffer: &GpuBuffer, len: usize) -> Result<()> {
        if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < len {
            bail!("V4.1 projection adjoint buffer geometry or device mismatch");
        }
        Ok(())
    }
    /// The FP32 matrix the input adjoint contracts against.
    ///
    /// FP32 and BF16 dense storage already hold the exact values their forward
    /// consumed; packed storage is decoded with its UE8M0 weight scales. Both
    /// cases are the same matrix the forward's arithmetic reduced against, so
    /// the input gradient needs no storage-specific staging.
    fn base_matrix<'a>(&self, weight: &'a DeepseekV41ResidentLinear) -> Result<Cow<'a, GpuBuffer>> {
        let elements = count(weight.rows, weight.cols)?;
        match &weight.storage {
            LinearStorage::Dense(base, _) => {
                self.check(base, elements)?;
                Ok(Cow::Borrowed(base))
            }
            LinearStorage::Packed(packed, _) => Ok(Cow::Owned(
                self.forward
                    .decode_packed(packed, crate::deepseek_v41::DeepseekV41WeightOutput::Fp32)?,
            )),
        }
    }
    fn snapshot(&self, input: &GpuBuffer, len: usize) -> Result<GpuBuffer> {
        self.check(input, len)?;
        let copy = GpuBuffer::zeros_f32(&self.device, len)?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.copy_f32_range(input, 0, &copy, 0, len)?;
        commands.submit()?;
        Ok(copy)
    }

    fn input_kernel(&self, rows: usize, input: usize, output: usize) -> &vulkan::ComputeKernel {
        let width = crate::transformer::aten_vector_width();
        if width >= 16 && rows == 1 {
            return &self.input_grad_block8;
        }
        // CPU non-transposed mm uses adjacent four-product groups for M=2/3
        // even at wide N. Narrow LoRA results use the same path at larger M.
        // The group kernel requires complete four-term reduction groups.
        if width >= 16
            && rows >= 2
            && output.is_multiple_of(4)
            && ((rows < 4 && input.is_multiple_of(4)) || (input <= 8 && input.is_multiple_of(2)))
        {
            &self.input_grad_group4
        } else {
            &self.input_grad
        }
    }

    fn weight_kernel(&self, input: usize, output: usize) -> &vulkan::ComputeKernel {
        if crate::transformer::aten_vector_width() >= 16
            && (output == 1
                || (output >= 2
                    && ((input <= 8 && input.is_multiple_of(2))
                        || (output < 4 && input.is_multiple_of(4)))))
        {
            &self.weight_grad_cpu_short
        } else {
            &self.weight_grad
        }
    }

    /// Evaluate the actual zero-dropout FP32 projection, retaining only its
    /// activation and factor snapshots. No frozen weight is owned by the tape.
    pub fn forward(
        &self,
        input: &GpuBuffer,
        rows: usize,
        weight: &DeepseekV41ResidentLinear,
    ) -> Result<DeepseekV41LinearTape> {
        self.forward_mixed(
            input,
            rows,
            weight,
            crate::deepseek_v41::DeepseekV41WeightOutput::Fp32,
        )
    }

    /// Evaluate the projection at the released mixed boundary. `stream` is the
    /// enclosing graph's stream precision: it fixes the input cast (and, for
    /// dense BF16 storage, the required storage precision). Packed storage
    /// quantizes that cast activation and emits BF16 exactly as the released
    /// block-scaled GEMM does; adapters merge at the existing output boundary.
    pub fn forward_mixed(
        &self,
        input: &GpuBuffer,
        rows: usize,
        weight: &DeepseekV41ResidentLinear,
        stream: crate::deepseek_v41::DeepseekV41WeightOutput,
    ) -> Result<DeepseekV41LinearTape> {
        geometry(rows, weight.cols, weight.rows)?;
        let storage = match &weight.storage {
            LinearStorage::Dense(base, bf16) => {
                self.check(base, count(weight.rows, weight.cols)?)?;
                LinearStorage::Dense(base.clone(), *bf16)
            }
            // Forward consumes packed bytes directly. Materializing the full
            // decoded matrix here wastes staging memory and is only needed by
            // the frozen input adjoint during backward.
            LinearStorage::Packed(packed, block) => LinearStorage::Packed(packed.clone(), *block),
        };
        let input = self.snapshot(input, count(rows, weight.cols)?)?;
        let lora = match &weight.lora {
            None => None,
            Some(lora) => {
                geometry(rows, weight.cols, lora.rank)?;
                geometry(rows, lora.rank, weight.rows)?;
                if !lora.scale.is_finite() || lora.scale <= 0.0 {
                    bail!("V4.1 projection adapter scale is invalid");
                }
                Some(DeepseekV41ResidentLora {
                    a: self.snapshot(&lora.a, count(lora.rank, weight.cols)?)?,
                    b: self.snapshot(&lora.b, count(weight.rows, lora.rank)?)?,
                    rank: lora.rank,
                    scale: lora.scale,
                })
            }
        };
        // This transient view drops its storage reference before returning.
        let view = DeepseekV41ResidentLinear {
            rows: weight.rows,
            cols: weight.cols,
            storage,
            lora: lora.as_ref().map(|l| DeepseekV41ResidentLora {
                a: l.a.clone(),
                b: l.b.clone(),
                rank: l.rank,
                scale: l.scale,
            }),
        };
        let output = self.forward.apply(&input, rows, &view, stream)?;
        Ok(DeepseekV41LinearTape {
            output,
            input,
            lora,
            rows,
            input_dim: weight.cols,
            output_dim: weight.rows,
        })
    }

    /// Differentiate a retained evaluation. The caller supplies the same frozen
    /// base values, potentially restaged from disk; adapter values come from
    /// the tape even if a newer adapter has since been uploaded.
    pub fn backward(
        &self,
        tape: &DeepseekV41LinearTape,
        weight: &DeepseekV41ResidentLinear,
        upstream: &GpuBuffer,
    ) -> Result<DeepseekV41LinearGradients> {
        if weight.shape() != [tape.output_dim, tape.input_dim] {
            bail!("V4.1 projection backward base shape differs from its tape");
        }
        // Packed storage is decoded for this call only; the tape pins no base.
        let base = self.base_matrix(weight)?;
        let base = base.as_ref();
        let rows = tape.rows;
        self.check(&tape.input, count(rows, tape.input_dim)?)?;
        self.check(upstream, count(rows, tape.output_dim)?)?;
        let base_grad = GpuBuffer::zeros_f32(&self.device, count(rows, tape.input_dim)?)?;
        let base_push = LinearPush {
            rows: rows as u32,
            input_dim: tape.input_dim as u32,
            output_dim: tape.output_dim as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.input_kernel(rows, tape.input_dim, tape.output_dim)
            .record_dispatch(
                &mut commands,
                &[upstream, base, &base_grad],
                bytemuck::bytes_of(&base_push),
                input_grid(rows, tape.input_dim),
            )?;
        let Some(lora) = &tape.lora else {
            commands.submit()?;
            return Ok(DeepseekV41LinearGradients {
                grad_input: base_grad,
                grad_a: None,
                grad_b: None,
                a_len: 0,
                b_len: 0,
            });
        };
        // Retire before the recomputation's synchronous submission. Keeping
        // the frozen base alive is limited to this backward call.
        commands.submit()?;
        let hidden = self.forward.dense.apply(
            &self.device,
            &tape.input,
            &lora.a,
            rows,
            tape.input_dim,
            lora.rank,
        )?;
        let scaled = GpuBuffer::zeros_f32(&self.device, count(rows, tape.output_dim)?)?;
        let grad_hidden = GpuBuffer::zeros_f32(&self.device, count(rows, lora.rank)?)?;
        let grad_lora_input = GpuBuffer::zeros_f32(&self.device, count(rows, tape.input_dim)?)?;
        let grad_input = GpuBuffer::zeros_f32(&self.device, count(rows, tape.input_dim)?)?;
        let a_len = count(lora.rank, tape.input_dim)?;
        let b_len = count(tape.output_dim, lora.rank)?;
        let grad_a = GpuBuffer::zeros_f32(&self.device, a_len)?;
        let grad_b = GpuBuffer::zeros_f32(&self.device, b_len)?;
        let a_push = LinearPush {
            rows: rows as u32,
            input_dim: tape.input_dim as u32,
            output_dim: lora.rank as u32,
        };
        let b_push = LinearPush {
            rows: rows as u32,
            input_dim: lora.rank as u32,
            output_dim: tape.output_dim as u32,
        };
        let scale = VectorPush {
            len: count(rows, tape.output_dim)? as u32,
            scale: lora.scale,
            add: 0,
        };
        let add = VectorPush {
            len: count(rows, tape.input_dim)? as u32,
            scale: 1.0,
            add: 1,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.vector.record_dispatch(
            &mut commands,
            &[upstream, upstream, &scaled],
            bytemuck::bytes_of(&scale),
            vector_grid(scale.len as usize),
        )?;
        self.weight_kernel(lora.rank, tape.output_dim)
            .record_dispatch(
                &mut commands,
                &[&hidden, &scaled, &grad_b],
                bytemuck::bytes_of(&b_push),
                weight_grid(lora.rank, tape.output_dim),
            )?;
        // Preserve the existing AVX2 LoRA-B policy. The new wide, short-row
        // policy above is qualified only on the pinned AVX512 CPU oracle.
        let b_kernel = if crate::transformer::aten_vector_width() < 16 && lora.rank <= 8 {
            &self.input_grad_lane2
        } else {
            self.input_kernel(rows, lora.rank, tape.output_dim)
        };
        b_kernel.record_dispatch(
            &mut commands,
            &[&scaled, &lora.b, &grad_hidden],
            bytemuck::bytes_of(&b_push),
            input_grid(rows, lora.rank),
        )?;
        self.weight_kernel(tape.input_dim, lora.rank)
            .record_dispatch(
                &mut commands,
                &[&tape.input, &grad_hidden, &grad_a],
                bytemuck::bytes_of(&a_push),
                weight_grid(tape.input_dim, lora.rank),
            )?;
        self.input_kernel(rows, tape.input_dim, lora.rank)
            .record_dispatch(
                &mut commands,
                &[&grad_hidden, &lora.a, &grad_lora_input],
                bytemuck::bytes_of(&a_push),
                input_grid(rows, tape.input_dim),
            )?;
        self.vector.record_dispatch(
            &mut commands,
            &[&base_grad, &grad_lora_input, &grad_input],
            bytemuck::bytes_of(&add),
            vector_grid(add.len as usize),
        )?;
        commands.submit()?;
        Ok(DeepseekV41LinearGradients {
            grad_input,
            grad_a: Some(grad_a),
            grad_b: Some(grad_b),
            a_len,
            b_len,
        })
    }
}

fn count(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| a != 0 && b != 0 && n <= u32::MAX as usize)
        .context("V4.1 projection adjoint has zero/overflowing geometry")
}
fn geometry(rows: usize, input: usize, output: usize) -> Result<()> {
    count(rows, input)?;
    count(rows, output)?;
    count(input, output)?;
    if rows > 65_535 || input.div_ceil(16) > 65_535 || output.div_ceil(16) > 65_535 {
        bail!("V4.1 projection adjoint exceeds dispatch capacity");
    }
    Ok(())
}
fn input_grid(rows: usize, input: usize) -> [u32; 3] {
    [input.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1]
}
fn weight_grid(input: usize, output: usize) -> [u32; 3] {
    [input.div_ceil(16) as u32, output.div_ceil(16) as u32, 1]
}
fn vector_grid(len: usize) -> [u32; 3] {
    let groups = len.div_ceil(64);
    [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1]
}
