//! Engram's retained normalized-gate and frozen/adapter projection adjoints.
//!
//! Token hashes and embedding rows are frozen inputs. Backward uses retained
//! lookup/stream/factor snapshots and a restaged projection, so no embedding
//! table or checkpoint payload is needed after forward. Mixed rounding uses
//! the projection module's documented FP32 straight-through contract.

use super::*;
use crate::deepseek_v41::{
    DeepseekV41LinearAdjoints, DeepseekV41LinearGradients, DeepseekV41LinearTape,
};

#[cfg(test)]
#[path = "engram_grad_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV41EngramGateConfig {
    pub rows: usize,
    pub hc: usize,
    pub hidden: usize,
    pub eps: f32,
    pub precision: DeepseekV41WeightOutput,
}
impl DeepseekV41EngramGateConfig {
    fn lengths(self) -> Result<(usize, usize, usize)> {
        let copies = count(self.rows, self.hc)?;
        let stream = count(copies, self.hidden)?;
        let projection = count(
            self.hc.checked_add(1).context("V4.1 Engram HC overflow")?,
            self.hidden,
        )?;
        let kv = count(self.rows, projection)?;
        if copies > 65_535
            || self.rows > 65_535
            || self.hidden.div_ceil(64) > 65_535
            || !self.eps.is_finite()
            || self.eps <= 0.0
        {
            bail!("V4.1 Engram adjoint configuration exceeds its dispatch/epsilon bounds");
        }
        Ok((stream, kv, projection))
    }
    fn push(self, mask: bool) -> Result<GatePush> {
        Ok(GatePush {
            rows: self.rows as u32,
            hc: self.hc as u32,
            dim: self.hidden as u32,
            proj_width: self.lengths()?.2 as u32,
            eps: self.eps,
            scale: (self.hidden as f64).powf(-0.5) as f32,
            has_mask: mask as u32,
        })
    }
}

pub struct DeepseekV41EngramGateTape {
    pub output: GpuBuffer,
    stream: GpuBuffer,
    kv: GpuBuffer,
    weight: GpuBuffer,
    mask: Option<GpuBuffer>,
    config: DeepseekV41EngramGateConfig,
}
pub struct DeepseekV41EngramGateGradients {
    pub grad_stream: GpuBuffer,
    /// Per-copy keys followed by the shared value, matching wkv's output.
    pub grad_kv: GpuBuffer,
}
pub struct DeepseekV41EngramTape {
    pub output: GpuBuffer,
    gate: DeepseekV41EngramGateTape,
    projection: DeepseekV41LinearTape,
}
pub struct DeepseekV41EngramGradients {
    pub grad_stream: GpuBuffer,
    pub grad_kv: GpuBuffer,
    /// Input gradients for the selected embedded rows and optional adapter
    /// factor VJPs. This is not a trainable embedding-table gradient.
    pub projection: DeepseekV41LinearGradients,
}

pub struct DeepseekV41EngramAdjoints {
    device: VulkanDevice,
    linear: DeepseekV41LinearAdjoints,
    mixed: DeepseekV41MixedLinearKernel,
    forward_gate: vulkan::ComputeKernel,
    gate: vulkan::ComputeKernel,
    value: vulkan::ComputeKernel,
}
impl DeepseekV41EngramAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        use vulkan::BindingAccess::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
        Ok(Self {
            linear: DeepseekV41LinearAdjoints::new(device.clone())?,
            mixed: DeepseekV41MixedLinearKernel::new(device.clone())?,
            forward_gate: vulkan::ComputeKernel::new_with_access(
                &device,
                GATE,
                &[R, R, R, R, W],
                std::mem::size_of::<GatePush>() as u32,
            )?,
            gate: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate_grad.spv"),
                &[R, R, R, R, R, W, W, W],
                std::mem::size_of::<GatePush>() as u32,
            )?,
            value: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_engram_value_grad.spv"),
                &[R, R, RW],
                std::mem::size_of::<GatePush>() as u32,
            )?,
            device,
        })
    }
    fn check(&self, buffer: &GpuBuffer, elements: usize) -> Result<()> {
        if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < elements {
            bail!("V4.1 Engram adjoint buffer geometry or device mismatch");
        }
        Ok(())
    }
    fn snapshot(&self, buffer: &GpuBuffer, elements: usize) -> Result<GpuBuffer> {
        self.check(buffer, elements)?;
        let copied = GpuBuffer::zeros_f32(&self.device, elements)?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.copy_f32_range(buffer, 0, &copied, 0, elements)?;
        commands.submit()?;
        Ok(copied)
    }

    /// Evaluate the unchanged Engram gate with already projected keys/values.
    /// The stream is cast at its explicit precision boundary. KV and norm
    /// products are already the FP32 views consumed by the released gate.
    pub fn gate_forward(
        &self,
        stream: &GpuBuffer,
        kv: &GpuBuffer,
        weight: &GpuBuffer,
        mask: Option<&GpuBuffer>,
        config: DeepseekV41EngramGateConfig,
    ) -> Result<DeepseekV41EngramGateTape> {
        let (stream_len, kv_len, _) = config.lengths()?;
        let stream = self.snapshot(
            &self
                .mixed
                .cast_buffer(stream, stream_len, config.precision)?,
            stream_len,
        )?;
        let kv = self.snapshot(kv, kv_len)?;
        let weight = self.snapshot(weight, count(config.hc, config.hidden)?)?;
        let mask = mask.map(|m| self.snapshot(m, config.rows)).transpose()?;
        let empty = GpuBuffer::zeros_u32(&self.device, 1)?;
        let output = GpuBuffer::zeros_f32(&self.device, stream_len)?;
        let push = config.push(mask.is_some())?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.forward_gate.record_dispatch(
            &mut commands,
            &[
                &stream,
                &kv,
                &weight,
                mask.as_ref().unwrap_or(&empty),
                &output,
            ],
            bytemuck::bytes_of(&push),
            [(config.rows * config.hc) as u32, 1, 1],
        )?;
        commands.submit()?;
        let output = self
            .mixed
            .cast_buffer(&output, stream_len, config.precision)?;
        Ok(DeepseekV41EngramGateTape {
            output,
            stream,
            kv,
            weight,
            mask,
            config,
        })
    }

    pub fn gate_backward(
        &self,
        tape: &DeepseekV41EngramGateTape,
        upstream: &GpuBuffer,
    ) -> Result<DeepseekV41EngramGateGradients> {
        let c = tape.config;
        let (stream_len, kv_len, _) = c.lengths()?;
        self.check(upstream, stream_len)?;
        self.check(&tape.stream, stream_len)?;
        self.check(&tape.kv, kv_len)?;
        self.check(&tape.weight, count(c.hc, c.hidden)?)?;
        if let Some(mask) = &tape.mask {
            self.check(mask, c.rows)?;
        }
        let grad_stream = GpuBuffer::zeros_f32(&self.device, stream_len)?;
        let grad_kv = GpuBuffer::zeros_f32(&self.device, kv_len)?;
        let gates = GpuBuffer::zeros_f32(&self.device, c.rows * c.hc)?;
        let empty = GpuBuffer::zeros_u32(&self.device, 1)?;
        let push = c.push(tape.mask.is_some())?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.gate.record_dispatch(
            &mut commands,
            &[
                upstream,
                &tape.stream,
                &tape.kv,
                &tape.weight,
                tape.mask.as_ref().unwrap_or(&empty),
                &grad_stream,
                &grad_kv,
                &gates,
            ],
            bytemuck::bytes_of(&push),
            [(c.rows * c.hc) as u32, 1, 1],
        )?;
        self.value.record_dispatch(
            &mut commands,
            &[upstream, &gates, &grad_kv],
            bytemuck::bytes_of(&push),
            [c.hidden.div_ceil(64) as u32, c.rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(DeepseekV41EngramGateGradients {
            grad_stream,
            grad_kv,
        })
    }

    /// Retain one actual lookup/projection/gate evaluation. No table or frozen
    /// projection payload survives in the tape, including streamed tables.
    pub fn forward(
        &self,
        engram: &DeepseekV41Engram,
        stream: &GpuBuffer,
        hashes: &GpuBuffer,
        mask: Option<&GpuBuffer>,
        rows: usize,
    ) -> Result<DeepseekV41EngramTape> {
        let precision = match &engram.wkv {
            EngramProjection::Mixed {
                stream_precision, ..
            } => *stream_precision,
            _ => DeepseekV41WeightOutput::Fp32,
        };
        let config = DeepseekV41EngramGateConfig {
            rows,
            hc: engram.hc,
            hidden: engram.hidden,
            eps: engram.eps,
            precision,
        };
        let (length, _, width) = config.lengths()?;
        self.check(stream, length)?;
        self.check(&engram.weight, count(engram.hc, engram.hidden)?)?;
        let vectors = count(rows, engram.cols)?;
        if vectors > 65_535 {
            bail!("V4.1 Engram retained lookup exceeds dispatch capacity");
        }
        self.check(hashes, vectors)?;
        if let Some(mask) = mask {
            self.check(mask, rows)?;
        }
        let embedded = engram.lookup_rows(hashes, vectors)?;
        let projection = match &engram.wkv {
            EngramProjection::Fp32(w) => {
                let view = DeepseekV41ResidentLinear::fp32_view(
                    w,
                    width,
                    count(engram.cols, engram.head_dim)?,
                )?;
                self.linear.forward(&embedded, rows, &view)?
            }
            EngramProjection::Mixed {
                weight,
                input_precision,
                ..
            } => self
                .linear
                .forward_mixed(&embedded, rows, weight, *input_precision)?,
        };
        let gate = self.gate_forward(stream, &projection.output, &engram.weight, mask, config)?;
        Ok(DeepseekV41EngramTape {
            output: gate.output.clone(),
            gate,
            projection,
        })
    }

    /// Restage only the same frozen projection. Hashes, masks, norm products,
    /// selected rows and adapter factors are snapshots of the original call.
    /// No table or hash-state lookup is performed during backward.
    pub fn backward(
        &self,
        tape: &DeepseekV41EngramTape,
        projection: &DeepseekV41ResidentLinear,
        upstream: &GpuBuffer,
    ) -> Result<DeepseekV41EngramGradients> {
        let gradient = self.gate_backward(&tape.gate, upstream)?;
        let projection = self
            .linear
            .backward(&tape.projection, projection, &gradient.grad_kv)?;
        Ok(DeepseekV41EngramGradients {
            grad_stream: gradient.grad_stream,
            grad_kv: gradient.grad_kv,
            projection,
        })
    }
}

impl DeepseekV41EngramRuntime {
    pub(crate) fn forward_retained(
        &self,
        layer_id: usize,
        stream: &GpuBuffer,
        rows: usize,
        adjoints: &DeepseekV41EngramAdjoints,
    ) -> Result<Option<DeepseekV41EngramTape>> {
        let Some(index) = self.layout.layer_hash_index(layer_id) else {
            return Ok(None);
        };
        let hashes = self.hash.hashes(index)?;
        let layer = self.layers[layer_id]
            .as_ref()
            .context("V4.1 retained Engram parameters missing")?;
        adjoints
            .forward(layer, stream, &hashes, self.token_mask.as_ref(), rows)
            .map(Some)
    }
    pub(crate) fn backward_retained(
        &self,
        layer_id: usize,
        tape: &DeepseekV41EngramTape,
        upstream: &GpuBuffer,
        adjoints: &DeepseekV41EngramAdjoints,
    ) -> Result<(
        GpuBuffer,
        Vec<crate::deepseek_v41::DeepseekV41AdapterGradient>,
    )> {
        let layer = self.layers[layer_id]
            .as_ref()
            .context("V4.1 retained Engram parameters missing")?;
        let gradient = match &layer.wkv {
            EngramProjection::Fp32(w) => {
                let width = layer.hidden * (layer.hc + 1);
                let view =
                    DeepseekV41ResidentLinear::fp32_view(w, width, layer.cols * layer.head_dim)?;
                adjoints.backward(tape, &view, upstream)?
            }
            EngramProjection::Mixed { weight, .. } => adjoints.backward(tape, weight, upstream)?,
        };
        let adapters = if gradient.projection.grad_a.is_some() {
            vec![gradient
                .projection
                .read_adapter_gradient(&format!("layers.{layer_id}.engram.wkv"))?]
        } else {
            Vec::new()
        };
        Ok((gradient.grad_stream, adapters))
    }
}

fn count(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| a != 0 && b != 0 && n <= u32::MAX as usize)
        .context("V4.1 Engram adjoint has zero/overflowing geometry")
}
