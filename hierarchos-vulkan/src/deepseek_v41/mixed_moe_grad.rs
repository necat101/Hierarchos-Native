//! Retained mixed MoE graph: packed projections, SwiGLU, routing and compaction.
//!
//! Forward calls the same gate/pack/SwiGLU/scatter/combine kernels as inference
//! and the projection tape's unchanged mixed forward. Backward uses the
//! projection module's explicit FP32 straight-through rounding contract. No
//! frozen expert payload is retained by a tape; streamed sources restage one
//! expert triplet at a time. Dropout and a complete model trainer are separate.

use super::*;
use crate::deepseek_v41::{
    DeepseekV41AdapterGradient, DeepseekV41LinearAdjoints, DeepseekV41LinearGradients,
    DeepseekV41LinearTape,
};

#[cfg(test)]
#[path = "mixed_moe_grad_tests.rs"]
mod tests;

struct ExpertTape {
    rows: usize,
    w1: DeepseekV41LinearTape,
    w2: DeepseekV41LinearTape,
    w3: DeepseekV41LinearTape,
    route: GpuBuffer,
    mapping: Option<GpuBuffer>,
}

/// Evaluation snapshots. Selected indices are held fixed by the adjoint;
/// routing scores remain differentiable through the frozen gate projection.
pub struct DeepseekV41MixedMoeTape {
    pub output: GpuBuffer,
    input: GpuBuffer,
    route: GpuBuffer,
    indices: GpuBuffer,
    counts: GpuBuffer,
    experts: Vec<Option<ExpertTape>>,
    shared: ExpertTape,
    rows: usize,
    config: DeepseekV41MoeConfig,
    precision: DeepseekV41WeightOutput,
}

pub struct DeepseekV41MixedMoeGradients {
    pub grad_input: GpuBuffer,
    /// Gradients in the original token/top-k slot order, before gate backward.
    pub grad_route_weights: GpuBuffer,
    adapters: Vec<(u32, DeepseekV41AdapterGradient)>,
    counts: GpuBuffer,
    experts: usize,
}
impl DeepseekV41MixedMoeGradients {
    /// Materialize canonical host adapter gradients, e.g. prefix
    /// `layers.0.ffn`. Unvisited expert parameters are absent, matching
    /// autograd's None gradients; they must not receive a zero-gradient AdamW
    /// update. Only this optimizer boundary reads the small GPU count vector.
    pub fn adapter_gradients(&self, prefix: &str) -> Result<Vec<DeepseekV41AdapterGradient>> {
        if prefix.is_empty() || prefix.ends_with('.') || prefix.chars().any(char::is_whitespace) {
            bail!("V4.1 mixed MoE gradient prefix is invalid");
        }
        if self.adapters.is_empty() {
            return Ok(Vec::new());
        }
        let counts = self.counts.read_u32(self.experts)?;
        let mut gradients = Vec::new();
        for (id, gradient) in &self.adapters {
            if *id != u32::MAX && counts[*id as usize] == 0 {
                continue;
            }
            let mut gradient = gradient.clone();
            gradient.target = format!("{prefix}.{}", gradient.target);
            gradients.push(gradient);
        }
        gradients.sort_by(|a, b| a.target.cmp(&b.target));
        Ok(gradients)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddPush {
    len: u32,
    scale: f32,
    add: u32,
}

pub struct DeepseekV41MixedMoeAdjoints {
    device: VulkanDevice,
    linear: DeepseekV41LinearAdjoints,
    swiglu: vulkan::ComputeKernel,
    scatter: vulkan::ComputeKernel,
    gate: vulkan::ComputeKernel,
    vector: vulkan::ComputeKernel,
}
impl DeepseekV41MixedMoeAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        use vulkan::BindingAccess::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
        Ok(Self {
            linear: DeepseekV41LinearAdjoints::new(device.clone())?,
            swiglu: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu_grad.spv"),
                &[R, R, R, R, R, W, W, W],
                std::mem::size_of::<MixedSwigluPush>() as u32,
            )?,
            scatter: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_mixed_moe_scatter_grad.spv"),
                &[R, R, R, R, R, RW, W],
                std::mem::size_of::<MoePackPush>() as u32,
            )?,
            gate: vulkan::ComputeKernel::new_with_access(
                &device,
                MOE_GATE_GRAD,
                &[R, R, R, R, W, R],
                std::mem::size_of::<MoeGatePush>() as u32,
            )?,
            vector: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_projection_grad_vector.spv"),
                &[R, R, W],
                std::mem::size_of::<AddPush>() as u32,
            )?,
            device,
        })
    }

    fn check(&self, buffer: &GpuBuffer, elements: usize) -> Result<()> {
        if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < elements {
            bail!("V4.1 mixed MoE adjoint buffer geometry or device mismatch");
        }
        Ok(())
    }
    fn geometry(&self, moe: &DeepseekV41MixedMoe, rows: usize) -> Result<(usize, usize)> {
        let c = &moe.config;
        let stream = checked_size("mixed MoE adjoint stream", &[rows, c.hidden_size])?;
        let slots = checked_size("mixed MoE adjoint slots", &[rows, c.activated])?;
        checked_size("mixed MoE adjoint routed", &[slots, c.hidden_size])?;
        checked_size("mixed MoE adjoint activation", &[rows, c.intermediate_size])?;
        if rows == 0
            || slots > 65_535
            || c.hidden_size.div_ceil(64) > 65_535
            || c.intermediate_size.div_ceil(64) > 65_535
        {
            bail!("V4.1 mixed MoE adjoint exceeds dispatch capacity");
        }
        self.check(&moe.gate_weight, c.experts * c.hidden_size)?;
        self.check(&moe.gate_bias, c.experts)?;
        Ok((stream, slots))
    }
    fn snapshot(&self, input: &GpuBuffer, elements: usize) -> Result<GpuBuffer> {
        self.check(input, elements)?;
        let copy = GpuBuffer::zeros_f32(&self.device, elements)?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.copy_f32_range(input, 0, &copy, 0, elements)?;
        commands.submit()?;
        Ok(copy)
    }

    // Resident and checkpoint-staged paths share the identical callback.
    // The owned staging triplet drops before the next expert is loaded.
    fn with_expert<T>(
        &self,
        moe: &DeepseekV41MixedMoe,
        id: u32,
        run: impl FnOnce(&MixedExpert) -> Result<T>,
    ) -> Result<T> {
        if id == u32::MAX {
            if let Some(shared) = &moe.shared {
                return run(shared);
            }
        } else if let ExpertSource::Resident(experts) = &moe.experts {
            return run(&experts[id as usize]);
        }
        let ExpertSource::Streamed { checkpoint, layer } = &moe.experts else {
            bail!("V4.1 mixed MoE adjoint has no shared expert source");
        };
        let payload = {
            let mut checkpoint = checkpoint
                .lock()
                .map_err(|_| anyhow::anyhow!("V4.1 streamed checkpoint lock poisoned"))?;
            if id == u32::MAX {
                checkpoint.load_mixed_shared_expert(*layer)?
            } else {
                checkpoint.load_mixed_expert(*layer, id as usize)?
            }
        };
        run(&MixedExpert::upload(&self.device, payload)?)
    }

    fn expert_forward(
        &self,
        moe: &DeepseekV41MixedMoe,
        x: &GpuBuffer,
        rows: usize,
        id: u32,
        route: GpuBuffer,
        counts: &GpuBuffer,
        mapping: Option<GpuBuffer>,
        w: &MixedExpert,
    ) -> Result<ExpertTape> {
        let w1 = self.linear.forward_mixed(x, rows, &w.w1, moe.precision)?;
        let w3 = self.linear.forward_mixed(x, rows, &w.w3, moe.precision)?;
        let activation = GpuBuffer::zeros_f32(&self.device, rows * moe.config.intermediate_size)?;
        let push = MixedSwigluPush {
            rows: rows as u32,
            inter: moe.config.intermediate_size as u32,
            expert: id,
            bf16: u32::from(moe.precision == DeepseekV41WeightOutput::Bf16),
            limit: moe.config.swiglu_limit,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        moe.swiglu.record_dispatch(
            &mut commands,
            &[&w1.output, &w3.output, &route, counts, &activation],
            bytemuck::bytes_of(&push),
            [
                moe.config.intermediate_size.div_ceil(64) as u32,
                rows as u32,
                1,
            ],
        )?;
        commands.submit()?;
        let w2 = self
            .linear
            .forward_mixed(&activation, rows, &w.w2, moe.precision)?;
        #[cfg(test)]
        {
            crate::deepseek_v41::training_trace::record(
                &format!("expert/{id}/input"),
                x,
                rows * moe.config.hidden_size,
            );
            crate::deepseek_v41::training_trace::record(
                &format!("expert/{id}/w1"),
                &w1.output,
                rows * moe.config.intermediate_size,
            );
            crate::deepseek_v41::training_trace::record(
                &format!("expert/{id}/w3"),
                &w3.output,
                rows * moe.config.intermediate_size,
            );
            crate::deepseek_v41::training_trace::record(
                &format!("expert/{id}/activation"),
                &activation,
                rows * moe.config.intermediate_size,
            );
            crate::deepseek_v41::training_trace::record(
                &format!("expert/{id}/w2"),
                &w2.output,
                rows * moe.config.hidden_size,
            );
        }
        Ok(ExpertTape {
            rows,
            w1,
            w2,
            w3,
            route,
            mapping,
        })
    }

    /// Retain only visited routed experts. A small integer count vector is
    /// read after routing; inactive experts allocate no activation/factor tape
    /// and are never staged from the package. Frozen base weights are not kept.
    pub fn forward(
        &self,
        moe: &DeepseekV41MixedMoe,
        input: &GpuBuffer,
        rows: usize,
    ) -> Result<DeepseekV41MixedMoeTape> {
        let (stream, slots) = self.geometry(moe, rows)?;
        let c = moe.config;
        let x = self.snapshot(
            &moe.linear.cast_buffer(input, stream, moe.precision)?,
            stream,
        )?;
        let route = GpuBuffer::zeros_f32(&self.device, slots)?;
        let indices = GpuBuffer::zeros_u32(&self.device, slots)?;
        let counts = GpuBuffer::zeros_u32(&self.device, c.experts)?;
        let push = gate_push(c, rows);
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        moe.gate.record_dispatch(
            &mut commands,
            &[
                &x,
                &moe.gate_weight,
                &moe.gate_bias,
                &route,
                &indices,
                &counts,
                &moe.cpu_sqrt_seeds,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        let routed = GpuBuffer::zeros_f32(&self.device, slots * c.hidden_size)?;
        let mut experts = Vec::with_capacity(c.experts);
        let visited = counts.read_u32(c.experts)?;
        for id in 0..c.experts {
            if visited[id] == 0 {
                experts.push(None);
                continue;
            }
            let compact = GpuBuffer::zeros_f32(&self.device, stream)?;
            let weights = GpuBuffer::zeros_f32(&self.device, rows)?;
            let mapping = GpuBuffer::zeros_u32(&self.device, rows)?;
            let push = MoePackPush {
                rows: rows as u32,
                dim: c.hidden_size as u32,
                topk: c.activated as u32,
                expert: id as u32,
            };
            let grid = [c.hidden_size.div_ceil(64) as u32, rows as u32, 1];
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            moe.pack.record_dispatch(
                &mut commands,
                &[&x, &indices, &route, &compact, &weights, &mapping],
                bytemuck::bytes_of(&push),
                grid,
            )?;
            commands.submit()?;
            let tape = self.with_expert(moe, id as u32, |w| {
                self.expert_forward(
                    moe,
                    &compact,
                    visited[id] as usize,
                    id as u32,
                    weights,
                    &counts,
                    Some(mapping.clone()),
                    w,
                )
            })?;
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            moe.scatter.record_dispatch(
                &mut commands,
                &[&tape.w2.output, &indices, &mapping, &routed],
                bytemuck::bytes_of(&push),
                grid,
            )?;
            commands.submit()?;
            experts.push(Some(tape));
        }
        let shared = self.with_expert(moe, u32::MAX, |w| {
            self.expert_forward(moe, &x, rows, u32::MAX, route.clone(), &counts, None, w)
        })?;
        let combined = GpuBuffer::zeros_f32(&self.device, stream)?;
        let push = MoeCombinePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            topk: c.activated as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        moe.combine.record_dispatch(
            &mut commands,
            &[&routed, &shared.w2.output, &combined, &indices],
            bytemuck::bytes_of(&push),
            [c.hidden_size.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        let output = moe.linear.cast_buffer(&combined, stream, moe.precision)?;
        Ok(DeepseekV41MixedMoeTape {
            output,
            input: x,
            route,
            indices,
            counts,
            experts,
            shared,
            rows,
            config: c,
            precision: moe.precision,
        })
    }

    fn add(&self, left: &GpuBuffer, right: &GpuBuffer, len: usize) -> Result<GpuBuffer> {
        let output = GpuBuffer::zeros_f32(&self.device, len)?;
        let push = AddPush {
            len: len as u32,
            scale: 1.0,
            add: 1,
        };
        let groups = len.div_ceil(64);
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.vector.record_dispatch(
            &mut commands,
            &[left, right, &output],
            bytemuck::bytes_of(&push),
            [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }

    fn expert_backward(
        &self,
        moe: &DeepseekV41MixedMoe,
        tape: &DeepseekV41MixedMoeTape,
        expert: &ExpertTape,
        id: u32,
        w: &MixedExpert,
        upstream: &GpuBuffer,
    ) -> Result<(
        DeepseekV41LinearGradients,
        DeepseekV41LinearGradients,
        DeepseekV41LinearGradients,
        GpuBuffer,
    )> {
        let rows = expert.rows;
        let inter = moe.config.intermediate_size;
        let w2 = self.linear.backward(&expert.w2, &w.w2, upstream)?;
        let dg = GpuBuffer::zeros_f32(&self.device, rows * inter)?;
        let du = GpuBuffer::zeros_f32(&self.device, rows * inter)?;
        let dr = GpuBuffer::zeros_f32(&self.device, rows)?;
        let push = MixedSwigluPush {
            rows: rows as u32,
            inter: inter as u32,
            expert: id,
            bf16: u32::from(moe.precision == DeepseekV41WeightOutput::Bf16),
            limit: moe.config.swiglu_limit,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.swiglu.record_dispatch(
            &mut commands,
            &[
                &expert.w1.output,
                &expert.w3.output,
                &expert.route,
                &tape.counts,
                &w2.grad_input,
                &dg,
                &du,
                &dr,
            ],
            bytemuck::bytes_of(&push),
            [rows.div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        let w1 = self.linear.backward(&expert.w1, &w.w1, &dg)?;
        let w3 = self.linear.backward(&expert.w3, &w.w3, &du)?;
        Ok((w1, w2, w3, dr))
    }

    /// Differentiate a retained evaluation with the same frozen weights,
    /// optionally restaged from a checkpoint. Only small adapter gradients
    /// are read back; routing stays on GPU until adapter_gradients is called.
    pub fn backward(
        &self,
        moe: &DeepseekV41MixedMoe,
        tape: &DeepseekV41MixedMoeTape,
        upstream: &GpuBuffer,
    ) -> Result<DeepseekV41MixedMoeGradients> {
        let (stream, slots) = self.geometry(moe, tape.rows)?;
        if moe.config != tape.config
            || moe.precision != tape.precision
            || tape.experts.len() != moe.config.experts
        {
            bail!("V4.1 mixed MoE backward configuration differs from its tape");
        }
        self.check(upstream, stream)?;
        self.check(&tape.input, stream)?;
        let c = moe.config;
        let rows = tape.rows;
        let mut adapters = Vec::new();
        let (w1, w2, w3, _) = self.with_expert(moe, u32::MAX, |w| {
            self.expert_backward(moe, tape, &tape.shared, u32::MAX, w, upstream)
        })?;
        let grad_tokens = self.add(&w1.grad_input, &w3.grad_input, stream)?;
        collect_adapters(&mut adapters, u32::MAX, &w1, &w2, &w3)?;
        drop((w1, w2, w3));
        let grad_route_weights = GpuBuffer::zeros_f32(&self.device, slots)?;
        // Shared was created last by the CPU graph; autograd visits routed
        // experts in reverse creation order when accumulating into x.
        for id in (0..c.experts).rev() {
            if tape.experts[id].is_none() {
                continue;
            }
            let compact_grad = GpuBuffer::zeros_f32(&self.device, stream)?;
            let scratch_weights = GpuBuffer::zeros_f32(&self.device, rows)?;
            let mapping = GpuBuffer::zeros_u32(&self.device, rows)?;
            let push = MoePackPush {
                rows: rows as u32,
                dim: c.hidden_size as u32,
                topk: c.activated as u32,
                expert: id as u32,
            };
            let grid = [c.hidden_size.div_ceil(64) as u32, rows as u32, 1];
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            // Combine/scatter's adjoint gathers the upstream into the same
            // torch.where compact-row order as the forward gather.
            moe.pack.record_dispatch(
                &mut commands,
                &[
                    upstream,
                    &tape.indices,
                    &tape.route,
                    &compact_grad,
                    &scratch_weights,
                    &mapping,
                ],
                bytemuck::bytes_of(&push),
                grid,
            )?;
            commands.submit()?;
            let expert = tape.experts[id].as_ref().unwrap();
            let (w1, w2, w3, dr) = self.with_expert(moe, id as u32, |w| {
                self.expert_backward(moe, tape, expert, id as u32, w, &compact_grad)
            })?;
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            self.scatter.record_dispatch(
                &mut commands,
                &[
                    &w1.grad_input,
                    &w3.grad_input,
                    &tape.indices,
                    expert.mapping.as_ref().unwrap(),
                    &dr,
                    &grad_tokens,
                    &grad_route_weights,
                ],
                bytemuck::bytes_of(&push),
                grid,
            )?;
            commands.submit()?;
            collect_adapters(&mut adapters, id as u32, &w1, &w2, &w3)?;
        }
        let grad_gate = GpuBuffer::zeros_f32(&self.device, stream)?;
        let push = gate_push(c, rows);
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.gate.record_dispatch(
            &mut commands,
            &[
                &tape.input,
                &moe.gate_weight,
                &tape.indices,
                &grad_route_weights,
                &grad_gate,
                &moe.cpu_sqrt_seeds,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(DeepseekV41MixedMoeGradients {
            grad_input: self.add(&grad_tokens, &grad_gate, stream)?,
            grad_route_weights,
            adapters,
            counts: tape.counts.clone(),
            experts: c.experts,
        })
    }
}

fn gate_push(c: DeepseekV41MoeConfig, rows: usize) -> MoeGatePush {
    MoeGatePush {
        rows: rows as u32,
        dim: c.hidden_size as u32,
        experts: c.experts as u32,
        topk: c.activated as u32,
        score_func: c.score_func.code(),
        gate_temp: c.gate_temp,
        norm_topk: c.norm_topk_prob as u32,
        route_scale: c.route_scale,
    }
}

fn collect_adapters(
    out: &mut Vec<(u32, DeepseekV41AdapterGradient)>,
    id: u32,
    w1: &DeepseekV41LinearGradients,
    w2: &DeepseekV41LinearGradients,
    w3: &DeepseekV41LinearGradients,
) -> Result<()> {
    let prefix = if id == u32::MAX {
        "shared_experts".to_owned()
    } else {
        format!("experts.{id}")
    };
    for (name, gradient) in [("w1", w1), ("w2", w2), ("w3", w3)] {
        if gradient.grad_a.is_some() {
            out.push((
                id,
                gradient.read_adapter_gradient(&format!("{prefix}.{name}"))?,
            ));
        }
    }
    Ok(())
}
