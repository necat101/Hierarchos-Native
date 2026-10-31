//! Projection/norm/rotary edges on either side of the sparse/cache tape.
//! Frozen bases are supplied again at backward, so they can be checkpoint
//! staged. Adapter factors and norm gains are forward snapshots. Rounding
//! boundaries use the established FP32 STE contract, with fixed sparse indices.

use super::*;
use crate::deepseek_v41::{
    DeepseekV41AdapterGradient, DeepseekV41AttentionAdjoints, DeepseekV41LinearAdjoints,
    DeepseekV41LinearGradients, DeepseekV41LinearTape, DeepseekV41TerminalAdjoints,
};

#[cfg(test)]
#[path = "attention_projection_grad_tests.rs"]
mod tests;

struct Geometry {
    config: DeepseekV41AttentionConfig,
    stream: DeepseekV41WeightOutput,
    query: DeepseekV41WeightOutput,
    batch: usize,
    seq: usize,
    position: usize,
}
impl Geometry {
    fn rows(&self) -> usize {
        self.batch * self.seq
    }
}

pub struct DeepseekV41AttentionInputTape {
    pub query: GpuBuffer,
    pub window_kv: GpuBuffer,
    /// The exact query latent shared with the discrete indexer at forward.
    pub query_latent: GpuBuffer,
    qa: DeepseekV41LinearTape,
    qb: DeepseekV41LinearTape,
    kv: DeepseekV41LinearTape,
    q_norm: GpuBuffer,
    kv_norm: GpuBuffer,
    geometry: Geometry,
}
pub struct DeepseekV41AttentionOutputTape {
    pub output: GpuBuffer,
    ob: DeepseekV41LinearTape,
    geometry: Geometry,
}
pub struct DeepseekV41AttentionProjectionGradients {
    /// Input tape: dX of the normalized attention stream, or None if no branch
    /// is used. Output tape: d(sparse head output), always Some.
    pub grad_input: Option<GpuBuffer>,
    /// Input tape only: total d(query latent), including an optional auxiliary
    /// branch. No gradient is invented for the discrete index selection.
    pub grad_query_latent: Option<GpuBuffer>,
    adapters: Vec<(&'static str, DeepseekV41LinearGradients)>,
}
impl DeepseekV41AttentionProjectionGradients {
    /// Read small adapter gradients at the optimizer boundary. Across calls,
    /// accumulate matching targets before issuing a single optimizer step.
    pub fn adapter_gradients(&self, prefix: &str) -> Result<Vec<DeepseekV41AdapterGradient>> {
        if prefix.is_empty() || prefix.ends_with('.') || prefix.chars().any(char::is_whitespace) {
            bail!("V4.1 attention adapter gradient prefix is invalid");
        }
        let mut result = Vec::new();
        for (name, gradient) in &self.adapters {
            if gradient.grad_a.is_some() {
                result.push(gradient.read_adapter_gradient(&format!("{prefix}.{name}"))?);
            }
        }
        result.sort_by(|a, b| a.target.cmp(&b.target));
        Ok(result)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddPush {
    len: u32,
    scale: f32,
    add: u32,
}

pub struct DeepseekV41AttentionProjectionAdjoints {
    device: VulkanDevice,
    linear: DeepseekV41LinearAdjoints,
    norm: DeepseekV41TerminalAdjoints,
    attention: DeepseekV41AttentionAdjoints,
    vector: vulkan::ComputeKernel,
}
impl DeepseekV41AttentionProjectionAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        use vulkan::BindingAccess::{ReadOnly as R, WriteOnly as W};
        Ok(Self {
            linear: DeepseekV41LinearAdjoints::new(device.clone())?,
            norm: DeepseekV41TerminalAdjoints::new(device.clone())?,
            attention: DeepseekV41AttentionAdjoints::new(device.clone())?,
            vector: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_projection_grad_vector.spv"),
                &[R, R, W],
                std::mem::size_of::<AddPush>() as u32,
            )?,
            device,
        })
    }
    fn check(&self, buffer: &GpuBuffer, len: usize) -> Result<()> {
        if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < len {
            bail!("V4.1 attention projection tape buffer geometry or device mismatch");
        }
        Ok(())
    }
    fn geometry(
        &self,
        a: &DeepseekV41Attention,
        batch: usize,
        seq: usize,
        position: usize,
    ) -> Result<Geometry> {
        let rows = checked(&[batch, seq])?;
        checked(&[rows, a.config.hidden_size])?;
        checked(&[rows, a.config.heads, a.config.head_dim])?;
        checked(&[rows, a.config.q_lora_rank])?;
        checked(&[rows, a.config.o_groups, a.config.o_lora_rank])?;
        if !a.device.shares_logical_device_with(&self.device)
            || batch > a.window_cache.batch()
            || rows > 65_535
            || a.config.o_groups > 65_535
            || a.config.o_lora_rank.div_ceil(64) > 65_535
            || position
                .checked_add(seq)
                .is_none_or(|end| end > a.config.rotary.max_positions)
        {
            bail!("V4.1 attention projection tape exceeds device, context or dispatch capacity");
        }
        a.projections
            .as_ref()
            .context("V4.1 attention projections are not loaded")?;
        Ok(Geometry {
            config: a.config.clone(),
            stream: a.stream_precision,
            query: a.query_precision,
            batch,
            seq,
            position,
        })
    }
    fn validate(&self, a: &DeepseekV41Attention, g: &Geometry) -> Result<()> {
        self.geometry(a, g.batch, g.seq, g.position)?;
        if a.config != g.config || a.stream_precision != g.stream || a.query_precision != g.query {
            bail!("V4.1 attention projection backward configuration differs from its tape");
        }
        Ok(())
    }
    pub(crate) fn add(&self, left: &GpuBuffer, right: &GpuBuffer, len: usize) -> Result<GpuBuffer> {
        self.check(left, len)?;
        self.check(right, len)?;
        let out = GpuBuffer::zeros_f32(&self.device, len)?;
        let groups = len.div_ceil(64);
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.vector.record_dispatch(
            &mut commands,
            &[left, right, &out],
            bytemuck::bytes_of(&AddPush {
                len: len as u32,
                scale: 1.0,
                add: 1,
            }),
            [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1],
        )?;
        commands.submit()?;
        Ok(out)
    }

    /// Same Q and window-KV preparation as native attention, with retained
    /// projection operands. The cache tape owns the ensuing seed/decode writes.
    pub fn forward_inputs(
        &self,
        a: &DeepseekV41Attention,
        x: &GpuBuffer,
        batch: usize,
        seq: usize,
        position: usize,
    ) -> Result<DeepseekV41AttentionInputTape> {
        let g = self.geometry(a, batch, seq, position)?;
        let c = &g.config;
        let rows = g.rows();
        self.check(x, rows * c.hidden_size)?;
        let p = a.projections.as_ref().unwrap();
        let qa = self.linear.forward_mixed(
            x,
            rows,
            &p.wq_a,
            p.wq_a.input_precision(a.stream_precision),
        )?;
        let q_norm = p.q_norm.duplicate_f32(c.q_lora_rank)?;
        let qr = a.norm.apply(
            &self.device,
            &qa.output,
            &q_norm,
            rows,
            c.q_lora_rank,
            c.norm_eps,
        )?;
        let query_latent =
            a.linear
                .cast_buffer(&qr, rows * c.q_lora_rank, p.wq_a.output_precision())?;
        let qb = self.linear.forward_mixed(
            &query_latent,
            rows,
            &p.wq_b,
            p.wq_b.input_precision(a.query_precision),
        )?;
        let query = a.rotary.apply_repeated_buffer(
            &qb.output,
            rows * c.heads,
            c.head_dim,
            c.heads,
            seq,
            position,
            1,
            false,
        )?;
        let query = a.linear.cast_buffer(
            &query,
            rows * c.heads * c.head_dim,
            p.wq_b.output_precision(),
        )?;
        let kv = self.linear.forward_mixed(
            x,
            rows,
            &p.wkv,
            p.wkv.input_precision(a.stream_precision),
        )?;
        let kv_norm = p.kv_norm.duplicate_f32(c.head_dim)?;
        let normalized = a.norm.apply(
            &self.device,
            &kv.output,
            &kv_norm,
            rows,
            c.head_dim,
            c.norm_eps,
        )?;
        let normalized =
            a.linear
                .cast_buffer(&normalized, rows * c.head_dim, p.wkv.output_precision())?;
        let rotated = a.rotary.apply_grouped_buffer(
            &normalized,
            rows,
            c.head_dim,
            seq,
            position,
            1,
            false,
        )?;
        let rotated =
            a.linear
                .cast_buffer(&rotated, rows * c.head_dim, p.wkv.output_precision())?;
        let quantized = a.quantizer.quantize_buffer(
            &rotated,
            rows,
            c.head_dim,
            KV_QUANT_BLOCK,
            DeepseekV41QuantMode::Fp8E4M3,
            DeepseekV41ScaleMode::Ue8m0,
        )?;
        let window_kv = a.linear.cast_buffer(
            &quantized.values,
            rows * c.head_dim,
            p.wkv.output_precision(),
        )?;
        Ok(DeepseekV41AttentionInputTape {
            query,
            window_kv,
            query_latent,
            qa,
            qb,
            kv,
            q_norm,
            kv_norm,
            geometry: g,
        })
    }

    /// Q and window source gradients are supplied by the completed cache tape,
    /// after all readers and later steps have joined. Extra latent gradients
    /// are optional explicit edges (e.g. an independently qualified aux loss).
    pub fn backward_inputs(
        &self,
        a: &DeepseekV41Attention,
        t: &DeepseekV41AttentionInputTape,
        grad_query: Option<&GpuBuffer>,
        grad_window: Option<&GpuBuffer>,
        extra_latent: Option<&GpuBuffer>,
    ) -> Result<DeepseekV41AttentionProjectionGradients> {
        self.validate(a, &t.geometry)?;
        let g = &t.geometry;
        let c = &g.config;
        let rows = g.rows();
        // Validate every external edge before any branch executes.
        for (edge, n) in [
            (grad_query, rows * c.heads * c.head_dim),
            (grad_window, rows * c.head_dim),
            (extra_latent, rows * c.q_lora_rank),
        ] {
            if let Some(edge) = edge {
                self.check(edge, n)?;
            }
        }
        let p = a.projections.as_ref().unwrap();
        let mut adapters = Vec::new();
        let mut latent = match grad_query {
            Some(upstream) => {
                let rotated = a.rotary.input_gradient_repeated(
                    upstream,
                    rows * c.heads,
                    c.head_dim,
                    c.heads,
                    g.seq,
                    g.position,
                    1,
                    false,
                )?;
                let qb = self.linear.backward(&t.qb, &p.wq_b, &rotated)?;
                #[cfg(test)]
                {
                    crate::deepseek_v41::training_trace::record("qb/input", &t.query_latent, rows*c.q_lora_rank);
                    crate::deepseek_v41::training_trace::record("qb/upstream", &rotated, rows*c.heads*c.head_dim);
                    if let Some(a) = &qb.grad_a { crate::deepseek_v41::training_trace::record("qb/grad_a",a,a.f32_capacity()); }
                }
                let result = qb.grad_input.clone();
                adapters.push(("wq_b", qb));
                Some(result)
            }
            None => None,
        };
        if let Some(extra) = extra_latent {
            latent = Some(match latent {
                Some(existing) => self.add(&existing, extra, rows * c.q_lora_rank)?,
                None => extra.duplicate_f32(rows * c.q_lora_rank)?,
            });
        }
        let mut input = match &latent {
            Some(upstream) => {
                let norm = self.norm.norm_input_grad(
                    upstream,
                    &t.qa.output,
                    &t.q_norm,
                    rows,
                    c.q_lora_rank,
                    c.norm_eps,
                )?;
                let qa = self.linear.backward(&t.qa, &p.wq_a, &norm)?;
                let result = qa.grad_input.clone();
                adapters.push(("wq_a", qa));
                Some(result)
            }
            None => None,
        };
        if let Some(upstream) = grad_window {
            let rotated = a.rotary.input_gradient_repeated(
                upstream, rows, c.head_dim, 1, g.seq, g.position, 1, false,
            )?;
            let norm = self.norm.norm_input_grad(
                &rotated,
                &t.kv.output,
                &t.kv_norm,
                rows,
                c.head_dim,
                c.norm_eps,
            )?;
            let kv = self.linear.backward(&t.kv, &p.wkv, &norm)?;
            input = Some(match input {
                Some(q) => self.add(&q, &kv.grad_input, rows * c.hidden_size)?,
                None => kv.grad_input.clone(),
            });
            adapters.push(("wkv", kv));
        }
        Ok(DeepseekV41AttentionProjectionGradients {
            grad_input: input,
            grad_query_latent: latent,
            adapters,
        })
    }

    /// Native inverse RoPE, FP8 rounding, frozen grouped wo_a and adapted wo_b.
    pub fn forward_output(
        &self,
        a: &DeepseekV41Attention,
        sparse: &GpuBuffer,
        batch: usize,
        seq: usize,
        position: usize,
    ) -> Result<DeepseekV41AttentionOutputTape> {
        let g = self.geometry(a, batch, seq, position)?;
        let c = &g.config;
        let rows = g.rows();
        let n = rows * c.heads * c.head_dim;
        self.check(sparse, n)?;
        let p = a.projections.as_ref().unwrap();
        let sparse = a.linear.cast_buffer(sparse, n, p.wq_b.output_precision())?;
        let rotated = a.rotary.apply_repeated_buffer(
            &sparse,
            rows * c.heads,
            c.head_dim,
            c.heads,
            seq,
            position,
            1,
            true,
        )?;
        let rotated = a
            .linear
            .cast_buffer(&rotated, n, p.wq_b.output_precision())?;
        let quantized = a.quantizer.quantize_buffer(
            &rotated,
            rows,
            c.heads * c.head_dim,
            KV_QUANT_BLOCK,
            DeepseekV41QuantMode::Fp8E4M3,
            DeepseekV41ScaleMode::Ue8m0,
        )?;
        let grouped_input = a
            .linear
            .cast_buffer(&quantized.values, n, p.wo_a_precision)?;
        let grouped = GpuBuffer::zeros_f32(&self.device, rows * c.o_groups * c.o_lora_rank)?;
        let push = GroupedPush {
            rows: rows as u32,
            groups: c.o_groups as u32,
            group_width: a.group_width as u32,
            output_rank: c.o_lora_rank as u32,
            bf16: u32::from(p.wo_a_precision == DeepseekV41WeightOutput::Bf16),
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        a.grouped_kernel.record_dispatch(
            &mut commands,
            &[&grouped_input, &p.wo_a, &grouped],
            bytemuck::bytes_of(&push),
            [
                c.o_lora_rank.div_ceil(64) as u32,
                c.o_groups as u32,
                rows as u32,
            ],
        )?;
        commands.submit()?;
        let grouped = a.linear.cast_buffer(
            &grouped,
            rows * c.o_groups * c.o_lora_rank,
            p.wo_a_precision,
        )?;
        let ob = self.linear.forward_mixed(
            &grouped,
            rows,
            &p.wo_b,
            p.wo_b.input_precision(p.wo_a_precision),
        )?;
        Ok(DeepseekV41AttentionOutputTape {
            output: ob.output.clone(),
            ob,
            geometry: g,
        })
    }
    pub fn backward_output(
        &self,
        a: &DeepseekV41Attention,
        t: &DeepseekV41AttentionOutputTape,
        upstream: &GpuBuffer,
    ) -> Result<DeepseekV41AttentionProjectionGradients> {
        self.validate(a, &t.geometry)?;
        let g = &t.geometry;
        let c = &g.config;
        let rows = g.rows();
        self.check(upstream, rows * c.hidden_size)?;
        let p = a.projections.as_ref().unwrap();
        let ob = self.linear.backward(&t.ob, &p.wo_b, upstream)?;
        let grouped = self.attention.grouped_output(
            &ob.grad_input,
            &p.wo_a,
            rows,
            c.o_groups,
            a.group_width,
            c.o_lora_rank,
        )?;
        let grad = a.rotary.input_gradient_repeated(
            &grouped,
            rows * c.heads,
            c.head_dim,
            c.heads,
            g.seq,
            g.position,
            1,
            true,
        )?;
        Ok(DeepseekV41AttentionProjectionGradients {
            grad_input: Some(grad),
            grad_query_latent: None,
            adapters: vec![("wo_b", ob)],
        })
    }
}
fn checked(shape: &[usize]) -> Result<usize> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
        .context("V4.1 attention projection tape zero/overflowing geometry")?;
    if n > u32::MAX as usize {
        bail!("V4.1 attention projection tape exceeds u32 addressing");
    }
    Ok(n)
}
