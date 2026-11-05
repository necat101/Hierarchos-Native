//! Native Gemma4 vision -> multimodal embedder -> Gemma4 text training.
//!
//! First qualified geometry: fixed 12x12 RGB, 2x2 patches, 6x6 grid, 3x3
//! pooling and four soft tokens. It owns the Gemma4-specific arithmetic:
//! learned 2-D position tables, axial H-H-W-W RoPE, Q/K RMSNorm, a
//! parameter-free V RMSNorm, four sandwich RMSNorms per layer, non-causal
//! unit-scale attention, spatial pooling and a no-scale RMSNorm embedder.
//! Padding, variable grids, standardize and clipped linears fail closed.
use super::vision::{
    cpu_avx2_input_gradient_gemm, cpu_avx2_vision_gemm, tensor, PatchPush, VisionKernels,
};
use super::*;
use crate::weight_offload::ModelPackage;

#[cfg(test)]
#[path = "gemma4_vision_cpu_tests.rs"]
mod cpu_tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PositionsPush {
    rows: u32,
    dim: u32,
    position_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopePush {
    rows: u32,
    heads: u32,
    head_dim: u32,
    inverse: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalePush {
    len: u32,
    scale: f32,
    offset: f32,
    inverse: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PoolPush {
    images: u32,
    grid: u32,
    kernel: u32,
    dim: u32,
    mode: u32,
}

#[derive(Debug, Serialize)]
pub struct Gemma4Forward {
    pub logits: Vec<f32>,
    pub vision_features: Vec<f32>,
    pub projected_features: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct Gemma4TrainResult {
    pub text: VulkanTransformerStepResult,
    pub patch_gradient: Vec<f32>,
    pub projected_gradient: Vec<f32>,
}

const N1: usize = 0;
const Q: usize = 1;
const QN: usize = 2;
const QR: usize = 3;
const K: usize = 4;
const KN: usize = 5;
const KR: usize = 6;
const V: usize = 7;
const VN: usize = 8;
const A: usize = 9;
const O: usize = 10;
const N2: usize = 11;
const R: usize = 12;
const N3: usize = 13;
const G: usize = 14;
const GA: usize = 15;
const U: usize = 16;
const P: usize = 17;
const D: usize = 18;
const N4: usize = 19;
const OUT: usize = 20;
const DI: usize = 21;
const D_DOWN: usize = 22;
const D_P: usize = 23;
const D_GA: usize = 24;
const D_G: usize = 25;
const D_U: usize = 26;
const D_N3: usize = 27;
const D_R: usize = 28;
const D_O: usize = 29;
const D_A: usize = 30;
const D_QR: usize = 31;
const D_KR: usize = 32;
const D_VN: usize = 33;
const D_QN: usize = 34;
const D_KN: usize = 35;
const D_Q: usize = 36;
const D_K: usize = 37;
const D_V: usize = 38;
const D_N1: usize = 39;
const D_TMP: usize = 40;
const D_Q_IN: usize = 41;
const D_K_IN: usize = 42;
const D_V_IN: usize = 43;
const DR: usize = 44;
const DIN: usize = 45;
const TAPE_LEN: usize = 46;

struct Gemma4VisionLayer {
    q: VulkanLinear,
    k: VulkanLinear,
    v: VulkanLinear,
    o: VulkanLinear,
    gate: VulkanLinear,
    up: VulkanLinear,
    down: VulkanLinear,
    ln1: VulkanLayerNorm,
    ln2: VulkanLayerNorm,
    ln3: VulkanLayerNorm,
    ln4: VulkanLayerNorm,
    q_norm: VulkanLayerNorm,
    k_norm: VulkanLayerNorm,
    v_norm: VulkanLayerNorm,
    tape: Vec<GpuBuffer>,
    ln1_mean: GpuBuffer,
    ln1_rstd: GpuBuffer,
    ln2_mean: GpuBuffer,
    ln2_rstd: GpuBuffer,
    ln3_mean: GpuBuffer,
    ln3_rstd: GpuBuffer,
    ln4_mean: GpuBuffer,
    ln4_rstd: GpuBuffer,
    q_mean: GpuBuffer,
    q_rstd: GpuBuffer,
    k_mean: GpuBuffer,
    k_rstd: GpuBuffer,
    v_mean: GpuBuffer,
    v_rstd: GpuBuffer,
    prefix: String,
}

impl Gemma4VisionLayer {
    fn linear_no_bias(
        package: &mut ModelPackage,
        device: &VulkanDevice,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<VulkanLinear> {
        let weight = tensor(package, &format!("{name}.weight"), &[output, input])?;
        VulkanLinear::new_no_bias(device, input, output, &weight)
    }

    fn rms(
        package: &mut ModelPackage,
        device: &VulkanDevice,
        name: &str,
        dim: usize,
        eps: f32,
        trainable: bool,
    ) -> Result<VulkanLayerNorm> {
        let weight = if trainable {
            tensor(package, &format!("{name}.weight"), &[dim])?
        } else {
            vec![1.0; dim]
        };
        let mut norm = VulkanLayerNorm::new_rms_pow(device, dim, eps, &weight, 0.0)?
            .use_vector_cpu_rsqrt(true);
        if !trainable {
            norm.freeze();
        }
        Ok(norm)
    }

    fn load(
        package: &mut ModelPackage,
        device: &VulkanDevice,
        prefix: String,
        rows: usize,
        dim: usize,
        intermediate: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        let rows_dim = rows * dim;
        let rows_intermediate = rows * intermediate;
        let rows_heads = rows * heads;
        let sizes = (0..TAPE_LEN).map(|index| match index {
            G | GA | U | P | D_GA | D_G | D_U | D_P => rows_intermediate,
            _ => rows_dim,
        });
        Ok(Self {
            q: Self::linear_no_bias(
                package,
                device,
                &format!("{prefix}.self_attn.q_proj.linear"),
                dim,
                dim,
            )?,
            k: Self::linear_no_bias(
                package,
                device,
                &format!("{prefix}.self_attn.k_proj.linear"),
                dim,
                dim,
            )?,
            v: Self::linear_no_bias(
                package,
                device,
                &format!("{prefix}.self_attn.v_proj.linear"),
                dim,
                dim,
            )?,
            o: Self::linear_no_bias(
                package,
                device,
                &format!("{prefix}.self_attn.o_proj.linear"),
                dim,
                dim,
            )?,
            gate: Self::linear_no_bias(
                package,
                device,
                &format!("{prefix}.mlp.gate_proj.linear"),
                dim,
                intermediate,
            )?,
            up: Self::linear_no_bias(
                package,
                device,
                &format!("{prefix}.mlp.up_proj.linear"),
                dim,
                intermediate,
            )?,
            down: Self::linear_no_bias(
                package,
                device,
                &format!("{prefix}.mlp.down_proj.linear"),
                intermediate,
                dim,
            )?,
            ln1: Self::rms(
                package,
                device,
                &format!("{prefix}.input_layernorm"),
                dim,
                eps,
                true,
            )?,
            ln2: Self::rms(
                package,
                device,
                &format!("{prefix}.post_attention_layernorm"),
                dim,
                eps,
                true,
            )?,
            ln3: Self::rms(
                package,
                device,
                &format!("{prefix}.pre_feedforward_layernorm"),
                dim,
                eps,
                true,
            )?,
            ln4: Self::rms(
                package,
                device,
                &format!("{prefix}.post_feedforward_layernorm"),
                dim,
                eps,
                true,
            )?,
            q_norm: Self::rms(
                package,
                device,
                &format!("{prefix}.self_attn.q_norm"),
                head_dim,
                eps,
                true,
            )?,
            k_norm: Self::rms(
                package,
                device,
                &format!("{prefix}.self_attn.k_norm"),
                head_dim,
                eps,
                true,
            )?,
            v_norm: Self::rms(
                package,
                device,
                &format!("{prefix}.self_attn.v_norm"),
                head_dim,
                eps,
                false,
            )?,
            tape: sizes
                .map(|len| GpuBuffer::zeros_f32(device, len))
                .collect::<Result<_>>()?,
            ln1_mean: GpuBuffer::zeros_f32(device, rows)?,
            ln1_rstd: GpuBuffer::zeros_f32(device, rows)?,
            ln2_mean: GpuBuffer::zeros_f32(device, rows)?,
            ln2_rstd: GpuBuffer::zeros_f32(device, rows)?,
            ln3_mean: GpuBuffer::zeros_f32(device, rows)?,
            ln3_rstd: GpuBuffer::zeros_f32(device, rows)?,
            ln4_mean: GpuBuffer::zeros_f32(device, rows)?,
            ln4_rstd: GpuBuffer::zeros_f32(device, rows)?,
            q_mean: GpuBuffer::zeros_f32(device, rows_heads)?,
            q_rstd: GpuBuffer::zeros_f32(device, rows_heads)?,
            k_mean: GpuBuffer::zeros_f32(device, rows_heads)?,
            k_rstd: GpuBuffer::zeros_f32(device, rows_heads)?,
            v_mean: GpuBuffer::zeros_f32(device, rows_heads)?,
            v_rstd: GpuBuffer::zeros_f32(device, rows_heads)?,
            prefix,
        })
    }

    fn linear_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        linear: &VulkanLinear,
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        if cpu_avx2_vision_gemm(
            rows,
            linear.input_dim,
            linear.output_dim,
            aten_vector_width(),
        ) {
            let push = [
                rows as u32,
                linear.input_dim as u32,
                linear.output_dim as u32,
                u32::from(linear.has_bias),
            ];
            vk.cpu_avx2_linear.record_dispatch(
                commands,
                &[input, &linear.weight.values, &linear.bias.values, output],
                bytemuck::cast_slice(&push),
                [
                    div_ceil_u32(linear.output_dim, 16),
                    div_ceil_u32(rows, 16),
                    1,
                ],
            )
        } else {
            linear.record_forward(commands, k, input, output, rows)
        }
    }

    fn linear_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        linear: &VulkanLinear,
        input: &GpuBuffer,
        gradient: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        linear.record_backward(commands, k, input, gradient, output, rows)?;
        if cpu_avx2_input_gradient_gemm(
            rows,
            linear.output_dim,
            linear.input_dim,
            aten_vector_width(),
        ) && matches!(rows % 6, 1..=3)
        {
            let push = [
                rows as u32,
                linear.output_dim as u32,
                linear.input_dim as u32,
                0,
            ];
            vk.cpu_avx2_input_grad.record_dispatch(
                commands,
                &[gradient, &linear.weight.values, &linear.bias.values, output],
                bytemuck::cast_slice(&push),
                [
                    div_ceil_u32(linear.input_dim, 16),
                    div_ceil_u32(rows, 16),
                    1,
                ],
            )?;
        }
        Ok(())
    }

    fn rms_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        norm: &VulkanLayerNorm,
        input: &GpuBuffer,
        output: &GpuBuffer,
        mean: &GpuBuffer,
        rstd: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        norm.record_forward(commands, k, input, output, mean, rstd, rows)
    }

    fn forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        x: &GpuBuffer,
        cos: &GpuBuffer,
        sin: &GpuBuffer,
        mask: &GpuBuffer,
        dummy: &GpuBuffer,
        push: AttentionPush,
        activation: &str,
        rows: usize,
        dim: usize,
        heads: usize,
        head_dim: usize,
        rope: &vulkan::ComputeKernel,
        cpu_attention: Option<&vulkan::ComputeKernel>,
    ) -> Result<()> {
        let t = &self.tape;
        let rows_heads = rows * heads;
        self.rms_forward(
            commands,
            k,
            &self.ln1,
            x,
            &t[N1],
            &self.ln1_mean,
            &self.ln1_rstd,
            rows,
        )?;
        self.linear_forward(commands, k, vk, &self.q, &t[N1], &t[Q], rows)?;
        self.rms_forward(
            commands,
            k,
            &self.q_norm,
            &t[Q],
            &t[QN],
            &self.q_mean,
            &self.q_rstd,
            rows_heads,
        )?;
        rope.record_dispatch(
            commands,
            &[&t[QN], cos, sin, &t[QR]],
            bytemuck::bytes_of(&RopePush {
                rows: rows as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                inverse: 0,
            }),
            [div_ceil_u32(rows_heads * head_dim, 256), 1, 1],
        )?;
        self.linear_forward(commands, k, vk, &self.k, &t[N1], &t[K], rows)?;
        self.rms_forward(
            commands,
            k,
            &self.k_norm,
            &t[K],
            &t[KN],
            &self.k_mean,
            &self.k_rstd,
            rows_heads,
        )?;
        rope.record_dispatch(
            commands,
            &[&t[KN], cos, sin, &t[KR]],
            bytemuck::bytes_of(&RopePush {
                rows: rows as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                inverse: 0,
            }),
            [div_ceil_u32(rows_heads * head_dim, 256), 1, 1],
        )?;
        self.linear_forward(commands, k, vk, &self.v, &t[N1], &t[V], rows)?;
        self.rms_forward(
            commands,
            k,
            &self.v_norm,
            &t[V],
            &t[VN],
            &self.v_mean,
            &self.v_rstd,
            rows_heads,
        )?;
        if let Some(attention) = cpu_attention {
            attention.record_dispatch(
                commands,
                &[&t[QR], &t[KR], &t[VN], &t[A]],
                bytemuck::cast_slice(&[
                    push.batch_size,
                    push.query_seq_len,
                    push.query_hidden_size,
                    push.num_heads,
                ]),
                [push.query_seq_len, push.num_heads, push.batch_size],
            )?;
        } else {
            vk.attention_forward.record_dispatch(
                commands,
                &[
                    &t[QR], &t[KR], &t[VN], mask, mask, dummy, &t[A], dummy, dummy, mask, mask,
                ],
                bytemuck::bytes_of(&push),
                [push.query_seq_len, push.num_heads, push.batch_size],
            )?;
        }
        self.linear_forward(commands, k, vk, &self.o, &t[A], &t[O], rows)?;
        self.rms_forward(
            commands,
            k,
            &self.ln2,
            &t[O],
            &t[N2],
            &self.ln2_mean,
            &self.ln2_rstd,
            rows,
        )?;
        record_add(commands, k, x, &t[N2], &t[R], rows * dim)?;
        self.rms_forward(
            commands,
            k,
            &self.ln3,
            &t[R],
            &t[N3],
            &self.ln3_mean,
            &self.ln3_rstd,
            rows,
        )?;
        self.linear_forward(commands, k, vk, &self.gate, &t[N3], &t[G], rows)?;
        vk.activation(
            k,
            activation,
            commands,
            &t[G],
            None,
            &t[GA],
            rows * self.gate.output_dim,
        )?;
        self.linear_forward(commands, k, vk, &self.up, &t[N3], &t[U], rows)?;
        kernels_mul(
            commands,
            k,
            &t[GA],
            &t[U],
            &t[P],
            rows * self.gate.output_dim,
        )?;
        self.linear_forward(commands, k, vk, &self.down, &t[P], &t[D], rows)?;
        self.rms_forward(
            commands,
            k,
            &self.ln4,
            &t[D],
            &t[N4],
            &self.ln4_mean,
            &self.ln4_rstd,
            rows,
        )?;
        record_add(commands, k, &t[R], &t[N4], &t[OUT], rows * dim)
    }

    #[allow(clippy::too_many_arguments)]
    fn backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        x: &GpuBuffer,
        dy: &GpuBuffer,
        cos: &GpuBuffer,
        sin: &GpuBuffer,
        mask: &GpuBuffer,
        dummy: &GpuBuffer,
        push: AttentionPush,
        activation: &str,
        rows: usize,
        dim: usize,
        heads: usize,
        head_dim: usize,
        rope: &vulkan::ComputeKernel,
        cpu_attention: Option<&vulkan::ComputeKernel>,
    ) -> Result<()> {
        let t = &self.tape;
        let rows_dim = rows * dim;
        let rows_intermediate = rows * self.gate.output_dim;
        let rows_heads = rows * heads;
        self.ln4.record_backward(
            commands,
            k,
            &t[D],
            dy,
            &self.ln4_mean,
            &self.ln4_rstd,
            &t[D_DOWN],
            rows,
        )?;
        self.linear_backward(
            commands, k, vk, &self.down, &t[P], &t[D_DOWN], &t[D_P], rows,
        )?;
        kernels_mul_backward(
            commands,
            k,
            &t[D_P],
            &t[GA],
            &t[U],
            &t[D_GA],
            &t[D_U],
            rows_intermediate,
        )?;
        vk.activation(
            k,
            activation,
            commands,
            &t[G],
            Some(&t[D_GA]),
            &t[D_G],
            rows_intermediate,
        )?;
        self.linear_backward(
            commands, k, vk, &self.gate, &t[N3], &t[D_G], &t[D_Q_IN], rows,
        )?;
        self.linear_backward(commands, k, vk, &self.up, &t[N3], &t[D_U], &t[D_K_IN], rows)?;
        record_add(commands, k, &t[D_Q_IN], &t[D_K_IN], &t[D_N3], rows_dim)?;
        self.ln3.record_backward(
            commands,
            k,
            &t[R],
            &t[D_N3],
            &self.ln3_mean,
            &self.ln3_rstd,
            &t[D_R],
            rows,
        )?;
        record_add(commands, k, &t[D_R], dy, &t[DR], rows_dim)?;
        self.ln2.record_backward(
            commands,
            k,
            &t[O],
            &t[DR],
            &self.ln2_mean,
            &self.ln2_rstd,
            &t[D_O],
            rows,
        )?;
        self.linear_backward(commands, k, vk, &self.o, &t[A], &t[D_O], &t[D_A], rows)?;
        if let Some(attention) = cpu_attention {
            attention.record_dispatch(
                commands,
                &[
                    &t[QR], &t[KR], &t[VN], &t[D_A], &t[D_QR], &t[D_KR], &t[D_VN],
                ],
                bytemuck::cast_slice(&[
                    push.batch_size,
                    push.query_seq_len,
                    push.query_hidden_size,
                    push.num_heads,
                ]),
                [push.num_heads, push.batch_size, 1],
            )?;
        } else {
            vk.attention_backward.record_dispatch(
                commands,
                &[
                    &t[QR], &t[KR], &t[VN], mask, mask, dummy, &t[D_A], &t[D_QR], &t[D_KR],
                    &t[D_VN], dummy, dummy, dummy, dummy, mask,
                ],
                bytemuck::bytes_of(&push.as_backward_push(0, 1)),
                [push.num_heads, push.batch_size, 1],
            )?;
        }
        rope.record_dispatch(
            commands,
            &[&t[D_QR], cos, sin, &t[D_QN]],
            bytemuck::bytes_of(&RopePush {
                rows: rows as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                inverse: 1,
            }),
            [div_ceil_u32(rows_heads * head_dim, 256), 1, 1],
        )?;
        rope.record_dispatch(
            commands,
            &[&t[D_KR], cos, sin, &t[D_KN]],
            bytemuck::bytes_of(&RopePush {
                rows: rows as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                inverse: 1,
            }),
            [div_ceil_u32(rows_heads * head_dim, 256), 1, 1],
        )?;
        self.q_norm.record_backward(
            commands,
            k,
            &t[Q],
            &t[D_QN],
            &self.q_mean,
            &self.q_rstd,
            &t[D_Q],
            rows_heads,
        )?;
        self.k_norm.record_backward(
            commands,
            k,
            &t[K],
            &t[D_KN],
            &self.k_mean,
            &self.k_rstd,
            &t[D_K],
            rows_heads,
        )?;
        self.v_norm.record_backward(
            commands,
            k,
            &t[V],
            &t[D_VN],
            &self.v_mean,
            &self.v_rstd,
            &t[D_V],
            rows_heads,
        )?;
        self.linear_backward(commands, k, vk, &self.q, &t[N1], &t[D_Q], &t[D_Q_IN], rows)?;
        self.linear_backward(commands, k, vk, &self.k, &t[N1], &t[D_K], &t[D_K_IN], rows)?;
        self.linear_backward(commands, k, vk, &self.v, &t[N1], &t[D_V], &t[D_V_IN], rows)?;
        record_add(commands, k, &t[D_Q_IN], &t[D_K_IN], &t[D_TMP], rows_dim)?;
        record_add(commands, k, &t[D_TMP], &t[D_V_IN], &t[D_N1], rows_dim)?;
        self.ln1.record_backward(
            commands,
            k,
            x,
            &t[D_N1],
            &self.ln1_mean,
            &self.ln1_rstd,
            &t[DI],
            rows,
        )?;
        record_add(commands, k, &t[DR], &t[DI], &t[DIN], rows_dim)
    }

    fn step(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        step: u32,
        hyper: AdamWHyperParams,
    ) -> Result<()> {
        for linear in [
            &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
        ] {
            linear.record_step(commands, k, step, hyper)?;
        }
        for norm in [
            &self.ln1,
            &self.ln2,
            &self.ln3,
            &self.ln4,
            &self.q_norm,
            &self.k_norm,
        ] {
            norm.record_step(commands, k, step, hyper)?;
        }
        Ok(())
    }

    fn export(&self, values: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> Result<()> {
        for (suffix, linear) in [
            ("self_attn.q_proj.linear", &self.q),
            ("self_attn.k_proj.linear", &self.k),
            ("self_attn.v_proj.linear", &self.v),
            ("self_attn.o_proj.linear", &self.o),
            ("mlp.gate_proj.linear", &self.gate),
            ("mlp.up_proj.linear", &self.up),
            ("mlp.down_proj.linear", &self.down),
        ] {
            values.insert(
                format!("{}.{suffix}.weight", self.prefix),
                (
                    vec![linear.output_dim, linear.input_dim],
                    linear.weight.read()?,
                ),
            );
        }
        for (suffix, norm) in [
            ("self_attn.q_norm", &self.q_norm),
            ("self_attn.k_norm", &self.k_norm),
            ("input_layernorm", &self.ln1),
            ("post_attention_layernorm", &self.ln2),
            ("pre_feedforward_layernorm", &self.ln3),
            ("post_feedforward_layernorm", &self.ln4),
        ] {
            values.insert(
                format!("{}.{suffix}.weight", self.prefix),
                (vec![norm.dim], norm.weight.read()?),
            );
        }
        Ok(())
    }
}

fn kernels_mul(
    commands: &mut vulkan::ComputeBatch,
    kernels: &TransformerKernels,
    left: &GpuBuffer,
    right: &GpuBuffer,
    output: &GpuBuffer,
    len: usize,
) -> Result<()> {
    kernels.elementwise_mul_forward.record_dispatch(
        commands,
        &[left, right, output],
        bytemuck::bytes_of(&LenPush { len: len as u32 }),
        [div_ceil_u32(len, 256), 1, 1],
    )
}

fn kernels_mul_backward(
    commands: &mut vulkan::ComputeBatch,
    kernels: &TransformerKernels,
    gradient: &GpuBuffer,
    left: &GpuBuffer,
    right: &GpuBuffer,
    grad_left: &GpuBuffer,
    grad_right: &GpuBuffer,
    len: usize,
) -> Result<()> {
    kernels.elementwise_mul_backward.record_dispatch(
        commands,
        &[gradient, left, right, grad_left, grad_right],
        bytemuck::bytes_of(&LenPush { len: len as u32 }),
        [div_ceil_u32(len, 256), 1, 1],
    )
}

fn rope_tables_gpu(
    device: &VulkanDevice,
    positions: &[u32],
    head_dim: usize,
    theta: f32,
) -> Result<(GpuBuffer, GpuBuffer)> {
    let quarter = head_dim / 4;
    let inv_freq = (0..quarter)
        .map(|index| 1.0 / theta.powf(index as f32 / quarter as f32))
        .collect::<Vec<_>>();
    let rows = positions.len() / 2;
    let positions = GpuBuffer::from_u32(device, positions)?;
    let frequencies = GpuBuffer::from_f32(device, &inv_freq)?;
    let cos = GpuBuffer::zeros_f32(device, rows * head_dim)?;
    let sin = GpuBuffer::zeros_f32(device, rows * head_dim)?;
    let table = vulkan::ComputeKernel::new(
        device,
        include_bytes!("../../shaders/vision_gemma4_rope_tables.spv"),
        4,
        8,
    )?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    table.record_dispatch(
        &mut commands,
        &[&positions, &frequencies, &cos, &sin],
        bytemuck::cast_slice(&[rows as u32, head_dim as u32]),
        [div_ceil_u32(rows * head_dim, 256), 1, 1],
    )?;
    commands.submit()?;
    Ok((cos, sin))
}

pub struct VulkanGemma4Vision {
    device: VulkanDevice,
    language: VulkanTransformer,
    vk: VisionKernels,
    image_count: usize,
    side: usize,
    patch_size: usize,
    channels: usize,
    dim: usize,
    intermediate: usize,
    heads: u32,
    head_dim: usize,
    grid: usize,
    kernel: usize,
    tokens_per_image: usize,
    rows: usize,
    patch_dim: usize,
    image_token: u32,
    eps: f32,
    position_size: usize,
    text_dim: usize,
    processor: Option<serde_json::Value>,
    activation: String,
    vision_prefix: String,
    projector_prefix: String,
    patch: VulkanLinear,
    position: TrainableParameter,
    layers: Vec<Gemma4VisionLayer>,
    projector_norm: VulkanLayerNorm,
    projector: VulkanLinear,
    pixels: GpuBuffer,
    patches: GpuBuffer,
    patch_input: GpuBuffer,
    patch_output: GpuBuffer,
    tokens: GpuBuffer,
    positions: GpuBuffer,
    position_values: Vec<u32>,
    cos: GpuBuffer,
    sin: GpuBuffer,
    pooled: GpuBuffer,
    pooled_scaled: GpuBuffer,
    pooled_norm: GpuBuffer,
    projected: GpuBuffer,
    projector_mean: GpuBuffer,
    projector_rstd: GpuBuffer,
    mask: GpuBuffer,
    dummy: GpuBuffer,
    grad_projected: GpuBuffer,
    grad_pooled_norm: GpuBuffer,
    grad_pooled: GpuBuffer,
    grad_pooled_scaled: GpuBuffer,
    grad_tokens: GpuBuffer,
    grad_patch_input: GpuBuffer,
    grad_patches: GpuBuffer,
    grad_pixels: GpuBuffer,
    position_kernel: vulkan::ComputeKernel,
    position_grad_kernel: vulkan::ComputeKernel,
    rope_kernel: vulkan::ComputeKernel,
    avg_pool: vulkan::ComputeKernel,
    cpu_attention: Option<(vulkan::ComputeKernel, vulkan::ComputeKernel)>,
    cpu_weight_grad: bool,
    scale_kernel: vulkan::ComputeKernel,
}

impl VulkanGemma4Vision {
    pub fn from_hf_package(
        device: VulkanDevice,
        directory: impl AsRef<Path>,
        batch_size: usize,
        seq_len: usize,
        image_count: usize,
        side: usize,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        if config["model_type"].as_str() != Some("gemma4") {
            bail!("native Gemma4 image pipeline requires model_type=gemma4");
        }
        let vision = &config["vision_config"];
        if vision["model_type"].as_str() != Some("gemma4_vision") {
            bail!("native Gemma4 requires a gemma4_vision tower");
        }
        let number = |value: &serde_json::Value, name: &str| -> Result<usize> {
            let x = usize::try_from(
                value[name]
                    .as_u64()
                    .with_context(|| format!("missing positive {name}"))?,
            )?;
            if x == 0 || x > u32::MAX as usize {
                bail!("invalid {name}");
            }
            Ok(x)
        };
        let patch_size = number(vision, "patch_size")?;
        let channels = 3usize;
        let dim = number(vision, "hidden_size")?;
        let intermediate = number(vision, "intermediate_size")?;
        let num_layers = number(vision, "num_hidden_layers")?;
        let heads = number(vision, "num_attention_heads")?;
        let kv_heads = number(vision, "num_key_value_heads")?;
        let head_dim = number(vision, "head_dim")?;
        let kernel = number(vision, "pooling_kernel_size")?;
        let position_size = number(vision, "position_embedding_size")?;
        if image_count != 1 {
            bail!("native Gemma4 vision currently qualifies exactly one image");
        }
        if !side.is_multiple_of(patch_size) {
            bail!("Gemma4 image side must be divisible by patch_size");
        }
        let grid = side / patch_size;
        if !grid.is_multiple_of(kernel) || position_size < grid {
            bail!("Gemma4 pooling grid or position table is too small");
        }
        let tokens_per_image = (grid / kernel) * (grid / kernel);
        let rows = grid * grid;
        let patch_dim = channels * patch_size * patch_size;
        let rope = vision["rope_parameters"]
            .as_object()
            .context("Gemma4 vision rope_parameters must be an object")?;
        if rope.get("rope_type").and_then(|v| v.as_str()) != Some("axial") {
            bail!("native Gemma4 vision requires axial RoPE");
        }
        let theta = rope
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .context("missing Gemma4 vision rope_theta")? as f32;
        if !theta.is_finite() || theta <= 1.0 || !head_dim.is_multiple_of(4) {
            bail!("invalid Gemma4 axial RoPE geometry");
        }
        if heads != kv_heads {
            bail!("native Gemma4 vision currently requires equal Q/KV head counts");
        }
        if heads.checked_mul(head_dim) != Some(dim) {
            bail!("Gemma4 vision hidden size must equal heads*head_dim");
        }
        if vision["standardize"].as_bool().unwrap_or(false)
            || vision["use_clipped_linears"].as_bool().unwrap_or(false)
        {
            bail!("Gemma4 vision standardize/clipped linears are not yet represented");
        }
        if vision["attention_dropout"].as_f64().unwrap_or(0.0) != 0.0 {
            bail!("native Gemma4 vision training requires attention_dropout=0");
        }
        let activation = match vision["hidden_activation"]
            .as_str()
            .or_else(|| vision["hidden_act"].as_str())
            .unwrap_or("gelu_pytorch_tanh")
        {
            "gelu_pytorch_tanh" => "gelu_pytorch_tanh".to_string(),
            "gelu_new" => "gelu_pytorch_tanh_ha".to_string(),
            "gelu" => "gelu".to_string(),
            other => bail!("unsupported Gemma4 vision activation {other}"),
        };
        let eps = vision["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        if !eps.is_finite() || eps <= 0.0 {
            bail!("invalid Gemma4 vision RMS epsilon");
        }
        let mut language =
            VulkanTransformer::from_hf_package(device.clone(), directory, batch_size, seq_len)?;
        if language.config.architecture != VulkanTransformerArchitecture::Gemma4 {
            bail!("native Gemma4 vision requires its qualified Gemma4 text backbone");
        }
        if language.config.embedding_dropout != 0.0
            || language.config.attention_dropout != 0.0
            || language.config.mlp_output_dropout != 0.0
            || language.config.attention_output_dropout != 0.0
        {
            bail!("native Gemma4 training requires deterministic zero-dropout text configuration");
        }
        language.kernels.cpu_row_gemm = Some((
            vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
                4,
                16,
            )?,
            vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv"),
                4,
                16,
            )?,
        ));
        let text_dim = language.config.effective_embedding_size();
        let cpu_weight_grad = aten_vector_width() == 8
            && rows == 36
            && dim == 16
            && intermediate == 32
            && heads == 4
            && head_dim == 4
            && batch_size == 1
            && seq_len == 8
            && text_dim == 16
            && language.config.intermediate_size == 32
            && language.config.query_hidden_size() == 16
            && language.config.key_value_hidden_size() == 16
            && language.config.num_heads == 2
            && language.config.vocab_size == 64;
        if cpu_weight_grad {
            language.kernels.linear_weight_grad = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_linear_weight_grad_cpu_avx2.spv"),
                3,
                12,
            )?;
            language.kernels.attention_forward = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/transformer_attention_forward_gemma3_cpu_avx2.spv"),
                11,
                std::mem::size_of::<AttentionPush>() as u32,
            )?;
            language.kernels.attention_backward = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/transformer_attention_backward_gemma4_cpu_avx2.spv"),
                15,
                std::mem::size_of::<AttentionBackwardPush>() as u32,
            )?;
        }
        let image_token = u32::try_from(
            config["image_token_id"]
                .as_u64()
                .context("missing image_token_id")?,
        )?;
        if image_token as usize >= language.config.vocab_size {
            bail!("image token must be in the text vocabulary");
        }
        let mut package = ModelPackage::open_directory(directory)?;
        let names = package.tensor_names()?;
        let vision_prefix = ["model.vision_tower", "vision_tower"]
            .into_iter()
            .find(|p| names.contains(&format!("{p}.patch_embedder.input_proj.weight")))
            .context("cannot locate the Gemma4 vision tower namespace")?
            .to_string();
        let projector_prefix = ["model.embed_vision", "embed_vision"]
            .into_iter()
            .find(|p| names.contains(&format!("{p}.embedding_projection.weight")))
            .context("cannot locate the Gemma4 multimodal embedder")?
            .to_string();
        let patch_weight = tensor(
            &mut package,
            &format!("{vision_prefix}.patch_embedder.input_proj.weight"),
            &[dim, patch_dim],
        )?;
        let position_values = tensor(
            &mut package,
            &format!("{vision_prefix}.patch_embedder.position_embedding_table"),
            &[2, position_size, dim],
        )?;
        let projector_values = tensor(
            &mut package,
            &format!("{projector_prefix}.embedding_projection.weight"),
            &[text_dim, dim],
        )?;
        let positions = (0..grid)
            .flat_map(|y| (0..grid).map(move |x| [x as u32, y as u32]))
            .flatten()
            .collect::<Vec<_>>();
        let (cos, sin) = rope_tables_gpu(&device, &positions, head_dim, theta)?;
        let buffer = |len| GpuBuffer::zeros_f32(&device, len);
        let mut projector_norm =
            VulkanLayerNorm::new_rms_pow(&device, dim, eps, &vec![1.0; dim], 0.0)?
                .use_vector_cpu_rsqrt(true);
        projector_norm.freeze();
        let vk = VisionKernels::new(&device)?;
        Ok(Self {
            patch: VulkanLinear::new_no_bias(&device, patch_dim, dim, &patch_weight)?,
            position: TrainableParameter::new(&device, &position_values, true)?,
            layers: (0..num_layers)
                .map(|i| {
                    Gemma4VisionLayer::load(
                        &mut package,
                        &device,
                        format!("{vision_prefix}.encoder.layers.{i}"),
                        rows,
                        dim,
                        intermediate,
                        heads,
                        head_dim,
                        eps,
                    )
                })
                .collect::<Result<_>>()?,
            projector_norm,
            projector: VulkanLinear::new_no_bias(&device, dim, text_dim, &projector_values)?,
            pixels: buffer(channels * side * side)?,
            patches: buffer(rows * patch_dim)?,
            patch_input: buffer(rows * patch_dim)?,
            patch_output: buffer(rows * dim)?,
            tokens: buffer(rows * dim)?,
            positions: GpuBuffer::from_u32(&device, &positions)?,
            position_values: positions,
            cos,
            sin,
            pooled: buffer(tokens_per_image * dim)?,
            pooled_scaled: buffer(tokens_per_image * dim)?,
            pooled_norm: buffer(tokens_per_image * dim)?,
            projected: buffer(tokens_per_image * text_dim)?,
            projector_mean: buffer(tokens_per_image)?,
            projector_rstd: buffer(tokens_per_image)?,
            mask: GpuBuffer::from_f32(&device, &vec![1.0; rows])?,
            dummy: buffer(rows.max(heads))?,
            grad_projected: buffer(tokens_per_image * text_dim)?,
            grad_pooled_norm: buffer(tokens_per_image * dim)?,
            grad_pooled: buffer(tokens_per_image * dim)?,
            grad_pooled_scaled: buffer(tokens_per_image * dim)?,
            grad_tokens: buffer(rows * dim)?,
            grad_patch_input: buffer(rows * patch_dim)?,
            grad_patches: buffer(rows * patch_dim)?,
            grad_pixels: buffer(channels * side * side)?,
            position_kernel: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_gemma4_positions.spv"),
                4,
                std::mem::size_of::<PositionsPush>() as u32,
            )?,
            position_grad_kernel: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_gemma4_position_grad.spv"),
                3,
                std::mem::size_of::<PositionsPush>() as u32,
            )?,
            rope_kernel: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_gemma4_rope.spv"),
                4,
                std::mem::size_of::<RopePush>() as u32,
            )?,
            avg_pool: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_gemma4_pool.spv"),
                2,
                std::mem::size_of::<PoolPush>() as u32,
            )?,
            cpu_attention: if aten_vector_width() == 8
                && rows == 36
                && dim == 16
                && heads == 4
                && head_dim == 4
            {
                Some((
                    vulkan::ComputeKernel::new(
                        &device,
                        include_bytes!("../../shaders/vision_gemma4_attention_cpu_avx2.spv"),
                        4,
                        16,
                    )?,
                    vulkan::ComputeKernel::new(
                        &device,
                        include_bytes!(
                            "../../shaders/vision_gemma4_attention_cpu_avx2_backward.spv"
                        ),
                        7,
                        16,
                    )?,
                ))
            } else {
                None
            },
            cpu_weight_grad,
            scale_kernel: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_gemma4_scale.spv"),
                2,
                std::mem::size_of::<ScalePush>() as u32,
            )?,
            device,
            language,
            vk,
            image_count,
            side,
            patch_size,
            channels,
            dim,
            intermediate,
            heads: heads as u32,
            head_dim,
            grid,
            kernel,
            tokens_per_image,
            rows,
            patch_dim,
            image_token,
            eps,
            position_size,
            text_dim,
            processor: directory
                .join("preprocessor_config.json")
                .is_file()
                .then(|| {
                    serde_json::from_slice(
                        &fs::read(directory.join("preprocessor_config.json")).ok()?,
                    )
                    .ok()
                })
                .flatten(),
            activation,
            vision_prefix,
            projector_prefix,
        })
    }

    fn record_boundary_weight_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        linear: &VulkanLinear,
        input: &GpuBuffer,
        gradient: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        if self.cpu_weight_grad {
            return Ok(());
        }
        let push = [
            rows as u32,
            linear.input_dim as u32,
            linear.output_dim as u32,
        ];
        self.vk.weight_grad_cpu_avx2.record_dispatch(
            commands,
            &[input, gradient, linear.weight.grad_target()?],
            bytemuck::cast_slice(&push),
            [
                div_ceil_u32(linear.input_dim, 16),
                div_ceil_u32(linear.output_dim, 16),
                1,
            ],
        )?;
        Ok(())
    }

    fn embedding_scale(&self) -> f32 {
        self.language.config.token_embedding_scale()
    }

    fn attention_push(&self) -> AttentionPush {
        let mut push = AttentionPush::zeroed();
        push.batch_size = self.image_count as u32;
        push.query_seq_len = self.rows as u32;
        push.key_value_seq_len = self.rows as u32;
        push.query_hidden_size = self.dim as u32;
        push.value_hidden_size = self.dim as u32;
        push.num_heads = self.heads;
        push.num_key_value_heads = self.heads;
        push.scale = 1.0;
        push.value_scale = 1.0;
        push.causal = 0;
        push.unfused_qk = 9;
        push.kv_page_size = 1;
        push.kv_page_table_stride = 1;
        push
    }

    pub fn positions(&self) -> Vec<u32> {
        self.position_values.clone()
    }

    pub fn text_config_summary(&self) -> String {
        format!("{:?}", self.language.config)
    }

    /// Text-only diagnostic: no image rows or embeddings are injected.
    pub fn text_logits(&self, input_ids: &[u32], attention_mask: &[f32]) -> Result<Vec<f32>> {
        self.language.forward_logits(input_ids, attention_mask)
    }

    /// Raw patch vectors in the checkpoint/processor order `(kh, kw, c)`.
    pub fn patches(&self) -> Result<Vec<f32>> {
        self.patches.read_f32(self.rows * self.patch_dim)
    }

    pub fn preprocess_rgb8(&self, rgb: &[u8]) -> Result<Vec<f32>> {
        let p = self
            .processor
            .as_ref()
            .context("raw RGB input requires preprocessor_config.json")?;
        if p["do_rescale"].as_bool() != Some(true)
            || p["do_normalize"].as_bool() != Some(false)
            || p["do_resize"].as_bool() == Some(true)
        {
            bail!("native Gemma4 RGB processor requires fixed-size rescale without normalize");
        }
        if rgb.len() != self.channels * self.side * self.side {
            bail!("RGB8 input must match the fixed Gemma4 image dimensions");
        }
        let rescale = p["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0);
        if !rescale.is_finite() || rescale <= 0.0 {
            bail!("Gemma4 RGB rescale_factor must be finite and positive");
        }
        if p["patch_size"]
            .as_u64()
            .is_some_and(|v| v as usize != self.patch_size)
            || p["pooling_kernel_size"]
                .as_u64()
                .is_some_and(|v| v as usize != self.kernel)
        {
            bail!("Gemma4 processor patch/pooling geometry differs from the model");
        }
        let area = self.side * self.side;
        let mut values = vec![0.0; rgb.len()];
        for image in 0..self.image_count {
            for pixel in 0..area {
                for channel in 0..self.channels {
                    values[(image * self.channels + channel) * area + pixel] =
                        (rgb[(image * area + pixel) * self.channels + channel] as f64 * rescale)
                            as f32;
                }
            }
        }
        Ok(values)
    }

    fn validate(&self, ids: &[u32], mask: &[f32], pixels: &[f32]) -> Result<Vec<usize>> {
        if ids.len() != self.language.rows || mask.len() != ids.len() {
            bail!("invalid Gemma4 text batch geometry");
        }
        if mask.iter().any(|&v| v != 0.0 && v != 1.0) {
            bail!("Gemma4 attention mask must be binary");
        }
        if ids
            .iter()
            .any(|&id| id as usize >= self.language.config.vocab_size)
        {
            bail!("Gemma4 token outside vocabulary");
        }
        if pixels.len() != self.image_count * self.channels * self.side * self.side
            || pixels.iter().any(|v| !v.is_finite())
        {
            bail!("Gemma4 pixel values must be finite and match the fixed NCHW image dimensions");
        }
        let rows = ids
            .iter()
            .enumerate()
            .filter_map(|(index, &id)| (id == self.image_token).then_some(index))
            .collect::<Vec<_>>();
        if rows.len() != self.image_count * self.tokens_per_image
            || rows.iter().any(|&index| mask[index] != 1.0)
        {
            bail!("image placeholders must match every projected token and be visible");
        }
        Ok(rows)
    }

    fn encode(&self, pixels: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let k = &self.language.kernels;
        if pixels.len() != self.image_count * self.channels * self.side * self.side
            || pixels.iter().any(|v| !v.is_finite())
        {
            bail!("Gemma4 encoder requires finite fixed-size NCHW pixel values");
        }
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.pixels, pixels)?;
        let patch = PatchPush::new(
            self.image_count as u32,
            self.side as u32,
            self.patch_size as u32,
            self.channels as u32,
            0,
            1,
        );
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.pixels, &self.patches],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixels.len(), 256), 1, 1],
        )?;
        self.scale_kernel.record_dispatch(
            &mut commands,
            &[&self.patches, &self.patch_input],
            bytemuck::bytes_of(&ScalePush {
                len: (self.rows * self.patch_dim) as u32,
                scale: 2.0,
                offset: -0.5,
                inverse: 0,
            }),
            [div_ceil_u32(self.rows * self.patch_dim, 256), 1, 1],
        )?;
        self.layers[0].linear_forward(
            &mut commands,
            k,
            &self.vk,
            &self.patch,
            &self.patch_input,
            &self.patch_output,
            self.rows,
        )?;
        self.position_kernel.record_dispatch(
            &mut commands,
            &[
                &self.patch_output,
                &self.position.values,
                &self.positions,
                &self.tokens,
            ],
            bytemuck::bytes_of(&PositionsPush {
                rows: self.rows as u32,
                dim: self.dim as u32,
                position_size: self.position_size as u32,
            }),
            [div_ceil_u32(self.rows * self.dim, 256), 1, 1],
        )?;
        let push = self.attention_push();
        let mut hidden = &self.tokens;
        for layer in &self.layers {
            layer.forward(
                &mut commands,
                k,
                &self.vk,
                hidden,
                &self.cos,
                &self.sin,
                &self.mask,
                &self.dummy,
                push,
                &self.activation,
                self.rows,
                self.dim,
                self.heads as usize,
                self.head_dim,
                &self.rope_kernel,
                self.cpu_attention.as_ref().map(|(forward, _)| forward),
            )?;
            hidden = &layer.tape[OUT];
        }
        self.avg_pool.record_dispatch(
            &mut commands,
            &[hidden, &self.pooled],
            bytemuck::bytes_of(&PoolPush {
                images: self.image_count as u32,
                grid: self.grid as u32,
                kernel: self.kernel as u32,
                dim: self.dim as u32,
                mode: 0,
            }),
            [div_ceil_u32(self.tokens_per_image * self.dim, 256), 1, 1],
        )?;
        self.scale_kernel.record_dispatch(
            &mut commands,
            &[&self.pooled, &self.pooled_scaled],
            bytemuck::bytes_of(&ScalePush {
                len: (self.tokens_per_image * self.dim) as u32,
                scale: (self.dim as f32).sqrt(),
                offset: 0.0,
                inverse: 0,
            }),
            [div_ceil_u32(self.tokens_per_image * self.dim, 256), 1, 1],
        )?;
        self.projector_norm.record_forward(
            &mut commands,
            k,
            &self.pooled_scaled,
            &self.pooled_norm,
            &self.projector_mean,
            &self.projector_rstd,
            self.tokens_per_image,
        )?;
        self.projector.record_forward(
            &mut commands,
            k,
            &self.pooled_norm,
            &self.projected,
            self.tokens_per_image,
        )?;
        commands.submit()?;
        Ok((
            self.pooled_scaled
                .read_f32(self.tokens_per_image * self.dim)?,
            self.projected
                .read_f32(self.tokens_per_image * self.text_dim)?,
        ))
    }

    /// Read the image graph's ordinary forward boundaries for CPU localization.
    pub fn stage_dump(&self, pixels: &[f32]) -> Result<serde_json::Value> {
        self.encode(pixels)?;
        let mut stages = serde_json::Map::new();
        for (name, buffer, len) in [
            ("patches", &self.patches, self.rows * self.patch_dim),
            ("patch_input", &self.patch_input, self.rows * self.patch_dim),
            ("patch_output", &self.patch_output, self.rows * self.dim),
            ("tokens", &self.tokens, self.rows * self.dim),
            ("pooled", &self.pooled, self.tokens_per_image * self.dim),
            (
                "vision_features",
                &self.pooled_scaled,
                self.tokens_per_image * self.dim,
            ),
            (
                "pooled_norm",
                &self.pooled_norm,
                self.tokens_per_image * self.dim,
            ),
            (
                "projected",
                &self.projected,
                self.tokens_per_image * self.text_dim,
            ),
        ] {
            stages.insert(name.into(), serde_json::json!(buffer.read_f32(len)?));
        }
        for (i, l) in self.layers.iter().enumerate() {
            for (name, index) in [
                ("n1", N1),
                ("q", Q),
                ("qn", QN),
                ("qr", QR),
                ("k", K),
                ("kn", KN),
                ("kr", KR),
                ("v", V),
                ("vn", VN),
                ("a", A),
                ("o", O),
                ("n2", N2),
                ("r", R),
                ("n3", N3),
                ("g", G),
                ("ga", GA),
                ("u", U),
                ("p", P),
                ("d", D),
                ("n4", N4),
                ("out", OUT),
            ] {
                let width = if [G, GA, U, P].contains(&index) {
                    self.intermediate
                } else {
                    self.dim
                };
                stages.insert(
                    format!("layer_{i}/{name}"),
                    serde_json::json!(l.tape[index].read_f32(self.rows * width)?),
                );
            }
        }
        Ok(serde_json::Value::Object(stages))
    }
    /// Read ordinary decoder boundaries after the composed forward.
    pub fn text_stage_dump(
        &self,
        ids: &[u32],
        mask: &[f32],
        pixels: &[f32],
    ) -> Result<serde_json::Value> {
        let forward = self.forward(ids, mask, pixels)?;
        let config = &self.language.config;
        let (rows, dim, qdim, kdim, intermediate) = (
            ids.len(),
            config.hidden_size,
            config.query_hidden_size(),
            config.key_value_hidden_size(),
            config.intermediate_size,
        );
        let mut stages = serde_json::Map::new();
        stages.insert(
            "input".into(),
            serde_json::json!(self.language.input_hidden.read_f32(rows * dim)?),
        );
        for (i, l) in self.language.layers.iter().enumerate() {
            let t = &l.tape;
            for (name, buffer, width) in [
                ("n1", &t.ln1, dim),
                ("q", &t.q, qdim),
                ("k", &t.k, kdim),
                ("v", &t.v, kdim),
                ("qn", &t.q_norm, qdim),
                ("kn", &t.k_norm, kdim),
                ("vn", &t.v_norm, kdim),
                ("qr", &t.q_rotary, qdim),
                ("kr", &t.k_rotary, kdim),
                ("a", &t.attention, qdim),
                ("o", &t.attention_projection, dim),
                ("n2", &t.post_attention_norm, dim),
                ("r", &t.residual1, dim),
                ("n3", &t.ln2, dim),
                ("g", &t.mlp_gate_pre, intermediate),
                ("u", &t.mlp_pre, intermediate),
                ("p", &t.mlp_product, intermediate),
                ("d", &t.mlp_output, dim),
                ("n4", &t.post_mlp_norm, dim),
                ("out", &t.output, dim),
            ] {
                stages.insert(
                    format!("layer_{i}/{name}"),
                    serde_json::json!(buffer.read_f32(rows * width)?),
                );
            }
        }
        stages.insert(
            "final_norm".into(),
            serde_json::json!(self.language.final_norm_output.read_f32(rows * dim)?),
        );
        Ok(serde_json::json!({"stages":stages,"logits":forward.logits}))
    }
    /// Compose preprocessed NCHW pixel values with the text graph. Use
    /// `preprocess_rgb8` for processor-compatible RGB8 rescaling.
    pub fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixels: &[f32],
    ) -> Result<Gemma4Forward> {
        let rows = self.validate(input_ids, attention_mask, pixels)?;
        let (vision_features, projected_features) = self.encode(pixels)?;
        let scale = self.embedding_scale();
        let injected = projected_features
            .iter()
            .map(|value| value / scale)
            .collect::<Vec<_>>();
        let logits = self.language.forward_logits_with_modality_embeddings(
            input_ids,
            attention_mask,
            ModalityEmbeddingRows {
                rows: &rows,
                values: &injected,
            },
        )?;
        Ok(Gemma4Forward {
            logits,
            vision_features,
            projected_features,
        })
    }

    pub fn train_step_with_optimizer(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixels: &[f32],
        targets: &[u32],
        loss_weights: &[f32],
        optimizer: crate::CpuAdamWHyperParams,
    ) -> Result<Gemma4TrainResult> {
        optimizer.validate()?;
        let hyper = optimizer.as_f32();
        self.language.set_cpu_adamw_policy(Some(optimizer))?;
        hyper.validate()?;
        let rows = self.validate(input_ids, attention_mask, pixels)?;
        if targets.len() != input_ids.len()
            || loss_weights.len() != input_ids.len()
            || targets
                .iter()
                .zip(loss_weights)
                .any(|(&id, &weight)| id == self.image_token && weight > 0.0)
        {
            bail!("Gemma4 targets must match text rows and exclude image-token supervision");
        }
        let (_, projected) = self.encode(pixels)?;
        let scale = self.embedding_scale();
        let injected = projected
            .iter()
            .map(|value| value / scale)
            .collect::<Vec<_>>();
        let result = self.language.train_step_with_modality_embeddings(
            input_ids,
            targets,
            attention_mask,
            loss_weights,
            hyper,
            ModalityEmbeddingRows {
                rows: &rows,
                values: &injected,
            },
        )?;
        let step = result.text.step;
        let k = &self.language.kernels;
        let projected_gradient = result
            .modality_embedding_grad
            .iter()
            .map(|value| value / scale)
            .collect::<Vec<_>>();
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.grad_projected, &projected_gradient)?;
        self.projector.record_backward(
            &mut commands,
            k,
            &self.pooled_norm,
            &self.grad_projected,
            &self.grad_pooled_norm,
            self.tokens_per_image,
        )?;
        self.record_boundary_weight_grad(
            &mut commands,
            &self.projector,
            &self.pooled_norm,
            &self.grad_projected,
            self.tokens_per_image,
        )?;
        self.projector_norm.record_backward(
            &mut commands,
            k,
            &self.pooled_scaled,
            &self.grad_pooled_norm,
            &self.projector_mean,
            &self.projector_rstd,
            &self.grad_pooled,
            self.tokens_per_image,
        )?;
        self.scale_kernel.record_dispatch(
            &mut commands,
            &[&self.grad_pooled, &self.grad_pooled_scaled],
            bytemuck::bytes_of(&ScalePush {
                len: (self.tokens_per_image * self.dim) as u32,
                scale: (self.dim as f32).sqrt(),
                offset: 0.0,
                inverse: 1,
            }),
            [div_ceil_u32(self.tokens_per_image * self.dim, 256), 1, 1],
        )?;
        self.avg_pool.record_dispatch(
            &mut commands,
            &[&self.grad_pooled_scaled, &self.grad_tokens],
            bytemuck::bytes_of(&PoolPush {
                images: self.image_count as u32,
                grid: self.grid as u32,
                kernel: self.kernel as u32,
                dim: self.dim as u32,
                mode: 1,
            }),
            [div_ceil_u32(self.rows * self.dim, 256), 1, 1],
        )?;
        let push = self.attention_push();
        let mut grad = &self.grad_tokens;
        for index in (0..self.layers.len()).rev() {
            let input = if index == 0 {
                &self.tokens
            } else {
                &self.layers[index - 1].tape[OUT]
            };
            let layer = &self.layers[index];
            layer.backward(
                &mut commands,
                k,
                &self.vk,
                input,
                grad,
                &self.cos,
                &self.sin,
                &self.mask,
                &self.dummy,
                push,
                &self.activation,
                self.rows,
                self.dim,
                self.heads as usize,
                self.head_dim,
                &self.rope_kernel,
                self.cpu_attention.as_ref().map(|(_, backward)| backward),
            )?;
            grad = &layer.tape[DIN];
        }
        self.position_grad_kernel.record_dispatch(
            &mut commands,
            &[grad, &self.positions, self.position.grad_buffer()?],
            bytemuck::bytes_of(&PositionsPush {
                rows: self.rows as u32,
                dim: self.dim as u32,
                position_size: self.position_size as u32,
            }),
            [
                div_ceil_u32(self.dim, 256),
                2 * self.position_size as u32,
                1,
            ],
        )?;
        self.layers[0].linear_backward(
            &mut commands,
            k,
            &self.vk,
            &self.patch,
            &self.patch_input,
            grad,
            &self.grad_patch_input,
            self.rows,
        )?;
        self.record_boundary_weight_grad(
            &mut commands,
            &self.patch,
            &self.patch_input,
            grad,
            self.rows,
        )?;
        self.scale_kernel.record_dispatch(
            &mut commands,
            &[&self.grad_patch_input, &self.grad_patches],
            bytemuck::bytes_of(&ScalePush {
                len: (self.rows * self.patch_dim) as u32,
                scale: 2.0,
                offset: 0.0,
                inverse: 1,
            }),
            [div_ceil_u32(self.rows * self.patch_dim, 256), 1, 1],
        )?;
        let patch = PatchPush::new(
            self.image_count as u32,
            self.side as u32,
            self.patch_size as u32,
            self.channels as u32,
            1,
            1,
        );
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.grad_patches, &self.grad_pixels],
            bytemuck::bytes_of(&patch),
            [
                div_ceil_u32(self.channels * self.side * self.side, 256),
                1,
                1,
            ],
        )?;
        self.projector.record_step(&mut commands, k, step, hyper)?;
        for layer in &self.layers {
            layer.step(&mut commands, k, step, hyper)?;
        }
        self.patch.record_step(&mut commands, k, step, hyper)?;
        self.position.record_step(&mut commands, k, step, hyper)?;
        commands.submit()?;
        Ok(Gemma4TrainResult {
            text: result.text,
            patch_gradient: self.grad_patches.read_f32(self.rows * self.patch_dim)?,
            projected_gradient,
        })
    }

    pub fn export_hf_package(
        &self,
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
    ) -> Result<()> {
        self.language
            .export_hf_package(source.as_ref(), destination.as_ref())?;
        if source.as_ref().join("preprocessor_config.json").is_file() {
            fs::copy(
                source.as_ref().join("preprocessor_config.json"),
                destination.as_ref().join("preprocessor_config.json"),
            )?;
        }
        let mut values = BTreeMap::new();
        values.insert(
            format!("{}.patch_embedder.input_proj.weight", self.vision_prefix),
            (vec![self.dim, self.patch_dim], self.patch.weight.read()?),
        );
        values.insert(
            format!(
                "{}.patch_embedder.position_embedding_table",
                self.vision_prefix
            ),
            (vec![2, self.position_size, self.dim], self.position.read()?),
        );
        for layer in &self.layers {
            layer.export(&mut values)?;
        }
        values.insert(
            format!("{}.embedding_projection.weight", self.projector_prefix),
            (vec![self.text_dim, self.dim], self.projector.weight.read()?),
        );
        let replacements = values
            .iter()
            .map(|(name, (shape, values))| (name.as_str(), shape.as_slice(), values.as_slice()))
            .collect::<Vec<_>>();
        let path = destination.as_ref().join("model.safetensors");
        crate::replace_f32_tensors(&path, &path, &replacements)
    }
}
