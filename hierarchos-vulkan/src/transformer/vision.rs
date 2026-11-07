//! Native CLIP -> LLaVA projector -> language-model training. Model arithmetic
//! stays on Vulkan; the host validates shapes and transfers graph boundaries.
use super::*;
use crate::weight_offload::ModelPackage;

#[cfg(test)]
#[path = "vision_cpu_gemm_tests.rs"]
mod cpu_gemm_tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PatchLinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
    channels: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PatchPush {
    images: u32,
    side: u32,
    patch: u32,
    channels: u32,
    inverse: u32,
    /// 0 = checkpoint (channel, kh, kw), 1 = convolution (kh, kw, channel).
    layout: u32,
}
impl PatchPush {
    pub(super) fn new(
        images: u32,
        side: u32,
        patch: u32,
        channels: u32,
        inverse: u32,
        layout: u32,
    ) -> Self {
        Self {
            images,
            side,
            patch,
            channels,
            inverse,
            layout,
        }
    }
}

/// Reorder patch-projection columns between the checkpoint's
/// (channel, kh, kw) reduction and the (kh, kw, channel) order oneDNN's AVX2
/// direct convolution accumulates. Bias-free and biased `Conv2d` patch
/// embeddings both follow it; the tower's serial-K GEMM then reproduces the
/// convolution's rounding exactly. `to_convolution` selects the direction; the
/// export path reverses it so the HF package stays bit-identical.
pub(super) fn permute_patch_weights(
    values: &[f32],
    output: usize,
    channels: usize,
    patch: usize,
    to_convolution: bool,
) -> Vec<f32> {
    let k = channels * patch * patch;
    let mut permuted = vec![0.0; values.len()];
    for o in 0..output {
        for c in 0..channels {
            for kh in 0..patch {
                for kw in 0..patch {
                    let checkpoint = o * k + (c * patch + kh) * patch + kw;
                    let convolution = o * k + (kh * patch + kw) * channels + c;
                    if to_convolution {
                        permuted[convolution] = values[checkpoint];
                    } else {
                        permuted[checkpoint] = values[convolution];
                    }
                }
            }
        }
    }
    permuted
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct TokenPush {
    images: u32,
    seq: u32,
    dim: u32,
    mode: u32,
}
impl TokenPush {
    pub(super) fn new(images: u32, seq: u32, dim: u32, mode: u32) -> Self {
        Self {
            images,
            seq,
            dim,
            mode,
        }
    }
}

pub(super) fn tensor(package: &mut ModelPackage, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let (actual, values) = package.read_f32(name)?;
    if actual != shape {
        bail!("{name}: expected {shape:?}, found {actual:?}");
    }
    Ok(values)
}
pub(super) fn linear(
    package: &mut ModelPackage,
    device: &VulkanDevice,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<VulkanLinear> {
    let w = tensor(package, &format!("{prefix}.weight"), &[output, input])?;
    if bias {
        VulkanLinear::new(
            device,
            input,
            output,
            &w,
            &tensor(package, &format!("{prefix}.bias"), &[output])?,
        )
    } else {
        VulkanLinear::new_no_bias(device, input, output, &w)
    }
}
pub(super) fn norm(
    package: &mut ModelPackage,
    device: &VulkanDevice,
    prefix: &str,
    dim: usize,
    eps: f32,
) -> Result<VulkanLayerNorm> {
    VulkanLayerNorm::new(
        device,
        dim,
        eps,
        &tensor(package, &format!("{prefix}.weight"), &[dim])?,
        &tensor(package, &format!("{prefix}.bias"), &[dim])?,
    )
}
pub(super) fn export_linear(
    values: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    prefix: &str,
    linear: &VulkanLinear,
) -> Result<()> {
    values.insert(
        format!("{prefix}.weight"),
        (
            vec![linear.output_dim, linear.input_dim],
            linear.weight.read()?,
        ),
    );
    if linear.has_bias {
        values.insert(
            format!("{prefix}.bias"),
            (vec![linear.output_dim], linear.bias.read()?),
        );
    }
    Ok(())
}
pub(super) fn export_norm(
    values: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    prefix: &str,
    norm: &VulkanLayerNorm,
) -> Result<()> {
    values.insert(
        format!("{prefix}.weight"),
        (vec![norm.dim], norm.weight.read()?),
    );
    values.insert(
        format!("{prefix}.bias"),
        (vec![norm.dim], norm.bias.read()?),
    );
    Ok(())
}

pub(super) struct VisionKernels {
    pub(super) patch: vulkan::ComputeKernel,
    tokens: vulkan::ComputeKernel,
    quick_forward: vulkan::ComputeKernel,
    quick_backward: vulkan::ComputeKernel,
    tanh_high_accuracy_forward: vulkan::ComputeKernel,
    tanh_high_accuracy_backward: vulkan::ComputeKernel,
    /// The vision softmax denominator uses the correctly rounded reciprocal; the
    /// text graph's default kernel takes the hardware approximation, which is
    /// several hundred nano-units off across a two-layer tower.
    pub(super) attention_forward: vulkan::ComputeKernel,
    pub(super) attention_backward: vulkan::ComputeKernel,
    pub(super) cpu_avx2_linear: vulkan::ComputeKernel,
    pub(super) patch_linear: vulkan::ComputeKernel,
    pub(super) gelu_cpu_forward: vulkan::ComputeKernel,
    pub(super) gelu_cpu_backward: vulkan::ComputeKernel,
    cpu_avx2_attention: vulkan::ComputeKernel,
    cpu_avx2_attention_seq5: vulkan::ComputeKernel,
    cpu_avx2_attention_seq5_backward: vulkan::ComputeKernel,
    pub(super) cpu_avx2_input_grad: vulkan::ComputeKernel,
    pub(super) weight_grad_cpu_avx2: vulkan::ComputeKernel,
    /// RDNA3-only hardware-FMA schedules of the exact CPU-arithmetic vision
    /// kernels. Every other adapter leaves these unset and keeps the committed
    /// modules, so the Gen9 path is untouched.
    rdna3_cpu_linear: Option<vulkan::ComputeKernel>,
    rdna3_patch_linear: Option<vulkan::ComputeKernel>,
    rdna3_cpu_input_grad: Option<vulkan::ComputeKernel>,
    rdna3_weight_grad: Option<vulkan::ComputeKernel>,
    rdna3_tanh_backward: Option<vulkan::ComputeKernel>,
    rdna3_avx2_siglip_backward: Option<vulkan::ComputeKernel>,
    rdna3_avx512_gelu: Option<[vulkan::ComputeKernel; 2]>,
    rdna3_avx512_weight_grad: Option<vulkan::ComputeKernel>,
    rdna3_avx512_siglip_attention: Option<vulkan::ComputeKernel>,
    rdna3_avx512_siglip_backward: Option<vulkan::ComputeKernel>,
    rdna3_avx512_clip_backward: Option<vulkan::ComputeKernel>,
}
impl VisionKernels {
    pub(super) fn new(device: &VulkanDevice) -> Result<Self> {
        let rdna3 = aten_vector_width() == 8 && crate::vendor::rdna3_vision_variants(device);
        let rdna3_kernel = |spirv: &'static [u8],
                            bindings: usize,
                            push: u32|
         -> Result<Option<vulkan::ComputeKernel>> {
            if !rdna3 {
                return Ok(None);
            }
            Ok(Some(vulkan::ComputeKernel::new(
                device, spirv, bindings, push,
            )?))
        };
        Ok(Self {
            rdna3_avx2_siglip_backward: if crate::vendor::rdna3_frozen_avx2_variants(
                device,
                aten_vector_width(),
            ) {
                Some(vulkan::ComputeKernel::new(
                    device,
                    include_bytes!("../../shaders/vision_siglip_backward_avx2_amd_rdna3.spv"),
                    9,
                    36,
                )?)
            } else {
                None
            },
            // This rounding repair was qualified with the frozen AVX2 image
            // track. Preserve the native AVX512 graph's established staging.
            rdna3_tanh_backward: if crate::vendor::rdna3_frozen_avx2_variants(
                device,
                aten_vector_width(),
            ) {
                Some(vulkan::ComputeKernel::new(
                    device,
                    include_bytes!("../../shaders/vision_gelu_tanh_backward_amd_rdna3.spv"),
                    3,
                    4,
                )?)
            } else {
                None
            },
            rdna3_avx512_clip_backward: if crate::vendor::rdna3_avx512_variants(
                device,
                aten_vector_width(),
            ) {
                Some(vulkan::ComputeKernel::new(
                    device,
                    include_bytes!("../../shaders/image_text_clip_backward_avx512_amd_rdna3.spv"),
                    9,
                    36,
                )?)
            } else {
                None
            },
            rdna3_avx512_siglip_backward: if crate::vendor::rdna3_avx512_variants(
                device,
                aten_vector_width(),
            ) {
                Some(vulkan::ComputeKernel::new(
                    device,
                    include_bytes!("../../shaders/image_text_eager_backward_avx512_amd_rdna3.spv"),
                    9,
                    36,
                )?)
            } else {
                None
            },
            rdna3_avx512_siglip_attention: if crate::vendor::rdna3_avx512_variants(
                device,
                aten_vector_width(),
            ) {
                Some(vulkan::ComputeKernel::new(
                    device,
                    include_bytes!("../../shaders/vision_attention_avx512_amd_rdna3.spv"),
                    4,
                    16,
                )?)
            } else {
                None
            },
            rdna3_avx512_weight_grad: if crate::vendor::rdna3_avx512_variants(
                device,
                aten_vector_width(),
            ) {
                Some(vulkan::ComputeKernel::new(
                    device,
                    include_bytes!("../../shaders/vision_weight_grad_avx512_amd_rdna3.spv"),
                    3,
                    12,
                )?)
            } else {
                None
            },
            rdna3_avx512_gelu: if crate::vendor::rdna3_avx512_variants(device, aten_vector_width())
            {
                Some([
                    vulkan::ComputeKernel::new(
                        device,
                        include_bytes!("../../shaders/image_text_gelu_avx512_amd_rdna3.spv"),
                        2,
                        4,
                    )?,
                    vulkan::ComputeKernel::new(
                        device,
                        include_bytes!(
                            "../../shaders/image_text_gelu_backward_avx512_amd_rdna3.spv"
                        ),
                        3,
                        4,
                    )?,
                ])
            } else {
                None
            },
            cpu_avx2_linear: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
                4,
                16,
            )?,
            patch_linear: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_patch_linear_cpu_avx2.spv"),
                3,
                16,
            )?,
            gelu_cpu_forward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_gelu_erf_cpu_forward.spv"),
                2,
                4,
            )?,
            gelu_cpu_backward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_gelu_erf_cpu_backward.spv"),
                3,
                4,
            )?,
            cpu_avx2_attention: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_attention_cpu_avx2.spv"),
                4,
                16,
            )?,
            cpu_avx2_attention_seq5: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/seq5_attention_cpu_avx2.spv"),
                4,
                20,
            )?,
            cpu_avx2_attention_seq5_backward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/seq5_attention_cpu_avx2_backward.spv"),
                15,
                std::mem::size_of::<AttentionBackwardPush>() as u32,
            )?,
            cpu_avx2_input_grad: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv"),
                4,
                16,
            )?,
            weight_grad_cpu_avx2: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_linear_weight_grad.spv"),
                3,
                12,
            )?,
            rdna3_cpu_linear: rdna3_kernel(
                include_bytes!("../../shaders/vision_linear_hw_fma_amd_rdna3.spv"),
                4,
                16,
            )?,
            rdna3_patch_linear: rdna3_kernel(
                include_bytes!("../../shaders/vision_patch_linear_hw_fma_amd_rdna3.spv"),
                3,
                16,
            )?,
            rdna3_cpu_input_grad: rdna3_kernel(
                include_bytes!("../../shaders/vision_linear_input_grad_hw_fma_amd_rdna3.spv"),
                4,
                16,
            )?,
            rdna3_weight_grad: rdna3_kernel(
                include_bytes!("../../shaders/vision_linear_weight_grad_hw_fma_amd_rdna3.spv"),
                3,
                12,
            )?,
            patch: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_patchify.spv"),
                2,
                24,
            )?,
            tokens: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_clip_tokens.spv"),
                4,
                16,
            )?,
            quick_forward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_quick_gelu.spv"),
                2,
                4,
            )?,
            quick_backward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_quick_gelu_backward.spv"),
                3,
                4,
            )?,
            // torch's gelu_pytorch_tanh dispatches through the same
            // high-accuracy tanh reconstruction the GPT-2 CPU route uses; the
            // default kernel is a faster approximation that leaves the SigLIP
            // tower several hundred nano-units away from the oracle.
            tanh_high_accuracy_forward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/transformer_gelu_tanh_ha_forward.spv"),
                2,
                4,
            )?,
            tanh_high_accuracy_backward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/transformer_gelu_tanh_ha_backward.spv"),
                3,
                4,
            )?,
            attention_forward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/transformer_attention_forward_cpu_reciprocal.spv"),
                11,
                std::mem::size_of::<AttentionPush>() as u32,
            )?,
            attention_backward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/transformer_attention_backward_cpu_reciprocal.spv"),
                15,
                std::mem::size_of::<AttentionBackwardPush>() as u32,
            )?,
        })
    }

    /// The exact vision kernel selected on this adapter. The RDNA3 module
    /// substitutes correctly rounded hardware FMA for the software construction
    /// and keeps every other step, so selection cannot change a result bit.
    fn cpu_linear_kernel(&self) -> &vulkan::ComputeKernel {
        self.rdna3_cpu_linear
            .as_ref()
            .unwrap_or(&self.cpu_avx2_linear)
    }

    fn cpu_input_grad_kernel(&self) -> &vulkan::ComputeKernel {
        self.rdna3_cpu_input_grad
            .as_ref()
            .unwrap_or(&self.cpu_avx2_input_grad)
    }

    pub(super) fn patch_linear_kernel(&self) -> &vulkan::ComputeKernel {
        self.rdna3_patch_linear
            .as_ref()
            .unwrap_or(&self.patch_linear)
    }

    pub(super) fn weight_grad_kernel(&self) -> &vulkan::ComputeKernel {
        self.rdna3_avx512_weight_grad
            .as_ref()
            .or(self.rdna3_weight_grad.as_ref())
            .unwrap_or(&self.weight_grad_cpu_avx2)
    }

    /// Replay only the independently measured AVX2 NN-linear geometries.
    /// Convolution patch projections retain their own reduction order.
    fn cpu_linear(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
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
            self.cpu_linear_kernel().record_dispatch(
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

    fn cpu_input_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        linear: &VulkanLinear,
        gradient: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
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
            self.cpu_input_grad_kernel().record_dispatch(
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

    pub(super) fn activation(
        &self,
        k: &TransformerKernels,
        name: &str,
        commands: &mut vulkan::ComputeBatch,
        x: &GpuBuffer,
        gradient: Option<&GpuBuffer>,
        output: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        let kernel = match (name, gradient.is_some()) {
            ("quick_gelu", false) => &self.quick_forward,
            ("quick_gelu", true) => &self.quick_backward,
            ("gelu", false) => &k.gelu_erf_forward,
            ("gelu", true) => &k.gelu_erf_backward,
            ("gelu_pytorch_tanh" | "gelu_new", false) => &k.gelu_tanh_forward,
            ("gelu_pytorch_tanh" | "gelu_new", true) => self
                .rdna3_tanh_backward
                .as_ref()
                .unwrap_or(&k.gelu_tanh_backward),
            ("gelu_pytorch_tanh_ha", false) => &self.tanh_high_accuracy_forward,
            ("gelu_pytorch_tanh_ha", true) => &self.tanh_high_accuracy_backward,
            _ => bail!("unsupported native vision activation {name}"),
        };
        let bindings = if let Some(g) = gradient {
            vec![x, g, output]
        } else {
            vec![x, output]
        };
        kernel.record_dispatch(
            commands,
            &bindings,
            bytemuck::bytes_of(&LenPush { len: len as u32 }),
            [div_ceil_u32(len, 256), 1, 1],
        )
    }
}

const N1: usize = 0;
const Q: usize = 1;
const K: usize = 2;
const V: usize = 3;
const A: usize = 4;
const O: usize = 5;
const R: usize = 6;
const N2: usize = 7;
const F: usize = 8;
const ACT: usize = 9;
pub(super) const OUT: usize = 10;
const M1: usize = 11;
const S1: usize = 12;
const M2: usize = 13;
const S2: usize = 14;
const DACT: usize = 15;
const DF: usize = 16;
const DN2: usize = 17;
const DR: usize = 18;
const DA: usize = 19;
const DQ: usize = 20;
const DK: usize = 21;
const DV: usize = 22;
const DQI: usize = 23;
const DKI: usize = 24;
const DVI: usize = 25;
const DN1: usize = 26;
pub(super) const DI: usize = 27;
const TMP: usize = 28;

pub(super) fn cpu_avx2_vision_gemm(
    rows: usize,
    input: usize,
    output: usize,
    vector_width: u32,
) -> bool {
    vector_width == 8
        && (2..=36).contains(&rows)
        && (matches!(input, 8 | 12 | 16 | 32) || (rows == 8 && input == 64 && output == 64))
        && matches!(output, 16 | 32 | 64)
}

pub(super) fn cpu_avx2_input_gradient_gemm(
    rows: usize,
    input: usize,
    output: usize,
    vector_width: u32,
) -> bool {
    cpu_avx2_vision_gemm(rows, input, output, vector_width)
        || (vector_width == 8
            && (2..=36).contains(&rows)
            && input == 64
            && matches!(output, 16 | 32))
}

pub(super) struct ClipLayer {
    cpu_reference_math: bool,
    reverse_qkv_adjoint: bool,
    q: VulkanLinear,
    key: VulkanLinear,
    v: VulkanLinear,
    out: VulkanLinear,
    fc1: VulkanLinear,
    fc2: VulkanLinear,
    ln1: VulkanLayerNorm,
    ln2: VulkanLayerNorm,
    pub(super) tape: Vec<GpuBuffer>,
    prefix: String,
}
impl ClipLayer {
    pub(super) fn load(
        package: &mut ModelPackage,
        device: &VulkanDevice,
        prefix: String,
        rows: usize,
        dim: usize,
        intermediate: usize,
        eps: f32,
        cpu_reference_math: bool,
        reverse_qkv_adjoint: bool,
    ) -> Result<Self> {
        let sizes = (0..29).map(|index| match index {
            F | ACT | DACT | DF => rows * intermediate,
            M1 | S1 | M2 | S2 => rows,
            _ => rows * dim,
        });
        Ok(Self {
            cpu_reference_math,
            reverse_qkv_adjoint,
            q: linear(
                package,
                device,
                &format!("{prefix}.self_attn.q_proj"),
                dim,
                dim,
                true,
            )?,
            key: linear(
                package,
                device,
                &format!("{prefix}.self_attn.k_proj"),
                dim,
                dim,
                true,
            )?,
            v: linear(
                package,
                device,
                &format!("{prefix}.self_attn.v_proj"),
                dim,
                dim,
                true,
            )?,
            out: linear(
                package,
                device,
                &format!("{prefix}.self_attn.out_proj"),
                dim,
                dim,
                true,
            )?,
            fc1: linear(
                package,
                device,
                &format!("{prefix}.mlp.fc1"),
                dim,
                intermediate,
                true,
            )?,
            fc2: linear(
                package,
                device,
                &format!("{prefix}.mlp.fc2"),
                intermediate,
                dim,
                true,
            )?,
            ln1: norm(package, device, &format!("{prefix}.layer_norm1"), dim, eps)?
                .use_cpu_welford(cpu_reference_math),
            ln2: norm(package, device, &format!("{prefix}.layer_norm2"), dim, eps)?
                .use_cpu_welford(cpu_reference_math),
            tape: sizes
                .map(|len| GpuBuffer::zeros_f32(device, len))
                .collect::<Result<_>>()?,
            prefix,
        })
    }
    /// Diagnostic boundary dump used by the SigLIP stage diff. Not part of the
    /// qualification path.
    pub(super) fn stage_dump(
        &self,
        rows: usize,
        dim: usize,
        intermediate: usize,
    ) -> Result<Vec<(&'static str, Vec<f32>)>> {
        let mut stages = Vec::new();
        for (name, index, len) in [
            ("ln1", N1, rows * dim),
            ("q", Q, rows * dim),
            ("k", K, rows * dim),
            ("v", V, rows * dim),
            ("attention", A, rows * dim),
            ("out_proj", O, rows * dim),
            ("residual1", R, rows * dim),
            ("ln2", N2, rows * dim),
            ("fc1", F, rows * intermediate),
            ("gte", ACT, rows * intermediate),
            ("ln1_mean", M1, rows),
            ("ln1_rstd", S1, rows),
            ("ln2_mean", M2, rows),
            ("ln2_rstd", S2, rows),
        ] {
            stages.push((name, self.tape[index].read_f32(len)?));
        }
        Ok(stages)
    }

    pub(super) fn backward_stage_dump(
        &self,
        rows: usize,
        dim: usize,
        intermediate: usize,
    ) -> Result<Vec<(&'static str, Vec<f32>)>> {
        let mut stages = self.stage_dump(rows, dim, intermediate)?;
        for (name, index, len) in [
            ("out", OUT, rows * dim),
            ("dactivation", DACT, rows * intermediate),
            ("dfc1", DF, rows * intermediate),
            ("dln2", DN2, rows * dim),
            ("dresidual1", DR, rows * dim),
            ("dattention", DA, rows * dim),
            ("dq", DQ, rows * dim),
            ("dk", DK, rows * dim),
            ("dv", DV, rows * dim),
            ("dqi", DQI, rows * dim),
            ("dki", DKI, rows * dim),
            ("dvi", DVI, rows * dim),
            ("dln1", DN1, rows * dim),
            ("dinput", DI, rows * dim),
        ] {
            stages.push((name, self.tape[index].read_f32(len)?));
        }
        Ok(stages)
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
        if self.cpu_reference_math {
            vk.cpu_linear(commands, k, linear, input, output, rows)
        } else {
            linear.record_forward(commands, k, input, output, rows)
        }
    }

    pub(super) fn forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        x: &GpuBuffer,
        mask: &GpuBuffer,
        dummy: &GpuBuffer,
        push: AttentionPush,
        activation: &str,
    ) -> Result<()> {
        let rows = (push.batch_size * push.query_seq_len) as usize;
        let dim = push.query_hidden_size as usize;
        let t = &self.tape;
        self.ln1
            .record_forward(commands, k, x, &t[N1], &t[M1], &t[S1], rows)?;
        self.linear_forward(commands, k, vk, &self.q, &t[N1], &t[Q], rows)?;
        self.linear_forward(commands, k, vk, &self.key, &t[N1], &t[K], rows)?;
        self.linear_forward(commands, k, vk, &self.v, &t[N1], &t[V], rows)?;
        if let Some(kernel) = vk.rdna3_avx512_siglip_attention.as_ref().filter(|_| {
            self.cpu_reference_math
                && matches!(push.batch_size, 1 | 2)
                && matches!(push.query_seq_len, 4 | 5 | 16 | 17)
                && push.key_value_seq_len == push.query_seq_len
                && push.query_hidden_size == 16
                && push.value_hidden_size == 16
                && push.num_heads == 4
                && push.num_key_value_heads == 4
        }) {
            let cpu_push = [
                push.batch_size,
                push.query_seq_len,
                push.query_hidden_size,
                push.num_heads,
            ];
            let rows = push.batch_size * push.num_heads * push.query_seq_len;
            kernel.record_dispatch(
                commands,
                &[&t[Q], &t[K], &t[V], &t[A]],
                bytemuck::cast_slice(&cpu_push),
                [rows.div_ceil(64), 1, 1],
            )?;
        } else if self.cpu_reference_math
            && aten_vector_width() == 8
            && matches!(push.batch_size, 1 | 2)
            && push.query_seq_len == 5
            && push.key_value_seq_len == 5
            && push.query_hidden_size == 16
            && push.value_hidden_size == 16
            && push.num_heads == 2
        {
            let cpu_push = [
                push.batch_size,
                push.query_seq_len,
                push.query_hidden_size,
                push.num_heads,
                push.scale.to_bits(),
            ];
            vk.cpu_avx2_attention_seq5.record_dispatch(
                commands,
                &[&t[Q], &t[K], &t[V], &t[A]],
                bytemuck::cast_slice(&cpu_push),
                [push.query_seq_len, push.num_heads, push.batch_size],
            )?;
        } else if self.cpu_reference_math
            && aten_vector_width() == 8
            && matches!(push.batch_size, 1 | 2)
            && push.query_seq_len == 16
            && push.key_value_seq_len == 16
            && push.query_hidden_size == 16
            && push.value_hidden_size == 16
            && push.num_heads == 4
        {
            let cpu_push = [
                push.batch_size,
                push.query_seq_len,
                push.query_hidden_size,
                push.num_heads,
            ];
            vk.cpu_avx2_attention.record_dispatch(
                commands,
                &[&t[Q], &t[K], &t[V], &t[A]],
                bytemuck::cast_slice(&cpu_push),
                [push.query_seq_len, push.num_heads, push.batch_size],
            )?;
        } else {
            vk.attention_forward.record_dispatch(
                commands,
                &[
                    &t[Q], &t[K], &t[V], mask, mask, dummy, &t[A], dummy, dummy, mask, mask,
                ],
                bytemuck::bytes_of(&push),
                [push.query_seq_len, push.num_heads, push.batch_size],
            )?;
        }
        self.linear_forward(commands, k, vk, &self.out, &t[A], &t[O], rows)?;
        record_add(commands, k, x, &t[O], &t[R], rows * dim)?;
        self.ln2
            .record_forward(commands, k, &t[R], &t[N2], &t[M2], &t[S2], rows)?;
        self.linear_forward(commands, k, vk, &self.fc1, &t[N2], &t[F], rows)?;
        vk.activation(
            k,
            activation,
            commands,
            &t[F],
            None,
            &t[ACT],
            rows * self.fc1.output_dim,
        )?;
        self.linear_forward(commands, k, vk, &self.fc2, &t[ACT], &t[OUT], rows)?;
        record_add(commands, k, &t[R], &t[OUT], &t[OUT], rows * dim)
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
        // Keep parameter gradients under the shared linear's ownership. Replay
        // only the short-panel input-gradient GEMM before its consumer runs.
        linear.record_backward(commands, k, input, gradient, output, rows)?;
        if self.cpu_reference_math {
            vk.cpu_input_grad(commands, linear, gradient, output, rows)?;
        }
        Ok(())
    }

    pub(super) fn backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        x: &GpuBuffer,
        dy: &GpuBuffer,
        mask: &GpuBuffer,
        dummy: &GpuBuffer,
        push: AttentionPush,
        activation: &str,
    ) -> Result<()> {
        let rows = (push.batch_size * push.query_seq_len) as usize;
        let dim = push.query_hidden_size as usize;
        let t = &self.tape;
        self.linear_backward(commands, k, vk, &self.fc2, &t[ACT], dy, &t[DACT], rows)?;
        vk.activation(
            k,
            activation,
            commands,
            &t[F],
            Some(&t[DACT]),
            &t[DF],
            rows * self.fc1.output_dim,
        )?;
        self.linear_backward(commands, k, vk, &self.fc1, &t[N2], &t[DF], &t[DN2], rows)?;
        self.ln2
            .record_backward(commands, k, &t[R], &t[DN2], &t[M2], &t[S2], &t[DR], rows)?;
        record_add(commands, k, dy, &t[DR], &t[DR], rows * dim)?;
        self.linear_backward(commands, k, vk, &self.out, &t[A], &t[DR], &t[DA], rows)?;
        if let Some(kernel) = vk.rdna3_avx2_siglip_backward.as_ref().filter(|_| {
            self.cpu_reference_math
                && push.batch_size == 1
                && push.query_seq_len == 16
                && push.key_value_seq_len == 16
                && push.query_hidden_size == 16
                && push.value_hidden_size == 16
                && push.num_heads == 4
                && push.num_key_value_heads == 4
        }) {
            let packed = rdna3_eager_push(&push);
            kernel.record_dispatch(
                commands,
                &[
                    &t[Q], &t[K], &t[V], mask, mask, &t[DA], &t[DQ], &t[DK], &t[DV],
                ],
                bytemuck::cast_slice(&packed),
                [4, 1, 1],
            )?;
        } else if let Some(kernel) = vk.rdna3_avx512_clip_backward.as_ref().filter(|_| {
            self.cpu_reference_math
                && k.rdna3_avx512_gpt2.is_some()
                && matches!(push.batch_size, 1..=3)
                && push.query_seq_len == 5
                && push.key_value_seq_len == 5
                && push.query_hidden_size == 16
                && push.value_hidden_size == 16
                && push.num_heads == 2
                && push.num_key_value_heads == 2
        }) {
            let packed = rdna3_eager_push(&push);
            kernel.record_dispatch(
                commands,
                &[
                    &t[Q], &t[K], &t[V], mask, mask, &t[DA], &t[DQ], &t[DK], &t[DV],
                ],
                bytemuck::cast_slice(&packed),
                [2, push.batch_size, 1],
            )?;
        } else if let Some(kernel) = vk.rdna3_avx512_siglip_backward.as_ref().filter(|_| {
            self.cpu_reference_math
                && matches!(push.batch_size, 1 | 2)
                && push.query_seq_len == 16
                && push.key_value_seq_len == 16
                && push.query_hidden_size == 16
                && push.value_hidden_size == 16
                && push.num_heads == 4
                && push.num_key_value_heads == 4
        }) {
            let packed = rdna3_eager_push(&push);
            kernel.record_dispatch(
                commands,
                &[
                    &t[Q], &t[K], &t[V], mask, mask, &t[DA], &t[DQ], &t[DK], &t[DV],
                ],
                bytemuck::cast_slice(&packed),
                [4, push.batch_size, 1],
            )?;
        } else {
            let attention_backward = if self.cpu_reference_math
                && aten_vector_width() == 8
                && push.batch_size == 2
                && push.query_seq_len == 5
                && push.query_hidden_size == 16
                && push.num_heads == 2
            {
                &vk.cpu_avx2_attention_seq5_backward
            } else {
                &vk.attention_backward
            };
            attention_backward.record_dispatch(
                commands,
                &[
                    &t[Q], &t[K], &t[V], mask, mask, dummy, &t[DA], &t[DQ], &t[DK], &t[DV], dummy,
                    dummy, dummy, dummy, mask,
                ],
                bytemuck::bytes_of(&push.as_backward_push(0, 1)),
                [push.num_heads, push.batch_size, 1],
            )?;
        }
        self.linear_backward(commands, k, vk, &self.q, &t[N1], &t[DQ], &t[DQI], rows)?;
        self.linear_backward(commands, k, vk, &self.key, &t[N1], &t[DK], &t[DKI], rows)?;
        self.linear_backward(commands, k, vk, &self.v, &t[N1], &t[DV], &t[DVI], rows)?;
        let (first, last) = if self.reverse_qkv_adjoint
            && (aten_vector_width() == 8 || k.rdna3_avx512_gpt2.is_some())
        {
            (&t[DVI], &t[DQI])
        } else {
            (&t[DQI], &t[DVI])
        };
        record_add(commands, k, first, &t[DKI], &t[TMP], rows * dim)?;
        record_add(commands, k, &t[TMP], last, &t[DN1], rows * dim)?;
        self.ln1
            .record_backward(commands, k, x, &t[DN1], &t[M1], &t[S1], &t[DI], rows)?;
        record_add(commands, k, &t[DR], &t[DI], &t[DI], rows * dim)
    }
    pub(super) fn step(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        step: u32,
        hyper: AdamWHyperParams,
    ) -> Result<()> {
        for linear in [&self.q, &self.key, &self.v, &self.out, &self.fc1, &self.fc2] {
            linear.record_step(commands, k, step, hyper)?;
        }
        self.ln1.record_step(commands, k, step, hyper)?;
        self.ln2.record_step(commands, k, step, hyper)
    }
    pub(super) fn export(
        &self,
        values: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    ) -> Result<()> {
        for (suffix, linear) in [
            ("self_attn.q_proj", &self.q),
            ("self_attn.k_proj", &self.key),
            ("self_attn.v_proj", &self.v),
            ("self_attn.out_proj", &self.out),
            ("mlp.fc1", &self.fc1),
            ("mlp.fc2", &self.fc2),
        ] {
            export_linear(values, &format!("{}.{suffix}", self.prefix), linear)?;
        }
        export_norm(values, &format!("{}.layer_norm1", self.prefix), &self.ln1)?;
        export_norm(values, &format!("{}.layer_norm2", self.prefix), &self.ln2)
    }
}

#[derive(Debug, Serialize)]
pub struct LlavaForward {
    pub logits: Vec<f32>,
    pub vision_features: Vec<f32>,
    pub projector_pre: Vec<f32>,
    pub projector_act: Vec<f32>,
    pub projected_features: Vec<f32>,
}
#[derive(Debug, Serialize)]
pub struct LlavaTrainResult {
    pub text: VulkanTransformerStepResult,
    pub pixel_gradient: Vec<f32>,
    pub projected_gradient: Vec<f32>,
}

/// Fixed-resolution CLIP/LLaVA FP32 graph. Image tensors use NCHW layout.
/// The standard HF package supplies the actual tower and two-layer projector.
/// Dynamic positional interpolation, video, multiple selected feature layers,
/// dropout and non-CLIP towers fail closed pending separate implementations.
pub struct VulkanLlava {
    pub(super) language: VulkanTransformer,
    pub(super) image: VulkanClipImageAdapter,
}

pub(super) struct ClipAdapterGradients {
    pub(super) pixel_gradient: Vec<f32>,
    pub(super) projected_gradient: Vec<f32>,
}

impl VulkanLlava {
    pub fn from_hf_package(
        device: VulkanDevice,
        directory: impl AsRef<Path>,
        batch_size: usize,
        seq_len: usize,
        image_count: usize,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        if config["model_type"].as_str() != Some("llava") {
            bail!("native LLaVA pipeline requires model_type=llava");
        }
        let language =
            VulkanTransformer::from_hf_package(device.clone(), directory, batch_size, seq_len)?;
        if language.config.architecture != VulkanTransformerArchitecture::Llama {
            bail!("native LLaVA pipeline requires its qualified Llama backbone");
        }
        Self::from_language(device, directory, image_count, language)
    }
    pub(super) fn from_language(
        device: VulkanDevice,
        directory: &Path,
        image_count: usize,
        mut language: VulkanTransformer,
    ) -> Result<Self> {
        if language.config.embedding_dropout != 0.0
            || language.config.attention_dropout != 0.0
            || language.config.mlp_output_dropout != 0.0
            || language.config.attention_output_dropout != 0.0
        {
            bail!("native LLaVA training requires deterministic zero-dropout text configuration");
        }
        let image = VulkanClipImageAdapter::load(
            device,
            directory,
            image_count,
            language.rows,
            language.config.vocab_size,
            language.config.effective_embedding_size(),
            &mut language.kernels,
        )?;
        Ok(Self { language, image })
    }

    pub fn preprocess_rgb8(&self, rgb: &[u8]) -> Result<Vec<f32>> {
        self.image.preprocess_rgb8(rgb)
    }
    pub(super) fn validate(&self, ids: &[u32], mask: &[f32], pixels: &[f32]) -> Result<Vec<usize>> {
        self.image.validate(ids, mask, pixels)
    }
    pub(super) fn encode_device(&self, pixels: &[f32]) -> Result<()> {
        self.image.encode_device(pixels, &self.language.kernels)
    }
    pub(super) fn forward_result(&self, logits: Vec<f32>) -> Result<LlavaForward> {
        self.image.forward_result(logits)
    }
    pub fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
    ) -> Result<LlavaForward> {
        let rows = self.validate(input_ids, attention_mask, pixel_values)?;
        self.encode_device(pixel_values)?;
        let logits = self
            .language
            .forward_logits_with_device_modality_embeddings(
                input_ids,
                attention_mask,
                &rows,
                &self.image.projected,
            )?;
        self.forward_result(logits)
    }
    pub fn train_step(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
        targets: &[u32],
        loss_weights: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
        self.train_step_with_optimizer(
            input_ids,
            attention_mask,
            pixel_values,
            targets,
            loss_weights,
            hyper.into(),
        )
    }
    pub fn train_step_with_optimizer(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
        targets: &[u32],
        loss_weights: &[f32],
        optimizer: crate::CpuAdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
        optimizer.validate()?;
        let hyper = optimizer.as_f32();
        self.language.set_cpu_adamw_policy(Some(optimizer))?;
        hyper.validate()?;
        let rows = self.validate(input_ids, attention_mask, pixel_values)?;
        if targets.len() != input_ids.len()
            || loss_weights.len() != input_ids.len()
            || targets
                .iter()
                .zip(loss_weights)
                .any(|(&id, &w)| id == self.image.image_token && w > 0.)
        {
            bail!("LLaVA targets must match text rows and exclude image-token supervision");
        }
        self.encode_device(pixel_values)?;
        let result = self.language.train_step_with_device_modality_embeddings(
            input_ids,
            targets,
            attention_mask,
            loss_weights,
            hyper,
            &rows,
            &self.image.projected,
        )?;
        self.backward_image(result, pixel_values, hyper)
    }
    pub(super) fn backward_image(
        &mut self,
        result: DeviceMultimodalStepResult,
        pixels: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
        let gradient = result
            .gradient
            .as_ref()
            .context("image-text backward omitted its device gradient")?;
        let gradients = self.image.backward_image(
            gradient,
            result.text.step,
            pixels,
            &self.language.kernels,
            hyper,
        )?;
        Ok(LlavaTrainResult {
            text: result.text,
            pixel_gradient: gradients.pixel_gradient,
            projected_gradient: gradients.projected_gradient,
        })
    }
    pub fn export_hf_package(
        &self,
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
    ) -> Result<()> {
        self.language
            .export_hf_package(source.as_ref(), destination.as_ref())?;
        self.export_image_parameters(source.as_ref(), destination.as_ref())
    }
    pub(super) fn debug_image_training_stages(&self) -> Result<serde_json::Value> {
        self.image.debug_image_training_stages()
    }
    pub(super) fn export_image_parameters(&self, source: &Path, destination: &Path) -> Result<()> {
        self.image.export_image_parameters(source, destination)
    }
}

pub(super) struct VulkanClipImageAdapter {
    device: VulkanDevice,
    text_rows: usize,
    vocab_size: usize,
    vk: VisionKernels,
    image_count: usize,
    side: usize,
    patch_size: usize,
    channels: usize,
    dim: usize,
    heads: u32,
    seq: usize,
    features_per_image: usize,
    drop_cls: bool,
    image_token: u32,
    cpu_reference_math: bool,
    activation: String,
    projector_activation: String,
    processor: Option<serde_json::Value>,
    vision_prefix: String,
    projector_prefix: String,
    patch: VulkanLinear,
    cls: TrainableParameter,
    position: TrainableParameter,
    pre_norm: VulkanLayerNorm,
    layers: Vec<ClipLayer>,
    projector1: VulkanLinear,
    projector2: VulkanLinear,
    pixels: GpuBuffer,
    patches: GpuBuffer,
    patch_output: GpuBuffer,
    tokens: GpuBuffer,
    pre_output: GpuBuffer,
    pre_mean: GpuBuffer,
    pre_rstd: GpuBuffer,
    mask: GpuBuffer,
    dummy: GpuBuffer,
    selected: GpuBuffer,
    projector_pre: GpuBuffer,
    projector_act: GpuBuffer,
    pub(super) projected: GpuBuffer,
    grad_projected: GpuBuffer,
    grad_act: GpuBuffer,
    grad_projector_pre: GpuBuffer,
    grad_selected: GpuBuffer,
    grad_vision_output: GpuBuffer,
    grad_tokens: GpuBuffer,
    grad_patch_output: GpuBuffer,
    grad_patches: GpuBuffer,
    grad_pixels: GpuBuffer,
}

impl VulkanClipImageAdapter {
    fn vision_heads(&self) -> u32 {
        self.heads
    }
    pub(super) fn load(
        device: VulkanDevice,
        directory: &Path,
        image_count: usize,
        text_rows: usize,
        vocab_size: usize,
        text_dim: usize,
        kernels: &mut TransformerKernels,
    ) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        if !matches!(
            config["model_type"].as_str(),
            Some("llava" | "hierarchos_clip_adapter")
        ) {
            bail!("image adapter requires llava or hierarchos_clip_adapter metadata");
        }
        let cpu_reference_math = config["model_type"].as_str() == Some("hierarchos_clip_adapter");
        if cpu_reference_math && aten_vector_width() == 8 {
            kernels.cpu_layer_norm_adjoint = true;
            kernels.bias_grad = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_bias_grad_cpu_avx2.spv"),
                2,
                8,
            )?;
        }
        let vision = &config["vision_config"];
        if vision["model_type"].as_str() != Some("clip_vision_model") {
            bail!("native LLaVA currently requires a CLIP vision tower");
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
        let side = number(vision, "image_size")?;
        let patch_size = number(vision, "patch_size")?;
        let channels = number(vision, "num_channels")?;
        let dim = number(vision, "hidden_size")?;
        let intermediate = number(vision, "intermediate_size")?;
        let num_layers = number(vision, "num_hidden_layers")?;
        let heads = number(vision, "num_attention_heads")?;
        if image_count == 0 || !side.is_multiple_of(patch_size) || !dim.is_multiple_of(heads) {
            bail!("invalid image count, patch grid or CLIP head geometry");
        }
        if vision["attention_dropout"].as_f64().unwrap_or(0.0) != 0.0 {
            bail!("native CLIP training requires attention_dropout=0");
        }
        let activation = vision["hidden_act"]
            .as_str()
            .unwrap_or("quick_gelu")
            .to_string();
        let projector_activation = config["projector_hidden_act"]
            .as_str()
            .unwrap_or("gelu")
            .to_string();
        for name in [&activation, &projector_activation] {
            if !matches!(
                name.as_str(),
                "quick_gelu" | "gelu" | "gelu_pytorch_tanh" | "gelu_new"
            ) {
                bail!("unsupported vision/projector activation {name}");
            }
        }
        let layer = config["vision_feature_layer"]
            .as_i64()
            .context("vision_feature_layer must be one integer")?;
        let active = if layer < 0 {
            (num_layers as i64 + 1).checked_add(layer)
        } else {
            Some(layer)
        }
        .context("vision feature layer overflow")?;
        if active < 0 || active > num_layers as i64 {
            bail!("vision feature layer is outside CLIP hidden states");
        }
        let drop_cls = match config["vision_feature_select_strategy"]
            .as_str()
            .unwrap_or("default")
        {
            "default" => true,
            "full" => false,
            _ => bail!("unsupported vision feature selection"),
        };
        let seq = (side / patch_size)
            .checked_pow(2)
            .and_then(|x| x.checked_add(1))
            .context("vision grid overflow")?;
        let features_per_image = seq - usize::from(drop_cls);
        let rows = image_count
            .checked_mul(seq)
            .context("vision row overflow")?;
        let hidden = rows
            .checked_mul(dim)
            .context("vision hidden size overflow")?;
        let pixel_len = image_count
            .checked_mul(channels)
            .and_then(|x| x.checked_mul(side))
            .and_then(|x| x.checked_mul(side))
            .context("pixel size overflow")?;
        for size in [
            hidden,
            pixel_len,
            image_count * features_per_image,
            rows * intermediate,
        ] {
            if size > u32::MAX as usize {
                bail!("CLIP graph exceeds shader indexing range");
            }
        }
        let eps = vision["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        if !eps.is_finite() || eps <= 0.0 {
            bail!("invalid CLIP layer norm epsilon");
        }
        let image_token = u32::try_from(
            config["image_token_index"]
                .as_u64()
                .context("missing image_token_index")?,
        )?;
        if image_token as usize >= vocab_size {
            bail!("image token must be in the text vocabulary");
        }
        let mut package = ModelPackage::open_directory(directory)?;
        let names = package.tensor_names()?;
        let vision_prefix = [
            "model.vision_tower",
            "model.vision_tower.vision_model",
            "vision_tower.vision_model",
            "vision_tower",
        ]
        .into_iter()
        .find(|p| names.contains(&format!("{p}.embeddings.class_embedding")))
        .context("cannot locate the CLIP tower namespace")?
        .to_string();
        let projector_prefix = ["model.multi_modal_projector", "multi_modal_projector"]
            .into_iter()
            .find(|p| names.contains(&format!("{p}.linear_1.weight")))
            .context("cannot locate LLaVA projector")?
            .to_string();
        let patch_input = channels * patch_size * patch_size;
        let patch_weights = tensor(
            &mut package,
            &format!("{vision_prefix}.embeddings.patch_embedding.weight"),
            &[dim, channels, patch_size, patch_size],
        )?;
        let bias = config["multimodal_projector_bias"]
            .as_bool()
            .unwrap_or(true);
        let selected_rows = image_count * features_per_image;
        let buffer = |len| GpuBuffer::zeros_f32(&device, len);
        Ok(Self {
            patch: VulkanLinear::new_no_bias(&device, patch_input, dim, &patch_weights)?,
            cls: TrainableParameter::new(
                &device,
                &tensor(
                    &mut package,
                    &format!("{vision_prefix}.embeddings.class_embedding"),
                    &[dim],
                )?,
                false,
            )?,
            position: TrainableParameter::new(
                &device,
                &tensor(
                    &mut package,
                    &format!("{vision_prefix}.embeddings.position_embedding.weight"),
                    &[seq, dim],
                )?,
                true,
            )?,
            pre_norm: norm(
                &mut package,
                &device,
                &format!("{vision_prefix}.pre_layrnorm"),
                dim,
                eps,
            )?
            .use_cpu_welford(cpu_reference_math),
            layers: (0..active as usize)
                .map(|i| {
                    ClipLayer::load(
                        &mut package,
                        &device,
                        format!("{vision_prefix}.encoder.layers.{i}"),
                        rows,
                        dim,
                        intermediate,
                        eps,
                        cpu_reference_math,
                        cpu_reference_math,
                    )
                })
                .collect::<Result<_>>()?,
            projector1: linear(
                &mut package,
                &device,
                &format!("{projector_prefix}.linear_1"),
                dim,
                text_dim,
                bias,
            )?,
            projector2: linear(
                &mut package,
                &device,
                &format!("{projector_prefix}.linear_2"),
                text_dim,
                text_dim,
                bias,
            )?,
            pixels: buffer(pixel_len)?,
            patches: buffer(pixel_len)?,
            patch_output: buffer(image_count * (seq - 1) * dim)?,
            tokens: buffer(hidden)?,
            pre_output: buffer(hidden)?,
            pre_mean: buffer(rows)?,
            pre_rstd: buffer(rows)?,
            mask: GpuBuffer::from_f32(&device, &vec![1.; rows])?,
            dummy: buffer(rows.max(heads))?,
            selected: buffer(selected_rows * dim)?,
            projector_pre: buffer(selected_rows * text_dim)?,
            projector_act: buffer(selected_rows * text_dim)?,
            projected: buffer(selected_rows * text_dim)?,
            grad_projected: buffer(selected_rows * text_dim)?,
            grad_act: buffer(selected_rows * text_dim)?,
            grad_projector_pre: buffer(selected_rows * text_dim)?,
            grad_selected: buffer(selected_rows * dim)?,
            grad_vision_output: buffer(hidden)?,
            grad_tokens: buffer(hidden)?,
            grad_patch_output: buffer(image_count * (seq - 1) * dim)?,
            grad_patches: buffer(pixel_len)?,
            grad_pixels: buffer(pixel_len)?,
            processor: if directory.join("preprocessor_config.json").is_file() {
                Some(serde_json::from_slice(&fs::read(
                    directory.join("preprocessor_config.json"),
                )?)?)
            } else {
                None
            },
            heads: heads as u32,
            vk: VisionKernels::new(&device)?,
            device,
            text_rows,
            vocab_size,
            image_count,
            side,
            patch_size,
            channels,
            dim,
            seq,
            features_per_image,
            drop_cls,
            image_token,
            cpu_reference_math,
            activation,
            projector_activation,
            vision_prefix,
            projector_prefix,
        })
    }

    fn attention_push(&self) -> AttentionPush {
        let mut push = AttentionPush::zeroed();
        push.batch_size = self.image_count as u32;
        push.query_seq_len = self.seq as u32;
        push.key_value_seq_len = self.seq as u32;
        push.query_hidden_size = self.dim as u32;
        push.value_hidden_size = self.dim as u32;
        // Every layer has the same validated CLIP head geometry.
        // Stored separately from the text graph, whose GQA may differ.
        push.num_heads = self.vision_heads();
        push.num_key_value_heads = push.num_heads;
        push.scale = ((self.dim as f64 / push.num_heads as f64).sqrt().recip()) as f32;
        push.value_scale = 1.;
        push.unfused_qk = 9;
        push.kv_page_size = 1;
        push.kv_page_table_stride = 1;
        push
    }
    pub fn preprocess_rgb8(&self, rgb: &[u8]) -> Result<Vec<f32>> {
        let p = self
            .processor
            .as_ref()
            .context("raw RGB input requires preprocessor_config.json")?;
        if self.channels != 3 || rgb.len() != self.image_count * self.side * self.side * 3 {
            bail!("RGB8 input must match the fixed CLIP image dimensions");
        }
        if p["do_resize"].as_bool() != Some(false)
            || p["do_center_crop"].as_bool() != Some(false)
            || p["do_rescale"].as_bool() != Some(true)
            || p["do_normalize"].as_bool() != Some(true)
        {
            bail!(
                "native RGB processor requires fixed-size input with rescale and normalize enabled"
            );
        }
        let factor = p["rescale_factor"]
            .as_f64()
            .context("missing rescale factor")?;
        let means = p["image_mean"].as_array().context("missing RGB mean")?;
        let stds = p["image_std"]
            .as_array()
            .context("missing RGB standard deviation")?;
        if means.len() != 3 || stds.len() != 3 || !factor.is_finite() || factor <= 0. {
            bail!("invalid RGB processor parameters");
        }
        let mut mean = [0.; 3];
        let mut std = [0.; 3];
        for c in 0..3 {
            mean[c] = means[c].as_f64().context("invalid RGB mean")? as f32;
            std[c] = stds[c].as_f64().context("invalid RGB standard deviation")? as f32;
            if !mean[c].is_finite() || !std[c].is_finite() || std[c] <= 0. {
                bail!("invalid RGB normalization");
            }
        }
        let area = self.side * self.side;
        let mut values = vec![0.; rgb.len()];
        // The width-16 RDNA3 oracle uses Transformers' Torchvision backend:
        // normalize RGB directly with FP32 mean/std multiplied by 1/factor.
        // Keep the frozen Gen9/AVX2 processor's operation order unchanged.
        let rdna3_avx512 =
            crate::vendor::rdna3_avx512_cpu_contract(&self.device, aten_vector_width());
        let inverse_factor = (1.0 / factor) as f32;
        let fused_mean = mean.map(|v| v * inverse_factor);
        let fused_std = std.map(|v| v * inverse_factor);
        for image in 0..self.image_count {
            for pixel in 0..area {
                for channel in 0..3 {
                    if rdna3_avx512 {
                        values[(image * 3 + channel) * area + pixel] =
                            (rgb[(image * area + pixel) * 3 + channel] as f32
                                - fused_mean[channel])
                                / fused_std[channel];
                        continue;
                    }
                    let scaled = (rgb[(image * area + pixel) * 3 + channel] as f64 * factor) as f32;
                    values[(image * 3 + channel) * area + pixel] =
                        (scaled - mean[channel]) / std[channel];
                }
            }
        }
        Ok(values)
    }
    pub(super) fn validate(&self, ids: &[u32], mask: &[f32], pixels: &[f32]) -> Result<Vec<usize>> {
        if ids.len() != self.text_rows || mask.len() != ids.len() {
            bail!("invalid LLaVA text batch geometry");
        }
        if mask.iter().any(|&v| v != 0. && v != 1.) {
            bail!("LLaVA attention mask must be binary");
        }
        if ids.iter().any(|&id| id as usize >= self.vocab_size) {
            bail!("LLaVA token outside vocabulary");
        }
        if pixels.len() != self.image_count * self.channels * self.side * self.side
            || pixels.iter().any(|v| !v.is_finite())
        {
            bail!("pixel_values must be finite NCHW at the configured CLIP resolution");
        }
        let rows = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.image_token).then_some(i))
            .collect::<Vec<_>>();
        if rows.len() != self.image_count * self.features_per_image
            || rows.iter().any(|&i| mask[i] != 1.)
        {
            bail!("image placeholders must match every projected feature and be visible");
        }
        Ok(rows)
    }
    pub(super) fn encode_device(&self, pixels: &[f32], k: &TransformerKernels) -> Result<()> {
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.pixels, pixels)?;
        let patch = PatchPush {
            images: self.image_count as u32,
            side: self.side as u32,
            patch: self.patch_size as u32,
            channels: self.channels as u32,
            inverse: 0,
            layout: 0,
        };
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.pixels, &self.patches],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixels.len(), 256), 1, 1],
        )?;
        let patch_rows = self.image_count * (self.seq - 1);
        if self.cpu_reference_math {
            let push = PatchLinearPush {
                rows: patch_rows as u32,
                input_dim: (self.channels * self.patch_size * self.patch_size) as u32,
                output_dim: self.dim as u32,
                channels: self.channels as u32,
            };
            self.vk.patch_linear_kernel().record_dispatch(
                &mut commands,
                &[&self.patches, &self.patch.weight.values, &self.patch_output],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(self.dim, 16), div_ceil_u32(patch_rows, 16), 1],
            )?;
        } else {
            self.patch.record_forward(
                &mut commands,
                k,
                &self.patches,
                &self.patch_output,
                patch_rows,
            )?;
        }
        let tokens = TokenPush {
            images: self.image_count as u32,
            seq: self.seq as u32,
            dim: self.dim as u32,
            mode: 0,
        };
        self.vk.tokens.record_dispatch(
            &mut commands,
            &[
                &self.patch_output,
                &self.cls.values,
                &self.position.values,
                &self.tokens,
            ],
            bytemuck::bytes_of(&tokens),
            [
                div_ceil_u32(self.image_count * self.seq * self.dim, 256),
                1,
                1,
            ],
        )?;
        self.pre_norm.record_forward(
            &mut commands,
            k,
            &self.tokens,
            &self.pre_output,
            &self.pre_mean,
            &self.pre_rstd,
            self.image_count * self.seq,
        )?;
        let mut hidden = &self.pre_output;
        for layer in &self.layers {
            layer.forward(
                &mut commands,
                k,
                &self.vk,
                hidden,
                &self.mask,
                &self.dummy,
                self.attention_push(),
                &self.activation,
            )?;
            hidden = &layer.tape[OUT];
        }
        for image in 0..self.image_count {
            commands.copy_f32_range(
                hidden,
                (image * self.seq + usize::from(self.drop_cls)) * self.dim,
                &self.selected,
                image * self.features_per_image * self.dim,
                self.features_per_image * self.dim,
            )?;
        }
        let rows = self.image_count * self.features_per_image;
        self.projector1.record_forward(
            &mut commands,
            k,
            &self.selected,
            &self.projector_pre,
            rows,
        )?;
        if self.cpu_reference_math && self.projector_activation == "gelu" {
            let len = rows * self.projector1.output_dim;
            self.vk
                .rdna3_avx512_gelu
                .as_ref()
                .map(|family| &family[0])
                .unwrap_or(&self.vk.gelu_cpu_forward)
                .record_dispatch(
                    &mut commands,
                    &[&self.projector_pre, &self.projector_act],
                    bytemuck::cast_slice(&[len as u32]),
                    [div_ceil_u32(len, 256), 1, 1],
                )?;
        } else {
            self.vk.activation(
                k,
                &self.projector_activation,
                &mut commands,
                &self.projector_pre,
                None,
                &self.projector_act,
                rows * self.projector1.output_dim,
            )?;
        }
        self.projector2.record_forward(
            &mut commands,
            k,
            &self.projector_act,
            &self.projected,
            rows,
        )?;
        commands.submit()
    }
    pub(super) fn forward_result(&self, logits: Vec<f32>) -> Result<LlavaForward> {
        let feature_rows = self.image_count * self.features_per_image;
        Ok(LlavaForward {
            logits,
            vision_features: self.selected.read_f32(feature_rows * self.dim)?,
            projector_pre: self
                .projector_pre
                .read_f32(feature_rows * self.projector1.output_dim)?,
            projector_act: self
                .projector_act
                .read_f32(feature_rows * self.projector1.output_dim)?,
            projected_features: self
                .projected
                .read_f32(feature_rows * self.projector2.output_dim)?,
        })
    }
    pub(super) fn backward_image(
        &mut self,
        projected_gradient: &GpuBuffer,
        step: u32,
        pixel_values: &[f32],
        k: &TransformerKernels,
        hyper: AdamWHyperParams,
    ) -> Result<ClipAdapterGradients> {
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        let feature_rows = self.image_count * self.features_per_image;
        commands.copy_f32(
            projected_gradient,
            &self.grad_projected,
            feature_rows * self.projector2.output_dim,
        )?;
        self.projector2.record_backward(
            &mut commands,
            k,
            &self.projector_act,
            &self.grad_projected,
            &self.grad_act,
            feature_rows,
        )?;
        if self.cpu_reference_math && self.projector_activation == "gelu" {
            let len = feature_rows * self.projector1.output_dim;
            self.vk
                .rdna3_avx512_gelu
                .as_ref()
                .map(|family| &family[1])
                .unwrap_or(&self.vk.gelu_cpu_backward)
                .record_dispatch(
                    &mut commands,
                    &[
                        &self.projector_pre,
                        &self.grad_act,
                        &self.grad_projector_pre,
                    ],
                    bytemuck::cast_slice(&[len as u32]),
                    [div_ceil_u32(len, 256), 1, 1],
                )?;
        } else {
            self.vk.activation(
                k,
                &self.projector_activation,
                &mut commands,
                &self.projector_pre,
                Some(&self.grad_act),
                &self.grad_projector_pre,
                feature_rows * self.projector1.output_dim,
            )?;
        }
        self.projector1.record_backward(
            &mut commands,
            k,
            &self.selected,
            &self.grad_projector_pre,
            &self.grad_selected,
            feature_rows,
        )?;
        commands.fill_zero_f32(
            &self.grad_vision_output,
            self.image_count * self.seq * self.dim,
        )?;
        for image in 0..self.image_count {
            commands.copy_f32_range(
                &self.grad_selected,
                image * self.features_per_image * self.dim,
                &self.grad_vision_output,
                (image * self.seq + usize::from(self.drop_cls)) * self.dim,
                self.features_per_image * self.dim,
            )?;
        }
        let mut grad = &self.grad_vision_output;
        for index in (0..self.layers.len()).rev() {
            let input = if index == 0 {
                &self.pre_output
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
                &self.mask,
                &self.dummy,
                self.attention_push(),
                &self.activation,
            )?;
            grad = &layer.tape[DI];
        }
        self.pre_norm.record_backward(
            &mut commands,
            k,
            &self.tokens,
            grad,
            &self.pre_mean,
            &self.pre_rstd,
            &self.grad_tokens,
            self.image_count * self.seq,
        )?;
        let mut push = TokenPush {
            images: self.image_count as u32,
            seq: self.seq as u32,
            dim: self.dim as u32,
            mode: 1,
        };
        for (mode, output, len) in [
            (
                1,
                &self.grad_patch_output,
                self.image_count * (self.seq - 1) * self.dim,
            ),
            (2, self.cls.grad_buffer()?, self.dim),
            (3, self.position.grad_buffer()?, self.seq * self.dim),
        ] {
            push.mode = mode;
            self.vk.tokens.record_dispatch(
                &mut commands,
                &[
                    &self.grad_tokens,
                    &self.cls.values,
                    &self.position.values,
                    output,
                ],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(len, 256), 1, 1],
            )?;
        }
        self.patch.record_backward(
            &mut commands,
            k,
            &self.patches,
            &self.grad_patch_output,
            &self.grad_patches,
            self.image_count * (self.seq - 1),
        )?;
        let patch = PatchPush {
            images: self.image_count as u32,
            side: self.side as u32,
            patch: self.patch_size as u32,
            channels: self.channels as u32,
            inverse: 1,
            layout: 0,
        };
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.grad_patches, &self.grad_pixels],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixel_values.len(), 256), 1, 1],
        )?;
        self.projector1.record_step(&mut commands, k, step, hyper)?;
        self.projector2.record_step(&mut commands, k, step, hyper)?;
        for layer in &self.layers {
            layer.step(&mut commands, k, step, hyper)?;
        }
        self.pre_norm.record_step(&mut commands, k, step, hyper)?;
        self.patch.record_step(&mut commands, k, step, hyper)?;
        self.cls.record_step(&mut commands, k, step, hyper)?;
        self.position.record_step(&mut commands, k, step, hyper)?;
        commands.submit()?;
        Ok(ClipAdapterGradients {
            pixel_gradient: self.grad_pixels.read_f32(pixel_values.len())?,
            projected_gradient: projected_gradient
                .read_f32(feature_rows * self.projector2.output_dim)?,
        })
    }
    pub(super) fn debug_image_training_stages(&self) -> Result<serde_json::Value> {
        let mut stages = serde_json::Map::new();
        let mut gradients = serde_json::Map::new();
        let rows = self.image_count * self.seq;
        let feature_rows = self.image_count * self.features_per_image;
        for (name, forward, gradient, len) in [
            (
                "selected",
                &self.selected,
                &self.grad_selected,
                feature_rows * self.dim,
            ),
            (
                "projector_pre",
                &self.projector_pre,
                &self.grad_projector_pre,
                feature_rows * self.projector1.output_dim,
            ),
            (
                "projector_act",
                &self.projector_act,
                &self.grad_act,
                feature_rows * self.projector1.output_dim,
            ),
            (
                "projected",
                &self.projected,
                &self.grad_projected,
                feature_rows * self.projector2.output_dim,
            ),
            ("tokens", &self.tokens, &self.grad_tokens, rows * self.dim),
        ] {
            stages.insert(name.into(), serde_json::json!(forward.read_f32(len)?));
            gradients.insert(name.into(), serde_json::json!(gradient.read_f32(len)?));
        }
        stages.insert(
            "pre_output".into(),
            serde_json::json!(self.pre_output.read_f32(rows * self.dim)?),
        );
        let first_grad = self
            .layers
            .first()
            .map(|l| &l.tape[DI])
            .unwrap_or(&self.grad_vision_output);
        gradients.insert(
            "pre_output".into(),
            serde_json::json!(first_grad.read_f32(rows * self.dim)?),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            let t = &layer.tape;
            let mid = layer.fc1.output_dim;
            for (name, forward, gradient, width) in [
                ("n1", N1, DN1, self.dim),
                ("q", Q, DQ, self.dim),
                ("k", K, DK, self.dim),
                ("v", V, DV, self.dim),
                ("a", A, DA, self.dim),
                ("o", O, DR, self.dim),
                ("r", R, DR, self.dim),
                ("n2", N2, DN2, self.dim),
                ("f", F, DF, mid),
                ("act", ACT, DACT, mid),
            ] {
                stages.insert(
                    format!("layer_{i}/{name}"),
                    serde_json::json!(t[forward].read_f32(rows * width)?),
                );
                gradients.insert(
                    format!("layer_{i}/{name}"),
                    serde_json::json!(t[gradient].read_f32(rows * width)?),
                );
            }
            let incoming = self
                .layers
                .get(i + 1)
                .map(|l| &l.tape[DI])
                .unwrap_or(&self.grad_vision_output);
            stages.insert(
                format!("layer_{i}/out"),
                serde_json::json!(t[OUT].read_f32(rows * self.dim)?),
            );
            gradients.insert(
                format!("layer_{i}/out"),
                serde_json::json!(incoming.read_f32(rows * self.dim)?),
            );
            gradients.insert(
                format!("layer_{i}/out_linear"),
                serde_json::json!(incoming.read_f32(rows * self.dim)?),
            );
        }
        Ok(
            serde_json::json!({"stages":stages,"gradients":gradients,"pixel_gradient":self.grad_pixels.read_f32(self.image_count*self.channels*self.side*self.side)?}),
        )
    }
    pub(super) fn export_image_parameters(&self, source: &Path, destination: &Path) -> Result<()> {
        if source.join("preprocessor_config.json").is_file() {
            fs::copy(
                source.join("preprocessor_config.json"),
                destination.join("preprocessor_config.json"),
            )?;
        }
        let mut values = BTreeMap::new();
        let prefix = &self.vision_prefix;
        values.insert(
            format!("{prefix}.embeddings.class_embedding"),
            (vec![self.dim], self.cls.read()?),
        );
        values.insert(
            format!("{prefix}.embeddings.position_embedding.weight"),
            (vec![self.seq, self.dim], self.position.read()?),
        );
        values.insert(
            format!("{prefix}.embeddings.patch_embedding.weight"),
            (
                vec![self.dim, self.channels, self.patch_size, self.patch_size],
                self.patch.weight.read()?,
            ),
        );
        export_norm(
            &mut values,
            &format!("{prefix}.pre_layrnorm"),
            &self.pre_norm,
        )?;
        for layer in &self.layers {
            layer.export(&mut values)?;
        }
        export_linear(
            &mut values,
            &format!("{}.linear_1", self.projector_prefix),
            &self.projector1,
        )?;
        export_linear(
            &mut values,
            &format!("{}.linear_2", self.projector_prefix),
            &self.projector2,
        )?;
        let replacements = values
            .iter()
            .map(|(name, (shape, values))| (name.as_str(), shape.as_slice(), values.as_slice()))
            .collect::<Vec<_>>();
        let path = destination.join("model.safetensors");
        crate::replace_f32_tensors(&path, &path, &replacements)
    }
}
