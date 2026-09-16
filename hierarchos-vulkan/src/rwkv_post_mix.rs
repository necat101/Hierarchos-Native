use std::{
    collections::HashMap,
    path::Path,
    sync::{Mutex, OnceLock},
};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::rwkv_optimizer::{RwkvDecayClass, RwkvTrainableRef};
use crate::{read_f32_tensor, vulkan, GpuBuffer, VulkanDevice};

const GROUP_NORM_FORWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_group_norm_forward.spv");
const GROUP_NORM_BONUS_GATE_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_group_norm_bonus_gate_forward_fused.spv");
const GROUP_NORM_BONUS_GATE_LINEAR_RESIDUAL_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_group_norm_bonus_gate_linear_residual_forward_fused.spv");
const GROUP_NORM_BONUS_GATE_LINEAR_RESIDUAL_FORWARD_FUSED_COMPACT_ONE_ROW_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_group_norm_bonus_gate_linear_residual_forward_fused_compact_one_row.spv"
);
const GROUP_NORM_BONUS_GATE_LINEAR_RESIDUAL_FORWARD_FUSED_TWO_ROWS_SPV: &[u8] = include_bytes!(
    "../shaders/rwkv_group_norm_bonus_gate_linear_residual_forward_fused_two_rows.spv"
);
const HIERARCHOS_RWKV_POST_MIX_DISABLE_FORWARD_AUTOTUNE_ENV: &str =
    "HIERARCHOS_RWKV_POST_MIX_DISABLE_FORWARD_AUTOTUNE";
const HIERARCHOS_RWKV_POST_MIX_FORWARD_AUTOTUNE_LOG_ENV: &str =
    "HIERARCHOS_RWKV_POST_MIX_FORWARD_AUTOTUNE_LOG";
const GROUP_NORM_INPUT_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_group_norm_input_grad.spv");
const GROUP_NORM_PARAM_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_group_norm_param_grad.spv");
const BONUS_GATE_FORWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_bonus_gate_forward.spv");
const BONUS_GATE_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_bonus_gate_backward.spv");
const CHANNEL_REDUCE_SPV: &[u8] = include_bytes!("../shaders/channel_reduce.spv");
const LINEAR_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_forward.spv");
const LINEAR_RESIDUAL_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_residual_forward.spv");
const LINEAR_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
const LINEAR_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_weight_grad.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupNormPush {
    batch: u32,
    width: u32,
    head_size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HeadPush {
    batch: u32,
    width: u32,
    head_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReducePush {
    batch: u32,
    width: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PostMixForwardTopology {
    OneRow,
    CompactOneRow,
    TwoRows,
}

impl PostMixForwardTopology {
    fn label(self) -> &'static str {
        match self {
            Self::OneRow => "post-mix-one-row",
            Self::CompactOneRow => "post-mix-compact-one-row",
            Self::TwoRows => "post-mix-two-rows",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PostMixForwardAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    width: usize,
    head_size: usize,
    batch_pairs: usize,
    has_unpaired_tail: bool,
}

#[derive(Clone, Copy, Debug)]
struct PostMixForwardDecision {
    topology: PostMixForwardTopology,
    autotuned: bool,
}

struct PostMixForwardProbeBuffers {
    tmix: GpuBuffer,
    r: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    g: GpuBuffer,
    residual: GpuBuffer,
    group_normed: GpuBuffer,
    group_mean: GpuBuffer,
    group_rstd: GpuBuffer,
    gated: GpuBuffer,
    bonus_scalar: GpuBuffer,
    output: GpuBuffer,
}

static POST_MIX_FORWARD_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<PostMixForwardAutotuneKey, PostMixForwardDecision>>,
> = OnceLock::new();

fn select_post_mix_forward_topology(
    timings: &[(PostMixForwardTopology, f64)],
    structural_default: PostMixForwardTopology,
) -> PostMixForwardTopology {
    let Some((_, default_ms)) = timings
        .iter()
        .find(|(topology, _)| *topology == structural_default)
    else {
        return structural_default;
    };
    let Some(&(best, best_ms)) = timings
        .iter()
        .filter(|(_, ms)| ms.is_finite() && *ms > 0.0)
        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
    else {
        return structural_default;
    };
    if best_ms < *default_ms * 0.98 {
        best
    } else {
        structural_default
    }
}

#[derive(Debug)]
pub struct RwkvPostMixResult {
    pub output: Vec<f32>,
    pub group_normed: Vec<f32>,
    pub grad_tmix: Vec<f32>,
    pub grad_r: Vec<f32>,
    pub grad_k: Vec<f32>,
    pub grad_v: Vec<f32>,
    pub grad_g: Vec<f32>,
    pub grad_r_k: Vec<f32>,
    pub grad_output_weight: Vec<f32>,
    pub grad_group_norm_weight: Vec<f32>,
    pub grad_group_norm_bias: Vec<f32>,
}

/// Vulkan-native RWKV-v8 post-recurrence time-mix slice:
///
/// `GroupNorm(tmix) + ((r * k * r_k).sum(head) * v)`
/// `-> multiply by g -> output projection`.
///
/// All trainable tensors retain their PyTorch checkpoint layouts. In
/// particular `output.weight` is `[out_features, in_features]`, while `r_k`
/// remains `[heads, head_size]` in SafeTensors and is only flattened in GPU
/// storage.
pub struct RwkvPostMixOp {
    device: VulkanDevice,
    width: usize,
    head_size: usize,
    heads: usize,
    max_batch: usize,

    group_norm_weight: GpuBuffer,
    group_norm_bias: GpuBuffer,
    r_k: GpuBuffer,
    output_weight: GpuBuffer,

    group_normed: GpuBuffer,
    group_mean: GpuBuffer,
    group_rstd: GpuBuffer,
    bonus_scalar: GpuBuffer,
    gated: GpuBuffer,
    output: GpuBuffer,
    grad_output: GpuBuffer,
    grad_gated: GpuBuffer,
    grad_group_normed: GpuBuffer,
    grad_tmix: GpuBuffer,
    grad_r: GpuBuffer,
    grad_k: GpuBuffer,
    grad_v: GpuBuffer,
    grad_g: GpuBuffer,
    grad_r_k_partial: GpuBuffer,
    grad_r_k: GpuBuffer,
    grad_output_weight: GpuBuffer,
    grad_group_norm_weight: GpuBuffer,
    grad_group_norm_bias: GpuBuffer,

    output_readback: GpuBuffer,
    group_normed_readback: GpuBuffer,
    grad_tmix_readback: GpuBuffer,
    grad_r_readback: GpuBuffer,
    grad_k_readback: GpuBuffer,
    grad_v_readback: GpuBuffer,
    grad_g_readback: GpuBuffer,
    grad_r_k_readback: GpuBuffer,
    grad_output_weight_readback: GpuBuffer,
    grad_group_norm_weight_readback: GpuBuffer,
    grad_group_norm_bias_readback: GpuBuffer,

    group_norm_forward: vulkan::ComputeKernel,
    group_norm_bonus_gate_forward_fused: Option<vulkan::ComputeKernel>,
    group_norm_bonus_gate_linear_residual_forward_fused: Option<vulkan::ComputeKernel>,
    group_norm_bonus_gate_linear_residual_forward_fused_compact_one_row:
        Option<vulkan::ComputeKernel>,
    group_norm_bonus_gate_linear_residual_forward_fused_two_rows: Option<vulkan::ComputeKernel>,
    group_norm_input_grad: vulkan::ComputeKernel,
    group_norm_param_grad: vulkan::ComputeKernel,
    bonus_gate_forward: vulkan::ComputeKernel,
    bonus_gate_backward: vulkan::ComputeKernel,
    channel_reduce: vulkan::ComputeKernel,
    linear_forward: vulkan::ComputeKernel,
    linear_residual_forward: vulkan::ComputeKernel,
    linear_input_grad: vulkan::ComputeKernel,
    linear_weight_grad: vulkan::ComputeKernel,
}

impl RwkvPostMixOp {
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        head_size: usize,
        max_batch: usize,
    ) -> Result<Self> {
        if prefix.trim().is_empty() {
            bail!("RWKV tensor prefix must not be empty");
        }
        let path = model_dir.as_ref().join("model.safetensors");
        let (norm_shape, norm_weight) = read_f32_tensor(&path, &format!("{prefix}.ln_x.weight"))?;
        let width = match norm_shape.as_slice() {
            [width] if *width > 0 => *width,
            _ => bail!("RWKV tensor {prefix}.ln_x.weight must have shape [C], got {norm_shape:?}"),
        };
        let norm_bias = read_vector(&path, &format!("{prefix}.ln_x.bias"), width)?;
        let (rk_shape, r_k) = read_f32_tensor(&path, &format!("{prefix}.r_k"))?;
        if head_size == 0 || !width.is_multiple_of(head_size) {
            bail!("RWKV post-mix width {width} must be divisible by head_size {head_size}");
        }
        let heads = width / head_size;
        if rk_shape != [heads, head_size] && rk_shape != [width] {
            bail!(
                "RWKV tensor {prefix}.r_k has shape {rk_shape:?}; expected [{heads}, {head_size}]"
            );
        }
        let output_weight = read_matrix(&path, &format!("{prefix}.output.weight"), width)?;
        Self::new(
            device,
            width,
            head_size,
            max_batch,
            &norm_weight,
            &norm_bias,
            &r_k,
            &output_weight,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: VulkanDevice,
        width: usize,
        head_size: usize,
        max_batch: usize,
        group_norm_weight: &[f32],
        group_norm_bias: &[f32],
        r_k: &[f32],
        output_weight: &[f32],
    ) -> Result<Self> {
        if width == 0 || head_size == 0 || max_batch == 0 {
            bail!("RWKV post-mix dimensions and max_batch must be positive");
        }
        if !width.is_multiple_of(head_size) {
            bail!("RWKV post-mix width {width} must be divisible by head_size {head_size}");
        }
        validate_len("group_norm_weight", group_norm_weight, width)?;
        validate_len("group_norm_bias", group_norm_bias, width)?;
        validate_len("r_k", r_k, width)?;
        validate_len("output_weight", output_weight, width * width)?;
        let heads = width / head_size;
        let vector_len = max_batch
            .checked_mul(width)
            .context("RWKV post-mix vector capacity overflow")?;
        let group_len = max_batch
            .checked_mul(heads)
            .context("RWKV post-mix group capacity overflow")?;
        let weight_len = width
            .checked_mul(width)
            .context("RWKV post-mix output weight size overflow")?;

        let group_norm_bonus_gate_forward_fused = if device.supports_storage_buffer_bindings(13) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                GROUP_NORM_BONUS_GATE_FORWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<GroupNormPush>() as u32,
            )?)
        } else {
            None
        };
        let group_norm_bonus_gate_linear_residual_forward_fused = if width <= 512
            && device.supports_storage_buffer_bindings(16)
            && device.supports_compute_work_group_size_x(256)
        {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                GROUP_NORM_BONUS_GATE_LINEAR_RESIDUAL_FORWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<GroupNormPush>() as u32,
            )?)
        } else {
            None
        };
        let group_norm_bonus_gate_linear_residual_forward_fused_compact_one_row = if width <= 32
            && device.supports_storage_buffer_bindings(16)
            && device.supports_compute_work_group_size_x(64)
            && std::env::var_os("HIERARCHOS_RWKV_POST_MIX_DISABLE_COMPACT_ONE_ROW_FORWARD_FUSION")
                .is_none()
        {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                GROUP_NORM_BONUS_GATE_LINEAR_RESIDUAL_FORWARD_FUSED_COMPACT_ONE_ROW_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<GroupNormPush>() as u32,
            )?)
        } else {
            None
        };
        let group_norm_bonus_gate_linear_residual_forward_fused_two_rows = if width <= 32
            && device.supports_storage_buffer_bindings(16)
            && device.supports_compute_work_group_size_x(64)
            && std::env::var_os("HIERARCHOS_RWKV_POST_MIX_DISABLE_TWO_ROW_FORWARD_FUSION").is_none()
        {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                GROUP_NORM_BONUS_GATE_LINEAR_RESIDUAL_FORWARD_FUSED_TWO_ROWS_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<GroupNormPush>() as u32,
            )?)
        } else {
            None
        };

        Ok(Self {
            group_norm_forward: vulkan::ComputeKernel::new(
                &device,
                GROUP_NORM_FORWARD_SPV,
                6,
                std::mem::size_of::<GroupNormPush>() as u32,
            )?,
            group_norm_bonus_gate_forward_fused,
            group_norm_bonus_gate_linear_residual_forward_fused,
            group_norm_bonus_gate_linear_residual_forward_fused_compact_one_row,
            group_norm_bonus_gate_linear_residual_forward_fused_two_rows,
            group_norm_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                GROUP_NORM_INPUT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<GroupNormPush>() as u32,
            )?,
            group_norm_param_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                GROUP_NORM_PARAM_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<GroupNormPush>() as u32,
            )?,
            bonus_gate_forward: vulkan::ComputeKernel::new(
                &device,
                BONUS_GATE_FORWARD_SPV,
                8,
                std::mem::size_of::<HeadPush>() as u32,
            )?,
            bonus_gate_backward: vulkan::ComputeKernel::new(
                &device,
                BONUS_GATE_BACKWARD_SPV,
                14,
                std::mem::size_of::<HeadPush>() as u32,
            )?,
            channel_reduce: vulkan::ComputeKernel::new(
                &device,
                CHANNEL_REDUCE_SPV,
                2,
                std::mem::size_of::<ReducePush>() as u32,
            )?,
            linear_forward: vulkan::ComputeKernel::new(
                &device,
                LINEAR_FORWARD_SPV,
                3,
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear_residual_forward: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_RESIDUAL_FORWARD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_INPUT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear_weight_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_WEIGHT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            group_norm_weight: GpuBuffer::from_f32(&device, group_norm_weight)?,
            group_norm_bias: GpuBuffer::from_f32(&device, group_norm_bias)?,
            r_k: GpuBuffer::from_f32(&device, r_k)?,
            output_weight: GpuBuffer::from_f32(&device, output_weight)?,
            group_normed: GpuBuffer::zeros_f32(&device, vector_len)?,
            group_mean: GpuBuffer::zeros_f32(&device, group_len)?,
            group_rstd: GpuBuffer::zeros_f32(&device, group_len)?,
            bonus_scalar: GpuBuffer::zeros_f32(&device, group_len)?,
            gated: GpuBuffer::zeros_f32(&device, vector_len)?,
            output: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_gated: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_group_normed: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_tmix: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_r: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_k: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_v: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_g: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_r_k_partial: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_r_k: GpuBuffer::zeros_f32(&device, width)?,
            grad_output_weight: GpuBuffer::zeros_f32(&device, weight_len)?,
            grad_group_norm_weight: GpuBuffer::zeros_f32(&device, width)?,
            grad_group_norm_bias: GpuBuffer::zeros_f32(&device, width)?,
            output_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            group_normed_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_tmix_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_r_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_k_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_v_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_g_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_r_k_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_output_weight_readback: GpuBuffer::zeros_host_f32(&device, weight_len)?,
            grad_group_norm_weight_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_group_norm_bias_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            device,
            width,
            head_size,
            heads,
            max_batch,
        })
    }

    pub fn forward_backward(
        &mut self,
        batch: usize,
        tmix: &[f32],
        r: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        grad_output: &[f32],
    ) -> Result<RwkvPostMixResult> {
        let vector_len = self.validate_batch_inputs(batch, tmix, r, k, v, g, grad_output)?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let tmix_buffer = GpuBuffer::from_f32(&self.device, tmix)?;
        let r_buffer = GpuBuffer::from_f32(&self.device, r)?;
        let k_buffer = GpuBuffer::from_f32(&self.device, k)?;
        let v_buffer = GpuBuffer::from_f32(&self.device, v)?;
        let g_buffer = GpuBuffer::from_f32(&self.device, g)?;
        commands.upload_f32(&self.grad_output, grad_output)?;
        self.record_forward(
            &mut commands,
            batch,
            &tmix_buffer,
            &r_buffer,
            &k_buffer,
            &v_buffer,
            &g_buffer,
        )?;
        self.record_backward(
            &mut commands,
            batch,
            &tmix_buffer,
            &r_buffer,
            &k_buffer,
            &v_buffer,
            &g_buffer,
            &self.grad_output,
        )?;
        self.record_readback(&mut commands, batch)?;
        commands.submit()?;
        debug_assert_eq!(vector_len, batch * self.width);
        self.read_result(batch)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        tmix: &GpuBuffer,
        r: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        g: &GpuBuffer,
    ) -> Result<()> {
        self.record_gated_forward(commands, batch, tmix, r, k, v, g)?;
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        self.linear_forward.record_dispatch(
            commands,
            &[&self.gated, &self.output_weight, &self.output],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1],
        )
    }

    fn available_fused_forward_topologies(&self) -> Vec<PostMixForwardTopology> {
        let mut topologies = Vec::with_capacity(3);
        if self
            .group_norm_bonus_gate_linear_residual_forward_fused
            .is_some()
        {
            topologies.push(PostMixForwardTopology::OneRow);
        }
        if self
            .group_norm_bonus_gate_linear_residual_forward_fused_compact_one_row
            .is_some()
        {
            topologies.push(PostMixForwardTopology::CompactOneRow);
        }
        if self
            .group_norm_bonus_gate_linear_residual_forward_fused_two_rows
            .is_some()
        {
            topologies.push(PostMixForwardTopology::TwoRows);
        }
        topologies
    }

    fn structural_fused_forward_topology(&self) -> Option<PostMixForwardTopology> {
        if self
            .group_norm_bonus_gate_linear_residual_forward_fused_two_rows
            .is_some()
        {
            Some(PostMixForwardTopology::TwoRows)
        } else if self
            .group_norm_bonus_gate_linear_residual_forward_fused_compact_one_row
            .is_some()
        {
            Some(PostMixForwardTopology::CompactOneRow)
        } else if self
            .group_norm_bonus_gate_linear_residual_forward_fused
            .is_some()
        {
            Some(PostMixForwardTopology::OneRow)
        } else {
            None
        }
    }

    fn fused_forward_kernel(
        &self,
        topology: PostMixForwardTopology,
    ) -> Option<&vulkan::ComputeKernel> {
        match topology {
            PostMixForwardTopology::OneRow => self
                .group_norm_bonus_gate_linear_residual_forward_fused
                .as_ref(),
            PostMixForwardTopology::CompactOneRow => self
                .group_norm_bonus_gate_linear_residual_forward_fused_compact_one_row
                .as_ref(),
            PostMixForwardTopology::TwoRows => self
                .group_norm_bonus_gate_linear_residual_forward_fused_two_rows
                .as_ref(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_fused_forward_topology(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        tmix: &GpuBuffer,
        r: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        g: &GpuBuffer,
        residual: &GpuBuffer,
        output: &GpuBuffer,
        topology: PostMixForwardTopology,
    ) -> Result<()> {
        self.record_fused_forward_topology_into(
            commands,
            batch,
            tmix,
            r,
            k,
            v,
            g,
            residual,
            &self.group_normed,
            &self.group_mean,
            &self.group_rstd,
            &self.gated,
            &self.bonus_scalar,
            output,
            topology,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_fused_forward_topology_into(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        tmix: &GpuBuffer,
        r: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        g: &GpuBuffer,
        residual: &GpuBuffer,
        group_normed: &GpuBuffer,
        group_mean: &GpuBuffer,
        group_rstd: &GpuBuffer,
        gated: &GpuBuffer,
        bonus_scalar: &GpuBuffer,
        output: &GpuBuffer,
        topology: PostMixForwardTopology,
    ) -> Result<()> {
        let kernel = self.fused_forward_kernel(topology).with_context(|| {
            format!(
                "RWKV post-mix fused forward topology {} is unavailable",
                topology.label()
            )
        })?;
        let push = GroupNormPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
            eps: 64.0e-5,
        };
        kernel.record_dispatch(
            commands,
            &[
                tmix,
                &self.group_norm_weight,
                &self.group_norm_bias,
                r,
                k,
                v,
                g,
                &self.r_k,
                &self.output_weight,
                residual,
                group_normed,
                group_mean,
                group_rstd,
                gated,
                bonus_scalar,
                output,
            ],
            bytemuck::bytes_of(&push),
            [
                match topology {
                    PostMixForwardTopology::OneRow | PostMixForwardTopology::CompactOneRow => {
                        batch as u32
                    }
                    PostMixForwardTopology::TwoRows => batch.div_ceil(2) as u32,
                },
                1,
                1,
            ],
        )
    }

    fn allocate_fused_forward_autotune_probe(
        &self,
        batch: usize,
    ) -> Result<PostMixForwardProbeBuffers> {
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV post-mix autotune vector size overflow")?;
        let group_len = batch
            .checked_mul(self.heads)
            .context("RWKV post-mix autotune group size overflow")?;
        Ok(PostMixForwardProbeBuffers {
            tmix: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            r: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            k: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            v: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            g: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            residual: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            group_normed: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            group_mean: GpuBuffer::zeros_f32(&self.device, group_len)?,
            group_rstd: GpuBuffer::zeros_f32(&self.device, group_len)?,
            gated: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            bonus_scalar: GpuBuffer::zeros_f32(&self.device, group_len)?,
            output: GpuBuffer::zeros_f32(&self.device, vector_len)?,
        })
    }

    fn time_fused_forward_topology_ms(
        &self,
        batch: usize,
        probe: &PostMixForwardProbeBuffers,
        topology: PostMixForwardTopology,
    ) -> Result<f64> {
        let repetitions = if batch >= 64 { 4 } else { 16 };
        let elapsed_ms = self.device.time_compute_batch_ms(|commands| {
            for _ in 0..repetitions {
                self.record_fused_forward_topology_into(
                    commands,
                    batch,
                    &probe.tmix,
                    &probe.r,
                    &probe.k,
                    &probe.v,
                    &probe.g,
                    &probe.residual,
                    &probe.group_normed,
                    &probe.group_mean,
                    &probe.group_rstd,
                    &probe.gated,
                    &probe.bonus_scalar,
                    &probe.output,
                    topology,
                )?;
            }
            Ok(())
        })?;
        Ok(elapsed_ms / repetitions as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn choose_fused_forward_topology(&self, batch: usize) -> Result<PostMixForwardDecision> {
        let candidates = self.available_fused_forward_topologies();
        let structural_default = self
            .structural_fused_forward_topology()
            .context("RWKV fused post-mix forward is unavailable")?;
        if candidates.len() == 1
            || std::env::var_os(HIERARCHOS_RWKV_POST_MIX_DISABLE_FORWARD_AUTOTUNE_ENV).is_some()
        {
            return Ok(PostMixForwardDecision {
                topology: structural_default,
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let key = PostMixForwardAutotuneKey {
            device_name: self.device.name().to_owned(),
            subgroup_size,
            width: self.width,
            head_size: self.head_size,
            batch_pairs: batch.div_ceil(2),
            has_unpaired_tail: batch % 2 != 0,
        };
        let cache = POST_MIX_FORWARD_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(decision) = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("RWKV post-mix forward autotune cache lock was poisoned"))?
            .get(&key)
            .copied()
        {
            return Ok(decision);
        }

        // Autotuning can run while the caller is still recording a larger
        // command batch. Probe on isolated buffers so the synchronous timing
        // submissions cannot observe not-yet-produced inputs or overwrite
        // residual/state buffers owned by that pending graph.
        let probe = self.allocate_fused_forward_autotune_probe(batch)?;
        let time_topology = |topology| self.time_fused_forward_topology_ms(batch, &probe, topology);
        if let Err(err) = time_topology(structural_default) {
            if std::env::var_os(HIERARCHOS_RWKV_POST_MIX_FORWARD_AUTOTUNE_LOG_ENV).is_some() {
                eprintln!(
                    "RWKV post-mix forward autotune warmup failed device={} batch={batch}: {err:#}; using {}",
                    self.device.name(),
                    structural_default.label()
                );
            }
            return Ok(PostMixForwardDecision {
                topology: structural_default,
                autotuned: false,
            });
        }

        let mut samples = candidates
            .iter()
            .copied()
            .map(|topology| (topology, Vec::with_capacity(3)))
            .collect::<HashMap<_, _>>();
        for round in 0..3 {
            for offset in 0..candidates.len() {
                let index = if round % 2 == 0 {
                    offset
                } else {
                    candidates.len() - 1 - offset
                };
                let topology = candidates[index];
                match time_topology(topology) {
                    Ok(ms) => samples
                        .get_mut(&topology)
                        .expect("autotune candidate sample bucket must exist")
                        .push(ms),
                    Err(err) => {
                        if std::env::var_os(HIERARCHOS_RWKV_POST_MIX_FORWARD_AUTOTUNE_LOG_ENV)
                            .is_some()
                        {
                            eprintln!(
                                "RWKV post-mix forward autotune failed device={} batch={batch} candidate={}: {err:#}; using {}",
                                self.device.name(),
                                topology.label(),
                                structural_default.label()
                            );
                        }
                        return Ok(PostMixForwardDecision {
                            topology: structural_default,
                            autotuned: false,
                        });
                    }
                }
            }
        }
        let mut timings = Vec::with_capacity(candidates.len());
        for topology in candidates {
            let values = samples
                .get_mut(&topology)
                .expect("autotune candidate sample bucket must exist");
            values.sort_by(f64::total_cmp);
            timings.push((topology, values[values.len() / 2]));
        }
        let selected = select_post_mix_forward_topology(&timings, structural_default);
        let decision = PostMixForwardDecision {
            topology: selected,
            autotuned: true,
        };

        if std::env::var_os(HIERARCHOS_RWKV_POST_MIX_FORWARD_AUTOTUNE_LOG_ENV).is_some() {
            let summary = timings
                .iter()
                .map(|(topology, ms)| format!("{}={ms:.5}ms", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV post-mix forward autotune device={} subgroup={} width={} head_size={} batch_pairs={} tail={} {} selected={} autotuned={}",
                self.device.name(),
                subgroup_size,
                self.width,
                self.head_size,
                key.batch_pairs,
                key.has_unpaired_tail,
                summary,
                selected.label(),
                decision.autotuned
            );
        }

        cache
            .lock()
            .map_err(|_| anyhow::anyhow!("RWKV post-mix forward autotune cache lock was poisoned"))?
            .insert(key, decision);
        Ok(decision)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_forward_with_residual(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        tmix: &GpuBuffer,
        r: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        g: &GpuBuffer,
        residual: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        if self.structural_fused_forward_topology().is_some() {
            let decision = self.choose_fused_forward_topology(batch)?;
            return self.record_fused_forward_topology(
                commands,
                batch,
                tmix,
                r,
                k,
                v,
                g,
                residual,
                output,
                decision.topology,
            );
        }
        self.record_gated_forward(commands, batch, tmix, r, k, v, g)?;
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        self.linear_residual_forward.record_dispatch(
            commands,
            &[&self.gated, &self.output_weight, residual, output],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_forward_optional_residual(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        tmix: &GpuBuffer,
        r: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        g: &GpuBuffer,
        residual_output: Option<(&GpuBuffer, &GpuBuffer)>,
    ) -> Result<()> {
        if let Some((residual, output)) = residual_output {
            self.record_forward_with_residual(commands, batch, tmix, r, k, v, g, residual, output)
        } else {
            self.record_forward(commands, batch, tmix, r, k, v, g)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_gated_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        tmix: &GpuBuffer,
        r: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        g: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let norm_push = GroupNormPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
            eps: 64.0e-5,
        };
        if let Some(kernel) = &self.group_norm_bonus_gate_forward_fused {
            return kernel.record_dispatch(
                commands,
                &[
                    tmix,
                    &self.group_norm_weight,
                    &self.group_norm_bias,
                    r,
                    k,
                    v,
                    g,
                    &self.r_k,
                    &self.group_normed,
                    &self.group_mean,
                    &self.group_rstd,
                    &self.gated,
                    &self.bonus_scalar,
                ],
                bytemuck::bytes_of(&norm_push),
                [div_ceil_u32(batch * self.heads, 64), 1, 1],
            );
        }
        self.group_norm_forward.record_dispatch(
            commands,
            &[
                tmix,
                &self.group_norm_weight,
                &self.group_norm_bias,
                &self.group_normed,
                &self.group_mean,
                &self.group_rstd,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(batch * self.heads, 64), 1, 1],
        )?;
        let head_push = HeadPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        self.bonus_gate_forward.record_dispatch(
            commands,
            &[
                &self.group_normed,
                r,
                k,
                v,
                g,
                &self.r_k,
                &self.gated,
                &self.bonus_scalar,
            ],
            bytemuck::bytes_of(&head_push),
            [div_ceil_u32(batch * self.heads, 64), 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        tmix: &GpuBuffer,
        r: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        g: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.width as u32,
        };
        self.linear_weight_grad.record_dispatch(
            commands,
            &[&self.gated, grad_output, &self.grad_output_weight],
            bytemuck::bytes_of(&linear_push),
            [
                div_ceil_u32(self.width, 16),
                div_ceil_u32(self.width, 16),
                1,
            ],
        )?;
        self.linear_input_grad.record_dispatch(
            commands,
            &[grad_output, &self.output_weight, &self.grad_gated],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1],
        )?;
        let head_push = HeadPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        self.bonus_gate_backward.record_dispatch(
            commands,
            &[
                &self.grad_gated,
                &self.group_normed,
                r,
                k,
                v,
                g,
                &self.r_k,
                &self.bonus_scalar,
                &self.grad_group_normed,
                &self.grad_r,
                &self.grad_k,
                &self.grad_v,
                &self.grad_g,
                &self.grad_r_k_partial,
            ],
            bytemuck::bytes_of(&head_push),
            [div_ceil_u32(batch * self.heads, 64), 1, 1],
        )?;
        let norm_push = GroupNormPush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
            eps: 64.0e-5,
        };
        self.group_norm_input_grad.record_dispatch(
            commands,
            &[
                &self.grad_group_normed,
                tmix,
                &self.group_norm_weight,
                &self.group_mean,
                &self.group_rstd,
                &self.grad_tmix,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(batch * self.heads, 64), 1, 1],
        )?;
        self.group_norm_param_grad.record_dispatch(
            commands,
            &[
                &self.grad_group_normed,
                tmix,
                &self.group_mean,
                &self.group_rstd,
                &self.grad_group_norm_weight,
                &self.grad_group_norm_bias,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(self.width, 256), 1, 1],
        )?;
        let reduce_push = ReducePush {
            batch: batch as u32,
            width: self.width as u32,
        };
        self.channel_reduce.record_dispatch(
            commands,
            &[&self.grad_r_k_partial, &self.grad_r_k],
            bytemuck::bytes_of(&reduce_push),
            [div_ceil_u32(self.width, 256), 1, 1],
        )
    }

    pub(crate) fn upload_grad_output(
        &self,
        commands: &mut vulkan::ComputeBatch,
        grad_output: &[f32],
    ) -> Result<()> {
        commands.upload_f32(&self.grad_output, grad_output)
    }

    pub(crate) fn grad_output_buffer(&self) -> &GpuBuffer {
        &self.grad_output
    }

    pub(crate) fn grad_tmix_buffer(&self) -> &GpuBuffer {
        &self.grad_tmix
    }

    pub(crate) fn grad_r_buffer(&self) -> &GpuBuffer {
        &self.grad_r
    }

    pub(crate) fn grad_k_buffer(&self) -> &GpuBuffer {
        &self.grad_k
    }

    pub(crate) fn grad_v_buffer(&self) -> &GpuBuffer {
        &self.grad_v
    }

    pub(crate) fn grad_g_buffer(&self) -> &GpuBuffer {
        &self.grad_g
    }

    pub(crate) fn trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        vec![
            RwkvTrainableRef {
                name: "ln_x.weight",
                parameter: &self.group_norm_weight,
                gradient: &self.grad_group_norm_weight,
                len: self.width,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "ln_x.bias",
                parameter: &self.group_norm_bias,
                gradient: &self.grad_group_norm_bias,
                len: self.width,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "r_k",
                parameter: &self.r_k,
                gradient: &self.grad_r_k,
                len: self.width,
                decay_class: RwkvDecayClass::Decay,
            },
            RwkvTrainableRef {
                name: "output.weight",
                parameter: &self.output_weight,
                gradient: &self.grad_output_weight,
                len: self.width * self.width,
                decay_class: RwkvDecayClass::Decay,
            },
        ]
    }

    pub(crate) fn record_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        commands.readback_f32(&self.output, &self.output_readback, vector_len)?;
        commands.readback_f32(&self.group_normed, &self.group_normed_readback, vector_len)?;
        commands.readback_f32(&self.grad_tmix, &self.grad_tmix_readback, vector_len)?;
        commands.readback_f32(&self.grad_r, &self.grad_r_readback, vector_len)?;
        commands.readback_f32(&self.grad_k, &self.grad_k_readback, vector_len)?;
        commands.readback_f32(&self.grad_v, &self.grad_v_readback, vector_len)?;
        commands.readback_f32(&self.grad_g, &self.grad_g_readback, vector_len)?;
        commands.readback_f32(&self.grad_r_k, &self.grad_r_k_readback, self.width)?;
        commands.readback_f32(
            &self.grad_output_weight,
            &self.grad_output_weight_readback,
            self.width * self.width,
        )?;
        commands.readback_f32(
            &self.grad_group_norm_weight,
            &self.grad_group_norm_weight_readback,
            self.width,
        )?;
        commands.readback_f32(
            &self.grad_group_norm_bias,
            &self.grad_group_norm_bias_readback,
            self.width,
        )
    }

    pub(crate) fn read_result(&self, batch: usize) -> Result<RwkvPostMixResult> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        Ok(RwkvPostMixResult {
            output: self.output_readback.read_f32(vector_len)?,
            group_normed: self.group_normed_readback.read_f32(vector_len)?,
            grad_tmix: self.grad_tmix_readback.read_f32(vector_len)?,
            grad_r: self.grad_r_readback.read_f32(vector_len)?,
            grad_k: self.grad_k_readback.read_f32(vector_len)?,
            grad_v: self.grad_v_readback.read_f32(vector_len)?,
            grad_g: self.grad_g_readback.read_f32(vector_len)?,
            grad_r_k: self.grad_r_k_readback.read_f32(self.width)?,
            grad_output_weight: self
                .grad_output_weight_readback
                .read_f32(self.width * self.width)?,
            grad_group_norm_weight: self.grad_group_norm_weight_readback.read_f32(self.width)?,
            grad_group_norm_bias: self.grad_group_norm_bias_readback.read_f32(self.width)?,
        })
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn head_size(&self) -> usize {
        self.head_size
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn validate_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV post-mix batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_batch_inputs(
        &self,
        batch: usize,
        tmix: &[f32],
        r: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        grad_output: &[f32],
    ) -> Result<usize> {
        self.validate_batch(batch)?;
        let len = batch * self.width;
        for (name, values) in [
            ("tmix", tmix),
            ("r", r),
            ("k", k),
            ("v", v),
            ("g", g),
            ("grad_output", grad_output),
        ] {
            validate_len(name, values, len)?;
        }
        Ok(len)
    }
}

fn read_vector(path: &Path, name: &str, width: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if shape != [width] && shape != [1, width] {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{width}]");
    }
    Ok(values)
}

fn read_matrix(path: &Path, name: &str, width: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if shape != [width, width] {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{width}, {width}]");
    }
    Ok(values)
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "RWKV post-mix {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("RWKV post-mix {name} contains non-finite values");
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_mix_forward_autotune_requires_meaningful_speedup() {
        let structural_default = PostMixForwardTopology::TwoRows;
        assert_eq!(
            select_post_mix_forward_topology(
                &[
                    (PostMixForwardTopology::OneRow, 0.099),
                    (PostMixForwardTopology::TwoRows, 0.100),
                ],
                structural_default,
            ),
            structural_default,
        );
        assert_eq!(
            select_post_mix_forward_topology(
                &[
                    (PostMixForwardTopology::OneRow, 0.097),
                    (PostMixForwardTopology::TwoRows, 0.100),
                ],
                structural_default,
            ),
            PostMixForwardTopology::OneRow,
        );
    }

    #[test]
    fn fused_post_mix_residual_projection_matches_legacy_three_dispatch_chain() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        if !device.supports_storage_buffer_bindings(16)
            || !device.supports_compute_work_group_size_x(256)
        {
            return Ok(());
        }

        let batch = 3usize;
        let width = 8usize;
        let head_size = 4usize;
        let heads = width / head_size;
        let vector_len = batch * width;
        let group_len = batch * heads;
        let group_norm_weight = (0..width)
            .map(|index| 0.8 + index as f32 * 0.05)
            .collect::<Vec<_>>();
        let group_norm_bias = (0..width)
            .map(|index| -0.03 + index as f32 * 0.01)
            .collect::<Vec<_>>();
        let r_k = (0..width)
            .map(|index| -0.15 + index as f32 * 0.035)
            .collect::<Vec<_>>();
        let output_weight = (0..width * width)
            .map(|index| ((index as f32 * 0.19).cos() - 0.25) * 0.13)
            .collect::<Vec<_>>();
        let op = RwkvPostMixOp::new(
            device.clone(),
            width,
            head_size,
            batch,
            &group_norm_weight,
            &group_norm_bias,
            &r_k,
            &output_weight,
        )?;
        assert!(op
            .group_norm_bonus_gate_linear_residual_forward_fused
            .is_some());
        assert!(op
            .group_norm_bonus_gate_linear_residual_forward_fused_two_rows
            .is_some());

        let make_values = |scale: f32, bias: f32| {
            (0..vector_len)
                .map(|index| ((index as f32 * 0.31).sin() + bias) * scale)
                .collect::<Vec<_>>()
        };
        let tmix = GpuBuffer::from_f32(&device, &make_values(0.7, -0.2))?;
        let r = GpuBuffer::from_f32(&device, &make_values(0.6, 0.15))?;
        let k = GpuBuffer::from_f32(&device, &make_values(0.5, -0.25))?;
        let v = GpuBuffer::from_f32(&device, &make_values(0.9, 0.05))?;
        let g = GpuBuffer::from_f32(&device, &make_values(0.4, 1.25))?;
        let residual = GpuBuffer::from_f32(&device, &make_values(0.3, 0.4))?;

        let legacy_normed = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_mean = GpuBuffer::zeros_f32(&device, group_len)?;
        let legacy_rstd = GpuBuffer::zeros_f32(&device, group_len)?;
        let legacy_gated = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_bonus = GpuBuffer::zeros_f32(&device, group_len)?;
        let legacy_output = GpuBuffer::zeros_f32(&device, vector_len)?;
        let fused_output = GpuBuffer::zeros_f32(&device, vector_len)?;
        let norm_push = GroupNormPush {
            batch: batch as u32,
            width: width as u32,
            head_size: head_size as u32,
            eps: 64.0e-5,
        };
        let head_push = HeadPush {
            batch: batch as u32,
            width: width as u32,
            head_size: head_size as u32,
        };
        let linear_push = LinearPush {
            rows: batch as u32,
            input_dim: width as u32,
            output_dim: width as u32,
        };

        let mut legacy_batch = vulkan::ComputeBatch::new(&device)?;
        op.group_norm_forward.record_dispatch(
            &mut legacy_batch,
            &[
                &tmix,
                &op.group_norm_weight,
                &op.group_norm_bias,
                &legacy_normed,
                &legacy_mean,
                &legacy_rstd,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(group_len, 64), 1, 1],
        )?;
        op.bonus_gate_forward.record_dispatch(
            &mut legacy_batch,
            &[
                &legacy_normed,
                &r,
                &k,
                &v,
                &g,
                &op.r_k,
                &legacy_gated,
                &legacy_bonus,
            ],
            bytemuck::bytes_of(&head_push),
            [div_ceil_u32(group_len, 64), 1, 1],
        )?;
        op.linear_residual_forward.record_dispatch(
            &mut legacy_batch,
            &[&legacy_gated, &op.output_weight, &residual, &legacy_output],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(width, 16), div_ceil_u32(batch, 16), 1],
        )?;
        assert_eq!(legacy_batch.dispatch_count(), 3);
        legacy_batch.submit()?;

        let mut fused_batch = vulkan::ComputeBatch::new(&device)?;
        op.record_forward_with_residual(
            &mut fused_batch,
            batch,
            &tmix,
            &r,
            &k,
            &v,
            &g,
            &residual,
            &fused_output,
        )?;
        assert_eq!(fused_batch.dispatch_count(), 1);
        assert_eq!(fused_batch.shader_barrier_count(), 0);
        fused_batch.submit()?;

        for (name, legacy_values, fused_values) in [
            (
                "group_normed",
                legacy_normed.read_f32(vector_len)?,
                op.group_normed.read_f32(vector_len)?,
            ),
            (
                "group_mean",
                legacy_mean.read_f32(group_len)?,
                op.group_mean.read_f32(group_len)?,
            ),
            (
                "group_rstd",
                legacy_rstd.read_f32(group_len)?,
                op.group_rstd.read_f32(group_len)?,
            ),
            (
                "bonus_scalar",
                legacy_bonus.read_f32(group_len)?,
                op.bonus_scalar.read_f32(group_len)?,
            ),
            (
                "gated",
                legacy_gated.read_f32(vector_len)?,
                op.gated.read_f32(vector_len)?,
            ),
            (
                "residual_output",
                legacy_output.read_f32(vector_len)?,
                fused_output.read_f32(vector_len)?,
            ),
        ] {
            let max_abs = legacy_values
                .iter()
                .zip(&fused_values)
                .map(|(legacy, fused)| (legacy - fused).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= 1.0e-6, "fused {name} drifted by {max_abs}");
        }

        Ok(())
    }

    #[test]
    fn fused_group_norm_bonus_gate_matches_legacy_and_removes_dispatch_seam() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        if !device.supports_storage_buffer_bindings(13) {
            return Ok(());
        }

        let batch = 3usize;
        let width = 8usize;
        let head_size = 4usize;
        let heads = width / head_size;
        let vector_len = batch * width;
        let group_len = batch * heads;
        let group_norm_weight = (0..width)
            .map(|index| 0.75 + index as f32 * 0.07)
            .collect::<Vec<_>>();
        let group_norm_bias = (0..width)
            .map(|index| -0.08 + index as f32 * 0.015)
            .collect::<Vec<_>>();
        let r_k = (0..width)
            .map(|index| -0.2 + index as f32 * 0.04)
            .collect::<Vec<_>>();
        let output_weight = vec![0.0; width * width];
        let op = RwkvPostMixOp::new(
            device.clone(),
            width,
            head_size,
            batch,
            &group_norm_weight,
            &group_norm_bias,
            &r_k,
            &output_weight,
        )?;

        let make_values = |scale: f32, bias: f32| {
            (0..vector_len)
                .map(|index| ((index as f32 * 0.37).sin() + bias) * scale)
                .collect::<Vec<_>>()
        };
        let tmix = GpuBuffer::from_f32(&device, &make_values(0.8, -0.1))?;
        let r = GpuBuffer::from_f32(&device, &make_values(0.7, 0.2))?;
        let k = GpuBuffer::from_f32(&device, &make_values(0.6, -0.3))?;
        let v = GpuBuffer::from_f32(&device, &make_values(0.9, 0.1))?;
        let g = GpuBuffer::from_f32(&device, &make_values(0.5, 1.1))?;

        let legacy_normed = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_mean = GpuBuffer::zeros_f32(&device, group_len)?;
        let legacy_rstd = GpuBuffer::zeros_f32(&device, group_len)?;
        let legacy_gated = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_bonus = GpuBuffer::zeros_f32(&device, group_len)?;
        let norm_push = GroupNormPush {
            batch: batch as u32,
            width: width as u32,
            head_size: head_size as u32,
            eps: 64.0e-5,
        };
        let head_push = HeadPush {
            batch: batch as u32,
            width: width as u32,
            head_size: head_size as u32,
        };

        let mut legacy_batch = vulkan::ComputeBatch::new(&device)?;
        op.group_norm_forward.record_dispatch(
            &mut legacy_batch,
            &[
                &tmix,
                &op.group_norm_weight,
                &op.group_norm_bias,
                &legacy_normed,
                &legacy_mean,
                &legacy_rstd,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(group_len, 64), 1, 1],
        )?;
        op.bonus_gate_forward.record_dispatch(
            &mut legacy_batch,
            &[
                &legacy_normed,
                &r,
                &k,
                &v,
                &g,
                &op.r_k,
                &legacy_gated,
                &legacy_bonus,
            ],
            bytemuck::bytes_of(&head_push),
            [div_ceil_u32(group_len, 64), 1, 1],
        )?;
        assert_eq!(legacy_batch.dispatch_count(), 2);
        legacy_batch.submit()?;

        let mut fused_batch = vulkan::ComputeBatch::new(&device)?;
        op.record_gated_forward(&mut fused_batch, batch, &tmix, &r, &k, &v, &g)?;
        assert_eq!(fused_batch.dispatch_count(), 1);
        assert_eq!(fused_batch.shader_barrier_count(), 0);
        fused_batch.submit()?;

        for (name, legacy_values, fused_values) in [
            (
                "group_normed",
                legacy_normed.read_f32(vector_len)?,
                op.group_normed.read_f32(vector_len)?,
            ),
            (
                "group_mean",
                legacy_mean.read_f32(group_len)?,
                op.group_mean.read_f32(group_len)?,
            ),
            (
                "group_rstd",
                legacy_rstd.read_f32(group_len)?,
                op.group_rstd.read_f32(group_len)?,
            ),
            (
                "bonus_scalar",
                legacy_bonus.read_f32(group_len)?,
                op.bonus_scalar.read_f32(group_len)?,
            ),
            (
                "gated",
                legacy_gated.read_f32(vector_len)?,
                op.gated.read_f32(vector_len)?,
            ),
        ] {
            let max_abs = legacy_values
                .iter()
                .zip(&fused_values)
                .map(|(legacy, fused)| (legacy - fused).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= 1.0e-6, "fused {name} drifted by {max_abs}");
        }

        Ok(())
    }
}
