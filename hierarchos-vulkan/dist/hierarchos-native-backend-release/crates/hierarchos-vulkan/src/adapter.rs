use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::rwkv_optimizer::{RwkvDecayClass, RwkvTrainableRef};
use crate::{
    read_f32_tensor, replace_f32_tensors, vulkan, AdamWHyperParams, GpuBuffer, VulkanDevice,
};

const LINEAR_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_forward.spv");
const LAYER_NORM_LINEAR_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_linear_forward_fused.spv");
const LAYER_NORM_LINEAR_SILU_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_linear_silu_forward_fused.spv");
const LAYER_NORM_ADAPTER_FORWARD_FUSED_64_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_adapter_forward_fused_64.spv");
const LAYER_NORM_ADAPTER_FORWARD_FUSED_256_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_adapter_forward_fused_256.spv");
const LAYER_NORM_ADAPTER_FORWARD_FUSED_512_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_adapter_forward_fused_512.spv");
const LINEAR_BIAS_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_bias_forward.spv");
const LINEAR_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_weight_grad.spv");
const LINEAR_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
const LAYER_NORM_FORWARD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_forward.spv");
const LAYER_NORM_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_input_grad.spv");
const SILU_FORWARD_SPV: &[u8] = include_bytes!("../shaders/silu_forward.spv");
const SILU_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/silu_backward.spv");
const BIAS_GRAD_SPV: &[u8] = include_bytes!("../shaders/bias_grad.spv");
const ADAMW_SPV: &[u8] = include_bytes!("../shaders/adamw.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormForwardPush {
    rows: u32,
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormLinearForwardPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormAdapterForwardPush {
    rows: u32,
    input_dim: u32,
    rank: u32,
    output_dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormBackwardPush {
    rows: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VectorPush {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasPush {
    rows: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWPush {
    len: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

#[derive(Debug)]
pub struct AdapterStepResult {
    pub step: u32,
    pub output: Vec<f32>,
    pub input_grad: Vec<f32>,
    pub dispatch_count: usize,
    pub shader_barrier_count: usize,
}

/// Vulkan implementation of coherent-v9's `SharedTokenAdapter`:
///
/// `bias + up(silu(down(layer_norm_affine_free(token_features))))`
///
/// The gradients returned through `input_grad` are the contribution that must
/// be accumulated into the tied token embedding. DeepEmbed callers should pass
/// `matrix_weight_decay = 0.0`; ROSA callers should pass the configured RWKV
/// matrix decay while the adapter bias remains in the vector/no-decay group.
pub struct SharedTokenAdapterTrainer {
    device: VulkanDevice,
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    max_rows: usize,
    step: u32,
    matrix_weight_decay: f32,

    norm_weight: GpuBuffer,
    norm_bias: GpuBuffer,
    down_weight: GpuBuffer,
    up_weight: GpuBuffer,
    bias: GpuBuffer,

    input: GpuBuffer,
    normalized: GpuBuffer,
    norm_mean: GpuBuffer,
    norm_rstd: GpuBuffer,
    down_preact: GpuBuffer,
    hidden: GpuBuffer,
    output: GpuBuffer,
    grad_output: GpuBuffer,
    grad_up_weight: GpuBuffer,
    grad_hidden: GpuBuffer,
    grad_down_preact: GpuBuffer,
    grad_down_weight: GpuBuffer,
    grad_normalized: GpuBuffer,
    grad_input: GpuBuffer,
    grad_bias: GpuBuffer,

    down_exp_avg: GpuBuffer,
    down_exp_avg_sq: GpuBuffer,
    up_exp_avg: GpuBuffer,
    up_exp_avg_sq: GpuBuffer,
    bias_exp_avg: GpuBuffer,
    bias_exp_avg_sq: GpuBuffer,

    output_readback: GpuBuffer,
    grad_input_readback: GpuBuffer,

    layer_norm_adapter_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_linear_silu_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_linear_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_forward: vulkan::ComputeKernel,
    layer_norm_input_grad: vulkan::ComputeKernel,
    linear_forward: vulkan::ComputeKernel,
    linear_bias_forward: vulkan::ComputeKernel,
    linear_weight_grad: vulkan::ComputeKernel,
    linear_input_grad: vulkan::ComputeKernel,
    silu_forward: vulkan::ComputeKernel,
    silu_backward: vulkan::ComputeKernel,
    bias_grad: vulkan::ComputeKernel,
    adamw: vulkan::ComputeKernel,
}

impl SharedTokenAdapterTrainer {
    const LAYER_NORM_EPS: f32 = 1.0e-5;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: VulkanDevice,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        max_rows: usize,
        down_weight: &[f32],
        up_weight: &[f32],
        bias: &[f32],
        matrix_weight_decay: f32,
    ) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 || rank == 0 || max_rows == 0 {
            bail!("SharedTokenAdapter dimensions and max_rows must be positive");
        }
        if !matrix_weight_decay.is_finite() || matrix_weight_decay < 0.0 {
            bail!("adapter matrix weight decay must be finite and non-negative");
        }
        let down_len = rank
            .checked_mul(input_dim)
            .context("adapter down weight size overflow")?;
        let up_len = output_dim
            .checked_mul(rank)
            .context("adapter up weight size overflow")?;
        if down_weight.len() != down_len {
            bail!(
                "adapter down weight has {} values; expected {} for [{}, {}]",
                down_weight.len(),
                down_len,
                rank,
                input_dim
            );
        }
        if up_weight.len() != up_len {
            bail!(
                "adapter up weight has {} values; expected {} for [{}, {}]",
                up_weight.len(),
                up_len,
                output_dim,
                rank
            );
        }
        if bias.len() != output_dim {
            bail!(
                "adapter bias has {} values; expected {}",
                bias.len(),
                output_dim
            );
        }
        if down_weight
            .iter()
            .chain(up_weight)
            .chain(bias)
            .any(|value| !value.is_finite())
        {
            bail!("SharedTokenAdapter parameters contain non-finite values");
        }

        let input_len = max_rows
            .checked_mul(input_dim)
            .context("adapter input capacity overflow")?;
        let hidden_len = max_rows
            .checked_mul(rank)
            .context("adapter hidden capacity overflow")?;
        let output_len = max_rows
            .checked_mul(output_dim)
            .context("adapter output capacity overflow")?;
        let norm_weight_host = vec![1.0f32; input_dim];
        let norm_bias_host = vec![0.0f32; input_dim];
        let adapter_up_fusion_disabled =
            std::env::var_os("HIERARCHOS_VULKAN_DISABLE_ADAPTER_UP_FUSION").is_some();
        let adapter_fused_spv = if !adapter_up_fusion_disabled
            && rank <= 64
            && output_dim <= 512
            && device.supports_storage_buffer_bindings(12)
        {
            if output_dim <= 64 && device.supports_compute_work_group_size_x(64) {
                Some(LAYER_NORM_ADAPTER_FORWARD_FUSED_64_SPV)
            } else if output_dim > 256 && device.supports_compute_work_group_size_x(512) {
                Some(LAYER_NORM_ADAPTER_FORWARD_FUSED_512_SPV)
            } else if device.supports_compute_work_group_size_x(256) {
                Some(LAYER_NORM_ADAPTER_FORWARD_FUSED_256_SPV)
            } else {
                None
            }
        } else {
            None
        };
        let layer_norm_adapter_forward_fused = if let Some(spirv) = adapter_fused_spv {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                spirv,
                &[
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
                std::mem::size_of::<LayerNormAdapterForwardPush>() as u32,
            )?)
        } else {
            None
        };
        let layer_norm_linear_silu_forward_fused = if device.supports_storage_buffer_bindings(9) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_LINEAR_SILU_FORWARD_FUSED_SPV,
                &[
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
                std::mem::size_of::<LayerNormLinearForwardPush>() as u32,
            )?)
        } else {
            None
        };
        let layer_norm_linear_forward_fused = if device.supports_storage_buffer_bindings(8) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_LINEAR_FORWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LayerNormLinearForwardPush>() as u32,
            )?)
        } else {
            None
        };

        Ok(Self {
            layer_norm_adapter_forward_fused,
            layer_norm_linear_silu_forward_fused,
            layer_norm_linear_forward_fused,
            layer_norm_forward: vulkan::ComputeKernel::new(
                &device,
                LAYER_NORM_FORWARD_SPV,
                6,
                std::mem::size_of::<LayerNormForwardPush>() as u32,
            )?,
            layer_norm_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_INPUT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LayerNormBackwardPush>() as u32,
            )?,
            linear_forward: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_FORWARD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear_bias_forward: vulkan::ComputeKernel::new(
                &device,
                LINEAR_BIAS_FORWARD_SPV,
                4,
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
            silu_forward: vulkan::ComputeKernel::new(
                &device,
                SILU_FORWARD_SPV,
                2,
                std::mem::size_of::<VectorPush>() as u32,
            )?,
            silu_backward: vulkan::ComputeKernel::new(
                &device,
                SILU_BACKWARD_SPV,
                3,
                std::mem::size_of::<VectorPush>() as u32,
            )?,
            bias_grad: vulkan::ComputeKernel::new(
                &device,
                BIAS_GRAD_SPV,
                2,
                std::mem::size_of::<BiasPush>() as u32,
            )?,
            adamw: vulkan::ComputeKernel::new(
                &device,
                ADAMW_SPV,
                4,
                std::mem::size_of::<AdamWPush>() as u32,
            )?,
            norm_weight: GpuBuffer::from_f32(&device, &norm_weight_host)?,
            norm_bias: GpuBuffer::from_f32(&device, &norm_bias_host)?,
            down_weight: GpuBuffer::from_f32(&device, down_weight)?,
            up_weight: GpuBuffer::from_f32(&device, up_weight)?,
            bias: GpuBuffer::from_f32(&device, bias)?,
            input: GpuBuffer::zeros_f32(&device, input_len)?,
            normalized: GpuBuffer::zeros_f32(&device, input_len)?,
            norm_mean: GpuBuffer::zeros_f32(&device, max_rows)?,
            norm_rstd: GpuBuffer::zeros_f32(&device, max_rows)?,
            down_preact: GpuBuffer::zeros_f32(&device, hidden_len)?,
            hidden: GpuBuffer::zeros_f32(&device, hidden_len)?,
            output: GpuBuffer::zeros_f32(&device, output_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, output_len)?,
            grad_up_weight: GpuBuffer::zeros_f32(&device, up_len)?,
            grad_hidden: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_down_preact: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_down_weight: GpuBuffer::zeros_f32(&device, down_len)?,
            grad_normalized: GpuBuffer::zeros_f32(&device, input_len)?,
            grad_input: GpuBuffer::zeros_f32(&device, input_len)?,
            grad_bias: GpuBuffer::zeros_f32(&device, output_dim)?,
            down_exp_avg: GpuBuffer::zeros_f32(&device, down_len)?,
            down_exp_avg_sq: GpuBuffer::zeros_f32(&device, down_len)?,
            up_exp_avg: GpuBuffer::zeros_f32(&device, up_len)?,
            up_exp_avg_sq: GpuBuffer::zeros_f32(&device, up_len)?,
            bias_exp_avg: GpuBuffer::zeros_f32(&device, output_dim)?,
            bias_exp_avg_sq: GpuBuffer::zeros_f32(&device, output_dim)?,
            output_readback: GpuBuffer::zeros_host_f32(&device, output_len)?,
            grad_input_readback: GpuBuffer::zeros_host_f32(&device, input_len)?,
            device,
            input_dim,
            output_dim,
            rank,
            max_rows,
            step: 0,
            matrix_weight_decay,
        })
    }

    /// Load one of coherent-v9's real shared-factorized adapter prefixes
    /// (`h_deepembed_adapter`, `l_deepembed_adapter`, or `rosa_adapter`) from a
    /// standard model package without changing tensor names or layouts.
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        max_rows: usize,
        matrix_weight_decay: f32,
    ) -> Result<Self> {
        validate_prefix(prefix)?;
        let tensor_path = model_dir.as_ref().join("model.safetensors");
        let down_name = format!("{prefix}.down.weight");
        let up_name = format!("{prefix}.up.weight");
        let bias_name = format!("{prefix}.bias");
        let (down_shape, down_weight) = read_f32_tensor(&tensor_path, &down_name)?;
        let (up_shape, up_weight) = read_f32_tensor(&tensor_path, &up_name)?;
        let (bias_shape, bias) = read_f32_tensor(&tensor_path, &bias_name)?;
        if down_shape.len() != 2 || up_shape.len() != 2 || bias_shape.len() != 1 {
            bail!(
                "adapter {prefix:?} requires down/up rank-2 and bias rank-1 tensors; got down={down_shape:?} up={up_shape:?} bias={bias_shape:?}"
            );
        }
        let rank = down_shape[0];
        let input_dim = down_shape[1];
        let output_dim = up_shape[0];
        if up_shape[1] != rank || bias_shape[0] != output_dim {
            bail!(
                "adapter {prefix:?} shapes are inconsistent: down={down_shape:?} up={up_shape:?} bias={bias_shape:?}"
            );
        }
        Self::new(
            device,
            input_dim,
            output_dim,
            rank,
            max_rows,
            &down_weight,
            &up_weight,
            &bias,
            matrix_weight_decay,
        )
    }

    /// Record the adapter forward pass into a caller-owned Vulkan command
    /// buffer. This is the composition seam used by the recurrent cell: the
    /// returned DeepEmbed tensor remains device-local and can be bound directly
    /// by channel-mix without a host readback or a second queue submission.
    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        input: &GpuBuffer,
    ) -> Result<()> {
        self.validate_rows(rows)?;
        if let Some(kernel) = &self.layer_norm_adapter_forward_fused {
            let fused_push = LayerNormAdapterForwardPush {
                rows: rows as u32,
                input_dim: self.input_dim as u32,
                rank: self.rank as u32,
                output_dim: self.output_dim as u32,
                eps: Self::LAYER_NORM_EPS,
            };
            kernel.record_dispatch(
                commands,
                &[
                    input,
                    &self.norm_weight,
                    &self.norm_bias,
                    &self.down_weight,
                    &self.up_weight,
                    &self.bias,
                    &self.normalized,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.down_preact,
                    &self.hidden,
                    &self.output,
                ],
                bytemuck::bytes_of(&fused_push),
                [1, rows as u32, 1],
            )?;
            return Ok(());
        }

        let hidden_is_ready = self.record_norm_down_forward(commands, rows, input)?;

        if !hidden_is_ready {
            let hidden_len = rows * self.rank;
            let silu_push = VectorPush {
                len: hidden_len as u32,
            };
            self.silu_forward.record_dispatch(
                commands,
                &[&self.down_preact, &self.hidden],
                bytemuck::bytes_of(&silu_push),
                [div_ceil_u32(hidden_len, 256), 1, 1],
            )?;
        }

        let up_linear = LinearPush {
            rows: rows as u32,
            input_dim: self.rank as u32,
            output_dim: self.output_dim as u32,
        };
        self.linear_bias_forward.record_dispatch(
            commands,
            &[&self.hidden, &self.up_weight, &self.bias, &self.output],
            bytemuck::bytes_of(&up_linear),
            [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1],
        )
    }

    fn record_norm_down_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        input: &GpuBuffer,
    ) -> Result<bool> {
        if let Some(kernel) = &self.layer_norm_linear_silu_forward_fused {
            let fused_push = LayerNormLinearForwardPush {
                rows: rows as u32,
                input_dim: self.input_dim as u32,
                output_dim: self.rank as u32,
                eps: Self::LAYER_NORM_EPS,
            };
            kernel.record_dispatch(
                commands,
                &[
                    input,
                    &self.norm_weight,
                    &self.norm_bias,
                    &self.down_weight,
                    &self.normalized,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.down_preact,
                    &self.hidden,
                ],
                bytemuck::bytes_of(&fused_push),
                [div_ceil_u32(self.rank, 64), rows as u32, 1],
            )?;
            return Ok(true);
        }

        if let Some(kernel) = &self.layer_norm_linear_forward_fused {
            let fused_push = LayerNormLinearForwardPush {
                rows: rows as u32,
                input_dim: self.input_dim as u32,
                output_dim: self.rank as u32,
                eps: Self::LAYER_NORM_EPS,
            };
            kernel.record_dispatch(
                commands,
                &[
                    input,
                    &self.norm_weight,
                    &self.norm_bias,
                    &self.down_weight,
                    &self.normalized,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.down_preact,
                ],
                bytemuck::bytes_of(&fused_push),
                [div_ceil_u32(self.rank, 64), rows as u32, 1],
            )?;
            return Ok(false);
        }

        let norm_forward = LayerNormForwardPush {
            rows: rows as u32,
            dim: self.input_dim as u32,
            eps: Self::LAYER_NORM_EPS,
        };
        self.layer_norm_forward.record_dispatch(
            commands,
            &[
                input,
                &self.norm_weight,
                &self.norm_bias,
                &self.normalized,
                &self.norm_mean,
                &self.norm_rstd,
            ],
            bytemuck::bytes_of(&norm_forward),
            [div_ceil_u32(rows, 64), 1, 1],
        )?;

        let down_linear = LinearPush {
            rows: rows as u32,
            input_dim: self.input_dim as u32,
            output_dim: self.rank as u32,
        };
        self.linear_forward.record_dispatch(
            commands,
            &[&self.normalized, &self.down_weight, &self.down_preact],
            bytemuck::bytes_of(&down_linear),
            [div_ceil_u32(self.rank, 16), div_ceil_u32(rows, 16), 1],
        )?;
        Ok(false)
    }

    /// Record the adapter backward pass against a caller-owned gradient
    /// buffer. In the fused DeepEmbed/channel-mix path `grad_output` is the
    /// channel-mix `grad_deepembed` buffer, so no activation-gradient shuttle
    /// through Python or CPU memory is required.
    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        input: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Result<()> {
        self.validate_rows(rows)?;
        let bias_push = BiasPush {
            rows: rows as u32,
            dim: self.output_dim as u32,
        };
        self.bias_grad.record_dispatch(
            commands,
            &[grad_output, &self.grad_bias],
            bytemuck::bytes_of(&bias_push),
            [div_ceil_u32(self.output_dim, 256), 1, 1],
        )?;

        let up_linear = LinearPush {
            rows: rows as u32,
            input_dim: self.rank as u32,
            output_dim: self.output_dim as u32,
        };
        self.linear_weight_grad.record_dispatch(
            commands,
            &[&self.hidden, grad_output, &self.grad_up_weight],
            bytemuck::bytes_of(&up_linear),
            [
                div_ceil_u32(self.rank, 16),
                div_ceil_u32(self.output_dim, 16),
                1,
            ],
        )?;
        self.linear_input_grad.record_dispatch(
            commands,
            &[grad_output, &self.up_weight, &self.grad_hidden],
            bytemuck::bytes_of(&up_linear),
            [div_ceil_u32(self.rank, 16), div_ceil_u32(rows, 16), 1],
        )?;

        let hidden_len = rows * self.rank;
        let silu_push = VectorPush {
            len: hidden_len as u32,
        };
        self.silu_backward.record_dispatch(
            commands,
            &[&self.grad_hidden, &self.down_preact, &self.grad_down_preact],
            bytemuck::bytes_of(&silu_push),
            [div_ceil_u32(hidden_len, 256), 1, 1],
        )?;

        let down_linear = LinearPush {
            rows: rows as u32,
            input_dim: self.input_dim as u32,
            output_dim: self.rank as u32,
        };
        self.linear_weight_grad.record_dispatch(
            commands,
            &[
                &self.normalized,
                &self.grad_down_preact,
                &self.grad_down_weight,
            ],
            bytemuck::bytes_of(&down_linear),
            [
                div_ceil_u32(self.input_dim, 16),
                div_ceil_u32(self.rank, 16),
                1,
            ],
        )?;
        self.linear_input_grad.record_dispatch(
            commands,
            &[
                &self.grad_down_preact,
                &self.down_weight,
                &self.grad_normalized,
            ],
            bytemuck::bytes_of(&down_linear),
            [div_ceil_u32(self.input_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;

        let norm_backward = LayerNormBackwardPush {
            rows: rows as u32,
            dim: self.input_dim as u32,
        };
        self.layer_norm_input_grad.record_dispatch(
            commands,
            &[
                &self.grad_normalized,
                input,
                &self.norm_weight,
                &self.norm_mean,
                &self.norm_rstd,
                &self.grad_input,
            ],
            bytemuck::bytes_of(&norm_backward),
            [div_ceil_u32(rows, 64), 1, 1],
        )
    }

    pub(crate) fn record_grad_input_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
    ) -> Result<()> {
        self.validate_rows(rows)?;
        commands.readback_f32(
            &self.grad_input,
            &self.grad_input_readback,
            rows * self.input_dim,
        )
    }

    pub(crate) fn read_grad_input(&self, rows: usize) -> Result<Vec<f32>> {
        self.validate_rows(rows)?;
        self.grad_input_readback.read_f32(rows * self.input_dim)
    }

    pub(crate) fn output_buffer(&self) -> &GpuBuffer {
        &self.output
    }

    pub(crate) fn grad_input_buffer(&self) -> &GpuBuffer {
        &self.grad_input
    }

    pub(crate) fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub(crate) fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub(crate) fn deepembed_trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        vec![
            RwkvTrainableRef {
                name: "deepembed.down.weight",
                parameter: &self.down_weight,
                gradient: &self.grad_down_weight,
                len: self.rank * self.input_dim,
                // `build_hierarchos_optimizer` v2 classifies factorized
                // DeepEmbed adapter matrices by dimensionality. The legacy
                // vocabulary-sized `h_deepemb`/`l_deepemb` tables have their
                // own special no-decay rule, but these ordinary rank matrices
                // are AdamW-decayed on the PyTorch training path.
                decay_class: RwkvDecayClass::Decay,
            },
            RwkvTrainableRef {
                name: "deepembed.up.weight",
                parameter: &self.up_weight,
                gradient: &self.grad_up_weight,
                len: self.output_dim * self.rank,
                decay_class: RwkvDecayClass::Decay,
            },
            RwkvTrainableRef {
                name: "deepembed.bias",
                parameter: &self.bias,
                gradient: &self.grad_bias,
                len: self.output_dim,
                decay_class: RwkvDecayClass::NoDecay,
            },
        ]
    }

    fn validate_rows(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.max_rows {
            bail!(
                "adapter batch has {rows} rows; capacity is 1..={}",
                self.max_rows
            );
        }
        Ok(())
    }

    pub fn train_step(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<AdapterStepResult> {
        hyper.validate()?;
        if !input.len().is_multiple_of(self.input_dim) {
            bail!(
                "adapter input length {} is not divisible by input_dim {}",
                input.len(),
                self.input_dim
            );
        }
        let rows = input.len() / self.input_dim;
        if rows == 0 || rows > self.max_rows {
            bail!(
                "adapter batch has {rows} rows; capacity is 1..={}",
                self.max_rows
            );
        }
        let expected_grad_output = rows
            .checked_mul(self.output_dim)
            .context("adapter gradient size overflow")?;
        if grad_output.len() != expected_grad_output {
            bail!(
                "adapter grad_output has {} values; expected {} for [{}, {}]",
                grad_output.len(),
                expected_grad_output,
                rows,
                self.output_dim
            );
        }
        if input
            .iter()
            .chain(grad_output)
            .any(|value| !value.is_finite())
        {
            bail!("adapter input/gradient contains non-finite values");
        }

        let mut batch = vulkan::ComputeBatch::new(&self.device)?;
        batch.upload_f32(&self.input, input)?;
        batch.upload_f32(&self.grad_output, grad_output)?;

        self.record_forward(&mut batch, rows, &self.input)?;
        self.record_backward(&mut batch, rows, &self.input, &self.grad_output)?;

        let next_step = self.step.checked_add(1).context("AdamW step overflow")?;
        let down_adam = AdamWPush {
            len: (self.rank * self.input_dim) as u32,
            step: next_step,
            lr: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay: self.matrix_weight_decay,
        };
        self.adamw.record_dispatch(
            &mut batch,
            &[
                &self.down_weight,
                &self.grad_down_weight,
                &self.down_exp_avg,
                &self.down_exp_avg_sq,
            ],
            bytemuck::bytes_of(&down_adam),
            [div_ceil_u32(self.rank * self.input_dim, 256), 1, 1],
        )?;
        let up_adam = AdamWPush {
            len: (self.output_dim * self.rank) as u32,
            step: next_step,
            lr: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay: self.matrix_weight_decay,
        };
        self.adamw.record_dispatch(
            &mut batch,
            &[
                &self.up_weight,
                &self.grad_up_weight,
                &self.up_exp_avg,
                &self.up_exp_avg_sq,
            ],
            bytemuck::bytes_of(&up_adam),
            [div_ceil_u32(self.output_dim * self.rank, 256), 1, 1],
        )?;
        let bias_adam = AdamWPush {
            len: self.output_dim as u32,
            step: next_step,
            lr: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay: 0.0,
        };
        self.adamw.record_dispatch(
            &mut batch,
            &[
                &self.bias,
                &self.grad_bias,
                &self.bias_exp_avg,
                &self.bias_exp_avg_sq,
            ],
            bytemuck::bytes_of(&bias_adam),
            [div_ceil_u32(self.output_dim, 256), 1, 1],
        )?;

        let dispatch_count = batch.dispatch_count();
        let shader_barrier_count = batch.shader_barrier_count();
        batch.readback_f32(&self.output, &self.output_readback, rows * self.output_dim)?;
        batch.readback_f32(
            &self.grad_input,
            &self.grad_input_readback,
            rows * self.input_dim,
        )?;
        batch.submit()?;
        self.step = next_step;

        Ok(AdapterStepResult {
            step: self.step,
            output: self.output_readback.read_f32(rows * self.output_dim)?,
            input_grad: self.grad_input_readback.read_f32(rows * self.input_dim)?,
            dispatch_count,
            shader_barrier_count,
        })
    }

    pub fn down_weights(&self) -> Result<Vec<f32>> {
        self.down_weight.read_f32(self.rank * self.input_dim)
    }

    pub fn up_weights(&self) -> Result<Vec<f32>> {
        self.up_weight.read_f32(self.output_dim * self.rank)
    }

    pub fn bias(&self) -> Result<Vec<f32>> {
        self.bias.read_f32(self.output_dim)
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Export updated adapter parameters back under their original PyTorch
    /// names while preserving every unrelated tensor and package sidecar.
    pub fn export_model_package(
        &self,
        source_model_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        prefix: &str,
    ) -> Result<()> {
        validate_prefix(prefix)?;
        let source_model_dir = source_model_dir.as_ref();
        let output_dir = output_dir.as_ref();
        if source_model_dir == output_dir {
            bail!("export_model_package requires a distinct output directory");
        }
        std::fs::create_dir_all(output_dir)?;
        for entry in std::fs::read_dir(source_model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path.file_name().and_then(|name| name.to_str()) != Some("model.safetensors")
            {
                std::fs::copy(&path, output_dir.join(entry.file_name()))?;
            }
        }

        let down_weight = self.down_weights()?;
        let up_weight = self.up_weights()?;
        let bias = self.bias()?;
        let down_name = format!("{prefix}.down.weight");
        let up_name = format!("{prefix}.up.weight");
        let bias_name = format!("{prefix}.bias");
        let down_shape = [self.rank, self.input_dim];
        let up_shape = [self.output_dim, self.rank];
        let bias_shape = [self.output_dim];
        replace_f32_tensors(
            &source_model_dir.join("model.safetensors"),
            &output_dir.join("model.safetensors"),
            &[
                (&down_name, &down_shape, &down_weight),
                (&up_name, &up_shape, &up_weight),
                (&bias_name, &bias_shape, &bias),
            ],
        )?;
        Ok(())
    }
}

fn validate_prefix(prefix: &str) -> Result<()> {
    if prefix.trim().is_empty() {
        bail!("adapter tensor prefix must not be empty");
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
