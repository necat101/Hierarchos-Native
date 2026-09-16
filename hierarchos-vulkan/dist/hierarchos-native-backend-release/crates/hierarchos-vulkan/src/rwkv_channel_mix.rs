use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::rwkv_optimizer::{RwkvDecayClass, RwkvTrainableRef};
use crate::{read_f32_tensor, vulkan, GpuBuffer, VulkanDevice};

const LAYER_NORM_CHANNEL_MIX_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_channel_mix_forward_fused.spv");
const LAYER_NORM_CHANNEL_MIX_KEY_RELU2_DEEPEMBED_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_channel_mix_key_relu2_deepembed_forward_fused.spv");
const LAYER_NORM_CHANNEL_MIX_FULL_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_channel_mix_full_forward_fused.spv");
const LAYER_NORM_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_input_grad.spv");
const LAYER_NORM_PARAM_GRAD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_param_grad.spv");
const LAYER_NORM_BACKWARD_FUSED_ADD_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_backward_fused_add.spv");
const LAYER_NORM_BACKWARD_FUSED_ADD_RESIDUAL_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_backward_fused_add_residual.spv");
const CHANNEL_MIX_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_channel_mix_backward.spv");
const RELU2_DEEPEMBED_FORWARD_SPV: &[u8] = include_bytes!("../shaders/relu2_deepembed_forward.spv");
const RELU2_DEEPEMBED_BACKWARD_SPV: &[u8] =
    include_bytes!("../shaders/relu2_deepembed_backward.spv");
const LINEAR_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_forward.spv");
const LINEAR_RESIDUAL_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_residual_forward.spv");
const LINEAR_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
const LINEAR_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_weight_grad.spv");
const VECTOR_ADD_SPV: &[u8] = include_bytes!("../shaders/vector_add.spv");
const LAYER_NORM_FUSED_ADD_ACCESS: [vulkan::BindingAccess; 9] = [
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::MayWrite,
    vulkan::BindingAccess::MayWrite,
    vulkan::BindingAccess::MayWrite,
];
const LAYER_NORM_CHANNEL_MIX_FORWARD_FUSED_ACCESS: [vulkan::BindingAccess; 9] = [
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::ReadOnly,
    vulkan::BindingAccess::MayWrite,
    vulkan::BindingAccess::MayWrite,
    vulkan::BindingAccess::MayWrite,
    vulkan::BindingAccess::MayWrite,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormForwardPush {
    rows: u32,
    dim: u32,
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
struct MixPush {
    batch: u32,
    width: u32,
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
struct ActivationPush {
    len: u32,
    key_clamp: f32,
    deepembed_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChannelMixProducerPush {
    rows: u32,
    width: u32,
    hidden_width: u32,
    eps: f32,
    key_clamp: f32,
    deepembed_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

#[derive(Debug)]
pub struct RwkvChannelMixResult {
    pub output: Vec<f32>,
    pub grad_x: Vec<f32>,
    pub grad_previous: Vec<f32>,
    pub grad_deepembed: Vec<f32>,
    pub grad_mix_k: Vec<f32>,
    pub grad_key_weight: Vec<f32>,
    pub grad_value_weight: Vec<f32>,
    pub grad_layer_norm_weight: Vec<f32>,
    pub grad_layer_norm_bias: Vec<f32>,
}

/// Vulkan-native coherent-v9 channel-mix slice:
///
/// `ln2(x) -> mix(x_k_cm, previous) -> key_cm -> ReLU^2`
/// `-> DeepEmbed multiply -> value_cm -> residual add`.
///
/// DeepEmbed is represented as an input buffer of width `4*C`. Passing an
/// all-ones vector exactly selects the no-DeepEmbed path, while a future fused
/// caller can bind the output buffer of `SharedTokenAdapter` directly without
/// changing this graph or checkpoint layout.
pub struct RwkvChannelMixOp {
    device: VulkanDevice,
    width: usize,
    hidden_width: usize,
    max_batch: usize,
    key_clamp: f32,
    deepembed_clamp: f32,

    layer_norm_weight: GpuBuffer,
    layer_norm_bias: GpuBuffer,
    mix_k: GpuBuffer,
    key_weight: GpuBuffer,
    value_weight: GpuBuffer,

    x: GpuBuffer,
    previous: GpuBuffer,
    deepembed: GpuBuffer,
    grad_output: GpuBuffer,
    normalized: GpuBuffer,
    norm_mean: GpuBuffer,
    norm_rstd: GpuBuffer,
    mixed: GpuBuffer,
    cm_key: GpuBuffer,
    ffn: GpuBuffer,
    output: GpuBuffer,
    grad_ffn: GpuBuffer,
    grad_cm_key: GpuBuffer,
    grad_deepembed: GpuBuffer,
    grad_mixed: GpuBuffer,
    grad_normalized: GpuBuffer,
    grad_previous: GpuBuffer,
    grad_mix_k: GpuBuffer,
    grad_layer_norm_input: GpuBuffer,
    grad_x: GpuBuffer,
    grad_key_weight: GpuBuffer,
    grad_value_weight: GpuBuffer,
    grad_layer_norm_weight: GpuBuffer,
    grad_layer_norm_bias: GpuBuffer,

    output_readback: GpuBuffer,
    grad_x_readback: GpuBuffer,
    grad_previous_readback: GpuBuffer,
    grad_deepembed_readback: GpuBuffer,
    grad_mix_k_readback: GpuBuffer,
    grad_key_weight_readback: GpuBuffer,
    grad_value_weight_readback: GpuBuffer,
    grad_layer_norm_weight_readback: GpuBuffer,
    grad_layer_norm_bias_readback: GpuBuffer,

    layer_norm_channel_mix_forward_fused: vulkan::ComputeKernel,
    layer_norm_channel_mix_key_relu2_deepembed_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_channel_mix_full_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_input_grad: vulkan::ComputeKernel,
    layer_norm_param_grad: vulkan::ComputeKernel,
    layer_norm_backward_fused_add: vulkan::ComputeKernel,
    layer_norm_backward_fused_add_residual: Option<vulkan::ComputeKernel>,
    channel_mix_backward: vulkan::ComputeKernel,
    activation_forward: vulkan::ComputeKernel,
    activation_backward: vulkan::ComputeKernel,
    linear_forward: vulkan::ComputeKernel,
    linear_residual_forward: vulkan::ComputeKernel,
    linear_input_grad: vulkan::ComputeKernel,
    linear_weight_grad: vulkan::ComputeKernel,
    vector_add: vulkan::ComputeKernel,
}

impl RwkvChannelMixOp {
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        max_batch: usize,
        key_clamp: f32,
        deepembed_clamp: f32,
    ) -> Result<Self> {
        if prefix.trim().is_empty() {
            bail!("RWKV tensor prefix must not be empty");
        }
        let path = model_dir.as_ref().join("model.safetensors");
        let (norm_shape, layer_norm_weight) =
            read_f32_tensor(&path, &format!("{prefix}.ln2.weight"))?;
        let width = match norm_shape.as_slice() {
            [width] if *width > 0 => *width,
            _ => bail!("RWKV tensor {prefix}.ln2.weight must have shape [C], got {norm_shape:?}"),
        };
        let layer_norm_bias = read_vector(&path, &format!("{prefix}.ln2.bias"), width)?;
        let mix_k = read_vector(&path, &format!("{prefix}.x_k_cm"), width)?;
        let key_weight = read_matrix(&path, &format!("{prefix}.key_cm.weight"), width * 4, width)?;
        let value_weight = read_matrix(
            &path,
            &format!("{prefix}.value_cm.weight"),
            width,
            width * 4,
        )?;
        Self::new(
            device,
            width,
            max_batch,
            &layer_norm_weight,
            &layer_norm_bias,
            &mix_k,
            &key_weight,
            &value_weight,
            key_clamp,
            deepembed_clamp,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: VulkanDevice,
        width: usize,
        max_batch: usize,
        layer_norm_weight: &[f32],
        layer_norm_bias: &[f32],
        mix_k: &[f32],
        key_weight: &[f32],
        value_weight: &[f32],
        key_clamp: f32,
        deepembed_clamp: f32,
    ) -> Result<Self> {
        if width == 0 || max_batch == 0 {
            bail!("RWKV channel-mix width and max_batch must be positive");
        }
        if !key_clamp.is_finite() || key_clamp < 0.0 {
            bail!("RWKV channel-mix key clamp must be finite and non-negative");
        }
        if !deepembed_clamp.is_finite() || deepembed_clamp < 0.0 {
            bail!("RWKV channel-mix DeepEmbed clamp must be finite and non-negative");
        }
        let hidden_width = width
            .checked_mul(4)
            .context("channel-mix hidden width overflow")?;
        validate_len("layer_norm_weight", layer_norm_weight, width)?;
        validate_len("layer_norm_bias", layer_norm_bias, width)?;
        validate_len("mix_k", mix_k, width)?;
        validate_len("key_weight", key_weight, hidden_width * width)?;
        validate_len("value_weight", value_weight, width * hidden_width)?;
        let vector_len = max_batch
            .checked_mul(width)
            .context("channel-mix vector capacity overflow")?;
        let hidden_len = max_batch
            .checked_mul(hidden_width)
            .context("channel-mix hidden capacity overflow")?;
        let matrix_len = width
            .checked_mul(hidden_width)
            .context("channel-mix matrix size overflow")?;

        let layer_norm_channel_mix_key_relu2_deepembed_forward_fused = if width <= 512
            && device.supports_storage_buffer_bindings(13)
            && device.supports_compute_work_group_size_x(256)
        {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_CHANNEL_MIX_KEY_RELU2_DEEPEMBED_FORWARD_FUSED_SPV,
                &[
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
                std::mem::size_of::<ChannelMixProducerPush>() as u32,
            )?)
        } else {
            None
        };
        let layer_norm_channel_mix_full_forward_fused = if width <= 512
            && device.supports_storage_buffer_bindings(15)
            && device.supports_compute_work_group_size_x(256)
        {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_CHANNEL_MIX_FULL_FORWARD_FUSED_SPV,
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
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<ChannelMixProducerPush>() as u32,
            )?)
        } else {
            None
        };

        Ok(Self {
            layer_norm_channel_mix_forward_fused: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_CHANNEL_MIX_FORWARD_FUSED_SPV,
                &LAYER_NORM_CHANNEL_MIX_FORWARD_FUSED_ACCESS,
                std::mem::size_of::<LayerNormForwardPush>() as u32,
            )?,
            layer_norm_channel_mix_key_relu2_deepembed_forward_fused,
            layer_norm_channel_mix_full_forward_fused,
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
            layer_norm_param_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_PARAM_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LayerNormBackwardPush>() as u32,
            )?,
            layer_norm_backward_fused_add: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_BACKWARD_FUSED_ADD_SPV,
                &LAYER_NORM_FUSED_ADD_ACCESS,
                std::mem::size_of::<LayerNormBackwardPush>() as u32,
            )?,
            layer_norm_backward_fused_add_residual: if device.supports_storage_buffer_bindings(10) {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    LAYER_NORM_BACKWARD_FUSED_ADD_RESIDUAL_SPV,
                    &[
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
                    ],
                    std::mem::size_of::<LayerNormBackwardPush>() as u32,
                )?)
            } else {
                None
            },
            channel_mix_backward: vulkan::ComputeKernel::new(
                &device,
                CHANNEL_MIX_BACKWARD_SPV,
                7,
                std::mem::size_of::<MixPush>() as u32,
            )?,
            activation_forward: vulkan::ComputeKernel::new(
                &device,
                RELU2_DEEPEMBED_FORWARD_SPV,
                3,
                std::mem::size_of::<ActivationPush>() as u32,
            )?,
            activation_backward: vulkan::ComputeKernel::new(
                &device,
                RELU2_DEEPEMBED_BACKWARD_SPV,
                5,
                std::mem::size_of::<ActivationPush>() as u32,
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
            vector_add: vulkan::ComputeKernel::new_with_access(
                &device,
                VECTOR_ADD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            layer_norm_weight: GpuBuffer::from_f32(&device, layer_norm_weight)?,
            layer_norm_bias: GpuBuffer::from_f32(&device, layer_norm_bias)?,
            mix_k: GpuBuffer::from_f32(&device, mix_k)?,
            key_weight: GpuBuffer::from_f32(&device, key_weight)?,
            value_weight: GpuBuffer::from_f32(&device, value_weight)?,
            x: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            deepembed: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            normalized: GpuBuffer::zeros_f32(&device, vector_len)?,
            norm_mean: GpuBuffer::zeros_f32(&device, max_batch)?,
            norm_rstd: GpuBuffer::zeros_f32(&device, max_batch)?,
            mixed: GpuBuffer::zeros_f32(&device, vector_len)?,
            cm_key: GpuBuffer::zeros_f32(&device, hidden_len)?,
            ffn: GpuBuffer::zeros_f32(&device, hidden_len)?,
            output: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_ffn: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_cm_key: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_deepembed: GpuBuffer::zeros_f32(&device, hidden_len)?,
            grad_mixed: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_normalized: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_mix_k: GpuBuffer::zeros_f32(&device, width)?,
            grad_layer_norm_input: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_x: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_key_weight: GpuBuffer::zeros_f32(&device, matrix_len)?,
            grad_value_weight: GpuBuffer::zeros_f32(&device, matrix_len)?,
            grad_layer_norm_weight: GpuBuffer::zeros_f32(&device, width)?,
            grad_layer_norm_bias: GpuBuffer::zeros_f32(&device, width)?,
            output_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_x_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_previous_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_deepembed_readback: GpuBuffer::zeros_host_f32(&device, hidden_len)?,
            grad_mix_k_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_key_weight_readback: GpuBuffer::zeros_host_f32(&device, matrix_len)?,
            grad_value_weight_readback: GpuBuffer::zeros_host_f32(&device, matrix_len)?,
            grad_layer_norm_weight_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_layer_norm_bias_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            device,
            width,
            hidden_width,
            max_batch,
            key_clamp,
            deepembed_clamp,
        })
    }

    pub fn forward_backward(
        &mut self,
        batch: usize,
        x: &[f32],
        previous: &[f32],
        deepembed: &[f32],
        grad_output: &[f32],
    ) -> Result<RwkvChannelMixResult> {
        self.validate_inputs(batch, x, previous, deepembed, grad_output)?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.x, x)?;
        commands.upload_f32(&self.previous, previous)?;
        commands.upload_f32(&self.deepembed, deepembed)?;
        commands.upload_f32(&self.grad_output, grad_output)?;
        self.record_forward(
            &mut commands,
            batch,
            &self.x,
            &self.previous,
            &self.deepembed,
        )?;
        self.record_backward(
            &mut commands,
            batch,
            &self.x,
            &self.previous,
            &self.deepembed,
            &self.grad_output,
        )?;
        self.record_readback(&mut commands, batch)?;
        commands.submit()?;
        self.read_result(batch)
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn hidden_width(&self) -> usize {
        self.hidden_width
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub(crate) fn output_buffer(&self) -> &GpuBuffer {
        &self.output
    }

    pub(crate) fn grad_x_buffer(&self) -> &GpuBuffer {
        &self.grad_x
    }

    pub(crate) fn grad_previous_buffer(&self) -> &GpuBuffer {
        &self.grad_previous
    }

    pub(crate) fn grad_deepembed_buffer(&self) -> &GpuBuffer {
        &self.grad_deepembed
    }

    pub(crate) fn normalized_buffer(&self) -> &GpuBuffer {
        &self.normalized
    }

    pub(crate) fn supports_full_forward_fusion(&self) -> bool {
        self.layer_norm_channel_mix_full_forward_fused.is_some()
    }

    pub(crate) fn layer_norm_weight_buffer(&self) -> &GpuBuffer {
        &self.layer_norm_weight
    }

    pub(crate) fn layer_norm_bias_buffer(&self) -> &GpuBuffer {
        &self.layer_norm_bias
    }

    pub(crate) fn mix_k_buffer(&self) -> &GpuBuffer {
        &self.mix_k
    }

    pub(crate) fn key_weight_buffer(&self) -> &GpuBuffer {
        &self.key_weight
    }

    pub(crate) fn value_weight_buffer(&self) -> &GpuBuffer {
        &self.value_weight
    }

    pub(crate) fn norm_mean_buffer(&self) -> &GpuBuffer {
        &self.norm_mean
    }

    pub(crate) fn norm_rstd_buffer(&self) -> &GpuBuffer {
        &self.norm_rstd
    }

    pub(crate) fn mixed_buffer(&self) -> &GpuBuffer {
        &self.mixed
    }

    pub(crate) fn key_buffer(&self) -> &GpuBuffer {
        &self.cm_key
    }

    pub(crate) fn ffn_buffer(&self) -> &GpuBuffer {
        &self.ffn
    }

    pub(crate) fn key_clamp(&self) -> f32 {
        self.key_clamp
    }

    pub(crate) fn deepembed_clamp(&self) -> f32 {
        self.deepembed_clamp
    }

    pub(crate) fn trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        vec![
            RwkvTrainableRef {
                name: "ln2.weight",
                parameter: &self.layer_norm_weight,
                gradient: &self.grad_layer_norm_weight,
                len: self.width,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "ln2.bias",
                parameter: &self.layer_norm_bias,
                gradient: &self.grad_layer_norm_bias,
                len: self.width,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "x_k_cm",
                parameter: &self.mix_k,
                gradient: &self.grad_mix_k,
                len: self.width,
                decay_class: RwkvDecayClass::Decay,
            },
            RwkvTrainableRef {
                name: "key_cm.weight",
                parameter: &self.key_weight,
                gradient: &self.grad_key_weight,
                len: self.hidden_width * self.width,
                decay_class: RwkvDecayClass::Decay,
            },
            RwkvTrainableRef {
                name: "value_cm.weight",
                parameter: &self.value_weight,
                gradient: &self.grad_value_weight,
                len: self.width * self.hidden_width,
                decay_class: RwkvDecayClass::Decay,
            },
        ]
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous: &GpuBuffer,
        deepembed: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        if let Some(kernel) = &self.layer_norm_channel_mix_full_forward_fused {
            let push = ChannelMixProducerPush {
                rows: batch as u32,
                width: self.width as u32,
                hidden_width: self.hidden_width as u32,
                eps: 1.0e-5,
                key_clamp: self.key_clamp,
                deepembed_clamp: self.deepembed_clamp,
            };
            return kernel.record_dispatch(
                commands,
                &[
                    x,
                    &self.layer_norm_weight,
                    &self.layer_norm_bias,
                    previous,
                    &self.mix_k,
                    &self.key_weight,
                    deepembed,
                    &self.value_weight,
                    &self.normalized,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.mixed,
                    &self.cm_key,
                    &self.ffn,
                    &self.output,
                ],
                bytemuck::bytes_of(&push),
                [batch as u32, 1, 1],
            );
        }
        self.record_forward_producer(commands, batch, x, previous, deepembed)?;
        let value_push = LinearPush {
            rows: batch as u32,
            input_dim: self.hidden_width as u32,
            output_dim: self.width as u32,
        };
        self.linear_residual_forward.record_dispatch(
            commands,
            &[&self.ffn, &self.value_weight, x, &self.output],
            bytemuck::bytes_of(&value_push),
            [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1],
        )
    }

    fn record_forward_producer(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous: &GpuBuffer,
        deepembed: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        if let Some(kernel) = &self.layer_norm_channel_mix_key_relu2_deepembed_forward_fused {
            let push = ChannelMixProducerPush {
                rows: batch as u32,
                width: self.width as u32,
                hidden_width: self.hidden_width as u32,
                eps: 1.0e-5,
                key_clamp: self.key_clamp,
                deepembed_clamp: self.deepembed_clamp,
            };
            return kernel.record_dispatch(
                commands,
                &[
                    x,
                    &self.layer_norm_weight,
                    &self.layer_norm_bias,
                    previous,
                    &self.mix_k,
                    &self.key_weight,
                    deepembed,
                    &self.normalized,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.mixed,
                    &self.cm_key,
                    &self.ffn,
                ],
                bytemuck::bytes_of(&push),
                [batch as u32, 1, 1],
            );
        }

        let hidden_len = batch * self.hidden_width;
        let norm_push = LayerNormForwardPush {
            rows: batch as u32,
            dim: self.width as u32,
            eps: 1.0e-5,
        };
        self.layer_norm_channel_mix_forward_fused.record_dispatch(
            commands,
            &[
                x,
                &self.layer_norm_weight,
                &self.layer_norm_bias,
                previous,
                &self.mix_k,
                &self.normalized,
                &self.norm_mean,
                &self.norm_rstd,
                &self.mixed,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(batch, 64), 1, 1],
        )?;
        let key_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.hidden_width as u32,
        };
        self.linear_forward.record_dispatch(
            commands,
            &[&self.mixed, &self.key_weight, &self.cm_key],
            bytemuck::bytes_of(&key_push),
            [
                div_ceil_u32(self.hidden_width, 16),
                div_ceil_u32(batch, 16),
                1,
            ],
        )?;
        let activation_push = ActivationPush {
            len: hidden_len as u32,
            key_clamp: self.key_clamp,
            deepembed_clamp: self.deepembed_clamp,
        };
        self.activation_forward.record_dispatch(
            commands,
            &[&self.cm_key, deepembed, &self.ffn],
            bytemuck::bytes_of(&activation_push),
            [div_ceil_u32(hidden_len, 256), 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous: &GpuBuffer,
        deepembed: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Result<()> {
        self.record_backward_with_normalized_grad(
            commands,
            batch,
            x,
            previous,
            deepembed,
            grad_output,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward_with_normalized_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous: &GpuBuffer,
        deepembed: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_normalized_external: Option<&GpuBuffer>,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let hidden_len = batch * self.hidden_width;
        let value_push = LinearPush {
            rows: batch as u32,
            input_dim: self.hidden_width as u32,
            output_dim: self.width as u32,
        };
        self.linear_weight_grad.record_dispatch(
            commands,
            &[&self.ffn, grad_output, &self.grad_value_weight],
            bytemuck::bytes_of(&value_push),
            [
                div_ceil_u32(self.hidden_width, 16),
                div_ceil_u32(self.width, 16),
                1,
            ],
        )?;
        self.linear_input_grad.record_dispatch(
            commands,
            &[grad_output, &self.value_weight, &self.grad_ffn],
            bytemuck::bytes_of(&value_push),
            [
                div_ceil_u32(self.hidden_width, 16),
                div_ceil_u32(batch, 16),
                1,
            ],
        )?;
        let activation_push = ActivationPush {
            len: hidden_len as u32,
            key_clamp: self.key_clamp,
            deepembed_clamp: self.deepembed_clamp,
        };
        self.activation_backward.record_dispatch(
            commands,
            &[
                &self.grad_ffn,
                &self.cm_key,
                deepembed,
                &self.grad_cm_key,
                &self.grad_deepembed,
            ],
            bytemuck::bytes_of(&activation_push),
            [div_ceil_u32(hidden_len, 256), 1, 1],
        )?;
        let key_push = LinearPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.hidden_width as u32,
        };
        self.linear_weight_grad.record_dispatch(
            commands,
            &[&self.mixed, &self.grad_cm_key, &self.grad_key_weight],
            bytemuck::bytes_of(&key_push),
            [
                div_ceil_u32(self.width, 16),
                div_ceil_u32(self.hidden_width, 16),
                1,
            ],
        )?;
        self.linear_input_grad.record_dispatch(
            commands,
            &[&self.grad_cm_key, &self.key_weight, &self.grad_mixed],
            bytemuck::bytes_of(&key_push),
            [div_ceil_u32(self.width, 16), div_ceil_u32(batch, 16), 1],
        )?;
        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        self.channel_mix_backward.record_dispatch(
            commands,
            &[
                &self.normalized,
                previous,
                &self.mix_k,
                &self.grad_mixed,
                &self.grad_normalized,
                &self.grad_previous,
                &self.grad_mix_k,
            ],
            bytemuck::bytes_of(&mix_push),
            [div_ceil_u32(self.width, 64), 1, 1],
        )?;
        let len_push = LenPush {
            len: vector_len as u32,
        };
        let norm_push = LayerNormBackwardPush {
            rows: batch as u32,
            dim: self.width as u32,
        };
        let residual_fused = if let Some(external) = grad_normalized_external {
            if let Some(kernel) = &self.layer_norm_backward_fused_add_residual {
                kernel.record_dispatch(
                    commands,
                    &[
                        &self.grad_normalized,
                        external,
                        grad_output,
                        x,
                        &self.layer_norm_weight,
                        &self.norm_mean,
                        &self.norm_rstd,
                        &self.grad_x,
                        &self.grad_layer_norm_weight,
                        &self.grad_layer_norm_bias,
                    ],
                    bytemuck::bytes_of(&norm_push),
                    [div_ceil_u32(batch.max(self.width), 256), 1, 1],
                )?;
                true
            } else {
                self.layer_norm_backward_fused_add.record_dispatch(
                    commands,
                    &[
                        &self.grad_normalized,
                        external,
                        x,
                        &self.layer_norm_weight,
                        &self.norm_mean,
                        &self.norm_rstd,
                        &self.grad_layer_norm_input,
                        &self.grad_layer_norm_weight,
                        &self.grad_layer_norm_bias,
                    ],
                    bytemuck::bytes_of(&norm_push),
                    [div_ceil_u32(batch.max(self.width), 256), 1, 1],
                )?;
                false
            }
        } else {
            self.layer_norm_input_grad.record_dispatch(
                commands,
                &[
                    &self.grad_normalized,
                    x,
                    &self.layer_norm_weight,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.grad_layer_norm_input,
                ],
                bytemuck::bytes_of(&norm_push),
                [div_ceil_u32(batch, 64), 1, 1],
            )?;
            self.layer_norm_param_grad.record_dispatch(
                commands,
                &[
                    &self.grad_normalized,
                    x,
                    &self.norm_mean,
                    &self.norm_rstd,
                    &self.grad_layer_norm_weight,
                    &self.grad_layer_norm_bias,
                ],
                bytemuck::bytes_of(&norm_push),
                [div_ceil_u32(self.width, 256), 1, 1],
            )?;
            false
        };
        if residual_fused {
            return Ok(());
        }
        self.vector_add.record_dispatch(
            commands,
            &[grad_output, &self.grad_layer_norm_input, &self.grad_x],
            bytemuck::bytes_of(&len_push),
            [div_ceil_u32(vector_len, 256), 1, 1],
        )
    }

    pub(crate) fn record_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let hidden_len = batch * self.hidden_width;
        let matrix_len = self.width * self.hidden_width;
        commands.readback_f32(&self.output, &self.output_readback, vector_len)?;
        commands.readback_f32(&self.grad_x, &self.grad_x_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_previous,
            &self.grad_previous_readback,
            vector_len,
        )?;
        commands.readback_f32(
            &self.grad_deepembed,
            &self.grad_deepembed_readback,
            hidden_len,
        )?;
        commands.readback_f32(&self.grad_mix_k, &self.grad_mix_k_readback, self.width)?;
        commands.readback_f32(
            &self.grad_key_weight,
            &self.grad_key_weight_readback,
            matrix_len,
        )?;
        commands.readback_f32(
            &self.grad_value_weight,
            &self.grad_value_weight_readback,
            matrix_len,
        )?;
        commands.readback_f32(
            &self.grad_layer_norm_weight,
            &self.grad_layer_norm_weight_readback,
            self.width,
        )?;
        commands.readback_f32(
            &self.grad_layer_norm_bias,
            &self.grad_layer_norm_bias_readback,
            self.width,
        )
    }

    pub(crate) fn read_result(&self, batch: usize) -> Result<RwkvChannelMixResult> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let hidden_len = batch * self.hidden_width;
        let matrix_len = self.width * self.hidden_width;
        Ok(RwkvChannelMixResult {
            output: self.output_readback.read_f32(vector_len)?,
            grad_x: self.grad_x_readback.read_f32(vector_len)?,
            grad_previous: self.grad_previous_readback.read_f32(vector_len)?,
            grad_deepembed: self.grad_deepembed_readback.read_f32(hidden_len)?,
            grad_mix_k: self.grad_mix_k_readback.read_f32(self.width)?,
            grad_key_weight: self.grad_key_weight_readback.read_f32(matrix_len)?,
            grad_value_weight: self.grad_value_weight_readback.read_f32(matrix_len)?,
            grad_layer_norm_weight: self.grad_layer_norm_weight_readback.read_f32(self.width)?,
            grad_layer_norm_bias: self.grad_layer_norm_bias_readback.read_f32(self.width)?,
        })
    }

    fn validate_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV channel-mix batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        Ok(())
    }

    fn validate_inputs(
        &self,
        batch: usize,
        x: &[f32],
        previous: &[f32],
        deepembed: &[f32],
        grad_output: &[f32],
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        validate_len("x", x, vector_len)?;
        validate_len("previous", previous, vector_len)?;
        validate_len("deepembed", deepembed, batch * self.hidden_width)?;
        validate_len("grad_output", grad_output, vector_len)
    }
}

fn read_vector(path: &Path, name: &str, width: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if shape != [width] && shape != [1, width] {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{width}] or [1, {width}]");
    }
    Ok(values)
}

fn read_matrix(path: &Path, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if shape != [rows, cols] {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{rows}, {cols}]");
    }
    Ok(values)
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "RWKV channel-mix {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("RWKV channel-mix {name} contains non-finite values");
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
    fn fused_channel_mix_full_forward_matches_legacy_four_dispatch_chain() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        if !device.supports_storage_buffer_bindings(15)
            || !device.supports_compute_work_group_size_x(256)
        {
            return Ok(());
        }

        let width = 8usize;
        let batch = 2usize;
        let hidden_width = width * 4;
        let vector_len = batch * width;
        let hidden_len = batch * hidden_width;
        let layer_norm_weight = deterministic_values(width, 0.07, 109)
            .into_iter()
            .map(|value| value + 1.0)
            .collect::<Vec<_>>();
        let layer_norm_bias = deterministic_values(width, 0.025, 113);
        let mix_k = deterministic_values(width, 0.06, 127)
            .into_iter()
            .map(|value| value + 0.5)
            .collect::<Vec<_>>();
        let key_weight = deterministic_values(hidden_width * width, 0.08, 131);
        let value_weight = deterministic_values(width * hidden_width, 0.065, 137);
        let key_clamp = 0.4;
        let deepembed_clamp = 0.5;
        let op = RwkvChannelMixOp::new(
            device.clone(),
            width,
            batch,
            &layer_norm_weight,
            &layer_norm_bias,
            &mix_k,
            &key_weight,
            &value_weight,
            key_clamp,
            deepembed_clamp,
        )?;
        assert!(op.layer_norm_channel_mix_full_forward_fused.is_some());

        let x = GpuBuffer::from_f32(&device, &deterministic_values(vector_len, 0.21, 139))?;
        let previous = GpuBuffer::from_f32(&device, &deterministic_values(vector_len, 0.18, 149))?;
        let deepembed_values = deterministic_values(hidden_len, 0.58, 151)
            .into_iter()
            .map(|value| value + 0.15)
            .collect::<Vec<_>>();
        let deepembed = GpuBuffer::from_f32(&device, &deepembed_values)?;

        let legacy_normalized = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_mean = GpuBuffer::zeros_f32(&device, batch)?;
        let legacy_rstd = GpuBuffer::zeros_f32(&device, batch)?;
        let legacy_mixed = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_key = GpuBuffer::zeros_f32(&device, hidden_len)?;
        let legacy_ffn = GpuBuffer::zeros_f32(&device, hidden_len)?;
        let legacy_output = GpuBuffer::zeros_f32(&device, vector_len)?;

        let norm_push = LayerNormForwardPush {
            rows: batch as u32,
            dim: width as u32,
            eps: 1.0e-5,
        };
        let key_push = LinearPush {
            rows: batch as u32,
            input_dim: width as u32,
            output_dim: hidden_width as u32,
        };
        let activation_push = ActivationPush {
            len: hidden_len as u32,
            key_clamp,
            deepembed_clamp,
        };
        let value_push = LinearPush {
            rows: batch as u32,
            input_dim: hidden_width as u32,
            output_dim: width as u32,
        };

        let mut legacy_batch = vulkan::ComputeBatch::new(&device)?;
        op.layer_norm_channel_mix_forward_fused.record_dispatch(
            &mut legacy_batch,
            &[
                &x,
                &op.layer_norm_weight,
                &op.layer_norm_bias,
                &previous,
                &op.mix_k,
                &legacy_normalized,
                &legacy_mean,
                &legacy_rstd,
                &legacy_mixed,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(batch, 64), 1, 1],
        )?;
        op.linear_forward.record_dispatch(
            &mut legacy_batch,
            &[&legacy_mixed, &op.key_weight, &legacy_key],
            bytemuck::bytes_of(&key_push),
            [div_ceil_u32(hidden_width, 16), div_ceil_u32(batch, 16), 1],
        )?;
        op.activation_forward.record_dispatch(
            &mut legacy_batch,
            &[&legacy_key, &deepembed, &legacy_ffn],
            bytemuck::bytes_of(&activation_push),
            [div_ceil_u32(hidden_len, 256), 1, 1],
        )?;
        op.linear_residual_forward.record_dispatch(
            &mut legacy_batch,
            &[&legacy_ffn, &op.value_weight, &x, &legacy_output],
            bytemuck::bytes_of(&value_push),
            [div_ceil_u32(width, 16), div_ceil_u32(batch, 16), 1],
        )?;
        assert_eq!(legacy_batch.dispatch_count(), 4);
        legacy_batch.submit()?;

        let mut fused_batch = vulkan::ComputeBatch::new(&device)?;
        op.record_forward(&mut fused_batch, batch, &x, &previous, &deepembed)?;
        assert_eq!(fused_batch.dispatch_count(), 1);
        assert_eq!(fused_batch.shader_barrier_count(), 0);
        fused_batch.submit()?;

        for (name, legacy_values, fused_values) in [
            (
                "normalized",
                legacy_normalized.read_f32(vector_len)?,
                op.normalized.read_f32(vector_len)?,
            ),
            (
                "mean",
                legacy_mean.read_f32(batch)?,
                op.norm_mean.read_f32(batch)?,
            ),
            (
                "rstd",
                legacy_rstd.read_f32(batch)?,
                op.norm_rstd.read_f32(batch)?,
            ),
            (
                "mixed",
                legacy_mixed.read_f32(vector_len)?,
                op.mixed.read_f32(vector_len)?,
            ),
            (
                "cm_key",
                legacy_key.read_f32(hidden_len)?,
                op.cm_key.read_f32(hidden_len)?,
            ),
            (
                "ffn",
                legacy_ffn.read_f32(hidden_len)?,
                op.ffn.read_f32(hidden_len)?,
            ),
            (
                "output",
                legacy_output.read_f32(vector_len)?,
                op.output.read_f32(vector_len)?,
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
    fn fused_channel_mix_producer_matches_legacy_three_dispatch_chain() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        if !device.supports_storage_buffer_bindings(13)
            || !device.supports_compute_work_group_size_x(256)
        {
            return Ok(());
        }

        let width = 8usize;
        let batch = 3usize;
        let hidden_width = width * 4;
        let vector_len = batch * width;
        let hidden_len = batch * hidden_width;
        let layer_norm_weight = deterministic_values(width, 0.08, 71)
            .into_iter()
            .map(|value| value + 1.0)
            .collect::<Vec<_>>();
        let layer_norm_bias = deterministic_values(width, 0.03, 73);
        let mix_k = deterministic_values(width, 0.07, 79)
            .into_iter()
            .map(|value| value + 0.5)
            .collect::<Vec<_>>();
        let key_weight = deterministic_values(hidden_width * width, 0.09, 83);
        let value_weight = deterministic_values(width * hidden_width, 0.02, 89);
        let key_clamp = 0.35;
        let deepembed_clamp = 0.45;
        let op = RwkvChannelMixOp::new(
            device.clone(),
            width,
            batch,
            &layer_norm_weight,
            &layer_norm_bias,
            &mix_k,
            &key_weight,
            &value_weight,
            key_clamp,
            deepembed_clamp,
        )?;
        assert!(op
            .layer_norm_channel_mix_key_relu2_deepembed_forward_fused
            .is_some());

        let x = GpuBuffer::from_f32(&device, &deterministic_values(vector_len, 0.19, 97))?;
        let previous = GpuBuffer::from_f32(&device, &deterministic_values(vector_len, 0.16, 101))?;
        let deepembed_values = deterministic_values(hidden_len, 0.53, 103)
            .into_iter()
            .map(|value| value + 0.2)
            .collect::<Vec<_>>();
        let deepembed = GpuBuffer::from_f32(&device, &deepembed_values)?;

        let legacy_normalized = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_mean = GpuBuffer::zeros_f32(&device, batch)?;
        let legacy_rstd = GpuBuffer::zeros_f32(&device, batch)?;
        let legacy_mixed = GpuBuffer::zeros_f32(&device, vector_len)?;
        let legacy_key = GpuBuffer::zeros_f32(&device, hidden_len)?;
        let legacy_ffn = GpuBuffer::zeros_f32(&device, hidden_len)?;

        let norm_push = LayerNormForwardPush {
            rows: batch as u32,
            dim: width as u32,
            eps: 1.0e-5,
        };
        let key_push = LinearPush {
            rows: batch as u32,
            input_dim: width as u32,
            output_dim: hidden_width as u32,
        };
        let activation_push = ActivationPush {
            len: hidden_len as u32,
            key_clamp,
            deepembed_clamp,
        };
        let mut legacy_batch = vulkan::ComputeBatch::new(&device)?;
        op.layer_norm_channel_mix_forward_fused.record_dispatch(
            &mut legacy_batch,
            &[
                &x,
                &op.layer_norm_weight,
                &op.layer_norm_bias,
                &previous,
                &op.mix_k,
                &legacy_normalized,
                &legacy_mean,
                &legacy_rstd,
                &legacy_mixed,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(batch, 64), 1, 1],
        )?;
        op.linear_forward.record_dispatch(
            &mut legacy_batch,
            &[&legacy_mixed, &op.key_weight, &legacy_key],
            bytemuck::bytes_of(&key_push),
            [div_ceil_u32(hidden_width, 16), div_ceil_u32(batch, 16), 1],
        )?;
        op.activation_forward.record_dispatch(
            &mut legacy_batch,
            &[&legacy_key, &deepembed, &legacy_ffn],
            bytemuck::bytes_of(&activation_push),
            [div_ceil_u32(hidden_len, 256), 1, 1],
        )?;
        assert_eq!(legacy_batch.dispatch_count(), 3);
        legacy_batch.submit()?;

        let mut fused_batch = vulkan::ComputeBatch::new(&device)?;
        op.record_forward_producer(&mut fused_batch, batch, &x, &previous, &deepembed)?;
        assert_eq!(fused_batch.dispatch_count(), 1);
        assert_eq!(fused_batch.shader_barrier_count(), 0);
        fused_batch.submit()?;

        for (name, legacy_values, fused_values) in [
            (
                "normalized",
                legacy_normalized.read_f32(vector_len)?,
                op.normalized.read_f32(vector_len)?,
            ),
            (
                "mean",
                legacy_mean.read_f32(batch)?,
                op.norm_mean.read_f32(batch)?,
            ),
            (
                "rstd",
                legacy_rstd.read_f32(batch)?,
                op.norm_rstd.read_f32(batch)?,
            ),
            (
                "mixed",
                legacy_mixed.read_f32(vector_len)?,
                op.mixed.read_f32(vector_len)?,
            ),
            (
                "cm_key",
                legacy_key.read_f32(hidden_len)?,
                op.cm_key.read_f32(hidden_len)?,
            ),
            (
                "ffn",
                legacy_ffn.read_f32(hidden_len)?,
                op.ffn.read_f32(hidden_len)?,
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
    fn fused_external_norm_gradient_matches_zero_external_reference() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let width = 4usize;
        let batch = 2usize;
        let hidden_width = width * 4;
        let layer_norm_weight = [1.0, 0.9, 1.1, 0.8];
        let layer_norm_bias = [0.05, -0.03, 0.02, -0.01];
        let mix_k = [0.2, 0.4, 0.6, 0.8];
        let key_weight = deterministic_values(hidden_width * width, 0.031, 3);
        let value_weight = deterministic_values(width * hidden_width, 0.027, 5);
        let x = deterministic_values(batch * width, 0.17, 7);
        let previous = deterministic_values(batch * width, 0.13, 11);
        let deepembed = deterministic_values(batch * hidden_width, 0.09, 13)
            .into_iter()
            .map(|value| value + 1.0)
            .collect::<Vec<_>>();
        let grad_output = deterministic_values(batch * width, 0.07, 17);
        let external = vec![0.0f32; batch * width];

        let mut reference = RwkvChannelMixOp::new(
            device.clone(),
            width,
            batch,
            &layer_norm_weight,
            &layer_norm_bias,
            &mix_k,
            &key_weight,
            &value_weight,
            10.0,
            10.0,
        )?;
        let mut fused = RwkvChannelMixOp::new(
            device.clone(),
            width,
            batch,
            &layer_norm_weight,
            &layer_norm_bias,
            &mix_k,
            &key_weight,
            &value_weight,
            10.0,
            10.0,
        )?;

        let reference_metrics = run_recorded_backward(
            &mut reference,
            batch,
            &x,
            &previous,
            &deepembed,
            &grad_output,
            None,
        )?;
        let external_buffer = GpuBuffer::from_f32(&device, &external)?;
        let fused_metrics = run_recorded_backward(
            &mut fused,
            batch,
            &x,
            &previous,
            &deepembed,
            &grad_output,
            Some(&external_buffer),
        )?;

        assert_eq!(fused_metrics.1 + 1, reference_metrics.1);
        assert_eq!(fused_metrics.0 + 2, reference_metrics.0);
        assert_results_close(&reference.read_result(batch)?, &fused.read_result(batch)?)?;
        Ok(())
    }

    #[test]
    fn fused_layer_norm_backward_matches_legacy_nonzero_external_chain() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let width = 4usize;
        let batch = 2usize;
        let hidden_width = width * 4;
        let layer_norm_weight = [1.0, 0.9, 1.1, 0.8];
        let layer_norm_bias = [0.0; 4];
        let mix_k = [0.25; 4];
        let key_weight = deterministic_values(hidden_width * width, 0.02, 19);
        let value_weight = deterministic_values(width * hidden_width, 0.02, 23);
        let op = RwkvChannelMixOp::new(
            device.clone(),
            width,
            batch,
            &layer_norm_weight,
            &layer_norm_bias,
            &mix_k,
            &key_weight,
            &value_weight,
            10.0,
            10.0,
        )?;

        let len = batch * width;
        let grad_primary = GpuBuffer::from_f32(&device, &deterministic_values(len, 0.17, 29))?;
        let grad_external = GpuBuffer::from_f32(&device, &deterministic_values(len, 0.11, 31))?;
        let x = GpuBuffer::from_f32(&device, &deterministic_values(len, 0.23, 37))?;
        let mean = GpuBuffer::from_f32(&device, &[0.03, -0.07])?;
        let rstd = GpuBuffer::from_f32(&device, &[1.15, 0.91])?;
        let legacy_total = GpuBuffer::zeros_f32(&device, len)?;
        let legacy_grad_input = GpuBuffer::zeros_f32(&device, len)?;
        let legacy_grad_weight = GpuBuffer::zeros_f32(&device, width)?;
        let legacy_grad_bias = GpuBuffer::zeros_f32(&device, width)?;
        let fused_grad_input = GpuBuffer::zeros_f32(&device, len)?;
        let fused_grad_weight = GpuBuffer::zeros_f32(&device, width)?;
        let fused_grad_bias = GpuBuffer::zeros_f32(&device, width)?;
        let legacy_input_readback = GpuBuffer::zeros_host_f32(&device, len)?;
        let legacy_weight_readback = GpuBuffer::zeros_host_f32(&device, width)?;
        let legacy_bias_readback = GpuBuffer::zeros_host_f32(&device, width)?;
        let fused_input_readback = GpuBuffer::zeros_host_f32(&device, len)?;
        let fused_weight_readback = GpuBuffer::zeros_host_f32(&device, width)?;
        let fused_bias_readback = GpuBuffer::zeros_host_f32(&device, width)?;
        let len_push = LenPush { len: len as u32 };
        let norm_push = LayerNormBackwardPush {
            rows: batch as u32,
            dim: width as u32,
        };

        let mut legacy = vulkan::ComputeBatch::new(&device)?;
        op.vector_add.record_dispatch(
            &mut legacy,
            &[&grad_primary, &grad_external, &legacy_total],
            bytemuck::bytes_of(&len_push),
            [div_ceil_u32(len, 256), 1, 1],
        )?;
        op.layer_norm_input_grad.record_dispatch(
            &mut legacy,
            &[
                &legacy_total,
                &x,
                &op.layer_norm_weight,
                &mean,
                &rstd,
                &legacy_grad_input,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(batch, 64), 1, 1],
        )?;
        op.layer_norm_param_grad.record_dispatch(
            &mut legacy,
            &[
                &legacy_total,
                &x,
                &mean,
                &rstd,
                &legacy_grad_weight,
                &legacy_grad_bias,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(width, 256), 1, 1],
        )?;
        legacy.readback_f32(&legacy_grad_input, &legacy_input_readback, len)?;
        legacy.readback_f32(&legacy_grad_weight, &legacy_weight_readback, width)?;
        legacy.readback_f32(&legacy_grad_bias, &legacy_bias_readback, width)?;
        assert_eq!(legacy.dispatch_count(), 3);
        assert_eq!(legacy.shader_barrier_count(), 1);
        legacy.submit()?;

        let mut fused = vulkan::ComputeBatch::new(&device)?;
        op.layer_norm_backward_fused_add.record_dispatch(
            &mut fused,
            &[
                &grad_primary,
                &grad_external,
                &x,
                &op.layer_norm_weight,
                &mean,
                &rstd,
                &fused_grad_input,
                &fused_grad_weight,
                &fused_grad_bias,
            ],
            bytemuck::bytes_of(&norm_push),
            [div_ceil_u32(batch.max(width), 256), 1, 1],
        )?;
        fused.readback_f32(&fused_grad_input, &fused_input_readback, len)?;
        fused.readback_f32(&fused_grad_weight, &fused_weight_readback, width)?;
        fused.readback_f32(&fused_grad_bias, &fused_bias_readback, width)?;
        assert_eq!(fused.dispatch_count(), 1);
        assert_eq!(fused.shader_barrier_count(), 0);
        fused.submit()?;

        for (name, legacy_values, fused_values) in [
            (
                "grad_input",
                legacy_input_readback.read_f32(len)?,
                fused_input_readback.read_f32(len)?,
            ),
            (
                "grad_weight",
                legacy_weight_readback.read_f32(width)?,
                fused_weight_readback.read_f32(width)?,
            ),
            (
                "grad_bias",
                legacy_bias_readback.read_f32(width)?,
                fused_bias_readback.read_f32(width)?,
            ),
        ] {
            let max_diff = legacy_values
                .iter()
                .zip(&fused_values)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            if max_diff > 1.0e-6 {
                bail!("fused LayerNorm {name} max abs diff {max_diff} exceeds tolerance");
            }
        }
        Ok(())
    }

    fn run_recorded_backward(
        op: &mut RwkvChannelMixOp,
        batch: usize,
        x: &[f32],
        previous: &[f32],
        deepembed: &[f32],
        grad_output: &[f32],
        external: Option<&GpuBuffer>,
    ) -> Result<(usize, usize)> {
        let x_buffer = GpuBuffer::from_f32(&op.device, x)?;
        let previous_buffer = GpuBuffer::from_f32(&op.device, previous)?;
        let deepembed_buffer = GpuBuffer::from_f32(&op.device, deepembed)?;
        let grad_output_buffer = GpuBuffer::from_f32(&op.device, grad_output)?;
        let mut commands = vulkan::ComputeBatch::new(&op.device)?;
        op.record_forward(
            &mut commands,
            batch,
            &x_buffer,
            &previous_buffer,
            &deepembed_buffer,
        )?;
        op.record_backward_with_normalized_grad(
            &mut commands,
            batch,
            &x_buffer,
            &previous_buffer,
            &deepembed_buffer,
            &grad_output_buffer,
            external,
        )?;
        op.record_readback(&mut commands, batch)?;
        let metrics = (commands.dispatch_count(), commands.shader_barrier_count());
        commands.submit()?;
        Ok(metrics)
    }

    fn assert_results_close(lhs: &RwkvChannelMixResult, rhs: &RwkvChannelMixResult) -> Result<()> {
        for (name, left, right) in [
            ("output", &lhs.output, &rhs.output),
            ("grad_x", &lhs.grad_x, &rhs.grad_x),
            ("grad_previous", &lhs.grad_previous, &rhs.grad_previous),
            ("grad_deepembed", &lhs.grad_deepembed, &rhs.grad_deepembed),
            ("grad_mix_k", &lhs.grad_mix_k, &rhs.grad_mix_k),
            (
                "grad_key_weight",
                &lhs.grad_key_weight,
                &rhs.grad_key_weight,
            ),
            (
                "grad_value_weight",
                &lhs.grad_value_weight,
                &rhs.grad_value_weight,
            ),
            (
                "grad_layer_norm_weight",
                &lhs.grad_layer_norm_weight,
                &rhs.grad_layer_norm_weight,
            ),
            (
                "grad_layer_norm_bias",
                &lhs.grad_layer_norm_bias,
                &rhs.grad_layer_norm_bias,
            ),
        ] {
            let max_diff = left
                .iter()
                .zip(right)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if max_diff > 1.0e-6 {
                bail!("fused channel-mix {name} max abs diff {max_diff} exceeds tolerance");
            }
        }
        Ok(())
    }

    fn deterministic_values(len: usize, scale: f32, phase: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let centered = ((index * 37 + phase * 19) % 101) as f32 - 50.0;
                centered * scale / 50.0
            })
            .collect()
    }
}
