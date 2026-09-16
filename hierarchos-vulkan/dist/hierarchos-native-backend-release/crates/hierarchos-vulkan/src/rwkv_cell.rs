use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::rwkv_low_rank::{RwkvLowRankFp16ParameterMirrors, RwkvLowRankParameterGradArithmetic};
use crate::rwkv_optimizer::{RwkvDecayClass, RwkvTrainableRef};
use crate::{
    read_f32_tensor, replace_f32_tensors, vulkan, GpuBuffer, RwkvChannelMixOp,
    RwkvChannelMixResult, RwkvNumericsPolicy, RwkvPackedStateOp, RwkvParameterSnapshot,
    RwkvStateReadoutMode, RwkvTimeMixCoreOp, SharedTokenAdapterTrainer, VulkanDevice,
};

const LAYER_NORM_FORWARD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_forward.spv");
const LAYER_NORM_INPUT_GRAD_RESIDUAL_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_input_grad_residual_fused.spv");
const LAYER_NORM_PARAM_GRAD_SPV: &[u8] = include_bytes!("../shaders/layer_norm_param_grad.spv");
const VECTOR_ADD_SPV: &[u8] = include_bytes!("../shaders/vector_add.spv");
const PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/packed_cell_channel_mix_state_forward_fused.spv");
const PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_WG32_SPV: &[u8] =
    include_bytes!("../shaders/packed_cell_channel_mix_state_forward_fused_wg32.spv");
const PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_WG64_SPV: &[u8] =
    include_bytes!("../shaders/packed_cell_channel_mix_state_forward_fused_wg64.spv");
const PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_WG128_SPV: &[u8] =
    include_bytes!("../shaders/packed_cell_channel_mix_state_forward_fused_wg128.spv");
const HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE_ENV: &str =
    "HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE";
const PACKED_CELL_FORWARD_ONLY_FLAG: u32 = 1;
const PACKED_CELL_MATRIX_ALREADY_PACKED_FLAG: u32 = 2;

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
struct LenPush {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackedCellChannelMixStatePush {
    rows: u32,
    width: u32,
    hidden_width: u32,
    head_size: u32,
    matrix_offset: u32,
    eps: f32,
    key_clamp: f32,
    deepembed_clamp: f32,
    state_clamp: f32,
    flags: u32,
}

#[derive(Debug)]
pub struct RwkvCellSliceResult {
    pub output: Vec<f32>,
    pub new_matrix_state: Vec<f32>,
    pub grad_x: Vec<f32>,
    pub grad_matrix_state: Vec<f32>,
    pub grad_previous_tm: Vec<f32>,
    pub grad_previous_cm: Vec<f32>,
    pub token_feature_grad: Vec<f32>,
    pub grad_ln1_weight: Vec<f32>,
    pub grad_ln1_bias: Vec<f32>,
    pub channel_mix: RwkvChannelMixResult,
}

#[derive(Debug)]
pub struct RwkvPackedCellResult {
    pub output: Vec<f32>,
    pub packed_new_state: Vec<f32>,
    pub grad_x: Vec<f32>,
    pub grad_packed_state: Vec<f32>,
    pub token_feature_grad: Vec<f32>,
}

/// Vulkan-native coherent-v9 cell slice from raw residual input through LN1,
/// full RWKV time-mix, the time residual, SharedTokenAdapter DeepEmbed,
/// channel-mix, and the channel residual.
///
/// The recurrent caches are still supplied as separate GPU uploads in this
/// slice. The next state-ownership layer packs/clamps them into the public
/// Hierarchos state contract; all cell-internal activations already remain on
/// Vulkan here.
pub struct RwkvCellSliceOp {
    device: VulkanDevice,
    width: usize,
    head_size: usize,
    max_batch: usize,
    token_feature_width: usize,

    time_mix: RwkvTimeMixCoreOp,
    channel_mix: RwkvChannelMixOp,
    deepembed_adapter: SharedTokenAdapterTrainer,

    ln1_weight: GpuBuffer,
    ln1_bias: GpuBuffer,
    x: GpuBuffer,
    previous_tm: GpuBuffer,
    previous_cm: GpuBuffer,
    matrix_state: GpuBuffer,
    token_features: GpuBuffer,
    grad_matrix_state_out: GpuBuffer,
    grad_output: GpuBuffer,

    x_norm: GpuBuffer,
    ln1_mean: GpuBuffer,
    ln1_rstd: GpuBuffer,
    time_residual: GpuBuffer,
    grad_x_norm_total: GpuBuffer,
    grad_x: GpuBuffer,
    grad_ln1_weight: GpuBuffer,
    grad_ln1_bias: GpuBuffer,

    new_matrix_state_readback: GpuBuffer,
    grad_x_readback: GpuBuffer,
    grad_matrix_state_readback: GpuBuffer,
    grad_previous_tm_readback: GpuBuffer,
    grad_ln1_weight_readback: GpuBuffer,
    grad_ln1_bias_readback: GpuBuffer,

    layer_norm_forward: vulkan::ComputeKernel,
    layer_norm_input_grad_residual_fused: vulkan::ComputeKernel,
    layer_norm_param_grad: vulkan::ComputeKernel,
    vector_add: vulkan::ComputeKernel,
}

impl RwkvCellSliceOp {
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        cell_prefix: &str,
        deepembed_adapter_prefix: &str,
        head_size: usize,
        max_batch: usize,
        key_clamp: f32,
        deepembed_clamp: f32,
    ) -> Result<Self> {
        if cell_prefix.trim().is_empty() || deepembed_adapter_prefix.trim().is_empty() {
            bail!("RWKV cell and DeepEmbed adapter prefixes must not be empty");
        }
        let model_dir = model_dir.as_ref();
        let path = model_dir.join("model.safetensors");
        let (ln1_shape, ln1_weight) = read_f32_tensor(&path, &format!("{cell_prefix}.ln1.weight"))?;
        let width = match ln1_shape.as_slice() {
            [width] if *width > 0 => *width,
            _ => {
                bail!("RWKV tensor {cell_prefix}.ln1.weight must have shape [C], got {ln1_shape:?}")
            }
        };
        let (ln1_bias_shape, ln1_bias) =
            read_f32_tensor(&path, &format!("{cell_prefix}.ln1.bias"))?;
        if ln1_bias_shape != [width] {
            bail!(
                "RWKV tensor {cell_prefix}.ln1.bias has shape {ln1_bias_shape:?}; expected [{width}]"
            );
        }

        let time_mix = RwkvTimeMixCoreOp::from_model_package_full(
            device.clone(),
            model_dir,
            cell_prefix,
            head_size,
            max_batch,
        )?;
        let channel_mix = RwkvChannelMixOp::from_model_package(
            device.clone(),
            model_dir,
            cell_prefix,
            max_batch,
            key_clamp,
            deepembed_clamp,
        )?;
        let deepembed_adapter = SharedTokenAdapterTrainer::from_model_package(
            device.clone(),
            model_dir,
            deepembed_adapter_prefix,
            max_batch,
            0.0,
        )?;
        Self::new(
            device,
            time_mix,
            channel_mix,
            deepembed_adapter,
            &ln1_weight,
            &ln1_bias,
        )
    }

    pub(crate) fn new(
        device: VulkanDevice,
        time_mix: RwkvTimeMixCoreOp,
        channel_mix: RwkvChannelMixOp,
        deepembed_adapter: SharedTokenAdapterTrainer,
        ln1_weight: &[f32],
        ln1_bias: &[f32],
    ) -> Result<Self> {
        let width = time_mix.width();
        if channel_mix.width() != width {
            bail!(
                "RWKV channel-mix width {} does not match time-mix width {width}",
                channel_mix.width()
            );
        }
        if deepembed_adapter.output_dim() != channel_mix.hidden_width() {
            bail!(
                "DeepEmbed adapter output width {} does not match channel-mix hidden width {}",
                deepembed_adapter.output_dim(),
                channel_mix.hidden_width()
            );
        }
        if ln1_weight.len() != width || ln1_bias.len() != width {
            bail!("LN1 weight/bias must both have {width} values");
        }
        if ln1_weight
            .iter()
            .chain(ln1_bias)
            .any(|value| !value.is_finite())
        {
            bail!("LN1 parameters contain non-finite values");
        }
        let max_batch = time_mix
            .max_batch()
            .min(channel_mix.max_batch())
            .min(deepembed_adapter.max_rows());
        if max_batch == 0 {
            bail!("RWKV cell Vulkan capacity must be positive");
        }
        let head_size = time_mix.head_size();
        let vector_len = max_batch
            .checked_mul(width)
            .context("RWKV cell vector capacity overflow")?;
        let state_len = vector_len
            .checked_mul(head_size)
            .context("RWKV cell matrix-state capacity overflow")?;
        let token_feature_width = deepembed_adapter.input_dim();
        let token_feature_len = max_batch
            .checked_mul(token_feature_width)
            .context("RWKV cell token-feature capacity overflow")?;

        Ok(Self {
            layer_norm_forward: vulkan::ComputeKernel::new(
                &device,
                LAYER_NORM_FORWARD_SPV,
                6,
                std::mem::size_of::<LayerNormForwardPush>() as u32,
            )?,
            layer_norm_input_grad_residual_fused: vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_INPUT_GRAD_RESIDUAL_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
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
            ln1_weight: GpuBuffer::from_f32(&device, ln1_weight)?,
            ln1_bias: GpuBuffer::from_f32(&device, ln1_bias)?,
            x: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous_tm: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous_cm: GpuBuffer::zeros_f32(&device, vector_len)?,
            matrix_state: GpuBuffer::zeros_f32(&device, state_len)?,
            token_features: GpuBuffer::zeros_f32(&device, token_feature_len)?,
            grad_matrix_state_out: GpuBuffer::zeros_f32(&device, state_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            ln1_mean: GpuBuffer::zeros_f32(&device, max_batch)?,
            ln1_rstd: GpuBuffer::zeros_f32(&device, max_batch)?,
            time_residual: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_x_norm_total: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_x: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_ln1_weight: GpuBuffer::zeros_f32(&device, width)?,
            grad_ln1_bias: GpuBuffer::zeros_f32(&device, width)?,
            new_matrix_state_readback: GpuBuffer::zeros_host_f32(&device, state_len)?,
            grad_x_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_matrix_state_readback: GpuBuffer::zeros_host_f32(&device, state_len)?,
            grad_previous_tm_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_ln1_weight_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_ln1_bias_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            device,
            width,
            head_size,
            max_batch,
            token_feature_width,
            time_mix,
            channel_mix,
            deepembed_adapter,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward(
        &mut self,
        batch: usize,
        x: &[f32],
        previous_tm: &[f32],
        previous_cm: &[f32],
        matrix_state: &[f32],
        token_features: &[f32],
        grad_matrix_state_out: &[f32],
        grad_output: &[f32],
    ) -> Result<RwkvCellSliceResult> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let state_len = vector_len * self.head_size;
        validate_len("x", x, vector_len)?;
        validate_len("previous_tm", previous_tm, vector_len)?;
        validate_len("previous_cm", previous_cm, vector_len)?;
        validate_len("matrix_state", matrix_state, state_len)?;
        validate_len(
            "token_features",
            token_features,
            batch * self.token_feature_width,
        )?;
        validate_len("grad_matrix_state_out", grad_matrix_state_out, state_len)?;
        validate_len("grad_output", grad_output, vector_len)?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.x, x)?;
        commands.upload_f32(&self.previous_tm, previous_tm)?;
        commands.upload_f32(&self.previous_cm, previous_cm)?;
        commands.upload_f32(&self.matrix_state, matrix_state)?;
        commands.upload_f32(&self.token_features, token_features)?;
        commands.upload_f32(&self.grad_matrix_state_out, grad_matrix_state_out)?;
        commands.upload_f32(&self.grad_output, grad_output)?;

        self.record_forward(
            &mut commands,
            batch,
            &self.x,
            &self.previous_tm,
            &self.previous_cm,
            &self.matrix_state,
            &self.token_features,
        )?;
        self.record_backward(
            &mut commands,
            batch,
            &self.x,
            &self.previous_tm,
            &self.previous_cm,
            &self.matrix_state,
            &self.token_features,
            &self.grad_matrix_state_out,
            &self.grad_output,
            None,
            None,
            None,
        )?;

        self.channel_mix.record_readback(&mut commands, batch)?;
        self.deepembed_adapter
            .record_grad_input_readback(&mut commands, batch)?;
        commands.readback_f32(
            self.time_mix.new_state_buffer(),
            &self.new_matrix_state_readback,
            state_len,
        )?;
        commands.readback_f32(&self.grad_x, &self.grad_x_readback, vector_len)?;
        commands.readback_f32(
            self.time_mix.grad_state_buffer(),
            &self.grad_matrix_state_readback,
            state_len,
        )?;
        commands.readback_f32(
            self.time_mix.full_grad_previous_buffer(),
            &self.grad_previous_tm_readback,
            vector_len,
        )?;
        commands.readback_f32(
            &self.grad_ln1_weight,
            &self.grad_ln1_weight_readback,
            self.width,
        )?;
        commands.readback_f32(
            &self.grad_ln1_bias,
            &self.grad_ln1_bias_readback,
            self.width,
        )?;
        commands.submit()?;

        let channel_mix = self.channel_mix.read_result(batch)?;
        Ok(RwkvCellSliceResult {
            output: channel_mix.output.clone(),
            new_matrix_state: self.new_matrix_state_readback.read_f32(state_len)?,
            grad_x: self.grad_x_readback.read_f32(vector_len)?,
            grad_matrix_state: self.grad_matrix_state_readback.read_f32(state_len)?,
            grad_previous_tm: self.grad_previous_tm_readback.read_f32(vector_len)?,
            grad_previous_cm: channel_mix.grad_previous.clone(),
            token_feature_grad: self.deepembed_adapter.read_grad_input(batch)?,
            grad_ln1_weight: self.grad_ln1_weight_readback.read_f32(self.width)?,
            grad_ln1_bias: self.grad_ln1_bias_readback.read_f32(self.width)?,
            channel_mix,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous_tm: &GpuBuffer,
        previous_cm: &GpuBuffer,
        matrix_state: &GpuBuffer,
        token_features: &GpuBuffer,
    ) -> Result<()> {
        self.record_forward_before_channel_mix(
            commands,
            batch,
            x,
            previous_tm,
            matrix_state,
            token_features,
        )?;
        self.record_channel_mix_forward(commands, batch, previous_cm)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_forward_before_channel_mix(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous_tm: &GpuBuffer,
        matrix_state: &GpuBuffer,
        token_features: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        self.deepembed_adapter
            .record_forward(commands, batch, token_features)?;

        if self.time_mix.can_fuse_layer_norm_forward() {
            self.time_mix.record_full_forward_with_layer_norm_residual(
                commands,
                batch,
                matrix_state,
                x,
                &self.ln1_weight,
                &self.ln1_bias,
                &self.x_norm,
                &self.ln1_mean,
                &self.ln1_rstd,
                previous_tm,
                x,
                &self.time_residual,
                1.0e-5,
            )?;
        } else {
            let norm_forward = LayerNormForwardPush {
                rows: batch as u32,
                dim: self.width as u32,
                eps: 1.0e-5,
            };
            self.layer_norm_forward.record_dispatch(
                commands,
                &[
                    x,
                    &self.ln1_weight,
                    &self.ln1_bias,
                    &self.x_norm,
                    &self.ln1_mean,
                    &self.ln1_rstd,
                ],
                bytemuck::bytes_of(&norm_forward),
                [div_ceil_u32(batch, 64), 1, 1],
            )?;
            self.time_mix.record_full_forward_with_residual(
                commands,
                batch,
                matrix_state,
                &self.x_norm,
                previous_tm,
                x,
                &self.time_residual,
            )?;
        }
        Ok(())
    }

    fn record_channel_mix_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        previous_cm: &GpuBuffer,
    ) -> Result<()> {
        self.channel_mix.record_forward(
            commands,
            batch,
            &self.time_residual,
            previous_cm,
            self.deepembed_adapter.output_buffer(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous_tm: &GpuBuffer,
        previous_cm: &GpuBuffer,
        matrix_state: &GpuBuffer,
        token_features: &GpuBuffer,
        grad_matrix_state_out: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_x_norm_state: Option<&GpuBuffer>,
        grad_x_norm2_state: Option<&GpuBuffer>,
        grad_v_first_state: Option<&GpuBuffer>,
    ) -> Result<()> {
        self.record_backward_inner(
            commands,
            batch,
            x,
            previous_tm,
            previous_cm,
            matrix_state,
            None,
            token_features,
            grad_matrix_state_out,
            grad_output,
            grad_x_norm_state,
            grad_x_norm2_state,
            grad_v_first_state,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_backward_from_packed_state(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous_tm: &GpuBuffer,
        previous_cm: &GpuBuffer,
        dense_matrix_state_scratch: &GpuBuffer,
        packed_state: &GpuBuffer,
        matrix_offset: usize,
        token_features: &GpuBuffer,
        grad_matrix_state_out: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_x_norm_state: Option<&GpuBuffer>,
        grad_x_norm2_state: Option<&GpuBuffer>,
        grad_v_first_state: Option<&GpuBuffer>,
    ) -> Result<()> {
        self.record_backward_inner(
            commands,
            batch,
            x,
            previous_tm,
            previous_cm,
            dense_matrix_state_scratch,
            Some((packed_state, matrix_offset)),
            token_features,
            grad_matrix_state_out,
            grad_output,
            grad_x_norm_state,
            grad_x_norm2_state,
            grad_v_first_state,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_backward_inner(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        previous_tm: &GpuBuffer,
        previous_cm: &GpuBuffer,
        matrix_state: &GpuBuffer,
        packed_state: Option<(&GpuBuffer, usize)>,
        token_features: &GpuBuffer,
        grad_matrix_state_out: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_x_norm_state: Option<&GpuBuffer>,
        grad_x_norm2_state: Option<&GpuBuffer>,
        grad_v_first_state: Option<&GpuBuffer>,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        self.channel_mix.record_backward_with_normalized_grad(
            commands,
            batch,
            &self.time_residual,
            previous_cm,
            self.deepembed_adapter.output_buffer(),
            grad_output,
            grad_x_norm2_state,
        )?;
        self.deepembed_adapter.record_backward(
            commands,
            batch,
            token_features,
            self.channel_mix.grad_deepembed_buffer(),
        )?;
        let grad_x_norm_state_fused = if let Some((packed_state, matrix_offset)) = packed_state {
            self.time_mix
                .record_full_backward_with_v_grad_from_packed_state(
                    commands,
                    batch,
                    packed_state,
                    matrix_offset,
                    matrix_state,
                    &self.x_norm,
                    previous_tm,
                    grad_matrix_state_out,
                    self.channel_mix.grad_x_buffer(),
                    grad_v_first_state,
                    grad_x_norm_state,
                )?
        } else {
            self.time_mix.record_full_backward_with_v_grad(
                commands,
                batch,
                matrix_state,
                &self.x_norm,
                previous_tm,
                grad_matrix_state_out,
                self.channel_mix.grad_x_buffer(),
                grad_v_first_state,
                grad_x_norm_state,
            )?
        };

        let len_push = LenPush {
            len: vector_len as u32,
        };
        let add_groups = [div_ceil_u32(vector_len, 256), 1, 1];
        let grad_x_norm = if let Some(state_grad) = grad_x_norm_state {
            if grad_x_norm_state_fused {
                self.time_mix.full_grad_x_norm_buffer()
            } else {
                self.vector_add.record_dispatch(
                    commands,
                    &[
                        self.time_mix.full_grad_x_norm_buffer(),
                        state_grad,
                        &self.grad_x_norm_total,
                    ],
                    bytemuck::bytes_of(&len_push),
                    add_groups,
                )?;
                &self.grad_x_norm_total
            }
        } else {
            self.time_mix.full_grad_x_norm_buffer()
        };
        let norm_backward = LayerNormBackwardPush {
            rows: batch as u32,
            dim: self.width as u32,
        };
        self.layer_norm_input_grad_residual_fused.record_dispatch(
            commands,
            &[
                grad_x_norm,
                x,
                &self.ln1_weight,
                &self.ln1_mean,
                &self.ln1_rstd,
                self.channel_mix.grad_x_buffer(),
                &self.grad_x,
            ],
            bytemuck::bytes_of(&norm_backward),
            [div_ceil_u32(batch, 64), 1, 1],
        )?;
        self.layer_norm_param_grad.record_dispatch(
            commands,
            &[
                grad_x_norm,
                x,
                &self.ln1_mean,
                &self.ln1_rstd,
                &self.grad_ln1_weight,
                &self.grad_ln1_bias,
            ],
            bytemuck::bytes_of(&norm_backward),
            [div_ceil_u32(self.width, 256), 1, 1],
        )?;
        Ok(())
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn head_size(&self) -> usize {
        self.head_size
    }

    pub fn backward_segment_schedule_label(&self, batch: usize) -> Option<String> {
        self.time_mix.backward_segment_schedule_label(batch)
    }

    pub fn backward_segment_schedule_was_autotuned(&self, batch: usize) -> bool {
        self.time_mix.backward_segment_schedule_was_autotuned(batch)
    }

    pub fn backward_kernel_geometry_label(&self, batch: usize) -> Option<&'static str> {
        self.time_mix.backward_kernel_geometry_label(batch)
    }

    pub fn numerics_policy(&self) -> RwkvNumericsPolicy {
        self.time_mix.numerics_policy()
    }

    pub(crate) fn supports_numerics_policy(&self, policy: RwkvNumericsPolicy) -> bool {
        self.time_mix.supports_numerics_policy(policy)
    }

    pub fn set_numerics_policy(&mut self, policy: RwkvNumericsPolicy) -> Result<()> {
        self.time_mix.set_numerics_policy(policy)
    }

    pub(crate) fn available_backward_kernel_geometry_labels(
        &self,
        batch: usize,
    ) -> Result<Vec<String>> {
        self.time_mix
            .available_backward_kernel_geometry_labels(batch)
    }

    pub(crate) fn available_backward_kernel_geometry_labels_for_numerics(
        &self,
        batch: usize,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<Vec<String>> {
        self.time_mix
            .available_backward_kernel_geometry_labels_for_numerics(batch, numerics_policy)
    }

    pub(crate) fn backward_segment_schedule_geometry_pair_available(
        &self,
        batch: usize,
        schedule_label: &str,
        geometry_label: &str,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<bool> {
        self.time_mix
            .backward_segment_schedule_geometry_pair_available(
                batch,
                schedule_label,
                geometry_label,
                numerics_policy,
            )
    }

    pub(crate) fn set_backward_kernel_geometry_label(
        &mut self,
        batch: usize,
        label: &str,
    ) -> Result<()> {
        self.time_mix
            .set_backward_kernel_geometry_label(batch, label)
    }

    pub(crate) fn set_backward_kernel_geometry_label_for_numerics(
        &mut self,
        batch: usize,
        label: &str,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<()> {
        self.time_mix
            .set_backward_kernel_geometry_label_for_numerics(batch, label, numerics_policy)
    }

    pub(crate) fn available_backward_segment_schedule_labels(
        &self,
        batch: usize,
    ) -> Result<Vec<String>> {
        self.time_mix
            .available_backward_segment_schedule_labels(batch)
    }

    pub(crate) fn backward_segment_schedule_factor_labels(
        &self,
        batch: usize,
        label: &str,
    ) -> Result<(String, String, Option<String>)> {
        self.time_mix
            .backward_segment_schedule_factor_labels(batch, label)
    }

    pub(crate) fn compose_backward_segment_schedule_label(
        &self,
        batch: usize,
        state_label: &str,
        projection_label: &str,
        low_rank_fan_in_label: Option<&str>,
    ) -> Result<Option<String>> {
        self.time_mix.compose_backward_segment_schedule_label(
            batch,
            state_label,
            projection_label,
            low_rank_fan_in_label,
        )
    }

    pub(crate) fn backward_segment_fusion_depth_neighbor_labels_for_geometry(
        &self,
        batch: usize,
        current_label: &str,
        geometry_label: &str,
    ) -> Result<Vec<String>> {
        self.time_mix
            .backward_segment_fusion_depth_neighbor_labels_for_geometry(
                batch,
                current_label,
                geometry_label,
            )
    }

    pub(crate) fn set_backward_segment_schedule_label(
        &mut self,
        batch: usize,
        label: &str,
    ) -> Result<()> {
        self.time_mix
            .set_backward_segment_schedule_label(batch, label)
    }

    pub fn low_rank_ranks(&self) -> Option<(usize, usize, usize)> {
        self.time_mix.low_rank_ranks()
    }

    pub(crate) fn low_rank_fp16_parameter_storage_active(&self) -> bool {
        self.time_mix.low_rank_fp16_parameter_storage_active()
    }

    pub(crate) fn low_rank_native_fp16_backward_compute_active(&self) -> bool {
        self.time_mix.low_rank_native_fp16_backward_compute_active()
    }

    pub(crate) fn low_rank_native_fp16_parameter_grad_compute_active(&self) -> bool {
        self.time_mix
            .low_rank_native_fp16_parameter_grad_compute_active()
    }

    pub(crate) fn low_rank_parameter_grad_arithmetic(&self) -> RwkvLowRankParameterGradArithmetic {
        self.time_mix.low_rank_parameter_grad_arithmetic()
    }

    pub(crate) fn projection_native_fp16_backward_compute_active(&self) -> bool {
        self.time_mix
            .projection_native_fp16_backward_compute_active()
    }

    pub(crate) fn low_rank_fp16_full_forward_first_stage_arm_label(&self) -> Option<&'static str> {
        self.time_mix
            .low_rank_fp16_full_forward_first_stage_arm_label()
    }

    pub(crate) fn trainables(&self) -> Result<Vec<RwkvTrainableRef<'_>>> {
        let mut trainables = vec![
            RwkvTrainableRef {
                name: "ln1.weight",
                parameter: &self.ln1_weight,
                gradient: &self.grad_ln1_weight,
                len: self.width,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "ln1.bias",
                parameter: &self.ln1_bias,
                gradient: &self.grad_ln1_bias,
                len: self.width,
                decay_class: RwkvDecayClass::NoDecay,
            },
        ];
        trainables.extend(self.time_mix.trainables()?);
        trainables.extend(self.channel_mix.trainables());
        trainables.extend(self.deepembed_adapter.deepembed_trainables());
        Ok(trainables)
    }

    fn validate_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV cell Vulkan batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        Ok(())
    }
}

/// Public-state owner around `RwkvCellSliceOp`. The caller supplies and
/// receives only Hierarchos' packed recurrent state; cache/matrix splitting,
/// finite-preserving state clamp, and all state-edge gradient routing remain
/// Vulkan-local in the same submission as the cell graph.
pub struct RwkvPackedCellOp {
    device: VulkanDevice,
    cell: RwkvCellSliceOp,
    state: RwkvPackedStateOp,
    max_batch: usize,

    x: GpuBuffer,
    token_features: GpuBuffer,
    packed_state: GpuBuffer,
    grad_packed_new_state: GpuBuffer,
    grad_output: GpuBuffer,
    grad_output_total: GpuBuffer,
    zero_previous_v_first_grad: GpuBuffer,

    packed_channel_mix_state_forward_fused: Option<vulkan::ComputeKernel>,
    packed_forward_only_enabled: bool,
    packed_backward_rematerialization_enabled: bool,

    output_readback: GpuBuffer,
    grad_x_readback: GpuBuffer,
}

impl RwkvPackedCellOp {
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        cell_prefix: &str,
        deepembed_adapter_prefix: &str,
        head_size: usize,
        max_batch: usize,
        key_clamp: f32,
        deepembed_clamp: f32,
        state_mode: RwkvStateReadoutMode,
        state_clamp: f32,
    ) -> Result<Self> {
        let cell = RwkvCellSliceOp::from_model_package(
            device.clone(),
            model_dir,
            cell_prefix,
            deepembed_adapter_prefix,
            head_size,
            max_batch,
            key_clamp,
            deepembed_clamp,
        )?;
        let state = RwkvPackedStateOp::new(
            device.clone(),
            cell.width,
            cell.head_size,
            max_batch,
            state_mode,
            state_clamp,
        )?;
        Self::new(device, cell, state, max_batch)
    }

    fn new(
        device: VulkanDevice,
        cell: RwkvCellSliceOp,
        state: RwkvPackedStateOp,
        max_batch: usize,
    ) -> Result<Self> {
        if max_batch == 0 || max_batch > cell.max_batch {
            bail!(
                "packed cell max_batch must be in 1..={}; got {max_batch}",
                cell.max_batch
            );
        }
        if state.matrix_offset() + cell.head_size != state.state_size() {
            bail!("packed cell state geometry is inconsistent");
        }
        let vector_len = max_batch
            .checked_mul(cell.width)
            .context("packed cell vector capacity overflow")?;
        let token_len = max_batch
            .checked_mul(cell.token_feature_width)
            .context("packed cell token-feature capacity overflow")?;
        let packed_len = vector_len
            .checked_mul(state.state_size())
            .context("packed cell state capacity overflow")?;
        let forward_workgroup_size = match std::env::var(
            HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE_ENV,
        ) {
            Ok(raw) => match raw.parse::<u32>() {
                Ok(value @ (32 | 64 | 128 | 256)) => value,
                _ => bail!(
                    "{HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE_ENV} must be 32, 64, 128, or 256, got {raw:?}"
                ),
            },
            Err(std::env::VarError::NotPresent) => 256,
            Err(err) => bail!(
                "reading {HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE_ENV}: {err}"
            ),
        };
        let forward_spirv = match forward_workgroup_size {
            32 => PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_WG32_SPV,
            64 => PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_WG64_SPV,
            128 => PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_WG128_SPV,
            256 => PACKED_CELL_CHANNEL_MIX_STATE_FORWARD_FUSED_SPV,
            _ => unreachable!("validated packed-cell forward workgroup size"),
        };
        let packed_channel_mix_state_forward_fused =
            if cell.channel_mix.supports_full_forward_fusion()
                && device.supports_storage_buffer_bindings(19)
                && device.supports_compute_work_group_size_x(forward_workgroup_size)
            {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    forward_spirv,
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
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                        vulkan::BindingAccess::MayWrite,
                    ],
                    std::mem::size_of::<PackedCellChannelMixStatePush>() as u32,
                )?)
            } else {
                None
            };
        Ok(Self {
            x: GpuBuffer::zeros_f32(&device, vector_len)?,
            token_features: GpuBuffer::zeros_f32(&device, token_len)?,
            packed_state: GpuBuffer::zeros_f32(&device, packed_len)?,
            grad_packed_new_state: GpuBuffer::zeros_f32(&device, packed_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_output_total: GpuBuffer::zeros_f32(&device, vector_len)?,
            zero_previous_v_first_grad: GpuBuffer::zeros_f32(&device, vector_len)?,
            packed_channel_mix_state_forward_fused,
            packed_forward_only_enabled: true,
            packed_backward_rematerialization_enabled: false,
            output_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_x_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            device,
            cell,
            state,
            max_batch,
        })
    }

    pub fn forward_backward(
        &mut self,
        batch: usize,
        x: &[f32],
        token_features: &[f32],
        packed_state: &[f32],
        grad_output: &[f32],
        grad_packed_new_state: &[f32],
    ) -> Result<RwkvPackedCellResult> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.cell.width;
        let packed_len = vector_len * self.state.state_size();
        validate_len("packed-cell x", x, vector_len)?;
        validate_len(
            "packed-cell token_features",
            token_features,
            batch * self.cell.token_feature_width,
        )?;
        validate_len("packed-cell state", packed_state, packed_len)?;
        validate_len("packed-cell grad_output", grad_output, vector_len)?;
        validate_len(
            "packed-cell grad_packed_new_state",
            grad_packed_new_state,
            packed_len,
        )?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.x, x)?;
        commands.upload_f32(&self.token_features, token_features)?;
        commands.upload_f32(&self.packed_state, packed_state)?;
        commands.upload_f32(&self.grad_output, grad_output)?;
        commands.upload_f32(&self.grad_packed_new_state, grad_packed_new_state)?;

        self.record_forward(
            &mut commands,
            batch,
            &self.x,
            &self.token_features,
            &self.packed_state,
        )?;
        self.state.record_pack_backward_fused_add(
            &mut commands,
            batch,
            &self.cell.x_norm,
            self.cell.channel_mix.normalized_buffer(),
            self.cell.time_mix.value_buffer(),
            self.cell.channel_mix.output_buffer(),
            self.cell.time_mix.new_state_buffer(),
            &self.grad_packed_new_state,
            &self.grad_output,
            &self.grad_output_total,
        )?;
        self.cell.record_backward(
            &mut commands,
            batch,
            &self.x,
            self.state.previous_tm_buffer(),
            self.state.previous_cm_buffer(),
            self.state.matrix_state_buffer(),
            &self.token_features,
            self.state.grad_matrix_state_buffer(),
            &self.grad_output_total,
            Some(self.state.grad_x_norm_buffer()),
            Some(self.state.grad_x_norm2_buffer()),
            Some(self.state.grad_v_first_buffer()),
        )?;
        self.state.record_pack_input_grad(
            &mut commands,
            batch,
            self.cell.time_mix.full_grad_previous_buffer(),
            self.cell.channel_mix.grad_previous_buffer(),
            &self.zero_previous_v_first_grad,
            self.cell.time_mix.grad_state_buffer(),
        )?;

        commands.readback_f32(
            self.cell.channel_mix.output_buffer(),
            &self.output_readback,
            vector_len,
        )?;
        commands.readback_f32(&self.cell.grad_x, &self.grad_x_readback, vector_len)?;
        self.state.record_readback(&mut commands, batch)?;
        self.state
            .record_grad_packed_input_readback(&mut commands, batch)?;
        self.cell
            .deepembed_adapter
            .record_grad_input_readback(&mut commands, batch)?;
        commands.submit()?;

        let state_result = self.state.read_result(batch)?;
        Ok(RwkvPackedCellResult {
            output: self.output_readback.read_f32(vector_len)?,
            packed_new_state: state_result.packed_new_state,
            grad_x: self.grad_x_readback.read_f32(vector_len)?,
            grad_packed_state: self.state.read_grad_packed_input(batch)?,
            token_feature_grad: self.cell.deepembed_adapter.read_grad_input(batch)?,
        })
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        token_features: &GpuBuffer,
        packed_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        self.state.record_unpack(commands, batch, packed_state)?;
        self.cell.record_forward_before_channel_mix(
            commands,
            batch,
            x,
            self.state.previous_tm_buffer(),
            self.state.matrix_state_buffer(),
            token_features,
        )?;
        self.record_channel_mix_and_pack(commands, batch)
    }

    /// Forward-only transition with caller-owned destinations. TBPTT uses this
    /// to make the next history slot the recurrent kernel's actual output,
    /// removing the packed-state and cell-output copy seams between timesteps.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_forward_transition_into(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        token_features: &GpuBuffer,
        packed_state: &GpuBuffer,
        packed_new_state: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        if !self.packed_forward_only_active() {
            self.record_forward(commands, batch, x, token_features, packed_state)?;
            commands.copy_f32(
                self.state.packed_new_state_buffer(),
                packed_new_state,
                batch * self.cell.width * self.state.state_size(),
            )?;
            return commands.copy_f32(
                self.cell.channel_mix.output_buffer(),
                output,
                batch * self.cell.width,
            );
        }

        self.state
            .record_unpack_vectors(commands, batch, packed_state)?;
        self.cell
            .deepembed_adapter
            .record_forward(commands, batch, token_features)?;
        let fast_recorded = self
            .cell
            .time_mix
            .record_packed_forward_only_with_layer_norm_residual(
                commands,
                batch,
                packed_state,
                self.state.matrix_offset(),
                self.state.state_clamp(),
                x,
                &self.cell.ln1_weight,
                &self.cell.ln1_bias,
                &self.cell.x_norm,
                &self.cell.ln1_mean,
                &self.cell.ln1_rstd,
                self.state.previous_tm_buffer(),
                x,
                &self.cell.time_residual,
                packed_new_state,
                1.0e-5,
            )?;
        if !fast_recorded {
            bail!("packed-cell forward-only capability changed while recording a transition");
        }

        self.record_channel_mix_and_pack_with_flags(
            commands,
            batch,
            PACKED_CELL_FORWARD_ONLY_FLAG | PACKED_CELL_MATRIX_ALREADY_PACKED_FLAG,
            output,
            packed_new_state,
        )
    }

    pub(crate) fn packed_forward_only_active(&self) -> bool {
        self.packed_forward_only_enabled && self.cell.time_mix.can_record_packed_forward_only()
    }

    pub(crate) fn set_packed_forward_only_enabled(&mut self, enabled: bool) {
        self.packed_forward_only_enabled = enabled;
    }

    pub(crate) fn packed_backward_rematerialization_active(&self) -> bool {
        self.packed_backward_rematerialization_enabled
            && self
                .cell
                .time_mix
                .can_record_packed_backward_rematerialization()
    }

    pub(crate) fn set_packed_backward_rematerialization_enabled(&mut self, enabled: bool) {
        self.packed_backward_rematerialization_enabled = enabled;
    }

    fn record_channel_mix_and_pack(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
    ) -> Result<()> {
        self.record_channel_mix_and_pack_with_flags(
            commands,
            batch,
            0,
            self.cell.channel_mix.output_buffer(),
            self.state.packed_new_state_buffer(),
        )
    }

    fn record_channel_mix_and_pack_with_flags(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        flags: u32,
        output: &GpuBuffer,
        packed_new_state: &GpuBuffer,
    ) -> Result<()> {
        if let Some(kernel) = &self.packed_channel_mix_state_forward_fused {
            let channel_mix = &self.cell.channel_mix;
            let push = PackedCellChannelMixStatePush {
                rows: batch as u32,
                width: self.cell.width as u32,
                hidden_width: channel_mix.hidden_width() as u32,
                head_size: self.cell.head_size as u32,
                matrix_offset: self.state.matrix_offset() as u32,
                eps: 1.0e-5,
                key_clamp: channel_mix.key_clamp(),
                deepembed_clamp: channel_mix.deepembed_clamp(),
                state_clamp: self.state.state_clamp(),
                flags,
            };
            return kernel.record_dispatch(
                commands,
                &[
                    &self.cell.time_residual,
                    channel_mix.layer_norm_weight_buffer(),
                    channel_mix.layer_norm_bias_buffer(),
                    self.state.previous_cm_buffer(),
                    channel_mix.mix_k_buffer(),
                    channel_mix.key_weight_buffer(),
                    self.cell.deepembed_adapter.output_buffer(),
                    channel_mix.value_weight_buffer(),
                    &self.cell.x_norm,
                    self.cell.time_mix.value_buffer(),
                    self.cell.time_mix.new_state_buffer(),
                    channel_mix.normalized_buffer(),
                    channel_mix.norm_mean_buffer(),
                    channel_mix.norm_rstd_buffer(),
                    channel_mix.mixed_buffer(),
                    channel_mix.key_buffer(),
                    channel_mix.ffn_buffer(),
                    output,
                    packed_new_state,
                ],
                bytemuck::bytes_of(&push),
                [batch as u32, 1, 1],
            );
        }

        let forward_only_flags =
            PACKED_CELL_FORWARD_ONLY_FLAG | PACKED_CELL_MATRIX_ALREADY_PACKED_FLAG;
        if flags != 0 && flags != forward_only_flags {
            bail!("unsupported packed-cell channel/state flags {flags:#x}");
        }

        self.cell
            .record_channel_mix_forward(commands, batch, self.state.previous_cm_buffer())?;
        if flags == forward_only_flags {
            self.state.record_pack_vectors_into(
                commands,
                batch,
                &self.cell.x_norm,
                self.cell.channel_mix.normalized_buffer(),
                self.cell.time_mix.value_buffer(),
                self.cell.channel_mix.output_buffer(),
                packed_new_state,
            )?;
            return commands.copy_f32(
                self.cell.channel_mix.output_buffer(),
                output,
                batch * self.cell.width,
            );
        }
        self.state.record_pack(
            commands,
            batch,
            &self.cell.x_norm,
            self.cell.channel_mix.normalized_buffer(),
            self.cell.time_mix.value_buffer(),
            self.cell.channel_mix.output_buffer(),
            self.cell.time_mix.new_state_buffer(),
        )
    }

    /// Replay one reverse-mode cell from its packed history slot and immediately
    /// run backward. The opt-in packed arm restores only vector caches, rebuilds
    /// the legacy training tape from packed matrix state, and feeds that same
    /// packed matrix state directly into the fused state-backward kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_rematerialized_forward_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        token_features: &GpuBuffer,
        packed_state: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_packed_new_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        if !self.packed_backward_rematerialization_active() {
            self.record_forward(commands, batch, x, token_features, packed_state)?;
            return self.record_backward_after_forward(
                commands,
                batch,
                x,
                token_features,
                grad_output,
                grad_packed_new_state,
            );
        }

        self.state
            .record_unpack_vectors(commands, batch, packed_state)?;
        self.cell
            .deepembed_adapter
            .record_forward(commands, batch, token_features)?;
        let rematerialized = self
            .cell
            .time_mix
            .record_packed_backward_rematerialization_with_layer_norm_residual(
                commands,
                batch,
                packed_state,
                self.state.matrix_offset(),
                x,
                &self.cell.ln1_weight,
                &self.cell.ln1_bias,
                &self.cell.x_norm,
                &self.cell.ln1_mean,
                &self.cell.ln1_rstd,
                self.state.previous_tm_buffer(),
                x,
                &self.cell.time_residual,
                1.0e-5,
            )?;
        if !rematerialized {
            bail!("packed backward rematerialization capability changed while recording TBPTT");
        }
        self.cell
            .record_channel_mix_forward(commands, batch, self.state.previous_cm_buffer())?;

        self.state.record_pack_backward_fused_add(
            commands,
            batch,
            &self.cell.x_norm,
            self.cell.channel_mix.normalized_buffer(),
            self.cell.time_mix.value_buffer(),
            self.cell.channel_mix.output_buffer(),
            self.cell.time_mix.new_state_buffer(),
            grad_packed_new_state,
            grad_output,
            &self.grad_output_total,
        )?;
        self.cell.record_backward_from_packed_state(
            commands,
            batch,
            x,
            self.state.previous_tm_buffer(),
            self.state.previous_cm_buffer(),
            self.state.matrix_state_buffer(),
            packed_state,
            self.state.matrix_offset(),
            token_features,
            self.state.grad_matrix_state_buffer(),
            &self.grad_output_total,
            Some(self.state.grad_x_norm_buffer()),
            Some(self.state.grad_x_norm2_buffer()),
            Some(self.state.grad_v_first_buffer()),
        )?;
        self.state.record_pack_input_grad(
            commands,
            batch,
            self.cell.time_mix.full_grad_previous_buffer(),
            self.cell.channel_mix.grad_previous_buffer(),
            &self.zero_previous_v_first_grad,
            self.cell.time_mix.grad_state_buffer(),
        )
    }

    pub(crate) fn record_backward_after_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        token_features: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_packed_new_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        self.state.record_pack_backward_fused_add(
            commands,
            batch,
            &self.cell.x_norm,
            self.cell.channel_mix.normalized_buffer(),
            self.cell.time_mix.value_buffer(),
            self.cell.channel_mix.output_buffer(),
            self.cell.time_mix.new_state_buffer(),
            grad_packed_new_state,
            grad_output,
            &self.grad_output_total,
        )?;
        self.cell.record_backward(
            commands,
            batch,
            x,
            self.state.previous_tm_buffer(),
            self.state.previous_cm_buffer(),
            self.state.matrix_state_buffer(),
            token_features,
            self.state.grad_matrix_state_buffer(),
            &self.grad_output_total,
            Some(self.state.grad_x_norm_buffer()),
            Some(self.state.grad_x_norm2_buffer()),
            Some(self.state.grad_v_first_buffer()),
        )?;
        self.state.record_pack_input_grad(
            commands,
            batch,
            self.cell.time_mix.full_grad_previous_buffer(),
            self.cell.channel_mix.grad_previous_buffer(),
            &self.zero_previous_v_first_grad,
            self.cell.time_mix.grad_state_buffer(),
        )
    }

    pub(crate) fn grad_x_buffer(&self) -> &GpuBuffer {
        &self.cell.grad_x
    }

    pub(crate) fn grad_packed_state_buffer(&self) -> &GpuBuffer {
        self.state.grad_packed_input_buffer()
    }

    pub(crate) fn token_feature_grad_buffer(&self) -> &GpuBuffer {
        self.cell.deepembed_adapter.grad_input_buffer()
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn width(&self) -> usize {
        self.cell.width
    }

    pub fn state_size(&self) -> usize {
        self.state.state_size()
    }

    pub fn token_feature_width(&self) -> usize {
        self.cell.token_feature_width
    }

    pub fn backward_segment_schedule_label(&self, batch: usize) -> Option<String> {
        self.cell.backward_segment_schedule_label(batch)
    }

    pub fn backward_segment_schedule_was_autotuned(&self, batch: usize) -> bool {
        self.cell.backward_segment_schedule_was_autotuned(batch)
    }

    pub fn backward_kernel_geometry_label(&self, batch: usize) -> Option<&'static str> {
        self.cell.backward_kernel_geometry_label(batch)
    }

    pub fn numerics_policy(&self) -> RwkvNumericsPolicy {
        self.cell.numerics_policy()
    }

    pub(crate) fn supports_numerics_policy(&self, policy: RwkvNumericsPolicy) -> bool {
        self.cell.supports_numerics_policy(policy)
    }

    pub fn set_numerics_policy(&mut self, policy: RwkvNumericsPolicy) -> Result<()> {
        self.cell.set_numerics_policy(policy)
    }

    pub(crate) fn available_backward_kernel_geometry_labels(
        &self,
        batch: usize,
    ) -> Result<Vec<String>> {
        self.cell.available_backward_kernel_geometry_labels(batch)
    }

    pub(crate) fn available_backward_kernel_geometry_labels_for_numerics(
        &self,
        batch: usize,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<Vec<String>> {
        self.cell
            .available_backward_kernel_geometry_labels_for_numerics(batch, numerics_policy)
    }

    pub(crate) fn backward_segment_schedule_geometry_pair_available(
        &self,
        batch: usize,
        schedule_label: &str,
        geometry_label: &str,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<bool> {
        self.cell.backward_segment_schedule_geometry_pair_available(
            batch,
            schedule_label,
            geometry_label,
            numerics_policy,
        )
    }

    pub(crate) fn set_backward_kernel_geometry_label(
        &mut self,
        batch: usize,
        label: &str,
    ) -> Result<()> {
        self.cell.set_backward_kernel_geometry_label(batch, label)
    }

    pub(crate) fn set_backward_kernel_geometry_label_for_numerics(
        &mut self,
        batch: usize,
        label: &str,
        numerics_policy: RwkvNumericsPolicy,
    ) -> Result<()> {
        self.cell
            .set_backward_kernel_geometry_label_for_numerics(batch, label, numerics_policy)
    }

    pub(crate) fn available_backward_segment_schedule_labels(
        &self,
        batch: usize,
    ) -> Result<Vec<String>> {
        self.cell.available_backward_segment_schedule_labels(batch)
    }

    pub(crate) fn backward_segment_schedule_factor_labels(
        &self,
        batch: usize,
        label: &str,
    ) -> Result<(String, String, Option<String>)> {
        self.cell
            .backward_segment_schedule_factor_labels(batch, label)
    }

    pub(crate) fn compose_backward_segment_schedule_label(
        &self,
        batch: usize,
        state_label: &str,
        projection_label: &str,
        low_rank_fan_in_label: Option<&str>,
    ) -> Result<Option<String>> {
        self.cell.compose_backward_segment_schedule_label(
            batch,
            state_label,
            projection_label,
            low_rank_fan_in_label,
        )
    }

    pub(crate) fn backward_segment_fusion_depth_neighbor_labels_for_geometry(
        &self,
        batch: usize,
        current_label: &str,
        geometry_label: &str,
    ) -> Result<Vec<String>> {
        self.cell
            .backward_segment_fusion_depth_neighbor_labels_for_geometry(
                batch,
                current_label,
                geometry_label,
            )
    }

    pub(crate) fn set_backward_segment_schedule_label(
        &mut self,
        batch: usize,
        label: &str,
    ) -> Result<()> {
        self.cell.set_backward_segment_schedule_label(batch, label)
    }

    pub fn low_rank_ranks(&self) -> Option<(usize, usize, usize)> {
        self.cell.low_rank_ranks()
    }

    pub(crate) fn low_rank_fp16_parameter_storage_active(&self) -> bool {
        self.cell.low_rank_fp16_parameter_storage_active()
    }

    pub(crate) fn low_rank_native_fp16_backward_compute_active(&self) -> bool {
        self.cell.low_rank_native_fp16_backward_compute_active()
    }

    pub(crate) fn low_rank_native_fp16_parameter_grad_compute_active(&self) -> bool {
        self.cell
            .low_rank_native_fp16_parameter_grad_compute_active()
    }

    pub(crate) fn low_rank_parameter_grad_arithmetic(&self) -> RwkvLowRankParameterGradArithmetic {
        self.cell.low_rank_parameter_grad_arithmetic()
    }

    pub(crate) fn projection_native_fp16_backward_compute_active(&self) -> bool {
        self.cell.projection_native_fp16_backward_compute_active()
    }

    pub(crate) fn low_rank_fp16_full_forward_first_stage_arm_label(&self) -> Option<&'static str> {
        self.cell.low_rank_fp16_full_forward_first_stage_arm_label()
    }

    pub(crate) fn install_low_rank_fp16_parameter_mirrors(
        &mut self,
        mirrors: RwkvLowRankFp16ParameterMirrors,
    ) -> Result<()> {
        self.cell
            .time_mix
            .install_low_rank_fp16_parameter_mirrors(mirrors)
    }

    pub(crate) fn enable_low_rank_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.cell
            .time_mix
            .enable_low_rank_native_fp16_backward_compute()
    }

    pub(crate) fn enable_low_rank_native_fp16_parameter_grad_compute(
        &mut self,
        widen_product: bool,
        compensated_operands: bool,
    ) -> Result<()> {
        self.cell
            .time_mix
            .enable_low_rank_native_fp16_parameter_grad_compute(widen_product, compensated_operands)
    }

    pub(crate) fn enable_projection_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.cell
            .time_mix
            .enable_projection_native_fp16_backward_compute()
    }

    pub(crate) fn configure_backward_source_scale(
        &mut self,
        source_scale: f32,
        source_scaled_backward_domain: bool,
    ) -> Result<()> {
        self.cell
            .time_mix
            .configure_backward_source_scale(source_scale, source_scaled_backward_domain)
    }

    pub(crate) fn trainables(&self) -> Result<Vec<RwkvTrainableRef<'_>>> {
        self.cell.trainables()
    }

    pub fn parameter_snapshots(&self) -> Result<Vec<RwkvParameterSnapshot>> {
        self.trainables()?
            .into_iter()
            .map(|trainable| {
                Ok(RwkvParameterSnapshot {
                    name: trainable.name.to_string(),
                    values: trainable.parameter.read_f32(trainable.len)?,
                })
            })
            .collect()
    }

    pub fn export_model_package(
        &self,
        source_model_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        cell_prefix: &str,
        deepembed_adapter_prefix: &str,
    ) -> Result<()> {
        if cell_prefix.trim().is_empty() || deepembed_adapter_prefix.trim().is_empty() {
            bail!("RWKV cell and DeepEmbed adapter export prefixes must not be empty");
        }
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

        let source_checkpoint = source_model_dir.join("model.safetensors");
        let snapshots = self.parameter_snapshots()?;
        let mut replacements_owned = Vec::with_capacity(snapshots.len());
        for snapshot in snapshots {
            let tensor_name = if let Some(suffix) = snapshot.name.strip_prefix("deepembed.") {
                format!("{deepembed_adapter_prefix}.{suffix}")
            } else {
                format!("{cell_prefix}.{}", snapshot.name)
            };
            let (shape, _) = read_f32_tensor(&source_checkpoint, &tensor_name)
                .with_context(|| format!("reading source shape for {tensor_name}"))?;
            replacements_owned.push((tensor_name, shape, snapshot.values));
        }
        let replacements: Vec<(&str, &[usize], &[f32])> = replacements_owned
            .iter()
            .map(|(name, shape, values)| (name.as_str(), shape.as_slice(), values.as_slice()))
            .collect();
        replace_f32_tensors(
            &source_checkpoint,
            &output_dir.join("model.safetensors"),
            &replacements,
        )
    }

    fn validate_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "packed RWKV cell batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        Ok(())
    }
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "RWKV cell {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("RWKV cell {name} contains non-finite values");
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
