// These record seams are intentionally ahead of their caller: the ownership
// graph is live now, while the next TBPTT refactor will consume the forward /
// backward methods from one caller-owned command buffer.
#![allow(dead_code)]

use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::mixed_precision::VulkanParameterStorageMirror;
use crate::rwkv_optimizer::{
    RwkvDecayClass, RwkvParameterStorageMirrorBinding, RwkvPersistentAdamW, RwkvTrainableRef,
};
use crate::{
    read_f32_tensor, vulkan, AdamWHyperParams, AdamWOptimizerState, GpuBuffer,
    RwkvOptimizerStepResult, RwkvParameterSnapshot, VulkanDevice, VulkanParameterStorageFormat,
};

const LINEAR_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_forward.spv");
const LINEAR_FORWARD_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/linear_forward_fp16_packed.spv");
const LINEAR_BIAS_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_bias_forward.spv");
const LINEAR_BIAS_FORWARD_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/linear_bias_forward_fp16_packed.spv");
const LINEAR_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_weight_grad.spv");
const LINEAR_WEIGHT_GRAD_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/linear_weight_grad_fp16_native_compute.spv");
const LINEAR_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
const LINEAR_INPUT_GRAD_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/linear_input_grad_fp16_packed.spv");
const LINEAR_INPUT_GRAD_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/linear_input_grad_fp16_native_compute.spv");
const BIAS_GRAD_SPV: &[u8] = include_bytes!("../shaders/bias_grad.spv");
const BIAS_GRAD_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/bias_grad_fp16_native_compute.spv");
const PACKED_STATE_SLOT_FORWARD_SPV: &[u8] =
    include_bytes!("../shaders/packed_state_slot_forward.spv");
const PACKED_STATE_SLOT_BACKWARD_SPV: &[u8] =
    include_bytes!("../shaders/packed_state_slot_backward.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasPush {
    rows: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackedStateSlotPush {
    vector_len: u32,
    state_size: u32,
    slot: u32,
}

/// Differentiable view of one `[batch, width, state_size]` packed-state slot.
/// Hierarchos' Python `state_hidden()` is exactly this operation: coherent-v9
/// explicit-output reads slot 3, while the legacy input-cache contract reads
/// slot 0. The backward kernel adds into an existing packed-state gradient so
/// recurrent BPTT and manager/worker feedback can meet without a CPU scatter.
pub(crate) struct GraphPackedStateSlotOp {
    width: usize,
    state_size: usize,
    slot: usize,
    max_batch: usize,
    hidden: GpuBuffer,
    forward: vulkan::ComputeKernel,
    backward: vulkan::ComputeKernel,
}

impl GraphPackedStateSlotOp {
    pub(crate) fn new(
        device: VulkanDevice,
        width: usize,
        state_size: usize,
        slot: usize,
        max_batch: usize,
    ) -> Result<Self> {
        if width == 0 || state_size == 0 || max_batch == 0 || slot >= state_size {
            bail!(
                "packed-state hidden view requires positive width/state/batch and slot < state_size"
            );
        }
        let hidden_len = width
            .checked_mul(max_batch)
            .context("packed-state hidden view capacity overflow")?;
        Ok(Self {
            width,
            state_size,
            slot,
            max_batch,
            hidden: GpuBuffer::zeros_f32(&device, hidden_len)?,
            forward: vulkan::ComputeKernel::new(
                &device,
                PACKED_STATE_SLOT_FORWARD_SPV,
                2,
                std::mem::size_of::<PackedStateSlotPush>() as u32,
            )?,
            backward: vulkan::ComputeKernel::new(
                &device,
                PACKED_STATE_SLOT_BACKWARD_SPV,
                2,
                std::mem::size_of::<PackedStateSlotPush>() as u32,
            )?,
        })
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        packed_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let push = self.push(vector_len);
        self.forward.record_dispatch(
            commands,
            &[packed_state, &self.hidden],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(vector_len, 256), 1, 1],
        )
    }

    pub(crate) fn record_backward_accumulate(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        grad_hidden: &GpuBuffer,
        grad_packed_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let push = self.push(vector_len);
        self.backward.record_dispatch(
            commands,
            &[grad_hidden, grad_packed_state],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(vector_len, 256), 1, 1],
        )
    }

    pub(crate) fn hidden_buffer(&self) -> &GpuBuffer {
        &self.hidden
    }

    fn validate_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "packed-state hidden view batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        Ok(())
    }

    fn push(&self, vector_len: usize) -> PackedStateSlotPush {
        PackedStateSlotPush {
            vector_len: vector_len as u32,
            state_size: self.state_size as u32,
            slot: self.slot as u32,
        }
    }
}

/// One checkpoint-bound `nn.Linear` node designed for a larger caller-owned
/// Vulkan graph. Its backward buffers are scratch: a full-model scheduler must
/// call `record_accumulate_gradients` immediately after every reuse so repeated
/// manager/worker invocations add into the persistent optimizer instead of
/// overwriting one another.
pub(crate) struct GraphProjectionOp {
    prefix: String,
    weight_name: String,
    bias_name: Option<String>,
    input_dim: usize,
    output_dim: usize,
    max_rows: usize,
    weight: GpuBuffer,
    bias: Option<GpuBuffer>,
    output: GpuBuffer,
    grad_weight: GpuBuffer,
    grad_bias: Option<GpuBuffer>,
    grad_input: GpuBuffer,
    fp16_weight_mirror: Option<VulkanParameterStorageMirror>,
    native_fp16_backward_compute: bool,
    native_fp16_input_adjoint_compute: bool,
    source_scaled_backward_domain: bool,
    linear_forward: vulkan::ComputeKernel,
    linear_forward_fp16_packed: vulkan::ComputeKernel,
    linear_bias_forward: vulkan::ComputeKernel,
    linear_bias_forward_fp16_packed: vulkan::ComputeKernel,
    linear_weight_grad: vulkan::ComputeKernel,
    linear_weight_grad_fp16_native_compute: Option<vulkan::ComputeKernel>,
    linear_input_grad: vulkan::ComputeKernel,
    linear_input_grad_fp16_packed: vulkan::ComputeKernel,
    linear_input_grad_fp16_native_compute: Option<vulkan::ComputeKernel>,
    bias_grad: vulkan::ComputeKernel,
    bias_grad_fp16_native_compute: Option<vulkan::ComputeKernel>,
}

impl GraphProjectionOp {
    pub(crate) fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        has_bias: bool,
        max_rows: usize,
    ) -> Result<Self> {
        if prefix.trim().is_empty() || max_rows == 0 {
            bail!("graph projection prefix and max_rows must be non-empty/positive");
        }
        let tensor_path = model_dir.as_ref().join("model.safetensors");
        let weight_name = format!("{prefix}.weight");
        let (shape, weight) = read_f32_tensor(&tensor_path, &weight_name)?;
        let [output_dim, input_dim] = shape.as_slice() else {
            bail!("projection {weight_name:?} must have shape [out, in], got {shape:?}");
        };
        let bias_name = has_bias.then(|| format!("{prefix}.bias"));
        let bias = if let Some(name) = bias_name.as_deref() {
            let (bias_shape, values) = read_f32_tensor(&tensor_path, name)?;
            if bias_shape != vec![*output_dim] {
                bail!(
                    "projection {name:?} has shape {bias_shape:?}; expected [{}]",
                    output_dim
                );
            }
            Some(values)
        } else {
            None
        };
        Self::new(
            device,
            prefix,
            *input_dim,
            *output_dim,
            max_rows,
            &weight,
            bias.as_deref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: VulkanDevice,
        prefix: &str,
        input_dim: usize,
        output_dim: usize,
        max_rows: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 || max_rows == 0 {
            bail!("graph projection dimensions and max_rows must be positive");
        }
        let weight_len = input_dim
            .checked_mul(output_dim)
            .context("graph projection weight size overflow")?;
        if weight.len() != weight_len || weight.iter().any(|value| !value.is_finite()) {
            bail!("graph projection {prefix:?} has invalid/non-finite weight data");
        }
        if let Some(values) = bias {
            if values.len() != output_dim || values.iter().any(|value| !value.is_finite()) {
                bail!("graph projection {prefix:?} has invalid/non-finite bias data");
            }
        }
        let output_len = max_rows
            .checked_mul(output_dim)
            .context("graph projection output capacity overflow")?;
        let input_len = max_rows
            .checked_mul(input_dim)
            .context("graph projection input capacity overflow")?;
        Ok(Self {
            prefix: prefix.to_string(),
            weight_name: format!("{prefix}.weight"),
            bias_name: bias.map(|_| format!("{prefix}.bias")),
            input_dim,
            output_dim,
            max_rows,
            weight: GpuBuffer::from_f32(&device, weight)?,
            bias: bias
                .map(|values| GpuBuffer::from_f32(&device, values))
                .transpose()?,
            output: GpuBuffer::zeros_f32(&device, output_len)?,
            grad_weight: GpuBuffer::zeros_f32(&device, weight_len)?,
            grad_bias: bias
                .map(|_| GpuBuffer::zeros_f32(&device, output_dim))
                .transpose()?,
            grad_input: GpuBuffer::zeros_f32(&device, input_len)?,
            fp16_weight_mirror: None,
            native_fp16_backward_compute: false,
            native_fp16_input_adjoint_compute: false,
            source_scaled_backward_domain: false,
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
            linear_forward_fp16_packed: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_FORWARD_FP16_PACKED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear_bias_forward: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_BIAS_FORWARD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear_bias_forward_fp16_packed: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_BIAS_FORWARD_FP16_PACKED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
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
            linear_weight_grad_fp16_native_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        LINEAR_WEIGHT_GRAD_FP16_NATIVE_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<LinearPush>() as u32,
                    )
                })
                .transpose()?,
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
            linear_input_grad_fp16_packed: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_INPUT_GRAD_FP16_PACKED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LinearPush>() as u32,
            )?,
            linear_input_grad_fp16_native_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        LINEAR_INPUT_GRAD_FP16_NATIVE_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<LinearPush>() as u32,
                    )
                })
                .transpose()?,
            bias_grad: vulkan::ComputeKernel::new(
                &device,
                BIAS_GRAD_SPV,
                2,
                std::mem::size_of::<BiasPush>() as u32,
            )?,
            bias_grad_fp16_native_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new(
                        &device,
                        BIAS_GRAD_FP16_NATIVE_COMPUTE_SPV,
                        2,
                        std::mem::size_of::<BiasPush>() as u32,
                    )
                })
                .transpose()?,
        })
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        input: &GpuBuffer,
    ) -> Result<()> {
        self.validate_rows(rows)?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: self.input_dim as u32,
            output_dim: self.output_dim as u32,
        };
        if let Some(mirror) = self.fp16_weight_mirror.as_ref() {
            if let Some(bias) = &self.bias {
                self.linear_bias_forward_fp16_packed.record_dispatch(
                    commands,
                    &[input, mirror.packed_storage(), bias, &self.output],
                    bytemuck::bytes_of(&push),
                    [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1],
                )?;
            } else {
                self.linear_forward_fp16_packed.record_dispatch(
                    commands,
                    &[input, mirror.packed_storage(), &self.output],
                    bytemuck::bytes_of(&push),
                    [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1],
                )?;
            }
        } else if let Some(bias) = &self.bias {
            self.linear_bias_forward.record_dispatch(
                commands,
                &[input, &self.weight, bias, &self.output],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
        } else {
            self.linear_forward.record_dispatch(
                commands,
                &[input, &self.weight, &self.output],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
        }
        Ok(())
    }

    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        input: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Result<()> {
        self.validate_rows(rows)?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: self.input_dim as u32,
            output_dim: self.output_dim as u32,
        };
        // GradScaler-style source scaling can make a tiny, cancelling
        // projection gradient representable in FP16 while still changing its
        // sign after the optimizer-side unscale. Keep the promoted native-FP16
        // dW/db path for ordinary backward domains, but use the canonical FP32
        // reduction when the source domain is explicitly scaled.
        let use_native_fp16_parameter_grad =
            self.native_fp16_backward_compute && !self.source_scaled_backward_domain;
        let weight_grad = if use_native_fp16_parameter_grad {
            self.linear_weight_grad_fp16_native_compute
                .as_ref()
                .context("native-FP16 projection dW was enabled without a kernel")?
        } else {
            &self.linear_weight_grad
        };
        weight_grad.record_dispatch(
            commands,
            &[input, grad_output, &self.grad_weight],
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(self.input_dim, 16),
                div_ceil_u32(self.output_dim, 16),
                1,
            ],
        )?;
        if let Some(mirror) = self.fp16_weight_mirror.as_ref() {
            let input_grad = if self.native_fp16_input_adjoint_compute {
                self.linear_input_grad_fp16_native_compute
                    .as_ref()
                    .context("native-FP16 projection dX was enabled without a kernel")?
            } else {
                &self.linear_input_grad_fp16_packed
            };
            input_grad.record_dispatch(
                commands,
                &[grad_output, mirror.packed_storage(), &self.grad_input],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(self.input_dim, 32), div_ceil_u32(rows, 16), 1],
            )?;
        } else {
            self.linear_input_grad.record_dispatch(
                commands,
                &[grad_output, &self.weight, &self.grad_input],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(self.input_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
        }
        if let Some(grad_bias) = &self.grad_bias {
            let bias_push = BiasPush {
                rows: rows as u32,
                dim: self.output_dim as u32,
            };
            let bias_grad = if use_native_fp16_parameter_grad {
                self.bias_grad_fp16_native_compute
                    .as_ref()
                    .context("native-FP16 projection db was enabled without a kernel")?
            } else {
                &self.bias_grad
            };
            bias_grad.record_dispatch(
                commands,
                &[grad_output, grad_bias],
                bytemuck::bytes_of(&bias_push),
                [div_ceil_u32(self.output_dim, 256), 1, 1],
            )?;
        }
        Ok(())
    }

    pub(crate) fn record_accumulate_gradients(
        &self,
        commands: &mut vulkan::ComputeBatch,
        optimizer: &RwkvPersistentAdamW,
    ) -> Result<()> {
        let trainables = self.trainables();
        optimizer.record_accumulate_many(commands, &trainables)
    }

    pub(crate) fn trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        let mut trainables = vec![RwkvTrainableRef {
            name: &self.weight_name,
            parameter: &self.weight,
            gradient: &self.grad_weight,
            len: self.input_dim * self.output_dim,
            decay_class: RwkvDecayClass::Decay,
        }];
        if let (Some(name), Some(parameter), Some(gradient)) =
            (&self.bias_name, &self.bias, &self.grad_bias)
        {
            trainables.push(RwkvTrainableRef {
                name,
                parameter,
                gradient,
                len: self.output_dim,
                decay_class: RwkvDecayClass::NoDecay,
            });
        }
        trainables
    }

    pub(crate) fn output_buffer(&self) -> &GpuBuffer {
        &self.output
    }

    pub(crate) fn grad_input_buffer(&self) -> &GpuBuffer {
        &self.grad_input
    }

    pub(crate) fn grad_weight_buffer(&self) -> &GpuBuffer {
        &self.grad_weight
    }

    pub(crate) fn weight_buffer(&self) -> &GpuBuffer {
        &self.weight
    }

    pub(crate) fn grad_bias_buffer(&self) -> Option<&GpuBuffer> {
        self.grad_bias.as_ref()
    }

    pub(crate) fn install_fp16_parameter_storage_mirror(
        &mut self,
        mirror: VulkanParameterStorageMirror,
    ) -> Result<()> {
        if mirror.format() != VulkanParameterStorageFormat::Fp16 {
            bail!(
                "projection {:?} execution mirror must be fp16; got {}",
                self.prefix,
                mirror.format().label()
            );
        }
        let expected_len = self.input_dim * self.output_dim;
        if mirror.len() != expected_len {
            bail!(
                "projection {:?} execution mirror has {} elements; expected {}",
                self.prefix,
                mirror.len(),
                expected_len
            );
        }
        if let Some(existing) = self.fp16_weight_mirror.as_ref() {
            if !existing
                .packed_storage()
                .shares_allocation_with(mirror.packed_storage())
            {
                bail!(
                    "projection {:?} already has a different fp16 execution mirror",
                    self.prefix
                );
            }
            return Ok(());
        }
        self.fp16_weight_mirror = Some(mirror);
        Ok(())
    }

    pub(crate) fn fp16_parameter_storage_active(&self) -> bool {
        self.fp16_weight_mirror.is_some()
    }

    pub(crate) fn enable_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.fp16_weight_mirror
            .as_ref()
            .context("native-FP16 projection backward requires an FP16 weight mirror")?;
        self.linear_weight_grad_fp16_native_compute
            .as_ref()
            .context("device cannot create native-FP16 projection dW")?;
        if self.bias.is_some() {
            self.bias_grad_fp16_native_compute
                .as_ref()
                .context("device cannot create native-FP16 projection db")?;
        }
        self.native_fp16_backward_compute = true;
        Ok(())
    }

    pub(crate) fn enable_native_fp16_input_adjoint_compute(&mut self) -> Result<()> {
        self.fp16_weight_mirror
            .as_ref()
            .context("native-FP16 projection dX requires an FP16 weight mirror")?;
        self.linear_input_grad_fp16_native_compute
            .as_ref()
            .context("device cannot create native-FP16 projection dX")?;
        self.native_fp16_input_adjoint_compute = true;
        Ok(())
    }

    pub(crate) fn configure_backward_source_domain(&mut self, source_scaled: bool) {
        self.source_scaled_backward_domain = source_scaled;
    }

    pub(crate) fn native_fp16_backward_compute_active(&self) -> bool {
        self.native_fp16_backward_compute
    }

    pub(crate) fn native_fp16_input_adjoint_compute_active(&self) -> bool {
        self.native_fp16_input_adjoint_compute
    }

    pub(crate) fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub(crate) fn output_dim(&self) -> usize {
        self.output_dim
    }

    fn validate_rows(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.max_rows {
            bail!(
                "graph projection {:?} rows must be in 1..={}; got {rows}",
                self.prefix,
                self.max_rows
            );
        }
        Ok(())
    }
}

/// Persistent owner for the six learned affine seams surrounding Hierarchos'
/// manager and worker recurrent cells. All ten tensors (six matrices + four
/// biases) are registered in one AdamW state, so repeated projection uses can
/// accumulate during a future end-to-end command graph and step exactly once.
pub(crate) struct ManagerWorkerProjectionGraph {
    pub(crate) l_feedback_proj: GraphProjectionOp,
    pub(crate) h_to_context: GraphProjectionOp,
    pub(crate) h_halt_proj: GraphProjectionOp,
    pub(crate) l_input_proj: GraphProjectionOp,
    pub(crate) context_drift_proj: GraphProjectionOp,
    pub(crate) l_to_out: GraphProjectionOp,
    pub(crate) optimizer: RwkvPersistentAdamW,
}

impl ManagerWorkerProjectionGraph {
    pub(crate) fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        max_rows: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let l_feedback_proj = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "l_feedback_proj",
            false,
            max_rows,
        )?;
        let h_to_context = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "h_to_context",
            true,
            max_rows,
        )?;
        let h_halt_proj = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "h_halt_proj",
            true,
            max_rows,
        )?;
        let l_input_proj = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "l_input_proj",
            true,
            max_rows,
        )?;
        let context_drift_proj = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "context_drift_proj",
            false,
            max_rows,
        )?;
        let l_to_out = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "l_to_out",
            true,
            max_rows,
        )?;

        let mut trainables = Vec::with_capacity(10);
        trainables.extend(l_feedback_proj.trainables());
        trainables.extend(h_to_context.trainables());
        trainables.extend(h_halt_proj.trainables());
        trainables.extend(l_input_proj.trainables());
        trainables.extend(context_drift_proj.trainables());
        trainables.extend(l_to_out.trainables());
        let optimizer = RwkvPersistentAdamW::new(device, &trainables)?;

        Ok(Self {
            l_feedback_proj,
            h_to_context,
            h_halt_proj,
            l_input_proj,
            context_drift_proj,
            l_to_out,
            optimizer,
        })
    }

    pub(crate) fn validate_topology(
        &self,
        context_dim: usize,
        h_hidden: usize,
        l_hidden: usize,
    ) -> Result<()> {
        let expected = [
            (
                "l_feedback_proj",
                self.l_feedback_proj.input_dim(),
                l_hidden,
                self.l_feedback_proj.output_dim(),
                h_hidden,
            ),
            (
                "h_to_context",
                self.h_to_context.input_dim(),
                h_hidden,
                self.h_to_context.output_dim(),
                context_dim,
            ),
            (
                "h_halt_proj",
                self.h_halt_proj.input_dim(),
                h_hidden,
                self.h_halt_proj.output_dim(),
                1,
            ),
            (
                "l_input_proj",
                self.l_input_proj.input_dim(),
                context_dim * 2,
                self.l_input_proj.output_dim(),
                l_hidden,
            ),
            (
                "context_drift_proj",
                self.context_drift_proj.input_dim(),
                l_hidden,
                self.context_drift_proj.output_dim(),
                context_dim,
            ),
            (
                "l_to_out",
                self.l_to_out.input_dim(),
                l_hidden,
                self.l_to_out.output_dim(),
                context_dim,
            ),
        ];
        for (name, actual_in, expected_in, actual_out, expected_out) in expected {
            if actual_in != expected_in || actual_out != expected_out {
                bail!(
                    "{name} has Vulkan shape [{actual_out}, {actual_in}]; expected [{expected_out}, {expected_in}]"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn tensor_count(&self) -> usize {
        10
    }

    pub(crate) fn trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        let mut trainables = Vec::with_capacity(self.tensor_count());
        trainables.extend(self.l_feedback_proj.trainables());
        trainables.extend(self.h_to_context.trainables());
        trainables.extend(self.h_halt_proj.trainables());
        trainables.extend(self.l_input_proj.trainables());
        trainables.extend(self.context_drift_proj.trainables());
        trainables.extend(self.l_to_out.trainables());
        trainables
    }

    pub(crate) fn install_fp16_parameter_storage_mirrors(
        &mut self,
        bindings: &[RwkvParameterStorageMirrorBinding],
    ) -> Result<()> {
        const PROJECTION_WEIGHT_NAMES: [&str; 6] = [
            "l_feedback_proj.weight",
            "h_to_context.weight",
            "h_halt_proj.weight",
            "l_input_proj.weight",
            "context_drift_proj.weight",
            "l_to_out.weight",
        ];
        if bindings.len() != PROJECTION_WEIGHT_NAMES.len() {
            bail!(
                "projection fp16 storage requires {} mirror bindings; got {}",
                PROJECTION_WEIGHT_NAMES.len(),
                bindings.len()
            );
        }
        let resolve = |name: &str| {
            bindings
                .iter()
                .find(|binding| binding.name == name)
                .map(|binding| binding.mirror.clone())
                .with_context(|| {
                    format!("missing projection mixed-precision mirror binding {name:?}")
                })
        };

        self.l_feedback_proj
            .install_fp16_parameter_storage_mirror(resolve(PROJECTION_WEIGHT_NAMES[0])?)?;
        self.h_to_context
            .install_fp16_parameter_storage_mirror(resolve(PROJECTION_WEIGHT_NAMES[1])?)?;
        self.h_halt_proj
            .install_fp16_parameter_storage_mirror(resolve(PROJECTION_WEIGHT_NAMES[2])?)?;
        self.l_input_proj
            .install_fp16_parameter_storage_mirror(resolve(PROJECTION_WEIGHT_NAMES[3])?)?;
        self.context_drift_proj
            .install_fp16_parameter_storage_mirror(resolve(PROJECTION_WEIGHT_NAMES[4])?)?;
        self.l_to_out
            .install_fp16_parameter_storage_mirror(resolve(PROJECTION_WEIGHT_NAMES[5])?)?;

        let Self {
            l_feedback_proj,
            h_to_context,
            h_halt_proj,
            l_input_proj,
            context_drift_proj,
            l_to_out,
            optimizer,
        } = self;
        let mut trainables = Vec::with_capacity(10);
        trainables.extend(l_feedback_proj.trainables());
        trainables.extend(h_to_context.trainables());
        trainables.extend(h_halt_proj.trainables());
        trainables.extend(l_input_proj.trainables());
        trainables.extend(context_drift_proj.trainables());
        trainables.extend(l_to_out.trainables());
        optimizer.attach_parameter_storage_mirrors(
            &trainables,
            VulkanParameterStorageFormat::Fp16,
            bindings,
        )?;
        Ok(())
    }

    pub(crate) fn fp16_parameter_storage_active(&self) -> bool {
        [
            &self.l_feedback_proj,
            &self.h_to_context,
            &self.h_halt_proj,
            &self.l_input_proj,
            &self.context_drift_proj,
            &self.l_to_out,
        ]
        .into_iter()
        .all(GraphProjectionOp::fp16_parameter_storage_active)
    }

    pub(crate) fn enable_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.l_feedback_proj.enable_native_fp16_backward_compute()?;
        self.h_to_context.enable_native_fp16_backward_compute()?;
        self.h_halt_proj.enable_native_fp16_backward_compute()?;
        self.l_input_proj.enable_native_fp16_backward_compute()?;
        self.context_drift_proj
            .enable_native_fp16_backward_compute()?;
        self.l_to_out.enable_native_fp16_backward_compute()?;
        Ok(())
    }

    pub(crate) fn enable_native_fp16_input_adjoint_compute(&mut self) -> Result<()> {
        self.l_feedback_proj
            .enable_native_fp16_input_adjoint_compute()?;
        self.h_to_context
            .enable_native_fp16_input_adjoint_compute()?;
        self.h_halt_proj
            .enable_native_fp16_input_adjoint_compute()?;
        self.l_input_proj
            .enable_native_fp16_input_adjoint_compute()?;
        self.context_drift_proj
            .enable_native_fp16_input_adjoint_compute()?;
        self.l_to_out.enable_native_fp16_input_adjoint_compute()?;
        Ok(())
    }

    pub(crate) fn configure_backward_source_domain(&mut self, source_scaled: bool) {
        self.l_feedback_proj
            .configure_backward_source_domain(source_scaled);
        self.h_to_context
            .configure_backward_source_domain(source_scaled);
        self.h_halt_proj
            .configure_backward_source_domain(source_scaled);
        self.l_input_proj
            .configure_backward_source_domain(source_scaled);
        self.context_drift_proj
            .configure_backward_source_domain(source_scaled);
        self.l_to_out
            .configure_backward_source_domain(source_scaled);
    }

    pub(crate) fn native_fp16_backward_compute_active(&self) -> bool {
        [
            &self.l_feedback_proj,
            &self.h_to_context,
            &self.h_halt_proj,
            &self.l_input_proj,
            &self.context_drift_proj,
            &self.l_to_out,
        ]
        .into_iter()
        .all(GraphProjectionOp::native_fp16_backward_compute_active)
    }

    pub(crate) fn native_fp16_input_adjoint_compute_active(&self) -> bool {
        [
            &self.l_feedback_proj,
            &self.h_to_context,
            &self.h_halt_proj,
            &self.l_input_proj,
            &self.context_drift_proj,
            &self.l_to_out,
        ]
        .into_iter()
        .all(GraphProjectionOp::native_fp16_input_adjoint_compute_active)
    }

    pub(crate) fn record_zero_grad(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        self.optimizer.record_zero_grad(commands)
    }

    pub(crate) fn record_step_and_readback(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        hyper: AdamWHyperParams,
    ) -> Result<RwkvOptimizerStepResult> {
        let Self {
            l_feedback_proj,
            h_to_context,
            h_halt_proj,
            l_input_proj,
            context_drift_proj,
            l_to_out,
            optimizer,
        } = self;
        let mut trainables = Vec::with_capacity(10);
        trainables.extend(l_feedback_proj.trainables());
        trainables.extend(h_to_context.trainables());
        trainables.extend(h_halt_proj.trainables());
        trainables.extend(l_input_proj.trainables());
        trainables.extend(context_drift_proj.trainables());
        trainables.extend(l_to_out.trainables());
        let result = optimizer.record_step(commands, &trainables, hyper)?;
        optimizer.record_parameter_readback(commands, &trainables)?;
        Ok(result)
    }

    pub(crate) fn parameter_snapshots(&self) -> Result<Vec<RwkvParameterSnapshot>> {
        self.trainables()
            .into_iter()
            .map(|trainable| {
                Ok(RwkvParameterSnapshot {
                    name: trainable.name.to_string(),
                    values: trainable.parameter.read_f32(trainable.len)?,
                })
            })
            .collect()
    }

    pub(crate) fn optimizer_state(&self) -> Result<AdamWOptimizerState> {
        self.optimizer.state_snapshot()
    }

    pub(crate) fn load_optimizer_state(&mut self, state: &AdamWOptimizerState) -> Result<()> {
        self.optimizer.load_state(state)
    }
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
