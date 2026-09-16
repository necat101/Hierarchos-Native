use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{
    read_f32_tensor, replace_f32_tensors, vulkan, AdamWHyperParams, GpuBuffer, VulkanDevice,
};

const LINEAR_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_forward.spv");
const LINEAR_BIAS_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_bias_forward.spv");
const LINEAR_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_weight_grad.spv");
const LINEAR_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
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
pub struct ProjectionStepResult {
    pub step: u32,
    pub output: Vec<f32>,
    pub input_grad: Vec<f32>,
}

/// Trainable Vulkan affine projection with exact PyTorch `nn.Linear` row-major
/// `[out_features, in_features]` storage. This primitive covers the Hierarchos
/// manager/worker projections and the q/in/router projections without changing
/// checkpoint names or layouts.
pub struct LinearProjectionTrainer {
    device: VulkanDevice,
    input_dim: usize,
    output_dim: usize,
    max_rows: usize,
    step: u32,
    matrix_weight_decay: f32,

    weight: GpuBuffer,
    bias: Option<GpuBuffer>,
    input: GpuBuffer,
    output: GpuBuffer,
    grad_output: GpuBuffer,
    grad_weight: GpuBuffer,
    grad_bias: Option<GpuBuffer>,
    grad_input: GpuBuffer,
    weight_exp_avg: GpuBuffer,
    weight_exp_avg_sq: GpuBuffer,
    bias_exp_avg: Option<GpuBuffer>,
    bias_exp_avg_sq: Option<GpuBuffer>,
    output_readback: GpuBuffer,
    grad_input_readback: GpuBuffer,

    linear_forward: vulkan::ComputeKernel,
    linear_bias_forward: vulkan::ComputeKernel,
    linear_weight_grad: vulkan::ComputeKernel,
    linear_input_grad: vulkan::ComputeKernel,
    bias_grad: vulkan::ComputeKernel,
    adamw: vulkan::ComputeKernel,
}

impl LinearProjectionTrainer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: VulkanDevice,
        input_dim: usize,
        output_dim: usize,
        max_rows: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        matrix_weight_decay: f32,
    ) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 || max_rows == 0 {
            bail!("linear projection dimensions and max_rows must be positive");
        }
        if !matrix_weight_decay.is_finite() || matrix_weight_decay < 0.0 {
            bail!("projection matrix weight decay must be finite and non-negative");
        }
        let weight_len = output_dim
            .checked_mul(input_dim)
            .context("projection weight size overflow")?;
        if weight.len() != weight_len {
            bail!(
                "projection weight has {} values; expected {} for [{}, {}]",
                weight.len(),
                weight_len,
                output_dim,
                input_dim
            );
        }
        if let Some(bias) = bias {
            if bias.len() != output_dim {
                bail!(
                    "projection bias has {} values; expected {}",
                    bias.len(),
                    output_dim
                );
            }
            if bias.iter().any(|value| !value.is_finite()) {
                bail!("projection bias contains non-finite values");
            }
        }
        if weight.iter().any(|value| !value.is_finite()) {
            bail!("projection weight contains non-finite values");
        }

        let input_len = max_rows
            .checked_mul(input_dim)
            .context("projection input capacity overflow")?;
        let output_len = max_rows
            .checked_mul(output_dim)
            .context("projection output capacity overflow")?;

        let bias_buffer = bias
            .map(|values| GpuBuffer::from_f32(&device, values))
            .transpose()?;
        let grad_bias = bias
            .map(|_| GpuBuffer::zeros_f32(&device, output_dim))
            .transpose()?;
        let bias_exp_avg = bias
            .map(|_| GpuBuffer::zeros_f32(&device, output_dim))
            .transpose()?;
        let bias_exp_avg_sq = bias
            .map(|_| GpuBuffer::zeros_f32(&device, output_dim))
            .transpose()?;

        Ok(Self {
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
            weight: GpuBuffer::from_f32(&device, weight)?,
            bias: bias_buffer,
            input: GpuBuffer::zeros_f32(&device, input_len)?,
            output: GpuBuffer::zeros_f32(&device, output_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, output_len)?,
            grad_weight: GpuBuffer::zeros_f32(&device, weight_len)?,
            grad_bias,
            grad_input: GpuBuffer::zeros_f32(&device, input_len)?,
            weight_exp_avg: GpuBuffer::zeros_f32(&device, weight_len)?,
            weight_exp_avg_sq: GpuBuffer::zeros_f32(&device, weight_len)?,
            bias_exp_avg,
            bias_exp_avg_sq,
            output_readback: GpuBuffer::zeros_host_f32(&device, output_len)?,
            grad_input_readback: GpuBuffer::zeros_host_f32(&device, input_len)?,
            device,
            input_dim,
            output_dim,
            max_rows,
            step: 0,
            matrix_weight_decay,
        })
    }

    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        has_bias: bool,
        max_rows: usize,
        matrix_weight_decay: f32,
    ) -> Result<Self> {
        validate_prefix(prefix)?;
        let tensor_path = model_dir.as_ref().join("model.safetensors");
        let weight_name = format!("{prefix}.weight");
        let (weight_shape, weight) = read_f32_tensor(&tensor_path, &weight_name)?;
        if weight_shape.len() != 2 {
            bail!("projection {weight_name:?} must be rank-2 [out, in], got {weight_shape:?}");
        }
        let output_dim = weight_shape[0];
        let input_dim = weight_shape[1];
        let bias = if has_bias {
            let bias_name = format!("{prefix}.bias");
            let (bias_shape, bias) = read_f32_tensor(&tensor_path, &bias_name)?;
            if bias_shape != vec![output_dim] {
                bail!("projection {bias_name:?} has shape {bias_shape:?}; expected [{output_dim}]");
            }
            Some(bias)
        } else {
            None
        };
        Self::new(
            device,
            input_dim,
            output_dim,
            max_rows,
            &weight,
            bias.as_deref(),
            matrix_weight_decay,
        )
    }

    pub fn train_step(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<ProjectionStepResult> {
        hyper.validate()?;
        if !input.len().is_multiple_of(self.input_dim) {
            bail!(
                "projection input length {} is not divisible by input_dim {}",
                input.len(),
                self.input_dim
            );
        }
        let rows = input.len() / self.input_dim;
        if rows == 0 || rows > self.max_rows {
            bail!(
                "projection batch has {rows} rows; capacity is 1..={}",
                self.max_rows
            );
        }
        let expected_grad = rows
            .checked_mul(self.output_dim)
            .context("projection gradient size overflow")?;
        if grad_output.len() != expected_grad {
            bail!(
                "projection grad_output has {} values; expected {} for [{}, {}]",
                grad_output.len(),
                expected_grad,
                rows,
                self.output_dim
            );
        }
        if input
            .iter()
            .chain(grad_output)
            .any(|value| !value.is_finite())
        {
            bail!("projection input/gradient contains non-finite values");
        }

        let mut batch = vulkan::ComputeBatch::new(&self.device)?;
        batch.upload_f32(&self.input, input)?;
        batch.upload_f32(&self.grad_output, grad_output)?;
        let linear_push = LinearPush {
            rows: rows as u32,
            input_dim: self.input_dim as u32,
            output_dim: self.output_dim as u32,
        };
        if let Some(bias) = &self.bias {
            let bias_push = BiasPush {
                rows: rows as u32,
                dim: self.output_dim as u32,
            };
            self.linear_bias_forward.record_dispatch(
                &mut batch,
                &[&self.input, &self.weight, bias, &self.output],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
            self.bias_grad.record_dispatch(
                &mut batch,
                &[
                    &self.grad_output,
                    self.grad_bias.as_ref().expect("bias grad exists"),
                ],
                bytemuck::bytes_of(&bias_push),
                [div_ceil_u32(self.output_dim, 256), 1, 1],
            )?;
        } else {
            self.linear_forward.record_dispatch(
                &mut batch,
                &[&self.input, &self.weight, &self.output],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
        }

        self.linear_weight_grad.record_dispatch(
            &mut batch,
            &[&self.input, &self.grad_output, &self.grad_weight],
            bytemuck::bytes_of(&linear_push),
            [
                div_ceil_u32(self.input_dim, 16),
                div_ceil_u32(self.output_dim, 16),
                1,
            ],
        )?;
        self.linear_input_grad.record_dispatch(
            &mut batch,
            &[&self.grad_output, &self.weight, &self.grad_input],
            bytemuck::bytes_of(&linear_push),
            [div_ceil_u32(self.input_dim, 16), div_ceil_u32(rows, 16), 1],
        )?;

        let next_step = self.step.checked_add(1).context("AdamW step overflow")?;
        let weight_adam = AdamWPush {
            len: (self.output_dim * self.input_dim) as u32,
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
                &self.weight,
                &self.grad_weight,
                &self.weight_exp_avg,
                &self.weight_exp_avg_sq,
            ],
            bytemuck::bytes_of(&weight_adam),
            [div_ceil_u32(self.output_dim * self.input_dim, 256), 1, 1],
        )?;
        if let (Some(bias), Some(grad_bias), Some(exp_avg), Some(exp_avg_sq)) = (
            &self.bias,
            &self.grad_bias,
            &self.bias_exp_avg,
            &self.bias_exp_avg_sq,
        ) {
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
                &[bias, grad_bias, exp_avg, exp_avg_sq],
                bytemuck::bytes_of(&bias_adam),
                [div_ceil_u32(self.output_dim, 256), 1, 1],
            )?;
        }
        batch.readback_f32(&self.output, &self.output_readback, rows * self.output_dim)?;
        batch.readback_f32(
            &self.grad_input,
            &self.grad_input_readback,
            rows * self.input_dim,
        )?;
        batch.submit()?;
        self.step = next_step;

        Ok(ProjectionStepResult {
            step: self.step,
            output: self.output_readback.read_f32(rows * self.output_dim)?,
            input_grad: self.grad_input_readback.read_f32(rows * self.input_dim)?,
        })
    }

    pub fn weights(&self) -> Result<Vec<f32>> {
        self.weight.read_f32(self.output_dim * self.input_dim)
    }

    pub fn bias_values(&self) -> Result<Option<Vec<f32>>> {
        self.bias
            .as_ref()
            .map(|bias| bias.read_f32(self.output_dim))
            .transpose()
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

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

        let weight = self.weights()?;
        let weight_name = format!("{prefix}.weight");
        let weight_shape = [self.output_dim, self.input_dim];
        if let Some(bias) = self.bias_values()? {
            let bias_name = format!("{prefix}.bias");
            let bias_shape = [self.output_dim];
            replace_f32_tensors(
                &source_model_dir.join("model.safetensors"),
                &output_dir.join("model.safetensors"),
                &[
                    (&weight_name, &weight_shape, &weight),
                    (&bias_name, &bias_shape, &bias),
                ],
            )?;
        } else {
            replace_f32_tensors(
                &source_model_dir.join("model.safetensors"),
                &output_dir.join("model.safetensors"),
                &[(&weight_name, &weight_shape, &weight)],
            )?;
        }
        Ok(())
    }
}

fn validate_prefix(prefix: &str) -> Result<()> {
    if prefix.trim().is_empty() {
        bail!("projection tensor prefix must not be empty");
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
