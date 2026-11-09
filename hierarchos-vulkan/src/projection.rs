use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{
    read_f32_tensor, replace_f32_tensors, vendor, vulkan, wmma, AdamWHyperParams, GpuBuffer,
    VulkanDevice,
};

const LINEAR_WEIGHT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_weight_grad.spv");
const LINEAR_INPUT_GRAD_SPV: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
const BIAS_GRAD_SPV: &[u8] = include_bytes!("../shaders/bias_grad.spv");

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

    linear_forward: vendor::VendorMatmulKernel,
    linear_bias_forward: vendor::VendorMatmulKernel,
    linear_weight_grad: vulkan::ComputeKernel,
    linear_input_grad: vulkan::ComputeKernel,
    bias_grad: vulkan::ComputeKernel,
    adamw: vulkan::ComputeKernel,
    cpu_adamw: crate::cpu_adamw::CpuAdamW,
    cpu_small_row_gemm: Option<(vulkan::ComputeKernel, vulkan::ComputeKernel)>,
    wmma: Option<ProjectionWmma>,
}

/// Opt-in cooperative-matrix execution for one projection.
///
/// The canonical FP32 master parameter, its AdamW moments and the checkpoint
/// layout are untouched: `weight_f16` is an execution-only mirror refreshed from
/// the master after every optimizer step, exactly like the existing
/// `VulkanFp32MasterParameterMirror` contract. Forward, weight-gradient and
/// input-gradient GEMMs share it, so one projection step costs one mirror
/// refresh rather than one conversion per GEMM.
struct ProjectionWmma {
    gemm: wmma::WmmaGemm,
    weight_f16: GpuBuffer,
    input_f16: GpuBuffer,
    grad_output_f16: GpuBuffer,
}

impl ProjectionWmma {
    fn new(
        device: &VulkanDevice,
        mode: wmma::WmmaMode,
        input_len: usize,
        output_len: usize,
        weight_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            gemm: wmma::WmmaGemm::new(device, mode)?,
            weight_f16: wmma::WmmaGemm::new_half_buffer(device, weight_len)?,
            input_f16: wmma::WmmaGemm::new_half_buffer(device, input_len)?,
            grad_output_f16: wmma::WmmaGemm::new_half_buffer(device, output_len)?,
        })
    }

    /// Refresh the FP16 execution mirror from the FP32 master inside `batch`.
    fn record_refresh_weight(
        &self,
        batch: &mut vulkan::ComputeBatch,
        master: &GpuBuffer,
        weight_len: usize,
    ) -> Result<()> {
        self.gemm
            .record_conversion(batch, master, &self.weight_f16, weight_len)
    }
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

        // The policy is opt-in through HIERARCHOS_WMMA_MODE (set by
        // `--wmma-mode`). When it is off, or when this projection carries a bias
        // the fused FP32 kernel handles, the existing arithmetic is unchanged.
        let wmma_execution = match wmma::WmmaMode::from_env()? {
            wmma::WmmaMode::Off => None,
            mode => Some(ProjectionWmma::new(
                &device,
                mode,
                input_len,
                output_len,
                weight_len,
            )?),
        };

        Ok(Self {
            linear_forward: vendor::VendorMatmulKernel::new_with_access(
                &device,
                vendor::VendorKernelFamily::LinearForward,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
            )?,
            linear_bias_forward: vendor::VendorMatmulKernel::new(
                &device,
                vendor::VendorKernelFamily::LinearBiasForward,
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
            adamw: vendor::adamw_kernel(&device)?,
            cpu_adamw: crate::cpu_adamw::CpuAdamW::new(&device)?,
            cpu_small_row_gemm: None,
            wmma: wmma_execution,
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

    /// Match the CPU reference's measured small-row GEMM arithmetic.
    ///
    /// On the AVX2 oracle, three rows with eight input features select two
    /// interleaved FMA accumulators, including when the output is wide. Other
    /// shapes retain their regular native kernel. This is an explicit policy
    /// so existing projection users retain their established arithmetic.
    pub fn set_cpu_reference_math(&mut self, enabled: bool) -> Result<()> {
        self.cpu_small_row_gemm = if enabled {
            Some((
                vulkan::ComputeKernel::new(
                    &self.device,
                    include_bytes!("../shaders/linear_forward_lane2.spv"),
                    3,
                    12,
                )?,
                vulkan::ComputeKernel::new(
                    &self.device,
                    include_bytes!("../shaders/linear_bias_forward_lane2.spv"),
                    4,
                    12,
                )?,
            ))
        } else {
            None
        };
        Ok(())
    }

    fn record_forward(&self, batch: &mut vulkan::ComputeBatch, rows: usize) -> Result<()> {
        // A fused-bias projection keeps its existing kernel: the WMMA GEMM emits
        // the unbiased product only, and silently dropping the bias would change
        // the model. The fallback is counted so profiling shows it.
        if let Some(execution) = &self.wmma {
            if self.bias.is_none() {
                if let Some(arm) =
                    execution
                        .gemm
                        .select_arm(rows, self.output_dim, self.input_dim)
                {
                    execution.gemm.record_conversion(
                        batch,
                        &self.input,
                        &execution.input_f16,
                        rows * self.input_dim,
                    )?;
                    execution.gemm.record_gemm(
                        batch,
                        arm,
                        &execution.input_f16,
                        &execution.weight_f16,
                        &self.output,
                        rows,
                        self.output_dim,
                        self.input_dim,
                        wmma::GemmLayout {
                            transposed_a: false,
                            transposed_b: true,
                        },
                    )?;
                    return Ok(());
                }
            }
            execution.gemm.note_shape_fallback();
        }
        let push = LinearPush {
            rows: rows as u32,
            input_dim: self.input_dim as u32,
            output_dim: self.output_dim as u32,
        };
        let grid = [div_ceil_u32(self.output_dim, 16), div_ceil_u32(rows, 16), 1];
        let cpu = self.cpu_small_row_gemm.as_ref().filter(|_| {
            crate::transformer::aten_vector_width() == 8
                && rows == 3
                && self.input_dim == 8
                && matches!(self.output_dim, 8 | 16 | 32 | 64)
        });
        if let Some(bias) = &self.bias {
            let bindings = [&self.input, &self.weight, bias, &self.output];
            if let Some((_, kernel)) = cpu {
                kernel.record_dispatch(batch, &bindings, bytemuck::bytes_of(&push), grid)
            } else {
                self.linear_bias_forward.record_dispatch(
                    batch,
                    &bindings,
                    bytemuck::bytes_of(&push),
                    grid,
                )
            }
        } else {
            let bindings = [&self.input, &self.weight, &self.output];
            if let Some((kernel, _)) = cpu {
                kernel.record_dispatch(batch, &bindings, bytemuck::bytes_of(&push), grid)
            } else {
                self.linear_forward.record_dispatch(
                    batch,
                    &bindings,
                    bytemuck::bytes_of(&push),
                    grid,
                )
            }
        }
    }

    /// Read-only projection for a composed training graph. A downstream loss
    /// can return its adjoint to `train_step`; this call advances no optimizer.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if !input.len().is_multiple_of(self.input_dim) {
            bail!("projection input length is not divisible by input_dim");
        }
        let rows = input.len() / self.input_dim;
        if rows == 0 || rows > self.max_rows || input.iter().any(|value| !value.is_finite()) {
            bail!(
                "projection input must contain 1..={} finite rows",
                self.max_rows
            );
        }
        let mut batch = vulkan::ComputeBatch::new(&self.device)?;
        batch.upload_f32(&self.input, input)?;
        self.record_forward(&mut batch, rows)?;
        batch.readback_f32(&self.output, &self.output_readback, rows * self.output_dim)?;
        batch.submit()?;
        self.output_readback.read_f32(rows * self.output_dim)
    }

    pub fn train_step(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<ProjectionStepResult> {
        self.train_step_impl(input, grad_output, hyper, None)
    }

    pub fn train_step_with_optimizer(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        hyper: crate::CpuAdamWHyperParams,
    ) -> Result<ProjectionStepResult> {
        hyper.validate()?;
        if hyper.weight_decay as f32 != self.matrix_weight_decay {
            bail!("projection CPU optimizer weight decay differs from construction");
        }
        self.train_step_impl(input, grad_output, hyper.as_f32(), Some(hyper))
    }

    fn train_step_impl(
        &mut self,
        input: &[f32],
        grad_output: &[f32],
        hyper: AdamWHyperParams,
        cpu_policy: Option<crate::CpuAdamWHyperParams>,
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
        self.record_forward(&mut batch, rows)?;
        if self.bias.is_some() {
            let bias_push = BiasPush {
                rows: rows as u32,
                dim: self.output_dim as u32,
            };
            self.bias_grad.record_dispatch(
                &mut batch,
                &[
                    &self.grad_output,
                    self.grad_bias.as_ref().expect("bias grad exists"),
                ],
                bytemuck::bytes_of(&bias_push),
                [div_ceil_u32(self.output_dim, 256), 1, 1],
            )?;
        }

        // Weight gradient: dW[out, in] = sum_batch grad_out[batch, out] * x[batch, in].
        // As a GEMM this is rows=out, columns=in, reduction=batch with A stored
        // [batch, out] (transposed) and B stored [batch, in] (not transposed).
        let wmma_grads = self.wmma.as_ref().filter(|_| self.bias.is_none());
        if let Some(execution) = wmma_grads {
            execution.gemm.record_conversion(
                &mut batch,
                &self.grad_output,
                &execution.grad_output_f16,
                rows * self.output_dim,
            )?;
        }
        let mut weight_grad_done = false;
        if let Some(execution) = wmma_grads {
            if let Some(arm) = execution
                .gemm
                .select_arm(self.output_dim, self.input_dim, rows)
            {
                execution.gemm.record_gemm(
                    &mut batch,
                    arm,
                    &execution.grad_output_f16,
                    &execution.input_f16,
                    &self.grad_weight,
                    self.output_dim,
                    self.input_dim,
                    rows,
                    wmma::GemmLayout {
                        transposed_a: true,
                        transposed_b: false,
                    },
                )?;
                weight_grad_done = true;
            } else {
                execution.gemm.note_shape_fallback();
            }
        }
        if !weight_grad_done {
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
        }
        let mut input_grad_done = false;
        if let Some(execution) = wmma_grads {
            if let Some(arm) = execution
                .gemm
                .select_arm(rows, self.input_dim, self.output_dim)
            {
                execution.gemm.record_gemm(
                    &mut batch,
                    arm,
                    &execution.grad_output_f16,
                    &execution.weight_f16,
                    &self.grad_input,
                    rows,
                    self.input_dim,
                    self.output_dim,
                    wmma::GemmLayout::default(),
                )?;
                input_grad_done = true;
            } else {
                execution.gemm.note_shape_fallback();
            }
        }
        if !input_grad_done {
            self.linear_input_grad.record_dispatch(
                &mut batch,
                &[&self.grad_output, &self.weight, &self.grad_input],
                bytemuck::bytes_of(&linear_push),
                [div_ceil_u32(self.input_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
        }

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
        if let Some(policy) = cpu_policy {
            self.cpu_adamw.record_step(
                &mut batch,
                &self.weight,
                &self.grad_weight,
                &self.weight_exp_avg,
                &self.weight_exp_avg_sq,
                self.output_dim * self.input_dim,
                next_step,
                policy,
                true,
            )?;
        } else {
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
        }
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
            if let Some(policy) = cpu_policy {
                self.cpu_adamw.record_step(
                    &mut batch,
                    bias,
                    grad_bias,
                    exp_avg,
                    exp_avg_sq,
                    self.output_dim,
                    next_step,
                    policy,
                    false,
                )?;
            } else {
                self.adamw.record_dispatch(
                    &mut batch,
                    &[bias, grad_bias, exp_avg, exp_avg_sq],
                    bytemuck::bytes_of(&bias_adam),
                    [div_ceil_u32(self.output_dim, 256), 1, 1],
                )?;
            }
        }
        // The FP32 master was just updated in this command buffer; refresh the
        // FP16 execution mirror behind it so the next step's GEMMs read the new
        // parameters without a separate submission or a second mirror.
        if let Some(execution) = &self.wmma {
            execution.record_refresh_weight(
                &mut batch,
                &self.weight,
                self.output_dim * self.input_dim,
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

    /// Cooperative-matrix (WMMA) dispatch accounting for this projection, or
    /// `None` when the policy is off. Shape/fused-bias fallbacks are counted so a
    /// profiler can see where the WMMA path did not apply.
    pub fn wmma_dispatch_counts(&self) -> Option<wmma::WmmaDispatchCounts> {
        self.wmma.as_ref().map(|execution| execution.gemm.dispatch_counts())
    }

    /// Which WMMA policy this projection negotiated.
    pub fn wmma_mode(&self) -> wmma::WmmaMode {
        self.wmma
            .as_ref()
            .map(|execution| execution.gemm.mode())
            .unwrap_or(wmma::WmmaMode::Off)
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
