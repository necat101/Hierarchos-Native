//! Terminal training adjoints for the dedicated V4.1 graph.
//!
//! A LoRA step needs the loss gradient at the top of the graph before any block
//! adjoint can run: cross entropy over the logits, the frozen vocabulary head's
//! input gradient, and the final RMSNorm's input gradient. Each stage reuses the
//! arithmetic contract its forward kernel was qualified with, so the chain is
//! `logits -> head -> final norm -> Hyper-Connection collapse` with no
//! rearranged algebra. The Hyper-Connection, attention and MoE adjoints are
//! separate passes; nothing here claims a trainable adapter on its own.

use std::mem::size_of;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

const CROSS_ENTROPY: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_cross_entropy.spv");
const CROSS_ENTROPY_LANES16: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_cross_entropy_lanes16.spv");
const RMS_NORM_INPUT_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_rms_norm_input_grad.spv");
const LINEAR_INPUT_GRAD: &[u8] = include_bytes!("../../shaders/linear_input_grad.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CrossEntropyPush {
    rows: u32,
    vocab_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearInputGradPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormInputGradPush {
    rows: u32,
    dim: u32,
    eps: f32,
}

/// Loss and the head's input gradient for one batch of rows.
pub struct DeepseekV41TerminalGradients {
    /// Per-row cross entropy in row order.
    pub row_loss: Vec<f32>,
    /// Mean cross entropy over the rows, matching the graph's reduction.
    pub loss: f32,
    /// `d loss / d logits`, `rows x vocab`, FP32.
    pub grad_logits: GpuBuffer,
    /// `d loss / d hidden`, `rows x hidden`, FP32. The head weight is frozen,
    /// so this is the whole head backward.
    pub grad_hidden: GpuBuffer,
}

/// The kernels every V4.1 training step starts with.
pub struct DeepseekV41TerminalAdjoints {
    device: VulkanDevice,
    cross_entropy: vulkan::ComputeKernel,
    head_input_grad: vulkan::ComputeKernel,
    norm_input_grad: vulkan::ComputeKernel,
}

impl DeepseekV41TerminalAdjoints {
    /// Loss over non-ignored target rows plus its full-layout logits adjoint.
    /// Compaction changes no row's softmax arithmetic. The loss fold keeps
    /// ignored zero rows in their original positions to preserve CPU NLL order.
    pub fn masked_cross_entropy(
        &self,
        logits: &GpuBuffer,
        targets: &[u32],
        vocab: usize,
    ) -> Result<(f32, GpuBuffer)> {
        let rows = targets.len();
        let elements = rows
            .checked_mul(vocab)
            .context("V4.1 masked loss shape overflow")?;
        if rows == 0
            || vocab == 0
            || elements > u32::MAX as usize
            || !logits.belongs_to_device(&self.device)
            || logits.f32_capacity() < elements
        {
            bail!("V4.1 masked loss geometry/device mismatch");
        }
        let mut live = Vec::new();
        for (row, &target) in targets.iter().enumerate() {
            if target == u32::MAX {
                continue;
            }
            if target as usize >= vocab {
                bail!("V4.1 loss target exceeds vocabulary");
            }
            live.push(row);
        }
        if live.is_empty() {
            bail!("V4.1 training batch has no non-ignored targets");
        }
        let compact = GpuBuffer::zeros_f32(&self.device, live.len() * vocab)?;
        let compact_targets = GpuBuffer::from_u32(
            &self.device,
            &live.iter().map(|&row| targets[row]).collect::<Vec<_>>(),
        )?;
        let grad = GpuBuffer::zeros_f32(&self.device, live.len() * vocab)?;
        let row_loss = GpuBuffer::zeros_f32(&self.device, live.len())?;
        let mut commands = self.device.compute_batch()?;
        for (row, &source) in live.iter().enumerate() {
            commands.copy_f32_range(logits, source * vocab, &compact, row * vocab, vocab)?;
        }
        self.cross_entropy.record_dispatch(
            &mut commands,
            &[&compact, &compact_targets, &grad, &row_loss],
            bytemuck::bytes_of(&CrossEntropyPush {
                rows: live.len() as u32,
                vocab_size: vocab as u32,
            }),
            [live.len().div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        let row_loss = row_loss.read_f32(live.len())?;
        let mut losses = vec![0.0f32; rows];
        for (i, &row) in live.iter().enumerate() {
            losses[row] = row_loss[i];
        }
        let level_power = ((usize::BITS - (rows - 1).leading_zeros()) / 8).max(4);
        let mask = (1usize << level_power) - 1;
        let mut partial = [0.0f32; 8];
        for (row, &loss) in losses.iter().enumerate() {
            partial[0] += loss;
            for level in 0..7 {
                if row & (mask << (level as u32 * level_power)) != 0 {
                    break;
                }
                partial[level + 1] += partial[level];
                partial[level] = 0.0;
            }
        }
        let loss = partial.into_iter().fold(0.0f32, |a, b| a + b) / live.len() as f32;
        let full = GpuBuffer::zeros_f32(&self.device, elements)?;
        let mut commands = self.device.compute_batch()?;
        for (row, &destination) in live.iter().enumerate() {
            commands.copy_f32_range(&grad, row * vocab, &full, destination * vocab, vocab)?;
        }
        commands.submit()?;
        Ok((loss, full))
    }
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            cross_entropy: vulkan::ComputeKernel::new_with_access(
                &device,
                if crate::transformer::aten_vector_width() >= 16 {
                    CROSS_ENTROPY_LANES16
                } else {
                    CROSS_ENTROPY
                },
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                size_of::<CrossEntropyPush>() as u32,
            )?,
            head_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                LINEAR_INPUT_GRAD,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                size_of::<LinearInputGradPush>() as u32,
            )?,
            norm_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                RMS_NORM_INPUT_GRAD,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                size_of::<NormInputGradPush>() as u32,
            )?,
            device,
        })
    }

    /// Cross entropy over `logits` and `d loss / d hidden` through the frozen
    /// head `[vocab, hidden]`.
    ///
    /// `targets` holds one token id per row. The loss is the mean of the row
    /// losses, and the returned logit gradient already carries the `1 / rows`
    /// factor, so the head adjoint is the complete head backward.
    pub fn loss_and_input_grad(
        &self,
        logits: &GpuBuffer,
        targets: &GpuBuffer,
        head: &GpuBuffer,
        rows: usize,
        vocab_size: usize,
        hidden: usize,
    ) -> Result<DeepseekV41TerminalGradients> {
        if rows == 0 || vocab_size == 0 || hidden == 0 {
            bail!("V4.1 terminal adjoint geometry must be positive");
        }
        if rows > u32::MAX as usize || vocab_size > u32::MAX as usize {
            bail!("V4.1 terminal adjoint exceeds u32 dispatch geometry");
        }
        for (label, buffer) in [("logits", logits), ("targets", targets), ("head", head)] {
            if !buffer.belongs_to_device(&self.device) {
                bail!("V4.1 terminal adjoint {label} belongs to another Vulkan device");
            }
        }
        let logit_elements = rows
            .checked_mul(vocab_size)
            .context("V4.1 terminal adjoint logit overflow")?;
        let head_elements = vocab_size
            .checked_mul(hidden)
            .context("V4.1 terminal adjoint head overflow")?;
        if logits.f32_capacity() < logit_elements {
            bail!("V4.1 terminal adjoint logits buffer is too small");
        }
        if head.f32_capacity() < head_elements {
            bail!("V4.1 terminal adjoint head buffer is too small");
        }
        // Token ids are 32-bit too, so the f32 element capacity is the id count.
        if targets.f32_capacity() < rows {
            bail!("V4.1 terminal adjoint target buffer is too small");
        }

        let grad_logits = GpuBuffer::zeros_f32(&self.device, logit_elements)?;
        let row_loss = GpuBuffer::zeros_f32(&self.device, rows)?;
        let cross_entropy = CrossEntropyPush {
            rows: rows as u32,
            vocab_size: vocab_size as u32,
        };
        let mut commands = self.device.compute_batch()?;
        self.cross_entropy.record_dispatch(
            &mut commands,
            &[logits, targets, &grad_logits, &row_loss],
            bytemuck::bytes_of(&cross_entropy),
            [rows.div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;

        let row_loss = row_loss.read_f32(rows)?;
        // Preserve this API's pinned row_loss.mean() contract. Reduced NLL
        // has a different FP32 fold, covered by the method below.
        let loss = row_loss
            .iter()
            .copied()
            .fold(0.0f32, |sum, value| sum + value)
            / rows as f32;

        let grad_hidden = GpuBuffer::zeros_f32(&self.device, rows * hidden)?;
        let head_grad = LinearInputGradPush {
            rows: rows as u32,
            input_dim: hidden as u32,
            output_dim: vocab_size as u32,
        };
        let mut commands = self.device.compute_batch()?;
        self.head_input_grad.record_dispatch(
            &mut commands,
            &[&grad_logits, head, &grad_hidden],
            bytemuck::bytes_of(&head_grad),
            [hidden.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
        )?;
        commands.submit()?;

        Ok(DeepseekV41TerminalGradients {
            row_loss,
            loss,
            grad_logits,
            grad_hidden,
        })
    }

    /// CPU F.cross_entropy(..., reduction="mean") contract. Its eight-level
    /// NLL cascade differs from unreduced losses followed by mean(). Both
    /// have the same loss-to-logit derivative used by the GPU pass.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_entropy_and_input_grad(
        &self,
        logits: &GpuBuffer,
        targets: &GpuBuffer,
        head: &GpuBuffer,
        rows: usize,
        vocab_size: usize,
        hidden: usize,
    ) -> Result<DeepseekV41TerminalGradients> {
        let mut result =
            self.loss_and_input_grad(logits, targets, head, rows, vocab_size, hidden)?;
        // CPU source: aten/src/ATen/native/LossNLL.cpp, nll_loss_out_frame.
        let level_power = ((usize::BITS - (rows - 1).leading_zeros()) / 8).max(4);
        let mask = (1usize << level_power) - 1;
        let mut partial = [0.0f32; 8];
        for (row, value) in result.row_loss.iter().enumerate() {
            partial[0] += value;
            for level in 0..7 {
                if row & (mask << (level as u32 * level_power)) != 0 {
                    break;
                }
                partial[level + 1] += partial[level];
                partial[level] = 0.0;
            }
        }
        result.loss = partial.into_iter().fold(0.0f32, |sum, value| sum + value) / rows as f32;
        Ok(result)
    }

    /// `d loss / d x` for the final RMSNorm, from `d loss / d hidden` at its
    /// output. `x` is the pre-norm input and `weight` the frozen norm gain.
    ///
    /// The kernel recomputes the forward `rstd` from `x`, so this cannot
    /// disagree with `rms_norm.comp` about the normalization factor.
    pub fn norm_input_grad(
        &self,
        grad_hidden: &GpuBuffer,
        x: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<GpuBuffer> {
        if rows == 0 || hidden == 0 {
            bail!("V4.1 terminal norm geometry must be positive");
        }
        if !eps.is_finite() || eps <= 0.0 {
            bail!("V4.1 terminal norm epsilon is invalid");
        }
        let elements = rows
            .checked_mul(hidden)
            .context("V4.1 terminal norm overflow")?;
        for (label, buffer) in [("grad", grad_hidden), ("input", x)] {
            if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < elements {
                bail!("V4.1 terminal norm {label} geometry or device mismatch");
            }
        }
        if !weight.belongs_to_device(&self.device) || weight.f32_capacity() < hidden {
            bail!("V4.1 terminal norm weight geometry or device mismatch");
        }
        let grad_input = GpuBuffer::zeros_f32(&self.device, elements)?;
        let push = NormInputGradPush {
            rows: rows as u32,
            dim: hidden as u32,
            eps,
        };
        let mut commands = self.device.compute_batch()?;
        self.norm_input_grad.record_dispatch(
            &mut commands,
            &[grad_hidden, x, weight, &grad_input],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(grad_input)
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
}

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod tests;
