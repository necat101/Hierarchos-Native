use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

fn validate_nonnegative_source_scale(label: &str, source_scale: f32) -> Result<()> {
    if !source_scale.is_finite() || source_scale < 0.0 {
        bail!("{label} source scale must be finite and non-negative; got {source_scale}");
    }
    Ok(())
}

const HARD_ACT_SELECT_SPV: &[u8] = include_bytes!("../shaders/hard_act_select.spv");
const HARD_ACT_DEPTH_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/hard_act_depth_backward.spv");
const FINITE_CLAMP_FORWARD_SPV: &[u8] = include_bytes!("../shaders/finite_clamp_forward.spv");
const FINITE_CLAMP_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/finite_clamp_backward.spv");
const INDEXED_STEP_GATHER_SPV: &[u8] = include_bytes!("../shaders/indexed_step_gather.spv");
const INDEXED_STEP_SCATTER_BACKWARD_SPV: &[u8] =
    include_bytes!("../shaders/indexed_step_scatter_backward.spv");
const CONTEXT_LERP_CONCAT_FORWARD_SPV: &[u8] =
    include_bytes!("../shaders/context_lerp_concat_forward.spv");
const CONTEXT_LERP_CONCAT_BACKWARD_SPV: &[u8] =
    include_bytes!("../shaders/context_lerp_concat_backward.spv");
const DRIFT_UPDATE_FORWARD_SPV: &[u8] = include_bytes!("../shaders/drift_update_forward.spv");
const DRIFT_UPDATE_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/drift_update_backward.spv");
const ROW_KEEP_FORWARD_SPV: &[u8] = include_bytes!("../shaders/row_keep_forward.spv");
const ROW_KEEP_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/row_keep_backward.spv");
const WORKER_CONVERGENCE_SPV: &[u8] = include_bytes!("../shaders/worker_convergence.spv");
const COMMITMENT_ACCUMULATE_SPV: &[u8] = include_bytes!("../shaders/commitment_accumulate.spv");
const COMMITMENT_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/commitment_backward.spv");
const COMMITMENT_FINALIZE_SPV: &[u8] = include_bytes!("../shaders/commitment_finalize.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HardActSelectPush {
    steps: u32,
    batch: u32,
    min_steps: u32,
    threshold: f32,
    halt_logit_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HardActDepthPush {
    steps: u32,
    batch: u32,
    min_steps: u32,
    threshold: f32,
    temperature: f32,
    halt_logit_clamp: f32,
    source_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexedStepPush {
    steps: u32,
    batch: u32,
    width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FiniteClampPush {
    len: u32,
    max_abs: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ContextLerpPush {
    rows: u32,
    dim: u32,
    alpha: f32,
    context_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DriftUpdatePush {
    rows: u32,
    dim: u32,
    add_current: u32,
    delta_scale: f32,
    state_clamp: f32,
    norm_clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RowKeepPush {
    rows: u32,
    width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorkerConvergencePush {
    rows: u32,
    dim: u32,
    delta_scale: f32,
    conv_atol: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CommitmentPush {
    rows: u32,
    dim: u32,
    mean_square: u32,
    threshold: f32,
    source_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RowsPush {
    rows: u32,
}

#[derive(Debug)]
pub struct HardActResult {
    pub halt_probabilities: Vec<f32>,
    pub selected_index: Vec<u32>,
    pub executed_steps: Vec<f32>,
    pub selected_output: Vec<f32>,
    pub grad_halt_logits: Vec<f32>,
    pub grad_step_outputs: Vec<f32>,
}

/// Exact device-side counterpart of Python's `_finite_clamp` for graph
/// composition. Finite values are clipped symmetrically while NaN/Inf values
/// are preserved so the model's fail-closed loss/logit guards can reject the
/// trajectory. Backward is the corresponding clamp mask; the non-finite branch
/// is an identity, matching `torch.where(isfinite(x), clamp(x), x)`.
pub(crate) struct FiniteClampVulkanOp {
    forward: vulkan::ComputeKernel,
    backward: vulkan::ComputeKernel,
}

impl FiniteClampVulkanOp {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            forward: vulkan::ComputeKernel::new(
                device,
                FINITE_CLAMP_FORWARD_SPV,
                2,
                std::mem::size_of::<FiniteClampPush>() as u32,
            )?,
            backward: vulkan::ComputeKernel::new(
                device,
                FINITE_CLAMP_BACKWARD_SPV,
                3,
                std::mem::size_of::<FiniteClampPush>() as u32,
            )?,
        })
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        len: usize,
        input: &GpuBuffer,
        output: &GpuBuffer,
        max_abs: f32,
    ) -> Result<()> {
        self.validate(len, max_abs)?;
        let push = FiniteClampPush {
            len: len as u32,
            max_abs,
        };
        self.forward.record_dispatch(
            commands,
            &[input, output],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(len, 256), 1, 1],
        )
    }

    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        len: usize,
        input: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_input: &GpuBuffer,
        max_abs: f32,
    ) -> Result<()> {
        self.validate(len, max_abs)?;
        let push = FiniteClampPush {
            len: len as u32,
            max_abs,
        };
        self.backward.record_dispatch(
            commands,
            &[input, grad_output, grad_input],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(len, 256), 1, 1],
        )
    }

    fn validate(&self, len: usize, max_abs: f32) -> Result<()> {
        if len == 0 {
            bail!("finite clamp length must be positive");
        }
        if !max_abs.is_finite() || max_abs < 0.0 {
            bail!("finite clamp max_abs must be finite and non-negative");
        }
        if len > u32::MAX as usize {
            bail!("finite clamp length exceeds Vulkan u32 indexing");
        }
        Ok(())
    }
}

/// Vulkan implementation of coherent-v9's hard manager ACT primitives.
///
/// Forward matches `hard_act_selection`: halt logits are finite-clamped,
/// sigmoid hazards are accumulated into a CDF, and the first eligible threshold
/// crossing is selected with final-step fallback. Backward contains both the
/// selected-output gather/scatter gradient and the differentiable quantile-depth
/// surrogate used by `hard_act_depth_straight_through`.
pub struct HardActVulkanOp {
    device: VulkanDevice,
    max_steps: usize,
    max_batch: usize,
    width: usize,
    halt_logits: GpuBuffer,
    halt_probabilities: GpuBuffer,
    selected_index: GpuBuffer,
    executed_steps: GpuBuffer,
    step_outputs: GpuBuffer,
    selected_output: GpuBuffer,
    grad_selected_output: GpuBuffer,
    grad_step_outputs: GpuBuffer,
    grad_depth: GpuBuffer,
    grad_halt_logits: GpuBuffer,
    halt_probabilities_readback: GpuBuffer,
    selected_index_readback: GpuBuffer,
    executed_steps_readback: GpuBuffer,
    selected_output_readback: GpuBuffer,
    grad_step_outputs_readback: GpuBuffer,
    grad_halt_logits_readback: GpuBuffer,
    hard_act_select: vulkan::ComputeKernel,
    hard_act_depth_backward: vulkan::ComputeKernel,
    indexed_step_gather: vulkan::ComputeKernel,
    indexed_step_scatter_backward: vulkan::ComputeKernel,
}

impl HardActVulkanOp {
    pub fn new(
        device: VulkanDevice,
        max_steps: usize,
        max_batch: usize,
        width: usize,
    ) -> Result<Self> {
        if max_steps == 0 || max_batch == 0 || width == 0 {
            bail!("hard ACT max_steps, max_batch, and width must be positive");
        }
        let halt_len = max_steps
            .checked_mul(max_batch)
            .context("hard ACT halt capacity overflow")?;
        let selected_len = max_batch
            .checked_mul(width)
            .context("hard ACT selected-output capacity overflow")?;
        let stack_len = halt_len
            .checked_mul(width)
            .context("hard ACT output-stack capacity overflow")?;
        Ok(Self {
            hard_act_select: vulkan::ComputeKernel::new(
                &device,
                HARD_ACT_SELECT_SPV,
                4,
                std::mem::size_of::<HardActSelectPush>() as u32,
            )?,
            hard_act_depth_backward: vulkan::ComputeKernel::new(
                &device,
                HARD_ACT_DEPTH_BACKWARD_SPV,
                4,
                std::mem::size_of::<HardActDepthPush>() as u32,
            )?,
            indexed_step_gather: vulkan::ComputeKernel::new(
                &device,
                INDEXED_STEP_GATHER_SPV,
                3,
                std::mem::size_of::<IndexedStepPush>() as u32,
            )?,
            indexed_step_scatter_backward: vulkan::ComputeKernel::new(
                &device,
                INDEXED_STEP_SCATTER_BACKWARD_SPV,
                3,
                std::mem::size_of::<IndexedStepPush>() as u32,
            )?,
            halt_logits: GpuBuffer::zeros_f32(&device, halt_len)?,
            halt_probabilities: GpuBuffer::zeros_f32(&device, halt_len)?,
            selected_index: GpuBuffer::zeros_f32(&device, max_batch)?,
            executed_steps: GpuBuffer::zeros_f32(&device, max_batch)?,
            step_outputs: GpuBuffer::zeros_f32(&device, stack_len)?,
            selected_output: GpuBuffer::zeros_f32(&device, selected_len)?,
            grad_selected_output: GpuBuffer::zeros_f32(&device, selected_len)?,
            grad_step_outputs: GpuBuffer::zeros_f32(&device, stack_len)?,
            grad_depth: GpuBuffer::zeros_f32(&device, max_batch)?,
            grad_halt_logits: GpuBuffer::zeros_f32(&device, halt_len)?,
            halt_probabilities_readback: GpuBuffer::zeros_host_f32(&device, halt_len)?,
            selected_index_readback: GpuBuffer::zeros_host_f32(&device, max_batch)?,
            executed_steps_readback: GpuBuffer::zeros_host_f32(&device, max_batch)?,
            selected_output_readback: GpuBuffer::zeros_host_f32(&device, selected_len)?,
            grad_step_outputs_readback: GpuBuffer::zeros_host_f32(&device, stack_len)?,
            grad_halt_logits_readback: GpuBuffer::zeros_host_f32(&device, halt_len)?,
            device,
            max_steps,
            max_batch,
            width,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward(
        &mut self,
        halt_logits: &[f32],
        step_outputs: &[f32],
        grad_selected_output: &[f32],
        grad_depth: &[f32],
        steps: usize,
        batch: usize,
        min_steps: usize,
        threshold: f32,
        temperature: f32,
        halt_logit_clamp: f32,
    ) -> Result<HardActResult> {
        self.validate_control(
            steps,
            batch,
            min_steps,
            threshold,
            temperature,
            halt_logit_clamp,
        )?;
        let halt_len = steps * batch;
        let selected_len = batch * self.width;
        let stack_len = halt_len * self.width;
        validate_finite("hard ACT halt_logits", halt_logits, halt_len)?;
        validate_finite("hard ACT step_outputs", step_outputs, stack_len)?;
        validate_finite(
            "hard ACT grad_selected_output",
            grad_selected_output,
            selected_len,
        )?;
        validate_finite("hard ACT grad_depth", grad_depth, batch)?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.halt_logits, halt_logits)?;
        commands.upload_f32(&self.step_outputs, step_outputs)?;
        commands.upload_f32(&self.grad_selected_output, grad_selected_output)?;
        commands.upload_f32(&self.grad_depth, grad_depth)?;

        self.record_select(
            &mut commands,
            steps,
            batch,
            min_steps,
            &self.halt_logits,
            &self.halt_probabilities,
            &self.selected_index,
            &self.executed_steps,
            threshold,
            halt_logit_clamp,
        )?;
        self.record_selected_output_gather(
            &mut commands,
            steps,
            batch,
            &self.step_outputs,
            &self.selected_index,
            &self.selected_output,
        )?;
        self.record_selected_output_scatter_backward(
            &mut commands,
            steps,
            batch,
            &self.grad_selected_output,
            &self.selected_index,
            &self.grad_step_outputs,
        )?;
        self.record_depth_backward(
            &mut commands,
            steps,
            batch,
            min_steps,
            &self.halt_logits,
            &self.halt_probabilities,
            &self.grad_depth,
            &self.grad_halt_logits,
            threshold,
            temperature,
            halt_logit_clamp,
            1.0,
        )?;

        commands.readback_f32(
            &self.halt_probabilities,
            &self.halt_probabilities_readback,
            halt_len,
        )?;
        commands.readback_f32(&self.selected_index, &self.selected_index_readback, batch)?;
        commands.readback_f32(&self.executed_steps, &self.executed_steps_readback, batch)?;
        commands.readback_f32(
            &self.selected_output,
            &self.selected_output_readback,
            selected_len,
        )?;
        commands.readback_f32(
            &self.grad_step_outputs,
            &self.grad_step_outputs_readback,
            stack_len,
        )?;
        commands.readback_f32(
            &self.grad_halt_logits,
            &self.grad_halt_logits_readback,
            halt_len,
        )?;
        commands.submit()?;

        let selected_index = self
            .selected_index_readback
            .read_f32(batch)?
            .into_iter()
            .map(|value| value as u32)
            .collect();
        Ok(HardActResult {
            halt_probabilities: self.halt_probabilities_readback.read_f32(halt_len)?,
            selected_index,
            executed_steps: self.executed_steps_readback.read_f32(batch)?,
            selected_output: self.selected_output_readback.read_f32(selected_len)?,
            grad_halt_logits: self.grad_halt_logits_readback.read_f32(halt_len)?,
            grad_step_outputs: self.grad_step_outputs_readback.read_f32(stack_len)?,
        })
    }

    /// Record coherent-v9 hard-ACT threshold selection directly against
    /// caller-owned Vulkan buffers. The selected index is stored as an exact
    /// integer-valued FP32 lane to preserve compatibility with the existing
    /// gather/scatter kernels while the surrounding graph stays device-local.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_select(
        &self,
        commands: &mut vulkan::ComputeBatch,
        steps: usize,
        batch: usize,
        min_steps: usize,
        halt_logits: &GpuBuffer,
        halt_probabilities: &GpuBuffer,
        selected_index: &GpuBuffer,
        executed_steps: &GpuBuffer,
        threshold: f32,
        halt_logit_clamp: f32,
    ) -> Result<()> {
        self.validate_control(steps, batch, min_steps, threshold, 1.0, halt_logit_clamp)?;
        let push = HardActSelectPush {
            steps: steps as u32,
            batch: batch as u32,
            min_steps: min_steps as u32,
            threshold,
            halt_logit_clamp,
        };
        self.hard_act_select.record_dispatch(
            commands,
            &[
                halt_logits,
                halt_probabilities,
                selected_index,
                executed_steps,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(batch, 64), 1, 1],
        )
    }

    /// Gather the selected candidate output without materializing ACT state on
    /// the host. `step_outputs` is laid out `[steps, batch, width]`.
    pub(crate) fn record_selected_output_gather(
        &self,
        commands: &mut vulkan::ComputeBatch,
        steps: usize,
        batch: usize,
        step_outputs: &GpuBuffer,
        selected_index: &GpuBuffer,
        selected_output: &GpuBuffer,
    ) -> Result<()> {
        self.record_indexed_step_gather(
            commands,
            steps,
            batch,
            self.width,
            step_outputs,
            selected_index,
            selected_output,
        )
    }

    /// Generic form of the hard-selection gather. Manager state commit uses
    /// the same selected index as the H output but a packed-state row width.
    pub(crate) fn record_indexed_step_gather(
        &self,
        commands: &mut vulkan::ComputeBatch,
        steps: usize,
        batch: usize,
        width: usize,
        step_values: &GpuBuffer,
        selected_index: &GpuBuffer,
        selected_values: &GpuBuffer,
    ) -> Result<()> {
        self.validate_shape(steps, batch)?;
        if width == 0 {
            bail!("indexed hard-ACT gather width must be positive");
        }
        let push = IndexedStepPush {
            steps: steps as u32,
            batch: batch as u32,
            width: width as u32,
        };
        self.indexed_step_gather.record_dispatch(
            commands,
            &[step_values, selected_index, selected_values],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(width, 16), div_ceil_u32(batch, 16), 1],
        )
    }

    /// Scatter the selected-output adjoint back into the candidate stack. This
    /// is the hard gather's exact reverse-mode edge; non-selected candidates
    /// receive zero from this path.
    pub(crate) fn record_selected_output_scatter_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        steps: usize,
        batch: usize,
        grad_selected_output: &GpuBuffer,
        selected_index: &GpuBuffer,
        grad_step_outputs: &GpuBuffer,
    ) -> Result<()> {
        self.record_indexed_step_scatter_backward(
            commands,
            steps,
            batch,
            self.width,
            grad_selected_output,
            selected_index,
            grad_step_outputs,
        )
    }

    /// Generic reverse edge for an indexed hard-selection gather. Manager
    /// packed-state commitment uses the same selected indices as the output
    /// path but a different per-row width.
    pub(crate) fn record_indexed_step_scatter_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        steps: usize,
        batch: usize,
        width: usize,
        grad_selected_values: &GpuBuffer,
        selected_index: &GpuBuffer,
        grad_step_values: &GpuBuffer,
    ) -> Result<()> {
        self.validate_shape(steps, batch)?;
        if width == 0 {
            bail!("indexed hard-ACT scatter width must be positive");
        }
        let push = IndexedStepPush {
            steps: steps as u32,
            batch: batch as u32,
            width: width as u32,
        };
        let stack_len = steps * batch * width;
        self.indexed_step_scatter_backward.record_dispatch(
            commands,
            &[grad_selected_values, selected_index, grad_step_values],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(stack_len, 256), 1, 1],
        )
    }

    /// Reverse the straight-through differentiable ACT depth surrogate into
    /// caller-owned halt-logit gradients while preserving hard forward choice.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_depth_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        steps: usize,
        batch: usize,
        min_steps: usize,
        halt_logits: &GpuBuffer,
        halt_probabilities: &GpuBuffer,
        grad_depth: &GpuBuffer,
        grad_halt_logits: &GpuBuffer,
        threshold: f32,
        temperature: f32,
        halt_logit_clamp: f32,
        source_scale: f32,
    ) -> Result<()> {
        self.validate_control(
            steps,
            batch,
            min_steps,
            threshold,
            temperature,
            halt_logit_clamp,
        )?;
        validate_nonnegative_source_scale("hard-ACT depth", source_scale)?;
        let push = HardActDepthPush {
            steps: steps as u32,
            batch: batch as u32,
            min_steps: min_steps as u32,
            threshold,
            temperature,
            halt_logit_clamp,
            source_scale,
        };
        self.hard_act_depth_backward.record_dispatch(
            commands,
            &[
                halt_logits,
                halt_probabilities,
                grad_depth,
                grad_halt_logits,
            ],
            bytemuck::bytes_of(&push),
            // The shader owns one complete ACT row per invocation and emits all
            // step derivatives with a reverse suffix scan.
            [div_ceil_u32(batch, 64), 1, 1],
        )
    }

    fn validate_shape(&self, steps: usize, batch: usize) -> Result<()> {
        if steps == 0 || steps > self.max_steps || batch == 0 || batch > self.max_batch {
            bail!(
                "hard ACT shape must fit steps=1..={} batch=1..={}; got steps={steps} batch={batch}",
                self.max_steps,
                self.max_batch
            );
        }
        Ok(())
    }

    fn validate_control(
        &self,
        steps: usize,
        batch: usize,
        min_steps: usize,
        threshold: f32,
        temperature: f32,
        halt_logit_clamp: f32,
    ) -> Result<()> {
        self.validate_shape(steps, batch)?;
        if min_steps == 0 || min_steps > steps {
            bail!("hard ACT min_steps must be in 1..={steps}; got {min_steps}");
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            bail!("hard ACT threshold must be finite and in 0..=1");
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            bail!("hard ACT temperature must be finite and positive");
        }
        if !halt_logit_clamp.is_finite() || halt_logit_clamp <= 0.0 {
            bail!("hard ACT halt_logit_clamp must be finite and positive");
        }
        Ok(())
    }
}

/// Generic differentiable row-freeze edge used by static Vulkan control loops.
/// Rows with an active lane > 0 keep the candidate; inactive rows retain their
/// previous value. Backward splits the adjoint along the same hard mask.
pub(crate) struct RowKeepVulkanOp {
    max_rows: usize,
    row_keep_forward: vulkan::ComputeKernel,
    row_keep_backward: vulkan::ComputeKernel,
}

impl RowKeepVulkanOp {
    pub(crate) fn new(device: &VulkanDevice, max_rows: usize) -> Result<Self> {
        if max_rows == 0 {
            bail!("row-keep max_rows must be positive");
        }
        Ok(Self {
            max_rows,
            row_keep_forward: vulkan::ComputeKernel::new(
                device,
                ROW_KEEP_FORWARD_SPV,
                4,
                std::mem::size_of::<RowKeepPush>() as u32,
            )?,
            row_keep_backward: vulkan::ComputeKernel::new(
                device,
                ROW_KEEP_BACKWARD_SPV,
                4,
                std::mem::size_of::<RowKeepPush>() as u32,
            )?,
        })
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        width: usize,
        candidate: &GpuBuffer,
        previous: &GpuBuffer,
        active: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<()> {
        self.validate(rows, width)?;
        let push = RowKeepPush {
            rows: rows as u32,
            width: width as u32,
        };
        self.row_keep_forward.record_dispatch(
            commands,
            &[candidate, previous, active, output],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(rows * width, 256), 1, 1],
        )
    }

    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        width: usize,
        grad_output: &GpuBuffer,
        active: &GpuBuffer,
        grad_candidate: &GpuBuffer,
        grad_previous: &GpuBuffer,
    ) -> Result<()> {
        self.validate(rows, width)?;
        let push = RowKeepPush {
            rows: rows as u32,
            width: width as u32,
        };
        self.row_keep_backward.record_dispatch(
            commands,
            &[grad_output, active, grad_candidate, grad_previous],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(rows * width, 256), 1, 1],
        )
    }

    fn validate(&self, rows: usize, width: usize) -> Result<()> {
        if rows == 0 || rows > self.max_rows || width == 0 {
            bail!(
                "row-keep shape requires rows=1..={} and positive width; got rows={rows} width={width}",
                self.max_rows
            );
        }
        Ok(())
    }
}

/// Device-resident worker convergence and commitment-control primitives.
pub(crate) struct WorkerRefinementControlVulkanOp {
    dim: usize,
    max_rows: usize,
    convergence: vulkan::ComputeKernel,
    commitment_accumulate: vulkan::ComputeKernel,
    commitment_backward: vulkan::ComputeKernel,
    commitment_finalize: vulkan::ComputeKernel,
}

impl WorkerRefinementControlVulkanOp {
    pub(crate) fn new(device: &VulkanDevice, dim: usize, max_rows: usize) -> Result<Self> {
        if dim == 0 || max_rows == 0 {
            bail!("worker-control dim and max_rows must be positive");
        }
        Ok(Self {
            dim,
            max_rows,
            convergence: vulkan::ComputeKernel::new(
                device,
                WORKER_CONVERGENCE_SPV,
                3,
                std::mem::size_of::<WorkerConvergencePush>() as u32,
            )?,
            commitment_accumulate: vulkan::ComputeKernel::new(
                device,
                COMMITMENT_ACCUMULATE_SPV,
                4,
                std::mem::size_of::<CommitmentPush>() as u32,
            )?,
            commitment_backward: vulkan::ComputeKernel::new(
                device,
                COMMITMENT_BACKWARD_SPV,
                5,
                std::mem::size_of::<CommitmentPush>() as u32,
            )?,
            commitment_finalize: vulkan::ComputeKernel::new(
                device,
                COMMITMENT_FINALIZE_SPV,
                4,
                std::mem::size_of::<RowsPush>() as u32,
            )?,
        })
    }

    pub(crate) fn record_convergence(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        projected: &GpuBuffer,
        active: &GpuBuffer,
        next_active: &GpuBuffer,
        delta_scale: f32,
        conv_atol: f32,
    ) -> Result<()> {
        self.validate(rows)?;
        if !delta_scale.is_finite() || delta_scale < 0.0 {
            bail!("worker convergence delta_scale must be finite and non-negative");
        }
        if !conv_atol.is_finite() || conv_atol <= 0.0 {
            bail!("worker convergence atol must be finite and positive");
        }
        let push = WorkerConvergencePush {
            rows: rows as u32,
            dim: self.dim as u32,
            delta_scale,
            conv_atol,
        };
        self.convergence.record_dispatch(
            commands,
            &[projected, active, next_active],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_commitment_accumulate(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        candidate_drift: &GpuBuffer,
        active: &GpuBuffer,
        cost_sum: &GpuBuffer,
        cost_count: &GpuBuffer,
        mean_square: bool,
        threshold: f32,
    ) -> Result<()> {
        self.validate_commitment(rows, threshold)?;
        let push = self.commitment_push(rows, mean_square, threshold, 1.0);
        self.commitment_accumulate.record_dispatch(
            commands,
            &[candidate_drift, active, cost_sum, cost_count],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_commitment_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        candidate_drift: &GpuBuffer,
        active: &GpuBuffer,
        cost_count: &GpuBuffer,
        grad_cost: &GpuBuffer,
        grad_drift: &GpuBuffer,
        mean_square: bool,
        threshold: f32,
        source_scale: f32,
    ) -> Result<()> {
        self.validate_commitment(rows, threshold)?;
        validate_nonnegative_source_scale("worker commitment", source_scale)?;
        let push = self.commitment_push(rows, mean_square, threshold, source_scale);
        self.commitment_backward.record_dispatch(
            commands,
            &[candidate_drift, active, cost_count, grad_cost, grad_drift],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(rows * self.dim, 256), 1, 1],
        )
    }

    pub(crate) fn record_commitment_finalize(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        cost_sum: &GpuBuffer,
        cost_count: &GpuBuffer,
        cost: &GpuBuffer,
        effective_steps: &GpuBuffer,
    ) -> Result<()> {
        self.validate(rows)?;
        let push = RowsPush { rows: rows as u32 };
        self.commitment_finalize.record_dispatch(
            commands,
            &[cost_sum, cost_count, cost, effective_steps],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(rows, 64), 1, 1],
        )
    }

    fn commitment_push(
        &self,
        rows: usize,
        mean_square: bool,
        threshold: f32,
        source_scale: f32,
    ) -> CommitmentPush {
        CommitmentPush {
            rows: rows as u32,
            dim: self.dim as u32,
            mean_square: u32::from(mean_square),
            threshold,
            source_scale,
        }
    }

    fn validate_commitment(&self, rows: usize, threshold: f32) -> Result<()> {
        self.validate(rows)?;
        if !threshold.is_finite() || threshold < 0.0 {
            bail!("worker commitment threshold must be finite and non-negative");
        }
        Ok(())
    }

    fn validate(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.max_rows {
            bail!(
                "worker-control rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ContextLerpConcatResult {
    pub output: Vec<f32>,
    pub grad_enc: Vec<f32>,
    pub grad_previous: Vec<f32>,
    pub grad_target: Vec<f32>,
    pub grad_drift: Vec<f32>,
}

#[derive(Debug)]
pub struct DriftUpdateResult {
    pub output: Vec<f32>,
    pub grad_current: Vec<f32>,
    pub grad_projected: Vec<f32>,
}

/// Vulkan context-control primitives around the worker input. These kernels are
/// deliberately separate from the learned projections so repeated drift steps
/// can later be recomputed during reverse-mode TBPTT without retaining a large
/// activation tape.
pub struct ContextDriftVulkanOp {
    device: VulkanDevice,
    dim: usize,
    max_rows: usize,
    enc: GpuBuffer,
    previous: GpuBuffer,
    target: GpuBuffer,
    drift: GpuBuffer,
    concat_output: GpuBuffer,
    grad_concat: GpuBuffer,
    grad_enc: GpuBuffer,
    grad_previous: GpuBuffer,
    grad_target: GpuBuffer,
    grad_drift: GpuBuffer,
    projected: GpuBuffer,
    drift_output: GpuBuffer,
    grad_drift_output: GpuBuffer,
    grad_current: GpuBuffer,
    grad_projected: GpuBuffer,
    concat_output_readback: GpuBuffer,
    grad_enc_readback: GpuBuffer,
    grad_previous_readback: GpuBuffer,
    grad_target_readback: GpuBuffer,
    grad_drift_readback: GpuBuffer,
    drift_output_readback: GpuBuffer,
    grad_current_readback: GpuBuffer,
    grad_projected_readback: GpuBuffer,
    context_lerp_concat_forward: vulkan::ComputeKernel,
    context_lerp_concat_backward: vulkan::ComputeKernel,
    drift_update_forward: vulkan::ComputeKernel,
    drift_update_backward: vulkan::ComputeKernel,
}

impl ContextDriftVulkanOp {
    pub fn new(device: VulkanDevice, dim: usize, max_rows: usize) -> Result<Self> {
        if dim == 0 || max_rows == 0 {
            bail!("context/drift dim and max_rows must be positive");
        }
        let vector_len = dim
            .checked_mul(max_rows)
            .context("context/drift vector capacity overflow")?;
        let concat_len = vector_len
            .checked_mul(2)
            .context("context/drift concat capacity overflow")?;
        Ok(Self {
            context_lerp_concat_forward: vulkan::ComputeKernel::new(
                &device,
                CONTEXT_LERP_CONCAT_FORWARD_SPV,
                5,
                std::mem::size_of::<ContextLerpPush>() as u32,
            )?,
            context_lerp_concat_backward: vulkan::ComputeKernel::new(
                &device,
                CONTEXT_LERP_CONCAT_BACKWARD_SPV,
                7,
                std::mem::size_of::<ContextLerpPush>() as u32,
            )?,
            drift_update_forward: vulkan::ComputeKernel::new(
                &device,
                DRIFT_UPDATE_FORWARD_SPV,
                3,
                std::mem::size_of::<DriftUpdatePush>() as u32,
            )?,
            drift_update_backward: vulkan::ComputeKernel::new(
                &device,
                DRIFT_UPDATE_BACKWARD_SPV,
                5,
                std::mem::size_of::<DriftUpdatePush>() as u32,
            )?,
            enc: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            target: GpuBuffer::zeros_f32(&device, vector_len)?,
            drift: GpuBuffer::zeros_f32(&device, vector_len)?,
            concat_output: GpuBuffer::zeros_f32(&device, concat_len)?,
            grad_concat: GpuBuffer::zeros_f32(&device, concat_len)?,
            grad_enc: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_target: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_drift: GpuBuffer::zeros_f32(&device, vector_len)?,
            projected: GpuBuffer::zeros_f32(&device, vector_len)?,
            drift_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_drift_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_current: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_projected: GpuBuffer::zeros_f32(&device, vector_len)?,
            concat_output_readback: GpuBuffer::zeros_host_f32(&device, concat_len)?,
            grad_enc_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_previous_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_target_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_drift_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            drift_output_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_current_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_projected_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            device,
            dim,
            max_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lerp_concat_forward_backward(
        &mut self,
        enc: &[f32],
        previous: &[f32],
        target: &[f32],
        drift: &[f32],
        grad_output: &[f32],
        rows: usize,
        alpha: f32,
        context_clamp: f32,
    ) -> Result<ContextLerpConcatResult> {
        self.validate_rows(rows)?;
        if !alpha.is_finite() {
            bail!("context interpolation alpha must be finite");
        }
        if !context_clamp.is_finite() || context_clamp <= 0.0 {
            bail!("context interpolation clamp must be finite and positive");
        }
        let vector_len = rows * self.dim;
        let concat_len = vector_len * 2;
        validate_finite("context enc", enc, vector_len)?;
        validate_finite("context previous", previous, vector_len)?;
        validate_finite("context target", target, vector_len)?;
        validate_finite("context drift", drift, vector_len)?;
        validate_finite("context grad_output", grad_output, concat_len)?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.enc, enc)?;
        commands.upload_f32(&self.previous, previous)?;
        commands.upload_f32(&self.target, target)?;
        commands.upload_f32(&self.drift, drift)?;
        commands.upload_f32(&self.grad_concat, grad_output)?;
        self.record_lerp_concat_forward(
            &mut commands,
            rows,
            &self.enc,
            &self.previous,
            &self.target,
            &self.drift,
            &self.concat_output,
            alpha,
            context_clamp,
        )?;
        self.record_lerp_concat_backward(
            &mut commands,
            rows,
            &self.previous,
            &self.target,
            &self.grad_concat,
            &self.grad_enc,
            &self.grad_previous,
            &self.grad_target,
            &self.grad_drift,
            alpha,
            context_clamp,
        )?;
        commands.readback_f32(
            &self.concat_output,
            &self.concat_output_readback,
            concat_len,
        )?;
        commands.readback_f32(&self.grad_enc, &self.grad_enc_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_previous,
            &self.grad_previous_readback,
            vector_len,
        )?;
        commands.readback_f32(&self.grad_target, &self.grad_target_readback, vector_len)?;
        commands.readback_f32(&self.grad_drift, &self.grad_drift_readback, vector_len)?;
        commands.submit()?;
        Ok(ContextLerpConcatResult {
            output: self.concat_output_readback.read_f32(concat_len)?,
            grad_enc: self.grad_enc_readback.read_f32(vector_len)?,
            grad_previous: self.grad_previous_readback.read_f32(vector_len)?,
            grad_target: self.grad_target_readback.read_f32(vector_len)?,
            grad_drift: self.grad_drift_readback.read_f32(vector_len)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn drift_update_forward_backward(
        &mut self,
        current: &[f32],
        projected: &[f32],
        grad_output: &[f32],
        rows: usize,
        add_current: bool,
        delta_scale: f32,
        state_clamp: f32,
        norm_clamp: f32,
    ) -> Result<DriftUpdateResult> {
        self.validate_rows(rows)?;
        if !delta_scale.is_finite() || delta_scale < 0.0 {
            bail!("drift delta scale must be finite and non-negative");
        }
        if !state_clamp.is_finite() || state_clamp <= 0.0 {
            bail!("drift state clamp must be finite and positive");
        }
        if !norm_clamp.is_finite() || norm_clamp < 0.0 {
            bail!("drift norm clamp must be finite and non-negative");
        }
        let vector_len = rows * self.dim;
        validate_finite("current drift", current, vector_len)?;
        validate_finite("projected drift", projected, vector_len)?;
        validate_finite("drift grad_output", grad_output, vector_len)?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.drift, current)?;
        commands.upload_f32(&self.projected, projected)?;
        commands.upload_f32(&self.grad_drift_output, grad_output)?;
        self.record_drift_update_forward(
            &mut commands,
            rows,
            &self.drift,
            &self.projected,
            &self.drift_output,
            add_current,
            delta_scale,
            state_clamp,
            norm_clamp,
        )?;
        self.record_drift_update_backward(
            &mut commands,
            rows,
            &self.drift,
            &self.projected,
            &self.grad_drift_output,
            &self.grad_current,
            &self.grad_projected,
            add_current,
            delta_scale,
            state_clamp,
            norm_clamp,
        )?;
        commands.readback_f32(&self.drift_output, &self.drift_output_readback, vector_len)?;
        commands.readback_f32(&self.grad_current, &self.grad_current_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_projected,
            &self.grad_projected_readback,
            vector_len,
        )?;
        commands.submit()?;
        Ok(DriftUpdateResult {
            output: self.drift_output_readback.read_f32(vector_len)?,
            grad_current: self.grad_current_readback.read_f32(vector_len)?,
            grad_projected: self.grad_projected_readback.read_f32(vector_len)?,
        })
    }

    /// Record the context interpolation + `[enc, context + drift]` concat using
    /// caller-owned Vulkan buffers. No upload, submission, or readback occurs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_lerp_concat_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        enc: &GpuBuffer,
        previous: &GpuBuffer,
        target: &GpuBuffer,
        drift: &GpuBuffer,
        output: &GpuBuffer,
        alpha: f32,
        context_clamp: f32,
    ) -> Result<()> {
        self.validate_lerp(rows, alpha, context_clamp)?;
        let push = ContextLerpPush {
            rows: rows as u32,
            dim: self.dim as u32,
            alpha,
            context_clamp,
        };
        self.context_lerp_concat_forward.record_dispatch(
            commands,
            &[enc, previous, target, drift, output],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(self.dim, 16), div_ceil_u32(rows, 16), 1],
        )
    }

    /// Reverse the context interpolation/concat into caller-owned gradient
    /// buffers. This is the device-resident seam needed to propagate worker
    /// refinement gradients back to enc, temporal context, and drift.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_lerp_concat_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        previous: &GpuBuffer,
        target: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_enc: &GpuBuffer,
        grad_previous: &GpuBuffer,
        grad_target: &GpuBuffer,
        grad_drift: &GpuBuffer,
        alpha: f32,
        context_clamp: f32,
    ) -> Result<()> {
        self.validate_lerp(rows, alpha, context_clamp)?;
        let push = ContextLerpPush {
            rows: rows as u32,
            dim: self.dim as u32,
            alpha,
            context_clamp,
        };
        self.context_lerp_concat_backward.record_dispatch(
            commands,
            &[
                grad_output,
                previous,
                target,
                grad_enc,
                grad_previous,
                grad_target,
                grad_drift,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(self.dim, 16), div_ceil_u32(rows, 16), 1],
        )
    }

    /// Record one coherent-v9 drift transition on caller-owned Vulkan buffers.
    /// `projected` is the raw output of `context_drift_proj`; tanh, scaling,
    /// finite clamp, and optional L2 norm clamp remain inside the Vulkan kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_drift_update_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        current: &GpuBuffer,
        projected: &GpuBuffer,
        output: &GpuBuffer,
        add_current: bool,
        delta_scale: f32,
        state_clamp: f32,
        norm_clamp: f32,
    ) -> Result<()> {
        self.validate_drift(rows, delta_scale, state_clamp, norm_clamp)?;
        let push = DriftUpdatePush {
            rows: rows as u32,
            dim: self.dim as u32,
            add_current: u32::from(add_current),
            delta_scale,
            state_clamp,
            norm_clamp,
        };
        self.drift_update_forward.record_dispatch(
            commands,
            &[current, projected, output],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )
    }

    /// Reverse one caller-owned drift transition without submitting or reading
    /// back. The projected gradient can feed `context_drift_proj` immediately.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_drift_update_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        current: &GpuBuffer,
        projected: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_current: &GpuBuffer,
        grad_projected: &GpuBuffer,
        add_current: bool,
        delta_scale: f32,
        state_clamp: f32,
        norm_clamp: f32,
    ) -> Result<()> {
        self.validate_drift(rows, delta_scale, state_clamp, norm_clamp)?;
        let push = DriftUpdatePush {
            rows: rows as u32,
            dim: self.dim as u32,
            add_current: u32::from(add_current),
            delta_scale,
            state_clamp,
            norm_clamp,
        };
        self.drift_update_backward.record_dispatch(
            commands,
            &[
                current,
                projected,
                grad_output,
                grad_current,
                grad_projected,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )
    }

    fn validate_lerp(&self, rows: usize, alpha: f32, context_clamp: f32) -> Result<()> {
        self.validate_rows(rows)?;
        if !alpha.is_finite() {
            bail!("context interpolation alpha must be finite");
        }
        if !context_clamp.is_finite() || context_clamp <= 0.0 {
            bail!("context interpolation clamp must be finite and positive");
        }
        Ok(())
    }

    fn validate_drift(
        &self,
        rows: usize,
        delta_scale: f32,
        state_clamp: f32,
        norm_clamp: f32,
    ) -> Result<()> {
        self.validate_rows(rows)?;
        if !delta_scale.is_finite() || delta_scale < 0.0 {
            bail!("drift delta scale must be finite and non-negative");
        }
        if !state_clamp.is_finite() || state_clamp <= 0.0 {
            bail!("drift state clamp must be finite and positive");
        }
        if !norm_clamp.is_finite() || norm_clamp < 0.0 {
            bail!("drift norm clamp must be finite and non-negative");
        }
        Ok(())
    }

    fn validate_rows(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.max_rows {
            bail!(
                "context/drift rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        Ok(())
    }
}

fn validate_finite(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!("{name} has {} values; expected {expected}", values.len());
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{name} contains non-finite values");
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUALIFICATION_DEVICE_INDEX_ENV: &str = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX";

    fn test_device() -> Result<Option<VulkanDevice>> {
        match std::env::var(QUALIFICATION_DEVICE_INDEX_ENV) {
            Ok(raw) => {
                let index = raw.parse::<usize>().with_context(|| {
                    format!("{QUALIFICATION_DEVICE_INDEX_ENV} must be a non-negative device index")
                })?;
                Ok(Some(VulkanDevice::new_with_index(index)?))
            }
            Err(std::env::VarError::NotPresent) => Ok(VulkanDevice::new().ok()),
            Err(err) => {
                Err(err).with_context(|| format!("reading {QUALIFICATION_DEVICE_INDEX_ENV}"))
            }
        }
    }

    #[test]
    fn auxiliary_source_scale_accepts_zero_gradient_mask() {
        validate_nonnegative_source_scale("auxiliary", 0.0)
            .expect("a zero source scale is a valid derivative mask");
        validate_nonnegative_source_scale("auxiliary", 1.0)
            .expect("positive source scales remain valid");

        for invalid in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_nonnegative_source_scale("auxiliary", invalid).is_err());
        }
    }

    #[test]
    fn hard_act_zero_source_mask_produces_zero_gradients() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let op = HardActVulkanOp::new(device.clone(), 2, 1, 1)?;
        let halt_logits = GpuBuffer::from_f32(&device, &[0.0, 0.0])?;
        let halt_probabilities = GpuBuffer::from_f32(&device, &[0.5, 0.5])?;
        let grad_depth = GpuBuffer::from_f32(&device, &[1.0])?;
        let grad_halt_logits = GpuBuffer::zeros_f32(&device, 2)?;
        let readback = GpuBuffer::zeros_host_f32(&device, 2)?;

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        op.record_depth_backward(
            &mut commands,
            2,
            1,
            1,
            &halt_logits,
            &halt_probabilities,
            &grad_depth,
            &grad_halt_logits,
            0.5,
            1.0,
            30.0,
            0.0,
        )?;
        commands.readback_f32(&grad_halt_logits, &readback, 2)?;
        commands.submit()?;

        assert_eq!(readback.read_f32(2)?, vec![0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn hard_act_rowwise_depth_backward_matches_prefix_reference_at_deeper_horizon() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let steps = 11usize;
        let batch = 3usize;
        let min_steps = 3usize;
        let threshold = 0.71f32;
        let temperature = 0.13f32;
        let halt_logit_clamp = 3.0f32;
        let source_scale = 0.625f32;
        let logits = (0..steps * batch)
            .map(|index| {
                let centered = ((index * 29 + 17) % 67) as f32 - 33.0;
                centered * 0.115
            })
            .collect::<Vec<_>>();
        let probabilities = logits
            .iter()
            .map(|raw| {
                let bounded = raw.clamp(-halt_logit_clamp, halt_logit_clamp);
                (1.0 / (1.0 + (-bounded).exp())).clamp(1.0e-6, 1.0 - 1.0e-6)
            })
            .collect::<Vec<_>>();
        let grad_depth = [0.7f32, -0.4, 1.1];
        let expected = hard_act_depth_prefix_reference(
            &logits,
            &probabilities,
            &grad_depth,
            steps,
            batch,
            min_steps,
            threshold,
            temperature,
            halt_logit_clamp,
            source_scale,
        );

        let op = HardActVulkanOp::new(device.clone(), steps, batch, 1)?;
        let logits_gpu = GpuBuffer::from_f32(&device, &logits)?;
        let probabilities_gpu = GpuBuffer::from_f32(&device, &probabilities)?;
        let grad_depth_gpu = GpuBuffer::from_f32(&device, &grad_depth)?;
        let grad_logits_gpu = GpuBuffer::zeros_f32(&device, steps * batch)?;
        let readback = GpuBuffer::zeros_host_f32(&device, steps * batch)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        op.record_depth_backward(
            &mut commands,
            steps,
            batch,
            min_steps,
            &logits_gpu,
            &probabilities_gpu,
            &grad_depth_gpu,
            &grad_logits_gpu,
            threshold,
            temperature,
            halt_logit_clamp,
            source_scale,
        )?;
        commands.readback_f32(&grad_logits_gpu, &readback, steps * batch)?;
        commands.submit()?;
        let actual = readback.read_f32(steps * batch)?;

        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 2.0e-5,
            "row-wise hard-ACT depth backward diverged from prefix reference: max_abs={max_abs}"
        );
        for row in 0..batch {
            assert_eq!(actual[(steps - 1) * batch + row], 0.0);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn hard_act_depth_prefix_reference(
        logits: &[f32],
        probabilities: &[f32],
        grad_depth: &[f32],
        steps: usize,
        batch: usize,
        min_steps: usize,
        threshold: f32,
        temperature: f32,
        halt_logit_clamp: f32,
        source_scale: f32,
    ) -> Vec<f32> {
        let mut grad_logits = vec![0.0f32; steps * batch];
        for step_k in 0..steps {
            for row in 0..batch {
                let mut grad_p = 0.0f32;
                if step_k + 1 < steps {
                    let first_cdf = step_k.max(min_steps - 1);
                    for cdf_step in first_cdf..steps - 1 {
                        let mut survival = 1.0f32;
                        for j in 0..=cdf_step {
                            survival *= 1.0 - probabilities[j * batch + row];
                        }
                        let cdf = 1.0 - survival;
                        let soft_continue =
                            1.0 / (1.0 + (-((threshold - cdf) / temperature)).exp());
                        let dterm_dsurvival = soft_continue * (1.0 - soft_continue) / temperature;
                        let denom = (1.0 - probabilities[step_k * batch + row]).max(1.0e-8);
                        grad_p += dterm_dsurvival * (-survival / denom);
                    }
                }
                grad_p *= grad_depth[row] * source_scale;

                let index = step_k * batch + row;
                let raw = logits[index];
                let bounded = raw.clamp(-halt_logit_clamp, halt_logit_clamp);
                let sigmoid_value = 1.0 / (1.0 + (-bounded).exp());
                let finite_mask = if raw.is_finite() { 1.0 } else { 0.0 };
                let logit_clamp_mask = if raw >= -halt_logit_clamp && raw <= halt_logit_clamp {
                    1.0
                } else {
                    0.0
                };
                let probability_clamp_mask = if (1.0e-6..=1.0 - 1.0e-6).contains(&sigmoid_value) {
                    1.0
                } else {
                    0.0
                };
                grad_logits[index] = grad_p
                    * sigmoid_value
                    * (1.0 - sigmoid_value)
                    * finite_mask
                    * logit_clamp_mask
                    * probability_clamp_mask;
            }
        }
        grad_logits
    }

    #[test]
    fn finite_clamp_matches_pytorch_boundaries_and_preserves_nonfinite_branch() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let op = FiniteClampVulkanOp::new(&device)?;
        let values = [
            -3.0f32,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let value_bits = values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let grad_output_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input = GpuBuffer::from_u32(&device, &value_bits)?;
        let grad_output = GpuBuffer::from_f32(&device, &grad_output_values)?;
        let output = GpuBuffer::zeros_f32(&device, values.len())?;
        let grad_input = GpuBuffer::zeros_f32(&device, values.len())?;
        let output_readback = GpuBuffer::zeros_host_f32(&device, values.len())?;
        let grad_input_readback = GpuBuffer::zeros_host_f32(&device, values.len())?;

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        op.record_forward(&mut commands, values.len(), &input, &output, 2.0)?;
        op.record_backward(
            &mut commands,
            values.len(),
            &input,
            &grad_output,
            &grad_input,
            2.0,
        )?;
        commands.readback_f32(&output, &output_readback, values.len())?;
        commands.readback_f32(&grad_input, &grad_input_readback, values.len())?;
        commands.submit()?;

        let actual_output = output_readback.read_f32(values.len())?;
        assert_eq!(&actual_output[..7], &[-2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0]);
        assert!(actual_output[7].is_nan());
        assert_eq!(actual_output[8], f32::INFINITY);
        assert_eq!(actual_output[9], f32::NEG_INFINITY);

        // torch.clamp's derivative is inclusive at both clamp boundaries. The
        // torch.where non-finite branch is an identity, so NaN/Inf inputs pass
        // their upstream gradient through even though the batch is rejected by
        // the outer numerical-health guard.
        assert_eq!(
            grad_input_readback.read_f32(values.len())?,
            vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0, 10.0]
        );
        Ok(())
    }
}
