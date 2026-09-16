use std::{cell::RefCell, path::Path};

use anyhow::{bail, Context, Result};

use crate::rwkv_low_rank::{RwkvLowRankFp16ParameterMirrors, RwkvLowRankParameterGradArithmetic};
use crate::rwkv_optimizer::{
    RwkvParameterStorageMirrorBinding, RwkvPersistentAdamW, RwkvTrainableRef,
};
use crate::{
    vulkan, AdamWHyperParams, GpuBuffer, RwkvNumericsPolicy, RwkvOptimizerStepResult,
    RwkvPackedCellOp, RwkvParameterSnapshot, RwkvStateReadoutMode, SharedLmHeadParameter,
    TiedTokenEmbeddingOp, VulkanDevice, VulkanParameterStorageFormat,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RwkvTbpttSchedule {
    pub detach_every_n_steps: Option<usize>,
}

impl RwkvTbpttSchedule {
    pub fn new(detach_every_n_steps: Option<usize>) -> Result<Self> {
        if detach_every_n_steps == Some(0) {
            bail!("TBPTT detach interval must be positive when enabled");
        }
        Ok(Self {
            detach_every_n_steps,
        })
    }

    pub const fn full_bptt() -> Self {
        Self {
            detach_every_n_steps: None,
        }
    }

    pub fn is_detach_boundary(self, timestep: usize) -> bool {
        timestep > 0
            && self
                .detach_every_n_steps
                .is_some_and(|interval| timestep.is_multiple_of(interval))
    }
}

#[derive(Debug)]
pub struct RwkvTbpttSequenceResult {
    pub steps: usize,
    pub batch: usize,
    pub outputs: Vec<f32>,
    pub final_packed_state: Vec<f32>,
    pub grad_x: Vec<f32>,
    pub token_feature_grad: Vec<f32>,
    pub grad_initial_packed_state: Vec<f32>,
}

#[derive(Debug)]
pub struct RwkvTbpttTrainStepResult {
    pub sequence: RwkvTbpttSequenceResult,
    pub optimizer: RwkvOptimizerStepResult,
    pub parameters: Vec<RwkvParameterSnapshot>,
    /// Present when the sequence was driven from token IDs through the tied
    /// `lm_head.weight` embedding path.
    pub tied_embedding_optimizer: Option<RwkvOptimizerStepResult>,
}

/// One host-described recurrent branch for deferred multi-branch training.
/// Branches share the same physical RWKV/DeepEmbed/tied-embedding parameters,
/// but each branch owns its own recurrent input/state/gradient boundary.
pub struct RwkvTbpttBranchInput<'a> {
    pub batch: usize,
    pub steps: usize,
    pub x_sequence: &'a [f32],
    pub token_id_sequence: &'a [u32],
    pub initial_packed_state: &'a [f32],
    pub grad_output_sequence: &'a [f32],
    pub final_packed_state_grad: Option<&'a [f32]>,
    pub schedule: RwkvTbpttSchedule,
}

/// Host-finalization ticket for a TBPTT sequence recorded into a
/// caller-owned Vulkan command buffer. The batch must be submitted before the
/// ticket is finalized.
#[derive(Debug)]
pub(crate) struct RwkvTbpttRecordedSequence {
    batch: usize,
    steps: usize,
    vector_len: usize,
    token_len: usize,
    state_len: usize,
    optimizer_step: Option<RwkvOptimizerStepResult>,
    tied_embedding_optimizer_step: Option<RwkvOptimizerStepResult>,
}

impl RwkvTbpttRecordedSequence {
    pub(crate) fn with_recurrent_optimizer_step(
        mut self,
        optimizer_step: RwkvOptimizerStepResult,
    ) -> Self {
        self.optimizer_step = Some(optimizer_step);
        self
    }

    fn with_tied_embedding_optimizer_step(
        mut self,
        optimizer_step: RwkvOptimizerStepResult,
    ) -> Self {
        self.tied_embedding_optimizer_step = Some(optimizer_step);
        self
    }
}

/// Mutable ticket for token-level graph recording. Unlike the compatibility
/// sequence recorder, this deliberately leaves a seam between every forward
/// and backward timestep so higher-level Hierarchos projections can consume
/// recurrent outputs and later feed their Vulkan-resident gradients back into
/// the cell without a host round trip.
pub(crate) struct RwkvTbpttGraphTicket {
    batch: usize,
    steps: usize,
    vector_len: usize,
    token_len: usize,
    state_len: usize,
    schedule: RwkvTbpttSchedule,
    next_forward: usize,
    next_backward: usize,
    backward_started: bool,
}

/// Independent device-resident activation/state arena for an additional live
/// recurrent graph ticket. The cell kernels and optimizer state remain shared;
/// only values that must survive until reverse-mode replay are duplicated.
/// This lets a shadow recurrence remain intact while the committed recurrence
/// restarts from the original real state in the primary scheduler workspace.
pub(crate) struct RwkvTbpttGraphWorkspace {
    x_steps: Vec<GpuBuffer>,
    token_steps: Vec<GpuBuffer>,
    token_id_steps: Vec<GpuBuffer>,
    grad_output_steps: Vec<GpuBuffer>,
    state_history: Vec<GpuBuffer>,
    output_steps: Vec<GpuBuffer>,
    grad_x_steps: Vec<GpuBuffer>,
    grad_token_steps: Vec<GpuBuffer>,
    state_grad_carry: GpuBuffer,
}

/// Controls the shared `lm_head.weight` phase for token-ID recurrent training.
/// Cell parameters still perform their normal per-sequence AdamW step; this
/// mode only controls the tied LM-head gradient/optimizer so H-DeepEmbed,
/// L-DeepEmbed, and the final LM loss can contribute to one shared update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedLmHeadTrainMode {
    /// Clear prior tied gradients and step the shared LM head in this call.
    Standalone,
    /// Clear prior tied gradients, accumulate this branch, but defer the LM step.
    BeginAccumulation,
    /// Preserve prior tied gradients and add this branch without stepping.
    Accumulate,
}

impl SharedLmHeadTrainMode {
    fn reset_gradient(self) -> bool {
        matches!(self, Self::Standalone | Self::BeginAccumulation)
    }

    fn step_parameter(self) -> bool {
        matches!(self, Self::Standalone)
    }
}

/// Controls whether a recurrent graph ticket starts a fresh cell-gradient
/// accumulation window or contributes to one that is already open. Optimizer
/// stepping is deliberately separate from this mode so forked recurrences can
/// record several backward branches before advancing AdamW exactly once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RwkvRecurrentGradientMode {
    /// Clear the persistent recurrent gradient accumulator before this branch.
    Reset,
    /// Preserve the existing accumulator and add this branch's gradients.
    Accumulate,
}

impl RwkvRecurrentGradientMode {
    fn reset_gradient(self) -> bool {
        matches!(self, Self::Reset)
    }
}

#[derive(Clone, Copy)]
enum TbpttTokenInput<'a> {
    Features(&'a [f32]),
    TokenIds(&'a [u32]),
}

/// Vulkan-resident recurrent sequence scheduler.
///
/// The forward sweep stores only packed/clamped states. The reverse sweep
/// recomputes each cell forward immediately before backward, carrying one
/// packed state-gradient buffer and zeroing it at Hierarchos TBPTT detach
/// boundaries. `run` is weights-readonly; `train_step` accumulates every
/// per-timestep cell parameter gradient on-device and performs one persistent
/// AdamW update after the reverse sweep.
pub struct RwkvTbpttSequenceOp {
    device: VulkanDevice,
    cell: RwkvPackedCellOp,
    max_batch: usize,
    max_steps: usize,
    width: usize,
    token_feature_width: usize,
    state_size: usize,
    optimizer: RwkvPersistentAdamW,
    tied_embedding: Option<TiedTokenEmbeddingOp>,

    x_steps: Vec<GpuBuffer>,
    token_steps: Vec<GpuBuffer>,
    token_id_steps: Vec<GpuBuffer>,
    grad_output_steps: Vec<GpuBuffer>,
    state_history: Vec<GpuBuffer>,
    output_steps: Vec<GpuBuffer>,
    grad_x_steps: Vec<GpuBuffer>,
    grad_token_steps: Vec<GpuBuffer>,
    state_grad_carry: GpuBuffer,
    zero_state_grad: GpuBuffer,

    output_readbacks: Vec<GpuBuffer>,
    grad_x_readbacks: Vec<GpuBuffer>,
    grad_token_readbacks: Vec<GpuBuffer>,
    final_state_readback: GpuBuffer,
    initial_state_grad_readback: GpuBuffer,
    current_gradient_trace: RefCell<Option<RwkvCurrentGradientTrace>>,
}

struct RwkvCurrentGradientTrace {
    trainable_name: String,
    len: usize,
    device_snapshots: Vec<GpuBuffer>,
    readbacks: Vec<GpuBuffer>,
}

impl RwkvTbpttSequenceOp {
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        cell_prefix: &str,
        deepembed_adapter_prefix: &str,
        head_size: usize,
        max_batch: usize,
        max_steps: usize,
        key_clamp: f32,
        deepembed_clamp: f32,
        state_mode: RwkvStateReadoutMode,
        state_clamp: f32,
    ) -> Result<Self> {
        if max_batch == 0 || max_steps == 0 {
            bail!("TBPTT max_batch and max_steps must be positive");
        }
        let cell = RwkvPackedCellOp::from_model_package(
            device.clone(),
            model_dir,
            cell_prefix,
            deepembed_adapter_prefix,
            head_size,
            max_batch,
            key_clamp,
            deepembed_clamp,
            state_mode,
            state_clamp,
        )?;
        Self::new(device, cell, max_batch, max_steps, None)
    }

    /// Construct the recurrent scheduler with a device-resident view of the
    /// tied token embedding. `run_with_token_ids` and
    /// `train_step_with_token_ids` can then gather DeepEmbed token features
    /// directly from standard `lm_head.weight` without Python materializing
    /// the embedding tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_package_with_tied_embedding(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        cell_prefix: &str,
        deepembed_adapter_prefix: &str,
        head_size: usize,
        max_batch: usize,
        max_steps: usize,
        key_clamp: f32,
        deepembed_clamp: f32,
        state_mode: RwkvStateReadoutMode,
        state_clamp: f32,
    ) -> Result<Self> {
        if max_batch == 0 || max_steps == 0 {
            bail!("TBPTT max_batch and max_steps must be positive");
        }
        let model_dir = model_dir.as_ref();
        let cell = RwkvPackedCellOp::from_model_package(
            device.clone(),
            model_dir,
            cell_prefix,
            deepembed_adapter_prefix,
            head_size,
            max_batch,
            key_clamp,
            deepembed_clamp,
            state_mode,
            state_clamp,
        )?;
        let tied_embedding =
            TiedTokenEmbeddingOp::from_model_package(device.clone(), model_dir, max_batch)?;
        if tied_embedding.dim() != cell.token_feature_width() {
            bail!(
                "tied embedding width {} does not match DeepEmbed token-feature width {}",
                tied_embedding.dim(),
                cell.token_feature_width()
            );
        }
        Self::new(device, cell, max_batch, max_steps, Some(tied_embedding))
    }

    /// Construct a recurrent scheduler that aliases an existing shared
    /// `lm_head.weight` parameter identity. Multiple schedulers created with
    /// clones of the same `SharedLmHeadParameter` use one Vulkan allocation,
    /// one gradient accumulator, and one AdamW moment/step state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_package_with_shared_tied_embedding(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        cell_prefix: &str,
        deepembed_adapter_prefix: &str,
        head_size: usize,
        max_batch: usize,
        max_steps: usize,
        key_clamp: f32,
        deepembed_clamp: f32,
        state_mode: RwkvStateReadoutMode,
        state_clamp: f32,
        shared_lm_head: SharedLmHeadParameter,
    ) -> Result<Self> {
        if max_batch == 0 || max_steps == 0 {
            bail!("TBPTT max_batch and max_steps must be positive");
        }
        let model_dir = model_dir.as_ref();
        let cell = RwkvPackedCellOp::from_model_package(
            device.clone(),
            model_dir,
            cell_prefix,
            deepembed_adapter_prefix,
            head_size,
            max_batch,
            key_clamp,
            deepembed_clamp,
            state_mode,
            state_clamp,
        )?;
        let tied_embedding =
            TiedTokenEmbeddingOp::from_shared_parameter(shared_lm_head, max_batch)?;
        if tied_embedding.dim() != cell.token_feature_width() {
            bail!(
                "shared tied embedding width {} does not match DeepEmbed token-feature width {}",
                tied_embedding.dim(),
                cell.token_feature_width()
            );
        }
        Self::new(device, cell, max_batch, max_steps, Some(tied_embedding))
    }

    fn new(
        device: VulkanDevice,
        cell: RwkvPackedCellOp,
        max_batch: usize,
        max_steps: usize,
        tied_embedding: Option<TiedTokenEmbeddingOp>,
    ) -> Result<Self> {
        let width = cell.width();
        let token_feature_width = cell.token_feature_width();
        let state_size = cell.state_size();
        let vector_capacity = max_batch
            .checked_mul(width)
            .context("TBPTT vector capacity overflow")?;
        let token_capacity = max_batch
            .checked_mul(token_feature_width)
            .context("TBPTT token-feature capacity overflow")?;
        let state_capacity = vector_capacity
            .checked_mul(state_size)
            .context("TBPTT packed-state capacity overflow")?;
        let optimizer = {
            let trainables = cell.trainables()?;
            RwkvPersistentAdamW::new(device.clone(), &trainables)?
        };
        Ok(Self {
            x_steps: device_buffers(&device, max_steps, vector_capacity)?,
            token_steps: device_buffers(&device, max_steps, token_capacity)?,
            token_id_steps: u32_device_buffers(&device, max_steps, max_batch)?,
            grad_output_steps: device_buffers(&device, max_steps, vector_capacity)?,
            state_history: device_buffers(&device, max_steps + 1, state_capacity)?,
            output_steps: device_buffers(&device, max_steps, vector_capacity)?,
            grad_x_steps: device_buffers(&device, max_steps, vector_capacity)?,
            grad_token_steps: device_buffers(&device, max_steps, token_capacity)?,
            state_grad_carry: GpuBuffer::zeros_f32(&device, state_capacity)?,
            zero_state_grad: GpuBuffer::zeros_f32(&device, state_capacity)?,
            output_readbacks: host_buffers(&device, max_steps, vector_capacity)?,
            grad_x_readbacks: host_buffers(&device, max_steps, vector_capacity)?,
            grad_token_readbacks: host_buffers(&device, max_steps, token_capacity)?,
            final_state_readback: GpuBuffer::zeros_host_f32(&device, state_capacity)?,
            initial_state_grad_readback: GpuBuffer::zeros_host_f32(&device, state_capacity)?,
            current_gradient_trace: RefCell::new(None),
            device,
            cell,
            max_batch,
            max_steps,
            width,
            token_feature_width,
            state_size,
            optimizer,
            tied_embedding,
        })
    }

    /// Allocate another live recurrent graph arena with the scheduler's full
    /// configured capacity. Forked Hierarchos recurrence currently needs one
    /// such arena for the shadow chain while the ordinary buffers hold the
    /// committed branch.
    pub(crate) fn create_graph_workspace(&self) -> Result<RwkvTbpttGraphWorkspace> {
        let vector_capacity = self
            .max_batch
            .checked_mul(self.width)
            .context("TBPTT graph-workspace vector capacity overflow")?;
        let token_capacity = self
            .max_batch
            .checked_mul(self.token_feature_width)
            .context("TBPTT graph-workspace token capacity overflow")?;
        let state_capacity = vector_capacity
            .checked_mul(self.state_size)
            .context("TBPTT graph-workspace state capacity overflow")?;
        Ok(RwkvTbpttGraphWorkspace {
            x_steps: device_buffers(&self.device, self.max_steps, vector_capacity)?,
            token_steps: device_buffers(&self.device, self.max_steps, token_capacity)?,
            token_id_steps: u32_device_buffers(&self.device, self.max_steps, self.max_batch)?,
            grad_output_steps: device_buffers(&self.device, self.max_steps, vector_capacity)?,
            state_history: device_buffers(&self.device, self.max_steps + 1, state_capacity)?,
            output_steps: device_buffers(&self.device, self.max_steps, vector_capacity)?,
            grad_x_steps: device_buffers(&self.device, self.max_steps, vector_capacity)?,
            grad_token_steps: device_buffers(&self.device, self.max_steps, token_capacity)?,
            state_grad_carry: GpuBuffer::zeros_f32(&self.device, state_capacity)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_feature_sequence: &[f32],
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
    ) -> Result<RwkvTbpttSequenceResult> {
        let (sequence, _, _, _) = self.run_internal(
            batch,
            steps,
            x_sequence,
            TbpttTokenInput::Features(token_feature_sequence),
            initial_packed_state,
            grad_output_sequence,
            final_packed_state_grad,
            schedule,
            None,
            SharedLmHeadTrainMode::Standalone,
        )?;
        Ok(sequence)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_with_token_ids(
        &mut self,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_id_sequence: &[u32],
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
    ) -> Result<RwkvTbpttSequenceResult> {
        let (sequence, _, _, _) = self.run_internal(
            batch,
            steps,
            x_sequence,
            TbpttTokenInput::TokenIds(token_id_sequence),
            initial_packed_state,
            grad_output_sequence,
            final_packed_state_grad,
            schedule,
            None,
            SharedLmHeadTrainMode::Standalone,
        )?;
        Ok(sequence)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_step(
        &mut self,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_feature_sequence: &[f32],
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
        optimizer_hyper: AdamWHyperParams,
    ) -> Result<RwkvTbpttTrainStepResult> {
        let (sequence, optimizer, _, parameters) = self.run_internal(
            batch,
            steps,
            x_sequence,
            TbpttTokenInput::Features(token_feature_sequence),
            initial_packed_state,
            grad_output_sequence,
            final_packed_state_grad,
            schedule,
            Some(optimizer_hyper),
            SharedLmHeadTrainMode::Standalone,
        )?;
        Ok(RwkvTbpttTrainStepResult {
            sequence,
            optimizer: optimizer.context("TBPTT training step did not produce optimizer state")?,
            parameters,
            tied_embedding_optimizer: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_step_with_token_ids(
        &mut self,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_id_sequence: &[u32],
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
        optimizer_hyper: AdamWHyperParams,
    ) -> Result<RwkvTbpttTrainStepResult> {
        let result = self.train_step_with_token_ids_shared_lm_mode(
            batch,
            steps,
            x_sequence,
            token_id_sequence,
            initial_packed_state,
            grad_output_sequence,
            final_packed_state_grad,
            schedule,
            optimizer_hyper,
            SharedLmHeadTrainMode::Standalone,
        )?;
        if result.tied_embedding_optimizer.is_none() {
            bail!("standalone token-ID TBPTT did not step the shared lm_head optimizer");
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_step_with_token_ids_shared_lm_mode(
        &mut self,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_id_sequence: &[u32],
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
        optimizer_hyper: AdamWHyperParams,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttTrainStepResult> {
        let (sequence, optimizer, tied_embedding_optimizer, parameters) = self.run_internal(
            batch,
            steps,
            x_sequence,
            TbpttTokenInput::TokenIds(token_id_sequence),
            initial_packed_state,
            grad_output_sequence,
            final_packed_state_grad,
            schedule,
            Some(optimizer_hyper),
            shared_lm_mode,
        )?;
        Ok(RwkvTbpttTrainStepResult {
            sequence,
            optimizer: optimizer.context("TBPTT training step did not produce optimizer state")?,
            parameters,
            tied_embedding_optimizer,
        })
    }

    /// Accumulate gradients from several recurrent branches into one deferred
    /// RWKV-cell AdamW step and one tied `lm_head.weight` AdamW step. The branch
    /// computations are encoded into a single Vulkan submission, scratch
    /// storage is reused between branches, and only the final branch schedules
    /// sequence readback. This is the optimizer-lifecycle primitive needed by
    /// forked Hierarchos recurrence; graph-level drift/state dependencies are
    /// layered above it rather than being approximated as linear timesteps.
    pub fn train_step_with_token_ids_accumulated_branches(
        &mut self,
        branches: &[RwkvTbpttBranchInput<'_>],
        optimizer_hyper: AdamWHyperParams,
    ) -> Result<RwkvTbpttTrainStepResult> {
        optimizer_hyper.validate()?;
        if branches.is_empty() {
            bail!("deferred recurrent training requires at least one branch");
        }
        self.tied_embedding
            .as_ref()
            .context("deferred token-ID TBPTT requires a tied embedding")?;

        let vector_capacity = self
            .max_batch
            .checked_mul(self.width)
            .context("deferred recurrent vector capacity overflow")?;
        let state_capacity = vector_capacity
            .checked_mul(self.state_size)
            .context("deferred recurrent state capacity overflow")?;
        let x_scratch = GpuBuffer::zeros_f32(&self.device, vector_capacity)?;
        let grad_output_scratch = GpuBuffer::zeros_f32(&self.device, vector_capacity)?;
        let final_state_grad_scratch = GpuBuffer::zeros_f32(&self.device, state_capacity)?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let mut final_recorded = None;

        for (branch_index, branch) in branches.iter().enumerate() {
            if branch.batch == 0 || branch.batch > self.max_batch {
                bail!(
                    "deferred recurrent branch {branch_index} batch must be in 1..={}; got {}",
                    self.max_batch,
                    branch.batch
                );
            }
            if branch.steps == 0 || branch.steps > self.max_steps {
                bail!(
                    "deferred recurrent branch {branch_index} steps must be in 1..={}; got {}",
                    self.max_steps,
                    branch.steps
                );
            }
            let vector_len = branch.batch * self.width;
            let state_len = vector_len * self.state_size;
            validate_sequence(
                "deferred branch x_sequence",
                branch.x_sequence,
                branch.steps * vector_len,
            )?;
            validate_sequence(
                "deferred branch grad_output_sequence",
                branch.grad_output_sequence,
                branch.steps * vector_len,
            )?;
            validate_sequence(
                "deferred branch initial_packed_state",
                branch.initial_packed_state,
                state_len,
            )?;
            validate_token_sequence(
                "deferred branch token_id_sequence",
                branch.token_id_sequence,
                branch.steps * branch.batch,
                self.tied_embedding
                    .as_ref()
                    .context("deferred token-ID TBPTT is missing tied embedding state")?
                    .vocab_size(),
            )?;
            if let Some(final_grad) = branch.final_packed_state_grad {
                validate_sequence(
                    "deferred branch final_packed_state_grad",
                    final_grad,
                    state_len,
                )?;
            }

            let recurrent_mode = if branch_index == 0 {
                RwkvRecurrentGradientMode::Reset
            } else {
                RwkvRecurrentGradientMode::Accumulate
            };
            let lm_mode = if branch_index == 0 {
                SharedLmHeadTrainMode::BeginAccumulation
            } else {
                SharedLmHeadTrainMode::Accumulate
            };
            let mut ticket = self.record_graph_begin_with_token_ids_accumulating(
                &mut commands,
                branch.batch,
                branch.steps,
                branch.initial_packed_state,
                branch.schedule,
                recurrent_mode,
                lm_mode,
            )?;

            for timestep in 0..branch.steps {
                let vector_start = timestep * vector_len;
                commands.upload_f32(
                    &x_scratch,
                    &branch.x_sequence[vector_start..vector_start + vector_len],
                )?;
                let token_start = timestep * branch.batch;
                self.record_graph_forward_token_ids(
                    &mut commands,
                    &mut ticket,
                    timestep,
                    &x_scratch,
                    &branch.token_id_sequence[token_start..token_start + branch.batch],
                )?;
            }

            let final_state_grad = if let Some(final_grad) = branch.final_packed_state_grad {
                commands.upload_f32(&final_state_grad_scratch, final_grad)?;
                Some(&final_state_grad_scratch)
            } else {
                None
            };
            self.record_graph_begin_backward(&mut commands, &mut ticket, final_state_grad)?;
            for timestep in (0..branch.steps).rev() {
                let vector_start = timestep * vector_len;
                commands.upload_f32(
                    &grad_output_scratch,
                    &branch.grad_output_sequence[vector_start..vector_start + vector_len],
                )?;
                self.record_graph_backward_step(
                    &mut commands,
                    &mut ticket,
                    timestep,
                    &grad_output_scratch,
                )?;
            }

            if branch_index + 1 == branches.len() {
                final_recorded =
                    Some(self.record_graph_finish_accumulation(&mut commands, ticket)?);
            } else {
                self.record_graph_finish_shadow_accumulation(&mut commands, ticket)?;
            }
        }

        let recurrent_optimizer = self
            .record_recurrent_optimizer_step_after_accumulation(&mut commands, optimizer_hyper)?;
        let embedding = self
            .tied_embedding
            .as_ref()
            .context("deferred token-ID TBPTT is missing tied embedding state")?;
        let tied_parameter = embedding.shared_parameter();
        let tied_step = tied_parameter.record_step(&mut commands, optimizer_hyper)?;
        tied_parameter.record_readback(&mut commands)?;
        let recorded = final_recorded
            .context("deferred recurrent training produced no final branch")?
            .with_recurrent_optimizer_step(recurrent_optimizer)
            .with_tied_embedding_optimizer_step(RwkvOptimizerStepResult {
                step: tied_step,
                tensor_count: 1,
            });
        commands.submit()?;
        self.finalize_recorded_train_step(recorded)
    }

    /// Execute the core recurrent shape of Hierarchos worker refinement: a
    /// multi-step shadow chain remains live in its own activation/state arena,
    /// then a committed chain restarts from its independently supplied real
    /// state in the primary arena. Both backwards contribute to one recurrent
    /// and one tied-embedding AdamW update. Higher graph layers are responsible
    /// for making the committed input depend on the shadow drift trajectory;
    /// this method establishes that the recurrent fork itself no longer aliases
    /// or overwrites the shadow tape.
    pub fn train_step_with_token_ids_forked_shadow_commit(
        &mut self,
        shadow: RwkvTbpttBranchInput<'_>,
        committed: RwkvTbpttBranchInput<'_>,
        optimizer_hyper: AdamWHyperParams,
    ) -> Result<RwkvTbpttTrainStepResult> {
        optimizer_hyper.validate()?;
        self.tied_embedding
            .as_ref()
            .context("forked token-ID TBPTT requires a tied embedding")?;
        self.validate_host_branch(&shadow, "shadow")?;
        self.validate_host_branch(&committed, "committed")?;

        let vector_capacity = self
            .max_batch
            .checked_mul(self.width)
            .context("forked recurrent vector capacity overflow")?;
        let state_capacity = vector_capacity
            .checked_mul(self.state_size)
            .context("forked recurrent state capacity overflow")?;
        let x_scratch = GpuBuffer::zeros_f32(&self.device, vector_capacity)?;
        let grad_output_scratch = GpuBuffer::zeros_f32(&self.device, vector_capacity)?;
        let final_state_grad_scratch = GpuBuffer::zeros_f32(&self.device, state_capacity)?;
        let shadow_workspace = self.create_graph_workspace()?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;

        let mut shadow_ticket = self.record_workspace_graph_begin_with_token_ids(
            &mut commands,
            &shadow_workspace,
            shadow.batch,
            shadow.steps,
            shadow.initial_packed_state,
            shadow.schedule,
            RwkvRecurrentGradientMode::Reset,
            SharedLmHeadTrainMode::BeginAccumulation,
        )?;
        for timestep in 0..shadow.steps {
            let vector_len = shadow.batch * self.width;
            let vector_start = timestep * vector_len;
            commands.upload_f32(
                &x_scratch,
                &shadow.x_sequence[vector_start..vector_start + vector_len],
            )?;
            let token_start = timestep * shadow.batch;
            self.record_workspace_graph_forward_token_ids(
                &mut commands,
                &shadow_workspace,
                &mut shadow_ticket,
                timestep,
                &x_scratch,
                &shadow.token_id_sequence[token_start..token_start + shadow.batch],
            )?;
        }

        let mut committed_ticket = self.record_graph_begin_with_token_ids_accumulating(
            &mut commands,
            committed.batch,
            committed.steps,
            committed.initial_packed_state,
            committed.schedule,
            RwkvRecurrentGradientMode::Accumulate,
            SharedLmHeadTrainMode::Accumulate,
        )?;
        for timestep in 0..committed.steps {
            let vector_len = committed.batch * self.width;
            let vector_start = timestep * vector_len;
            commands.upload_f32(
                &x_scratch,
                &committed.x_sequence[vector_start..vector_start + vector_len],
            )?;
            let token_start = timestep * committed.batch;
            self.record_graph_forward_token_ids(
                &mut commands,
                &mut committed_ticket,
                timestep,
                &x_scratch,
                &committed.token_id_sequence[token_start..token_start + committed.batch],
            )?;
        }

        let committed_final_grad = if let Some(final_grad) = committed.final_packed_state_grad {
            commands.upload_f32(&final_state_grad_scratch, final_grad)?;
            Some(&final_state_grad_scratch)
        } else {
            None
        };
        self.record_graph_begin_backward(
            &mut commands,
            &mut committed_ticket,
            committed_final_grad,
        )?;
        for timestep in (0..committed.steps).rev() {
            let vector_len = committed.batch * self.width;
            let vector_start = timestep * vector_len;
            commands.upload_f32(
                &grad_output_scratch,
                &committed.grad_output_sequence[vector_start..vector_start + vector_len],
            )?;
            self.record_graph_backward_step(
                &mut commands,
                &mut committed_ticket,
                timestep,
                &grad_output_scratch,
            )?;
        }
        let committed_recorded =
            self.record_graph_finish_accumulation(&mut commands, committed_ticket)?;

        let shadow_final_grad = if let Some(final_grad) = shadow.final_packed_state_grad {
            commands.upload_f32(&final_state_grad_scratch, final_grad)?;
            Some(&final_state_grad_scratch)
        } else {
            None
        };
        self.record_workspace_graph_begin_backward(
            &mut commands,
            &shadow_workspace,
            &mut shadow_ticket,
            shadow_final_grad,
        )?;
        for timestep in (0..shadow.steps).rev() {
            let vector_len = shadow.batch * self.width;
            let vector_start = timestep * vector_len;
            commands.upload_f32(
                &grad_output_scratch,
                &shadow.grad_output_sequence[vector_start..vector_start + vector_len],
            )?;
            self.record_workspace_graph_backward_step(
                &mut commands,
                &shadow_workspace,
                &mut shadow_ticket,
                timestep,
                &grad_output_scratch,
            )?;
        }
        self.record_workspace_graph_finish_shadow_accumulation(
            &mut commands,
            &shadow_workspace,
            shadow_ticket,
        )?;

        let recurrent_optimizer = self
            .record_recurrent_optimizer_step_after_accumulation(&mut commands, optimizer_hyper)?;
        let tied_parameter = self
            .tied_embedding
            .as_ref()
            .context("forked token-ID TBPTT is missing tied embedding state")?
            .shared_parameter();
        let tied_step = tied_parameter.record_step(&mut commands, optimizer_hyper)?;
        tied_parameter.record_readback(&mut commands)?;
        let recorded = committed_recorded
            .with_recurrent_optimizer_step(recurrent_optimizer)
            .with_tied_embedding_optimizer_step(RwkvOptimizerStepResult {
                step: tied_step,
                tensor_count: 1,
            });
        commands.submit()?;
        self.finalize_recorded_train_step(recorded)
    }

    fn validate_host_branch(&self, branch: &RwkvTbpttBranchInput<'_>, label: &str) -> Result<()> {
        if branch.batch == 0 || branch.batch > self.max_batch {
            bail!(
                "{label} recurrent branch batch must be in 1..={}; got {}",
                self.max_batch,
                branch.batch
            );
        }
        if branch.steps == 0 || branch.steps > self.max_steps {
            bail!(
                "{label} recurrent branch steps must be in 1..={}; got {}",
                self.max_steps,
                branch.steps
            );
        }
        let vector_len = branch.batch * self.width;
        let state_len = vector_len * self.state_size;
        validate_sequence(
            &format!("{label} branch x_sequence"),
            branch.x_sequence,
            branch.steps * vector_len,
        )?;
        validate_sequence(
            &format!("{label} branch grad_output_sequence"),
            branch.grad_output_sequence,
            branch.steps * vector_len,
        )?;
        validate_sequence(
            &format!("{label} branch initial_packed_state"),
            branch.initial_packed_state,
            state_len,
        )?;
        validate_token_sequence(
            &format!("{label} branch token_id_sequence"),
            branch.token_id_sequence,
            branch.steps * branch.batch,
            self.tied_embedding
                .as_ref()
                .context("forked token-ID TBPTT is missing tied embedding state")?
                .vocab_size(),
        )?;
        if let Some(final_grad) = branch.final_packed_state_grad {
            validate_sequence(
                &format!("{label} branch final_packed_state_grad"),
                final_grad,
                state_len,
            )?;
        }
        Ok(())
    }

    /// Record a token-ID TBPTT training sequence into a caller-owned Vulkan
    /// batch. This performs uploads, forward/recompute/backward, gradient
    /// accumulation, optional tied-LM accumulation, optimizer recording, and
    /// readback-copy recording without submitting the queue.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_train_step_with_token_ids_shared_lm_mode(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_id_sequence: &[u32],
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
        optimizer_hyper: AdamWHyperParams,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttRecordedSequence> {
        self.record_internal(
            commands,
            batch,
            steps,
            x_sequence,
            TbpttTokenInput::TokenIds(token_id_sequence),
            initial_packed_state,
            grad_output_sequence,
            final_packed_state_grad,
            schedule,
            Some(optimizer_hyper),
            shared_lm_mode,
        )
    }

    /// Finish a training result after the caller-owned batch containing the
    /// sequence has completed.
    pub(crate) fn finalize_recorded_train_step(
        &self,
        recorded: RwkvTbpttRecordedSequence,
    ) -> Result<RwkvTbpttTrainStepResult> {
        let (sequence, optimizer, tied_embedding_optimizer, parameters) =
            self.finalize_recorded(recorded)?;
        Ok(RwkvTbpttTrainStepResult {
            sequence,
            optimizer: optimizer
                .context("recorded TBPTT training step did not produce optimizer state")?,
            parameters,
            tied_embedding_optimizer,
        })
    }

    /// Begin a token-ID TBPTT training sequence whose residual inputs and
    /// upstream gradients will be supplied by Vulkan buffers one timestep at a
    /// time. This is the graph-composition path used by manager/worker
    /// projections; it intentionally does not record any recurrent forward
    /// work until `record_graph_forward_token_ids` is called.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_graph_begin_with_token_ids(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        steps: usize,
        initial_packed_state: &[f32],
        schedule: RwkvTbpttSchedule,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttGraphTicket> {
        self.record_graph_begin_with_token_ids_accumulating(
            commands,
            batch,
            steps,
            initial_packed_state,
            schedule,
            RwkvRecurrentGradientMode::Reset,
            shared_lm_mode,
        )
    }

    /// Begin a graph ticket that either resets or preserves the recurrent
    /// cell-gradient accumulator. `Accumulate` is the forked-recurrence seam:
    /// shadow worker/manager branches can reuse the same physical recurrent
    /// parameter identity and contribute gradients without advancing moments
    /// or the global AdamW step.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_graph_begin_with_token_ids_accumulating(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        steps: usize,
        initial_packed_state: &[f32],
        schedule: RwkvTbpttSchedule,
        recurrent_gradient_mode: RwkvRecurrentGradientMode,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttGraphTicket> {
        if batch == 0 || batch > self.max_batch {
            bail!("TBPTT batch must be in 1..={}; got {batch}", self.max_batch);
        }
        if steps == 0 || steps > self.max_steps {
            bail!("TBPTT steps must be in 1..={}; got {steps}", self.max_steps);
        }
        if schedule.detach_every_n_steps == Some(0) {
            bail!("TBPTT detach interval must be positive when enabled");
        }
        self.tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT requires a tied embedding")?;

        let vector_len = batch * self.width;
        let token_len = batch * self.token_feature_width;
        let state_len = vector_len * self.state_size;
        validate_sequence("initial_packed_state", initial_packed_state, state_len)?;
        commands.upload_f32(&self.state_history[0], initial_packed_state)?;
        if recurrent_gradient_mode.reset_gradient() {
            self.optimizer.record_zero_grad(commands)?;
        }
        if shared_lm_mode.reset_gradient() {
            self.tied_embedding
                .as_ref()
                .context("token-ID graph TBPTT is missing tied embedding state")?
                .record_zero_grad(commands)?;
        }

        Ok(RwkvTbpttGraphTicket {
            batch,
            steps,
            vector_len,
            token_len,
            state_len,
            schedule,
            next_forward: 0,
            next_backward: steps,
            backward_started: false,
        })
    }

    /// Begin the primary graph arena from an already device-resident packed
    /// state. This is the committed-state handoff seam used by higher-level
    /// sequence owners: a previous token can leave its selected/final state on
    /// the GPU and the next token can start without a host readback/re-upload.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_graph_begin_with_token_ids_from_state_buffer_accumulating(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        steps: usize,
        initial_packed_state: &GpuBuffer,
        schedule: RwkvTbpttSchedule,
        recurrent_gradient_mode: RwkvRecurrentGradientMode,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttGraphTicket> {
        if batch == 0 || batch > self.max_batch {
            bail!("TBPTT batch must be in 1..={}; got {batch}", self.max_batch);
        }
        if steps == 0 || steps > self.max_steps {
            bail!("TBPTT steps must be in 1..={}; got {steps}", self.max_steps);
        }
        if schedule.detach_every_n_steps == Some(0) {
            bail!("TBPTT detach interval must be positive when enabled");
        }
        self.tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT requires a tied embedding")?;

        let vector_len = batch * self.width;
        let token_len = batch * self.token_feature_width;
        let state_len = vector_len * self.state_size;
        commands.copy_f32(initial_packed_state, &self.state_history[0], state_len)?;
        if recurrent_gradient_mode.reset_gradient() {
            self.optimizer.record_zero_grad(commands)?;
        }
        if shared_lm_mode.reset_gradient() {
            self.tied_embedding
                .as_ref()
                .context("token-ID graph TBPTT is missing tied embedding state")?
                .record_zero_grad(commands)?;
        }

        Ok(RwkvTbpttGraphTicket {
            batch,
            steps,
            vector_len,
            token_len,
            state_len,
            schedule,
            next_forward: 0,
            next_backward: steps,
            backward_started: false,
        })
    }

    /// Begin a ticket in an independent graph workspace. This has the same
    /// optimizer-gradient lifecycle as the primary graph path, but none of its
    /// activation/state storage aliases the primary ticket.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_workspace_graph_begin_with_token_ids(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        workspace: &RwkvTbpttGraphWorkspace,
        batch: usize,
        steps: usize,
        initial_packed_state: &[f32],
        schedule: RwkvTbpttSchedule,
        recurrent_gradient_mode: RwkvRecurrentGradientMode,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttGraphTicket> {
        if batch == 0 || batch > self.max_batch {
            bail!("TBPTT batch must be in 1..={}; got {batch}", self.max_batch);
        }
        if steps == 0 || steps > self.max_steps {
            bail!("TBPTT steps must be in 1..={}; got {steps}", self.max_steps);
        }
        if schedule.detach_every_n_steps == Some(0) {
            bail!("TBPTT detach interval must be positive when enabled");
        }
        self.tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT requires a tied embedding")?;

        let vector_len = batch * self.width;
        let token_len = batch * self.token_feature_width;
        let state_len = vector_len * self.state_size;
        validate_sequence("initial_packed_state", initial_packed_state, state_len)?;
        commands.upload_f32(&workspace.state_history[0], initial_packed_state)?;
        if recurrent_gradient_mode.reset_gradient() {
            self.optimizer.record_zero_grad(commands)?;
        }
        if shared_lm_mode.reset_gradient() {
            self.tied_embedding
                .as_ref()
                .context("token-ID graph TBPTT is missing tied embedding state")?
                .record_zero_grad(commands)?;
        }

        Ok(RwkvTbpttGraphTicket {
            batch,
            steps,
            vector_len,
            token_len,
            state_len,
            schedule,
            next_forward: 0,
            next_backward: steps,
            backward_started: false,
        })
    }

    /// Begin an independent graph workspace from an already device-resident
    /// packed state. Manager pondering uses this seam to start its shadow H
    /// chain from the real H transition without a state readback/re-upload.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_workspace_graph_begin_with_token_ids_from_state_buffer(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        workspace: &RwkvTbpttGraphWorkspace,
        batch: usize,
        steps: usize,
        initial_packed_state: &GpuBuffer,
        schedule: RwkvTbpttSchedule,
        recurrent_gradient_mode: RwkvRecurrentGradientMode,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttGraphTicket> {
        if batch == 0 || batch > self.max_batch {
            bail!("TBPTT batch must be in 1..={}; got {batch}", self.max_batch);
        }
        if steps == 0 || steps > self.max_steps {
            bail!("TBPTT steps must be in 1..={}; got {steps}", self.max_steps);
        }
        if schedule.detach_every_n_steps == Some(0) {
            bail!("TBPTT detach interval must be positive when enabled");
        }
        self.tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT requires a tied embedding")?;

        let vector_len = batch * self.width;
        let token_len = batch * self.token_feature_width;
        let state_len = vector_len * self.state_size;
        commands.copy_f32(initial_packed_state, &workspace.state_history[0], state_len)?;
        if recurrent_gradient_mode.reset_gradient() {
            self.optimizer.record_zero_grad(commands)?;
        }
        if shared_lm_mode.reset_gradient() {
            self.tied_embedding
                .as_ref()
                .context("token-ID graph TBPTT is missing tied embedding state")?
                .record_zero_grad(commands)?;
        }

        Ok(RwkvTbpttGraphTicket {
            batch,
            steps,
            vector_len,
            token_len,
            state_len,
            schedule,
            next_forward: 0,
            next_backward: steps,
            backward_started: false,
        })
    }

    /// Record exactly one recurrent forward timestep from a device-resident
    /// residual input. The output/state remain resident and can be consumed by
    /// projection nodes before the next graph operation is recorded.
    pub(crate) fn record_graph_forward_token_ids(
        &self,
        commands: &mut vulkan::ComputeBatch,
        ticket: &mut RwkvTbpttGraphTicket,
        timestep: usize,
        x: &GpuBuffer,
        token_ids: &[u32],
    ) -> Result<()> {
        if ticket.backward_started {
            bail!("TBPTT graph forward cannot continue after backward has started");
        }
        if timestep != ticket.next_forward || timestep >= ticket.steps {
            bail!(
                "TBPTT graph forward expected timestep {}; got {timestep}",
                ticket.next_forward
            );
        }
        let embedding = self
            .tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT is missing tied embedding state")?;
        validate_token_sequence(
            "token_id_step",
            token_ids,
            ticket.batch,
            embedding.vocab_size(),
        )?;

        commands.copy_f32(x, &self.x_steps[timestep], ticket.vector_len)?;
        commands.upload_u32(&self.token_id_steps[timestep], token_ids)?;
        embedding.record_forward(
            commands,
            ticket.batch,
            &self.token_id_steps[timestep],
            &self.token_steps[timestep],
        )?;
        self.cell.record_forward_transition_into(
            commands,
            ticket.batch,
            &self.x_steps[timestep],
            &self.token_steps[timestep],
            &self.state_history[timestep],
            &self.state_history[timestep + 1],
            &self.output_steps[timestep],
        )?;
        ticket.next_forward += 1;
        Ok(())
    }

    pub(crate) fn record_workspace_graph_forward_token_ids(
        &self,
        commands: &mut vulkan::ComputeBatch,
        workspace: &RwkvTbpttGraphWorkspace,
        ticket: &mut RwkvTbpttGraphTicket,
        timestep: usize,
        x: &GpuBuffer,
        token_ids: &[u32],
    ) -> Result<()> {
        if ticket.backward_started {
            bail!("TBPTT graph forward cannot continue after backward has started");
        }
        if timestep != ticket.next_forward || timestep >= ticket.steps {
            bail!(
                "TBPTT graph forward expected timestep {}; got {timestep}",
                ticket.next_forward
            );
        }
        let embedding = self
            .tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT is missing tied embedding state")?;
        validate_token_sequence(
            "token_id_step",
            token_ids,
            ticket.batch,
            embedding.vocab_size(),
        )?;

        commands.copy_f32(x, &workspace.x_steps[timestep], ticket.vector_len)?;
        commands.upload_u32(&workspace.token_id_steps[timestep], token_ids)?;
        embedding.record_forward(
            commands,
            ticket.batch,
            &workspace.token_id_steps[timestep],
            &workspace.token_steps[timestep],
        )?;
        self.cell.record_forward_transition_into(
            commands,
            ticket.batch,
            &workspace.x_steps[timestep],
            &workspace.token_steps[timestep],
            &workspace.state_history[timestep],
            &workspace.state_history[timestep + 1],
            &workspace.output_steps[timestep],
        )?;
        ticket.next_forward += 1;
        Ok(())
    }

    /// Seal the forward sweep and initialize the reverse state-gradient carry.
    /// A device-resident final-state gradient may be supplied by a higher-level
    /// recurrent graph; otherwise the reverse sweep starts from zero.
    pub(crate) fn record_graph_begin_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        ticket: &mut RwkvTbpttGraphTicket,
        final_packed_state_grad: Option<&GpuBuffer>,
    ) -> Result<()> {
        if ticket.next_forward != ticket.steps {
            bail!(
                "TBPTT graph backward requires {} recorded forward steps; have {}",
                ticket.steps,
                ticket.next_forward
            );
        }
        if ticket.backward_started {
            bail!("TBPTT graph backward has already started");
        }
        if let Some(final_grad) = final_packed_state_grad {
            commands.copy_f32(final_grad, &self.state_grad_carry, ticket.state_len)?;
        } else {
            commands.copy_f32(
                &self.zero_state_grad,
                &self.state_grad_carry,
                ticket.state_len,
            )?;
        }
        ticket.backward_started = true;
        Ok(())
    }

    pub(crate) fn record_workspace_graph_begin_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        workspace: &RwkvTbpttGraphWorkspace,
        ticket: &mut RwkvTbpttGraphTicket,
        final_packed_state_grad: Option<&GpuBuffer>,
    ) -> Result<()> {
        if ticket.next_forward != ticket.steps {
            bail!(
                "TBPTT graph backward requires {} recorded forward steps; have {}",
                ticket.steps,
                ticket.next_forward
            );
        }
        if ticket.backward_started {
            bail!("TBPTT graph backward has already started");
        }
        if let Some(final_grad) = final_packed_state_grad {
            commands.copy_f32(final_grad, &workspace.state_grad_carry, ticket.state_len)?;
        } else {
            commands.copy_f32(
                &self.zero_state_grad,
                &workspace.state_grad_carry,
                ticket.state_len,
            )?;
        }
        ticket.backward_started = true;
        Ok(())
    }

    /// Record exactly one reverse timestep. `grad_output` is copied from the
    /// caller's device buffer, so gradients assembled by projection/backbone
    /// nodes can flow into RWKV without being materialized on the host.
    pub(crate) fn record_graph_backward_step(
        &self,
        commands: &mut vulkan::ComputeBatch,
        ticket: &mut RwkvTbpttGraphTicket,
        timestep: usize,
        grad_output: &GpuBuffer,
    ) -> Result<()> {
        if !ticket.backward_started {
            bail!("TBPTT graph backward must be initialized before reverse steps");
        }
        if ticket.next_backward == 0 || timestep + 1 != ticket.next_backward {
            bail!(
                "TBPTT graph backward expected timestep {}; got {timestep}",
                ticket.next_backward.saturating_sub(1)
            );
        }
        let next_timestep = timestep + 1;
        if next_timestep < ticket.steps && ticket.schedule.is_detach_boundary(next_timestep) {
            commands.copy_f32(
                &self.zero_state_grad,
                &self.state_grad_carry,
                ticket.state_len,
            )?;
        }
        commands.copy_f32(
            grad_output,
            &self.grad_output_steps[timestep],
            ticket.vector_len,
        )?;
        self.cell.record_rematerialized_forward_backward(
            commands,
            ticket.batch,
            &self.x_steps[timestep],
            &self.token_steps[timestep],
            &self.state_history[timestep],
            &self.grad_output_steps[timestep],
            &self.state_grad_carry,
        )?;
        let trainables = self.cell.trainables()?;
        self.optimizer.record_accumulate(commands, &trainables)?;
        commands.copy_f32(
            self.cell.grad_x_buffer(),
            &self.grad_x_steps[timestep],
            ticket.vector_len,
        )?;
        commands.copy_f32(
            self.cell.token_feature_grad_buffer(),
            &self.grad_token_steps[timestep],
            ticket.token_len,
        )?;
        commands.copy_f32(
            self.cell.grad_packed_state_buffer(),
            &self.state_grad_carry,
            ticket.state_len,
        )?;
        ticket.next_backward -= 1;
        Ok(())
    }

    pub(crate) fn record_workspace_graph_backward_step(
        &self,
        commands: &mut vulkan::ComputeBatch,
        workspace: &RwkvTbpttGraphWorkspace,
        ticket: &mut RwkvTbpttGraphTicket,
        timestep: usize,
        grad_output: &GpuBuffer,
    ) -> Result<()> {
        if !ticket.backward_started {
            bail!("TBPTT graph backward must be initialized before reverse steps");
        }
        if ticket.next_backward == 0 || timestep + 1 != ticket.next_backward {
            bail!(
                "TBPTT graph backward expected timestep {}; got {timestep}",
                ticket.next_backward.saturating_sub(1)
            );
        }
        let next_timestep = timestep + 1;
        if next_timestep < ticket.steps && ticket.schedule.is_detach_boundary(next_timestep) {
            commands.copy_f32(
                &self.zero_state_grad,
                &workspace.state_grad_carry,
                ticket.state_len,
            )?;
        }
        commands.copy_f32(
            grad_output,
            &workspace.grad_output_steps[timestep],
            ticket.vector_len,
        )?;
        self.cell.record_rematerialized_forward_backward(
            commands,
            ticket.batch,
            &workspace.x_steps[timestep],
            &workspace.token_steps[timestep],
            &workspace.state_history[timestep],
            &workspace.grad_output_steps[timestep],
            &workspace.state_grad_carry,
        )?;
        let trainables = self.cell.trainables()?;
        self.optimizer.record_accumulate(commands, &trainables)?;
        commands.copy_f32(
            self.cell.grad_x_buffer(),
            &workspace.grad_x_steps[timestep],
            ticket.vector_len,
        )?;
        commands.copy_f32(
            self.cell.token_feature_grad_buffer(),
            &workspace.grad_token_steps[timestep],
            ticket.token_len,
        )?;
        commands.copy_f32(
            self.cell.grad_packed_state_buffer(),
            &workspace.state_grad_carry,
            ticket.state_len,
        )?;
        ticket.next_backward -= 1;
        Ok(())
    }

    pub(crate) fn record_workspace_graph_finish_shadow_accumulation(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        workspace: &RwkvTbpttGraphWorkspace,
        ticket: RwkvTbpttGraphTicket,
    ) -> Result<()> {
        if !ticket.backward_started || ticket.next_backward != 0 {
            bail!(
                "TBPTT graph finish requires a complete reverse sweep; {} steps remain",
                ticket.next_backward
            );
        }
        let embedding = self
            .tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT is missing tied embedding state")?;
        for timestep in 0..ticket.steps {
            embedding.record_backward_accumulate(
                commands,
                ticket.batch,
                &workspace.token_id_steps[timestep],
                &workspace.grad_token_steps[timestep],
            )?;
        }
        Ok(())
    }

    fn record_graph_finish_gradients(
        &self,
        commands: &mut vulkan::ComputeBatch,
        ticket: &RwkvTbpttGraphTicket,
    ) -> Result<()> {
        if !ticket.backward_started || ticket.next_backward != 0 {
            bail!(
                "TBPTT graph finish requires a complete reverse sweep; {} steps remain",
                ticket.next_backward
            );
        }
        let embedding = self
            .tied_embedding
            .as_ref()
            .context("token-ID graph TBPTT is missing tied embedding state")?;
        for timestep in 0..ticket.steps {
            embedding.record_backward_accumulate(
                commands,
                ticket.batch,
                &self.token_id_steps[timestep],
                &self.grad_token_steps[timestep],
            )?;
        }
        Ok(())
    }

    /// Consume a completed shadow graph branch after its recurrent cell
    /// gradients have already been accumulated by the reverse sweep. This adds
    /// its tied DeepEmbed gradient but deliberately records no optimizer step
    /// and no host readback, allowing another ticket to reuse the scheduler's
    /// scratch buffers in the same queue submission.
    #[allow(dead_code)]
    pub(crate) fn record_graph_finish_shadow_accumulation(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        ticket: RwkvTbpttGraphTicket,
    ) -> Result<()> {
        self.record_graph_finish_gradients(commands, &ticket)
    }

    /// Finish a graph branch by accumulating its tied-embedding gradients and
    /// scheduling sequence readbacks, but do not advance the recurrent or tied
    /// LM optimizer. The returned sequence can later be paired with the one
    /// recurrent AdamW step recorded after every contributing branch.
    pub(crate) fn record_graph_finish_accumulation(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        ticket: RwkvTbpttGraphTicket,
    ) -> Result<RwkvTbpttRecordedSequence> {
        self.record_graph_finish_gradients(commands, &ticket)?;

        for timestep in 0..ticket.steps {
            commands.readback_f32(
                &self.output_steps[timestep],
                &self.output_readbacks[timestep],
                ticket.vector_len,
            )?;
            commands.readback_f32(
                &self.grad_x_steps[timestep],
                &self.grad_x_readbacks[timestep],
                ticket.vector_len,
            )?;
            commands.readback_f32(
                &self.grad_token_steps[timestep],
                &self.grad_token_readbacks[timestep],
                ticket.token_len,
            )?;
        }
        commands.readback_f32(
            &self.state_history[ticket.steps],
            &self.final_state_readback,
            ticket.state_len,
        )?;
        commands.readback_f32(
            &self.state_grad_carry,
            &self.initial_state_grad_readback,
            ticket.state_len,
        )?;

        Ok(RwkvTbpttRecordedSequence {
            batch: ticket.batch,
            steps: ticket.steps,
            vector_len: ticket.vector_len,
            token_len: ticket.token_len,
            state_len: ticket.state_len,
            optimizer_step: None,
            tied_embedding_optimizer_step: None,
        })
    }

    /// Finish a graph branch after its reverse sweep without scheduling any
    /// scheduler-owned host readbacks. This is the outer-token tape path: the
    /// next token may immediately reuse every recurrent scratch/readback slot
    /// in the same command buffer, while sequence-owned state/adjoint/loss
    /// buffers preserve the only values that must outlive the token.
    pub(crate) fn record_graph_finish_accumulation_without_readback(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        ticket: RwkvTbpttGraphTicket,
    ) -> Result<()> {
        self.record_graph_finish_gradients(commands, &ticket)
    }

    /// Advance the recurrent cell's persistent AdamW state once using all
    /// gradients accumulated since the most recent reset, then schedule
    /// parameter readback. This is intentionally independent from graph-ticket
    /// completion so several recurrence branches can share one optimizer step.
    pub(crate) fn record_recurrent_optimizer_step_after_accumulation(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        hyper: AdamWHyperParams,
    ) -> Result<RwkvOptimizerStepResult> {
        let trainables = self.cell.trainables()?;
        let optimizer_step = self.optimizer.record_step(commands, &trainables, hyper)?;
        self.optimizer
            .record_parameter_readback(commands, &trainables)?;
        Ok(optimizer_step)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_internal(
        &mut self,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_input: TbpttTokenInput<'_>,
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
        optimizer_hyper: Option<AdamWHyperParams>,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<(
        RwkvTbpttSequenceResult,
        Option<RwkvOptimizerStepResult>,
        Option<RwkvOptimizerStepResult>,
        Vec<RwkvParameterSnapshot>,
    )> {
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let recorded = self.record_internal(
            &mut commands,
            batch,
            steps,
            x_sequence,
            token_input,
            initial_packed_state,
            grad_output_sequence,
            final_packed_state_grad,
            schedule,
            optimizer_hyper,
            shared_lm_mode,
        )?;
        commands.submit()?;
        self.finalize_recorded(recorded)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_internal(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        steps: usize,
        x_sequence: &[f32],
        token_input: TbpttTokenInput<'_>,
        initial_packed_state: &[f32],
        grad_output_sequence: &[f32],
        final_packed_state_grad: Option<&[f32]>,
        schedule: RwkvTbpttSchedule,
        optimizer_hyper: Option<AdamWHyperParams>,
        shared_lm_mode: SharedLmHeadTrainMode,
    ) -> Result<RwkvTbpttRecordedSequence> {
        if batch == 0 || batch > self.max_batch {
            bail!("TBPTT batch must be in 1..={}; got {batch}", self.max_batch);
        }
        if steps == 0 || steps > self.max_steps {
            bail!("TBPTT steps must be in 1..={}; got {steps}", self.max_steps);
        }
        if schedule.detach_every_n_steps == Some(0) {
            bail!("TBPTT detach interval must be positive when enabled");
        }
        let vector_len = batch * self.width;
        let token_len = batch * self.token_feature_width;
        let state_len = vector_len * self.state_size;
        validate_sequence("x_sequence", x_sequence, steps * vector_len)?;
        let using_token_ids = matches!(token_input, TbpttTokenInput::TokenIds(_));
        match token_input {
            TbpttTokenInput::Features(token_feature_sequence) => validate_sequence(
                "token_feature_sequence",
                token_feature_sequence,
                steps * token_len,
            )?,
            TbpttTokenInput::TokenIds(token_id_sequence) => {
                let embedding = self
                    .tied_embedding
                    .as_ref()
                    .context("token-ID TBPTT requires from_model_package_with_tied_embedding")?;
                validate_token_sequence(
                    "token_id_sequence",
                    token_id_sequence,
                    steps * batch,
                    embedding.vocab_size(),
                )?;
            }
        }
        validate_sequence("initial_packed_state", initial_packed_state, state_len)?;
        validate_sequence(
            "grad_output_sequence",
            grad_output_sequence,
            steps * vector_len,
        )?;
        if let Some(final_grad) = final_packed_state_grad {
            validate_sequence("final_packed_state_grad", final_grad, state_len)?;
        }

        commands.upload_f32(&self.state_history[0], initial_packed_state)?;
        for timestep in 0..steps {
            let x_start = timestep * vector_len;
            commands.upload_f32(
                &self.x_steps[timestep],
                &x_sequence[x_start..x_start + vector_len],
            )?;
            match token_input {
                TbpttTokenInput::Features(token_feature_sequence) => {
                    let token_start = timestep * token_len;
                    commands.upload_f32(
                        &self.token_steps[timestep],
                        &token_feature_sequence[token_start..token_start + token_len],
                    )?;
                }
                TbpttTokenInput::TokenIds(token_id_sequence) => {
                    let token_start = timestep * batch;
                    commands.upload_u32(
                        &self.token_id_steps[timestep],
                        &token_id_sequence[token_start..token_start + batch],
                    )?;
                }
            }
            commands.upload_f32(
                &self.grad_output_steps[timestep],
                &grad_output_sequence[x_start..x_start + vector_len],
            )?;
        }

        for timestep in 0..steps {
            if let TbpttTokenInput::TokenIds(_) = token_input {
                self.tied_embedding
                    .as_ref()
                    .context("token-ID TBPTT is missing tied embedding state")?
                    .record_forward(
                        commands,
                        batch,
                        &self.token_id_steps[timestep],
                        &self.token_steps[timestep],
                    )?;
            }
            self.cell.record_forward_transition_into(
                commands,
                batch,
                &self.x_steps[timestep],
                &self.token_steps[timestep],
                &self.state_history[timestep],
                &self.state_history[timestep + 1],
                &self.output_steps[timestep],
            )?;
        }

        if let Some(final_grad) = final_packed_state_grad {
            commands.upload_f32(&self.state_grad_carry, final_grad)?;
        } else {
            commands.copy_f32(&self.zero_state_grad, &self.state_grad_carry, state_len)?;
        }
        if optimizer_hyper.is_some() {
            self.optimizer.record_zero_grad(commands)?;
            if using_token_ids && shared_lm_mode.reset_gradient() {
                self.tied_embedding
                    .as_ref()
                    .context("token-ID TBPTT is missing tied embedding state")?
                    .record_zero_grad(commands)?;
            }
        }

        for timestep in (0..steps).rev() {
            let next_timestep = timestep + 1;
            if next_timestep < steps && schedule.is_detach_boundary(next_timestep) {
                commands.copy_f32(&self.zero_state_grad, &self.state_grad_carry, state_len)?;
            }
            self.cell.record_rematerialized_forward_backward(
                commands,
                batch,
                &self.x_steps[timestep],
                &self.token_steps[timestep],
                &self.state_history[timestep],
                &self.grad_output_steps[timestep],
                &self.state_grad_carry,
            )?;
            if optimizer_hyper.is_some() {
                let trainables = self.cell.trainables()?;
                self.optimizer.record_accumulate(commands, &trainables)?;
            }
            commands.copy_f32(
                self.cell.grad_x_buffer(),
                &self.grad_x_steps[timestep],
                vector_len,
            )?;
            commands.copy_f32(
                self.cell.token_feature_grad_buffer(),
                &self.grad_token_steps[timestep],
                token_len,
            )?;
            commands.copy_f32(
                self.cell.grad_packed_state_buffer(),
                &self.state_grad_carry,
                state_len,
            )?;
        }

        if optimizer_hyper.is_some() && using_token_ids {
            let embedding = self
                .tied_embedding
                .as_ref()
                .context("token-ID TBPTT is missing tied embedding state")?;
            for timestep in 0..steps {
                embedding.record_backward_accumulate(
                    commands,
                    batch,
                    &self.token_id_steps[timestep],
                    &self.grad_token_steps[timestep],
                )?;
            }
        }

        let mut optimizer_step = None;
        let mut tied_embedding_optimizer_step = None;
        if let Some(hyper) = optimizer_hyper {
            let trainables = self.cell.trainables()?;
            optimizer_step = Some(self.optimizer.record_step(commands, &trainables, hyper)?);
            self.optimizer
                .record_parameter_readback(commands, &trainables)?;

            if using_token_ids && shared_lm_mode.step_parameter() {
                let embedding = self
                    .tied_embedding
                    .as_ref()
                    .context("token-ID TBPTT is missing tied embedding state")?;
                let parameter = embedding.shared_parameter();
                let step = parameter.record_step(commands, hyper)?;
                parameter.record_readback(commands)?;
                tied_embedding_optimizer_step = Some(RwkvOptimizerStepResult {
                    step,
                    tensor_count: 1,
                });
            }
        }

        for timestep in 0..steps {
            commands.readback_f32(
                &self.output_steps[timestep],
                &self.output_readbacks[timestep],
                vector_len,
            )?;
            commands.readback_f32(
                &self.grad_x_steps[timestep],
                &self.grad_x_readbacks[timestep],
                vector_len,
            )?;
            commands.readback_f32(
                &self.grad_token_steps[timestep],
                &self.grad_token_readbacks[timestep],
                token_len,
            )?;
        }
        commands.readback_f32(
            &self.state_history[steps],
            &self.final_state_readback,
            state_len,
        )?;
        commands.readback_f32(
            &self.state_grad_carry,
            &self.initial_state_grad_readback,
            state_len,
        )?;

        Ok(RwkvTbpttRecordedSequence {
            batch,
            steps,
            vector_len,
            token_len,
            state_len,
            optimizer_step,
            tied_embedding_optimizer_step,
        })
    }

    fn finalize_recorded(
        &self,
        recorded: RwkvTbpttRecordedSequence,
    ) -> Result<(
        RwkvTbpttSequenceResult,
        Option<RwkvOptimizerStepResult>,
        Option<RwkvOptimizerStepResult>,
        Vec<RwkvParameterSnapshot>,
    )> {
        let RwkvTbpttRecordedSequence {
            batch,
            steps,
            vector_len,
            token_len,
            state_len,
            optimizer_step,
            tied_embedding_optimizer_step,
        } = recorded;
        let mut parameter_snapshots = if optimizer_step.is_some() {
            self.optimizer.read_parameter_snapshots()?
        } else {
            Vec::new()
        };
        if tied_embedding_optimizer_step.is_some() {
            let parameter = self
                .tied_embedding
                .as_ref()
                .context("token-ID TBPTT is missing tied embedding state")?
                .shared_parameter();
            parameter_snapshots.push(RwkvParameterSnapshot {
                name: "lm_head.weight".to_string(),
                values: parameter.read_recorded_weights()?,
            });
        }

        let mut outputs = Vec::with_capacity(steps * vector_len);
        let mut grad_x = Vec::with_capacity(steps * vector_len);
        let mut token_feature_grad = Vec::with_capacity(steps * token_len);
        for timestep in 0..steps {
            outputs.extend(self.output_readbacks[timestep].read_f32(vector_len)?);
            grad_x.extend(self.grad_x_readbacks[timestep].read_f32(vector_len)?);
            token_feature_grad.extend(self.grad_token_readbacks[timestep].read_f32(token_len)?);
        }
        Ok((
            RwkvTbpttSequenceResult {
                steps,
                batch,
                outputs,
                final_packed_state: self.final_state_readback.read_f32(state_len)?,
                grad_x,
                token_feature_grad,
                grad_initial_packed_state: self.initial_state_grad_readback.read_f32(state_len)?,
            },
            optimizer_step,
            tied_embedding_optimizer_step,
            parameter_snapshots,
        ))
    }

    pub(crate) fn graph_output_step_buffer(&self, timestep: usize) -> Result<&GpuBuffer> {
        self.output_steps
            .get(timestep)
            .with_context(|| format!("TBPTT output timestep {timestep} exceeds graph capacity"))
    }

    pub(crate) fn graph_grad_x_step_buffer(&self, timestep: usize) -> Result<&GpuBuffer> {
        self.grad_x_steps
            .get(timestep)
            .with_context(|| format!("TBPTT grad-x timestep {timestep} exceeds graph capacity"))
    }

    pub(crate) fn graph_state_step_buffer(&self, timestep: usize) -> Result<&GpuBuffer> {
        self.state_history
            .get(timestep)
            .with_context(|| format!("TBPTT state timestep {timestep} exceeds graph capacity"))
    }

    pub(crate) fn graph_state_grad_carry_buffer(&self) -> &GpuBuffer {
        &self.state_grad_carry
    }

    #[allow(dead_code)]
    pub(crate) fn workspace_graph_output_step_buffer<'a>(
        &self,
        workspace: &'a RwkvTbpttGraphWorkspace,
        timestep: usize,
    ) -> Result<&'a GpuBuffer> {
        workspace.output_steps.get(timestep).with_context(|| {
            format!("TBPTT workspace output timestep {timestep} exceeds graph capacity")
        })
    }

    #[allow(dead_code)]
    pub(crate) fn workspace_graph_grad_x_step_buffer<'a>(
        &self,
        workspace: &'a RwkvTbpttGraphWorkspace,
        timestep: usize,
    ) -> Result<&'a GpuBuffer> {
        workspace.grad_x_steps.get(timestep).with_context(|| {
            format!("TBPTT workspace grad-x timestep {timestep} exceeds graph capacity")
        })
    }

    #[allow(dead_code)]
    pub(crate) fn workspace_graph_state_step_buffer<'a>(
        &self,
        workspace: &'a RwkvTbpttGraphWorkspace,
        timestep: usize,
    ) -> Result<&'a GpuBuffer> {
        workspace.state_history.get(timestep).with_context(|| {
            format!("TBPTT workspace state timestep {timestep} exceeds graph capacity")
        })
    }

    #[allow(dead_code)]
    pub(crate) fn workspace_graph_state_grad_carry_buffer<'a>(
        &self,
        workspace: &'a RwkvTbpttGraphWorkspace,
    ) -> &'a GpuBuffer {
        &workspace.state_grad_carry
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn state_size(&self) -> usize {
        self.state_size
    }

    /// Whether first-pass TBPTT forward transitions keep recurrent matrix
    /// state packed and suppress backward-only tape stores on this device.
    pub fn packed_forward_only_active(&self) -> bool {
        self.cell.packed_forward_only_active()
    }

    /// Enable or disable the packed first-forward optimization. This is mainly
    /// useful for parity/benchmark A/B checks; backward rematerialization is
    /// always recorded through the full training path.
    pub fn set_packed_forward_only_enabled(&mut self, enabled: bool) {
        self.cell.set_packed_forward_only_enabled(enabled);
    }

    /// Whether reverse TBPTT can rebuild its training tape and consume matrix
    /// state directly from packed history on this Vulkan device.
    pub fn packed_backward_rematerialization_active(&self) -> bool {
        self.cell.packed_backward_rematerialization_active()
    }

    /// Opt into/out of the packed reverse-rematerialization A/B arm. It remains
    /// disabled by default until parity has been established for a target.
    pub fn set_packed_backward_rematerialization_enabled(&mut self, enabled: bool) {
        self.cell
            .set_packed_backward_rematerialization_enabled(enabled);
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

    pub fn available_backward_kernel_geometry_labels(&self, batch: usize) -> Result<Vec<String>> {
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

    pub fn set_backward_kernel_geometry_label(&mut self, batch: usize, label: &str) -> Result<()> {
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

    pub fn low_rank_fp16_parameter_storage_active(&self) -> bool {
        self.cell.low_rank_fp16_parameter_storage_active()
    }

    pub fn low_rank_native_fp16_backward_compute_active(&self) -> bool {
        self.cell.low_rank_native_fp16_backward_compute_active()
    }

    pub fn low_rank_native_fp16_parameter_grad_compute_active(&self) -> bool {
        self.cell
            .low_rank_native_fp16_parameter_grad_compute_active()
    }

    pub fn low_rank_parameter_grad_arithmetic(&self) -> RwkvLowRankParameterGradArithmetic {
        self.cell.low_rank_parameter_grad_arithmetic()
    }

    pub fn projection_native_fp16_backward_compute_active(&self) -> bool {
        self.cell.projection_native_fp16_backward_compute_active()
    }

    pub(crate) fn low_rank_fp16_full_forward_first_stage_arm_label(&self) -> Option<&'static str> {
        self.cell.low_rank_fp16_full_forward_first_stage_arm_label()
    }

    pub(crate) fn install_low_rank_fp16_parameter_mirrors(
        &mut self,
        mirrors: RwkvLowRankFp16ParameterMirrors,
        local_optimizer_bindings: &[RwkvParameterStorageMirrorBinding],
    ) -> Result<()> {
        let Self {
            cell, optimizer, ..
        } = self;
        let trainables = cell.trainables()?;
        optimizer.attach_parameter_storage_mirrors(
            &trainables,
            VulkanParameterStorageFormat::Fp16,
            local_optimizer_bindings,
        )?;
        cell.install_low_rank_fp16_parameter_mirrors(mirrors)
    }

    pub(crate) fn enable_low_rank_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.cell.enable_low_rank_native_fp16_backward_compute()
    }

    pub(crate) fn enable_low_rank_native_fp16_parameter_grad_compute(
        &mut self,
        widen_product: bool,
        compensated_operands: bool,
    ) -> Result<()> {
        self.cell
            .enable_low_rank_native_fp16_parameter_grad_compute(widen_product, compensated_operands)
    }

    pub(crate) fn enable_projection_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.cell.enable_projection_native_fp16_backward_compute()
    }

    pub(crate) fn configure_backward_source_scale(
        &mut self,
        source_scale: f32,
        source_scaled_backward_domain: bool,
    ) -> Result<()> {
        self.cell
            .configure_backward_source_scale(source_scale, source_scaled_backward_domain)
    }

    pub(crate) fn optimizer_trainables(&self) -> Result<Vec<RwkvTrainableRef<'_>>> {
        self.cell.trainables()
    }

    /// Begin an opt-in trace of one per-use cell scratch gradient. Every later
    /// full-model accumulation copies that scratch tensor into a fresh
    /// device-local snapshot before the cell can reuse it. A transfer into a
    /// host-visible readback buffer is also recorded in the same command stream,
    /// so observing the trace never requires a synchronization between recurrent
    /// backward uses.
    pub(crate) fn begin_current_gradient_trace(&self, trainable_name: &str) -> Result<()> {
        let trainables = self.cell.trainables()?;
        let trainable = trainables
            .iter()
            .find(|trainable| trainable.name == trainable_name)
            .with_context(|| {
                format!("unknown recurrent gradient trace tensor {trainable_name:?}")
            })?;
        *self.current_gradient_trace.borrow_mut() = Some(RwkvCurrentGradientTrace {
            trainable_name: trainable_name.to_string(),
            len: trainable.len,
            device_snapshots: Vec::new(),
            readbacks: Vec::new(),
        });
        Ok(())
    }

    /// Consume a completed trace after the owning compute batch has submitted.
    /// The outer vector is chronological command-recording order: entry zero is
    /// the first recurrent backward use that reached the full-model accumulator.
    pub(crate) fn take_current_gradient_trace(&self) -> Result<Vec<Vec<f32>>> {
        let trace = self
            .current_gradient_trace
            .borrow_mut()
            .take()
            .context("recurrent gradient trace was not started")?;
        trace
            .readbacks
            .iter()
            .map(|readback| readback.read_f32(trace.len))
            .collect()
    }

    fn record_current_gradient_trace(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
    ) -> Result<()> {
        let mut trace_ref = self.current_gradient_trace.borrow_mut();
        let Some(trace) = trace_ref.as_mut() else {
            return Ok(());
        };
        let trainable = trainables
            .iter()
            .find(|trainable| trainable.name == trace.trainable_name)
            .with_context(|| {
                format!(
                    "recurrent gradient trace tensor {:?} disappeared from the cell registry",
                    trace.trainable_name
                )
            })?;
        if trainable.len != trace.len {
            bail!(
                "recurrent gradient trace tensor {:?} changed length from {} to {}",
                trace.trainable_name,
                trace.len,
                trainable.len
            );
        }

        let snapshot = GpuBuffer::zeros_f32(&self.device, trace.len)?;
        let readback = GpuBuffer::zeros_host_f32(&self.device, trace.len)?;
        commands.copy_f32(trainable.gradient, &snapshot, trace.len)?;
        commands.readback_f32(&snapshot, &readback, trace.len)?;
        trace.device_snapshots.push(snapshot);
        trace.readbacks.push(readback);
        Ok(())
    }

    /// Mirror the current per-timestep cell scratch gradients into a caller-
    /// owned optimizer registry using canonical model tensor names. The cell
    /// backward kernels overwrite these scratch gradients at the next reverse
    /// timestep, so a full-model optimizer must consume them immediately.
    pub(crate) fn record_accumulate_current_gradients_into(
        &self,
        commands: &mut vulkan::ComputeBatch,
        optimizer: &RwkvPersistentAdamW,
        recurrent_prefix: &str,
        deepembed_prefix: &str,
    ) -> Result<()> {
        let trainables = self.cell.trainables()?;
        self.record_current_gradient_trace(commands, &trainables)?;
        let canonical_names = trainables
            .iter()
            .map(|trainable| {
                if let Some(suffix) = trainable.name.strip_prefix("deepembed.") {
                    format!("{deepembed_prefix}.{suffix}")
                } else {
                    format!("{recurrent_prefix}.{}", trainable.name)
                }
            })
            .collect::<Vec<_>>();
        let named = canonical_names
            .iter()
            .map(String::as_str)
            .zip(trainables)
            .collect::<Vec<_>>();
        optimizer.record_accumulate_many_named(commands, &named)
    }

    pub(crate) fn parameter_snapshots(&self) -> Result<Vec<RwkvParameterSnapshot>> {
        self.cell.parameter_snapshots()
    }

    pub(crate) fn finalize_recorded_train_step_with_external_optimizer(
        &self,
        recorded: RwkvTbpttRecordedSequence,
        optimizer: RwkvOptimizerStepResult,
        parameters: Vec<RwkvParameterSnapshot>,
    ) -> Result<RwkvTbpttTrainStepResult> {
        let (sequence, _, _, _) = self.finalize_recorded(recorded)?;
        Ok(RwkvTbpttTrainStepResult {
            sequence,
            optimizer,
            parameters,
            tied_embedding_optimizer: None,
        })
    }

    pub fn shared_lm_head(&self) -> Option<SharedLmHeadParameter> {
        self.tied_embedding
            .as_ref()
            .map(TiedTokenEmbeddingOp::shared_parameter)
    }

    pub fn export_model_package(
        &self,
        source_model_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        cell_prefix: &str,
        deepembed_adapter_prefix: &str,
    ) -> Result<()> {
        let source_model_dir = source_model_dir.as_ref();
        let output_dir = output_dir.as_ref();
        self.cell.export_model_package(
            source_model_dir,
            output_dir,
            cell_prefix,
            deepembed_adapter_prefix,
        )?;
        if let Some(embedding) = self.tied_embedding.as_ref() {
            let weights = embedding.weights()?;
            let checkpoint = output_dir.join("model.safetensors");
            crate::replace_f32_tensor(
                &checkpoint,
                &checkpoint,
                "lm_head.weight",
                &[embedding.vocab_size(), embedding.dim()],
                &weights,
            )?;
        }
        Ok(())
    }
}

fn device_buffers(device: &VulkanDevice, count: usize, len: usize) -> Result<Vec<GpuBuffer>> {
    (0..count)
        .map(|_| GpuBuffer::zeros_f32(device, len))
        .collect()
}

fn u32_device_buffers(device: &VulkanDevice, count: usize, len: usize) -> Result<Vec<GpuBuffer>> {
    (0..count)
        .map(|_| GpuBuffer::zeros_u32(device, len))
        .collect()
}

fn host_buffers(device: &VulkanDevice, count: usize, len: usize) -> Result<Vec<GpuBuffer>> {
    (0..count)
        .map(|_| GpuBuffer::zeros_host_f32(device, len))
        .collect()
}

fn validate_sequence(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "TBPTT {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("TBPTT {name} contains non-finite values");
    }
    Ok(())
}

fn validate_token_sequence(
    name: &str,
    values: &[u32],
    expected: usize,
    vocab_size: usize,
) -> Result<()> {
    if values.len() != expected {
        bail!(
            "TBPTT {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if let Some(&bad) = values.iter().find(|&&token| token as usize >= vocab_size) {
        bail!("TBPTT {name} contains token {bad} outside vocabulary size {vocab_size}");
    }
    Ok(())
}
