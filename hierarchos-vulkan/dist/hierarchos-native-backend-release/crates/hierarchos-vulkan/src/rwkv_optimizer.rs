use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::mixed_precision::{VulkanParameterStorageMirror, VulkanParameterStorageMirrorRefresher};
use crate::training_numerics::{VulkanGradientL2NormReducer, VulkanGradientNonfiniteDetector};
use crate::VulkanParameterStorageFormat;
use crate::{vulkan, AdamWHyperParams, GpuBuffer, VulkanDevice};

const GRADIENT_ACCUMULATE_SPV: &[u8] = include_bytes!("../shaders/gradient_accumulate.spv");
const GRADIENT_ACCUMULATE4_SPV: &[u8] = include_bytes!("../shaders/gradient_accumulate4.spv");
const GRADIENT_SCALE_SPV: &[u8] = include_bytes!("../shaders/gradient_scale.spv");
const GRADIENT_SCALE_FROM_BUFFER_SPV: &[u8] =
    include_bytes!("../shaders/gradient_scale_from_buffer.spv");
const GRADIENT_SCALE_FROM_BUFFER_INDEXED_SPV: &[u8] =
    include_bytes!("../shaders/gradient_scale_from_buffer_indexed.spv");
const ADAMW_SPV: &[u8] = include_bytes!("../shaders/adamw.spv");
const ADAMW_RANGE_SPV: &[u8] = include_bytes!("../shaders/adamw_range.spv");
const ADAMW_RANGE_CONTROLLED_SPV: &[u8] = include_bytes!("../shaders/adamw_range_controlled.spv");
const ADAMW_RANGE_GRAD_SCALER_CONTROLLED_SPV: &[u8] =
    include_bytes!("../shaders/adamw_range_grad_scaler_controlled.spv");
const ADAMW_STEP_GRAD_SCALER_CONTROLLED_SPV: &[u8] =
    include_bytes!("../shaders/adamw_step_grad_scaler_controlled.spv");
const ADAMW_FP16_MIRROR_SPV: &[u8] = include_bytes!("../shaders/adamw_fp16_mirror.spv");
const ADAMW_CONTROLLED_SPV: &[u8] = include_bytes!("../shaders/adamw_controlled.spv");
const ADAMW_FP16_MIRROR_CONTROLLED_SPV: &[u8] =
    include_bytes!("../shaders/adamw_fp16_mirror_controlled.spv");

/// Amortize queue-submit/host-wakeup overhead while retaining chunk-level
/// retirement safety. Broadcast workers run independently of the optimizer, so
/// waiting for a short consecutive run cannot deadlock transport; it only lets
/// a few more already-in-flight ranges retire before one command buffer mutates
/// all of them. The final partial run is always accepted.
const ADAMW_WAVEFRONT_MIN_COALESCED_RANGES: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdamWDecayClass {
    Decay,
    NoDecay,
}

impl AdamWDecayClass {
    pub const fn checkpoint_label(self) -> &'static str {
        match self {
            Self::Decay => "decay",
            Self::NoDecay => "no-decay",
        }
    }

    pub(crate) fn from_checkpoint_label(label: &str) -> Result<Self> {
        match label {
            "decay" => Ok(Self::Decay),
            "no-decay" => Ok(Self::NoDecay),
            other => bail!("unsupported AdamW decay class {other:?}"),
        }
    }
}

pub(crate) type RwkvDecayClass = AdamWDecayClass;

#[derive(Clone, Copy)]
pub(crate) struct RwkvTrainableRef<'a> {
    pub name: &'a str,
    pub parameter: &'a GpuBuffer,
    pub gradient: &'a GpuBuffer,
    pub len: usize,
    pub decay_class: RwkvDecayClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RwkvParameterSnapshot {
    pub name: String,
    pub values: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct RwkvOptimizerStepResult {
    pub step: u32,
    pub tensor_count: usize,
}

#[derive(Clone, Debug)]
pub struct AdamWOptimizerSlotState {
    pub name: String,
    /// PyTorch AdamW keeps an independent step tensor per parameter and does
    /// not advance it when `grad is None`. This is therefore deliberately
    /// separate from [`AdamWOptimizerState::step`], which remains the outer
    /// optimizer-step counter used by the training scheduler/checkpoint UI.
    pub step: u32,
    /// Portable optimizer-group topology. Legacy v1/v2 checkpoints did not
    /// record this field and therefore restore as `None`; current snapshots
    /// always carry the live registry's decay class so another backend cannot
    /// silently move a no-decay tensor into a decayed parameter group.
    pub decay_class: Option<AdamWDecayClass>,
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Vec<f32>,
}

/// Portable optimizer state at a completed training-step boundary. Parameter
/// values remain in the ordinary model SafeTensors package; this companion
/// state contains only AdamW's global step and first/second moments, keyed by
/// the exact model tensor names used by the Vulkan registry.
#[derive(Clone, Debug)]
pub struct AdamWOptimizerState {
    pub step: u32,
    pub slots: Vec<AdamWOptimizerSlotState>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradientScalePush {
    len: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradientScaleFromBufferIndexedPush {
    len: u32,
    scale_index: u32,
    multiplier: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradientAccumulate4Push {
    len0: u32,
    len1: u32,
    len2: u32,
    len3: u32,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWRangePush {
    offset: u32,
    len: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWRangeControlledPush {
    offset: u32,
    len: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    is_active: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWRangeGradScalerControlledPush {
    offset: u32,
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    gradient_scale: f32,
    is_active: u32,
    apply_control_unscale: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWControlledPush {
    len: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    gradient_scale: f32,
    active: u32,
    apply_control_unscale: u32,
}

pub(crate) struct RwkvDeviceControlledStepPending {
    /// Exact host metadata at record time when the Rust mirror was current.
    /// Queue-resident GradScaler wavefronts may deliberately leave these empty
    /// after a previous step was deferred; their device counters remain exact.
    previous_step: Option<u32>,
    next_step: Option<u32>,
    active_slot_indices: Vec<usize>,
    optimizer_generation: u64,
    /// Range wavefronts can publish the next hazard generation as soon as their
    /// tail submission has a timeline completion token. That publication is
    /// independent of the numerical step/skip decision: on a skipped AMP
    /// window the new generation simply names identical parameter/moment data
    /// after the device-side gradient clear has completed.
    generation_committed: bool,
    /// True when the Vulkan-resident global and per-slot Adam clocks were
    /// predicated by the same control decision as parameter mutation. Other
    /// controlled paths still use host step push constants and invalidate the
    /// lazy device clock mirror after a successful update.
    device_steps_committed: bool,
    tensor_count: usize,
}

struct OptimizerSlot {
    name: String,
    len: usize,
    decay_class: RwkvDecayClass,
    step: u32,
    device_step: GpuBuffer,
    device_step_authoritative: bool,
    accumulated_grad: GpuBuffer,
    exp_avg: GpuBuffer,
    exp_avg_sq: GpuBuffer,
    parameter_readback: GpuBuffer,
    parameter_storage_mirror: Option<VulkanParameterStorageMirror>,
}

#[derive(Clone)]
struct RwkvReplicaStepSource {
    host_value: u32,
    device_value: Option<GpuBuffer>,
}

impl RwkvReplicaStepSource {
    fn host(host_value: u32) -> Self {
        Self {
            host_value,
            device_value: None,
        }
    }

    fn device(host_value: u32, device_value: &GpuBuffer) -> Self {
        Self {
            host_value,
            device_value: Some(device_value.clone()),
        }
    }

    fn is_device_authoritative(&self) -> bool {
        self.device_value.is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RwkvReplicaStatePlane {
    Parameter,
    ExpAvg,
    ExpAvgSq,
}

#[derive(Clone)]
struct RwkvReplicaStateSourceSlot {
    name: String,
    len: usize,
    step: RwkvReplicaStepSource,
    decay_class: AdamWDecayClass,
    parameter: GpuBuffer,
    exp_avg: GpuBuffer,
    exp_avg_sq: GpuBuffer,
}

#[derive(Clone)]
struct RwkvPendingGradientSourceSlot {
    name: String,
    len: usize,
    gradient: GpuBuffer,
}

/// Immutable Vulkan-resident view of one open optimizer gradient registry.
///
/// Each slot clones the exact canonical gradient buffer selected at capture
/// time, including the tied LM-head override when that topology is active. The
/// source therefore carries no borrow of the parent training graph and can be
/// handed to transport workers without making TBPTT tracing state `Sync`.
#[derive(Clone)]
pub(crate) struct RwkvPendingGradientSource {
    slots: Vec<RwkvPendingGradientSourceSlot>,
}

impl RwkvPendingGradientSource {
    pub(crate) fn gradient_layout(&self) -> Vec<(String, usize)> {
        self.slots
            .iter()
            .map(|slot| (slot.name.clone(), slot.len))
            .collect()
    }

    pub(crate) fn record_gradient_range_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        tensor_index: usize,
        offset: usize,
        len: usize,
        readback: &GpuBuffer,
    ) -> Result<()> {
        let slot = self.slots.get(tensor_index).with_context(|| {
            format!("pending gradient source tensor index {tensor_index} is out of range")
        })?;
        let end = offset
            .checked_add(len)
            .context("pending gradient source read range overflow")?;
        if len == 0 || end > slot.len {
            bail!(
                "pending gradient source read range {offset}..{end} is outside tensor {:?} length {}",
                slot.name,
                slot.len
            );
        }
        if len > readback.f32_capacity() {
            bail!(
                "pending gradient source chunk length {len} exceeds reusable readback capacity {}",
                readback.f32_capacity()
            );
        }
        commands.readback_f32_range(&slot.gradient, offset, readback, 0, len)?;
        Ok(())
    }

    pub(crate) fn record_gradient_range_copy(
        &self,
        commands: &mut vulkan::ComputeBatch,
        tensor_index: usize,
        offset: usize,
        len: usize,
        destination: &GpuBuffer,
    ) -> Result<()> {
        let slot = self.slots.get(tensor_index).with_context(|| {
            format!("pending gradient source tensor index {tensor_index} is out of range")
        })?;
        let end = offset
            .checked_add(len)
            .context("pending gradient source copy range overflow")?;
        if len == 0 || end > slot.len {
            bail!(
                "pending gradient source copy range {offset}..{end} is outside tensor {:?} length {}",
                slot.name,
                slot.len
            );
        }
        if len > destination.f32_capacity() {
            bail!(
                "pending gradient source chunk length {len} exceeds destination capacity {}",
                destination.f32_capacity()
            );
        }
        commands.copy_f32_range(&slot.gradient, offset, destination, 0, len)
    }
}

#[derive(Clone)]
struct OptimizerGenerationPredeclaredTimelineConsumer {
    first_wait: vulkan::DeviceGroupTimelineWait,
    range_count: usize,
}

/// Immutable queueing plan for a generation whose remaining retirement
/// dependencies are already expressible as Vulkan timeline values. The common
/// device-group path stores only one contiguous timeline span per replica; an
/// optional per-range snapshot exists solely for mixed/legacy transports that
/// had already published individual GPU waits before the plan was frozen.
struct OptimizerGenerationPredeclaredGpuSchedule {
    range_limit: usize,
    run_ranges: usize,
    range_gpu_waits: Option<Vec<Vec<vulkan::DeviceGroupTimelineWait>>>,
    timeline_consumers: Vec<OptimizerGenerationPredeclaredTimelineConsumer>,
}

impl OptimizerGenerationPredeclaredGpuSchedule {
    fn range_run(
        &self,
        first_range: usize,
    ) -> Result<(usize, Vec<vulkan::DeviceGroupTimelineWait>)> {
        if first_range >= self.range_limit {
            bail!(
                "optimizer generation predeclared GPU schedule start {first_range} is outside range limit {}",
                self.range_limit
            );
        }
        let ready_end = first_range
            .checked_add(self.run_ranges)
            .context("optimizer generation predeclared GPU schedule end overflow")?
            .min(self.range_limit);
        let mut waits = if let Some(range_gpu_waits) = self.range_gpu_waits.as_ref() {
            range_gpu_waits[first_range..ready_end]
                .iter()
                .flat_map(|waits| waits.iter().cloned())
                .collect::<Vec<_>>()
        } else {
            Vec::with_capacity(self.timeline_consumers.len())
        };
        let final_range = ready_end - 1;
        for consumer in &self.timeline_consumers {
            if ready_end > consumer.range_count {
                bail!(
                    "optimizer generation GPU timeline span covers {} ranges; scheduled run ends at {ready_end}",
                    consumer.range_count
                );
            }
            waits.push(consumer.first_wait.advanced_by(final_range)?);
        }
        Ok((ready_end, waits))
    }

    #[cfg(test)]
    fn run_count(&self) -> usize {
        self.range_limit.div_ceil(self.run_ranges)
    }
}

#[derive(Default)]
struct OptimizerGenerationState {
    generation: u64,
    readers: usize,
    range_readers: Vec<usize>,
    range_gpu_waits: Vec<Vec<vulkan::DeviceGroupTimelineWait>>,
    range_gpu_waits_predeclared: Vec<bool>,
    predeclared_timeline_consumers: Vec<OptimizerGenerationPredeclaredTimelineConsumer>,
    ready_after: Option<vulkan::SubmissionTimelineWait>,
}

#[derive(Default)]
struct OptimizerGenerationGuard {
    state: Mutex<OptimizerGenerationState>,
    readers_retired: Condvar,
}

impl OptimizerGenerationGuard {
    fn lock_state(&self) -> MutexGuard<'_, OptimizerGenerationState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn acquire_read_lease(self: &Arc<Self>) -> Result<Arc<OptimizerGenerationReadLease>> {
        let mut state = self.lock_state();
        state.readers = state
            .readers
            .checked_add(1)
            .context("optimizer generation reader count overflow")?;
        let generation = state.generation;
        let ready_after = state.ready_after.clone();
        drop(state);
        Ok(Arc::new(OptimizerGenerationReadLease {
            guard: Arc::clone(self),
            generation,
            ready_after,
            mode: Mutex::new(OptimizerGenerationReadLeaseMode::Full),
        }))
    }

    fn split_read_lease_into_ranges(
        self: &Arc<Self>,
        expected_generation: u64,
        range_count: usize,
        consumer_count: usize,
    ) -> Result<Vec<OptimizerGenerationRangeConsumer>> {
        if range_count == 0 {
            bail!("optimizer generation range lease requires at least one range");
        }
        if consumer_count == 0 {
            bail!("optimizer generation range lease requires at least one consumer");
        }
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before range-lease split: expected {expected_generation}, found {}",
                state.generation
            );
        }
        if state.readers == 0 {
            bail!("optimizer generation read lease disappeared before range-lease split");
        }
        if !state.range_readers.is_empty() {
            bail!(
                "optimizer generation {expected_generation} already has an active range-retirement plan"
            );
        }
        state.readers -= 1;
        state.range_readers = vec![consumer_count; range_count];
        state.range_gpu_waits = (0..range_count).map(|_| Vec::new()).collect();
        state.range_gpu_waits_predeclared = vec![false; range_count];
        state.predeclared_timeline_consumers.clear();
        drop(state);
        self.readers_retired.notify_all();
        Ok((0..consumer_count)
            .map(|_| OptimizerGenerationRangeConsumer {
                guard: Arc::clone(self),
                generation: expected_generation,
                range_count,
                retired: None,
                predeclared_all: false,
            })
            .collect())
    }

    /// Convert one range consumer into a compact future GPU retirement span.
    /// `range_readers` deliberately keeps the original consumer count; the
    /// number of compact spans is its retirement floor. This avoids touching
    /// every range merely to publish a dependency whose values are already a
    /// contiguous monotonic timeline.
    fn predeclare_device_group_timeline_span(
        &self,
        expected_generation: u64,
        range_count: usize,
        first_wait: vulkan::DeviceGroupTimelineWait,
    ) -> Result<()> {
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before GPU timeline-span predeclaration: expected {expected_generation}, found {}",
                state.generation
            );
        }
        if range_count == 0 || range_count != state.range_readers.len() {
            bail!(
                "optimizer generation GPU timeline span covers {range_count} ranges; active plan has {}",
                state.range_readers.len()
            );
        }
        if state.range_gpu_waits.len() != range_count
            || state.range_gpu_waits_predeclared.len() != range_count
        {
            bail!("optimizer generation GPU retirement plan is internally inconsistent");
        }
        let retirement_floor = state.predeclared_timeline_consumers.len();
        if state
            .range_readers
            .iter()
            .any(|readers| *readers <= retirement_floor)
        {
            bail!(
                "optimizer generation has no unclaimed range consumer for GPU timeline-span predeclaration"
            );
        }
        state
            .predeclared_timeline_consumers
            .push(OptimizerGenerationPredeclaredTimelineConsumer {
                first_wait,
                range_count,
            });
        drop(state);
        self.readers_retired.notify_all();
        Ok(())
    }

    fn retirement_floor(state: &OptimizerGenerationState) -> usize {
        state.predeclared_timeline_consumers.len()
    }

    fn remaining_cpu_range_readers(state: &OptimizerGenerationState) -> Result<usize> {
        let retirement_floor = Self::retirement_floor(state);
        state.range_readers.iter().try_fold(0usize, |total, readers| {
            let remaining = readers.checked_sub(retirement_floor).context(
                "optimizer generation range reader count fell below its predeclared GPU retirement floor",
            )?;
            total
                .checked_add(remaining)
                .context("optimizer generation remaining range-reader count overflow")
        })
    }

    fn collect_gpu_waits_for_range_run(
        state: &OptimizerGenerationState,
        first_range: usize,
        ready_end: usize,
    ) -> Result<Vec<vulkan::DeviceGroupTimelineWait>> {
        if first_range >= ready_end || ready_end > state.range_readers.len() {
            bail!("optimizer generation GPU wait collection range is invalid");
        }
        let mut waits = state.range_gpu_waits[first_range..ready_end]
            .iter()
            .flat_map(|waits| waits.iter().cloned())
            .collect::<Vec<_>>();
        let final_range = ready_end - 1;
        for consumer in &state.predeclared_timeline_consumers {
            if ready_end > consumer.range_count {
                bail!(
                    "optimizer generation GPU timeline span covers {} ranges; wait requested through {ready_end}",
                    consumer.range_count
                );
            }
            waits.push(consumer.first_wait.advanced_by(final_range)?);
        }
        Ok(waits)
    }

    fn retire_range(&self, expected_generation: u64, range_index: usize) -> Result<()> {
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before range retirement: expected {expected_generation}, found {}",
                state.generation
            );
        }
        let retirement_floor = Self::retirement_floor(&state);
        let readers = state.range_readers.get_mut(range_index).with_context(|| {
            format!(
                "optimizer generation range {range_index} is outside the active retirement plan"
            )
        })?;
        if *readers <= retirement_floor {
            bail!("optimizer generation range {range_index} was retired more than once");
        }
        *readers -= 1;
        drop(state);
        self.readers_retired.notify_all();
        Ok(())
    }

    fn retire_range_after_device_group_timeline(
        &self,
        expected_generation: u64,
        range_index: usize,
        wait: vulkan::DeviceGroupTimelineWait,
    ) -> Result<()> {
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before GPU range retirement: expected {expected_generation}, found {}",
                state.generation
            );
        }
        if state.range_gpu_waits.len() != state.range_readers.len()
            || state.range_gpu_waits_predeclared.len() != state.range_readers.len()
        {
            bail!("optimizer generation GPU retirement plan is internally inconsistent");
        }
        let readers = state
            .range_readers
            .get(range_index)
            .copied()
            .with_context(|| {
                format!(
                "optimizer generation GPU range {range_index} is outside the active retirement plan"
            )
            })?;
        let retirement_floor = Self::retirement_floor(&state);
        if readers <= retirement_floor {
            bail!("optimizer generation range {range_index} was retired more than once");
        }
        state.range_readers[range_index] -= 1;
        state.range_gpu_waits[range_index].push(wait);
        drop(state);
        // Wake the fallback path too. Device-group-only training normally
        // consumes the published timeline values through the non-blocking fast
        // path below, while opaque/host peers retain the Condvar contract.
        self.readers_retired.notify_all();
        Ok(())
    }

    #[cfg(test)]
    fn wait_for_range_to_retire(&self, expected_generation: u64, range_index: usize) -> Result<()> {
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before range mutation guard acquisition: expected {expected_generation}, found {}",
                state.generation
            );
        }
        if range_index >= state.range_readers.len() {
            // No split plan exists for this generation. This is the ordinary
            // single-device/legacy path. A coarse source may still be alive,
            // so preserve the old whole-generation safety contract.
            if state.range_readers.is_empty() {
                while state.readers != 0 {
                    state = self
                        .readers_retired
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if state.generation != expected_generation {
                        bail!(
                            "optimizer generation changed while waiting for legacy readers: expected {expected_generation}, found {}",
                            state.generation
                        );
                    }
                }
                return Ok(());
            }
            bail!(
                "optimizer generation range {range_index} is outside active plan of {} ranges",
                state.range_readers.len()
            );
        }
        while state.readers != 0
            || state.range_readers[range_index] != Self::retirement_floor(&state)
        {
            state = self
                .readers_retired
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.generation != expected_generation {
                bail!(
                    "optimizer generation changed while waiting for range {range_index}: expected {expected_generation}, found {}",
                    state.generation
                );
            }
        }
        Ok(())
    }

    /// Wait until at least `min_run_ranges` consecutive ranges starting at
    /// `first_range` are safe to mutate (or the final partial run is ready),
    /// then return the exclusive end of the largest consecutive run already
    /// retired. Range reader counts only move toward zero, so every range in
    /// the returned run remains safe after the mutex is released.
    fn wait_for_ready_range_run(
        &self,
        expected_generation: u64,
        first_range: usize,
        range_limit: usize,
        min_run_ranges: usize,
    ) -> Result<usize> {
        if first_range >= range_limit {
            bail!(
                "optimizer generation ready-run start {first_range} is outside range limit {range_limit}"
            );
        }
        if min_run_ranges == 0 {
            bail!("optimizer generation ready-run coalescing size must be positive");
        }
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before ready-run acquisition: expected {expected_generation}, found {}",
                state.generation
            );
        }

        if state.range_readers.is_empty() {
            // Legacy/coarse generation readers have no per-range visibility.
            // Once they are gone, every range in this optimizer step is safe
            // and can be collapsed into one queue submission.
            while state.readers != 0 {
                state = self
                    .readers_retired
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if state.generation != expected_generation {
                    bail!(
                        "optimizer generation changed while waiting for legacy readers: expected {expected_generation}, found {}",
                        state.generation
                    );
                }
            }
            return Ok(range_limit);
        }

        if range_limit > state.range_readers.len() {
            bail!(
                "optimizer generation ready-run limit {range_limit} exceeds active plan of {} ranges",
                state.range_readers.len()
            );
        }
        let required_end = first_range
            .checked_add(min_run_ranges)
            .context("optimizer generation ready-run coalescing end overflow")?
            .min(range_limit);
        while state.readers != 0
            || state.range_readers[first_range..required_end]
                .iter()
                .any(|readers| *readers != Self::retirement_floor(&state))
        {
            state = self
                .readers_retired
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.generation != expected_generation {
                bail!(
                    "optimizer generation changed while waiting for coalesced ranges {first_range}..{required_end}: expected {expected_generation}, found {}",
                    state.generation
                );
            }
        }

        let bounded_by_predeclared_timeline = !state.predeclared_timeline_consumers.is_empty()
            || state.range_gpu_waits_predeclared[first_range..required_end]
                .iter()
                .any(|predeclared| *predeclared);
        let retirement_floor = Self::retirement_floor(&state);
        let mut ready_end = required_end;
        while !bounded_by_predeclared_timeline
            && ready_end < range_limit
            && state.range_readers[ready_end] == retirement_floor
        {
            ready_end += 1;
        }
        Ok(ready_end)
    }

    /// Return a ready coalesced run without sleeping the host thread. Ranges
    /// retired by device-group transport contribute timeline waits that AdamW
    /// must attach to its queue submission; host/opaque retirements contribute
    /// no GPU wait. `None` means a CPU-published fallback reader is still live.
    fn try_ready_range_run_with_gpu_waits(
        &self,
        expected_generation: u64,
        first_range: usize,
        range_limit: usize,
        min_run_ranges: usize,
    ) -> Result<Option<(usize, Vec<vulkan::DeviceGroupTimelineWait>)>> {
        if first_range >= range_limit {
            bail!(
                "optimizer generation GPU ready-run start {first_range} is outside range limit {range_limit}"
            );
        }
        if min_run_ranges == 0 {
            bail!("optimizer generation GPU ready-run coalescing size must be positive");
        }
        let state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before GPU ready-run acquisition: expected {expected_generation}, found {}",
                state.generation
            );
        }
        if state.range_readers.is_empty() {
            return Ok((state.readers == 0).then(|| (range_limit, Vec::new())));
        }
        if range_limit > state.range_readers.len()
            || state.range_gpu_waits.len() != state.range_readers.len()
            || state.range_gpu_waits_predeclared.len() != state.range_readers.len()
        {
            bail!("optimizer generation GPU ready-run plan is internally inconsistent");
        }
        let required_end = first_range
            .checked_add(min_run_ranges)
            .context("optimizer generation GPU ready-run coalescing end overflow")?
            .min(range_limit);
        let retirement_floor = Self::retirement_floor(&state);
        if state.readers != 0
            || state.range_readers[first_range..required_end]
                .iter()
                .any(|readers| *readers != retirement_floor)
        {
            return Ok(None);
        }
        let mut ready_end = required_end;
        let required_has_predeclared_wait = !state.predeclared_timeline_consumers.is_empty()
            || state.range_gpu_waits_predeclared[first_range..required_end]
                .iter()
                .any(|predeclared| *predeclared);
        while !required_has_predeclared_wait
            && ready_end < range_limit
            && state.range_readers[ready_end] == retirement_floor
            && !state.range_gpu_waits_predeclared[ready_end]
        {
            ready_end += 1;
        }
        let waits = Self::collect_gpu_waits_for_range_run(&state, first_range, ready_end)?;
        Ok(Some((ready_end, waits)))
    }

    /// Snapshot a complete bounded wavefront schedule when every range in the
    /// generation has already been retired onto a predeclared device-group
    /// timeline. Unlike `try_ready_range_run_with_gpu_waits`, this takes the
    /// guard only once: subsequent AdamW submissions consume immutable future
    /// semaphore values and never consult CPU worker progress again.
    fn predeclared_gpu_range_schedule(
        &self,
        expected_generation: u64,
        range_limit: usize,
        run_ranges: usize,
    ) -> Result<Option<OptimizerGenerationPredeclaredGpuSchedule>> {
        if range_limit == 0 {
            bail!("optimizer generation predeclared GPU schedule requires at least one range");
        }
        if run_ranges == 0 {
            bail!("optimizer generation predeclared GPU schedule run size must be positive");
        }
        let state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before predeclared GPU schedule snapshot: expected {expected_generation}, found {}",
                state.generation
            );
        }
        if state.readers != 0 || state.range_readers.is_empty() {
            return Ok(None);
        }
        if range_limit > state.range_readers.len()
            || state.range_gpu_waits.len() != state.range_readers.len()
            || state.range_gpu_waits_predeclared.len() != state.range_readers.len()
        {
            bail!("optimizer generation predeclared GPU schedule is internally inconsistent");
        }
        let retirement_floor = Self::retirement_floor(&state);
        if state.range_readers[..range_limit]
            .iter()
            .any(|readers| *readers != retirement_floor)
        {
            return Ok(None);
        }
        if state.predeclared_timeline_consumers.is_empty()
            && state.range_gpu_waits_predeclared[..range_limit]
                .iter()
                .any(|predeclared| !*predeclared)
        {
            return Ok(None);
        }

        let range_gpu_waits = state.range_gpu_waits[..range_limit]
            .iter()
            .any(|waits| !waits.is_empty())
            .then(|| state.range_gpu_waits[..range_limit].to_vec());
        Ok(Some(OptimizerGenerationPredeclaredGpuSchedule {
            range_limit,
            run_ranges,
            range_gpu_waits,
            timeline_consumers: state.predeclared_timeline_consumers.clone(),
        }))
    }

    fn gpu_waits_for_retired_range_run(
        &self,
        expected_generation: u64,
        first_range: usize,
        ready_end: usize,
    ) -> Result<Vec<vulkan::DeviceGroupTimelineWait>> {
        let state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before GPU wait collection: expected {expected_generation}, found {}",
                state.generation
            );
        }
        if state.range_readers.is_empty() {
            if state.readers != 0 {
                bail!("optimizer generation legacy readers are still live after ready-run wait");
            }
            return Ok(Vec::new());
        }
        if ready_end > state.range_readers.len()
            || first_range >= ready_end
            || state.range_gpu_waits.len() != state.range_readers.len()
            || state.range_gpu_waits_predeclared.len() != state.range_readers.len()
        {
            bail!("optimizer generation GPU wait collection range is invalid");
        }
        let retirement_floor = Self::retirement_floor(&state);
        if state.readers != 0
            || state.range_readers[first_range..ready_end]
                .iter()
                .any(|readers| *readers != retirement_floor)
        {
            bail!("optimizer generation range run is not fully retired for GPU wait collection");
        }
        Self::collect_gpu_waits_for_range_run(&state, first_range, ready_end)
    }

    fn current_generation(&self) -> u64 {
        self.lock_state().generation
    }

    fn wait_for_readers_to_retire(&self, expected_generation: u64) -> Result<()> {
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before mutation guard acquisition: expected {expected_generation}, found {}",
                state.generation
            );
        }
        while state.readers != 0
            || state
                .range_readers
                .iter()
                .any(|readers| *readers != Self::retirement_floor(&state))
        {
            state = self
                .readers_retired
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.generation != expected_generation {
                bail!(
                    "optimizer generation changed while waiting for broadcast readers: expected {expected_generation}, found {}",
                    state.generation
                );
            }
        }
        Ok(())
    }

    fn advance_after_mutation(&self, expected_generation: u64) -> Result<u64> {
        self.advance_after_mutation_with_completion(expected_generation, None)
    }

    fn advance_after_mutation_after_submission(
        &self,
        expected_generation: u64,
        completion: vulkan::SubmissionTimelineWait,
    ) -> Result<u64> {
        self.advance_after_mutation_with_completion(expected_generation, Some(completion))
    }

    fn advance_after_mutation_with_completion(
        &self,
        expected_generation: u64,
        ready_after: Option<vulkan::SubmissionTimelineWait>,
    ) -> Result<u64> {
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before mutation commit: expected {expected_generation}, found {}",
                state.generation
            );
        }
        let range_readers = Self::remaining_cpu_range_readers(&state)?;
        if state.readers != 0 || range_readers != 0 {
            bail!(
                "optimizer generation {expected_generation} still has {} full and {range_readers} ranged broadcast readers at mutation commit",
                state.readers,
            );
        }
        state.range_readers.clear();
        state.range_gpu_waits.clear();
        state.range_gpu_waits_predeclared.clear();
        state.predeclared_timeline_consumers.clear();
        state.generation = state
            .generation
            .checked_add(1)
            .context("optimizer generation overflow")?;
        state.ready_after = ready_after;
        Ok(state.generation)
    }

    fn release_after_skipped_mutation(&self, expected_generation: u64) -> Result<()> {
        let mut state = self.lock_state();
        if state.generation != expected_generation {
            bail!(
                "optimizer generation changed before skipped mutation release: expected {expected_generation}, found {}",
                state.generation
            );
        }
        let range_readers = Self::remaining_cpu_range_readers(&state)?;
        if state.readers != 0 || range_readers != 0 {
            bail!(
                "optimizer generation {expected_generation} still has {} full and {range_readers} ranged broadcast readers at skipped mutation release",
                state.readers,
            );
        }
        state.range_readers.clear();
        state.range_gpu_waits.clear();
        state.range_gpu_waits_predeclared.clear();
        state.predeclared_timeline_consumers.clear();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OptimizerGenerationReadLeaseMode {
    Full,
    Ranged,
}

struct OptimizerGenerationReadLease {
    guard: Arc<OptimizerGenerationGuard>,
    generation: u64,
    ready_after: Option<vulkan::SubmissionTimelineWait>,
    mode: Mutex<OptimizerGenerationReadLeaseMode>,
}

impl OptimizerGenerationReadLease {
    fn ready_after(&self) -> Option<vulkan::SubmissionTimelineWait> {
        self.ready_after.clone()
    }

    fn split_into_range_consumers(
        &self,
        range_count: usize,
        consumer_count: usize,
    ) -> Result<Vec<OptimizerGenerationRangeConsumer>> {
        let mut mode = self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *mode != OptimizerGenerationReadLeaseMode::Full {
            bail!("optimizer generation read lease has already been split into ranges");
        }
        let consumers = self.guard.split_read_lease_into_ranges(
            self.generation,
            range_count,
            consumer_count,
        )?;
        *mode = OptimizerGenerationReadLeaseMode::Ranged;
        Ok(consumers)
    }
}

impl Drop for OptimizerGenerationReadLease {
    fn drop(&mut self) {
        let mode = *self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if mode == OptimizerGenerationReadLeaseMode::Ranged {
            return;
        }
        let mut state = self.guard.lock_state();
        debug_assert_eq!(state.generation, self.generation);
        debug_assert!(state.readers > 0);
        state.readers = state.readers.saturating_sub(1);
        if state.readers == 0 {
            self.guard.readers_retired.notify_all();
        }
    }
}

pub(crate) struct OptimizerGenerationRangeConsumer {
    guard: Arc<OptimizerGenerationGuard>,
    generation: u64,
    range_count: usize,
    /// Allocated only for CPU-published or per-range GPU retirements. The
    /// predeclared device-group fast path publishes one monotonic timeline span
    /// and therefore never needs an O(range_count) consumer bitmap.
    retired: Option<Vec<bool>>,
    predeclared_all: bool,
}

impl OptimizerGenerationRangeConsumer {
    pub(crate) fn range_count(&self) -> usize {
        self.range_count
    }

    pub(crate) fn retire(&mut self, range_index: usize) -> Result<()> {
        if self.predeclared_all {
            bail!("optimizer range consumer is already covered by a predeclared GPU timeline");
        }
        if range_index >= self.range_count {
            bail!(
                "optimizer range consumer received out-of-range retirement {range_index}/{}",
                self.range_count
            );
        }
        if self
            .retired
            .as_ref()
            .is_some_and(|retired| retired[range_index])
        {
            bail!("optimizer range consumer retired range {range_index} twice");
        }
        self.guard.retire_range(self.generation, range_index)?;
        self.retired
            .get_or_insert_with(|| vec![false; self.range_count])[range_index] = true;
        Ok(())
    }

    pub(crate) fn retire_after_device_group_timeline(
        &mut self,
        range_index: usize,
        wait: vulkan::DeviceGroupTimelineWait,
    ) -> Result<()> {
        if self.predeclared_all {
            bail!("optimizer range consumer is already covered by a predeclared GPU timeline");
        }
        if range_index >= self.range_count {
            bail!(
                "optimizer range consumer received out-of-range GPU retirement {range_index}/{}",
                self.range_count
            );
        }
        if self
            .retired
            .as_ref()
            .is_some_and(|retired| retired[range_index])
        {
            bail!("optimizer range consumer retired range {range_index} twice");
        }
        self.guard
            .retire_range_after_device_group_timeline(self.generation, range_index, wait)?;
        self.retired
            .get_or_insert_with(|| vec![false; self.range_count])[range_index] = true;
        Ok(())
    }

    pub(crate) fn predeclare_device_group_timeline_span(
        &mut self,
        first_wait: vulkan::DeviceGroupTimelineWait,
        range_count: usize,
    ) -> Result<()> {
        if range_count != self.range_count {
            bail!(
                "optimizer range consumer has {} ranges; GPU timeline span covers {range_count}",
                self.range_count
            );
        }
        if self.predeclared_all {
            bail!("optimizer range consumer GPU timeline span was predeclared twice");
        }
        if self
            .retired
            .as_ref()
            .is_some_and(|retired| retired.iter().any(|retired| *retired))
        {
            bail!("cannot predeclare a GPU timeline span after individual range retirement began");
        }
        self.guard.predeclare_device_group_timeline_span(
            self.generation,
            range_count,
            first_wait,
        )?;
        self.predeclared_all = true;
        self.retired = None;
        Ok(())
    }

    pub(crate) fn retire_all(&mut self) -> Result<()> {
        if self.predeclared_all {
            return Ok(());
        }
        for range_index in 0..self.range_count {
            if !self
                .retired
                .as_ref()
                .is_some_and(|retired| retired[range_index])
            {
                self.retire(range_index)?;
            }
        }
        Ok(())
    }
}

impl Drop for OptimizerGenerationRangeConsumer {
    fn drop(&mut self) {
        // A worker that abandons a broadcast will perform no further reads.
        // Releasing its outstanding ranges here prevents a failed replica from
        // permanently wedging the canonical optimizer generation.
        let _ = self.retire_all();
    }
}

/// Immutable, transport-only view of one closed AdamW boundary.
///
/// Every Vulkan buffer is an `Arc`-backed clone of canonical live storage, so
/// this object can be shared across replica transfer threads without borrowing
/// the optimizer (or the non-`Sync` TBPTT graph that owns it). The source is
/// The source owns a generation read lease. Forward/backward may continue to
/// read the same canonical parameters while that lease is live, but AdamW
/// parameter/moment mutation blocks at the generation boundary until the final
/// source clone retires. This makes transport lifetime independent of the
/// non-`Sync` training graph and permits asynchronous replica fanout.
#[derive(Clone)]
pub(crate) struct RwkvReplicaStateSource {
    device: VulkanDevice,
    step: RwkvReplicaStepSource,
    slots: Vec<RwkvReplicaStateSourceSlot>,
    _read_lease: Arc<OptimizerGenerationReadLease>,
}

impl RwkvReplicaStateSource {
    pub(crate) fn gradient_layout(&self) -> Vec<(String, usize)> {
        self.slots
            .iter()
            .map(|slot| (slot.name.clone(), slot.len))
            .collect()
    }

    pub(crate) fn step_metadata_word_count(&self) -> usize {
        self.slots.len() + 1
    }

    fn has_device_step_metadata(&self) -> bool {
        self.step.is_device_authoritative()
            || self
                .slots
                .iter()
                .any(|slot| slot.step.is_device_authoritative())
    }

    fn exact_step_metadata(&self) -> Result<(u32, Vec<u32>)> {
        let mut words = Vec::with_capacity(self.step_metadata_word_count());
        words.push(self.step.host_value);
        words.extend(self.slots.iter().map(|slot| slot.step.host_value));
        if !self.has_device_step_metadata() {
            return Ok((words[0], words[1..].to_vec()));
        }

        let readback = GpuBuffer::zeros_host_f32(&self.device, words.len())?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        if let Some(device_step) = self.step.device_value.as_ref() {
            commands.copy_f32_range(device_step, 0, &readback, 0, 1)?;
        }
        for (slot_index, slot) in self.slots.iter().enumerate() {
            if let Some(device_step) = slot.step.device_value.as_ref() {
                commands.copy_f32_range(device_step, 0, &readback, slot_index + 1, 1)?;
            }
        }
        commands.submit()?;
        let device_words = readback
            .read_f32(words.len())?
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        if self.step.is_device_authoritative() {
            words[0] = device_words[0];
        }
        for (slot_index, slot) in self.slots.iter().enumerate() {
            if slot.step.is_device_authoritative() {
                words[slot_index + 1] = device_words[slot_index + 1];
            }
        }
        if words[1..].iter().any(|&slot_step| slot_step > words[0]) {
            bail!("replica source device AdamW metadata contains a per-slot step beyond its global step");
        }
        Ok((words[0], words[1..].to_vec()))
    }

    /// Pack the exact global/per-slot Adam clocks into one peer-visible word
    /// row without resolving device-authoritative counters on the CPU. Host-
    /// authoritative words are staged first; device-owned words overwrite their
    /// positions with raw four-byte Vulkan copies in the same command buffer.
    pub(crate) fn record_step_metadata_pack(
        &self,
        commands: &mut vulkan::ComputeBatch,
        destination: &GpuBuffer,
    ) -> Result<()> {
        let word_count = self.step_metadata_word_count();
        if destination.f32_capacity() < word_count {
            bail!(
                "replica step-metadata transport capacity {} is smaller than required {word_count}",
                destination.f32_capacity()
            );
        }
        let mut host_words = Vec::with_capacity(word_count);
        host_words.push(self.step.host_value);
        host_words.extend(self.slots.iter().map(|slot| slot.step.host_value));
        commands.upload_u32(destination, &host_words)?;
        if let Some(device_step) = self.step.device_value.as_ref() {
            commands.copy_f32_range(device_step, 0, destination, 0, 1)?;
        }
        for (slot_index, slot) in self.slots.iter().enumerate() {
            if let Some(device_step) = slot.step.device_value.as_ref() {
                commands.copy_f32_range(device_step, 0, destination, slot_index + 1, 1)?;
            }
        }
        Ok(())
    }

    pub(crate) fn generation_ready_after(&self) -> Option<vulkan::SubmissionTimelineWait> {
        self._read_lease.ready_after()
    }

    pub(crate) fn state_chunk_count(&self, max_chunk_values: usize) -> Result<usize> {
        if max_chunk_values == 0 {
            bail!("replica-state chunk size must be positive");
        }
        self.slots.iter().try_fold(0usize, |count, slot| {
            count
                .checked_add(slot.len.div_ceil(max_chunk_values))
                .context("replica-state chunk count overflow")
        })
    }

    pub(crate) fn prepare_range_retirement_consumers(
        &self,
        max_chunk_values: usize,
        consumer_count: usize,
    ) -> Result<Vec<OptimizerGenerationRangeConsumer>> {
        let range_count = self.state_chunk_count(max_chunk_values)?;
        self._read_lease
            .split_into_range_consumers(range_count, consumer_count)
    }

    /// Materialize the portable host representation only for a peer that could
    /// not use a direct Vulkan transport. The generation lease stays live for
    /// the entire readback, so the source buffers cannot cross an AdamW
    /// mutation while the fallback snapshot is being assembled.
    pub(crate) fn portable_state_snapshot(
        &self,
    ) -> Result<(Vec<RwkvParameterSnapshot>, AdamWOptimizerState)> {
        let (step, slot_steps) = self.exact_step_metadata()?;
        let mut parameters = Vec::with_capacity(self.slots.len());
        let mut optimizer_slots = Vec::with_capacity(self.slots.len());
        for (slot, slot_step) in self.slots.iter().zip(slot_steps) {
            parameters.push(RwkvParameterSnapshot {
                name: slot.name.clone(),
                values: slot.parameter.read_f32(slot.len)?,
            });
            optimizer_slots.push(AdamWOptimizerSlotState {
                name: slot.name.clone(),
                step: slot_step,
                decay_class: Some(slot.decay_class),
                exp_avg: slot.exp_avg.read_f32(slot.len)?,
                exp_avg_sq: slot.exp_avg_sq.read_f32(slot.len)?,
            });
        }
        Ok((
            parameters,
            AdamWOptimizerState {
                step,
                slots: optimizer_slots,
            },
        ))
    }

    pub(crate) fn record_range_copy(
        &self,
        commands: &mut vulkan::ComputeBatch,
        tensor_index: usize,
        plane: RwkvReplicaStatePlane,
        offset: usize,
        len: usize,
        destination: &GpuBuffer,
    ) -> Result<()> {
        let slot = self.slots.get(tensor_index).with_context(|| {
            format!("replica-state source tensor index {tensor_index} is out of range")
        })?;
        let end = offset
            .checked_add(len)
            .context("replica-state source copy range overflow")?;
        if len == 0 || end > slot.len {
            bail!(
                "replica-state source copy range {offset}..{end} is outside tensor {:?} length {}",
                slot.name,
                slot.len
            );
        }
        if len > destination.f32_capacity() {
            bail!(
                "replica-state source copy chunk length {len} exceeds destination capacity {}",
                destination.f32_capacity()
            );
        }
        let source = match plane {
            RwkvReplicaStatePlane::Parameter => &slot.parameter,
            RwkvReplicaStatePlane::ExpAvg => &slot.exp_avg,
            RwkvReplicaStatePlane::ExpAvgSq => &slot.exp_avg_sq,
        };
        commands.copy_f32_range(source, offset, destination, 0, len)
    }
}

#[derive(Clone)]
pub(crate) struct RwkvParameterStorageMirrorBinding {
    pub name: String,
    pub mirror: VulkanParameterStorageMirror,
}

/// Persistent, all-device AdamW state for the fused RWKV cell.
///
/// Gradient buffers owned by the cell are per-recompute scratch. This object
/// accumulates them after every reverse timestep, then performs one AdamW step
/// after the whole TBPTT sequence. First/second moments persist across calls.
pub(crate) struct RwkvPersistentAdamW {
    device: VulkanDevice,
    step: u32,
    device_step: GpuBuffer,
    device_step_authoritative: bool,
    host_step_metadata_authoritative: bool,
    step_metadata_readback: GpuBuffer,
    slots: Vec<OptimizerSlot>,
    permanently_inactive_names: HashSet<String>,
    gradient_accumulate: vulkan::ComputeKernel,
    gradient_accumulate4: Option<vulkan::ComputeKernel>,
    gradient_scale: vulkan::ComputeKernel,
    gradient_scale_from_buffer: vulkan::ComputeKernel,
    gradient_scale_from_buffer_indexed: vulkan::ComputeKernel,
    adamw: vulkan::ComputeKernel,
    adamw_range: vulkan::ComputeKernel,
    adamw_range_controlled: vulkan::ComputeKernel,
    adamw_range_grad_scaler_controlled: vulkan::ComputeKernel,
    adamw_step_grad_scaler_controlled: vulkan::ComputeKernel,
    adamw_fp16_mirror: vulkan::ComputeKernel,
    adamw_controlled: vulkan::ComputeKernel,
    adamw_fp16_mirror_controlled: vulkan::ComputeKernel,
    gradient_nonfinite_detector: VulkanGradientNonfiniteDetector,
    gradient_l2_norm_reducer: VulkanGradientL2NormReducer,
    parameter_mirror_refresher: Option<VulkanParameterStorageMirrorRefresher>,
    generation_guard: Arc<OptimizerGenerationGuard>,
}

impl RwkvPersistentAdamW {
    pub fn new(device: VulkanDevice, trainables: &[RwkvTrainableRef<'_>]) -> Result<Self> {
        if trainables.is_empty() {
            bail!("RWKV persistent optimizer requires at least one trainable tensor");
        }
        let mut slots = Vec::with_capacity(trainables.len());
        let mut names = HashSet::with_capacity(trainables.len());
        for trainable in trainables {
            if trainable.len == 0 {
                bail!("RWKV trainable {} has zero elements", trainable.name);
            }
            if !names.insert(trainable.name) {
                bail!(
                    "RWKV persistent optimizer registry contains duplicate tensor {:?}",
                    trainable.name
                );
            }
            slots.push(OptimizerSlot {
                name: trainable.name.to_string(),
                len: trainable.len,
                decay_class: trainable.decay_class,
                step: 0,
                device_step: GpuBuffer::zeros_u32(&device, 1)?,
                device_step_authoritative: true,
                accumulated_grad: GpuBuffer::zeros_f32(&device, trainable.len)?,
                exp_avg: GpuBuffer::zeros_f32(&device, trainable.len)?,
                exp_avg_sq: GpuBuffer::zeros_f32(&device, trainable.len)?,
                parameter_readback: GpuBuffer::zeros_host_f32(&device, trainable.len)?,
                parameter_storage_mirror: None,
            });
        }
        Ok(Self {
            gradient_accumulate: vulkan::ComputeKernel::new_with_access(
                &device,
                GRADIENT_ACCUMULATE_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            gradient_accumulate4: if device.supports_storage_buffer_bindings(8) {
                Some(vulkan::ComputeKernel::new_with_access(
                    &device,
                    GRADIENT_ACCUMULATE4_SPV,
                    &[
                        vulkan::BindingAccess::ReadWrite,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadWrite,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadWrite,
                        vulkan::BindingAccess::ReadOnly,
                        vulkan::BindingAccess::ReadWrite,
                        vulkan::BindingAccess::ReadOnly,
                    ],
                    std::mem::size_of::<GradientAccumulate4Push>() as u32,
                )?)
            } else {
                None
            },
            gradient_scale: vulkan::ComputeKernel::new(
                &device,
                GRADIENT_SCALE_SPV,
                1,
                std::mem::size_of::<GradientScalePush>() as u32,
            )?,
            gradient_scale_from_buffer: vulkan::ComputeKernel::new_with_access(
                &device,
                GRADIENT_SCALE_FROM_BUFFER_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            gradient_scale_from_buffer_indexed: vulkan::ComputeKernel::new_with_access(
                &device,
                GRADIENT_SCALE_FROM_BUFFER_INDEXED_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<GradientScaleFromBufferIndexedPush>() as u32,
            )?,
            adamw: vulkan::ComputeKernel::new(
                &device,
                ADAMW_SPV,
                4,
                std::mem::size_of::<AdamWPush>() as u32,
            )?,
            adamw_range: vulkan::ComputeKernel::new(
                &device,
                ADAMW_RANGE_SPV,
                4,
                std::mem::size_of::<AdamWRangePush>() as u32,
            )?,
            adamw_range_controlled: vulkan::ComputeKernel::new_with_access(
                &device,
                ADAMW_RANGE_CONTROLLED_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<AdamWRangeControlledPush>() as u32,
            )?,
            adamw_range_grad_scaler_controlled: vulkan::ComputeKernel::new_with_access(
                &device,
                ADAMW_RANGE_GRAD_SCALER_CONTROLLED_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<AdamWRangeGradScalerControlledPush>() as u32,
            )?,
            adamw_step_grad_scaler_controlled: vulkan::ComputeKernel::new_with_access(
                &device,
                ADAMW_STEP_GRAD_SCALER_CONTROLLED_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                ],
                0,
            )?,
            adamw_fp16_mirror: vulkan::ComputeKernel::new_with_access(
                &device,
                ADAMW_FP16_MIRROR_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<AdamWPush>() as u32,
            )?,
            adamw_controlled: vulkan::ComputeKernel::new_with_access(
                &device,
                ADAMW_CONTROLLED_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<AdamWControlledPush>() as u32,
            )?,
            adamw_fp16_mirror_controlled: vulkan::ComputeKernel::new_with_access(
                &device,
                ADAMW_FP16_MIRROR_CONTROLLED_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<AdamWControlledPush>() as u32,
            )?,
            gradient_nonfinite_detector: VulkanGradientNonfiniteDetector::new(device.clone())?,
            gradient_l2_norm_reducer: VulkanGradientL2NormReducer::new(
                &device,
                &trainables
                    .iter()
                    .map(|trainable| trainable.len)
                    .collect::<Vec<_>>(),
            )?,
            step: 0,
            device_step: GpuBuffer::zeros_u32(&device, 1)?,
            device_step_authoritative: true,
            host_step_metadata_authoritative: true,
            step_metadata_readback: GpuBuffer::zeros_host_f32(&device, slots.len() + 1)?,
            device,
            slots,
            permanently_inactive_names: HashSet::new(),
            parameter_mirror_refresher: None,
            generation_guard: Arc::new(OptimizerGenerationGuard::default()),
        })
    }

    /// Restrict optimizer mutation to tensors whose canonical name is exactly
    /// one of `prefixes` or is nested beneath one of them. Forward/backward
    /// execution remains unchanged, but every other parameter, moment tensor,
    /// and per-slot Adam clock is frozen. This is the primitive used by the
    /// native parameter-efficient fine-tuning frontend.
    pub(crate) fn set_active_parameter_prefixes(
        &mut self,
        prefixes: &[String],
    ) -> Result<(usize, usize)> {
        if prefixes.is_empty() {
            self.permanently_inactive_names.clear();
            return Ok((self.slots.len(), 0));
        }
        let normalized = prefixes
            .iter()
            .map(|prefix| prefix.trim())
            .collect::<Vec<_>>();
        if normalized.iter().any(|prefix| prefix.is_empty()) {
            bail!("optimizer trainable-prefix selection contains an empty prefix");
        }
        let unique = normalized.iter().copied().collect::<HashSet<_>>();
        if unique.len() != normalized.len() {
            bail!("optimizer trainable-prefix selection contains duplicate prefixes");
        }

        let matches_prefix = |name: &str, prefix: &str| {
            name == prefix
                || name
                    .strip_prefix(prefix)
                    .is_some_and(|suffix| suffix.starts_with('.'))
        };
        for prefix in &normalized {
            if !self
                .slots
                .iter()
                .any(|slot| matches_prefix(&slot.name, prefix))
            {
                bail!("optimizer trainable prefix {prefix:?} matches no registered tensor");
            }
        }

        self.permanently_inactive_names = self
            .slots
            .iter()
            .filter(|slot| {
                !normalized
                    .iter()
                    .any(|prefix| matches_prefix(&slot.name, prefix))
            })
            .map(|slot| slot.name.clone())
            .collect();
        let inactive = self.permanently_inactive_names.len();
        let active = self.slots.len().saturating_sub(inactive);
        if active == 0 {
            bail!("optimizer trainable-prefix selection froze every registered tensor");
        }
        Ok((active, inactive))
    }

    fn effective_inactive_names(&self, inactive_names: &[&str]) -> Result<HashSet<String>> {
        let requested = inactive_names.iter().copied().collect::<HashSet<_>>();
        if requested.len() != inactive_names.len() {
            bail!("inactive optimizer-slot list contains duplicate tensor names");
        }
        for name in &requested {
            if !self.slots.iter().any(|slot| slot.name == *name) {
                bail!("persistent optimizer has no registered inactive tensor {name:?}");
            }
        }
        let mut inactive = self.permanently_inactive_names.clone();
        inactive.extend(requested.into_iter().map(str::to_string));
        Ok(inactive)
    }

    /// Allocate compact execution mirrors for selected canonical trainables and
    /// initialize them from the current FP32 masters in one submission.
    ///
    /// AdamW moments and parameters remain FP32. The returned bindings are
    /// cheap clones of the compact buffer identities so graph consumers and
    /// secondary/local optimizer registries can target the exact same storage.
    pub(crate) fn enable_parameter_storage_mirrors(
        &mut self,
        trainables: &[RwkvTrainableRef<'_>],
        format: VulkanParameterStorageFormat,
        names: &[&str],
    ) -> Result<Vec<RwkvParameterStorageMirrorBinding>> {
        self.validate_registry(trainables)?;
        if names.is_empty() {
            return Ok(Vec::new());
        }
        let requested: HashSet<&str> = names.iter().copied().collect();
        if requested.len() != names.len() {
            bail!("mixed-precision mirror request contains duplicate tensor names");
        }
        for &name in names {
            if !self.slots.iter().any(|slot| slot.name == name) {
                bail!("persistent optimizer has no registered tensor {name:?} for mixed-precision mirror");
            }
        }
        if let Some(existing_format) = self
            .slots
            .iter()
            .filter_map(|slot| slot.parameter_storage_mirror.as_ref())
            .map(VulkanParameterStorageMirror::format)
            .next()
        {
            if existing_format != format {
                bail!(
                    "persistent optimizer already owns {} mirrors; cannot attach {} in the same registry",
                    existing_format.label(),
                    format.label()
                );
            }
        }
        if self.parameter_mirror_refresher.is_none() {
            self.parameter_mirror_refresher = Some(VulkanParameterStorageMirrorRefresher::new(
                &self.device,
                format,
            )?);
        }

        for slot in &mut self.slots {
            if requested.contains(slot.name.as_str()) && slot.parameter_storage_mirror.is_none() {
                slot.parameter_storage_mirror = Some(VulkanParameterStorageMirror::new(
                    &self.device,
                    format,
                    slot.len,
                )?);
            }
        }

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let refresher = self
            .parameter_mirror_refresher
            .as_ref()
            .context("mixed-precision mirror refresher was not initialized")?;
        for (slot, trainable) in self.slots.iter().zip(trainables) {
            if requested.contains(slot.name.as_str()) {
                let mirror = slot
                    .parameter_storage_mirror
                    .as_ref()
                    .context("requested mixed-precision mirror was not allocated")?;
                refresher.record_refresh(&mut commands, trainable.parameter, mirror)?;
            }
        }
        commands.submit()?;

        self.parameter_storage_mirror_bindings(names)
    }

    /// Attach already-initialized compact storage identities to another
    /// optimizer registry. This is used by the legacy per-recurrent AdamW path
    /// so it refreshes the same buffers consumed by the full-model graph.
    pub(crate) fn attach_parameter_storage_mirrors(
        &mut self,
        trainables: &[RwkvTrainableRef<'_>],
        format: VulkanParameterStorageFormat,
        bindings: &[RwkvParameterStorageMirrorBinding],
    ) -> Result<()> {
        self.validate_registry(trainables)?;
        if bindings.is_empty() {
            return Ok(());
        }
        let mut seen = HashSet::with_capacity(bindings.len());
        for binding in bindings {
            if !seen.insert(binding.name.as_str()) {
                bail!(
                    "mixed-precision mirror bindings contain duplicate tensor {:?}",
                    binding.name
                );
            }
            if binding.mirror.format() != format {
                bail!(
                    "mixed-precision binding {:?} is {}, expected {}",
                    binding.name,
                    binding.mirror.format().label(),
                    format.label()
                );
            }
            let slot = self
                .slots
                .iter_mut()
                .find(|slot| slot.name == binding.name)
                .with_context(|| {
                    format!(
                        "persistent optimizer has no registered tensor {:?} for mixed-precision binding",
                        binding.name
                    )
                })?;
            if slot.len != binding.mirror.len() {
                bail!(
                    "mixed-precision binding {:?} has {} elements; optimizer slot expects {}",
                    binding.name,
                    binding.mirror.len(),
                    slot.len
                );
            }
            slot.parameter_storage_mirror = Some(binding.mirror.clone());
        }
        self.parameter_mirror_refresher = Some(VulkanParameterStorageMirrorRefresher::new(
            &self.device,
            format,
        )?);
        Ok(())
    }

    pub(crate) fn parameter_storage_mirror_bindings(
        &self,
        names: &[&str],
    ) -> Result<Vec<RwkvParameterStorageMirrorBinding>> {
        names
            .iter()
            .map(|&name| {
                let slot = self
                    .slots
                    .iter()
                    .find(|slot| slot.name == name)
                    .with_context(|| {
                        format!("persistent optimizer has no registered tensor {name:?}")
                    })?;
                let mirror = slot.parameter_storage_mirror.as_ref().with_context(|| {
                    format!("persistent optimizer tensor {name:?} has no mixed-precision mirror")
                })?;
                Ok(RwkvParameterStorageMirrorBinding {
                    name: slot.name.clone(),
                    mirror: mirror.clone(),
                })
            })
            .collect()
    }

    pub fn record_zero_grad(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        for slot in &self.slots {
            commands.fill_zero_f32(&slot.accumulated_grad, slot.len)?;
        }
        Ok(())
    }

    /// Clear the canonical gradient registry except for one slot whose AdamW
    /// step will consume an external device-resident gradient directly. This is
    /// useful for very large tied parameters where staging into a second
    /// optimizer-owned accumulator would be pure bandwidth overhead.
    pub(crate) fn record_zero_grad_except_named(
        &self,
        commands: &mut vulkan::ComputeBatch,
        excluded_name: &str,
    ) -> Result<()> {
        let mut found = false;
        for slot in &self.slots {
            if slot.name == excluded_name {
                found = true;
                continue;
            }
            commands.fill_zero_f32(&slot.accumulated_grad, slot.len)?;
        }
        if !found {
            bail!(
                "persistent optimizer has no registered tensor {excluded_name:?} to exclude from zero-grad"
            );
        }
        Ok(())
    }

    pub(crate) fn current_step(&self) -> u32 {
        self.step
    }

    /// Read exact global/per-parameter Adam clocks in one Vulkan submission.
    /// Device-authoritative entries are copied into one compact host-visible
    /// row; entries whose device mirror was invalidated by a host-owned path
    /// continue to use their exact Rust value. This is the synchronization seam
    /// for checkpointing/telemetry after queue-resident optimizer windows.
    fn exact_step_metadata(&self) -> Result<(u32, Vec<u32>)> {
        if self.host_step_metadata_authoritative {
            return Ok((self.step, self.slots.iter().map(|slot| slot.step).collect()));
        }

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let mut copied_any = false;
        if self.device_step_authoritative {
            commands.copy_f32_range(&self.device_step, 0, &self.step_metadata_readback, 0, 1)?;
            copied_any = true;
        }
        for (slot_index, slot) in self.slots.iter().enumerate() {
            if slot.device_step_authoritative {
                commands.copy_f32_range(
                    &slot.device_step,
                    0,
                    &self.step_metadata_readback,
                    slot_index + 1,
                    1,
                )?;
                copied_any = true;
            }
        }
        let words = if copied_any {
            commands.submit()?;
            self.step_metadata_readback
                .read_f32(self.slots.len() + 1)?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        } else {
            vec![0; self.slots.len() + 1]
        };
        let step = if self.device_step_authoritative {
            words[0]
        } else {
            self.step
        };
        let slot_steps = self
            .slots
            .iter()
            .enumerate()
            .map(|(slot_index, slot)| {
                if slot.device_step_authoritative {
                    words[slot_index + 1]
                } else {
                    slot.step
                }
            })
            .collect::<Vec<_>>();
        if slot_steps.iter().any(|&slot_step| slot_step > step) {
            bail!("device AdamW metadata contains a per-slot step beyond its global step");
        }
        Ok((step, slot_steps))
    }

    pub(crate) fn synchronize_device_step_metadata(&mut self) -> Result<RwkvOptimizerStepResult> {
        let (step, slot_steps) = self.exact_step_metadata()?;
        self.step = step;
        for (slot, slot_step) in self.slots.iter_mut().zip(slot_steps) {
            slot.step = slot_step;
        }
        self.host_step_metadata_authoritative = true;
        Ok(RwkvOptimizerStepResult {
            step,
            tensor_count: self.slots.len(),
        })
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.slots.len()
    }

    /// Bind the AMP overflow detector's device-local flag to the enclosing
    /// training graph's aliased working-set arena. Optimizer masters, moments,
    /// and canonical gradients remain persistent and checkpoint-identical.
    pub(crate) fn bind_nonfinite_flag_buffer(&mut self, flag: GpuBuffer) -> Result<()> {
        self.gradient_nonfinite_detector.bind_flag_buffer(flag)
    }

    pub(crate) fn nonfinite_flag_buffer(&self) -> &GpuBuffer {
        self.gradient_nonfinite_detector.flag_buffer()
    }

    pub(crate) fn record_accumulated_gradient_nonfinite_scan_with_named_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<()> {
        self.record_accumulated_gradient_nonfinite_scan_with_named_override_and_inactive_names(
            commands,
            gradient_override,
            &[],
        )
    }

    pub(crate) fn record_accumulated_gradient_nonfinite_scan_with_named_override_and_inactive_names(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
    ) -> Result<()> {
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let inactive = self.effective_inactive_names(inactive_names)?;
        let gradients = self
            .slots
            .iter()
            .filter(|slot| !inactive.contains(slot.name.as_str()))
            .map(|slot| {
                let gradient = gradient_override
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, gradient)| gradient)
                    .unwrap_or(&slot.accumulated_grad);
                (gradient, slot.len)
            })
            .collect::<Vec<_>>();
        self.gradient_nonfinite_detector
            .record_scan_device_only(commands, &gradients)
    }

    /// Reduce the exact canonical accumulated-gradient registry that the next
    /// AdamW step would consume. This is the native AMP/GradScaler overflow
    /// boundary; scratch gradients are deliberately irrelevant here.
    pub(crate) fn accumulated_gradients_have_nonfinite(&self) -> Result<bool> {
        let gradients = self
            .slots
            .iter()
            .filter(|slot| !self.permanently_inactive_names.contains(slot.name.as_str()))
            .map(|slot| (&slot.accumulated_grad, slot.len))
            .collect::<Vec<_>>();
        self.gradient_nonfinite_detector.has_nonfinite(&gradients)
    }

    pub(crate) fn accumulated_gradients_have_nonfinite_with_named_override(
        &self,
        override_name: &str,
        override_gradient: &GpuBuffer,
    ) -> Result<bool> {
        self.validate_gradient_override(override_name, override_gradient)?;
        let gradients = self
            .slots
            .iter()
            .map(|slot| {
                let gradient = if slot.name == override_name {
                    override_gradient
                } else {
                    &slot.accumulated_grad
                };
                (gradient, slot.len)
            })
            .collect::<Vec<_>>();
        self.gradient_nonfinite_detector.has_nonfinite(&gradients)
    }

    pub(crate) fn read_accumulated_gradient_l2_norm(&self) -> Result<f64> {
        self.gradient_l2_norm_reducer.read_l2_norm()
    }

    /// Record the global-L2 reduction and derive the PyTorch clipping scalar on
    /// Vulkan. The scalar remains device-resident so a following gradient-scale
    /// pass can be recorded into the same command stream without a CPU decision.
    pub(crate) fn record_accumulated_gradient_l2_norm_and_clip_coefficient_with_override_and_inactive_names(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
        max_norm: f32,
    ) -> Result<()> {
        self.record_accumulated_gradient_l2_norm_and_clip_coefficient_impl(
            commands,
            gradient_override,
            inactive_names,
            max_norm,
            true,
        )
    }

    /// Queue-resident clipping reduction for production AMP windows. The
    /// coefficient and non-finite bit stay on Vulkan; unlike the telemetry form
    /// above this records no norm/coefficient/safety copies into host-visible
    /// buffers, so the optimizer wavefront can consume the result without a
    /// hidden CPU synchronization point.
    pub(crate) fn record_accumulated_gradient_l2_norm_and_clip_coefficient_device_only_with_override_and_inactive_names(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
        max_norm: f32,
    ) -> Result<()> {
        self.record_accumulated_gradient_l2_norm_and_clip_coefficient_impl(
            commands,
            gradient_override,
            inactive_names,
            max_norm,
            false,
        )
    }

    fn record_accumulated_gradient_l2_norm_and_clip_coefficient_impl(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
        max_norm: f32,
        record_host_telemetry: bool,
    ) -> Result<()> {
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let inactive = self.effective_inactive_names(inactive_names)?;
        let gradients = self
            .slots
            .iter()
            .filter(|slot| !inactive.contains(slot.name.as_str()))
            .map(|slot| {
                let gradient = gradient_override
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, gradient)| gradient)
                    .unwrap_or(&slot.accumulated_grad);
                (gradient, slot.len)
            })
            .collect::<Vec<_>>();
        if record_host_telemetry {
            self.gradient_l2_norm_reducer
                .record_l2_norm_and_clip_coefficient(commands, &gradients, max_norm)
        } else {
            self.gradient_l2_norm_reducer
                .record_l2_norm_and_clip_coefficient_device_only(commands, &gradients, max_norm)
        }
    }

    pub(crate) fn read_accumulated_gradient_clip_coefficient(&self) -> Result<f32> {
        self.gradient_l2_norm_reducer.read_clip_coefficient()
    }

    pub(crate) fn read_accumulated_gradient_clip_has_nonfinite(&self) -> Result<bool> {
        self.gradient_l2_norm_reducer.read_clip_has_nonfinite()
    }

    pub(crate) fn accumulated_gradient_clip_nonfinite_buffer(&self) -> &GpuBuffer {
        self.gradient_l2_norm_reducer.clip_nonfinite_buffer()
    }

    /// Scale the canonical accumulated-gradient registry in place. Sequence
    /// owners use this immediately before AdamW to turn a sum of token or
    /// microbatch gradients into the corresponding mean without any host
    /// materialization.
    pub(crate) fn record_scale_gradients(
        &self,
        commands: &mut vulkan::ComputeBatch,
        scale: f32,
    ) -> Result<()> {
        self.record_scale_gradients_with_override(commands, scale, None)
    }

    pub(crate) fn record_scale_gradients_with_named_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        scale: f32,
        override_name: &str,
        override_gradient: &GpuBuffer,
    ) -> Result<()> {
        self.record_scale_gradients_with_override(
            commands,
            scale,
            Some((override_name, override_gradient)),
        )
    }

    /// Scale every canonical accumulated gradient by the device-side clipping
    /// coefficient emitted by the most recent global-norm reduction.
    pub(crate) fn record_scale_gradients_from_device_clip_coefficient(
        &self,
        commands: &mut vulkan::ComputeBatch,
    ) -> Result<()> {
        self.record_scale_gradients_from_device_clip_coefficient_with_override(commands, None)
    }

    pub(crate) fn record_scale_gradients_from_device_clip_coefficient_with_named_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        override_name: &str,
        override_gradient: &GpuBuffer,
    ) -> Result<()> {
        self.record_scale_gradients_from_device_clip_coefficient_with_override(
            commands,
            Some((override_name, override_gradient)),
        )
    }

    /// Multiply every canonical accumulated gradient by one scalar selected
    /// from a device-resident control buffer, composed with a host-known
    /// multiplier. This is the queue-resident AMP preparation seam: GradScaler
    /// can own the reciprocal loss scale while sequence normalization remains a
    /// deterministic host-side schedule constant.
    pub(crate) fn record_scale_gradients_from_indexed_device_factor_with_named_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        factor: &GpuBuffer,
        factor_index: usize,
        multiplier: f32,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<()> {
        if !multiplier.is_finite() || multiplier <= 0.0 {
            bail!(
                "indexed device gradient-scale multiplier must be finite and positive; got {multiplier}"
            );
        }
        if factor_index >= factor.f32_capacity() {
            bail!(
                "indexed device gradient-scale factor {factor_index} is outside buffer capacity {}",
                factor.f32_capacity()
            );
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let scale_index = u32::try_from(factor_index)
            .context("indexed device gradient-scale factor index exceeds Vulkan u32")?;
        for slot in &self.slots {
            let gradient = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            let push = GradientScaleFromBufferIndexedPush {
                len: u32::try_from(slot.len)
                    .context("indexed device gradient-scale tensor length exceeds Vulkan u32")?,
                scale_index,
                multiplier,
            };
            self.gradient_scale_from_buffer_indexed.record_dispatch(
                commands,
                &[gradient, factor],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(slot.len, 256), 1, 1],
            )?;
        }
        Ok(())
    }

    fn record_scale_gradients_from_device_clip_coefficient_with_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<()> {
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let coefficient = self.gradient_l2_norm_reducer.clip_coefficient_buffer();
        for slot in &self.slots {
            let gradient = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            self.gradient_scale_from_buffer.record_dispatch(
                commands,
                &[gradient, coefficient],
                bytemuck::bytes_of(&LenPush {
                    len: slot.len as u32,
                }),
                [div_ceil_u32(slot.len, 256), 1, 1],
            )?;
        }
        Ok(())
    }

    /// Scale one canonical accumulated-gradient slot without touching the rest
    /// of the registry. Native auxiliary controllers use this after the common
    /// sequence normalization so an already-mean auxiliary is not diluted by
    /// the token count used by the language-model objective.
    pub(crate) fn record_scale_named_gradient(
        &self,
        commands: &mut vulkan::ComputeBatch,
        name: &str,
        scale: f32,
    ) -> Result<()> {
        if !scale.is_finite() || scale < 0.0 {
            bail!("named gradient scale must be finite and non-negative; got {scale}");
        }
        let slot = self
            .slots
            .iter()
            .find(|slot| slot.name == name)
            .with_context(|| format!("persistent optimizer has no registered tensor {name:?}"))?;
        if scale == 1.0 {
            return Ok(());
        }
        let push = GradientScalePush {
            len: slot.len as u32,
            scale,
        };
        self.gradient_scale.record_dispatch(
            commands,
            &[&slot.accumulated_grad],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(slot.len, 256), 1, 1],
        )
    }

    fn record_scale_gradients_with_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        scale: f32,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<()> {
        if !scale.is_finite() || scale <= 0.0 {
            bail!("gradient scale must be finite and positive; got {scale}");
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        if scale == 1.0 {
            return Ok(());
        }
        for slot in &self.slots {
            let gradient = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            let push = GradientScalePush {
                len: slot.len as u32,
                scale,
            };
            self.gradient_scale.record_dispatch(
                commands,
                &[gradient],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(slot.len, 256), 1, 1],
            )?;
        }
        Ok(())
    }

    pub fn record_accumulate(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
    ) -> Result<()> {
        self.validate_registry(trainables)?;
        let entries = self
            .slots
            .iter()
            .zip(trainables)
            .map(|(slot, trainable)| (slot, trainable.gradient))
            .collect::<Vec<_>>();
        self.record_accumulate_entries(commands, &entries)
    }

    /// Accumulate an arbitrary set of live scratch gradients by registry name.
    /// Four independent tensors are packed into one dispatch when the device
    /// exposes enough storage bindings; the portable single-tensor kernel is
    /// retained for the tail and lower-binding devices.
    pub(crate) fn record_accumulate_many(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
    ) -> Result<()> {
        let mut entries = Vec::with_capacity(trainables.len());
        for trainable in trainables {
            let slot = self
                .slots
                .iter()
                .find(|slot| slot.name == trainable.name)
                .with_context(|| {
                    format!(
                        "persistent optimizer has no registered tensor {:?}",
                        trainable.name
                    )
                })?;
            validate_slot(slot, trainable)?;
            entries.push((slot, trainable.gradient));
        }
        self.record_accumulate_entries(commands, &entries)
    }

    pub(crate) fn record_accumulate_many_named(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[(&str, RwkvTrainableRef<'_>)],
    ) -> Result<()> {
        let mut entries = Vec::with_capacity(trainables.len());
        for (canonical_name, trainable) in trainables {
            let slot = self
                .slots
                .iter()
                .find(|slot| slot.name == *canonical_name)
                .with_context(|| {
                    format!(
                        "persistent optimizer has no registered tensor {:?}",
                        canonical_name
                    )
                })?;
            let named = RwkvTrainableRef {
                name: *canonical_name,
                ..*trainable
            };
            validate_slot(slot, &named)?;
            entries.push((slot, trainable.gradient));
        }
        self.record_accumulate_entries(commands, &entries)
    }

    fn record_accumulate_entries(
        &self,
        commands: &mut vulkan::ComputeBatch,
        entries: &[(&OptimizerSlot, &GpuBuffer)],
    ) -> Result<()> {
        let Some(kernel) = self.gradient_accumulate4.as_ref() else {
            for (slot, gradient) in entries {
                self.record_accumulate_entry(commands, slot, gradient)?;
            }
            return Ok(());
        };

        let mut groups = entries.chunks_exact(4);
        for group in groups.by_ref() {
            let push = GradientAccumulate4Push {
                len0: group[0].0.len as u32,
                len1: group[1].0.len as u32,
                len2: group[2].0.len as u32,
                len3: group[3].0.len as u32,
            };
            let group_count = group
                .iter()
                .map(|(slot, _)| div_ceil_u32(slot.len, 256))
                .sum();
            kernel.record_dispatch(
                commands,
                &[
                    &group[0].0.accumulated_grad,
                    group[0].1,
                    &group[1].0.accumulated_grad,
                    group[1].1,
                    &group[2].0.accumulated_grad,
                    group[2].1,
                    &group[3].0.accumulated_grad,
                    group[3].1,
                ],
                bytemuck::bytes_of(&push),
                [group_count, 1, 1],
            )?;
        }
        for (slot, gradient) in groups.remainder() {
            self.record_accumulate_entry(commands, slot, gradient)?;
        }
        Ok(())
    }

    fn record_accumulate_entry(
        &self,
        commands: &mut vulkan::ComputeBatch,
        slot: &OptimizerSlot,
        gradient: &GpuBuffer,
    ) -> Result<()> {
        let push = LenPush {
            len: slot.len as u32,
        };
        self.gradient_accumulate.record_dispatch(
            commands,
            &[&slot.accumulated_grad, gradient],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(slot.len, 256), 1, 1],
        )
    }

    /// Accumulate one scratch gradient immediately after the graph branch that
    /// produced it. This is the composition seam for parameters (notably the
    /// manager/worker projections) that are reused several times before the
    /// full-model optimizer step and whose local backward kernels overwrite
    /// scratch storage on every invocation.
    #[allow(dead_code)]
    pub fn record_accumulate_one(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainable: RwkvTrainableRef<'_>,
    ) -> Result<()> {
        let slot = self
            .slots
            .iter()
            .find(|slot| slot.name == trainable.name)
            .with_context(|| {
                format!(
                    "persistent optimizer has no registered tensor {:?}",
                    trainable.name
                )
            })?;
        validate_slot(slot, &trainable)?;
        self.record_accumulate_entry(commands, slot, trainable.gradient)
    }

    pub fn record_step(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
    ) -> Result<RwkvOptimizerStepResult> {
        self.record_step_with_gradient_override(commands, trainables, hyper, None)
    }

    /// PyTorch `zero_grad(set_to_none=True)` semantics for a fixed Vulkan
    /// registry: named inactive slots are left completely untouched, including
    /// their parameter, first/second moments, and per-slot Adam step counter.
    pub(crate) fn record_step_with_inactive_names(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        inactive_names: &[&str],
    ) -> Result<RwkvOptimizerStepResult> {
        self.record_step_with_gradient_override_and_inactive_names(
            commands,
            trainables,
            hyper,
            None,
            inactive_names,
        )
    }

    pub(crate) fn record_step_with_named_gradient_override_and_inactive_names(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        override_name: &str,
        override_gradient: &GpuBuffer,
        inactive_names: &[&str],
    ) -> Result<RwkvOptimizerStepResult> {
        self.record_step_with_gradient_override_and_inactive_names(
            commands,
            trainables,
            hyper,
            Some((override_name, override_gradient)),
            inactive_names,
        )
    }

    fn record_step_with_gradient_override(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<RwkvOptimizerStepResult> {
        self.record_step_with_gradient_override_and_inactive_names(
            commands,
            trainables,
            hyper,
            gradient_override,
            &[],
        )
    }

    /// Record one AdamW step whose mutation/clear decision is read from a
    /// Vulkan-resident optimizer-control buffer. Host optimizer counters are
    /// intentionally committed only after submission via
    /// [`Self::finalize_device_controlled_step`].
    pub(crate) fn record_device_controlled_step(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        control: &GpuBuffer,
        gradient_scale: f32,
        apply_control_unscale: bool,
        named_gradient_scale: Option<(&str, f32)>,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
    ) -> Result<RwkvDeviceControlledStepPending> {
        hyper.validate()?;
        self.validate_registry(trainables)?;
        if control.f32_capacity() < crate::training_numerics::DYNAMIC_LOSS_SCALE_CONTROL_WORDS {
            bail!(
                "device-controlled AdamW control buffer has capacity {}, expected at least {} words",
                control.f32_capacity(),
                crate::training_numerics::DYNAMIC_LOSS_SCALE_CONTROL_WORDS
            );
        }
        if !gradient_scale.is_finite() || gradient_scale <= 0.0 {
            bail!("device-controlled AdamW gradient scale must be finite and positive");
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        if let Some((name, scale)) = named_gradient_scale {
            if !scale.is_finite() || scale <= 0.0 {
                bail!("device-controlled AdamW named gradient scale for {name:?} is invalid");
            }
            if !self.slots.iter().any(|slot| slot.name == name) {
                bail!("persistent optimizer has no registered tensor {name:?} for named gradient scaling");
            }
        }
        let inactive = self.effective_inactive_names(inactive_names)?;

        let previous_step = self.step;
        let next_step = previous_step
            .checked_add(1)
            .context("RWKV device-controlled AdamW step overflow")?;
        let optimizer_generation = self.generation_guard.current_generation();
        self.generation_guard
            .wait_for_readers_to_retire(optimizer_generation)?;
        let parameter_mirror_refresher = self.parameter_mirror_refresher.as_ref();
        let mut active_slot_indices = Vec::with_capacity(self.slots.len());

        for (slot_index, (slot, trainable)) in self.slots.iter().zip(trainables).enumerate() {
            let active = !inactive.contains(slot.name.as_str());
            let slot_next_step = if active {
                active_slot_indices.push(slot_index);
                slot.step
                    .checked_add(1)
                    .with_context(|| format!("RWKV AdamW slot {:?} step overflow", slot.name))?
            } else {
                slot.step.saturating_add(1).max(1)
            };
            let gradient = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            let per_tensor_scale = named_gradient_scale
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, scale)| scale)
                .unwrap_or(1.0);
            let combined_scale = gradient_scale * per_tensor_scale;
            if !combined_scale.is_finite() || combined_scale <= 0.0 {
                bail!(
                    "device-controlled AdamW combined gradient scale for {:?} is invalid",
                    slot.name
                );
            }
            let weight_decay = match slot.decay_class {
                RwkvDecayClass::Decay => hyper.weight_decay,
                RwkvDecayClass::NoDecay => 0.0,
            };
            let push = AdamWControlledPush {
                len: u32::try_from(slot.len)
                    .context("device-controlled AdamW tensor length exceeds Vulkan u32")?,
                step: slot_next_step,
                lr: hyper.lr,
                beta1: hyper.beta1,
                beta2: hyper.beta2,
                eps: hyper.eps,
                weight_decay,
                gradient_scale: combined_scale,
                active: u32::from(active),
                apply_control_unscale: u32::from(apply_control_unscale),
            };
            if let Some(mirror) = slot.parameter_storage_mirror.as_ref() {
                if mirror.format() == VulkanParameterStorageFormat::Fp16 {
                    self.adamw_fp16_mirror_controlled.record_dispatch(
                        commands,
                        &[
                            trainable.parameter,
                            gradient,
                            &slot.exp_avg,
                            &slot.exp_avg_sq,
                            mirror.packed_storage(),
                            control,
                        ],
                        bytemuck::bytes_of(&push),
                        [div_ceil_u32(mirror.packed_words(), 256), 1, 1],
                    )?;
                } else {
                    self.adamw_controlled.record_dispatch(
                        commands,
                        &[
                            trainable.parameter,
                            gradient,
                            &slot.exp_avg,
                            &slot.exp_avg_sq,
                            control,
                        ],
                        bytemuck::bytes_of(&push),
                        [div_ceil_u32(slot.len, 256), 1, 1],
                    )?;
                    parameter_mirror_refresher
                        .context("optimizer mirror exists without a matching refresher")?
                        .record_refresh(commands, trainable.parameter, mirror)?;
                }
            } else {
                self.adamw_controlled.record_dispatch(
                    commands,
                    &[
                        trainable.parameter,
                        gradient,
                        &slot.exp_avg,
                        &slot.exp_avg_sq,
                        control,
                    ],
                    bytemuck::bytes_of(&push),
                    [div_ceil_u32(slot.len, 256), 1, 1],
                )?;
            }
        }

        Ok(RwkvDeviceControlledStepPending {
            previous_step: Some(previous_step),
            next_step: Some(next_step),
            active_slot_indices,
            optimizer_generation,
            generation_committed: false,
            device_steps_committed: false,
            tensor_count: self.slots.len(),
        })
    }

    /// Record a whole-model AdamW step gated directly by the robust clipping
    /// pass's non-finite flag. This is the single-queue FP32 analogue of the
    /// GradScaler-controlled path: `nonfinite_flag[0] == 0` commits every
    /// active tensor, while a non-zero flag clears the pending gradients and
    /// leaves parameters/moments untouched. Host step metadata is finalized
    /// only after the already-predicated Vulkan work has completed.
    pub(crate) fn record_device_controlled_nonfinite_step(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        nonfinite_flag: &GpuBuffer,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
    ) -> Result<RwkvDeviceControlledStepPending> {
        hyper.validate()?;
        self.validate_registry(trainables)?;
        if nonfinite_flag.f32_capacity() < 1 {
            bail!("device-controlled AdamW non-finite flag is empty");
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let inactive = self.effective_inactive_names(inactive_names)?;

        let previous_step = self.step;
        let next_step = previous_step
            .checked_add(1)
            .context("RWKV device-controlled non-finite AdamW step overflow")?;
        let optimizer_generation = self.generation_guard.current_generation();
        self.generation_guard
            .wait_for_readers_to_retire(optimizer_generation)?;
        let mut active_slot_indices = Vec::with_capacity(self.slots.len());

        for (slot_index, (slot, trainable)) in self.slots.iter().zip(trainables).enumerate() {
            let active = !inactive.contains(slot.name.as_str());
            let slot_next_step = if active {
                active_slot_indices.push(slot_index);
                slot.step
                    .checked_add(1)
                    .with_context(|| format!("RWKV AdamW slot {:?} step overflow", slot.name))?
            } else {
                slot.step.saturating_add(1).max(1)
            };
            let gradient = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            let weight_decay = match slot.decay_class {
                RwkvDecayClass::Decay => hyper.weight_decay,
                RwkvDecayClass::NoDecay => 0.0,
            };
            let push = AdamWRangeControlledPush {
                offset: 0,
                len: u32::try_from(slot.len)
                    .context("device-controlled AdamW tensor length exceeds Vulkan u32")?,
                step: slot_next_step,
                lr: hyper.lr,
                beta1: hyper.beta1,
                beta2: hyper.beta2,
                eps: hyper.eps,
                weight_decay,
                is_active: u32::from(active),
            };
            self.adamw_range_controlled.record_dispatch(
                commands,
                &[
                    trainable.parameter,
                    gradient,
                    &slot.exp_avg,
                    &slot.exp_avg_sq,
                    nonfinite_flag,
                ],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(slot.len, 256), 1, 1],
            )?;
        }

        // The non-finite-controlled range kernel intentionally updates the
        // canonical FP32 masters first. Refresh any compact execution mirrors
        // behind those writes in the same command buffer; on a skipped window
        // this simply republishes the unchanged masters.
        self.record_refresh_parameter_mirrors(commands, trainables)?;

        Ok(RwkvDeviceControlledStepPending {
            previous_step: Some(previous_step),
            next_step: Some(next_step),
            active_slot_indices,
            optimizer_generation,
            generation_committed: false,
            device_steps_committed: false,
            tensor_count: self.slots.len(),
        })
    }

    pub(crate) fn finalize_device_controlled_step(
        &mut self,
        pending: RwkvDeviceControlledStepPending,
        stepped: bool,
    ) -> Result<RwkvOptimizerStepResult> {
        let RwkvDeviceControlledStepPending {
            previous_step,
            next_step,
            active_slot_indices,
            optimizer_generation,
            generation_committed,
            device_steps_committed,
            tensor_count,
        } = pending;
        let previous_step = previous_step.context(
            "device-controlled AdamW host metadata was deferred; synchronize device step metadata before host finalization",
        )?;
        let next_step = next_step.context(
            "device-controlled AdamW host metadata was deferred; synchronize device step metadata before host finalization",
        )?;
        if self.step != previous_step {
            bail!(
                "device-controlled AdamW host step changed before finalize: expected {}, found {}",
                previous_step,
                self.step
            );
        }
        if stepped {
            for slot_index in active_slot_indices {
                let slot = self
                    .slots
                    .get_mut(slot_index)
                    .context("device-controlled AdamW active slot index is out of range")?;
                slot.step = slot
                    .step
                    .checked_add(1)
                    .with_context(|| format!("RWKV AdamW slot {:?} step overflow", slot.name))?;
                if !device_steps_committed {
                    slot.device_step_authoritative = false;
                }
            }
            self.step = next_step;
            if !device_steps_committed {
                self.device_step_authoritative = false;
            }
        }
        if !generation_committed {
            if stepped {
                self.generation_guard
                    .advance_after_mutation(optimizer_generation)?;
            } else {
                self.generation_guard
                    .release_after_skipped_mutation(optimizer_generation)?;
            }
        }
        self.host_step_metadata_authoritative = true;
        Ok(RwkvOptimizerStepResult {
            step: self.step,
            tensor_count,
        })
    }

    /// Consume a queue-resident GradScaler wavefront without resolving its
    /// finite/overflow decision on the CPU. Parameter mutation, generation
    /// retirement, and both global/per-parameter Adam clocks have already been
    /// predicated by the Vulkan control buffer. The Rust step counters become a
    /// deliberately stale cache until an explicit checkpoint/telemetry sync.
    pub(crate) fn defer_device_controlled_step_host_metadata(
        &mut self,
        pending: RwkvDeviceControlledStepPending,
    ) -> Result<()> {
        anyhow::ensure!(
            pending.generation_committed,
            "cannot defer device-controlled AdamW metadata before its hazard generation is committed"
        );
        anyhow::ensure!(
            pending.device_steps_committed,
            "cannot defer device-controlled AdamW metadata when its step clocks were not updated on device"
        );
        self.host_step_metadata_authoritative = false;
        Ok(())
    }

    fn record_step_with_gradient_override_and_inactive_names(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
    ) -> Result<RwkvOptimizerStepResult> {
        hyper.validate()?;
        self.validate_registry(trainables)?;
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let inactive = self.effective_inactive_names(inactive_names)?;
        let next_step = self
            .step
            .checked_add(1)
            .context("RWKV AdamW step overflow")?;
        // A replica transport source is a read lease over the exact canonical
        // parameter/moment buffers that AdamW mutates. Waiting here (rather
        // than at forward/backward entry) deliberately permits the next
        // gradient window to compute while the preceding generation is still
        // being drained to replicas, but prevents the mutation itself from
        // crossing that captured generation.
        let optimizer_generation = self.generation_guard.current_generation();
        self.generation_guard
            .wait_for_readers_to_retire(optimizer_generation)?;
        let parameter_mirror_refresher = self.parameter_mirror_refresher.as_ref();
        for (slot, trainable) in self.slots.iter_mut().zip(trainables) {
            if inactive.contains(slot.name.as_str()) {
                continue;
            }
            let gradient = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            let slot_next_step = slot
                .step
                .checked_add(1)
                .with_context(|| format!("RWKV AdamW slot {:?} step overflow", slot.name))?;
            let weight_decay = match slot.decay_class {
                RwkvDecayClass::Decay => hyper.weight_decay,
                RwkvDecayClass::NoDecay => 0.0,
            };
            let push = AdamWPush {
                len: slot.len as u32,
                step: slot_next_step,
                lr: hyper.lr,
                beta1: hyper.beta1,
                beta2: hyper.beta2,
                eps: hyper.eps,
                weight_decay,
            };
            if let Some(mirror) = slot.parameter_storage_mirror.as_ref() {
                if mirror.format() == VulkanParameterStorageFormat::Fp16 {
                    self.adamw_fp16_mirror.record_dispatch(
                        commands,
                        &[
                            trainable.parameter,
                            gradient,
                            &slot.exp_avg,
                            &slot.exp_avg_sq,
                            mirror.packed_storage(),
                        ],
                        bytemuck::bytes_of(&push),
                        [div_ceil_u32(mirror.packed_words(), 256), 1, 1],
                    )?;
                } else {
                    self.adamw.record_dispatch(
                        commands,
                        &[
                            trainable.parameter,
                            gradient,
                            &slot.exp_avg,
                            &slot.exp_avg_sq,
                        ],
                        bytemuck::bytes_of(&push),
                        [div_ceil_u32(slot.len, 256), 1, 1],
                    )?;
                    parameter_mirror_refresher
                        .context("optimizer mirror exists without a matching refresher")?
                        .record_refresh(commands, trainable.parameter, mirror)?;
                }
            } else {
                self.adamw.record_dispatch(
                    commands,
                    &[
                        trainable.parameter,
                        gradient,
                        &slot.exp_avg,
                        &slot.exp_avg_sq,
                    ],
                    bytemuck::bytes_of(&push),
                    [div_ceil_u32(slot.len, 256), 1, 1],
                )?;
            }
            slot.step = slot_next_step;
            slot.device_step_authoritative = false;
        }
        self.step = next_step;
        self.device_step_authoritative = false;
        self.host_step_metadata_authoritative = true;
        self.generation_guard
            .advance_after_mutation(optimizer_generation)?;
        Ok(RwkvOptimizerStepResult {
            step: next_step,
            tensor_count: self.slots.len(),
        })
    }

    /// Execute AdamW as a tensor-range wavefront behind an in-flight replica
    /// broadcast. Each canonical state chunk waits only for its own parameter,
    /// exp_avg, and exp_avg_sq readers to retire before mutation. No second
    /// model-sized parameter or optimizer generation is allocated.
    ///
    /// The caller must have already applied any gradient normalization/unscale
    /// operations. `max_chunk_values` must match the chunk geometry used when
    /// the closed replica source was split into range-retirement consumers.
    pub(crate) fn step_wavefront_with_named_gradient_override_and_inactive_names(
        &mut self,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
        max_chunk_values: usize,
    ) -> Result<(RwkvOptimizerStepResult, usize, usize)> {
        hyper.validate()?;
        self.validate_registry(trainables)?;
        if max_chunk_values == 0 {
            bail!("AdamW wavefront chunk size must be positive");
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let inactive = self.effective_inactive_names(inactive_names)?;

        let next_step = self
            .step
            .checked_add(1)
            .context("RWKV AdamW step overflow")?;
        let optimizer_generation = self.generation_guard.current_generation();
        let range_count = self.slots.iter().try_fold(0usize, |count, slot| {
            count
                .checked_add(slot.len.div_ceil(max_chunk_values))
                .context("AdamW wavefront range count overflow")
        })?;
        let mut range_index = 0usize;
        let mut queue_submissions = 0usize;
        let mut pending_submissions = Vec::<vulkan::SubmittedComputeBatch>::new();
        let predeclared_schedule = self.generation_guard.predeclared_gpu_range_schedule(
            optimizer_generation,
            range_count,
            ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
        )?;

        while range_index < range_count {
            let (ready_end, gpu_waits) = if let Some(schedule) = predeclared_schedule.as_ref() {
                schedule.range_run(range_index)?
            } else if let Some(ready) = self.generation_guard.try_ready_range_run_with_gpu_waits(
                optimizer_generation,
                range_index,
                range_count,
                ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
            )? {
                ready
            } else {
                // A host-staged/opaque peer (or a device-group worker that has
                // not yet submitted its source copy) still owns one of the
                // required ranges. Preserve the legacy retirement wait for that
                // fallback, then collect any device-group timeline dependencies
                // that were published alongside it.
                let ready_end = self.generation_guard.wait_for_ready_range_run(
                    optimizer_generation,
                    range_index,
                    range_count,
                    ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
                )?;
                let waits = self.generation_guard.gpu_waits_for_retired_range_run(
                    optimizer_generation,
                    range_index,
                    ready_end,
                )?;
                (ready_end, waits)
            };
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            let mut recorded_dispatch = false;
            let mut current_range = 0usize;

            'slots: for (slot, trainable) in self.slots.iter().zip(trainables) {
                let slot_range_count = slot.len.div_ceil(max_chunk_values);
                let slot_range_end = current_range
                    .checked_add(slot_range_count)
                    .context("AdamW wavefront tensor range end overflow")?;
                if slot_range_end <= range_index {
                    current_range = slot_range_end;
                    continue;
                }
                if current_range >= ready_end {
                    break;
                }

                let slot_next_step = slot
                    .step
                    .checked_add(1)
                    .with_context(|| format!("RWKV AdamW slot {:?} step overflow", slot.name))?;
                let weight_decay = match slot.decay_class {
                    RwkvDecayClass::Decay => hyper.weight_decay,
                    RwkvDecayClass::NoDecay => 0.0,
                };
                let gradient = gradient_override
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, gradient)| gradient)
                    .unwrap_or(&slot.accumulated_grad);

                let first_slot_range = range_index.saturating_sub(current_range);
                let last_slot_range = ready_end.min(slot_range_end) - current_range;
                if !inactive.contains(slot.name.as_str()) {
                    for slot_range in first_slot_range..last_slot_range {
                        let offset = slot_range
                            .checked_mul(max_chunk_values)
                            .context("AdamW wavefront range offset overflow")?;
                        let len = max_chunk_values.min(slot.len - offset);
                        let push = AdamWRangePush {
                            offset: u32::try_from(offset)
                                .context("AdamW wavefront range offset exceeds Vulkan u32")?,
                            len: u32::try_from(len)
                                .context("AdamW wavefront range length exceeds Vulkan u32")?,
                            step: slot_next_step,
                            lr: hyper.lr,
                            beta1: hyper.beta1,
                            beta2: hyper.beta2,
                            eps: hyper.eps,
                            weight_decay,
                        };
                        self.adamw_range.record_dispatch(
                            &mut commands,
                            &[
                                trainable.parameter,
                                gradient,
                                &slot.exp_avg,
                                &slot.exp_avg_sq,
                            ],
                            bytemuck::bytes_of(&push),
                            [div_ceil_u32(len, 256), 1, 1],
                        )?;
                        recorded_dispatch = true;
                    }
                }
                current_range = slot_range_end;
                if current_range >= ready_end {
                    break 'slots;
                }
            }

            if recorded_dispatch || !gpu_waits.is_empty() {
                // Even a fully inactive range must consume its published GPU
                // retirement waits before the generation can advance. It does
                // not mutate this step, but a later generation may reactivate
                // that tensor and must never race an older broadcast read.
                // Keep the submission in flight instead of waiting its fence on
                // the host. Predeclared device-group ranges can therefore queue
                // the whole bounded AdamW wavefront behind future retirement
                // values before broadcast workers reach those ranges. All of
                // these batches use the same primary queue, so their FIFO order
                // preserves range mutation order while independent replica
                // source queues make progress toward the timeline signals.
                pending_submissions
                    .push(commands.submit_async_wait_device_group_timeline(&gpu_waits)?);
                queue_submissions = queue_submissions
                    .checked_add(1)
                    .context("AdamW wavefront queue-submission count overflow")?;
            }
            range_index = ready_end;
        }

        for slot in &mut self.slots {
            if !inactive.contains(slot.name.as_str()) {
                slot.step = slot
                    .step
                    .checked_add(1)
                    .with_context(|| format!("RWKV AdamW slot {:?} step overflow", slot.name))?;
                slot.device_step_authoritative = false;
            }
        }

        // The range kernel deliberately keeps compact execution mirrors out of
        // the DMA/AdamW critical path. Refresh them once after all FP32 masters
        // have advanced; this preserves the portable canonical optimizer state
        // and avoids packed-half races at odd range boundaries.
        if self
            .slots
            .iter()
            .any(|slot| slot.parameter_storage_mirror.is_some())
        {
            let mut refresh = vulkan::ComputeBatch::new(&self.device)?;
            self.record_refresh_parameter_mirrors(&mut refresh, trainables)?;
            // The refresh shares the primary queue with every AdamW range, so
            // queue order is the dependency. Enqueue it now and use its fence
            // as the tail drain when one is present.
            pending_submissions.push(refresh.submit_async()?);
            queue_submissions = queue_submissions
                .checked_add(1)
                .context("AdamW wavefront mirror-refresh submission count overflow")?;
        }

        self.step = next_step;
        self.device_step_authoritative = false;
        self.host_step_metadata_authoritative = true;
        if let Some(completion) = pending_submissions
            .last()
            .and_then(vulkan::SubmittedComputeBatch::timeline_wait)
        {
            // Queue FIFO makes the newest timeline value the completion point
            // for the entire AdamW wavefront and optional mirror refresh. The
            // next logical generation may therefore be published immediately;
            // independent replica source queues inherit this wait through their
            // generation lease instead of forcing a CPU fence drain here.
            drop(pending_submissions);
            self.generation_guard
                .advance_after_mutation_after_submission(optimizer_generation, completion)?;
        } else {
            // Fence-only devices cannot export a lightweight queue dependency.
            // Preserve the conservative host retirement path for that legacy
            // backend while timeline-capable training stays queue-resident.
            while let Some(submission) = pending_submissions.pop() {
                submission.wait()?;
            }
            self.generation_guard
                .advance_after_mutation(optimizer_generation)?;
        }
        Ok((
            RwkvOptimizerStepResult {
                step: next_step,
                tensor_count: self.slots.len(),
            },
            range_index,
            queue_submissions,
        ))
    }

    /// Queue range-addressed AdamW behind replica-retirement dependencies while
    /// taking the finite/overflow decision from a device-resident flag. Host
    /// Adam-step metadata remains pending until the caller observes the flag,
    /// but timeline-capable wavefronts publish the next hazard generation as
    /// soon as their GPU tail is queued. A skipped window therefore advances
    /// only the hazard epoch, never the numerical Adam clock.
    pub(crate) fn step_wavefront_device_controlled_nonfinite_with_named_gradient_override_and_inactive_names(
        &mut self,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        nonfinite_flag: &GpuBuffer,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
        max_chunk_values: usize,
    ) -> Result<(RwkvDeviceControlledStepPending, usize, usize)> {
        hyper.validate()?;
        self.validate_registry(trainables)?;
        if max_chunk_values == 0 {
            bail!("device-controlled AdamW wavefront chunk size must be positive");
        }
        if nonfinite_flag.f32_capacity() < 1 {
            bail!("device-controlled AdamW wavefront non-finite flag is empty");
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let inactive = self.effective_inactive_names(inactive_names)?;

        let previous_step = self.step;
        let next_step = previous_step
            .checked_add(1)
            .context("RWKV device-controlled AdamW wavefront step overflow")?;
        let optimizer_generation = self.generation_guard.current_generation();
        let range_count = self.slots.iter().try_fold(0usize, |count, slot| {
            count
                .checked_add(slot.len.div_ceil(max_chunk_values))
                .context("device-controlled AdamW wavefront range count overflow")
        })?;
        let active_slot_indices = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| (!inactive.contains(slot.name.as_str())).then_some(index))
            .collect::<Vec<_>>();
        let mut range_index = 0usize;
        let mut queue_submissions = 0usize;
        let mut pending_submissions = Vec::<vulkan::SubmittedComputeBatch>::new();
        let predeclared_schedule = self.generation_guard.predeclared_gpu_range_schedule(
            optimizer_generation,
            range_count,
            ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
        )?;

        while range_index < range_count {
            let (ready_end, gpu_waits) = if let Some(schedule) = predeclared_schedule.as_ref() {
                schedule.range_run(range_index)?
            } else if let Some(ready) = self.generation_guard.try_ready_range_run_with_gpu_waits(
                optimizer_generation,
                range_index,
                range_count,
                ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
            )? {
                ready
            } else {
                let ready_end = self.generation_guard.wait_for_ready_range_run(
                    optimizer_generation,
                    range_index,
                    range_count,
                    ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
                )?;
                let waits = self.generation_guard.gpu_waits_for_retired_range_run(
                    optimizer_generation,
                    range_index,
                    ready_end,
                )?;
                (ready_end, waits)
            };
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            let mut current_range = 0usize;

            'slots: for (slot, trainable) in self.slots.iter().zip(trainables) {
                let slot_range_count = slot.len.div_ceil(max_chunk_values);
                let slot_range_end = current_range
                    .checked_add(slot_range_count)
                    .context("device-controlled AdamW wavefront tensor range end overflow")?;
                if slot_range_end <= range_index {
                    current_range = slot_range_end;
                    continue;
                }
                if current_range >= ready_end {
                    break;
                }

                let active = !inactive.contains(slot.name.as_str());
                let slot_next_step = slot
                    .step
                    .checked_add(1)
                    .with_context(|| format!("RWKV AdamW slot {:?} step overflow", slot.name))?;
                let weight_decay = match slot.decay_class {
                    RwkvDecayClass::Decay => hyper.weight_decay,
                    RwkvDecayClass::NoDecay => 0.0,
                };
                let gradient = gradient_override
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, gradient)| gradient)
                    .unwrap_or(&slot.accumulated_grad);

                let first_slot_range = range_index.saturating_sub(current_range);
                let last_slot_range = ready_end.min(slot_range_end) - current_range;
                for slot_range in first_slot_range..last_slot_range {
                    let offset = slot_range
                        .checked_mul(max_chunk_values)
                        .context("device-controlled AdamW wavefront range offset overflow")?;
                    let len = max_chunk_values.min(slot.len - offset);
                    let push = AdamWRangeControlledPush {
                        offset: u32::try_from(offset).context(
                            "device-controlled AdamW wavefront range offset exceeds Vulkan u32",
                        )?,
                        len: u32::try_from(len).context(
                            "device-controlled AdamW wavefront range length exceeds Vulkan u32",
                        )?,
                        step: slot_next_step,
                        lr: hyper.lr,
                        beta1: hyper.beta1,
                        beta2: hyper.beta2,
                        eps: hyper.eps,
                        weight_decay,
                        is_active: u32::from(active),
                    };
                    self.adamw_range_controlled.record_dispatch(
                        &mut commands,
                        &[
                            trainable.parameter,
                            gradient,
                            &slot.exp_avg,
                            &slot.exp_avg_sq,
                            nonfinite_flag,
                        ],
                        bytemuck::bytes_of(&push),
                        [div_ceil_u32(len, 256), 1, 1],
                    )?;
                }
                current_range = slot_range_end;
                if current_range >= ready_end {
                    break 'slots;
                }
            }

            pending_submissions.push(commands.submit_async_wait_device_group_timeline(&gpu_waits)?);
            queue_submissions = queue_submissions
                .checked_add(1)
                .context("device-controlled AdamW wavefront queue-submission count overflow")?;
            range_index = ready_end;
        }

        if self
            .slots
            .iter()
            .any(|slot| slot.parameter_storage_mirror.is_some())
        {
            let mut refresh = vulkan::ComputeBatch::new(&self.device)?;
            self.record_refresh_parameter_mirrors(&mut refresh, trainables)?;
            pending_submissions.push(refresh.submit_async()?);
            queue_submissions = queue_submissions.checked_add(1).context(
                "device-controlled AdamW wavefront mirror-refresh submission count overflow",
            )?;
        }

        if let Some(completion) = pending_submissions
            .last()
            .and_then(vulkan::SubmittedComputeBatch::timeline_wait)
        {
            // The hazard generation is not the Adam successful-step counter.
            // Publishing it here is safe for both mutation and skip: the next
            // replica source inherits this tail dependency and cannot read the
            // new epoch until the predicated AdamW/gradient-clear work finishes.
            drop(pending_submissions);
            self.generation_guard
                .advance_after_mutation_after_submission(optimizer_generation, completion)?;
        } else {
            while let Some(submission) = pending_submissions.pop() {
                submission.wait()?;
            }
            self.generation_guard
                .advance_after_mutation(optimizer_generation)?;
        }

        Ok((
            RwkvDeviceControlledStepPending {
                previous_step: Some(previous_step),
                next_step: Some(next_step),
                active_slot_indices,
                optimizer_generation,
                generation_committed: true,
                device_steps_committed: false,
                tensor_count: self.slots.len(),
            },
            range_index,
            queue_submissions,
        ))
    }

    /// GradScaler-controlled range wavefront. `gradient_scale` is the ordinary
    /// sequence-normalization factor; an optional named scale handles the LTM
    /// value projection, while `apply_control_unscale` consumes the reciprocal
    /// stored by the device loss-scale controller. The control buffer decides
    /// step versus skip only when each queued range executes. The hazard
    /// generation is published from the queued GPU tail, so host GradScaler
    /// telemetry is no longer part of replica-generation retirement.
    pub(crate) fn step_wavefront_grad_scaler_controlled_with_named_gradient_override_and_inactive_names(
        &mut self,
        trainables: &[RwkvTrainableRef<'_>],
        hyper: AdamWHyperParams,
        control: &GpuBuffer,
        gradient_scale: f32,
        apply_control_unscale: bool,
        named_gradient_scale: Option<(&str, f32)>,
        gradient_override: Option<(&str, &GpuBuffer)>,
        inactive_names: &[&str],
        max_chunk_values: usize,
    ) -> Result<(RwkvDeviceControlledStepPending, usize, usize)> {
        hyper.validate()?;
        self.validate_registry(trainables)?;
        if control.f32_capacity() < crate::training_numerics::DYNAMIC_LOSS_SCALE_CONTROL_WORDS {
            bail!("GradScaler-controlled AdamW wavefront control buffer is too small");
        }
        if max_chunk_values == 0 {
            bail!("GradScaler-controlled AdamW wavefront chunk size must be positive");
        }
        if !gradient_scale.is_finite() || gradient_scale <= 0.0 {
            bail!(
                "GradScaler-controlled AdamW wavefront gradient scale must be finite and positive"
            );
        }
        if let Some((name, scale)) = named_gradient_scale {
            if !scale.is_finite() || scale <= 0.0 {
                bail!("GradScaler-controlled AdamW wavefront named scale for {name:?} is invalid");
            }
            if !self.slots.iter().any(|slot| slot.name == name) {
                bail!("persistent optimizer has no registered tensor {name:?} for named gradient scaling");
            }
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let inactive = self.effective_inactive_names(inactive_names)?;

        let (previous_step, next_step) = if self.host_step_metadata_authoritative {
            let previous_step = self.step;
            let next_step = previous_step
                .checked_add(1)
                .context("RWKV GradScaler-controlled AdamW wavefront step overflow")?;
            (Some(previous_step), Some(next_step))
        } else {
            (None, None)
        };
        let optimizer_generation = self.generation_guard.current_generation();
        let range_count = self.slots.iter().try_fold(0usize, |count, slot| {
            count
                .checked_add(slot.len.div_ceil(max_chunk_values))
                .context("GradScaler-controlled AdamW wavefront range count overflow")
        })?;
        let active_slot_indices = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| (!inactive.contains(slot.name.as_str())).then_some(index))
            .collect::<Vec<_>>();
        for &slot_index in &active_slot_indices {
            if self.host_step_metadata_authoritative && self.slots[slot_index].step == u32::MAX {
                bail!(
                    "RWKV AdamW slot {:?} step overflow",
                    self.slots[slot_index].name
                );
            }
        }
        let seed_device_step = !self.device_step_authoritative;
        let seeded_device_step_slots = active_slot_indices
            .iter()
            .copied()
            .filter(|&slot_index| !self.slots[slot_index].device_step_authoritative)
            .collect::<Vec<_>>();
        let mut range_index = 0usize;
        let mut queue_submissions = 0usize;
        let mut pending_submissions = Vec::<vulkan::SubmittedComputeBatch>::new();
        let predeclared_schedule = self.generation_guard.predeclared_gpu_range_schedule(
            optimizer_generation,
            range_count,
            ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
        )?;

        while range_index < range_count {
            let (ready_end, gpu_waits) = if let Some(schedule) = predeclared_schedule.as_ref() {
                schedule.range_run(range_index)?
            } else if let Some(ready) = self.generation_guard.try_ready_range_run_with_gpu_waits(
                optimizer_generation,
                range_index,
                range_count,
                ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
            )? {
                ready
            } else {
                let ready_end = self.generation_guard.wait_for_ready_range_run(
                    optimizer_generation,
                    range_index,
                    range_count,
                    ADAMW_WAVEFRONT_MIN_COALESCED_RANGES,
                )?;
                let waits = self.generation_guard.gpu_waits_for_retired_range_run(
                    optimizer_generation,
                    range_index,
                    ready_end,
                )?;
                (ready_end, waits)
            };
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            let first_wavefront_submission = range_index == 0;
            if first_wavefront_submission {
                // Seed only after a host-owned optimizer path or checkpoint
                // restore invalidates a mirror. Once seeded, both the global
                // and per-slot clocks remain device-owned across GradScaler
                // wavefronts until an explicit metadata synchronization.
                if seed_device_step {
                    anyhow::ensure!(
                        self.host_step_metadata_authoritative,
                        "device global AdamW step is stale while host step metadata is not authoritative"
                    );
                    commands.upload_u32(&self.device_step, &[self.step])?;
                }
                for &slot_index in &seeded_device_step_slots {
                    let slot = &self.slots[slot_index];
                    commands.upload_u32(&slot.device_step, &[slot.step])?;
                }
                self.adamw_step_grad_scaler_controlled.record_dispatch(
                    &mut commands,
                    &[control, &self.device_step],
                    &[],
                    [1, 1, 1],
                )?;
                for &slot_index in &active_slot_indices {
                    let slot = &self.slots[slot_index];
                    self.adamw_step_grad_scaler_controlled.record_dispatch(
                        &mut commands,
                        &[control, &slot.device_step],
                        &[],
                        [1, 1, 1],
                    )?;
                }
            }
            let mut current_range = 0usize;

            'slots: for (slot, trainable) in self.slots.iter().zip(trainables) {
                let slot_range_count = slot.len.div_ceil(max_chunk_values);
                let slot_range_end = current_range
                    .checked_add(slot_range_count)
                    .context("GradScaler-controlled AdamW wavefront tensor range end overflow")?;
                if slot_range_end <= range_index {
                    current_range = slot_range_end;
                    continue;
                }
                if current_range >= ready_end {
                    break;
                }

                let active = !inactive.contains(slot.name.as_str());
                let weight_decay = match slot.decay_class {
                    RwkvDecayClass::Decay => hyper.weight_decay,
                    RwkvDecayClass::NoDecay => 0.0,
                };
                let gradient = gradient_override
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, gradient)| gradient)
                    .unwrap_or(&slot.accumulated_grad);
                let per_tensor_scale = named_gradient_scale
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, scale)| scale)
                    .unwrap_or(1.0);
                let combined_scale = gradient_scale * per_tensor_scale;
                if !combined_scale.is_finite() || combined_scale <= 0.0 {
                    bail!(
                        "GradScaler-controlled AdamW wavefront combined scale for {:?} is invalid",
                        slot.name
                    );
                }

                let first_slot_range = range_index.saturating_sub(current_range);
                let last_slot_range = ready_end.min(slot_range_end) - current_range;
                for slot_range in first_slot_range..last_slot_range {
                    let offset = slot_range
                        .checked_mul(max_chunk_values)
                        .context("GradScaler-controlled AdamW wavefront range offset overflow")?;
                    let len = max_chunk_values.min(slot.len - offset);
                    let push = AdamWRangeGradScalerControlledPush {
                        offset: u32::try_from(offset).context(
                            "GradScaler-controlled AdamW wavefront range offset exceeds Vulkan u32",
                        )?,
                        len: u32::try_from(len).context(
                            "GradScaler-controlled AdamW wavefront range length exceeds Vulkan u32",
                        )?,
                        lr: hyper.lr,
                        beta1: hyper.beta1,
                        beta2: hyper.beta2,
                        eps: hyper.eps,
                        weight_decay,
                        gradient_scale: combined_scale,
                        is_active: u32::from(active),
                        apply_control_unscale: u32::from(apply_control_unscale),
                    };
                    self.adamw_range_grad_scaler_controlled.record_dispatch(
                        &mut commands,
                        &[
                            trainable.parameter,
                            gradient,
                            &slot.exp_avg,
                            &slot.exp_avg_sq,
                            control,
                            &slot.device_step,
                        ],
                        bytemuck::bytes_of(&push),
                        [div_ceil_u32(len, 256), 1, 1],
                    )?;
                }
                current_range = slot_range_end;
                if current_range >= ready_end {
                    break 'slots;
                }
            }

            pending_submissions.push(commands.submit_async_wait_device_group_timeline(&gpu_waits)?);
            if first_wavefront_submission {
                self.device_step_authoritative = true;
                for &slot_index in &seeded_device_step_slots {
                    self.slots[slot_index].device_step_authoritative = true;
                }
            }
            queue_submissions = queue_submissions
                .checked_add(1)
                .context("GradScaler-controlled AdamW wavefront queue-submission count overflow")?;
            range_index = ready_end;
        }

        if self
            .slots
            .iter()
            .any(|slot| slot.parameter_storage_mirror.is_some())
        {
            let mut refresh = vulkan::ComputeBatch::new(&self.device)?;
            self.record_refresh_parameter_mirrors(&mut refresh, trainables)?;
            pending_submissions.push(refresh.submit_async()?);
            queue_submissions = queue_submissions.checked_add(1).context(
                "GradScaler-controlled AdamW wavefront mirror-refresh submission count overflow",
            )?;
        }

        if let Some(completion) = pending_submissions
            .last()
            .and_then(vulkan::SubmittedComputeBatch::timeline_wait)
        {
            drop(pending_submissions);
            self.generation_guard
                .advance_after_mutation_after_submission(optimizer_generation, completion)?;
        } else {
            while let Some(submission) = pending_submissions.pop() {
                submission.wait()?;
            }
            self.generation_guard
                .advance_after_mutation(optimizer_generation)?;
        }

        Ok((
            RwkvDeviceControlledStepPending {
                previous_step,
                next_step,
                active_slot_indices,
                optimizer_generation,
                generation_committed: true,
                device_steps_committed: true,
                tensor_count: self.slots.len(),
            },
            range_index,
            queue_submissions,
        ))
    }

    pub fn record_parameter_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
    ) -> Result<()> {
        self.validate_registry(trainables)?;
        for (slot, trainable) in self.slots.iter().zip(trainables) {
            commands.readback_f32(trainable.parameter, &slot.parameter_readback, slot.len)?;
        }
        Ok(())
    }

    pub fn read_parameter_snapshots(&self) -> Result<Vec<RwkvParameterSnapshot>> {
        self.slots
            .iter()
            .map(|slot| {
                Ok(RwkvParameterSnapshot {
                    name: slot.name.to_string(),
                    values: slot.parameter_readback.read_f32(slot.len)?,
                })
            })
            .collect()
    }

    pub fn state_snapshot(&self) -> Result<AdamWOptimizerState> {
        let (step, slot_steps) = self.exact_step_metadata()?;
        let mut slots = Vec::with_capacity(self.slots.len());
        for (slot, slot_step) in self.slots.iter().zip(slot_steps) {
            slots.push(AdamWOptimizerSlotState {
                name: slot.name.clone(),
                step: slot_step,
                decay_class: Some(slot.decay_class),
                exp_avg: slot.exp_avg.read_f32(slot.len)?,
                exp_avg_sq: slot.exp_avg_sq.read_f32(slot.len)?,
            });
        }
        Ok(AdamWOptimizerState { step, slots })
    }

    /// Snapshot the canonical pending-gradient registry in exact optimizer
    /// order. A named override is used by the tied LM-head fast path, where the
    /// live gradient intentionally resides in the shared embedding/head buffer
    /// instead of the optimizer-owned staging allocation.
    pub(crate) fn gradient_state_snapshot_with_override(
        &self,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<Vec<RwkvParameterSnapshot>> {
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        self.slots
            .iter()
            .map(|slot| {
                let gradient = gradient_override
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, gradient)| gradient)
                    .unwrap_or(&slot.accumulated_grad);
                let values = gradient.read_f32(slot.len)?;
                if values.iter().any(|value| !value.is_finite()) {
                    bail!(
                        "pending gradient registry tensor {:?} contains non-finite values",
                        slot.name
                    );
                }
                Ok(RwkvParameterSnapshot {
                    name: slot.name.clone(),
                    values,
                })
            })
            .collect()
    }

    /// Canonical pending-gradient tensor geometry without reading any tensor
    /// payload back to the host. This is the control-plane half of streamed
    /// cross-adapter reduction: peers can validate exact name/order/length
    /// parity before the first bounded data chunk mutates the destination.
    pub(crate) fn gradient_layout(&self) -> Vec<(String, usize)> {
        self.slots
            .iter()
            .map(|slot| (slot.name.clone(), slot.len))
            .collect()
    }

    /// Capture only the immutable Vulkan buffers and tiny AdamW step metadata
    /// required to broadcast a closed optimizer boundary. This deliberately
    /// severs replica transport from the parent training graph's TBPTT tracing
    /// state while preserving exact parameter/optimizer storage identity.
    pub(crate) fn replica_state_source(
        &self,
        trainables: &[RwkvTrainableRef<'_>],
    ) -> Result<RwkvReplicaStateSource> {
        self.validate_registry(trainables)?;
        let read_lease = self.generation_guard.acquire_read_lease()?;
        let step = if self.host_step_metadata_authoritative || !self.device_step_authoritative {
            RwkvReplicaStepSource::host(self.step)
        } else {
            RwkvReplicaStepSource::device(self.step, &self.device_step)
        };
        let slots = self
            .slots
            .iter()
            .zip(trainables)
            .map(|(slot, trainable)| RwkvReplicaStateSourceSlot {
                name: slot.name.clone(),
                len: slot.len,
                step: if self.host_step_metadata_authoritative || !slot.device_step_authoritative {
                    RwkvReplicaStepSource::host(slot.step)
                } else {
                    RwkvReplicaStepSource::device(slot.step, &slot.device_step)
                },
                decay_class: slot.decay_class,
                parameter: trainable.parameter.clone(),
                exp_avg: slot.exp_avg.clone(),
                exp_avg_sq: slot.exp_avg_sq.clone(),
            })
            .collect();
        Ok(RwkvReplicaStateSource {
            device: self.device.clone(),
            step,
            slots,
            _read_lease: read_lease,
        })
    }

    /// Capture the exact Vulkan buffers that represent this optimizer's open
    /// pending-gradient registry. This is a transport-only snapshot of buffer
    /// identity: no model-sized host payload is materialized.
    pub(crate) fn pending_gradient_source_with_override(
        &self,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<RwkvPendingGradientSource> {
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let slots = self
            .slots
            .iter()
            .map(|slot| {
                let gradient = gradient_override
                    .filter(|(name, _)| *name == slot.name)
                    .map(|(_, gradient)| gradient)
                    .unwrap_or(&slot.accumulated_grad)
                    .clone();
                RwkvPendingGradientSourceSlot {
                    name: slot.name.clone(),
                    len: slot.len,
                    gradient,
                }
            })
            .collect();
        Ok(RwkvPendingGradientSource { slots })
    }

    /// Write one bounded slice of closed-step replica state from a Vulkan
    /// transport window into this optimizer registry. Parameter mirrors are
    /// refreshed once after the complete streamed payload has landed.
    pub(crate) fn record_replica_state_range_write(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
        tensor_index: usize,
        plane: RwkvReplicaStatePlane,
        offset: usize,
        len: usize,
        source: &GpuBuffer,
    ) -> Result<()> {
        self.validate_registry(trainables)?;
        let (slot, trainable) = self
            .slots
            .get(tensor_index)
            .zip(trainables.get(tensor_index))
            .with_context(|| {
                format!("replica-state tensor index {tensor_index} is out of range")
            })?;
        let end = offset
            .checked_add(len)
            .context("replica-state write range overflow")?;
        if len == 0 || end > slot.len {
            bail!(
                "replica-state write range {offset}..{end} is outside tensor {:?} length {}",
                slot.name,
                slot.len
            );
        }
        if len > source.f32_capacity() {
            bail!(
                "replica-state write chunk length {len} exceeds source capacity {}",
                source.f32_capacity()
            );
        }
        let destination = match plane {
            RwkvReplicaStatePlane::Parameter => trainable.parameter,
            RwkvReplicaStatePlane::ExpAvg => &slot.exp_avg,
            RwkvReplicaStatePlane::ExpAvgSq => &slot.exp_avg_sq,
        };
        commands.copy_f32_range(source, 0, destination, offset, len)
    }

    pub(crate) fn record_refresh_parameter_mirrors(
        &self,
        commands: &mut vulkan::ComputeBatch,
        trainables: &[RwkvTrainableRef<'_>],
    ) -> Result<()> {
        self.validate_registry(trainables)?;
        if !self
            .slots
            .iter()
            .any(|slot| slot.parameter_storage_mirror.is_some())
        {
            return Ok(());
        }
        let refresher = self
            .parameter_mirror_refresher
            .as_ref()
            .context("mixed-precision parameter mirrors have no refresher")?;
        for (slot, trainable) in self.slots.iter().zip(trainables) {
            if let Some(mirror) = slot.parameter_storage_mirror.as_ref() {
                refresher.record_refresh(commands, trainable.parameter, mirror)?;
            }
        }
        Ok(())
    }

    /// Copy the source generation's packed Adam clocks into this replica's
    /// device-resident clock buffers. Any source word that is already known on
    /// the host keeps the cheap host mirror path; device-authoritative words
    /// remain device-authoritative on the replica and never cross a CPU readback.
    pub(crate) fn record_replica_step_metadata_write(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        source: &RwkvReplicaStateSource,
        packed: &GpuBuffer,
    ) -> Result<()> {
        if self.slots.len() != source.slots.len() {
            bail!(
                "replica optimizer registry has {} slots; source has {}",
                self.slots.len(),
                source.slots.len()
            );
        }
        if packed.f32_capacity() < source.step_metadata_word_count() {
            bail!(
                "replica packed step metadata has capacity {}, expected at least {} words",
                packed.f32_capacity(),
                source.step_metadata_word_count()
            );
        }
        for (index, (destination, source_slot)) in self.slots.iter().zip(&source.slots).enumerate()
        {
            if destination.name != source_slot.name || destination.len != source_slot.len {
                bail!(
                    "replica optimizer registry mismatch at slot {index}: destination={:?}/{} source={:?}/{}",
                    destination.name,
                    destination.len,
                    source_slot.name,
                    source_slot.len
                );
            }
        }

        let mut any_device_authoritative = false;
        if source.step.is_device_authoritative() {
            commands.copy_f32_range(packed, 0, &self.device_step, 0, 1)?;
            self.device_step_authoritative = true;
            any_device_authoritative = true;
        } else {
            self.step = source.step.host_value;
            self.device_step_authoritative = false;
        }
        for (slot_index, (destination, source_slot)) in
            self.slots.iter_mut().zip(&source.slots).enumerate()
        {
            if source_slot.step.is_device_authoritative() {
                commands.copy_f32_range(packed, slot_index + 1, &destination.device_step, 0, 1)?;
                destination.device_step_authoritative = true;
                any_device_authoritative = true;
            } else {
                destination.step = source_slot.step.host_value;
                destination.device_step_authoritative = false;
            }
        }
        self.host_step_metadata_authoritative = !any_device_authoritative;
        Ok(())
    }

    /// Add one bounded source chunk into a range of the destination canonical
    /// pending-gradient registry. The existing offset-zero accumulation shader
    /// is intentionally reused: the destination range is copied into a bounded
    /// scratch slice, accumulated there on Vulkan, then copied back in place.
    /// This keeps the data plane GPU-native without requiring a model-sized
    /// staging allocation or a new shader ABI.
    pub(crate) fn record_accumulate_gradient_range_with_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        tensor_index: usize,
        offset: usize,
        source: &GpuBuffer,
        scratch: &GpuBuffer,
        len: usize,
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<()> {
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        let slot = self.slots.get(tensor_index).with_context(|| {
            format!("pending gradient tensor index {tensor_index} is out of range")
        })?;
        let end = offset
            .checked_add(len)
            .context("pending gradient accumulation range overflow")?;
        if len == 0 || end > slot.len {
            bail!(
                "pending gradient accumulation range {offset}..{end} is outside tensor {:?} length {}",
                slot.name,
                slot.len
            );
        }
        if len > source.f32_capacity() || len > scratch.f32_capacity() {
            bail!(
                "pending gradient accumulation chunk length {len} exceeds source/scratch capacities {}/{}",
                source.f32_capacity(),
                scratch.f32_capacity()
            );
        }
        let destination = gradient_override
            .filter(|(name, _)| *name == slot.name)
            .map(|(_, gradient)| gradient)
            .unwrap_or(&slot.accumulated_grad);
        commands.copy_f32_range(destination, offset, scratch, 0, len)?;
        let push = LenPush {
            len: u32::try_from(len).context("pending gradient chunk exceeds Vulkan u32 length")?,
        };
        self.gradient_accumulate.record_dispatch(
            commands,
            &[scratch, source],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(len, 256), 1, 1],
        )?;
        commands.copy_f32_range(scratch, 0, destination, offset, len)?;
        Ok(())
    }

    /// Restore a complete pending-gradient registry snapshot. The checkpoint is
    /// canonical by parameter name, so the LM-head entry can be materialized
    /// into either the optimizer slot or the shared tied-gradient allocation on
    /// resume without changing the portable file format.
    pub(crate) fn load_gradient_state_with_override(
        &self,
        state: &[RwkvParameterSnapshot],
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<()> {
        if state.len() != self.slots.len() {
            bail!(
                "pending gradient state has {} slots; registry requires {}",
                state.len(),
                self.slots.len()
            );
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }
        for (index, (slot, saved)) in self.slots.iter().zip(state).enumerate() {
            if saved.name != slot.name {
                bail!(
                    "pending gradient state slot {index} is {:?}; registry requires {:?}",
                    saved.name,
                    slot.name
                );
            }
            if saved.values.len() != slot.len {
                bail!(
                    "pending gradient state {:?} has {} values; expected {}",
                    saved.name,
                    saved.values.len(),
                    slot.len
                );
            }
            if saved.values.iter().any(|value| !value.is_finite()) {
                bail!(
                    "pending gradient state {:?} contains non-finite values",
                    saved.name
                );
            }
            let gradient = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            gradient.write_f32(&saved.values)?;
        }
        Ok(())
    }

    /// Add a canonical host-side pending-gradient snapshot into the live
    /// Vulkan accumulation registry. The snapshot is transported through host
    /// memory today, but the reduction arithmetic itself is performed by the
    /// same Vulkan accumulation kernel used by ordinary graph backward passes.
    ///
    /// This is the first cross-device reduction seam: another Vulkan replica
    /// can read its canonical named registry, and this device can accumulate it
    /// without teaching either side a device-specific tensor layout.
    pub(crate) fn record_accumulate_gradient_state_with_override(
        &self,
        commands: &mut vulkan::ComputeBatch,
        state: &[RwkvParameterSnapshot],
        gradient_override: Option<(&str, &GpuBuffer)>,
    ) -> Result<Vec<GpuBuffer>> {
        if state.len() != self.slots.len() {
            bail!(
                "pending gradient shard has {} slots; registry requires {}",
                state.len(),
                self.slots.len()
            );
        }
        if let Some((name, gradient)) = gradient_override {
            self.validate_gradient_override(name, gradient)?;
        }

        // Validate the complete shard before recording any mutation so a bad
        // replica cannot partially contaminate the destination window.
        for (index, (slot, saved)) in self.slots.iter().zip(state).enumerate() {
            if saved.name != slot.name {
                bail!(
                    "pending gradient shard slot {index} is {:?}; registry requires {:?}",
                    saved.name,
                    slot.name
                );
            }
            if saved.values.len() != slot.len {
                bail!(
                    "pending gradient shard {:?} has {} values; expected {}",
                    saved.name,
                    saved.values.len(),
                    slot.len
                );
            }
            if saved.values.iter().any(|value| !value.is_finite()) {
                bail!(
                    "pending gradient shard {:?} contains non-finite values",
                    saved.name
                );
            }
        }

        // Keep every transient source alive until submit; descriptor sets
        // reference these Vulkan allocations after record_dispatch returns.
        // The buffers themselves come from the device timeline arena and are
        // returned to its reusable pool when the final owner drops.
        let mut uploaded = Vec::with_capacity(state.len());
        for (slot, saved) in self.slots.iter().zip(state) {
            let source = GpuBuffer::transient_f32(&self.device, saved.values.len())?;
            commands.upload_f32(&source, &saved.values)?;
            let destination = gradient_override
                .filter(|(name, _)| *name == slot.name)
                .map(|(_, gradient)| gradient)
                .unwrap_or(&slot.accumulated_grad);
            let push = LenPush {
                len: slot.len as u32,
            };
            self.gradient_accumulate.record_dispatch(
                commands,
                &[destination, &source],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(slot.len, 256), 1, 1],
            )?;
            uploaded.push(source);
        }
        Ok(uploaded)
    }

    /// Replace all canonical FP32 model masters from a named snapshot and
    /// refresh any compact execution mirrors (for example FP16 storage) before
    /// returning. AdamW moments/steps are deliberately untouched; callers use
    /// `load_state` separately when synchronizing a complete replica.
    pub(crate) fn load_parameter_snapshots(
        &self,
        trainables: &[RwkvTrainableRef<'_>],
        state: &[RwkvParameterSnapshot],
    ) -> Result<()> {
        self.validate_registry(trainables)?;
        if state.len() != self.slots.len() {
            bail!(
                "parameter snapshot has {} slots; registry requires {}",
                state.len(),
                self.slots.len()
            );
        }
        for (index, ((slot, trainable), saved)) in
            self.slots.iter().zip(trainables).zip(state).enumerate()
        {
            if saved.name != slot.name {
                bail!(
                    "parameter snapshot slot {index} is {:?}; registry requires {:?}",
                    saved.name,
                    slot.name
                );
            }
            if saved.values.len() != slot.len {
                bail!(
                    "parameter snapshot {:?} has {} values; expected {}",
                    saved.name,
                    saved.values.len(),
                    slot.len
                );
            }
            if saved.values.iter().any(|value| !value.is_finite()) {
                bail!(
                    "parameter snapshot {:?} contains non-finite values",
                    saved.name
                );
            }
            // Registry validation above guarantees this is the exact canonical
            // parameter paired with the slot.
            let _ = trainable;
        }

        for (trainable, saved) in trainables.iter().zip(state) {
            trainable.parameter.write_f32(&saved.values)?;
        }

        if self
            .slots
            .iter()
            .any(|slot| slot.parameter_storage_mirror.is_some())
        {
            let refresher = self
                .parameter_mirror_refresher
                .as_ref()
                .context("mixed-precision parameter mirrors have no refresher")?;
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            for (slot, trainable) in self.slots.iter().zip(trainables) {
                if let Some(mirror) = slot.parameter_storage_mirror.as_ref() {
                    refresher.record_refresh(&mut commands, trainable.parameter, mirror)?;
                }
            }
            commands.submit()?;
        }
        Ok(())
    }

    pub fn load_state(&mut self, state: &AdamWOptimizerState) -> Result<()> {
        if state.slots.len() != self.slots.len() {
            bail!(
                "AdamW state has {} slots; registry requires {}",
                state.slots.len(),
                self.slots.len()
            );
        }
        for (index, (slot, saved)) in self.slots.iter().zip(&state.slots).enumerate() {
            if saved.name != slot.name {
                bail!(
                    "AdamW state slot {index} is {:?}; registry requires {:?}",
                    saved.name,
                    slot.name
                );
            }
            if let Some(saved_decay_class) = saved.decay_class {
                if saved_decay_class != slot.decay_class {
                    bail!(
                        "AdamW state slot {:?} decay class {:?} does not match live registry {:?}",
                        saved.name,
                        saved_decay_class,
                        slot.decay_class
                    );
                }
            }
            if saved.exp_avg.len() != slot.len || saved.exp_avg_sq.len() != slot.len {
                bail!(
                    "AdamW state {:?} moment lengths are {}/{}; expected {}",
                    saved.name,
                    saved.exp_avg.len(),
                    saved.exp_avg_sq.len(),
                    slot.len
                );
            }
            if saved
                .exp_avg
                .iter()
                .chain(&saved.exp_avg_sq)
                .any(|value| !value.is_finite())
            {
                bail!("AdamW state {:?} contains non-finite moments", saved.name);
            }
            slot.exp_avg.write_f32(&saved.exp_avg)?;
            slot.exp_avg_sq.write_f32(&saved.exp_avg_sq)?;
            if saved.step > state.step {
                bail!(
                    "AdamW state {:?} has per-slot step {} beyond global step {}",
                    saved.name,
                    saved.step,
                    state.step
                );
            }
        }
        for (slot, saved) in self.slots.iter_mut().zip(&state.slots) {
            slot.step = saved.step;
            slot.device_step_authoritative = false;
        }
        self.step = state.step;
        self.device_step_authoritative = false;
        self.host_step_metadata_authoritative = true;
        Ok(())
    }

    fn validate_registry(&self, trainables: &[RwkvTrainableRef<'_>]) -> Result<()> {
        if trainables.len() != self.slots.len() {
            bail!(
                "RWKV trainable registry changed from {} to {} tensors",
                self.slots.len(),
                trainables.len()
            );
        }
        for (index, (slot, trainable)) in self.slots.iter().zip(trainables).enumerate() {
            if let Err(err) = validate_slot(slot, trainable) {
                bail!("RWKV trainable registry mismatch at index {index}: {err}");
            }
        }
        Ok(())
    }

    fn validate_gradient_override(&self, name: &str, gradient: &GpuBuffer) -> Result<()> {
        let slot = self
            .slots
            .iter()
            .find(|slot| slot.name == name)
            .with_context(|| {
                format!(
                    "persistent optimizer has no registered tensor {name:?} for gradient override"
                )
            })?;
        if gradient.f32_capacity() < slot.len {
            bail!(
                "gradient override for {:?} has capacity {}; expected at least {}",
                slot.name,
                gradient.f32_capacity(),
                slot.len
            );
        }
        Ok(())
    }
}

fn validate_slot(slot: &OptimizerSlot, trainable: &RwkvTrainableRef<'_>) -> Result<()> {
    if slot.name != trainable.name
        || slot.len != trainable.len
        || slot.decay_class != trainable.decay_class
    {
        bail!(
            "optimizer=({}, {}, {:?}) live=({}, {}, {:?})",
            slot.name,
            slot.len,
            slot.decay_class,
            trainable.name,
            trainable.len,
            trainable.decay_class
        );
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_numerics::{
        VulkanDynamicLossScaleController, VulkanDynamicLossScaleSeed,
        DYNAMIC_LOSS_SCALE_CONTROL_UNSCALE_FACTOR_WORD,
    };

    #[test]
    fn optimizer_generation_waits_for_every_broadcast_source_clone() -> Result<()> {
        use std::sync::mpsc;
        use std::time::Duration;

        let guard = Arc::new(OptimizerGenerationGuard::default());
        let lease = guard.acquire_read_lease()?;
        let lease_clone = Arc::clone(&lease);
        let generation = lease.generation;
        let waiter = Arc::clone(&guard);
        let (finished_tx, finished_rx) = mpsc::channel();
        let writer = std::thread::spawn(move || -> Result<()> {
            waiter.wait_for_readers_to_retire(generation)?;
            let next_generation = waiter.advance_after_mutation(generation)?;
            finished_tx
                .send(next_generation)
                .map_err(|_| anyhow::anyhow!("generation test receiver disappeared"))?;
            Ok(())
        });

        assert!(finished_rx.recv_timeout(Duration::from_millis(25)).is_err());
        drop(lease);
        assert!(finished_rx.recv_timeout(Duration::from_millis(25)).is_err());
        drop(lease_clone);
        assert_eq!(finished_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        writer
            .join()
            .map_err(|_| anyhow::anyhow!("generation writer test thread panicked"))??;
        Ok(())
    }

    #[test]
    fn optimizer_generation_range_retirement_unblocks_as_a_wavefront() -> Result<()> {
        use std::sync::mpsc;
        use std::time::Duration;

        let guard = Arc::new(OptimizerGenerationGuard::default());
        let lease = guard.acquire_read_lease()?;
        let generation = lease.generation;
        let mut consumers = lease.split_into_range_consumers(3, 2)?;
        drop(lease);

        let waiter = Arc::clone(&guard);
        let (retired_tx, retired_rx) = mpsc::channel();
        let writer = std::thread::spawn(move || -> Result<()> {
            for range_index in 0..3 {
                waiter.wait_for_range_to_retire(generation, range_index)?;
                retired_tx
                    .send(range_index)
                    .map_err(|_| anyhow::anyhow!("range-retirement test receiver disappeared"))?;
            }
            waiter.advance_after_mutation(generation)?;
            Ok(())
        });

        assert!(retired_rx.recv_timeout(Duration::from_millis(25)).is_err());
        consumers[0].retire(0)?;
        assert!(retired_rx.recv_timeout(Duration::from_millis(25)).is_err());
        consumers[1].retire(0)?;
        assert_eq!(retired_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 0);

        consumers[1].retire(2)?;
        assert!(retired_rx.recv_timeout(Duration::from_millis(25)).is_err());
        consumers[0].retire(1)?;
        assert!(retired_rx.recv_timeout(Duration::from_millis(25)).is_err());
        consumers[1].retire(1)?;
        assert_eq!(retired_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        assert!(retired_rx.recv_timeout(Duration::from_millis(25)).is_err());
        consumers[0].retire(2)?;
        assert_eq!(retired_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 2);

        writer
            .join()
            .map_err(|_| anyhow::anyhow!("range-retirement writer test thread panicked"))??;
        assert_eq!(guard.current_generation(), generation + 1);
        Ok(())
    }

    #[test]
    fn optimizer_generation_ready_run_stops_at_first_live_range() -> Result<()> {
        let guard = Arc::new(OptimizerGenerationGuard::default());
        let lease = guard.acquire_read_lease()?;
        let generation = lease.generation;
        let mut consumers = lease.split_into_range_consumers(4, 1)?;
        drop(lease);

        consumers[0].retire(0)?;
        consumers[0].retire(1)?;
        consumers[0].retire(3)?;
        assert_eq!(guard.wait_for_ready_range_run(generation, 0, 4, 1)?, 2);

        consumers[0].retire(2)?;
        assert_eq!(guard.wait_for_ready_range_run(generation, 2, 4, 1)?, 4);
        Ok(())
    }

    #[test]
    fn optimizer_generation_nonblocking_ready_run_preserves_coalescing() -> Result<()> {
        let guard = Arc::new(OptimizerGenerationGuard::default());
        let lease = guard.acquire_read_lease()?;
        let generation = lease.generation;
        let mut consumers = lease.split_into_range_consumers(4, 1)?;
        drop(lease);

        assert!(guard
            .try_ready_range_run_with_gpu_waits(generation, 0, 4, 2)?
            .is_none());
        consumers[0].retire(0)?;
        assert!(guard
            .try_ready_range_run_with_gpu_waits(generation, 0, 4, 2)?
            .is_none());
        consumers[0].retire(1)?;
        consumers[0].retire(2)?;
        let (ready_end, waits) = guard
            .try_ready_range_run_with_gpu_waits(generation, 0, 4, 2)?
            .context("retired CPU ranges should be visible without a Condvar wait")?;
        assert_eq!(ready_end, 3);
        assert!(waits.is_empty());

        consumers[0].retire(3)?;
        let (ready_end, waits) = guard
            .try_ready_range_run_with_gpu_waits(generation, 3, 4, 2)?
            .context("final partial run should be immediately ready")?;
        assert_eq!(ready_end, 4);
        assert!(waits.is_empty());
        Ok(())
    }

    #[test]
    fn optimizer_generation_predeclared_gpu_ranges_keep_wavefront_bounded() -> Result<()> {
        let guard = Arc::new(OptimizerGenerationGuard::default());
        let lease = guard.acquire_read_lease()?;
        let generation = lease.generation;
        let mut consumers = lease.split_into_range_consumers(4, 1)?;
        drop(lease);

        // Actual DeviceGroupTimelineWait handles require multi-device Vulkan
        // hardware. This unit test isolates the scheduler bit carried alongside
        // those waits: future values must not make the ready-run coalescer merge
        // the entire model into one wait on the final timeline value.
        consumers[0].retire_all()?;
        {
            let mut state = guard.lock_state();
            state.range_gpu_waits_predeclared.fill(true);
        }
        let (first_end, first_waits) = guard
            .try_ready_range_run_with_gpu_waits(generation, 0, 4, 2)?
            .context("predeclared first range run should be immediately queue-ready")?;
        assert_eq!(first_end, 2);
        assert!(first_waits.is_empty());
        let (second_end, second_waits) = guard
            .try_ready_range_run_with_gpu_waits(generation, 2, 4, 2)?
            .context("predeclared second range run should be immediately queue-ready")?;
        assert_eq!(second_end, 4);
        assert!(second_waits.is_empty());

        let schedule = guard
            .predeclared_gpu_range_schedule(generation, 4, 2)?
            .context("fully predeclared ranges should snapshot one immutable GPU schedule")?;
        assert_eq!(schedule.run_count(), 2);
        let (first_end, first_waits) = schedule.range_run(0)?;
        assert_eq!(first_end, 2);
        assert!(first_waits.is_empty());
        let (second_end, second_waits) = schedule.range_run(first_end)?;
        assert_eq!(second_end, 4);
        assert!(second_waits.is_empty());
        Ok(())
    }

    #[test]
    fn optimizer_generation_predeclared_gpu_schedule_rejects_host_owned_suffix() -> Result<()> {
        let guard = Arc::new(OptimizerGenerationGuard::default());
        let lease = guard.acquire_read_lease()?;
        let generation = lease.generation;
        let mut consumers = lease.split_into_range_consumers(4, 1)?;
        drop(lease);

        consumers[0].retire(0)?;
        consumers[0].retire(1)?;
        {
            let mut state = guard.lock_state();
            state.range_gpu_waits_predeclared[0] = true;
            state.range_gpu_waits_predeclared[1] = true;
        }
        assert!(guard
            .predeclared_gpu_range_schedule(generation, 4, 2)?
            .is_none());

        consumers[0].retire(2)?;
        consumers[0].retire(3)?;
        assert!(guard
            .predeclared_gpu_range_schedule(generation, 4, 2)?
            .is_none());
        Ok(())
    }

    #[test]
    fn optimizer_generation_ready_run_waits_for_coalescing_floor() -> Result<()> {
        use std::sync::mpsc;
        use std::time::Duration;

        let guard = Arc::new(OptimizerGenerationGuard::default());
        let lease = guard.acquire_read_lease()?;
        let generation = lease.generation;
        let mut consumers = lease.split_into_range_consumers(4, 1)?;
        drop(lease);

        let waiter = Arc::clone(&guard);
        let (ready_tx, ready_rx) = mpsc::channel();
        let writer = std::thread::spawn(move || -> Result<()> {
            let ready_end = waiter.wait_for_ready_range_run(generation, 0, 4, 3)?;
            ready_tx
                .send(ready_end)
                .map_err(|_| anyhow::anyhow!("ready-run coalescing test receiver disappeared"))?;
            Ok(())
        });

        consumers[0].retire(0)?;
        consumers[0].retire(1)?;
        assert!(ready_rx.recv_timeout(Duration::from_millis(25)).is_err());
        consumers[0].retire(2)?;
        assert_eq!(ready_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 3);

        writer
            .join()
            .map_err(|_| anyhow::anyhow!("ready-run coalescing test thread panicked"))??;
        Ok(())
    }

    #[test]
    fn adamw_range_wavefront_matches_full_tensor_update() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan AdamW range-wavefront parity test: {err:#}");
                return Ok(());
            }
        };
        let initial = [1.0f32, -2.0, 0.5, 4.0, -0.25];
        let grads = [0.25f32, -0.5, 1.0, -0.125, 0.75];
        let full_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let wave_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let full_gradient = GpuBuffer::from_f32(&device, &grads)?;
        let wave_gradient = GpuBuffer::from_f32(&device, &grads)?;
        let full_trainables = [RwkvTrainableRef {
            name: "parity.weight",
            parameter: &full_parameter,
            gradient: &full_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let wave_trainables = [RwkvTrainableRef {
            name: "parity.weight",
            parameter: &wave_parameter,
            gradient: &wave_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let mut full = RwkvPersistentAdamW::new(device.clone(), &full_trainables)?;
        let mut wave = RwkvPersistentAdamW::new(device.clone(), &wave_trainables)?;
        let hyper = AdamWHyperParams {
            lr: 3.0e-3,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1.0e-8,
            weight_decay: 0.1,
        };

        let mut full_commands = vulkan::ComputeBatch::new(&device)?;
        full.record_zero_grad(&mut full_commands)?;
        full.record_accumulate(&mut full_commands, &full_trainables)?;
        full.record_step(&mut full_commands, &full_trainables, hyper)?;
        full_commands.submit()?;

        let mut wave_commands = vulkan::ComputeBatch::new(&device)?;
        wave.record_zero_grad(&mut wave_commands)?;
        wave.record_accumulate(&mut wave_commands, &wave_trainables)?;
        wave_commands.submit()?;
        let source = wave.replica_state_source(&wave_trainables)?;
        let mut consumers = source.prepare_range_retirement_consumers(2, 2)?;
        for consumer in &mut consumers {
            consumer.retire_all()?;
        }
        drop(consumers);
        drop(source);
        let (wave_step, ranges, queue_submissions) = wave
            .step_wavefront_with_named_gradient_override_and_inactive_names(
                &wave_trainables,
                hyper,
                None,
                &[],
                2,
            )?;
        assert_eq!(wave_step.step, 1);
        assert_eq!(ranges, initial.len().div_ceil(2));
        assert_eq!(queue_submissions, 1);

        let full_state = full.state_snapshot()?;
        let wave_state = wave.state_snapshot()?;
        assert_eq!(full_state.step, wave_state.step);
        assert_eq!(full_state.slots.len(), wave_state.slots.len());
        for (full_slot, wave_slot) in full_state.slots.iter().zip(&wave_state.slots) {
            assert_eq!(full_slot.step, wave_slot.step);
            for (actual, expected) in full_slot.exp_avg.iter().zip(&wave_slot.exp_avg) {
                assert!((actual - expected).abs() <= 2.0e-7);
            }
            for (actual, expected) in full_slot.exp_avg_sq.iter().zip(&wave_slot.exp_avg_sq) {
                assert!((actual - expected).abs() <= 2.0e-8);
            }
        }
        let full_values = full_parameter.read_f32(initial.len())?;
        let wave_values = wave_parameter.read_f32(initial.len())?;
        for (actual, expected) in full_values.iter().zip(&wave_values) {
            assert!((actual - expected).abs() <= 2.0e-7);
        }
        Ok(())
    }

    #[test]
    fn adamw_predeclared_wavefront_queues_multiple_bounded_runs() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan predeclared AdamW queue test: {err:#}");
                return Ok(());
            }
        };
        let initial = (0..17)
            .map(|index| 0.25 + index as f32 * 0.125)
            .collect::<Vec<_>>();
        let grads = (0..17)
            .map(|index| (index as f32 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let full_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let wave_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let full_gradient = GpuBuffer::from_f32(&device, &grads)?;
        let wave_gradient = GpuBuffer::from_f32(&device, &grads)?;
        let full_trainables = [RwkvTrainableRef {
            name: "predeclared-parity.weight",
            parameter: &full_parameter,
            gradient: &full_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let wave_trainables = [RwkvTrainableRef {
            name: "predeclared-parity.weight",
            parameter: &wave_parameter,
            gradient: &wave_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let mut full = RwkvPersistentAdamW::new(device.clone(), &full_trainables)?;
        let mut wave = RwkvPersistentAdamW::new(device.clone(), &wave_trainables)?;
        let hyper = AdamWHyperParams {
            lr: 2.0e-3,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1.0e-8,
            weight_decay: 0.05,
        };

        let mut full_commands = vulkan::ComputeBatch::new(&device)?;
        full.record_zero_grad(&mut full_commands)?;
        full.record_accumulate(&mut full_commands, &full_trainables)?;
        full.record_step(&mut full_commands, &full_trainables, hyper)?;
        full_commands.submit()?;

        let mut wave_commands = vulkan::ComputeBatch::new(&device)?;
        wave.record_zero_grad(&mut wave_commands)?;
        wave.record_accumulate(&mut wave_commands, &wave_trainables)?;
        wave_commands.submit()?;
        let source = wave.replica_state_source(&wave_trainables)?;
        let mut consumers = source.prepare_range_retirement_consumers(1, 1)?;
        consumers[0].retire_all()?;
        {
            // Unit-test the scheduler property independently of whether this
            // machine exposes a multi-physical-device timeline semaphore. The
            // production path sets this bit while publishing the real waits.
            let mut state = wave.generation_guard.lock_state();
            state.range_gpu_waits_predeclared.fill(true);
        }
        drop(consumers);
        drop(source);

        let (wave_step, ranges, queue_submissions) = wave
            .step_wavefront_with_named_gradient_override_and_inactive_names(
                &wave_trainables,
                hyper,
                None,
                &[],
                1,
            )?;
        assert_eq!(wave_step.step, 1);
        assert_eq!(ranges, 17);
        assert_eq!(queue_submissions, 3);

        let full_state = full.state_snapshot()?;
        let wave_state = wave.state_snapshot()?;
        assert_eq!(full_state.step, wave_state.step);
        assert_eq!(full_state.slots.len(), wave_state.slots.len());
        for (full_slot, wave_slot) in full_state.slots.iter().zip(&wave_state.slots) {
            assert_eq!(full_slot.step, wave_slot.step);
            for (actual, expected) in full_slot.exp_avg.iter().zip(&wave_slot.exp_avg) {
                assert!((actual - expected).abs() <= 2.0e-7);
            }
            for (actual, expected) in full_slot.exp_avg_sq.iter().zip(&wave_slot.exp_avg_sq) {
                assert!((actual - expected).abs() <= 2.0e-8);
            }
        }
        let full_values = full_parameter.read_f32(initial.len())?;
        let wave_values = wave_parameter.read_f32(initial.len())?;
        for (actual, expected) in full_values.iter().zip(&wave_values) {
            assert!((actual - expected).abs() <= 2.0e-7);
        }
        Ok(())
    }

    #[test]
    fn scaled_adamw_range_wavefront_matches_unscaled_full_tensor_update() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan scaled AdamW range-wavefront parity test: {err:#}");
                return Ok(());
            }
        };
        let initial = [1.0f32, -2.0, 0.5, 4.0, -0.25];
        let scaled_grads = [2.0f32, -4.0, 8.0, -1.0, 6.0];
        let full_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let wave_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let full_gradient = GpuBuffer::from_f32(&device, &scaled_grads)?;
        let wave_gradient = GpuBuffer::from_f32(&device, &scaled_grads)?;
        let full_trainables = [RwkvTrainableRef {
            name: "scaled-parity.weight",
            parameter: &full_parameter,
            gradient: &full_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let wave_trainables = [RwkvTrainableRef {
            name: "scaled-parity.weight",
            parameter: &wave_parameter,
            gradient: &wave_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let mut full = RwkvPersistentAdamW::new(device.clone(), &full_trainables)?;
        let mut wave = RwkvPersistentAdamW::new(device.clone(), &wave_trainables)?;
        let hyper = AdamWHyperParams {
            lr: 3.0e-3,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1.0e-8,
            weight_decay: 0.1,
        };

        let mut full_commands = vulkan::ComputeBatch::new(&device)?;
        full.record_zero_grad(&mut full_commands)?;
        full.record_accumulate(&mut full_commands, &full_trainables)?;
        full.record_scale_gradients(&mut full_commands, 1.0 / 8.0)?;
        full.record_step(&mut full_commands, &full_trainables, hyper)?;
        full_commands.submit()?;

        let mut wave_commands = vulkan::ComputeBatch::new(&device)?;
        wave.record_zero_grad(&mut wave_commands)?;
        wave.record_accumulate(&mut wave_commands, &wave_trainables)?;
        wave_commands.submit()?;
        let source = wave.replica_state_source(&wave_trainables)?;
        let mut consumers = source.prepare_range_retirement_consumers(2, 2)?;
        for consumer in &mut consumers {
            consumer.retire_all()?;
        }
        drop(consumers);
        drop(source);

        let mut unscale_commands = vulkan::ComputeBatch::new(&device)?;
        wave.record_scale_gradients(&mut unscale_commands, 1.0 / 8.0)?;
        unscale_commands.submit()?;
        let (wave_step, ranges, queue_submissions) = wave
            .step_wavefront_with_named_gradient_override_and_inactive_names(
                &wave_trainables,
                hyper,
                None,
                &[],
                2,
            )?;
        assert_eq!(wave_step.step, 1);
        assert_eq!(ranges, initial.len().div_ceil(2));
        assert_eq!(queue_submissions, 1);

        let full_state = full.state_snapshot()?;
        let wave_state = wave.state_snapshot()?;
        assert_eq!(full_state.step, wave_state.step);
        assert_eq!(full_state.slots.len(), wave_state.slots.len());
        for (full_slot, wave_slot) in full_state.slots.iter().zip(&wave_state.slots) {
            assert_eq!(full_slot.step, wave_slot.step);
            for (actual, expected) in full_slot.exp_avg.iter().zip(&wave_slot.exp_avg) {
                assert!((actual - expected).abs() <= 2.0e-7);
            }
            for (actual, expected) in full_slot.exp_avg_sq.iter().zip(&wave_slot.exp_avg_sq) {
                assert!((actual - expected).abs() <= 2.0e-8);
            }
        }
        let full_values = full_parameter.read_f32(initial.len())?;
        let wave_values = wave_parameter.read_f32(initial.len())?;
        for (actual, expected) in full_values.iter().zip(&wave_values) {
            assert!((actual - expected).abs() <= 2.0e-7);
        }
        Ok(())
    }

    #[test]
    fn device_grad_scaler_gates_predeclared_adamw_wavefront_and_releases_generation() -> Result<()>
    {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan GradScaler wavefront parity test: {err:#}");
                return Ok(());
            }
        };
        let initial = (0..17)
            .map(|index| 0.5 + index as f32 * 0.0625)
            .collect::<Vec<_>>();
        let gradients = (0..17)
            .map(|index| (index as f32 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let scaled_gradients = gradients
            .iter()
            .map(|gradient| gradient * 8.0)
            .collect::<Vec<_>>();
        let reference_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let wave_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let reference_gradient = GpuBuffer::from_f32(&device, &gradients)?;
        let wave_gradient = GpuBuffer::from_f32(&device, &scaled_gradients)?;
        let reference_trainables = [RwkvTrainableRef {
            name: "grad-scaler-wavefront.weight",
            parameter: &reference_parameter,
            gradient: &reference_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let wave_trainables = [RwkvTrainableRef {
            name: "grad-scaler-wavefront.weight",
            parameter: &wave_parameter,
            gradient: &wave_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let mut reference = RwkvPersistentAdamW::new(device.clone(), &reference_trainables)?;
        let mut wave = RwkvPersistentAdamW::new(device.clone(), &wave_trainables)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let hyper = AdamWHyperParams {
            lr: 2.0e-3,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1.0e-8,
            weight_decay: 0.05,
        };

        let mut reference_commands = vulkan::ComputeBatch::new(&device)?;
        reference.record_zero_grad(&mut reference_commands)?;
        reference.record_accumulate(&mut reference_commands, &reference_trainables)?;
        reference.record_step(&mut reference_commands, &reference_trainables, hyper)?;
        reference_commands.submit()?;

        let mut accumulate = vulkan::ComputeBatch::new(&device)?;
        wave.record_zero_grad(&mut accumulate)?;
        wave.record_accumulate(&mut accumulate, &wave_trainables)?;
        accumulate.submit()?;
        let source = wave.replica_state_source(&wave_trainables)?;
        let mut consumers = source.prepare_range_retirement_consumers(1, 1)?;
        consumers[0].retire_all()?;
        {
            // Production device-group broadcasts publish real future timeline
            // waits. Mark the already-retired test plan predeclared so the
            // scheduler must retain the same bounded 8+8+1 submission shape.
            let mut state = wave.generation_guard.lock_state();
            state.range_gpu_waits_predeclared.fill(true);
        }
        drop(consumers);
        drop(source);

        let mut prepare = vulkan::ComputeBatch::new(&device)?;
        wave.record_accumulated_gradient_nonfinite_scan_with_named_override(&mut prepare, None)?;
        let finite_flag = wave.nonfinite_flag_buffer().clone();
        scaler.record_resolve(
            &mut prepare,
            &finite_flag,
            VulkanDynamicLossScaleSeed {
                scale: 8.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2,
                growth_tracker: 0,
                pending_gradients_scaled: true,
            },
        )?;
        prepare.submit()?;
        let control = scaler.control_buffer().clone();
        let finite_generation = wave.generation_guard.current_generation();
        let (finite_pending, ranges, queue_submissions) = wave
            .step_wavefront_grad_scaler_controlled_with_named_gradient_override_and_inactive_names(
                &wave_trainables,
                hyper,
                &control,
                1.0,
                true,
                None,
                None,
                &[],
                1,
            )?;
        assert_eq!(ranges, initial.len());
        assert_eq!(queue_submissions, 3);
        assert_eq!(
            wave.step, 0,
            "host Adam clock must remain pending until telemetry finalize"
        );
        assert_eq!(
            wave.generation_guard.current_generation(),
            finite_generation + 1,
            "GPU wavefront tail must release the next hazard generation before host decision readback"
        );

        let mut finite_readback = vulkan::ComputeBatch::new(&device)?;
        scaler.record_readback(&mut finite_readback)?;
        finite_readback.submit()?;
        let finite_decision = scaler.read_decision()?;
        assert!(finite_decision.should_step);
        assert!(!finite_decision.overflowed);
        assert_eq!(finite_decision.unscale_factor, 1.0 / 8.0);
        assert_eq!(
            wave.finalize_device_controlled_step(finite_pending, true)?
                .step,
            1
        );

        let reference_state = reference.state_snapshot()?;
        let finite_state = wave.state_snapshot()?;
        assert_eq!(reference_state.step, finite_state.step);
        assert_eq!(reference_state.slots[0].step, finite_state.slots[0].step);
        for (actual, expected) in finite_state.slots[0]
            .exp_avg
            .iter()
            .zip(&reference_state.slots[0].exp_avg)
        {
            assert!((actual - expected).abs() <= 3.0e-7);
        }
        for (actual, expected) in finite_state.slots[0]
            .exp_avg_sq
            .iter()
            .zip(&reference_state.slots[0].exp_avg_sq)
        {
            assert!((actual - expected).abs() <= 3.0e-8);
        }
        let reference_values = reference_parameter.read_f32(initial.len())?;
        let finite_values = wave_parameter.read_f32(initial.len())?;
        for (actual, expected) in finite_values.iter().zip(&reference_values) {
            assert!((actual - expected).abs() <= 3.0e-7);
        }

        // Overflow must still publish a new hazard epoch: the predicated range
        // kernels clear their gradients but leave Adam state untouched. The
        // host successful-step clock remains at one after finalization.
        let nonfinite_gradient_words = (0..initial.len())
            .map(|index| {
                if index == 5 {
                    f32::INFINITY.to_bits()
                } else {
                    ((index as f32 + 1.0) * 8.0).to_bits()
                }
            })
            .collect::<Vec<_>>();
        let nonfinite_gradient = GpuBuffer::from_u32(&device, &nonfinite_gradient_words)?;
        let nonfinite_trainables = [RwkvTrainableRef {
            name: "grad-scaler-wavefront.weight",
            parameter: &wave_parameter,
            gradient: &nonfinite_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::Decay,
        }];
        let mut overflow_accumulate = vulkan::ComputeBatch::new(&device)?;
        wave.record_zero_grad(&mut overflow_accumulate)?;
        wave.record_accumulate(&mut overflow_accumulate, &nonfinite_trainables)?;
        overflow_accumulate.submit()?;
        let source = wave.replica_state_source(&nonfinite_trainables)?;
        let mut consumers = source.prepare_range_retirement_consumers(1, 1)?;
        consumers[0].retire_all()?;
        {
            let mut state = wave.generation_guard.lock_state();
            state.range_gpu_waits_predeclared.fill(true);
        }
        drop(consumers);
        drop(source);

        let mut overflow_prepare = vulkan::ComputeBatch::new(&device)?;
        wave.record_accumulated_gradient_nonfinite_scan_with_named_override(
            &mut overflow_prepare,
            None,
        )?;
        let overflow_flag = wave.nonfinite_flag_buffer().clone();
        scaler.record_resolve(
            &mut overflow_prepare,
            &overflow_flag,
            VulkanDynamicLossScaleSeed {
                scale: finite_decision.scale_after,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2,
                growth_tracker: finite_decision.growth_tracker,
                pending_gradients_scaled: true,
            },
        )?;
        overflow_prepare.submit()?;
        let control = scaler.control_buffer().clone();
        let overflow_generation = wave.generation_guard.current_generation();
        let (overflow_pending, ranges, queue_submissions) = wave
            .step_wavefront_grad_scaler_controlled_with_named_gradient_override_and_inactive_names(
                &nonfinite_trainables,
                hyper,
                &control,
                1.0,
                true,
                None,
                None,
                &[],
                1,
            )?;
        assert_eq!(ranges, initial.len());
        assert_eq!(queue_submissions, 3);
        assert_eq!(wave.step, 1);
        assert_eq!(
            wave.generation_guard.current_generation(),
            overflow_generation + 1
        );

        let mut overflow_readback = vulkan::ComputeBatch::new(&device)?;
        scaler.record_readback(&mut overflow_readback)?;
        overflow_readback.submit()?;
        let overflow_decision = scaler.read_decision()?;
        assert!(overflow_decision.overflowed);
        assert!(!overflow_decision.should_step);
        assert_eq!(overflow_decision.scale_after, 4.0);
        assert_eq!(
            wave.finalize_device_controlled_step(overflow_pending, false)?
                .step,
            1
        );
        assert_eq!(wave_parameter.read_f32(initial.len())?, finite_values);
        let overflow_state = wave.state_snapshot()?;
        assert_eq!(overflow_state.step, finite_state.step);
        assert_eq!(overflow_state.slots[0].step, finite_state.slots[0].step);
        assert_eq!(
            overflow_state.slots[0].exp_avg,
            finite_state.slots[0].exp_avg
        );
        assert_eq!(
            overflow_state.slots[0].exp_avg_sq,
            finite_state.slots[0].exp_avg_sq
        );
        Ok(())
    }

    #[test]
    fn loss_unscale_and_adamw_share_one_command_stream() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan fused loss-unscale AdamW test: {err:#}");
                return Ok(());
            }
        };
        let parameter = GpuBuffer::from_f32(&device, &[1.0, -2.0])?;
        // This is an 8x loss-scaled gradient. The optimizer must observe
        // [1.0, -0.5] after the in-stream unscale dispatch below.
        let gradient = GpuBuffer::from_f32(&device, &[8.0, -4.0])?;
        let trainables = [RwkvTrainableRef {
            name: "scaled.weight",
            parameter: &parameter,
            gradient: &gradient,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let mut optimizer = RwkvPersistentAdamW::new(device.clone(), &trainables)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut commands)?;
        optimizer.record_accumulate(&mut commands, &trainables)?;
        optimizer.record_scale_gradients(&mut commands, 1.0 / 8.0)?;
        optimizer.record_step(
            &mut commands,
            &trainables,
            AdamWHyperParams {
                lr: 1.0e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1.0e-8,
                weight_decay: 0.0,
            },
        )?;
        commands.submit()?;

        let state = optimizer.state_snapshot()?;
        assert_eq!(state.step, 1);
        assert_eq!(state.slots[0].step, 1);
        for (actual, expected) in state.slots[0].exp_avg.iter().zip([0.1, -0.05]) {
            assert!(
                (actual - expected).abs() <= 2.0e-7,
                "exp_avg={actual} expected {expected}"
            );
        }
        for (actual, expected) in state.slots[0].exp_avg_sq.iter().zip([0.001, 0.00025]) {
            assert!(
                (actual - expected).abs() <= 2.0e-8,
                "exp_avg_sq={actual} expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn device_grad_scaler_gates_adamw_without_host_branch() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan device-controlled GradScaler test: {err:#}");
                return Ok(());
            }
        };
        let parameter = GpuBuffer::from_f32(&device, &[1.0, -2.0])?;
        let finite_scaled_gradient = GpuBuffer::from_f32(&device, &[8.0, -4.0])?;
        let finite_trainables = [RwkvTrainableRef {
            name: "device-controlled.weight",
            parameter: &parameter,
            gradient: &finite_scaled_gradient,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let mut optimizer = RwkvPersistentAdamW::new(device.clone(), &finite_trainables)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let hyper = AdamWHyperParams {
            lr: 1.0e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
        };

        // Finite 8x-scaled gradients: the shader resolves the reciprocal and
        // AdamW consumes it later in the same command buffer.
        let mut finite_commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut finite_commands)?;
        optimizer.record_accumulate(&mut finite_commands, &finite_trainables)?;
        optimizer.record_accumulated_gradient_nonfinite_scan_with_named_override(
            &mut finite_commands,
            None,
        )?;
        let finite_flag = optimizer.nonfinite_flag_buffer().clone();
        scaler.record_resolve(
            &mut finite_commands,
            &finite_flag,
            VulkanDynamicLossScaleSeed {
                scale: 8.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2,
                growth_tracker: 0,
                pending_gradients_scaled: true,
            },
        )?;
        let control = scaler.control_buffer().clone();
        let finite_pending = optimizer.record_device_controlled_step(
            &mut finite_commands,
            &finite_trainables,
            hyper,
            &control,
            1.0,
            true,
            None,
            None,
            &[],
        )?;
        scaler.record_readback(&mut finite_commands)?;
        finite_commands.submit()?;

        let finite_decision = scaler.read_decision()?;
        assert!(!finite_decision.overflowed);
        assert!(finite_decision.should_step);
        assert_eq!(finite_decision.unscale_factor, 1.0 / 8.0);
        assert_eq!(finite_decision.scale_after, 8.0);
        assert_eq!(finite_decision.growth_tracker, 1);
        let finite_step = optimizer.finalize_device_controlled_step(finite_pending, true)?;
        assert_eq!(finite_step.step, 1);
        let finite_state = optimizer.state_snapshot()?;
        assert_eq!(finite_state.step, 1);
        assert_eq!(finite_state.slots[0].step, 1);
        for (actual, expected) in finite_state.slots[0].exp_avg.iter().zip([0.1, -0.05]) {
            assert!((actual - expected).abs() <= 2.0e-7);
        }
        let parameter_after_finite = parameter.read_f32(2)?;

        // A subsequent non-finite window stays in the same control state. The
        // device backs the scale off, clears the accumulated gradient, and
        // leaves parameters/moments/step untouched without any CPU decision
        // between the scan and the predicated AdamW dispatch.
        let nonfinite_gradient =
            GpuBuffer::from_u32(&device, &[f32::INFINITY.to_bits(), 1.0f32.to_bits()])?;
        let nonfinite_trainables = [RwkvTrainableRef {
            name: "device-controlled.weight",
            parameter: &parameter,
            gradient: &nonfinite_gradient,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let mut overflow_commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut overflow_commands)?;
        optimizer.record_accumulate(&mut overflow_commands, &nonfinite_trainables)?;
        optimizer.record_accumulated_gradient_nonfinite_scan_with_named_override(
            &mut overflow_commands,
            None,
        )?;
        let overflow_flag = optimizer.nonfinite_flag_buffer().clone();
        scaler.record_resolve(
            &mut overflow_commands,
            &overflow_flag,
            VulkanDynamicLossScaleSeed {
                scale: finite_decision.scale_after,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2,
                growth_tracker: finite_decision.growth_tracker,
                pending_gradients_scaled: true,
            },
        )?;
        let control = scaler.control_buffer().clone();
        let overflow_pending = optimizer.record_device_controlled_step(
            &mut overflow_commands,
            &nonfinite_trainables,
            hyper,
            &control,
            1.0,
            true,
            None,
            None,
            &[],
        )?;
        scaler.record_readback(&mut overflow_commands)?;
        overflow_commands.submit()?;

        let overflow_decision = scaler.read_decision()?;
        assert!(overflow_decision.overflowed);
        assert!(!overflow_decision.should_step);
        assert_eq!(overflow_decision.scale_before, 8.0);
        assert_eq!(overflow_decision.scale_after, 4.0);
        assert_eq!(overflow_decision.growth_tracker, 0);
        let overflow_step = optimizer.finalize_device_controlled_step(overflow_pending, false)?;
        assert_eq!(overflow_step.step, 1);
        assert_eq!(parameter.read_f32(2)?, parameter_after_finite);
        let overflow_state = optimizer.state_snapshot()?;
        assert_eq!(overflow_state.step, finite_state.step);
        assert_eq!(overflow_state.slots[0].step, finite_state.slots[0].step);
        assert_eq!(
            overflow_state.slots[0].exp_avg,
            finite_state.slots[0].exp_avg
        );
        assert_eq!(
            overflow_state.slots[0].exp_avg_sq,
            finite_state.slots[0].exp_avg_sq
        );
        Ok(())
    }

    #[test]
    fn device_grad_scaler_gates_prepared_clipped_adamw_without_double_unscale() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan device-controlled clipped GradScaler test: {err:#}");
                return Ok(());
            }
        };
        let parameter = GpuBuffer::from_f32(&device, &[1.0, -2.0])?;
        // 8x source scaling: after unscale this is the classic [3,4] vector,
        // whose norm is 5 and whose max_norm=2 PyTorch coefficient is ~0.4.
        let finite_scaled_gradient = GpuBuffer::from_f32(&device, &[24.0, 32.0])?;
        let finite_trainables = [RwkvTrainableRef {
            name: "device-controlled-clipped.weight",
            parameter: &parameter,
            gradient: &finite_scaled_gradient,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let mut optimizer = RwkvPersistentAdamW::new(device.clone(), &finite_trainables)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let hyper = AdamWHyperParams {
            lr: 1.0e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
        };

        let mut finite_commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut finite_commands)?;
        optimizer.record_accumulate(&mut finite_commands, &finite_trainables)?;
        optimizer.record_scale_gradients(&mut finite_commands, 1.0 / 8.0)?;
        optimizer.record_accumulated_gradient_l2_norm_and_clip_coefficient_with_override_and_inactive_names(
            &mut finite_commands,
            None,
            &[],
            2.0,
        )?;
        optimizer.record_scale_gradients_from_device_clip_coefficient(&mut finite_commands)?;
        let clip_nonfinite = optimizer
            .accumulated_gradient_clip_nonfinite_buffer()
            .clone();
        scaler.record_resolve(
            &mut finite_commands,
            &clip_nonfinite,
            VulkanDynamicLossScaleSeed {
                scale: 8.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2,
                growth_tracker: 0,
                pending_gradients_scaled: true,
            },
        )?;
        let control = scaler.control_buffer().clone();
        let finite_pending = optimizer.record_device_controlled_step(
            &mut finite_commands,
            &finite_trainables,
            hyper,
            &control,
            1.0,
            false,
            None,
            None,
            &[],
        )?;
        scaler.record_readback(&mut finite_commands)?;
        finite_commands.submit()?;

        let finite_decision = scaler.read_decision()?;
        assert!(!finite_decision.overflowed);
        assert!(finite_decision.should_step);
        assert_eq!(finite_decision.unscale_factor, 1.0 / 8.0);
        assert_eq!(
            optimizer
                .finalize_device_controlled_step(finite_pending, true)?
                .step,
            1
        );
        let norm = optimizer.read_accumulated_gradient_l2_norm()?;
        let coefficient = optimizer.read_accumulated_gradient_clip_coefficient()?;
        let expected_coefficient = (2.0f64 / (5.0 + 1.0e-6)) as f32;
        assert!((norm - 5.0).abs() <= 1.0e-6);
        assert!((coefficient - expected_coefficient).abs() <= 1.0e-7);
        let finite_state = optimizer.state_snapshot()?;
        let expected_gradient = [3.0 * coefficient, 4.0 * coefficient];
        for (actual, expected) in finite_state.slots[0]
            .exp_avg
            .iter()
            .zip(expected_gradient.map(|gradient| 0.1 * gradient))
        {
            assert!(
                (actual - expected).abs() <= 3.0e-7,
                "prepared clipped exp_avg={actual} expected={expected}"
            );
        }
        let parameter_after_finite = parameter.read_f32(2)?;

        // The same norm reducer owns the overflow bit. Its coefficient is zero
        // for Inf/NaN, and the controlled AdamW skip clears any resulting NaNs
        // without advancing moments, parameters, or the Adam clock.
        let nonfinite_gradient =
            GpuBuffer::from_u32(&device, &[f32::INFINITY.to_bits(), 8.0f32.to_bits()])?;
        let nonfinite_trainables = [RwkvTrainableRef {
            name: "device-controlled-clipped.weight",
            parameter: &parameter,
            gradient: &nonfinite_gradient,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let mut overflow_commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut overflow_commands)?;
        optimizer.record_accumulate(&mut overflow_commands, &nonfinite_trainables)?;
        optimizer.record_scale_gradients(&mut overflow_commands, 1.0 / 8.0)?;
        optimizer.record_accumulated_gradient_l2_norm_and_clip_coefficient_with_override_and_inactive_names(
            &mut overflow_commands,
            None,
            &[],
            2.0,
        )?;
        optimizer.record_scale_gradients_from_device_clip_coefficient(&mut overflow_commands)?;
        let clip_nonfinite = optimizer
            .accumulated_gradient_clip_nonfinite_buffer()
            .clone();
        scaler.record_resolve(
            &mut overflow_commands,
            &clip_nonfinite,
            VulkanDynamicLossScaleSeed {
                scale: finite_decision.scale_after,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2,
                growth_tracker: finite_decision.growth_tracker,
                pending_gradients_scaled: true,
            },
        )?;
        let control = scaler.control_buffer().clone();
        let overflow_pending = optimizer.record_device_controlled_step(
            &mut overflow_commands,
            &nonfinite_trainables,
            hyper,
            &control,
            1.0,
            false,
            None,
            None,
            &[],
        )?;
        scaler.record_readback(&mut overflow_commands)?;
        overflow_commands.submit()?;

        let overflow_decision = scaler.read_decision()?;
        assert!(overflow_decision.overflowed);
        assert!(!overflow_decision.should_step);
        assert_eq!(overflow_decision.scale_after, 4.0);
        assert_eq!(
            optimizer
                .finalize_device_controlled_step(overflow_pending, false)?
                .step,
            1
        );
        assert_eq!(parameter.read_f32(2)?, parameter_after_finite);
        let overflow_state = optimizer.state_snapshot()?;
        assert_eq!(overflow_state.step, finite_state.step);
        assert_eq!(overflow_state.slots[0].step, finite_state.slots[0].step);
        assert_eq!(
            overflow_state.slots[0].exp_avg,
            finite_state.slots[0].exp_avg
        );
        assert_eq!(
            overflow_state.slots[0].exp_avg_sq,
            finite_state.slots[0].exp_avg_sq
        );
        Ok(())
    }

    #[test]
    fn indexed_device_gradient_scale_combines_control_word_and_normalization() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping indexed device gradient-scale test: {err:#}");
                return Ok(());
            }
        };
        let parameter = GpuBuffer::from_f32(&device, &[1.0, -2.0])?;
        let gradient = GpuBuffer::from_f32(&device, &[24.0, 32.0])?;
        let trainables = [RwkvTrainableRef {
            name: "indexed-device-scale.weight",
            parameter: &parameter,
            gradient: &gradient,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let optimizer = RwkvPersistentAdamW::new(device.clone(), &trainables)?;
        // Deliberately place the desired factor away from word zero. The
        // dynamic GradScaler ABI stores scale-after at word zero and the
        // current window's reciprocal at a later control word.
        let control = GpuBuffer::from_f32(&device, &[16.0, 0.125, 99.0])?;

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut commands)?;
        optimizer.record_accumulate(&mut commands, &trainables)?;
        optimizer.record_scale_gradients_from_indexed_device_factor_with_named_override(
            &mut commands,
            &control,
            1,
            0.5,
            None,
        )?;
        optimizer.record_accumulated_gradient_l2_norm_and_clip_coefficient_with_override_and_inactive_names(
            &mut commands,
            None,
            &[],
            100.0,
        )?;
        commands.submit()?;

        // [24, 32] * 0.125 * 0.5 = [1.5, 2.0], whose L2 norm is 2.5.
        assert!((optimizer.read_accumulated_gradient_l2_norm()? - 2.5).abs() <= 1.0e-6);
        Ok(())
    }

    #[test]
    fn clipped_grad_scaler_keeps_scale_and_adam_clocks_device_resident_across_windows() -> Result<()>
    {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping queue-resident clipped GradScaler test: {err:#}");
                return Ok(());
            }
        };
        let parameter = GpuBuffer::from_f32(&device, &[1.0, -2.0])?;
        let gradient8 = GpuBuffer::from_f32(&device, &[24.0, 32.0])?;
        let gradient16 = GpuBuffer::from_f32(&device, &[48.0, 64.0])?;
        let first_trainables = [RwkvTrainableRef {
            name: "queue-resident-clipped.weight",
            parameter: &parameter,
            gradient: &gradient8,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let mut optimizer = RwkvPersistentAdamW::new(device.clone(), &first_trainables)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let hyper = AdamWHyperParams {
            lr: 1.0e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
        };
        let stale_host_seed = VulkanDynamicLossScaleSeed {
            scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1,
            growth_tracker: 0,
            pending_gradients_scaled: true,
        };

        let run_window = |optimizer: &mut RwkvPersistentAdamW,
                          scaler: &mut VulkanDynamicLossScaleController,
                          trainables: &[RwkvTrainableRef<'_>]|
         -> Result<()> {
            let mut prepare = vulkan::ComputeBatch::new(&device)?;
            optimizer.record_zero_grad(&mut prepare)?;
            optimizer.record_accumulate(&mut prepare, trainables)?;
            optimizer.record_accumulated_gradient_nonfinite_scan_with_named_override(
                &mut prepare,
                None,
            )?;
            let nonfinite = optimizer.nonfinite_flag_buffer().clone();
            // The same stale host seed is intentionally reused on every window.
            // Device-resident resolve must preserve the live scale/tracker once
            // ownership has moved to Vulkan.
            scaler.record_resolve_device_resident(&mut prepare, &nonfinite, stale_host_seed)?;
            let control = scaler.control_buffer().clone();
            optimizer.record_scale_gradients_from_indexed_device_factor_with_named_override(
                &mut prepare,
                &control,
                DYNAMIC_LOSS_SCALE_CONTROL_UNSCALE_FACTOR_WORD,
                1.0,
                None,
            )?;
            optimizer.record_accumulated_gradient_l2_norm_and_clip_coefficient_device_only_with_override_and_inactive_names(
                &mut prepare,
                None,
                &[],
                2.0,
            )?;
            optimizer.record_scale_gradients_from_device_clip_coefficient(&mut prepare)?;
            prepare.submit()?;

            let (pending, ranges, _) = optimizer
                .step_wavefront_grad_scaler_controlled_with_named_gradient_override_and_inactive_names(
                    trainables,
                    hyper,
                    &control,
                    1.0,
                    false,
                    None,
                    None,
                    &[],
                    1024,
                )?;
            assert_eq!(ranges, 1);
            optimizer.defer_device_controlled_step_host_metadata(pending)?;
            Ok(())
        };

        run_window(&mut optimizer, &mut scaler, &first_trainables)?;
        let second_trainables = [RwkvTrainableRef {
            name: "queue-resident-clipped.weight",
            parameter: &parameter,
            gradient: &gradient16,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        run_window(&mut optimizer, &mut scaler, &second_trainables)?;

        // Replica-state transport must be able to clone the still-device-owned
        // Adam clocks without forcing the source optimizer through a metadata
        // readback. Use an ordinary device buffer here; the cross-adapter paths
        // wrap the same pack/write operations in peer/external-memory handoffs.
        let replica_parameter = GpuBuffer::from_f32(&device, &[0.0, 0.0])?;
        let replica_gradient = GpuBuffer::zeros_f32(&device, 2)?;
        let replica_trainables = [RwkvTrainableRef {
            name: "queue-resident-clipped.weight",
            parameter: &replica_parameter,
            gradient: &replica_gradient,
            len: 2,
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let mut replica = RwkvPersistentAdamW::new(device.clone(), &replica_trainables)?;
        let source = optimizer.replica_state_source(&second_trainables)?;
        assert!(source.step.is_device_authoritative());
        assert!(source.slots[0].step.is_device_authoritative());
        let packed_steps = GpuBuffer::zeros_f32(&device, source.step_metadata_word_count())?;
        let mut replica_commands = vulkan::ComputeBatch::new(&device)?;
        source.record_step_metadata_pack(&mut replica_commands, &packed_steps)?;
        replica.record_replica_step_metadata_write(
            &mut replica_commands,
            &source,
            &packed_steps,
        )?;
        replica_commands.submit()?;
        drop(source);
        assert!(!replica.host_step_metadata_authoritative);
        let replica_step = replica.synchronize_device_step_metadata()?;
        assert_eq!(replica_step.step, 2);
        assert_eq!(replica.state_snapshot()?.slots[0].step, 2);

        // Only now do checkpoint-style synchronization. Two finite windows with
        // growth_interval=1 advance scale 8 -> 16 -> 32 and Adam step 0 -> 2.
        let mut readback = vulkan::ComputeBatch::new(&device)?;
        scaler.record_readback(&mut readback)?;
        readback.submit()?;
        let scaler_decision = scaler.read_decision()?;
        assert_eq!(scaler_decision.scale_before, 16.0);
        assert_eq!(scaler_decision.scale_after, 32.0);
        assert_eq!(optimizer.synchronize_device_step_metadata()?.step, 2);
        Ok(())
    }

    #[test]
    fn device_norm_clip_scale_and_adamw_match_host_reference() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan device-resident clipping AdamW test: {err:#}");
                return Ok(());
            }
        };
        let initial = [1.0, -2.0];
        let gradient_values = [3.0, 4.0];
        let device_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let host_parameter = GpuBuffer::from_f32(&device, &initial)?;
        let device_gradient = GpuBuffer::from_f32(&device, &gradient_values)?;
        let host_gradient = GpuBuffer::from_f32(&device, &gradient_values)?;
        let device_trainables = [RwkvTrainableRef {
            name: "clip.weight",
            parameter: &device_parameter,
            gradient: &device_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let host_trainables = [RwkvTrainableRef {
            name: "clip.weight",
            parameter: &host_parameter,
            gradient: &host_gradient,
            len: initial.len(),
            decay_class: RwkvDecayClass::NoDecay,
        }];
        let hyper = AdamWHyperParams {
            lr: 1.0e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
        };
        let max_norm = 1.0f32;
        let host_clip = max_norm / (5.0f32 + 1.0e-6);
        let mut device_optimizer = RwkvPersistentAdamW::new(device.clone(), &device_trainables)?;
        let mut host_optimizer = RwkvPersistentAdamW::new(device.clone(), &host_trainables)?;

        let mut device_commands = vulkan::ComputeBatch::new(&device)?;
        device_optimizer.record_zero_grad(&mut device_commands)?;
        device_optimizer.record_accumulate(&mut device_commands, &device_trainables)?;
        device_optimizer
            .record_accumulated_gradient_l2_norm_and_clip_coefficient_with_override_and_inactive_names(
                &mut device_commands,
                None,
                &[],
                max_norm,
            )?;
        device_optimizer
            .record_scale_gradients_from_device_clip_coefficient(&mut device_commands)?;
        device_optimizer.record_step(&mut device_commands, &device_trainables, hyper)?;
        device_commands.submit()?;

        let mut host_commands = vulkan::ComputeBatch::new(&device)?;
        host_optimizer.record_zero_grad(&mut host_commands)?;
        host_optimizer.record_accumulate(&mut host_commands, &host_trainables)?;
        host_optimizer.record_scale_gradients(&mut host_commands, host_clip)?;
        host_optimizer.record_step(&mut host_commands, &host_trainables, hyper)?;
        host_commands.submit()?;

        let device_norm = device_optimizer.read_accumulated_gradient_l2_norm()?;
        let device_clip = device_optimizer.read_accumulated_gradient_clip_coefficient()?;
        assert!(!device_optimizer.read_accumulated_gradient_clip_has_nonfinite()?);
        assert!((device_norm - 5.0).abs() <= 1.0e-6, "norm={device_norm}");
        assert!(
            (device_clip - host_clip).abs() <= 2.0e-7,
            "device_clip={device_clip} host_clip={host_clip}"
        );

        let device_state = device_optimizer.state_snapshot()?;
        let host_state = host_optimizer.state_snapshot()?;
        assert_eq!(device_state.step, host_state.step);
        for (actual, expected) in device_state.slots[0]
            .exp_avg
            .iter()
            .zip(&host_state.slots[0].exp_avg)
        {
            assert!((actual - expected).abs() <= 2.0e-7);
        }
        for (actual, expected) in device_state.slots[0]
            .exp_avg_sq
            .iter()
            .zip(&host_state.slots[0].exp_avg_sq)
        {
            assert!((actual - expected).abs() <= 2.0e-8);
        }
        let device_values = device_parameter.read_f32(initial.len())?;
        let host_values = host_parameter.read_f32(initial.len())?;
        for (actual, expected) in device_values.iter().zip(&host_values) {
            assert!((actual - expected).abs() <= 2.0e-7);
        }
        Ok(())
    }

    #[test]
    fn inactive_slot_preserves_parameter_moments_and_adam_clock() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan inactive-slot AdamW test: {err:#}");
                return Ok(());
            }
        };
        let first_parameter = GpuBuffer::from_f32(&device, &[1.0, -0.5])?;
        let second_parameter = GpuBuffer::from_f32(&device, &[2.0, -3.0])?;
        let first_gradient = GpuBuffer::from_f32(&device, &[0.25, -0.125])?;
        let second_gradient = GpuBuffer::from_f32(&device, &[0.5, -0.75])?;
        let trainables = [
            RwkvTrainableRef {
                name: "always.weight",
                parameter: &first_parameter,
                gradient: &first_gradient,
                len: 2,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "intermittent.weight",
                parameter: &second_parameter,
                gradient: &second_gradient,
                len: 2,
                decay_class: RwkvDecayClass::NoDecay,
            },
        ];
        let hyper = AdamWHyperParams {
            lr: 1.0e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
        };
        let mut optimizer = RwkvPersistentAdamW::new(device.clone(), &trainables)?;

        let mut first_commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut first_commands)?;
        optimizer.record_accumulate(&mut first_commands, &trainables)?;
        optimizer.record_step(&mut first_commands, &trainables, hyper)?;
        first_commands.submit()?;
        let state_after_first = optimizer.state_snapshot()?;
        let intermittent_parameter_after_first = second_parameter.read_f32(2)?;

        let mut second_commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut second_commands)?;
        optimizer.record_accumulate_one(&mut second_commands, trainables[0])?;
        optimizer.record_step_with_inactive_names(
            &mut second_commands,
            &trainables,
            hyper,
            &["intermittent.weight"],
        )?;
        second_commands.submit()?;
        let state_after_second = optimizer.state_snapshot()?;

        assert_eq!(state_after_second.step, 2);
        assert_eq!(state_after_second.slots[0].step, 2);
        assert_eq!(state_after_second.slots[1].step, 1);
        assert_eq!(
            state_after_second.slots[1].exp_avg,
            state_after_first.slots[1].exp_avg
        );
        assert_eq!(
            state_after_second.slots[1].exp_avg_sq,
            state_after_first.slots[1].exp_avg_sq
        );
        assert_eq!(
            second_parameter.read_f32(2)?,
            intermittent_parameter_after_first
        );
        Ok(())
    }

    #[test]
    fn inactive_gradient_override_is_accepted_and_remains_frozen() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan inactive override AdamW test: {err:#}");
                return Ok(());
            }
        };
        let active_parameter = GpuBuffer::from_f32(&device, &[1.0, -0.5])?;
        let frozen_parameter = GpuBuffer::from_f32(&device, &[2.0, -3.0])?;
        let active_gradient = GpuBuffer::from_f32(&device, &[0.25, -0.125])?;
        let frozen_gradient = GpuBuffer::from_f32(&device, &[0.0, 0.0])?;
        let override_gradient = GpuBuffer::from_f32(&device, &[9.0, -11.0])?;
        let trainables = [
            RwkvTrainableRef {
                name: "active.weight",
                parameter: &active_parameter,
                gradient: &active_gradient,
                len: 2,
                decay_class: RwkvDecayClass::NoDecay,
            },
            RwkvTrainableRef {
                name: "frozen.weight",
                parameter: &frozen_parameter,
                gradient: &frozen_gradient,
                len: 2,
                decay_class: RwkvDecayClass::NoDecay,
            },
        ];
        let hyper = AdamWHyperParams {
            lr: 1.0e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.0,
        };
        let mut optimizer = RwkvPersistentAdamW::new(device.clone(), &trainables)?;
        optimizer.set_active_parameter_prefixes(&["active".to_string()])?;

        let frozen_before = frozen_parameter.read_f32(2)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut commands)?;
        optimizer.record_accumulate(&mut commands, &trainables)?;
        optimizer.record_step_with_named_gradient_override_and_inactive_names(
            &mut commands,
            &trainables,
            hyper,
            "frozen.weight",
            &override_gradient,
            &[],
        )?;
        commands.submit()?;

        let state = optimizer.state_snapshot()?;
        assert_eq!(state.step, 1);
        assert_eq!(state.slots[0].step, 1);
        assert_eq!(state.slots[1].step, 0);
        assert_eq!(state.slots[1].exp_avg, vec![0.0, 0.0]);
        assert_eq!(state.slots[1].exp_avg_sq, vec![0.0, 0.0]);
        assert_eq!(frozen_parameter.read_f32(2)?, frozen_before);
        Ok(())
    }

    #[test]
    fn canonical_gradient_shard_accumulation_reuses_transient_arena_buffers() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan transient gradient-shard test: {err:#}");
                return Ok(());
            }
        };
        let parameter = GpuBuffer::from_f32(&device, &[1.0, -2.0, 3.0, -4.0])?;
        let live_gradient = GpuBuffer::from_f32(&device, &[0.0; 4])?;
        let trainables = [RwkvTrainableRef {
            name: "weight",
            parameter: &parameter,
            gradient: &live_gradient,
            len: 4,
            decay_class: RwkvDecayClass::Decay,
        }];
        let optimizer = RwkvPersistentAdamW::new(device.clone(), &trainables)?;

        let first_shard = [RwkvParameterSnapshot {
            name: "weight".to_string(),
            values: vec![0.25, -0.5, 0.75, -1.0],
        }];
        let mut first = vulkan::ComputeBatch::new(&device)?;
        let first_sources = optimizer.record_accumulate_gradient_state_with_override(
            &mut first,
            &first_shard,
            None,
        )?;
        first.submit()?;
        drop(first_sources);
        let after_first = device.submission_arena_stats()?;
        assert!(after_first.scratch_slab_count >= 1);
        assert!(after_first.scratch_lease_allocated >= 1);

        let second_shard = [RwkvParameterSnapshot {
            name: "weight".to_string(),
            values: vec![1.0, 0.5, -0.25, 2.0],
        }];
        let mut second = vulkan::ComputeBatch::new(&device)?;
        let second_sources = optimizer.record_accumulate_gradient_state_with_override(
            &mut second,
            &second_shard,
            None,
        )?;
        second.submit()?;
        drop(second_sources);
        let after_second = device.submission_arena_stats()?;
        assert!(after_second.scratch_lease_reused > after_first.scratch_lease_reused);
        assert!(after_second.descriptor_pool_reused > after_first.descriptor_pool_reused);

        let accumulated = optimizer.gradient_state_snapshot_with_override(None)?;
        assert_eq!(accumulated.len(), 1);
        for (actual, expected) in accumulated[0].values.iter().zip([1.25, 0.0, 0.5, 1.0]) {
            assert!(
                (actual - expected).abs() <= 2.0e-7,
                "accumulated gradient {actual} did not match {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn fp16_storage_mirror_refreshes_in_same_adamw_command_stream() -> Result<()> {
        let device = match VulkanDevice::new() {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping Vulkan optimizer mirror test: {err:#}");
                return Ok(());
            }
        };
        let parameter = GpuBuffer::from_f32(&device, &[1.0, -2.0, 0.25, 3.5, -0.125])?;
        let gradient = GpuBuffer::from_f32(&device, &[0.2, -0.4, 0.1, 0.8, -0.05])?;
        let trainables = [RwkvTrainableRef {
            name: "weight",
            parameter: &parameter,
            gradient: &gradient,
            len: 5,
            decay_class: RwkvDecayClass::Decay,
        }];
        let mut optimizer = RwkvPersistentAdamW::new(device.clone(), &trainables)?;
        let bindings = optimizer.enable_parameter_storage_mirrors(
            &trainables,
            VulkanParameterStorageFormat::Fp16,
            &["weight"],
        )?;
        let mirror = bindings[0].mirror.clone();

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        optimizer.record_zero_grad(&mut commands)?;
        optimizer.record_accumulate(&mut commands, &trainables)?;
        optimizer.record_step(
            &mut commands,
            &trainables,
            AdamWHyperParams {
                lr: 1.0e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1.0e-8,
                weight_decay: 0.01,
            },
        )?;
        commands.submit()?;

        let master = parameter.read_f32(5)?;
        let packed_words = mirror
            .packed_storage()
            .read_f32(mirror.packed_words())?
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let compact = (0..mirror.len())
            .map(|index| {
                let word = packed_words[index / 2];
                let bits = if index % 2 == 0 {
                    word as u16
                } else {
                    (word >> 16) as u16
                };
                fp16_bits_to_f32(bits)
            })
            .collect::<Vec<_>>();

        for (index, (&master_value, &mirror_value)) in master.iter().zip(&compact).enumerate() {
            let tolerance = 1.0e-3_f32.max(master_value.abs() * 1.0e-3);
            assert!(
                (master_value - mirror_value).abs() <= tolerance,
                "mirror[{index}]={mirror_value} did not refresh from AdamW master {master_value}"
            );
        }
        assert_ne!(master, vec![1.0, -2.0, 0.25, 3.5, -0.125]);
        Ok(())
    }

    fn fp16_bits_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = ((bits >> 10) & 0x1f) as i32;
        let fraction = (bits & 0x03ff) as u32;
        match exponent {
            0 if fraction == 0 => sign * 0.0,
            0 => sign * (fraction as f32 / 1024.0) * 2.0_f32.powi(-14),
            31 if fraction == 0 => sign * f32::INFINITY,
            31 => f32::NAN,
            _ => sign * (1.0 + fraction as f32 / 1024.0) * 2.0_f32.powi(exponent - 15),
        }
    }
}
