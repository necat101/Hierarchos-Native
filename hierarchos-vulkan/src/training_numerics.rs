use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

const GRADIENT_NONFINITE_FLAG_SPV: &[u8] = include_bytes!("../shaders/gradient_nonfinite_flag.spv");
const GRADIENT_LASSQ_PARTIALS_SPV: &[u8] = include_bytes!("../shaders/gradient_lassq_partials.spv");
const GRADIENT_LASSQ_REDUCE_SPV: &[u8] = include_bytes!("../shaders/gradient_lassq_reduce.spv");
const GRADIENT_CLIP_COEFFICIENT_SPV: &[u8] =
    include_bytes!("../shaders/gradient_clip_coefficient.spv");
const ORDERED_F32_SUM_SPV: &[u8] = include_bytes!("../shaders/ordered_f32_sum.spv");
const DYNAMIC_LOSS_SCALE_CONTROL_SPV: &[u8] =
    include_bytes!("../shaders/dynamic_loss_scale_control.spv");
const LTM_ALIGNMENT_CONTROL_SPV: &[u8] = include_bytes!("../shaders/ltm_alignment_control.spv");
const BUFFER_ZERO_ON_OPTIMIZER_SKIP_SPV: &[u8] =
    include_bytes!("../shaders/buffer_zero_on_optimizer_skip.spv");
const GRADIENT_SCALE_FROM_BUFFER_SPV: &[u8] =
    include_bytes!("../shaders/gradient_scale_from_buffer.spv");
const GRADIENT_SCALE_STRIDED_FROM_BUFFER_SPV: &[u8] =
    include_bytes!("../shaders/gradient_scale_strided_from_buffer.spv");

pub(crate) const DYNAMIC_LOSS_SCALE_CONTROL_WORDS: usize = 8;
const LTM_ALIGNMENT_CONTROL_WORDS: usize = 12;

const CONTROL_SCALE_AFTER: usize = 0;
const CONTROL_GROWTH_TRACKER_LO: usize = 1;
const CONTROL_GROWTH_TRACKER_HI: usize = 2;
const CONTROL_SHOULD_STEP: usize = 3;
const CONTROL_OVERFLOWED: usize = 4;
pub(crate) const DYNAMIC_LOSS_SCALE_CONTROL_UNSCALE_FACTOR_WORD: usize = 5;
const CONTROL_SCALE_BEFORE: usize = 6;
const CONTROL_STATUS: usize = 7;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StridedScalePush {
    len: u32,
    stride: u32,
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LassqPartialsPush {
    len: u32,
    output_pair_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LassqReducePush {
    pair_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradientClipPush {
    max_norm: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DynamicLossScalePush {
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval_lo: u32,
    growth_interval_hi: u32,
    pending_gradients_scaled: u32,
    _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LtmAlignmentControlPush {
    ema_decay: f32,
    ready_threshold: f32,
    writer_max_norm: f32,
    cost_len: u32,
    exact_pytorch_tbptt: u32,
    sampled_rows_lo: u32,
    sampled_rows_hi: u32,
    controller_rows_lo: u32,
    controller_rows_hi: u32,
    min_updates_lo: u32,
    min_updates_hi: u32,
    step_predicate_mode: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VulkanDynamicLossScaleSeed {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u64,
    pub growth_tracker: u64,
    pub pending_gradients_scaled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VulkanDynamicLossScaleDecision {
    pub overflowed: bool,
    pub should_step: bool,
    pub unscale_factor: f32,
    pub scale_before: f32,
    pub scale_after: f32,
    pub growth_tracker: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VulkanLtmAlignmentSeed {
    pub updates: u64,
    pub last: Option<f32>,
    pub ema: Option<f32>,
    pub best: Option<f32>,
    pub writer_norm: Option<f32>,
    pub ready: bool,
    pub last_step_sampled_rows: u64,
    pub last_step_controller_sampled_rows: u64,
    pub sampled_rows: u64,
    pub controller_sampled_rows: u64,
    pub min_updates: u64,
    pub ready_threshold: f32,
    pub ema_decay: f32,
    pub writer_max_norm: f32,
    pub exact_pytorch_tbptt: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VulkanLtmAlignmentSnapshot {
    pub updates: u64,
    pub last: Option<f32>,
    pub ema: Option<f32>,
    pub best: Option<f32>,
    pub writer_norm: Option<f32>,
    pub ready: bool,
    pub last_step_sampled_rows: u64,
    pub last_step_controller_sampled_rows: u64,
}

/// Small deterministic Vulkan reduction used for parity-sensitive controller
/// metadata. Unlike the throughput-oriented parallel reducers, this intentionally
/// sums in increasing index order so replacing a Rust `Iterator::sum::<f32>()`
/// does not change the portable training trajectory's rounding order.
pub(crate) struct VulkanOrderedF32SumReducer {
    kernel: vulkan::ComputeKernel,
    scalar: GpuBuffer,
    readback: GpuBuffer,
}

impl VulkanOrderedF32SumReducer {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new_with_access(
                device,
                ORDERED_F32_SUM_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            scalar: GpuBuffer::zeros_f32(device, 1)?,
            readback: GpuBuffer::uninitialized_host_f32(device, 1)?,
        })
    }

    pub(crate) fn record_sum(
        &self,
        commands: &mut vulkan::ComputeBatch,
        values: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        if len == 0 || len > u32::MAX as usize || values.f32_capacity() < len {
            bail!(
                "ordered Vulkan FP32 sum length {len} is invalid for buffer capacity {}",
                values.f32_capacity()
            );
        }
        let push = LenPush { len: len as u32 };
        self.kernel.record_dispatch(
            commands,
            &[values, &self.scalar],
            bytemuck::bytes_of(&push),
            [1, 1, 1],
        )?;
        commands.readback_f32(&self.scalar, &self.readback, 1)
    }

    pub(crate) fn read_sum(&self) -> Result<f32> {
        self.readback
            .read_f32(1)?
            .into_iter()
            .next()
            .context("ordered Vulkan FP32 sum readback is empty")
    }
}

/// Persistent device-side GradScaler control state. The current scale and
/// growth tracker remain Vulkan-resident between optimizer windows. A host
/// upload happens only on first use or when checkpoint/resume state differs
/// from the last device result. The tiny readback is deliberately recorded at
/// the tail of the optimizer submission, after the control decision has already
/// gated parameter mutation; it exists only to mirror portable checkpoint and
/// telemetry state back into the Rust/PyTorch ABI.
pub(crate) struct VulkanDynamicLossScaleController {
    resolve_kernel: vulkan::ComputeKernel,
    zero_if_skip_kernel: vulkan::ComputeKernel,
    source_scale_kernel: vulkan::ComputeKernel,
    source_scale_strided_kernel: vulkan::ComputeKernel,
    state: GpuBuffer,
    readback: GpuBuffer,
    mirrored_scale_bits: Option<u32>,
    mirrored_growth_tracker: Option<u64>,
    authority: DynamicLossScaleAuthority,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DynamicLossScaleAuthority {
    Uninitialized,
    HostMirrored,
    DeviceOnly,
}

impl VulkanDynamicLossScaleController {
    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            resolve_kernel: vulkan::ComputeKernel::new_with_access(
                device,
                DYNAMIC_LOSS_SCALE_CONTROL_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadWrite,
                ],
                std::mem::size_of::<DynamicLossScalePush>() as u32,
            )?,
            zero_if_skip_kernel: vulkan::ComputeKernel::new_with_access(
                device,
                BUFFER_ZERO_ON_OPTIMIZER_SKIP_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            source_scale_kernel: vulkan::ComputeKernel::new_with_access(
                device,
                GRADIENT_SCALE_FROM_BUFFER_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            source_scale_strided_kernel: vulkan::ComputeKernel::new_with_access(
                device,
                GRADIENT_SCALE_STRIDED_FROM_BUFFER_SPV,
                &[
                    vulkan::BindingAccess::ReadWrite,
                    vulkan::BindingAccess::ReadOnly,
                ],
                std::mem::size_of::<StridedScalePush>() as u32,
            )?,
            state: GpuBuffer::zeros_u32(device, DYNAMIC_LOSS_SCALE_CONTROL_WORDS)?,
            // The readback copies raw u32 state words. As with the non-finite
            // detector, `read_f32().to_bits()` preserves those exact bytes.
            readback: GpuBuffer::zeros_host_f32(device, DYNAMIC_LOSS_SCALE_CONTROL_WORDS)?,
            mirrored_scale_bits: None,
            mirrored_growth_tracker: None,
            authority: DynamicLossScaleAuthority::Uninitialized,
        })
    }

    fn validate_seed(seed: VulkanDynamicLossScaleSeed) -> Result<()> {
        for (name, value) in [
            ("scale", seed.scale),
            ("growth_factor", seed.growth_factor),
            ("backoff_factor", seed.backoff_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("device dynamic loss-scale {name} must be finite and positive; got {value}");
            }
        }
        if seed.growth_interval == 0 {
            bail!("device dynamic loss-scale growth_interval must be positive");
        }
        let unscale = if seed.pending_gradients_scaled {
            1.0 / seed.scale
        } else {
            1.0
        };
        if !unscale.is_finite() || unscale <= 0.0 {
            bail!("device dynamic loss-scale reciprocal is non-finite or non-positive");
        }
        // Validate both possible branches before any optimizer mutation. This
        // keeps an invalid scaler configuration a host-side validation error
        // rather than allowing the safety shader to fail after gradients have
        // already been consumed/cleared.
        for (name, value) in [
            ("growth", seed.scale * seed.growth_factor),
            ("backoff", seed.scale * seed.backoff_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("device dynamic loss-scale {name} transition is invalid: {value}");
            }
        }
        Ok(())
    }

    fn record_seed_upload(
        &self,
        commands: &mut vulkan::ComputeBatch,
        seed: VulkanDynamicLossScaleSeed,
    ) -> Result<()> {
        let tracker_bytes = seed.growth_tracker.to_le_bytes();
        let tracker_lo = u32::from_le_bytes(tracker_bytes[0..4].try_into().unwrap());
        let tracker_hi = u32::from_le_bytes(tracker_bytes[4..8].try_into().unwrap());
        commands.upload_u32(
            &self.state,
            &[seed.scale.to_bits(), tracker_lo, tracker_hi, 0, 0, 0, 0, 0],
        )
    }

    fn record_transition(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        overflow_flag: &GpuBuffer,
        seed: VulkanDynamicLossScaleSeed,
    ) -> Result<()> {
        let interval_bytes = seed.growth_interval.to_le_bytes();
        let push = DynamicLossScalePush {
            growth_factor: seed.growth_factor,
            backoff_factor: seed.backoff_factor,
            growth_interval_lo: u32::from_le_bytes(interval_bytes[0..4].try_into().unwrap()),
            growth_interval_hi: u32::from_le_bytes(interval_bytes[4..8].try_into().unwrap()),
            pending_gradients_scaled: u32::from(seed.pending_gradients_scaled),
            _reserved: 0,
        };
        self.resolve_kernel.record_dispatch(
            commands,
            &[overflow_flag, &self.state],
            bytemuck::bytes_of(&push),
            [1, 1, 1],
        )?;
        // Once this dispatch is queued, scale/tracker ownership moves to the
        // Vulkan command stream. The host mirror is no longer authoritative
        // until an explicit readback completes.
        self.authority = DynamicLossScaleAuthority::DeviceOnly;
        Ok(())
    }

    /// Resolve one scaler transition while treating `seed` as host-authoritative.
    /// This preserves the public checkpoint/PyTorch-state contract: an explicit
    /// host scaler edit or restore overrides any device-only progress that has
    /// not been mirrored back yet.
    pub(crate) fn record_resolve(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        overflow_flag: &GpuBuffer,
        seed: VulkanDynamicLossScaleSeed,
    ) -> Result<()> {
        Self::validate_seed(seed)?;
        if overflow_flag.f32_capacity() < 1 {
            bail!("device dynamic loss-scale overflow flag is empty");
        }
        let seed_matches_host_mirror = self.mirrored_scale_bits == Some(seed.scale.to_bits())
            && self.mirrored_growth_tracker == Some(seed.growth_tracker);
        if self.authority != DynamicLossScaleAuthority::HostMirrored || !seed_matches_host_mirror {
            self.record_seed_upload(commands, seed)?;
        }
        self.record_transition(commands, overflow_flag, seed)
    }

    /// Resolve one scaler transition while keeping the current scale/tracker
    /// device-authoritative. The supplied seed is used only to initialize a new
    /// controller; after that, stale host scale/tracker values are deliberately
    /// ignored. Growth/backoff configuration and the per-window scaled-gradient
    /// bit still come from the seed's push constants.
    ///
    /// This is the queue-resident training mode: many optimizer windows can run
    /// without a scaler readback, and checkpoint/telemetry can synchronize the
    /// tiny state only when needed.
    pub(crate) fn record_resolve_device_resident(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        overflow_flag: &GpuBuffer,
        seed: VulkanDynamicLossScaleSeed,
    ) -> Result<()> {
        Self::validate_seed(seed)?;
        if overflow_flag.f32_capacity() < 1 {
            bail!("device dynamic loss-scale overflow flag is empty");
        }
        let seed_matches_host_mirror = self.mirrored_scale_bits == Some(seed.scale.to_bits())
            && self.mirrored_growth_tracker == Some(seed.growth_tracker);
        if self.authority == DynamicLossScaleAuthority::Uninitialized
            || (self.authority == DynamicLossScaleAuthority::HostMirrored
                && !seed_matches_host_mirror)
        {
            self.record_seed_upload(commands, seed)?;
        }
        self.record_transition(commands, overflow_flag, seed)
    }

    pub(crate) fn control_buffer(&self) -> &GpuBuffer {
        &self.state
    }

    /// Multiply an already-materialized backward source by the current
    /// device-resident GradScaler scale. `state[0]` is deliberately both the
    /// previous transition's scale-after and the next window's scale-before,
    /// so this dispatch can feed the next backward pass without a four-byte
    /// host readback between optimizer windows.
    pub(crate) fn record_scale_source_by_current_scale(
        &self,
        commands: &mut vulkan::ComputeBatch,
        source: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        if self.authority == DynamicLossScaleAuthority::Uninitialized {
            bail!("device dynamic loss-scale source requested before controller initialization");
        }
        if len == 0 || len > u32::MAX as usize || source.f32_capacity() < len {
            bail!(
                "device dynamic loss-scale source length {len} is invalid for buffer capacity {}",
                source.f32_capacity()
            );
        }
        self.source_scale_kernel.record_dispatch(
            commands,
            &[source, &self.state],
            bytemuck::bytes_of(&LenPush { len: len as u32 }),
            [len.div_ceil(256) as u32, 1, 1],
        )
    }

    /// Multiply one field from each fixed-width device record by the current
    /// GradScaler scale without disturbing neighboring forward/telemetry data.
    pub(crate) fn record_scale_source_by_current_scale_strided(
        &self,
        commands: &mut vulkan::ComputeBatch,
        source: &GpuBuffer,
        len: usize,
        stride: usize,
        offset: usize,
    ) -> Result<()> {
        if self.authority == DynamicLossScaleAuthority::Uninitialized {
            bail!("device dynamic loss-scale source requested before controller initialization");
        }
        if len == 0 || stride == 0 {
            bail!("device dynamic loss-scale strided source requires positive length/stride");
        }
        let last_index = offset
            .checked_add((len - 1).checked_mul(stride).ok_or_else(|| {
                anyhow::anyhow!("device dynamic loss-scale strided source index overflow")
            })?)
            .ok_or_else(|| {
                anyhow::anyhow!("device dynamic loss-scale strided source index overflow")
            })?;
        if len > u32::MAX as usize
            || stride > u32::MAX as usize
            || offset > u32::MAX as usize
            || last_index >= source.f32_capacity()
        {
            bail!(
                "device dynamic loss-scale strided source len={len} stride={stride} offset={offset} exceeds buffer capacity {} or Vulkan u32 indexing",
                source.f32_capacity()
            );
        }
        let push = StridedScalePush {
            len: len as u32,
            stride: stride as u32,
            offset: offset as u32,
        };
        self.source_scale_strided_kernel.record_dispatch(
            commands,
            &[source, &self.state],
            bytemuck::bytes_of(&push),
            [len.div_ceil(256) as u32, 1, 1],
        )
    }

    pub(crate) fn host_scale_metadata_stale(&self) -> bool {
        self.authority == DynamicLossScaleAuthority::DeviceOnly
    }

    pub(crate) fn record_zero_if_skipped(
        &self,
        commands: &mut vulkan::ComputeBatch,
        buffer: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        if len == 0 || len > u32::MAX as usize || buffer.f32_capacity() < len {
            bail!(
                "device controlled zero length {len} is invalid for buffer capacity {}",
                buffer.f32_capacity()
            );
        }
        self.zero_if_skip_kernel.record_dispatch(
            commands,
            &[buffer, &self.state],
            bytemuck::bytes_of(&LenPush { len: len as u32 }),
            [len.div_ceil(256) as u32, 1, 1],
        )
    }

    pub(crate) fn record_readback(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        commands.readback_f32(
            &self.state,
            &self.readback,
            DYNAMIC_LOSS_SCALE_CONTROL_WORDS,
        )
    }

    pub(crate) fn read_decision(&mut self) -> Result<VulkanDynamicLossScaleDecision> {
        self.read_decision_impl(true)
    }

    /// Read a previously requested controller observation without transferring
    /// scaler ownership back to the host. This is used by host-owned auxiliary
    /// controllers (currently LTM readiness) that need the step/skip bit while
    /// the next optimizer window must continue sourcing its scale from Vulkan.
    pub(crate) fn read_decision_preserving_device_authority(
        &mut self,
    ) -> Result<VulkanDynamicLossScaleDecision> {
        self.read_decision_impl(false)
    }

    fn read_decision_impl(
        &mut self,
        mirror_host_metadata: bool,
    ) -> Result<VulkanDynamicLossScaleDecision> {
        let words = self
            .readback
            .read_f32(DYNAMIC_LOSS_SCALE_CONTROL_WORDS)?
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let status = words[CONTROL_STATUS];
        if status != 0 {
            bail!("device dynamic loss-scale controller reported status {status}");
        }
        let scale_before = f32::from_bits(words[CONTROL_SCALE_BEFORE]);
        let scale_after = f32::from_bits(words[CONTROL_SCALE_AFTER]);
        let unscale_factor = f32::from_bits(words[DYNAMIC_LOSS_SCALE_CONTROL_UNSCALE_FACTOR_WORD]);
        for (name, value) in [
            ("scale_before", scale_before),
            ("scale_after", scale_after),
            ("unscale_factor", unscale_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("device dynamic loss-scale result {name} is invalid: {value}");
            }
        }
        let tracker = u64::from(words[CONTROL_GROWTH_TRACKER_LO])
            | (u64::from(words[CONTROL_GROWTH_TRACKER_HI]) << 32);
        let overflowed = words[CONTROL_OVERFLOWED] != 0;
        let should_step = words[CONTROL_SHOULD_STEP] != 0;
        if should_step == overflowed {
            bail!(
                "device dynamic loss-scale result has inconsistent step/overflow bits: step={should_step}, overflow={overflowed}"
            );
        }
        if mirror_host_metadata {
            self.mirrored_scale_bits = Some(scale_after.to_bits());
            self.mirrored_growth_tracker = Some(tracker);
            self.authority = DynamicLossScaleAuthority::HostMirrored;
        }
        Ok(VulkanDynamicLossScaleDecision {
            overflowed,
            should_step,
            unscale_factor,
            scale_before,
            scale_after,
            growth_tracker: tracker,
        })
    }
}

const LTM_FLAG_HAS_LAST: u32 = 1 << 0;
const LTM_FLAG_HAS_EMA: u32 = 1 << 1;
const LTM_FLAG_HAS_BEST: u32 = 1 << 2;
const LTM_FLAG_HAS_WRITER_NORM: u32 = 1 << 3;
const LTM_FLAG_READY: u32 = 1 << 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LtmAlignmentAuthority {
    Uninitialized,
    HostMirrored,
    DeviceOnly,
}

/// Persistent Vulkan-side copy of the small LTM writer-readiness controller.
/// The objective rows, writer norm, EMA/best tracking, update counter, and
/// ready bit advance behind the same GradScaler step predicate as AdamW. The
/// host mirror is intentionally stale during ordinary training and is only
/// refreshed for checkpoint/telemetry boundaries.
pub(crate) struct VulkanLtmAlignmentController {
    kernel: vulkan::ComputeKernel,
    state: GpuBuffer,
    readback: GpuBuffer,
    authority: LtmAlignmentAuthority,
}

impl VulkanLtmAlignmentController {
    const STEP_PREDICATE_GRAD_SCALER: u32 = 0;
    const STEP_PREDICATE_NONFINITE_FLAG: u32 = 1;
    const STEP_PREDICATE_COMMITTED: u32 = 2;

    pub(crate) fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new_with_access(
                device,
                LTM_ALIGNMENT_CONTROL_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadWrite,
                ],
                std::mem::size_of::<LtmAlignmentControlPush>() as u32,
            )?,
            state: GpuBuffer::zeros_u32(device, LTM_ALIGNMENT_CONTROL_WORDS)?,
            readback: GpuBuffer::zeros_host_f32(device, LTM_ALIGNMENT_CONTROL_WORDS)?,
            authority: LtmAlignmentAuthority::Uninitialized,
        })
    }

    fn split_u64(value: u64) -> (u32, u32) {
        (value as u32, (value >> 32) as u32)
    }

    fn validate_optional(name: &str, value: Option<f32>) -> Result<()> {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            bail!("device LTM alignment {name} must be finite and non-negative");
        }
        Ok(())
    }

    fn validate_seed(seed: VulkanLtmAlignmentSeed) -> Result<()> {
        Self::validate_optional("last", seed.last)?;
        Self::validate_optional("ema", seed.ema)?;
        Self::validate_optional("best", seed.best)?;
        Self::validate_optional("writer_norm", seed.writer_norm)?;
        if seed.min_updates == 0
            || !seed.ready_threshold.is_finite()
            || seed.ready_threshold < 0.0
            || !seed.ema_decay.is_finite()
            || !(0.0..1.0).contains(&seed.ema_decay)
            || !seed.writer_max_norm.is_finite()
            || seed.writer_max_norm < 0.0
        {
            bail!("invalid device LTM alignment readiness policy");
        }
        Ok(())
    }

    fn record_seed_upload(
        &self,
        commands: &mut vulkan::ComputeBatch,
        seed: VulkanLtmAlignmentSeed,
    ) -> Result<()> {
        let (updates_lo, updates_hi) = Self::split_u64(seed.updates);
        let (sampled_lo, sampled_hi) = Self::split_u64(seed.last_step_sampled_rows);
        let (controller_lo, controller_hi) =
            Self::split_u64(seed.last_step_controller_sampled_rows);
        let mut flags = 0u32;
        if seed.last.is_some() {
            flags |= LTM_FLAG_HAS_LAST;
        }
        if seed.ema.is_some() {
            flags |= LTM_FLAG_HAS_EMA;
        }
        if seed.best.is_some() {
            flags |= LTM_FLAG_HAS_BEST;
        }
        if seed.writer_norm.is_some() {
            flags |= LTM_FLAG_HAS_WRITER_NORM;
        }
        if seed.ready {
            flags |= LTM_FLAG_READY;
        }
        commands.upload_u32(
            &self.state,
            &[
                updates_lo,
                updates_hi,
                seed.last.unwrap_or(0.0).to_bits(),
                seed.ema.unwrap_or(0.0).to_bits(),
                seed.best.unwrap_or(0.0).to_bits(),
                seed.writer_norm.unwrap_or(0.0).to_bits(),
                flags,
                0,
                sampled_lo,
                sampled_hi,
                controller_lo,
                controller_hi,
            ],
        )
    }

    pub(crate) fn record_step_device_resident(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        costs: &GpuBuffer,
        cost_len: usize,
        writer_l2_pair: &GpuBuffer,
        grad_scaler_control: &GpuBuffer,
        seed: VulkanLtmAlignmentSeed,
    ) -> Result<()> {
        self.record_step_device_resident_impl(
            commands,
            costs,
            cost_len,
            writer_l2_pair,
            grad_scaler_control,
            seed,
            Self::STEP_PREDICATE_GRAD_SCALER,
        )
    }

    /// FP32 counterpart of the GradScaler-controlled device transition. A
    /// zero non-finite flag means AdamW committed, while any non-zero value
    /// leaves the persistent readiness history untouched. This lets ordinary
    /// clipped training keep its optimizer safety decision and LTM retirement
    /// on Vulkan without manufacturing or mirroring a fake GradScaler state.
    pub(crate) fn record_step_device_resident_with_nonfinite_flag(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        costs: &GpuBuffer,
        cost_len: usize,
        writer_l2_pair: &GpuBuffer,
        nonfinite_flag: &GpuBuffer,
        seed: VulkanLtmAlignmentSeed,
    ) -> Result<()> {
        self.record_step_device_resident_impl(
            commands,
            costs,
            cost_len,
            writer_l2_pair,
            nonfinite_flag,
            seed,
            Self::STEP_PREDICATE_NONFINITE_FLAG,
        )
    }

    /// Advance readiness after an optimizer boundary that is guaranteed to
    /// commit. Plain FP32 AdamW has no GradScaler/non-finite skip predicate, so
    /// keeping this transition on Vulkan must not manufacture a host-visible
    /// control word merely to express `true`. Binding the already-required
    /// writer pair twice keeps the descriptor ABI fixed; predicate mode 2 never
    /// reads binding 2.
    pub(crate) fn record_step_device_resident_committed(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        costs: &GpuBuffer,
        cost_len: usize,
        writer_l2_pair: &GpuBuffer,
        seed: VulkanLtmAlignmentSeed,
    ) -> Result<()> {
        self.record_step_device_resident_impl(
            commands,
            costs,
            cost_len,
            writer_l2_pair,
            writer_l2_pair,
            seed,
            Self::STEP_PREDICATE_COMMITTED,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_step_device_resident_impl(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        costs: &GpuBuffer,
        cost_len: usize,
        writer_l2_pair: &GpuBuffer,
        step_predicate: &GpuBuffer,
        seed: VulkanLtmAlignmentSeed,
        step_predicate_mode: u32,
    ) -> Result<()> {
        Self::validate_seed(seed)?;
        if cost_len == 0 || cost_len > u32::MAX as usize || costs.f32_capacity() < cost_len {
            bail!(
                "device LTM alignment cost length {cost_len} is invalid for buffer capacity {}",
                costs.f32_capacity()
            );
        }
        if writer_l2_pair.f32_capacity() < 2 {
            bail!("device LTM alignment writer L2 pair is incomplete");
        }
        match step_predicate_mode {
            Self::STEP_PREDICATE_GRAD_SCALER
                if step_predicate.f32_capacity() < DYNAMIC_LOSS_SCALE_CONTROL_WORDS =>
            {
                bail!("device LTM alignment requires the complete GradScaler control buffer");
            }
            Self::STEP_PREDICATE_NONFINITE_FLAG if step_predicate.f32_capacity() < 1 => {
                bail!("device LTM alignment requires a non-empty non-finite flag buffer");
            }
            Self::STEP_PREDICATE_GRAD_SCALER
            | Self::STEP_PREDICATE_NONFINITE_FLAG
            | Self::STEP_PREDICATE_COMMITTED => {}
            _ => bail!("device LTM alignment received an unknown step-predicate mode"),
        }
        if self.authority != LtmAlignmentAuthority::DeviceOnly {
            self.record_seed_upload(commands, seed)?;
        }
        let (sampled_rows_lo, sampled_rows_hi) = Self::split_u64(seed.sampled_rows);
        let (controller_rows_lo, controller_rows_hi) =
            Self::split_u64(seed.controller_sampled_rows);
        let (min_updates_lo, min_updates_hi) = Self::split_u64(seed.min_updates);
        let push = LtmAlignmentControlPush {
            ema_decay: seed.ema_decay,
            ready_threshold: seed.ready_threshold,
            writer_max_norm: seed.writer_max_norm,
            cost_len: cost_len as u32,
            exact_pytorch_tbptt: u32::from(seed.exact_pytorch_tbptt),
            sampled_rows_lo,
            sampled_rows_hi,
            controller_rows_lo,
            controller_rows_hi,
            min_updates_lo,
            min_updates_hi,
            step_predicate_mode,
        };
        self.kernel.record_dispatch(
            commands,
            &[costs, writer_l2_pair, step_predicate, &self.state],
            bytemuck::bytes_of(&push),
            [1, 1, 1],
        )?;
        self.authority = LtmAlignmentAuthority::DeviceOnly;
        Ok(())
    }

    pub(crate) fn host_metadata_stale(&self) -> bool {
        self.authority == LtmAlignmentAuthority::DeviceOnly
    }

    /// A host-side restore/legacy update becomes authoritative at the next
    /// device transition. No GPU mutation is needed until that transition.
    pub(crate) fn invalidate_device_authority(&mut self) {
        self.authority = LtmAlignmentAuthority::Uninitialized;
    }

    pub(crate) fn record_readback(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        commands.readback_f32(&self.state, &self.readback, LTM_ALIGNMENT_CONTROL_WORDS)
    }

    pub(crate) fn read_snapshot(&mut self) -> Result<VulkanLtmAlignmentSnapshot> {
        let words = self
            .readback
            .read_f32(LTM_ALIGNMENT_CONTROL_WORDS)?
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let status = words[7];
        if status != 0 {
            bail!("device LTM alignment controller reported status {status}");
        }
        let flags = words[6];
        let decode_optional = |bit: u32, index: usize| -> Result<Option<f32>> {
            if flags & bit == 0 {
                return Ok(None);
            }
            let value = f32::from_bits(words[index]);
            if !value.is_finite() || value < 0.0 {
                bail!("device LTM alignment controller produced invalid scalar {value}");
            }
            Ok(Some(value))
        };
        let snapshot = VulkanLtmAlignmentSnapshot {
            updates: u64::from(words[0]) | (u64::from(words[1]) << 32),
            last: decode_optional(LTM_FLAG_HAS_LAST, 2)?,
            ema: decode_optional(LTM_FLAG_HAS_EMA, 3)?,
            best: decode_optional(LTM_FLAG_HAS_BEST, 4)?,
            writer_norm: decode_optional(LTM_FLAG_HAS_WRITER_NORM, 5)?,
            ready: flags & LTM_FLAG_READY != 0,
            last_step_sampled_rows: u64::from(words[8]) | (u64::from(words[9]) << 32),
            last_step_controller_sampled_rows: u64::from(words[10]) | (u64::from(words[11]) << 32),
        };
        self.authority = LtmAlignmentAuthority::HostMirrored;
        Ok(snapshot)
    }
}

/// Robust device-side L2 reduction for the complete optimizer gradient
/// registry. The reduction carries LAPACK-style `(scale, ssq)` pairs rather
/// than raw fp32 sums of squares, so a finite gradient vector can have a norm
/// above `f32::MAX` without spuriously turning into an overflow decision.
///
/// Only the final two-float pair plus tiny clip/safety telemetry are read by the
/// host. This is deliberately the same control-plane shape as PyTorch's global
/// `clip_grad_norm_`: individual gradient tensors stay Vulkan-resident, and the
/// reduction itself owns non-finite detection for clipped optimizer steps.
pub(crate) struct VulkanGradientL2NormReducer {
    partials: vulkan::ComputeKernel,
    reduce: vulkan::ComputeKernel,
    clip_coefficient_kernel: vulkan::ComputeKernel,
    scratch_a: GpuBuffer,
    scratch_b: GpuBuffer,
    readback: GpuBuffer,
    clip_coefficient: GpuBuffer,
    clip_coefficient_readback: GpuBuffer,
    clip_nonfinite: GpuBuffer,
    clip_nonfinite_readback: GpuBuffer,
    max_partial_pairs: usize,
}

impl VulkanGradientL2NormReducer {
    pub(crate) fn new(device: &VulkanDevice, gradient_lengths: &[usize]) -> Result<Self> {
        if gradient_lengths.is_empty() || gradient_lengths.contains(&0) {
            bail!("gradient L2 reducer requires non-empty positive tensor lengths");
        }
        let max_partial_pairs = gradient_lengths.iter().try_fold(0usize, |count, &len| {
            if len > u32::MAX as usize {
                bail!("gradient L2 tensor length exceeds Vulkan u32 range: {len}");
            }
            count
                .checked_add(len.div_ceil(256))
                .ok_or_else(|| anyhow::anyhow!("gradient L2 partial-pair count overflow"))
        })?;
        if max_partial_pairs == 0 || max_partial_pairs > u32::MAX as usize {
            bail!(
                "gradient L2 partial-pair capacity must be in 1..=u32::MAX; got {max_partial_pairs}"
            );
        }
        let scratch_values = max_partial_pairs
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("gradient L2 scratch size overflow"))?;
        Ok(Self {
            partials: vulkan::ComputeKernel::new_with_access(
                device,
                GRADIENT_LASSQ_PARTIALS_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<LassqPartialsPush>() as u32,
            )?,
            reduce: vulkan::ComputeKernel::new_with_access(
                device,
                GRADIENT_LASSQ_REDUCE_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<LassqReducePush>() as u32,
            )?,
            clip_coefficient_kernel: vulkan::ComputeKernel::new_with_access(
                device,
                GRADIENT_CLIP_COEFFICIENT_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<GradientClipPush>() as u32,
            )?,
            scratch_a: GpuBuffer::zeros_f32(device, scratch_values)?,
            scratch_b: GpuBuffer::zeros_f32(device, scratch_values)?,
            readback: GpuBuffer::zeros_host_f32(device, 2)?,
            clip_coefficient: GpuBuffer::zeros_f32(device, 1)?,
            clip_coefficient_readback: GpuBuffer::zeros_host_f32(device, 1)?,
            clip_nonfinite: GpuBuffer::zeros_u32(device, 1)?,
            clip_nonfinite_readback: GpuBuffer::zeros_host_f32(device, 1)?,
            max_partial_pairs,
        })
    }

    fn record_l2_pair<'a>(
        &'a self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
    ) -> Result<&'a GpuBuffer> {
        if gradients.is_empty() {
            bail!("gradient L2 reduction requires at least one active gradient tensor");
        }
        let mut pair_count = 0usize;
        for &(gradient, len) in gradients {
            if len == 0 || len > u32::MAX as usize {
                bail!("gradient L2 scan length must be in 1..=u32::MAX; got {len}");
            }
            if gradient.f32_capacity() < len {
                bail!(
                    "gradient L2 scan length {len} exceeds buffer capacity {}",
                    gradient.f32_capacity()
                );
            }
            let groups = len.div_ceil(256);
            let next_pair_count = pair_count
                .checked_add(groups)
                .ok_or_else(|| anyhow::anyhow!("gradient L2 partial-pair offset overflow"))?;
            if next_pair_count > self.max_partial_pairs || next_pair_count > u32::MAX as usize {
                bail!(
                    "gradient L2 reduction needs {next_pair_count} partial pairs; capacity is {}",
                    self.max_partial_pairs
                );
            }
            self.partials.record_dispatch(
                commands,
                &[gradient, &self.scratch_a],
                bytemuck::bytes_of(&LassqPartialsPush {
                    len: len as u32,
                    output_pair_offset: pair_count as u32,
                }),
                [groups as u32, 1, 1],
            )?;
            pair_count = next_pair_count;
        }

        let mut input_is_a = true;
        while pair_count > 1 {
            let output_pairs = pair_count.div_ceil(256);
            let (input, output) = if input_is_a {
                (&self.scratch_a, &self.scratch_b)
            } else {
                (&self.scratch_b, &self.scratch_a)
            };
            self.reduce.record_dispatch(
                commands,
                &[input, output],
                bytemuck::bytes_of(&LassqReducePush {
                    pair_count: pair_count as u32,
                }),
                [output_pairs as u32, 1, 1],
            )?;
            pair_count = output_pairs;
            input_is_a = !input_is_a;
        }
        Ok(if input_is_a {
            &self.scratch_a
        } else {
            &self.scratch_b
        })
    }

    /// Record only the Vulkan reduction and return a retained handle to the
    /// final `(scale, ssq)` pair. Auxiliary device controllers can consume the
    /// pair directly without introducing a host-visible norm observation.
    pub(crate) fn record_l2_pair_device_only(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
    ) -> Result<GpuBuffer> {
        Ok(self.record_l2_pair(commands, gradients)?.clone())
    }

    /// Record a robust L2 reduction and copy only the final `(scale, ssq)` pair
    /// to the host. This is also useful for small control-plane observations
    /// such as LTM writer readiness: the source tensor remains device-resident
    /// regardless of its size.
    pub(crate) fn record_l2_norm(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
    ) -> Result<()> {
        let result = self.record_l2_pair(commands, gradients)?;
        commands.readback_f32(result, &self.readback, 2)
    }

    /// Record robust global-norm reduction plus the PyTorch clipping scalar and
    /// non-finite safety bit entirely on Vulkan. Norm/coefficient readbacks are
    /// telemetry. Device-controlled GradScaler callers feed the safety bit
    /// directly into their optimizer-control buffer; legacy host-resolved paths
    /// may still inspect its tiny readback before parameter mutation.
    pub(crate) fn record_l2_norm_and_clip_coefficient(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
        max_norm: f32,
    ) -> Result<()> {
        let result =
            self.record_l2_norm_and_clip_coefficient_inner(commands, gradients, max_norm)?;
        commands.readback_f32(result, &self.readback, 2)?;
        commands.readback_f32(&self.clip_coefficient, &self.clip_coefficient_readback, 1)?;
        commands.readback_f32(&self.clip_nonfinite, &self.clip_nonfinite_readback, 1)
    }

    /// Device-only form of the clipping reduction. The coefficient and safety
    /// bit feed later Vulkan dispatches directly; host-visible norm telemetry
    /// is deliberately omitted from ordinary queue-resident training steps.
    pub(crate) fn record_l2_norm_and_clip_coefficient_device_only(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
        max_norm: f32,
    ) -> Result<()> {
        self.record_l2_norm_and_clip_coefficient_inner(commands, gradients, max_norm)
            .map(|_| ())
    }

    fn record_l2_norm_and_clip_coefficient_inner<'a>(
        &'a self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
        max_norm: f32,
    ) -> Result<&'a GpuBuffer> {
        if !max_norm.is_finite() || max_norm < 0.0 {
            bail!("gradient clip max norm must be finite and non-negative; got {max_norm}");
        }
        let result = self.record_l2_pair(commands, gradients)?;
        self.clip_coefficient_kernel.record_dispatch(
            commands,
            &[result, &self.clip_coefficient, &self.clip_nonfinite],
            bytemuck::bytes_of(&GradientClipPush { max_norm }),
            [1, 1, 1],
        )?;
        Ok(result)
    }

    pub(crate) fn read_l2_norm(&self) -> Result<f64> {
        let pair = self.readback.read_f32(2)?;
        let scale = f64::from(pair[0]);
        let ssq = f64::from(pair[1]);
        if !scale.is_finite() || !ssq.is_finite() || scale < 0.0 || ssq < 0.0 {
            return Ok(f64::NAN);
        }
        Ok(scale * ssq.sqrt())
    }

    pub(crate) fn clip_coefficient_buffer(&self) -> &GpuBuffer {
        &self.clip_coefficient
    }

    pub(crate) fn clip_nonfinite_buffer(&self) -> &GpuBuffer {
        &self.clip_nonfinite
    }

    pub(crate) fn read_clip_coefficient(&self) -> Result<f32> {
        let coefficient = self.clip_coefficient_readback.read_f32(1)?[0];
        if !coefficient.is_finite() || !(0.0..=1.0).contains(&coefficient) {
            bail!("device gradient clip coefficient is invalid: {coefficient}");
        }
        Ok(coefficient)
    }

    pub(crate) fn read_clip_has_nonfinite(&self) -> Result<bool> {
        Ok(self.clip_nonfinite_readback.read_f32(1)?[0].to_bits() != 0)
    }
}

/// Device-side reduction for AMP overflow detection. All gradient buffers are
/// scanned in one queue submission and atomically OR into one 32-bit flag.
/// Host-resolved callers can mirror four bytes; device-controlled GradScaler
/// paths consume the flag in-place without a safety synchronization.
pub struct VulkanGradientNonfiniteDetector {
    device: VulkanDevice,
    kernel: vulkan::ComputeKernel,
    flag: GpuBuffer,
    readback: GpuBuffer,
}

impl VulkanGradientNonfiniteDetector {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new_with_access(
                &device,
                GRADIENT_NONFINITE_FLAG_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
            flag: GpuBuffer::zeros_u32(&device, 1)?,
            // Readback copies raw 32-bit bytes. `read_f32()[0].to_bits()` below
            // intentionally interprets those bytes as the original uint flag.
            readback: GpuBuffer::zeros_host_f32(&device, 1)?,
            device,
        })
    }

    pub fn has_nonfinite(&self, gradients: &[(&GpuBuffer, usize)]) -> Result<bool> {
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.record_scan(&mut commands, gradients)?;
        commands.submit()?;
        self.read_has_nonfinite()
    }

    /// Record the non-finite reduction into an existing command stream. This is
    /// used by the optimizer safety boundary so normalization/unscale and the
    /// four-byte overflow decision can share one submission.
    pub(crate) fn record_scan(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
    ) -> Result<()> {
        self.record_scan_device_only(commands, gradients)?;
        self.record_readback(commands)
    }

    /// Device-only form used by GPU-resident optimizer control. The flag stays
    /// in device-local memory so a following control shader can decide whether
    /// AdamW mutates parameters without an intervening CPU synchronization.
    pub(crate) fn record_scan_device_only(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradients: &[(&GpuBuffer, usize)],
    ) -> Result<()> {
        commands.fill_zero_f32(&self.flag, 1)?;
        for &(gradient, len) in gradients {
            if len == 0 || len > u32::MAX as usize {
                bail!("gradient non-finite scan length must be in 1..=u32::MAX; got {len}");
            }
            if gradient.f32_capacity() < len {
                bail!(
                    "gradient non-finite scan length {len} exceeds buffer capacity {}",
                    gradient.f32_capacity()
                );
            }
            self.kernel.record_dispatch(
                commands,
                &[gradient, &self.flag],
                bytemuck::bytes_of(&LenPush { len: len as u32 }),
                [len.div_ceil(256) as u32, 1, 1],
            )?;
        }
        Ok(())
    }

    pub(crate) fn record_readback(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        commands.readback_f32(&self.flag, &self.readback, 1)
    }

    pub(crate) fn flag_buffer(&self) -> &GpuBuffer {
        &self.flag
    }

    pub(crate) fn read_has_nonfinite(&self) -> Result<bool> {
        Ok(self.readback.read_f32(1)?[0].to_bits() != 0)
    }

    /// Rebind the device-local reduction flag to graph-owned transient storage.
    /// The host-visible readback remains detector-owned because it belongs to a
    /// different Vulkan memory class and cannot participate in the device-local
    /// lifetime arena.
    pub(crate) fn bind_flag_buffer(&mut self, flag: GpuBuffer) -> Result<()> {
        if flag.f32_capacity() < 1 {
            bail!("gradient non-finite flag buffer must contain at least one 32-bit value");
        }
        self.flag = flag;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUALIFICATION_DEVICE_INDEX_ENV: &str = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX";

    fn test_device() -> Result<Option<VulkanDevice>> {
        match std::env::var(QUALIFICATION_DEVICE_INDEX_ENV) {
            Ok(raw) => {
                let index = raw.parse::<usize>().map_err(|err| {
                    anyhow::anyhow!(
                        "{QUALIFICATION_DEVICE_INDEX_ENV} must be a non-negative device index: {err}"
                    )
                })?;
                Ok(Some(VulkanDevice::new_with_index(index)?))
            }
            Err(std::env::VarError::NotPresent) => Ok(VulkanDevice::new().ok()),
            Err(err) => Err(anyhow::anyhow!(
                "reading {QUALIFICATION_DEVICE_INDEX_ENV}: {err}"
            )),
        }
    }

    #[test]
    fn vulkan_ordered_f32_sum_matches_rust_left_to_right_rounding() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        // Reassociation would produce 16_777_218 here, while Rust's ordinary
        // left-to-right FP32 iterator sum produces 16_777_216. Keep this vector
        // deliberately rounding-sensitive so the portability contract is
        // covered rather than merely checking an easy exact sum.
        let values = [16_777_216.0f32, 1.0, 1.0];
        let input = GpuBuffer::from_f32(&device, &values)?;
        let reducer = VulkanOrderedF32SumReducer::new(&device)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        reducer.record_sum(&mut commands, &input, values.len())?;
        commands.submit()?;

        let observed = reducer.read_sum()?;
        let expected = values.into_iter().sum::<f32>();
        assert_eq!(observed.to_bits(), expected.to_bits());
        assert_eq!(observed, 16_777_216.0);
        Ok(())
    }

    #[test]
    fn vulkan_nonfinite_detector_finds_inf_and_nan_written_on_device() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let detector = VulkanGradientNonfiniteDetector::new(device.clone())?;
        let finite = GpuBuffer::from_f32(&device, &[1.0, -2.0, 3.0, 0.0])?;
        assert!(!detector.has_nonfinite(&[(&finite, 4)])?);

        // `write_f32` deliberately rejects non-finite host uploads, so create
        // the exact IEEE bit patterns through the u32 upload path. The shader
        // still sees the storage buffer as float values.
        let nonfinite = GpuBuffer::from_u32(
            &device,
            &[
                1.0f32.to_bits(),
                f32::INFINITY.to_bits(),
                f32::NAN.to_bits(),
            ],
        )?;
        assert!(detector.has_nonfinite(&[(&finite, 4), (&nonfinite, 3)])?);
        Ok(())
    }

    #[test]
    fn vulkan_gradient_l2_reducer_is_robust_for_huge_finite_values() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let first = GpuBuffer::from_f32(&device, &[3.0, 4.0])?;
        let second = GpuBuffer::from_f32(&device, &[1.0e30, -1.0e30])?;
        let reducer = VulkanGradientL2NormReducer::new(&device, &[2, 2])?;

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        reducer.record_l2_norm(&mut commands, &[(&first, 2), (&second, 2)])?;
        commands.submit()?;
        let norm = reducer.read_l2_norm()?;
        let expected = 2.0f64.sqrt() * 1.0e30;
        let relative = (norm - expected).abs() / expected;
        assert!(
            norm.is_finite(),
            "robust finite-gradient norm became {norm}"
        );
        assert!(
            relative < 2.0e-6,
            "norm={norm} expected={expected} relative={relative}"
        );
        Ok(())
    }

    #[test]
    fn vulkan_gradient_l2_reducer_propagates_nonfinite_inputs() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let nonfinite = GpuBuffer::from_u32(
            &device,
            &[
                1.0f32.to_bits(),
                f32::INFINITY.to_bits(),
                f32::NAN.to_bits(),
            ],
        )?;
        let reducer = VulkanGradientL2NormReducer::new(&device, &[3])?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        reducer.record_l2_norm(&mut commands, &[(&nonfinite, 3)])?;
        commands.submit()?;
        assert!(!reducer.read_l2_norm()?.is_finite());
        Ok(())
    }

    #[test]
    fn vulkan_gradient_clip_coefficient_matches_pytorch_formula() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let gradient = GpuBuffer::from_f32(&device, &[3.0, 4.0])?;
        let reducer = VulkanGradientL2NormReducer::new(&device, &[2])?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        reducer.record_l2_norm_and_clip_coefficient(&mut commands, &[(&gradient, 2)], 2.0)?;
        commands.submit()?;

        let norm = reducer.read_l2_norm()?;
        let coefficient = reducer.read_clip_coefficient()?;
        let expected = (2.0f64 / (5.0 + 1.0e-6)) as f32;
        assert!(!reducer.read_clip_has_nonfinite()?);
        assert!((norm - 5.0).abs() < 1.0e-6, "device norm={norm}");
        assert!(
            (coefficient - expected).abs() <= 1.0e-7,
            "device coefficient={coefficient} expected={expected}"
        );
        Ok(())
    }

    #[test]
    fn vulkan_gradient_clip_reduction_owns_nonfinite_safety_flag() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let nonfinite = GpuBuffer::from_u32(
            &device,
            &[
                1.0f32.to_bits(),
                f32::INFINITY.to_bits(),
                f32::NAN.to_bits(),
            ],
        )?;
        let reducer = VulkanGradientL2NormReducer::new(&device, &[3])?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        reducer.record_l2_norm_and_clip_coefficient(&mut commands, &[(&nonfinite, 3)], 1.0)?;
        commands.submit()?;

        assert!(reducer.read_clip_has_nonfinite()?);
        assert!(!reducer.read_l2_norm()?.is_finite());
        assert_eq!(reducer.read_clip_coefficient()?, 0.0);
        Ok(())
    }

    #[test]
    fn vulkan_dynamic_loss_scale_can_advance_twice_without_intermediate_readback() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let finite_flag = GpuBuffer::zeros_u32(&device, 1)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let stale_seed = VulkanDynamicLossScaleSeed {
            scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
            growth_tracker: 0,
            pending_gradients_scaled: true,
        };

        let mut first = vulkan::ComputeBatch::new(&device)?;
        scaler.record_resolve_device_resident(&mut first, &finite_flag, stale_seed)?;
        first.submit()?;

        // Deliberately reuse the original host seed. A host-authoritative
        // implementation would re-upload tracker=0 here and lose the first
        // transition. Device-resident mode must instead consume tracker=1 from
        // the prior Vulkan dispatch and grow the scale on this second window.
        let mut second = vulkan::ComputeBatch::new(&device)?;
        scaler.record_resolve_device_resident(&mut second, &finite_flag, stale_seed)?;
        scaler.record_readback(&mut second)?;
        second.submit()?;

        let decision = scaler.read_decision()?;
        assert!(decision.should_step);
        assert!(!decision.overflowed);
        assert_eq!(decision.scale_before, 8.0);
        assert_eq!(decision.scale_after, 16.0);
        assert_eq!(decision.growth_tracker, 0);
        Ok(())
    }

    #[test]
    fn vulkan_dynamic_loss_scale_drives_next_window_source_without_host_readback() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let finite_flag = GpuBuffer::zeros_u32(&device, 1)?;
        let source = GpuBuffer::from_f32(&device, &[0.5, -1.25])?;
        let source_readback = GpuBuffer::zeros_host_f32(&device, 2)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let seed = VulkanDynamicLossScaleSeed {
            scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1,
            growth_tracker: 0,
            pending_gradients_scaled: true,
        };

        let mut first_window = vulkan::ComputeBatch::new(&device)?;
        scaler.record_resolve_device_resident(&mut first_window, &finite_flag, seed)?;
        first_window.submit()?;
        assert!(scaler.host_scale_metadata_stale());

        // The first transition grows 8 -> 16. No scaler readback occurs before
        // this second submission: the next-window source multiplier is loaded
        // directly from control state word zero on the GPU.
        let mut next_window = vulkan::ComputeBatch::new(&device)?;
        scaler.record_scale_source_by_current_scale(&mut next_window, &source, 2)?;
        next_window.readback_f32(&source, &source_readback, 2)?;
        next_window.submit()?;
        assert_eq!(source_readback.read_f32(2)?, vec![8.0, -20.0]);
        Ok(())
    }

    #[test]
    fn vulkan_dynamic_loss_scale_observation_can_preserve_device_authority() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let finite_flag = GpuBuffer::zeros_u32(&device, 1)?;
        let source = GpuBuffer::from_f32(&device, &[0.25])?;
        let source_readback = GpuBuffer::zeros_host_f32(&device, 1)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let seed = VulkanDynamicLossScaleSeed {
            scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1,
            growth_tracker: 0,
            pending_gradients_scaled: true,
        };

        let mut transition = vulkan::ComputeBatch::new(&device)?;
        scaler.record_resolve_device_resident(&mut transition, &finite_flag, seed)?;
        scaler.record_readback(&mut transition)?;
        transition.submit()?;

        let observed = scaler.read_decision_preserving_device_authority()?;
        assert!(observed.should_step);
        assert_eq!(observed.scale_after, 16.0);
        assert!(scaler.host_scale_metadata_stale());

        let mut next = vulkan::ComputeBatch::new(&device)?;
        scaler.record_scale_source_by_current_scale(&mut next, &source, 1)?;
        next.readback_f32(&source, &source_readback, 1)?;
        next.submit()?;
        assert_eq!(source_readback.read_f32(1)?, vec![4.0]);
        Ok(())
    }

    #[test]
    fn vulkan_dynamic_loss_scale_scales_only_strided_backward_source_field() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let finite_flag = GpuBuffer::zeros_u32(&device, 1)?;
        let records = GpuBuffer::from_f32(
            &device,
            &[
                1.0, 2.0, 3.0, 4.0, 0.5, // row 0
                5.0, 6.0, 7.0, 8.0, -1.25, // row 1
            ],
        )?;
        let readback = GpuBuffer::zeros_host_f32(&device, 10)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let seed = VulkanDynamicLossScaleSeed {
            scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1,
            growth_tracker: 0,
            pending_gradients_scaled: true,
        };

        let mut first_window = vulkan::ComputeBatch::new(&device)?;
        scaler.record_resolve_device_resident(&mut first_window, &finite_flag, seed)?;
        first_window.submit()?;

        let mut next_window = vulkan::ComputeBatch::new(&device)?;
        scaler.record_scale_source_by_current_scale_strided(&mut next_window, &records, 2, 5, 4)?;
        next_window.readback_f32(&records, &readback, 10)?;
        next_window.submit()?;

        assert_eq!(
            readback.read_f32(10)?,
            vec![1.0, 2.0, 3.0, 4.0, 8.0, 5.0, 6.0, 7.0, 8.0, -20.0]
        );
        Ok(())
    }

    #[test]
    fn vulkan_dynamic_loss_scale_host_resolve_overrides_device_only_state() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let finite_flag = GpuBuffer::zeros_u32(&device, 1)?;
        let mut scaler = VulkanDynamicLossScaleController::new(&device)?;
        let initial = VulkanDynamicLossScaleSeed {
            scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 4,
            growth_tracker: 0,
            pending_gradients_scaled: true,
        };
        let mut resident = vulkan::ComputeBatch::new(&device)?;
        scaler.record_resolve_device_resident(&mut resident, &finite_flag, initial)?;
        resident.submit()?;

        let restored = VulkanDynamicLossScaleSeed {
            scale: 32.0,
            growth_tracker: 2,
            ..initial
        };
        let mut host_authoritative = vulkan::ComputeBatch::new(&device)?;
        scaler.record_resolve(&mut host_authoritative, &finite_flag, restored)?;
        scaler.record_readback(&mut host_authoritative)?;
        host_authoritative.submit()?;

        let decision = scaler.read_decision()?;
        assert_eq!(decision.scale_before, 32.0);
        assert_eq!(decision.scale_after, 32.0);
        assert_eq!(decision.growth_tracker, 3);
        Ok(())
    }

    fn ltm_test_seed() -> VulkanLtmAlignmentSeed {
        VulkanLtmAlignmentSeed {
            updates: 0,
            last: None,
            ema: None,
            best: None,
            writer_norm: None,
            ready: false,
            last_step_sampled_rows: 0,
            last_step_controller_sampled_rows: 0,
            sampled_rows: 2,
            controller_sampled_rows: 2,
            min_updates: 2,
            ready_threshold: 0.95,
            ema_decay: 0.5,
            writer_max_norm: 64.0,
            exact_pytorch_tbptt: false,
        }
    }

    #[test]
    fn vulkan_ltm_alignment_controller_advances_twice_without_intermediate_readback() -> Result<()>
    {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let writer = GpuBuffer::from_f32(&device, &[3.0, 4.0])?;
        let reducer = VulkanGradientL2NormReducer::new(&device, &[2])?;
        let scaler_control = GpuBuffer::from_u32(&device, &[0, 0, 0, 1, 0, 0, 0, 0])?;
        let mut controller = VulkanLtmAlignmentController::new(&device)?;
        let seed = ltm_test_seed();

        let first_costs = GpuBuffer::from_f32(&device, &[1.0, 1.0])?;
        let mut first = vulkan::ComputeBatch::new(&device)?;
        let first_pair = reducer.record_l2_pair_device_only(&mut first, &[(&writer, 2)])?;
        controller.record_step_device_resident(
            &mut first,
            &first_costs,
            2,
            &first_pair,
            &scaler_control,
            seed,
        )?;
        first.submit()?;
        assert!(controller.host_metadata_stale());

        // Reuse the deliberately stale host seed. The persistent Vulkan state
        // must carry update=1 / ema=1.0 into this second transition.
        let second_costs = GpuBuffer::from_f32(&device, &[0.8, 0.8])?;
        let mut second = vulkan::ComputeBatch::new(&device)?;
        let second_pair = reducer.record_l2_pair_device_only(&mut second, &[(&writer, 2)])?;
        controller.record_step_device_resident(
            &mut second,
            &second_costs,
            2,
            &second_pair,
            &scaler_control,
            seed,
        )?;
        controller.record_readback(&mut second)?;
        second.submit()?;

        let snapshot = controller.read_snapshot()?;
        assert_eq!(snapshot.updates, 2);
        assert_eq!(snapshot.last, Some(0.8));
        assert_eq!(snapshot.best, Some(0.8));
        assert_eq!(snapshot.writer_norm, Some(5.0));
        assert_eq!(snapshot.last_step_sampled_rows, 2);
        assert_eq!(snapshot.last_step_controller_sampled_rows, 2);
        assert!(snapshot.ema.is_some_and(|ema| (ema - 0.9).abs() <= 1.0e-6));
        assert!(snapshot.ready);
        Ok(())
    }

    #[test]
    fn vulkan_ltm_alignment_controller_committed_step_needs_no_host_predicate() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let writer = GpuBuffer::from_f32(&device, &[3.0, 4.0])?;
        let reducer = VulkanGradientL2NormReducer::new(&device, &[2])?;
        let costs = GpuBuffer::from_f32(&device, &[0.5, 1.0])?;
        let mut controller = VulkanLtmAlignmentController::new(&device)?;
        let seed = VulkanLtmAlignmentSeed {
            min_updates: 1,
            ready_threshold: 1.0,
            ..ltm_test_seed()
        };

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        let writer_pair = reducer.record_l2_pair_device_only(&mut commands, &[(&writer, 2)])?;
        controller.record_step_device_resident_committed(
            &mut commands,
            &costs,
            2,
            &writer_pair,
            seed,
        )?;
        controller.record_readback(&mut commands)?;
        commands.submit()?;

        let snapshot = controller.read_snapshot()?;
        assert_eq!(snapshot.updates, 1);
        assert_eq!(snapshot.last, Some(0.75));
        assert_eq!(snapshot.ema, Some(0.75));
        assert_eq!(snapshot.best, Some(0.75));
        assert_eq!(snapshot.writer_norm, Some(5.0));
        assert!(snapshot.ready);
        Ok(())
    }

    #[test]
    fn vulkan_ltm_alignment_controller_obeys_grad_scaler_skip_without_host_decision() -> Result<()>
    {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let writer_pair = GpuBuffer::from_f32(&device, &[5.0, 1.0])?;
        let costs = GpuBuffer::from_f32(&device, &[0.1, 0.1])?;
        let skipped_scaler_control = GpuBuffer::from_u32(&device, &[0, 0, 0, 0, 1, 0, 0, 0])?;
        let mut controller = VulkanLtmAlignmentController::new(&device)?;
        let seed = VulkanLtmAlignmentSeed {
            updates: 7,
            last: Some(0.25),
            ema: Some(0.3),
            best: Some(0.2),
            writer_norm: Some(4.0),
            ready: true,
            last_step_sampled_rows: 9,
            last_step_controller_sampled_rows: 3,
            ..ltm_test_seed()
        };

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        controller.record_step_device_resident(
            &mut commands,
            &costs,
            2,
            &writer_pair,
            &skipped_scaler_control,
            seed,
        )?;
        controller.record_readback(&mut commands)?;
        commands.submit()?;
        let snapshot = controller.read_snapshot()?;

        assert_eq!(snapshot.updates, 7);
        assert_eq!(snapshot.last, Some(0.25));
        assert_eq!(snapshot.ema, Some(0.3));
        assert_eq!(snapshot.best, Some(0.2));
        assert_eq!(snapshot.writer_norm, Some(4.0));
        assert_eq!(snapshot.last_step_sampled_rows, 9);
        assert_eq!(snapshot.last_step_controller_sampled_rows, 3);
        assert!(snapshot.ready);
        Ok(())
    }
}
