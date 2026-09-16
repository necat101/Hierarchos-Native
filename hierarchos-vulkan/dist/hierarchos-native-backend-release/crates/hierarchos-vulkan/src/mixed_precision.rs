use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

const FP32_TO_FP16_PACKED_SPV: &[u8] = include_bytes!("../shaders/fp32_to_fp16_packed.spv");
const FP16_PACKED_TO_FP32_SPV: &[u8] = include_bytes!("../shaders/fp16_packed_to_fp32.spv");
const FP32_TO_BF16_PACKED_SPV: &[u8] = include_bytes!("../shaders/fp32_to_bf16_packed.spv");
const BF16_PACKED_TO_FP32_SPV: &[u8] = include_bytes!("../shaders/bf16_packed_to_fp32.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

/// Compact device-storage format for a compute-parameter mirror.
///
/// The canonical trainable/checkpoint tensor remains FP32. These formats are
/// execution-only mirrors, so PyTorch/CUDA and native Rust continue to exchange
/// the same SafeTensors parameter values and AdamW master state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VulkanParameterStorageFormat {
    Fp16,
    Bf16,
}

impl VulkanParameterStorageFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16-storage-fp32-master",
            Self::Bf16 => "bf16-storage-fp32-master",
        }
    }
}

/// Shareable compact execution-storage identity for one FP32 master parameter.
///
/// The buffer is intentionally independent from the kernel object that refreshes
/// it. Hierarchos has both local recurrent AdamW registries and a canonical
/// full-model registry; both can therefore target the same compact storage
/// allocation without duplicating the execution-side parameter or changing the
/// FP32 checkpoint identity.
#[derive(Clone)]
pub(crate) struct VulkanParameterStorageMirror {
    format: VulkanParameterStorageFormat,
    len: usize,
    packed_words: usize,
    packed_storage: GpuBuffer,
}

impl VulkanParameterStorageMirror {
    pub(crate) fn new(
        device: &VulkanDevice,
        format: VulkanParameterStorageFormat,
        len: usize,
    ) -> Result<Self> {
        if len == 0 {
            bail!("mixed-precision parameter mirror requires at least one element");
        }
        if len > u32::MAX as usize {
            bail!("mixed-precision parameter mirror length {len} exceeds u32 shader indexing");
        }
        let packed_words = len.div_ceil(2);
        Ok(Self {
            format,
            len,
            packed_words,
            packed_storage: GpuBuffer::zeros_u32(device, packed_words)?,
        })
    }

    pub(crate) fn format(&self) -> VulkanParameterStorageFormat {
        self.format
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn packed_words(&self) -> usize {
        self.packed_words
    }

    pub(crate) fn packed_storage(&self) -> &GpuBuffer {
        &self.packed_storage
    }

    pub(crate) fn read_expanded_f32(&self, device: &VulkanDevice) -> Result<Vec<f32>> {
        let unpack_spv = match self.format {
            VulkanParameterStorageFormat::Fp16 => FP16_PACKED_TO_FP32_SPV,
            VulkanParameterStorageFormat::Bf16 => BF16_PACKED_TO_FP32_SPV,
        };
        let unpack = vulkan::ComputeKernel::new_with_access(
            device,
            unpack_spv,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<LenPush>() as u32,
        )?;
        // Both buffers are fully overwritten by this submission. Keep them in
        // the device timeline arena instead of creating/destroying VkBuffers on
        // every parity/readback probe.
        let expanded = GpuBuffer::transient_f32(device, self.len)?;
        let readback = GpuBuffer::transient_host_f32(device, self.len)?;
        let mut commands = vulkan::ComputeBatch::new(device)?;
        unpack.record_dispatch(
            &mut commands,
            &[self.packed_storage(), &expanded],
            bytemuck::bytes_of(&LenPush {
                len: self.len as u32,
            }),
            [div_ceil_u32(self.packed_words, 256), 1, 1],
        )?;
        commands.readback_f32(&expanded, &readback, self.len)?;
        commands.submit()?;
        readback.read_f32(self.len)
    }
}

/// Stateless-per-parameter writer for a compact execution mirror. One writer
/// can refresh every registry slot using the same storage format.
pub(crate) struct VulkanParameterStorageMirrorRefresher {
    format: VulkanParameterStorageFormat,
    pack: vulkan::ComputeKernel,
}

impl VulkanParameterStorageMirrorRefresher {
    pub(crate) fn new(device: &VulkanDevice, format: VulkanParameterStorageFormat) -> Result<Self> {
        let pack_spv = match format {
            VulkanParameterStorageFormat::Fp16 => FP32_TO_FP16_PACKED_SPV,
            VulkanParameterStorageFormat::Bf16 => FP32_TO_BF16_PACKED_SPV,
        };
        Ok(Self {
            format,
            pack: vulkan::ComputeKernel::new_with_access(
                device,
                pack_spv,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
        })
    }

    pub(crate) fn record_refresh(
        &self,
        commands: &mut vulkan::ComputeBatch,
        fp32_master: &GpuBuffer,
        mirror: &VulkanParameterStorageMirror,
    ) -> Result<()> {
        if mirror.format() != self.format {
            bail!(
                "mixed-precision refresher format {} cannot update {} mirror",
                self.format.label(),
                mirror.format().label()
            );
        }
        if fp32_master.f32_capacity() < mirror.len() {
            bail!(
                "FP32 master capacity {} is smaller than mixed-precision mirror length {}",
                fp32_master.f32_capacity(),
                mirror.len()
            );
        }
        self.pack.record_dispatch(
            commands,
            &[fp32_master, mirror.packed_storage()],
            bytemuck::bytes_of(&LenPush {
                len: mirror.len() as u32,
            }),
            [div_ceil_u32(mirror.packed_words(), 256), 1, 1],
        )
    }
}

/// A 16-bit execution-storage mirror sourced from a canonical FP32 parameter.
///
/// Two 16-bit elements are packed into each `uint` storage word. That gives us
/// a portable Vulkan 1.1-compatible storage boundary before every consumer
/// shader has native FP16/BF16 buffer declarations. `expanded_f32` is a
/// transition buffer for today's FP32 kernels; native half consumers can later
/// bind `packed_storage` directly without changing the FP32 optimizer master.
pub struct VulkanFp32MasterParameterMirror {
    device: VulkanDevice,
    storage: VulkanParameterStorageMirror,
    expanded_f32: GpuBuffer,
    refresher: VulkanParameterStorageMirrorRefresher,
    unpack: vulkan::ComputeKernel,
}

impl VulkanFp32MasterParameterMirror {
    pub fn new(
        device: VulkanDevice,
        format: VulkanParameterStorageFormat,
        len: usize,
    ) -> Result<Self> {
        let storage = VulkanParameterStorageMirror::new(&device, format, len)?;
        let unpack_spv = match format {
            VulkanParameterStorageFormat::Fp16 => FP16_PACKED_TO_FP32_SPV,
            VulkanParameterStorageFormat::Bf16 => BF16_PACKED_TO_FP32_SPV,
        };
        Ok(Self {
            device: device.clone(),
            storage,
            expanded_f32: GpuBuffer::zeros_f32(&device, len)?,
            refresher: VulkanParameterStorageMirrorRefresher::new(&device, format)?,
            unpack: vulkan::ComputeKernel::new_with_access(
                &device,
                unpack_spv,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LenPush>() as u32,
            )?,
        })
    }

    pub fn format(&self) -> VulkanParameterStorageFormat {
        self.storage.format()
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Logical bytes occupied by the 16-bit representation, excluding the
    /// two-byte alignment pad used when `len` is odd.
    pub fn logical_storage_bytes(&self) -> usize {
        self.storage.len() * 2
    }

    pub fn allocated_storage_bytes(&self) -> usize {
        self.storage.packed_words() * std::mem::size_of::<u32>()
    }

    /// Packed two-elements-per-word storage intended for future native-half
    /// consumers. Its bit layout is IEEE FP16 or BF16 according to `format`.
    pub fn packed_storage(&self) -> &GpuBuffer {
        self.storage.packed_storage()
    }

    /// FP32 transition buffer for kernels that have not yet gained a native
    /// half-storage specialization.
    pub fn expanded_f32(&self) -> &GpuBuffer {
        &self.expanded_f32
    }

    /// Refresh only the compact mirror in one Vulkan submission. The canonical
    /// FP32 master is never rebound or written by this operation.
    pub fn refresh_storage_from_fp32_master(&self, fp32_master: &GpuBuffer) -> Result<()> {
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.record_refresh_from_fp32_master(&mut commands, fp32_master)?;
        commands.submit()
    }

    /// Transitional convenience path for existing FP32 compute kernels:
    /// quantize the master into 16-bit storage and expand it again in the same
    /// submission. Native-half consumers will use only the refresh half.
    pub fn refresh_and_expand_from_fp32_master(&self, fp32_master: &GpuBuffer) -> Result<()> {
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.record_refresh_from_fp32_master(&mut commands, fp32_master)?;
        self.record_expand_for_fp32_compute(&mut commands)?;
        commands.submit()
    }

    /// Quantize the current canonical FP32 parameter into compact execution
    /// storage. The master buffer is read-only by construction.
    pub(crate) fn record_refresh_from_fp32_master(
        &self,
        commands: &mut vulkan::ComputeBatch,
        fp32_master: &GpuBuffer,
    ) -> Result<()> {
        self.refresher
            .record_refresh(commands, fp32_master, &self.storage)
    }

    /// Expand the compact representation into FP32 for an existing kernel.
    /// This is deliberately separate from `record_refresh_from_fp32_master` so
    /// a future native-half kernel can skip the expansion dispatch entirely.
    pub(crate) fn record_expand_for_fp32_compute(
        &self,
        commands: &mut vulkan::ComputeBatch,
    ) -> Result<()> {
        self.unpack
            .record_dispatch(
                commands,
                &[self.storage.packed_storage(), &self.expanded_f32],
                bytemuck::bytes_of(&LenPush {
                    len: self.storage.len() as u32,
                }),
                [div_ceil_u32(self.storage.packed_words(), 256), 1, 1],
            )
            .with_context(|| format!("expanding {} parameter mirror", self.format().label()))
    }
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(format: VulkanParameterStorageFormat, tolerance: f32) -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let values = [0.0f32, 1.0, -2.0, 0.333_251_95, 17.125, -0.007_812_5, 65.5];
        let master = GpuBuffer::from_f32(&device, &values)?;
        let mirror = VulkanFp32MasterParameterMirror::new(device.clone(), format, values.len())?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        mirror.record_refresh_from_fp32_master(&mut commands, &master)?;
        mirror.record_expand_for_fp32_compute(&mut commands)?;
        commands.submit()?;

        assert_eq!(master.read_f32(values.len())?, values);
        let expanded = mirror.expanded_f32().read_f32(values.len())?;
        for (index, (&expected, &actual)) in values.iter().zip(&expanded).enumerate() {
            assert!(
                (expected - actual).abs() <= tolerance,
                "{} element {index}: expected {expected}, got {actual}",
                format.label()
            );
        }
        assert_eq!(mirror.logical_storage_bytes(), values.len() * 2);
        assert_eq!(
            mirror.allocated_storage_bytes(),
            values.len().div_ceil(2) * 4
        );
        Ok(())
    }

    #[test]
    fn fp16_storage_roundtrip_preserves_fp32_master() -> Result<()> {
        roundtrip(VulkanParameterStorageFormat::Fp16, 2.0e-3)
    }

    #[test]
    fn fp16_storage_uses_ieee_round_to_nearest_even() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let min_subnormal = 2.0f32.powi(-24);
        let values = [
            1.000_488_281_25,
            1.001_464_843_75,
            -1.000_488_281_25,
            2.0f32.powi(-25),
            3.0 * 2.0f32.powi(-25),
            min_subnormal,
            65_504.0,
            65_520.0,
        ];
        let expected = [
            1.0,
            1.001_953_125,
            -1.0,
            0.0,
            2.0 * min_subnormal,
            min_subnormal,
            65_504.0,
            f32::INFINITY,
        ];
        let master = GpuBuffer::from_f32(&device, &values)?;
        let mirror = VulkanFp32MasterParameterMirror::new(
            device.clone(),
            VulkanParameterStorageFormat::Fp16,
            values.len(),
        )?;
        mirror.refresh_and_expand_from_fp32_master(&master)?;
        let expanded = mirror.expanded_f32().read_f32(values.len())?;
        assert_eq!(expanded, expected);
        assert_eq!(master.read_f32(values.len())?, values);
        Ok(())
    }

    #[test]
    fn bf16_storage_roundtrip_preserves_fp32_master() -> Result<()> {
        roundtrip(VulkanParameterStorageFormat::Bf16, 2.0e-2)
    }

    #[test]
    fn mirror_readback_reuses_timeline_arena_scratch() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        let values = [1.0f32, -2.0, 0.5, 7.25, -0.125, 3.0, 0.0, 16.0];
        let master = GpuBuffer::from_f32(&device, &values)?;
        let storage = VulkanParameterStorageMirror::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
            values.len(),
        )?;
        let refresher = VulkanParameterStorageMirrorRefresher::new(
            &device,
            VulkanParameterStorageFormat::Fp16,
        )?;
        let mut refresh = vulkan::ComputeBatch::new(&device)?;
        refresher.record_refresh(&mut refresh, &master, &storage)?;
        refresh.submit()?;

        let first = storage.read_expanded_f32(&device)?;
        let after_first = device.submission_arena_stats()?;
        assert!(after_first.scratch_slab_count >= 2);
        assert!(after_first.scratch_lease_allocated >= 2);
        let second = storage.read_expanded_f32(&device)?;
        let after_second = device.submission_arena_stats()?;

        assert_eq!(first, second);
        assert!(
            after_second.scratch_lease_reused >= after_first.scratch_lease_reused.saturating_add(2),
            "mirror readback did not reuse both device-local and host-visible slab leases"
        );
        Ok(())
    }
}
