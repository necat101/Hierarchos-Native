//! Opt-in RDNA3 hardware cooperative-matrix (WMMA) GEMM support.
//!
//! The default Hierarchos Vulkan path is unchanged FP32. This module adds an
//! orthogonal, explicitly requested acceleration policy:
//!
//! * [`WmmaMode::Off`] — every existing kernel runs exactly as before. The
//!   device is not asked for any cooperative-matrix feature and no WMMA shader
//!   is created.
//! * [`WmmaMode::Mixed`] — FP16 matrix operands with FP32 accumulation. FP32
//!   master parameters, FP32 optimizer state, and FP32 normalization, softmax
//!   and loss reductions stay where they already are; only the GEMM operands are
//!   narrowed (`mock-AMP`).
//! * [`WmmaMode::FullFp16`] — FP16 operands and FP16 accumulation where the
//!   driver genuinely advertises an `f16 x f16 -> f16` cooperative configuration.
//!   FP32 masters and optimizer state are still retained. This arm is
//!   experimental: FP16 accumulation can overflow to infinity and loses mantissa bits.
//!
//! Nothing here is inferred from the adapter's vendor or name. The device must
//! expose `VK_KHR_cooperative_matrix`, the `cooperativeMatrix` feature, the
//! Vulkan memory model, 16-bit storage/arithmetic, and a cooperative
//! configuration whose scope is `SUBGROUP` with 16x16x16 tiles; the WMMA
//! pipeline is then created with a forced 32-wide subgroup. A device that does
//! not meet all of that produces a diagnostic instead of running a shader that
//! only looks like FP16.
//!
//! Kernels that use ordinary FP16 arithmetic without cooperative matrices are
//! [`WmmaGemmArm::PlainF16Storage`]. They are never reported as hardware WMMA,
//! and they exist so an ablation can separate "FP16 GEMM" from "matrix-core
//! GEMM" instead of attributing one to the other.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use ash::vk;

use crate::vulkan;
use crate::{GpuBuffer, VulkanDevice};

const WMMA_GEMM_F32ACC_SPV: &[u8] = include_bytes!("../shaders/wmma_gemm_f16_f32acc.spv");
const WMMA_GEMM_F16ACC_SPV: &[u8] = include_bytes!("../shaders/wmma_gemm_f16_f16acc.spv");
const GEMM_F16_F32_PLAIN_SPV: &[u8] = include_bytes!("../shaders/gemm_f16_f32_plain.spv");
const FP32_TO_FP16_NATIVE_SPV: &[u8] = include_bytes!("../shaders/fp32_to_fp16_native.spv");
/// The product's existing FP32 linear kernel, used as the paired baseline arm.
/// Its contract is `out[row, col] = sum_t a[row, t] * b[col, t]`, i.e. operand B
/// stored `[columns, reduction]` with an increasing-K FP32 FMA chain.
const FP32_LINEAR_FORWARD_SPV: &[u8] = include_bytes!("../shaders/linear_forward.spv");

/// Cooperative-matrix tile edge used by every WMMA shader in this repository.
pub const WMMA_TILE_M: u32 = 16;
pub const WMMA_TILE_N: u32 = 16;
pub const WMMA_TILE_K: u32 = 16;

/// RDNA3 matrix cores execute 16x16x16 cooperative operations on a wave32.
/// The device reports a default subgroup size of 64, so the WMMA pipelines
/// force 32 through `VK_EXT_subgroup_size_control` rather than assuming it.
pub const WMMA_SUBGROUP_SIZE: u32 = 32;

/// Environment variable that selects the policy for binaries that do not take
/// an explicit flag. Unset/empty means [`WmmaMode::Off`].
pub const WMMA_MODE_ENV: &str = "HIERARCHOS_WMMA_MODE";

/// Qualification targets for the small deterministic fixtures.
///
/// These are initial engineering gates, not universal tolerances: larger models,
/// longer contexts and different logit scales need their own scaled relative
/// checks. See `RDNA3_WMMA.md`.
pub const MIXED_VS_FP32_MAX_ABS: f64 = 2.0e-2;
pub const MIXED_VS_FP32_RMS: f64 = 3.0e-3;
pub const MIXED_VS_MATCHED_ORACLE_MAX_ABS: f64 = 1.0e-3;
pub const FULL_VS_FP32_MAX_ABS: f64 = 5.0e-2;
pub const FULL_VS_FP32_RMS: f64 = 1.0e-2;
pub const FULL_VS_MATCHED_ORACLE_MAX_ABS: f64 = 4.0e-3;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WmmaMode {
    /// Unchanged behaviour: no cooperative-matrix feature is requested.
    #[default]
    Off,
    /// FP16 operands, FP32 accumulation, FP32 master parameters.
    Mixed,
    /// FP16 operands and FP16 accumulation where genuinely advertised.
    FullFp16,
}

impl WmmaMode {
    pub const ALL: [Self; 3] = [Self::Off, Self::Mixed, Self::FullFp16];

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "fp32" => Ok(Self::Off),
            "mixed" | "mock-amp" | "amp" => Ok(Self::Mixed),
            "full-fp16" | "full_fp16" | "fullfp16" | "fp16" => Ok(Self::FullFp16),
            other => bail!(
                "unsupported --wmma-mode {other:?}; expected off, mixed, or full-fp16"
            ),
        }
    }

    /// Read [`WMMA_MODE_ENV`]. An absent or empty value is `off`, so a process
    /// that never sets it keeps the pre-existing FP32 device contract.
    pub fn from_env() -> Result<Self> {
        match std::env::var(WMMA_MODE_ENV) {
            Ok(value) if !value.trim().is_empty() => Self::parse(&value),
            _ => Ok(Self::Off),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Mixed => "mixed",
            Self::FullFp16 => "full-fp16",
        }
    }

    pub fn requested(self) -> bool {
        self != Self::Off
    }

    /// Human-readable contract, used by CLI diagnostics and receipts.
    pub fn description(self) -> &'static str {
        match self {
            Self::Off => {
                "FP32 GEMM, FP32 parameters, FP32 optimizer state; no cooperative-matrix feature requested"
            }
            Self::Mixed => {
                "FP16 GEMM operands with FP32 accumulation; FP32 masters, FP32 optimizer state, FP32 norm/softmax/loss"
            }
            Self::FullFp16 => {
                "FP16 GEMM operands and FP16 accumulation (experimental); FP32 masters and FP32 optimizer state retained"
            }
        }
    }
}

/// One `VkCooperativeMatrixPropertiesKHR` row, reduced to the fields this
/// repository selects on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WmmaTile {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_is_f16: bool,
    pub b_is_f16: bool,
    pub c_is_f16: bool,
    pub c_is_f32: bool,
    pub subgroup_scope: bool,
    pub saturating: bool,
}

impl WmmaTile {
    pub fn is_f16_f16_f32(self) -> bool {
        self.a_is_f16 && self.b_is_f16 && self.c_is_f32 && self.subgroup_scope
    }

    pub fn is_f16_f16_f16(self) -> bool {
        self.a_is_f16 && self.b_is_f16 && self.c_is_f16 && self.subgroup_scope
    }

    pub fn matches_repo_tile(self) -> bool {
        self.m == WMMA_TILE_M && self.n == WMMA_TILE_N && self.k == WMMA_TILE_K
    }
}

impl std::fmt::Display for WmmaTile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = |high: bool, low: bool| match (high, low) {
            (true, _) => "f16",
            (false, true) => "f32",
            (false, false) => "other",
        };
        write!(
            f,
            "{}x{}x{} A={} B={} C={}{}{}",
            self.m,
            self.n,
            self.k,
            ty(self.a_is_f16, false),
            ty(self.b_is_f16, false),
            ty(self.c_is_f16, self.c_is_f32),
            if self.subgroup_scope { " scope=subgroup" } else { " scope!=subgroup" },
            if self.saturating { " saturating" } else { "" },
        )
    }
}

/// What this logical device can actually do. Reported by `wmma-probe`, the CLI
/// `doctor`/`devices` diagnostics, and every qualification receipt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanWmmaCapabilities {
    /// `VK_KHR_cooperative_matrix` appears in the device extension list.
    pub extension_exposed: bool,
    /// `VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix`.
    pub feature_supported: bool,
    /// `vulkanMemoryModel`, required by the SPIR-V these shaders compile to.
    pub memory_model_supported: bool,
    /// `VK_EXT_subgroup_size_control` accepts 32 for compute.
    pub subgroup_size_control_supported: bool,
    /// 16-bit storage-buffer access and FP16 arithmetic on the logical device.
    pub storage_16bit: bool,
    pub shader_float16: bool,
    /// The default subgroup size the driver reports for this device.
    pub subgroup_size: u32,
    /// Cooperative-matrix feature was requested and enabled for this device.
    pub enabled: bool,
    /// Selected 16x16x16 subgroup-scope f16/f16 -> f32 configuration.
    pub mixed_tile: Option<WmmaTile>,
    /// Selected 16x16x16 subgroup-scope f16/f16 -> f16 configuration.
    pub full_fp16_tile: Option<WmmaTile>,
    /// Number of cooperative-matrix configurations the driver advertised.
    pub configuration_count: u32,
}

impl VulkanWmmaCapabilities {
    /// Everything the mixed arm needs, including an enabled feature.
    pub fn mixed_supported(&self) -> bool {
        self.enabled
            && self.storage_16bit
            && self.shader_float16
            && self.subgroup_size_control_supported
            && self
                .mixed_tile
                .is_some_and(|tile| tile.matches_repo_tile() && tile.is_f16_f16_f32())
    }

    /// Everything the full-FP16 arm needs. Deliberately separate from
    /// [`Self::mixed_supported`]: an adapter may expose only FP32 accumulation.
    pub fn full_fp16_supported(&self) -> bool {
        self.enabled
            && self.storage_16bit
            && self.shader_float16
            && self.subgroup_size_control_supported
            && self
                .full_fp16_tile
                .is_some_and(|tile| tile.matches_repo_tile() && tile.is_f16_f16_f16())
    }

    pub fn supports_mode(&self, mode: WmmaMode) -> bool {
        match mode {
            WmmaMode::Off => true,
            WmmaMode::Mixed => self.mixed_supported(),
            WmmaMode::FullFp16 => self.full_fp16_supported(),
        }
    }

    /// Multi-line, driver-truth diagnostic. Never claims WMMA when the feature
    /// was not enabled.
    pub fn diagnostics(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "cooperative matrix: extension_exposed={} feature_supported={} memory_model={} subgroup_size_control={} enabled={}\n",
            self.extension_exposed,
            self.feature_supported,
            self.memory_model_supported,
            self.subgroup_size_control_supported,
            self.enabled
        ));
        out.push_str(&format!(
            "  storage_16bit={} shader_float16={} driver_subgroup_size={} configurations={}\n",
            self.storage_16bit, self.shader_float16, self.subgroup_size, self.configuration_count
        ));
        out.push_str(&format!(
            "  mixed (f16xf16+fp32) tile: {}\n",
            match self.mixed_tile {
                Some(tile) => tile.to_string(),
                None => "unavailable".to_owned(),
            }
        ));
        out.push_str(&format!(
            "  full-fp16 (f16xf16+f16) tile: {}\n",
            match self.full_fp16_tile {
                Some(tile) => tile.to_string(),
                None => "unavailable".to_owned(),
            }
        ));
        if !self.mixed_supported() {
            out.push_str(&format!(
                "  hardware WMMA mixed: UNAVAILABLE — {}; use --wmma-mode off\n",
                self.unavailable_reason(WmmaMode::Mixed)
            ));
        }
        if !self.full_fp16_supported() {
            out.push_str(&format!(
                "  hardware WMMA full-fp16: UNAVAILABLE — {}; use --wmma-mode off\n",
                self.unavailable_reason(WmmaMode::FullFp16)
            ));
        }
        out
    }

    /// Why a mode is unusable, in the order the device contract is checked.
    pub fn unavailable_reason(&self, mode: WmmaMode) -> &'static str {
        if mode == WmmaMode::Off {
            return "mode off never needs cooperative matrices";
        }
        if !self.extension_exposed {
            return "device does not expose VK_KHR_cooperative_matrix";
        }
        if !self.feature_supported {
            return "VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix is false";
        }
        if !self.memory_model_supported {
            return "vulkanMemoryModel is unsupported, but the SPIR-V requires it";
        }
        if !self.storage_16bit {
            return "storageBuffer16BitAccess is unavailable";
        }
        if !self.shader_float16 {
            return "shaderFloat16 is unavailable";
        }
        if !self.enabled {
            return "cooperative-matrix feature was not enabled on this logical device";
        }
        if !self.subgroup_size_control_supported {
            return "VK_EXT_subgroup_size_control cannot force a 32-wide compute subgroup";
        }
        match mode {
            WmmaMode::Mixed => {
                "no SUBGROUP-scope 16x16x16 f16/f16 -> f32 cooperative configuration"
            }
            WmmaMode::FullFp16 => {
                "no SUBGROUP-scope 16x16x16 f16/f16 -> f16 cooperative configuration"
            }
            WmmaMode::Off => unreachable!(),
        }
    }

    /// Fail with the diagnostic unless `mode` is genuinely usable. Used by
    /// explicit requests so an unsupported adapter reports instead of silently
    /// running the fallback.
    pub fn require(&self, mode: WmmaMode) -> Result<()> {
        if self.supports_mode(mode) {
            return Ok(());
        }
        bail!(
            "--wmma-mode {} is not supported by this Vulkan device: {}\n{}",
            mode.label(),
            self.unavailable_reason(mode),
            self.diagnostics()
        )
    }
}

/// Query every `VK_KHR_cooperative_matrix` configuration for a physical device.
///
/// Property enumeration does not require the feature to be enabled, so a probe
/// process can report an unsupported device truthfully.
pub(crate) fn probe_cooperative_matrix_tiles(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<Vec<WmmaTile>> {
    let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let properties = unsafe {
        loader.get_physical_device_cooperative_matrix_properties(physical_device)
    }
    .map_err(|error| anyhow::anyhow!("querying cooperative matrix properties: {error:?}"))?;
    Ok(properties
        .iter()
        .map(|property| WmmaTile {
            m: property.m_size,
            n: property.n_size,
            k: property.k_size,
            a_is_f16: property.a_type == vk::ComponentTypeKHR::FLOAT16,
            b_is_f16: property.b_type == vk::ComponentTypeKHR::FLOAT16,
            c_is_f16: property.c_type == vk::ComponentTypeKHR::FLOAT16,
            c_is_f32: property.c_type == vk::ComponentTypeKHR::FLOAT32,
            subgroup_scope: property.scope == vk::ScopeKHR::SUBGROUP,
            saturating: property.saturating_accumulation != vk::FALSE,
        })
        .collect())
}

/// Select the repository tile from the advertised rows.
///
/// A saturating row is only accepted when no ordinary row of the same shape and
/// component types exists, and that choice is reported through
/// [`WmmaTile::saturating`] so a receipt never hides it.
pub(crate) fn select_tiles(tiles: &[WmmaTile]) -> (Option<WmmaTile>, Option<WmmaTile>) {
    let pick = |predicate: fn(WmmaTile) -> bool| -> Option<WmmaTile> {
        let mut candidates = tiles
            .iter()
            .copied()
            .filter(|tile| predicate(*tile) && tile.matches_repo_tile());
        let first = candidates.next()?;
        let mut chosen = first;
        if chosen.saturating {
            // Prefer a non-saturating row of the same shape when one exists.
            for candidate in tiles.iter().copied() {
                if predicate(candidate) && candidate.matches_repo_tile() && !candidate.saturating {
                    chosen = candidate;
                    break;
                }
            }
        }
        Some(chosen)
    };
    (
        pick(WmmaTile::is_f16_f16_f32),
        pick(WmmaTile::is_f16_f16_f16),
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GemmPush {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    flags: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

/// Which GEMM kernel actually ran. Reported in receipts and benchmarks so an
/// ablation cannot be mistaken for a hardware-WMMA measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WmmaGemmArm {
    /// Cooperative matrix, FP16 operands, FP32 accumulation.
    MixedFp32Accumulate,
    /// Cooperative matrix, FP16 operands, FP16 accumulation.
    FullFp16Accumulate,
    /// Ordinary FP16-storage GEMM without cooperative matrices.
    PlainF16Storage,
    /// The repository's existing FP32 linear kernel. Paired baseline only; it
    /// requires B stored `[columns, reduction]` (`transposed_b`).
    Fp32Baseline,
}

impl WmmaGemmArm {
    pub fn label(self) -> &'static str {
        match self {
            Self::MixedFp32Accumulate => "wmma-mixed-f32acc",
            Self::FullFp16Accumulate => "wmma-full-f16acc",
            Self::PlainF16Storage => "plain-f16-storage-f32acc",
            Self::Fp32Baseline => "fp32-baseline",
        }
    }

    pub fn uses_cooperative_matrix(self) -> bool {
        matches!(
            self,
            Self::MixedFp32Accumulate | Self::FullFp16Accumulate
        )
    }
}

/// Shape admission for the cooperative-matrix kernel.
///
/// Below these sizes a 32x32 output tile cannot be filled and the staging cost
/// dominates, so the caller keeps its existing kernel. The thresholds are a
/// dispatch policy, not a numerical contract.
pub fn wmma_shape_eligible(rows: usize, columns: usize, reduction: usize) -> bool {
    rows >= 16
        && columns >= 16
        && reduction >= 16
        && rows.saturating_mul(columns) >= 1024
        && rows.div_ceil(32).saturating_mul(columns.div_ceil(32)) >= 4
}

/// Opt-in GEMM encodings used by callers that already hold FP16 storage.
#[derive(Clone, Copy, Debug, Default)]
pub struct GemmLayout {
    pub transposed_a: bool,
    pub transposed_b: bool,
}

struct WmmaKernels {
    convert: vulkan::ComputeKernel,
    mixed: Option<vulkan::ComputeKernel>,
    full: Option<vulkan::ComputeKernel>,
    plain: vulkan::ComputeKernel,
    fp32: vulkan::ComputeKernel,
}

/// Device-side WMMA GEMM executor for one precision policy.
///
/// Instances are cheap to share: dispatch counters use atomics and the shader
/// kernels are the device's interned pipelines.
pub struct WmmaGemm {
    device: VulkanDevice,
    mode: WmmaMode,
    capabilities: VulkanWmmaCapabilities,
    kernels: WmmaKernels,
    cooperative_dispatches: AtomicU64,
    plain_dispatches: AtomicU64,
    shape_fallbacks: AtomicU64,
}

impl WmmaGemm {
    /// Build the executor for `mode`.
    ///
    /// `Off` still succeeds and still provides the plain FP16 control arm used
    /// by ablation benchmarks; it simply never creates a cooperative-matrix
    /// kernel. `Mixed`/`FullFp16` fail with the full diagnostic when the device
    /// does not satisfy [`VulkanWmmaCapabilities::require`].
    pub fn new(device: &VulkanDevice, mode: WmmaMode) -> Result<Self> {
        let capabilities = device.wmma_capabilities();
        capabilities.require(mode)?;
        let convert = vulkan::ComputeKernel::new_with_access(
            device,
            FP32_TO_FP16_NATIVE_SPV,
            &[vulkan::BindingAccess::ReadOnly, vulkan::BindingAccess::MayWrite],
            std::mem::size_of::<LenPush>() as u32,
        )
        .context("creating FP32->FP16 native conversion kernel")?;
        let plain = vulkan::ComputeKernel::new_with_access_and_subgroup(
            device,
            GEMM_F16_F32_PLAIN_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            std::mem::size_of::<GemmPush>() as u32,
            None,
        )
        .context("creating plain FP16-storage GEMM kernel")?;
        let fp32 = vulkan::ComputeKernel::new_with_access(
            device,
            FP32_LINEAR_FORWARD_SPV,
            &[
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::ReadOnly,
                vulkan::BindingAccess::MayWrite,
            ],
            3 * std::mem::size_of::<u32>() as u32,
        )
        .context("creating FP32 baseline linear kernel")?;
        let gemm_kernel = |spirv: &[u8], label: &str| -> Result<vulkan::ComputeKernel> {
            vulkan::ComputeKernel::new_with_access_and_subgroup(
                device,
                spirv,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<GemmPush>() as u32,
                Some(WMMA_SUBGROUP_SIZE),
            )
            .with_context(|| format!("creating {label} cooperative-matrix kernel"))
        };
        let mixed = if capabilities.mixed_supported() {
            Some(gemm_kernel(WMMA_GEMM_F32ACC_SPV, "mixed")?)
        } else {
            None
        };
        let full = if capabilities.full_fp16_supported() {
            Some(gemm_kernel(WMMA_GEMM_F16ACC_SPV, "full-fp16")?)
        } else {
            None
        };
        Ok(Self {
            device: device.clone(),
            mode,
            capabilities,
            kernels: WmmaKernels {
                convert,
                mixed,
                full,
                plain,
                fp32,
            },
            cooperative_dispatches: AtomicU64::new(0),
            plain_dispatches: AtomicU64::new(0),
            shape_fallbacks: AtomicU64::new(0),
        })
    }

    pub fn mode(&self) -> WmmaMode {
        self.mode
    }

    pub fn capabilities(&self) -> VulkanWmmaCapabilities {
        self.capabilities
    }

    /// Number of substitutions this policy applied: cooperative-matrix GEMMs
    /// plus plain-FP16 GEMMs, and calls that kept the caller's kernel.
    pub fn dispatch_counts(&self) -> WmmaDispatchCounts {
        WmmaDispatchCounts {
            cooperative: self.cooperative_dispatches.load(Ordering::Relaxed),
            plain_f16: self.plain_dispatches.load(Ordering::Relaxed),
            shape_fallbacks: self.shape_fallbacks.load(Ordering::Relaxed),
        }
    }

    /// Pick the kernel for a shape under this policy, or `None` to keep the
    /// caller's existing FP32 kernel.
    ///
    /// `Mixed` falls back to the plain FP16 GEMM (same operand precision, no
    /// matrix cores) for shapes the cooperative tiling cannot fill. `FullFp16`
    /// does the same but never silently downgrades a requested FP16
    /// accumulation to an FP32-accumulating arm: the plain fallback is reported
    /// as FP32 accumulation in the receipt.
    pub fn select_arm(
        &self,
        rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Option<WmmaGemmArm> {
        let eligible = wmma_shape_eligible(rows, columns, reduction);
        match self.mode {
            WmmaMode::Off => None,
            WmmaMode::Mixed => {
                if eligible && self.kernels.mixed.is_some() {
                    Some(WmmaGemmArm::MixedFp32Accumulate)
                } else {
                    Some(WmmaGemmArm::PlainF16Storage)
                }
            }
            WmmaMode::FullFp16 => {
                if eligible && self.kernels.full.is_some() {
                    Some(WmmaGemmArm::FullFp16Accumulate)
                } else {
                    Some(WmmaGemmArm::PlainF16Storage)
                }
            }
        }
    }

    /// Note that a caller kept its own kernel because no arm was applicable.
    pub fn note_shape_fallback(&self) {
        self.shape_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Allocate a native `float16_t` storage buffer holding `len` elements.
    ///
    /// Vulkan requires the binding range to cover `2 * len` bytes; two halves
    /// share each 32-bit word, so the allocation is `ceil(len / 2)` words.
    pub fn new_half_buffer(device: &VulkanDevice, len: usize) -> Result<GpuBuffer> {
        if len == 0 {
            bail!("native FP16 GEMM buffer requires at least one element");
        }
        GpuBuffer::zeros_u32(device, len.div_ceil(2))
    }

    /// Record an FP32 -> native FP16 conversion for `len` elements.
    pub fn record_conversion(
        &self,
        batch: &mut vulkan::ComputeBatch,
        source_f32: &GpuBuffer,
        destination_f16: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        self.kernels.convert.record_dispatch(
            batch,
            &[source_f32, destination_f16],
            bytemuck::bytes_of(&LenPush {
                len: len as u32,
            }),
            [div_ceil_u32(len, 256), 1, 1],
        )
    }

    /// Record one GEMM without any host round trip.
    ///
    /// `a`/`b` must already hold native FP16 storage. Shapes outside the tile
    /// contract are the caller's responsibility; use [`Self::select_arm`].
    #[allow(clippy::too_many_arguments)]
    pub fn record_gemm(
        &self,
        batch: &mut vulkan::ComputeBatch,
        arm: WmmaGemmArm,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        rows: usize,
        columns: usize,
        reduction: usize,
        layout: GemmLayout,
    ) -> Result<()> {
        if rows == 0 || columns == 0 || reduction == 0 {
            bail!("GEMM dimensions must be positive");
        }
        let kernel = match arm {
            WmmaGemmArm::MixedFp32Accumulate => self
                .kernels
                .mixed
                .as_ref()
                .context("mixed WMMA kernel is unavailable on this device")?,
            WmmaGemmArm::FullFp16Accumulate => self
                .kernels
                .full
                .as_ref()
                .context("full-FP16 WMMA kernel is unavailable on this device")?,
            WmmaGemmArm::PlainF16Storage => &self.kernels.plain,
            WmmaGemmArm::Fp32Baseline => &self.kernels.fp32,
        };
        if arm == WmmaGemmArm::Fp32Baseline {
            // The existing FP32 kernel consumes A `[rows, k]` and W `[columns, k]`
            // and has a three-field push constant, so it only expresses the
            // `transposed_b` layout. Refuse the rest instead of reinterpreting
            // memory under a different contract.
            if layout.transposed_a || !layout.transposed_b {
                bail!(
                    "FP32 baseline arm requires A row-major [rows, k] and B stored [columns, k]"
                );
            }
            let push = [
                rows as u32,
                reduction as u32,
                columns as u32,
            ];
            self.kernels.fp32.record_dispatch(
                batch,
                &[a, b, c],
                bytemuck::cast_slice(&push),
                [
                    div_ceil_u32(columns, 16),
                    div_ceil_u32(rows, 16),
                    1,
                ],
            )?;
            self.plain_dispatches.fetch_add(0, Ordering::Relaxed);
            return Ok(());
        }
        let lda = if layout.transposed_a {
            rows
        } else {
            reduction
        };
        let ldb = if layout.transposed_b {
            reduction
        } else {
            columns
        };
        let push = GemmPush {
            m: rows as u32,
            n: columns as u32,
            k: reduction as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: columns as u32,
            flags: (u32::from(layout.transposed_a)) | (u32::from(layout.transposed_b) << 1),
            reserved: 0,
        };
        let tile = if arm.uses_cooperative_matrix() {
            WMMA_SUBGROUP_SIZE
        } else {
            16
        };
        let groups = [
            div_ceil_u32(columns, tile),
            div_ceil_u32(rows, tile),
            1,
        ];
        kernel.record_dispatch(batch, &[a, b, c], bytemuck::bytes_of(&push), groups)?;
        if arm.uses_cooperative_matrix() {
            self.cooperative_dispatches.fetch_add(1, Ordering::Relaxed);
        } else {
            self.plain_dispatches.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Host-facing GEMM used by the qualification and benchmark harnesses.
    ///
    /// Inputs are FP32 row-major matrices; they are narrowed exactly once by the
    /// conversion kernel, mirroring the training path where FP32 masters are
    /// refreshed into FP16 mirrors.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        arm: WmmaGemmArm,
        a: &[f32],
        b: &[f32],
        rows: usize,
        columns: usize,
        reduction: usize,
        layout: GemmLayout,
    ) -> Result<Vec<f32>> {
        let expected_a = rows
            .checked_mul(reduction)
            .context("GEMM A size overflow")?;
        let expected_b = reduction
            .checked_mul(columns)
            .context("GEMM B size overflow")?;
        if a.len() != expected_a {
            bail!("GEMM A has {} values; expected {expected_a}", a.len());
        }
        if b.len() != expected_b {
            bail!("GEMM B has {} values; expected {expected_b}", b.len());
        }
        let out_len = rows.checked_mul(columns).context("GEMM C size overflow")?;
        let a_f32 = GpuBuffer::transient_f32(&self.device, expected_a)?;
        let b_f32 = GpuBuffer::transient_f32(&self.device, expected_b)?;
        let c_f32 = GpuBuffer::transient_f32(&self.device, out_len)?;
        let readback = GpuBuffer::transient_host_f32(&self.device, out_len)?;
        let mut batch = vulkan::ComputeBatch::new(&self.device)?;
        batch.upload_f32(&a_f32, a)?;
        batch.upload_f32(&b_f32, b)?;
        if arm == WmmaGemmArm::Fp32Baseline {
            self.record_gemm(
                &mut batch, arm, &a_f32, &b_f32, &c_f32, rows, columns, reduction, layout,
            )?;
        } else {
            let a_f16 = Self::new_half_buffer(&self.device, expected_a)?;
            let b_f16 = Self::new_half_buffer(&self.device, expected_b)?;
            self.record_conversion(&mut batch, &a_f32, &a_f16, expected_a)?;
            self.record_conversion(&mut batch, &b_f32, &b_f16, expected_b)?;
            self.record_gemm(
                &mut batch, arm, &a_f16, &b_f16, &c_f32, rows, columns, reduction, layout,
            )?;
        }
        batch.readback_f32(&c_f32, &readback, out_len)?;
        batch.submit()?;
        readback.read_f32(out_len)
    }

    /// Paired-benchmark helper: one allocation and one FP32 -> FP16 conversion,
    /// then `iterations` independent submissions that each dispatch exactly one
    /// GEMM and wait for it.
    ///
    /// This isolates the GEMM submission from conversion and allocation cost so
    /// a kernel-latency claim is about the kernel. Wall-clock includes the
    /// per-iteration command-buffer allocation and fence wait, which every arm
    /// pays equally; it is reported as "GEMM call latency", never as an
    /// end-to-end training-step speedup.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_repeated(
        &self,
        arm: WmmaGemmArm,
        a: &[f32],
        b: &[f32],
        rows: usize,
        columns: usize,
        reduction: usize,
        layout: GemmLayout,
        iterations: usize,
    ) -> Result<Vec<f64>> {
        if iterations == 0 {
            bail!("benchmark needs at least one iteration");
        }
        let expected_a = rows.checked_mul(reduction).context("GEMM A size overflow")?;
        let expected_b = reduction
            .checked_mul(columns)
            .context("GEMM B size overflow")?;
        if a.len() != expected_a || b.len() != expected_b {
            bail!("benchmark operand sizes do not match the requested shape");
        }
        let out_len = rows.checked_mul(columns).context("GEMM C size overflow")?;
        let a_f32 = GpuBuffer::from_f32(&self.device, a)?;
        let b_f32 = GpuBuffer::from_f32(&self.device, b)?;
        let c_f32 = GpuBuffer::zeros_f32(&self.device, out_len)?;
        let a_f16 = Self::new_half_buffer(&self.device, expected_a)?;
        let b_f16 = Self::new_half_buffer(&self.device, expected_b)?;
        if arm != WmmaGemmArm::Fp32Baseline {
            let mut warmup = vulkan::ComputeBatch::new(&self.device)?;
            self.record_conversion(&mut warmup, &a_f32, &a_f16, expected_a)?;
            self.record_conversion(&mut warmup, &b_f32, &b_f16, expected_b)?;
            warmup.submit()?;
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let mut batch = vulkan::ComputeBatch::new(&self.device)?;
            let (a_bind, b_bind) = if arm == WmmaGemmArm::Fp32Baseline {
                (&a_f32, &b_f32)
            } else {
                (&a_f16, &b_f16)
            };
            let started = std::time::Instant::now();
            self.record_gemm(
                &mut batch, arm, a_bind, b_bind, &c_f32, rows, columns, reduction, layout,
            )?;
            batch.submit()?;
            samples.push(started.elapsed().as_secs_f64());
        }
        Ok(samples)
    }
}

/// Dispatch accounting for one policy instance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WmmaDispatchCounts {
    pub cooperative: u64,
    pub plain_f16: u64,
    pub shape_fallbacks: u64,
}

fn div_ceil_u32(value: usize, divisor: u32) -> u32 {
    (value as u64).div_ceil(divisor as u64) as u32
}

// ---------------------------------------------------------------------------
// Matched reference oracles
// ---------------------------------------------------------------------------

/// IEEE binary16 round-to-nearest-even conversion of an `f32`.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exponent == 0xFF {
        // Inf / NaN: preserve NaN-ness, else infinity.
        return sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let unbiased = exponent - 127 + 15;
    if unbiased >= 0x1F {
        return sign | 0x7C00;
    }
    if unbiased <= 0 {
        // Subnormal half, or underflow to zero.
        if unbiased < -10 {
            return sign;
        }
        let mantissa_with_implicit = mantissa | 0x80_0000;
        let shift = (14 - unbiased) as u32;
        let shifted = mantissa_with_implicit >> shift;
        let remainder = mantissa_with_implicit & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut result = shifted as u16;
        if remainder > halfway || (remainder == halfway && (shifted & 1) == 1) {
            result += 1;
        }
        return sign | result;
    }
    let mut result = sign | (((unbiased as u32) << 10) as u16) | ((mantissa >> 13) as u16);
    let remainder = mantissa & 0x1FFF;
    if remainder > 0x1000 || (remainder == 0x1000 && (result & 1) == 1) {
        result = result.wrapping_add(1);
    }
    result
}

/// Exact `f32` value of a binary16 bit pattern (no rounding: halves are a
/// subset of floats).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    let out = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal: normalize mantissa into a float32.
            let mut e = -1i32;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let exponent32 = (127 - 15 + 1 + e + 1) as u32;
            sign | (exponent32 << 23) | ((m & 0x3FF) << 13)
        }
    } else if exponent == 0x1F {
        sign | 0x7F80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

/// Narrow `value` to FP16 and widen it back, i.e. `float(float16_t(value))`.
pub fn round_to_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

/// One binary16 unit-in-the-last-place at the magnitude of `value`.
///
/// FP16 accumulation has no fixed absolute error bound: the observable error
/// scales with the accumulator magnitude, so a receipt derived from these
/// fixtures must be expressed in ULPs rather than one absolute number. Every
/// value in the normal half range has a ULP of `2^(floor(log2|x|) - 10)`; below
/// the normal range the subnormal step `2^-24` applies.
pub fn f16_ulp(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude == 0.0 || !magnitude.is_finite() {
        return 2f32.powi(-24);
    }
    let exponent = magnitude.log2().floor() as i32;
    let ulp = 2f32.powi(exponent - 10);
    ulp.max(2f32.powi(-24))
}

/// Accumulation model used by a qualification receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleAccumulation {
    /// FP32 accumulator (the `mixed` contract).
    Fp32,
    /// Historical model with one rounding per 16-wide reduction tile: the
    /// intra-tile dot product is summed in FP32 and the accumulator is rounded
    /// to binary16 once per `k0` step. Retained so the original 37-gate
    /// qualification, including its failures, remains reproducible. A matrix
    /// instruction's K dimension alone does not specify its rounding cadence.
    Fp16TileK,
    /// RDNA3 dot2 model: two FP16 products plus the incoming FP16 accumulator,
    /// rounded to binary16 after each pair. AMD's RDNA3 ISA section 7.9 states
    /// that WMMA internally uses DOT instructions. Independent cadence sweeps
    /// match this model within the original two-ULP bound on every fixture;
    /// the historical tile16 model does not. Internal hardware order remains
    /// implementation dependent, so this is not a bitwise emulation claim.
    Fp16Dot2,
}

/// Matched mixed-precision reference for `C = op(A) * op(B)`.
///
/// Casting is the same IEEE RNE half conversion the conversion kernel performs,
/// layouts follow `layout`, and accumulation is a serial `k` order in the
/// selected precision. This is the oracle the WMMA kernels are compared against
/// so that expected FP16 quantization drift relative to FP32 is not confused
/// with a kernel defect.
pub fn oracle_gemm(
    a: &[f32],
    b: &[f32],
    rows: usize,
    columns: usize,
    reduction: usize,
    layout: GemmLayout,
    accumulation: OracleAccumulation,
) -> Result<Vec<f32>> {
    let expected_a = rows * reduction;
    let expected_b = reduction * columns;
    if a.len() != expected_a || b.len() != expected_b {
        bail!(
            "oracle GEMM operand sizes are {} and {}; expected {expected_a} and {expected_b}",
            a.len(),
            b.len()
        );
    }
    let a_h: Vec<f32> = a.iter().copied().map(round_to_f16).collect();
    let b_h: Vec<f32> = b.iter().copied().map(round_to_f16).collect();
    let mut out = vec![0.0f32; rows * columns];
    for row in 0..rows {
        for col in 0..columns {
            let mut acc = 0.0f32;
            let mut partial = 0.0f32;
            for t in 0..reduction {
                let av = if layout.transposed_a {
                    a_h[t * rows + row]
                } else {
                    a_h[row * reduction + t]
                };
                let bv = if layout.transposed_b {
                    b_h[col * reduction + t]
                } else {
                    b_h[t * columns + col]
                };
                match accumulation {
                    OracleAccumulation::Fp32 => acc = (av * bv) + acc,
                    OracleAccumulation::Fp16TileK | OracleAccumulation::Fp16Dot2 => partial = (av * bv) + partial,
                }
                let cadence = if accumulation == OracleAccumulation::Fp16Dot2 {2} else {WMMA_TILE_K as usize};
                if accumulation != OracleAccumulation::Fp32
                    && (t + 1) % cadence == 0
                {
                    acc = round_to_f16(acc + partial);
                    partial = 0.0;
                }
            }
            if partial != 0.0 {
                acc = round_to_f16(acc + partial);
            }
            out[row * columns + col] = acc;
        }
    }
    Ok(out)
}

/// Ordinary FP32 reference GEMM with `f64` accumulation.
///
/// This is the reference the reduced-precision arms are measured against when
/// the question is "how far from FP32 is this?", as opposed to
/// [`oracle_gemm`], which answers "is the kernel implementing the precision it
/// claims?".
pub fn reference_fp32_gemm(
    a: &[f32],
    b: &[f32],
    rows: usize,
    columns: usize,
    reduction: usize,
    layout: GemmLayout,
) -> Result<Vec<f32>> {
    let expected_a = rows * reduction;
    let expected_b = reduction * columns;
    if a.len() != expected_a || b.len() != expected_b {
        bail!(
            "FP32 reference operand sizes are {} and {}; expected {expected_a} and {expected_b}",
            a.len(),
            b.len()
        );
    }
    let mut out = vec![0.0f32; rows * columns];
    for row in 0..rows {
        for col in 0..columns {
            let mut acc = 0.0f64;
            for t in 0..reduction {
                let av = if layout.transposed_a {
                    a[t * rows + row]
                } else {
                    a[row * reduction + t]
                };
                let bv = if layout.transposed_b {
                    b[col * reduction + t]
                } else {
                    b[t * columns + col]
                };
                acc += f64::from(av) * f64::from(bv);
            }
            out[row * columns + col] = acc as f32;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Drift metrics
// ---------------------------------------------------------------------------

/// Absolute/RMS/relative drift of one measurement against a reference.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WmmaDrift {
    pub max_abs: f64,
    pub rms: f64,
    pub max_relative: f64,
    pub nonfinite: usize,
    pub samples: usize,
}

impl WmmaDrift {
    pub fn all_finite(&self) -> bool {
        self.nonfinite == 0
    }

    /// Relative measure that stays meaningful when the reference has a small
    /// magnitude: the error is scaled by `max(|reference|, reference_rms)` so a
    /// near-zero reference cannot manufacture an infinite ratio.
    pub fn scaled_relative(&self) -> f64 {
        self.max_relative
    }
}

/// Compare `actual` against `reference`.
pub fn drift(reference: &[f32], actual: &[f32]) -> Result<WmmaDrift> {
    if reference.len() != actual.len() {
        bail!(
            "drift comparison needs equal lengths, got {} and {}",
            reference.len(),
            actual.len()
        );
    }
    if reference.is_empty() {
        bail!("drift comparison needs at least one sample");
    }
    let reference_rms = (reference
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / reference.len() as f64)
        .sqrt();
    let floor = reference_rms.max(f64::MIN_POSITIVE);
    let mut max_abs = 0.0f64;
    let mut max_relative = 0.0f64;
    let mut squared = 0.0f64;
    let mut nonfinite = 0usize;
    for (expected, observed) in reference.iter().zip(actual) {
        if !observed.is_finite() {
            nonfinite += 1;
        }
        let error = (f64::from(*observed) - f64::from(*expected)).abs();
        max_abs = max_abs.max(error);
        squared += error * error;
        let scale = f64::from(*expected).abs().max(floor);
        max_relative = max_relative.max(error / scale);
    }
    Ok(WmmaDrift {
        max_abs,
        rms: (squared / reference.len() as f64).sqrt(),
        max_relative,
        nonfinite,
        samples: reference.len(),
    })
}

/// Token-level agreement between two `rows x columns` logit matrices.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogitAgreement {
    pub rows: usize,
    pub top1_matches: usize,
    pub mean_top5_overlap: f64,
    pub max_softmax_abs: f64,
    pub max_softmax_kl: f64,
    /// Rows where the reference top-1 and the candidate top-1 are within one
    /// floating-point epsilon of each other, i.e. a near-tie rather than a
    /// genuine disagreement.
    pub near_tie_top1: usize,
}

pub fn logit_agreement(
    reference: &[f32],
    actual: &[f32],
    rows: usize,
    columns: usize,
) -> Result<LogitAgreement> {
    if reference.len() != rows * columns || actual.len() != rows * columns {
        bail!("logit agreement requires rows*columns values in both matrices");
    }
    let mut report = LogitAgreement {
        rows,
        ..Default::default()
    };
    let mut overlap_total = 0.0f64;
    let softmax = |values: &[f32]| -> Vec<f64> {
        let max = values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f64> = values
            .iter()
            .map(|value| (f64::from(*value) - f64::from(max)).exp())
            .collect();
        let sum: f64 = exp.iter().sum();
        exp.into_iter()
            .map(|value| value / sum)
            .collect()
    };
    for row in 0..rows {
        let reference_row = &reference[row * columns..(row + 1) * columns];
        let actual_row = &actual[row * columns..(row + 1) * columns];
        let mut reference_order: Vec<usize> = (0..columns).collect();
        reference_order.sort_by(|a, b| {
            reference_row[*b]
                .partial_cmp(&reference_row[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut actual_order: Vec<usize> = (0..columns).collect();
        actual_order.sort_by(|a, b| {
            actual_row[*b]
                .partial_cmp(&actual_row[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if reference_order[0] == actual_order[0] {
            report.top1_matches += 1;
        } else {
            let gap = (f64::from(reference_row[reference_order[0]])
                - f64::from(reference_row[actual_order[0]]))
            .abs();
            if gap <= f64::EPSILON * 8.0 || gap <= 1.0e-6 {
                report.near_tie_top1 += 1;
            }
        }
        let k = 5.min(columns);
        let overlap = reference_order[..k]
            .iter()
            .filter(|index| actual_order[..k].contains(index))
            .count();
        overlap_total += overlap as f64 / k as f64;

        let reference_softmax = softmax(reference_row);
        let actual_softmax = softmax(actual_row);
        for (expected, observed) in reference_softmax.iter().zip(&actual_softmax) {
            report.max_softmax_abs = report.max_softmax_abs.max((expected - observed).abs());
        }
        let kl: f64 = reference_softmax
            .iter()
            .zip(&actual_softmax)
            .map(|(p, q)| {
                if *p > 0.0 {
                    p * (p / q.max(f64::MIN_POSITIVE)).ln()
                } else {
                    0.0
                }
            })
            .sum();
        report.max_softmax_kl = report.max_softmax_kl.max(kl);
    }
    report.mean_top5_overlap = overlap_total / rows as f64;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_matrix(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = ((state >> 8) & 0xFFFF) as f32 / 65_535.0;
                // Keep logits in a plausible range: [-1, 1) with a sinusoid so
                // the values are not all the same magnitude.
                (unit * 2.0 - 1.0) * (1.0 + (index % 7) as f32 * 0.15)
            })
            .collect()
    }

    fn device_or_skip(mode: WmmaMode) -> Option<VulkanDevice> {
        match VulkanDevice::new_for_wmma(mode) {
            Ok(device) => Some(device),
            Err(_) => None,
        }
    }

    #[test]
    fn mode_parsing_is_explicit_and_rejects_unknown_values() -> Result<()> {
        assert_eq!(WmmaMode::parse("off")?, WmmaMode::Off);
        assert_eq!(WmmaMode::parse("MIXED")?, WmmaMode::Mixed);
        assert_eq!(WmmaMode::parse("full-fp16")?, WmmaMode::FullFp16);
        assert!(WmmaMode::parse("half").is_err());
        assert!(!WmmaMode::Off.requested());
        assert!(WmmaMode::Mixed.requested());
        Ok(())
    }

    #[test]
    fn half_conversion_is_exact_round_to_nearest_even() {
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(1.0)), 1.0);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(-2.0)), -2.0);
        // 1 + 2^-11 is halfway between 1.0 and 1.0009765625; ties-to-even picks 1.0.
        assert_eq!(round_to_f16(1.0 + 2f32.powi(-11)), 1.0);
        // 1 + 3*2^-11 is halfway between 1.0009765625 and 1.001953125; ties-to-even rounds up.
        assert_eq!(round_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 1.001_953_125f32);
        // 65520 overflows the half range to infinity, as the hardware conversion does.
        assert!(round_to_f16(65_520.0).is_infinite());
        assert!(round_to_f16(-65_520.0).is_infinite());
        // Smallest positive subnormal half.
        assert_eq!(round_to_f16(2f32.powi(-25)), 0.0);
        assert_eq!(round_to_f16(2f32.powi(-24)), 2f32.powi(-24));
    }

    #[test]
    fn shape_admission_rejects_tiles_that_cannot_be_filled() {
        assert!(!wmma_shape_eligible(8, 4096, 4096));
        assert!(!wmma_shape_eligible(1024, 1024, 8));
        assert!(!wmma_shape_eligible(16, 16, 16));
        assert!(wmma_shape_eligible(256, 256, 128));
        assert!(wmma_shape_eligible(64, 1024, 64));
    }

    #[test]
    fn drift_and_agreement_metrics_are_self_consistent() -> Result<()> {
        let reference = vec![0.5f32, -1.0, 2.0, 3.0];
        assert_eq!(drift(&reference, &reference)?.max_abs, 0.0);
        let perturbed = vec![0.5f32, -1.0, 2.0, 3.25];
        let measured = drift(&reference, &perturbed)?;
        assert!((measured.max_abs - 0.25).abs() < 1e-7);
        assert!(measured.all_finite());
        let agreement = logit_agreement(&reference, &perturbed, 1, 4)?;
        assert_eq!(agreement.top1_matches, 1);
        assert_eq!(agreement.mean_top5_overlap, 1.0);
        Ok(())
    }

    /// Cooperative matrix versus its matched oracle. The oracle shares casting,
    /// layout and accumulation policy, so this isolates kernel correctness.
    #[test]
    fn wmma_mixed_matches_matched_oracle() -> Result<()> {
        let Some(device) = device_or_skip(WmmaMode::Mixed) else {
            return Ok(());
        };
        let gemm = WmmaGemm::new(&device, WmmaMode::Mixed)?;
        if !gemm.capabilities().mixed_supported() {
            return Ok(());
        }
        for (rows, columns, reduction) in [(64usize, 48usize, 40usize), (33, 17, 19), (128, 128, 96)]
        {
            for layout in [
                GemmLayout::default(),
                GemmLayout {
                    transposed_a: true,
                    transposed_b: false,
                },
                GemmLayout {
                    transposed_a: false,
                    transposed_b: true,
                },
                GemmLayout {
                    transposed_a: true,
                    transposed_b: true,
                },
            ] {
                let a = deterministic_matrix(rows * reduction, 0x1234);
                let b = deterministic_matrix(reduction * columns, 0x5678);
                let oracle = oracle_gemm(
                    &a,
                    &b,
                    rows,
                    columns,
                    reduction,
                    layout,
                    OracleAccumulation::Fp32,
                )?;
                let observed = gemm.matmul(
                    WmmaGemmArm::MixedFp32Accumulate,
                    &a,
                    &b,
                    rows,
                    columns,
                    reduction,
                    layout,
                )?;
                let measured = drift(&oracle, &observed)?;
                assert!(measured.all_finite());
                assert!(
                    measured.max_abs <= MIXED_VS_MATCHED_ORACLE_MAX_ABS,
                    "mixed WMMA {}x{}x{} {:?} vs matched oracle max_abs {} > {}",
                    rows,
                    columns,
                    reduction,
                    layout,
                    measured.max_abs,
                    MIXED_VS_MATCHED_ORACLE_MAX_ABS
                );
            }
        }
        Ok(())
    }

    /// The full-FP16 arm must satisfy its own contract: finite output and the
    /// matched-oracle bound, not merely "a number came back".
    #[test]
    fn wmma_full_fp16_matches_matched_oracle() -> Result<()> {
        let Some(device) = device_or_skip(WmmaMode::FullFp16) else {
            return Ok(());
        };
        let gemm = WmmaGemm::new(&device, WmmaMode::FullFp16)?;
        if !gemm.capabilities().full_fp16_supported() {
            return Ok(());
        }
        for (rows, columns, reduction) in [(64usize, 64usize, 32usize), (37, 21, 48)] {
            let a = deterministic_matrix(rows * reduction, 0x2468);
            let b = deterministic_matrix(reduction * columns, 0x1357);
            let oracle = oracle_gemm(
                &a,
                &b,
                rows,
                columns,
                reduction,
                GemmLayout::default(),
                OracleAccumulation::Fp16TileK,
            )?;
            let observed = gemm.matmul(
                WmmaGemmArm::FullFp16Accumulate,
                &a,
                &b,
                rows,
                columns,
                reduction,
                GemmLayout::default(),
            )?;
            let measured = drift(&oracle, &observed)?;
            assert!(measured.all_finite());
            let magnitude = oracle.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let limit = (2.0 * f64::from(f16_ulp(magnitude))).max(FULL_VS_MATCHED_ORACLE_MAX_ABS);
            assert!(
                measured.max_abs <= limit,
                "full-FP16 WMMA {}x{}x{} vs matched oracle max_abs {} > {}",
                rows,
                columns,
                reduction,
                measured.max_abs,
                limit
            );
        }
        Ok(())
    }

    /// Plain FP16 storage must agree with the mixed oracle too: it is the
    /// ablation control, so its own accuracy has to be established.
    #[test]
    fn plain_fp16_gemm_matches_mixed_oracle() -> Result<()> {
        let Some(device) = device_or_skip(WmmaMode::Off) else {
            return Ok(());
        };
        let gemm = WmmaGemm::new(&device, WmmaMode::Off)?;
        let rows = 40usize;
        let columns = 24usize;
        let reduction = 35usize;
        let a = deterministic_matrix(rows * reduction, 0x9E37);
        let b = deterministic_matrix(reduction * columns, 0x79B9);
        let oracle = oracle_gemm(
            &a,
            &b,
            rows,
            columns,
            reduction,
            GemmLayout::default(),
            OracleAccumulation::Fp32,
        )?;
        let observed = gemm.matmul(
            WmmaGemmArm::PlainF16Storage,
            &a,
            &b,
            rows,
            columns,
            reduction,
            GemmLayout::default(),
        )?;
        let measured = drift(&oracle, &observed)?;
        assert!(
            measured.max_abs <= MIXED_VS_MATCHED_ORACLE_MAX_ABS,
            "plain FP16 GEMM vs mixed oracle max_abs {}",
            measured.max_abs
        );
        Ok(())
    }

    /// An explicit unsupported request must produce a diagnostic, never a silent
    /// FP16 shader that pretends to be WMMA.
    #[test]
    fn unsupported_explicit_mode_reports_diagnostic() -> Result<()> {
        let Ok(device) = VulkanDevice::new_for_wmma(WmmaMode::Mixed) else {
            return Ok(());
        };
        let capabilities = device.wmma_capabilities();
        if capabilities.mixed_supported() {
            // Hardware is capable; the negative control cannot be built here.
            return Ok(());
        }
        let error = match WmmaGemm::new(&device, WmmaMode::Mixed) {
            Ok(_) => bail!("explicit unsupported WMMA request unexpectedly succeeded"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("not supported by this Vulkan device"),
            "unexpected diagnostic: {error}"
        );
        Ok(())
    }
}
