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
//! Admission requires an AMD RDNA3-or-newer PCI ID; names and vendor overrides
//! cannot admit an older GPU. The device must additionally
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
/// Phase LXIX two-bank bank-reuse module. Selectable through
/// [`FULL_FP16_VARIANT_ENV`]; never the default until a promotion decision is
/// taken, because it is still slower than the FP32 kernel it replaces.
const WMMA_GEMM_F16ACC_TWOBANK_SPV: &[u8] =
    include_bytes!("../shaders/wmma_gemm_f16_f16acc_twobank.spv");
/// Phase LXXI two-wave tiling module. Same reduction contract as the two-bank
/// module, but two wave32 execution groups per workgroup and two output tiles
/// per wave, so a single wave keeps four instead of eight 16x16 FP16
/// accumulators live. Selectable through [`FULL_FP16_VARIANT_ENV`]; never the
/// default.
const WMMA_GEMM_F16ACC_TWOWAVE_SPV: &[u8] =
    include_bytes!("../shaders/wmma_gemm_f16_f16acc_twowave.spv");
/// Phase LXXIX P3 epilogue on the Phase LXXI two-wave tiling. Staging,
/// workgroup shape, K-slice stream, bank assignment and per-bank accumulation
/// are byte-for-byte the two-wave contract; the 4 + 3 rounding merge tree is
/// replaced by one float32 left-to-right sum of the eight binary16 bank
/// partials and a single binary16 rounding. This is the production Full FP16
/// module; see `RDNA3_P3_RELEASE.md`.
const WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV: &[u8] =
    include_bytes!("../shaders/wmma_gemm_f16_f16acc_twowave_p3.spv");
/// Phase LXXXVIII production **staging** module (candidate C1) for the same P3
/// contract. The arithmetic, the two-wave workgroup, the K-slice stream, the
/// per-bank accumulation order, the float32 bank sum and the single binary16
/// rounding are identical to the module above; only the staging schedule
/// changes, and only inside a workgroup-uniform predicate that proves every
/// packed 32-bit load aligned and in bounds. It is selected per dispatch by
/// [`batched_staging_eligible`] and the module above stays the fallback. See
/// `RDNA3_P3_RELEASE.md`.
const WMMA_GEMM_F16ACC_TWOWAVE_P3_C1_SPV: &[u8] =
    include_bytes!("../shaders/wmma_gemm_f16_f16acc_twowave_p3_c1.spv");
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

/// Environment variable that selects which committed Full FP16 shader the
/// `FullFp16` arm loads. Unset/empty means [`FullFp16Variant::P3`], the module
/// the production route dispatches. Setting it to `split8`, `two-bank` or
/// `two-wave` selects one of the earlier modules; those remain available as a
/// fallback and for comparison, and are not removed by the P3 default.
pub const FULL_FP16_VARIANT_ENV: &str = "HIERARCHOS_WMMA_FULL_FP16_VARIANT";

/// The committed Full FP16 shader variants this build can dispatch.
///
/// The variant selects the actual SPIR-V module, not a label: each variant is a
/// distinct `include_bytes!` blob, so the pipelines they create have different
/// shader signatures (the FNV-1a of the SPIR-V is the shader component of the
/// pipeline cache key in [`vulkan::ComputeKernel`]). Two variants therefore
/// cannot share a cached pipeline.
///
/// [`Self::Split8`], [`Self::TwoBank`] and [`Self::TwoWave`] evaluate the same
/// reduction contract (bank `b` owns slices `b, b+8, ...`, pairwise FP16 merge
/// tree, identical operand order and rounding points) and are expected to
/// produce bit-identical FP32 outputs. [`Self::P3`] keeps that accumulation
/// contract and changes the epilogue only: the eight bank partials are summed
/// left-to-right in float32 and rounded to binary16 once, so its output is
/// deliberately **not** bit-identical to the tree — it is closer to the
/// correctly rounded FP16 result. See `RDNA3_P3_RELEASE.md`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FullFp16Variant {
    /// The eight-bank split8 module committed as `wmma_gemm_f16_f16acc.spv`.
    /// The pre-P3 production arm; still selectable as a fallback.
    Split8,
    /// The Phase LXIX two-bank bank-reuse module
    /// (`wmma_gemm_f16_f16acc_twobank.spv`). Same arithmetic, fewer live
    /// accumulator banks, smaller partial array, different pipeline.
    TwoBank,
    /// The Phase LXXI two-wave tiling module
    /// (`wmma_gemm_f16_f16acc_twowave.spv`). Same arithmetic as [`Self::TwoBank`]
    /// and the same per-workgroup output tile, but the four 16x16 output tiles
    /// are split across two wave32 execution groups so each wave keeps two
    /// tiles' accumulators live instead of four. Costs two real workgroup
    /// barriers per K slice, which every single-wave variant elides.
    TwoWave,
    /// The Phase LXXIX P3 epilogue module
    /// (`wmma_gemm_f16_f16acc_twowave_p3.spv`), and the production Full FP16
    /// route. Same two-wave workgroup, staging, barriers and per-bank binary16
    /// accumulation; one float32 left-to-right sum of the eight bank partials
    /// and a single binary16 rounding instead of the 4 + 3 rounding tree.
    /// Qualified on the physical RDNA3 device over 32 K=96 cases: 31/32 closer
    /// to the independently computed FP16 reference than the tree, none worse.
    #[default]
    P3,
}

impl FullFp16Variant {
    pub const ALL: [Self; 4] = [Self::Split8, Self::TwoBank, Self::TwoWave, Self::P3];

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "split8" | "eight-bank" | "eightbank" | "eight_bank" | "8" => Ok(Self::Split8),
            "two-bank" | "twobank" | "two_bank" | "2" => Ok(Self::TwoBank),
            "two-wave" | "twowave" | "two_wave" | "64" => Ok(Self::TwoWave),
            "p3" | "twowave-p3" | "two-wave-p3" => Ok(Self::P3),
            other => bail!(
                "unsupported full-fp16 variant {other:?}; expected p3, split8, two-bank or two-wave"
            ),
        }
    }

    /// Resolve a raw environment value. Absent or empty selects the default.
    ///
    /// Split out from [`Self::from_env`] so the environment contract is testable
    /// without mutating process state.
    pub fn from_env_value(value: Option<String>) -> Result<Self> {
        match value {
            Some(value) if !value.trim().is_empty() => Self::parse(&value),
            _ => Ok(Self::default()),
        }
    }

    /// Read [`FULL_FP16_VARIANT_ENV`]. An absent or empty value selects
    /// [`Self::P3`].
    pub fn from_env() -> Result<Self> {
        Self::from_env_value(std::env::var(FULL_FP16_VARIANT_ENV).ok())
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Split8 => "split8",
            Self::TwoBank => "two-bank",
            Self::TwoWave => "two-wave",
            Self::P3 => "p3",
        }
    }

    /// The optional batched-staging module this variant can dispatch instead of
    /// [`Self::spirv`].
    ///
    /// Only the production P3 contract has one: C1 is the same P3 arithmetic
    /// with the four dependent global-load/shared-store pairs of each K slice
    /// replaced by four independent 32-bit loads issued before any store. Every
    /// other variant returns `None`, so a non-P3 selection cannot reach it.
    pub fn staged_spirv(self) -> Option<&'static [u8]> {
        match self {
            Self::Split8 | Self::TwoBank | Self::TwoWave => None,
            Self::P3 => Some(WMMA_GEMM_F16ACC_TWOWAVE_P3_C1_SPV),
        }
    }

    /// This variant's batched-staging shader name, or `None` when it has no
    /// batched-staging module.
    pub fn staged_shader_name(self) -> Option<&'static str> {
        match self {
            Self::Split8 | Self::TwoBank | Self::TwoWave => None,
            Self::P3 => Some("wmma_gemm_f16_f16acc_twowave_p3_c1"),
        }
    }

    /// Name this variant's SPIR-V carries in the build-generated shader
    /// registry (`shaders/<name>.spv`).
    pub fn shader_name(self) -> &'static str {
        match self {
            Self::Split8 => "wmma_gemm_f16_f16acc",
            Self::TwoBank => "wmma_gemm_f16_f16acc_twobank",
            Self::TwoWave => "wmma_gemm_f16_f16acc_twowave",
            Self::P3 => "wmma_gemm_f16_f16acc_twowave_p3",
        }
    }

    /// The SPIR-V module this variant dispatches.
    pub fn spirv(self) -> &'static [u8] {
        match self {
            Self::Split8 => WMMA_GEMM_F16ACC_SPV,
            Self::TwoBank => WMMA_GEMM_F16ACC_TWOBANK_SPV,
            Self::TwoWave => WMMA_GEMM_F16ACC_TWOWAVE_SPV,
            Self::P3 => WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV,
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
    /// Physical AMD RDNA3-or-newer PCI admission, independent of driver features.
    pub rdna3_or_newer: bool,
    /// WMMA was requested on admitted hardware, so optional features were queried.
    pub probed: bool,
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
        self.rdna3_or_newer && self.probed && self.enabled
            && self.extension_exposed && self.feature_supported && self.memory_model_supported
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
        self.rdna3_or_newer && self.probed && self.enabled
            && self.extension_exposed && self.feature_supported && self.memory_model_supported
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
        out.push_str(&format!("WMMA admission: rdna3_or_newer={} probed={}\n", self.rdna3_or_newer, self.probed));
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
        if !self.rdna3_or_newer {
            return "WMMA requires AMD RDNA3 or newer; this device retains its existing FP32 path";
        }
        if !self.probed {
            return "WMMA is off; cooperative-matrix features were not probed";
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

/// Measured minimum work for the cooperative-matrix kernel to beat the FP32
/// vendor GEMM.
///
/// Paired ABBA sweeps of `wmma_bench` on RDNA3 (16x16x16 cooperative tile) show
/// the FP32 kernel winning every small shape (0.59x-0.82x) and the cooperative
/// kernel winning every shape from roughly 16 MMAC upward (1.08x-2.07x). This
/// floor keeps a ~2x margin above the largest measured losing shape
/// (128x256x256 = 8.4 MMAC) and declines only the marginal 5-10 MMAC wins, so a
/// graph never pays a dispatch overhead it cannot amortize.
pub const WMMA_MIN_PROFITABLE_MACS: u64 = 16_000_000;

/// Whether a shape is admissible and large enough for admission to pay off.
///
/// This is a dispatch policy, not a numerical contract: a declining caller keeps
/// its existing FP32 kernel and its results are unchanged.
pub fn wmma_shape_profitable(rows: usize, columns: usize, reduction: usize) -> bool {
    wmma_shape_eligible(rows, columns, reduction)
        && (rows as u64)
            .saturating_mul(columns as u64)
            .saturating_mul(reduction as u64)
            >= WMMA_MIN_PROFITABLE_MACS
}

/// Opt-in GEMM encodings used by callers that already hold FP16 storage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GemmLayout {
    pub transposed_a: bool,
    pub transposed_b: bool,
}

/// Everything the batched-staging (C1) contract is a statement about.
///
/// This is a *dispatch* description, not a numerical one: it names the GEMM
/// geometry, the two operand strides the selector derives from it, and the byte
/// capacity of the two bound operand buffers. [`batched_staging_eligible`] is a
/// pure function of these fields, so the production predicate can be tested
/// without a device and cannot drift from the shader it guards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StagingGeometry {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub layout: GemmLayout,
    /// `GpuBuffer::capacity_bytes()` of the buffer bound as operand A.
    pub a_capacity_bytes: usize,
    /// `GpuBuffer::capacity_bytes()` of the buffer bound as operand B.
    pub b_capacity_bytes: usize,
}

impl StagingGeometry {
    /// A's leading dimension, exactly as [`WmmaGemm::record_gemm`] derives it:
    /// the row stride is `rows` when A is `[k, m]` and `reduction` when A is
    /// `[m, k]`.
    pub fn lda(&self) -> usize {
        if self.layout.transposed_a {
            self.rows
        } else {
            self.reduction
        }
    }

    /// B's leading dimension: `reduction` when B is `[n, k]` and `columns` when
    /// B is `[k, n]`.
    pub fn ldb(&self) -> usize {
        if self.layout.transposed_b {
            self.reduction
        } else {
            self.columns
        }
    }
}

/// Whether the C1 batched-staging module is provably safe for this dispatch.
///
/// The shader's fast path is taken per K slice and per workgroup, and it issues
/// four packed 32-bit global loads before any shared store. Selecting it here
/// therefore requires that, for *every* dispatched workgroup and *every* slice,
/// those loads are aligned and in bounds. The conditions below are exactly that
/// derivation, and they are stated conservatively: a geometry that satisfies
/// only some of them keeps the original P3 module, which is what the measured
/// odd-stride regression in Phase LXXXVII requires.
///
/// 1. **Tile coverage.** `rows % 32 == 0`, `columns % 32 == 0`, `reduction % 16
///    == 0`. The workgroup grid is `ceil(columns/32) x ceil(rows/32)` and each
///    group stages rows `m0..m0+32` and columns `n0..n0+32` for K slices of 16,
///    so these divisibilities are what make `m0 + 32 <= m`, `n0 + 32 <= n` and
///    `k0 + 16 <= k` *uniformly* true. With them the shader's per-element
///    bounds tests are provably never taken, which is the only reason the fast
///    path may drop them.
/// 2. **Packed-pair stride parity.** `lda % 2 == 0` and `ldb % 2 == 0`. A packed
///    load is `a[off >> 1]` on an `f16vec2[]`, so it needs an even element
///    index. For row-major A the index is `gm * lda + gk` with even `gm`
///    implied by nothing — it is `gm * lda` that must stay even for odd `gm`,
///    hence even `lda`; symmetrically row-major B needs even `ldb`. The
///    transposed layouts put the pair along the contiguous axis, so parity
///    there is structural, and requiring both strides even keeps the predicate
///    uniform across the four branches (it is also the shader's own
///    `wide_a`/`wide_b` gate). Condition 1 already implies both parities for
///    every layout this crate can build; the check is kept explicit so a future
///    caller-supplied stride cannot slip through it.
/// 3. **No base offset exists to be misaligned.** The kernel has no offset
///    operand: its push constant is `m, n, k, lda, ldb, ldc, flags, reserved`
///    and `reserved` is unused, and `record_dispatch` binds whole buffers at
///    descriptor offset zero. The only address term is `row * stride + col`, so
///    the byte address of a packed load is `allocation_start + 2 * element`,
///    with `allocation_start` four-byte aligned by the `GpuBuffer` ABI this
///    crate's every `float`/`u32` kernel already depends on.
/// 4. **Complete 32-bit access bounds.** The index arithmetic in the shader is
///    `uint`, so the largest element index it can form must fit in 32 bits, and
///    the buffer bound at that binding must actually cover the elements the
///    fast path reads. `a_last`/`b_last` below are the highest element index the
///    kernel can address for the geometry (`rows * lda` style products, which is
///    the full operand), and both are compared against the bound capacity.
///
/// Batch strides and per-batch element offsets are deliberately absent: this
/// kernel is a single-matrix GEMM with no batch dimension in any address
/// expression, and per-batch admission happens above the kernel by dispatching
/// each batch separately. There is nothing here for a batch term to misalign.
pub fn batched_staging_eligible(geometry: &StagingGeometry) -> bool {
    let rows = geometry.rows as u64;
    let columns = geometry.columns as u64;
    let reduction = geometry.reduction as u64;
    if rows == 0 || columns == 0 || reduction == 0 {
        return false;
    }
    // (1) Every tile and every K slice is full.
    if geometry.rows % 32 != 0 || geometry.columns % 32 != 0 || geometry.reduction % 16 != 0 {
        return false;
    }
    let lda = geometry.lda();
    let ldb = geometry.ldb();
    // (2) Packed pair parity, stated for the strides the shader actually reads.
    if lda % 2 != 0 || ldb % 2 != 0 {
        return false;
    }
    let lda = lda as u64;
    let ldb = ldb as u64;
    // (4) Highest addressable element, per layout, plus the packed pair's second
    // element (the kernel reads `i` and `i + 1`, and `i + 1 <= last` is implied
    // by the full-tile coverage in (1)).
    let a_last = if geometry.layout.transposed_a {
        (reduction - 1) * lda + (rows - 1)
    } else {
        (rows - 1) * lda + (reduction - 1)
    };
    let b_last = if geometry.layout.transposed_b {
        (columns - 1) * ldb + (reduction - 1)
    } else {
        (reduction - 1) * ldb + (columns - 1)
    };
    // The shader forms these in 32-bit `uint` arithmetic; `off + 1` is the
    // highest value it can produce in the masked path, so leave one element of
    // headroom below the type's maximum.
    let ceiling = u32::MAX as u64 - 2;
    if a_last > ceiling || b_last > ceiling {
        return false;
    }
    // The bound regions must cover every element the kernel can read. A buffer
    // shorter than the geometry claims would make the fast path read past its
    // binding, so it is a fallback rather than a silent over-read.
    let a_bytes = match (a_last + 1).checked_mul(2) {
        Some(bytes) => bytes,
        None => return false,
    };
    let b_bytes = match (b_last + 1).checked_mul(2) {
        Some(bytes) => bytes,
        None => return false,
    };
    a_bytes <= geometry.a_capacity_bytes as u64 && b_bytes <= geometry.b_capacity_bytes as u64
}

/// One dispatch's staging choice, for the receipts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WmmaStagingCounts {
    /// Dispatches that used the batched (C1) staging module.
    pub batched: u64,
    /// Dispatches that used the original masked-staging P3 module.
    pub masked: u64,
}

struct WmmaKernels {
    convert: vulkan::ComputeKernel,
    mixed: Option<vulkan::ComputeKernel>,
    /// [`FullFp16Variant::spirv`] — the contract module and the staging fallback.
    full: Option<vulkan::ComputeKernel>,
    /// [`FullFp16Variant::staged_spirv`] — C1, dispatched only when
    /// [`batched_staging_eligible`] proves it safe for this geometry.
    full_staging: Option<vulkan::ComputeKernel>,
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
    full_fp16_variant: FullFp16Variant,
    capabilities: VulkanWmmaCapabilities,
    kernels: WmmaKernels,
    cooperative_dispatches: AtomicU64,
    plain_dispatches: AtomicU64,
    shape_fallbacks: AtomicU64,
    staging_batched: AtomicU64,
    staging_masked: AtomicU64,
}

impl WmmaGemm {
    /// Build the executor for `mode`.
    ///
    /// `Off` still succeeds and still provides the plain FP16 control arm used
    /// by ablation benchmarks; it simply never creates a cooperative-matrix
    /// kernel. `Mixed`/`FullFp16` fail with the full diagnostic when the device
    /// does not satisfy [`VulkanWmmaCapabilities::require`].
    pub fn new(device: &VulkanDevice, mode: WmmaMode) -> Result<Self> {
        Self::new_with_full_fp16_variant(device, mode, FullFp16Variant::from_env()?)
    }

    /// As [`Self::new`], with the Full FP16 shader variant chosen explicitly.
    ///
    /// The variant selects the SPIR-V module the `FullFp16` arm loads. It never
    /// changes the requested mode: `Mixed` and `Off` do not load or dispatch the
    /// Full FP16 module at all, and asking for `full-fp16` without a variant
    /// selects [`FullFp16Variant::P3`], the qualified production epilogue.
    /// [`FullFp16Variant::Split8`] and the two earlier variants remain
    /// selectable through [`FULL_FP16_VARIANT_ENV`] and through this
    /// constructor, so the previous pipeline is preserved as a fallback.
    pub fn new_with_full_fp16_variant(
        device: &VulkanDevice,
        mode: WmmaMode,
        full_fp16_variant: FullFp16Variant,
    ) -> Result<Self> {
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
            Some(gemm_kernel(
                full_fp16_variant.spirv(),
                full_fp16_variant.shader_name(),
            )?)
        } else {
            None
        };
        // The batched-staging module is built only for a variant that has one and
        // only on a device that supports the Full FP16 contract at all, so a
        // device that cannot run P3 never creates a C1 pipeline either.
        let full_staging = match (full_fp16_variant.staged_spirv(), full.is_some()) {
            (Some(spirv), true) => Some(gemm_kernel(
                spirv,
                full_fp16_variant
                    .staged_shader_name()
                    .unwrap_or("full-fp16-staging"),
            )?),
            _ => None,
        };
        Ok(Self {
            device: device.clone(),
            mode,
            full_fp16_variant,
            capabilities,
            kernels: WmmaKernels {
                convert,
                mixed,
                full,
                full_staging,
                plain,
                fp32,
            },
            cooperative_dispatches: AtomicU64::new(0),
            plain_dispatches: AtomicU64::new(0),
            shape_fallbacks: AtomicU64::new(0),
            staging_batched: AtomicU64::new(0),
            staging_masked: AtomicU64::new(0),
        })
    }

    pub fn mode(&self) -> WmmaMode {
        self.mode
    }

    /// The Full FP16 variant whose SPIR-V this executor loaded.
    ///
    /// Every constructor assigns this from a committed [`FullFp16Variant`], so
    /// the value identifies the module exactly; `wmma_qualify` and `wmma_bench`
    /// print it beside [`Self::full_fp16_pipeline_identity`]. There is no
    /// caller-supplied or qualification-only module in this build, so no
    /// executor can report a variant it is not dispatching.
    pub fn full_fp16_variant(&self) -> FullFp16Variant {
        self.full_fp16_variant
    }

    /// Identity of the pipeline the Full FP16 arm actually dispatches.
    ///
    /// The shader component is the FNV-1a signature of the exact SPIR-V bytes,
    /// which is also the shader component of the pipeline cache key, so two
    /// variants with different identities cannot reuse each other's interned
    /// pipeline. Reported by `wmma_bench` beside the selected mode.
    pub fn full_fp16_pipeline_identity(&self) -> Option<String> {
        self.kernels.full.as_ref().map(|kernel| {
            format!(
                "{}/spirv-fnv1a64:0x{:016x}/len:{}",
                kernel.shader_name().unwrap_or("unknown"),
                kernel.shader_signature(),
                self.full_fp16_variant.spirv().len()
            )
        })
    }

    pub fn capabilities(&self) -> VulkanWmmaCapabilities {
        self.capabilities
    }

    /// Identity of the batched-staging module, when this executor has one.
    ///
    /// `None` for every variant that has no batched-staging module and for a
    /// device that cannot run the Full FP16 contract at all. Reported beside
    /// [`Self::full_fp16_pipeline_identity`] so a receipt names both the P3
    /// contract module and the C1 module, and [`Self::staging_counts`] says
    /// which of them each dispatch actually used.
    pub fn full_fp16_staging_identity(&self) -> Option<String> {
        self.kernels.full_staging.as_ref().map(|kernel| {
            format!(
                "{}/spirv-fnv1a64:0x{:016x}/len:{}",
                kernel.shader_name().unwrap_or("unknown"),
                kernel.shader_signature(),
                self.full_fp16_variant.staged_spirv().map_or(0, |spirv| spirv.len())
            )
        })
    }

    /// Which staging module this executor dispatched, counted per dispatch.
    ///
    /// Both counters are zero until a Full FP16 dispatch happens, and they stay
    /// zero for every other arm: the Mixed, plain-FP16 and FP32 kernels share
    /// the P3 arithmetic contract but never select a staging module.
    pub fn staging_counts(&self) -> WmmaStagingCounts {
        WmmaStagingCounts {
            batched: self.staging_batched.load(Ordering::Relaxed),
            masked: self.staging_masked.load(Ordering::Relaxed),
        }
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
        // The leading dimensions are derived here, ahead of the kernel choice, so
        // the staging predicate and the push constants cannot disagree about the
        // strides the shader will read.
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
        // The C1 module replaces the staging schedule only, so it is selected per
        // dispatch from the proven geometry contract; anything it cannot prove
        // keeps the original masked-staging P3 module.
        let mut batched_staging = false;
        let kernel = match arm {
            WmmaGemmArm::MixedFp32Accumulate => self
                .kernels
                .mixed
                .as_ref()
                .context("mixed WMMA kernel is unavailable on this device")?,
            WmmaGemmArm::FullFp16Accumulate => {
                let masked = self
                    .kernels
                    .full
                    .as_ref()
                    .context("full-FP16 WMMA kernel is unavailable on this device")?;
                match self.kernels.full_staging.as_ref() {
                    Some(batched)
                        if batched_staging_eligible(&StagingGeometry {
                            rows,
                            columns,
                            reduction,
                            layout,
                            a_capacity_bytes: a.capacity_bytes(),
                            b_capacity_bytes: b.capacity_bytes(),
                        }) =>
                    {
                        batched_staging = true;
                        batched
                    }
                    _ => masked,
                }
            }
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
            if arm == WmmaGemmArm::FullFp16Accumulate {
                if batched_staging {
                    self.staging_batched.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.staging_masked.fetch_add(1, Ordering::Relaxed);
                }
            }
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
    /// Two-bank split model: alternating 16-wide K slices accumulate into two
    /// separate FP16 accumulators with the RDNA3 dot2 cadence, and the two banks
    /// are merged once, in binary16, per output element.
    ///
    /// This was the contract of the Phase LXIV two-bank module. A matched
    /// oracle has to model the accumulation order the kernel actually issues;
    /// `Fp16TileK` models a single accumulator rounded once per K slice, which
    /// is neither what that kernel did nor what the hardware does, so it could
    /// not serve as the implementation-correctness reference for it. Retained
    /// so the two-bank receipt stays reproducible.
    Fp16Split2,
    /// Eight-bank split model: consecutive 16-wide K slices are assigned to
    /// eight FP16 accumulator banks round-robin with the RDNA3 dot2 cadence,
    /// and the banks are merged once, in binary16, by the pairwise tree
    /// `((b0+b1)+(b2+b3)) + ((b4+b5)+(b6+b7))`.
    ///
    /// This is the contract the current `wmma_gemm_f16_f16acc` module issues.
    /// Blocked summation bounds the rounding error of a fixed-precision
    /// reduction: each bank carries 1/8 of the terms, so each running sum is
    /// about eight times smaller and the roundings it absorbs are
    /// proportionally finer relative to the final result.
    Fp16Split8,
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
            // The split models keep one accumulator per K-slice bank; the
            // other models keep a single accumulator at index 0. Unused banks
            // stay exactly zero, and adding binary16 zero is exact, so a
            // reduction shorter than the bank count reduces to the same value.
            let mut banks = [0.0f32; 8];
            let bank_count: usize = match accumulation {
                OracleAccumulation::Fp16Split2 => 2,
                OracleAccumulation::Fp16Split8 => 8,
                _ => 1,
            };
            let mut bank_of_last = 0usize;
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
                    _ => partial = (av * bv) + partial,
                }
                let cadence = match accumulation {
                    OracleAccumulation::Fp32 => 0,
                    OracleAccumulation::Fp16TileK => WMMA_TILE_K as usize,
                    OracleAccumulation::Fp16Dot2
                    | OracleAccumulation::Fp16Split2
                    | OracleAccumulation::Fp16Split8 => 2,
                };
                if cadence != 0 && (t + 1) % cadence == 0 {
                    let bank = if bank_count > 1 {
                        (t / WMMA_TILE_K as usize) % bank_count
                    } else {
                        0
                    };
                    banks[bank] = round_to_f16(banks[bank] + partial);
                    bank_of_last = bank;
                    partial = 0.0;
                }
            }
            if partial != 0.0 {
                banks[bank_of_last] = round_to_f16(banks[bank_of_last] + partial);
            }
            out[row * columns + col] = match accumulation {
                OracleAccumulation::Fp16Split2 => round_to_f16(banks[0] + banks[1]),
                OracleAccumulation::Fp16Split8 => {
                    // Pairwise merge tree, exactly as the shader epilogue does.
                    let l0 = round_to_f16(banks[0] + banks[1]);
                    let l1 = round_to_f16(banks[2] + banks[3]);
                    let l2 = round_to_f16(banks[4] + banks[5]);
                    let l3 = round_to_f16(banks[6] + banks[7]);
                    let m0 = round_to_f16(l0 + l1);
                    let m1 = round_to_f16(l2 + l3);
                    round_to_f16(m0 + m1)
                }
                OracleAccumulation::Fp32 => acc,
                _ => banks[0],
            };
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

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(bytes);
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

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
    fn full_fp16_variant_parsing_is_explicit_and_rejects_unknown_values() -> Result<()> {
        assert_eq!(FullFp16Variant::parse("split8")?, FullFp16Variant::Split8);
        assert_eq!(FullFp16Variant::parse("eight-bank")?, FullFp16Variant::Split8);
        assert_eq!(FullFp16Variant::parse("TWO-BANK")?, FullFp16Variant::TwoBank);
        assert_eq!(FullFp16Variant::parse(" twobank ")?, FullFp16Variant::TwoBank);
        assert_eq!(FullFp16Variant::parse("P3")?, FullFp16Variant::P3);
        let error = FullFp16Variant::parse("four-bank").unwrap_err().to_string();
        assert!(
            error.contains("unsupported full-fp16 variant") && error.contains("two-bank"),
            "an unavailable variant must fail with the supported values: {error}"
        );
        assert!(FullFp16Variant::parse("").is_err());
        Ok(())
    }

    #[test]
    fn full_fp16_variant_defaults_to_the_production_p3_module() -> Result<()> {
        assert_eq!(FullFp16Variant::default(), FullFp16Variant::P3);
        assert_eq!(FullFp16Variant::from_env_value(None)?, FullFp16Variant::P3);
        assert_eq!(
            FullFp16Variant::from_env_value(Some("   ".to_owned()))?,
            FullFp16Variant::P3
        );
        // The pre-P3 module stays selectable: the fallback is not removed by
        // the default change.
        assert_eq!(
            FullFp16Variant::from_env_value(Some("split8".to_owned()))?,
            FullFp16Variant::Split8
        );
        assert_eq!(
            FullFp16Variant::from_env_value(Some("p3".to_owned()))?,
            FullFp16Variant::P3
        );
        assert_eq!(
            FullFp16Variant::from_env_value(Some("two-bank".to_owned()))?,
            FullFp16Variant::TwoBank
        );
        assert_eq!(
            FullFp16Variant::from_env_value(Some("two-wave".to_owned()))?,
            FullFp16Variant::TwoWave
        );
        assert_eq!(
            FullFp16Variant::from_env_value(Some("two_wave".to_owned()))?,
            FullFp16Variant::TwoWave
        );
        assert_eq!(
            FullFp16Variant::from_env_value(Some("64".to_owned()))?,
            FullFp16Variant::TwoWave
        );
        assert!(FullFp16Variant::from_env_value(Some("split4".to_owned())).is_err());
        assert!(FullFp16Variant::from_env_value(Some("four-bank".to_owned())).is_err());
        // A qualification-only module is not a variant and cannot be named
        // through the production selector.
        assert!(FullFp16Variant::from_env_value(Some("experimental".to_owned())).is_err());
        assert!(
            FullFp16Variant::from_env_value(Some("phase-lxxix-p3".to_owned())).is_err()
        );
        Ok(())
    }

    /// P3 is the production Full FP16 arm, and the profitability predicate both
    /// production routes apply before consulting the selector rejects the
    /// documented direct-kernel regression geometry. The selector keeps its
    /// historical tile-eligibility gate; admission is enforced by the callers
    /// (`ProjectionWmma::admits` at construction and per recorded batch, and the
    /// Transformer graph's construction gate).
    #[test]
    fn p3_is_selected_only_for_profitable_geometry() -> Result<()> {
        let Some(device) = device_or_skip(WmmaMode::FullFp16) else {
            return Ok(());
        };
        let gemm = WmmaGemm::new(&device, WmmaMode::FullFp16)?;
        if !gemm.capabilities().full_fp16_supported() {
            return Ok(());
        }
        assert_eq!(gemm.full_fp16_variant(), FullFp16Variant::P3);

        for (rows, columns, reduction) in [
            (2048usize, 768usize, 2048usize),
            (512, 2048, 768),
            (128, 8192, 768),
            (512, 768, 768),
        ] {
            assert!(
                wmma_shape_profitable(rows, columns, reduction),
                "{rows}x{columns}x{reduction} is a protected production shape"
            );
            assert_eq!(
                gemm.select_arm(rows, columns, reduction),
                Some(WmmaGemmArm::FullFp16Accumulate),
                "{rows}x{columns}x{reduction} must take the P3 arm"
            );
        }

        // Eligible under the tile contract but below the 16 MMAC floor: the
        // documented direct-kernel regression shape, a marginal 8.4 MMAC
        // shape, and the LoRA control. Every production selection point asks
        // this predicate first, and it refuses all three.
        for (rows, columns, reduction) in [
            (16usize, 128usize, 32usize),
            (128, 256, 256),
            (64, 64, 64),
        ] {
            assert!(
                wmma_shape_eligible(rows, columns, reduction),
                "{rows}x{columns}x{reduction} must still pass the tile contract"
            );
            assert!(
                !wmma_shape_profitable(rows, columns, reduction),
                "{rows}x{columns}x{reduction} must be refused below the 16 MMAC floor"
            );
        }
        // Boundary of the predicate itself: 400 x 400 x 99 is 15.84 MMAC and
        // 400 x 400 x 100 is exactly the 16 MMAC floor.
        assert!(!wmma_shape_profitable(400, 400, 99));
        assert!(wmma_shape_profitable(400, 400, 100));
        Ok(())
    }

    /// Outside the production contract the executor keeps its previous routes:
    /// `off` has no cooperative arm, `mixed` keeps its own arm and fallback, and
    /// an unsupported Full FP16 request fails at the capability boundary.
    #[test]
    fn unsupported_modes_keep_their_existing_route() -> Result<()> {
        let Some(off_device) = device_or_skip(WmmaMode::Off) else {
            return Ok(());
        };
        let off = WmmaGemm::new(&off_device, WmmaMode::Off)?;
        assert_eq!(off.select_arm(2048, 768, 2048), None);
        assert!(off.full_fp16_pipeline_identity().is_none());

        // Probing is per requested mode, so the mixed arm needs its own device.
        let Some(device) = device_or_skip(WmmaMode::Mixed) else {
            return Ok(());
        };
        let mixed =
            WmmaGemm::new_with_full_fp16_variant(&device, WmmaMode::Mixed, FullFp16Variant::P3)?;
        if mixed.capabilities().mixed_supported() {
            assert_eq!(
                mixed.select_arm(2048, 768, 2048),
                Some(WmmaGemmArm::MixedFp32Accumulate),
                "a Full FP16 variant must not change Mixed dispatch"
            );
        }
        // Mixed keeps its existing eligibility gate: it is not the arm this
        // change tightens, and its fallback still covers sub-tile shapes.
        assert_eq!(
            mixed.select_arm(16, 128, 32),
            Some(WmmaGemmArm::MixedFp32Accumulate)
        );
        assert_eq!(
            mixed.select_arm(8, 8, 8),
            Some(WmmaGemmArm::PlainF16Storage)
        );

        // Capability boundary, independent of the physical device: an
        // unnegotiated or feature-incomplete device cannot be asked for the
        // Full FP16 arm at all.
        let mut unsupported = VulkanWmmaCapabilities::default();
        assert!(unsupported.require(WmmaMode::FullFp16).is_err());
        unsupported.probed = true;
        unsupported.enabled = true;
        assert!(unsupported.require(WmmaMode::FullFp16).is_err());
        Ok(())
    }

    /// The production constructor loads the exact qualified P3 module — same
    /// bytes the Phase LXXIX GPU campaign dispatched — and reports a pipeline
    /// identity that is not an experimental route. The previous module remains
    /// loadable through the same constructor as a fallback.
    #[test]
    fn production_default_dispatches_the_p3_module() -> Result<()> {
        let Some(device) = device_or_skip(WmmaMode::FullFp16) else {
            return Ok(());
        };
        let production = WmmaGemm::new(&device, WmmaMode::FullFp16)?;
        if !production.capabilities().full_fp16_supported() {
            return Ok(());
        }

        // The committed module is the artifact Phase LXXIX qualified; the hash
        // is repeated in `RDNA3_P3_RELEASE.md` and the release receipts.
        assert_eq!(
            FullFp16Variant::P3.spirv().len(),
            23_280,
            "the production P3 module must be the qualified 23,280-byte artifact"
        );
        assert_eq!(
            sha256_hex(FullFp16Variant::P3.spirv()),
            "2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2",
            "the production P3 module must hash to the qualified Phase LXXIX SPIR-V"
        );
        assert_eq!(production.full_fp16_variant(), FullFp16Variant::P3);
        let identity = production
            .full_fp16_pipeline_identity()
            .expect("production P3 pipeline identity");
        assert!(
            identity.starts_with(&format!("{}/", FullFp16Variant::P3.shader_name())),
            "the production identity must name the P3 module: {identity}"
        );
        assert!(
            identity.contains("/len:23280"),
            "the production identity must describe the qualified module: {identity}"
        );
        assert!(
            !identity.contains("/experimental/"),
            "the production identity must not be a qualification route: {identity}"
        );

        let (rows, columns, reduction) = (64usize, 96usize, 128usize);
        let a = deterministic_matrix(rows * reduction, 0x5EED_0101);
        let b = deterministic_matrix(reduction * columns, 0x5EED_0102);
        let layout = GemmLayout {
            transposed_a: false,
            transposed_b: true,
        };
        let output = production.matmul(
            WmmaGemmArm::FullFp16Accumulate,
            &a,
            &b,
            rows,
            columns,
            reduction,
            layout,
        )?;
        assert_eq!(output.len(), rows * columns);
        assert!(output.iter().all(|value| value.is_finite()));

        let split8 = WmmaGemm::new_with_full_fp16_variant(
            &device,
            WmmaMode::FullFp16,
            FullFp16Variant::Split8,
        )?;
        assert_eq!(split8.full_fp16_variant(), FullFp16Variant::Split8);
        let split8_identity = split8
            .full_fp16_pipeline_identity()
            .expect("split8 fallback pipeline identity");
        assert!(
            split8_identity.starts_with(&format!("{}/", FullFp16Variant::Split8.shader_name())),
            "the fallback must still resolve to its own module: {split8_identity}"
        );
        assert_ne!(identity, split8_identity);
        Ok(())
    }

    #[test]
    fn full_fp16_variants_ship_distinct_spirv_modules() {
        assert_eq!(FullFp16Variant::Split8.spirv(), WMMA_GEMM_F16ACC_SPV);
        assert_eq!(
            FullFp16Variant::TwoBank.spirv(),
            WMMA_GEMM_F16ACC_TWOBANK_SPV
        );
        assert_eq!(
            FullFp16Variant::TwoWave.spirv(),
            WMMA_GEMM_F16ACC_TWOWAVE_SPV
        );
        assert_eq!(FullFp16Variant::P3.spirv(), WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV);
        // Every shipped variant must resolve to its own module, so no two
        // variants can share a pipeline cache entry.
        for (index, left) in FullFp16Variant::ALL.iter().enumerate() {
            for right in &FullFp16Variant::ALL[index + 1..] {
                assert_ne!(
                    left.spirv(),
                    right.spirv(),
                    "{} and {} must not resolve to the same SPIR-V",
                    left.label(),
                    right.label()
                );
                assert_ne!(left.shader_name(), right.shader_name());
            }
        }
    }

    /// The selector must change the pipeline that actually runs, not just the
    /// label, and both variants must remain bit-identical to each other.
    #[test]
    fn full_fp16_variant_selects_the_shader_it_dispatches() -> Result<()> {
        let Some(device) = device_or_skip(WmmaMode::FullFp16) else {
            return Ok(());
        };
        let split8 = WmmaGemm::new_with_full_fp16_variant(
            &device,
            WmmaMode::FullFp16,
            FullFp16Variant::Split8,
        )?;
        if !split8.capabilities().full_fp16_supported() {
            return Ok(());
        }
        let two_bank = WmmaGemm::new_with_full_fp16_variant(
            &device,
            WmmaMode::FullFp16,
            FullFp16Variant::TwoBank,
        )?;
        let two_wave = WmmaGemm::new_with_full_fp16_variant(
            &device,
            WmmaMode::FullFp16,
            FullFp16Variant::TwoWave,
        )?;

        assert_eq!(split8.full_fp16_variant(), FullFp16Variant::Split8);
        assert_eq!(two_bank.full_fp16_variant(), FullFp16Variant::TwoBank);
        assert_eq!(two_wave.full_fp16_variant(), FullFp16Variant::TwoWave);
        // The arm label is deliberately unchanged so every historical receipt
        // keyed on `wmma-full-f16acc` stays comparable.
        assert_eq!(WmmaGemmArm::FullFp16Accumulate.label(), "wmma-full-f16acc");

        let split8_identity = split8
            .full_fp16_pipeline_identity()
            .expect("split8 pipeline identity");
        let two_bank_identity = two_bank
            .full_fp16_pipeline_identity()
            .expect("two-bank pipeline identity");
        assert!(
            split8_identity.starts_with(&format!(
                "{}/",
                FullFp16Variant::Split8.shader_name()
            )),
            "split8 identity must name its own shader: {split8_identity}"
        );
        assert!(
            two_bank_identity.starts_with(&format!(
                "{}/",
                FullFp16Variant::TwoBank.shader_name()
            )),
            "two-bank identity must name its own shader: {two_bank_identity}"
        );
        assert_ne!(
            split8_identity, two_bank_identity,
            "the two variants must not share a pipeline identity"
        );
        let two_wave_identity = two_wave
            .full_fp16_pipeline_identity()
            .expect("two-wave pipeline identity");
        assert!(
            two_wave_identity.starts_with(&format!(
                "{}/",
                FullFp16Variant::TwoWave.shader_name()
            )),
            "two-wave identity must name its own shader: {two_wave_identity}"
        );
        assert_ne!(two_wave_identity, split8_identity);
        assert_ne!(two_wave_identity, two_bank_identity);

        let (rows, columns, reduction) = (64usize, 96usize, 128usize);
        let a = deterministic_matrix(rows * reduction, 0x5EED_0001);
        let b = deterministic_matrix(reduction * columns, 0x5EED_0002);
        let layout = GemmLayout {
            transposed_a: false,
            transposed_b: true,
        };
        let from_split8 = split8.matmul(
            WmmaGemmArm::FullFp16Accumulate,
            &a,
            &b,
            rows,
            columns,
            reduction,
            layout,
        )?;
        let from_two_bank = two_bank.matmul(
            WmmaGemmArm::FullFp16Accumulate,
            &a,
            &b,
            rows,
            columns,
            reduction,
            layout,
        )?;
        let from_two_wave = two_wave.matmul(
            WmmaGemmArm::FullFp16Accumulate,
            &a,
            &b,
            rows,
            columns,
            reduction,
            layout,
        )?;
        assert_eq!(from_split8.len(), from_two_bank.len());
        assert_eq!(from_split8.len(), from_two_wave.len());
        for (index, (left, right)) in from_split8.iter().zip(&from_two_bank).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "element {index} differs between the split8 and two-bank variants"
            );
        }
        for (index, (left, right)) in from_split8.iter().zip(&from_two_wave).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "element {index} differs between the split8 and two-wave variants"
            );
        }

        // Mixed routing must not consult the Full FP16 variant at all.
        let mixed = WmmaGemm::new_with_full_fp16_variant(
            &device,
            WmmaMode::Mixed,
            FullFp16Variant::TwoBank,
        )?;
        assert_eq!(
            mixed.select_arm(64, 64, 64),
            Some(WmmaGemmArm::MixedFp32Accumulate),
            "a Full FP16 variant must not change Mixed dispatch"
        );
        Ok(())
    }

    #[test]
    fn feature_flags_cannot_admit_a_non_rdna3_device() {
        let tile = WmmaTile { m:16, n:16, k:16, a_is_f16:true, b_is_f16:true,
            c_is_f16:false, c_is_f32:true, subgroup_scope:true, saturating:false };
        let mut caps = VulkanWmmaCapabilities {
            rdna3_or_newer:true, probed:true, enabled:true,
            extension_exposed:true, feature_supported:true, memory_model_supported:true,
            subgroup_size_control_supported:true, storage_16bit:true, shader_float16:true,
            mixed_tile:Some(tile), full_fp16_tile:Some(WmmaTile { c_is_f16:true, c_is_f32:false, ..tile }),
            ..Default::default()
        };
        assert!(caps.mixed_supported() && caps.full_fp16_supported());
        caps.rdna3_or_newer = false;
        for mode in [WmmaMode::Mixed, WmmaMode::FullFp16] {
            assert!(!caps.supports_mode(mode));
            assert!(caps.require(mode).unwrap_err().to_string().contains("RDNA3 or newer"));
        }
        assert!(caps.supports_mode(WmmaMode::Off));
        assert!(caps.require(WmmaMode::Off).is_ok());
    }

    #[test]
    fn off_device_cannot_report_an_unprobed_wmma_mode_as_supported() {
        let caps = VulkanWmmaCapabilities { rdna3_or_newer:true, ..Default::default() };
        assert!(!caps.mixed_supported() && !caps.full_fp16_supported());
        assert!(caps.unavailable_reason(WmmaMode::Mixed).contains("not probed"));
        assert!(caps.supports_mode(WmmaMode::Off));
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

    /// The performance gate must match the shapes `wmma_bench` measured, and it
    /// may only narrow the tile contract, never widen it.
    #[test]
    fn profitable_shape_policy_matches_measured_crossover() {
        for (rows, columns, reduction) in [
            (16, 64, 64),
            (16, 320, 64),
            (16, 64, 256),
            (32, 320, 64),
            (64, 320, 64),
            (64, 256, 256),
            (128, 256, 256),
            (128, 320, 64),
        ] {
            assert!(
                !wmma_shape_profitable(rows, columns, reduction),
                "{rows}x{columns}x{reduction} measured slower in cooperative WMMA and must keep FP32"
            );
        }
        for (rows, columns, reduction) in [
            (256, 768, 256),
            (128, 768, 768),
            (256, 512, 512),
            (512, 768, 768),
            (2048, 768, 2048),
            (128, 8192, 768),
        ] {
            assert!(
                wmma_shape_profitable(rows, columns, reduction),
                "{rows}x{columns}x{reduction} measured faster in cooperative WMMA"
            );
        }
        // A large reduction and column count cannot rescue an unusable row count,
        // and the profitability gate never widens the tile contract.
        for (rows, columns, reduction) in [(1, 8192, 8192), (8, 4096, 4096), (1024, 1024, 8)] {
            assert!(!wmma_shape_profitable(rows, columns, reduction));
        }
        assert!(WMMA_MIN_PROFITABLE_MACS > 128 * 256 * 256);
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
        // Shapes are chosen to cover staging boundaries, not just sizes. The
        // all-odd fixture forces every operand onto the masked 16-bit fallback
        // because an odd leading dimension makes a pair base odd. `64x48x19`
        // keeps A's transposed leading dimension even (64) while the reduction
        // axis is odd, so the final K slice packs words whose second half is out
        // of range. `19x48x64` keeps A's row-major leading dimension even (64)
        // while only 19 output rows exist, so the last row tile packs a pair at
        // the matrix edge. Both must return the same masked values the scalar
        // path returned, with nothing read past the logical matrix.
        for (rows, columns, reduction) in [
            (64usize, 48usize, 40usize),
            (33, 17, 19),
            (128, 128, 96),
            (64, 48, 19),
            (19, 48, 64),
        ]
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
    ///
    /// The matched oracle models the order the current module issues: eight
    /// FP16 accumulator banks over consecutive K slices, merged by a pairwise
    /// FP16 tree. The bound is the unchanged two-ULP budget.
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
                OracleAccumulation::Fp16Split8,
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

    /// The long-reduction geometry whose single-accumulator form failed the
    /// unchanged FP32 budget must now meet it in every operand layout.
    ///
    /// `128x128x96` is the fixture whose three transposed cases previously
    /// measured 0.0631-0.0731 against the unchanged `FULL_VS_FP32_MAX_ABS` of
    /// 0.05. This is an accuracy gate, so it is checked against the f64
    /// reference rather than against any accumulation model.
    #[test]
    fn wmma_full_fp16_meets_fp32_budget_on_long_reductions() -> Result<()> {
        let Some(device) = device_or_skip(WmmaMode::FullFp16) else {
            return Ok(());
        };
        let gemm = WmmaGemm::new(&device, WmmaMode::FullFp16)?;
        if !gemm.capabilities().full_fp16_supported() {
            return Ok(());
        }
        let (rows, columns, reduction) = (128usize, 128usize, 96usize);
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
            let a = deterministic_matrix(rows * reduction, 0xA101_0001);
            let b = deterministic_matrix(reduction * columns, 0xB202_0002);
            let reference =
                reference_fp32_gemm(&a, &b, rows, columns, reduction, layout)?;
            let observed = gemm.matmul(
                WmmaGemmArm::FullFp16Accumulate,
                &a,
                &b,
                rows,
                columns,
                reduction,
                layout,
            )?;
            let measured = drift(&reference, &observed)?;
            assert!(measured.all_finite());
            assert!(
                measured.max_abs <= FULL_VS_FP32_MAX_ABS,
                "full-FP16 WMMA {}x{}x{} {:?} vs FP32 max_abs {} > {}",
                rows,
                columns,
                reduction,
                layout,
                measured.max_abs,
                FULL_VS_FP32_MAX_ABS
            );
            assert!(
                measured.rms <= FULL_VS_FP32_RMS,
                "full-FP16 WMMA {}x{}x{} {:?} vs FP32 rms {} > {}",
                rows,
                columns,
                reduction,
                layout,
                measured.rms,
                FULL_VS_FP32_RMS
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

    /// Bytes a bound operand needs for this geometry, as the selector checks it.
    fn capacities(geometry: StagingGeometry) -> StagingGeometry {
        let a_len = if geometry.layout.transposed_a {
            geometry.lda() * geometry.reduction
        } else {
            geometry.lda() * geometry.rows
        };
        let b_len = if geometry.layout.transposed_b {
            geometry.ldb() * geometry.columns
        } else {
            geometry.ldb() * geometry.reduction
        };
        StagingGeometry {
            a_capacity_bytes: a_len * 2,
            b_capacity_bytes: b_len * 2,
            ..geometry
        }
    }

    fn geometry(
        rows: usize,
        columns: usize,
        reduction: usize,
        layout: GemmLayout,
    ) -> StagingGeometry {
        capacities(StagingGeometry {
            rows,
            columns,
            reduction,
            layout,
            a_capacity_bytes: 0,
            b_capacity_bytes: 0,
        })
    }

    /// Every protected production geometry, in every layout the graph builds,
    /// is inside the C1 contract.
    #[test]
    fn batched_staging_admits_every_protected_geometry() {
        let layouts = [
            GemmLayout {
                transposed_a: false,
                transposed_b: false,
            },
            GemmLayout {
                transposed_a: false,
                transposed_b: true,
            },
            GemmLayout {
                transposed_a: true,
                transposed_b: false,
            },
            GemmLayout {
                transposed_a: true,
                transposed_b: true,
            },
        ];
        for (rows, columns, reduction) in [
            (2048usize, 768usize, 2048usize),
            (512, 2048, 768),
            (512, 768, 768),
            (128, 8192, 768),
        ] {
            for layout in layouts {
                let geometry = geometry(rows, columns, reduction, layout);
                assert!(
                    batched_staging_eligible(&geometry),
                    "{rows}x{columns}x{reduction} {layout:?} must be C1-eligible"
                );
                // The two stride parity conditions are structural for these
                // layouts, so state them explicitly.
                assert_eq!(geometry.lda() % 2, 0);
                assert_eq!(geometry.ldb() % 2, 0);
            }
        }
    }

    /// Every geometry the contract cannot prove keeps the original P3 module.
    ///
    /// The odd-K and partial-tile cases are the ones Phase LXXXVII measured as
    /// regressions or inconclusive, so they are fallbacks by contract, not by
    /// benchmark.
    #[test]
    fn batched_staging_rejects_every_unproven_geometry() {
        let row_major = GemmLayout::default();
        let cases: [(usize, usize, usize, &str); 8] = [
            (512, 768, 767, "odd reduction: lda = k = 767 is odd"),
            (511, 768, 768, "rows not a multiple of 32"),
            (512, 767, 768, "columns not a multiple of 32"),
            (16, 128, 32, "rows below one 32-row tile row"),
            (512, 768, 760, "reduction not a multiple of 16"),
            (0, 768, 768, "empty geometry"),
            (512, 0, 768, "empty geometry"),
            (512, 768, 0, "empty geometry"),
        ];
        for (rows, columns, reduction, reason) in cases {
            let geometry = geometry(rows, columns, reduction, row_major);
            assert!(
                !batched_staging_eligible(&geometry),
                "{rows}x{columns}x{reduction} must fall back: {reason}"
            );
        }

        // An odd stride that no tile-divisibility rule can produce: a custom
        // layout where B is `[k, n]` with an odd column count is rejected on the
        // column rule, and one where the reduction is odd is rejected on both.
        assert!(!batched_staging_eligible(&geometry(
            512,
            768,
            767,
            GemmLayout {
                transposed_a: true,
                transposed_b: false,
            }
        )));

        // A buffer one element short of the geometry's own operand is not the
        // operand the kernel would read.
        let exact = geometry(512, 768, 768, row_major);
        assert!(!batched_staging_eligible(&StagingGeometry {
            a_capacity_bytes: exact.a_capacity_bytes - 2,
            ..exact
        }));
        assert!(!batched_staging_eligible(&StagingGeometry {
            b_capacity_bytes: exact.b_capacity_bytes - 2,
            ..exact
        }));

        // 32-bit index arithmetic: a geometry whose element index cannot fit in
        // the shader's `uint` arithmetic is a fallback even though every tile is
        // full and every stride is even.
        let huge = StagingGeometry {
            rows: 2_000_000_000,
            columns: 32,
            reduction: 32,
            layout: row_major,
            a_capacity_bytes: usize::MAX / 2,
            b_capacity_bytes: 32 * 32 * 2,
        };
        assert!(!batched_staging_eligible(&huge));
    }

    /// The two modules the production P3 variant can dispatch are exactly the
    /// qualified P3 contract and the C1 staging module, by byte identity.
    #[test]
    fn production_staging_modules_are_the_qualified_bytes() {
        assert_eq!(
            sha256_hex(FullFp16Variant::P3.spirv()),
            "2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2"
        );
        assert_eq!(FullFp16Variant::P3.spirv().len(), 23_280);
        let staged = FullFp16Variant::P3
            .staged_spirv()
            .expect("the production variant has a batched-staging module");
        assert_eq!(
            sha256_hex(staged),
            "b072242034ba7bd0f53690c38c03b11ed587b538480c04ed13ecf19fe2a8b27d"
        );
        assert_eq!(staged.len(), 36_912);
        assert_eq!(
            FullFp16Variant::P3.staged_shader_name(),
            Some("wmma_gemm_f16_f16acc_twowave_p3_c1")
        );
        // The production pipeline identity still names the P3 contract module, so
        // the v0.2.5 release gate's identity is unchanged; the staging identity is
        // reported separately.
        assert_eq!(
            FullFp16Variant::P3.shader_name(),
            "wmma_gemm_f16_f16acc_twowave_p3"
        );
        // No other committed variant may reach the C1 module.
        for variant in [
            FullFp16Variant::Split8,
            FullFp16Variant::TwoBank,
            FullFp16Variant::TwoWave,
        ] {
            assert!(variant.staged_spirv().is_none(), "{variant:?}");
            assert!(variant.staged_shader_name().is_none(), "{variant:?}");
        }
    }

    /// The two production routes must both be reachable from the real selector.
    #[test]
    fn production_selector_reaches_both_staging_modules() -> Result<()> {
        let Ok(device) = VulkanDevice::new_for_wmma(WmmaMode::FullFp16) else {
            return Ok(());
        };
        let Ok(gemm) = WmmaGemm::new(&device, WmmaMode::FullFp16) else {
            return Ok(());
        };
        let identity = gemm
            .full_fp16_staging_identity()
            .expect("a capable device builds the C1 kernel");
        assert!(identity.starts_with("wmma_gemm_f16_f16acc_twowave_p3_c1/"), "{identity}");
        assert!(identity.ends_with("/len:36912"), "{identity}");
        // Both routes are recorded separately; nothing has run yet.
        assert_eq!(gemm.staging_counts(), WmmaStagingCounts::default());

        let rows = 64usize;
        let columns = 64usize;
        let reduction = 64usize;
        let a = deterministic_matrix(rows * reduction, 7);
        let b = deterministic_matrix(columns * reduction, 9);
        let aligned = gemm.matmul(
            WmmaGemmArm::FullFp16Accumulate,
            &a,
            &b,
            rows,
            columns,
            reduction,
            GemmLayout::default(),
        )?;
        let batched = gemm.staging_counts();
        assert_eq!(batched.batched, 1, "aligned geometry must take C1");
        assert_eq!(batched.masked, 0);

        // The same size with an odd reduction cannot use C1; the output must be
        // finite and the masked counter must move instead.
        let odd_reduction = 63usize;
        let a_odd = deterministic_matrix(rows * odd_reduction, 11);
        let b_odd = deterministic_matrix(columns * odd_reduction, 13);
        let fallback = gemm.matmul(
            WmmaGemmArm::FullFp16Accumulate,
            &a_odd,
            &b_odd,
            rows,
            columns,
            odd_reduction,
            GemmLayout::default(),
        )?;
        assert!(aligned.iter().all(|value| value.is_finite()));
        assert!(fallback.iter().all(|value| value.is_finite()));
        let after = gemm.staging_counts();
        assert_eq!(after.batched, 1, "odd reduction must not take C1");
        assert_eq!(after.masked, 1);
        Ok(())
    }
}
