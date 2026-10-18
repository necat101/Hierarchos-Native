//! Vendor-aware Vulkan kernel selection.
//!
//! Hierarchos ships one portable SPIR-V module per kernel, which is what every
//! backend uses by default. Vendor silicon and drivers still differ enough that
//! a single memory/compute schedule is not optimal everywhere:
//!
//! * AMD (RDNA/CU/KHR-SPIR-V lowering) is the historic primary development
//!   target (wave64, 64 KiB LDS, large workgroups).
//! * Intel Gen9 integrated parts (HD Graphics 520-class) are bandwidth-starved
//!   with 32 KiB of shared memory, `SIMD8/16` execution, and a different image
//!   compiler lowering path.
//!
//! This module keeps those choices separate from mainline code. Call sites keep
//! their existing dispatch contract and only ask for the vendor-aware kernel
//! slot for their family:
//!
//! ```ignore
//! let kernel = vendor::VendorMatmulKernel::new(&device, VendorKernelFamily::LinearForward)?;
//! kernel.record_dispatch(&mut batch, &[&x, &w, &out], push, grid)?;
//! ```
//!
//! [`select_kernel`] resolves the vendor from `VkPhysicalDeviceProperties`
//! (PCI vendor ID, never the marketing name), applies capability gates, and
//! falls back to the exact portable module when anything is missing. Matmul
//! geometry is only known when a dispatch is recorded, so
//! [`VendorMatmulKernel`] keeps both the portable and the vendor pipeline and
//! picks per dispatch from [`intel_matmul_geometry_supports`]; every other
//! family still resolves once. Every selected variant preserves the host-side
//! dispatch contract and the per-element FP32 math, so switching vendors cannot
//! change results:
//!
//! * [`VendorKernelFamily::workgroup_elements`] is the number of data elements
//!   one workgroup covers, so existing `div_ceil(len, 256)` style grids stay
//!   valid for both the portable and the Intel modules.
//! * Intel variants are bit-identical to the portable kernels; the regression
//!   tests below assert that on the executing device, not just on paper.
//!
//! Two environment variables exist for qualification work:
//!
//! * `HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS` forces the portable modules.
//! * `HIERARCHOS_VULKAN_FORCE_VENDOR` overrides classification (for A/B runs on
//!   one physical adapter); accepted values are the [`GpuVendor::label`]s plus
//!   `portable`.

use anyhow::Result;

use crate::vulkan::{BindingAccess, ComputeBatch, ComputeKernel};
use crate::{GpuBuffer, VulkanDevice};

/// Force every kernel family back to its portable SPIR-V module.
pub const HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS_ENV: &str =
    "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS";
/// Override vendor classification for A/B qualification on one adapter.
pub const HIERARCHOS_VULKAN_FORCE_VENDOR_ENV: &str = "HIERARCHOS_VULKAN_FORCE_VENDOR";
/// Qualification override: run the packed-FP16 LM adjoint kernels even on the
/// Intel Gen9 parts where [`native_fp16_lm_compute_reliable`] measures a driver
/// fault, so a newer driver can be re-qualified with the existing tests.
pub const HIERARCHOS_VULKAN_FORCE_NATIVE_FP16_LM_COMPUTE_ENV: &str =
    "HIERARCHOS_VULKAN_FORCE_NATIVE_FP16_LM_COMPUTE";

pub const PCI_VENDOR_AMD: u32 = 0x1002;
pub const PCI_VENDOR_INTEL: u32 = 0x8086;
pub const PCI_VENDOR_NVIDIA: u32 = 0x10DE;
pub const PCI_VENDOR_ARM: u32 = 0x13B5;
pub const PCI_VENDOR_QUALCOMM: u32 = 0x5143;
pub const PCI_VENDOR_IMAGINATION: u32 = 0x1010;
pub const PCI_VENDOR_MICROSOFT: u32 = 0x1414;
pub const PCI_VENDOR_APPLE: u32 = 0x106B;

/// Hardware vendor of a Vulkan physical device.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GpuVendor {
    Amd,
    Intel,
    Nvidia,
    Arm,
    Qualcomm,
    Imagination,
    Microsoft,
    Apple,
    Other,
}

impl GpuVendor {
    /// Classify a raw PCI vendor ID. Unknown IDs stay `Other` so a new vendor
    /// can never be mistaken for one with tuned kernels.
    pub const fn from_pci_vendor_id(vendor_id: u32) -> Self {
        match vendor_id {
            PCI_VENDOR_AMD => Self::Amd,
            PCI_VENDOR_INTEL => Self::Intel,
            PCI_VENDOR_NVIDIA => Self::Nvidia,
            PCI_VENDOR_ARM => Self::Arm,
            PCI_VENDOR_QUALCOMM => Self::Qualcomm,
            PCI_VENDOR_IMAGINATION => Self::Imagination,
            PCI_VENDOR_MICROSOFT => Self::Microsoft,
            PCI_VENDOR_APPLE => Self::Apple,
            _ => Self::Other,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Amd => "amd",
            Self::Intel => "intel",
            Self::Nvidia => "nvidia",
            Self::Arm => "arm",
            Self::Qualcomm => "qualcomm",
            Self::Imagination => "imagination",
            Self::Microsoft => "microsoft",
            Self::Apple => "apple",
            Self::Other => "other",
        }
    }

    /// Parse a [`Self::label`] (or the literal `portable`) from an environment
    /// override. `portable` resolves to `Other`, which selects portable modules.
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "amd" => Some(Self::Amd),
            "intel" => Some(Self::Intel),
            "nvidia" => Some(Self::Nvidia),
            "arm" => Some(Self::Arm),
            "qualcomm" => Some(Self::Qualcomm),
            "imagination" => Some(Self::Imagination),
            "microsoft" => Some(Self::Microsoft),
            "apple" => Some(Self::Apple),
            "other" | "portable" => Some(Self::Other),
            _ => None,
        }
    }

    pub const fn is_intel(self) -> bool {
        matches!(self, Self::Intel)
    }
}

/// Kernel families that have vendor-specific variants.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VendorKernelFamily {
    /// `out[row, col] = sum_k x[row, k] * w[col, k]`
    LinearForward,
    /// [`Self::LinearForward`] plus `bias[col]`
    LinearBiasForward,
    /// [`Self::LinearForward`] plus `residual[row, col]`
    LinearResidualForward,
    /// Three same-shaped row-major projections (RWKV time-mix r/k/v) in one
    /// dispatch, each with its own serial `fma` chain.
    Linear3Forward,
    /// `out[i] = silu(x[i])`
    SiluForward,
    /// `grad_input[i] = grad_output[i] * silu'(x[i])`
    SiluBackward,
    /// Decoupled AdamW elementwise update.
    AdamW,
}

impl VendorKernelFamily {
    pub const ALL: [Self; 7] = [
        Self::LinearForward,
        Self::LinearBiasForward,
        Self::LinearResidualForward,
        Self::Linear3Forward,
        Self::SiluForward,
        Self::SiluBackward,
        Self::AdamW,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::LinearForward => "linear-forward",
            Self::LinearBiasForward => "linear-bias-forward",
            Self::LinearResidualForward => "linear-residual-forward",
            Self::Linear3Forward => "linear3-forward",
            Self::SiluForward => "silu-forward",
            Self::SiluBackward => "silu-backward",
            Self::AdamW => "adamw",
        }
    }

    /// True for the families whose module is a `(rows, input_dim, output_dim)`
    /// matmul slot: they share the 16x16 workgroup, the 12-byte geometry push
    /// block, and the per-dispatch geometry gate, so they are the families
    /// [`VendorMatmulKernel`] can hold.
    pub const fn is_matmul(self) -> bool {
        matches!(
            self,
            Self::LinearForward
                | Self::LinearBiasForward
                | Self::LinearResidualForward
                | Self::Linear3Forward
        )
    }

    /// The portable module every vendor falls back to. These are the exact
    /// bytes the pre-vendor-tuning backend embedded.
    pub const fn portable_spirv(self) -> &'static [u8] {
        match self {
            Self::LinearForward => include_bytes!("../shaders/linear_forward.spv"),
            Self::LinearBiasForward => include_bytes!("../shaders/linear_bias_forward.spv"),
            Self::LinearResidualForward => {
                include_bytes!("../shaders/linear_residual_forward.spv")
            }
            Self::Linear3Forward => include_bytes!("../shaders/linear3_forward.spv"),
            Self::SiluForward => include_bytes!("../shaders/silu_forward.spv"),
            Self::SiluBackward => include_bytes!("../shaders/silu_backward.spv"),
            Self::AdamW => include_bytes!("../shaders/adamw.spv"),
        }
    }

    pub const fn binding_count(self) -> usize {
        match self {
            Self::LinearForward => 3,
            Self::LinearBiasForward | Self::LinearResidualForward | Self::AdamW => 4,
            Self::Linear3Forward => 9,
            Self::SiluForward => 2,
            Self::SiluBackward => 3,
        }
    }

    pub const fn push_constant_bytes(self) -> u32 {
        match self {
            Self::LinearForward
            | Self::LinearBiasForward
            | Self::LinearResidualForward
            | Self::Linear3Forward => 12,
            Self::SiluForward | Self::SiluBackward => 4,
            Self::AdamW => 28,
        }
    }

    /// Number of data elements one workgroup covers. Every host call site
    /// computes its dispatch grid from this width, so a variant that changes its
    /// internal per-invocation layout must not change this value.
    pub const fn workgroup_elements(self) -> u32 {
        match self {
            Self::LinearForward
            | Self::LinearBiasForward
            | Self::LinearResidualForward
            | Self::Linear3Forward => 16 * 16,
            Self::SiluForward | Self::SiluBackward | Self::AdamW => 256,
        }
    }

    /// True when this family has a measured Intel Gen9 module. Families without
    /// one keep the portable module on every vendor, because their candidates
    /// were measured on the HD 520 and rejected (see
    /// `shaders/vendor_experiments/` and `VENDOR_TUNING.md`):
    ///
    /// * `silu-forward` / `silu-backward` measured 0.78x / 0.85x.
    /// * `adamw` was within timing noise but not bit-exact (1 ulp at element 64).
    /// * the transposed-weight parameter matmuls measured 0.79x at 8x448x512 and
    ///   0.98x at 6x96x320: that portable addressing is already coalesced, so
    ///   the extra workgroup-memory staging buys nothing.
    pub const fn has_intel_variant(self) -> bool {
        matches!(
            self,
            Self::LinearForward
                | Self::LinearBiasForward
                | Self::LinearResidualForward
                | Self::Linear3Forward
        )
    }

    /// Minimum `maxComputeSharedMemorySize` the Intel variant needs, or `None`
    /// when the family has no workgroup-memory variant.
    ///
    /// The three single-projection families carry the 32-wide K stage ported
    /// from the portable modules: a 16-row input tile with a 33-float row stride
    /// and a 32-wide weight tile with a 17-float row stride. `linear3-forward`
    /// still stages six 16x17 tiles. `spirv_workgroup_bytes` decodes these
    /// totals back out of the committed modules in
    /// `committed_vendor_modules_declare_their_capability_contract`, so the
    /// constant cannot drift from the artifacts it gates.
    pub const fn intel_shared_memory_bytes(self) -> Option<u32> {
        match self {
            Self::LinearForward | Self::LinearBiasForward | Self::LinearResidualForward => {
                Some(16 * 33 * 4 + 32 * 17 * 4)
            }
            Self::Linear3Forward => Some(6 * 16 * 17 * 4),
            Self::SiluForward | Self::SiluBackward | Self::AdamW => None,
        }
    }

    /// Workgroup dimensions the Intel variant's SPIR-V declares, used for
    /// capability gating before the module is handed to
    /// `vkCreateComputePipelines`.
    pub const fn intel_local_size(self) -> [u32; 3] {
        [16, 16, 1]
    }

    pub const fn intel_variant_label(self) -> &'static str {
        "intel-gen9-lds"
    }

    /// The pre-tuning scalar schedule of one matmul family, vendored unchanged
    /// from the initial publish of this crate.
    ///
    /// The portable modules stage a k-slice in workgroup memory and fall back to
    /// this exact dot-product schedule below eight rows. That fallback is not
    /// free: the shared-memory declaration is part of the module, so a
    /// single-row dispatch pays it even while it runs the scalar path, and on the
    /// Gen9 target that measured 0.81x against the pre-tuning module at
    /// 1x448x512. Dispatching the scalar module itself for those shapes recovers
    /// the pre-tuning number, and it is bit-identical by construction -- it *is*
    /// the module those kernels were recorded against.
    ///
    /// The artifacts live in `shaders/portable_scalar/` rather than `shaders/`
    /// because they are byte-identical to the pre-tuning portable modules, and
    /// `build.rs` refuses two artifacts with one SPIR-V signature (the registry it
    /// generates maps signature to name). The nested directory is outside that
    /// registry on purpose; the parity test compares these bytes against the
    /// portable modules on the device instead.
    const fn scalar_spirv(self) -> Option<&'static [u8]> {
        match self {
            Self::LinearForward => Some(include_bytes!(
                "../shaders/portable_scalar/linear_forward.spv"
            )),
            Self::LinearBiasForward => Some(include_bytes!(
                "../shaders/portable_scalar/linear_bias_forward.spv"
            )),
            Self::LinearResidualForward => Some(include_bytes!(
                "../shaders/portable_scalar/linear_residual_forward.spv"
            )),
            Self::Linear3Forward => Some(include_bytes!(
                "../shaders/portable_scalar/linear3_forward.spv"
            )),
            Self::SiluForward | Self::SiluBackward | Self::AdamW => None,
        }
    }

    /// True when this family has a measured AMD RDNA3 module. Every matmul
    /// family has one, including `linear3-forward`, because the RDNA3 stage
    /// covers all six of that family's operands in the 64 KiB LDS budget.
    pub const fn has_amd_variant(self) -> bool {
        matches!(
            self,
            Self::LinearForward
                | Self::LinearBiasForward
                | Self::LinearResidualForward
                | Self::Linear3Forward
        )
    }

    /// Workgroup storage the RDNA3 module declares.
    ///
    /// The three single-projection families stage a 64-wide K slice: a 16-row
    /// input tile with a 65-float row stride and a 64-wide weight tile with a
    /// 17-float row stride, so `16 * 65 * 4 + 64 * 17 * 4 = 8512` bytes.
    /// `linear3-forward` stages six 16x65 tiles, `6 * 16 * 65 * 4 = 24960`.
    /// Both fit inside RDNA3's 64 KiB LDS; `spirv_workgroup_bytes` decodes the
    /// totals back out of the committed modules, so these cannot drift from the
    /// artifacts they gate.
    pub const fn amd_shared_memory_bytes(self) -> Option<u32> {
        match self {
            Self::LinearForward | Self::LinearBiasForward | Self::LinearResidualForward => {
                Some(16 * 65 * 4 + 64 * 17 * 4)
            }
            Self::Linear3Forward => Some(6 * 16 * 65 * 4),
            Self::SiluForward | Self::SiluBackward | Self::AdamW => None,
        }
    }

    /// Workgroup dimensions the RDNA3 module declares; the host grid contract is
    /// the same 16x16 element tile as the portable module.
    pub const fn amd_local_size(self) -> [u32; 3] {
        [16, 16, 1]
    }

    pub const fn amd_variant_label(self) -> &'static str {
        "amd-rdna3-lds64"
    }

    const fn amd_spirv(self) -> Option<&'static [u8]> {
        match self {
            Self::LinearForward => Some(include_bytes!("../shaders/linear_forward_amd_rdna3.spv")),
            Self::LinearBiasForward => Some(include_bytes!(
                "../shaders/linear_bias_forward_amd_rdna3.spv"
            )),
            Self::LinearResidualForward => Some(include_bytes!(
                "../shaders/linear_residual_forward_amd_rdna3.spv"
            )),
            Self::Linear3Forward => {
                Some(include_bytes!("../shaders/linear3_forward_amd_rdna3.spv"))
            }
            Self::SiluForward | Self::SiluBackward | Self::AdamW => None,
        }
    }

    const fn intel_spirv(self) -> Option<&'static [u8]> {
        match self {
            Self::LinearForward => Some(include_bytes!("../shaders/linear_forward_intel_gen9.spv")),
            Self::LinearBiasForward => Some(include_bytes!(
                "../shaders/linear_bias_forward_intel_gen9.spv"
            )),
            Self::LinearResidualForward => Some(include_bytes!(
                "../shaders/linear_residual_forward_intel_gen9.spv"
            )),
            Self::Linear3Forward => {
                Some(include_bytes!("../shaders/linear3_forward_intel_gen9.spv"))
            }
            Self::SiluForward | Self::SiluBackward | Self::AdamW => None,
        }
    }
}

/// Geometry gate for the Intel Gen9 tiled matmul variants, in units of the
/// `(rows, input_dim, output_dim)` push constants every linear-family call site
/// already passes. The variants stage k-slices in workgroup memory, which pays
/// for itself only when a dispatch has enough row lanes, enough k to amortize
/// the two barriers per 16-wide slice, and enough output tiles to fill the GPU.
/// Every bound below is a measured crossover on the HD Graphics 520 target
/// (hardware, not a simulator); `VENDOR_TUNING.md` records the full matrix.
///
/// Measured shape verdicts:
///
/// | rows | k | n | Intel/portable |
/// |------|---|-----|----------------|
/// | 1 | 448 | 512 | 0.61x |
/// | 2 | 448 | 512 | 1.12x (below the row bar) |
/// | 4 | 64 | 96 | 0.92x |
/// | 4 | 64 | 128 | 0.99x |
/// | 4 | 32 | 128 | 0.87x |
/// | 4 | 32 | 256 | 1.14x |
/// | 4 | 32 | 512 | 1.44x |
/// | 4 | 64 | 192 | 1.17x |
/// | 6 | 96 | 320 | 2.18x |
/// | 8 | 16 | 512 | 0.57x |
/// | 8 | 32 | 512 | 2.23x |
/// | 8 | 64 | 128 | 1.39x |
/// | 8 | 448 | 512 | 3.16x |
/// | 16 | 448 | 2048 | 6.43x |
///
/// The row bar excludes single-row decode, which loses hard (0.61x) because a
/// 16-row tile can never fill its lanes; the output-tile bar is higher below
/// eight rows because half-empty workgroups need more tiles to keep the GPU
/// busy.
pub const INTEL_MATMUL_MIN_ROWS: u32 = 4;
/// Shortest reduction dimension that amortizes the two barriers per 16-wide
/// k-slice. `shaders/vendor_experiments/` records the rejected short-k module
/// that tried to solve this in-shader instead (0.70x at k=8, 0.57x at k=16).
pub const INTEL_MATMUL_MIN_INPUT_DIM: u32 = 32;
/// Fewest output columns for dispatches with at least eight rows.
pub const INTEL_MATMUL_MIN_OUTPUT_DIM: u32 = 128;
/// Fewest output columns for dispatches with four to seven rows.
pub const INTEL_MATMUL_MIN_OUTPUT_DIM_LOW_ROWS: u32 = 192;

/// Rows below which a matmul dispatch uses the scalar schedule instead of the
/// tiled portable module. This is the portable modules' own scalar-vs-tiled
/// threshold, so the two decisions cannot disagree about which schedule a shape
/// was measured under; a vendor variant admitted for the geometry still wins
/// first, because its gate is the measured crossover for that part.
pub const SCALAR_MATMUL_MAX_ROWS: u32 = 8;

/// Pure geometry gate shared by per-dispatch selection and its regression
/// tests. `false` always means "use the portable module".
pub const fn intel_matmul_geometry_supports(rows: u32, input_dim: u32, output_dim: u32) -> bool {
    if rows < INTEL_MATMUL_MIN_ROWS || input_dim < INTEL_MATMUL_MIN_INPUT_DIM {
        return false;
    }
    if rows >= 8 {
        output_dim >= INTEL_MATMUL_MIN_OUTPUT_DIM
    } else {
        output_dim >= INTEL_MATMUL_MIN_OUTPUT_DIM_LOW_ROWS
    }
}

/// One resolved kernel: the family, the vendor decision that applied, and the
/// SPIR-V module the caller should create its pipeline from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VendorKernelSelection {
    pub family: VendorKernelFamily,
    /// Vendor classification that produced this selection (after any override).
    pub vendor: GpuVendor,
    /// Stable label for logs, tests, and qualification reports.
    pub variant: &'static str,
    pub spirv: &'static [u8],
    /// False for the portable module shared by every vendor.
    pub vendor_specific: bool,
    /// True when the variant is only used for dispatches that pass
    /// [`intel_matmul_geometry_supports`]; the plan therefore lists the
    /// capability, not a per-dispatch promise.
    pub geometry_gated: bool,
}

/// Resolve the vendor used for kernel selection, honoring the qualification
/// override. Unknown override values fall back to the device-reported vendor.
pub fn effective_vendor(device: &VulkanDevice) -> GpuVendor {
    match std::env::var(HIERARCHOS_VULKAN_FORCE_VENDOR_ENV) {
        Ok(raw) => GpuVendor::from_label(&raw)
            .unwrap_or_else(|| GpuVendor::from_pci_vendor_id(device.vendor_id())),
        Err(_) => GpuVendor::from_pci_vendor_id(device.vendor_id()),
    }
}

fn vendor_kernels_disabled() -> bool {
    std::env::var_os(HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS_ENV).is_some()
}

/// Pure Intel Gen9/Gen9.5 PCI device-ID test: the one Gen9 classification the
/// crate publishes, shared by the FP16 LM reliability policy below and the
/// watchdog submission-slicing policy ported onto this part class.
///
/// The ID table itself lives in `vulkan.rs`, because the shader harnesses
/// compile that module standalone and cannot resolve `crate::vendor`; this is
/// the vendor layer's view of the same list, and
/// `vulkan::gen9_pci_table_is_the_one_the_vendor_layer_publishes` asserts the
/// two agree across every range boundary.
pub const fn is_intel_gen9_pci_id(vendor_id: u32, device_id: u32) -> bool {
    crate::vulkan::is_intel_gen9_pci_id(vendor_id, device_id)
}

/// Whether the opt-in native-FP16-compute LM backward tranche is usable on this
/// device.
///
/// Measured on the target Intel HD Graphics 520 (`0x8086:0x1916`, Windows driver
/// 31.0.101.2115): the packed-FP16 streaming LM adjoint kernels
/// (`cross_entropy_linear_{row_stats,input_grad}_streaming_fp16_packed`) fault
/// the GPU (`VK_ERROR_DEVICE_LOST`) at the production width 448 whenever the
/// `Fp16StorageFp16LmBackward` policy is selected, while the same modules pass at
/// width 5 and every FP32-compute arm passes. The byte-identical modules are
/// green on the AMD target, so this is a Gen9 driver/compiler fault, not a
/// kernel defect: the backend therefore keeps the FP32-compute/FP16-storage arm
/// on these parts instead of hanging a training run. Re-qualify with a newer
/// driver before lifting the restriction.
pub fn native_fp16_lm_compute_reliable(device: &VulkanDevice) -> bool {
    if std::env::var_os(HIERARCHOS_VULKAN_FORCE_NATIVE_FP16_LM_COMPUTE_ENV).is_some() {
        return true;
    }
    native_fp16_lm_compute_reliable_for(device.vendor_id(), device.device_id())
}

/// Device-independent form of [`native_fp16_lm_compute_reliable`], so the
/// policy is asserted by regression tests without a physical adapter.
pub const fn native_fp16_lm_compute_reliable_for(vendor_id: u32, device_id: u32) -> bool {
    !is_intel_gen9_pci_id(vendor_id, device_id)
}

// The packed-FP16 rows16 cross-row LM arms (`Fp16CeTapeRows16*`) are admitted by
// capability alone. An earlier Gen9 reliability policy withheld them after a
// `VK_ERROR_DEVICE_LOST` in
// `native_fp16_lm_input_grad_reuse_arms_match_packed_at_width_448`. Bisection
// (`lm_rows16_batch_bisect_microprofile`) traced that fault to the regression
// creating compute pipelines between recorded dispatches of an open command
// batch, not to the kernels: every rows16 module passes in isolation and in
// production-shaped compositions, production creates every kernel before
// recording, and the regression now hoists pipeline creation. Every rows16
// variant measures green on the Gen9 target (see VENDOR_TUNING.md, "Gen9
// rows16 driver quirk (resolved)").

/// PCI device IDs of the AMD RDNA3 parts this layer tunes for: Navi 31
/// (`gfx1100`), Navi 32 (`gfx1101`), Navi 33 (`gfx1102`) and the RDNA3
/// integrated parts Phoenix / Hawk Point (`gfx1103`). From the PCI ID database
/// and the AMDGPU device list, not from the marketing name.
///
/// Vulkan exposes no architecture-generation property, so an ID table is the
/// only reliable test; an ID outside it keeps the portable module, which is the
/// safe direction for a part this layer has not been tuned for.
pub const AMD_RDNA3_PCI_IDS: &[u32] = &[
    // Navi 31 / gfx1100
    0x7448, 0x7449, 0x744A, 0x744B, 0x744C, 0x745E, // Navi 32 / gfx1101
    0x7460, 0x7461, 0x7470, 0x747E, // Navi 33 / gfx1102
    0x7480, 0x7481, 0x7483, 0x7487, 0x7489, 0x748B, 0x7499, 0x749F,
    // Phoenix / Hawk Point integrated / gfx1103
    0x15BF, 0x15C8, 0x1900, 0x1901,
];

/// Pure RDNA3 classification: AMD vendor ID plus one of the tuned device IDs.
pub const fn is_amd_rdna3_pci_id(vendor_id: u32, device_id: u32) -> bool {
    if vendor_id != PCI_VENDOR_AMD {
        return false;
    }
    let mut index = 0;
    while index < AMD_RDNA3_PCI_IDS.len() {
        if AMD_RDNA3_PCI_IDS[index] == device_id {
            return true;
        }
        index += 1;
    }
    false
}

/// Whether this adapter is an RDNA3 part, i.e. whether the RDNA3 modules may be
/// selected. Classification is by PCI ID, never by the reported device name.
pub fn is_amd_rdna3(device: &VulkanDevice) -> bool {
    is_amd_rdna3_pci_id(device.vendor_id(), device.device_id())
}

/// Geometry gate for the RDNA3 matmul modules.
///
/// Deliberately the same structural threshold the portable modules use to pick
/// their own 32-wide tiled schedule over the scalar path: below it the portable
/// module is a scalar dot product, which is the measured right answer for
/// single-row decode, short reduction dimensions and thin output tiles (see
/// [`intel_matmul_geometry_supports`] for the measured crossover on the other
/// tuned part). No RDNA3-specific crossover has been measured -- there is no
/// RDNA3 adapter on the machine this layer was tuned on -- so the gate claims
/// only what is structural: use the tuned module exactly where the portable
/// module would itself tile. Every admitted RDNA3 module is compared bit-exact
/// with the portable one on the executing adapter.
pub const fn amd_matmul_geometry_supports(rows: u32, input_dim: u32, output_dim: u32) -> bool {
    rows >= 8 && input_dim >= 64 && output_dim >= 16
}

/// Capability gate for the RDNA3 variant of one family on one device.
fn amd_variant_supported(device: &VulkanDevice, family: VendorKernelFamily) -> bool {
    if family.amd_spirv().is_none() {
        return false;
    }
    if !device.supports_storage_buffer_bindings(family.binding_count() as u32) {
        return false;
    }
    if !device.supports_compute_work_group_size(family.amd_local_size()) {
        return false;
    }
    match family.amd_shared_memory_bytes() {
        Some(bytes) => device.max_compute_shared_memory_bytes() >= bytes,
        None => true,
    }
}

/// Capability gate for the tuned module of a family on this device, whichever
/// vendor's module the selection resolved to.
fn variant_supported(device: &VulkanDevice, family: VendorKernelFamily) -> bool {
    if family.intel_spirv().is_some()
        && effective_vendor(device).is_intel()
        && intel_variant_supported(device, family)
    {
        return true;
    }
    effective_vendor(device) == GpuVendor::Amd
        && is_amd_rdna3(device)
        && amd_variant_supported(device, family)
}

/// Capability gate for the Intel variant of one family on one device.
fn intel_variant_supported(device: &VulkanDevice, family: VendorKernelFamily) -> bool {
    let Some(_) = family.intel_spirv() else {
        return false;
    };
    if !device.supports_storage_buffer_bindings(family.binding_count() as u32) {
        return false;
    }
    if !device.supports_compute_work_group_size(family.intel_local_size()) {
        return false;
    }
    match family.intel_shared_memory_bytes() {
        Some(bytes) => device.max_compute_shared_memory_bytes() >= bytes,
        None => true,
    }
}

/// Choose the module for one kernel family on one physical device.
///
/// This is the *capability* decision: the vendor classification and the device
/// limits, with no geometry input. Matmul call sites run the variant through
/// [`VendorMatmulKernel`], which additionally applies the per-dispatch geometry
/// gate; [`vendor_kernel_plan`] prints this capability view.
///
/// The decision is deliberately pure: no measuring, no caching, and no state
/// mutation, so it is safe to call during every graph construction and cheap to
/// assert in regression tests.
pub fn select_kernel(device: &VulkanDevice, family: VendorKernelFamily) -> VendorKernelSelection {
    let vendor = effective_vendor(device);
    let portable = VendorKernelSelection {
        family,
        vendor,
        variant: "portable",
        spirv: family.portable_spirv(),
        vendor_specific: false,
        geometry_gated: false,
    };
    if vendor_kernels_disabled() {
        return portable;
    }
    if vendor.is_intel() {
        if !intel_variant_supported(device, family) {
            return portable;
        }
        let Some(spirv) = family.intel_spirv() else {
            return portable;
        };
        return VendorKernelSelection {
            family,
            vendor,
            variant: family.intel_variant_label(),
            spirv,
            vendor_specific: true,
            // Only the matmul families carry a variant, and their variant is used
            // only when the dispatch geometry clears the measured crossover.
            geometry_gated: family.is_matmul(),
        };
    }
    // AMD only leaves the portable module on an RDNA3 part: every other AMD
    // generation, and every non-AMD vendor, keeps the exact pre-tuning bytes.
    if vendor == GpuVendor::Amd && is_amd_rdna3(device) && amd_variant_supported(device, family) {
        if let Some(spirv) = family.amd_spirv() {
            return VendorKernelSelection {
                family,
                vendor,
                variant: family.amd_variant_label(),
                spirv,
                vendor_specific: true,
                geometry_gated: family.is_matmul(),
            };
        }
    }
    portable
}

/// Diagnostic view of the kernel plan for one device. The device-listing CLI and
/// qualification runs print this to make the vendor decision auditable without
/// reading the source.
pub fn vendor_kernel_plan(device: &VulkanDevice) -> Vec<VendorKernelSelection> {
    VendorKernelFamily::ALL
        .into_iter()
        .map(|family| select_kernel(device, family))
        .collect()
}

/// A matmul kernel slot whose concrete SPIR-V module is chosen per dispatch.
///
/// Every call site learns the true `(rows, input_dim, output_dim)` of a matmul
/// only when it records the dispatch, and the Intel Gen9 modules win only on
/// production-shaped work: single-row decode, short reduction dimensions, and
/// thin output tiles all measured slower than the portable module (see
/// [`intel_matmul_geometry_supports`]). The wrapper therefore owns the portable
/// pipeline always, plus one Intel pipeline when the capability gates admit it,
/// and picks between them from the push constants' leading three words — the
/// `(rows, input_dim, output_dim)` layout every linear-family push struct in
/// this crate uses. Host grids, bindings, and results are unchanged.
pub(crate) struct VendorMatmulKernel {
    /// Read by the dispatch counter that proves the vendor path ran, not just
    /// that the plan selected it. Always compiled: the logits probe asserts the
    /// counter at runtime, so a release build needs the family too.
    family: VendorKernelFamily,
    portable: ComputeKernel,
    /// The tuned pipeline and the per-dispatch geometry gate it is admitted by.
    /// The gate is chosen by the vendor the selection resolved to, so one slot
    /// can hold either the Intel or the RDNA3 module without the call sites
    /// knowing which.
    variant: Option<(ComputeKernel, fn(u32, u32, u32) -> bool)>,
    /// The scalar schedule the portable modules fall back to, dispatched directly
    /// below [`SCALAR_MATMUL_MAX_ROWS`] so a small dispatch does not carry the
    /// tiled module's workgroup-memory reservation.
    scalar: Option<ComputeKernel>,
}

impl VendorMatmulKernel {
    /// Build the slot for `family`, which must be a matmul family.
    pub(crate) fn new(device: &VulkanDevice, family: VendorKernelFamily) -> Result<Self> {
        Self::build(device, family, None, family.push_constant_bytes())
    }

    /// [`Self::new`] with explicit binding-access annotations for call sites
    /// whose kernels are verified against a stricter buffer-access contract.
    pub(crate) fn new_with_access(
        device: &VulkanDevice,
        family: VendorKernelFamily,
        binding_accesses: &[BindingAccess],
    ) -> Result<Self> {
        Self::build(
            device,
            family,
            Some(binding_accesses),
            family.push_constant_bytes(),
        )
    }

    /// [`Self::new`] for call sites whose push block is wider than the family
    /// minimum, because their struct carries extra fields after the three
    /// geometry words (for example the shared head trainer's loss parameters).
    /// Both the portable and the vendor module read only the geometry words, so
    /// the wider declared range is valid for either pipeline.
    pub(crate) fn new_with_push_constant_bytes(
        device: &VulkanDevice,
        family: VendorKernelFamily,
        push_constant_bytes: u32,
    ) -> Result<Self> {
        Self::build(device, family, None, push_constant_bytes)
    }

    fn build(
        device: &VulkanDevice,
        family: VendorKernelFamily,
        binding_accesses: Option<&[BindingAccess]>,
        push_constant_bytes: u32,
    ) -> Result<Self> {
        debug_assert!(
            family.is_matmul(),
            "{} is not a matmul family",
            family.label()
        );
        let create = |spirv: &'static [u8]| -> Result<ComputeKernel> {
            match binding_accesses {
                Some(accesses) => {
                    ComputeKernel::new_with_access(device, spirv, accesses, push_constant_bytes)
                }
                None => {
                    ComputeKernel::new(device, spirv, family.binding_count(), push_constant_bytes)
                }
            }
        };
        let portable = create(family.portable_spirv())?;
        let selection = select_kernel(device, family);
        let variant = if selection.vendor_specific {
            let gate: fn(u32, u32, u32) -> bool = if selection.vendor.is_intel() {
                intel_matmul_geometry_supports
            } else {
                amd_matmul_geometry_supports
            };
            Some((create(selection.spirv)?, gate))
        } else {
            None
        };
        let scalar = match family.scalar_spirv() {
            Some(spirv) => Some(create(spirv)?),
            None => None,
        };
        Ok(Self {
            family,
            portable,
            variant,
            scalar,
        })
    }

    /// True when this slot holds a vendor module for qualifying dispatches.
    #[cfg(test)]
    pub(crate) fn variant_available(&self) -> bool {
        self.variant.is_some()
    }

    /// The portable pipeline this slot always owns. Tests that compare kernel
    /// layouts or drive the portable module directly use this instead of
    /// re-creating the pipeline.
    #[cfg(test)]
    pub(crate) fn portable_kernel(&self) -> &ComputeKernel {
        &self.portable
    }

    /// The module [`Self::record_dispatch`] would use for this geometry. Call
    /// sites outside the vendor layer use this as a prediction gate: the fused
    /// RWKV time-mix projection yields to the composed `linear3-forward` module
    /// exactly where this returns `true` (see
    /// `RwkvTimeMixCoreOp::prefer_composed_projection`), so the two decisions
    /// cannot disagree about whether a tuned module would run.
    pub(crate) fn selects_variant_for(&self, rows: u32, input_dim: u32, output_dim: u32) -> bool {
        self.variant
            .as_ref()
            .is_some_and(|(_, gate)| gate(rows, input_dim, output_dim))
    }

    /// True when these rows are below the scalar threshold. A vendor module
    /// admitted for the same dispatch still wins, so this is the fallback order's
    /// second rung, not a promise that the scalar module runs.
    #[cfg(test)]
    pub(crate) fn selects_scalar_for(&self, rows: u32) -> bool {
        self.scalar.is_some() && rows < SCALAR_MATMUL_MAX_ROWS
    }

    /// True when this slot holds the scalar schedule.
    #[cfg(test)]
    pub(crate) fn scalar_available(&self) -> bool {
        self.scalar.is_some()
    }

    /// Dispatch through whichever module the geometry selects.
    pub(crate) fn record_dispatch(
        &self,
        batch: &mut ComputeBatch,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        let geometry = push_geometry(push_constants);
        let kernel = match &self.variant {
            Some((variant, gate))
                if geometry.is_some_and(|(rows, input_dim, output_dim)| {
                    gate(rows, input_dim, output_dim)
                }) =>
            {
                record_variant_dispatch(self.family);
                variant
            }
            // The scalar fallback is the portable modules' own below-eight-row
            // schedule, dispatched directly: the tiled module would run the same
            // arithmetic while still reserving its workgroup memory.
            _ if geometry.is_some_and(|(rows, _, _)| rows < SCALAR_MATMUL_MAX_ROWS)
                && self.scalar.is_some() =>
            {
                self.scalar.as_ref().expect("checked above")
            }
            _ => &self.portable,
        };
        kernel.record_dispatch(batch, buffers, push_constants, groups)
    }
}

/// Environment switch that disables the attention-backward watchdog budget.
pub const HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET_ENV: &str =
    "HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET";

/// Serial work one attention-backward invocation performs.
///
/// Attention backward launches one invocation per (batch row, KV head) and each
/// invocation walks the whole query and key range for the query heads mapped to
/// its KV head, so its inner loops are proportional to this product. Strict
/// adjoint modes (`unfused_qk` 1, 2 and 9) measured 1.44x the cost of the same
/// geometry in mode 0, which is folded in here.
pub const fn attention_backward_work(
    queries_per_kv: u64,
    head_dim: u64,
    query_seq_len: u64,
    key_value_seq_len: u64,
    strict: bool,
) -> u64 {
    let base = queries_per_kv
        .saturating_mul(head_dim)
        .saturating_mul(query_seq_len)
        .saturating_mul(key_value_seq_len);
    if strict {
        base.saturating_mul(144) / 100
    } else {
        base
    }
}

/// Work units one dispatch admits on the adapter class the budget was measured on.
///
/// Calibrated on Intel HD Graphics 520 (Gen9), 2026-10-15, one invocation per
/// dispatch, where the driver's watchdog interval is about three seconds: mode 0
/// passes seq 256 (4.19e6 units, 2576 ms) and loses the device at seq 320
/// (6.55e6), while strict mode passes seq 240 (5.31e6, 2973 ms) and loses it at
/// seq 248 (5.67e6). The budget sits below the first loss and about a tenth below
/// the last passing point, so what it refuses is what runs within a tenth of the
/// watchdog.
pub const ATTENTION_BACKWARD_WORK_BUDGET: u64 = 5_000_000;

/// Shared-memory size of the adapter class the budget was measured on.
const ATTENTION_BACKWARD_BUDGET_CLASS_SHARED_BYTES: u32 = 32 * 1024;

/// The attention-backward work budget for `device`, or `None` when none applies.
///
/// `None` means "refuse nothing". Only the class the budget was measured on is
/// bounded, because a bound derived from the slowest measured adapter would reject
/// sequences that run in well under a second on a discrete GPU -- a functional
/// regression rather than a safety net. Extend the class list when a boundary is
/// measured on another adapter.
pub fn attention_backward_work_budget(device: &VulkanDevice) -> Option<u64> {
    if std::env::var(HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET_ENV).is_ok() {
        return None;
    }
    let measured_class = GpuVendor::from_pci_vendor_id(device.vendor_id()).is_intel()
        && device.max_compute_shared_memory_bytes() <= ATTENTION_BACKWARD_BUDGET_CLASS_SHARED_BYTES;
    measured_class.then_some(ATTENTION_BACKWARD_WORK_BUDGET)
}

/// Most key-range blocks one attention-backward dispatch is split into.
///
/// A split turns a dispatch that would reset the device into several sequential
/// submissions, each covering its own slice of the key range. The cap stops a caller
/// from turning one unusable geometry into a long sequence of tiny submissions for no
/// gain; beyond it the dispatch is refused as before. It also bounds how wide the
/// rescue reaches: at 257 queries and head dim 8 the measured loss at key span 16
/// needs six blocks, and a geometry needing more than eight is still refused.
pub const ATTENTION_BACKWARD_MAX_KEY_BLOCKS: u32 = 8;

/// Rows of row statistics the strict adjoint caches invocation-privately.
///
/// The backward module keeps a row's softmax statistics while `(batch rows) *
/// queries per KV head * query length` fits here and streams them above it, so this
/// is also the boundary between the two adjoints a split has to respect.
pub const ATTENTION_BACKWARD_STATS_CACHE_SIZE: u64 = 512;

/// True when the strict adjoint would keep its row statistics in the invocation's
/// private cache instead of streaming them, mirroring the module's own test.
pub const fn attention_backward_statistics_are_cached(
    queries_per_kv: u64,
    query_seq_len: u64,
    batch_rows: u64,
) -> bool {
    batch_rows
        .saturating_mul(queries_per_kv)
        .saturating_mul(query_seq_len)
        <= ATTENTION_BACKWARD_STATS_CACHE_SIZE
}

/// Streamed-adjoint work units for `key_span` outer keys of a `key_total`-long range.
///
/// The streamed strict adjoint has no shared row statistics to reuse: it replays the
/// softmax for every outer key, and it does that inside the head-dimension loop, so
/// its cost carries the full key span and a second head-dimension factor. A key-range
/// block bounds only the outer key loop, which is why the full span stays in the
/// model and a split divides the total by the block count rather than by its square.
pub const fn attention_backward_streamed_work(
    queries_per_kv: u64,
    head_dim: u64,
    query_seq_len: u64,
    key_span: u64,
    key_total: u64,
) -> u64 {
    queries_per_kv
        .saturating_mul(head_dim)
        .saturating_mul(head_dim)
        .saturating_mul(query_seq_len)
        .saturating_mul(key_span)
        .saturating_mul(key_total)
}

/// Work units the streamed strict adjoint admits on the measured adapter class.
///
/// This is a different model from the cached budget above, so it is a different number
/// for the same adapter, and it is bracketed on both sides by measurement. On Intel HD
/// Graphics 520 (Gen9), at 257 queries, head dim 8 and 2 queries per KV head, the
/// streamed adjoint completed key spans 1, 4 and 8 in 61 ms, 395 ms and 1394 ms, and
/// lost the device at key span 16, so its boundary lies between 2.11e6 and 8.42e6 of
/// these units and the budget below is a tenth under the last span measured to
/// complete. Head dim 32 is out of reach at every span driven -- 320 x 320 and 512 x
/// 512 both lost the device -- which this model reproduces.
///
/// A first version of this bound was anchored on the losing side alone and came out
/// about a thousand times too generous, admitting shapes that reset the device.
/// Read by the tests that pin the derived percentage above to this measured
/// number, so it is a test-build oracle rather than a value production reads.
#[cfg(test)]
pub const ATTENTION_BACKWARD_STREAMED_WORK_BUDGET: u64 = 1_900_000;

/// The streamed bound as a percentage of the cached budget of the same class.
///
/// The two models count different work, so the streamed bound scales the budget the
/// caller derived from its device (5.0e6 cached -> 1.9e6 streamed) instead of
/// replacing it with a constant, which would ignore the caller's device gating and
/// bound every adapter as if it were this one.
pub const ATTENTION_BACKWARD_STREAMED_BUDGET_PERCENT: u64 = 38;

/// Sequential key-range blocks this geometry is dispatched in.
///
/// Returns 1 whenever splitting would not help or would not be safe:
///
/// * The fused path (`unfused_qk` 0 or 3) is never split. Its `grad_q`, `grad_k`
///   and `grad_v` all come from the shared primary pass, whose `softmax_dot` is a
///   running prefix feeding `ds` across the whole key range, so blocking keys would
///   change the reduction order. The shader ignores the split grid for that path,
///   and this keeps the caller from paying for duplicate full-range blocks.
/// * A device whose boundary has not been measured (`None`) is never split, so its
///   dispatch is exactly what it was before the split existed.
/// * A geometry that already fits its budget is not split; the extra dispatches
///   would only add overhead.
///
/// The count is derived from the largest slice rather than from the total, so every
/// block's own work -- not just the average -- lands under `budget`.
pub const fn attention_backward_key_blocks(
    budget: Option<u64>,
    queries_per_kv: u64,
    head_dim: u64,
    query_seq_len: u64,
    key_value_seq_len: u64,
    batch_rows: u64,
    strict: bool,
) -> u32 {
    let budget = match budget {
        Some(budget) => budget,
        None => return 1,
    };
    if !strict || budget == 0 || key_value_seq_len == 0 {
        return 1;
    }
    // A split cannot use the adjoint's cached row statistics, which are private to
    // the invocation that computed them, so it takes the streaming adjoint instead.
    // That one is far more expensive per key than the cached path -- measured at
    // 2365 ms against 22.3 ms at 1024 rows and head dim 257 -- so splitting a
    // geometry whose statistics *are* cacheable would make it slower, not safer:
    // strict seq 248 took the device out at 4 blocks where it had been refused
    // cleanly. Only geometries that already stream, where the blocks divide the work
    // the unsplit dispatch would have done, are split.
    if attention_backward_statistics_are_cached(queries_per_kv, query_seq_len, batch_rows) {
        return 1;
    }
    // A streamed geometry is split only as far as the measured boundary allows, and it
    // is split at all only because each block is its own submission: a grid partition
    // inside one submission leaves that submission's duration unchanged, and measured
    // at 257 queries and head dim 8 one dispatch and eight key-range blocks of the same
    // span both lost the device. That is why the caller submits the blocks one at a
    // time instead of sizing a grid.
    //
    // The streamed adjoint replays the whole key range for every outer key, so a
    // block's cost carries the full key length and its own span:
    //
    //     block work = queries per KV head * head dim^2 * queries * block keys * keys
    //
    // The keys one block may own is therefore the streamed bound divided by that
    // per-key cost, and the block count follows from the range. When a single key
    // already costs more than the bound, no block count can bring a submission inside
    // it: at 257 queries and head dim 8 the whole sequence is such a shape, one key
    // costing 8.45e6 units against a 1.9e6 bound, so it keeps its refusal.
    let limit = budget.saturating_mul(ATTENTION_BACKWARD_STREAMED_BUDGET_PERCENT) / 100;
    let per_key = attention_backward_streamed_work(
        queries_per_kv,
        head_dim,
        query_seq_len,
        1,
        key_value_seq_len,
    );
    if per_key == 0 {
        return 1;
    }
    let max_keys = limit / per_key;
    if max_keys == 0 {
        return 1;
    }
    let blocks = key_value_seq_len.saturating_add(max_keys - 1) / max_keys;
    if blocks == 0 {
        return 1;
    }
    let cap = ATTENTION_BACKWARD_MAX_KEY_BLOCKS as u64;
    (if blocks > cap { cap } else { blocks }) as u32
}

/// The refusal message for an attention-backward geometry that still exceeds
/// `budget` once split into `key_blocks` sequential dispatches, or `None` when it
/// fits.
///
/// Split out from the dispatch site so the refusal itself can be driven by a test:
/// the only other way to reach it is to dispatch a geometry that resets the
/// device, which is what it exists to prevent. `key_blocks` is the count the site is
/// about to dispatch, so the bound applies to the widest block rather than to the
/// whole geometry.
pub fn attention_backward_budget_error(
    budget: u64,
    queries_per_kv: u64,
    head_dim: u64,
    query_seq_len: u64,
    key_value_seq_len: u64,
    batch_rows: u64,
    strict: bool,
    key_blocks: u32,
) -> Option<String> {
    let blocks = key_blocks.max(1) as u64;
    // The shader partitions the key range as [b * len / blocks, (b + 1) * len /
    // blocks), so the widest block is the ceiling of that division. A block bounds
    // the outer key loop only, so the streamed model keeps the whole span in its cost.
    let max_block_keys = (key_value_seq_len + blocks - 1) / blocks;
    let streams_statistics = strict
        && !attention_backward_statistics_are_cached(queries_per_kv, query_seq_len, batch_rows);
    let (work, limit) = if streams_statistics {
        (
            // Each block is its own submission, so the bound applies to the widest block
            // rather than to one submission holding every block.
            attention_backward_streamed_work(
                queries_per_kv,
                head_dim,
                query_seq_len,
                max_block_keys,
                key_value_seq_len,
            ),
            budget.saturating_mul(ATTENTION_BACKWARD_STREAMED_BUDGET_PERCENT) / 100,
        )
    } else {
        (
            attention_backward_work(
                queries_per_kv,
                head_dim,
                query_seq_len,
                max_block_keys,
                strict,
            ),
            budget,
        )
    };
    (work > limit).then(|| {
        if blocks > 1 {
            format!(
                "attention backward for {query_seq_len} queries x {key_value_seq_len} keys at head dim {head_dim} (strict {strict}) needs {blocks} key-range blocks and still leaves {work} work units in its widest block, past the {limit} this adapter resets the device at; shorten the sequence, or set {HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET_ENV} to dispatch anyway"
            )
        } else if streams_statistics
            && attention_backward_streamed_work(
                queries_per_kv,
                head_dim,
                query_seq_len,
                1,
                key_value_seq_len,
            ) > limit
        {
            // The useful diagnosis for a shape a split cannot rescue: one key of it is
            // already a full replay of the key range, so blocking keys buys nothing.
            format!(
                "attention backward for {query_seq_len} queries x {key_value_seq_len} keys at head dim {head_dim} (strict {strict}) streams its row statistics, and a single one of its keys already costs {per_key} work units against the {limit} this adapter resets the device above, because the streaming adjoint replays the whole key range for every key -- no key-range split can bring a submission inside that bound; shorten the sequence, or set {HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET_ENV} to dispatch anyway",
                per_key = attention_backward_streamed_work(
                    queries_per_kv,
                    head_dim,
                    query_seq_len,
                    1,
                    key_value_seq_len,
                ),
            )
        } else {
            format!(
                "attention backward for {query_seq_len} queries x {key_value_seq_len} keys at head dim {head_dim} (strict {strict}) needs {work} work units and this adapter resets the device above {limit}; shorten the sequence, or set {HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET_ENV} to dispatch anyway"
            )
        }
    })
}

const VARIANT_DISPATCH_SLOTS: usize = 9;

thread_local! {
    /// Dispatches routed to a vendor variant on this thread, per family, so a
    /// device test or a probe binary can prove a model consumed the vendor
    /// module instead of silently falling back to the portable one - and that
    /// the *specific* families a graph is supposed to use were really
    /// exercised. Compiled into release too, because the logits probe asserts
    /// it at runtime: a one-word thread-local bump per admitted dispatch.
    static VARIANT_DISPATCH_COUNTS: std::cell::Cell<[u64; VARIANT_DISPATCH_SLOTS]> =
        const { std::cell::Cell::new([0; VARIANT_DISPATCH_SLOTS]) };
}

fn record_variant_dispatch(family: VendorKernelFamily) {
    let index = family as usize;
    VARIANT_DISPATCH_COUNTS.with(|counts| {
        let mut slots = counts.get();
        slots[index] += 1;
        counts.set(slots);
    });
}

/// Per-family vendor-variant dispatch counts for the current thread.
///
/// Only dispatches that cleared the family's geometry gate are counted, so a
/// non-zero entry is proof that the tuned module ran rather than that the plan
/// merely selected it. Probes print this when
/// `HIERARCHOS_VULKAN_VARIANT_TRACE` is set; see `transformer_logits`.
pub fn variant_dispatch_counts() -> Vec<(&'static str, u64)> {
    VARIANT_DISPATCH_COUNTS.with(|counts| {
        let slots = counts.get();
        VendorKernelFamily::ALL
            .iter()
            .map(|family| (family.label(), slots[*family as usize]))
            .collect()
    })
}

/// Test-only read of the per-thread Intel dispatch count across all families.
#[cfg(test)]
pub(crate) fn variant_dispatch_count() -> u64 {
    VARIANT_DISPATCH_COUNTS.with(|counts| counts.get().iter().sum())
}

/// Test-only read of one family's per-thread Intel dispatch count.
#[cfg(test)]
pub(crate) fn variant_dispatch_count_for(family: VendorKernelFamily) -> u64 {
    VARIANT_DISPATCH_COUNTS.with(|counts| counts.get()[family as usize])
}

/// Test-only reset of the per-thread Intel dispatch counts.
#[cfg(test)]
pub(crate) fn reset_variant_dispatch_count() {
    VARIANT_DISPATCH_COUNTS.with(|counts| counts.set([0; VARIANT_DISPATCH_SLOTS]));
}

/// Reads the leading `(rows, input_dim, output_dim)` words of a linear-family
/// push-constant block, as written by `bytemuck::bytes_of` on the host. A block
/// too short to hold them keeps the dispatch on the portable module.
fn push_geometry(push_constants: &[u8]) -> Option<(u32, u32, u32)> {
    let word = |start: usize| -> Option<u32> {
        Some(u32::from_ne_bytes(
            push_constants.get(start..start + 4)?.try_into().ok()?,
        ))
    };
    Some((word(0)?, word(4)?, word(8)?))
}

fn build_kernel(device: &VulkanDevice, family: VendorKernelFamily) -> Result<ComputeKernel> {
    let selection = select_kernel(device, family);
    // Re-check the gate here so a future selection change cannot hand an
    // unsupported module to the driver.
    if selection.vendor_specific && !variant_supported(device, family) {
        return ComputeKernel::new(
            device,
            family.portable_spirv(),
            family.binding_count(),
            family.push_constant_bytes(),
        );
    }
    ComputeKernel::new(
        device,
        selection.spirv,
        family.binding_count(),
        family.push_constant_bytes(),
    )
}

pub(crate) fn silu_forward_kernel(device: &VulkanDevice) -> Result<ComputeKernel> {
    build_kernel(device, VendorKernelFamily::SiluForward)
}

pub(crate) fn silu_backward_kernel(device: &VulkanDevice) -> Result<ComputeKernel> {
    build_kernel(device, VendorKernelFamily::SiluBackward)
}

pub(crate) fn adamw_kernel(device: &VulkanDevice) -> Result<ComputeKernel> {
    build_kernel(device, VendorKernelFamily::AdamW)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::ComputeBatch;
    use crate::GpuBuffer;
    use sha2::{Digest, Sha256};

    const WARMUP_DISPATCHES: usize = 3;
    const TIMED_DISPATCHES: usize = 20;
    const QUALIFICATION_DEVICE_INDEX_ENV: &str = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX";

    fn test_device() -> Result<Option<VulkanDevice>> {
        match std::env::var(QUALIFICATION_DEVICE_INDEX_ENV) {
            Ok(raw) => Ok(Some(VulkanDevice::new_with_index(raw.parse::<usize>()?)?)),
            Err(std::env::VarError::NotPresent) => Ok(VulkanDevice::new().ok()),
            Err(err) => Err(anyhow::anyhow!(
                "reading {QUALIFICATION_DEVICE_INDEX_ENV}: {err}"
            )),
        }
    }

    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn vendor_classification_uses_pci_ids_not_names() {
        assert_eq!(GpuVendor::from_pci_vendor_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_pci_vendor_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_pci_vendor_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_pci_vendor_id(0x13B5), GpuVendor::Arm);
        // An unknown vendor must never be classified as a tuned vendor.
        assert_eq!(GpuVendor::from_pci_vendor_id(0xDEAD), GpuVendor::Other);
        assert_eq!(GpuVendor::from_pci_vendor_id(0), GpuVendor::Other);
    }

    #[test]
    fn vendor_labels_round_trip_and_portable_alias_resolves_to_other() {
        for vendor in [
            GpuVendor::Amd,
            GpuVendor::Intel,
            GpuVendor::Nvidia,
            GpuVendor::Arm,
            GpuVendor::Qualcomm,
            GpuVendor::Imagination,
            GpuVendor::Microsoft,
            GpuVendor::Apple,
            GpuVendor::Other,
        ] {
            assert_eq!(GpuVendor::from_label(vendor.label()), Some(vendor));
        }
        assert_eq!(GpuVendor::from_label(" Intel "), Some(GpuVendor::Intel));
        assert_eq!(GpuVendor::from_label("portable"), Some(GpuVendor::Other));
        assert_eq!(GpuVendor::from_label("mystery-gpu"), None);
    }

    /// A variant may change how one workgroup computes, but not what it covers:
    /// every existing call site keeps its `div_ceil(len, 256)` grid.
    #[test]
    fn vendor_variants_preserve_the_host_dispatch_contract() {
        for family in VendorKernelFamily::ALL {
            assert_eq!(
                family.workgroup_elements(),
                256,
                "{} changed its elements-per-workgroup contract",
                family.label()
            );
            let intel_local: u32 = family.intel_local_size().iter().product();
            assert!(
                intel_local <= family.workgroup_elements(),
                "{} declares a workgroup larger than its coverage",
                family.label()
            );
            assert_eq!(
                family.workgroup_elements() % intel_local,
                0,
                "{} cannot tile its coverage with the Intel workgroup",
                family.label()
            );
        }
    }

    fn families_with_intel_variants() -> Vec<VendorKernelFamily> {
        VendorKernelFamily::ALL
            .into_iter()
            .filter(|family| family.has_intel_variant())
            .collect()
    }

    fn families_with_amd_variants() -> Vec<VendorKernelFamily> {
        VendorKernelFamily::ALL
            .into_iter()
            .filter(|family| family.has_amd_variant())
            .collect()
    }

    fn families_with_scalar_schedule() -> Vec<VendorKernelFamily> {
        VendorKernelFamily::ALL
            .into_iter()
            .filter(|family| family.scalar_spirv().is_some())
            .collect()
    }

    /// The scalar schedule is the pre-tuning module, so it has to agree with the
    /// portable module raw-bit at every geometry -- and it is the module a
    /// below-eight-row dispatch now runs, which is what recovers the 0.81x that
    /// the tiled module's workgroup-memory reservation cost on single-row decode.
    #[test]
    fn scalar_schedule_reproduces_the_portable_kernels() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        for family in families_with_scalar_schedule() {
            let module = family
                .scalar_spirv()
                .expect("a family with a scalar schedule has a module");
            for (rows, input_dim, output_dim) in linear_layout_geometries() {
                if family == VendorKernelFamily::Linear3Forward {
                    linear3_forward_case_with(&device, rows, input_dim, output_dim, module)?;
                } else {
                    linear_layout_case_with(&device, family, rows, input_dim, output_dim, module)?;
                }
            }
            let slot = VendorMatmulKernel::new(&device, family)?;
            assert!(
                slot.scalar_available(),
                "{} has no scalar pipeline",
                family.label()
            );
            assert!(slot.selects_scalar_for(1), "single-row decode is scalar");
            assert!(slot.selects_scalar_for(SCALAR_MATMUL_MAX_ROWS - 1));
            assert!(!slot.selects_scalar_for(SCALAR_MATMUL_MAX_ROWS));
        }
        Ok(())
    }

    /// The RDNA3 classification is an AMD PCI device-ID table, never a name.
    #[test]
    fn rdna3_classification_is_amd_pci_ids_only() {
        assert!(is_amd_rdna3_pci_id(PCI_VENDOR_AMD, 0x744C), "Navi 31");
        assert!(is_amd_rdna3_pci_id(PCI_VENDOR_AMD, 0x747E), "Navi 32");
        assert!(is_amd_rdna3_pci_id(PCI_VENDOR_AMD, 0x7480), "Navi 33");
        assert!(is_amd_rdna3_pci_id(PCI_VENDOR_AMD, 0x15BF), "Phoenix");
        // The vendor ID is part of the test: the same device ID on another
        // vendor is not an RDNA3 part.
        assert!(!is_amd_rdna3_pci_id(PCI_VENDOR_INTEL, 0x744C));
        // Earlier AMD generations keep the portable module.
        assert!(
            !is_amd_rdna3_pci_id(PCI_VENDOR_AMD, 0x731F),
            "Navi 10, RDNA1"
        );
        assert!(
            !is_amd_rdna3_pci_id(PCI_VENDOR_AMD, 0x73BF),
            "Navi 21, RDNA2"
        );
        assert!(!is_amd_rdna3_pci_id(PCI_VENDOR_AMD, 0x744D));
    }

    /// Every admitted RDNA3 module has to fit the 64 KiB LDS budget and declare
    /// the local size and workgroup footprint the capability gate publishes; the
    /// footprint is decoded from the artifact, not trusted from the constant.
    #[test]
    fn committed_rdna3_modules_declare_their_capability_contract() {
        const RDNA3_LDS_BYTES: u32 = 64 * 1024;
        for family in families_with_amd_variants() {
            let module = family
                .amd_spirv()
                .expect("a family with an RDNA3 variant has a module");
            assert_eq!(
                spirv_local_size(module),
                family.amd_local_size(),
                "{} RDNA3 module declares an unexpected local size",
                family.label()
            );
            let bytes = family
                .amd_shared_memory_bytes()
                .expect("an RDNA3 module always stages workgroup memory");
            assert!(
                bytes <= RDNA3_LDS_BYTES,
                "{} stages {bytes} bytes against the {RDNA3_LDS_BYTES}-byte RDNA3 LDS",
                family.label()
            );
            assert_eq!(
                spirv_workgroup_bytes(module),
                bytes,
                "{} RDNA3 module's declared workgroup footprint disagrees with the capability gate",
                family.label()
            );
            let (barriers, workgroup_variables) = spirv_workgroup_usage(module);
            assert!(
                workgroup_variables >= 2 && barriers >= 2,
                "{} RDNA3 module claims workgroup memory but declares {workgroup_variables} \
                 variables / {barriers} barriers",
                family.label()
            );
            // The 64-wide stage is the whole point of the module, so the
            // declared footprint has to be larger than the portable 32-wide one
            // could explain by padding alone.
            assert!(
                bytes > family.intel_shared_memory_bytes().unwrap_or(0),
                "{} RDNA3 stage is not wider than the 32-wide stage it replaces",
                family.label()
            );
        }
    }

    /// The RDNA3 gate is exactly the portable modules' own tiled-vs-scalar
    /// threshold, so the tuned module is used where the portable schedule would
    /// tile and never where it would fall back to a scalar dot product.
    #[test]
    fn rdna3_geometry_gate_matches_the_portable_tiling_threshold() {
        assert!(amd_matmul_geometry_supports(8, 64, 16));
        assert!(amd_matmul_geometry_supports(16, 448, 2048));
        assert!(amd_matmul_geometry_supports(32, 448, 2048));
        assert!(
            !amd_matmul_geometry_supports(1, 448, 512),
            "single-row decode"
        );
        assert!(!amd_matmul_geometry_supports(7, 448, 512));
        assert!(!amd_matmul_geometry_supports(8, 63, 512), "short k");
        assert!(!amd_matmul_geometry_supports(8, 64, 15), "thin n");
        // Every geometry the gate admits has to have its local size covered by
        // the 16x16 workgroup the module declares.
        for family in families_with_amd_variants() {
            assert_eq!(
                family.amd_local_size().iter().product::<u32>(),
                family.workgroup_elements()
            );
        }
    }

    #[test]
    fn portable_fallback_bytes_are_the_mainline_artifacts() {
        let tuned = families_with_intel_variants();
        assert!(
            !tuned.is_empty(),
            "at least one family must carry a measured vendor variant"
        );
        for family in VendorKernelFamily::ALL {
            assert_eq!(family.portable_spirv().len() % 4, 0);
            match family.intel_spirv() {
                // A distinct vendor module must exist for every tuned family, and
                // it must not be a copy of the portable one.
                Some(intel) => assert_ne!(
                    family.portable_spirv(),
                    intel,
                    "{} Intel variant is a byte-for-byte copy of the portable module",
                    family.label()
                ),
                // Families whose candidates were measured and rejected must not
                // silently regain a module without a new measurement.
                None => assert!(
                    !family.has_intel_variant(),
                    "{} claims an Intel variant without one",
                    family.label()
                ),
            }
        }
    }

    /// The scalar rung is the pre-tuning module for every family, so a dispatch
    /// below [`SCALAR_MATMUL_MAX_ROWS`] reproduces exactly what the publish commit
    /// shipped -- on every vendor, RDNA3 included. The retile that won the measured
    /// geometry replaced the *mainline* portable modules, so identity with the
    /// published bytes is asserted from the frozen copies in
    /// `shaders/portable_scalar/`: each digest below is that artifact's SHA-256 at
    /// the publish commit, and the source beside it rebuilds it byte-for-byte with
    /// glslc 2023.8 (`training_shader_manifest.json` pins both).
    #[test]
    fn scalar_fallbacks_are_the_published_modules() {
        let published = [
            (
                VendorKernelFamily::LinearForward,
                "74313348d543b2048dbf3bbfe62e895e8d43b9693bdfe4de5907e17a39f1af29",
            ),
            (
                VendorKernelFamily::LinearBiasForward,
                "332972cee5de8929e0eef1a8640c801d2cd04ebc1ddcab570ada2cdfdbf8dbad",
            ),
            (
                VendorKernelFamily::LinearResidualForward,
                "daf250cb5c167e1a1cd94911400ce8a6d9eed2e09f24842d03bbdb25d352d6de",
            ),
            (
                VendorKernelFamily::Linear3Forward,
                "d18345aab759196e8365deb08b85012501ef0ab03ce3002172007acc4bc285bc",
            ),
        ];
        for (family, expected) in published {
            let bytes = family
                .scalar_spirv()
                .unwrap_or_else(|| panic!("{} has no scalar rung", family.label()));
            assert_eq!(
                format!("{:x}", Sha256::digest(bytes)),
                expected,
                "{} scalar fallback is no longer the published module",
                family.label()
            );
        }
    }

    /// The per-dispatch geometry gate is what keeps every measured below-parity
    /// shape on the portable streaming module while production matmuls keep the
    /// measured 1.4x-6.4x Intel wins. Each row is a shape measured on the HD
    /// Graphics 520 target; the comment records the measured ratio.
    #[test]
    fn intel_matmul_geometry_gate_matches_the_measured_matrix() {
        // (rows, input_dim, output_dim, intel_is_faster)
        let measured = [
            (1, 448, 512, false),  // 0.61x: single-row decode starves the 16-row tile
            (2, 448, 512, false),  // 1.12x, below the row bar
            (2, 128, 512, false),  // 1.08x, below the row bar
            (2, 32, 512, false),   // 0.97x
            (4, 32, 128, false),   // 0.87x: too few output tiles below eight rows
            (4, 64, 96, false),    // 0.92x
            (4, 64, 128, false),   // 0.99x
            (5, 64, 96, false),    // 0.91x
            (4, 32, 256, true),    // 1.14x
            (4, 64, 192, true),    // 1.17x
            (4, 64, 256, true),    // 1.25x
            (4, 32, 512, true),    // 1.44x
            (6, 96, 320, true),    // 2.18x
            (8, 16, 512, false),   // 0.57x: short-k barrier overhead
            (8, 64, 128, true),    // 1.39x
            (8, 32, 512, true),    // 2.23x
            (8, 448, 512, true),   // 3.16x
            (16, 448, 2048, true), // 6.43x
            (32, 448, 2048, true), // 6.76x
        ];
        for (rows, input_dim, output_dim, expected) in measured {
            assert_eq!(
                intel_matmul_geometry_supports(rows, input_dim, output_dim),
                expected,
                "geometry gate mis-classified rows={rows} k={input_dim} n={output_dim}"
            );
        }
        // Boundary conditions of each constant.
        assert!(!intel_matmul_geometry_supports(
            0,
            INTEL_MATMUL_MIN_INPUT_DIM,
            INTEL_MATMUL_MIN_OUTPUT_DIM_LOW_ROWS
        ));
        assert!(!intel_matmul_geometry_supports(
            INTEL_MATMUL_MIN_ROWS,
            INTEL_MATMUL_MIN_INPUT_DIM - 1,
            INTEL_MATMUL_MIN_OUTPUT_DIM_LOW_ROWS
        ));
        assert!(!intel_matmul_geometry_supports(
            INTEL_MATMUL_MIN_ROWS,
            INTEL_MATMUL_MIN_INPUT_DIM,
            INTEL_MATMUL_MIN_OUTPUT_DIM_LOW_ROWS - 1
        ));
        assert!(intel_matmul_geometry_supports(
            INTEL_MATMUL_MIN_ROWS,
            INTEL_MATMUL_MIN_INPUT_DIM,
            INTEL_MATMUL_MIN_OUTPUT_DIM_LOW_ROWS
        ));
        assert!(!intel_matmul_geometry_supports(
            7,
            INTEL_MATMUL_MIN_INPUT_DIM,
            INTEL_MATMUL_MIN_OUTPUT_DIM_LOW_ROWS - 1
        ));
        assert!(intel_matmul_geometry_supports(
            8,
            INTEL_MATMUL_MIN_INPUT_DIM,
            INTEL_MATMUL_MIN_OUTPUT_DIM
        ));
        assert!(!intel_matmul_geometry_supports(
            8,
            INTEL_MATMUL_MIN_INPUT_DIM,
            INTEL_MATMUL_MIN_OUTPUT_DIM - 1
        ));
    }

    /// The wrapper reads geometry from the push-constant bytes a call site
    /// actually writes, so pin the layout contract and the short-block fallback.
    #[test]
    fn push_geometry_reads_the_leading_rows_input_output_words() {
        let push = [8u32, 448, 512];
        assert_eq!(
            push_geometry(bytemuck::cast_slice(&push)),
            Some((8, 448, 512))
        );
        assert_eq!(push_geometry(&[0u8; 12]), Some((0, 0, 0)));
        assert_eq!(push_geometry(&[0u8; 11]), None);
        assert_eq!(push_geometry(&[]), None);
    }

    /// The native-FP16 LM backward reliability policy must match exactly the
    /// Intel Gen9/Gen9.5 PCI IDs measured to fault, and nothing else: the
    /// feature is verified green on AMD and on newer Intel generations, so a
    /// broad vendor gate would remove working coverage.
    #[test]
    fn native_fp16_lm_reliability_policy_is_scoped_to_intel_gen9() {
        // The measured target and its siblings.
        assert!(!native_fp16_lm_compute_reliable_for(0x8086, 0x1916));
        assert!(!native_fp16_lm_compute_reliable_for(0x8086, 0x1906));
        assert!(!native_fp16_lm_compute_reliable_for(0x8086, 0x1926));
        assert!(!native_fp16_lm_compute_reliable_for(0x8086, 0x591C));
        assert!(!native_fp16_lm_compute_reliable_for(0x8086, 0x5A85));
        // Every other vendor, and newer Intel generations, keep the feature.
        assert!(native_fp16_lm_compute_reliable_for(0x1002, 0x1916));
        assert!(native_fp16_lm_compute_reliable_for(0x10DE, 0x1916));
        assert!(native_fp16_lm_compute_reliable_for(0x8086, 0x8A52)); // Ice Lake
        assert!(native_fp16_lm_compute_reliable_for(0x8086, 0x9A49)); // Xe
        assert!(native_fp16_lm_compute_reliable_for(0x8086, 0x56A0)); // Arc
        assert!(native_fp16_lm_compute_reliable_for(0x8086, 0x22B0)); // Braswell Gen8
    }

    /// On this Intel target the wrapper must hold the variant and route by
    /// geometry; on any other vendor, or with the kill switch, it must behave
    /// exactly like the portable kernel slot.
    #[test]
    fn vendor_matmul_kernel_routes_dispatch_geometry_to_the_right_module() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        for family in VendorKernelFamily::ALL {
            if !family.is_matmul() {
                continue;
            }
            let kernel = VendorMatmulKernel::new(&device, family)?;
            let expects_variant = select_kernel(&device, family).vendor_specific;
            assert_eq!(
                kernel.variant_available(),
                expects_variant,
                "{} variant availability disagrees with the capability plan",
                family.label()
            );
            for (rows, input_dim, output_dim, intel_is_faster) in [
                (8u32, 448u32, 512u32, true),
                (4, 32, 512, true),
                (1, 448, 512, false),
                (8, 16, 512, false),
                (4, 64, 96, false),
                (4, 32, 128, false),
                (8, 64, 128, true),
            ] {
                assert_eq!(
                    kernel.selects_variant_for(rows, input_dim, output_dim),
                    expects_variant && intel_is_faster,
                    "{} routed rows={rows} k={input_dim} n={output_dim} incorrectly",
                    family.label()
                );
            }
        }
        Ok(())
    }

    /// Families without a variant must resolve to the portable module on every
    /// device, including when the vendor is forced to Intel.
    #[test]
    fn families_without_variants_always_use_the_portable_module() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        for family in VendorKernelFamily::ALL {
            if family.has_intel_variant() {
                continue;
            }
            let selection = select_kernel(&device, family);
            assert!(!selection.vendor_specific);
            assert!(!selection.geometry_gated);
            assert_eq!(selection.spirv, family.portable_spirv());
        }
        Ok(())
    }

    /// Minimal SPIR-V word reader. The vendor modules are committed binaries, so
    /// the regression suite decodes the declarations that justify each
    /// capability gate instead of trusting a comment.
    fn spirv_words(spirv: &[u8]) -> Vec<u32> {
        assert_eq!(spirv.len() % 4, 0, "SPIR-V length must be word aligned");
        spirv
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    /// `(local_size_x, local_size_y, local_size_z)` from `OpExecutionMode`.
    fn spirv_local_size(spirv: &[u8]) -> [u32; 3] {
        const OP_EXECUTION_MODE: u32 = 16;
        const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
        let words = spirv_words(spirv);
        assert_eq!(words[0], 0x0723_0203, "bad SPIR-V magic");
        let mut index = 5usize;
        while index < words.len() {
            let instruction = words[index];
            let word_count = (instruction >> 16) as usize;
            let opcode = instruction & 0xffff;
            assert!(word_count > 0, "zero-length SPIR-V instruction");
            if opcode == OP_EXECUTION_MODE
                && word_count >= 6
                && words[index + 2] == EXECUTION_MODE_LOCAL_SIZE
            {
                return [words[index + 3], words[index + 4], words[index + 5]];
            }
            index += word_count;
        }
        panic!("SPIR-V module declares no LocalSize execution mode");
    }

    /// Count `OpControlBarrier` and workgroup-storage `OpVariable` declarations.
    fn spirv_workgroup_usage(spirv: &[u8]) -> (usize, usize) {
        const OP_VARIABLE: u32 = 59;
        const OP_CONTROL_BARRIER: u32 = 224;
        const STORAGE_CLASS_WORKGROUP: u32 = 4;
        let words = spirv_words(spirv);
        let mut index = 5usize;
        let mut barriers = 0usize;
        let mut workgroup_variables = 0usize;
        while index < words.len() {
            let instruction = words[index];
            let word_count = (instruction >> 16) as usize;
            let opcode = instruction & 0xffff;
            if opcode == OP_CONTROL_BARRIER {
                barriers += 1;
            }
            if opcode == OP_VARIABLE
                && word_count >= 4
                && words[index + 3] == STORAGE_CLASS_WORKGROUP
            {
                workgroup_variables += 1;
            }
            if word_count == 0 {
                break;
            }
            index += word_count;
        }
        (barriers, workgroup_variables)
    }

    /// Byte size of one SPIR-V type, following array and vector wrapping down to
    /// the scalar it is built from. `0` for anything this decoder does not
    /// recognize, which makes an unrecognized declaration fail the footprint
    /// assertion below rather than pass it silently.
    fn spirv_type_bytes(
        type_id: u32,
        scalar_bytes: &std::collections::HashMap<u32, u32>,
        vector: &std::collections::HashMap<u32, (u32, u32)>,
        array: &std::collections::HashMap<u32, (u32, u32)>,
        constant: &std::collections::HashMap<u32, u32>,
    ) -> u32 {
        if let Some(bytes) = scalar_bytes.get(&type_id) {
            return *bytes;
        }
        if let Some((component, count)) = vector.get(&type_id) {
            return spirv_type_bytes(*component, scalar_bytes, vector, array, constant) * *count;
        }
        if let Some((element, length_id)) = array.get(&type_id) {
            let length = *constant.get(length_id).unwrap_or(&0);
            return spirv_type_bytes(*element, scalar_bytes, vector, array, constant) * length;
        }
        0
    }

    /// Total bytes of workgroup storage a module declares, decoded from its
    /// `Workgroup` `OpVariable`s and the types they point at. GLSL `shared`
    /// arrays are plain `OpTypeArray`s over a scalar, so the sum of
    /// `length * element_bytes` is exactly what `maxComputeSharedMemorySize`
    /// has to cover. This is what makes
    /// [`VendorKernelFamily::intel_shared_memory_bytes`] a checked claim rather
    /// than a comment: the capability gate is compared against the artifact it
    /// actually gates.
    fn spirv_workgroup_bytes(spirv: &[u8]) -> u32 {
        const OP_TYPE_INT: u32 = 21;
        const OP_TYPE_FLOAT: u32 = 22;
        const OP_TYPE_VECTOR: u32 = 23;
        const OP_TYPE_ARRAY: u32 = 28;
        const OP_TYPE_POINTER: u32 = 32;
        const OP_CONSTANT: u32 = 43;
        const OP_VARIABLE: u32 = 59;
        const STORAGE_CLASS_WORKGROUP: u32 = 4;
        use std::collections::HashMap;
        let words = spirv_words(spirv);
        let mut scalar_bytes: HashMap<u32, u32> = HashMap::new();
        let mut vector: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut array: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut pointer: HashMap<u32, u32> = HashMap::new();
        let mut constant: HashMap<u32, u32> = HashMap::new();
        let mut variables: Vec<u32> = Vec::new();
        let mut index = 5usize;
        while index < words.len() {
            let instruction = words[index];
            let word_count = (instruction >> 16) as usize;
            let opcode = instruction & 0xffff;
            if word_count == 0 {
                break;
            }
            // Types precede the variables that use them, and constants precede
            // the array types that reference them, so one forward pass is
            // enough to have every operand resolved when it is read.
            match opcode {
                OP_TYPE_INT | OP_TYPE_FLOAT if word_count >= 3 => {
                    scalar_bytes.insert(words[index + 1], words[index + 2] / 8);
                }
                OP_TYPE_VECTOR if word_count >= 4 => {
                    vector.insert(words[index + 1], (words[index + 2], words[index + 3]));
                }
                OP_TYPE_ARRAY if word_count >= 4 => {
                    array.insert(words[index + 1], (words[index + 2], words[index + 3]));
                }
                OP_TYPE_POINTER if word_count >= 4 => {
                    // OpTypePointer: result id, storage class, pointee type.
                    pointer.insert(words[index + 1], words[index + 3]);
                }
                OP_CONSTANT if word_count >= 4 => {
                    // OpConstant: result type, result id, value.
                    constant.insert(words[index + 2], words[index + 3]);
                }
                OP_VARIABLE if word_count >= 4 && words[index + 3] == STORAGE_CLASS_WORKGROUP => {
                    variables.push(*pointer.get(&words[index + 1]).unwrap_or(&0));
                }
                _ => {}
            }
            index += word_count;
        }
        variables
            .into_iter()
            .map(|type_id| spirv_type_bytes(type_id, &scalar_bytes, &vector, &array, &constant))
            .sum()
    }

    /// The committed Intel modules must actually declare what the capability
    /// gates and the host dispatch assume. A stale or mistargeted `.spv` (wrong
    /// local size, missing shared-memory staging, missing barriers) fails here
    /// instead of surfacing as a driver validation error on a user machine.
    #[test]
    fn committed_vendor_modules_declare_their_capability_contract() {
        for family in families_with_intel_variants() {
            let intel = family.intel_spirv().expect("tuned family has a module");
            assert_eq!(
                spirv_local_size(intel),
                family.intel_local_size(),
                "{} Intel module declares an unexpected local size",
                family.label()
            );
            let (barriers, workgroup_variables) = spirv_workgroup_usage(intel);
            match family.intel_shared_memory_bytes() {
                Some(bytes) => {
                    assert!(
                        workgroup_variables >= 2 && barriers >= 2,
                        "{} Intel module claims workgroup memory but declares {workgroup_variables} variables / {barriers} barriers",
                        family.label()
                    );
                    // The declared footprint is what `maxComputeSharedMemorySize`
                    // has to cover, so decode it from the artifact instead of
                    // trusting the constant next to it.
                    assert_eq!(
                        spirv_workgroup_bytes(intel),
                        bytes,
                        "{} Intel module's declared workgroup footprint disagrees with the capability gate",
                        family.label()
                    );
                }
                None => {
                    assert_eq!(
                        workgroup_variables,
                        0,
                        "{} Intel module uses workgroup memory without declaring its footprint",
                        family.label()
                    );
                }
            }
            // The portable module is the dispatch-contract reference: one
            // invocation per element, so its workgroup size *is* its coverage.
            assert_eq!(
                spirv_local_size(family.portable_spirv())
                    .iter()
                    .product::<u32>(),
                family.workgroup_elements(),
                "{} portable local size does not match the declared coverage",
                family.label()
            );
        }
    }

    /// The budget must admit every shape the device completes and refuse the
    /// shapes that reset it, so it is pinned to the measured ladder rather than
    /// to a fitted curve.
    #[test]
    fn attention_backward_budget_matches_the_measured_boundary() {
        let work = |seq: u64, strict: bool| attention_backward_work(2, 32, seq, seq, strict);
        assert!(work(64, false) <= ATTENTION_BACKWARD_WORK_BUDGET);
        assert!(
            work(256, false) <= ATTENTION_BACKWARD_WORK_BUDGET,
            "mode 0 seq 256 completes in 2576 ms"
        );
        assert!(
            work(320, false) > ATTENTION_BACKWARD_WORK_BUDGET,
            "mode 0 seq 320 loses the device"
        );
        assert!(
            work(224, true) <= ATTENTION_BACKWARD_WORK_BUDGET,
            "strict seq 224 completes in 2714 ms"
        );
        assert!(
            work(248, true) > ATTENTION_BACKWARD_WORK_BUDGET,
            "strict seq 248 loses the device"
        );
        assert!(
            work(512, false) > ATTENTION_BACKWARD_WORK_BUDGET,
            "mode 0 seq 512 loses the device"
        );
        assert_eq!(
            attention_backward_work(2, 32, 128, 128, true),
            attention_backward_work(2, 32, 128, 128, false) * 144 / 100,
            "strict modes carry their measured 1.44x factor"
        );
    }

    /// The dispatch site refuses through this function, so its decision and the
    /// message it carries are checked here, without dispatching a geometry that
    /// would reset the device -- the only other way to reach the refusal.
    #[test]
    fn attention_backward_budget_error_names_the_geometry_and_the_override() {
        let budget = ATTENTION_BACKWARD_WORK_BUDGET;
        assert!(
            attention_backward_budget_error(budget, 2, 32, 256, 256, 1, false, 1).is_none(),
            "seq 256 completes in 2576 ms"
        );

        let message = attention_backward_budget_error(budget, 2, 32, 512, 512, 1, false, 1)
            .expect("seq 512 resets this device");
        assert!(message.contains("512 queries x 512 keys"), "{message}");
        assert!(message.contains("head dim 32"), "{message}");
        assert!(message.contains(&budget.to_string()), "{message}");
        assert!(
            message.contains(HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET_ENV),
            "the override has to be discoverable from the failure: {message}"
        );

        let strict = attention_backward_budget_error(budget, 2, 32, 248, 248, 1, true, 1)
            .expect("strict seq 248 resets this device");
        assert!(strict.contains("strict true"), "{strict}");

        // An adapter with no measured boundary is never refused.
        assert!(attention_backward_budget_error(u64::MAX, 2, 32, 4096, 4096, 1, true, 1).is_none());
    }

    /// The split turns the strict shapes the guard used to refuse into several
    /// sequential key-range dispatches, and leaves every shape it cannot help alone.
    #[test]
    fn attention_backward_key_blocks_split_only_the_strict_adjoint() {
        let budget = ATTENTION_BACKWARD_WORK_BUDGET;
        assert_eq!(
            budget.saturating_mul(ATTENTION_BACKWARD_STREAMED_BUDGET_PERCENT) / 100,
            ATTENTION_BACKWARD_STREAMED_WORK_BUDGET,
            "the streamed bound is the measured cached bound scaled to its model"
        );
        let blocks = |seq: u64, strict: bool| {
            attention_backward_key_blocks(Some(budget), 2, 32, seq, seq, 1, strict)
        };
        // Shapes that already fit are never split.
        assert_eq!(blocks(256, false), 1);
        assert_eq!(blocks(224, true), 1);
        // The fused shape measured to lose the device at seq 320 cannot be split:
        // its gradients come from the shared primary pass, whose softmax_dot is a
        // running prefix across the whole key range.
        assert_eq!(blocks(320, false), 1);
        // A strict shape whose row statistics are cached keeps its refusal: splitting
        // it would move it onto the streamed adjoint, and 4 blocks of that took the
        // device out at seq 248 where a single dispatch had been refused cleanly. The
        // measured shape has to stay refused until the cached adjoint itself is
        // splittable.
        assert_eq!(blocks(248, true), 1);
        assert!(
            attention_backward_budget_error(budget, 2, 32, 248, 248, 1, true, 1).is_some(),
            "strict seq 248 must stay refused rather than split onto the streamed adjoint"
        );
        // Past the statistics cache the adjoint streams, and that is where a split
        // earns its keep: each block is its own submission, so the block count is what
        // decides whether a submission fits the measured boundary.
        assert_eq!(blocks(512, true), 1);
        assert!(
            attention_backward_streamed_work(2, 8, 257, 4, 4)
                <= ATTENTION_BACKWARD_STREAMED_WORK_BUDGET
        );
        // The shape the ladder measured losing the device -- 257 queries, head dim 8,
        // key span 16 -- is now admitted as six submissions. Its streamed cost carries
        // the full key span, so one block may own three keys and six of those stay
        // inside the bound.
        let blocks_for = |queries: u64, key_span: u64, head_dim: u64| {
            attention_backward_key_blocks(Some(budget), 2, head_dim, queries, key_span, 1, true)
        };
        assert_eq!(
            blocks_for(257, 4, 8),
            1,
            "span 4 completed in 395 ms unsplit"
        );
        assert_eq!(blocks_for(257, 16, 8), 6);
        assert!(
            attention_backward_budget_error(budget, 2, 8, 257, 16, 1, true, 6).is_none(),
            "six blocks of the span that lost the device have to fit the streamed bound"
        );
        // This is the regression the test exists to catch, stated as an invariant: a
        // credited count is sound only when the widest block it leaves behind fits the
        // streamed bound, and a count that cannot get there has to be the policy's cap
        // with the guard refusing the geometry. An earlier policy credited eight blocks
        // to the shape above while its blocks were workgroups of one submission, which
        // left that submission exactly as long as it had been.
        let streamed_limit =
            budget.saturating_mul(ATTENTION_BACKWARD_STREAMED_BUDGET_PERCENT) / 100;
        for (queries, key_span, head_dim) in [
            (257u64, 4u64, 8u64),
            (257, 16, 8),
            (257, 21, 8),
            (257, 64, 8),
            (257, 257, 8),
            (512, 512, 32),
            (4096, 4096, 32),
        ] {
            let count = blocks_for(queries, key_span, head_dim);
            assert!(count >= 1);
            if count > 1 {
                let widest = (key_span + count as u64 - 1) / count as u64;
                let widest_work =
                    attention_backward_streamed_work(2, head_dim, queries, widest, key_span);
                if widest_work <= streamed_limit {
                    assert!(
                        attention_backward_budget_error(
                            budget, 2, head_dim, queries, key_span, 1, true, count
                        )
                        .is_none(),
                        "the guard has to admit streamed {queries}x{key_span} at head dim {head_dim}, which the policy split into {count} blocks"
                    );
                } else {
                    assert_eq!(
                        count, ATTENTION_BACKWARD_MAX_KEY_BLOCKS,
                        "streamed {queries}x{key_span} at head dim {head_dim} left {widest_work} work units in its widest block, past the {streamed_limit} bound, without being split to the cap"
                    );
                    assert!(
                        attention_backward_budget_error(
                            budget, 2, head_dim, queries, key_span, 1, true, count
                        )
                        .is_some(),
                        "a geometry split to the cap that still does not fit has to be refused"
                    );
                }
            } else {
                // A count of one has to mean no split could help: either the geometry
                // fits unsplit, or even a single key of it costs more than the bound.
                let whole_span =
                    attention_backward_streamed_work(2, head_dim, queries, key_span, key_span);
                let one_key = attention_backward_streamed_work(2, head_dim, queries, 1, key_span);
                assert!(
                    whole_span <= streamed_limit || one_key > streamed_limit,
                    "streamed {queries}x{key_span} at head dim {head_dim} was left unsplit although a split could have brought a submission inside the bound"
                );
            }
        }
        // The shapes measured to lose the device at head dim 32 stay refused even at the
        // cap, because their streamed work needs more blocks than it allows.
        for seq in [320u64, 512] {
            let count = attention_backward_key_blocks(Some(budget), 2, 32, seq, seq, 1, true);
            assert_eq!(count, 1, "a streamed geometry is never split");
            assert!(
                attention_backward_budget_error(budget, 2, 32, seq, seq, 1, true, count).is_some(),
                "strict seq {seq} is past the streamed bound and stays refused"
            );
        }
        // The longest shape here streams too, so it is refused un-split.
        let capped = attention_backward_key_blocks(Some(budget), 2, 32, 1024, 1024, 1, true);
        assert_eq!(capped, 1);
        let message = attention_backward_budget_error(budget, 2, 32, 1024, 1024, 1, true, capped)
            .expect("strict seq 1024 streams and is far past its bound");
        assert!(message.contains("1024 queries x 1024 keys"), "{message}");
        // An adapter with no measured boundary is never split.
        assert_eq!(
            attention_backward_key_blocks(None, 2, 32, 4096, 4096, 1, true),
            1
        );
        assert_eq!(
            attention_backward_key_blocks(Some(budget), 2, 32, 8192, 8192, 1, true),
            1,
            "a length at head dim 32 costs more in one key than the bound allows"
        );
    }

    /// On a non-Intel adapter the selection must return the exact pre-existing
    /// module bytes for every family, which is what guarantees zero AMD/NVIDIA
    /// behavioral change from this module's introduction.
    #[test]
    fn non_intel_devices_keep_the_portable_modules() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        if std::env::var(HIERARCHOS_VULKAN_FORCE_VENDOR_ENV).is_ok() {
            return Ok(());
        }
        if GpuVendor::from_pci_vendor_id(device.vendor_id()).is_intel() {
            return Ok(());
        }
        // An RDNA3 adapter is the one AMD class that leaves the portable
        // modules, and `rdna3_devices_select_the_rdna3_modules` is its claim.
        if is_amd_rdna3(&device) {
            return Ok(());
        }
        for family in VendorKernelFamily::ALL {
            let selection = select_kernel(&device, family);
            assert!(
                !selection.vendor_specific,
                "{} selected {}",
                family.label(),
                selection.variant
            );
            assert_eq!(selection.spirv, family.portable_spirv());
        }
        Ok(())
    }

    #[test]
    fn intel_variants_are_bit_exact_with_portable_kernels() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        // The Intel modules are executed and compared on every vendor, so this
        // evidence does not depend on owning Gen9 hardware, but the *selection*
        // claim is only meaningful when the plan actually picked them.
        let plan = vendor_kernel_plan(&device);
        println!(
            "vendor-kernel plan on {} (vendor=0x{:04x} device=0x{:04x} driver={}):",
            device.name(),
            device.vendor_id(),
            device.device_id(),
            device.driver_version()
        );
        for selection in &plan {
            println!(
                "  {:<20} {} ({})",
                selection.family.label(),
                selection.variant,
                selection.vendor.label()
            );
        }

        linear_forward_matches_portable(&device)?;
        linear_bias_forward_matches_portable(&device)?;
        linear_residual_forward_matches_portable(&device)?;
        linear3_forward_matches_portable(&device)?;
        Ok(())
    }

    /// The plan on an actual RDNA3 adapter: every family with a tuned module is
    /// selected there, and nothing else is.
    #[test]
    fn rdna3_devices_select_the_rdna3_modules() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        if !is_amd_rdna3(&device) || std::env::var(HIERARCHOS_VULKAN_FORCE_VENDOR_ENV).is_ok() {
            return Ok(());
        }
        for family in VendorKernelFamily::ALL {
            let selection = select_kernel(&device, family);
            if family.has_amd_variant() {
                assert_eq!(
                    selection.variant,
                    family.amd_variant_label(),
                    "{} keeps the portable module on an RDNA3 adapter",
                    family.label()
                );
                assert_eq!(selection.spirv, family.amd_spirv().expect("module"));
            } else {
                assert!(!selection.vendor_specific, "{}", family.label());
                assert_eq!(selection.spirv, family.portable_spirv());
            }
        }
        Ok(())
    }

    /// The RDNA3 modules are executed and compared with the portable ones on
    /// *every* adapter, so their numeric contract does not depend on owning
    /// RDNA3 hardware -- only their timing does. Every geometry the gate admits
    /// is covered, plus every geometry the gate rejects, because a module must
    /// be correct where it is not selected too.
    #[test]
    fn rdna3_variants_are_bit_exact_with_portable_kernels() -> Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        for family in families_with_amd_variants() {
            let module = family
                .amd_spirv()
                .expect("a family with an RDNA3 variant has a module");
            for (rows, input_dim, output_dim) in linear_layout_geometries() {
                if family == VendorKernelFamily::Linear3Forward {
                    linear3_forward_case_with(&device, rows, input_dim, output_dim, module)?;
                } else {
                    linear_layout_case_with(&device, family, rows, input_dim, output_dim, module)?;
                }
            }
        }
        Ok(())
    }

    struct AbOutcome {
        portable_outputs: Vec<Vec<f32>>,
        variant_outputs: Vec<Vec<f32>>,
        portable_ms: f64,
        variant_ms: f64,
    }

    /// A/B harness: identical buffers, push constants, and dispatch grid are
    /// driven through both modules. Timing passes exclude uploads; the parity
    /// pass re-seeds the initial state so in-place kernels start identically.
    #[allow(clippy::too_many_arguments)]
    fn ab_dispatch(
        device: &VulkanDevice,
        portable: &ComputeKernel,
        variant: &ComputeKernel,
        initial: &[(&GpuBuffer, &[f32])],
        bindings: &[&GpuBuffer],
        push: &[u32],
        grid: [u32; 3],
        outputs: &[(&GpuBuffer, usize)],
        timed_dispatches: usize,
    ) -> Result<AbOutcome> {
        /// One isolated submission per sample, timed with queue timestamp
        /// queries when the adapter supports them and host wall time otherwise.
        /// The returned value is per-dispatch milliseconds.
        fn timing_pass(
            device: &VulkanDevice,
            kernel: &ComputeKernel,
            initial: &[(&GpuBuffer, &[f32])],
            bindings: &[&GpuBuffer],
            push: &[u32],
            grid: [u32; 3],
            timed_dispatches: usize,
        ) -> Result<f64> {
            {
                let mut commands = ComputeBatch::new(device)?;
                for (buffer, values) in initial {
                    commands.upload_f32(buffer, values)?;
                }
                commands.submit()?;
            }
            for _ in 0..WARMUP_DISPATCHES {
                let mut commands = ComputeBatch::new(device)?;
                kernel.record_dispatch(
                    &mut commands,
                    bindings,
                    bytemuck::cast_slice(push),
                    grid,
                )?;
                commands.submit()?;
            }
            let mut samples = Vec::with_capacity(timed_dispatches);
            for _ in 0..timed_dispatches {
                samples.push(device.time_compute_batch_ms(|commands| {
                    kernel.record_dispatch(commands, bindings, bytemuck::cast_slice(push), grid)
                })?);
            }
            samples.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
            Ok(samples[samples.len() / 2])
        }

        fn parity_pass(
            device: &VulkanDevice,
            kernel: &ComputeKernel,
            initial: &[(&GpuBuffer, &[f32])],
            bindings: &[&GpuBuffer],
            push: &[u32],
            grid: [u32; 3],
            outputs: &[(&GpuBuffer, usize)],
        ) -> Result<Vec<Vec<f32>>> {
            let mut commands = ComputeBatch::new(device)?;
            for (buffer, values) in initial {
                commands.upload_f32(buffer, values)?;
            }
            kernel.record_dispatch(&mut commands, bindings, bytemuck::cast_slice(push), grid)?;
            let readbacks = outputs
                .iter()
                .map(|(_, len)| GpuBuffer::zeros_host_f32(device, *len))
                .collect::<Result<Vec<_>>>()?;
            for ((source, len), readback) in outputs.iter().zip(readbacks.iter()) {
                commands.readback_f32(source, readback, *len)?;
            }
            commands.submit()?;
            readbacks
                .iter()
                .zip(outputs.iter())
                .map(|(readback, (_, len))| readback.read_f32(*len))
                .collect()
        }

        let portable_ms = timing_pass(
            device,
            portable,
            initial,
            bindings,
            push,
            grid,
            timed_dispatches,
        )?;
        let variant_ms = timing_pass(
            device,
            variant,
            initial,
            bindings,
            push,
            grid,
            timed_dispatches,
        )?;
        let portable_outputs =
            parity_pass(device, portable, initial, bindings, push, grid, outputs)?;
        let variant_outputs = parity_pass(device, variant, initial, bindings, push, grid, outputs)?;
        Ok(AbOutcome {
            portable_outputs,
            variant_outputs,
            portable_ms,
            variant_ms,
        })
    }

    /// Logit-drift ceiling every reference-parity test in this crate uses
    /// (`hierarchos_causal_lm_tests`, `rwkv_optimizer`, `transformer`). The
    /// vendor modules are held to the *tighter* raw-bit contract below, so they
    /// can never turn a green reference model red.
    const REFERENCE_LOGIT_DRIFT: f32 = 2.0e-7;

    fn assert_bit_exact(family: VendorKernelFamily, outcome: &AbOutcome, case: &str) {
        let mut max_abs_delta = 0.0f32;
        let mut max_ulp_delta = 0u32;
        for (index, (lhs, rhs)) in outcome
            .portable_outputs
            .iter()
            .zip(outcome.variant_outputs.iter())
            .enumerate()
        {
            assert_eq!(
                lhs.len(),
                rhs.len(),
                "{} {case} output {index} length mismatch",
                family.label()
            );
            for (position, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert_eq!(
                    lhs.to_bits(),
                    rhs.to_bits(),
                    "{} {case} output {index}[{position}] differs: portable={lhs} variant={rhs}",
                    family.label()
                );
                max_abs_delta = max_abs_delta.max((lhs - rhs).abs());
                max_ulp_delta = max_ulp_delta.max(lhs.to_bits().abs_diff(rhs.to_bits()));
            }
        }
        // The bit-exactness assert above already implies this, but the parity
        // claim is stated in the drift units the reference model tests use, so
        // the guarantee is machine-checked here rather than implied.
        assert!(
            max_abs_delta <= REFERENCE_LOGIT_DRIFT,
            "{} {case} vendor drift {max_abs_delta} exceeds the {REFERENCE_LOGIT_DRIFT} reference-parity ceiling",
            family.label()
        );
        println!(
            "vendor-kernel A/B {} [{case}]: portable={:.4}ms variant={:.4}ms speedup={:.3}x bit_exact=true max_abs_delta={max_abs_delta:.1e} (<= {REFERENCE_LOGIT_DRIFT:.1e}) max_ulp_delta={max_ulp_delta}",
            family.label(),
            outcome.portable_ms,
            outcome.variant_ms,
            outcome.portable_ms / outcome.variant_ms
        );
    }

    /// The portable/variant pair, from an explicitly named tuned module, so the
    /// RDNA3 modules can be executed on an adapter that will never select them.
    fn kernel_pair_with(
        device: &VulkanDevice,
        family: VendorKernelFamily,
        variant_spirv: &'static [u8],
    ) -> Result<(ComputeKernel, ComputeKernel)> {
        let portable = ComputeKernel::new(
            device,
            family.portable_spirv(),
            family.binding_count(),
            family.push_constant_bytes(),
        )?;
        let variant = ComputeKernel::new(
            device,
            variant_spirv,
            family.binding_count(),
            family.push_constant_bytes(),
        )?;
        Ok((portable, variant))
    }

    /// The `div_ceil(16)` grids every linear-family call site uses.
    fn linear_grid(rows: usize, output_dim: usize) -> [u32; 3] {
        [
            (output_dim as u32).div_ceil(16),
            (rows as u32).div_ceil(16),
            1,
        ]
    }

    /// A/B body for the matmul families that share the row-major `x * w^T`
    /// layout and the `(rows, input_dim, output_dim)` push block. The bias or
    /// residual binding is supplied only for the families whose portable module
    /// declares it, in the portable binding order, so the comparison runs the
    /// real dispatch contract rather than a reduced one.
    fn linear_layout_case(
        device: &VulkanDevice,
        family: VendorKernelFamily,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        let variant_spirv = family
            .intel_spirv()
            .ok_or_else(|| anyhow::anyhow!("{} has no Intel variant", family.label()))?;
        linear_layout_case_with(device, family, rows, input_dim, output_dim, variant_spirv)
    }

    fn linear_layout_case_with(
        device: &VulkanDevice,
        family: VendorKernelFamily,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        variant_spirv: &'static [u8],
    ) -> Result<()> {
        let (portable, variant) = kernel_pair_with(device, family, variant_spirv)?;
        let input = pseudo_random(rows * input_dim, 0x51ed);
        let weight = pseudo_random(output_dim * input_dim, 0x9a13);
        let bias = pseudo_random(output_dim, 0x77);
        let residual = pseudo_random(rows * output_dim, 0x2f1d);
        let x = GpuBuffer::from_f32(device, &input)?;
        let w = GpuBuffer::from_f32(device, &weight)?;
        let b = GpuBuffer::from_f32(device, &bias)?;
        let r = GpuBuffer::from_f32(device, &residual)?;
        let out = GpuBuffer::zeros_f32(device, rows * output_dim)?;
        let push = [rows as u32, input_dim as u32, output_dim as u32];
        let grid = linear_grid(rows, output_dim);
        let outputs = [(&out, rows * output_dim)];
        let outcome = match family {
            VendorKernelFamily::LinearForward => ab_dispatch(
                device,
                &portable,
                &variant,
                &[(&x, &input), (&w, &weight)],
                &[&x, &w, &out],
                &push,
                grid,
                &outputs,
                TIMED_DISPATCHES,
            )?,
            VendorKernelFamily::LinearBiasForward => ab_dispatch(
                device,
                &portable,
                &variant,
                &[(&x, &input), (&w, &weight), (&b, &bias)],
                &[&x, &w, &b, &out],
                &push,
                grid,
                &outputs,
                TIMED_DISPATCHES,
            )?,
            VendorKernelFamily::LinearResidualForward => ab_dispatch(
                device,
                &portable,
                &variant,
                &[(&x, &input), (&w, &weight), (&r, &residual)],
                &[&x, &w, &r, &out],
                &push,
                grid,
                &outputs,
                TIMED_DISPATCHES,
            )?,
            other => anyhow::bail!("{} is not a row-major linear family", other.label()),
        };
        assert_bit_exact(
            family,
            &outcome,
            &format!("rows={rows},k={input_dim},n={output_dim}"),
        );
        Ok(())
    }

    /// Experiment hook used to qualify *candidate* modules before they are
    /// committed into the selection table. It loads two arbitrary SPIR-V files
    /// from disk, so shader tuning does not require rebuilding the crate:
    ///
    /// ```text
    /// HIERARCHOS_VULKAN_VENDOR_BENCH=linear-forward
    /// HIERARCHOS_VULKAN_VENDOR_BENCH_A=shaders/linear_forward.spv
    /// HIERARCHOS_VULKAN_VENDOR_BENCH_B=/tmp/exp/candidate.spv
    /// HIERARCHOS_VULKAN_VENDOR_BENCH_SHAPE=8x448x512
    /// ```
    #[test]
    fn candidate_module_experiment_from_env() -> Result<()> {
        let Ok(family_label) = std::env::var("HIERARCHOS_VULKAN_VENDOR_BENCH") else {
            return Ok(());
        };
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let family = match family_label.as_str() {
            "linear-forward" => VendorKernelFamily::LinearForward,
            "linear-bias-forward" => VendorKernelFamily::LinearBiasForward,
            "linear-residual-forward" => VendorKernelFamily::LinearResidualForward,
            "linear3-forward" => VendorKernelFamily::Linear3Forward,
            "silu-forward" => VendorKernelFamily::SiluForward,
            "silu-backward" => VendorKernelFamily::SiluBackward,
            "adamw" => VendorKernelFamily::AdamW,
            other => anyhow::bail!("unknown HIERARCHOS_VULKAN_VENDOR_BENCH family {other:?}"),
        };
        let load = |name: &str, default: Option<&str>| -> Result<Vec<u8>> {
            let path = match std::env::var(name) {
                Ok(path) => path,
                Err(_) => default
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow::anyhow!("{name} is required"))?,
            };
            Ok(std::fs::read(&path)?)
        };
        let default_a = None;
        let default_b = None;
        let a_bytes = load("HIERARCHOS_VULKAN_VENDOR_BENCH_A", default_a)?;
        let b_bytes = load("HIERARCHOS_VULKAN_VENDOR_BENCH_B", default_b)?;
        let shape = std::env::var("HIERARCHOS_VULKAN_VENDOR_BENCH_SHAPE").ok();
        let dispatches = std::env::var("HIERARCHOS_VULKAN_VENDOR_BENCH_ITERS")
            .ok()
            .map(|raw| raw.parse::<usize>())
            .transpose()?
            .unwrap_or(TIMED_DISPATCHES);
        let a = ComputeKernel::new(
            &device,
            &a_bytes,
            family.binding_count(),
            family.push_constant_bytes(),
        )?;
        let b = ComputeKernel::new(
            &device,
            &b_bytes,
            family.binding_count(),
            family.push_constant_bytes(),
        )?;

        match family {
            VendorKernelFamily::LinearForward
            | VendorKernelFamily::LinearBiasForward
            | VendorKernelFamily::LinearResidualForward
            | VendorKernelFamily::Linear3Forward => {
                let (rows, input_dim, output_dim) = match shape.as_deref() {
                    Some(shape) => {
                        let parts = shape
                            .split('x')
                            .map(|part| part.parse::<usize>())
                            .collect::<Result<Vec<_>, _>>()?;
                        if parts.len() != 3 {
                            anyhow::bail!(
                                "expected SHAPE=rowsxinput_dimxoutput_dim, got {shape:?}"
                            );
                        }
                        (parts[0], parts[1], parts[2])
                    }
                    None => (8, 448, 512),
                };
                let input = pseudo_random(rows * input_dim, 0x51ed);
                let weight = pseudo_random(output_dim * input_dim, 0x9a13);
                let bias = pseudo_random(output_dim, 0x77);
                let residual = pseudo_random(rows * output_dim, 0x2f1d);
                let x = GpuBuffer::from_f32(&device, &input)?;
                let w = GpuBuffer::from_f32(&device, &weight)?;
                let b_bias = GpuBuffer::from_f32(&device, &bias)?;
                let r = GpuBuffer::from_f32(&device, &residual)?;
                let out = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
                let push = [rows as u32, input_dim as u32, output_dim as u32];
                let grid = linear_grid(rows, output_dim);
                let outputs = [(&out, rows * output_dim)];
                let outcome = match family {
                    VendorKernelFamily::LinearForward => ab_dispatch(
                        &device,
                        &a,
                        &b,
                        &[(&x, &input), (&w, &weight)],
                        &[&x, &w, &out],
                        &push,
                        grid,
                        &outputs,
                        dispatches,
                    )?,
                    VendorKernelFamily::LinearBiasForward => ab_dispatch(
                        &device,
                        &a,
                        &b,
                        &[(&x, &input), (&w, &weight), (&b_bias, &bias)],
                        &[&x, &w, &b_bias, &out],
                        &push,
                        grid,
                        &outputs,
                        dispatches,
                    )?,
                    VendorKernelFamily::LinearResidualForward => ab_dispatch(
                        &device,
                        &a,
                        &b,
                        &[(&x, &input), (&w, &weight), (&r, &residual)],
                        &[&x, &w, &r, &out],
                        &push,
                        grid,
                        &outputs,
                        dispatches,
                    )?,
                    VendorKernelFamily::Linear3Forward => {
                        let out_k = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
                        let out_v = GpuBuffer::zeros_f32(&device, rows * output_dim)?;
                        let lanes = rows * output_dim;
                        // The bench hook shares one x/w pair across the three
                        // branches; the per-branch parity proof lives in
                        // `linear3_forward_case`.
                        ab_dispatch(
                            &device,
                            &a,
                            &b,
                            &[(&x, &input), (&w, &weight)],
                            &[&x, &w, &x, &w, &x, &w, &out, &out_k, &out_v],
                            &push,
                            grid,
                            &[(&out, lanes), (&out_k, lanes), (&out_v, lanes)],
                            dispatches,
                        )?
                    }
                    other => anyhow::bail!("{} is not a bench-matmul family", other.label()),
                };
                assert_bit_exact(family, &outcome, shape.as_deref().unwrap_or("8x448x512"));
            }
            VendorKernelFamily::SiluForward
            | VendorKernelFamily::SiluBackward
            | VendorKernelFamily::AdamW => {
                let len = match shape.as_deref() {
                    Some(shape) => shape
                        .strip_prefix("len=")
                        .ok_or_else(|| anyhow::anyhow!("expected SHAPE=len=N, got {shape:?}"))?
                        .parse::<usize>()?,
                    None => 256 * 9 + 137,
                };
                let input = pseudo_random(len, 0xfeed);
                let x = GpuBuffer::from_f32(&device, &input)?;
                let grid = [(len as u32).div_ceil(256), 1, 1];
                let outcome = match family {
                    VendorKernelFamily::SiluForward => {
                        let out = GpuBuffer::zeros_f32(&device, len)?;
                        ab_dispatch(
                            &device,
                            &a,
                            &b,
                            &[(&x, &input)],
                            &[&x, &out],
                            &[len as u32],
                            grid,
                            &[(&out, len)],
                            dispatches,
                        )?
                    }
                    VendorKernelFamily::SiluBackward => {
                        let grad = pseudo_random(len, 0x1357);
                        let grad_buffer = GpuBuffer::from_f32(&device, &grad)?;
                        let grad_input = GpuBuffer::zeros_f32(&device, len)?;
                        ab_dispatch(
                            &device,
                            &a,
                            &b,
                            &[(&x, &input), (&grad_buffer, &grad)],
                            &[&grad_buffer, &x, &grad_input],
                            &[len as u32],
                            grid,
                            &[(&grad_input, len)],
                            dispatches,
                        )?
                    }
                    _ => {
                        let gradient = pseudo_random(len, 0x2bad);
                        let exp_avg = pseudo_random(len, 0x3bad);
                        let exp_avg_sq: Vec<f32> = pseudo_random(len, 0x4bad)
                            .into_iter()
                            .map(|value| value.abs() + 0.25)
                            .collect();
                        let parameter = pseudo_random(len, 0x1bad);
                        let p = GpuBuffer::from_f32(&device, &parameter)?;
                        let g = GpuBuffer::from_f32(&device, &gradient)?;
                        let m = GpuBuffer::from_f32(&device, &exp_avg)?;
                        let v = GpuBuffer::from_f32(&device, &exp_avg_sq)?;
                        let push = [
                            len as u32,
                            7,
                            0.001f32.to_bits(),
                            0.9f32.to_bits(),
                            0.999f32.to_bits(),
                            1.0e-8f32.to_bits(),
                            0.01f32.to_bits(),
                        ];
                        ab_dispatch(
                            &device,
                            &a,
                            &b,
                            &[
                                (&p, &parameter),
                                (&g, &gradient),
                                (&m, &exp_avg),
                                (&v, &exp_avg_sq),
                            ],
                            &[&p, &g, &m, &v],
                            &push,
                            grid,
                            &[(&p, len), (&m, len), (&v, len)],
                            dispatches,
                        )?
                    }
                };
                assert_bit_exact(family, &outcome, shape.as_deref().unwrap_or("default"));
            }
        }
        Ok(())
    }

    /// Production-shaped, tiled, and non-multiple-of-tile geometries for a
    /// row-major linear family: the tiled module must reproduce the portable
    /// module at every shape, including ones the geometry gate keeps on the
    /// portable module in production.
    fn linear_layout_geometries() -> [(usize, usize, usize); 6] {
        [
            (8, 448, 512),
            (6, 96, 320),
            (2, 64, 96),
            (3, 13, 7),
            (1, 1, 1),
            (9, 33, 193),
        ]
    }

    /// A/B body for the RWKV time-mix triple projection: three same-shaped
    /// row-major projections in one dispatch, each with its own serial chain
    /// and its own output buffer.
    fn linear3_forward_case(
        device: &VulkanDevice,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        let family = VendorKernelFamily::Linear3Forward;
        let variant_spirv = family
            .intel_spirv()
            .ok_or_else(|| anyhow::anyhow!("{} has no Intel variant", family.label()))?;
        linear3_forward_case_with(device, rows, input_dim, output_dim, variant_spirv)
    }

    fn linear3_forward_case_with(
        device: &VulkanDevice,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        variant_spirv: &'static [u8],
    ) -> Result<()> {
        let family = VendorKernelFamily::Linear3Forward;
        let (portable, variant) = kernel_pair_with(device, family, variant_spirv)?;
        let input_r = pseudo_random(rows * input_dim, 0x71ed);
        let weight_r = pseudo_random(output_dim * input_dim, 0x7a13);
        let input_k = pseudo_random(rows * input_dim, 0x72ed);
        let weight_k = pseudo_random(output_dim * input_dim, 0x7b13);
        let input_v = pseudo_random(rows * input_dim, 0x73ed);
        let weight_v = pseudo_random(output_dim * input_dim, 0x7c13);
        let xr = GpuBuffer::from_f32(device, &input_r)?;
        let wr = GpuBuffer::from_f32(device, &weight_r)?;
        let xk = GpuBuffer::from_f32(device, &input_k)?;
        let wk = GpuBuffer::from_f32(device, &weight_k)?;
        let xv = GpuBuffer::from_f32(device, &input_v)?;
        let wv = GpuBuffer::from_f32(device, &weight_v)?;
        let r = GpuBuffer::zeros_f32(device, rows * output_dim)?;
        let k = GpuBuffer::zeros_f32(device, rows * output_dim)?;
        let v = GpuBuffer::zeros_f32(device, rows * output_dim)?;
        let push = [rows as u32, input_dim as u32, output_dim as u32];
        let grid = linear_grid(rows, output_dim);
        let lanes = rows * output_dim;
        let outcome = ab_dispatch(
            device,
            &portable,
            &variant,
            &[
                (&xr, &input_r),
                (&wr, &weight_r),
                (&xk, &input_k),
                (&wk, &weight_k),
                (&xv, &input_v),
                (&wv, &weight_v),
            ],
            &[&xr, &wr, &xk, &wk, &xv, &wv, &r, &k, &v],
            &push,
            grid,
            &[(&r, lanes), (&k, lanes), (&v, lanes)],
            TIMED_DISPATCHES,
        )?;
        assert_bit_exact(
            family,
            &outcome,
            &format!("rows={rows},k={input_dim},n={output_dim}"),
        );
        Ok(())
    }

    fn linear3_forward_matches_portable(device: &VulkanDevice) -> Result<()> {
        for (rows, input_dim, output_dim) in linear_layout_geometries() {
            linear3_forward_case(device, rows, input_dim, output_dim)?;
        }
        Ok(())
    }

    fn linear_forward_matches_portable(device: &VulkanDevice) -> Result<()> {
        for (rows, input_dim, output_dim) in linear_layout_geometries() {
            linear_layout_case(
                device,
                VendorKernelFamily::LinearForward,
                rows,
                input_dim,
                output_dim,
            )?;
        }
        Ok(())
    }

    fn linear_bias_forward_matches_portable(device: &VulkanDevice) -> Result<()> {
        for (rows, input_dim, output_dim) in linear_layout_geometries() {
            linear_layout_case(
                device,
                VendorKernelFamily::LinearBiasForward,
                rows,
                input_dim,
                output_dim,
            )?;
        }
        Ok(())
    }

    fn linear_residual_forward_matches_portable(device: &VulkanDevice) -> Result<()> {
        for (rows, input_dim, output_dim) in linear_layout_geometries() {
            linear_layout_case(
                device,
                VendorKernelFamily::LinearResidualForward,
                rows,
                input_dim,
                output_dim,
            )?;
        }
        Ok(())
    }
}
