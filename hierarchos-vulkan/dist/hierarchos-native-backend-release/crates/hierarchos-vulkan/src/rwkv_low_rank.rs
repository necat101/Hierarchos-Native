use std::{
    collections::HashMap,
    path::Path,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::mixed_precision::VulkanParameterStorageMirror;
use crate::rwkv_optimizer::{RwkvDecayClass, RwkvTrainableRef};
use crate::{read_f32_tensor, vulkan, GpuBuffer, VulkanDevice, VulkanParameterStorageFormat};

const TIME_MIX_FORWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_time_mix3_forward.spv");
const TIME_MIX_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_time_mix3_backward.spv");
const TIME_MIX_BACKWARD_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_wg32.spv");
const TIME_MIX_BACKWARD_WG128_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_wg128.spv");
const TIME_MIX_BACKWARD_FUSED_ADD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add.spv");
const TIME_MIX_BACKWARD_FUSED_ADD_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_wg32.spv");
const TIME_MIX_BACKWARD_FUSED_ADD_WG128_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_wg128.spv");
const TIME_MIX_BACKWARD_FUSED_ADD_OUTER_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_outer.spv");
const TIME_MIX_BACKWARD_FUSED_ADD_OUTER_WG32_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_outer_wg32.spv");
const TIME_MIX_BACKWARD_FUSED_ADD_OUTER_WG128_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_outer_wg128.spv");
const LOW_RANK_PRODUCER_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_low_rank_producer_forward_fused.spv");
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_low_rank_producer_forward_fused.spv");
const LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_low_rank_producer_forward_fused_fp16_packed.spv");
const LOW_RANK_FULL_FORWARD_FUSED_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_low_rank_full_forward_fused_fp16_packed.spv");
const LOW_RANK_FULL_FORWARD_FUSED_FP16_PACKED_SUBGROUP_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_low_rank_full_forward_fused_fp16_packed_subgroup.spv");
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed.spv");
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_CACHED_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed_cached.spv");
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_RANK128_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128.spv");
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_TWO_ROWS_SPV: &[u8] = include_bytes!(
    "../shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows.spv"
);
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_TWO_ROWS_WIDE_SPV: &[u8] = include_bytes!(
    "../shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide.spv"
);
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_FOUR_ROWS_SPV: &[u8] = include_bytes!(
    "../shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows.spv"
);
const HIERARCHOS_RWKV_LOW_RANK_DISABLE_LN_FORWARD_AUTOTUNE_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_DISABLE_LN_FORWARD_AUTOTUNE";
const HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG";
const PARAMETER_MATMUL_FORWARD_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_forward.spv");
const PARAMETER_MATMUL_FORWARD_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_forward_fp16_packed.spv");
const PARAMETER_MATMUL_BIAS_FORWARD_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_bias_forward.spv");
const PARAMETER_MATMUL_BIAS_FORWARD_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_bias_forward_fp16_packed.spv");
const PARAMETER_MATMUL_INPUT_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_input_grad.spv");
const PARAMETER_MATMUL_INPUT_GRAD_FP16_PACKED_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_input_grad_fp16_packed.spv");
const PARAMETER_MATMUL_INPUT_GRAD_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_input_grad_fp16_native_compute.spv");
const PARAMETER_MATMUL_WEIGHT_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_weight_grad.spv");
const PARAMETER_MATMUL_WEIGHT_GRAD_FP16_NATIVE_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_weight_grad_fp16_native_compute.spv");
const PARAMETER_MATMUL_WEIGHT_GRAD_FP16_WIDENED_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_weight_grad_fp16_widened_compute.spv");
const PARAMETER_MATMUL_WEIGHT_GRAD_FP16_COMPENSATED_COMPUTE_SPV: &[u8] =
    include_bytes!("../shaders/parameter_matmul_weight_grad_fp16_compensated_compute.spv");
const BIAS_GRAD_SPV: &[u8] = include_bytes!("../shaders/bias_grad.spv");
const SIGMOID_FORWARD_SPV: &[u8] = include_bytes!("../shaders/sigmoid_forward.spv");
const SIGMOID_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/sigmoid_backward.spv");
const TANH_FORWARD_SPV: &[u8] = include_bytes!("../shaders/tanh_forward.spv");
const TANH_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/tanh_backward.spv");
const DECAY_FORWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_decay_forward.spv");
const DECAY_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_decay_backward.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MixPush {
    batch: u32,
    width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulPush {
    rows: u32,
    input_dim: u32,
    output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LowRankProducerPush {
    rows: u32,
    width: u32,
    w_rank: u32,
    a_rank: u32,
    g_rank: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormLowRankProducerPush {
    rows: u32,
    width: u32,
    w_rank: u32,
    a_rank: u32,
    g_rank: u32,
    eps: f32,
}

struct LayerNormProducerInput<'a> {
    x: &'a GpuBuffer,
    weight: &'a GpuBuffer,
    bias: &'a GpuBuffer,
    mean: &'a GpuBuffer,
    rstd: &'a GpuBuffer,
    eps: f32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum LayerNormLowRankForwardTopology {
    OneRow,
    OneRowCached,
    OneRowRank128,
    TwoRows,
    TwoRowsWide,
    FourRows,
}

impl LayerNormLowRankForwardTopology {
    fn label(self) -> &'static str {
        match self {
            Self::OneRow => "ln-low-rank-fp16-one-row",
            Self::OneRowCached => "ln-low-rank-fp16-one-row-cached",
            Self::OneRowRank128 => "ln-low-rank-fp16-one-row-rank128",
            Self::TwoRows => "ln-low-rank-fp16-two-rows",
            Self::TwoRowsWide => "ln-low-rank-fp16-two-rows-wide",
            Self::FourRows => "ln-low-rank-fp16-four-rows",
        }
    }

    fn rows_per_workgroup(self) -> usize {
        match self {
            Self::OneRow | Self::OneRowCached | Self::OneRowRank128 => 1,
            Self::TwoRows | Self::TwoRowsWide => 2,
            Self::FourRows => 4,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LayerNormLowRankForwardAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    width: usize,
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    batch_pairs: usize,
    has_unpaired_tail: bool,
}

#[derive(Clone, Copy, Debug)]
struct LayerNormLowRankForwardDecision {
    topology: LayerNormLowRankForwardTopology,
    autotuned: bool,
}

struct LayerNormLowRankForwardOutputs<'a> {
    x_norm: &'a GpuBuffer,
    xw: &'a GpuBuffer,
    xa: &'a GpuBuffer,
    xg: &'a GpuBuffer,
    w_hidden: &'a GpuBuffer,
    a_hidden: &'a GpuBuffer,
    g_hidden: &'a GpuBuffer,
    w_tanh: &'a GpuBuffer,
    w_pre: &'a GpuBuffer,
    w: &'a GpuBuffer,
    a_pre: &'a GpuBuffer,
    a: &'a GpuBuffer,
    g_sigmoid: &'a GpuBuffer,
    g: &'a GpuBuffer,
}

struct LayerNormLowRankForwardProbeBuffers {
    x: GpuBuffer,
    previous: GpuBuffer,
    x_norm: GpuBuffer,
    mean: GpuBuffer,
    rstd: GpuBuffer,
    xw: GpuBuffer,
    xa: GpuBuffer,
    xg: GpuBuffer,
    w_hidden: GpuBuffer,
    a_hidden: GpuBuffer,
    g_hidden: GpuBuffer,
    w_tanh: GpuBuffer,
    w_pre: GpuBuffer,
    w: GpuBuffer,
    a_pre: GpuBuffer,
    a: GpuBuffer,
    g_sigmoid: GpuBuffer,
    g: GpuBuffer,
}

static LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<LayerNormLowRankForwardAutotuneKey, LayerNormLowRankForwardDecision>>,
> = OnceLock::new();

fn select_layer_norm_low_rank_forward_topology(
    timings: &[(LayerNormLowRankForwardTopology, f64)],
    structural_default: LayerNormLowRankForwardTopology,
) -> LayerNormLowRankForwardTopology {
    let Some((_, default_ms)) = timings
        .iter()
        .find(|(topology, _)| *topology == structural_default)
    else {
        return structural_default;
    };
    let Some(&(best, best_ms)) = timings
        .iter()
        .filter(|(_, ms)| ms.is_finite() && *ms > 0.0)
        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
    else {
        return structural_default;
    };

    // Two-row ownership is the established compact-width default. Require a
    // material (>2%) win before a device-specific probe displaces it so submit
    // jitter cannot turn effectively equivalent kernels into schedule churn.
    if best_ms < *default_ms * 0.98 {
        best
    } else {
        structural_default
    }
}

/// Dispatch topology for the shared-input fan-in at the end of the a/w/g
/// backward graph. The math is identical across variants; only where the
/// low-rank contribution is accumulated changes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub(crate) enum RwkvLowRankFanInSchedule {
    /// Materialize the low-rank input adjoints, then let the caller add them to
    /// the recurrent/projection input adjoints with standalone vector kernels.
    Split,
    /// Fold the recurrent/projection input adjoints into the a/w/g time-mix
    /// backward kernel.
    FusedBase,
    /// Also fold the enclosing cell's additional normalized-input adjoint into
    /// the same a/w/g time-mix backward dispatch.
    FusedOuter,
}

impl RwkvLowRankFanInSchedule {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Split => "low-rank-split-fan-in",
            Self::FusedBase => "low-rank-fused-base-fan-in",
            Self::FusedOuter => "low-rank-fused-outer-fan-in",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RwkvLowRankFirstStageArm {
    Portable,
    SubgroupPackedShare,
}

impl RwkvLowRankFirstStageArm {
    const fn label(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::SubgroupPackedShare => "subgroup-packed-share",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasPush {
    rows: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

#[derive(Debug)]
pub struct RwkvLowRankResult {
    pub a: Vec<f32>,
    pub w: Vec<f32>,
    pub g: Vec<f32>,
    pub grad_x_norm: Vec<f32>,
    pub grad_previous: Vec<f32>,
    pub grad_mix_w: Vec<f32>,
    pub grad_mix_a: Vec<f32>,
    pub grad_mix_g: Vec<f32>,
    pub grad_w0: Vec<f32>,
    pub grad_w1: Vec<f32>,
    pub grad_w2: Vec<f32>,
    pub grad_a0: Vec<f32>,
    pub grad_a1: Vec<f32>,
    pub grad_a2: Vec<f32>,
    pub grad_g1: Vec<f32>,
    pub grad_g2: Vec<f32>,
}

/// Arithmetic used by the RWKV low-rank matrix parameter-gradient (`dW`)
/// reduction. All variants write the same canonical FP32 gradient buffers and
/// therefore preserve the PyTorch/SafeTensors optimizer/checkpoint boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RwkvLowRankParameterGradArithmetic {
    /// Canonical FP32 multiply/FMA and FP32 accumulation.
    #[default]
    Fp32,
    /// Round both operands to FP16, multiply in FP16, widen the product, and
    /// accumulate in FP32.
    NativeFp16,
    /// Round both operands to FP16, widen the rounded operands, then multiply
    /// and accumulate in FP32.
    NativeFp16WidenedProduct,
    /// Split each FP32 operand into high/low FP16 terms and reconstruct the
    /// dominant product with three native FP16 multiplies before FP32
    /// accumulation.
    NativeFp16CompensatedOperands,
}

impl RwkvLowRankParameterGradArithmetic {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::NativeFp16 => "native-fp16",
            Self::NativeFp16WidenedProduct => "native-fp16-widened-product",
            Self::NativeFp16CompensatedOperands => "native-fp16-compensated-operands",
        }
    }

    /// Native half multiplies issued for each mathematical dW multiply-add.
    /// The widened-product arm quantizes operands to half but performs its
    /// product in FP32, so its native-half multiply count is zero.
    pub const fn native_fp16_products_per_mac(self) -> usize {
        match self {
            Self::NativeFp16 => 1,
            Self::NativeFp16CompensatedOperands => 3,
            Self::Fp32 | Self::NativeFp16WidenedProduct => 0,
        }
    }
}

/// Result of an isolated low-rank dW throughput qualification. The timed
/// region reuses resident buffers and contains only repeated dW dispatches in
/// one queue submission; the final validation readback is deliberately outside
/// that region.
#[derive(Debug)]
pub struct RwkvLowRankWeightGradBenchmark {
    pub arithmetic: RwkvLowRankParameterGradArithmetic,
    pub rows: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub elapsed_seconds: f64,
    pub dispatches_per_second: f64,
    pub macs_per_second: f64,
    pub native_fp16_products_per_second: f64,
    /// Logical device-resident bytes touched by the kernel fixture itself:
    /// input + incoming adjoint + dW destination. This excludes the validation
    /// readback buffer.
    pub kernel_resident_bytes: usize,
    /// Exact logical live-buffer increase reported by Hierarchos' Vulkan
    /// allocator while the benchmark fixture and validation readback are live.
    pub allocator_live_buffer_bytes_delta: usize,
    /// Additional pooled VkDeviceMemory reserved for this fixture. This can be
    /// zero when an existing arena block has sufficient reusable slack.
    pub allocator_reserved_bytes_delta: usize,
    pub allocator_driver_allocation_count_delta: usize,
    /// Final dW from an untimed validation dispatch using the same mode.
    pub gradient: Vec<f32>,
}

/// Compact matrix mirrors used by the RWKV low-rank consumer specialization.
/// Vector/bias parameters remain FP32; the six bandwidth-heavy matrices are
/// stored as packed IEEE FP16 while all arithmetic and gradient accumulation
/// remain FP32.
#[derive(Clone)]
pub(crate) struct RwkvLowRankFp16ParameterMirrors {
    pub w1: VulkanParameterStorageMirror,
    pub w2: VulkanParameterStorageMirror,
    pub a1: VulkanParameterStorageMirror,
    pub a2: VulkanParameterStorageMirror,
    pub g1: VulkanParameterStorageMirror,
    pub g2: VulkanParameterStorageMirror,
}

/// Vulkan-native RWKV-v8 low-rank `a`, `w`, and `g` branches.
///
/// The parameter matrices retain PyTorch's raw `nn.Parameter` layouts used by
/// Hierarchos: first-stage matrices are `[width, rank]`, second-stage matrices
/// are `[rank, width]`, and the operation computes row-vector matrix products
/// exactly like `left @ parameter`. No transpose is introduced at load time.
pub struct RwkvLowRankOp {
    device: VulkanDevice,
    width: usize,
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    max_batch: usize,

    mix_w: GpuBuffer,
    mix_a: GpuBuffer,
    mix_g: GpuBuffer,
    w0: GpuBuffer,
    w1: GpuBuffer,
    w2: GpuBuffer,
    a0: GpuBuffer,
    a1: GpuBuffer,
    a2: GpuBuffer,
    g1: GpuBuffer,
    g2: GpuBuffer,

    x_norm: GpuBuffer,
    previous: GpuBuffer,
    grad_a_out: GpuBuffer,
    grad_w_out: GpuBuffer,
    grad_g_out: GpuBuffer,

    xw: GpuBuffer,
    xa: GpuBuffer,
    xg: GpuBuffer,
    w_hidden: GpuBuffer,
    w_tanh: GpuBuffer,
    w_pre: GpuBuffer,
    w: GpuBuffer,
    a_hidden: GpuBuffer,
    a_pre: GpuBuffer,
    a: GpuBuffer,
    g_hidden: GpuBuffer,
    g_sigmoid: GpuBuffer,
    g: GpuBuffer,

    grad_w_pre: GpuBuffer,
    grad_w_tanh: GpuBuffer,
    grad_w_hidden: GpuBuffer,
    grad_xw: GpuBuffer,
    grad_a_pre: GpuBuffer,
    grad_a_hidden: GpuBuffer,
    grad_xa: GpuBuffer,
    grad_g_sigmoid: GpuBuffer,
    grad_g_hidden: GpuBuffer,
    grad_xg: GpuBuffer,
    grad_x_norm: GpuBuffer,
    grad_previous: GpuBuffer,
    grad_mix_w: GpuBuffer,
    grad_mix_a: GpuBuffer,
    grad_mix_g: GpuBuffer,
    grad_w0: GpuBuffer,
    grad_w1: GpuBuffer,
    grad_w2: GpuBuffer,
    grad_a0: GpuBuffer,
    grad_a1: GpuBuffer,
    grad_a2: GpuBuffer,
    grad_g1: GpuBuffer,
    grad_g2: GpuBuffer,

    a_readback: GpuBuffer,
    w_readback: GpuBuffer,
    g_readback: GpuBuffer,
    grad_x_norm_readback: GpuBuffer,
    grad_previous_readback: GpuBuffer,
    grad_mix_w_readback: GpuBuffer,
    grad_mix_a_readback: GpuBuffer,
    grad_mix_g_readback: GpuBuffer,
    grad_w0_readback: GpuBuffer,
    grad_w1_readback: GpuBuffer,
    grad_w2_readback: GpuBuffer,
    grad_a0_readback: GpuBuffer,
    grad_a1_readback: GpuBuffer,
    grad_a2_readback: GpuBuffer,
    grad_g1_readback: GpuBuffer,
    grad_g2_readback: GpuBuffer,

    time_mix_forward: vulkan::ComputeKernel,
    low_rank_producer_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused: Option<vulkan::ComputeKernel>,
    low_rank_producer_forward_fused_fp16_packed: Option<vulkan::ComputeKernel>,
    low_rank_full_forward_fused_fp16_packed: Option<vulkan::ComputeKernel>,
    low_rank_full_forward_first_stage_arm: Option<RwkvLowRankFirstStageArm>,
    layer_norm_low_rank_producer_forward_fused_fp16_packed: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused_fp16_packed_cached: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide:
        Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows: Option<vulkan::ComputeKernel>,
    time_mix_backward: vulkan::ComputeKernel,
    time_mix_backward_wg32: Option<vulkan::ComputeKernel>,
    time_mix_backward_wg128: Option<vulkan::ComputeKernel>,
    time_mix_backward_fused_add: Option<vulkan::ComputeKernel>,
    time_mix_backward_fused_add_wg32: Option<vulkan::ComputeKernel>,
    time_mix_backward_fused_add_wg128: Option<vulkan::ComputeKernel>,
    time_mix_backward_fused_add_outer: Option<vulkan::ComputeKernel>,
    time_mix_backward_fused_add_outer_wg32: Option<vulkan::ComputeKernel>,
    time_mix_backward_fused_add_outer_wg128: Option<vulkan::ComputeKernel>,
    parameter_matmul_forward: vulkan::ComputeKernel,
    parameter_matmul_forward_fp16_packed: vulkan::ComputeKernel,
    parameter_matmul_bias_forward: vulkan::ComputeKernel,
    parameter_matmul_bias_forward_fp16_packed: vulkan::ComputeKernel,
    parameter_matmul_input_grad: vulkan::ComputeKernel,
    parameter_matmul_input_grad_fp16_packed: vulkan::ComputeKernel,
    parameter_matmul_input_grad_fp16_native_compute: Option<vulkan::ComputeKernel>,
    parameter_matmul_weight_grad: vulkan::ComputeKernel,
    parameter_matmul_weight_grad_fp16_native_compute: Option<vulkan::ComputeKernel>,
    parameter_matmul_weight_grad_fp16_widened_compute: Option<vulkan::ComputeKernel>,
    parameter_matmul_weight_grad_fp16_compensated_compute: Option<vulkan::ComputeKernel>,
    bias_grad: vulkan::ComputeKernel,
    sigmoid_forward: vulkan::ComputeKernel,
    sigmoid_backward: vulkan::ComputeKernel,
    tanh_forward: vulkan::ComputeKernel,
    tanh_backward: vulkan::ComputeKernel,
    decay_forward: vulkan::ComputeKernel,
    decay_backward: vulkan::ComputeKernel,
    fp16_parameter_mirrors: Option<RwkvLowRankFp16ParameterMirrors>,
    native_fp16_backward_compute: bool,
    native_fp16_parameter_grad_compute: bool,
    native_fp16_parameter_grad_widen_product: bool,
    native_fp16_parameter_grad_compensated_operands: bool,
    backward_source_scale: f32,
    source_scaled_backward_domain: bool,
}

fn create_time_mix_backward_geometry_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
    binding_count: usize,
    first_write_binding: usize,
    workgroup_size: u32,
) -> Result<Option<vulkan::ComputeKernel>> {
    if !device.supports_storage_buffer_bindings(binding_count as u32)
        || !device.supports_compute_work_group_size_x(workgroup_size)
    {
        return Ok(None);
    }

    let mut accesses = vec![vulkan::BindingAccess::ReadOnly; binding_count];
    accesses[first_write_binding..].fill(vulkan::BindingAccess::WriteOnly);
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &accesses,
        std::mem::size_of::<MixPush>() as u32,
    )?))
}

fn create_low_rank_producer_forward_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
) -> Result<Option<vulkan::ComputeKernel>> {
    if !device.supports_storage_buffer_bindings(14) {
        return Ok(None);
    }
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
        ],
        std::mem::size_of::<LowRankProducerPush>() as u32,
    )?))
}

fn create_low_rank_full_forward_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
) -> Result<Option<vulkan::ComputeKernel>> {
    if w_rank > 128
        || a_rank > 128
        || g_rank > 128
        || !device.supports_storage_buffer_bindings(26)
        || !device.supports_compute_work_group_size_x(128)
    {
        return Ok(None);
    }
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
        ],
        std::mem::size_of::<LowRankProducerPush>() as u32,
    )?))
}

fn create_layer_norm_low_rank_producer_forward_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
) -> Result<Option<vulkan::ComputeKernel>> {
    create_layer_norm_low_rank_producer_forward_kernel_with_max_rank(
        device, spirv, w_rank, a_rank, g_rank, 64,
    )
}

fn create_layer_norm_low_rank_producer_forward_kernel_with_max_rank(
    device: &VulkanDevice,
    spirv: &[u8],
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    max_rank: usize,
) -> Result<Option<vulkan::ComputeKernel>> {
    if w_rank > max_rank
        || a_rank > max_rank
        || g_rank > max_rank
        || !device.supports_storage_buffer_bindings(31)
    {
        return Ok(None);
    }
    Ok(Some(vulkan::ComputeKernel::new_with_access(
        device,
        spirv,
        &[
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::MayWrite,
            vulkan::BindingAccess::ReadOnly,
            vulkan::BindingAccess::MayWrite,
        ],
        std::mem::size_of::<LayerNormLowRankProducerPush>() as u32,
    )?))
}

impl RwkvLowRankOp {
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        prefix: &str,
        max_batch: usize,
    ) -> Result<Self> {
        if prefix.trim().is_empty() {
            bail!("RWKV tensor prefix must not be empty");
        }
        let path = model_dir.as_ref().join("model.safetensors");
        let (mix_w_shape, mix_w) = read_f32_tensor(&path, &format!("{prefix}.x_w"))?;
        let width = vector_width(&mix_w_shape).with_context(|| {
            format!("RWKV tensor {prefix}.x_w must have shape [C] or [1, C], got {mix_w_shape:?}")
        })?;
        let mix_a = read_vector(&path, &format!("{prefix}.x_a"), width)?;
        let mix_g = read_vector(&path, &format!("{prefix}.x_g"), width)?;
        let w0 = read_vector(&path, &format!("{prefix}.w0"), width)?;
        let a0 = read_vector(&path, &format!("{prefix}.a0"), width)?;
        let (w_rank, w1) = read_first_matrix(&path, &format!("{prefix}.w1"), width)?;
        let w2 = read_second_matrix(&path, &format!("{prefix}.w2"), w_rank, width)?;
        let (a_rank, a1) = read_first_matrix(&path, &format!("{prefix}.a1"), width)?;
        let a2 = read_second_matrix(&path, &format!("{prefix}.a2"), a_rank, width)?;
        let (g_rank, g1) = read_first_matrix(&path, &format!("{prefix}.g1"), width)?;
        let g2 = read_second_matrix(&path, &format!("{prefix}.g2"), g_rank, width)?;

        Self::new(
            device, width, w_rank, a_rank, g_rank, max_batch, &mix_w, &mix_a, &mix_g, &w0, &w1,
            &w2, &a0, &a1, &a2, &g1, &g2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: VulkanDevice,
        width: usize,
        w_rank: usize,
        a_rank: usize,
        g_rank: usize,
        max_batch: usize,
        mix_w: &[f32],
        mix_a: &[f32],
        mix_g: &[f32],
        w0: &[f32],
        w1: &[f32],
        w2: &[f32],
        a0: &[f32],
        a1: &[f32],
        a2: &[f32],
        g1: &[f32],
        g2: &[f32],
    ) -> Result<Self> {
        if width == 0 || w_rank == 0 || a_rank == 0 || g_rank == 0 || max_batch == 0 {
            bail!("RWKV low-rank dimensions and max_batch must be positive");
        }
        validate_len("mix_w", mix_w, width)?;
        validate_len("mix_a", mix_a, width)?;
        validate_len("mix_g", mix_g, width)?;
        validate_len("w0", w0, width)?;
        validate_len("a0", a0, width)?;
        validate_len("w1", w1, width * w_rank)?;
        validate_len("w2", w2, w_rank * width)?;
        validate_len("a1", a1, width * a_rank)?;
        validate_len("a2", a2, a_rank * width)?;
        validate_len("g1", g1, width * g_rank)?;
        validate_len("g2", g2, g_rank * width)?;

        let vector_len = max_batch
            .checked_mul(width)
            .context("RWKV vector capacity overflow")?;
        let w_hidden_len = max_batch
            .checked_mul(w_rank)
            .context("RWKV w hidden capacity overflow")?;
        let a_hidden_len = max_batch
            .checked_mul(a_rank)
            .context("RWKV a hidden capacity overflow")?;
        let g_hidden_len = max_batch
            .checked_mul(g_rank)
            .context("RWKV g hidden capacity overflow")?;

        let low_rank_producer_forward_fused = if device.supports_storage_buffer_bindings(14) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                LOW_RANK_PRODUCER_FORWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LowRankProducerPush>() as u32,
            )?)
        } else {
            None
        };
        let layer_norm_low_rank_producer_forward_fused = if w_rank <= 64
            && a_rank <= 64
            && g_rank <= 64
            && device.supports_storage_buffer_bindings(31)
        {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<LayerNormLowRankProducerPush>() as u32,
            )?)
        } else {
            None
        };
        let low_rank_producer_forward_fused_fp16_packed = create_low_rank_producer_forward_kernel(
            &device,
            LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_SPV,
        )?;
        let (low_rank_full_forward_fused_fp16_packed, low_rank_full_forward_first_stage_arm) =
            if std::env::var_os("HIERARCHOS_RWKV_LOW_RANK_DISABLE_FULL_FORWARD_FUSION").is_some() {
                (None, None)
            } else {
                // Packed-word subgroup sharing is deliberately opt-in. On the
                // production wave64 Radeon target, removing the duplicate u32
                // W1/A1/G1 loads cost more in shuffle overhead than it saved in
                // memory traffic. Keep the variant available for wave32/NVIDIA
                // profiling without regressing the established portable path.
                let subgroup_packed_share_enabled =
                    std::env::var_os("HIERARCHOS_RWKV_LOW_RANK_ENABLE_SUBGROUP_PACKED_SHARE")
                        .is_some()
                        && device.supports_compute_subgroup_shuffle();
                let first_stage_arm = if subgroup_packed_share_enabled {
                    RwkvLowRankFirstStageArm::SubgroupPackedShare
                } else {
                    RwkvLowRankFirstStageArm::Portable
                };
                let spirv = match first_stage_arm {
                    RwkvLowRankFirstStageArm::Portable => {
                        LOW_RANK_FULL_FORWARD_FUSED_FP16_PACKED_SPV
                    }
                    RwkvLowRankFirstStageArm::SubgroupPackedShare => {
                        LOW_RANK_FULL_FORWARD_FUSED_FP16_PACKED_SUBGROUP_SPV
                    }
                };
                let kernel =
                    create_low_rank_full_forward_kernel(&device, spirv, w_rank, a_rank, g_rank)?;
                let selected_arm = kernel.as_ref().map(|_| first_stage_arm);
                (kernel, selected_arm)
            };
        let layer_norm_low_rank_producer_forward_fused_fp16_packed =
            create_layer_norm_low_rank_producer_forward_kernel(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_SPV,
                w_rank,
                a_rank,
                g_rank,
            )?;
        let layer_norm_low_rank_producer_forward_fused_fp16_packed_cached = if width <= 1024
            && device.max_compute_shared_memory_bytes() >= 5_640
            && std::env::var_os("HIERARCHOS_RWKV_LOW_RANK_DISABLE_CACHED_LN_FORWARD_FUSION")
                .is_none()
        {
            create_layer_norm_low_rank_producer_forward_kernel(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_CACHED_SPV,
                w_rank,
                a_rank,
                g_rank,
            )?
        } else {
            None
        };
        let layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128 =
            if (w_rank > 64 || a_rank > 64 || g_rank > 64)
                && w_rank <= 128
                && a_rank <= 128
                && g_rank <= 128
                && device.supports_compute_work_group_size_x(64)
                && std::env::var_os("HIERARCHOS_RWKV_LOW_RANK_DISABLE_RANK128_LN_FORWARD_FUSION")
                    .is_none()
            {
                create_layer_norm_low_rank_producer_forward_kernel_with_max_rank(
                    &device,
                    LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_RANK128_SPV,
                    w_rank,
                    a_rank,
                    g_rank,
                    128,
                )?
            } else {
                None
            };
        let layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows = if width <= 32
            && w_rank <= 32
            && a_rank <= 32
            && g_rank <= 32
            && device.supports_compute_work_group_size_x(64)
            && std::env::var_os("HIERARCHOS_RWKV_LOW_RANK_DISABLE_TWO_ROW_LN_FORWARD_FUSION")
                .is_none()
        {
            create_layer_norm_low_rank_producer_forward_kernel(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_TWO_ROWS_SPV,
                w_rank,
                a_rank,
                g_rank,
            )?
        } else {
            None
        };
        let layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide = if width > 32
            && w_rank <= 64
            && a_rank <= 64
            && g_rank <= 64
            && device.supports_compute_work_group_size_x(128)
            && std::env::var_os("HIERARCHOS_RWKV_LOW_RANK_DISABLE_TWO_ROW_WIDE_LN_FORWARD_FUSION")
                .is_none()
        {
            create_layer_norm_low_rank_producer_forward_kernel(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_TWO_ROWS_WIDE_SPV,
                w_rank,
                a_rank,
                g_rank,
            )?
        } else {
            None
        };
        let layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows = if width <= 32
            && w_rank <= 32
            && a_rank <= 32
            && g_rank <= 32
            && device.supports_compute_work_group_size_x(128)
            && std::env::var_os("HIERARCHOS_RWKV_LOW_RANK_DISABLE_FOUR_ROW_LN_FORWARD_FUSION")
                .is_none()
        {
            create_layer_norm_low_rank_producer_forward_kernel(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_FP16_PACKED_FOUR_ROWS_SPV,
                w_rank,
                a_rank,
                g_rank,
            )?
        } else {
            None
        };

        let time_mix_backward_wg32 = create_time_mix_backward_geometry_kernel(
            &device,
            TIME_MIX_BACKWARD_WG32_SPV,
            13,
            8,
            32,
        )?;
        let time_mix_backward_wg128 = create_time_mix_backward_geometry_kernel(
            &device,
            TIME_MIX_BACKWARD_WG128_SPV,
            13,
            8,
            128,
        )?;
        let time_mix_backward_fused_add_wg32 = create_time_mix_backward_geometry_kernel(
            &device,
            TIME_MIX_BACKWARD_FUSED_ADD_WG32_SPV,
            15,
            10,
            32,
        )?;
        let time_mix_backward_fused_add_wg128 = create_time_mix_backward_geometry_kernel(
            &device,
            TIME_MIX_BACKWARD_FUSED_ADD_WG128_SPV,
            15,
            10,
            128,
        )?;
        let time_mix_backward_fused_add_outer_wg32 = create_time_mix_backward_geometry_kernel(
            &device,
            TIME_MIX_BACKWARD_FUSED_ADD_OUTER_WG32_SPV,
            16,
            11,
            32,
        )?;
        let time_mix_backward_fused_add_outer_wg128 = create_time_mix_backward_geometry_kernel(
            &device,
            TIME_MIX_BACKWARD_FUSED_ADD_OUTER_WG128_SPV,
            16,
            11,
            128,
        )?;

        let time_mix_backward_fused_add = if device.supports_storage_buffer_bindings(15) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                TIME_MIX_BACKWARD_FUSED_ADD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<MixPush>() as u32,
            )?)
        } else {
            None
        };
        let time_mix_backward_fused_add_outer = if device.supports_storage_buffer_bindings(16) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                TIME_MIX_BACKWARD_FUSED_ADD_OUTER_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<MixPush>() as u32,
            )?)
        } else {
            None
        };

        Ok(Self {
            time_mix_forward: vulkan::ComputeKernel::new(
                &device,
                TIME_MIX_FORWARD_SPV,
                8,
                std::mem::size_of::<MixPush>() as u32,
            )?,
            low_rank_producer_forward_fused,
            layer_norm_low_rank_producer_forward_fused,
            low_rank_producer_forward_fused_fp16_packed,
            low_rank_full_forward_fused_fp16_packed,
            low_rank_full_forward_first_stage_arm,
            layer_norm_low_rank_producer_forward_fused_fp16_packed,
            layer_norm_low_rank_producer_forward_fused_fp16_packed_cached,
            layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128,
            layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows,
            layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide,
            layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows,
            time_mix_backward: vulkan::ComputeKernel::new(
                &device,
                TIME_MIX_BACKWARD_SPV,
                13,
                std::mem::size_of::<MixPush>() as u32,
            )?,
            time_mix_backward_wg32,
            time_mix_backward_wg128,
            time_mix_backward_fused_add,
            time_mix_backward_fused_add_wg32,
            time_mix_backward_fused_add_wg128,
            time_mix_backward_fused_add_outer,
            time_mix_backward_fused_add_outer_wg32,
            time_mix_backward_fused_add_outer_wg128,
            parameter_matmul_forward: vulkan::ComputeKernel::new(
                &device,
                PARAMETER_MATMUL_FORWARD_SPV,
                3,
                std::mem::size_of::<MatmulPush>() as u32,
            )?,
            parameter_matmul_forward_fp16_packed: vulkan::ComputeKernel::new_with_access(
                &device,
                PARAMETER_MATMUL_FORWARD_FP16_PACKED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<MatmulPush>() as u32,
            )?,
            parameter_matmul_bias_forward: vulkan::ComputeKernel::new_with_access(
                &device,
                PARAMETER_MATMUL_BIAS_FORWARD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<MatmulPush>() as u32,
            )?,
            parameter_matmul_bias_forward_fp16_packed: vulkan::ComputeKernel::new_with_access(
                &device,
                PARAMETER_MATMUL_BIAS_FORWARD_FP16_PACKED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<MatmulPush>() as u32,
            )?,
            parameter_matmul_input_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                PARAMETER_MATMUL_INPUT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<MatmulPush>() as u32,
            )?,
            parameter_matmul_input_grad_fp16_packed: vulkan::ComputeKernel::new_with_access(
                &device,
                PARAMETER_MATMUL_INPUT_GRAD_FP16_PACKED_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<MatmulPush>() as u32,
            )?,
            parameter_matmul_input_grad_fp16_native_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        PARAMETER_MATMUL_INPUT_GRAD_FP16_NATIVE_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<MatmulPush>() as u32,
                    )
                })
                .transpose()?,
            parameter_matmul_weight_grad: vulkan::ComputeKernel::new_with_access(
                &device,
                PARAMETER_MATMUL_WEIGHT_GRAD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<MatmulPush>() as u32,
            )?,
            parameter_matmul_weight_grad_fp16_native_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        PARAMETER_MATMUL_WEIGHT_GRAD_FP16_NATIVE_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<MatmulPush>() as u32,
                    )
                })
                .transpose()?,
            parameter_matmul_weight_grad_fp16_widened_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        PARAMETER_MATMUL_WEIGHT_GRAD_FP16_WIDENED_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<MatmulPush>() as u32,
                    )
                })
                .transpose()?,
            parameter_matmul_weight_grad_fp16_compensated_compute: device
                .mixed_precision_capabilities()
                .native_fp16_storage_compute_ready()
                .then(|| {
                    vulkan::ComputeKernel::new_with_access(
                        &device,
                        PARAMETER_MATMUL_WEIGHT_GRAD_FP16_COMPENSATED_COMPUTE_SPV,
                        &[
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::ReadOnly,
                            vulkan::BindingAccess::MayWrite,
                        ],
                        std::mem::size_of::<MatmulPush>() as u32,
                    )
                })
                .transpose()?,
            bias_grad: vulkan::ComputeKernel::new(
                &device,
                BIAS_GRAD_SPV,
                2,
                std::mem::size_of::<BiasPush>() as u32,
            )?,
            sigmoid_forward: vulkan::ComputeKernel::new(
                &device,
                SIGMOID_FORWARD_SPV,
                2,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            sigmoid_backward: vulkan::ComputeKernel::new(
                &device,
                SIGMOID_BACKWARD_SPV,
                3,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            tanh_forward: vulkan::ComputeKernel::new(
                &device,
                TANH_FORWARD_SPV,
                2,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            tanh_backward: vulkan::ComputeKernel::new(
                &device,
                TANH_BACKWARD_SPV,
                3,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            decay_forward: vulkan::ComputeKernel::new(
                &device,
                DECAY_FORWARD_SPV,
                2,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            decay_backward: vulkan::ComputeKernel::new(
                &device,
                DECAY_BACKWARD_SPV,
                3,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            fp16_parameter_mirrors: None,
            native_fp16_backward_compute: false,
            native_fp16_parameter_grad_compute: false,
            native_fp16_parameter_grad_widen_product: false,
            native_fp16_parameter_grad_compensated_operands: false,
            backward_source_scale: 1.0,
            source_scaled_backward_domain: false,
            mix_w: GpuBuffer::from_f32(&device, mix_w)?,
            mix_a: GpuBuffer::from_f32(&device, mix_a)?,
            mix_g: GpuBuffer::from_f32(&device, mix_g)?,
            w0: GpuBuffer::from_f32(&device, w0)?,
            w1: GpuBuffer::from_f32(&device, w1)?,
            w2: GpuBuffer::from_f32(&device, w2)?,
            a0: GpuBuffer::from_f32(&device, a0)?,
            a1: GpuBuffer::from_f32(&device, a1)?,
            a2: GpuBuffer::from_f32(&device, a2)?,
            g1: GpuBuffer::from_f32(&device, g1)?,
            g2: GpuBuffer::from_f32(&device, g2)?,
            x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_a_out: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_w_out: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_g_out: GpuBuffer::zeros_f32(&device, vector_len)?,
            xw: GpuBuffer::zeros_f32(&device, vector_len)?,
            xa: GpuBuffer::zeros_f32(&device, vector_len)?,
            xg: GpuBuffer::zeros_f32(&device, vector_len)?,
            w_hidden: GpuBuffer::zeros_f32(&device, w_hidden_len)?,
            w_tanh: GpuBuffer::zeros_f32(&device, w_hidden_len)?,
            w_pre: GpuBuffer::zeros_f32(&device, vector_len)?,
            w: GpuBuffer::zeros_f32(&device, vector_len)?,
            a_hidden: GpuBuffer::zeros_f32(&device, a_hidden_len)?,
            a_pre: GpuBuffer::zeros_f32(&device, vector_len)?,
            a: GpuBuffer::zeros_f32(&device, vector_len)?,
            g_hidden: GpuBuffer::zeros_f32(&device, g_hidden_len)?,
            g_sigmoid: GpuBuffer::zeros_f32(&device, g_hidden_len)?,
            g: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_w_pre: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_w_tanh: GpuBuffer::zeros_f32(&device, w_hidden_len)?,
            grad_w_hidden: GpuBuffer::zeros_f32(&device, w_hidden_len)?,
            grad_xw: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_a_pre: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_a_hidden: GpuBuffer::zeros_f32(&device, a_hidden_len)?,
            grad_xa: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_g_sigmoid: GpuBuffer::zeros_f32(&device, g_hidden_len)?,
            grad_g_hidden: GpuBuffer::zeros_f32(&device, g_hidden_len)?,
            grad_xg: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_mix_w: GpuBuffer::zeros_f32(&device, width)?,
            grad_mix_a: GpuBuffer::zeros_f32(&device, width)?,
            grad_mix_g: GpuBuffer::zeros_f32(&device, width)?,
            grad_w0: GpuBuffer::zeros_f32(&device, width)?,
            grad_w1: GpuBuffer::zeros_f32(&device, width * w_rank)?,
            grad_w2: GpuBuffer::zeros_f32(&device, w_rank * width)?,
            grad_a0: GpuBuffer::zeros_f32(&device, width)?,
            grad_a1: GpuBuffer::zeros_f32(&device, width * a_rank)?,
            grad_a2: GpuBuffer::zeros_f32(&device, a_rank * width)?,
            grad_g1: GpuBuffer::zeros_f32(&device, width * g_rank)?,
            grad_g2: GpuBuffer::zeros_f32(&device, g_rank * width)?,
            a_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            w_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            g_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_x_norm_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_previous_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_mix_w_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_mix_a_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_mix_g_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_w0_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_w1_readback: GpuBuffer::zeros_host_f32(&device, width * w_rank)?,
            grad_w2_readback: GpuBuffer::zeros_host_f32(&device, w_rank * width)?,
            grad_a0_readback: GpuBuffer::zeros_host_f32(&device, width)?,
            grad_a1_readback: GpuBuffer::zeros_host_f32(&device, width * a_rank)?,
            grad_a2_readback: GpuBuffer::zeros_host_f32(&device, a_rank * width)?,
            grad_g1_readback: GpuBuffer::zeros_host_f32(&device, width * g_rank)?,
            grad_g2_readback: GpuBuffer::zeros_host_f32(&device, g_rank * width)?,
            device,
            width,
            w_rank,
            a_rank,
            g_rank,
            max_batch,
        })
    }

    pub fn forward_backward(
        &mut self,
        batch: usize,
        x_norm: &[f32],
        previous: &[f32],
        grad_a: &[f32],
        grad_w: &[f32],
        grad_g: &[f32],
    ) -> Result<RwkvLowRankResult> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV low-rank batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let vector_len = batch * self.width;
        validate_len("x_norm", x_norm, vector_len)?;
        validate_len("previous", previous, vector_len)?;
        validate_len("grad_a", grad_a, vector_len)?;
        validate_len("grad_w", grad_w, vector_len)?;
        validate_len("grad_g", grad_g, vector_len)?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.x_norm, x_norm)?;
        commands.upload_f32(&self.previous, previous)?;
        commands.upload_f32(&self.grad_a_out, grad_a)?;
        commands.upload_f32(&self.grad_w_out, grad_w)?;
        commands.upload_f32(&self.grad_g_out, grad_g)?;
        self.record_forward(&mut commands, batch, &self.x_norm, &self.previous)?;
        self.record_backward(
            &mut commands,
            batch,
            &self.x_norm,
            &self.previous,
            &self.grad_a_out,
            &self.grad_w_out,
            &self.grad_g_out,
        )?;
        self.record_readback(&mut commands, batch)?;
        commands.submit()?;
        self.read_result(batch)
    }

    fn weight_grad_kernel_for_arithmetic(
        &self,
        arithmetic: RwkvLowRankParameterGradArithmetic,
    ) -> Result<&vulkan::ComputeKernel> {
        match arithmetic {
            RwkvLowRankParameterGradArithmetic::Fp32 => Ok(&self.parameter_matmul_weight_grad),
            RwkvLowRankParameterGradArithmetic::NativeFp16 => self
                .parameter_matmul_weight_grad_fp16_native_compute
                .as_ref()
                .context("device cannot create native-FP16 RWKV low-rank dW"),
            RwkvLowRankParameterGradArithmetic::NativeFp16WidenedProduct => self
                .parameter_matmul_weight_grad_fp16_widened_compute
                .as_ref()
                .context("device cannot create widened-product native-FP16 RWKV low-rank dW"),
            RwkvLowRankParameterGradArithmetic::NativeFp16CompensatedOperands => self
                .parameter_matmul_weight_grad_fp16_compensated_compute
                .as_ref()
                .context("device cannot create compensated native-FP16 RWKV low-rank dW"),
        }
    }

    /// Run one isolated low-rank parameter-gradient dispatch with an explicit
    /// arithmetic policy.
    ///
    /// This intentionally bypasses the rest of the RWKV backward graph so a
    /// caller can distinguish local dW arithmetic drift from error accumulated
    /// across recurrent uses. Inputs and the destination remain canonical FP32
    /// buffers for every policy; only the shader's multiply path changes.
    pub fn diagnose_weight_grad(
        &self,
        arithmetic: RwkvLowRankParameterGradArithmetic,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        input: &[f32],
        grad_output: &[f32],
    ) -> Result<Vec<f32>> {
        if rows == 0 || input_dim == 0 || output_dim == 0 {
            bail!("RWKV low-rank dW diagnostic dimensions must all be non-zero");
        }
        validate_len("dW diagnostic input", input, rows * input_dim)?;
        validate_len("dW diagnostic grad_output", grad_output, rows * output_dim)?;

        let kernel = self.weight_grad_kernel_for_arithmetic(arithmetic)?;
        let push = MatmulPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        };
        let groups = [div_ceil_u32(output_dim, 16), div_ceil_u32(input_dim, 16), 1];
        let weight_len = input_dim * output_dim;
        let input_buffer = GpuBuffer::from_f32(&self.device, input)?;
        let grad_output_buffer = GpuBuffer::from_f32(&self.device, grad_output)?;
        let native_grad = GpuBuffer::zeros_f32(&self.device, weight_len)?;
        let native_readback = GpuBuffer::zeros_host_f32(&self.device, weight_len)?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&input_buffer, &grad_output_buffer, &native_grad],
            bytemuck::bytes_of(&push),
            groups,
        )?;
        commands.readback_f32(&native_grad, &native_readback, weight_len)?;
        commands.submit()?;

        native_readback.read_f32(weight_len)
    }

    /// Compatibility wrapper for the original single-dispatch native-FP16 dW
    /// diagnostic.
    pub fn diagnose_native_fp16_weight_grad(
        &self,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        input: &[f32],
        grad_output: &[f32],
    ) -> Result<Vec<f32>> {
        self.diagnose_weight_grad(
            RwkvLowRankParameterGradArithmetic::NativeFp16,
            rows,
            input_dim,
            output_dim,
            input,
            grad_output,
        )
    }

    /// Benchmark one isolated dW arithmetic path with reusable resident
    /// buffers. Warmups and measured dispatches are each recorded into one
    /// submission; command recording is outside the measured interval and the
    /// validation readback is performed afterwards. Set
    /// `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` to additionally emit Vulkan GPU
    /// timestamp-query results for the measured submission.
    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_weight_grad(
        &self,
        arithmetic: RwkvLowRankParameterGradArithmetic,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        input: &[f32],
        grad_output: &[f32],
        warmup_iterations: usize,
        measured_iterations: usize,
    ) -> Result<RwkvLowRankWeightGradBenchmark> {
        if rows == 0 || input_dim == 0 || output_dim == 0 {
            bail!("RWKV low-rank dW benchmark dimensions must all be non-zero");
        }
        if measured_iterations == 0 {
            bail!("RWKV low-rank dW benchmark requires at least one measured iteration");
        }
        validate_len("dW benchmark input", input, rows * input_dim)?;
        validate_len("dW benchmark grad_output", grad_output, rows * output_dim)?;

        let kernel = self.weight_grad_kernel_for_arithmetic(arithmetic)?;
        let weight_len = input_dim
            .checked_mul(output_dim)
            .context("RWKV low-rank dW benchmark weight length overflow")?;
        let input_len = rows
            .checked_mul(input_dim)
            .context("RWKV low-rank dW benchmark input length overflow")?;
        let grad_output_len = rows
            .checked_mul(output_dim)
            .context("RWKV low-rank dW benchmark grad-output length overflow")?;
        let kernel_elements = input_len
            .checked_add(grad_output_len)
            .and_then(|value| value.checked_add(weight_len))
            .context("RWKV low-rank dW benchmark resident element count overflow")?;
        let kernel_resident_bytes = kernel_elements
            .checked_mul(std::mem::size_of::<f32>())
            .context("RWKV low-rank dW benchmark resident byte count overflow")?;
        let macs_per_dispatch = rows
            .checked_mul(input_dim)
            .and_then(|value| value.checked_mul(output_dim))
            .context("RWKV low-rank dW benchmark MAC count overflow")?;
        let push = MatmulPush {
            rows: u32::try_from(rows).context("dW benchmark rows exceed u32 range")?,
            input_dim: u32::try_from(input_dim)
                .context("dW benchmark input_dim exceeds u32 range")?,
            output_dim: u32::try_from(output_dim)
                .context("dW benchmark output_dim exceeds u32 range")?,
        };
        let groups = [div_ceil_u32(output_dim, 16), div_ceil_u32(input_dim, 16), 1];

        let memory_before = self.device.memory_stats()?;
        let input_buffer = GpuBuffer::from_f32(&self.device, input)?;
        let grad_output_buffer = GpuBuffer::from_f32(&self.device, grad_output)?;
        let grad_weight = GpuBuffer::zeros_f32(&self.device, weight_len)?;
        let grad_readback = GpuBuffer::zeros_host_f32(&self.device, weight_len)?;
        let memory_fixture = self.device.memory_stats()?;

        if warmup_iterations > 0 {
            let mut warmup = vulkan::ComputeBatch::new(&self.device)?;
            for _ in 0..warmup_iterations {
                kernel.record_dispatch(
                    &mut warmup,
                    &[&input_buffer, &grad_output_buffer, &grad_weight],
                    bytemuck::bytes_of(&push),
                    groups,
                )?;
            }
            warmup.submit()?;
        }

        let mut measured = vulkan::ComputeBatch::new(&self.device)?;
        for _ in 0..measured_iterations {
            kernel.record_dispatch(
                &mut measured,
                &[&input_buffer, &grad_output_buffer, &grad_weight],
                bytemuck::bytes_of(&push),
                groups,
            )?;
        }
        let started = Instant::now();
        measured.submit()?;
        let elapsed_seconds = started.elapsed().as_secs_f64();

        let mut validation = vulkan::ComputeBatch::new(&self.device)?;
        kernel.record_dispatch(
            &mut validation,
            &[&input_buffer, &grad_output_buffer, &grad_weight],
            bytemuck::bytes_of(&push),
            groups,
        )?;
        validation.readback_f32(&grad_weight, &grad_readback, weight_len)?;
        validation.submit()?;
        let gradient = grad_readback.read_f32(weight_len)?;

        let dispatches_per_second = measured_iterations as f64 / elapsed_seconds;
        let macs_per_second = macs_per_dispatch as f64 * dispatches_per_second;
        let native_fp16_products_per_second =
            macs_per_second * arithmetic.native_fp16_products_per_mac() as f64;

        Ok(RwkvLowRankWeightGradBenchmark {
            arithmetic,
            rows,
            input_dim,
            output_dim,
            warmup_iterations,
            measured_iterations,
            elapsed_seconds,
            dispatches_per_second,
            macs_per_second,
            native_fp16_products_per_second,
            kernel_resident_bytes,
            allocator_live_buffer_bytes_delta: memory_fixture
                .live_buffer_bytes
                .saturating_sub(memory_before.live_buffer_bytes),
            allocator_reserved_bytes_delta: memory_fixture
                .reserved_bytes
                .saturating_sub(memory_before.reserved_bytes),
            allocator_driver_allocation_count_delta: memory_fixture
                .driver_allocation_count
                .saturating_sub(memory_before.driver_allocation_count),
            gradient,
        })
    }

    /// Record only the low-rank a/w/g forward graph into an existing command
    /// buffer. This is the composition seam used by the fused RWKV cell: the
    /// caller owns x/previous and can feed `a_buffer`, `w_buffer`, and
    /// `g_buffer` directly into downstream Vulkan kernels without a host
    /// round-trip or a second queue submission.
    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
    ) -> Result<()> {
        self.record_forward_inner(commands, batch, x_norm, previous, None)
    }

    pub(crate) fn install_fp16_parameter_mirrors(
        &mut self,
        mirrors: RwkvLowRankFp16ParameterMirrors,
    ) -> Result<()> {
        for (name, mirror, expected_len) in [
            ("w1", &mirrors.w1, self.width * self.w_rank),
            ("w2", &mirrors.w2, self.w_rank * self.width),
            ("a1", &mirrors.a1, self.width * self.a_rank),
            ("a2", &mirrors.a2, self.a_rank * self.width),
            ("g1", &mirrors.g1, self.width * self.g_rank),
            ("g2", &mirrors.g2, self.g_rank * self.width),
        ] {
            if mirror.format() != VulkanParameterStorageFormat::Fp16 {
                bail!("RWKV low-rank {name} mirror must use FP16 storage");
            }
            if mirror.len() != expected_len {
                bail!(
                    "RWKV low-rank {name} mirror has {} elements; expected {expected_len}",
                    mirror.len()
                );
            }
        }
        if self.low_rank_producer_forward_fused.is_some()
            && self.low_rank_producer_forward_fused_fp16_packed.is_none()
        {
            bail!("RWKV FP16 low-rank producer specialization is unavailable on this device");
        }
        if self.layer_norm_low_rank_producer_forward_fused.is_some()
            && self
                .layer_norm_low_rank_producer_forward_fused_fp16_packed
                .is_none()
        {
            bail!("RWKV FP16 LN1/low-rank producer specialization is unavailable on this device");
        }
        self.fp16_parameter_mirrors = Some(mirrors);
        Ok(())
    }

    pub(crate) fn fp16_parameter_storage_active(&self) -> bool {
        self.fp16_parameter_mirrors.is_some()
    }

    /// Enable FP16 multiplies for low-rank input adjoints while keeping gradient
    /// destinations, optimizer masters/moments, and recurrent/state-sensitive
    /// operations in FP32. Unscaled callers retain the conservative
    /// first-stage-only experiment and FP32 parameter-gradient products. A
    /// trainer marked as a source-scaled backward domain additionally runs the
    /// w2/a2/g2 -> w1/a1/g1 inter-stage dX multiplies in native half.
    pub(crate) fn enable_native_fp16_backward_compute(&mut self) -> Result<()> {
        self.fp16_parameter_mirrors
            .as_ref()
            .context("native-FP16 RWKV low-rank backward requires FP16 parameter mirrors")?;
        self.parameter_matmul_input_grad_fp16_native_compute
            .as_ref()
            .context("device cannot create native-FP16 RWKV low-rank dX")?;
        self.native_fp16_backward_compute = true;
        Ok(())
    }

    /// Experimental source-scaled low-rank parameter-gradient arithmetic. The
    /// activation and incoming-adjoint product is performed in Float16, widened
    /// immediately, and accumulated into the canonical FP32 gradient tensor.
    /// This is deliberately separate from the established low-rank dX policy so
    /// a failed optimizer-trajectory qualification cannot silently change the
    /// portable checkpoint contract.
    pub(crate) fn enable_native_fp16_parameter_grad_compute(
        &mut self,
        widen_product: bool,
        compensated_operands: bool,
    ) -> Result<()> {
        if widen_product && compensated_operands {
            bail!(
                "native-FP16 RWKV low-rank dW cannot enable widened-product and compensated-operand modes together"
            );
        }
        if compensated_operands {
            self.parameter_matmul_weight_grad_fp16_compensated_compute
                .as_ref()
                .context("device cannot create compensated native-FP16 RWKV low-rank dW")?;
        } else if widen_product {
            self.parameter_matmul_weight_grad_fp16_widened_compute
                .as_ref()
                .context("device cannot create widened-product native-FP16 RWKV low-rank dW")?;
        } else {
            self.parameter_matmul_weight_grad_fp16_native_compute
                .as_ref()
                .context("device cannot create native-FP16 RWKV low-rank dW")?;
        }
        self.native_fp16_parameter_grad_compute = true;
        self.native_fp16_parameter_grad_widen_product = widen_product;
        self.native_fp16_parameter_grad_compensated_operands = compensated_operands;
        Ok(())
    }

    pub(crate) fn configure_backward_source_scale(
        &mut self,
        source_scale: f32,
        source_scaled_backward_domain: bool,
    ) -> Result<()> {
        if !source_scale.is_finite() || source_scale <= 0.0 {
            bail!("RWKV low-rank backward source scale must be finite and positive");
        }
        self.backward_source_scale = source_scale;
        self.source_scaled_backward_domain = source_scaled_backward_domain;
        Ok(())
    }

    pub(crate) fn native_fp16_backward_compute_active(&self) -> bool {
        self.native_fp16_backward_compute
    }

    pub(crate) fn native_fp16_parameter_grad_compute_active(&self) -> bool {
        self.native_fp16_parameter_grad_compute
    }

    pub(crate) fn parameter_grad_arithmetic(&self) -> RwkvLowRankParameterGradArithmetic {
        if !self.native_fp16_parameter_grad_compute {
            RwkvLowRankParameterGradArithmetic::Fp32
        } else if self.native_fp16_parameter_grad_compensated_operands {
            RwkvLowRankParameterGradArithmetic::NativeFp16CompensatedOperands
        } else if self.native_fp16_parameter_grad_widen_product {
            RwkvLowRankParameterGradArithmetic::NativeFp16WidenedProduct
        } else {
            RwkvLowRankParameterGradArithmetic::NativeFp16
        }
    }

    pub(crate) fn fp16_full_forward_first_stage_arm_label(&self) -> Option<&'static str> {
        if !self.fp16_parameter_storage_active() {
            return None;
        }
        self.low_rank_full_forward_first_stage_arm
            .map(RwkvLowRankFirstStageArm::label)
    }

    fn available_fp16_layer_norm_forward_topologies(&self) -> Vec<LayerNormLowRankForwardTopology> {
        let mut topologies = Vec::with_capacity(6);
        if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardTopology::OneRow);
        }
        if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_cached
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardTopology::OneRowCached);
        }
        if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardTopology::OneRowRank128);
        }
        if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardTopology::TwoRows);
        }
        if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardTopology::TwoRowsWide);
        }
        if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardTopology::FourRows);
        }
        topologies
    }

    fn structural_fp16_layer_norm_forward_topology(
        &self,
    ) -> Option<LayerNormLowRankForwardTopology> {
        if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows
            .is_some()
        {
            Some(LayerNormLowRankForwardTopology::TwoRows)
        } else if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows
            .is_some()
        {
            Some(LayerNormLowRankForwardTopology::FourRows)
        } else if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed
            .is_some()
        {
            Some(LayerNormLowRankForwardTopology::OneRow)
        } else if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_cached
            .is_some()
        {
            Some(LayerNormLowRankForwardTopology::OneRowCached)
        } else if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128
            .is_some()
        {
            Some(LayerNormLowRankForwardTopology::OneRowRank128)
        } else if self
            .layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide
            .is_some()
        {
            Some(LayerNormLowRankForwardTopology::TwoRowsWide)
        } else {
            None
        }
    }

    fn fp16_layer_norm_forward_kernel(
        &self,
        topology: LayerNormLowRankForwardTopology,
    ) -> Option<&vulkan::ComputeKernel> {
        match topology {
            LayerNormLowRankForwardTopology::OneRow => self
                .layer_norm_low_rank_producer_forward_fused_fp16_packed
                .as_ref(),
            LayerNormLowRankForwardTopology::OneRowCached => self
                .layer_norm_low_rank_producer_forward_fused_fp16_packed_cached
                .as_ref(),
            LayerNormLowRankForwardTopology::OneRowRank128 => self
                .layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128
                .as_ref(),
            LayerNormLowRankForwardTopology::TwoRows => self
                .layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows
                .as_ref(),
            LayerNormLowRankForwardTopology::TwoRowsWide => self
                .layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide
                .as_ref(),
            LayerNormLowRankForwardTopology::FourRows => self
                .layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows
                .as_ref(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_fp16_layer_norm_forward_topology_into(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        previous: &GpuBuffer,
        norm: &LayerNormProducerInput<'_>,
        mirrors: &RwkvLowRankFp16ParameterMirrors,
        outputs: &LayerNormLowRankForwardOutputs<'_>,
        topology: LayerNormLowRankForwardTopology,
    ) -> Result<()> {
        let kernel = self
            .fp16_layer_norm_forward_kernel(topology)
            .with_context(|| {
                format!(
                    "RWKV FP16 LN1/low-rank forward topology {} is unavailable",
                    topology.label()
                )
            })?;
        let producer_push = LayerNormLowRankProducerPush {
            rows: batch as u32,
            width: self.width as u32,
            w_rank: self.w_rank as u32,
            a_rank: self.a_rank as u32,
            g_rank: self.g_rank as u32,
            eps: norm.eps,
        };
        kernel.record_dispatch(
            commands,
            &[
                norm.x,
                norm.weight,
                norm.bias,
                previous,
                &self.mix_w,
                &self.mix_a,
                &self.mix_g,
                mirrors.w1.packed_storage(),
                mirrors.a1.packed_storage(),
                mirrors.g1.packed_storage(),
                outputs.x_norm,
                norm.mean,
                norm.rstd,
                outputs.xw,
                outputs.xa,
                outputs.xg,
                outputs.w_hidden,
                outputs.a_hidden,
                outputs.g_hidden,
                outputs.w_tanh,
                mirrors.w2.packed_storage(),
                &self.w0,
                outputs.w_pre,
                outputs.w,
                mirrors.a2.packed_storage(),
                &self.a0,
                outputs.a_pre,
                outputs.a,
                outputs.g_sigmoid,
                mirrors.g2.packed_storage(),
                outputs.g,
            ],
            bytemuck::bytes_of(&producer_push),
            [batch.div_ceil(topology.rows_per_workgroup()) as u32, 1, 1],
        )
    }

    fn allocate_fp16_layer_norm_forward_autotune_probe(
        &self,
        batch: usize,
    ) -> Result<LayerNormLowRankForwardProbeBuffers> {
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV LN1/low-rank autotune vector size overflow")?;
        let w_hidden_len = batch
            .checked_mul(self.w_rank)
            .context("RWKV LN1/low-rank autotune w hidden size overflow")?;
        let a_hidden_len = batch
            .checked_mul(self.a_rank)
            .context("RWKV LN1/low-rank autotune a hidden size overflow")?;
        let g_hidden_len = batch
            .checked_mul(self.g_rank)
            .context("RWKV LN1/low-rank autotune g hidden size overflow")?;
        Ok(LayerNormLowRankForwardProbeBuffers {
            x: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            previous: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            x_norm: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            mean: GpuBuffer::zeros_f32(&self.device, batch)?,
            rstd: GpuBuffer::zeros_f32(&self.device, batch)?,
            xw: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            xa: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            xg: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            w_hidden: GpuBuffer::zeros_f32(&self.device, w_hidden_len)?,
            a_hidden: GpuBuffer::zeros_f32(&self.device, a_hidden_len)?,
            g_hidden: GpuBuffer::zeros_f32(&self.device, g_hidden_len)?,
            w_tanh: GpuBuffer::zeros_f32(&self.device, w_hidden_len)?,
            w_pre: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            w: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            a_pre: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            a: GpuBuffer::zeros_f32(&self.device, vector_len)?,
            g_sigmoid: GpuBuffer::zeros_f32(&self.device, g_hidden_len)?,
            g: GpuBuffer::zeros_f32(&self.device, vector_len)?,
        })
    }

    fn time_fp16_layer_norm_forward_topology_ms(
        &self,
        batch: usize,
        probe: &LayerNormLowRankForwardProbeBuffers,
        norm: &LayerNormProducerInput<'_>,
        mirrors: &RwkvLowRankFp16ParameterMirrors,
        topology: LayerNormLowRankForwardTopology,
    ) -> Result<f64> {
        // These kernels are only a few dozen microseconds on compact fixtures.
        // Repeat inside one submission so queue/fence latency does not dominate
        // the topology comparison.
        let repetitions = if batch >= 64 { 4 } else { 16 };
        let probe_norm = LayerNormProducerInput {
            x: &probe.x,
            weight: norm.weight,
            bias: norm.bias,
            mean: &probe.mean,
            rstd: &probe.rstd,
            eps: norm.eps,
        };
        let outputs = LayerNormLowRankForwardOutputs {
            x_norm: &probe.x_norm,
            xw: &probe.xw,
            xa: &probe.xa,
            xg: &probe.xg,
            w_hidden: &probe.w_hidden,
            a_hidden: &probe.a_hidden,
            g_hidden: &probe.g_hidden,
            w_tanh: &probe.w_tanh,
            w_pre: &probe.w_pre,
            w: &probe.w,
            a_pre: &probe.a_pre,
            a: &probe.a,
            g_sigmoid: &probe.g_sigmoid,
            g: &probe.g,
        };
        let elapsed_ms = self.device.time_compute_batch_ms(|commands| {
            for _ in 0..repetitions {
                self.record_fp16_layer_norm_forward_topology_into(
                    commands,
                    batch,
                    &probe.previous,
                    &probe_norm,
                    mirrors,
                    &outputs,
                    topology,
                )?;
            }
            Ok(())
        })?;
        Ok(elapsed_ms / repetitions as f64)
    }

    fn choose_fp16_layer_norm_forward_topology(
        &self,
        batch: usize,
        norm: &LayerNormProducerInput<'_>,
        mirrors: &RwkvLowRankFp16ParameterMirrors,
    ) -> Result<LayerNormLowRankForwardDecision> {
        let candidates = self.available_fp16_layer_norm_forward_topologies();
        let structural_default = self
            .structural_fp16_layer_norm_forward_topology()
            .context("RWKV FP16 LN1/low-rank producer fusion is unavailable")?;
        if candidates.len() == 1
            || std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_DISABLE_LN_FORWARD_AUTOTUNE_ENV).is_some()
        {
            return Ok(LayerNormLowRankForwardDecision {
                topology: structural_default,
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let key = LayerNormLowRankForwardAutotuneKey {
            device_name: self.device.name().to_owned(),
            subgroup_size,
            width: self.width,
            w_rank: self.w_rank,
            a_rank: self.a_rank,
            g_rank: self.g_rank,
            batch_pairs: batch.div_ceil(2),
            has_unpaired_tail: batch % 2 != 0,
        };
        let cache =
            LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(decision) = cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV LN1/low-rank forward autotune cache lock was poisoned")
            })?
            .get(&key)
            .copied()
        {
            return Ok(decision);
        }

        // The caller may still be recording the full training graph. Keep
        // synchronous autotune submissions off its pending x/previous/state
        // buffers so probes cannot consume stale producer output or overwrite
        // scratch that the pending command buffer still needs.
        let probe = self.allocate_fp16_layer_norm_forward_autotune_probe(batch)?;
        let time_topology = |topology| {
            self.time_fp16_layer_norm_forward_topology_ms(batch, &probe, norm, mirrors, topology)
        };
        if let Err(err) = time_topology(structural_default) {
            if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some() {
                eprintln!(
                    "RWKV LN1/low-rank forward autotune warmup failed device={} batch={batch}: {err:#}; using {}",
                    self.device.name(),
                    structural_default.label()
                );
            }
            return Ok(LayerNormLowRankForwardDecision {
                topology: structural_default,
                autotuned: false,
            });
        }

        let mut samples = candidates
            .iter()
            .copied()
            .map(|topology| (topology, Vec::with_capacity(3)))
            .collect::<HashMap<_, _>>();
        for round in 0..3 {
            for offset in 0..candidates.len() {
                let index = if round % 2 == 0 {
                    offset
                } else {
                    candidates.len() - 1 - offset
                };
                let topology = candidates[index];
                match time_topology(topology) {
                    Ok(ms) => samples
                        .get_mut(&topology)
                        .expect("autotune candidate sample bucket must exist")
                        .push(ms),
                    Err(err) => {
                        if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV)
                            .is_some()
                        {
                            eprintln!(
                                "RWKV LN1/low-rank forward autotune failed device={} batch={batch} candidate={}: {err:#}; using {}",
                                self.device.name(),
                                topology.label(),
                                structural_default.label()
                            );
                        }
                        return Ok(LayerNormLowRankForwardDecision {
                            topology: structural_default,
                            autotuned: false,
                        });
                    }
                }
            }
        }
        let mut timings = Vec::with_capacity(candidates.len());
        for topology in candidates {
            let values = samples
                .get_mut(&topology)
                .expect("autotune candidate sample bucket must exist");
            values.sort_by(f64::total_cmp);
            timings.push((topology, values[values.len() / 2]));
        }
        let selected = select_layer_norm_low_rank_forward_topology(&timings, structural_default);
        let decision = LayerNormLowRankForwardDecision {
            topology: selected,
            autotuned: true,
        };

        if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some() {
            let summary = timings
                .iter()
                .map(|(topology, ms)| format!("{}={ms:.5}ms", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV LN1/low-rank forward autotune device={} subgroup={} width={} ranks={}/{}/{} batch_pairs={} tail={} {} selected={} autotuned={}",
                self.device.name(),
                subgroup_size,
                self.width,
                self.w_rank,
                self.a_rank,
                self.g_rank,
                key.batch_pairs,
                key.has_unpaired_tail,
                summary,
                selected.label(),
                decision.autotuned
            );
        }

        cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV LN1/low-rank forward autotune cache lock was poisoned")
            })?
            .insert(key, decision);
        Ok(decision)
    }

    pub(crate) fn can_fuse_layer_norm_forward(&self) -> bool {
        if self.fp16_parameter_storage_active() {
            self.layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows
                .is_some()
                || self
                    .layer_norm_low_rank_producer_forward_fused_fp16_packed_cached
                    .is_some()
                || self
                    .layer_norm_low_rank_producer_forward_fused_fp16_packed_rank128
                    .is_some()
                || self
                    .layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows_wide
                    .is_some()
                || self
                    .layer_norm_low_rank_producer_forward_fused_fp16_packed_four_rows
                    .is_some()
                || self
                    .layer_norm_low_rank_producer_forward_fused_fp16_packed
                    .is_some()
        } else {
            self.layer_norm_low_rank_producer_forward_fused.is_some()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_forward_from_layer_norm(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x: &GpuBuffer,
        weight: &GpuBuffer,
        bias: &GpuBuffer,
        x_norm: &GpuBuffer,
        mean: &GpuBuffer,
        rstd: &GpuBuffer,
        previous: &GpuBuffer,
        eps: f32,
    ) -> Result<()> {
        self.record_forward_inner(
            commands,
            batch,
            x_norm,
            previous,
            Some(LayerNormProducerInput {
                x,
                weight,
                bias,
                mean,
                rstd,
                eps,
            }),
        )
    }

    fn record_forward_inner(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        layer_norm: Option<LayerNormProducerInput<'_>>,
    ) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV low-rank batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let vector_len = batch * self.width;

        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let vector_push = LenPush {
            len: vector_len as u32,
        };
        let w_hidden_push = LenPush {
            len: (batch * self.w_rank) as u32,
        };
        let g_hidden_push = LenPush {
            len: (batch * self.g_rank) as u32,
        };
        let vector_groups = [div_ceil_u32(vector_len, 64), 1, 1];
        let activation_groups = [div_ceil_u32(vector_len, 256), 1, 1];
        let w_activation_groups = [div_ceil_u32(batch * self.w_rank, 256), 1, 1];
        let g_activation_groups = [div_ceil_u32(batch * self.g_rank, 256), 1, 1];
        let mut low_rank_outputs_fused = layer_norm.is_some();

        let w1_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.w_rank as u32,
        };
        let w2_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.w_rank as u32,
            output_dim: self.width as u32,
        };
        let a1_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.a_rank as u32,
        };
        let a2_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.a_rank as u32,
            output_dim: self.width as u32,
        };
        let g1_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.g_rank as u32,
        };
        let g2_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.g_rank as u32,
            output_dim: self.width as u32,
        };
        let fp16_mirrors = self.fp16_parameter_mirrors.as_ref();
        let low_rank_fused_kernel = if fp16_mirrors.is_some() {
            self.low_rank_producer_forward_fused_fp16_packed.as_ref()
        } else {
            self.low_rank_producer_forward_fused.as_ref()
        };
        let (layer_norm_fused_kernel, layer_norm_rows_per_workgroup) =
            if let Some(mirrors) = fp16_mirrors {
                let topology = if let Some(norm) = layer_norm.as_ref() {
                    self.choose_fp16_layer_norm_forward_topology(batch, norm, mirrors)?
                        .topology
                } else {
                    self.structural_fp16_layer_norm_forward_topology()
                        .context("RWKV FP16 LN1/low-rank producer fusion is unavailable")?
                };
                (
                    self.fp16_layer_norm_forward_kernel(topology),
                    topology.rows_per_workgroup(),
                )
            } else {
                (self.layer_norm_low_rank_producer_forward_fused.as_ref(), 1)
            };

        if let Some(norm) = layer_norm {
            let kernel = layer_norm_fused_kernel.context(
                "RWKV LN1/low-rank producer fusion is unavailable on this device/rank shape",
            )?;
            let (w1, a1, g1, w2, a2, g2) = if let Some(mirrors) = fp16_mirrors {
                (
                    mirrors.w1.packed_storage(),
                    mirrors.a1.packed_storage(),
                    mirrors.g1.packed_storage(),
                    mirrors.w2.packed_storage(),
                    mirrors.a2.packed_storage(),
                    mirrors.g2.packed_storage(),
                )
            } else {
                (&self.w1, &self.a1, &self.g1, &self.w2, &self.a2, &self.g2)
            };
            let producer_push = LayerNormLowRankProducerPush {
                rows: batch as u32,
                width: self.width as u32,
                w_rank: self.w_rank as u32,
                a_rank: self.a_rank as u32,
                g_rank: self.g_rank as u32,
                eps: norm.eps,
            };
            kernel.record_dispatch(
                commands,
                &[
                    norm.x,
                    norm.weight,
                    norm.bias,
                    previous,
                    &self.mix_w,
                    &self.mix_a,
                    &self.mix_g,
                    w1,
                    a1,
                    g1,
                    x_norm,
                    norm.mean,
                    norm.rstd,
                    &self.xw,
                    &self.xa,
                    &self.xg,
                    &self.w_hidden,
                    &self.a_hidden,
                    &self.g_hidden,
                    &self.w_tanh,
                    w2,
                    &self.w0,
                    &self.w_pre,
                    &self.w,
                    a2,
                    &self.a0,
                    &self.a_pre,
                    &self.a,
                    &self.g_sigmoid,
                    g2,
                    &self.g,
                ],
                bytemuck::bytes_of(&producer_push),
                [batch.div_ceil(layer_norm_rows_per_workgroup) as u32, 1, 1],
            )?;
        } else if let (Some(kernel), Some(mirrors)) = (
            self.low_rank_full_forward_fused_fp16_packed.as_ref(),
            fp16_mirrors,
        ) {
            let producer_push = LowRankProducerPush {
                rows: batch as u32,
                width: self.width as u32,
                w_rank: self.w_rank as u32,
                a_rank: self.a_rank as u32,
                g_rank: self.g_rank as u32,
            };
            kernel.record_dispatch(
                commands,
                &[
                    x_norm,
                    previous,
                    &self.mix_w,
                    &self.mix_a,
                    &self.mix_g,
                    mirrors.w1.packed_storage(),
                    mirrors.a1.packed_storage(),
                    mirrors.g1.packed_storage(),
                    &self.xw,
                    &self.xa,
                    &self.xg,
                    &self.w_hidden,
                    &self.a_hidden,
                    &self.g_hidden,
                    &self.w_tanh,
                    mirrors.w2.packed_storage(),
                    &self.w0,
                    &self.w_pre,
                    &self.w,
                    mirrors.a2.packed_storage(),
                    &self.a0,
                    &self.a_pre,
                    &self.a,
                    &self.g_sigmoid,
                    mirrors.g2.packed_storage(),
                    &self.g,
                ],
                bytemuck::bytes_of(&producer_push),
                [batch as u32, 1, 1],
            )?;
            low_rank_outputs_fused = true;
        } else if let Some(kernel) = low_rank_fused_kernel {
            let (w1, a1, g1) = if let Some(mirrors) = fp16_mirrors {
                (
                    mirrors.w1.packed_storage(),
                    mirrors.a1.packed_storage(),
                    mirrors.g1.packed_storage(),
                )
            } else {
                (&self.w1, &self.a1, &self.g1)
            };
            let producer_push = LowRankProducerPush {
                rows: batch as u32,
                width: self.width as u32,
                w_rank: self.w_rank as u32,
                a_rank: self.a_rank as u32,
                g_rank: self.g_rank as u32,
            };
            let max_rank = self.w_rank.max(self.a_rank).max(self.g_rank);
            let producer_output_lanes = if fp16_mirrors.is_some() {
                max_rank.div_ceil(2)
            } else {
                max_rank
            };
            kernel.record_dispatch(
                commands,
                &[
                    x_norm,
                    previous,
                    &self.mix_w,
                    &self.mix_a,
                    &self.mix_g,
                    w1,
                    a1,
                    g1,
                    &self.xw,
                    &self.xa,
                    &self.xg,
                    &self.w_hidden,
                    &self.a_hidden,
                    &self.g_hidden,
                ],
                bytemuck::bytes_of(&producer_push),
                [
                    div_ceil_u32(producer_output_lanes, 16),
                    div_ceil_u32(batch, 16),
                    1,
                ],
            )?;
        } else {
            self.time_mix_forward.record_dispatch(
                commands,
                &[
                    x_norm,
                    previous,
                    &self.mix_w,
                    &self.mix_a,
                    &self.mix_g,
                    &self.xw,
                    &self.xa,
                    &self.xg,
                ],
                bytemuck::bytes_of(&mix_push),
                vector_groups,
            )?;

            self.matmul_forward(
                commands,
                &self.xw,
                &self.w1,
                fp16_mirrors.map(|mirrors| mirrors.w1.packed_storage()),
                &self.w_hidden,
                w1_push,
            )?;
            self.matmul_forward(
                commands,
                &self.xa,
                &self.a1,
                fp16_mirrors.map(|mirrors| mirrors.a1.packed_storage()),
                &self.a_hidden,
                a1_push,
            )?;
            self.matmul_forward(
                commands,
                &self.xg,
                &self.g1,
                fp16_mirrors.map(|mirrors| mirrors.g1.packed_storage()),
                &self.g_hidden,
                g1_push,
            )?;
        }

        if !low_rank_outputs_fused {
            self.tanh_forward.record_dispatch(
                commands,
                &[&self.w_hidden, &self.w_tanh],
                bytemuck::bytes_of(&w_hidden_push),
                w_activation_groups,
            )?;
            self.matmul_bias_forward(
                commands,
                &self.w_tanh,
                &self.w2,
                fp16_mirrors.map(|mirrors| mirrors.w2.packed_storage()),
                &self.w0,
                &self.w_pre,
                w2_push,
            )?;
            self.decay_forward.record_dispatch(
                commands,
                &[&self.w_pre, &self.w],
                bytemuck::bytes_of(&vector_push),
                activation_groups,
            )?;
            self.matmul_bias_forward(
                commands,
                &self.a_hidden,
                &self.a2,
                fp16_mirrors.map(|mirrors| mirrors.a2.packed_storage()),
                &self.a0,
                &self.a_pre,
                a2_push,
            )?;
            self.sigmoid_forward.record_dispatch(
                commands,
                &[&self.a_pre, &self.a],
                bytemuck::bytes_of(&vector_push),
                activation_groups,
            )?;

            self.sigmoid_forward.record_dispatch(
                commands,
                &[&self.g_hidden, &self.g_sigmoid],
                bytemuck::bytes_of(&g_hidden_push),
                g_activation_groups,
            )?;
            self.matmul_forward(
                commands,
                &self.g_sigmoid,
                &self.g2,
                fp16_mirrors.map(|mirrors| mirrors.g2.packed_storage()),
                &self.g,
                g2_push,
            )?;
        }
        Ok(())
    }

    /// Record the reverse low-rank graph into a caller-owned command buffer.
    /// `grad_a`, `grad_w`, and `grad_g` are expected to be produced by later
    /// fused-cell kernels in the same command stream.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_a: &GpuBuffer,
        grad_w: &GpuBuffer,
        grad_g: &GpuBuffer,
    ) -> Result<()> {
        self.record_backward_impl(
            commands, batch, x_norm, previous, grad_a, grad_w, grad_g, None, None, None, 64,
        )?;
        Ok(())
    }

    /// Record the reverse low-rank graph and, when supported by the device,
    /// fold its final time-mix input adjoints directly into an earlier
    /// producer's adjoints. `true` means the two trailing vector-add
    /// dispatches can be omitted by the caller.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward_accumulating(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_a: &GpuBuffer,
        grad_w: &GpuBuffer,
        grad_g: &GpuBuffer,
        base_grad_x_norm: &GpuBuffer,
        base_grad_previous: &GpuBuffer,
        output_grad_x_norm: &GpuBuffer,
        output_grad_previous: &GpuBuffer,
    ) -> Result<bool> {
        let (base_accumulated, _) = self.record_backward_impl(
            commands,
            batch,
            x_norm,
            previous,
            grad_a,
            grad_w,
            grad_g,
            Some((
                base_grad_x_norm,
                base_grad_previous,
                output_grad_x_norm,
                output_grad_previous,
            )),
            None,
            None,
            64,
        )?;
        Ok(base_accumulated)
    }

    /// As `record_backward_accumulating`, but also attempts to fold the
    /// enclosing cell's additional normalized-input adjoint into the same
    /// dispatch. The second return value reports whether that outer adjoint was
    /// consumed, allowing descriptor-limited callers to retain their existing
    /// standalone vector-add fallback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward_accumulating_outer_x(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_a: &GpuBuffer,
        grad_w: &GpuBuffer,
        grad_g: &GpuBuffer,
        base_grad_x_norm: &GpuBuffer,
        base_grad_previous: &GpuBuffer,
        outer_grad_x_norm: Option<&GpuBuffer>,
        output_grad_x_norm: &GpuBuffer,
        output_grad_previous: &GpuBuffer,
    ) -> Result<(bool, bool)> {
        self.record_backward_impl(
            commands,
            batch,
            x_norm,
            previous,
            grad_a,
            grad_w,
            grad_g,
            Some((
                base_grad_x_norm,
                base_grad_previous,
                output_grad_x_norm,
                output_grad_previous,
            )),
            outer_grad_x_norm,
            None,
            64,
        )
    }

    /// Available shared-input fan-in depths for the current device. Ordering is
    /// intentionally shallow-to-deep so callers can use the final entry as the
    /// conservative "deepest compatible fusion" fallback.
    pub(crate) fn available_backward_fan_in_schedules(
        &self,
        include_outer: bool,
    ) -> Vec<RwkvLowRankFanInSchedule> {
        let mut schedules = vec![RwkvLowRankFanInSchedule::Split];
        if self.time_mix_backward_fused_add.is_some() {
            schedules.push(RwkvLowRankFanInSchedule::FusedBase);
        }
        if include_outer && self.time_mix_backward_fused_add_outer.is_some() {
            schedules.push(RwkvLowRankFanInSchedule::FusedOuter);
        }
        schedules
    }

    fn time_mix_backward_kernel(&self, workgroup_size: usize) -> Option<&vulkan::ComputeKernel> {
        match workgroup_size {
            32 => self.time_mix_backward_wg32.as_ref(),
            64 => Some(&self.time_mix_backward),
            128 => self.time_mix_backward_wg128.as_ref(),
            _ => None,
        }
    }

    fn time_mix_backward_fused_add_kernel(
        &self,
        workgroup_size: usize,
    ) -> Option<&vulkan::ComputeKernel> {
        match workgroup_size {
            32 => self.time_mix_backward_fused_add_wg32.as_ref(),
            64 => self.time_mix_backward_fused_add.as_ref(),
            128 => self.time_mix_backward_fused_add_wg128.as_ref(),
            _ => None,
        }
    }

    fn time_mix_backward_fused_add_outer_kernel(
        &self,
        workgroup_size: usize,
    ) -> Option<&vulkan::ComputeKernel> {
        match workgroup_size {
            32 => self.time_mix_backward_fused_add_outer_wg32.as_ref(),
            64 => self.time_mix_backward_fused_add_outer.as_ref(),
            128 => self.time_mix_backward_fused_add_outer_wg128.as_ref(),
            _ => None,
        }
    }

    pub(crate) fn backward_fan_in_geometry_available(
        &self,
        schedule: RwkvLowRankFanInSchedule,
        workgroup_size: usize,
    ) -> bool {
        match schedule {
            RwkvLowRankFanInSchedule::Split => {
                self.time_mix_backward_kernel(workgroup_size).is_some()
            }
            RwkvLowRankFanInSchedule::FusedBase => self
                .time_mix_backward_fused_add_kernel(workgroup_size)
                .is_some(),
            RwkvLowRankFanInSchedule::FusedOuter => self
                .time_mix_backward_fused_add_outer_kernel(workgroup_size)
                .is_some(),
        }
    }

    /// Record a specific fan-in depth and local-size specialization. This is
    /// the seam used by the full-cell and token-tape autotuners.
    /// Each invocation still owns one channel and executes the batch reduction
    /// serially, so local-size changes affect occupancy/ownership only and do
    /// not change the PyTorch-visible FP32 accumulation order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_backward_with_fan_in_schedule_and_workgroup_size(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_a: &GpuBuffer,
        grad_w: &GpuBuffer,
        grad_g: &GpuBuffer,
        base_grad_x_norm: &GpuBuffer,
        base_grad_previous: &GpuBuffer,
        outer_grad_x_norm: Option<&GpuBuffer>,
        output_grad_x_norm: &GpuBuffer,
        output_grad_previous: &GpuBuffer,
        schedule: RwkvLowRankFanInSchedule,
        workgroup_size: usize,
    ) -> Result<(bool, bool)> {
        self.record_backward_impl(
            commands,
            batch,
            x_norm,
            previous,
            grad_a,
            grad_w,
            grad_g,
            Some((
                base_grad_x_norm,
                base_grad_previous,
                output_grad_x_norm,
                output_grad_previous,
            )),
            outer_grad_x_norm,
            Some(schedule),
            workgroup_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_backward_impl(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        previous: &GpuBuffer,
        grad_a: &GpuBuffer,
        grad_w: &GpuBuffer,
        grad_g: &GpuBuffer,
        accumulation: Option<(&GpuBuffer, &GpuBuffer, &GpuBuffer, &GpuBuffer)>,
        outer_grad_x_norm: Option<&GpuBuffer>,
        fan_in_schedule: Option<RwkvLowRankFanInSchedule>,
        fan_in_workgroup_size: usize,
    ) -> Result<(bool, bool)> {
        debug_assert!(self.backward_source_scale.is_finite());
        debug_assert!(self.backward_source_scale > 0.0);
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV low-rank batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let vector_len = batch * self.width;
        let mix_push = MixPush {
            batch: batch as u32,
            width: self.width as u32,
        };
        let width_bias_push = BiasPush {
            rows: batch as u32,
            dim: self.width as u32,
        };
        let vector_push = LenPush {
            len: vector_len as u32,
        };
        let w_hidden_push = LenPush {
            len: (batch * self.w_rank) as u32,
        };
        let g_hidden_push = LenPush {
            len: (batch * self.g_rank) as u32,
        };
        if !matches!(fan_in_workgroup_size, 32 | 64 | 128) {
            bail!(
                "RWKV low-rank fan-in workgroup size must be 32, 64, or 128; got {fan_in_workgroup_size}"
            );
        }
        if let Some(schedule) = fan_in_schedule {
            if !self.backward_fan_in_geometry_available(schedule, fan_in_workgroup_size) {
                bail!(
                    "RWKV low-rank fan-in geometry {}@wg{} is unavailable on device {}",
                    schedule.label(),
                    fan_in_workgroup_size,
                    self.device.name()
                );
            }
        }
        let channel_groups = [div_ceil_u32(self.width, fan_in_workgroup_size), 1, 1];
        let activation_groups = [div_ceil_u32(vector_len, 256), 1, 1];
        let w_activation_groups = [div_ceil_u32(batch * self.w_rank, 256), 1, 1];
        let g_activation_groups = [div_ceil_u32(batch * self.g_rank, 256), 1, 1];
        let bias_grad_groups = [div_ceil_u32(self.width, 256), 1, 1];
        let w1_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.w_rank as u32,
        };
        let w2_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.w_rank as u32,
            output_dim: self.width as u32,
        };
        let a1_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.a_rank as u32,
        };
        let a2_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.a_rank as u32,
            output_dim: self.width as u32,
        };
        let g1_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.width as u32,
            output_dim: self.g_rank as u32,
        };
        let g2_push = MatmulPush {
            rows: batch as u32,
            input_dim: self.g_rank as u32,
            output_dim: self.width as u32,
        };
        let fp16_mirrors = self.fp16_parameter_mirrors.as_ref();

        self.decay_backward.record_dispatch(
            commands,
            &[grad_w, &self.w_pre, &self.grad_w_pre],
            bytemuck::bytes_of(&vector_push),
            activation_groups,
        )?;
        self.bias_grad.record_dispatch(
            commands,
            &[&self.grad_w_pre, &self.grad_w0],
            bytemuck::bytes_of(&width_bias_push),
            bias_grad_groups,
        )?;
        self.matmul_backward(
            commands,
            &self.w_tanh,
            &self.grad_w_pre,
            &self.w2,
            fp16_mirrors.map(|mirrors| mirrors.w2.packed_storage()),
            &self.grad_w2,
            &self.grad_w_tanh,
            w2_push,
            false,
        )?;
        self.tanh_backward.record_dispatch(
            commands,
            &[&self.grad_w_tanh, &self.w_hidden, &self.grad_w_hidden],
            bytemuck::bytes_of(&w_hidden_push),
            w_activation_groups,
        )?;
        self.matmul_backward(
            commands,
            &self.xw,
            &self.grad_w_hidden,
            &self.w1,
            fp16_mirrors.map(|mirrors| mirrors.w1.packed_storage()),
            &self.grad_w1,
            &self.grad_xw,
            w1_push,
            true,
        )?;

        self.sigmoid_backward.record_dispatch(
            commands,
            &[grad_a, &self.a_pre, &self.grad_a_pre],
            bytemuck::bytes_of(&vector_push),
            activation_groups,
        )?;
        self.bias_grad.record_dispatch(
            commands,
            &[&self.grad_a_pre, &self.grad_a0],
            bytemuck::bytes_of(&width_bias_push),
            bias_grad_groups,
        )?;
        self.matmul_backward(
            commands,
            &self.a_hidden,
            &self.grad_a_pre,
            &self.a2,
            fp16_mirrors.map(|mirrors| mirrors.a2.packed_storage()),
            &self.grad_a2,
            &self.grad_a_hidden,
            a2_push,
            false,
        )?;
        self.matmul_backward(
            commands,
            &self.xa,
            &self.grad_a_hidden,
            &self.a1,
            fp16_mirrors.map(|mirrors| mirrors.a1.packed_storage()),
            &self.grad_a1,
            &self.grad_xa,
            a1_push,
            true,
        )?;

        self.matmul_backward(
            commands,
            &self.g_sigmoid,
            grad_g,
            &self.g2,
            fp16_mirrors.map(|mirrors| mirrors.g2.packed_storage()),
            &self.grad_g2,
            &self.grad_g_sigmoid,
            g2_push,
            false,
        )?;
        self.sigmoid_backward.record_dispatch(
            commands,
            &[&self.grad_g_sigmoid, &self.g_hidden, &self.grad_g_hidden],
            bytemuck::bytes_of(&g_hidden_push),
            g_activation_groups,
        )?;
        self.matmul_backward(
            commands,
            &self.xg,
            &self.grad_g_hidden,
            &self.g1,
            fp16_mirrors.map(|mirrors| mirrors.g1.packed_storage()),
            &self.grad_g1,
            &self.grad_xg,
            g1_push,
            true,
        )?;

        let allow_outer_fusion = fan_in_schedule
            .map(|schedule| schedule == RwkvLowRankFanInSchedule::FusedOuter)
            .unwrap_or(true);
        if allow_outer_fusion {
            if let (
                Some(kernel),
                Some((base_x, base_previous, output_x, output_previous)),
                Some(outer_x),
            ) = (
                self.time_mix_backward_fused_add_outer_kernel(fan_in_workgroup_size),
                accumulation,
                outer_grad_x_norm,
            ) {
                kernel.record_dispatch(
                    commands,
                    &[
                        x_norm,
                        previous,
                        &self.mix_w,
                        &self.mix_a,
                        &self.mix_g,
                        &self.grad_xw,
                        &self.grad_xa,
                        &self.grad_xg,
                        base_x,
                        base_previous,
                        outer_x,
                        output_x,
                        output_previous,
                        &self.grad_mix_w,
                        &self.grad_mix_a,
                        &self.grad_mix_g,
                    ],
                    bytemuck::bytes_of(&mix_push),
                    channel_groups,
                )?;
                return Ok((true, true));
            }
        }

        let allow_base_fusion = fan_in_schedule
            .map(|schedule| schedule != RwkvLowRankFanInSchedule::Split)
            .unwrap_or(true);
        if allow_base_fusion {
            if let (Some(kernel), Some((base_x, base_previous, output_x, output_previous))) = (
                self.time_mix_backward_fused_add_kernel(fan_in_workgroup_size),
                accumulation,
            ) {
                kernel.record_dispatch(
                    commands,
                    &[
                        x_norm,
                        previous,
                        &self.mix_w,
                        &self.mix_a,
                        &self.mix_g,
                        &self.grad_xw,
                        &self.grad_xa,
                        &self.grad_xg,
                        base_x,
                        base_previous,
                        output_x,
                        output_previous,
                        &self.grad_mix_w,
                        &self.grad_mix_a,
                        &self.grad_mix_g,
                    ],
                    bytemuck::bytes_of(&mix_push),
                    channel_groups,
                )?;
                return Ok((true, false));
            }
        }

        self.time_mix_backward_kernel(fan_in_workgroup_size)
            .context("RWKV low-rank split fan-in geometry is unavailable")?
            .record_dispatch(
                commands,
                &[
                    x_norm,
                    previous,
                    &self.mix_w,
                    &self.mix_a,
                    &self.mix_g,
                    &self.grad_xw,
                    &self.grad_xa,
                    &self.grad_xg,
                    &self.grad_x_norm,
                    &self.grad_previous,
                    &self.grad_mix_w,
                    &self.grad_mix_a,
                    &self.grad_mix_g,
                ],
                bytemuck::bytes_of(&mix_push),
                channel_groups,
            )?;
        Ok((false, false))
    }

    pub(crate) fn record_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
    ) -> Result<()> {
        let vector_len = batch * self.width;
        commands.readback_f32(&self.a, &self.a_readback, vector_len)?;
        commands.readback_f32(&self.w, &self.w_readback, vector_len)?;
        commands.readback_f32(&self.g, &self.g_readback, vector_len)?;
        commands.readback_f32(&self.grad_x_norm, &self.grad_x_norm_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_previous,
            &self.grad_previous_readback,
            vector_len,
        )?;
        commands.readback_f32(&self.grad_mix_w, &self.grad_mix_w_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_a, &self.grad_mix_a_readback, self.width)?;
        commands.readback_f32(&self.grad_mix_g, &self.grad_mix_g_readback, self.width)?;
        commands.readback_f32(&self.grad_w0, &self.grad_w0_readback, self.width)?;
        commands.readback_f32(
            &self.grad_w1,
            &self.grad_w1_readback,
            self.width * self.w_rank,
        )?;
        commands.readback_f32(
            &self.grad_w2,
            &self.grad_w2_readback,
            self.w_rank * self.width,
        )?;
        commands.readback_f32(&self.grad_a0, &self.grad_a0_readback, self.width)?;
        commands.readback_f32(
            &self.grad_a1,
            &self.grad_a1_readback,
            self.width * self.a_rank,
        )?;
        commands.readback_f32(
            &self.grad_a2,
            &self.grad_a2_readback,
            self.a_rank * self.width,
        )?;
        commands.readback_f32(
            &self.grad_g1,
            &self.grad_g1_readback,
            self.width * self.g_rank,
        )?;
        commands.readback_f32(
            &self.grad_g2,
            &self.grad_g2_readback,
            self.g_rank * self.width,
        )?;
        Ok(())
    }

    pub(crate) fn read_result(&self, batch: usize) -> Result<RwkvLowRankResult> {
        let vector_len = batch * self.width;
        Ok(RwkvLowRankResult {
            a: self.a_readback.read_f32(vector_len)?,
            w: self.w_readback.read_f32(vector_len)?,
            g: self.g_readback.read_f32(vector_len)?,
            grad_x_norm: self.grad_x_norm_readback.read_f32(vector_len)?,
            grad_previous: self.grad_previous_readback.read_f32(vector_len)?,
            grad_mix_w: self.grad_mix_w_readback.read_f32(self.width)?,
            grad_mix_a: self.grad_mix_a_readback.read_f32(self.width)?,
            grad_mix_g: self.grad_mix_g_readback.read_f32(self.width)?,
            grad_w0: self.grad_w0_readback.read_f32(self.width)?,
            grad_w1: self.grad_w1_readback.read_f32(self.width * self.w_rank)?,
            grad_w2: self.grad_w2_readback.read_f32(self.w_rank * self.width)?,
            grad_a0: self.grad_a0_readback.read_f32(self.width)?,
            grad_a1: self.grad_a1_readback.read_f32(self.width * self.a_rank)?,
            grad_a2: self.grad_a2_readback.read_f32(self.a_rank * self.width)?,
            grad_g1: self.grad_g1_readback.read_f32(self.width * self.g_rank)?,
            grad_g2: self.grad_g2_readback.read_f32(self.g_rank * self.width)?,
        })
    }

    pub(crate) fn a_buffer(&self) -> &GpuBuffer {
        &self.a
    }

    pub(crate) fn w_buffer(&self) -> &GpuBuffer {
        &self.w
    }

    pub(crate) fn g_buffer(&self) -> &GpuBuffer {
        &self.g
    }

    pub(crate) fn grad_x_norm_buffer(&self) -> &GpuBuffer {
        &self.grad_x_norm
    }

    pub(crate) fn grad_previous_buffer(&self) -> &GpuBuffer {
        &self.grad_previous
    }

    pub(crate) fn trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        let decay = RwkvDecayClass::Decay;
        vec![
            RwkvTrainableRef {
                name: "x_w",
                parameter: &self.mix_w,
                gradient: &self.grad_mix_w,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "x_a",
                parameter: &self.mix_a,
                gradient: &self.grad_mix_a,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "x_g",
                parameter: &self.mix_g,
                gradient: &self.grad_mix_g,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "w0",
                parameter: &self.w0,
                gradient: &self.grad_w0,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "w1",
                parameter: &self.w1,
                gradient: &self.grad_w1,
                len: self.width * self.w_rank,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "w2",
                parameter: &self.w2,
                gradient: &self.grad_w2,
                len: self.w_rank * self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "a0",
                parameter: &self.a0,
                gradient: &self.grad_a0,
                len: self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "a1",
                parameter: &self.a1,
                gradient: &self.grad_a1,
                len: self.width * self.a_rank,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "a2",
                parameter: &self.a2,
                gradient: &self.grad_a2,
                len: self.a_rank * self.width,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "g1",
                parameter: &self.g1,
                gradient: &self.grad_g1,
                len: self.width * self.g_rank,
                decay_class: decay,
            },
            RwkvTrainableRef {
                name: "g2",
                parameter: &self.g2,
                gradient: &self.grad_g2,
                len: self.g_rank * self.width,
                decay_class: decay,
            },
        ]
    }

    fn matmul_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        fp16_weight: Option<&GpuBuffer>,
        output: &GpuBuffer,
        push: MatmulPush,
    ) -> Result<()> {
        let (kernel, weight, output_lanes) = if let Some(fp16_weight) = fp16_weight {
            (
                &self.parameter_matmul_forward_fp16_packed,
                fp16_weight,
                (push.output_dim as usize).div_ceil(2),
            )
        } else {
            (
                &self.parameter_matmul_forward,
                weight,
                push.output_dim as usize,
            )
        };
        kernel.record_dispatch(
            commands,
            &[input, weight, output],
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(output_lanes, 16),
                div_ceil_u32(push.rows as usize, 16),
                1,
            ],
        )
    }

    fn matmul_bias_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        fp16_weight: Option<&GpuBuffer>,
        bias: &GpuBuffer,
        output: &GpuBuffer,
        push: MatmulPush,
    ) -> Result<()> {
        let (kernel, weight, output_lanes) = if let Some(fp16_weight) = fp16_weight {
            (
                &self.parameter_matmul_bias_forward_fp16_packed,
                fp16_weight,
                (push.output_dim as usize).div_ceil(2),
            )
        } else {
            (
                &self.parameter_matmul_bias_forward,
                weight,
                push.output_dim as usize,
            )
        };
        kernel.record_dispatch(
            commands,
            &[input, weight, bias, output],
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(output_lanes, 16),
                div_ceil_u32(push.rows as usize, 16),
                1,
            ],
        )
    }

    fn matmul_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        grad_output: &GpuBuffer,
        weight: &GpuBuffer,
        fp16_weight: Option<&GpuBuffer>,
        grad_weight: &GpuBuffer,
        grad_input: &GpuBuffer,
        push: MatmulPush,
        allow_native_fp16_input_grad: bool,
    ) -> Result<()> {
        let weight_grad_arithmetic = if self.native_fp16_backward_compute
            && self.native_fp16_parameter_grad_compute
            && self.source_scaled_backward_domain
        {
            self.parameter_grad_arithmetic()
        } else {
            RwkvLowRankParameterGradArithmetic::Fp32
        };
        let weight_grad_kernel = self.weight_grad_kernel_for_arithmetic(weight_grad_arithmetic)?;
        weight_grad_kernel.record_dispatch(
            commands,
            &[input, grad_output, grad_weight],
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(push.output_dim as usize, 16),
                div_ceil_u32(push.input_dim as usize, 16),
                1,
            ],
        )?;
        let (input_grad_kernel, input_grad_weight) = if let Some(fp16_weight) = fp16_weight {
            let kernel = if self.native_fp16_backward_compute && allow_native_fp16_input_grad {
                self.parameter_matmul_input_grad_fp16_native_compute
                    .as_ref()
                    .context("native-FP16 RWKV low-rank dX was enabled without a kernel")?
            } else {
                &self.parameter_matmul_input_grad_fp16_packed
            };
            (kernel, fp16_weight)
        } else {
            if self.native_fp16_backward_compute && allow_native_fp16_input_grad {
                bail!("native-FP16 RWKV low-rank dX requires an FP16 execution weight");
            }
            (&self.parameter_matmul_input_grad, weight)
        };
        input_grad_kernel.record_dispatch(
            commands,
            &[grad_output, input_grad_weight, grad_input],
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(push.input_dim as usize, 16),
                div_ceil_u32(push.rows as usize, 16),
                1,
            ],
        )
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn ranks(&self) -> (usize, usize, usize) {
        (self.w_rank, self.a_rank, self.g_rank)
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "RWKV {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("RWKV {name} contains non-finite values");
    }
    Ok(())
}

fn vector_width(shape: &[usize]) -> Option<usize> {
    match shape {
        [width] if *width > 0 => Some(*width),
        [1, width] if *width > 0 => Some(*width),
        _ => None,
    }
}

fn read_vector(path: &Path, name: &str, width: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if vector_width(&shape) != Some(width) {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{width}] or [1, {width}]");
    }
    Ok(values)
}

fn read_first_matrix(path: &Path, name: &str, width: usize) -> Result<(usize, Vec<f32>)> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if shape.len() != 2 || shape[0] != width || shape[1] == 0 {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{width}, rank]");
    }
    Ok((shape[1], values))
}

fn read_second_matrix(path: &Path, name: &str, rank: usize, width: usize) -> Result<Vec<f32>> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if shape != [rank, width] {
        bail!("RWKV tensor {name:?} has shape {shape:?}; expected [{rank}, {width}]");
    }
    Ok(values)
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
