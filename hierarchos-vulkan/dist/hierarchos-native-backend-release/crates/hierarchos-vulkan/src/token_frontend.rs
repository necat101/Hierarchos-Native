use std::{
    collections::HashMap,
    path::Path,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use hierarchos_inference::{ModelConfig, RosaState};

use crate::control::FiniteClampVulkanOp;
use crate::projection_graph::GraphProjectionOp;
use crate::rwkv_optimizer::{RwkvDecayClass, RwkvPersistentAdamW, RwkvTrainableRef};
use crate::training_numerics::VulkanDynamicLossScaleController;
use crate::{
    read_f32_tensor, vulkan, GpuBuffer, SharedLmHeadParameter, SharedTokenAdapterTrainer,
    TiedTokenEmbeddingOp, VulkanDevice,
};

const TOKEN_FRONTEND_ASSEMBLE_SPV: &[u8] = include_bytes!("../shaders/token_frontend_assemble.spv");
const GELU_FORWARD_SPV: &[u8] = include_bytes!("../shaders/gelu_forward.spv");
const GELU_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/gelu_backward.spv");
const TOKEN_FRONTEND_SPLIT_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/token_frontend_split_grad.spv");
const ROSA_PREDICT_BOUNDED_SPV: &[u8] = include_bytes!("../shaders/rosa_predict_bounded.spv");
const ROSA_PREDICT_BOUNDED_LANES_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_lanes.spv");
const ROSA_PREDICT_BOUNDED_SUBGROUP_32_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_subgroup_32.spv");
const ROSA_PREDICT_BOUNDED_SUBGROUP_64_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_subgroup_64.spv");
const ROSA_PREDICT_BOUNDED_SUBGROUP_128_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_subgroup_128.spv");
const ROSA_PREDICT_BOUNDED_SUBGROUP_256_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_subgroup_256.spv");
const ROSA_PREDICT_BOUNDED_SUBGROUP_256_SINGLE_PAIR_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_subgroup_256_single_pair.spv");
const ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_32_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_lanes_subgroup_32.spv");
const ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_64_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_lanes_subgroup_64.spv");
const ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_lanes_subgroup_128.spv");
const ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_lanes_subgroup_256.spv");
const ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_lanes_subgroup_256_single_pair.spv");
#[cfg(test)]
const ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_CACHE_TILED_SPV: &[u8] =
    include_bytes!("../shaders/rosa_predict_bounded_lanes_subgroup_128_cache_tiled.spv");
#[cfg(test)]
const ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_CACHE_TILED_SPV: &[u8] = include_bytes!(
    "../shaders/rosa_predict_bounded_lanes_subgroup_256_single_pair_cache_tiled.spv"
);
const ROSA_GATE_MIX_SPV: &[u8] = include_bytes!("../shaders/rosa_gate_mix.spv");
const CONCAT_TOKEN_CONTEXT_SPV: &[u8] = include_bytes!("../shaders/concat_token_context.spv");
const LTM_SIMILARITY_SPV: &[u8] = include_bytes!("../shaders/ltm_similarity.spv");
const LTM_TOPK_SPV: &[u8] = include_bytes!("../shaders/ltm_topk.spv");
const LTM_GATHER_GATE_SPV: &[u8] = include_bytes!("../shaders/ltm_gather_gate.spv");
const LTM_GATHER_GATE_BACKWARD_SPV: &[u8] =
    include_bytes!("../shaders/ltm_gather_gate_backward.spv");
const LTM_GATHER_GATE_BACKWARD_REDUCE_SPV: &[u8] =
    include_bytes!("../shaders/ltm_gather_gate_backward_reduce.spv");
const LTM_SIMILARITY_QUERY_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/ltm_similarity_query_grad.spv");
const LTM_SIMILARITY_KEY_GRAD_SPV: &[u8] = include_bytes!("../shaders/ltm_similarity_key_grad.spv");
const LTM_VALUE_ALIGNMENT_BACKWARD_SPV: &[u8] =
    include_bytes!("../shaders/ltm_value_alignment_backward.spv");
const SPLIT_TOKEN_CONTEXT_GRAD_SPV: &[u8] =
    include_bytes!("../shaders/split_token_context_grad.spv");
const ROSA_GATE_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/rosa_gate_backward.spv");
const ROSA_GATE_GRAD_REDUCE_SPV: &[u8] = include_bytes!("../shaders/rosa_gate_grad_reduce.spv");
const VECTOR_ADD_SPV: &[u8] = include_bytes!("../shaders/vector_add.spv");
const VECTOR_ADD3_SPV: &[u8] = include_bytes!("../shaders/vector_add3.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrontendPush {
    rows: u32,
    context_dim: u32,
    persistent_dim: u32,
    ltm_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RosaGatePush {
    rows: u32,
    dim: u32,
    gate_floor: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RosaPredictPush {
    rows: u32,
    max_context: u32,
}

struct PreparedRosaHostState {
    next_state: RosaState,
    prediction_ids: Vec<i64>,
}

#[derive(Clone, Copy)]
struct RosaKernelChoice {
    bounded_spirv: &'static [u8],
    lanes_spirv: &'static [u8],
    workgroup_size: u32,
    subgroup_reduction: bool,
    label: &'static str,
    autotuned: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RosaAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    max_context: usize,
    rows: usize,
}

static ROSA_AUTOTUNE_CACHE: OnceLock<Mutex<HashMap<RosaAutotuneKey, u32>>> = OnceLock::new();

fn rosa_subgroup_width(subgroup_size: u32) -> u32 {
    match subgroup_size {
        0..=32 => 32,
        33..=64 => 64,
        65..=128 => 128,
        _ => 256,
    }
}

fn automatic_rosa_workgroup_size(subgroup_size: u32, max_context: usize) -> u32 {
    let subgroup_width = rosa_subgroup_width(subgroup_size);
    // Packed bounded ROSA owns ceil(C / 2) independent state words. Prefer the
    // smallest compiled power-of-two workgroup that lets the steady-state end
    // of a segment assign at most one packed word to each lane. This keeps short
    // histories at one native subgroup while allowing the 512-token production
    // geometry to use all 256 compiled lanes when the device can sustain them.
    let packed_pair_count = max_context.max(1).div_ceil(2) as u32;
    packed_pair_count
        .next_power_of_two()
        .max(subgroup_width)
        .min(256)
}

fn rosa_subgroup_choice(
    workgroup_size: u32,
    max_context: usize,
    autotuned: bool,
) -> RosaKernelChoice {
    let single_packed_pair_per_lane = workgroup_size == 256 && max_context <= 512;
    let (bounded_spirv, lanes_spirv, label) = match workgroup_size {
        32 => (
            ROSA_PREDICT_BOUNDED_SUBGROUP_32_SPV,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_32_SPV,
            "subgroup32",
        ),
        64 => (
            ROSA_PREDICT_BOUNDED_SUBGROUP_64_SPV,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_64_SPV,
            "subgroup64",
        ),
        128 => (
            ROSA_PREDICT_BOUNDED_SUBGROUP_128_SPV,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_SPV,
            "subgroup128",
        ),
        256 if single_packed_pair_per_lane => (
            ROSA_PREDICT_BOUNDED_SUBGROUP_256_SINGLE_PAIR_SPV,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_SPV,
            "subgroup256-single-pair",
        ),
        256 => (
            ROSA_PREDICT_BOUNDED_SUBGROUP_256_SPV,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SPV,
            "subgroup256",
        ),
        _ => unreachable!(),
    };
    RosaKernelChoice {
        bounded_spirv,
        lanes_spirv,
        workgroup_size,
        subgroup_reduction: true,
        label,
        autotuned,
    }
}

fn time_rosa_lane_kernel(
    device: &VulkanDevice,
    spirv: &[u8],
    rows: usize,
    max_context: usize,
) -> Result<f64> {
    let kernel = vulkan::ComputeKernel::new(
        device,
        spirv,
        7,
        std::mem::size_of::<RosaPredictPush>() as u32,
    )?;
    let generation_stride = (max_context + 1) / 2;
    let history = GpuBuffer::zeros_u32(device, rows * max_context)?;
    let history_len = GpuBuffer::zeros_u32(device, rows)?;
    let token_ids = GpuBuffer::zeros_u32(device, rows)?;
    let reset_lanes = GpuBuffer::zeros_u32(device, rows)?;
    let match_state = GpuBuffer::zeros_u32(device, rows * generation_stride * 2)?;
    let predictions = GpuBuffer::zeros_u32(device, rows)?;
    let valid = GpuBuffer::zeros_u32(device, rows)?;
    let push = RosaPredictPush {
        rows: rows as u32,
        max_context: max_context as u32,
    };
    let mut commands = vulkan::ComputeBatch::new(device)?;
    for _ in 0..max_context {
        kernel.record_dispatch(
            &mut commands,
            &[
                &history,
                &history_len,
                &token_ids,
                &reset_lanes,
                &match_state,
                &predictions,
                &valid,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
    }
    let started = Instant::now();
    commands.submit()?;
    Ok(started.elapsed().as_secs_f64() * 1_000.0)
}

fn median_rosa_lane_kernel_ms(
    device: &VulkanDevice,
    spirv: &[u8],
    rows: usize,
    max_context: usize,
) -> Result<f64> {
    let _ = time_rosa_lane_kernel(device, spirv, rows, max_context)?;
    let mut samples = [0.0f64; 3];
    for sample in &mut samples {
        *sample = time_rosa_lane_kernel(device, spirv, rows, max_context)?;
    }
    samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    Ok(samples[1])
}

fn autotuned_rosa_workgroup_size(
    device: &VulkanDevice,
    max_context: usize,
    profile_rows: usize,
) -> Result<Option<u32>> {
    if max_context <= 256
        || max_context > 512
        || std::env::var_os("HIERARCHOS_ROSA_DISABLE_AUTOTUNE").is_some()
        || !device.supports_compute_work_group_size_x(128)
        || !device.supports_compute_work_group_size_x(256)
    {
        return Ok(None);
    }

    let caps = device.subgroup_capabilities();
    let rows = profile_rows.clamp(1, 32);
    let key = RosaAutotuneKey {
        device_name: device.name().to_owned(),
        subgroup_size: caps.subgroup_size,
        max_context,
        rows,
    };
    let cache = ROSA_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(width) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("ROSA autotune cache lock was poisoned"))?
        .get(&key)
        .copied()
    {
        return Ok(Some(width));
    }

    let ms_128 = median_rosa_lane_kernel_ms(
        device,
        ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_SPV,
        rows,
        max_context,
    )?;
    let ms_256 = median_rosa_lane_kernel_ms(
        device,
        ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_SPV,
        rows,
        max_context,
    )?;

    // Require a small win before selecting the wider group. This keeps startup
    // noise from flipping a device to 256 lanes when 128 is effectively tied,
    // while still selecting the measured Radeon optimum when its advantage is
    // material. NVIDIA and other subgroup geometries make the same decision
    // from their own timings rather than inheriting an AMD-specific heuristic.
    let width = if ms_256 < ms_128 * 0.98 { 256 } else { 128 };
    if std::env::var_os("HIERARCHOS_ROSA_AUTOTUNE_LOG").is_some() {
        eprintln!(
            "ROSA autotune device={} subgroup={} context={} rows={} subgroup128_ms={:.3} subgroup256_ms={:.3} selected={}",
            device.name(),
            caps.subgroup_size,
            max_context,
            rows,
            ms_128,
            ms_256,
            width
        );
    }
    cache
        .lock()
        .map_err(|_| anyhow::anyhow!("ROSA autotune cache lock was poisoned"))?
        .insert(key, width);
    Ok(Some(width))
}

fn rosa_kernel_choice(
    device: &VulkanDevice,
    max_context: usize,
    profile_rows: usize,
) -> Result<RosaKernelChoice> {
    let fallback = RosaKernelChoice {
        bounded_spirv: ROSA_PREDICT_BOUNDED_SPV,
        lanes_spirv: ROSA_PREDICT_BOUNDED_LANES_SPV,
        workgroup_size: 64,
        subgroup_reduction: false,
        label: "shared64",
        autotuned: false,
    };
    if !device.supports_compute_subgroup_arithmetic() {
        return Ok(fallback);
    }

    let caps = device.subgroup_capabilities();
    let requested = match std::env::var("HIERARCHOS_ROSA_WORKGROUP_SIZE") {
        Ok(raw) => {
            let parsed = raw.parse::<u32>().with_context(|| {
                format!(
                    "HIERARCHOS_ROSA_WORKGROUP_SIZE must be one of 32, 64, 128, or 256; got {raw:?}"
                )
            })?;
            if !matches!(parsed, 32 | 64 | 128 | 256) {
                bail!(
                    "HIERARCHOS_ROSA_WORKGROUP_SIZE must be one of 32, 64, 128, or 256; got {parsed}"
                );
            }
            Some(parsed)
        }
        Err(std::env::VarError::NotPresent) => None,
        Err(err) => bail!("reading HIERARCHOS_ROSA_WORKGROUP_SIZE: {err}"),
    };

    let subgroup_width = rosa_subgroup_width(caps.subgroup_size);
    let mut automatic_width = automatic_rosa_workgroup_size(caps.subgroup_size, max_context);
    while automatic_width > subgroup_width
        && !device.supports_compute_work_group_size_x(automatic_width)
    {
        automatic_width >>= 1;
    }
    let autotuned_width = if requested.is_none() {
        match autotuned_rosa_workgroup_size(device, max_context, profile_rows) {
            Ok(width) => width,
            Err(err) => {
                if std::env::var_os("HIERARCHOS_ROSA_AUTOTUNE_LOG").is_some() {
                    eprintln!(
                        "ROSA autotune failed on device={} context={}: {err:#}; using geometry fallback",
                        device.name(),
                        max_context
                    );
                }
                None
            }
        }
    } else {
        None
    };
    let workgroup_size = requested.or(autotuned_width).unwrap_or(automatic_width);
    if !device.supports_compute_work_group_size_x(workgroup_size) {
        if requested.is_some() {
            bail!(
                "requested ROSA workgroup width {workgroup_size} exceeds the selected Vulkan device limits"
            );
        }
        return Ok(fallback);
    }

    Ok(rosa_subgroup_choice(
        workgroup_size,
        max_context,
        autotuned_width.is_some(),
    ))
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConcatPush {
    rows: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LtmSimilarityPush {
    rows: u32,
    key_dim: u32,
    slots: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LtmTopkPush {
    rows: u32,
    slots: u32,
    topk: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LtmGatherGatePush {
    rows: u32,
    topk: u32,
    val_dim: u32,
    slots: u32,
    gate_floor: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LtmGatherGateBackwardPush {
    rows: u32,
    topk: u32,
    val_dim: u32,
    slots: u32,
    gate_floor: f32,
    score_grad_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LtmSimilarityBackwardPush {
    rows: u32,
    key_dim: u32,
    slots: u32,
    topk: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LtmValueAlignmentPush {
    rows: u32,
    context_dim: u32,
    val_dim: u32,
    topk: u32,
    in_proj_input_dim: u32,
    memory_offset: u32,
    loss_scale: f32,
    loss_normalizer: f32,
}

/// Host description of the current outer boundary of the Vulkan token front-end.
///
/// `gated_ltm_values` is the row-major `[rows, ltm_topk * ltm_val_dim]` payload
/// after Hierarchos' LTM retrieval/gating. That retrieval remains the next
/// front-end cut. `token_residual` is an optional row-major `[rows, context_dim]`
/// additive feature (currently the natural seam for a gated ROSA contribution).
/// The tied token embedding itself is gathered from `lm_head.weight` on Vulkan.
pub struct HierarchosTokenFrontendInput<'a> {
    pub token_ids: &'a [u32],
    pub token_residual: Option<&'a [f32]>,
    pub gated_ltm_values: &'a [f32],
}

/// Sequence-shaped input for the stateful coherent-v9 memory front-end.
///
/// The caller supplies raw token IDs plus the recurrent `prev_context` visible
/// at each token. Bounded coherent-v9 ROSA prediction/state is advanced by a
/// persistent Vulkan kernel; legacy unbounded ROSA retains a native-Rust
/// fallback because it has no finite device-history contract. All learned work
/// runs on Vulkan, including ROSA embedding/adapter/gating, qproj, LTM
/// similarity/top-k/gather/gating, and `in_proj -> GELU`.
pub struct HierarchosTokenMemoryFrontendInput<'a> {
    pub token_ids: &'a [u32],
    pub prev_context: &'a [f32],
}

/// Batch-lane input for the stateful coherent-v9 memory front-end.
///
/// Unlike [`HierarchosTokenMemoryFrontendInput`], whose rows are consecutive
/// tokens from one sequence, each row here is one independent sequence lane at
/// one time step. `reset_lanes[row] != 0` starts that lane from an empty bounded
/// ROSA segment before consuming the current token.
pub struct HierarchosTokenMemoryFrontendLaneInput<'a> {
    pub token_ids: &'a [u32],
    pub prev_context: &'a [f32],
    pub reset_lanes: &'a [u32],
}

#[derive(Debug)]
pub struct HierarchosTokenFrontendForwardResult {
    pub rows: usize,
    pub token_features: Vec<f32>,
    pub enc: Vec<f32>,
    pub queue_submissions: u32,
}

#[derive(Debug)]
pub struct HierarchosTokenMemoryFrontendForwardResult {
    pub rows: usize,
    pub rosa_prediction_ids: Vec<i64>,
    pub raw_token_features: Vec<f32>,
    pub token_features: Vec<f32>,
    pub query: Vec<f32>,
    pub topk_indices: Vec<u32>,
    pub gated_ltm_values: Vec<f32>,
    pub enc: Vec<f32>,
    pub queue_submissions: u32,
}

#[derive(Debug)]
pub struct HierarchosTokenMemoryFrontendBackwardResult {
    pub forward: HierarchosTokenMemoryFrontendForwardResult,
    pub grad_prev_context: Vec<f32>,
    pub grad_persistent: Vec<f32>,
    pub grad_lm_head_weight: Vec<f32>,
    pub grad_rosa_adapter_down_weight: Vec<f32>,
    pub grad_rosa_adapter_up_weight: Vec<f32>,
    pub grad_rosa_adapter_bias: Vec<f32>,
    pub grad_rosa_gate_logit: f32,
    pub grad_rosa_router_weight: Vec<f32>,
    pub grad_rosa_router_bias: Vec<f32>,
    pub grad_qproj_weight: Vec<f32>,
    pub grad_ltm_keys: Vec<f32>,
    pub grad_ltm_vals: Vec<f32>,
    pub grad_ltm_gate_logit: f32,
    pub grad_ltm_router_weight: Vec<f32>,
    pub grad_ltm_router_bias: Vec<f32>,
    pub grad_in_proj_weight: Vec<f32>,
    pub grad_in_proj_bias: Vec<f32>,
}

/// PyTorch-parity result for the optional LTM value-alignment auxiliary.
///
/// The target hidden state and the `in_proj`-derived readout are detached by
/// contract, so the only learned gradient returned here is
/// `d(val_proj.weight)`.
#[derive(Debug)]
pub struct HierarchosLtmValueAlignmentResult {
    pub rows: usize,
    pub row_cost: Vec<f32>,
    pub grad_val_proj_weight: Vec<f32>,
    pub queue_submissions: u32,
}

#[derive(Debug)]
pub struct HierarchosTokenFrontendBackwardResult {
    pub rows: usize,
    pub token_features: Vec<f32>,
    pub enc: Vec<f32>,
    /// Gradient for the post-ROSA token feature. This is both the tied embedding
    /// gather adjoint and the adjoint returned to an eventual ROSA graph.
    pub grad_token_features: Vec<f32>,
    pub grad_gated_ltm_values: Vec<f32>,
    pub grad_persistent: Vec<f32>,
    pub grad_in_proj_weight: Vec<f32>,
    pub grad_in_proj_bias: Vec<f32>,
    pub grad_lm_head_weight: Vec<f32>,
    pub queue_submissions: u32,
}

/// Vulkan-native first half of Hierarchos' token preprocessing path.
///
/// This moves the tied token gather plus the static MAC assembly and
/// `in_proj -> GELU -> finite_clamp(30)` edge off the host while retaining the
/// exact checkpoint ABI used by PyTorch CUDA and `hierarchos-inference`.
/// Backward produces parameter/input gradients but intentionally does not own an
/// optimizer step yet; that keeps this new graph composable with the canonical
/// full-model AdamW registry instead of creating another optimizer island.
pub struct HierarchosTokenFrontendOp {
    device: VulkanDevice,
    config: ModelConfig,
    max_rows: usize,
    ltm_dim: usize,
    input_dim: usize,

    embedding: TiedTokenEmbeddingOp,
    rosa_embedding: TiedTokenEmbeddingOp,
    rosa_adapter: SharedTokenAdapterTrainer,
    rosa_router: GraphProjectionOp,
    qproj: GraphProjectionOp,
    val_proj: GraphProjectionOp,
    ltm_router: GraphProjectionOp,
    rosa_state: RosaState,
    memory_gate_warmup_step: u64,
    memory_gate_floor: f32,
    rosa_gate_logit: GpuBuffer,
    ltm_gate_logit: GpuBuffer,
    ltm_keys: GpuBuffer,
    ltm_vals: GpuBuffer,
    ltm_fast_vals: GpuBuffer,
    persistent: GpuBuffer,
    in_proj: GraphProjectionOp,
    finite_clamp: FiniteClampVulkanOp,

    token_ids: GpuBuffer,
    token_features: GpuBuffer,
    rosa_history: GpuBuffer,
    rosa_history_len: GpuBuffer,
    rosa_match_state: GpuBuffer,
    rosa_reset_lanes: GpuBuffer,
    rosa_token_ids: GpuBuffer,
    rosa_valid: GpuBuffer,
    rosa_raw_features: GpuBuffer,
    memory_token_features: GpuBuffer,
    prev_context: GpuBuffer,
    q_input: GpuBuffer,
    query: GpuBuffer,
    similarity: GpuBuffer,
    topk_indices: GpuBuffer,
    token_residual: GpuBuffer,
    gated_ltm_values: GpuBuffer,
    mac_input: GpuBuffer,
    gelu_output: GpuBuffer,
    enc: GpuBuffer,
    grad_enc: GpuBuffer,
    grad_gelu: GpuBuffer,
    grad_linear: GpuBuffer,
    grad_token_features: GpuBuffer,
    grad_gated_ltm_values: GpuBuffer,
    grad_persistent: GpuBuffer,
    grad_ltm_router_output: GpuBuffer,
    grad_selected_score: GpuBuffer,
    grad_query: GpuBuffer,
    grad_qproj_output: GpuBuffer,
    grad_token_from_q: GpuBuffer,
    grad_prev_context: GpuBuffer,
    grad_ltm_keys: GpuBuffer,
    grad_ltm_vals: GpuBuffer,
    grad_ltm_gate_logit: GpuBuffer,
    grad_token_after_memory: GpuBuffer,
    grad_rosa_feature: GpuBuffer,
    grad_rosa_router_output: GpuBuffer,
    grad_rosa_gate_contribution: GpuBuffer,
    grad_rosa_gate_logit: GpuBuffer,
    grad_raw_token: GpuBuffer,
    ltm_value_alignment_target: GpuBuffer,
    ltm_value_alignment_row_mask: GpuBuffer,
    ltm_value_alignment_grad_value: GpuBuffer,
    ltm_value_alignment_row_cost: GpuBuffer,

    token_features_readback: GpuBuffer,
    rosa_token_ids_readback: GpuBuffer,
    rosa_valid_readback: GpuBuffer,
    memory_token_features_readback: GpuBuffer,
    query_readback: GpuBuffer,
    topk_indices_readback: GpuBuffer,
    gated_ltm_readback: GpuBuffer,
    enc_readback: GpuBuffer,
    grad_token_readback: GpuBuffer,
    grad_ltm_readback: GpuBuffer,
    grad_persistent_readback: GpuBuffer,
    grad_in_proj_weight_readback: GpuBuffer,
    grad_in_proj_bias_readback: GpuBuffer,
    grad_lm_head_readback: GpuBuffer,
    grad_prev_context_readback: GpuBuffer,
    grad_ltm_keys_readback: GpuBuffer,
    grad_ltm_vals_readback: GpuBuffer,
    grad_ltm_gate_logit_readback: GpuBuffer,
    grad_rosa_gate_logit_readback: GpuBuffer,
    grad_qproj_weight_readback: GpuBuffer,
    grad_rosa_router_weight_readback: GpuBuffer,
    grad_rosa_router_bias_readback: GpuBuffer,
    grad_ltm_router_weight_readback: GpuBuffer,
    grad_ltm_router_bias_readback: GpuBuffer,
    grad_rosa_adapter_down_readback: GpuBuffer,
    grad_rosa_adapter_up_readback: GpuBuffer,
    grad_rosa_adapter_bias_readback: GpuBuffer,
    ltm_value_alignment_row_cost_readback: GpuBuffer,
    grad_val_proj_weight_readback: GpuBuffer,

    assemble: vulkan::ComputeKernel,
    rosa_predict_bounded: vulkan::ComputeKernel,
    rosa_predict_bounded_lanes: vulkan::ComputeKernel,
    rosa_workgroup_size: u32,
    rosa_subgroup_reduction: bool,
    rosa_kernel_label: &'static str,
    rosa_autotuned: bool,
    rosa_gate_mix: vulkan::ComputeKernel,
    concat_token_context: vulkan::ComputeKernel,
    ltm_similarity: vulkan::ComputeKernel,
    ltm_topk: vulkan::ComputeKernel,
    ltm_gather_gate: vulkan::ComputeKernel,
    ltm_gather_gate_backward: vulkan::ComputeKernel,
    ltm_gather_gate_backward_reduce: vulkan::ComputeKernel,
    ltm_similarity_query_grad: vulkan::ComputeKernel,
    ltm_similarity_key_grad: vulkan::ComputeKernel,
    ltm_value_alignment_backward: vulkan::ComputeKernel,
    split_token_context_grad: vulkan::ComputeKernel,
    rosa_gate_backward: vulkan::ComputeKernel,
    rosa_gate_grad_reduce: vulkan::ComputeKernel,
    vector_add: vulkan::ComputeKernel,
    vector_add3: vulkan::ComputeKernel,
    gelu_forward: vulkan::ComputeKernel,
    gelu_backward: vulkan::ComputeKernel,
    split_grad: vulkan::ComputeKernel,
}

impl HierarchosTokenFrontendOp {
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        max_rows: usize,
    ) -> Result<Self> {
        let shared_lm_head = SharedLmHeadParameter::from_model_package(device.clone(), &model_dir)?;
        Self::from_shared_lm_head(model_dir, max_rows, shared_lm_head)
    }

    pub fn from_shared_lm_head(
        model_dir: impl AsRef<Path>,
        max_rows: usize,
        shared_lm_head: SharedLmHeadParameter,
    ) -> Result<Self> {
        if max_rows == 0 {
            bail!("token front-end max_rows must be positive");
        }
        let device = shared_lm_head.device();
        let model_dir = model_dir.as_ref();
        let config = ModelConfig::from_model_dir(model_dir)
            .context("validating Vulkan token front-end model contract")?;
        if config.context_dim != shared_lm_head.context_dim()
            || config.vocab_size != shared_lm_head.vocab_size()
        {
            bail!(
                "token front-end shared lm_head geometry [{}, {}] does not match config [{}, {}]",
                shared_lm_head.vocab_size(),
                shared_lm_head.context_dim(),
                config.vocab_size,
                config.context_dim
            );
        }
        if !config.use_rosa
            || !config.memory_token_routers
            || config.rosa_embedding_mode != "shared-factorized"
            || config.ltm_time_feature_mode != "metadata-only"
        {
            bail!(
                "memory-native Vulkan token front-end requires coherent-v9 shared-factorized ROSA, token routers, and metadata-only LTM timestamps"
            );
        }
        if config.ltm_topk > 16 {
            bail!(
                "memory-native Vulkan top-k currently supports ltm_topk <= 16; got {}",
                config.ltm_topk
            );
        }

        let tensor_path = model_dir.join("model.safetensors");
        let (persistent_shape, persistent_values) = read_f32_tensor(&tensor_path, "persistent")?;
        if persistent_shape != vec![config.persistent_dim] {
            bail!(
                "persistent has shape {persistent_shape:?}; expected [{}]",
                config.persistent_dim
            );
        }
        let (ltm_key_shape, ltm_keys) = read_f32_tensor(&tensor_path, "ltm.keys")?;
        if ltm_key_shape != vec![config.ltm_slots, config.ltm_key_dim] {
            bail!(
                "ltm.keys has shape {ltm_key_shape:?}; expected [{}, {}]",
                config.ltm_slots,
                config.ltm_key_dim
            );
        }
        let (ltm_val_shape, ltm_vals) = read_f32_tensor(&tensor_path, "ltm.vals")?;
        if ltm_val_shape != vec![config.ltm_slots, config.ltm_val_dim] {
            bail!(
                "ltm.vals has shape {ltm_val_shape:?}; expected [{}, {}]",
                config.ltm_slots,
                config.ltm_val_dim
            );
        }
        let (ltm_fast_shape, ltm_fast_vals) = read_f32_tensor(&tensor_path, "ltm.fast_vals")?;
        if ltm_fast_shape != vec![config.ltm_slots, config.ltm_val_dim] {
            bail!(
                "ltm.fast_vals has shape {ltm_fast_shape:?}; expected [{}, {}]",
                config.ltm_slots,
                config.ltm_val_dim
            );
        }
        let rosa_gate_logit = read_scalar_f32(&tensor_path, "rosa_gate_logit")?;
        let ltm_gate_logit = read_scalar_f32(&tensor_path, "ltm_gate_logit")?;
        let memory_gate_warmup_step_value =
            read_scalar_f32(&tensor_path, "memory_gate_warmup_step")?;
        if memory_gate_warmup_step_value < 0.0
            || memory_gate_warmup_step_value.fract() != 0.0
            || memory_gate_warmup_step_value > u64::MAX as f32
        {
            bail!(
                "memory_gate_warmup_step must be a finite nonnegative integer; got {memory_gate_warmup_step_value}"
            );
        }
        let memory_gate_warmup_step = memory_gate_warmup_step_value as u64;
        let memory_gate_floor = resolved_memory_gate_floor(&config, memory_gate_warmup_step as f32);
        let ltm_dim = config
            .ltm_topk
            .checked_mul(config.ltm_val_dim)
            .context("token front-end LTM width overflow")?;
        let input_dim = config
            .context_dim
            .checked_add(config.persistent_dim)
            .and_then(|value| value.checked_add(ltm_dim))
            .context("token front-end input width overflow")?;
        let token_len = max_rows
            .checked_mul(config.context_dim)
            .context("token front-end token capacity overflow")?;
        let ltm_len = max_rows
            .checked_mul(ltm_dim)
            .context("token front-end LTM capacity overflow")?;
        let mac_len = max_rows
            .checked_mul(input_dim)
            .context("token front-end MAC capacity overflow")?;
        let q_input_len = max_rows
            .checked_mul(config.context_dim)
            .and_then(|value| value.checked_mul(2))
            .context("token front-end q-input capacity overflow")?;
        let query_len = max_rows
            .checked_mul(config.ltm_key_dim)
            .context("token front-end query capacity overflow")?;
        let similarity_len = max_rows
            .checked_mul(config.ltm_slots)
            .context("token front-end similarity capacity overflow")?;
        let topk_len = max_rows
            .checked_mul(config.ltm_topk)
            .context("token front-end top-k capacity overflow")?;
        let ltm_key_len = config
            .ltm_slots
            .checked_mul(config.ltm_key_dim)
            .context("token front-end LTM key capacity overflow")?;
        let ltm_val_store_len = config
            .ltm_slots
            .checked_mul(config.ltm_val_dim)
            .context("token front-end LTM value capacity overflow")?;
        let in_proj_weight_len = config
            .context_dim
            .checked_mul(input_dim)
            .context("token front-end in_proj weight capacity overflow")?;
        let qproj_weight_len = config
            .ltm_key_dim
            .checked_mul(config.context_dim * 2)
            .context("token front-end qproj weight capacity overflow")?;
        let val_proj_weight_len = config
            .ltm_val_dim
            .checked_mul(config.context_dim)
            .context("token front-end val_proj weight capacity overflow")?;
        let alignment_value_len = max_rows
            .checked_mul(config.ltm_val_dim)
            .context("token front-end LTM value-alignment capacity overflow")?;
        let rosa_adapter_down_len = config
            .token_adapter_rank
            .checked_mul(config.context_dim)
            .context("token front-end ROSA down-weight capacity overflow")?;
        let rosa_adapter_up_len = config
            .context_dim
            .checked_mul(config.token_adapter_rank)
            .context("token front-end ROSA up-weight capacity overflow")?;
        let lm_head_len = config
            .vocab_size
            .checked_mul(config.context_dim)
            .context("token front-end lm_head capacity overflow")?;

        let in_proj = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "in_proj",
            true,
            max_rows,
        )?;
        if in_proj.input_dim() != input_dim || in_proj.output_dim() != config.context_dim {
            bail!(
                "in_proj geometry [{}, {}] does not match coherent-v9 token front-end [{}, {}]",
                in_proj.output_dim(),
                in_proj.input_dim(),
                config.context_dim,
                input_dim
            );
        }
        let rosa_adapter = SharedTokenAdapterTrainer::from_model_package(
            device.clone(),
            model_dir,
            "rosa_adapter",
            max_rows,
            0.0,
        )?;
        if rosa_adapter.input_dim() != config.context_dim
            || rosa_adapter.output_dim() != config.context_dim
        {
            bail!(
                "rosa_adapter geometry [{}, {}] does not match context_dim {}",
                rosa_adapter.output_dim(),
                rosa_adapter.input_dim(),
                config.context_dim
            );
        }
        let rosa_router = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "rosa_router",
            true,
            max_rows,
        )?;
        if rosa_router.input_dim() != config.context_dim || rosa_router.output_dim() != 1 {
            bail!("rosa_router must have geometry [1, context_dim]");
        }
        let qproj = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "qproj",
            false,
            max_rows,
        )?;
        if qproj.input_dim() != config.context_dim * 2 || qproj.output_dim() != config.ltm_key_dim {
            bail!("qproj geometry does not match [ltm_key_dim, 2 * context_dim]");
        }
        let val_proj = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "val_proj",
            false,
            max_rows,
        )?;
        if val_proj.input_dim() != config.context_dim || val_proj.output_dim() != config.ltm_val_dim
        {
            bail!("val_proj geometry does not match [ltm_val_dim, context_dim]");
        }
        let ltm_router = GraphProjectionOp::from_model_package(
            device.clone(),
            model_dir,
            "ltm_router",
            true,
            max_rows,
        )?;
        if ltm_router.input_dim() != config.context_dim || ltm_router.output_dim() != 1 {
            bail!("ltm_router must have geometry [1, context_dim]");
        }

        let embedding =
            TiedTokenEmbeddingOp::from_shared_parameter(shared_lm_head.clone(), max_rows)?;
        let rosa_embedding = TiedTokenEmbeddingOp::from_shared_parameter(shared_lm_head, max_rows)?;
        let rosa_history_capacity = max_rows
            .checked_mul(config.rosa_max_context.max(1))
            .context("token front-end ROSA history capacity overflow")?;
        // Match lengths never exceed the configured bounded segment. The
        // Vulkan kernels therefore pack two <=16-bit lengths into one u32 for
        // the ordinary coherent-v9 range (including the 512-token default),
        // while retaining the previous u32 layout above that safe bound.
        let rosa_match_generation_stride = if config.rosa_max_context.max(1) <= u16::MAX as usize {
            (config.rosa_max_context.max(1) + 1) / 2
        } else {
            config.rosa_max_context.max(1)
        };
        let rosa_match_state_capacity = max_rows
            .checked_mul(rosa_match_generation_stride)
            .and_then(|value| value.checked_mul(2))
            .context("token front-end ROSA suffix-state capacity overflow")?;
        let rosa_kernel = rosa_kernel_choice(&device, config.rosa_max_context, max_rows)?;

        Ok(Self {
            assemble: vulkan::ComputeKernel::new(
                &device,
                TOKEN_FRONTEND_ASSEMBLE_SPV,
                5,
                std::mem::size_of::<FrontendPush>() as u32,
            )?,
            rosa_predict_bounded: vulkan::ComputeKernel::new(
                &device,
                rosa_kernel.bounded_spirv,
                6,
                std::mem::size_of::<RosaPredictPush>() as u32,
            )?,
            rosa_predict_bounded_lanes: vulkan::ComputeKernel::new(
                &device,
                rosa_kernel.lanes_spirv,
                7,
                std::mem::size_of::<RosaPredictPush>() as u32,
            )?,
            rosa_workgroup_size: rosa_kernel.workgroup_size,
            rosa_subgroup_reduction: rosa_kernel.subgroup_reduction,
            rosa_kernel_label: rosa_kernel.label,
            rosa_autotuned: rosa_kernel.autotuned,
            gelu_forward: vulkan::ComputeKernel::new(
                &device,
                GELU_FORWARD_SPV,
                2,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            gelu_backward: vulkan::ComputeKernel::new(
                &device,
                GELU_BACKWARD_SPV,
                3,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            split_grad: vulkan::ComputeKernel::new(
                &device,
                TOKEN_FRONTEND_SPLIT_GRAD_SPV,
                4,
                std::mem::size_of::<FrontendPush>() as u32,
            )?,
            rosa_gate_mix: vulkan::ComputeKernel::new(
                &device,
                ROSA_GATE_MIX_SPV,
                6,
                std::mem::size_of::<RosaGatePush>() as u32,
            )?,
            concat_token_context: vulkan::ComputeKernel::new(
                &device,
                CONCAT_TOKEN_CONTEXT_SPV,
                3,
                std::mem::size_of::<ConcatPush>() as u32,
            )?,
            ltm_similarity: vulkan::ComputeKernel::new(
                &device,
                LTM_SIMILARITY_SPV,
                3,
                std::mem::size_of::<LtmSimilarityPush>() as u32,
            )?,
            ltm_topk: vulkan::ComputeKernel::new(
                &device,
                LTM_TOPK_SPV,
                2,
                std::mem::size_of::<LtmTopkPush>() as u32,
            )?,
            ltm_gather_gate: vulkan::ComputeKernel::new(
                &device,
                LTM_GATHER_GATE_SPV,
                6,
                std::mem::size_of::<LtmGatherGatePush>() as u32,
            )?,
            ltm_gather_gate_backward: vulkan::ComputeKernel::new(
                &device,
                LTM_GATHER_GATE_BACKWARD_SPV,
                10,
                std::mem::size_of::<LtmGatherGateBackwardPush>() as u32,
            )?,
            ltm_gather_gate_backward_reduce: vulkan::ComputeKernel::new(
                &device,
                LTM_GATHER_GATE_BACKWARD_REDUCE_SPV,
                7,
                std::mem::size_of::<LtmGatherGateBackwardPush>() as u32,
            )?,
            ltm_similarity_query_grad: vulkan::ComputeKernel::new(
                &device,
                LTM_SIMILARITY_QUERY_GRAD_SPV,
                4,
                std::mem::size_of::<LtmSimilarityBackwardPush>() as u32,
            )?,
            ltm_similarity_key_grad: vulkan::ComputeKernel::new(
                &device,
                LTM_SIMILARITY_KEY_GRAD_SPV,
                4,
                std::mem::size_of::<LtmSimilarityBackwardPush>() as u32,
            )?,
            ltm_value_alignment_backward: vulkan::ComputeKernel::new(
                &device,
                LTM_VALUE_ALIGNMENT_BACKWARD_SPV,
                6,
                std::mem::size_of::<LtmValueAlignmentPush>() as u32,
            )?,
            split_token_context_grad: vulkan::ComputeKernel::new(
                &device,
                SPLIT_TOKEN_CONTEXT_GRAD_SPV,
                3,
                std::mem::size_of::<ConcatPush>() as u32,
            )?,
            rosa_gate_backward: vulkan::ComputeKernel::new(
                &device,
                ROSA_GATE_BACKWARD_SPV,
                8,
                std::mem::size_of::<RosaGatePush>() as u32,
            )?,
            rosa_gate_grad_reduce: vulkan::ComputeKernel::new(
                &device,
                ROSA_GATE_GRAD_REDUCE_SPV,
                2,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            vector_add: vulkan::ComputeKernel::new(
                &device,
                VECTOR_ADD_SPV,
                3,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            vector_add3: vulkan::ComputeKernel::new(
                &device,
                VECTOR_ADD3_SPV,
                4,
                std::mem::size_of::<LenPush>() as u32,
            )?,
            embedding,
            rosa_embedding,
            rosa_adapter,
            rosa_router,
            qproj,
            val_proj,
            ltm_router,
            rosa_state: RosaState::new(),
            memory_gate_warmup_step,
            memory_gate_floor,
            rosa_gate_logit: GpuBuffer::from_f32(&device, &[rosa_gate_logit])?,
            ltm_gate_logit: GpuBuffer::from_f32(&device, &[ltm_gate_logit])?,
            ltm_keys: GpuBuffer::from_f32(&device, &ltm_keys)?,
            ltm_vals: GpuBuffer::from_f32(&device, &ltm_vals)?,
            ltm_fast_vals: GpuBuffer::from_f32(&device, &ltm_fast_vals)?,
            persistent: GpuBuffer::from_f32(&device, &persistent_values)?,
            in_proj,
            finite_clamp: FiniteClampVulkanOp::new(&device)?,
            token_ids: GpuBuffer::zeros_u32(&device, max_rows)?,
            token_features: GpuBuffer::zeros_f32(&device, token_len)?,
            rosa_history: GpuBuffer::zeros_u32(&device, rosa_history_capacity)?,
            rosa_history_len: GpuBuffer::zeros_u32(&device, max_rows)?,
            rosa_match_state: GpuBuffer::zeros_u32(&device, rosa_match_state_capacity)?,
            rosa_reset_lanes: GpuBuffer::zeros_u32(&device, max_rows)?,
            rosa_token_ids: GpuBuffer::zeros_u32(&device, max_rows)?,
            rosa_valid: GpuBuffer::zeros_u32(&device, max_rows)?,
            rosa_raw_features: GpuBuffer::zeros_f32(&device, token_len)?,
            memory_token_features: GpuBuffer::zeros_f32(&device, token_len)?,
            prev_context: GpuBuffer::zeros_f32(&device, token_len)?,
            q_input: GpuBuffer::zeros_f32(&device, q_input_len)?,
            query: GpuBuffer::zeros_f32(&device, query_len)?,
            similarity: GpuBuffer::zeros_f32(&device, similarity_len)?,
            topk_indices: GpuBuffer::zeros_u32(&device, topk_len)?,
            token_residual: GpuBuffer::zeros_f32(&device, token_len)?,
            gated_ltm_values: GpuBuffer::zeros_f32(&device, ltm_len)?,
            mac_input: GpuBuffer::zeros_f32(&device, mac_len)?,
            gelu_output: GpuBuffer::zeros_f32(&device, token_len)?,
            enc: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_enc: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_gelu: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_linear: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_token_features: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_gated_ltm_values: GpuBuffer::zeros_f32(&device, ltm_len)?,
            grad_persistent: GpuBuffer::zeros_f32(&device, config.persistent_dim)?,
            grad_ltm_router_output: GpuBuffer::zeros_f32(&device, max_rows)?,
            grad_selected_score: GpuBuffer::zeros_f32(&device, topk_len)?,
            grad_query: GpuBuffer::zeros_f32(&device, query_len)?,
            grad_qproj_output: GpuBuffer::zeros_f32(&device, query_len)?,
            grad_token_from_q: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_prev_context: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_ltm_keys: GpuBuffer::zeros_f32(&device, ltm_key_len)?,
            grad_ltm_vals: GpuBuffer::zeros_f32(&device, ltm_val_store_len)?,
            grad_ltm_gate_logit: GpuBuffer::zeros_f32(&device, 1)?,
            grad_token_after_memory: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_rosa_feature: GpuBuffer::zeros_f32(&device, token_len)?,
            grad_rosa_router_output: GpuBuffer::zeros_f32(&device, max_rows)?,
            grad_rosa_gate_contribution: GpuBuffer::zeros_f32(&device, max_rows)?,
            grad_rosa_gate_logit: GpuBuffer::zeros_f32(&device, 1)?,
            grad_raw_token: GpuBuffer::zeros_f32(&device, token_len)?,
            ltm_value_alignment_target: GpuBuffer::zeros_f32(&device, token_len)?,
            ltm_value_alignment_row_mask: GpuBuffer::zeros_f32(&device, max_rows)?,
            ltm_value_alignment_grad_value: GpuBuffer::zeros_f32(&device, alignment_value_len)?,
            ltm_value_alignment_row_cost: GpuBuffer::zeros_f32(&device, max_rows)?,
            token_features_readback: GpuBuffer::zeros_host_f32(&device, token_len)?,
            rosa_token_ids_readback: GpuBuffer::zeros_host_f32(&device, max_rows)?,
            rosa_valid_readback: GpuBuffer::zeros_host_f32(&device, max_rows)?,
            memory_token_features_readback: GpuBuffer::zeros_host_f32(&device, token_len)?,
            query_readback: GpuBuffer::zeros_host_f32(&device, query_len)?,
            topk_indices_readback: GpuBuffer::zeros_host_f32(&device, topk_len)?,
            gated_ltm_readback: GpuBuffer::zeros_host_f32(&device, ltm_len)?,
            enc_readback: GpuBuffer::zeros_host_f32(&device, token_len)?,
            grad_token_readback: GpuBuffer::zeros_host_f32(&device, token_len)?,
            grad_ltm_readback: GpuBuffer::zeros_host_f32(&device, ltm_len)?,
            grad_persistent_readback: GpuBuffer::zeros_host_f32(&device, config.persistent_dim)?,
            grad_in_proj_weight_readback: GpuBuffer::zeros_host_f32(&device, in_proj_weight_len)?,
            grad_in_proj_bias_readback: GpuBuffer::zeros_host_f32(&device, config.context_dim)?,
            grad_lm_head_readback: GpuBuffer::zeros_host_f32(&device, lm_head_len)?,
            grad_prev_context_readback: GpuBuffer::zeros_host_f32(&device, token_len)?,
            grad_ltm_keys_readback: GpuBuffer::zeros_host_f32(&device, ltm_key_len)?,
            grad_ltm_vals_readback: GpuBuffer::zeros_host_f32(&device, ltm_val_store_len)?,
            grad_ltm_gate_logit_readback: GpuBuffer::zeros_host_f32(&device, 1)?,
            grad_rosa_gate_logit_readback: GpuBuffer::zeros_host_f32(&device, 1)?,
            grad_qproj_weight_readback: GpuBuffer::zeros_host_f32(&device, qproj_weight_len)?,
            grad_rosa_router_weight_readback: GpuBuffer::zeros_host_f32(
                &device,
                config.context_dim,
            )?,
            grad_rosa_router_bias_readback: GpuBuffer::zeros_host_f32(&device, 1)?,
            grad_ltm_router_weight_readback: GpuBuffer::zeros_host_f32(
                &device,
                config.context_dim,
            )?,
            grad_ltm_router_bias_readback: GpuBuffer::zeros_host_f32(&device, 1)?,
            grad_rosa_adapter_down_readback: GpuBuffer::zeros_host_f32(
                &device,
                rosa_adapter_down_len,
            )?,
            grad_rosa_adapter_up_readback: GpuBuffer::zeros_host_f32(&device, rosa_adapter_up_len)?,
            grad_rosa_adapter_bias_readback: GpuBuffer::zeros_host_f32(
                &device,
                config.context_dim,
            )?,
            ltm_value_alignment_row_cost_readback: GpuBuffer::zeros_host_f32(&device, max_rows)?,
            grad_val_proj_weight_readback: GpuBuffer::zeros_host_f32(&device, val_proj_weight_len)?,
            device,
            config,
            max_rows,
            ltm_dim,
            input_dim,
        })
    }

    fn uses_gpu_bounded_rosa(&self) -> bool {
        self.config.enforce_rosa_max_context && self.config.rosa_max_context > 0
    }

    /// Record the discrete ROSA prediction/state transition at the head of the
    /// token graph. Bounded coherent-v9 keeps both history and transition on
    /// Vulkan. The legacy unbounded contract still uses the native Rust suffix
    /// automaton because an unbounded persistent GPU history needs an explicit
    /// growth/checkpoint policy before it can be made coherent.
    fn record_rosa_predictions(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        token_ids: &[u32],
        readback_predictions: bool,
    ) -> Result<Option<PreparedRosaHostState>> {
        let rows = token_ids.len();
        commands.upload_u32(&self.token_ids, token_ids)?;
        if self.uses_gpu_bounded_rosa() {
            let push = RosaPredictPush {
                rows: rows as u32,
                max_context: self.config.rosa_max_context as u32,
            };
            self.rosa_predict_bounded.record_dispatch(
                commands,
                &[
                    &self.rosa_history,
                    &self.rosa_history_len,
                    &self.token_ids,
                    &self.rosa_match_state,
                    &self.rosa_token_ids,
                    &self.rosa_valid,
                ],
                bytemuck::bytes_of(&push),
                [1, 1, 1],
            )?;
            if readback_predictions {
                commands.readback_f32(&self.rosa_token_ids, &self.rosa_token_ids_readback, rows)?;
                commands.readback_f32(&self.rosa_valid, &self.rosa_valid_readback, rows)?;
            }
            return Ok(None);
        }

        let mut next_state = self.rosa_state.clone();
        let mut predictions = Vec::with_capacity(rows);
        let mut prediction_ids = Vec::with_capacity(rows);
        let mut valid = Vec::with_capacity(rows);
        for &token in token_ids {
            match next_state
                .predict_and_push(token, 0)
                .filter(|&predicted| (predicted as usize) < self.config.vocab_size)
            {
                Some(predicted) => {
                    predictions.push(i64::from(predicted));
                    prediction_ids.push(predicted);
                    valid.push(1);
                }
                None => {
                    predictions.push(-1);
                    prediction_ids.push(0);
                    valid.push(0);
                }
            }
        }
        commands.upload_u32(&self.rosa_token_ids, &prediction_ids)?;
        commands.upload_u32(&self.rosa_valid, &valid)?;
        Ok(Some(PreparedRosaHostState {
            next_state,
            prediction_ids: predictions,
        }))
    }

    fn record_rosa_lane_predictions(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        token_ids: &[u32],
        reset_lanes: &[u32],
        readback_predictions: bool,
    ) -> Result<()> {
        if !self.uses_gpu_bounded_rosa() {
            bail!(
                "independent Vulkan ROSA lanes require enforce_rosa_max_context with a positive rosa_max_context"
            );
        }
        if reset_lanes.len() != token_ids.len() {
            bail!(
                "ROSA reset mask must contain one entry per lane; got {} resets for {} tokens",
                reset_lanes.len(),
                token_ids.len()
            );
        }
        if reset_lanes.iter().any(|&value| value > 1) {
            bail!("ROSA reset mask values must be 0 or 1");
        }
        let rows = token_ids.len();
        commands.upload_u32(&self.token_ids, token_ids)?;
        commands.upload_u32(&self.rosa_reset_lanes, reset_lanes)?;
        let push = RosaPredictPush {
            rows: rows as u32,
            max_context: self.config.rosa_max_context as u32,
        };
        self.rosa_predict_bounded_lanes.record_dispatch(
            commands,
            &[
                &self.rosa_history,
                &self.rosa_history_len,
                &self.token_ids,
                &self.rosa_reset_lanes,
                &self.rosa_match_state,
                &self.rosa_token_ids,
                &self.rosa_valid,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        if readback_predictions {
            commands.readback_f32(&self.rosa_token_ids, &self.rosa_token_ids_readback, rows)?;
            commands.readback_f32(&self.rosa_valid, &self.rosa_valid_readback, rows)?;
        }
        Ok(())
    }

    fn resolve_rosa_prediction_ids(
        &self,
        rows: usize,
        host_state: Option<&PreparedRosaHostState>,
    ) -> Result<Vec<i64>> {
        if let Some(host_state) = host_state {
            return Ok(host_state.prediction_ids.clone());
        }
        let ids = self.rosa_token_ids_readback.read_f32(rows)?;
        let valid = self.rosa_valid_readback.read_f32(rows)?;
        Ok(ids
            .into_iter()
            .zip(valid)
            .map(|(id_bits, valid_bits)| {
                if valid_bits.to_bits() == 0 {
                    -1
                } else {
                    i64::from(id_bits.to_bits())
                }
            })
            .collect())
    }

    pub fn forward(
        &mut self,
        input: HierarchosTokenFrontendInput<'_>,
    ) -> Result<HierarchosTokenFrontendForwardResult> {
        let rows = self.validate_input(&input)?;
        let token_len = rows * self.config.context_dim;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.record_input_uploads(&mut commands, &input, rows)?;
        self.record_forward(&mut commands, rows)?;
        commands.readback_f32(
            &self.token_features,
            &self.token_features_readback,
            token_len,
        )?;
        commands.readback_f32(&self.enc, &self.enc_readback, token_len)?;
        commands.submit()?;
        Ok(HierarchosTokenFrontendForwardResult {
            rows,
            token_features: self.token_features_readback.read_f32(token_len)?,
            enc: self.enc_readback.read_f32(token_len)?,
            queue_submissions: 1,
        })
    }

    /// Advance the coherent-v9 memory front-end from raw token IDs and the
    /// recurrent context visible at each row.
    ///
    /// Bounded coherent-v9 records ROSA prediction/state plus every learned
    /// tensor operation through `enc` into one Vulkan queue submission. Legacy
    /// unbounded ROSA keeps its native-Rust state fallback and uploads only the
    /// resulting predicted IDs.
    pub fn forward_memory(
        &mut self,
        input: HierarchosTokenMemoryFrontendInput<'_>,
    ) -> Result<HierarchosTokenMemoryFrontendForwardResult> {
        self.forward_memory_impl(input, None)
    }

    /// Advance one token for each independent batch lane while preserving a
    /// separate bounded ROSA history per row across calls.
    pub fn forward_memory_lanes(
        &mut self,
        input: HierarchosTokenMemoryFrontendLaneInput<'_>,
    ) -> Result<HierarchosTokenMemoryFrontendForwardResult> {
        let memory_input = HierarchosTokenMemoryFrontendInput {
            token_ids: input.token_ids,
            prev_context: input.prev_context,
        };
        self.forward_memory_impl(memory_input, Some(input.reset_lanes))
    }

    fn forward_memory_impl(
        &mut self,
        input: HierarchosTokenMemoryFrontendInput<'_>,
        reset_lanes: Option<&[u32]>,
    ) -> Result<HierarchosTokenMemoryFrontendForwardResult> {
        let rows = self.validate_memory_input(&input)?;
        let token_len = rows * self.config.context_dim;
        let query_len = rows * self.config.ltm_key_dim;
        let topk_len = rows * self.config.ltm_topk;
        let ltm_len = rows * self.ltm_dim;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let prepared_rosa = if let Some(reset_lanes) = reset_lanes {
            self.record_rosa_lane_predictions(&mut commands, input.token_ids, reset_lanes, true)?;
            None
        } else {
            self.record_rosa_predictions(&mut commands, input.token_ids, true)?
        };
        commands.upload_f32(&self.prev_context, input.prev_context)?;

        self.embedding.record_forward(
            &mut commands,
            rows,
            &self.token_ids,
            &self.token_features,
        )?;
        self.rosa_embedding.record_forward(
            &mut commands,
            rows,
            &self.rosa_token_ids,
            &self.rosa_raw_features,
        )?;
        self.rosa_adapter
            .record_forward(&mut commands, rows, &self.rosa_raw_features)?;
        self.rosa_router
            .record_forward(&mut commands, rows, &self.token_features)?;

        let rosa_push = RosaGatePush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
            gate_floor: self.memory_gate_floor,
        };
        self.rosa_gate_mix.record_dispatch(
            &mut commands,
            &[
                &self.token_features,
                self.rosa_adapter.output_buffer(),
                self.rosa_router.output_buffer(),
                &self.rosa_valid,
                &self.rosa_gate_logit,
                &self.memory_token_features,
            ],
            bytemuck::bytes_of(&rosa_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;

        let concat_push = ConcatPush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
        };
        self.concat_token_context.record_dispatch(
            &mut commands,
            &[
                &self.memory_token_features,
                &self.prev_context,
                &self.q_input,
            ],
            bytemuck::bytes_of(&concat_push),
            [div_ceil_u32(rows * self.config.context_dim * 2, 256), 1, 1],
        )?;
        self.qproj
            .record_forward(&mut commands, rows, &self.q_input)?;
        self.finite_clamp.record_forward(
            &mut commands,
            query_len,
            self.qproj.output_buffer(),
            &self.query,
            12.0,
        )?;

        let similarity_push = LtmSimilarityPush {
            rows: rows as u32,
            key_dim: self.config.ltm_key_dim as u32,
            slots: self.config.ltm_slots as u32,
            scale: (self.config.ltm_key_dim as f32).powf(-0.5),
        };
        self.ltm_similarity.record_dispatch(
            &mut commands,
            &[&self.query, &self.ltm_keys, &self.similarity],
            bytemuck::bytes_of(&similarity_push),
            [div_ceil_u32(rows * self.config.ltm_slots, 64), 1, 1],
        )?;
        let topk_push = LtmTopkPush {
            rows: rows as u32,
            slots: self.config.ltm_slots as u32,
            topk: self.config.ltm_topk as u32,
        };
        self.ltm_topk.record_dispatch(
            &mut commands,
            &[&self.similarity, &self.topk_indices],
            bytemuck::bytes_of(&topk_push),
            [rows as u32, 1, 1],
        )?;

        self.ltm_router
            .record_forward(&mut commands, rows, &self.memory_token_features)?;
        let gather_push = LtmGatherGatePush {
            rows: rows as u32,
            topk: self.config.ltm_topk as u32,
            val_dim: self.config.ltm_val_dim as u32,
            slots: self.config.ltm_slots as u32,
            gate_floor: self.memory_gate_floor,
        };
        self.ltm_gather_gate.record_dispatch(
            &mut commands,
            &[
                &self.topk_indices,
                &self.ltm_vals,
                &self.ltm_fast_vals,
                self.ltm_router.output_buffer(),
                &self.ltm_gate_logit,
                &self.gated_ltm_values,
            ],
            bytemuck::bytes_of(&gather_push),
            [div_ceil_u32(ltm_len, 256), 1, 1],
        )?;

        commands.fill_zero_f32(&self.token_residual, token_len)?;
        self.record_projection_forward(&mut commands, rows, &self.memory_token_features)?;

        commands.readback_f32(
            &self.token_features,
            &self.token_features_readback,
            token_len,
        )?;
        commands.readback_f32(
            &self.memory_token_features,
            &self.memory_token_features_readback,
            token_len,
        )?;
        commands.readback_f32(&self.query, &self.query_readback, query_len)?;
        commands.readback_f32(&self.topk_indices, &self.topk_indices_readback, topk_len)?;
        commands.readback_f32(&self.gated_ltm_values, &self.gated_ltm_readback, ltm_len)?;
        commands.readback_f32(&self.enc, &self.enc_readback, token_len)?;
        commands.submit()?;
        if let Some(prepared) = prepared_rosa.as_ref() {
            self.rosa_state = prepared.next_state.clone();
        }

        let topk_indices = self
            .topk_indices_readback
            .read_f32(topk_len)?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let rosa_prediction_ids = self.resolve_rosa_prediction_ids(rows, prepared_rosa.as_ref())?;
        Ok(HierarchosTokenMemoryFrontendForwardResult {
            rows,
            rosa_prediction_ids,
            raw_token_features: self.token_features_readback.read_f32(token_len)?,
            token_features: self.memory_token_features_readback.read_f32(token_len)?,
            query: self.query_readback.read_f32(query_len)?,
            topk_indices,
            gated_ltm_values: self.gated_ltm_readback.read_f32(ltm_len)?,
            enc: self.enc_readback.read_f32(token_len)?,
            queue_submissions: 1,
        })
    }

    pub fn reset_rosa_state(&mut self) -> Result<()> {
        self.rosa_state = RosaState::new();
        self.rosa_history_len.write_u32(&vec![0; self.max_rows])?;
        Ok(())
    }

    /// Snapshot the live bounded GPU ROSA histories in the same lane-local form
    /// accepted by `restore_rosa_token_histories`. This is the portable
    /// checkpoint boundary: the suffix-match cache stays Vulkan-private and is
    /// deterministically rebuilt from these histories on restore.
    pub fn snapshot_rosa_token_histories(&self, rows: usize) -> Result<Vec<Vec<u32>>> {
        if !self.uses_gpu_bounded_rosa() {
            bail!(
                "portable lane-local ROSA snapshot requires enforce_rosa_max_context with a positive rosa_max_context"
            );
        }
        if rows == 0 || rows > self.max_rows {
            bail!(
                "portable ROSA snapshot must contain 1..={} lanes; got {rows}",
                self.max_rows
            );
        }

        let max_context = self.config.rosa_max_context;
        let history_len = self.rosa_history_len.read_u32(rows)?;
        let history = self.rosa_history.read_u32(
            rows.checked_mul(max_context)
                .context("ROSA snapshot size overflow")?,
        )?;
        extract_bounded_rosa_histories(
            &history,
            &history_len,
            rows,
            max_context,
            self.config.vocab_size,
        )
    }

    /// Restore the bounded per-lane ROSA recurrence from backend-neutral token
    /// histories. The Vulkan predictor carries both the visible history and a
    /// two-generation suffix-match cache, so restoring only `rosa_history_len`
    /// would make the first post-resume prediction depend on stale scratch.
    /// Rebuilding the cache here gives PyTorch/CUDA checkpoints the exact same
    /// next-token state as a continuously running Vulkan stream.
    pub fn restore_rosa_token_histories(&mut self, histories: &[Vec<u32>]) -> Result<()> {
        if !self.uses_gpu_bounded_rosa() {
            bail!(
                "portable lane-local ROSA restore requires enforce_rosa_max_context with a positive rosa_max_context"
            );
        }
        if histories.is_empty() || histories.len() > self.max_rows {
            bail!(
                "portable ROSA restore must contain 1..={} lane histories; got {}",
                self.max_rows,
                histories.len()
            );
        }

        let max_context = self.config.rosa_max_context;
        let (history, history_len, match_state) = rebuild_bounded_rosa_device_state(
            histories,
            self.max_rows,
            max_context,
            self.config.vocab_size,
        )?;
        self.rosa_history.write_u32(&history)?;
        self.rosa_history_len.write_u32(&history_len)?;
        self.rosa_match_state.write_u32(&match_state)?;
        self.rosa_reset_lanes.write_u32(&vec![0; self.max_rows])?;
        Ok(())
    }

    /// Replace the transient coherent-v9 fast-memory values from a portable
    /// PyTorch/CUDA running-state carrier. Native Vulkan currently models this
    /// store as one shared [slots, value_dim] runtime buffer; batch-isolated
    /// writable LTM states are rejected by the graph-level compatibility check
    /// rather than silently collapsing independent rows.
    pub fn restore_ltm_fast_vals(&mut self, values: &[f32]) -> Result<()> {
        let expected = self
            .config
            .ltm_slots
            .checked_mul(self.config.ltm_val_dim)
            .context("LTM fast-state size overflow")?;
        if values.len() != expected {
            bail!(
                "portable LTM fast state contains {} values; expected {}x{}={expected}",
                values.len(),
                self.config.ltm_slots,
                self.config.ltm_val_dim
            );
        }
        if values.iter().any(|value| !value.is_finite()) {
            bail!("portable LTM fast state contains non-finite values");
        }
        self.ltm_fast_vals.write_f32(values)
    }

    /// Snapshot the shared runtime fast-memory values at the backend-neutral
    /// checkpoint boundary.
    pub fn snapshot_ltm_fast_vals(&self) -> Result<Vec<f32>> {
        let expected = self
            .config
            .ltm_slots
            .checked_mul(self.config.ltm_val_dim)
            .context("LTM fast-state size overflow")?;
        self.ltm_fast_vals.read_f32(expected)
    }

    /// Match `HierarchosCore.set_training_step`: the schedule position is an
    /// explicit global-batch coordinate, independent of AdamW's update count.
    pub fn set_training_step(&mut self, step: u64) {
        self.memory_gate_warmup_step = step;
        self.memory_gate_floor = resolved_memory_gate_floor(&self.config, step as f32);
    }

    pub fn training_step(&self) -> u64 {
        self.memory_gate_warmup_step
    }

    /// Forward and reverse the coherent-v9 memory front-end in one Vulkan
    /// submission. Hard top-k selection remains detached exactly like PyTorch;
    /// LTM addressing is trained through Hierarchos' selected-score gradient
    /// injection, so query/key gradients are nonzero without differentiating
    /// the discrete index choice itself.
    pub fn forward_memory_backward(
        &mut self,
        input: HierarchosTokenMemoryFrontendInput<'_>,
        grad_enc: &[f32],
    ) -> Result<HierarchosTokenMemoryFrontendBackwardResult> {
        let rows = self.validate_memory_input(&input)?;
        let token_len = rows * self.config.context_dim;
        let query_len = rows * self.config.ltm_key_dim;
        let topk_len = rows * self.config.ltm_topk;
        let ltm_len = rows * self.ltm_dim;
        if grad_enc.len() != token_len || grad_enc.iter().any(|value| !value.is_finite()) {
            bail!(
                "memory-native grad_enc must contain {token_len} finite values; got {}",
                grad_enc.len()
            );
        }

        let ltm_key_len = self.config.ltm_slots * self.config.ltm_key_dim;
        let ltm_val_store_len = self.config.ltm_slots * self.config.ltm_val_dim;
        let in_proj_weight_len = self.config.context_dim * self.input_dim;
        let qproj_weight_len = self.config.ltm_key_dim * self.config.context_dim * 2;
        let lm_head_len = self.config.vocab_size * self.config.context_dim;
        let rosa_adapter_down_len = self.config.token_adapter_rank * self.config.context_dim;
        let rosa_adapter_up_len = self.config.context_dim * self.config.token_adapter_rank;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let prepared_rosa = self.record_rosa_predictions(&mut commands, input.token_ids, true)?;
        commands.upload_f32(&self.prev_context, input.prev_context)?;
        commands.upload_f32(&self.grad_enc, grad_enc)?;
        self.embedding.record_zero_grad(&mut commands)?;
        commands.fill_zero_f32(&self.grad_ltm_keys, ltm_key_len)?;
        commands.fill_zero_f32(&self.grad_ltm_vals, ltm_val_store_len)?;
        commands.fill_zero_f32(&self.grad_ltm_gate_logit, 1)?;
        commands.fill_zero_f32(&self.grad_rosa_gate_logit, 1)?;

        // Forward: tied token lookup -> learned ROSA contribution -> qproj ->
        // hard LTM retrieval/gating -> MAC in_proj/GELU.
        self.embedding.record_forward(
            &mut commands,
            rows,
            &self.token_ids,
            &self.token_features,
        )?;
        self.rosa_embedding.record_forward(
            &mut commands,
            rows,
            &self.rosa_token_ids,
            &self.rosa_raw_features,
        )?;
        self.rosa_adapter
            .record_forward(&mut commands, rows, &self.rosa_raw_features)?;
        self.rosa_router
            .record_forward(&mut commands, rows, &self.token_features)?;
        let rosa_push = RosaGatePush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
            gate_floor: self.memory_gate_floor,
        };
        self.rosa_gate_mix.record_dispatch(
            &mut commands,
            &[
                &self.token_features,
                self.rosa_adapter.output_buffer(),
                self.rosa_router.output_buffer(),
                &self.rosa_valid,
                &self.rosa_gate_logit,
                &self.memory_token_features,
            ],
            bytemuck::bytes_of(&rosa_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        let concat_push = ConcatPush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
        };
        self.concat_token_context.record_dispatch(
            &mut commands,
            &[
                &self.memory_token_features,
                &self.prev_context,
                &self.q_input,
            ],
            bytemuck::bytes_of(&concat_push),
            [div_ceil_u32(rows * self.config.context_dim * 2, 256), 1, 1],
        )?;
        self.qproj
            .record_forward(&mut commands, rows, &self.q_input)?;
        self.finite_clamp.record_forward(
            &mut commands,
            query_len,
            self.qproj.output_buffer(),
            &self.query,
            12.0,
        )?;
        let similarity_scale = (self.config.ltm_key_dim as f32).powf(-0.5);
        let similarity_push = LtmSimilarityPush {
            rows: rows as u32,
            key_dim: self.config.ltm_key_dim as u32,
            slots: self.config.ltm_slots as u32,
            scale: similarity_scale,
        };
        self.ltm_similarity.record_dispatch(
            &mut commands,
            &[&self.query, &self.ltm_keys, &self.similarity],
            bytemuck::bytes_of(&similarity_push),
            [div_ceil_u32(rows * self.config.ltm_slots, 64), 1, 1],
        )?;
        let topk_push = LtmTopkPush {
            rows: rows as u32,
            slots: self.config.ltm_slots as u32,
            topk: self.config.ltm_topk as u32,
        };
        self.ltm_topk.record_dispatch(
            &mut commands,
            &[&self.similarity, &self.topk_indices],
            bytemuck::bytes_of(&topk_push),
            [rows as u32, 1, 1],
        )?;
        self.ltm_router
            .record_forward(&mut commands, rows, &self.memory_token_features)?;
        let gather_push = LtmGatherGatePush {
            rows: rows as u32,
            topk: self.config.ltm_topk as u32,
            val_dim: self.config.ltm_val_dim as u32,
            slots: self.config.ltm_slots as u32,
            gate_floor: self.memory_gate_floor,
        };
        self.ltm_gather_gate.record_dispatch(
            &mut commands,
            &[
                &self.topk_indices,
                &self.ltm_vals,
                &self.ltm_fast_vals,
                self.ltm_router.output_buffer(),
                &self.ltm_gate_logit,
                &self.gated_ltm_values,
            ],
            bytemuck::bytes_of(&gather_push),
            [div_ceil_u32(ltm_len, 256), 1, 1],
        )?;
        commands.fill_zero_f32(&self.token_residual, token_len)?;
        self.record_projection_forward(&mut commands, rows, &self.memory_token_features)?;

        // Reverse the MAC tail first, yielding token, persistent, and gated-LTM
        // adjoints without leaving the command buffer.
        self.finite_clamp.record_backward(
            &mut commands,
            token_len,
            &self.gelu_output,
            &self.grad_enc,
            &self.grad_gelu,
            30.0,
        )?;
        let token_push = LenPush {
            len: token_len as u32,
        };
        self.gelu_backward.record_dispatch(
            &mut commands,
            &[
                self.in_proj.output_buffer(),
                &self.grad_gelu,
                &self.grad_linear,
            ],
            bytemuck::bytes_of(&token_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.in_proj
            .record_backward(&mut commands, rows, &self.mac_input, &self.grad_linear)?;
        let frontend_push = self.push(rows);
        let split_work = token_len.max(ltm_len).max(self.config.persistent_dim);
        self.split_grad.record_dispatch(
            &mut commands,
            &[
                self.in_proj.grad_input_buffer(),
                &self.grad_token_features,
                &self.grad_gated_ltm_values,
                &self.grad_persistent,
            ],
            bytemuck::bytes_of(&frontend_push),
            [div_ceil_u32(split_work, 256), 1, 1],
        )?;

        // Reverse LTM gate/gather and the selected-score surrogate. Slow vals
        // receive the value gradient; fast_vals are runtime state, not AdamW
        // parameters. Query/key gradients follow only the injected score edge.
        let gather_backward_push = LtmGatherGateBackwardPush {
            rows: rows as u32,
            topk: self.config.ltm_topk as u32,
            val_dim: self.config.ltm_val_dim as u32,
            slots: self.config.ltm_slots as u32,
            gate_floor: self.memory_gate_floor,
            score_grad_scale: self.config.ltm_score_grad_scale,
        };
        self.ltm_gather_gate_backward.record_dispatch(
            &mut commands,
            &[
                &self.topk_indices,
                &self.ltm_vals,
                &self.ltm_fast_vals,
                self.ltm_router.output_buffer(),
                &self.ltm_gate_logit,
                &self.grad_gated_ltm_values,
                &self.grad_ltm_vals,
                &self.grad_ltm_router_output,
                &self.grad_selected_score,
                &self.grad_ltm_gate_logit,
            ],
            bytemuck::bytes_of(&gather_backward_push),
            [rows as u32, 1, 1],
        )?;
        self.ltm_gather_gate_backward_reduce.record_dispatch(
            &mut commands,
            &[
                &self.topk_indices,
                self.ltm_router.output_buffer(),
                &self.ltm_gate_logit,
                &self.grad_gated_ltm_values,
                &self.grad_ltm_vals,
                &self.grad_ltm_router_output,
                &self.grad_ltm_gate_logit,
            ],
            bytemuck::bytes_of(&gather_backward_push),
            [div_ceil_u32(self.config.ltm_val_dim, 64), 1, 1],
        )?;
        self.ltm_router.record_backward(
            &mut commands,
            rows,
            &self.memory_token_features,
            &self.grad_ltm_router_output,
        )?;
        let similarity_backward_push = LtmSimilarityBackwardPush {
            rows: rows as u32,
            key_dim: self.config.ltm_key_dim as u32,
            slots: self.config.ltm_slots as u32,
            topk: self.config.ltm_topk as u32,
            scale: similarity_scale,
        };
        self.ltm_similarity_query_grad.record_dispatch(
            &mut commands,
            &[
                &self.ltm_keys,
                &self.topk_indices,
                &self.grad_selected_score,
                &self.grad_query,
            ],
            bytemuck::bytes_of(&similarity_backward_push),
            [div_ceil_u32(query_len, 256), 1, 1],
        )?;
        self.ltm_similarity_key_grad.record_dispatch(
            &mut commands,
            &[
                &self.query,
                &self.topk_indices,
                &self.grad_selected_score,
                &self.grad_ltm_keys,
            ],
            bytemuck::bytes_of(&similarity_backward_push),
            [
                div_ceil_u32(self.config.ltm_slots * self.config.ltm_key_dim, 256),
                1,
                1,
            ],
        )?;
        self.finite_clamp.record_backward(
            &mut commands,
            query_len,
            self.qproj.output_buffer(),
            &self.grad_query,
            &self.grad_qproj_output,
            12.0,
        )?;
        self.qproj
            .record_backward(&mut commands, rows, &self.q_input, &self.grad_qproj_output)?;
        self.split_token_context_grad.record_dispatch(
            &mut commands,
            &[
                self.qproj.grad_input_buffer(),
                &self.grad_token_from_q,
                &self.grad_prev_context,
            ],
            bytemuck::bytes_of(&concat_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;

        // Every learned memory consumer of token_x meets before ROSA's
        // residual gate is reversed.
        self.vector_add3.record_dispatch(
            &mut commands,
            &[
                &self.grad_token_features,
                &self.grad_token_from_q,
                self.ltm_router.grad_input_buffer(),
                &self.grad_token_after_memory,
            ],
            bytemuck::bytes_of(&token_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.rosa_gate_backward.record_dispatch(
            &mut commands,
            &[
                self.rosa_adapter.output_buffer(),
                self.rosa_router.output_buffer(),
                &self.rosa_valid,
                &self.rosa_gate_logit,
                &self.grad_token_after_memory,
                &self.grad_rosa_feature,
                &self.grad_rosa_router_output,
                &self.grad_rosa_gate_contribution,
            ],
            bytemuck::bytes_of(&rosa_push),
            [rows as u32, 1, 1],
        )?;
        let rosa_reduce_push = LenPush { len: rows as u32 };
        self.rosa_gate_grad_reduce.record_dispatch(
            &mut commands,
            &[
                &self.grad_rosa_gate_contribution,
                &self.grad_rosa_gate_logit,
            ],
            bytemuck::bytes_of(&rosa_reduce_push),
            [1, 1, 1],
        )?;
        self.rosa_router.record_backward(
            &mut commands,
            rows,
            &self.token_features,
            &self.grad_rosa_router_output,
        )?;
        self.rosa_adapter.record_backward(
            &mut commands,
            rows,
            &self.rosa_raw_features,
            &self.grad_rosa_feature,
        )?;
        self.vector_add.record_dispatch(
            &mut commands,
            &[
                &self.grad_token_after_memory,
                self.rosa_router.grad_input_buffer(),
                &self.grad_raw_token,
            ],
            bytemuck::bytes_of(&token_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.embedding.record_backward_accumulate(
            &mut commands,
            rows,
            &self.token_ids,
            &self.grad_raw_token,
        )?;
        self.rosa_embedding.record_backward_accumulate(
            &mut commands,
            rows,
            &self.rosa_token_ids,
            self.rosa_adapter.grad_input_buffer(),
        )?;

        // One command-buffer readback batch makes this probe suitable for
        // parity testing without introducing intermediate host synchronization.
        commands.readback_f32(
            &self.token_features,
            &self.token_features_readback,
            token_len,
        )?;
        commands.readback_f32(
            &self.memory_token_features,
            &self.memory_token_features_readback,
            token_len,
        )?;
        commands.readback_f32(&self.query, &self.query_readback, query_len)?;
        commands.readback_f32(&self.topk_indices, &self.topk_indices_readback, topk_len)?;
        commands.readback_f32(&self.gated_ltm_values, &self.gated_ltm_readback, ltm_len)?;
        commands.readback_f32(&self.enc, &self.enc_readback, token_len)?;
        commands.readback_f32(
            &self.grad_prev_context,
            &self.grad_prev_context_readback,
            token_len,
        )?;
        commands.readback_f32(
            &self.grad_persistent,
            &self.grad_persistent_readback,
            self.config.persistent_dim,
        )?;
        commands.readback_f32(
            self.embedding.shared_parameter().gradient_buffer(),
            &self.grad_lm_head_readback,
            lm_head_len,
        )?;
        commands.readback_f32(
            &self.grad_ltm_keys,
            &self.grad_ltm_keys_readback,
            ltm_key_len,
        )?;
        commands.readback_f32(
            &self.grad_ltm_vals,
            &self.grad_ltm_vals_readback,
            ltm_val_store_len,
        )?;
        commands.readback_f32(
            &self.grad_ltm_gate_logit,
            &self.grad_ltm_gate_logit_readback,
            1,
        )?;
        commands.readback_f32(
            &self.grad_rosa_gate_logit,
            &self.grad_rosa_gate_logit_readback,
            1,
        )?;
        commands.readback_f32(
            self.qproj.grad_weight_buffer(),
            &self.grad_qproj_weight_readback,
            qproj_weight_len,
        )?;
        commands.readback_f32(
            self.rosa_router.grad_weight_buffer(),
            &self.grad_rosa_router_weight_readback,
            self.config.context_dim,
        )?;
        commands.readback_f32(
            self.rosa_router
                .grad_bias_buffer()
                .context("rosa_router unexpectedly has no bias gradient")?,
            &self.grad_rosa_router_bias_readback,
            1,
        )?;
        commands.readback_f32(
            self.ltm_router.grad_weight_buffer(),
            &self.grad_ltm_router_weight_readback,
            self.config.context_dim,
        )?;
        commands.readback_f32(
            self.ltm_router
                .grad_bias_buffer()
                .context("ltm_router unexpectedly has no bias gradient")?,
            &self.grad_ltm_router_bias_readback,
            1,
        )?;
        commands.readback_f32(
            self.in_proj.grad_weight_buffer(),
            &self.grad_in_proj_weight_readback,
            in_proj_weight_len,
        )?;
        commands.readback_f32(
            self.in_proj
                .grad_bias_buffer()
                .context("in_proj unexpectedly has no bias gradient")?,
            &self.grad_in_proj_bias_readback,
            self.config.context_dim,
        )?;
        {
            let adapter_trainables = self.rosa_adapter.deepembed_trainables();
            commands.readback_f32(
                adapter_trainables[0].gradient,
                &self.grad_rosa_adapter_down_readback,
                rosa_adapter_down_len,
            )?;
            commands.readback_f32(
                adapter_trainables[1].gradient,
                &self.grad_rosa_adapter_up_readback,
                rosa_adapter_up_len,
            )?;
            commands.readback_f32(
                adapter_trainables[2].gradient,
                &self.grad_rosa_adapter_bias_readback,
                self.config.context_dim,
            )?;
        }
        commands.submit()?;
        if let Some(prepared) = prepared_rosa.as_ref() {
            self.rosa_state = prepared.next_state.clone();
        }

        let topk_indices = self
            .topk_indices_readback
            .read_f32(topk_len)?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let rosa_prediction_ids = self.resolve_rosa_prediction_ids(rows, prepared_rosa.as_ref())?;
        let forward = HierarchosTokenMemoryFrontendForwardResult {
            rows,
            rosa_prediction_ids,
            raw_token_features: self.token_features_readback.read_f32(token_len)?,
            token_features: self.memory_token_features_readback.read_f32(token_len)?,
            query: self.query_readback.read_f32(query_len)?,
            topk_indices,
            gated_ltm_values: self.gated_ltm_readback.read_f32(ltm_len)?,
            enc: self.enc_readback.read_f32(token_len)?,
            queue_submissions: 1,
        };
        Ok(HierarchosTokenMemoryFrontendBackwardResult {
            forward,
            grad_prev_context: self.grad_prev_context_readback.read_f32(token_len)?,
            grad_persistent: self
                .grad_persistent_readback
                .read_f32(self.config.persistent_dim)?,
            grad_lm_head_weight: self.grad_lm_head_readback.read_f32(lm_head_len)?,
            grad_rosa_adapter_down_weight: self
                .grad_rosa_adapter_down_readback
                .read_f32(rosa_adapter_down_len)?,
            grad_rosa_adapter_up_weight: self
                .grad_rosa_adapter_up_readback
                .read_f32(rosa_adapter_up_len)?,
            grad_rosa_adapter_bias: self
                .grad_rosa_adapter_bias_readback
                .read_f32(self.config.context_dim)?,
            grad_rosa_gate_logit: self.grad_rosa_gate_logit_readback.read_f32(1)?[0],
            grad_rosa_router_weight: self
                .grad_rosa_router_weight_readback
                .read_f32(self.config.context_dim)?,
            grad_rosa_router_bias: self.grad_rosa_router_bias_readback.read_f32(1)?,
            grad_qproj_weight: self.grad_qproj_weight_readback.read_f32(qproj_weight_len)?,
            grad_ltm_keys: self.grad_ltm_keys_readback.read_f32(ltm_key_len)?,
            grad_ltm_vals: self.grad_ltm_vals_readback.read_f32(ltm_val_store_len)?,
            grad_ltm_gate_logit: self.grad_ltm_gate_logit_readback.read_f32(1)?[0],
            grad_ltm_router_weight: self
                .grad_ltm_router_weight_readback
                .read_f32(self.config.context_dim)?,
            grad_ltm_router_bias: self.grad_ltm_router_bias_readback.read_f32(1)?,
            grad_in_proj_weight: self
                .grad_in_proj_weight_readback
                .read_f32(in_proj_weight_len)?,
            grad_in_proj_bias: self
                .grad_in_proj_bias_readback
                .read_f32(self.config.context_dim)?,
        })
    }

    pub fn forward_backward(
        &mut self,
        input: HierarchosTokenFrontendInput<'_>,
        grad_enc: &[f32],
    ) -> Result<HierarchosTokenFrontendBackwardResult> {
        let rows = self.validate_input(&input)?;
        let token_len = rows * self.config.context_dim;
        if grad_enc.len() != token_len || grad_enc.iter().any(|value| !value.is_finite()) {
            bail!(
                "token front-end grad_enc must contain {token_len} finite values; got {}",
                grad_enc.len()
            );
        }
        let ltm_len = rows * self.ltm_dim;
        let weight_len = self.config.context_dim * self.input_dim;
        let lm_head_len = self.config.vocab_size * self.config.context_dim;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.record_input_uploads(&mut commands, &input, rows)?;
        commands.upload_f32(&self.grad_enc, grad_enc)?;
        self.embedding.record_zero_grad(&mut commands)?;
        self.record_forward(&mut commands, rows)?;

        self.finite_clamp.record_backward(
            &mut commands,
            token_len,
            &self.gelu_output,
            &self.grad_enc,
            &self.grad_gelu,
            30.0,
        )?;
        let len_push = LenPush {
            len: token_len as u32,
        };
        self.gelu_backward.record_dispatch(
            &mut commands,
            &[
                self.in_proj.output_buffer(),
                &self.grad_gelu,
                &self.grad_linear,
            ],
            bytemuck::bytes_of(&len_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.in_proj
            .record_backward(&mut commands, rows, &self.mac_input, &self.grad_linear)?;
        let push = self.push(rows);
        let split_work = token_len.max(ltm_len).max(self.config.persistent_dim);
        self.split_grad.record_dispatch(
            &mut commands,
            &[
                self.in_proj.grad_input_buffer(),
                &self.grad_token_features,
                &self.grad_gated_ltm_values,
                &self.grad_persistent,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(split_work, 256), 1, 1],
        )?;
        self.embedding.record_backward_accumulate(
            &mut commands,
            rows,
            &self.token_ids,
            &self.grad_token_features,
        )?;

        commands.readback_f32(
            &self.token_features,
            &self.token_features_readback,
            token_len,
        )?;
        commands.readback_f32(&self.enc, &self.enc_readback, token_len)?;
        commands.readback_f32(
            &self.grad_token_features,
            &self.grad_token_readback,
            token_len,
        )?;
        commands.readback_f32(
            &self.grad_gated_ltm_values,
            &self.grad_ltm_readback,
            ltm_len,
        )?;
        commands.readback_f32(
            &self.grad_persistent,
            &self.grad_persistent_readback,
            self.config.persistent_dim,
        )?;
        commands.readback_f32(
            self.in_proj.grad_weight_buffer(),
            &self.grad_in_proj_weight_readback,
            weight_len,
        )?;
        commands.readback_f32(
            self.in_proj
                .grad_bias_buffer()
                .context("in_proj unexpectedly has no bias gradient")?,
            &self.grad_in_proj_bias_readback,
            self.config.context_dim,
        )?;
        commands.readback_f32(
            self.embedding.shared_parameter().gradient_buffer(),
            &self.grad_lm_head_readback,
            lm_head_len,
        )?;
        commands.submit()?;

        Ok(HierarchosTokenFrontendBackwardResult {
            rows,
            token_features: self.token_features_readback.read_f32(token_len)?,
            enc: self.enc_readback.read_f32(token_len)?,
            grad_token_features: self.grad_token_readback.read_f32(token_len)?,
            grad_gated_ltm_values: self.grad_ltm_readback.read_f32(ltm_len)?,
            grad_persistent: self
                .grad_persistent_readback
                .read_f32(self.config.persistent_dim)?,
            grad_in_proj_weight: self.grad_in_proj_weight_readback.read_f32(weight_len)?,
            grad_in_proj_bias: self
                .grad_in_proj_bias_readback
                .read_f32(self.config.context_dim)?,
            grad_lm_head_weight: self.grad_lm_head_readback.read_f32(lm_head_len)?,
            queue_submissions: 1,
        })
    }

    pub fn shared_lm_head(&self) -> SharedLmHeadParameter {
        self.embedding.shared_parameter()
    }

    /// Evaluate and backpropagate the optional PyTorch LTM value-alignment
    /// auxiliary on Vulkan without taking an optimizer step.
    ///
    /// `target_hidden` is row-major `[rows, context_dim]` and is treated as a
    /// detached target exactly like `sequence_enc.detach()` in PyTorch. The
    /// readout assembled from the live `in_proj.weight` memory columns is also
    /// detached; consequently this call returns only `d(val_proj.weight)`.
    pub fn ltm_value_alignment_backward(
        &mut self,
        target_hidden: &[f32],
        loss_scale: f32,
    ) -> Result<HierarchosLtmValueAlignmentResult> {
        if !target_hidden.len().is_multiple_of(self.config.context_dim) {
            bail!(
                "LTM value-alignment target length {} is not divisible by context_dim {}",
                target_hidden.len(),
                self.config.context_dim
            );
        }
        let rows = target_hidden.len() / self.config.context_dim;
        if rows == 0 || rows > self.max_rows {
            bail!(
                "LTM value-alignment rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.ltm_value_alignment_target, target_hidden)?;
        let row_mask = vec![1.0f32; rows];
        commands.upload_f32(&self.ltm_value_alignment_row_mask, &row_mask)?;
        self.record_ltm_value_alignment_backward(
            &mut commands,
            rows,
            &self.ltm_value_alignment_target,
            &self.ltm_value_alignment_row_mask,
            loss_scale,
            rows as f32,
        )?;
        commands.readback_f32(
            &self.ltm_value_alignment_row_cost,
            &self.ltm_value_alignment_row_cost_readback,
            rows,
        )?;
        let val_proj_weight_len = self.config.ltm_val_dim * self.config.context_dim;
        commands.readback_f32(
            self.val_proj.grad_weight_buffer(),
            &self.grad_val_proj_weight_readback,
            val_proj_weight_len,
        )?;
        commands.submit()?;
        Ok(HierarchosLtmValueAlignmentResult {
            rows,
            row_cost: self.ltm_value_alignment_row_cost_readback.read_f32(rows)?,
            grad_val_proj_weight: self
                .grad_val_proj_weight_readback
                .read_f32(val_proj_weight_len)?,
            queue_submissions: 1,
        })
    }

    fn record_ltm_value_alignment_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        target_hidden: &GpuBuffer,
        row_mask: &GpuBuffer,
        loss_scale: f32,
        loss_normalizer: f32,
    ) -> Result<()> {
        if rows == 0 || rows > self.max_rows {
            bail!(
                "LTM value-alignment rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        if self.config.context_dim > 2048 {
            bail!(
                "Vulkan LTM value-alignment shared-error kernel currently supports context_dim <= 2048; got {}",
                self.config.context_dim
            );
        }
        if !loss_scale.is_finite() || loss_scale < 0.0 {
            bail!("LTM value-alignment loss scale must be finite and non-negative");
        }
        if !loss_normalizer.is_finite() || loss_normalizer <= 0.0 {
            bail!("LTM value-alignment loss normalizer must be finite and positive");
        }
        self.val_proj
            .record_forward(commands, rows, target_hidden)?;
        let push = LtmValueAlignmentPush {
            rows: rows as u32,
            context_dim: self.config.context_dim as u32,
            val_dim: self.config.ltm_val_dim as u32,
            topk: self.config.ltm_topk as u32,
            in_proj_input_dim: self.input_dim as u32,
            memory_offset: (self.config.context_dim + self.config.persistent_dim) as u32,
            loss_scale,
            loss_normalizer,
        };
        self.ltm_value_alignment_backward.record_dispatch(
            commands,
            &[
                target_hidden,
                self.val_proj.output_buffer(),
                self.in_proj.weight_buffer(),
                &self.ltm_value_alignment_grad_value,
                &self.ltm_value_alignment_row_cost,
                row_mask,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        self.val_proj.record_backward(
            commands,
            rows,
            target_hidden,
            &self.ltm_value_alignment_grad_value,
        )
    }

    /// Record the value-alignment gradient and immediately preserve it in the
    /// canonical persistent optimizer before `val_proj` scratch is reused.
    pub(crate) fn record_ltm_value_alignment_into_optimizer(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        target_hidden: &GpuBuffer,
        row_mask: Option<&[f32]>,
        loss_scale: f32,
        loss_normalizer: f32,
        optimizer: &RwkvPersistentAdamW,
    ) -> Result<()> {
        if loss_scale == 0.0 {
            return Ok(());
        }
        let owned_mask;
        let mask = if let Some(mask) = row_mask {
            if mask.len() != rows {
                bail!(
                    "LTM value-alignment row mask has {} entries; expected {rows}",
                    mask.len()
                );
            }
            if mask
                .iter()
                .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
            {
                bail!("LTM value-alignment row mask values must be finite 0/1 values");
            }
            mask
        } else {
            owned_mask = vec![1.0f32; rows];
            owned_mask.as_slice()
        };
        commands.upload_f32(&self.ltm_value_alignment_row_mask, mask)?;
        self.record_ltm_value_alignment_backward(
            commands,
            rows,
            target_hidden,
            &self.ltm_value_alignment_row_mask,
            loss_scale,
            loss_normalizer,
        )?;
        for trainable in self.val_proj_trainables() {
            optimizer.record_accumulate_one(commands, trainable)?;
        }
        Ok(())
    }

    /// Device-authoritative GradScaler form of the LTM writer auxiliary. The
    /// LTM shader applies only the deterministic objective multiplier; the
    /// freshly produced val_proj gradient is then multiplied by the live Vulkan
    /// loss scale immediately before it is accumulated into persistent AdamW.
    /// This is equivalent to `scaler.scale(aux_loss).backward()` without
    /// materializing the current scale on the CPU.
    pub(crate) fn record_ltm_value_alignment_into_optimizer_device_scaled(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        target_hidden: &GpuBuffer,
        row_mask: Option<&[f32]>,
        objective_scale: f32,
        loss_normalizer: f32,
        optimizer: &RwkvPersistentAdamW,
        loss_scaler: &VulkanDynamicLossScaleController,
    ) -> Result<()> {
        if objective_scale == 0.0 {
            return Ok(());
        }
        let owned_mask;
        let mask = if let Some(mask) = row_mask {
            if mask.len() != rows {
                bail!(
                    "LTM value-alignment row mask has {} entries; expected {rows}",
                    mask.len()
                );
            }
            if mask
                .iter()
                .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
            {
                bail!("LTM value-alignment row mask values must be finite 0/1 values");
            }
            mask
        } else {
            owned_mask = vec![1.0f32; rows];
            owned_mask.as_slice()
        };
        commands.upload_f32(&self.ltm_value_alignment_row_mask, mask)?;
        self.record_ltm_value_alignment_backward(
            commands,
            rows,
            target_hidden,
            &self.ltm_value_alignment_row_mask,
            objective_scale,
            loss_normalizer,
        )?;
        for trainable in self.val_proj_trainables() {
            loss_scaler.record_scale_source_by_current_scale(
                commands,
                trainable.gradient,
                trainable.len,
            )?;
            optimizer.record_accumulate_one(commands, trainable)?;
        }
        Ok(())
    }

    pub(crate) fn ltm_value_alignment_row_cost_buffer(&self) -> &GpuBuffer {
        &self.ltm_value_alignment_row_cost
    }

    /// Device-resident value-writer matrix. Control-plane consumers should
    /// reduce this buffer on Vulkan rather than materializing the matrix on the
    /// host merely to derive scalar readiness telemetry.
    pub(crate) fn val_proj_weight_buffer(&self) -> &GpuBuffer {
        self.val_proj.weight_buffer()
    }

    fn val_proj_trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        self.val_proj
            .trainables()
            .into_iter()
            .map(|trainable| RwkvTrainableRef {
                decay_class: RwkvDecayClass::NoDecay,
                ..trainable
            })
            .collect()
    }

    /// Canonical learned tensors owned by the coherent-v9 token/memory front
    /// end, excluding the tied `lm_head.weight` identity. The tied matrix is
    /// registered exactly once by the full output graph; raw-token and ROSA
    /// embedding gradients accumulate into that same physical gradient buffer.
    ///
    /// Decay classes intentionally mirror `build_hierarchos_optimizer`: every
    /// ordinary matrix is AdamW-decayed, while biases, scalars, and the
    /// persistent vector are not. ROSA's adapter is *not* a recurrent
    /// DeepEmbed gate, so its two matrices use the ordinary matrix policy.
    pub(crate) fn optimizer_trainables(&self) -> Vec<RwkvTrainableRef<'_>> {
        let mut trainables = Vec::with_capacity(16);
        trainables.push(RwkvTrainableRef {
            name: "persistent",
            parameter: &self.persistent,
            gradient: &self.grad_persistent,
            len: self.config.persistent_dim,
            decay_class: RwkvDecayClass::NoDecay,
        });

        let adapter = self.rosa_adapter.deepembed_trainables();
        trainables.push(RwkvTrainableRef {
            name: "rosa_adapter.down.weight",
            decay_class: RwkvDecayClass::Decay,
            ..adapter[0]
        });
        trainables.push(RwkvTrainableRef {
            name: "rosa_adapter.up.weight",
            decay_class: RwkvDecayClass::Decay,
            ..adapter[1]
        });
        trainables.push(RwkvTrainableRef {
            name: "rosa_adapter.bias",
            decay_class: RwkvDecayClass::NoDecay,
            ..adapter[2]
        });
        trainables.push(RwkvTrainableRef {
            name: "rosa_gate_logit",
            parameter: &self.rosa_gate_logit,
            gradient: &self.grad_rosa_gate_logit,
            len: 1,
            decay_class: RwkvDecayClass::NoDecay,
        });
        trainables.extend(self.rosa_router.trainables());
        trainables.extend(self.qproj.trainables());
        trainables.extend(self.val_proj_trainables());
        trainables.push(RwkvTrainableRef {
            name: "ltm.keys",
            parameter: &self.ltm_keys,
            gradient: &self.grad_ltm_keys,
            len: self.config.ltm_slots * self.config.ltm_key_dim,
            decay_class: RwkvDecayClass::Decay,
        });
        trainables.push(RwkvTrainableRef {
            name: "ltm.vals",
            parameter: &self.ltm_vals,
            gradient: &self.grad_ltm_vals,
            len: self.config.ltm_slots * self.config.ltm_val_dim,
            decay_class: RwkvDecayClass::Decay,
        });
        trainables.push(RwkvTrainableRef {
            name: "ltm_gate_logit",
            parameter: &self.ltm_gate_logit,
            gradient: &self.grad_ltm_gate_logit,
            len: 1,
            decay_class: RwkvDecayClass::NoDecay,
        });
        trainables.extend(self.ltm_router.trainables());
        trainables.extend(self.in_proj.trainables());
        trainables
    }

    /// Mirror the current per-token frontend scratch gradients into the one
    /// persistent full-model optimizer immediately, before the next token can
    /// overwrite projection/adapter scratch storage.
    pub(crate) fn record_accumulate_gradients(
        &self,
        commands: &mut vulkan::ComputeBatch,
        optimizer: &RwkvPersistentAdamW,
    ) -> Result<()> {
        for trainable in self.optimizer_trainables() {
            // val_proj is an auxiliary-only parameter. Its gradient scratch is
            // produced and accumulated immediately by
            // record_ltm_value_alignment_into_optimizer on sampled tokens.
            if trainable.name == "val_proj.weight" {
                continue;
            }
            optimizer.record_accumulate_one(commands, trainable)?;
        }
        Ok(())
    }

    pub(crate) fn enc_buffer(&self) -> &GpuBuffer {
        &self.enc
    }

    pub(crate) fn grad_prev_context_buffer(&self) -> &GpuBuffer {
        &self.grad_prev_context
    }

    /// Snapshot the discrete ROSA decision for the current token rows. The
    /// values are uint32 payloads, but `ComputeBatch::copy_f32` is a raw
    /// four-byte buffer copy, so preserving them this way is bit-exact and does
    /// not require a host readback.
    pub(crate) fn record_rosa_prediction_checkpoint(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        prediction_ids: &GpuBuffer,
        valid: &GpuBuffer,
    ) -> Result<()> {
        if rows == 0 || rows > self.max_rows {
            bail!(
                "ROSA prediction checkpoint rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        commands.copy_f32(&self.rosa_token_ids, prediction_ids, rows)?;
        commands.copy_f32(&self.rosa_valid, valid, rows)
    }

    /// Record one token per independent sequence lane into a caller-owned
    /// command buffer. No readback or queue submission occurs; bounded ROSA
    /// history advances as part of the same eventual Vulkan submission.
    pub(crate) fn record_memory_forward_lanes(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        input: HierarchosTokenMemoryFrontendInput<'_>,
        reset_lanes: &[u32],
    ) -> Result<usize> {
        let rows = self.validate_memory_input(&input)?;
        self.record_rosa_lane_predictions(commands, input.token_ids, reset_lanes, false)?;
        self.record_memory_tensor_forward(commands, input, rows)?;
        Ok(rows)
    }

    /// Device-context variant used by the outer labeled-sequence tape. The
    /// recurrent `prev_context` boundary is already resident on Vulkan, so
    /// keeping it on device avoids turning every token into a host round trip.
    pub(crate) fn record_memory_forward_lanes_from_device_prev_context(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        token_ids: &[u32],
        prev_context: &GpuBuffer,
        reset_lanes: &[u32],
    ) -> Result<usize> {
        let rows = self.validate_memory_token_rows(token_ids)?;
        let context_len = rows * self.config.context_dim;
        if prev_context.f32_capacity() < context_len {
            bail!(
                "memory-native device prev_context capacity {} is smaller than required {context_len}",
                prev_context.f32_capacity()
            );
        }
        self.record_rosa_lane_predictions(commands, token_ids, reset_lanes, false)?;
        commands.copy_f32(prev_context, &self.prev_context, context_len)?;
        self.record_memory_tensor_forward_loaded(commands, rows)?;
        Ok(rows)
    }

    /// Rematerialize the learned token/memory frontend for reverse-mode replay
    /// using a discrete ROSA prediction that was checkpointed during the true
    /// forward sweep. This deliberately does not touch persistent ROSA history;
    /// the suffix transition is nondifferentiable, while every learned consumer
    /// of its prediction is reconstructed exactly from the saved ID/valid pair.
    pub(crate) fn record_memory_forward_from_rosa_checkpoint(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        input: HierarchosTokenMemoryFrontendInput<'_>,
        prediction_ids: &GpuBuffer,
        valid: &GpuBuffer,
    ) -> Result<usize> {
        let rows = self.validate_memory_input(&input)?;
        commands.upload_u32(&self.token_ids, input.token_ids)?;
        commands.copy_f32(prediction_ids, &self.rosa_token_ids, rows)?;
        commands.copy_f32(valid, &self.rosa_valid, rows)?;
        self.record_memory_tensor_forward(commands, input, rows)?;
        Ok(rows)
    }

    /// Replay counterpart of `record_memory_forward_lanes_from_device_prev_context`.
    /// The nondifferentiable ROSA decision comes from the forward checkpoint,
    /// while the differentiable context boundary remains Vulkan-resident.
    pub(crate) fn record_memory_forward_from_rosa_checkpoint_device_prev_context(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        token_ids: &[u32],
        prev_context: &GpuBuffer,
        prediction_ids: &GpuBuffer,
        valid: &GpuBuffer,
    ) -> Result<usize> {
        let rows = self.validate_memory_token_rows(token_ids)?;
        let context_len = rows * self.config.context_dim;
        if prev_context.f32_capacity() < context_len {
            bail!(
                "memory-native device prev_context capacity {} is smaller than required {context_len}",
                prev_context.f32_capacity()
            );
        }
        commands.upload_u32(&self.token_ids, token_ids)?;
        commands.copy_f32(prediction_ids, &self.rosa_token_ids, rows)?;
        commands.copy_f32(valid, &self.rosa_valid, rows)?;
        commands.copy_f32(prev_context, &self.prev_context, context_len)?;
        self.record_memory_tensor_forward_loaded(commands, rows)?;
        Ok(rows)
    }

    fn record_memory_tensor_forward(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        input: HierarchosTokenMemoryFrontendInput<'_>,
        rows: usize,
    ) -> Result<()> {
        commands.upload_f32(&self.prev_context, input.prev_context)?;
        self.record_memory_tensor_forward_loaded(commands, rows)
    }

    fn record_memory_tensor_forward_loaded(
        &mut self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
    ) -> Result<()> {
        let token_len = rows * self.config.context_dim;
        let query_len = rows * self.config.ltm_key_dim;
        let ltm_len = rows * self.ltm_dim;
        self.embedding
            .record_forward(commands, rows, &self.token_ids, &self.token_features)?;
        self.rosa_embedding.record_forward(
            commands,
            rows,
            &self.rosa_token_ids,
            &self.rosa_raw_features,
        )?;
        self.rosa_adapter
            .record_forward(commands, rows, &self.rosa_raw_features)?;
        self.rosa_router
            .record_forward(commands, rows, &self.token_features)?;

        let rosa_push = RosaGatePush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
            gate_floor: self.memory_gate_floor,
        };
        self.rosa_gate_mix.record_dispatch(
            commands,
            &[
                &self.token_features,
                self.rosa_adapter.output_buffer(),
                self.rosa_router.output_buffer(),
                &self.rosa_valid,
                &self.rosa_gate_logit,
                &self.memory_token_features,
            ],
            bytemuck::bytes_of(&rosa_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        let concat_push = ConcatPush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
        };
        self.concat_token_context.record_dispatch(
            commands,
            &[
                &self.memory_token_features,
                &self.prev_context,
                &self.q_input,
            ],
            bytemuck::bytes_of(&concat_push),
            [div_ceil_u32(rows * self.config.context_dim * 2, 256), 1, 1],
        )?;
        self.qproj.record_forward(commands, rows, &self.q_input)?;
        self.finite_clamp.record_forward(
            commands,
            query_len,
            self.qproj.output_buffer(),
            &self.query,
            12.0,
        )?;
        let similarity_push = LtmSimilarityPush {
            rows: rows as u32,
            key_dim: self.config.ltm_key_dim as u32,
            slots: self.config.ltm_slots as u32,
            scale: (self.config.ltm_key_dim as f32).powf(-0.5),
        };
        self.ltm_similarity.record_dispatch(
            commands,
            &[&self.query, &self.ltm_keys, &self.similarity],
            bytemuck::bytes_of(&similarity_push),
            [div_ceil_u32(rows * self.config.ltm_slots, 64), 1, 1],
        )?;
        let topk_push = LtmTopkPush {
            rows: rows as u32,
            slots: self.config.ltm_slots as u32,
            topk: self.config.ltm_topk as u32,
        };
        self.ltm_topk.record_dispatch(
            commands,
            &[&self.similarity, &self.topk_indices],
            bytemuck::bytes_of(&topk_push),
            [rows as u32, 1, 1],
        )?;
        self.ltm_router
            .record_forward(commands, rows, &self.memory_token_features)?;
        let gather_push = LtmGatherGatePush {
            rows: rows as u32,
            topk: self.config.ltm_topk as u32,
            val_dim: self.config.ltm_val_dim as u32,
            slots: self.config.ltm_slots as u32,
            gate_floor: self.memory_gate_floor,
        };
        self.ltm_gather_gate.record_dispatch(
            commands,
            &[
                &self.topk_indices,
                &self.ltm_vals,
                &self.ltm_fast_vals,
                self.ltm_router.output_buffer(),
                &self.ltm_gate_logit,
                &self.gated_ltm_values,
            ],
            bytemuck::bytes_of(&gather_push),
            [div_ceil_u32(ltm_len, 256), 1, 1],
        )?;
        commands.fill_zero_f32(&self.token_residual, token_len)?;
        self.record_projection_forward(commands, rows, &self.memory_token_features)?;
        Ok(())
    }

    /// Reverse a previously recorded memory front-end using a GPU-resident
    /// `d(enc)` buffer. Scratch gradients are produced but not stepped here;
    /// the caller immediately mirrors them into the canonical full-model
    /// accumulator with `record_accumulate_gradients`.
    pub(crate) fn record_memory_backward_from_device(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        grad_enc: &GpuBuffer,
    ) -> Result<()> {
        if rows == 0 || rows > self.max_rows {
            bail!(
                "memory-native backward rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        let token_len = rows * self.config.context_dim;
        let query_len = rows * self.config.ltm_key_dim;
        let ltm_len = rows * self.ltm_dim;
        if grad_enc.f32_capacity() < token_len {
            bail!(
                "memory-native device grad_enc has capacity {}; expected at least {token_len}",
                grad_enc.f32_capacity()
            );
        }
        let ltm_key_len = self.config.ltm_slots * self.config.ltm_key_dim;
        let ltm_val_store_len = self.config.ltm_slots * self.config.ltm_val_dim;
        commands.fill_zero_f32(&self.grad_ltm_keys, ltm_key_len)?;
        commands.fill_zero_f32(&self.grad_ltm_vals, ltm_val_store_len)?;
        commands.fill_zero_f32(&self.grad_ltm_gate_logit, 1)?;
        commands.fill_zero_f32(&self.grad_rosa_gate_logit, 1)?;

        self.finite_clamp.record_backward(
            commands,
            token_len,
            &self.gelu_output,
            grad_enc,
            &self.grad_gelu,
            30.0,
        )?;
        let token_push = LenPush {
            len: token_len as u32,
        };
        self.gelu_backward.record_dispatch(
            commands,
            &[
                self.in_proj.output_buffer(),
                &self.grad_gelu,
                &self.grad_linear,
            ],
            bytemuck::bytes_of(&token_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.in_proj
            .record_backward(commands, rows, &self.mac_input, &self.grad_linear)?;
        let frontend_push = self.push(rows);
        let split_work = token_len.max(ltm_len).max(self.config.persistent_dim);
        self.split_grad.record_dispatch(
            commands,
            &[
                self.in_proj.grad_input_buffer(),
                &self.grad_token_features,
                &self.grad_gated_ltm_values,
                &self.grad_persistent,
            ],
            bytemuck::bytes_of(&frontend_push),
            [div_ceil_u32(split_work, 256), 1, 1],
        )?;

        let gather_backward_push = LtmGatherGateBackwardPush {
            rows: rows as u32,
            topk: self.config.ltm_topk as u32,
            val_dim: self.config.ltm_val_dim as u32,
            slots: self.config.ltm_slots as u32,
            gate_floor: self.memory_gate_floor,
            score_grad_scale: self.config.ltm_score_grad_scale,
        };
        self.ltm_gather_gate_backward.record_dispatch(
            commands,
            &[
                &self.topk_indices,
                &self.ltm_vals,
                &self.ltm_fast_vals,
                self.ltm_router.output_buffer(),
                &self.ltm_gate_logit,
                &self.grad_gated_ltm_values,
                &self.grad_ltm_vals,
                &self.grad_ltm_router_output,
                &self.grad_selected_score,
                &self.grad_ltm_gate_logit,
            ],
            bytemuck::bytes_of(&gather_backward_push),
            [rows as u32, 1, 1],
        )?;
        self.ltm_gather_gate_backward_reduce.record_dispatch(
            commands,
            &[
                &self.topk_indices,
                self.ltm_router.output_buffer(),
                &self.ltm_gate_logit,
                &self.grad_gated_ltm_values,
                &self.grad_ltm_vals,
                &self.grad_ltm_router_output,
                &self.grad_ltm_gate_logit,
            ],
            bytemuck::bytes_of(&gather_backward_push),
            [div_ceil_u32(self.config.ltm_val_dim, 64), 1, 1],
        )?;
        self.ltm_router.record_backward(
            commands,
            rows,
            &self.memory_token_features,
            &self.grad_ltm_router_output,
        )?;
        let similarity_scale = (self.config.ltm_key_dim as f32).powf(-0.5);
        let similarity_backward_push = LtmSimilarityBackwardPush {
            rows: rows as u32,
            key_dim: self.config.ltm_key_dim as u32,
            slots: self.config.ltm_slots as u32,
            topk: self.config.ltm_topk as u32,
            scale: similarity_scale,
        };
        self.ltm_similarity_query_grad.record_dispatch(
            commands,
            &[
                &self.ltm_keys,
                &self.topk_indices,
                &self.grad_selected_score,
                &self.grad_query,
            ],
            bytemuck::bytes_of(&similarity_backward_push),
            [div_ceil_u32(query_len, 256), 1, 1],
        )?;
        self.ltm_similarity_key_grad.record_dispatch(
            commands,
            &[
                &self.query,
                &self.topk_indices,
                &self.grad_selected_score,
                &self.grad_ltm_keys,
            ],
            bytemuck::bytes_of(&similarity_backward_push),
            [
                div_ceil_u32(self.config.ltm_slots * self.config.ltm_key_dim, 256),
                1,
                1,
            ],
        )?;
        self.finite_clamp.record_backward(
            commands,
            query_len,
            self.qproj.output_buffer(),
            &self.grad_query,
            &self.grad_qproj_output,
            12.0,
        )?;
        self.qproj
            .record_backward(commands, rows, &self.q_input, &self.grad_qproj_output)?;
        let concat_push = ConcatPush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
        };
        self.split_token_context_grad.record_dispatch(
            commands,
            &[
                self.qproj.grad_input_buffer(),
                &self.grad_token_from_q,
                &self.grad_prev_context,
            ],
            bytemuck::bytes_of(&concat_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.vector_add3.record_dispatch(
            commands,
            &[
                &self.grad_token_features,
                &self.grad_token_from_q,
                self.ltm_router.grad_input_buffer(),
                &self.grad_token_after_memory,
            ],
            bytemuck::bytes_of(&token_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        let rosa_push = RosaGatePush {
            rows: rows as u32,
            dim: self.config.context_dim as u32,
            gate_floor: self.memory_gate_floor,
        };
        self.rosa_gate_backward.record_dispatch(
            commands,
            &[
                self.rosa_adapter.output_buffer(),
                self.rosa_router.output_buffer(),
                &self.rosa_valid,
                &self.rosa_gate_logit,
                &self.grad_token_after_memory,
                &self.grad_rosa_feature,
                &self.grad_rosa_router_output,
                &self.grad_rosa_gate_contribution,
            ],
            bytemuck::bytes_of(&rosa_push),
            [rows as u32, 1, 1],
        )?;
        let rosa_reduce_push = LenPush { len: rows as u32 };
        self.rosa_gate_grad_reduce.record_dispatch(
            commands,
            &[
                &self.grad_rosa_gate_contribution,
                &self.grad_rosa_gate_logit,
            ],
            bytemuck::bytes_of(&rosa_reduce_push),
            [1, 1, 1],
        )?;
        self.rosa_router.record_backward(
            commands,
            rows,
            &self.token_features,
            &self.grad_rosa_router_output,
        )?;
        self.rosa_adapter.record_backward(
            commands,
            rows,
            &self.rosa_raw_features,
            &self.grad_rosa_feature,
        )?;
        self.vector_add.record_dispatch(
            commands,
            &[
                &self.grad_token_after_memory,
                self.rosa_router.grad_input_buffer(),
                &self.grad_raw_token,
            ],
            bytemuck::bytes_of(&token_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.embedding.record_backward_accumulate(
            commands,
            rows,
            &self.token_ids,
            &self.grad_raw_token,
        )?;
        self.rosa_embedding.record_backward_accumulate(
            commands,
            rows,
            &self.rosa_token_ids,
            self.rosa_adapter.grad_input_buffer(),
        )
    }

    pub(crate) fn memory_gate_warmup_step_value(&self) -> f32 {
        self.memory_gate_warmup_step as f32
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn persistent_values(&self) -> Result<Vec<f32>> {
        self.persistent.read_f32(self.config.persistent_dim)
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn rosa_workgroup_size(&self) -> u32 {
        self.rosa_workgroup_size
    }

    pub fn rosa_uses_subgroup_reduction(&self) -> bool {
        self.rosa_subgroup_reduction
    }

    pub fn rosa_kernel_label(&self) -> &'static str {
        self.rosa_kernel_label
    }

    pub fn rosa_was_autotuned(&self) -> bool {
        self.rosa_autotuned
    }

    fn record_input_uploads(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &HierarchosTokenFrontendInput<'_>,
        rows: usize,
    ) -> Result<()> {
        commands.upload_u32(&self.token_ids, input.token_ids)?;
        let token_len = rows * self.config.context_dim;
        match input.token_residual {
            Some(values) => commands.upload_f32(&self.token_residual, values)?,
            None => commands.fill_zero_f32(&self.token_residual, token_len)?,
        }
        commands.upload_f32(&self.gated_ltm_values, input.gated_ltm_values)?;
        Ok(())
    }

    fn record_forward(&self, commands: &mut vulkan::ComputeBatch, rows: usize) -> Result<()> {
        self.embedding
            .record_forward(commands, rows, &self.token_ids, &self.token_features)?;
        self.record_projection_forward(commands, rows, &self.token_features)
    }

    fn record_projection_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        rows: usize,
        token_source: &GpuBuffer,
    ) -> Result<()> {
        let token_len = rows * self.config.context_dim;
        let mac_len = rows * self.input_dim;
        let push = self.push(rows);
        self.assemble.record_dispatch(
            commands,
            &[
                token_source,
                &self.token_residual,
                &self.persistent,
                &self.gated_ltm_values,
                &self.mac_input,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(mac_len, 256), 1, 1],
        )?;
        self.in_proj
            .record_forward(commands, rows, &self.mac_input)?;
        let len_push = LenPush {
            len: token_len as u32,
        };
        self.gelu_forward.record_dispatch(
            commands,
            &[self.in_proj.output_buffer(), &self.gelu_output],
            bytemuck::bytes_of(&len_push),
            [div_ceil_u32(token_len, 256), 1, 1],
        )?;
        self.finite_clamp
            .record_forward(commands, token_len, &self.gelu_output, &self.enc, 30.0)
    }

    fn validate_memory_input(
        &self,
        input: &HierarchosTokenMemoryFrontendInput<'_>,
    ) -> Result<usize> {
        let rows = self.validate_memory_token_rows(input.token_ids)?;
        let context_len = rows * self.config.context_dim;
        if input.prev_context.len() != context_len
            || input.prev_context.iter().any(|value| !value.is_finite())
        {
            bail!(
                "memory-native prev_context must contain {context_len} finite values; got {}",
                input.prev_context.len()
            );
        }
        Ok(rows)
    }

    fn validate_memory_token_rows(&self, token_ids: &[u32]) -> Result<usize> {
        self.embedding.validate_token_ids(token_ids)?;
        let rows = token_ids.len();
        if rows == 0 || rows > self.max_rows {
            bail!(
                "memory-native token front-end rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        Ok(rows)
    }

    fn validate_input(&self, input: &HierarchosTokenFrontendInput<'_>) -> Result<usize> {
        self.embedding.validate_token_ids(input.token_ids)?;
        let rows = input.token_ids.len();
        if rows == 0 || rows > self.max_rows {
            bail!(
                "token front-end rows must be in 1..={}; got {rows}",
                self.max_rows
            );
        }
        let token_len = rows * self.config.context_dim;
        if let Some(values) = input.token_residual {
            if values.len() != token_len || values.iter().any(|value| !value.is_finite()) {
                bail!(
                    "token_residual must contain {token_len} finite values; got {}",
                    values.len()
                );
            }
        }
        let ltm_len = rows * self.ltm_dim;
        if input.gated_ltm_values.len() != ltm_len
            || input
                .gated_ltm_values
                .iter()
                .any(|value| !value.is_finite())
        {
            bail!(
                "gated_ltm_values must contain {ltm_len} finite values; got {}",
                input.gated_ltm_values.len()
            );
        }
        Ok(rows)
    }

    fn push(&self, rows: usize) -> FrontendPush {
        FrontendPush {
            rows: rows as u32,
            context_dim: self.config.context_dim as u32,
            persistent_dim: self.config.persistent_dim as u32,
            ltm_dim: self.ltm_dim as u32,
        }
    }
}

fn rebuild_bounded_rosa_device_state(
    histories: &[Vec<u32>],
    max_rows: usize,
    max_context: usize,
    vocab_size: usize,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    if max_context == 0 {
        bail!("bounded ROSA restore requires a positive max_context");
    }
    if histories.is_empty() || histories.len() > max_rows {
        bail!(
            "bounded ROSA restore requires 1..={max_rows} lane histories; got {}",
            histories.len()
        );
    }

    let generation_stride = if max_context <= u16::MAX as usize {
        (max_context + 1) / 2
    } else {
        max_context
    };
    let mut history = vec![0u32; max_rows * max_context];
    let mut history_len = vec![0u32; max_rows];
    let mut match_state = vec![0u32; max_rows * generation_stride * 2];

    for (lane, tokens) in histories.iter().enumerate() {
        if tokens.len() > max_context {
            bail!(
                "portable ROSA lane {lane} contains {} tokens, exceeding bounded max_context {max_context}",
                tokens.len()
            );
        }
        if let Some((index, token)) = tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token as usize >= vocab_size)
        {
            bail!(
                "portable ROSA lane {lane} token {index} has id {token}, outside vocab_size {vocab_size}"
            );
        }
        history_len[lane] =
            u32::try_from(tokens.len()).context("ROSA history length exceeds u32")?;
        let history_base = lane * max_context;
        history[history_base..history_base + tokens.len()].copy_from_slice(tokens);

        let lane_state_base = lane * generation_stride * 2;
        let mut previous = vec![0u32; max_context];
        for current in 1..tokens.len() {
            let mut next = vec![0u32; max_context];
            for prior in 0..current {
                if tokens[current] == tokens[prior] {
                    next[prior] = 1 + if prior == 0 { 0 } else { previous[prior - 1] };
                }
            }

            let generation_base = lane_state_base + (current & 1) * generation_stride;
            if max_context <= u16::MAX as usize {
                for pair in 0..current.div_ceil(2) {
                    let first = next[pair * 2];
                    let second = next.get(pair * 2 + 1).copied().unwrap_or(0);
                    debug_assert!(first <= u16::MAX as u32 && second <= u16::MAX as u32);
                    match_state[generation_base + pair] = first | (second << 16);
                }
            } else {
                match_state[generation_base..generation_base + current]
                    .copy_from_slice(&next[..current]);
            }
            previous = next;
        }
    }

    Ok((history, history_len, match_state))
}

fn extract_bounded_rosa_histories(
    history: &[u32],
    history_len: &[u32],
    rows: usize,
    max_context: usize,
    vocab_size: usize,
) -> Result<Vec<Vec<u32>>> {
    if rows == 0 || max_context == 0 {
        bail!("bounded ROSA snapshot requires positive rows and max_context");
    }
    let expected_history = rows
        .checked_mul(max_context)
        .context("ROSA snapshot geometry overflow")?;
    if history.len() != expected_history || history_len.len() != rows {
        bail!(
            "bounded ROSA snapshot geometry mismatch: history={} lengths={} expected_history={expected_history} rows={rows}",
            history.len(),
            history_len.len()
        );
    }

    let mut histories = Vec::with_capacity(rows);
    for lane in 0..rows {
        let len =
            usize::try_from(history_len[lane]).context("ROSA history length exceeds usize")?;
        if len > max_context {
            bail!(
                "bounded ROSA lane {lane} reports history length {len}, exceeding max_context {max_context}"
            );
        }
        let base = lane * max_context;
        let tokens = history[base..base + len].to_vec();
        if let Some((index, token)) = tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token as usize >= vocab_size)
        {
            bail!(
                "bounded ROSA lane {lane} token {index} has id {token}, outside vocab_size {vocab_size}"
            );
        }
        histories.push(tokens);
    }
    Ok(histories)
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

fn read_scalar_f32(path: &Path, name: &str) -> Result<f32> {
    let (shape, values) = read_f32_tensor(path, name)?;
    if !(shape.is_empty() || shape == vec![1]) || values.len() != 1 || !values[0].is_finite() {
        bail!("{name} must be one finite FP32 scalar; got shape {shape:?}");
    }
    Ok(values[0])
}

fn resolved_memory_gate_floor(config: &ModelConfig, warmup_step: f32) -> f32 {
    if config.memory_gate_warmup_steps <= 0.0 || config.memory_gate_warmup_floor <= 0.0 {
        return 0.0;
    }
    let floor = config.memory_gate_warmup_floor.clamp(0.0, 0.95);
    let progress = (warmup_step / config.memory_gate_warmup_steps).clamp(0.0, 1.0);
    floor * (1.0 - progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosa_workgroup_tracks_packed_state_geometry() {
        assert_eq!(automatic_rosa_workgroup_size(64, 1), 64);
        assert_eq!(automatic_rosa_workgroup_size(64, 128), 64);
        assert_eq!(automatic_rosa_workgroup_size(64, 129), 128);
        assert_eq!(automatic_rosa_workgroup_size(64, 256), 128);
        assert_eq!(automatic_rosa_workgroup_size(64, 257), 256);
        assert_eq!(automatic_rosa_workgroup_size(64, 512), 256);

        assert_eq!(automatic_rosa_workgroup_size(32, 64), 32);
        assert_eq!(automatic_rosa_workgroup_size(32, 65), 64);
        assert_eq!(automatic_rosa_workgroup_size(32, 512), 256);

        assert_eq!(automatic_rosa_workgroup_size(128, 1), 128);
        assert_eq!(automatic_rosa_workgroup_size(128, 256), 128);
        assert_eq!(automatic_rosa_workgroup_size(128, 512), 256);
    }

    #[test]
    fn rebuilt_bounded_rosa_state_matches_shader_generation_layout() -> Result<()> {
        let histories = vec![vec![2, 5, 2], vec![3, 3]];
        let (history, history_len, match_state) =
            rebuild_bounded_rosa_device_state(&histories, 2, 8, 16)?;

        assert_eq!(history_len, vec![3, 2]);
        assert_eq!(&history[..8], &[2, 5, 2, 0, 0, 0, 0, 0]);
        assert_eq!(&history[8..], &[3, 3, 0, 0, 0, 0, 0, 0]);

        // max_context=8 packs two u16 match lengths into each u32, with four
        // words per generation. Lane zero's latest transition is current=2,
        // therefore generation zero contains match[0]=1. Lane one's latest
        // transition is current=1, therefore generation one contains the same
        // one-token match. This is the src generation each next shader step
        // consumes after a portable resume.
        assert_eq!(match_state.len(), 16);
        assert_eq!(match_state[0], 1);
        assert_eq!(&match_state[1..8], &[0; 7]);
        assert_eq!(&match_state[8..12], &[0; 4]);
        assert_eq!(match_state[12], 1);
        assert_eq!(&match_state[13..], &[0; 3]);
        assert_eq!(
            extract_bounded_rosa_histories(&history, &history_len, 2, 8, 16)?,
            histories
        );
        Ok(())
    }

    fn run_rosa_lane_steps(
        device: &VulkanDevice,
        spirv: &[u8],
        steps: &[Vec<u32>],
        max_context: usize,
    ) -> Result<Vec<Vec<(u32, u32)>>> {
        let rows = steps.first().map_or(0, Vec::len);
        anyhow::ensure!(rows > 0, "ROSA test needs at least one lane");
        anyhow::ensure!(
            steps.iter().all(|step| step.len() == rows),
            "ROSA test lane width changed between steps"
        );
        let kernel = vulkan::ComputeKernel::new(
            device,
            spirv,
            7,
            std::mem::size_of::<RosaPredictPush>() as u32,
        )?;
        let generation_stride = (max_context + 1) / 2;
        let history = GpuBuffer::zeros_u32(device, rows * max_context)?;
        let history_len = GpuBuffer::zeros_u32(device, rows)?;
        let token_ids = GpuBuffer::zeros_u32(device, rows)?;
        let reset_lanes = GpuBuffer::zeros_u32(device, rows)?;
        let match_state = GpuBuffer::zeros_u32(device, rows * generation_stride * 2)?;
        let predictions = GpuBuffer::zeros_u32(device, rows)?;
        let valid = GpuBuffer::zeros_u32(device, rows)?;
        let push = RosaPredictPush {
            rows: rows as u32,
            max_context: max_context as u32,
        };
        let mut actual = Vec::with_capacity(steps.len());
        for (step_index, step) in steps.iter().enumerate() {
            token_ids.write_u32(step)?;
            reset_lanes.write_u32(&vec![u32::from(step_index == 0); rows])?;
            let mut commands = vulkan::ComputeBatch::new(device)?;
            kernel.record_dispatch(
                &mut commands,
                &[
                    &history,
                    &history_len,
                    &token_ids,
                    &reset_lanes,
                    &match_state,
                    &predictions,
                    &valid,
                ],
                bytemuck::bytes_of(&push),
                [rows as u32, 1, 1],
            )?;
            commands.submit()?;
            let prediction_bits = predictions
                .read_f32(rows)?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let valid_bits = valid
                .read_f32(rows)?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            actual.push(
                prediction_bits
                    .into_iter()
                    .zip(valid_bits)
                    .collect::<Vec<_>>(),
            );
        }
        Ok(actual)
    }

    #[test]
    fn rosa_cache_tiled_candidates_match_untiled_recurrence() -> Result<()> {
        let device = VulkanDevice::new()?;
        if !device.supports_compute_subgroup_arithmetic()
            || !device.supports_compute_work_group_size_x(256)
        {
            return Ok(());
        }
        let steps = (0..73usize)
            .map(|step| {
                vec![
                    [2u32, 5, 2, 5, 2, 5, 7][step % 7],
                    [3u32, 4, 3, 4, 9, 3, 4, 3][step % 8],
                    [11u32, 6, 11, 8, 11, 6, 11, 9, 11][step % 9],
                ]
            })
            .collect::<Vec<_>>();
        let max_context = 512usize;

        let untiled_128 = run_rosa_lane_steps(
            &device,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_SPV,
            &steps,
            max_context,
        )?;
        let tiled_128 = run_rosa_lane_steps(
            &device,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_CACHE_TILED_SPV,
            &steps,
            max_context,
        )?;
        assert_eq!(tiled_128, untiled_128);

        let untiled_256 = run_rosa_lane_steps(
            &device,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_SPV,
            &steps,
            max_context,
        )?;
        let tiled_256 = run_rosa_lane_steps(
            &device,
            ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_CACHE_TILED_SPV,
            &steps,
            max_context,
        )?;
        assert_eq!(tiled_256, untiled_256);
        assert_eq!(untiled_256, untiled_128);
        Ok(())
    }

    #[test]
    #[ignore = "GPU occupancy microprofile; run explicitly on the target Vulkan device"]
    fn profile_rosa_subgroup_workgroups_512() -> Result<()> {
        let device = VulkanDevice::new()?;
        let caps = device.subgroup_capabilities();
        println!(
            "ROSA profile device={} subgroup_size={} compute={} basic={} arithmetic={}",
            device.name(),
            caps.subgroup_size,
            caps.compute_supported,
            caps.basic_supported,
            caps.arithmetic_supported
        );

        let rows = 32usize;
        let max_context = 512usize;
        let mut variants = vec![("shared64", 64u32, ROSA_PREDICT_BOUNDED_LANES_SPV)];
        if device.supports_compute_subgroup_arithmetic() {
            variants.extend_from_slice(&[
                ("subgroup32", 32, ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_32_SPV),
                ("subgroup64", 64, ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_64_SPV),
                (
                    "subgroup128",
                    128,
                    ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_SPV,
                ),
                (
                    "subgroup128-cache-tiled",
                    128,
                    ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_128_CACHE_TILED_SPV,
                ),
                (
                    "subgroup256",
                    256,
                    ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SPV,
                ),
                (
                    "subgroup256-single-pair",
                    256,
                    ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_SPV,
                ),
                (
                    "subgroup256-single-pair-cache-tiled",
                    256,
                    ROSA_PREDICT_BOUNDED_LANES_SUBGROUP_256_SINGLE_PAIR_CACHE_TILED_SPV,
                ),
            ]);
        }

        for (label, width, spirv) in variants {
            if !device.supports_compute_work_group_size_x(width) {
                println!("{label}: skipped (workgroup width {width} unsupported)");
                continue;
            }
            // Prime shader/pipeline caches and clocks before collecting a wider
            // median. The timing itself still covers only command submission and
            // device completion for the 512-step recorded recurrence.
            let _ = time_rosa_lane_kernel(&device, spirv, rows, max_context)?;
            let mut samples = Vec::with_capacity(5);
            for _ in 0..5 {
                samples.push(time_rosa_lane_kernel(&device, spirv, rows, max_context)?);
            }
            samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
            println!(
                "{label}: width={width} median_ms={:.3} samples_ms={samples:?}",
                samples[2]
            );
        }
        Ok(())
    }
}
