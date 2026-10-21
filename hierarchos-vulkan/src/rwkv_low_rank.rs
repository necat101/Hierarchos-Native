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

/// These are distinct FP32 programs on the qualified AMD driver. Timing must
/// never select between them: the historical trajectory fixes the association.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum RwkvFanInArithmetic {
    LegacyFused,
    Materialized,
}

#[derive(Deserialize)]
struct FanInReferenceProfiles {
    schema: u32,
    device: String,
    materialized: Vec<FanInReferenceGeometry>,
}

#[derive(Deserialize)]
struct FanInReferenceGeometry {
    width: usize,
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    max_batch: usize,
}

fn fan_in_reference_arithmetic(
    device: &str,
    width: usize,
    ranks: (usize, usize, usize),
    max_batch: usize,
) -> Option<RwkvFanInArithmetic> {
    static PROFILES: OnceLock<FanInReferenceProfiles> = OnceLock::new();
    let profiles = PROFILES.get_or_init(|| {
        let profiles: FanInReferenceProfiles = serde_json::from_str(include_str!(
            "../validation/rwkv_fan_in_reference_profiles.json"
        ))
        .expect("valid independently established fan-in reference profiles");
        assert_eq!(profiles.schema, 1);
        profiles
    });
    if device != profiles.device {
        return None;
    }
    Some(
        if profiles.materialized.iter().any(|row| {
            row.width == width
                && (row.w_rank, row.a_rank, row.g_rank) == ranks
                && row.max_batch == max_batch
        }) {
            RwkvFanInArithmetic::Materialized
        } else {
            // The original fused module is the conservative reference for shapes
            // without a recorded materialized trajectory. It is never rewritten.
            RwkvFanInArithmetic::LegacyFused
        },
    )
}

fn time_mix_fan_in_spirv(rdna3: bool, outer: bool, workgroup: usize) -> &'static [u8] {
    let index = match workgroup {
        32 => 0,
        64 => 1,
        128 => 2,
        _ => unreachable!("validated fan-in workgroup"),
    };
    match (rdna3, outer) {
        (true, false) => [
            include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_amd_rdna3_wg32.spv")
                .as_slice(),
            include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_amd_rdna3.spv").as_slice(),
            include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_amd_rdna3_wg128.spv")
                .as_slice(),
        ][index],
        (true, true) => [
            include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_outer_amd_rdna3_wg32.spv")
                .as_slice(),
            include_bytes!("../shaders/rwkv_time_mix3_backward_fused_add_outer_amd_rdna3.spv")
                .as_slice(),
            include_bytes!(
                "../shaders/rwkv_time_mix3_backward_fused_add_outer_amd_rdna3_wg128.spv"
            )
            .as_slice(),
        ][index],
        (false, false) => [
            TIME_MIX_BACKWARD_FUSED_ADD_WG32_SPV,
            TIME_MIX_BACKWARD_FUSED_ADD_SPV,
            TIME_MIX_BACKWARD_FUSED_ADD_WG128_SPV,
        ][index],
        (false, true) => [
            TIME_MIX_BACKWARD_FUSED_ADD_OUTER_WG32_SPV,
            TIME_MIX_BACKWARD_FUSED_ADD_OUTER_SPV,
            TIME_MIX_BACKWARD_FUSED_ADD_OUTER_WG128_SPV,
        ][index],
    }
}
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
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_TWO_ROWS_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_low_rank_producer_forward_fused_two_rows.spv");
const LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_COLUMN_SLICES_SPV: &[u8] =
    include_bytes!("../shaders/layer_norm_low_rank_producer_forward_fused_column_slices.spv");
const HIERARCHOS_RWKV_LOW_RANK_DISABLE_LN_FORWARD_AUTOTUNE_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_DISABLE_LN_FORWARD_AUTOTUNE";
const HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG";
/// Force one LN1/low-rank producer topology instead of the measured choice, in either
/// family. The FP32 variants are `one-row`, `two-rows` and `column-slices`; the
/// packed-FP16 family -- the one the FP16-storage policy actually dispatches -- offers
/// `one-row`, `one-row-cached`, `one-row-rank128`, `two-rows`, `two-rows-wide` and
/// `four-rows`. One variable covers both because they are the same scheduling trade on
/// the same shapes, and `one-row`/`two-rows` deliberately name one variant in each: a
/// run under the FP16-storage policy then exercises the variant it asked for wherever
/// that variant exists. Every variant is arithmetic-identical to the kernel its own
/// timing selects, and the committed trajectories were recorded with the structural
/// default, so this is how a verification run exercises a variant the device's own
/// timing would not choose.
const HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY";
/// Name a file to append every measured LN1/low-rank topology decision to, one JSON
/// object per line. The producer stays the source of the measurement while the
/// checked-in record and its policy live in `validation/topology_decisions.json`,
/// owned by `validation/verify_topology_decisions.py`.
const HIERARCHOS_RWKV_LOW_RANK_TOPOLOGY_RECORD_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_TOPOLOGY_RECORD";
/// Hold the dense forward on the fused LN1/low-rank producer at every geometry,
/// ignoring the measured record. This is the first arm of the rank-2 A/B in
/// `validation/FUSED_FORWARD_PRODUCER_AUDIT.md`; setting both this and
/// [`HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN_ENV`] is a contradiction
/// and construction fails instead of silently picking one.
pub const HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN";
/// Hold the dense forward on the staged chain at every geometry: the standalone
/// layer norm, then the interpolation, the three first-stage matmuls and the
/// second-stage matmuls as separate dispatches. This is the second arm of the
/// rank-2 A/B. Like the force switch it is a construction-time measurement
/// switch; the shipped default follows
/// `validation/low_rank_full_forward_decisions.json` instead.
pub const HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN";
/// Measurement-only switch: count this op's forward kernels, so the rank-2 A/B
/// can prove each leg dispatched the graph it claims. The counter is off by
/// default; a training step must not pay for an atomic per forward just so a
/// benchmark can read the number.
pub const HIERARCHOS_RWKV_LOW_RANK_COUNT_FORWARD_DISPATCHES_ENV: &str =
    "HIERARCHOS_RWKV_LOW_RANK_COUNT_FORWARD_DISPATCHES";

/// The per-device answer to "does fusing LN1 with the low-rank forward beat
/// running the chain step by step at this geometry?". The table is the record the
/// rank-2 A/B writes; `compose` means the staged chain won for that
/// (adapter, width, batch).
fn low_rank_full_forward_decisions() -> &'static crate::decision_records::DecisionTable {
    static TABLE: std::sync::OnceLock<crate::decision_records::DecisionTable> =
        std::sync::OnceLock::new();
    crate::decision_records::table(
        &TABLE,
        include_str!("../validation/low_rank_full_forward_decisions.json"),
        "rwkv-low-rank-full-forward",
    )
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            character if (character as u32) < 0x20 => {
                escaped.push_str(&format!("\\u{:04x}", character as u32))
            }
            character => escaped.push(character),
        }
    }
    escaped
}

/// One autotune's verdict in the shape the record writes it.
///
/// The evidence travels as one value rather than as seven positional arguments, so a
/// geometry with more candidates and the next producer's autotuner write the same fields
/// without either call site having to keep two argument lists in step. Everything here is
/// a measurement: the medians the choice was made from, how many rounds each candidate
/// won, the narrowest per-round lead of the selected variant, and whether the rounds were
/// quiet enough to adjudicate at all.
#[derive(Clone, Copy)]
struct LayerNormLowRankForwardTopologyEvidence<'a> {
    /// Median per candidate: the set the record carries as `timings_ms`.
    timings: &'a [(String, f64)],
    /// How many rounds each candidate beat the default by the material margin.
    round_wins: &'a [(String, usize)],
    /// Rounds measured: a candidate that won all of them is unanimous.
    rounds: usize,
    /// The narrowest per-round lead of the selected variant, or `0.0` when the default was
    /// kept.
    decisive_margin: f64,
    /// Adapter/geometry-specific performance admission margin, not a numerical tolerance.
    required_margin: f64,
    /// The variant to dispatch.
    selected: &'a str,
    /// The variant the topology rule names at this geometry, before any measurement.
    structural_default: &'a str,
    /// How far the structural default's own per-round samples spread (slowest/fastest).
    contention_spread: f64,
    /// True when that spread says the rounds did not all measure the same machine, so the
    /// default was kept because the measurement was not evidence.
    contended: bool,
}

/// Publish one measured topology decision as a JSON line, when asked to.
///
/// `timings` is the same median-per-candidate set the decision was made from, so
/// the record carries both the choice and what it was chosen against. `round_wins` is
/// how many of the `rounds` interleaved rounds each candidate beat the default by the
/// material margin, and `decisive_margin` is the narrowest per-round lead of the
/// selected variant, so the record shows whether a variant was chosen unanimously or the
/// default was kept because no variant won every round. `contended` records the one case
/// where the rounds were not evidence at all -- the default's own samples moved by more
/// than the contention threshold -- so a replay on a busy device says that instead of
/// handing the reader a decision that the load, not the hardware, produced. Lines are
/// appended rather than rewritten: a decision is emitted once per process (the cache holds
/// it afterwards), and an interrupted run still says what it measured.
fn emit_topology_decision(
    family: &str,
    device: &str,
    subgroup_size: u32,
    key: &LayerNormLowRankForwardAutotuneKey,
    evidence: &LayerNormLowRankForwardTopologyEvidence<'_>,
) {
    let LayerNormLowRankForwardTopologyEvidence {
        timings,
        round_wins,
        rounds,
        decisive_margin,
        required_margin,
        selected,
        structural_default,
        contention_spread,
        contended,
    } = *evidence;
    let Some(path) = std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_TOPOLOGY_RECORD_ENV) else {
        return;
    };
    let best_ms = timings
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let default_ms = timings
        .iter()
        .find(|(label, _)| label == structural_default)
        .map(|(_, ms)| *ms);
    let margin_fraction = match default_ms {
        Some(default_ms) if default_ms.is_finite() && default_ms > 0.0 && best_ms.is_finite() => {
            (default_ms - best_ms) / default_ms
        }
        _ => 0.0,
    };
    let timings = timings
        .iter()
        .map(|(label, ms)| format!("\"{}\":{ms:.6}", json_escape(label)))
        .collect::<Vec<_>>()
        .join(",");
    let round_wins = round_wins
        .iter()
        .map(|(label, wins)| format!("\"{}\":{wins}", json_escape(label)))
        .collect::<Vec<_>>()
        .join(",");
    let line = format!(
        concat!(
            "{{\"family\":\"{}\",\"device\":\"{}\",\"subgroup_size\":{},",
            "\"width\":{},\"w_rank\":{},\"a_rank\":{},\"g_rank\":{},",
            "\"batch_pairs\":{},\"has_unpaired_tail\":{},",
            "\"timings_ms\":{{{}}},\"rounds\":{},\"round_wins\":{{{}}},",
            "\"decisive_margin_fraction\":{:.6},",
            "\"required_margin_fraction\":{:.6},",
            "\"selected\":\"{}\",\"structural_default\":\"{}\",",
            "\"margin_fraction\":{:.6},",
            "\"contention_spread\":{:.4},\"contended\":{}}}"
        ),
        json_escape(family),
        json_escape(device),
        subgroup_size,
        key.width,
        key.w_rank,
        key.a_rank,
        key.g_rank,
        key.batch_pairs,
        key.has_unpaired_tail,
        timings,
        rounds,
        round_wins,
        decisive_margin,
        required_margin,
        json_escape(selected),
        json_escape(structural_default),
        margin_fraction,
        contention_spread,
        contended
    );
    use std::io::Write as _;
    if let Err(error) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| writeln!(file, "{line}"))
    {
        eprintln!(
            "{}={}: {error}",
            HIERARCHOS_RWKV_LOW_RANK_TOPOLOGY_RECORD_ENV,
            path.to_string_lossy()
        );
    }
}
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

#[derive(Clone, Copy)]
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

    /// The `HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY` value that selects this
    /// topology, exactly as the FP32 table does it: the parser, the rejection message
    /// and the acceptance message all read this one string, so a value cannot name one
    /// topology when it is parsed and another when the run reports what it ran. The
    /// verification harness credits an arm from those reports.
    fn env_value(self) -> &'static str {
        match self {
            Self::OneRow => "one-row",
            Self::OneRowCached => "one-row-cached",
            Self::OneRowRank128 => "one-row-rank128",
            Self::TwoRows => "two-rows",
            Self::TwoRowsWide => "two-rows-wide",
            Self::FourRows => "four-rows",
        }
    }

    const ALL: [Self; 6] = [
        Self::OneRow,
        Self::OneRowCached,
        Self::OneRowRank128,
        Self::TwoRows,
        Self::TwoRowsWide,
        Self::FourRows,
    ];
}

/// Dispatch topology for the FP32 LN1/low-rank producer. The FP32 path shipped
/// one kernel: `rows` 64-lane workgroups, one row each. The pair variant gives one
/// workgroup two rows whose weight reads are shared, which halves the weight bytes
/// per row at the cost of half the workgroups, so the choice is measured on the
/// device rather than assumed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum LayerNormLowRankForwardF32Topology {
    OneRow,
    TwoRows,
    ColumnSlices,
}

impl LayerNormLowRankForwardF32Topology {
    fn label(self) -> &'static str {
        match self {
            Self::OneRow => "ln-low-rank-f32-one-row",
            Self::TwoRows => "ln-low-rank-f32-two-rows",
            Self::ColumnSlices => "ln-low-rank-f32-column-slices",
        }
    }

    fn rows_per_workgroup(self) -> usize {
        match self {
            Self::OneRow => 1,
            Self::TwoRows => 2,
            // Column slices replicate a row across workgroups rather than grouping
            // rows inside one: the grid is [ceil(width/64), rows, 1].
            Self::ColumnSlices => 1,
        }
    }

    /// The dispatch grid this topology's kernel reads. Only `ColumnSlices` maps a
    /// second grid dimension: its kernel takes the row from `gl_WorkGroupID.y` and
    /// its column slice from `gl_WorkGroupID.x`, so dispatching it with the
    /// row-per-workgroup grid would leave every row but the first unwritten. Both the
    /// timing path and the production path ask this method, because a grid that
    /// disagrees with the kernel's own mapping is a wrong answer rather than a slow
    /// one, and only a golden catches it.
    fn groups(self, batch: usize, width: usize) -> [u32; 3] {
        match self {
            Self::ColumnSlices => [(width as u32).div_ceil(64), batch as u32, 1],
            _ => [batch.div_ceil(self.rows_per_workgroup()) as u32, 1, 1],
        }
    }

    /// The `HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY` value that selects this
    /// topology. The parser, the rejection message and the acceptance message all
    /// read the same string, so a value cannot name one topology when parsed and
    /// another when reported: the verification harness reads the request the producer
    /// acknowledged out of that log, and an arm is only credited where the value it
    /// asked for is the value that ran.
    fn env_value(self) -> &'static str {
        match self {
            Self::OneRow => "one-row",
            Self::TwoRows => "two-rows",
            Self::ColumnSlices => "column-slices",
        }
    }

    const ALL: [Self; 3] = [Self::OneRow, Self::TwoRows, Self::ColumnSlices];
}

#[derive(Clone, Copy, Debug)]
struct LayerNormLowRankForwardF32Decision {
    topology: LayerNormLowRankForwardF32Topology,
    autotuned: bool,
}

/// The FP32 topology cache is separate from the FP16 one: their keys describe the
/// same shapes, but the two families select among different kernels.
static LAYER_NORM_LOW_RANK_FORWARD_F32_AUTOTUNE_CACHE: OnceLock<
    Mutex<HashMap<LayerNormLowRankForwardAutotuneKey, LayerNormLowRankForwardF32Decision>>,
> = OnceLock::new();

/// Every value the override accepts, from either family's table.
///
/// The two families accept different subsets -- `column-slices` is FP32-only,
/// `one-row-cached` and the other four are FP16-only, and `one-row`/`two-rows` appear in
/// both -- so a value that one family does not offer is normal and is answered by that
/// family's own rejection path, which the verification harness reads. Only a value in no
/// table at all is a typo, and that is the only case worth a message. The list is
/// deduplicated so the message names each value once.
fn known_forced_topology_values() -> Vec<&'static str> {
    let mut values: Vec<&'static str> = Vec::with_capacity(9);
    for value in LayerNormLowRankForwardF32Topology::ALL
        .into_iter()
        .map(LayerNormLowRankForwardF32Topology::env_value)
        .chain(
            LayerNormLowRankForwardTopology::ALL
                .into_iter()
                .map(LayerNormLowRankForwardTopology::env_value),
        )
    {
        if !values.contains(&value) {
            values.push(value);
        }
    }
    values
}

/// The requested override value, or `None` when unset, blank, or unknown.
fn forced_topology_value() -> Option<String> {
    let raw = std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY_ENV)?;
    let value = raw.to_str()?.trim();
    if value.is_empty() {
        return None;
    }
    if known_forced_topology_values().contains(&value) {
        return Some(value.to_owned());
    }
    let expected = known_forced_topology_values().join(", ");
    eprintln!(
        "{} expects one of {expected}; ignoring {value:?}",
        HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY_ENV
    );
    None
}

fn forced_layer_norm_forward_f32_topology() -> Option<LayerNormLowRankForwardF32Topology> {
    let value = forced_topology_value()?;
    LayerNormLowRankForwardF32Topology::ALL
        .into_iter()
        .find(|topology| topology.env_value() == value)
}

fn forced_layer_norm_forward_fp16_topology() -> Option<LayerNormLowRankForwardTopology> {
    let value = forced_topology_value()?;
    LayerNormLowRankForwardTopology::ALL
        .into_iter()
        .find(|topology| topology.env_value() == value)
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

/// Rounds each LN1/low-rank autotuner measures every candidate in, interleaved.
///
/// Seven, not three, because unanimity is what makes the choice stable on a geometry whose
/// lead sits inside the run-to-run spread. On the packed-FP16 producer's width-128 shape
/// the shipped default and `one-row-cached` differ by about 3%, which is the same size as
/// the round-to-round noise: with three rounds a run still came out unanimous about 40% of
/// the time, so the schedule flipped. Seven carries that below a percent for a lead that is
/// a coin flip per round, and a variant that is genuinely faster (the FP32 column-slice
/// shape leads by 12-17%) stays unanimous. The cost is one more timing sweep per geometry
/// per process -- these kernels are tens of microseconds and the decision is cached, so it
/// is a few milliseconds against a training step's milliseconds.
const LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_ROUNDS: usize = 7;

/// How far a variant must beat the structural default in *every* round before it
/// displaces it, as a fraction of the default's own time in that round.
///
/// The median-only rule this replaced accepted a single 2%-margin test on the median of
/// three samples: on the packed-FP16 producer the same geometry selected `one-row-cached`
/// in three of five identical runs, which is a schedule that changes with nothing changed.
/// A variant now has to win the same way in every round, so a lead that lives inside the
/// run-to-run spread keeps the default and the variation shows up as the evidence rather
/// than as a different kernel.
const LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_MARGIN: f64 = 0.02;

fn f32_topology_adoption_margin(device: &str, width: usize, ranks: (usize, usize, usize)) -> f64 {
    // Repeated quiet Ally replays alternated between one-row and column-slices
    // on 3-6% leads while the default's own samples moved by up to 13.7%.
    // Require a lead beyond that admitted noise envelope in every round.
    // Preserve Intel and unmeasured adapter/driver/shape policies verbatim.
    if device == "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]"
        && width == 128 && ranks == (32, 32, 64)
    {
        LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_CONTENTION_SPREAD - 1.0
    } else {
        LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_MARGIN
    }
}

/// How many times each candidate is timed inside one round. The round takes the fastest
/// repeat, because a dispatcher that shares the GPU can only *add* time: contention makes a
/// sample slower, never faster, so the minimum is the closest estimate of the undisturbed
/// cost and one interrupted repeat cannot decide a round. Three is enough to drop a single
/// bad repeat without turning a decision the whole fixture waits on (the FP16 producer's
/// 3% lead) into a long sweep.
const LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_REPEATS: usize = 3;

/// How far the structural default's own per-round samples may spread, slowest over fastest,
/// before the rounds are declared too contended to adjudicate.
///
/// The unanimity rule protects the decision from a lead that lives inside the noise, but it
/// cannot protect it from rounds that measured a busy machine: with another process sharing
/// the GPU every round is slower by a factor the interleaving cannot cancel, and a variant
/// whose true lead is real can fail the margin in all seven. That is how the recorded
/// column-slice win at width 128 turned into "this device now selects one-row" on a loaded
/// machine. Fifteen percent is above the spread a quiet run shows on the Gen9 part (the
/// per-round medians move by a few percent) and below what a second process on the GPU
/// produces, so a loaded run keeps the structural default and says `contended` instead of
/// recording a choice the hardware did not make.
const LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_CONTENTION_SPREAD: f64 = 1.15;

/// One autotuner's verdict, with the evidence it was reached from.
struct LayerNormLowRankForwardAutotune<T> {
    /// The variant to dispatch: a candidate that beat the structural default in every
    /// round, or the structural default when no candidate did.
    selected: T,
    /// Median per candidate: the set the record carries as `timings_ms`.
    medians: Vec<(T, f64)>,
    /// How many rounds each candidate beat the default by the material margin.
    round_wins: Vec<(T, usize)>,
    /// Rounds measured: a candidate that won all of them is unanimous.
    rounds: usize,
    /// The narrowest per-round lead of the selected variant over the default. `None` means
    /// the default was kept, which is also the answer when the rounds disagree or when they
    /// were contended.
    decisive_margin: Option<f64>,
    /// How far the structural default's own per-round samples spread (slowest/fastest).
    /// `1.0` means every round measured the same default cost.
    contention_spread: f64,
    /// True when `contention_spread` crossed the threshold: the rounds did not all measure
    /// the same machine, so no candidate was allowed to displace the default and the record
    /// says the decision was not measured rather than recording the load.
    contended: bool,
}

/// Measure every candidate in interleaved rounds, reversing the candidate order each
/// round so a drifting clock cannot favour one position. A failed measurement is an error
/// for the whole autotune: a round with a missing candidate cannot be adjudicated.
fn measure_layer_norm_low_rank_forward_rounds<T, F>(
    candidates: &[T],
    mut time_ms: F,
) -> Result<Vec<Vec<(T, f64)>>>
where
    T: Copy,
    F: FnMut(T) -> Result<f64>,
{
    let mut rounds = Vec::with_capacity(LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_ROUNDS);
    for round in 0..LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_ROUNDS {
        let mut samples = Vec::with_capacity(candidates.len());
        for offset in 0..candidates.len() {
            let index = if round % 2 == 0 {
                offset
            } else {
                candidates.len() - 1 - offset
            };
            let candidate = candidates[index];
            // Repeats inside the round, keeping the fastest: contention only ever adds
            // time, so the minimum is the closest estimate of the undisturbed cost and a
            // single interrupted repeat cannot make a round read as a candidate's loss.
            let repeats = (0..LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_REPEATS)
                .map(|_| time_ms(candidate))
                .collect::<Result<Vec<_>>>()?;
            let fastest = repeats
                .iter()
                .copied()
                .filter(|ms| ms.is_finite() && *ms > 0.0)
                .fold(f64::INFINITY, f64::min);
            // A round whose every repeat was unusable carries the last one, so adjudication
            // sees it as unmeasured instead of as an implausibly fast candidate.
            let sample = if fastest.is_finite() {
                fastest
            } else {
                *repeats
                    .last()
                    .expect("a repeat count of zero would be a mistake here")
            };
            samples.push((candidate, sample));
        }
        rounds.push(samples);
    }
    Ok(rounds)
}

/// The structural default's own sample per round, in round order. These are the samples
/// contention is judged from: the default runs in every round by construction, so its own
/// spread is what says whether the rounds measured one machine or several.
fn default_round_samples<T: Copy + PartialEq>(
    rounds: &[Vec<(T, f64)>],
    structural_default: T,
) -> Vec<f64> {
    rounds
        .iter()
        .filter_map(|round| {
            round
                .iter()
                .find(|(measured, _)| *measured == structural_default)
                .map(|(_, ms)| *ms)
        })
        .collect()
}

/// Adjudicate interleaved rounds: a variant is selected only when it beats the structural
/// default by the material margin in every round, otherwise the default is kept.
///
/// The medians are still reported, because they are what the record holds, but they no
/// longer decide. A round is won by the fastest candidate that clears the margin against
/// that round's own default sample, ties going to the earlier candidate in `candidates`,
/// so the verdict is deterministic and does not depend on measurement order.
///
/// Before any of that, the rounds have to have measured one machine. Contention is not a
/// candidate that wins or loses, it is the ground moving under every candidate at once, and
/// a sweep taken under it is not evidence about the kernels: the default's own samples then
/// spread by more than the contention threshold in the round order they were taken, so the
/// verdict is the structural default with `contended` set, and the record says the decision
/// was not measured rather than recording what the load did.
fn adjudicate_layer_norm_low_rank_forward_rounds<T: Copy + PartialEq>(
    rounds: &[Vec<(T, f64)>],
    candidates: &[T],
    structural_default: T,
) -> LayerNormLowRankForwardAutotune<T> {
    adjudicate_layer_norm_low_rank_forward_rounds_with_margin(
        rounds, candidates, structural_default, LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_MARGIN,
    )
}

fn adjudicate_layer_norm_low_rank_forward_rounds_with_margin<T: Copy + PartialEq>(
    rounds: &[Vec<(T, f64)>],
    candidates: &[T],
    structural_default: T,
    required_margin: f64,
) -> LayerNormLowRankForwardAutotune<T> {
    let mut medians = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut samples = rounds
            .iter()
            .filter_map(|round| round.iter().find(|(measured, _)| measured == candidate))
            .map(|(_, ms)| *ms)
            .collect::<Vec<_>>();
        samples.sort_by(f64::total_cmp);
        medians.push((
            *candidate,
            if samples.is_empty() {
                f64::INFINITY
            } else {
                samples[samples.len() / 2]
            },
        ));
    }

    let mut round_wins = candidates
        .iter()
        .map(|candidate| (*candidate, 0usize))
        .collect::<Vec<_>>();
    for round in rounds {
        let Some(default_ms) = round
            .iter()
            .find(|(measured, _)| *measured == structural_default)
            .map(|(_, ms)| *ms)
        else {
            // A round without the default cannot be compared against it.
            continue;
        };
        let mut best: Option<(usize, f64)> = None;
        for (candidate, ms) in round {
            let position = candidates
                .iter()
                .position(|known| known == candidate)
                .expect("every measured candidate comes from the candidate list");
            if !ms.is_finite() || *ms <= 0.0 {
                continue;
            }
            if *ms >= default_ms * (1.0 - required_margin) {
                continue;
            }
            match best {
                // A tie goes to the earlier candidate rather than to whichever one the
                // round happened to measure first: the next round reverses that order, so
                // a verdict that followed it would depend on the measurement order.
                Some((best_position, best_ms)) => {
                    if *ms < best_ms || (*ms == best_ms && position < best_position) {
                        best = Some((position, *ms));
                    }
                }
                None => best = Some((position, *ms)),
            }
        }
        if let Some((position, _)) = best {
            round_wins[position].1 += 1;
        }
    }

    let rounds_measured = rounds.len();
    let default_samples = default_round_samples(rounds, structural_default);
    let contention_spread = {
        let fastest = default_samples
            .iter()
            .copied()
            .filter(|ms| ms.is_finite() && *ms > 0.0)
            .fold(f64::INFINITY, f64::min);
        let slowest = default_samples
            .iter()
            .copied()
            .filter(|ms| ms.is_finite() && *ms > 0.0)
            .fold(f64::NEG_INFINITY, f64::max);
        if fastest.is_finite() && slowest.is_finite() && fastest > 0.0 {
            slowest / fastest
        } else {
            // Nothing usable to judge from; the round-level rules below already keep the
            // default for that case.
            1.0
        }
    };
    let contended = contention_spread > LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_CONTENTION_SPREAD;
    let mut decided: Option<(T, usize)> = None;
    if !contended {
        for (position, candidate) in candidates.iter().enumerate() {
            if *candidate == structural_default || round_wins[position].1 == 0 {
                continue;
            }
            if round_wins[position].1 == rounds_measured {
                decided = Some((*candidate, position));
                break;
            }
        }
    }
    let decisive_margin = decided.map(|(_, position)| {
        let candidate = candidates[position];
        rounds
            .iter()
            .map(|round| {
                let default_ms = round
                    .iter()
                    .find(|(measured, _)| *measured == structural_default)
                    .map(|(_, ms)| *ms)
                    .unwrap_or_default();
                let candidate_ms = round
                    .iter()
                    .find(|(measured, _)| *measured == candidate)
                    .map(|(_, ms)| *ms)
                    .unwrap_or_default();
                (default_ms - candidate_ms) / default_ms
            })
            .fold(f64::INFINITY, f64::min)
    });
    LayerNormLowRankForwardAutotune {
        selected: decided
            .map(|(candidate, _)| candidate)
            .unwrap_or(structural_default),
        medians,
        round_wins,
        rounds: rounds_measured,
        decisive_margin,
        contention_spread,
        contended,
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
    fan_in_arithmetic: Option<RwkvFanInArithmetic>,
    materialized_fan_in_fusion: bool,

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
    /// `HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN`: skip the record and
    /// keep the fused producer wherever its kernels exist.
    fused_forward_forced: bool,
    /// `HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN`: skip the record and
    /// always dispatch the staged chain.
    fused_forward_disabled: bool,
    /// `HIERARCHOS_RWKV_LOW_RANK_COUNT_FORWARD_DISPATCHES`: measure-only counter of
    /// this op's forward kernels.
    count_forward_dispatches: bool,
    forward_dispatches: std::sync::atomic::AtomicU64,
    /// Dispatches of the deep FP16 full-forward producer specifically. The
    /// total above cannot tell it from the first-stage producer because both
    /// are one dispatch; this counter is what proves which arm ran.
    full_forward_dispatches: std::sync::atomic::AtomicU64,
    low_rank_producer_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused_two_rows: Option<vulkan::ComputeKernel>,
    layer_norm_low_rank_producer_forward_fused_column_slices: Option<vulkan::ComputeKernel>,
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
        // The two A/B switches name opposite graphs, so honoring both would make
        // the measured leg depend on which check ran first.
        let fused_forward_forced =
            std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN_ENV).is_some();
        let fused_forward_disabled =
            std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN_ENV).is_some();
        if fused_forward_forced && fused_forward_disabled {
            bail!(
                "{} and {} are both set; an A/B leg cannot hold both forward graphs",
                HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN_ENV,
                HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN_ENV
            );
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
        let layer_norm_low_rank_producer_forward_fused_two_rows =
            create_layer_norm_low_rank_producer_forward_kernel(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_TWO_ROWS_SPV,
                w_rank,
                a_rank,
                g_rank,
            )?;
        let layer_norm_low_rank_producer_forward_fused_column_slices =
            create_layer_norm_low_rank_producer_forward_kernel(
                &device,
                LAYER_NORM_LOW_RANK_PRODUCER_FORWARD_FUSED_COLUMN_SLICES_SPV,
                w_rank,
                a_rank,
                g_rank,
            )?;
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

        let fan_in_arithmetic = fan_in_reference_arithmetic(
            device.decision_device_name(),
            width,
            (w_rank, a_rank, g_rank),
            max_batch,
        );
        let materialized_fan_in_fusion = fan_in_arithmetic
            == Some(RwkvFanInArithmetic::Materialized)
            && crate::vendor::rdna3_module_enabled(&device);
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
            time_mix_fan_in_spirv(materialized_fan_in_fusion, false, 32),
            15,
            10,
            32,
        )?;
        let time_mix_backward_fused_add_wg128 = create_time_mix_backward_geometry_kernel(
            &device,
            time_mix_fan_in_spirv(materialized_fan_in_fusion, false, 128),
            15,
            10,
            128,
        )?;
        let time_mix_backward_fused_add_outer_wg32 = create_time_mix_backward_geometry_kernel(
            &device,
            time_mix_fan_in_spirv(materialized_fan_in_fusion, true, 32),
            16,
            11,
            32,
        )?;
        let time_mix_backward_fused_add_outer_wg128 = create_time_mix_backward_geometry_kernel(
            &device,
            time_mix_fan_in_spirv(materialized_fan_in_fusion, true, 128),
            16,
            11,
            128,
        )?;

        let time_mix_backward_fused_add = if device.supports_storage_buffer_bindings(15) {
            Some(vulkan::ComputeKernel::new_with_access(
                &device,
                time_mix_fan_in_spirv(materialized_fan_in_fusion, false, 64),
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
                time_mix_fan_in_spirv(materialized_fan_in_fusion, true, 64),
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
            fan_in_arithmetic,
            materialized_fan_in_fusion,
            time_mix_forward: vulkan::ComputeKernel::new(
                &device,
                TIME_MIX_FORWARD_SPV,
                8,
                std::mem::size_of::<MixPush>() as u32,
            )?,
            fused_forward_forced,
            fused_forward_disabled,
            count_forward_dispatches: std::env::var_os(
                HIERARCHOS_RWKV_LOW_RANK_COUNT_FORWARD_DISPATCHES_ENV,
            )
            .is_some(),
            forward_dispatches: std::sync::atomic::AtomicU64::new(0),
            full_forward_dispatches: std::sync::atomic::AtomicU64::new(0),
            low_rank_producer_forward_fused,
            layer_norm_low_rank_producer_forward_fused,
            layer_norm_low_rank_producer_forward_fused_two_rows,
            layer_norm_low_rank_producer_forward_fused_column_slices,
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

    /// Quantize this op's canonical FP32 low-rank matrices into the packed FP16
    /// execution mirrors the mixed-precision producer arms bind, and switch the
    /// op to FP16 parameter storage.
    ///
    /// This is the op-level analogue of the full training graph's
    /// `fp16-storage-fp32-compute` policy: the six matrices keep their FP32
    /// buffers as the canonical checkpoint/optimizer identity, and the mirrors
    /// are execution-only. One submission packs all six from the current FP32
    /// values, so a caller that mutates the FP32 masters afterwards must
    /// refresh them again (the training graph owns a per-step refresher for
    /// exactly that reason).
    ///
    /// The device must have created the FP16 producer specializations; the
    /// install step fails rather than silently leaving the FP32 arm active.
    pub fn enable_fp16_parameter_storage(&mut self) -> Result<()> {
        if self.fp16_parameter_mirrors.is_some() {
            return Ok(());
        }
        let mirror = |len: usize| {
            crate::mixed_precision::VulkanParameterStorageMirror::new(
                &self.device,
                VulkanParameterStorageFormat::Fp16,
                len,
            )
        };
        let w1 = mirror(self.width * self.w_rank)?;
        let w2 = mirror(self.w_rank * self.width)?;
        let a1 = mirror(self.width * self.a_rank)?;
        let a2 = mirror(self.a_rank * self.width)?;
        let g1 = mirror(self.width * self.g_rank)?;
        let g2 = mirror(self.g_rank * self.width)?;
        let refresher = crate::mixed_precision::VulkanParameterStorageMirrorRefresher::new(
            &self.device,
            VulkanParameterStorageFormat::Fp16,
        )?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        refresher.record_refresh(&mut commands, &self.w1, &w1)?;
        refresher.record_refresh(&mut commands, &self.w2, &w2)?;
        refresher.record_refresh(&mut commands, &self.a1, &a1)?;
        refresher.record_refresh(&mut commands, &self.a2, &a2)?;
        refresher.record_refresh(&mut commands, &self.g1, &g1)?;
        refresher.record_refresh(&mut commands, &self.g2, &g2)?;
        commands.submit()?;
        self.install_fp16_parameter_mirrors(RwkvLowRankFp16ParameterMirrors {
            w1,
            w2,
            a1,
            a2,
            g1,
            g2,
        })
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

    /// Whether this op was switched to FP16 execution-storage mirrors, i.e. the
    /// storage half of the training graph's `fp16-storage-fp32-compute` policy.
    pub fn fp16_parameter_storage_active(&self) -> bool {
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
        // The FP32 family has had this arm since its own variants were audited; the FP16
        // family needs it for the same reason. A forced variant this device cannot offer
        // is reported and answered by the structural default rather than silently
        // ignored: the equivalence suite reads these two lines to know whether an arm ran
        // the variant it asked for, and a run that fell back passes while proving nothing.
        if let Some(forced) = forced_layer_norm_forward_fp16_topology() {
            if candidates.contains(&forced) {
                if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some()
                {
                    eprintln!(
                        "RWKV LN1/low-rank FP16 forward forced topology={} requested={} device={} batch={batch}",
                        forced.label(),
                        forced.env_value(),
                        self.device.decision_device_name()
                    );
                }
                return Ok(LayerNormLowRankForwardDecision {
                    topology: forced,
                    autotuned: false,
                });
            }
            eprintln!(
                "RWKV LN1/low-rank FP16 forward topology={} requested={} is unavailable on device={}; using {}",
                forced.label(),
                forced.env_value(),
                self.device.decision_device_name(),
                structural_default.label()
            );
        }
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
            device_name: self.device.decision_device_name().to_owned(),
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
                    self.device.decision_device_name(),
                    structural_default.label()
                );
            }
            return Ok(LayerNormLowRankForwardDecision {
                topology: structural_default,
                autotuned: false,
            });
        }

        let rounds = match measure_layer_norm_low_rank_forward_rounds(&candidates, |topology| {
            time_topology(topology).with_context(|| {
                format!(
                    "timing RWKV LN1/low-rank FP16 forward candidate {}",
                    topology.label()
                )
            })
        }) {
            Ok(rounds) => rounds,
            Err(err) => {
                if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some()
                {
                    eprintln!(
                        "RWKV LN1/low-rank forward autotune failed device={} batch={batch}: {err:#}; using {}",
                        self.device.decision_device_name(),
                        structural_default.label()
                    );
                }
                return Ok(LayerNormLowRankForwardDecision {
                    topology: structural_default,
                    autotuned: false,
                });
            }
        };
        let adjudication =
            adjudicate_layer_norm_low_rank_forward_rounds(&rounds, &candidates, structural_default);
        let selected = adjudication.selected;
        let decision = LayerNormLowRankForwardDecision {
            topology: selected,
            autotuned: true,
        };

        if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some() {
            let summary = adjudication
                .medians
                .iter()
                .map(|(topology, ms)| format!("{}={ms:.5}ms", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            let wins = adjudication
                .round_wins
                .iter()
                .map(|(topology, wins)| format!("{}:{wins}", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV LN1/low-rank forward autotune device={} subgroup={} width={} ranks={}/{}/{} batch_pairs={} tail={} {} rounds={} round_wins={} selected={} contended={} autotuned={}",
                self.device.decision_device_name(),
                subgroup_size,
                self.width,
                self.w_rank,
                self.a_rank,
                self.g_rank,
                key.batch_pairs,
                key.has_unpaired_tail,
                summary,
                adjudication.rounds,
                wins,
                selected.label(),
                adjudication.contended,
                decision.autotuned
            );
        }
        let labels = adjudication
            .medians
            .iter()
            .map(|(topology, ms)| (topology.label().to_string(), *ms))
            .collect::<Vec<_>>();
        let wins = adjudication
            .round_wins
            .iter()
            .map(|(topology, wins)| (topology.label().to_string(), *wins))
            .collect::<Vec<_>>();
        emit_topology_decision(
            "ln-low-rank-fp16",
            self.device.decision_device_name(),
            subgroup_size,
            &key,
            &LayerNormLowRankForwardTopologyEvidence {
                timings: &labels,
                round_wins: &wins,
                rounds: adjudication.rounds,
                decisive_margin: adjudication.decisive_margin.unwrap_or(0.0),
                required_margin: LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_MARGIN,
                selected: selected.label(),
                structural_default: structural_default.label(),
                contention_spread: adjudication.contention_spread,
                contended: adjudication.contended,
            },
        );

        cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV LN1/low-rank forward autotune cache lock was poisoned")
            })?
            .insert(key, decision);
        Ok(decision)
    }

    fn available_f32_layer_norm_forward_topologies(
        &self,
    ) -> Vec<LayerNormLowRankForwardF32Topology> {
        let mut topologies = Vec::with_capacity(2);
        if self.layer_norm_low_rank_producer_forward_fused.is_some() {
            topologies.push(LayerNormLowRankForwardF32Topology::OneRow);
        }
        if self
            .layer_norm_low_rank_producer_forward_fused_two_rows
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardF32Topology::TwoRows);
        }
        if self
            .layer_norm_low_rank_producer_forward_fused_column_slices
            .is_some()
        {
            topologies.push(LayerNormLowRankForwardF32Topology::ColumnSlices);
        }
        topologies
    }

    /// The FP32 path keeps the one-row kernel -- the shape the committed
    /// trajectories were recorded with -- unless a measured candidate beats it by
    /// the material margin the selector applies.
    fn structural_f32_layer_norm_forward_topology(
        &self,
    ) -> Option<LayerNormLowRankForwardF32Topology> {
        if self.layer_norm_low_rank_producer_forward_fused.is_some() {
            Some(LayerNormLowRankForwardF32Topology::OneRow)
        } else if self
            .layer_norm_low_rank_producer_forward_fused_two_rows
            .is_some()
        {
            Some(LayerNormLowRankForwardF32Topology::TwoRows)
        } else {
            None
        }
    }

    fn f32_layer_norm_forward_kernel(
        &self,
        topology: LayerNormLowRankForwardF32Topology,
    ) -> Option<&vulkan::ComputeKernel> {
        match topology {
            LayerNormLowRankForwardF32Topology::OneRow => {
                self.layer_norm_low_rank_producer_forward_fused.as_ref()
            }
            LayerNormLowRankForwardF32Topology::TwoRows => self
                .layer_norm_low_rank_producer_forward_fused_two_rows
                .as_ref(),
            LayerNormLowRankForwardF32Topology::ColumnSlices => self
                .layer_norm_low_rank_producer_forward_fused_column_slices
                .as_ref(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_f32_layer_norm_forward_topology_into(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        previous: &GpuBuffer,
        norm: &LayerNormProducerInput<'_>,
        outputs: &LayerNormLowRankForwardOutputs<'_>,
        topology: LayerNormLowRankForwardF32Topology,
    ) -> Result<()> {
        let kernel = self
            .f32_layer_norm_forward_kernel(topology)
            .with_context(|| {
                format!(
                    "RWKV LN1/low-rank FP32 forward topology {} is unavailable",
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
        let groups = topology.groups(batch, self.width);
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
                &self.w1,
                &self.a1,
                &self.g1,
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
                &self.w2,
                &self.w0,
                outputs.w_pre,
                outputs.w,
                &self.a2,
                &self.a0,
                outputs.a_pre,
                outputs.a,
                outputs.g_sigmoid,
                &self.g2,
                outputs.g,
            ],
            bytemuck::bytes_of(&producer_push),
            groups,
        )
    }

    fn time_f32_layer_norm_forward_topology_ms(
        &self,
        batch: usize,
        probe: &LayerNormLowRankForwardProbeBuffers,
        norm: &LayerNormProducerInput<'_>,
        topology: LayerNormLowRankForwardF32Topology,
    ) -> Result<f64> {
        // Same shape as the FP16 probe: these kernels are only a few dozen
        // microseconds on compact fixtures, so repeat inside one submission rather
        // than let queue and fence latency dominate the comparison.
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
                self.record_f32_layer_norm_forward_topology_into(
                    commands,
                    batch,
                    &probe.previous,
                    &probe_norm,
                    &outputs,
                    topology,
                )?;
            }
            Ok(())
        })?;
        Ok(elapsed_ms / repetitions as f64)
    }

    fn choose_f32_layer_norm_forward_topology(
        &self,
        batch: usize,
        norm: &LayerNormProducerInput<'_>,
    ) -> Result<LayerNormLowRankForwardF32Decision> {
        let candidates = self.available_f32_layer_norm_forward_topologies();
        // The caller's own message for this shape is kept: this decision is only
        // asked for when a layer norm is being fused, and the dispatch site reports
        // an unavailable kernel with it.
        let structural_default = self.structural_f32_layer_norm_forward_topology().context(
            "RWKV LN1/low-rank producer fusion is unavailable on this device/rank shape",
        )?;
        if let Some(forced) = forced_layer_norm_forward_f32_topology() {
            if candidates.contains(&forced) {
                if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some()
                {
                    eprintln!(
                        "RWKV LN1/low-rank FP32 forward forced topology={} requested={} device={} batch={batch}",
                        forced.label(),
                        forced.env_value(),
                        self.device.decision_device_name()
                    );
                }
                return Ok(LayerNormLowRankForwardF32Decision {
                    topology: forced,
                    autotuned: false,
                });
            }
            eprintln!(
                "RWKV LN1/low-rank FP32 forward topology={} requested={} is unavailable on device={}; using {}",
                forced.label(),
                forced.env_value(),
                self.device.decision_device_name(),
                structural_default.label()
            );
        }
        if candidates.len() == 1
            || std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_DISABLE_LN_FORWARD_AUTOTUNE_ENV).is_some()
        {
            return Ok(LayerNormLowRankForwardF32Decision {
                topology: structural_default,
                autotuned: false,
            });
        }

        let subgroup_size = self.device.subgroup_capabilities().subgroup_size;
        let key = LayerNormLowRankForwardAutotuneKey {
            device_name: self.device.decision_device_name().to_owned(),
            subgroup_size,
            width: self.width,
            w_rank: self.w_rank,
            a_rank: self.a_rank,
            g_rank: self.g_rank,
            batch_pairs: batch.div_ceil(2),
            has_unpaired_tail: batch % 2 != 0,
        };
        let cache = LAYER_NORM_LOW_RANK_FORWARD_F32_AUTOTUNE_CACHE
            .get_or_init(|| Mutex::new(HashMap::new()));
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

        // Keep synchronous autotune submissions off the caller's pending
        // x/previous buffers, exactly as the FP16 probe does.
        let probe = self.allocate_fp16_layer_norm_forward_autotune_probe(batch)?;
        let time_topology =
            |topology| self.time_f32_layer_norm_forward_topology_ms(batch, &probe, norm, topology);
        if let Err(err) = time_topology(structural_default) {
            if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some() {
                eprintln!(
                    "RWKV LN1/low-rank FP32 forward autotune warmup failed device={} batch={batch}: {err:#}; using {}",
                    self.device.decision_device_name(),
                    structural_default.label()
                );
            }
            return Ok(LayerNormLowRankForwardF32Decision {
                topology: structural_default,
                autotuned: false,
            });
        }

        let rounds = match measure_layer_norm_low_rank_forward_rounds(&candidates, |topology| {
            time_topology(topology).with_context(|| {
                format!(
                    "timing RWKV LN1/low-rank FP32 forward candidate {}",
                    topology.label()
                )
            })
        }) {
            Ok(rounds) => rounds,
            Err(err) => {
                if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some()
                {
                    eprintln!(
                        "RWKV LN1/low-rank FP32 forward autotune failed device={} batch={batch}: {err:#}; using {}",
                        self.device.decision_device_name(),
                        structural_default.label()
                    );
                }
                return Ok(LayerNormLowRankForwardF32Decision {
                    topology: structural_default,
                    autotuned: false,
                });
            }
        };
        let required_margin = f32_topology_adoption_margin(
            self.device.decision_device_name(), self.width, (self.w_rank, self.a_rank, self.g_rank),
        );
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds_with_margin(
            &rounds, &candidates, structural_default, required_margin,
        );
        let selected = adjudication.selected;
        let decision = LayerNormLowRankForwardF32Decision {
            topology: selected,
            autotuned: true,
        };

        if std::env::var_os(HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG_ENV).is_some() {
            let summary = adjudication
                .medians
                .iter()
                .map(|(topology, ms)| format!("{}={ms:.5}ms", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            let wins = adjudication
                .round_wins
                .iter()
                .map(|(topology, wins)| format!("{}:{wins}", topology.label()))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "RWKV LN1/low-rank FP32 forward autotune device={} subgroup={} width={} ranks={}/{}/{} batch_pairs={} tail={} {} rounds={} round_wins={} selected={} contended={} autotuned={}",
                self.device.decision_device_name(),
                subgroup_size,
                self.width,
                self.w_rank,
                self.a_rank,
                self.g_rank,
                key.batch_pairs,
                key.has_unpaired_tail,
                summary,
                adjudication.rounds,
                wins,
                selected.label(),
                adjudication.contended,
                decision.autotuned
            );
        }
        let labels = adjudication
            .medians
            .iter()
            .map(|(topology, ms)| (topology.label().to_string(), *ms))
            .collect::<Vec<_>>();
        let wins = adjudication
            .round_wins
            .iter()
            .map(|(topology, wins)| (topology.label().to_string(), *wins))
            .collect::<Vec<_>>();
        emit_topology_decision(
            "ln-low-rank-f32",
            self.device.decision_device_name(),
            subgroup_size,
            &key,
            &LayerNormLowRankForwardTopologyEvidence {
                timings: &labels,
                round_wins: &wins,
                rounds: adjudication.rounds,
                decisive_margin: adjudication.decisive_margin.unwrap_or(0.0),
                required_margin,
                selected: selected.label(),
                structural_default: structural_default.label(),
                contention_spread: adjudication.contention_spread,
                contended: adjudication.contended,
            },
        );

        cache
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("RWKV LN1/low-rank forward autotune cache lock was poisoned")
            })?
            .insert(key, decision);
        Ok(decision)
    }

    /// Whether this geometry should dispatch the fused LN1/low-rank producer
    /// rather than the staged chain: the rank-2 decision of
    /// `validation/FUSED_FORWARD_PRODUCER_AUDIT.md`.
    ///
    /// The fused graph is the structural default, because the producer exists
    /// exactly where the staged chain's separate dispatches would otherwise run.
    /// A measured record entry can withdraw it per (adapter, width, batch), and
    /// the two A/B switches pin either graph for a measurement. The decision is
    /// consulted by the dense cell forward and by the x_norm-in producer route it
    /// calls; the TBPTT packed forward-only and packed-rematerialization paths keep
    /// their own fused chains, which the packed-cell composition A/B measures.
    pub fn prefers_fused_forward_chain(&self, batch: usize) -> bool {
        let measured = if self.fused_forward_forced || self.fused_forward_disabled {
            None
        } else {
            // The record is keyed by parameter-storage policy where a policy has
            // its own measurement: the deep FP16 arm's crossover sits at different
            // geometries from the FP32 route's, and an FP16-only win must not move
            // the FP32 default. A policy with no entry falls back to the shared
            // verdict.
            low_rank_full_forward_decisions().route_scoped(
                self.device.decision_device_name(),
                self.width as u32,
                batch as u32,
                self.parameter_storage_policy(),
            )
        };
        crate::decision_records::prefers_fused_forward(
            self.fused_forward_disabled,
            self.fused_forward_kernel_available(),
            measured,
        )
    }

    /// Whether the fused producer's kernels exist for this device, rank shape and
    /// storage policy. Both the deep FP16 arm and the portable first-stage
    /// producer count: either one replaces the chain the staged route would run.
    fn fused_forward_kernel_available(&self) -> bool {
        if self.fp16_parameter_storage_active() {
            self.low_rank_full_forward_fused_fp16_packed.is_some()
                || self.low_rank_producer_forward_fused_fp16_packed.is_some()
        } else {
            self.low_rank_producer_forward_fused.is_some()
        }
    }

    /// Whether the cell should fuse LN1 into that producer. The layer norm fuses
    /// exactly when the producer does: a staged route that kept the LN inside would
    /// not be the chain the record measured.
    pub(crate) fn prefers_layer_norm_forward_fusion(&self, batch: usize) -> bool {
        self.can_fuse_layer_norm_forward() && self.prefers_fused_forward_chain(batch)
    }

    /// The graph this geometry dispatches, in the shared A/B vocabulary.
    pub fn forward_chain_label(&self, batch: usize) -> &'static str {
        if self.prefers_fused_forward_chain(batch) {
            "fused"
        } else {
            "composed"
        }
    }

    /// The compiled-in route for this adapter, geometry and storage policy, or
    /// `None` where the record has no entry and the structural default decides.
    pub fn recorded_forward_chain_route(
        &self,
        batch: usize,
    ) -> Option<crate::decision_records::Route> {
        low_rank_full_forward_decisions().route_scoped(
            self.device.decision_device_name(),
            self.width as u32,
            batch as u32,
            self.parameter_storage_policy(),
        )
    }

    /// The recorded rank-2 route in the record's own vocabulary.
    pub fn recorded_forward_chain_route_label(&self, batch: usize) -> Option<&'static str> {
        self.recorded_forward_chain_route(batch)
            .map(crate::decision_records::Route::label)
    }

    /// The compiled-in decision for this adapter and geometry, or `None` where the
    /// record has no entry and the structural default decides. `Some(true)` means
    /// the record names the staged chain.
    pub fn recorded_forward_chain_decision(&self, batch: usize) -> Option<bool> {
        self.recorded_forward_chain_route(batch)
            .map(crate::decision_records::Route::is_staged)
    }

    /// The parameter-storage policy this op is running, as the decision records
    /// key it. `fp32` is the canonical dense policy; `fp16-storage` is the
    /// execution-storage policy the training graph installs under any
    /// `fp16-storage-*` precision policy. The record canonicalizes its older
    /// `fp16-packed-mirror` spelling into the same verdict.
    pub fn parameter_storage_policy(&self) -> &'static str {
        if self.fp16_parameter_storage_active() {
            "fp16-storage"
        } else {
            "fp32"
        }
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
        // The A/B's dispatch accounting: the fused arm is one producer dispatch,
        // and the staged arm is every kernel it replaces. Counted only when the
        // measurement switch is on.
        let before = commands.dispatch_count();
        let result = self.record_forward_inner_impl(commands, batch, x_norm, previous, layer_norm);
        if self.count_forward_dispatches {
            let dispatched = commands.dispatch_count().saturating_sub(before) as u64;
            if dispatched > 0 {
                self.forward_dispatches
                    .fetch_add(dispatched, std::sync::atomic::Ordering::Relaxed);
            }
        }
        result
    }

    /// Forward kernels this op dispatched since construction; see
    /// [`HIERARCHOS_RWKV_LOW_RANK_COUNT_FORWARD_DISPATCHES_ENV`].
    pub fn low_rank_forward_dispatches(&self) -> u64 {
        self.forward_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Dispatches of the deep FP16 full-forward producer (the x_norm-in arm that
    /// keeps one row's w/a/g rank activations resident). The route's total above
    /// counts it and the first-stage producer as one dispatch each, so a receipt
    /// that names the deep arm needs this count to prove it ran.
    pub fn deep_forward_arm_dispatches(&self) -> u64 {
        self.full_forward_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The fused producer the pre-normalized (`record_forward`) route dispatches
    /// for this geometry, or `None` where the route is the staged chain.
    ///
    /// This names the arm, it does not time it: the deep FP16 producer is the
    /// first branch when FP16 storage is active and its kernel exists, and the
    /// first-stage producer is the fallback. The layer-norm route names its own
    /// arms (`layer_norm_low_rank_producer_forward_fused*`), which is why this
    /// label is scoped to the x_norm-in graph rather than trying to describe
    /// both.
    pub fn x_norm_forward_arm_label(&self, batch: usize) -> Option<&'static str> {
        if !self.prefers_fused_forward_chain(batch) {
            return None;
        }
        if self.fp16_parameter_storage_active() {
            if self.low_rank_full_forward_fused_fp16_packed.is_some() {
                Some("low-rank-full-forward-fused-fp16-packed")
            } else if self.low_rank_producer_forward_fused_fp16_packed.is_some() {
                Some("low-rank-producer-forward-fused-fp16-packed")
            } else {
                None
            }
        } else if self.low_rank_producer_forward_fused.is_some() {
            Some("low-rank-producer-forward-fused")
        } else {
            None
        }
    }

    fn record_forward_inner_impl(
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
        // The rank-2 route decides both halves of the fusion: the deep FP16
        // producer and the portable first-stage producer are the fused arm, and a
        // staged route dispatches neither.
        let fused_forward_route = self.prefers_fused_forward_chain(batch);
        let full_forward_kernel = if fused_forward_route {
            self.low_rank_full_forward_fused_fp16_packed.as_ref()
        } else {
            None
        };
        let low_rank_fused_kernel = if !fused_forward_route {
            None
        } else if fp16_mirrors.is_some() {
            self.low_rank_producer_forward_fused_fp16_packed.as_ref()
        } else {
            self.low_rank_producer_forward_fused.as_ref()
        };
        let (layer_norm_fused_kernel, layer_norm_groups) = if let Some(mirrors) = fp16_mirrors {
            let topology = if let Some(norm) = layer_norm.as_ref() {
                self.choose_fp16_layer_norm_forward_topology(batch, norm, mirrors)?
                    .topology
            } else {
                self.structural_fp16_layer_norm_forward_topology()
                    .context("RWKV FP16 LN1/low-rank producer fusion is unavailable")?
            };
            (
                self.fp16_layer_norm_forward_kernel(topology),
                // Every FP16 topology groups rows inside one workgroup, so its grid
                // is the row count over that grouping and nothing else.
                [batch.div_ceil(topology.rows_per_workgroup()) as u32, 1, 1],
            )
        } else if let Some(norm) = layer_norm.as_ref() {
            let topology = self
                .choose_f32_layer_norm_forward_topology(batch, norm)?
                .topology;
            (
                self.f32_layer_norm_forward_kernel(topology),
                topology.groups(batch, self.width),
            )
        } else {
            // No layer norm to fuse: the plain producer branches below own this
            // dispatch, and the FP32 LN producer kernel may or may not exist.
            (None, [1, 1, 1])
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
                layer_norm_groups,
            )?;
        } else if let (Some(kernel), Some(mirrors)) = (full_forward_kernel, fp16_mirrors) {
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
            if self.count_forward_dispatches {
                self.full_forward_dispatches
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
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
        schedules.retain(|schedule| match self.fan_in_arithmetic {
            None => true,
            Some(RwkvFanInArithmetic::LegacyFused) => *schedule != RwkvLowRankFanInSchedule::Split,
            Some(RwkvFanInArithmetic::Materialized) => {
                *schedule == RwkvLowRankFanInSchedule::Split || self.materialized_fan_in_fusion
            }
        });
        schedules
    }

    pub(crate) fn backward_fan_in_arithmetic(&self) -> Option<RwkvFanInArithmetic> {
        self.fan_in_arithmetic
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
        if !self
            .available_backward_fan_in_schedules(true)
            .contains(&schedule)
        {
            return false;
        }
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
        let fan_in_schedule = fan_in_schedule.or_else(|| {
            accumulation.and_then(|_| {
                self.available_backward_fan_in_schedules(outer_grad_x_norm.is_some())
                    .last()
                    .copied()
            })
        });
        if let Some(schedule) = fan_in_schedule {
            if !self.backward_fan_in_geometry_available(schedule, fan_in_workgroup_size) {
                bail!(
                    "RWKV low-rank fan-in geometry {}@wg{} is unavailable on device {}",
                    schedule.label(),
                    fan_in_workgroup_size,
                    self.device.decision_device_name()
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

    pub fn decision_device_name(&self) -> &str {
        self.device.decision_device_name()
    }

    pub fn ranks(&self) -> (usize, usize, usize) {
        (self.w_rank, self.a_rank, self.g_rank)
    }

    pub fn width(&self) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ally_f32_topology_admission_rejects_weak_wins_and_preserves_other_profiles() {
        use LayerNormLowRankForwardF32Topology::{OneRow, ColumnSlices};
        let device = "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]";
        let margin = f32_topology_adoption_margin(device, 128, (32, 32, 64));
        assert!(margin > 0.14 && margin < 0.16);
        for (name, width, ranks) in [
            ("Intel(R) HD Graphics 520", 128, (32, 32, 64)),
            ("AMD Radeon Graphics [pci=1002:15bf;driver=00800185]", 128, (32, 32, 64)),
            (device, 64, (8, 8, 8)),
        ] {
            assert_eq!(f32_topology_adoption_margin(name, width, ranks), 0.02);
        }
        let weak = vec![vec![(OneRow, 1.0), (ColumnSlices, 0.95)]; 7];
        let candidates = [OneRow, ColumnSlices];
        assert_eq!(adjudicate_layer_norm_low_rank_forward_rounds_with_margin(
            &weak, &candidates, OneRow, margin).selected, OneRow);
        assert_eq!(adjudicate_layer_norm_low_rank_forward_rounds(
            &weak, &candidates, OneRow).selected, ColumnSlices);
        let strong = vec![vec![(OneRow, 1.0), (ColumnSlices, 0.8)]; 7];
        assert_eq!(adjudicate_layer_norm_low_rank_forward_rounds_with_margin(
            &strong, &candidates, OneRow, margin).selected, ColumnSlices);
    }

    #[test]
    fn fan_in_reference_profile_is_adapter_driver_and_geometry_scoped() {
        let device = "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]";
        assert_eq!(
            fan_in_reference_arithmetic(device, 128, (32, 32, 64), 32),
            Some(RwkvFanInArithmetic::Materialized)
        );
        for (width, ranks, batch) in [
            (128, (32, 32, 64), 2),
            (128, (32, 32, 64), 64),
            (256, (32, 32, 64), 32),
            (128, (64, 32, 64), 32),
        ] {
            assert_eq!(
                fan_in_reference_arithmetic(device, width, ranks, batch),
                Some(RwkvFanInArithmetic::LegacyFused)
            );
        }
        for other in [
            "Intel(R) HD Graphics 520",
            "AMD Radeon Graphics [pci=1002:15bf;driver=00800185]",
        ] {
            assert_eq!(
                fan_in_reference_arithmetic(other, 128, (32, 32, 64), 32),
                None
            );
        }
    }

    #[test]
    fn fan_in_fusion_preserves_materialized_fp32_rounding() -> Result<()> {
        let Ok(device) = VulkanDevice::new() else {
            return Ok(());
        };
        if !crate::vendor::is_amd_rdna3(&device) {
            return Ok(());
        }
        let split = vulkan::ComputeKernel::new(&device, TIME_MIX_BACKWARD_SPV, 13, 8)?;
        let add =
            vulkan::ComputeKernel::new(&device, include_bytes!("../shaders/vector_add.spv"), 3, 4)?;
        // External modules are accepted only by this diagnostic, so arithmetic
        // repairs can be checked before rebuilding the production library.
        let directory = std::env::var_os("HIERARCHOS_VALIDATION_FAN_IN_DIR");
        let load = |name: &str, fallback: &[u8]| -> Result<Vec<u8>> {
            Ok(match directory.as_ref() {
                Some(path) => std::fs::read(Path::new(path).join(name))?,
                None => fallback.to_vec(),
            })
        };
        let mut seed = 0x7a193d51u32;
        let mut random = |len: usize| {
            (0..len)
                .map(|index| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    match index % 257 {
                        0 => -0.0,
                        1 => 0.0,
                        _ => (seed >> 8) as f32 / 16777216.0 - 0.5,
                    }
                })
                .collect::<Vec<_>>()
        };
        let workgroups: &[usize] = if directory.is_some() {
            &[64]
        } else {
            &[32, 64, 128]
        };
        for &workgroup in workgroups {
            let suffix = if workgroup == 64 {
                String::new()
            } else {
                format!("_wg{workgroup}")
            };
            let base = vulkan::ComputeKernel::new(
                &device,
                &load(
                    &format!("rwkv_time_mix3_backward_fused_add{suffix}.spv"),
                    time_mix_fan_in_spirv(true, false, workgroup),
                )?,
                15,
                8,
            )?;
            let outer = vulkan::ComputeKernel::new(
                &device,
                &load(
                    &format!("rwkv_time_mix3_backward_fused_add_outer{suffix}.spv"),
                    time_mix_fan_in_spirv(true, true, workgroup),
                )?,
                16,
                8,
            )?;
            for (batch, width) in [(1usize, 32usize), (3, 65), (17, 128), (64, 128), (192, 128)] {
                let len = batch * width;
                let inputs = (0..11)
                    .map(|index| {
                        GpuBuffer::from_f32(
                            &device,
                            &random(if (2..5).contains(&index) { width } else { len }),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let outputs = (0..14)
                    .map(|_| GpuBuffer::zeros_f32(&device, len))
                    .collect::<Result<Vec<_>>>()?;
                let push = [batch as u32, width as u32];
                let element_push = [len as u32];
                let groups = [div_ceil_u32(width, workgroup), 1, 1];
                let element_groups = [div_ceil_u32(len, 256), 1, 1];
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                split.record_dispatch(
                    &mut commands,
                    &[
                        &inputs[0],
                        &inputs[1],
                        &inputs[2],
                        &inputs[3],
                        &inputs[4],
                        &inputs[5],
                        &inputs[6],
                        &inputs[7],
                        &outputs[0],
                        &outputs[1],
                        &outputs[2],
                        &outputs[3],
                        &outputs[4],
                    ],
                    bytemuck::cast_slice(&push),
                    [div_ceil_u32(width, 64), 1, 1],
                )?;
                add.record_dispatch(
                    &mut commands,
                    &[&inputs[8], &outputs[0], &outputs[5]],
                    bytemuck::cast_slice(&element_push),
                    element_groups,
                )?;
                add.record_dispatch(
                    &mut commands,
                    &[&inputs[9], &outputs[1], &outputs[6]],
                    bytemuck::cast_slice(&element_push),
                    element_groups,
                )?;
                add.record_dispatch(
                    &mut commands,
                    &[&outputs[5], &inputs[10], &outputs[7]],
                    bytemuck::cast_slice(&element_push),
                    element_groups,
                )?;
                base.record_dispatch(
                    &mut commands,
                    &[
                        &inputs[0],
                        &inputs[1],
                        &inputs[2],
                        &inputs[3],
                        &inputs[4],
                        &inputs[5],
                        &inputs[6],
                        &inputs[7],
                        &inputs[8],
                        &inputs[9],
                        &outputs[8],
                        &outputs[9],
                        &outputs[2],
                        &outputs[3],
                        &outputs[4],
                    ],
                    bytemuck::cast_slice(&push),
                    groups,
                )?;
                outer.record_dispatch(
                    &mut commands,
                    &[
                        &inputs[0],
                        &inputs[1],
                        &inputs[2],
                        &inputs[3],
                        &inputs[4],
                        &inputs[5],
                        &inputs[6],
                        &inputs[7],
                        &inputs[8],
                        &inputs[9],
                        &inputs[10],
                        &outputs[10],
                        &outputs[11],
                        &outputs[2],
                        &outputs[3],
                        &outputs[4],
                    ],
                    bytemuck::cast_slice(&push),
                    groups,
                )?;
                commands.submit()?;
                for (label, expected, actual) in [
                    ("base-x", 5, 8),
                    ("base-previous", 6, 9),
                    ("outer-x", 7, 10),
                    ("outer-previous", 6, 11),
                ] {
                    let expected = outputs[expected].read_f32(len)?;
                    let actual = outputs[actual].read_f32(len)?;
                    let different = expected
                        .iter()
                        .zip(&actual)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let first = expected
                        .iter()
                        .zip(&actual)
                        .enumerate()
                        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                        .take(8)
                        .map(|(index, (a, b))| (index, a.to_bits(), b.to_bits(), *a, *b))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        different, 0,
                    "{label} changed materialized adjoint bits at {batch}x{width} wg={workgroup}: {first:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn row_per_workgroup_topologies_keep_one_grid_dimension() {
        assert_eq!(
            LayerNormLowRankForwardF32Topology::OneRow.groups(3, 128),
            [3, 1, 1]
        );
        assert_eq!(
            LayerNormLowRankForwardF32Topology::TwoRows.groups(3, 128),
            [2, 1, 1]
        );
        assert_eq!(
            LayerNormLowRankForwardF32Topology::TwoRows.groups(4, 128),
            [2, 1, 1]
        );
    }

    #[test]
    fn column_slices_grid_carries_rows_in_the_second_dimension() {
        // The kernel reads the row from `gl_WorkGroupID.y` and its column slice from
        // `gl_WorkGroupID.x`, so a row belongs in the second grid dimension and the
        // first one has to cover every 64-column slice of the row.
        assert_eq!(
            LayerNormLowRankForwardF32Topology::ColumnSlices.groups(2, 128),
            [2, 2, 1]
        );
        for (batch, width) in [(1usize, 64usize), (2, 128), (9, 256), (32, 100)] {
            let groups = LayerNormLowRankForwardF32Topology::ColumnSlices.groups(batch, width);
            assert_eq!(groups[1], batch as u32);
            assert_eq!(groups[2], 1);
            assert!(groups[0] >= (width as u32).div_ceil(64));
        }
    }

    #[test]
    fn column_slices_is_never_dispatched_with_a_row_per_workgroup_grid() {
        // The production site used to reduce a topology to its rows per workgroup and
        // build `[batch / rows, 1, 1]` from it. For the sliced kernel that grid is a
        // wrong answer rather than a slow one -- the kernel reads the row from
        // `gl_WorkGroupID.y`, so every row but the first would go unwritten -- and the
        // same reduction stays correct for the topologies that do take their row from
        // the first grid dimension.
        let batch = 2usize;
        let width = 128usize;
        assert_eq!(
            LayerNormLowRankForwardF32Topology::ColumnSlices.rows_per_workgroup(),
            1
        );
        assert_ne!(
            LayerNormLowRankForwardF32Topology::ColumnSlices.groups(batch, width),
            [batch as u32, 1, 1]
        );
        assert_eq!(
            LayerNormLowRankForwardF32Topology::OneRow.groups(batch, width),
            [batch as u32, 1, 1]
        );
    }

    /// Every FP16 variant is addressed by one distinct override value and identified by
    /// one distinct record label: the equivalence harness credits an arm from the value
    /// the producer reports, and the decision record keys an entry by the label, so a
    /// duplicate in either table would silently fold two variants into one.
    #[test]
    fn every_fp16_variant_has_a_distinct_override_value_and_label() {
        let mut values = LayerNormLowRankForwardTopology::ALL
            .into_iter()
            .map(LayerNormLowRankForwardTopology::env_value)
            .collect::<Vec<_>>();
        let mut labels = LayerNormLowRankForwardTopology::ALL
            .into_iter()
            .map(LayerNormLowRankForwardTopology::label)
            .collect::<Vec<_>>();
        for list in [&mut values, &mut labels] {
            list.sort_unstable();
            list.dedup();
        }
        assert_eq!(values.len(), 6, "{values:?}");
        assert_eq!(labels.len(), 6, "{labels:?}");
        // The rows-per-workgroup each dispatch divides the batch by. Every FP16 kernel
        // takes its row from the first grid dimension -- the FP32 column-slice kernel is
        // the one that reads `gl_WorkGroupID.y` -- so this is the whole grid difference
        // between the six variants.
        for (topology, rows) in [
            (LayerNormLowRankForwardTopology::OneRow, 1usize),
            (LayerNormLowRankForwardTopology::OneRowCached, 1),
            (LayerNormLowRankForwardTopology::OneRowRank128, 1),
            (LayerNormLowRankForwardTopology::TwoRows, 2),
            (LayerNormLowRankForwardTopology::TwoRowsWide, 2),
            (LayerNormLowRankForwardTopology::FourRows, 4),
        ] {
            assert_eq!(topology.rows_per_workgroup(), rows);
        }
    }

    /// The two families share one override variable, so the parser's vocabulary is the
    /// union of their tables. A value that only one family offers is normal and is
    /// answered by that family's own rejection path; only a value in neither table is a
    /// typo worth a message.
    #[test]
    fn the_override_vocabulary_is_the_union_of_both_families() {
        let known = known_forced_topology_values();
        for value in LayerNormLowRankForwardTopology::ALL
            .into_iter()
            .map(LayerNormLowRankForwardTopology::env_value)
            .chain(
                LayerNormLowRankForwardF32Topology::ALL
                    .into_iter()
                    .map(LayerNormLowRankForwardF32Topology::env_value),
            )
        {
            assert!(known.contains(&value), "{value} is missing from {known:?}");
        }
        // `one-row` and `two-rows` are the shared names -- each family offers one variant
        // under them -- so the union is 7 values rather than 6 + 3.
        assert_eq!(known.len(), 7, "{known:?}");
        assert!(known.contains(&"column-slices"), "{known:?}");
        assert!(known.contains(&"one-row-cached"), "{known:?}");
    }

    fn rounds(
        samples: &[&[(LayerNormLowRankForwardTopology, f64)]],
    ) -> Vec<Vec<(LayerNormLowRankForwardTopology, f64)>> {
        samples.iter().map(|round| round.to_vec()).collect()
    }

    /// A variant that beats the default by the material margin in every round is selected,
    /// and the record says how narrow its worst round was.
    #[test]
    fn a_unanimous_win_displaces_the_structural_default() {
        use LayerNormLowRankForwardTopology::{OneRow, OneRowCached};
        let measured = rounds(&[
            &[(OneRow, 1.00), (OneRowCached, 0.90)],
            &[(OneRow, 0.99), (OneRowCached, 0.88)],
            &[(OneRow, 1.01), (OneRowCached, 0.91)],
        ]);
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds(
            &measured,
            &[OneRow, OneRowCached],
            OneRow,
        );
        assert_eq!(adjudication.selected, OneRowCached);
        assert_eq!(adjudication.rounds, 3);
        assert_eq!(
            adjudication.round_wins,
            vec![(OneRow, 0), (OneRowCached, 3)]
        );
        // The narrowest lead is the third round's: 1.01 - 0.91 over 1.01.
        let decisive = adjudication
            .decisive_margin
            .expect("a unanimous winner has a margin");
        assert!((decisive - 0.09900990099).abs() < 1e-6, "{decisive}");
    }

    /// The rule this replaced selected on the median of three rounds alone. Here the
    /// candidate leads the medians by 10% yet fails the material margin in one round, so
    /// the old rule would have dispatched it and the unanimous rule keeps the default.
    #[test]
    fn a_lead_that_loses_a_round_keeps_the_default() {
        use LayerNormLowRankForwardTopology::{OneRow, OneRowCached};
        let measured = rounds(&[
            &[(OneRow, 1.000), (OneRowCached, 0.900)],
            &[(OneRow, 1.000), (OneRowCached, 0.990)],
            &[(OneRow, 1.000), (OneRowCached, 0.900)],
        ]);
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds(
            &measured,
            &[OneRow, OneRowCached],
            OneRow,
        );
        assert_eq!(adjudication.selected, OneRow);
        assert_eq!(
            adjudication.round_wins,
            vec![(OneRow, 0), (OneRowCached, 2)]
        );
        assert!(adjudication.decisive_margin.is_none());
        // The medians are still reported, because that is what the record holds, and this
        // is the pair the record shows beside a kept default.
        assert_eq!(
            adjudication.medians,
            vec![(OneRow, 1.000), (OneRowCached, 0.900)]
        );
    }

    /// Two variants that each win a round leave the default in place: the rounds have to
    /// agree, so a lead that depends on which variant happened to be measured is not a
    /// decision.
    #[test]
    fn split_rounds_leave_the_default() {
        use LayerNormLowRankForwardTopology::{OneRow, OneRowCached, TwoRows};
        let measured = rounds(&[
            &[(OneRow, 1.0), (OneRowCached, 0.90), (TwoRows, 0.95)],
            &[(OneRow, 1.0), (OneRowCached, 0.96), (TwoRows, 0.90)],
            &[(OneRow, 1.0), (OneRowCached, 0.95), (TwoRows, 0.97)],
        ]);
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds(
            &measured,
            &[OneRow, OneRowCached, TwoRows],
            OneRow,
        );
        assert_eq!(adjudication.selected, OneRow);
        assert_eq!(
            adjudication.round_wins,
            vec![(OneRow, 0), (OneRowCached, 2), (TwoRows, 1)]
        );
    }

    /// Ties go to the earlier candidate, so the verdict does not depend on the order the
    /// interleaved rounds happened to measure in, and a missing round withholds a win
    /// rather than inventing one.
    #[test]
    fn the_verdict_is_order_independent_and_withholds_unmeasured_rounds() {
        use LayerNormLowRankForwardTopology::{OneRow, OneRowCached, TwoRows};
        let measured = rounds(&[
            &[(TwoRows, 0.90), (OneRowCached, 0.90), (OneRow, 1.0)],
            &[(OneRowCached, 0.90), (OneRow, 1.0), (TwoRows, 0.90)],
        ]);
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds(
            &measured,
            &[OneRow, OneRowCached, TwoRows],
            OneRow,
        );
        assert_eq!(adjudication.selected, OneRowCached);
        assert_eq!(
            adjudication.round_wins,
            vec![(OneRow, 0), (OneRowCached, 2), (TwoRows, 0)]
        );
        // `TwoRows` was never measured, so it cannot win a round it is absent from.
        let incomplete = rounds(&[&[(OneRow, 1.0), (OneRowCached, 0.90)], &[(OneRow, 1.0)]]);
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds(
            &incomplete,
            &[OneRow, OneRowCached],
            OneRow,
        );
        assert_eq!(adjudication.selected, OneRow);
    }

    /// No rounds at all is not a win: the default stands.
    #[test]
    fn no_measured_rounds_keeps_the_default() {
        use LayerNormLowRankForwardTopology::{OneRow, OneRowCached};
        let adjudication =
            adjudicate_layer_norm_low_rank_forward_rounds(&[], &[OneRow, OneRowCached], OneRow);
        assert_eq!(adjudication.selected, OneRow);
        assert_eq!(adjudication.rounds, 0);
        assert!(adjudication.decisive_margin.is_none());
        assert!(
            !adjudication.contended,
            "nothing measured is not contention"
        );
    }

    /// The rounds have to have measured one machine. Here the candidate wins every round by
    /// the material margin, yet the default's own samples move by 25% across the sweep --
    /// another process was sharing the GPU -- so the sweep is not evidence about the kernels
    /// and the structural default stands with the contention recorded.
    #[test]
    fn a_contended_sweep_keeps_the_default_and_says_so() {
        use LayerNormLowRankForwardTopology::{OneRow, OneRowCached};
        let measured = rounds(&[
            &[(OneRow, 1.00), (OneRowCached, 0.90)],
            &[(OneRow, 1.20), (OneRowCached, 0.95)],
            &[(OneRow, 1.25), (OneRowCached, 0.96)],
        ]);
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds(
            &measured,
            &[OneRow, OneRowCached],
            OneRow,
        );
        assert_eq!(
            adjudication.selected, OneRow,
            "a sweep taken under load must not move the decision"
        );
        assert!(adjudication.contended);
        assert!(
            (adjudication.contention_spread - 1.25).abs() < 1e-9,
            "{}",
            adjudication.contention_spread
        );
        assert!(adjudication.decisive_margin.is_none());
        // The wins are still reported: they are what the record shows beside the kept
        // default, and they are how a reader sees that the load, not the kernels, decided.
        assert_eq!(
            adjudication.round_wins,
            vec![(OneRow, 0), (OneRowCached, 3)]
        );
    }

    /// A sweep whose default samples stay inside the threshold is adjudicated normally, and
    /// the spread it measured is what the record carries, so a run that was quiet enough to
    /// decide says how quiet it was.
    #[test]
    fn a_quiet_sweep_still_decides_and_records_its_spread() {
        use LayerNormLowRankForwardTopology::{OneRow, OneRowCached};
        let measured = rounds(&[
            &[(OneRow, 1.00), (OneRowCached, 0.90)],
            &[(OneRow, 1.04), (OneRowCached, 0.92)],
            &[(OneRow, 1.02), (OneRowCached, 0.91)],
        ]);
        let adjudication = adjudicate_layer_norm_low_rank_forward_rounds(
            &measured,
            &[OneRow, OneRowCached],
            OneRow,
        );
        assert_eq!(adjudication.selected, OneRowCached);
        assert!(!adjudication.contended);
        assert!(
            (adjudication.contention_spread - 1.04).abs() < 1e-9,
            "{}",
            adjudication.contention_spread
        );
    }
}
