use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const HIERARCHOS_VULKAN_LM_EXECUTION_ARM_ENV: &str = "HIERARCHOS_VULKAN_LM_EXECUTION_ARM";
pub const HIERARCHOS_VULKAN_LM_BACKWARD_TOPOLOGY_ENV: &str =
    "HIERARCHOS_VULKAN_LM_BACKWARD_TOPOLOGY";
pub const HIERARCHOS_VULKAN_LM_FUSED_ADJOINT_TOPOLOGY_ENV: &str =
    "HIERARCHOS_VULKAN_LM_FUSED_ADJOINT_TOPOLOGY";
pub const HIERARCHOS_VULKAN_LM_AUTOTUNE_DISABLE_ENV: &str = "HIERARCHOS_VULKAN_LM_AUTOTUNE_DISABLE";
pub const HIERARCHOS_VULKAN_LM_REAUTOTUNE_ENV: &str = "HIERARCHOS_VULKAN_LM_REAUTOTUNE";
pub const HIERARCHOS_VULKAN_LM_AUTOTUNE_LOG_ENV: &str = "HIERARCHOS_VULKAN_LM_AUTOTUNE_LOG";
pub const HIERARCHOS_VULKAN_LM_AUTOTUNE_CACHE_PATH_ENV: &str =
    "HIERARCHOS_VULKAN_LM_AUTOTUNE_CACHE_PATH";
pub const HIERARCHOS_VULKAN_LM_DISABLE_PERSISTENT_CACHE_ENV: &str =
    "HIERARCHOS_VULKAN_LM_DISABLE_PERSISTENT_CACHE";

const PERSISTENT_CACHE_VERSION: u32 = 11;

const LM_NATIVE_FP16_VOCAB_TILE_ROWS: u32 = 64;
const LM_NATIVE_FP16_FIXED_SHARED_BYTES: u32 = (64 + 1024) * 4;
const LM_NATIVE_FP16_REUSE_ARMS: [HierarchosLmExecutionArm; 4] = [
    HierarchosLmExecutionArm::Fp16Native,
    HierarchosLmExecutionArm::Fp16NativeReuse64,
    HierarchosLmExecutionArm::Fp16NativeReuse128,
    HierarchosLmExecutionArm::Fp16NativeReuse224,
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HierarchosLmExecutionArm {
    Fp32,
    Fp16Packed,
    Fp16CeTape,
    Fp16CeTapeRows8,
    Fp16CeTapeRows16,
    Fp16CeTapeRows16Dot4,
    Fp16CeTapeRows16FusedAdjoints,
    Fp16CeTapeRows16Dot4FusedAdjoints,
    Fp16CeTapeRows16Cluster4FusedAdjoints,
    Fp16Native,
    Fp16NativeReuse64,
    Fp16NativeReuse128,
    Fp16NativeReuse224,
}

impl HierarchosLmExecutionArm {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16Packed => "fp16-packed",
            Self::Fp16CeTape => "fp16-ce-tape",
            Self::Fp16CeTapeRows8 => "fp16-ce-tape-rows8",
            Self::Fp16CeTapeRows16 => "fp16-ce-tape-rows16",
            Self::Fp16CeTapeRows16Dot4 => "fp16-ce-tape-rows16-dot4",
            Self::Fp16CeTapeRows16FusedAdjoints => "fp16-ce-tape-rows16-fused-adjoints",
            Self::Fp16CeTapeRows16Dot4FusedAdjoints => "fp16-ce-tape-rows16-dot4-fused-adjoints",
            Self::Fp16CeTapeRows16Cluster4FusedAdjoints => {
                "fp16-ce-tape-rows16-cluster4-fused-adjoints"
            }
            Self::Fp16Native => "fp16-native",
            Self::Fp16NativeReuse64 => "fp16-native-reuse64",
            Self::Fp16NativeReuse128 => "fp16-native-reuse128",
            Self::Fp16NativeReuse224 => "fp16-native-reuse224",
        }
    }

    pub const fn uses_fp16_weights(self) -> bool {
        !matches!(self, Self::Fp32)
    }

    pub const fn native_fp16_reuse_pairs(self) -> Option<u32> {
        match self {
            Self::Fp16Native => Some(32),
            Self::Fp16NativeReuse64 => Some(64),
            Self::Fp16NativeReuse128 => Some(128),
            Self::Fp16NativeReuse224 => Some(224),
            Self::Fp32
            | Self::Fp16Packed
            | Self::Fp16CeTape
            | Self::Fp16CeTapeRows8
            | Self::Fp16CeTapeRows16
            | Self::Fp16CeTapeRows16Dot4
            | Self::Fp16CeTapeRows16FusedAdjoints
            | Self::Fp16CeTapeRows16Dot4FusedAdjoints
            | Self::Fp16CeTapeRows16Cluster4FusedAdjoints => None,
        }
    }

    pub const fn fuses_ce_adjoints(self) -> bool {
        matches!(
            self,
            Self::Fp16CeTapeRows16FusedAdjoints
                | Self::Fp16CeTapeRows16Dot4FusedAdjoints
                | Self::Fp16CeTapeRows16Cluster4FusedAdjoints
        )
    }

    pub const fn native_fp16_shared_memory_bytes(self) -> Option<u32> {
        match self.native_fp16_reuse_pairs() {
            Some(pairs) => {
                Some(LM_NATIVE_FP16_FIXED_SHARED_BYTES + LM_NATIVE_FP16_VOCAB_TILE_ROWS * pairs * 4)
            }
            None => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HierarchosLmWeightGradTopology {
    VocabRows4,
    VocabRows8,
    VocabRows16,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum HierarchosLmFusedAdjointTopology {
    SharedHidden,
    PrivateHidden,
    PrivateHiddenTile256,
}

impl HierarchosLmFusedAdjointTopology {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SharedHidden => "fused-shared-hidden",
            Self::PrivateHidden => "fused-private-hidden",
            Self::PrivateHiddenTile256 => "fused-private-hidden-tile256-wg256",
        }
    }

    pub const fn vocab_tile(self) -> usize {
        match self {
            Self::SharedHidden | Self::PrivateHidden => 64,
            Self::PrivateHiddenTile256 => 256,
        }
    }
}

impl HierarchosLmWeightGradTopology {
    pub const fn label(self) -> &'static str {
        match self {
            Self::VocabRows4 => "dw-vocab4",
            Self::VocabRows8 => "dw-vocab8",
            Self::VocabRows16 => "dw-vocab16",
        }
    }

    pub const fn vocab_rows_per_group(self) -> u32 {
        match self {
            Self::VocabRows4 => 4,
            Self::VocabRows8 => 8,
            Self::VocabRows16 => 16,
        }
    }

    pub const fn local_size(self) -> [u32; 3] {
        [32, self.vocab_rows_per_group(), 1]
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub(crate) struct HierarchosLmBackwardPlan {
    pub input_grad_arm: HierarchosLmExecutionArm,
    pub weight_grad_topology: HierarchosLmWeightGradTopology,
    pub fused_adjoint_topology: HierarchosLmFusedAdjointTopology,
}

impl HierarchosLmBackwardPlan {
    fn label(self) -> String {
        if self.input_grad_arm.fuses_ce_adjoints() {
            format!(
                "{}+{}+{}",
                self.input_grad_arm.label(),
                self.weight_grad_topology.label(),
                self.fused_adjoint_topology.label()
            )
        } else {
            format!(
                "{}+{}",
                self.input_grad_arm.label(),
                self.weight_grad_topology.label()
            )
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct LmExecutionAutotuneKey {
    device_name: String,
    subgroup_size: u32,
    context_dim: usize,
    vocab_size: usize,
    rows: usize,
    native_fp16_candidate: bool,
    max_compute_shared_memory_bytes: u32,
    ce_tape_candidate: bool,
    ce_tape_rows8_candidate: bool,
    ce_tape_rows16_candidate: bool,
    ce_tape_rows16_fused_adjoints_candidate: bool,
    fused_adjoints_private_hidden_candidate: bool,
    fused_adjoints_private_hidden_tile256_candidate: bool,
    ce_tape_rows16_cluster4_candidate: bool,
    dw_vocab4_candidate: bool,
    dw_vocab8_candidate: bool,
    dw_vocab16_candidate: bool,
    kernel_signature: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistentEntry {
    key: LmExecutionAutotuneKey,
    plan: HierarchosLmBackwardPlan,
}

#[derive(Debug, Deserialize, Serialize)]
struct PersistentCache {
    version: u32,
    entries: Vec<PersistentEntry>,
}

impl Default for PersistentCache {
    fn default() -> Self {
        Self {
            version: PERSISTENT_CACHE_VERSION,
            entries: Vec::new(),
        }
    }
}

static AUTOTUNE_CACHE: OnceLock<Mutex<HashMap<LmExecutionAutotuneKey, HierarchosLmBackwardPlan>>> =
    OnceLock::new();
static PERSISTENT_CACHE_IO: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) struct LmExecutionAutotuneGeometry<'a> {
    pub device_name: &'a str,
    pub subgroup_size: u32,
    pub context_dim: usize,
    pub vocab_size: usize,
    pub rows: usize,
    pub native_fp16_candidate: bool,
    pub max_compute_shared_memory_bytes: u32,
    pub ce_tape_candidate: bool,
    pub ce_tape_rows8_candidate: bool,
    pub ce_tape_rows16_candidate: bool,
    pub ce_tape_rows16_fused_adjoints_candidate: bool,
    pub fused_adjoints_private_hidden_candidate: bool,
    pub fused_adjoints_private_hidden_tile256_candidate: bool,
    pub ce_tape_rows16_cluster4_candidate: bool,
    pub dw_vocab4_candidate: bool,
    pub dw_vocab8_candidate: bool,
    pub dw_vocab16_candidate: bool,
    pub kernel_signature: u64,
}

pub(crate) fn choose_fp16_backward_plan<F>(
    geometry: LmExecutionAutotuneGeometry<'_>,
    mut measure_ms: F,
) -> Result<HierarchosLmBackwardPlan>
where
    F: FnMut(HierarchosLmBackwardPlan) -> Result<f64>,
{
    let mut input_candidates = fp16_execution_candidates(&geometry);
    if let Some(forced) = forced_fp16_execution_arm()? {
        if !input_candidates.contains(&forced) {
            bail!(
                "forced LM execution arm {} is unavailable on device {}",
                forced.label(),
                geometry.device_name
            );
        }
        input_candidates.retain(|candidate| *candidate == forced);
    }

    let mut weight_grad_candidates = fp16_weight_grad_candidates(&geometry);
    if weight_grad_candidates.is_empty() {
        bail!(
            "device {} cannot launch the portable 32x4 LM dW workgroup",
            geometry.device_name
        );
    }
    if let Some(forced) = forced_fp16_weight_grad_topology()? {
        if !weight_grad_candidates.contains(&forced) {
            bail!(
                "forced LM backward topology {} is unavailable on device {}",
                forced.label(),
                geometry.device_name
            );
        }
        weight_grad_candidates.retain(|candidate| *candidate == forced);
    }

    let default_input = if input_candidates.contains(&HierarchosLmExecutionArm::Fp16Packed) {
        HierarchosLmExecutionArm::Fp16Packed
    } else {
        input_candidates[0]
    };
    let default_weight_grad =
        if weight_grad_candidates.contains(&HierarchosLmWeightGradTopology::VocabRows8) {
            HierarchosLmWeightGradTopology::VocabRows8
        } else {
            weight_grad_candidates[0]
        };
    let mut fused_topology_candidates = fp16_fused_adjoint_candidates(&geometry);
    let mut selected_fused_topology = HierarchosLmFusedAdjointTopology::SharedHidden;
    if let Some(forced) = forced_fp16_fused_adjoint_topology()? {
        if !fused_topology_candidates.contains(&forced) {
            bail!(
                "forced LM fused-adjoint topology {} is unavailable on device {}",
                forced.label(),
                geometry.device_name
            );
        }
        selected_fused_topology = forced;
        fused_topology_candidates.retain(|candidate| *candidate == forced);
    }
    let default_plan = HierarchosLmBackwardPlan {
        input_grad_arm: default_input,
        weight_grad_topology: default_weight_grad,
        fused_adjoint_topology: if default_input.fuses_ce_adjoints() {
            selected_fused_topology
        } else {
            HierarchosLmFusedAdjointTopology::SharedHidden
        },
    };

    if std::env::var_os(HIERARCHOS_VULKAN_LM_AUTOTUNE_DISABLE_ENV).is_some()
        || (input_candidates.len() == 1
            && weight_grad_candidates.len() == 1
            && (!input_candidates[0].fuses_ce_adjoints() || fused_topology_candidates.len() == 1))
    {
        return Ok(default_plan);
    }

    let key = LmExecutionAutotuneKey {
        device_name: geometry.device_name.to_owned(),
        subgroup_size: geometry.subgroup_size,
        context_dim: geometry.context_dim,
        vocab_size: geometry.vocab_size,
        rows: geometry.rows,
        native_fp16_candidate: geometry.native_fp16_candidate,
        max_compute_shared_memory_bytes: geometry.max_compute_shared_memory_bytes,
        ce_tape_candidate: geometry.ce_tape_candidate,
        ce_tape_rows8_candidate: geometry.ce_tape_rows8_candidate,
        ce_tape_rows16_candidate: geometry.ce_tape_rows16_candidate,
        ce_tape_rows16_fused_adjoints_candidate: geometry.ce_tape_rows16_fused_adjoints_candidate,
        fused_adjoints_private_hidden_candidate: geometry.fused_adjoints_private_hidden_candidate,
        fused_adjoints_private_hidden_tile256_candidate: geometry
            .fused_adjoints_private_hidden_tile256_candidate,
        ce_tape_rows16_cluster4_candidate: geometry.ce_tape_rows16_cluster4_candidate,
        dw_vocab4_candidate: geometry.dw_vocab4_candidate,
        dw_vocab8_candidate: geometry.dw_vocab8_candidate,
        dw_vocab16_candidate: geometry.dw_vocab16_candidate,
        kernel_signature: geometry.kernel_signature,
    };
    let cache = AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let reautotune = std::env::var_os(HIERARCHOS_VULKAN_LM_REAUTOTUNE_ENV).is_some();
    if !reautotune {
        if let Some(plan) = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("LM execution autotune cache lock was poisoned"))?
            .get(&key)
            .copied()
            .filter(|plan| {
                input_candidates.contains(&plan.input_grad_arm)
                    && weight_grad_candidates.contains(&plan.weight_grad_topology)
                    && (!plan.input_grad_arm.fuses_ce_adjoints()
                        || fused_topology_candidates.contains(&plan.fused_adjoint_topology))
            })
        {
            return Ok(plan);
        }
    }

    if !reautotune {
        match load_persistent_plan(
            &key,
            &input_candidates,
            &weight_grad_candidates,
            &fused_topology_candidates,
        ) {
            Ok(Some(plan)) => {
                cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("LM execution autotune cache lock was poisoned"))?
                    .insert(key, plan);
                if autotune_log_enabled() {
                    eprintln!(
                        "Hierarchos LM autotune persistent-hit device={} subgroup={} shared={}B context_dim={} vocab={} rows={} selected={}",
                        geometry.device_name,
                        geometry.subgroup_size,
                        geometry.max_compute_shared_memory_bytes,
                        geometry.context_dim,
                        geometry.vocab_size,
                        geometry.rows,
                        plan.label()
                    );
                }
                return Ok(plan);
            }
            Ok(None) => {}
            Err(err) => {
                if autotune_log_enabled() {
                    eprintln!("Hierarchos LM persistent autotune cache read failed: {err:#}");
                }
            }
        }
    }

    // Warm one stable anchor before the coordinate descent. Every measured plan
    // consumes the same packed-FP16 execution mirror and retains FP32 CE/dW
    // arithmetic, so topology selection cannot change the PyTorch oracle.
    for _ in 0..2 {
        let _ = measure_ms(default_plan)?;
    }

    // Choose the fused-adjoint hidden-value placement once before comparing
    // projection arms. All fused projection arms feed the same adjoint kernel,
    // so this isolates the shared-memory-vs-register choice without multiplying
    // the whole projection candidate space.
    let fused_probe_arm = input_candidates
        .iter()
        .copied()
        .find(|arm| arm.fuses_ce_adjoints());
    let fused_topology_timings = if let Some(probe_arm) = fused_probe_arm {
        if forced_fp16_fused_adjoint_topology()?.is_some() {
            Vec::new()
        } else {
            let timings = profile_medians(&fused_topology_candidates, |fused_adjoint_topology| {
                measure_ms(HierarchosLmBackwardPlan {
                    input_grad_arm: probe_arm,
                    weight_grad_topology: default_weight_grad,
                    fused_adjoint_topology,
                })
            })?;
            selected_fused_topology =
                select_candidate(&timings, HierarchosLmFusedAdjointTopology::SharedHidden);
            timings
        }
    } else {
        Vec::new()
    };

    // First choose the W^T/input-adjoint topology while holding dW stable,
    // then choose the dW fanout while holding that winner stable. This avoids a
    // 5x3 Cartesian explosion but the thing persisted and installed is the
    // complete backward plan, including combinations such as reuse224+dw16.
    let input_timings = profile_medians(&input_candidates, |input_grad_arm| {
        measure_ms(HierarchosLmBackwardPlan {
            input_grad_arm,
            weight_grad_topology: default_weight_grad,
            fused_adjoint_topology: if input_grad_arm.fuses_ce_adjoints() {
                selected_fused_topology
            } else {
                HierarchosLmFusedAdjointTopology::SharedHidden
            },
        })
    })?;
    let selected_input = fastest_candidate(&input_timings, default_input);
    // The fused rows16 adjoint arm produces dW inside the same bounded replay
    // kernel as W^T, so the standalone dW topology is intentionally inert.
    // Do not benchmark identical fused plans three times and let timing noise
    // persist a meaningless dW fanout choice.
    let (weight_grad_timings, selected_weight_grad) = if selected_input.fuses_ce_adjoints() {
        (Vec::new(), default_weight_grad)
    } else {
        let timings = profile_medians(&weight_grad_candidates, |weight_grad_topology| {
            measure_ms(HierarchosLmBackwardPlan {
                input_grad_arm: selected_input,
                weight_grad_topology,
                fused_adjoint_topology: HierarchosLmFusedAdjointTopology::SharedHidden,
            })
        })?;
        let selected = fastest_candidate(&timings, default_weight_grad);
        (timings, selected)
    };
    let coordinate_winner = HierarchosLmBackwardPlan {
        input_grad_arm: selected_input,
        weight_grad_topology: selected_weight_grad,
        fused_adjoint_topology: if selected_input.fuses_ce_adjoints() {
            selected_fused_topology
        } else {
            HierarchosLmFusedAdjointTopology::SharedHidden
        },
    };
    let final_timings = if coordinate_winner == default_plan {
        Vec::new()
    } else {
        profile_medians(&[default_plan, coordinate_winner], |plan| measure_ms(plan))?
    };
    let selected = if final_timings.is_empty() {
        default_plan
    } else {
        // Apply the persistence/noise margin only to the complete backward
        // plans. Two individually sub-2% improvements are allowed to combine
        // into one durable win instead of being discarded at each axis.
        select_candidate(&final_timings, default_plan)
    };

    if autotune_log_enabled() {
        let input_summary = input_timings
            .iter()
            .map(|(arm, ms)| format!("{}={ms:.4}ms", arm.label()))
            .collect::<Vec<_>>()
            .join(" ");
        let weight_grad_summary = weight_grad_timings
            .iter()
            .map(|(topology, ms)| format!("{}={ms:.4}ms", topology.label()))
            .collect::<Vec<_>>()
            .join(" ");
        let fused_summary = fused_topology_timings
            .iter()
            .map(|(topology, ms)| format!("{}={ms:.4}ms", topology.label()))
            .collect::<Vec<_>>()
            .join(" ");
        let final_summary = final_timings
            .iter()
            .map(|(plan, ms)| format!("{}={ms:.4}ms", plan.label()))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "Hierarchos LM autotune device={} subgroup={} shared={}B context_dim={} vocab={} rows={} fused=[{}] input=[{}] dW=[{}] final=[{}] selected={}",
            geometry.device_name,
            geometry.subgroup_size,
            geometry.max_compute_shared_memory_bytes,
            geometry.context_dim,
            geometry.vocab_size,
            geometry.rows,
            fused_summary,
            input_summary,
            weight_grad_summary,
            final_summary,
            selected.label()
        );
    }

    cache
        .lock()
        .map_err(|_| anyhow::anyhow!("LM execution autotune cache lock was poisoned"))?
        .insert(key.clone(), selected);
    if let Err(err) = store_persistent_plan(&key, selected) {
        if autotune_log_enabled() {
            eprintln!("Hierarchos LM persistent autotune cache write failed: {err:#}");
        }
    }
    Ok(selected)
}

fn profile_medians<T, F>(candidates: &[T], mut measure_ms: F) -> Result<Vec<(T, f64)>>
where
    T: Copy,
    F: FnMut(T) -> Result<f64>,
{
    let mut samples = vec![Vec::with_capacity(5); candidates.len()];
    for round in 0..5 {
        if round % 2 == 0 {
            for (index, &candidate) in candidates.iter().enumerate() {
                samples[index].push(measure_ms(candidate)?);
            }
        } else {
            for index in (0..candidates.len()).rev() {
                samples[index].push(measure_ms(candidates[index])?);
            }
        }
    }
    let mut timings = Vec::with_capacity(candidates.len());
    for (&candidate, values) in candidates.iter().zip(&mut samples) {
        values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
        timings.push((candidate, values[values.len() / 2]));
    }
    Ok(timings)
}

fn fp16_execution_candidates(
    geometry: &LmExecutionAutotuneGeometry<'_>,
) -> Vec<HierarchosLmExecutionArm> {
    let mut candidates = vec![HierarchosLmExecutionArm::Fp16Packed];
    if geometry.ce_tape_candidate {
        candidates.push(HierarchosLmExecutionArm::Fp16CeTape);
    }
    if geometry.ce_tape_rows8_candidate {
        candidates.push(HierarchosLmExecutionArm::Fp16CeTapeRows8);
    }
    if geometry.ce_tape_rows16_candidate {
        candidates.push(HierarchosLmExecutionArm::Fp16CeTapeRows16);
        if geometry.max_compute_shared_memory_bytes >= 16_384 {
            candidates.push(HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4);
        }
    }
    if geometry.ce_tape_rows16_fused_adjoints_candidate {
        candidates.push(HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints);
        if geometry.ce_tape_rows16_candidate && geometry.max_compute_shared_memory_bytes >= 16_384 {
            candidates.push(HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints);
        }
        if geometry.ce_tape_rows16_cluster4_candidate {
            candidates.push(HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints);
        }
    }
    if geometry.native_fp16_candidate {
        let required_pairs = geometry.context_dim.div_ceil(2) as u32;
        for arm in LM_NATIVE_FP16_REUSE_ARMS {
            let reuse_pairs = arm
                .native_fp16_reuse_pairs()
                .expect("native FP16 reuse arm must declare a reuse width");
            let shared_bytes = arm
                .native_fp16_shared_memory_bytes()
                .expect("native FP16 reuse arm must declare shared-memory use");
            if shared_bytes <= geometry.max_compute_shared_memory_bytes {
                candidates.push(arm);
            }
            if reuse_pairs >= required_pairs {
                break;
            }
        }
    }
    candidates
}

fn fp16_weight_grad_candidates(
    geometry: &LmExecutionAutotuneGeometry<'_>,
) -> Vec<HierarchosLmWeightGradTopology> {
    let mut candidates = Vec::with_capacity(3);
    if geometry.dw_vocab4_candidate {
        candidates.push(HierarchosLmWeightGradTopology::VocabRows4);
    }
    if geometry.dw_vocab8_candidate {
        candidates.push(HierarchosLmWeightGradTopology::VocabRows8);
    }
    if geometry.dw_vocab16_candidate {
        candidates.push(HierarchosLmWeightGradTopology::VocabRows16);
    }
    candidates
}

fn fp16_fused_adjoint_candidates(
    geometry: &LmExecutionAutotuneGeometry<'_>,
) -> Vec<HierarchosLmFusedAdjointTopology> {
    let mut candidates = vec![HierarchosLmFusedAdjointTopology::SharedHidden];
    if geometry.fused_adjoints_private_hidden_candidate {
        candidates.push(HierarchosLmFusedAdjointTopology::PrivateHidden);
    }
    if geometry.fused_adjoints_private_hidden_tile256_candidate {
        candidates.push(HierarchosLmFusedAdjointTopology::PrivateHiddenTile256);
    }
    candidates
}

fn forced_fp16_execution_arm() -> Result<Option<HierarchosLmExecutionArm>> {
    let raw = match std::env::var(HIERARCHOS_VULKAN_LM_EXECUTION_ARM_ENV) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(err) => bail!("reading {HIERARCHOS_VULKAN_LM_EXECUTION_ARM_ENV}: {err}"),
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(None),
        "packed" | "fp16-packed" | "portable-packed" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16Packed))
        }
        "tape" | "ce-tape" | "fp16-ce-tape" => Ok(Some(HierarchosLmExecutionArm::Fp16CeTape)),
        "tape-rows8" | "ce-tape-rows8" | "fp16-ce-tape-rows8" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16CeTapeRows8))
        }
        "tape-rows16" | "ce-tape-rows16" | "fp16-ce-tape-rows16" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16CeTapeRows16))
        }
        "tape-rows16-dot4" | "ce-tape-rows16-dot4" | "fp16-ce-tape-rows16-dot4" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4))
        }
        "tape-rows16-fused"
        | "ce-tape-rows16-fused"
        | "fp16-ce-tape-rows16-fused"
        | "fp16-ce-tape-rows16-fused-adjoints" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints))
        }
        "tape-rows16-dot4-fused"
        | "ce-tape-rows16-dot4-fused"
        | "fp16-ce-tape-rows16-dot4-fused"
        | "fp16-ce-tape-rows16-dot4-fused-adjoints" => Ok(Some(
            HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints,
        )),
        "tape-rows16-cluster4-fused"
        | "ce-tape-rows16-cluster4-fused"
        | "fp16-ce-tape-rows16-cluster4-fused"
        | "fp16-ce-tape-rows16-cluster4-fused-adjoints" => Ok(Some(
            HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints,
        )),
        "native" | "fp16-native" | "native-half" | "native32" | "fp16-native-reuse32" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16Native))
        }
        "native64" | "fp16-native-reuse64" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16NativeReuse64))
        }
        "native128" | "fp16-native-reuse128" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16NativeReuse128))
        }
        "native224" | "fp16-native-reuse224" => {
            Ok(Some(HierarchosLmExecutionArm::Fp16NativeReuse224))
        }
        "fp32" => bail!(
            "{HIERARCHOS_VULKAN_LM_EXECUTION_ARM_ENV}=fp32 crosses the training numerical policy; select HIERARCHOS_VULKAN_TRAINING_PRECISION=fp32 instead"
        ),
        _ => bail!(
            "{HIERARCHOS_VULKAN_LM_EXECUTION_ARM_ENV} must be auto, fp16-packed, fp16-ce-tape, fp16-ce-tape-rows8, fp16-ce-tape-rows16, fp16-ce-tape-rows16-dot4, fp16-ce-tape-rows16-fused-adjoints, fp16-ce-tape-rows16-dot4-fused-adjoints, fp16-ce-tape-rows16-cluster4-fused-adjoints, fp16-native[-reuse32], fp16-native-reuse64, fp16-native-reuse128, or fp16-native-reuse224; got {raw:?}"
        ),
    }
}

fn forced_fp16_weight_grad_topology() -> Result<Option<HierarchosLmWeightGradTopology>> {
    let raw = match std::env::var(HIERARCHOS_VULKAN_LM_BACKWARD_TOPOLOGY_ENV) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(err) => bail!("reading {HIERARCHOS_VULKAN_LM_BACKWARD_TOPOLOGY_ENV}: {err}"),
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(None),
        "4" | "dw4" | "dw-vocab4" | "vocab4" => {
            Ok(Some(HierarchosLmWeightGradTopology::VocabRows4))
        }
        "8" | "dw8" | "dw-vocab8" | "vocab8" => {
            Ok(Some(HierarchosLmWeightGradTopology::VocabRows8))
        }
        "16" | "dw16" | "dw-vocab16" | "vocab16" => {
            Ok(Some(HierarchosLmWeightGradTopology::VocabRows16))
        }
        _ => bail!(
            "{HIERARCHOS_VULKAN_LM_BACKWARD_TOPOLOGY_ENV} must be auto, dw-vocab4, dw-vocab8, or dw-vocab16; got {raw:?}"
        ),
    }
}

fn forced_fp16_fused_adjoint_topology() -> Result<Option<HierarchosLmFusedAdjointTopology>> {
    let raw = match std::env::var(HIERARCHOS_VULKAN_LM_FUSED_ADJOINT_TOPOLOGY_ENV) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(err) => bail!("reading {HIERARCHOS_VULKAN_LM_FUSED_ADJOINT_TOPOLOGY_ENV}: {err}"),
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(None),
        "shared" | "shared-hidden" | "fused-shared-hidden" => {
            Ok(Some(HierarchosLmFusedAdjointTopology::SharedHidden))
        }
        "private" | "private-hidden" | "fused-private-hidden" => {
            Ok(Some(HierarchosLmFusedAdjointTopology::PrivateHidden))
        }
        "private-tile256"
        | "private-hidden-tile256"
        | "fused-private-hidden-tile256"
        | "fused-private-hidden-tile256-wg256" => {
            Ok(Some(HierarchosLmFusedAdjointTopology::PrivateHiddenTile256))
        }
        _ => bail!(
            "{HIERARCHOS_VULKAN_LM_FUSED_ADJOINT_TOPOLOGY_ENV} must be auto, shared-hidden, private-hidden, or private-hidden-tile256; got {raw:?}"
        ),
    }
}

fn select_candidate<T>(timings: &[(T, f64)], default: T) -> T
where
    T: Copy + PartialEq,
{
    let Some((_, default_ms)) = timings.iter().find(|(candidate, _)| *candidate == default) else {
        return default;
    };
    let mut selected = default;
    let mut selected_ms = *default_ms;
    for &(candidate, ms) in timings {
        // Require a real margin before replacing the portable baseline. This
        // prevents one-time DVFS/submission noise from becoming a persistent
        // device decision.
        if ms < selected_ms * 0.98 {
            selected = candidate;
            selected_ms = ms;
        }
    }
    selected
}

fn fastest_candidate<T>(timings: &[(T, f64)], default: T) -> T
where
    T: Copy,
{
    timings
        .iter()
        .min_by(|(_, lhs_ms), (_, rhs_ms)| lhs_ms.total_cmp(rhs_ms))
        .map(|(candidate, _)| *candidate)
        .unwrap_or(default)
}

fn autotune_log_enabled() -> bool {
    std::env::var_os(HIERARCHOS_VULKAN_LM_AUTOTUNE_LOG_ENV).is_some()
}

fn persistent_cache_path() -> Option<PathBuf> {
    if std::env::var_os(HIERARCHOS_VULKAN_LM_DISABLE_PERSISTENT_CACHE_ENV).is_some() {
        return None;
    }
    if let Some(path) = std::env::var_os(HIERARCHOS_VULKAN_LM_AUTOTUNE_CACHE_PATH_ENV) {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
        return None;
    }
    if let Some(root) = std::env::var_os("LOCALAPPDATA") {
        return Some(
            PathBuf::from(root)
                .join("Hierarchos")
                .join("vulkan-lm-execution-v5.json"),
        );
    }
    if let Some(root) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(
            PathBuf::from(root)
                .join("hierarchos")
                .join("vulkan-lm-execution-v5.json"),
        );
    }
    std::env::var_os("HOME").map(|root| {
        PathBuf::from(root)
            .join(".cache")
            .join("hierarchos")
            .join("vulkan-lm-execution-v5.json")
    })
}

fn load_persistent_plan(
    key: &LmExecutionAutotuneKey,
    input_candidates: &[HierarchosLmExecutionArm],
    weight_grad_candidates: &[HierarchosLmWeightGradTopology],
    fused_topology_candidates: &[HierarchosLmFusedAdjointTopology],
) -> Result<Option<HierarchosLmBackwardPlan>> {
    let Some(path) = persistent_cache_path() else {
        return Ok(None);
    };
    let _guard = PERSISTENT_CACHE_IO
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| anyhow::anyhow!("LM execution persistent cache lock was poisoned"))?;
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("reading {}", path.display())),
    };
    let cache: PersistentCache =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if cache.version != PERSISTENT_CACHE_VERSION {
        return Ok(None);
    }
    Ok(cache
        .entries
        .iter()
        .rev()
        .find(|entry| {
            &entry.key == key
                && input_candidates.contains(&entry.plan.input_grad_arm)
                && weight_grad_candidates.contains(&entry.plan.weight_grad_topology)
                && (!entry.plan.input_grad_arm.fuses_ce_adjoints()
                    || fused_topology_candidates.contains(&entry.plan.fused_adjoint_topology))
        })
        .map(|entry| entry.plan))
}

fn store_persistent_plan(
    key: &LmExecutionAutotuneKey,
    plan: HierarchosLmBackwardPlan,
) -> Result<()> {
    let Some(path) = persistent_cache_path() else {
        return Ok(());
    };
    let _guard = PERSISTENT_CACHE_IO
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| anyhow::anyhow!("LM execution persistent cache lock was poisoned"))?;
    let mut cache = match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<PersistentCache>(&bytes)
            .ok()
            .filter(|cache| cache.version == PERSISTENT_CACHE_VERSION)
            .unwrap_or_default(),
        Err(_) => PersistentCache::default(),
    };
    if let Some(entry) = cache.entries.iter_mut().find(|entry| entry.key == *key) {
        entry.plan = plan;
    } else {
        cache.entries.push(PersistentEntry {
            key: key.clone(),
            plan,
        });
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "creating LM persistent autotune cache directory {}",
                parent.display()
            )
        })?;
    }
    let bytes =
        serde_json::to_vec_pretty(&cache).context("serializing LM persistent autotune cache")?;
    fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        fp16_execution_candidates, fp16_weight_grad_candidates, select_candidate,
        HierarchosLmBackwardPlan, HierarchosLmExecutionArm, HierarchosLmFusedAdjointTopology,
        HierarchosLmWeightGradTopology, LmExecutionAutotuneGeometry, LM_NATIVE_FP16_REUSE_ARMS,
    };

    #[test]
    fn lm_autotune_requires_two_percent_margin_over_portable_baseline() {
        let default = HierarchosLmExecutionArm::Fp16Packed;
        assert_eq!(
            select_candidate(
                &[
                    (HierarchosLmExecutionArm::Fp16Packed, 10.0),
                    (HierarchosLmExecutionArm::Fp16Native, 9.85),
                ],
                default,
            ),
            default
        );
        assert_eq!(
            select_candidate(
                &[
                    (HierarchosLmExecutionArm::Fp16Packed, 10.0),
                    (HierarchosLmExecutionArm::Fp16Native, 9.7),
                ],
                default,
            ),
            HierarchosLmExecutionArm::Fp16Native
        );
    }

    #[test]
    fn lm_autotune_applies_margin_to_the_combined_backward_plan() {
        let portable = HierarchosLmBackwardPlan {
            input_grad_arm: HierarchosLmExecutionArm::Fp16Packed,
            weight_grad_topology: HierarchosLmWeightGradTopology::VocabRows8,
            fused_adjoint_topology: HierarchosLmFusedAdjointTopology::SharedHidden,
        };
        let combined = HierarchosLmBackwardPlan {
            input_grad_arm: HierarchosLmExecutionArm::Fp16NativeReuse64,
            weight_grad_topology: HierarchosLmWeightGradTopology::VocabRows16,
            fused_adjoint_topology: HierarchosLmFusedAdjointTopology::SharedHidden,
        };

        // The input-axis win alone is 1.9%, so an axis-local 2% gate would
        // retain packed. The measured complete plan clears 2% once dW16 is
        // combined with reuse64 and must therefore be eligible to persist.
        assert_eq!(
            select_candidate(
                &[
                    (HierarchosLmExecutionArm::Fp16Packed, 66.0914),
                    (HierarchosLmExecutionArm::Fp16NativeReuse64, 64.8329),
                ],
                HierarchosLmExecutionArm::Fp16Packed,
            ),
            HierarchosLmExecutionArm::Fp16Packed
        );
        assert_eq!(
            select_candidate(&[(portable, 66.0914), (combined, 64.6450)], portable),
            combined
        );
    }

    #[test]
    fn native_fp16_reuse_arms_match_shared_memory_envelope() {
        let expected = [(32, 12_544), (64, 20_736), (128, 37_120), (224, 61_696)];
        for (arm, (pairs, bytes)) in LM_NATIVE_FP16_REUSE_ARMS.into_iter().zip(expected) {
            assert_eq!(arm.native_fp16_reuse_pairs(), Some(pairs));
            assert_eq!(arm.native_fp16_shared_memory_bytes(), Some(bytes));
        }
    }

    #[test]
    fn native_fp16_reuse_candidates_stop_after_covering_context_width() {
        let geometry = LmExecutionAutotuneGeometry {
            device_name: "test",
            subgroup_size: 32,
            context_dim: 192,
            vocab_size: 50_257,
            rows: 2,
            native_fp16_candidate: true,
            max_compute_shared_memory_bytes: 65_536,
            ce_tape_candidate: true,
            ce_tape_rows8_candidate: true,
            ce_tape_rows16_candidate: true,
            ce_tape_rows16_fused_adjoints_candidate: true,
            fused_adjoints_private_hidden_candidate: true,
            fused_adjoints_private_hidden_tile256_candidate: true,
            ce_tape_rows16_cluster4_candidate: true,
            dw_vocab4_candidate: true,
            dw_vocab8_candidate: true,
            dw_vocab16_candidate: true,
            kernel_signature: 7,
        };
        assert_eq!(
            fp16_execution_candidates(&geometry),
            vec![
                HierarchosLmExecutionArm::Fp16Packed,
                HierarchosLmExecutionArm::Fp16CeTape,
                HierarchosLmExecutionArm::Fp16CeTapeRows8,
                HierarchosLmExecutionArm::Fp16CeTapeRows16,
                HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4,
                HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints,
                HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints,
                HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints,
                HierarchosLmExecutionArm::Fp16Native,
                HierarchosLmExecutionArm::Fp16NativeReuse64,
                HierarchosLmExecutionArm::Fp16NativeReuse128,
            ]
        );
    }

    #[test]
    fn native_fp16_reuse_candidates_respect_device_shared_memory() {
        let portable = LmExecutionAutotuneGeometry {
            device_name: "portable",
            subgroup_size: 32,
            context_dim: 448,
            vocab_size: 50_257,
            rows: 2,
            native_fp16_candidate: true,
            max_compute_shared_memory_bytes: 16_384,
            ce_tape_candidate: true,
            ce_tape_rows8_candidate: true,
            ce_tape_rows16_candidate: false,
            ce_tape_rows16_fused_adjoints_candidate: false,
            fused_adjoints_private_hidden_candidate: false,
            fused_adjoints_private_hidden_tile256_candidate: false,
            ce_tape_rows16_cluster4_candidate: false,
            dw_vocab4_candidate: true,
            dw_vocab8_candidate: true,
            dw_vocab16_candidate: false,
            kernel_signature: 11,
        };
        assert_eq!(
            fp16_execution_candidates(&portable),
            vec![
                HierarchosLmExecutionArm::Fp16Packed,
                HierarchosLmExecutionArm::Fp16CeTape,
                HierarchosLmExecutionArm::Fp16CeTapeRows8,
                HierarchosLmExecutionArm::Fp16Native,
            ]
        );

        let wide = LmExecutionAutotuneGeometry {
            max_compute_shared_memory_bytes: 65_536,
            ce_tape_rows16_candidate: true,
            ce_tape_rows16_fused_adjoints_candidate: true,
            ce_tape_rows16_cluster4_candidate: true,
            device_name: "wide",
            ..portable
        };
        assert_eq!(
            fp16_execution_candidates(&wide),
            vec![
                HierarchosLmExecutionArm::Fp16Packed,
                HierarchosLmExecutionArm::Fp16CeTape,
                HierarchosLmExecutionArm::Fp16CeTapeRows8,
                HierarchosLmExecutionArm::Fp16CeTapeRows16,
                HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4,
                HierarchosLmExecutionArm::Fp16CeTapeRows16FusedAdjoints,
                HierarchosLmExecutionArm::Fp16CeTapeRows16Dot4FusedAdjoints,
                HierarchosLmExecutionArm::Fp16CeTapeRows16Cluster4FusedAdjoints,
                HierarchosLmExecutionArm::Fp16Native,
                HierarchosLmExecutionArm::Fp16NativeReuse64,
                HierarchosLmExecutionArm::Fp16NativeReuse128,
                HierarchosLmExecutionArm::Fp16NativeReuse224,
            ]
        );
    }

    #[test]
    fn lm_weight_grad_candidates_follow_workgroup_capabilities() {
        let geometry = LmExecutionAutotuneGeometry {
            device_name: "portable",
            subgroup_size: 32,
            context_dim: 448,
            vocab_size: 50_257,
            rows: 2,
            native_fp16_candidate: true,
            max_compute_shared_memory_bytes: 32_768,
            ce_tape_candidate: true,
            ce_tape_rows8_candidate: true,
            ce_tape_rows16_candidate: true,
            ce_tape_rows16_fused_adjoints_candidate: false,
            fused_adjoints_private_hidden_candidate: false,
            fused_adjoints_private_hidden_tile256_candidate: false,
            ce_tape_rows16_cluster4_candidate: false,
            dw_vocab4_candidate: true,
            dw_vocab8_candidate: true,
            dw_vocab16_candidate: false,
            kernel_signature: 19,
        };
        assert_eq!(
            fp16_weight_grad_candidates(&geometry),
            vec![
                HierarchosLmWeightGradTopology::VocabRows4,
                HierarchosLmWeightGradTopology::VocabRows8,
            ]
        );

        let wide = LmExecutionAutotuneGeometry {
            dw_vocab16_candidate: true,
            ..geometry
        };
        assert_eq!(
            fp16_weight_grad_candidates(&wide),
            vec![
                HierarchosLmWeightGradTopology::VocabRows4,
                HierarchosLmWeightGradTopology::VocabRows8,
                HierarchosLmWeightGradTopology::VocabRows16,
            ]
        );
    }
}
