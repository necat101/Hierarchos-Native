use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const HIERARCHOS_TOKEN_TAPE_PROFILE_ENV: &str = "HIERARCHOS_VULKAN_TAPE_PROFILE_DB";
pub const HIERARCHOS_TOKEN_TAPE_PROFILE_DISABLE_ENV: &str =
    "HIERARCHOS_VULKAN_DISABLE_TAPE_PROFILES";
pub const HIERARCHOS_TOKEN_TAPE_PROFILE_LOG_ENV: &str = "HIERARCHOS_VULKAN_TAPE_PROFILE_LOG";
pub const HIERARCHOS_TOKEN_TAPE_ONLINE_AUTOTUNE_DISABLE_ENV: &str =
    "HIERARCHOS_VULKAN_DISABLE_TAPE_ONLINE_AUTOTUNE";
pub const HIERARCHOS_TOKEN_TAPE_EXPLORE_EVERY_ENV: &str = "HIERARCHOS_VULKAN_TAPE_EXPLORE_EVERY";
pub const HIERARCHOS_TOKEN_TAPE_PROFILE_FILENAME: &str =
    "vulkan_training_submission_profiles.v1.jsonl";

const PROFILE_SCHEMA_VERSION: u32 = 1;
pub const BACKWARD_KERNEL_GEOMETRY_POLICY_REVISION: u32 = 2;
pub(crate) const DEFAULT_TOKEN_TAPE_EXPLORE_EVERY: u64 = 16;
const PROFILE_CONFIDENCE_Z: f64 = 1.645;
const PROFILE_RELATIVE_NOISE_FLOOR: f64 = 0.10;
// Online observations arrive only on the bounded autotune cadence. Discounting
// by matching-geometry observation count (rather than wall clock) makes the
// policy deterministic across restarts while still letting fresh thermal/DVFS
// behavior overtake old benchmark evidence. 0.90 is a ~6.6-observation
// half-life, or roughly 100 ordinary optimizer steps at the default cadence.
const PROFILE_OBSERVATION_DECAY: f64 = 0.90;
const PROFILE_UCB_EXPLORATION_SCALE: f64 = 0.10;
const PROFILE_MIN_EFFECTIVE_ITERATIONS: f64 = 0.25;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HierarchosTokenTapeProfileGeometry {
    pub device: String,
    /// Exact Vulkan hardware/driver identity for newly collected evidence.
    /// Persistent records predating these fields remain valid as name-based
    /// fallback, but fingerprinted records never cross device/driver stacks.
    pub device_uuid: String,
    pub driver_uuid: String,
    pub subgroup_size: u32,
    /// Trainable-parameter execution storage/compute contract. Precision is
    /// geometry-level rather than a within-graph arm because changing it can
    /// require allocating different persistent mirrors and pipelines. Keeping
    /// it in the key prevents FP16 evidence from contaminating FP32 rankings.
    pub training_precision_policy: String,
    pub backward_kernel_geometry_revision: u32,
    pub device_local_pressure_bucket: Option<u8>,
    pub architecture_revision: String,
    pub batch: usize,
    pub context_dim: usize,
    pub persistent_dim: usize,
    pub ltm_slots: usize,
    pub ltm_key_dim: usize,
    pub ltm_val_dim: usize,
    pub ltm_topk: usize,
    pub vocab_size: usize,
    pub h_hidden: usize,
    pub l_hidden: usize,
    pub h_width: usize,
    pub l_width: usize,
    pub h_state_size: usize,
    pub l_state_size: usize,
    pub h_rwkv_head_size: usize,
    pub l_rwkv_head_size: usize,
    pub h_low_rank_ranks: Option<(usize, usize, usize)>,
    pub l_low_rank_ranks: Option<(usize, usize, usize)>,
    pub token_adapter_rank: usize,
    pub max_h_steps: usize,
    pub max_l_steps: usize,
    pub tokens_per_sequence: usize,
    pub sequences: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosTokenTapeProfileScore {
    pub sequence_microbatch_size: usize,
    pub state_checkpoint_stride: usize,
    pub device_local_pressure_bucket: Option<u8>,
    /// Optional H/L full-cell backward topology labels. `None` denotes a
    /// legacy geometry-only profile and is treated as topology-agnostic by the
    /// runtime for backwards compatibility with existing profile databases.
    pub h_backward_segment_schedule: Option<String>,
    pub l_backward_segment_schedule: Option<String>,
    /// Optional compiled local-size labels for the H/L fused RWKV backward
    /// kernels. Missing fields preserve compatibility with profile databases
    /// written before kernel geometry became a policy arm.
    pub h_backward_kernel_geometry: Option<String>,
    pub l_backward_kernel_geometry: Option<String>,
    /// RWKV reduction-order contract. Missing values in older profile files are
    /// interpreted as `strict-parity`, which was the only policy at the time.
    pub rwkv_numerics_policy: String,
    pub median_tokens_per_second: f64,
    /// Recency-weighted throughput estimate. Unlike the raw all-history median,
    /// this can move quickly when new online observations show a thermal/DVFS
    /// regime change.
    pub adaptive_tokens_per_second: f64,
    /// Conservative throughput estimate used for scheduler ranking. This is a
    /// lower-confidence bound around the adaptive estimate, so a noisy one-off
    /// win does not immediately displace a repeatable candidate.
    pub confidence_adjusted_tokens_per_second: f64,
    /// Upper-confidence score used only on bounded online-autotune steps. A
    /// stale candidate's effective evidence decays, increasing this score until
    /// it earns another real training-step probe.
    pub exploration_score_tokens_per_second: f64,
    pub relative_uncertainty: f64,
    pub effective_measured_iterations: f64,
    pub observations_since_last_measurement: usize,
    pub profile_records: usize,
    pub measured_iterations: usize,
}

/// Marginal evidence for one recurrent branch's backward topology. H and L
/// deliberately use the same shape so persistent observations can be projected
/// onto either branch without coupling the sibling branch or tape geometry.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct HierarchosTokenTapeBranchTopology {
    pub backward_segment_schedule: Option<String>,
    pub backward_kernel_geometry: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HierarchosTokenTapeBranchTopologyScore {
    pub topology: HierarchosTokenTapeBranchTopology,
    pub median_tokens_per_second: f64,
    pub adaptive_tokens_per_second: f64,
    pub confidence_adjusted_tokens_per_second: f64,
    pub exploration_score_tokens_per_second: f64,
    pub relative_uncertainty: f64,
    pub effective_measured_iterations: f64,
    pub observations_since_last_measurement: usize,
    pub profile_records: usize,
    pub measured_iterations: usize,
}

/// Marginal evidence for one independently controlled recurrent-branch factor.
/// Keeping schedule and kernel geometry separate lets the runtime synthesize a
/// schedule/geometry pair that was never present as an exact profile arm while
/// retaining the same recency/confidence/UCB policy as the composite history.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HierarchosTokenTapeBranchFactorScore {
    pub value: Option<String>,
    pub median_tokens_per_second: f64,
    pub adaptive_tokens_per_second: f64,
    pub confidence_adjusted_tokens_per_second: f64,
    pub exploration_score_tokens_per_second: f64,
    pub relative_uncertainty: f64,
    pub effective_measured_iterations: f64,
    pub observations_since_last_measurement: usize,
    pub profile_records: usize,
    pub measured_iterations: usize,
}

/// Marginal evidence for one tape-memory geometry coordinate. Microbatch width
/// and checkpoint stride are intentionally scored independently so persistent
/// observations can recommend a pair that was never benchmarked as one exact
/// Cartesian arm. Numerics remains fixed while these coordinates are projected
/// because changing reduction order is part of the parity contract.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HierarchosTokenTapeGeometryFactorScore {
    pub value: usize,
    pub median_tokens_per_second: f64,
    pub adaptive_tokens_per_second: f64,
    pub confidence_adjusted_tokens_per_second: f64,
    pub exploration_score_tokens_per_second: f64,
    pub relative_uncertainty: f64,
    pub effective_measured_iterations: f64,
    pub observations_since_last_measurement: usize,
    pub profile_records: usize,
    pub measured_iterations: usize,
}

/// Optional steady-state diagnostics attached to an online exploration record.
/// The scheduler still consumes the stable median-throughput / iteration pair;
/// these fields preserve the evidence needed to distinguish a repeatable win
/// from warmup, queue pressure, memory pressure, or a thermally degrading GPU.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct HierarchosTokenTapeOnlineTelemetry {
    pub warmup_windows: usize,
    pub measured_windows: usize,
    pub aggregate_tokens_per_second: f64,
    pub mean_tokens_per_second: f64,
    pub sample_variance_tokens_per_second: f64,
    pub relative_standard_deviation: f64,
    pub first_tokens_per_second: f64,
    pub last_tokens_per_second: f64,
    pub timeline_retirement_latency_samples: u64,
    pub timeline_retirement_latency_ns_average: Option<f64>,
    pub kernel_timestamp_profile_samples: u64,
    pub kernel_timestamp_dispatches: u64,
    pub kernel_gpu_ns_per_token: Option<f64>,
    pub first_kernel_gpu_ns_per_token: Option<f64>,
    pub last_kernel_gpu_ns_per_token: Option<f64>,
    pub initial_device_local_pressure_bucket: Option<u8>,
    pub final_device_local_pressure_bucket: Option<u8>,
    pub peak_device_local_usage_ratio: Option<f64>,
    pub throughput_slowdown_ratio: f64,
    pub possible_gpu_throttling: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BranchFactorKind {
    SegmentSchedule,
    SegmentState,
    SegmentProjection,
    SegmentLowRankFanIn,
    KernelGeometry,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TapeGeometryFactorKind {
    SequenceMicrobatch,
    StateCheckpointStride,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BackwardSegmentScheduleFactors {
    state: String,
    projection: String,
    low_rank_fan_in: Option<String>,
}

const BACKWARD_SEGMENT_STATE_LABELS: [&str; 3] = [
    "rkv-add3+key+reduce",
    "rkv-add3-key+reduce",
    "rkv-add3-key-reduce",
];
const BACKWARD_SEGMENT_PROJECTION_LABELS: [&str; 4] = [
    "weight->fused-input-mix",
    "fused-input-mix->weight",
    "weight->split-input-mix",
    "split-input-mix->weight",
];
const BACKWARD_SEGMENT_LOW_RANK_FAN_IN_LABELS: [&str; 3] = [
    "low-rank-split-fan-in",
    "low-rank-fused-base-fan-in",
    "low-rank-fused-outer-fan-in",
];

/// Decode the stable string representation emitted by `BackwardSegmentSchedule`
/// without making the persistent profile format depend on the private runtime
/// enum types. Older/noncanonical labels are deliberately left opaque so they
/// continue to participate in whole-schedule ranking without contaminating the
/// newer marginal state/projection/fan-in populations.
fn parse_backward_segment_schedule_factors(label: &str) -> Option<BackwardSegmentScheduleFactors> {
    let (without_fan_in, low_rank_fan_in) = if let Some(fan_in) =
        BACKWARD_SEGMENT_LOW_RANK_FAN_IN_LABELS
            .iter()
            .find(|fan_in| label.ends_with(&format!("+{fan_in}")))
    {
        let suffix_len = fan_in.len() + 1;
        (
            &label[..label.len() - suffix_len],
            Some((*fan_in).to_owned()),
        )
    } else {
        (label, None)
    };
    let projection = BACKWARD_SEGMENT_PROJECTION_LABELS
        .iter()
        .find(|projection| without_fan_in.ends_with(&format!("+{projection}")))?;
    let projection_suffix_len = projection.len() + 1;
    let state = &without_fan_in[..without_fan_in.len() - projection_suffix_len];
    if !BACKWARD_SEGMENT_STATE_LABELS.contains(&state) {
        return None;
    }
    Some(BackwardSegmentScheduleFactors {
        state: state.to_owned(),
        projection: (*projection).to_owned(),
        low_rank_fan_in,
    })
}

#[derive(Clone, Debug)]
pub struct HierarchosTokenTapeProfileDatabase {
    source_path: PathBuf,
    observations: Vec<StoredProfileObservation>,
}

impl HierarchosTokenTapeProfileDatabase {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading Vulkan tape profile database {}", path.display()))?;
        let observations = parse_profile_jsonl(&text)
            .with_context(|| format!("parsing Vulkan tape profile database {}", path.display()))?;
        Ok(Self {
            source_path: path.to_path_buf(),
            observations,
        })
    }

    pub fn load_default() -> Result<Option<Self>> {
        if std::env::var_os(HIERARCHOS_TOKEN_TAPE_PROFILE_DISABLE_ENV).is_some() {
            return Ok(None);
        }
        if let Some(path) = std::env::var_os(HIERARCHOS_TOKEN_TAPE_PROFILE_ENV) {
            return Self::load(PathBuf::from(path)).map(Some);
        }

        let candidates = default_profile_candidates()?;
        for path in candidates {
            if path.is_file() {
                return Self::load(path).map(Some);
            }
        }
        Ok(None)
    }

    /// Open the configured/default profile database for online autotuning. In
    /// contrast to `load_default`, this may return an empty database backed by
    /// a not-yet-created JSONL path so the first online exploration result can
    /// bootstrap persistent scheduler knowledge.
    pub(crate) fn open_default_for_online() -> Result<Option<Self>> {
        if std::env::var_os(HIERARCHOS_TOKEN_TAPE_PROFILE_DISABLE_ENV).is_some()
            || std::env::var_os(HIERARCHOS_TOKEN_TAPE_ONLINE_AUTOTUNE_DISABLE_ENV).is_some()
        {
            return Ok(None);
        }
        if let Some(path) = std::env::var_os(HIERARCHOS_TOKEN_TAPE_PROFILE_ENV) {
            let path = PathBuf::from(path);
            return if path.is_file() {
                Self::load(path).map(Some)
            } else {
                Ok(Some(Self {
                    source_path: path,
                    observations: Vec::new(),
                }))
            };
        }

        let candidates = default_profile_candidates()?;
        if let Some(path) = candidates.iter().find(|path| path.is_file()) {
            return Self::load(path).map(Some);
        }
        let path = if candidates[0].parent().is_some_and(|parent| parent.is_dir())
            || !candidates[1].parent().is_some_and(|parent| parent.is_dir())
        {
            candidates[0].clone()
        } else {
            candidates[1].clone()
        };
        Ok(Some(Self {
            source_path: path,
            observations: Vec::new(),
        }))
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Legacy databases used only the display name to identify a Vulkan
    /// adapter. Keep that evidence usable until this exact device/driver stack
    /// has produced its first fingerprinted observation; after that point the
    /// exact population becomes authoritative so historical measurements from
    /// another driver cannot dilute or bias learned selectors.
    fn has_exact_fingerprint_population(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
    ) -> bool {
        self.observations.iter().any(|observation| {
            observation.key.has_exact_fingerprint() && observation.key.matches_geometry(geometry)
        })
    }

    pub(crate) fn append_online_observation(
        &mut self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        sequence_microbatch_size: usize,
        state_checkpoint_stride: usize,
        h_backward_segment_schedule: Option<&str>,
        l_backward_segment_schedule: Option<&str>,
        h_backward_kernel_geometry: Option<&str>,
        l_backward_kernel_geometry: Option<&str>,
        rwkv_numerics_policy: &str,
        tokens_per_second: f64,
        measured_iterations: usize,
        telemetry: Option<&HierarchosTokenTapeOnlineTelemetry>,
    ) -> Result<()> {
        anyhow::ensure!(
            tokens_per_second.is_finite() && tokens_per_second > 0.0,
            "online Vulkan tape profile throughput must be finite and positive"
        );
        let key = StoredProfileKey::from_geometry(
            geometry,
            sequence_microbatch_size,
            state_checkpoint_stride,
            h_backward_segment_schedule,
            l_backward_segment_schedule,
            h_backward_kernel_geometry,
            l_backward_kernel_geometry,
            rwkv_numerics_policy,
        );
        let measured_iterations = measured_iterations.max(1);
        let record = serde_json::json!({
            "schema_version": PROFILE_SCHEMA_VERSION,
            "status": "ok",
            "model_source": "hierarchos-vulkan-online-autotune",
            "case_source": "live-training-step",
            "profile_key": &key,
            "result": {
                "plan_mode": "online-explore",
                "median_tokens_per_second": tokens_per_second,
                "measured_iterations": measured_iterations,
                "steady_state_telemetry": telemetry,
            }
        });
        let mut encoded =
            serde_json::to_vec(&record).context("serializing online Vulkan tape profile")?;
        encoded.push(b'\n');
        if let Some(parent) = self.source_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "creating Vulkan tape profile directory {}",
                    parent.display()
                )
            })?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.source_path)
            .with_context(|| {
                format!(
                    "opening Vulkan tape profile database {} for append",
                    self.source_path.display()
                )
            })?;
        file.write_all(&encoded).with_context(|| {
            format!(
                "appending Vulkan tape profile database {}",
                self.source_path.display()
            )
        })?;
        file.sync_data().with_context(|| {
            format!(
                "syncing Vulkan tape profile database {}",
                self.source_path.display()
            )
        })?;
        self.observations.push(StoredProfileObservation {
            key,
            median_tokens_per_second: tokens_per_second,
            measured_iterations,
        });
        Ok(())
    }

    pub(crate) fn ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
    ) -> Vec<HierarchosTokenTapeProfileScore> {
        let mut ranked = self.scored_candidates(geometry);
        ranked.sort_by(|lhs, rhs| {
            rhs.confidence_adjusted_tokens_per_second
                .total_cmp(&lhs.confidence_adjusted_tokens_per_second)
                .then_with(|| {
                    rhs.adaptive_tokens_per_second
                        .total_cmp(&lhs.adaptive_tokens_per_second)
                })
                .then_with(|| {
                    rhs.median_tokens_per_second
                        .total_cmp(&lhs.median_tokens_per_second)
                })
                .then_with(|| rhs.measured_iterations.cmp(&lhs.measured_iterations))
                .then_with(|| {
                    rhs.sequence_microbatch_size
                        .cmp(&lhs.sequence_microbatch_size)
                })
                .then_with(|| {
                    lhs.state_checkpoint_stride
                        .cmp(&rhs.state_checkpoint_stride)
                })
        });
        ranked
    }

    pub(crate) fn exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
    ) -> Vec<HierarchosTokenTapeProfileScore> {
        let mut ranked = self.scored_candidates(geometry);
        ranked.sort_by(|lhs, rhs| {
            rhs.exploration_score_tokens_per_second
                .total_cmp(&lhs.exploration_score_tokens_per_second)
                .then_with(|| {
                    rhs.observations_since_last_measurement
                        .cmp(&lhs.observations_since_last_measurement)
                })
                .then_with(|| {
                    rhs.confidence_adjusted_tokens_per_second
                        .total_cmp(&lhs.confidence_adjusted_tokens_per_second)
                })
                .then_with(|| {
                    rhs.sequence_microbatch_size
                        .cmp(&lhs.sequence_microbatch_size)
                })
                .then_with(|| {
                    lhs.state_checkpoint_stride
                        .cmp(&rhs.state_checkpoint_stride)
                })
        });
        ranked
    }

    pub(crate) fn sequence_microbatch_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeGeometryFactorScore> {
        self.tape_geometry_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            TapeGeometryFactorKind::SequenceMicrobatch,
        )
    }

    pub(crate) fn state_checkpoint_stride_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeGeometryFactorScore> {
        self.tape_geometry_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            TapeGeometryFactorKind::StateCheckpointStride,
        )
    }

    /// Collapse full training observations onto one tape geometry coordinate.
    /// The sibling tape coordinate and both recurrent branch topologies are
    /// marginalized out. This is the tape-memory analogue of the H/L marginal
    /// selectors below and prevents `microbatch × stride` from becoming a new
    /// composite search surface after recurrent topology was factorized.
    fn tape_geometry_factor_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
        factor: TapeGeometryFactorKind,
    ) -> Vec<HierarchosTokenTapeGeometryFactorScore> {
        let mut grouped: HashMap<usize, CandidateMeasurements> = HashMap::new();
        let mut geometry_observation_count = 0usize;
        let require_exact_fingerprint = self.has_exact_fingerprint_population(geometry);
        for observation in &self.observations {
            if !observation
                .key
                .matches_geometry_population(geometry, require_exact_fingerprint)
                || !observation.median_tokens_per_second.is_finite()
                || observation.median_tokens_per_second <= 0.0
            {
                continue;
            }
            let observation_numerics = observation
                .key
                .rwkv_numerics_policy
                .as_deref()
                .unwrap_or("strict-parity");
            if observation_numerics != rwkv_numerics_policy {
                continue;
            }
            let value = match factor {
                TapeGeometryFactorKind::SequenceMicrobatch => {
                    observation.key.sequence_microbatch_size
                }
                TapeGeometryFactorKind::StateCheckpointStride => {
                    observation.key.state_checkpoint_stride
                }
            };
            geometry_observation_count = geometry_observation_count.saturating_add(1);
            let measurements = grouped.entry(value).or_default();
            measurements.observations.push(CandidateMeasurement {
                throughput: observation.median_tokens_per_second,
                measured_iterations: observation.measured_iterations.max(1),
                geometry_ordinal: geometry_observation_count,
            });
            measurements.measured_iterations = measurements
                .measured_iterations
                .saturating_add(observation.measured_iterations.max(1));
        }

        let mut ranked = grouped
            .into_iter()
            .map(|(value, measurements)| {
                let profile_records = measurements.observations.len();
                let median_tokens_per_second = median(
                    measurements
                        .observations
                        .iter()
                        .map(|measurement| measurement.throughput)
                        .collect(),
                );
                let adaptive = adaptive_profile_statistics(
                    &measurements.observations,
                    geometry_observation_count,
                );
                let ucb_sampling_bonus = adaptive.adaptive_tokens_per_second
                    * PROFILE_UCB_EXPLORATION_SCALE
                    * (((geometry_observation_count.saturating_add(1)) as f64).ln()
                        / adaptive
                            .effective_measured_iterations
                            .max(PROFILE_MIN_EFFECTIVE_ITERATIONS))
                    .sqrt();
                HierarchosTokenTapeGeometryFactorScore {
                    value,
                    median_tokens_per_second,
                    adaptive_tokens_per_second: adaptive.adaptive_tokens_per_second,
                    confidence_adjusted_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 - PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty).max(0.0),
                    exploration_score_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 + PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty)
                        + ucb_sampling_bonus,
                    relative_uncertainty: adaptive.relative_uncertainty,
                    effective_measured_iterations: adaptive.effective_measured_iterations,
                    observations_since_last_measurement: adaptive
                        .observations_since_last_measurement,
                    profile_records,
                    measured_iterations: measurements.measured_iterations,
                }
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|lhs, rhs| {
            rhs.exploration_score_tokens_per_second
                .total_cmp(&lhs.exploration_score_tokens_per_second)
                .then_with(|| {
                    rhs.observations_since_last_measurement
                        .cmp(&lhs.observations_since_last_measurement)
                })
                .then_with(|| {
                    rhs.confidence_adjusted_tokens_per_second
                        .total_cmp(&lhs.confidence_adjusted_tokens_per_second)
                })
                .then_with(|| rhs.value.cmp(&lhs.value))
        });
        ranked
    }

    pub(crate) fn h_topology_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchTopologyScore> {
        self.branch_topology_exploration_ranked_candidates(geometry, rwkv_numerics_policy, true)
    }

    pub(crate) fn l_topology_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchTopologyScore> {
        self.branch_topology_exploration_ranked_candidates(geometry, rwkv_numerics_policy, false)
    }

    pub(crate) fn h_segment_schedule_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            true,
            BranchFactorKind::SegmentSchedule,
        )
    }

    pub(crate) fn l_segment_schedule_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            false,
            BranchFactorKind::SegmentSchedule,
        )
    }

    pub(crate) fn h_segment_state_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            true,
            BranchFactorKind::SegmentState,
        )
    }

    pub(crate) fn l_segment_state_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            false,
            BranchFactorKind::SegmentState,
        )
    }

    pub(crate) fn h_segment_projection_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            true,
            BranchFactorKind::SegmentProjection,
        )
    }

    pub(crate) fn l_segment_projection_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            false,
            BranchFactorKind::SegmentProjection,
        )
    }

    pub(crate) fn h_segment_low_rank_fan_in_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            true,
            BranchFactorKind::SegmentLowRankFanIn,
        )
    }

    pub(crate) fn l_segment_low_rank_fan_in_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            false,
            BranchFactorKind::SegmentLowRankFanIn,
        )
    }

    pub(crate) fn h_kernel_geometry_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            true,
            BranchFactorKind::KernelGeometry,
        )
    }

    pub(crate) fn l_kernel_geometry_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        self.branch_factor_exploration_ranked_candidates(
            geometry,
            rwkv_numerics_policy,
            false,
            BranchFactorKind::KernelGeometry,
        )
    }

    /// Marginalize the persistent observation stream down to one recurrent
    /// branch coordinate. The full segment schedule can remain an opaque legacy
    /// arm, or be projected further into state fusion depth, projection order,
    /// and low-rank fan-in depth. Tape geometry, sibling-branch choices, and
    /// every nonselected branch coordinate are integrated out. Numerics stays
    /// fixed because reduction order remains a mathematical/performance
    /// contract rather than a freely composable axis.
    fn branch_factor_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
        h_branch: bool,
        factor: BranchFactorKind,
    ) -> Vec<HierarchosTokenTapeBranchFactorScore> {
        let mut grouped: HashMap<Option<String>, CandidateMeasurements> = HashMap::new();
        let mut geometry_observation_count = 0usize;
        let require_exact_fingerprint = self.has_exact_fingerprint_population(geometry);
        for observation in &self.observations {
            if !observation
                .key
                .matches_geometry_population(geometry, require_exact_fingerprint)
                || !observation.median_tokens_per_second.is_finite()
                || observation.median_tokens_per_second <= 0.0
            {
                continue;
            }
            let observation_numerics = observation
                .key
                .rwkv_numerics_policy
                .as_deref()
                .unwrap_or("strict-parity");
            if observation_numerics != rwkv_numerics_policy {
                continue;
            }
            let segment_schedule = if h_branch {
                observation.key.h_backward_segment_schedule.as_deref()
            } else {
                observation.key.l_backward_segment_schedule.as_deref()
            };
            let kernel_geometry = if h_branch {
                observation.key.h_backward_kernel_geometry.clone()
            } else {
                observation.key.l_backward_kernel_geometry.clone()
            };
            let value = match factor {
                BranchFactorKind::SegmentSchedule => Some(segment_schedule.map(str::to_owned)),
                BranchFactorKind::KernelGeometry => Some(kernel_geometry),
                BranchFactorKind::SegmentState => segment_schedule
                    .and_then(parse_backward_segment_schedule_factors)
                    .map(|factors| Some(factors.state)),
                BranchFactorKind::SegmentProjection => segment_schedule
                    .and_then(parse_backward_segment_schedule_factors)
                    .map(|factors| Some(factors.projection)),
                BranchFactorKind::SegmentLowRankFanIn => segment_schedule
                    .and_then(parse_backward_segment_schedule_factors)
                    .map(|factors| factors.low_rank_fan_in),
            };
            let Some(value) = value else {
                continue;
            };
            geometry_observation_count = geometry_observation_count.saturating_add(1);
            let measurements = grouped.entry(value).or_default();
            measurements.observations.push(CandidateMeasurement {
                throughput: observation.median_tokens_per_second,
                measured_iterations: observation.measured_iterations.max(1),
                geometry_ordinal: geometry_observation_count,
            });
            measurements.measured_iterations = measurements
                .measured_iterations
                .saturating_add(observation.measured_iterations.max(1));
        }

        let mut ranked = grouped
            .into_iter()
            .map(|(value, measurements)| {
                let profile_records = measurements.observations.len();
                let median_tokens_per_second = median(
                    measurements
                        .observations
                        .iter()
                        .map(|measurement| measurement.throughput)
                        .collect(),
                );
                let adaptive = adaptive_profile_statistics(
                    &measurements.observations,
                    geometry_observation_count,
                );
                let ucb_sampling_bonus = adaptive.adaptive_tokens_per_second
                    * PROFILE_UCB_EXPLORATION_SCALE
                    * (((geometry_observation_count.saturating_add(1)) as f64).ln()
                        / adaptive
                            .effective_measured_iterations
                            .max(PROFILE_MIN_EFFECTIVE_ITERATIONS))
                    .sqrt();
                HierarchosTokenTapeBranchFactorScore {
                    value,
                    median_tokens_per_second,
                    adaptive_tokens_per_second: adaptive.adaptive_tokens_per_second,
                    confidence_adjusted_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 - PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty).max(0.0),
                    exploration_score_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 + PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty)
                        + ucb_sampling_bonus,
                    relative_uncertainty: adaptive.relative_uncertainty,
                    effective_measured_iterations: adaptive.effective_measured_iterations,
                    observations_since_last_measurement: adaptive
                        .observations_since_last_measurement,
                    profile_records,
                    measured_iterations: measurements.measured_iterations,
                }
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|lhs, rhs| {
            rhs.exploration_score_tokens_per_second
                .total_cmp(&lhs.exploration_score_tokens_per_second)
                .then_with(|| {
                    rhs.observations_since_last_measurement
                        .cmp(&lhs.observations_since_last_measurement)
                })
                .then_with(|| {
                    rhs.confidence_adjusted_tokens_per_second
                        .total_cmp(&lhs.confidence_adjusted_tokens_per_second)
                })
                .then_with(|| lhs.value.cmp(&rhs.value))
        });
        ranked
    }

    /// Project the same device/model observation stream onto only H or only L.
    /// Tape microbatch/stride and the sibling branch are marginalized out, while
    /// numerics remains fixed because reduction order can materially change the
    /// performance and parity contract of both branches.
    fn branch_topology_exploration_ranked_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        rwkv_numerics_policy: &str,
        h_branch: bool,
    ) -> Vec<HierarchosTokenTapeBranchTopologyScore> {
        let mut grouped: HashMap<HierarchosTokenTapeBranchTopology, CandidateMeasurements> =
            HashMap::new();
        let mut geometry_observation_count = 0usize;
        let require_exact_fingerprint = self.has_exact_fingerprint_population(geometry);
        for observation in &self.observations {
            if !observation
                .key
                .matches_geometry_population(geometry, require_exact_fingerprint)
                || !observation.median_tokens_per_second.is_finite()
                || observation.median_tokens_per_second <= 0.0
            {
                continue;
            }
            let observation_numerics = observation
                .key
                .rwkv_numerics_policy
                .as_deref()
                .unwrap_or("strict-parity");
            if observation_numerics != rwkv_numerics_policy {
                continue;
            }
            geometry_observation_count = geometry_observation_count.saturating_add(1);
            let topology = if h_branch {
                HierarchosTokenTapeBranchTopology {
                    backward_segment_schedule: observation.key.h_backward_segment_schedule.clone(),
                    backward_kernel_geometry: observation.key.h_backward_kernel_geometry.clone(),
                }
            } else {
                HierarchosTokenTapeBranchTopology {
                    backward_segment_schedule: observation.key.l_backward_segment_schedule.clone(),
                    backward_kernel_geometry: observation.key.l_backward_kernel_geometry.clone(),
                }
            };
            let measurements = grouped.entry(topology).or_default();
            measurements.observations.push(CandidateMeasurement {
                throughput: observation.median_tokens_per_second,
                measured_iterations: observation.measured_iterations.max(1),
                geometry_ordinal: geometry_observation_count,
            });
            measurements.measured_iterations = measurements
                .measured_iterations
                .saturating_add(observation.measured_iterations.max(1));
        }

        let mut ranked = grouped
            .into_iter()
            .map(|(topology, measurements)| {
                let profile_records = measurements.observations.len();
                let median_tokens_per_second = median(
                    measurements
                        .observations
                        .iter()
                        .map(|measurement| measurement.throughput)
                        .collect(),
                );
                let adaptive = adaptive_profile_statistics(
                    &measurements.observations,
                    geometry_observation_count,
                );
                let ucb_sampling_bonus = adaptive.adaptive_tokens_per_second
                    * PROFILE_UCB_EXPLORATION_SCALE
                    * (((geometry_observation_count.saturating_add(1)) as f64).ln()
                        / adaptive
                            .effective_measured_iterations
                            .max(PROFILE_MIN_EFFECTIVE_ITERATIONS))
                    .sqrt();
                HierarchosTokenTapeBranchTopologyScore {
                    topology,
                    median_tokens_per_second,
                    adaptive_tokens_per_second: adaptive.adaptive_tokens_per_second,
                    confidence_adjusted_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 - PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty).max(0.0),
                    exploration_score_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 + PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty)
                        + ucb_sampling_bonus,
                    relative_uncertainty: adaptive.relative_uncertainty,
                    effective_measured_iterations: adaptive.effective_measured_iterations,
                    observations_since_last_measurement: adaptive
                        .observations_since_last_measurement,
                    profile_records,
                    measured_iterations: measurements.measured_iterations,
                }
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|lhs, rhs| {
            rhs.exploration_score_tokens_per_second
                .total_cmp(&lhs.exploration_score_tokens_per_second)
                .then_with(|| {
                    rhs.observations_since_last_measurement
                        .cmp(&lhs.observations_since_last_measurement)
                })
                .then_with(|| {
                    rhs.confidence_adjusted_tokens_per_second
                        .total_cmp(&lhs.confidence_adjusted_tokens_per_second)
                })
                .then_with(|| {
                    lhs.topology
                        .backward_segment_schedule
                        .cmp(&rhs.topology.backward_segment_schedule)
                })
                .then_with(|| {
                    lhs.topology
                        .backward_kernel_geometry
                        .cmp(&rhs.topology.backward_kernel_geometry)
                })
        });
        ranked
    }

    fn scored_candidates(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
    ) -> Vec<HierarchosTokenTapeProfileScore> {
        let mut grouped: HashMap<ProfileArmKey, CandidateMeasurements> = HashMap::new();
        let mut geometry_observation_count = 0usize;
        let require_exact_fingerprint = self.has_exact_fingerprint_population(geometry);
        for observation in &self.observations {
            if !observation
                .key
                .matches_geometry_population(geometry, require_exact_fingerprint)
                || !observation.median_tokens_per_second.is_finite()
                || observation.median_tokens_per_second <= 0.0
            {
                continue;
            }
            geometry_observation_count = geometry_observation_count.saturating_add(1);
            let candidate = ProfileArmKey {
                sequence_microbatch_size: observation.key.sequence_microbatch_size,
                state_checkpoint_stride: observation.key.state_checkpoint_stride,
                device_local_pressure_bucket: observation.key.device_local_pressure_bucket,
                h_backward_segment_schedule: observation.key.h_backward_segment_schedule.clone(),
                l_backward_segment_schedule: observation.key.l_backward_segment_schedule.clone(),
                h_backward_kernel_geometry: observation.key.h_backward_kernel_geometry.clone(),
                l_backward_kernel_geometry: observation.key.l_backward_kernel_geometry.clone(),
                rwkv_numerics_policy: observation
                    .key
                    .rwkv_numerics_policy
                    .clone()
                    .unwrap_or_else(|| "strict-parity".to_string()),
            };
            let measurements = grouped.entry(candidate).or_default();
            measurements.observations.push(CandidateMeasurement {
                throughput: observation.median_tokens_per_second,
                measured_iterations: observation.measured_iterations.max(1),
                geometry_ordinal: geometry_observation_count,
            });
            measurements.measured_iterations = measurements
                .measured_iterations
                .saturating_add(observation.measured_iterations.max(1));
        }

        grouped
            .into_iter()
            .map(|(candidate, measurements)| {
                let profile_records = measurements.observations.len();
                let median_tokens_per_second = median(
                    measurements
                        .observations
                        .iter()
                        .map(|measurement| measurement.throughput)
                        .collect(),
                );
                let adaptive = adaptive_profile_statistics(
                    &measurements.observations,
                    geometry_observation_count,
                );
                let ucb_sampling_bonus = adaptive.adaptive_tokens_per_second
                    * PROFILE_UCB_EXPLORATION_SCALE
                    * (((geometry_observation_count.saturating_add(1)) as f64).ln()
                        / adaptive
                            .effective_measured_iterations
                            .max(PROFILE_MIN_EFFECTIVE_ITERATIONS))
                    .sqrt();
                HierarchosTokenTapeProfileScore {
                    sequence_microbatch_size: candidate.sequence_microbatch_size,
                    state_checkpoint_stride: candidate.state_checkpoint_stride,
                    device_local_pressure_bucket: candidate.device_local_pressure_bucket,
                    h_backward_segment_schedule: candidate.h_backward_segment_schedule,
                    l_backward_segment_schedule: candidate.l_backward_segment_schedule,
                    h_backward_kernel_geometry: candidate.h_backward_kernel_geometry,
                    l_backward_kernel_geometry: candidate.l_backward_kernel_geometry,
                    rwkv_numerics_policy: candidate.rwkv_numerics_policy,
                    median_tokens_per_second,
                    adaptive_tokens_per_second: adaptive.adaptive_tokens_per_second,
                    confidence_adjusted_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 - PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty).max(0.0),
                    exploration_score_tokens_per_second: adaptive.adaptive_tokens_per_second
                        * (1.0 + PROFILE_CONFIDENCE_Z * adaptive.relative_uncertainty)
                        + ucb_sampling_bonus,
                    relative_uncertainty: adaptive.relative_uncertainty,
                    effective_measured_iterations: adaptive.effective_measured_iterations,
                    observations_since_last_measurement: adaptive
                        .observations_since_last_measurement,
                    profile_records,
                    measured_iterations: measurements.measured_iterations,
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ProfileArmKey {
    sequence_microbatch_size: usize,
    state_checkpoint_stride: usize,
    device_local_pressure_bucket: Option<u8>,
    h_backward_segment_schedule: Option<String>,
    l_backward_segment_schedule: Option<String>,
    h_backward_kernel_geometry: Option<String>,
    l_backward_kernel_geometry: Option<String>,
    rwkv_numerics_policy: String,
}

#[derive(Default)]
struct CandidateMeasurements {
    observations: Vec<CandidateMeasurement>,
    measured_iterations: usize,
}

#[derive(Clone, Copy)]
struct CandidateMeasurement {
    throughput: f64,
    measured_iterations: usize,
    geometry_ordinal: usize,
}

#[derive(Clone, Copy)]
struct AdaptiveProfileStatistics {
    adaptive_tokens_per_second: f64,
    relative_uncertainty: f64,
    effective_measured_iterations: f64,
    observations_since_last_measurement: usize,
}

#[derive(Clone, Debug)]
struct StoredProfileObservation {
    key: StoredProfileKey,
    median_tokens_per_second: f64,
    measured_iterations: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredProfileKey {
    device: String,
    #[serde(default)]
    device_uuid: Option<String>,
    #[serde(default)]
    driver_uuid: Option<String>,
    subgroup_size: u32,
    #[serde(default = "default_training_precision_policy")]
    training_precision_policy: String,
    #[serde(default)]
    backward_kernel_geometry_revision: u32,
    #[serde(default)]
    device_local_pressure_bucket: Option<u8>,
    architecture_revision: String,
    batch: usize,
    context_dim: usize,
    persistent_dim: usize,
    ltm_slots: usize,
    ltm_key_dim: usize,
    ltm_val_dim: usize,
    ltm_topk: usize,
    vocab_size: usize,
    h_hidden: usize,
    l_hidden: usize,
    h_width: usize,
    l_width: usize,
    h_state_size: usize,
    l_state_size: usize,
    h_rwkv_head_size: usize,
    l_rwkv_head_size: usize,
    h_low_rank_ranks: Option<(usize, usize, usize)>,
    l_low_rank_ranks: Option<(usize, usize, usize)>,
    token_adapter_rank: usize,
    max_h_steps: usize,
    max_l_steps: usize,
    tokens_per_sequence: usize,
    sequences: usize,
    sequence_microbatch_size: usize,
    state_checkpoint_stride: usize,
    #[serde(default)]
    h_backward_segment_schedule: Option<String>,
    #[serde(default)]
    l_backward_segment_schedule: Option<String>,
    #[serde(default)]
    h_backward_kernel_geometry: Option<String>,
    #[serde(default)]
    l_backward_kernel_geometry: Option<String>,
    #[serde(default)]
    rwkv_numerics_policy: Option<String>,
}

impl StoredProfileKey {
    fn has_exact_fingerprint(&self) -> bool {
        self.device_uuid.is_some() && self.driver_uuid.is_some()
    }

    fn matches_geometry_population(
        &self,
        geometry: &HierarchosTokenTapeProfileGeometry,
        require_exact_fingerprint: bool,
    ) -> bool {
        self.matches_geometry(geometry)
            && (!require_exact_fingerprint || self.has_exact_fingerprint())
    }

    fn from_geometry(
        geometry: &HierarchosTokenTapeProfileGeometry,
        sequence_microbatch_size: usize,
        state_checkpoint_stride: usize,
        h_backward_segment_schedule: Option<&str>,
        l_backward_segment_schedule: Option<&str>,
        h_backward_kernel_geometry: Option<&str>,
        l_backward_kernel_geometry: Option<&str>,
        rwkv_numerics_policy: &str,
    ) -> Self {
        Self {
            device: geometry.device.clone(),
            device_uuid: Some(geometry.device_uuid.clone()),
            driver_uuid: Some(geometry.driver_uuid.clone()),
            subgroup_size: geometry.subgroup_size,
            training_precision_policy: geometry.training_precision_policy.clone(),
            backward_kernel_geometry_revision: geometry.backward_kernel_geometry_revision,
            device_local_pressure_bucket: geometry.device_local_pressure_bucket,
            architecture_revision: geometry.architecture_revision.clone(),
            batch: geometry.batch,
            context_dim: geometry.context_dim,
            persistent_dim: geometry.persistent_dim,
            ltm_slots: geometry.ltm_slots,
            ltm_key_dim: geometry.ltm_key_dim,
            ltm_val_dim: geometry.ltm_val_dim,
            ltm_topk: geometry.ltm_topk,
            vocab_size: geometry.vocab_size,
            h_hidden: geometry.h_hidden,
            l_hidden: geometry.l_hidden,
            h_width: geometry.h_width,
            l_width: geometry.l_width,
            h_state_size: geometry.h_state_size,
            l_state_size: geometry.l_state_size,
            h_rwkv_head_size: geometry.h_rwkv_head_size,
            l_rwkv_head_size: geometry.l_rwkv_head_size,
            h_low_rank_ranks: geometry.h_low_rank_ranks,
            l_low_rank_ranks: geometry.l_low_rank_ranks,
            token_adapter_rank: geometry.token_adapter_rank,
            max_h_steps: geometry.max_h_steps,
            max_l_steps: geometry.max_l_steps,
            tokens_per_sequence: geometry.tokens_per_sequence,
            sequences: geometry.sequences,
            sequence_microbatch_size,
            state_checkpoint_stride,
            h_backward_segment_schedule: h_backward_segment_schedule.map(str::to_owned),
            l_backward_segment_schedule: l_backward_segment_schedule.map(str::to_owned),
            h_backward_kernel_geometry: h_backward_kernel_geometry.map(str::to_owned),
            l_backward_kernel_geometry: l_backward_kernel_geometry.map(str::to_owned),
            rwkv_numerics_policy: Some(rwkv_numerics_policy.to_owned()),
        }
    }

    fn matches_geometry(&self, geometry: &HierarchosTokenTapeProfileGeometry) -> bool {
        self.device == geometry.device
            && match (&self.device_uuid, &self.driver_uuid) {
                (Some(device_uuid), Some(driver_uuid)) => {
                    device_uuid == &geometry.device_uuid && driver_uuid == &geometry.driver_uuid
                }
                // Profile databases written before Vulkan UUID scoping are
                // intentionally retained as lower-specificity historical
                // evidence. A partially fingerprinted record is ambiguous and
                // fails closed rather than leaking evidence across drivers.
                (None, None) => true,
                _ => false,
            }
            && self.subgroup_size == geometry.subgroup_size
            && self.training_precision_policy == geometry.training_precision_policy
            && (self.backward_kernel_geometry_revision
                == geometry.backward_kernel_geometry_revision
                || (self.backward_kernel_geometry_revision == 0
                    && self.h_backward_kernel_geometry.is_none()
                    && self.l_backward_kernel_geometry.is_none()))
            && (self.device_local_pressure_bucket.is_none()
                || self.device_local_pressure_bucket == geometry.device_local_pressure_bucket)
            && self.architecture_revision == geometry.architecture_revision
            && self.batch == geometry.batch
            && self.context_dim == geometry.context_dim
            && self.persistent_dim == geometry.persistent_dim
            && self.ltm_slots == geometry.ltm_slots
            && self.ltm_key_dim == geometry.ltm_key_dim
            && self.ltm_val_dim == geometry.ltm_val_dim
            && self.ltm_topk == geometry.ltm_topk
            && self.vocab_size == geometry.vocab_size
            && self.h_hidden == geometry.h_hidden
            && self.l_hidden == geometry.l_hidden
            && self.h_width == geometry.h_width
            && self.l_width == geometry.l_width
            && self.h_state_size == geometry.h_state_size
            && self.l_state_size == geometry.l_state_size
            && self.h_rwkv_head_size == geometry.h_rwkv_head_size
            && self.l_rwkv_head_size == geometry.l_rwkv_head_size
            && self.h_low_rank_ranks == geometry.h_low_rank_ranks
            && self.l_low_rank_ranks == geometry.l_low_rank_ranks
            && self.token_adapter_rank == geometry.token_adapter_rank
            && self.max_h_steps == geometry.max_h_steps
            && self.max_l_steps == geometry.max_l_steps
            && self.tokens_per_sequence == geometry.tokens_per_sequence
            && self.sequences == geometry.sequences
    }
}

fn default_training_precision_policy() -> String {
    "fp32".to_string()
}

#[derive(Deserialize)]
struct StoredProfileRecord {
    schema_version: u32,
    status: String,
    profile_key: Option<StoredProfileKey>,
    result: Option<StoredProfileResult>,
}

#[derive(Deserialize)]
struct StoredProfileResult {
    plan_mode: String,
    median_tokens_per_second: f64,
    measured_iterations: usize,
}

fn parse_profile_jsonl(text: &str) -> Result<Vec<StoredProfileObservation>> {
    let mut observations = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: StoredProfileRecord = serde_json::from_str(line)
            .with_context(|| format!("invalid profile JSONL record at line {}", line_index + 1))?;
        if record.schema_version != PROFILE_SCHEMA_VERSION || record.status != "ok" {
            continue;
        }
        let (Some(key), Some(result)) = (record.profile_key, record.result) else {
            continue;
        };
        // Only controlled candidate runs are training data for the scheduler.
        // Re-ingesting automatic decisions would double-count the currently
        // selected plan and create a self-reinforcing feedback loop.
        if result.plan_mode != "explicit" && result.plan_mode != "online-explore" {
            continue;
        }
        observations.push(StoredProfileObservation {
            key,
            median_tokens_per_second: result.median_tokens_per_second,
            measured_iterations: result.measured_iterations,
        });
    }
    Ok(observations)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn adaptive_profile_statistics(
    measurements: &[CandidateMeasurement],
    geometry_observation_count: usize,
) -> AdaptiveProfileStatistics {
    debug_assert!(!measurements.is_empty());
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    let mut effective_records = 0.0;
    let mut last_ordinal = 0usize;
    for measurement in measurements {
        let age = geometry_observation_count.saturating_sub(measurement.geometry_ordinal);
        let decay = PROFILE_OBSERVATION_DECAY.powi(i32::try_from(age).unwrap_or(i32::MAX));
        let iteration_weight = measurement.measured_iterations.max(1) as f64;
        let weight = decay * iteration_weight;
        weighted_sum += weight * measurement.throughput;
        total_weight += weight;
        effective_records += decay;
        last_ordinal = last_ordinal.max(measurement.geometry_ordinal);
    }
    let adaptive_tokens_per_second = weighted_sum / total_weight.max(f64::MIN_POSITIVE);
    let mut weighted_variance = 0.0;
    for measurement in measurements {
        let age = geometry_observation_count.saturating_sub(measurement.geometry_ordinal);
        let decay = PROFILE_OBSERVATION_DECAY.powi(i32::try_from(age).unwrap_or(i32::MAX));
        let weight = decay * measurement.measured_iterations.max(1) as f64;
        let delta = measurement.throughput - adaptive_tokens_per_second;
        weighted_variance += weight * delta * delta;
    }
    weighted_variance /= total_weight.max(f64::MIN_POSITIVE);
    let dispersion_relative_standard_error = if adaptive_tokens_per_second <= f64::EPSILON {
        0.0
    } else {
        weighted_variance.sqrt() / adaptive_tokens_per_second / effective_records.max(1.0).sqrt()
    };
    let sampling_floor =
        PROFILE_RELATIVE_NOISE_FLOOR / total_weight.max(PROFILE_MIN_EFFECTIVE_ITERATIONS).sqrt();
    AdaptiveProfileStatistics {
        adaptive_tokens_per_second,
        relative_uncertainty: dispersion_relative_standard_error.hypot(sampling_floor),
        effective_measured_iterations: total_weight,
        observations_since_last_measurement: geometry_observation_count
            .saturating_sub(last_ordinal),
    }
}

fn default_profile_candidates() -> Result<[PathBuf; 2]> {
    let cwd = std::env::current_dir().context("resolving current directory for tape profiles")?;
    Ok([
        cwd.join("benchmark_results")
            .join(HIERARCHOS_TOKEN_TAPE_PROFILE_FILENAME),
        cwd.join("..")
            .join("benchmark_results")
            .join(HIERARCHOS_TOKEN_TAPE_PROFILE_FILENAME),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> HierarchosTokenTapeProfileGeometry {
        HierarchosTokenTapeProfileGeometry {
            device: "TEST GPU".to_string(),
            device_uuid: "test-device-uuid".to_string(),
            driver_uuid: "test-driver-uuid".to_string(),
            subgroup_size: 64,
            training_precision_policy: "fp32".to_string(),
            backward_kernel_geometry_revision: BACKWARD_KERNEL_GEOMETRY_POLICY_REVISION,
            device_local_pressure_bucket: Some(2),
            architecture_revision: "coherent-v9".to_string(),
            batch: 2,
            context_dim: 32,
            persistent_dim: 8,
            ltm_slots: 16,
            ltm_key_dim: 8,
            ltm_val_dim: 8,
            ltm_topk: 2,
            vocab_size: 64,
            h_hidden: 32,
            l_hidden: 32,
            h_width: 32,
            l_width: 32,
            h_state_size: 36,
            l_state_size: 36,
            h_rwkv_head_size: 32,
            l_rwkv_head_size: 32,
            h_low_rank_ranks: Some((8, 8, 8)),
            l_low_rank_ranks: Some((8, 8, 8)),
            token_adapter_rank: 32,
            max_h_steps: 3,
            max_l_steps: 2,
            tokens_per_sequence: 8,
            sequences: 4,
        }
    }

    fn record(device: &str, microbatch: usize, stride: usize, throughput: f64) -> String {
        format!(
            r#"{{"schema_version":1,"status":"ok","profile_key":{{"device":"{device}","subgroup_size":64,"architecture_revision":"coherent-v9","batch":2,"context_dim":32,"persistent_dim":8,"ltm_slots":16,"ltm_key_dim":8,"ltm_val_dim":8,"ltm_topk":2,"vocab_size":64,"h_hidden":32,"l_hidden":32,"h_width":32,"l_width":32,"h_state_size":36,"l_state_size":36,"h_rwkv_head_size":32,"l_rwkv_head_size":32,"h_low_rank_ranks":[8,8,8],"l_low_rank_ranks":[8,8,8],"token_adapter_rank":32,"max_h_steps":3,"max_l_steps":2,"tokens_per_sequence":8,"sequences":4,"sequence_microbatch_size":{microbatch},"state_checkpoint_stride":{stride}}},"result":{{"plan_mode":"explicit","median_tokens_per_second":{throughput},"measured_iterations":3}}}}"#
        )
    }

    fn fingerprinted_record(
        device_uuid: &str,
        driver_uuid: &str,
        microbatch: usize,
        stride: usize,
        throughput: f64,
    ) -> String {
        record("TEST GPU", microbatch, stride, throughput).replacen(
            "\"subgroup_size\":64,",
            &format!(
                "\"subgroup_size\":64,\"device_uuid\":\"{device_uuid}\",\"driver_uuid\":\"{driver_uuid}\"," 
            ),
            1,
        )
    }

    fn topology_record(
        microbatch: usize,
        stride: usize,
        h_schedule: &str,
        l_schedule: &str,
        throughput: f64,
    ) -> String {
        record("TEST GPU", microbatch, stride, throughput).replace(
            &format!(r#""state_checkpoint_stride":{stride}"#),
            &format!(
                r#""state_checkpoint_stride":{stride},"h_backward_segment_schedule":"{h_schedule}","l_backward_segment_schedule":"{l_schedule}""#
            ),
        )
    }

    fn kernel_geometry_record(
        microbatch: usize,
        stride: usize,
        h_geometry: &str,
        l_geometry: &str,
        throughput: f64,
    ) -> String {
        record("TEST GPU", microbatch, stride, throughput).replace(
            &format!(r#""state_checkpoint_stride":{stride}"#),
            &format!(
                r#""state_checkpoint_stride":{stride},"backward_kernel_geometry_revision":{BACKWARD_KERNEL_GEOMETRY_POLICY_REVISION},"h_backward_kernel_geometry":"{h_geometry}","l_backward_kernel_geometry":"{l_geometry}""#
            ),
        )
    }

    fn topology_kernel_record(
        microbatch: usize,
        stride: usize,
        h_schedule: &str,
        l_schedule: &str,
        h_geometry: &str,
        l_geometry: &str,
        throughput: f64,
    ) -> String {
        record("TEST GPU", microbatch, stride, throughput).replace(
            &format!(r#""state_checkpoint_stride":{stride}"#),
            &format!(
                r#""state_checkpoint_stride":{stride},"backward_kernel_geometry_revision":{BACKWARD_KERNEL_GEOMETRY_POLICY_REVISION},"h_backward_segment_schedule":"{h_schedule}","l_backward_segment_schedule":"{l_schedule}","h_backward_kernel_geometry":"{h_geometry}","l_backward_kernel_geometry":"{l_geometry}""#
            ),
        )
    }

    fn pressure_record(microbatch: usize, stride: usize, bucket: u8, throughput: f64) -> String {
        record("TEST GPU", microbatch, stride, throughput).replace(
            &format!(r#""state_checkpoint_stride":{stride}"#),
            &format!(
                r#""state_checkpoint_stride":{stride},"device_local_pressure_bucket":{bucket}"#
            ),
        )
    }

    #[test]
    fn ranks_matching_geometry_by_median_end_to_end_throughput() {
        let text = [
            record("TEST GPU", 4, 1, 100.0),
            record("TEST GPU", 2, 2, 140.0),
            record("TEST GPU", 2, 2, 160.0),
            record("OTHER GPU", 1, 1, 1000.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };

        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].sequence_microbatch_size, 2);
        assert_eq!(ranked[0].state_checkpoint_stride, 2);
        assert_eq!(ranked[0].median_tokens_per_second, 150.0);
        assert_eq!(ranked[0].profile_records, 2);
        assert_eq!(ranked[0].measured_iterations, 6);
    }

    #[test]
    fn device_driver_fingerprint_is_exact_when_present_and_legacy_records_remain_fallback() {
        let legacy_database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("legacy-test.jsonl"),
            observations: parse_profile_jsonl(&record("TEST GPU", 2, 2, 100.0)).unwrap(),
        };
        let legacy_ranked = legacy_database.ranked_candidates(&geometry());
        assert_eq!(legacy_ranked.len(), 1);
        assert_eq!(legacy_ranked[0].sequence_microbatch_size, 2);

        let text = [
            // Historical name-only evidence is superseded after this exact
            // Vulkan stack begins producing fingerprinted measurements.
            record("TEST GPU", 2, 2, 100.0),
            // Fresh evidence from this exact Vulkan device/driver wins.
            fingerprinted_record("test-device-uuid", "test-driver-uuid", 4, 2, 300.0),
            // Same display name and physical UUID, different driver/compiler
            // stack: this must not contaminate the live ranking.
            fingerprinted_record("test-device-uuid", "other-driver-uuid", 8, 2, 900.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };

        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].sequence_microbatch_size, 4);
        assert!(!ranked
            .iter()
            .any(|score| score.sequence_microbatch_size == 2));
        assert!(!ranked
            .iter()
            .any(|score| score.sequence_microbatch_size == 8));
    }

    #[test]
    fn confidence_ranking_prefers_repeatable_candidate_over_marginal_one_off_win() {
        let text = [
            record("TEST GPU", 4, 1, 165.0),
            record("TEST GPU", 2, 2, 160.0),
            record("TEST GPU", 2, 2, 160.0),
            record("TEST GPU", 2, 2, 160.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };

        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked[0].sequence_microbatch_size, 2);
        assert_eq!(ranked[0].state_checkpoint_stride, 2);
        assert_eq!(ranked[0].profile_records, 3);
        assert!(
            ranked[0].confidence_adjusted_tokens_per_second
                > ranked[1].confidence_adjusted_tokens_per_second
        );
        assert!(ranked[0].relative_uncertainty < ranked[1].relative_uncertainty);
    }

    #[test]
    fn adaptive_ranking_lets_recent_slowdown_overtake_old_history() {
        let mut records = Vec::new();
        records.extend((0..5).map(|_| record("TEST GPU", 4, 1, 240.0)));
        records.extend((0..3).map(|_| record("TEST GPU", 2, 2, 180.0)));
        records.extend((0..2).map(|_| record("TEST GPU", 4, 1, 100.0)));
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&records.join("\n")).unwrap(),
        };

        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked[0].sequence_microbatch_size, 2);
        assert_eq!(ranked[0].state_checkpoint_stride, 2);
        let historically_fast = ranked
            .iter()
            .find(|score| score.sequence_microbatch_size == 4)
            .unwrap();
        assert_eq!(historically_fast.median_tokens_per_second, 240.0);
        assert!(historically_fast.adaptive_tokens_per_second < 190.0);
        assert!(
            ranked[0].confidence_adjusted_tokens_per_second
                > historically_fast.confidence_adjusted_tokens_per_second
        );
    }

    #[test]
    fn decayed_ucb_reopens_stale_candidate_for_measurement() {
        let mut records = vec![record("TEST GPU", 4, 1, 200.0)];
        records.extend((0..10).map(|_| record("TEST GPU", 2, 2, 180.0)));
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&records.join("\n")).unwrap(),
        };

        let exploit = database.ranked_candidates(&geometry());
        assert_eq!(exploit[0].sequence_microbatch_size, 2);
        let explore = database.exploration_ranked_candidates(&geometry());
        assert_eq!(explore[0].sequence_microbatch_size, 4);
        assert!(explore[0].observations_since_last_measurement >= 10);
        assert!(explore[0].effective_measured_iterations < 1.1);
        assert!(
            explore[0].exploration_score_tokens_per_second
                > explore[1].exploration_score_tokens_per_second
        );
    }

    #[test]
    fn ignores_rejected_and_unknown_schema_records() {
        let mut rejected = record("TEST GPU", 4, 1, 100.0);
        rejected = rejected.replace("\"status\":\"ok\"", "\"status\":\"rejected\"");
        let unknown = record("TEST GPU", 2, 1, 200.0)
            .replace("\"schema_version\":1", "\"schema_version\":999");
        let observations = parse_profile_jsonl(&format!("{rejected}\n{unknown}")).unwrap();
        assert!(observations.is_empty());
    }

    #[test]
    fn ignores_automatic_scheduler_observations() {
        let automatic = record("TEST GPU", 4, 1, 1000.0)
            .replace("\"plan_mode\":\"explicit\"", "\"plan_mode\":\"automatic\"");
        let explicit = record("TEST GPU", 2, 1, 200.0);
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&format!("{automatic}\n{explicit}")).unwrap(),
        };
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].sequence_microbatch_size, 2);
        assert_eq!(ranked[0].median_tokens_per_second, 200.0);
    }

    #[test]
    fn accepts_online_exploration_as_controlled_scheduler_training_data() {
        let online = record("TEST GPU", 3, 2, 210.0).replace(
            "\"plan_mode\":\"explicit\"",
            "\"plan_mode\":\"online-explore\"",
        );
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&online).unwrap(),
        };
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].sequence_microbatch_size, 3);
        assert_eq!(ranked[0].state_checkpoint_stride, 2);
    }

    #[test]
    fn topology_variants_are_distinct_policy_arms_while_legacy_records_remain_valid() {
        let text = [
            record("TEST GPU", 2, 2, 100.0),
            topology_record(2, 2, "h-fast", "l-base", 220.0),
            topology_record(2, 2, "h-base", "l-fast", 180.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 3);
        assert_eq!(
            ranked[0].h_backward_segment_schedule.as_deref(),
            Some("h-fast")
        );
        assert_eq!(
            ranked[0].l_backward_segment_schedule.as_deref(),
            Some("l-base")
        );
        assert!(ranked.iter().any(|score| {
            score.h_backward_segment_schedule.is_none()
                && score.l_backward_segment_schedule.is_none()
        }));
    }

    #[test]
    fn factorized_h_l_topology_scores_can_compose_an_unmeasured_pair() {
        let text = [
            topology_record(2, 2, "h-fast", "l-base", 300.0),
            topology_record(2, 2, "h-base", "l-fast", 290.0),
            topology_record(2, 2, "h-base", "l-base", 100.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };

        let h_ranked =
            database.h_topology_exploration_ranked_candidates(&geometry(), "strict-parity");
        let l_ranked =
            database.l_topology_exploration_ranked_candidates(&geometry(), "strict-parity");
        assert_eq!(
            h_ranked[0].topology.backward_segment_schedule.as_deref(),
            Some("h-fast")
        );
        assert_eq!(
            l_ranked[0].topology.backward_segment_schedule.as_deref(),
            Some("l-fast")
        );
        let global = database.ranked_candidates(&geometry());
        assert!(!global.iter().any(|score| {
            score.h_backward_segment_schedule.as_deref() == Some("h-fast")
                && score.l_backward_segment_schedule.as_deref() == Some("l-fast")
        }));
    }

    #[test]
    fn factorized_tape_geometry_scores_can_compose_an_unmeasured_pair() {
        let text = [
            record("TEST GPU", 4, 2, 300.0),
            record("TEST GPU", 2, 4, 290.0),
            record("TEST GPU", 2, 2, 100.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };

        let microbatches = database
            .sequence_microbatch_exploration_ranked_candidates(&geometry(), "strict-parity");
        let strides = database
            .state_checkpoint_stride_exploration_ranked_candidates(&geometry(), "strict-parity");
        let best_microbatch = microbatches
            .iter()
            .max_by(|left, right| {
                left.confidence_adjusted_tokens_per_second
                    .total_cmp(&right.confidence_adjusted_tokens_per_second)
            })
            .unwrap();
        let best_stride = strides
            .iter()
            .max_by(|left, right| {
                left.confidence_adjusted_tokens_per_second
                    .total_cmp(&right.confidence_adjusted_tokens_per_second)
            })
            .unwrap();
        assert_eq!(best_microbatch.value, 4);
        assert_eq!(best_stride.value, 4);

        let global = database.ranked_candidates(&geometry());
        assert!(!global.iter().any(|score| {
            score.sequence_microbatch_size == 4 && score.state_checkpoint_stride == 4
        }));
    }

    #[test]
    fn factorized_branch_schedule_and_kernel_scores_can_compose_unmeasured_pair() {
        let text = [
            topology_kernel_record(
                2,
                2,
                "h-base",
                "l-base",
                "rwkv-state-bwd-wg32",
                "rwkv-state-bwd-wg64",
                50.0,
            ),
            topology_kernel_record(
                2,
                2,
                "h-fast",
                "l-base",
                "rwkv-state-bwd-wg32",
                "rwkv-state-bwd-wg64",
                300.0,
            ),
            topology_kernel_record(
                2,
                2,
                "h-base",
                "l-base",
                "rwkv-state-bwd-wg128",
                "rwkv-state-bwd-wg64",
                250.0,
            ),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };

        let schedules =
            database.h_segment_schedule_exploration_ranked_candidates(&geometry(), "strict-parity");
        let kernels =
            database.h_kernel_geometry_exploration_ranked_candidates(&geometry(), "strict-parity");
        let best_schedule = schedules
            .iter()
            .max_by(|left, right| {
                left.confidence_adjusted_tokens_per_second
                    .total_cmp(&right.confidence_adjusted_tokens_per_second)
            })
            .unwrap();
        let best_kernel = kernels
            .iter()
            .max_by(|left, right| {
                left.confidence_adjusted_tokens_per_second
                    .total_cmp(&right.confidence_adjusted_tokens_per_second)
            })
            .unwrap();
        assert_eq!(best_schedule.value.as_deref(), Some("h-fast"));
        assert_eq!(best_kernel.value.as_deref(), Some("rwkv-state-bwd-wg128"));

        let global = database.ranked_candidates(&geometry());
        assert!(!global.iter().any(|score| {
            score.h_backward_segment_schedule.as_deref() == Some("h-fast")
                && score.h_backward_kernel_geometry.as_deref() == Some("rwkv-state-bwd-wg128")
        }));
    }

    #[test]
    fn backward_segment_schedule_labels_split_into_independent_runtime_factors() {
        let parsed = parse_backward_segment_schedule_factors(
            "rkv-add3-key-reduce+fused-input-mix->weight+low-rank-fused-outer-fan-in",
        )
        .unwrap();
        assert_eq!(parsed.state, "rkv-add3-key-reduce");
        assert_eq!(parsed.projection, "fused-input-mix->weight");
        assert_eq!(
            parsed.low_rank_fan_in.as_deref(),
            Some("low-rank-fused-outer-fan-in")
        );
        assert!(parse_backward_segment_schedule_factors("historical-opaque-arm").is_none());
    }

    #[test]
    fn factorized_segment_subscores_can_compose_an_unmeasured_full_schedule() {
        let states = ["rkv-add3+key+reduce", "rkv-add3-key-reduce"];
        let projections = ["split-input-mix->weight", "fused-input-mix->weight"];
        let fan_ins = ["low-rank-split-fan-in", "low-rank-fused-outer-fan-in"];
        let l_schedule = "rkv-add3+key+reduce+split-input-mix->weight+low-rank-split-fan-in";
        let mut records = Vec::new();
        for state_fast in 0..=1usize {
            for projection_fast in 0..=1usize {
                for fan_in_fast in 0..=1usize {
                    if state_fast == 1 && projection_fast == 1 && fan_in_fast == 1 {
                        continue;
                    }
                    let h_schedule = format!(
                        "{}+{}+{}",
                        states[state_fast], projections[projection_fast], fan_ins[fan_in_fast]
                    );
                    let throughput = 100.0
                        + state_fast as f64 * 120.0
                        + projection_fast as f64 * 90.0
                        + fan_in_fast as f64 * 60.0;
                    records.push(topology_kernel_record(
                        2,
                        2,
                        &h_schedule,
                        l_schedule,
                        "rwkv-state-bwd-wg64",
                        "rwkv-state-bwd-wg64",
                        throughput,
                    ));
                }
            }
        }
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&records.join("\n")).unwrap(),
        };

        let best_value = |scores: Vec<HierarchosTokenTapeBranchFactorScore>| {
            scores
                .into_iter()
                .max_by(|left, right| {
                    left.confidence_adjusted_tokens_per_second
                        .total_cmp(&right.confidence_adjusted_tokens_per_second)
                })
                .and_then(|score| score.value)
                .unwrap()
        };
        let state = best_value(
            database.h_segment_state_exploration_ranked_candidates(&geometry(), "strict-parity"),
        );
        let projection = best_value(
            database
                .h_segment_projection_exploration_ranked_candidates(&geometry(), "strict-parity"),
        );
        let fan_in = best_value(
            database.h_segment_low_rank_fan_in_exploration_ranked_candidates(
                &geometry(),
                "strict-parity",
            ),
        );
        assert_eq!(state, states[1]);
        assert_eq!(projection, projections[1]);
        assert_eq!(fan_in, fan_ins[1]);

        let synthesized = format!("{state}+{projection}+{fan_in}");
        let global = database.ranked_candidates(&geometry());
        assert!(!global.iter().any(|score| {
            score.h_backward_segment_schedule.as_deref() == Some(synthesized.as_str())
        }));
    }

    #[test]
    fn kernel_geometry_variants_are_distinct_policy_arms_while_legacy_records_remain_valid() {
        let text = [
            record("TEST GPU", 2, 2, 100.0),
            kernel_geometry_record(2, 2, "rwkv-state-bwd-wg32", "rwkv-state-bwd-wg64", 230.0),
            kernel_geometry_record(2, 2, "rwkv-state-bwd-wg64", "rwkv-state-bwd-wg128", 190.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 3);
        assert_eq!(
            ranked[0].h_backward_kernel_geometry.as_deref(),
            Some("rwkv-state-bwd-wg32")
        );
        assert_eq!(
            ranked[0].l_backward_kernel_geometry.as_deref(),
            Some("rwkv-state-bwd-wg64")
        );
        assert!(ranked.iter().any(|score| {
            score.h_backward_kernel_geometry.is_none() && score.l_backward_kernel_geometry.is_none()
        }));
    }

    #[test]
    fn pre_fan_in_kernel_geometry_observations_do_not_seed_revised_geometry_policy() {
        let legacy_geometry = record("TEST GPU", 2, 2, 500.0).replace(
            r#""state_checkpoint_stride":2"#,
            r#""state_checkpoint_stride":2,"h_backward_kernel_geometry":"rwkv-state-bwd-wg32","l_backward_kernel_geometry":"rwkv-state-bwd-wg64""#,
        );
        let legacy_geometry_only = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&legacy_geometry).unwrap(),
        };
        assert!(legacy_geometry_only
            .ranked_candidates(&geometry())
            .is_empty());

        let legacy_geometry_plus_agnostic =
            [legacy_geometry, record("TEST GPU", 2, 2, 100.0)].join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&legacy_geometry_plus_agnostic).unwrap(),
        };
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 1);
        assert!(ranked[0].h_backward_kernel_geometry.is_none());
        assert!(ranked[0].l_backward_kernel_geometry.is_none());
    }

    #[test]
    fn numerics_policy_is_a_distinct_arm_and_legacy_records_mean_strict_parity() {
        let legacy_strict = record("TEST GPU", 2, 2, 200.0);
        let fast_subgroup = record("TEST GPU", 2, 2, 250.0).replace(
            r#""state_checkpoint_stride":2"#,
            r#""state_checkpoint_stride":2,"rwkv_numerics_policy":"fast-subgroup""#,
        );
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&[legacy_strict, fast_subgroup].join("\n")).unwrap(),
        };
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].rwkv_numerics_policy, "fast-subgroup");
        assert!(ranked
            .iter()
            .any(|score| score.rwkv_numerics_policy == "strict-parity"));
    }

    #[test]
    fn precision_policy_separates_profile_populations_and_legacy_records_mean_fp32() {
        let legacy_fp32 = record("TEST GPU", 2, 2, 200.0);
        let fp16_storage = record("TEST GPU", 2, 2, 500.0).replace(
            r#""subgroup_size":64"#,
            r#""subgroup_size":64,"training_precision_policy":"fp16-storage-fp32-compute""#,
        );
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&[legacy_fp32, fp16_storage].join("\n")).unwrap(),
        };

        let fp32_ranked = database.ranked_candidates(&geometry());
        assert_eq!(fp32_ranked.len(), 1);
        assert_eq!(fp32_ranked[0].median_tokens_per_second, 200.0);

        let mut fp16_geometry = geometry();
        fp16_geometry.training_precision_policy = "fp16-storage-fp32-compute".to_string();
        let fp16_ranked = database.ranked_candidates(&fp16_geometry);
        assert_eq!(fp16_ranked.len(), 1);
        assert_eq!(fp16_ranked[0].median_tokens_per_second, 500.0);
    }

    #[test]
    fn memory_pressure_context_filters_specific_profiles_and_keeps_legacy_fallback() {
        let text = [
            record("TEST GPU", 2, 2, 90.0),
            pressure_record(2, 2, 2, 210.0),
            pressure_record(2, 2, 6, 500.0),
        ]
        .join("\n");
        let database = HierarchosTokenTapeProfileDatabase {
            source_path: PathBuf::from("test.jsonl"),
            observations: parse_profile_jsonl(&text).unwrap(),
        };
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 2);
        assert!(ranked
            .iter()
            .any(|score| score.device_local_pressure_bucket == Some(2)));
        assert!(ranked
            .iter()
            .any(|score| score.device_local_pressure_bucket.is_none()));
        assert!(!ranked
            .iter()
            .any(|score| score.device_local_pressure_bucket == Some(6)));
    }

    #[test]
    fn online_observation_is_appended_and_immediately_visible() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "hierarchos-vulkan-online-profile-{}-{unique}.jsonl",
            std::process::id()
        ));
        let mut database = HierarchosTokenTapeProfileDatabase {
            source_path: path.clone(),
            observations: Vec::new(),
        };
        let telemetry = HierarchosTokenTapeOnlineTelemetry {
            warmup_windows: 1,
            measured_windows: 4,
            aggregate_tokens_per_second: 218.0,
            mean_tokens_per_second: 220.0,
            sample_variance_tokens_per_second: 16.0,
            relative_standard_deviation: 4.0 / 220.0,
            first_tokens_per_second: 224.0,
            last_tokens_per_second: 216.0,
            timeline_retirement_latency_samples: 8,
            timeline_retirement_latency_ns_average: Some(250.0),
            kernel_timestamp_profile_samples: 4,
            kernel_timestamp_dispatches: 32,
            kernel_gpu_ns_per_token: Some(12.5),
            first_kernel_gpu_ns_per_token: Some(12.0),
            last_kernel_gpu_ns_per_token: Some(13.0),
            initial_device_local_pressure_bucket: Some(2),
            final_device_local_pressure_bucket: Some(2),
            peak_device_local_usage_ratio: Some(0.61),
            throughput_slowdown_ratio: 216.0 / 224.0,
            possible_gpu_throttling: false,
        };

        database
            .append_online_observation(
                &geometry(),
                2,
                2,
                Some("h-fast"),
                Some("l-fast"),
                Some("rwkv-state-bwd-wg32"),
                Some("rwkv-state-bwd-wg64"),
                "strict-parity",
                222.0,
                4,
                Some(&telemetry),
            )
            .unwrap();
        database
            .append_online_observation(
                &geometry(),
                2,
                2,
                Some("h-fast"),
                Some("l-fast"),
                Some("rwkv-state-bwd-wg32"),
                Some("rwkv-state-bwd-wg64"),
                "strict-parity",
                198.0,
                1,
                None,
            )
            .unwrap();
        assert_eq!(database.observation_count(), 2);
        let ranked = database.ranked_candidates(&geometry());
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].sequence_microbatch_size, 2);
        assert_eq!(ranked[0].state_checkpoint_stride, 2);
        assert_eq!(ranked[0].device_local_pressure_bucket, Some(2));
        assert_eq!(
            ranked[0].h_backward_segment_schedule.as_deref(),
            Some("h-fast")
        );
        assert_eq!(
            ranked[0].l_backward_segment_schedule.as_deref(),
            Some("l-fast")
        );
        assert_eq!(
            ranked[0].h_backward_kernel_geometry.as_deref(),
            Some("rwkv-state-bwd-wg32")
        );
        assert_eq!(
            ranked[0].l_backward_kernel_geometry.as_deref(),
            Some("rwkv-state-bwd-wg64")
        );
        assert_eq!(ranked[0].profile_records, 2);
        assert_eq!(ranked[0].measured_iterations, 5);
        assert!(ranked[0].adaptive_tokens_per_second < 222.0);
        assert!(ranked[0].adaptive_tokens_per_second > 198.0);
        let persisted = fs::read_to_string(&path).unwrap();
        let parsed = parse_profile_jsonl(&persisted).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].median_tokens_per_second, 222.0);
        assert_eq!(parsed[1].median_tokens_per_second, 198.0);
        let raw_records = persisted
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            raw_records[0]["profile_key"]["device_uuid"],
            "test-device-uuid"
        );
        assert_eq!(
            raw_records[0]["profile_key"]["driver_uuid"],
            "test-driver-uuid"
        );
        assert_eq!(
            raw_records[0]["result"]["steady_state_telemetry"]["measured_windows"],
            4
        );
        assert_eq!(
            raw_records[0]["result"]["steady_state_telemetry"]
                ["timeline_retirement_latency_ns_average"],
            250.0
        );
        assert_eq!(
            raw_records[0]["result"]["steady_state_telemetry"]["possible_gpu_throttling"],
            false
        );
        let _ = fs::remove_file(path);
    }
}
