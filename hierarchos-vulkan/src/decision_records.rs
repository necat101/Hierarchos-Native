//! Measured per-device kernel-graph decisions, compiled in from checked-in records.
//!
//! Each record is one JSON file under `validation/`, keyed by adapter name,
//! width and batch, naming which complete Vulkan graph won at that geometry. A
//! binary compiles its record in with `include_str!`, so the shipped decision and
//! the checked-in measurement cannot drift, and a hand-edited entry whose route
//! contradicts its own candidate timings fails at build time.
//!
//! The host never computes any of the arithmetic: the decision selects which of
//! the complete GPU shader graphs the default dispatches, and every candidate is
//! measured bit-for-bit against the others before a record entry is written.
//! Recording happens offline (the A/B drivers write the record); a training graph
//! must not time itself on the hot path.
//!
//! # One route, not a flag plus an arm
//!
//! A record entry carries a single `route` naming the graph that won, not a
//! boolean composition flag plus a separate fused-arm field. The key/state stage
//! has three candidates -- the wider-group fusion, the one-workgroup-per-head
//! fusion and the two-dispatch chain -- and "compose" is only one of them, so a
//! boolean could not name the arm without a second field that had to be kept
//! consistent with it. [`Route`] is the closed vocabulary, and the entry's
//! `candidates` map is the measurement that produced it: the recorded route must
//! be the fastest candidate it lists, and the loader rejects an entry that says
//! otherwise.
//!
//! # Parameter-storage policy
//!
//! An entry may carry an optional `parameter_storage` policy. A policy-less entry
//! is the default every policy falls back to; a policy-scoped entry only applies
//! to a consumer that is running that storage policy, including the fused arm it
//! names. The two are looked up separately, so an FP16-storage measurement can
//! name its own winner without moving the FP32 route — the reason the dimension
//! exists is that the low-rank producer's crossover sits at different geometries
//! under the two policies.
//!
//! Policy names are canonicalized to what they actually select: the several
//! `fp16-storage-*` training policies differ in compute, not in storage, so they
//! all read the `fp16-storage` verdict, and `fp16-packed-mirror` (the op-level
//! label the low-rank producer reports) is the same storage policy under an older
//! name.

use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

use serde::Deserialize;

/// The complete graph a measured geometry dispatches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    /// A single fused producer, where the record's producer has exactly one fused
    /// kernel: the time-mix projection, the channel mix, and the rank-2 low-rank
    /// route.
    Fused,
    /// The wider-group fused key/state kernel: `64 / head_size` heads per
    /// workgroup, every lane advancing a row.
    WideFused,
    /// The one-workgroup-per-head fused key/state kernel.
    NarrowFused,
    /// The two-dispatch chain (or, for the projection and channel mix, the
    /// `linear3`/`linear_residual` module composition).
    Staged,
    /// The packed TBPTT chain: recurrent matrix state stays packed through the
    /// first forward sweep (or through reverse rematerialization) and the
    /// backward-only tape stores are suppressed.
    Packed,
    /// The legacy TBPTT chain: unpacked forward state and a full recorded tape.
    Legacy,
    /// The fully folded state-backward rung: the recurrence/vector-add3 kernel, the
    /// key transform and the key-parameter reduction in one dispatch. It is only
    /// recordable where the sweep measured it bit-exact against the two-dispatch
    /// reduction (`grad_k_k`/`grad_k_a` exactly equal), and only where the device
    /// compiles that depth.
    Folded,
}

impl Route {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::WideFused => "wide-fused",
            Self::NarrowFused => "narrow-fused",
            Self::Staged => "staged",
            Self::Packed => "packed",
            Self::Legacy => "legacy",
            Self::Folded => "folded",
        }
    }

    fn parse(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "fused" => Some(Self::Fused),
            "wide-fused" => Some(Self::WideFused),
            "narrow-fused" => Some(Self::NarrowFused),
            "staged" | "composed" => Some(Self::Staged),
            "packed" => Some(Self::Packed),
            "legacy" => Some(Self::Legacy),
            "folded" => Some(Self::Folded),
            _ => None,
        }
    }

    /// Whether the route is the decomposed graph rather than a fused producer.
    pub fn is_staged(self) -> bool {
        matches!(self, Self::Staged)
    }
}

/// The route vocabulary each record kind may use. A closed set rather than a
/// free-form string: the runtime can only dispatch the graphs it builds, so an
/// entry naming a fourth candidate is a typo that must fail at load instead of
/// silently falling back to a default nobody measured.
fn allowed_routes(kind: &str) -> &'static [Route] {
    match kind {
        "rwkv-key-state-forward" => &[Route::WideFused, Route::NarrowFused, Route::Staged],
        "rwkv-tbptt-packed-forward" | "rwkv-tbptt-packed-rematerialization" => {
            &[Route::Packed, Route::Legacy]
        }
        // The backward-fusion table names the autotuner's depth (`fused`), the most
        // decomposed chain (`staged`) and, where the sweep measured it exact, the
        // fully folded rung (`folded`).
        "rwkv-tbptt-backward-fusion" => &[Route::Fused, Route::Staged, Route::Folded],
        _ => &[Route::Fused, Route::Staged],
    }
}

#[derive(Deserialize)]
struct DecisionRecord {
    schema: u32,
    kind: String,
    decisions: Vec<Decision>,
    /// Routes that were measured and won, but whose graph changed the values
    /// enough that a training trajectory cannot adopt them. The entry stays in
    /// `decisions` so the measurement is auditable; the runtime skips it and
    /// keeps the structural default. A scoped entry withholds that policy's
    /// lookup entirely -- including the shared fallback -- so a policy the
    /// measurement was not taken under can refuse it without moving the policy
    /// that was.
    #[serde(default)]
    withheld: Vec<Withheld>,
}

#[derive(Deserialize)]
struct Withheld {
    device: String,
    width: u32,
    batch: u32,
    /// The route the run measured. It must be the fastest candidate in the
    /// decision entry this policy would otherwise adopt (its own scoped entry,
    /// or the shared one it falls back to), so a withheld route cannot be
    /// invented beside the measurement either.
    route: String,
    #[serde(default, rename = "parameter_storage")]
    parameter_storage: Option<String>,
    reason: String,
}

#[derive(Deserialize)]
struct Decision {
    device: String,
    width: u32,
    batch: u32,
    /// The graph that won at this geometry, from the kind's route vocabulary.
    route: String,
    /// Per-candidate median milliseconds. This is the measurement the route
    /// claims, and the loader checks the claim against it.
    candidates: HashMap<String, f64>,
    #[serde(default)]
    source: Option<String>,
    /// The parameter-storage policy this decision was measured under, e.g.
    /// `fp16-storage`. Absent means the default entry, which applies to every
    /// policy that has no entry of its own.
    #[serde(default, rename = "parameter_storage")]
    parameter_storage: Option<String>,
}

/// One compiled-in record: the measured route per (adapter, width, batch), plus
/// the policy-scoped entries that override it for one storage policy.
pub(crate) struct DecisionTable {
    entries: HashMap<(String, u32, u32), Route>,
    scoped: HashMap<(String, u32, u32, String), Route>,
    /// Keys whose measured route is not adopted by the runtime. A shared (no
    /// policy) key withholds the entry from every policy that would otherwise
    /// fall back to it; a scoped key withholds that policy's lookup even when
    /// its only evidence is the shared entry, so the structural default decides
    /// there while other policies keep reading the measurement.
    withheld: HashSet<(String, u32, u32, String)>,
}

impl DecisionTable {
    /// Parse and validate a record. `kind` is checked against the file's own
    /// `kind` so a record cannot be loaded into the wrong consumer.
    pub(crate) fn load(json: &str, kind: &str) -> Self {
        let record: DecisionRecord =
            serde_json::from_str(json).expect("embedded decision record must parse");
        assert_eq!(
            record.schema, 2,
            "unsupported decision schema {} for {kind}",
            record.schema
        );
        assert_eq!(
            record.kind, kind,
            "decision record is {:?}, not the {kind} consumer's record",
            record.kind
        );
        // An empty record is valid and means "no adapter has been measured yet":
        // every device keeps the structural default until the first A/B records
        // an entry. It is never treated as permission to stage anywhere.
        let allowed = allowed_routes(kind);
        let mut entries = HashMap::with_capacity(record.decisions.len());
        let mut scoped = HashMap::new();
        for decision in record.decisions {
            if let Some(source) = decision.source.as_deref() {
                assert!(
                    source == "measured",
                    "decision for {} at width {} batch {} must be measured, not {source:?}",
                    decision.device,
                    decision.width,
                    decision.batch
                );
            }
            let route = Route::parse(&decision.route).unwrap_or_else(|| {
                panic!(
                    "decision for {} at width {} batch {} names route {:?}, which is not one of \
                     {allowed:?}",
                    decision.device, decision.width, decision.batch, decision.route
                )
            });
            assert!(
                allowed.contains(&route),
                "decision for {} at width {} batch {} names route {:?}, which the {kind} consumer \
                 does not dispatch: it builds {allowed:?}",
                decision.device,
                decision.width,
                decision.batch,
                route.label()
            );
            assert!(
                !decision.candidates.is_empty(),
                "decision for {} at width {} batch {} carries no candidate timings",
                decision.device,
                decision.width,
                decision.batch
            );
            let mut measured = Vec::with_capacity(decision.candidates.len());
            for (label, milliseconds) in &decision.candidates {
                let candidate = Route::parse(label).unwrap_or_else(|| {
                    panic!(
                        "decision for {} at width {} batch {} measured candidate {label:?}, which \
                         is not one of {allowed:?}",
                        decision.device, decision.width, decision.batch
                    )
                });
                assert!(
                    allowed.contains(&candidate),
                    "decision for {} at width {} batch {} measured candidate {label:?}, which the \
                     {kind} consumer does not dispatch",
                    decision.device,
                    decision.width,
                    decision.batch
                );
                assert!(
                    milliseconds.is_finite() && *milliseconds > 0.0,
                    "decision for {} at width {} batch {} has a non-positive {label:?} timing \
                     ({milliseconds})",
                    decision.device,
                    decision.width,
                    decision.batch
                );
                measured.push((candidate, *milliseconds));
            }
            let chosen = decision.candidates.get(route.label()).unwrap_or_else(|| {
                panic!(
                    "decision for {} at width {} batch {} names route {:?} but lists no timing for \
                     it",
                    decision.device,
                    decision.width,
                    decision.batch,
                    route.label()
                )
            });
            let fastest = measured
                .iter()
                .map(|(_, milliseconds)| *milliseconds)
                .fold(f64::INFINITY, f64::min);
            assert!(
                *chosen <= fastest,
                "decision for {} at width {} batch {} says {:?} won at {chosen} ms but another \
                 candidate measured {fastest} ms: the route has to be the measurement, not an \
                 assertion beside it",
                decision.device,
                decision.width,
                decision.batch,
                route.label()
            );
            let key = decision_key(&decision.device, decision.width, decision.batch);
            match decision.parameter_storage.as_deref() {
                None => {
                    assert!(
                        entries.insert(key, route).is_none(),
                        "duplicate decision for one device/width/batch in the {kind} record"
                    );
                }
                Some(policy) => {
                    let policy = decision_policy(policy);
                    assert!(
                        scoped
                            .insert((key.0, key.1, key.2, policy.clone()), route)
                            .is_none(),
                        "duplicate {policy:?} decision for one device/width/batch in the {kind} \
                         record"
                    );
                }
            }
        }
        let mut withheld = HashSet::with_capacity(record.withheld.len());
        for entry in record.withheld {
            assert!(
                !entry.reason.trim().is_empty(),
                "withheld decision for {} at width {} batch {} carries no reason",
                entry.device,
                entry.width,
                entry.batch
            );
            let route = Route::parse(&entry.route).unwrap_or_else(|| {
                panic!(
                    "withheld decision for {} at width {} batch {} names route {:?}, which is not \
                     one of {allowed:?}",
                    entry.device, entry.width, entry.batch, entry.route
                )
            });
            assert!(
                allowed.contains(&route),
                "withheld decision for {} at width {} batch {} names route {:?}, which the {kind} \
                 consumer does not dispatch",
                entry.device,
                entry.width,
                entry.batch,
                route.label()
            );
            let key = decision_key(&entry.device, entry.width, entry.batch);
            let policy = entry
                .parameter_storage
                .as_deref()
                .map(decision_policy)
                .unwrap_or_default();
            // The route a policy would otherwise adopt: its own entry when one
            // was measured under it, else the shared verdict it falls back to.
            // Withholding the shared fallback for one policy is the case a
            // frozen trajectory creates -- the measurement is real and stays in
            // `decisions`, but that policy may not move to it.
            let measured = match entry.parameter_storage.as_deref() {
                Some(_) => scoped
                    .get(&(key.0.clone(), key.1, key.2, policy.clone()))
                    .or_else(|| entries.get(&key)),
                None => entries.get(&key),
            };
            assert_eq!(
                measured.copied(),
                Some(route),
                "withheld decision for {} at width {} batch {} names {:?}, but neither its \
                 scoped entry nor the shared fallback does; a withheld route has to be the \
                 route this policy would otherwise adopt",
                entry.device,
                entry.width,
                entry.batch,
                route.label()
            );
            assert!(
                withheld.insert((key.0, key.1, key.2, policy)),
                "duplicate withheld decision for one device/width/batch in the {kind} record"
            );
        }
        Self {
            entries,
            scoped,
            withheld,
        }
    }

    /// The measured route for this adapter and geometry under the policy-less
    /// read, if one was recorded. `None` means no measurement exists, so the
    /// caller falls back to its own structural default.
    ///
    /// Production consumers always name their parameter-storage policy through
    /// [`Self::route_scoped`], because a scoped refusal must be visible to them;
    /// this view exists for the loader's own tests.
    #[cfg(test)]
    pub(crate) fn route(&self, device: &str, width: u32, batch: u32) -> Option<Route> {
        let key = decision_key(device, width, batch);
        if self
            .withheld
            .contains(&(key.0.clone(), key.1, key.2, String::new()))
        {
            return None;
        }
        self.entries.get(&key).copied()
    }

    /// The measured route for this adapter, geometry and parameter-storage
    /// policy: a policy-scoped entry wins where one exists, and the policy-less
    /// default applies otherwise. That fallback is what keeps a policy measured
    /// only at some geometries using the shared verdict at the rest instead of
    /// silently reverting to the structural default.
    pub(crate) fn route_scoped(
        &self,
        device: &str,
        width: u32,
        batch: u32,
        policy: &str,
    ) -> Option<Route> {
        let key = decision_key(device, width, batch);
        let scoped_key = (key.0.clone(), key.1, key.2, decision_policy(policy));
        // A withheld scoped route does not fall back to the shared verdict: the
        // reason it was withheld (a numerical difference that the trajectory
        // rejects) applies to any route that changes the graph, and the shared
        // entry is a different policy's evidence. A scoped withhold with no
        // scoped entry is exactly that case -- it names the shared verdict it
        // refuses, and only this policy stops reading it.
        if self.withheld.contains(&scoped_key) {
            return None;
        }
        if let Some(route) = self.scoped.get(&scoped_key) {
            return Some(*route);
        }
        if self
            .withheld
            .contains(&(key.0.clone(), key.1, key.2, String::new()))
        {
            return None;
        }
        self.entries.get(&key).copied()
    }

    /// Whether any verdict in this table names `route` for this adapter and width,
    /// at any batch or storage policy.
    ///
    /// A consumer uses this to answer "can a record select this graph here?" before
    /// paying to build kernels nothing may dispatch. It is deliberately not a
    /// per-geometry lookup: the question is whether the route is selectable at all
    /// on this adapter, which is settled before any batch is known.
    pub(crate) fn any_route(&self, device: &str, width: u32, route: Route) -> bool {
        let device = device.trim().to_ascii_lowercase();
        let shared = self
            .entries
            .iter()
            .any(|((name, entry_width, _), measured)| {
                *name == device && *entry_width == width && *measured == route
            });
        let scoped = self
            .scoped
            .iter()
            .any(|((name, entry_width, _, _), measured)| {
                *name == device && *entry_width == width && *measured == route
            });
        shared || scoped
    }

    /// Number of recorded decisions, for coverage tests that must fail when the
    /// record is emptied or truncated.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len() + self.scoped.len()
    }

    /// Policy-scoped entries only, for tests that check the dimension is used
    /// rather than merely accepted by the parser.
    #[cfg(test)]
    pub(crate) fn scoped_len(&self) -> usize {
        self.scoped.len()
    }
}

fn decision_key(device: &str, width: u32, batch: u32) -> (String, u32, u32) {
    // Driver names differ only in surrounding whitespace and case across the
    // backends a record can come from.
    (device.trim().to_ascii_lowercase(), width, batch)
}

/// Canonicalize a storage-policy name to the storage it selects.
///
/// The training policies `fp16-storage-fp32-compute`, `fp16-storage-parity` and
/// `fp16-storage-fp16-lm-backward` install the same packed mirrors and differ only
/// in compute, and the low-rank op reports `fp16-packed-mirror` for the mirrors it
/// was handed. A route measured under any of them is a measurement of FP16
/// parameter storage, so they share one verdict; recording them as four separate
/// scopes would claim a resolution the storage decision does not have.
fn decision_policy(policy: &str) -> String {
    let policy = policy.trim().to_ascii_lowercase();
    if policy.starts_with("fp16") {
        "fp16-storage".to_owned()
    } else if policy.starts_with("bf16") {
        "bf16-storage".to_owned()
    } else {
        policy
    }
}

/// The composition rule, factored out so its cases are testable without a GPU:
/// the fused graph is forced when the A/B switch says so; the composition only
/// ever runs where the tuned module is admitted; and within that, a recorded
/// measurement decides, with "compose" as the default for an unmeasured device.
pub(crate) fn prefers_composition(forced: bool, admitted: bool, measured: Option<Route>) -> bool {
    !forced && admitted && measured.map(Route::is_staged).unwrap_or(true)
}

/// The rank-2 route rule, factored out for the same reason: the fused LN1/low-rank
/// producer wins wherever its kernels exist, unless a measurement recorded a staged
/// win for this geometry or the A/B forced the staged chain. Unlike
/// [`prefers_composition`] there is no vendor gate in front of it — the producer is
/// plain Vulkan code, so "a kernel exists" is the whole availability question, and
/// the structural default is the fusion rather than the decomposition.
pub(crate) fn prefers_fused_forward(
    forced_staged: bool,
    available: bool,
    measured: Option<Route>,
) -> bool {
    available && !forced_staged && !measured.map(Route::is_staged).unwrap_or(false)
}

/// Lazily loaded tables: one per consumer, so an unused record is never parsed.
pub(crate) fn table(
    lock: &'static OnceLock<DecisionTable>,
    json: &'static str,
    kind: &str,
) -> &'static DecisionTable {
    lock.get_or_init(|| DecisionTable::load(json, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIME_MIX: &str = include_str!("../validation/time_mix_projection_decisions.json");
    const CHANNEL_MIX: &str = include_str!("../validation/channel_mix_forward_decisions.json");
    const LOW_RANK_FULL_FORWARD: &str =
        include_str!("../validation/low_rank_full_forward_decisions.json");
    const KEY_STATE_FORWARD: &str = include_str!("../validation/key_state_forward_decisions.json");
    const TBPTT_PACKED_FORWARD: &str =
        include_str!("../validation/tbptt_packed_forward_decisions.json");
    const TBPTT_PACKED_REMATERIALIZATION: &str =
        include_str!("../validation/tbptt_packed_rematerialization_decisions.json");
    const TBPTT_BACKWARD_FUSION: &str =
        include_str!("../validation/tbptt_backward_fusion_decisions.json");

    #[test]
    fn rule_prefers_the_record_over_the_gate() {
        // A recorded fused win vetoes a geometry the gate would admit.
        assert!(!prefers_composition(false, true, Some(Route::Fused)));
        // A recorded staged win keeps the composition.
        assert!(prefers_composition(false, true, Some(Route::Staged)));
        // An unmeasured device inherits the shipped default.
        assert!(prefers_composition(false, true, None));
        // The composition only wins as a tuned module, so a recorded decision
        // cannot compose where the module would be portable.
        assert!(!prefers_composition(false, false, Some(Route::Staged)));
        // The A/B switch still forces the fused graph everywhere.
        assert!(!prefers_composition(true, true, Some(Route::Staged)));
    }

    #[test]
    fn record_parses_and_is_non_empty() {
        let table = DecisionTable::load(TIME_MIX, "rwkv-time-mix-projection");
        assert!(
            table.len() >= 2,
            "the embedded record must carry measured decisions"
        );
    }

    #[test]
    fn lookup_is_exact_on_device_and_geometry() {
        let table = DecisionTable::load(TIME_MIX, "rwkv-time-mix-projection");
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 8),
            Some(Route::Staged)
        );
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 4),
            None,
            "a geometry measured inside the noise band has its entry withdrawn and the gate \
             decides it, so it must not read as a recorded fused win"
        );
        assert_eq!(
            table.route("  intel(r) hd graphics 520 ", 128, 8),
            Some(Route::Staged),
            "device names are compared case-insensitively and trimmed"
        );
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 256, 8),
            None,
            "a width that was never measured must not borrow another width's decision"
        );
        assert_eq!(
            table.route("Some Other GPU", 128, 8),
            None,
            "an unrecorded adapter must fall back to the vendor gate"
        );
    }

    /// One table implementation serves every producer, and the rule each record
    /// documents is the same: a geometry measured outside the band carries the graph
    /// that won there, and one measured inside it carries nothing at all.
    #[test]
    fn every_producer_record_shares_one_loader_and_one_band_rule() {
        let time_mix = DecisionTable::load(TIME_MIX, "rwkv-time-mix-projection");
        let channel_mix = DecisionTable::load(CHANNEL_MIX, "rwkv-channel-mix-forward");
        assert!(
            time_mix.len() >= 2 && channel_mix.len() >= 2,
            "both records must carry measured decisions"
        );
        for (name, table) in [("time-mix", &time_mix), ("channel-mix", &channel_mix)] {
            assert_eq!(
                table.route("Intel(R) HD Graphics 520", 128, 8),
                Some(Route::Staged),
                "{name}: batch 8 of width 128 is a recorded composed win"
            );
            assert_eq!(
                table.route("Intel(R) HD Graphics 520", 128, 4),
                None,
                "{name}: batch 4 measures inside the noise band, so it has no decision"
            );
        }
    }

    #[test]
    fn rank_two_rule_keeps_the_fusion_unless_measured_or_forced() {
        // The structural default: the producer exists, so it wins.
        assert!(prefers_fused_forward(false, true, None));
        // A recorded staged win withdraws it for that geometry only.
        assert!(!prefers_fused_forward(false, true, Some(Route::Staged)));
        // A recorded fused win keeps the fusion.
        assert!(prefers_fused_forward(false, true, Some(Route::Fused)));
        // No producer on this device/rank shape: the chain is the only graph.
        assert!(!prefers_fused_forward(false, false, None));
        assert!(!prefers_fused_forward(false, false, Some(Route::Fused)));
        // The A/B switch still pins the staged chain everywhere.
        assert!(!prefers_fused_forward(true, true, None));
        assert!(!prefers_fused_forward(true, true, Some(Route::Fused)));
    }

    #[test]
    fn low_rank_forward_record_loads_into_its_own_consumer() {
        let table = DecisionTable::load(LOW_RANK_FULL_FORWARD, "rwkv-low-rank-full-forward");
        assert_eq!(
            table.route("Some Other GPU", 128, 8),
            None,
            "the record must never name a decision for an adapter it did not measure"
        );
        let wrong = std::panic::catch_unwind(|| {
            DecisionTable::load(LOW_RANK_FULL_FORWARD, "rwkv-channel-mix-forward");
        });
        assert!(
            wrong.is_err(),
            "a record loaded into the wrong consumer must fail"
        );
    }

    /// The three-way dimension: the wider-group fusion, the narrow fusion it
    /// replaced, and the chain are all routes of one record, so a geometry can
    /// carry the fusion *and* say which fused kernel it means in the same field.
    /// The wider group now wins every measured geometry, batch 32 included: the
    /// all-lane transform removed the batch-16 crossover, and staging the state
    /// phase's channel-invariant operands removed the batch-32 collapse, so no
    /// entry names the chain or the narrow arm on this adapter.
    #[test]
    fn key_state_record_names_the_route_that_won_each_geometry() {
        let table = DecisionTable::load(KEY_STATE_FORWARD, "rwkv-key-state-forward");
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 1),
            Some(Route::WideFused),
            "the wider group wins batch 1 by 12% once its state phase stages the per-channel \
             operands, so this is a measured decision rather than the old tie"
        );
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 4),
            Some(Route::WideFused),
            "the wider group is the fastest fused candidate here; the narrow arm no longer wins a \
             single measured geometry"
        );
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 8),
            Some(Route::WideFused),
            "the wider group wins the middle of the range as well"
        );
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 16),
            Some(Route::WideFused),
            "batch 16 was the crossover: the wider group removed it, so the record names the wide route rather than the chain"
        );
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 32),
            Some(Route::WideFused),
            "batch 32 was the state-phase collapse: with the operands staged the fused route leads \
             the chain, so the record no longer stages this geometry"
        );
        assert_eq!(
            table.route("Some Other GPU", 128, 8),
            None,
            "an unmeasured adapter keeps the shipped wider-group route"
        );
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 256),
            None,
            "a route recorded for one width must not be borrowed by another"
        );
        assert!(table.route("Intel(R) HD Graphics 520", 128, 32).is_some());
    }

    /// The backward-fusion ladder's third rung. The shipped record documents the
    /// folded rung as a measured candidate at every geometry it keeps, but names
    /// it as the winner nowhere: on this adapter the three depths sit inside the 2%
    /// band, so no measurement justified moving off the autotuner's schedule and
    /// the fold's kernels are not compiled into the shipped graph.
    #[test]
    fn the_backward_fusion_record_measures_the_folded_rung_without_adopting_it() {
        let table = DecisionTable::load(TBPTT_BACKWARD_FUSION, "rwkv-tbptt-backward-fusion");
        assert!(table.len() >= 1, "the record has to carry its measurements");
        assert_eq!(
            table.route("Intel(R) HD Graphics 520", 128, 4),
            Some(Route::Fused),
            "batch 4 keeps the autotuner's depth"
        );
        assert!(
            !table.any_route("Intel(R) HD Graphics 520", 128, Route::Folded),
            "the folded rung is a candidate everywhere but wins nowhere, so the cell does not \
             compile a depth nothing dispatches"
        );
        assert!(
            !table.any_route("Some Other GPU", 128, Route::Folded),
            "an unmeasured adapter has no folded verdict at all"
        );
    }

    /// The folded route is recordable: an entry that chose it is admitted, reads
    /// back for its own policy, and is visible to the cell's "can anything select
    /// this depth" question. It is still only a backward-fusion route -- a chain
    /// record that builds two graphs may not name it.
    #[test]
    fn a_folded_entry_is_loadable_and_selectable() {
        let json = r#"{"schema":2,"kind":"rwkv-tbptt-backward-fusion","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"folded",
             "candidates":{"staged":1.02,"fused":1.0,"folded":0.97}}]}"#;
        let table = DecisionTable::load(json, "rwkv-tbptt-backward-fusion");
        assert_eq!(
            table.route_scoped("GPU", 128, 8, "fp32"),
            Some(Route::Folded)
        );
        assert!(table.any_route("GPU", 128, Route::Folded));
        assert_eq!(table.route_scoped("GPU", 128, 16, "fp32"), None);

        let packed_forward = r#"{"schema":2,"kind":"rwkv-tbptt-packed-forward","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"folded",
             "candidates":{"packed":1.0,"legacy":1.02,"folded":0.97}}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(packed_forward, "rwkv-tbptt-packed-forward");
        });
        assert!(
            result.is_err(),
            "the packed-forward chain builds two graphs, so a folded route there is a typo"
        );
    }

    /// The closed vocabulary is what removes the old flag-plus-arm pair: a route
    /// the kind does not dispatch fails at load rather than being ignored.
    #[test]
    fn a_route_outside_the_kind_vocabulary_is_rejected() {
        let json = r#"{"schema":2,"kind":"rwkv-key-state-forward","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"fused",
             "candidates":{"fused":1.0,"staged":1.2}}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-key-state-forward");
        });
        assert!(
            result.is_err(),
            "the key/state consumer builds two fused kernels, so a bare `fused` route cannot say \
             which one ran"
        );
    }

    /// An entry is evidence, so its route has to match its own candidate timings.
    #[test]
    fn a_route_that_is_not_the_fastest_candidate_is_rejected() {
        let json = r#"{"schema":2,"kind":"rwkv-key-state-forward","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"staged",
             "candidates":{"wide-fused":1.0,"narrow-fused":1.1,"staged":1.2}}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-key-state-forward");
        });
        assert!(
            result.is_err(),
            "a recorded staged win whose own timings put the fusion 20% ahead is hand-edited \
             evidence"
        );
    }

    #[test]
    fn a_route_with_no_timing_of_its_own_is_rejected() {
        let json = r#"{"schema":2,"kind":"rwkv-time-mix-projection","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"staged",
             "candidates":{"fused":1.0}}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-time-mix-projection");
        });
        assert!(result.is_err());
    }

    #[test]
    fn the_low_rank_record_mixes_candidate_sets_by_geometry() {
        // The FP16 entries of that record are two-way (fused vs staged) while the
        // whole record is two-way too: a candidate key the consumer does not build
        // is the same class of error as an unknown route.
        let json = r#"{"schema":2,"kind":"rwkv-low-rank-full-forward","decisions":[
            {"device":"GPU","width":128,"batch":16,"route":"fused",
             "candidates":{"fused":1.0,"wide-fused":0.9}}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-low-rank-full-forward");
        });
        assert!(result.is_err());
    }

    /// The storage-policy dimension: an entry scoped to one policy wins for that
    /// policy and nowhere else, and the policy-less entry stays the fallback the
    /// other policies keep reading.
    #[test]
    fn policy_scoped_entries_override_only_their_own_policy() {
        let json = r#"{"schema":2,"kind":"rwkv-low-rank-full-forward","decisions":[
            {"device":"GPU","width":128,"batch":16,"route":"fused",
             "candidates":{"fused":1.0,"staged":1.031}},
            {"device":"GPU","width":128,"batch":16,"route":"staged",
             "candidates":{"fused":1.0,"staged":0.976},
             "parameter_storage":"fp16-storage"},
            {"device":"GPU","width":128,"batch":32,"route":"staged",
             "candidates":{"fused":1.0,"staged":0.961},
             "parameter_storage":"FP16-Packed-Mirror "}]}"#;
        let table = DecisionTable::load(json, "rwkv-low-rank-full-forward");
        assert_eq!(
            table.route("GPU", 128, 16),
            Some(Route::Fused),
            "the policy-less lookup reads the shared verdict, not the scoped one"
        );
        assert_eq!(
            table.route("GPU", 128, 32),
            None,
            "a geometry the shared table never measured has no shared entry"
        );
        assert_eq!(
            table.route_scoped("GPU", 128, 16, "fp16-storage"),
            Some(Route::Staged),
            "the policy-scoped entry wins for the policy it was measured under"
        );
        assert_eq!(
            table.route_scoped("GPU", 128, 16, "fp32"),
            Some(Route::Fused),
            "another policy still reads the shared verdict"
        );
        assert_eq!(
            table.route_scoped("GPU", 128, 32, "fp16-storage-fp32-compute"),
            Some(Route::Staged),
            "every fp16-storage training policy shares one storage verdict, and the older \
             `fp16-packed-mirror` label canonicalizes into it"
        );
        assert_eq!(
            table.route_scoped("GPU", 128, 32, "fp32"),
            None,
            "a policy with no entry and no shared geometry keeps the structural default"
        );
        assert_eq!(table.scoped_len(), 2);
    }

    /// The shipped record has to carry the FP16 route this dimension was added for:
    /// the deep packed-mirror producer measured the chain a hair faster where the
    /// dense route keeps the fusion, but that measurement is not bit-identical, so
    /// it is recorded and withheld and the dense route must not have moved.
    #[test]
    fn the_fp16_route_reads_its_own_entries_and_the_dense_route_does_not() {
        let table = DecisionTable::load(LOW_RANK_FULL_FORWARD, "rwkv-low-rank-full-forward");
        assert!(
            table.scoped_len() >= 2,
            "the record carries no policy-scoped entry, so the dimension is unused"
        );
        assert_eq!(
            table.route_scoped("Intel(R) HD Graphics 520", 128, 16, "fp16-storage"),
            None,
            "the FP16 staged win at batch 16 is measured but withheld, so the runtime \
             keeps the structural fused graph"
        );
        assert_eq!(
            table.route_scoped("Intel(R) HD Graphics 520", 128, 8, "fp16-storage"),
            None,
            "a withheld scoped route does not fall back to the shared FP32 verdict"
        );
        assert_eq!(
            table.route_scoped("Intel(R) HD Graphics 520", 128, 4, "fp32"),
            Some(Route::Fused),
            "a withheld FP16 route does not withhold another policy's own verdict"
        );
        assert_eq!(
            table.route_scoped("Intel(R) HD Graphics 520", 128, 16, "fp32"),
            None,
            "the dense route measured nothing at batch 16, so it keeps the structural default"
        );
        assert_eq!(
            table.route_scoped("Intel(R) HD Graphics 520", 128, 4, "fp16-storage"),
            Some(Route::Fused),
            "the FP16 route keeps the fusion where it measured the fusion faster"
        );
        assert_eq!(
            table.route_scoped("Intel(R) HD Graphics 520", 128, 4, "fp32"),
            Some(Route::Fused),
            "the dense route's own batch-4 verdict is unchanged"
        );
    }

    /// The storage dimension is not the low-rank producer's private idea: every
    /// record the four producers compile in reads a scoped entry, and every one of
    /// them names its own route vocabulary.
    #[test]
    fn every_record_accepts_a_policy_scope() {
        let table = DecisionTable::load(KEY_STATE_FORWARD, "rwkv-key-state-forward");
        assert_eq!(
            table.route_scoped("Some Other GPU", 128, 8, "fp16-storage"),
            None,
            "a scoped lookup on an unmeasured adapter still has no entry"
        );
        let scoped = r#"{"schema":2,"kind":"rwkv-key-state-forward","decisions":[
            {"device":"GPU","width":128,"batch":16,"route":"narrow-fused",
             "candidates":{"wide-fused":1.04,"narrow-fused":0.99,"staged":1.03},
             "parameter_storage":"fp16-storage"}]}"#;
        let table = DecisionTable::load(scoped, "rwkv-key-state-forward");
        assert_eq!(
            table.route_scoped("GPU", 128, 16, "fp16-storage"),
            Some(Route::NarrowFused),
            "a key/state entry can scope the fused arm itself, not only the route"
        );
        assert_eq!(
            table.route_scoped("GPU", 128, 16, "fp32"),
            None,
            "the scoped arm must not leak into the dense route"
        );
        assert_eq!(table.route("GPU", 128, 16), None);
    }

    #[test]
    fn key_state_forward_record_loads_into_its_own_consumer() {
        let table = DecisionTable::load(KEY_STATE_FORWARD, "rwkv-key-state-forward");
        assert_eq!(
            table.route("Some Other GPU", 128, 8),
            None,
            "the record must never name a decision for an adapter it did not measure"
        );
        let wrong = std::panic::catch_unwind(|| {
            DecisionTable::load(KEY_STATE_FORWARD, "rwkv-low-rank-full-forward");
        });
        assert!(
            wrong.is_err(),
            "a key/state record loaded into the low-rank consumer must fail"
        );
    }

    /// The two TBPTT chains use their own vocabulary: `packed` and `legacy`.
    /// Batch 8 carries no first-forward entry (the legacy graph led inside the
    /// 2% band, so the run withdrew it), while batches 16 and 32 record the
    /// legacy transition as the winner. Rematerialization keeps legacy where it
    /// was measured and leaves the structural default elsewhere.
    #[test]
    fn tbptt_chain_records_name_packed_or_legacy_per_geometry() {
        let forward = DecisionTable::load(TBPTT_PACKED_FORWARD, "rwkv-tbptt-packed-forward");
        assert_eq!(
            forward.route("Intel(R) HD Graphics 520", 128, 16),
            Some(Route::Legacy),
            "the legacy first-forward transition led batch 16 outside the noise band"
        );
        assert_eq!(
            forward.route("Intel(R) HD Graphics 520", 128, 32),
            Some(Route::Legacy)
        );
        assert_eq!(
            forward.route("Intel(R) HD Graphics 520", 128, 8),
            None,
            "batch 8 was inside the band, so it has no entry and the packed default decides it"
        );
        assert_eq!(forward.route("Some Other GPU", 128, 16), None);

        let remat = DecisionTable::load(
            TBPTT_PACKED_REMATERIALIZATION,
            "rwkv-tbptt-packed-rematerialization",
        );
        assert_eq!(
            remat.route("Intel(R) HD Graphics 520", 128, 8),
            Some(Route::Legacy),
            "packed reverse rematerialization lost every measured batch outside the band"
        );
        assert_eq!(remat.route("Intel(R) HD Graphics 520", 128, 1), None);
    }

    /// A policy reads the shared measurement, and a policy that measured the
    /// transition itself reads its own entry: the FP16 chain A/B removed the
    /// refusal this record used to carry by recording the same crossover under
    /// `fp16-storage`, so both policies now follow their own receipt.
    #[test]
    fn a_chain_policy_reads_the_measurement_it_is_entitled_to() {
        let forward = DecisionTable::load(TBPTT_PACKED_FORWARD, "rwkv-tbptt-packed-forward");
        for batch in [16, 32] {
            assert_eq!(
                forward.route("Intel(R) HD Graphics 520", 128, batch),
                Some(Route::Legacy),
                "the default policy's measured verdict stays recorded at batch {batch}"
            );
            assert_eq!(
                forward.route_scoped("Intel(R) HD Graphics 520", 128, batch, "fp16-storage"),
                Some(Route::Legacy),
                "the FP16 policy measured this transition itself at batch {batch}, so it \
                 adopts its own verdict instead of falling back to the structural route"
            );
            assert_eq!(
                forward.route_scoped("Intel(R) HD Graphics 520", 128, batch, "fp32"),
                Some(Route::Legacy),
                "another policy still reads the shared measurement"
            );
        }
        assert_eq!(
            forward.route_scoped("Intel(R) HD Graphics 520", 128, 1, "fp16-storage"),
            Some(Route::Packed),
            "a geometry whose measured verdict is the structural route needs no refusal"
        );
        let remat = DecisionTable::load(
            TBPTT_PACKED_REMATERIALIZATION,
            "rwkv-tbptt-packed-rematerialization",
        );
        assert_eq!(
            remat.route_scoped("Intel(R) HD Graphics 520", 128, 8, "fp16-storage"),
            Some(Route::Legacy),
            "rematerialization's measured verdicts are the structural default, so no policy \
             refuses anything there"
        );
    }

    /// A scoped withhold has to name the route its policy would otherwise adopt:
    /// either its own scoped entry or the shared fallback. A record that
    /// withholds a route the policy never measured is fabricating a refusal.
    #[test]
    fn a_scoped_withhold_must_name_a_route_the_policy_has() {
        let json = r#"{"schema":2,"kind":"rwkv-tbptt-packed-forward","decisions":[
            {"device":"GPU","width":128,"batch":16,"route":"legacy",
             "candidates":{"packed":1.2,"legacy":1.0}}],
            "withheld":[{"device":"GPU","width":128,"batch":16,"route":"legacy",
             "parameter_storage":"fp16-storage","reason":"the golden is packed"}]}"#;
        let table = DecisionTable::load(json, "rwkv-tbptt-packed-forward");
        assert_eq!(table.route_scoped("GPU", 128, 16, "fp16-storage"), None);
        assert_eq!(
            table.route_scoped("GPU", 128, 16, "fp32"),
            Some(Route::Legacy),
            "withholding the fallback for one policy must not withdraw the shared entry"
        );

        let json = r#"{"schema":2,"kind":"rwkv-tbptt-packed-forward","decisions":[
            {"device":"GPU","width":128,"batch":16,"route":"legacy",
             "candidates":{"packed":1.2,"legacy":1.0}}],
            "withheld":[{"device":"GPU","width":128,"batch":16,"route":"packed",
             "parameter_storage":"fp16-storage","reason":"the golden is packed"}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-tbptt-packed-forward");
        });
        assert!(
            result.is_err(),
            "a scoped withhold that names a route the policy does not have must be rejected"
        );
    }

    #[test]
    fn a_projection_route_is_rejected_by_a_chain_record() {
        let json = r#"{"schema":2,"kind":"rwkv-tbptt-packed-forward","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"fused",
             "candidates":{"packed":1.0,"legacy":1.2}}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-tbptt-packed-forward");
        });
        assert!(
            result.is_err(),
            "a TBPTT chain record can only name the graphs that chain builds"
        );
    }

    /// A withheld route is claimed beside a measurement, so it has to be that
    /// measurement: a record that withholds a route the decision did not name is
    /// fabricating a verdict, and one with no reason gives the runtime no record
    /// of why the faster route was not adopted.
    #[test]
    fn a_withheld_route_must_match_its_measured_decision() {
        let json = r#"{"schema":2,"kind":"rwkv-low-rank-full-forward","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"staged",
             "candidates":{"fused":1.0,"staged":0.9}}],
            "withheld":[{"device":"GPU","width":128,"batch":8,"route":"fused",
             "reason":"not bit-identical"}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-low-rank-full-forward");
        });
        assert!(
            result.is_err(),
            "a withheld route that is not the measured decision must be rejected"
        );

        let json = r#"{"schema":2,"kind":"rwkv-low-rank-full-forward","decisions":[
            {"device":"GPU","width":128,"batch":8,"route":"staged",
             "candidates":{"fused":1.0,"staged":0.9}}],
            "withheld":[{"device":"GPU","width":128,"batch":8,"route":"staged",
             "reason":" "}]}"#;
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(json, "rwkv-low-rank-full-forward");
        });
        assert!(
            result.is_err(),
            "a withheld route with no reason must be rejected, not adopted silently"
        );
    }

    #[test]
    fn wrong_kind_is_rejected() {
        let result = std::panic::catch_unwind(|| {
            DecisionTable::load(TIME_MIX, "rwkv-channel-mix-forward");
        });
        assert!(
            result.is_err(),
            "a record loaded into the wrong consumer must fail"
        );
    }
}
