//! Measured per-device crossover for the packed-cell time-mix projection.
//!
//! The promotion rule used to be static: compose wherever the tuned
//! `linear3-forward` module is admitted. That rule was measured on one adapter,
//! so its crossover (batch 8 here) was that host's, not every GPU's. This module
//! carries the measured decision table, keyed by adapter name, width and batch,
//! and `RwkvTimeMixCoreOp::prefer_composed_projection` consults it before falling
//! back to the vendor gate.
//!
//! The table is compiled in from
//! `validation/time_mix_projection_decisions.json`, so the decision a binary
//! ships is the decision the checked-in record names -- there is no path lookup
//! at runtime and no way for the two to drift. Recording is the offline
//! measurement step (`bench_rwkv_packed_cell_time_mix.py --record-decisions`),
//! mirroring how the LN1 low-rank topology decisions are published, because a
//! training graph must not time itself on the hot path.
//!
//! A device with no entry keeps the conservative default: compose only where its
//! own tuned module is admitted. The measured data behind that default says the
//! composition is the module's win, never the split's, so a device without a
//! tuned module has nothing to gain from composing.

use std::{collections::HashMap, sync::OnceLock};

use serde::Deserialize;

/// The checked-in record this binary ships. `include_str!` makes the build fail
/// if the table is removed, and makes Cargo rebuild when it moves.
const DECISIONS_JSON: &str = include_str!("../validation/time_mix_projection_decisions.json");

#[derive(Deserialize)]
struct DecisionRecord {
    schema: u32,
    decisions: Vec<Decision>,
}

#[derive(Deserialize)]
struct Decision {
    device: String,
    width: u32,
    batch: u32,
    compose: bool,
    #[serde(default)]
    composed_over_fused: Option<f64>,
    #[serde(default)]
    source: Option<String>,
}

fn decisions() -> &'static HashMap<(String, u32, u32), Decision> {
    static TABLE: OnceLock<HashMap<(String, u32, u32), Decision>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let record: DecisionRecord = serde_json::from_str(DECISIONS_JSON)
            .expect("embedded time-mix projection decision record must parse");
        assert_eq!(
            record.schema, 1,
            "unsupported time-mix projection decision schema {}",
            record.schema
        );
        let mut table = HashMap::with_capacity(record.decisions.len());
        for decision in record.decisions {
            if let Some(source) = decision.source.as_deref() {
                assert!(
                    source == "measured",
                    "time-mix projection decision for {} at width {} batch {} must be measured, \
                     not {source:?}",
                    decision.device,
                    decision.width,
                    decision.batch
                );
            }
            if let Some(ratio) = decision.composed_over_fused {
                // A recorded entry has to agree with its own measurement, or the
                // table is hand-edited evidence rather than measured evidence.
                assert!(
                    decision.compose == (ratio < 1.0),
                    "time-mix projection decision for {} at width {} batch {} says compose={} but \
                     its recorded ratio is {ratio}",
                    decision.device,
                    decision.width,
                    decision.batch,
                    decision.compose
                );
            }
            let key = decision_key(&decision.device, decision.width, decision.batch);
            assert!(
                table.insert(key, decision).is_none(),
                "duplicate time-mix projection decision for one device/width/batch"
            );
        }
        table
    })
}

fn decision_key(device: &str, width: u32, batch: u32) -> (String, u32, u32) {
    // Driver names differ only in surrounding whitespace and case across the
    // backends this record can come from.
    (device.trim().to_ascii_lowercase(), width, batch)
}

/// The measured decision for this adapter and geometry, if one was recorded.
/// `Some(true)` means the composition won; `Some(false)` means the fused kernel
/// did. `None` means no measurement exists for this (device, width, batch), so
/// the caller must fall back to the vendor gate.
pub(crate) fn measured_compose_decision(device: &str, width: u32, batch: u32) -> Option<bool> {
    decisions()
        .get(&decision_key(device, width, batch))
        .map(|decision| decision.compose)
}

/// The rule itself, factored out so its four cases are testable without a GPU:
/// the fused kernel is forced when the A/B switch says so; the composition only
/// ever runs where the tuned module is admitted; and within that, a recorded
/// measurement decides, with "compose" as the default for an unmeasured device.
pub(crate) fn prefers_composition(forced: bool, admitted: bool, measured: Option<bool>) -> bool {
    !forced && admitted && measured.unwrap_or(true)
}

/// Number of recorded decisions, for coverage tests that must fail when the
/// table is emptied or truncated.
#[cfg(test)]
pub(crate) fn recorded_decision_count() -> usize {
    decisions().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_parses_and_is_non_empty() {
        assert!(
            recorded_decision_count() >= 2,
            "the embedded time-mix projection record must carry measured decisions"
        );
    }

    #[test]
    fn rule_prefers_the_record_over_the_gate() {
        // A recorded fused win vetoes a geometry the gate would admit.
        assert!(!prefers_composition(false, true, Some(false)));
        // A recorded composed win keeps the composition.
        assert!(prefers_composition(false, true, Some(true)));
        // An unmeasured device inherits the shipped default.
        assert!(prefers_composition(false, true, None));
        // The composition only wins as a tuned module, so a recorded decision
        // cannot compose where the module would be portable.
        assert!(!prefers_composition(false, false, Some(true)));
        // The A/B switch still forces the fused kernel everywhere.
        assert!(!prefers_composition(true, true, Some(true)));
    }

    #[test]
    fn lookup_is_exact_on_device_and_geometry() {
        let compose = measured_compose_decision("Intel(R) HD Graphics 520", 128, 8);
        assert_eq!(compose, Some(true));
        assert_eq!(
            measured_compose_decision("Intel(R) HD Graphics 520", 128, 4),
            Some(false)
        );
        assert_eq!(
            measured_compose_decision("  intel(r) hd graphics 520 ", 128, 8),
            Some(true),
            "device names are compared case-insensitively and trimmed"
        );
        assert_eq!(
            measured_compose_decision("Intel(R) HD Graphics 520", 256, 8),
            None,
            "a width that was never measured must not borrow another width's decision"
        );
        assert_eq!(
            measured_compose_decision("Some Other GPU", 128, 8),
            None,
            "an unrecorded adapter must fall back to the vendor gate"
        );
    }
}
