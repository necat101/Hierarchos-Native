//! One implementation of the per-device coverage record contract.
//!
//! Two records answer the same question about a different surface:
//!
//! * `validation/training_state_coverage.json` -- which topology arm ran which golden case,
//!   and how much each case compared, on a device.
//! * `validation/kernel_golden_coverage.json` -- which invocations the dense, reduction and
//!   attention golden harnesses exercised on a device.
//!
//! Both exist because a *changed* result is already caught by the goldens (they are keyed by
//! device), while a *lost* invocation is not: deleting a case, or a device losing a kernel it
//! used to run, only lowers a count in the pass line while every digest that remains still
//! matches. Both records therefore have to answer three questions the same way, and before
//! this module they answered them twice, in two languages, with two sets of tests:
//!
//! 1. **Audit** -- everything the record holds for this device must still hold, and growth is
//!    a note. A device with no entry fails rather than defaulting to green.
//! 2. **Merge** -- recording replaces only the entries this run measured and leaves every
//!    other device's entry exactly as it was.
//! 3. **Refusal** -- a merge that loses coverage is reported, not applied, unless the caller
//!    asked for the loss with `--allow-coverage-shrink`.
//!
//! What differs between the records is only what "loses coverage" means for one entry, which
//! is [`CoverageModel::diff`]. The envelope, the audit, the merge, the refusal vocabulary, the
//! bootstrap and the on-disk format live here once, and the test suite at the bottom of this
//! file covers both models plus the shared machinery.
//!
//! **Bootstrap.** A device with no entry cannot audit, and the fail-closed rule is what makes
//! the record evidence rather than a formality. `--bootstrap-coverage` is the deliberate,
//! narrow way out: it creates the entries this run is missing and touches nothing else. It
//! cannot shrink, cannot rewrite an existing entry, and prints exactly what it added so the
//! diff is reviewable before the record is committed.
//!
//! This module is pure -- no device, no Vulkan, no subprocess -- so `cargo test --lib` checks
//! the whole contract without a GPU.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Bump when a record's shape changes; an older file then fails loudly.
pub const SCHEMA: u32 = 1;

/// One entry of a coverage record: a keyed claim about what a device exercised.
///
/// `diff` is the whole per-record half of the contract. It is called with `self` being what
/// this run measured and `recorded` being what the record holds, and returns the losses (a
/// failure, and what a merge refuses unless told otherwise) and the growth (a note).
pub trait CoverageModel: Clone + Serialize + DeserializeOwned {
    /// Extra run evidence a loss can be judged against; `()` where the entry is the evidence.
    type Evidence: Default;
    /// The record's own explanation, written into the file's `note`.
    const NOTE: &'static str;

    /// `(record, device)` -- the pair that identifies one entry across the whole file.
    fn key(&self) -> (String, String);

    /// `(losses, growth)` of `self` against the recorded entry.
    fn diff(&self, recorded: &Self, evidence: &Self::Evidence) -> (Vec<String>, Vec<String>);

    /// One line for a printed diff: what this entry covers.
    fn summary(&self) -> String;

    /// `dense on 'Intel(R) HD Graphics 520'` -- the prefix every message carries.
    fn label(&self) -> String {
        let (record, device) = self.key();
        format!("{record} on '{device}'")
    }
}

/// The on-disk record: an envelope both records share.
#[derive(Debug, Serialize, Deserialize)]
pub struct CoverageFile<E> {
    pub schema: u32,
    pub note: String,
    pub coverage: Vec<E>,
}

/// Read a record, or `None` when the file does not exist yet.
///
/// A record that cannot be trusted is an error rather than an empty list: an unsupported
/// schema or a duplicated entry means the file was written by something other than
/// [`write`], and auditing against it would compare a claim with itself.
pub fn read<E: CoverageModel>(path: &Path) -> Result<Option<CoverageFile<E>>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading coverage record {}", path.display()))?;
    let parsed: CoverageFile<E> = serde_json::from_str(&text)
        .with_context(|| format!("parsing coverage record {}", path.display()))?;
    if parsed.schema != SCHEMA {
        bail!(
            "coverage record {} has schema {}, this record is schema {SCHEMA}",
            path.display(),
            parsed.schema
        );
    }
    let mut keys = BTreeSet::new();
    for entry in &parsed.coverage {
        if !keys.insert(entry.key()) {
            bail!(
                "coverage record {} holds duplicate entries; write it with --record-coverage \
                 rather than by hand",
                path.display()
            );
        }
    }
    Ok(Some(parsed))
}

/// Write a record, creating its directory if needed.
pub fn write<E: CoverageModel>(path: &Path, coverage: Vec<E>) -> Result<CoverageFile<E>> {
    let document = CoverageFile {
        schema: SCHEMA,
        note: E::NOTE.to_string(),
        coverage,
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&document)? + "\n")
        .with_context(|| format!("writing coverage record {}", path.display()))?;
    Ok(document)
}

/// Audit this run against the record.
#[derive(Debug, Default)]
pub struct Audit {
    pub problems: Vec<String>,
    pub notes: Vec<String>,
}

impl Audit {
    /// The audit as `(problems, notes)`, the shape both callers report with.
    pub fn into_pair(self) -> (Vec<String>, Vec<String>) {
        (self.problems, self.notes)
    }
}

/// Everything the record holds must still hold; an entry this run is missing from the record
/// fails, because that is exactly the case the record exists to make visible.
pub fn audit<E: CoverageModel>(recorded: &[E], candidates: &[E], evidence: &E::Evidence) -> Audit {
    let by_key: BTreeMap<(String, String), &E> =
        recorded.iter().map(|entry| (entry.key(), entry)).collect();
    let mut outcome = Audit::default();
    for candidate in candidates {
        match by_key.get(&candidate.key()) {
            Some(previous) => {
                let (problems, notes) = candidate.diff(previous, evidence);
                outcome.problems.extend(problems);
                outcome.notes.extend(notes);
            }
            None => outcome.problems.push(format!(
                "{}: no coverage recorded for this device; run with --record-coverage on this \
                 machine to add its entry, or --bootstrap-coverage to create only the entries \
                 this run is missing",
                candidate.label()
            )),
        }
    }
    outcome
}

/// What a merge did: the record it would write, and what that costs.
#[derive(Debug)]
pub struct Merge<E> {
    pub merged: Vec<E>,
    /// Entries this record did not hold before.
    pub added: Vec<E>,
    /// Entries this record held and this run re-measured.
    pub replaced: Vec<E>,
    /// What replacing them would lose, as messages. Non-empty is a refusal unless asked for.
    pub shrink: Vec<String>,
}

/// Replace the entries this run measured, keeping every other device's entry as it was.
///
/// A loss is reported rather than applied, because the record is what makes a dropped
/// invocation a failure; the caller decides whether `--allow-coverage-shrink` was passed.
pub fn merge<E: CoverageModel>(
    existing: &[E],
    candidates: &[E],
    evidence: &E::Evidence,
) -> Merge<E> {
    let mut by_key: BTreeMap<(String, String), E> = existing
        .iter()
        .map(|entry| (entry.key(), entry.clone()))
        .collect();
    let mut outcome = Merge {
        merged: Vec::new(),
        added: Vec::new(),
        replaced: Vec::new(),
        shrink: Vec::new(),
    };
    for candidate in candidates {
        match by_key.get(&candidate.key()) {
            Some(previous) => {
                let (problems, _) = candidate.diff(previous, evidence);
                outcome.shrink.extend(problems);
                outcome.replaced.push(candidate.clone());
            }
            None => outcome.added.push(candidate.clone()),
        }
        by_key.insert(candidate.key(), candidate.clone());
    }
    outcome.merged = by_key.into_values().collect();
    outcome
}

impl<E> Merge<E> {
    /// The refusal message for a merge that loses coverage, in the one wording both records
    /// use, or `None` when nothing was lost.
    pub fn refusal(&self) -> Option<String> {
        if self.shrink.is_empty() {
            return None;
        }
        Some(format!(
            "refusing to record a shrunken entry: {}\npass --allow-coverage-shrink if losing \
             this coverage is intended",
            self.shrink.join("; ")
        ))
    }
}

/// What `--bootstrap-coverage` did: entries created, and entries left alone.
#[derive(Debug)]
pub struct Bootstrap<E> {
    pub merged: Vec<E>,
    pub added: Vec<E>,
    /// The entries this record already held, which bootstrap never rewrites.
    pub already: Vec<E>,
}

impl<E: CoverageModel> Bootstrap<E> {
    /// The printed diff: one line per entry created, then what the file now holds.
    ///
    /// Bootstrap always writes to the record it was pointed at, so the diff is the only thing
    /// that stands between a new device and silently blessing whatever its first run happened
    /// to exercise. It is printed for that reason and is part of the flag's contract.
    pub fn diff_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if self.added.is_empty() {
            lines.push(format!(
                "bootstrap found every entry of this run already recorded ({} left untouched); \
                 nothing was written",
                self.already.len()
            ));
            return lines;
        }
        lines.push(format!(
            "bootstrap added {} coverage entr{} ({} already recorded): the record now holds {} \
             entries",
            self.added.len(),
            if self.added.len() == 1 { "y" } else { "ies" },
            self.already.len(),
            self.merged.len()
        ));
        for entry in &self.added {
            lines.push(format!("+ {}", entry.summary()));
        }
        lines.push(
            "review the added entries and commit the record; the next run audits it instead of \
             trusting this one"
                .to_string(),
        );
        lines
    }
}

/// Create only the entries this record is missing, and leave every existing entry alone.
///
/// This is the deliberate first run on a new device: nothing is replaced, so there is nothing
/// to shrink and nothing to refuse. It returns the entries it created so the caller can print
/// the diff.
pub fn bootstrap<E: CoverageModel>(existing: &[E], candidates: &[E]) -> Bootstrap<E> {
    let mut by_key: BTreeMap<(String, String), E> = existing
        .iter()
        .map(|entry| (entry.key(), entry.clone()))
        .collect();
    let mut outcome = Bootstrap {
        merged: Vec::new(),
        added: Vec::new(),
        already: Vec::new(),
    };
    for candidate in candidates {
        if by_key.contains_key(&candidate.key()) {
            outcome.already.push(candidate.clone());
            continue;
        }
        by_key.insert(candidate.key(), candidate.clone());
        outcome.added.push(candidate.clone());
    }
    outcome.merged = by_key.into_values().collect();
    outcome
}

// ---------------------------------------------------------------------------------------
// The training-state record's entry.
// ---------------------------------------------------------------------------------------

/// One `(suite, device)` entry of `validation/training_state_coverage.json`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct TrainingStateEntry {
    pub suite: String,
    pub device: String,
    /// What the measured arm compared per golden case.
    #[serde(default)]
    pub cases: BTreeMap<String, CaseCounts>,
    /// Where each forced topology arm ran, and where the variant is not there.
    #[serde(default)]
    pub arms: BTreeMap<String, ArmCoverage>,
}

/// How much one golden case compares.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CaseCounts {
    #[serde(default)]
    pub steps: u64,
    #[serde(default)]
    pub tensor_comparisons: u64,
}

/// One forced arm's verdicts in a `(suite, device)` entry.
///
/// `exercised` is where the forced variant ran. The two absence verdicts are kept apart
/// because they mean different things: the producer rejected the request (`unavailable`), or
/// the fused producer never dispatched at that shape at all (`not_dispatched`). `asserted` is
/// the subset a later run carried from this record instead of measuring again -- re-checked by
/// one canary per arm -- so a carried claim never looks like fresh evidence.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ArmCoverage {
    #[serde(default)]
    pub exercised: Vec<String>,
    #[serde(default)]
    pub unavailable: Vec<String>,
    #[serde(default)]
    pub not_dispatched: Vec<String>,
    #[serde(default)]
    pub asserted: Vec<String>,
}

impl ArmCoverage {
    /// The cases where this record says the variant is not running on this device.
    fn absent(&self) -> BTreeSet<&str> {
        self.unavailable
            .iter()
            .chain(self.not_dispatched.iter())
            .map(String::as_str)
            .collect()
    }
}

fn quoted(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("'{value}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

impl TrainingStateEntry {
    pub const RECORD_NOTE: &'static str =
        "Topology arms and golden cases each device exercises, one \
entry per (suite, device). Written only by the coverage record tool (`--record-coverage` on \
`validation/verify_training_state.py`), which merges what this machine measured and leaves every \
other device's entries as they were; a suite is a training numerical policy and an arm is one \
forced LN1/low-rank topology. `cases` is what the measured arm compares per golden case. Per arm, \
`exercised` is where the forced variant ran, `unavailable` is where the producer rejected the \
request, `not_dispatched` is where the fused producer never saw it at that shape, and `asserted` \
is the subset of the two absence verdicts a later run carried from this record instead of \
measuring again; one canary -- the widest absent case, replayed for one update -- re-checks an \
arm's absence on this device every run. A run fails if any of it shrinks: an arm, a case, a case \
that compares fewer steps or tensors than it did, or an absence verdict that is neither \
re-measured nor carried. `--bootstrap-coverage` creates the entries a device is missing and \
touches nothing else.";
}

impl CoverageModel for TrainingStateEntry {
    type Evidence = ();
    const NOTE: &'static str = TrainingStateEntry::RECORD_NOTE;

    fn key(&self) -> (String, String) {
        (self.suite.clone(), self.device.clone())
    }

    fn diff(&self, recorded: &Self, _: &()) -> (Vec<String>, Vec<String>) {
        let label = self.label();
        let mut problems = Vec::new();
        let mut notes = Vec::new();
        for (name, counts) in &recorded.cases {
            let Some(measured) = self.cases.get(name) else {
                problems.push(format!(
                    "{label}: golden case '{name}' was covered and this run did not measure it"
                ));
                continue;
            };
            for (field, now, before) in [
                ("steps", measured.steps, counts.steps),
                (
                    "tensor_comparisons",
                    measured.tensor_comparisons,
                    counts.tensor_comparisons,
                ),
            ] {
                if now < before {
                    problems.push(format!(
                        "{label}: golden case '{name}' now compares {now} {field} against the \
                         recorded {before}; the case got smaller"
                    ));
                }
            }
        }
        for name in self.cases.keys() {
            if !recorded.cases.contains_key(name) {
                notes.push(format!(
                    "{label}: golden case '{name}' is new to the record"
                ));
            }
        }
        for (arm, was) in &recorded.arms {
            let Some(now) = self.arms.get(arm) else {
                problems.push(format!(
                    "{label}: arm '{arm}' was exercised here and this run does not run it"
                ));
                continue;
            };
            let verified: BTreeSet<&str> = was.exercised.iter().map(String::as_str).collect();
            let exercised: BTreeSet<&str> = now.exercised.iter().map(String::as_str).collect();
            let lost: Vec<&str> = verified.difference(&exercised).copied().collect();
            if !lost.is_empty() {
                let rejected: BTreeSet<&str> = now.unavailable.iter().map(String::as_str).collect();
                let reason = if lost.iter().all(|case| rejected.contains(case)) {
                    "this device now reports the variant unavailable"
                } else {
                    "the producer no longer dispatched it"
                };
                let lost: Vec<String> = lost.iter().map(|case| (*case).to_string()).collect();
                problems.push(format!(
                    "{label}: arm '{arm}' no longer runs the forced variant in {} ({reason})",
                    lost.join(", ")
                ));
            }
            for case in &now.exercised {
                if !verified.contains(case.as_str()) {
                    notes.push(format!(
                        "{label}: arm '{arm}' now runs in '{case}', which was not covered"
                    ));
                }
            }
            // An absence the record holds is evidence too: it may be carried unmeasured, but it
            // must still be accounted for, either re-measured or asserted beside its canary.
            let absent_now = now.absent();
            let already_carried: BTreeSet<&str> = now.asserted.iter().map(String::as_str).collect();
            let was_carried: BTreeSet<&str> = was.asserted.iter().map(String::as_str).collect();
            for case in was.absent().union(&was_carried) {
                if exercised.contains(case) {
                    continue;
                }
                if absent_now.contains(case) || already_carried.contains(case) {
                    continue;
                }
                problems.push(format!(
                    "{label}: arm '{arm}' had case '{case}' recorded with the variant absent and \
                     this run reports neither a rejection, nor a missing dispatch, nor a \
                     measurement of it"
                ));
            }
            for case in &now.asserted {
                if verified.contains(case.as_str()) {
                    problems.push(format!(
                        "{label}: arm '{arm}' carried case '{case}' unmeasured although the record \
                         has that variant verified there"
                    ));
                } else if !was.absent().contains(case.as_str()) && !was.asserted.contains(case) {
                    problems.push(format!(
                        "{label}: arm '{arm}' carried case '{case}' unmeasured although the record \
                         does not report the variant absent there"
                    ));
                }
            }
        }
        for arm in self.arms.keys() {
            if !recorded.arms.contains_key(arm) {
                notes.push(format!("{label}: arm '{arm}' is new to the record"));
            }
        }
        (problems, notes)
    }

    fn summary(&self) -> String {
        format!(
            "{}: {} case(s), {} arm(s){}",
            self.label(),
            self.cases.len(),
            self.arms.len(),
            if self.cases.is_empty() {
                String::new()
            } else {
                format!(
                    "; cases {}",
                    quoted(&self.cases.keys().cloned().collect::<Vec<_>>())
                )
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed(labels: &[&str]) -> BTreeSet<String> {
        labels.iter().map(|label| label.to_string()).collect()
    }

    fn kernel_entry(labels: &[&str]) -> crate::CoverageEntry {
        crate::CoverageEntry {
            golden: "dense".to_string(),
            device: "Intel(R) HD Graphics 520".to_string(),
            labels: labels.iter().map(|label| label.to_string()).collect(),
        }
    }

    // -----------------------------------------------------------------------------------
    // The kernel-golden model.
    // -----------------------------------------------------------------------------------

    /// No entry is a failure, and the message names the flag that creates one.
    #[test]
    fn a_device_with_no_recorded_coverage_fails() {
        let outcome = audit(
            &[],
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &observed(&["linear_forward:1x1x1"]),
        );
        assert_eq!(outcome.notes, Vec::<String>::new());
        assert_eq!(outcome.problems.len(), 1, "{:?}", outcome.problems);
        assert!(
            outcome.problems[0].contains("no coverage recorded"),
            "{:?}",
            outcome.problems
        );
        assert!(
            outcome.problems[0].contains("--bootstrap-coverage"),
            "{:?}",
            outcome.problems
        );
    }

    /// The regression this record exists for: an invocation this device used to exercise that
    /// the run no longer does, whichever half of the evidence still lists it.
    #[test]
    fn an_invocation_that_stops_being_exercised_fails() {
        let outcome = audit(
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &observed(&["linear_forward:1x1x1", "linear_forward:8x64x16"]),
        );
        assert_eq!(outcome.problems.len(), 1, "{:?}", outcome.problems);
        assert!(
            outcome.problems[0].contains("has a digest recorded"),
            "{:?}",
            outcome.problems
        );

        let outcome = audit(
            &[kernel_entry(&[
                "linear_forward:1x1x1",
                "linear_forward:8x64x16",
            ])],
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &observed(&["linear_forward:1x1x1"]),
        );
        assert_eq!(outcome.problems.len(), 1, "{:?}", outcome.problems);
        assert!(
            outcome.problems[0].contains("was covered and this run did not exercise it"),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn a_newly_exercised_invocation_is_a_note() {
        let outcome = audit(
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &[kernel_entry(&[
                "linear_forward:1x1x1",
                "linear_forward:8x64x16",
            ])],
            &observed(&["linear_forward:1x1x1", "linear_forward:8x64x16"]),
        );
        assert_eq!(outcome.problems, Vec::<String>::new());
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("newly exercised")),
            "{:?}",
            outcome.notes
        );
    }

    /// An invocation that is exercised and verified but missing from the record is a note, so a
    /// device's record can be completed without having to be exactly what it was.
    #[test]
    fn a_verified_invocation_missing_from_the_record_is_a_note() {
        let outcome = audit(
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &[kernel_entry(&[
                "linear_forward:1x1x1",
                "linear_forward:8x64x16",
            ])],
            &observed(&["linear_forward:1x1x1", "linear_forward:8x64x16"]),
        );
        assert_eq!(
            outcome.problems,
            Vec::<String>::new(),
            "{:?}",
            outcome.problems
        );
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("is verified but missing from the coverage record")),
            "{:?}",
            outcome.notes
        );
    }

    // -----------------------------------------------------------------------------------
    // The shared merge.
    // -----------------------------------------------------------------------------------

    #[test]
    fn recording_merges_this_device_and_lists_its_invocations() {
        let mut other = kernel_entry(&["adamw len=1 step=1"]);
        other.device = "AMD Radeon RX 7900 XTX".to_string();
        let merged = merge(
            &[kernel_entry(&["linear_forward:1x1x1"]), other],
            &[kernel_entry(&[
                "linear_forward:1x1x1",
                "linear_forward:8x64x16",
            ])],
            &observed(&["linear_forward:1x1x1", "linear_forward:8x64x16"]),
        );
        assert_eq!(merged.shrink, Vec::<String>::new());
        assert_eq!(merged.added.len(), 0);
        assert_eq!(merged.replaced.len(), 1);
        assert_eq!(
            merged.replaced[0].labels,
            vec!["linear_forward:1x1x1", "linear_forward:8x64x16"]
        );
        assert_eq!(merged.merged.len(), 2);
        // The other device's entry is carried byte for byte.
        assert!(merged
            .merged
            .iter()
            .any(|entry| entry.device == "AMD Radeon RX 7900 XTX"
                && entry.labels == vec!["adamw len=1 step=1".to_string()]));
        // ... and the record is sorted by (record, device) whichever order it arrived in.
        assert_eq!(merged.merged[0].device, "AMD Radeon RX 7900 XTX");
    }

    #[test]
    fn recording_a_shrunken_entry_is_reported_before_it_is_applied() {
        let merged = merge(
            &[kernel_entry(&[
                "linear_forward:1x1x1",
                "linear_forward:8x64x16",
            ])],
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &observed(&["linear_forward:1x1x1"]),
        );
        assert_eq!(merged.shrink.len(), 1, "{:?}", merged.shrink);
        assert!(
            merged.shrink[0].contains("this run did not exercise it"),
            "{:?}",
            merged.shrink
        );
        // The merge still reflects the run; refusing to write it is the caller's job, so that
        // an intended loss can be recorded with --allow-coverage-shrink.
        assert_eq!(merged.merged.len(), 1);
        assert_eq!(
            merged.merged[0].labels,
            vec!["linear_forward:1x1x1".to_string()]
        );
        let refusal = merged.refusal().expect("a loss is refused");
        assert!(
            refusal.starts_with("refusing to record a shrunken entry: "),
            "{refusal}"
        );
        assert!(refusal.contains("--allow-coverage-shrink"), "{refusal}");
    }

    #[test]
    fn a_clean_merge_has_nothing_to_refuse() {
        let merged = merge(
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &observed(&["linear_forward:1x1x1"]),
        );
        assert!(merged.refusal().is_none());
    }

    // -----------------------------------------------------------------------------------
    // The shared bootstrap.
    // -----------------------------------------------------------------------------------

    #[test]
    fn bootstrap_creates_only_the_entries_that_are_missing() {
        let recorded = TrainingStateEntry {
            suite: "fp32".to_string(),
            device: "Intel(R) HD Graphics 520".to_string(),
            ..TrainingStateEntry::default()
        };
        let mut new_device = recorded.clone();
        new_device.device = "AMD Radeon RX 7900 XTX".to_string();
        let bootstrapped = bootstrap(&[recorded.clone()], &[recorded.clone(), new_device.clone()]);
        assert_eq!(bootstrapped.added.len(), 1);
        assert_eq!(bootstrapped.added[0].device, "AMD Radeon RX 7900 XTX");
        assert_eq!(bootstrapped.already.len(), 1);
        assert_eq!(bootstrapped.merged.len(), 2);
        // The entry that already existed is the one the record held, not the run's copy.
        assert!(bootstrapped.merged.iter().any(|entry| *entry == recorded));
    }

    #[test]
    fn bootstrap_prints_the_diff_it_wrote() {
        let mut entry = TrainingStateEntry {
            suite: "fp16-storage-fp32-compute".to_string(),
            device: "AMD Radeon RX 7900 XTX".to_string(),
            ..TrainingStateEntry::default()
        };
        entry.cases.insert(
            "batch2".to_string(),
            CaseCounts {
                steps: 6,
                tensor_comparisons: 1752,
            },
        );
        entry
            .arms
            .insert("one-row".to_string(), ArmCoverage::default());
        let bootstrapped = bootstrap(&[], &[entry]);
        let lines = bootstrapped.diff_lines();
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(
            lines[0].contains("bootstrap added 1 coverage entry"),
            "{lines:?}"
        );
        assert!(lines[1].starts_with("+ "), "{lines:?}");
        assert!(lines[1].contains("AMD Radeon RX 7900 XTX"), "{lines:?}");
        assert!(lines[2].contains("review the added entries"), "{lines:?}");
    }

    #[test]
    fn bootstrap_with_nothing_missing_writes_nothing() {
        let entry = kernel_entry(&["linear_forward:1x1x1"]);
        let bootstrapped = bootstrap(&[entry.clone()], &[entry]);
        assert!(bootstrapped.added.is_empty());
        assert_eq!(bootstrapped.already.len(), 1);
        let lines = bootstrapped.diff_lines();
        assert!(lines[0].contains("nothing was written"), "{lines:?}");
    }

    // -----------------------------------------------------------------------------------
    // The training-state model.
    // -----------------------------------------------------------------------------------

    fn coverage(device: &str, suite: &str) -> TrainingStateEntry {
        let mut entry = TrainingStateEntry {
            suite: suite.to_string(),
            device: device.to_string(),
            ..TrainingStateEntry::default()
        };
        entry.cases.insert(
            "batch2".to_string(),
            CaseCounts {
                steps: 6,
                tensor_comparisons: 1752,
            },
        );
        entry.arms.insert(
            "one-row".to_string(),
            ArmCoverage {
                exercised: vec!["batch2".to_string()],
                ..ArmCoverage::default()
            },
        );
        entry
    }

    fn arm(exercised: &[&str], unavailable: &[&str], asserted: &[&str]) -> ArmCoverage {
        ArmCoverage {
            exercised: exercised.iter().map(|case| case.to_string()).collect(),
            unavailable: unavailable.iter().map(|case| case.to_string()).collect(),
            not_dispatched: Vec::new(),
            asserted: asserted.iter().map(|case| case.to_string()).collect(),
        }
    }

    fn state(suite: &str, arms: Vec<(&str, ArmCoverage)>) -> TrainingStateEntry {
        let mut entry = coverage("Intel(R) HD Graphics 520", suite);
        entry.arms = arms
            .into_iter()
            .map(|(name, arm)| (name.to_string(), arm))
            .collect();
        entry
    }

    #[test]
    fn the_recorded_coverage_reproducing_is_clean() {
        let entry = coverage("Intel(R) HD Graphics 520", "fp32");
        let outcome = audit(&[entry.clone()], &[entry], &());
        assert_eq!(
            outcome.problems,
            Vec::<String>::new(),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn an_unrecorded_device_fails_rather_than_defaulting() {
        let outcome = audit(&[], &[coverage("AMD Radeon RX 7900 XTX", "fp32")], &());
        assert_eq!(outcome.problems.len(), 1, "{:?}", outcome.problems);
        assert!(
            outcome.problems[0].contains("--record-coverage"),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn a_carried_claim_must_be_in_the_record() {
        let recorded = coverage("Intel(R) HD Graphics 520", "fp32");
        let carried = state("fp32", vec![("one-row", arm(&["batch2"], &[], &["odd"]))]);
        let outcome = audit(&[recorded], &[carried], &());
        assert!(
            outcome
                .problems
                .iter()
                .any(|problem| problem.contains("does not report the variant absent")),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn carrying_a_case_the_record_verified_fails() {
        let recorded = coverage("Intel(R) HD Graphics 520", "fp32");
        let carried = state("fp32", vec![("one-row", arm(&[], &[], &["batch2"]))]);
        let outcome = audit(&[recorded], &[carried], &());
        assert!(
            outcome
                .problems
                .iter()
                .any(|problem| problem.contains("has that variant verified there")),
            "{:?}",
            outcome.problems
        );
        // ... and it is a shrink too: the verification the record holds was dropped.
        let lost = merge(
            &[coverage("Intel(R) HD Graphics 520", "fp32")],
            &[state("fp32", vec![("one-row", arm(&[], &[], &["batch2"]))])],
            &(),
        );
        assert!(
            lost.shrink
                .iter()
                .any(|problem| problem.contains("no longer runs the forced variant")),
            "{:?}",
            lost.shrink
        );
    }

    #[test]
    fn a_recorded_rejection_that_is_neither_measured_nor_carried_fails() {
        let recorded = state(
            "fp32",
            vec![("one-row", arm(&["batch2"], &["batch32"], &[]))],
        );
        let dropped = state("fp32", vec![("one-row", arm(&[], &[], &[]))]);
        let outcome = audit(&[recorded], &[dropped], &());
        assert!(
            outcome
                .problems
                .iter()
                .any(|problem| problem
                    .contains("reports neither a rejection, nor a missing dispatch")),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn a_recorded_missing_dispatch_that_vanishes_also_fails() {
        let mut recorded = state("fp32", vec![("one-row", arm(&["batch2"], &[], &[]))]);
        recorded
            .arms
            .get_mut("one-row")
            .expect("the arm is recorded")
            .not_dispatched = vec!["odd".to_string()];
        let dropped = state("fp32", vec![("one-row", arm(&["batch2"], &[], &[]))]);
        let outcome = audit(&[recorded], &[dropped], &());
        assert!(
            outcome
                .problems
                .iter()
                .any(|problem| problem.contains("missing dispatch")),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn a_rejection_this_run_confirmed_is_clean() {
        let entry = state("fp32", vec![("one-row", arm(&["batch2"], &[], &[]))]);
        let confirmed = state("fp32", vec![("one-row", arm(&["batch2"], &[], &[]))]);
        let outcome = audit(&[entry], &[confirmed], &());
        assert_eq!(
            outcome.problems,
            Vec::<String>::new(),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn a_rejection_carried_past_its_canary_is_clean() {
        let recorded = state("fp32", vec![("one-row", arm(&["batch2"], &["odd"], &[]))]);
        let carried = state("fp32", vec![("one-row", arm(&["batch2"], &[], &["odd"]))]);
        let outcome = audit(&[recorded], &[carried], &());
        assert_eq!(
            outcome.problems,
            Vec::<String>::new(),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn an_arm_that_stops_running_fails() {
        let recorded = coverage("Intel(R) HD Graphics 520", "fp32");
        let shrunk = state("fp32", vec![("one-row", arm(&[], &[], &[]))]);
        let outcome = audit(&[recorded], &[shrunk], &());
        assert_eq!(outcome.problems.len(), 1, "{:?}", outcome.problems);
        assert!(
            outcome.problems[0].contains("no longer runs the forced variant in batch2"),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn an_arm_that_became_unavailable_where_it_ran_fails_with_that_reason() {
        let recorded = coverage("Intel(R) HD Graphics 520", "fp32");
        let shrunk = state("fp32", vec![("one-row", arm(&[], &["batch2"], &[]))]);
        let outcome = audit(&[recorded], &[shrunk], &());
        assert_eq!(outcome.problems.len(), 1, "{:?}", outcome.problems);
        assert!(
            outcome.problems[0].contains("now reports the variant unavailable"),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn a_golden_case_that_stops_being_measured_fails() {
        let recorded = coverage("Intel(R) HD Graphics 520", "fp32");
        let mut shrunk = coverage("Intel(R) HD Graphics 520", "fp32");
        shrunk.cases.clear();
        let outcome = audit(&[recorded], &[shrunk], &());
        assert!(
            outcome
                .problems
                .iter()
                .any(|problem| problem.contains("was covered and this run did not measure it")),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn a_case_that_compares_less_than_it_did_fails() {
        let recorded = coverage("Intel(R) HD Graphics 520", "fp32");
        let mut shrunk = coverage("Intel(R) HD Graphics 520", "fp32");
        shrunk
            .cases
            .get_mut("batch2")
            .expect("the case is recorded")
            .steps = 3;
        let outcome = audit(&[recorded], &[shrunk], &());
        assert_eq!(outcome.problems.len(), 1, "{:?}", outcome.problems);
        assert!(
            outcome.problems[0].contains("the case got smaller"),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn grown_coverage_is_a_note() {
        let recorded = coverage("Intel(R) HD Graphics 520", "fp32");
        let mut grown = coverage("Intel(R) HD Graphics 520", "fp32");
        grown.cases.insert(
            "odd".to_string(),
            CaseCounts {
                steps: 3,
                tensor_comparisons: 900,
            },
        );
        grown.arms.insert(
            "one-row".to_string(),
            ArmCoverage {
                exercised: vec!["batch2".to_string(), "odd".to_string()],
                ..ArmCoverage::default()
            },
        );
        let outcome = audit(&[recorded], &[grown], &());
        assert_eq!(
            outcome.problems,
            Vec::<String>::new(),
            "{:?}",
            outcome.problems
        );
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("now runs in 'odd'")),
            "{:?}",
            outcome.notes
        );
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("golden case 'odd' is new")),
            "{:?}",
            outcome.notes
        );
    }

    // -----------------------------------------------------------------------------------
    // The shared envelope.
    // -----------------------------------------------------------------------------------

    #[test]
    fn a_record_round_trips_through_disk() {
        let scratch = std::env::temp_dir().join("coverage-record-test");
        let path = scratch.join("coverage.json");
        let _ = std::fs::remove_dir_all(&scratch);
        let merged = merge(
            &[],
            &[kernel_entry(&["adamw len=1 step=1"])],
            &observed(&[]),
        );
        let written = write(&path, merged.merged).expect("the record writes");
        assert_eq!(written.schema, SCHEMA);
        assert!(
            written.note.contains("Invocations each device exercises"),
            "{written:?}"
        );
        let read_back = read::<crate::CoverageEntry>(&path)
            .expect("the record reads")
            .expect("the record exists");
        assert_eq!(read_back.coverage.len(), 1);
        assert_eq!(read_back.coverage[0].golden, "dense");
        assert_eq!(
            read_back.coverage[0].labels,
            vec!["adamw len=1 step=1".to_string()]
        );

        let mut unsupported = read_back;
        unsupported.schema = SCHEMA + 1;
        std::fs::write(
            &path,
            serde_json::to_string(&unsupported).expect("serializes"),
        )
        .expect("the doctored record writes");
        let error =
            read::<crate::CoverageEntry>(&path).expect_err("an unsupported schema is rejected");
        assert!(format!("{error:#}").contains("has schema"), "{error:#}");

        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn a_duplicated_entry_is_rejected_rather_than_audited() {
        let scratch = std::env::temp_dir().join("coverage-record-duplicate-test");
        let path = scratch.join("coverage.json");
        let _ = std::fs::remove_dir_all(&scratch);
        let entry = kernel_entry(&["linear_forward:1x1x1"]);
        write(&path, vec![entry.clone(), entry]).expect("the doctored record writes");
        let error = read::<crate::CoverageEntry>(&path).expect_err("duplicates are rejected");
        assert!(
            format!("{error:#}").contains("duplicate entries"),
            "{error:#}"
        );
        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn both_records_refuse_a_loss_in_the_same_words() {
        let kernel = merge(
            &[kernel_entry(&[
                "linear_forward:1x1x1",
                "linear_forward:8x64x16",
            ])],
            &[kernel_entry(&["linear_forward:1x1x1"])],
            &observed(&["linear_forward:1x1x1"]),
        )
        .refusal()
        .expect("the kernel record refuses");
        let state = merge(
            &[coverage("Intel(R) HD Graphics 520", "fp32")],
            &[state("fp32", vec![("one-row", arm(&[], &[], &[]))])],
            &(),
        )
        .refusal()
        .expect("the training-state record refuses");
        for refusal in [kernel, state] {
            assert!(
                refusal.starts_with("refusing to record a shrunken entry: "),
                "{refusal}"
            );
            assert!(
                refusal
                    .contains("pass --allow-coverage-shrink if losing this coverage is intended"),
                "{refusal}"
            );
        }
    }
}
