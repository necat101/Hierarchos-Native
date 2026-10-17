//! Per-device invocation coverage for the kernel golden harnesses.
//!
//! The golden files already key a digest by device, which catches a *changed* kernel. It
//! does not catch a harness that stops exercising an invocation on a device: deleting a
//! case, or a device losing a kernel it used to run, only lowers the invocation count in
//! the pass line while every digest that remains still matches. That is the same gap the
//! training-state trajectory check closed with `training_state_coverage.json`, and this is
//! the entry type for the dense, reduction and attention goldens.
//!
//! The record's semantics -- the envelope, the audit, the merge, the shrink refusal and the
//! bootstrap -- are [`crate::coverage_record`], shared with the training-state record; what
//! this module adds is what "loses coverage" means for one kernel-golden entry: a recorded
//! invocation the run did not exercise, or one the golden has a digest for on this device
//! that the run did not reproduce.
//!
//! Enforcement runs in the harness: `kernel_golden::Golden` audits this record in `finish`
//! and fails when a recorded invocation is no longer exercised on the device, when the
//! golden holds a digest for the device that the run did not reproduce, or when the device
//! has no entry at all. `--record-coverage` merges this machine's coverage in, refusing to
//! shrink an entry unless `--allow-coverage-shrink` says so; `--bootstrap-coverage` creates
//! only the entries this machine is missing and prints the diff.
//!
//! This module is pure -- no device, no Vulkan -- and lives in the library so
//! `cargo test --lib` can check the audit and the merge without a GPU. The rest of the
//! golden contract (`kernel_golden.rs`, and the shaders it drives) is included by path
//! from the harnesses instead.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::coverage_record::{self, CoverageModel};
pub use crate::coverage_record::{Bootstrap, CoverageFile, Merge};

/// Bump when the coverage record's shape changes; an older file then fails loudly.
pub const COVERAGE_SCHEMA: u32 = coverage_record::SCHEMA;

/// The explanation this record writes into its own file.
pub const COVERAGE_NOTE: &str = <CoverageEntry as CoverageModel>::NOTE;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoverageEntry {
    /// The golden this coverage belongs to, by default the golden file's own name.
    pub golden: String,
    pub device: String,
    /// Invocations this device exercised, sorted.
    pub labels: Vec<String>,
}

impl CoverageModel for CoverageEntry {
    /// Invocations the golden already has a digest for *on this device*. Those are
    /// invocations this device has been verified against, so a run that no longer exercises
    /// one has stopped testing it whether or not the record still lists it.
    type Evidence = BTreeSet<String>;
    const NOTE: &'static str =
        "Invocations each device exercises in the kernel golden harnesses, one entry per \
(golden, device). Written only by a harness run with --record-coverage, which merges this \
machine's coverage and leaves every other device's entry as it was; --bootstrap-coverage \
creates only the entries a device is missing and prints what it added. A run fails when a \
recorded invocation is no longer exercised on that device, when the golden holds a digest \
for the device that the run did not reproduce, or when the device has no entry here; a \
newly exercised invocation is a note, and --allow-coverage-shrink is required to record a \
loss.";

    fn key(&self) -> (String, String) {
        (self.golden.clone(), self.device.clone())
    }

    fn diff(&self, recorded: &Self, digests: &Self::Evidence) -> (Vec<String>, Vec<String>) {
        let label = self.label();
        let covered: BTreeSet<&str> = recorded.labels.iter().map(String::as_str).collect();
        let observed: BTreeSet<&str> = self.labels.iter().map(String::as_str).collect();
        let mut problems = Vec::new();
        let mut notes = Vec::new();
        for invocation in &covered {
            if !observed.contains(invocation) {
                problems.push(format!(
                    "{label}: invocation {invocation} was covered and this run did not exercise it"
                ));
            }
        }
        for invocation in &observed {
            if !covered.contains(invocation) {
                notes.push(format!(
                    "{label}: invocation {invocation} is newly exercised and not yet recorded"
                ));
            }
        }
        for invocation in digests {
            if !observed.contains(invocation.as_str()) {
                problems.push(format!(
                    "{label}: invocation {invocation} has a digest recorded for this device and \
                     this run did not exercise it"
                ));
            } else if !covered.contains(invocation.as_str()) {
                notes.push(format!(
                    "{label}: invocation {invocation} is verified but missing from the coverage \
                     record"
                ));
            }
        }
        (problems, notes)
    }

    fn summary(&self) -> String {
        format!("{}: {} invocation(s)", self.label(), self.labels.len())
    }
}

/// The checked-in record the harnesses audit against when `--coverage` does not name one.
pub fn default_coverage_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("validation")
        .join("kernel_golden_coverage.json")
}

/// The coverage flags a harness exposes, resolved.
pub struct CoverageArgs {
    pub path: PathBuf,
    pub key: Option<String>,
    pub record: bool,
    pub allow_shrink: bool,
    pub bootstrap: bool,
}

/// Parse the coverage flags, which every harness accepts.
///
/// `--coverage FILE` names the record (default: the checked-in one), `--coverage-key`
/// overrides the entry key (default: the golden file's own name), `--record-coverage`
/// merges this run's coverage in instead of auditing it, `--bootstrap-coverage` creates only
/// the entries this device is missing and prints the diff, and `--allow-coverage-shrink`
/// permits a merge that loses an invocation.
pub fn coverage_args(arguments: &[String]) -> Result<CoverageArgs> {
    let mut resolved = CoverageArgs {
        path: default_coverage_path(),
        key: None,
        record: false,
        allow_shrink: false,
        bootstrap: false,
    };
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--coverage" => {
                resolved.path = PathBuf::from(
                    arguments
                        .get(index + 1)
                        .context("missing coverage file path")?,
                );
                index += 2;
            }
            "--coverage-key" => {
                resolved.key = Some(
                    arguments
                        .get(index + 1)
                        .context("missing coverage key")?
                        .clone(),
                );
                index += 2;
            }
            "--record-coverage" => {
                resolved.record = true;
                index += 1;
            }
            "--allow-coverage-shrink" => {
                resolved.allow_shrink = true;
                index += 1;
            }
            "--bootstrap-coverage" => {
                resolved.bootstrap = true;
                index += 1;
            }
            _ => index += 1,
        }
    }
    Ok(resolved)
}

/// The key a golden's coverage is recorded under: its own file name, unless overridden.
pub fn coverage_key(golden: &Path, override_key: Option<&str>) -> String {
    if let Some(key) = override_key {
        return key.to_string();
    }
    golden
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| golden.display().to_string())
}

/// Read the coverage record, or `None` when it does not exist yet.
pub fn read_coverage(path: &Path) -> Result<Option<CoverageFile<CoverageEntry>>> {
    coverage_record::read(path)
}

/// Write the coverage record, creating its directory if needed.
pub fn write_coverage(
    path: &Path,
    coverage: Vec<CoverageEntry>,
) -> Result<CoverageFile<CoverageEntry>> {
    coverage_record::write(path, coverage)
}

fn entry(key: &str, device: &str, observed: &BTreeSet<String>) -> CoverageEntry {
    CoverageEntry {
        golden: key.to_string(),
        device: device.to_string(),
        labels: observed.iter().cloned().collect(),
    }
}

/// Audit one run against its device's recorded coverage, as `(problems, notes)`.
pub fn coverage_problems(
    key: &str,
    device: &str,
    observed: &BTreeSet<String>,
    digests: &BTreeSet<String>,
    recorded: Option<&CoverageEntry>,
) -> (Vec<String>, Vec<String>) {
    let candidate = entry(key, device, observed);
    match recorded {
        Some(recorded) => candidate.diff(recorded, digests),
        None => coverage_record::audit(&[], &[candidate], digests).into_pair(),
    }
}

/// Record this run's invocation coverage, keeping every other device's entry.
///
/// The returned [`Merge`] carries what the merge would write and, in the shared wording,
/// what it would lose.
pub fn merge_coverage(
    existing: &[CoverageEntry],
    key: &str,
    device: &str,
    observed: &BTreeSet<String>,
) -> Merge<CoverageEntry> {
    let candidate = entry(key, device, observed);
    coverage_record::merge(existing, &[candidate], &BTreeSet::new())
}

/// Create this run's coverage entry when the record does not hold it yet.
///
/// Only the entries this device is missing are created; an entry that already existed is
/// left exactly as recorded, which is why bootstrap has nothing to refuse.
pub fn bootstrap_coverage(
    existing: &[CoverageEntry],
    key: &str,
    device: &str,
    observed: &BTreeSet<String>,
) -> Bootstrap<CoverageEntry> {
    let candidate = entry(key, device, observed);
    coverage_record::bootstrap(existing, &[candidate])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The audit and merge rules for this entry type are covered once, with the shared
    /// machinery, in `coverage_record::tests`; what is left here is flag plumbing.
    #[test]
    fn the_coverage_flags_resolve_their_defaults() {
        let args = coverage_args(&[]).expect("no flags is the default audit");
        assert!(
            args.path.ends_with("kernel_golden_coverage.json"),
            "{:?}",
            args.path
        );
        assert!(!args.record && !args.allow_shrink && !args.bootstrap && args.key.is_none());
        let args = coverage_args(&[
            "--coverage".to_string(),
            "tmp/coverage.json".to_string(),
            "--coverage-key".to_string(),
            "reduction".to_string(),
            "--record-coverage".to_string(),
            "--allow-coverage-shrink".to_string(),
            "--bootstrap-coverage".to_string(),
        ])
        .expect("the flags parse");
        assert_eq!(args.path, PathBuf::from("tmp/coverage.json"));
        assert_eq!(args.key.as_deref(), Some("reduction"));
        assert!(args.record && args.allow_shrink && args.bootstrap);
    }

    #[test]
    fn the_coverage_key_is_the_goldens_name() {
        assert_eq!(
            coverage_key(Path::new("validation/kernel_golden/dense.json"), None),
            "dense"
        );
        assert_eq!(
            coverage_key(
                Path::new("validation/kernel_golden/dense.json"),
                Some("attention")
            ),
            "attention"
        );
    }
}
