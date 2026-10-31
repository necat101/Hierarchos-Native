//! Golden output digests for the kernel parity harnesses.
//!
//! A parity harness normally proves a kernel change is bit-exact by comparing
//! against the previous artifact directory. That proof is unavailable exactly
//! where it matters most: CI, or any machine that does not ship the pre-change
//! `.spv` files. Recording a digest of every invocation's output vector turns
//! the same proof into a portable one — hash today's outputs and compare them
//! with the recorded digests, with no baseline directory present.
//!
//! Digests are keyed by device because bit-identical arithmetic is only
//! guaranteed within one driver and ISA. A golden file therefore accumulates one
//! digest per device name, so an AMD, Intel, or NVIDIA runner each check the
//! entry recorded on their own hardware; an invocation recorded on a different
//! device is reported as unverified instead of quietly passing.
//!
//! An unverified count is not the only way a harness can quietly stop covering a
//! device, so the invocations each device exercises are checked in as well, in
//! `validation/kernel_golden_coverage.json`. A run fails when an invocation its device
//! is recorded as exercising is no longer exercised, when the golden holds a digest for
//! this device that the run did not reproduce, and when this device has no coverage
//! recorded at all; a new invocation is a note and `--record-coverage` merges this
//! machine's coverage in, refusing to shrink an entry unless `--allow-coverage-shrink`
//! says the loss is intended. Without that rule, deleting a case from a harness -- or a
//! device losing a kernel it used to run -- would only lower the invocation count in the
//! pass line while every remaining digest still matched. `--bootstrap-coverage` is the
//! deliberate way onto a new device: it creates only the entries this run is missing,
//! prints what it added, and leaves every other entry as recorded. The record's audit,
//! merge and refusal are the shared implementation in `coverage_record`, the same one the
//! training-state record uses.
//!
//! This module is included by path from the harness binaries, the same way they
//! include `vulkan.rs`, so it adds no library surface.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bump when the digest contract changes; an older file then fails loudly
/// instead of comparing incomparable digests.
pub const SCHEMA: u32 = 1;

#[derive(Default, Serialize, Deserialize)]
pub struct GoldenFile {
    pub schema: u32,
    /// Invocation label -> device name -> hex digest.
    pub entries: BTreeMap<String, BTreeMap<String, String>>,
}

/// Hex SHA-256 over the little-endian bit patterns of one output vector.
pub fn digest_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Hex SHA-256 over one byte range, used for buffers read as `u32`.
pub fn digest_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Digest of several output vectors that belong to one invocation.
pub fn digest_parts(parts: &[&[f32]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((parts.len() as u64).to_le_bytes());
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        for value in part.iter() {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// The per-device coverage half of this contract is pure -- no device, no Vulkan -- so it
/// lives in the library where `cargo test --lib` checks it without a GPU. It is re-exported
/// here so a harness keeps one import site for everything golden-related.
pub use hierarchos_vulkan::{
    bootstrap_coverage, coverage_args, coverage_key, coverage_problems, merge_coverage,
    read_coverage, write_coverage, CoverageArgs,
};

#[derive(Clone, Copy, PartialEq)]
pub enum GoldenMode {
    Compare,
    Write,
}

pub struct Golden {
    mode: GoldenMode,
    path: PathBuf,
    file: GoldenFile,
    device: String,
    strict: bool,
    matched: usize,
    recorded: usize,
    unverified: usize,
    mismatches: Vec<String>,
    missing: Vec<String>,
    /// Every invocation this run exercised on this device, whether or not a digest was
    /// recorded for it here. This is the run's half of the coverage audit.
    observed: BTreeSet<String>,
    coverage: Option<CoverageArgs>,
    key: String,
    restricted: bool,
}

impl Golden {
    pub fn compare(path: impl AsRef<Path>, device: &str) -> Result<Self> {
        Self::load(path.as_ref(), device, GoldenMode::Compare)
    }

    pub fn write(path: impl AsRef<Path>, device: &str) -> Result<Self> {
        Self::load(path.as_ref(), device, GoldenMode::Write)
    }

    fn load(path: &Path, device: &str, mode: GoldenMode) -> Result<Self> {
        let file = if path.exists() {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading golden file {}", path.display()))?;
            let parsed: GoldenFile = serde_json::from_str(&text)
                .with_context(|| format!("parsing golden file {}", path.display()))?;
            if parsed.schema != SCHEMA {
                bail!(
                    "golden file {} has schema {}, this harness records schema {SCHEMA}",
                    path.display(),
                    parsed.schema
                );
            }
            parsed
        } else {
            if mode == GoldenMode::Compare {
                bail!(
                    "golden file {} does not exist; record it first with --write-golden",
                    path.display()
                );
            }
            GoldenFile {
                schema: SCHEMA,
                ..Default::default()
            }
        };
        Ok(Self {
            mode,
            path: path.to_path_buf(),
            file,
            device: device.to_string(),
            strict: false,
            matched: 0,
            recorded: 0,
            unverified: 0,
            mismatches: Vec::new(),
            missing: Vec::new(),
            observed: BTreeSet::new(),
            coverage: None,
            key: String::new(),
            restricted: false,
        })
    }

    /// Refuse to pass when any invocation had no digest recorded for this
    /// device. A CI runner on unseen hardware would otherwise verify nothing
    /// and still exit zero.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Audit this run's invocation coverage against the record `args` names. The entry key
    /// is the golden's own file name unless `--coverage-key` overrides it, so a run that
    /// points `--golden` somewhere else still audits the same device's coverage.
    pub fn with_coverage(mut self, args: CoverageArgs, golden: &Path) -> Self {
        self.key = coverage_key(golden, args.key.as_deref());
        self.coverage = Some(args);
        self
    }

    /// A run that deliberately exercises only part of the harness cannot be audited for
    /// coverage: it exercises less on purpose, which is not the shrink the record exists to
    /// catch. Such a run says so instead.
    pub fn restricted(mut self, restricted: bool) -> Self {
        self.restricted = restricted;
        self
    }

    /// Record or verify one invocation's outputs. `label` must identify the
    /// kernel and its case; the caller owns the naming convention.
    pub fn observe_digest(&mut self, label: &str, digest: String) {
        match self.mode {
            GoldenMode::Write => {
                self.observed.insert(label.to_string());
                self.file
                    .entries
                    .entry(label.to_string())
                    .or_default()
                    .insert(self.device.clone(), digest);
                self.recorded += 1;
            }
            GoldenMode::Compare => {
                self.observed.insert(label.to_string());
                let Some(by_device) = self.file.entries.get(label) else {
                    self.missing.push(label.to_string());
                    return;
                };
                match by_device.get(&self.device) {
                    Some(expected) if *expected == digest => self.matched += 1,
                    Some(expected) => self.mismatches.push(format!(
                        "{label}: digest {digest} != recorded {expected} on {}",
                        self.device
                    )),
                    None => {
                        self.unverified += 1;
                        if self.unverified <= 8 {
                            eprintln!(
                                "golden unverified {label}: no digest recorded for {} (file has {})",
                                self.device,
                                by_device.keys().cloned().collect::<Vec<_>>().join(", ")
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn observe(&mut self, label: &str, values: &[f32]) {
        let digest = digest_f32(values);
        self.observe_digest(label, digest);
    }

    pub fn observe_parts(&mut self, label: &str, parts: &[&[f32]]) {
        let digest = digest_parts(parts);
        self.observe_digest(label, digest);
    }

    /// The invocations the golden holds a digest for on this device: what this device has
    /// actually been verified against, whether or not the coverage record lists it.
    fn device_digests(&self) -> BTreeSet<String> {
        self.file
            .entries
            .iter()
            .filter(|(_, by_device)| by_device.contains_key(&self.device))
            .map(|(label, _)| label.clone())
            .collect()
    }

    /// Audit this run's coverage, or explain why it was not audited.
    ///
    /// `--bootstrap-coverage` is handled here, before the audit: it creates only the entry
    /// this device is missing, leaves every other entry exactly as recorded, and returns the
    /// printed diff as notes. The fail-closed rule still applies to everything else -- a
    /// device with an entry is audited, never rewritten.
    fn coverage_audit(&self) -> (Vec<String>, Vec<String>) {
        let Some(args) = &self.coverage else {
            return (Vec::new(), Vec::new());
        };
        if self.restricted {
            return (
                Vec::new(),
                vec![format!(
                    "{}: coverage was not audited because this run exercises only part of the harness",
                    self.key
                )],
            );
        }
        let mut notes = Vec::new();
        let mut existing = match read_coverage(&args.path) {
            Ok(Some(document)) => document.coverage,
            Ok(None) => {
                if !args.bootstrap {
                    return (
                        vec![format!(
                            "{} is missing; create it with --record-coverage on this machine, or \
                             --bootstrap-coverage to create only the entries this run is missing",
                            args.path.display()
                        )],
                        Vec::new(),
                    );
                }
                Vec::new()
            }
            Err(err) => return (vec![format!("{err:#}")], Vec::new()),
        };
        if args.bootstrap {
            let created = bootstrap_coverage(&existing, &self.key, &self.device, &self.observed);
            if !created.added.is_empty() {
                match write_coverage(&args.path, created.merged.clone()) {
                    Ok(_) => {
                        notes.extend(created.diff_lines());
                        existing = created.merged;
                    }
                    Err(err) => return (vec![format!("{err:#}")], notes),
                }
            } else {
                notes.extend(created.diff_lines());
            }
        }
        let recorded = existing
            .iter()
            .find(|entry| entry.golden == self.key && entry.device == self.device);
        let (mut problems, audit_notes) = coverage_problems(
            &self.key,
            &self.device,
            &self.observed,
            &self.device_digests(),
            recorded,
        );
        // A bootstrapped entry was created from this very run, so it cannot disagree with it;
        // the audit above is what proves that rather than assuming it.
        notes.extend(audit_notes);
        (std::mem::take(&mut problems), notes)
    }

    /// Merge this run's coverage into the record, refusing a shrink unless it was asked for.
    fn record_coverage(&self, allow_shrink: bool) -> Result<Vec<String>> {
        let args = self
            .coverage
            .as_ref()
            .context("coverage flags are resolved before recording")?;
        let existing = read_coverage(&args.path)?
            .map(|document| document.coverage)
            .unwrap_or_default();
        let merged = merge_coverage(&existing, &self.key, &self.device, &self.observed);
        if !allow_shrink {
            if let Some(refusal) = merged.refusal() {
                bail!("{refusal}");
            }
        }
        let document = write_coverage(&args.path, merged.merged)?;
        Ok(vec![format!(
            "recorded coverage for {} on {}: {} invocation(s); the record now holds {} entry(ies)",
            self.key,
            self.device,
            self.observed.len(),
            document.coverage.len()
        )])
    }

    /// Write the accumulated digests, or report the comparison outcome. Any
    /// mismatch, any invocation missing a digest for this device, and any loss of
    /// recorded coverage fails.
    pub fn finish(mut self) -> Result<()> {
        match self.mode {
            GoldenMode::Write => {
                let notes = if self.coverage.as_ref().is_some_and(|args| args.record) {
                    let allow_shrink = self.coverage.as_ref().is_some_and(|args| args.allow_shrink);
                    self.record_coverage(allow_shrink)?
                } else {
                    vec![format!(
                        "{}: coverage was not recorded; pass --record-coverage to merge this run's coverage into the record",
                        self.key
                    )]
                };
                self.file.schema = SCHEMA;
                let text = serde_json::to_string_pretty(&self.file)?;
                std::fs::write(&self.path, text)
                    .with_context(|| format!("writing golden file {}", self.path.display()))?;
                for note in &notes {
                    println!("note  {note}");
                }
                println!(
                    "GOLDEN RECORDED: {} invocation(s) for {} -> {}",
                    self.recorded,
                    self.device,
                    self.path.display()
                );
                Ok(())
            }
            GoldenMode::Compare => {
                for mismatch in &self.mismatches {
                    eprintln!("golden mismatch {mismatch}");
                }
                for label in &self.missing {
                    eprintln!("golden missing {label}: no digest recorded for any device");
                }
                // Recording replaces the audit, exactly as the training-state verifier's
                // `--record-coverage` does: a device that has no entry yet cannot audit,
                // and this is how its entry comes into existence.
                let (coverage_problems, coverage_notes) =
                    if self.coverage.as_ref().is_some_and(|args| args.record) {
                        let allow_shrink =
                            self.coverage.as_ref().is_some_and(|args| args.allow_shrink);
                        match self.record_coverage(allow_shrink) {
                            Ok(notes) => (Vec::new(), notes),
                            Err(err) => (vec![format!("{err:#}")], Vec::new()),
                        }
                    } else {
                        self.coverage_audit()
                    };
                for note in &coverage_notes {
                    println!("note  {note}");
                }
                let mut problems = coverage_problems;
                if self.strict && self.unverified > 0 {
                    problems.push(format!(
                        "{} invocation(s) have no digest recorded for {}; record them with --write-golden before enforcing --golden-strict",
                        self.unverified,
                        self.device
                    ));
                }
                if !self.mismatches.is_empty() || !self.missing.is_empty() {
                    problems.push(format!(
                        "{} golden digest mismatch(es), {} unrecorded invocation(s), {} verified, {} unverified on another device",
                        self.mismatches.len(),
                        self.missing.len(),
                        self.matched,
                        self.unverified
                    ));
                }
                if !problems.is_empty() {
                    bail!("{}", problems.join("\n"));
                }
                println!(
                    "PASS: golden digests verified for {} invocation(s) on {} ({} unverified on another device)",
                    self.matched, self.device, self.unverified
                );
                if let Some(args) = &self.coverage {
                    println!(
                        "PASS: {} invocation(s) exercised on this device match {} for {}",
                        self.observed.len(),
                        args.path.display(),
                        self.key
                    );
                }
                Ok(())
            }
        }
    }
}

/// Parse the golden flags a harness exposes. Returns `(compare, write, strict)`. The
/// coverage flags are resolved by `coverage_args`, which every harness calls as well.
pub fn golden_flags(arguments: &[String]) -> Result<(Option<PathBuf>, Option<PathBuf>, bool)> {
    let mut compare = None;
    let mut write = None;
    let mut strict = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--golden" => {
                compare = Some(PathBuf::from(
                    arguments
                        .get(index + 1)
                        .context("missing golden file path")?,
                ));
                index += 2;
            }
            "--write-golden" => {
                write = Some(PathBuf::from(
                    arguments
                        .get(index + 1)
                        .context("missing golden file path")?,
                ));
                index += 2;
            }
            "--golden-strict" => {
                strict = true;
                index += 1;
            }
            _ => index += 1,
        }
    }
    if compare.is_some() && write.is_some() {
        bail!("--golden and --write-golden are mutually exclusive");
    }
    if strict && compare.is_none() {
        bail!("--golden-strict needs a --golden file to verify against");
    }
    Ok((compare, write, strict))
}
