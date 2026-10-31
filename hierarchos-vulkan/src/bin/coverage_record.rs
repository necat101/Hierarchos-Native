//! The coverage-record tool: the shared record engine, for drivers that are not Rust.
//!
//! The dense, reduction and attention harnesses audit their record in process
//! ([`hierarchos_vulkan::coverage_problems`]). `validation/verify_training_state.py` is a
//! Python driver, so it needs the same engine out of process rather than a second copy of
//! the audit and merge rules in Python -- which is what it used to have, and what let the two
//! records' shrink rules drift apart.
//!
//! Usage: one JSON request on stdin, one JSON response on stdout.
//!
//! ```text
//! {"mode": "audit" | "record", "file": "...", "bootstrap": false, "allow_shrink": false,
//!  "entries": [ ... training_state_coverage.json entries ... ]}
//! ```
//!
//! `audit` compares `entries` with the record and reports problems and notes; `record`
//! replaces them, refusing a shrunken entry unless `allow_shrink`. `bootstrap` (valid with
//! either mode) first creates the entries the record is missing and prints the diff, and
//! never rewrites an entry that exists.
//!
//! ```text
//! {"problems": [...], "notes": [...], "added": 0, "replaced": 1, "total": 2, "written": true}
//! ```
//!
//! The process exits 0 even when the audit fails: the request was answered. A non-zero exit
//! means the request itself was malformed, which is the only failure the caller cannot report
//! as a coverage problem.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use hierarchos_vulkan::coverage_record;
use hierarchos_vulkan::TrainingStateEntry;

#[derive(Deserialize)]
struct Request {
    mode: String,
    file: PathBuf,
    #[serde(default)]
    bootstrap: bool,
    #[serde(default)]
    allow_shrink: bool,
    #[serde(default)]
    entries: Vec<TrainingStateEntry>,
}

#[derive(Default, Serialize)]
struct Response {
    problems: Vec<String>,
    notes: Vec<String>,
    added: usize,
    replaced: usize,
    total: usize,
    written: bool,
}

fn load(path: &std::path::Path) -> Result<Option<Vec<TrainingStateEntry>>> {
    match coverage_record::read::<TrainingStateEntry>(path) {
        Ok(Some(document)) => Ok(Some(document.coverage)),
        Ok(None) => Ok(None),
        Err(err) => Err(err),
    }
}

fn run(request: Request) -> Result<Response> {
    // Validated before anything is read or written: a mistyped mode must not look like a
    // coverage problem the caller can report as a red run.
    if !matches!(request.mode.as_str(), "audit" | "record") {
        bail!(
            "unknown mode {:?}; expected audit or record (bootstrap is a flag, not a mode)",
            request.mode
        );
    }
    let mut response = Response::default();
    let mut existing = match load(&request.file) {
        Ok(Some(existing)) => existing,
        Ok(None) => {
            if !request.bootstrap {
                response.problems.push(format!(
                    "{} is missing; create it with --record-coverage on this machine, or \
                     --bootstrap-coverage to create only the entries this run is missing",
                    request.file.display()
                ));
                return Ok(response);
            }
            Vec::new()
        }
        Err(err) => {
            response.problems.push(format!("{err:#}"));
            return Ok(response);
        }
    };

    if request.bootstrap {
        let created = coverage_record::bootstrap(&existing, &request.entries);
        response.added = created.added.len();
        response.notes.extend(created.diff_lines());
        if !created.added.is_empty() {
            let document = coverage_record::write(&request.file, created.merged.clone())
                .with_context(|| format!("writing {}", request.file.display()))?;
            response.total = document.coverage.len();
            response.written = true;
            existing = created.merged;
        }
    }

    match request.mode.as_str() {
        "audit" => {
            let outcome = coverage_record::audit(&existing, &request.entries, &());
            response.problems.extend(outcome.problems);
            response.notes.extend(outcome.notes);
        }
        "record" => {
            let merged = coverage_record::merge(&existing, &request.entries, &());
            if let Some(refusal) = merged.refusal() {
                if !request.allow_shrink {
                    response.problems.push(refusal);
                    return Ok(response);
                }
            }
            response.replaced = merged.replaced.len();
            let added = merged.added.len();
            let document = coverage_record::write(&request.file, merged.merged)
                .with_context(|| format!("writing {}", request.file.display()))?;
            response.added += added;
            response.total = document.coverage.len();
            response.written = true;
        }
        _ => unreachable!("the mode is validated above"),
    }
    Ok(response)
}

fn main() -> Result<()> {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .context("reading the request from stdin")?;
    let request: Request = serde_json::from_str(&text).context("parsing the request")?;
    let response = run(request)?;
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
