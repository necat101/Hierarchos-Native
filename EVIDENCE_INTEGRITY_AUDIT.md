# Repository-wide evidence integrity audit

Scope: every qualification root in this repository that declares digest bindings.
Method: `hierarchos-vulkan/validation/check_evidence_integrity.py --audit-all`, run
offline against the working tree and Git history. Nothing in an audited root was
modified, restored or deleted.

```bash
python hierarchos-vulkan/validation/check_evidence_integrity.py --audit-all
python hierarchos-vulkan/validation/check_evidence_integrity.py --latest-release --strict
```

## 1. What was audited

- 3,822 JSON files under `hierarchos-vulkan/validation/release_parity` were parsed.
- **94 roots** declare at least one digest binding the guard understands — an
  `evidence`/`evidence_files` map, `retained.*.qualification_sha256`,
  `components[*].{path,sha256}`, or `sources`/`source_files`.
- 4 sealed unified releases exist; the newest is
  `multimodal_training/20261115-phase25-current-source/release-qualification.json`.
- Roots whose `evidence` block is a metadata summary
  (`{"file": "evidence.zip", "bytes": …, "files": …}`) are not digest roots and are
  not counted here; the guard distinguishes the two shapes.

## 2. Results

| Classification | Roots |
|---|---|
| Integrity verified (no byte mismatch, no missing artifact) | **81** |
| Integrity mismatch (declared bytes differ or never existed) | **4** |
| Evidence missing / unverifiable (no matching artifact under the root) | **11** |
| Provenance unsupported (declares binaries or producer digests with no tracked artifact) | **66** |
| Referenced by the current unified release | **1** (its closure is clean) |

These are not exclusive: a root can be integrity-mismatched and provenance-unsupported.
81 roots have no integrity finding at all. 66 of those still cannot show a producer
ran, because the executable digests they declare are not tracked in the repository —
that is reported as provenance, not as an integrity failure.

## 3. The 13 roots with integrity findings

| Root | Finding | Classification |
|---|---|---|
| `20261108-rdna3-phase15-qualification` | 12 evidence + 6 retained mismatches | **Provenance unsupported, historical** — the `9e636fb` rebinding; see `PHASE15_EVIDENCE_REBINDING_AUDIT.md` |
| `20261108-rdna3-gpt2-current` | 1 evidence mismatch (`v41native/primitives.log`, `668eb9d8…`) | **Provenance unsupported, historical** — same signature, single binding |
| `20261106-rdna3-gpt2-shapes` | 3 mismatches, 17 missing | Mixed: stale path bindings plus artifacts never committed |
| `20261113-phase22-recovery/top-before-classifier-refresh` | 6 mismatches, 326 missing | Unverifiable snapshot: the manifest was preserved, the evidence tree was not |
| `20261113-phase22-recovery/seal-before-avx512-host-scope` | 163 missing | Unverifiable snapshot |
| `20261113-phase22-recovery/seal-before-ci-exit-fix` | 163 missing | Unverifiable snapshot |
| `20261114-main-compatibility` | 164 missing | Unverifiable snapshot (`before-release-qualification.json` kept, evidence tree absent) |
| `20261108-rdna3-phase18-path-smoke-repair/gpt2` | 20 missing | Historical copy, evidence not carried |
| `20261108-rdna3-phase18-replay-reliability/gpt2` | 20 missing | Historical copy, evidence not carried |
| `20261112-phase19y-full-attention/historical` | 3 missing | Historical copy |
| `20261112-unified-release/phase19y/historical` | 3 missing | Historical copy |
| `20261113-phase22-recovery/top-before-classifier-refresh/y/historical` | 3 missing | Historical copy |
| `20261115-phase25-current-source/phase19y/historical` | 3 missing | Historical copy inside the current release tree; **not referenced** by its manifest (0 occurrences) |

Missing evidence in a preserved snapshot is not evidence of tampering: a snapshot that
kept its manifest but not its evidence tree is unverifiable, and the report says so
rather than promoting it to "verified".

## 4. Newly discovered anomalies

**`20261108-rdna3-gpt2-current`** (commit `3806c8c`, "Seal the current-source RDNA3
GPT-2 qualification") declares `668eb9d8…` for `v41native/primitives.log`. No committed
revision of that path ever held that content, and the log under the root hashes
differently. This is the Phase XV failure mode at a single binding. Severity is low — the
root is historical and nothing current depends on it — but it is recorded, not waived.

The other mismatch roots (`20261106-rdna3-gpt2-shapes`,
`top-before-classifier-refresh`) are mixed: some bindings are stale paths whose content
exists at an earlier revision, others name content never committed. They are
distinguished per binding by the guard's `history/declared-evidence-content` finding.

## 5. Current release

`--latest-release --strict` on `20261115-phase25-current-source`:

```
findings: 2197  failures: 0  valid: 2164  missing: 0  provenance: 33
```

- 24 components, 140 `evidence_files`, 2,000 bound sources and 33 binary declarations
  all resolve; every byte match is exact.
- The 33 binaries are reported as provenance-unestablished: they are built artifacts,
  not tracked files, so their digests cannot be re-derived offline. `check_unified_release.py`
  remains the authority that binds them to the recorded run.
- The historical sub-directory `phase19y/historical` inside that tree is outside the
  manifest's dependency closure (0 references) and does not affect the result.

**No current-release dependency defect was found.** No reseal is required.

## 6. Limits of this audit

- Audit mode does not re-hash bound sources (`--verify-sources` does). Source drift is
  expected for historical roots and would otherwise dominate the report; the release
  closure is always checked with sources included.
- Declared digests are SHA-256 content hashes. They are not Git object ids, and
  `git cat-file` cannot confirm them — the history check hashes each committed revision
  of the path instead. A conclusion drawn from `git cat-file` in an earlier draft of the
  Phase XV note was withdrawn on that basis; the byte-level contradiction is unchanged.
- A matching digest proves bytes, never execution. 66 roots declare binaries that no
  artifact can confirm, and the report keeps that separate from integrity.

## 7. Preservation

No audited root was altered. The Phase XV original remains byte-identical at `9e636fb~1`
and `origin/main` (`7bbba8cd…`); the rebound receipt (`8d4a4299…`) remains as committed.
Both are reachable, and `test_evidence_integrity.py::PhaseFifteenRegression` fails if
either the defect or the guard's detection of it disappears.
