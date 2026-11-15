# Merge plan — scoped RDNA3 release (Phase XXXI final)

Read-only plan for merging `codex/rdna3-phase19-instrumented` into `main`. Nothing here
authorises a merge: the branch is prepared for review, and the merge waits for explicit
approval.

## 1. Identity

| | |
|---|---|
| Branch | `codex/rdna3-phase19-instrumented` |
| **Reviewed candidate** | `e8eb5316f09ee415ea2d5fadddd57d5acbe166cb` |
| Target `main` | `ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5` |
| Merge base | `ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5` = tip of `main` |
| Fast-forward | **yes** — `main` is an ancestor of the candidate |
| Commits in range | **226** |
| Files changed | **4,208** — 4,131 added, 77 modified, **0 deleted** |
| Added bytes | 477.7 MB across the added files; 68 files above 1 MB |

The merge base equals the tip of `main`, so this integrates as a **fast-forward** with no
merge commit, no reconciliation of divergent history, and a rollback that loses no work.

Three revisions are distinct and must not be conflated: the candidate `e8eb531` is what
the evidence describes; any documentation commit made after it (including the one
carrying this file) adds no code and no evidence; a CI run only proves something about
the commit its `headSha` names.

## 2. What the range contains

| Subsystem | Files | Purpose |
|---|---|---|
| `hierarchos-vulkan/validation/release_parity/**` | 3,959 | Sealed evidence: Phase XV–XXV receipts, packed-cell matrices, unified release manifest and its component copies |
| `hierarchos-vulkan/src/**` | 58 `.rs` | WMMA admission and device policy, packed-cell kernels, `bench_ab`/`decision_records`, probes and benchmark bins |
| `hierarchos-vulkan/validation/*.py` | 105 | Producers, checkers, negative controls, the digest guard |
| `hierarchos-vulkan/shaders/**` | 14 | Isolated RDNA3 key-state variant, attention backward Q/K, WMMA GEMM, fp32→fp16 |
| `.github/workflows/*.yml` | 7 | The four mandatory gates, the AMD performance run, the self-hosted A/B, and the Phase XXIX evidence-integrity gate |
| `hierarchos-native-cli`, `hierarchos-gui` | 8 | Vision runtime wiring and the multimodal Start surface |
| Root contracts and audits | 13 | `RDNA3_RELEASE_CONTRACT.md`, `RDNA3_WMMA.md`, `RDNA3_PRECISION_COMPLETION.md`, `MAIN_INTEGRATION_READINESS.md`, this plan, the resealing procedure, the Phase XV audit, the evidence audit, the A/B runbook |

## 3. Focused review of the sensitive contents

**3.1 Configuration files — 8 changed (7 workflows + `.gitattributes`).**
`architecture-compatibility.yml`, `phase19-instrumented.yml`, `phase19y-oracles.yml`,
`workspace-check.yml` are the mandatory gates; `rdna3-performance.yml` is the AMD
performance run; `time-mix-ab.yml` is the self-hosted A/B (pending, see §7);
`evidence-integrity.yml` is the Phase XXIX digest guard added on this branch.
`.gitattributes` sets line-ending policy. A change here changes what "green" means, so
these are the first thing a reviewer should read. All five workflows on the candidate
concluded `success`; the new guard passed on its first run.

**3.2 `src/device_policy.rs` — admission is narrow and device-independent where it must
be.** `wmma_hardware_eligible` accepts only vendor `0x1002` with an explicit PCI-ID list
(RDNA3 `0x73a8…0x749f`, `0x15bf`, `0x15c8`, `0x1900/1901`; RDNA3.5 `0x1114`, `0x150e`,
`0x1586`, `0x1902`; RDNA4 `0x7550/7551/7590`), sourced from AMD GPUOpen `device_info` at
a pinned commit. Vulkan cooperative-matrix features remain mandatory regardless of ID.
`composition_admitted` is recovered verbatim from pre-WMMA `main` so an Intel Gen9 or
RDNA1/2 dispatch decision cannot change. `wmma_requires_amd_rdna3_or_newer` rejects
Intel, NVIDIA, ARM and unknown-vendor devices; 6 tests pass.

**3.3 The isolated SPIR-V pair.** `shaders/rwkv_key_state_forward_fused_wide.spv`
(12,488 B, sha256 `8481b7c7…`) is **byte-identical to pre-WMMA `main`** — verified by
hashing the blob at `origin/main`. The RDNA3 arithmetic fix lives only in
`shaders/rwkv_key_state_forward_fused_wide_amd_rdna3.spv` (11,980 B, sha256
`8aab7de3…`), so other devices keep their existing bytes. The provenance manifest
requires 728 identical rebuilds including both artifacts.

**3.4 Sealed-root literals.** The sealed root `20261115-phase25-current-source` is a
literal in 19 validation scripts; the three assemblers name their sub-roots directly —
`qualify_phase19x_current.py:26`, `qualify_phase19z_current.py:28` and `:39`. The
Y assembler is `qualify_phase19y_full_attention.py` (there is no
`qualify_phase19y_current.py`); it takes `--output` and carries the root literal as well.
This is why advancing a root is a source change, not a flag (§3 of
`RELEASE_RESEALING_PROCEDURE.md`).

**3.5 Large artifacts — all required evidence, none removable.**
- `release_parity/multimodal_training/20261113-phase22-recovery/numerical-before-20261114.zip`
  (59.6 MB, 1,827 members) is the preserved pre-2026-11-14 numerical state, byte-verified
  against its own SHA-256 manifest; it is ZIPped because the unpacked Windows paths
  exceed 260 characters. Its README states it is historical evidence, not the refreshed
  qualification.
- 291 `.safetensors` files, 48.9 MB total (72 in the sealed root, whose whole tree is
  52 MB), are LoRA adapter weights and AdamW optimizer-state snapshots. Checkers read
  them directly — `check_phase19t_current.py` counts `required_safetensors`, the
  stability checks resume from them — so deleting or trimming them invalidates receipts.
- `evidence/unified/raw/primitives/operands.json` (15.6 MB) and
  `raw/fallback/model-export.json` (6.8 MB) exist twice because each sealed root carries
  its own copy; no producer reads the older one, but the copy is part of that root's
  manifest.

## 4. Historical findings a reviewer must accept

Both are historical-only, outside the current release's dependency closure, and neither
is rewritten or re-sealed.

- **Phase XV rebinding (`9e636fb`, 2026-11-09).** One receipt file, 32 values rewritten
  in place: `source_head`, `producer_sha256`, five binary digests, 12 of 14 evidence
  digests, six `retained.*.qualification_sha256`, and `gemma4.worst_logit`
  `1.6391277323e-07 → 1.4156103134e-07`. All 134 source hashes are genuine; the new
  evidence is not preserved — hashing every committed revision of each referenced path
  finds **18 of 18** bindings naming content that path never held, while the family
  receipts still carry the old digests and the old `1.639e-07`. Original recoverable at
  `9e636fb~1` = `origin/main` (`7bbba8cd…`). Details: `PHASE15_EVIDENCE_REBINDING_AUDIT.md`.
- **`20261108-rdna3-gpt2-current` (`3806c8c`).** One binding
  (`v41native/primitives.log`, `668eb9d8…`) names content no revision of that path held.
  Same failure class at one entry, found by the Phase XXIX audit
  (`EVIDENCE_INTEGRITY_AUDIT.md` §4).

Neither root is referenced by `release-qualification.json` (0 occurrences).

## 5. Release evidence, its bindings, and the integrity gate

`20261115-phase25-current-source/release-qualification.json`:
`status: PASS_SCOPED_RELEASE`, 24 components, 2,000 bound sources, 140 evidence files,
33 binaries; `source_revision: 0779b3d…` (an ancestor, which is what the checkers
require). Text166 166/166 at max abs `1.788139343e-07 < 2e-7` with `rtol=0`; XIX-X/Y/Z
checker-passing; packed-cell 13/13 on `amd-rdna3-lds64`; precision 20 steps; wide 9
results; shaders 728/728 and WMMA 4/4.

Independent verification at the candidate: `check_unified_release.py` →
`PASS_SCOPED_RELEASE`; `test_unified_release.py` → 33 passed; the Phase XXIX guard
(`check_evidence_integrity.py --latest-release --strict`) → 2,197 findings, **0
failures**; its 18 negative controls pass. The guard's own files sit **outside** the
sealed `source_files` closure, which is why adding them did not disturb the receipt.

## 6. CI on the exact candidate `e8eb531`

All `push` events, `2026-11-15T20:46:46Z`, all `success`:

| Workflow | Run |
|---|---|
| Workspace check / unified release | 43759967266 |
| Phase XIX instrumented qualification | 43759966990 |
| Phase XIX-Y oracle bytes | 43759967031 |
| Phase XIX-W architecture compatibility | 43759966978 |
| **Evidence integrity (new)** | 43759967292 |

URLs: `https://github.com/necat101/Hierarchos-Native/actions/runs/<id>`.

Any documentation commit after this point gets its own runs; read them with
`gh run list --branch codex/rdna3-phase19-instrumented --limit 8`. That commit contains
no code and no evidence change, so it cannot invalidate the qualification — only the
runs' head SHA changes.

## 7. Outstanding qualifications

| Item | Status | Classification |
|---|---|---|
| Self-hosted packed-cell A/B (`time-mix-ab.yml`) | **Pending** — `total_count: 0` runners | Deferred cross-check of the locally measured 13/13 matrix; resume with `PACKED_CELL_AB_RUNNER_RUNBOOK.md` |
| Physical Intel Gen9 | **Pending** — no host | Deferred platform qualification; Gen9 keeps the pre-WMMA FP32 selector |
| Physical RDNA1/RDNA2 | **Pending** — no host | Deferred; FP32 selector retained |
| AVX2-host fixture reproduction | **Pending** — no host | Deferred; frozen AVX2 oracle fixtures are used, not a live host |

None is a blocker for the **scoped RDNA3** release. They are blockers only for a
universal or cross-platform claim, which this branch does not make.

## 8. Merge sequence (executes only on approval)

```bash
set -e
git fetch origin
git merge-base --is-ancestor origin/main FETCH_HEAD && echo fast-forward-ok

git switch --detach origin/codex/rdna3-phase19-instrumented
test -z "$(git status --porcelain --untracked-files=no)"

python hierarchos-vulkan/validation/check_phase19x_current.py
python hierarchos-vulkan/validation/check_phase19z_current.py
python hierarchos-vulkan/validation/check_unified_release.py
python -m pytest hierarchos-vulkan/validation/test_unified_release.py -q
python hierarchos-vulkan/validation/check_evidence_integrity.py --latest-release --strict
python hierarchos-vulkan/validation/check_main_compatibility.py

git switch main
git merge --ff-only origin/codex/rdna3-phase19-instrumented
git push origin main
```

Post-merge: `git rev-parse HEAD` equals the candidate; the five workflows run and
conclude `success` **on `main`'s new head**; re-run the checker and the digest guard from
a fresh clone of `main` to prove the evidence is in the merged tree.

Do not delete `codex/rdna3-phase19-instrumented` — it is the audit trail the receipts
name — and `codex/rdna3-wmma` (`0483bca`) stays untouched.

## 9. Rollback

```bash
git switch main
git reset --hard ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5   # local only
git push --force-with-lease origin main                     # only on approval
```

Every `release_parity` root is additive, so the rollback deletes no evidence; the
qualified branch ref still points at the candidate; `codex/rdna3-wmma` is never involved.

## 10. Recommendation

**READY TO MERGE — AWAITING USER AUTHORIZATION.**

- No WMMA implementation gap remains in the declared RDNA3 scope (Phase XXX verified
  admission, `off`/`mixed`/`full-fp16`, FP32 masters and optimizer state, reduction
  limits and FP32 fallback, CLI `--amp` compatibility, GUI forwarding, and
  unsupported-device refusal).
- The candidate is a clean fast-forward of `main` with five green workflows and a
  digest-clean release closure.
- Accepted, documented, and outside the closure: the two historical evidence findings
  in §4, the four pending hardware qualifications in §7, and the intentionally scoped
  Full-FP16 package limits.

`MERGED` would only be claimed after an actual merge.
