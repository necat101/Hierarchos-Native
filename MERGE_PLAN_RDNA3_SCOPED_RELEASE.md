# Merge plan — scoped RDNA3 release (Phase XXVIII review)

Read-only plan for reviewing `codex/rdna3-phase19-instrumented` against `main`.
Nothing here authorises a merge: the branch is prepared for review, and the merge
itself waits for explicit approval.

## 1. Identity

| | |
|---|---|
| Branch under review | `codex/rdna3-phase19-instrumented` |
| **Release qualification source** | `306c7db9afbf5de3840c0e6c63876c6c67ab18fc` — last code change; the SHA the sealed evidence describes |
| **Phase XXVII handoff commit** | `ad1b7b0feafe8674b6fc1b446f1a5a2adf159891` — documentation only |
| **Reviewed candidate tip** | tip of the branch at review time: `git rev-parse origin/codex/rdna3-phase19-instrumented`. Every commit after `306c7db` is documentation-only |
| Target | `main` = `origin/main` = `ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5` |
| Merge base | `ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5` |
| Commits in code range | **223** (`ead6cd3..306c7db`) |
| Files changed in code range | **4,200** — 4,123 added, 77 modified, 0 deleted |
| Commits including handoff docs | 224 (`ead6cd3..ad1b7b0`), 4,202 files |

Three revisions must not be conflated. `306c7db` is what was qualified and what the
receipts bind. `ad1b7b0` adds two root Markdown files and changes no code, evidence or
threshold. The tip at review time may be later still. **A CI run only proves anything
about the commit its `headSha` names**, so §6 records which SHA each run covers rather
than treating "CI is green" as a property of the branch.

The merge base equals the tip of `main`, so `main` is a strict ancestor of the
qualified branch: the integration is a **fast-forward**, not a merge commit. That is
the safest possible shape here — it introduces no reconciliation of divergent history
and makes rollback a single ref move.

## 2. What the range contains, by subsystem

| Subsystem | Files | Purpose |
|---|---|---|
| `hierarchos-vulkan/validation/release_parity/**` | 3,959 | Sealed evidence: Phase XV–XXV receipts, packed-cell matrices, unified release manifest and its 24 component copies |
| `hierarchos-vulkan/src/**` | 47 | WMMA admission and device policy, RWKV packed-cell kernels, `bench_ab`/`decision_records`, probes and benchmark bins |
| `hierarchos-vulkan/validation/*.py` | 105 | Producers, checkers, negative controls, oracle verifiers |
| `hierarchos-vulkan/shaders/**` | 14 | Isolated RDNA3 key-state variant, attention backward Q/K variants, WMMA GEMM, fp32→fp16 conversion |
| `.github/workflows/*.yml` | 6 | The four mandatory gates plus the self-hosted A/B and the AMD performance run |
| `hierarchos-native-cli`, `hierarchos-gui` | 8 | Vision runtime wiring and the multimodal Start surface exercised by the GUI evidence |
| Root contracts and audits | 13 | `RDNA3_RELEASE_CONTRACT.md`, `RDNA3_WMMA.md`, `RDNA3_MIXED_GUI.md`, `RDNA3_PRECISION_COMPLETION.md`, `MAIN_INTEGRATION_READINESS.md`, audit documents |

Added evidence in the code range (`ead6cd3..306c7db`) totals **477.6 MB across 4,123
files**, with 68 files above 1 MB. The
largest is `20261113-phase22-recovery/numerical-before-20261114.zip` at **59.6 MB**;
`evidence/unified/raw/primitives/operands.json` (15.6 MB) and
`evidence/unified/raw/fallback/model-export.json` (6.8 MB) are each stored twice because
every sealed root carries its own copy. That duplication is evidence, not code, and no
producer reads the older copy — but a reviewer should decide deliberately whether this
volume belongs on `main` before approving, because the merge is a fast-forward and the
archive cannot be dropped afterwards without deleting sealed evidence.

## 3. Sensitive files a reviewer must read

- `.github/workflows/{architecture-compatibility,phase19-instrumented,phase19y-oracles,workspace-check}.yml`
  — these define the gates that make the release claim; a change here changes what
  "green" means.
- `.github/workflows/time-mix-ab.yml` — self-hosted A/B, permanently queued without a
  runner (see §7).
- `hierarchos-vulkan/src/device_policy.rs` and the five `src/bin` plus
  `validation/*.rs` `#[path]` include sites — RDNA3-only admission.
- `hierarchos-vulkan/shaders/rwkv_key_state_forward_fused_wide.spv` (shared, byte-identical
  to pre-WMMA `ead6cd3`) versus `..._amd_rdna3.spv` (isolated fix).
- `hierarchos-vulkan/src/rwkv_time_mix_core.rs` — carries the restored portable
  composition-selection expression `measured || (qualified && variant)`.
- `hierarchos-vulkan/validation/emit_two_pass_evidence.py` — the attention parser fix
  described in `RELEASE_RESEALING_PROCEDURE.md`.
- `hierarchos-vulkan/validation/qualify_phase19{x,y,z}_current.py`, `check_*`,
  `check_unified_release.py`, `qualify_unified_release.py` — the sealed roots are
  literals inside these files.

## 4. Phase XV provenance discrepancy — adjudicated

One historical root in the range is not a clean regeneration. Full evidence:
`PHASE15_EVIDENCE_REBINDING_AUDIT.md`. The review-relevant summary:

- Commit `9e636fb` changed **one file**,
  `20261108-rdna3-phase15-qualification/current-qualification.json`, rewriting 32
  values: `source_head`, `producer_sha256`, five binary digests, **12 of 14** `evidence`
  digests, six `retained.*.qualification_sha256` values, and one numeric —
  `retained.gemma4.worst_logit` `1.6391277313232422e-07 → 1.4156103134155273e-07`.
- The **source binding is genuine**: all 134 `sources` digests match the tree at
  `9e636fb` exactly.
- The **regenerated evidence is not preserved**. Hashing every committed revision of
  each referenced path finds no revision that ever held the declared content (18 of 18
  mismatched bindings), and the family receipts still on disk carry the *old* digests
  and the *old* `1.639e-07`. The aggregate therefore contradicts the evidence it cites,
  and `verify.py` does not check the `evidence` map or the family hashes — which is why
  it went unnoticed.
- `SHA256SUMS` was refreshed (`b1d6f49`) to match the edited receipt, so the root is
  consistent as a manifest but not as a provenance chain.

Impact on this merge: none for the scoped release. The root is read only by
`emit_spirv_gate.py` (through `gates/spirv.json`) and by
`check_phase16_receipts.py` / `emit_phase16_spirv_gate.py`;
`20261115-phase25-current-source/release-qualification.json` contains **0** references
to it, and `verify.py` already fails on bound-source drift, so there is no current
passing state to protect.

This is a **historical evidence-integrity defect**, not a release blocker. The original
receipt is byte-recoverable at `9e636fb~1` (sha `7bbba8cd…`, identical to `origin/main`),
nothing was rewritten or deleted in this phase, and the justified corrective action is
an audit note plus a quarantine label — not restoration, which would rewrite history a
second time. A reviewer must consciously accept the branch with this known defect, or
require a follow-up root before merge.

No other file in the range deletes or rewrites sealed evidence: 0 deletions, and the
remaining 73 modifications are source, workflow, CLI/GUI, shader and documentation
files.

## 5. Release evidence and its bindings

`release-qualification.json` (`20261115-phase25-current-source`):

- `status: PASS_SCOPED_RELEASE`, `source_revision: 0779b3d…` (ancestor of the reviewed
  head, which is all the checkers require), 24 components, ~2,000 bound source files,
  33 binaries.
- `text166` — re-executed at `306c7db` scope; 166/166, max abs `1.788139343e-07`,
  `< 2e-07`, `rtol 0`, executable `ac675c4f…`.
- Phase XIX-X / Y / Z — executed in Phase XXV, checker-passing, bound to 397 / 1,577
  sources and their own evidence counts.
- `packed` — `phase19z/packed-cell/status.json`, 13/13 on `amd-rdna3-lds64`.
- `precision`, `wide`, `tests`, `gui_cli`, `wmma_*`, `shader727` (728/728),
  `shader_wmma` (4/4) — executed in Phase XXVI against the current binaries.

Independent verification at the reviewed head: `check_unified_release.py` →
`PASS_SCOPED_RELEASE`; `test_unified_release.py` → 33 passed.

## 6. CI evidence, per SHA

Independently read back through the GitHub API on 2026-11-15. Every run below is a
`push` event that concluded `success`.

**On the documentation handoff `ad1b7b0`:**

| Workflow | Run | Head | Created |
|---|---|---|---|
| Workspace check / unified release | 43699252188 | `ad1b7b0` | 2026-11-15T09:26:20Z |
| Phase XIX instrumented qualification | 43699252204 | `ad1b7b0` | 2026-11-15T09:26:20Z |
| Phase XIX-Y oracle bytes | 43699252175 | `ad1b7b0` | 2026-11-15T09:26:20Z |
| Phase XIX-W architecture compatibility | 43699252211 | `ad1b7b0` | 2026-11-15T09:26:20Z |

URLs follow `https://github.com/necat101/Hierarchos-Native/actions/runs/<id>`.

**On the qualified code candidate `306c7db`** (Phase XXVI final):

| Workflow | Run |
|---|---|
| Workspace check | 43693998223 |
| Phase XIX instrumented qualification | 43693998237 |
| Phase XIX-Y oracle bytes | 43693998268 |
| Phase XIX-W architecture compatibility | 43693998290 |

`ad1b7b0` differs from `306c7db` only by two root Markdown files, so its green runs
cover the same code and evidence; they do not re-qualify anything. Neither set validates
a later commit. When this review's documentation commit lands, read its own runs with
`gh run list --branch codex/rdna3-phase19-instrumented --limit 8`; this table is
deliberately not rewritten to chase a self-referential final SHA.

## 7. Outstanding qualifications

| Item | Status | Blocker classification |
|---|---|---|
| Self-hosted packed-cell producer A/B (`time-mix-ab.yml`) | **Pending** — no runner | Deferred. `GET /repos/necat101/Hierarchos-Native/actions/runners` returns `total_count: 0`; `producer-ab` requires `[self-hosted, windows, vulkan]` and `cross-adapter-ab` adds `amd`, both with tuned `linear3-forward`/`linear-residual-forward` modules. Resume procedure: `PACKED_CELL_AB_RUNNER_RUNBOOK.md`. It is a cross-check of the locally measured 13/13 matrix, not a substitute for it. |
| Physical Intel Gen9 | **Pending** — no host | Deferred platform qualification; Gen9 keeps the pre-WMMA FP32 selector, so no RDNA3 claim covers it. |
| Physical RDNA1/RDNA2 | **Pending** — no host | Deferred platform qualification; FP32 selector retained. |
| AVX2-host fixture reproduction | **Pending** — no host | Deferred; the release uses the frozen AVX2 oracle fixtures, not a live AVX2 host. |

None of these is a blocker for the **scoped RDNA3** release, which is what the sealed
receipt claims. They are blockers for any statement of universal or cross-platform
qualification, and no such statement is made.

## 8. Merge sequence (executes only on approval)

```bash
set -e
git fetch origin
git rev-parse origin/main                      # expect ead6cd3 (or re-base the plan if moved)
git merge-base --is-ancestor origin/main FETCH_HEAD && echo fast-forward-ok

# 1. Re-confirm the candidate is the reviewed commit and clean.
git switch --detach origin/codex/rdna3-phase19-instrumented
test -z "$(git status --porcelain --untracked-files=no)"

# 2. Re-run the local gates on the exact candidate.
python hierarchos-vulkan/validation/check_phase19x_current.py
python hierarchos-vulkan/validation/check_phase19z_current.py
python hierarchos-vulkan/validation/check_unified_release.py
python -m pytest hierarchos-vulkan/validation/test_unified_release.py -q
python hierarchos-vulkan/validation/check_main_compatibility.py
python hierarchos-vulkan/validation/check_shader_provenance.py \
  --manifest hierarchos-vulkan/validation/training_shader_manifest.json \
  --compiler-path "glslangValidator=C:/VulkanSDK/1.4.321.1/Bin/glslangValidator.exe"

# 3. Fast-forward main and push only after step 2 is green.
git switch main
git merge --ff-only origin/codex/rdna3-phase19-instrumented
git push origin main
```

Post-merge verification: `git rev-parse HEAD` equals the reviewed SHA; the four
workflows must run and conclude `success` **on `main`'s new head**; then re-run
`check_unified_release.py` and `check_main_compatibility.py` from a fresh clone of
`main` to prove the evidence is in the merged tree, not only in the branch worktree.

Do not delete `codex/rdna3-phase19-instrumented` after merging — it is the audit trail
the receipts name, and `codex/rdna3-wmma` (`0483bca`) must stay untouched.

## 9. Rollback

Because the integration is a fast-forward, rollback is a ref move that loses no work:

```bash
git switch main
git reset --hard ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5   # local only
git push --force-with-lease origin main                     # only on approval
```

`codex/rdna3-phase19-instrumented` still points at the qualified commit, every
`release_parity` root is additive and unchanged by the rollback, and
`codex/rdna3-wmma` is never involved. No evidence is deleted by reverting `main`.

## 10. Recommendation

**READY FOR REVIEW.** The scoped RDNA3 release is internally consistent: one sealed
manifest, checker and 33 tests passing, and four mandatory workflows green on both
`306c7db` and `ad1b7b0`. It is not `READY TO MERGE`: the Phase XV provenance defect in
§4 is a known, unresolved audit finding that a reviewer must accept explicitly, and the
self-hosted A/B (§7) has never executed. `MERGED` would only be claimed after an actual
merge.
