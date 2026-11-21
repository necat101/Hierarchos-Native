# Phase LXXXII — PR #9 CI closeout, historical gate correction, clean-clone manifest verification and PR #8 cross-link

**Phase:** LXXXII
**Date:** 2026-11-21
**Repository:** `necat101/Hierarchos-Native`
**Release branch:** `codex/rdna3-p3-release-20261121`
**Corrective commit:** `c71272078658e7732a99cc67200d4bf4d08d4b91`
**Terminal status:** `MANIFEST VERIFICATION FAILED — RELEASE NOT MERGE-READY`

No kernel, shader, SPIR-V, selector, threshold, fallback, GPU tensor, timing
sample or historical Phase LXXVIII–LXXXI manifest was modified by this phase.
`main` was not touched, neither PR was merged, and nothing was force-pushed,
rebased or amended.

---

## 1. Starting state (re-fetched, not assumed)

| Item | Value |
|---|---|
| PR #9 head | `67278c3d5b2041d63697af199e12dd61272bb2a9` |
| PR #9 base | `main` @ `94c4db1f18e57680d691bcafc9dc6f5b0ab292f6` |
| PR #9 state | `OPEN`, `mergeable: MERGEABLE`, `mergeStateStatus: UNSTABLE` |
| PR #8 head | `4f115f2e3b968eabc42872769001e6c7b49f2e50` |
| PR #8 base | `81ac82fa972d84025129e7c3f0e58fdece7df786` |
| PR #8 state | `OPEN`, `mergeStateStatus: CLEAN` |
| Release branch head | `67278c3d5b2041d63697af199e12dd61272bb2a9` |
| Two-wave candidate | `bd7434ed8de2055f34dfd6fabdb94d2c2ccb2e1a` |

Failing workflows at that head, on both the `push` and `pull_request` events:
`phase lxv` (`44900715351` / `44900710218`), `candidate a`
(`44900715319` / `44900710282`) and `workspace check`
(`44900715389` / `44900710285`). `evidence integrity` passed.

## 2. Ending state

| Item | Value |
|---|---|
| PR #9 head | `c71272078658e7732a99cc67200d4bf4d08d4b91` |
| PR #9 state | `OPEN`, `mergeable: MERGEABLE`, `mergeStateStatus: CLEAN` |
| PR #9 base | `main` @ `94c4db1f18e57680d691bcafc9dc6f5b0ab292f6` (unchanged) |
| PR #8 head | `4f115f2e3b968eabc42872769001e6c7b49f2e50` (unchanged) |
| PR #8 state | `OPEN` (deliberately not closed, not retargeted) |
| Release branch head | `c71272078658e7732a99cc67200d4bf4d08d4b91` |

## 3. Ancestry: PR #8 and PR #9

Both PRs re-fetched and compared at the corrected head.

| Measurement | Value |
|---|---|
| Merge base | `4f115f2e3b968eabc42872769001e6c7b49f2e50` — exactly PR #8's head |
| PR #9 ahead of PR #8 | **6 commits** |
| PR #9 behind PR #8 | **0 commits** |
| Ancestor test | `git merge-base --is-ancestor 4f115f2e c7127207` → true |

```
$ git merge-base 4f115f2e3b968eabc42872769001e6c7b49f2e50 c71272078658e7732a99cc67200d4bf4d08d4b91
4f115f2e3b968eabc42872769001e6c7b49f2e50
$ git rev-list --count 4f115f2e..c7127207   # ahead
6
$ git rev-list --count c7127207..4f115f2e   # behind
0
```

The five release commits verified in Phase LXXXI, plus the one corrective
commit from this phase:

```
c712720 Scope the historical CI gates to their own revisions and add the P3 release gate
67278c3 Record the evidence commit hash and pin the evidence bytes
9e92b71 Add the Phase LXXXI release evidence
88a1146 Make the Phase LXXIX P3 epilogue the production Full FP16 WMMA module
bd7434e Add a selectable Phase LXXI two-wave Full FP16 WMMA variant
f97d8de Make the Full FP16 shader a selectable committed variant instead of a swap
```

PR #9 is the cumulative successor of PR #8: it contains PR #8's complete
history and diverges nowhere. Merging both separately is unnecessary when PR #9
is merged as the cumulative descendant.

## 4. The three CI failures, root cause by root cause

All three were the same class of defect: a **historical** gate reading the
**current** tree of a branch that inherits every later phase.

### 4.1 `phase lxv` — `FAIL: unapproved file changed by Phase LXV: .gitattributes`

*Root cause.* `.github/workflows/phase-lxv.yml` computed
`git diff --name-only b86c29e HEAD` and passed the whole result to
`check_phase_lxv.py --changed`. On the release branch that range contains every
later phase, so the release's own `.gitattributes` rule
(`phase_lxxxi_evidence/** -text`, required to keep the manifest valid across
Windows and Linux checkouts) and the whole Phase LXXXI evidence tree read as
unapproved Phase LXV changes.

*Historical range established from the record, not guessed.* `b86c29e` is the
commit immediately before the first Phase LXV commit, and `cf98af6` is the last
of the six Phase LXV commits (`9f554c6`, `3844385`, `32a54d8`, `216c449`,
`8673eb8`, `cf98af6`); `759da5c` (Phase LXVI) follows it. `git diff --name-only
b86c29e..cf98af6` returns **203 files**, every one of them inside the approved
prefixes (`hierarchos-vulkan/validation/`, `PHASE_LXV`, `.github/workflows/`).
`cf98af6` is therefore the unambiguous endpoint, and no commit was invented.

### 4.2 `candidate a` / `current-candidate` — protected source changed outside the qualified Phase LXIV paths

*Root cause.* `check_current_candidate.py` grades the Phase LXIV change against
a fixed protected-path inventory. The cumulative release legitimately adds
`wmma_gemm_f16_f16acc_twobank.{comp,spv}`,
`..._twowave.{comp,spv}`, `..._twowave_p3.{comp,spv}` and the P3/selector work
in `projection.rs` and `vulkan.rs`, none of which belong to Phase LXIV. The
job could never pass on this branch while remaining a Phase LXIV gate.

*Historical revision established.* `a70cef4c6629f715f98cd42fb3ab29d8cbb9d575`
is the commit that recorded the 37/37 Full FP16 result and bound the gate to the
eight-bank bytes — the last commit of the Phase LXIV series, immediately before
`b86c29e`. Running the unmodified checker in a detached worktree at that
revision prints:

```
PHASE LXIV CURRENT-CANDIDATE VALIDATION: PASS (36 path(s) changed against 81ac82fa972d,
Phase LXII f32acc artifacts byte-identical, 32 cooperative multiply-adds in the
eight-bank split module, 16,000,000-MAC floor and every tolerance unchanged,
full FP16 37/37 under the historical default oracle, recorded reproduction receipt
verified; no GPU timing is claimed by hosted CI)
```

### 4.3 `workspace check` / `Unified FP32 Mixed and scoped Full FP16 release`

*Root cause.* `check_unified_release.py::verify()` built `required_sources` with
`git ls-files` **against the current checkout** and required it to be a subset of
the receipt's historical `source_files`. The receipt declares
`source_revision = 2ef9b8d9642cbbc41dd6d3f3e41b84c14803fe5a`; its inventory is a
statement about that revision's bytes, exactly as the comment immediately below
the check said. The only delta was the six P3 shader files:

```
MISSING FROM RECEIPT (6)
  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twobank.comp
  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twobank.spv
  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave.comp
  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave.spv
  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.comp
  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.spv
```

## 5. Modifications

| File | Change |
|---|---|
| `hierarchos-vulkan/validation/check_phase_lxv.py` | Adds `PHASE_LXV_SCOPE_BASE` / `PHASE_LXV_ENDPOINT`, `historical_scope_paths()` and `--write-scope`. `check_scope()` is unchanged. |
| `hierarchos-vulkan/validation/test_phase_lxv.py` | Adds `PhaseLxvHistoricalScope` (6 controls). All 19 pre-existing tests (17 negative controls and 2 positives) are untouched. |
| `.github/workflows/phase-lxv.yml` | Splits the historical scope check from the evidence gate; the evidence gate now runs against the PR head with no scope file. |
| `hierarchos-vulkan/validation/check_p3_release.py` | **New.** Current-release gate for the Phase LXXXI P3 arm. |
| `hierarchos-vulkan/validation/test_p3_release.py` | **New.** 23 negative controls for the above. |
| `.github/workflows/candidate-a.yml` | Adds the `phase-lxiv-historical` job (detached worktree at `a70cef4`); repoints `current-candidate` at `check_p3_release.py`; `candidate-a` (Phase LXII) untouched. |
| `hierarchos-vulkan/validation/check_unified_release.py` | Derives the required inventory from `git ls-tree -r` at `document['source_revision']`; adds `post_receipt_required_sources()`. `source_provenance.verify_bindings()` unchanged. |
| `hierarchos-vulkan/validation/test_unified_release.py` | Adds 6 historical-provenance controls (98 tests total). |

### 5.1 Phase LXV: scope and evidence separated, gate not weakened

Two questions are now asked separately:

* **Scope** — `python check_phase_lxv.py --write-scope …` then
  `--changed …` over `b86c29e..cf98af6` only.
* **Evidence** — `python check_phase_lxv.py` with no scope file, against the PR
  head, so the P3/SPIR-V bytes, static ISA counts, code objects, RDNA3
  register/LDS/scratch facts, the eight-seed accuracy receipts, the recomputed
  margins, the dispatch accounting and the ABBA statistics are still re-derived
  on every run.

The scope control itself was not loosened: an unapproved path **inside the
range** is still rejected, and `test_unapproved_source_file` is unchanged.
No commit was guessed — both endpoints come from the committed record, and
`test_endpoint_is_a_committed_revision` asserts the base is an ancestor of the
endpoint.

The focused control requested for the inherited-diff case is
`test_later_release_commit_is_not_a_scope_violation`: it asserts that
`.gitattributes` and `phase_lxxxi_evidence/**` are **absent** from the
historical range and **present** in `cf98af6..HEAD`, that the historical range
passes `check_scope`, and that the descendant diff is exactly what used to be
misread as a violation.

### 5.2 Phase LXIV and the current release separated

`phase-lxiv-historical` runs the **unmodified** `check_current_candidate.py` and
its 28 controls in a detached worktree at `a70cef4` — the revision whose
protected-path inventory it actually describes.

`current-candidate` now runs `check_p3_release.py` on the release tree. It is not
a path allowlist; it pins bytes and structure:

| Claim | How it is enforced |
|---|---|
| Exact P3 artifact | source `4e448e3c…d12`, SPIR-V `2aff4123…7d2`, 23,280 B, 17,941 B source |
| P3 SPIR-V structure | SPIR-V 1.5, 4 cooperative multiply-adds, 3 cooperative matrix types |
| Runtime artifact identity | `include_bytes!` of the committed module; `P3 → WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV`; `shader_name()` = `wmma_gemm_f16_f16acc_twowave_p3`; identity derived from the loaded kernel's name and signature and the selected variant's own SPIR-V length |
| `FullFp16Variant::P3` selection | `#[default]` on `P3`; `from_env_value` defers to `Self::default()` |
| Qualification route not reachable | `FullFp16Variant` has exactly `Split8, TwoBank, TwoWave, P3`; `parse()` has exactly four `Ok(Self::…)` arms and rejects every other name |
| WMMA stays opt-in | `WmmaMode` default `Off`; unset `HIERARCHOS_WMMA_MODE` → `Ok(Self::Off)` |
| Profitability at every admission point | `wmma_shape_profitable` requires eligibility and `>= WMMA_MIN_PROFITABLE_MACS`; `ProjectionWmma::admits` calls it, the constructor and all three recorded-batch `.filter(...)` paths call `admits`, and `TransformerKernels::wmma_linear` calls it directly |
| Threshold unchanged | `pub const WMMA_MIN_PROFITABLE_MACS: u64 = 16_000_000;` and the `> 128 * 256 * 256` assertion |
| Safe routing preserved | `select_arm` returns `PlainF16Storage` in both ineligible branches; `WmmaMode::Off → None`; declined projections record `note_policy_fallback()` |
| Mixed unchanged | `wmma_gemm_f16_f32acc.{comp,spv}` bytes, `gemm_kernel(WMMA_GEMM_F32ACC_SPV, "mixed")`, `MixedFp32Accumulate`, three Mixed limits |
| Fallbacks available | `split8` (`dbed2a52…`), `two-bank` (`169ff486…`), `two-wave` (`1b007264…`) committed, embedded, each with its own `spirv()` arm, shader name and `HIERARCHOS_WMMA_FULL_FP16_VARIANT` |

The checker never runs an adapter and says so; the GPU qualification remains the
recorded Phase LXXIX/LXXXI RDNA3 evidence.

### 5.3 Unified release: revision-scoped inventory

`historical_required_sources(revision)` reads
`git ls-tree -r --name-only <revision> -- <five source directories>` and unions
the seven explicitly required historical files. It fails closed on a non-hex,
wrong-length, or unavailable revision.

`post_receipt_required_sources(revision, declared)` then walks the current-tree
inventory: any path the receipt does not bind is looked up **at that revision**.
Absent means "introduced later" and is accepted; present means the receipt lost a
binding and still fails with `unified runtime/source inventory incomplete`.

`source_provenance.verify_bindings()` still hashes every declared path's
committed blob at the declared revision, so a missing historical file, an altered
digest, a malformed digest or an unresolvable revision all still fail. The
receipt's `source_revision` (`2ef9b8d9642cbbc41dd6d3f3e41b84c14803fe5a`), its
2,004-entry `source_files` map and every component digest are unchanged.

## 6. Evidence that the historical gates remain enforced

| Gate | Evidence |
|---|---|
| Phase LXV scope still rejects | `test_unapproved_source_file` (unchanged) plus `test_unapproved_path_inside_the_historical_range_still_fails` |
| Phase LXV evidence still recomputes | The workflow runs the checker on the PR head with no scope file; 25/25 controls pass |
| Phase LXV range still resolvable | `test_unavailable_range_fails_closed` raises when Git cannot read the range |
| Phase LXII record untouched | `candidate-a` job unchanged, still materialises `7c49558` and passes |
| Phase LXIV record untouched | `phase-lxiv-historical` runs the original checker and all 28 controls at `a70cef4` |
| Unified receipt still fails closed | `test_missing_required_historical_source_still_fails`, `test_altered_historical_inventory_digest_still_fails`, `test_required_release_file_missing_from_the_historical_inventory_fails`, `test_invalid_source_revision_still_fails` (4 revisions) |
| Historical receipt not resealed | `test_new_release_files_do_not_invalidate_the_pinned_receipt` pins `source_revision` and the 2,004-entry inventory |

## 7. New P3-specific CI coverage

`current-candidate` on every push and PR now runs `check_p3_release.py` plus 23
negative controls, including the three the release contract calls out:

* `test_altered_p3_source_bytes` and `test_altered_p3_spirv_bytes`
* `test_removed_production_admission` (and the Transformer and recorded-batch variants)
* `test_unintended_qualification_route_exposure` (a fifth variant) and
  `test_qualification_name_accepted_by_the_selector`

The remaining controls cover: a P3 variant embedding or dispatching a different
module, a reassigned shader name, P3 losing its `#[default]`, a shrunk committed
inventory, an opt-out WMMA default, a moved 16-M-N floor, a predicate that admits
every shape, dropped shape eligibility, an ineligible shape reaching a
cooperative arm, a declined projection that stops recording its fallback, an
altered Mixed artifact, a removed two-wave fallback, altered two-bank bytes and
a removed fallback selection variable.

## 8. Unified historical source-provenance results

```
$ python hierarchos-vulkan/validation/check_unified_release.py
UNIFIED RELEASE: PASS_SCOPED_RELEASE (2ef9b8d9642c, 2004 sources verified at their
declared revision, 26 file(s) differ in this checkout)

$ python -m unittest discover -s hierarchos-vulkan/validation -p "test_unified_release.py"
Ran 98 tests in 78.9s
OK
```

Six new controls:

| Control | Proves |
|---|---|
| `test_new_release_files_do_not_invalidate_the_pinned_receipt` | The P3 files are absent from both the receipt and its revision, and are classified as post-receipt |
| `test_missing_required_historical_source_still_fails` | A file that existed at the revision cannot drop out of the inventory |
| `test_altered_historical_inventory_digest_still_fails` | An altered historical digest still fails |
| `test_required_release_file_missing_from_the_historical_inventory_fails` | `.gitattributes` is still a required historical binding |
| `test_invalid_source_revision_still_fails` | `ffff…`, `0000…`, a short id and a non-hex string all fail |
| `test_current_source_p3_validation_is_independent_of_provenance` | `check_p3_release.py` reads no historical receipt and still passes |

## 9. Clean-clone manifest verification — FAILED

### 9.1 Method

```
git clone --shared --no-checkout \
  --branch codex/rdna3-p3-release-20261121 \
  C:/Users/User/Downloads/Multimodal/Hierarchos-Native-lxxxi-release \
  C:/Users/User/Downloads/Multimodal/_lxxxii_clean
git sparse-checkout init --no-cone
# /phase_lxxxi_evidence/
# /hierarchos-vulkan/shaders/
# /.gitattributes
git checkout c71272078658e7732a99cc67200d4bf4d08d4b91
```

`--shared` keeps the object database by reference, so the clone costs only its
checkout (15.0 MB). The checkout is sparse because a full one (1,505.3 MB of
tracked content) would have taken free space below the 10 GiB floor; the full
committed inventory was therefore cross-checked with `git ls-tree` instead of a
materialised tree. `.gitattributes` is checked out at the root, so the
byte-preserving rules apply exactly as in a full clone.

A small independent script (not the producer, since deleted) recomputed every
digest from raw bytes and compared the manifest against the committed tree.

### 9.2 Results

| Measurement | Value |
|---|---|
| Actual commit SHA | `c71272078658e7732a99cc67200d4bf4d08d4b91` |
| `HEAD` equals the release commit | yes |
| Working tree | clean |
| Manifest entries | **612** |
| Entries verifying | **213** |
| Entries mismatching | **397** |
| Referenced files missing | **2** |
| Unexplained mismatches | **0** |
| Committed but not listed | 1 (`SHA256SUMS_phase_lxxxi.txt` itself) |
| Untracked files in the evidence tree | 0 |
| P3 source SHA-256 | `4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12` ✅ |
| P3 SPIR-V SHA-256 | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` ✅ |
| Free disk before / after | 10.34 GiB / 10.33 GiB |

### 9.3 The 397 mismatches are line endings

For each of the 397, `sha256(blob.replace(b"\n", b"\r\n"))` reproduces the
manifest digest **exactly**. The evidence was committed in `9e92b71`, *before*
`phase_lxxxi_evidence/** -text` existed; that rule arrived in `67278c3`, the same
commit as the manifest. Git normalised CRLF→LF on the way into the blobs, while
the manifest hashed the CRLF working-tree bytes. Content is intact; the stored
encoding is not what the manifest declares. The rule prevents *future*
conversion — it cannot restore bytes already stored.

### 9.4 The 2 missing files are gitignored

`analysis.log` and `final_rerun.log` are listed in the manifest and exist on the
producing host with **matching** digests, but `.gitignore:34` (`*.log`) means
they were never committed.

```
$ git check-ignore -v phase_lxxxi_evidence/analysis.log
.gitignore:34:*.log	phase_lxxxi_evidence/analysis.log
```

### 9.5 In the producing worktree the manifest is correct

```
worktree: entries 612 match 612 mismatch 0 missing 0
```

So the manifest truthfully describes the evidence; **the repository does not
contain those bytes**. The `.gitattributes` protection is present and active
(`git check-attr text` → `text: unset`), but it cannot repair history.

### 9.6 Why CI never caught it

No workflow or script references this manifest. The only mention is prose in
`PHASE_LXXXI_RELEASE_REPORT.md` telling a reader to run `sha256sum -c`.
`evidence integrity` verifies the release receipt's own bindings, not this file,
which is why it passes.

### 9.7 Not repaired here

Restoring the 397 CRLF blobs and adding the two logs would change Phase LXXXI
evidence files and exceed the authorized CI/documentation scope of this phase.
The deterministic repair, for the maintainer to authorize, is either:

1. re-add the 397 files (each verified byte-exactly by the manifest digest) and
   the two logs with `-text` in force, or
2. narrow the manifest to the bytes the repository actually stores and re-seal it.

Until then the manifest does not verify in any checkout, and the release should
not be merged.

## 10. Full CI status at `c71272078658e7732a99cc67200d4bf4d08d4b91`

| Workflow | Run | Event | Result |
|---|---|---|---|
| `phase lxv` | `44941733902` | pull_request | ✅ success |
| `candidate a` | `44941733879` | pull_request | ✅ success |
| `workspace check` | `44941733843` | pull_request | ✅ success |
| `evidence integrity` | `44941733845` | pull_request | ✅ success |
| `phase lxv` | `44941729721` | push | ✅ success |
| `candidate a` | `44941729697` | push | ✅ success |
| `workspace check` | `44941729731` | push | ✅ success |
| `evidence integrity` | `44941729710` | push | ✅ success |

Jobs:

| Workflow | Job | Result |
|---|---|---|
| `candidate a` | `candidate-a` (Phase LXII, unchanged) | ✅ success |
| `candidate a` | `phase-lxiv-historical` (**new**) | ✅ success |
| `candidate a` | `current-candidate` (P3 gate, **new content**) | ✅ success |
| `phase lxv` | `phase-lxv-gate` | ✅ success |
| `workspace check` | `workspace-check` | ✅ success |
| `workspace check` | `workflow-validation` (actionlint) | ✅ success |
| `workspace check` | `phase19t-historical` | ✅ success |
| `evidence integrity` | `digest-guard` | ✅ success |

The Windows runner logged the corrected inventory behaviour:

```
UNIFIED RELEASE: PASS_SCOPED_RELEASE (2ef9b8d9642c, 2004 sources verified at
their declared revision, 26 file(s) differ in this checkout)
```

Every reported status belongs to `c712720…`; all earlier failing runs are at
`67278c3…` or `9e92b71…`.

## 11. PR cross-links

| PR | Comment |
|---|---|
| #8 | https://github.com/necat101/Hierarchos-Native/pull/8#issuecomment-7068136585 — declares #9 the cumulative successor, records the merge base / ahead-behind table, and states that merging both separately is unnecessary |
| #9 | https://github.com/necat101/Hierarchos-Native/pull/9#issuecomment-7068142115 — links #8, records the same ancestry, explains all three fixes, and flags the manifest failure |

Neither PR was closed, retargeted, rebased or merged. PR #8 remains open for the
maintainer to close after merging #9.

## 12. Corrective commits

| Commit | Contents |
|---|---|
| `c71272078658e7732a99cc67200d4bf4d08d4b91` | The eight workflow / checker / test files below |

Baseline commit `67278c3d5b2041d63697af199e12dd61272bb2a9` and every earlier
commit are untouched.

## 13. Remaining limitations

1. **The manifest does not verify in any clean clone** (397 line-ending blobs,
   2 gitignored files). Unresolved by design; see §9.7.
2. **The clean clone was sparse**, forced by the 10 GiB disk floor. The full
   committed inventory was validated with `git ls-tree`, but a full checkout was
   not materialised. A sparse clone's `git status` cannot report modifications
   to files outside the sparse set.
3. **The complete Windows workspace-check job was not executed locally.**
   PowerShell 7 (`pwsh`) is not installed on this host, and a full cargo build of
   the four-crate workspace would have exceeded the free-disk budget. The job ran
   and passed on `windows-latest` in run `44941733843`; locally only its Python
   steps were run (`check_unified_release.py`, `test_unified_release.py`).
4. **No GPU was executed.** Every GPU claim in this phase rests on the recorded
   Phase LXXIX/LXXXI RDNA3 evidence; hosted CI executed no adapter.
5. The `phase_lxxxi_evidence` manifest is not gated by any workflow, so a future
   regression of this kind would remain invisible to CI. Adding such a gate
   requires fixing the manifest first.
6. `test_p3_release.py` and `check_p3_release.py` assert exact source text for
   the selector, policy and admission wiring; a deliberate refactor of those
   constructs requires updating the expectations in the same commit.

## 14. SHA-256 manifest

Digests of the corrective files as committed in `c71272078658e7732a99cc67200d4bf4d08d4b91`
(`git show <commit>:<path> | sha256sum`):

```
0df1e75a0d527caa088263654e4ad9806a8214b183547500cea7af5bf1bbed28  .github/workflows/phase-lxv.yml
69d9a6f9c39147c5b0f298d50f654a592627f32cfd108779057ca6ed62ce82cd  .github/workflows/candidate-a.yml
6dc207f9d895c019d997adffa1d934e3821b099099c0d398ecad3b5c17eea8cb  hierarchos-vulkan/validation/check_phase_lxv.py
65fc639df24550d7ca3903f644ba328022af7b9b2e258cf9dc590d75170430c4  hierarchos-vulkan/validation/test_phase_lxv.py
44a8ab787cf387f6354b4f16f739693a50bffd785b8d3f4e0d77775d77f1160e  hierarchos-vulkan/validation/check_unified_release.py
9ffeb4b69db65b1778161bbc7f96ab7d8a452fa02aa708db3f55eb6dd9881ac3  hierarchos-vulkan/validation/test_unified_release.py
2b85233c5b8f6de2d11acf164c79cf7ce533b7ba1276988563cb84163bb6f5a3  hierarchos-vulkan/validation/check_p3_release.py
66b01e89d0019da3ced5ea639a4f0f5f78caa6115e5f86421b727fb41ea5bee6  hierarchos-vulkan/validation/test_p3_release.py
```

Unchanged qualified P3 artifacts (re-verified independently in the clean clone):

```
4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.comp
2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2  hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.spv
```

## 15. Terminal status

```
MANIFEST VERIFICATION FAILED — RELEASE NOT MERGE-READY
```

All four workflows are green at `c712720…`, the historical gates are still
enforced, and the two PRs are cross-linked. The release is blocked solely by the
Phase LXXXI SHA-256 manifest, which references 397 line-ending-normalised blobs
and 2 files that were never committed, and therefore cannot verify in a clean
checkout.
