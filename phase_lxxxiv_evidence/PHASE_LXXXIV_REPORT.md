# Phase LXXXIV — Final v0.2.5 qualification, workspace cleanup and portable release

Status: **fresh RDNA3 accuracy and performance qualification PASS on the
qualified adapter.** The portable package build, its extracted-package acceptance
and the release assets are recorded in `BUILD_MANIFEST.json`, `SHA256SUMS` and
`hierarchos-vulkan/standalone/CLEAN_MACHINE_TEST.md`; they are produced from the
commit that contains this directory and are not restated here.

No kernel, shader, SPIR-V, selector, fallback, threshold, reference tensor,
numerical tolerance or production-dispatch path was modified in this phase. The
only source changes are documentation, the `.gitattributes`/`.gitignore` evidence
bindings, and this evidence directory.

---

## A. Source state

| Item | Value |
| --- | --- |
| Qualified source commit | `b165c2994e9ef5929c9e36fca76192db8dd3df35` |
| Branch | `codex/rdna3-p3-release-20261121` |
| Working tree at qualification | clean (only this evidence directory untracked) |
| Remote head of the release branch | `b165c299…` (fetch performed this phase; `git rev-list --left-right --count origin/…` = `0 0`) |
| `origin/main` | `94c4db1f18e57680d691bcafc9dc6f5b0ab292f6` |
| PR #9 | open, non-draft, `MERGEABLE`, merge state `CLEAN`, head `b165c299…` |

The remote head was unchanged from the previously reported state, so no new diff
review was required and nothing was built from stale `main`. No merge, history
rewrite, force-push or release tag was created; PR #9 remains unmerged and the
publishing step stays outstanding.

Qualification harnesses (archived in `bin/`, digests below):

| Artifact | SHA-256 |
| --- | --- |
| `bin/wmma_qualify.exe` (accuracy harness) | `138c2866af5eb499bf2e5c9e67505ea47dea9af2cfb7248ed6125b37e3773726` |
| `bin/wmma_bench.exe` (performance harness, release arm) | `6b291c1f9e19f625a516d4d25a0071db6f139f6101b41927ab0ed8ea7100ef12` |
| Preserved Main baseline `wmma_bench_main.exe` | `f41caea0c0a97fa017063de17e2f902722bcec6fa4c758362af2f504aa7538e0` |

`wmma_bench.exe` reproduced the documented release-binary hash exactly, and the
Main baseline reproduced its documented arm hash exactly.

### Harness byte-reproducibility, stated precisely

`wmma_bench` rebuilds byte-identically on this toolchain. `wmma_qualify` does
**not**: two consecutive `cargo build --release` runs from the same clean source
produced

- `0c2ef4007ba99942207f112fd8fcb94ba40431782b787e00148709eef8e6a01b`
- then `138c2866af5eb499bf2e5c9e67505ea47dea9af2cfb7248ed6125b37e3773726`
- then `0e48f99d14b3a108e4d2b6f8dfc583ac35f3dfb676055ddedaff1ed0e94c5d5f`.

All three are the same size (955,392 B) and differ in **20 bytes**: offsets 248
and 830,436–830,847. The PE debug directory begins at file offset 830,432
(84 bytes, `.rdata`) and the CodeView `RSDS` record follows it, so the entire
difference is link metadata (PDB signature and its path record), not code or
data. To prove that behaviourally rather than by inspection, the complete
accuracy matrix was executed twice, once with each of the first two binaries:
**all 40 receipts and all 8 operand dumps are byte-identical across the two
builds.** The archived harness used for the committed receipts is the second
digest above.

`check_unified_release.py` has a local-only guard, `if p.exists(): require(...)`,
that compares any locally present release executable against a digest recorded at
`source_revision 2ef9b8d9642c…` (a Phase 42 receipt). Because `wmma_qualify` is
not byte-reproducible, that guard fails on this host whenever a local release
build exists, and hosted CI never trips it because CI builds no binaries. The
gate suite was therefore run in the CI-equivalent condition — no local release
executable present — and passes; the guard itself is unaffected by this phase.

---

## B. Fresh accuracy sweep

Device: `AMD Radeon Graphics`, vendor `0x1002`, device `0x15bf`, RDNA3,
`PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`, Vulkan `1.4.344`, driver `2.0.388`,
driverInfo `26.10.07.06 (LLPC)`, Vulkan instance `1.4.341`, loader from
`C:\WINDOWS\system32\vulkan-1.dll`. Timestamps reported `timestamp_period_ns =
10.0` in every stream. This is the same adapter, Vulkan version and driver
version as the recorded Phase LXXXI qualification, so the phase is a fresh run on
the original qualification host, not a transfer to different hardware.

Admission and cooperative contract, from the production harness:

```text
WMMA admission: rdna3_or_newer=true probed=true
cooperative matrix: extension_exposed=true feature_supported=true memory_model=true
  subgroup_size_control=true enabled=true
  storage_16bit=true shader_float16=true driver_subgroup_size=64 configurations=11
  mixed (f16xf16+fp32) tile: 16x16x16 A=f16 B=f16 C=f32 scope=subgroup
  full-fp16 (f16xf16+f16) tile: 16x16x16 A=f16 B=f16 C=f16 scope=subgroup
```

Subgroup contract: forced subgroup size 32 (`WMMA_SUBGROUP_SIZE`) against a
driver default of 64.

### Full matrix — 8 seeds x 5 mode/oracle configurations, 37 gates each

Seeds: `1593426484, 42, 1337, 271828, 314159, 777, 8675309, 20261119`.

| Seed | full-fp16 / tile16 | full-fp16 / split8 | full-fp16 / dot2 | full-fp16 / split2 | mixed / tile16 |
| --- | --- | --- | --- | --- | --- |
| 1593426484 | 37/37 | 37/37 | 29/37 | 34/37 | 37/37 |
| 42 | 37/37 | 37/37 | 31/37 | 37/37 | 37/37 |
| 1337 | 37/37 | 37/37 | 31/37 | 36/37 | 37/37 |
| 271828 | 37/37 | 37/37 | 31/37 | 36/37 | 37/37 |
| 314159 | 37/37 | 37/37 | 33/37 | 36/37 | 37/37 |
| 777 | 37/37 | 37/37 | 32/37 | 36/37 | 37/37 |
| 8675309 | 37/37 | 37/37 | 32/37 | 35/37 | 37/37 |
| 20261119 | 37/37 | 37/37 | 31/37 | 34/37 | 37/37 |

Totals: **40 receipts, 25 with every gate passing, 15 with failing gates**,
confined to the two alternative-oracle diagnostic columns.

Dispatch accounting was identical in every one of the 40 receipts:
`cooperative=15 plain_f16=1 shape_fallbacks=0`.

### The 15 receipts with failing gates, and why they are not a release failure

Every failing gate is an **oracle-order diagnostic**, not a production gate. The
`--full-oracle` flag selects which accumulation order the observed P3 output is
compared against; it does not change the dispatched kernel, which is P3 in all
four Full FP16 configurations.

| Failures | Oracle | Gate |
| ---: | --- | --- |
| 8 seeds | rdna3-dot2 | `128x128x96 matched fp16 oracle vs FP32` |
| 8 | rdna3-dot2 | `128x128x96 wmma-full-fp16 vs matched oracle` |
| 8 | rdna3-dot2 | `128x128x96 trans_a=false trans_b=true vs oracle` |
| 8 | rdna3-dot2 | `128x128x96 trans_a=true trans_b=true vs oracle` |
| 7 | rdna3-dot2 | `16x1024x64 wmma-full-fp16 vs matched oracle` |
| 3 | rdna3-dot2 | `37x8192x48 wmma-full-fp16 vs matched oracle` |
| 2 | rdna3-dot2 | `128x128x96 trans_a=true trans_b=false vs oracle` |
| 2 | rdna3-dot2 | `64x48x40 trans_a=true trans_b=false vs oracle` |
| 5 | split2 | `128x128x96 trans_a=false trans_b=true vs oracle` |
| 3 | split2 | `128x128x96 trans_a=true trans_b=true vs oracle` |
| 3 | split2 | `128x128x96 wmma-full-fp16 vs matched oracle` |
| 1 | split2 | `128x128x96 trans_a=true trans_b=false vs oracle` |

Two facts make these diagnoses of the *oracles*, not of the release:

1. `128x128x96 matched fp16 oracle vs FP32` compares the **oracle with itself**
   (`fp32` reference against the oracle tensor) and never touches a GPU output.
   It fails on every seed for `rdna3-dot2`, so that accumulation order is itself
   outside the documented `0.05` FP32 bound at this fixture magnitude.
2. The historical Phase LXV record already contains exactly this profile, on the
   pre-P3 kernel: the committed receipts report 4–8 failing gates for
   `rdna3-dot2` at all eight seeds and 1–3 for `split2` at all eight. The
   alternative oracles have never been release gates; `check_phase_lxv.py`
   requires only `split8` and `mixed` to be clean at every seed, and requires the
   recorded seed to be clean under `tile16`.

The fresh numbers differ slightly from the historical ones (`split2`/seed 42 is
now clean, and `dot2` improves on five seeds) because the observed tensor is now
P3's single-rounding epilogue rather than the earlier multi-rounding reduction.
That is an improvement in the measured distance to each alternative oracle, not a
relaxation: no limit, oracle, reference or seed was changed.

### Production P3 route on the 32 preserved K=96 cases

`phase_lxxxiv_evidence/lxxxiv_production_smoke.json`, produced with **no
override**: `--wmma-mode full-fp16` and no variant or qualification flag, so the
production selector resolved P3.

| Check | Result |
| --- | --- |
| Inputs identical to the preserved operands | **32/32** |
| Production output identical to the Phase LXXIX GPU P3 tensor | **32/32** (0 differing elements) |
| Numerical gates, all eight seeds | **37/37, 0 failed** |
| Dispatch counts per seed | `cooperative=15 plain_f16=1 shape_fallbacks=0` |
| Hard numerical gate (`ULP_GATE_LIMITS`) | `0.03125`, unchanged |
| Maximum tile16 matched-oracle distance | `0.03125` — exactly at the gate, passing (`<=`) |
| Maximum `128x128x96` distance vs FP32 | `0.02548027` |
| 25 % tile16 headroom target | informational; not met (`0.0234375` required) |

Because all 32 output tensors are bit-identical to the Phase LXXIX tensors, every
derived Phase LXXIX metric stands unchanged — including the 31/32 improved and
0/32 worsened counts against REF64F16 and the `0.02593803` maximum REF64F16
error. Those figures are reproduced by identity of bytes, not recomputed from a
different reference.

The preserved reference is a local byte-copy of
`phase_lxxix_evidence/k96_input_output_hashes.json`, SHA-256
`df8a523249bee3db518070fa71bb25bc20b7d4bc66631f860eed17305e4cb14a`, identical
to the source file.

The profitability predicate excludes the `16x128x32` control: it is 65,536 MACs
against `WMMA_MIN_PROFITABLE_MACS = 16_000_000` (`src/wmma.rs:626`), applied at
`ProjectionWmma::admits` (`src/projection.rs:108`) and at its three recorded-batch
call sites. The floor is unchanged.

---

## C. Fresh performance sweep

Protocol: the existing GPU timestamp harness
(`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`,
`HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`), `--rounds 1 --iterations 16`, GPU
timestamp queries around the actual compute dispatch, three warm-up dispatches
dropped per shape by `parse_gpu_profile.steady_state_gpu_ms`, strictly serial
one-process-at-a-time execution, 24 paired blocks per shape, position-balanced
**ABBA/BAAB** ordering (12 quads: ABBA, BAAB, ABBA, …), two-sided 95 % paired
Student-t over per-pair ratios from the canonical
`hierarchos-vulkan/validation/abba_stats.py` (no normal-quantile fallback).
All 48 runs exited 0 and self-reported `timestamp_period_ns = 10.0`.

Ratio convention: Main ms / P3 ms, so > 1.0 means P3 is faster. Raw per-iteration
samples: `abba_main_vs_p3_balanced_24p/` (48 `*_samples.jsonl` files, 194 files
total, `SCHEDULE.json`, `index.tsv` with per-run executable digests).

### Protected production shapes

| Shape | Pairs | Main ms | P3 ms | Speedup | Corrected 95 % CI | Verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 2048x768x2048 | 24 | 5.4193 | 5.3109 | **1.0243** | [1.0159, 1.0326] | PASS |
| 512x2048x768 | 24 | 1.4945 | 1.3332 | **1.1246** | [1.1167, 1.1325] | PASS |
| 512x768x768 | 24 | 0.6088 | 0.5486 | **1.1163** | [1.0879, 1.1446] | PASS |
| 128x8192x768 | 24 | 1.8102 | 1.4260 | **1.2722** | [1.2584, 1.2860] | PASS |

Every protected shape passes independently: point estimate >= 1.0 **and**
corrected CI lower bound >= 1.0. No shape pools another, no average is used, and
no geometric-mean substitution is made.

Note on the fourth label: the historical summary table mislabelled this row
`512x768x768` and duplicated it. The fourth protected shape is **`128x8192x768`**,
which is what this sweep measured.

### Controls, recorded but not acceptance-bearing

| Shape | Pairs | Main ms | P3 ms | Speedup | Corrected 95 % CI | Note |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 64x64x64 | 24 | 0.0465 | 0.0420 | 1.1069 | [1.1004, 1.1134] | faster than Main |
| 16x128x32 | 24 | 0.0271 | 0.0355 | **0.7566** | [0.7504, 0.7627] | disclosed direct-kernel regression |

The `16x128x32` regression is preserved, not hidden. At 65,536 MACs it is
approximately 244x below the 16,000,000-MAC admission floor, so the production
route never dispatches WMMA there and the regression is excluded by the unchanged
predicate rather than by a reclassification.

These are **per-dispatch GEMM kernel timings**. No end-to-end training-step
speedup is claimed.

### Agreement with a second, independent campaign

An identical 24-pair campaign was run before the host was quiesced
(`presweep_stats.json`). It agrees within 0.007 on every ratio and passes the same
four shapes, e.g. `2048x768x2048` 1.0304 [1.0202, 1.0405] and
`128x8192x768` 1.2732 [1.2605, 1.2860]. Only the quiesced campaign is reported as
the result; the second campaign is retained to show the ordering effect is small
relative to machine load.

---

## D. Existing CI and local test results

`release_gates.txt` records every command, its full output and its exit status.
**26 of 26 commands exited 0.**

| Gate | Result |
| --- | --- |
| `check_p3_release.py` | PASS |
| `check_phase_lxv.py` | PASS |
| `check_phase_lxxxi_manifest.py --expect-manifest-sha256 a7979f4f…` | PASS — 612/612 working tree **and** 612/612 Git blobs, resolved revision `b165c299…` |
| `check_evidence_integrity.py --latest-release --strict` | PASS — 0 findings failures; latest sealed root `20261117-phase42-current-source` |
| `check_unified_release.py` | PASS (CI-equivalent condition, see A) |
| `check_main_compatibility.py` | PASS |
| `check_rdna3_multimodal_receipts.py` | PASS |
| `check_rdna3_gpt2_receipts.py --historical-only` | PASS |
| `check_phase19x_current.py`, `check_attention_phase19x.py`, `check_phase19y_full_attention.py`, `check_phase19z_current.py` | PASS |
| `check_phase19u_audit.py` | PASS |
| `render_ab_workflow.py --check` | PASS |
| `test_rdna3_workflow_paths`, `test_phase19x_current`, `test_wmma_drift_epoch_reuse`, `test_attention_phase19x_ci`, `test_phase19y_full_attention`, `test_phase19z_current`, `test_unified_release`, `test_phase19u_audit`, `test_rdna3_gpt2_receipts`, `test_phase16_receipts` | PASS |
| `pytest test_phase_lxxxi_manifest.py -q` | PASS |
| `pytest test_evidence_integrity.py -q` | PASS |

Rust and shader:

| Check | Result |
| --- | --- |
| `cargo test --lib` | **1,177 passed, 0 failed, 40 ignored** (74.51 s) |
| `cargo clippy --lib --no-deps` | exit 0, pre-existing warnings only |
| `spirv-val --target-env vulkan1.2` on the committed P3 module | clean |
| `glslc` regeneration of the committed module | byte-identical |
| `check-workspace.ps1` (four-crate junction workspace build) | **not run** — see limitations |

Artifact identity, re-derived this phase:

| Item | Value |
| --- | --- |
| P3 source SHA-256 | `4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12` |
| P3 SPIR-V SHA-256 | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` |
| P3 SPIR-V size | 23,280 B |
| Regenerated `glslc --target-env=vulkan1.2` SHA-256 | `2aff4123…` (identical) |
| Toolchain | shaderc v2023.8/v2025.3, spirv-tools v2025.3, glslang 11.1.0-1253 |

Production selection was observed, not inferred: `wmma_bench` reported
`full_fp16_variant: p3` and pipeline
`wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280`,
and the profile stream contained actual
`shader=wmma_gemm_f16_f16acc_twowave_p3` dispatch records. Global WMMA remains
opt-in: `WmmaMode::Off` is the process default and `check_p3_release.py`
re-verifies that.

---

## E. Storage audit

Command used: `git worktree list --porcelain` (11 registered worktrees, all
retained), `git status --short --ignored`, per-directory size reports, and
process checks for active builds/benchmarks (none during the sweep).

Removed — one directory, verified on every rule:

| Directory | Prior size | Why it was safe |
| --- | ---: | --- |
| `_lxxxiii_clean/` | **1,976 MB** | Standalone clone of `github.com/necat101/Hierarchos-Native` at `b165c299…`, i.e. the published release-branch head; not a registered worktree (absent from `git worktree list`); `git status --porcelain --ignored` showed **only** `__pycache__/` and `.pytest_cache/`, no uncommitted, untracked or user data; no reference to it anywhere in the release tree, evidence, or tool directories; deleting it cannot invalidate a receipt because no receipt cites it. |

Disk free: **3.78 GB → 5.58 GB**, i.e. **1.80 GB actually freed** (the 1,976 MB
directory minus filesystem overhead/report rounding).

Retained, with reasons:

| Directory | Size | Reason retained |
| --- | ---: | --- |
| `Hierarchos-Native/` | 23.9 GB | Canonical repository (holds `.git`); never a removal candidate |
| `Hierarchos-Native-lxxxi-release/` | 6.4 GB | The qualified release worktree; contains the protected `hierarchos-vulkan/target/release/*.exe` |
| `Hierarchos-Native-lxvii-bankreuse/` | 6.8 GB | Holds the original Phase LXXXI/LXXIX recovery source and `phase_lxxix_evidence/` (386 MB dump tree, preserved referents); not removed pending its own confirmation sweep |
| `Hierarchos-Native-phase48-probe/` | 13.3 GB | Registered worktree with no evidence sweep of its own performed here |
| `Hierarchos-Native-lxvi-k96/`, `-lxxii-test/`, `-lxxix-p3/`, `-phase47-20261117/`, `-phase48-20261117/`, `-wmma-lab-20261117/`, `-wmma-promotion/` | 1.5–6.0 GB each | Registered Git worktrees; removal requires `git worktree remove` plus a per-tree confirmation that was not in scope for this phase |
| `_lxxxii_clean2/` | 17.4 MB | Cited by path in `phase_lxxxii_evidence/PHASE_LXXXII_CI_CLOSEOUT.md:406`; small enough that removing it would risk a dangling evidence reference for negligible gain |
| `_lxxxiii_tools/`, `_lxxxii_tools/` | 0.1 MB | Contain the Phase LXXXII/LXXXIII verification reports (`restore_report.json`, `closeout.json`, `precommit.json`); unique provenance, too small to justify a risk |
| `lxv-rgp/` | 475 MB | Raw RGP capture material behind the Phase LXV hardware record |
| `release-v0.2.0/`, `release-evidence/`, `lxvi-k96/` | 413 MB / 0.8 MB / 0.1 MB | Prior release artifacts and evidence; ownership unchanged |

Deliberately **not** deleted, per the resealing procedure:
`hierarchos-vulkan/target/release/*.exe`,
`target/unified-oracle/Scripts/python.exe`, frozen fixture trees, oracle archives,
every `release_parity` root, the shared Cargo registry and the Rust toolchains.
No Git worktree metadata was removed by hand.

Note on disk pressure: the host was at 100 % capacity (3.8 GB free) at the start
of this phase. The single removal above was the only fully-verified candidate
available without a per-worktree confirmation sweep, and it was enough to run the
portable build.

---

## F. Limitations, stated plainly

- **`check-workspace.ps1` was not run.** It creates a four-crate junction
  workspace and builds every binary and target; on a host that was already at
  full capacity this would have consumed the free space reserved for the portable
  release build. Its component coverage is not lost: `cargo test --lib` (1,177
  tests) and `cargo clippy --lib --no-deps` ran in the release worktree, and the
  26 workflow gates above all passed. The four-crate workspace link check itself
  is **unverified this phase**.
- **`check_unified_release.py` ran in the CI-equivalent condition** (no local
  release executables). The local-only binary-digest guard is discussed in A and
  is not satisfiable on this host for a non-byte-reproducible binary.
- **No RGP capture or occupancy measurement.** No Radeon Developer Panel is
  installed; register/LDS figures in the historical record are static
  code-object facts.
- **No end-to-end training-step timing.** All performance numbers are
  per-dispatch GEMM kernel timings.
- **Other hardware is untested.** The qualification applies to `1002:15bf` only.
- **`cargo fmt --check` is not clean on the untouched baseline** and is not an
  enforced repository gate; nothing was reformatted here.
- The `16x128x32` direct-kernel regression is measured and excluded by policy,
  not fixed.
