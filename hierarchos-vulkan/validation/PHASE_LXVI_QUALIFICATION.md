# Phase LXVI — pre-merge qualification status

Recorded 2026-11-19 on the ROG Ally host: AMD Radeon 780M (RDNA3, `gfx1103`,
PCI `1002:15bf`), driver `32.0.31007.6002`, Windows 11 26100. Branch
`codex/rdna3-wmma-vector-load-20261118` at `cf98af6`.

This document says what was measured, what was recomputed, and what is still
missing. It does not upgrade a missing measurement into a claim.

## 1. What this phase changed

| Change | Status |
|---|---|
| One canonical paired-ABBA statistics implementation (`abba_stats.py`) | **done, tested** |
| Dated correction of every recomputable historical interval | **done** |
| K=96 accuracy study: independent simulation plus a GPU-tested schedule candidate | **done, candidate rejected, gap open** |
| Phase LXVI structural gate and its negative controls | **done** |
| RGP dynamic per-instruction hit counts | **not obtained** (GUI-only reduction) |
| Steady-state training-step measurement | **not obtained** (no verified checkpoint in this tree) |
| Full FP16 bank-reuse candidate | **not built** |

## 2. Hardware instruction execution (RGP)

| Item | Value |
|---|---|
| RGP | Radeon GPU Profiler 2.7.1.13, RDTS `2026-10-02-1872` |
| Adapter | AMD Radeon Graphics, RDNA3 `gfx1103`, PCI `1002:15bf` |
| Traced capture | `lxv-rgp/cap-final-full-trace.rgp`, 379,492,979 B, SQTT instruction tracing, `[Status] Done` |

The captures are real and their code objects were already extracted and
disassembled with AMD COMGR (Phase LXV). What is **not** present is the
GUI-side reduction of the SQTT stream into per-instruction hit counts. The
`.rgp` container is AMD's `AMD_RDF` v3 binary format — not a zip, not SQLite —
and the Radeon GPU Profiler ships no command-line analyser: `-m profiling` only
*captures*. Instruction timing, the Hit Count column, the Instruction Types
`WMMA` row and the Wavefront Statistics panel are Radeon GPU Profiler GUI
views.

Consequently this phase reports **no dynamic hit count**, and it does not
manufacture one from the static ISA or SPIR-V counts. `cap-final-full-trace.rgp`
is preserved with its hash so the reduction can be completed in the GUI without
recapturing.

What the committed static evidence does establish, and what it does not:

* 32 static `v_wmma_f16_16x16x16_f16` sites, eight bank branches x four tiles.
* The uniform selector `bank = slice & 7` means **four** WMMA instructions per K
  slice, not 32. For K=96 that is six active slices, banks 0–5, and 24 WMMA
  executions per output tile group.
* That is a control-flow expectation, not a measured hit count. It is not
  presented as one.

`phase_lxvi_evidence/rgp_instruction_timing.json` does not exist. The gate
reports its absence; it cannot be satisfied by a forged file, because any future
record must carry the capture SHA-256, the event and pipeline hash, the
single-shader-engine sampling scope, total and analyzed wavefronts, and
per-instruction counts, and must not present sampled counts as whole-GPU totals.

## 3. Training-step measurement

Not measured. The exact blocker:

* no SmolLM2-135M weights, tokenizer, configuration or dataset exist in this
  tree or in the local Hugging Face cache (the cache holds only DeepSeek-V4.1 and
  Kimi-K3 partial downloads);
* `wmma-lxi-checkpoint/` does not exist in any preserved worktree;
* the earlier SmolLM2 reference (batch 8 x sequence 8, seed 42, 451 cooperative
  dispatches) therefore cannot be re-established from preserved artifacts.

Downloading a fresh checkpoint is possible (huggingface.co answered 200), but a
training-step number produced from an unverified checkpoint acquired in this
phase would not be comparable to the earlier reference, and the disk headroom is
12 GiB. No training-step claim is made, and no Amdahl bound is estimated.

## 4. K=96 accuracy margin

### 4.1 Where the four zero-margin gates come from

They are measured against the **retained historical `tile16` oracle**: one FP16
accumulator rounded once per 16-wide K slice. The kernel no longer implements
that structure. It keeps eight FP16 banks and merges them with a pairwise tree.
For K=96, six banks hold one slice each and two hold exact zero.

The same fixture under the kernel's own `split8` contract, measured on this GPU:

| Gate | split8 receipt | limit | ULP |
|---|---:|---:|---:|
| `128x128x96 wmma-full-fp16 vs matched oracle` | 0.0078125–0.015625 | 0.03125 | 0.5–1 |
| `128x128x96 trans_a=true trans_b=false vs oracle` | 0.0078125–0.015625 | 0.03125–0.0625 | 0.5–1 |
| `128x128x96 trans_a=false trans_b=true vs oracle` | 0.0078125–0.015625 | 0.03125 | 0.5–1 |
| `128x128x96 trans_a=true trans_b=true vs oracle` | 0.0078125–0.015625 | 0.03125 | 0.5–1 |

The zero-margin readings belong to a reference model the kernel deliberately
stopped implementing. That is a model mismatch, not a rounding error.

### 4.2 Independent simulation

`phase_lxvi_k96_margin.py` reimplements the fixtures and nine reduction
structures. It predicts the *qualitative* result (no schedule or merge order
moves the K=96 gates materially) but not the ranking: it over-predicts the
current kernel's error against FP64 by about 10 % (0.0304 predicted vs 0.0334
measured) and it predicted that a stride-3 bank permutation would improve the
K=96 oracle gates. It does not. The simulation is therefore recorded as
exploration only, never as a promotion instrument.

### 4.3 The GPU-tested candidate

Isolated worktree `Hierarchos-Native-lxvi-k96` (detached at `cf98af6`), one
change: `bank = (slice * 3u) & 7u`.

| Item | Value |
|---|---|
| Baseline SPIR-V (unchanged source recompiled) | `dbed2a528f674a37fd6672ff83a786eb28804a77a85befd076acc80d701106fe` (reproduced byte for byte) |
| Candidate SPIR-V | `66516c496d067fee2093975f53ccc9d22653832f172ce6908f59e855d232c54e`, 37,388 B |
| Build flags | `-fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O` |
| Gates | 37/37 passed on all eight seeds |
| K=96 oracle comparisons vs the current kernel | **23 worse, 1 equal, 0 better** |

Decision: **REJECTED.** It degrades the very margin it was meant to improve.
The current eight-bank kernel remains the implementation; its source, SPIR-V and
receipts are untouched.

The 25 % headroom target (0.0234375) is **not met** and the gap remains open
explicitly. No limit, tolerance, seed, layout or historical reference was
removed or changed.

## 5. Full FP16 performance

Unchanged and still a real, mode-specific regression: 5,082 ISA instructions,
32 static WMMA sites, 256 VGPRs (architectural maximum), 4,608 B LDS, 768 B
scratch per wave, 715 spill instructions, and 7.6x–10.6x lower throughput than
the single-accumulator kernel.

No bank-reuse candidate was built in this phase, so there is no new resource or
throughput number to report and no claim that the regression is addressed. The
Mixed Candidate A path was not touched.

## 6. Statistical analyzer

The defect is fixed and consolidated:

* `abba_stats.py` is the single implementation. The t quantile is computed from
  the regularized incomplete beta function; there is no table and no `1.96`
  fallback on a live path. Degrees of freedom outside `[1, 100000]`, fewer than
  two observations, duplicate pair identifiers, missing arms, nonzero exit
  codes, empty sample arrays, malformed JSON and non-finite or non-positive
  timings all raise.
* `wmma_abba_analyze.py` and `phase_lxv_stats.py` both delegate to it.
* `test_abba_stats.py` checks reference quantiles at df 7, 11, 12, 13, 15, 23,
  31, 32, 33, 47 and 63 and the negative controls.
* `phase_lxvi_historical_ci.py` recomputes all 72 recomputable intervals from
  the preserved raw samples and writes `phase_lxvi_statistics_correction.json`
  plus `PHASE_LXVI_STATISTICS_CORRECTION.md`.

Result: intervals widen, as they must. One qualitative verdict changes — the
same-executable null control at `512x2048x768` no longer shows a "speedup"
(legacy CI `[1.0025, 1.1805]`, corrected `[0.9947, 1.1883]`), which strengthens
rather than weakens the Phase LXV conclusion that the harness cannot resolve
FINAL against Candidate A at this shape. The FINAL-versus-Candidate-A
non-inferiority verdict remains `INCONCLUSIVE`.

The Phase LXI/LXII/LXIV raw samples are not present in this tree, so those
intervals are not recomputed and are not used for any new decision. That is
recorded, not silently skipped.

## 7. Reproduction

| Command | Result |
|---|---|
| `python check_phase_lxv.py` | PASS (archived receipt verified, corrected intervals permitted through the correction record) |
| `python check_phase_lxvi.py` | PASS, with the three missing-evidence items listed |
| `python -m pytest test_abba_stats.py` | 18 passed |
| `python -m pytest test_phase_lxvi.py` | 14 passed |
| `python -m pytest test_phase_lxv.py test_evidence_integrity.py` | passed |
| `wmma_qualify --seed 42 --full-oracle split8` on the rebuilt branch binary | 37/37 gates, zero value differences against the committed receipt |

`test_candidate_a.py::test_positive_candidate_validates` fails in this worktree
at `cf98af6` **before any Phase LXVI change** as well, because that gate
materialises the Phase LXII revision (`7c49558`) into its own worktree; the
workflow `candidate-a.yml` does the same. It is not a Phase LXVI regression.

## 8. Verdict

* statistics: fixed, consolidated, recomputed, tested;
* K=96 margin: studied, one hardware-tested candidate rejected, target still not
  met, gap open;
* Full FP16: regression still present, no new candidate retained;
* RGP dynamic counts: not obtained;
* training step: not measured.

PR #8 stays open and unmerged.
