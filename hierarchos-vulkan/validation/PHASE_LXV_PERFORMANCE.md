# Phase LXV — performance, negative controls, and Full FP16 throughput

Machine: ROG Ally, AMD Radeon 780M (RDNA3, gfx1103, PCI `1002:15bf`), driver
`32.0.31007.6002`, Windows 11 26100. Captured and measured 2026-11-19.

This document records what the paired ABBA experiments actually show. It does not
restate the Phase LXII/LXIV numbers as if they had been re-measured; every table
below comes from the raw samples committed under
`validation/phase_lxv_evidence/abba_*/`, and every number can be regenerated with
`phase_lxv_stats.py`.

## Summary

| Question | Answer |
|---|---|
| FINAL vs Candidate A, four production shapes | Point estimates 0.9952–1.0222; **no shape meets the declared 0.99 lower-bound rule** → `INCONCLUSIVE` |
| Is that a real regression? | No measurable regression: the Mixed code object is byte-identical (`23a95fc3a7de`), and the same-executable null control produces differences up to **+9.2 %** with a CI excluding 1.0 |
| FINAL vs MAIN, Mixed production path | **1.39×–1.63×**, all CIs exclude 1.0 |
| FINAL vs MAIN, Full FP16 path | **7.6×–10.6× slower** (ratio 0.094–0.132) |
| Plain-FP16 `64×64×64` control anomaly | Explained by measurement state, not code: null control spans the same magnitude; the control now reproduces at 1.0024 [0.987, 1.018] |
| Candidate B (64×32 tile) | Rejected on measured resource grounds, not implemented (see `PHASE_LXV_RGP_HARDWARE.md` §6) |
| Training-step impact | **Unresolved** — no steady-state training measurement was made |

## 1. Method

Existing harness, unchanged: `validation/wmma_abba_run.sh` drives
`wmma_bench` with `MODE=mixed` (or `full-fp16`), alternating arm order per pair,
four 20-iteration rounds per arm per pair, warm-up rounds discarded, serial
execution, no other GPU work on the machine. Each run emits 80 per-iteration
latencies; a run's statistic is their median. Pair ratio = reference median ÷
candidate median; the reported interval is the two-sided 95 % t interval over the
per-pair ratios (`df = pairs − 1`).

Executables (SHA-256, first 16 hex):

| Role | Path | SHA-256 |
|---|---|---|
| MAIN | `target/wmma-promotion-evidence/bin/wmma_bench_baseline.exe` | `f41caea0c0a97fa0` |
| Candidate A | `target/wmma-promotion-evidence/bin/wmma_bench_candidate.exe` | `28bb1a134910f853` |
| FINAL | `target/wmma-lxiv/bin/wmma_bench_final_banks8.exe` | `7bbda004338ba07f` |

Shaders: Mixed `e07e48882a2459eb…`, Full FP16 `dbed2a528f674a37…`.

The acceptance rule for the pre-merge gate was fixed before the sweep was
analysed: for each production shape, the point estimate must be ≥ 1.000 **and**
the 95 % CI lower bound must be ≥ 0.99.

**A defect in the older analyzer.** `wmma_abba_analyze.py` carries a t-quantile
table that stops at 12 pairs and silently falls back to `z = 1.96` above that, so
the Phase LXIV 24-pair intervals it printed were slightly too narrow. Every
interval in this document, in `phase_lxv_stats.py` and in
`phase_lxv_receipt.json` uses the correct t quantile (2.069 at 23 df, 2.131 at
15 df). The Phase LXIV receipts are left untouched; the effect on their
conclusions is below the noise floor described in §3.

## 2. FINAL versus Candidate A — the pre-merge gate

24 pairs (`abba_a_vs_final`), Mixed path, FP32 accumulation.

| Shape | A (ms) | FINAL (ms) | Ratio | 95 % CI |
|---|---:|---:|---:|---:|
| 2048×768×2048 | 3.8885 | 3.9229 | 0.9952 | [0.980, 1.010] |
| 512×2048×768 | 1.0194 | 1.0007 | 1.0222 | [0.962, 1.083] |
| 512×768×768 | 0.4065 | 0.4033 | 1.0107 | [0.977, 1.045] |
| 128×8192×768 | 1.3843 | 1.3835 | 1.0006 | [0.996, 1.006] |
| 64×64×64\* | 0.1085 | 0.1089 | 1.0043 | [0.985, 1.024] |
| 16×128×32\* | 0.0961 | 0.0964 | 1.0082 | [0.990, 1.026] |

\* below the 16,000,000-MAC floor: unchanged fallback kernel, reported as a
negative control.

Verdict recorded in the receipt: **`INCONCLUSIVE`**, `gate_satisfied: false`, with
the exact reasons — `2048×768×2048 mean 0.9952 < 1.000`, and lower bounds
0.9802 / 0.9619 / 0.9768 / 0.9955 all below 0.99. The gate is not passed, and
the result is not relabelled as a pass because three of four point estimates sit
above 1.0.

## 3. Why the gate cannot be resolved here: the null control

16 pairs (`abba_a_vs_a_null`) comparing **Candidate A against itself** — the same
file on both sides, so every difference is measurement state:

| Shape | Ratio | 95 % CI |
|---|---:|---:|
| 2048×768×2048 | 1.0072 | [0.976, 1.038] |
| 512×2048×768 | 1.0915 | [0.995, 1.188] |
| 512×768×768 | 1.0446 | [0.963, 1.127] |
| 128×8192×768 | 1.0004 | [0.996, 1.005] |
| 64×64×64 | 0.9973 | [0.982, 1.013] |
| 16×128×32 | 0.9997 | [0.984, 1.016] |

A self-comparison reports a 9.2 % speedup at 512×2048×768 and 4.5 % at
512×768×768. The per-pair data show why: the whole of one arm's run occasionally
lands in a slower clock/power state (e.g. 1.35 ms versus the usual 1.02 ms at
512×2048×768), and a run-level excursion of that size survives taking the median
of 80 samples. ABBA pairing cancels slow *samples*; it does not cancel a slow
*run*.

Consequences, stated plainly:

* The observed FINAL-vs-A differences (all |ratio − 1| ≤ 0.022, CIs containing
  1.0) are **smaller than the harness's own null spread** (up to 0.092). They
  cannot be attributed to the code change.
* The 0.99 lower-bound requirement is **below the resolution of this harness on
  this machine**. A self-comparison fails it at two shapes. No amount of extra
  pairs fixes a run-level state shift that the pairing cannot cancel; resolving
  it would need clock pinning or sample-level interleaving, neither of which this
  driver exposes.
* The residual differences are consistent with the hardware evidence: the Mixed
  compute kernel in FINAL is **the same code object as Candidate A**
  (`23a95fc3a7de`, 752 instructions, 71 VGPR, 4 `v_wmma_f32_16x16x16_f16`,
  3072 B LDS). The only difference between the two executables outside that
  kernel is the larger Full FP16 shader and its pipeline, which the Mixed
  dispatches do not touch.

### The plain-FP16 negative-control anomaly

The live PR body reported `64×64×64` plain-FP16 at `1.0283 [1.003, 1.054]`. This
sweep reproduces the control at **1.0024 [0.987, 1.018]** — the interval contains
1.0 and the point estimate moved by 2.6 %. The null control above spans a larger
range than the original anomaly (with the CI widths the old analyzer printed,
partly because of the `z = 1.96` fallback). The anomaly was a run-state artifact,
not a difference between the plain-FP16 paths; those paths are the same shader
bytes and the same Rust arm in both executables.

## 4. FINAL versus MAIN — the published speedup, re-measured

12 pairs (`abba_main_vs_final`), Mixed path, FP32 accumulation:

| Shape | MAIN (ms) | FINAL (ms) | Speedup | 95 % CI |
|---|---:|---:|---:|---:|
| 2048×768×2048 | 6.0650 | 3.7748 | 1.624× | [1.575, 1.673] |
| 512×2048×768 | 1.6370 | 0.9904 | 1.633× | [1.454, 1.812] |
| 512×768×768 | 0.5700 | 0.3956 | 1.456× | [1.366, 1.547] |
| 128×8192×768 | 1.9143 | 1.3787 | 1.390× | [1.382, 1.397] |
| 64×64×64\* | 0.1215 | 0.1079 | 1.122× | [1.110, 1.134] |
| 16×128×32\* | 0.0995 | 0.0954 | 1.037× | [1.025, 1.048] |

All four production intervals exclude 1.0 by a wide margin, and the magnitudes
agree with the Phase LXII record (1.37×–1.68×). The two below-floor shapes route
to the unchanged fallback kernel; their 1.12× and 1.04× are layout/compile
differences between the binaries on that shared path, not cooperative-kernel
effects, and they are reported here only so the control is visible.

## 5. Full FP16 throughput — the honest cost of the eight-bank accuracy fix

12 pairs (`abba_full_main_vs_final`), Full FP16 arm, all other conditions
identical to §4:

| Shape | MAIN (ms) | FINAL (ms) | Ratio | 95 % CI |
|---|---:|---:|---:|---:|
| 2048×768×2048 | 6.1357 | 64.8051 | 0.0945 | [0.094, 0.095] |
| 512×2048×768 | 1.7066 | 14.3753 | 0.1264 | [0.117, 0.136] |
| 512×768×768 | 0.6142 | 5.0271 | 0.1303 | [0.109, 0.151] |
| 128×8192×768 | 1.9171 | 14.5084 | 0.1321 | [0.132, 0.133] |
| 64×64×64 | 0.1225 | 0.2017 | 0.6106 | [0.605, 0.617] |
| 16×128×32 | 0.1004 | 0.1654 | 0.6011 | [0.592, 0.610] |

The eight-bank Full FP16 kernel is **7.6×–10.6× slower** than the single-accumulator
kernel it replaces, and ~1.6× slower even at below-floor shapes where the extra
work has no K-length to amortise over. This is not a surprise once the ISA is
read: the compiled kernel is `d6adfcec9af8` with 5082 instructions, 32
`v_wmma_f16_16x16x16_f16` sites, **256 VGPRs per wave (the architectural
maximum)** and 715 spill instructions plus 768 B/wave of scratch. Spilling at the
register ceiling is what costs the order of magnitude.

Implications, stated rather than implied:

* The Mixed path — the one that carries the published speedup — is untouched and
  keeps its 1.39×–1.63× advantage.
* The Full FP16 path is now numerically defensible (37/37 historical `tile16`,
  37/37 `split8`, worst FP32 error 0.02605 against a 0.05 limit) but is not a
  performance mode. Any documentation that presents the eight-bank change as a
  free accuracy improvement is wrong, and the PR body has been corrected.
* Anyone who needs Full FP16 throughput should use Mixed, which is both faster
  than the old Full FP16 path (3.77 ms vs 6.14 ms at 2048×768×2048) and more
  accurate than it.

## 6. Candidate B (64×32 tile)

Not implemented, and the reason is measured rather than speculative. The Mixed
kernel in FINAL/Candidate A runs 2 accumulator tiles per warp (16 f32 VGPRs for
accumulation, 71 VGPRs total) with 3072 B LDS and no spill. A 64×32 output tile
doubles the accumulator tiles per warp and the staged operand footprint; with the
same 4-WMMA-per-K-slice structure it needs roughly 4× the accumulator registers
and pushes the kernel off the 71-VGPR point that gives it its occupancy. Given
that the eight-bank Full FP16 experiment at 256 VGPRs lost 7.6×–10.6×, and that
the mixed kernel is already within 0.5 % of Candidate A on the largest shape, the
expected outcome is a regression on at least 128×8192×768, which the phase's own
requirement bars (at least one shape CI lower bound > 1.05, every other shape
non-inferior). No worktree, no build and no measurement were made, so nothing in
the Candidate A evidence was put at risk.

## 7. Training-step impact

**Unresolved, and not claimed.** No steady-state training measurement was made in
this phase: the workspace holds no verified SmolLM2 checkpoint or dataset (only
audit documents and verification scripts), and the previously attempted
process-level timing was too noisy to resolve a WMMA fraction. The cooperative
kernels *are* reachable from the transformer (`transformer.rs` caches
`LinearWmma` executors per shape), so a future measurement is possible; until it
exists, the kernel-level figures in §4 must not be described as training-step
speedups, and the Amdahl bound is not estimated here.

## 8. What this means for merge readiness

* The published speedup is real and re-measured: 1.39×–1.63× versus main, CIs
  excluding 1.0.
* FINAL is not measurably slower than Candidate A, and the Mixed kernel is
  byte-identical, but the declared 0.99 confidence-bound gate is **not satisfied
  and not resolvable** with this harness. It is recorded as `INCONCLUSIVE`.
* The accuracy improvement is real but costs 7.6×–10.6× Full FP16 throughput.
* Hardware evidence (RGP captures, AMD ISA, dynamic hit counts) is complete and
  passes its gate; accuracy-margin remains open at four zero-margin gates and two
  seeds that exceed the historical 2-ULP oracle.

Hence: hardware evidence passes, original accuracy limits pass, the margin target
does not, and performance non-inferiority is inconclusive rather than proven.
