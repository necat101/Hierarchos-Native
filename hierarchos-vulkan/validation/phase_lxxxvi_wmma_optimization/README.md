# Phase LXXXVI — v0.2.6 WMMA optimization campaign

Status: **NO QUALIFIED OPTIMIZATION FOUND — v0.2.5 PRESERVED.** Two candidates were
implemented, compiled, measured on the qualified adapter and rejected. Neither is
committed to a production route; `main`'s P3 implementation, SPIR-V, selector and
policy are byte-unchanged and the v0.2.5 release gate passes unmodified.

No release, tag, version bump or merge was produced by this phase.

---

## A. Immutable baseline

| Item | Value |
| --- | --- |
| Repository | `https://github.com/necat101/Hierarchos-Native` |
| Release | `v0.2.5` |
| Tag object | `c4125f36f3ba9f1311a446d4e576a26af173fee5` |
| Tag target / merge commit | `d0f48c9637ac8d0fdfc1418c2b90de81196f9d4d` |
| `origin/main` at campaign start | `d0f48c9637ac8d0fdfc1418c2b90de81196f9d4d` (contains the tag target) |
| Campaign branch | `codex/wmma-v026-optimization`, created from `d0f48c9` |
| P3 source SHA-256 | `4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12` |
| P3 SPIR-V SHA-256 | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` |
| P3 SPIR-V size | 23,280 B |
| P3 runtime identity | `wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280` |
| Working tree at start | clean |

All four values were re-derived on this host, not copied from the release notes.
The archived v0.2.5 harness reported the qualified pipeline identity verbatim:

```text
full-fp16 variant: p3 (shader wmma_gemm_f16_f16acc_twowave_p3, pipeline
wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280)
```

### Shader build recipe (established, then used for every candidate)

`glslc -fshader-stage=compute --target-env=vulkan1.2 <src> -o <out>` reproduces the
committed P3 module **byte-identically** (`2aff4123…`, 23,280 B) with the installed
`shaderc v2023.8 v2025.3 / spirv-tools v2025.3 / glslang 11.1.0-1253`. Adding
`--target-spv=spv1.4/1.6`, switching to `glslangValidator`, or targeting
`vulkan1.3` all produce different bytes, so the flag set is pinned by measurement,
not assumption.

### Baseline harnesses (archived in `bin/`)

| Artifact | SHA-256 |
| --- | --- |
| `bin/wmma_bench_v025.exe` | `8fcdc4a8453b2cdb529fe895130e2ec94f3b13a1ba00b15ebd264d2f1704862b` |
| `bin/wmma_qualify_v025.exe` | `9761361bdb5f5e31ef8f2f7f0b67f115aea05ba99f129b6a0be0950c0f734169` |
| `bin/p3_preserved.spv` | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` |

Built with `CARGO_TARGET_DIR=target-lxxxvi-baseline` from the unmodified `main`
tree, so no candidate build can overwrite them. Candidate builds used
`target-lxxxvi-cand`; the committed P3 module was restored and re-hashed after
every single candidate build, and each candidate executable had to report its own
`spirv-fnv1a64` signature before it was allowed to be measured.

### Baseline numerical qualification, re-run on this host

`run_lxxxiv_accuracy.py` (the preserved Phase LXXXIV driver, unmodified) over the
original 8 seeds × 5 mode/oracle configurations:

| Seed | full-fp16 / tile16 | dispatch accounting |
| --- | --- | --- |
| 1593426484 | 37/37, 0 failed | `cooperative=15 plain_f16=1 shape_fallbacks=0` |
| 42 | 37/37 | same |
| 1337 | 37/37 | same |
| 271828 | 37/37 | same |
| 314159 | 37/37 | same |
| 777 | 37/37 | same |
| 8675309 | 37/37 | same |
| 20261119 | 37/37 | same |

Totals: 40 receipts, 15 with failing gates, confined to `full-fp16/rdna3-dot2` and
`full-fp16/split2` — exactly the documented historical diagnostic profiles, whose
gate counts are of the same order as the recorded Phase LXV/LXXXIV runs. No
production gate fails, no reference, tolerance, oracle or seed was changed, and
`ULP_GATE_LIMITS` remains `0.03125`.

---

## B. Environment, stated precisely

| Item | v0.2.5 qualification | this host, this campaign |
| --- | --- | --- |
| Adapter | AMD Radeon Graphics | AMD Radeon Graphics |
| PCI ID | `1002:15bf` | `1002:15bf` (`VEN_1002&DEV_15BF`) |
| Architecture | RDNA3 / Phoenix | RDNA3 / Phoenix |
| Driver | `2.0.388`, packaging `26.10.07.06 (LLPC)` | `2.0.416`, packaging `26.20.15.14 (LLPC)` |
| Vulkan (device apiVersion) | `1.4.344` | `1.4.359` |
| Cooperative tile | `16x16x16` f16xf16+f16, scope=subgroup | identical (11 configurations advertised) |
| Subgroup | forced 32 (driver default 64) | identical |

**The driver is not the version the release was qualified on.** Every number in
this record is a *paired* comparison of two executables measured minutes apart in
the same driver session, so the driver difference does not invalidate a ratio; it
does mean these absolute milliseconds are not comparable to the release notes'
absolute milliseconds. That is stated rather than glossed.

### Profiler availability

- **RGP is available on this host** (Radeon Developer Tool Suite 2026-10-02-1872,
  recorded and used by Phase LXV). This campaign did **not** produce a new RGP
  capture: extracting a code object's AMDPAL metadata and ISA from an `.rgp`
  requires the COMGR RDF tooling that Phase LXV built outside the Git tree
  (`lxv-rgp/` is not present in this checkout).
- **No register, LDS, spill or occupancy figure is reported in this record for the
  P3 module.** None was measured. Everything attributed below is attributed from
  GPU-timestamp ablation measurements and from the shader's own structure; where a
  hypothesis depends on occupancy it is labelled a hypothesis.

---

## C. Profiling: what was actually measured

Accepted protocol for every measurement in this phase (unchanged from Phase
LXXIII/LXXXI/LXXXIV):

* GPU timestamp queries around the real compute dispatch
  (`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, `PROFILE_SAMPLE_STRIDE=1`);
* `--rounds 1 --iterations 16`, three warm-up dispatches dropped per shape;
* strictly serial execution, one process at a time, no competing GPU work;
* predeclared **ABBA / BAAB** quad schedule, position-balanced;
* corrected two-sided 95 % Student-t interval over per-pair ratios from the
  canonical `validation/abba_stats.py` (regularized incomplete beta, no
  normal-quantile fallback);
* identical input tensors and workload geometry on both arms;
* exact executable and shader signatures recorded per run in `index.tsv` /
  `*_meta.json`.

Ratio convention: `v0.2.5 P3 ms / arm ms`; **> 1.0 means the arm is faster**.

### The four ablations' question

Static instruction counts could not explain the production kernel's absolute time:
a staging-instruction model predicts ~0.5 ms for the `2048×768×2048` shape at this
GPU's issue rate, against 5.01 ms measured. Rather than guess, three ablation
modules were compiled from the qualified P3 source and measured against it. They
are **profiling instruments with wrong numerical output**, and are preserved in
`ablations/` as such.

| Ablation | What it removes | SPIR-V SHA-256 |
| --- | --- | --- |
| `abl_staging_only` | cooperative load, all 4 multiply-adds, the whole bank-sum epilogue | `6899b603…` |
| `abl_no_global` | the four global operand reads (replaced by values derived from the same address; every index computation, bounds test, LDS store, barrier, multiply-add and epilogue survives) | `6f9cd6c7…` |
| `abl_no_epilogue` | the per-pair LDS round trip, float32 bank sum and its barriers | `93168bd6…` |

`abl_no_epilogue` could not be built inside the time budget after two interrupted
runs (`ablation_build_abl_no_epilogue.log`), so its **two completed sibling
ablations carry the conclusion**; it is recorded as not measured rather than
implied.

Measured (4 paired blocks per ablation; every pair a full ABBA/BAAB quad leg):

| Shape | P3 ms | staging-only ms | ratio | no-global ms | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2048×768×2048` | 5.0132 | 3.9877 | 1.259 | 1.8764 | **2.678** |
| `512×2048×768` | 1.2475 | 0.8822 | 1.419 | 0.5249 | **2.363** |
| `512×768×768` | 0.5503 | 0.2729 | 1.943 | 0.3006 | **1.846** |
| `128×8192×768` | 1.3898 | 1.4284 | 0.972 | 0.5527 | **2.566** |

### Bottleneck conclusion (measured)

1. **The cooperative multiply-add and the P3 epilogue together are ~20 % of the
   time** on the largest shape: deleting both leaves 3.99 of 5.01 ms.
2. **The global operand staging path is 60–73 % of the time**: deleting only the
   four global reads — keeping every index computation, bounds test, shared-memory
   store, barrier, multiply-add and epilogue — makes the kernel **1.85×–2.68×
   faster**.
3. It is **not DRAM bandwidth**. All four protected shapes run at 549–645 GMAC/s
   regardless of whether their operands fit in cache. `128×8192×768` has a 196 KB
   A operand (L2-resident) and streams only B, yet runs at the same GMAC/s as the
   4 MB-A shape. Bandwidth would separate those two; the measurement does not.
4. It is **not shared-memory or instruction issue** either: the staging-only
   ablation keeps all 1,024 shared stores and all 32 barriers per workgroup-K-slice
   and is only 26 % faster than P3 on the largest shape, while removing the loads
   alone is 168 % faster.

The consistent reading is that the per-K-slice staging round trip is **latency
exposed**: each 16-deep slice issues eight dependent 32-bit loads per thread,
waits, then hits two workgroup barriers, and the workgroup has only two wave32
groups with which to cover that latency.

### Optimization inventory (ranked before any change)

| Candidate | Suspected bottleneck | Supporting evidence | Potential benefit | Numerical risk | Measurement |
| --- | --- | --- | --- | --- | --- |
| Wider output tile per workgroup (64×32, 4 waves) | staging work and barriers per MAC; too few waves to hide load latency | staging = 80 % of time; 1,024 staging halves + 2 barriers per 16,384 MACs | −25 % staging halves/MAC, −50 % barriers/MAC, 2× waves | none — arithmetic per output tile untouched | 24-pair ABBA vs v0.2.5 |
| Trim LDS (`lean`): epilogue stages through dead A/B tiles | workgroup residency — more concurrent workgroups cover the exposed latency | 4-wave (17.4 KB LDS, 3 WGs/CU) regressed to 0.58–0.73×, i.e. residency, not width, is the shortage | 9,216 B → 6,144 B LDS at unchanged shape, count and registers | none — same values, order, barriers | 8-pair screen, then 24-pair ABBA |
| Double-buffered single-barrier K pipeline | barrier-serialised loads with no overlap of slice *s+1* loads against slice *s* MMA | same ablations | −50 % barriers, overlapped latency | none | not attempted — see §F |

Double buffering was ranked third because a one-slice lookahead covers only the
duration of two `v_wmma` instructions, roughly 1/10 of one L2 round trip, and
because it *raises* LDS — which the first two candidates' results identify as the
scarce resource.

---

## D. Candidate 1 — `fourwave` (64×32 tile, 128 threads): REJECTED

**Hypothesis.** Halving the staging work and barrier count per MAC and doubling the
waves per workgroup would attack the measured staging bottleneck.

**Implementation.** The P3 source with the workgroup widened to four wave32 groups,
`a_tile[4][256]`, `b_tile[2][256]`, `c_tile[4][256]`, `bank_tile[8][256]`,
`bank_sum[8][256]`, `m0 = wg.y * 64`, 17,408 B LDS. Bank assignment, slice stream,
per-bank accumulation order, MMA operands and the single-rounding P3 epilogue are
unchanged by construction, so outputs were required to be bit-identical.

| Item | Value |
| --- | --- |
| SPIR-V SHA-256 | `a85b15abdd86a5bd2116b46f95d2943304e0c9720ce8c540222de9933860caf5` |
| SPIR-V size | 23,344 B |
| FNV-1a64 | `0x574e051d008ffd6c` |
| Harness exe | `bin/wmma_bench_fourwave.exe`, `21435d3978c917dc2299162e747925fdca9b1e8dd3208087d1cecc44ccd563cc` |
| `spirv-val` | clean |

**Result — 24 paired blocks, corrected 95 % t CI:**

| Shape | P3 ms | fourwave ms | speedup | 95 % CI | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| `2048×768×2048` | 5.0342 | 7.5082 | 0.6713 | [0.6688, 0.6738] | **regression** |
| `512×2048×768` | 1.2446 | 1.8469 | 0.6730 | [0.6695, 0.6765] | **regression** |
| `512×768×768` | 0.5343 | 0.9263 | 0.5791 | [0.5613, 0.5970] | **regression** |
| `128×8192×768` | 1.4140 | 1.9133 | 0.7287 | [0.7185, 0.7388] | **regression** |
| `64×64×64` (control) | 0.0386 | 0.0361 | 1.0695 | [1.0683, 1.0707] | — |
| `16×128×32` (control, below 16M-MAC floor) | 0.0322 | 0.0321 | 1.0027 | [1.0010, 1.0045] | — |

Numbers below 1.0 mean slower. All four protected shapes regressed.

**Rejection reason.** Not marginal and not noise: 1.4–1.7× slower with intervals
that exclude 1.0 by a wide margin. The result also *refutes* the first hypothesis
in an informative direction: the workgroup-level savings (−25 % staging halves,
−50 % barriers per MAC) are outweighed by having fewer independent workgroups
(768 vs 1,536) and fewer resident per CU (17,408 B of LDS admits ~3 of 64 KB
against 7 for P3). The candidate's arithmetic was never in question; it was
rejected before its accuracy sweep, and no accuracy claim is made for it.

**Source not retained.** The rejected module's `.comp` was deleted when the
`FullFp16Variant` entry was reverted; its SPIR-V digest, size, FNV, harness
executable digest and full measurement are preserved here and the executable can
still be re-measured. This is a gap in reproducibility for this one rejected
candidate and is stated rather than papered over.

---

## E. Candidate 2 — `lean` (P3 with the epilogue scratch removed): REJECTED

**Hypothesis.** If residency is what the kernel is short of (candidate 1's
inference), then removing dead LDS — without changing the workgroup shape, count,
register budget or arithmetic — should raise residency and reduce exposed latency.

**Implementation.** The P3 source with `c_tile[2][256]` and `bank_tile[4][256]`
(3,072 B) deleted; the pair epilogue stages through `a_tile[wave]` / `b_tile[wave]`,
which are dead between the end of a pair's K loop and the start of the next one's
staging. LDS per workgroup: 9,216 B → 6,144 B. Staging loops, barriers, bank
assignment, slice stream, MMA operands, accumulation order and rounding points are
untouched, so the output was required to be bit-identical.

| Item | Value |
| --- | --- |
| Source SHA-256 | `a75db2d3de7de7ae6d1c49412dde1f67d48580960b0e4e7750b88e2e3f8b335b` |
| SPIR-V SHA-256 | `fdafb2d6b45b5ff1679db60019decda3aae8b7bd447239dcaafabe56ea4e7b99` |
| SPIR-V size | 23,172 B |
| FNV-1a64 | `0xcf2949a95f7eee92` |
| Harness exes | `bin/wmma_bench_lean.exe`, `bin/wmma_qualify_lean.exe` |
| `spirv-val` | clean |
| Preserved source/binary | `candidates/wmma_gemm_f16_f16acc_twowave_p3_lean.comp` / `.spv` |

### Accuracy — PASS, and stronger than the gate

* Eight seeds × five mode/oracle configurations: **40 receipts, 15 with failing
  gates, confined to `full-fp16/rdna3-dot2` and `full-fp16/split2`** — identical
  to the v0.2.5 baseline in count and location. Production `full-fp16 / tile16`:
  **37/37 on all eight seeds, 0 failed**. Dispatch accounting
  `cooperative=15 plain_f16=1 shape_fallbacks=0`, unchanged.
* `ULP_GATE_LIMITS` unchanged at `0.03125`; no reference, oracle, seed or
  tolerance altered.
* **Production outputs are bit-identical to v0.2.5 P3**: 120 of 120 production
  cases (15 cases × 8 seeds) matched **every element** of the preserved v0.2.5
  GPU output, compared per case with no aggregation (`compare_dumps.py`). Inputs
  are verified equal on both sides, so the comparison is of the kernel, not of the
  fixture. There is no per-case regression to bound because there is no differing
  element at all.
* Small-workload controls keep their behaviour: `16x128x32` (65,536 MACs, ~244×
  below the unchanged 16,000,000-MAC floor) still takes the fallback path —
  dispatch accounting is unchanged and the floor is untouched.

### Performance — 24 paired blocks, corrected 95 % t CI

| Shape | P3 ms | lean ms | speedup | 95 % CI | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| `2048×768×2048` | 5.0144 | 4.7429 | **1.0601** | [1.0569, 1.0632] | faster, CI lower > 1.0 |
| `512×2048×768` | 1.2504 | 1.1669 | **1.0730** | [1.0674, 1.0786] | faster, CI lower > 1.0 |
| `512×768×768` | 0.5507 | 0.4769 | **1.1485** | [1.1087, 1.1883] | faster, CI lower > 1.0 |
| `128×8192×768` | 1.4012 | 1.4629 | **0.9628** | [0.9533, 0.9724] | **significant regression** |
| `64×64×64` (control) | 0.0386 | 0.0389 | 0.9925 | [0.9909, 0.9940] | slightly slower |
| `16×128×32` (control, below floor) | 0.0321 | 0.0324 | 0.9903 | [0.9880, 0.9927] | slightly slower |

At the tighter sample the regression is smaller than the 8-pair screen suggested
(`0.9628` vs `0.9439`) but its interval still excludes 1.0 by a wide margin and its
lower bound is **below the `0.98` floor** the acceptance rule sets for a protected
shape.

**Rejection reason.** The release bar is: at least one protected workload ≥ 3 %
faster with its corrected CI lower bound above 1.0 — which `lean` meets three
times over, including `+6.0 %` on the largest shape — and *no other protected
workload may regress*, with every other CI lower bound at or above `0.98`.
`128×8192×768` regresses 3.7 % with an upper bound of `0.9724`, i.e. slower with
high confidence. The candidate is therefore rejected despite being numerically
bit-identical and faster on three of four protected shapes.

The likely mechanism, stated as hypothesis because it was not counter-measured:
raising resident workgroups raises the aggregate streamed working set. That shape
streams a 12.6 MB B operand while its A operand (196 KB) is L2-resident; the three
shapes that improved are the ones whose operands are re-read from cache far more
often per byte. Whatever the mechanism, the measured outcome is a real, shape-
dependent regression and that is what the acceptance rule forbids.

The module is **not** committed to any production route: the `FullFp16Variant`
entry was reverted and the source/SPIR-V are preserved only under `candidates/` as
rejected artifacts. `git diff` against `d0f48c9` is empty.

---

## F. What was not done, and why

* **No RGP capture, no ISA read-out, no occupancy/VGPR/LDS measurement for P3.**
  RGP is installed but its RDF/COMGR extraction tooling from Phase LXV is not in
  this checkout. Every occupancy statement above is labelled an inference from
  measurement, never a hardware counter.
* **No double-buffered pipeline.** After candidates 1 and 2 established that LDS
  per workgroup is the scarce resource, a pipeline that adds 4,096 B of LDS to
  halve the barrier count is the wrong trade on this evidence; it would also have
  been a second, larger structural change on top of a rejected direction.
* **No FP32, Mixed, split8, two-bank or two-wave change.** Their modules, routes
  and limits are untouched.
* **No hardware admission change.** `1002:15bf` remains the confirmed boundary;
  nothing in this record claims any other device.
* **No end-to-end training timing.** Every number is a per-dispatch GEMM kernel
  time.
* **`abl_no_epilogue` was not measured** (build interrupted twice).
* **No v0.2.6 tag, version bump, release asset, merge or push.**

---

## G. Dispatch, fallback and production-contract verification

Run against the restored tree, after every candidate was reverted:

| Check | Result |
| --- | --- |
| `validation/check_p3_release.py` | **PASS** — P3 source and SPIR-V are the qualified bytes (23,280 B, SPIR-V 0x00010500, 4 cooperative multiply-adds over 3 cooperative types), 4 committed variants, WMMA opt-in with `off` as the external default, 16,000,000-MAC predicate at every admission point, plain/FP32 fallbacks intact, Mixed route unchanged, split8/two-bank/two-wave still committed and selectable |
| Global WMMA default | `WmmaMode::Off`, unchanged |
| Full FP16 selection | still explicitly opt-in, unchanged |
| P3 as default Full FP16 variant | unchanged |
| Profitability floor | `WMMA_MIN_PROFITABLE_MACS = 16_000_000`, unchanged |
| Reduction / K limits | unchanged |
| Per-batch admission | unchanged; dispatch accounting re-observed as `cooperative=15 plain_f16=1 shape_fallbacks=0` on every seed |
| Small-workload `16×128×32` | below the floor, still falls back (no cooperative dispatch) |
| Production result without any override | P3, reported as `spirv-fnv1a64:0x076216ad97d7605f/len:23280` |

The candidate measurements did use `HIERARCHOS_WMMA_FULL_FP16_VARIANT` to select a
non-default module; that is the repository's established comparison mechanism and
is not part of any production result. No test-only or override-based behaviour is
claimed as the production result, because no candidate became the production
result.

---

## H. Conclusion

Two genuinely different, individually justified attacks on the measured bottleneck
were implemented and measured. The wider-tile candidate made the kernel 1.4–1.7×
slower on every protected shape. The LDS-trimmed candidate was numerically perfect
— bit-identical on 120/120 production cases — and faster on three protected shapes
(`2048×768×2048` +5.5 %, `512×2048×768` +7.1 %, `512×768×768` +17.0 %) but
regressed `128×8192×768`, which the acceptance bar forbids.

A kernel that is faster on the shapes you benchmarked and slower on the one you did
not is exactly the outcome this campaign's acceptance rule exists to reject. The
v0.2.5 P3 implementation is therefore preserved unchanged, and no v0.2.6 candidate
is justified by this phase's evidence.

The measured inventory in §C remains the durable result: the cooperative
multiply-add and epilogue are ~20 % of the kernel, the global staging path is
60–73 %, and the exposure is latency, not bandwidth or issue. A future phase that
wants a real win should measure occupancy directly (RGP + COMGR on this host) and
target the staging latency with evidence about how many workgroups actually
reside — not with a wider tile, which this phase measured and rejected.

---

## I. Reproducing this record

```bash
cd hierarchos-vulkan
export E=validation/phase_lxxxvi_wmma_optimization

# baseline harnesses (archived, verified by hash)
sha256sum $E/bin/wmma_bench_v025.exe $E/bin/wmma_qualify_v025.exe

# shader recipe: must reproduce 2aff4123…
glslc -fshader-stage=compute --target-env=vulkan1.2 \
  shaders/wmma_gemm_f16_f16acc_twowave_p3.comp -o /tmp/p3.spv

# baseline accuracy (8 seeds x 5 configs + production dumps)
python ../phase_lxxxiv_evidence/run_lxxxiv_accuracy.py \
  --exe $E/bin/wmma_qualify_v025.exe --out $E/baseline_accuracy

# ablation measurements
python $E/run_balanced.py $E/ablation/abl_staging_only \
  $E/bin/wmma_bench_v025.exe p3 $E/bin/wmma_bench_abl_staging_only.exe p3 4 16 1
python $E/run_balanced.py $E/ablation/abl_no_global \
  $E/bin/wmma_bench_v025.exe p3 $E/bin/wmma_bench_abl_no_global.exe p3 4 16 1

# candidate campaigns
python $E/run_balanced.py $E/abba_p3_vs_fourwave \
  $E/bin/wmma_bench_v025.exe p3 $E/bin/wmma_bench_fourwave.exe fourwave 24 16 1
python $E/run_balanced.py $E/abba_p3_vs_lean \
  $E/bin/wmma_bench_v025.exe p3 $E/bin/wmma_bench_lean.exe lean 24 16 1

# statistics and bit-identical production comparison
python $E/analyze.py $E abba_p3_vs_fourwave abba_p3_vs_lean
python $E/compare_dumps.py $E/baseline_accuracy/dumps $E/candidate_lean_accuracy/dumps
```

`run_balanced.py` reuses `phase_lxxxi_evidence/parse_gpu_profile.py` for the
GPU-timestamp stream and `validation/abba_stats.py` for the interval, so there is
no second implementation of either rule. `make_ablations.py` regenerates the
ablation modules from the committed P3 source with exact-match assertions.
