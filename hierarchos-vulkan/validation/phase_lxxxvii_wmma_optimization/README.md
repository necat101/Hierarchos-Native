# Phase LXXXVII — Geometry-aware WMMA optimization and P3 staging investigation

Status: **OPTIMIZATION VERIFIED — ACCURACY PRESERVED.** One candidate is fully
qualified and it is a staging change, not an epilogue change: batching the eight
global operand loads of each K slice into one unlatched group makes the
production P3 kernel **1.34×–2.04× faster on every protected shape**, with output
that is **bit-identical to v0.2.5 on 120/120 production cases** and an accuracy
matrix that is byte-for-byte the same as the v0.2.5 P3 baseline.

The Phase LXXXVI `lean` candidate was recovered byte-for-byte and re-measured. It
is **dominated** by the staging candidate on all four protected shapes, and the
tall-K regression is now explained by a measured, instruction-identical control
rather than by a reduction in LDS bytes.

No tag, release, version bump, merge or push was produced. `main`'s P3 source,
SPIR-V, selector, policy and the v0.2.5 release gate are unchanged, and the gate
still passes on this tree.

---

## A. Starting state, re-derived on this host

| Item | Value |
| --- | --- |
| Repository | `necat101/Hierarchos-Native` |
| Branch | `codex/wmma-v026-optimization` |
| `HEAD` at start | `d221daa` — "Record the Phase LXXXVI WMMA optimization campaign…" |
| Release tag | `v0.2.5` → tag object `c4125f36…` → commit `d0f48c9637ac8d0fdfc1418c2b90de81196f9d4d` |
| Working tree at start | clean |
| `git diff d0f48c9 d221daa -- hierarchos-vulkan/src hierarchos-vulkan/shaders` | **empty** (LXXXVI committed evidence only) |
| P3 source SHA-256 | `4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12` |
| P3 SPIR-V SHA-256 | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2`, 23,280 B |
| P3 runtime identity | `wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280` |
| Phase LXXXVI evidence | `hierarchos-vulkan/validation/phase_lxxxvi_wmma_optimization/` — unmodified |

Every hash above was recomputed here. The archived v0.2.5 harness is still the
binary that produced the release receipts
(`bin/wmma_bench_v025.exe` `8fcdc4a8…`, `bin/wmma_qualify_v025.exe` `9761361b…`),
and it reports the qualified pipeline verbatim:

```text
full-fp16 variant: p3 (shader wmma_gemm_f16_f16acc_twowave_p3, pipeline
wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280)
```

`validation/check_p3_release.py` **PASSES** on the tree as delivered by this
phase (see §K).

### Shader build recipe

`glslc -fshader-stage=compute --target-env=vulkan1.2 <src> -o <out>` with
`shaderc v2023.8 v2025.3 / spirv-tools v2025.3 / glslang 11.1.0-1253`
reproduces the committed P3 module byte-identically (`2aff4123…`, 23,280 B). Every
module in this phase was built with that exact recipe and `spirv-val`-clean.

---

## B. Environment

| Item | Value |
| --- | --- |
| Adapter | AMD Radeon Graphics, `1002:15bf`, RDNA3 / Phoenix |
| Driver (host) | `32.0.32015.14001` (Win32_VideoController) — **not** the v0.2.5 release driver |
| Vulkan | device apiVersion 1.4.x, cooperative tile `16x16x16`, subgroup forced 32 |
| Toolchain | `rustc 1.93.0`, `cargo 1.93.0`, `glslc v2023.8 v2025.3` |
| Measurement harness | `target-lxxxvii` build of `wmma_bench` / `wmma_qualify`; hashes in `identities.json` |

Every number in this record is a *paired* comparison of two modules measured
minutes apart in the same driver session on the same host. Absolute milliseconds
are therefore not comparable to the v0.2.5 release notes, exactly as in Phase
LXXXVI. Ratios are.

**No RGP capture, ISA read-out, VGPR count or hardware occupancy counter was
obtained.** The Phase LXV RDF/COMGR extraction tooling is still not in this
checkout (`lxv-rgp/` absent), so it was not rebuilt. Everything attributed below
is attributed from paired GPU-timestamp measurements of modules that differ only
in ways that are stated, and every residency statement is labelled as an
inference from those measurements.

---

## C. Recovery of the Phase LXXXVI artifacts

The phase brief warned that the original lean candidate source might be gone. It
is not.

| Artifact | Result |
| --- | --- |
| `phase_lxxxvi…/candidates/…_p3_lean.comp` | present; SHA-256 `a75db2d3de7de7ae6d1c49412dde1f67d48580960b0e4e7750b88e2e3f8b335b` — **matches the Phase LXXXVI record exactly** |
| Recompiled from that source | `fdafb2d6b45b5ff1679db60019decda3aae8b7bd447239dcaafabe56ea4e7b99`, 23,172 B, FNV `0xcf2949a95f7eee92` — **byte-identical to the archived module** |
| `bin/wmma_bench_lean.exe` / `wmma_qualify_lean.exe` | present and executable |
| `ablations/abl_no_epilogue.{comp,spv}` | present; regenerated here to `93168bd6…`, 20,692 B — **byte-identical** |
| `ablations/abl_no_global.spv`, `abl_staging_only.spv` | present; regenerated to `6f9cd6c7…` / `6899b603…` — **byte-identical** |
| Profiling dumps | **missing.** `dumps_manifest.json` pins 24 files (8 seeds × 3 arms, ~16 MB each); **none of the 24 exist on disk**, so those hashes cannot be verified. No substitute was generated and none is claimed: the 120/120 bit-identity claim below rests on dumps produced *in this phase*, not on LXXXVI's. |

So the lean candidate was measured from its **original bytes**, not from a
reconstruction. This phase's candidate modules live in `candidates/` with their
own hashes; the v0.2.5 baseline binaries, shader and SPIR-V are untouched and
were never overwritten.

### Harness change (measurement tooling only)

`src/bin/wmma_bench.rs` gained two optional flags used by the sweeps:
`--shapes MxNxK,…` (an explicit geometry list) and `--skip-reference` (skip the
host FP32 reference GEMM, used only for wide sweeps that make no numerical
claim). With neither flag the benchmark behaves exactly as before. The
production kernel, selector, modules and policy are not touched by either flag.

---

## D. Investigation A — the tall-K regression, measured

### D.1 What lean actually changes

`lean` deletes `c_tile[2][256]` and `bank_tile[4][256]` (3,072 B) and stages the
pair epilogue through `a_tile[wave]` / `b_tile[wave]`, which are dead in exactly
that window. Instructions, barriers, bank order and MMA operands are otherwise
identical, so its output must be — and is — bit-identical.

| Module | LDS per workgroup | Composition |
| --- | ---: | --- |
| P3 | 9,216 B | a 1,024 + b 1,024 + c 1,024 + bank_tile 2,048 + bank_sum 4,096 |
| `lean` | 6,144 B | a 1,024 + b 1,024 + bank_sum 4,096 |
| `leanpad` | 9,216 B | `lean` with `bank_sum[4][448]` instead of `[4][256]` |
| `pad` | 14,336 B | P3 with `bank_sum[4][576]` instead of `[4][256]` |

`leanpad` and `pad` are the point of this phase's Investigation A. Widening
`bank_sum`'s *column count* changes the LDS footprint and nothing else: no extra
instruction executes, no access pattern changes beyond the row stride, and the
code that could plausibly explain a difference (the epilogue) is untouched.

### D.2 Measurements — 24 paired blocks, corrected 95 % t CI

`p3` vs `lean` (replication of Phase LXXXVI on this session):

| Shape | P3 ms | lean ms | ratio | 95 % CI |
| --- | ---: | ---: | ---: | --- |
| `2048×768×2048` | 5.0301 | 4.7446 | 1.0592 | [1.0550, 1.0634] |
| `512×2048×768` | 1.2459 | 1.1630 | 1.0723 | [1.0655, 1.0791] |
| `512×768×768` | 0.5328 | 0.4641 | 1.1502 | [1.1207, 1.1797] |
| `128×8192×768` | 1.3960 | 1.4831 | **0.9427** | [0.9318, 0.9536] |
| `64×64×64` | 0.0386 | 0.0389 | 0.9938 | [0.9924, 0.9953] |
| `16×128×32` | 0.0322 | 0.0324 | 0.9914 | [0.9892, 0.9937] |

Directionally identical to Phase LXXXVI (`+5.9 %`, `+7.2 %`, `+15.0 %`,
`−5.7 %` here against `+6.0 %`, `+7.3 %`, `+14.9 %`, `−3.7 %` there) on a
different driver build. The rejection is reproducible.

### D.3 The LDS-matched control — the mechanism

8 paired blocks each, same session:

| Shape | `leanpad` (9,216 B) ratio | `pad` (14,336 B) ratio |
| --- | ---: | ---: |
| `2048×768×2048` | 1.0025 [0.9989, 1.0061] | **0.7270** [0.7179, 0.7360] |
| `512×2048×768` | 0.9947 [0.9848, 1.0046] | **0.7642** [0.7555, 0.7730] |
| `512×768×768` | 1.0235 [0.9828, 1.0643] | **0.6998** [0.6571, 0.7426] |
| `128×8192×768` | 1.0141 [0.9925, 1.0357] | **0.8032** [0.7850, 0.8214] |
| `64×64×64` | 1.0084 [1.0042, 1.0127] | 1.0066 [1.0036, 1.0096] |
| `16×128×32` | 1.0134 [1.0098, 1.0170] | 1.0059 [1.0021, 1.0098] |

Two facts follow directly, and neither depends on an occupancy counter:

1. **Restoring lean's LDS footprint to 9,216 B restores P3's behaviour on every
   shape, including the tall-K one** (`1.0141`, CI spanning 1.0). Since `leanpad`
   keeps lean's epilogue-reuse code, the difference between `lean` and P3 cannot
   be an epilogue-code effect. Any explanation of the tall-K regression that rests
   on "the epilogue now runs through `a_tile`/`b_tile`" is refuted.
2. **LDS footprint alone moves runtime by 25–43 %** with no instruction change
   (`pad`, 9,216 → 14,336 B, is 0.70×–0.80× on the four protected shapes). The
   kernel's performance is therefore sharply sensitive to resident workgroup
   count, which is the resource the footprint controls.

Arithmetic, explicitly labelled as arithmetic rather than a counter read: if the
CU exposes 64 KiB of LDS in 512 B allocations, 9,216 B admits 7 workgroups,
6,144 B admits 10 and 14,336 B admits 4. The measured `pad` penalty (7 → 4, −43 %
residency, −25 % throughput) and `lean` gain (7 → 10, +43 % residency) are
consistent with that, but the residency numbers are inferred, not read.

### D.4 Where the sign flips

`lean` vs P3, 8 paired blocks, explicit geometry sweep (labels verified against
the requested list — see §J):

| Shape | workgroups | B operand | ratio | 95 % CI |
| --- | ---: | ---: | ---: | --- |
| `128×2048×768` | 256 | 3.1 MB | 1.1732 | [1.0455, 1.3010] |
| `128×4096×768` | 512 | 6.3 MB | 1.0068 | [0.9696, 1.0441] |
| `128×8192×768` | 1,024 | 12.6 MB | 0.9467 | [0.9173, 0.9761] |
| `128×16384×768` | 2,048 | 25.2 MB | 0.9650 | [0.9539, 0.9760] |
| `64×8192×768` | 512 | 12.6 MB | 1.0148 | [0.9586, 1.0710] |
| `256×8192×768` | 2,048 | 12.6 MB | 0.9403 | [0.9252, 0.9554] |
| `512×8192×768` | 4,096 | 12.6 MB | 0.9218 | [0.9165, 0.9271] |

At fixed `M=128, K=768` the sign of `lean`'s effect is **monotone in N**: +17 %
at 3.1 MB of streamed B, neutral at 6.3 MB, negative from 12.6 MB upward. It is
not workgroup count on its own — `512×2048×768` (1,024 workgroups, 3.1 MB B) is
`+7.2 %` while `128×8192×768` (1,024 workgroups, 12.6 MB B) is `−5.7 %`. Nor is
it K depth: both are `K=768`.

**Measured cause, as supported by the evidence:** the regression is the
interaction between extra resident workgroups and a streamed B operand that
greatly exceeds L2. Raising residency helps when the operand working set is
cache-resident and hurts when it is not, because the additional concurrency
converts into additional concurrent DRAM streams rather than into cache reuse.
The instruction-identical LDS controls establish the residency half of that
statement; the sweep establishes the footprint dependence. The DRAM-stream part
remains an inference — no bandwidth counter was read, and the LXXXVI bandwidth
argument (549–645 GMAC/s across shapes) is about averages, not about stream
concurrency.

### D.5 Consequence for the geometry-aware lean selector (Investigation B)

The brief asked whether lean could be retained where it helps. It can, but it
should not, because it is **dominated by this phase's staging candidate on every
protected shape**:

| Shape | `lean` | `batch` |
| --- | ---: | ---: |
| `2048×768×2048` | 1.0592 | **1.7078** |
| `512×2048×768` | 1.0723 | **1.3365** |
| `512×768×768` | 1.1502 | **2.0446** |
| `128×8192×768` | 0.9427 | **1.3819** |

A selector that dispatched `lean` on the three shapes where it wins would be
strictly slower than simply dispatching the staging candidate everywhere, and it
would carry a second module, a predicate and a boundary to defend. **No
geometry-aware lean selector is proposed.** The selector question is answered by
dominance, not by budget.

---

## E. The missing epilogue ablation, completed

Phase LXXXVI could not build `abl_no_epilogue` (two interrupted builds). It is
built here from the preserved source, at the recorded bytes (`93168bd6…`,
20,692 B), and measured against P3 — 8 paired blocks, same session:

| Shape | P3 ms | no-epilogue ms | ratio | 95 % CI |
| --- | ---: | ---: | ---: | --- |
| `2048×768×2048` | 5.0355 | 4.8771 | **1.0326** | [1.0262, 1.0390] |
| `512×2048×768` | 1.2408 | 1.1074 | **1.1228** | [1.1127, 1.1328] |
| `512×768×768` | 0.5567 | 0.3855 | 1.4600 | [1.3819, 1.5381] |
| `128×8192×768` | 1.4190 | 1.4723 | **0.9601** | [0.9432, 0.9769] |
| `64×64×64` | 0.0386 | 0.0257 | 1.4989 | [1.4925, 1.5053] |
| `16×128×32` | 0.0321 | 0.0193 | 1.6699 | [1.6600, 1.6798] |

The instrument is numerically wrong by construction and produces no result; it
exists only to attribute time.

**Revised decomposition.** Phase LXXXVI measured `abl_staging_only` (no MMA, no
epilogue) at `1.259×` on `2048×768×2048` and read that as "MMA + epilogue ≈ 20 %".
With the epilogue isolated at `1.0326×`, that 20 % splits into **≈ 3 % epilogue
and ≈ 17–18 % cooperative MMA**. The epilogue is not a meaningful cost on any
protected shape, and on `128×8192×768` removing it is *slower* (`0.9601`), i.e.
the measurement has no usable positive signal there at all.

This is the durable correction to LXXXVI's reading: the epilogue was never the
lever, and the LDS-trimmed candidate's gains could not have come from epilogue
instructions — which §D.3 confirms independently.

---

## F. Investigation C — batched operand staging (candidate C1)

### F.1 What the measurement said the problem was

The staging loop per K slice is

```glsl
for (uint p = tid; p < 256u; p += threads) {   // threads == 64, so four iterations
    ... address arithmetic, one 32-bit global load, two shared stores ...
}
barrier();
coopMatLoad ×3; coopMatMultiplyAdd ×2;
barrier();
```

The trip count is four and each iteration is a **dependent load → store pair**, so
per slice and per thread the eight global loads (four for A, four for B) issue one
at a time behind a shared-memory store that waits on each. `abl_no_global` making
the kernel `2.68×` faster without removing a single store, barrier or multiply-add
is exactly the signature of that chain, not of bandwidth.

### F.2 The change

When the staged 32×16 slab is fully in bounds and the operand is pair-aligned, the
four iterations are emitted unrolled with **all four 32-bit loads issued before
any shared store**, and with the per-element bounds tests removed because the
uniform predicate makes them provably unnecessary:

```glsl
if (wide_a && m0 + 32u <= pc.m && k0 + 16u <= pc.k) {
    ... four f16vec2 loads ...
    ... eight shared stores ...
} else {
    ... the original masked loop, byte-for-byte ...
}
```

Four such fast paths exist, one per (operand × layout) branch, and four masked
fallbacks remain. `layout(local_size_x = 64)` makes `threads == 64` a
compile-time fact, so the mapping `p = tid, tid+64, tid+128, tid+192` →
`(tile, e) = (0,tid), (0,tid+64), (1,tid), (1,tid+64)` is exact; the fast path
writes the same values to the same shared slots from the same threads.

**Why the output bits must be unchanged.** No arithmetic statement is touched: not
the slice stream, not the bank assignment, not a `coopMatMultiplyAdd` operand, not
the float32 left-to-right bank sum, not the single binary16 rounding, not the
store masking. The fast path changes *when* a global load is issued and removes
bounds tests that the predicate proves are never taken. Barrier placement and
count are unchanged. Bit-identity is therefore a property of the construction, and
it is then verified against the GPU in §I.

One implementation note worth recording because it cost a build: in the
**transposed-B** branch the rank pair runs along K, so the K offset of the pair
comes from the shared-memory *row* index and the N offset from the *column*
index. The first generated version used the column index for both, produced a
uniformly wrong result that the drift column caught immediately
(`3.086e-1` vs `3.236e-4`), and was discarded. The generator in
`make_candidates.py` now documents that mapping at the point of use.

### F.3 Cost

The module is larger: **36,912 B of SPIR-V against P3's 23,280 B**, because both
the batched path and the masked fallback exist for all four branches. That is the
price of leaving the unaligned path untouched, and it is recorded rather than
minimised — it is also the most plausible explanation for the small effect on
odd-stride shapes in §H.

### F.4 Performance — 24 paired blocks per protected shape, corrected 95 % t CI

Same executable, same host, same session; the only difference between arms is
`HIERARCHOS_WMMA_FULL_FP16_VARIANT`.

| Shape | P3 ms | `batch` ms | speedup | 95 % CI | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| `2048×768×2048` | 5.0298 | 2.9417 | **1.7078** | [1.6995, 1.7162] | faster |
| `512×2048×768` | 1.2484 | 0.9332 | **1.3365** | [1.3298, 1.3432] | faster |
| `512×768×768` | 0.5389 | 0.2656 | **2.0446** | [1.9902, 2.0990] | faster |
| `128×8192×768` | 1.3954 | 1.0105 | **1.3819** | [1.3701, 1.3937] | faster |
| `64×64×64` (control) | 0.0386 | 0.0311 | 1.2444 | [1.2414, 1.2474] | faster |
| `16×128×32` (control, below the 16 MMAC floor) | 0.0322 | 0.0303 | 1.0615 | [1.0587, 1.0642] | faster |

Against the standing acceptance rule:

* at least one protected workload ≥ 3 % faster with corrected CI lower bound
  > 1.0 — **met by all four**, the smallest at `+33.6 %`;
* every other protected workload CI lower bound ≥ 0.98 — **met**, the smallest at
  `1.3298`;
* no statistically significant regression on any protected shape — **met**;
* each shape reported independently — done above.

The controls are reported but are not part of the acceptance rule; both are
slightly faster, and the floor itself is unchanged (§K).

---

## G. Boundary and neighbour geometries

Both tables below are 8-paired-block screens against the same P3 baseline, with
`--skip-reference` (so they make a *performance* claim only — no numerical claim
is made for these runs). They exist to test §F's fast-path predicate at and
around its boundary, and to check that the win is not specific to the four
benchmark shapes.

### G.1 Fast path taken (aligned, in-bounds, even strides)

| Shape | P3 ms | `batch` ms | ratio | 95 % CI |
| --- | ---: | ---: | ---: | --- |
| `128×4096×768` | 0.8541 | 0.5544 | 1.5327 | [1.4958, 1.5696] |
| `128×16384×768` | 2.8318 | 1.9956 | 1.4095 | [1.3945, 1.4245] |
| `256×8192×768` | 2.7318 | 1.9811 | 1.3796 | [1.3678, 1.3913] |
| `512×1024×768` | 0.6533 | 0.3137 | 2.0850 | [1.9555, 2.2145] |
| `1024×768×768` | 0.7300 | 0.3449 | 2.1329 | [2.0585, 2.2073] |
| `512×768×1536` | 1.0596 | 0.4978 | 2.1101 | [2.0003, 2.2199] |
| `2048×768×1024` | 1.8338 | 0.8641 | 2.1254 | [2.0646, 2.1862] |
| `480×768×768` (M just below 512) | 0.5706 | 0.2661 | 2.1027 | [1.9825, 2.2230] |
| `512×736×768` (N just below 768) | 0.5758 | 0.2727 | 2.0978 | [1.9547, 2.2409] |
| `512×768×752` (K just below 768) | 0.6628 | 0.3088 | 2.2602 | [1.9787, 2.5416] |

The boundary is not a cliff at the protected sizes: `M=480`, `N=736` and `K=752`
are one tile-step below the protected `512×768×768` and all keep the full gain.

### G.2 Fast path *not* uniformly taken

| Shape | P3 ms | `batch` ms | ratio | 95 % CI |
| --- | ---: | ---: | ---: | --- |
| `511×768×768` (M not a multiple of 32) | 0.5635 | 0.3036 | 1.8356 | [1.7034, 1.9678] |
| `255×768×768` | 0.5363 | 0.2911 | 1.8788 | [1.6323, 2.1252] |
| `512×767×768` (N not a multiple of 32) | 0.5817 | 0.3144 | 1.8862 | [1.6471, 2.1253] |
| `512×2047×768` (N = 2047) | 1.2603 | 0.9880 | 1.2636 | [1.2349, 1.2922] |
| `512×768×767` (K odd ⇒ odd `lda`) | 0.8270 | 0.8151 | 1.0163 | [0.9093, 1.1233] |
| `2048×768×2047` (K odd ⇒ odd `lda`) | 7.9309 | 8.0235 | **0.9898** | [0.9860, 0.9937] |

The last two rows are the honest limitation. With an odd reduction the
non-transposed A stride `lda = K` is odd, `wide_a` is false, and the A staging
cannot be batched at all — only B benefits. There the gain collapses to `+1.6 %`
(inconclusive) and `−1.0 %` (CI excluding 1.0) respectively.

That is a real, reproducible, statistically significant **1 % regression on
odd-stride shapes**, outside the protected set. It is not hidden here and it
shapes the promotion recommendation: the module's validated condition is *both
operand strides even*. A production promotion should either accept that 1 % on
odd-stride shapes or gate on stride parity and keep P3 otherwise — the predicate
is derived from the mechanism (`wide_a`/`wide_b`), not from the benchmark list.

---

## H. Accuracy and the numerical contract

Everything below was produced by the **unmodified** Phase LXXXIV driver
(`phase_lxxxiv_evidence/run_lxxxiv_accuracy.py`) over the original 8 seeds × 5
mode/oracle configurations, on the same harness as the baseline.

| Check | Baseline (P3) | Candidate (`batch`) |
| --- | --- | --- |
| Receipts | 40 | 40 |
| Receipts with any failing gate | 15 | 15 |
| Configurations with failing gates | `full-fp16/rdna3-dot2`, `full-fp16/split2` | identical |
| `full-fp16 / tile16` (production) | **37/37, 0 failed on all eight seeds** | **37/37, 0 failed on all eight seeds** |
| `mixed / tile16`, `full-fp16 / split8` | 37/37, 0 failed | identical |
| Per-seed gate/failure log | — | **byte-identical** (`diff` of the two 50-line sweep logs is empty) |
| `accuracy_matrix.json` | — | **byte-identical** (whole document, not a summary) |

Numeric contract, all unchanged and all re-observed:

* `ULP_GATE_LIMITS = 0.03125`, original eight seeds, references, inputs,
  tolerances and test definitions untouched;
* diagnostic `rdna3-dot2` and `split2` failures remain recorded as diagnostics and
  are **not** reclassified, suppressed or altered;
* no new seed-dependent instability: the failure counts per seed are identical,
  not merely similar;
* the maximum REF64F16 error cannot have moved, because every production output
  tensor is bit-identical (§I) and the reference is the same reference; the v0.2.5
  bound `0.02593803` therefore still holds. (The dump records `actual` values, not
  per-case error scalars, so this is argued from bit-identity rather than quoted
  from a field.)

### Bit-identity, 120/120

`compare_dumps.py` (Phase LXXXVI, unmodified) over the eight production dumps of
each arm:

```text
cases=120 bit_identical=120 differing=0 missing_files=0
BIT-IDENTICAL COMPARISON: PASS (every production case, every element)
```

Both arms are the same executable with no oracle override; inputs are verified
equal on both sides by the driver, so this compares the kernel, not the fixture.

---

## I. Shader and module identities

| Module | Source SHA-256 | SPIR-V SHA-256 | Bytes | FNV-1a64 |
| --- | --- | --- | ---: | --- |
| P3 (production, untouched) | `4e448e3c…786d12` | `2aff4123…c517d2` | 23,280 | `0x076216ad97d7605f` |
| `lean` (Phase LXXXVI, recovered) | `a75db2d3…8b335b` | `fdafb2d6…4e7b99` | 23,172 | `0xcf2949a95f7eee92` |
| `batch` (candidate C1) | `9857a8ad…953da4` | `b0722420…a8b27d` | 36,912 | `0xfecb2a74c4bbbce4` |
| `leanpad` (control) | `f3007dc1…6264696` | `0ce19e04…c369a9c` | 23,188 | `0xbe3b324461241bfa` |
| `pad` (probe) | `14059af1…1510f841` | `023220e7…b9999949` | 23,296 | `0xf8e7f2896a24af7e` |
| `ablnoepi` (instrument) | `a1a528ef…f3e11f` | `93168bd6…82c7f9` | 20,692 | `0xf4fd5b2aadfec2a9` |

Full hashes, harness binaries and toolchain versions: `identities.json`.
Measurement harnesses: `bin/wmma_bench_lxxxvii_candidates.exe`
`08d93219…f8f8fce`, `bin/wmma_qualify_lxxxvii_candidates.exe`
`8eca5952…908d0a0e`.

---

## J. Evidence hygiene

* Phase LXXXVI evidence is unmodified (no file in that directory was written).
* The v0.2.5 baseline binaries, P3 shader and P3 SPIR-V were not overwritten; the
  production source and module hashes in §A are the release hashes.
* Candidate modules are `candidates/*.comp` + `*.spv`, generated by
  `make_candidates.py` from the two pinned sources by exact textual edit
  (`require_once` raises on any drift).
* The scratch registration that makes the five modules selectable through
  `HIERARCHOS_WMMA_FULL_FP16_VARIANT` is **not** in the delivered tree. It is
  preserved as `candidates/register_candidates.patch`; apply it with `git apply`,
  rebuild, and the harness binaries above are reproducible. Without it the tree is
  v0.2.5 plus the two `wmma_bench` flags.
* Production dumps are 259,500,114 B (16 files). They are **not** committed;
  `dumps_manifest.json` pins each file's byte count and SHA-256, and
  `validation/phase_lxxxvii_wmma_optimization/.gitignore` keeps them out of Git.
  They remain on disk at `baseline_accuracy/dumps/` and
  `candidate_batch_accuracy/dumps/`.
* **One correction to record against the tooling:** the preserved Phase LXXXI
  parser (`phase_lxxxi_evidence/parse_gpu_profile.py`) segments a profile stream
  *positionally* against a fixed six-shape tuple. A `--shapes` run must hand it
  the tuple it actually requested. The first iteration of this phase's sweeps did
  not, so three sweeps were discarded — one produced silently mislabelled blocks,
  two aborted on the block-count check. `run_arm_campaign.py` now passes the
  requested tuple; the segmentation check itself is unchanged and still
  fail-closed. Every sweep quoted above was re-run after that fix, from scratch
  (`rm -rf` of the affected directories).

---

## K. What was not done, and why

* **No geometry-aware lean selector.** Dominated on every protected shape by C1
  (§D.5). Building it would add a module and a predicate for a strictly slower
  result.
* **No production promotion.** The candidate is qualified but this phase's brief
  keeps v0.2.5 as the production route: the Full FP16 default is still `P3`, the
  four-variant inventory is unchanged, and `validation/check_p3_release.py`
  **passes** on the delivered tree (P3 source and SPIR-V are the qualified bytes,
  4 committed variants, WMMA opt-in with `off` external default, the
  16,000,000-MAC predicate at every admission point, plain/FP32 fallbacks,
  Mixed route, split8/two-bank/two-wave intact). Promotion is a one-module,
  one-predicate change for the next phase, not a redesign.
* **No prefetch-across-slices second staging candidate.** With the per-slice
  dependent chain removed, a cross-slice lookahead would be a third structural
  change on top of a candidate that already clears the bar by 11× the required
  margin. The phase limit is two serious candidates and this is the second.
* **No RGP, ISA, VGPR or occupancy counter.** Tooling unavailable (§B); every
  residency claim is labelled as inference from instruction-identical controls.
* **No change to any other arm.** FP32, Mixed, split8, two-bank, two-wave,
  reduction/K limits, per-batch admission and the subgroup policy are untouched.
* **No end-to-end training timing.** Every number is a per-dispatch GEMM kernel
  time.
* **No v0.2.6 tag, release asset, version bump, merge or push.**

---

## L. Conclusion

The Phase LXXXVI campaign concluded that the staging path was the dominant cost
but then attacked residency instead, and lost on one shape. Completing its missing
ablation and adding instruction-identical LDS controls shows why that could not
have worked: **the epilogue is ~3 % of the kernel**, and the LDS-trimmed
candidate's gains came entirely from extra resident workgroups — which help when
the streamed operand fits in cache and hurt when it does not.

Attacking the staging measurement directly, rather than the resource it merely
exposed, produces a change that is faster on **every** protected shape
(`+33.6 %` to `+104.5 %`, all corrected CI lower bounds ≥ 1.3298), bit-identical
on **120/120** production cases, and byte-identical in its accuracy matrix. Its
documented weakness is a **1 % regression on odd-stride shapes**, where only one
operand's staging can be batched.

A v0.2.6 release candidate is justified. The recommended promotion is
`P3` → the batched module as the default Full FP16 variant, with `P3` retained as
the selectable fallback and, optionally, a stride-parity predicate that keeps P3
for odd `lda`/`ldb`. Everything needed to make that decision — source, byte
identities, accuracy, 24-pair receipts, boundary sweeps and the predicate's
mechanism — is in this directory.

---

## M. Reproducing this record

```bash
cd hierarchos-vulkan
export D=validation/phase_lxxxvii_wmma_optimization
export CARGO_TARGET_DIR=target-lxxxvii

# 1. candidate modules from the two pinned sources (requires glslc + spirv-val)
python $D/make_candidates.py $D/candidates

# 2. harness with the candidate modules selectable
git apply $D/candidates/register_candidates.patch
cp $D/candidates/*.spv shaders/                       # build registry needs them
cargo build --release --bin wmma_bench --bin wmma_qualify

# 3. the primary receipt (24 paired blocks, built-in six shapes)
python $D/run_arm_campaign.py --out $D/abba_p3_vs_batch \
    --exe $CARGO_TARGET_DIR/release/wmma_bench.exe --base p3 --cand p3-batch \
    --pairs 24 --iterations 16

# 4. the rest of the campaigns
python $D/run_arm_campaign.py --out $D/abba_p3_vs_lean    --exe $CARGO_TARGET_DIR/release/wmma_bench.exe --base p3 --cand p3-lean    --pairs 24 --iterations 16
python $D/run_arm_campaign.py --out $D/abba_p3_vs_leanpad --exe $CARGO_TARGET_DIR/release/wmma_bench.exe --base p3 --cand p3-leanpad --pairs 8  --iterations 16
python $D/run_arm_campaign.py --out $D/abba_p3_vs_pad     --exe $CARGO_TARGET_DIR/release/wmma_bench.exe --base p3 --cand p3-pad     --pairs 8  --iterations 16
python $D/run_arm_campaign.py --out $D/abba_p3_vs_ablnoepi --exe $CARGO_TARGET_DIR/release/wmma_bench.exe --base p3 --cand p3-ablnoepi --pairs 8 --iterations 16

# 5. boundary sweeps (performance only; no numerical claim)
python $D/run_arm_campaign.py --out $D/sweep_batch_aligned --exe $CARGO_TARGET_DIR/release/wmma_bench.exe \
    --base p3 --cand p3-batch --pairs 8 --iterations 16 --skip-reference \
    --shapes 128x4096x768,128x16384x768,256x8192x768,512x1024x768,1024x768x768,512x768x1536,2048x768x1024,480x768x768,512x736x768,512x768x752
python $D/run_arm_campaign.py --out $D/sweep_batch_boundary --exe $CARGO_TARGET_DIR/release/wmma_bench.exe \
    --base p3 --cand p3-batch --pairs 8 --iterations 16 --skip-reference \
    --shapes 512x768x767,512x767x768,511x768x768,255x768x768,512x2047x768,2048x768x2047
python $D/run_arm_campaign.py --out $D/sweep_lean_residency --exe $CARGO_TARGET_DIR/release/wmma_bench.exe \
    --base p3 --cand p3-lean --pairs 8 --iterations 16 --skip-reference \
    --shapes 64x8192x768,128x2048x768,128x4096x768,128x8192x768,128x16384x768,256x8192x768,512x8192x768

# 6. accuracy, both arms, and the 120-case comparison
HIERARCHOS_WMMA_FULL_FP16_VARIANT=p3 \
  python ../phase_lxxxiv_evidence/run_lxxxiv_accuracy.py --exe $CARGO_TARGET_DIR/release/wmma_qualify.exe --out $D/baseline_accuracy
HIERARCHOS_WMMA_FULL_FP16_VARIANT=p3-batch \
  python ../phase_lxxxiv_evidence/run_lxxxiv_accuracy.py --exe $CARGO_TARGET_DIR/release/wmma_qualify.exe --out $D/candidate_batch_accuracy
python validation/phase_lxxxvi_wmma_optimization/compare_dumps.py \
  $D/baseline_accuracy/dumps $D/candidate_batch_accuracy/dumps

# 7. identities, dump pins, and the release gate on the restored tree
python $D/collect_identities.py $D/identities.json
python $D/hash_dumps.py $D/dumps_manifest.json $D/baseline_accuracy $D/candidate_batch_accuracy
rm -f shaders/wmma_gemm_f16_f16acc_twowave_p3_{lean,leanpad,pad,batch,ablnoepi}.spv
git checkout -- src/wmma.rs
python validation/check_p3_release.py
```

`run_arm_campaign.py` imports `schedule()` from the Phase LXXXVI driver, the
GPU-timestamp parser from Phase LXXXI and the interval from
`validation/abba_stats.py`, so the ABBA/BAAB order, the steady-state rule and the
corrected Student-t interval each still have exactly one implementation.
