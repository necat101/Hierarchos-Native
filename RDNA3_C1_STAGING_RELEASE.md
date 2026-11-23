# RDNA3 Full FP16 WMMA — v0.2.6 batched-staging (C1) release

Status: **production Full FP16 staging schedule, opt-in mode, qualified on the
measured RDNA3 device.** This is the release note for the Phase LXXXVIII
promotion of the Phase LXXXVII candidate C1. v0.2.5 remains the previous stable
release and its documentation, receipts, binaries and P3 module are unchanged.

This document does not rewrite any historical receipt. [RDNA3_P3_RELEASE.md](RDNA3_P3_RELEASE.md)
stays the v0.2.5 record for the P3 epilogue, the Phase LXXXVII directory stays
the discovery record for the staging investigation, and this document is the
record for what shipped in v0.2.6.

---

## 1. What changed

Inside `--wmma-mode full-fp16` the production route now chooses its **staging
schedule** per dispatch. Arithmetic, the two-wave workgroup, the K-slice stream,
the per-bank binary16 accumulation, the float32 sum of the eight bank partials
and the single binary16 rounding are the P3 contract and did not move.

| | v0.2.5 (P3) | v0.2.6 (C1, selected when proven) |
| --- | --- | --- |
| Per K slice, per thread | four dependent global-load → shared-store pairs | four 32-bit global loads issued, then eight shared stores |
| Bounds tests inside staging | per element, every iteration | removed in the fast path, because a uniform predicate proves them unnecessary |
| Masked staging path | the loop as written | **unchanged**, kept as the fallback |
| Arithmetic / reduction / rounding | P3 | **identical** |
| LDS per workgroup | 9,216 B | **9,216 B** |
| Barriers per K slice | 2 | **2** (none added, none moved) |
| SPIR-V | 23,280 B | 23,280 B (P3) and 36,912 B (C1) |

The mechanism is memory-level parallelism. The staging loop's trip count is
four, and the original form makes every iteration a load that a shared store
depends on, so per slice and per thread the eight global loads (four for A, four
for B) issue one at a time behind a store. C1 issues the four loads first and
then the stores. Nothing else in the kernel changes.

**The original P3 module is the fallback, not a legacy option.** It is still
embedded in the same executables, still dispatched by the same
`FullFp16Variant::P3`, and it serves every geometry the C1 contract cannot prove
(§3). There is no new user-facing mode, flag or variant: users select
`full-fp16` exactly as before and cannot tell which staging schedule ran except
by reading the staging counters in a receipt.

## 2. Numerical qualification

The v0.2.5 numerical contract is unchanged: same eight seeds, same fixtures, same
oracles, same references, same `ULP_GATE_LIMITS = 0.03125`, same gates, same
gate arithmetic. Nothing was re-tuned.

Both arms were run from the **unmodified** Phase LXXXIV driver
(`phase_lxxxiv_evidence/run_lxxxiv_accuracy.py`) over eight seeds × five
mode/oracle configurations:

| Check | v0.2.5 P3 binary | v0.2.6 candidate binary |
| --- | --- | --- |
| Receipts | 40 | 40 |
| Receipts with any failing gate | 15 | 15 |
| Configurations with failing gates | `full-fp16/rdna3-dot2`, `full-fp16/split2` | identical |
| `full-fp16 / tile16` (production) | 37/37, 0 failed, all eight seeds | 37/37, 0 failed, all eight seeds |
| `mixed / tile16`, `full-fp16 / split8` | 37/37, 0 failed | identical |
| Per-seed failure counts | `8,6,6,6,4,5,5,6` (dot2) / `3,0,1,1,1,1,2,3` (split2) | **identical** |
| Maximum tile16 `max_abs` | `0.03125` (the hard gate value, at `128x128x96`) | `0.03125`, same gate |

The historical `split2` and `rdna3-dot2` diagnostic failures remain recorded as
diagnostics. They are not suppressed, reclassified or "fixed" by this release.

### Bit identity, 120/120

The two arms are the two **release harness binaries** — the archived v0.2.5
`wmma_qualify.exe` (`9761361b…`) and the v0.2.6 build — each producing its own
eight production dumps with no oracle and no variant override. The Phase LXXXVI
comparator, unmodified, over those dumps:

```text
cases=120 bit_identical=120 differing=0 missing_files=0
BIT-IDENTICAL COMPARISON: PASS (every production case, every element)
```

Every production output element on every one of the 120 cases is bit-identical
to what v0.2.5 produced. Because each REF64F16 error is a function of that output
tensor and the same preserved reference, no individual REF64F16 error can be
worse than v0.2.5's and the recorded maximum `0.02593803` still holds. That is a
consequence of bit identity, not a re-measurement, and it is stated that way.

### Both routes were exercised

A bit-identity claim is only worth something if the new path actually ran. The
receipts carry per-dispatch staging counters, and every one of the eight
production receipts reports **5 batched + 10 masked** of its 15 cooperative
dispatches: the qualification fixture set itself exercises both the C1 fast path
(the aligned `128x128x96` and `256x64x16` geometries in three layouts each) and
the P3 fallback (the partial-tile and odd-reduction fixtures).

## 3. The selection contract

C1 is selected only when the host can prove, for **every** workgroup and
**every** K slice of the dispatch, that its packed 32-bit loads are aligned and
in bounds. `batched_staging_eligible` in `hierarchos-vulkan/src/wmma.rs` is that
proof, and it is a pure function of the dispatch geometry:

1. **Full tile coverage** — `rows % 32 == 0`, `columns % 32 == 0`,
   `reduction % 16 == 0`. The grid is `ceil(columns/32) × ceil(rows/32)`, so
   these divisibilities are exactly what make `m0 + 32 ≤ m`, `n0 + 32 ≤ n` and
   `k0 + 16 ≤ k` uniformly true. With them the shader's per-element bounds tests
   are provably never taken — which is the only justification for removing them.
2. **Packed-pair stride parity** — `lda % 2 == 0` and `ldb % 2 == 0`. A packed
   load is `a[off >> 1]` on an `f16vec2[]`, so the element index must be even.
   Row-major A forms `gm * lda + gk` for odd `gm`, so `lda` must be even;
   row-major B is the symmetric case on `ldb`. The transposed layouts put the
   pair on the contiguous axis, where parity is structural, and the predicate
   states both parities anyway so no future caller-supplied stride slips through.
   For every layout this crate builds, condition 1 already implies both.
3. **No base offset exists to be misaligned.** The kernel's push constant is
   `m, n, k, lda, ldb, ldc, flags, reserved`; `reserved` is unused and there is
   no offset operand. `record_dispatch` binds whole buffers at descriptor offset
   zero. The only address term is `row * stride + col`, so a packed load's byte
   address is `allocation_start + 2 * element`, and `allocation_start` is
   four-byte aligned by the `GpuBuffer` ABI that every existing `float`/`u32`
   kernel in this crate already depends on.
4. **Complete 32-bit access bounds.** The shader's index arithmetic is `uint`, so
   the largest element index must fit in 32 bits (one element of headroom is
   left below the ceiling for the masked path's `off + 1`), it must not overflow
   during the `checked_mul(2)` byte conversion, and the *bound buffers' actual
   capacities* must cover every element the fast path reads. A buffer one
   element short is a fallback, not a silent over-read.

Batch strides and per-batch offsets do not appear: this kernel is a single-matrix
GEMM whose address expressions have no batch term, and per-batch admission
happens above it by dispatching each batch separately.

**Workgroup uniformity.** The fast-path predicate is built from `m0`, `n0` (from
`gl_WorkGroupID`), `k0` (a uniform loop variable) and push constants — all
workgroup-uniform — so the branch is uniform and no barrier sits inside it. Both
the fast path and the masked fallback converge before the existing `barrier()`
calls, whose count and placement are identical to P3's. The release gate checks
that the two sources have the same barrier count and exactly four added fast
paths.

**Why the predicate is this narrow.** Phase LXXXVII measured a reproducible
**1 % regression on odd-stride shapes**, where only one operand's staging can be
batched. Those shapes are not routed into C1 to see whether they happen to pass;
they are outside the contract and take P3.

## 4. Performance

`v0.2.5 P3` vs `v0.2.6 production selector`, 24 paired blocks per shape,
position-balanced ABBA/BAAB ordering, GPU timestamp queries around the compute
dispatch, strictly serial execution, identical tensors and dispatch geometry.
Ratio = baseline ms / candidate ms; **> 1.0 means the candidate is faster.**

| Shape | P3 ms | C1 ms | Speedup | Corrected 95 % CI |
| --- | ---: | ---: | ---: | --- |
| 2048×768×2048 | 5.0217 | 2.9347 | **1.7074×** | [1.6976, 1.7171] |
| 512×2048×768 | 1.2376 | 0.9316 | **1.3286×** | [1.3182, 1.3391] |
| 512×768×768 | 0.5352 | 0.2570 | **2.0920×** | [2.0415, 2.1425] |
| 128×8192×768 | 1.3955 | 1.0067 | **1.3782×** | [1.3580, 1.3983] |
| 64×64×64 (control) | 0.0387 | 0.0311 | 1.2450× | [1.2423, 1.2477] |
| 16×128×32 (control, below the floor) | 0.0322 | 0.0321 | 1.0021× | [0.9998, 1.0044] |

Acceptance: at least 3 % faster on every protected shape, corrected CI lower
bound above 1.0 on every protected shape, no significant regression — all four
protected shapes satisfy every clause, the smallest at `+32.9 %`.

These are **per-dispatch GPU GEMM timings, not end-to-end training-step
speedups**. No training-step or application speedup is claimed.

The two controls are reported, not claimed: `64×64×64` is far below the 16M-MAC
floor and production never dispatches there, and `16×128×32` is below the floor
*and* fails the tile-fill requirement, so the production route runs FP32.

### The staging module each shape actually dispatched

Taken from the GPU profile of the measured runs, not from a label:

| Shapes | Baseline (v0.2.5 binary) | Candidate (v0.2.6 binary) |
| --- | --- | --- |
| four protected + `64×64×64` | `wmma_gemm_f16_f16acc_twowave_p3` | `wmma_gemm_f16_f16acc_twowave_p3_c1` |
| `16×128×32` | `wmma_gemm_f16_f16acc_twowave_p3` | `wmma_gemm_f16_f16acc_twowave_p3` |

### Boundary and selector sweep

Seventeen geometries measured through the production binary with no override:
the ten aligned neighbours Phase LXXXVII measured and seven unaligned
neighbours. **17/17 dispatched the module the contract requires** — all ten
aligned geometries selected C1, and `511×768×768`, `255×768×768`,
`512×767×768`, `512×2047×768`, `512×768×767`, `2048×768×2047` and `16×128×32`
selected the original P3 module. Every shape's cooperative arm stayed within
`8.9e-4` of the harness's FP32 reference, i.e. in the same band as the P3
fallback on the same shapes.

## 5. Which hardware path is enabled

* Mode: `--wmma-mode full-fp16` / `HIERARCHOS_WMMA_MODE=full-fp16`. WMMA remains
  **opt-in**: `WmmaMode::Off` is still the process default and the FP32 Vulkan
  path is unchanged.
* Admission is unchanged: AMD RDNA3-or-newer PCI ID plus the full cooperative
  feature contract (`VK_KHR_cooperative_matrix`, `vulkanMemoryModel`, 16-bit
  storage, `shaderFloat16`, subgroup-size control forcing 32, and a
  SUBGROUP-scope 16x16x16 f16/f16→f16 tile).
* The 16,000,000-MAC profitability predicate is unchanged and still gates every
  production admission point.
* Mixed is a separate, unchanged arm. Unsupported adapters, unsupported shapes
  and below-floor projections keep their existing fallbacks and counters.
* `split8`, `two-bank` and `two-wave` remain committed and selectable through
  `HIERARCHOS_WMMA_FULL_FP16_VARIANT`, exactly as in v0.2.5.

## 6. Module identities

| Module | Source SHA-256 | SPIR-V SHA-256 | Bytes | FNV-1a64 |
| --- | --- | --- | ---: | --- |
| P3 (contract + fallback, unchanged) | `4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12` | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` | 23,280 | `0x076216ad97d7605f` |
| C1 (batched staging) | `9857a8ad457f3b119e0465dde7c4bb36635053e9ee25522bce0a208452953da4` | `b072242034ba7bd0f53690c38c03b11ed587b538480c04ed13ecf19fe2a8b27d` | 36,912 | `0xfecb2a74c4bbbce4` |

Both are built with `glslc -fshader-stage=compute --target-env=vulkan1.2` under
`shaderc v2023.8 v2025.3 / spirv-tools v2025.3 / glslang 11.1.0-1253`, and
`spirv-val --target-env vulkan1.2` accepts both. The C1 bytes are the Phase
LXXXVII candidate's bytes: the promotion renamed the artifact and changed
nothing in it.

## 7. Hardware scope

Every measurement in this document was taken on one adapter: AMD RDNA3, PCI
**`1002:15bf`**, `AMD Radeon Graphics`, driver **`32.0.32015.14001`**, Vulkan
1.4.321 SDK with `spirv-tools v2025.3`, on Windows 11 x64. This is a different
driver build from the v0.2.5 release measurements, so absolute milliseconds are
**not** comparable across the two documents; the ratios here are paired
measurements inside one driver session and are.

The qualification does **not** generalize to other RDNA3 adapters, RDNA3.5,
RDNA4 or non-AMD devices. **Independent second-device acceptance remains
unverified.**

No RGP capture, ISA read-out, VGPR count or hardware occupancy counter was
obtained. Nothing in this document is derived from one.

## 8. Reproducing the record

```bash
cd hierarchos-vulkan
export CARGO_TARGET_DIR=target-v026

# 1. the two modules (needs glslc + spirv-val)
glslc -fshader-stage=compute --target-env=vulkan1.2 \
    shaders/wmma_gemm_f16_f16acc_twowave_p3.comp -o /tmp/p3.spv
glslc -fshader-stage=compute --target-env=vulkan1.2 \
    shaders/wmma_gemm_f16_f16acc_twowave_p3_c1.comp -o /tmp/c1.spv
# 2aff4123… / b0722420… respectively

# 2. release gate and its controls, CPU-side
python validation/check_c1_release.py
python -m unittest discover -s validation -p "test_c1_release.py"

# 3. build the qualification harnesses, then reproduce §2
cargo build --release --bin wmma_bench --bin wmma_qualify
python ../phase_lxxxiv_evidence/run_lxxxiv_accuracy.py \
    --exe $CARGO_TARGET_DIR/release/wmma_qualify.exe --out validation/phase_lxxxviii_c1_promotion/candidate_accuracy
python validation/phase_lxxxvi_wmma_optimization/compare_dumps.py \
    validation/phase_lxxxviii_c1_promotion/baseline_accuracy/dumps \
    validation/phase_lxxxviii_c1_promotion/candidate_accuracy/dumps

# 4. reproduce §4 and §3's sweep
python validation/phase_lxxxviii_c1_promotion/run_production_campaign.py \
    --out validation/phase_lxxxviii_c1_promotion/campaign_protected \
    --base-exe validation/phase_lxxxvi_wmma_optimization/bin/wmma_bench_v025.exe \
    --cand-exe $CARGO_TARGET_DIR/release/wmma_bench.exe --pairs 24 --iterations 16
python validation/phase_lxxxviii_c1_promotion/run_boundary_sweep.py \
    --exe $CARGO_TARGET_DIR/release/wmma_bench.exe \
    --out validation/phase_lxxxviii_c1_promotion/boundary_sweep --iterations 4
```

The full receipts, dump pins and module identities are in
`hierarchos-vulkan/validation/phase_lxxxviii_c1_promotion/`.
