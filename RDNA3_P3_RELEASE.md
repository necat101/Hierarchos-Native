# RDNA3 Full FP16 WMMA — P3 epilogue release

Status: **production Full FP16 epilogue, opt-in mode, qualified on the measured
RDNA3 device.** Read §6 for the disclosed direct-kernel limitation and §4 for the
informational tile16 headroom before merging.

This document is the release note for the Phase LXXXI promotion of the Phase LXXIX
P3 epilogue. It does not rewrite any historical receipt. The Phase LXXIX,
LXXX, LXXIII and LXXV evidence directories remain the source of record for the
numerical investigation and for the performance campaigns that preceded this
release.

---

## 1. What changed

The Full FP16 WMMA arm (FP16 operands, FP16 accumulation, FP32 masters) previously
dispatched one of three committed shaders, defaulting to `split8`
(`wmma_gemm_f16_f16acc.spv`). It now defaults to the **P3 epilogue module**:

| | split8 / two-bank / two-wave | P3 (production) |
| --- | --- | --- |
| Bank partials | 8 binary16 banks, slices `b, b+8, …` | **identical** |
| Workgroup / staging / barriers | as committed | **identical to two-wave** |
| Level 1 | `l_i = f16(bank[2i] + bank[2i+1])`, 4 roundings | — |
| Level 2 | `f16(f16(l0+l1) + f16(l2+l3))`, 3 roundings | — |
| Merge | — | `s = ((((((b0+b1)+b2)+b3)+b4)+b5)+b6)+b7` in **float32** |
| Output rounding | 7 binary16 roundings | **1** binary16 rounding: `f16(s)` |

P3 is an epilogue-only change. Bank assignment, the K-slice stream, the
two-wave workgroup structure, the cooperative-matrix multiply, the A/B staging,
the barrier placement and the per-bank binary16 accumulation are unchanged.
Total LDS stays at 9,216 B: the `part[4][3][256]` tree scratch becomes one
binary16 staging slot per tile plus a float32 accumulator tile.

`split8`, `two-bank` and `two-wave` remain available and selectable through
`HIERARCHOS_WMMA_FULL_FP16_VARIANT`; the previous pipeline is preserved as a
rollback path and for comparison. `wmma-full-f16acc` remains the arm label, so
existing receipts stay comparable.

No threshold moved:

| Constant | Value | File |
| --- | ---: | --- |
| `WMMA_MIN_PROFITABLE_MACS` | `16_000_000` | `hierarchos-vulkan/src/wmma.rs` |
| `ULP_GATE_LIMITS` | `0.03125` | `hierarchos-vulkan/validation/check_phase_lxv.py` |
| `MARGIN_TARGET` | `0.25` (historical, informational) | `hierarchos-vulkan/validation/check_phase_lxv.py` |

## 2. Which hardware path is enabled

* Mode: `--wmma-mode full-fp16` / `HIERARCHOS_WMMA_MODE=full-fp16`. WMMA remains
  **opt-in**: `WmmaMode::Off` is still the process default, and the FP32 Vulkan
  path is unchanged.
* Admission is unchanged: AMD RDNA3-or-newer PCI ID plus the full cooperative
  feature contract (`VK_KHR_cooperative_matrix`, `vulkanMemoryModel`, 16-bit
  storage, `shaderFloat16`, subgroup-size control forcing 32, and a
  SUBGROUP-scope 16x16x16 f16/f16→f16 tile).
* Selection is the production selector, not a qualification route: `WmmaGemm::new`
  resolves `FullFp16Variant::from_env()` and now defaults to
  `FullFp16Variant::P3`. `wmma_bench` reports
  `full_fp16_variant: p3` and the pipeline identity below.
* Profitability is unchanged and is applied at every production selection point:
  `ProjectionWmma::admits` at construction **and** per recorded batch (new: a
  smaller batch than `max_rows` must clear the same floor), and the Transformer
  graph's `wmma_shape_profitable` gate. `LinearWmma` additionally retains
  `FULL_FP16_MAX_REDUCTION = 64` per phase.
* The Mixed arm is untouched: `MixedFp32Accumulate` still dispatches
  `wmma_gemm_f16_f32acc.spv` and does not consult the Full FP16 variant.

## 3. Artifact identity

| Item | Value |
| --- | --- |
| Source | `hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.comp` |
| Source SHA-256 | `4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12` |
| SPIR-V | `hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.spv` |
| SPIR-V SHA-256 | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` |
| SPIR-V size | 23,280 B |
| SPIR-V signature (FNV-1a64) | `0x076216ad97d7605f` |
| Pipeline identity (runtime) | `wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280` |

The connection to the qualified Phase LXXIX artifact is exact, not nominal:
Phase LXXIX's experimental receipt for seed 42 recorded
`phase-lxxix-p3/experimental/sha256:2aff4123…/spirv-fnv1a64:0x076216ad97d7605f/len:23280`,
and the production receipt reports the same `sha256:2aff4123…`, the same
`spirv-fnv1a64:0x076216ad97d7605f` and the same `len:23280`. The production
selector loads the same 23,280 bytes through `include_bytes!`; no environment
variable or qualification flag is involved.

Reproducibility: the committed `.spv` is regenerated **byte-for-byte** from the
committed `.comp` on this machine with

```
glslc --target-env=vulkan1.2 wmma_gemm_f16_f16acc_twowave_p3.comp -o wmma_gemm_f16_f16acc_twowave_p3.spv
```

(shaderc v2023.8 / v2025.3, spirv-tools v2025.3, glslang 11.1.0-1253; no
diagnostics). `spirv-val --target-env vulkan1.2` is clean on both the committed
and the regenerated module. Unlike the older two-wave module, P3 is byte
reproducible with the installed SDK.

## 4. Numerical acceptance

The production route was re-executed on all 32 preserved K=96 cases (8 seeds × 4
layouts) with **no override and no experimental flag**. Every output tensor is
bit-identical to the Phase LXXIX hardware P3 tensor, and the input tensors hash
to the preserved operand hashes:

| Check | Result |
| --- | --- |
| Inputs identical to the preserved operands | **32/32** |
| Production output == Phase LXXIX GPU P3 output | **32/32** (0 differing elements) |
| Existing numerical gates | **37/37, 0 failed, all eight seeds** |
| Dispatch counts per seed | `cooperative=15 plain_f16=1 shape_fallbacks=0` |

The Phase LXXIX numerical results are unchanged and remain the numerical record
for P3:

| Measurement | Value |
| --- | ---: |
| Genuine GPU output tensors | 32 |
| Cases improved vs REF64F16 | **31/32** |
| Cases worsened vs REF64F16 | **0/32** |
| Maximum REF64F16 error | `0.03035259` → **`0.02593803`** |
| Maximum tile16 distance | `0.0390625` → **`0.03125`** |
| Existing gates | **37/37 for every seed** |
| Repeated GPU dispatch | bit-identical |
| Hardware-bank model vs GPU output | **0 / 524,288** differing |

The hard correctness gate is `ULP_GATE_LIMITS = 0.03125` and is unchanged.
The kernel meets it on every gate, every seed.

### The tile16 25 % headroom is informational, not a gate

`MARGIN_TARGET = 0.25` is a historical target recorded by the Phase LXV checker;
its provenance is `hierarchos-vulkan/validation/check_phase_lxv.py` and
`validation/PHASE_LXVI_ACCURACY_MARGIN.md`. It asks for `0.0234375` on the tile16
distance. That target is **not met and is not a release blocker**:

* 7/32 cases meet `0.0234375`; the P3 maximum is the hard limit `0.03125`.
* At 12 coordinates in 11/32 cases, every representable value within `0.0234375`
  of tile16 is at least one ULP away from the correctly rounded exact sum. Meeting
  the stricter margin would require deliberately departing from the correctly
  rounded result.
* `check_phase_lxv.py` reports `target_met` but never `require`s it; no checker
  treats the headroom as a hard blocker, and none was changed here.

The independent REF64 and REF64F16 references, the coordinate-level lower-bound
receipts (`phase_lxxix_evidence/k96_lower_bound_coords.json`) and the historical
`MARGIN_TARGET` value are preserved unmodified.

## 5. Performance qualification

Protocol (same as Phases LXXIII/LXXV): Vulkan GPU timestamp queries bracket every
compute dispatch (`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`,
`HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`), `--rounds 1 --iterations 16`, three
warm-up dispatches dropped per shape, strictly serial one-process-at-a-time
execution, 24 paired blocks per shape, two-sided 95 % paired Student-t over
per-pair ratios (`hierarchos-vulkan/validation/abba_stats.py`, unmodified). Two
designs were run: alternating `ABBA`, and the position-balanced `ABBA`/`BAAB`
schedule that removes the arm/position confound identified in Phase LXXV.

Arms: Original Main `81ac82fa…` (`f41caea0…`), the preserved two-wave candidate
`bd7434ed…` (`178fb339…`), and the release binary (`6b291c1f…`, variant `p3`).
Every run's self-reported variant and shader matched its declared arm (`p3` /
`wmma_gemm_f16_f16acc_twowave_p3` for the release arm; Main reports no selector
and dispatches its split8 module), 48 runs per campaign completed with rc 0, and
the device reported `timestamp_period_ns = 10.0` in every stream.

### P3 vs Original Main — protected production shapes

| Shape | Main ms | P3 ms | Speedup | Corrected 95 % CI (balanced) | ABBA speedup | ABBA 95 % CI |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 2048×768×2048 | 5.4178 | 5.3140 | **1.0239** | [1.0149, 1.0330] | 1.0196 | [1.0102, 1.0290] |
| 512×2048×768 | 1.4998 | 1.3336 | **1.1220** | [1.1120, 1.1319] | 1.1186 | [1.1122, 1.1249] |
| 128×8192×768 | 1.8065 | 1.4257 | **1.2717** | [1.2550, 1.2884] | 1.2759 | [1.2612, 1.2905] |
| 512×768×768 | 0.6228 | 0.5470 | **1.1481** | [1.1178, 1.1784] | 1.1357 | [1.1098, 1.1616] |

Every lower bound is strictly above 1.0 in both designs. The acceptance rule
(point estimate and corrected CI lower bound ≥ 1.0 on each protected shape) is
met on all four. No pooling, no geometric-mean substitution, no shape carries
another.

### P3 vs the two-wave candidate — cost of the epilogue

| Shape | Two-wave ms | P3 ms | P3 speedup vs two-wave | 95 % CI (balanced) | ABBA 95 % CI |
| --- | ---: | ---: | ---: | --- | --- |
| 2048×768×2048 | 5.2849 | 5.3114 | 0.9981 | [0.9895, 1.0066] | [0.9811, 0.9951] |
| 512×2048×768 | 1.3182 | 1.3379 | 0.9859 | [0.9761, 0.9957] | [0.9814, 0.9975] |
| 128×8192×768 | 1.4165 | 1.4238 | 0.9928 | [0.9804, 1.0052] | [0.9800, 0.9964] |
| 512×768×768 | 0.5326 | 0.5509 | 0.9628 | [0.9415, 0.9842] | [0.9637, 1.0068] |

The single-rounding epilogue costs 0.2–3.7 % against the two-wave tree (≈1 % on
three of the four shapes). That cost buys the accuracy improvement and still
leaves P3 faster than Original Main on every protected shape.

### Controls, recorded but not acceptance-bearing

| Shape | P3 vs Main (balanced) | 95 % CI | P3 vs Main (ABBA) | Note |
| --- | ---: | --- | ---: | --- |
| 64×64×64 | 1.1056 | [1.1024, 1.1089] | 1.1035 | faster than Main's split8 |
| 16×128×32 | **0.7593** | [0.7549, 0.7637] | 0.7543 | direct-kernel regression, see §6 |

## 6. Direct-kernel limitation and the profitability safeguard

The `16×128×32` control is a real regression and is disclosed: P3 is ~32 % slower
than Original Main at this geometry, and ~16 % slower than the two-wave candidate.
The historical two-wave figure was `0.87768 [0.87136, 0.88401]`; P3's extra
float32 accumulator tile makes the tiny shape worse.

The geometry is **65,536 MACs**, roughly **244× below**
`WMMA_MIN_PROFITABLE_MACS = 16,000,000`, and it fails `wmma_shape_eligible`'s
tile-fill requirement. Both production admission sites apply the profitability
predicate before an executor is consulted, and `ProjectionWmma` applies it again
to each recorded batch, so the production route never dispatches WMMA here. The
regression is excluded from production acceptance by policy, not hidden: the
benchmark arm is explicit, and the measurement is preserved.

The predicate was not weakened. `WMMA_MIN_PROFITABLE_MACS` is unchanged.

## 7. Hardware-specific scope

All measurements in this document were taken on one device:

| Property | Value |
| --- | --- |
| Adapter | `AMD Radeon Graphics` |
| PCI | vendor `0x1002`, device `0x15bf` (RDNA3, Phoenix) |
| Vulkan | 1.4.344, driver 2.0.388 |
| Cooperative tile | 16×16×16 subgroup-scope f16/f16→f16, subgroup forced to 32 |

No claim is made for any other RDNA3 part, for RDNA3.5 or RDNA4, or for any
non-AMD adapter. The route is gated by the existing capability contract; this
release adds no new hardware admission and no new hardware claim.

## 8. What was verified, and what was not

Verified:

* `cargo test --lib`: **1,177 passed, 0 failed, 40 ignored**, including the new
  constructor/selector tests and the unchanged Mixed tests.
* `cargo clippy --lib --no-deps`: exit 0 (pre-existing warnings only).
* Shader compile + `spirv-val` (committed and regenerated modules).
* Production-route GPU smoke over all 32 K=96 cases: bit-identical to the
  qualified P3 tensors; 37/37 gates on all eight seeds.
* Mixed qualification: 37/37 gates, seed 42, unchanged path.
* Four protected shapes: Main parity or better in both ABBA and balanced designs.

Not verified, stated plainly:

* No RGP capture and no occupancy measurement (no Radeon Developer Panel on this
  machine); register/LDS figures are static code-object facts only.
* No end-to-end training-step speedup is claimed; the numbers are per-dispatch
  GPU GEMM timings.
* Other RDNA3 parts, RDNA3.5/RDNA4 and non-AMD devices are untested.
* The `16×128×32` kernel regression is measured, not fixed.
* `cargo fmt --check` is not clean on the untouched baseline, so it is not an
  enforced repository gate; the added code is rustfmt-canonical (checked by
  diffing the formatter's output against the added regions).
