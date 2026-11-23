# Phase LXXXVIII — C1 staging promotion, qualification and v0.2.6

Status: **PROMOTED AND QUALIFIED.** Candidate C1 from Phase LXXXVII is
integrated into the production Full FP16 route, requalified after integration,
and shipped as v0.2.6. The original v0.2.5 P3 module is preserved byte for byte
and remains the production fallback.

The release note is [RDNA3_C1_STAGING_RELEASE.md](../../../RDNA3_C1_STAGING_RELEASE.md).
This directory is the evidence.

---

## A. Recovery of the candidate

| Artifact | Result |
| --- | --- |
| Candidate source (`candidates/…_p3_batch.comp` in Phase LXXXVII) | SHA-256 `9857a8ad457f3b119e0465dde7c4bb36635053e9ee25522bce0a208452953da4`, 26,229 B |
| Recompiled with `glslc -fshader-stage=compute --target-env=vulkan1.2` | SPIR-V `b072242034ba7bd0f53690c38c03b11ed587b538480c04ed13ecf19fe2a8b27d`, 36,912 B — **byte-identical to the archived module** |
| `spirv-val --target-env vulkan1.2` | clean |
| Production artifact | `shaders/wmma_gemm_f16_f16acc_twowave_p3_c1.{comp,spv}` — the same bytes, renamed |

The recovery rebuilt the module from the preserved source rather than trusting
the archive; both hashes match the Phase LXXXVII record, so the integrated
candidate is the measured candidate and not a reconstruction.

`identities.json` pins both modules, all four harness binaries and the toolchain.

## B. Integration

`hierarchos-vulkan/src/wmma.rs`:

* `WMMA_GEMM_F16ACC_TWOWAVE_P3_C1_SPV` embeds the C1 module;
* `FullFp16Variant::staged_spirv`/`staged_shader_name` give **only** `P3` a
  batched-staging module; `spirv()`/`shader_name()` still name the P3 contract
  module, so the v0.2.5 identity and its gate are unchanged;
* `StagingGeometry` + `batched_staging_eligible` are the pure, device-free
  selector;
* `WmmaGemm::record_gemm` chooses the staging kernel per dispatch inside the
  `FullFp16Accumulate` arm only, and counts both routes
  (`staging_batched` / `staging_masked`);
* every other arm, mode, threshold, fallback and admission point is untouched.

There is no new mode, flag, variant name or environment variable. C1 is not
selectable by a user; it is selected by proof.

## C. Selector and boundary evidence

`run_boundary_sweep.py` runs each geometry through the v0.2.6 production binary
with no override and reads the dispatched shader from the GPU profile stream.
`SELECTOR_SWEEP.json` records 17/17 correct:

| Group | Shapes | Module dispatched |
| --- | --- | --- |
| Aligned neighbours (10) | `128x4096x768`, `128x16384x768`, `256x8192x768`, `512x1024x768`, `1024x768x768`, `512x768x1536`, `2048x768x1024`, `480x768x768`, `512x736x768`, `512x768x752` | `…_p3_c1` |
| Unaligned neighbours (7) | `511x768x768`, `255x768x768`, `512x767x768`, `512x2047x768`, `512x768x767`, `2048x768x2047`, `16x128x32` | `…_p3` |

Every cooperative arm stayed within `8.9e-4` of the harness FP32 reference.

The unit tests in `wmma.rs` cover the same contract without a device
(`batched_staging_admits_every_protected_geometry`,
`batched_staging_rejects_every_unproven_geometry`,
`production_staging_modules_are_the_qualified_bytes`), plus one device test that
asserts both routes are reachable and counted
(`production_selector_reaches_both_staging_modules`).

## D. Numerical qualification

Produced by the **unmodified** `phase_lxxxiv_evidence/run_lxxxiv_accuracy.py`
over eight seeds × five mode/oracle configurations, run twice: once with the
archived v0.2.5 harness (`9761361b…`), once with the v0.2.6 build.

| Check | Baseline | Candidate |
| --- | --- | --- |
| Receipts | 40 | 40 |
| Receipts with a failing gate | 15 | 15 |
| Configurations with failing gates | `full-fp16/rdna3-dot2`, `full-fp16/split2` | identical |
| `full-fp16/tile16` | 37/37 on all eight seeds | 37/37 on all eight seeds |
| Maximum tile16 `max_abs` | `0.03125` | `0.03125` |
| Per-seed failure counts (dot2 / split2) | `8,6,6,6,4,5,5,6` / `3,0,1,1,1,1,2,3` | identical |
| Production dumps | 8 | 8 |

`compare_dumps.py` over the two dump sets:

```text
cases=120 bit_identical=120 differing=0 missing_files=0
BIT-IDENTICAL COMPARISON: PASS (every production case, every element)
```

Every production receipt reports **5 batched + 10 masked** of 15 cooperative
dispatches, so both routes ran inside the qualification set. The 16 dumps are
259,500,114 B and are pinned by size and SHA-256 in `dumps_manifest.json` rather
than committed.

The `split2` and `rdna3-dot2` diagnostic failures remain recorded as diagnostics.
Nothing was reclassified.

## E. Performance qualification

`campaign_protected/` — 24 paired blocks per shape, ABBA/BAAB position balance,
GPU timestamp queries, strictly serial, two executables (v0.2.5
`wmma_bench_v025.exe` `8fcdc4a8…` vs the v0.2.6 build). The candidate arm never
sees a variant override; it is the production selector.

| Shape | P3 ms | C1 ms | Speedup | Corrected 95 % CI | Module |
| --- | ---: | ---: | ---: | --- | --- |
| 2048×768×2048 | 5.0217 | 2.9347 | 1.7074 | [1.6976, 1.7171] | C1 |
| 512×2048×768 | 1.2376 | 0.9316 | 1.3286 | [1.3182, 1.3391] | C1 |
| 512×768×768 | 0.5352 | 0.2570 | 2.0920 | [2.0415, 2.1425] | C1 |
| 128×8192×768 | 1.3955 | 1.0067 | 1.3782 | [1.3580, 1.3983] | C1 |
| 64×64×64 (control) | 0.0387 | 0.0311 | 1.2450 | [1.2423, 1.2477] | C1 |
| 16×128×32 (control) | 0.0322 | 0.0321 | 1.0021 | [0.9998, 1.0044] | P3 |

Both controls are below the unchanged 16M-MAC profitability floor, so production
never dispatches them; they are reported and excluded from the acceptance rule.

`run_production_campaign.py` is the driver. It differs from Phase LXXXVII's in
two ways, both recorded here rather than hidden: the arms are two executables
(there is no variant override to abuse any more), and the profile stream is
segmented positionally with a checked split because the candidate's staging
module legitimately changes per shape.

## F. Gate

`validation/check_c1_release.py` runs `check_p3_release.py` in full and
unmodified first — the v0.2.5 gate is a precondition, not a fallback — then pins
the C1 bytes, the variant containment, the predicate's conditions, the fallback,
the counters, the absence of any forced route, and the barrier count. Its 20
negative controls are `test_c1_release.py`. The 23 inherited controls still run
as `test_p3_release.py`.

## G. What was not done

* **No RGP, ISA, VGPR or occupancy counter.** Not available in this checkout, and
  none is claimed.
* **No second physical device.** Independent second-device acceptance remains
  unverified.
* **No end-to-end training-step timing.** Every number is a per-dispatch GPU GEMM
  time.
* **No new optimization search.** C1 was integrated, qualified and released; no
  further candidate was explored.
* **The Phase LXXXVI profiling dumps remain unavailable** and their historical
  hashes remain unreverified. Nothing in this phase depends on them: the dumps
  used here were generated in this phase and pinned in `dumps_manifest.json`.

## H. Reproducing

See §8 of [RDNA3_C1_STAGING_RELEASE.md](../../../RDNA3_C1_STAGING_RELEASE.md).
