# PHASE LXXXI — FINAL RDNA3 RELEASE REPORT

**Terminal status: `RDNA3 RELEASE PR READY — P3 PRODUCTION VERIFIED — FOUR-SHAPE PERFORMANCE QUALIFIED`**

Date: 2026-11-21. Device: `AMD Radeon Graphics`, PCI `1002:15bf` (RDNA3,
Phoenix), Vulkan 1.4.344, driver 2.0.388.

This report is the release record for the Phase LXXXI work. It supersedes
nothing: every Phase LXXVIII–LXXX receipt, tensor, manifest and report is
untouched, and no historical number was rewritten.

---

## 1. Starting and ending repository state

| Item | Starting (verified at phase start) | Ending (verified at phase end) |
| --- | --- | --- |
| Original Main (local) | `81ac82fa972d84025129e7c3f0e58fdece7df786` | unchanged |
| `origin/main` | `94c4db1f18e57680d691bcafc9dc6f5b0ab292f6` | `94c4db1f18e57680d691bcafc9dc6f5b0ab292f6` (re-fetched) |
| Two-wave candidate | `bd7434ed8de2055f34dfd6fabdb94d2c2ccb2e1a` | unchanged, preserved |
| Two-wave parent | `f97d8de2e7e8db6b24b41daa115eb63656845836` | unchanged |
| PR #8 | OPEN, MERGEABLE, head `4f115f2e3b968eabc42872769001e6c7b49f2e50` | OPEN, MERGEABLE, head unchanged; untouched by this phase |
| PR #8 ancestry | ancestor of the candidate (`git merge-base --is-ancestor` exit 0) | still an ancestor of the release head |
| Release branch | did not exist | `codex/rdna3-p3-release-20261121` |

No branch was deleted, rebased, amended or force-pushed. Mixed Candidate A
(`23a95fc3…`) was not discarded or replaced. Local `main` was not moved and
nothing was pushed to `main`.

## 2. Release branch and exact commit hashes

| Role | Ref |
| --- | --- |
| Release branch | `codex/rdna3-p3-release-20261121` |
| Release commit (production integration, tests, docs) | `88a114670bb0ef384b4cf9b1007df1787cf9376e` |
| Evidence commit (this directory) | `9e92b71f898a0085e90bd955e43790667ceff57b` |
| Parent (preserved two-wave candidate) | `bd7434ed8de2055f34dfd6fabdb94d2c2ccb2e1a` |
| Base | `main` @ `94c4db1f18e57680d691bcafc9dc6f5b0ab292f6` |

The release commit is additive: one clean commit on top of the preserved
two-wave candidate, with the evidence directory added by a second commit. No
history was rewritten to produce either.

## 3. P3 production source and SPIR-V hashes

| Artifact | Path | SHA-256 | Size |
| --- | --- | --- | ---: |
| Shader source | `hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.comp` | `4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12` | — |
| SPIR-V (committed, embedded) | `hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.spv` | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` | 23,280 B |
| Runtime pipeline identity | reported by `wmma_bench` | `wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280` | — |
| `hierarchos-vulkan/src/wmma.rs` | production selector + tests | `4ebdbd2291781cb50db7bb200feb1fc9dbbaf92798e6e425d569a731db7a891e` | 94,874 B |
| `hierarchos-vulkan/src/projection.rs` | per-batch admission | `1364fe4e58ce15390f4d3469d69a8cf4f6ba1896e3d09a3a6ce6d24c0da97397` | 30,967 B |
| `wmma_bench.exe` (release) | campaign binary, variant `p3` | `6b291c1f9e19f625a516d4d25a0071db6f139f6101b41927ab0ed8ea7100ef12` | — |
| `wmma_qualify.exe` (release) | smoke binary | `0c2ef4007ba99942207f112fd8fcb94ba40431782b787e00148709eef8e6a01b` | — |
| `wmma_bench_main.exe` (preserved) | Original Main arm | `f41caea0c0a97fa017063de17e2f902722bcec6fa4c758362af2f504aa7538e0` | — |
| `wmma_bench_twowave.exe` (preserved) | two-wave arm | `178fb339e8372b99ff3c921e8886683b5632999f05c7ba698722085e48dff222` | — |

**Qualified artifact → production artifact.** Phase LXXIX's experimental receipt
for seed 42 recorded
`phase-lxxix-p3/experimental/sha256:2aff4123…/spirv-fnv1a64:0x076216ad97d7605f/len:23280`.
The production receipt on the release build records the same
`sha256:2aff4123…`, the same FNV-1a64 signature `0x076216ad97d7605f` and the
same length 23,280 B. The bytes embedded by `include_bytes!` are the qualified
bytes; the selector differs, the module does not.

**Reproducibility.** `glslc --target-env=vulkan1.2
wmma_gemm_f16_f16acc_twowave_p3.comp -o /tmp/p3.spv` reproduces the committed
SPIR-V byte-for-byte (`cmp` clean, SHA-256 identical).
Compiler receipt: `shaderc v2023.8 v2025.3`, `spirv-tools v2025.3
v2022.4-833-g33e02568`, `glslang 11.1.0-1253-gefd24d75`, no diagnostics.
`spirv-val --target-env vulkan1.2` (SPIRV-Tools v2025.3) is clean on the
committed and the regenerated module.

## 4. Constructor and selector changes

| Change | Detail |
| --- | --- |
| `FullFp16Variant::P3` | new variant, `#[default]`; `shader_name() = "wmma_gemm_f16_f16acc_twowave_p3"`; `spirv() = WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV` |
| Enumeration order | `Split8, TwoBank, TwoWave, P3` — `Default` is explicit, so order does not decide dispatch |
| Env parsing | `p3`, `twowave-p3`, `two-wave-p3` accepted; unknown values still return `None` (fail closed to the default) |
| `full_fp16_variant()` (experimental executor) | doc now states that it reports the executor's own chosen variant and never the production default; semantics unchanged, no test was bent to it |
| `ProjectionWmma` per-call admission | the three recording sites (`record_forward`, `record_weight_grad`, `record_input_grad`) filter `select_arm` with `ProjectionWmma::admits(rows, input_dim, output_dim)`, so a small batch cannot reach the cooperative arm after the constructor admitted `max_rows` |
| Removed | the dead `admits_batch` helper (it was never called); its contract is now enforced inline at every site that records a batch |
| Unchanged | `WMMA_MIN_PROFITABLE_MACS = 16_000_000`, `FULL_FP16_MAX_REDUCTION = 64`, `wmma_shape_eligible`, `wmma_shape_profitable`, the capability gate, the Mixed arm and every FP32 path |
| Misplaced doc comment (Phase LXXX H.1) | not present in the release tree; nothing to correct |
| Qualification-only entry point | the Phase LXXIX experimental executor is not referenced by any production constructor; production dispatch imports only the committed `.spv` |

## 5. Test results and exact command receipts

All commands run in `hierarchos-vulkan/` on the release commit.

```
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test --lib
   test result: ok. 1177 passed; 0 failed; 40 ignored; 0 measured; 0 filtered out
   exit 0

CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test --lib wmma::tests
   test result: ok. 19 passed; 0 failed
CARGO_INCREMENTAL=0 ... cargo test --lib wmma_linear::tests
   test result: ok. 4 passed; 0 failed
CARGO_INCREMENTAL=0 ... cargo test --lib projection::tests
   test result: ok. 1 passed; 0 failed

cargo clippy --lib --no-deps
   exit 0 (587 pre-existing warnings, none from the added code)
```

Baseline was `1,174 passed, 0 failed, 40 ignored`; the release adds three tests
and re-uses the existing test structure (no new framework). The new tests:

1. `full_fp16_variant_defaults_to_the_production_p3_module` — `Default::default()`
   is `P3`, its label and shader name are the production ones, and the three
   pre-P3 variants remain resolvable.
2. `p3_is_selected_only_for_profitable_geometry` — `ProjectionWmma::admits`
   accepts the four protected shapes and rejects `16×128×32`, `16×64×320`,
   `1×1×1` and `0`-dimension geometry.
3. `unsupported_modes_keep_their_existing_route` — `Mixed`, `Off` and
   `PlainF16` routing is unchanged by any Full FP16 variant, and `split8`
   remains selectable.
4. `production_default_dispatches_the_p3_module` — the embedded module's
   SHA-256, `/len:23280` signature and absence of `/experimental/` in the
   reported identity.
5. Extended `full_fp16_variant_selects_the_shader_it_dispatches` (distinct
   modules per variant) and the `ProjectionWmma` test above.

Formatting: `cargo fmt --check` is **not** clean on the untouched baseline, so
it is not an enforced repository gate. The added code is rustfmt-canonical
(verified by diffing `rustfmt` output against the added regions; one collapsed
`assert_eq!` was fixed).

## 6. GPU production-route smoke results

`phase_lxxxi_evidence/run_lxxxi_production_smoke.sh` runs the release
`wmma_qualify` with **no variant override and no experimental flag** over the 32
preserved K=96 cases (8 seeds × 4 layouts) and compares against the preserved
Phase LXXIX hardware P3 tensor hashes.

| Check | Result |
| --- | --- |
| Input tensors match the preserved operands | **32/32** |
| Production output == Phase LXXIX GPU P3 output | **32/32**, 0 differing elements |
| Numerical gates | **37/37, 0 failed**, every seed |
| Device / mode | `AMD Radeon Graphics` / `full-fp16` |
| Dispatch counts per seed | `{cooperative: 15, plain_f16: 1, shape_fallbacks: 0}` |
| Variant / shader reported by the campaign binary | `p3` / `wmma_gemm_f16_f16acc_twowave_p3` |
| Mixed arm re-qualification (seed 42) | 37/37, 0 failed, unchanged path |

The production route therefore executes the same arithmetic, workgroup
structure and hardware-bank behavior as the qualified experiment. The hardware
bank discrepancy stays documented (104 / 98,304 bank elements, 1,144 / 524,288
final elements vs the offline model; hardware-bank calculation matches the GPU
0 / 524,288; repeated dispatch bit-identical) and is not a rejection reason.

## 7. Four-shape performance table with corrected confidence intervals

Method (unchanged from Phases LXXIII/LXXV): GPU timestamp queries around every
compute dispatch, `HIERARCHOS_VULKAN_PROFILE_KERNELS=1`,
`HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`, `--rounds 1 --iterations 16`, three
warm-up dispatches dropped, serial single-process execution, 24 paired blocks
per shape, two-sided 95 % paired Student-t over per-pair ratios
(`validation/abba_stats.py`, unmodified). Two designs: alternating `ABBA` and
position-balanced `ABBA`/`BAAB`.

### P3 vs Original Main `81ac82fa…`

| Shape | Main ms | P3 ms | Speedup | Corrected 95 % CI (balanced) | ABBA speedup | ABBA 95 % CI |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 2048×768×2048 | 5.4178 | 5.3140 | **1.0239** | [1.0149, 1.0330] | 1.0196 | [1.0102, 1.0290] |
| 512×2048×768 | 1.4998 | 1.3336 | **1.1220** | [1.1120, 1.1319] | 1.1186 | [1.1122, 1.1249] |
| 128×8192×768 | 1.8065 | 1.4257 | **1.2717** | [1.2550, 1.2884] | 1.2759 | [1.2612, 1.2905] |
| 512×768×768 | 0.6228 | 0.5470 | **1.1481** | [1.1178, 1.1784] | 1.1357 | [1.1098, 1.1616] |

Acceptance: point estimate **and** corrected CI lower bound ≥ 1.0 per protected
shape — **passes on all four**, in both designs (lowest lower bound 1.0102).
No pooling, no geometric mean, no cross-shape compensation.

### P3 vs the preserved two-wave candidate `bd7434ed…`

| Shape | Two-wave ms | P3 ms | P3 speedup | 95 % CI (balanced) | ABBA 95 % CI |
| --- | ---: | ---: | ---: | --- | --- |
| 2048×768×2048 | 5.2849 | 5.3114 | 0.9981 | [0.9895, 1.0066] | [0.9811, 0.9951] |
| 512×2048×768 | 1.3182 | 1.3379 | 0.9859 | [0.9761, 0.9957] | [0.9814, 0.9975] |
| 128×8192×768 | 1.4165 | 1.4238 | 0.9928 | [0.9804, 1.0052] | [0.9800, 0.9964] |
| 512×768×768 | 0.5326 | 0.5509 | 0.9628 | [0.9415, 0.9842] | [0.9637, 1.0068] |

The single-rounding epilogue costs 0.2–3.7 % against the two-wave tree (≈1 % on
three of four shapes). That is the measured price of the accuracy improvement,
not a production regression: P3 beats Original Main on every protected shape.

### Controls (recorded, not acceptance-bearing)

| Shape | P3 vs Main (balanced) | 95 % CI | P3 vs Main (ABBA) | P3 vs two-wave |
| --- | ---: | --- | ---: | ---: |
| 64×64×64 | 1.1056 | [1.1024, 1.1089] | 1.1035 | 0.8864 |
| 16×128×32 | 0.7593 | [0.7549, 0.7637] | 0.7543 | 0.8636 |

### Campaign integrity

* 48 runs per campaign, every `rc = 0`; `abba_stats.load_dir` rejects anything
  else.
* One executable SHA-256 per arm per campaign (table in §3); the release arm is
  `6b291c1f…` in all four campaigns.
* The release arm self-reported `variant = p3` and shader
  `wmma_gemm_f16_f16acc_twowave_p3` in every sample stream; the Main arm reports
  no selector and its split8 module; the two-wave arm reports `two-wave` and
  `wmma_gemm_f16_f16acc_twowave`.
* 384 timestamp samples per arm per shape per campaign (24 × 16);
  `timestamp_period_ns = 10.0` everywhere.
* The earlier campaigns run against the pre-final build (`9f804b35…`, comment
  and blank-line differences only) are preserved under
  `superseded_9f804b35/`; they are not used for acceptance.

## 8. Numerical acceptance results

| Measurement | Result |
| --- | --- |
| Hard gate `ULP_GATE_LIMITS` | `0.03125` — **unchanged** |
| Gates passed | **37/37, 0 failed**, all eight seeds (GPU, production route) |
| REF64F16 maximum error | `0.02593803` (was `0.03035259`) |
| Cases improved vs REF64F16 | **31/32** |
| Cases worsened vs REF64F16 | **0/32** |
| Maximum tile16 distance | `0.03125` (was `0.0390625`) |
| Independent references | REF64 and REF64F16 preserved, unmodified |
| Coordinate-level lower-bound receipts | `phase_lxxix_evidence/k96_lower_bound_coords.json`, unmodified |
| Historical `MARGIN_TARGET` | `0.25` at `validation/check_phase_lxv.py`, provenance preserved |
| Historical `16×128×32` figure | `0.87768 [0.87136, 0.88401]`, preserved |

The P3 arithmetic change is an epilogue change only: the 4 + 3 rounding tree is
replaced by a float32 left-to-right sum of the eight binary16 bank partials and
one binary16 rounding. Bank assignment, K-slice order, workgroup structure,
staging, barriers and per-bank accumulation are identical to the two-wave
module, and no intentional numerical error was introduced.

## 9. D1 and D2 decisions as applied

**D1 — production-scoped performance acceptance: applied.** Acceptance was
scoped to the four protected shapes; the direct-kernel `16×128×32` regression is
documented in `RDNA3_P3_RELEASE.md` §6 and in this report (§7 controls) and is
not concealed. `WMMA_MIN_PROFITABLE_MACS` remains `16_000_000`; the geometry is
65,536 MACs, ~244× below the floor. Both production admission sites apply the
predicate, and the release additionally applies it to every recorded batch, so
unprofitable shapes cannot enter P3 by any path. No predicate was weakened.

**D2 — tile16 headroom: applied as informational.** The hard gate stays
`0.03125`. `MARGIN_TARGET = 0.25` (target `0.0234375`) is reported, not
required: 7/32 cases meet it; at 12 coordinates in 11/32 cases every value within
`0.0234375` of tile16 is ≥ 1 ULP from the correctly rounded sum. The checker
`validation/check_phase_lxv.py` already only *reports* `target_met` and never
`require`s it, so no acceptance code needed changing. No numerical calculation,
reference implementation, hard limit or historical receipt was altered, and no
error was introduced to satisfy the historical margin. The distinction between
the hard gate and the informational headroom is stated explicitly in
`RDNA3_P3_RELEASE.md` §4 and `RDNA3_WMMA.md` (release-status section).

## 10. PR number and URL

**PR #9 — https://github.com/necat101/Hierarchos-Native/pull/9**

* Base `main` @ `94c4db1f18e57680d691bcafc9dc6f5b0ab292f6`, head
  `codex/rdna3-p3-release-20261121`.
* Separate from PR #8 (https://github.com/necat101/Hierarchos-Native/pull/8,
  OPEN/MERGEABLE at `4f115f2e…`, untouched) because #8's branch could not be
  updated by fast-forward without rewriting its published commit, and #8's
  subject is the aligned-load change. The relationship is stated in the PR body:
  the release branch is a descendant of #8's head, so if #8 merges first the
  release diff collapses to the release commit.
* Not merged, not force-pushed, nothing pushed to `main`. The merge decision is
  left to the maintainer.

## 11. Remaining limitations and anything not verified

* One device only (`1002:15bf`). No claim for other RDNA3 parts, RDNA3.5, RDNA4
  or non-AMD adapters; no new hardware admission was added.
* No RGP capture and no occupancy measurement (Radeon Developer Panel is not
  available on this machine). Register/LDS statements are static facts.
* The measurements are per-dispatch GEMM timings; no end-to-end training-step
  speedup is claimed or measured.
* The `16×128×32` kernel regression is measured, not fixed; it is excluded from
  production by the unchanged profitability predicate.
* The tile16 25 % headroom is unmet and is now informational by decision D2.
* `cargo fmt --check` is not clean on the untouched baseline and is not enforced;
  the added code was verified rustfmt-canonical by hand.
* The one-ULP behavioral-model discrepancy (104 bank elements, 1,144 output
  elements) remains documented; hardware-bank calculation matches the GPU.
* WMMA remains opt-in; the release does not change any default mode other than
  the Full FP16 variant.
* The loaded/qualified connection is established by SHA-256 and FNV-1a64
  identity; the embedded bytes were not re-disassembled for this report.

## 12. SHA-256 manifest

`SHA256SUMS_phase_lxxxi.txt` in this directory lists every file shipped with
this evidence, relative to `phase_lxxxi_evidence/`, plus the committed release
sources and shader artifacts. The manifest is generated from the files as
committed; verify with `sha256sum -c SHA256SUMS_phase_lxxxi.txt` from this
directory (`--ignore-missing` for the release-repo entries that live outside
it).

---

### Terminal status

`RDNA3 RELEASE PR READY — P3 PRODUCTION VERIFIED — FOUR-SHAPE PERFORMANCE QUALIFIED`

Basis: production integration selects the qualified P3 module and reproduces its
outputs bit-for-bit (32/32); the library suite passes (1,177 / 0 / 40); the hard
numerical gate passes 37/37 on all seeds; and every protected production shape
shows a corrected 95 % CI lower bound above 1.0 against Original Main in two
independent paired designs. PR #9 is open for the maintainer's review.
