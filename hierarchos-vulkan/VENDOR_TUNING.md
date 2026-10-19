# Vendor-specific kernel tuning

The Vulkan training backend keeps one portable SPIR-V module per kernel and a
small, explicit selection layer that swaps in a vendor variant when the physical
device is known. This document describes the contract that makes that swap safe,
the current vendor matrix, the measured results, and how to add or qualify a
variant.

## Why not one kernel for everything

The backend was developed primarily against an AMD Radeon (RDNA, `wave64`,
64 KiB LDS) target. Intel Gen9 integrated parts (HD Graphics 520-class, the 2016
ThinkPad E560 target) differ materially:

| Property | AMD Radeon Graphics (target A) | Intel HD Graphics 520 (target B) |
| --- | --- | --- |
| Execution width | wave64 | SIMD8/16 (subgroup reported as 32) |
| Shared memory | 64 KiB | 32 KiB (`maxComputeSharedMemorySize`) |
| Memory system | dedicated VRAM | shared system DDR (bandwidth-starved) |
| Driver/compiler | AMDVLK/RADV-class | Intel Windows image compiler |

A schedule that is optimal for one is not optimal for the other, but the *math*
must not change. Every vendor variant here is therefore required to be
**bit-identical** to the portable kernel for the same inputs, and the regression
suite asserts that on the executing device with raw `f32` bit patterns.

## Selection layer

`src/vendor.rs` owns the whole decision:

* `GpuVendor` classifies `VkPhysicalDeviceProperties::vendor_id` (PCI IDs, never
  marketing names).
* `VendorKernelFamily` enumerates the kernels the backend constructs.
* `select_kernel` resolves vendor -> capability gates -> SPIR-V bytes. Anything
  missing falls back to the portable module. It is the *capability* view printed
  by `vendor_kernel_plan` / `--kernel-plan`.
* `VendorMatmulKernel` is the matmul slot call sites hold. It owns the portable
  pipeline always plus the vendor pipeline when the capability gates admit it,
  and picks per dispatch from `intel_matmul_geometry_supports_for` (see below).
* Elementwise entry points `silu_forward_kernel`, `silu_backward_kernel`, and
  `adamw_kernel` resolve once at construction; they have no variants and always
  return the portable module.

Mainline code stays vendor-neutral; only the module bytes can differ:

```rust
// before
linear_forward: vulkan::ComputeKernel::new(&device, LINEAR_FORWARD_SPV, 3, 12)?,
// after
linear_forward: vendor::VendorMatmulKernel::new(&device, vendor::VendorKernelFamily::LinearForward)?,
```

Call sites keep calling `.record_dispatch(batch, buffers, push, grid)`; the
wrapper forwards to whichever module the dispatch geometry selects. Call sites
whose push struct carries fields after the three geometry words (the shared head
trainer appends loss parameters) use `new_with_push_constant_bytes` so the
declared pipeline push range still matches what they write.

### Dispatch contract

A variant may change *how* a workgroup computes, never *what* it covers.
`VendorKernelFamily::workgroup_elements()` is 256 for every family, so all
existing host grids (`div_ceil(len, 256)`, `div_ceil(rows, 16)`,
`div_ceil(output_dim, 16)`) stay valid on every vendor. `linear-forward` and
`linear-bias-forward` keep the 16x16 workgroup and the one-output-per-invocation
mapping, which is why call sites need no dispatch changes at all.

### Geometry gate

The matmul geometry is only known when a dispatch is recorded — one shared
kernel serves every layer, and `rows` is runtime data (training/prefill batches
as well as single-row cached decode). `VendorMatmulKernel::record_dispatch`
therefore reads the leading `(rows, input_dim, output_dim)` words of the push
constants every linear-family struct already writes, and routes the dispatch
with the pure gate `intel_matmul_geometry_supports_for`:

```text
rows >= 4  AND  input_dim >= 32  AND  output_dim >= (rows >= 8 ? 128 : 192)
```

Each bound is a measured crossover on the HD Graphics 520 (the full matrix is
below). The row bar exists because a 16-row tile cannot fill its lanes at low
row counts: single-row decode measured **0.61x** (39% slower) with the Intel
module, which is exactly the kind of silent regression the gate prevents. The
short-k bar removes the k=8/16/24 class (0.70x/0.57x/0.55x). The higher output
bar below eight rows keeps half-empty workgroups from running on too few tiles.
A push-constant block shorter than three words, or a geometry below any bar,
dispatches the portable module — never a guess.

That expression is only the *vendor-independent* half of the decision. The
other half asks which portable schedule the dispatch will land on, because the
module the bounds above were measured against is not the module that runs today.

#### The portable modules gained the same tile, so three families' gates moved

The 2026-10-15 portable tiling pass gave `linear_forward`,
`linear_bias_forward` and `linear_residual_forward` the same 16x16
workgroup-staged schedule the Intel variants had been measured against, with a
32-float K stage instead of 16 and therefore half the barriers per K step. That
moved the baseline every number above was measured against. Re-measuring the
same A/B on the same HD Graphics 520 on 2026-10-15 (median GPU timestamp delta
of 60 isolated dispatches per side,
`HIERARCHOS_VULKAN_VENDOR_BENCH_ITERS=60`), `intel/portable` below 1.0 means the
*portable* module is faster:

| rows | k | n | intel/portable | variant selected |
| --- | --- | --- | --- | --- |
| 8 | 64 / 96 / 128 / 192 / 256 / 1024 | 512 | 0.97 / 0.97 / 0.95 / 0.92 / 0.95 / 0.95 | no |
| 8 | 448 | 128 / 256 / 384 / 512 / 2048 | 0.96 / 0.93 / 0.91 / 0.93-0.94 / 0.91-0.93 | no |
| 9 / 10 / 12 | 448 | 512 | 0.95 / 0.96 / 0.97 | no |
| 17 | 448 | 512 | 0.92 | no |
| 24 | 448 | 512 | 0.98 / 0.98 / 0.98 | no |
| 16 | 64 / 128 / 256 / 448 | 512 | 1.03 / 1.04 / 1.03 / 1.02-1.03 | yes |
| 16 | 448 | 2048 | 1.04 | yes |
| 32 | 448 | 512 | 1.03 / 1.03 / 1.05 | yes |
| 48 / 64 | 448 | 512 | 1.06 / 1.04-1.05 | yes |
| 64 | 448 | 2048 | 1.05 | yes |
| 128 | 448 | 512 | 1.07 | yes |
| 128 / 256 | 1024 / 448 | 2048 | 1.05 | yes |
| 8 | 32 | 512 | 2.21 | yes (portable is scalar) |
| 4-7 | 32-96 | 192-320 | 1.12-2.21 | yes (portable is scalar) |
| 15 | 448 | 512 | 1.01 / 1.01 / 1.03 | no (withheld deliberately) |

Read across the top rows: once the portable module tiles (`rows >= 8`,
`input_dim >= 64`, `output_dim >= 16`) the Intel variant lost at every tiled
shape, because it staged only a 16-float K slice — 32 barriers per 512-wide K
step — while the portable module staged 32 and paid half as many. The
interleaved 32-wide K stage that the portable modules gained in the same pass
was therefore ported into the three Intel Gen9 variants, keeping the
instruction-level accumulation order (ascending `k`, one `fma` per element, same
operand order) unchanged.

That closed the gap. Re-measuring the same A/B on the same HD Graphics 520
after the port, `intel/portable` in the tiled regime:

| rows | k | n | intel/portable after the port |
| --- | --- | --- | --- |
| 8 | 448 | 512 | 1.019 / 1.022 / 1.035 (was 0.93-0.94) |
| 8 | 448 | 128 / 256 / 384 / 2048 | 1.005-1.035 (was 0.91-0.96) |
| 16 / 32 / 64 / 128 | 448 | 512 | at or above the pre-port 1.02-1.07 |

The variant now wins or ties across the regime where the portable module tiles,
so no row-count restriction is justified: `intel_matmul_geometry_supports_for`
keeps its original crossover unchanged (`rows >= 8`, `input >= 32`,
`output >= 128`), and the 15-row shape that a `rows % 16 == 0` rule would have
declined is simply admitted on its measured merits.

`linear3-forward` is deliberately exempt. Its portable module still computes one
scalar dot product per invocation — it has no `shared` staging at all — so the
variant keeps winning there (3.60x at 8x448x512, 2.20x at 6x96x320 measured in
the same session) and its gate is unchanged.

Two regression tests keep the host constants and the shader guard from drifting
apart: `intel_matmul_family_gate_matches_the_portable_tiled_measurements` asserts
every row of the table above, and
`portable_tiled_schedule_thresholds_match_the_committed_shader_guards` reads the
committed GLSL and requires its scalar fallback guard to be exactly the one the
host mirrors, and requires `linear3_forward.comp` to still be scalar.

One fixture had to follow the gate: the model-level vendor parity test used to
run eight rows per timestep, which the gate now keeps on the portable module, so
it runs sixteen (see `hierarchos_causal_lm_vendor_kernels_match_portable`).

#### What this is not

The variant was not retired: it still wins where the portable module is scalar
(2.2x at 8x32x512) and it now wins-or-ties in the tiled regime. Nothing else was
re-tuned — no register blocking, no `subgroup` ops, no workgroup-size change — so
this stays a schedule-port that is bit-exact against both the pre-port Intel
module and the portable module (`max_ulp_delta = 0`).

### Environment switches

| Variable | Effect |
| --- | --- |
| `HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS` | Force the portable modules everywhere. |
| `HIERARCHOS_VULKAN_FORCE_VENDOR` | Override classification on one adapter (`amd`, `intel`, `nvidia`, ..., `portable`). |
| `HIERARCHOS_VULKAN_FORCE_NATIVE_FP16_LM_COMPUTE` | Re-enable the packed-FP16 LM adjoint kernels on Intel Gen9 for driver re-qualification (see below). |
| `HIERARCHOS_ATEN_VECTOR_WIDTH` | Pin the host ATen vector shape (`8` AVX2 / `16` AVX-512) that the narrow-GEMM and log-softmax modules mirror; default is the CPU capability probe. |

All are read at kernel-construction time, so the same binary can be A/B'd
without a rebuild.

### Host ATen vector shape is not a vendor choice

The narrow-output GEMM (`output_dim == 8`) and the materialized log-softmax fold
reproduce PyTorch's CPU kernels, and ATen selects those kernels by CPU ISA. The
Gen9 target is an AVX2-only machine, so its tuned modules are the 8-float shape
(`linear_forward_lane2`, `falcon_h1_cross_entropy.spv` with an 8-lane fold),
while an AVX-512 host runs the 16-float shape (`linear_forward_lane4`, and
`falcon_h1_cross_entropy_lanes16.spv`, built from the same source with
`-DHIERARCHOS_LOG_SOFTMAX_LANES=16`). Selection follows the CPU capability probe,
not the GPU vendor, so both hosts stay bit-exact to their own oracle; on an
AVX2-only host the probe resolves to 8 and the Gen9-tuned modules are dispatched
unchanged. See `COMPATIBILITY.md`, "Host ATen vector shape (AVX2 vs AVX-512)".

### Intel Gen9 FP16 LM reliability policy

`native_fp16_lm_compute_reliable` keeps the opt-in
`Fp16StorageFp16LmBackward` LM tranche on the verified FP32-compute/FP16-storage
arm for Intel Gen9/Gen9.5 PCI IDs. Measured on the HD Graphics 520 target
(`0x8086:0x1916`, driver 31.0.101.2115): the packed-FP16 streaming LM adjoint
kernels fault the GPU (`VK_ERROR_DEVICE_LOST`) at the production width 448,
which the regression test
`native_fp16_lm_input_grad_reuse_arms_match_packed_at_width_448` reproduces on
both the pre-tuning and tuned binaries. The same byte-identical modules pass at
width 5, the FP32-compute adjoint passes at width 448, and every arm is green on
AMD, so the fault is a Gen9 driver/compiler issue rather than a kernel defect.
The policy is pure and asserted by `native_fp16_lm_reliability_policy_is_scoped_to_intel_gen9`;
newer Intel generations (Ice Lake, Xe, Arc) and every other vendor keep the
feature. Set `HIERARCHOS_VULKAN_FORCE_NATIVE_FP16_LM_COMPUTE=1` to re-qualify the
path on a newer driver.

#### Gen9 rows16 driver quirk (resolved)

The packed-FP16 rows16 cross-row input arms (`Fp16CeTapeRows16*`) were briefly
withheld from the auto-tuner on Gen9/Gen9.5 after a `VK_ERROR_DEVICE_LOST` in
the rows16 arm of
`native_fp16_lm_input_grad_reuse_arms_match_packed_at_width_448`. Bisecting that
batch with the `lm_rows16_batch_bisect_microprofile` harness showed every
module passes in isolation and in every production-shaped composition, and the
fault reproduced only when the test created the fused/reduce compute pipelines
between recorded dispatches of an open command batch. Hoisting kernel creation
ahead of command recording makes the same arm green deterministically, the
width-448 topology microprofile measures every rows16 variant green on the same
driver (selecting `fp16-ce-tape-rows16-cluster4-fused-adjoints`, about 3.9x
faster than the rows8 plan), and production never creates a kernel while
recording (no `record_*`/`train_*` path calls `ComputeKernel::new_with_access`),
so the quirk was a test-harness artifact rather than a kernel or driver fault.
The arms are therefore admitted by capability on every vendor, exactly like the
other rows16 variants, and the former `fp16_rows16_lm_arms_reliable` policy
together with its `HIERARCHOS_VULKAN_FORCE_FP16_ROWS16_LM` re-qualification
switch was removed along with the fix. The plan-level guarantee that a cache
entry can never dispatch an arm the geometry does not offer is still enforced
by the shared `plan_admitted` filter, asserted by
`rows16_cross_row_arms_are_offered_and_cache_admission_requires_the_arm`. The
opt-in LM microprofiles now exercise the rows16 arms unconditionally:
`lm_width448_backward_topology_microprofile`,
`lm_rows16_forward_stats_seam_microprofile`, and
`lm_fused_adjoint_dx_tile_microprofile`.

## Vendor matrix

| Family | Portable (every other vendor) | Intel Gen9 | AMD RDNA3 (gfx11) |
| --- | --- | --- | --- |
| `linear-forward` | `shaders/linear_forward.spv` | `shaders/linear_forward_intel_gen9.spv` | `shaders/linear_forward_amd_rdna3.spv` |
| `linear-bias-forward` | `shaders/linear_bias_forward.spv` | `shaders/linear_bias_forward_intel_gen9.spv` | `shaders/linear_bias_forward_amd_rdna3.spv` |
| `linear-residual-forward` | `shaders/linear_residual_forward.spv` | `shaders/linear_residual_forward_intel_gen9.spv` | `shaders/linear_residual_forward_amd_rdna3.spv` |
| `linear3-forward` | `shaders/linear3_forward.spv` | `shaders/linear3_forward_intel_gen9.spv` | `shaders/linear3_forward_amd_rdna3.spv` |
| `silu-forward`, `silu-backward`, `adamw` | own portable module | none (candidates measured slower; see below) | none |

AMD is the one vendor whose tuned modules are selected by *silicon generation*, not
by vendor alone: only an RDNA3 part (PCI device ID in the table documented below)
leaves the portable module. Every earlier AMD generation, and every non-AMD vendor,
resolves to the exact pre-tuning `.spv` bytes, which is what keeps the AMD target the
portable modules were green on unchanged.

The Intel matmul variants keep the portable 16x16 workgroup, output mapping, and
ascending-k `fma()` chain, and stage their `x`/`w` k-slices in workgroup memory.
Global loads per FMA drop ~8x (6 -> 6/16 for the three-way `linear3-forward`),
the per-family epilogues are unchanged and in the portable operand order
(`sum + bias[col]`, `residual[index] + sum`), and the arithmetic sequence is
untouched, hence bit-identical results.

The three two-tile families stage a **32-float** K slice, matching the portable
module's interleave: a 16-row input tile with a 33-float row stride and a
32-wide weight tile with a 17-float row stride, so the 16 lanes of the workgroup
saturate a 32-wide slice in one pass and each 32-wide K step pays two barriers
instead of four. Workgroup bytes are `16 * 33 * 4 + 32 * 17 * 4 = 4288` for
those three families and `6 * 16 * 17 * 4 = 6528` for the unported
`linear3-forward`; both are inside the Gen9 32 KiB limit and are decoded back out
of the committed module by the capability test rather than hand-asserted.

`linear-residual-forward` and `linear3-forward` are the RWKV-native
channel-mix / post-mix and time-mix projection slots, so the tuning covers the
hottest matmuls of both served graphs (the Transformer training path and the
native Hierarchos path) rather than only the Hugging Face forward projections.

### AMD RDNA3 (gfx11) variant (bit-exact here, timing unmeasured on RDNA3)

RDNA3 parts carry 64 KiB of LDS -- twice the Gen9 budget -- and execute in wave64, so
their tuned modules stage a **64-wide** K slice where the portable and Gen9 modules
stage 32: each of the 16 lanes loads four k values, a 64-wide slice is still filled in
one coalesced pass, and the workgroup pays two barriers per 64 k instead of two per 32.
The padding that keeps the reads conflict-free is the portable one (65-float input rows,
17-float weight rows). The three single-projection families declare
`16 * 65 * 4 + 64 * 17 * 4 = 8512` bytes; `linear3-forward` stages all six of its
operands, `6 * 16 * 65 * 4 = 24960` bytes. Both fit the RDNA3 LDS, and
`committed_rdna3_modules_declare_their_capability_contract` decodes those totals, the
`LocalSize` and the barrier count back out of the committed modules rather than trusting
the constants next to them.

Selection is `vendor == AMD && is_amd_rdna3(device)`: RDNA3 is classified by an AMD PCI
device-ID table (Navi 31 `gfx1100`, Navi 32 `gfx1101`, Navi 33 `gfx1102`, and the
integrated Phoenix / Hawk Point `gfx1103` parts), because Vulkan exposes no
architecture-generation property; a name or a driver string is not used. Anything
outside that table -- including RDNA1 and RDNA2, whose IDs are pinned as negative
controls -- keeps the portable module.

The Gen9 modules are not candidates on an RDNA3 device at all: `select_kernel` enters
the Intel branch only when `effective_vendor(device).is_intel()`, which is the PCI
vendor ID (`0x8086`), and a non-Intel device reaches the RDNA3 branch or the portable
bytes. The per-dispatch gate is chosen by the vendor the selection resolved to, so an
AMD slot can only ever hold the RDNA3 module and the RDNA3 gate. The Intel modules are
still *executed* on an RDNA3 adapter by
`intel_variants_are_bit_exact_with_portable_kernels`, but nothing there dispatches
them in production.

The per-dispatch gate is `rows >= 8 && input_dim >= 64 && output_dim >= 16`, which is
exactly the threshold at which the *portable* modules pick their own tiled schedule over
the scalar path. That is deliberate: no RDNA3-specific crossover has been measured (there
is no RDNA3 adapter on the machine this layer is tuned on), so the gate claims only the
structural statement -- use the tuned module where the portable module would itself
tile -- and never a speedup.

**What is proven here.** All four modules are executed on this Gen9 adapter and compared
raw-bit with the portable module at the same six geometries the Intel A/B uses, including
the geometries the gate keeps on the portable module: 24 comparisons, every one
`max_ulp_delta=0` (`rdna3_variants_are_bit_exact_with_portable_kernels`, which runs on
whatever adapter is present). The timing that run prints is recorded but is *not* RDNA3
evidence: on Gen9 the 64-wide stage loses where k is long (0.71-0.73x at 8x448x512) and
wins on `linear3-forward` (2.52x at 8x448x512, 1.77x at 6x96x320), because it trades
per-lane loads for barriers. That is also why the Gen9 `linear3` module stays at the
32-wide-shaped 16-wide stage (3.58x on the same shape) and the two families do not share
one module.

**What is not proven.** RDNA3 timing. The module is selected there and is numerically
identical here, but its speedup on gfx11 has not been measured, so it is recorded as a
structural tuning, not a measured win. The first run on an RDNA3 adapter should print the
plan (`vendor_kernel_plan`) and `--bootstrap-coverage` its coverage entries; measuring it
is the next step, and if a shape loses there the same A/B hook
(`HIERARCHOS_VULKAN_VENDOR_BENCH_*`) and the same gate are what admit or reject it.

### Below eight rows: the pre-tuning scalar schedule (measured)

The 2026-10-15 tiling pass gave the portable modules a 32-wide shared-memory k-slice with
a scalar fallback. That fallback is not free: the `shared` declaration is part of the
module, so a single-row dispatch reserves it even while it runs the scalar path, and on
the Gen9 target at `1x448x512` the tiled portable module measured 0.1839 ms against the
pre-tuning module's 0.1502 ms -- 0.82x, an 18% regression on exactly the decode shapes
that matter outside training.

`VendorMatmulKernel` now holds the pre-tuning scalar artifact for each of the four matmul
families (`shaders/portable_scalar/*.spv`, byte-for-byte the initial-publish modules) and
dispatches it below `SCALAR_MATMUL_MAX_ROWS = 8`, which is the portable modules' own
scalar-vs-tiled threshold. Nothing is re-derived: those are the modules the pre-optimization
goldens were recorded against, and `scalar_schedule_reproduces_the_portable_kernels`
executes them against the portable modules on the executing adapter (raw-bit, every
geometry) as a guard against a stale copy. Selection order is: an admitted vendor variant
first (its gate is the measured crossover for that part), then the scalar schedule below
eight rows, then the tiled portable module. The shape recovers to **0.1502 ms, 1.22x
against the tiled portable and parity with the pre-tuning module**, measured on the same
adapter in the same session.

The artifacts live in a nested directory because they are byte-identical to the
pre-tuning portable modules and `build.rs` refuses two shaders with one SPIR-V signature
(the registry it generates maps signature to name); the nested directory is outside that
registry on purpose.

The frozen copies carry their sources now: `shaders/portable_scalar/*.comp` are the
pre-tuning GLSL, listed beside the artifacts in `training_shader_manifest.json` and
rebuilt byte-for-byte by the provenance check with the pinned `glslc`. Identity with
the publish commit is additionally asserted without a compiler by
`scalar_fallbacks_are_the_published_modules`, which holds each scalar artifact to the
SHA-256 it had at `a100338` -- so a retile or an artifact swap cannot silently move the
decode path off the bytes the pre-optimization goldens were recorded against.

## Measured results

Hardware: Intel(R) HD Graphics 520 (0x8086:0x1916), driver 31.0.101.2115,
Vulkan 1.3.215, Windows. Method: `cargo test --release --lib vendor:: --
--nocapture --test-threads=1`; each sample is one isolated dispatch timed with
queue timestamps, median of 20-40 dispatches after 3 warmups. `speedup` is
portable / Intel.

Every A/B row is compared as raw `f32` bit patterns against the *unmodified
portable module that was green on the AMD target* — the strictest form of the
`2e-7` logit-drift ceiling the reference model tests use, and the test prints
both `max_abs_delta` (asserted `<= 2e-7`) and `max_ulp_delta` (0) for each row.
That is why vendor tuning cannot move a model that was green at `2e-7`.

The gate's numbers are driven by this matrix; `intel_matmul_geometry_supports`
is asserted against every row in `intel_matmul_geometry_gate_matches_the_measured_matrix`,
and its portable-schedule half in
`intel_matmul_family_gate_matches_the_portable_tiled_measurements`.

| Geometry (rows x k x n) | Portable | Intel Gen9 | Speedup | Gate |
| --- | --- | --- | --- | --- |
| 16 x 448 x 2048 | 5.74 ms | 0.89 ms | **6.43x** | Intel |
| 32 x 448 x 2048 | 11.36 ms | 1.68 ms | **6.76x** | Intel |
| 8 x 448 x 512 | 0.770 ms | 0.246 ms | **3.16x** | Intel |
| 8 x 128 x 512 | 0.219 ms | 0.077 ms | **2.83x** | Intel |
| 8 x 96 x 512 | 0.178 ms | 0.062 ms | **2.85x** | Intel |
| 8 x 64 x 512 | 0.114 ms | 0.044 ms | **2.61x** | Intel |
| 8 x 32 x 512 | 0.061 ms | 0.028 ms | **2.23x** | Intel |
| 8 x 64 x 256 | 0.062 ms | 0.028 ms | **2.26x** | Intel |
| 8 x 64 x 192 | 0.049 ms | 0.026 ms | **1.87x** | Intel |
| 8 x 64 x 128 | 0.034 ms | 0.025 ms | **1.39x** | Intel |
| 12 x 32 x 256 | 0.049 ms | 0.020 ms | **2.48x** | Intel |
| 6 x 96 x 320 (bias) | 0.091 ms | 0.041 ms | **2.18x** | Intel |
| 4 x 32 x 512 | 0.039 ms | 0.027 ms | **1.44x** | Intel |
| 4 x 64 x 256 | 0.035 ms | 0.028 ms | **1.25x** | Intel |
| 4 x 64 x 192 | 0.029 ms | 0.025 ms | **1.17x** | Intel |
| 4 x 32 x 256 | 0.022 ms | 0.019 ms | **1.14x** | Intel |
| 4 x 96 x 128 | 0.033 ms | 0.031 ms | 1.05x | portable (thin n at four rows) |
| 4 x 64 x 128 | 0.0234 ms | 0.0237 ms | 0.99x | portable |
| 4 x 32 x 128 | 0.0160 ms | 0.0183 ms | 0.87x | portable |
| 4 x 64 x 96 | 0.0219 ms | 0.0239 ms | 0.92x | portable |
| 5 x 64 x 96 | 0.0229 ms | 0.0252 ms | 0.91x | portable |
| 2 x 448 x 512 | 0.274 ms | 0.245 ms | 1.12x | portable (row bar) |
| 2 x 128 x 512 | 0.0832 ms | 0.0768 ms | 1.08x | portable (row bar) |
| 2 x 64 x 96 | 0.0212 ms | 0.0233 ms | 0.91x | portable |
| 1 x 448 x 512 | 0.148 ms | 0.243 ms | **0.61x** | portable (row bar) |
| 8 x 16 x 512 | 0.0345 ms | 0.0605 ms | 0.57x | portable (k bar) |

Every row compared bit-identical outputs. Shapes below a bar keep the portable
module, so no dispatch on this hardware is slower than the pre-tuning backend
while production geometry runs 1.4x-6.8x faster. The `Portable` column is the
**pre-tuning** module (the initial publish), not today's tiled portable module.

#### Today's selection against the pre-tuning artifacts (measured 2026-10-16)

Same adapter, same harness (`HIERARCHOS_VULKAN_VENDOR_BENCH_*`), 20 timed dispatches
after 3 warmups, median, every row asserted bit-identical. `A` is the module from the
initial publish of this crate (`git show a100338:.../<family>.spv`); `B` is the module
this machine selects today. Timing comes from queue timestamps, so the ratios do not
depend on the host build profile; these rows were measured in a debug binary.

| Family | Geometry | Pre-tuning | Today | Speedup |
| --- | --- | --- | --- | --- |
| `linear-forward` | 16x448x2048 | 5.7041 ms | 0.9197 ms | **6.20x** |
| `linear-forward` | 8x448x512 | 0.7595 ms | 0.2277 ms | **3.34x** |
| `linear-forward` | 6x96x320 | 0.0917 ms | 0.0436 ms | **2.10x** |
| `linear-bias-forward` | 16x448x2048 | 5.6945 ms | 0.9197 ms | **6.19x** |
| `linear-bias-forward` | 8x448x512 | 0.7680 ms | 0.2254 ms | **3.41x** |
| `linear-bias-forward` | 6x96x320 | 0.0912 ms | 0.0442 ms | **2.06x** |
| `linear-residual-forward` | 16x448x2048 | 5.7380 ms | 0.9217 ms | **6.23x** |
| `linear-residual-forward` | 8x448x512 | 0.7663 ms | 0.2256 ms | **3.40x** |
| `linear-residual-forward` | 6x96x320 | 0.0912 ms | 0.0435 ms | **2.10x** |
| `linear3-forward` | 16x448x2048 | 17.0464 ms | 2.3921 ms | **7.13x** |
| `linear3-forward` | 8x448x512 | 2.2253 ms | 0.6355 ms | **3.50x** |
| `linear3-forward` | 6x96x320 | 0.2411 ms | 0.1027 ms | **2.35x** |

The portable path -- what every non-tuned vendor runs -- also improved against the
pre-tuning artifacts without a vendor module: 6.24x at 16x448x2048 and 3.34x at
8x448x512 for `linear-forward`, and 1.00x for `linear3-forward`, whose portable module
was never tiled. Single-row decode is the one shape the tiled portable module lost
(0.82x at 1x448x512); it now dispatches the scalar schedule and measures 1.00x against
the pre-tuning module, which is the section above.

### Extended families (same hardware, same method)

The A/B harness now also drives every other admitted family, and the same
`--nocapture` run prints one line per case
(`cargo test --release --lib intel_variants_are_bit_exact_with_portable_kernels --
--nocapture --test-threads=1`). All rows below compared raw `f32` bits:
`bit_exact=true`, `max_abs_delta=0.0`, `max_ulp_delta=0`.

| Family | Geometry | Portable | Intel Gen9 | Speedup | Gate |
| --- | --- | --- | --- | --- | --- |
| `linear-residual-forward` | 8 x 448 x 512 | 0.791 ms | 0.244 ms | **3.24x** | Intel |
| `linear-residual-forward` | 6 x 96 x 320 | 0.094 ms | 0.041 ms | **2.30x** | Intel |
| `linear-residual-forward` | 9 x 33 x 193 | 0.030 ms | 0.018 ms | **1.64x** | Intel |
| `linear-residual-forward` | 2 x 64 x 96 | 0.021 ms | 0.024 ms | 0.91x | portable |
| `linear-residual-forward` | 1 x 1 x 1 | 0.008 ms | 0.011 ms | 0.73x | portable |
| `linear3-forward` | 8 x 448 x 512 | 2.248 ms | 0.638 ms | **3.52x** | Intel |
| `linear3-forward` | 6 x 96 x 320 | 0.256 ms | 0.103 ms | **2.49x** | Intel |
| `linear3-forward` | 9 x 33 x 193 | 0.075 ms | 0.036 ms | **2.08x** | Intel |
| `linear3-forward` | 2 x 64 x 96 | 0.031 ms | 0.039 ms | 0.79x | portable |
| `linear3-forward` | 1 x 1 x 1 | 0.009 ms | 0.015 ms | 0.62x | portable |
| `parameter-matmul-forward` | 8 x 448 x 512 | 0.193 ms | 0.244 ms | 0.79x | portable (rejected) |
| `parameter-matmul-bias-forward` | 8 x 448 x 512 | 0.195 ms | 0.245 ms | 0.80x | portable (rejected) |

The sub-gate rows confirm the three constants transfer unchanged to the new
families: every shape the gate keeps on the portable module measured slower on
the Intel module, so no new geometry class needed its own bar. `linear3-forward`
moves three times the data per FMA, which is why its gains at the top shape
(3.52x) exceed the single-projection kernels (3.16x) that share the same tile. The two-to-four-row rows are a
deliberate trade: they measured 1.08x-1.12x, but they sit next to the
catastrophic single-row case and the loss at `4 x 32 x 128` (0.87x), so the row
bar stays at four.

### What the Intel work has not covered

The four forward matmul families are decided, and `silu-forward`, `silu-backward`
and `adamw` are deliberately left portable. The backward GEMM families
(`linear_input_grad`, `linear_weight_grad`, `parameter_matmul_input_grad`,
`parameter_matmul_weight_grad`) have no variant at all. They reduce over the
output dimension, so the same 16x16 output tile applies and a tiled module can
keep each output element's serial k order -- bit-exactness is achievable by
construction, unlike a reassociated reduction -- but admitting one still needs the
same two pieces of evidence every module here has: bit-exactness against the
portable kernel on device, and a measured crossover at training shapes. That is
the next Intel target; measuring RDNA3 timing is the next AMD one.

### Candidates that were measured and rejected

| Candidate | Result | Decision |
| --- | --- | --- |
| `silu-forward` / `silu-backward` 64 invocations x 4 elements | 0.78x / 0.85x | Rejected: slower on Gen9. |
| `adamw` 64 invocations x 4 elements | **not bit-exact** (1 ulp at element 64) | Rejected: the different loop shape changed the compiler's contraction of the update expression. The bit-exactness contract is non-negotiable. |
| short-k direct path inside the tiled module | 0.70x at k=8 (module-level occupancy cost, not loop cost) | Rejected: a host-side geometry gate is strictly better. |
| `parameter-matmul-forward` / `-bias-forward` (transposed weight) | 0.79x / 0.80x at 8x448x512, 0.98x / 1.00x at 6x96x320, 1.12x at 5x64x192 | Rejected: that portable addressing is already coalesced (0.19 ms vs 0.77 ms for the row-major kernel at the same shape), so the extra staging and barriers buy nothing at production shapes. |
Their GLSL and SPIR-V are kept under `shaders/vendor_experiments/` so a future
Intel generation (Xe/Arc) can re-run the same A/B without rewriting the
kernels:

```powershell
$env:HIERARCHOS_VULKAN_VENDOR_BENCH='silu-forward'
$env:HIERARCHOS_VULKAN_VENDOR_BENCH_A='shaders/silu_forward.spv'
$env:HIERARCHOS_VULKAN_VENDOR_BENCH_B='shaders/vendor_experiments/silu_forward_intel_gen9.spv'
$env:HIERARCHOS_VULKAN_VENDOR_BENCH_SHAPE='len=16384'
cargo test --release --lib candidate_module_experiment_from_env -- --nocapture --test-threads=1
```

## Shader provenance

Every admitted Intel module is committed as GLSL and SPIR-V, assembled with

```bash
glslang -V --target-env vulkan1.0 shaders/<name>_intel_gen9.comp -o shaders/<name>_intel_gen9.spv
```

with an official `glslang` release build; both 16.6.0 and 16.5.0 reproduce these
admitted modules byte-for-byte, and 16.5.0 is the pin that also reproduces the
attention kernels (see *Toolchain pin* below).

**Every embedded module is pinned, not just the vendor ones.** All twelve modules
above are in `validation/training_shader_manifest.json` -- the four `*_intel_gen9`
and four `*_amd_rdna3` artifacts under `glslangValidator` 16.5.0, and the four
pre-tuning scalar fallbacks in `shaders/portable_scalar/` (artifact and source,
`glslc` 2023.8) -- and so is every other `include_bytes!` module the crate
compiles: **467 entries, 467 `identical`**, covering 466 of the 470 embedded
artifacts. `validation/verify_training_shaders.py` rebuilds, byte-compares and
`spirv-val`s the same **467 variants**. The recipes were recovered from each
source's own macro table, so a variant like
`rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_tree_wg32.spv` is
pinned as its base source with `-DFUSE_KEY_PARAM_REDUCE=1
-DRWKV_FAST_TREE_RECURRENT_REDUCTIONS=1
-DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32`, not as an unexplained blob.

Four embedded modules cannot be pinned, and each is listed with its reason in
`validation/shader_provenance_exceptions.json` rather than silently skipped:
`falcon_h1_adamw.spv`, `falcon_h1_cross_entropy.spv` and
`falcon_h1_cross_entropy_lanes16.spv` were committed without GLSL (no revision of
the tree ever carried one, and no committed source reproduces them), and
`parameter_matmul_weight_grad.spv` is kept as the stable oracle's bytes (see
below). `test_shader_provenance.py` fails if an embedded module is neither pinned
nor listed, and if a listed module is no longer embedded, so the exception list
cannot quietly grow or go stale.

`transformer_qwen_gdn_params.spv` used to be the fourth: it had been built before
its source grew a `beta_scale` push member, so the shipped module read an 8-byte
push block and multiplied nothing, while the host pushed 12 bytes and the backward
and shared-gradient kernels already applied the scale. The source and host were
right and the module was stale, so it was rebuilt (see below) rather than
excepted -- a forward kernel silently disagreeing with its own backward kernel is
a training correctness bug, not a provenance detail.

Two artifacts are not simply the pinned build. `transformer_qwen_gdn_params.spv`
was rebuilt from its committed GLSL to restore the `beta_scale` term; the diff
against the old module is the third push member, the load, and
`pc.beta_scale / (1 + exp(-b))` in place of `1 / (1 + exp(-b))`. That is
bit-identical at `beta_scale = 1.0` (so Qwen3.5 is unchanged, IEEE division by one
being exact) and exactly 2x the beta update for OLMo Hybrid with
`linear_allow_neg_eigval`, which is what the host always pushed and what the
backward kernel already assumed.
`parameter_matmul_weight_grad.spv` is **restored from the stable oracle tree**,
not rebuilt. A regenerated copy had been shipped during this sweep and that was a
mistake: the oracle build inlines `accumulate_backward_product()` into `main`,
while the pinned `glslc` outlines it into a `Function`-scope helper that stages its
operands through local memory -- 404 bytes and a different instruction schedule,
not the `OpName`-only difference originally claimed. No release or flag set on
offer reproduces the oracle bytes (default, `-O`, `-Os` and `-O0` all differ), so
the validated shipping module is kept byte-for-byte and recorded as an exception
instead of a pin.
#### The stable tree as the numerical oracle

`C:\Users\User\Hierarchos-Native-stable\Hierarchos-Native` is the pre-optimization
tree the tuned kernels are validated against: it carries the AVX512/RDNA3 work and
predates the Gen9 and lane-2 sweeps. It is **read-only** -- nothing here writes,
builds or checks anything out there. Portable artifacts are compared to it by
bytes, and the rule is that the portable path ships the oracle's module unless a
change is deliberate and its numerics are requalified against it.

The current comparison of every `.spv` and GLSL file in `shaders/` finds the two
trees identical except for: the additive vendor modules (`*_intel_gen9`,
`*_amd_rdna3`, `portable_scalar/`, the vendor experiments), the attention backward
key-range split (a portable kernel, but its single-block path is the oracle's
arithmetic and the recorded digests below did not move), and
`transformer_qwen_gdn_params.spv` (the `beta_scale` ABI fix).
`parameter_matmul_weight_grad.spv` is the oracle's bytes exactly -- a regenerated
copy that had drifted to a 404-byte-larger, outlined build was reverted, because
`spirv-diff` had only compared instruction mnemonics and missed the function
outlining.

The numerical checks that run against the oracle:

| check | result |
| --- | --- |
| `kernel_golden/*.json` AMD digests, stable vs this tree | 406/406 identical (144 dense, 183 reduction, 79 attention) |
| golden harnesses on this Intel device | 406/406 verified, 0 failures |
| `training_step_probe` at the ladder's probe args | mean_loss `5.5432177` and 6368 dispatches, both exactly the recorded Intel expectation; the oracle's AMD recording is `5.543217` for the same fixture |

The AMD digests cannot be re-executed on this machine, so they are a carried
record rather than a fresh measurement; the Intel harness runs are the local
proof, and the harnesses' `--golden-strict` mode cannot consume the stable file
directly because that file has no entry for this device and says so instead of
silently skipping.

`validation/check_oracle_parity.py` automates the byte half of this, so the
drift that shipped once cannot ship again. It opens both trees read-only and
compares every portable source and artifact present in both, normalizing line
endings so a checkout's CRLF is not drift; vendor modules (`*_intel_gen9`,
`*_amd_rdna3`, `vendor_experiments/`, `portable_scalar/`) are reported separately
and excluded, because differing from a baseline that predates them is the vendor
layer's job and its own tests qualify them. A shared file that differs without an
entry in `validation/oracle_divergences.json` fails, and so does an entry that no
longer differs, so the allowlist cannot rot into blanket permission. The current
run compares 810 portable files, reports the three documented divergences below,
and passes; putting the outlined `parameter_matmul_weight_grad.spv` back fails it
with that artifact's name. Exit 2 means the oracle is absent on this machine -- a
skip, never a pass. It runs as the `oracle parity` step of
`verify_training_stack.py`, blocked rather than green when the tree is missing.

`validation/audit_shader_artifacts.py` re-compiles GLSL sources
with the compiler the caller configures (`--compiler glslc` by default); run it
with a matching toolchain when auditing byte provenance, and note that `glslc`
produces equivalent modules with a different SPIR-V generator word.

#### Toolchain pin

The committed artifacts reproduce **byte-for-byte** with the official `glslang`
**16.5.0** release build (`glslangValidator`, Windows x86-64):

| module(s) | result with 16.5.0 | result with 16.6.0 |
| --- | --- | --- |
| `linear_forward_intel_gen9.spv`, `linear_bias_forward_intel_gen9.spv`, `linear_residual_forward_intel_gen9.spv`, `linear3_forward_intel_gen9.spv` | byte-identical | byte-identical |
| `transformer_attention_forward.spv`, `transformer_attention_backward.spv` | byte-identical | **differs** |

The admitted vendor modules are insensitive to the 16.5.0/16.6.0 difference, so
the 16.6.0 instruction in `README.md` stays correct for them. The attention kernels
are not: 16.6.0 emits one extra instruction for both (`OpBitcast` where the
committed bytes carry `OpUConvert`, 16 bytes larger), so they pin **16.5.0**.

The full embedded set spans **both** releases, and the manifest records which one
each artifact needs: 27 artifacts reproduce only with 16.5.0 and five
(`silu_forward`, `transformer_gelu_tanh_forward`, `transformer_gelu_tanh_backward`,
`transformer_kimi_attn_res_softmax`, `transformer_moe_router_topk`) reproduce only
with **16.6.0** -- their committed bytes carry a dead-constant shape that 16.5.0
eliminates. Running the check with only one release reports `wrong_version` for the
other set rather than a byte difference, which is why
`install_shader_toolchain.sh` installs both (`glslang-16.6.0`'s asset names its
binary `glslang.exe`; the script also exposes it as `glslangValidator.exe`) and
`check_shader_provenance.py` discovers every `glslang-*` release under `.tools/`
and selects per artifact. The bulk of the tree (435 artifacts) is the pinned
`glslc` (shaderc 2025.3, which reports `2023.8`).

Re-measured on the module as it now stands (after the attention-backward
rewrite, `transformer_attention_backward.spv` is itself the 16.5.0 build), the
difference between the two releases for that kernel is where the `+Inf` clamp
constant comes from: 16.5.0 emits `OpBitcast %float %uint_2139095040`
(`0x7F800000`, the same `+Inf`) where 16.6.0 emits a `%float_0x1p_128 =
OpConstant %float 0x1p+128` and returns it twice. Floating-point opcode counts are
identical between the two, and for this module 16.5.0 is the *larger* one by 16
bytes because of that constant's path; `0x1p+128` and `0x7F800000` are both
`+Inf` in binary32, so the two builds compute the same kernel. The generator word is
`Khronos Glslang Reference Front End` at version `11` for the glslang-built
modules, versus `Google Shaderc over Glslang` at `11` for the `glslc`-built bulk
of the tree, which is how a module's compiler family can be identified before
re-deriving its flags.

The crate does not trust the binary: `vendor.rs` decodes the committed modules in
its test suite and asserts that the declared `LocalSize`, workgroup-memory
variables, and barriers match the capability gates and the host dispatch
contract, so a stale or mistargeted `.spv` fails the suite rather than a user's
driver validation.

The same idea covers the push block, which byte provenance does not: a module that
rebuilds byte-for-byte from its source is only as truthful as that source is about
the host. `validation/check_push_constant_layouts.py` therefore compares three
descriptions of every kernel's push block -- the committed SPIR-V, the GLSL that
builds it (with `#include`s expanded and `#if` branches resolved for the artifact's
pinned `-D` flags), and the host struct whose `bytes_of` is pushed -- and reports
any disagreement. A struct wider than the block is allowed, because it is
documented (the shared head trainer declares its full loss block while both vendor
and portable modules read only the leading geometry words); a host narrower than
the shader is an error, and so is a different leading word layout. This is the
check that fails on the pre-fix `transformer_qwen_gdn_params.spv`: `SPIR-V declares
[rows@0, num_value_heads@4] but transformer_qwen_gdn_params.comp declares
[rows@0, num_value_heads@4, beta_scale@8]`. It is a ladder step in
`verify_training_stack.py` (`push-constant layouts`) and needs no toolchain, since
it reads the committed binaries rather than rebuilding them.

#### Installing the pin locally (no administrator rights)

`validation/install_shader_toolchain.sh` installs the three binaries above into
the repository's gitignored `.tools/` directory and writes `.tools/env.sh` to put
them on `PATH`:

```bash
cd hierarchos-vulkan
bash validation/install_shader_toolchain.sh
source ../.tools/env.sh
python validation/check_shader_provenance.py
python validation/verify_training_shaders.py
python validation/check_push_constant_layouts.py
```

Neither command needs a flag to pin a release: each artifact records the release
that reproduces it and the checker selects that toolchain itself. `--record-versions`
writes the release that reproduced an artifact back into the manifest, so a pin is
recorded by a verified rebuild rather than by hand.

It takes `glslangValidator` **16.5.0** and **16.6.0** from the official Khronos
Windows release assets, checking each published SHA-256 before extracting, because
those two releases together reproduce every committed artifact byte-for-byte.
`glslc` and `spirv-val` come from conda-forge packages unpacked by a micromamba the
script downloads: neither SPIRV-Tools nor shaderc publishes Windows binaries of its
own.
The pinned `glslc` is **shaderc 2025.3**, not the newest conda build. That is a
measurement, not a preference: with shaderc 2026.4 two artifacts
(`transformer_rms_norm_input_grad.spv`, `transformer_rms_norm_input_grad_residual.spv`)
rebuild 136 bytes smaller with six fewer integer index adds and one fewer
`OpBitcast`, i.e. a later optimizer, so the tree is spanned by two shaderc
releases and only 2025.3 explains all of it. Its `--version` banner reports
`shaderc v2023.8 unknown hash, 2025-08-31`: the release that build reports about
itself is upstream **2023.8**, so that is what the `glslc` artifacts record even
though the package installed is `shaderc=2025.3`. The check compares the reported
release, and the newest package reports `v2026.4`, so the two are distinguishable
from the banner alone.

`PATH` order matters and `.tools/env.sh` gets it right: the pinned glslang
directory comes first, so `glslangValidator` is 16.5.0 rather than the glslang the
shaderc package depends on (15.4.0 for the pinned 2025.3 package; the 2026.4
package pulls 16.6.0, which does not reproduce the attention kernels — see the pin
table above). With the pin in place, `check_shader_provenance.py` reports
`{"identical": 86}` over the whole manifest and the training-shader release check
validates every rebuild with `spirv-val` and asserts each recorded storage policy.

The release is recorded **per artifact**, which is how the split above stops being
something the tree merely survives. Both checkers offer every candidate binary for
a compiler name (`--compiler-path NAME=PATH`, repeatable, else `PATH`) and compile
each artifact with the candidate that reports the release that artifact records,
so a tree whose artifacts came from two releases is checked in one run instead of
reporting the untranslated half as drift. An installed toolchain that is not the
recorded release fails as `wrong_version` — naming the candidates it saw — rather
than as a byte difference, and `--record-versions` writes the release that
reproduced each artifact back into the manifest so the pins come from a verified
rebuild. Verified by recording `adamw.spv` at 2026.4 in a copy of the manifest and
offering both `glslc` installs **with 2026.4 first**: 86/86 identical, that one
artifact compiled by 2026.4 and the other 82 by 2023.8. With only 2026.4 available
the same manifest reports 82 `wrong_version` and exits 1.

### Intel Gen9 FP32 LN1/low-rank producer topology (measured)

The FP32 LN1/low-rank producer shipped as `rows` 64-lane workgroups with one row
per workgroup. `layer_norm_low_rank_producer_forward_fused_two_rows.comp` gives one
workgroup a pair of rows and shares every weight read between them: the first-stage
weight of a (column, rank) pair and the w2/a2/g2 element of an output column each
serve two FMAs instead of one, so a row pays half the weight bytes it used to. The
price is half the workgroups, which is a real cost on a 24-EU part, so the FP32
path times both variants on the device and keeps the one-row kernel unless the pair
beats it by more than the 2% material margin in every one of seven interleaved rounds --
the packed-FP16 section below states that rule and measures why seven.

On the HD Graphics 520 the pair loses, and the autotuner declines it. The opposite
trade wins: `layer_norm_low_rank_producer_forward_fused_column_slices.comp` gives one
workgroup a 64-column slice of one row instead of a whole row, so the grid is
`[ceil(width/64), rows, 1]` and each lane produces exactly one output column rather
than two at `width = 128`. Every stage that depends only on the row is recomputed
identically by each slice of that row, and the shared backward tape is written by
slice 0 alone, so the split stays bit-identical. The producer's fused form is only
enabled for ranks `<= 64`, which in this model family is `width = 128, ranks
32/32/64`, so that is the whole measurable domain:

| geometry (`--hidden 128`, ranks 32/32/64) | one row | two rows | column slices | selected |
| --- | --- | --- | --- | --- |
| batch 2 (1 pair) | 0.2328 ms | 0.2797 ms | 0.1956 ms | column slices, 16% faster |
| batch 8 (4 pairs) | 0.2308 ms | 0.2773 ms | 0.1972 ms | column slices, 15% faster |
| batch 32 (16 pairs) | 0.2376 ms | 0.2799 ms | 0.2037 ms | column slices, 14% faster |

It is not faster everywhere, and that is why the choice is per shape rather than per
variant. At `width 64` (ranks 8/8/8) the row is one 64-column slice, so the split has
nothing to divide and pays for the extra workgroups: `column slices` 0.08131 ms
against `one row` 0.07704 ms -- a 1.9% median lead that won one round in seven, which is
why the replay reports it as a kept default and not as a variant. Both
shapes are in the checked-in record for this device (`width64-batch2` selecting
`one-row`, `width128-batch2`, `width128-batch3-tail` and `width128-batch32` selecting
`column-slices`), and `validation/verify_topology_decisions.py` replays all four so
either direction moving is a failure.

Each row is the median of seven interleaved rounds of a single submission of the
same kernel repeated in place, which is the set the choice is adjudicated on
(`HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG=1` prints the medians and each
candidate's round wins). The pair loses because it halves the parallelism: at `width = 128` the
second stage's weight working set is small enough to be served from cache whatever
the schedule does, so sharing its reads saves nothing measurable, and one batch row is
1-4 workgroups on a 24-EU part. The column split adds workgroups instead and wins at
every batch measured. Both variants stay available because that trade can move the
other way on a wider geometry or a device with more resident workgroups per EU, and
the choice is made per device and shape rather than assumed. What does not vary is the
arithmetic: all three kernels keep each row's reduction order and FMA order, so the
measurement only picks a schedule, and `validation/verify_training_state.py` now runs
its 3,817 exact tensor digests once per arm -- shipped choice, then each variant forced
-- so each one's equivalence is a ladder step instead of a claim from one run.

Their grids do not all have the same shape, and the grid is not a scheduling detail:
the sliced kernel reads its row from `gl_WorkGroupID.y` and its column slice from
`gl_WorkGroupID.x`, so dispatching it with the row-per-workgroup grid leaves every
row but the first unwritten. Both the timing path and the production path build
their grid through `LayerNormLowRankForwardF32Topology::groups(batch, width)` for
that reason, and three unit tests pin the mapping -- the first production version of
the sliced variant did get the row grid and the training-state goldens failed at
step 0 (1,533 of 1,752 tensor digests at batch 2) on exactly the two geometries that
use the fused producer, while the other two topologies passed.

### Intel Gen9 packed-FP16 LN1/low-rank producer variants (equivalent, not a decision)

The producer the FP16-storage policy actually dispatches is the packed one, and it has
six variants: `one-row`, `one-row-cached` (the row's normalized inputs stay in shared
memory), `one-row-rank128` (for a rank above 64), `two-rows` and `two-rows-wide`
(pairing rows the way the FP32 pair does) and `four-rows`. Which of them the device can
offer is gated on width and ranks: on this model family width 32 admits `one-row`,
`one-row-cached`, `two-rows` and `four-rows`, width 128 admits `one-row`,
`one-row-cached` and `two-rows-wide`, and none of the covered shapes reaches the rank
above 64 that `one-row-rank128` needs.

Unlike the FP32 split, this family's choice used to be a coin flip, which is why the
rule that picks it changed. Under the earlier median-only rule, five consecutive runs of
the width-128 geometry measured:

| run | `one-row` | `one-row-cached` | `two-rows-wide` | selected |
| --- | --- | --- | --- | --- |
| 1 | 0.27867 ms | 0.27181 ms | 0.28252 ms | one-row-cached (2.5% ahead) |
| 2 | 0.27874 ms | 0.27470 ms | 0.28336 ms | one-row (1.5% ahead) |
| 3 | 0.27829 ms | 0.27665 ms | 0.28264 ms | one-row (0.6% ahead) |
| 4 | 0.28302 ms | 0.27270 ms | 0.28608 ms | one-row-cached (3.6% ahead) |
| 5 | 0.27892 ms | 0.27206 ms | 0.28235 ms | one-row-cached (2.5% ahead) |

Three runs out of five picked the cached arm and two kept `one-row`, because the cached
arm's lead straddles the 2% material margin: the effect is inside the run-to-run spread,
and the schedule flipped with it.

The rule now is that a candidate may only displace the structural default by beating it
by the material margin in **every** one of seven interleaved rounds. The round count is
what makes the verdict stable: if each round's winner is a fair coin flip, some candidate
still wins every round about a quarter of the time in three rounds and about one run in
sixty-four in seven, so three rounds is not enough and seven is. The same geometry, on
this part: **thirty-five consecutive runs -- thirty, then five -- all kept `one-row`**, and
the producer's record line says why rather than leaving an unmeasured default:

```json
{"family": "ln-low-rank-fp16", "device": "Intel(R) HD Graphics 520", "subgroup_size": 32,
 "width": 128, "w_rank": 32, "a_rank": 32, "g_rank": 64, "batch_pairs": 1,
 "has_unpaired_tail": false,
 "timings_ms": {"ln-low-rank-fp16-one-row": 0.281082,
                "ln-low-rank-fp16-one-row-cached": 0.276186,
                "ln-low-rank-fp16-two-rows-wide": 0.283327},
 "rounds": 7, "round_wins": {"ln-low-rank-fp16-one-row": 0,
                             "ln-low-rank-fp16-one-row-cached": 4,
                             "ln-low-rank-fp16-two-rows-wide": 0},
 "decisive_margin_fraction": 0.0, "selected": "ln-low-rank-fp16-one-row",
 "structural_default": "ln-low-rank-fp16-one-row", "margin_fraction": 0.017418}
```

That is a 1.7% median lead that won four of seven rounds, recorded as the default kept
rather than as a variant the device does not reproduce. The cost is seven rounds of a few
tens of microseconds per candidate, once per geometry per process, against a step that
measures 361-376 ms; and the three width-128 FP32 geometries still select
`column-slices` unanimously at 12-17%, so `topology_decisions.json` records the same
choices it did before the rule changed. `validation/verify_topology_decisions.py` audits
the new evidence as well as the choice: a record that claims a variant which did not win
every round, a kept default that claims a round it won, or a candidate that stops being
timed at a geometry are each failures, while a kept default with a median lead is
reported as exactly that.

What *is* checked in for this family is that all six variants are arithmetic-identical,
and the FP16 trajectory suite is what says so. The end-to-end view agrees that this choice is not
where the remaining time is: at width 128 the whole producer costs ~0.28 ms of a step
that measures 361-376 ms at batch 2 and 923-926 ms at batch 32, so even the fastest
variant is inside the run-to-run spread of the step itself (shipped 361.2/376.3 ms
against `one-row` forced 367.6/381.0 ms at batch 2, and 923.0/926.2 against 930.2/941.4
at batch 32 -- the shipped choice is never slower, and the difference is below what two
repeats resolve).

That suite is `validation/kernel_golden/training_state_fp16.json`, recorded under
`HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-storage-fp32-compute` (the policy is stored
in the golden's own `env` block so the recorder and the verifier read it from one
place). It runs the full trajectory -- every parameter, both Adam moments, the
optimizer clocks, the losses and the recurrent adjoints -- once for the shipped choice
and once for each of the six variants forced through
`HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY`, which is one variable for both families
because `one-row` and `two-rows` name one variant in each. Its cases are `--batch 2` and
`--batch 32` at width 128 plus a `--hidden 32` fixture, so the width- and rank-gated
variants are each exercised somewhere: `one-row` 3/3 cases, `one-row-cached` 3/3,
`two-rows-wide` 2/3, `two-rows` 1/3, `four-rows` 1/3 and `one-row-rank128` nowhere on
this part. Which of those each device exercises is checked in as
`validation/training_state_coverage.json`; a run fails if any of it shrinks, and the
verdict for the cases an arm is absent in is checked in too, so the arm only pays for a
one-update canary there rather than a full replay (see the state-step section above).

### The topology autotuner cannot flip under load (measured)

The unanimous-round rule removed one way for the schedule to move with nothing changed --
a lead inside the run-to-run spread -- but not the way that actually bit: a sweep taken
while something else was using the GPU. The ladder recorded it once, on the
`width128-batch2` decision, whose release pass reported `one-row` while an abandoned
background run still held the adapter and then reproduced `column-slices` on the clean
pass. The mechanism is not subtle: with a second process on the adapter every round is
slower by a factor the interleaving cannot cancel, so a variant whose true lead is 12-16%
can fail the 2% margin in all seven rounds. Nothing about the kernels changed; the machine
did.

Two changes, both in `src/rwkv_low_rank.rs`:

* Each round now times each candidate **three times and keeps the fastest**. Contention is
  additive -- it can make a sample slower, never faster -- so the minimum is the closest
  estimate of the undisturbed cost, and one interrupted repeat no longer decides a round.
* The round set is checked before it is adjudicated on. The structural default runs in every
  round by construction, so its own per-round samples are the ground truth for whether the
  sweep measured one machine: if the slowest is more than **1.15x** the fastest, no candidate
  may displace the default and the record says `"contended": true` with the spread it
  measured instead of recording what the load did. A quiet sweep records `"contended":
  false` and the spread it saw.

`validation/verify_topology_decisions.py` treats a contended replay as evidence about the
machine rather than about the kernels: it is reported as a note, it never moves the record,
and `--record` refuses to write one, so a run on a busy adapter can no longer turn another
process's timing into this device's decision. The record's own stability audit is extended
the same way -- an entry marked contended that selected a variant, or one whose spread
contradicts its own flag, is a failure -- and `validation/test_topology_decisions.py` pins
both directions of that check without a GPU.

The four FP32 decisions were then re-recorded on the quiet laptop, with the new fields:

| geometry (`--hidden 128`, ranks 32/32/64 unless noted) | selected | round wins | decisive lead | default spread | contended |
| --- | --- | --- | --- | --- | --- |
| width 64, ranks 8/8/8, batch 2 | `one-row` (default kept) | every candidate 0/7 | -- | 1.027x | no |
| batch 2 (1 pair) | `column-slices` | 7/7 | 12.2% | 1.010x | no |
| batch 3 (2 pairs + tail) | `column-slices` | 7/7 | 3.4% | 1.025x | no |
| batch 32 (16 pairs) | `column-slices` | 7/7 | 9.2% | 1.050x | no |

The choices are the same as before the rule changed -- the `width 64` geometry is the
1.9%-median lead that wins no round, the three `width 128` shapes are unanimous -- so
nothing in `topology_decisions.json` moved for a measurement reason. What moved is that
each entry now carries its evidence, including how quiet its sweep was. The decisive lead
recorded at the two-pair geometry (3.4%) is narrower than that row's 14-16% median would
suggest, which is the honest direction: the rule records the narrowest round, and the
per-round minimum is a tighter estimate than a single sample, so a real lead is recorded
as smaller than it looked rather than as larger.

### Kernel-golden invocation coverage (checked in)

The dense, reduction and attention harnesses already caught a changed kernel: every
invocation's output digest is recorded per device. What they could not catch was a
harness or a device *stopping* to exercise an invocation -- deleting a case, or a part
losing a kernel it used to run -- because that only lowers the invocation count in the
pass line while every remaining digest still matches. That is the same gap the
trajectory check closed, so the same rule now covers these goldens, in
`validation/kernel_golden_coverage.json`. One entry per (golden, device) lists the
invocations that device exercises: on the Intel Gen9 part, `dense` 144, `reduction` 183
and `attention` 79, which is exactly what the goldens hold digests for there. A harness
run fails when an invocation its device is recorded as exercising is no longer
exercised, when the golden holds a digest for that device the run did not reproduce, or
when the device has no entry; growth is a note; `--record-coverage` merges this
machine's coverage in (leaving every other device's entry alone) and refuses to shrink
one unless `--allow-coverage-shrink` says the loss is intended. A `--only` run exercises
one kernel on purpose, so it reports `coverage was not audited` instead of a shrink.

The audit and its merge are pure -- no device, no Vulkan -- and live in
`src/coverage_record.rs`, shared with the trajectory record (see *One implementation,
one test suite, one bootstrap* below) so `cargo test --lib` checks them without a GPU;
the record is written only by a harness run. All four paths were proven on the harness
itself: a record claiming `linear_forward:ghost_case`, a record with the device's entry
removed, and a golden holding a digest for `linear_forward:ghost_case` on this device
each exit 1 with their own message, and `--record-coverage` against the first refuses
with `refusing to record a shrunken entry` before writing anything, which
`--allow-coverage-shrink` then accepts.

#### One implementation, one test suite, one bootstrap (2026-10-16)

The two records answered the same three questions -- audit, merge, refusal -- in two
languages, with two sets of rules and two sets of tests. They now share
`src/coverage_record.rs`: the envelope (`schema`, `note`, `coverage`), the keyed audit,
the merge that replaces only the entries this run measured, the shrink refusal and its
wording (`refusing to record a shrunken entry: ... pass --allow-coverage-shrink if
losing this coverage is intended`), and the bootstrap. What each record still owns is
one method, `diff`, which says what a loss means for one of its entries: a kernel-golden
entry compares invocation labels (plus the digests the golden holds for this device), a
trajectory entry compares golden cases, per-case comparison counts and each arm's
verdicts. Both models and the shared machinery are covered by the single test module at
the bottom of that file -- 26 tests: the kernel rules, the trajectory rules, the merge,
the refusal, the envelope, and the property that both records refuse a loss in the same
words. The Python driver kept only what is its own (which arm ran where, which cases a
canary stands in for) and asks the engine through the `coverage_record` tool, built
beside `training_step_probe`; `validation/test_training_state.py` no longer has an audit
or a merge to test.

`--bootstrap-coverage` is the deliberate way onto a device the record has never seen.
The default stays fail-closed: a device with no entry fails, because an entry is what
makes the record evidence. Bootstrap creates *only* the entries this run is missing, so
it has nothing to shrink and nothing to refuse, never rewrites an entry that exists, and
prints the diff it wrote -- one line per added entry, then the new total, then a reminder
to review and commit. It is deliberately narrow: `--record-coverage` and
`--bootstrap-coverage` are mutually exclusive, and a bootstrap cannot be combined with
`--allow-coverage-shrink` because it never replaces an entry.

Proven both ways: the trajectory tool bootstrapped a scratch record from one entry
(`bootstrap added 1 coverage entry (0 already recorded)`, the entry on disk), audited it
clean, reported a shrunken case and a lost arm as two problems, refused
`mode: record` with the shared wording, wrote it with `allow_shrink`, and a second
bootstrap over the same record reported `every entry of this run already recorded; ...
nothing was written`. The kernel harness has the same flag in `GOLDEN RECORDED`-style
use on `--record-coverage` and in the audit path otherwise.

The same step got faster without losing coverage. The trajectory verifier snapshots
after every optimizer step, and each of those snapshots used to read every Adam moment
with its own blocking copy -- two staging allocations, submissions and fences per
parameter. Both moments are now copied back in one submission
(`RwkvPersistentAdamW::state_snapshot`) with every value read from the same buffer it
was read from before, which is why the digests are unchanged rather than merely close.
The state step went from **402.7 s to 346.8 s** (6m42.7s to 5m46.8s, -13.9%) on
this part with the same 33 trajectory runs, the same 43,884 tensor digests and the same
coverage audit, and every digest still matched.

#### The arms then stopped replaying cases they cannot show anything in (measured)

Every arm used to replay every case of both goldens, including the cases where this
device does not have the variant at all: `one-row-rank128` was forced in all three FP16
cases and reported unavailable in each, `two-rows` and `four-rows` were forced in the
two width-128 cases where only the wide pairing exists, and all three FP32 arms were
forced in the width-256 `odd_multi_sequence` case, where the FP32 producer is not in the
shape at all and the override changed nothing. A run in one of those cases can only
re-confirm a verdict the record already holds, and a probe run of one case is 10-26 s of
wall time, most of it model load and pipeline setup the arm has no influence on.

So the coverage record now also holds the two ways an arm can be absent from a case --
`unavailable` (the producer rejected the request) and `not_dispatched` (the fused
producer never saw it at that shape, which is the FP32 producer above rank 64) -- and a
forced arm runs every case the record has it verified in, every case the record says
nothing about, and one **canary** per arm: the widest absent case, replayed for a single
update, because existence is settled by the first dispatch and there is no trajectory to
compare in a case the variant never ran in. The other absent cases are written as
`asserted` and carried from the record, with the canary as their re-check; a canary that
disagrees makes the run measure that arm's carried cases in full instead, so a device or
a gate that changes restores the coverage it invalidated in the same run.

Each case's model package is shared by every arm now as well. The probe initializes a
package when it is missing and only reads it afterwards, verified directly: the package's
content hash is identical before and after two consecutive six-update runs and the two
snapshots are byte-identical files, and the same case takes 10.85 s from a fresh
directory against 9.64 s from one that already existed.

Measured on this part, both suites, the two runs back to back in one session so the
machine's regime is shared: **33 runs / 346.97 s against 29 runs / 281.02 s (-19.0%)**,
same trajectories compared, same coverage audited, every digest matching. The
pre-change side is measured the same way the change is -- by pointing the verifier at a
record whose absence verdicts are all blank, which makes every arm replay every case
exactly as it used to -- and it reports the old `33 run(s), 43884 exact tensor digests`.
The row-level digest count falls to 32,162 because a canary compares one update: the
fallback trajectory it stands in for is still compared in full by the measured arm and
by every arm that dispatches in that case, and the record's per-case counts still come
from the full measured run.

### Intel Gen9 module-level FMA de-fusion (measured)

On the HD Graphics 520 target a compute module that contains *any* NoContraction
arithmetic -- a GLSL `precise` variable whose value can be formed by contracting
a multiply into an add -- has every `GLSL.std.450 Fma` ext-instruction in that
module lowered to a separate multiply and add. The effect is per module, not per
expression. Round 7's `transformer_moe_expert_forward` probe measured a module
whose only seven `NoContraction` decorations (`OpDecorate` 42, verified by
decoding the committed SPIR-V) belonged to one branch, while two other branches
decorated nowhere still accumulated as unfused muladd; removing the `precise`
qualifiers from that same module (zero decorations, same branches) restored the
fused behaviour, so the split is triggered by the module's decoration set rather
than by the branch that carries it. Modules without contractible products are
unaffected: `linear_forward.spv` and `linear_forward_lane2.spv` carry zero
`NoContraction` decorations and are verified fused on-device (lane2's `precise`
accumulators emit none because nothing in them is contractible). The practical
rule is that a module may either materialize products or fuse FMAs, never both,
which is why the products-materialized expert topology lives in
`transformer_moe_expert_forward_muladd.comp` and the fused topologies in
`transformer_moe_expert_forward.comp`. `linear_forward_lane2.comp`'s header
records the first observed instance of the rule.

## Regression tests

`cargo test --release --lib vendor::` runs:

* `vendor_classification_uses_pci_ids_not_names`
* `vendor_labels_round_trip_and_portable_alias_resolves_to_other`
* `vendor_variants_preserve_the_host_dispatch_contract`
* `portable_fallback_bytes_are_the_mainline_artifacts`
* `intel_matmul_geometry_gate_matches_the_measured_matrix`
* `push_geometry_reads_the_leading_rows_input_output_words`
* `vendor_matmul_kernel_routes_dispatch_geometry_to_the_right_module` (device test)
* `native_fp16_lm_reliability_policy_is_scoped_to_intel_gen9`
* `committed_vendor_modules_declare_their_capability_contract`
* `families_without_variants_always_use_the_portable_module` (uses a device when present)
* `non_intel_devices_keep_the_portable_modules` (device test)
* `intel_variants_are_bit_exact_with_portable_kernels` (device test, prints the A/B table for all four admitted families at production, tiled, and sub-gate geometries)
* `candidate_module_experiment_from_env` (opt-in A/B harness for new candidates)

The device tests execute both modules on whatever adapter is present, so the
Intel modules are numerically verified on AMD/NVIDIA hosts too.

The selection layer was also checked end to end on the Gen9 target, because a
module can win its isolation shape and still lose a whole step. Interleaved
`training_step_probe` runs against `HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1`
on the same fixture, with the arms alternating:

| rows | vendor plan on | forced portable | delta |
| --- | --- | --- | --- |
| 2 | 271.8, 274.5 ms | 269.8, 270.2 ms | +0.7 to +1.2% (vendor slower) |
| 8 | 369.0, 356.8, 359.0 ms | 357.0, 359.3, 359.8 ms | within noise, one pair reversed |
| 32 | 829.5, 831.8, 831.3, 853.5 ms | 830.5, 829.2, 839.4, 853.8 ms | within noise, mixed signs |

The small-row residual is the gate doing its job: at these shapes most of the
dispatches are already on the portable modules by geometry, so forcing the
portable path changes little, and the difference that remains is the families
above the gate. The plan is therefore neutral end to end at the geometries where
it is exercised, and no family is admitted on the strength of a step-level
measurement it does not have.

The native Hierarchos architecture is pinned at the model level by
`hierarchos_causal_lm_vendor_kernels_match_portable`
(`cargo test --release --lib hierarchos_causal_lm_vendor_kernels_match_portable`):
it runs one causal-LM accumulation step at vendor-eligible geometry (width 128,
batch 8) with the Intel modules and again with
`HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1`, requires identical losses,
recurrent states, and parameter gradients (`2e-7` ceiling), and uses a
test-only per-thread, **per-family** dispatch counter to prove the vendor run
really executed the Intel modules while the portable run executed none, so the
comparison cannot pass vacuously. Adapters without an Intel variant (the AMD
target included) skip the comparison instead of failing it.

Batch 8 is the fixture's vendor-eligible geometry: the step dispatches its
channel mix and projections as `(rows = batch, k = 128, n = 512)`, and the
fixture asserts its own geometry still reaches the variant, so a gate change that
quietly stopped exercising the Intel modules fails the test instead of passing on
a portable-versus-portable comparison. A later tuning pass moved the eight-row
`(rows = batch, k = 128, n = 512)` shape onto the Intel module by measurement
(`0.93x-0.95x` on the portable schedule), which is why this batch is eligible
again; the assertion is what keeps that fact honest rather than a number in this
file.

The counter histogram is printed on every run and asserts the families the
fixture actually owns. At the fixture width and batch both measured promotions are
live: its time-mix projection takes the geometry-aware composition (the tuned
`linear3-forward` module is admitted there) and, since the channel-mix A/B landed,
its channel mix composes through the tuned `linear-residual-forward` module as
well. The measured histogram is therefore
`linear-forward=44, linear-bias-forward=56, linear-residual-forward=60,
linear3-forward=20` (180 Intel dispatches, portable run 0), and the test asserts
`linear3-forward == 20` and `linear-residual-forward == 60` instead of discarding
them, so a regression that reroutes either call site fails on the count rather
than only on the numbers.

That histogram is also the record of a promotion, not just of coverage. When this
section was first written the channel-mix family was the scorer's zero: the
fixture kept the fused producer and `linear-residual-forward` was pinned by
forcing the composition in the packed-cell check instead. The channel-mix A/B
(`validation/FUSED_FORWARD_PRODUCER_AUDIT.md`, rank 1) then measured the
composition faster at width 128 for batches 8/16/32, the per-device record took
those geometries, and the fixture -- width 128 at batch 8 -- started exercising
the family for real. So the run that first failed this assertion was the promotion
working: the count moved because the graph did, and the test has been updated to
the new measurement rather than the measurement weakened to the old count. The
families that are still zero are the ones no fixture geometry reaches
(`silu-forward`, `silu-backward`, `adamw`), and they stay in the printed histogram
for the same reason.

The LM auto-tuner suite (`cargo test --release --lib lm_execution::`) pins the
plan level: `rows16_cross_row_arms_are_offered_and_cache_admission_requires_the_arm`
builds the capability-ready geometry, asserts all five rows16 cross-row arms are
offered by the candidate list, each is admitted by the shared `plan_admitted`
filter, and a plan naming an arm the geometry does not offer is rejected.

## Adding a variant

1. Add the GLSL source as `shaders/<kernel>_<vendor>[_<generation>].comp`,
   keeping the push-constant layout, binding order, per-element math, and
   workgroup coverage.
2. Compile to `.spv` with the command above and commit both files.
3. Extend `VendorKernelFamily` with the module, its capability gates, and its
   `local_size`.
4. Extend `select_kernel` (capability gates) and
   `intel_matmul_geometry_supports_for` (per-dispatch geometry gate) as needed.
5. Measure with the experiment hook on the target hardware; admit it only if it
   wins and stays bit-exact. If it loses, move it to
   `shaders/vendor_experiments/` and record the result here. The one admitted
   module without a measured win is the RDNA3 set, which is gated on the portable
   tile threshold and documented as structural (`AMD RDNA3 (gfx11) variant` above);
   do not add another unmeasured variant without that same kind of gate.
6. Run `cargo test --release --lib vendor::` on the new vendor's hardware and on
   a second vendor. The parity half of that suite executes the module on whatever
   adapter is present, so bit-exactness is checked on a second vendor even when
   the selection cannot be exercised there.

`validation/verify_vendor_parity.py` runs layers 1 and 3 end to end on the
executing adapter: it prints the resolved kernel plan, runs the full forward
registry (75 comparisons, including the gate-width `gpt2_wide`/`llama_wide`
fixtures) with that plan and again with
`HIERARCHOS_VULKAN_FORCE_VENDOR=amd`, and fails unless both runs pass and every
comparison reports the same `max_abs` and `failing_values`. It works on any
vendor - on AMD itself the forced run is the real run, so the check degrades to
the plain qualification plus a portable-path regression guard.

```powershell
python hierarchos-vulkan/validation/verify_vendor_parity.py
```

### Machine-level qualification on the Gen9 target

Measured on the tuning target itself (Intel HD Graphics 520, `0x8086:0x1916`,
driver 31.0.101.2115, Windows, Vulkan 1.3.215), re-run on 2026-10-17 with the
Intel modules selected for production geometry:

| Check | Result |
| --- | --- |
| `hierarchos-vulkan-devices -- --kernel-plan` | four matmul families `intel-gen9-lds` (geometry-gated), `silu-forward`/`silu-backward`/`adamw` `portable` |
| `cargo test --lib` (debug) | 762 passed / 0 failed / 9 ignored |
| `cargo test --release --lib` | 762 passed / 0 failed / 9 ignored |
| `cargo test --release --lib vendor::` | 23 passed / 0 failed, all admitted modules bit-exact on device |
| Golden harnesses (`--golden-strict`) | 406/406 invocations verified (144 dense, 183 reduction, 79 attention) |
| `training_step_probe` at the ladder's probe args | `mean_loss 5.5432177`, 6368 dispatches, 2980 barriers (bound 3076) |
| `check_shader_provenance.py` | 467/467 identical |
| `check_push_constant_layouts.py` | PASS, 356 push blocks / 109 host structs |
| `check_oracle_parity.py` | PASS, 810 portable files, 3 documented divergences |
| Headline forward logits (`--headline-strict`) | exit 0, 21 comparisons, worst `5.96e-8` at the unchanged `2e-7` gate |
| Headline two-step AdamW (`--headline-strict`) | exit 0 (same 14-family claim as the AMD target) |
| Forced-portable logits (`verify_vendor_parity.py`, full registry) | exit 0, identical `max_abs` and `failing_values` on all 75 rows (worst `1.79e-7`), including the gate-width `gpt2_wide`/`llama_wide` rows where the tuned modules are actually dispatched |
| PEFT green matrix (`verify_peft_green_matrix.py --stage all --jobs 2`) | 32/32 LoRA, 32/32 switching, 25/32 saved, provenance `inputs_unchanged=true` |

The forced-portable harness used to run only `--headline-strict`; it now runs the
whole forward registry, so the two gate-width fixtures are compared across both
kernel paths as well. Those two rows also read the per-family dispatch counter:
`transformer_logits` prints it to stderr when `HIERARCHOS_VULKAN_VARIANT_TRACE` is
set, and the harness fails unless the tuned family the graph uses was dispatched
(`linear-bias-forward` for GPT-2, `linear-forward` for Llama - 4 and 5 dispatches
per case). The rule is two-sided: a family the plan resolves to a vendor module
must be dispatched, and a family planned `portable` (which is what the forced run
reports) must be dispatched zero times, so the assertion also holds on an adapter
with no module for that family. The counter is a thread-local incremented only
when a dispatch clears the geometry gate, so it is compiled into release builds
rather than test-only. Below-gate rows select the portable module on both paths,
so only those two rows exercise a real tuned-vs-portable comparison through the
logit path; the remaining tuned families stay covered by
`vendor::intel_variants_are_bit_exact_with_portable_kernels`.

The seven red saved-module fixtures are gemma3, gemma4, minimax_m2,
minimax_m3, mistral4, qwen3_5_full and smollm3 - the same rows listed in
`COMPATIBILITY.md`, which were measured with these very binaries. They are
therefore **not** an Intel gap: every one of those rows is computed by modules
the vendor layer does not touch (it only substitutes the row-major matmul
families), and each failing value is a 1-2 ulp adjoint at a shared
embedding / `lm_head` / input-norm head amplified by the RMSNorm backward. The
vendor layer cannot move them in either direction: it is bit-exact where it is
selected and byte-identical where it is not. A fresh AMD-machine run at this
exact oracle pin was not possible from this host (there is no AMD adapter here),
so the AMD-side record remains the 2026-09-30 32/32 all-stage statement on the
older `5.16.0.dev0` oracle documented in `COMPATIBILITY.md`; what this machine
contributes is that the portable (AMD-target) modules and the tuned Intel
modules produce identical numbers on every surface both of them can run.

The seven red rows above were measured while the narrow-GEMM and log-softmax
modules were hard-coded to the AVX2 shape on every host. Selection now follows
the host CPU ISA (see `COMPATIBILITY.md`, "Host ATen vector shape"), which on
this AVX2-only laptop resolves to the same 8-lane modules, so the record above
still describes the Gen9 target exactly. Because its red set overlaps five of the
six rows those same kernels pushed past the gate on an AVX-512 host, re-running
one of them here with `HIERARCHOS_ATEN_VECTOR_WIDTH=16` is the cheapest way to
confirm the oracle-boundary explanation above.

### Gen9 host-shape verification after the ATen-shape change

The experiment the previous paragraph proposes was run on the tuning target
itself (Intel i5-6200U / HD Graphics 520, AVX2-only: `avx512f` absent, so
`aten_vector_width()` resolves to 8) once the host-ISA change (`09db9ff`) landed.
The `2e-7` gate is unchanged.

| Check | Result |
| --- | --- |
| Committed 8-lane and vendor modules | unmodified by the commit; `transformer_cross_entropy.spv` and `falcon_h1_cross_entropy.spv` rebuild byte-for-byte from the committed sources with the local glslang 16.6.0, and the pre- and post-commit `transformer_cross_entropy.comp` emit byte-identical SPIR-V at 8 lanes |
| `cargo test --lib` (debug) | `693 passed / 0 failed / 9 ignored` |
| Full PEFT matrix (`--stage all --jobs 2`) | `32 families, 0 failures`, 32/32 LoRA, 32/32 switching, **32/32 saved**, `inputs_unchanged=true` |
| Same matrix vs the pre-commit report | all 32 families bit-identical field-by-field (peft, gradient, two-step AdamW, frozen base, resume, lifecycle) |
| `verify_hf_logits.py --headline-strict` | exit 0, zero failing values |
| `HIERARCHOS_ATEN_VECTOR_WIDTH=8` on `gemma4` `saved` | byte-identical to the probe default |
| `HIERARCHOS_ATEN_VECTOR_WIDTH=16` on `gemma4` / `smollm3` `saved` | both fail (`model.embed_tokens` adjoint `3.2187e-6` / `2.98e-7`), the mirror image of the wrong-shape damage the AVX-512 target measured |

The dated `25/32 saved` row above is the 2026-10-05 measurement, superseded by
the repair rounds: this laptop's saved stage has been 32/32 since its 2026-10-06
matrix runs, so the seven rows are not a live Gen9 gap, and the forced-16
failures show its oracle is AVX2-shaped - the oracle-boundary explanation above
stands, and the host-ISA change leaves the Gen9 dispatches exactly as tuned.

The 16-lane `falcon_h1_cross_entropy_lanes16.spv` was built by the AVX-512
machine's glslang and does not reproduce with the local 16.6.0 build (one extra
type/id and one `+inf` materialisation differ); it is dispatched only where
`avx512f` is present, so it is left untouched rather than swapped without a
dual-host re-qualification.

## Watchdog-safe submissions ported onto the Gen9 target

The AMD target (target A) protects `amdgpu`'s scheduler timeout by retiring each
compute dispatch as its own synchronous submission. The equivalent failure is
bound by the Windows driver's timer on target B, whose resets this project has
measured, so the same pattern is now selected by device class rather than by
vendor alone: `watchdog_submission_slicing_enabled_for` keeps AMD on Linux and
adds the Intel Gen9/Gen9.5 PCI IDs on Windows at a *measured* length-based slice
point instead of the AMD target's per-dispatch one, using the one Gen9 ID table the
crate publishes (`vendor::is_intel_gen9_pci_id`, also used by the FP16 LM
reliability policy, and asserted against the table's definition in `vulkan.rs` by
`gen9_pci_table_is_the_one_the_vendor_layer_publishes`). Other Intel generations,
other vendors and AMD outside Linux keep the legacy single-submit path.

A watchdog bounds a submitted job in *time*, so the bound is measured rather than
chosen: each retirement is timed on the host and the budget becomes the count
whose measured cost fits `GEN9_WATCHDOG_SLICE_MICROS` (400 ms), floored at the
1024 dispatches the port shipped at and guarded at 8192. The target is the safety
bound; the guard is only there so a cheap sample cannot size a submission without
limit, and it is deliberately above the fixture's own 6,368-dispatch step -- a
guard below one step (the earlier 4,096) makes every step pay a retirement for a
submission that is already five times inside the driver's timeout, which is where
the 0.9% regression this pass fixed came from. A sample can only widen the bound,
never tighten it below the shipped count, and the widest measurement a device has
taken is published for the batches after it, so a bound learned while the graph
warms up is not re-learned on every step. This is what the port's own note asked
for: a boundary that responds to submission length instead of a count that has to
be re-measured by hand when the kernels change.

Measured on this target (Intel HD Graphics 520, `0x8086:0x1916`, driver
31.0.101.2115, Windows, debug build). Median step of `training_step_probe --batch
2 --tokens 8 --steps 6`, every arm interleaved in five rounds so the machine's own
drift reaches every arm equally:

| Arm | Median step | vs legacy |
| --- | --- | --- |
| legacy single submit | 381.8 ms | -- |
| 8192 dispatches, forced (what the measurement converges to) | 381.7 ms | -0.0% |
| **shipped default (measured bound, 400 ms target)** | **382.7 ms** | **+0.2%** |
| 4096 dispatches, forced (the port's earlier ceiling) | 385.1-399.5 ms | +1.2 to +1.8% |
| 1024 dispatches, forced (the fixed count the port shipped) | 405.6 ms | +5.7% |
| 512 dispatches, forced | 426.1 ms | +12% |
| 256 dispatches, forced | 449.7 ms | +18% |
| 64 dispatches, forced | 556.7 ms | +46% |
| 1 dispatch (the AMD slice point) | 1928.3 ms | +407% |

The last four rows are the earlier forced sweep (legacy step 380.0 ms in that
session); the ratios reproduce and the absolute milliseconds belong to the session
that took them. The first five are one interleaved session. The shipped arm's
+0.2% is one calibration retirement per process -- the budget starts at the floor,
the first retirement sizes it, and it then sits at the guard for the rest of the
process -- so a seven-step probe amortizes it while a real run does not see it at
all. Against the fixed 1,024-dispatch count that shipped first, the measured bound
is 5.5 points of step time; against the intermediate 4,096 guard, 0.7 points.

The second fixture makes the net effect a speedup rather than a recovery:
`cargo test --lib -- hierarchos_causal_lm_` (six tests, ten seconds of real
training work) runs in **10.28 s** with the measured bound live against **10.49 s**
with `HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0` (median of three interleaved
repetitions each) -- about 2% faster than the same binary's unsliced path, because
retiring a slice starts the GPU on it while the host records the next one.

Every arm writes byte-identical training state -- the golden's own three cases
hash to `95f48308...`, `6c97733e...` and `cc4cb821...` under the legacy path, the
measured default and a forced 1024-dispatch point alike -- so the granularity is a
cost decision and not a numerical one. Copying the AMD slice point across would
have cost about 5x per step on this part, so the port retires on accumulated
length: 1024 dispatches is about 60 ms of this fixture's work and holds a
submission under a three-second watchdog for every per-dispatch cost measured
here except the single-dispatch attention-backward class, which only the
kernel-level split can shorten -- the resets recorded here are that class, and
they are addressed by the key-range split's per-block submissions, which use the
same `submit_and_restart` primitive. The class also retires at the transformer
recorders' phase boundaries whatever the budget says.

Machine-level checks with the port live:

| Check | Result |
| --- | --- |
| `cargo test --lib` | 717 passed / 0 failed / 9 ignored |
| `cargo test --lib -- hierarchos_causal_lm_`, live versus unsliced, three interleaved repetitions each | **10.28 s versus 10.49 s** -- this fixture is about 2% *faster* with the measured bound live than with slicing off |
| `validation/verify_training_state.py`, default, `WATCHDOG_SLICE_DISPATCHES=1024` and `WATCHDOG_SUBMISSIONS=0` | PASS on all three arms, 13 steps, 3,817 exact tensor digests, and the same three trajectory dumps by sha256 |
| `validation/attention_kernel_cache_check --golden validation/kernel_golden/attention.json` | 79/79 recorded digests PASS on this adapter |
| `validation/verify_training_stack.py --skip-build` | 6/7 at this checkpoint -- every runnable step PASS (goldens, training-step probe, training-state parity); shader provenance was BLOCKED because `spirv-val` was not installed then. The toolchain pin section installs it, and the ladder is 7/7 with that step running |
| `validation/verify_hf_logits.py --families smollm3 smollm3_yarn phi3 --atol 2e-7 --rtol 0` (`.venv-vulkan` oracle, source rebuilt by the harness) | PASS with the measured bound live -- 6 cases, 0 failing values, worst `max_abs` `5.96e-08` against the fixed `2e-07` ceiling |
| `watchdog_slicing_is_wired_into_the_training_path_on_this_adapter`, `watchdog_slice_budget_retires_on_accumulated_length` | ok -- the training-path constructor follows the policy and its budget, the plain one never slices unasked |
| `watchdog_slice_budget_is_a_measured_bound_inside_a_measured_window`, `watchdog_slice_budget_follows_the_measured_retirement_on_this_adapter` | ok -- the extrapolation, both clamps and the unusable samples without a device; on device, a real retirement widens the floor, stays inside the window and is published for the batches after it |
| `python -m unittest discover -s validation -p test_training_stack.py` | 6 tests ok -- a run is compared against its own device's entry, another device's count is never substituted, and an unrecorded device fails |

The structural counters a caller reads (`dispatch_count`, barrier, pipeline-bind,
descriptor-bind, push-constant and descriptor-set counts) are carried across a
retirement, so they still describe the recorded stream rather than the slice that
happened to be open at the end; one extra barrier per slice boundary is the only
counter a split adds (2,983 versus 2,980 on a two-step probe). The stack
check's probe expectation is device-keyed for the same reason the trajectory
golden is: this target's graph is not the AMD one (the vendor layer selects Gen9
kernels here), and the same probe arguments dispatch 6,368 times on this part
against 6,464 as recorded on AMD, on both arms. That number used to be one
top-level value "compared for every device", which left a second device able only
to fail it or overwrite the first device's record; it now sits in the device entry
beside `mean_loss`, `--record-expectation` merges instead of rewriting, and the
barrier bound is a device-independent bound that only widens. Both arms PASS the
probe step against it, and AMD's `6,464` is untouched.

The target's trajectory in `validation/kernel_golden/training_state.json` is what
makes that verify check possible on this machine; `validation/record_training_state.py`
added it without touching the AMD trajectories. See `COMPATIBILITY.md`,
"Watchdog-safe submissions" and "The Gen9 target's own training-state
trajectory", for the policy, the overrides and the full measured record.

## Qualification checklist

```powershell
# 1. Confirm the vendor plan on this machine.
cargo run --release --bin hierarchos-vulkan-devices -- --kernel-plan

# 2. Portable-vs-vendor parity and timing on the executing adapter.
cargo test --release --lib vendor:: -- --nocapture --test-threads=1

# 3. Store the full training-graph regression (unchanged by vendor tuning).
cargo test --release

# 3a. The automated shader checks: every embedded module rebuilds, every push
#     block matches its host struct, and no portable artifact drifts from the
#     read-only stable baseline.
python validation/check_shader_provenance.py
python validation/check_push_constant_layouts.py
python validation/check_oracle_parity.py

# 3b. On an adapter with no digests recorded yet, record that device's digests and
#     then enforce them. `--write-golden` loads the file first and merges this
#     device's entries, so every other device's digests are left alone; coverage
#     is recorded with the same merge and refuses an unexplained shrink.
./target/release/matmul_kernel_validation.exe --write-golden validation/kernel_golden/dense.json
./target/release/reduction_kernel_validation.exe shaders --write-golden validation/kernel_golden/reduction.json
./target/release/attention_kernel_cache_check.exe --write-golden validation/kernel_golden/attention.json
python validation/verify_training_stack.py --only golden --skip-build

# 3c. A machine without the stable baseline reports `oracle parity` as blocked, not
#     green: it never checked what it could not see. Point it at a copy with
#     `--oracle PATH` (or `HIERARCHOS_STABLE_ORACLE`) to run the comparison there.

# 4. On an RDNA3 adapter (Navi 31/32/33 or the integrated Phoenix/Hawk Point parts):
#    the same three steps, with the plan expected to read `amd-rdna3-lds` for the four
#    matmul families and `portable` for every other family.
#    `rdna3_devices_select_the_rdna3_modules` asserts that selection on the executing
#    adapter, and `--bootstrap-coverage` records the device's own golden-coverage
#    entries instead of defaulting to green (see *Kernel-golden invocation coverage*).
```

## Cross-vendor parity evidence

Vendor tuning is only safe if the *other* vendor's execution is provably
unchanged. Four machine-checked layers, in increasing strength:

1. `non_intel_devices_keep_the_portable_modules` and
   `portable_fallback_bytes_are_the_mainline_artifacts` assert that a non-Intel
   device resolves to the exact pre-tuning module bytes for every family, so the
   AMD path executes byte-identical SPIR-V to the vendored AMD target.
2. `intel_variants_are_bit_exact_with_portable_kernels` and
   `rdna3_variants_are_bit_exact_with_portable_kernels` execute both modules on
   whatever adapter is present and require raw-bit equality at every geometry
   (including geometries the production gate keeps on the portable module), so
   neither vendor's modules can move a result even where they are not selected.
   The RDNA3 half is what makes its unmeasured timing safe to ship: the selection
   is new, the numbers are not.
3. The model-level surfaces are re-qualified on both paths.
   `HIERARCHOS_VULKAN_FORCE_VENDOR=amd` (classification override -> portable
   modules) and the default (Intel modules selected) runs of
   `validation/verify_hf_logits.py --headline-strict` produced the same 21
   unmasked/mixed-padding comparisons with the same `max_abs` and
   `failing_values` on every row (worst `5.96e-8` against the unchanged `2e-7`
   ceiling), i.e. the AMD-green forward surfaces are green and numerically
   identical on this Intel machine.
