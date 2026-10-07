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
  and picks per dispatch from `intel_matmul_geometry_supports` (see below).
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
with the pure gate `intel_matmul_geometry_supports`:

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

| Family | Portable (every vendor, including AMD) | Intel Gen9 |
| --- | --- | --- |
| `linear-forward` | `shaders/linear_forward.spv` | `shaders/linear_forward_intel_gen9.spv` |
| `linear-bias-forward` | `shaders/linear_bias_forward.spv` | `shaders/linear_bias_forward_intel_gen9.spv` |
| `linear-residual-forward` | `shaders/linear_residual_forward.spv` | `shaders/linear_residual_forward_intel_gen9.spv` |
| `linear3-forward` | `shaders/linear3_forward.spv` | `shaders/linear3_forward_intel_gen9.spv` |
| `silu-forward`, `silu-backward`, `adamw` | own portable module | none (candidates measured slower; see below) |

The Intel matmul variants keep the portable 16x16 workgroup, output mapping, and
ascending-k `fma()` chain, and stage each 16x16 `x`/`w` k-slice in workgroup
memory. Global loads per FMA drop ~8x (6 -> 6/16 for the three-way
`linear3-forward`), the per-family epilogues are unchanged and in the portable
operand order (`sum + bias[col]`, `residual[index] + sum`), and the arithmetic
sequence is untouched, hence bit-identical results. Workgroup bytes:
`2 * 16 * 17 * 4 = 2176` for the two-tile families, `6 * 16 * 17 * 4 = 6528`
for `linear3-forward` (both inside the Gen9 32 KiB limit, and asserted against
the decoded module by the capability test).

`linear-residual-forward` and `linear3-forward` are the RWKV-native
channel-mix / post-mix and time-mix projection slots, so the tuning covers the
hottest matmuls of both served graphs (the Transformer training path and the
native Hierarchos path) rather than only the Hugging Face forward projections.

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
is asserted against every row in `intel_matmul_geometry_gate_matches_the_measured_matrix`.

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
while production geometry runs 1.4x-6.8x faster.

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

(glslang 16.6.0). `validation/audit_shader_artifacts.py` re-compiles GLSL sources
with the compiler the caller configures (`--compiler glslc` by default); run it
with a matching toolchain when auditing byte provenance, and note that `glslc`
produces equivalent modules with a different SPIR-V generator word.

The crate does not trust the binary: `vendor.rs` decodes the committed modules in
its test suite and asserts that the declared `LocalSize`, workgroup-memory
variables, and barriers match the capability gates and the host dispatch
contract, so a stale or mistargeted `.spv` fails the suite rather than a user's
driver validation.

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

The counter histogram is printed on every run and asserts the families the
fixture actually owns. At the fixture width the native graph routes its
time-mix and channel-mix projections through the fused producers, so the
measured histogram is
`linear-forward=44, linear-bias-forward=56, linear-residual-forward=0,
linear3-forward=0` (100 Intel dispatches, portable run 0). The two zero families
are therefore pinned by the on-device A/B test at production geometry instead,
and they are the slots a wider (non-fused) native configuration uses. This is
recorded rather than asserted away so a future configuration change that starts
exercising them shows up as a histogram change, not as a silent loss of
coverage.

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
   `intel_matmul_geometry_supports` (per-dispatch geometry gate) as needed.
5. Measure with the experiment hook on the target hardware; admit it only if it
   wins and stays bit-exact. If it loses, move it to
   `shaders/vendor_experiments/` and record the result here.
6. Run `cargo test --release --lib vendor::` on the new vendor's hardware and on
   a second vendor.

`validation/verify_vendor_parity.py` runs layers 1 and 3 end to end on the
executing adapter: it prints the resolved kernel plan, runs the headline
forward qualification with that plan and again with
`HIERARCHOS_VULKAN_FORCE_VENDOR=amd`, and fails unless both runs pass and every
comparison reports the same `max_abs` and `failing_values`. It works on any
vendor - on AMD itself the forced run is the real run, so the check degrades to
the plain qualification plus a portable-path regression guard.

```powershell
python hierarchos-vulkan/validation/verify_vendor_parity.py
```

### Machine-level qualification on the Gen9 target

Measured on the tuning target itself (Intel HD Graphics 520, `0x8086:0x1916`,
driver 31.0.101.2115, Windows, Vulkan 1.3.215) on 2026-10-05, with the Intel
modules selected for production geometry:

| Check | Result |
| --- | --- |
| `cargo test --lib` (debug) | 689 passed / 0 failed / 9 ignored |
| `cargo test --release --lib` | 689 passed / 0 failed / 9 ignored |
| `cargo test --release --lib vendor::` | 13 passed / 0 failed, all admitted modules bit-exact on device |
| Headline forward logits (`--headline-strict`) | exit 0, 21 comparisons, worst `5.96e-8` at the unchanged `2e-7` gate |
| Headline two-step AdamW (`--headline-strict`) | exit 0 (same 14-family claim as the AMD target) |
| Forced-portable logits (`FORCE_VENDOR=amd`) | exit 0, identical `max_abs` and `failing_values` on all 21 rows |
| PEFT green matrix (`verify_peft_green_matrix.py --stage all --jobs 2`) | 32/32 LoRA, 32/32 switching, 25/32 saved, provenance `inputs_unchanged=true` |

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

## Qualification checklist

```powershell
# 1. Confirm the vendor plan on this machine.
cargo run --release --bin hierarchos-vulkan-devices -- --kernel-plan

# 2. Portable-vs-vendor parity and timing on the executing adapter.
cargo test --release --lib vendor:: -- --nocapture --test-threads=1

# 3. Store the full training-graph regression (unchanged by vendor tuning).
cargo test --release
```

## Cross-vendor parity evidence

Vendor tuning is only safe if the *other* vendor's execution is provably
unchanged. Three machine-checked layers, in increasing strength:

1. `non_intel_devices_keep_the_portable_modules` and
   `portable_fallback_bytes_are_the_mainline_artifacts` assert that a non-Intel
   device resolves to the exact pre-tuning module bytes for every family, so the
   AMD path executes byte-identical SPIR-V to the vendored AMD target.
2. `intel_variants_are_bit_exact_with_portable_kernels` executes both modules on
   whatever adapter is present and requires raw-bit equality at every geometry
   (including geometries the production gate keeps on the portable module), so
   the Intel modules cannot move a result even where they are not selected.
3. The model-level surfaces are re-qualified on both paths.
   `HIERARCHOS_VULKAN_FORCE_VENDOR=amd` (classification override -> portable
   modules) and the default (Intel modules selected) runs of
   `validation/verify_hf_logits.py --headline-strict` produced the same 21
   unmasked/mixed-padding comparisons with the same `max_abs` and
   `failing_values` on every row (worst `5.96e-8` against the unchanged `2e-7`
   ceiling), i.e. the AMD-green forward surfaces are green and numerically
   identical on this Intel machine.
