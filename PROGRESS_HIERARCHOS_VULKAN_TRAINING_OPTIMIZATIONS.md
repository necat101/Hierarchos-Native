# Hierarchos Vulkan training-kernel optimizations — 2026-10-15

Scope: the Hierarchos training compute kernels and the compute-dispatch runtime.
Model mathematics, objective definitions, optimizer equations, and trainer
behavior are unchanged. Every kernel whose artifact changed is compared
bit-for-bit against its pre-change artifact (`git show HEAD:...`) on the local
AMD Radeon Graphics device, because the project treats exact FP32 reproduction
as the contract for a kernel optimization.

This continues the work left mid-verification at 03:57 (dense/attention
candidates compiled, comparisons not yet run). The separately documented
objective/data-boundary port in `PROGRESS_HIERARCHOS_OBJECTIVES_PORT.md` is
untouched by this pass.

## Changed kernels

| Kernel family | Change |
| --- | --- |
| `linear_forward`, `linear_bias_forward`, `linear_residual_forward`, `linear_*_fp16_packed` | 16×16 output tile with shared 32-wide K tiles; each input/weight value is loaded once and reused across the tile. Small/recurrent shapes keep the original scalar schedule. Increasing-K FMA order preserved. |
| `parameter_matmul_forward`, `parameter_matmul_bias_forward`, `_fp16_packed` | Same tiling for the weight-major projection; leading dimension read once per K tile. |
| `parameter_matmul_input_grad`, `_fp16_packed`, `_fp16_native_compute` | Same tiling; FP16 weights are unpacked once per tile instead of once per output element. |
| `transformer_attention_forward` | Per-row score/softmax tile cached in shared memory (rows ≤ 1024 keys), removing the recomputation of every QK dot product for every value channel. Longer rows keep the streaming path. |
| `transformer_attention_backward` | K/V adjoints evaluate each (query, key) score, probability and `dp` once per row and reuse row statistics instead of replaying the full softmax per key and channel. Strict-shape fallbacks retained. |
| `transformer_qwen_gdn_gate_norm_param_grad` | The `head_dim`-fold mean-square reduction is computed once per row tile and shared across features; previously it was recomputed inside every feature, making the pass quadratic in `head_dim`. |
| `transformer_rms_norm_input_grad`, `_residual` | The eight-lane topology is selected before reading the row, removing a wasted scalar prepass that was always discarded when `unfused_products == 2`. |
| `adamw` | The two bias-correction powers depend only on the step; they are evaluated once per workgroup and broadcast, instead of once per parameter. |

## Dispatch runtime

- **Pipeline interning.** Kernels that share the same SPIR-V module and pipeline
  layout now share one `vk::Pipeline` per device, held through a weak interner
  entry so driver memory is released with the last kernel. Identical layers in
  the same graph previously created separate pipelines.
- **Barrier elision from SPIR-V reflection.** `src/spirv_access.rs` reads
  `NonWritable`/`NonReadable` decorations, including per-member decorations on
  storage blocks, and reports conservative `MayWrite` for anything it does not
  recognize or for contradictory aliased declarations. The command recorder
  inserts a compute-to-compute barrier only when the next dispatch can alias a
  buffer touched by any dispatch since the last barrier, so read/read overlaps
  (activation reused by two consumers) no longer serialize.

## Verification

| Check | Result |
| --- | --- |
| Dense-kernel parity + reference model (`matmul_kernel_validation`) | 12 kernels × 12 shapes = **144/144 bit-identical** to HEAD |
| Attention parity (`attention_kernel_cache_check`) | **79 parity cases × forward/backward bit-identical**, including cache boundaries 512/513 and 1024/1025, windowed, paged, sink, relative-bias, and both strict-head/stats fallbacks |
| Reduction parity (`reduction_kernel_validation`) | **183 GPU cases bit-identical**: Qwen weight grad 8 shapes, RMSNorm input grad ×2 kernels × 5 shapes × 20 config combinations, AdamW 5 lengths × 3 steps |
| Runtime unit tests | 6/6 pass (pipeline sharing and lifetime, interning key separation, reflected-access barrier elision) |
| Maintained regressions | Causal LM 6 passed; CE 3 passed; Hierarchos trainer 40 passed; attention/linear/parameter-matmul library tests 71 passed |
| Full library suite (`cargo test --lib`) | **704 passed, 0 failed, 9 ignored** |
| Shader provenance | Every changed artifact reproduces byte-for-byte from its source (`glslc`, `glslc -O`, or `glslangValidator -V` per shader family) and validates with `spirv-val --target-env vulkan1.1`; CE: 32/32 variants |

### Measured speedups (GPU timestamp median of 5 samples)

Dense projections — 72 timed configurations, 12 kernels × 6 shapes:

| Rows | Median | Range |
| --- | --- | --- |
| ≥ 16 (training-like) | **2.67×** | 0.99× – 4.43× |
| < 16 | 1.06× | 0.89× – 2.56× |

Best cases: `parameter_matmul_input_grad_fp16_packed` 4.43× (64×1024×256),
`parameter_matmul_forward_fp16_packed` 4.29× (16×256×256),
`parameter_matmul_input_grad` 4.07× (64×1024×256), `linear_forward` 3.23×
(32×512×512). Batch-1 `(1, 512, 512)` and the smallest `(8, 64, 16)` shapes are
within ±11% of the baseline, as expected when the tile reuse cannot amortize the
two extra barriers.

Attention (`transformer_attention_forward` / `_backward`, 64 queries × 64 keys,
4 heads, dim 32):

| Configuration | Baseline | Optimized | Speedup |
| --- | --- | --- | --- |
| Gemma4 mode 0, forward | 4.49 ms | 0.73 ms | **6.2×** |
| Gemma4 mode 1, forward | 3.73 ms | 0.81 ms | **4.6×** |
| Gemma4 mode 0, backward | 53.4 ms | 45.0 ms | 1.19× |

Reductions:

| Kernel | Shape | Baseline | Optimized | Speedup |
| --- | --- | --- | --- | --- |
| Qwen gated-RMS weight grad | rows 128, heads 4, dim 128 | 2.73 ms | 0.232 ms | **11.8×** |
| Qwen gated-RMS weight grad | rows 256, heads 4, dim 192 | 4.74 ms | 0.462 ms | **10.3×** |
| RMSNorm input grad | rows 128, dim 256 | 0.276 ms | 0.218 ms | 1.27× |
| RMSNorm input grad + residual | rows 128, dim 256 | 0.435 ms | 0.374 ms | 1.16× |
| AdamW | len 65537 | 0.0219 ms | 0.0171 ms | 1.28× |

## Findings made while finishing the verification

- **Qwen gated-RMS weight gradient: a 1-ULP regression, found and fixed.** The
  candidate broadcast each row's already-evaluated `inversesqrt(...)` through
  workgroup memory; on this driver that shifted roughly half of the reduced
  elements by one ULP against the pre-change kernel. Publishing the raw
  mean-square sum and evaluating `inversesqrt(... / head_dim + epsilon)` in the
  consuming lane restores bit parity (183/183, 11.8×/10.3× retained), because
  the reciprocal square root then executes in the same lane context and
  instruction sequence as before. The `head_dim` multiplier on the mean-square
  reduction is what the pass removes; the per-feature root is not the cost.
- **Compile-flag provenance.** The pre-change Qwen artifact was built without
  `-O` while the candidate was built with `-O`. A flag-only probe (HEAD source
  compiled both ways, compared on GPU) is bit-identical, so the flag is
  numerically inert here and was not the cause; the source restructure was.
- **Reference-model calibration for native-FP16 dense kernels.** The dense
  harness's independent FP32 emulation cannot reproduce exact Float16 product
  rounding on awkward shapes: the *pre-change* `parameter_matmul_input_grad_fp16_native_compute`
  already exceeds its `5e-4` tolerance (max 2.6e-3). The harness now enforces
  bit-parity with the supplied baseline first and calibrates the advisory model
  against that baseline's own error, so a change that widens the deviation still
  fails while an environment limitation cannot mask a real mismatch.
- **Harness diagnostics.** The reduction harness now reports every mismatch with
  case context instead of stopping at the first one, and accepts
  `--candidate-dir` so any two artifacts can be compared; the attention harness
  accepts `--candidate-dir` and `--bench-repeats` and times with GPU timestamps
  instead of host wall time.

## Limitations

- **Attention backward mode 1 was not benchmarked.** The harness resets the
  device (`VK_ERROR_DEVICE_LOST`) on that case on this machine. The reset
  reproduces identically with unoptimized artifacts on both sides
  (`--candidate-dir .attention-optimization-baseline`), so it is a driver /
  watchdog limitation of the harness's long scalar backward dispatches, not a
  property of the optimization. Forward and mode-0 backward numbers are
  unaffected, and parity for that case passes.
- Timings are kernel-level on one AMD device with the debug-profile harnesses;
  no end-to-end trainer wall-clock A/B was measured in this pass, and no
  cross-vendor parity is claimed.
- The dense benchmark's small shapes (rows < 16) include a few sub-1.0×
  configurations; those shapes deliberately keep the scalar schedule, so the
  residual delta is launch/measurement noise rather than a systematic loss.

## Reproduce

```powershell
cd hierarchos-vulkan

# Parity + benchmarks (baseline artifacts are HEAD blobs; see the note below)
cargo run --bin matmul_kernel_validation -- --baseline-dir target/shader-matmul-baseline --bench --output target/matmul_report.json
cargo run --bin reduction_kernel_validation -- target/shader-reduction-baseline
cargo run --bin attention_kernel_cache_check -- .attention-optimization-baseline --bench

# Baseline reconstruction (once)
git show HEAD:hierarchos-vulkan/shaders/<name>.spv > target/shader-matmul-baseline/<name>.spv

# Shader provenance and validation
python validation/verify_hierarchos_ce_shaders.py
python validation/audit_shader_artifacts.py linear_forward parameter_matmul_input_grad ... --output target/shader_audit.json
glslangValidator -V --target-env vulkan1.0 shaders/transformer_attention_forward.comp -o <out>
glslc -O shaders/adamw.comp -o <out>
spirv-val --target-env vulkan1.1 shaders/<name>.spv

# Runtime and regression suites
cargo test --lib -j 1 -- --test-threads=1
cargo test --lib -j 1 hierarchos_causal_lm -- --nocapture --test-threads=1
cargo test --lib -j 1 hierarchos_ce_ -- --test-threads=1
cargo test --bin hierarchos-vulkan-train -j 1 -- --test-threads=1
```

The `target/shader-*-baseline` directories are scratch reconstruction of
pre-change artifacts (`target/` is ignored) and are not part of the tree.

---

# Second pass — 2026-10-15: binding-access audit, CI golden digests, end-to-end throughput

The first pass verified the kernel and runtime changes. This pass answers four
questions the first pass left open: whether the remaining shaders can declare
narrower storage access so the scheduler elides barriers, whether the parity
harnesses can run in CI without the original artifacts, what the attention
backward regression actually costs a whole training step, and what the
end-to-end effect of all of it is.

## Storage-access audit and barrier accounting

`validation/audit_shader_binding_access.py` reads each compiled module as ground
truth: a binding that no `OpStore`, atomic, or `OpImageWrite` can reach is
provably never written, and one that no `OpLoad`, atomic, or `OpImageRead`
reaches is provably never read. It reports where that proof is narrower than the
declared qualifier, which is exactly where the dispatch scheduler pays for an
unqualified block.

`validation/narrow_shader_binding_access.py` closes those gaps. For each affected
artifact it resolves the GLSL source and the build command that produced it,
**requires that command to reproduce the checked-in artifact byte-for-byte**
before editing anything, recompiles every variant built from that source, then
requires `spirv-val` to accept it, the module's read/write usage sets to be
unchanged, and the reported declaration to match the proof. It rolls the source
back and rebuilds if any of that fails.

Result: 41 bindings over 24 sources narrowed, and the audit now reports
`0 provably narrower binding(s) over 0 shader(s) out of 467 modules`.

Two toolchain facts surfaced on the way: the README's command for the CE fused
family omits `--target-env=vulkan1.1`, which the artifact needs, and the
`..._packed_fast_weight_reuse{4,8}` / `..._wg32` variants are built from the
`_weight_reuse2.comp` source with `-DRWKV_WEIGHT_REUSE_ROWS` / `-DRWKV_WEIGHT_REUSE_TILE`
and are not documented at all. The tool finds a working command by requiring
byte-reproduction, and says when the documented one no longer works.

### The measured barrier reduction is zero, and the reason is structural

(The fifth pass below closes this question: the rule is already exactly the hazard
set, so no declaration can lower the count, and the remaining levers are graph
structure and per-barrier cost.)

`src/bin/training_step_probe.rs` measures a real training step (raw-token
microbatch: forward, reverse, canonical AdamW) and prints dispatch and barrier
counts. On the fixed fixture (batch 2, 8 tokens, context 128, hidden 128):

| build | dispatches/step | barriers/step | step ms | tokens/s |
| --- | --- | --- | --- | --- |
| HEAD (`5004a8a`) | 6912 | 3412 | 172.76 | 92.62 |
| current tree | 6912 | 3076 | 167.98 | 95.25 |

Before and after the 41 narrowings on the same tree the count was 3076 both
times, so the 336-barrier improvement (and the 2.8% step-time gain) belongs to
the already-present reflection/runtime change, not to the narrowing. That is not
an accident of which shaders were left; it follows from the elision rule:

```rust
let overlaps_pending = buffers.iter().zip(binding_accesses).any(|(buffer, access)| {
    let region = buffer.buffer_region_key();
    (access.may_read() && self.pending_shader_writes.contains(&region))
        || (access.may_write() && self.pending_shader_buffers.contains(&region))
});
```

`pending_shader_buffers` is every region any earlier dispatch touched, and every
write also inserts there. So the second term reduces to "this binding may write
and the region was touched", and `writeonly` keeps `may_write` — a `writeonly`
declaration can never remove a barrier. Only `readonly` can, by clearing
`may_write`; of the remaining unqualified declarations exactly 5 are provably
read-only, all in `falcon_h1.comp`, which is not on the Hierarchos RWKV training
path. The narrowing is still worth landing as accuracy (it is what makes a
finer-grained rule possible, and it documents the access each kernel has), but it
could not move this number, and the measurement says so rather than the other way
round.

The "non-aliasing region" half of that question is already expressed where it can
be: `buffer_region_key()` is `(VkBuffer, view offset, view size)`, so two disjoint
views of one allocation are already distinct regions and nothing has to be
declared for that. The barrier rate is the sanity check that this granularity is
live — if every binding collapsed onto one region, every dispatch after the first
would barrier (6911 of 6912); the step reports 3076.

Parity after the narrowing: full library suite **704 passed, 0 failed, 9
ignored**; dense `144/144`, reduction `183/183`, attention `79 × fwd/bwd` all
bit-identical to their baselines; and the end-to-end probe's mean loss is
bit-identical between HEAD and the current tree over six real optimizer steps
from the same weights (4.6006656 both sides).

Reproduce:

```powershell
cargo run --bin training_step_probe -- --batch 2 --tokens 8 --warmup 1 --steps 5
python validation/audit_shader_binding_access.py
python validation/narrow_shader_binding_access.py            # dry run
python validation/narrow_shader_binding_access.py --apply    # rebuilds artifacts
```

## Golden digests: CI parity without the pre-change artifacts

`src/kernel_golden.rs` (included by path, like `vulkan.rs`) records a SHA-256 of
each invocation's output vector, keyed by invocation label and device name.
`--write-golden FILE` records, `--golden FILE` verifies, and in either mode the
baseline artifact directory becomes optional — the digests *are* the baseline.
A file accumulates one digest per device, so an AMD, Intel, or NVIDIA runner each
checks the entry recorded on its own hardware and reports another device's entry
as unverified instead of passing quietly.

Verified end to end on this device: dense 144, reduction 183, attention 79
invocations recorded and re-verified. The check is not vacuous — rebuilding the
gated-RMS weight-gradient kernel with a perturbed input scale produced exactly 8
digest mismatches on its 8 invocations (175 others verified) and exit status 1.

```powershell
cargo run --bin matmul_kernel_validation -- --write-golden validation/kernel_golden/dense.json
cargo run --bin matmul_kernel_validation -- --golden validation/kernel_golden/dense.json --golden-strict
```

A CI runner on hardware nobody has recorded yet would otherwise verify nothing and
still exit zero, so `--golden-strict` makes an unrecorded-for-this-device
invocation fatal. Both halves of that were checked: strict mode passes on the
recording device (144/183/79 verified, 0 unverified), and with the device key
rewritten to a foreign GPU the loose mode reports `0 verified, 144 unverified`
while strict mode exits 1 with `144 invocation(s) have no digest recorded for ...`.

## End-to-end trainer throughput, HEAD versus optimized

Both builds are the same probe binary built from the same source, one against
HEAD and one against the working tree, started from **the same model package** so
no weight difference can explain the gap.

| | HEAD | current | ratio |
| --- | --- | --- | --- |
| median step | 172.76 ms | 167.98 ms | 1.028× |
| steps/s | 5.788 | 5.953 | 1.028× |
| tokens/s (16 tokens/step) | 92.62 | 95.25 | 1.028× |
| dispatches/step | 6912 | 6912 | — |
| shader barriers/step | 3412 | 3076 | 0.901× |
| mean loss after 6 identical steps | 4.6006656 | 4.6006656 | bit-identical |

So the kernel and runtime work is worth 2.8% on this small fixture, where the
step is dominated by thousands of tiny dispatches rather than by the dense and
reduction kernels that carry the 2.7–11.8× kernel-level wins. The third pass below
measures that directly and confirms it: a step is 6912 dispatches at 175 µs for its
most expensive launch and 1–4 µs for its median one. The bit-identical
loss over six optimizer steps is the strongest end-to-end parity evidence
available here: same weights, same data, six AdamW updates, no divergence.

## Attention backward: the dominant cost, measured

`attention_kernel_cache_check` gained `--bench-{batch,query,key,heads,kv-heads}`
so one run can separate "not enough workgroups" from "too much serial work per
workgroup". Backward mode 0, `query=key=64`, 4 query heads, 2 KV heads:

| batch | workgroups | HEAD ms | current ms | ratio |
| --- | --- | --- | --- | --- |
| 2 | 4 | 54.6 | 46.3 | 1.18× |
| 8 | 16 | 56.2 | 45.4 | 1.24× |
| 32 | 64 | 56.5 | 88.6 | 0.64× |
| 128 | 256 | 78.3 | 262.9 | 0.30× |

Three facts fall out of this, and none of them flatter the current artifact:

1. The backward kernel declares `local_size = 1×1×1` and dispatches one
   workgroup per `(KV head, batch row)`, so at realistic shapes the grid is tiny
   and every element of the per-key reduction runs as a dependent single-lane
   chain. Step time at 4 workgroups is ~46 ms and is *independent of the
   workgroup count* up to 16: the device is idle, not saturated.
2. The current artifact uses workgroup memory the HEAD artifact did not (10
   workgroup variables versus 0), which lowers residency: it wins only while the
   grid is tiny and loses badly from batch 32 upward, 3.4× slower than HEAD at
   batch 128.
3. The forward path, restructured the same way, is 3.9–6.1× faster at every
   shape, because it dispatches `query × heads × batch` workgroups instead of
   `KV heads × batch`.

A bit-exact fix is available and needs no atomics: in the strict-adjoint path the
per-key phase zeroes `shared_k_cache`/`shared_v_cache` per key, reduces that key
over `(head, query)` in a fixed order, and writes that key's gradient before
moving on — so keys are independent accumulators, and splitting them across lanes
or across a `(KV head, batch, key tile)` grid reorders no FP sum. Independent
keys per lane also break the dependency chain that is doing the damage. Landing
it means moving the shared row statistics to a global pass (or sizing per-lane
scratch) plus the host dispatch/allocation change, and verifying against the
79-case parity harness.

It is **not implemented in this pass**: the change spans the 796-line kernel and
its host call sites, and the backward benchmark already trips
`VK_ERROR_DEVICE_LOST` in mode 1 on this device, which makes long verification
runs unreliable here. The finding to act on is the regression in the table: the
current backward artifact is slower than HEAD at every batch size that matters,
so the next change to it should start from the HEAD artifact's occupancy rather
than from the current one.

One correction from the third pass below: the canonical raw-token trainer never
dispatches this kernel. "Dominant cost" and "highest-value target" here are true
within the attention path's own benchmark, and the same *signature* — cost flat in
the workgroup count — turns out to be the trainer's top cost as well, in a
different kernel.

## Limitations of this pass

- The end-to-end numbers are one small fixture (batch 2, 8 tokens, 16 tokens per
  step) on one AMD device, in an unoptimized build. They say what the kernel and
  runtime work is worth at that shape, not what it is worth at production shapes;
  the backward table above is the counterexample to extrapolating from it.
- Attention backward mode 1 still cannot be benchmarked: the harness resets the
  device, identically with unoptimized artifacts on both sides.
- The barrier count is a property of one graph and one command-batch discipline.
  A different graph (transformer trainer, LoRA) would have a different mix, and
  the audit tool is how to check it there.
- Digest-based parity is device-keyed by design, so a CI runner on new hardware
  reports its invocations as unverified until someone records a golden file for
  that device.

---

# Third pass — 2026-10-15: what the ~6900 dispatches of a step actually cost

The first two passes optimized kernels and then measured each one in isolation.
This pass answers the prior question — where does a whole training step spend its
GPU time — because the answer decides which of those kernel numbers matter. The
instrument already existed: `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` writes a GPU
timestamp delta around every dispatch. [rank_kernel_profile.py](hierarchos-vulkan/validation/rank_kernel_profile.py)
turns that stream into a per-step ranking and reports µs/dispatch — the column that
distinguishes arithmetic from per-launch cost. Selecting the submissions to rank is
the one subtlety: a batch-2 run logs 1733 submissions, of which 1693 dispatch
nothing at all, 37 carry only warmup work (4–64 dispatches), and **3 carry a
complete 6912-dispatch step**. That is what the tool's `--min-dispatches 512`
threshold is for, and the category lines of each selected submission sum to exactly
its `gpu_ms`, which checks the selection.

One step of `training_step_probe --batch 2 --tokens 8` is 6912 dispatches and
88.5 ms of GPU time. Full tables for both batch sizes are in
[kernel_ranking.txt](hierarchos-vulkan/validation/training_step_probe/kernel_ranking.txt).

| category | dispatches/step | GPU ms/step | share | µs/dispatch |
| --- | --- | --- | --- | --- |
| normalization.layer-norm | 536 | 37.690 | 42.6% | 70.3 |
| rwkv.recurrent | 1152 | 21.689 | 24.5% | 18.8 |
| projection.linear | 1739 | 17.027 | 19.2% | 9.8 |
| control | 432 | 5.056 | 5.7% | 11.7 |
| optimizer.gradient-bookkeeping | 1346 | 3.118 | 3.5% | 2.3 |
| loss.lm-projection | 24 | 1.671 | 1.9% | 69.6 |
| elementwise-reduction | 1184 | 1.065 | 1.2% | 0.9 |
| other | 164 | 0.504 | 0.6% | 3.1 |
| token.embedding | 192 | 0.325 | 0.4% | 1.7 |
| optimizer.adamw | 95 | 0.227 | 0.3% | 2.4 |
| memory.rosa | 40 | 0.167 | 0.2% | 4.2 |
| loss.cross-entropy | 8 | 0.008 | 0.0% | 1.0 |
| **total** | **6912** | **88.548** | **100.0%** | **12.8** |

| shader | dispatches/step | GPU ms/step | share | µs/dispatch |
| --- | --- | --- | --- | --- |
| layer_norm_low_rank_producer_forward_fused | 144 | 25.201 | 28.5% | 175.0 |
| packed_cell_channel_mix_state_forward_fused | 144 | 9.757 | 11.0% | 67.8 |
| layer_norm_adapter_forward_fused_512 | 144 | 5.583 | 6.3% | 38.8 |
| linear_bias_forward | 192 | 4.703 | 5.3% | 24.5 |
| linear_input_grad | 385 | 4.095 | 4.6% | 10.6 |
| rwkv_time_mix3_linear3_key_state_forward_packed_fast | 96 | 3.741 | 4.2% | 39.0 |
| rwkv_group_norm_bonus_gate_linear_residual_forward_fused | 144 | 3.195 | 3.6% | 22.2 |
| gradient_accumulate4 | 768 | 2.770 | 3.1% | 3.6 |
| layer_norm_input_grad | 64 | 2.537 | 2.9% | 39.6 |
| drift_update_forward | 48 | 2.522 | 2.8% | 52.5 |

Both tables are abridged. The 10 shader rows above cover 72.4% of the step, and all
25 shaders the runtime printed for these submissions cover 93.8%; the remaining gap
is a property of the instrument rather than of the step, because the runtime prints
only the top 24 shaders per submission (`KERNEL_PROFILE_TOP_SHADERS`). Read this as
a ranking of hot spots, not as an account of every microsecond.

## Four findings

**1. One kernel is 28.5% of the step, and it is not one we had optimized.**
`layer_norm_low_rank_producer_forward_fused` is 144 dispatches at 25.20 ms/step,
175.0 µs each. The step's per-dispatch average is 12.8 µs, and the kernels that
carry its dispatch *volume* are cheaper still — `elementwise-reduction` 0.9 µs per
dispatch, `gradient_accumulate4` 3.6 — so this one launch is 13.7× the step's
average and up to ~190× the cheapest. The `normalization.layer-norm` category is
42.6% of the step and this single kernel is two thirds of it. The second-largest
category, `rwkv.recurrent`, already puts 45% of itself in one kernel;
`projection.linear` spreads 19.2% over 1739 dispatches at 9.8 µs each.

**2. The step's GPU cost is nearly independent of the token count.** Going from
batch 2 to batch 16 with the same 8 tokens per sequence is 8× the rows, and it
costs 88.5 → 97.0 ms of GPU per step: +8.4 ms, +9.5%. Per dispatch, the top kernel
gets *cheaper* (175.0 → 148.5 µs), and so do `linear_bias_forward` (24.5 → 19.0),
`linear_input_grad` (10.6 → 8.7) and the whole `projection.linear` category
(17.0 → 13.8 ms). Only the recurrent state work and the token-dependent elementwise
work grow (`rwkv.recurrent` 21.7 → 31.0 ms, +43%). A step is therefore dominated by
per-launch constants rather than by token-proportional arithmetic, which is the
measured explanation for pass 2's modest 2.8% end-to-end win from kernel-level
changes that were 2.7–11.8× in isolation.

**3. Attention is not on this path at all.** No `transformer_attention_*` kernel
appears in either ranking, at either batch size. Pass 2 measured a real regression
in the attention backward artifact, but the canonical raw-token trainer does not
dispatch that kernel, so the regression is not a training-step cost. "Highest-value
target by a wide margin" was true within the attention benchmark and is corrected
here: within the trainer, the target is the fused low-rank LayerNorm producer.

**4. The hot kernel's cost is a fixed one, and its geometry explains that better
than its arithmetic does.** In [rwkv_low_rank.rs](hierarchos-vulkan/src/rwkv_low_rank.rs#L2420)
the dispatch is `[batch.div_ceil(rows_per_workgroup), 1, 1]` with 64 lanes, and
`rows_per_workgroup` is 1 whenever the fp16 parameter mirrors are absent, which is
the case on this path — so a launch exposes exactly `rows` workgroups. The shader
body's per-lane work is on the order of `width` guarded FMAs plus one barrier per
64-column tile, which cannot cost 150–175 µs. What the measurement *proves* is that
the cost does not scale with rows at all; it does not yet say which mechanism makes
it fixed. Two candidates are worth one experiment each: only `rows` workgroups are
exposed so latency is unhidden — the same signature pass 2 found in the attention
backward kernel, whose cost was also flat in the workgroup count — and/or the
barrier plus cache-invalidation cost of its ~32 bindings, since
`HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1` shows it producing the step's largest
hazard edge (288 raw edges into `rwkv_time_mix3_linear3_key_state_forward_packed_fast`).
Raising `rows_per_workgroup` would test both at once, and the fp16 path already has
that machinery; it would also cut barriers, which ties this back to the elision
rule in pass 2.

## Method, and what it does not prove

- The timestamps bracket each dispatch (`record_kernel_timestamp_begin/end` in
  [vulkan.rs](hierarchos-vulkan/src/vulkan.rs#L6495)), so these are GPU times
  rather than host gaps. Each submission's category lines sum to exactly its
  `gpu_ms` — 100.00% in both runs — which is the internal check that the
  aggregation captures the whole submission.
- These runs use the debug binary (210.7 ms/step wall at batch 2), so the GPU
  attribution holds but the wall-clock numbers are not comparable with pass 2's
  end-to-end measurement, which used a different build.
- Three batch-2 profiles of the same step agree — the earlier one at 90.120 ms of
  GPU per step, two from this pass at 88.186 and 88.548 — and the top kernel is
  25.132, 25.235 and 25.201 ms in them. The ranking is reproducible, not noise.
- The probe's model package is reused across runs and its weights advance every
  run, so `mean_loss` is not comparable between runs. The kernel profile is
  unaffected.
- One device, one fixture: this ranks the canonical trainer's small-model step,
  and finding 2 is the reason not to extrapolate the *proportions* to other shapes
  even though the per-launch constants will travel.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 ./target/debug/training_step_probe --batch 2  --tokens 8 --steps 2 --warmup 1 2> target/probe_b2.log
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 ./target/debug/training_step_probe --batch 16 --tokens 8 --steps 2 --warmup 1 2> target/probe_b16.log
python validation/rank_kernel_profile.py target/probe_b2.log  --top 24
python validation/rank_kernel_profile.py target/probe_b16.log --top 24
HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1 ./target/debug/training_step_probe --steps 1 --warmup 0 2> target/trace_probe.log
```

The raw profile logs stay in `target/`, which is ignored; the ranking itself is
recorded as [validation/training_step_probe/kernel_ranking.txt](hierarchos-vulkan/validation/training_step_probe/kernel_ranking.txt).

---

# Fourth pass — 2026-10-15: the ladder as one command, and golden mode as a gate

The checks in the first three passes were run by hand, in an order that matters,
because each one constrains a different layer and a change can satisfy one while
breaking another. [verify_training_stack.py](hierarchos-vulkan/validation/verify_training_stack.py)
runs the whole ladder in one pass and exits non-zero unless every step passed:

```
PASS     cargo build (ladder binaries)  finished (debug)
PASS     binding-access audit           0 provably narrower binding(s) over 0 shader(s) out of 467 modules
PASS     shader provenance              32 variant(s) match their sources and flags
PASS     golden dense (matmul)          PASS: golden digests verified for 144 invocation(s) on AMD Radeon Graphics (0 unverified on another device)
PASS     golden reduction               PASS: golden digests verified for 183 invocation(s) on AMD Radeon Graphics (0 unverified on another device)
PASS     golden attention               PASS: golden digests verified for 79 invocation(s) on AMD Radeon Graphics (0 unverified on another device)
PASS     training-step probe            6912 dispatches, 3076 barriers (upper bound 3076), loss 5.543217 on AMD Radeon Graphics

7/7 step(s) passed
```

Two design choices are worth stating, because both exist to stop the ladder from
passing vacuously:

- A step that cannot run — missing toolchain, missing binary, no recorded digest or
expectation for this device — is reported `BLOCKED` and fails the run. Green means
verified. The `--write-golden` and `--golden` modes could otherwise report an
unrecorded device as success, and `--golden-strict` is what turns that into a
failure; the ladder always passes `--golden-strict`.
- The ladder builds exactly the four binaries it runs, not `cargo build --bins`.
Building every bin makes an unrelated harness that does not compile fail a run
about the training stack, which is a real failure mode in a checkout where several
worktrees are being edited at once.

## The probe is now a repeatable end-to-end check

The probe previously reused one model package, and since every run performs AdamW
updates, each run measured a different point in training — so its loss could not be
asserted. `--model-dir` now initializes a directory that does not already hold a
package ([training_step_probe.rs](hierarchos-vulkan/src/bin/training_step_probe.rs#L133)),
so a caller can point it at a scratch directory it deletes first. Two such runs
give bit-identical results: 6912 dispatches, 3076 barriers, loss **5.543217** both
times. The ladder already deletes and re-creates `target/verify_training_stack/fresh_fixture`
for every run, and compares against
[step_expectation.json](hierarchos-vulkan/validation/training_step_probe/step_expectation.json):
dispatch count must match, the barrier count is an upper bound (so the elision work
below cannot fail the ladder by improving), and the loss is compared bit-for-bit
per device name. `--record-expectation` adds that device's entry and leaves every
other device's entry as recorded. `dispatch_count` is recorded per device too now,
for the same reason the loss is: the vendor layer dispatches the fixture a
different number of times on different parts (see the 2026-10-15 slice-point
section below).

## The gate is non-vacuous, checked six ways

| perturbation | observed |
| --- | --- |
| one digest nibble flipped in `dense.json` | 1 mismatch reported, exit 1 |
| `dense.json` re-keyed to another vendor, `--golden-strict` | exit 1, "144 invocation(s) have no digest recorded for AMD Radeon Graphics" |
| same file, loose `--golden` | exit 0 but "0 verified, 144 unverified on another device" — not a pass |
| expectation's `dispatch_count` set to 1 | exit 1, "dispatch count 6912 != recorded 1" |
| device entry removed from the expectation | exit 1, "no expectation recorded for AMD Radeon Graphics" |
| release binaries absent with `--skip-build` | 4 × BLOCKED, exit 1 |

## Recording goldens for a second GPU vendor

The expensive half of this is hardware: it cannot be recorded from here, and no
fabricated entry would be worth anything. What is verified is the mechanism that
makes it a one-command step on that machine. Recording **merges** instead of
overwriting: taking a copy of `dense.json`, re-keying it to
`Intel(R) Arc(TM) A770 Graphics`, and re-recording on this AMD machine produced a
file holding both `AMD Radeon Graphics` and `Intel(R) Arc(TM) A770 Graphics`
entries for all 144 invocations, and `--golden-strict` then passed against it. So
the procedure on a new vendor's runner is exactly:

```powershell
./target/debug/matmul_kernel_validation     --write-golden validation/kernel_golden/dense.json
./target/debug/reduction_kernel_validation  --write-golden validation/kernel_golden/reduction.json
./target/debug/attention_kernel_cache_check --write-golden validation/kernel_golden/attention.json
python validation/verify_training_stack.py --record-expectation
```

and the existing vendor's entries survive it. Until that is done, a run on a new
device is `BLOCKED` rather than green, which is the intended behaviour.

## Wiring, and what the project does not have

There is no CI configuration in the repository — no `.github/workflows`, no other
pipeline file — so the gate is the command itself:

```powershell
cd hierarchos-vulkan
python validation/verify_training_stack.py --receipt target/verify_training_stack/receipt.json
```

`--receipt` writes a machine-readable per-step summary. A runner needs `glslc` and
`spirv-val` on `PATH` (the provenance step compiles and validates the CE variants)
and a Vulkan device whose digests and expectation are recorded; both are reported
as failures rather than skipped when absent.

---

# Fifth pass — 2026-10-15: why the barrier count cannot be lowered by narrowing access

Pass 2 found that narrowing a binding to `writeonly` can never remove a barrier,
because the elision rule's second term keys on "may be written", which `writeonly`
preserves, and measured that the 41 narrowings bought exactly zero barriers. The
task here was to change the rule so that narrowed access does buy barriers. It does
not work, and this pass establishes that with a measurement instead of an argument.
The barrier count on the representative step is unchanged at **3076 per step**.

## The rule is already exactly the hazard set

[prepare_shader_dispatch](hierarchos-vulkan/src/vulkan.rs#L6663) inserts a barrier
when any bound buffer satisfies:

```rust
(access.may_read()  && self.pending_shader_writes.contains(&region))
    || (access.may_write() && self.pending_shader_buffers.contains(&region))
```

and [finish_shader_dispatch](hierarchos-vulkan/src/vulkan.rs#L6762) records every
bound region in `pending_shader_buffers` and, additionally, in the read or write set
according to its access. So `pending_shader_buffers` is precisely "read or written by
some dispatch since the last barrier", and the two terms are exactly:

- term 1 = **read-after-write**,
- term 2 = `write && (read || written)` = **write-after-read** ∪ **write-after-write**.

That is the complete set of hazards a compute-to-compute barrier can be protecting
against. There is no conservatism left to remove, which is why no declaration can
lower the count. The dependency trace over one step agrees with the shape of the
rule: 69 read-after-write edges, 37 write-after-read, 21 write-after-write — all
three classes are present, so all three terms are load-bearing.

Two further levers were ruled out by measurement rather than by reasoning:

- **Finer region granularity.** `BufferRegionKey` is already
  `(VkBuffer, offset, size)` of the buffer view, and the engine binds whole views,
  so the key is as precise as the addressing the shaders use. There is no coarser
  key to refine.
- **Unused bindings.** A dispatch that binds a region it never touches still records
  it, which would be a false hazard. `audit_shader_binding_access.py` already proves
  per-binding access from SPIR-V, and its `unused` class is empty where it matters:
  **0 of 135 bindings** across the 13 hottest shaders of the step, and only 6 across
  all 467 modules. Skipping provably-unused bindings would buy nothing here.

## The rule's one real gap is in the strict direction, and this graph does not hit it

An exact-key comparison also means two *different* views of one `VkBuffer` whose byte
ranges overlap are invisible to the rule: the hazard is real but the keys do not
match, so no barrier is inserted. That would be a latent race, and it is now
something the engine can report. `HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1` gained a
recorded-but-inert diagnostic:
[record_overlapping_view_hazards](hierarchos-vulkan/src/vulkan.rs#L6730) compares each
bound view against every region touched since the last barrier, and the submission
summary reports how many pairs it examined against how many overlapped.

On the representative step the answer is **0 overlaps out of 352 same-buffer view
pairs examined, identical across all four profiled steps**. The population counter is
the point: a bare zero would be indistinguishable from a detector that never saw a
comparable pair. The predicate is unit-tested by
`overlapping_view_detection_distinguishes_overlap_from_distinct_ranges`, which asserts
that `(buf 1, 0, 4096)` and `(buf 1, 2048, 4096)` overlap, that identical keys and
adjacent or disjoint ranges do not, and that the same range in two buffers is not a
hazard.

The diagnostic changes no decision: it records only, and the probe's 6912 dispatches,
3076 barriers and bit-identical loss are unchanged with it enabled. It exists because
the check would otherwise be an unverified assumption in the elision rule.

## What would actually reduce barriers

Since the count is not reachable from the rule, the remaining levers are graph
structure and barrier *cost*, not barrier count:

- **Fewer real hazards.** The step executes 6912 dispatches and 3076 barriers, so
  about 44% of its dispatches sit behind a new ordering point. Reducing that means
  changing buffer reuse — for example having a producer write where its predecessor's
  output is not still pending — which is graph work, not scheduler work.
- **Cheaper barriers.** `shader_barrier()` emits an unscoped global memory barrier
  (`src=SHADER_WRITE`, `dst=SHADER_READ|SHADER_WRITE`). Scoping each barrier to the
  regions that actually hazard, via buffer memory barriers, would not change the
  count but would reduce what each one costs. That is a measurement nobody has taken
  here, and it is the honest next experiment on this axis.

## Not done in this pass: the attention backward lane split

The per-key phase of `transformer_attention_backward.comp` is the largest remaining
kernel-level opportunity, and it was not implemented. Reading the kernel closely enough
to be sure of a bit-exact split turned up three obstacles that pass 2's design note did
not capture, and each one has to be solved before the change is worth attempting:

1. **The per-key phase is only independent of itself.** In the strict-adjoint branch
   (lines 565–640) the body of `for (uint key …)` reads `row_max`, `row_inverse` and
   `row_softmax_dot` from shared memory. Those are computed earlier in the same
   invocation by iterating over *all* keys — a cross-key reduction whose FP32 order is
   part of the bit-exactness contract. Splitting keys across lanes therefore cannot be
   done by widening the grid: the row statistics have to stay in one lane, in the
   original order, with a barrier before the split.
2. **Every other path in the file assumes one lane.** The kernel declares
   `local_size = 1×1×1` and handles MHA/GQA, causal and bidirectional, sinks, relative
   bias, paged KV, and three unfused-QK variants in the same entry point, all of them
   single-lane and several of them accumulating `grad_k`/`grad_v` with `+=` in query
   order. Raising `local_size_x` to get lanes puts every one of those paths behind a
   `lane == 0` guard and adds the barriers they never needed.
3. **The per-lane workspace does not fit the shader's own budget.** The accumulators
   being split (`shared_k_cache`, `shared_v_cache`, `head_k_cache`, `head_v_cache`) are
   single-lane scratch today. Per-lane copies need `lanes × max_head_dim × 2` floats,
   and the file's header explicitly keeps its working set inside Vulkan 1.0's 16 KiB
   shared-memory guarantee — so a lane count has to be chosen against
   `HEAD_CACHE_SIZE` and the oversized-row fallback kept for everything else.

None of that is a reason not to do it; it is the work the change actually involves,
and it spans the kernel plus the host dispatch and allocation. The verification gate is
already in place: the parity harness's 79 invocations (forward and backward) compare
bit-for-bit, and the same invocations are covered by golden digests, so a split that
changes any FP32 sum fails immediately rather than silently. A later pass should start
from the current artifact's occupancy — not from the narrower declaration experiment —
and treat obstacle 1 as the design constraint rather than a detail.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
# hazard-class histogram and the overlapping-view population for one step
$env:HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS=1
./target/debug/training_step_probe --steps 1 --warmup 0 2> target/trace_probe.log
Select-String -Path target/trace_probe.log -Pattern "overlapping_view_total|hazard=" | Select-Object -First 20
# per-binding proof of access, including the unused class
python validation/audit_shader_binding_access.py --json target/audit_full.json
cargo test --lib -j 1 overlapping_view_detection -- --test-threads=1
```

---

# Sixth pass — 2026-10-15: what a step costs per token, and which knob amortizes it

The third pass found the step's GPU cost is nearly flat in the tokens it carries. If
that is true, the cheapest throughput win available is not a kernel at all: it is
carrying more tokens per step so the same dispatches amortize over more work.
[sweep_step_tokens.py](hierarchos-vulkan/validation/sweep_step_tokens.py) measures that,
running `training_step_probe` from a fresh model package per configuration so every
configuration starts at the same weights.

Release build, 8 tokens per sequence, rows varying (wall and GPU milliseconds per
step, `tokens/s` = tokens per step divided by wall time):

| rows | tokens/step | wall ms | GPU ms | GPU share | tokens/s | vs 16 | dispatches | barriers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 16 | 156.06 | 87.87 | 56.3% | **102.5** | 1.00× | 6912 | 3076 |
| 8 | 64 | 163.94 | 90.30 | 55.1% | 390.4 | 3.81× | 7056 | 3172 |
| 32 | 256 | 200.28 | 108.65 | 54.2% | 1278.2 | 12.47× | 7056 | 3172 |
| 128 | 1024 | 502.24 | 351.35 | 70.0% | 2038.9 | 19.89× | 7120 | 3204 |
| 256 | 2048 | 863.20 | 638.06 | 73.9% | 2372.6 | 23.14× | 7120 | 3204 |
| 512 | 4096 | 1575.67 | 1206.72 | 76.6% | **2599.5** | **25.35×** | 7120 | 3204 |
| 1024 | 8192 | 3011.93 | 2339.60 | 77.7% | 2719.9 | 26.53× | 7120 | 3204 |

**The answer is 25.4× tokens/s, and it costs nothing but a configuration change.** 256×
the tokens per step leaves the dispatch count almost unchanged (6912 → 7120, +3%) and
the barrier count essentially unchanged (3076 → 3204). GPU time per token falls 18.6×
(5.492 → 0.295 ms per token in GPU time), which says the *GPU* side of a small step is
also dominated by per-launch overhead rather than by arithmetic — the same effect the
third pass measured as a flat cost in rows.

The curve flattens past 2048 tokens per step: doubling again to 8192 buys 4.6%, because
the GPU share has climbed from 56% to 78% and the step is now doing real work per
token. 4096 tokens per step (512 rows) is the practical point — within 4.6% of the
8192-token result for half the memory and step latency.

## The other knob is the wrong one

Tokens per step can also grow by lengthening sequences instead of adding rows, and that
is 2.4× worse for the same tokens:

| configuration | tokens/step | wall ms | dispatches | barriers | tokens/s |
| --- | --- | --- | --- | --- | --- |
| 32 rows × 8 tokens | 256 | 200.28 | 7056 | 3172 | 1278.2 |
| 8 rows × 32 tokens | 256 | 480.87 | 27642 | 12673 | 532.4 |

The reason is structural: the step dispatches roughly per sequence position, so
sequence length multiplies the dispatch count (7056 → 27642, 3.9×) and therefore the
per-launch overhead, while rows add work inside dispatches that already exist. The
knee in the row curve is the point where per-token GPU work overtakes that fixed cost.
The practical rule this measurement gives: **raise the step's row count to add tokens,
and keep the sequence length moderate.**

That lands on a real trainer flag rather than an abstract knob. `train.rs` exposes
`--batch-size` (rows carried in one graph — the amortizing knob) and
`--gradient-accumulation-steps` (more graphs before one optimizer update — which
multiplies dispatches instead), and its `--batch-size` default was 1. At 1 row per step
the trainer sits left of this table entirely, where the per-launch cost is the whole
step, so the default is now **32** and the `--help` text carries the measured curve
above. `--gradient-accumulation-steps` is deliberately left at 1: it replays the whole
graph per micro-batch and pays the same per-dispatch price, so it cannot amortize
anything.

Measured through the trainer itself, not the probe — 384 dataset rows of 32 tokens
(12 288 tokens), one epoch, release, one AMD Radeon, per-step `tokens_per_sec` medians
taken from the `HIERARCHOS_EVENT` stream:

| `--batch-size` | steps | wall s | median tokens/s | first step | last step |
| --- | --- | --- | --- | --- | --- |
| 1 | 384 | 129.8 | **113.9** | 75.8 | 115.7 |
| 32 (new default) | 12 | 6.4 | **2430.8** | 1752.7 | 2506.6 |

Same tokens, same epochs: **21.3× the per-step token rate, 20.3× the wall clock**. The
first step at 32 rows is 28% below the last (graph setup and the first dispatches are
not amortized yet), which is the shape the probe table predicts.

Two things this does not fix. A larger batch is a different optimizer problem — fewer,
fatter updates — so the learning rate that suited one row per step is not the learning
rate that suits 32; `--batch-size 1` still exists for debugging and for anyone whose
schedule assumes it. And the default change is a training-semantics change as much as a
throughput one; it is recorded here rather than presented as free.

## One incidental result worth keeping

The release build reproduces the debug-recorded probe expectation exactly — 6912
dispatches, 3076 barriers, loss **5.543217** — so the ladder's recorded expectation,
and the bit-exactness it asserts, are build-flavor independent. The ladder can be run
with `--release` and compare against the same numbers.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe
python validation/sweep_step_tokens.py --release --batches 2,8,32,128,512 --tokens 8 --steps 5 --warmup 1
python validation/sweep_step_tokens.py --release --batches 8,32 --tokens 32 --steps 3 --warmup 1
python validation/rank_kernel_profile.py target/token_sweep/profile_b512_t8_s1.log
```

# Seventh pass — 2026-10-15: the producer kernel, and the pass that says no

The third pass named `layer_norm_low_rank_producer_forward_fused` as the step's top
cost at 175 µs per dispatch. This pass tried to make it cheap, and the honest result
is that it is now 5% cheap: **132.24 µs per dispatch instead of 139.35 — not the 175 µs
that was asked for, and not a fix.** What follows is the measurement that changed my
mind about why the number is what it is.

The same kernel, same 144 dispatches per step, at three geometries (release, 8 tokens
per sequence, one sequence):

| rows | producer ms | µs/dispatch | step GPU ms | producer share of step |
| --- | --- | --- | --- | --- |
| 2 (third pass) | 25.35 | 175.0 | 88.5 | 28.6% |
| 32 | 20.07 | 139.35 | 106.83 | 18.8% |
| 512 | 27.68 | 192.25 | 1203.38 | **2.3%** |

**The cost is nearly flat in the rows it processes.** 16× the rows (32 → 512) costs
1.38×, while in the same two runs every control kernel scales with rows as it should:
`packed_cell_channel_mix_state_forward_fused` 79.16 → 854.93 µs (10.8×),
`vector_add` 2.91 → 60.87 (20.9×), `linear_weight_grad` 6.00 → 48.61 (8.1×),
`layer_norm_adapter_forward_fused_512` 56.15 → 200.98 (3.6×). A kernel that costs the
same for 2 rows as for 512 is not paying for its arithmetic, and the third pass's
mechanism — a fixed per-launch cost — is confirmed while the third pass's *target* is
not.

## What was changed, and what it bought

The one serial structure in the kernel that is specific to it rather than to the
launch path was the LayerNorm reduction: lane 0 added the whole row to itself, one
global load per addition, while the other 63 lanes waited at the barrier. It now stages
the row through shared memory in 64-float tiles (one coalesced load per lane) and lane 0
adds the same values in the same order out of shared memory, so the mean, the
variance and the fused outputs are unchanged. Measured: **139.35 → 132.24 µs per
dispatch, 20.07 → 19.04 ms per step**, with the control kernels moving less than 0.5%
(`packed_cell` 79.16 → 79.19, `vector_add` 2.91 → 2.89), so the 5% is above this
machine's noise floor and real. The change is kept because it is bit-exact and
monotone, not because it answers the question.

## Why the remaining 132 µs is not this kernel's arithmetic

The hypothesis that the loads themselves were unpipelined is wrong: if a dependent
global-load chain of `width` round trips were sitting on one lane, removing it would
have bought tens of microseconds, and it bought 7. The compiler already hoists
independent loads. Three further facts point away from arithmetic as the explanation:

* The cost is a property of the *shape class*, not of this kernel. At 32 rows the
  same-geometry kernels in the same category — layer-norm backward/add-residual,
  adapter forward, packed cell channel mix — all sit in a 47–79 µs per-dispatch band,
  while kernels dispatched over many narrow workgroups (`vector_add`
  `linear_weight_grad`, `gradient_accumulate4`, `bias_grad`) sit at 3–6 µs. The
  producer is the widest member of the first class, not an outlier in it.
* At a small row count the grid is tiny — 32 workgroups of 64 lanes — so there is
  almost nothing resident to overlap memory latency with, and the 512-row run recovers
  16× the work for 1.38× the time, which is what latency-hiding looks like when it
  starts working.
* The sub-dispatch attribution cannot be separated from the neighbouring dispatches
  with the instrumentation this project has. The runtime's per-category milliseconds do
  sum to the submission's `gpu_ms`, so the *total* is real and the producer's share of
  it is 2.3% of the recommended step; but among adjacent timestamp brackets, some of a
  wide kernel's bracket can belong to the work it inherits across the preceding
  barrier. Resolving that needs per-dispatch counters, and there is no `rocprof` or
  Nsight on this machine.

So the requested outcome — "stop costing 175 µs per dispatch" — was not achieved. I
can say what the number is (139, not 175, at the trainer's default batch, and 2.3% of a
step rather than 29%), what it is not (this kernel's arithmetic), and what would settle
it (counters, or a profile that separates brackets from neighbours). The rest is the
negative result, reported as one.

## Verifying a bit-exact kernel change

`glslc` with its default target environment reproduces the embedded `.spv`
byte-for-byte from the source, before and after the edit
(`python validation/audit_shader_artifacts.py layer_norm_low_rank_producer_forward_fused`),
`spirv-val --target-env vulkan1.0` accepts the result, and the ladder is unaffected in
the ways that matter: **7/7, loss 5.543217, 6912 dispatches, 3076 barriers**, plus the
three golden harnesses at 144/183/79 invocations. The 32-row geometry used for the
timing comparison reports loss 5.3915977 with the same 7056 dispatches and 3172
barriers, bit-identical to the values recorded before the edit (compared on the full
recorded value, not on the table's 4-decimal print). `cargo test --lib -j 1` is 705 passed / 0 failed / 9 ignored, unchanged.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe --bin hierarchos-vulkan-train
glslc shaders/layer_norm_low_rank_producer_forward_fused.comp -o shaders/layer_norm_low_rank_producer_forward_fused.spv
spirv-val --target-env vulkan1.0 shaders/layer_norm_low_rank_producer_forward_fused.spv
python validation/audit_shader_artifacts.py layer_norm_low_rank_producer_forward_fused
python validation/sweep_step_tokens.py --release --batches 32,512 --tokens 8 --steps 5 --warmup 1
python validation/rank_kernel_profile.py target/token_sweep/profile_b32_t8_s1.log
python validation/verify_training_stack.py
# trainer-level before/after, 384 rows of 32 tokens each
./target/release/hierarchos-vulkan-train.exe --model target/train_fixture/model --dataset target/train_fixture/tokens384.jsonl --output target/train_fixture/out_b1 --batch-size 1 --epochs 1 --json-events
./target/release/hierarchos-vulkan-train.exe --model target/train_fixture/model --dataset target/train_fixture/tokens384.jsonl --output target/train_fixture/out_bdefault --epochs 1 --json-events
```

# Eighth pass — 2026-10-15: what a dispatch bracket leaves out, and where the host time goes

The third pass ranked kernels by `gpu_ms`, and that number is the *sum of per-dispatch
timestamps*: a bracket is written immediately before and immediately after each
`vkCmdDispatch`. So `gpu_ms` is execution and nothing else — which is the good news —
but it also means a step can be slower than `gpu_ms` for reasons no kernel row shows.
Nothing in the profile bounded the submission, so the unattributed part had no name and
silently belonged to whichever kernel sat next to it. Two additions fix that.

## The span, and the share no kernel owns

`hierarchos_vulkan_kernel_profile_span` now reports the submission's first bracket begin,
its last bracket end, and the difference. Release, 8 tokens per sequence, one sequence,
four submissions averaged:

| rows | dispatches | GPU span ms | dispatch brackets ms | unattributed ms | share | wall ms | host = wall − span |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 6912 | 100.369 | 88.639 | 11.730 | 11.7% | 157.10 | 56.7 ms (8.2 µs/dispatch) |
| 32 | 7056 | 127.076 | 107.594 | 19.482 | 15.3% | 200.36 | 73.3 ms (10.4 µs/dispatch) |
| 512 | 7120 | 1335.095 | 1208.929 | 126.166 | 9.4% | ~1576 | ~241 ms (readback-dominated) |

**9–15% of a step's GPU span is inside no kernel bracket**: barrier flush, launch
latency, and idle between dispatches. At 32 rows that is 19.5 ms of a 127 ms span. It is
the bucket that made the third pass's flat-per-launch cost look like a kernel property,
and it is not the same lever as the dispatch count: removing dispatches removes brackets
and their host cost, not the gap between the remaining ones.

The span also separates host time for the first time. Wall minus span at 32 rows is
73.3 ms over 7056 dispatches — **10.4 µs per recorded dispatch** (8.2 µs at 2 rows).
At 512 rows the same subtraction gives ~241 ms, but that figure is not comparable: the
probe's `--readback full` copies back tensors proportional to the batch, so it measures
readback I/O as well.

**That 10.4 µs is an upper bound, and the profiler is the reason.** Profiling writes two
timestamps per dispatch — 14112 extra host calls per step — so the same 32-row
configuration runs in **115.6 ms unprofiled against 200.4 ms profiled**, and the
profiled wall-minus-span includes the profiler's own recording. The genuine per-dispatch
host cost is therefore single-digit microseconds and, with the instrument that can see
it also perturbing it, it is not pinned down here. This also corrects the sixth pass:
its absolute tokens/s column was measured with profiling on (`sweep_step_tokens.py`
always enables it), so those figures are lower bounds. The *ratios* in that table survive
because the overhead tracks the dispatch count, which barely moves with rows.

The step result also exposes the recording counters, so they are now in the probe's
report rather than in a comment. At 32 rows, per 7056 dispatches: **216 descriptor sets
allocated, 2352 descriptor binds, 1902 pipeline binds**. Binding work is therefore *not*
per-dispatch — the descriptor cache absorbs 96.9% of it, and only a third of dispatches
rebind at all. Whatever the ~10 µs is spent on, it is the recording calls themselves, not
descriptor churn, which removes the second reason to expect a single gradient slab to pay
for itself.

## Cold and warm: does a kernel pay for itself?

`hierarchos_vulkan_kernel_profile_placement` splits each shader's dispatches by whether
they were recorded directly after a barrier or back to back with another dispatch. Same
step, 32 rows, per dispatch:

| shader | after barrier | after dispatch | difference |
| --- | --- | --- | --- |
| `layer_norm_low_rank_producer_forward_fused` | 154.3 µs | **123.7 µs** | +30.6 µs |
| `layer_norm_adapter_forward_fused_512` | 73.8 µs | **27.7 µs** | +46.1 µs |
| `linear_bias_forward` | 20.7 µs | 4.7 µs | +15.9 µs |
| `row_keep_forward` | 14.4 µs | 0.005 µs | +14.4 µs |
| `layer_norm_stats` | 26.0 µs | 17.3 µs | +8.8 µs |
| `indexed_step_gather` | 2.6 µs | 13.8 µs | **−11.2 µs** |

This answers the question the seventh pass had to leave open, and it partly corrects
that pass's conclusion. The producer is mostly paying for its own work: 123.7 µs of its
133.9 µs average is execution, and only ~31 µs of a cold dispatch is inherited. So the
third pass's ranking of it as a hot spot was not a measurement artifact, and the seventh
pass's 5% was a real 5% of a real cost. Where the seventh pass went too far was in
concluding that the residual "is not this kernel's arithmetic": once the inherited share
is subtracted, the remaining 123.7 µs *is* time the kernel spends executing, just spent
in a latency-bound structure rather than in useful flops — which means a shader-level fix
remains possible in principle and the 5% was a small win in the right direction, not a
wrong target. By contrast `layer_norm_adapter_forward_fused_512` looks 2.7× more
expensive than it is: 63% of its cold bracket is the barrier in front of it.

Two caveats, both visible in the table. `row_keep_forward` shows the effect at its
purest — 14.4 µs cold against 5 ns warm, the same kernel and the same grid, so that
kernel's cost is essentially all inherited — while `indexed_step_gather` goes the other
way, because which dispatches land cold correlates with *what* they process. The split
is an observation about placement, not a controlled experiment, and the tool says so.
And a small warm measurement is not proof of free work: 5 ns of bracket is a dispatch
whose grid is a handful of workgroups that the hardware retires between two timestamp
writes. One more precision, since it is a label rather than a measurement: the first
dispatch of a submission follows neither a barrier nor another dispatch, and it is
counted as warm. That is one dispatch per submission out of ~7000, and its bracket can
include the previous submission's tail, so the cold/warm split is unaffected at this
scale — but the label is approximate for that dispatch.

## Why the dispatch count cannot be cut yet

The eighth pass's own numbers size the prize: **8–10 µs of host time per recorded
dispatch**, so the ~1346 per-slot dispatches in `optimizer.gradient-bookkeeping` (one per
optimizer slot, in `record_scale_gradients_*`, `record_accumulate*`, `record_step`) cost
roughly 13 ms of the 200 ms step at 32 rows. They cannot be folded into one dispatch as
the buffers stand: every slot owns its own allocation — `GpuBuffer::zeros_f32(&device,
trainable.len)` per slot — and Vulkan storage bindings cannot be indexed without
descriptor-indexing support, so one dispatch physically cannot touch N slots' gradients.
Cutting the count honestly needs one gradient slab with per-slot sub-ranges (the
barrier-elision rule already keys on `(buffer, offset, size)`, so sub-ranges stay
distinct) plus an index table and a `*_many` kernel per operation. That is an allocation
and checkpoint-layout change, not a batching tweak, and it was not attempted here.

The same reasoning bounds the other route — fusing equal-shape elementwise kernels
(`finite_clamp_forward` 288, `vector_add` 308, `bias_grad` 240 dispatches per step) into
one dispatch per group — and the `dist` trees under `hierarchos-vulkan/` show this
project already treats layout changes as a deliberate migration. What is *verified* here
is the measurement, so that whoever does that work can show the single-digit
microseconds per dispatch it removes — against a step that is 115.6 ms unprofiled, not
the 200 ms the profiled runs show.

## Verified

Ladder 7/7 with loss **5.543217**, 6912 dispatches and 3076 barriers unchanged;
`cargo test --lib -j 1` 705 passed / 0 failed / 9 ignored. The profiler is inactive
unless `HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, so the ladder and the golden harnesses run
the same code they ran before. `rank_kernel_profile.py` reads the two new lines and
`validation/training_step_probe/kernel_ranking.txt` is regenerated from a 512-row
profile.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 ./target/release/training_step_probe.exe --model-dir target/prof_check/m32 --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/b32.json 2> target/prof_check/b32.log
python validation/rank_kernel_profile.py target/prof_check/b32.log --top 8
grep _span target/prof_check/b32.log
grep _placement target/prof_check/b32.log
```

# Ninth pass — 2026-10-15: the profiler was measuring itself too

The eighth pass ended with a caveat it could not fix: profiling adds ~82% to the wall
clock, because two timestamps per dispatch is 14112 extra host calls in a step, and every
number it produced came from a run distorted by the instrument. This pass samples the
brackets and re-measures what that distortion was hiding.

## Sampling, and why it is per shader

Every `sample_stride`-th dispatch *of each shader* is timestamped, and a shader's first
dispatch always is. Sampling per shader rather than globally is what keeps the ranking
intact: a global stride would leave kernels that appear fewer times than the stride with
no bracket at all, and they would silently vanish from the hot-spot table. The default
stride is **17**, prime so the sample does not phase-lock with a per-layer dispatch
pattern, and `HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1` restores exact timing. The
scheduler-driven collection stays at stride 1 by construction: the trainer consumes that
profile's totals when it decides how to schedule, so that path must not become an
estimate.

Same 32-row step, same machine:

| profiler | step ms | overhead |
| --- | --- | --- |
| stride 1 (previous behaviour) | 202.49 | +82% |
| stride 17 (new default) | 119.82 | **+7.8%** |
| off | 111.11 | — |

**The instrument now costs 7.8% instead of 82%**, and the sampled run is 448 dispatches
of 7056 timestamped. A bracket is therefore a mean; each shader's reported `gpu_ms` is
that mean scaled by its exact dispatch count, and the `samples` column says how many
brackets the estimate rests on (the producer: 9 of 144).

## What the distortion was hiding

The span was the surprise. It is now written by its own query pair so that sampling
cannot move it, and on the same 32-row step it falls from **129.0 ms at stride 1 to
88.8 ms at stride 17** — the profiler's own timestamp writes were inside the span it
reported, so a fifth of what the eighth pass called "time no kernel owns" was the
profiler writing timestamps. Averaged over four submissions at 512 rows the unattributed
share is **9.4% at stride 1 against 5.1% at stride 17**, because the extra captures the
profiler without inflating the work between them. The barriers are real; they were just
being measured through an instrument that added more of its own serialization than they
cost.

> **Superseded by the tenth pass.** Those two unattributed figures do not reproduce:
> on the same configuration the share reads 13.8% at batch 32 and 4.2% at batch 512 at
> stride 17, and 15.3% / 9.5% at stride 1. The reason is that the share is an exact span
> minus an *estimated* bracket sum, so it inherits the sampling error of the estimate.
> The tenth pass re-measures it and states the range instead of a point value; the span
> itself, and the mechanism described here, are unchanged.

The per-kernel totals move the same way and the *ranking does not*: the producer reads
**133.9 µs per dispatch at stride 1 and 109.7 µs at stride 17**, while its share of the
step is 17.8% against 17.9%. So every absolute microsecond figure in the third, seventh
and eighth passes is an overestimate from a perturbed run, and their shares are the part
that survives.

## The placement comparison is now shape-matched

The eighth pass's cold/warm split compared dispatches that happened to sit in different
places, and flagged that cold dispatches could simply be different shapes. A placement
line is now keyed by the dispatch's shape — its grid, push constants and binding count —
and both populations print on one line under that key, so a cold and a warm bracket are
only ever compared when they ran the same work. The grid is printed so the match is
visible rather than asserted:

| shader | grid | after barrier | after dispatch | difference | cold n | warm n |
| --- | --- | --- | --- | --- | --- | --- |
| `layer_norm_low_rank_producer_forward_fused` | 32x1x1 | 129.7 µs | 99.6 µs | +30.1 µs | 3 | 6 |
| `layer_norm_adapter_forward_fused_512` | 1x32x1 | 58.7 µs | 20.5 µs | +38.2 µs | 6 | 3 |
| `linear_bias_forward` | 1x2x1 | 11.0 µs | 2.6 µs | +8.5 µs | 3 | 3 |
| `bias_grad` | 1x1x1 | 8.7 µs | 0.3 µs | +8.4 µs | 9 | 5 |
| `linear_forward` | 8x2x1 | 8.3 µs | 10.2 µs | −1.9 µs | 5 | 1 |

Both numbers for the producer fall relative to the eighth pass (154.3/123.7 µs) because
those brackets carried the profiler's own timestamps, but the **+30 µs inherited** and
the 80/20 split between own work and barrier are unchanged. `linear_forward`'s −1.9 µs
is what the confound looks like when it is gone: with matched shapes the remaining
difference is within the noise of 5 cold and 1 warm sample, and the counts are printed
exactly so that can be judged instead of guessed. A small sample is not proof of free
work in either direction.

## Limitations of this pass

The shape key removes the confound it targets and no more. It covers the grid, the push
constants and the binding *count*, not which buffers were bound, so a matched cold/warm
pair can still differ in the tensors it touches — a dispatch that reads a buffer written
moments earlier differs from one reading a cold allocation in cache and TLB state, and no
shape signature sees that. Matching also removes a confound by *construction*: two
dispatches only stay on one line because the graph happened to record them at the same
shape, so the surviving cold population is still selected by the graph's hazards rather
than assigned by the experimenter. A genuinely controlled version would force placement:
dispatch the same call site both after a barrier and back to back, in both orders, and
compare. That is a one-off harness against a single kernel, not something the per-step
profile can do, and it is the test to run before trusting a residual of a few
microseconds.

Sampled totals are estimates: means scaled by exact counts, valid when the stride does
not correlate with the cost, which is why the stride is prime and the sample sizes are
reported. The residual 7.8% is not zero — the shape hash allocates a small buffer per
dispatch (the shape is hashed for every dispatch while profiling is on, not only for the
ones that are sampled, because the sampling decision is made after it), the span pair is
written per submission, and the sampling decision costs a hash lookup per dispatch. Junk
or non-positive values of `HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE` fall back to 17 rather
than disabling or exacting the profiler: verified for `0`, `junk` and the empty string
(all resolve to 17) against an explicit `1` (all 7056 dispatches bracketed). The first dispatch of a submission is still labelled warm although
it follows neither a barrier nor another dispatch. And the new report paths have no unit
tests; they were verified against logs with both line formats, including one written
before either line existed.

## Verified

Ladder 7/7 with loss **5.543217**, 6912 dispatches, 3076 barriers, goldens 144/183/79;
`cargo test --lib -j 1` 705 passed / 0 failed / 9 ignored. Profiling stays inactive unless
`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, so the ladder exercises the untouched path.
`rank_kernel_profile.py` reads both the old and the new line formats and
`validation/training_step_probe/kernel_ranking.txt` is regenerated from a sampled
512-row profile. The stride override was exercised across `1`, `0`, `junk` and empty
(see the limitations above) after the last code edit.

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe
$env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "1"; ./target/release/training_step_probe.exe --model-dir target/prof_check/s1 --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/s1.json 2> target/prof_check/s1.log
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "17"; ./target/release/training_step_probe.exe --model-dir target/prof_check/s17 --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/s17.json 2> target/prof_check/s17.log
Remove-Item Env:HIERARCHOS_VULKAN_PROFILE_KERNELS; ./target/release/training_step_probe.exe --model-dir target/prof_check/none --batch 32 --tokens 8 --steps 3 --warmup 1 > target/prof_check/none.json
python validation/rank_kernel_profile.py target/prof_check/s17.log --top 6
```

# Tenth pass — 2026-10-15: the whole distribution re-ranked, and placement assigned
rather than observed

Two things were left over. The ranking was built from a table the runtime truncates to
the top 24 shaders, so the long tail of cheap kernels — the part whose *dispatch count*
decides whether it is worth batching — had never been ranked at all. And the cold/warm
split was still an observation: the ninth pass could match two brackets by shape, but
*which* dispatches landed cold was still the graph's decision, so the comparison could
not separate the kernel's response to placement from the work that happened to be
placed there. This pass fixes the first by making the table complete and the second by
taking placement away from the graph.

## Every kernel, not the top 24

`HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS` now bounds both the per-shader and the placement
tables (the placement loop had its own hard-coded copy of the same constant). The default
stays 24 so a diagnostic run's stderr stays short; a ranking run sets it high. The 512-row
table goes from 12 printed shaders to 92, and a per-submission table from 24 to ~103.

## The re-rank

Same synthetic step, default stride 17, complete table. Category shares of the GPU step
(the step total is 61.716 ms at batch 2, 85.054 ms at 32, 1284.871 ms at 512):

| category | batch 2 | batch 32 | batch 512 |
| --- | --- | --- | --- |
| rwkv.recurrent | 24.73% | 36.89% | 77.60% |
| normalization.layer-norm | 43.82% | 34.04% | 7.73% |
| projection.linear | 17.08% | 11.77% | 4.61% |
| control | 4.95% | 4.69% | 2.18% |
| elementwise-reduction | 1.22% | 4.31% | 5.15% |
| optimizer.gradient-bookkeeping | 4.54% | 3.70% | 0.38% |
| loss.lm-projection | 1.97% | 1.70% | 0.71% |
| token.embedding | 0.42% | 1.65% | 0.62% |
| other | 0.88% | 0.95% | 0.96% |

(Stride-17 tables throughout. The exact-bracket stride-1 run at batch 512 moves the
leaders by less than half a point — rwkv.recurrent 77.89%, layer-norm 7.44% — because at
that batch size the instrument barely moves the span.)

So there is no single priority order, and the earlier passes' lists were all read at one
batch size. At the trainer's default batch the leader is
`layer_norm_low_rank_producer_forward_fused` (144 dispatches, 15.459 ms/step, 18.18%),
followed by `packed_cell_channel_mix_state_forward_fused` (10.99%),
`rwkv_time_mix3_linear3_key_state_forward_packed_fast` (8.55%) and
`layer_norm_adapter_forward_fused_512` (7.56%) — the top four are 45.28% of the step. At
batch 512 the same four kernels are reordered and reweighted under the packed recurrent
kernels: 26.38 / 22.09 / 10.31 / 7.90%, top four 66.68%. At batch 2 the producer alone is
30.01%. The rebuilt list, with the lever and the measured basis for each entry, is
`validation/training_step_probe/kernel_priority.txt`; the generated tables for all three
regimes are `kernel_ranking_b2/b32/b512.txt`.

**A stride-5 run confirms the order at batch 32**: 8 steps, 29 samples per 144-dispatch
shader, same four kernels, same order, every level within 4% (17.52 / 10.79 / 8.07 /
7.38%). The batch-32 ordering is not a sampling artifact.

## What stride 1 actually measures

Running the same batch-32 step at three strides separates sampling error from instrument
effect, and the answer is that stride 1 is not a noisy version of the truth — it is a
different physical regime:

| kernel (batch 32) | stride 17 | stride 5 | stride 1 |
| --- | --- | --- | --- |
| layer_norm_low_rank_producer_forward_fused | 107.4 µs | 105.1 µs | 134.5 µs |
| packed_cell_channel_mix_state_forward_fused | 64.9 | 64.8 | 81.4 |
| rwkv_time_mix3_..._packed_fast | 75.7 | 72.7 | 108.6 |
| layer_norm_adapter_forward_fused_512 | 44.7 | 44.3 | 58.7 |

Strides 5 and 17 agree within 4%, so the sampling error at 17 is small once the sample
count is adequate; stride 1 is 20-30% above both, and the gap is 110.384 ms of brackets
at stride 1 against 86.431 ms at stride 5 with the wall clock moving the *opposite* way
(204.220 ms against 124.148 ms). Two timestamps per dispatch at the compute stage force
the pipeline to drain around every dispatch, so each bracket measures an *isolated*
dispatch and the step no longer overlaps its own work. That is the same mechanism the
harness below finds dominating the wide kernels, seen from the other side: the stride-17
number is the better estimate of what the step pays, and it is still an upper bound
because 1 dispatch in 17 is still isolated by its own timestamps.

At batch 512 the instrument does not move the span at all (1341.762 ms at stride 17
against 1341.146 ms at stride 1) because the step is long enough that the isolated
dispatches still fit; at batch 32 the span moves 98.646 → 130.311 ms. The distortion is
worst exactly where the per-dispatch cost is smallest, which is why the small-batch
regime needed re-measuring before its priority list could be trusted.

It also corrects the ninth pass's unattributed share, which is an exact span minus an
*estimated* bracket sum and therefore moves with the estimate: 13.8% at batch 32 and 4.2%
at batch 512 at stride 17, 15.3% and 9.5% at stride 1, against the 6.7% and 5.1% reported
before. The span is the number to compare across runs; the difference from the brackets
is a range, not a value.

## Sampling erases placement rows, not just samples

The placement table is hit harder than the cost table, because a row exists only if the
sampled dispatches of one (shader, shape) landed in *both* populations:

* `packed_state_slot_backward` (8 cold / 8 warm at stride 1) has no row at all at
  stride 17.
* At batch 512 the producer reads +0.464 µs at stride 17 (3 cold, 6 warm) against
  **+20.196 µs** at stride 1 (48 cold, 96 warm) — a 40× difference in the same run
  configuration.
* At batch 32 the same comparison is well behaved: producer +27.269 (3/6) against
  +29.921 (48/96), `bias_grad` +8.273 (9/5) against +10.107 (96/56), `vector_add`
  +0.997 (11/6) against +1.055 (140/120), `finite_clamp_forward` +0.303 (7/4) against
  +0.287 (144/80).

So the ranking is taken from stride 17 and the placement ratios from stride 1, and every
placement number in the priority list carries its sample counts.

## Placement as an independent variable

`hierarchos-vulkan/src/bin/placement_probe.rs` dispatches one kernel both ways with
placement assigned:

* The cold arm records an explicit barrier before every dispatch, the warm arm none.
  A barrier count read back from the command stream is checked against the expected
  value before a block is timed (measured: one per dispatch cold, zero warm), so the
  assignment is verified rather than assumed.
* Both arms use the same shared inputs and the same shape, and each dispatch writes its
  own output buffer, so no aliasing confound is introduced by the arm that is meant to
  be free of barriers.
* Arms alternate in both orders (warm-cold-cold-warm, then cold-warm-warm-cold) and each
  block is timed as a whole, so the instrument costs one query pair per block instead of
  two timestamp writes per dispatch. Medians over 8 alternating blocks per arm, on
  blocks of 32, 64 and 128 so that anything that scales as 1/blocks is visible.

Per dispatch, at the largest block size (µs):

| kernel | shape | grid | warm | cold | inherited |
| --- | --- | --- | --- | --- | --- |
| `linear_forward` | 32x128x128 | 8x2x1 | 3.72 | 15.44 | +11.7 |
| `linear_bias_forward` | 32x128x128 | 8x2x1 | 4.34 | 16.33 | +12.0 |
| `linear_bias_forward` | 512x128x16 | 1x32x1 | 8.49 | 19.29 | +10.8 |
| `vector_add` | 1x65536x1 | 256x1x1 | 3.90 | 6.03 | +2.1 |
| `finite_clamp_forward` | 1x65536x1 | 256x1x1 | 3.24 | 3.80 | +0.6 |
| `linear_forward` | 512x128x128 | 8x32x1 | 30.06 | 34.12 | +4.1 |
| `linear_bias_forward` | 32x512x512 | 32x2x1 | 20.53 | 50.27 | +29.7 |

Three results, none of which the observed split can show:

1. **Inheritance is a property of the shape class.** The narrow matmul shapes pay
   +10.8…+12.0 µs and are stable to within 0.3 µs across block sizes; a two-stream
   elementwise kernel at 256 workgroups pays +2.1…+2.7; a one-stream one pays +0.1…+0.8,
   indistinguishable from zero. The producer's +20.196 µs (batch 512, 48 cold / 96
   warm) is well above that band, which is the expected direction: it is a much larger
   dispatch than the harness shapes (190 µs warm against 8-19) with ~24 bindings, and
   inheritance tracks how much memory the dispatch first touches after the barrier —
   the same ordering the 2.1 µs for one input stream against 11.7 µs for two shows.
2. **For wide compute-bound shapes it is also a property of the block, so no cost per
   barrier exists.** The same dispatch is 50.44 µs warm at 32 dispatches per submission
   and 20.53 µs at 128, and its cold cost 75.55 → 50.27 µs over the same range: most of
   what a barrier costs there is the inter-dispatch overlap it prevents, not a flush.
   Inheritance for that shape *grows* (25.11 → 26.27 → 29.74 µs) as the warm arm gets
   faster than the cold arm can follow. This is the mechanism behind the stride-1 result
   above, measured with placement under control.
3. **The profile's difference carries the neighbour, not only the placement.** With
   identical neighbours in both arms, `finite_clamp_forward` at 256x1x1 pays +0.6 µs,
   while the step's profile reports **−4.588 µs** for the same kernel at the same grid
   (144 cold, 80 warm). A dispatch's bracket is affected by which kernel preceded it, and
   the placement table has no way to express that.

The harness also pins the host cost that the eighth and ninth passes could only bound:
wall minus GPU is 1.7-5.4 µs per recorded dispatch for these shapes at blocks of 64 and
128, typically 2-3.5 (`vector_add` 6.87 − 3.57 µs at 32 dispatches and 5.65 − 3.75 at 64;
`finite_clamp_forward` 1.7-2.0; `linear_bias_forward` at 1x32x1 3.4-5.4). The step-level
subtraction agrees: wall minus span at batch 32 is 25.5 ms over 7056 dispatches, 3.6 µs
each, and 1.5 µs at batch 2. The eighth pass's ~10 µs was an upper bound from a run its
own instrumentation inflated; the honest level is 1.5-3.6 µs, single-digit as predicted.

Limits: the registry covers only kernels whose dispatch parameters are already pinned by
an existing validation harness, so the numbers transfer to the wide recurrent kernels by
argument rather than measurement; the GPU reading goes through the device's own
per-dispatch timestamps, so its absolute levels carry the same isolation as stride 1 and
only the cold/warm difference is comparable; and each block's dispatches write distinct
output buffers, which the real step does not do.

## What the priority list now says

The count lever is the one this pass can quantify end to end. Select every shader
dispatched at least 240 times per step whose GPU time is at most 6 µs per dispatch:
at batch 32 that is nine shaders — `gradient_accumulate4` (768, 3.5 µs),
`gradient_accumulate` (482, 0.9), `linear_input_grad` (385, 5.1), `linear_weight_grad`
(385, 4.5), `vector_add` (308, 2.1), `parameter_matmul_weight_grad` (288, 4.4),
`parameter_matmul_input_grad` (288, 0.1), `finite_clamp_forward` (288, 1.7), `bias_grad`
(240, 5.3) — and they are 3432 of 7056 dispatches (48.6%) but only 10.523 ms of 85.054 ms
(12.37%) of the GPU step. Their recording adds 5.1-12.4 ms at 1.5-3.6 µs per dispatch, so
the class is 15.6-22.9 ms of a ~124 ms step (13-18%), and *execution* would not pay for
restructuring them — the recording is what makes them worth attacking.

The dispatch counts are batch-independent (6912-7120 per step at every batch size), but
the same kernels' per-dispatch cost is not, so the class itself shrinks as the batch
grows: 8 shaders and 45.2% of the dispatches at batch 2 (7.570 ms of 61.716, plus 4.7-11.2
ms of recording), the 9 above at batch 32, and only 3 shaders and 1538 of 7120 dispatches
at batch 512, where they are 0.49% of the GPU step and under 1% of its wall clock. The
lever therefore belongs to the small-batch regimes, which is the opposite of the
conventional reading that a larger batch is where overhead matters. `gradient_accumulate4`
is the cleanest single candidate: one shader, 768 call sites, one operation. The 1346
`optimizer.gradient-bookkeeping` dispatches stay blocked on the gradient slab, as the
eighth pass established.

Everything else in the top ten is an execution problem with no cheaper structural fix,
and the two regimes disagree about which of them matters. The producer's own cost is
bounded but not explained: about +20 µs of its 107 µs at batch 32 is placement (stride-1
sample), and the harness cannot instantiate it to go further.

## Verified

Ladder 7/7 — loss **5.543217**, 6912 dispatches, 3076 barriers, goldens 144/183/79 — and
`cargo test --lib -j 1` 705 passed / 0 failed / 9 ignored, both after the changes to
`vulkan.rs` (`shader_barrier` is now crate-visible for the harness; the top-shaders
override is read per submission). The new binary self-verifies its own placement before
measuring it, and the profiles behind the ranking were parsed by
`rank_kernel_profile.py` unchanged, which reads every shader line and reports sample
counts. `validation/training_step_probe/kernel_ranking.txt` is regenerated from the
512-row stride-17 run (84 shaders instead of 12).

## Reproduce this pass

```powershell
cd hierarchos-vulkan
cargo build --release --bin training_step_probe --bin placement_probe
$env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"
$env:HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS = "4096"
foreach ($b in 2, 32, 512) { ./target/release/training_step_probe.exe --model-dir target/prof_check/rank_b$b --batch $b --tokens 8 --steps 5 --warmup 1 > target/prof_check/rank_b$b.json 2> target/prof_check/rank_b$b.log }
python validation/rank_kernel_profile.py target/prof_check/rank_b32.log --top 120
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "5"; ./target/release/training_step_probe.exe --model-dir target/prof_check/rank_s5_b32 --batch 32 --tokens 8 --steps 8 --warmup 1 2> target/prof_check/rank_s5_b32.log
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "1"; ./target/release/training_step_probe.exe --model-dir target/prof_check/rank_s1_b32 --batch 32 --tokens 8 --steps 3 --warmup 1 2> target/prof_check/rank_s1_b32.log
Remove-Item Env:HIERARCHOS_VULKAN_PROFILE_KERNELS, Env:HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS, Env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE
./target/release/placement_probe.exe --blocks 32,64,128 --trials 4 --json validation/placement_probe/placement_probe.json
```


# Attention-backward regression closed — 2026-10-15

The previously documented mode-0 large-batch slowdown is fixed on the AMD device.
The cached backward kernel used ten Workgroup variables (14,400 bytes) despite
launching one invocation per workgroup. Moving those caches to Private storage
retains every operation, reduction order, cache bound and dispatch ABI.

Paired GPU timestamp measurements at 64 queries/keys and head dimension 32:
batch 32 falls from 88.8 to 48.4 ms (original artifact 55.8 ms); batch 128 falls
from 263.8 to about 71 ms (original 78.6 ms). Batches 2, 8 and 16 preserve the
small-batch benefit, within about one percent of the earlier cached kernel.
The fixed artifact is faster than the original at all five measured batches.

All 79 attention comparisons remain bit-identical, with unchanged frozen goldens.
The harness now rejects even a one-bit difference in direct A/B mode, and requires
benchmark outputs to match too. An injected one-bit gradient error is rejected.
The shader provenance manifest also guards local size [1,1,1] and zero Workgroup
variables, so the earlier storage regression fails the release gate.

After the final code changes: both debug and release ladders pass all eight checks,
including 86 shader variants and 3,817 full-state digests across 13 training steps;
706 library and 40 trainer tests pass, with nine microprofiles explicitly ignored.
All binary targets pass cargo check. No existing output golden was re-recorded.

Full timings, scope and commands: hierarchos-vulkan/validation/ATTENTION_BACKWARD_PERFORMANCE.md.
Raw paired results: hierarchos-vulkan/validation/release_parity/attention_backward/.
Cross-vendor performance remains unqualified. Long strict-mode jobs can still hit
the driver watchdog; the safe 8-query/8-key strict check passes exactly.


# Gen9 watchdog slice point is length-based and shipped on — 2026-10-15

> Superseded by the next section: the fixed 1,024-dispatch count described here
> was replaced later the same day by a bound measured from each retirement, with a
> 400 ms target and an 8,192-dispatch mis-measurement guard. The numbers below are
> what this pass shipped and are kept as the record it worked from.

The ported Gen9 slice point retired every dispatch: bit-exact on the Intel HD
Graphics 520 target, but priced at about 5x per training step (1,928.3 ms against
the same binary's 380.0 ms legacy step on `training_step_probe --batch 2 --tokens
8 --steps 6`), which is why it stayed reachable instead of being imposed. It now
retires on accumulated submission length: `watchdog_checkpoint()` consults a
dispatch budget per class, the transformer recorders' phase boundaries call
`force_watchdog_checkpoint()`, and the AMD target's budget is 1 — its behaviour is
unchanged by construction rather than by measurement.

Median step against that legacy baseline, same fixture and binary:

| slice point | median step | vs legacy |
| --- | --- | --- |
| 1 (the AMD point) | 1,928.3 ms | +407% |
| 64 | 556.7 ms | +46% |
| 256 | 449.7 ms | +18% |
| 512 | 426.1 ms | +12% |
| **1024 (shipped)** | 401.7 ms | **+6%** |
| 4096 | 384.3 ms | +1% |

1024 is what ships and `GEN9_WATCHDOG_SLICING_DEFAULT` is now `true`;
`HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0` restores the legacy single-submit path.
The causal-LM suite no longer separates the arms (10.36 s / 10.30 s sliced versus
10.56 s / 10.31 s legacy), so the 2.2x it used to cost is gone.

Two defects surfaced only because the policy went live, and both are fixed.
Retiring a submission reset the structural counters, so a sliced run reported the
slice that happened to be open rather than the recorded stream — 224 dispatches
where the fixture records 6,368; they are now carried across a retirement and both
arms report 6,368. The ladder's `step_expectation.json` then turned out to hold
one device's `dispatch_count` as a single value "compared for every device", and
this part dispatches the same fixture 6,368 times against the AMD-recorded 6,464 on
both arms — a graph difference, not a policy one. `mean_loss` and `dispatch_count`
are now per device, `--record-expectation` merges this machine's entry and leaves
every other entry as recorded, and the barrier bound only widens.
`validation/test_training_stack.py` (6 tests) pins that comparison.

The A/B no longer needs two live runs: `validation/kernel_golden/training_state.json`
carries this target's own trajectories for all three cases (13 steps, 3,817 exact
tensor digests), recorded through the legacy path by
`validation/record_training_state.py`, and `validation/verify_training_state.py`
passes on both arms against it.

After the final code changes: `cargo test --lib` 715 passed / 0 failed / 9 ignored;
the ladder is 6/7 with every runnable step PASS and shader provenance BLOCKED
because `spirv-val` is not installed on this machine (it PASSES on the release
machine, per `validation/release_parity/release_final_receipt.json`).

Policy, overrides and the full record: `hierarchos-vulkan/COMPATIBILITY.md`,
"Watchdog-safe submissions" and "The Gen9 target's own training-state trajectory";
the machine-level check table is in `hierarchos-vulkan/VENDOR_TUNING.md`.


# The Gen9 slice point is measured, not chosen as a count — 2026-10-15

The pass above shipped a *fixed* 1024-dispatch slice point. It is safe and it
still costs about 6% per training step on a step whose whole submission is an
order of magnitude short of the watchdog — on the Gen9/AVX2 target that is roughly
2.55 against 2.40 training steps/s, or 40.7 against 38.3 tokens/s at this fixture's
16 tokens per step, paid for protection nothing on this path needed. It was fixed
because a count was the only bound the mechanism had; a watchdog, though, bounds a
submitted job in *time*, so the count is now derived from the time a retirement
actually takes.

Each retirement is timed where it is already synchronous (`submit_and_restart`),
and `watchdog_slice_budget_after_retirement` turns that sample into the count
whose measured cost fits `GEN9_WATCHDOG_SLICE_MICROS` (400 ms), floored at the
1024 dispatches the port shipped at (179 ms at the slowest per-dispatch class
measured here, `layer_norm_low_rank_producer_forward_fused` at 175.0 us) and
guarded at 8192. The target is the safety bound and the guard is not: the guard
exists so that a cheap sample cannot size a submission without limit, and it is set
above the fixture's own 6,368-dispatch step, because a guard below one step forces
every step to pay for a retirement the measurement would not have asked for -- that
is the 0.9% regression this pass fixed (below). Two properties keep it honest. A
sample can only *widen*
the bound, so the policy can never cost more than the fixed count it replaced, and
no cold first slice can pin a run at its own warmup granularity; and the widest
bound a device has measured is published on the device for the batches after it,
so a graph that warms up once does not re-learn the bound on every step. A forced
`HIERARCHOS_VULKAN_WATCHDOG_SLICE_DISPATCHES` pins the budget and skips the
measurement entirely, which is what keeps the granularity A/B and the per-device
validation arms reproducible. AMD has no window at all: its per-dispatch point is
a policy constant, so no sample can loosen it.

Measured on this target (Intel HD Graphics 520, `0x8086:0x1916`, driver
31.0.101.2115, Windows, debug build), `training_step_probe --batch 2 --tokens 8
--steps 6`, every arm interleaved in five rounds so the machine's own drift reaches
every arm equally:

| arm | median step | vs legacy |
| --- | --- | --- |
| legacy single submit | 381.8 ms | -- |
| 8192 dispatches, forced (what the measurement converges to) | 381.7 ms | -0.0% |
| **shipped default (measured bound, 400 ms target)** | **382.7 ms** | **+0.2%** |
| 4096 dispatches, forced (the port's earlier guard) | 385.1-399.5 ms | +1.2 to +1.8% |
| 1024 dispatches, forced (the fixed count it replaced) | 405.6 ms | +5.7% |

The measured bound gives back all but half a point of the 6% the fixed count cost —
about **5.5% per training step** — and it costs nothing at all once it has
calibrated: the forced-8192 row measures -0.0%, and the +0.2% the shipped arm still
shows on a *seven-step* probe is the single calibration retirement per process
(the budget starts at the floor, the first retirement sizes it, and it then sits at
the guard). A real run amortizes that over thousands of steps and does not see it.

This pass also fixed a regression the first version of the measured bound
introduced, which is worth recording because it is the same mistake the port made
twice: a bound that binds a submission the measurement has already priced is a cost
with no protection attached. With the guard at 4096 — below the fixture's own
6,368-dispatch step — every step retired once regardless of how cheap its work was,
and an interleaved six-round A/B against the unsliced path measured **387.8 ms
against 384.4 ms, +0.9%**. Raising the guard above one step and giving the target
the 400 ms that one step's measured work needs takes the same comparison to
**382.7 ms against 381.8 ms, +0.2%**, and the forced-8192 control to -0.0%.

Absolute milliseconds drift between sessions on this laptop (380.0 ms for the same
legacy arm in the previous pass, 392.6 and 378.6 ms in two earlier sessions of this
one), which is why every table above shares one interleaved session and the earlier
forced sweep is quoted as ratios.

On the second fixture the port is a speedup rather than a recovered regression:
`cargo test --lib -- hierarchos_causal_lm_` (six tests, ten seconds of real
training work) runs in **10.28 s** with the measured bound live against **10.49 s**
with `HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0` (median of three interleaved
repetitions each) -- about 2% *faster* than the same binary's unsliced path,
because retiring a slice starts the GPU on it while the host records the next one.
The per-dispatch reading of the same policy cost 1,928 ms per step on the probe
fixture, so this is the first configuration of the port that is faster than not
having it at all.

Nothing numerical moved: the golden's own three cases run under the legacy path,
the measured default and a forced 1024-dispatch point hash byte-for-byte alike
(`95f48308...` batch2, `6c97733e...` batch32, `cc4cb821...` odd_multi_sequence),
and `validation/verify_training_state.py` passes on all three arms against the
recorded trajectories (13 steps, 3,817 exact tensor digests). The strict `2e-7`
model gate was then re-exercised on the shipped default rather than inherited from
an earlier pass: `validation/verify_hf_logits.py --families smollm3 smollm3_yarn
phi3 --atol 2e-7 --rtol 0` in the repo's `.venv-vulkan` oracle -- which rebuilds
the logits binary from the current source first -- reports 0 failing values over
six cases with the worst `max_abs` `5.96e-08`, four times inside the ceiling, on
this adapter with the measured bound live. `smollm3` was re-run once more after the
guard and target change, on the final build, with the same result (`2.98e-08`
unmasked, `5.96e-08` padded, 0 failing values).

```powershell
cd hierarchos-vulkan
python validation/slice_point_ab.py --rounds 5   # interleaved timing + per-arm trajectory hashes
python validation/verify_training_stack.py --skip-build
cargo test --lib
```

After the change: `cargo test --lib` 717 passed / 0 failed / 9 ignored (two new
tests: the pure bound's window and clamps, and a real retirement on this adapter
widening the floor, staying inside the window and publishing what it measured); the
ladder is 6/7 with every runnable step PASS — including the training-state parity
check on the new default — and shader provenance BLOCKED because `spirv-val` is not
installed here. The bound does not touch the one class it never could: a single
dispatch longer than the watchdog is still only shortened at the kernel level.

Policy, overrides and the full record: `hierarchos-vulkan/COMPATIBILITY.md`,
"Watchdog-safe submissions"; the machine-level check table is in
`hierarchos-vulkan/VENDOR_TUNING.md`.
---

# Gen9 shader-free lever inventory: what is left above the kernels — 2026-10-16

The slice-point work moved submission granularity. This pass asks the next
question the port owes before it ships: with no SPIR-V compiler on this machine,
which remaining lever actually moves a Gen9 training step, and by how much?
Every number below is either a runtime switch on one binary or an interleaved
A/B of two binaries on the same fixture, because this machine's absolute level
drifts by more than the effects being measured.

**Machine state, stated up front.** The same fixture
(`training_step_probe --batch 2 --tokens 8 --steps 6 --warmup 1`) that the
earlier passes recorded at 156-168 ms per step now measures 269-275 ms, and at
32 rows 833 ms against a recorded 200 ms. Nothing in the tree explains a factor
of 1.7-4: `HEAD` (`git archive HEAD`, built into its own target directory) and
the working tree are within 0.2% of each other on both fixtures, the vendor and
portable kernel paths are within 1%, and the slice path retires zero times per
step at these sizes. The absolute level is therefore this machine's current
power/clock state, not a code change, and only the interleaved deltas below are
used as evidence. The recorded absolute tables remain the reference for
comparable machine state.

## The step is 93% GPU, and the host is no longer a lever

`hierarchos_vulkan_kernel_profile` at rows 2 (stride 17, release):

| quantity | ms/step | share |
| --- | --- | --- |
| wall | 270.4 | 100% |
| GPU span (first bracket begin to last bracket end) | 249.8 | 92.4% |
| of which dispatch brackets | 234.2 | 86.6% |
| of which no bracket covers (barrier flush, launch latency, idle) | 15.5 | 5.7% |
| host (wall minus span: recording, submission, readback, digests) | ~20 | ~7% |

The eighth pass's ~10 us per recorded dispatch was an upper bound from a run its
own instrumentation inflated; the tenth pass put the honest range at 1.5-3.6 us.
6,368 dispatches at that rate is 9.5-23 ms, which is the whole of the host
column — so even a *free* recording path, and therefore even a fully overlapped
one, is worth at most the 7% the host holds. That closes the async-retirement
idea the slice port made tempting: pipelining the host against the GPU can only
collect host time, and here the host is not what the step spends its time on.

What the GPU does with the 234 ms, by shader:

| shader | dispatches | us/dispatch | ms/step | share |
| --- | --- | --- | --- | --- |
| `layer_norm_low_rank_producer_forward_fused` | 125 | 271.2 | 34.1 | 16.2% |
| `packed_cell_channel_mix_state_forward_fused` | 125 | 238.7 | 30.0 | 14.3% |
| `linear_input_grad` | 334 | 57.5 | 19.2 | 9.1% |
| `layer_norm_adapter_forward_fused_512` | 125 | 88.5 | 11.1 | 5.3% |
| `linear_weight_grad` | 334 | 31.9 | 10.7 | 5.1% |
| `gradient_accumulate8` | 499 | 20.3 | 10.2 | 4.8% |

By category the same step is `rwkv.recurrent` 59.2 ms (28.1%),
`normalization.layer-norm` 55.4 (26.3%), `projection.linear` 53.5 (25.5%),
`control` 12.0, `optimizer.gradient-bookkeeping` 11.6 and
`elementwise-reduction` 10.3. Two fused kernels are 30.5% of the GPU step, and
at rows 2 each dispatch of them launches only `rows` workgroups, so they are
latency-bound rather than arithmetic-bound. That is the frontier, and it is a
kernel-shape problem: the producer's f32 path is one row per workgroup (its
multi-row modules exist only for the fp16-packed path, which is different
arithmetic), and the packed-cell channel mix is one row per workgroup too.

## Three shader-free levers, measured and rejected

**Workgroup size for the packed-cell channel mix (14.3% of the step).** Four
modules exist and are numerically identical: `wg32`, `wg64`, `wg128` and the
shipped 256 differ in exactly two SPIR-V words (`OpExecutionMode LocalSize` and
the `gl_WorkGroupSize` constant) and nothing else, and the host already selects
between them by `HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE`. Two
interleaved rounds per arm at rows 2:

| workgroup | step ms | vs shipped 256 |
| --- | --- | --- |
| 256 (shipped) | 282.9, 281.3 | — |
| 128 | 276.8, 285.5 | within noise |
| 64 | 313.0, 307.5 | +9 to +11% |
| 32 | 382.4, 385.4 | +36% |

The kernel spreads its element loops across the workgroup, so shrinking the
workgroup removes parallelism instead of adding occupancy: the shipped 256 is
already the best of the four on this target. No change.

**The Gen9 vendor modules (25.5% of the step is `projection.linear`).**
`HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS` switches the whole admitted Gen9 set
(tiled `linear-forward`, `linear-bias-forward`, `linear-residual-forward`,
`linear3-forward`) for the portable modules, interleaved: rows 2 +1.2% slower
*with* the Gen9 modules, rows 8 -0.1%, rows 32 +-0.5% with mixed signs. Neutral,
and the reason is already in the selection layer: the geometry gate keeps the
small shapes this probe uses on the portable modules (`VENDOR_TUNING.md` records
the eight-row measurement at 0.93x-0.95x for the Intel module), so the arms
mostly agree by construction and the residual is the families above the gate.
There is no regression to revert and no win to take here; the table is in
`hierarchos-vulkan/VENDOR_TUNING.md`.

**The pending-region hasher.** `prepare_shader_dispatch` probes and
`finish_shader_dispatch` fills three `HashSet<BufferRegionKey>` once per binding
per recorded dispatch, with the default SipHash. Those probes are the hottest
hashed structure in a step, so the sets now carry a local FNV-1a hasher
(`Fnv1a64Hasher` in `vulkan.rs`, the same function the crate already uses for
shader signatures and `build.rs` uses for the artifact registry, no new
dependency). It is membership-equivalent by construction — the sets are only
probed for containment and cleared as a whole, never iterated for anything that
reaches a barrier decision — and four interleaved pairs per fixture put the
step-time effect inside this machine's noise:

| fixture | pre-change ms | post-change ms | paired median |
| --- | --- | --- | --- |
| rows 2 | 268.51, 278.61, 270.15, 269.61 | 269.21, 270.25, 271.53, 268.00 | -0.06% |
| rows 8 | 364.34, 362.29, 359.28, 359.52 | 357.22, 365.24, 359.07, 357.23 | -0.8%, one pair reversed |

It is kept for the work it removes per probe, not for a measured win, and it is
verified inert where it matters: dispatch and barrier counts are *identical*
between the arms (6,368 / 2,980 at rows 2, 6,560 / 3,124 at rows 8), the loss is
`5.290753` on both, and `--state-output` from the two binaries hashes
byte-for-byte alike (`dec3e2bf...`, 162,302 bytes). SipHash's randomized seed
also made that set layout vary per process; FNV's fixed seed does not.

## What the row sweep says today

Same probe, 8 tokens per sequence, fresh fixture per point:

| rows | tokens/step | step ms | tokens/s |
| --- | --- | --- | --- |
| 2 | 16 | 275.3 | 58.1 |
| 8 | 64 | 363.8 | 175.9 |
| 16 | 128 | 529.2 | 241.9 |
| 32 | 256 | 833.0 | 307.3 |
| 64 | 512 | 1475.1 | 347.1 |

The shape matches the fourth pass's conclusion — rows amortize the fixed
per-dispatch cost, sequence length does not — and on today's machine the curve is
still climbing at the trainer's default of 32 rows (+13% tokens/s at 64 rows).
The knee itself has moved with the clocks: the earlier sweep put the practical
point at 512 rows, this one is still rising at 64, so the default stays where it
is and the curve is the argument for raising rows, not a constant.

## The frontier this pass cannot cross

The remaining 30% of the step is two latency-bound fused kernels, and every
cheaper shape for them is a *new SPIR-V module*: a multi-row f32 producer, more
rows per workgroup for the packed-cell channel mix, and the tenth pass's
`gradient_accumulate8` fusion candidate (499 dispatches, 4.8% of the GPU step,
one operation). This machine has neither `glslang` nor `spirv-val` — no Vulkan
SDK is installed — so no shader here can be compiled, and the tree's own rule
that every `.spv` must be byte-reproducible from the command recorded in
`README.md` cannot be met for a new one. That is the whole blocker: with a
toolchain, the producer and the packed-cell channel mix are each worth up to a
measured 15% of the GPU step, and nothing else on the shelf is.

To reproduce the interleaved switch A/Bs (run them on a quiet machine; the arms
must alternate, and a concurrent build moves the numbers by more than the effect):

```
cd hierarchos-vulkan
HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE=128 ./target/release/training_step_probe.exe --model-dir target/ff/wg128 --batch 2 --tokens 8 --steps 6 --warmup 1
HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1 ./target/release/training_step_probe.exe --model-dir target/ff/voff --batch 32 --tokens 8 --steps 4 --warmup 1
# HEAD-vs-tree A/B: git archive HEAD into a scratch directory, build it there, alternate the two binaries
```

Verified on the changed build: `cargo test --lib` **717 passed / 0 failed / 9
ignored** in 108.74 s, and the ladder is **6/7** with every runnable step PASS —
binding-access audit (0 narrower bindings over 468 modules), the three golden
digest sets (144 / 183 / 79 invocations), the training-step probe at **6,368
dispatches and 2,980 barriers against the 3,076 upper bound, loss 5.5432177**,
and the training-state parity over 13 steps and 3,817 tensor digests — with
shader provenance BLOCKED exactly as before, because `spirv-val` is not on this
machine. The strict model gate was re-run on the changed build rather than
inherited: `validation/verify_hf_logits.py --families smollm3 --atol 2e-7
--rtol 0` in the repo's `.venv-vulkan` oracle exits 0 with both cases at **0
failing values** (`max_abs` 2.98e-08 unmasked, 5.96e-08 padded), the same values
the slice-point pass recorded on the pre-change binary. The state-digest
equality above is the direct check on the change; nothing else moved because it
touches no ordering, no barrier decision and no shader.
---

# Pinned shader toolchain installed locally, and what it found — 2026-10-16

Every pass so far ended with the same caveat: shader provenance was **BLOCKED**
because `spirv-val` is not installed on this machine, and no shader could be
compiled here at all. That gate is not decorative — the tree's rule is that one
recorded toolchain reproduces every committed artifact byte-for-byte — so this
pass installs that toolchain inside the repository, without administrator
rights, and runs the gates it unlocks.

## What is installed, and where

`validation/install_shader_toolchain.sh` puts the pinned toolchain under the
repository's gitignored `.tools/` and writes `.tools/env.sh` to put it on `PATH`:

| binary | source | version |
| --- | --- | --- |
| `glslangValidator` | official Khronos glslang Windows release asset, SHA-256 checked before extracting | **16.5.0** |
| `glslc` | conda-forge `shaderc` package, unpacked by a micromamba the script downloads | **2025.3** |
| `spirv-val`, `spirv-dis` | conda-forge `spirv-tools` package, same environment | SPIRV-Tools 2025.5 |

Neither SPIRV-Tools nor shaderc publishes Windows binaries of its own, which is
why two of the three arrive as conda packages; nothing is installed system-wide
and no path outside the checkout is touched. `PATH` order is part of the pin:
the glslang directory comes first so `glslangValidator` is the pinned release
rather than the glslang that the shaderc package depends on (15.4.0 for the pinned
2025.3 package; the 2026.4 package pulls 16.6.0, which does not reproduce the
attention kernels). A fresh install was tested end to end by
deleting `.tools/glslang-16.5.0` and `.tools/env`, re-running the script, and
re-checking provenance from what it installed.

```bash
cd hierarchos-vulkan
bash validation/install_shader_toolchain.sh
source ../.tools/env.sh
python validation/check_shader_provenance.py
python validation/verify_training_shaders.py
```

## The gate's first real run found a toolchain split, not a stale artifact

The first install took the newest conda shaderc (2026.4) and reported 83 of 86
artifacts `identical` with three `drifted`. Each was diagnosed rather than
assumed:

* **`transformer_rms_norm_input_grad.spv` and
  `transformer_rms_norm_input_grad_residual.spv`** (manifest compiler `glslc`,
  flags `-O`) rebuild to 40,780 and 41,540 bytes under shaderc 2026.4 against
  committed 40,916 and 41,676, and rebuild to the committed bytes **exactly**
  under **shaderc 2025.3**. Disassembly puts the difference at six fewer integer
  index adds and one fewer float `OpBitcast` — a later optimizer, no floating-
  point opcode difference, no source difference. So the tree is spanned by two
  shaderc releases, and 2025.3 is the one that explains all of it; with it
  installed, the check reports `{"identical": 86}` and passes.
* **`transformer_attention_backward.spv`** (manifest compiler
  `glslangValidator`, flags `-V --target-env vulkan1.0 -Ishaders`) rebuilds to
  104,544 bytes under the pinned 16.5.0 but the committed working-tree bytes were
  104,528 — and those bytes reproduce **exactly** under glslang **16.6.0**
  (`c428846b...`). The session had rebuilt that kernel with a different glslang
  release than the pin names. No single glslang release explains the tree in that
  state: 16.6.0 reproduces the backward kernel but drifts
  `transformer_attention_forward.spv`, which 16.5.0 reproduces.

The repair is the pin's own recipe, applied to the artifact that was out of
step: `glslangValidator -V --target-env vulkan1.0 -Ishaders
shaders/transformer_attention_backward.comp -o shaders/transformer_attention_backward.spv`
with 16.5.0 (`e78cb292...`, 104,544 bytes). One toolchain now explains every
artifact, and the manifest needed no change because it already recorded that
recipe.

## Why the swap is arithmetic-preserving, measured not assumed

The two builds of the same source differ only in how the kernel's `+Inf` clamp
constant is materialized. Disassembling both (`spirv-dis`, IDs normalized):

| | 16.5.0 build (shipped) | 16.6.0 build (what the session had) |
| --- | --- | --- |
| size / ID bound | 104,544 bytes / 4368 | 104,528 bytes / 4367 |
| the constant | `OpBitcast %float %uint_2139095040` | `%float_0x1p_128 = OpConstant %float 0x1p+128` |
| its uses | 2 x `OpReturnValue` | 2 x `OpReturnValue %float_0x1p_128` |
| floating-point opcode counts | **identical** | **identical** |

`0x7F800000` is exactly `+Inf` in binary32 and `0x1p+128` is `+Inf` as well, so
the two modules return the same value; the 16-byte difference is that constant's
path, not the kernel's arithmetic. (The pin table above records the 16.6.0
difference as making a module "16 bytes larger"; for this module as it now
stands the pinned build is the larger one, for the reason in the table here.)

## Verified on the rebuilt module

* `python validation/check_shader_provenance.py` → `{"identical": 86}`, **PASS:
every checked artifact reproduces byte-for-byte**, exit 0. Run twice: against the
in-place toolchain and again from a from-scratch install of
`install_shader_toolchain.sh`. (At the time of that run the command named the
glslang pin on its command line; the release is now recorded per artifact and
neither checker needs the flag — see the next section.)
* `python validation/verify_training_stack.py` → **7/7**, with the provenance step
running instead of blocked: `shader provenance 86 variant(s) match their sources
and flags`, binding-access audit clean over 468 modules, goldens
144/183/79 invocations, probe `6,368 dispatches, 2,980 barriers (upper bound
3,076), loss 5.5432177`, training-state parity 13 steps / 3,817 exact tensor
digests. The attention golden step ran against the harness binary linked *after*
the module was rebuilt (module 02:10, binary 02:20), so those 79 invocations are
the rebuilt kernel's.
* `cargo test --lib -j 1 -- --test-threads=1` → **717 passed / 0 failed / 9
ignored**, exit 0, on the rebuilt module.
* Strict model gate `verify_hf_logits.py --families smollm3 --atol 2e-7 --rtol 0`
→ exit 0, both cases **0 failing values** (`max_abs` 2.98e-08 unmasked, 5.96e-08
padded), unchanged from the pre-rebuild binary.

The rebuilt module is therefore verified numerically (goldens, state digests,
model gate, unit suite) and textually (one toolchain reproduces it), which is what
makes it safe to ship in place of the 16.6.0 build the session had produced from
the same source.

## The pin is recorded per artifact, not per compiler

The toolchain split found above was survived rather than expressed: the manifest
named a compiler per artifact and the check resolved one binary per name, so a
tree whose artifacts came from two releases could only be checked one release at a
time. Entries in `training_shader_manifest.json` now also record
`compiler_version` — the release that compiler reported about itself when the
artifact was rebuilt — and both checkers choose the toolchain per artifact.

* `check_shader_provenance.py` accepts repeated `--compiler-path NAME=PATH`
  candidates (else `PATH`), compiles each artifact with the candidate that reports
  the release that artifact records, and reports an installed-but-wrong toolchain
  as `wrong_version` listing the candidates it saw instead of as drift. A
  `--require-version NAME=VERSION` flag still pins entries that record no release
  of their own; contradicting a recorded release is a `pin_conflict` — the
  manifest wins, loudly — rather than a silent override.
* `--record-versions` writes the release that reproduced each artifact back into
  the manifest, so a pin comes from a verified rebuild instead of hand editing. It
  records only artifacts that were byte-identical and leaves the others alone. All
  86 entries were recorded this way: 3 `glslangValidator` at `16.5.0` and 83
  `glslc` at `2023.8` (84 once the two-row producer below joined the manifest), the
  release the pinned shaderc package reports about itself
  (the package version is 2025.3; the newest package reports `v2026.4`, so the two
  are distinguishable from the banner alone).
* `verify_training_shaders.py` — the ladder's shader-provenance step — uses the
  same selection, so a spanned tree passes the ladder rather than being reported as
  drift.

Verified against the recorded pins, with the toolchain that reports them on
`PATH`:

* `python validation/check_shader_provenance.py` with no flags →
  `{"identical": 86}`, **PASS**, exit 0.
* A copy of the manifest with `adamw.spv` recorded at `2026.4`, run with **both**
  shaderc installs offered and the 2026.4 one first: 86/86 identical, `adamw.spv`
  compiled by 2026.4 and the other 82 by 2023.8 — the recorded release chooses the
  binary, not the candidate order.
* The same copy with only 2026.4 available: 82 `wrong_version`, 4 identical, exit
  1 — a wrong toolchain is never reported as an artifact difference.
* `python validation/verify_training_shaders.py` → PASS 86 on the pinned
  environment and on the spanned copy given two candidates; with only 2026.4 it
  fails naming the artifact, the release it records and what the compiler reports.
* `python -m unittest discover -s hierarchos-vulkan/validation -p 'test_*.py'` in
  `.venv-vulkan` → 41 tests OK, 1 skipped. The provenance suite is 26 of them; it
  grew cases for candidate selection, a spanned tree, `pin_conflict`, recording
  only what reproduced, and the release lines real compilers print.

## The FP32 LN1/low-rank producer has a two-row variant, and the measurement declined it

`layer_norm_low_rank_producer_forward_fused` is the FP32 LN1/low-rank producer, and
on this device it is the step's normalization class: 144 dispatches and 25.19 ms of
a step (26.2%) in `validation/training_step_probe/profile_step.log`, and the class
the watchdog prices at 175.0 us. It dispatched `rows` 64-lane workgroups with one
row each, so every row read the whole `w1/a1/g1` and `w2/a2/g2` working set again.

The new `layer_norm_low_rank_producer_forward_fused_two_rows.comp` gives one
workgroup a pair of rows and shares every weight read between them: the first-stage
weight of a (column, rank) pair and the w2/a2/g2 element of an output column each
serve two FMAs, so a row pays half the weight bytes. Each row keeps its own scalar
LayerNorm reduction order -- lane 0 reduces the first row of the pair and lane 1 the
second, which cannot change either row's values -- and its own ascending FMA chain,
so the variant is bit-identical to the one-row kernel and only schedules
differently. An odd row count leaves the pair's second row idle.

The FP32 path now measures the two topologies on the device the way the FP16 path
already did: the same probe buffers, the same three interleaved rounds, the same
median and the same 2% material-win guard, with its own cache because the two
families select among different kernels. The one-row kernel stays the default and
the pair only displaces it where it is faster.
`HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY=one-row|two-rows` forces one of them
for verification and for bisecting a scheduling regression; the column-slice
variant in the later section inherits both this measurement machinery and the
cache, and is the variant this device now selects.

Measured on the HD Graphics 520. The producer's fused form is enabled only for ranks
`<= 64`, which in this family is `width = 128, ranks 32/32/64`, so that is the whole
measurable domain:

| batch | one row | two rows | selected |
| --- | --- | --- | --- |
| 2 (1 pair) | 0.2285 ms | 0.2771 ms | one-row, pair 21% slower |
| 8 (4 pairs) | 0.2283 ms | 0.2847 ms | one-row, pair 25% slower |
| 32 (16 pairs) | 0.2391 ms | 0.2866 ms | one-row, pair 20% slower |

The pairing loses here for two reasons the table shows: at `width = 128` the second
stage's weight working set is small enough that cache serves it whatever the
schedule does, so sharing those reads saves nothing, and halving the workgroups
halves the parallelism the kernel actually runs on (1 and 4 workgroups in the first
two rows). The variant is kept because that trade can move the other way on a wider
geometry or a part with more resident workgroups per EU, and because the choice is
per device and shape; it is not selected on this one, which is the result of
measuring it rather than assuming it. The variant this device does select takes the
opposite trade (see *The FP32 LN1/low-rank producer splits its second stage across
workgroups* below).

Verified:

* Provenance: `check_shader_provenance.py` → `{"identical": 87}` PASS, and the new
  artifact recorded its own release through `--record-versions` instead of having it
  typed in. `spirv-val` accepts the module and `audit_shader_artifacts.py` reports
  it matching its source.
* Ladder: `verify_training_stack.py --skip-build` → **7/7**, the binding-access
  audit clean over 469 modules (468 plus the new one), goldens 144/183/79, probe
  unchanged at 6,368 dispatches / 2,980 barriers / loss 5.5432177, training-state
  parity 13 steps / 3,817 digests. A topology change alters workgroups per dispatch,
  not the number of dispatches, so the probe's per-device expectation still holds.
* Arithmetic: `verify_training_state.py` → PASS, 13 steps and 3,817 exact tensor
  digests, under `..._TOPOLOGY=two-rows`, under `=one-row`, and with no override
  (the measured choice). The golden was recorded with the one-row kernel, so the
  pair variant reproducing it exactly is the equivalence proof.
* `cargo test --lib -j 1 -- --test-threads=1` → 717 passed / 0 failed / 9 ignored.
* Strict model gate `verify_hf_logits.py --families smollm3 --atol 2e-7 --rtol 0`
  (`.venv-vulkan`) → exit 0, 288 values, **0 failing**, worst `max_abs`
  5.96e-08 against the fixed 2e-07 ceiling, unchanged from the pre-change result.
* `python -m unittest discover -s hierarchos-vulkan/validation -p 'test_*.py'` in
  `.venv-vulkan` → 41 tests OK, 1 skipped.
* `verify_vendor_parity.py` (`.venv-vulkan`) → PASS, 21 comparisons on both the
  resolved plan and the forced-portable path, worst `max_abs` 5.960e-08 on both,
  exit 0 -- a second kernel-selection choice in this graph is still numerically
  inert. (Run it with the venv python: it drives `verify_hf_logits.py` with
  `sys.executable`, so a bare interpreter without torch exits 1 in a second.)

## The autotuner's decisions are now a checked-in, replayable record

A topology choice used to exist only as an env-gated log line, so a device that
began selecting a different variant changed every training step's schedule with
nothing on disk saying it had. The producer now appends every **measured** decision
as one JSON line when `HIERARCHOS_RWKV_LOW_RANK_TOPOLOGY_RECORD` names a file (the
Rust side stays the source of the measurement; the record's format and policy live
in Python), and `validation/topology_decisions.json` is the checked-in result: one
entry per (family, device, geometry) with the medians the choice was made from,
`selected`, and `margin_fraction`. Both families publish through it -- the FP16 and
FP32 LN1/low-rank autotuners.

`validation/verify_topology_decisions.py` replays several probe geometries --
`REPLAY_GEOMETRIES`, each with the reporter pointed at its own scratch file and with
the force and disable variables stripped so it always records a measurement -- and
then:

* every decision the replay measured must be recorded for this device and must still
  select the same variant -- a decision that moves is a failure, because the
  scheduling choice is what the record documents;
* timing drift is a note with the recorded/observed ratio, never a failure: a median
  from a shared GPU moves between runs even when the choice does not;
* every geometry in the replay must measure something. A geometry is a promise that
  this device decides a kernel at that shape, and the record is only as wide as the
  geometries that produced it: a shape that goes quiet -- the fused producer went
  away, ranks grew past the kernel's limit -- would otherwise shrink the audited
  coverage while the step kept passing, so it fails instead and names the geometry;
* recorded decisions the replay does not exercise (another device, or a geometry the
  replay no longer drives) are notes, so auditing one device does not erase or
  mis-check the rest;
* `--record` merges this run's decisions in and leaves every other device's entries
  untouched. Only measured decisions are ever written.

The replay used to be one fixture at `width 128, batch 2`, which is one cell of the
space a decision lives in and not the one where the newest variant loses. It now
drives four shapes whose keys differ in every field the autotuners use -- width,
ranks, `batch_pairs` and `has_unpaired_tail` -- two of which exist because they
disagree with the others:

| geometry | width | ranks | pairs | selected | margin |
| --- | --- | --- | --- | --- | --- |
| `width64-batch2` | 64 | 8/8/8 | 1 | one-row | 0.000 |
| `width128-batch2` | 128 | 32/32/64 | 1 | column-slices | 0.125 |
| `width128-batch3-tail` | 128 | 32/32/64 | 2 + tail | column-slices | 0.160 |
| `width128-batch32` | 128 | 32/32/64 | 16 | column-slices | 0.170 |

The `width 64` row is why the set is not just `batch 2` at several sizes: there the
sliced kernel has a single 64-column slice, measures 0.08131 ms against the one-row
kernel's 0.07704 ms, and the material margin keeps the kernel the device shipped
with. So the record now documents that this device selects **different** variants at
different shapes, and the audit fails if either row moves in either direction.

Verified: the record started as a single measurement (`ln-low-rank-f32`, width 128,
ranks 32/32/64, batch_pairs 1, one-row 0.226572 ms against two-rows 0.277692 ms,
selected one-row) and now holds this device's four geometries above, written by
`--record`; the first multi-geometry audit run failed on the three shapes the record
did not have yet (`measured here but not recorded`, exit 1), which is the guard
working end to end, and the audit passes on all four afterwards. A copy of the record
with `selected` flipped fails naming both variants and exits 1; a copy with the
timings tripled passes with `0.33x` drift notes and exits 0; 20 unit tests pin the
parser, the format checks (duplicates and unsorted entries mean hand editing), the
comparison rules and the geometry-coverage rule.

## The ladder installs and reports the pinned shader toolchain itself

A fresh clone used to get `BLOCKED shader provenance ... glslc must be on PATH`,
which says nothing about what to do. The ladder now has a toolchain step ahead of
it: it prepends the repository's `.tools/` install to `PATH` when it exists, runs
`validation/install_shader_toolchain.sh` when the pins are still missing, and
reports the releases it ended up using. `--no-toolchain-install` reports instead of
installing. When the pins cannot be produced the step is BLOCKED with the exact
commands, and the provenance step that follows inherits the `PATH` either way.

Verified: a run without sourcing `.tools/env.sh` reports `prepended
...glslang-16.5.0\bin, ...env\Library\bin to PATH: glslangValidator 16.5.0; glslc
2023.8; spirv-val 2025.5` and the ladder is **9/9**, with provenance checking 87
variants in the same run. With `.tools` moved aside and installs disabled the step
is BLOCKED with the install commands and exits 1, and `.tools` was restored
afterwards. 5 unit tests drive the injected lookup/directories/installer paths.

## The FP32 LN1/low-rank producer splits its second stage across workgroups

`layer_norm_low_rank_producer_forward_fused` runs one 64-lane workgroup per batch
row, so a small batch is one to four workgroups on a 24-EU part, and at
`width = 128` each lane produces two output columns. The stages before the second
one cannot be shared across workgroups without changing their order -- lane 0 owns
the row's LayerNorm reduction, and each lane's first-stage sums are chains over the
whole row -- so `layer_norm_low_rank_producer_forward_fused_column_slices.comp`
trades replicated work for parallelism instead: the grid is `[ceil(width/64), rows,
1]`, every workgroup owns one 64-column slice of one row, and each lane produces
exactly one output column through the same chain, in the same order.

Replication is safe here because the stages that depend only on the row compute the
same values in every slice: the reduction's addition order, the mixed
`x_norm/xw/xa/xg` values and the w/a/g rank activations are all recomputed
identically, and nothing is reduced across workgroups. Only the writes are
partitioned -- identical bytes written by several workgroups would still be a race
-- so slice 0 owns the materialized backward tape and the caller's hidden vectors,
and each w/a/g column is written by the slice that owns its range.

Measured on the HD Graphics 520 (`width = 128`, ranks 32/32/64, medians of three
interleaved rounds; `HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG=1` prints
them):

| batch | one row | two rows | column slices | selected |
| --- | --- | --- | --- | --- |
| 2 | 0.2328 ms | 0.2797 ms | 0.1956 ms | column slices, 16% faster |
| 8 | 0.2308 ms | 0.2773 ms | 0.1972 ms | column slices, 15% faster |
| 32 | 0.2376 ms | 0.2799 ms | 0.2037 ms | column slices, 14% faster |

The win is parallelism rather than reuse: the two-row variant halves the number of
workgroups and loses 18-20%, while splitting one row across workgroups adds them and
wins at every measured batch. The variant only exists where the fused producer does,
which is ranks `<= 64`, so this is the whole measurable domain in this family. The
device's decision moved with it -- `validation/topology_decisions.json` now records
`ln-low-rank-f32-column-slices` at 0.191937 ms against one-row's 0.232166 ms
(`margin_fraction` 0.173) instead of the one-row entry the pair variant left behind
-- and `HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY=one-row|two-rows|column-slices`
forces any of the three for verification or bisection.

**The first version of this shipped a wrong grid, and the goldens caught it.** The
kernel takes its row from `gl_WorkGroupID.y` and its column slice from
`gl_WorkGroupID.x`, but the production dispatch site reduced a topology to its
`rows_per_workgroup()` -- 1 for the sliced kernel -- and built `[batch / rows, 1, 1]`
from it, so every row but the first went unwritten and the rows it did write were
written by the wrong slices. The timing path built its own, correct two-dimensional
grid, so the autotuner measured the variant honestly, selected it, and then the step
it selected it for computed the wrong answer: `verify_training_state.py` failed at
step 0 on both geometries that use the fused producer (1,533 of 1,752 tensor digests
at batch 2) while the other two topologies passed, which is what localized it to
this one rather than to the selection machinery.

The fix is that both sites now ask one method,
`LayerNormLowRankForwardF32Topology::groups(batch, width)`, for the grid, so a grid
that disagrees with the kernel's own row/slice mapping is not expressible at one
site and not at the other. Three unit tests pin that method, including the specific
mistake: the sliced topology's grid is asserted different from the row-per-workgroup
grid for a shape with more than one slice.

Verified:

* Arithmetic: `verify_training_state.py` PASS, 13 training steps and 3,817 exact
  tensor digests, under `..._TOPOLOGY=column-slices`, under `=two-rows`, under
  `=one-row`, and with no override -- and the no-override run selects column slices
  on this device. The golden was recorded with the one-row kernel, so the other two
  reproducing it exactly is the equivalence proof.
* Ladder: `verify_training_stack.py` → **9/9** with the toolchain step, 88 variants
  reproducing their sources and flags, the binding-access audit clean over 470
  modules, goldens 144/183/79, the probe unchanged at 6,368 dispatches / 2,980
  barriers / loss 5.5432177, topology decisions matching the refreshed record, and
  training-state parity. A topology change alters workgroups per dispatch, not the
  number of dispatches, so the probe's per-device expectation still holds.
* `cargo test --lib` → 720 passed / 0 failed / 9 ignored (717 before, plus the three
  grid tests).
* The failure itself was reproduced twice: a forced `column-slices` run and the
  default run (which selects it) both failed the same goldens at step 0 before the
  fix, and a scratch kernel that only moved the one-row kernel's row to
  `gl_WorkGroupID.y` under a two-dimensional grid failed identically -- which is how
  the dispatch, not the arithmetic, was identified as the cause while the two
  kernels' code-only diffs were empty apart from the row/slice mapping and the write
  guards.

## Training-trajectory parity runs once per LN1/low-rank topology arm

Every LN1/low-rank variant is supposed to be bit-identical to the kernel the committed
trajectories were recorded with, but that equivalence was proved by hand: three runs of
`verify_training_state.py` with `HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY` set to a
different value, done once and written down. The ladder now runs it, so a later change
that breaks one variant's arithmetic cannot land behind a green default run.

The state step runs the suite once per arm -- the shipped choice first, then
`one-row`, `two-rows` and `column-slices` forced -- and every arm must reproduce the
same 13 steps and 3,817 exact tensor digests as the recorded baseline.

**Forcing is not proof, though.** A variant this device cannot offer is rejected by the
producer, which then runs the default kernel: that arm would pass while proving nothing
about the variant it named. So the producer now logs both sides of that decision -- the
request it acknowledged (`forced topology=<label> requested=<value>`) and the one it
had to fall back from (`... topology=<label> requested=<value> is unavailable on
device=...; using <default>`) -- and the values come from the topology's own
`env_value()`, so the string that selects a kernel, the string that is printed and the
string a harness matches cannot drift apart. `verify_training_state.py` reads the
probe's stream and credits an arm only where the value it asked for is the value that
ran; `--topologies` narrows the arms, and with no values runs only the shipped choice.

Coverage is reported rather than assumed, case by case:

    note  one-row: the forced variant ran in 2/3 case(s) (batch2, batch32)
    note  two-rows: the forced variant ran in 2/3 case(s) (batch2, batch32)
    note  column-slices: the forced variant ran in 2/3 case(s) (batch2, batch32)
    PASS: 4 arm(s) x 13 training steps, 3817 exact tensor digests each (measured first);
          forced arms exercised: one-row 2/3, two-rows 2/3, column-slices 2/3

The third case (`odd_multi_sequence`, width 256) does not fuse this producer at all --
the FP16 family owns that shape -- so no arm is credited for it and the line reads
`2/3` instead of implying the arm covered the suite. The rules are:

* an arm that ran the variant in at least one case is verified there, and the cases it
  fell back in are named next to it;
* an arm this device rejects in every case is a note: the device cannot exercise the
  variant, and saying so is more useful than a red step;
* an arm that ran nowhere *and* was never rejected is a failure, because then the fused
  producer never dispatched at all and nothing was verified.

Verified:

* Ladder: `verify_training_stack.py` → **10/10**, with the state step reporting `4
  arm(s) x 13 training steps, 3817 exact tensor digests each (measured first); forced
  arms exercised: one-row 2/3, two-rows 2/3, column-slices 2/3` and the topology step
  auditing four measured decisions. Provenance 88 variants, binding audit 470 modules,
  goldens 144/183/79, probe 6,368 dispatches / 2,980 barriers / loss 5.5432177.
* The failure path end to end: `--topologies nonsense` exits 1 with `nonsense: the
  forced variant ran in no case and was never reported unavailable, so the fused
  producer did not dispatch and nothing was verified; pass --topologies for the arms
  this device can run`, and `--topologies` with no values runs only the measured arm
  and passes.
* A value nobody defines reports what is available instead of silently ignoring it:
  `HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY expects one of one-row, two-rows,
  column-slices; ignoring "nonsense"`.
* `python -m unittest discover -s hierarchos-vulkan/validation -p 'test_*.py'` in
  `.venv-vulkan` → 74 tests OK, 1 skipped (63 before): 8 cases pin which request
  credits an arm and which arms fail, 3 pin the new geometry-coverage rule.
* `cargo test --lib` → 720 passed / 0 failed / 9 ignored.

## The packed-FP16 producer has the same arms, and coverage is checked in

The FP32 arms above can never exercise the packed-FP16 producer, and that is not an
oversight in the arm list: the training numerical policy picks the family. Under the
default FP32 policy the probe's LN1/low-rank decisions are the FP32 family's alone, and
under `HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-storage-fp32-compute` the FP32 producer
does not dispatch at all -- the packed kernel prints instead. So the FP16 family needed
its own suite, not another arm in the FP32 one.

`HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY` now names a variant in either family. The
FP16 table is `one-row`, `one-row-cached`, `one-row-rank128`, `two-rows`,
`two-rows-wide` and `four-rows`, and the vocabulary is their union with the FP32 table
(seven values, because `one-row` and `two-rows` deliberately name one variant in each --
a single arm then exercises both families when both dispatch). A value in neither table
is reported once with the whole vocabulary, and a value only one family offers is
answered by that family's own path: an accepted request prints `forced topology=<label>
requested=<value>`, and a request this device cannot offer prints `topology=<label>
requested=<value> is unavailable on device=...; using <default>` and falls back. Both
strings come from the topology's own `env_value()`, exactly as the FP32 side does it, so
the arm the harness credits is the arm that ran.

The FP16 suite is a second golden, `validation/kernel_golden/training_state_fp16.json`,
because FP16 storage rounds the operands and a trajectory recorded under one policy
cannot be compared with the other. Its three cases are `batch2` and `batch32` at width
128 -- where this part offers `one-row`, `one-row-cached` and `two-rows-wide` -- and
`small_hidden32`, where the width/rank gates admit `two-rows` and `four-rows` as well, so
every variant that exists at a covered shape is exercised by at least one case. The
policy a golden was recorded under lives in the golden's own `env` block, read by both
`record_training_state.py` and `verify_training_state.py`, so the recorder cannot record
one policy's arithmetic and the verifier replay it under another.

This golden is not a pre-optimization baseline and does not pretend to be one: it was
recorded on the shipped kernels, so it pins the FP16 variants to *each other* while
`training_state.json` keeps pinning the FP32 trajectory to the earlier baseline. The
packed family's arm coverage on the Gen9 part is:

| arm | batch2 (w128) | batch32 (w128) | small_hidden32 |
| --- | --- | --- | --- |
| `one-row` | exercised | exercised | exercised |
| `one-row-cached` | exercised | exercised | exercised |
| `one-row-rank128` | unavailable | unavailable | unavailable |
| `two-rows` | unavailable | unavailable | exercised |
| `two-rows-wide` | exercised | exercised | unavailable |
| `four-rows` | unavailable | unavailable | exercised |

`one-row-rank128` is gated on a rank above 64 and this device's shapes are 8/8/8 and
32/32/64, so the arm verifies nothing here and its record says so: the coverage entry is
`unavailable` in all three cases, which is also what makes it a failure if the variant
ever disappears.

### What each device exercises is checked in

The per-arm fraction above used to be a line in the run's output and nothing more, so a
device that quietly stopped exercising a variant read as `2/3` forever and no step went
red. `validation/training_state_coverage.json` closes that: one entry per (suite,
device) holding the golden cases the measured arm compares, how much each one compares,
and where every forced arm ran and where this device reported it unavailable. A run now
fails when any of it shrinks -- an arm that no longer runs where it ran, a golden case
that is no longer measured, or a case that compares fewer steps or tensors than the
record has. Growth is a note. A device with no entry fails rather than passing by
default, and `--record-coverage` merges this machine's evidence in, replacing only the
entries this run measured and refusing to shrink an existing entry unless
`--allow-coverage-shrink` says the loss is intended -- removing coverage from a file that
exists to prove a variant is still tested has to be an explicit edit. `--topologies`
restricts the arms and skips the audit, because a run that forces fewer arms than the
record covers has not verified what the record claims.

### The FP16 family stays out of the decision record, and that is a measurement

(This was written under the median-only rule. The unanimous-round rule landed after it and
is measured in the 2026-10-16 pass at the end of this file: the choice is now stable at the
structural default -- thirty-five consecutive runs, all `one-row` -- and the reason the
family is still absent from the record is structural: the replay that writes the record runs
at the FP32 policy, where the packed producer never dispatches.)

The FP32 decisions are replayed because a different variant is 14-17% faster there and a
selection that moved would be a scheduling change worth failing on. The packed family
does not have that headroom on this part. At width 128 the measured candidates were
`one-row-cached` 0.2713 ms, `one-row` 0.2813 ms and `two-rows-wide` 0.2852 ms in one run,
and 0.27789, 0.28005 and 0.28442 in another, where the cached arm's 0.8% lead was inside
the 2% material margin and the selector kept `one-row`. Runs disagree about which variant
this part should ship, which means the decision is not stable enough to check in:
recording it would turn ordinary timing noise into a red ladder step. So the FP16
producer ships its structural default with every variant proven bit-identical, and the
few-percent question stays open rather than being answered with a coin flip. That is the
honest shape of the speed claim: the FP32 split is a stable 14-17% kernel-level decision
that is recorded and replayed, and the FP16 margin is not.

Verified:

* Ladder: `verify_training_stack.py` → **10/10**. The state step reports `2 suite(s), 33
  run(s), 43884 exact tensor digests compared; coverage audited against
  validation/training_state_coverage.json`; the topology step still audits its four
  FP32 decisions, and the probe still reports 6,368 dispatches / 2,980 barriers / loss
  5.5432177 unchanged, because the override is opt-in and the measured path never reads
  it.
* The packed family's six arms reproduce the FP16 trajectory exactly where they run:
  `one-row 3/3`, `one-row-cached 3/3`, `two-rows-wide 2/3`, `two-rows 1/3`, `four-rows
  1/3`, `one-row-rank128 0/3` (unavailable on this part, and the record says so).
* The coverage rule, failing end to end: the run before the record existed reported
  `no coverage recorded for this device` for both suites, and a doctored record claiming
  `column-slices` also ran in `odd_multi_sequence` plus a `ghost_case` failed with
  exactly those two messages while the suite's 15,268 digests still matched. The
  recorder refuses the same shrink: `refusing to record a shrunken entry` plus `pass
  --allow-coverage-shrink if losing this coverage is intended`.
* `--topologies` with no values runs only the shipped choice in both suites, passes, and
  says `coverage not audited` rather than claiming a check it did not perform; a value
  naming no variant at all still lists the seven the override accepts.
* Decision stability, five consecutive width-128 runs: `one-row-cached` 0.2718 / 0.2747
  / 0.2767 / 0.2727 / 0.2721 ms against `one-row` 0.2787 / 0.2787 / 0.2783 / 0.2830 /
  0.2789 ms, selecting the cached arm three times and `one-row` twice. The lead straddles
  the 2% material margin, which is why those geometries are not in the decision record.
* End to end, shipped versus `one-row` forced, two runs each: batch 2 361.2 / 376.3 ms
  against 367.6 / 381.0 ms, batch 32 923.0 / 926.2 ms against 930.2 / 941.4 ms. Never
  slower, and the difference is smaller than the repeat-to-repeat spread of the step, so
  the LN1/low-rank choice is documented as the kernel-level 12-17% the record measures
  rather than as an end-to-end speedup.
* `cargo test --lib` → 722 passed / 0 failed / 9 ignored (720 before, plus the FP16
  table's distinctness and the shared override vocabulary).
* `python -m unittest discover -s hierarchos-vulkan/validation -p 'test_*.py'` in
  `.venv-vulkan` → 93 tests OK, 1 skipped (74 before): the new cases pin the coverage
  entry's shape, an arm that stops running, an arm that becomes unavailable where it
  ran, a case that stops being measured, a case that compares less than it did, growth
  as a note, the recorder's refusal to shrink, and each suite's golden naming its own
  policy.
* Cost: the state step is now the ladder's longest by far -- 33 trajectory runs, about
  six minutes in debug -- because every arm replays every case of both goldens. Superseded
  by the 2026-10-16 pass below: 29 runs and 281.0 s with the same coverage, measured on the
  same machine.
# The parity step stops replaying what it cannot show, and the kernel goldens get the coverage rule -- 2026-10-16

Three changes are in this pass: the training-state parity step no longer pays for the cases
an arm cannot exist in, the dense/reduction/attention kernel goldens now fail when a device
stops exercising an invocation its record says it exercises, and the LN1/low-rank autotuners
only displace the structural default on a unanimous round. The third was already built --
the rule was in the code and the record carried the evidence -- so this pass measured it,
and what it measured is why the packed family stays out of the decision record.

## The trajectory step lost the runs that could not show anything

Every arm used to replay every case of both goldens. Some of those runs could only
re-confirm a verdict the record already held: `one-row-rank128` was forced in all three FP16
cases and reported unavailable in each, `two-rows` and `four-rows` were forced in the two
width-128 cases where only the wide pairing exists, and all three FP32 arms were forced in
the width-256 `odd_multi_sequence` case, where the FP32 producer is not in the shape at all
and the override changed nothing. A probe run of one case is 10-26 s of wall time, most of
it model load and pipeline setup the arm has no influence on.

The coverage record now says which cases an arm is absent from, and keeps the two reasons
apart because they are not the same claim: `unavailable` is the producer rejecting the
request on this device, `not_dispatched` is the fused producer never seeing it at that shape
at all -- the FP32 producer above rank 64, which is the width-256 case. Forcing a variant in
either kind of case measures the fallback kernel, and reporting that as an arm result is what
made those runs look like coverage while they were not.

So a forced arm now runs every case the record has it verified in, every case the record says
nothing about, and one **canary**: the widest absent case, replayed for a *single* update.
The canary is what keeps the claim device evidence instead of an assumption -- existence at a
shape is settled by the first dispatch, and a case the variant never ran in has no trajectory
to compare, so a full replay there buys nothing -- and the other absent cases are written as
`asserted` and carried from the record. If the canary disagrees, the variant ran where the
record says it is absent, so the record's claim is stale: that run then measures every
carried case of that arm in full instead of trusting it, which is what makes a device or a
gate change restore the coverage it just invalidated rather than passing on a stale record.

Two details keep the accounting honest. A shortened canary counts only the digests it
actually compared, so it cannot inflate the run's total, and the record's per-case counts
still come from the measured arm, which always replays every case in full. And the model
package is now shared by every arm of a case instead of being re-created per arm: the probe
initializes a package when it is missing and only ever reads it, which was verified directly
-- the package's content hash is identical before and after two consecutive six-update runs
(`110db052739d39257b8018c15757e660` both times) and the two snapshots are byte-identical
files -- and the same case takes **10.85 s from a fresh directory against 9.64 s from one
that already existed**.

Measured on this part, both suites, the two runs back to back in one session so the machine's
regime is shared: **33 runs / 346.97 s against 29 runs / 281.02 s (-19.0%)**, with the same
trajectories compared, the same coverage audited and every digest matching. The pre-change
side is measured the same way the change is: the verifier is pointed at a record whose
absence verdicts are all blank, which makes every arm replay every case exactly as it used
to, and it prints the old `2 suite(s), 33 run(s), 43884 exact tensor digests compared`. That
number also cross-checks the Adam copyback pass above, which measured the same step at
346.8 s against the 346.97 s of the all-cases run here. The shipped side reports `2 suite(s),
29 run(s), 32162 exact tensor digests compared`.

The row-level digest count falls 43,884 to 32,162 because a canary compares one update. That
is not lost evidence: in a case an arm never runs in, the fallback trajectory is compared in
full by the measured arm and by every arm that does dispatch there, so the canary only stands
in for a verdict about the *variant's* existence, and every run re-checks that verdict on
this machine.

## The kernel goldens now fail when a device stops exercising an invocation

The dense, reduction and attention harnesses already caught a changed kernel, because every
invocation's output digest is recorded per device. What they could not catch was a harness or
a device *stopping* to exercise an invocation -- a deleted case, or a part losing a kernel it
used to run -- because that only lowers the invocation count in the pass line while every
remaining digest still matches. That is the same gap the trajectory check closed, so the same
rule now covers these goldens, in `validation/kernel_golden_coverage.json`: one entry per
(golden, device), on this part `dense` 144, `reduction` 183 and `attention` 79, which is
exactly what the goldens hold digests for there.

A harness run fails when an invocation its device is recorded as exercising is no longer
exercised, when the golden holds a digest for that device the run did not reproduce, or when
the device has no entry at all; growth is a note. `--record-coverage` merges this machine's
coverage in and leaves every other device's entry alone, refusing to shrink one unless
`--allow-coverage-shrink` says the loss is intended. A `--only` run exercises one kernel on
purpose, so it reports `coverage was not audited` instead of a shrink. The audit and its merge
are pure -- no device, no Vulkan -- and live in `src/kernel_golden_coverage.rs` so
`cargo test --lib` checks them without a GPU.

Proven by making it fail first, on the harnesses themselves and not only in unit tests:

* a record claiming `adamw len=1 step=ghost` in `reduction` exits 1 with `Error: reduction on
  Intel(R) HD Graphics 520: invocation adamw len=1 step=ghost was covered and this run did not
  exercise it`, printed after that same run's `PASS: 183 GPU cases, all compared outputs
  bit-identical` -- the digests all still matched, which is exactly the hole the rule closes;
* the same record with this device's `reduction` entry removed exits 1 with `no coverage
  recorded for this device; run with --record-coverage on this machine to add its entry`;* a `dense` record with a ghost label exits 1 the same way (`invocation gemm rows=8 ghost
  was covered and this run did not exercise it`) and so does an `attention` one
  (`invocation forward_cache_1024 backward=ghost was covered and this run did not exercise
  it`), so the rule is wired into every harness rather than into the one that was drilled;
* all three harnesses exit 0 against the checked-in record and report `144`, `183` and `79`
  invocations audited.

## The LN1/low-rank rule is unanimous or nothing, and the FP16 family measures why

The autotuners pick between kernels by timing them, and the rule is that a candidate may only
displace the structural default by beating it by the 2% material margin in **every** one of
seven interleaved rounds. The round count is what makes the verdict stable: if each round's
winner is a fair coin flip, some candidate still wins every round about a quarter of the time
in three rounds and about one run in sixty-four in seven, so three rounds is not enough and
seven is.

The packed-FP16 family is the case that motivated it. Under the earlier median-only rule, five
consecutive width-128 runs selected `one-row-cached` three times and `one-row` twice, with a
0.6-3.6% median lead straddling the material margin. Thirty-five consecutive runs under the
rule -- thirty, then five -- all kept `one-row` instead, and the five were adjudicated on the
narrowest evidence of the set:

| run | `one-row` | `one-row-cached` | `two-rows-wide` | cached round wins (of 7) | selected |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.28326 ms | 0.27299 ms | 0.28329 ms | 4 | `one-row` (default kept) |
| 2 | 0.27960 ms | 0.27492 ms | 0.28535 ms | 3 | `one-row` (default kept) |
| 3 | 0.28728 ms | 0.27496 ms | 0.28233 ms | 3 | `one-row` (default kept) |
| 4 | 0.28150 ms | 0.27356 ms | 0.28562 ms | 5 | `one-row` (default kept) |
| 5 | 0.28495 ms | 0.27642 ms | 0.28319 ms | 4 | `one-row` (default kept) |

Every one of those is a 1.7-4.3% median lead that never held in all seven rounds, so the
producer records the default with its round wins and the narrowest per-round lead (`"rounds":
7, "round_wins": {...}, "decisive_margin_fraction": 0.0, "structural_default":
"ln-low-rank-fp16-one-row"`) rather than a variant this device does not actually reproduce.
`validation/verify_topology_decisions.py` audits that evidence as well as the choice: a
variant claimed after winning fewer than every round, a kept default that claims a round it
won, or a candidate that stops being timed at a geometry are each failures, and a record
written before the fields existed is audited for its choice alone.

The four FP32 decisions are unchanged and still unanimous where they are recorded
(`width128-batch2`, `width128-batch3-tail` and `width128-batch32` select `column-slices` at
12-17%, `width64-batch2` keeps `one-row`), so the rule costs the ladder nothing. And the
packed family is not in `topology_decisions.json` for a structural reason rather than a
stability one: the replay that writes that record runs the probe at the default FP32 precision
policy, where the packed producer is never dispatched, so it emits no `ln-low-rank-fp16`
decision line at all -- the replay's own scratch record is eight `ln-low-rank-f32` lines and
nothing else.

## Verified

* Ladder on this part: `validation/verify_training_stack.py` -> **10/10**, with the state step
  reporting `2 suite(s), 29 run(s), 32162 exact tensor digests compared; coverage audited
  against validation/training_state_coverage.json`. The dense, reduction and attention steps
  report `144`, `183` and `79` invocations audited against
  `validation/kernel_golden_coverage.json`; the topology step still audits its four FP32
  decisions; the probe still reports 6,368 dispatches / 2,980 barriers / loss 5.5432177.
* The runtime pair, one session, back to back: **33 runs / 346.97 s -> 29 runs / 281.02 s
  (-19.0%)**, every digest matching on both sides. Earlier sessions measured the same all-cases
  step at 402.7 s and 365.8 s, so the absolute seconds belong to the session while the run
  count and the ratio do not.
* Coverage is not merely audited but *earned*: the shipped run reports `one-row-rank128 0/1 +2
  carried`, `two-rows 1/2 +1 carried`, `two-rows-wide 2/3`, `four-rows 1/2 +1 carried` for the
  packed suite and `2/3 +1 carried` for each FP32 arm, so exactly the cases the record says the
  variant is absent in are the ones carried.
* The package is read-only across runs: content hash identical before and after two six-update
  runs, snapshots byte-identical, and 10.85 s versus 9.64 s for the same case from a fresh and
  an existing directory.
* Kernel-golden coverage, all three harnesses: exit 0 against the checked-in record;
  `reduction` exits 1 on a ghost invocation and on a missing device entry, `dense` and
  `attention` exit 1 on a ghost invocation each, with the 183 / 144 GPU cases still reporting
  bit-identical outputs in the failing runs and the attention harness's 79 invocations
  audited on the passing side.
* Decision stability: 35 consecutive width-128 runs (30, then 5) all kept the structural
  default, against 3 of 5 selecting the cached arm under the median-only rule; the FP32 record
  replays unchanged.
* `cargo test --lib` -> **736 passed / 0 failed / 9 ignored** (722 before, plus the coverage
  record's own tests). `python -m unittest` over the four affected validation modules ->
  **109 tests OK** (`test_training_state` 45, `test_topology_decisions` 27,
  `test_shader_provenance` 26, `test_training_stack` 11). A full `unittest discover` over
  `validation/` runs 114 and reports one pre-existing environment error --
  `test_peft_oracle_provenance` imports `torch`, which is not installed in this interpreter --
  plus one skip, so that module is not part of the numbers above.

## One coverage implementation, an RDNA3 module set, and the decode shape that got slower (2026-10-16)

This pass did three things: it gave both per-device coverage records a single implementation
and a deliberate first-run path, it added AMD RDNA3 kernels next to the Gen9 ones with the
portable module still universal, and it closed the one shape where the tuned tree was slower
than the artifacts it started from.

### The two coverage records are one implementation now

`validation/training_state_coverage.json` and `validation/kernel_golden_coverage.json`
answered the same three questions -- audit, merge, refusal -- in two languages, with two sets
of rules and two sets of tests. They now share `src/coverage_record.rs`: the envelope
(`schema`, `note`, `coverage`), the keyed audit (everything recorded still holds, growth is a
note, an unrecorded device fails), the merge that replaces only the entries this run measured,
the shrink refusal and its wording, and the bootstrap. Each record contributes one method,
`diff`, saying what a loss is for one of its entries -- invocation labels plus the digests the
golden holds for this device, or golden cases, per-case comparison counts and each arm's
verdicts -- and `summary` for the printed diff.

The Python driver kept only what is its own (which arm ran where, which cases a canary stands
in for). It asks the engine through a new `coverage_record` tool built beside
`training_step_probe`; `validation/test_training_state.py` lost the 20 tests that pinned the
Python copy (45 -> 25), and the shared suite in `src/coverage_record.rs` covers both models
plus the machinery: 26 tests.

`--bootstrap-coverage` is the deliberate first run on a device the record has never seen. The
default stays fail-closed -- no entry is a failure, because an entry is what makes the record
evidence -- and bootstrap creates *only* the entries this run is missing, leaves every existing
entry exactly as recorded, cannot shrink anything (so it has nothing to refuse), prints the
diff it wrote, and is mutually exclusive with `--record-coverage` and `--allow-coverage-shrink`.

Proven on the GPU: `matmul_kernel_validation` against a scratch record printed
`bootstrap added 1 coverage entry (0 already recorded)` and
`+ dense on 'Intel(R) HD Graphics 520': 144 invocation(s)`, passed, and audited clean on the
next run; without the flag the same run exited 1 with `... is missing; create it with
--record-coverage on this machine, or --bootstrap-coverage to create only the entries this run
is missing`. The trajectory driver bootstrapped one suite end to end:
`--suite fp32 --bootstrap-coverage` added `fp32 3 case(s), 3 arm(s)` and passed with
`1 suite(s), 12 run(s), 15268 exact tensor digests compared`. Tool-level checks: a scratch
record bootstrapped from one entry, audited clean, reported a shrunken case and a lost arm as
two problems, refused `mode: record` with the shared wording, wrote it with `allow_shrink`, and
a second bootstrap over the same record reported `every entry of this run already recorded; ...
nothing was written`.

### AMD RDNA3 kernels, with the portable module still universal

Four new modules -- `linear_forward`, `linear_bias_forward`, `linear_residual_forward` and
`linear3_forward`, suffixed `_amd_rdna3` -- stage a **64-wide** K slice (four k values per lane)
where the portable and Gen9 modules stage 32, which halves the barriers per k on a part with
64 KiB of LDS. Footprints: `16*65*4 + 64*17*4 = 8512` bytes for the three single-projection
families and `6*16*65*4 = 24960` for `linear3-forward`; the capability test decodes those
totals, `LocalSize` and the barrier count out of the committed `.spv` files rather than trusting
the constants. Workgroup, dispatch mapping, epilogues and the ascending-k `fma` chain are
unchanged, so the numeric contract is the portable one.

Selection is `vendor == AMD && is_amd_rdna3(device)`, classified by an AMD PCI device-ID table
(Navi 31/32/33 and the Phoenix/Hawk Point integrated parts), because Vulkan exposes no
architecture-generation property; RDNA1/RDNA2 IDs are pinned as negative controls, and every
other vendor and AMD generation keeps the exact portable bytes. The per-dispatch gate is the
portable modules' own tiled-vs-scalar threshold (`rows >= 8, k >= 64, n >= 16`), because no
RDNA3 crossover has been measured here -- there is no RDNA3 adapter on this machine. That is
recorded as a structural tuning, not a measured win.

What is proven here: all four modules are executed on this Gen9 adapter against the portable
module at the same six geometries the Intel A/B uses, **24 comparisons, every one
`max_ulp_delta=0`**. The RDNA3 timing that run prints is not RDNA3 evidence: on Gen9 the 64-wide
stage loses where k is long (0.71-0.73x at 8x448x512) and wins on `linear3-forward` (2.52x at
8x448x512, 1.77x at 6x96x320), which is why the two families keep different modules and the
selection is per vendor.

### The deliverable against the artifacts it started from (measured)

`A` is the module from the initial publish (`git show a100338:.../<family>.spv`), `B` is the
module this machine selects today; same adapter, same session, 20 timed dispatches after 3
warmups, median, every row bit-exact. Queue timestamps, so the ratios are host-build
independent (these rows were measured in a debug binary).

| Family | Geometry | Pre-tuning | Today | Speedup |
| --- | --- | --- | --- | --- |
| `linear-forward` | 16x448x2048 | 5.7041 ms | 0.9197 ms | 6.20x |
| `linear-forward` | 8x448x512 | 0.7595 ms | 0.2277 ms | 3.34x |
| `linear-forward` | 6x96x320 | 0.0917 ms | 0.0436 ms | 2.10x |
| `linear-bias-forward` | 8x448x512 | 0.7680 ms | 0.2254 ms | 3.41x |
| `linear-residual-forward` | 16x448x2048 | 5.7380 ms | 0.9217 ms | 6.23x |
| `linear-residual-forward` | 8x448x512 | 0.7663 ms | 0.2256 ms | 3.40x |
| `linear3-forward` | 16x448x2048 | 17.0464 ms | 2.3921 ms | 7.13x |
| `linear3-forward` | 8x448x512 | 2.2253 ms | 0.6355 ms | 3.50x |

The portable path -- every vendor without a tuned module -- also beats the pre-tuning artifacts
at training geometry (6.24x at 16x448x2048 and 3.34x at 8x448x512 for `linear-forward`), but it
lost single-row decode: 0.1839 ms against the pre-tuning 0.1502 ms at `1x448x512` (**0.82x**),
because the tiled module's `shared` declaration is reserved even when the scalar path runs.

That regression is closed rather than documented: `VendorMatmulKernel` now holds the pre-tuning
scalar artifact for each matmul family (`shaders/portable_scalar/*.spv`, byte-for-byte the
initial-publish modules -- vendored in a nested directory because `build.rs` refuses two shaders
with one SPIR-V signature) and dispatches it below `SCALAR_MATMUL_MAX_ROWS = 8`, the portable
modules' own threshold. Selection order is vendor variant first (its gate is that part's
measured crossover), then scalar below eight rows, then the tiled portable module. The shape
measures **0.1502 ms, 1.22x against the tiled portable, parity with the pre-tuning module**, and
`scalar_schedule_reproduces_the_portable_kernels` executes the scalar modules against the
portable ones on the executing adapter (raw-bit, every geometry) so a stale copy cannot ship.

### Verified

* Ladder on this part: **10/10**, state step `2 suite(s), 29 run(s), 32162 exact tensor digests
  compared; coverage audited against validation/training_state_coverage.json` (audited through
  the shared tool), dense/reduction/attention at 144/183/79 invocations, probe unchanged at
  6368 dispatches / 2980 barriers / loss 5.5432177.
* `cargo test --lib` -> **761 passed / 0 failed / 9 ignored** (26 of them the shared coverage
  suite; the 24 RDNA3 A/B comparisons and the scalar-schedule A/B run in the same suite).
* `python -m unittest test_training_state test_topology_decisions test_shader_provenance
  test_training_stack` -> **89 tests OK**; the coverage-semantic half moved to Rust.
* One caveat measured, not hidden: the first full ladder run in this session failed the topology
  step with `recorded ln-low-rank-f32-column-slices but this device now selects
  ln-low-rank-f32-one-row` immediately after the five-minute parity step; two standalone
  replays and a second full ladder run were green. The decision rule needs all seven rounds to
  agree, so a post-parity thermal regime can still flip a 2% margin.

### Packed-cell fusion removal, linear3 coverage and a one-pass workspace gate

The packed-cell fused channel/state kernel was measured once more on its last
plausible niche, the TBPTT forward-only transition: at batch 8 the composition
was `1.664 ms` against `1.870 ms` fused (`0.890x`), and `forward_backward` was
`3.871 ms` against `4.039 ms` (`0.958x`), bit-identical in both runs. The fused
kernel, its four SPIR-V variants, its workgroup switch, the A/B bench and its
receipt were removed; `validation/PACKED_CELL_COMPOSITION.md` records the final
table. The remaining packed cell composes its channel mix from the vendor
`linear-forward`/`linear-residual-forward` modules, and its non-fused time-mix
projection from `linear3-forward`. A new
`HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION`
switch forces that time-mix composition, and
`validation/verify_rwkv_packed_cell_composition.py` now runs vendor-vs-portable
over three steps and asserts all three dispatch families; the CPU bridge still
passes independently at its usual `7.6e-6` worst drift.

`check-workspace.ps1` builds a reusable workspace from directory junctions and
runs `cargo check --workspace --bins` in one pass: **4 crates, 55 binaries**,
with a warm run around one minute.

### Time-mix projection A/B and a geometry-aware promotion

The packed cell's time-mix projection still had a fused projection/key/state
kernel next to the `linear3-forward` composition. The three-leg bench
(`hierarchos-vulkan-rwkv-packed-cell-time-mix-bench` plus
`validation/bench_rwkv_packed_cell_time_mix.py`) runs fused-forced,
composition-forced and production-default graphs over one fixture, interleaved
with a rotated order, and records each leg's reported graph plus its vendor
`linear3-forward` dispatch count. All five returned tensors were bit-identical
(`0.0`) across all six batch sizes, so the choice is scheduling only.

| batch | fused | composed | ratio | tuned `linear3-forward` | default |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.592 | 2.617 | 1.009 | no | fused |
| 2 | 2.795 | 2.819 | 1.009 | no | fused |
| 4 | 3.232 | 3.253 | 1.006 | no | fused |
| 8 | 4.067 | 3.979 | 0.978 | yes | composed |
| 16 | 6.382 | 6.159 | 0.965 | yes | composed |
| 32 | 11.095 | 10.647 | 0.960 | yes | composed |

The composition wins exactly where the tuned module is admitted and the fused
kernel wins where the composition falls back to the portable module, so the
promotion is the vendor layer's own geometry gate, not a new constant:
`prefer_composed_projection` asks the `linear3-forward` slot whether it would
select its variant for the dispatch geometry. Both outcomes are bit-identical,
so no golden state moved. `HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION` keeps
the fused kernel at every batch for A/B work.

`validation/test_rwkv_time_mix_receipt.py` enforces the contract on the receipt:
the default must compose wherever the composed leg dispatched the tuned module,
keep the fused kernel elsewhere, and the per-group timing medians must favor the
chosen graph. The live leg (enabled by `HIERARCHOS_RUN_GPU_VALIDATION=1`)
re-runs the bench and fails on a promotion or parity regression.

### The promotion read end to end, recorded per device, and gated in CI

A single `forward_backward` is not the unit a training run pays for, so the same
A/B was re-read through a whole 8-step TBPTT window
(`hierarchos-vulkan-rwkv-tbptt-time-mix-bench` plus
`validation/bench_rwkv_tbptt_time_mix.py`, 8 samples per leg, rotated order).
Every returned tensor over the whole sequence stayed bit-identical (`0.0`).

| batch | fused window | composed window | ratio | per step | default |
| --- | --- | --- | --- | --- | --- |
| 1 | 21.685 | 21.551 | 0.994 | 2.711 -> 2.694 | fused |
| 4 | 25.974 | 25.988 | 1.001 | 3.247 -> 3.248 | fused |
| 8 | 31.820 | 31.217 | 0.981 | 3.977 -> 3.902 | composed |
| 16 | 49.548 | 47.348 | 0.956 | 6.194 -> 5.919 | composed |

The window keeps the split (`0.956x` at batch 16 is `2.2 ms` of a `49.5 ms`
sequence, `0.28 ms` per step) and turns the small-batch result into an honest
tie: the composition is not faster there, and the fused kernel is not visibly
faster either. The record therefore keeps the fused kernel at those geometries
and the receipt test asserts "the composition does not win", not a win the
window does not show.

The static gate rule became a measured per-device record.
`validation/time_mix_projection_decisions.json` holds one entry per
(adapter, width, batch) with `compose`, the ratio that produced it and
`source: measured`; `src/rwkv_time_mix_decisions.rs` compiles it in, so the
shipped decision and the checked-in record cannot drift. The runtime still asks
the vendor gate first (the composition only wins as a tuned module) and then the
record decides that device's crossover; an unrecorded adapter keeps the gate
rule. Recording is offline (`--record-decisions`), not a hot-path autotune, and
`verify_time_mix_projection_decisions.py` replays all six entries on the
attached adapter and checks the shipped default builds the graph the record
names.

Both A/Bs now fail closed when the default stops following the record, and
`.github/workflows/time-mix-ab.yml` runs them, the replay and both receipt
suites on a self-hosted Vulkan runner, so the rule is measured on hardware and
not only against a checked-in JSON file.

### Full ladder on the promoted default

`verify_training_stack.py` passed **12/12** in debug and **11/11** in release
(`--skip-build`, release bins built first, exit 0). Receipts that moved:

* `training_step_probe/step_expectation.json` (Intel entry): 6368 -> **6512
  dispatches**, 2980 -> **3020 barriers** (bound 3076 unchanged), `mean_loss`
  **unchanged** at 5.5432177. The 144 extra dispatches are the promotion: the
  composed projection dispatches `linear3-forward` plus the key/state kernels
  where the fused kernel was one dispatch, over 72 layer-timesteps.
* `oracle_divergences.json`: five `state: "removed"` entries for the retired
  packed-cell channel-mix/state kernel and its four artifacts.
  `check_oracle_parity.py` now supports and validates removals (a `removed`
  entry fails when the file comes back, a `modified` one when it stops
  differing), so a retirement is recorded rather than worked around.
  Oracle: 805 portable files, 3 documented divergences, 5 retired, 20 only here.
* Shader provenance: 467 -> **463** variants, four artifacts retired with that
  kernel.

Unchanged: kernel goldens (144 dense / 183 reduction / 79 attention), training
state (2 suites, 29 runs, 32162 exact digests), topology decisions (4 entries),
push-constant layouts, binding-access audit. One flake is recorded rather than
hidden: the first release pass reported the `ln-low-rank-f32` decision moving
from `column-slices` to `one-row` while an abandoned background ladder was still
running on the adapter; the same step replayed and the clean second pass both
passed with all four decisions matching. The autotuner's 2% margin can react to
GPU contention; the recorded decision did not change.

The remaining fused forward producers were audited against their compositions
and ranked (`validation/FUSED_FORWARD_PRODUCER_AUDIT.md`): the packed path's
channel-mix producer (one existing dispatch replacing two), the LN1/low-rank
full-forward producer (one replacing the staged chain), and
`rwkv_key_state_forward_fused` (one replacing two). No A/B has been run for
those yet; the audit ranks them by expected payoff and states that explicitly.
