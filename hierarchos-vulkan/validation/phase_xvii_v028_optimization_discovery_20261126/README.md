# Phase XVII — v0.2.8 optimization discovery and first candidate experiment

**Result: GO for the query-pass dQ accumulator, with a measured complete-step
gain of 1.0608× on the FP32 arm and 1.0762× on the opt-in Mixed arm, bit-exact
against the shipped v0.2.7 artifact on every case tested.** Nothing was promoted:
`--wmma-mode` still defaults to `off`, no production shader, selector, threshold,
default or precision policy was modified, and the only source change is 27 lines
inside `#[cfg(test)]`.

The leading hypothesis in the brief (Candidate A, attention-backward query-pass
computation reuse) was investigated and **not** selected. The fresh profile and
the shipped SPIR-V show that the query pass is dominated by something else: its
`dq[128]` accumulator is a dynamically indexed function-storage array, so the
backend places it in scratch memory, and the per-key loop touches it 16,384 times
per lane. Removing the repeated score and value-gradient products would delete
arithmetic from a kernel that is not arithmetic-bound. The experiment therefore
targets the accumulator instead, and Candidate A is retained as the strongest
follow-on (see §11).

---

## 1. Repository baseline

| item | value |
| --- | --- |
| base branch / commit | `main` `9b0091e02e976441e343d6c445101f9c29395bd6` |
| `v0.2.7` annotated tag object | `85b64a1eb9be1e925e791c1c29b69ced74e23404` |
| `v0.2.7^{commit}` | `9b0091e02e976441e343d6c445101f9c29395bd6` |
| research branch | `research/v0.2.8-optimization-discovery` (created at that commit) |
| working tree at start | clean; no staged changes; no untracked files |
| `stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` (`pre-v0.2.6 untracked validation evidence`) |
| `rustc` / `cargo` | 1.93.0 (254b59607 2026-01-19) / 1.93.0 (083ac5135 2025-12-15) |
| `glslc` / `spirv-val` / `spirv-dis` | VulkanSDK 1.4.321.1 |

`git rev-parse v0.2.7` returns the **tag object** `85b64a1e…`, not a commit. That
is a two-line annotated-tag indirection, not a moved tag: dereferencing gives
exactly the published commit `9b0091e…`. Any audit that compares `rev-parse`
output directly against a commit id will report a false mismatch here.

**Device.** `AMD Radeon Graphics`, vendor `0x1002`, device `0x15bf`, integrated,
driverID `DRIVER_ID_AMD_PROPRIETARY`, `driverInfo 26.20.15.14 (LLPC)`, Vulkan
instance 1.4.357, device API 1.4.359, `subgroupSize 64`,
`VK_KHR_cooperative_matrix` with a `SUBGROUP`-scope 16×16×16
f16×f16→f32 configuration, `shaderFloat16 = true`, `maxComputeSharedMemorySize
32768`.

This is the **same adapter, PCI id and driver string** the Phase XIII receipt
records, and the fresh profile reproduces it to within 0.6% (§2), so the numbers
in this phase are a matched baseline rather than a cross-adapter substitution.

---

## 2. Fresh baseline

### 2.1 Profiled attribution (9 steps per arm)

`HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`,
`raw/50_profile_off.log`, `raw/51_profile_mixed.log`, reduced by the Phase XIII
`attribute.py` (copied unchanged) into `raw/52_attribution.json`. Fixture:
`target/gqa-e2e/fixture-8q2kv`, 2 layers, hidden 1024, 8 query heads × 128, 2 KV
heads × 128, batch 2 × seq 64 (1024 attention rows), LoRA rank 8 / alpha 16,
`unfused_qk = 9`, causal.

| shader | FP32 ms/step | Mixed ms/step | FP32 dispatch | Mixed dispatch |
| --- | ---: | ---: | ---: | ---: |
| `transformer_attention_backward_query_rdna3` | **8.417** | **9.256** | 2 | 2 |
| `linear_forward` | 8.247 | 0.023 | 17 | 2 |
| `linear_input_grad` | 7.406 | 0.009 | 17 | 2 |
| `wmma_gemm_f16_f32acc` | — | **7.380** | 0 | 30 |
| `fp32_to_fp16_native` | — | **1.186** | 0 | 45 |
| `transformer_attention_forward_rowpacked64_cached_rdna3` | 4.733 | 5.065 | 2 | 2 |
| `transformer_rms_norm_forward` | 1.966 | 2.079 | 5 | 5 |
| `transformer_rms_norm_input_grad` | 1.716 | 1.840 | 5 | 5 |
| `falcon_h1_cross_entropy_lanes16` | 1.521 | 1.629 | 1 | 1 |
| `transformer_attention_backward_key_channel4_rdna3` | 0.572 | 0.576 | 2 | 2 |
| `linear_forward_lane4` | 0.216 | 0.062 | 2 | 1.33 |
| `linear_input_grad_group4_muladd` | 0.131 | 0.138 | 2 | 2 |
| `linear_weight_grad` | 0.058 | 0.062 | 4 | 4 |
| `vector_add` | 0.220 | 0.183 | 16 | 16 |
| everything else | 0.088 | 0.084 | — | — |
| **attributed total** | **35.511** | **29.775** | 143 | 143 |

Phase XIII published 35.490 / 29.623 for the same two arms; every row is within
noise of its published counterpart (query 8.411→8.417, forward 4.725→4.733,
`wmma` 7.411→7.380, conversion 1.169→1.186).

Shares of the Mixed step: query pass **31.1%**, dense projection stack **29.8%**
(`wmma` 7.380 + conversion 1.186 + 0.29 residual FP32 adapter), attention forward
**17.0%**, RMSNorm fwd+grad **13.2%**, cross-entropy 5.5%, channel-4 key pass
1.9%, everything else 1.5%.

### 2.2 Memory, scratch and conversion structure

* Two-pass backward scratch: `65536` floats per buffer × 2 buffers =
  **524,288 bytes per layer**, 1 MiB per step, reported per layer by the route
  line (`scratch_elements=65536 scratch_bytes=524288`).
* `fp32_to_fp16_native` is **45 dispatches/step**: 30 at `grid=512x1x1`
  (the 15 weight mirrors, converted once per projection and reused by that
  projection's input-gradient GEMM) and 15 at `grid=128x1x1` (activations,
  one per projection). Weight mirrors are therefore ~2/3 of conversion time by
  grid volume, not the 1/3 a dispatch count suggests.
* `wmma_gemm` is **30 dispatches/step** (15 forward + 15 input-gradient; LoRA
  records no dense base-weight gradient because the base is frozen).
* Attention route is identical in both arms: `forward_row_packed=2`,
  `backward_query=2`, `backward_key_channel4=2`, `backward_key_original=0`,
  `backward_scalar=0`.
* WMMA workgroups are **16 per query dispatch** (`div_ceil(1024, 64)` with
  `local_size_x = 64`): sixteen 64-lane waves for the whole device.

### 2.3 Unprofiled complete step (the performance baseline)

`raw/s7-{off,mixed}-r*-p*-{A,B}.log`, three warmup steps, 26 measured steps per
process, five rounds of `A B B A`, 20 separate release-mode processes, arm A =
the shipped v0.2.7 query artifact. Reduced by `reduce_step.py` into
`raw/63_abba.json`.

| arm | median complete step | 26-step stdev | Phase XIII published |
| --- | ---: | ---: | ---: |
| `--wmma-mode off` | **28.477 ms** | 0.41 ms | 28.481 ms |
| `--wmma-mode mixed` | **23.035 ms** | 0.74 ms | 23.001 ms |

---

## 3. What the production query shader actually does

`shaders/transformer_attention_backward_query_rdna3.comp`, 300 lines, one lane
per `(batch, query head, query position)`, `local_size_x = 64`. The lane makes
**four serial traversals of the key sequence**:

| pass | computes | per visible key |
| --- | --- | --- |
| 1 | row maximum | `score()` = 128-element dot |
| 2 | softmax normalizer | `score()` again |
| 3 | `softmax_dot` and `kv_softmax_dot` | `score()` **and** `value_dot()` |
| 4 | dQ accumulation + scratch stores | `score()` **and** `value_dot()` |

So per visible key the lane performs **4 × 128 + 2 × 128 + 128 = 896** dependent
FMA-class operations, of which 384 (the three repeated `score()` calls) and 128
(one repeated `value_dot()`) are arithmetically removable. That is Candidate A's
premise, and it is real: **57% of the arithmetic is recomputation.**

### 3.1 Why that is not the binding constraint

The device issues 16 waves for this kernel. Each lane's `dq` accumulator is
declared `float dq[MAX_HEAD_DIM]` and indexed with `d`, a loop variable bounded
by the push constant `qk_head_dim`, which is not a compile-time constant. The
shipped artifact reflects that:

```text
$ spirv-dis shaders/transformer_attention_backward_query_rdna3.spv | grep arr_float_uint_128
%_arr_float_uint_128          = OpTypeArray %float %uint_128
%_ptr_Function__arr_float_uint_128 = OpTypePointer Function %_arr_float_uint_128
%1297 = OpVariable %_ptr_Function__arr_float_uint_128 Function
%1299 = OpAccessChain %_ptr_Function_float %1297 %5820   <- dynamic index
```

A 128-float function-storage array with dynamic indices cannot be promoted to
registers, so it is lowered to scratch ("private") memory. Private memory is
per-lane strided: one access from a 64-lane wave becomes 64 separate
transactions, exactly the opposite of what the K/V reads get (see §4,
Candidate B). Inside the key loop the array is touched `2 × 128 × 64 = 16,384`
times per lane, against 768 K/V reads per lane over the same loop.

The measured arithmetic rate corroborates this: 1024 rows × 64 keys × 896 ops ≈
58.7 M FMA-class operations in 4.6 ms per dispatch is ≈ 26 GFLOP/s on a part whose
FP32 peak is in the TFLOP/s range. The kernel is not short of arithmetic; it is
short of useful memory transactions and of latency-hiding waves.

Reproducing the same mechanism on a kernel where it *was* measured: Phase XI/XII
built the key pass with `XI_UNROLL_CHANNEL=1` specifically so that "the
accumulator arrays are promoted out of function storage", and that was the
configuration promoted (`shaders/transformer_attention_backward_key_channel4_rdna3.comp`).

---

## 4. Candidate ranking

Shares are of the measured complete step (§2.1). "Modeled" values are Amdahl
projections from a measured share and a *measured* kernel ratio; they are labeled
and are not measurements.

**Candidate B (GQA K/V reuse) is ranked last on evidence, not on effort.** The
shader maps `row = ((b * num_heads + head) * query_seq_len) + query`, and
`query_seq_len = 64` equals the wave width, so **every 64-lane wave is exactly
one `(batch, head)` covering all 64 query positions of that head**. All lanes in
a wave read the *same* `k_base` and `v_base`. The production kernel therefore
already broadcasts one K/V row to the whole wave; a shared-memory tile or a
query-head workgroup cannot beat a hardware broadcast, and "four logical reads per
KV head" is not four memory transactions. This also explains why Phase VIII's
destination-key partitioning was rejected and why it is not reopened.

| # | Candidate | Bottleneck (measured) | Redundant work | Expected complete-step benefit | Cost | Numerical risk | Memory | Hardware scope | Qualification cost | Confidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **C′ query dQ accumulator register promotion** (the candidate that was built and measured) | query pass 8.417 / 9.256 ms = 23.7% / 31.1%; 16 waves | 16,384 scratch accesses per lane from a dynamic-index private array | **measured 1.0608× (FP32) / 1.0762× (Mixed)**; modeled 1.062× / 1.082× before measuring | Low: one shader, same bindings, same dispatch, same push block | **None observed: bit-exact, 10/10 isolated cases, 9/9 in-graph steps** | none new; unchanged 1 MiB scratch | head_dim 128 (both qk and value); any RDNA2/3 wave64; no cooperative matrix needed | bit-exact vs shipped + complete-step ABBA on at least one other geometry | **High** |
| 2 | F Mixed WMMA staging (C1-style load batching, K-slice order, tile/occupancy) | `wmma_gemm` 7.380 ms = 24.8% of the Mixed step | operand reload and staging inside one cooperative GEMM | up to 1.05–1.10× **on the opt-in Mixed arm only**; 1.00× on the default FP32 arm | High: cooperative-matrix shader, 37-gate contract, three routes (fwd/dW/dX) | Medium–High: same accumulation type but any staging change re-opens the gate contract | shared-memory staging, no new global buffers | needs the 16×16×16 f16→f32 cooperative configuration | full 37-gate requalification + three-phase graph test + ABBA on both arms | Medium |
| 3 | A query-pass scratch reuse (retain score and `value_dot`) | same kernel; after C′ the remaining steady-state cost is 5.05 ms/step (Mixed) | 3 of 4 `score()` passes and 1 of 2 `value_dot()` per key = 57% of the arithmetic | modeled 1.03–1.05× complete step **if** the kernel is arithmetic-bound after C′; genuinely unknown, could be ~0 | Medium: rewrite all four traversals, keep two distinct softmax dots, add scratch or private storage | Low–Medium: arithmetic is preserved elementwise but the two softmax-dot reductions must stay separate | +8 bytes/(row,key) if backed by global scratch, or +512 B/lane if private | same as C′ | bit-exact vs shipped, both modes, masked/skipped/padded rows | Low–Medium |
| 4 | D frozen LoRA base-weight mirror retention | conversion stack 1.186 ms = 4.0% of the Mixed step; 15 of 45 conversion dispatches at `grid=512` | base weights are immutable under LoRA, yet every optimizer epoch re-converts all 15 masters | modeled 1.02–1.03× **on the Mixed arm only** (≈0.8 ms/step at most); 1.00× on FP32 | Low–Medium: distinguish immutable from trainable mirrors, keep invalidation | Medium: a stale mirror is silently wrong; allocation identity + load/reload invalidation must be proven | none (retains existing mirrors longer) | Mixed only | proof that no reload/checkpoint path can serve a stale mirror; 29-step bitwise A/B | Medium |
| 5 | G attention-forward multi-query reuse | forward 4.733 / 5.065 ms = 13.3% / 17.0% | one lane per query row already caches its key-dependent state | modeled 1.02–1.04× if K/V reuse across rows were free; the 7×-slower streaming candidate shows the design space is tight | High: new mapping, register footprint, mask/causal handling | Medium | shared tiling | RDNA3 wave64 row-packed route | must beat the cached row-packed kernel outright, not a strawman | Low |
| 6 | H RMSNorm + activation-conversion fusion | RMS fwd + grad 3.682 / 3.919 ms = 10.4% / 13.2% | 10 dispatches/step; the forward could emit its FP16 mirror in the same pass | modeled ≤1.01× | Medium: touches architecture-specific norm semantics | Medium–High: strict order / model-specific norm behaviour | one extra output binding | Mixed only (the mirror exists only then) | per-architecture norm parity | Low–Medium |
| 7 | E shared activation conversion across projections | 15 activation conversions = ~1/3 of 1.186 ms = 1.3% | Q/K/V share one normed activation; gate/up share one | modeled ≤1.003× (saves ~6 of 30 activation dispatches ≈ 0.08 ms/step) | Low | Low | graph-level cache keyed on contents, not allocation | Mixed only | lifetime/identity proof that the same bytes are still there | Medium (small) |
| 8 | B GQA K/V reuse | 8.417 / 9.256 ms | none remaining: the wave already broadcasts one K/V row to 64 lanes | ≈1.000× | — | — | — | — | — | **High that there is nothing here** |

Ordering rationale. C′ wins because it is the only candidate that (a) attacks the
largest or second-largest measured item, (b) removes a *mechanism* rather than
reordering work, (c) was measured to be bit-exact with zero tolerance changes, and
(d) leaves production dispatch, precision policy and fallbacks untouched. F has a
comparable share but is opt-in-only, carries a full requalification contract, and
touches the route the release does not default to. A is the natural next step but
its payoff is contingent on what C′ leaves behind: it deletes arithmetic from a
kernel that was not arithmetic-bound, so the honest expectation before measuring
is "somewhere between nothing and modest".

---

## 5. Selected experiment

### 5.1 Hypothesis

> On the admitted head-dimension-128 geometry, indexing the `dq` accumulator
> only with compile-time literals lets the backend promote it out of
> function storage into VGPRs. The per-key loop then stops issuing 256
> per-lane-strided scratch accesses per key, and the query pass gets faster
> without any change to its arithmetic.

Null hypothesis: the extra register pressure, code size (109 KB vs 48 KB of
SPIR-V) and occupancy cost cancel the scratch saving.

### 5.2 Production files inspected (none modified for the production path)

| file | what was taken from it |
| --- | --- |
| `shaders/transformer_attention_backward_query_rdna3.comp` | the arithmetic contract, the four traversals, both softmax-dot reductions, the zeroing/skipping rules — copied statement for statement |
| `shaders/transformer_attention_backward_query_rdna3.spv` | the arm every measurement is against; also `sha256` `f9652f52…` |
| `shaders/transformer_attention_backward_key_channel4_rdna3.comp` | the literal-index repeat-macro idiom (`XI_UNROLL_CHANNEL`) reused as `QX_R*` |
| `src/transformer.rs` | `AttentionPush`, the query dispatch grid `[div_ceil(rows,64),1,1]`, `two_pass_attention_scratch_elements`, `TwoPassAttentionPass::spirv()` |
| `src/vulkan.rs` | `ComputeKernel::new`, `record_dispatch`, `time_compute_batch_ms` |
| `validation/phase_xiii…/attribute.py` | the profile reducer, copied unchanged |
| `validation/phase_xi…` | the isolation pattern: candidate SPIR-V in the validation tree, `#[cfg(test)]` module as the only dispatcher |

### 5.3 Experimental files created

| path | role |
| --- | --- |
| `validation/phase_xvii…/shaders/transformer_attention_backward_query_rdna3_dqunroll.comp` | the candidate source |
| `validation/phase_xvii…/shaders/q_dqunroll.spv` | candidate artifact (109,300 B) |
| `validation/phase_xvii…/shaders/q_rebuild.spv` | **control**: the unmodified production source compiled with the experiment's exact `glslc` invocation |
| `src/phase_xvii_v028_optimization_discovery_tests.rs` | the whole dispatch path, `#[cfg(test)]` |
| `validation/phase_xvii…/profile_baseline.sh`, `run_abba_step.sh`, `reduce_step.py`, `attribute.py` | drivers and reducers |
| `validation/phase_xvii…/raw/` | every log, artifact, reduction |

Source change to `src/transformer.rs`: **27 lines, all inside `#[cfg(test)]`** — a
test-only module declaration, and a seam that returns the candidate bytes when
`HIERARCHOS_PHASE_XVII_QUERY_SPV=dqunroll`. Unset, `TwoPassAttentionPass::spirv()`
returns `ATTENTION_BACKWARD_QUERY_PARALLEL_SPV` exactly as before; a production
build does not compile the seam at all.

### 5.4 Design

The candidate is the production shader plus one structural change: on
`qk_head_dim == 128 && value_head_dim == 128`, the dQ initialization, the
accumulation and the store are expanded over integer literals with a repeat-macro
list (`QX_R128(QX_INIT)`, `QX_R128(QX_ACC0)`, `QX_R128(QX_ACC9)`,
`QX_R128(QX_STORE)`). Every other geometry falls through to a byte-for-byte copy
of the production body, so the artifact is a superset rather than a replacement.

Arithmetic is untouched:

* mode 0 accumulates `dq[i] += pc.scale * ds_q * k_values[k_base + i]`, the same
  left-associated expression the production loop contains, so the compiler
  contracts it the same way;
* mode 9 keeps `precise float score_scale = pc.scale * 1.0; score_scale *=
  ds_q;` once per key, then `precise float contribution = score_scale *
  k_values[...]; dq[i] += contribution;` per element, in the same order;
* the four traversals, `softmax_dot` (plain, for dQ) and `kv_softmax_dot`
  (`precise`, for dK/dV) are **not** merged and **not** cached — no score or
  value product is reused anywhere, so nothing measured here belongs to
  Candidate A;
* exponentiation, reciprocal, masked-key skipping, skipped-row scratch clearing,
  the pre-loop `grad_q` zeroing and the store order are unchanged.

Memory behaviour: the accumulator is the same 128 floats; on the admitted
geometry they are SSA values rather than scratch slots. Bindings (9), push
constants (`size_of::<AttentionPush>()`), dispatch grid (16×1×1), dispatch count
(2/step), scratch allocation (65,536 × 2 floats/layer) and the second-pass key
shader's scratch semantics are all unchanged — pass 2 reads exactly the slots
pass 1 writes, in the same representation.

### 5.5 Compiler and shader validation

```text
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders
  ..._dqunroll.comp -o q_dqunroll.spv          # exit 0
spirv-val --target-env vulkan1.2 q_dqunroll.spv  # exit 0
```

The control proves the flags are the production flags: `q_rebuild.spv`
(production `.comp`, this invocation) and the shipped
`transformer_attention_backward_query_rdna3.spv` differ by **0 lines** of
`spirv-dis` output (`raw/58_control_diff.txt`). The candidate still contains one
`_arr_float_uint_128` variable — that is the untouched generic fallback body —
while the specialized body indexes only by `OpConstant`.

### 5.6 Numerical results

`raw/60_parity.log`, ten cases, `--test-threads=1`, sentinel-free fresh buffers:

| case | heads | kv | mode | seq | head_dim | causal | masks | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gqa8q2kv_m0_causal | 8 | 2 | 0 | 64 | 128 | 1 | all | PASS |
| gqa8q2kv_m9_causal | 8 | 2 | 9 | 64 | 128 | 1 | all | PASS |
| mha8q8kv_m9_causal | 8 | 8 | 9 | 64 | 128 | 1 | all | PASS |
| mha8q8kv_m0_noncausal | 8 | 8 | 0 | 64 | 128 | 0 | all | PASS |
| gqa8q2kv_m9_masked | 8 | 2 | 9 | 64 | 128 | 1 | dropped query + key | PASS |
| gqa8q2kv_m0_masked_noncausal | 8 | 2 | 0 | 64 | 128 | 0 | dropped query + key | PASS |
| gqa4q2kv_m0_seq37 | 4 | 2 | 0 | 37 | 128 | 1 | dropped | PASS |
| gqa4q2kv_m9_seq37 | 4 | 2 | 9 | 37 | 128 | 0 | all | PASS |
| gqa8q2kv_hd64_m0 (fallback) | 8 | 2 | 0 | 64 | 64 | 1 | dropped | PASS |
| gqa8q2kv_hd64_m9 (fallback) | 8 | 2 | 9 | 64 | 64 | 1 | all | PASS |

Each case compares **three** tensors (`dQ`, `scratch_probability`, `scratch_ds`)
twice: the rebuilt control against the shipped artifact, and the candidate against
the shipped artifact. Every comparison is **bitwise** (`to_bits()` equality, no
tolerance), for 60 tensor comparisons, 0 divergences, 0 non-finite values. `seq
37` exercises a partial final workgroup (296 rows → 5 workgroups) and a boundary
key length; the masked cases cover both skipped-row paths, including the row whose
`query_mask` drops it after `grad_q` has already been zeroed.

`head_dim 64` is included to prove the artifact degrades to the production body
rather than silently doing something else.

End-to-end numerics (`raw/65_profile_{A,B}.log`): `loss_bits` are **identical on
all 9 steps** between the shipped arm and the candidate arm
(`40e44309, 40e43f10, 40e30547, 40e757ac, 40e5465d, 40e437d3, 40df8192,
40e11b1a, 40e25bf7`). Only `step_ms` differs. Determinism and checkpoint resume
therefore hold by construction: the candidate's outputs are bit-identical to the
shipped ones, so no tolerance, no resume probe and no per-element master
comparison is required or claimed.

### 5.7 Performance results

**Isolated kernel** (`raw/61_timing.log`, 8 back-to-back dispatches per sample,
median of 7 ABBA rounds, GPU timestamps):

| mode | production | candidate | ratio | speedup | rounds improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | — | — | 0.7044 | **1.4197×** | 7/7 |
| 9 | — | — | 0.7547 | **1.3251×** | 7/7 |

**In-graph, under the profiler** (`raw/65_profile_{A,B}.log`, steady-state steps
4–8, 2 dispatches/step):

| arm | query ms/step | step GPU total ms |
| --- | ---: | ---: |
| shipped v0.2.7 artifact | 6.638 – 6.710 (median 6.705) | 27.366 – 27.992 |
| candidate | 4.922 – 5.327 (median 5.084) | 26.007 – 26.900 |

Steady-state ratio 5.084 / 6.705 = **0.758**, matching the isolated mode-9 result
(0.7547) and confirming the in-graph effect is the same mechanism, not a
benchmark artefact. Mean-over-nine-batches attribution gives 8.427 → 6.504
(0.772), which is the number comparable to the historical table.

**Complete optimizer step** (`raw/s7-*.log`, `raw/63_abba.json`; three warmups,
26 measured steps, five rounds of `A B B A`, 20 separate release-mode processes,
unprofiled, digest off):

| arm | A (shipped) | B (candidate) | median ratio | 95% CI of round ratio | speedup | rounds improved |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `--wmma-mode off` | **28.477 ms** | **26.831 ms** | 0.942697 | 0.941696 – 0.943871 | **1.0608×** | 5/5 |
| `--wmma-mode mixed` | **23.035 ms** | **21.415 ms** | 0.929234 | 0.925698 – 0.933281 | **1.0762×** | 5/5 |

Per-round ratios, FP32: 0.943627, 0.943694, 0.942204, 0.942697, 0.941694.
Per-round ratios, Mixed: 0.924883, 0.928896, 0.932845, 0.929234, 0.931590.
Neither interval includes 1.0, and both arms improve in every round.

The absolute saving is ~1.65 ms/step in both arms, which is what the
kernel-level delta predicts (6.705 → 5.084 under the profiler; 1.62 ms), so the
gain is correctly attributed rather than hidden inside a step-level artefact. The
Mixed arm gains a larger *ratio* only because its step is smaller; the kernel
saving is the same, as it must be — the attention shaders are mode-independent and
the A-arm change is confined to pass 1.

Pre-measurement Amdahl projections from the profiled shares and the isolated
mode-9 ratio were 1.0616× (FP32) and 1.0822× (Mixed). The measured 1.0608× and
1.0762× land on those, which is the strongest available evidence that the step
delta is the kernel delta and not a redistribution.

---

## 6. Decision

**GO.**

The candidate is worth developing into a production-promotion phase: it is
bit-exact on 60/60 tensor comparisons across ten geometries and on 9/9 training
steps, it improves the complete optimizer step by 6.1% on the default FP32 route
and 7.6% on the opt-in Mixed route with every round improving and a CI that
excludes 1.0, and it does so without touching production dispatch, bindings, push
constants, scratch semantics, precision policy or the second-pass key shader.

### 6.1 Explicit conditions on the GO

* **Geometry scope is head dimension 128 only.** The candidate cannot represent
  other head dimensions; a promotion must keep the shipped shader for every other
  geometry and add the selector, which this phase deliberately did not write.
* **One adapter, one fixture.** All numbers are from `0x1002:0x15bf`, driver
  `26.20.15.14`, on 2 layers / hidden 1024 / 8q-2kv / batch 2 × seq 64. Another
  geometry (4q-2kv, MHA 8q-8kv, longer sequences, full-parameter training) has
  parity coverage but no performance measurement.
* **The profiler shows the candidate as `shader=unknown`** because its module has
  no registered signature. A promotion should register the name so future
  attributions are readable.
* No claim is made about full-parameter training performance or about Mixed
  beyond the LoRA step; the two arms are reported separately and never multiplied.

---

## 7. Next phase recommendation

**Phase XVIII — promote the head-128 register-resident dQ accumulator, and
re-rank Candidate A behind it.**

Scope of that phase, in order:

1. Ship the candidate as a distinct production artifact
   (`..._dqunroll_rdna3.spv`) and add a selector gated on
   `qk_head_dim == 128 && value_head_dim == 128`, with the shipped shader as the
   unconditional fallback for every other geometry. Default = the promoted route,
   since it is bit-exact; keep `HIERARCHOS_ATTENTION_BACKWARD_*`-style opt-outs
   consistent with the existing selectors.
2. Prove the selector never changes the arithmetic for non-admitted geometry by
   keeping the existing two-pass parity coverage and adding the head-dimension
   matrix to it.
3. Measure complete-step ABBA on at least MHA 8q-8kv and 4q-2kv 8q-2kv at
   seq 64 and seq 128, plus full-parameter training, before any release
   decision. Only then does the promotion claim extend beyond this fixture.
4. Directly implement the Candidate A experiment only **after** C′ lands, and
   re-measure: with the accumulator in registers the remaining steady-state cost
   is ~5.1 ms/step (Mixed), and the question becomes whether caching score and
   `value_dot` in `scratch_probability`/`scratch_ds` costs less than recomputing
   them. The two reductions must stay separate; `kv_softmax_dot` is `precise` and
   feeds dK/dV.

**Strongest runner-up if C′ were rejected: Candidate F (Mixed WMMA staging).** It
is the second-largest measured item (7.380 ms/step, 24.8% of the Mixed step) and
the only one with a plausible double-digit isolated ratio left, but it is opt-in
only, needs the full 37-gate requalification plus the three-phase graph test, and
sits on the route the release does not default to — which is exactly why C′ was
taken first.

**Candidate A is not abandoned, it is re-ordered.** Its premise (57% of the query
pass arithmetic is recomputed) is confirmed in the source, but the measurement
says the pass is not arithmetic-bound, so its expected value must be re-measured
against the post-C′ kernel rather than assumed from the old one.

---

## 8. Preservation report

| check | result |
| --- | --- |
| final branch / HEAD | `research/v0.2.8-optimization-discovery` / `9b0091e02e976441e343d6c445101f9c29395bd6` |
| `main` | `9b0091e…` — unchanged |
| `v0.2.7` tag object / dereferenced commit | `85b64a1e…` / `9b0091e…` — unchanged |
| commits created | **none** (`git log -1` is still the v0.2.7 merge) |
| pushes / merges / tags / releases | **none** |
| tracked changes | `hierarchos-vulkan/src/transformer.rs`, +27 lines, all inside `#[cfg(test)]` |
| new untracked paths | `src/phase_xvii_v028_optimization_discovery_tests.rs`, `validation/phase_xvii_v028_optimization_discovery_20261126/` |
| `stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` — unchanged |
| reset / clean / drop / rebase | **none** |

Production shader and source hashes, identical to the Phase XIII published values
(`raw/01_production_hashes.txt`):

| artifact | sha256 | matches Phase XIII |
| --- | --- | --- |
| `transformer_attention_backward_query_rdna3.comp` | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` | yes |
| `transformer_attention_backward_query_rdna3.spv` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` | yes |
| `transformer_attention_backward_key_channel4_rdna3.spv` | `f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8` | yes |
| `transformer_attention_forward_rowpacked64_cached_rdna3.spv` | `c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd` | yes |
| `wmma_gemm_f16_f32acc.comp` | `239b63144bdd7c7aef703d88622201bc9c6c4832000e7dc86cec3317034863fc` | yes |

Prior receipt directory identities (`raw/02_preservation.txt`, recursive sha256
over every file in sorted path order) reproduce their Phase XIII values exactly,
including `phase_xiii_mixed_wmma_requalification_20261125` =
`78a5c8b076955c683d8551f562a5d76d3f9e8230b450730c619fdbc061a8ed44`
(172 files, 4,967,740 bytes) and `phase_xii_channel4_key_production_20261125` =
`92656e87…3143c`. Nothing in any of them was read-modify-written, overwritten or
deleted.

No production selector, default, threshold, precision policy or eligibility rule
was altered; `--wmma-mode` still defaults to `off` and Mixed remains opt-in.

---

## 9. Reproduction

```text
cd hierarchos-vulkan
bash validation/phase_xvii_v028_optimization_discovery_20261126/profile_baseline.sh
cargo test --release --lib --no-run
./target/release/deps/hierarchos_vulkan-*.exe phase_xvii_query_dqunroll_parity \
    --nocapture --test-threads=1
HIERARCHOS_PHASE_XVII_KERNEL_TIMING=1 \
  ./target/release/deps/hierarchos_vulkan-*.exe phase_xvii_query_dqunroll_timing \
    --nocapture --test-threads=1
bash validation/phase_xvii_v028_optimization_discovery_20261126/run_abba_step.sh
python validation/phase_xvii_v028_optimization_discovery_20261126/reduce_step.py \
    validation/phase_xvii_v028_optimization_discovery_20261126/raw 5
```

The candidate SPIR-V is rebuilt with the `glslc` line in §5.5; the control is the
production `.comp` built with the same line.

## 10. Files added

| file | role |
| --- | --- |
| `README.md` | this receipt |
| `profile_baseline.sh` | identity, preservation hashes, profiled attribution for both arms |
| `run_abba_step.sh` | the complete-step ABBA driver (both `--wmma-mode` arms) |
| `reduce_step.py` | per-round paired ratios, CI, verdict |
| `attribute.py` | the Phase XIII profile reducer, copied unchanged |
| `shaders/…_dqunroll.comp`, `q_dqunroll.spv`, `q_rebuild.spv` | candidate source, candidate artifact, build-flag control |
| `raw/` | every log, hash, counter dump and reduction cited above |
