# Phase XIII — RDNA3 Mixed WMMA requalification on the optimized Transformer production path

**Result: GO for the production GQA LoRA path.** With the Phase III–XII tree
unchanged, `--wmma-mode mixed` now measures **1.2385×** on the exact Phase XII
release step harness (28.481 ms → 23.001 ms, five paired ABBA rounds,
20 separate release-mode processes, 95% CI of the round ratio
0.8059–0.8101) and **1.2288×** on an independent counter-carrying harness.
Full-parameter training measures **1.2792×** (37.564 → 29.360 ms) and its
numerics qualify at a stable learning rate, but its fixture learning rate is
unstable in the FP32 oracle itself, so the full-parameter *numerical* result is
left unqualified. Nothing was promoted, no shader or policy was modified, and
the default remains `off`.

This is a re-measurement, not a kernel change: the Mixed WMMA implementation,
its thresholds, `FULL_FP16_MAX_REDUCTION`, the 37-gate qualification contract
and every attention path are byte-identical to Phase XII. What changed since the
earlier neutral/negative Mixed measurements is the graph around them — the
attention-backward key pass no longer dominates, so the same projection kernels
are now a large enough share of the step to show up end to end.

---

## 1. Preservation evidence (Phase III–XII intact)

Base `560809eead186f832344c15d1a9b70795da3b75d`, branch `main`. Captured before
any measurement (`raw/00_identity.txt`) and again at receipt time
(`raw/09_final_identity.txt`):

| item | before | at receipt time |
| --- | --- | --- |
| HEAD | `560809eead186f832344c15d1a9b70795da3b75d` | identical |
| branch | `main` | identical |
| staged index tree | `4709d4508ca02c854741617b379795929f673760` | identical |
| staged diff sha256 | `e3b0c442…` (empty) | identical |
| `stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` | identical |
| full diff (`--binary`) sha256 | `3a671c752ccf8bb7ad714781fae3f11e6264cd6085001da9caf9896c7244853f` | identical |
| binary-only diff (`*.spv`) sha256 | `cfa0d1c059d1dab6500f23ec981e2e4fdfc6c47bd43c1d86957ee686f8d7319a` | identical |
| tracked text diff sha256 | `1176e36db6d2542deddc85b28d0070f39ab9ce71bea575e15ba84ee6a8e96db4` | identical |
| tracked files changed by this phase | — | **none** (`git diff --stat` unchanged) |

The full-diff and binary-diff digests are the two Phase XII published values
(`3a671c75…`, `cfa0d1c0…`); both reproduce exactly, so the working tree is
byte-identical to the state Phase XII measured. The only addition is this
`validation/phase_xiii_mixed_wmma_requalification_20261125/` directory
(`raw/09_final_identity.txt` lists it as the sole new untracked path).

Prior receipts, recursive sha256 over every file in sorted path order
(`raw/02_preservation.txt`):

| directory | files | bytes | identity |
| --- | ---: | ---: | --- |
| `mha_attention_backward_closeout_20261124` | 36 | 787,175 | `bf2bde0f…a15e2` |
| `gqa_attention_backward_experiment_20261124` | 13 | 249,967 | `ebde677a…8852d` |
| `gqa_production_promotion_20261124` | 14 | 553,326 | `25e2a128…09107` |
| `gqa_end_to_end_training_20261124` | 178 | 4,559,007 | `4c1b4d96…df21a` |
| `phase_viii_key_blocks` | 26 | 2,242,018 | `b15767c3…fcd5f` |
| `phase_ix_row_packed_forward_20261124` | 61 | 1,771,186 | `5d24674f…594f8b` |
| `phase_x_row_packed_forward_promotion_20261125` | 66 | 2,538,532 | `b6efb85e…9ef49f` |
| `phase_xi_channel_tiling_20261125` | 69 | 2,068,107 | `048ee479…3b6948` |
| `phase_xii_channel4_key_production_20261125` | 59 | 2,058,249 | `92656e87…143c` |

Every hash equals the value Phase XII recorded for that directory; the Phase XII
directory itself is hashed here for the first time because a receipt cannot
contain its own digest.

Production attention artifacts (`raw/02_preservation.txt`), all unchanged:

| artifact | sha256 |
| --- | --- |
| `transformer_attention_backward_key_rdna3.comp` | `e4179354…92ff78` |
| `transformer_attention_backward_key_rdna3.spv` | `ccf60984…1cd925` |
| `transformer_attention_backward_query_rdna3.comp` | `2e3f5720…d7eb56` |
| `transformer_attention_backward_query_rdna3.spv` | `f9652f52…104256` |
| `transformer_attention_forward_rowpacked64_cached_rdna3.comp` | `8f7bf06f…3439e6ed` |
| `transformer_attention_forward_rowpacked64_cached_rdna3.spv` | `c053503e…43764fd` |
| `transformer_attention_backward_key_channel4_rdna3.comp` | `e1373db7…6e78f9` |
| `transformer_attention_backward_key_channel4_rdna3.spv` | `f5480cce…7f7b2c8` |

No reset, clean, rebase, discard, overwrite, commit, push or release was
performed. No production source, shader, dispatch threshold, default or
precision-policy file was edited; the WMMA implementation is exactly the one
inherited from Phase III–XII.

---

## 2. The Mixed WMMA implementation and its selector

`src/wmma.rs` owns the primitive, `src/wmma_linear.rs` the graph-facing executor,
and `src/transformer.rs` the admission points. The chain is short and every link
is countable.

**Policy.** `WmmaMode::{Off, Mixed, FullFp16}` from `HIERARCHOS_WMMA_MODE`
(`WmmaMode::from_env`); `Off` is the default and requests no cooperative feature,
creates no pipeline and allocates no FP16 mirror. `Mixed` is
`FP16 × FP16 → FP32 accumulation` with FP32 masters, FP32 AdamW state and FP32
norm/softmax/loss. It is **not** Full FP16: `FULL_FP16_MAX_REDUCTION = 64`
(`src/precision_policy.rs`) is untouched and never consulted on this route,
because `LinearWmma::admits` short-circuits only for `FullFp16`.

**Device admission.** `VulkanWmmaCapabilities` requires an AMD RDNA3-or-newer PCI
ID plus `VK_KHR_cooperative_matrix`, the `cooperativeMatrix` feature, the Vulkan
memory model, 16-bit storage/arithmetic and a `SUBGROUP`-scope 16×16×16
configuration. `raw/03_wmma_probe_off.json` / `raw/04_wmma_probe_mixed.json`:

| field | `off` | `mixed` |
| --- | --- | --- |
| `probed` | false | true |
| `configuration_count` | 0 | 11 |
| `mixed_tile` | null | `16×16×16 f16×f16→f32`, `subgroup_scope`, non-saturating |
| `full_fp16_tile` | null | `16×16×16 f16×f16→f16` |
| `enabled` | false | true |

**Shape policy (unchanged).**

```text
wmma_shape_eligible(rows, cols, red) = rows>=16 && cols>=16 && red>=16
                                     && rows*cols>=1024
                                     && ceil(rows/32)*ceil(cols/32)>=4
wmma_shape_profitable(...)           = eligible(...) && rows*cols*red >= 16_000_000
```

`WMMA_MIN_PROFITABLE_MACS = 16_000_000`, `FULL_FP16_MAX_REDUCTION = 64`, the
Mixed arm and every FP32 path are exactly as Phase XII left them
(`src/wmma.rs` sha256 `77f23697…ba9a6`, `src/wmma_linear.rs` `03aac057…83c5d`).

**Graph admission.** `TransformerKernels::wmma_linear(rows, input_dim,
output_dim)` is the single gate. It returns `None` when the mode is `Off` or when
`wmma_shape_profitable(rows, output_dim, input_dim)` declines, counting the
latter as a policy fallback. Executors are cached per `(rows, in, out)` geometry
and share FP32→FP16 weight mirrors keyed by `(allocation, length, epoch)`; the
epoch advances only after a submitted optimizer step, so a mirror can never be
stale and a sibling parameter of the same shape can never read it.

**Three routes, three counters.** `LinearWmma::record_forward`,
`record_weight_grad`, `record_input_grad` each call `admits(rows, reduction)`,
which requires the projection geometry to be eligible *and*
`select_arm(...).uses_cooperative_matrix()`. Ineligible shapes keep the caller's
own kernel and increment `shape_fallbacks`; `Mixed` never routes a shape to the
plain-FP16 arm because `admits` rejects `PlainF16Storage`.

| route | GEMM | layout |
| --- | --- | --- |
| forward | `out = in · Wᵀ` — M=`rows`, N=`out_dim`, K=`in_dim` | A row-major, B transposed |
| weight gradient | `dW = dOutᵀ · in` — M=`out_dim`, N=`in_dim`, K=`rows` | A transposed, B row-major |
| input gradient | `dX = dOut · W` — M=`rows`, N=`in_dim`, K=`out_dim` | A row-major, B row-major |

`record_weight_grad` is only reached when `VulkanLinear::train_base` is true, so
a LoRA run never records a dense base weight gradient. Adapter gradients take
`linear_weight_grad` (FP32) regardless of mode.

**Kernel identity.** `shaders/wmma_gemm_f16_f32acc.comp`
sha256 `239b63144bdd7c7aef703d88622201bc9c6c4832000e7dc86cec3317034863fc`,
SPIR-V `e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65`
(both in `raw/01_wmma_source_and_shader_hashes.txt`). The FP32 kernels that Mixed
replaces are `linear_forward.spv` `7f9a3868…a649da49` and
`linear_input_grad`/`linear_weight_grad`; the operand conversion is
`fp32_to_fp16_native.spv` `51d229f1…84139f`.

---

## 3. Per-projection route and eligibility table

Fixture geometry: 2 layers, hidden 1024, intermediate 1024, vocab 1024, 8 query
heads × head dim 128 = 1024, 2 KV heads × 128 = 256, batch 2 × sequence 64 =
**128 GEMM rows**, LoRA rank 8 / alpha 16. Every prediction below is produced by
`projection_table.py` from the verbs of the Rust predicates, and the script
**fails** if the predicted cooperative-dispatch totals differ from the counters
the graph recorded (`raw/07_projection_table.json`).

| projection | in | out | forward M/N/K | MMAC | forward | dW | dX |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `layers.{0,1}.self_attn.q_proj` | 1024 | 1024 | 128/1024/1024 | 134.22 | **WMMA** | not recorded (base frozen) | **WMMA** |
| `layers.{0,1}.self_attn.k_proj` | 1024 | 256 | 128/256/1024 | 33.55 | **WMMA** | not recorded | **WMMA** |
| `layers.{0,1}.self_attn.v_proj` | 1024 | 256 | 128/256/1024 | 33.55 | **WMMA** | not recorded | **WMMA** |
| `layers.{0,1}.self_attn.o_proj` | 1024 | 1024 | 128/1024/1024 | 134.22 | **WMMA** | not recorded | **WMMA** |
| `layers.{0,1}.mlp.gate_proj` | 1024 | 1024 | 128/1024/1024 | 134.22 | **WMMA** | not recorded | **WMMA** |
| `layers.{0,1}.mlp.up_proj` | 1024 | 1024 | 128/1024/1024 | 134.22 | **WMMA** | not recorded | **WMMA** |
| `layers.{0,1}.mlp.down_proj` | 1024 | 1024 | 128/1024/1024 | 134.22 | **WMMA** | not recorded | **WMMA** |
| `lm_head` | 1024 | 1024 | 128/1024/1024 | 134.22 | **WMMA** | not recorded | **WMMA** |
| `layers.{0,1}.q_proj.lora_A` | 1024 | 8 | 128/8/1024 | 1.05 | FP32 narrow-output lane4 | FP32 delta | FP32 group4-muladd |
| `layers.{0,1}.q_proj.lora_B` | 8 | 1024 | 128/1024/8 | 1.05 | FP32 `linear_forward` | FP32 delta | FP32 `linear_input_grad` |

Totals per step: **15 forward + 15 input-gradient cooperative GEMMs** in LoRA,
**15 + 15 + 15** in full-parameter training; **0 shape fallbacks** in both. The
dense base weight gradient is the only route LoRA does not exercise, because the
base weights are frozen — not because WMMA declines it.

Every otherwise-eligible projection does dispatch. The only FP32 remainders are
the four adapter GEMMs per step, which fail `wmma_shape_eligible` on their own
geometry:

| adapter route | failing condition |
| --- | --- |
| `lora_A` forward/dX/dW | `out_dim = 8 < 16`; 1.05 MMAC also below the 16 MMAC floor |
| `lora_B` forward/dX/dW | `reduction = 8 < 16` (LoRA rank); same 1.05 MMAC |

This is the per-shape trap in the brief: an operator could naively read "LoRA
rank 8" as "adapter GEMMs are small" and still expect ten of them to be
accelerated; they are not, by construction.

Full-parameter training adds the dense weight gradients at the same geometry
(M=`out_dim`, N=`in_dim`, K=128 → 134.22 MMAC for the wide projections, 33.55
MMAC for K/V), all of which clear both the tile contract and the 16 MMAC floor,
which is why `wmma_weight_grad` goes from 0 to 15 per step.

---

## 4. Actual dispatch counters

**Qualification pass** (`raw/11_stability_lora_mixed.log`, 3 steps, same fixture,
seed 7, batch 2, seq 64, LoRA rank 8 / alpha 16, lr 1e-2):

| counter | `--wmma-mode off` | `--wmma-mode mixed` |
| --- | ---: | ---: |
| forward cooperative | 0 | **15 / step** |
| weight-gradient cooperative | 0 | **0 / step** (LoRA) |
| input-gradient cooperative | 0 | **15 / step** |
| total cooperative | 0 | **30 / step** |
| shape fallbacks | 0 | **0** |
| weight-mirror conversions | 0 | 15 / step |
| weight-mirror reuses | 0 | 15 / step |
| mixed accumulation vs full-FP16 | 0 / 0 | only the Mixed arm exists here |

The off arm proves zero cooperative dispatches, so it is genuinely the FP32
graph. The mixed arm's 15 conversions + 15 reuses per step are the expected
mirror contract: one conversion per projection master per optimizer epoch,
then reuse by that projection's input-gradient GEMM.

**Every timed process carries its own proof.** `training-stability` prints each
step's dispatch deltas, and `raw/stab-r*-p*-{A,B}.log` shows the same signature in
**20/20** ABBA processes: arm A `(forward=0, dW=0, dX=0, cooperative=0,
fallbacks=0)` for all 29 steps, arm B `(15, 0, 15, 30, 0)` for all 29 steps.

**The attention route is identical in both arms** (Phase XII harness
`[phase-xii-step-run-end]`, every round): `forward_row_packed=58`,
`backward_query=58`, `backward_key_channel4=58`, `backward_key_original=0`,
`backward_scalar=0` for 29 steps. The only variable in the experiment is WMMA
mode.

**Graph-level FP32 kernels that survive Mixed** (per step, profiled): two
`linear_forward` and two `linear_input_grad` dispatches — the LoRA B adapter —
plus two `linear_forward_lane4` (LoRA A narrow output) and four
`linear_weight_grad` (adapter deltas). These are the same in both arms and are
never labelled WMMA.

Full-parameter counters (`raw/41_full_mixed_9.json`, 9 steps): forward 135,
weight-gradient 135, input-gradient 135 = **405 cooperative**, 0 fallbacks.

---

## 5. Numerical qualification

**Established 37/37 Mixed gates** — `wmma_qualify --wmma-mode mixed --json`
(`raw/05_wmma_qualify_mixed.json`), unmodified contract, exit 0:

```text
device = AMD Radeon Graphics        mode = mixed
gates  = 37                         failed = 0        status: PASS
```

Per-gate errors are in the same file (`results[]`), including the worst
`vs FP32` row: `128x128x96 trans_a=true trans_b=true`, `max_abs 8.63e-3` against
the `2.0e-2` limit and `rms 2.08e-3` against `3.0e-3`. No tolerance was changed.

**Graph-path three-phase gate** — `cargo test --release --lib wmma_linear`
(`raw/06_wmma_linear_graph_tests.log`), the test that dispatches the exact
`record_forward` / `record_weight_grad` / `record_input_grad` entry points
against host FP32 references:

```text
test wmma_linear::tests::mixed_linear_graph_path_matches_fp32_in_three_phases ... ok
test wmma_linear::tests::weight_mirror_is_reused_within_an_epoch_and_refreshed_after_a_step ... ok
test wmma_linear::tests::ineligible_shapes_keep_their_own_kernel_and_are_counted ... ok
test wmma_linear::tests::full_fp16_graph_widens_accumulators_in_all_three_phases ... ok
test result: ok. 4 passed; 0 failed
```

**Training-level comparison** against the FP32 oracle, using the existing Mixed
contract (`max_abs < 0.02`, `rms < 0.003`) with nothing relaxed
(`raw/60_numerics.json`):

| quantity (29 steps, LoRA) | max_abs | rms | within contract |
| --- | ---: | ---: | --- |
| loss trajectory (29 values) | **4.02e-3** | **1.37e-3** | yes |
| native LoRA gradient norm (29 values) | 7.00e-3 | 1.90e-3 | yes |
| adapter weight, worst of 4 tensors | 2.60e-2 | 2.01e-3 | **no** |
| adapter AdamW moment, worst of 8 tensors | 4.39e-4 | 9.85e-5 | yes |
| base parameters (frozen) | identical | — | yes |
| non-finite loss / gradient / parameter | 0 | — | yes |

The one number outside the envelope is the elementwise adapter weight of the
second layer's `q_proj.lora_B` after 29 steps at lr 1e-2 (max 0.0260). Its RMS is
inside the contract and the trajectory that produced it is inside the contract.
The established Mixed producer (`validation/qualify_precision_training.py`) does
not assert FP32↔Mixed adapter *elementwise* equality — it asserts the loss
trajectory against FP32 and exact within-mode resume — so this is recorded as an
observation, not hidden and not reclassified as a pass. It is the single
caveat attached to the GO recommendation.

**Determinism and resume** (`raw/60_numerics.json`):

| check | result |
| --- | --- |
| Mixed rerun, 29 steps | `loss_bits` identical (29/29), `grad_digest` identical, `parameter_end_digest` `d79a12d8…` both runs |
| FP32 rerun, 29 steps | `loss_bits` identical, `grad_digest` identical, digest `d79a12d8…` |
| checkpoint save (6 steps) + exact resume (6 steps) vs one 12-step run | `loss_bits` identical 12/12, `grad_digest` identical 12/12, `parameter_end_digest` `d79a12d8…` in both |

Checkpoint identities are in `raw/10_fixture_and_checkpoint_hashes.txt`
(`adapter_config.json` `1fa65722…`, `adapter_model.safetensors` `2800bc5c…`,
`hierarchos_adapter_optimizer.safetensors` `f4ca6167…`). The 29-step adapters
differ between arms by construction (off `8a64128b…`, mixed `252c9fc5…`); that
is the measured drift above, not a schema difference.

No unsupported operation enters the WMMA path: `shape_fallbacks = 0`, every
cooperative dispatch is `MixedFp32Accumulate`, and the adapter GEMMs never
dispatch WMMA.

---

## 6. Kernel-level profile (GPU timestamps, ms per training step)

Two profiled release runs, 9 steps each, `HIERARCHOS_VULKAN_PROFILE_KERNELS=1`
`HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`, reduced by `attribute.py` over the
nine batches that contain the two-layer backward query pass
(`raw/50_profile_off.log`, `raw/51_profile_mixed.log`,
`raw/52_attribution.json`). A profiled run perturbs the step it measures; these
numbers attribute work between components, the headline timings are the
unprofiled ABBA in §7.

| shader | off ms/step | mixed ms/step | Δ | dispatches/step (off → mixed) |
| --- | ---: | ---: | ---: | --- |
| `wmma_gemm_f16_f32acc` | — | **7.411** | **+7.411** | 0 → 30 |
| `fp32_to_fp16_native` | — | **1.169** | **+1.169** | 0 → 45 |
| `linear_forward` (FP32 dense fwd) | **8.203** | 0.024 | **−8.179** | 17 → 2 |
| `linear_input_grad` (FP32 dX) | **7.324** | 0.009 | **−7.316** | 17 → 2 |
| `linear_forward_lane4` (LoRA A) | 0.214 | 0.062 | −0.152 | 2 → 1.6 |
| `linear_input_grad_group4_muladd` (LoRA B) | 0.133 | 0.137 | +0.004 | 2 → 2 |
| `linear_weight_grad` (adapter deltas) | 0.059 | 0.062 | +0.003 | 4 → 4 |
| attention backward query | 8.411 | 9.173 | +0.762 | 2 → 2 |
| attention forward (row-packed) | 4.725 | 5.023 | +0.298 | 2 → 2 |
| attention backward channel4 key | 0.538 | 0.583 | +0.045 | 2 → 2 |
| RMS norm forward + input grad | 3.824 | 3.897 | +0.073 | 10 → 10 |
| cross-entropy (lanes16) | 1.524 | 1.614 | +0.090 | 1 → 1 |
| everything else | 0.535 | 0.459 | −0.076 | — |
| **attributed total** | **35.490** | **29.623** | **−5.867** | |

Read plainly: the projection stack costs **15.933 ms/step** in FP32 and
**8.873 ms/step** in Mixed (`wmma_gemm` 7.411 + conversion 1.169 + the 0.293 of
remaining FP32 adapter work) — **−7.06 ms** of projection work for **+7.41 ms**
of cooperative GEMM and **+1.17 ms** of operand conversion, i.e. the Mixed
kernel is roughly 1.8× the FP32 GEMM it replaces *including* the mirror
conversion.

The attention rows move slightly between the two profiled processes
(query +0.76, forward +0.30) although both use the identical SPIR-V with
identical dispatch counts. That is profiler perturbation, not a WMMA effect:
the attention shaders are byte-identical (§1) and cannot observe the WMMA mode.

**Isolated GEMM ratio at the production shapes** (`wmma_bench --shapes`,
`raw/09_gemm_bench_{off,mixed}.json`, 3 rounds × 10 iterations):

| shape (M×N×K) | role | FP32 ms | Mixed ms | speedup |
| --- | --- | ---: | ---: | ---: |
| 128×1024×1024 | q/o/gate/up/down forward, lm_head | 0.887 | 0.568 | **1.89×** |
| 128×256×1024 | k/v forward | 0.346 | 0.415 | **0.93×** |
| 128×1024×256 | k/v input gradient | 0.365 | 0.230 | **1.72×** |
| 1024×1024×128 | dense weight gradient | 0.829 | 0.345 | **2.47×** |

The narrow K/V forward is the one shape where the cooperative kernel loses, and
the graph pays it: four such GEMMs per step at −0.07 ms each. The wide
projections win 1.9–2.5×, which is where the end-to-end gain comes from. These
isolated ratios are **not** extrapolated to throughput (§7 is the measurement
that counts).

---

## 7. Release-mode LoRA ABBA (the Phase XII protocol, fresh baseline)

Protocol: three warmup steps, 26 measured steps per process, five rounds of
A B B A, 20 separate release-mode processes, unprofiled, same device and driver,
same fixture/model/seed/optimizer/LoRA configuration, production row-packed
attention forward, production GQA query pass and production channel4 key pass in
both arms. Arm A `--wmma-mode off`, arm B `--wmma-mode mixed`.

| quantity | arm A (off) | arm B (mixed) |
| --- | ---: | ---: |
| median complete-step | **28.481 ms** | **23.001 ms** |
| mean complete-step | 28.474 ms | 23.007 ms |
| 95% CI of the mean | 28.458–28.490 ms | 22.971–23.043 ms |
| P95 of the 26-step samples | 29.406 ms | 24.362 ms |
| standard deviation | 0.410 ms | 0.735 ms |
| parameter-end digest | `d79a12d8…` | `d79a12d8…` |
| attention dispatch counters (29 steps) | 58/58/58 | 58/58/58 |
| cooperative projection dispatches | 0 | 15 fwd + 15 dX per step |

Per-round B/A ratios (mean of the two pairings in each round):

| round | A (ms) | B (ms) | paired ratios | round ratio |
| --- | --- | --- | --- | ---: |
| 0 | 28.499 / 28.445 | 23.115 / 23.016 | 0.811078 / 0.809155 | 0.810117 |
| 1 | 28.437 / 28.497 | 22.983 / 22.986 | 0.808206 / 0.806599 | 0.807402 |
| 2 | 28.471 / 28.501 | 22.976 / 22.968 | 0.807004 / 0.805865 | 0.806434 |
| 3 | 28.441 / 28.496 | 23.065 / 23.023 | 0.810969 / 0.807934 | 0.809452 |
| 4 | 28.463 / 28.490 | 23.029 / 22.910 | 0.809086 / 0.804144 | 0.806615 |

Median round ratio **0.807402**, mean 0.808004, **95% CI 0.805916–0.810092**,
min 0.806434, max 0.810117, **5/5 rounds improved**, speedup **1.2385×**. The
interval excludes 1.0, so the result is a measurable benefit rather than an
inconclusive one.

The fresh baseline matches the Phase XII published 28.460 ms (28.481 here), so
this is a matched comparison and no historical number is substituted.

**Independent cross-check with the counter-carrying harness**
(`raw/stab-r*-p*-{A,B}.log`, same 3+26 protocol, `--digest-every 0`, every step
printing its own dispatch deltas): A 28.313 ms median / 28.413 mean
(CI 28.182–28.644), B 23.097 / 23.045 (CI 22.958–23.131), median ratio
**0.813787**, CI 0.800196–0.822215, **5/5 rounds**, speedup **1.2288×**. Two
harnesses, two processes per pairing, same sign and magnitude.

---

## 8. Full-parameter training (secondary workload)

Same model geometry, same fixture, seed 7, batch 2, seq 64, lr 1e-2, 29 steps per
process, 5 rounds × ABBA, separate processes, `--digest-every 0`
(`raw/full-r*-p*-{A,B}.log`, `raw/20_abba.json` → `full_parameter_harness`).
This arm exercises the dense weight-gradient route that LoRA never records:
**15 forward + 15 dW + 15 dX cooperative GEMMs per step, 0 fallbacks**.

| quantity | arm A (off) | arm B (mixed) |
| --- | ---: | ---: |
| median complete-step | **37.564 ms** | **29.360 ms** |
| mean complete-step | 37.560 ms | 29.366 ms |
| 95% CI of the mean | 37.524–37.597 ms | 29.340–29.392 ms |
| P95 | 38.640 ms | 30.432 ms |
| standard deviation | 0.534 ms | 0.385 ms |

Per-round ratios: 0.7844, 0.7819, 0.7817, 0.7806, 0.7806 — median **0.781712**,
mean 0.781838, **95% CI 0.779900–0.783777**, **5/5 rounds**, speedup **1.2792×**.

The timing result is qualified. The **numerical** result is not, and the reason
is in the oracle rather than in WMMA: at the fixture's lr 1e-2 the FP32 arm
itself does not descend (loss 7.13 → 6.71 → 7.98 → 7.60 → 16.80 → 12.64 → 19.56
→ 14.09 → 14.38), so a trajectory difference of 0.81 and a master difference of
0.103 measure divergence amplification, not precision. Following the brief, the
full-parameter numerical comparison is therefore **left unqualified at lr 1e-2**.

A stable-learning-rate control makes the precision question answerable
(`raw/44_full_off_lr1e4_export.json`, `raw/45_full_mixed_lr1e4_export.json`,
9 steps, lr 1e-4, both arms bounded 6.46–7.17):

| quantity | max_abs | rms | within contract |
| --- | ---: | ---: | --- |
| loss trajectory (9 values) | **6.80e-5** | **4.11e-5** | yes |
| master weights, worst of 21 tensors (`embed_tokens`) | **7.99e-4** | 6.23e-6 | yes |
| all 21 master tensors | — | — | **yes** |

So the dense weight-gradient route is numerically sound; what is unqualified is
full-parameter *training* at this fixture's aggressive learning rate, which is a
property of the fixture. Do not average this with the LoRA result (§9 of the
brief): they are reported separately and neither is combined.

---

## 9. Loss and training-state comparison (summary)

| comparison | off | mixed | delta |
| --- | --- | --- | --- |
| LoRA loss, step 0 → 28 | 7.133183 → 6.552530 | 7.133153 → 6.551390 | max 4.02e-3 |
| LoRA gradient norm range | 0.10929 – 1.20100 | 0.10929 – 1.20271 | max 7.00e-3 |
| LoRA base parameters | `d79a12d8…` (frozen) | `d79a12d8…` (frozen) | identical |
| LoRA adapters (29 steps) | `8a64128b…` | `252c9fc5…` | 0.0260 max element (see §5) |
| full-param loss (lr 1e-2, unstable) | 7.133 → 14.380 | 7.133 → 13.570 | 0.810 (unqualified) |
| full-param masters (lr 1e-4 control) | reference | — | 7.99e-4 max |

---

## 10. Confidence intervals and raw samples

Every raw process log is in `raw/`: `abba-r{0..4}-p{0..3}-{A,B}.log` (Phase XII
harness), `stab-r{0..4}-p{0..3}-{A,B}.log` (counter-carrying harness),
`full-r{0..4}-p{0..3}-{A,B}.log` (full-parameter). Each contains one
`[phase-xii-step]` / `qual_step` line per step and 26 measured samples per
process. The reductions are `raw/20_abba.json` (per-round pairs, CIs, P95,
standard deviations, dispatch evidence) and `raw/52_attribution.json`.

| series | ratio median | 95% CI | rounds improved | verdict |
| --- | ---: | --- | ---: | --- |
| LoRA, Phase XII harness | 0.807402 | 0.805916–0.810092 | 5/5 | PASS |
| LoRA, counter harness | 0.813787 | 0.800196–0.822215 | 5/5 | PASS |
| full-parameter | 0.781712 | 0.779900–0.783777 | 5/5 | PASS (timing) |

No interval includes 1.0.

---

## 11. GO / NO-GO

**GO — Mixed WMMA is measurably beneficial on the current production training
path.** 1.2385× on the Phase XII harness with a matched fresh baseline, 1.2288×
on an independent harness, 5/5 rounds in both, CIs entirely below 1.0, and the
kernel-level cause is explicit: two FP32 GEMM kernels totalling 15.5 ms/step are
replaced by one 7.4 ms cooperative kernel plus 1.2 ms of conversion, with the
narrow K/V forward as the only losing shape.

Conditions attached to that verdict:

* **LoRA path:** GO. Numerical trajectory, gradient norms, determinism and exact
  resume are inside the established Mixed contract. The single elementwise
  adapter deviation (0.0260 on one tensor after 29 steps at lr 1e-2) is reported
  above and should be considered by whoever decides the promotion.
* **Full-parameter path:** timing GO (1.2792×), numerics **unqualified at
  lr 1e-2** because the FP32 oracle diverges there; qualified at lr 1e-4
  (loss 6.8e-5, masters 8.0e-4). Do not promote full-parameter Mixed on this
  evidence alone.
* **Nothing was promoted.** `--wmma-mode` still defaults to `off`,
  `FULL_FP16_MAX_REDUCTION`, `WMMA_MIN_PROFITABLE_MACS`, the Mixed tile, the
  staging schedule and every attention path are unchanged, and the next phase
  owns the decision.

This overturns the *earlier* neutral/negative end-to-end Mixed measurements on
their own terms: those were taken when the attention-backward key pass consumed
~34% of the step and buried a ~5.5 ms projection win. With the key pass at 1.5%,
the same win is 19% of the step.

---

## 12. Refreshed attribution and the next bottleneck

Mixed-arm profile shares of the 29.62 ms/step attributed total (§6):

| component | ms/step | share |
| --- | ---: | ---: |
| attention backward query pass | **9.173** | **31.0%** |
| dense projections (WMMA + conversion + FP32 adapters) | **8.874** | **30.0%** |
| attention forward (row-packed) | 5.023 | 17.0% |
| RMS norm forward + input grad | 3.897 | 13.2% |
| cross-entropy | 1.614 | 5.4% |
| channel4 key pass | 0.583 | 2.0% |
| everything else | 0.459 | 1.5% |

**The next dominant single kernel is the GQA backward query pass** (9.17 ms/step,
31.0%), followed by the dense projection stack (8.87 ms, 30.0%), then attention
forward (5.02 ms, 17.0%) and RMSNorm (3.90 ms, 13.2%). The query pass is the
largest remaining item and the only one whose share *grew* because the key pass
left the distribution in Phase XII. Attention forward is the next-largest single
kernel after that.

If Mixed were promoted, its own headroom is now the staging schedule of the
cooperative kernel (7.41 ms of GEMM plus 1.17 ms of conversion for 3.22 GMAC of
projection work per step) — the C1-style load batching the brief defers to a
later phase.

---

## 13. Reproduction

```text
cd hierarchos-vulkan
bash validation/phase_xiii_mixed_wmma_requalification_20261125/reproduce.sh
```

The four 54 MB exported master-weight packages compared in §8 are **not** kept in
`raw/`; their identities are in `raw/11_exported_master_hashes.txt` and the
per-tensor comparison they produced is in `raw/60_numerics.json`. `numerics.py`
regenerates nothing, so run it through `reproduce.sh` (steps 9–10 produce the
exports first). The LoRA adapter, optimizer and checkpoint files are small and
are kept in full.

Sections run in order: identity and preservation, capability probes, the 37
Mixed gates, the graph-path tests, the dispatch proof, the LoRA numerics, the
profiled attribution, the three ABBA series, and the reductions. Every artifact
lands in `raw/`.

Fast targeted runs:

```text
HIERARCHOS_WMMA_MODE=mixed ./target/release/wmma_probe --wmma-mode mixed --json
./target/release/wmma_qualify --wmma-mode mixed --json
cargo test --release --lib wmma_linear -- --nocapture --test-threads=1
cargo test --release --lib phase_xii_channel4_key_full_step -- --nocapture \
  HIERARCHOS_PHASE_XII_STEP_ARM=channel4 HIERARCHOS_WMMA_MODE=mixed \
  HIERARCHOS_PHASE_XII_STEP_STEPS=29
```

Device: `AMD Radeon Graphics`, vendor `0x1002`, device `0x15bf` (integrated
RDNA3), single driver, quiet measurement environment. Executable identities
(`raw/08_executable_hashes.txt`): lib test binary `125a6956…a834b`,
`training-stability` `374e6235…255b1`, `wmma_probe` `6f01d9cc…878e6`,
`wmma_qualify` `fe0b6496…2a0031`, `wmma_bench` `38075f3f…6d7574`. Fixture
(`raw/10_fixture_and_checkpoint_hashes.txt`): `config.json` `f5f1703b…8f68e`,
`model.safetensors` `a77d99d8…2754b`, `dataset.jsonl` `0c5d3e90…6af9c`.

---

## 14. Files added

| file | role |
| --- | --- |
| `README.md` | this receipt |
| `reproduce.sh` | end-to-end reproduction of every section |
| `run_abba_phase_xii_harness.sh` | the three ABBA series driver |
| `reduce.py` | per-round ratios, CIs, P95, dispatch extraction |
| `attribute.py` | per-step kernel attribution from profiled batches |
| `numerics.py` | loss/gradient/adapter/master comparison against the Mixed contract |
| `projection_table.py` | per-projection geometry, eligibility, route and counter check |
| `raw/` | every log, hash, counter dump and reduction; `raw/11_exported_master_hashes.txt` names the master exports that were compared and then removed for size |

No production file was created or modified.
