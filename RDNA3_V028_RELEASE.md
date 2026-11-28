# RDNA3 query-pass optimizations — v0.2.8 release note

v0.2.8 promotes two qualified query-pass changes and one FP32 input-gradient
kernel into the default RDNA3 training route: the Phase XVIII literal-index dQ
accumulator, the Phase XX scratch-reuse variant, and the Phase XXII output-tiled
`linear_input_grad` kernel. The original query shader and the original
input-gradient kernel remain the fallbacks, Mixed WMMA remains **off by
default**, and no precision threshold or existing eligibility allowlist moved.

Every performance number below is a paired complete-step result on one adapter.
The ratios are independent measurements and are never multiplied into an
overall training speedup.

## 1. What changed

| # | Change | Production artifacts | Eligibility | Receipt |
| ---: | --- | --- | --- | --- |
| 1 | Phase XVIII literal-index dQ accumulator | `transformer_attention_backward_query_rdna3_dqunroll.{comp,spv}`, `src/transformer.rs` | existing query selector: QK and value head dimension 128 | `validation/phase_xviii_v028_dq_accumulator_promotion_20261126/` |
| 2 | Phase XX scratch-reuse query pass | `transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.{comp,spv}`, `src/transformer.rs` | Phase XVIII selector, unchanged | `validation/phase_xix_query_scratch_reuse_20261127/`, `validation/phase_xx_query_scratchreuse_promotion_20261127/` |
| 3 | Phase XXII output-tiled FP32 input gradient | `linear_input_grad_output_tile32.{comp,spv}`, `src/transformer.rs` | rows, input and output dimension all multiples of 32 | `validation/phase_xxii_linear_input_grad_tile_20261128/` |

Pass 1 now selects, in order:

1. `query_dqunroll_scratchreuse` when the Phase XVIII selector admits the
   geometry, scratch reuse is enabled (the default) and the artifact loads;
2. `query_dqunroll` when scratch reuse is disabled or its pipeline is refused;
3. the original `query_rdna3` shader when the dQ-unroll path is disabled or the
   geometry is not admitted.

Pass 2 is unchanged: the Phase XII four-channel key shader on its qualified
head-dimension-128 path, the original two-pass key shader otherwise, and the
scalar backward when the enclosing two-pass gate refuses.

## 2. Eligibility and fallbacks

The enclosing two-pass gate is unchanged: modes 0/9, `key_blocks == 1`, no
window, softcap, ALiBi, bias, dropout, sink, offset, sparse or compressed keys,
paged KV or value scale; `head_dim <= 128`, `kv_seq <= head_dim`; grouped
geometries remain limited to `8q/2kv`, `4q/2kv`, `8q/4kv` and `16q/4kv` at head
dimension 128 with `q_seq == kv_seq == 64` and causal masking. MHA keeps its own
bypass. The query selector itself still requires QK and value head dimension 128.

Runtime control arms:

| variable | effect |
| --- | --- |
| `HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0` | Phase XVIII dQ-unroll fallback |
| `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0` | original query shader (wins over either scratch-reuse value) |
| `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0` | scalar attention backward |
| `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` | original two-pass key shader |
| `HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0` | original 16 x 16 FP32 input-gradient kernel |

A module-specific load refusal is an ordinary fallback. Device-lost,
out-of-memory and poisoned-layout failures stay fatal.

## 3. Numerical qualification

- **Phase XIX parity:** 28 cases, 140 tensor comparisons of `dQ`,
  `scratch_probability`, `scratch_ds`, `dK` and `dV`; **140/140 bitwise equal**,
  zero non-finite values. Specialized cases 26, generic fallback 2.
- **Phase XX in-graph equivalence:** 29 complete steps, 87 comparisons, zero
  mismatches; 29/29 loss bits, LoRA gradient digests and parameter digests equal;
  `parameter_end_digest = d79a12d8…9c4655` in both arms. Both arms ran the real
  production dispatch, selector and buffers.
- **Phase XX fallback requalification:** the Phase XVIII matrix was re-read
  against its own module with scratch reuse pinned off — 30 cases, 150 tensor
  comparisons, zero failures. No Phase XVIII campaign was re-run and no
  tolerance changed.
- **Phase XXII input-gradient parity:** 6 shapes, 272,038/272,038 output words
  bitwise equal to the original kernel, zero unwritten elements and zero
  non-finite values. The matrix includes 128x1024x1024, 128x1024x256,
  64x128x128, a 33x37x45 boundary, a 5x96x160 narrow case and 1x1x1.

## 4. Performance

Phase XVIII, five-round ABBA, 3 warmup + 26 measured steps per process, eight
independent geometry/mode series:

| mode | geometry | seq | A (ms) | B (ms) | speedup | rounds | 95 % CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 8q/2kv | 64 | 28.494 | 26.821 | 1.063× | 5/5 | 0.9399–0.9427 |
| FP32 | 8q/8kv MHA | 64 | 32.279 | 30.704 | 1.051× | 5/5 | 0.9477–0.9546 |
| FP32 | 8q/8kv MHA | 128 | 110.475 | 105.912 | 1.042× | 5/5 | 0.9549–0.9647 |
| FP32 | 4q/2kv | 64 | 17.196 | 15.348 | 1.121× | 5/5 | 0.8909–0.8930 |
| Mixed | 8q/2kv | 64 | 23.020 | 21.374 | 1.077× | 5/5 | 0.9270–0.9293 |
| Mixed | 8q/8kv MHA | 64 | 24.829 | 23.238 | 1.068× | 5/5 | 0.9352–0.9374 |
| Mixed | 8q/8kv MHA | 128 | 90.889 | 86.246 | 1.054× | 5/5 | 0.9477–0.9498 |
| Mixed | 4q/2kv | 64 | 17.114 | 15.253 | 1.124× | 5/5 | 0.8851–0.8994 |

Phase XIX, the scratch-reuse step measured against the Phase XVIII module with
the same protocol (160 processes across five rounds per series):

| mode | fixture | A (ms) | B (ms) | speedup | rounds | 95 % CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 8q/2kv seq 64 (primary) | 26.799 | 24.643 | **1.088×** | 5/5 | 0.9176–0.9210 |
| Mixed | 8q/2kv seq 64 (primary) | 21.386 | 19.111 | **1.120×** | 5/5 | 0.8874–0.8977 |

The six additional regression series also improved (1.071×–1.148×). The isolated
pass-1 kernel measured 1.53× in mode 0 and 1.76× in mode 9; the complete-step
figures above are the ones that support a training claim.

Phase XXII, five-round ABBA on the same complete-step fixture:

| mode | kernel | A (ms) | B (ms) | speedup | rounds | 95 % CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FP32 | `linear_input_grad` | 24.875 | 21.716 | **1.1462×** | 5/5 | 0.8706–0.8742 |

Arm A is the original 16 x 16 kernel (`HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0`),
arm B the promoted 32 x 32 output tile. The 29-step parameter-end digest is
identical in both arms (`d79a12d8…9c4655`).

**Phase XXI is component attribution, not a performance campaign.** It profiled
the committed Phase XX tree in three independent processes per mode. Its
per-shader numbers (for example 6.453 ms for FP32 `linear_input_grad` over the
29 profiled batches, 5.948 ms over the 26 measured batches) attribute where a
step's GPU time goes; they are not a before/after measurement and are not used
as a speedup claim here.

## 5. What stayed put

- Mixed WMMA remains opt-in and **default-off**; its thresholds, full-parameter
  learning-rate caveat and adapter-drift observations are unchanged.
- `FULL_FP16_MAX_REDUCTION = 64`, `WMMA_MIN_PROFITABLE_MACS = 16_000_000` and
  every numerical threshold are unchanged.
- The Phase XII four-channel key pass, the row-packed forward and the GQA
  grouped key pass keep their existing artifacts and gates.
- The original query shader and the original FP32 input-gradient kernel are
  still present and remain the fallbacks.

## 6. Supported scope

One measured adapter: integrated AMD RDNA3, PCI `1002:15bf`, driver
`32.0.32015.14001`, Windows 11 x64. Query-pass eligibility is head dimension 128
inside the existing two-pass gate. No universal AMD, NVIDIA, Intel or
cross-architecture qualification is implied. Timings are per-step milliseconds
on one machine; no throughput, power or multi-GPU claim is made.
