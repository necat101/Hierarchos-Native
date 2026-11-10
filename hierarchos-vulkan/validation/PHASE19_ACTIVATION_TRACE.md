# Phase XIX — activation-level mixed-WMMA tracing

Observation-only instrumentation of the production Vulkan graph. No shader,
kernel, optimizer rule or tolerance changed; tracing is off unless
`HIERARCHOS_ACTIVATION_TRACE=1`.

- Source revision: see `current-qualification.json` (`source_clean: true`).
- Receipt: `validation/release_parity/multimodal_training/20261110-phase19-activation-trace/`.
- Device: AMD Radeon Graphics (`1002:15bf`).
- Geometry: batch 8, sequence 64, one validation batch (512 rows).
- Checkpoints: initial, step-1 FP32, step-1 mixed, step-10 FP32, step-100 FP32
  (both step-10/100 are weights-only Phase XVIII exports).

## What is captured

Twelve buffers per decoder layer (24 total), plus three graph-level buffers:

`token_embedding`, `pre_attention_norm`, `q_proj`, `k_proj`, `v_proj`,
`attention_context`, `attention_output_projection`, `attention_residual`,
`pre_mlp_norm`, `mlp_gate_up_projection`, `mlp_activation`, `mlp_output`,
`layer_output`, `final_norm`, `logits`.

Shapes are `[8, 64, width]`: q/k/v width 64, MLP hidden 256, vocabulary 1024.
Each tap is a `copy_f32` of the already-materialized production buffer recorded
in the same batch, so it observes the real dispatch output. Softmax scores and
probabilities are fused inside the attention shader and are **not** separately
materialized; no claim is made about them beyond the attention context they
produce.

Each capture records layer, operation, shape, elements, SHA-256, max-abs, RMS,
mean-abs and nonfinite count. Raw tensors stay in `target/phase19-activation/`;
the receipt binds digests and compact statistics.

## Same-weight forward comparisons

FP32 arms dispatched 0 cooperative GEMMs; mixed arms dispatched 15 forward
GEMMs, 0 shape fallbacks. Both arms report distinct bundle digests.

| Checkpoint | First divergence | Largest normalized error | Location |
| --- | --- | ---: | --- |
| initial | layer 0 `q_proj` | 9.294e-04 | layer 1 `mlp_output` |
| step-1 FP32 | layer 0 `q_proj` | 9.123e-04 | layer 1 `mlp_output` |
| step-1 mixed | layer 0 `q_proj` | 9.118e-04 | layer 1 `mlp_output` |
| step-10 | layer 0 `q_proj` | 6.856e-04 | `final_norm` |
| step-100 | layer 0 `q_proj` | 4.950e-04 | `final_norm` |

Divergence is bitwise-zero through the embedding and normalization. It first
appears in the first mixed GEMM (`q_proj`, layer 0), as expected from FP16
operand rounding; every later buffer inherits it. The largest normalized error
does not exceed `9.3e-04` at any traced checkpoint.

## Six-cell first-step cross-evaluation

All cells use the same 288 validation windows and tokenizer settings.

| Weights | Arithmetic | CE |
| --- | --- | ---: |
| initial | FP32 | 7.119862882 |
| initial | mixed | 7.119873485 |
| step-1 FP32 | FP32 | 6.757843914 |
| step-1 FP32 | mixed | 6.757825218 |
| step-1 mixed | FP32 | 6.757539822 |
| step-1 mixed | mixed | 6.757522782 |

Path decomposition of the observed step-1 gap:

- observed = `6.757522782 - 6.757843914 = -3.21e-04`
- arithmetic at fixed FP32 weights = `-1.87e-05`
- weight effect at fixed mixed arithmetic = `-3.02e-04`
- closure residual = `0.0`

**The Phase XVIII +0.302353 step-1 gap does not reproduce.** Re-evaluating the
exact exported step-1 weights gives `6.75752` (mixed) versus `6.75784` (FP32),
and the FP32 arm's step-1 CE matches Phase XVIII's `6.757844` to the last
digit. The mixed arm's training loss also matches (`7.148527`) and its step
dispatched 45 cooperative GEMMs with 15 mirror conversions. The old mixed
step-1 *evaluation* (`7.060197`) is therefore not a property of the weights; it
was an evaluation-path anomaly in that earlier run, not a 0.3-nat precision
penalty. Later Phase XVIII checkpoints remain historical measurements but are
not used here to claim a first-step causal mechanism.

## Gradient evidence

Raw gradient buffers remain unexposed; this phase compares activations only and
does not claim gradient equality. Phase XVIII's post-AdamW update comparison
stands as the available optimizer evidence.

## Verification

```powershell
python validation/qualify_activation_phase19.py --reuse --skip-gpu --install
python validation/check_activation_phase19.py
python -m unittest discover -s validation -p "test_activation_phase19.py"
```

The checker re-derives each error RMS from committed sums, recomputes first
divergence and peak location, verifies evidence digests, mode labels, dispatch
counts, checkpoint distinctness, and the six-cell closure. Seventeen controls
each fail for their intended rule.

## Limitations

- Weight packages are `target/` artifacts; CI verifies their hashes inside the
  receipt, not the package bytes.
- Raw activation tensors are retained only under `target/`.
- Softmax scores/probabilities are fused and not separately traced.
- Only one validation batch is traced per checkpoint.
- Step-10/100 comparisons use Phase XVIII weights-only exports, not exact
  optimizer-resume checkpoints.
