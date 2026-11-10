# Phase XIX — diagnostic findings (instrumentation not retained)

This phase implemented and ran activation-level tracing, then reverted it. The
taps required edits to `src/transformer.rs`, which is bound byte-for-byte by the
sealed Phase XVI, XVII and XVIII receipts. Those receipts cannot be regenerated
without re-running their GPU qualifications, and modifying them is forbidden.
The instrumented source and its receipt were therefore removed rather than
leaving evidence bound to a source state that no longer exists.

The measurements below were produced by the instrumented build before the
revert. They are diagnostic, not a source-bound qualification.

## Trace

27 tensors per forward (12 per layer plus embedding, final norm and logits),
batch 8 x sequence 64, one validation batch. Taps copied materialized buffers;
no shader arithmetic changed. FP32 arms dispatched 0 cooperative GEMMs, mixed
arms 15 forward GEMMs with 0 fallbacks.

- First bitwise divergence: layer 0 `q_proj`, at every checkpoint.
- Largest normalized activation error: `9.294e-04` (initial weights, layer 1
  `mlp_output`); `9.123e-04` at step-1 FP32; `9.118e-04` at step-1 mixed;
  `6.856e-04` at step 10; `4.950e-04` at step 100.
- Embedding and normalization buffers were bitwise equal before the first GEMM.

Softmax scores/probabilities are fused inside the attention shader and were not
traced; raw gradients remain unexposed.

## Six-cell first-step matrix

| Weights | FP32 | Mixed |
| --- | ---: | ---: |
| initial | 7.119862882 | 7.119873485 |
| step-1 FP32 | 6.757843914 | 6.757825218 |
| step-1 mixed | 6.757539822 | 6.757522782 |

Observed gap `-3.21e-04`; arithmetic at fixed FP32 weights `-1.87e-05`; weight
effect at fixed mixed arithmetic `-3.02e-04`; closure residual 0.

Phase XVIII's `+0.302353` step-1 gap did not reproduce. The FP32 arm's step-1 CE
matches Phase XVIII exactly, the mixed training loss matches (`7.148527`), and
the mixed step dispatched 45 cooperative GEMMs with 15 mirror conversions. The
old mixed step-1 *evaluation* (`7.060197`) is not a property of the exported
weights; it was an evaluation-path anomaly in that earlier run. Phase XVIII's
later checkpoints remain historical measurements.

## Required regeneration

To retain activation tracing as qualified evidence, Phase XVI current-source,
three Phase XVI training/stability arms, Phase XVII held-out and Phase XVIII
drift must be re-run against the instrumented revision, with new receipts. That
was not done here.
