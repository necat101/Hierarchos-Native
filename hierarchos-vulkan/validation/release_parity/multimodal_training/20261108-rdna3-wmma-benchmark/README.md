# RDNA3 mixed-WMMA performance investigation: 2026-11-09

**Verdict: mixed WMMA is a statistically significant regression for LoRA and
statistically indistinguishable from FP32 for full fine-tuning. It should stay
off by default, and it is not an accelerator on this workload.**

## Paired optimizer-step results (5 ABBA rounds, warmups excluded, all runs exit 0)

Per-step latency is `(t48 - t4) / 44`, which cancels fixed startup but not thermal drift.

| Workload | FP32 median ms/step | mixed median ms/step | FP32 / mixed | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Full fine-tuning | 689.16 | 712.52 | **0.9954x** | 0.9123-1.086 |
| LoRA rank 8 (frozen base) | 629.64 | 705.52 | **0.9051x** | 0.8542-0.9591 |

The LoRA interval lies entirely below 1.0: mixed precision costs about 9.5% per
step on this workload. The full-fine-tuning interval straddles 1.0 and is
**INCONCLUSIVE**. The earlier `1.087x` figure predates the mirror cache and is
not reproduced by either workload.

The previous benchmark was labelled LoRA but ran with no `--lora-rank`, so it was
full fine-tuning; its 720 `weight_grad` dispatches are correct for that mode. The
trainer now prints `WMMA mode: training=... weight_grad_consistent=...` so the
mode cannot be misread from dispatch counts again.

## Why: the step is attention, not GEMM

Per-step GPU attribution (medians of 12 profiled optimizer steps,
`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`):

| Arm | GPU dispatch sum ms | Wall step ms | Non-dispatch ms | GPU share |
| --- | ---: | ---: | ---: | ---: |
| `fp32-lora` | 608.9 | 629.6 | 20.7 | 96.7% |
| `mixed-lora` | 558.7 | 705.5 | 146.8 | 79.2% |
| `fp32-full` | 651.9 | 689.2 | 37.3 | 94.6% |
| `mixed-full` | 576.5 | 712.5 | 136.0 | 80.9% |

Top kernels, FP32 full fine-tuning (sum of the measured step, not a normalised
share):

| Kernel | ms/step | Share |
| --- | ---: | ---: |
| `transformer_attention_backward` | 472.4 | 73.4% |
| `transformer_attention_forward` | 53.2 | 8.3% |
| `linear_weight_grad` | 40.3 | 6.3% |
| `linear_input_grad` | 34.1 | 5.3% |
| `linear_forward` | 33.8 | 5.2% |
| `adamw` | 4.4 | 0.7% |
| `transformer_rms_norm_param_grad` | 1.8 | 0.3% |
| `transformer_rms_norm_forward` | 1.6 | 0.2% |

`transformer_attention_backward` alone is **472 ms of a
689 ms step**. Both attention kernels run
`local_size_x = 1`: one invocation per (batch, KV head) executes every query and
key serially, so a 12-CU part runs ~32 single-thread workgroups. All projections
together are about 115 ms, so no projection optimisation can recover the step.

That is also why mixed precision cannot win here. In LoRA the WMMA path replaces
~115 ms of projection work with ~43 ms of `wmma_gemm_f16_f32acc`, yet the step
gets **slower**: the GPU dispatch sum falls while wall time rises, which places
the loss in submission, barrier and host scheduling around the extra WMMA
dispatches, not in the kernels.

## What was not done

A parallel attention backward kernel is the single change that would move this
step by hundreds of milliseconds, and it is a kernel project in its own right -
it has to reproduce the scalar kernel's reduction order to keep the qualified
FP32 gates, or be qualified as a separate numerically distinct path. It was not
attempted here rather than risk the numerical contract. The existing parallel
`transformer_attention_short_amd_rdna3` kernel is gated at `seq <= 16` and has no
backward counterpart, so it cannot serve a 64-token step.

## Limitations

- Process wall clock plus GPU timestamps; no GPU temperature, clock or power
  telemetry is available on this host.
- Five paired rounds. The LoRA CI is tight enough to exclude parity; the full-FT
  CI is not.
- Profiling runs are separate from the timed runs, as they must be.

## Track A: the non-dispatch gap was a measurement artifact

Keying host timing to the actual submissions showed the step is **one queue
submission** in both modes, with 0.07 ms of host submit time and 99.9% of the
step spent blocked in the fence wait. The previously reported 137-147 ms of
"non-dispatch" time came from subtracting an instrumented run's GPU sum from a
different run's wall clock; in a single run the submission span (563.9 ms) and
the wall clock (563.90 ms) agree to 0.03 ms. There is no host submission overhead
to remove. Details and the raw logs: `track_a/`.
