# Track A: where mixed-WMMA step time actually goes

## Method

Host timing was added around every queue submission (`vkQueueSubmit` plus the
fence wait) and sampled around each optimizer step, next to the existing GPU
timestamp profiler. Both arms are the real trainer on the Llama-dense fixture
(batch 8, seq 64, LoRA rank 8, alpha 16, 12 steps, alternating order), so wall,
submission and GPU time all come from one process rather than two benchmark
arms. Medians are over steps 3-12.

## Result

| Arm | wall ms | submissions/step | host submit ms | fence wait ms | residual ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `off` | 608.61 | 1.0 | 0.07 | 608.14 | 0.40 |
| `mixed` | 563.90 | 1.0 | 0.07 | 563.31 | 0.53 |

- **One queue submission per optimizer step in both modes.** The mixed path adds
  no submissions, no extra command-buffer boundaries and no additional fences.
- **Host submit cost is 0.07 ms per step.** Recording the whole step's commands
  costs about 0.5 ms; descriptor and pipeline binds are not measurable here.
- **99.9% of the step is the fence wait.** The host blocks on the GPU for the
  whole step, so the host cannot be on the critical path.

## Consequence for the earlier number

The "137-147 ms per step outside dispatched GPU work" figure from the previous
report does not survive same-run measurement. It came from subtracting a GPU
timestamp sum collected in an *instrumented* run from a wall clock collected in
a *different timed* run, which mixes timing domains. Within one run the profiled
submission span for the mixed arm is 563.9 ms and the same-run wall clock is
563.90 ms: the GPU span covers the submission, and the residual is under a
millisecond.

Track A therefore has **no host-side optimization to land**. Submissions are
already one per step, host submit cost is negligible, and there is no stall to
remove without inventing one.

## The remaining lead, stated as unresolved

In this same-binary 12-step pair the mixed arm was **44.7 ms faster** than FP32
(563.90 vs 608.61). The 48-step ABBA measurement found the opposite sign: mixed
75.9 ms slower (0.9051x, CI 0.8542-0.9591). Both were measured on the same host
and binary family; they differ in protocol and run length.

The 48-step ABBA design should govern any headline, because it warm-starts each
arm and measures steady state. The 12-step pair suggests the mixed arm is faster
before steady state, which means something accumulates across steps. Candidates,
none of them measured yet: per-step mirror refresh (the weight epoch advances
every optimizer step, so all 15 projections reconvert per step), scratch or
descriptor growth, or longer-arm clock drift. This is not a host overhead and it
is not explained here.

## Files

`host-off-12.txt`, `host-mixed-12.txt` (and the two 2-step warmups) are the raw
per-step lines from that instrumented build, `meta.txt` records the binary
hash, and `results.txt` the exit status of every arm.
