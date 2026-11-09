# RDNA3 mixed-WMMA paired optimizer-step benchmark: 2026-11-09

**Verdict: INCONCLUSIVE. No end-to-end speedup is claimed.**

This is the first whole-optimizer-step measurement taken *after* the versioned
FP16 mirror cache landed. The earlier `1.087×` figure predates the cache and is
not restated as a result here.

## Method

Same fixture and inputs for both arms (Llama dense 1024, 16.78M parameters,
LoRA rank 8, batch 8, sequence 64, lr 1e-2, seed 7). Five paired ABBA rounds,
two warmups excluded, every run exited 0:

`off 4, mixed 4, mixed 48, off 48` per round. Per-step latency is estimated as
`(t48 − t4) / 44`, which cancels fixed startup cost but not thermal drift.
Timer is process wall clock. Binary:

`target/release/hierarchos-vulkan-transformer-train.exe`
SHA-256 recorded in `benchmark.json`.

## Results

| Round | FP32 ms/step | mixed ms/step | FP32 / mixed |
| ---: | ---: | ---: | ---: |
| 1 | 678.43 | 625.98 | 1.0838 |
| 2 | 683.25 | 719.05 | 0.9502 |
| 3 | 689.16 | 712.52 | 0.9672 |
| 4 | 690.70 | 651.05 | 1.0609 |
| 5 | 690.86 | 747.14 | 0.9247 |

| Measure | FP32 | mixed |
| --- | ---: | ---: |
| Median ms/step | 689.16 | 712.52 |
| Min / max | 678.43 / 690.86 | 625.98 / 747.14 |

Geometric mean ratio **0.9954×**, 95% CI **0.9123–1.0860** (t, df = 4). The
interval contains 1.0, so the two arms are not distinguishable at this sample
size. Note the dispersion: the FP32 arm is stable within 1.8%, while the mixed
arm ranges over 19%. That spread, not a code difference, drives the interval.

## Telemetry from the mixed arm

Per 48-step run: `cooperative_dispatches=2160` (forward 720, weight_grad 720,
input_grad 720; 15 projections per step), `shape_fallbacks=0`,
`weight_conversions=720`, `fp32_bytes_read=3019898880`, `mirror_reuses=720`.
One conversion per parameter per step with one reuse, i.e. the cache halves
conversions relative to the pre-cache path, but that saving is small next to
the per-step GPU work.

## Limitations

- Process wall clock, not GPU timestamps; no GPU temperature, clock or power
  telemetry was captured on this host.
- The per-step estimator removes startup cost but inherits thermal drift; the
  19% mixed spread is consistent with throttling on a handheld APU.
- Five rounds only.
- No FP16-storage-without-cooperative-matrices arm exists for this workload,
  so the requested three-way comparison is two arms.

## What this means

Mixed WMMA is numerically qualified and functionally correct on this path, but
it is **not** shown to be faster end-to-end. Claiming a speedup would contradict
this measurement. The prior `1.087×` figure should be treated as unreproduced
on this session and superseded by this INCONCLUSIVE result.
