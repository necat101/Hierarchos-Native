# Native V4.1 resident experts: routing-readback finding

## Question

Can the remaining per-layer routing-count readbacks be batched into one
transfer per step?

## Finding

No, not under streamed experts. `apply_with_text_mask_routed` reads
`counts.read_u32(c.experts)` once per MoE layer, and the host uses that result
to decide which expert payloads to load from the checkpoint. The angularity is
`forward -> gate counts -> upload set -> expert dispatch`, so the transfer is a
true per-layer producer/consumer dependency. Collapsing it needs either every
expert uploaded every step or device-side payload selection; neither is a
batching change.

The profile confirms the size: 192 reads over the benchmark (12 MoE layers x 16
forwards, two native cases), reported as 27.43 per profile step.

The fused `PackedExpertBatch` path is the one design that removes the counts
read while keeping streamed uploads bounded, but it is not available here:
`PackedExpertBatch::upload` refuses on this RDNA3 adapter, so `fused` stays
`None` even with resident experts and `HIERARCHOS_VULKAN_V41_FUSED_EXPERTS=1`.
Forcing that flag changed neither the 192 reads nor the 169 submissions.

## Measured alternative: resident experts

Resident experts do remove the per-step expert uploads and 72 of the 241
submissions. Same binary, four ABBA rounds, seven measured updates, two
warmups, software FMA:

| Case | Streamed, ms | Resident, ms | Ratio | Submissions/step | Peak live buffers, streamed -> resident |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 240.907 | 205.099 | 1.175x | 241 -> 169 | 9.09 -> 9.23 MB |
| `native_vit_6x4` | 307.671 | 273.383 | 1.125x | 241 -> 169 | 9.14 -> 9.28 MB |
| `llama` control | 13.306 | 13.394 | 0.993x | 5 -> 5 | - |
| `gpt2` control | 10.934 | 10.908 | 1.002x | 5 -> 5 | - |
| `t5` control | 32.853 | 32.687 | 1.005x | 7 -> 7 | - |

`pass: true`, observed `max_abs = 0.0`, and the shared controls are unchanged.
Unlike the session ablation, this one is also bit-exact: every image and adapter
digest matched between the arms. Peak live buffer bytes are the benchmark's
`device.memory_stats()` value, 9.09 -> 9.23 MB at 4x4 and 9.14 -> 9.28 MB at
6x4. The frozen fixture's expert payloads are small, so the resident delta is
only about 0.14 MB here; it scales with the real expert payload size.

The attributed profile drops from 5,344 retirements (streamed) to 3,232
(resident). The 192 counts reads remain in both.

## Verification

- Native 30-test qualifier under resident experts: 30/30 pass, 1,360 logit
  values, `max_abs_logit = 0.0`, checkpoint/resume/export verified.
- Frozen gate unchanged: `max_abs < 2e-7`, `rtol = 0`.
- Nine-stage replay was not rerun; the previous receipt remains the latest full
  matrix and applies to that binary.

## Scope and state

- The residency switch is benchmark-only: `setup` reads
  `HIERARCHOS_MULTIMODAL_BENCH_RESIDENT_EXPERTS` and every other path keeps the
  production streamed configuration. No default changed.
- The benchmark record now carries `memory.peak_live_buffer_bytes`.
- Read attribution (`read_f32`, `read_u32`, `read_many_f32`) is test-only and is
  what produced the 192/27.43 attribution above.

## Reproduce

```powershell
python validation/benchmark_rdna3_v41_residency.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261106/v41-resident-ab-r1 `
  --rounds 4 --steps 7

$env:HIERARCHOS_MULTIMODAL_BENCH_RESIDENT_EXPERTS = "1"
python validation/qualify_deepseek_v41_native_pipeline.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --output target/rdna3-port/full-multimodal-20261106/v41-resident-qualifier-r1 `
  --target rdna3
```
