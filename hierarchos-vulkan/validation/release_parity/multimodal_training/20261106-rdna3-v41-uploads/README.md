# Native V4.1 session-wide uploads

## Summary

The native V4.1 training step no longer retires its recording session for
every host upload, metadata buffer or tape snapshot. `GpuBuffer::from_f32`,
`from_u32` and `duplicate_f32` now record into the open session when one is
active, and keep their previous immediate path otherwise. Host reads still
retire the session before reading, so ordering and visibility are unchanged.
Frozen attention norm gains were shared as refcounted device views instead of
being copied once per step; this was reverted (see Correction).

Arithmetic, kernel selection, dispatch order and buffer contents are unchanged.
The attributed profile of the throughput benchmark falls from 8,748 to 5,344
session retirements end to end, and the native step falls from 956 to 241
submissions.

## Correction (2026-11-07)

The attention norm-gain sharing above was reverted. The retained-tape chain test
overwrites `q_norm` and `kv_norm` after forward and requires backward to use the
forward values; live views instead exposed the overwritten gains, failing
`attention_projection_cache_chain_and_all_factor_vjps_match_cpu`. Restoring
`duplicate_f32` snapshots fixes the test with the session on and off. The upload
path and its measurements are unchanged, and the 241-per-step submission count
is unaffected because the restored copies ride the existing session. The
[release-parity refresh](../20261107-rdna3-v41-release-parity/README.md) records
the corrected binary's nine-stage replay and new ABBA numbers. The compressor's
frozen gain is now snapshotted too; see the
[compressor-snapshot refresh](../20261107-rdna3-v41-compressor-snapshot/README.md).

## Measurement

Same-binary ABBA, four rounds, seven measured updates, two warmups, software FMA
in both arms:

| Case | Baseline, ms | Session, ms | Ratio | Submissions/step |
| --- | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 373.5460 | 255.1060 | 1.464x | 956 -> 241 |
| `native_vit_6x4` | 540.3973 | 395.6551 | 1.366x | 956 -> 241 |
| `llama` control | 17.0041 | 16.8634 | 1.008x | 5 -> 5 |
| `gpt2` control | 13.5445 | 13.6293 | 0.994x | 5 -> 5 |
| `t5` control | 39.8033 | 39.5286 | 1.007x | 7 -> 7 |

`pass: true`, `word_exact: true`, and every image and adapter digest matched
across all eight runs. The baseline arm is the same binary with
`HIERARCHOS_VULKAN_LAYER_SESSION=0`. The baseline submission count moved from
970 to 956 because the shared norm-gain copies are gone from both arms.

Against the prior receipt's session arm at the same geometries, the medians move
from 291.8544 to 255.1060 ms at 4x4 (1.144x) and from 439.3253 to 395.6551 ms at
6x4 (1.110x). Those are cross-run comparisons, not a paired arm of this run.

## Profile

`profile.log` is the attributed run over the same five cases. Host uploads and
snapshot copies no longer appear as their own retirement sites: the remaining
rows are the session's own retirement plus the three readback funnels
(`read_u32` bulk and single-value, `read_f32`) and a few shared-image control
sites. The routing-count readback funnel is unchanged by this increment.

## Verification

- Native 30-test qualifier: 30/30 pass, 1,360 logit values,
  `max_abs_logit = 0.0`, checkpoint/resume/export verified.
- Benchmark: `pass: true`, `word_exact: true`, identical digests across all
  eight runs.
- Gate unchanged: `max_abs < 2e-7`, `rtol = 0`.
- The nine-stage replay was not rerun for this increment. The corrected
  binary's replay is in the
  [release-parity refresh](../20261107-rdna3-v41-release-parity/README.md).

## Source changes

- `vulkan.rs`: `from_f32` and `from_u32` record their copy into the active
  session; `duplicate_f32` records a `copy_f32_range` transfer there. Outside a
  session both keep the immediate upload/copy boundary.
- `vulkan.rs`: the submission profiler attributes copies to their caller's
  source line instead of a fixed operation label.
- `attention_projection_grad.rs`: frozen attention norm gains were shared
  (`clone`) rather than duplicated with a device copy; this was reverted on
  2026-11-07 (see Correction). `compressor_tape.rs` now also snapshots its
  frozen gain; see the compressor-snapshot refresh.

## Reproduce

```powershell
python validation/qualify_deepseek_v41_native_pipeline.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --output target/rdna3-port/full-multimodal-20261106/v41-upload-r1 `
  --target rdna3

python validation/benchmark_rdna3_v41_session.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261106/v41-upload-ab-r2 `
  --rounds 4 --steps 7

$env:HIERARCHOS_VULKAN_SUBMISSION_REPORT = "1"
$env:HIERARCHOS_VULKAN_RDNA3_V41 = "0"
$env:HIERARCHOS_MULTIMODAL_BENCH_STEPS = "5"
$env:HIERARCHOS_MULTIMODAL_BENCH_OUTPUT = "target/rdna3-port/upload-profile-r3/bench.json"
$env:HIERARCHOS_MULTIMODAL_BENCH_FIXTURES = "target/rdna3-port/gen9-frozen/target/multimodal-gen9"
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  multimodal_training_throughput_benchmark --ignored --nocapture
```

## Files and limits

- `benchmark.json`: four-round ABBA medians, submission counts and word parity.
- `native-qualification.json`: the 30-test native gate for this binary.
- `profile.log`: attributed session-retirement profile.
- `receipt-hashes.json`: SHA-256 bindings for this receipt.

The measurements cover the frozen tiny training geometries and the published
control cases. They do not certify production-size throughput.
