# Native V4.1 batch-2 decode-snapshot release parity

## Summary

The rebuilt binary
(`5ae2f32da93a97bd4a67b66c734405145595765c248590d386bb7c5a9ba04205`)
passes the complete nine-stage replay on this RDNA3 host:

- all nine published stages green, `pass: true`, `complete: true`
- 217/217 active DeepSeek tests (243 enumerated, 26 ignored)
- 30/30 native V4.1 tests, `max_abs_logit = 0.0`
- 2,748 frozen files unchanged
- frozen manifest SHA-256
  `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`
- strict gate unchanged: `max_abs < 2e-7`, `rtol = 0`

This receipt supersedes the
[decode-snapshot refresh](../20261107-rdna3-v41-decode-snapshot/README.md),
which is bound to the prior `0bd45568...` binary and ran the decode test at batch
1 only. The 243/217 inventory is unchanged: the existing decode regression now
drives both batch 1 and batch 2.

## Batch-2 decode coverage

The ratio-2 sequence remains prefill two tokens, decode the pending cached token
at position 2, then decode position 3 to complete one group. The retained tape
from that completed step is run backward once as recorded, the live norm gain is
overwritten with `17.0`, and the same tape is run backward again.

The test now runs that sequence twice, at batch 1 and batch 2. Batch 2 supplies a
two-row upstream with one `dim` vector per row, checks a `batch * hidden`
gradient readback, and therefore exercises the independent per-row pool state
and the absence of cross-row mixing. The same-tape and restarted-sequence checks
apply to both batch sizes:

1. Same-tape gradients must agree within `2e-7` after the live gain changes.
2. A tape captured after the overwrite must differ by more than `1e-4`, proving
   the test does not pass by ignoring the gain.

`compressor_tape.rs` captures `p.norm_weight.duplicate_f32(c.dim)?` and snapshots
the ratio-2 pool state, so neither the gain nor the cached state is a live view.

## Measurement

Same-binary ABBA, four rounds, seven measured updates, two warmups, software FMA
in both arms:

| Case | Streamed, ms | Resident, ms | Ratio | Submissions/step |
| --- | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 312.3666 | 262.8894 | 1.188x | 241 -> 169 |
| `native_vit_6x4` | 324.0386 | 271.9519 | 1.192x | 241 -> 169 |
| `llama` control | 13.4326 | 13.2098 | 1.017x | 5 -> 5 |
| `gpt2` control | 10.6366 | 10.6002 | 1.003x | 5 -> 5 |
| `t5` control | 29.8029 | 30.1516 | 0.988x | 7 -> 7 |

`pass: true`, `digests_identical: true`, and `observed_max_abs = 0.0`. Absolute
medians vary between runs, so only the paired ratio is claimed. The resident arm
is a benchmark-only switch; the production default remains streamed.

## Profile

`profile.log` is the default streamed arm with seven measured updates, two
warmups and the submission report enabled. Its top attributed site records
6,816 events (757.33/step); the adapter-gradient readback is 1,296 (144/step).
The submission count remains 241 per step.

## Verification

- `full-replay-qualification.json`: nine stages, all green, 2,748 frozen files,
  manifest hash above.
- The replay binds the test binary and all five fixed probes to their recorded
  SHA-256 values throughout the run.
- `deepseek-suite-qualification.json`: 243 expected tests, 217 verified passes,
  26 ignored, `binary_sha256` equal to the replay binary.
- `native-qualification.json`: 30/30 tests, `max_abs_logit = 0.0`;
  `native-run-status.json` records checkpoint, resume and export.
- `benchmark.json`: `pass: true`, identical digests, `observed_max_abs = 0.0`.
- `compressor-regression-tests.log`: both prefill and decode snapshot tests pass
  (`2 passed`); the decode case covers batch 1 and batch 2.

## Source changes

- `src/deepseek_v41/compressor_tape_tests.rs`: the cached-decode regression now
  runs its full sequence at batch 1 and batch 2 through one helper, with batch
  size in the pending, completed, drift and sensitivity assertions.
- No production compressor code changed in this revision; the norm and pool-state
  snapshots are the implementation introduced by the preceding receipt.
- No suite inventory gates changed: 243 enumerated / 217 active / 26 ignored.

## Reproduce

```powershell
python validation/qualify_rdna3_full_multimodal.py `
  --frozen target/rdna3-port/gen9-frozen `
  --output target/rdna3-port/full-multimodal-20261107/batch2-decode-snapshot-r2 `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe

python validation/benchmark_rdna3_v41_residency.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261107/batch2-decode-snapshot-ab-r1 `
  --rounds 4 --steps 7
```

The profile command sets `HIERARCHOS_VULKAN_SUBMISSION_REPORT=1`,
`HIERARCHOS_MULTIMODAL_BENCH_STEPS=7`, `HIERARCHOS_MULTIMODAL_BENCH_WARMUP=2`
and the frozen fixture/output paths, then runs
`multimodal_training_throughput_benchmark --ignored --nocapture`.

## Files and limits

- `full-replay-qualification.json`, `full-replay.log`: the nine-stage replay.
- `deepseek-suite-qualification.json`, `deepseek-suite.log`: the 217-test stage.
- `compressor-regression-tests.log`: both compressor-tape snapshot tests.
- `native-qualification.json`, `native-run-status.json`,
  `native-qualifier.log`: the 30-test native gate.
- `benchmark.json`, `benchmark.log`: the same-binary streamed/resident ABBA.
- `profile.log`: the attributed default-arm submission profile.
- `receipt-hashes.json`: SHA-256 bindings for this receipt.

The measurements cover the frozen tiny training geometries and the published
replay matrix. They do not certify production-size throughput. The batch-2 test
checks gradient sensitivity on this adapter, not every possible batch geometry.
