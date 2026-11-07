# Native V4.1 decode-snapshot release parity

## Summary

The rebuilt binary
(`0bd455687ed18f3198af8caa2ca6231960c82e79df37d552e50a974f0bfb0a8c`)
passes the complete nine-stage replay on this RDNA3 host:

- all nine published stages green, `pass: true`, `complete: true`
- 217/217 active DeepSeek tests (243 enumerated, 26 ignored)
- 30/30 native V4.1 tests, `max_abs_logit = 0.0`
- 2,748 frozen files unchanged
- frozen manifest SHA-256
  `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`
- strict gate unchanged: `max_abs < 2e-7`, `rtol = 0`

This receipt supersedes the
[compressor-snapshot refresh](../20261107-rdna3-v41-compressor-snapshot/README.md),
which is bound to the prior `ff5e4806...` binary and its 216-test inventory. The
new inventory includes the decode-path regression
`deepseek_v41::compressor::tape_tests::deepseek_v41_compressor_decode_tape_snapshots_norm_gain`.

## Decode snapshot coverage

The prefill test already overwrites the live compressor norm gain after a
retained forward and requires backward to use the captured value. The new test
covers the cached decode path, where the completed group is separated from the
pending cached step:

1. Prefill two tokens at ratio 2.
2. Decode position 2, which completes no group but updates the pool state.
3. Decode position 3, which completes one group and retains its tape.
4. Record `grad_input` from that tape, overwrite the live gain with `17.0`, and
   run the same tape backward again.

The recorded and snapshot gradients must agree within `2e-7`. The test then
restarts the sequence after the overwrite and reaches the same completed group;
that gradient must differ by more than `1e-4`, so it cannot pass by ignoring the
gain. `compressor_tape.rs` captures `p.norm_weight.duplicate_f32(c.dim)?` and
also snapshots ratio-2 pool state, so neither the gain nor the cached state is a
live view.

The DeepSeek suite inventory constants were updated from 242/216 to 243/217
(26 ignored), and the replay gate now requires 217 passes.

## Measurement

Same-binary ABBA, four rounds, seven measured updates, two warmups, software FMA
in both arms:

| Case | Streamed, ms | Resident, ms | Ratio | Submissions/step |
| --- | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 253.0860 | 214.1784 | 1.182x | 241 -> 169 |
| `native_vit_6x4` | 390.7058 | 358.8235 | 1.089x | 241 -> 169 |
| `llama` control | 17.1112 | 17.5834 | 0.973x | 5 -> 5 |
| `gpt2` control | 13.3223 | 13.9087 | 0.958x | 5 -> 5 |
| `t5` control | 40.0532 | 39.2578 | 1.020x | 7 -> 7 |

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
- The replay binds the model binary and all five fixed probes to their recorded
  SHA-256 values throughout the run.
- `deepseek-suite-qualification.json`: 243 expected tests, 217 verified passes,
  26 ignored, `binary_sha256` equal to the replay binary.
- `native-qualification.json`: 30/30 tests, `max_abs_logit = 0.0`;
  `native-run-status.json` records checkpoint, resume and export.
- `benchmark.json`: `pass: true`, identical digests, `observed_max_abs = 0.0`.
- `compressor-regression-tests.log`: both prefill and decode snapshot tests pass
  (`2 passed`).

## Source changes

- `src/deepseek_v41/compressor_tape_tests.rs`: new ratio-2 cached-decode
  regression test that completes a group, overwrites the live norm gain, and
  checks both snapshot stability and post-overwrite sensitivity.
- `validation/qualify_rdna3_deepseek_suite.py`,
  `validation/qualify_rdna3_full_multimodal.py`: inventory gates updated to
  243 total / 217 active / 26 ignored.
- `src/deepseek_v41/compressor_tape.rs`: retains the `duplicate_f32` norm and
  state snapshots introduced by the preceding compressor-snapshot refresh.

## Reproduce

```powershell
python validation/qualify_rdna3_full_multimodal.py `
  --frozen target/rdna3-port/gen9-frozen `
  --output target/rdna3-port/full-multimodal-20261107/decode-snapshot-r1 `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe

python validation/benchmark_rdna3_v41_residency.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261107/decode-snapshot-ab-r1 `
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
replay matrix. They do not certify production-size throughput.
