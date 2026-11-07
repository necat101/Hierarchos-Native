# Native V4.1 compressor-snapshot release parity

## Summary

The current binary
(`ff5e4806c0158617f2ee3739c7a7c4fcbc0d67b2b14f0b10829e8652092b39e4`)
passes the complete nine-stage replay on this RDNA3 host:

- all nine published stages green, `pass: true`, `complete: true`
- 216/216 active DeepSeek tests (242 enumerated, 26 ignored)
- 30/30 native V4.1 tests, `max_abs_logit = 0.0`
- 2,748 frozen files unchanged
- frozen manifest SHA-256
  `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`
- strict gate unchanged: `max_abs < 2e-7`, `rtol = 0`

This receipt supersedes the
[earlier release-parity refresh](../20261107-rdna3-v41-release-parity/README.md),
which is bound to the prior `60f9fc91...` binary. The new inventory includes
`deepseek_v41::compressor::tape_tests::deepseek_v41_compressor_tape_snapshots_norm_gain`.

## Compressor snapshot correction

The compressor tape had the same live-view pattern that previously broke the
attention projection tape: it retained `p.norm_weight.clone()` for backward.
The new regression test overwrites the live compressor norm gain after forward
and requires backward to use the value forward consumed. A second tape captured
after the overwrite must produce a different gradient, so the test cannot pass
by ignoring the gain.

`compressor_tape.rs` now captures `p.norm_weight.duplicate_f32(c.dim)?`.
The attention projection tape already snapshots its gains from the prior
refresh. The DeepSeek suite inventory constants were updated from 241/215 to
242/216 (26 ignored), and the replay gate now requires 216 passes.

## Measurement

Same-binary ABBA, four rounds, seven measured updates, two warmups, software FMA
in both arms:

| Case | Streamed, ms | Resident, ms | Ratio | Submissions/step |
| --- | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 240.0778 | 204.5660 | 1.174x | 241 -> 169 |
| `native_vit_6x4` | 371.0949 | 337.8474 | 1.098x | 241 -> 169 |
| `llama` control | 16.3582 | 16.4265 | 0.996x | 5 -> 5 |
| `gpt2` control | 13.5915 | 13.7484 | 0.989x | 5 -> 5 |
| `t5` control | 38.1535 | 38.1363 | 1.000x | 7 -> 7 |

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
- `deepseek-suite-qualification.json`: 242 expected tests, 216 verified passes,
  26 ignored, `binary_sha256` equal to the replay binary.
- `native-qualification.json`: 30/30 tests, `max_abs_logit = 0.0`;
  `native-run-status.json` records checkpoint, resume and export.
- `benchmark.json`: `pass: true`, identical digests, `observed_max_abs = 0.0`.
- `compressor-regression-test.log`: the new snapshot test passes.

## Source changes

- `src/deepseek_v41/compressor_tape.rs`: captures the frozen compressor norm
  gain with `duplicate_f32` instead of a refcounted live view.
- `src/deepseek_v41/compressor_tape_tests.rs`: new post-forward overwrite
  regression test and sensitivity check.
- `validation/qualify_rdna3_deepseek_suite.py`,
  `validation/qualify_rdna3_full_multimodal.py`: inventory gates updated to
  242 total / 216 active / 26 ignored.

## Reproduce

```powershell
python validation/qualify_rdna3_full_multimodal.py `
  --frozen target/rdna3-port/gen9-frozen `
  --output target/rdna3-port/full-multimodal-20261106/final-replay-current-r3 `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe

python validation/benchmark_rdna3_v41_residency.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261106/v41-compressor-snapshot-ab-r1 `
  --rounds 4 --steps 7
```

The profile command sets `HIERARCHOS_VULKAN_SUBMISSION_REPORT=1`,
`HIERARCHOS_MULTIMODAL_BENCH_STEPS=7`, `HIERARCHOS_MULTIMODAL_BENCH_WARMUP=2`
and the frozen fixture/output paths, then runs
`multimodal_training_throughput_benchmark --ignored --nocapture`.

## Files and limits

- `full-replay-qualification.json`, `full-replay.log`: the nine-stage replay.
- `deepseek-suite-qualification.json`, `deepseek-suite.log`: the 216-test stage.
- `compressor-regression-test.log`: the new compressor-tape check.
- `native-qualification.json`, `native-run-status.json`,
  `native-qualifier.log`: the 30-test native gate.
- `benchmark.json`, `benchmark.log`: the same-binary streamed/resident ABBA.
- `profile.log`: the attributed default-arm submission profile.
- `receipt-hashes.json`: SHA-256 bindings for this receipt.

The measurements cover the frozen tiny training geometries and the published
replay matrix. They do not certify production-size throughput.
