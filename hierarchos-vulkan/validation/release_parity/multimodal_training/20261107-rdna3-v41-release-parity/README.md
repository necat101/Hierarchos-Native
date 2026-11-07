# Native V4.1 release-parity refresh

## Summary

The current binary
(`60f9fc91527495e3ba762eb143e832b6f7675aaa80c4ac3cff7bd153e05d942b`)
passes the complete nine-stage replay on this RDNA3 host. All nine published
stages are green, including 215/215 active DeepSeek tests and 30/30 native V4.1
tests. The 2,748-file frozen manifest is unchanged, and the strict gate remains
`max_abs < 2e-7` with `rtol = 0`.

This refresh closes a regression introduced by the session-wide upload
increment. `attention_projection_grad.rs` had replaced the retained tape's
q/kv norm snapshots with refcounted live views. The DeepSeek chain test
overwrites `q_norm` and `kv_norm` after forward and requires backward to use
the forward values, so the pre-fix binary failed
`deepseek_v41::attention::projection_grad::tests::attention_projection_cache_chain_and_all_factor_vjps_match_cpu`
with dX errors up to 4.699e-1 and adapter-factor errors up to 1.651. Restoring
`duplicate_f32` snapshots fixes the test with the session on and off, and the
whole matrix is green on the rebuilt binary. `compressor_tape.rs` still retains
its frozen gain as a refcounted view; it has no post-forward mutation path.

## Measurement

Same-binary ABBA, four rounds, seven measured updates, two warmups, software FMA
in both arms:

| Case | Streamed, ms | Resident, ms | Ratio | Submissions/step |
| --- | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 251.6107 | 214.1620 | 1.175x | 241 -> 169 |
| `native_vit_6x4` | 395.3938 | 347.9104 | 1.136x | 241 -> 169 |
| `llama` control | 16.9548 | 16.9468 | 1.000x | 5 -> 5 |
| `gpt2` control | 13.7665 | 14.2263 | 0.968x | 5 -> 5 |
| `t5` control | 39.9112 | 39.8649 | 1.001x | 7 -> 7 |

`pass: true`, `digests_identical: true`, and `observed_max_abs = 0.0`. Absolute
medians vary between runs, so the paired ratio is the measurement; this receipt
does not claim a cross-run speedup. The resident arm is a benchmark-only switch;
the production default remains streamed.

## Profile

`profile.log` is the default streamed arm with seven measured updates, two
warmups and the submission report enabled. Its top attributed site records
6,816 events (757.33/step); the adapter-gradient readback is 1,296 (144/step).
The submission count remains 241 per step, unchanged from the upload receipt.

## Verification

- `full-replay-qualification.json`: `pass: true`, `complete: true`, nine stages,
  all 2,748 frozen files unchanged, manifest
  `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`.
- The replay binds every stage to the binary above; the model binary and all
  five fixed probes kept their recorded SHA-256 values throughout the run.
- `deepseek-suite-qualification.json`: 241 expected tests, 215 verified passes,
  26 ignored, `binary_sha256` equal to the replay binary.
- `native-qualification.json`: 30/30 tests, `max_abs_logit = 0.0`;
  `native-run-status.json` records checkpoint, resume and export.
- `benchmark.json`: `pass: true`, identical digests and `observed_max_abs = 0.0`.
- `snapshot-regression-test.log`: the formerly failing chain test passes with
  the session on and with `HIERARCHOS_VULKAN_LAYER_SESSION=0`.

## Source change

- `src/deepseek_v41/attention_projection_grad.rs`: the attention projection
  tape snapshots `q_norm` and `kv_norm` with `duplicate_f32` again, so backward
  consumes the gains forward used even if the live buffers are overwritten.

No other source changed for this refresh. The compressor tape's frozen
norm-gain view is unchanged.

## Reproduce

```powershell
python validation/qualify_rdna3_full_multimodal.py `
  --frozen target/rdna3-port/gen9-frozen `
  --output target/rdna3-port/full-multimodal-20261106/final-replay-current-r2 `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe

python validation/benchmark_rdna3_v41_residency.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261106/v41-snapshot-fix-ab-r1 `
  --rounds 4 --steps 7
```

The profile command sets `HIERARCHOS_VULKAN_SUBMISSION_REPORT=1`,
`HIERARCHOS_MULTIMODAL_BENCH_STEPS=7`, `HIERARCHOS_MULTIMODAL_BENCH_WARMUP=2`
and the frozen fixture/output paths, then runs
`multimodal_training_throughput_benchmark --ignored --nocapture`.

## Files and limits

- `full-replay-qualification.json`, `full-replay.log`: the nine-stage replay.
- `deepseek-suite-qualification.json`, `deepseek-suite.log`: the 215-test stage.
- `native-qualification.json`, `native-run-status.json`,
  `native-qualifier.log`: the 30-test native gate.
- `benchmark.json`, `benchmark.log`: the same-binary streamed/resident ABBA.
- `profile.log`: the attributed default-arm submission profile.
- `snapshot-regression-test.log`: the restored-snapshot regression check.
- `receipt-hashes.json`: SHA-256 bindings for this receipt.

The measurements cover the frozen tiny training geometries and the published
replay matrix. They do not certify production-size throughput.
