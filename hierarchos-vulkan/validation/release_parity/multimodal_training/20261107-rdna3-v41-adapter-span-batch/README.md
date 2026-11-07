# Adapter-span batch optimization receipt

## Summary

Source commit: `722acd09626c592ead883b8c7aa652dd0d6ceeac` (parent `24424bf`).

The native V4.1 mixed-MoE backward pass now retains each visited expert's small
adapter-gradient spans and performs one gathered device-to-host read after the
whole layer, instead of one read per expert projection. The large input-gradient
buffers are released before the gather; accumulation order remains shared expert,
then routed experts in reverse creation order, then `w1`, `w2`, `w3` within each
expert.

The optimized test binary is
`8ec108f10107e2e104b987e57c81bd433e125d21f4274e6687dbe500fa274c48`.
The three performance comparisons are kept separate below and are not
interchangeable.

## Optimization result

Same host, same five frozen cases, two warmups, seven measured updates, six ABBA
rounds, software FMA in both arms. Confidence intervals are paired bootstrap
95% intervals over round-level median ratios.

### Before versus after this optimization

| Case | Before, ms | After, ms | Speedup | 95% CI | p95 before/after | Relative MAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 258.4650 | 247.2001 | **1.0456x** | 1.0312–1.0533 | 271.16 / 259.85 ms | 0.40% / 0.36% |
| `native_vit_6x4` | 269.3806 | 257.1283 | **1.0477x** | 1.0328–1.0587 | 285.89 / 263.93 ms | 0.45% / 0.34% |
| `llama` control | 12.0621 | 12.0934 | 0.9974x | 0.9824–1.0571 | 13.44 / 12.90 ms | 0.52% / 1.48% |
| `gpt2` control | 9.5263 | 9.4335 | 1.0098x | 0.9963–1.0183 | 10.42 / 10.08 ms | 0.74% / 0.28% |
| `t5` control | 28.1578 | 28.2356 | 0.9972x | 0.9893–1.0119 | 29.26 / 29.36 ms | 0.59% / 0.72% |

Native submissions fall from 241 to 187 per step. The control intervals cross
1.0; there is no statistically significant control regression. Final logits,
image digests and adapter digests are word-identical in every run.

### Session OFF versus ON in the optimized binary

| Case | OFF, ms | ON, ms | Speedup | 95% CI | Submissions/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 389.6883 | 248.7083 | **1.5668x** | 1.5316–1.5875 | 952 -> 187 |
| `native_vit_6x4` | 404.6915 | 263.0636 | **1.5384x** | 1.5236–1.5526 | 952 -> 187 |
| `llama` control | 12.1092 | 12.5116 | 0.9678x | 0.9526–1.0080 | 5 -> 5 |
| `gpt2` control | 9.5652 | 9.5332 | 1.0034x | 0.9926–1.0421 | 5 -> 5 |
| `t5` control | 28.7058 | 28.5479 | 1.0055x | 0.9814–1.0135 | 7 -> 7 |

This is a same-binary ablation of reverse-graph session granularity. It is not a
historical speedup. The Llama interval is inconclusive, not a significant
regression.

### Historical `372407b` versus optimized final

| Case | `372407b`, ms | Final, ms | Speedup | 95% CI | Submissions/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 383.1195 | 248.3555 | **1.5426x** | 1.5339–1.5914 | 982 -> 187 |
| `native_vit_6x4` | 400.2438 | 257.0319 | **1.5572x** | 1.5305–1.5789 | 982 -> 187 |
| `llama` control | 12.0831 | 12.0314 | 1.0043x | 0.9825–1.0104 | 5 -> 5 |
| `gpt2` control | 10.8538 | 9.3880 | 1.1561x | 1.1056–1.2159 | 5 -> 5 |
| `t5` control | 28.4169 | 28.3327 | 1.0030x | 1.0001–1.0186 | 7 -> 7 |

The historical GPT-2 final words differ by one or two ULP in a few logits; this
predates adapter batching and is recorded as the only cross-arm mismatch. The
other four cases are word-identical. The historical timing comparison uses the
same five cases, fixtures, precision, warmups, steps, and software-FMA arm; the
difference is disclosed rather than hidden.

## End-to-end prefill and decode

Frozen V4.1 forward-probe checkpoint, paired baseline `372407b` versus current,
three alternating pairs, two measured steps, final-row digest guards. These are
batch-1 probes; they do not establish production batch-2/4 throughput.

| Residency | Phase | `372407b`, ms | Final, ms | Reduction |
| --- | --- | ---: | ---: | ---: |
| resident | prefill | 565.8670 | 565.9875 | -0.02% |
| resident | decode | 257.6855 | 255.0955 | 1.01% |
| streamed/default | prefill | 1958.9080 | 1901.2935 | **2.94%** |
| streamed/default | decode | 742.2735 | 722.1300 | **2.71%** |

All frozen digests matched. The streamed/default arm is the shipping loader; the
resident prefill result is neutral within noise.

## Profile

The optimized streamed profile still records 630.4 session-retirement
submissions per native step. The gathered adapter read is 72 events per step
over the five-case run; the next attributed sites are 72.6, 28.8 and 20.8 events.
The optimization removed the per-expert adapter readback submissions, but the
remaining session retirements are the dominant residual and were not changed
without a transfer-hazard design.

## Correctness matrix

| Gate | Result |
| --- | --- |
| Nine-stage Gen9 frozen replay | 9/9 pass, 2,748 frozen files unchanged |
| DeepSeek suite | 218/218 active, 244 enumerated, 26 ignored |
| Native V4.1 | 30/30, `max_abs_logit = 0.0` |
| Native AVX-512 multimodal track | 8/8 stages; 95 retained and 16 expanded cases |
| Compressor snapshot regressions | 2/2 pass, strict `max_abs < 2e-7`, `rtol = 0` |
| Python multimodal suites | 33 passed, 3 skipped |
| Adapter-span word/order regression | 1/1 pass |
| Formatting and diff checks | `cargo fmt --check` and `git diff --check` pass |

The compressor decode log includes batch-2 and batch-4 state-chain rows and the
live gain/cache overwrite isolation checks. Batch-2/4 are correctness gates; the
five-case training benchmark and forward probe are batch 1, so no batch-2/4
throughput claim is made.

## Files

- `benchmark-before-after.json`, `before-after/`: six-round raw A/B/before/after data.
- `benchmark-session.json`, `session/`: six-round same-binary session ablation.
- `benchmark-historical.json`, `historical/`: six-round `372407b` comparison.
- `prefill-decode/`, `correctness/prefill-decode.json`: paired prefill/decode raw logs.
- `correctness/full-replay-qualification.json`: nine-stage replay summary.
- `correctness/deepseek-suite-qualification.json`: 244/218/26 inventory and shard rows.
- `correctness/native-v41-qualification.json`: 30/30 native V4.1 result.
- `correctness/native-multimodal-qualification.json`: eight-stage native track.
- `correctness/compressor-regression-tests.log`: both strict snapshot tests.
- `correctness/adapter-span-test.log`: exact-word/order regression.
- `correctness/python-unittest.log`: 33 tests, 3 documented skips.
- `profile.log`, `profile.json`: optimized submission profile.
- `binaries.json`, `build.json`, `hardware.json`, `commit.json`: identities.
- `source-sha256.json`: every tracked source/artifact file at the source commit.
- `receipt-hashes.json`: integrity manifest for this directory.

## Hardware and build

- AMD Ryzen Z1 Extreme, 8 cores / 16 threads.
- AMD Radeon Graphics, PCI `1002:15bf`, driver `32.0.31007.6002`.
- Windows 11 Enterprise 10.0.26300.
- rustc 1.93.0, cargo 1.93.0, `x86_64-pc-windows-msvc`, LLVM 21.1.8.
- Test profile (`cargo test --locked --lib --no-run`), no release-only claim.
- `HIERARCHOS_ATEN_VECTOR_WIDTH=16`; `HIERARCHOS_VULKAN_RDNA3_V41=0`.

## Reproduce

The commands and interpretation rules are documented in
`validation/PERFORMANCE_REGRESSION.md`. Use fresh output directories, one GPU job
at a time, and the same fixtures:

```powershell
python validation/benchmark_rdna3_v41_paired.py `
  --baseline-binary target/rdna3-port/before-adapter-batch-2e9c.exe `
  --candidate-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/abba-before-after `
  --rounds 6 --steps 7 --require-native-improvement --forbid-control-regression
```

Run `python validation/check_rdna3_v41_optimization_receipt.py` to verify this
directory. It re-derives every median and speedup from raw samples, checks the
binary and source hashes, enforces native intervals and control non-regression,
allows only the documented historical GPT-2 mismatch, and checks the correctness
inventories.

## Limits

These are tiny frozen training geometries, not production-size throughput.
The full native performance fixtures are batch 1, and the forward probe is batch
1; batch-2/4 correctness is covered but batch-2/4 throughput is not claimed.
Physical Intel Gen9 execution remains outside this RDNA3 run. Prior sealed
receipts remain unchanged and retain their original 217/243 inventory.
