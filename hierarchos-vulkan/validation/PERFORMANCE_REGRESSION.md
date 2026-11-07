# RDNA3 V4.1 performance regression guide

This guide covers the frozen five-case native V4.1 training benchmark and the
three comparisons that must stay separate:

1. historical `372407b` versus the optimized tree;
2. session OFF versus session ON in the same optimized binary;
3. before versus after a new optimization in the same source lineage.

Do not describe the session ratio as a historical speedup, and do not use the
tiny frozen training geometry as a production-throughput claim.

## Measurement protocol

The runner is `validation/benchmark_rdna3_v41_paired.py`. Every measured arm:

- runs the same executable test, `multimodal_training_throughput_benchmark`;
- uses `HIERARCHOS_ATEN_VECTOR_WIDTH=16` and software FMA
  (`HIERARCHOS_VULKAN_RDNA3_V41=0`) in both arms;
- performs two warmup updates and seven measured updates per case;
- runs six ABBA rounds (the order reverses every round);
- keeps every raw JSON sample and child-process log;
- checks that final logits, image digests and adapter digests are word-identical
  within each arm and, unless explicitly allowed, across arms;
- reports the median, p95, relative MAD of round medians, sample CV, submission
  count, and a paired bootstrap 95% confidence interval for the speedup.

The confidence interval is the performance gate. There is no fixed one-percent
failure threshold. A native optimization passes only when the 95% interval lies
wholly above `1.0`; a control regression is reported only when the interval lies
wholly below `1.0`. An inconclusive control interval is not evidence of a
regression and must not be converted into one.

Run one GPU job at a time. Build and finish all CPU work before starting a
benchmark. The host must be on its normal power profile, and the receipt records
the adapter name returned by the benchmark plus the binary hashes. A run with a
changed binary, missing case, wrong step count, non-finite sample, signature
change, or incomplete test result fails closed.

## Commands

Before/after optimization, same host and harness:

```powershell
python validation/benchmark_rdna3_v41_paired.py `
  --baseline-binary target/rdna3-port/before-adapter-batch-2e9c.exe `
  --candidate-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/abba-before-after `
  --rounds 6 --steps 7 `
  --scope "adapter-span batching before vs after" `
  --require-native-improvement --forbid-control-regression
```

Same optimized binary, session OFF versus ON:

```powershell
python validation/benchmark_rdna3_v41_paired.py `
  --baseline-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --candidate-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --baseline-env HIERARCHOS_VULKAN_LAYER_SESSION=0 `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/abba-session `
  --rounds 6 --steps 7 `
  --scope "reverse-graph session OFF vs ON" `
  --require-native-improvement --forbid-control-regression
```

Historical tree versus optimized tree:

```powershell
python validation/benchmark_rdna3_v41_paired.py `
  --baseline-binary target/rdna3-port/baseline-372407b.exe `
  --candidate-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/abba-historical `
  --rounds 6 --steps 7 `
  --scope "historical 372407b vs optimized final" `
  --allow-cross-arm-signature-difference `
  --require-native-improvement --forbid-control-regression
```

`--allow-cross-arm-signature-difference` is only for an explicitly documented
historical semantic difference. It does not relax within-arm word exactness.
The current historical comparison has a one-to-two-ULP GPT-2 difference caused
by code between `372407b` and the optimized tree; the four native/control cases
other than GPT-2 are word-exact. The receipt records the mismatch rather than
hiding it.

## Correctness gates

Performance is meaningful only after these pass:

```powershell
cargo test --locked --lib gathered_adapter_spans_preserve_individual_words_and_order
cargo test --locked --lib deepseek_v41_compressor_tape_snapshots_norm_gain
cargo test --locked --lib deepseek_v41_compressor_decode_tape_snapshots_norm_gain
python validation/qualify_rdna3_full_multimodal.py `
  --frozen target/rdna3-port/gen9-frozen `
  --output target/rdna3-port/full-replay `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe
python validation/qualify_rdna3_native_multimodal.py `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --output target/rdna3-port/native-multimodal
python -m unittest discover -s validation -p "test_*.py"
```

The DeepSeek inventory changes when a test is added. The current tree is exactly
244 enumerated tests, 218 active and 26 ignored; the qualifier derives that list
from the test binary instead of assuming the previous 217/243 counts. Existing
sealed receipts keep their original counts and must not be rewritten.

The compressor acceptance gate remains `max_abs < 2e-7` with `rtol = 0`. Gen9
frozen files, protected fixtures and prior receipt contents are read-only.

## Interpreting a result

- **native improvement**: the entire speedup interval is above 1.0.
- **control regression**: the entire speedup interval is below 1.0.
- **inconclusive**: the interval crosses 1.0; report it as inconclusive.
- **dispersion**: relative MAD is the round-median scatter. A large value means
  the host was not stable enough for a narrow conclusion, even if the point
  estimate looks favorable.
- **p95**: per-update latency tail over all measured samples, not a guarantee for
  production batch sizes.

The raw files are the evidence. Summaries may be regenerated, but the JSON
samples and child logs must not be edited after a run. A failed or superseded
attempt stays in the local work directory with its error and is not silently
reused as a passing receipt.
