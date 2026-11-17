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

## FP32 non-regression against previous main (2026-11-17, Phase XLII)

The RDNA3 WMMA work must not slow the existing FP32 path. This is the
seven-workload comparison against the `main` revision the feature branch
diverged from, measured with the harness above (`bench_transformer_inference.py`)
on the qualified adapter. Both arms run the default precision mode, so this is
the `WmmaMode::Off` path.

| Item | Value |
| --- | --- |
| baseline | `main` at `ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5`, isolated detached worktree |
| candidate | `codex/rdna3-phase19-instrumented` at `5d40f6883b5b562d229db565ce09dbeafbed1c73` |
| adapter | `AMD Radeon Graphics`, `1002:15bf`, driver `8388996` (`32.0.31007.6002`) |
| build | `--release`; observer `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `-j 1`; one observer executable per arm |
| warm-up / measured | 5 warm-up forwards, then 30 timed forwards per arm per round |
| pairing | 12 alternating ABBA rounds per workload; the arm order reverses every round |
| statistic | median of the 12 per-round `candidate / stable` latency ratios |
| power | `Turbo` before and after; unchanged during the run |
| binaries | stable `2dfbd04a38e7d107…`, candidate `df84ef1f6d06f21a…`; distinct, and unchanged during the run |

`R = candidate FP32 latency / previous-main FP32 latency`; lower is faster. The
latency columns are the median over the 12 round medians of each arm.

| Workload | previous main (ms) | candidate (ms) | R | round range | relMAD | result |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `gpt2-unmasked` | 0.5556 | 0.5527 | 0.9996 | 0.9617–1.0419 | 0.010 | equivalent |
| `gpt2_wide-unmasked` | 0.8544 | 0.8537 | 1.0007 | 0.9516–1.1681 | 0.028 | equivalent |
| `gpt2_wide-padded` | 0.8488 | 0.8579 | 0.9996 | 0.9438–1.1110 | 0.016 | equivalent |
| `llama_wide-unmasked` | 1.1437 | 1.0682 | 0.9617 | 0.8884–1.0888 | 0.040 | faster |
| `llama_wide-padded` | 1.1185 | 1.1202 | 0.9903 | 0.9374–1.1190 | 0.021 | equivalent |
| `gpt2-wide-decode` | 0.6285 | 0.6258 | 0.9962 | 0.9734–1.0159 | 0.007 | equivalent |
| `gpt2-wide-batch32` | 4.1968 | 4.1887 | 0.9987 | 0.9871–1.0969 | 0.007 | equivalent |

Every ratio is at or below `1.0007`, well inside the ±2% parity band, and no
workload's median crosses it. `llama_wide-unmasked` is 3.8% faster; the other six
are equivalent. A ratio above `1.0` is reported as equivalent, never as faster.
`gpt2_wide-unmasked` and `gpt2_wide-padded` carry single-round outliers
(`1.1681`, `1.1110`) that do not move their medians; the per-round spread is why
12 rounds are used rather than 3.

FP32 output equivalence: in all 12 rounds of every workload both arms produced a
single FP32 logit digest, and the two arms' digests were bit-identical to each
other.

| Workload | `logits_fp32_sha256` (both arms, every round) |
| --- | --- |
| `gpt2-unmasked` | `954cfd192ad8f7b1f9272836391c0678ce10a998fa39f79034e7f768a276dcae` |
| `gpt2_wide-unmasked` | `b73201463483219c8412720a0824f2dd86ac1c5baa73d25dbf2188e4548c3cd8` |
| `gpt2_wide-padded` | `a6d31a27bb9cb8afcc3ca9bea1f8190b717340516b0cfa8dd75fbad7b517e21f` |
| `llama_wide-unmasked` | `45fc5f2083faf216f7c892946ea05b1eb999aebf0dec3e70d77ab390b3b83192` |
| `llama_wide-padded` | `e6ea0abb26ff596cacb606cf82176a373dd6d365c0a7442c81e07b957ffbc0dd` |
| `gpt2-wide-decode` | `f89785f841be1aab5658d9e185f67cd2632bcffdda9ca1268d52feb4709ab6cc` |
| `gpt2-wide-batch32` | `668827d6e395f34f1ae0c4ecbbb587c8c9fd5262bf69c98876fc64f9ddefff3e` |

The receipt (including every raw per-round sample and the two generated
sub-batch fixtures) is committed in
`validation/performance/fp32-nonregression/20261117-phase42/`.

Two earlier sets were recorded and are deliberately not used as the headline:
a 6-round run whose aggregate step did not complete (its raw samples are not
committed), and a second 6-round run taken while other host work was running,
which produced a `3.17` single-round outlier and three medians above `1.02`.
Both are the reason the protocol above fixes one GPU job at a time and 12
rounds. This section is a non-regression statement for FP32 latency and FP32
logits; it is not a WMMA speedup claim. Mixed training-step performance is
reported separately in `RDNA3_WMMA.md` section 5 and is slower than FP32 there.
