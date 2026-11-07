# AVX512 / RDNA3 GPT2 shape qualification

Physical target: Ryzen Z1 Extreme, CPUID `00a70f41`, Windows Torch
`2.6.0+cpu`, Radeon `1002:15bf`, driver `00800184`.
This is a worktree qualification; no new commit or publication is implied.

All sixteen cases pass the unchanged strict `max_abs < 2e-7`, `rtol=0`
policy through two composed updates, image sensitivity, RGB gradients,
trainable-parameter checks, export and reload. The largest required GPT2 logit
error is `5.960464477539063e-8`. Fifteen final-logit comparisons have zero
numeric error; the original batch-2 / length-8 / four-head unmasked control
has that maximum error. Original expected values and acceptance gates were not
changed.

| Batch | Sequence | Heads | Unmasked | Padded |
| ---: | ---: | ---: | --- | --- |
| 2 | 8 | 4 | PASS | PASS |
| 1 | 8 | 4 | PASS | PASS |
| 3 | 8 | 4 | PASS | PASS |
| 2 | 6 | 4 | PASS | PASS |
| 2 | 10 | 4 | PASS | PASS |
| 2 | 16 | 4 | PASS | PASS |
| 2 | 8 | 2 | PASS | PASS |
| 2 | 8 | 8 | PASS | PASS |

All have hidden width 64, inner width 256, vocabulary 320, two text blocks,
and the actual trainable CLIP tower/projector. This certifies eight geometries,
not their Cartesian product or production-size models. The control CPU
fixtures reproduce the original alternate, both training-step and final CPU
logits word-for-word before any GPU replay.

## Numerical contracts and dispatch

The retained CPU model uses **SDPA**. The earlier eager operation-local
diagnostic did not establish the CPU model's backend. The corrected tools
include padding and require exact frozen CPU replay; the text observer can
also load CPU step-1 parameters to diagnose the second update.

The separate GPT2 modules preserve the measured CPU GEMM and SDPA schedules,
CRT `logf`, MKL piecewise `tanh`, the sum operator's Vec8 cascade reduction,
and MKL AdamW square-root staging. CPU capability `AVX512` alone does not
identify each operator's arithmetic. The instruction seed table exhaustively
covers all mantissas in the two normalized exponent bins; the log tables are
generated from mathematical constants. The tanh coefficients come from the
fingerprinted Torch DLL. None are corrections fitted to model values.

Nine primitive tests pass. These include sixteen SDPA cases, three CLIP
adjoints, six cross-entropy cases, twelve input-gradient GEMMs, eighty bias
reductions, 8,192 NewGELU forward/backward values, 6,147 scalar-log inputs and
thirty-six AdamW updates with complete parameter/moment/variance states.
Attention comparisons permit the two signs of zero; other primitive arrays
require raw-word equality. Negative admission tests reject unmeasured attention
features and shapes.

The new family requires the pinned GPU/driver and CPU profile. Its eight
attention geometries have explicit admission checks. Separate artifacts and
fallbacks preserve the existing paths; the original shared backward shader and
the original RDNA3 short-forward source are byte-preserved, with the inherited
GPT2 edits moved into separate source files.

## Timings

Eight paired ABBA rounds, thirty-two dispatches per batch, persistent buffers
and warm pipelines produced 128 GPU timestamp samples. Every serial/candidate
output is word-identical. Times below are batch medians, not model throughput.

| Sequence | Kernel | Serial control, ms | Selected, ms | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 8 | forward, eight-thread packing | 3.35154 | 3.11800 | 1.075x |
| 8 | backward, padded LDS reuse | 12.93330 | 2.45902 | 5.260x |
| 16 | forward, serial rows retained | 1.09132 | 1.08868 | 1.002x |
| 16 | backward, padded LDS reuse | 22.05122 | 2.08128 | 10.595x |

The workgroup sweep includes 1/2/4/8/16/32/64 threads. The original packed-64
candidate regressed forward timing and was rejected. Eight-thread forward
packing is selected only for batch 2 / sequence 8 / four heads. Other forward
shapes retain serial rows. Backward uses 64 threads with a padded 17-float LDS
stride and no float atomics. The timed packed-forward artifact is byte-identical
to the selected production artifact. No end-to-end or production-size throughput
claim follows from these isolated measurements.

## Retained regressions and provenance

The final binaries pass 74 shared cases, three Kimi cases, six seq2seq cases,
four native Gemma3 cases, two native Gemma4 cases and four native LLaVA cases.
Native V4.1 passes all thirty tests and its two-case shared-inventory bridge,
with zero logit error. The union retains all 44 shared profiles and 85 image
cases. V4.1 keeps its established frozen-language-base LoRA contract.

All 43 RDNA3 artifacts in the local glslang 15.3 qualification set rebuild
byte-for-byte and pass SPIR-V validation. The upstream preservation receipt
checks 1,341 shader/fixture files and 672 original recipes with no changes.
No physical Gen9 execution was performed.

`receipt-hashes.json` binds every retained result and the relevant source,
artifact, include and primitive-fixture bytes. `cpu-manifest.json` binds the
independently frozen model cases and source snapshots. The new receipt guard
also derives the case/profile totals, checks the strict logit limits and checks
primitive, provenance and GPU timing coverage. Three negative controls verify
rejection of missing cases, false subchecks and relaxed/boundary acceptance.

## Reproduce

Run from `hierarchos-vulkan`:

```powershell
& .venv-torch/Scripts/python.exe validation/check_rdna3_gpt2_receipts.py
& .venv-torch/Scripts/python.exe -m unittest discover -s validation -p test_rdna3_gpt2_receipts.py
cargo test --lib rdna3_gpt2_ -- --nocapture --test-threads=1
& .venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_shapes.py run --fixtures target/rdna3-port/gpt2-shapes-final-cpu --binary target/debug/image_text_pipeline_probe.exe --output target/rdna3-port/gpt2-shapes-replay
& .venv-torch/Scripts/python.exe validation/benchmark_rdna3_gpt2.py --output target/rdna3-port/gpt2-timing-replay
```

The frozen oracle bundle is
`target/rdna3-port/gpt2-shapes-avx512-oracles.zip`; its SHA-256 and size are in
`summary.json`. It contains all sixteen cases, models, optimizer expectations
and source snapshots. Restore these frozen expectations for regression replay.
The CPU-only preparation command uses the maintained CPU environment and the
original AVX512 shared fixture's `gpt2_wide/unmasked/model` directory; its sibling
masked/unmasked CPU controls must also be present. CPU preparation enforces SDPA
and original-control equality. It does not consume native output.

Audio/video and production geometry remain outside this receipt.
