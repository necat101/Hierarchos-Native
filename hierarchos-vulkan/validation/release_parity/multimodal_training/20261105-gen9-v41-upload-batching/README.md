# Gen9 parameter-upload optimization, 2026-11-05

Grouped parameter uploads improve the two frozen native V4.1 training cases by
**1.052x and 1.067x** on Intel HD Graphics 520. All **44 previously green text
profiles** retain joint image-training qualification at **`max_abs < 2e-7`,
`rtol=0`**. The final source passes 95 image cases and 166 text cases; no frozen
CPU expectations were regenerated.

## Measured change

| Native case | Individual uploads | Grouped uploads | Throughput ratio |
| --- | ---: | ---: | ---: |
| `native_vit_4x4` | 601.843 ms | 572.105 ms | 1.0520x |
| `native_vit_6x4` | 618.413 ms | 579.502 ms | 1.0671x |

Both native cases use **982 submissions per timed training step**, down from
**1,102**: 120 fewer, or **10.89%**. Measurements compare the old and new upload
boundaries in the same executable, in ABBA order across four separate processes.
Each process runs two warmup and seven timed updates. Each number is the median
of two per-process medians; raw samples and both round medians remain in
[`benchmark/benchmark.json`](benchmark/benchmark.json). No compiler, concurrent
GPU job, kernel timestamps or submission-attribution report ran during timing.
The executable uses Cargo's unoptimized test profile with debug information
disabled. These are frozen small-geometry measurements, not production-size
throughput claims. Earlier optimization timings used different build conditions;
their ratios are not multiplied into this result.

All five benchmark cases have identical final logit words, image-weight digests
and adapter-weight digests after nine updates in all four processes. Shared
LLaMA, GPT-2 and T5 are controls: their submission counts are unchanged and their
timing variation is not claimed as a speedup from this change.

## Implementation

Private `ParameterUploadBatch` groups parameters admitted by one component:
HC coefficients, attention projections and auxiliary tensors, resident/shared
experts and MoE gates, and FP32 vocabulary parameters. It records uploads in one
command batch and uses the existing host staging arena, whose chunks are 4 MiB.
Pending payload is capped at 16 MiB, with allocation rounding to arena chunks.
A tensor meeting or exceeding the budget flushes the group and keeps its
original upload boundary. Streamed experts and the packed-expert arena retain
their established paths. GPU parameters still have their individual buffers
and descriptors. This pass changes no floating-point arithmetic or shaders.

The earlier Optimization 25/26 upload attribution was too broad: attaching a
LoRA adapter to a CPU checkpoint performs no GPU upload, and each linear
projection already records its base and both factors together. This change
groups the remaining component admissions and shares host staging; it does not
require GPU-buffer views into a new parameter arena. The test-only
`HIERARCHOS_V41_INDIVIDUAL_UPLOADS=1` switch restores the old upload boundaries
for comparison. Production always uses the bounded grouped path.

## Final-source parity and regression

| Gate | Passing cases | Worst visible-logit difference |
| --- | ---: | ---: |
| Shared image training, 40 profiles | 77 | `1.7136335372924805e-07` |
| Joint BART/T5/Switch, 3 profiles | 6 | `5.960464477539063e-08` |
| Native DeepSeek V4.1, final profile | 2 | `0` |
| Native Gemma4 raw patches | 1 | `1.341104507446289e-07` |
| Native Gemma4 RGB8 | 1 | `1.341104507446289e-07` |
| Native Gemma3/SigLIP | 4 | `1.9650906324386597e-07` |
| Native CLIP/LLaVA | 4 | `7.450580596923828e-08` |
| Resident/streaming text regression | 166 | `1.7881393432617188e-07` |

All image cases execute two joint AdamW updates, image sensitivity, gradient
flow and native export/reload checks. The shared and encoder-decoder pipelines
train their text parameters and full image tower/projector. Native V4.1 trains
the full native ViT, aligner and image delimiters with frozen-language-base LoRA.
Its 30-execution qualifier also covers optimizer checkpoint/resume, continued
updates, corrupt/foreign-state rejection, stale-gradient rejection, text
controls and forced host vector width 16. Its 1,866 comparison rows cover
155,444 values, with 155,179 word-exact, worst other difference
`1.4901161193847656e-8`, and all 1,360 logits word-exact.

The broader DeepSeek suite verifies **215 tests**, with **26 opt-in tests
ignored**. The initial invocation returned 101: 214 passed and one test failed
while writing its diagnostic trace to a missing directory in the isolated
build tree, after every numerical comparison had matched. Creating that output
directory and rerunning the sole failed test on the same executable passed.
[`deepseek-suite-qualification.json`](deepseek-suite-qualification.json), the
original [`logs/deepseek-suite.log`](logs/deepseek-suite.log), and
[`logs/schedule-output-recovery.log`](logs/schedule-output-recovery.log) retain
this setup failure and its recovery separately. There is no claim that the
original entire suite invocation returned zero.

The upload unit test checks signed zero and finite values word-for-word,
submission boundaries at a small budget, and actual reuse of one staging chunk.
The broader tests cover dense/packed, resident/streamed paths and resource
release. All **672 shader recipes** rebuild and validate, including the inherited
cached HC-adjoint variant; all **81 historical V4.1 SPVs and 127 CPU fixtures**
remain byte-identical. Some native Gemma diagnostics remain above the tolerance
at intermediate tensors; their visible logits and updated parameters pass.

[`qualification.json`](qualification.json) binds all component receipts and the
44-profile inventory. [`native-source-sha256.json`](native-source-sha256.json)
records the source used by both the libtest executable and all five regression
probes. Inventory metadata and this new evidence were promoted afterward;
compiled Rust, shipped shaders and frozen oracles did not change.

## Replay

Build the locked library tests and the five probes listed in the
[baseline replay guide](../20261104-gen9-full-image-training/README.md#reproduction).
Use an explicit executable, a fresh output directory, and serial GPU jobs.
From the repository root:

```text
python hierarchos-vulkan/validation/benchmark_deepseek_v41_training.py --binary PATH_TO_LIBTEST_EXE --fixtures target/multimodal-gen9 --output NEW_BENCHMARK_DIRECTORY --rounds 2 --steps 7
python hierarchos-vulkan/validation/qualify_deepseek_v41_native_pipeline.py --binary PATH_TO_LIBTEST_EXE --output NEW_NATIVE_QUALIFICATION_DIRECTORY
```

The benchmark pins the executable hash and requires exactly one benchmark test
and all five case records on every invocation. Its maintained runner SHA-256 is
`95b7a60d7f79168b38676d50047c4c0acc46c5f1ae5e4ad1426083029bacf71d`; the measured executable SHA-256 is
`0567b0266abd25265408087bce41da0455c51bd8d819737cde5dcc3650e7f23b`. Follow the baseline guide for the 77 shared,
6 encoder-decoder, 10 native and 166 text regression cases. On isolated builds,
create `target/deepseek-v41-reference` next to the compiled crate before running
the broader suite's trace-writing schedule test.

## RDNA3 handoff and limits

Physical RDNA3 replay remains pending; that machine was not accessed. The local
branch is `codex/gen9-multimodal-training` at base
`4c67507cab6fe86b334e2709c1260ee81320d061`. No commit or push was made. The
previously synchronized RDNA3 GPT-2 fix/README and the November 4 qualification
and transfer archives are preserved.

The new `target/multimodal-gen9/gen9-multimodal-upload-20261105-source.zip`
contains changed and untracked source plus current evidence and per-file hashes.
It reuses the unchanged November 4 oracle archive, SHA-256
`f52a62d55790728e232f45feac3ebce1f42d0d42844a34bcab9dfa3aa62b519a`.
The matching November 5 transfer JSON records the new source archive's hash.
Apply/reconcile against the recorded base, rebuild on RDNA3, use `--target rdna3`
where supported, and set `HIERARCHOS_ATEN_VECTOR_WIDTH=8` for the Gen9 AVX2 oracle
bundle. Native V4.1 pins its private math scope and runs on the local Vulkan GPU.
Retain fresh receipts before marking physical replay qualified.

Qualification remains limited to frozen FP32 geometry. Production-size parity
and throughput, BF16 training, audio/video and dynamic image shapes remain
unqualified. Native V4.1 accepts one complete image span per prefill and one
fixed patch grid per trainer. Shared towers implement the authorized image
training contract for text backbones; their coverage does not establish parity
with separately named upstream VL models. Existing native towers remain
available and pass their own regressions.
