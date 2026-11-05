# Full Gen9 image-training qualification, 2026-11-04

The final source passes the strict visible-logit and updated-parameter gate
`max_abs < 2e-7`, `rtol=0`, for all 44 previously green text profiles. This is
85 backbone image cases plus 10 separate native-tower regression cases and
166 resident/streaming text cases. The inventory maps every profile to its
passing cases. `qualification.json` binds the component receipts and inventory.

| Path | Cases | Worst visible logit |
| --- | ---: | ---: |
| Shared full-parameter CLIP + projector + text | 77 | 1.7136335372924805e-7 |
| Real BART/T5/Switch encoder-decoder image training | 6 | 5.960464477539063e-8 |
| Native DeepSeek V4.1 ViT/aligner/delimiters + language LoRA | 2 | 0 |
| Native Gemma4 | 2 | 1.341104507446289e-7 |
| Native Gemma3/SigLIP | 4 | 1.9650906324386597e-7 |
| Native CLIP/LLaVA | 4 | 7.450580596923828e-8 |
| Text regression | 166 | 1.7881393432617188e-7 |

Image cases execute two actual joint AdamW updates and native export/reload.
The shared and encoder-decoder paths train text parameters and the full image
tower/projector. V4.1 retains its frozen-language-base LoRA contract and trains
every native image parameter. Checkpoint/resume testing covers V4.1 weights,
AdamW moments, continued updates and byte-exact reloaded logits. This does not
extend optimizer-resume claims to the other native image families.

## V4.1 implementation and evidence

`DeepseekV41MultimodalTrainer` in `src/deepseek_v41/vision_train.rs` runs the
released model's native ViT, 2-D rotary attention, spatial aligner, learned
image-span delimiters, HC-repeat adjoint, Engram image mask and image-specific
MoE correction biases. Its public operations include `gradients`,
`apply_gradients`, `train_step`, `forward_logits`, `save_checkpoint`, `resume`,
`export_native` and `from_native_export`. Construction takes the checkpoint,
language adapter/options and a fixed patch-grid height/width. Patches and span
types follow the pinned released processor, rather than a shared tower.

The unchanged CPU fixture executes the exact released model/vision/processor
at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, with Torch's pinned
Windows AVX2, single-thread FP32 arithmetic. Its SHA-256 remains
`9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b`.
The official sources, including Engram and license, are preserved with hashes
under `validation/reference/deepseek_v41_vl/`.

The final uninterrupted trajectory compares 155,444 values in 1,866 rows, with
155,179 word-exact values and worst difference 1.4901161193847656e-8. All 1,360
training/inference logits are word-exact, including after the second update.
These runs retain native parameters between updates; the separate CPU-state
forward diagnostic is not used as a qualification substitute.

Native-only kernels reproduce measured AVX2 matrix panels, reduction tails,
unary boundaries and adjoints. Input VJPs preserve CPU autograd's accumulation
order for trained LoRA base/adapter branches across experts, window attention,
query projections and compression. Existing GPU allocations are reused; no
CPU numerical fallback was added. The private nested thread-local scope pins
native image math to width 8 without changing ordinary text dispatch or its
buffer retention. Thirty final test executions pass, including repeated native
vision/joint checks with the host width forced to 16. The four text-modality
controls and old FP32 training/resume/export gate also pass.

All 81 historical V4.1 SPVs and 127 historical CPU fixtures are unchanged.
`shader-verification.json` records successful rebuilding and SPIR-V validation
of 671 variants; their artifact hashes were rechecked at promotion. Python
pipeline controls pass 22 tests, with three pre-existing optional integration
tests skipped; the native runtime qualifiers independently execute those image
pipelines. `git diff --check` passes.

## Reproduction

Build from the crate with its locked dependency set. The libtest package was
built with debug=0 to fit this laptop; no arithmetic or optimization flag was
relaxed. Cargo's parent-Git scan stalls in this checkout, so the recorded runs
used an identical source copy outside any Git ancestor and the same target
directory. The compiled-source hash manifest is included here. Both the
libtest executable and five regression probes were built from that same map.
The inventory and new evidence files were promoted afterward; Rust, shaders
and frozen CPU inputs did not change during qualification.

Build the library tests, then pass the just-built executable explicitly:

```text
cargo test --locked --offline --lib deepseek_v41_native --no-run
python validation/qualify_deepseek_v41_native_pipeline.py --binary PATH_TO_LIBTEST_EXE --output NEW_OUTPUT_DIRECTORY
```

For the other gates, build `image_text_pipeline_probe`, `gemma4_vision_probe`,
`gemma3_vision_probe`, `llava_pipeline_probe` and
`hierarchos-vulkan-transformer-logits`. Each corresponding qualifier accepts
`run --fixtures FROZEN_ROOT --binary PROBE --output NEW_DIRECTORY --target gen9`:

| Qualifier under `validation/` | Frozen root relative to the repository |
| --- | --- |
| `qualify_shared_image_pipeline.py` | `target/multimodal-gen9/shared-image-cpu-v3` |
| `qualify_seq2seq_image_pipeline.py` | `target/multimodal-gen9/seq2seq-image-cpu-v1` |
| `qualify_gemma4_pipeline.py` | `validation/tmp-gemma4-fixtures/gen4-cpu` within the crate |
| `qualify_gemma4_pipeline.py` | `target/multimodal-gen9/gemma4-rgb-cpu` |
| `qualify_gemma3_pipeline.py` | `target/multimodal-gen9/gemma3-cpu` |
| `qualify_llava_pipeline.py` | `target/multimodal-gen9/llava-cpu` |
| `qualify_gen9_architectures.py` | `target/gen9-phase3/cpu-fixtures-reproduced` |

Run GPU jobs serially. Frozen manifests verify their inputs. Preserve failed
receipts and write every run to a new directory. `source-sha256` and executable
hashes bind the maintained results to the tested source and binaries.

## RDNA3 handoff and limits

The RDNA3 machine was not accessed; physical replay is pending. The local
branch is `codex/gen9-multimodal-training`, based on
`4c67507cab6fe86b334e2709c1260ee81320d061`. The earlier RDNA3 GPT-2 fix and
README synchronization at `d41ebeee295d598840438ae03757419d624985ac` remain
preserved. No commit or push was made.

Transfer archives are prepared under `target/multimodal-gen9/`: the
`gen9-full-image-training-20261104-source.zip` contains changed/new source and
evidence plus a file-hash manifest, and the matching `-oracles.zip` preserves
the seven generated frozen oracle roots above. The source archive includes
untracked implementation files; a tracked-only Git diff is insufficient.
Apply/reconcile the source against its recorded base on the other machine,
rebuild there and rerun the qualifiers. For the shared/native/text matrix use
`--target rdna3`, and set `HIERARCHOS_ATEN_VECTOR_WIDTH=8` when replaying these
Gen9 AVX2 oracle bundles. The V4.1 native scope itself pins width 8 and its
qualifier runs the local Vulkan device. Do not mark the physical replay green
until the new machine's complete receipts pass.

Qualification is for frozen tiny FP32 geometry and training contracts, not
production-size parity or throughput, BF16 training, audio/video or dynamic
image shapes. Shared-tower coverage establishes image training for the listed
text backbone, not equivalence to a separately named upstream VL model.
Existing native Gemma4/Gemma3/LLaVA paths remain available and pass their own
regressions. Some of those native paths retain above-limit intermediate
diagnostics; their visible logits and updated parameters pass the required
strict gates. V4.1 accepts one complete image span per prefill and a fixed
patch grid per trainer. Its released BF16 processor patches are explicitly
promoted to FP32 at the native qualification boundary.
