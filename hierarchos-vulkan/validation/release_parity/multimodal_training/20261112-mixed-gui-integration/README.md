# Mixed desktop precision integration — 2026-11-12

Overall unified release status: **INCOMPLETE**. The additional mixed GUI component is **PASS_SCOPED**.

Tested runtime source: `fabbd2f6f2e9d154ea709b0e0e81e3644022af4f`.
Main/merge base: `ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5`.
Fetched WMMA development baseline: `a2d90ffb8970184afeff4e34b3a20559ea97ebba`.
The protected historical branch remains at `0483bcab17c0ad01872afb8bd5aa8bdc90e6b657`.

## Current execution

| Component | Result |
| --- | --- |
| GUI unit tests | 15 passed, 3 opt-in tests ignored |
| GUI Start hardware test | 1 passed; four actual workers: vision, full Transformer, full fine-tuning, LoRA |
| CLI unit tests | 50 passed, 0 failed/ignored |
| Rust Vulkan library suite | 1,150 passed, 2 failed, 40 ignored; all 1,152 active tests executed |
| Original frozen FP32 image matrix | 93/93 passed: shared 77, encoder-decoder 6, LLaVA 4, Gemma3 4, Gemma4 2 |
| Frozen text comparisons | 164/166 passed |
| Native V4.1 separate execution matrix | 30/30 passed; overlaps the library suite, not 30 additional unique tests |
| Shader provenance | All 727 artifacts reproduced byte-for-byte with the pinned toolchains |
| Additional mixed component | Full, LoRA, and joint image training; two optimizer updates per run; FP32 exports; physical cooperative dispatches |

The actual GUI workers used the newly built native CLI. In the cross-mode fixture, full text training recorded 42 cooperative dispatches (14 in each phase), LoRA recorded 28 (14 forward, 14 input-gradient, zero base-weight gradients), and vision CLI training recorded 94 (30 forward, 32 weight-gradient, 32 input-gradient). LoRA's zero base-weight-gradient count is expected.

Both FP32 and mixed vision CLI checkpoints matched the direct probe's image and text parameters exactly. Live/reloaded mixed logits and image features were bit-identical after the image-mirror invalidation fix. Full/LoRA mixed losses agreed with their FP32 counterparts within the retained mixed limits and decreased over the two updates. Actual fallback counts remain in the reports.

## Remaining release failures and limits

- Packed target-parameter LoRA merge unit test fails with incorrectly associated/layout-dependent deltas.
- The padded wide GPT-2 oracle fails at `2.086162567138672e-7` in resident and streaming modes, above the unchanged strict `< 2e-7` limit.
- The additional 128-wide CPU comparisons are diagnostic and remain unqualified: logits reach `4.172325134277344e-7`. They do not replace the original frozen oracles or extend the original FP32 PASS scope.
- Phase XIX-Z still rejects the changed packed-cell shader binding. Its training/stability/held-out/drift/activation evidence was not regenerated here.
- The 13 packed-cell benchmark/guard pairs and immutable-main runtime baseline were not rerun here.
- The broader native mixed-architecture matrix, long convergence and clean-machine packaging gates remain unexecuted.
- Automatic loss scaling and native DeepSeek V4.1 WMMA training are unimplemented; the GUI exposes the corresponding boundaries.
- CI status is separate from these local results. No complete release PASS is asserted.

## Evidence and reproduction

`current-qualification.json` binds component receipts, executables, environment, failure identities and artifact hashes. `mixed-qualification.json` binds 1,577 source/dependency files, generated fixture/oracle hashes, executed commands and numerical observations. Its source, evidence and binary bindings were checked again during assembly. Each image component was independently checked with the maintained complete-matrix coverage/policy verifier and against its actual executable hash. Every file in the original frozen HANDOFF manifest reverified with zero errors.

Large numerical payloads are losslessly gzip-compressed with deterministic headers. Their raw SHA-256 values and sizes are retained in `compressed_artifacts`. Read them with Python's `gzip.open(path, "rt")` and `json.load`; the source qualification scripts continue to produce ordinary JSON. Temporary model/checkpoint and build-cache files remain in local `target/` directories and are not committed.

Reproduce the additional component from the repository root with the new locked release builds:

```text
.phase19w-venv/Scripts/python.exe hierarchos-vulkan/validation/qualify_mixed_gui_integration.py --output target/NEW_RUN --probe hierarchos-vulkan/target/release/image_text_pipeline_probe.exe --cli hierarchos-vulkan/target/release/hierarchos-native-cli.exe
```

Use the maintained `qualify_rdna3_full_multimodal.py` and its component runners against the immutable frozen Gen9 handoff for the original matrix. Do not substitute generated GPU outputs for those oracles. See `RDNA3_MIXED_GUI.md` for the precision contract and limitations.
