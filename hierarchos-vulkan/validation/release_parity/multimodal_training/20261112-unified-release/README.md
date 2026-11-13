# Unified multimodal and WMMA release verification - 2026-11-12

Implementation revision: `719b9c5430bf1ac15471b7d4e9024d832d7fdbfa`
Branch: `codex/rdna3-phase19-instrumented`
Device: `AMD Radeon Graphics [pci=1002:15bf;driver=00800184]`

This is a current-source verification record, not a claim that Phase XIX-Z
current-source qualification is complete. The existing Phase XIX-Z checker
remains **BLOCKED** on stale source bindings and is not replaced here. Raw
receipts live under the repository's ignored `target/`; this document records
their status and the exact scope that was run.

## FP32 multimodal

| Scope | Status | Result |
|---|---|---|
| 93 maintained image cases (`shared77`, `seq2seq`, `gemma4-legacy`, `gemma4-rgb`, `gemma3`, `llava`) | **PASS** | 93/93 cases; all six stages exit 0; strict main gate `max_abs < 2e-7`, `rtol=0` |
| 166 text cases | **PASS** | 166/166; strict maximum `1.7881393432617188e-07` |
| Strict intermediate vision/feature diagnostics | **FAIL / UNQUALIFIED** | 194 of 1,116 diagnostic checks fail; worst `4.57763671875e-05` (Gemma4 vision features). These are not part of the frozen main gate and are not hidden. |

Receipts: `target/unified-release-images-719b9c5/qualification.json`,
`target/unified-fp32-text166-final-719b9c5.log`.

Image binaries: `image_text_pipeline_probe.exe`
`49b7b2baa410103ce22c822ab60b301da5495c4d7ed2bb2c685e0502260cf318`,
`llava_pipeline_probe.exe`
`3dfc64ebf79070ba38523a89edab6982d80d94b8bd8600d9fbe87fd54c32eba0`,
`gemma3_vision_probe.exe`
`afbede84ec70a5f4f6548b5c4ae9becea959c9eaa336501e1c6de3628d2caae1`,
`gemma4_vision_probe.exe`
`dec5024754bfa0e55c5fd11b45829eb7b3dc51daf6fe04e5d64e211b26125714`,
test runner `46e1616020dc0c45f6dd5f2aa84831a4ceaa00cd3d7ddc20a65646eb3f01c3a4`.

Native V4.1 bounded Mixed remains **PASS_BOUNDED_NATIVE_VISION** with 65 strict
rows (maximum `1.564621925354004e-07`), 32 cooperative dispatches and exact
reload. The separate stress fixture remains **FAIL / UNQUALIFIED** with 65
strict failures (maximum `4.76837158203125e-06`). Receipt:
`target/unified-release-v41-final/qualification.json`.

## Mixed WMMA

| Scope | Status | Result |
|---|---|---|
| Llama64 full-parameter and LoRA training | **PASS_SCOPED** | 20 steps each; 480/320 cooperative dispatches; FP32 masters and optimizer state; loss scaling enabled |
| Shared CLIP64 image/text CLI and direct probe | **PASS_SCOPED** | 94 CLI cooperative dispatches (30 forward, 32 dW, 32 dX); six documented shape fallbacks remain FP32; image/text parameter equality `0.0` against the direct probe |
| T5, BART, Switch wide cases (6) | **PASS_SCOPED** | 6/6 pass; 352 cooperative dispatches |
| LLaVA wide | **PASS_SCOPED** | Cooperative forward/dW/dX confirmed |
| Gemma3 wide | **PASS_SCOPED** | Cooperative forward/dW/dX confirmed; its raw FP32 diagnostics exceed strict `2e-7` |
| Gemma4 wide RGB | **FAIL** | Unchanged policy: step-1 vision RMS `0.006074178677798042` > `0.003`, maximum `0.017984390258789062`; not promoted |
| Mixed primitive qualification | **PASS** | 37/37 gates; 15 cooperative, 1 plain FP16, 0 shape fallbacks |

Receipts: `target/unified-release-precision-final/qualification.json`,
`target/unified-release-mixed-gui-clean/qualification.json`,
`target/unified-wide-multimodal-719b9c5/qualification.json`,
`target/mixed-primitive-719b9c5.json`.

Gradient accumulation, overflow skipping, scaler growth/backoff, solo optimizer
skipping and exact LoRA resume pass for the scoped fixture. A forced joint
overflow leaves image weights, language weights, AdamW moments and clocks
unchanged and backs the scale off.

## Scoped Full FP16

| Scope | Status | Result |
|---|---|---|
| Llama64 full-parameter | **PASS_SCOPED** | 480 actual FP16-accumulating dispatches (160 forward, 160 dW, 160 dX); 0 Mixed; 0 FP32 reduction fallbacks |
| Llama64 LoRA | **PASS_SCOPED** | 320 FP16-accumulating forward/dX dispatches; frozen base weights have no dW |
| Shared CLIP64 image/text | **PASS_SCOPED** | Two training steps; actual FP16 accumulation; image and text updates; zero reload divergence |
| Gradient accumulation | **PASS_SCOPED** | 960 full-training / 640 LoRA dispatches over 40 microbatches; deterministic equal-weight windows |
| Oversized reduction | **FP32 fallback by design** | Reductions greater than 64 are rejected from cooperative Full FP16 and use the explicit FP32 path; they are not reported as Full FP16 dispatches |
| Broad original primitive | **HISTORICAL / FAIL** | 9 of 37 primitive gates fail under the original cadence oracle; unchanged and preserved |
| Matched dot2 oracle, applicable short reductions | **PASS** | 28/28 applicable gates, unchanged drift limits |
| Native V4.1 Full FP16 | **REJECTED** | Not admitted |

Full-training drift versus FP32: max logit `3.662109375e-04`, RMS
`1.3759e-04`; LoRA max `1.068115234375e-04`, RMS `5.2882e-05`.
Maximum admitted reduction: 64. Full FP16 means FP16 GEMM accumulation, not
FP16 graph arithmetic; normalization, softmax, loss, masters and AdamW remain
FP32.

Receipt: `target/unified-release-precision-final/qualification.json`.
Original primitive failure: `target/fullfp16-primitive-719b9c5.json`.

## GUI and CLI

- Actual GUI Start action: eight workflows executed (Mixed and Full FP16, each
  vision, full Transformer, full-parameter fine-tune, LoRA) through
  `multimodal_ui`; both ignored Start tests pass.
- Mixed GUI reports: 84 vision, 48 full/fine-tune, 32 LoRA cooperative
  dispatches; Full FP16 reports: 84, 48, 32, with FP16 accumulation nonzero.
- `hierarchos-native-cli` release binary:
  `743feeb89204d6131bab13e64ead6394a616b4fd9006f9121cc7ff0511a1eee0`.
- CLI tests: 50 passed, 0 failed. GUI tests: 17 passed, 0 failed, 4 ignored;
  the two precision Start tests were run separately and passed.
- Full Rust library suite: 1,157 passed, 0 failed, 40 ignored, exit 0
  (`target/rust-lib-final2.log`).

## Packed cell and shader provenance

- Packed-cell composition: **PASS**. `worst_max_abs = 0.0`, qualified AMD plan,
  vendor dispatches for `linear-forward`, `linear-residual-forward` and
  `linear3-forward`, negative control rejected. Receipt:
  `target/packed-cell-719b9c5.json`.
- The 13-route AMD receipt remains **HISTORICAL ONLY** at
  `b03bf0919ea0b3bfb4fcbc6964674131bbc4464d`; the required current-source
  Phase XIX-Z regeneration is not complete.
- Shader provenance: **BLOCKED**. 674 of 727 training variants reproduce
  byte-for-byte with the installed toolchains; 53 glslangValidator artifacts
  require the pinned 15.3.0 release, while only 16.5.0/16.6.0 are installed.
  `glslc` artifacts and the other 213 glslang artifacts pass. Receipt:
  `target/training-shaders-719b9c5.json`.

## Remaining limitations

- Phase XIX-Z current-source qualification: **BLOCKED**; the old receipt fails
  closed because `rwkv_key_state_forward_fused_wide.comp` changed.
- Shader provenance: **BLOCKED** on the pinned glslang 15.3.0 toolchain.
- Full-model optimizer resume, image gradient accumulation and partial-window
  resume remain unsupported.
- The 128-wide exploratory image geometry retains its post-training strict
  difference of approximately `2.24e-7`; it is not added to the 93-case gate.
- Windows Vulkan AMD CI runner availability and GitHub Actions conclusions are
  not claimed here.
