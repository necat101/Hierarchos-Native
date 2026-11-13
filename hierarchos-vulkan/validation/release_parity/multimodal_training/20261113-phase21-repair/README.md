# Phase XXI numerical repair - 2026-11-13

Implementation revision: `91dda255d56d47fb4182a117b9c104480694a966`
Branch: `codex/rdna3-phase19-instrumented`
Device: `AMD Radeon Graphics [pci=1002:15bf]`

This record covers the Phase XXI repair only. It does not replace the
`20261112-unified-release` Phase XIX-W/X/Y/Z receipts, which remain absent.

## Numerical policy change

Gemma4 wide RGB Mixed previously failed the unchanged vision limit because
rounding Q, K and V operands to FP16 is amplified by the parameter-free V norm.
Independent eager replay isolated the contributions:

| Rounded projections | Vision max | Vision RMS |
|---|---:|---:|
| Q only | `5.34058e-05` | `1.43791e-05` |
| K only | `1.21117e-04` | `4.27206e-05` |
| Q + K | `1.36375e-04` | `4.58877e-05` |
| V only | `8.26836e-03` | `2.56412e-03` |
| Q + K + V | `8.29124e-03` | `2.56489e-03` |

The policy therefore keeps Q and K on cooperative WMMA, and keeps V, the output
projection, and the MLP projections on their FP32 kernels. Skipped projections
are counted separately as `policy_fallbacks`; they are never added to the
cooperative dispatch counters. No acceptance threshold was changed
(`RMS <= 0.003`, `max <= 0.02`, strict FP32 `< 2e-7`).

## Gemma4 wide Mixed

**PASS_SCOPED** for `gemma4_rgb8`:

- Mixed dispatches `44`; phases forward/dW/dX `[28, 8, 8]`.
- FP16-accumulating dispatches `0`; FP32 masters and optimizer state.
- Policy fallbacks `114`.
- Step-1 vision max `1.3828277587890625e-04`, RMS
  `4.6337672045397e-05`.
- Step-2 vision max `1.407623291015625e-03`, RMS
  `3.1492973139237037e-04`.
- Step-2 logits max `4.801154136657715e-05`, RMS
  `8.569628412135932e-06`.
- Reload logits and vision features differ by `0.0`.

The earlier all-Q/K/V attempt failed with step-2 vision max
`0.2756843566894531` and RMS `0.04629850713155045`; the all-FP32 attempt failed
the cooperative-dispatch qualifier with zero dispatches. Neither result is
promoted.

Receipt: `gemma4-wide.json`, SHA-256
`70d9e823d41ab7189761e946e8f800df1040dfe27c7bb315b6b2f08770a8c786`.
Runner source-file manifest is embedded in the receipt (1,549 files).
Gemma4 binary SHA-256:
`d5c76370dcd1e2cad4cbc24a15218675469a556e9d5621091373d967b3c1b957`.

## Full wide multimodal matrix

**PASS_SCOPED**, 9 of 9 cases under the current source:

`llava clip_last_layer_padded`, `gemma3 siglip_full_grid`,
`gemma4 gemma4_rgb8`, `seq2seq t5/unmasked`, `t5/padded`,
`bart/unmasked`, `bart/padded`, `switch_transformers/unmasked`,
`switch_transformers/padded`.

Receipt: `wide-multimodal.json`, SHA-256
`0346e2abf64528d7699241b588551e450339eb0358e55c303a453e85d99ccc51`.
Binary SHA-256 values are recorded in the receipt.

## Scoped Full FP16 oversized reduction

**PASS**, real GPU test
`wmma_linear::tests::full_fp16_graph_widens_accumulators_in_all_three_phases`:
1 passed, 0 failed, exit 0. The test exercises forward, dW and dX at K=64 with
FP16 accumulation, then verifies a K=128 projection is rejected into the
explicit FP32 path, increments `full_fp16_reduction_fallbacks` once, and does
not increment either cooperative precision counter.

Log: `full-fp16-oversized.log`, SHA-256
`0c5f4abcfa93d35924b6e054ab4c6f313ab238404c5a7028daa125743f5c201d`.

## Shader provenance

**PASS**, `727/727` artifacts reproduce byte-for-byte.

- Compiler: `C:\VulkanSDK\1.4.321.1\Bin\glslangValidator.exe`
- Version: `11:15.3.0`
- Compiler SHA-256:
  `64c49955b902968b5c0ab9cfb82a9bf441ae45fbce12876fcb64cac459cdf749`
- Log: `shader-provenance-15.3.log`, SHA-256
  `4d9a0e5e6bc0e3c0e9a65f884d17419e82a1644c258bb96d3dba0a4a8630d344`

Expected hashes were not edited.

## Rust library suite

**PASS**, current source: `1,157 passed; 0 failed; 40 ignored`, exit 0.

Log: `rust-lib.log`, SHA-256
`b6f8c254f207a0c69a887bb069ada4accee8b8a2ccfa3d681c44e66c8ecb764a`.

## Not run or not qualified here

| Scope | Status | Detail |
|---|---|---|
| CLI suite | **NOT RUN** | Previous count at `719b9c5` was 50 passed; not rerun after `91dda25`. |
| GUI suite | **NOT RUN** | Previous count at `719b9c5` was 17 passed, 0 failed, 4 ignored; not rerun. |
| 93 maintained FP32 image cases | **HISTORICAL ONLY** | 93/93 passed at `719b9c5`; FP32 arithmetic and its thresholds were not changed, but this root was not rerun at `91dda25`. |
| 166 FP32 text cases | **HISTORICAL ONLY** | 166/166 passed at `719b9c5`; not rerun at `91dda25`. |
| 1,116 strict intermediate diagnostics | **FAIL / UNQUALIFIED (historical)** | 194 failures at `719b9c5`, dominated by post-training feature drift. Not part of the frozen observable/update gate and not reset or retuned. |
| 128-wide exploratory image geometry | **HISTORICAL / UNQUALIFIED** | Post-training strict difference approximately `2.24e-7`; it remains outside the 93-case gate. |
| DeepSeek V4.1 stress | **HISTORICAL / UNQUALIFIED** | 32 strict failures, maximum `4.76837158203125e-06`; not promoted. |
| Phase XIX-W matrix | **BLOCKED** | Final-root `phase19w-matrix.json` was not regenerated from current-source forward/generation evidence. |
| Phase XIX-X activation | **BLOCKED** | Final-root `phase19x/current-qualification.json` and activation evidence are absent. |
| Phase XIX-Y oracle bytes | **BLOCKED** | Final-root `phase19y/current-qualification.json` is absent. |
| Phase XIX-Z | **BLOCKED** | Current-source `phase19z/current-qualification.json` remains absent; the old receipt fails closed on a changed shader. |
| GitHub Actions | **NOT RUN / BLOCKED** | No new workflow conclusion is claimed; the self-hosted Windows Vulkan AMD packed-cell runner remains unavailable. |
