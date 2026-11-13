# Phase XXI numerical and compatibility repair - 2026-11-13

Implementation revision: `b84bc91c71dba828f60ca06d77126869c4aec184`
Branch: `codex/rdna3-phase19-instrumented`
Device: `AMD Radeon Graphics [pci=1002:15bf]`

This record covers the Phase XXI repair. Phase XIX-W and the attention half of
Phase XIX-X were regenerated; Phase XIX-X activity/drift, Phase XIX-Y, Phase
XIX-Z, and GitHub Actions remain blocked.

## Gemma4 wide Mixed

**PASS_SCOPED** for `gemma4_rgb8`. Rounding Q and K operands stays inside the
unchanged wide limits; V is the dominant error source because its parameter-free
norm amplifies FP16 query/value quantization. V, output, and MLP projections
therefore remain FP32 and are counted as policy fallbacks, not cooperative
dispatches.

| Rounded projections | Vision max | Vision RMS |
|---|---:|---:|
| Q only | `5.34058e-05` | `1.43791e-05` |
| K only | `1.21117e-04` | `4.27206e-05` |
| Q + K | `1.36375e-04` | `4.58877e-05` |
| V only | `8.26836e-03` | `2.56412e-03` |
| Q + K + V | `8.29124e-03` | `2.56489e-03` |

Result:

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
- No threshold changed: RMS `<= 0.003`, max `<= 0.02`, strict FP32 `< 2e-7`.

The rejected all-Q/K/V attempt reached step-2 vision max
`0.2756843566894531` / RMS `0.04629850713155045`; the all-FP32 attempt had zero
cooperative dispatches and failed the qualifier. Neither is promoted.

Receipt: `gemma4-wide.json`, SHA-256
`ee019aeb7a65b1687460e9e745c1c10ac9e44bafaedec55998129d7b34cc038e`.
Gemma4 binary SHA-256:
`8a77ab0350a4f0cbb97e4d03edbf4311dd1431f359612faec120b1db76cb8fcb`.

## Full wide multimodal matrix

**PASS_SCOPED**, 9 of 9 cases under the current source:

`llava clip_last_layer_padded`, `gemma3 siglip_full_grid`,
`gemma4 gemma4_rgb8`, `seq2seq t5/unmasked`, `t5/padded`,
`bart/unmasked`, `bart/padded`, `switch_transformers/unmasked`,
`switch_transformers/padded`.

Receipt: `wide-multimodal.json`, SHA-256
`af1b5b959ea7c299d1e20d029fa9cea334e15f00facfc7ccf0f521f3da24dee5`.

## Scoped Full FP16 oversized reduction

**PASS**, real GPU test
`wmma_linear::tests::full_fp16_graph_widens_accumulators_in_all_three_phases`:
1 passed, 0 failed, exit 0. Forward, dW, and dX at K=64 use FP16 accumulation;
K=128 is rejected into the explicit FP32 path, increments
`full_fp16_reduction_fallbacks` once, and leaves both cooperative precision
counters unchanged.

Log: `full-fp16-oversized.log`, SHA-256
`da8129f846c4380bc8b7aa5e6185cc92b70525185e80aea0becd2e306d45dbb5`.

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

**PASS**, current source: `1,158 passed; 0 failed; 40 ignored`, exit 0. This
includes the new packed-target geometry regression test.

Log: `rust-lib.log`, SHA-256
`7fc7d5c09052f71680981eaf1f040d8512b90943101ba4aabfb6cbe2b5569779`.

## Phase XIX-W

**PASS**, current-source compatibility matrix:

- 146 canonical types and 83 aliases.
- 166 forward replays over 43 historical green profiles, all pass.
- 72 generation requests over 18 profiles, all pass.
- 32 PEFT training profiles, all pass.
- 26 historical native architectures preserved; no regression.
- Matrix SHA-256:
  `517e4245635682272b30e1e2547a1cc7f3d0c3e2e38d4d9bb689d4fdca6641a4`
  at
  `hierarchos-vulkan/validation/release_parity/multimodal_training/20261112-unified-release/phase19w-matrix.json`.

The PEFT replay exposed an order-dependent Mixtral merge: a sorted
`target_parameters` list made `gate_up_proj` and `down_proj` resolve to the
other's tensor. The resolver now prefers the candidate whose packed geometry
uniquely accepts the adapter A/B dimensions, retaining declaration order only
when geometry is ambiguous. All 32 profiles then passed. Regression test:
`lora_packed_tests::packed_target_parameters_use_geometry_when_config_order_is_sorted`.

## Phase XIX-X

Attention half: **PASS**. Fresh two-pass attention-backward evidence and its
SPIR-V binding were installed at
`phase19x/attention/current-qualification.json`.

Activity/drift half: **BLOCKED**. The producer requires the missing
`hierarchos-vulkan/target/phase19x/drift/exports/step-0010-*` checkpoints; no
current-source receipt is claimed.

## Not run or not qualified here

| Scope | Status | Detail |
|---|---|---|
| CLI suite | **NOT RUN** | Previous count at `719b9c5` was 50 passed; not rerun at `b84bc91`. |
| GUI suite | **NOT RUN** | Previous count at `719b9c5` was 17 passed, 0 failed, 4 ignored; not rerun. |
| 93 maintained FP32 image cases | **HISTORICAL ONLY** | 93/93 passed at `719b9c5`; FP32 arithmetic was not changed, but this root was not rerun. |
| 166 FP32 text cases | **HISTORICAL ONLY** | 166/166 passed at `719b9c5`; not rerun. |
| 1,116 strict intermediate diagnostics | **FAIL / UNQUALIFIED (historical)** | 194 failures at `719b9c5`, dominated by post-training feature drift; not part of the frozen observable/update gate and not retuned. |
| 128-wide exploratory image geometry | **HISTORICAL / UNQUALIFIED** | Post-training strict difference approximately `2.24e-7`; remains outside the 93-case gate. |
| DeepSeek V4.1 stress | **HISTORICAL / UNQUALIFIED** | 32 strict failures, maximum `4.76837158203125e-06`; not promoted. |
| Phase XIX-X activity/drift | **BLOCKED** | Missing drift export checkpoints. |
| Phase XIX-Y | **BLOCKED** | The recovered shape archive (`gpt2-shapes-avx512-oracles.zip`) and a current image-text probe/fixture set are absent; the producer cannot rebind. |
| Phase XIX-Z | **BLOCKED** | Current-source packed-cell receipt and the authorized self-hosted Windows Vulkan AMD runner are absent. |
| GitHub Actions | **NOT RUN / BLOCKED** | No workflow conclusion is claimed. |
