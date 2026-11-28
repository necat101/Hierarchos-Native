# Phase XXIII — v0.2.8 release receipt

**Status: release prepared from `77b0b006760171207d6a8e3054fd2070e4bd49c0`.**
Published state and asset hashes are recorded after GitHub confirms them;
nothing in this file declares a release published before that confirmation.

## 1. Starting point and scope

- Branch: `release/v0.2.8`, cut from the committed Phase XX tree
  (`468a7ceada598dd613162323604f4240f1b411c6`) on
  `research/v0.2.8-optimization-discovery`.
- `main` and `origin/main` at the start: `9b0091e` (`v0.2.7` merge).
- Committed before this release: Phase XVIII dQ accumulator and Phase XX query
  scratch reuse. Neither was requalified; their receipts are unchanged.
- One bounded experiment, Phase XXII, was run and passed. No second candidate
  and no new profiling campaign were started.

## 2. Phase XXI closure

`validation/phase_xxi_postpromotion_profile_20261128/` is committed with its raw
logs. The amendment records that the attribution tables average **29 profiled
batches** (3 warmup + 26 measured), not 26; the 26-measured values are dense
13.034, `linear_forward` 6.751, `linear_input_grad` 5.948, query 2.894 and total
24.471 ms. Three independent profiled processes per mode and one unprofiled
process per mode were run. The profile is component attribution only; it is not
a performance campaign and no ratio is derived from it.

## 3. Phase XXII decision

GO, promoted. The candidate is a 32 x 32 output tile for FP32
`linear_input_grad`, eligible when rows, input dimension and output dimension are
all multiples of 32. It preserves each output's ascending-`k` single-`fma`
reduction and FP32 accumulation. `linear_input_grad_output_tile32.spv` is
4,992 bytes, SHA-256
`9316501e88a97a41da7a30c21e68eb31e80fd328cc6c8626790a7c8bbbb12c45`; its GLSL
source is `fb94829731afe81a1d77f8b62f9c937b5f767059b38e6cff3c3d900bed7c983d`.

- Parity: 272,038/272,038 output words bitwise equal over six shapes, zero
  non-finite values, zero unwritten elements.
- Complete-step ABBA: 24.875 → 21.716 ms, median ratio 0.872420, 95 % CI
  0.870612–0.874241, 5/5 rounds, **1.1462×**; identical 29-step
  `parameter_end_digest = d79a12d8…9c4655`.
- Control: `HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0` selects the original
  16 x 16 kernel for every shape. A failed candidate load also falls back.
- Full receipt: `validation/phase_xxii_linear_input_grad_tile_20261128/`.

## 4. v0.2.8 contents

1. Phase XVIII literal-index dQ accumulator, default on its existing
   head-dimension-128 path; original query shader remains the fallback.
2. Phase XX scratch reuse, default on the same path; Phase XVIII remains the
   scratch-reuse opt-out and the original query shader the overall fallback.
3. Phase XXII output-tiled FP32 input gradient, default for 32-aligned shapes;
   original kernel remains the fallback and opt-out.

Mixed WMMA is still default-off. No threshold, allowlist or existing eligibility
gate moved. The Phase XIX five-round, 160-process campaign remains the support
for the scratch-reuse claim.

## 5. Shader and layout verification

```text
python -m json.tool validation/training_shader_manifest.json   # OK
glslc -fshader-stage=compute --target-env=vulkan1.1 -O \
  shaders/linear_input_grad_output_tile32.comp -o /tmp/rebuild_tile32.spv
cmp /tmp/rebuild_tile32.spv shaders/linear_input_grad_output_tile32.spv  # identical
spirv-val --target-env vulkan1.1 shaders/linear_input_grad_output_tile32.spv  # OK
python validation/check_push_constant_layouts.py
# checked 534 shader push blocks, 112 host structs
# PASS: push-constant blocks agree across SPIR-V, GLSL and host structs
```

The v0.2.7 attention artifacts are unchanged; their byte-identity is pinned by
the release manifest rather than re-derived here.

## 6. Build and package

- Tests: `cargo test --release --lib --no-run` EXIT=0; bitwise parity and selector
  tests EXIT=0. Production compile: `cargo check --lib --bins` EXIT=0.
- Build: `package-standalone.ps1 -OutDir <stage> -StaticCrt`, default target
  `x86_64-pc-windows-msvc`, release profile, static MSVC CRT, dependency scan
  enabled → `PASS`.
- Source state at packaging: `clean` (`77b0b006`), build timestamp
  `2026-11-28T10:46:31Z`.
- Archive: `Hierarchos-Native-Windows-x64-Portable-v0.2.8.zip`, 34,294,807
  bytes, 16 entries, SHA-256
  `aa2d8d7feda9b106120d9abb4b0be28986d554ad6d915af000a0be89db0ec61c`.
- `BUILD_MANIFEST.json`: 11,911 bytes, SHA-256
  `c0af79f5156a1840e3328e63c769275387a7a8bb80ac51585d76547ad095702d`;
  `SHA256SUMS` verifies both.
- Sidecars committed under `phase_lxxxiv_evidence/release_assets_v028/`; the
  archive is not committed.

## 7. Extracted-package acceptance

The archive was extracted to a path containing spaces. `sha256sum -c
SHA256SUMS` passed for both files; `doctor`, `devices`, `precision-info`,
`architectures` and `--help` all exit 0; the dependency scan is `PASS`.

| Arm | Route evidence | Loss |
| --- | --- | --- |
| default | `linear_input_grad_output_tile32` ×15, `…query…scratchreuse` ×2, `…key_channel4` ×2 | 6.937828063964844 |
| `…OUTPUT_TILE=0` | `linear_input_grad` ×15 | 6.937828063964844 |
| `…SCRATCH_REUSE=0` | `…query…dqunroll` ×2 | 6.937828063964844 |
| `…QUERY_DQUNROLL=0` | `transformer_attention_backward_query_rdna3` ×2 | 6.937828063964844 |
| `…TWO_PASS=0` | scalar backward ×2, no scratch | 6.937828063964844 |
| `…KEY_CHANNEL4=0` | `…backward_key_rdna3` ×2 | 6.937828063964844 |

Full transcript: `raw/02_package_acceptance.txt`. The promoted window kernel is
bitwise-equal by the Phase XXII matrix; the package arms confirm route identity
and identical one-step loss only.

## 8. CI and publication

Recorded after the required checks pass and GitHub confirms the tag, release and
assets. The release is not published until that record replaces this paragraph.
