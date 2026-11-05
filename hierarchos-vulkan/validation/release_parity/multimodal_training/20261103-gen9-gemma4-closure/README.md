# Gemma4 image pipeline: Gen9 closure, 2026-11-03

Both fixed Gemma4 image cases pass unchanged `max_abs < 2e-7`, `rtol=0`,
strict-less-than logit and parameter gates against unmodified upstream CPU
`Gemma4ForConditionalGeneration`, with eager attention and the frozen AVX2 policy.
Each case has 130 required checks and two full composed AdamW updates.

| Case | Input domain | Worst visible logit | Worst parameter difference |
| --- | --- | ---: | ---: |
| `gemma4_default` | Original raw patch-value stress case | 1.341104507446289e-7 | 5.681067705154419e-8 |
| `gemma4_rgb8` | Processor-compatible rescaled RGB8 | 1.341104507446289e-7 | 5.960464477539063e-8 |

Both cases change the vision tower, vision embedder and text parameters, carry
nonzero loss gradients into image patches, respond to the alternate image and
reload their native HF exports with identical logit words. Each case retains two
failing strict intermediate diagnostics. These gates qualify visible logits and
parameter updates; they do not establish parity for every intermediate value.

The original DeepSeek receipt missed step-2 logits at 2.5704503059387207e-7.
The same immutable CPU fixture now reports 1.0058283805847168e-7 there. Failed
intermediate candidates are retained in `diagnostics.zip`.

## Pixel-domain correction and scope

The inherited prototype fed raw 0..255 patch values despite its processor's
rescale configuration. The public RGB8 API now applies rescaling and returns
NCHW FP32 input, consistent with the other image pipeline APIs. The old fixture
is preserved as an explicit `raw_patch_values` stress case; its model, inputs,
CPU parameters and CPU logits were not regenerated.

The additional `rescaled_rgb8` case uses the real upstream
`Gemma4ImageProcessorPil`, without resizing or normalization. Processor padding
is explicitly compacted to the 36 valid patches before both engines execute.
The upstream processor and model arithmetic are unmodified. This does not
qualify padded image attention. The two cases use different frozen seeds.

Geometry: one fixed 12x12 RGB image, patch size 2, 6x6 patch grid, 3x3 pooling to
four soft tokens; two width-16 vision layers with four D4 heads; two width-16
decoder layers with two D8 heads, vocabulary 64, sequence length 8, untied output,
no per-layer inputs or audio, eager attention, no dropout, FP32 on Intel HD
Graphics 520. Other shapes, standardized/clipped linears, resizing, multiple
images, audio/video, optimizer resume and production throughput are unqualified.
Physical RDNA3 replay remains pending.

## Numerical evidence

Seven GPU primitive tests reproduce 8,128 frozen CPU FP32 words exactly:
contiguous RMS, axial RoPE, vision attention/context and its QKV adjoints, sparse
pooling, nine transposed weight-gradient shapes, and decoder attention/context
and QKV adjoints. The new decoder and weight-gradient variants are selected only
for the measured Gemma4 geometry. The earlier eight-lane boundary weight-gradient
experiment remains unpromoted for this qualified profile.

Two final diagnostics replay the CPU model with observational hooks, first
requiring exact reproduction of frozen CPU logits. `diagnostics.zip` binds their
native traces to the final binary and records every observed image/text boundary.
It preserves remaining intermediate differences. Primitive fixtures live in
`validation/fixtures/gemma4_cpu_primitives/` with recorded upstream provenance.

All 30 affected shader variants rebuild byte for byte and pass SPIR-V validation;
shared attention variants with the new macro disabled retain their exact bytes.
Recovered compiler recipes describe the inherited cross-entropy and vision
artifacts without replacing their bytes. `shader-manifest.json` and
`shader-verification.json` retain releases, flags and hashes.

## Regression and build results

* SigLIP/Gemma3: 4/4, 684 checks, worst logit 1.9650906324386597e-7.
* CLIP/LLaVA: 4/4, 564 checks, worst logit 7.450580596923828e-8.
* Text: 166/166, worst 1.7881393432617188e-7; all 77,772 output words equal
  the preserved previous baseline.
* Projected-feature bridge: 63/77, unchanged 14 red cases and 30 failing rows.
  `regression-audit.json` verifies the exact failure set, with no new rows.
* Rust: seven Gemma4 primitive tests, nine existing vision tests, two GPT-2
  checks and two bridge-layout checks pass; two external shader diagnostics stay
  intentionally ignored.
* Python: 53 controls pass, including nine malformed/unsupported Gemma4 native
  input cases, four Gemma3 cases and six LLaVA cases.
* Locked offline probe builds and `cargo check --all-targets` pass. Existing
  compiler warnings are retained in the logs.

Cargo's Git source scan stalled in this checkout. Builds used a byte-verified
source copy outside any Git ancestor and the original target directory. The
library-test package alone used debug=0 to reduce memory; ordinary dependencies
retained their profiles. `compiled-source-sha256.json` verifies every Rust,
shader, Cargo manifest and lockfile against that checked build copy.

## Files and replay

`qualification.json` aggregates only these two Gemma4 cases; it explicitly leaves
the full multimodal inventory and RDNA3 qualification false. Individual gate
receipts, native exports/reports, CPU archives, traces and regression receipts are
retained. `source-sha256.json` binds current source and validation code;
`receipt-sha256.json` covers all receipt files except itself. Original historical
red and green receipts remain untouched.

Extract `cpu-legacy.zip` and `cpu-rgb.zip` into a fresh directory. Their roots are
`gen4-cpu/` and `gemma4-rgb-cpu/`. From the repository root, build the Gemma4 probe
and replay with standard-library Python, without re-preparing the CPU oracle:

```powershell
cargo build --locked --manifest-path hierarchos-vulkan/Cargo.toml --bin gemma4_vision_probe
python hierarchos-vulkan/validation/qualify_gemma4_pipeline.py run --fixtures C:/fixtures/gen4-cpu --binary hierarchos-vulkan/target/debug/gemma4_vision_probe.exe --output target/gemma4-legacy-replay --target gen9
python hierarchos-vulkan/validation/qualify_gemma4_pipeline.py run --fixtures C:/fixtures/gemma4-rgb-cpu --binary hierarchos-vulkan/target/debug/gemma4_vision_probe.exe --output target/gemma4-rgb-replay --target gen9
```

For RDNA3 use `--target rdna3 --device-id` with the actual device ID and a fresh
output directory. The runners explicitly pin the frozen AVX2 width, including on
AVX-512 hosts. Archive physical RDNA3 results separately before promoting them.
