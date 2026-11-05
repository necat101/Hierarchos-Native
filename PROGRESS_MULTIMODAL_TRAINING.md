# Multimodal training progress

## Current phase: Gen9 upload optimization qualified, 44/44 preserved

Bounded parameter-upload batches and shared host staging improve native V4.1
training by **1.052x / 1.067x** at the frozen 4x4 / 6x4 fixture geometries on
Intel HD Graphics 520. Same-executable ABBA measurements use two warmup plus
seven timed updates in each of four processes; trained logit/image/adapter words
are identical after all nine updates. Timed submissions fall **1,102 -> 982**
per native step (**120 fewer, 10.89%**). Shared LLaMA/GPT-2/T5 are unchanged
controls; no performance gain is claimed for those paths.

The final source again passes **all 44 green text profiles**, **95 image cases**
and **166 text cases**, with the unchanged **`max_abs < 2e-7`, `rtol=0`** gates.
The 30-execution native V4.1 qualifier retains all 1,360 logits word-exact,
checkpoint/resume and forced-host-width coverage. The broad DeepSeek suite
verifies 215 tests, with 26 opt-in tests ignored; one trace-writing test needed
its missing output directory created and passed in a targeted rerun. Both the
original failure log and recovery are retained. All 672 shader variants rebuild
and validate; the 81 historical V4.1 SPVs and 127 CPU fixtures remain unchanged.

This change groups component admissions with a 16 MiB payload budget, using the
existing 4 MiB host staging chunks and individual GPU buffers. Large tensors
keep their old boundaries. No arithmetic or shaders changed in this pass.
Timings use an unoptimized libtest build with debug information disabled;
production-size throughput is unqualified. **Physical RDNA3 replay is pending;
the RDNA3 machine was not accessed.**

- [Current qualification](hierarchos-vulkan/validation/release_parity/multimodal_training/20261105-gen9-v41-upload-batching/qualification.json)
- [Measurements, recovery evidence and replay](hierarchos-vulkan/validation/release_parity/multimodal_training/20261105-gen9-v41-upload-batching/README.md)
- [44-profile inventory](hierarchos-vulkan/validation/multimodal_training_inventory.json)

The new November 5 source handoff includes untracked implementation files;
the unchanged November 4 oracle archive is reused. Earlier evidence and archives,
including the synchronized RDNA3 GPT-2 fix/README, remain preserved. No commit
or push was made.


## Previous phase: Gen9 full image training gate closed, 44/44

All **44 previously green text profiles** now have qualified joint image training
at the frozen Gen9 fixture geometry. The unchanged visible-logit and parameter
gates are **`max_abs < 2e-7`, `rtol=0`**. No CPU expectation was regenerated to
obtain these passes. The RDNA3 GPT-2 fix and README already integrated from
`d41ebeee295d598840438ae03757419d624985ac` remain preserved.

Final-source build19 results on Intel HD Graphics 520:

| Gate | Passing cases | Worst visible logit |
| --- | ---: | ---: |
| Shared trainable CLIP tower/projector, 40 text profiles | 77 | `1.7136335372924805e-7` |
| Joint BART/T5/Switch encoder-decoder image training, 3 profiles | 6 | `5.960464477539063e-8` |
| Dedicated DeepSeek V4.1 native image training, final profile | 2 | **`0`** |
| Separate native Gemma4, Gemma3/SigLIP and CLIP/LLaVA regression | 10 | `1.9650906324386597e-7` |
| Resident/streaming text regression | 166 | `1.7881393432617188e-7` |

Every image case includes two joint AdamW updates, image sensitivity, gradient
flow into the trainable tower/projector and native export/reload checks. V4.1
trains all native ViT, aligner and delimiter parameters alongside its established
frozen-language-base LoRA contract. Its 1,866 comparisons cover 155,444 values;
all 1,360 logits are word-exact and the worst other difference is
`1.4901161193847656e-8`. Its 30 final test executions also cover lossless optimizer
resume, continued updates, corrupt/foreign checkpoint rejection, stale-gradient
guards, preserved text behavior and forced host vector width 16.

The final V4.1 drift was the accumulation of trained base/adapter input-gradient
edges. Native experts and attention now preserve released autograd order,
including compressor contributions, using existing GPU buffers. Private AVX2
selection pins native image math across host CPU vector widths. The original
text path retains its dispatch and memory behavior. All 81 historical V4.1
shader artifacts and 127 historical CPU fixtures remain byte-identical;
all 671 maintained shader recipes rebuild and validate.

Maintained receipt, per-profile inventory, reproduction and handoff instructions:

- [Full qualification](hierarchos-vulkan/validation/release_parity/multimodal_training/20261104-gen9-full-image-training/qualification.json)
- [Replay and scope](hierarchos-vulkan/validation/release_parity/multimodal_training/20261104-gen9-full-image-training/README.md)
- [44-profile inventory](hierarchos-vulkan/validation/multimodal_training_inventory.json)

**Physical RDNA3 replay is pending.** The RDNA3 machine was not accessed.
Qualification covers the frozen FP32 geometries; it does not establish
production-size parity or throughput, BF16 training, audio/video, or equivalence
to separately named upstream VL models when the qualified text backbone uses
the authorized shared trainable tower. Existing native towers remain available
and their regression cases pass. V4.1 accepts one complete image span per prefill
and a fixed patch grid per trainer, with released processor patches promoted to
FP32 at the native qualification boundary.

The local branch is `codex/gen9-multimodal-training`; no commit or push was made.
The chronological work log remains in
`target/multimodal-gen9/shared-recovery-20261104-journal.md`.

## Previous phase: Gemma 4 vision logit gate closed on Gen9

The native Gemma4 image pipeline now passes the unchanged strict
`max_abs < 2e-7`, `rtol=0` logit and parameter-update gates against the pinned,
unmodified upstream `Gemma4ForConditionalGeneration` CPU implementation.
Both frozen cases run the axial-RoPE vision tower, Q/K/V normalization,
sandwich RMS blocks, 3x3 pooler, vision embedder and Gemma4 decoder, with full
backward and two complete AdamW updates.

| Frozen case | Pixel domain | Worst visible logit | Worst parameter difference |
| --- | --- | ---: | ---: |
| `gemma4_default` | Original raw model-value stress case | `1.341104507446289e-7` | `5.681067705154419e-8` |
| `gemma4_rgb8` | Real RGB8 processor rescaling | `1.341104507446289e-7` | `5.960464477539063e-8` |

The original failing `step-2/logits` row was `2.5704503059387207e-7`; it is now
`1.0058283805847168e-7`. Each case passes 130 required checks, including all
parameter tensors after each update, changed tower/embedder/text parameter
groups, gradient flow into patches, alternate-image sensitivity and bit-exact
native export/reload. Two strict intermediate diagnostic rows remain above
`2e-7` in each case and are retained in the receipts.

The inherited RGB wrapper omitted rescaling. `preprocess_rgb8` now applies the
processor's rescale factor and returns NCHW FP32 values for the public pipeline.
The original CPU fixture is immutable and explicitly replays as
`raw_patch_values`; the additional `rescaled_rgb8` fixture uses the actual
upstream `Gemma4ImageProcessorPil`. Its padding is explicitly compacted to the
36 valid patches before both engines run. This does not qualify padded image
attention or resizing.

The numerical repairs follow independently measured CPU operation order:
contiguous RMS reductions, GPU SLEEF axial-RoPE tables, matched vision and text
eager attention/adjoints, sparse pool GEMM rounding and transposed weight-gradient
panels. Seven primitive tests reproduce **8,128 frozen CPU FP32 words exactly**.
The decoder and weight-gradient variants are enabled only for the measured
Gemma4 geometry; ordinary text, CLIP and Gemma3 graphs keep their existing routes.
All 30 affected shader variants rebuild byte for byte and pass SPIR-V validation.

Final-source regression replays retain:

* **Gemma3/SigLIP: 4/4**, worst visible logit `1.9650906324386597e-7`.
* **CLIP/LLaVA: 4/4**, worst visible logit `7.450580596923828e-8`.
* **Text: 166/166**, worst `1.7881393432617188e-7`; all 77,772 output words
  equal the previous baseline.
* **Projected-feature bridge: 63/77**, with the same 14 red cases and 30 failing
  metric rows; no new failure rows.

Receipt, CPU archives, source/binary hashes, diagnostics and regression evidence:
[`20261103-gen9-gemma4-closure/`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261103-gen9-gemma4-closure/README.md).

Qualification covers fixed 12x12 RGB, one image, 2x2 patches, a 6x6 patch grid,
four pooled tokens, two width-16 vision layers and two width-16 decoder layers
at sequence length 8, FP32, AVX2 CPU policy and zero dropout on Intel HD Graphics
520. Resize, standardized/clipped vision, multiple images, broader geometries,
audio/video, optimizer resume and production throughput remain unqualified.
Physical RDNA3 replay is still pending.

Next native image families remain Qwen2.5-VL, Pixtral/Mistral3, Phi4 multimodal,
Kimi and MiniMax M3VL. Registry aliases do not add qualified architectures;
SmolVLM's Llama decoder does not qualify SmolLM3. BART/T5/Switch and the dedicated
DeepSeek V4.1 graph still need their embedding bridges. The full multimodal
architecture inventory is not yet qualified.

## Previous phase: Gemma 3 vision closed

The second real image stack is wired end to end on Gen9: raw RGB8 -> SigLIP
patch projection -> two SigLIP encoder layers -> Gemma3 average-pool
soft-embedding projector -> the real Gemma3 text backbone -> logits/loss, with
full backward through the tower and projector and two composed AdamW updates.
Measured against the pinned, unmodified upstream `Gemma3ForConditionalGeneration`
CPU oracle (pinned eager attention), **all 4 frozen cases pass** at
`max_abs < 2e-7`, `rtol=0`.

| Frozen case | Geometry | Worst visible logit | Result |
| --- | --- | ---: | --- |
| `siglip_default` | width 16, patch 2, 4 tokens | `7.451e-8` | **pass** |
| `siglip_batch_padded` | two images, left/right padding | `1.639e-7` | **pass** |
| `siglip_full_grid` | 16 tokens, kernel 1 | `1.965e-7` | **pass** |
| `siglip_wider` | width 32, patch 4, 4 tokens | `1.788e-7` | **pass** |

Each case carries 171 required checks (visible logits on the alternate image,
two training steps and the final forward, every parameter tensor after both
updates, group-update assertions, loss, pixel gradient, byte-exact
export/reload). The policy is unchanged: `max_abs < 2e-7`, `rtol=0`,
`strict_less_than`, fixtures hash-bound and not regenerated.

### Two kernel defects closed

* **Stale precompiled attention module.** The committed
  `transformer_attention_forward_cpu_reciprocal.spv` no longer matched
  `transformer_attention_forward.comp`. Recompiling it with
  `-DHIERARCHOS_CPU_ATTENTION_RECIPROCAL` made the SigLIP attention context and
  `out_proj` bit-exact (`2.384e-7` -> `0.0`). An independent replay of the CPU
  eager path reproduces both `torch.softmax` and the oracle context exactly, so
  the shader now matches the reference arithmetic.
* **An RDNA3-only GELU module was running on Gen9.**
  `transformer_gelu_tanh_ha_{forward,backward}.comp` forced
  `#define fp32_fma_exact fma`; Gen9's MAD truncates the product, so the
  compensated SLEEF tanh staging drifted one ulp at `gte`. Both modules now use
  the software exact FMA. With that, every tower boundary of `siglip_wider` and
  `siglip_default` is `0.0` and the wider case clears the gate.

### M=32 tail panel closed

With two images the first projection runs at `M=32, K=16, N=16`. There
`layer_0/q`, `k` and `v` differed by `2.384e-7` while `ln1` was exact, so the
seed was that projection. `F.linear`/`addmm` was not an ascending-K chain: ATen
AVX2 SGEMM uses a six-row panel, and for a one/two-row tail it materializes each
product in two interleaved accumulators; a three-row tail uses two FMA lanes.
`vision_linear_cpu_avx2.comp` now reproduces that short-panel rule, gated to the
measured `rows 2..=36`, `K in {8,16,32}`, `N in {16,32,64}` shapes. The patch
projection separately selects ATen bias-last or oneDNN bias-initial order from
the measured geometry. With the M=32 projection on the tail kernel,
`step-2/logits` moves from `2.757e-7` to `1.639e-7` and the case passes. The
strict post-update `vision_features` diagnostics still reach `1.31e-6`; they
remain diagnostics, not gate rows, and the projected features and every visible
logit stay below `2e-7`.

Verification for this receipt: all four frozen cases pass with 684 required
checks, including every parameter tensor after both updates and byte-exact
export/reload. CLIP/LLaVA was rerun on the same shader set and still passes all
four cases (564 required checks, worst `7.450580596923828e-8`); the text gate
was not rerun in this session.

Receipt: [`release_parity/multimodal_training/20261103-gen9-gemma3-siglip-tail/`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261103-gen9-gemma3-siglip-tail/).

LLaVA regression receipt: [`release_parity/multimodal_training/20261103-gen9-clip-llava-tail/`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261103-gen9-clip-llava-tail/).

## Previous phase: working CLIP/LLaVA image pipeline

On 2026-11-02 the native **RGB8 -> normalization -> CLIP -> LLaVA projector ->
Llama -> logits/loss** pipeline passes all four independent CPU-reference
cases on Intel HD Graphics 520. Each case includes two complete native AdamW
updates, real image-tower/projector/backbone parameter changes, gradients into
image pixels, alternate-image controls and byte-exact native logit replay after
HF package export/reload.

**22,784 visible logits pass `max_abs < 2e-7`, `rtol=0`; worst drift is
`7.450580596923828e-8`.** All 130,688 named parameter comparisons pass, worst
`2.9802322387695312e-8`. The independent CPU implementation is the unmodified
upstream `LlavaForConditionalGeneration`, including its actual CLIP tower and
processor; the oracle observes activations without replacing model math.

| Frozen case | Images | CLIP width | Patch size / tokens per image | Feature selection | Worst logit drift |
| --- | ---: | ---: | --- | --- | ---: |
| `clip_default_layer` | 1 | 16 | 2 / 16 | default penultimate layer, drop CLS | `7.450580596923828e-8` |
| `clip_last_layer_padded` | 2 | 16 | 2 / 16 | last layer, drop CLS, left/right padding | `7.450580596923828e-8` |
| `clip_full_tokens` | 1 | 16 | 2 / 17 | last layer, preserve CLS | `6.705522537231445e-8` |
| `clip_wider_grid` | 1 | 32 | 4 / 4 | last layer, tanh GELU | `7.450580596923828e-8` |

These are fixed-resolution 8x8 RGB, two-layer CLIP and two-layer/width-16 Llama
scalable-checkpoint correctness gates. Inverting only the image changes visible
text logits by **0.0465–0.1481**. Image-tower, projector and text parameters each
change by approximately `0.002`; this is full composed training, not frozen
features or a text-only checkpoint preservation check. RGB normalization matches
the CPU processor exactly. All 5,952 native export/reload logit words match.

The gate is specifically logit parity plus parameter-update parity. Intermediate
activations and raw feature adjoints retain separately reported strict `2e-7`
diagnostics; some exceed that ceiling. They are not promoted to bit parity or
hidden by the logit qualification. The older feature-bridge gate below continues
to require its stricter intermediate checks.

### Numerical repairs and broader matrix

The initial actual-image trial passed inference but three cases exceeded the
logit ceiling after the second update. Two implementation differences were
identified and repaired without changing any frozen inputs or expectations:

* Modality training now uses the already validated materialized CPU log-softmax
  adjoint, preserving rounding before embedding/tower backward.
* `CpuAdamWHyperParams` retains declared f64 scalar coefficients, and a Vulkan
  single-tensor AdamW kernel reproduces CPU lerp, moment, sqrt/division and
  update staging. Parameter and moment math stays on the GPU. Identical-gradient
  fixtures cover vector blocks, scalar tails and three beta configurations over
  three steps: **all 1,044 parameter/moment words are bit-exact**.

The same optimizer policy improves the original frozen feature matrix from
**44/77 to 63/77 passing cases**, with **30/40 profiles** passing every attempted
case. Four wide GPT-2/Llama cases still fail logits; ten further cases fail only
the stricter raw feature-adjoint rows. The 14 cases remain red in their receipt.
BART/T5/Switch and the dedicated V4.1 graph still need bridge integration; other
actual modality stacks remain pending.

### Full-matrix continuation and the Gemma4 pow repair

A second complete 77-case replay with the two-accumulator projector and the
pinned AVX2 reference width finished on Gen9. It still reports **63/77 cases**
and **30/40 fully passing profiles**, but the failure surface shrank from 33 to
**30 failing metric rows**:

| Metric kind | Before | After |
| --- | ---: | ---: |
| `modality_embedding_grad` | 18 | 18 |
| logits (all three points) | 15 | 12 |

**Gemma4's pow-RMS derivative was modelling the wrong reference.** Measured on
the pinned oracle over 2000 sampled FP32 means, `torch.pow(mean_squared, -1.5)`
agrees with the correctly rounded real value only **1969/2000** times, while
`torch.pow(mean_squared, -0.5)` equals `1/sqrt(mean_squared)` **2000/2000** times.
The scalar CPU pow therefore follows the vectorized u10 approximation, and the
correctly rounded `-3/2` value the selector previously fed into Gemma4's RMS
input gradient was the wrong model of the reference. Both backward entry points
now select the SLEEF paraphrase; the correctly rounded path remains in the
shader for fixtures that provably reach a scalar pow.

Effect, recorded as a trade rather than a pass: Gemma4's post-update and final
logits move from `2.51e-7`-`3.13e-7` to passing, and its first-step feature
adjoint improves from `3.87e-7` to `2.38e-7`. Its second-step adjoint moves from
`4.77e-7` to `8.64e-7`, so the case stays red. The primary logit and parameter
gates improved; the raw adjoint diagnostic did not improve uniformly.

| Profile | Remaining red metric rows |
| --- | --- |
| `gpt2_wide`, `llama_wide` (4 cases) | 12 logit rows, plus 4 adjoint rows on the padded/wide variants |
| `gemma4` (2 cases) | 4 adjoint rows |
| `gemma3_softcap_tied` (2 cases) | 3 adjoint rows |
| `minimax_m2/padded` | 2 adjoint rows |
| `qwen2_5_sliding_tied`, `mistral`, `kimi_k25_text`, `qwen3_5_linear`, `qwen3_5_mixed` (padded) | 1 adjoint row each, all at `2.384185791015625e-7` |

### Where the remaining drift lives

`diagnose_multimodal_arithmetic.py` replays the frozen oracle through the
unmodified Transformers model with the fixture's own embedding hook, then swaps
in native normalization arithmetic. The unpatched replay reproduces `cpu.json`
**bit-exactly** (`0.0`), so any remaining difference is native drift rather than
a harness artifact:

| `gpt2_wide/unmasked` | vs frozen CPU oracle | vs native report |
| --- | ---: | ---: |
| Unpatched | `0.0` | `2.086e-7` |
| Native LayerNorm arithmetic substituted | `2.384e-7` | `1.788e-7` |

The existing text gate already reports `1.490e-7`-`1.788e-7` for these same
64-wide packages in both resident and streaming modes. Substituting three
embedding rows with projected features moves a handful of logits from that noise
floor to `2.086e-7`. The wide failures are therefore the text path's own
forward reordering margin, not a bridge defect: closing them needs a
bit-exact wide reduction/GEMM order, not another bridge change.

The adjoint rows are a different shape of problem. Nearly every element of a
failing adjoint differs again (37-46 of 48 values), typically by 2-5 ULP of its
own magnitude, with a few small-magnitude elements at higher ULP counts. That is
whole-backward accumulation order, not a single primitive: parameter updates
pass their own `2e-7` gate at ~1 ULP, and the next step's adjoints amplify that
seed at these magnitudes. Five of the ten adjoint-only cases miss by a single
row of exactly 4 ULP at `|v| ~= 0.5-0.7`.

Receipts, the run log and a per-row before/after comparison live in
[`release_parity/multimodal_training/20261102-gen9-matrix-continuation/`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261102-gen9-matrix-continuation/).
The receipt's `binary_sha256` is the probe as it ran, before the final
comment-only cleanup; rebuilding the committed source reproduces the recorded
`gemma4/unmasked` report byte for byte, and only the debug line tables differ.

The GPT-2 tracing pass suggested those kernels were a lead for the wide pairs.
Both a full three-kernel promotion and a two-kernel subset were then built and
run against the complete 166-run text gate and the 77-case matrix; the result is
in
[GPT-2 CPU-kernel promotion](#gpt-2-cpu-kernel-promotion),
and the subset that clears both gates is what now ships.

### GPT-2 CPU-kernel promotion

Three kernels model the CPU reference for GPT-2: the fused-Welford LayerNorm,
the correctly rounded reciprocal inside attention, and MKL VML tanh inside
`gelu_new`. Making all three the architecture default was measured on Gen9
across four text cases, the multimodal wide cases and the frozen
`gpt2_rdna3_parity` regression. **No combination is a uniform improvement**, and
on the full gates the three-kernel set regresses: the 166-run text regression
drops to **164/166** because `gpt2_wide/padded` moves from `1.490e-7` to
`2.086e-7`, and the frozen Rust regression fails on the same case. That build
was rejected; its patch and logs are kept as `all-three-kernels.patch` in the
receipt below.

**The Welford norm and the attention reciprocal now ship as the GPT-2
default** (`gpt2_welford_norm`, `gpt2_attention_reciprocal`), verified on the
same two gates: text **166/166** with `gpt2_wide` at `1.788e-7` in both modes,
the frozen Rust regression passing, LLaVA **4/4**, and the matrix at **63/77
cases with 29 failing rows** (from 30). The tanh GELU and the RDNA3 k-block GEMM
stay on their adapter gate, because the GELU is what breaks the padded text
case.

That matrix row count hides a trade worth recording: unmasked `step-2/logits`
(`2.086e-7` to `1.788e-7`) and padded `step-1/logits` (`2.086e-7` to `1.639e-7`)
clear, while padded `step-2/modality_embedding_grad` rises from `1.974e-7` to
`2.198e-7`, crossing the line inside a case that already fails on logits. Two
rows gained, one lost.

Receipt: [`release_parity/multimodal_training/20261102-gen9-gpt2-promotion/`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261102-gen9-gpt2-promotion/).

Text columns are the qualification harness's bit-word comparison; modality
columns use the training runner's own float comparison. **Bold** values clear
`< 2e-7`.

| Selection | `gpt2/unm` | `gpt2/pad` | `gpt2_wide/unm` | `gpt2_wide/pad` | modality `wide/unm` | modality `wide/pad` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous default | `2.98e-8` | `2.98e-8` | **`1.79e-7`** | **`1.49e-7`** | `2.09e-7` | `2.09e-7` |
| Welford | `2.24e-8` | `2.98e-8` | `2.09e-7` | **`1.49e-7`** | `2.09e-7` | `2.09e-7` |
| Welford + reciprocal | `2.98e-8` | `2.98e-8` | **`1.79e-7`** | **`1.79e-7`** | `2.09e-7` | **`1.64e-7`** |
| Welford + tanh | `2.98e-8` | `2.98e-8` | **`1.79e-7`** | **`1.79e-7`** | `2.38e-7` | **`1.79e-7`** |
| Welford + reciprocal + tanh | `4.47e-8` | `4.47e-8` | **`1.94e-7`** | `2.09e-7` | **`1.79e-7`** | **`1.79e-7`** |
| tanh | `2.98e-8` | `2.98e-8` | **`1.97e-7`** | **`1.79e-7`** | `2.09e-7` | `2.09e-7` |
| reciprocal | `2.98e-8` | `2.98e-8` | **`1.79e-7`** | **`1.57e-7`** | **`1.79e-7`** | `2.09e-7` |
| reciprocal + tanh | `2.98e-8` | `2.98e-8` | **`1.79e-7`** | **`1.79e-7`** | `2.09e-7` | `2.09e-7` |

`welford + reciprocal` (now shipped) clears both wide text cases at
`1.79e-7`. Every configuration that also clears the multimodal wide columns
pushes a text gate over: `welford + reciprocal + tanh` reaches `1.79e-7` on
both modality columns but moves `gpt2_wide/padded` to `2.09e-7`, which is
exactly the case the repository's frozen
`gpt2_wide_frozen_cpu_logits_resident_and_streamed` regression asserts is
below `2e-7`. Its model file is byte-identical to the text `gpt2_wide` fixture
(`29f3a1856af812713715...`), so the two columns are the same gate.

Every row sits at the noise floor: the same kernel is a few ULP better on one
input set and worse on another, and the tanh GELU swings the padded text case
by 6 ULP while leaving the unmasked one alone. The shipped subset is chosen
because it clears both full gates, not because it is mathematically closer to
the reference. What the wide pairs still need is a reduction and GEMM order
that is uniformly closer, since `gpt2_wide` still carries six over-limit
multimodal rows and its `final_logits` stays at `2.384e-7`.

### RDNA3 update integrated

At the user's request, this branch was fast-forwarded to `origin/main`
**`d41ebeee295d598840438ae03757419d624985ac`**, including GPT-2 fix **`88f60d6`**,
the RDNA3 regression records and the restyled README. Tonight's work was backed
up, restored, and merged with the new README and CI controls. The GPT-2 repair's
device routing remains intact. Its two new Rust regression tests pass on Gen9.

After integration, all **166/166** existing text forward runs pass at worst
`1.7881393432617188e-7`. Every **77,772** output word matches the opening-phase
executable. The legacy streamed-vocabulary LoRA save/resume regression and the
GPU bridge tests pass. All five new shaders rebuild identically with pinned
glslang 16.6.0 and pass SPIR-V validation.

### Qualified entry points and continuation

`VulkanLlava::from_hf_package`, `preprocess_rgb8`, `forward`,
`train_step_with_optimizer` and `export_hf_package` provide the native image
pipeline. `llava_pipeline_probe` exercises those same APIs. It loads the actual
standard HF vision/projector tensors; model math has no Python dependency.

The current processor contract takes RGB images already at the configured
resolution with resize/crop disabled in `preprocessor_config.json`. Dynamic
resize/crop/interpolation, video/audio, multiple selected vision layers, dropout,
PEFT and composed optimizer checkpoint/resume remain outside this gate. The
qualified application surface is the native API/probe; dataset-driven image
training in the main CLI/GUI is further integration work. Full production-size
model throughput/capacity and physical RDNA3 replay are unmeasured in this phase.

Frozen CPU fixtures, native exports, logs, pre-repair failures, optimizer replay,
and the post-sync source snapshot are retained under
`hierarchos-vulkan/validation/release_parity/multimodal_training/20261102-gen9-image-pipeline/`.
Two independent preparations reproduce all **32** image case files exactly.
Replay needs standard-library Python; preparation additionally uses the pinned
Torch/Transformers environment and Pillow `12.0.0`. Replay pins the original
ATen vector width so RDNA3 consumes the same CPU reference arithmetic.

```powershell
cargo build --locked --manifest-path hierarchos-vulkan/Cargo.toml -j1 --bin llava_pipeline_probe --bin cpu_adamw_probe
python -m zipfile -e hierarchos-vulkan/validation/release_parity/multimodal_training/20261102-gen9-image-pipeline/cpu-fixtures.zip target/llava-repeat/cpu
python hierarchos-vulkan/validation/qualify_llava_pipeline.py run --fixtures target/llava-repeat/cpu --binary hierarchos-vulkan/target/debug/llava_pipeline_probe.exe --output target/llava-repeat/gen9 --target gen9
```

On RDNA3 use a new output directory, `--target rdna3 --device-id 0x...` with the
physical adapter's PCI ID. Keep the frozen CPU expectations unchanged. The local
`.tools/gen9-multimodal-image-pipeline-20261102.zip` contains the source/evidence
handoff against base `d41ebee` and an application helper.

## Historical opening-phase record

Started 2026-11-02 on the Intel HD Graphics 520 / i5-6200U, branch
`codex/gen9-multimodal-training`. RDNA3 execution is deferred to the user's
later device session. Earlier text parity receipts remain unchanged.

**Current result: the new strict training matrix is red.** The shared bridge
and projector execute 77 cases across 40 profiles: **44 pass, 33 fail**.
Seventeen profiles pass every case attempted for that profile. Actual modality
towers remain unimplemented/unqualified. The existing text forward matrix
still passes **166/166** at worst `1.7881393432617188e-7`.

The [retained receipts and frozen CPU archives](hierarchos-vulkan/validation/release_parity/multimodal_training/20261102-gen9-opening/)
contain passing and failing results. These results do not promote the complete
multimodal architecture inventory to supported production training.

## Scope and acceptance

The [inventory](hierarchos-vulkan/validation/multimodal_training_inventory.json)
tracks all 43 profiles in the recorded Gen9 inference matrix and the separately
qualified DeepSeek V4.1 scalable graph: 44 profiles / 27 native identifiers.
Registry recognition and composite-package aliases do not add numerically
qualified architectures. Text parity alone does not qualify image/audio/video
preprocessing, towers, projectors, modality positions, masks or training.

Every new logit comparison must satisfy **max absolute error < 2e-7, rtol=0**.
The initial feature bridge also applies this strict ceiling to projected
features, feature adjoints, two native AdamW projector updates and every named
backbone parameter after each step. Loss retains the existing full-parameter
training harness's separate `2e-5` absolute tolerance. These are deterministic
FP32 reduced-checkpoint checks with no dropout and zero weight decay.

## Implementation

* `ModalityEmbeddingRows` replaces explicitly indexed flattened batch/sequence
  rows immediately after raw token lookup, before architecture scaling,
  positions, normalization and dropout. Row order is preserved. Invalid,
  duplicate, empty, incorrectly sized or non-finite inputs fail closed.
* `forward_logits_with_modality_embeddings` and
  `train_step_with_modality_embeddings` reuse the native Transformer graph.
  Input IDs are valid vocabulary placeholders; ID-dependent branches retain
  their original semantics. This ABI does not synthesize modality position IDs,
  image-block attention, audio lengths or architecture-specific insertion rules.
* Backward returns feature adjoints and removes the replaced rows from the
  token-lookup gradient. Ordinary text rows still accumulate, and tied LM heads
  retain their independent contribution. GPU buffer copies perform the scatter
  and adjoint split; no new shader or CPU model math is introduced.
* `LinearProjectionTrainer::forward` permits composition of the existing native
  affine forward/backward/AdamW primitive with a language-model loss. The
  experimental `multimodal_training_probe` runs two complete composed steps and
  exports every backbone parameter after each step.
* CPU preparation and native execution are separate. Frozen packages,
  expectations and generator sources are hash-bound. The native runner uses
  only standard-library Python, so RDNA3 replay needs no Torch installation.
  A partial or failed run cannot publish a completed passing receipt.

The initial candidate batch is Llama, Gemma3, Phi-4 Multimodal's text backbone
and Qwen3.5 hybrid, each unmasked and padded, with two batch elements and three
modality rows supplied in deliberately mixed order. Targets for modality
positions and padding are excluded from language supervision. Gemma3's CPU
embedding hook explicitly preserves its built-in embedding scale.

## Recorded Gen9 results

The full follow-up includes every generic decoder/encoder profile in the
qualified inventory. BART, T5 and Switch need the seq2seq integration; DeepSeek
V4.1 needs its dedicated graph integration. Kimi K3 retains its existing dense,
unpadded CPU-oracle scope. There are 77 cases rather than 80 because its three
padded cases are explicitly deferred.

| Comparison | Values | Worst absolute drift | Failing metric rows |
| --- | ---: | ---: | ---: |
| First forward with modality rows | 39,831 | `2.384185791015625e-7` | 3 |
| Forward after one composed update | 39,831 | `4.76837158203125e-7` | 6 |
| Forward after two composed updates | 39,831 | `9.5367431640625e-7` | 16 |
| Raw modality embedding adjoints | 9,120 | `2.130866050720215e-6` | 36 |
| Projector input adjoints | 3,696 | `2.7008354663848877e-7` | 1 |
| Updated projector weights | 24,320 | `3.3527612686157227e-8` | 0 |
| Updated projector biases | 3,040 | `2.514570951461792e-8` | 0 |
| Updated named backbone parameters | 2,544,104 | `1.1920928955078125e-7` | 0 |

Logit comparisons use observable query rows, matching the existing CPU padding
contract; non-finite values in any row fail. Every feature gradient is checked.
All 154 loss comparisons pass the separate existing loss tolerance. There are
62 failing metric rows across the 33 failing cases.

The 17 entirely passing profiles are `bert`, `deepseek_v4`, `falcon_h1_silu`,
`kimi_k3_kda`, `kimi_k3_mla`, `kimi_k3_text`, `llama`, `mistral`, `mixtral`,
`qwen2`, `qwen3_5_linear`, `qwen3_5_mixed`, `qwen3_5_moe`, `qwen4_exp_linear`,
`qwen4_exp_mixed_ple`, `qwen4_exp_ple` and `qwen4_exp_qsa`. Other variants of the
same architecture may fail; the inventory retains results per profile/case.

For the initial eight cases, five pass completely. Phi-4 padded fails the
second raw feature adjoint at `2.384185791015625e-7`; Gemma3 unmasked and padded
fail second-step adjoints and final logits. The wider GPT-2/Llama fixtures in
the full matrix also expose first-forward drift on the new feature inputs.
Those inputs are additional numerical coverage beyond the original text gate.

State-substitution diagnostics narrow the Gemma3 padded final-logit failure:

| Backbone after two updates | Projector after two updates | GPU vs frozen CPU final logits |
| --- | --- | ---: |
| CPU | CPU | `5.960464477539063e-8` |
| CPU | Native | `5.960464477539063e-8` |
| Native | CPU | `2.980232238769531e-7` |
| Native | Native | `3.2782554626464844e-7` |

At the second backward, CPU-trained backbone weights also keep the frozen
Gemma3/Phi-4 adjoints below the gate with either projector; native-trained
backbones exceed it with either projector. This isolates accumulated backbone
update differences as a material contributor in these cases. The exact
primitive/reduction responsible remains open. The original failing trajectories
remain intact; state substitutions are diagnostics, not qualification passes.

The two GPU scatter/adjoint tests, seven standard-library negative controls and
the existing streamed-vocabulary LoRA training/resume regression pass. The
library check and both probe builds pass. All **582** frozen shader artifact
hashes remain unchanged. New negative controls are included in workspace CI.
Two CPU preparations reproduce the initial 56 case files byte-for-byte; the
expanded matrix preserves those same 56 files. Oracle provenance is a clean
Transformers `469230357aab0f2b303b0d638c1f8d06edb14184`, Torch `2.14.1+cpu`, AVX2,
one thread, with the imported source files retained in each CPU archive.

## Remaining work across the inventory

1. Close the 30 recorded failing metric rows without changing the frozen inputs,
   expectations or strict ceiling. The wide logit rows need a wide
   reduction/GEMM order that matches the reference; the adjoint rows need a
   bit-exact backward accumulation order, since their parameter updates already
   differ by ~1 ULP and the next step amplifies that seed. Add dedicated
   encoder-decoder entry points for BART/T5/Switch and the separate V4.1 graph.
2. Implement and independently qualify each actual modality stack. Gemma3's
   SigLIP tower and pool/soft-embedding projector are implemented and green on
   all 4 frozen cases (20261103 receipt). Move next to Gemma 4, then Phi-4
   image/audio branches, Qwen vision/video position handling, Kimi's
   MoonViT/projectors and the remaining modality-capable package contracts.
   Text-only models can consume an explicit external-encoder adapter; they do
   not acquire an upstream multimodal model contract by alias.
3. Qualify native tower input/parameter gradients, multimodal masks and positions,
   mixed text/modality batches, multiple items, variable feature lengths and
   preprocessing against the pinned independent CPU implementation.
4. Add composed optimizer checkpoint/resume, projector export/reload and
   standard adapter naming. The probe's per-step model exports are numerical
   evidence, not a resumable composed training checkpoint. PEFT feature training
   currently fails closed pending its own regression gate.
5. Complete physical RDNA3 replay against the same frozen CPU expectations,
   then qualify actual tower training at larger resource budgets. Preserve the
   Gen9 receipt separately and rerun existing text/generation gates before
   promoting any production CLI/GUI capability.

## Reproduction and RDNA3 continuation

From the repository root, prepare a new independent fixture directory:

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_multimodal_training.py prepare --output target/multimodal-repeat/cpu --families llama gemma3 phi4_multimodal_text qwen3_5_mixed
cargo build --locked --manifest-path hierarchos-vulkan/Cargo.toml -j1 --bin multimodal_training_probe
python hierarchos-vulkan/validation/qualify_multimodal_training.py run --fixtures target/multimodal-repeat/cpu --binary hierarchos-vulkan/target/debug/multimodal_training_probe.exe --output target/multimodal-repeat/gen9 --target gen9
```

For RDNA3, copy/extract the retained CPU fixture archive and build the same
source on that machine. Use a new output directory, `--target rdna3`, and
`--device-id 0x...` with the physical adapter's PCI device ID. The runner checks
AMD vendor identity and the requested device ID; the operator must establish
that the selected adapter is RDNA3. Compare with the Gen9 receipt without
regenerating or overwriting expectations. There is no remote RDNA3 job.

The full frozen matrix is `matrix-cpu-fixtures.zip`; `cpu-fixtures.zip` preserves
the original eight-case preparation used by the state-substitution diagnostics.
Both sit beside the receipts linked above. After extracting the full matrix to
a new directory, replay all 77 cases with the standard-library runner. A
nonzero exit is expected until the recorded red cases are repaired. The local
`.tools/gen9-multimodal-handoff-20261102.zip` additionally contains the tracked
source patch and new-file overlay against base
`2d18ac7bfec1c83ef59462ebadc7731307fb44d4`, plus application instructions.

Targeted checks:

```powershell
cargo test --locked --manifest-path hierarchos-vulkan/Cargo.toml --lib multimodal::tests -j1 -- --test-threads=1
python -m unittest discover -s hierarchos-vulkan/validation -p test_multimodal_training.py -v
python -m unittest discover -s hierarchos-vulkan/validation -p test_llava_pipeline.py -v
```

To attribute a red case instead of guessing, replay the oracle under the pinned
Torch/Transformers environment. The unpatched replay must print `0`, otherwise
the harness itself is wrong and the patch comparison is meaningless:

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/diagnose_multimodal_arithmetic.py `
  --fixtures target/multimodal-gen9/matrix-cpu --case gpt2_wide/unmasked `
  --native target/multimodal-gen9/phase4/full-matrix-sleef
```

## Shared-image projector GELU: exact-source audit (2026-11-04)

Scope: the four remaining red wide shared-image cases (`gpt2_wide` and
`llama_wide`, masked and padded). Every other stage of the shared CLIP graph is
already byte-exact: `vision_features` 0/128 and `projector_pre` 0/512 against
the frozen CPU oracle.

### What is proven

- The CPU oracle is `torch.nn.Sequential(Linear, nn.GELU(), Linear)` in
  `qualify_shared_image_pipeline.py`, so the acceptance target is PyTorch's
  AVX2 `vectorized_gelu<float>`. Recomputing the fixture's
  `projected_features` from the GPU's byte-exact `projector_pre` with local
  `F.gelu` plus `linear_2` reproduces `cpu.json` with **0/512** differences,
  so the fixture and the local wheel agree exactly.
- The shipped kernel is `aten/src/ATen/native/cpu/Gelu.h`
  `x * 0.5 * (1 + (x * M_SQRT1_2).erf())`, where AVX2 `erf()`
  (`vec256_float.h`) is the Abramowitz-Stegun polynomial
  `t = 1/(fma(0.3275911, |v|, 1))`, a four-step FMA Horner in `t`, and
  `1 + ((-exp(-v*v)) * t) * r` with the sign restored by XOR. It does not use
  `Sleef_erff8_u10`.
- `Sleef_expf8_u10` is reproduced bit-exactly: the host reimplementation
  matches the shipped `sleef.lib` on **0/10240** differences over a dense
  sweep. (The earlier `1 + fma(s*s, u, s)` note was a transcription slip in the
  Python model, not in the shader; the shader matches the library.)
- The GPU stage dump and the local `F.gelu` reference are stable across runs,
  and the earlier "shader edits change nothing" observation was two shader
  variants that evaluate to identical bits, not a stale binary.

### What remains open

- The faithful reimplementation is still **147/512** (host, hardware FMA) and
  **229/512** (GPU) one-ULP away from `F.gelu`. Searching all 64 fused/separate
  orderings of the six arithmetic steps and all 15625 ULP perturbations of the
  six A&S coefficients never improves on that 147 baseline, so the residual is
  neither an ordering choice nor a constant difference.
- A correctly rounded (float64) GELU is strictly worse (**425/512**), so
  matching PyTorch requires its exact rounding, not a more accurate value.
- Downstream effect: `projected_features` differ on 264/512 by at most
  `3.7e-9`, which the text graph amplifies to `2.086e-7`..`3.278e-7` at the
  logits, just above the `2e-7` gate.

### Consequence

The wide shared-image cases cannot be closed by tuning the GPU arithmetic
further; the blocking question is which GELU the shipped wheel actually
executes, since it is observably not the published A&S kernel yet is only one
ULP away from it. Only the two wide profiles (`gpt2_wide`, `llama_wide`) are
affected: every narrow shared-image profile and every text-only case still
passes at `max_abs < 2e-7`.

## Receipt: fresh-build re-qualification (native V4.1 joint, 2026-11-05)

**Attempt-12 was measured on a stale binary.** The binary it ran was linked at
17:59; `native_vision_backward_tests.rs` and `orchestrator_tape.rs` changed at
18:01. Its reported `sparse-adjoints FAIL` is therefore not a property of the
current source.

Rebuilt from current source (`cargo test --lib --no-run`, exit 0, 7m44s,
`hierarchos_vulkan-25096d1b6b51d860.exe`), then re-ran on that one binary:

* **sparse adjoints: PASS** — 8 forward/adjoint cases, 7,720 exact gradient
  words, 1.99s. The attempt-12 failure was the stale binary, not arithmetic.
* **checkpoint/resume/export/freshness: PASS** — 45.35s.
* **joint two-AdamW-step gate: FAIL**, worst `3.203749656677246e-7`, 8 of 1,866
  comparisons (87,185 of 155,444 values already bit-exact).
* **The joint failure is deterministic.** Two runs on one binary produced
  byte-identical qualification JSON (`aa7504c7…`), so this is a fixed
  divergence, not atomic-accumulation noise. No deterministic-reduction work is
  needed.

All eight failures are downstream of **one** gradient:
`native_vit_4x4/0/image-gradient/vision.patch_embed.proj.bias` differs by 4 ULP
(2.384e-7 on 0.7739). The step-1/step-2 logits and the loss differ only because
that gradient shifts the first AdamW update.

**The bias-reduction order hypothesis is eliminated.** `torch 2.14.1+cpu`
(AVX2) `sum(dim=0)` on `(rows, 16)` float32 matches the shader's 4-lane
interleave `((p0+p1)+p2)+p3` bit-for-bit — 120 random trials at each of
rows ∈ {4,8,12,16,20,24,32,48,64}, 1920/1920 values every time. At the 4×4
geometry (`3×2×2` patches → 4 rows) that formula *is* a plain sequential sum,
and all candidate orders agree there. So the kernel is correct and the residual
is **input-side**: either the `grad_output` fed to the patch linear differs, or
the written bias-grad buffer is not `Σ rows`.

Harness fix: `run_v41_joint_phase.py` no longer hard-codes a binary path (the
relink changed the metadata hash and silently pointed it at a retired binary);
it now resolves the newest `hierarchos_vulkan-*.exe`.

Next probe: dump the patch linear's `grad_output` at 4×4 step 0 and compare
`Σ rows` against the bias grad actually written, which separates "dy differs"
from "kernel writes ≠ Σ dy".

### Addendum: the divergence is confined to the bias reduction

Narrowing on the fresh binary, at 4×4 step 0 the joint gate's **`patch-gradient`
and `input-gradient` both pass word-for-word**. That is decisive about scope:

* `input-gradient` exact ⇒ the language graph's gradient into the stream is
  exact.
* `patch-gradient` exact ⇒ the whole vision backward (two blocks, aligner, and
  the patch linear's input adjoint `dx = g·W`) is exact.

So neither the vision arithmetic nor the upstream gradient is at fault. The only
element that differs is the 16-word `vision.patch_embed.proj.bias`, i.e. the
reduction `Σ_rows g` for a 4-row matrix whose `dx = g·W` is simultaneously exact.

The component oracle's own bias gradient *does* agree with the shader's
reduction (that test is bit-exact), so the component oracle uses the same
sequential/4-lane sum. The joint oracle is therefore the outlier. Two remaining
possibilities: the joint oracle's reduction differs from the component's, or its
`g` differs by ~1 ULP and `dx = g·W` absorbed that difference into identical
words (possible but not yet demonstrated).

`feature_gradient` is not compared anywhere in the joint gate — that is the
missing intermediate. Next probe: dump the patch linear's `g` at 4×4 step 0 and
compare `Σ_rows g` against both the written bias and the joint oracle's bias;
that separates "oracle reduction differs" from "g differs".

### Correction to the addendum above

Two claims in the previous addendum were too strong and are corrected here:

1. `patch-gradient` and `input-gradient` passing does not mean the vision
   backward is wholly exact. The gate reports `bit_exact: 87,185` of
   `values: 155,444`, i.e. **68,259 values differ**; 1,858 of 1,866 comparisons
   are merely *inside* the 2e-7 budget. The patch-embed bias (2.384e-7) is the
   only one that crosses it, but it is the largest of a broad field of
   ULP-level differences, not an isolated outlier in an otherwise exact graph.
2. A `#[cfg(test)]` trace of the patch linear's `grad_output` was added to
   `deepseek_v41_vision.rs` and read back as 64 zero words, so the trace does
   not capture that buffer at the point of use. The instrumentation was
   reverted; the "g differs vs oracle reduction differs" question is therefore
   still open, and needs a trace that snapshots the buffer immediately rather
   than at `finish()`.

### Phase result: the image path is exonerated; the residual is in the language backward

Instrumenting the joint backward (probes reverted afterwards, tree compiles clean) gave
one answer that overturns the earlier attribution.

`input-gradient` is **not** word-exact: `512 values, worst 3.72529029846e-8 at 282`.
`patch-gradient` likewise: `48 values, worst 3.35276126862e-8 at 16`. Both passed only
because they sit inside the 2e-7 budget, and I had wrongly read "passed" as "bit-exact".
The stream gradient therefore already carries ~1 ULP of error before the image module
sees it, and every image-side error measured downstream is inherited:

* `image/merged` (after `hc_reduce`) differs from `stream_gradient[3,0] + [3,1]` in
  22 of 32 words, by 1-2 ULP — consistent with the input differing, not with a
  reduction bug.
* `image/projected` vs the frozen `feature_gradient`: 10 of 32 exact, worst 2.98e-8.
* `patch_embed/g` sums exactly to the bias the kernel writes (`0.773911714553833`),
  so the bias kernel is self-consistent; it is faithfully summing an already-off `g`.

Conclusion: the ViT, the aligner, the merge, the hc reduction and the bias kernel are
not the defect. The 3.2e-7 joint residual originates in the **language backward**, at
roughly 1 ULP on a subset of the stream gradient (first observed at flat index 282,
i.e. row 4 / hc 0 / dim 26 under the `(rows, hc, dim)` layout). The bias at 4x4 step 0
is simply the most exposed amplification of it, which is why it is the only comparison
that crosses 2e-7.

Search space for the remaining work is now the language graph's adjoint, not the
native vision path. Nothing in this phase changed a kernel, fixture, comparator or gate.

### The residual is one ULP at layer 2's output gradient

`compare_v41_native_backward.py` against the frozen language CPU trace, run on the
joint ladder capture (fresh binary, `native_vit_4x4`):

| stage | differing words | max |
|---|---|---|
| `0/5/...` (all stages) | 0 | 0.0 |
| `0/4/...` (all stages) | 0 | 0.0 |
| `0/3/...` (all stages) | 0 | 0.0 |
| `0/2/output/grad` | **99 / 512** | 7.450581e-09 |
| `0/2/mixed/grad` | 47 / 256 | 2.9802322e-08 |
| `0/1/output/grad` | 256 / 512 | 1.4901161e-08 |
| `0/0/output/grad` | 317 / 512 | 2.9802322e-08 |

Layers 5, 4 and 3 are word-exact in every recorded stage. The first word that differs
anywhere in the language graph is in `0/2/output/grad`, and it is exactly one ULP
(7.45e-9 = 2^-27). Every downstream language stage and, through the stream gradient,
every image-side number inherits that single rounding.

So the joint gate's 3.2e-7 is one first-rounding event at layer 2 amplified by two
AdamW steps — not a vision bug, not an hc-reduction bug, not a bias-kernel bug, and not
the tolerance. The remaining work is precisely scoped: find the reduction in layer 2's
block backward whose intermediate rounding differs from the CPU's, at 99 of 512 words.

### Where the first divergent word is actually born

`{l}/output/grad` is not produced by layer `l`. In `backbone_tape.rs::backward_retained`
it is the value of `stream_grad` read at the top of iteration `l`, and `stream_grad` is
built at the **end of iteration `l+1`**:

```
stream = add(post.grad_residual, input_collapse.grad_stream)
stream = add(stream, mix)
stream_grad = engram ? engram.backward_retained(..) : stream
```

So `0/2/output/grad` (first divergence, 1 ULP, 99/512) is written by the tail of **layer
3's** iteration. Every layer-3 value that *is* traced is exact — `output/grad`,
`mixed/grad`, `ffn_*`, `after_attention/grad`, `attended/grad`,
`attn_norm_collapsed/grad`. What is not traced is the attention-stage tail:
`hc_split(Attention)`, `hc_mixes(Attention)`, the two elementwise adds, and the optional
`engram.backward_retained`. Only the FFN-stage equivalents (`ffn_residual/grad`,
`ffn_collapse/grad`, `ffn_mix/grad`) are recorded, and those are exact.

That is the precise remaining target: the untraced attention-stage tail of layer 3, in
one of `hc_split` / `hc_mixes` / `add` / `engram`, producing a single 2^-27 rounding at
99 of 512 words. Note `hc_post_input_grad.comp` deliberately avoids `fma` and sums in
ascending copy order with `precise float`, so that shader is already written for
CPU-identical rounding; `hc_mixes_input_grad.comp` instead calls `v41_gen9_fma`, which
deserves the first look.

No kernel, fixture, comparator or gate was modified in reaching this conclusion.

### Final verification on the clean tree

Rebuilt from source with every probe reverted, then ran the V4.1 gates against that
exact executable. The joint numbers are byte-for-byte the ones from the instrumented
build, which also proves the probes did not perturb the arithmetic:

```
all_comparisons { bit_exact: 87185, comparisons: 1866,
                  max_abs: 3.203749656677246e-07, values: 155444 }
pass: false   completed_cases: 2/2   optimizer_updates: 4
failures: 8
  native_vit_4x4/0/image-gradient/vision.patch_embed.proj.bias  2.38418579101562e-7
  native_vit_4x4/1/{inference,training}-logits                   2.38418579101562e-7
  native_vit_4x4/1/loss                                          2.38418579101562e-7
  native_vit_4x4/2/inference-logits                              2.38418579101562e-7
  native_vit_6x4/1/{inference,training}-logits                   3.20374965667724e-7
  native_vit_6x4/2/inference-logits                              2.98023223876953e-7
```

Across roughly 100 selected tests run on this binary, the **only** non-zero exit is the
joint gate (`exit=101`). Everything else is 0, including
`deepseek_v41_attention_grad::tests::deepseek_v41_sparse_attention_adjoints_cpu_pytorch_strict`,
the four `backbone::block_grad_tests` (hc_post, hc_collapse, ffn_norm, moe), the
compressor/engram adjoints and the V4.1 checkpoint suites. So the sparse-adjoint failure
recorded in the handoff log is definitively a stale-binary artefact, not a regression.

State left behind: no kernel, shader, fixture, comparator or gate modified; no commit;
only this record changed. The joint multimodal gate remains open at 3.2037e-7 with the
residual traced to the untraced attention-stage tail of layer 3.

### Engram eliminated; the tail is the hc chain and two adds

The optional step in layer 3's tail is not present. Neither frozen fixture carries any
engram configuration, and the harness builds the pipeline with
`"engram_layer_ids": []` (`backbone_tests.rs:346`), so `t.engram` is `None` and
`engram.backward_retained` never executes. That removes the only asymmetric branch
between layer 3 and the exact layers 4 and 5.

What remains of the tail is:

```
input_collapse = hc_pre(norm, stream, previous_pre)        // grad_stream, grad_pre
split          = hc_split(Attention, projection, ...)      // grad_post, grad_comb
mix            = hc_mixes(Attention, split, ...)
stream         = add(post.grad_residual, input_collapse.grad_stream)
stream         = add(stream, mix)                          // -> 0/2/output/grad
```

All three inputs to that sum are derived from `attn_norm_collapsed/grad`, which is
word-exact. The divergence therefore arises inside this chain for layer 3's data, at a
1-ULP tie. The FFN-stage equivalents of the same three kernels (`ffn_residual/grad`,
`ffn_collapse/grad`, `ffn_mix/grad`) are exact for the same layer, which is consistent
with a tie that only layer 3's attention-stage operands happen to hit.

Further bisection needs CPU-side operands for these intermediates, which the frozen
traces do not contain and which cannot be regenerated without touching frozen roots.
The honest stopping point is here: the defect is one tie-breaking rounding in layer 3's
attention-stage hc chain, and closing it requires either a CPU operand trace produced
outside the frozen fixtures, or a bit-level argument for each of `hc_pre` / `hc_post` /
`hc_mixes` reductions against PyTorch's summation order at hc=2.

### The mix adjoint is provably correct; Sinkhorn is the last candidate

Fingerprinted `torch 2.14.1+cpu` (1 thread) on the exact joint geometry, a `[8, 8] @
[8, 64]` float32 product, 30 random trials, per-row exact-word counts:

| rows | sequential FMA chain | even/odd + mul/add | even/odd + FMA |
|---|---|---|---|
| 0-5 | **100.0%** | 36-39% | 41-43% |
| 6-7 | 38-39% | **100.0%** | 61% |

So torch splits the eight rows into a six-row panel plus a two-row tail, and evaluates
the tail with an even/odd split over separately rounded products. The shader's
`short_panel` branch (`tail = rows % 6`, `tail <= 2`, even/odd with `product = d*w;
a += product`) implements *exactly* that, and its else-branch is the FMA chain. At the
joint geometry `hc_mixes_input_grad.comp` therefore reproduces torch word-for-word on
100% of rows. The strongest remaining candidate is eliminated by measurement, not by
argument.

What is left in layer 3's untraced tail is the Sinkhorn chain
(`transformer_deepseek_v41_hc_split{,_grad}.comp`). Its header asserts "the released
implementation runs this in a TileLang kernel, so the scalar order below is the contract
rather than a PyTorch reduction" -- but the fixture's `comb` does not come from TileLang.
The oracle injects `hc_split_sinkhorn` into the model namespace from the training
harness (`training.KERNEL`) and `freeze_deepseek_v41_hc_backward_avx2.py` wraps that same
injected function to observe it. If that CPU-path scalar order differs from the shader's
by one rounding, `comb` is off by ~1 ULP, `hc_post`'s `sum_j comb[k,j]*grad_out[j]`
inherits it, and `stream_grad` carries it into `0/2/output/grad` -- the observed first
divergence, data-dependent and therefore layer-selective.

`hc_split_sinkhorn` is not defined in `validate_deepseek_v41_full_training.py`; it is
reached through that module's `KERNEL` binding, which is the next thing to read and
compare against the shader's normalization order.

No source file was modified in this phase.

### The Sinkhorn contract is mis-stated, and that is the live defect

The oracle's `hc_split_sinkhorn` resolved to
`target/deepseek-v41-reference/inference_kernel.py`, which is a **TileLang kernel**
(`@tilelang.jit` `hc_split_sinkhorn_kernel`) that writes `pre`, `post` and `comb` into
tensors allocated with `mixes.new_empty(...)` and returns them. A TileLang kernel call
registers no autograd node, so that path cannot produce the fixture's gradient for
`comb` -- and the joint fixture does carry one (`image-gradient`/`adapter-gradient` for
every `ffn` parameter, `0/3/ffn_comb/grad` among the traced words).

The joint fixture's own `contract` field says which path was actually used:

> "Unchanged released Block.hc_mixes with **CPU hc_split_sinkhorn fallback** and
> unmodified Torch autograd"

So the fixture's `comb` came from a pure-PyTorch fallback installed by
`training.install_functional`, and its gradient is plain torch autograd through
`softmax -> /(row_sum) + eps -> /(col_sum + eps) -> 19 x [/(row_sum+eps),
/(col_sum+eps)]`. With `hc = 2` every reduction there is a two-term sum, so order is
irrelevant; what matters is that each step's arithmetic is the *torch op*, not a
re-derived scalar chain.

`transformer_deepseek_v41_hc_split_grad.comp` nevertheless states:

> "The released implementation runs this in a TileLang kernel, so the scalar order below
> is the contract rather than a PyTorch reduction."

That premise is false for this fixture. The shader replays the normalization chain from
the stored softmax output with its own `precise` ordering, while the oracle's `comb` was
produced by torch's `exp`/`div`/`add` sequence. A one-ULP disagreement in `comb` enters
`hc_post`'s `grad_residual = sum_j comb[k, j] * grad_out[j]`, which is added into
`stream_grad`, which is exactly `0/2/output/grad` -- the observed first divergence, and
one that can be layer-selective because it only shows where the two evaluations round
differently.

This is the defect to fix, and it is a contract mismatch rather than a mystery: the
forward chain and its adjoint in `hc_split{,_grad}.comp` must be re-derived to match
`torch.nn.functional.softmax` plus the explicit `+ eps` / `/ (sum + eps)` steps that the
fallback actually executes, instead of the TileLang scalar order they mirror today.

No source file was modified in this phase; the tree still builds and the joint gate still
reads 3.203749656677246e-07.

#### Refinement: the forward is not the suspect, the adjoint is

The conclusion above needs one correction before it is acted on. At step 0 the gate
reports `native_vit_4x4/0/inference-logits worst 0.0`, `training-logits worst 0.0` and
`loss worst 0.0` -- the entire forward and the loss are bit-exact, and `comb` enters that
forward through `hc_post`. A one-ULP error in the forward `comb` would therefore have to
perturb the logits, and it does not. So the Sinkhorn *forward* chain is fine.

What is left is its adjoint: `transformer_deepseek_v41_hc_split_grad.comp`. Torch's
autograd through the fallback's `y = x / (sum(x) + eps)` step produces
`d_x = d_y / D + sum_axis(-(d_y * x) / D^2)` with `D = sum(x) + eps`, evaluated as
separate `div` and `mul` ops in torch's own order. The shader implements the same
algebra but as a hand-written scalar chain. With every reduction being two-term at
`hc = 2`, order cannot differ; only the *expression shape* can -- `d_y / D` versus
`d_y * (1/D)`, and `-(d_y * x) / D^2` versus `(-d_y * x) / (D * D)` each round once
differently.

That single rounding lands in `comb`'s adjoint, then in `hc_post`'s
`grad_residual = sum_j comb[k,j] * grad_out[j]`, then in `stream_grad` --
`0/2/output/grad`. This remains the live defect, and the fix is to reshape those two
expressions to torch's op order rather than to re-derive the forward.

#### Correction: the Sinkhorn adjoint division is also correct

The refinement above is wrong, and the measurement says so. `hc_split_grad.comp`'s AVX2
branch computes the div-backward as `q = (x/D)/D`, `tail += -g*q`, `d = g/D + tail`; the
scalar branch computes `tail += (g*x)/(D*D)`, `d = g/D - tail`. Those are different
roundings, so I fingerprinted torch's autograd on `y = x / (x.sum(-1, keepdim=True) +
eps)` at hc=2, 300 trials, 1200 cells:

| shape | exact |
|---|---|
| `(g/D) + sum(-g * ((x/D)/D))` | **1200/1200 (100.0%)** |
| `(g/D) - sum((g*x)/(D*D))` | 677/1200 (56.4%) |
| `(g/D) - sum(((g*x)/D)/D)` | 687/1200 (57.2%) |
| `(g - sum(g*x)/D)/D` | 381/1200 (31.8%) |

Torch uses exactly the shape the shader's AVX2 branch already implements. The Sinkhorn
adjoint division chain is therefore word-correct, and the live-defect claim above is
withdrawn.

Candidates eliminated by 100% bit-level fingerprints so far: the bias-grad reduction
order, the `hc_mixes` input adjoint panel structure (6-row FMA + 2-row even/odd), the
Sinkhorn div-backward shape, the hc_post input adjoint (mul+add, two terms), and Engram
(absent from the config). What has not been fingerprinted is the Sinkhorn's **softmax
Jacobian replay** (`s * (d - sum(d*s))` and the `+eps` placement across 19 iterations),
and the `hc_pre` collapse's `grad_pre` reduction over `dim = 32`.

The joint gate still reads 3.203749656677246e-07. No source file was modified.

#### The untested axis is Sinkhorn iteration count, not the chain's shape

`freeze_deepseek_v41_hc_backward_avx2.py` already generates exactly the CPU operands I
said were missing -- per case it stores `raw`, `grad_pre`, `grad_post`, `grad_comb`,
`mix_gradient` and `input_bits`, and the corresponding native gate
(`backbone::block_grad_tests::deepseek_v41_hc_*_adjoint_matches_the_pinned_cpu_fixture`)
**passes** for `rows in {1, 2, 3, 8, 9}` and `hc in {2, 4}`. So the whole
`hc_mixes -> sinkhorn -> mix_gradient -> hc_mixes_input_grad` chain, including the
Sinkhorn adjoint, is already proven word-exact at `rows = 8, hc = 2` -- the joint model's
geometry.

That fixture's config is not the joint model's, though:

| parameter | hc fixture (passes) | joint model (fails) |
|---|---|---|
| `hc_sinkhorn_iters` | **3** | **20** (`ModelArgs` default) |
| `norm_eps` | 1e-20 | base config value |
| `hc` | 2 and 4 | 2 |

`hc_split_grad.comp` replays the normalization chain to invert it ("the pre-step matrix a
step needs is replayed from the stored softmax output"), and the replay length is
`sinkhorn_iters - 1` = 2 in the passing fixture versus 19 in the joint model. A
rounding drift in that replay would therefore be invisible to every passing gate and
would appear only in the joint one, at exactly 1 ULP, layer-selectively -- which is the
observed signature.

Next step is concrete and does not touch any frozen root: add a 20-iteration case at
`rows = 8, hc = 2` to the hc-boundary harness and compare its `mix_gradient` words
against the native kernel, or run the existing native hc test with
`hc_sinkhorn_iters = 20`. That isolates the replay loop without another language-level
trace.

No source file was modified. The joint gate still reads 3.203749656677246e-07.

#### Final eliminations: reducer, softmax Jacobian and sigmoid ordering are all exact

Four more measurements, all against `torch 2.14.1+cpu` with one thread:

1. **The contiguous sum reducer.** `fp32_v41_reduce.glsl`'s fold is eight banks
   combined *sequentially*, which the fingerprint shows is torch's shape
   (`lane8_seq` 1200/1200; tree combine 577/1200). Re-implementing the shader's whole
   `v41_sum_squares` cascade and testing it against `x.square().sum()` gives
   **80/80 at width 32, 64, 128 and 256** -- the exact widths the hc tail uses
   (`dim`, `hc*dim`). `rstd` is bit-exact.
2. **The softmax Jacobian.** Against torch's own softmax output (not a reconstruction):
   `s * (g - dot)` is **1600/1600**; `s*g - s*dot` 41.6%; the fma form 49.5%.
   `hc_split_grad.comp` computes `softmax * (gradient - dot)` -- exactly the winning
   shape.
3. **The sigmoid branches.** torch's sigmoid backward is `grad * (1 - y) * y`; the
   shader's AVX2 branch is `(grad * (1-sigmoid)) * sigmoid`. Same order.
4. My earlier "0/60 at every width" reducer result was my own harness error -- it
   compared sum-of-squares against a plain sum. Corrected above; it is not a finding
   about the shader.

So every structural component of the hc tail is now bit-verified: reducer, hc_mixes
panel structure, Sinkhorn div-backward, softmax Jacobian, sigmoid ordering, forward
pass. The earlier "untested axis is iteration count" note stands only in the sense that
no fixture exercises `sinkhorn_iters = 20`; the replay's parity convention (`steps = 1 +
2*(iters-1)`, columns at odd steps) is consistent for any count, so an error there would
be gross, not 1 ULP.

What remains untested is not a shape but a **configuration**: the entire hc-boundary
fixture set runs `hc_sinkhorn_iters=3` and `norm_eps=1e-20`, while the joint model runs
20 and the base config's `norm_eps`. A 20-iteration case at `rows=8, hc=2` is the one
experiment that would close or re-open the tail, and it does not touch any frozen root.

The joint gate still reads 3.203749656677246e-07. No source file was modified.

#### Optimization 22: memoized Sinkhorn state in the HC adjoint (4.5-5.4% per training step)

This continues the throughput work begun in `optimization-training-batched21`.
The corpus profile (`optimization-baseline20/profile.json`) put
`transformer_deepseek_v41_hc_split_grad_avx2` first by GPU cost: 48 dispatches
for 119.76 ms, 2.57 ms each against a 0.035 ms corpus average. The cause is
structural, not arithmetic -- the reverse loop rebuilt the forward state for
every step:

    for (step = steps; step >= 1; --step) {
        state = matrix;
        for (applied = 1; applied < step; ++applied) forward_step(applied);
        backward_step(step);
    }

That is `steps(steps-1)/2` normalizations. At the model's `sinkhorn_iters = 20`
it is 741 replays per row per layer, 38 of which are distinct. The shader
already carried a bounded `HIERARCHOS_V41_HC_CACHE` path that retains every
forward state in 2,496 bytes of workgroup storage and replays nothing
(`_cached_avx2.spv`, compiled and listed in `training_shader_manifest.json`), but
nothing selected it: `cpu_math`'s pair table still mapped the native slot to the
replay SPV.

`src/deepseek_v41/backbone.rs` now selects it, and only where it is valid:

* the cached kernel is constructed only when `cpu_math::selected(device,
  HC_SPLIT_GRAD)` is not the base SPV, i.e. the pinned AVX2 arithmetic is what
  the device would run anyway;
* the dispatch uses it only for `hc <= 4 && iterations <= 20`, the geometry the
  variant pins; every other case keeps the replay kernel unchanged.

Verification is a same-binary A/B, so neither arm inherits build or clock
differences. `HIERARCHOS_V41_HC_REPLAY=1` pins the replay kernel under
`cfg(test)`; production builds ignore it (`!cfg!(test)` short-circuits before
the environment is read).

| case | replay median | memoized median | speedup | submissions |
| --- | ---: | ---: | ---: | ---: |
| native_vit_4x4 | 988.2 ms | 937.4 ms | 1.0542x | 1258 -> 1258 |
| native_vit_6x4 | 979.5 ms | 937.3 ms | 1.0450x | 1258 -> 1258 |

Medians over both rounds (four samples per arm per case). Per round:
1.0457x/1.0851x and 1.0574x/1.0268x. Submission count is unchanged, as expected:
this removes GPU work, not submissions.

`compare_training_benchmark.py` asserts `final_logits_bits`, `image_digest` and
`adapter_digest` are identical between arms, and passes in both rounds
(`COMPARE_EXIT=0`): nine updates leave the logits, every trained image parameter
and every adapter word identical. Memoizing a deterministic computation cannot
move a bit, and the measurement confirms it.

Gates on the same executable (`optimization-cache23-gates/`):

| gate | result |
| --- | --- |
| joint two-AdamW-step | pass -- 1,866 comparisons, 155,179/155,444 exact, max_abs 1.4901161193847656e-8, 1,360 logits, 4 updates |
| joint checkpoint/resume/export | pass |
| `block_grad_tests` (hc collapse/post fixtures) | 4/4 |
| `hc_adjoint` (released HC adjoint + modality repeat) | 2/2 |
| native vision two-step | 1/1 |

Honest limits of this receipt:

* Absolute milliseconds are from the default test profile, not the `debug=0`
  qualification build that produced the 655 ms figure. The claim is the **ratio
  on one binary**, not a new absolute.
* Timestamp-query profiling emitted no `hc_split_grad` lines in this libtest
  build, so the per-kernel drop is not re-measured here. The 2.57 ms figure is
  the motivation from the earlier profile, not a post-hoc result.
* The gain applies to `hc <= 4` and `<= 20` iterations. Larger geometry still
  takes the replay path, which is correct but unchanged in cost.
* No shader source or SPIR-V was modified: the memoized variant already existed
  and was already recorded in the shader manifest.

Operational note: this tree (`C:\Users\User\Hierarchos-Native`, branch
`codex/gen9-multimodal-training`) is the authoritative one. The editor opened
`C:\Users\User\jews\Hierarchos-Native`, a separate `main` clone from 2026-10-30
that has no `vision_train.rs`, no `target/multimodal-gen9` and no Gen9 evidence.
An edit initially landed there by path resolution and was reverted exactly; that
clone has zero references to the change and is otherwise untouched.

No commit or push was made.

Tested executable: `hierarchos_vulkan-25096d1b6b51d860.exe`, SHA-256 `fcd68e6584caeeebc9fde25e65dbff6941085b737f63505da0a36fa63f2a77d2`.

#### Optimization 24: where the 1,258 submissions actually go

Before changing submission granularity again, the split had to be measured rather
than guessed. `vulkan::submission_report` (test-only, compiled out of non-test
builds, inert unless `HIERARCHOS_VULKAN_SUBMISSION_REPORT` is set) attributes
every `ComputeBatch::submit` and `submit_and_restart` to its call site through
`#[track_caller]`, and the throughput benchmark dumps the histogram.

One run, 50 distinct sites, 24,461 submissions over 18 case-steps
(`optimization-report24/benchmark.log`). Normalised to one case-step:

| site | per case-step | share |
| --- | ---: | ---: |
| `vulkan.rs:5035` (`fn copy_buffer`) | 494.5 | 36.4% |
| `vulkan.rs:5239` (`BatchHandle::submit`, Owned) | 358.8 | 26.4% |
| `deepseek_v41/linear_grad.rs:471` | 98.0 | 7.2% |
| `deepseek_v41/attention_projection_grad.rs:169` (`add`) | 50.0 | 3.7% |
| `deepseek_v41/mixed_moe_grad.rs:410` (`add`) | 36.0 | 2.6% |
| `deepseek_v41/linear_grad.rs:270` | 26.0 | 1.9% |
| `mixed_moe_grad.rs:237`, `:457` | 24.0 each | 3.5% |
| `vulkan.rs:6633` (watchdog slice) | 24.0 | 1.8% |
| `vulkan.rs:5359` (session retirement) | 23.8 | 1.8% |
| four `backbone.rs` hc stages | 12-13 each | 3.6% |
| remaining 38 sites | ~180 | 13% |

Two things follow, and the second corrects the plan I started with.

1. **63% of submissions are transport, not arithmetic.** `copy_buffer` alone opens
   its own `ComputeBatch`, records one `vkCmdCopyBuffer` and submits: one queue
   submission and fence wait to move one buffer. It is the largest single
   reducible group by a wide margin, and it is a pure copy, so batching it cannot
   change any arithmetic.
2. Further kernel fusion is no longer the lever. The per-stage work has already
   been merged (batched21); what remains is one submission per stage handle
   (26%), one per buffer copy (36%), and a long thin tail. The next change must
   be submission *granularity*, not more dispatch merging.

The report is reproducible: build, then run the throughput benchmark with
`HIERARCHOS_VULKAN_SUBMISSION_REPORT=1`.

No behavior changed in this step; the instrumentation is `cfg(test)`-only and the
joint/checkpoint gates are unaffected by it.

#### Optimization 25: the 495 copies are the optimizer moving one tensor at a time

`copy_buffer` has four callers that all funnel through one `submit()`, so the
first histogram could not separate them. `submission_report::label` now tags the
funnel for the duration of each call (`LabelGuard`, thread-local, test-only), and
one more run splits the group (`optimization-label25/benchmark.log`):

| caller | per case-step | share of all submissions |
| --- | ---: | ---: |
| `BatchHandle::submit` (Owned) | 358.8 | 26.4% |
| `upload_bytes` | 222.4 | 16.4% |
| `read_f32` | 203.8 | 15.0% |
| `duplicate_f32` | 54.0 | 4.0% |
| `read_u32` | 23.8 | 1.7% |

Totals are unchanged (1,358.9 per case-step), which is the control that the
labelling re-attributes rather than adds.

The numbers identify the source rather than merely ranking it: 203.8 readbacks
and 222.4 uploads per step is one device-to-host copy and one host-to-device
upload per adapter factor. The model trains 106 adapter modules with two factors
each, so the CPU AdamW is moving 212 gradients down and 212 updated factors up
as 424 separate submissions -- each one a staging allocation, a
`vkCmdCopyBuffer`, a queue submission and a fence wait. `duplicate_f32` (54) is
the tape preserving operands, also one submission per tensor.

That makes the next change unambiguous and it is not kernel work: a batched
download/upload that packs N tensors into one staging slab, records every copy
into a single `ComputeBatch`, submits once and maps once. Both directions are
pure copies, so no arithmetic can move; the only requirement is that the caller
asks for all tensors before mapping, which the optimizer step does naturally.
Expected effect is up to ~30% of the step, since these 426 submissions are
31% of the total and the measured step is submission-bound.

Not yet implemented. It rewrites the optimizer's per-tensor transfer in
`model_train.rs` and the vision parameter path, and it must be verified by the
same-barrier byte-identity check the memoized Sinkhorn change used. Landing it
half-verified would be worse than leaving the measurement.

No arithmetic changed in this step: the label guard and the recorder are
`cfg(test)`-only, and the private `copy_buffer` signature gained a label that
non-test builds discard.

#### Optimization 26: one transfer per adapter gradient group

Measured in Optimization 25: 203.8 device-to-host submissions per step, two per
adapter module, because `read_f32` on a device-local buffer allocates staging,
opens a `ComputeBatch`, submits and fences for each small span.

`GpuBuffer::read_many_f32` now downloads an ordered list of spans with one
staging slab, one `ComputeBatch`, one `submit()` and one map.
`DeepseekV41LinearGradients::read_adapter_gradients_gathered` validates the
targets exactly as the single-module read does, gathers both factors of every
module, and slices the result. The two call sites that looped
(`attention_projection_grad.rs::adapter_gradients`,
`mixed_moe_grad.rs::collect_adapters`) now use it; the single-call Engram site
keeps the untouched per-module path.

Submissions, same binary, same fixtures (`optimization-batch26/benchmark.log`):

| | before | after |
| --- | ---: | ---: |
| total over 18 case-steps | 24,461 | 21,653 |
| per case-step | 1,358.9 | 1,202.9 |
| timed region | 1,258 | 1,102 |
| `read_f32` site | 203.8 | 0 (replaced by the batched submit) |
| batched download submit | 0 | 36.0 |

Timing, medians of two post-change rounds against the pre-change round
(`optimization-batch26`, `optimization-batch26b`), same executable:

| case | before | round 1 | round 2 | median speedup | saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 1006.0 ms | 983.9 ms | 919.0 ms | 1.0573x | 54.6 ms |
| `native_vit_6x4` | 1021.6 ms | 973.2 ms | 948.3 ms | 1.0634x | 60.9 ms |

Round-to-round spread on this device is several percent, so the two rounds are
reported rather than the better one. Only ratios measured on this one binary are
claimed.

Bit-exactness: `compare_training_benchmark.py` reports every case word-exact
after 9 updates in both rounds -- identical `final_logits_bits`, `image_digest`
and `adapter_digest` against the pre-change run.

Gates on the same binary (`optimization-batch26-gates/`): joint two-AdamW-step
pass (`max_abs 1.4901161193847656e-8`, 155,179/155,444 exact, 1,866 comparisons,
`pass: true`), checkpoint/resume/export pass, `block_grad_tests` pass,
`hc_adjoint` pass.

Remaining, and why it is not in this change: the 222.4 uploads per step are the
adapter factors being rebuilt into per-target GPU buffers, one allocation and
submission each. Collapsing them needs an arena the resident LoRA views into,
not a copy change -- every factor is currently its own `GpuBuffer` with its own
descriptor.

#### Optimization 28: bounded component uploads with shared host staging

The Optimization 25/26 attribution that every remaining upload was an individual
LoRA factor was incorrect. CPU adapter attachment performs no GPU upload, and
each linear projection already batches its base plus both factors. The remaining
cost included HC coefficients, auxiliary attention tensors, expert/gate and
vocabulary admissions. `ParameterUploadBatch` now groups those component
parameters, while preserving individual GPU buffers and descriptors. Its
16 MiB pending-payload budget flushes before overflow; oversized tensors retain
their old boundary. Small FP32 spans reuse the existing 4 MiB host upload arena.
Streamed expert and packed-expert arena policies stay intact.

Same-binary final-source ABBA comparison, two pairs, no competing GPU/compiler
work, seven measured steps after two warmups:

| Native case | Individual uploads | Grouped uploads | Throughput ratio |
| --- | ---: | ---: | ---: |
| `native_vit_4x4` | 601.843 ms | 572.105 ms | 1.0520x |
| `native_vit_6x4` | 618.413 ms | 579.502 ms | 1.0671x |

Timed submissions fall from 1,102 to 982 per native step. Both arms and both
rounds retain identical logit words and trained image/adapter digests after nine
updates for both native cases and shared LLaMA/GPT-2/T5 controls. Earlier timings
are retained as history; their build conditions differ, so ratios are not combined.

Final-source regressions: all 44 profiles, 95 image cases, 166 text cases; 30 native
qualifier executions, all 1,360 native logits word-exact; 215 broad DeepSeek tests
verified and 26 opt-in tests ignored; 672 shader recipes rebuilt and validated;
208 historical fixture/SPV files unchanged. The broad suite's first command had
214 passes and one trace-write failure from a missing isolated-build output
directory. After creating the directory the sole case passed on the same binary;
the original log and separate recovery receipt are preserved.

The current [qualification and replay guide](hierarchos-vulkan/validation/release_parity/multimodal_training/20261105-gen9-v41-upload-batching/README.md) binds source,
binaries, inventory, samples and logs. The maintained
`validation/benchmark_deepseek_v41_training.py` runner compares the test-only old
upload arm against production grouping in one executable. The new November 5
source transfer preserves the previous archives and reuses the unchanged oracle
bundle. RDNA3 was not accessed; physical replay and production-size throughput
remain pending.
