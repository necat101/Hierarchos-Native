# Multimodal training progress

## Current phase: working CLIP/LLaVA image pipeline

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
**44/77 to 63/77 passing cases**, with **31/40 profiles** passing every attempted
case. Four wide GPT-2/Llama cases still fail logits; ten other cases fail only
the stricter raw feature-adjoint checks. The 14 cases remain red in their
receipt. BART/T5/Switch and the dedicated V4.1 graph still need bridge integration;
other actual modality stacks remain pending.

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

1. Close the 33 recorded numerical failures without changing the frozen inputs,
   expectations or strict ceiling. Trace wider first-forward GEMMs separately
   from accumulated backbone updates and embedding adjoints. Add dedicated
   encoder-decoder entry points for BART/T5/Switch and the separate V4.1 graph.
2. Implement and independently qualify each actual modality stack. Start with
   Gemma3's image encoder/projector and Phi-4 image/audio branches, then Qwen
   vision/video position handling, Kimi's MoonViT/projectors and the remaining
   modality-capable package contracts. Text-only models can consume an explicit
   external-encoder adapter; they do not acquire an upstream multimodal model
   contract by alias.
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
```
