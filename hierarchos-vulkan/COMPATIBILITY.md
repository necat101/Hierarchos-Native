# Compatibility and validation

**Current-revision hardware scope:** RDNA3 is validated for DeepSeek V4.1 Flash
and the recorded parameter/MoE streaming checks only. Earlier broad Intel/Ally
results in this document are historical; they do not extend current-revision
RDNA3 coverage to every architecture. Gen9 qualification of V4.1/streaming is
pending. See [current validation status](../VALIDATION_STATUS.md).

## Strict base to PEFT follow-up inventory

This table is the machine-readable inventory consumed by
`validation/verify_peft_green_matrix.py`. Add a row whenever a new family passes
the native base gate; list every independently qualified graph fixture. A new
row automatically becomes required PEFT follow-up work. A passing base row
never promotes the independent PEFT result. On 2026-10-05 the complete matrix was
re-run against the current local checkout (Transformers `5.19.0.dev0`, commit
`469230357aab`) and the documented oracle stack (`peft==0.18.0`, torch
`2.14.1`); provenance with unchanged source/shader/binary/oracle hashes
(`inputs_unchanged=true`) is
`.peft-oracle-fixtures/green-matrix-report-provenance.json`. The rerun is green
at the unchanged absolute `max_abs <= 2e-7` gate for **32/32 fixtures on
ordinary LoRA and named-adapter switching** and for **25/32 fixtures on the
saved-module stage**. The seven red saved fixtures (nine module rows) are
native-side, not oracle-side: every failing value is a 1-2 ulp difference in the
adjoint presented to a shared head module (input embedding, input norm or
`lm_head`) and then amplified by the RMSNorm backward. That attribution is
measured, not assumed. Pinning the 2026-09-30 checkout (`transformers` tag
`v5.16.0`) with `HIERARCHOS_TRANSFORMERS_ROOT` regenerates a byte-identical
fixture (same base weights, adapter and input fixture hashes) and reproduces
Gemma3's saved-embedding gradient bit-for-bit (`4.768371582e-7` / `2.384185791e-7`),
so the current `5.19.0.dev0` checkout is excluded as a cause; the per-module
values are listed under "Saved-module stage status" below. See
[the full audit](../PROGRESS_PEFT_AUDIT.md)
for evidence and the [PEFT usage guide](../hierarchos-native-cli/README.md#peft-fine-tuning-guide)
for training, saved modules, resume, inference and merge commands. This is
FP32 tiny-model text-fixture qualification, not universal PEFT or production-size
checkpoint/device certification.

<!-- peft-native-green:start -->
| Architecture | Native model_type | Required PEFT fixtures | Base ceiling |
| --- | --- | --- | --- |
| GPT-2 | gpt2 | gpt2 | 2e-7 |
| Llama | llama | llama | 2e-7 |
| Mixtral | mixtral | mixtral | 2e-7 |
| Qwen3-Next | qwen3_next | qwen3_next | 2e-7 |
| DeepSeek V4 | deepseek_v4 | deepseek_v4 | 2e-7 |
| DeepSeek V4.1 Flash text | deepseek_v41 | deepseek_v41_scalable | 2e-7 |
| Phi-4 Multimodal text | phi4_multimodal | phi4_multimodal_text | 2e-7 |
| Phi-3 | phi3 | phi3 | 2e-7 |
| Kimi K2.5 text | kimi_k25 | kimi_k25_text | 2e-7 |
| Kimi K3 / KimiLinear text | kimi_linear | kimi_k3_kda, kimi_k3_mla, kimi_k3_text | 2e-7 |
| GPT-OSS | gpt_oss | gpt_oss | 2e-7 |
| SmolLM3 | smollm3 | smollm3 | 2e-7 |
| Qwen2.5 text | qwen2 | qwen2_5_gqa, qwen2_5_sliding_tied | 2e-7 |
| Qwen3.5 | qwen3_5, qwen3_5_moe | qwen3_5_full, qwen3_5_linear, qwen3_5_mixed, qwen3_5_moe | 2e-7 |
| Qwen4 Experimental | qwen4_exp | qwen4_exp_linear, qwen4_exp_qsa, qwen4_exp_ple, qwen4_exp_mixed_ple | 2e-7 |
| Mistral 4 | mistral4 | mistral4 | 2e-7 |
| MiniMax M3 text | minimax_m3_vl_text | minimax_m3_dense, minimax_m3 | 2e-7 |
| Gemma 4 | gemma4_text | gemma4 | 2e-7 |
| MiniMax M2 | minimax_m2 | minimax_m2 | 2e-7 |
| Gemma 3 text | gemma3_text | gemma3 | 2e-7 |
| Falcon H1/H1R | falcon_h1 | falcon_h1_gated_after, falcon_h1_gated_before, falcon_h1_silu | 2e-7 |
<!-- peft-native-green:end -->

### Saved-module stage status

Full matrix (`verify_peft_green_matrix.py --stage all --jobs 2`) against the
documented oracle pin: **32/32 fixtures pass ordinary LoRA, 32/32 pass
named-adapter switching, 32/32 pass the saved-module stage.**

The saved-module reports read at the start of this work item listed 11 red rows
(deepseek_v4, gemma3, gemma4, llama, minimax_m2, minimax_m3, mistral4,
mixtral, phi4_multimodal_text, qwen3_5_full, smollm3). Four of them are now
green: **llama, deepseek_v4, mixtral and phi4_multimodal_text**. The measured
root causes were native kernel defects, not oracle FP32 execution topology
(see `PROGRESS_PEFT_AUDIT.md`, checkpoint 2026-10-05 later):

* `shaders/silu_forward.comp` computed `x * (1/(1+exp(-x)))` with the driver's
  relaxed `exp()`, while ATen/HF compute `x / (1 + exp(-x))`. With the SLEEF
  exponential and the correctly rounded `fp32_div`, the module is bit-exact
  (0/128 on the llama layer-0 gate values, was 44/128).
* plain-RoPE layers now consume host-materialized FP32 inverse frequencies
  instead of the in-shader `pow` path.
* both attention shaders use the correctly rounded divider for the softmax
  reciprocal (`1/exp_sum` was one ulp high).
* the PEFT cross-entropy adjacency (`falcon_h1_cross_entropy.spv`, built with
  `HIERARCHOS_LOG_SOFTMAX_GRAD=1`) folded the AVX512-sized 16-float lanes as a
  lane-order sequential sum; PyTorch reduces them as a balanced binary tree.
  The sequential fold put 31 of 128 adjoint elements one ulp off, which the
  final RMSNorm backward amplified past the gate.

The remaining seven fixtures (nine module rows) were red at that checkpoint and
each failing value was a 1-2 ulp difference in the adjoint presented to a shared
head (embedding, input norm or `lm_head`), amplified by the RMSNorm backward
(`rstd` ~41.7) so the absolute drift landed just above the gate. They are now
**resolved**: the 2026-10-07 full matrix (`started_utc
2026-10-07T05:40:52Z`, `inputs_unchanged: true`) aggregates to **32/32 saved,
32/32 LoRA and 32/32 switch**, and every row below is under the unchanged `2e-7`
gate. No tolerance or fixture was changed, and `frozen base` stays `0.0` for
every row.

| Fixture | Formerly failing saved module | Then (A / B) | Adapter A now | Adapter B now |
| --- | --- | ---: | ---: | ---: |
| gemma3 | `model.embed_tokens` | `4.768371582e-7 / 2.384185791e-7` | `7.450580597e-9` | `7.450580597e-9` |
| gemma4 | `model.layers.0.input_layernorm` | `7.152557373e-7 / 1.192092896e-6` | `3.725290298e-9` | `7.450580597e-9` |
| gemma4 | `lm_head` | `1.192092896e-7 / 2.384185791e-7` | `7.450580597e-9` | `5.960464478e-8` |
| gemma4 | `model.embed_tokens` | `7.152557373e-7 / 4.768371582e-7` | `7.450580597e-9` | `7.450580597e-9` |
| minimax_m2 | `model.embed_tokens` | `3.576278687e-7 / 1.192092896e-7` | `1.192092896e-7` | `1.192092896e-7` |
| minimax_m3 | `model.embed_tokens` | `1.788139343e-7 / 2.086162567e-7` | `1.192092896e-7` | `1.043081284e-7` |
| mistral4 | `model.embed_tokens` | `2.384185791e-7 / 1.788139343e-7` | `1.490116119e-7` | `5.960464478e-8` |
| qwen3_5_full | `model.embed_tokens` | `2.384185791e-7 / 5.960464478e-8` | `1.192092896e-7` | `1.788139343e-7` |
| smollm3 | `model.embed_tokens` | `2.980232239e-7 / 1.788139343e-7` | `1.490116119e-7` | `1.192092896e-7` |

For reference, llama's saved embedding gradient moved from `2.384185791e-7`
(adapter A and B) to `1.192092896e-7` / `5.960464478e-8`, with `frozen base`
still exactly `0.0`; the remaining llama boundary is a single `2^-24` step inside
the layer-1 backward.

HF PEFT rejects Falcon H1 Mamba `out_proj` and `conv1d`; the interoperability
fixture targets all other linears, including Mamba `in_proj`. MiniMax M3 indexer
linears and Qwen4 QSA selectors have no gradient through discrete top-k selection
and are excluded from ordinary differentiable LoRA fixtures. These exclusions
do not excuse missing validation of the remaining modules or lifecycle stages.

This is a shared native Rust/Vulkan backend for Hierarchos and selected Hugging
Face Transformer graphs. Universal feature parity with the Python Transformers
library has **not** been reached. Model-type registration, task implementation,
and numerical validation are separate milestones.

## Headline: fourteen strict two-step AdamW checks

The strongest compatibility claim in this backend is intentionally small and
reproducible: fourteen current/high-value Transformer text architectures/families have live Vulkan
forward parity and pass **two full AdamW steps** against the local Hugging Face
Transformers source tree, with cross-entropy loss checked at each step and every
named trainable parameter compared after export. Falcon H1/H1R (parallel
Attention+Mamba2) is at the top of that list.

Verified on AMD Radeon Graphics against local Transformers `5.16.0.dev0`:

| Family | Verified scope | Max abs parameter error after 2 AdamW steps |
| --- | --- | ---: |
| Falcon H1 / H1R | parallel Attention+Mamba2 causal LM | `1.192092896e-7` |
| DeepSeek V4 | causal LM | `1.192092896e-7` |
| Phi-4 Multimodal | text backbone / causal LM | `1.192092896e-7` |
| Phi-3 | causal LM | `1.192092896e-7` |
| Kimi K2.5 | text backbone / causal LM | `1.192092896e-7` |
| Kimi K3 / KimiLinear | text backbone / causal LM | `5.963374861e-8` |
| GPT-OSS | causal LM | `1.192092896e-7` |
| SmolLM3 | causal LM; mixed RoPE/NoPE and YaRN | `1.192092896e-7` |
| Qwen3.5 | dense / DeltaNet / hybrid / MoE text causal LM | `2.980232239e-8` |
| Qwen4 Experimental | DeltaNet / sparse QSA / GR+PLE / hybrid / MoE text causal LM | `2.980232239e-8` |
| Mistral 4 | causal LM | `2.607703209e-8` |
| MiniMax M3 | text backbone / causal LM | `3.539025784e-8` |
| Gemma 4 | causal LM | `1.220032573e-7` |
| MiniMax M2 | causal LM | `1.192092896e-7` |

The acceptance ceiling is `2e-7`; it is enforced by `verify_hf_training.py` and
`verify_hf_logits.py`, not rounded into the documentation after the fact. On
2026-10-05 the entire forward registry (37 families: this headline set, the
Kimi K3 oracle family, and the previously broad/legacy set) was re-qualified
under a universal `atol=2e-7, rtol=0` clamp that the harness now applies to
every registered family; callers may tighten it but can never loosen it. The
default `verify_hf_logits.py` invocation ran all 71 unmasked and mixed-padding
comparisons with exit 0 and observed at most `7.450580597e-8` maximum logit
error (Kimi K2.5 text). T5 and Switch Transformers use a re-conditioned tiny
fixture (`initializer_factor=0.25`): at the paper init the oracle's own
fp32-vs-fp64 logit noise (`4.8e-7`/`5.7e-7`) exceeded the ceiling, so the gate
would have measured oracle rounding rather than native fidelity. Two-step AdamW
parameter parity is green at the same gate for every decoder-only fixture
family (34/34; worst observed `1.192092896e-7`).

On 2026-10-17 two gate-width forward fixtures were added, because every earlier
fixture is smaller than the Intel Gen9 matmul gate (`rows >= 4`, `input_dim >=
32`, `output_dim >= 128`) and therefore never dispatched the tuned modules it
was supposed to check. `gpt2_wide` and `llama_wide` use `hidden_size` 64,
`intermediate_size` 256 and `vocab_size` 320 with a 2x6 token batch, so their
attention, MLP and LM-head projections are admitted; the bias and no-bias
variants cover `linear-bias-forward` and `linear-forward`. The default registry
now runs 75 comparisons over 39 families, all at `atol=2e-7, rtol=0`, and the
widest observed drift is `1.788139343e-7` on those two fixtures -- inside the
ceiling, but the tightest margin in the set, because at 64-wide reductions the
oracle's own FP32 summation order is the dominant term. Those two rows also
prove at runtime that the tuned modules ran: the probe prints its per-family
dispatch counter under `HIERARCHOS_VULKAN_VARIANT_TRACE` and the harness fails
if the expected family is missing (`linear-bias-forward` 4 dispatches for
`gpt2_wide`, `linear-forward` 5 for `llama_wide`), because the geometry gate can
reject a dispatch the plan selected. The rule is two-sided: a family the plan
resolves to `portable` must be dispatched zero times, which is also how the
forced-portable run is checked and how the assertion stays valid on an adapter
without that module.

Run exactly the claim above with:

```powershell
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_logits.py --headline-strict
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_training.py --headline-strict
```

The training fixture uses AdamW with learning rate `0.001`, betas
`(0.9, 0.999)`, epsilon `0.001`, zero weight decay, and two optimizer steps.
These are deterministic FP32 tiny-model correctness checks, not throughput
benchmarks or certification of arbitrary production-size checkpoints.

### Additional verified and frontier families

Falcon H1 / Falcon H1R (`falcon_h1`) has a dedicated native parallel
Attention+Mamba2 graph. Three deterministic variants cover gated norm before/after
the SiLU gate, no gated norm, GQA/RoPE, non-unit MuP multipliers, padding,
chunk-aligned/nonaligned prefill, recurrent KV+conv+SSM decode, native backward,
two AdamW trajectories and SafeTensors reload. The strict absolute `2e-7` checks
are in `validation/verify_falcon_h1.py`; the audit is
[`PROGRESS_FALCON_H1_AUDIT.md`](../PROGRESS_FALCON_H1_AUDIT.md).
Measured forward drift is at most `8.940696716e-8`, cached decode across prefill
lengths 1/3/4/7/9 at most `7.450580597e-8`, and two-step parameter drift at most
`1.192092896e-7`. Cached native logits match full-prefix native logits at every
checked token; the harness also requires actual recurrent dispatches. The second
AdamW trajectory runs with trailing padding and its padded labels masked; a
mid-sequence mask gap is not scored because the reference's own fp32-vs-fp64
backward spread there is `4.209578e-7`, above the gate, and the harness comment
keeps those measurements as the reason.
Full-parameter training/fine-tuning uses the CLI and GUI. Falcon H1/H1R LoRA
and registered saved modules also pass all three PEFT fixtures; HF interoperability
excludes Mamba `out_proj` and `conv1d`, but includes Mamba `in_proj`. Official H1R-7B config/shard metadata was checked;
the full 7B weights were not run, so this qualification is limited to the fixtures.

The validator also carries additional verified and in-progress fixtures rather
than inflating the headline number with registry aliases:

- Gemma 3 remains source-backed and passes its two-step AdamW check
  (`1.797452569e-7` observed maximum parameter error), but the headline slot now
  prioritizes Gemma 4.
- SmolLM3 text causal-LM support is strict-parity verified with GQA, RMSNorm,
  SiLU/SwiGLU, tied embeddings, config-driven mixed RoPE/NoPE layers, sliding
  attention on configured NoPE layers, padding masks, and YaRN. Default and
  YaRN forward fixtures observed at most `5.960464478e-8` absolute logit error;
  four-step cached generation observed at most `5.960464478e-8` versus
  Transformers and exactly `0.0` versus native full-prefix decoding. Both
  variants reached `1.192092896e-7` maximum named-parameter drift after two
  AdamW steps. Native-trained SafeTensors reloads observed at most
  `5.960464478e-8` logit drift and preserve `no_rope_layers`, `layer_types`,
  RoPE parameters, sliding-window settings, and tied embeddings. The dedicated
  checks are `verify_smollm3_generation.py` and `verify_smollm3_roundtrip.py`.
- Qwen2.5 text checkpoints that use the standard Hugging Face `qwen2` model
  type reuse the native Qwen2 graph instead of a second compatibility graph.
  Strict fixtures cover 4-query/2-KV-head GQA, Qwen2.5-style
  `rope_theta=1_000_000`, mixed full/sliding attention, padding, and tied and
  untied LM heads. Forward logits observed at most `5.960464478e-8`; four-step
  cached generation observed at most `4.470348358e-8` versus Transformers and
  exactly `0.0` versus native full-prefix decoding; two-step AdamW parameter
  drift observed at most `1.192092896e-7`. Native-trained SafeTensors reloaded
  by Transformers and Vulkan observed at most `5.960464478e-8` logit drift.
  Qwen2.5-VL/Omni vision/audio execution is separate work.
- Qwen3.5 text causal-LM support is strict-parity verified across full-attention,
  pure Gated DeltaNet, the default hybrid DeltaNet/full-attention schedule, and
  Qwen3.5-MoE. Unmasked and mixed-padding logits observed at most
  `4.470348358e-8` absolute error; three-step cached generation observed at most
  `4.470348358e-8` versus Transformers and exactly `0.0` versus native
  full-prefix decoding; two-step AdamW observed at most `2.980232239e-8`
  parameter drift. A trained native SafeTensors export reloaded by both
  Transformers and Vulkan observed at most `5.215406418e-8` logit drift on the
  exact same checkpoint. These checks are enforced at `2e-7` absolute-only by
  `verify_hf_logits.py`, `verify_hf_training.py`,
  `verify_qwen35_generation.py`, and `verify_qwen35_roundtrip.py`.
- Qwen4 Experimental text support is strict-parity verified across Gated
  DeltaNet, genuinely sparse QSA, GR/hyper-connections, PLE n-gram and dilated
  convolution state, mixed DeltaNet/QSA schedules, and MoE/shared experts.
  Forward logits observed at most `2.980232239e-8` absolute error; four-step
  cached generation observed at most `3.725290298e-8` versus Transformers and
  exactly `0.0` versus native full-prefix decoding; two-step AdamW observed at
  most `2.980232239e-8` parameter drift. Native-trained SafeTensors reloaded by
  Transformers and Vulkan observed at most `2.235174179e-8` logit drift on the
  same checkpoint under the fixed `2e-7` absolute-only gate.
- Kimi K3 is validated against Moonshot's released `Kimi-K3` custom modeling
  source while still using the checked local Transformers runtime. The native
  production path remains Rust/Vulkan-only; Moonshot/Python code is an oracle,
  not a runtime dependency.
- Kimi K3 support is specifically the `KimiLinear` text backbone. Unquantized
  FP32/BF16 packages are supported. The official `compressed-tensors`
  `mxfp4-pack-quantized` checkpoint representation is rejected explicitly until
  that packed representation has a separately verified native implementation.
  MoonViT/vision/projector execution is likewise outside this text-backbone
  claim; unrelated composite tensors are preserved on package export.

### DeepSeek V4.1 Flash release parity gate

**Qualified on the AMD Ryzen Z1 Extreme / Radeon Graphics machine using the
accepted scalable-checkpoint criterion.** The normal package loader freshly
passes the small 40-layer, 384-expert/top-6 graph: 3,216 complete logits across
six prefill/decode/reset steps have zero CPU drift, with exact routing checks.
Its released compression ratios, source ownership, Engram placement/four-gram
features, candidate block size and routing density are asserted against the
Flash-1017 config. Production arithmetic widths are covered by the recorded
width probes below. Full production-weight execution is not a qualification
prerequisite. This qualifies native causal text inference; PEFT remains an
independent follow-up and is not promoted by the base result.

The durable [qualification record](validation/release_parity/deepseek_v41/qualification.json)
binds the current runtime/shaders, device/driver, CPU fixtures and every measured
logit. Its compressed native-output bundle can be rechecked after generated
large checkpoints are removed. The retained scalable package is about 45 MB;
it is the baseline for the forthcoming Intel Gen9 kernel port.

DeepSeek V4.1 Flash (`deepseek_v41`) has a dedicated Rust/Vulkan graph rather
than a generic Transformer package. Its release gate is a **production-width**
configuration that exercises inter-layer accumulation at released geometry: the
released hidden width 5120, 64x512 attention heads, Q/O ranks 1280/1024, eight
output groups, 2304-wide experts, HC 4 and the full 129,280-token vocabulary,
in twenty-four layers with two routed experts and two single-token steps.
Crucially, the package is loaded through the shipping default
`DeepseekV41CausalLM::from_hf_package`, not a hand-selected residency
constructor; that is the same loader `transformer-generate` uses.

The gate is the unchanged absolute ceiling `<2e-7`, `rtol=0`, compared against
the pinned single-threaded CPU PyTorch oracle. Measured results:

- **258,560 production-width logits, zero drift** (prefill and decode).
- **904,960 schedule-width logits, zero drift** across five prefill/decode/reset
  steps through the shipping loader (1331.88 s on AMD Radeon Graphics).
- Aggregate V4.1 recheck: **18 models, 108 steps, 1,472,649 logits**, worst drift
  `1.7881393432617188e-7`, still strictly inside the gate.
- Production checkpoint: 339 stored tensors plus 312 packed FP8/FP4 tensors,
  with the twenty shard hashes pinned in the fixture.
- Additive schedule package: eight shards, 75 stored tensors plus 107 packed
  FP8/FP4 matrices, 4.26 GB including metadata; the original package is unchanged.

Vocabulary residency is tiered, so the release path never holds both 1.32 GB
tables. The embedding gathers only the tokens of the call (uniqued, remapped
onto a compact table) and the head is projected in 8,192-row tiles read from
the package, so at most one 84 MB BF16 tile is on the device at a time and no
full-table `Vec` is built per operation. `Resident` is honored only when both
tables fit beside the already-loaded backbone, measured from the driver budget
with a 512 MiB reserve; otherwise the loader stages them before the allocation
that would fail, and the effective tier is reported by the model.

With that in place the production-width gate passes for the shipping loader and
five residency combinations -- default, streamed routed experts, streamed all
experts, streamed block projections and streamed vocabulary -- at **1,293,824
logits, zero drift**, `<2e-7`, `rtol=0`, plus the separate PEFT gate. The
twenty-four-layer fully resident variant stays unqualified on this device by
capacity: it needs about 6.97 GB of projections and experts before the
vocabulary is touched, against 7.11 GB available, and the driver refuses the
allocation while building layer 22.

The original accumulation case keeps twenty-four layers, two routed experts,
two single-token steps and no compression/index/Engram schedule. An additive
four-layer case checks 904,960 full-vocabulary logits with ratios `[2,2,1,1]`,
shared KV/index state, candidate filtering, Engram at 8x256/four-gram width,
six selected experts, masked history, cached decode and a shorter reset. It
uses the same shipping loader with no execution-policy flags. Its depth, table
budgets, expert payload count and context are explicitly reduced.
Twenty-four layers repeat the residual accumulation the released 40-layer stack
performs and extend past the released schedule's ratio and source transitions
(ratio 2 from layer 2, ratio 1 from layer 20, KV/index sources at 2/8/14/20),
rather than checking a single inter-layer hand-off. The remaining axes are
qualified in
composable probes rather than this graph: the 40-layer released schedule and
384-expert top-6 routing over hidden width 32, and the 2,048-token prefill at
the released 32x128 index heads. The router itself now also passes bit-exact
CPU checks over all 384 candidates at hidden width 5120, top-6 and 1/3/4 rows.
The full expert payload would need 6.8 GB of FP4 weights per layer (163 GB at
twenty-four layers), so payload cardinality and schedule depth remain checked
in reduced-width graphs. Full
510 GB geometry, million-token contexts, training, DSpark and vision remain
unqualified. This is the documented basis for scaling the qualified production
width to the released depth and expert count. This is a forward-logit gate;
the training state that has been qualified separately is described under
*DeepSeek V4.1 adapter training state* below. The generic `verify_hf_logits.py`
probe does not host the dedicated V4.1 graph, so this dedicated case is the
equivalent release check.

### DeepSeek V4.1 PEFT-LoRA

Additive unmerged PEFT-LoRA is qualified separately from the base gate. The
released package is loaded unchanged; each targeted projection computes
`base(x) + (alpha / r) * B(A(x))` in FP32 and rounds once at the existing stream
boundary. The gate is a 247 MB production-width package (5120 hidden, 64x512
heads, Q/O ranks 1280/1024, eight groups, 2304-wide FP4 experts, HC 4) with rank
16 adapters on packed FP8 `attn.wq_a/wq_b/wkv/wo_b`, the packed FP8 shared
expert and routed FP4 expert `0.w2`:

- **1,024 logits, zero drift** for both the base and adapted graphs against the
  frozen CPU PyTorch oracle, `<2e-7`, `rtol=0`; the adapter moves the logits by
  up to `2.543889`, so a silently ignored adapter cannot pass.
- 127 default V4.1 tests, 40 CLI tests and 517 reproduced shader artifacts pass;
  the PEFT gate itself runs opt-in because it loads the package twice.

Supported: additive LoRA, rsLoRA, rank/alpha patterns, BF16/F16 adapter storage,
both `attn` and `self_attn` HF spellings, adapter export/reload (factors,
patterns and rsLoRA survive the round trip; half-precision storage decodes to
the resident values), and targets on attention `wq_a/wq_b`
`wkv/wo_b`, indexer `wq_b/weights_proj/wk`, and routed/shared expert
`w1/w2/w3`. Fail-closed: `modules_to_save`, DoRA, QALoRA, adapter bias,
`target_parameters`, trainable tokens, ALoRA, arrow, layer replication,
grouped `wo_a`, compressor and Engram targets, F32-storage packages and
`merge-lora` for V4.1 packages. Native adapter training uses the same additive
LoRA/rsLoRA contract with zero dropout and is enabled through
`transformer-finetune`; its qualification is
*DeepSeek V4.1 full-model PEFT training qualification* below.
Multi-token prefill at released width is qualified too: the same
released-width package runs a four-token prefill and the decode that reads the
ring it seeded, `2,560` logits at zero drift through the default loader. That
case pins the CPU bmm's multi-row grouped body (contiguous 1024-wide K blocks,
odd-element-first pair folding), which differs from the single-row body at
BF16 rounding ties. See [the V4.1 audit](../PROGRESS_DEEPSEEK_V41_AUDIT.md).

The optional schedule trace compares 6,497,008 intermediate values. Fourteen
grouped BF16 projection values differ from the CPU trace (largest difference
0.015625); subsequent FP8 activation quantization removes those differences
in this case, and every final logit is identical. Thus the full-logit gate
passes, while complete intermediate bit parity at all production shapes
remains unqualified. These finite probes do not certify joint full-size execution.

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_scalable_release_strict_all_logits -- --ignored --nocapture --test-threads=1
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/qualify_deepseek_v41_scalable.py
```

The large packages were removed after recording their passing qualification.
Their frozen CPU expectations and shard hashes remain; regenerate the packages
only when intentionally repeating these older width gates:

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_production_width_strict_all_logits -- --ignored --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_schedule_width_strict_all_logits -- --ignored --nocapture --test-threads=1
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/summarize_deepseek_v41_qualification.py --output target/deepseek-v41-reference/release-summary.json
```

### DeepSeek V4.1 adapter training state

The loss-to-adapter graph is wired end to end and enabled through the normal
CLI (`train`, `resume`, `export`). LoRA and rsLoRA modules train with AdamW,
save/resume bit-exactly, and export as standard PEFT directories. The committed
release gate replays **four CPU PyTorch oracles x LoRA/rsLoRA x three steps**
with every factor and optimizer moment pinned to the frozen reference state, so
each step is qualified on its own arithmetic:

- every logit strictly `< 2e-7`, `rtol = 0`; all six production-ratio steps
  (256 logit values each) are **bit exact**, each step's 892,684 factor updates
  are bit exact to the documented pinned single-tensor AdamW policy, and its
  gradients stay within `8.4e-9`;
- every factor gradient and optimizer moment `< 2e-7`;
- the fixture residual stays inside the exactly computed policy response to the
  gated gradient difference, so AdamW's `eps` amplification is measured, not
  assumed away.

The most recent correction is the indexer's released dtype boundary, which the
CPU BF16 mirror had been running one rounding wider. The released indexer runs
on bf16 tensors, so its einsum output, its head-weighted product and its
head-sum are each rounded to bf16 before the two-level top-k. On the
production-ratio trajectory those roundings make compressed positions 1 and 4
exactly equal at the step-2 decode, which decides the selection; the fp32 mirror
kept a `3.5e-4` gap and therefore selected a different position, and that
discrete flip - not any arithmetic drift - was the `4.9e-2` decode logit
difference this gate found. The native side was already bit-exact to the
released bf16 scores at every sampled layer and position, so the fix is in
`validation/validate_deepseek_v41_full_bf16_training.py`, which now mirrors the
three roundings and is frozen as
`fixtures/deepseek_v41_production_ratio_training_v2.json` and
`fixtures/deepseek_v41_full_bf16_training_v3.json`; the superseded fixtures are
listed as rejected evidence in the training receipt. No tolerance, `rtol`,
threshold or native kernel changed.

Landed, in the graph's own arithmetic:

- `terminal.rs` implements the loss-side chain: cross entropy whose loss
  staging uses the host's 8/16-lane log-softmax map-reduce, the frozen
  vocabulary head's input gradient, and the final
  RMSNorm input gradient, which recomputes the forward `rstd` with the same
  pinned reducer and rsqrt instead of trusting a side buffer.
- Gated against CPU PyTorch autograd under `<2e-7`, `rtol=0`: row loss and mean
  loss **zero drift**, `grad_logits` `2.98e-8`, `grad_hidden` `5.96e-8`,
  `grad_x` `1.19e-7`.
- `train.rs` owns mutable adapter factor masters, both AdamW moments per factor
  and the step count. The update follows `torch.optim.AdamW`'s single-tensor CPU
  path operation by operation. Three pinned steps over two modules stay within
  `5.96e-8` of PyTorch for every factor and every moment, and a
  export/save/resume run reproduces the uninterrupted trajectory bit for bit.
  Export reuses the normal adapter writer, so a trained adapter reloads and
  validates as a standard PEFT directory. The default `step` skips omitted
  targets as `grad=None`; `step_complete` retains the explicit complete-cohort
  check for dense callers.
- `backbone.rs` adds the first block stage: `DeepseekV41BlockAdjoints` carries
  the Hyper-Connection post adjoint (sub-layer output, residual stream, `post`
  and `comb`) and the FP32 MoE chain (combine, routed SwiGLU, shared SwiGLU and
  gate). Each stage has its own CPU PyTorch gradient fixture; the gate's
  normalization factor and selected scores are recomputed with the forward's
  own staging rather than cached, so they cannot disagree with the routing that
  produced the indices.
- Gated under `<2e-7`, `rtol=0`: the Hyper-Connection post adjoint is **bit
  exact** over all 132 values, including `grad_comb`, where the released
  `sum(dim=2)` contracts comb's first copy axis. The five MoE cases cover
  `softmax`/`sigmoid`/`sqrtsoftplus`, top-k normalization on and off, active
  clamp limits, the vector and matrix dot paths, and a 12-expert router.
  2,360 values: routing weights and the combine adjoint **zero drift**,
  `grad_gate_x` `2.98e-8`, expert `grad_x` `5.96e-8`, routing-weight gradients
  `3.73e-8`, shared expert `5.96e-8`, and the chained MoE input gradient
  `1.19e-7`.
- `DeepseekV41BlockAdjoints` now also carries the FFN RMSNorm input gradient
  (the terminal kernel, reused in the FFN position), the Hyper-Connection
  collapse (`grad_stream`, `grad_pre`), the coefficient split's Sinkhorn and
  sigmoid adjoint (`grad_mix`) and the frozen `hc_fn` mix projection's input
  adjoint. The split inverts every normalization step in reverse order, applies
  the softmax Jacobian at the softmax output rather than at `softmax + eps`, and
  differentiates the `pre`/`post` sigmoids through their arguments; the
  projection recomputes `rstd` with the pinned reducer and walks the
  mean-square chain in autograd's order.
- Gated under `<2e-7`, `rtol=0`: the FFN norm and the collapse are **bit
  exact** (32 and 108 values), the split's `grad_mix` is `3.54e-8` over 172
  values across three Sinkhorn cases (`hc=3`, 3 iterations; `hc=2`, one
  iteration; `hc=4` with `eps=1e-3`), and the mix projection's `grad_stream` is
  `1.19e-7` over 208 values. The gate caught two real defects: the softmax
  Jacobian taken at `softmax + eps` (1.71e-6), and a workgroup/row mismatch in
  the projection that returned all zeros.

The existing eleven training-stage kernels are the two terminal ones plus
`transformer_deepseek_v41_hc_post_{input,comb}_grad`,
`transformer_deepseek_v41_hc_{pre_input,split,mixes_input}_grad` and
`transformer_deepseek_v41_moe_{combine,expert,gate,input_grad_combine}_grad`;
all rebuild byte-for-byte under `glslang` 16.6.0, which the provenance check
covers with the other recorded artifacts. The attention phase adds seven
artifacts; the projection phase adds two, the compressor-backward phase two,
Engram two and the decode phase two, bringing the manifest to **546**.

- `attention_grad.rs` adds `DeepseekV41AttentionAdjoints`: the gathered sparse
  attention VJP through both the key and value uses of the same KV tensor,
  deterministic duplicate-index scatter to window/compressed caches, the
  frozen grouped `wo_a` input adjoint, and cache overwrite adjoints for both
  source rows and previous cache state. Discrete indices and the sink are
  frozen. RoPE's input adjoint reuses the conjugate rotation with the identical
  head/batch/position mapping; channels outside its tail pass through.
- A new CPU PyTorch autograd fixture checks **152,210 gradient values** and
  **66,608 forward values** under `<2e-7`, `rtol=0`. Sparse attention's worst
  gradient drift is `5.96e-8`, grouped output `1.79e-7`, and RoPE/cache writes
  are bit exact. The sparse case with 64 heads and 512 channels is bit exact;
  this is an FP32 attention-component gate. Grouped projections are exercised
  at fixture widths, and packed/BF16 training is not covered by this gate.
- A separate identity-contraction test crosses the 65,535-workgroup dispatch
  axis limit with **4,194,257 values**, using the second dispatch axis. The
  frozen weight's storage size is validated independently of dispatch shape.

- `linear_grad.rs` adds immutable FP32 input/factor tapes and LoRA/rsLoRA
  projection VJPs. It reuses the actual mixed-linear forward. A tape retains
  no frozen base matrix: the residency test drops the original GPU weight,
  mutates the original caller input/factors, then restages a base without an
  adapter and still reproduces the original evaluation's gradients.
- A query-projection subgraph (`wq_a -> RMSNorm -> wq_b`) now computes native
  cross-entropy gradients through a frozen classification head, updates both
  adapters with AdamW, and saves/resumes/exports. LoRA and rsLoRA each run three
  real gradient steps. Together with the standalone projection and loss-fold
  regressions: **11,260** checked numerical values; worst drift **2.98e-8**
  under `<2e-7`, `rtol=0`. Updated logits
  stay within **2.98e-8**, factor gradients within **1.12e-8**, factors within
  **3.73e-9**, and moments within **1.40e-9**. Resume factors, moments and
  logits are compared as FP32 bits. This is a projection-subgraph gate with
  zero dropout; BF16/packed projection training is qualified end to end by the
  full-model gate below.
- `compressor_grad.rs` adds `DeepseekV41CompressorAdjoints::prefill`, the
  FP32 prefill adjoint of the released compressor chain: the ratio-2
  softmax-pool VJP (`transformer_deepseek_v41_compress_grad.comp`) and the
  pool-state routing (`transformer_deepseek_v41_compress_state_grad.comp`),
  reusing the input-gradient, norm and vector kernels the other adjoints
  already use. A CPU PyTorch autograd fixture checks **8,992 frozen
  gradient values** across 10 cases plus **6,784 values** re-driven through
  the native forward, worst drift **1.79e-7** under `<2e-7`, `rtol=0`
  (retained-operand drift 3.73e-9). Decode evaluations (`start_pos != 0`)
  are rejected, not approximated. The shared input-gradient kernel is not
  ATen-order-identical at head 512, so two fixture cases record smaller
  seed scales and the limit stays recorded in the fixture's
  `reduction_note`.
- `compressor_grad.rs` also adds `DeepseekV41CompressorAdjoints::decode`: one
  cached decode step, with the pool VJP and state routing
  (`transformer_deepseek_v41_compress_decode_grad.comp`). The written slot
  returns zero to the previous state, the untouched slot passes the incoming
  gradient plus its pool term, and a non-completing step sends the incoming
  gradient to the row it wrote. Replaying a sequence backwards and feeding each
  returned state gradient into the step before it is the cross-step chain. Ratio
  1 rejects state operands and position zero rejects as a prefill. A CPU
  PyTorch autograd fixture checks **6,808 frozen gradients** over ten chains and
  **32 decode steps** (ratio 1/2, prefill 1/2, pending and completing steps,
  eight-step chains, unseeded tails, batch 2, head 4..128, hidden 8..512):
  both the CPU-operand chain and the native-operand replay now have **zero
  drift** under `<2e-7`, `rtol=0`. A measured follow-up closed the former
  `4.77e-7` native replay gap: projections were already bit exact; the pool
  used vector SLEEF exp for CPU scalar-tail channels. Forward, prefill backward
  and decode backward now share the host SIMD-width/scalar-exp split. All
  6,808 gradients in each replay and 8,684 forward/state operands are gated.
  Twelve widths around the SIMD cutoffs add **4,640 bit-exact** pool/VJP values
  (widths 1/3/4/7/8/15/16/17/31/32/33/65). No oracle operands or tolerance
  changed. Both decode operand types are exported for model-tape callers.
- Decode rows change ATen's GEMM reduction order: one row walks eight-element
  partials with separately rounded products, two rows use four-product groups.
  The decode adjoint routes to `linear_input_grad_block8.spv` (new),
  `linear_input_grad_group4_muladd.spv` (existing) or the serial kernel for
  wider rows. Without the routing the same operands measured `1.67e-6` at
  head 128 / hidden 512 and `2.38e-7` at head 32.
- `linear_grad.rs` now covers the released packed (FP8 block32/block128, FP4
  row32) and BF16 dense storage. `forward_mixed` evaluates the mixed
  boundary (stream cast, block-scaled GEMM, one BF16 output rounding, PEFT
  merge) and `backward` contracts the input gradient against the decoded,
  weight-scaled matrix: the frozen activation quantizer's straight-through
  edge divides by an exact UE8M0 power-of-two scale, so that scale cancels
  exactly. A CPU PyTorch autograd fixture checks **6,816 gradient values**
  and **2,624 forward values** across 11 cases; every forward is bit exact,
  worst gradient drift **5.96e-8** under `<2e-7`, `rtol=0`. Adapter
  gradients ride the dense branch. The shared input/weight-gradient kernels
  are one-to-three ulps off ATen at larger magnitudes; the fixture records
  the pre-calibration measurements and stays in the standard regime.
- `terminal.rs` preserves its original unreduced-loss-then-mean contract and
  adds `cross_entropy_and_input_grad` for CPU reduced cross entropy's NLL
  cascade. Two immutable CPU fixtures make their scalar rounding difference
  observable. Vector-width selection and reduction order now follow the
  actual operator, rather than reusing the ordinary tensor-sum layout.
- `mixed_moe_grad.rs` connects retained packed projections to SwiGLU,
  differentiable routing weights, compact-row gather/scatter and shared/routed
  input-gradient accumulation. Its tape holds no frozen expert payload;
  checkpoint-streamed routed and shared experts restage for backward, including
  adapter snapshots loaded during forward. Unvisited expert adapter gradients
  are absent at the optimizer boundary, matching CPU autograd's `None`.
- Seven mixed MoE autograd graphs, four isolated SwiGLU cases, a scalar-clamp
  regression and streamed repeats check **35,346 values** under `<2e-7`,
  `rtol=0`: every checked forward is bit exact; worst input drift **5.96e-8**,
  routing-gradient drift **4.84e-8**, adapter-gradient drift **1.86e-8**.
  Cases include all scoring functions, negative/zero route scales, exact clamp
  bounds, inactive experts, all three packed formats, FP32/BF16 streams and a
  48:1 expert-to-active ratio at reduced dimensions. Rounding uses the documented
  FP32 STE contract and zero dropout. The gate caught the scalar-bound clamp's
  strict subgradient and the AVX512 two/three-row input-GEMM grouping; both are
  corrected without reducing fixture magnitudes. Frozen packed matrices are
  decoded only during backward. **542** recorded shader artifacts reproduce.
- `save_checkpoint` atomically publishes a complete adapter/optimizer directory;
  `resume_from_checkpoint` validates it. Optimizer schema **3** binds the exact
  trained factors/config and file payloads, checks moment shapes/finiteness,
  and refuses unbound schema-1 state. Gradient validation covers every module
  before updating any factor. Independent A/B update counters are U64 tensors
  in the same checksum-bound payload as their moments. Existing bound schema-2
  checkpoints migrate using their formerly complete-cohort global step;
  unbound schema-1 checkpoints remain rejected. Checkpoint destinations are
  never overwritten.
- Conditional optimization now skips a missing A or B gradient without decay,
  moment changes or counter increments; an explicitly present zero gradient
  still updates. Bias correction uses each factor's counter. All-active and
  all-absent calls work through the normal `step` API; `step_parameters` also
  represents A/B independently. Whole-cohort validation and a bounded-memory
  update dry run prevent partial mutation on malformed input, counter overflow
  or finite gradients whose arithmetic would produce non-finite state.
- Two nine-call LoRA/rsLoRA CPU trajectories plus a three-step computed-gradient
  mixed-MoE trajectory check **44,908 values**, worst drift **1.49e-8** under
  `<2e-7`, `rtol=0`. Native projection logits stay within **1.12e-8**; every
  checked mixed-MoE pre/post-update output is exact. The mixed graph streams
  frozen routed/shared experts and updates only its six visited adapter
  modules, preserving nine inactive modules. Resume factors, moments, counters
  and native outputs are compared as bits/exact integers; standard PEFT export
  reloads without changing the output. This qualifies the optimizer/component
  lifecycle; the full-model loss-to-adapter gate is described below.
- `engram_grad.rs` now retains the normalized gate and selected-row projection
  operands, with adjoints for the residual stream, per-copy keys, shared value,
  projection inputs and component LoRA factors. It differentiates both RMS
  statistics, the signed-square-root floor and sigmoid, and folds shared-value
  gradients across the HC copies without atomics. Masks and discrete hashes
  remain frozen. The tape owns no table or frozen projection payload.
- Nine gate and four lookup/projection CPU autograd cases, plus three streamed
  repeats, check **224,076 values** under `<2e-7`, `rtol=0`. Worst drift is
  **1.19e-7** (key/value gradients); stream gradients stay within **5.96e-8**,
  selected-row projection-input gradients **2.98e-8**, and adapter factors'
  gradients **3.73e-9**. The **5,120-channel / HC=4** gate reproduces its frozen
  BF16 forward exactly. Coverage includes signed zeros, exact clamp-floor
  equality, all-masked rows, zero norms, HC=16, width 544, repeated/invalid
  hashes, dense/packed tables and FP32/BF16/FP8 projections. Backward succeeds
  after the table file becomes unavailable; another forward correctly fails.
  These use the existing FP32 STE mixed-precision contract. The adapter target
  allowlist and full-model training status are unchanged. All **544** recorded
  shader artifacts reproduce; every prior shader artifact is unchanged.

- `DeepseekV41AttentionTape` now executes and retains the native gathered
  attention/cache-write graph. Private window rings and shared compressed
  sources have graph-local cache IDs; duplicate physical registration and
  handles from other graphs are rejected. Each read owns independent GPU
  snapshots of Q, window/compressed values, indices and the frozen sink. Later
  cache writes, scratch reuse and released attention projections cannot change
  its VJP. Backward joins readers in reverse event order and splits cache
  gradients at each overwrite, returning source gradients and incoming-cache
  gradients. Prefill's complete chunk and its truncated ring seed join at the
  same source. Omitted output seeds stay absent; explicit zero seeds remain
  graph uses. All cache writes during recording must pass through the tape.
- Three complete CPU autograd graphs check **42,848 values** across **71
  attention reads and 82 cache writes**, worst drift **1.19e-7** with the
  unchanged `<2e-7`, `rtol=0` gate. Native head outputs stay within **3.73e-8**,
  query gradients **5.59e-9**, source gradients **1.19e-7**, and incoming-cache
  gradients **1.49e-8**. Coverage includes batched three-reader sharing,
  independent ratio-1 sources, ratio-2 publication gaps, prefill longer/shorter
  than the ring, repeated wraps, masked/duplicate gathers, absent and zero
  seeds, and real native FP8/UE8M0 window / FP4/E4M3 compressed quantization.
  Quantized values/scales and every forward cache update also match as bits.
  These are VJPs of post-quantization operands with fixed indices, not a new
  quantizer differentiation rule or a complete attention-projection trainer.
  `retained_bytes` accounts snapshot payloads only; model memory planning and
  recomputation remain separate.
- `DeepseekV41AttentionProjectionAdjoints` now connects the sparse/cache graph
  to `wq_a -> q_norm -> wq_b -> RoPE`, `wkv -> kv_norm -> RoPE -> FP8`, and
  `inverse RoPE -> FP8 -> grouped wo_a -> wo_b`. Its input/output tapes use the
  unchanged native mixed forward and the established FP32 STE cast/quantizer
  contract. They snapshot activations, adapter factors and norm gains, retaining
  no frozen projection base. Backward accepts restaged bases without adapters;
  configuration, geometry and every external edge are validated. Omitted Q,
  KV or auxiliary-latent edges preserve absent projection gradients.
- Four connected CPU autograd graphs, **16 prefill/decode steps**, check
  **22,223 values** with worst drift **1.19e-7**, `<2e-7`, `rtol=0`. The measured
  Q, query-latent and window-KV forward values are bit-exact. All 16 native
  attention outputs match ordinary inference as bits, with output drift at
  most **1.12e-8** and **319 frozen logit probes** within **1.49e-8**. Input
  gradients stay within **1.12e-8** and the **5,632** accumulated adapter-factor
  gradients within **1.49e-8**. Cases cover FP32 LoRA, batched compressed FP32
  rsLoRA, dense BF16 and packed FP8/rsLoRA, grouped output, window wraps, an
  unseeded output whose KV is used later, and an explicit auxiliary latent
  edge. Compressed payloads are external inputs here; their finalizer and
  compressor backward composition remains separate. This qualifies the
  connected attention component with fixed indices, not the full model loss
  or optimizer lifecycle.

What remains before LoRA training can be enabled, in the order the chain must
be built backwards from the loss: integrate the connected attention component
into the model tape, then qualify the complete block/model loss chain. The
retained sparse/cache graph now aggregates shared
source readers across decode steps, and the compressor chains its own decode
state. Q/window-KV and grouped output projections, norms and RoPE now compose
with the cache graph; compressed finalizer/compressor composition still needs
model-tape wiring. The
attention-side Hyper-Connection stages need no new kernels: they are the
collapse, split and projection adjoints gated here. The mixed MoE backward is
component-qualified, including conditional optimizer/save/resume integration;
it still needs model-tape integration. Retained activations use
capacity-sized expert rows, so a model trainer must also budget or recompute
those tapes and bind frozen model identity at resume. `merge-lora`, the
CLI's training commands and the GUI's V4.1 adapter workflow stay fail-closed
until the chain exists and a full-loss gate passes.

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::terminal -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::backbone::block_grad_tests -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::train -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::attention_grad -- --nocapture
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention_tape.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib attention_tape -- --nocapture
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention_projection_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::attention::projection_grad -- --nocapture
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention_grad.py
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_projection_grad.py
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_compressor_grad.py
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_compressor_decode_grad.py
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_packed_grad.py
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_mixed_moe_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib mixed_moe_grad -- --nocapture
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_conditional_adamw.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib conditional_adamw -- --nocapture
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_engram_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::engram::gradients -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib packed_projection_tape_vjps_match_cpu_pytorch -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::compressor_grad -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::compressor_grad::decode_tests -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::mixed_linear::gradients -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

### DeepSeek V4.1 full-model PEFT training qualification

The loss-to-adapter chain, the model trainer, optimizer save/resume, adapter
export and the CLI route are landed and gated, so the "training is being built
from the loss downward" notes above describe the build-up rather than the
current contract. Four frozen CPU PyTorch oracles drive the end-to-end trainer
gate: dense FP32, FP32 with FP4 routed / FP8 shared experts, BF16 with the same
packed experts, and the production ratio geometry (40 layers, 384 routed
experts, top-6, HC4). Each runs LoRA and rsLoRA for three AdamW steps, saves,
resumes at step 2 and exports an adapter that reloads.

**Every compared step is replayed from the reference's own state.** Before each
step the gate pins every adapter factor *and* optimizer moment to the frozen
fixture state (`DeepseekV41AdapterTrainer::load_state_bits`, exposed as
`DeepseekV41ModelTrainer::load_adapter_state_bits`). Without the pin, AdamW's
`eps` division (`sqrt(exp_avg_sq)/bias_correction2_sqrt + eps`) turns a gated
one-or-two-ulp gradient difference at a near-zero element into a much larger
factor difference: the measured coefficient reaches
`step_size*(1-beta1)/eps = 1.0e5`, and over three steps it flipped a discrete
top-6 routing decision and a 7.79e-2 decode logit. Under the pin the gradient is
the only differing input, so each step is qualified on its own arithmetic.

The pinned gate asserts: every logit strictly below `2e-7` with `rtol=0`; every
factor gradient and moment below `2e-7`; every active factor's live result
**bit-exactly** the documented pinned single-tensor AdamW policy applied to the
native gradient, with the fixtures' own trajectories reproducing that policy;
factors without a gradient bit-identical; and a factor residual bounded by the
*exactly computed* policy response to the gated gradient difference plus
`8*2^-24` of the largest operand for each implementation's own roundings (ATen's
vectorized `sqrt` is not correctly rounded; measured within 1 ulp of `denom`).
Over the four oracles: logits worst `1.49e-7` (the production-ratio and BF16
fixtures are bit-exact at every step), gradients `1.12e-8`, moments `1.4e-9`,
and `5,585,376 / 5,585,376` compared factor updates bit-exact to that policy.
The residual factor difference against the fixture reaches `6.71e-7`; that is
the AdamW response to the gated gradient difference, which the receipt bounds
by computing the policy's own output exactly - the worst bound ratio over every
compared element is `0.999972 < 1`. Aggregate over 24 pinned step-comparisons:
`6,144` logit values (`96` logit points), `16,896` gradient, `87,384` factor and
`34,092` moment points.

The production-ratio case is `#[ignore]`d like the repository's other
40-layer/384-expert graphs. Its peak commit does not fit beside the rest of the
suite on this machine (10.7 GB of RAM, a 19 GB commit limit), so it is run
`--ignored` in its own process and its log is bound into the receipt;
`target/v41-completion/production-gate-final.log` records the final build.

The release checks therefore run as three processes on the final tree: **175
passed / 0 failed / 23 ignored** (`deepseek_v41::` with
`--skip model::tests::training`), **6 passed / 0 failed / 1 ignored** (the
`deepseek_v41::model::tests::training::` group) and **1 passed** for the
standalone production-ratio gate - **182 passed, 0 failed, 24 ignored** in
total. All three logs are hash-bound in the machine receipt
(`validation/release_parity/deepseek_v41/qualification.json`).

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41:: -- --skip model::tests::training --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::model::tests::training:: -- --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_full_model_cpu_autograd_adamw_resume_and_export_gate -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_full_model_packed_cpu_autograd_adamw_resume_and_export_gate -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_full_model_bf16_cpu_autograd_adamw_resume_and_export_gate -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_full_model_production_ratio_cpu_training_gate -- --ignored --nocapture
.venv-vulkan\Scripts\python.exe target/v41-completion/summarize_pinned_replay.py
.venv-vulkan\Scripts\python.exe target/v41-completion/collect_training_receipt.py
```

### Tiered weight residency across native architectures

The dedicated V4.1 graph is not the only place a released checkpoint has to be
staged; it was simply the first. `src/weight_offload.rs` makes the tiers every
native loader already depends on explicit and shared, and the common host reader
now uses it.

Three tiers are named and accounted:

- **device** - a tensor or tile uploaded into a `GpuBuffer`.
- **host** - a byte-capped tile cache of decoded payloads. Occupancy never
exceeds its budget; the least recently used tile is dropped to make room.
- **package** - the SafeTensors file, read through the OS page cache.

What changed for the non-V4.1 architectures is the **package** tier. The host
readers (`read_f32_tensor`, `read_f64_tensor`, `read_i64_tensor`,
`read_u32_tensor`, `tensor_exists` and the FP32-master dtype validator in
`checkpoint.rs`) previously called `fs::read` and deserialised the **whole
shard** for **every tensor** they wanted. A loader asking for fifty tensors
read and parsed the entire shard fifty times; the largest tensors were also
expanded into a full host `Vec<f32>` even where only a row range was needed.
They now read only the requested byte range of only the requested tensor, and
shard headers are parsed once and cached process-wide, keyed by file length and
modification time so a regenerated package is not served a stale header.

Measured, not asserted: loading fifty tensors from one shard reads exactly the
payload bytes once (`payload_reads == 50`, one header parse) instead of fifty
shard reads, and `single_tensor_image` returns an image that deserialises to the
same dtype, shape and values, so each reader's decode body is unchanged.
`contexts beyond the request` are no longer paid for.

`plan_weight_residency` is the tier decision any loader can make before it
allocates: `Resident` when the whole set fits beside what is already loaded,
`Tiered` when it must be staged. An unreadable or absent driver budget is
reported as unknown and maps to `Tiered`, so a missing budget degrades to the
bounded path instead of an out-of-memory failure. `OffloadLedger` records bytes
read from the package, bytes served from the host cache, cache hits/misses,
evictions and peak occupancy, so a tier claim can be checked rather than
believed.

The primitives above are available to every architecture for inference, training
and PEFT, and every architecture's package loading now goes through the bounded
reader.

### The first non-V4.1 consumer is converted

The heaviest single weight in the tree outside V4.1 is the Qwen4-Exp PLE n-gram
embedding table, `model.layers.N.ple.ple_embedding.ngram_embedding.weight`. The
released configuration sizes it in tens of GiB, which is why its loader already
kept a source descriptor instead of the decoded table. Its consumer is the PLE
layer's per-step row gather, and that gather now reads through the shared
process-wide `ModelPackage` rather than the private per-row reader it used
before. `read_row_gather` groups the requested row ids by host tile, pulls each
tile once for the whole request, serves duplicate and repeated rows from that
one read, and returns only the requested rows - which are the only rows uploaded
as a device tile.

Values are unchanged: the same bytes at the same offsets widen through the same
F32/F16/BF16 paths, and a non-finite payload is still rejected. Measured:

- The two strict Qwen4-Exp logits fixtures (`qwen4_exp_ple`,
  `qwen4_exp_mixed_ple`, unmasked and padded) produce **byte-identical** native
  reports before and after the conversion, and still report zero failing values
  against the Transformers reference at `atol=2e-7`, `rtol=0`.
- A negative control (`HIERARCHOS_TRANSFORMER_ZERO_GATHER_CONTROL`) makes the
  gather return zeros and moves those logits by up to `4.11e-3`, so the equality
  above is not vacuous: the gather really is on the logits path.
- A unit test decodes the same table through both readers and requires exact
  equality, then pins the batched read (`payload_reads == 1` for five scattered
  rows across two host tiles, and a repeat gather that reads nothing new).
- Four-token PLE generation, cached and full-prefix, matches Transformers to
  `1.86e-8`.

### The generic output head streams device tiles

The device tier's next consumer is the one weight every dense family in the tree
shares: the vocabulary-sized output head. It is not the largest weight in a
package, but it is the only one whose device cost scales with the **vocabulary**
instead of with the model width - a 129,280x5,120 FP32 head is 2.65 GB, and the
four allocations a trainable parameter of that size needs (weight, gradient and
both AdamW moments) are 10.6 GB - and the only one whose rows are consumed
independently, because every logit column is its own dot product against the
hidden row. A tile of head rows therefore produces a tile of logit columns and
nothing else is needed to finish them, which is what makes this weight
streamable where a weight reduced along its columns is not.

`VulkanTransformer::stream_lm_head_from_package` converts one loaded graph's head
to that tier, and `--stream-lm-head` exposes it in the logits binary. The
conversion is explicit and additive: `from_hf_package` and every other
constructor still keep the head resident, and the default path is unchanged. What
the conversion does:

- It proves the package tensor is the one the graph loaded, by tile-by-tile
equality with the resident values, before releasing anything. The per-family head
spellings (`lm_head.weight`, `output.decoder.weight`, ...) make a name list a
maintenance hazard, and a name-based guess could silently stream the embedding
table in its place; a package holding more than one tensor equal to the head is
refused rather than guessed at.
- It releases the resident payload - weight, gradient and both AdamW moments -
immediately, leaving a one-element placeholder that no projection binds. A weight
tile, one output tile and the per-column bias slice are allocated instead, and
every buffer is sized by the largest tile the plan can produce (8,207 rows),
never by the table: streamed, the head costs the same device memory whether the
vocabulary has 16,384 rows or a million.
- It projects the vocabulary in 8,192-row tiles, each through the same kernel the
full-width dispatch would have selected, and copies each tile's logit columns
into the logits as an FP32 buffer move, so the tile boundary never enters the
arithmetic. A vocabulary remainder below the 16-column minimum both kernels
select their schedule on is merged into the preceding tile instead of being
dispatched at a width the resident projection never used.
- It keeps the output bias resident - one value per column, about 0.5 MB at
129,280 tokens. Untied projections seed the accumulator with each bias slice;
tied projections use the ordinary linear kernel and a separate bias add,
preserving each resident path's rounding order.
- For a tied head, token lookup and projection share one immutable package
parameter. Lookup gathers only requested rows, including duplicates and stored
padding rows. The full resident embedding, gradient, moments and table-sized
readbacks are released. Raw resident-buffer access and parameter mutation fail
with errors. Cached decode and DoLa projection use the same streamed head.
- The output tile holds `graph_rows * max_tile_rows` values, independently of
hidden width. A batch with more rows than hidden dimensions cannot overrun it.
- It still answers `read()` with the complete table. Parameter export and the
round-trip comparisons materialize the head through the same bounded package
tiles the projection uses, so a streamed graph is not a one-way conversion.
- It fails closed everywhere else: training, a training-mode forward, PEFT
replication (`modules_to_save` on `lm_head`), acquiring training state, and
attaching an adapter are all refused with an explicit error until the head is
loaded resident again. A streamed parameter cannot silently take part in an
optimizer step it has no state for.

Measured, not asserted:

- A two-tile vocabulary (16,387 rows, whose three-row remainder is merged into
the second tile) produces **byte-identical** logits to the same package loaded
resident, both unbiased and with a biased head - `assert_eq!` on the whole logit
vector, not a tolerance.
- The tied-head regression also crosses the tile boundary, with 655,480 logits
  from 40 rows (more than the hidden width), biased projection, cached decoding,
  DoLa projection and export/reload all matching the resident graph exactly.
- The tied GPT-2 CLI fixture has identical resident/streamed/reloaded native
  logits (128 values) and 28 byte-identical exported tensors. Maximum drift from
  fresh CPU PyTorch and its frozen CPU oracle is `2.9802322387695312e-8`, strictly
  below `2e-7` with `rtol=0`. See
  `target/streamed-tied-head-check/qualification.json` for the measured report.
- The same equality holds end to end through the logits binary: the resident and
`--stream-lm-head` reports of the Falcon-H1 fixture agree exactly (369 logits,
`max_abs = 0.0`).
- Exporting from a streamed graph reproduces the package: all 37 tensors of the
exported checkpoint, `lm_head.weight` included, are bitwise equal to the source
package, and a resident load of the exported directory returns the same logits
again.

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib streamed_output_head -- --nocapture
cargo run --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-transformer-logits -- --model hierarchos-vulkan/.falcon-h1-fixtures/cli-trained --fixture hierarchos-vulkan/.falcon-h1-fixtures/cli-trained/input.json --output target/streamed-head-check/resident.json
cargo run --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-transformer-logits -- --model hierarchos-vulkan/.falcon-h1-fixtures/cli-trained --fixture hierarchos-vulkan/.falcon-h1-fixtures/cli-trained/input.json --stream-lm-head --output target/streamed-head-check/streamed.json
```

`VulkanTransformer::from_hf_package_with_streamed_lm_head` also supports
**deferred vocabulary construction** in the GPT-2/GPT-SW3 and Llama-layout loaders. The
loader retains the exact physical tensor name through wrapper rebasing and uses
bounded package tiles to validate payloads and any tied alias. It never creates
either table's full host vector, resident weight, gradient or AdamW buffers.
A tied graph shares one parameter; an untied graph retains separate input and
output sources and identities, even when their values happen to be equal.
Token lookup gathers only requested rows, preserving order, duplicates and
stored padding rows. F32, BF16 and F16 storage use the same widening as resident
loading. Physical Mllama tables may contain eight extra rows: lookup and parameter
reads enforce the logical text vocabulary, and export restores the reserved rows
from the source. The common full-vocabulary backward scratch
is also omitted in streamed construction and released by post-load conversion.

Without adapter arguments, the existing `--stream-lm-head` flag selects this
constructor. Other architecture loaders use the existing post-load conversion
and retain its initial allocation peak. The ordinary constructors remain
resident. The streamed vocabulary is a **frozen** parameter: adapter training
runs against it (see below), while training the table itself, `modules_to_save`
or PEFT replacement of it, and a training-mode forward of the streamed base stay
refused before mutation, with the resident paths unchanged.

The load-time regression uses a 65,539-row, 32-column table, with tied and untied
heads in each of F32/BF16/F16 storage. BF16 cases also use wrapped physical names
and three indexed shards separating the input table, output table and body.
Each case checks 524,312 bit-identical native logits, cached
decoding and export/reload. Loading peaks (logical live Vulkan buffer bytes,
measured on fresh devices before the first forward) are:

| Head | Resident loading | Deferred loading |
| --- | ---: | ---: |
| Tied | 71,811,328 bytes | 8,109,732 bytes |
| Untied | 105,367,312 bytes | 8,109,752 bytes |

Both deferred paths' largest buffer request is 2,097,248 bytes, below the
8,388,992-byte canonical FP32 table, including allocations already released.
The previous head-only deferred path needed 58,443,692 bytes for untied graphs;
staging the input table removes another 50,333,940 bytes in this fixture. These are
allocator buffer measurements, not process RSS or driver arena reservation.
GPT-2, Llama, Qwen2.5 GQA and Phi-3 CLI fixtures total 512 logits, with identical
resident/streamed/reloaded native outputs and at most `2.9802322387695312e-8`
drift against both fresh CPU PyTorch and unchanged frozen CPU references
(`rtol=0`, strictly below `2e-7`). A new independently untied GPT-2 fixture adds
128 fresh CPU comparisons, with `2.2351741790771484e-8` maximum drift. All 119
exported tensors across the five CPU fixtures match their source payloads byte
for byte. Mismatched aliases, non-finite input or output payloads and invalid
head shapes are rejected before device buffer allocation. Logical prefix tests
also reject out-of-range input rows and preserve Mllama's reserved rows through
export/reload. See `target/untied-embedding-check/qualification.json` and the
per-group logs. All 87 unique targeted tests passed.

Deferred vocabulary loading in the remaining architecture loaders, untied input
staging in the explicit post-load conversion, and tiering the remaining generic
attention/MLP weights remain unimplemented. A *trainable* vocabulary cannot
stream: its full gradient table and both AdamW moments are the arithmetic, not
an allocation choice.

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib weight_offload -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib qwen4_ple_row_gather_matches -- --nocapture
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_qwen4_exp_generation.py --families qwen4_exp_ple qwen4_exp_mixed_ple --steps 4
```

Storage is reclaimed separately, and the ledger accumulates across runs rather
than overwriting so the totals stay auditable. Large generated packages used only
by `#[ignore]`d gates are removed; the small `structure`/`wide` packages that
default tests need are kept, because deleting them fails seven non-ignored tests.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/reclaim_v41_scratch.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/summarize_scratch_cleanup.py
```

### Adapter training against a streamed vocabulary

A graph whose head or tied embedding is served from package tiles can now run
LoRA/rsLoRA training, resume from an optimizer checkpoint and export its
adapter. The frozen table is never resident and never staged in full:

- Backward through the frozen table uses `linear_input_grad_streamed.comp`,
  which accumulates `dX` tile by tile starting from the previous FP32
  accumulator. Tile boundaries are whole multiples of the resident kernel's
  32-term inner block, so every FMA keeps the resident order and the assembled
  adjoint is bit-identical to the resident one.
- Each tile's upload retires with the tile's own submission. Before this, a
  streamed projection released its device weight tiles but retained every tile's
  host-visible staging until the caller submitted, which is a table-sized hold
  on a large vocabulary. The regression asserts the surviving batch still
  reports the tile dispatches and reports `uploaded_bytes() == 0`.
- `lora_base_fingerprint` and the training/resume checkpoint hash streamed
  vocabulary rows in bounded 8,192-row chunks instead of materializing a
  table-sized host buffer.
- `--stream-lm-head` is accepted by `transformer_logits`, `transformer_parity`
  and `transformer_train`; the PEFT oracle appends it when
  `HIERARCHOS_PEFT_STREAM_LM_HEAD=1`, so identical fixtures drive both tiers.

Measured on this machine: streamed and resident training are **bit-exact**
(forward logits, losses, head adjoint, every LoRA gradient, post-step logits,
base fingerprint, resume state and adapter export, for tied and untied
vocabularies in F32/BF16/F16). All **32 PEFT oracle families pass** with the
streaming switch on - worst `peft_max_abs` `6.3981552111958e-8`, strictly below
`2e-7`, `rtol=0` - and the 29 families that had run before the per-tile staging
change reproduced **identical** oracle values, so the retirement is
arithmetic-neutral. The DeepSeek V4.1 production-width adapter case passes with
worst drift exactly `0.0` against its pinned CPU fixture and adapter delta
`2.543889`.

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib streamed_vocabulary_lora_training_resume -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib streamed_output_head -- --nocapture
$env:HIERARCHOS_KIMI_K3_REMOTE_DIR = (Resolve-Path target/kimi-k3-remote)
.\.venv-vulkan\Scripts\python.exe target/peft-offload-completion/qualify.py
```

## RDNA3 parameter/MoE streaming and V4.1 training follow-through (2026-10-30)

Application streaming is explicit: `hierarchos-native-cli infer` /
`transformer-generate` and frozen-base `transformer-finetune` accept
`--stream-parameters`. The GUI exposes an unchecked **Stream frozen parameters
and MoE experts from disk** checkbox. Application defaults are resident,
including the dedicated V4.1 route. Historical V4.1 library package constructors
retain their tiered behavior; application calls pass the new residency policy
explicitly. V4.1 resume must match the checkpoint's recorded residency.

The generic frozen traversal includes routed/shared experts and routers even
when those weights are outside an architecture's PEFT target subset. Package
views cover direct matrices, Conv1D transposes, contiguous expert slices,
GPT-OSS transposed/interleaved slices and Qwen query/gate row splits. Each
source in the post-load path is compared bit-for-bit with the loaded native
parameter before replacement. The source-first path described below binds the
architecture loader's exact tensor descriptor and validates its geometry, dtype
and finite values in bounded tiles. Projection and adjoint shaders are unchanged.

| Independent qualification on this machine | Result |
| --- | --- |
| All 32 previously-green CPU/PEFT profiles with parameter streaming | pass, worst metric `9.000000000813912e-8`, strict `<2e-7`, `rtol=0` |
| V4.1 uninterrupted LoRA and rsLoRA, three AdamW steps each, 384 experts/top-6/HC4 | 1,536 logits bit-exact to CPU; native factors/moments carry between steps without fixture resets |
| V4.1 released 40-layer inference schedule, streamed default package API | 3,216 logits bit-exact to CPU |
| CLI train/save/resume/export and cached generation, resident vs streamed | identical adapter bytes and generation for V4.1 LoRA/rsLoRA, Llama and Mixtral |

**Depth correction:** the production-ratio **training** fixture has six layers
and width 32. Earlier handoff prose called it forty layers. The separate
inference fixture exercises the released forty-layer schedule. Numeric records,
frozen expectations and the strict logit gate were not changed by this metadata
correction. The uninterrupted training gate is additional evidence beyond the
earlier pinned per-step optimizer replay.

Receipts: `validation/release_parity/parameter_streaming/qualification.json`,
`validation/release_parity/deepseek_v41/uninterrupted-training.json` and
`validation/release_parity/deepseek_v41/rdna3-streaming-qualification.json`.

Scope remains frozen-base streaming. Ordinary Llama/Mistral/Qwen2/Mixtral
packages now construct their body directly from checkpoint descriptors; other
generic body conversion follows initial resident loading. Specialized operators outside
`VulkanLinear` can remain resident. Full base-weight training offload and
encoder-decoder/cross-attention streaming are not admitted by the application
flag. This matrix does not qualify all 146 registry families, full production
weights, unsupported V4.1 adapter classes or Intel Gen9 performance.

## Source-first body loading (RDNA3, 2026-10-30)

With `--stream-parameters` (or the unchecked-by-default GUI checkbox), ordinary
`llama`, `mistral`, `qwen2` and `mixtral` packages now construct frozen projections
from typed physical tensor descriptors. No complete host weight vector or device
weight/gradient/AdamW matrix is created for those body projections. Packed and
ordinary/legacy per-expert Mixtral sources bind exact contiguous views. Other
families retain their qualified post-load conversion path.

The independent CPU/PEFT matrix passes all 32 profiles through the application
constructor, worst metric `9.000000000813912e-8`, strict `<2e-7`, `rtol=0`. Four
profiles use the new source-first route; this does not promote the remaining
families to bounded initial loading. The native five-case dense/MoE matrix also
matches resident logits and two-step AdamW adapter bytes exactly.

| Synthetic construction measurement | Resident buffer peak | Source-first buffer peak |
| --- | ---: | ---: |
| Llama, hidden 512 / intermediate 1024 | 41,063,772 bytes | 1,873,292 bytes |
| Llama, hidden 1024 / intermediate 3072 | 208,278,940 bytes | 4,068,812 bytes |
| Mixtral, hidden 512 / intermediate 1024 / 8 experts | 217,701,244 bytes | 2,284,548 bytes |

At fixed Mixtral width, increasing expert count from two to eight adds 1,176
construction buffer bytes. One dense projection exceeds the 8 MiB validation
tile; the largest returned validation vector is exactly 8 MiB. All measured
package reads stay within that bound and cache occupancy stays within 32 MiB.
These counters describe construction buffers and package requests, excluding
driver/pipeline memory and process RSS. Norms, biases, activations and adapter
state remain resident.

The loader/storage gate checks 23 cases and 7,168 logits, all bit-exact to
resident execution: F32/F16/BF16, sharded/wrapped dense backbones, tied vocabulary,
packed/current/legacy experts and a wrapped Mixtral source. Non-finite late
experts, wrong projection shapes and unequal tied aliases reject. Storage tests
compare with resident execution; the independent CPU/PEFT gate is separate.

The fresh CPU receipt is
`validation/release_parity/parameter_streaming/source-first-cpu-peft-qualification.json`.
The combined loading/application receipt is
`validation/release_parity/parameter_streaming/source-first-qualification.json`.
Previous receipts and their parent hashes remain intact. Full-base training
offload, encoder-decoder streaming, full production weights, all-146 coverage
and Intel Gen9 performance remain outside this qualification.

## V4.1 scheduled metadata uploads (RDNA3, 2026-10-30)

Attention window indices, candidate visibility lengths and retained attention
tape indices now stage their unchanged integer bytes into the enclosing layer
batch. This removes standalone uploads and premature layer retirements. Routing
counter placement, shader arithmetic, dispatch order and application defaults
are unchanged. Streaming remains opt-in through the CLI flag / GUI toggle.

Three alternating release A/B pairs on the existing 40-layer/384-expert,
vocabulary-67 fixture measured:

| Residency | Decode submissions | Decode latency, ms/token | Prefill latency, ms |
| --- | ---: | ---: | ---: |
| Resident | 252 -> 170 | 461.359 -> 370.847 | 709.547 -> 583.081 |
| Streamed | 4,680 -> 4,598 | 938.973 -> 932.490 | 2,648.382 -> 2,653.208 |

Resident decode latency falls about 20%; streamed timing is effectively
unchanged. These are paired measurements on a scaled fixture, with variable
host timing, not a released-checkpoint throughput claim.

**Timing correction:** the previous probe divided the median of individual
single-token decode calls by `--decode` again. Its old `decode_ms_per_token`
field was eight times too small at `--decode 8`. The corrected probe and new
comparison use raw phase records. Previous A/B speedup ratios retain their
meaning; old absolute ms/token labels should not be reused.

Fresh checks pass: 182 default V4.1 tests (25 ignored), 37 Vulkan tests,
3 generic-streaming tests, 16 offloading tests and 41 CLI tests. Explicit
40-layer joint resident/fully-streamed CPU gates compare 6,432 logits
bit-exactly; uninterrupted LoRA/rsLoRA adds 1,536 bit-exact CPU logits.
Parsed default-gate logits cover 201,714 values, worst `1.4901161193847656e-7`,
strictly below `2e-7`, `rtol=0`. All 72 archived training logit comparisons
retain identical reported drift. CLI save/resume/export and cached generation
remain identical across resident and streamed execution.

Receipt: `validation/release_parity/deepseek_v41/rdna3-metadata-upload.json`.
The old batching receipt is preserved. Its recorded executable hash differs
from the rebuilt executable present at pickup; the actual baseline executable
was frozen and re-probed, and all six recorded source hashes matched. The new
receipt binds the actual baseline and candidate binaries. Performance digests
cover the final decoded row; the separate CPU gates cover complete logits.
Routing readbacks remain; this does not implement a GPU indirect worklist.

## V4.1 expert uploads and checkpoint cache (RDNA3, 2026-10-30)

Packed data/scales and LoRA factors now share each linear upload batch. Each
expert's three projections share a startup batch; streamed uploads join the
expert's compute/scatter, with retirement before admitting another expert.
Exact-size transient staging is released at retirement. The existing lifetime
assertion caught an initial retained 4 MiB arena chunk; that version was rejected.
Shader arithmetic, reductions and expert accumulation order are unchanged.

Streaming enables one checkpoint-wide raw payload cache: at most 32 MiB and
32,768 ranges, shared across shards/clones. Every hit still checks shard size
and modification time; adapter factors are not cached. The initial 8,192-range
pilot churned before using its byte budget, so the independently bounded metadata
limit was increased. The final measured working set is 6,208,646 bytes / 10,669
ranges. Each measured run has 83,490 cache hits and zero payload reads after
warm-up. This is file-payload I/O, not a physical-SSD read counter.

Three fresh alternating release A/B pairs on the retained 40-layer/E384/vocab67
checkpoint, prefill8/decode8, compare the prior qualified executable:

| Measurement | Before | After |
| --- | ---: | ---: |
| Resident load, ms | 11,772.948 | 5,458.652 |
| Resident load submissions | 93,484 | 16,312 |
| Streamed prefill, ms | 2,673.513 | 1,800.810 |
| Streamed decode, ms/token | 936.6305 | 632.9385 |
| Streamed prefill submissions | 15,074 | 1,869 |
| Streamed decode submissions | 4,598 | 996 |

Streamed decode latency is about 32% lower (1.48x rate). Resident decode remains
170 submissions and 365.716 -> 355.626 ms/token, a small timing difference with
no claimed compute improvement. Default-loader startup is effectively unchanged
(456.655 -> 464.153 ms). Timings do not predict production-weight or Gen9 speed.

All 287 fresh tests pass: V4.1 default185 (25 ignored), Vulkan38, generic
streaming3, offload16, CLI41 and four explicit model/PEFT gates. Normal package,
joint resident and joint fully streamed gates each compare 3,216 CPU logits;
the uninterrupted PEFT trajectory adds 1,536, all bit-exact. Default-gate parsed
logits cover 201,714 values, worst `1.4901161193847656e-7`, strictly `<2e-7`,
`rtol=0`. All 72 archived training labels retain identical reported drift and
historical inference/PEFT CPU fixture hashes match. CLI LoRA/rsLoRA, Llama and
Mixtral training/resume/export bytes and generation remain identical across
residencies. Streaming remains opt-in through CLI/GUI.

Receipt: `validation/release_parity/deepseek_v41/rdna3-expert-upload.json`.
Evidence: `../target/v41-final-optimizations/`. The unfinished exploratory pilot
is excluded from final timings. The portable source/fixture packager and Gen9
gate runner are described in `validation/intel-handoff/DEEPSEEK_V41.md`.
Routing readbacks remain; this phase adds no new kernel arithmetic or full-base
training offload, and Gen9 must pass its own target gates.

## V4.1 final resident/streamed compute baseline (RDNA3, 2026-10-30)

The default MoE gate now uses one 64-lane workgroup per row. Independent expert
dots and unary scores run in parallel with 4 KiB shared scratch; lane zero keeps
the pinned CPU top-k/tie staging, softmax reduction and selected-weight sum.
Dot reduction panels, unary functions, ordering and rounding are unchanged.
Expert w1/w3 share their input rounding and matching-block FP8/UE8 preparation.
Each output retains its boundary; LoRA consumes the original module input.
Dense and different-block fallback paths remain. No activation persists across
experts, calls or optimizer updates. The existing staging-lifetime test passes.

Three alternating unprofiled A/B pairs against the prior qualified executable:

| Scaled 40-layer/E384/vocab67 measurement | Before | Final |
| --- | ---: | ---: |
| Resident prefill8, ms | 585.630 | 527.072 |
| Resident decode, ms/token | 372.1765 | 306.8150 |
| Streamed prefill8, ms | 1782.044 | 1285.738 |
| Streamed decode, ms/token | 630.4380 | 512.5925 |

Decode latency is 18% lower resident and 19% lower streamed; streamed prefill is
28% lower. Decode submissions remain 170/996 and loading is effectively
unchanged. The bounded payload cache retains zero warmed measured payload reads.
Full-stride profiling separately covers 18 model calls: resident GPU dispatch
brackets 5641.356 -> 3477.269 ms; gate 1923.572 -> 160.259 ms. Total dispatches
122220 -> 108648, eliminating 6786 quantizations and 6786 casts. Profiles perturb
timings; the unprofiled pairs establish latency. Remaining GPU shares are roughly
quantization 30%, HC split 30%, packed GEMM 26%, gate 5%: Gen9 must re-profile.

288 fresh tests pass: default V4.1 186 (25 ignored), Vulkan 38, generic streaming 3,
offload 16, CLI 41 and four explicit model/PEFT gates. Explicit CPU logits cover
11,184 values bit-exactly. Default parsed 201,714 logits have worst
`1.4901161193847656e-7`, strict `<2e-7`, `rtol=0`; all 166 previous comparisons
(106 labels) retain identical drift. Frozen inference/PEFT fixture hashes match.
The new paired-projection test checks 42 FP32/BF16, FP4/FP8, LoRA and block-layout
cases bit-for-bit against sequential execution and checks saved dispatches.
All 548 shader artifacts reproduce; push layouts 428 / host structs 109 pass.
CLI resident/streamed LoRA/rsLoRA and Llama/Mixtral training, resume, export and
generation remain identical. Application streaming remains opt-in.

Receipt: `validation/release_parity/deepseek_v41/rdna3-final-push.json`.
Evidence: `../target/v41-final-push/`. Gen9 instructions:
`validation/intel-handoff/DEEPSEEK_V41.md`. The final portable bundle includes
source, pinned SPIR-V, immutable CPU fixtures, the retained joint checkpoint,
qualification logs, profiles and before-source snapshots. Routing readbacks
remain; this adds no full-base training offload, production-weight throughput
claim or Intel Gen9 device qualification.

## Registry inventory

The exact inventory is intentionally generated instead of duplicated here.
`README_ARCHITECTURES.md` is the source-derived native architecture list, while
`audit_transformers_coverage.py` compares it with the adjacent Python
Transformers checkout and reports current causal-LM, masked-LM, seq2seq-LM, and
configuration-registry overlap. Those counts are not an assertion that every
configuration or task head associated with an overlapping model type executes
correctly.

```powershell
python hierarchos-vulkan/validation/audit_transformers_coverage.py --all-tasks
python hierarchos-vulkan/validation/audit_transformers_coverage.py --json
python -m unittest discover -s hierarchos-vulkan/validation -p 'test_*.py'
```

The audit parses Python syntax without importing or executing the model package,
including composed AutoModel registries. Additional task mappings report only
registry overlap and explicitly leave task implementation unverified.

## Numerical checks

Use Python dependencies compatible with the local Transformers checkout. The
reference scripts resolve `HIERARCHOS_TRANSFORMERS_CHECKOUT`, an adjacent
checkout, or `~/transformers`, then include the imported version/path in their
output so an older system installation cannot silently become the reference.
Rust, Cargo and a Vulkan compute device are required; model fixtures are tiny
and created locally, with no pretrained-model downloads.

An isolated Windows setup reusing an existing PyTorch installation is:

```powershell
python -m venv --system-site-packages .venv-vulkan
.\.venv-vulkan\Scripts\python.exe -m pip install 'huggingface-hub>=1.5.0,<2.0' 'tokenizers>=0.23.1,<0.24.0' 'safetensors>=0.8.0'
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_logits.py --headline-strict
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_training.py --headline-strict
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_training.py --jitter
```

On other platforms use the environment's Python executable. Cargo reports the
probe executable path directly, including custom target directories. Both
scripts accept `--device-index` and `--families` to select a device or subset.

Broader regression coverage on the same AMD Radeon Graphics device and
Transformers checkout includes:

| Check | Scope |
| --- | --- |
| Headline forward logits | DeepSeek V4, Phi-4 text, Phi-3, Kimi K2.5 text, Kimi K3/KimiLinear text, GPT-OSS, SmolLM3, Mistral 4, MiniMax M3 text, Gemma 4, MiniMax M2 |
| Additional forward logits | GPT-2, Llama, Qwen2 / Qwen2.5 text, Mistral, Mixtral, DBRX, BERT, T5, BART, Switch Transformers |
| Masking and batching | Unmasked batch one and mixed left/right padding in batch two; visible-token logits |
| Encoder-decoder | Independent source/decoder lengths; Switch encoder has three sparse layers, decoder has a different two-layer schedule |
| Headline training and export | Exactly two AdamW steps; cross-entropy loss and every named parameter for the headline families |
| Stochastic training | Mixtral, MiniMax-M2 and DBRX using shared Philox jitter draws in HF's otherwise unchanged forward/autograd |
| Jitter GPU regression | Scalar Philox reference, step replay, evaluation bypass, preservation of normalization input, finite-difference input gradients |

Every forward check in the table above passed at the `2e-7` absolute-only
ceiling; the whole registered forward set is re-qualified there (worst observed
`7.450580597e-8`). Two-step AdamW checks now cover every decoder-only fixture
family and observed parameter errors at most `1.192092896e-7`.
Kimi K3 and SmolLM3 remain capped at `2e-7` absolute error; the current SmolLM3
default/YaRN qualification observed at most `5.960464478e-8`.
These are FP32
tiny-model correctness checks, not performance claims or certification of
large checkpoints, mixed precision, every hyperparameter, generation policy,
loss, or optimizer. Stochastic checks control jitter randomness rather than
expecting native PyTorch RNG sequences to match Vulkan's.

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib
cargo test --manifest-path hierarchos-inference/Cargo.toml --lib
cargo test --manifest-path hierarchos-native-cli/Cargo.toml --lib
```

Rust GPU tests follow the existing convention of returning early when no device
is available; the Python numerical probes require an actual Vulkan device.
Run those probes when hardware execution is an acceptance requirement.

### Watchdog-safe submissions (Linux AMD, and the ported Intel Gen9 target)

Linux AMD training protects the stock `amdgpu` scheduler timeout by
retiring compute dispatches as dispatch-granular synchronous Vulkan jobs instead of
allowing an entire forward/backward/optimizer phase to accumulate into one
long-running submission. The policy is enabled automatically for AMD PCI vendor
`0x1002` on Linux and is deliberately not enabled by default on other
platforms or vendors.

The same failure class is bound by the Windows display driver's timer on the
Intel Gen9/Gen9.5 target, so that pattern was ported onto exactly the part class
the vendor layer already singles out (`vendor::is_intel_gen9_pci_id`, the same
IDs the FP16 LM reliability policy uses) on Windows -- at a *measured,
length-based* slice point rather than at the AMD target's per-dispatch one,
because porting the latter unchanged was measured costing about 5x per training
step here while buying no headroom the submission needed. A driver watchdog
bounds a submitted job in *time*, so the bound is not a count chosen once: every
retirement is timed on the host, and the budget becomes the count whose measured
cost fits `GEN9_WATCHDOG_SLICE_MICROS` (400 ms), inside a window bounded below by
the 1024 dispatches the port shipped at and above by a mis-measurement guard of
8192. The target is the safety bound and the guard is not: a training step of the
canonical fixture is 320-350 ms of measured work at the rate its own retirements
measure, so a target below one step forces a retirement on every step for a
submission that is already five times inside the driver's timeout, and a guard at
8192 cannot bind a submission the measurement has already priced (the step is
6,368 dispatches). What the guard still bounds is the composite in which a cheap
sample sizes the submission and every dispatch after it is the slowest class
measured here (`layer_norm_low_rank_producer_forward_fused`, 175.0 us):
8192 x 175.0 us is 1.43 s, and that class is 2.3% of a step's dispatches, so no
real graph composes a submission that way. A sample can only *widen* the bound, so
the policy costs at most what the fixed count cost, and the widest measurement a
device has taken is published for the batches after it, so a bound learned while
the graph warms up is not re-learned on every step. The measurement is below.
Every other
combination keeps the legacy single-submit path: other Intel generations, other
vendors, platforms outside each policy's own, and AMD outside Linux. The policy
reproduces no vendor variant and touches no shader: it changes submission
granularity, not shader math or optimizer ordering.

What the slice point costs and whether it buys anything here, measured on the
tuning target itself (Intel HD Graphics 520, `0x8086:0x1916`, driver
31.0.101.2115, Windows, debug build) with `training_step_probe --batch 2
--tokens 8 --steps 6`, every arm interleaved in five rounds so the machine's own
drift reaches every arm equally, median of the rounds:

| Arm | Median step | vs legacy |
| --- | --- | --- |
| legacy single submit | 381.8 ms | -- |
| 8192 dispatches, forced (what the measurement converges to) | 381.7 ms | -0.0% |
| **shipped default (measured bound, 400 ms target)** | **382.7 ms** | **+0.2%** |
| 4096 dispatches, forced (the port's earlier ceiling) | 385.1-399.5 ms | +1.2 to +1.8% |
| 1024 dispatches, forced (the fixed count the port shipped) | 405.6 ms | +5.7% |
| 512 dispatches, forced | 426.1 ms | +12% |
| 256 dispatches, forced | 449.7 ms | +18% |
| 64 dispatches, forced | 556.7 ms | +46% |
| 1 dispatch (the AMD slice point) | 1928.3 ms | +407% |

The last four rows are the earlier forced sweep, whose legacy step was 380.0 ms in
that session; their ratios reproduce, while their absolute milliseconds belong to
the session that took them. The first five rows are one interleaved session, which
is what makes them comparable.

The shipped arm's +0.2% is one *calibration* retirement per process: the budget
starts at the floor, the first retirement measures the stream, and the bound then
sits at the guard for the rest of the process, which is the -0.0% row. On a
seven-step probe that is one retirement amortized over seven steps; on a real run
it is amortized over thousands and is 0.00%. The fixed 1,024-dispatch count this
replaced costs +5.7% on the same fixture, and an intermediate configuration whose
guard was 4,096 dispatches -- below the fixture's own 6,368-dispatch step, so it
could never accept a step in one submission -- measured +0.9% before it was fixed.
That is the whole reason the guard is 8,192 and the target is 400 ms: a bound that
binds a submission the measurement has already priced is a cost with no protection
attached to it.

The policy is a *net speedup*, not merely a recovered regression, on the second
fixture: `cargo test --lib -- hierarchos_causal_lm_` (six tests, ten seconds of
real training work) runs in 10.28 s with the measured bound live against 10.49 s
with slicing off (`HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0`), median of three
interleaved repetitions each -- about 2% faster than the same binary's unsliced
path, because retiring a slice lets the GPU start on it while the host is still
recording the next one. On this fixture and the probe together, the port is
between -2% and +0.2% of the pre-port unsliced path rather than the +407% its
per-dispatch reading cost.
`validation/slice_point_ab.py` is the tool that takes the step numbers: it runs the
arms round-robin and then hashes the golden's own three cases under every arm,
failing if any arm disagrees.

So the per-dispatch slice point is what made this port unaffordable at about 5x
per step, and a *fixed* length bound still cost about 6% while meaning something:
at this fixture's measured per-dispatch cost 1024 dispatches is roughly 60 ms of
work, and it holds a submission under a three-second watchdog for every
per-dispatch cost measured here except the single-dispatch attention-backward
class that only the kernel-level split can shorten. The transformer recorders also
retire at their phase boundaries whatever the budget says, so a step's submission
count is the larger of its phase count and its dispatch count over the budget.

Every arm is **bit-exact**, and now by hash instead of by comparison: running the
golden's own three cases under the legacy single-submit path, under the measured
default and under a forced 1024-dispatch point produces the same three trajectory
dumps byte for byte -- `95f48308...` (`batch2`), `6c97733e...` (`batch32`),
`cc4cb821...` (`odd_multi_sequence`) -- the ladder's recorded-trajectory check
passes on all three arms (13 steps, 3,817 exact tensor digests each), the library
suite is 717 passed / 0 failed with the policy live, and
`watchdog_submission_slicing_preserves_dependent_dispatch_results` executes a
sliced stream with a data dependency across the slice boundary on the adapter
that is actually present.

Retiring a submission is a change of granularity, so the structural counters a
caller reads still describe the whole recorded stream: `dispatch_count`,
`shader_barrier_count`, `pipeline_bind_count`, `descriptor_bind_count`, the
push-constant writes and the allocated descriptor sets are carried across a
retirement, and only the since-retirement dispatch count drives the budget. That
is more than bookkeeping -- `validation/verify_training_stack.py` compares
`dispatch_count` against the expectation recorded for the device it ran on, and
reporting only the open slice made a sliced run look like it had recorded 224
dispatches where the fixture records 6,368. With the counters carried, that check
reports the same 6,368 on both arms, which is what makes it a check of the graph
rather than of the submission policy. A split does cost one extra barrier across
each slice boundary (2,983 versus 2,980 on the two-step probe), inside the
recorded upper bound.

The Gen9 slice point is therefore ported, wired and **shipped on**:
`GEN9_WATCHDOG_SLICING_DEFAULT` is `true` for this class at the measured +0.5%
per step, and `HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0` restores the legacy
single-submit path for anyone who wants that half point back. What decided it
against the per-dispatch reading, and then against a fixed count, was not taste
but the measurements above, and the constants that carry them --
`GEN9_WATCHDOG_SLICE_DISPATCHES`, its ceiling and `GEN9_WATCHDOG_SLICE_MICROS` --
live together in the source.

### The Gen9 target's own training-state trajectory

The golden is device-keyed, so a machine it has no trajectory for cannot be
verified against it -- which is why every claim above started as a live two-run
comparison. `validation/kernel_golden/training_state.json` now carries the Gen9
target's trajectory for all three cases (13 training steps, 3,817 exact tensor
digests), recorded through the legacy single-submit path so the stored baseline
is the un-sliced one, and `validation/record_training_state.py` is the recorder
that added it: it runs the probe exactly as the verifier does, adds one device's
trajectories without rewriting any other device's, refuses unless every case ran
and reported the same device, and refuses to overwrite an existing trajectory
without `--replace`. Stripping the new entry reproduces the previous file
byte-for-byte, so the AMD trajectories are untouched.

With that recorded, the A/B is checked against a repository artifact in both
directions instead of against two live runs:

| Arm | `validation/verify_training_state.py` |
| --- | --- |
| policy at its default (measured bound) | PASS, 13 steps, 3,817 digests |
| `HIERARCHOS_VULKAN_WATCHDOG_SLICE_DISPATCHES=1024` (the fixed count it replaced) | PASS, 13 steps, 3,817 digests |
| `HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0` | PASS, 13 steps, 3,817 digests |

The device trajectories are an input artifact, so
`validation/release_parity/source_inventory.json` still holds this file's
previous hash; that inventory is a release-audit snapshot and already disagrees
with this working tree on 470 of its 1,005 entries, so it is refreshed by the
release flow rather than by hand here.

The ladder's other fixture record, `validation/training_step_probe/step_expectation.json`,
is now device-keyed for the same reason the trajectory golden is. The graph this
target builds is not the AMD one -- the vendor layer selects Gen9 kernels here --
so the same probe arguments dispatch a different number of times: 6,368 on this
part against the 6,464 recorded on the AMD target, on **both** arms, which is what
says the difference is the graph and not the slicing policy. While
`dispatch_count` sat at the top level as one number "compared for every device",
the second device could only fail that comparison or overwrite the first device's
record when re-recorded. It now lives in the device entry beside `mean_loss`,
`--record-expectation` merges this machine's entry and leaves every other entry
as recorded, and the barrier bound is device-independent and only ever widens. AMD's
entry keeps `6,464` and `shader_barrier_count_max` keeps `3,076`; the Gen9 entry
records `6,368` and its own loss, and the training-step probe passes on both arms
against it.

The port is only real if the shipped constructors act on it, so
`watchdog_slicing_is_wired_into_the_training_path_on_this_adapter` asserts that
the plain constructor never slices a caller's stream unasked while
`ComputeBatch::new_with_persistent_upload_arena` -- the one the training graph
builds its phases with -- comes out slicing exactly when the pure policy says so
on the running adapter, with the same dispatch budget the policy resolves. Where
that adapter is this class, the same test asserts both directions of the slice
point, so the wiring cannot pass vacuously on a machine the policy does not
cover. The vendor/device/platform matrix is asserted without a device by
`gpu_job_boundary_policy_defaults_to_amd_and_scopes_the_gen9_port` and
`gpu_job_boundary_policy_honors_explicit_override`, the granularity by
`watchdog_slice_budget_is_per_dispatch_on_amd_and_length_based_on_gen9`, and the
gate itself on device by `watchdog_slice_budget_retires_on_accumulated_length`,
which requires a stream below the budget to stay batched, the budget's dispatch
to retire, the replacement batch to keep the policy and its granularity, and a
phase boundary to retire without waiting for the budget. The measured bound
itself is asserted twice more: without a device by
`watchdog_slice_budget_is_a_measured_bound_inside_a_measured_window` (the
extrapolation, both clamps, the unusable samples, and the sample that must *not*
tighten a wider budget) and on device by
`watchdog_slice_budget_follows_the_measured_retirement_on_this_adapter`, which
requires a real retirement of cheap dispatches to leave the floor behind, to stay
inside the window, to publish what it measured for the batches after it, and a
pinned budget to survive the same retirement untouched.

It does not reach a single dispatch that is itself longer than the watchdog:
slicing retires between dispatches, so a submission with one over-long dispatch
in it is shortened only at the kernel level (see the attention-backward
key-range split, whose per-block submissions use the same primitive). The Gen9
resets recorded for the attention backward are that single-dispatch class.

Slicing is activated only on synchronous training/inference command streams;
batches that acquire external or device-group semaphore dependencies retain
their original deferred-submission behavior. On the Gen9 target the ported slice
point is the shipped default, so `HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0` is
what restores the legacy path there, `=1` forces it on any device, and
`HIERARCHOS_VULKAN_WATCHDOG_SLICE_DISPATCHES` chooses the granularity for the
A/B the table above reports.

This changes submission granularity, not shader math or optimizer ordering. The
same path can be forced for validation on any platform with
`HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=1`, and users who have deliberately
raised the kernel watchdog and prefer the legacy larger submissions can restore
them with `HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS=0`.

The safeguard also makes a failed fence/timeline wait terminal for that
submission handle. In particular, a Vulkan device-loss error is propagated
without the handle's destructor entering a second unbounded wait during cleanup.
This addresses the failure mode where a lost training device could otherwise
leave the process consuming a CPU core instead of exiting.

### Vendor-specific kernels

Kernel selection is vendor-aware through `src/vendor.rs`: AMD and every other
vendor keep the portable SPIR-V modules unchanged, while Intel Gen9 integrated
parts select workgroup-tiled matmul modules that were measured on the HD
Graphics 520 target. The admitted Intel families are `linear-forward`,
`linear-bias-forward`, `linear-residual-forward` (RWKV channel-mix/post-mix) and
`linear3-forward` (RWKV time-mix r/k/v); every other family keeps its portable
module because its candidate either measured slower or was not bit-exact
(`shaders/vendor_experiments/`). Selection is driven by the PCI vendor ID plus
capability gates, and the matmul slot (`VendorMatmulKernel`) chooses per
dispatch from the push-constant geometry so single-row decode, short-k, and
thin-n shapes measure no slower than before (the full measured matrix is in the
tuning guide).

The AMD path is not merely "also green" but numerically identical: with
`HIERARCHOS_VULKAN_FORCE_VENDOR=amd` every matmul family resolves to its
exact pre-tuning portable module, and a default (Intel modules selected) and
forced-AMD run of the headline forward set produced the same 21
unmasked/mixed-padding comparisons with identical `max_abs` and zero failing
values on every row (worst `5.96e-8`, exit 0 on both) against the unchanged
`2e-7` ceiling. The force-vendor switch only overrides kernel classification;
the FP16 LM reliability policy remains keyed to the physical device and is
unaffected by it. The
saved-module rows below are computed by the same modules on both vendors, so
they are a vendor-independent oracle-boundary question, not an Intel-specific
gap.
`HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS` forces the portable modules and
`HIERARCHOS_VULKAN_FORCE_VENDOR` overrides the classification for A/B runs.
Every admitted variant is bit-identical to the portable kernel (raw `f32` bits,
stricter than the `2e-7` reference logit-drift ceiling), and
`cargo test --release --lib vendor::` asserts that plus the host dispatch
contract. The same vendor layer also keeps the opt-in packed-FP16 LM adjoint
tranche on the verified FP32-compute arm for Intel Gen9/Gen9.5 parts, where
those kernels fault the driver at production widths
(`HIERARCHOS_VULKAN_FORCE_NATIVE_FP16_LM_COMPUTE` re-qualifies on a newer
driver). The packed-FP16 rows16 cross-row LM candidate arms remain available
everywhere the capability gates admit them: the original `VK_ERROR_DEVICE_LOST`
was traced to the test creating compute pipelines while a command batch was
being recorded, not to a kernel fault, production creates every kernel before
recording, every rows16 module is green on that path, and the regression test
now hoists creation to match (see `VENDOR_TUNING.md`, "Gen9 rows16 driver quirk
(resolved)"). See [VENDOR_TUNING.md](VENDOR_TUNING.md) for the matrix, measured
results, rejected candidates, and the qualification checklist.

### Host ATen vector shape (AVX2 vs AVX-512)

Kernel *variants* are selected by GPU vendor. Two reductions whose oracle is the
host's PyTorch CPU library are selected by **host CPU capability** instead,
because ATen dispatches its vectorized kernels by ISA at run time and the
reduction shape changes with that dispatch:

| Host | Vector width | Narrow-output (`N == 8`) GEMM | Log-softmax fold |
| --- | --- | --- | --- |
| AVX2-only (Gen9 target class) | 8 `f32` | two interleaved fused FMA lanes | 8 interleaved lanes, sequential fold |
| AVX-512 (`avx512f` present) | 16 `f32` | four interleaved lanes, materialized products | 16 interleaved lanes, sequential fold |

Both shapes are correct; only one matches a given machine, and the wrong one is
not merely imprecise. Measured on the AMD Radeon target (Ryzen Z1 Extreme, an
AVX-512 host) with the AVX2 shape applied everywhere: the `saved`-stage
`model.embed_tokens` adjoint moved from `7.45e-9` to `3.22e-6` on Gemma 4 and
from `1.79e-7` to `2.98e-7` on Qwen2.5-sliding-tied, which pushed six
previously green PEFT saved-module fixtures (`gemma3`, `gemma4`, `minimax_m2`,
`minimax_m3`, `smollm3`, `qwen2_5_sliding_tied`) past the `2e-7` gate. The same
kernels are one ulp off in the other direction on an AVX2-only host, so neither
shape can be baked in.

Selection is a capability probe (`transformer::aten_vector_width`), cached once
per process and asserted by `aten_vector_width_follows_host_avx512_support_not_the_gpu_vendor`
and `narrow_gemm_and_log_softmax_kernels_track_the_host_vector_width`. It is not
a GPU-vendor choice: the modules are `linear_forward_lane2` /
`linear_forward_lane4` and the two `transformer_cross_entropy.comp` builds
(`falcon_h1_cross_entropy.spv` at 8 lanes, `falcon_h1_cross_entropy_lanes16.spv`
at 16). `HIERARCHOS_ATEN_VECTOR_WIDTH=8|16` pins the shape for qualification
when the reference wheel's kernels differ from the CPU's capability.

On an AVX2-only host the probe resolves to 8 and the dispatched modules and
parameters are exactly the Gen9-tuned ones, so the Intel target is unchanged —
verified by forcing `HIERARCHOS_ATEN_VECTOR_WIDTH=8` on the AMD machine and
reproducing the AVX2 numbers to the last bit (`3.219e-6`, `8.345e-7`,
`2.086e-7`, `1.639e-7` on Gemma 4, the exact values that run produces). With the
probe resolving to 16 on that same machine, all six fixtures return to the pass
band: Gemma 4's `saved` adjoints fall back to `7.451e-9` / `3.725e-9` /
`7.451e-9`, and Qwen2.5-sliding-tied's `model.embed_tokens` to `1.788e-7` against
a `1.937e-7`–`2.980e-7` AVX2-shaped range.

The same host-ISA choice moves a handful of fixture logits by one to three
`f32` ulps in the *forward* set, in both directions: on the AMD target the
21-comparison headline forward stays green with zero failing values, and its
worst row moves from `8.941e-8` (AVX2 shape) to `1.192e-7` (AVX-512 shape)
against the unchanged `2e-7` ceiling, while Gemma 4's unmasked row *improves*
from `8.941e-8` to `4.657e-8`. Per-family calibration of the narrow shape
(Gemma3/4 on the four-lane module, families whose fixtures measured better on
the two-lane one kept there) is the next refinement; the uniform rule above is
what the current evidence supports, because the saved-module stage is the
surface where a mismatched shape is amplified rather than lost in the noise.

## Remaining parity work

- Hundreds of unregistered architectures still require native graph operations,
  checkpoint loaders, backward passes and reference tests, including hybrid
  state-space models, MLA families, vision/audio towers and specialized models.
- Text-backbone extraction from multimodal packages does not implement their
  image/audio/video processing or non-text tensor inputs.
- AutoModel task heads beyond the implemented language-model paths need their
  own contracts and validation. The audit includes all task mappings so these
  gaps are visible.
- The native runtime is not registered as a PyTorch device or a drop-in backend
  for Python `AutoModel`, `Trainer`, or `pipeline`. Its API is Rust/native CLI.
- The seq2seq facade supports encoding, logits and batch-one generation; a joint
  encoder-and-decoder training facade is still needed. Batched teacher-forced
  evaluation does not add batched autoregressive generation to that facade.
- MoE auxiliary/router losses need execution and numerical coverage beyond the
  cross-entropy objective checked here; parsing their coefficients is not proof
  that they participate in training.
- Quantized caches remain an explicit unsupported path. Paged KV caching is
  available as an opt-in native Vulkan generation policy for
  compatible causal attention layers (`cache_implementation="paged"`). It uses
  a Vulkan-resident logical page table plus lazily grown physical K/V arenas,
  shares beam prefixes, performs copy-on-write for a shared tail page, and
  reuses freed physical pages. The ordinary contiguous native KV cache remains
  the default and special/recurrent cache topologies keep their existing cache
  implementations. The CLI strips package-level paged/continuous serving
  metadata before user overrides are applied, so package defaults cannot enable
  this path implicitly; an explicit CLI/GUI selection or user-supplied generation
  config is required. Decoder-only greedy and multinomial-sampling request batches
  can additionally opt into `continuous_batching_config`: active sequences share
  the per-layer physical paged-KV arenas, finished sequences release pages for
  waiting requests, and each survivor decode round is recorded into one Vulkan
  command submission. Continuous beam search, multiple returned sequences,
  external encoder context, and encoder-decoder request batching currently fail
  closed. The native page size is 16 tokens, so an explicit continuous-batching
  `block_size` must be 16. Only `block_size` and `max_requests_per_batch` are
  currently accepted from the Hugging Face continuous-batching config; other
  allocator, offload, CUDA-graph, or scheduler knobs fail closed. PEFT LoRA covers
  `bias="lora_only"`, DoRA, RS-LoRA scaling,
  per-module `rank_pattern` / `alpha_pattern`, and
  `modules_to_save` for every ordinary linear exposed by the native PEFT module
  topology, using Hugging Face suffix matching and fused-QKV layout conversion.
  `lm_head` is also supported for both untied and tied checkpoints; tied heads
  are cloned before adapter training so the frozen input embedding remains
  independent, matching PEFT's `ModulesToSaveWrapper` behavior. `bias="all"`
  has complete load/train/export coverage for BERT-family graphs,
  GPT-2/GPT-SW3, dense Llama-layout graphs, and GPT-OSS including its router
  bias plus packed `gate_up_proj_bias` / `down_proj_bias` expert parameters.
  Other sparse-MoE and architecture families still reject that mode until their
  complete trainable bias-name topology is wired and round-trip tested.
  The shared adapter-local `modules_to_save` bank also covers qualified
  LayerNorm/RMSNorm, `lm_head`, and input-embedding replacements in addition to
  ordinary Linear modules. All 32 documented native-green fixtures have independently
  passed the replacement-state gate, including named-adapter switching,
  disable/base restoration, save/reload, gradients, and AdamW state. Gemma3's
  saved-embedding gradients are A `1.1920928955078125e-7` / B
  `1.7881393432617188e-7`; Gemma4's are A `7.450580596923828e-9` / B
  `2.9802322387695312e-8`. Frozen-base and resume drift are exactly `0.0`.
  Registry LayerNorm includes canonical RMSNorm replacements, not arbitrary
  internal/per-head norms. Targets outside the canonical replacement graph
  remain unsupported rather than being approximated.

  PEFT qualification is a separate post-base-parity gate. A newly green base
  architecture must first expose its canonical PEFT module graph/names and
  declare any genuine exclusions, then pass the common PEFT logits, gradients,
  two-step AdamW, save/reload/resume, switching/disable, and merge/export checks
  before it can be advertised as PEFT-qualified. Base parity or registry
  presence by itself does not confer PEFT support.

Keep the existing Hierarchos graph and its checkpoint ABI covered while adding
these features. Adding aliases alone does not complete any of these milestones.
