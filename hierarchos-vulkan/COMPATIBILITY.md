# Compatibility and validation

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
checked token; the harness also requires actual recurrent dispatches.
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

### Linux AMD watchdog-safe submissions

Linux AMD training now protects the stock `amdgpu` scheduler timeout by
retiring compute dispatches as dispatch-granular synchronous Vulkan jobs instead of
allowing an entire forward/backward/optimizer phase to accumulate into one
long-running submission. The policy is enabled automatically for AMD PCI vendor
`0x1002` on Linux and is deliberately not enabled by default on other
platforms or vendors.

Slicing is activated only on synchronous training/inference command streams;
batches that acquire external or device-group semaphore dependencies retain
their original deferred-submission behavior.

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
