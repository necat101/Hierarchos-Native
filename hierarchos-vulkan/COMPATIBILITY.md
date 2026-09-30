# Compatibility and validation

## Strict base to PEFT follow-up inventory

This table is the machine-readable inventory consumed by
`validation/verify_peft_green_matrix.py`. Add a row whenever a new family passes
the native base gate; list every independently qualified graph fixture. A new
row automatically becomes required PEFT follow-up work. A passing base row
never promotes the independent PEFT result. On 2026-09-30, the fresh complete
**32/32 all-stage matrix passed**: ordinary LoRA, saved modules, and named-adapter
switching, at the unchanged absolute `max_abs <= 2e-7` gate, with exact frozen
base and resume state. Gemma3 and Gemma4 are included. The oracle imported
`C:\Users\User\transformers` (`5.16.0.dev0`); source/shader/binary/oracle hashes
were unchanged throughout qualification. See [the full audit](../PROGRESS_PEFT_AUDIT.md)
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

The acceptance ceiling is `2e-7`; it is enforced by
`verify_hf_training.py`, not rounded into the documentation after the fact.
The matching forward suite covers both unmasked batch-one and mixed
left/right-padded batch-two inputs for the legacy headline families and passed
at `atol=rtol=2e-4`. Kimi K3's Moonshot oracle, GPT-OSS, and SmolLM3 are held to
the stricter `atol=2e-7, rtol=0` contract. Kimi K3's mixed KDA/MLA text fixture
observed `5.215406418e-8` maximum logit error. GPT-OSS observed
`5.960464478e-8` maximum logit error, while the mixed RoPE/NoPE SmolLM3 fixture
and its YaRN variant each observed at most `5.960464478e-8` across unmasked and
mixed-padding inputs.

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

The headline forward checks passed at their documented tolerances. Kimi K3 and
SmolLM3 are independently capped at `2e-7` absolute error; the current SmolLM3
default/YaRN qualification observed at most `5.960464478e-8`.
The headline two-step training checks observed parameter errors below `2e-7`.
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
