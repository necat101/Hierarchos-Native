# Compatibility and validation

This is a shared native Rust/Vulkan backend for Hierarchos and selected Hugging
Face Transformer graphs. Universal feature parity with the Python Transformers
library has **not** been reached. Model-type registration, task implementation,
and numerical validation are separate milestones.

## Headline: nine strict two-step AdamW checks

The strongest compatibility claim in this backend is intentionally small and
reproducible: nine current/high-value Transformer text graphs have live Vulkan
forward parity and pass **two full AdamW steps** against the local Hugging Face
Transformers source tree, with cross-entropy loss checked at each step and every
named trainable parameter compared after export.

Verified on AMD Radeon Graphics against local Transformers `5.16.0.dev0`:

| Family | Verified scope | Max abs parameter error after 2 AdamW steps |
| --- | --- | ---: |
| DeepSeek V4 | causal LM | `1.192092896e-7` |
| Phi-4 Multimodal | text backbone / causal LM | `1.192092896e-7` |
| Phi-3 | causal LM | `1.192092896e-7` |
| Kimi K2.5 | text backbone / causal LM | `1.192092896e-7` |
| Kimi K3 / KimiLinear | text backbone / causal LM | `5.963374861e-8` |
| Mistral 4 | causal LM | `2.607703209e-8` |
| MiniMax M3 | text backbone / causal LM | `3.539025784e-8` |
| Gemma 4 | causal LM | `1.220032573e-7` |
| MiniMax M2 | causal LM | `1.192092896e-7` |

The acceptance ceiling is `2e-7`; it is enforced by
`verify_hf_training.py`, not rounded into the documentation after the fact.
The matching forward suite covers both unmasked batch-one and mixed
left/right-padded batch-two inputs for the legacy headline families and passed
at `atol=rtol=2e-4`. Kimi K3's Moonshot oracle currently exercises the dense
unmasked sequence path and is held to the stricter `atol=2e-7, rtol=0` contract;
its mixed KDA/MLA text fixture observed `5.215406418e-8` maximum logit error.

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

The validator also carries additional verified and in-progress fixtures rather
than inflating the headline number with registry aliases:

- Gemma 3 remains source-backed and passes its two-step AdamW check
  (`1.797452569e-7` observed maximum parameter error), but the headline slot now
  prioritizes Gemma 4.
- Qwen3.5 full attention is not yet counted; its current forward mismatch is
  materially larger (`1.046035439e-2` maximum absolute error in the tiny
  unmasked fixture).
- Qwen4 experimental linear attention is close but still fails the padded
  fixture (`2.943556756e-4` maximum absolute error), while its experimental
  sparse-attention path remains farther out (`4.634071141e-3` observed
  unmasked maximum absolute error).
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
| Headline forward logits | DeepSeek V4, Phi-4 text, Phi-3, Kimi K2.5 text, Kimi K3/KimiLinear text, Mistral 4, MiniMax M3 text, Gemma 4, MiniMax M2 |
| Additional forward logits | GPT-2, Llama, Qwen2, Mistral, Mixtral, DBRX, BERT, T5, BART, Switch Transformers |
| Masking and batching | Unmasked batch one and mixed left/right padding in batch two; visible-token logits |
| Encoder-decoder | Independent source/decoder lengths; Switch encoder has three sparse layers, decoder has a different two-layer schedule |
| Headline training and export | Exactly two AdamW steps; cross-entropy loss and every named parameter for the nine headline families |
| Stochastic training | Mixtral, MiniMax-M2 and DBRX using shared Philox jitter draws in HF's otherwise unchanged forward/autograd |
| Jitter GPU regression | Scalar Philox reference, step replay, evaluation bypass, preservation of normalization input, finite-difference input gradients |

The headline forward checks passed at their documented tolerances; the largest
observed absolute logit error among the nine was `1.937150955e-7` (Gemma 4,
unmasked), while Kimi K3 is independently capped at `2e-7` absolute error.
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
  Non-linear `modules_to_save` targets and other generation/PEFT options
  still need native parameter routing and broader checkpoint-level reference
  tests before full parity can be claimed.

Keep the existing Hierarchos graph and its checkpoint ABI covered while adding
these features. Adding aliases alone does not complete any of these milestones.
