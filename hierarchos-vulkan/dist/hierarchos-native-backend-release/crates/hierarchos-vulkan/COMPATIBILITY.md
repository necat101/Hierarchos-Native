# Compatibility and validation

This is a shared native Rust/Vulkan backend for Hierarchos and selected Hugging
Face Transformer graphs. Universal feature parity with the Python Transformers
library has **not** been reached. Model-type registration, task implementation,
and numerical validation are separate milestones.

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

Use Python dependencies compatible with the adjacent checkout. The reference
scripts import `../src/transformers` and include its version/path in their output,
so an older system installation cannot silently become the reference. Rust,
Cargo and a Vulkan compute device are required; model fixtures are tiny and
created locally, with no pretrained-model downloads.

An isolated Windows setup reusing an existing PyTorch installation is:

```powershell
python -m venv --system-site-packages .venv-vulkan
.\.venv-vulkan\Scripts\python.exe -m pip install 'huggingface-hub>=1.5.0,<2.0' 'tokenizers>=0.23.1,<0.24.0' 'safetensors>=0.8.0'
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_logits.py
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_training.py
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/verify_hf_training.py --jitter
```

On other platforms use the environment's Python executable. Cargo reports the
probe executable path directly, including custom target directories. Both
scripts accept `--device-index` and `--families` to select a device or subset.

Verified on AMD Radeon Graphics with this checkout reporting `5.16.0.dev0`:

| Check | Scope |
| --- | --- |
| Forward logits | GPT-2, Llama, Qwen2, Mistral, Mixtral, MiniMax-M2, DBRX, BERT, T5, BART, Switch Transformers |
| Masking and batching | Unmasked batch one and mixed left/right padding in batch two; visible-token logits |
| Encoder-decoder | Independent source/decoder lengths; Switch encoder has three sparse layers, decoder has a different two-layer schedule |
| Training and export | Two AdamW steps; cross-entropy loss and every named parameter for eight non-seq2seq families above |
| Stochastic training | Mixtral, MiniMax-M2 and DBRX using shared Philox jitter draws in HF's otherwise unchanged forward/autograd |
| Jitter GPU regression | Scalar Philox reference, step replay, evaluation bypass, preservation of normalization input, finite-difference input gradients |

The forward checks passed at `atol=rtol=2e-4`; observed maximum absolute error
was below `2e-6`. The two-step training checks observed parameter errors below
`2e-7`. These are FP32 tiny-model correctness checks, not performance claims or
certification of large checkpoints, mixed precision, every hyperparameter,
generation policy, loss, or optimizer. Training comparisons use zero weight
decay; stochastic checks control jitter randomness rather than expecting native
PyTorch RNG sequences to match Vulkan's.

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
