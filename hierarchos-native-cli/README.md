# Hierarchos Native CLI

`hierarchos-native-cli` is the framework-free command-line frontend for the
Hierarchos native stack. It links the pure-Rust inference engine and launches
only Hierarchos' Rust/Vulkan executables for GPU training and device discovery.
There is no Python or PyTorch compatibility dispatcher in this crate. Network
model/tokenizer/data acquisition is also implemented in Rust; Hugging Face Hub
downloads do not invoke `python`, `huggingface_hub`, Git LFS, or a framework
loader.

The canonical interchange boundary is a Hierarchos model package containing
`model.safetensors`, `hierarchos_rust_config.json`, `hierarchos_config.json`, and
local tokenizer assets. FP32 master tensors keep the same names and shapes across
the native trainer and external consumers, while native exact-resume state uses
backend-neutral sidecars owned by `hierarchos-vulkan`.

Build:

```powershell
cargo build --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train --bin hierarchos-vulkan-devices
cargo build --release --manifest-path hierarchos-native-cli/Cargo.toml
```

Typical commands:

Falcon H1 / H1R packages (`falcon_h1`) use `transformer-train` for full training
and `transformer-finetune --full-finetune` for full-parameter fine-tuning.
Both accept `--model-path DIR` or `--hf-model tiiuae/Falcon-H1R-7B`,
`--train DATA.jsonl`, and `--out-dir OUT`. Reload the output with `infer`.
Without `--full-finetune`, `transformer-finetune` selects qualified LoRA on
Falcon's canonical Linear subset, including Mamba `in_proj`; Mamba `out_proj`
and `conv1d` are excluded by HF PEFT. See the root [README](../README.md#falcon-h1--falcon-h1r-7b-native-training-and-full-fine-tuning)
for complete CLI/GUI commands and the strict fixture qualification scope.

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe devices

# Download a published canonical Hierarchos package directly from the Hub.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe pull `
  --repo YOUR_ORG/YOUR_HIERARCHOS_REPO `
  --revision main `
  --out-dir .\hierarchos_from_hf

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --model-path .\hierarchos_model `
  --train .\dataset.jsonl `
  --out-dir .\trained `
  --epochs 3 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --training-chunk-size 256 --precision fp32

# Native assistant-SFT recovery profile. Explicit values still override the preset.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --model-path .\hierarchos_model `
  --train .\alpaca.jsonl `
  --out-dir .\assistant_recovery `
  --assistant-recovery --precision fp32

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe finetune `
  --model-path .\trained `
  --train .\domain_dataset.jsonl `
  --out-dir .\finetuned `
  --epochs 1 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-5 --training-chunk-size 256 --precision fp32

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe chat `
  --model-path .\finetuned `
  --carry-chat-state --chat-state-file .\chat-state.json
```

The same executable also exposes the native Hugging Face Vulkan Transformer
path (causal LM for decoder families and masked LM for supported bidirectional
families such as EuroBERT):

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-train `
  --model-path .\hf_model `
  --train .\dataset.jsonl `
  --out-dir .\transformer_trained `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 5e-5 --device-index 0

# Stock checkpoints with supported non-zero training dropout run directly in Vulkan.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-train `
  --hf-model hf-internal-testing/tiny-random-gpt2 `
  --hf-dataset polinaeterna/jsonl_test `
  --hf-dataset-file data/train.jsonl `
  --out-dir .\transformer_hf_dropout `
  --epochs 1 --batch-size 1 --seq-len 128 --seed 42 --device-index 0

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-finetune `
  --model-path .\hf_model `
  --train .\dataset.jsonl `
  --out-dir .\transformer_lora `
  --lora-rank 8 --lora-alpha 16 `
  --device-index 0

# Exact native continuation restores adapter tensors, AdamW moments/step, and dropout RNG.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-finetune `
  --model-path .\hf_model `
  --train .\dataset.jsonl `
  --out-dir .\transformer_lora_resumed `
  --peft-resume .\transformer_lora\training-state `
  --device-index 0

# Run an unmerged saved adapter directly on the frozen base model.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-generate `
  --model-path .\hf_model `
  --peft-adapter .\transformer_lora_resumed `
  --adapter-name resumed `
  --prompt "Explain Vulkan compute in one paragraph." `
  --max-new-tokens 160 --device-index 0

# Or merge the adapter into a standalone HF-style SafeTensors package.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe merge-lora `
  --model-path .\hf_model `
  --lora-adapter-path .\transformer_lora_resumed `
  --out-dir .\transformer_lora_merged

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-generate `
  --model-path .\transformer_lora_merged `
  --prompt "Explain Vulkan compute in one paragraph." `
  --max-new-tokens 160 --device-index 0

# Native Transformer inference. infer/generate and transformer-infer/
# transformer-generate are equivalent entry points.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe infer `
  --model-path .\transformer_trained `
  --prompt "Explain Vulkan compute in one paragraph." `
  --max-new-tokens 160 --do-sample --temperature 0.7 --top-p 0.9 `
  --device-index 0

# A supported Hugging Face package can also be fetched and generated from directly.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe infer `
  --hf-model hf-internal-testing/tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 32 --no-do-sample
```

`transformer-finetune` defaults to PEFT/LoRA unless `--full-finetune` is supplied. Every PEFT training run saves exact continuation state to `OUT/training-state` unless `--save-lora-training DIR` overrides that location. `transformer-generate --peft-adapter DIR` loads the HF-compatible adapter without merging it first; `--adapter-name NAME` selects its in-memory name. Run `architectures --peft --json` for the independently qualified PEFT families and the separate `modules_to_save_validated` bit. Training and generation both fail closed when a replacement-module adapter targets a family whose saved-module gate is still red.

## PEFT fine-tuning guide

**Qualified on 2026-09-30:** all 32 documented native-green text fixtures pass
LoRA, registered saved modules, and adapter switching against the local
`C:\Users\User\transformers` source clone at absolute `max_abs <= 2e-7`.
Gemma3 and Gemma4 are green, including saved norms/heads/embeddings. Frozen-base
and resume drift remain exactly `0.0`. These deterministic FP32 tiny-model
checks do not certify arbitrary production-size checkpoints, dtypes, dropout
trajectories, devices, or non-text towers. See the
[qualification audit](../PROGRESS_PEFT_AUDIT.md).

The production path is Rust/Vulkan only. Python, Torch, and HF PEFT are used
by the validation oracles, not by these commands. Use `transformer-finetune`
for HF Transformer adapters; the separate `finetune` command trains
coherent-v9's model-defined factors, not arbitrary Transformer LoRA.

### Prepare a base model and data

Use the same base checkpoint and tokenizer for training, resume, and adapter
inference. Local packages need `config.json`, tokenizer assets, and
`model.safetensors` or its standard shard index/shards. `--hf-model OWNER/REPO`
can download a supported package; gated models require `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN`. Model math and text-wrapper support remain
architecture-specific: unrelated vision/audio towers are not PEFT-qualified.

JSONL accepts `{"text":"training example"}` or
`{"input_ids":[1,7,3,11,2]}`. Select prompt/completion columns with
`--prompt-column prompt --completion-column completion`, or use `--alpaca`
for instruction/input/output rows. Supply explicit `--epochs`, `--batch-size`,
`--seq-len`, `--lr`, `--seed`, and `--device-index` for reproducible runs.

### Choose targets and saved modules

Target suffixes must exist in the canonical HF model graph. Start with small,
explicit target sets rather than assuming every architecture has `q_proj`:

| Model graph | Example LoRA suffixes |
| --- | --- |
| Gemma3/4, Llama, Qwen split attention | `q_proj,v_proj` |
| GPT-2 fused attention | `c_attn,c_proj` |
| Phi-3 / Phi-4 Multimodal text | `qkv_proj,o_proj` |
| Kimi K2.5 MLA | `q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj` |
| Kimi K3 KDA/MLA | `o_proj` (common qualified fixture target) |
| Falcon H1/H1R | `q_proj,v_proj,in_proj` (`in_proj` is the Mamba input Linear) |

The [compatibility inventory](../hierarchos-vulkan/COMPATIBILITY.md#strict-base-to-peft-follow-up-inventory)
records all independently tested surfaces and genuine target exclusions. HF
PEFT rejects Falcon H1 Mamba `out_proj` and `conv1d`; do not use `all-linear`
there. Discrete top-k indexer/selector paths and packed expert parameters are
not implied by ordinary Linear qualification.

Use explicit rank and repeat `--lora-modules-to-save NAME` to train full
adapter-local Linear, registered LayerNorm/RMSNorm, head, or embedding
replacements. A Gemma3 norm example:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-finetune `
  --model-path .\gemma3_text --train .\dataset.jsonl --out-dir .\gemma3_adapter `
  --peft --lora-rank 8 --lora-alpha 16 --lora-target-modules q_proj,v_proj `
  --lora-modules-to-save model.layers.0.input_layernorm `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 1e-4 --seed 42 --device-index 0
```

Saved embeddings/heads allocate full trainable copies, not low-rank matrices;
plan memory accordingly. Saved tied-head/embedding behavior follows HF
replacement semantics without mutating the frozen base. Only registered
canonical norm nodes are targetable; an architecture-level norm class does not
mean arbitrary per-head/internal norms can be saved.

### Artifacts and continuation

`OUT/adapter_config.json` and `OUT/adapter_model.safetensors` are the
HF-compatible adapter. `OUT/training-state/` adds native AdamW moments/step
and dropout RNG state. Keep that directory for exact native continuation;
it is not an HF Trainer checkpoint. The frozen base is not copied into OUT.

- `--peft-config FILE`: create a new adapter from HF config, not saved weights.
- `--peft-adapter DIR` / `--lora-adapter-path DIR`: weights-only load/warm start,
  with a fresh optimizer when training.
- `--peft-resume DIR`: restore the native adapter/optimizer/RNG checkpoint;
  use the same base, optimizer, batch/sequence settings, and intended continuation
  data. The frontend does not restore a dataset cursor or epoch scheduler.
- `transformer-generate --peft-adapter DIR`: run the adapter on its frozen base.
- `merge-lora`: produce a standalone model; the current merge path requires
  a single-file base checkpoint. Generate from that output without adapter flags.

Do not combine `--full-finetune` with PEFT flags. HF config files and individual
LoRA configuration overrides are mutually exclusive. Only ordinary LoRA is
qualified across the entire strict inventory; DoRA, RS-LoRA, initializer, bias,
packed-parameter, token-training, and replication options are not promoted by
that result. Unsupported topology/configuration fails explicitly.

### Explicit PEFT configuration and automatic merge

```powershell
hierarchos-native-cli transformer-finetune --model-path .\hf_model `
  --train .\dataset.jsonl --out-dir .\adapter --peft `
  --lora-target-modules q_proj,v_proj --peft-parameter-percent 1 `
  --lora-alpha 16 --lora-dropout 0.05 --lora-merge
```

`--peft-parameter-percent P` chooses the largest integer uniform rank whose
native trainable adapter count fits the requested budget. `1` means one percent.
The denominator is the base checkpoint's serialized floating tensor elements;
a duplicate tied `lm_head` is counted once. This includes serialized floating
buffers and any auxiliary towers present in a multimodal package. The CLI prints
the actual count, rank and achieved percentage before training and checks the
attached count against the budget. A budget below rank 1 fails explicitly.
Automatic rank selection currently requires standard LoRA without bias, DoRA,
saved modules, rank patterns, trainable tokens, packed parameters or replication.
Use explicit rank for those configurations, subject to their qualification gates.

`--peft-config FILE` reads an HF `adapter_config.json` to initialize a new adapter;
`--peft-adapter DIR` loads existing adapter weights and `--peft-resume DIR`
restores native AdamW/RNG state. Config files cannot be mixed with individual
config flags. Direct controls include:

| Flag | HF setting |
| --- | --- |
| `--lora-rank N` / `--lora-r N` / `--lora_r N` | `r` |
| `--lora-alpha N`, `--lora-dropout F`, `--lora-bias MODE` | `lora_alpha`, `lora_dropout`, `bias` |
| `--lora-target-modules q_proj,v_proj` | `target_modules` suffix list; accepts `all-linear` where supported |
| `--lora-target-regex REGEX` | `target_modules` regex string |
| `--lora-exclude-modules NAME,...` | `exclude_modules` |
| `--lora-layers-to-transform 0,2`, `--lora-layers-pattern layers` | layer filtering |
| `--lora-rank-pattern JSON`, `--lora-alpha-pattern JSON` | per-module patterns |
| `--lora-modules-to-save NAME` (repeatable) | adapter-local replacement modules |

`--lora-merge` (alias `--lora-model-merge`) saves the adapter and exact training
state, then exports a standalone package at `OUT/merged-model`. It uses the
existing Rust merge implementation, requires a single-file base checkpoint,
and refuses an existing merge destination. Adapter and training-state files
remain available for continuation even if merging fails. Variant flags remain
subject to the native lifecycle gates; accepting HF config syntax does not
qualify unsupported PEFT methods.

This Transformer command surface is family-specific rather than a claim of
universal `transformers` compatibility. The backend's authoritative registry is
the Rust `VulkanTransformerArchitecture` registry; run `architectures` (or read
`../hierarchos-vulkan/README_ARCHITECTURES.md`) for the generated current list
and counts. Coverage includes GPT,
BERT/RoBERTa/XLM encoders, GPT-2/Neo/NeoX/J/Falcon/BLOOM/MPT/CodeGen/CTRL,
OPT/BioGPT/XGLM, Llama/Mistral/Ministral, Qwen, Phi, GLM, Gemma, Granite,
Cohere, OLMo, StarCoder2, T5/MT5/UMT5, BART/MBART/Marian/Pegasus-style
encoder-decoder models, several routed-MoE families, and their implemented
derivatives. It also recognizes supported text backbones inside multimodal and
speech packages while preserving their untouched auxiliary tensors on export,
including Qwen/LLaVA/Gemma/GLM wrappers plus AudioFlamingo3, Qwen3-ASR, Granite
Speech/Plus, GLM-ASR, MusicFlamingo, and VibeVoice-ASR. Legacy EXAONE-4.5 text
configs using `model_type=exaone4_5_text` are normalized onto the current
canonical EXAONE-4 graph. It preserves the matching Hugging Face SafeTensors
names/layout and tied-vs-untied head behavior.

The native graph covers LayerNorm/RMSNorm, exact and tanh-family GELU, SwiGLU,
RoPE and ALiBi, grouped-query attention, local/sliding attention, relative
position bias, encoder-decoder cross-attention, packed and split QKV layouts,
routed packed experts, and native dropout/backward/AdamW/LoRA paths for the
implemented contracts. Architectures requiring graph primitives not yet present
in the Vulkan runtime (for example hybrid SSM/convolution blocks, MLA/indexer
attention, Longformer/LongT5 sparse-global attention, or full non-text towers)
remain fail-closed instead of being silently approximated.
GPT-J uses its native shared pre-norm residual graph and adjacent-pair partial
RoPE, including its biased LM head and all-projection LoRA surface. StableLM
uses its native LayerNorm/GQA/partial-RoPE/SwiGLU contract, including optional
per-head Q/K LayerNorm and `use_parallel_residual=true` shared-pre-norm training.
Falcon supports its fused MQA/MHA/new-decoder-GQA QKV layouts and residual
topologies. BLOOM adds embedding LayerNorm and native Vulkan ALiBi; MPT adds
bias-free fused `Wqkv`, exact GELU, biasless LayerNorm, native MPT ALiBi, and
PEFT-compatible `Wqkv`/`out_proj`/`up_proj`/`down_proj` LoRA; CodeGen adds
its fused QKV/shared-pre-norm/partial-RoPE contract; Phi adds shared pre-norm,
biased split GQA projections, optional per-head Q/K LayerNorm, partial RoPE, and
its biased LM head. Phi Q/K LayerNorm training and checkpoint export remain in
the Vulkan graph.
Linear and dynamic/NTK RoPE scaling are implemented for the family/config combinations in
the native contract, including Phi-3 LongRoPE forward/backward frequency-table
semantics. Supported Transformer dropout is native Vulkan training math:
attention-probability masks use a reproducible stateless per-layer stream, while
embedding and residual/MLP masks use the backend-neutral Philox4x32-10 counter
contract. Forward/backward replay the same masks without Python RNG state, and
`--seed` selects the stream. `--disable-transformer-dropout` remains an explicit
ablation/debug policy that zeros recognized config fields for the run without
mutating the source/cache; full-model export records the resulting zero values.
OPT whole-layer `layerdrop` remains fail-closed because it is distinct from
tensor dropout. Rust `peft-rs`
supplies the HF LoRA config/key
interoperability contract while LoRA forward/backward/AdamW math stays in
Vulkan. Packages may use one `model.safetensors` file or standard Hugging Face
sharded SafeTensors described by `model.safetensors.index.json`; the Rust
downloader fetches referenced shards and trained full-model export consolidates
them into one canonical FP32 SafeTensors file. SafeTensors export canonicalizes
header metadata ordering, so repeated training or LoRA runs with identical input
and `--seed` produce byte-identical weight artifacts.

The Hub can also be used directly at the training boundary. `--hf-model` pulls
a canonical Hierarchos package, `--hf-tokenizer` pulls standard tokenizer
assets for Rust-only fresh initialization, and `--hf-dataset` discovers
JSONL/NDJSON or Parquet training files from Hub repository metadata. Parquet
shards are decoded and materialized to cached JSONL entirely in Rust before
tokenization. The root CLI spelling
`--tokenizer-path OWNER/REPO` is also recognized automatically when that value
is not an existing local path:

```powershell
# Warm-start a native Vulkan run from a canonical model and dataset on HF.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --hf-model YOUR_ORG/YOUR_HIERARCHOS_REPO `
  --hf-model-revision main `
  --hf-dataset YOUR_ORG/YOUR_DATASET `
  --hf_dataset_split train `
  --hf-dataset-file data/train.jsonl `
  --hf-dataset-revision main `
  --out-dir .\trained `
  --epochs 3 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --training-chunk-size 256 --precision fp32

# Or initialize coherent-v9 from an ordinary Hub tokenizer with no model input.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --hf-tokenizer openai-community/gpt2 `
  --train .\dataset.jsonl `
  --out-dir .\fresh_native `
  --context_dim 448 --h_hidden 448 --l_hidden 448 --rwkv-head-size 64
```

`--hf-cache-dir DIR` relocates the native cache; otherwise it uses
`.hierarchos-hf-cache` beside the repository. `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN` is attached as a bearer token for private/gated repos.
Revisions can be pinned independently with `--hf-model-revision`,
`--hf-tokenizer-revision`, and `--hf-dataset-revision`.

The release path has been exercised end-to-end with an ordinary Hub tokenizer
and with Hub-hosted native datasets: `openai-community/gpt2` can seed fresh
coherent-v9 initialization, while `polinaeterna/jsonl_test` with
`--hf-dataset-file data/train.jsonl` is downloaded, tokenized, and trained without
leaving the native Rust/Vulkan stack.

The native Hub path is deliberately file-oriented and fail-closed. Model pulls
must expose the canonical Hierarchos `model.safetensors`,
`hierarchos_rust_config.json`, `hierarchos_config.json`, and `tokenizer.json`.
Dataset discovery honors `--hf_dataset_config` and `--hf_dataset_split` as
selection hints. If a split consists of multiple JSONL/NDJSON/Parquet shards
they are downloaded in lexical shard order, Parquet is decoded natively, and
the rows are combined into one cached line stream. If discovery is ambiguous,
`--hf-dataset-file PATH` selects the exact file. The native CLI does not execute
remote dataset builder scripts, convert unsupported formats such as CSV, or
execute arbitrary remote code.

Raw JSONL preprocessing is native as well. The frontend recognizes `text`/`content`,
`instruction`/`output`, `prompt`/`completion`, and `question`/`answer` schemas,
appends the tokenizer EOS token, drops blank completions by default, preserves the
prompt suffix plus the beginning of the answer when an example must be truncated,
and supports `--min-response-tokens`, `--allow-empty-completions`, response-boundary
weights, and `--assistant-recovery`. The assistant-recovery preset applies the
supported root-CLI SFT defaults (Alpaca formatting, four epochs, `6e-5` LR,
`0.03` warmup ratio, `0.10/1.0` prompt/response weights, `2x` the first 32
response tokens, 16 reserved answer tokens, `0.003` ponder weight, and a
5000-step fresh-model memory-gate warmup). Independent framework-side LTM
optimizer/value-alignment knobs are not fabricated by the native frontend; use a
schema-v6 token cache for the exact already-tokenized cross-runtime data objective.

Set `HIERARCHOS_VULKAN_BIN_DIR` when the Vulkan trainer/device binaries are
installed outside the repository or are not placed next to the CLI executable.

Native-only behavior is fail-closed. Framework-object `.pt` checkpoints,
framework-style Hugging Face dataset builders, external Python benchmark
registries, and PEFT LoRA geometry injection are not silently delegated. Native
`finetune` trains
coherent-v9's existing recurrent low-rank factors, DeepEmbed/ROSA adapter factors
and routers, and slow-LTM tensors through the same Vulkan forward/backward graph;
repeat `--trainable-prefix` to replace that default selection. `merge-lora` is
implemented in Rust for a bound Hierarchos PEFT SafeTensors adapter package and
emits a standalone canonical model package without importing PEFT, Python, or
PyTorch. Creating a brand-new arbitrary PEFT-LoRA geometry at runtime remains
intentionally unsupported because that would change the canonical architecture.

The high-level native `train` frontend preserves the root CLI's ordinary
training defaults (`epochs=3`, `batch_size=64`, `seed=1337`, `min_lr=1e-6`, and
`ponder_loss_weight=0.01`) while still letting explicit arguments win. Legacy
`--amp` maps to the qualified `fp16-storage-parity` Vulkan policy and `--no-amp`
maps to `fp32`; the native binary never dispatches either option to a framework.
