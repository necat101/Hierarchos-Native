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
