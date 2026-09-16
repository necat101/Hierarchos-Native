# Hierarchos Native

Native Rust + Vulkan training and inference tooling for **Hierarchos coherent-v9** and a broad set of modern Hugging Face Transformer text architectures.

Hierarchos Native is built around a framework-free execution path: Rust handles model/package I/O, Hugging Face downloads, tokenization, datasets, checkpointing, and orchestration, while supported Transformer and Hierarchos training math runs through Vulkan compute shaders. Hierarchos inference has a separate pure-Rust runtime.

The current source-derived Transformer registry contains **143 canonical native architectures**, plus **81 Hugging Face package/config aliases** for **224 advertised `model_type` spellings**. The full generated inventory is in [hierarchos-vulkan/README_ARCHITECTURES.md](hierarchos-vulkan/README_ARCHITECTURES.md).

> [!IMPORTANT]
> Architecture support means the declared native **text graph** is implemented by this backend. A supported text backbone inside a multimodal/audio/vision package does not imply that the package's image, audio, video, processor, or other non-text towers execute natively. Unsupported graphs fail closed instead of being silently approximated.

## What is in this repository?

| Component | Purpose |
| --- | --- |
| [`hierarchos-vulkan`](hierarchos-vulkan/) | Native Vulkan training backend for Hierarchos and supported Transformer graphs. Includes forward/backward, AdamW, dropout, LoRA/PEFT paths, generation, checkpoint interchange, shaders, validation tools, and low-level utilities. |
| [`hierarchos-native-cli`](hierarchos-native-cli/) | Unified native CLI for Transformer training/fine-tuning/generation and Hierarchos training/fine-tuning/chat. Also handles Rust-native Hugging Face model/tokenizer/dataset acquisition. |
| [`hierarchos-inference`](hierarchos-inference/) | Pure-Rust Hierarchos coherent-v9 inference runtime using SafeTensors model packages. |
| [`hierarchos-gui`](hierarchos-gui/) | Native desktop frontend over the same CLI/runtime contracts. It does not contain a second model implementation. |

The native Transformer path is not a PyTorch device plugin and does not register itself as a Python `transformers` backend. Its public integration surface is Rust plus the native CLI.

## Highlights

- Raw Vulkan compute through Rust [`ash`](https://crates.io/crates/ash) for supported training and generation paths.
- No CUDA toolkit, PyTorch, Python runtime, or vendor-specific compute API is required by the native backend itself.
- Native Hugging Face Hub model, tokenizer, and dataset acquisition from Rust.
- SafeTensors-compatible model interchange with canonical PyTorch-style tensor names/layouts.
- Full-model Transformer training and native LoRA fine-tuning.
- Causal, bidirectional, and implemented encoder-decoder Transformer graph families.
- Dense and routed-MoE model families, including modern Llama, Qwen, Gemma, Mistral/Mixtral, DeepSeek, GLM, OLMo, Granite, Cohere, Phi, T5/BART-family, BERT-family, and many others.
- Greedy, sampling, and beam-style Transformer generation policies, including native KV-cached paths for qualified causal families.
- Native Hierarchos coherent-v9 training, checkpoint/resume state, mixed-precision policies, multi-device scheduling, and pure-Rust inference.
- Checked-in SPIR-V binaries for ordinary builds; a shader compiler is not required just to build and run the release backend.
- Explicit numerical/parity tooling against reference PyTorch/Transformers implementations for development qualification.

## Requirements

For normal native use you need:

- Rust and Cargo compatible with the checked-in lockfiles.
- A Vulkan 1.x loader/driver.
- A compute-capable Vulkan device for GPU training and native Transformer generation.

Python and PyTorch are used only by optional development/reference validation scripts. They are not runtime dependencies of the native training backend.

## Quick start

Clone the repository:

```powershell
git clone https://github.com/necat101/Hierarchos-Native.git
cd Hierarchos-Native
```

Build the unified CLI and the Vulkan companion binaries:

```powershell
cargo build --release --manifest-path hierarchos-native-cli/Cargo.toml

cargo build --release --manifest-path hierarchos-vulkan/Cargo.toml `
  --bin hierarchos-vulkan-train `
  --bin hierarchos-vulkan-devices `
  --bin hierarchos-vulkan-transformer-train `
  --bin hierarchos-vulkan-transformer-logits
```

Then inspect the machine and the live architecture registry:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe doctor
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe devices
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe architectures
```

On non-Windows systems, omit the `.exe` suffix and use your shell's normal line-continuation syntax.

### Portable standalone bundle

On Windows, the repository can stage the four native crates into one standalone Cargo workspace and build the user-facing binaries into one `bin/` directory:

```powershell
.\hierarchos-vulkan\package-standalone.ps1 `
  -OutDir C:\path\to\hierarchos-native-backend
```

Use `-SkipBuild` for a source-only bundle. The resulting package does not depend on a local Python Transformers checkout at build time or runtime.

## Transformer workflows

### Train a local Hugging Face model package

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-train `
  --model-path .\hf_model `
  --train .\dataset.jsonl `
  --out-dir .\transformer_trained `
  --epochs 1 `
  --batch-size 1 `
  --seq-len 128 `
  --lr 5e-5 `
  --device-index 0
```

### Train directly from Hugging Face Hub assets

The native CLI can download supported models and datasets without invoking Python, `huggingface_hub`, Git LFS, or a framework loader:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-train `
  --hf-model hf-internal-testing/tiny-random-gpt2 `
  --hf-dataset polinaeterna/jsonl_test `
  --hf-dataset-file data/train.jsonl `
  --out-dir .\transformer_hf `
  --epochs 1 `
  --batch-size 1 `
  --seq-len 128 `
  --seed 42 `
  --device-index 0
```

Private or gated Hub repositories can use `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`. Model, tokenizer, and dataset revisions can be pinned independently.

### Native LoRA fine-tuning

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-finetune `
  --model-path .\hf_model `
  --train .\dataset.jsonl `
  --out-dir .\transformer_lora `
  --lora-rank 8 `
  --lora-alpha 16 `
  --device-index 0
```

LoRA forward, backward, and AdamW math remain in the Vulkan graph. `peft-rs` is used for PEFT schema/key interoperability rather than as a framework execution backend.

### Generate from a trained/local model

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe infer `
  --model-path .\transformer_trained `
  --prompt "Explain Vulkan compute in one paragraph." `
  --max-new-tokens 160 `
  --do-sample `
  --temperature 0.7 `
  --top-p 0.9 `
  --device-index 0
```

You can also point `infer` at a supported Hub package with `--hf-model OWNER/REPO`.

## Hierarchos workflows

### Train or resume coherent-v9

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --model-path .\hierarchos_model `
  --train .\dataset.jsonl `
  --out-dir .\trained `
  --epochs 3 `
  --batch_size 4 `
  --accumulation-steps 4 `
  --starting-lr 1e-4 `
  --training-chunk-size 256 `
  --precision fp32
```

Fresh coherent-v9 initialization can use a normal Hub tokenizer:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --hf-tokenizer openai-community/gpt2 `
  --train .\dataset.jsonl `
  --out-dir .\fresh_native `
  --context_dim 448 `
  --h_hidden 448 `
  --l_hidden 448 `
  --rwkv-head-size 64
```

### Fine-tune Hierarchos

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe finetune `
  --model-path .\trained `
  --train .\domain_dataset.jsonl `
  --out-dir .\finetuned `
  --epochs 1 `
  --batch_size 4 `
  --accumulation-steps 4 `
  --starting-lr 1e-5 `
  --training-chunk-size 256 `
  --precision fp32
```

### Chat through the pure-Rust inference runtime

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe chat `
  --model-path .\finetuned `
  --carry-chat-state `
  --chat-state-file .\chat-state.json
```

## Dataset input

The native preprocessing path accepts common JSONL shapes including:

- `text` / `content`
- `instruction` + `output`
- `prompt` + `completion`
- `question` + `answer`

The CLI can also discover JSONL/NDJSON and Parquet shards from Hugging Face dataset repositories and materialize them through the Rust frontend. It intentionally does not execute remote dataset builder scripts or arbitrary remote code.

For low-level pre-tokenized Hierarchos training, `hierarchos-vulkan-train` accepts JSONL rows containing `input_ids` and optional same-length `labels`, `attention_mask`, and `loss_weights`.

## Supported Transformer architectures

Do not rely on a hand-maintained list copied into this front page. The authoritative inventory is generated from `VulkanTransformerArchitecture::ALL` and its explicit aliases:

```powershell
python hierarchos-vulkan/validation/generate_supported_architectures.py --check
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe architectures --json
```

At the current revision the registry reports:

- **143 canonical native Transformer text architectures**
- **81 package/config aliases**
- **224 advertised `model_type` spellings total**

See [the generated architecture matrix](hierarchos-vulkan/README_ARCHITECTURES.md) for the complete list.

The registry includes dense decoders, encoder families, encoder-decoder families, and routed-MoE families. It also resolves supported text backbones from a number of multimodal and speech packages while preserving unconsumed tensors on export where supported.

## Checkpoint and interoperability contract

The backend uses standard SafeTensors packages rather than a Vulkan-only weight format. Training operates with canonical FP32 masters and can use qualified lower-precision execution/storage policies internally. Exported model tensors retain normal row-major PyTorch/Hugging Face shapes and names expected by the implemented contracts.

Hierarchos training additionally uses backend-neutral sidecars for exact optimizer/replay state. These files remain separate from ordinary model tensors so the model package can still be consumed by the pure-Rust inference runtime and external PyTorch CPU/CUDA tooling.

Supported Hugging Face Transformer packages may use either a single `model.safetensors` or standard sharded SafeTensors with `model.safetensors.index.json`.

## Desktop GUI

Build the native desktop launcher with:

```powershell
cargo build --release --manifest-path hierarchos-gui/Cargo.toml
```

The GUI exposes Transformer training, Transformer LoRA fine-tuning, Transformer inference/generation, Hierarchos training/fine-tuning, and Hierarchos chat. It is a thin launcher over the same native contracts and shows the generated command before execution.

For a portable Windows bundle, keep `hierarchos-native.exe` and `hierarchos-native-cli.exe` together. Hierarchos training also needs `hierarchos-vulkan-train.exe`, or set `HIERARCHOS_VULKAN_BIN_DIR` to the directory containing the companion Vulkan binaries.

## Validation

The repository keeps registry coverage, Rust tests, and numerical reference checks separate so an architecture name is not confused with a parity claim.

Useful source-level checks include:

```powershell
cargo check --manifest-path hierarchos-vulkan/Cargo.toml --all-targets
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib
cargo test --manifest-path hierarchos-native-cli/Cargo.toml
cargo test --manifest-path hierarchos-gui/Cargo.toml
python -m unittest discover -s hierarchos-vulkan/validation -p "test_*.py"
python hierarchos-vulkan/validation/generate_supported_architectures.py --check
```

Development-only Hugging Face/PyTorch parity probes live under `hierarchos-vulkan/validation/`. See [hierarchos-vulkan/COMPATIBILITY.md](hierarchos-vulkan/COMPATIBILITY.md) for the tested scope, tolerances, and remaining gaps.

## Current compatibility boundaries

Hierarchos Native intentionally fails closed when a requested execution contract is not implemented. Important current boundaries include:

- This is not universal compatibility with every model and every task in Python `transformers`.
- A supported multimodal package alias currently represents its supported text backbone, not automatic native execution of every vision/audio/video tower.
- AutoModel task heads outside the implemented language-model paths need their own native contracts and validation.
- Some hybrid SSM/convolution, sparse/global-attention, MLA/indexer, non-text, quantized-cache, paged-cache, and continuous-batching paths remain outside the current native graph.
- The seq2seq facade has native encoding/logits and implemented generation paths, but the compatibility document should be checked before assuming a particular training/generation mode is covered.
- Reference parity results are tiny-model correctness/qualification evidence, not a blanket performance or large-checkpoint certification.

## Documentation map

- [Vulkan backend internals, milestones, performance work, and parity tooling](hierarchos-vulkan/README.md)
- [Generated Transformer architecture inventory](hierarchos-vulkan/README_ARCHITECTURES.md)
- [Compatibility, numerical checks, and known gaps](hierarchos-vulkan/COMPATIBILITY.md)
- [Unified native CLI](hierarchos-native-cli/README.md)
- [Pure-Rust Hierarchos inference](hierarchos-inference/README.md)
- [Native desktop GUI](hierarchos-gui/README.md)
- [Standalone distribution layout](hierarchos-vulkan/standalone/README.md)

## Project status

This repository is under active development. The architecture inventory and compatibility surface are generated/validated from source and will continue to evolve. When in doubt, prefer the live `architectures` command and the generated architecture matrix over copied architecture counts in third-party descriptions.
