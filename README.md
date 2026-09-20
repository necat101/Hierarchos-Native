# Hierarchos Native

Native Rust + Vulkan training and inference tooling for **Hierarchos coherent-v9** and a broad set of modern Hugging Face Transformer text architectures.

Hierarchos Native is built around a framework-free execution path: Rust handles model/package I/O, Hugging Face downloads, tokenization, datasets, checkpointing, and orchestration, while supported Transformer and Hierarchos training math runs through Vulkan compute shaders. Hierarchos inference has a separate pure-Rust runtime.

## Twelve architectures verified below `2e-7` training divergence

The headline compatibility target is strict numerical parity, not just architecture-name recognition. The current Vulkan backend has **twelve modern Transformer text architectures/families** that each pass **two full AdamW steps** against the reference implementation with **less than `2e-7` maximum absolute parameter divergence** after export:

| Architecture | Verified native scope | Max abs parameter error after 2 AdamW steps |
| --- | --- | ---: |
| **DeepSeek V4** | causal LM | `1.192092896e-7` |
| **Phi-4 Multimodal** | text backbone / causal LM | `1.192092896e-7` |
| **Phi-3** | causal LM | `1.192092896e-7` |
| **Kimi K2.5** | text backbone / causal LM | `1.192092896e-7` |
| **Kimi K3 / KimiLinear** | text backbone / causal LM | **`5.963374861e-8`** |
| **`gpt_oss`** | causal LM | `1.192092896e-7` |
| **Qwen3.5** | dense, Gated DeltaNet, hybrid, and MoE text causal LM | `2.980232239e-8` |
| **Qwen4 Experimental** | DeltaNet, sparse QSA, GR/PLE, hybrid, and MoE text causal LM | `2.980232239e-8` |
| **Mistral 4** | causal LM | `2.607703209e-8` |
| **MiniMax M3** | text backbone / causal LM | `3.539025784e-8` |
| **Gemma 4** | causal LM | `1.220032573e-7` |
| **MiniMax M2** | causal LM | `1.192092896e-7` |

The acceptance ceiling is enforced by the validation harness rather than rounded into the documentation after the fact. These are deterministic FP32 tiny-model correctness checks covering cross-entropy loss and every named trainable parameter across two optimizer steps; they are not a claim that every arbitrary production checkpoint, precision mode, or hyperparameter combination has been certified. See [the compatibility and validation record](hierarchos-vulkan/COMPATIBILITY.md) for the exact harness and broader parity results.

### Newly strict-qualified: GPT‑OSS, Qwen3.5, and Qwen4-Experimental

All three newly qualified families are held to an absolute-only `2e-7` gate (`rtol=0`) against the local Hugging Face Transformers source checkout for forward logits, cached generation, two native AdamW steps, and native-trained SafeTensors save/reload.

- **`gpt_oss`:** forward logits reached `5.960464478e-8` maximum absolute drift, cached generation reached `4.470348358e-8`, two-step AdamW parameter drift reached `1.192092896e-7`, and native-reload versus Transformers-reload logits reached `4.470348358e-8`.

- **Qwen3.5:** dense full attention, pure Gated DeltaNet, the default hybrid schedule, and MoE fixtures reached at most `4.470348358e-8` forward-logit drift, `4.470348358e-8` cached-generation drift, `2.980232239e-8` two-step AdamW parameter drift, and `5.215406418e-8` native-reload versus Transformers-reload logit drift.

- **Qwen4 Experimental:** DeltaNet, genuinely sparse QSA, GR/hyper-connections, PLE n-gram/dilated-convolution state, mixed DeltaNet/QSA scheduling, and MoE/shared-expert fixtures reached at most `2.980232239e-8` forward-logit drift, `3.725290298e-8` cached-generation drift, `2.980232239e-8` two-step AdamW parameter drift, and `2.235174179e-8` native-reload versus Transformers-reload logit drift.

For Qwen3.5 and Qwen4 Experimental, native cached decoding matched native full-prefix decoding exactly (`0.0` maximum drift) at every checked step.

## Kimi K3 / KimiLinear: fully implemented native text backbone

Kimi K3 now has a **fully implemented native `KimiLinear` text-backbone path** instead of being aliased to Kimi K2/K2.5, DeepSeek, Qwen, GLM, or another graph. A top-level `kimi_k3` package resolves its nested `text_config` to the canonical native `kimi_linear` architecture and executes through the Rust/Vulkan backend.

The completed path includes the hybrid **KDA + MLA** layer schedule, independent Q/K/V short convolutions and recurrent KDA state, full-rank KDA output gating, MLA output gating, **SiTU**, **Stable LatentMoE**, **AttnRes**, native forward/backward and AdamW training, cached token-by-token generation, and SafeTensors load/export round-tripping. The production runtime does not depend on Python, PyTorch, FLA, Triton, CUDA, or Moonshot remote code.

Current K3 validation against Moonshot's released Kimi K3 modeling semantics is well inside the `2e-7` target: isolated KDA reaches `5.587935448e-8` maximum absolute logit error, isolated MLA `6.146728992e-8`, the production-style mixed KDA/MLA graph `5.215406418e-8`, and the strict two-step AdamW check `5.963374861e-8` maximum absolute parameter error across 18,450 compared values.

The K3 claim is specifically for the **language/text backbone**. MoonViT/vision/projector execution is intentionally outside this contract. The official `compressed-tensors` `mxfp4-pack-quantized` checkpoint representation is also rejected explicitly until that packed format has its own native implementation and qualification; use an unquantized FP32/BF16 KimiLinear package for native execution.

The current source-derived Transformer registry contains **145 canonical native architectures**, plus **83 Hugging Face package/config aliases** for **228 advertised `model_type` spellings**. The full generated inventory is in [hierarchos-vulkan/README_ARCHITECTURES.md](hierarchos-vulkan/README_ARCHITECTURES.md).

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

### Using the newly strict-qualified families

There is no architecture-specific opt-in flag. Point the normal Transformer commands at a supported local Hugging Face package with `--model-path`, or at a Hub package with `--hf-model`; the native loader resolves the package model type and supported text-wrapper aliases automatically.

The newly qualified model types include `gpt_oss`, Qwen3.5 text/MoE packages (`qwen3_5_text` and `qwen3_5_moe_text` plus supported wrapper aliases), and `qwen4_exp_text`. Hybrid DeltaNet/attention scheduling and the Qwen4 QSA/GR/PLE path are taken from the model config, so users do not manually select those internal layer modes.

Python/PyTorch remains validation-only; it is not required for production training or generation through these native paths. The qualification applies to the implemented text causal-LM path, not unrelated vision/audio towers in a wrapper package.

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

- **145 canonical native Transformer text architectures**
- **83 package/config aliases**
- **228 advertised `model_type` spellings total**

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

Paged KV caching is an opt-in Vulkan generation optimization; the existing
contiguous native KV cache remains the default. From the CLI, add
`--cache-implementation paged` to `transformer-generate` (with caching enabled).
In the GUI's Transformer generation workflow, choose `Paged Vulkan KV` from the
`KV cache` selector. Choosing `Model/package default (non-paged)` preserves the
package's ordinary cache policy but deliberately ignores package-level
`cache_implementation="paged"` and `continuous_batching_config` metadata so a
downloaded model cannot silently opt the user into the new serving path.
`Contiguous native KV` explicitly selects the established native cache and
`Disabled (full-prefix)` disables KV caching. An explicit `--generation-config`
may still request paged/continuous behavior because supplying that file is an
end-user opt-in. An explicit paged request fails closed for generation graphs
that have no pageable K/V attention layers.

Continuous request scheduling is also opt-in. Repeat `--prompt` and add
`--continuous-batching` to `transformer-generate`, or enable `Continuous
batching` in the GUI and provide one additional request per line. This mode
requires paged KV, shares one physical paged-KV arena per pageable Transformer
layer across the active requests, returns finished requests' pages to the shared
free list, admits waiting requests up to the configured active-request limit,
and groups each active decode round into one Vulkan command submission. The
current scheduler supports decoder-only greedy or multinomial sampling with one
returned sequence per request; beam/encoder-decoder continuous batching fails
closed. Native paged KV uses 16-token pages; a supplied
`continuous_batching_config.block_size` must therefore be `16` in this build.
The native continuous scheduler currently accepts only `block_size` and
`max_requests_per_batch` from the Hugging Face continuous-batching config; other
allocator, offload, CUDA-graph, or scheduler knobs fail closed rather than being
silently ignored.

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

The registry integration test needs a local Hugging Face Transformers source
checkout. Set `TRANSFORMERS_CHECKOUT` to that checkout before running the Python
suite, or pass `--transformers-root <path>` directly to
`hierarchos-vulkan/validation/audit_transformers_coverage.py`.

Development-only Hugging Face/PyTorch parity probes live under `hierarchos-vulkan/validation/`. See [hierarchos-vulkan/COMPATIBILITY.md](hierarchos-vulkan/COMPATIBILITY.md) for the tested scope, tolerances, and remaining gaps.

## Current compatibility boundaries

Hierarchos Native intentionally fails closed when a requested execution contract is not implemented. Important current boundaries include:

- This is not universal compatibility with every model and every task in Python `transformers`.
- A supported multimodal package alias currently represents its supported text backbone, not automatic native execution of every vision/audio/video tower.
- AutoModel task heads outside the implemented language-model paths need their own native contracts and validation.
- Some other families' hybrid SSM/convolution, sparse/global-attention, MLA/indexer, non-text, quantized-cache, and advanced continuous-batching paths remain outside the current native graph. Kimi K3's KDA/MLA text path is fully implemented as documented above. Paged KV generation is available only for compatible native Vulkan attention layers and remains explicitly opt-in; the native continuous scheduler is currently limited to decoder-only greedy/sampling requests with paged KV.
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

## Support the Developer:
[Patreon](https://www.patreon.com/cw/MakhiBurroughs)

## Project status

This repository is under active development. The architecture inventory and compatibility surface are generated/validated from source and will continue to evolve. When in doubt, prefer the live `architectures` command and the generated architecture matrix over copied architecture counts in third-party descriptions.
