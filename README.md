<div align="center">

<h1>Hierarchos Native</h1>

**Train, fine-tune, and run language models with Rust + Vulkan.**

[![Workspace check](https://github.com/necat101/Hierarchos-Native/actions/workflows/workspace-check.yml/badge.svg?branch=main)](https://github.com/necat101/Hierarchos-Native/actions/workflows/workspace-check.yml)
[![Native Rust and Vulkan](https://img.shields.io/badge/native-Rust%20%2B%20Vulkan-d97706?style=flat-square)](#how-it-works)
[![Strict CPU logit parity](https://img.shields.io/badge/CPU%20logit%20drift-%3C%202e--7-2563eb?style=flat-square)](#validation)

[Quick start](#quick-start) · [Architectures](#supported-architectures) · [Documentation](#documentation) · [Validation](#validation)

**[☕ Support the developer on Buy Me a Coffee](https://buymeacoffee.com/netcat420)**

</div>

Hierarchos Native runs supported Hugging Face text models and **Hierarchos coherent-v9**
through a native stack. Rust handles packages, tokenization, datasets and checkpoints;
Vulkan handles supported Transformer training and generation. Use the CLI directly
or launch the same workflows from the desktop GUI.

**No Python, PyTorch, CUDA, ROCm or oneAPI installation is required by the native runtime.**
PyTorch and Transformers serve as development oracles for numerical qualification.

| **146** | **166/166** | **72/72** | **< `2e-7`** |
|:---:|:---:|:---:|:---:|
| registered text architectures | RDNA3 forward runs passed | RDNA3 generation requests passed | strict CPU logit drift gate |

<sub>Registry breadth and measured qualification are separate: the November 2, 2026 RDNA3 matrix covers 43 scaled profiles / 26 architecture identifiers, in resident and streamed modes. The CPU gate uses `rtol=0`. [Evidence and limits](hierarchos-vulkan/validation/RDNA3_GPT2_CLOSURE.md).</sub>

---

## Quick start

Install Rust/Cargo and a Vulkan compute driver, then build from source:

```powershell
git clone https://github.com/necat101/Hierarchos-Native.git
cd Hierarchos-Native

cargo build --release --manifest-path hierarchos-native-cli/Cargo.toml
cargo build --release --manifest-path hierarchos-vulkan/Cargo.toml `
  --bin hierarchos-vulkan-train `
  --bin hierarchos-vulkan-devices `
  --bin hierarchos-vulkan-transformer-train `
  --bin hierarchos-vulkan-transformer-logits

$cli = ".\hierarchos-native-cli\target\release\hierarchos-native-cli.exe"
& $cli doctor
& $cli devices
& $cli architectures
```

**Try a small model** through the native Hub downloader:

```powershell
& $cli infer --hf-model hf-internal-testing/tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 32 --no-do-sample
```

This randomly initialized fixture exercises the execution path.
Use `--model-path .\model` for a supported local package.
The examples below reuse `$cli`. On non-Windows systems, use the corresponding
executable path without `.exe` and your shell's line-continuation syntax.

Checked-in SPIR-V binaries are included; ordinary builds do not need a shader compiler.
[Build requirements](#getting-started) · [CLI reference](hierarchos-native-cli/README.md) · [Portable bundle](#portable-standalone-bundle)

---

## Table of contents

- [Why Hierarchos Native](#why-hierarchos-native)
- [How it works](#how-it-works)
- [Supported architectures](#supported-architectures)
- [Training and inference](#training-and-inference)
- [Streaming and offloading](#streaming-and-offloading)
- [Desktop GUI](#desktop-gui)
- [Validation](#validation)
- [Performance](#performance)
- [Getting started](#getting-started)
- [Project status](#project-status)
- [Documentation](#documentation)
- [License and notices](#license-and-notices)
- [Community and support](#community-and-support)

---

## Why Hierarchos Native

- **One native execution path.** The CLI and GUI use the same Rust/Vulkan model contracts.
- **Train as well as infer.** Supported graphs provide forward/backward, AdamW,
  full-parameter training and independently qualified LoRA/PEFT paths.
- **Keep standard model packages.** Load and export canonical Hugging Face tensor
  names and shapes using single-file or indexed, sharded SafeTensors.
- **Choose how frozen weights are stored.** Keep parameters resident, or explicitly
  stream supported projections, MoE experts and vocabulary from disk.
- **Use consumer Vulkan hardware.** Physical qualification includes Intel HD Graphics
  520 and AMD RDNA3, with separate device and revision receipts.
- **Check the numbers.** Frozen CPU references, shader provenance and training-state
  comparisons back the published qualification claims.

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## How it works

The Rust frontend resolves the model configuration, loads tokenizer and weight assets,
prepares data, and dispatches the implemented native graph. Transformer math runs in
Vulkan compute shaders; coherent-v9 also has a separate pure-Rust inference engine.

| Component | Responsibility |
| --- | --- |
| [Native CLI](hierarchos-native-cli/) | Model/tokenizer/dataset acquisition, training commands, generation and checkpoint orchestration |
| [Vulkan backend](hierarchos-vulkan/) | Native model graphs, forward/backward, AdamW, adapters, caches and streaming |
| [Pure-Rust inference](hierarchos-inference/) | Hierarchos coherent-v9 inference from native SafeTensors packages |
| [Desktop GUI](hierarchos-gui/) | Mode-aware forms, command previews and live output over the same CLI |

Model exports stay separate from optimizer/RNG continuation sidecars. A saved model
or adapter is an interchange artifact; exact native continuation additionally needs
its training state. The native path does not register a Python `transformers` backend
or execute remote model/dataset code.

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Supported architectures

The source-derived registry contains **146 canonical text architectures**, **83 package
aliases**, and **229 advertised `model_type` spellings**. It includes dense and MoE
decoders, encoder families, encoder-decoder families and implemented hybrid graphs.

| Native path | Highlights |
| --- | --- |
| **DeepSeek V4.1 Flash** | Dedicated `deepseek_v41` graph; hyper-connections, compression, shared KV/index state, Engram, routed experts and native LoRA/rsLoRA |
| **Kimi K3 / KimiLinear** | Native mixed KDA + MLA text backbone, SiTU, Stable LatentMoE, AttnRes, cached generation and AdamW |
| **Falcon H1 / H1R** | Parallel GQA/RoPE attention + Mamba2, full training, fine-tuning and qualified PEFT |
| **Llama, Qwen, Gemma, Mistral and others** | Family-specific text graphs and package configuration handling; see the generated inventory for individual contracts |
| **Hierarchos coherent-v9** | Native training/fine-tuning, replay state and pure-Rust inference/chat |

Inspect the live inventory and its separate PEFT capabilities:

```powershell
& $cli architectures --json
& $cli architectures --peft --json
```

**Start with the [generated architecture matrix](hierarchos-vulkan/README_ARCHITECTURES.md)
and [compatibility record](hierarchos-vulkan/COMPATIBILITY.md) for a particular model.**
Architecture selection comes from the package configuration; no architecture-specific
enable flag is needed. Registry presence does not imply every mode has passed its
independent numerical gate.

> [!IMPORTANT]
> A supported text backbone in a multimodal package does not imply native image,
> audio, video or processor execution. Unsupported graphs, formats and execution
> contracts fail closed. Scaled-checkpoint qualification does not certify arbitrary
> production checkpoints or every precision policy.

DeepSeek V4.1 qualification uses the accepted production-ratio, reduced-width
checkpoints; full production weights and CUDA are not required or claimed.
Kimi K3 coverage is for the text backbone; its official packed MXFP4 representation
remains outside the qualified path. See the [V4.1 audit](PROGRESS_DEEPSEEK_V41_AUDIT.md)
and [K3 audit](PROGRESS_K3_AUDIT.md) for the detailed contracts.

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Training and inference

Use a supported local package containing `config.json`, tokenizer assets and
SafeTensors weights, or use `--hf-model OWNER/REPO` for native Hub acquisition.
Private/gated packages can use `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.

### Full-parameter training

```powershell
& $cli transformer-train `
  --model-path .\model --train .\dataset.jsonl --out-dir .\trained `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 5e-5 --device-index 0
```

For continued full-parameter fine-tuning, use `transformer-finetune --full-finetune`.
Starting from exported model weights starts a new optimizer; it is distinct from
restoring exact training state.

### LoRA training, save/resume and inference

```powershell
& $cli transformer-finetune `
  --model-path .\model --train .\dataset.jsonl --out-dir .\adapter `
  --lora-rank 8 --lora-alpha 16 `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 1e-4 --device-index 0

& $cli transformer-finetune `
  --model-path .\model --train .\dataset.jsonl --out-dir .\adapter-resumed `
  --peft-resume .\adapter\training-state `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 1e-4 --device-index 0

& $cli infer --model-path .\model --peft-adapter .\adapter-resumed `
  --prompt "Explain Vulkan compute." --max-new-tokens 128 --device-index 0
```

PEFT is the default `transformer-finetune` mode. It exports `adapter_config.json`,
`adapter_model.safetensors` and native `training-state/` sidecars, preserving the
frozen base. Resume restores adapter tensors, AdamW moments/steps and dropout RNG;
it does not restore a high-level dataset cursor. Keep the same base and compatible
data, batch, sequence and optimizer settings.

Targets, saved modules and merge support are architecture-specific. DeepSeek V4.1
supports additive unmerged LoRA/rsLoRA and continuation/export; packed-base merging
and `modules_to_save` remain unqualified. Check `architectures --peft --json` and
the [PEFT guide](hierarchos-native-cli/README.md#peft-fine-tuning-guide) before
choosing replacement modules or merging an adapter.

### Dataset input

JSONL supports text rows such as `{"text":"your training example"}` and pretokenized
`input_ids`. Prompt/completion and Alpaca-style data have explicit CLI options.
The Rust frontend also handles supported JSONL/NDJSON and Parquet Hub shards.
[Dataset and training examples](hierarchos-native-cli/README.md).

### Hierarchos coherent-v9

Use `train` and `finetune` for native coherent-v9 training, and `chat` for pure-Rust
inference. Packages retain canonical FP32 master tensors plus separate native
continuation state. [CLI workflows](hierarchos-native-cli/README.md) ·
[Inference contract](hierarchos-inference/README.md).

### Falcon H1 / Falcon H1R-7B: native training and full fine-tuning

<details>
<summary><strong>Complete Falcon CLI/GUI examples and qualification scope</strong></summary>

`tiiuae/Falcon-H1R-7B` uses the canonical `falcon_h1` architecture. Each native
layer executes **parallel GQA/RoPE attention and Mamba2**, followed by the gated
MLP. Config-driven MuP/scaling, chunk prefill, selective recurrent decode,
KV/conv/SSM cache, backward, AdamW, and canonical SafeTensors save/reload run in
Rust/Vulkan. Python and Transformers are only the validation oracle.

The deterministic tiny fixtures pass the fixed absolute `2e-7` gate: forward
logits at most `8.940696716e-8`, cached logits at most `7.450580597e-8` across
five prefill lengths, and both AdamW trajectories at most `1.192092896e-7`
parameter drift. Every checked cached token matches native full-prefix logits.
These are fixture-level results, not a measured parity or performance claim
for all 7B weights. The official config and shard index were checked; the full
7B checkpoint was not downloaded for qualification. Full-parameter training
uses FP32 parameters, gradients and optimizer state and needs substantially
more memory than the roughly 15.2 GB BF16 checkpoint on disk.

The CLI and GUI command-builder integration also pass train → reload → full
fine-tune → save → generate checks. With nonzero weight decay, cumulative
parameter drift after training plus fine-tuning is `1.620501280e-7`; GUI and
CLI exports are bit-identical. This integration checks the GUI launch contract,
not automated visual interaction with its window.

After building the CLI as above, train/fine-tune the pretrained checkpoint:

```powershell
# Full-parameter training from the official Hub package.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-train `
  --hf-model tiiuae/Falcon-H1R-7B `
  --train .\falcon-data.jsonl --out-dir .\falcon-trained `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 5e-5 --device-index 0

# Continue with domain/instruction data, updating all parameters.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe transformer-finetune `
  --full-finetune --model-path .\falcon-trained `
  --train .\falcon-domain.jsonl --out-dir .\falcon-finetuned `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 1e-5 --device-index 0

# Reload the exported package and generate using the hybrid cache.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe infer `
  --model-path .\falcon-finetuned --prompt "Explain your reasoning step by step." `
  --max-new-tokens 128 --use-cache --no-do-sample --device-index 0
```

Use `--model-path C:\models\Falcon-H1R-7B` in place of `--hf-model` for an
existing package containing `config.json`, `tokenizer.json`, and either
`model.safetensors` or its shard index and shards. JSONL data can contain one
`{"text":"your training example"}` per line, or pretokenized `input_ids`.
Prompt/completion data can use `--prompt-column prompt --completion-column completion`;
Alpaca instruction/input/output rows can use `--alpaca`. Exported outputs are
complete model packages; starting another training command reloads weights
and starts a new optimizer, rather than resuming optimizer moments.

For the GUI, build and launch it with the matching CLI:

```powershell
cargo build --release --manifest-path hierarchos-gui/Cargo.toml
$env:HIERARCHOS_NATIVE_CLI = (Resolve-Path .\hierarchos-native-cli\target\release\hierarchos-native-cli.exe).Path
.\hierarchos-gui\target\release\hierarchos-native.exe
```

Choose **Transformer full training** or **Transformer full-parameter fine-tuning**.
Enter the local model folder, or enter `tiiuae/Falcon-H1R-7B` and check the
Hugging Face model-ID option. Select the JSONL training file, a separate output
folder, epochs, batch size, sequence length, learning rate and device; then
select **Start training**. For generation choose **Transformer inference / generation**,
point Model at the saved output, enter a prompt and choose **Contiguous native KV**
to use the combined KV/conv/SSM cache. The command preview shows the exact CLI call.

Falcon H1/H1R also supports qualified PEFT/LoRA on its canonical Linear
subset, including Mamba `in_proj`; HF PEFT excludes Mamba `out_proj` and
`conv1d`. Use explicit targets rather than `all-linear`. See the
[native PEFT guide](hierarchos-native-cli/README.md#peft-fine-tuning-guide).
Full fine-tuning cannot be combined with adapter flags.
See [PROGRESS_FALCON_H1_AUDIT.md](PROGRESS_FALCON_H1_AUDIT.md) for exact measurements,
commands, regression results and qualification limits.

</details>

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Streaming and offloading

**Parameters stay resident by default.** Add `--stream-parameters` to supported
inference or frozen-base LoRA commands to select package-backed frozen projections,
MoE experts and vocabulary:

```powershell
& $cli infer --model-path .\model --stream-parameters `
  --prompt "Explain mixture-of-experts routing." --max-new-tokens 128
```

The GUI exposes the same choice as **Stream frozen parameters and MoE experts from
disk**, unchecked by default. Mutable full-parameter training keeps weights resident;
streaming supports the frozen base used by PEFT. Norms, biases, activations and adapter
state remain resident.

Ordinary Llama, Mistral, Qwen2 and Mixtral packages avoid resident body-matrix
construction during loading. Other generic loaders retain their initial-loading
peak. Device residency, offload and cache controls have separate contracts; consult
the [CLI guide](hierarchos-native-cli/README.md) and [compatibility record](hierarchos-vulkan/COMPATIBILITY.md)
for the selected architecture and operation.

Resident and streamed outputs match word-for-word in the current **166-run forward**
and **72-request generation** matrices.
[RDNA3 closure](hierarchos-vulkan/validation/RDNA3_GPT2_CLOSURE.md) ·
[Gen9 qualification](hierarchos-vulkan/validation/GEN9_FINAL_RECEIPT.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Desktop GUI

```powershell
cargo build --release --manifest-path hierarchos-gui/Cargo.toml
$env:HIERARCHOS_NATIVE_CLI = (Resolve-Path $cli).Path
.\hierarchos-gui\target\release\hierarchos-native.exe
```

Choose Transformer training, full-parameter fine-tuning, LoRA, generation, or the
Hierarchos training/chat modes. The form shows the exact CLI command before launch
and streams the child process's output.

Contiguous native KV is the established cache path. **Paged Vulkan KV** and continuous
batching are explicit opt-ins for compatible graphs. The continuous scheduler supports
decoder-only greedy/sampling requests with 16-token pages; unsupported serving
settings fail closed. [GUI controls and packaging](hierarchos-gui/README.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Validation

**The CPU logit requirement is strict: `max_abs < 2e-7`, `rtol=0`.** The frozen inputs
and expectations stay unchanged when a kernel is repaired or optimized. Base-model
parity and PEFT qualification are separate gates.

| Physical qualification | Recorded result | Evidence |
| --- | --- | --- |
| **AMD RDNA3 · November 2, 2026** | 166/166 forward runs and 72/72 generation requests; worst forward drift `1.7881393432617188e-7` | [Closure receipt](hierarchos-vulkan/validation/RDNA3_GPT2_CLOSURE.md) |
| **AMD native regression gate** | 269 selected native tests, 50 GPT-2 tests and 41 CLI tests; resident/streamed CPU PEFT checks pass | [Logs, hashes and archive](hierarchos-vulkan/validation/release_parity/rdna3_gpt2_closure/20261102/qualification.json) |
| **DeepSeek V4.1 on AMD** | 201,714 default-gate logits, worst `1.4901161193847656e-7`; 11,184 explicit package/training logits bit-exact | [V4.1 results](hierarchos-vulkan/validation/RDNA3_GPT2_CLOSURE.md#measured-gates) |
| **Intel HD Graphics 520 · November 1, 2026** | 166/166 forward runs, 72/72 generation requests and 269 selected native tests | [Gen9 receipt](hierarchos-vulkan/validation/GEN9_FINAL_RECEIPT.md) |
| **Gen9 preservation in the AMD fix** | All 585 existing shader binaries unchanged; Intel pipeline-selection guard passes | [Preservation scope](hierarchos-vulkan/validation/RDNA3_GPT2_CLOSURE.md) |

These are retained scalable-checkpoint results. The AMD follow-up did not rerun on
physical Intel hardware. Test filters overlap, so the counts are not additive unique
test totals. The measured architecture matrix covers 26 identifiers, not the entire
146-entry registry; all-profile CPU PEFT training and production throughput are
outside that matrix. [Current validation scope](VALIDATION_STATUS.md).

<details>
<summary><strong>Earlier two-step CPU training qualifications: fourteen architecture families</strong></summary>

These earlier training records retain their linked device/revision scope; they are
separate from the current RDNA3 inference matrix above.

The headline compatibility target is strict numerical parity, not just architecture-name recognition. The current Vulkan backend has **fourteen modern Transformer text architectures/families** that each pass **two full AdamW steps** against the reference implementation with **less than `2e-7` maximum absolute parameter divergence** after export. **Falcon H1/H1R** leads the list: its parallel Attention+Mamba2 graph passes full-parameter training, full fine-tuning, and qualified LoRA/`modules_to_save` PEFT through the same CLI:

| Architecture | Verified native scope | Max abs parameter error after 2 AdamW steps |
| --- | --- | ---: |
| **Falcon H1 / H1R** | causal LM with parallel GQA/RoPE attention + Mamba2; full training, fine-tuning, LoRA, saved modules | `1.192092896e-7` |
| **DeepSeek V4** | causal LM | `1.192092896e-7` |
| **Phi-4 Multimodal** | text backbone / causal LM | `1.192092896e-7` |
| **Phi-3** | causal LM | `1.192092896e-7` |
| **Kimi K2.5** | text backbone / causal LM | `1.192092896e-7` |
| **Kimi K3 / KimiLinear** | text backbone / causal LM | **`5.963374861e-8`** |
| **`gpt_oss`** | causal LM | `1.192092896e-7` |
| **SmolLM3** | causal LM; mixed RoPE/NoPE and YaRN | `1.192092896e-7` |
| **Qwen3.5** | dense, Gated DeltaNet, hybrid, and MoE text causal LM | `2.980232239e-8` |
| **Qwen4 Experimental** | DeltaNet, sparse QSA, GR/PLE, hybrid, and MoE text causal LM | `2.980232239e-8` |
| **Mistral 4** | causal LM | `2.607703209e-8` |
| **MiniMax M3** | text backbone / causal LM | `3.539025784e-8` |
| **Gemma 4** | causal LM | `1.220032573e-7` |
| **MiniMax M2** | causal LM | `1.192092896e-7` |

The acceptance ceiling is enforced by the validation harness rather than rounded into the documentation after the fact. These are deterministic FP32 tiny-model correctness checks covering cross-entropy loss and every named trainable parameter across two optimizer steps; they are not a claim that every arbitrary production checkpoint, precision mode, or hyperparameter combination has been certified. See [the compatibility and validation record](hierarchos-vulkan/COMPATIBILITY.md) for the exact harness and broader parity results.

</details>

Run the source-level workspace gate, or use the device runbook to consume the frozen
CPU references without installing Torch:

```powershell
.\check-workspace.ps1 -AllTargets
```

[RDNA3 reproduction](RDNA3_REGRESSION.md) · [Gen9 evidence](hierarchos-vulkan/validation/GEN9_FINAL_RECEIPT.md) ·
[Fixture provenance](hierarchos-vulkan/validation/V41_FIXTURE_PROVENANCE.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Performance

Optimization is measured per device and shape, with numerical gates checked alongside
timings. Kernel speedups and small-checkpoint timings are not production-model
throughput claims.

| Recorded measurement | Result | Scope |
| --- | --- | --- |
| **Gen9 Llama-wide streaming** | **11.016 → 4.597 ms**, 58.3% lower median; 19 → 4 submissions | Five alternating rounds against the preceding pass |
| **Eight Gen9 architecture controls** | Streaming medians **42.4–78.4% lower**; resident medians **3.5–32.1% lower** | Combined optimization passes against the documented baseline |
| **AMD attention backward** | Batch-128 fixture **263.8 → ~71 ms** with bit-identical gradients | Individual-kernel regression repair |

[Gen9 measurements, memory bounds and tradeoffs](hierarchos-vulkan/validation/GEN9_FINAL_RECEIPT.md) ·
[Attention-backward evidence](hierarchos-vulkan/validation/ATTENTION_BACKWARD_PERFORMANCE.md) ·
[Vendor tuning and rejected routes](hierarchos-vulkan/VENDOR_TUNING.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Getting started

| Requirement | Notes |
| --- | --- |
| **Rust and Cargo** | Use the checked-in manifests and lockfiles |
| **Vulkan loader and compute driver** | Required for GPU training and native Transformer generation |
| **Model assets** | A supported config, tokenizer and single-file or sharded SafeTensors package; alternatively a supported Hub model ID |
| **Python / PyTorch** | Development/reference validation only; not native runtime dependencies |
| **Shader compiler** | Needed when changing shaders, not for ordinary builds using checked-in SPIR-V |

The documented device qualification is Windows-based. The backend uses Rust and
Vulkan; non-Windows builds use their platform's driver and executable conventions.
Availability on a Vulkan device is separate from recorded numerical qualification.

### Portable standalone bundle

Stage all four native crates and the user-facing binaries into one Windows bundle:

```powershell
.\hierarchos-vulkan\package-standalone.ps1 `
  -OutDir C:\path\to\hierarchos-native-backend
```

Use `-SkipBuild` for a source-only bundle. Keep the GUI and CLI together, and include
the Vulkan training companions for Hierarchos training, or set
`HIERARCHOS_VULKAN_BIN_DIR`. [Distribution layout](hierarchos-vulkan/standalone/README.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Project status

| Area | Current scope |
| --- | --- |
| **Development** | Active source development; build and qualification evidence are revision-specific |
| **DeepSeek V4.1 Flash** | Dedicated native text implementation, scaled inference/PEFT gates, save/resume/export and opt-in parameter/MoE streaming |
| **Latest AMD closure** | The padded GPT-2-wide gap is closed at the unchanged CPU threshold; original frozen references preserved |
| **Intel Gen9** | Completed physical port and optimization receipt; the AMD parity repair preserves existing Gen9 shader bytes and selection |
| **Compatibility boundaries** | Text-backbone support is not universal multimodal/task-head compatibility; formats, training modes and precision policies need their own gates |

Use the live `architectures` command for package recognition and the linked receipts
for numerical coverage. [Status and limitations](VALIDATION_STATUS.md) ·
[Compatibility](hierarchos-vulkan/COMPATIBILITY.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

### Multimodal training development

Native CLIP/LLaVA image processing, projector training and the shared modality
embedding bridge are under development. See the
[multimodal training audit](PROGRESS_MULTIMODAL_TRAINING.md) for measured scope,
strict parity results and the remaining architecture work.

## Documentation

| Start here | Reference |
| --- | --- |
| Commands, datasets, PEFT and continuation | [Native CLI guide](hierarchos-native-cli/README.md) |
| Native graphs, shaders and backend tooling | [Vulkan backend](hierarchos-vulkan/README.md) |
| Model recognition and execution limits | [Architecture inventory](hierarchos-vulkan/README_ARCHITECTURES.md) · [Compatibility](hierarchos-vulkan/COMPATIBILITY.md) |
| Desktop workflows | [GUI guide](hierarchos-gui/README.md) |
| Hierarchos coherent-v9 inference | [Pure-Rust runtime](hierarchos-inference/README.md) |
| Current qualification and reproduction | [Validation status](VALIDATION_STATUS.md) · [RDNA3 runbook](RDNA3_REGRESSION.md) |
| Device performance and numerical routing | [Vendor tuning](hierarchos-vulkan/VENDOR_TUNING.md) · [AMD audit](AMD_REGRESSION_AUDIT.md) |
| Original physical ROG Ally qualification | [Qualification](RDNA3_QUALIFICATION.md) · [Handoff](ROG_ALLY_HANDOFF.md) |
| Implementation history | [DeepSeek V4.1 audit](PROGRESS_DEEPSEEK_V41_AUDIT.md) · [Kimi K3 audit](PROGRESS_K3_AUDIT.md) |

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## License and notices

The native crate manifests declare **MIT** licensing; the standalone distribution
includes the [MIT license text](hierarchos-vulkan/standalone/LICENSE).
Third-party code and reference artifacts retain their own notices:
[third-party notices](THIRD_PARTY_NOTICES.md) · [license files](licenses/).

---

## Community and support

Found a bug or want to contribute a device result? [Open an issue](https://github.com/necat101/Hierarchos-Native/issues)
with the model/config, device and driver, exact command, and relevant logs. Numerical
changes should preserve the frozen references and pass the applicable parity gates.
A GitHub star helps others find the project.

<div align="center">

### Support native GPU development

If Hierarchos Native is useful to you, you can support continued kernel optimization,
architecture implementation and hardware qualification.

**[☕ Buy me a coffee — netcat420](https://buymeacoffee.com/netcat420)**

[Support on Patreon](https://www.patreon.com/cw/MakhiBurroughs) · [GitHub issues](https://github.com/necat101/Hierarchos-Native/issues)

</div>
