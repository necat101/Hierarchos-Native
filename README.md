<div align="center">

<h1>Hierarchos Native</h1>

</div>

## A personal note from the developer

I want to share a little context about the person maintaining this project and why community support would mean so much right now. I was recently laid off from my internship and am facing a serious financial shortfall. I currently have about **$190** to my name, with **$725 in rent due by November 30, 2026**. I've been skipping meals to stretch what I have, and without help or a change in circumstances, I may lose my housing and face homelessness beginning **December 1, 2026**.

I also deal with motor-cortex-damage-related coordination issues that affect my ability to drive safely. That limits some in-person work opportunities and may affect the pace and future trajectory of this project as I work to find a sustainable way forward.

If you find Hierarchos Native useful and are in a position to help, please consider making a voluntary contribution. Right now, support would help me cover basic needs like rent and food while also giving me a better chance to keep developing and maintaining the project. I understand that not everyone can donate, and there is absolutely no obligation or expectation. Sharing the project, giving it a GitHub star, reporting issues, or contributing code are also genuinely appreciated.

11/19/2026 2:13AM

Laptop died, I spilled water on it, having been incredibly sleep deprived and depressed, did not even notice it until I went to go power it on. I just watched the screen fade to white and die, I've never cried so hard in my life 3; I feel like someone ran over my dog. This was my first ever x86 computer, and now I killed it.

Intel gen9 development will unfortunately have to halt for the foreseeable future as my financial situation is far from capable of a replacement at the time. work will proceed with RDNA, and eventually, planned QUALCOMM support If i don't go homeless next month.

**[Support on Buy Me a Coffee](https://buymeacoffee.com/netcat420)** · **[Support on Patreon](https://www.patreon.com/cw/MakhiBurroughs)**

---

<div align="center">

**Train, fine-tune, and run language models with Rust + Vulkan.**

[![Workspace check](https://github.com/necat101/Hierarchos-Native/actions/workflows/workspace-check.yml/badge.svg?branch=main)](https://github.com/necat101/Hierarchos-Native/actions/workflows/workspace-check.yml)
[![Native Rust and Vulkan](https://img.shields.io/badge/native-Rust%20%2B%20Vulkan-d97706?style=flat-square)](#how-it-works)
[![Strict CPU logit parity](https://img.shields.io/badge/CPU%20logit%20drift-%3C%202e--7-2563eb?style=flat-square)](#validation)

[Quick start](#quick-start) · [Architectures](#supported-architectures) · [Documentation](#documentation) · [Validation](#validation)


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

**Joint image training:** all 44 previously green text profiles pass the frozen
Gen9 FP32 image-training gate below `2e-7`, including 95 image cases and 166 text
regressions. [Coverage, frozen oracles and Ally replay instructions](RDNA3_MULTIMODAL_HANDOFF.md).

**Phase XVIII replay:** two complete nine-stage producer runs exited 0 on the
qualified source: 77/77 shared-image, 6/6 seq2seq, 4/4 Gemma3, 4/4 LLaVA,
166/166 text, 218/218 active DeepSeek tests (26 opt-in ignored), and 30/30
native V4.1 tests. The complete numerical receipt is sealed at `d799890...`;
the later path-test-only repair is separately bound to `ce1757d...`, with final
workspace validation at `b173653...`. The dedicated RDNA3 GitHub Actions
hardware job remains unavailable because no self-hosted runner is registered;
these are committed qualification results, not a hardware-CI PASS. [Replay and path-repair evidence](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase18-replay-reliability/README.md).

**Final architecture verification (2026-11-08):** the current source
`3f0aaf5` completed one fresh nine-stage replay (9/9 PASS, 507 cases/tests),
the full Rust library suite (**1,139 passed, 40 ignored**), and the batch-1
five-workload ABBA comparison (**5/5 positive**). The same-binary session study
was positive for both ViT workloads but inconclusive for Llama, GPT-2 and T5;
those textual session deltas are not promoted. [Sealed receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-final-architecture-verification/README.md).

---

## Quick start

**Download a prebuilt package:** the **v0.2.8** Windows x64 package is published
with `Hierarchos-Native-Windows-x64-Portable-v0.2.8.zip`, `BUILD_MANIFEST.json`
and `SHA256SUMS`. It contains the CLI, the GUI and the Vulkan companion
executables. Extract it and run `bin\hierarchos-native-cli.exe doctor`. No Rust,
Python or Visual C++ Redistributable is required at runtime — only Windows x64
and a Vulkan 1.x compute driver.
[Jump to the package details](#windows-x64-portable-v028).

Or install Rust/Cargo and a Vulkan compute driver and build from source:

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
- [RDNA3 WMMA (opt-in)](#rdna3-wmma-opt-in)
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

## RDNA3 WMMA (opt-in)

`v0.2.8` promotes two **query-pass** changes into the default training route —
the Phase XVIII literal-index dQ accumulator and the Phase XX scratch-reuse
variant — and leaves the v0.2.6 **opt-in cooperative-matrix (WMMA) path for
RDNA3** unchanged, with the P3 Full FP16 contract and the **C1 batched staging
schedule** as the production route inside that mode. C1 changes only the staging schedule — four
32-bit operand loads are issued before the shared-memory stores that consume
them — and is selected per dispatch only when the host can prove every packed
load aligned and in bounds; every other geometry keeps the P3 module as its
fallback. WMMA remains off by default, and the default FP32 path keeps its
bit-identical logits and its previous-`main` performance (worst paired ratio
`1.0007` across seven workloads, all FP32 logits byte-identical).

Each query-pass promotion is its own paired measurement on one device: the
Phase XVIII series measured 1.042×–1.124× across eight geometries and modes, and
scratch reuse measured 1.088× FP32 and 1.120× Mixed on the primary 8q/2kv
complete step. **They are independent results, not multiplicative factors**,
and no overall training speedup is claimed. The full scope and its limits are in
[the v0.2.8 release note](RDNA3_V028_RELEASE.md).

```powershell
# Default: FP32, no cooperative-matrix feature queried.
& $cli transformer-train --model-path .\model --train .\data.jsonl --out-dir .\out

# Opt in: FP16 operands with FP32 accumulation, FP32 masters and FP32 AdamW.
& $cli transformer-train --model-path .\model --train .\data.jsonl --out-dir .\out --wmma-mode mixed

# What did this device actually negotiate?
& $cli precision-info --device-index 0
```

| Arm | `--wmma-mode` | Operands → accumulation | Masters / AdamW | Status |
| --- | --- | --- | --- | --- |
| FP32 | `off` (default) | FP32 → FP32 | FP32 | Qualified, unchanged |
| Mixed WMMA | `mixed` | FP16 → **FP32** | FP32 | Opt-in, numerically gated |
| Full FP16 | `full-fp16` | FP16 → **FP16** (reductions ≤ 64) | FP32 | Opt-in; P3 contract with C1 staging, CLI-qualified Llama-64 profile |

- **Hardware boundary.** Qualified on AMD RDNA3 PCI **`1002:15bf`** only, on
  Windows x64, with a cooperative `16x16x16` f16/f16→f32 tile. Other RDNA3
  parts, RDNA1/2, Intel Gen9 and multi-GPU device groups are **not** qualified;
  an unsupported explicit request fails with the probed reason instead of
  silently running FP32.
- **Per-dispatch C1 performance.** Against the v0.2.5 P3 binary, the integrated
  production selector is faster on all four protected shapes: `1.7074×`
  (2048×768×2048), `1.3286×` (512×2048×768), `2.0920×` (512×768×768) and
  `1.3782×` (128×8192×768), with every corrected 95% CI lower bound above 1.31.
  These are **per-dispatch GPU GEMM timings**, not end-to-end training-step or
  application speedups. The historical 25% tile16 headroom target remains
  informational and was not achieved.
- **Opt-in remains the default policy.** Mixed is a separate, unchanged arm;
  neither Mixed nor Full FP16 is enabled by default. A measured 512-row Mixed
  training step was slower than FP32 (ratio `0.8708`), so no training-step
  speedup is claimed.
- **Baked module identities.** Both modules are compiled into the CLI,
  transformer trainer and logits executables: P3 (23,280 B, FNV-1a64
  `0x076216ad97d7605f`) and C1 (36,912 B, FNV-1a64 `0xfecb2a74c4bbbce4`). The
  production selector picks between them from the dispatch geometry alone — no
  variant override, environment variable or flag can force either one — and
  reports which ran through per-dispatch staging counters.
- **Numerical identity with v0.2.5.** Eight seeds, 40 receipts, `37/37` gates on
  every seed, and **120/120 production outputs bit-identical** to the v0.2.5 P3
  baseline. No reference, tolerance, seed, oracle or threshold changed; the
  `split2` and `rdna3-dot2` diagnostic failures stay recorded as diagnostics.
- **Fallbacks are not dispatches.** Projections below
  `WMMA_MIN_PROFITABLE_MACS = 16,000,000`, and Full FP16 reductions above 64,
  keep their FP32 kernels and increment fallback counters. The released
  wide-multimodal and native V4.1 fixtures are below the floor and record zero
  cooperative dispatches.

**[Full WMMA guide](WMMA_GUIDE.md)** — modes, hardware admission, copyable
commands, dispatch/fallback semantics, safety, measurements and limitations.
Numerical scope: [release contract](RDNA3_RELEASE_CONTRACT.md). Engineering
history: [WMMA log](RDNA3_WMMA.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Desktop GUI

```powershell
cargo build --release --manifest-path hierarchos-gui/Cargo.toml
$env:HIERARCHOS_NATIVE_CLI = (Resolve-Path $cli).Path
.\hierarchos-gui\target\release\hierarchos-native.exe
```

Choose Transformer training, full-parameter fine-tuning, LoRA, generation, the
Hierarchos training/chat modes, native vision training, image-to-text, or
text-to-image. The form shows the exact CLI command before launch and streams the
child process's output.

Vision training and image-to-text run on the native Vulkan image backends and
accept the model package directly. Text-to-image is a separate workflow: it drives
a pinned [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
Vulkan engine against a Stable Diffusion single-file checkpoint. Install that
engine once with `hierarchos-native-cli/install-image-engine.ps1`, which checks the
archive SHA-256 before extracting. Image-generation numerics are that engine's,
not the text parity contract.

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
| **Phase XVIII complete replay** | Two complete nine-stage runs, 9/9 each: 77/77 shared-image, 6/6 seq2seq, 4/4 Gemma3, 4/4 LLaVA, 166/166 text, 218/218 active DeepSeek (26 ignored), 30/30 native V4.1 | [Phase XVIII receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase18-replay-reliability/README.md) · [path repair](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase18-path-smoke-repair/README.md) |
| **Current-source GPT-2** | 16/16 frozen shapes, worst `5.960464477539063e-08`; current-source guard PASS at `b173653...` | [Current qualification](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase18-path-smoke-repair/gpt2/current-qualification.json) |
| **Final architecture verification** | `3f0aaf5`: 9/9 stages, 507 cases/tests, worst logit `1.7881393432617188e-07`; full Rust suite **1,139 passed / 0 failed / 40 ignored**; batch-1 5/5 positive | [Final receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-final-architecture-verification/README.md) |
| **Chinese families and sparse attention** | `60a14d9`: 37/37 frozen Chinese cases, 218/218 DeepSeek tests, 30/30 native V4.1; worst strict logit `5.960464477539063e-08`; no production changes | [Audit receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-chinese-sparse-attention-audit/README.md) |
| **Full Rust library (Phase XIV)** | 1,135 passed, 40 ignored, 0 failed in three parallel and one serial run | [Phase XIV receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase14-qualification/README.md) |

These are retained scalable-checkpoint results. The AMD follow-up did not rerun on
physical Intel hardware. Test filters overlap, so the counts are not additive unique
test totals. The measured architecture matrix covers 26 identifiers, not the entire
146-entry registry; all-profile CPU PEFT training and production throughput are
outside that matrix. [Current validation scope](VALIDATION_STATUS.md).

### Chinese Model Families & Sparse Attention Qualification

The **2026-11-08 audit at `60a14d9`** checked the existing qualified Chinese-model
paths without reopening the optimization campaign. The frozen shared-image
matrix covers full-parameter image-text training; the native suites add sparse
cache, compressor, backward and export/resume checks.

| Group | Profiles | Cases | Verified sparse / state path | Worst logit |
| --- | ---: | ---: | --- | ---: |
| Qwen2 / Qwen2.5 | 3 | 6 | Dense/GQA control; sliding-window and tied embeddings | `5.960464477539063e-08` |
| Qwen3-Next | 1 | 2 | Linear/full hybrid | `5.960464477539063e-08` |
| Qwen3.5 | 4 | 8 | Full attention, Gated DeltaNet, hybrid and MoE | `5.960464477539063e-08` |
| Qwen4 Experimental | 4 | 8 | QSA compressed-block top-k, GR/PLE caches and MoE | `4.470348358154297e-08` |
| DeepSeek V4 | 1 | 2 | V4 text path | `5.960464477539063e-08` |
| DeepSeek V4.1 Flash | 1 | 218 + 30 native | Compressed top-k, ring cache, compressor and backward | `0.0` logits; `1.7881393432617188e-07` compressor |
| Kimi K2.5 / K3 | 4 | 5 | KDA, MLA and text backbone | `4.470348358154297e-08` |
| MiniMax M2 / M3 | 3 | 6 | Lightning Indexer block selection and local blocks | `5.960464477539063e-08` |
| **Total** | **20 shared + 1 native** | **37 shared + 248 native** | Sparse attention and sparse MoE kept as separate checks | **`5.960464477539063e-08` shared** |

All 37 shared-image cases passed the strict logit gate (`max_abs < 2e-7`,
`rtol=0`); the existing loss gate is `2e-5`, and parameter/gradient checks retain
their canonical metrics. Qwen4 QSA uses a two-token budget with compression ratio
2, complete-block selection and trailing visible tokens; MiniMax M3 uses a
one-block top-k with a local block; DeepSeek V4.1 uses explicit `-1` invalid slots
and window/compressed gathers. The first shared-image invocation used the wrong
executable and failed before comparison; it is preserved as `attempt1`, and the
corrected probe passed 37/37. No production source or shader was changed.

Live Transformers/PyTorch generation, AdamW and roundtrip oracles are **blocked**
because torch cannot load on this host. Other registered-only paths (GLM/GLM4-MoE,
Hunyuan, ERNIE, LongCat, Qwen2-MoE, Qwen3/Qwen3-MoE, DeepSeek V2/V3 and Kimi K2)
remain **registered / not fully qualified** for complete frozen-logit end-to-end
coverage. Batch-4 sparse-specific coverage is
not present. No new performance comparison is claimed; the established
five-workload ViT/Llama/GPT-2/T5 receipt remains separate. See the
[full audit receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-chinese-sparse-attention-audit/README.md)
for logs, hashes and the verifier.

### CI and hardware status

GitHub-hosted `workspace check` and workflow validation pass on the final
`b173653` revision; the path-smoke repair records 5/5 locally and a successful
remote path step. The dedicated RDNA3 performance workflow still has no
registered self-hosted runners, so no RDNA3 hardware CI PASS is claimed.
[Latest workspace run](https://github.com/necat101/Hierarchos-Native/actions/runs/42526469315).
The Gen9 result above is a preserved frozen-reference run; no new physical Intel
execution is claimed on this AMD host.

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

### Matched-profile Phase XIV results

The retained Phase XIV receipt compares release builds from exact revisions
with the same frozen fixtures, batch 1, six ABBA rounds and seven measured
updates. It is a historical matched-profile result for those fixtures, not a
guarantee for every model size or GPU. The receipt records the exact source and
executable hashes for each arm; consult it for revision scope.

| Workload | Speedup versus historical `372407b` | 95% CI |
| --- | ---: | --- |
| T5 | **1.3687×** | 1.2455–1.4275 |
| Native ViT 4×4 | **1.9792×** | 1.8016–2.0542 |
| Native ViT 6×4 | **2.0386×** | 1.8442–2.0698 |
| Llama | **1.1862×** | 1.0629–1.3371 |
| GPT-2 | **1.5150×** | 1.2108–1.5537 |

[Phase XIV performance receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase14-qualification/README.md).

### Final-source matched-profile verification

The final receipt re-ran the same five canonical workloads at source `3f0aaf5`
against the sealed `372407b` baseline (six ABBA rounds, two warmups, seven
measured updates):

| Workload | Speedup versus historical `372407b` | 95% CI |
| --- | ---: | --- |
| T5 | **1.2869×** | 1.1422–1.3827 |
| Native ViT 4×4 | **2.0131×** | 1.9921–2.0330 |
| Native ViT 6×4 | **1.9797×** | 1.9457–2.0174 |
| Llama | **1.1365×** | 1.0789–1.2449 |
| GPT-2 | **1.6069×** | 1.4460–1.8597 |

[Final performance and session evidence](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-final-architecture-verification/README.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Getting started

| Requirement | Notes |
| --- | --- |
| **Rust and Cargo** | Stable toolchain; qualification receipts record Rust 1.93.0. Use the checked-in manifests and lockfiles. |
| **Operating system** | Windows is the documented qualification platform. The Rust/Vulkan source is portable to other desktop Vulkan platforms, but physical qualification is device- and revision-specific. |
| **Vulkan loader and compute driver** | Required for GPU training and native Transformer generation |
| **Model assets** | A supported config, tokenizer and single-file or sharded SafeTensors package; alternatively a supported Hub model ID |
| **Python / PyTorch** | Development/reference validation only; not native runtime dependencies |
| **Shader compiler** | Needed when changing shaders, not for ordinary builds using checked-in SPIR-V |

The documented device qualification is Windows-based. The backend uses Rust and
Vulkan; non-Windows builds use their platform's driver and executable conventions.
Availability on a Vulkan device is separate from recorded numerical qualification.

### Windows x64 portable v0.2.8

**Status: published.** It is built from the qualified source commit of
`release/v0.2.8`. `BUILD_MANIFEST.json` records the exact packaging
commit, toolchain, flags, per-executable hashes, the promoted query artifact
hashes, the retained attention artifact hashes and the P3 and C1 WMMA artifact
hashes. The v0.2.5, v0.2.6 and v0.2.7 assets remain published and unchanged.

| Asset | Platform | Contents |
| --- | --- | --- |
| `Hierarchos-Native-Windows-x64-Portable-v0.2.8.zip` | Windows x64 (`x86_64-pc-windows-msvc`) | `bin\` with `hierarchos-native.exe` (GUI), `hierarchos-native-cli.exe`, `hierarchos-infer.exe`, `hierarchos-vulkan-train.exe`, `hierarchos-vulkan-devices.exe`, `hierarchos-vulkan-transformer-train.exe`, `hierarchos-vulkan-transformer-logits.exe`; `licenses\`, `README.md`, `README_ARCHITECTURES.md`, `RELEASE_NOTES.md`, `CLEAN_MACHINE_TEST.md`, `THIRD_PARTY_NOTICES.md`, `BUILD_INFO.txt`, `DEPENDENCIES.txt` |
| `BUILD_MANIFEST.json` | any | Exact source commit, target triple, toolchain, flags, executable hashes and the attention and WMMA module identities |
| `SHA256SUMS` | any | SHA-256 of the archive and the manifest |

The archive is runtime-only: it contains no Rust source tree, Cargo files, build
toolchain, shader sources, PDBs, fixtures, test harnesses or model weights.

Requirements: Windows x64 and a Vulkan 1.x compute-capable GPU driver. The
Vulkan loader (`vulkan-1.dll`) is supplied by the graphics environment and is
**not bundled**. All seven executables are statically linked against the MSVC C
runtime, so **no Visual C++ Redistributable is needed**. Rust, Cargo, Python,
PyTorch and MSVC build tools are not runtime requirements. Model weights are
**not** bundled — point the CLI at a supported local package (`--model-path`) or
let it download a public one (`--hf-model`).

```powershell
# Verify the archive against the sidecar before extracting.
$zip = ".\Hierarchos-Native-Windows-x64-Portable-v0.2.8.zip"
$expected = (Select-String -Path .\SHA256SUMS -Pattern ([regex]::Escape((Split-Path $zip -Leaf))) |
  Select-Object -First 1).Line.Split()[0]
$actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw "SHA-256 mismatch: $actual" }
Expand-Archive $zip -DestinationPath .\hierarchos-v028
cd .\hierarchos-v028

.\bin\hierarchos-native-cli.exe --help
.\bin\hierarchos-native-cli.exe doctor
.\bin\hierarchos-native-cli.exe devices
.\bin\hierarchos-native-cli.exe architectures
.\bin\hierarchos-native-cli.exe precision-info --device-index 0

# Inference against a small public fixture
.\bin\hierarchos-native-cli.exe infer --hf-model hf-internal-testing/tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 8 --no-do-sample

# One-step native Vulkan training, then checkpoint reload
.\bin\hierarchos-native-cli.exe transformer-train `
  --model-path C:\models\tiny-random-gpt2 `
  --train C:\data\one-row.jsonl `
  --out-dir C:\outputs\trained `
  --epochs 1 --batch-size 1 --seq-len 16 --lr 5e-5 --device-index 0

.\bin\hierarchos-native-cli.exe infer `
  --model-path C:\outputs\trained `
  --prompt "Reload checkpoint" --max-new-tokens 8 --no-do-sample
```

`doctor` reports Vulkan visibility and companion resolution. `precision-info`
reports capability/admission state and the measured adapter; a capability report
alone is not proof of a cooperative dispatch. The acceptance report records the
packaged dispatch test separately.

Troubleshooting: a missing Vulkan driver shows up in `doctor` and `devices`;
keep `bin\` intact because the CLI launches its Vulkan companions from there
(`HIERARCHOS_VULKAN_BIN_DIR` overrides discovery). [CLI reference](hierarchos-native-cli/README.md) ·
[WMMA guide](WMMA_GUIDE.md) · [issues](https://github.com/necat101/Hierarchos-Native/issues).

### Portable standalone bundle

Build the runtime-only Windows x64 package with the static MSVC C runtime:

```powershell
.\hierarchos-vulkan\package-standalone.ps1 `
  -OutDir C:\path\to\hierarchos-v025-stage `
  -StaticCrt
```

The packager uses the checked-in standalone lockfile, builds exactly seven release
executables, verifies their PE import closure, and writes `BUILD_INFO.txt` and
`DEPENDENCIES.txt`. Use `-SkipBuild` for source-only staging. Keep the GUI, CLI,
and Vulkan companions together; `HIERARCHOS_VULKAN_BIN_DIR` can override companion
discovery. [Distribution layout](hierarchos-vulkan/standalone/README.md).

### Acceptance status

This is a **same-host extracted-artifact acceptance pass**, not independent
clean-machine certification. The v0.2.6 archive was extracted into a path with
spaces, relocated to a deeper path with spaces, and exercised from the relocated
layout. Results:

- **PE closure PASS.** All seven executables are x64 PE32+, statically linked,
  with no `VCRUNTIME*`, `MSVCP*`, `MSVCR*`, `concrt*` or `ucrtbase` import.
- **Functional smoke PASS.** CLI help, `doctor`, `devices`, `architectures`,
  `precision-info`, local inference, one-step Vulkan training, checkpoint reload,
  relocation and checksum verification passed. Global WMMA stayed off in the
  default run.
- **Packaged C1 dispatch PASS.** A disposable Llama geometry with hidden/K = 64
  and 4096-wide projections gives `64 × 4096 × 64 = 16,777,216` MACs at `rows=64`,
  above the unchanged 16M floor while respecting the Full FP16 reduction cap and
  satisfying the C1 contract. The extracted trainer reported
  `requested=full-fp16 cooperative_dispatches=8` and the profile named
  `wmma_gemm_f16_f16acc_twowave_p3_c1`.
- **Packaged P3 fallback PASS.** The same fixture at `rows=80` (`80 % 32 = 16`)
  stays above the floor at 20,971,520 MACs but fails tile coverage, and the
  extracted trainer fell back to `wmma_gemm_f16_f16acc_twowave_p3` for its 4
  cooperative dispatches. No variant override, floor change or test flag was
  used, and WMMA `off` recorded zero cooperative dispatches on both geometries.
- **Packaged numerical comparison PASS.** The logits executable matched FP32
  within `max_abs 0.00079345` / `rms 0.00012127` on the C1 route (262,144
  elements) and `0.00079345` / `0.00012087` on the P3 route (327,680 elements).
  Those geometries are outside the CLI's frontend-qualified 64-wide training
  profile; the test uses the packaged standalone trainer.
- **BLOCKED.** Independent clean Windows installation, a second physical CPU/GPU,
  physical network isolation, different-drive relocation, GUI interaction, code
  signing and automated GUI “Check backend” activation were not performed. The
  native V4.1 trainer and `hierarchos-infer` were not exercised functionally: both
  need a V4.1 model package that this environment does not have, and neither is on
  the Full FP16 path.
- **Unified-release checker.** The checker reproduces the known local-only
  `wmma_qualify` digest mismatch: three builds differ in 20 bytes of PE
  debug-directory/CodeView metadata, while the 40 accuracy receipts and eight
  dumps are byte-identical. The checker passes in its CI-equivalent condition
  (no local release executable present). This is binary link metadata
  nondeterminism, not a source or functional-output difference.

The full per-executable receipts, commands and limitations are in
[the v0.2.6 acceptance report](hierarchos-vulkan/standalone/CLEAN_MACHINE_TEST.md).
No end-to-end application or training-step speedup is claimed.

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

## Project status

| Area | Current scope |
| --- | --- |
| **Development** | Active source development; build and qualification evidence are revision-specific |
| **DeepSeek V4.1 Flash** | Dedicated native text implementation, scaled inference/PEFT gates, save/resume/export and opt-in parameter/MoE streaming |
| **Latest AMD closure** | The padded GPT-2-wide gap is closed at the unchanged CPU threshold; original frozen references preserved |
| **Phase XVIII replay** | Two complete nine-stage runs, current-source GPT-2 16/16, path-smoke repair; final GitHub workspace check PASS; no RDNA3 self-hosted runner |
| **Intel Gen9** | Completed physical port and optimization receipt; the AMD parity repair preserves existing Gen9 shader bytes and selection |
| **Compatibility boundaries** | Text-backbone support is not universal multimodal/task-head compatibility; formats, training modes and precision policies need their own gates |

### Known limitations

- Physical RDNA3 CI is unavailable until a self-hosted runner is registered; the committed RDNA3 receipts are source-bound local qualification results, not a hardware-CI pass.
- Registry breadth is not qualification breadth: the measured matrix covers 43 scaled profiles / 26 architecture identifiers, not the full 146-entry registry or every mode and precision policy.
- The Gen9 result is a preserved frozen-reference result; this AMD host did not perform a new physical Intel execution.
- DeepSeek V4.1 evidence uses accepted reduced-width/scaled checkpoints; full production weights and all-profile production throughput are not claimed.
- Performance results are matched-profile fixture measurements, not universal speedups or production-size model throughput.

Use the live `architectures` command for package recognition and the linked receipts
for numerical coverage. [Status and limitations](VALIDATION_STATUS.md) ·
[Compatibility](hierarchos-vulkan/COMPATIBILITY.md).

<div align="right"><a href="#table-of-contents">↑ Back to contents</a></div>

---

### Multimodal training development

Native CLIP/LLaVA image processing, projector training and the shared modality
embedding bridge are under development. The same backends are reachable without
the GUI as `vision-train`, `vision-generate`, `vision-info`, `image-generate` and
`image-engine-info`. See the
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
| Nine-stage replay and path repair | [Phase XVIII receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase18-replay-reliability/README.md) · [repair evidence](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase18-path-smoke-repair/README.md) |
| Performance and full-suite history | [Phase XIV receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase14-qualification/README.md) |
| Final architecture and session verification | [Final receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-final-architecture-verification/README.md) |
| Chinese families and sparse attention | [Audit receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-chinese-sparse-attention-audit/README.md) |
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
