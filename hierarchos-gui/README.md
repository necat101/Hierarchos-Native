# Hierarchos Native GUI

`hierarchos-native` is the desktop frontend for the same pure-Rust/native-Vulkan
training and inference contracts exposed by `hierarchos-native-cli`. The GUI
intentionally does not reimplement model logic: it validates common inputs,
builds the CLI invocation, streams stdout/stderr, and can stop the child process.

It supports seven workflows from one screen:

- full native Vulkan Transformer training;
- full-parameter Transformer fine-tuning, including Falcon H1/H1R (`falcon_h1`);
- Transformer LoRA fine-tuning (rank 8 by default);
- native Vulkan Transformer inference/generation from a local package or
  Hugging Face model ID;
- fresh/resumed coherent-v9 Hierarchos training; and
- Hierarchos fine-tuning;
- pure-Rust Hierarchos inference/chat from a local native model package.

The form is mode-aware: Transformer datasets are selected as files, fresh
Hierarchos initialization requires a tokenizer when no model package is given,
LoRA rank is shown only for Transformer LoRA fine-tuning, and the
assistant-recovery preset is exposed only for the Hierarchos training mode that
implements it. Inference mode exposes prompt, token limit, temperature, top-k,
top-p, sampling policy, and (for Transformer generation) Vulkan device selection.
The output pane is read-only so model output and logs cannot be edited accidentally.

Transformer generation and LoRA fine-tuning expose **Stream frozen parameters
and MoE experts from disk**. It is unchecked by default and adds
`--stream-parameters` to the displayed CLI command. It selects package-backed
projection/expert/vocabulary execution, including the dedicated DeepSeek V4.1
path. Full-parameter training keeps mutable weights resident. Ordinary Llama,
Mistral, Qwen2 and Mixtral packages also avoid resident body-weight allocation
during loading. Other generic loaders retain their initial-loading peak.

Local model/tokenizer/dataset paths and Hugging Face `OWNER/REPO` model/dataset
IDs are supported. Common training controls (epochs, batch size, maximum sequence
length, learning rate, Vulkan device index and assistant-recovery SFT preset) are
available without requiring users to construct a long shell command.

The **Check backend** button runs `hierarchos-native-cli doctor`, while **List
architectures** displays the live native Transformer registry. The command
preview is shown before launch so advanced users can reproduce a GUI run exactly
from a terminal.

## Build

```powershell
cargo build --release --manifest-path hierarchos-gui/Cargo.toml
```

Keep `hierarchos-native.exe` and `hierarchos-native-cli.exe` in the same
directory for a portable bundle. Alternatively set `HIERARCHOS_NATIVE_CLI` to
the CLI executable. Hierarchos `train`/`finetune` additionally needs the
`hierarchos-vulkan-train` companion binary, normally colocated with the CLI or
provided through `HIERARCHOS_VULKAN_BIN_DIR`.

The GUI does not launch Python or PyTorch. Hugging Face model/dataset downloads,
tokenization, native Transformer training/generation and checkpoint export remain
handled by the Rust CLI/backend; Hierarchos inference remains handled by the
pure-Rust inference engine linked by that CLI.

For a distributable source + binary bundle that does not rely on the parent
Transformers checkout, run `hierarchos-vulkan/package-standalone.ps1` from the
repository root.

## Workflows

The desktop app exposes ten workflows: Transformer training, full-parameter
fine-tuning, LoRA, Hierarchos training, Hierarchos fine-tuning, Transformer
generation, native vision training, image-to-text, text-to-image, and Hierarchos
inference. Each form builds the exact CLI arguments shown above the launch button;
text-cache and continuous-batching controls are not forwarded to image workers.

- **Vision training** takes a native image package and an image/text JSONL, and
  writes a new or empty output directory.
- **Image-to-text** takes one image plus a prompt about it, with greedy or sampled
  decoding.
- **Text-to-image** takes a Stable Diffusion single-file checkpoint, an optional
  VAE and engine path, prompt/negative prompt, dimensions, steps, CFG and seed,
  and writes a new PNG with a preview. Install the engine with
  `hierarchos-native-cli/install-image-engine.ps1`; without it the workflow reports
  the missing engine instead of generating.

Native V4.1 keeps its base model and trains image parameters plus a language LoRA,
so its form carries the LoRA rank, streaming and resume/export fields. The other
families load a complete exported image package.

On Windows each run is placed in a job object with `KILL_ON_JOB_CLOSE`, so Stop —
or closing the window — terminates the CLI and any companion image engine it
spawned. On Unix the CLI is started in its own process group and the group is
signalled. A cancelled worker can still report exit code 0 on Windows; the test
therefore checks that the process is gone, not that it failed.

When the GUI and CLI are built separately, `discover_cli` falls back to the
repository's crate build directories before giving up, and the CLI path field
accepts an explicit binary.
