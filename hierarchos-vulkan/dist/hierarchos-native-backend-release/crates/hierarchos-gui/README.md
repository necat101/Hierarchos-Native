# Hierarchos Native GUI

`hierarchos-native` is the desktop frontend for the same pure-Rust/native-Vulkan
training and inference contracts exposed by `hierarchos-native-cli`. The GUI
intentionally does not reimplement model logic: it validates common inputs,
builds the CLI invocation, streams stdout/stderr, and can stop the child process.

It supports six workflows from one screen:

- full native Vulkan Transformer training;
- Transformer LoRA fine-tuning (rank 8 by default);
- native Vulkan Transformer inference/generation from a local package or
  Hugging Face model ID;
- fresh/resumed coherent-v9 Hierarchos training; and
- Hierarchos fine-tuning;
- pure-Rust Hierarchos inference/chat from a local native model package.

The form is mode-aware: Transformer datasets are selected as files, fresh
Hierarchos initialization requires a tokenizer when no model package is given,
LoRA rank is shown only for Transformer fine-tuning, and the
assistant-recovery preset is exposed only for the Hierarchos training mode that
implements it. Inference mode exposes prompt, token limit, temperature, top-k,
top-p, sampling policy, and (for Transformer generation) Vulkan device selection.
The output pane is read-only so model output and logs cannot be edited accidentally.

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
