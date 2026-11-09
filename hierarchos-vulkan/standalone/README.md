# Hierarchos Native Backend

This is the standalone source/binary layout for Hierarchos' native Transformer
backend. It is deliberately independent of the Python `transformers` source
tree at build time and runtime: model/tokenizer/dataset I/O is Rust, Transformer
and Hierarchos training is Vulkan compute, and Hierarchos inference is pure Rust.

The Python Transformers checkout remains a development reference for registry
audits and numerical parity tests. It is not a runtime dependency of this
package.

## Package layout

```text
Cargo.toml
Cargo.lock                 # checked-in standalone release lockfile
LICENSE
THIRD_PARTY_NOTICES.md
licenses/                  # bundled third-party license texts
RELEASE_NOTES.md
CLEAN_MACHINE_TEST.md      # measured acceptance results and BLOCKED checks
BUILD_INFO.txt             # source commit, toolchain and binary hashes
README.md
README_ARCHITECTURES.md    # generated native Transformer support inventory
bin/                       # exactly seven release executables
crates/
  hierarchos-inference/
  hierarchos-vulkan/
  hierarchos-native-cli/
  hierarchos-gui/
```

`crates/hierarchos-vulkan/README_ARCHITECTURES.md` is generated from the Rust
architecture registry and is the authoritative list of end-to-end native text
graph contracts and package aliases. `hierarchos-native-cli architectures`
reports the same live registry from the built backend.

## Requirements

- Windows 11 x64 for the documented binary package.
- **Microsoft Visual C++ 2015–2022 Redistributable (x64).** The executables
  import `VCRUNTIME140.dll`; the runtime is not bundled.
- A Vulkan 1.x loader and a compute-capable Vulkan device for GPU training.
- Rust/Cargo 1.93.0-compatible toolchain only when rebuilding from source.
- No Python, PyTorch, CUDA toolkit, or local Transformers checkout is required
  to build or run the native backend. No administrator rights are required for
  normal operation.

## First-run checks

```powershell
.\bin\hierarchos-native-cli.exe doctor
.\bin\hierarchos-native-cli.exe devices
.\bin\hierarchos-native-cli.exe architectures
.\bin\hierarchos-native-cli.exe infer --model-path .\model --prompt "Hello" --max-new-tokens 64
```

`doctor` checks Vulkan device visibility and the companion Hierarchos training
executable. It reports a missing companion as `not found`; the command currently
exits 0, so read the diagnostic rather than relying only on the exit status.
The Transformer training and generation paths are linked into
`hierarchos-native-cli` itself. `hierarchos-native.exe` exposes the same common
training and inference workflows through the desktop GUI.

The packager copies release executables into `bin/` and removes its Cargo
`target/` tree by default so the distributable does not retain compiler
intermediates. The archive includes `LICENSE`, `THIRD_PARTY_NOTICES.md`, and
`licenses/`; no model weights are bundled. Downloaded models are cached in the
per-user platform cache at runtime, not in the installation directory. The
default is `%LOCALAPPDATA%\Hierarchos\hf-cache`; override it with
`HIERARCHOS_HF_CACHE` or `--hf-cache-dir`.

For the measured acceptance results, supported and blocked checks, and
limitations, read `CLEAN_MACHINE_TEST.md`. The packaged binaries were built
from source `e8fc472b5a64cce89a028a5e17652d148be71f1b` with the checked-in
standalone lockfile and `--locked`.

## Build from source

```powershell
cargo build --release --locked -p hierarchos-native-cli --bin hierarchos-native-cli
cargo build --release --locked -p hierarchos-vulkan --bin hierarchos-vulkan-train --bin hierarchos-vulkan-devices --bin hierarchos-vulkan-transformer-train --bin hierarchos-vulkan-transformer-logits
cargo build --release --locked -p hierarchos-inference --bin hierarchos-infer
cargo build --release --locked -p hierarchos-gui --bin hierarchos-native
```

The Vulkan crate embeds its checked-in SPIR-V shader binaries at compile time;
shader compilers are not required for an ordinary release build.

For training examples and Hugging Face package/dataset support, see
`crates/hierarchos-native-cli/README.md`. Unsupported model types, packages, and
non-text towers fail closed instead of being silently approximated.

## Development reference tests

The validation scripts under the original repository's
`hierarchos-vulkan/validation/` compare this backend with the adjacent Python
Transformers checkout. They are development-only reference tooling and are not
included in the standalone runtime package.
