# Hierarchos Native — Windows x64 Release Candidate 1

This package is a portable source-and-binary distribution of the Hierarchos
native Rust/Vulkan runtime. The binaries are self-contained apart from the
operating system and the installed Vulkan loader/compute driver. Python,
PyTorch, CUDA, a Transformers checkout, the Vulkan SDK, and a shader compiler
are not runtime dependencies.

## Included executables

- `bin/hierarchos-native-cli.exe`
- `bin/hierarchos-native.exe`
- `bin/hierarchos-infer.exe`
- `bin/hierarchos-vulkan-train.exe`
- `bin/hierarchos-vulkan-devices.exe`
- `bin/hierarchos-vulkan-transformer-train.exe`
- `bin/hierarchos-vulkan-transformer-logits.exe`

The GUI and CLI discover one another beside `bin/`. `hierarchos-native-cli`
also discovers the Vulkan companion binaries in the same directory.

## Requirements

- Windows 11 x64 (Windows 10 x64 should work but is not the acceptance target).
- A Vulkan 1.x loader and a Vulkan compute-capable GPU driver.
- Enough RAM and disk for the selected model. The bundled `hf-internal-testing`
  smoke fixture is tiny; production models are much larger.
- No administrator rights are required for normal operation. Downloaded model
  data is cached under `%LOCALAPPDATA%\Hierarchos\hf-cache` by default.

## First run

```powershell
.\bin\hierarchos-native-cli.exe doctor
.\bin\hierarchos-native-cli.exe devices
.\bin\hierarchos-native-cli.exe architectures
```

Online smoke inference:

```powershell
.\bin\hierarchos-native-cli.exe infer `
  --hf-model hf-internal-testing/tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 32 --no-do-sample
```

Offline/local inference uses the same executable with `--model-path` pointing at
a supported local package. Set `HIERARCHOS_HF_CACHE` or pass `--hf-cache-dir` to
relocate the cache.

Launch the desktop frontend:

```powershell
.\bin\hierarchos-native.exe
```

## License and notices

Hierarchos Native is MIT licensed; see `LICENSE`. Third-party components and
reference-derived shader code retain their own terms in
`THIRD_PARTY_NOTICES.md` and `licenses/`. No model weights are bundled.

## Verification status

`BUILD_INFO.txt` records the exact source commit, build profile, Rust/Cargo
versions, and binary SHA-256 values for this package. See
`CLEAN_MACHINE_TEST.md` for the release acceptance results and limitations.
