# Hierarchos Native — Windows x64 Release Candidate 1

This package is a portable source-and-binary distribution of the Hierarchos
native Rust/Vulkan runtime. The binaries are self-contained apart from the
operating system, the Vulkan loader/compute driver, and the Microsoft Visual
C++ runtime. Python, PyTorch, CUDA, a Transformers checkout, the Vulkan SDK, and
a shader compiler are not runtime dependencies.

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
- **Microsoft Visual C++ 2015–2022 Redistributable (x64).** The executables
  import `VCRUNTIME140.dll` and the Universal C Runtime; these DLLs are not
  bundled. Install the x64 redistributable if the runtime is absent.
- A Vulkan 1.x loader and a Vulkan compute-capable GPU driver.
- Enough RAM and disk for the selected model. No model weights are bundled.
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

## Verification status

The developer-host acceptance run passed CLI startup, Vulkan discovery, the
architecture registry, online and local inference, a one-step Vulkan training
run with checkpoint reload, relocation to a path with spaces, and negative
model/tokenizer/config/download diagnostics. The GUI opened and closed cleanly;
its “Check backend” control was not activated by the automated harness.

The independent clean-machine, second-GPU, and physical-network-isolation checks
are **BLOCKED**; see `CLEAN_MACHINE_TEST.md`. The package was not code-signed.
No numerical algorithms, tolerances, checkpoint formats, or shaders were changed
for this packaging release.

## License and notices

Hierarchos Native is MIT licensed; see `LICENSE`. Third-party components and
reference-derived shader code retain their own terms in
`THIRD_PARTY_NOTICES.md` and `licenses/`. No model weights are bundled.

## Provenance

`BUILD_INFO.txt` records the binary build source, toolchain, profile, and
SHA-256 values for all seven executables. `SHA256SUMS.txt` beside the archive
records the archive checksum and integrity test.
