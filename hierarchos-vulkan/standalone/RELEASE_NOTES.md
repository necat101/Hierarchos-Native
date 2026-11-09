# Hierarchos Native — Windows x64 Portable Release Candidate 2

RC2 is a runtime-only, extract-and-run Windows x64 package. It removes the RC1
dependency on a separately installed Microsoft Visual C++ 2015–2022
Redistributable by linking the seven release executables with the static MSVC C
runtime. It also removes source, Cargo manifests, build trees, shader sources,
fixtures, caches, and development logs from the user-facing archive.

## Included executables

- `bin/hierarchos-native.exe`
- `bin/hierarchos-native-cli.exe`
- `bin/hierarchos-infer.exe`
- `bin/hierarchos-vulkan-train.exe`
- `bin/hierarchos-vulkan-devices.exe`
- `bin/hierarchos-vulkan-transformer-train.exe`
- `bin/hierarchos-vulkan-transformer-logits.exe`

The GUI and CLI discover one another beside `bin/`. `hierarchos-native-cli` also
discovers the Vulkan companion binaries in the same directory.

## Requirements

- Windows 11 x64 (Windows 10 x64 should work but is not the acceptance target).
- **No Microsoft Visual C++ Redistributable installation is required.** The
  release executables are statically linked against the MSVC C runtime. Their
  PE import tables are checked during packaging; `VCRUNTIME140.dll`,
  `VCRUNTIME140_1.dll`, `MSVCP140.dll`, and other Visual C++ runtime DLLs must
  not appear.
- A Vulkan 1.x loader and a Vulkan compute-capable GPU driver for native
  Transformer generation and training. The loader is supplied by the graphics
  software environment; it is not bundled.
- Enough RAM and disk for the selected model. No model weights are bundled.
- No administrator rights are required for normal operation. Downloaded model
  data is cached under `%LOCALAPPDATA%\Hierarchos\hf-cache` by default; override
  it with `HIERARCHOS_HF_CACHE` or `--hf-cache-dir`.

Windows system DLLs, the GPU driver, and its Vulkan loader are the intended
external runtime dependencies. RC2 does not include a Rust/Cargo/Python/PyTorch
toolchain, Visual Studio, an installer, a registry step, or a shader compiler.

## What changed from RC1

- Static MSVC CRT linkage for all seven release executables.
- New PE import-closure verifier that fails packaging on a Visual C++ runtime
  DLL or an unexpected non-system DLL.
- Runtime-only archive layout; source remains available in the Git repository.
- Source and Cargo.lock hashes, effective static-CRT flags, and per-executable
  hashes recorded in `BUILD_INFO.txt`.
- RC2 documentation distinguishes portable design, local acceptance, and
  unverified clean-machine/other-hardware scope.
- RC1 is preserved unchanged; this package does not replace its archive.

## Verification status

The RC2 archive was extracted on the available Windows 11 / AMD RDNA3 host and
tested from the extracted layout. The local acceptance run covered:

- All seven executables present, x64, with a clean PE import closure.
- CLI help, `doctor`, Vulkan device inventory, and the architecture registry.
- Local tiny-model inference, fresh-cache Hub inference, and checkpoint reload.
- One-step native Vulkan training with a measured loss.
- Relocation to paths containing spaces and deep nesting.
- Missing-model, unsupported-architecture, and missing-companion diagnostics.
- A write-denied installation copy.
- GUI launch and clean close.

The GUI “Check backend” control was not activated by the available mouse or UI
Automation harness. That interaction is **BLOCKED**, not a claimed pass; the same
backend check passes through the CLI.

The independent clean-machine, second physical CPU/GPU, physical
network-isolation, different-drive, and code-signing checks are also **BLOCKED**
or not applicable. This is a local acceptance pass, not clean-machine
certification. CPU compatibility is analytically checked through the default
x86_64 target and retained runtime feature dispatch.

## Measured host

- Windows 11 Enterprise x64, version 10.0.26300
- AMD Ryzen Z1 Extreme, 16 logical CPUs
- AMD Radeon Graphics, driver 32.0.31007.6002, Vulkan 1.4.341
- 11.5 GB visible RAM

No model weights, training fixtures, caches, or build intermediates are in the
archive.

## Provenance

`BUILD_INFO.txt` records the build source, `Cargo.lock` SHA-256, target, release
profile, static-CRT flags, toolchain, and SHA-256 values for all seven
executables. `DEPENDENCIES.txt` records the analyzed PE imports. The archive
SHA-256 is in the sidecar checksum file beside the archive, not inside the
archive.

## License and notices

Hierarchos Native is MIT licensed; see `LICENSE`. Third-party components and
reference-derived shader code retain their own terms in
`THIRD_PARTY_NOTICES.md` and `licenses/`. No model weights are bundled.

## Package limitations

- No installer, code signing, auto-updater, or uninstaller.
- No Vulkan acceleration is promised on a machine without a working Vulkan
  driver.
- Model recognition is not universal model certification. Unsupported packages
  fail closed.
- The numerical baseline remains source-bound; this packaging release did not
  change numerical algorithms, tolerances, checkpoint formats, or shaders.
