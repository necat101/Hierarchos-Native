# Hierarchos Native — Windows x64 Portable RC2

This is the runtime package for the Hierarchos native Transformer backend. It is
extract-and-run on a supported Windows x64 computer: no installer, no Rust or
Cargo, no Python or PyTorch, no Visual Studio, no administrator setup, and no
separate Microsoft Visual C++ Redistributable installation.

The release binaries are linked with the static MSVC C runtime
(`-C target-feature=+crt-static`). Their PE import tables are checked during
packaging; the package must not contain `VCRUNTIME140.dll`, `MSVCP140.dll`, or
another Visual C++ runtime DLL.

The archive is intentionally runtime-only. Source, Cargo manifests, shader
sources, build intermediates, fixtures, model weights, caches, and local test
logs are not included. Source remains available in the
[Hierarchos-Native repository](https://github.com/necat101/Hierarchos-Native).

## Package layout

```text
bin/
  hierarchos-native.exe                 # desktop GUI
  hierarchos-native-cli.exe             # CLI and linked Transformer training
  hierarchos-infer.exe                  # pure-Rust coherent-v9 inference
  hierarchos-vulkan-train.exe           # Hierarchos native training
  hierarchos-vulkan-devices.exe          # Vulkan device inventory
  hierarchos-vulkan-transformer-train.exe
  hierarchos-vulkan-transformer-logits.exe
LICENSE
THIRD_PARTY_NOTICES.md
licenses/
README.md
README_ARCHITECTURES.md                 # native architecture inventory
RELEASE_NOTES.md
CLEAN_MACHINE_TEST.md                   # measured local acceptance results
BUILD_INFO.txt                          # source, lockfile, flags, binary hashes
DEPENDENCIES.txt                        # PE import closure for all seven EXEs
```

Keep all seven executables together in `bin/`. The CLI and GUI resolve their
companions relative to the executable directory. Do not move individual
executables into a different directory.

## Requirements

- Windows 11 x64. Windows 10 x64 is expected to work but is not the measured
  acceptance target.
- A Vulkan 1.x loader and a compute-capable Vulkan GPU driver for native
  Transformer training and generation. The Vulkan loader (`vulkan-1.dll`) is
  supplied by the graphics software environment; it is not bundled, and it is
  not a substitute for the GPU driver.
- Enough RAM and disk for the selected model and output. No model weights are
  bundled.
- Models and outputs may be stored anywhere the user can read/write; they are
  not tied to the installation directory.

No Microsoft Visual C++ Redistributable is required by the RC2 executables. No
administrator rights, registry installation, machine-wide environment variable,
or installer is required for ordinary execution. Python, PyTorch, CUDA, a
Transformers checkout, the Vulkan SDK, and a shader compiler are not runtime
dependencies.

## First-run checks

From the extracted directory:

```powershell
.\bin\hierarchos-native-cli.exe --help
.\bin\hierarchos-native-cli.exe doctor
.\bin\hierarchos-native-cli.exe devices
.\bin\hierarchos-native-cli.exe architectures
```

`doctor` reports Vulkan visibility and whether the expected companion
executables resolve from `bin/`. A missing companion is printed as `not found`;
`doctor` currently exits 0 for that diagnostic, so read its output rather than
relying only on the exit code.

Local inference:

```powershell
.\bin\hierarchos-native-cli.exe infer `
  --model-path C:\models\tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 16 --no-do-sample
```

Hub inference and the per-user cache:

```powershell
.\bin\hierarchos-native-cli.exe infer `
  --hf-model hf-internal-testing/tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 8 --no-do-sample
```

The default cache is `%LOCALAPPDATA%\Hierarchos\hf-cache`. Override it with
`HIERARCHOS_HF_CACHE` or `--hf-cache-dir`. The application does not need write
access to its installation directory.

One-step native Vulkan training and checkpoint reload:

```powershell
.\bin\hierarchos-native-cli.exe transformer-train `
  --model-path C:\models\tiny-random-gpt2 `
  --train C:\data\one-row.jsonl `
  --out-dir C:\outputs\trained `
  --epochs 1 --batch-size 1 --seq-len 16 --lr 5e-5 --device-index 0

.\bin\hierarchos-native-cli.exe infer `
  --model-path C:\outputs\trained `
  --prompt "Reload checkpoint" --max-new-tokens 8 --no-do-sample
```

Launch the desktop frontend:

```powershell
.\bin\hierarchos-native.exe
```

## Portability and writable state

The extracted directory can be moved or renamed after extraction, including to
a path containing spaces or to a deeply nested directory. Keep the relative
layout intact.

Downloaded model data, Vulkan execution-cache metadata, temporary files, logs,
and training checkpoints use user-selected or per-user writable locations. The
application does not require installation-directory writes. If an explicit
output directory is supplied, that directory is used as requested.

The only external runtime DLL in the measured import closure besides Windows
system components is `vulkan-1.dll`. The GPU software stack supplies it.

## Verification status

The RC2 archive was extracted and exercised on the Windows 11 / AMD RDNA3 host
used for the prior RC1 acceptance. The local checks cover all seven executables,
the PE import closure, CLI startup, `doctor`, device enumeration, the
architecture registry, local and fresh-cache Hub inference, one-step Vulkan
training, checkpoint reload, relocation, missing-resource diagnostics, and a
write-denied installation copy.

The GUI window launches and closes. The “Check backend” control could not be
activated by the available mouse/UI Automation harness; that interaction is
reported as **BLOCKED**, not as a pass. The same `doctor` command passes from the
CLI.

No independent clean Windows installation, second physical CPU/GPU, physical
network-isolation test, or different-drive relocation was available. CPU
compatibility is analytically checked: the build uses the default
`x86_64-pc-windows-msvc` target, not `target-cpu=native`, and retains the
existing runtime feature dispatch. See `CLEAN_MACHINE_TEST.md` for the complete
result table and scope.

## Build from source

The release candidate was built from source commit
`164711020d9fe9f229756f0ff58d8160ba9c580a` with the checked-in standalone
`Cargo.lock` and `--locked`. The static portable build is produced by:

```powershell
.\hierarchos-vulkan\package-standalone.ps1 `
  -OutDir C:\path\to\hierarchos-rc2-stage `
  -StaticCrt
```

The packager sets `-C target-feature=+crt-static`, uses the default Windows x64
target, copies exactly seven release executables, runs the PE import verifier,
and writes `BUILD_INFO.txt` and `DEPENDENCIES.txt`. Normal development builds
remain dynamically linked; static CRT is a release-packaging option.

For a source checkout, the component build commands are documented in the root
[README](https://github.com/necat101/Hierarchos-Native#quick-start). The Vulkan
crate embeds its checked-in SPIR-V shader binaries at compile time; shader
compilers are not required for an ordinary release build.

## License and notices

Hierarchos Native is MIT licensed; see `LICENSE`. Third-party components and
reference-derived shader code retain their own terms in
`THIRD_PARTY_NOTICES.md` and `licenses/`. No model weights are bundled.
