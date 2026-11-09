# Hierarchos Native — Windows x64 Portable RC2 Acceptance Report

Date: 2026-11-09

Status: **PORTABLE RC — LOCAL ACCEPTANCE PASS**

This is a same-host extracted-artifact acceptance report, not a clean-machine
certification. An independent Windows installation, a second physical CPU/GPU,
and physical network isolation were not available. The result below is therefore
limited to the measured Windows 11 / AMD RDNA3 host.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-Portable-RC2.zip`
- Binary build source: `164711020d9fe9f229756f0ff58d8160ba9c580a`
- Target: `x86_64-pc-windows-msvc`
- Profile: release (`opt-level=3`, thin LTO, `codegen-units=1`)
- CRT linkage: static MSVC C runtime (`-C target-feature=+crt-static`)
- Standalone `Cargo.lock` SHA-256:
  `f4360adfc4a2127c0be817c71c08659c181f7ff04813b5f8bc882a23ca7132d7`
- Archive SHA-256: recorded in the checksum sidecar beside the archive. It is
  not embedded in this report because the report is inside the archive.

`BUILD_INFO.txt` records the toolchain and per-executable SHA-256 values.
`DEPENDENCIES.txt` records the analyzed PE import tables. The archive contains
seven runtime executables and supporting documentation/licenses only; it does
not contain Cargo manifests, source, shader sources, Cargo `target/`, PDBs,
fixtures, models, caches, or developer logs.

## Host used

- Windows 11 Enterprise x64, version 10.0.26300
- AMD Ryzen Z1 Extreme, 16 logical CPUs
- AMD Radeon Graphics, driver 32.0.31007.6002, Vulkan 1.4.341
- 11.5 GB visible RAM
- Rust/Cargo 1.93.0 used only for the build, not at runtime

## Results

| Check | Result | Evidence / scope |
| --- | --- | --- |
| Seven packaged executables | **PASS** | Exactly seven `.exe` files in `bin/`; no target, PDB, source, cache, or test fixture |
| Executable architecture | **PASS** | All seven are `IMAGE_FILE_MACHINE_AMD64` |
| PE import closure | **PASS** | All seven contain only Windows system DLL/API-set imports; no `VCRUNTIME*.dll`, `MSVCP*.dll`, or other Visual C++ runtime DLL |
| Developer-only DLLs | **PASS** | No unexpected third-party DLL imports in the seven PE import tables |
| Absolute developer paths | **PASS** | Zero occurrences of the build/user paths in each executable; dependency report uses `Directory: bin` |
| CLI startup/help | **PASS** | `hierarchos-native-cli.exe --help` exited 0 |
| Backend `doctor` | **PASS** | Exited 0; one AMD Radeon Graphics device; companion paths resolved from the extracted `bin/` |
| Vulkan device inventory | **PASS** | `devices` returned the AMD Radeon Graphics device, queue family 0 |
| Architecture registry | **PASS** | 146 canonical architectures + 83 package aliases |
| Local tiny-model inference | **PASS** | Local package generated tokens; exit 0 |
| Fresh-cache Hub inference | **PASS** | `hf-internal-testing/tiny-random-gpt2` generated tokens; exit 0; cache populated in a new directory |
| Offline local inference | **PASS (dead-proxy test)** | All proxy variables pointed at `127.0.0.1:9`; local inference exited 0. Physical network isolation was not performed. |
| One-step Vulkan training | **PASS** | One epoch / one row / batch 1 / sequence 16: `steps=1`, `mean_loss=6.905534` |
| Trained checkpoint reload | **PASS** | Exported model package loaded and generated tokens; exit 0 |
| GUI launch | **PASS (window)** | `hierarchos-native.exe` created a live `Hierarchos Native` window and was responsive; closed cleanly |
| GUI “Check backend” interaction | **BLOCKED** | Real mouse input and Windows UI Automation did not activate the control; UI Automation exposed zero descendants. The same backend check passes through the CLI. |
| Missing model directory | **PASS** | Exited 1 with a clear missing `config.json` path diagnostic |
| Unsupported architecture | **PASS** | Exited 1 with `unsupported Hugging Face model_type "definitely_unsupported"` and the supported list |
| Missing companion binary | **PASS (diagnostic)** | `doctor` printed `hierarchos-vulkan-train: not found (needed for Hierarchos train/finetune)` and still exited 0; this is diagnostic behavior, not fail-closed |
| Relocation to spaces | **PASS** | Copied package to `C:\Users\Public\Hierarchos Test\RC2 Relocated\portable copy`; `doctor` and local inference passed |
| Deeply nested path | **PASS** | Copied to a nested path with spaces under that tree; `doctor` and `architectures` passed |
| Read-only installation copy | **PASS** | ACL removed write access; a direct write attempt was denied; `doctor` and local inference still exited 0 |
| Installation-directory writes | **PASS** | Ordinary inference used an external model and a per-user cache; no package write was required |
| Different drive letter | **NOT TESTED** | Only the `C:` volume was available |
| Independent clean machine | **BLOCKED** | No second Windows computer was available |
| Second physical CPU/GPU | **BLOCKED** | No second qualifying host was available |
| Physical network isolation | **BLOCKED** | Not performed; dead-proxy local inference is reported instead |
| Code signing | **NOT PRESENT** | The archive and executables are unsigned |

## Visual C++ runtime conclusion

RC1 imported `VCRUNTIME140.dll` in every executable. RC2 does not. All seven
released executables were rebuilt with `-C target-feature=+crt-static`, and the
automated PE verifier reports **PASS** for a seven-executable closure with only
Windows system imports. **A Microsoft Visual C++ 2015–2022 Redistributable
installation is not required for RC2.**

The Vulkan loader is supplied by the installed graphics software and is loaded
at runtime; it is not a Microsoft C++ runtime component and is not bundled in the
ZIP. Windows system components and the GPU driver remain external dependencies.
No Vulkan acceleration is promised on a machine without a working Vulkan driver.

## CPU compatibility

The release uses the default `x86_64-pc-windows-msvc` target and does not use
`target-cpu=native` or a release-wide AVX-512 requirement. Existing runtime
feature dispatch and fallback code were not changed. CPU compatibility is
**analytically checked**, not physically certified on a second processor.

## Numerical and functional scope

The packaged executable completed a real one-step Vulkan training update with
`mean_loss=6.905534`, exported a model, reloaded it, and generated tokens. Local
and fresh-cache Hub inference also completed. This is a focused rebuilt-binary
smoke test, not a new full numerical qualification campaign.

The prior source-bound baseline remains the reference for the full replay and
parity suites: nine-stage replay, 1,139 Rust library tests, 218 active DeepSeek
tests, 30 native V4.1 tests, and strict `< 2e-7` logit parity. No numerical
algorithm, tolerance, checkpoint format, shader, or performance contract was
changed in this packaging release; prior executable-specific results are not
claimed automatically for the new static binaries.

## Verdict

**PORTABLE RC — LOCAL ACCEPTANCE PASS.**

RC2 is extract-and-run on the measured Windows x64 host, relocatable, free of a
Visual C++ Redistributable prerequisite, and has a verified seven-executable PE
import closure. It is not clean-machine-certified: independent Windows,
second-hardware, different-drive, physical network-isolation, code-signing, and
GUI-button interaction checks remain unverified or blocked.
