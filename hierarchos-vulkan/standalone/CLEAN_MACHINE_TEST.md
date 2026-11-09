# Release Candidate Acceptance Report

Date: 2026-11-09

Status: **developer-host acceptance PASS; independent clean-machine execution BLOCKED.**

This is not a clean-machine test. The package was built, exercised, relocated,
and re-exercised on the same Windows host. No second Windows machine or
independent GPU host was available. The results below must not be read as a
clean-machine or universal GPU certification.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-RC1.zip`
- Binary build source: `e8fc472b5a64cce89a028a5e17652d148be71f1b`
- Packaging revision: `fb19ebcdafa8c164a98080a8af637d17e09a5725`
- Target: `x86_64-pc-windows-msvc`
- Toolchain: Rust/Cargo 1.93.0
- Archive SHA-256 and integrity are recorded in `SHA256SUMS.txt` beside the
  archive. `BUILD_INFO.txt` records the per-binary SHA-256 values.

## Host used

- Windows 11 Enterprise x64, version 10.0.26300, build 26300
- AMD Ryzen Z1 Extreme, 16 logical CPUs
- AMD Radeon Graphics, driver 32.0.31007.6002 (Vulkan 1.4.341)
- 11.5 GB visible RAM
- Visual C++ runtime present: `VCRUNTIME140.dll` 14.44.35211.0

## Results

| Check | Result | Evidence / scope |
| --- | --- | --- |
| Seven packaged executables | **PASS** | Exactly seven `.exe` files in `bin/`; no `target/`, `.pdb`, or build logs |
| CLI starts and reports help | **PASS** | `hierarchos-native-cli.exe --help` exited 0 |
| Backend `doctor` | **PASS** | Exited 0; one AMD Radeon Graphics device; both companion paths resolved from `bin/` |
| Vulkan device inventory | **PASS** | `devices` returned the AMD integrated GPU with queue family 0 |
| Architecture registry | **PASS** | 146 canonical + 83 aliases; 229 advertised spellings |
| Online tiny-model inference | **PASS** | `hf-internal-testing/tiny-random-gpt2` generated tokens |
| Local tiny-model inference | **PASS** | Local package generated tokens |
| Offline local inference | **PASS (dead-proxy test)** | All proxy variables pointed at `127.0.0.1:9`; local inference still exited 0. Physical network isolation was not performed. |
| One-step Vulkan training | **PASS** | `transformer-train`, one epoch / one row / batch 1 / sequence 16: `steps=1`, `mean_loss=6.905534` |
| Trained checkpoint reload | **PASS** | Exported model package loaded and generated with `infer` |
| GUI launches | **PASS** | Window titled `Hierarchos Native`; clean close |
| GUI “Check backend” interaction | **BLOCKED** | Windows pointer/window-message injection produced hover but did not activate the control. The same `doctor` command passed from the CLI; no GUI button execution is claimed. |
| Relocation to a path with spaces | **PASS** | Copied package to `C:\Users\Public\Hierarchos Test\relocated RC1`; `doctor`, `architectures`, local inference, and GUI startup all passed from that path |
| Missing model directory | **PASS** | Exited 1 with a clear missing `config.json` diagnostic |
| Missing tokenizer assets | **PASS** | Exited 1 with the missing `tokenizer.json` path |
| Unsupported model type | **PASS** | Exited 1 with `unsupported Hugging Face model_type "definitely_unsupported"` |
| Missing companion binary | **PASS (diagnostic)** | `doctor` printed `hierarchos-vulkan-train: not found` and the optional devices probe status. The command exits 0; this is diagnostic behavior, not a fail-closed status. |
| Model-download failure | **PASS** | Nonexistent Hub repository exited 1 with `Hugging Face returned HTTP 401` |
| Read-only package copy | **NOT TESTED** | An ACL-based write-denied test was attempted but the test harness failed before executing the binaries. Normal local commands do not write beside the installation; downloaded models use `%LOCALAPPDATA%\Hierarchos\hf-cache`. |
| Independent clean-machine run | **BLOCKED** | No separate Windows machine was available |
| Physical GPU test outside this host | **BLOCKED** | No second Vulkan host was available |
| Physical network isolation | **BLOCKED** | Not available without host-level network changes; a dead-proxy test is reported above instead |

## Runtime and packaging limitations

- The executables import `VCRUNTIME140.dll` and the Universal C Runtime. The
  Microsoft Visual C++ 2015–2022 Redistributable (x64) is a prerequisite. It is
  **not bundled**. A machine without that runtime may fail to start.
- No model weights are bundled.
- The archive has no installer, code signing, or uninstaller.
- The GUI was launched and closed, but its “Check backend” button was not
  activated by the automated test harness. This is an unverified interaction,
  not a claimed failure of the GUI command path.
- No numerical qualification was rerun for this packaging change. The prior
  9/9 replay, 218/218 DeepSeek, 30/30 V4.1, and 1,139-pass Rust baseline remain
  the relevant source-bound baseline; packaging did not alter numerical code.
- Model recognition is not universal model certification. Unsupported packages
  fail closed.

## Verdict

**Developer-host release candidate PASS with documented BLOCKED checks.** The
archive is suitable for a user-observed Windows RC test, but an independent
clean-machine and second-GPU run remain required before a general-production
claim.
