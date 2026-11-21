# Hierarchos Native — Windows x64 Portable v0.2.5 Acceptance Report

Date: 2026-11-21

Status: **LOCAL ACCEPTANCE PASS, with explicitly blocked items.**

This is a same-host extracted-artifact acceptance report for the v0.2.5 binaries.
An independent Windows installation, a second physical CPU/GPU and physical
network isolation were not available. The result below is limited to the measured
Windows 11 / AMD RDNA3 host.

This report describes the **newly built v0.2.5 binaries**. It does not re-use, and
must not be confused with, the older RC2 acceptance report, which described a
different source commit and did not ship the production P3 epilogue.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-Portable-v0.2.5.zip`
- Packaging source commit: `f47989bd91328d8f94c61e9af2e868199a847401` (clean tree at packaging)
- Qualified kernel source commit: `b165c2994e9ef5929c9e36fca76192db8dd3df35`
- Relation between the two: the packaging commit adds only documentation, the
  evidence bindings and `phase_lxxxiv_evidence/` on top of the qualified commit.
  No file under `hierarchos-vulkan/src`, `hierarchos-vulkan/shaders`,
  `hierarchos-inference`, `hierarchos-native-cli` or `hierarchos-gui` differs.
- Target: `x86_64-pc-windows-msvc` (default target; **not** `target-cpu=native`)
- Profile: release (`opt-level=3`, thin LTO, `codegen-units=1`)
- CRT linkage: static MSVC C runtime (`-C target-feature=+crt-static`)
- Toolchain: `rustc 1.93.0 (254b59607 2026-01-19)` / LLVM 21.1.8,
  `cargo 1.93.0 (083ac5135 2025-12-15)`
- Standalone `Cargo.lock` SHA-256:
  `f4360adfc4a2127c0be817c71c08659c181f7ff04813b5f8bc882a23ca7132d7`
- Packager: `hierarchos-vulkan/package-standalone.ps1 -StaticCrt` with the
  dependency scan **enabled** (no `-SkipDependencyScan`)
- The archive's own SHA-256 is recorded in `BUILD_MANIFEST.json` and
  `SHA256SUMS` beside the ZIP, not inside it.

## Package inventory

Extracted into a directory whose path contains spaces
(`…\_v025_extract\portable test v0.2.5\`), then relocated to a deeper path that
also contains spaces.

Present — exactly the allowlist:

```text
bin/hierarchos-infer.exe
bin/hierarchos-native-cli.exe
bin/hierarchos-native.exe
bin/hierarchos-vulkan-devices.exe
bin/hierarchos-vulkan-train.exe
bin/hierarchos-vulkan-transformer-logits.exe
bin/hierarchos-vulkan-transformer-train.exe
BUILD_INFO.txt
CLEAN_MACHINE_TEST.md
DEPENDENCIES.txt
LICENSE
licenses/Microsoft-STL.txt
README.md
README_ARCHITECTURES.md
RELEASE_NOTES.md
THIRD_PARTY_NOTICES.md
```

Absent, as required: `crates/`, `Cargo.toml`, `Cargo.lock`, `target/`, any `.rs`,
`.comp`, `.glsl`, `.pdb`, `.lib`, `.rlib`, `.d`, fixtures, model weights,
caches and developer logs. The assembly step enforces this: it raises rather than
producing an archive that contains any path outside the runtime allowlist.

## PE closure

All seven executables:

| Check | Result |
| --- | --- |
| x64 PE architecture | **7/7** (`machine = 0x8664`, `PE32+`) |
| `VCRUNTIME*.dll`, `MSVCP*.dll`, `MSVCR*.dll`, `concrt*.dll`, `ucrtbase.dll` imports | **none** |
| Unexpected non-system DLL imports | **none** — import tables list only Windows system DLLs |
| Export verifier verdict | **`RESULT: PASS`** (7 executables, `DEPENDENCIES.txt`) |

**Correction to the expected dependency report.** `vulkan-1.dll` does **not**
appear in `DEPENDENCIES.txt`, because it is not a static import of any executable:
the Vulkan loader is resolved at runtime and each Vulkan-enabled binary contains
the `vulkan-1.dll` load string exactly once. The verifier's external-DLL allowlist
still permits `vulkan-1.dll` if it were imported. The loader and the GPU driver
remain **external runtime requirements and are not bundled**.

## Per-executable hashes

| Executable | SHA-256 |
| --- | --- |
| `hierarchos-native.exe` | `82125895d444cc347511874e22ec9fad17ad854d9d4abf57f15c8cdbe2c7fee8` |
| `hierarchos-native-cli.exe` | `870b34f5f539cc154d5a4173f00a25978016b696cdb63941cfbbb67610a77abc` |
| `hierarchos-infer.exe` | `82e98dc1f9934ae1241227fd191588f674de21c6ead388f7bc18d0c16fb28e61` |
| `hierarchos-vulkan-train.exe` | `a49d1c4519017fe2c27a409fae90a417c67a59fbec03068e12236f052dea2a73` |
| `hierarchos-vulkan-devices.exe` | `45077010b9a53c74c8c40243ec0cd702932185f3611f341699e44595aad6beee` |
| `hierarchos-vulkan-transformer-train.exe` | `c4182014f0207d9b87fc8f9502ffe13eba5355eb58d19002c1f7f89e222d392f` |
| `hierarchos-vulkan-transformer-logits.exe` | `6b662bcce831be3f8939ccd60133894a280d292fd553ae880e2353925ca8636c` |

## Startup and diagnostics

Run from the extracted directory (all exit statuses captured directly, not
through a pipe):

| Command | Exit | Observed |
| --- | ---: | --- |
| `bin\hierarchos-native-cli.exe --help` | 0 | full command surface printed |
| `… doctor` | 0 | `Vulkan compute devices: 1`, `[0] AMD Radeon Graphics (INTEGRATED_GPU, queue family 0)`, both companions resolved with absolute paths inside `bin\`, `Transformer registry: 146 canonical + 83 aliases` |
| `… devices` | 0 | one JSON device record, `"name":"AMD Radeon Graphics"`, `"device_type":"INTEGRATED_GPU"` |
| `… architectures` | 0 | `146 canonical + 83 package aliases` |
| `… precision-info` | 0 | `full_fp16_supported: true`, `full_fp16_qualified: true`, `full_fp16_max_reduction: 64`, `device_key "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]"` |

Companion executables resolve from `bin/` — confirmed by `doctor` printing the
two companion paths inside the extracted `bin\` directory, and by the seven-file
`bin\` listing.

## Functional smoke tests

Model fixture: `hf-internal-testing/tiny-random-gpt2` from the per-user cache
(`%LOCALAPPDATA%\Hierarchos\hf-cache`). It is an untrained random-weight model, so
gibberish output is the expected result; the test is that the path runs.

| Test | Result |
| --- | --- |
| Real inference, local model | **PASS** — `infer --prompt "Hello" --max-new-tokens 8 --no-do-sample` exit 0, emitted `" 200 200 200 reg reg regodod"` |
| Native Vulkan training | **PASS** — `transformer-train … --epochs 1 --batch-size 1 --seq-len 16 --lr 5e-5`: `steps=1 mean_loss=6.914904`, `backend=Vulkan device=AMD Radeon Graphics`, exit 0 |
| Checkpoint reload | **PASS** — `infer` against the produced directory exit 0; the output contains `config.json`, `model.safetensors`, tokenizer assets and `training_report.json` |
| Relocation | **PASS** — directory moved to a deeper path containing spaces; `--help`, `doctor` and `precision-info` all still exit 0 and still report the adapter |
| Global WMMA default | **PASS** — the training run's `WMMA evidence` block reports `"requested":"off"` with `cooperative_dispatches: 0`, confirming opt-in behaviour |

## Precision policy, as observed from the package

- Global WMMA is **off by default**: the training run with no mode flag reported
  `requested:"off"` and zero cooperative dispatches.
- `--wmma-mode full-fp16` is accepted and announces
  `WMMA: dense Transformer projections will request full-fp16`, then **refuses**
  (exit 1) because the packaged trainer qualifies Full FP16 only for the 64-wide,
  one-layer Llama profile. This is a deliberate gate, not a silent fallback.
- The exact qualified P3 module is embedded in the packaged binaries: the
  complete 23,280-byte SPIR-V blob (SHA-256 `2aff4123…`) appears **verbatim,
  exactly once**, in `hierarchos-native-cli.exe`,
  `hierarchos-vulkan-transformer-train.exe` and
  `hierarchos-vulkan-transformer-logits.exe`. The three binaries that never build
  a WMMA pipeline (`hierarchos-infer.exe`, `hierarchos-native.exe`,
  `hierarchos-vulkan-devices.exe`) contain zero copies.

## Proof of an actual P3 production dispatch — and what is blocked

An **above-threshold cooperative dispatch could not be exercised from inside the
packaged archive**, and this is reported as **BLOCKED**, not as a pass.

The reason is structural and is a property of the unchanged gating, not of this
package build:

- Cooperative admission requires `rows × columns × reduction ≥ 16,000,000`
  (`wmma::wmma_shape_profitable`).
- The packaged trainer's only qualified Full FP16 profile is 64-wide with
  33–64 rows, i.e. at most `64 × 64 × 64 = 262,144` MACs — about **61× below the
  floor** — so the same predicate that protects production declines it.
- No packaged executable ships a raw GEMM benchmark, so the above-threshold path
  cannot be re-driven from the archive alone.

The dispatch was therefore proven with the archived production harness on the
same adapter, and with the same binaries' code path:

| Evidence | Value |
| --- | --- |
| Harness | `wmma_bench` (`HIERARCHOS_WMMA_MODE` unset, `--wmma-mode full-fp16`), archived as `phase_lxxxiv_evidence/bin/wmma_bench.exe` |
| Self-reported variant | `full_fp16_variant: p3` |
| Self-reported pipeline identity | `wmma_gemm_f16_f16acc_twowave_p3/spirv-fnv1a64:0x076216ad97d7605f/len:23280` |
| Cooperative dispatches observed | 42 in the profile stream, with `shader=wmma_gemm_f16_f16acc_twowave_p3` records |
| Adapter | `1002:15bf`, same device the packaged `precision-info` reports |

A capability report alone would not prove a dispatch; the pipeline identity above
comes from an executed cooperative dispatch, and the same 23,280-byte module is
present in the packaged executables.

## Blocked or unverified

- **Above-threshold cooperative P3 dispatch from the packaged executables** —
  blocked by the profitability floor versus the only qualified training profile,
  as described above.
- **GUI window interaction.** The GUI executable is present and was not driven:
  no mouse/UI Automation harness is available in this environment. Reported as
  **BLOCKED**, not as a pass.
- **Clean-machine certification.** No independent Windows installation, second
  physical CPU/GPU, or physical network-isolation test was available. CPU
  compatibility is analytic only: default `x86_64-pc-windows-msvc` target, no
  `target-cpu=native`, existing runtime feature dispatch retained.
- **Different-drive relocation.** Relocation was tested within the same volume,
  into deeper paths containing spaces.
- **No end-to-end training-step speedup is claimed anywhere.** The qualification
  numbers are per-dispatch GEMM kernel timings.

## Numerical and performance qualification for these binaries

The full record is `phase_lxxxiv_evidence/` in the repository
(`PHASE_LXXXIV_REPORT.md`, 310-entry `SHA256SUMS_phase_lxxxiv.txt`, 40 accuracy
receipts, raw paired timing samples). Summary on the measured `1002:15bf`
adapter: 37/37 numerical gates at all eight seeds on the production route, 32/32
preserved K=96 inputs and outputs bit-identical to the Phase LXXIX P3 tensors,
and all four protected shapes above 1.0 on both the point estimate and the
corrected 95 % confidence lower bound.
