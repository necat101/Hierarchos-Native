# Hierarchos Native — Windows x64 Portable v0.2.5 Acceptance Report

Date: 2026-11-22

Status: **LOCAL ACCEPTANCE PASS, with explicitly blocked items. Packaged P3
dispatch: PASS on the measured adapter.**

This is a same-host extracted-artifact acceptance report for the v0.2.5 binaries.
An independent Windows installation, a second physical CPU/GPU and physical
network isolation were not available. The result below is limited to the measured
Windows 11 / AMD RDNA3 host.

This report describes the **newly built v0.2.5 binaries**. It does not re-use, and
must not be confused with, the older RC2 acceptance report, which described a
different source commit and did not ship the production P3 epilogue.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-Portable-v0.2.5.zip`
- Packaging source commit: the exact checked-out HEAD at packaging, recorded in
  the archive's `BUILD_INFO.txt` and in `BUILD_MANIFEST.json` beside the archive.
  It is the final commit of the release branch, not the earlier `f47989b` build.
- Qualified kernel source commit: `b165c2994e9ef5929c9e36fca76192db8dd3df35`
- Relation between the two: every commit after the qualified source adds only
  documentation, acceptance/evidence records and checksum sidecars. No file under
  `hierarchos-vulkan/src`, `hierarchos-vulkan/shaders`, `hierarchos-inference`,
  `hierarchos-native-cli` or `hierarchos-gui` differs from the qualified commit.
- Target: `x86_64-pc-windows-msvc` (default target; **not** `target-cpu=native`)
- Profile: release (`opt-level=3`, thin LTO, `codegen-units=1`)
- CRT linkage: static MSVC C runtime (`-C target-feature=+crt-static`)
- Toolchain: `rustc 1.93.0 (254b59607 2026-01-19)` / LLVM 21.1.8,
  `cargo 1.93.0 (083ac5135 2025-12-15)`
- Standalone `Cargo.lock` SHA-256:
  `f4360adfc4a2127c0be817c71c08659c181f7ff04813b5f8bc882a23ca7132d7`
- Packager: `hierarchos-vulkan/package-standalone.ps1 -StaticCrt` with the
  dependency scan **enabled** (no `-SkipDependencyScan`)
- The archive SHA-256 is recorded in `BUILD_MANIFEST.json` and `SHA256SUMS` beside
  the ZIP, not inside it.

## Package inventory

Extracted into a directory whose path contains spaces, then relocated to a deeper
path that also contains spaces. The final smoke tests were repeated from the
relocated package.

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
`.comp`, `.glsl`, `.pdb`, `.lib`, `.rlib`, `.d`, fixtures, model weights, caches,
test harnesses and developer logs. The assembly step enforces this: it raises
rather than producing an archive that contains any path outside the runtime
allowlist.

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

The archive's `BUILD_MANIFEST.json` records the byte count and SHA-256 of all seven
executables, and `BUILD_INFO.txt` repeats the hashes. This report does not
duplicate them, so it cannot drift from the shipped artifact.

## Startup and diagnostics

Run from the extracted and relocated directory (all exit statuses captured
directly, not through a pipe):

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
| Archive checksum | **PASS** — `SHA256SUMS` verification matched the ZIP and `BUILD_MANIFEST.json` |
| Global WMMA default | **PASS** — the training run's `WMMA evidence` block reports `"requested":"off"` with `cooperative_dispatches: 0`, confirming opt-in behaviour |

## Precision policy, as observed from the package

- Global WMMA is **off by default**: the training run with no mode flag reported
  `requested:"off"` and zero cooperative dispatches.
- `--wmma-mode full-fp16` is accepted by the CLI and announces the request, then
  **refuses** (exit 1) for a model outside the frontend-qualified 64-wide,
  one-layer Llama profile. This is a deliberate frontend gate, not a silent
  fallback. The standalone trainer and logits executables do not carry that
  frontend profile check, but they enforce the capability contract, the unchanged
  16M MAC floor and the Full FP16 reduction cap.
- The exact qualified P3 module is embedded in the packaged binaries: the
  complete 23,280-byte SPIR-V blob (SHA-256 `2aff4123…`, runtime FNV-1a64
  `0x076216ad97d7605f`) appears **verbatim, exactly once**, in
  `hierarchos-native-cli.exe`, `hierarchos-vulkan-transformer-train.exe` and
  `hierarchos-vulkan-transformer-logits.exe`. The three binaries that never build
  a WMMA pipeline (`hierarchos-infer.exe`, `hierarchos-native.exe`,
  `hierarchos-vulkan-devices.exe`) contain zero copies.

## Packaged P3 production dispatch: PASS

An above-threshold cooperative P3 dispatch was exercised **from the extracted,
relocated package**, using the packaged executables and the unchanged production
selector.

### Workload

A disposable deterministic Llama package was generated with
`hierarchos-vulkan/validation/make_wmma_fixture.py`:

```text
model_type=llama, hidden_size=64, intermediate_size=4096, vocab_size=4096,
num_hidden_layers=1, heads=8, kv_heads=8, head_dim=8, tie_word_embeddings=false
batch=1, seq_len=64, rows=64
```

It keeps the Full FP16 reduction at the qualified 64-element limit and reaches
the profitability floor through the projection width:

| Projection | Shape (rows × columns × reduction) | MACs | Route |
| --- | --- | ---: | --- |
| q/k/v/o | 64 × 64 × 64 | 262,144 | below floor — FP32 fallback |
| gate/up | 64 × 4096 × 64 | 16,777,216 | admitted — Full FP16 |
| down | 64 × 64 × 4096 | 16,777,216 | reduction 4096 > 64 — FP32 fallback |
| lm_head | 64 × 4096 × 64 | 16,777,216 | admitted — Full FP16 |

No floor, threshold, routing predicate or variant was changed. Only the
documented opt-in mode variable was set; `HIERARCHOS_WMMA_FULL_FP16_VARIANT` was
left unset, so the production default selected P3.

### Commands and observed evidence

```powershell
$env:HIERARCHOS_WMMA_MODE = "full-fp16"
.\bin\hierarchos-vulkan-transformer-train.exe `
  --model  C:\scratch\dispatch_fixture\model `
  --dataset C:\scratch\dispatch_fixture\dataset.jsonl `
  --output C:\scratch\dispatch_fixture\train_p3 `
  --batch-size 1 --seq-len 64 --epochs 1 --max-steps 1 `
  --device-index 0 --seed 42
```

Exit 0. Observed on the measured adapter:

```text
WMMA: full-fp16 selected for dense Transformer projections (... pci=1002:15bf ...)
WMMA evidence: requested=full-fp16 cooperative_dispatches=8 shape_fallbacks=4
  (forward=3 weight_grad=4 input_grad=1)
WMMA mode: training=full-finetune lora_rank=None weight_grad_expected=true
  weight_grad_consistent=true
hierarchos_vulkan_kernel_profile_shader shader=wmma_gemm_f16_f16acc_twowave_p3
  dispatches=8 ...
```

The log identifies the adapter (`AMD Radeon Graphics`, PCI `1002:15bf`), the
requested precision (`full-fp16`), the selected route (P3 shader), the
cooperative dispatch count (8), the fallback count (4) and the phase split. The
workload dimensions and MAC counts above come from the exact config and fixture
that the run consumed; they are recorded here rather than printed by the trainer.

The packaged forward path was checked independently with
`hierarchos-vulkan-transformer-logits.exe` on the same model and fixture:

| Run | Cooperative dispatches | P3 shader |
| --- | ---: | --- |
| `HIERARCHOS_WMMA_MODE=off` | 0 | absent |
| `HIERARCHOS_WMMA_MODE=full-fp16` | 3 | `wmma_gemm_f16_f16acc_twowave_p3` |

### Numerical comparison

The Full FP16 logits were compared with the FP32 logits over all 262,144
elements:

| Metric | Value |
| --- | ---: |
| finite | yes |
| max absolute difference | `0.000738` |
| RMS difference | `0.000122` |

That is well inside the project's `0.05` max-abs / `0.01` RMS Full FP16 component
comparison limits. This is a functional comparison on the synthetic geometry, not
the release oracle gate; the release numerical gate is the 37/37 per-seed result
on the qualified profile reported in `phase_lxxxiv_evidence/`.

### Scope limitation, stated explicitly

The synthetic geometry is **outside the CLI's frontend-qualified 64-wide training
profile**. The test therefore uses the packaged standalone trainer and logits
executables, which are production entry points that apply the library's capability
and profitability gates but not the CLI's additional profile whitelist. This
proves the distributed executable dispatches the qualified P3 module through the
normal production selector; it does not broaden the CLI's qualified profile, and
it is not a new numerical qualification of arbitrary Full FP16 geometries.

## Unified-release checker: known CodeView-only mismatch

`check_unified_release.py` was run unchanged in this checkout:

```powershell
python hierarchos-vulkan/validation/check_unified_release.py
```

It exits 1 with:

```text
ValueError: current executable changed: wmma_qualify
```

The local release build is
`0e48f99d14b3a108e4d2b6f8dfc583ac35f3dfb676055ddedaff1ed0e94c5d5f`; the archived
Phase LXXXIV harness is
`138c2866af5eb499bf2e5c9e67505ea47dea9af2cfb7248ed6125b37e3773726`; the Phase 42
receipt records `209c96bda5d47e1137ec6ebe5ca876ee9449b81c872df6886844ceff06dabada`.
All three are 955,392 bytes. The local build differs from the archived harness in
**20 bytes** at offsets 248, 830,436, 830,464, 830,492 and 830,832–830,847, which
lie in the PE debug directory and CodeView `RSDS` record. The accuracy matrix was
previously executed with two of these builds: all 40 receipts and all eight
operand dumps were byte-identical.

With the local release executable absent — the condition hosted CI runs in — the
same checker prints:

```text
UNIFIED RELEASE: PASS_SCOPED_RELEASE (2ef9b8d9642c, 2004 sources verified at
their declared revision, 26 file(s) differ in this checkout)
```

The checker was not edited and no assertion was weakened. The local failure is
binary link-metadata nondeterminism, not a source or functional-output
difference.

## Blocked or unverified

- **Independent clean-machine certification.** No independent Windows
  installation, second physical CPU/GPU, or physical network-isolation test was
  available. CPU compatibility is analytic only: default
  `x86_64-pc-windows-msvc` target, no `target-cpu=native`, existing runtime
  feature dispatch retained.
- **Different-drive relocation.** Relocation was tested within the same volume,
  into deeper paths containing spaces.
- **GUI window interaction.** The GUI executable is present and was not driven:
  no mouse/UI Automation harness is available in this environment. Reported as
  **BLOCKED**, not as a pass.
- **Code signing and automated GUI “Check backend” activation.** Not performed.
- **No end-to-end training-step or application speedup is claimed anywhere.**
  The qualification numbers are per-dispatch GPU GEMM kernel timings.

## Numerical and performance qualification for these binaries

The full record is `phase_lxxxiv_evidence/` in the repository
(`PHASE_LXXXIV_REPORT.md`, 310-entry `SHA256SUMS_phase_lxxxiv.txt`, 40 accuracy
receipts, raw paired timing samples). Summary on the measured `1002:15bf`
adapter: 37/37 numerical gates at all eight seeds on the production route, 32/32
preserved K=96 inputs and outputs bit-identical to the Phase LXXIX P3 tensors,
and all four protected shapes above 1.0 on both the point estimate and the
corrected 95 % confidence lower bound.

| Shape | Speedup | Corrected 95 % CI |
| --- | ---: | --- |
| 2048×768×2048 | 1.0243× | [1.0159, 1.0326] |
| 512×2048×768 | 1.1246× | [1.1167, 1.1325] |
| 512×768×768 | 1.1163× | [1.0879, 1.1446] |
| 128×8192×768 | 1.2722× | [1.2584, 1.2860] |

## Release status

The package is prepared and locally validated. It is **not published**: PR #9 is
open and the GitHub release/tag waits for that PR to merge. No public release is
claimed by this report.
