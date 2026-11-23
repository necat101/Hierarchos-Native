# Hierarchos Native — Windows x64 Portable v0.2.6 Acceptance Report

Status: **LOCAL ACCEPTANCE PASS, with explicitly blocked items. Packaged C1
dispatch and packaged P3 fallback both verified from the extracted, relocated
archive.**

This is a same-host extracted-artifact acceptance report for the v0.2.6 binaries.
It does not re-use, and does not replace, the v0.2.5 report: v0.2.6 changes the
Full FP16 staging schedule, so the dispatch evidence was collected again from the
new package.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-Portable-v0.2.6.zip`
- Packaging source commit: the exact checked-out HEAD at packaging, recorded in
  the archive's `BUILD_INFO.txt` and in `BUILD_MANIFEST.json` beside the archive.
- Qualified source commit: `b2f5bee` (`codex/wmma-v026-optimization`), which is
  the commit the numerical and performance qualification in
  `validation/phase_lxxxviii_c1_promotion/` was run against. Every later commit
  adds only documentation, acceptance evidence and checksum sidecars.
- Target: `x86_64-pc-windows-msvc` (default target; **not** `target-cpu=native`)
- Profile: release (`opt-level=3`, thin LTO, `codegen-units=1`)
- CRT linkage: static MSVC C runtime (`-C target-feature=+crt-static`)
- Toolchain: `rustc 1.93.0 (254b59607 2026-01-19)` / LLVM 21.1.8,
  `cargo 1.93.0 (083ac5135 2025-12-15)`
- Standalone `Cargo.lock` SHA-256:
  `f4360adfc4a2127c0be817c71c08659c181f7ff04813b5f8bc882a23ca7132d7` (unchanged
  from v0.2.5 — no dependency was added or upgraded for this release)
- Packager: `hierarchos-vulkan/package-standalone.ps1 -StaticCrt` with the
  dependency scan **enabled** (no `-SkipDependencyScan`)
- The archive SHA-256 is recorded in `BUILD_MANIFEST.json` and `SHA256SUMS` beside
  the ZIP, not inside it.

## Package inventory

```text
bin/hierarchos-native.exe                     (GUI)
bin/hierarchos-native-cli.exe
bin/hierarchos-infer.exe
bin/hierarchos-vulkan-train.exe
bin/hierarchos-vulkan-devices.exe
bin/hierarchos-vulkan-transformer-train.exe
bin/hierarchos-vulkan-transformer-logits.exe
LICENSE   THIRD_PARTY_NOTICES.md   licenses/
README.md   README_ARCHITECTURES.md   RELEASE_NOTES.md
CLEAN_MACHINE_TEST.md   BUILD_INFO.txt   DEPENDENCIES.txt
```

Exactly seven executables. No source tree, Cargo files, shader sources, PDBs,
fixtures, validation tree, test harness or model weights.

## PE closure

All seven executables are x64 PE32+, statically linked against the MSVC C
runtime, with no `VCRUNTIME*`, `MSVCP*`, `MSVCR*`, `concrt*` or `ucrtbase` import.
The packager's portable dependency scan passed with the scan enabled and wrote
the full per-executable import closure to `DEPENDENCIES.txt`. `vulkan-1.dll`
remains an external runtime requirement supplied by the graphics environment; it
is not bundled and is not a substitute for the driver.

## Embedding: both staging modules are inside the WMMA-enabled executables

The v0.2.6 Full FP16 route embeds two SPIR-V modules in the CLI, the transformer
trainer and the logits executable:

| Module | Bytes | SPIR-V SHA-256 | Runtime FNV-1a64 |
| --- | ---: | --- | --- |
| `wmma_gemm_f16_f16acc_twowave_p3` (contract, fallback) | 23,280 | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` | `0x076216ad97d7605f` |
| `wmma_gemm_f16_f16acc_twowave_p3_c1` (batched staging) | 36,912 | `b072242034ba7bd0f53690c38c03b11ed587b538480c04ed13ecf19fe2a8b27d` | `0xfecb2a74c4bbbce4` |

The P3 bytes are the v0.2.5 bytes, unchanged.

## Startup and diagnostics

| Command | Exit | Result |
| --- | ---: | --- |
| `bin\hierarchos-native-cli.exe --help` | 0 | usage rendered |
| `bin\hierarchos-native-cli.exe doctor` | 0 | `Vulkan compute devices: 1`; all companions discovered beside `bin\` |
| `bin\hierarchos-vulkan-devices.exe` | 0 | `AMD Radeon Graphics`, `INTEGRATED_GPU`, index 0 |
| `bin\hierarchos-native-cli.exe precision-info --device-index 0` | 0 | `full_fp16_supported=true`, `mixed_supported=true`, `full_fp16_max_reduction=64` |
| `bin\hierarchos-native-cli.exe architectures` | 0 | registry rendered |

`precision-info` reports `full_fp16_qualified=false` on this host. That field is
`precision_policy::full_fp16_hardware_qualified`, whose driver-key allowlist still
names the v0.2.5 driver key (`…;driver=00800184`) while this host now reports
`…;driver=008001a0`. It is the CLI's frontend profile qualification and it is
unchanged from v0.2.5 — no threshold, predicate or allowlist was edited for
v0.2.6. The production library route exercised below is governed by the
capability contract, the 16M-MAC floor and the reduction cap, which are
unchanged.

## Packaged C1 dispatch: PASS

An above-threshold, C1-eligible cooperative dispatch was exercised **from the
extracted, relocated package**, using the packaged executables and the unchanged
production selector.

### Workload

A disposable deterministic Llama package generated with
`hierarchos-vulkan/validation/make_wmma_fixture.py`:

```text
model_type=llama, hidden_size=64, intermediate_size=4096, vocab_size=4096,
num_hidden_layers=1, heads=8, kv_heads=8, head_dim=8, tie_word_embeddings=false
batch=1, seq_len=64, rows=64
```

`cols=4096`, `reduction=64`: full 32-row tile coverage, a multiple-of-16
reduction and even strides, so the C1 contract holds. Admitted projections reach
`64 × 4096 × 64 = 16,777,216` MACs, above the unchanged 16M floor and inside the
64-element Full FP16 reduction cap.

```powershell
$env:HIERARCHOS_WMMA_MODE = "full-fp16"
$env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"   # diagnostics only; selects nothing
.\bin\hierarchos-vulkan-transformer-train.exe `
  --model "C:\scratch\v026 acceptance\fixture_c1\model" `
  --dataset "C:\scratch\v026 acceptance\fixture_c1\dataset.jsonl" `
  --output "C:\scratch\v026 acceptance\train_c1" `
  --batch-size 1 --seq-len 64 --epochs 1 --max-steps 1 --device-index 0 --seed 42
```

Exit 0. Observed:

```text
WMMA: full-fp16 selected for dense Transformer projections (... pci=1002:15bf ...)
WMMA evidence: requested=full-fp16 cooperative_dispatches=8 shape_fallbacks=4
  (forward=3 weight_grad=4 input_grad=1)
hierarchos_vulkan_kernel_profile_shader shader=wmma_gemm_f16_f16acc_twowave_p3_c1
  category=other dispatches=8 gpu_ms=0.862720 ...
```

`HIERARCHOS_WMMA_FULL_FP16_VARIANT` was unset. No variant override, no floor
change, no test flag: the production selector chose the staging module.

## Packaged P3 fallback dispatch: PASS

The same fixture regenerated with `seq_len=80` gives `rows=80`, so `80 % 32 = 16`
and the C1 tile-coverage condition fails. The geometry stays above the floor
(`80 × 4096 × 64 = 20,971,520` MACs) and inside the reduction cap, so it is
admitted to Full FP16 and must fall back to the original module:

```text
WMMA evidence: requested=full-fp16 cooperative_dispatches=4 shape_fallbacks=8
  (forward=3 weight_grad=0 input_grad=1)
hierarchos_vulkan_kernel_profile_shader shader=wmma_gemm_f16_f16acc_twowave_p3
  category=other dispatches=4 gpu_ms=0.707040 ...
```

The mask-kernel difference between the two runs is the same selector acting on
rows: at 80 rows the weight-gradient projection is not tile-covered either, so it
falls back to FP32 as well. Nothing about the fallback counters changed.

## Packaged route table

| Run | Mode | Cooperative dispatches | Staging module observed |
| --- | --- | ---: | --- |
| logits, `rows=64`, `HIERARCHOS_WMMA_MODE=off` | off | 0 | absent |
| logits, `rows=64`, `full-fp16` | full-fp16 | 3 | `…_p3_c1` |
| logits, `rows=80`, `off` | off | 0 | absent |
| logits, `rows=80`, `full-fp16` | full-fp16 | 3 | `…_p3` |
| trainer, `rows=64`, `full-fp16` | full-fp16 | 8 | `…_p3_c1` |
| trainer, `rows=80`, `full-fp16` | full-fp16 | 4 | `…_p3` |

Zero cooperative dispatches with WMMA off, on both geometries.

## Numerical comparison from the package

`hierarchos-vulkan-transformer-logits.exe` was run with `off` and with
`full-fp16` on the same model and fixture, and the logits were compared element
by element:

| Geometry | Route | Elements | max abs | RMS | Finite |
| --- | --- | ---: | ---: | ---: | --- |
| `rows=64` | Full FP16 / C1 vs FP32 | 262,144 | `0.00079345` | `0.00012127` | yes |
| `rows=80` | Full FP16 / P3 fallback vs FP32 | 327,680 | `0.00079345` | `0.00012087` | yes |

Both are inside the project's `0.05` max-abs / `0.01` RMS Full FP16 component
comparison limits. This is a functional comparison on a synthetic geometry, not
the release oracle gate; the release numerical gate is the 37/37 per-seed result
and the 120/120 bit-identity comparison in
`validation/phase_lxxxviii_c1_promotion/`.

## Functional smoke tests

| Test | Result |
| --- | --- |
| full-finetune WMMA training, one step, `rows=64` | exit 0, 8 cooperative dispatches, `training_report.json` written |
| full-finetune WMMA training, one step, `rows=80` | exit 0, 4 cooperative dispatches |
| LoRA training, one step | exit 0, adapter + AdamW state written |
| exact checkpoint reload (`--resume-training-state`) | exit 0, `exact resume: adapter weights, AdamW moments, optimizer step and dropout seed restored` |
| FP32 logits inference + comparison | exit 0 (table above) |
| relocation to a deeper path with spaces | exit 0, same dispatch evidence |
| archive checksum verification | PASS |

`hierarchos-vulkan-train.exe` (the native V4.1 trainer) starts and prints usage,
but its functional smoke is **BLOCKED**: it requires a native V4.1 model package
(`--model` with that architecture's config) and no such package or weights are
available in this environment. `hierarchos-infer.exe` is **BLOCKED** for the same
reason — it loads a native runtime package rather than the Llama fixture used
here, and it exits with an I/O error on that fixture. Both executables are
unchanged by this release; neither is on the Full FP16 path.

## Precision policy, as observed from the package

- Global WMMA is off by default: the `off` runs recorded zero cooperative
  dispatches and no staging module in the profile.
- `full-fp16` is opt-in and was the only mode variable set.
- No environment variable, flag or variant name selected a staging module.
  `HIERARCHOS_WMMA_FULL_FP16_VARIANT` was unset in every run; C1 and P3 were
  chosen by the geometry alone.
- `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` was set for the dispatch-evidence runs.
  It enables GPU timestamp profiling output; it does not influence selection, and
  the same module was observed with it disabled (exit-0 runs without it produced
  identical `cooperative_dispatches` counts).

## Blocked or unverified

- Independent clean Windows installation. **BLOCKED.**
- A second physical CPU/GPU, or any non-AMD adapter. **BLOCKED.**
- Physical network isolation and different-drive relocation. **BLOCKED.**
- GUI interaction and automated GUI “Check backend” activation. **BLOCKED.**
- Code signing. The binaries are **unsigned**.
- `hierarchos-vulkan-train` and `hierarchos-infer` functional paths, as described
  above. **BLOCKED.**
- RGP, ISA, VGPR or occupancy counters. None were obtained; none are claimed.

## Numerical and performance qualification for these binaries

The release gates were run against the source commit this package was built from,
not against a different build:

- Eight seeds, 40 receipts, `37/37` gates on the production tile16 route at every
  seed, maximum tile16 distance `0.03125` — the unchanged hard gate value.
- `cases=120 bit_identical=120 differing=0` against the v0.2.5 P3 baseline.
- 24 paired blocks per protected shape: `1.7074×`, `2.0920×`, `1.3286×` and
  `1.3782×` over v0.2.5, every corrected 95 % CI lower bound above `1.31`.
- 17/17 selector sweep: ten aligned neighbours to C1, seven unaligned to P3.

Those are per-dispatch GPU GEMM timings, not end-to-end training-step speedups.
No end-to-end training speedup is claimed.

## Release status

Published as `v0.2.6`. Published contents are immutable; any further change is a
new version.
