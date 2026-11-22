# Hierarchos Native — Windows x64 Portable v0.2.5

**Package status: prepared and locally validated, not yet published.** The
GitHub release and tag wait for PR #9 to merge. `BUILD_MANIFEST.json` records the
exact packaging commit and hashes.

v0.2.5 is a runtime-only, extract-and-run Windows x64 package. It supersedes the
RC2 documentation, which described an older source commit and did not ship the
production P3 Full FP16 epilogue. The executables are linked with the static MSVC
C runtime, so no Microsoft Visual C++ Redistributable is required, and no Rust,
Cargo, Python, PyTorch, Visual Studio or shader compiler is needed to run them.

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

## What changed since RC2

- **Production P3 epilogue.** The Full FP16 WMMA arm now defaults to the P3
  epilogue module inside `--wmma-mode full-fp16`. Bank assignment, the K-slice
  stream, the two-wave workgroup structure, the cooperative multiply, the A/B
  staging and the barrier placement are unchanged; the change is the epilogue:
  the eight binary16 bank partials are merged in float32 and rounded once to
  binary16, instead of being reduced through seven binary16 roundings.
- **Accuracy.** The production route reproduces the qualified P3 output bit for
  bit on all 32 preserved K=96 cases and passes 37/37 numerical gates at all
  eight qualification seeds. The maximum tile16 distance is the unchanged hard
  gate value `0.03125`.
- **Performance.** P3 is faster than the preserved Main baseline on every one of
  the four protected production shapes, with a corrected 95 % confidence lower
  bound above 1.0 on each. See the table below.
- **Documentation.** The package documentation was rewritten for v0.2.5. The RC2
  claims, including its older source commit, are no longer shipped.

## WMMA is opt-in

Global WMMA is **off by default**. `--wmma-mode full-fp16` explicitly enables the
Full FP16 path, and P3 is the default variant inside that mode. Mixed mode is a
separate, unchanged arm. The full statement of the precision policy is in
`README.md`.

Admission requires AMD RDNA3-or-newer plus the complete cooperative feature
contract, and at least `WMMA_MIN_PROFITABLE_MACS = 16_000_000` multiply-
accumulate operations. The hard numerical gate remains `0.03125`; the historical
25 % tile16 headroom target remains informational and is not met.

## Measured performance (per-dispatch kernel timings)

P3 vs the preserved Main baseline, 24 paired blocks per shape, position-balanced
ABBA/BAAB ordering, GPU timestamp queries around the compute dispatch, strictly
serial execution. Ratio = Main ms / P3 ms; > 1.0 means P3 is faster.

| Shape | Main ms | P3 ms | Speedup | Corrected 95 % CI |
| --- | ---: | ---: | ---: | --- |
| 2048x768x2048 | 5.4193 | 5.3109 | 1.0243 | [1.0159, 1.0326] |
| 512x2048x768 | 1.4945 | 1.3332 | 1.1246 | [1.1167, 1.1325] |
| 512x768x768 | 0.6088 | 0.5486 | 1.1163 | [1.0879, 1.1446] |
| 128x8192x768 | 1.8102 | 1.4260 | 1.2722 | [1.2584, 1.2860] |

These are **per-dispatch GPU GEMM timings, not end-to-end training-step
speedups**. No end-to-end training speedup is claimed.

The `16x128x32` direct-kernel control is a disclosed regression (0.7566,
[0.7504, 0.7627]). At 65,536 MACs it is roughly 244x below the profitability
floor and fails the tile-fill requirement, so the production route never
dispatches WMMA there. The regression is excluded from production acceptance by
the unchanged predicate, not hidden.

## Hardware scope

All measurements were taken on one adapter: AMD RDNA3, PCI `1002:15bf`,
`AMD Radeon Graphics`, Vulkan 1.4.344, driver 2.0.388. The qualification does not
generalize to other RDNA3 adapters, RDNA3.5, RDNA4 or non-AMD devices.

## Requirements

- Windows x64. Windows 11 x64 is the measured acceptance target; Windows 10 x64
  should work but was not measured.
- **No Microsoft Visual C++ Redistributable installation is required.** The
  release executables are statically linked against the MSVC C runtime and their
  PE import tables are checked during packaging.
- A Vulkan 1.x loader and a Vulkan compute-capable GPU driver. The loader
  (`vulkan-1.dll`) is supplied by the graphics software environment and is **not
  bundled**; it is not a substitute for the driver.
- Enough RAM and disk for the selected model. **No model weights are bundled.**
- No administrator rights are required for normal operation. Downloaded model
  data is cached under `%LOCALAPPDATA%\Hierarchos\hf-cache` by default; override
  it with `HIERARCHOS_HF_CACHE` or `--hf-cache-dir`.

Windows system DLLs, the GPU driver and its Vulkan loader are the intended
external runtime dependencies. This package does not include a
Rust/Cargo/Python/PyTorch toolchain, Visual Studio, an installer, a registry
step, or a shader compiler.

## Verification status

The v0.2.5 archive was extracted on the available Windows 11 / AMD RDNA3 host and
tested from the extracted layout, including from a path containing spaces, then
relocated to a deeper path containing spaces. The local acceptance run covered
all seven executables, the PE import closure, CLI startup, `doctor`, device
enumeration, the architecture registry, precision reporting, a real inference
smoke test, a native Vulkan training and checkpoint-reload smoke test, archive
checksum verification, and relocation.

The packaged P3 dispatch check passed on the measured adapter. A disposable Llama
geometry with hidden/K = 64 and 4096-wide projections gives 16,777,216 MACs per
admitted projection, above the unchanged 16M floor and within the Full FP16
reduction cap. The extracted trainer reported `cooperative_dispatches=8`, the
profile named `wmma_gemm_f16_f16acc_twowave_p3`, and the packaged logits
executable matched FP32 within `max_abs 0.000738` / `rms 0.000122`. No variant
override or threshold change was used. The geometry is outside the CLI's
frontend-qualified 64-wide training profile; the test uses the packaged
standalone trainer's production path. See `CLEAN_MACHINE_TEST.md`.

Results, including items that could not be exercised and are reported as
**BLOCKED** rather than passed, are in `CLEAN_MACHINE_TEST.md`.

No independent clean Windows installation, second physical CPU/GPU, physical
network-isolation test, or different-drive relocation was available.
