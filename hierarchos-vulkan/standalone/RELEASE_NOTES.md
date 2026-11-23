# Hierarchos Native — Windows x64 Portable v0.2.6

v0.2.6 is a runtime-only, extract-and-run Windows x64 package. It ships the
production Full FP16 WMMA route with the **C1 batched staging schedule** in front
of the unchanged P3 contract, keeps the v0.2.5 P3 module as the automatic
fallback, and leaves global WMMA off by default. The executables are linked with
the static MSVC C runtime, so no Microsoft Visual C++ Redistributable is
required, and no Rust, Cargo, Python, PyTorch, Visual Studio or shader compiler
is needed to run them.

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

## What changed since v0.2.5

- **C1 batched staging.** Inside `--wmma-mode full-fp16` the per-slice staging
  loop used to issue four dependent global-load → shared-store pairs per thread.
  It now issues the four 32-bit operand loads first and the shared stores after
  them, so the loads overlap instead of queueing behind stores. That is the whole
  change: the two-wave workgroup, the K-slice stream, the per-bank binary16
  accumulation, the float32 sum of the eight bank partials, the single binary16
  rounding and the barrier placement are P3's, untouched.
- **The original P3 module is still shipped and still dispatched.** It serves
  every geometry the C1 contract cannot prove — partial tiles, K not a multiple
  of 16, odd packed-pair strides, operand buffers too short for the fast path's
  last element, and indices near the 32-bit ceiling. It is a fallback, not a
  legacy option, and both modules are embedded in the same executables.
- **No new mode, flag or variant.** Users still select `full-fp16`; nothing was
  added that could force one staging schedule or the other. Which one ran is
  visible in the per-dispatch staging counters of a receipt.
- **Accuracy.** Eight seeds, 40 receipts, 37/37 gates on the production tile16
  route at every seed, and all 120 production outputs **bit-identical** to the
  v0.2.5 P3 baseline. No reference, tolerance, seed, oracle or threshold changed;
  the `split2` and `rdna3-dot2` diagnostic failures remain recorded as
  diagnostics.
- **Performance.** The integrated production selector is `1.33x`–`2.09x` faster
  than the v0.2.5 P3 route on the four protected shapes, with every corrected
  95 % confidence lower bound above `1.31`.

## WMMA is opt-in

Global WMMA is **off by default**. `--wmma-mode full-fp16` explicitly enables the
Full FP16 path. Mixed mode is a separate, unchanged arm. The full statement of
the precision policy is in `README.md`.

Admission requires AMD RDNA3-or-newer plus the complete cooperative feature
contract, and at least `WMMA_MIN_PROFITABLE_MACS = 16_000_000` multiply-
accumulate operations. The hard numerical gate remains `0.03125`; the historical
25 % tile16 headroom target remains informational and is not met.

## Measured performance (per-dispatch kernel timings)

Integrated production selector (C1 where the contract holds, P3 otherwise) vs the
**v0.2.5 P3 release binary**, 24 paired blocks per shape, position-balanced
ABBA/BAAB ordering, GPU timestamp queries around the compute dispatch, strictly
serial execution, identical tensors and dispatch geometry. Ratio = v0.2.5 ms /
v0.2.6 ms; > 1.0 means v0.2.6 is faster.

| Shape | v0.2.5 ms | v0.2.6 ms | Speedup | Corrected 95 % CI |
| --- | ---: | ---: | ---: | --- |
| 2048x768x2048 | 5.0217 | 2.9347 | 1.7074 | [1.6976, 1.7171] |
| 512x2048x768 | 1.2376 | 0.9316 | 1.3286 | [1.3182, 1.3391] |
| 512x768x768 | 0.5352 | 0.2570 | 2.0920 | [2.0415, 2.1425] |
| 128x8192x768 | 1.3955 | 1.0067 | 1.3782 | [1.3580, 1.3983] |

These are **per-dispatch GPU GEMM timings, not end-to-end training-step
speedups**. No end-to-end training speedup is claimed.

The aligned 32×32-tile geometries above all dispatched the C1 module; the
boundary sweep in the validation tree records the same for ten aligned
neighbours and the original P3 module for seven unaligned ones. Shapes below the
16M-MAC floor — including the `16x128x32` direct-kernel control — never enter the
cooperative path at all, because the unchanged profitability predicate refuses
them.

## Hardware scope

All measurements were taken on one adapter: AMD RDNA3, PCI `1002:15bf`,
`AMD Radeon Graphics`, driver `32.0.32015.14001`, Windows 11 x64. That is a
different driver build from v0.2.5's, so absolute milliseconds are not comparable
between the two release notes; the ratios above are paired measurements inside
one driver session and are. The qualification does not generalize to other RDNA3
adapters, RDNA3.5, RDNA4 or non-AMD devices.

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

The v0.2.6 archive was extracted on the available Windows 11 / AMD RDNA3 host and
tested from the extracted layout, including from a path containing spaces, then
relocated to a deeper path containing spaces. The local acceptance run covered
all seven executables, the PE import closure, CLI startup, `doctor`, device
enumeration, the architecture registry, precision reporting, a real inference
smoke test, a native Vulkan training and checkpoint-reload smoke test, archive
checksum verification, and relocation.

The packaged dispatch check passed on the measured adapter, covering both routes:
an eligible geometry ran the C1 module and an ineligible one fell back to the
original P3 module, with no override and no threshold change. The exact geometry,
counters and profile stream are recorded in `CLEAN_MACHINE_TEST.md`.

Results, including items that could not be exercised and are reported as
**BLOCKED** rather than passed, are in `CLEAN_MACHINE_TEST.md`.

No independent clean Windows installation, second physical CPU/GPU, physical
network-isolation test, or different-drive relocation was available.
