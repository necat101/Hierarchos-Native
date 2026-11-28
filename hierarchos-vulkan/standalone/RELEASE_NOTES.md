# Hierarchos Native — Windows x64 Portable v0.2.8

v0.2.8 is a runtime-only, extract-and-run Windows x64 package. It promotes two
qualified query-pass changes and one FP32 input-gradient kernel into the default
RDNA3 training route: the Phase XVIII literal-index dQ accumulator, the Phase XX
scratch-reuse variant, and the Phase XXII output-tiled `linear_input_grad`
kernel. The original query shader and the original input-gradient kernel remain
the fallbacks. The WMMA precision
policy is unchanged from v0.2.7: global WMMA stays **off by default**, Mixed WMMA
stays **opt-in and default-off**, and both the C1 batched-staging module and the
P3 contract module remain embedded. The executables are linked with the static
MSVC C runtime, so no Microsoft Visual C++ Redistributable is required, and no
Rust, Cargo, Python, PyTorch, Visual Studio or shader compiler is needed to run
them.

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

## What changed since v0.2.7

Three kernel changes are promoted. They are **independent complete-step
measurements**, taken one at a time on the same host; the ratios are not
multiplied into an overall training speedup and no such number is claimed.

1. **Phase XVIII literal-index dQ accumulator.** On the geometry the existing
   query selector already admitted (QK and value head dimension 128), the query
   pass indexes its dQ accumulator with literals instead of a runtime chain.
   The Phase XVIII campaign measured eight independent geometry/mode series,
   every one 5/5 ABBA rounds improved, **1.042×–1.124×** on the complete step
   with a 95 % CI excluding 1.0 in every series. The original query shader is
   the fallback, selected by `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0`.
2. **Phase XX scratch reuse is the default on that qualified path.** The query
   pass now reuses its softmax/dS scratch instead of rewriting it. The Phase XIX
   five-round, 160-process campaign measured the complete step at **1.088×**
   FP32 (26.799 → 24.643 ms, 95 % CI 0.9176–0.9210) and **1.120×** Mixed
   (21.386 → 19.111 ms, 95 % CI 0.8874–0.8977), 5/5 rounds improved in both.
   The six additional regression series improved 1.071×–1.148×.
   `HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0` selects the Phase XVIII
   dQ-unroll module instead.
3. **The FP32 input gradient is output-tiled for 32-aligned shapes.** The new
   kernel owns a 32 x 32 output tile with four outputs per thread, preserving
   each output's ascending-`k` single-`fma` reduction. The five-round ABBA
   measured the complete step at **1.1462×** (24.875 → 21.716 ms, 95 % CI
   0.8706–0.8742, 5/5 rounds), with 272,038 of 272,038 output words bitwise
   equal over six shapes. Shapes outside the 32-aligned partition keep the
   original kernel; `HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0` restores it for
   every shape.

Selector order for pass 1:

| order | condition | route |
| ---: | --- | --- |
| 1 | Phase XVIII selector admits, scratch reuse enabled, artifact loads | `query_dqunroll_scratchreuse` |
| 2 | scratch reuse disabled or its pipeline refused | `query_dqunroll` |
| 3 | dQ-unroll disabled or geometry not admitted | original `query_rdna3` |

The enclosing two-pass gate is unchanged: modes 0/9, one key-range block, no
window, softcap, ALiBi, bias, dropout, sink, offset, sparse or compressed keys,
paged KV or value scale; `head_dim <= 128`, `kv_seq <= head_dim`; grouped
geometries remain limited to `8q/2kv`, `4q/2kv`, `8q/4kv` and `16q/4kv` at head
dimension 128 with `q_seq == kv_seq == 64` and causal masking. No eligibility
allowlist was widened. A module-specific load refusal is an ordinary fallback;
device-lost, out-of-memory and poisoned-layout failures stay fatal.

## Numerical qualification

- Phase XIX parity: 28 cases, 140 tensor comparisons of `dQ`,
  `scratch_probability`, `scratch_ds`, `dK` and `dV`; **140/140 bitwise equal**,
  zero non-finite values.
- Phase XX in-graph equivalence: 29 complete steps, 87 comparisons, zero
  mismatches; 29/29 loss bits, LoRA gradient digests and parameter digests
  equal; `parameter_end_digest = d79a12d8…9c4655` in both arms, through the real
  production selector, buffers and dispatch.
- Phase XVIII fallback matrix re-read with scratch reuse pinned off: 30 cases,
  150 tensor comparisons, zero failures. No campaign was re-run and no tolerance
  moved.

## WMMA and Mixed are unchanged

- **Global WMMA is off by default.** No cooperative-matrix dispatch happens
  unless a mode is requested.
- **`--wmma-mode full-fp16`** still runs the C1 batched-staging module when the
  host proves every packed load aligned and in bounds; the 23,280-byte P3 module
  remains the automatic fallback.
- **`--wmma-mode mixed`** remains a separate, opt-in arm and remains default-off.
  Its thresholds, full-parameter learning-rate caveat and adapter elementwise
  drift observations are unchanged from v0.2.7. No per-element adapter-weight
  parity is claimed.
- `FULL_FP16_MAX_REDUCTION = 64`, `WMMA_MIN_PROFITABLE_MACS = 16_000_000` and the
  hard numerical gate `0.03125` are unchanged.

## Scope and limitations

- **One device.** Every measurement was taken on the integrated AMD RDNA3
  adapter identified by PCI `1002:15bf` (`AMD Radeon Graphics`), driver
  `32.0.32015.14001`, Windows 11 x64. **No universal AMD, NVIDIA, Intel or
  cross-architecture qualification is implied.**
- Query-pass eligibility is head dimension 128 inside the existing two-pass
  gate. Other head dimensions keep the original query shader.
- The Phase XII four-channel key pass, row-packed forward and GQA grouped key
  pass keep their existing artifacts, gates and control arms.
- The original FP32 input-gradient kernel remains the fallback for shapes
  outside the 32-aligned partition and for the explicit opt-out.
- Timings are per-step milliseconds on one machine: no throughput, power or
  multi-GPU claim. Results, including items reported as **BLOCKED** rather than
  passed, are in `CLEAN_MACHINE_TEST.md`.

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
