# Hierarchos Native — Windows x64 Portable v0.2.7

v0.2.7 is a runtime-only, extract-and-run Windows x64 package. It promotes four
attention-path changes into the default training route — a two-pass MHA backward,
a narrowly gated grouped-query (GQA) backward key pass, a cached row-packed
attention forward, and a head-dimension-128 four-channel key pass — while the
WMMA precision policy is unchanged from v0.2.6: global WMMA stays **off by
default**, and both the C1 batched-staging module and the P3 contract module are
still embedded. The executables are linked with the static MSVC C runtime, so no
Microsoft Visual C++ Redistributable is required, and no Rust, Cargo, Python,
PyTorch, Visual Studio or shader compiler is needed to run them.

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

## What changed since v0.2.6

Four attention routes changed. They are **independent kernel and selector
changes**, measured one at a time on the same host; the ratios below are not
multiplied into an overall training speedup and no such number is claimed.

1. **Two-pass attention backward is the default for the existing gate.** The
   scalar fusion-free MHA attention backward was replaced by a query pass that
   writes softmax and `dS` scratch and a key/value pass that reduces it, for the
   geometry the gate already admitted (`unfused_qk` 0/9). Attention backward fell
   from 435.28 ms to 44.43 ms per step (9.7965× on the isolated backward) and the
   complete optimizer step from 560.78 ms to 171.10 ms, a paired **3.2492×**
   (95 % CI 3.1146–3.3838, five ABBA rounds, 230 samples per arm). Gradients are
   bit-exact against the scalar route over four geometries: 6,316,128 of
   6,316,128 FP32 elements, zero ULP. Scratch is sized exactly to
   `batch * heads * q_seq * kv_seq` and allocated only for eligible geometry.
2. **The GQA grouped key pass is promoted for four measured head groupings.**
   The key pass now owns the grouped reduction for `8q/2kv`, `4q/2kv`, `8q/4kv`
   and `16q/4kv` at head dimension 128, `q_seq = kv_seq = 64`, causal, and for
   nothing else. MHA is unchanged bit-for-bit and measured slightly faster with
   the same artifact. The 8q/2kv complete training step measured **19.76×**
   (95 % CI 19.64–19.89) against the scalar route. Every other grouped geometry
   keeps the scalar kernel with a `grouped_query_attention_*` reason and
   allocates no scratch.
3. **Attention forward is row-packed and cached for square sequences ≤ 64.** One
   lane per `(batch, head, query)` keeps its scores in registers and caches the
   `V` rows it reuses, replacing the generic per-key loop. The rule is
   `0 < q_seq == kv_seq <= 64`, equal QK/value head widths, `unfused_qk` in
   `{0, 1, 9}`, plain path only. Forward attention fell from 16.42 ms to
   4.38 ms per step in the Phase X attribution, and the complete shipped-selector
   step measured **1.2239×**. Outputs are bit-exact: 1.85 M of 1.85 M words over
   ten geometries and three arithmetic modes.
4. **The backward key pass is four-channel tiled at head dimension 128.** One
   lane owns a four-channel chunk of a single destination key row, so 32 lanes
   cover one row's 128 channels. Eligibility is `head_dim == value_head_dim ==
   128` with the compile-time partition `32 × 4 = 128`; head dimensions 16, 32
   and 64 keep the original key shader. The complete step measured **1.5297×**
   (43.53 ms → 28.46 ms, 5/5 ABBA rounds), with 20 parity cases over 60 tensors
   and 5,505,024 elements at zero failures, including an MHA control.
   `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` restores the original key
   shader for every geometry and is the control arm of the release A/B.

Every one of those routes keeps its original fallback. An ineligible geometry
dispatches the kernel it had in v0.2.6, the per-dispatch counters in a receipt
report which route actually ran, and `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0`
forces the scalar backward (which also disables the channel-tiled pipeline).
The rejected Phase VIII key-range partitioning is **not** in the production
path: a split backward submission is explicitly refused the two-pass pair and
keeps the scalar kernel.

## WMMA is opt-in, and Full FP16 still runs the P3 contract on C1 staging

Unchanged from v0.2.6:

- **Global WMMA is off by default.** No cooperative-matrix dispatch happens
  unless a mode is requested.
- **`--wmma-mode full-fp16` explicitly enables the Full FP16 path**; the C1
  batched-staging module runs when the host can prove every packed load aligned
  and in bounds, and the 23,280-byte P3 module remains the automatic fallback.
- **`--wmma-mode mixed` is a separate, opt-in arm** and remains default-off. See
  the Mixed WMMA section below for its measured result and its limits.
- Admission still requires AMD RDNA3-or-newer plus the complete cooperative
  feature contract and at least `WMMA_MIN_PROFITABLE_MACS = 16_000_000`
  multiply-accumulates; Full FP16 reductions above
  `FULL_FP16_MAX_REDUCTION = 64` keep their FP32 kernels. The hard numerical gate
  remains `0.03125`. Full FP16 and Mixed remain distinct precision paths.

## Mixed WMMA: requalified, still opt-in

The Mixed arm was requalified against current source in v0.2.7. It is **not**
promoted and its default is unchanged.

| Measurement | Result |
| --- | ---: |
| LoRA complete step, Phase XII harness | 28.481 ms → 23.001 ms = **1.2385×** |
| LoRA complete step, independent counter harness | **1.2288×** |
| Full-parameter complete step (timing evidence only) | **1.2792×** |

- Mixed remains **opt-in and default-off**. The policy switch is process-wide and
  cannot distinguish LoRA from full-parameter training, so default-on would
  promote Mixed to the workload whose numerical qualification is unresolved.
- The full-parameter numerical result at the fixture learning rate `1e-2` is
  **not qualified**: the FP32 reference itself diverges at that rate
  (7.13 → 19.56). The 1.2792× figure is timing evidence only and is not proof of
  numerical qualification. The separate `1e-4` control remains separate evidence
  and is not averaged in.
- The four LoRA adapter tensors show deterministic elementwise drift. Their
  maximum absolute differences are 0.023203, 0.022338, 0.021156 and 0.025953
  (nine elements of 32,768 above 0.02), while every tensor's RMS is inside the
  established `0.003` limit (worst 0.002012). **No per-element adapter-weight
  parity is claimed.** Every assertion the documented Mixed acceptance contract
  does make — loss trajectory, gradient norm, adapter RMS, AdamW moments, frozen
  base parameters, rerun determinism, exact checkpoint resume, the 37 gate rows
  and the four graph-path tests — passes.

## Measured performance (per-step, one device)

| Change | Isolated kernel | Complete step |
| --- | ---: | ---: |
| Two-pass MHA backward | 9.7965× (435.28 → 44.43 ms) | 3.2492× (560.78 → 171.10 ms) |
| GQA grouped key pass, 8q/2kv | — | 19.76× |
| Row-packed cached forward | 16.42 → 4.38 ms/step | 1.2239× |
| Four-channel key pass, head 128 | — | 1.5297× (43.53 → 28.46 ms) |
| Mixed WMMA, LoRA | — | 1.2385× (28.481 → 23.001 ms) |

Each row is a paired ABBA measurement inside one driver session against the
route it replaced. **These are independent results, not multiplicative factors.**
The isolated kernel figures and the complete-step figures answer different
questions and are never combined.

## Scope and limitations

- **One device.** Every measurement was taken on the integrated AMD RDNA3
  adapter identified by PCI `1002:15bf` (`AMD Radeon Graphics`), driver
  `32.0.32015.14001`, Windows 11 x64. No discrete RDNA3, RDNA3.5, RDNA4, NVIDIA
  or Intel device was measured, and **no universal AMD, NVIDIA, Intel or
  cross-architecture qualification is implied**.
- **GQA eligibility is narrow.** Only the four recorded head-count groupings at
  head dimension 128 with `q_seq = kv_seq = 64` and causal masking are promoted.
  Other head dimensions keep the original key shader; generalizing to
  `head_dim % 4 == 0` would need its own qualification.
- **Channel4 is head-dimension-128 only.** Head dimensions 16/32/64 and
  non-square or longer-than-64 attention keep their previous routes.
- **Row-packed forward requires square sequences ≤ 64.** Non-square,
  sliding-window, sparse, paged, compressed-key, ALiBi, relative-bias, softcap,
  sink and dropout geometries keep the generic kernel.
- **Mixed keeps its full-parameter caveat** described above, and WMMA numbers are
  per-step milliseconds on one machine: no throughput, power or multi-GPU claim.
- Results, including items that could not be exercised and are reported as
  **BLOCKED** rather than passed, are in `CLEAN_MACHINE_TEST.md`.

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
