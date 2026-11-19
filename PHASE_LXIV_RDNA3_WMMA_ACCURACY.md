# Phase LXIV — Full-FP16 split-reduction accuracy on the RDNA3 WMMA GEMM

This record changes one thing and states exactly what it does and does not
establish. The Phase LXII Candidate A staging optimisation is preserved byte for
byte; the Full FP16 arm's reduction order is changed so that the same declared
precision mode meets the numerical budget it was always measured against.

Three implementation references are kept distinct throughout:

| Reference | Meaning | Revision |
| --- | --- | --- |
| `MAIN` | original remote `main` implementation | `81ac82fa972d84025129e7c3f0e58fdece7df786` |
| `A` | Phase LXII Candidate A (packed 32-bit operand staging) | `7c49558b8155a351b30893a6d8f318d4e063d1bd` |
| `FINAL` | this phase: `A` plus the split-reduction accuracy change | branch head after this commit |

`MAIN` and `A` are **identical** for every numerical result in this document:
the Phase LXII change moves load instructions and leaves the staged bytes and the
arithmetic untouched, which is why its Full FP16 receipt is byte-identical to
`MAIN`'s. The nine-gate failure set therefore belongs to both, and the comparison
below is stated against `MAIN`/`A` jointly.

## 1. The nine failing gates, recovered from the receipts

The Full FP16 qualification runs 37 gates: six fixtures × three gates, three
fixtures × three operand layouts × two gates, and one plain-FP16-storage control.

The nine failures on `MAIN`/`A` were, without reconstruction:

| # | Gate | Shape | Layout (`trans_a`,`trans_b`) | Mode | Measured `max_abs` | Limit | Root cause |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | `128x128x96 wmma-full-fp16 vs matched oracle` | 128×128×96 | F,F | full-fp16 (f16 operands, f16 accum) | 0.0625 | 0.03125 (2 ULP) | single accumulator, 4 ULP from the tile16 model |
| 2 | `16x1024x64 wmma-full-fp16 vs matched oracle` | 16×1024×64 | F,F | full-fp16 | 0.0390625 | 0.03125 | single accumulator, 2.5 ULP |
| 3 | `37x8192x48 wmma-full-fp16 vs matched oracle` | 37×8192×48 | F,F | full-fp16 | 0.046875 | 0.03125 | single accumulator, 3 ULP |
| 4 | `128x128x96 trans_a=true trans_b=false vs oracle` | 128×128×96 | T,F | full-fp16 | 0.0703125 | 0.03125 | single accumulator, 4.5 ULP |
| 5 | `128x128x96 trans_a=true trans_b=false vs FP32` | 128×128×96 | T,F | full-fp16 | 0.0731049 | 0.05 | **FP16 accumulator drift vs the mathematical reference** |
| 6 | `128x128x96 trans_a=false trans_b=true vs oracle` | 128×128×96 | F,T | full-fp16 | 0.0625 | 0.03125 | single accumulator, 4 ULP |
| 7 | `128x128x96 trans_a=false trans_b=true vs FP32` | 128×128×96 | F,T | full-fp16 | 0.0637856 | 0.05 | **FP16 accumulator drift vs the mathematical reference** |
| 8 | `128x128x96 trans_a=true trans_b=true vs oracle` | 128×128×96 | T,T | full-fp16 | 0.0625 | 0.03125 | single accumulator, 4 ULP |
| 9 | `128x128x96 trans_a=true trans_b=true vs FP32` | 128×128×96 | T,T | full-fp16 | 0.0632477 | 0.05 | **FP16 accumulator drift vs the mathematical reference** |

Classification, with the two references the phase requires:

* **Gates 5, 7, 9 are accuracy failures.** They compare the kernel with an FP64
  reference and exceed the unchanged absolute budget. This is FP16 accumulator
  drift, and it is the failure class a numerical change can fix.
* **Gates 1, 2, 3, 4, 6, 8 are reference-model failures.** They compare the
  kernel with the historical `tile16` oracle, which assumes exactly one rounding
  of the accumulator per 16-wide K slice. That model does not describe the
  hardware: on the same dumped operands, the observed GPU output equals a
  dot2-cadence model (`round_to_f16` after every two products) to within a
  fraction of an ULP, while it sits 3–4.5 ULP away from `tile16`. The repository
  already recorded this (`RDNA3_PRECISION_COMPLETION.md`) and already exposes
  `--full-oracle rdna3-dot2`; it left `tile16` as the default so the historical
  failure set stays reproducible. This phase keeps that default.

No gate, tolerance, expected-output digest or precision label is changed here.

## 2. The change: two accumulator banks

`wmma_gemm_f16_f16acc.comp` now keeps **two** FP16 accumulator banks and assigns
alternating 16-wide K slices to them, merging them once per output tile:

```
acc0 = slices 0, 2, 4, ...      (FP16 accumulator, hardware dot2 cadence)
acc1 = slices 1, 3, 5, ...      (FP16 accumulator, hardware dot2 cadence)
out  = float16_t(acc0 + acc1)   (FP16 addition)
```

Everything that defines the mode is unchanged: the operands are still native
FP16, the instruction is still the FP16-accumulating cooperative matrix
multiply-add, every accumulator and every merge is a binary16 value, and no
product is formed in a wider precision. What changes is the **reduction order**,
which is the numerically meaningful part: each bank carries about half the terms,
so each running sum is smaller and the roundings it absorbs are finer relative to
the final result. The merge is staged through workgroup memory because GLSL
cooperative-matrix values cannot be added directly.

This is not an FP32 accumulation presented as an FP16 one. The gate asserts that
no `coopmat<float, …MatrixUseAccumulator>` exists in the module at all.

Cost: a fixed per-workgroup epilogue (four tiles × two fragment stores and one
merge each) that does not scale with the reduction length.

### 2.1 Matched-oracle surface

`OracleAccumulation` gains `Fp16Split2`, which models exactly the order above
(two banks, dot2 cadence, one FP16 merge), and `wmma_qualify` gains
`--full-oracle split2`. Nothing is removed: `tile16` is still the default and
`rdna3-dot2` still exists, so every historical comparison remains reproducible.

## 3. Result on the RDNA3 adapter

Adapter `AMD Radeon Graphics` `1002:15bf`, driver `26.10.07.06 (LLPC)`,
`VK_KHR_cooperative_matrix` rev 2, 16×16×16 subgroup-scope f16×f16→f32 and
f16×f16→f16.

| Run | `MAIN` | `A` | `FINAL` |
| --- | ---: | ---: | ---: |
| Mixed, 37 gates | 0 failed | 0 failed | **0 failed** |
| Mixed receipt | — | — | **byte-identical to `A`** |
| Full FP16, `tile16` (historical default) | 9 failed | 9 failed | **4 failed** |
| Full FP16, `rdna3-dot2` | all matched gates pass | identical | — |
| Full FP16, `split2` (contract-matched) | — | — | **0 failed** |
| Worst `vs FP32` `max_abs` | 0.0731049 | 0.0731049 | **0.0419369** |

Five gates are fixed and none regressed. The four that remain are the
`tile16`-model comparisons on `128×128×96`; they improved (0.0625 → 0.046875,
0.0703125 → 0.046875) and are the residual distance between two different but
both-approximate reductions, not an FP32-budget failure. Under the
contract-matched model the same kernel is 37/37, which is what
"implementation correctness" is supposed to measure.

The full 37-row comparison is in `PHASE_LXIV_GATE_TABLE.md`; the machine-readable
receipt, including every `max_abs`, `rms` and limit, is
`hierarchos-vulkan/validation/fixtures/wmma_lxiv_accuracy_receipt.json`.

### 3.1 Why the four residuals are not repaired by a looser reference

Because the default `tile16` model is 3 ULP from the hardware, a matched-oracle
gate against it cannot be satisfied by *any* genuinely FP16-accumulating kernel
at this magnitude: the kernel's own distance from FP32 (≥0.038 here) and the
model's distance from FP32 (0.020) add in quadrature to about the 0.03125 bound.
Reaching zero failures under that model would require either an FP32 accumulator
(the Mixed mode, which is a different contract and would destroy the meaning of
"Full FP16") or a widened tolerance. Neither is done.

## 4. What this phase does not establish

* **No new performance measurement.** The cooperative kernel that carries the
  measured speedup (`wmma_gemm_f16_f32acc.spv`, SHA-256
  `e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65`) is
  byte-identical to `A`, and the Mixed qualification receipt is byte-identical,
  so `A`'s recorded ABBA result applies unchanged. That is a proof of identity,
  not a new timing run.
* **No training-step throughput claim.** The Phase LXII limitation stands: at
  64 rows the process-level step is dominated by host and graph work.
* **No AMD ISA confirmation.** No gfx110x disassembler is available on this host.
* **Candidate B (64×32 tile) was not attempted** in this phase. The accuracy work
  changed the same shader family, and no profiling result in this phase gave a
  reason to reopen tile design.
* **Hosted CI recompiles nothing.** It verifies the recorded reproduction and the
  SPIR-V structure; the compile-and-byte-identity check is `--reproduce` on a host
  with the pinned shaderc.

## 5. Reproducing this

```bash
cd hierarchos-vulkan
export PATH="/path/to/VulkanSDK/1.4.321.1/Bin:$PATH"

# Both modules must reproduce byte-for-byte.
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O \
  -o shaders/wmma_gemm_f16_f32acc.spv shaders/wmma_gemm_f16_f32acc.comp
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O \
  -o shaders/wmma_gemm_f16_f16acc.spv shaders/wmma_gemm_f16_f16acc.comp
spirv-val --target-env vulkan1.2 shaders/wmma_gemm_f16_f32acc.spv
spirv-val --target-env vulkan1.2 shaders/wmma_gemm_f16_f16acc.spv

cargo build --release --bin wmma_qualify
./target/release/wmma_qualify.exe --wmma-mode mixed            --json
./target/release/wmma_qualify.exe --wmma-mode full-fp16        --json   # 4 failed
./target/release/wmma_qualify.exe --wmma-mode full-fp16 --full-oracle split2 --json  # 0 failed

python validation/check_current_candidate.py --reproduce
python -m unittest discover -s validation -p "test_current_candidate.py" -v
```
