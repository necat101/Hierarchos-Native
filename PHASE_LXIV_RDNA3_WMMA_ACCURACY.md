# Phase LXIV — Full-FP16 split-reduction accuracy on the RDNA3 WMMA GEMM

This record changes one thing in the accuracy arm and states exactly what it does
and does not establish. The Phase LXII Candidate A staging optimisation is
preserved byte for byte; the Full FP16 arm's reduction order is changed so that
the same declared precision mode meets the numerical budget it was always
measured against.

Three implementation references are kept distinct throughout:

| Reference | Meaning | Revision |
| --- | --- | --- |
| `MAIN` | original remote `main` implementation | `81ac82fa972d84025129e7c3f0e58fdece7df786` |
| `A` | Phase LXII Candidate A (packed 32-bit operand staging) | `7c49558b8155a351b30893a6d8f318d4e063d1bd` |
| `FINAL` | this phase: `A` plus the eight-bank accuracy change | branch head after this commit |

`MAIN` and `A` are **identical** for every numerical result in this document:
the Phase LXII change moves load instructions and leaves the staged bytes and the
arithmetic untouched, which is why its Full FP16 receipt is byte-identical to
`MAIN`'s. The nine-gate failure set therefore belongs to both, and the comparison
below is stated against `MAIN`/`A` jointly.

## 1. The nine failing gates, recovered from the receipts

The Full FP16 qualification runs 37 gates: six fixtures × three gates, three
fixtures × three operand layouts × two gates, and one plain-FP16-storage control.

The nine failures on `MAIN`/`A` were, without reconstruction
(`hierarchos-vulkan/target/wmma-promotion-evidence/qual_baseline_full-fp16.json`):

| # | Gate | Shape | Layout (`trans_a`,`trans_b`) | Measured before | Unchanged limit | Now |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `128x128x96 wmma-full-fp16 vs matched oracle` | 128×128×96 | F,F | 0.0625 | 0.03125 (2 ULP) | **0.03125** |
| 2 | `16x1024x64 wmma-full-fp16 vs matched oracle` | 16×1024×64 | F,F | 0.0390625 | 0.03125 | **0.03125** |
| 3 | `37x8192x48 wmma-full-fp16 vs matched oracle` | 37×8192×48 | F,F | 0.046875 | 0.03125 | **0.0234375** |
| 4 | `128x128x96 trans_a=true trans_b=false vs oracle` | 128×128×96 | T,F | 0.0703125 | 0.03125 | **0.03125** |
| 5 | `128x128x96 trans_a=true trans_b=false vs FP32` | 128×128×96 | T,F | 0.0731049 | 0.05 | **0.0229750** |
| 6 | `128x128x96 trans_a=false trans_b=true vs oracle` | 128×128×96 | F,T | 0.0625 | 0.03125 | **0.03125** |
| 7 | `128x128x96 trans_a=false trans_b=true vs FP32` | 128×128×96 | F,T | 0.0637856 | 0.05 | **0.0215597** |
| 8 | `128x128x96 trans_a=true trans_b=true vs oracle` | 128×128×96 | T,T | 0.0625 | 0.03125 | **0.03125** |
| 9 | `128x128x96 trans_a=true trans_b=true vs FP32` | 128×128×96 | T,T | 0.0632477 | 0.05 | **0.0236740** |

All nine now pass, under the original thresholds, with no gate removed, no
expected-output digest changed and no precision label altered. Five of the nine
land **exactly** on the two-ULP bound (`0.03125` at this fixture magnitude): they
pass with zero margin, and that is stated rather than rounded up.

Classification, with the two references the phase requires:

* **Gates 5, 7, 9 are accuracy failures.** They compare the kernel with an FP64
  reference and exceed the unchanged absolute budget. This is FP16 accumulator
  drift, and it is the failure class a reduction-order change can fix.
* **Gates 1, 2, 3, 4, 6, 8 are reference-model failures.** They compare the
  kernel with the historical `tile16` oracle, which assumes exactly one rounding
  of the accumulator per 16-wide K slice. That model does not describe the
  hardware: on the same dumped operands, the GPU output matches a dot2-cadence
  model to a fraction of an ULP while sitting 3–4.5 ULP from `tile16`. The
  repository already records this and exposes `--full-oracle rdna3-dot2`; it
  leaves `tile16` as the default so the historical failure set stays reproducible.
  This phase keeps that default.

## 2. The change: eight accumulator banks

`wmma_gemm_f16_f16acc.comp` keeps **eight** FP16 accumulator banks per output
tile and assigns consecutive 16-wide K slices to them round-robin, then merges
them once per tile with a pairwise binary16 tree:

```
bank b = slices b, b+8, b+16, ...                     (FP16, RDNA3 dot2 cadence)
out    = ((b0+b1)+(b2+b3)) + ((b4+b5)+(b6+b7))        (FP16 additions)
```

Everything that defines the mode is unchanged: the operands are native FP16, the
instruction is the FP16-accumulating cooperative matrix multiply-add, every
accumulator and every merge is a binary16 value, and no product is formed in a
wider precision. The gate asserts that no `coopmat<float, …MatrixUseAccumulator>`
exists in the module.

Why it fixes the failures: a single FP16 accumulator that runs over the whole
reduction keeps rounding a large partial sum. Blocked summation bounds that error
— each bank carries 1/8 of the terms, so each running sum is about eight times
smaller and the roundings it absorbs are proportionally finer relative to the
final result. The pairwise tree matters too: a left-to-right merge of eight banks
rounds seven times with a growing partial, while the tree rounds seven times with
partials of at most half the final magnitude.

Cost: a fixed per-workgroup epilogue (four tiles × eight fragment stores and seven
FP16 additions each) that does not scale with the reduction length. This arm is
not a production performance path.

### 2.1 This corrects the previous phase's conclusion

The two-bank implementation on the branch before this commit
(`7c49558` … `fe4c1a8`) left the same four `128×128×96` matched-oracle gates
failing at `0.046875` and this document previously argued they were unreachable
without an FP32 accumulator or a widened tolerance. That argument was an
inference from a two-bank experiment, not a proof. Measured: the eight-bank
reduction reaches `0.03125` on all four while remaining a genuinely
FP16-accumulating kernel. The earlier text is superseded by §1 and §3.

## 3. Result on the RDNA3 adapter

Adapter `AMD Radeon Graphics` `1002:15bf`, driver `26.10.07.06 (LLPC)`,
`VK_KHR_cooperative_matrix` rev 2, 16×16×16 subgroup-scope f16×f16→f32 and
f16×f16→f16.

| Run | `MAIN` | `A` | `FINAL` |
| --- | ---: | ---: | ---: |
| Mixed, 37 gates | 0 failed | 0 failed | **0 failed** |
| Full FP16, `tile16` (historical, unchanged default) | 9 failed | 9 failed | **0 failed** |
| Full FP16, `split8` (contract-matched) | — | — | **0 failed** |
| Worst `vs FP32` `max_abs` (limit 0.05) | 0.0731049 | 0.0731049 | **0.0260372** |

The machine-readable receipt, including every `max_abs`, `rms` and limit, is
`hierarchos-vulkan/validation/fixtures/wmma_lxiv_accuracy_receipt.json` (schema
2); the full 37-row comparison is in `PHASE_LXIV_GATE_TABLE.md`.

`Fp16Split8` was added to `OracleAccumulation` and `--full-oracle split8` to
`wmma_qualify`, so the kernel can be graded against the order it actually issues.
Nothing was removed: `tile16` is still the default and `rdna3-dot2` and
`split2` still exist.

## 4. Performance

The cooperative kernel that carries the measured speedup
(`wmma_gemm_f16_f32acc.spv`, SHA-256
`e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65`) is
byte-identical to `A`. Two paired ABBA sweeps were run fresh in this phase with a
binary built from the final tree, 8 pairs, strictly serial, alternating order,
4 rounds × 20 timed iterations per arm per round, raw per-iteration samples
retained under `hierarchos-vulkan/target/wmma-lxiv/abba_main_vs_final_banks8/`
and `…/abba_a_vs_final_banks8/`. Ratio is `reference_ms / final_ms`, so above 1.0
is faster.

Executables: `MAIN`
`f41caea0c0a97fa017063de17e2f902722bcec6fa4c758362af2f504aa7538e0`, `A`
`28bb1a134910f853f88a8170727b3a890149c309efd6e2e10a9198e75909d558`, `FINAL`
`7bbda004338ba07f1734be3c4a101a167fc1a7869caedd9d5c112a950d4144e6`.

`FINAL` vs `MAIN` (`wmma-mixed-f32acc` arm):

| Shape | MAIN ms | FINAL ms | ratio | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `2048x768x2048` | 5.6549 | 3.4893 | **1.5979** | [1.548, 1.647] |
| `512x2048x768` | 1.6017 | 0.9513 | **1.6772** | [1.602, 1.753] |
| `512x768x768` | 0.5564 | 0.3830 | **1.4541** | [1.436, 1.472] |
| `128x8192x768` | 1.9300 | 1.4127 | **1.3698** | [1.358, 1.381] |
| `64x64x64` (below floor, kernel-level) | 0.1272 | 0.1194 | 1.0668 | [1.026, 1.108] |
| `16x128x32` (below floor, kernel-level) | 0.1072 | 0.1034 | 1.0234 | [0.997, 1.050] |

`FINAL` vs `A`:

| Shape | A ms | FINAL ms | ratio | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `2048x768x2048` | 3.5610 | 3.5253 | 0.9997 | [0.981, 1.019] |
| `512x2048x768` | 0.9628 | 0.9647 | 1.0062 | [0.985, 1.028] |
| `512x768x768` | 0.3854 | 0.3864 | 1.0103 | [0.984, 1.037] |
| `128x8192x768` | 1.4418 | 1.4227 | 1.0114 | [1.004, 1.018] |
| `64x64x64` | 0.1186 | 0.1159 | 1.0202 | [0.997, 1.044] |
| `16x128x32` | 0.1041 | 0.1047 | 0.9980 | [0.963, 1.034] |

No protected shape has a `FINAL` median slower than either reference. Against
`MAIN` every production shape clears a 1.35 lower bound.

**One interval misses the phase's 0.99 lower-bound rule and is not hidden:**
`512x2048x768` against `A` measured `[0.985, 1.028]` with a median ratio of
`1.0062`. The same sweep's FP32 negative control on that shape is
`0.9867 [0.960, 1.013]`, so the harness floor at this shape is wider than the
0.005 by which the bound was missed; the two production arms embed byte-identical
SPIR-V and cannot differ by construction. It is reported as a resolution limit,
not as a regression.

Controls: the FP32 negative control is `1.0092 [0.986, 1.033]`,
`0.9668 [0.915, 1.018]`, `0.9895 [0.962, 1.017]`, `1.0037 [0.955, 1.052]`,
`1.0040 [0.970, 1.038]`, `1.0196 [0.986, 1.053]` (`MAIN` vs `FINAL`), and the
plain-FP16 control is `1.0092`, `1.0036`, `0.9917`, `1.0156`, `1.0249`, `1.0283`.
All but one contain 1.0; the exception is the plain-FP16 arm at `64x64x64`
(`1.0283 [1.003, 1.054]`), a 2.8 % shift on a kernel that is not part of this
change at a below-floor shape, and it is recorded as a control deviation.

The changed shader is the Full FP16 arm, which is not a production performance
path. Its epilogue cost is fixed per workgroup and does not scale with the
reduction length. It was not re-timed as a throughput claim and none is made.

## 5. What this phase does not establish

* **No additional WMMA optimization.** Candidate B (64×32 tile) was **not
  attempted** in this phase. No profiling result identified a specific new
  staging or occupancy opportunity, so nothing is claimed and nothing was
  rejected on measurements. Candidate A remains the performance record.
* **No training-step throughput claim.** The Phase LXII limitation stands: at 64
  rows the process-level step is dominated by host and graph work, and the
  fraction of a training step spent in cooperative dispatches was not
  re-established in this phase. The end-to-end figure is **unresolved**.
* **No AMD ISA confirmation.** No gfx110x disassembler is available on this host.
* **Hosted CI recompiles nothing.** It verifies the recorded reproduction and the
  SPIR-V structure; the compile-and-byte-identity check is `--reproduce` on a host
  with the pinned shaderc.
* **Zero margin on five gates.** They pass at exactly two ULP; a future change
  that moves any of them up by one ULP fails.

## 6. Reproducing this

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
./target/release/wmma_qualify.exe --wmma-mode mixed            --json   # 0 failed
./target/release/wmma_qualify.exe --wmma-mode full-fp16        --json   # 0 failed
./target/release/wmma_qualify.exe --wmma-mode full-fp16 --full-oracle split8 --json  # 0 failed

python validation/check_current_candidate.py --reproduce
python -m unittest discover -s validation -p "test_current_candidate.py" -v
```
