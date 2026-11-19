# Phase LXII — packed 32-bit operand staging for the RDNA3 WMMA GEMM (isolated promotion)

This record isolates one measured optimisation of the RDNA3 cooperative-matrix
GEMM staging path onto an exact `main` base, re-verifies it there, and states
what is and is not claimed. It imports nothing from the earlier feature branch
except the two shader sources, their compiled SPIR-V, the paired-benchmark
harness, and two boundary shapes in an existing test.

## 1. Starting state

| Item | Value |
| --- | --- |
| Base | `81ac82fa972d84025129e7c3f0e58fdece7df786` (freshly fetched `origin/main`) |
| Branch | `codex/rdna3-wmma-vector-load-20261118` |
| Adapter | `AMD Radeon Graphics`, driver `26.10.07.06 (LLPC)`, RDNA3 |
| Cooperative config | 16x16x16 `SUBGROUP` f16 x f16 -> f32 and -> f16 |
| Profitability floor | `WMMA_MIN_PROFITABLE_MACS = 16_000_000` (unchanged) |
| Trainer hash | `055d82ad05ee5ca82ec63d0d4e116c7a18be6b9bc922b956c8f4ea71b1118931` (unchanged; never relinked) |

The shader sources and their SPIR-V are **byte-identical between `origin/main`
and the Phase LX commit that preceded the original candidate**, so this diff
applies to `main` with no dependency on earlier branch history.

## 2. What changed

| Artifact | Baseline SHA-256 | Candidate SHA-256 |
| --- | --- | --- |
| `shaders/wmma_gemm_f16_f32acc.comp` | `68b3f24fdb34bb6012894626426ee3e250c980cee32f3088ce32eb2193b12317` | `239b63144bdd7c7aef703d88622201bc9c6c4832000e7dc86cec3317034863fc` |
| `shaders/wmma_gemm_f16_f32acc.spv` | `c6fd0f349bf66c731a43673ae410a6f41fd5609bf9d0b029fc15940042d8fcf5` | `e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65` |
| `shaders/wmma_gemm_f16_f16acc.comp` | `513a8b2df1a03fe63cecfa1fb7e1faeae2f4d3b82b921cd276ca13c3040b7dfe` | `43a62481f1cf23d96505b39ec8226deb9f4a02c3f054d3c6a7646ba10649f05e` |
| `shaders/wmma_gemm_f16_f16acc.spv` | `90724b47132fcbfbde3d7365727c04297922b16e7387ac460a434de8828531e1` | `b040095e481f9de7625df7a2d5bc743027629468e2bb553b571ee0f1237ad73a` |

Both candidate modules reproduce **bit-for-bit** from their `.comp` sources:

```
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O
```

No Rust file, dispatch rule, threshold, fallback or tolerance in the GEMM path
changed. The two non-shader changes are the paired-benchmark harness
(`src/bin/wmma_bench.rs --dump-samples`, `validation/wmma_abba_run.sh`,
`validation/wmma_abba_analyze.py`) and two extra shapes in the existing
`wmma_mixed_matches_matched_oracle` boundary test. Neither is on the GPU maths
path.

### 2.1 The change

The kernel was staging-issue bound: per 16-wide K slice one lane issued 16
staging iterations, each with a scalar 16-bit load for A and for B, against only
four cooperative multiply-adds. Both operands are now viewed as packed
`f16vec2` words, and a lane fetches a pair of adjacent halves with one aligned
32-bit load. Pairing follows the operand's globally contiguous axis:

| Operand / layout | Contiguous axis | Pair axis |
| --- | --- | --- |
| A `a[m][k]` (row-major) | k | `col` (two adjacent K) |
| A `a[k][m]` (transposed) | m | `row` (two adjacent output rows) |
| B `b[k][n]` (row-major) | n | `col` (two adjacent output columns) |
| B `b[n][k]` (transposed) | k | `row` (two adjacent K) |

Per lane per K slice, staging loop trips fall 16 -> 8, global loads 32 -> 16, and
leading-dimension multiplies 32 -> 16. Workgroup 16-bit stores (32), cooperative
multiply-adds (4), tile layout, lane-to-tile mapping, fragment loads and
workgroup geometry are unchanged.

`spirv-val --target-env vulkan1.2` is clean for both modules, and each still
contains exactly four `OpCooperativeMatrixMulAddKHR`.

## 3. Alignment and bounds proof

**Base alignment.** Both operands are `readonly buffer { f16vec2 a[]; }`. The
buffer is allocated by `WmmaGemm::new_half_buffer` as `len.div_ceil(2)` **u32
words**, so its size is a multiple of 4 bytes and its Vulkan memory binding is
word-aligned; the descriptor covers the whole buffer at offset 0. A word load at
halfword index `off` therefore has byte address `2*off` relative to a 4-byte
aligned base, and is aligned iff `off` is even.

**Index parity.** `off` is even in all four cases whenever the operand's leading
dimension is even, because the pair axis contributes an even amount and the
other axis is scaled by the even stride:

| Case | `off` | Pair axis parity | Result |
| --- | --- | --- | --- |
| A row-major | `gm*lda + gk` | `gk = k0 + col`, `col` even, `k0` multiple of 16 | even |
| A transposed | `gk*lda + gm` | `gm = m0 + tile*16 + row`, `row` even, `m0` multiple of 32 | even |
| B row-major | `gk*ldb + gn` | `gn = n0 + tile*16 + col`, `col` even, `n0` multiple of 32 | even |
| B transposed | `gn*ldb + gk` | `gk = k0 + row`, `row` even, `k0` multiple of 16 | even |

The shader gates the packed path on exactly this precondition (`wide_a =
(lda & 1) == 0`, `wide_b = (ldb & 1) == 0`), and `lda`/`ldb` are derived by
`record_gemm` from the shape alone (`lda` is `reduction` for row-major A and
`rows` for transposed A; `ldb` is `columns` for row-major B and `reduction` for
transposed B). There is no API path that supplies an arbitrary padded stride, so
an odd halfword base offset cannot be supplied.

**No straddling read.** The packed path is taken only when *both* halves are in
range (`gk + 1 < k`, `gm + 1 < m`, `gn + 1 < n`) and the guard compares against
the same dimension that bounds the pair axis, so a pair never crosses a row
boundary and never leaves the logical matrix. Every other case - odd leading
dimension, odd tail on the reduction or output axis, partial edge tile, a pair
with only one half in range - takes the fallback, which performs two
independently masked accesses. `spirv-dis` shows those fallbacks as scalar
`OpLoad %half` (16-bit), not as `%v2half`, so the final half of an odd-length
buffer is never read as part of a 4-byte word.

A consequence worth stating: whenever the packed path is reachable the leading
dimension is even, so the buffer holds an even number of halves and its
descriptor range is an exact multiple of 4 bytes. The packed path therefore can
never touch a word that extends past the descriptor range, and the odd-length
case is exclusively served by 16-bit loads.

**Bit patterns.** A packed pair is a plain load of the same two halves that the
scalar path loaded; it is never a re-rounding, so staged bytes are identical.

## 4. Boundaries exercised

`wmma_mixed_matches_matched_oracle` runs each shape in all four
`trans_a`/`trans_b` layouts against a matched oracle and against an FP32
reference. Its shape list now includes:

| Shape | What it forces |
| --- | --- |
| `33x17x19` | odd rows, columns and reduction: every leading dimension is odd, so all four cases take the masked 16-bit fallback; odd K tail; partial tiles on both axes |
| `64x48x40` | packed path on both operands; partial `n` tile where one half of a pair is in range and the other is not |
| `128x128x96` | several K slices, fully aligned |
| `64x48x19` | A transposed keeps an even leading dimension (64) while K is odd, so the final K slice packs words whose second half is out of range |
| `19x48x64` | A row-major keeps an even leading dimension (64) while only 19 output rows exist, so the last row tile packs at the matrix edge |

Out-of-range elements remain exactly zero because the fallback initialises both
halves to zero and only overwrites a half whose own guard passes; the padded
fragment therefore contributes nothing.

## 5. Correctness

Both arms were built in this session from one source tree differing only in the
embedded SPIR-V.

| Gate | Baseline | Candidate |
| --- | --- | --- |
| `wmma_qualify --wmma-mode mixed` | 37 gates, 0 failed, exit 0 | 37 gates, 0 failed, exit 0 |
| Mixed receipt | - | **byte-identical to baseline** |
| `wmma_qualify --wmma-mode full-fp16` | 37 gates, 9 failed (documented), exit 1 | 37 gates, 9 failed (documented), exit 1 |
| Full-FP16 receipt | - | **byte-identical to baseline** |
| `cargo test --release --lib wmma` | - | 20 passed, 0 failed |

Byte-identical receipts mean every drift value, relative error and gate verdict
matches to full printed precision, and that the failing set is the same nine
documented FP16-accumulation gates, not a different nine.

The test run was executed with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
forced through the loader. It reported **no descriptor-range, alignment or
out-of-bounds VUID**. It did emit module-level
`VUID-VkShaderModuleCreateInfo-pCode-08737/08742` and
`VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770` messages, which concern
shader-module/cooperative-matrix declaration rather than memory access; both
arms use modules of that family, and `spirv-val` is clean on them.

## 6. Performance

Design: the paired ABBA harness, unmodified. 8 pairs, strictly serial,
alternating baseline-first / candidate-first. Each run is 4 rounds x 20 timed
iterations per arm after warm-up; the 80 raw samples per pair and shape are
pooled to one median and the ratio is `baseline_ms / candidate_ms`, so >1.0
means the candidate is faster. Bumped shape list is the six required shapes.

| Arm | Executable SHA-256 |
| --- | --- |
| baseline | `f41caea0c0a97fa017063de17e2f902722bcec6fa4c758362af2f504aa7538e0` |
| candidate | `28bb1a134910f853f88a8170727b3a890149c309efd6e2e10a9198e75909d558` |

`wmma-mixed-f32acc` arm, candidate vs baseline:

| Shape | base ms | cand ms | ratio | 95% CI | geomean |
| --- | --- | --- | --- | --- | --- |
| `2048x768x2048` | 6.2334 | 4.1134 | 1.5270 | `[1.506, 1.548]` | 1.5269 |
| `128x8192x768` | 2.0891 | 1.5048 | 1.3919 | `[1.341, 1.443]` | 1.3907 |
| `512x2048x768` | 2.0353 | 1.1231 | 1.6459 | `[1.291, 2.001]` | 1.5949 |
| `512x768x768` | 0.8283 | 0.5225 | 1.8160 | `[1.345, 2.287]` | 1.7367 |
| `64x64x64` (control, below floor) | 0.1310 | 0.1133 | 1.1549 | `[1.140, 1.170]` | 1.1547 |
| `16x128x32` (control, below floor) | 0.1061 | 0.1008 | 1.0600 | `[1.008, 1.112]` | 1.0584 |

Every protected shape is faster, every point estimate is above 1.0, every 95%
interval lower bound is above 1.0 (the gate asks for at least 0.99), and all
four production-relevant shapes clear a lower bound above 1.05.

`128x8192x768` is the shape both rejected Phase LX candidates regressed to
0.8935 and 0.8340; it is now 1.3919 `[1.341, 1.443]`. That is consistent with
the Phase LX diagnosis: the rejected candidates changed the coalesced address
order, while this change leaves the address order per warp exactly as it was and
halves the instruction count that issues it.

### 6.1 Controls

- FP32 negative control (`fp32-baseline`, byte-identical kernel in both
  executables): `1.0034 [0.979, 1.028]`, `1.0102 [0.978, 1.042]`,
  `0.9984 [0.983, 1.014]`, `1.0032 [0.993, 1.014]`, `0.9391 [0.763, 1.115]`,
  `0.9979 [0.967, 1.029]`. Every interval contains 1.0; the widest is
  `512x768x768`, the intrinsically noisy shape Phase LX also flagged.
- Plain-FP16 control (`plain-f16-storage-f32acc`, a different unmodified
  shader): `1.0041`, `1.0132`, `1.0053`, `1.0007`, `1.0493`, `1.0125`, all with
  intervals containing 1.0. The change is confined to the cooperative kernel.

### 6.2 Independent replication

A second complete 8-pair x 4-round x 20-iteration sweep was run with the same two
executables into `abba_promotion_replicate/`. `wmma-mixed-f32acc` arm:

| Shape | base ms | cand ms | ratio | 95% CI | geomean |
| --- | --- | --- | --- | --- | --- |
| `2048x768x2048` | 6.2059 | 4.1103 | 1.5113 | `[1.500, 1.522]` | 1.5113 |
| `128x8192x768` | 2.1970 | 1.5111 | 1.4676 | `[1.327, 1.608]` | 1.4593 |
| `512x2048x768` | 2.1990 | 1.5226 | 1.5088 | `[1.234, 1.784]` | 1.4783 |
| `512x768x768` | 0.7342 | 0.5372 | 1.4652 | `[1.045, 1.886]` | 1.4002 |
| `64x64x64` (control) | 0.1313 | 0.1133 | 1.1485 | `[1.124, 1.173]` | 1.1482 |
| `16x128x32` (control) | 0.1038 | 0.0963 | 1.0634 | `[1.020, 1.107]` | 1.0623 |

The replication reproduces every verdict: all six shapes faster, every interval
lower bound above 1.0, no shape below 1.0 in either sweep.

One control caveat is reported rather than smoothed over. The FP32 negative
control in this second sweep measured `1.0223 [1.003, 1.041]` on `16x128x32`,
i.e. it marginally excluded 1.0, where the first sweep gave
`1.0102 [0.978, 1.042]`. Both arms embed byte-identical FP32 kernels, so this is
about 2% of residual bias or noise at the harness timer floor (~0.08 ms), not a
effect of the change. It bounds how much of the small-shape control gains can be
trusted but is an order of magnitude below the 35%-53% effects on the four
production shapes, and the FP32 arm is consistent with noise on every other
shape in both sweeps (`1.0034`, `1.0093`, `0.9984`, `1.0059`, `1.0032`,
`0.9981`, `0.9391 [0.763, 1.115]`, `0.8829 [0.682, 1.084]`, `0.9979`, `1.0132`).

Raw per-iteration samples: `hierarchos-vulkan/target/wmma-promotion-evidence/`
`abba_promotion/` and `abba_promotion_replicate/` (`index.tsv` plus
`p*_{base,cand}_samples.jsonl`). Those directories are untracked build output.

## 7. What is not claimed

- **No training-step speed claim.** The kernel is a GEMM-call latency result
  (`wmma_bench`), never training throughput. The Phase LXI record established
  real-model reachability (451 cooperative dispatches per step on a genuine
  SmolLM2-135M checkpoint) and bit-identical loss, but at 64 rows the
  process-level step time is dominated by host and graph work and its run-to-run
  variance swamps the kernel effect.
- **No AMD ISA confirmation.** This host has no gfx110x disassembler; the claim
  is attributed to the SPIR-V-level load/loop/address reductions and to the
  measurement, not to a disassembled instruction sequence.
- Candidate B (64x32 tile) was not attempted: no defect in this change gives a
  reason to reopen tile design.

## 8. Reproducing this

```bash
# candidate arm: candidate sources in place
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O \
  -o shaders/wmma_gemm_f16_f32acc.spv shaders/wmma_gemm_f16_f32acc.comp
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O \
  -o shaders/wmma_gemm_f16_f16acc.spv shaders/wmma_gemm_f16_f16acc.comp
cargo build --release --bin wmma_bench --bin wmma_qualify
cp target/release/wmma_bench.exe    $E/bin/wmma_bench_candidate.exe
cp target/release/wmma_qualify.exe  $E/bin/wmma_qualify_candidate.exe

# baseline arm: restore the two .spv from main, rebuild
git checkout origin/main -- shaders/wmma_gemm_f16_f32acc.spv shaders/wmma_gemm_f16_f16acc.spv
cargo build --release --bin wmma_bench --bin wmma_qualify
cp target/release/wmma_bench.exe    $E/bin/wmma_bench_baseline.exe
cp target/release/wmma_qualify.exe  $E/bin/wmma_qualify_baseline.exe

# correctness
./$E/bin/wmma_qualify_baseline.exe  --wmma-mode mixed     --json > qual_base_mixed.json
./$E/bin/wmma_qualify_candidate.exe --wmma-mode mixed     --json > qual_cand_mixed.json
cmp qual_base_mixed.json qual_cand_mixed.json

# paired ABBA
WMMA_EVIDENCE_DIR=$E BASE=$E/bin/wmma_bench_baseline.exe \
  CAND=$E/bin/wmma_bench_candidate.exe OUT_DIR=$E/abba PAIRS=8 ROUNDS=4 ITERS=20 \
  bash validation/wmma_abba_run.sh
python validation/wmma_abba_analyze.py $E/abba
```
