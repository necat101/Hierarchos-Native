# Phase LXV — Radeon GPU Profiler hardware evidence for the RDNA3 WMMA arms

This record replaces the earlier "no gfx110x disassembler here" conclusion. RGP is
installed on this host, it captures this RDNA3 APU headlessly, and the compiled AMD
ISA of every WMMA kernel below was read out of those captures.

## 1. Tool, adapter and capture path

| Item | Value |
| --- | --- |
| RGP | Radeon GPU Profiler **2.7.1.13** (RDTS `2026-10-02-1872`) |
| Path | `C:\Users\User\Downloads\rdts-dl\RadeonDeveloperToolSuite-2026-10-02-1872\RadeonGPUProfiler.exe` |
| Capture driver | Radeon Developer Panel CLI `1.0.0` (same suite) |
| GPU | AMD Radeon Graphics (Radeon 780M, gfx1103), PCI `1002:15bf` |
| Device driver | `32.0.31007.6002`, packaging `26.10.07.06-260512a-200783C-Asus` |
| OS | Windows 11 Enterprise 26100 |
| Capture method | `RadeonDeveloperPanelCLI.exe -m profiling --rgp-auto-capture=dispatch:1:N -p wmma_bench` |
| Analytical ISA read-out | `amd_comgr_3.dll` from the same RDTS, `AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE`, isa `amdgcn-amd-amdhsa--gfx1103` |

The capture is a **dispatch-range profile of a real run** of the unmodified benchmark
binary, not a synthetic/orphaned profile. RDP set the shader clock to peak for the
capture and restored it afterwards (`Successfully set clocks to peak` /
`Successfully restored clocks` in each capture log). No application code, shader or
dispatch was altered to obtain these profiles.

Reproduce: `lxv-rgp/capture.sh <name> <bench.exe> <mixed|full-fp16> [dispatches]`.
Raw profiles are kept outside the Git tree (they are 15–379 MB); their SHA-256 and
byte counts are recorded below and in `phase_lxv_receipt.json`.

## 2. Captures

| Capture | Arm | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `cap-main-mixed.rgp` | remote-main, mixed | 15,931,037 | `83174ae48fd8723e60cf9a74c0ee7f8d8e8543112b3769bf93e42a329d7c9d1c` |
| `cap-a-mixed.rgp` | Candidate A, mixed | 16,662,590 | `d48415cdbaa53acbd5fee11f66ba2b62ca43ea8b39def2857ee16824bed4961c` |
| `cap-mixed-s1.rgp` | Final, mixed | 49,352,244 | `84dd97846d4ac408…` (full hash in the receipt) |
| `cap-main-full.rgp` | remote-main, full FP16 | 16,669,405 | `91f92b7341412e49e32dbbd4f669d62b832ecfe860ccfbb9e68e301ad4daf7e1` |
| `cap-final-full.rgp` | Final eight-bank, full FP16 | 15,451,923 | `c983272f0cb19cd65e1fc2936a6680f2c5853d9dfab58c91fd326b79ab560793` |
| `cap-final-full-trace.rgp` | Final eight-bank, full FP16, **SQTT instruction tracing** | 379,492,979 | (hash in the receipt) |

Instruction tracing, which previously failed with `RGP trace failed with result: -2`,
completes on this host with `--rgp-instruction-tracing --rgp-sqtt-buffer-size=maximum`
(`[Status] Done`, `Profiling (RGP) trace completed successfully`). The resulting RDF
holds the raw SQTT stream; per-instruction **hit counts** are a GUI-side reduction of
that stream and are not decoded here, so no dynamic hit count is claimed from it.

## 3. Compiled AMD ISA, straight from the captures

Every code object below was extracted byte-for-byte from its `.rgp` and disassembled
by AMD's own COMGR target for gfx1103. Listings are committed under
`validation/phase_lxv_evidence/isa/`; the ELF objects are committed beside them.

| Arm | Code object SHA-256 (prefix) | `.text` | Instructions | WMMA mnemonics | Static WMMA |
| --- | --- | ---: | ---: | --- | ---: |
| remote-main mixed | `33d8a722afee` | 3,968 B | 503 | `v_wmma_f32_16x16x16_f16` | 4 |
| Candidate A mixed | `23a95fc3a7de` | 3,968 B | 752 | `v_wmma_f32_16x16x16_f16` | 4 |
| **Final mixed** | `23a95fc3a7de` | 3,968 B | 752 | `v_wmma_f32_16x16x16_f16` | 4 |
| remote-main full FP16 | `357332aba4cb` | 3,928 B | 491 | `v_wmma_f16_16x16x16_f16` | 4 |
| **Final eight-bank full FP16** | `d6adfcec9af8` | 42,496 B | 5,082 | `v_wmma_f16_16x16x16_f16` | **32** |

`Final mixed` and `Candidate A mixed` are the **same code object** (`23a95fc3a7de…`,
byte-identical, identical pipeline hash), captured from two different executables.
That is stronger than the "SPIR-V is byte-identical" claim in the earlier PR text: the
bytes the GPU actually executes are identical, not merely the bytes fed to the driver.

### 3.1 Decoded instruction examples

```
Candidate A / Final mixed          0x30C: CC404018 1C624D36  v_wmma_f32_16x16x16_f16 v[24:31], v[54:61], v[38:45], v[24:31]
Final eight-bank full FP16        0x1588: CC424061 1D8601A8  v_wmma_f16_16x16x16_f16 v[97:104], v[168:175], v[0:7], v[97:104]
Final eight-bank full FP16        0x1754: CC426061 1D8713A8  v_wmma_f16_16x16x16_f16 ... op_sel:[0,0,1]
```

AMD's documented WMMA lowering is the `v_wmma` instruction; the FP16-accumulating
form is `v_wmma_f16_16x16x16_f16` and the FP32-accumulating form is
`v_wmma_f32_16x16x16_f16`. The internal micro-operations of WMMA are **not** exposed
as separately disassembled shader instructions, and nothing below claims them.

## 4. Static versus dynamic WMMA

Static count is read from the ISA; dynamic count is not inferred from it.

* Mixed module: 4 `OpCooperativeMatrixMulAddKHR` in SPIR-V → 4 `v_wmma` in ISA.
* Full FP16 module: 32 `OpCooperativeMatrixMulAddKHR` → **32** `v_wmma` in ISA
  (eight banks × four 16×16 output tiles).
* The eight-bank kernel routes each K slice to exactly one bank with a uniform
  selector, so **four** WMMA instructions execute per K slice, not 32. The 32
  static sites exist because the bank is selected by control flow; the kernel
  contains 395 branches (uniform, workgroup-wide) to realise that.
* The SQTT trace capture proves tracing works on this host, but decoding hit counts
  is a GUI reduction of the trace, so **no per-instruction dynamic hit count is
  claimed**. Dynamic execution is bounded by the dispatch accounting of the harness
  (cooperative dispatches, K slices per dispatch) and by measured per-dispatch time.

## 5. Pipeline resources as compiled

Read from each code object's AMDPAL compute metadata (`hardware_stages[".cs"]`):

| Resource | main mixed | Cand A / Final mixed | main full FP16 | **Final 8-bank full FP16** |
| --- | ---: | ---: | ---: | ---: |
| Wavefront size | 32 | 32 | 32 | 32 |
| VGPRs / wave | 73 | 71 | 57 | **256 (architectural maximum)** |
| SGPRs / wave | 31 | 36 | 31 | 37 |
| LDS bytes / workgroup | 3,072 | 3,072 | 2,560 | **4,608** |
| Scratch enabled | no | no | no | **yes** |
| Scratch bytes / wave | 0 | 0 | 0 | **768** |
| VGPR spill instructions | 0 | 0 | 0 | **715** (`scratch_store_b32` 334, `scratch_load_b128` 252, `scratch_load_b32` 71) |
| Threadgroup | 32×1×1 | 32×1×1 | 32×1×1 | 32×1×1 |

The eight-bank Full FP16 kernel is at the ISA's VGPR ceiling: 256 registers per wave
is the largest allocation the architecture supports, i.e. the minimum occupancy a
kernel can request, and the compiler additionally spills 768 bytes per wave. Its
instruction count is 10.3× the one-bank kernel (5,082 vs 491) and its static WMMA
count is 8×.

This is the measured basis for two conclusions:

1. **The Full FP16 accuracy fix is not free.** It buys accuracy with a 3.6× register
   increase, a spill, +1,536 B LDS and +4,591 instructions relative to `main`.
2. **A 16-bank variant is not implementable on this ISA.** Eight banks already
   consume 8 banks × 4 tiles × 8 VGPRs = 256 VGPRs per lane; sixteen banks would need
   512, twice the per-wave ceiling. It is rejected on measured resources, not on
   speculation, and was therefore not built.

## 6. Candidate B (64×32 tile) from the same evidence

The mixed kernel uses 32 lanes, a 32×32 tile, 71 VGPRs and 3,072 B LDS. A 64×32 tile
doubles the accumulator fragments per lane (8 × 16×16 FP32 accumulators = 64 VGPRs
before operands) and requires 2 A fragments × 4 row groups of staging, so LDS rises to
8 KB and VGPRs to roughly 130–160. That is a legal tile, but the measured Mixed
baseline is already issue-bound rather than occupancy-bound at 71 VGPRs, so the
expected gain is smaller than the accuracy/risk cost of a new mixed kernel. A 64×32
candidate was **not implemented** in this phase; see `PHASE_LXV_PERFORMANCE.md` §5 for
the explicit status.

## 7. What these captures prove and do not prove

Proved:

* RGP 2.7.1 profiles real dispatches of these benchmarks on this APU.
* The compiled gfx1103 ISA contains `v_wmma_f32_16x16x16_f16` ×4 (mixed) and
  `v_wmma_f16_16x16x16_f16` ×32 (Final eight-bank full FP16).
* Final mixed and Candidate A mixed execute the same code object.
* Register, LDS, scratch, spill and wavefront facts in §5 are the driver's own
  recorded pipeline state.

Not proved here:

* Per-instruction dynamic hit counts (SQTT is captured, not decoded).
* Internal WMMA micro-operations or their rounding cadence. Rounding behaviour is
  supported by measured numerical output against independent references
  (`PHASE_LXV_ACCURACY.md`), not by architecture prose.
* Any throughput number. Profiled runs are not used for timing.
