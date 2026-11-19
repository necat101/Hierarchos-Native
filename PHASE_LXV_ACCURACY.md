# Phase LXV — accuracy margin across seeds, and what it actually says

The eight-bank Full FP16 arm is unchanged by this phase. What changed is the
*evidence*: the single-seed pass is now a fixed, recorded eight-seed sweep, and the
margin behind each recovered gate is computed instead of asserted.

Everything below is recomputable from the committed receipts in
`hierarchos-vulkan/validation/phase_lxv_evidence/accuracy/` by
`hierarchos-vulkan/validation/check_phase_lxv.py`.

## 1. The recorded result at the recorded seed

Seed `0x5EED1234` (`1593426484`), the seed every earlier receipt used:

| Comparison | Gates | Failed |
| --- | ---: | ---: |
| Mixed | 37 | 0 |
| Full FP16, historical `tile16` oracle | 37 | 0 |
| Full FP16, contract-matched `split8` oracle | 37 | 0 |

That is the 37/37 the Phase LXIV record claims, reproduced exactly and
deterministically (two consecutive runs produce byte-identical result rows).
Worst `vs FP32` value is **0.02604866** against the unchanged 0.05 limit.

**One correction to the PR text.** The live PR body lists
`16x1024x64 wmma-full-fp16 vs matched oracle` as `0.03125` and the worst FP32 value
as `0.0260372`. The current binary produces `0.015625` and `0.02604866` at the same
seed. The two gates still pass, but the PR table is not reproducible from the
committed binary and should not be read as a receipt; this document supersedes it.

## 2. Eight fixed seeds, and what breaks

The same suite was run at eight fixed seeds: `1593426484`, `42`, `1337`, `271828`,
`314159`, `777`, `8675309`, `20261119`, each under `tile16`, `split8`,
`rdna3-dot2`, `split2` (Full FP16) and `tile16` (Mixed).

| Seed | `tile16` failures | `split8` failures | Mixed failures |
| --- | ---: | ---: | ---: |
| 1593426484 | 0 | 0 | 0 |
| 42 | 0 | 0 | 0 |
| 1337 | 0 | 0 | 0 |
| 271828 | 0 | 0 | 0 |
| 314159 | 0 | 0 | 0 |
| **777** | **1** | 0 | 0 |
| 8675309 | 0 | 0 | 0 |
| **20261119** | **1** | 0 | 0 |

The failures are the same gate family and the same magnitude:

| Seed | Gate | Measured | Limit | ULP |
| --- | --- | ---: | ---: | ---: |
| 777 | `128x128x96 trans_a=false trans_b=true vs oracle` | 0.0390625 | 0.03125 | 2.5 |
| 20261119 | `128x128x96 trans_a=true trans_b=true vs oracle` | 0.0390625 | 0.03125 | 2.5 |

So "37/37" is a property of the recorded seed, not of the implementation. The
eight-bank arm is *much* better than `main` (which failed nine gates at every seed
tried), and it is not unconditionally inside the historical two-ULP bound.

The contract-matched `split8` oracle fails **nothing at any seed**. Its worst
hardware-versus-oracle gate is `37x8192x48` at 0.015625 against 0.03125 (exactly 50 %
of the limit, i.e. 1.0 ULP), and the worst value anywhere in the `split8` suite is
0.03338909 against the 0.05 FP32 budget (67 % of its limit). The hardware agrees with
the contract it implements with real margin; what it does not always do is agree with
a *single-accumulator* `tile16` model to within two ULP, which is a statement about
comparing two different FP16 reduction orders, not about a defect in this one.

The older `rdna3-dot2` and `split2` models are retained and still run; they fail
5–8 gates each, exactly as before. They are historical hypotheses about the
hardware cadence, not the contract this implementation targets, and they are kept
so the record does not silently drop an inconvenient reference.

## 3. Headroom on the nine originally failing gates

Default seed, unchanged limits. `ULP` is the distance in binary16 ULPs of the
largest reference magnitude (a two-ULP gate has limit `2 × ULP`, so
`ULP = 2 × measured/limit`).

| # | Gate | Measured | Limit | Ratio | ULP | Headroom | 25 % target |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `128x128x96 vs matched oracle` | 0.03125 | 0.03125 | 1.0000 | 2.0 | 0 % | **no** |
| 2 | `16x1024x64 vs matched oracle` | 0.015625 | 0.03125 | 0.5000 | 1.0 | 50 % | yes |
| 3 | `37x8192x48 vs matched oracle` | 0.0234375 | 0.03125 | 0.7500 | 1.5 | 25 % | yes |
| 4 | `128x128x96 t_a=true t_b=false vs oracle` | 0.03125 | 0.03125 | 1.0000 | 2.0 | 0 % | **no** |
| 5 | `128x128x96 t_a=true t_b=false vs FP32` | 0.02604866 | 0.05 | 0.5210 | — | 47.9 % | yes |
| 6 | `128x128x96 t_a=false t_b=true vs oracle` | 0.03125 | 0.03125 | 1.0000 | 2.0 | 0 % | **no** |
| 7 | `128x128x96 t_a=false t_b=true vs FP32` | 0.02227402 | 0.05 | 0.4455 | — | 55.5 % | yes |
| 8 | `128x128x96 t_a=true t_b=true vs oracle` | 0.03125 | 0.03125 | 1.0000 | 2.0 | 0 % | **no** |
| 9 | `128x128x96 t_a=true t_b=true vs FP32` | 0.02286172 | 0.05 | 0.4572 | — | 54.3 % | yes |

**Five of nine meet the declared 25 % target; four do not.** The four are the same
K=96 tile in four layouts, all at exactly two ULP, and two seeds push that family to
2.5 ULP. The target is therefore **not met**, and no implementation of the eight-bank
kind can be expected to meet it: two different orders of FP16 accumulation of the
same K=96 dot product differ by one to two ULPs of the result by construction, so a
1.5-ULP agreement with an arbitrary reference order is not achievable in binary16.
What *is* achievable, and is measured, is ≥ 33 % headroom against the contract-matched
model at every seed.

No tolerance was widened to produce any number above. The nine limits are the
originals, pinned in `check_phase_lxv.py::ORIGINAL_NINE`.

## 4. The 16-bank experiment: rejected on measured resources

A sixteen-bank pairwise merge was evaluated and **not built**. The reason is
measured, not speculative: RGP's own pipeline metadata for the eight-bank kernel
records **256 VGPRs per wave**, the architectural maximum, plus 768 bytes per wave
of scratch spill and 715 spill instructions (`validation/PHASE_LXV_RGP_HARDWARE.md`
§5). Eight banks already consume `8 banks × 4 tiles × 8 VGPRs = 256`; sixteen banks
would need 512 VGPRs per wave, twice the ceiling. The variant cannot exist on this
ISA without moving accumulators to memory, which would be slower than the thing it
is trying to improve and would almost certainly lose accuracy to the extra round
trips.

It also would not address the four zero-margin gates: those are a model-order
difference, not an accumulation-length problem, so more banks change the order again
without guaranteeing a closer match to `tile16`.

`check_phase_lxv.py` requires this rejection to stay recorded (`sixteen_bank`,
`implemented: false`, `reason: vgpr-wall`).

## 5. What the accuracy costs

Measured in `validation/PHASE_LXV_PERFORMANCE.md` §5: the eight-bank Full FP16
kernel is **7.6×–10.6× slower** than the single-accumulator kernel it replaces on
the four production shapes (e.g. 64.81 ms vs 6.14 ms at 2048×768×2048), with the
same 12-pair ABBA method used elsewhere. The compiled kernel runs at the 256-VGPR
architectural maximum with 715 spill instructions. The Mixed path is unaffected
and keeps its 1.39×–1.63× advantage over main, so the accuracy fix is paid for in
Full FP16 throughput only — but it is a real cost and is not presented here as
free.

## 6. What is still not covered

* **Model-graph operands.** The sweep uses the deterministic fixtures, not operands
  captured from a live model graph. The fixtures cover irregular tiles, partial
  tiles, long reductions (K = 8192), wide outputs (N = 8192) and all four
  transpose combinations, but not a real activation distribution.
* **Overflow and non-finite stress.** `nonfinite` is asserted to be zero on every
  gate, and the library's own Full FP16 documentation states that FP16 accumulation
  can overflow; no dedicated saturation/near-overflow fixture was added in this
  phase.
* **Negative controls on the numerical receipts as CI artifacts.** The gate rejects
  edited limits, forged pass flags, hidden failures, dropped seeds and a false margin
  claim (`validation/test_phase_lxv.py`), but it cannot re-run the GPU.
