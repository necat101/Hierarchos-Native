# Phase LXIV - full 37-gate Full FP16 comparison

Generated from the qualification receipts, not transcribed. `MAIN`/`A` are the
recorded single-accumulator result (`target/wmma-promotion-evidence/
qual_baseline_full-fp16.json`); `FINAL` is the eight-bank split reduction
measured in this phase. Limits are the recorded, unchanged values.

| # | Gate | limit | MAIN/A | FINAL (tile16) | FINAL (split8) | verdict MAIN/A | verdict FINAL |
|--:|---|---|--:|--:|--:|:-:|:-:|
| 1 | `64x48x40 wmma-full-fp16 vs FP32` | 0.05 | 0.022222519 | 0.013396263 | 0.013396263 | pass | pass |
| 2 | `64x48x40 wmma-full-fp16 vs matched oracle` | 0.03125 (2 ULP) | 0.03125 | 0.015625 | 0.0078125 | pass | pass |
| 3 | `64x48x40 matched fp16 oracle vs FP32` | 0.05 (2 ULP) | 0.0094032288 | 0.0094032288 | 0.013396263 | pass | pass |
| 4 | `33x17x19 wmma-full-fp16 vs FP32` | 0.05 | 0.0064210892 | 0.010744095 | 0.010744095 | pass | pass |
| 5 | `33x17x19 wmma-full-fp16 vs matched oracle` | 0.01562 (2 ULP) | 0.0078125 | 0.0078125 | 0 | pass | pass |
| 6 | `33x17x19 matched fp16 oracle vs FP32` | 0.05 (2 ULP) | 0.0055055618 | 0.0055055618 | 0.010744095 | pass | pass |
| 7 | `128x128x96 wmma-full-fp16 vs FP32` | 0.05 | 0.049953461 | 0.02555275 | 0.02555275 | pass | pass |
| 8 | `128x128x96 wmma-full-fp16 vs matched oracle` | 0.03125 (2 ULP) | 0.0625 | 0.03125 | 0.015625 | FAIL | pass |
| 9 | `128x128x96 matched fp16 oracle vs FP32` | 0.05 (2 ULP) | 0.020360947 | 0.020360947 | 0.02555275 | pass | pass |
| 10 | `16x1024x64 wmma-full-fp16 vs FP32` | 0.05 | 0.038728714 | 0.026037216 | 0.026037216 | pass | pass |
| 11 | `16x1024x64 wmma-full-fp16 vs matched oracle` | 0.03125 (2 ULP) | 0.0390625 | 0.03125 | 0.0078125 | FAIL | pass |
| 12 | `16x1024x64 matched fp16 oracle vs FP32` | 0.05 (2 ULP) | 0.016338348 | 0.016338348 | 0.026037216 | pass | pass |
| 13 | `256x64x16 wmma-full-fp16 vs FP32` | 0.05 | 0.0081558228 | 0.0081558228 | 0.0081558228 | pass | pass |
| 14 | `256x64x16 wmma-full-fp16 vs matched oracle` | 0.01562 (2 ULP) | 0.0078125 | 0.0078125 | 0.00390625 | pass | pass |
| 15 | `256x64x16 matched fp16 oracle vs FP32` | 0.05 (2 ULP) | 0.0053520203 | 0.0053520203 | 0.0081558228 | pass | pass |
| 16 | `37x8192x48 wmma-full-fp16 vs FP32` | 0.05 | 0.03730011 | 0.023813248 | 0.023813248 | pass | pass |
| 17 | `37x8192x48 wmma-full-fp16 vs matched oracle` | 0.03125 (2 ULP) | 0.046875 | 0.0234375 | 0.0078125 | FAIL | pass |
| 18 | `37x8192x48 matched fp16 oracle vs FP32` | 0.05 (2 ULP) | 0.017206192 | 0.017206192 | 0.023813248 | pass | pass |
| 19 | `64x48x40 trans_a=true trans_b=false vs oracle` | 0.03125 (2 ULP) | 0.0234375 | 0.015625 | 0.001953125 | pass | pass |
| 20 | `64x48x40 trans_a=true trans_b=false vs FP32` | 0.05 | 0.025880814 | 0.012126923 | 0.012126923 | pass | pass |
| 21 | `64x48x40 trans_a=false trans_b=true vs oracle` | 0.03125 (2 ULP) | 0.0234375 | 0.015625 | 0.0078125 | pass | pass |
| 22 | `64x48x40 trans_a=false trans_b=true vs FP32` | 0.05 | 0.020636559 | 0.017642975 | 0.017642975 | pass | pass |
| 23 | `64x48x40 trans_a=true trans_b=true vs oracle` | 0.03125 (2 ULP) | 0.017578125 | 0.015625 | 0.001953125 | pass | pass |
| 24 | `64x48x40 trans_a=true trans_b=true vs FP32` | 0.05 | 0.016999245 | 0.017324448 | 0.017324448 | pass | pass |
| 25 | `33x17x19 trans_a=true trans_b=false vs oracle` | 0.01562 (2 ULP) | 0.0078125 | 0.0078125 | 0 | pass | pass |
| 26 | `33x17x19 trans_a=true trans_b=false vs FP32` | 0.05 | 0.0076317787 | 0.0076317787 | 0.0076317787 | pass | pass |
| 27 | `33x17x19 trans_a=false trans_b=true vs oracle` | 0.01562 (2 ULP) | 0.0078125 | 0.0078125 | 0 | pass | pass |
| 28 | `33x17x19 trans_a=false trans_b=true vs FP32` | 0.05 | 0.012077332 | 0.012077332 | 0.012077332 | pass | pass |
| 29 | `33x17x19 trans_a=true trans_b=true vs oracle` | 0.01562 (2 ULP) | 0.01171875 | 0.0078125 | 0 | pass | pass |
| 30 | `33x17x19 trans_a=true trans_b=true vs FP32` | 0.05 | 0.0086784363 | 0.0086784363 | 0.0086784363 | pass | pass |
| 31 | `128x128x96 trans_a=true trans_b=false vs oracle` | 0.03125 (2 ULP) | 0.0703125 | 0.03125 | 0.015625 | FAIL | pass |
| 32 | `128x128x96 trans_a=true trans_b=false vs FP32` | 0.05 | 0.073104858 | 0.022974968 | 0.022974968 | FAIL | pass |
| 33 | `128x128x96 trans_a=false trans_b=true vs oracle` | 0.03125 (2 ULP) | 0.0625 | 0.03125 | 0.015625 | FAIL | pass |
| 34 | `128x128x96 trans_a=false trans_b=true vs FP32` | 0.05 | 0.063785553 | 0.021559715 | 0.021559715 | FAIL | pass |
| 35 | `128x128x96 trans_a=true trans_b=true vs oracle` | 0.03125 (2 ULP) | 0.0625 | 0.03125 | 0.0078125 | FAIL | pass |
| 36 | `128x128x96 trans_a=true trans_b=true vs FP32` | 0.05 | 0.063247681 | 0.023674011 | 0.023674011 | FAIL | pass |
| 37 | `64x64x64 plain-f16-storage vs matched oracle` | 0.001 (2 ULP) | 0 | 0 | 0 | pass | pass |

Totals: `MAIN`/`A` 37 gates / 9 failed; `FINAL` (historical `tile16` default) 37 / 0; `FINAL` (`--full-oracle split8`, contract-matched) 37 / 0; Mixed 37 / 0.

The `tile16` default is unchanged and still models one accumulator rounding per
16-wide K slice. The eight-bank reduction now meets it on every fixture: each
bank carries 1/8 of the terms, so its rounding error is small enough that the
kernel sits inside the two-ULP budget of a model whose own error it does not
share. Five of the nine recovered gates land exactly on that two-ULP bound.
