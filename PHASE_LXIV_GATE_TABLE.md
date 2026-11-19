# Phase LXIV - full 37-gate Full FP16 comparison

Generated from the qualification receipts, not transcribed. `MAIN` and `A` share every
value (the Phase XII change is load-only, and the Full FP16 receipt is byte-identical).
`FINAL tile16` is the unchanged default reference; `FINAL split2` is the same kernel
graded against the contract-matched model. Limits are the recorded, unchanged values.

| # | Gate | shape | layouts | oracle gate | limit | MAIN/A | FINAL tile16 | FINAL split2 | verdict A | verdict FINAL tile16 |
|--:|---|---|---|---|--:|--:|--:|--:|:-:|:-:|
| 1 | `64x48x40 wmma-full-fp16 vs FP32` | 64x48x40 | row-major | FP32 (f64) | 0.05 | 0.0222225 | 0.0160141 | 0.0160141 | pass | pass |
| 2 | `64x48x40 wmma-full-fp16 vs matched oracle` | 64x48x40 | row-major | matched oracle | 0.03125 (2 ULP) | 0.03125 | 0.015625 | 0.00390625 | pass | pass |
| 3 | `64x48x40 matched fp16 oracle vs FP32` | 64x48x40 | row-major | matched oracle | 0.05 (2 ULP) | 0.00940323 | 0.00940323 | 0.0160141 | pass | pass |
| 4 | `33x17x19 wmma-full-fp16 vs FP32` | 33x17x19 | row-major | FP32 (f64) | 0.05 | 0.00642109 | 0.0107441 | 0.0107441 | pass | pass |
| 5 | `33x17x19 wmma-full-fp16 vs matched oracle` | 33x17x19 | row-major | matched oracle | 0.01562 (2 ULP) | 0.0078125 | 0.0078125 | 0 | pass | pass |
| 6 | `33x17x19 matched fp16 oracle vs FP32` | 33x17x19 | row-major | matched oracle | 0.05 (2 ULP) | 0.00550556 | 0.00550556 | 0.0107441 | pass | pass |
| 7 | `128x128x96 wmma-full-fp16 vs FP32` | 128x128x96 | row-major | FP32 (f64) | 0.05 | 0.0499535 | 0.0389252 | 0.0389252 | pass | pass |
| 8 | `128x128x96 wmma-full-fp16 vs matched oracle` | 128x128x96 | row-major | matched oracle | 0.03125 (2 ULP) | 0.0625 | 0.046875 | 0.0078125 | FAIL | FAIL |
| 9 | `128x128x96 matched fp16 oracle vs FP32` | 128x128x96 | row-major | matched oracle | 0.05 (2 ULP) | 0.0203609 | 0.0203609 | 0.0389252 | pass | pass |
| 10 | `16x1024x64 wmma-full-fp16 vs FP32` | 16x1024x64 | row-major | FP32 (f64) | 0.05 | 0.0387287 | 0.0246086 | 0.0246086 | pass | pass |
| 11 | `16x1024x64 wmma-full-fp16 vs matched oracle` | 16x1024x64 | row-major | matched oracle | 0.03125 (2 ULP) | 0.0390625 | 0.03125 | 0.015625 | FAIL | pass |
| 12 | `16x1024x64 matched fp16 oracle vs FP32` | 16x1024x64 | row-major | matched oracle | 0.05 (2 ULP) | 0.0163383 | 0.0163383 | 0.0246086 | pass | pass |
| 13 | `256x64x16 wmma-full-fp16 vs FP32` | 256x64x16 | row-major | FP32 (f64) | 0.05 | 0.00815582 | 0.00815582 | 0.00815582 | pass | pass |
| 14 | `256x64x16 wmma-full-fp16 vs matched oracle` | 256x64x16 | row-major | matched oracle | 0.01562 (2 ULP) | 0.0078125 | 0.0078125 | 0.00390625 | pass | pass |
| 15 | `256x64x16 matched fp16 oracle vs FP32` | 256x64x16 | row-major | matched oracle | 0.05 (2 ULP) | 0.00535202 | 0.00535202 | 0.00815582 | pass | pass |
| 16 | `37x8192x48 wmma-full-fp16 vs FP32` | 37x8192x48 | row-major | FP32 (f64) | 0.05 | 0.0373001 | 0.0258274 | 0.0258274 | pass | pass |
| 17 | `37x8192x48 wmma-full-fp16 vs matched oracle` | 37x8192x48 | row-major | matched oracle | 0.03125 (2 ULP) | 0.046875 | 0.03125 | 0.0117188 | FAIL | pass |
| 18 | `37x8192x48 matched fp16 oracle vs FP32` | 37x8192x48 | row-major | matched oracle | 0.05 (2 ULP) | 0.0172062 | 0.0172062 | 0.0258274 | pass | pass |
| 19 | `64x48x40 trans_a=true trans_b=false vs oracle` | 64x48x40 | T=true,B=false | matched oracle | 0.03125 | 0.0234375 | 0.0234375 | 0.00195312 | pass | pass |
| 20 | `64x48x40 trans_a=true trans_b=false vs FP32` | 64x48x40 | T=true,B=false | FP32 (f64) | 0.05 | 0.0258808 | 0.0172453 | 0.0172453 | pass | pass |
| 21 | `64x48x40 trans_a=false trans_b=true vs oracle` | 64x48x40 | T=false,B=true | matched oracle | 0.03125 | 0.0234375 | 0.015625 | 0.00195312 | pass | pass |
| 22 | `64x48x40 trans_a=false trans_b=true vs FP32` | 64x48x40 | T=false,B=true | FP32 (f64) | 0.05 | 0.0206366 | 0.0155792 | 0.0155792 | pass | pass |
| 23 | `64x48x40 trans_a=true trans_b=true vs oracle` | 64x48x40 | T=true,B=true | matched oracle | 0.03125 | 0.0175781 | 0.015625 | 0 | pass | pass |
| 24 | `64x48x40 trans_a=true trans_b=true vs FP32` | 64x48x40 | T=true,B=true | FP32 (f64) | 0.05 | 0.0169992 | 0.0173244 | 0.0173244 | pass | pass |
| 25 | `33x17x19 trans_a=true trans_b=false vs oracle` | 33x17x19 | T=true,B=false | matched oracle | 0.01562 | 0.0078125 | 0.0078125 | 0 | pass | pass |
| 26 | `33x17x19 trans_a=true trans_b=false vs FP32` | 33x17x19 | T=true,B=false | FP32 (f64) | 0.05 | 0.00763178 | 0.00763178 | 0.00763178 | pass | pass |
| 27 | `33x17x19 trans_a=false trans_b=true vs oracle` | 33x17x19 | T=false,B=true | matched oracle | 0.01562 | 0.0078125 | 0.0078125 | 0 | pass | pass |
| 28 | `33x17x19 trans_a=false trans_b=true vs FP32` | 33x17x19 | T=false,B=true | FP32 (f64) | 0.05 | 0.0120773 | 0.0120773 | 0.0120773 | pass | pass |
| 29 | `33x17x19 trans_a=true trans_b=true vs oracle` | 33x17x19 | T=true,B=true | matched oracle | 0.01562 | 0.0117188 | 0.0078125 | 0 | pass | pass |
| 30 | `33x17x19 trans_a=true trans_b=true vs FP32` | 33x17x19 | T=true,B=true | FP32 (f64) | 0.05 | 0.00867844 | 0.00867844 | 0.00867844 | pass | pass |
| 31 | `128x128x96 trans_a=true trans_b=false vs oracle` | 128x128x96 | T=true,B=false | matched oracle | 0.03125 | 0.0703125 | 0.046875 | 0.015625 | FAIL | FAIL |
| 32 | `128x128x96 trans_a=true trans_b=false vs FP32` | 128x128x96 | T=true,B=false | FP32 (f64) | 0.05 | 0.0731049 | 0.0318184 | 0.0318184 | FAIL | pass |
| 33 | `128x128x96 trans_a=false trans_b=true vs oracle` | 128x128x96 | T=false,B=true | matched oracle | 0.03125 | 0.0625 | 0.046875 | 0.015625 | FAIL | FAIL |
| 34 | `128x128x96 trans_a=false trans_b=true vs FP32` | 128x128x96 | T=false,B=true | FP32 (f64) | 0.05 | 0.0637856 | 0.0419369 | 0.0419369 | FAIL | pass |
| 35 | `128x128x96 trans_a=true trans_b=true vs oracle` | 128x128x96 | T=true,B=true | matched oracle | 0.03125 | 0.0625 | 0.046875 | 0.015625 | FAIL | FAIL |
| 36 | `128x128x96 trans_a=true trans_b=true vs FP32` | 128x128x96 | T=true,B=true | FP32 (f64) | 0.05 | 0.0632477 | 0.0367794 | 0.0367794 | FAIL | pass |
| 37 | `64x64x64 plain-f16-storage vs matched oracle` | 64x64x64 | row-major | matched oracle | 0.001 (2 ULP) | 0 | 0 | 0 | pass | pass |

Totals: `MAIN`/`A` 37 gates / 9 failed; `FINAL` (default `tile16`) 37 gates / 4 failed;
`FINAL` (`--full-oracle split2`) 37 gates / 0 failed; Mixed 37 / 0 in every case.
