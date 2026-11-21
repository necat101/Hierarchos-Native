# Phase LXVI — K=96 accuracy margin, independent numerical simulation

**This is a simulation, not a GPU measurement.** It reimplements the
qualification fixtures and the accumulation structures in Python so that a
bounded set of alternative reductions can be evaluated before any shader is
changed. The GPU remains the authority; the simulation predicts, and the
existing eight-bank receipt is what confirms.

Seeds: 42, 777, 1337, 271828, 314159, 8675309, 20261119, 1593426484.
Layouts: trans_a=0, trans_b=0, trans_a=1, trans_b=0, trans_a=0, trans_b=1, trans_a=1, trans_b=1.
Unchanged two-ULP limit: 0.03125. 25 % headroom target: 0.0234375.

## K=96 (128x128x96), worst case over 8 seeds x 4 layouts

| Structure | worst abs vs FP64 | worst abs vs tile16 | tile16 gates over limit | 25 % target met |
|---|---:|---:|---:|---:|
| rdna3-dot2 | 0.078466 | 0.078125 | 32/32 | 0/32 |
| split2 | 0.041371 | 0.046875 | 23/32 | 0/32 |
| split6-compact-pairwise | 0.030353 | 0.039062 | 2/32 | 0/32 |
| split8-block-pairwise | 0.030353 | 0.039062 | 2/32 | 0/32 |
| split8-roundrobin-balanced | 0.034177 | 0.046875 | 5/32 | 0/32 |
| split8-roundrobin-left | 0.032652 | 0.046875 | 3/32 | 0/32 |
| split8-roundrobin-pairwise | 0.030353 | 0.039062 | 2/32 | 0/32 |
| split8-strided3-pairwise | 0.028611 | 0.031250 | 0/32 | 0/32 |
| tile16 | 0.033258 | 0.000000 | 0/32 | 32/32 |

## What the numbers say

* The current kernel structure matches the six-active-bank picture: for K=96 exactly
  six banks receive one 16-wide slice each and two stay zero, and the pairwise tree
  already merges six equal leaves with a maximum intermediate partial sum of two
  thirds of the result. No schedule or merge order for six leaves can do better than
  that; the alternatives below differ by at most a rounding step.
* Current structure versus FP64: worst **0.030353**, well inside the
  unchanged 0.05 gate and consistent with the eight-bank receipts.
* Current structure versus the historical `tile16` oracle: worst
  **0.039062** against a 0.03125 limit. 2 of 32 cases exceed the limit in the
  simulation; the GPU receipts record them as exactly at the limit.
* A structure closer to `tile16` is a structure that accumulates the whole reduction
  in one FP16 register — the pre-LXIV behaviour. That is the row measured as
  `rdna3-dot2`, and it is the row that failed nine original gates. The margin gap
  and the FP32-accuracy requirement therefore pull in opposite directions: closing
  the four zero-margin gates against the retained historical model would mean
  giving up the accuracy fix that the same phase is required to preserve.

## Conclusion

No FP16-only schedule or merge order moves the K=96 gates materially. The four
zero-margin gates are a mismatch between the current kernel and a superseded
reference model, not a rounding error that a better tree can remove. The gap is
recorded as **open**; the shader was not changed, and no gate, tolerance, seed or
layout was dropped.
