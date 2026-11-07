# RDNA3 V4.1 performance gate

Policy: `clear-regression-only`; result: **PASS**.

| Case | Role | Paired median | 95% CI | Conclusion |
|---|---:|---:|---:|---|
| native_vit_4x4 | native | 0.9966x | [0.9856, 1.0108] | inconclusive |
| native_vit_6x4 | native | 1.0009x | [0.9886, 1.0083] | inconclusive |
| llama | control | 1.0118x | [0.9826, 1.0327] | inconclusive |
| gpt2 | control | 0.9946x | [0.9642, 1.0119] | inconclusive |
| t5 | control | 1.0014x | [0.9875, 1.0161] | inconclusive |

Significant regressions: none

Inconclusive cases (not treated as regressions): native_vit_4x4, native_vit_6x4, llama, gpt2, t5

A candidate is faster only when its full interval is above 1.0. A regression fails only when its full interval is below 1.0.

