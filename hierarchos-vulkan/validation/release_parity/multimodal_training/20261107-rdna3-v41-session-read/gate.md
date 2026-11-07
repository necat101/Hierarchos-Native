# RDNA3 V4.1 performance gate

Policy: `clear-regression-only`; result: **PASS**.

| Case | Role | Paired median | 95% CI | Conclusion |
|---|---:|---:|---:|---|
| native_vit_4x4 | native | 1.0280x | [0.9771, 1.0378] | inconclusive |
| native_vit_6x4 | native | 1.0305x | [1.0160, 1.0450] | candidate_faster |
| llama | control | 1.0035x | [0.9863, 1.1826] | inconclusive |
| gpt2 | control | 1.0102x | [0.9935, 1.0884] | inconclusive |
| t5 | control | 1.0003x | [0.9898, 1.0161] | inconclusive |

Significant regressions: none

Inconclusive cases (not treated as regressions): native_vit_4x4, llama, gpt2, t5

A candidate is faster only when its full interval is above 1.0. A regression fails only when its full interval is below 1.0.

