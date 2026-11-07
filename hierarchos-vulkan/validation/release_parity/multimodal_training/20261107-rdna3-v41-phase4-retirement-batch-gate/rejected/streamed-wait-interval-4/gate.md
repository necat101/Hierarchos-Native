# RDNA3 V4.1 performance gate

Policy: `clear-regression-only`; result: **PASS**.

| Case | Role | Paired median | 95% CI | Conclusion |
|---|---:|---:|---:|---|
| native_vit_4x4 | native | 1.0201x | [0.9969, 1.0465] | inconclusive |
| native_vit_6x4 | native | 1.0055x | [0.9746, 1.1077] | inconclusive |
| llama | control | 1.0292x | [0.9891, 1.1177] | inconclusive |
| gpt2 | control | 1.0072x | [0.9666, 1.1144] | inconclusive |
| t5 | control | 1.0045x | [0.9490, 1.0606] | inconclusive |

Significant regressions: none

Inconclusive cases (not treated as regressions): native_vit_4x4, native_vit_6x4, llama, gpt2, t5

A candidate is faster only when its full interval is above 1.0. A regression fails only when its full interval is below 1.0.

