# RDNA3 V4.1 performance gate

Mode: `development`; state: **INCONCLUSIVE**.

Reason: 5 case(s) are not release-acceptable.

Batches: `1` (required: `1`).

| Batch | Case | Role | Paired median | 95% CI | Margin | State |
|---:|---|---:|---:|---:|---:|---|
| 1 | native_vit_4x4 | native | 1.0068x | [0.9914, 1.0860] | +/-0.0200 | INCONCLUSIVE |
| 1 | native_vit_6x4 | native | 1.0037x | [0.9891, 1.0763] | +/-0.0200 | INCONCLUSIVE |
| 1 | llama | control | 1.0148x | [0.9888, 1.0636] | +/-0.0200 | INCONCLUSIVE |
| 1 | gpt2 | control | 1.0033x | [0.9547, 1.2453] | +/-0.0200 | INCONCLUSIVE |
| 1 | t5 | control | 1.0089x | [1.0024, 1.0263] | +/-0.0200 | INCONCLUSIVE |

PASS requires conclusive evidence. REGRESSION and INVALID always fail. Development mode does not block on INCONCLUSIVE, but it does not label the performance qualification successful. Release mode requires PASS for every case at batches 1, 2, and 4.

