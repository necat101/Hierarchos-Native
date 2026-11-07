# RDNA3 V4.1 performance gate

Mode: `development`; state: **INCONCLUSIVE**.

Reason: 5 case(s) are not release-acceptable.

Batches: `1` (required: `1`).

| Batch | Case | Role | Paired median | 95% CI | Margin | State |
|---:|---|---:|---:|---:|---:|---|
| 1 | native_vit_4x4 | native | 1.0381x | [0.9404, 1.0837] | +/-0.0200 | INCONCLUSIVE |
| 1 | native_vit_6x4 | native | 1.0163x | [0.9530, 1.0965] | +/-0.0200 | INCONCLUSIVE |
| 1 | llama | control | 1.0016x | [0.9900, 1.1446] | +/-0.0200 | INCONCLUSIVE |
| 1 | gpt2 | control | 0.9995x | [0.8917, 1.0404] | +/-0.0200 | INCONCLUSIVE |
| 1 | t5 | control | 1.0011x | [0.9735, 1.1928] | +/-0.0200 | INCONCLUSIVE |

PASS requires conclusive evidence. REGRESSION and INVALID always fail. Development mode does not block on INCONCLUSIVE, but it does not label the performance qualification successful. Release mode requires PASS for every case at batches 1, 2, and 4.

