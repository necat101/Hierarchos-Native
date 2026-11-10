# Phase XVI full-fine-tuning stability — 2026-11-10

This extension closes the full-parameter training convergence gap without
changing the two-pass shaders or the AdamW arithmetic. The sealed
`20261110-phase16-training-qualification` receipt remains the qualification for
source `8a12af2d907c`; this receipt binds the new source and its validation.

- Source revision: `21027bf9699757859e7f2a9c07c437402d5aa2ae` (clean tree).
- Evidence run revision: `fafa54a21492f53dac972499384b79604963719b`.
- Receipt: `validation/release_parity/multimodal_training/20261110-phase16-training-stability/`.
- Device: AMD Radeon Graphics (Ryzen Z1 Extreme, `1002:15bf`).
- Checker: `STABILITY QUALIFICATION: PASS`; controls: **31/31**.
- Preserved Phase XVI checker: `PASS` (`8a12af2d907c`, 8 arms, 3 resume arms,
  8 performance arms).
- Shader gate: **45/45** byte-identical artifacts (43 sealed + 2 Phase XVI).

## Learning-rate sweep

The declared rule requires finite losses, final loss below half the first,
last-20% mean below half the first-20% mean, maximum no more than 1.5× the
first, and a negative least-squares slope. Among stable rates it selects the
lowest late-window mean, with the lower LR breaking ties.

| LR | First | Final | Min | Max | Early 20% | Late 20% | Stable |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1e-4 | 7.113463 | 0.017277505 | 0.017054603 | 7.113463 | 4.618341 | 0.020376778 | yes |
| 3e-4 | 7.113463 | 0.003029183 | 0.003029183 | 7.113463 | 3.288417 | 0.003548969 | yes |
| **1e-3** | **7.113463** | **0.000778376** | **0.000757654** | **7.113463** | **2.892176** | **0.000900910** | **yes** |
| 3e-3 | 7.113463 | 0.063013780 | 0.004672483 | 7.291460 | 4.936856 | 0.104517650 | yes |
| 1e-2 | 7.113463 | 2.837122400 | 2.825421000 | 16.402319 | 9.815657 | 3.179368740 | no |

`1e-3` was selected because it has the lowest stable late-window mean. The
`1e-2` control remains recorded as unstable: its maximum loss is 2.31× the
first loss and its late mean rose to 3.18. Seeds 13 and 21 reproduced the
selected-LR trajectory exactly.

## Sustained full-FT convergence and equivalence

All four arms ran 100 optimizer steps at `1e-3` with the same fixture, ordering,
seed and AdamW settings.

| Arm | Precision | Backward | First | Final | Reduction | Dispatch |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | FP32 | scalar | 7.113463 | 0.000778376 | 0.999891 | 200 scalar |
| B | FP32 | two-pass | 7.113463 | 0.000778376 | 0.999891 | 200 query + 200 key |
| C | mixed WMMA | scalar | 7.113452 | 0.000824489 | 0.999884 | 200 scalar |
| D | mixed WMMA | two-pass | 7.113452 | 0.000824489 | 0.999884 | 200 query + 200 key |

A/B and C/D are bitwise identical for every recorded loss bit, parameter digest,
update norm and maximum absolute update, and finish with identical parameter
digests. Mixed drift is recorded separately: B vs D first-step absolute drift is
`1.06e-5` (within the `2e-2` policy), maximum `1.7991e-3`, RMS `3.0501e-4`, and
final `4.611e-5`.

Per-step reports record the actual LR, beta1 `0.9`, beta2 `0.999`, epsilon
`1e-8`, weight decay `0`, optimizer step/status, loss bits, parameter digest,
update norm and maximum absolute update. Full-FT raw gradient tensors are not
exposed; their gradient norms remain unavailable rather than being inferred
from parameter digests.

## Resume, reuse and performance

FP32 scalar LoRA (E), FP32 two-pass LoRA (F) and mixed two-pass LoRA (H) each
compare an uninterrupted 12-step run with an 8-step checkpoint plus 4-step
resume. Loss bits, adapter bytes and optimizer bytes are exactly equal; resumed
parameters remain frozen; dispatch counts match the arm's real route (E: 8
scalar; F/H: 8 query + 8 key).

The repeated 20-step runs are bit-identical. A 20-step seq-96 A/B pair also
matches its dispatch proof. The two-pass scratch has no public sentinel/poison
hook, so sentinel contamination is **not claimed**; this is recorded in the
receipt limitations.

Fresh paired benchmark, batch 8, seq 64, ABBA, warmup 3, 20 steps:

| Pair | Scalar median | Two-pass median | Speedup | 95% CI |
| --- | ---: | ---: | ---: | --- |
| A | 671.791 ms | 275.278 ms | **2.440×** | 2.435–2.445 |
| B | 672.707 ms | 276.189 ms | **2.436×** | 2.432–2.448 |

These are current-source measurements, not reused old-benchmark claims.

## Verification

```powershell
cargo build --release --bin training-stability
python validation/assemble_training_stability.py
python validation/check_training_stability.py
python -m unittest discover -s validation -p "test_training_stability.py"
python validation/check_training_phase16.py
python validation/emit_phase16_spirv_gate.py --output target/phase16/training-stability-spirv.json
python validation/check_phase16_receipts.py
```

The control suite covers missing receipts/sources, fixture config/weights/
dataset bindings, per-step LR and every AdamW field, optimizer status/step,
forged steps and dispatches, scalar fallback, tampered profiles/losses/
parameters, failed-subprocess return codes, full-FT/LoRA mislabeling, missing
or changed adapter/optimizer bytes, wrong resumed steps/losses, selection
tampering and performance-sample tampering. The old test named
`changed_optimizer_configuration` only flipped the two-pass flag; it has been
replaced by real optimizer-field controls.

## Remaining limits

- Full-parameter exact checkpoint resume is still unsupported and explicitly
  refused; only the LoRA checkpoint contract is exact.
- The sweep measures training loss, not held-out generalization.
- Full-FT raw gradient tensors and a public scratch sentinel hook are not
  available.
- Mixed WMMA remains opt-in; the device/route scope is the tested AMD RDNA3
  machine.
