# Phase XVIII — mixed-WMMA numerical drift investigation

Phase XVII measured a mixed-WMMA held-out gap of +0.094 validation
cross-entropy and +0.107 test against FP32. This phase asks whether that gap is
an implementation defect, ordinary FP16 operand rounding, or optimizer-driven
trajectory divergence.

- Source revision (receipt): `81f997a1419487d583667f7b25dfdac8df919555`.
- Harness: `src/bin/wmma_drift.rs` (observation only; no kernel, shader or
  optimizer change).
- Receipt: `validation/release_parity/multimodal_training/20261110-phase18-wmma-drift/`.
- Device: AMD Radeon Graphics (Ryzen Z1 Extreme, `1002:15bf`).
- Scale: 6 matched runs (3 training-window orders x 2 precision modes), 100
  steps each, 16,782,336 parameters, two-pass attention backward throughout.
- Checker: `WMMA DRIFT QUALIFICATION: PASS`; negative controls: **26/26**.

## What was executed

Each run trains on the Phase XVII training partition with the qualified
protocol (LR `1e-3`, AdamW `(0.9, 0.999, 1e-8)`, weight decay 0, batch 8, seq
64), logs every step, and evaluates validation at steps
0/1/2/5/10/20/40/60/80/100 and test at 0 and 100. Order 0 additionally dumps
exact FP32 master parameters at steps 0/1/10/100 and exports weights-only
checkpoints at 10 and 100.

Same-weight forward error is measured by dumping the exact logits of the
initial checkpoint and of the step-10 and step-100 FP32 exports, once per
execution mode. The mode is captured when the graph is built, so each dump is
a separate process; an in-process dual-graph attempt silently ran FP32 twice
and was discarded.

## 1. Same-weight forward error

18,874,368 logits per comparison (288 validation windows, padded final batch
included in the tensor but excluded from the loss), identical FP32 master
digests in both modes:

| Checkpoint | Max abs logit | RMS | Token-weighted CE delta | Mixed dispatches |
| --- | ---: | ---: | ---: | ---: |
| step 0 (initial) | 0.002709 | 4.902e-04 | +1.060e-05 | 540 |
| step 10 (FP32 weights) | 0.055173 | 4.581e-04 | +2.622e-05 | 540 |
| step 100 (FP32 weights) | 0.027724 | 3.281e-04 | +6.175e-06 | 540 |

540 = 36 batches x 15 projections; zero shape fallbacks; FP32 graph zero
cooperative dispatches. Both graphs' parameter digests and the exporting run's
digest agree, so this is inference arithmetic on identical weights.

The step-10 maximum exceeds the *initial-fixture* mixed contract of `2e-2`.
That contract was qualified for the frozen initial logits, not for arbitrary
trained weights, so it is not used as a pass/fail gate here. The quantity that
matters for the held-out comparison is the token-weighted CE effect, which
stays at 1e-5 — three to four orders of magnitude below the 0.09–0.17 gaps
being explained.

## 2. First divergence and one-step update

Training losses first differ bitwise at step 1: FP32 `7.148552` versus mixed
`7.148527`, an absolute difference of `2.5e-05`. That is the same order as the
forward error, so the backward/AdamW update begins from an almost identical
loss.

Exact parameter snapshots at steps 0 and 1 give the applied AdamW update
(weight decay 0):

| Group | Update cosine | Sign-flip fraction | Mixed/FP32 update RMS | Step-1 weight-divergence RMS |
| --- | ---: | ---: | ---: | ---: |
| attn.q_proj | 0.99862 | 0.081% | 1.000003 | 5.258e-05 |
| attn.k_proj | 0.99858 | 0.083% | 1.000002 | 5.332e-05 |
| attn.v_proj | 0.99948 | 0.028% | 1.000002 | 3.226e-05 |
| attn.o_proj | 0.99945 | 0.029% | 1.000001 | 3.324e-05 |
| mlp.gate_proj | 0.99925 | 0.042% | 1.000002 | 3.876e-05 |
| mlp.up_proj | 0.99925 | 0.042% | 1.000001 | 3.871e-05 |
| mlp.down_proj | 0.99936 | 0.037% | 0.999998 | 3.587e-05 |
| embed | 0.99940 | 0.031% | 0.999990 | 1.320e-05 |
| lm_head | 0.99948 | 0.031% | 0.999997 | 3.239e-05 |
| norm | 0.99907 | 0.059% | 0.999875 | 4.320e-05 |

The update is almost the same vector: cosine at least 0.9986, fewer than 0.1%
of signs flipped, magnitude ratio within 0.02%. The largest per-parameter
differences are in the query and key projections; value, output and embedding
projections are smallest. This is consistent with FP16 operand rounding in the
projection GEMMs, not with a transposed, stale or missing dispatch.

The first *held-out* consequence is much larger than the parameter difference:
validation cross-entropy after step 1 is `6.7578` (FP32) versus `7.0602`
(mixed). A 5e-05 RMS weight difference moves a 7.1-loss model by 0.30, so the
early loss surface is steep and amplifies a small update difference.

## 3. Accumulated trajectory

Order 0 validation cross-entropy:

| Step | FP32 | Mixed | Gap |
| ---: | ---: | ---: | ---: |
| 0 | 7.119863 | 7.119873 | +0.000010 |
| 1 | 6.757844 | 7.060197 | +0.302353 |
| 2 | 6.666800 | 6.685841 | +0.019041 |
| 5 | 6.942471 | 7.066387 | +0.123916 |
| 10 | 5.700102 | 6.570716 | +0.870614 |
| 20 | 4.888866 | 4.812901 | -0.075965 |
| 40 | 3.939183 | 4.238646 | +0.299463 |
| 60 | 3.259062 | 3.364547 | +0.105485 |
| 80 | 2.892960 | 2.996094 | +0.103134 |
| 100 | 2.831961 | 2.926205 | +0.094244 |

The gap changes sign and magnitude repeatedly and is largest at step 10. This
is chaotic divergence of two nearby trajectories, not a constant inference or
gradient bias. Final weight divergence at step 100 (RMS): q `9.53e-03`,
k `8.38e-03`, embed `7.10e-03`, norm `4.72e-03`, down `4.28e-03`, lm_head
`4.00e-03`, gate `3.65e-03`, up `3.39e-03`, o `3.13e-03`, v `3.10e-03` — the
same ordering as the one-step update difference.

## 4. Is the 0.094 gap reproducible?

Three genuinely different training-window orders, each with a matched FP32 and
mixed run (same order within a pair):

| Order offset | Validation FP32 | Validation mixed | Validation gap | Test FP32 | Test mixed | Test gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.831961 | 2.926205 | **+0.094244** | 2.822131 | 2.929011 | **+0.106880** |
| 37 | 3.005721 | 2.979771 | **-0.025949** | 3.008919 | 2.994518 | **-0.014401** |
| 512 | 2.974943 | 3.111853 | **+0.136911** | 3.030603 | 3.197747 | **+0.167144** |

Mean paired gap: validation `+0.068402`, test `+0.086541`. Range: validation
`[-0.025949, +0.136911]`, test `[-0.014401, +0.167144]`. Mixed wins on one of
three orders.

With n=3 the receipt deliberately reports the full range and no confidence
interval. The Phase XVII +0.094 figure is reproduced exactly for its declared
order, but it is **not a stable precision penalty**: it is within the spread of
matched training-order variation and changes sign. A larger sweep would be
needed to estimate the mean gap precisely; this experiment does not claim one.

## 5. Defect assessment

Checked and negative:

- FP32 master weights remain authoritative; the receipt binds one initial
  parameter digest across all six runs.
- Every mixed step dispatches exactly 15 forward, 15 weight-gradient and 15
  input-gradient cooperative GEMMs, with zero shape fallbacks; FP32 runs
  dispatch zero.
- Weight mirrors convert on every epoch and are reused within it; a stale
  mirror would appear as a large forward error, but the same-weight RMS stays
  at 3e-04 to 5e-04.
- Parameters stay finite, the optimizer reaches step 100 in every run, and
  evaluation changes neither parameters nor gradient dispatch counts.
- The matched FP16/FP32-accumulation oracle and the Phase XVI 45-shader
  inventory are unchanged and still pass their checkers.

Not checked here: activation-level per-layer tracing. The receipt attributes
divergence to per-parameter projection groups, not to captured layer
activations, and states that limitation. Raw gradient tensors are also private;
the comparison uses the post-AdamW update.

Conclusion: **no numerical or implementation defect was found.** The evidence
supports ordinary FP16 operand rounding (RMS logit error ~4e-04) producing a
~5e-05 one-step weight difference, which a steep early loss surface and
AdamW's normalized update amplify into order-variable trajectory separation.

## 6. Remedy decision

No production change was adopted. The candidate remedy — selective FP32
execution for the query/key projections, which show the largest update error —
would trade cooperative-matrix throughput for a difference that is not a
consistent accuracy deficit (mixed is better on one of three orders) and whose
mean is smaller than the order-to-order spread. That fails the "reproducible
improvement without substantial throughput loss" bar. Mixed WMMA stays opt-in
and FP32 stays the default.

## Verification

```powershell
cargo build --release --bin wmma_drift
python validation/qualify_wmma_drift.py --install
python validation/check_wmma_drift.py
python -m unittest discover -s validation -p "test_wmma_drift.py"
```

The checker re-derives the partitions and rotated training streams, re-adds
every per-window evaluation loss, recomputes perplexities and order statistics,
and cross-binds each same-weight comparison to the exported checkpoint and the
run evaluation it must equal. It rejects the 26 forged controls in
`test_wmma_drift.py`, each for the rule it targets.

## Preserved qualification (re-run at this revision)

- `PHASE XVI CURRENT SOURCE: PASS (81f997a14194, 12 gradient tensors, 45 shader artifacts)`
- `TRAINING QUALIFICATION: PASS (8a12af2d907c, 8 arms, 3 resume arms, 8 performance arms)`
- `STABILITY QUALIFICATION: PASS (21027bf96997, lr=0.001, 4 arms, 3 resume arms, 2 performance arms)`
- `HELD-OUT QUALIFICATION: PASS (5cd0c0c6eb7f, 4 arms, test A=2.822131 D=2.929011)`

Historical receipts were not modified, and `v0.1.0-rc2` was not touched.

## Limitations

- The corpus is the synthetic three-family Phase XVII fixture; claims are scoped
  to it.
- Three training orders give a range, not a confidence interval.
- Parameter snapshots are weights-only; they are not optimizer-resume
  checkpoints, and AdamW moments are not directly observable.
- Layer-wise attribution is per parameter group, not per activation.
- Raw 67 MB parameter dumps and 75 MB logit tensors stay in `target/`; the
  receipt binds their SHA-256 and the compact statistics computed from them.
- CPU CI re-verifies committed evidence; it does not re-execute RDNA3 GPU work.
