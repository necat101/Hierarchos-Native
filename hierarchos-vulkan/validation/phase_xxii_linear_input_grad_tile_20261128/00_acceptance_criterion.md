# Phase XXII — predeclared acceptance rule

**Written before the candidate was measured.** This file is the GO/NO-GO rule
for the single Phase XXII experiment. It is not edited after the results.

## Candidate and control

* **Control (A):** the shipped production FP32 `linear_input_grad` kernel,
  `shaders/linear_input_grad.spv`, dispatched at the existing 16 x 16 grid.
* **Candidate (B):** the isolated 32 x 32 output-tiled kernel,
  `linear_input_grad_output_tile32.spv`, with four outputs per thread and the
  same ascending-`k` single-`fma` reduction per output.
* **Primary operation:** the verified 128 x 1024 x 1024 (rows x input_dim x
  output_dim) projection on the 8q/2kv, batch 2, sequence 64 fixture.
* **Arm selection:** `HIERARCHOS_PHASE_XXII_LINEAR_INPUT_GRAD_TILE=1` for B,
  unset for A. Test builds only; the production default never reads it.

## Required checks

1. **Bitwise parity.** Every output word of the candidate equals the control
   word on the parity matrix: 128x1024x1024, 128x1024x256, 64x128x128,
   33x37x45, 5x96x160 and 1x1x1. Zero sentinel survivors, zero non-finite
   values.
2. **End-to-end parity.** The 29-step complete-step driver's
   `parameter_end_digest` is identical in arms A and B (`d79a12d8…9c4655`),
   and the producer counters report the same routes in both arms.
3. **Protocol.** Five rounds of `A B B A`, 20 separate release-mode processes,
   3 warmup + 26 measured steps per process, no profiler, no per-step digest,
   same device and driver.
4. **Performance gate.** Reuse the Phase XIX predeclared gate, unchanged:
   * the 95 % CI of the five round ratios is entirely below 1.0;
   * at least 4 of 5 rounds improve;
   * the median complete-step ratio is <= 0.990 (at least 1.0 % gain);
   * every process exits 0 and no arm loses its route.

## Decision

**GO** only if all four checks pass. **NO-GO** on any failure. There is no
second candidate and no post-hoc threshold. A NO-GO experiment is excluded from
v0.2.8 without delaying the release.
