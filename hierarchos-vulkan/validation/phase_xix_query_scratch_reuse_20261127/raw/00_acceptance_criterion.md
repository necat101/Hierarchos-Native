# Phase XIX acceptance criterion — predeclared, before any candidate measurement

Written before any candidate arm was built into a test binary, before any
parity dispatch, and before any timing process. It is **not** edited after the
numbers are known; section 6 of the README restates it verbatim and section 9
applies it.

## Comparison

* **A** = Phase XVIII promoted query pass 1,
  `shaders/transformer_attention_backward_query_rdna3_dqunroll.spv`
  (`f0214b60…`), through its production selector and device gate.
* **B** = the same promoted shader plus the Candidate A scratch reuse,
  `validation/phase_xix_query_scratch_reuse_20261127/shaders/q_scratchreuse.spv`
  (`a6b3d6c6…`), reached through the same production selector, device gate,
  dispatch counter, grid, bindings and push block. The only difference between
  the arms is the SPIR-V the promoted loader builds.

No other optimization is part of this experiment.

## Primary fixture

8q/2kv, sequence 64, `target/gqa-e2e/fixture-8q2kv`, batch 2, LoRA rank 8 /
alpha 16, lr `1e-2`, seed 7. Measured independently in FP32 and in Mixed, exactly
as in Phase XVIII.

## Protocol (unchanged from Phase XII/XIII/XVII/XVIII)

Per mode: 5 ABBA rounds × 4 separate OS processes, schedule `A B B A`, 3 warmup
+ 26 measured complete steps per process, no profiler, no parameter readback
(`HIERARCHOS_PHASE_XIX_STEP_DIGEST=0`). `A` = median of the two arm-A process
medians, `B` = median of the two arm-B process medians, ratio `B/A`; the
confidence interval is the two-sided t interval (df = 4, `t = 2.776`) over the
five round ratios. Complete-step time, not isolated kernel time.

## Gate

**GO** requires all four, in both FP32 and Mixed on the primary fixture:

1. **Beyond noise.** The 95 % CI of the five round ratios lies entirely below
   1.0 (`ci_hi < 1.0`).
2. **Not one lucky round.** At least 4 of the 5 rounds improve (`ratio < 1.0`).
3. **Above the measurement floor.** The median complete-step ratio is
   `≤ 0.990`, i.e. a ≥ 1.0 % complete-step improvement. The floor is set from
   the established controls rather than chosen to fit a result: Phase XVIII's
   sham control bounded the `A B B A` schedule's own bias at 0.1 %, and its
   round ratios repeated to within 0.3 %.
4. **No credible regression elsewhere.** Each of the three regression series
   (8q/8kv MHA sequence 64, 8q/8kv MHA sequence 128, 4q/2kv sequence 64) in each
   mode must be reported, and none may show a 95 % CI whose lower bound exceeds
   1.0 (`ci_lo > 1.0` is a credible regression and blocks GO).

**NO-GO** if any gate above fails. The Phase XVIII promoted implementation is
then retained unchanged.

**BLOCKED** if the primary fixture cannot be measured under this protocol, or if
the candidate cannot be built or dispatched.

## Correctness precondition (not negotiable, checked before timing)

Every compared tensor must be bitwise equal between A and B, with zero
non-finite values, across attention mode 0 and mode 9, causal and non-causal,
masked and ordinary inputs, GQA and MHA, sequence 64 and 128 plus the
non-divisible sequence 37. In-graph, `loss_bits`, the LoRA gradient digest and the
parameter digest must be identical on every step. Any divergence fails the
candidate regardless of timing.

## Not a gate

* Isolated pass-1 kernel timing (`HIERARCHOS_PHASE_XIX_KERNEL_TIMING=1`) and the
  profiler attribution describe the mechanism. A faster isolated kernel is not
  sufficient, per the brief, and is not sufficient here either.
