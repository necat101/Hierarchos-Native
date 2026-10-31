# Verified Intel handoff evidence

**Historical evidence:** this archive covers the earlier Intel revision named
below. The current V4.1 Flash / streaming handoff has RDNA3 validation for those
scopes only and still requires its own Gen9 run. See
[current validation status](../../../VALIDATION_STATUS.md).

All 13 qualification stages passed on Intel HD Graphics 520 (driver 31.0.101.2115).
This evidence was recorded before committing the already verified working tree.
The candidate source fingerprint is `88f0e4883f1aa825a3bb2c9e7d64ceddb769ae095d066294549d9e581bf4abb2`.
The numerical oracle is commit `2e7893d4e3db17370304a8d67adacab963b22bbc`,
a Windows build-only adapter on stable `b519361`. The original stable checkout
was left clean. No numerical goldens were re-recorded and no tolerances widened.

- 75 registered Transformer comparisons: 36,096 FP32 values, zero differing bits.
- 8,067 exact recurrent tensor digests: zero differences, both storage policies,
  27 updates and 162 checkpoint-reloaded inference-logit vectors.
- Debug and release training ladders: 12/12 checks each, 32,162 exact trajectory
  digests per profile.
- Rust: 900 tests passed; Python: 345 tests run with 12 existing opt-in skips.
- Two-process TCP: parameter-server and allreduce exported identical bytes for
  all 290 model/optimizer tensors after two updates.

The user approved retaining the pre-existing OLMo Hybrid beta-scale correctness
repair as an explicit exception to old-stable parity. Its separate diagnostic
has 282/282 differing logits at beta scale 2 (maximum absolute difference
0.84882572); beta scale 1 gives zero differences. OLMo Hybrid is outside the
75-case registry. See the root ROG_ALLY_HANDOFF.md for the numerical contract.

Physical RDNA3 execution and two physical machines remain unverified.
Windows networking uses TCP; native Windows RDMA is not implemented.

`intel-results.zip` contains the original command logs, result JSONs, recurrent
snapshots and the small deterministic OLMo diagnostic. Its SHA-256 is
`30962cee93d314cec6568ec278c5ed398ea2cd4a8d273220e87562d7a14b5e71`. `validation_environment.txt` records the package versions
and exact clean Transformers source commit. Evidence is supplied as an archive
so recorded JSON files do not alter the source-fingerprint inputs.

Use the root ROG_ALLY_HANDOFF.md to clone both published branches and run the
strict qualification and AMD tuning gates on the Ally.
