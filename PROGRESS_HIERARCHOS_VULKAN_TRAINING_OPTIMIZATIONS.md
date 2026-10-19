

### Rank 3's batch-16 crossover, removed by a wider group

The rank-3 receipt recorded the two-dispatch chain winning batch 16 with the fused
kernel on the losing side. That is a kernel-shape result, not a merge result, and
the shape was the give-away: the fused kernel gave one 64-lane workgroup one
(batch, head), lane 0 ran the serial key normalization/scale while 63 lanes waited
at the barrier, and then only `head_size` (16 at width 128) of the 64 lanes
advanced state rows. Its cost per row is fixed per head, while the chain's
`key_transform_forward` spreads one head per lane and `state_forward` gives every
lane one of the `batch * width` rows. Two dispatches beat 1/4 lane utilization once
there are enough rows to amortize them; at batch 1 the extra dispatch cost still
won the other way, which is why the old table looked geometry-dependent.

`shaders/rwkv_key_state_forward_fused_wide.comp` packs `64 / head_size` heads into
the workgroup (four at this width). `lane / head_size` selects a head slot and
`lane % head_size` its row, so all 64 lanes advance rows and the grid shrinks by
the same factor (batch 1: 8 groups to 2; batch 16: 128 to 32). The transform phase
uses all 64 lanes as well: one lane per head publishes the normalization constant
in the standalone kernel's FMA order, then every lane writes one column of `kk` and
`scaled_k`, and stages the row loop's channel-invariant operands (`decay`, `kk * a`,
`r`) alongside them. At `head_size` 64 the mapping degenerates to the original
kernel's.

`bench_rwkv_key_state_forward.py` now measures four legs -- wide fused, narrow
fused, chain, default -- rotating through every sampling position in one process
(`sample_four_legs` in `src/bench_ab.rs`), with the arm label and the route's own
counter proving which kernel each leg dispatched:

| batch | wide | narrow | staged | narrow/wide | staged/narrow | staged/wide | default (arm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 36.9 us | 42.9 us | 51.5 us | 1.161 | 1.202 | 1.395 | 38.1 us (fused, wide) |
| 4 | 53.1 us | 58.4 us | 66.8 us | 1.100 | 1.144 | 1.259 | 53.7 us (fused, wide) |
| 8 | 71.3 us | 81.2 us | 82.5 us | 1.138 | 1.016 | 1.156 | 72.9 us (fused, wide) |
| 16 | 100.0 us | 119.8 us | 112.0 us | 1.199 | 0.935 | 1.120 | 101.6 us (fused, wide) |
| 32 | 161.7 us | 183.8 us | 183.0 us | 1.136 | 0.996 | 1.132 | 161.9 us (fused, wide) |

The narrow kernel still loses batch 16 to the chain (`0.943`, reproducing the old
`0.937-0.948`), and the wide kernel wins it (`1.145`). The crossover is removed, not
moved, and every measured geometry now names the wide route -- batch 1 and 4
included once the state phase stopped recomputing its channel-invariant operands.
All three graphs stay bit-identical (`worst_max_abs` and
`worst_max_abs_wide_vs_narrow` exactly 0), so the widening is scheduling and the
receipt asserts that rather than a tolerance.

### The records are now keyed by parameter-storage policy

Rank 2's FP16 scope gap was a record gap: the deep packed-mirror producer crosses
over at batch 8 while the dense route keeps the fusion there, and the table had no
dimension to hold that verdict, so the receipt had to report `diverges` at 8/16/32
rather than ship the win. `src/decision_records.rs` now loads an optional
`parameter_storage` on every entry: a policy-scoped entry wins for a consumer
running that policy, a policy-less entry is what every other policy falls back to,
and the two are replaced and withdrawn independently, so recording FP16 wins cannot
delete or rewrite an FP32 verdict.

`bench_rwkv_low_rank_fp16_forward.py --record-decisions` writes its verdicts under
`fp16-packed-mirror`, tagged with the driver that produced them (the file-level
`measured_by` stays the dense driver's): the FP16 route takes the chain at 8/16/32
(`0.977`/`0.971`/`0.969`) and the fusion at 1/4, and `route_following` now says
`agrees` at every geometry. The dense route is untouched -- batch 16 still has no
policy-less entry, batch 4 is still its fused win -- and
`src/rwkv_low_rank.rs` reads the scoped table through `compose_scoped` wherever it
consults the route.

### Verification

* `shaders/rwkv_key_state_forward_fused_wide.spv` compiled with the pinned shader
  toolchain and `spirv-val`-clean; `validation/verify_training_shaders.py` passes on
  464 variants, so the new artifact reproduces byte-for-byte from its source.
* `cargo check --lib --bins` clean; `cargo test --release --lib` filtered to
  `decision_records` (13 tests) and `bench_ab` (10 tests) green, including the new
  arm-lookup, arm-validation and policy-scoping tests.
* `bench_rwkv_key_state_forward.py --record --record-decisions` over batches
  1/4/8/16/32: receipt `validation/key_state_forward_bench.json` (`result: pass`,
  promotion `geometry-aware`), wide entries at 1/4/8/16/32, and every leg bit-identical
  (`worst_max_abs` and `worst_max_abs_wide_vs_narrow` exactly 0); re-running with the
  record in place dispatches the wide arm at every batch, including batch 32, which
  the earlier record staged.
* `bench_rwkv_low_rank_fp16_forward.py --record --record-decisions` over the same
  batches: receipt `validation/low_rank_fp16_forward_bench.json` with
  `follows_recorded_route_everywhere: true`, and the policy-less entries in
  `low_rank_full_forward_decisions.json` unchanged.
* `test_rwkv_key_state_forward_receipt.py` (10 tests, live re-run skipped without
  `HIERARCHOS_RUN_GPU_VALIDATION=1`) and `test_rwkv_low_rank_forward_receipt.py`
  (15 tests) green, plus the time-mix, channel-mix, composition, topology,
  training-state, training-stack and shader-provenance suites green.

### The transform phase joined the state phase on all 64 lanes

The wider group fixed which lanes advance state rows, but its transform phase still
gave one lane per head the whole serial normalization, so every group paid a fixed
transform cost on `slot_count` lanes. The shipped kernel now computes each head's
normalization constant on that head's row-0 lane (the standalone kernel's FMA order,
unchanged), publishes it through shared memory, and lets all 64 lanes write one
column each of `kk` and `scaled_k`. A first attempt that recomputed the norm in every
row lane -- one fewer barrier, four times the transform loads and FMAs -- was also
measured and rejected: 23% slower at batch 32 once the launch is throughput-bound.

**Then the row loop's own redundancy showed up.** Measured staged/wide on the Intel
Gen9 part: 1.183 / 1.174 / 1.153 / 1.139 at batches 1/4/8/16 (against 1.101 / 1.105 /
1.075 / 1.047 before), and 0.808 at batch 32. The
batch-16 crossover was gone by 14% and the narrow arm no longer won any measured
geometry, but batch 32 still lost by 19%: the profile put the whole difference in the
state phase (50.4 ns per row against the chain state kernel's 27.0 ns with the same
rows and lanes). The state loop was recomputing its channel-invariant operands per
row -- `exp(-exp(w))`, `kk * a` and `r` -- which is 16 times the transcendentals and
three extra global loads per row. Those are now staged once per channel by the
transform phase and published through shared memory, leaving 132 bytes of global
traffic per row instead of 320. Batch 32 went 232.1 -> 161.7 us (-30%, 50.4 -> 30.3
ns per row), staged/wide became 1.395 / 1.259 / 1.156 / 1.120 / 1.132 across
1/4/8/16/32, and the record now names the wide route at every measured geometry. All
three graphs stay bit-identical, including wide against narrow: the staged operands
are the same expressions on the same inputs, evaluated once.

### Phase attribution from the device's own timestamps

`validation/profile_rwkv_key_state_forward.py` runs each leg with kernel timestamps
on and one dispatch per submission, so each number is GPU time for that kernel:
`validation/key_state_forward_profile.json`. The chain's transform phase is nearly
flat in batch (14.8 -> 21.6 us from batch 1 to 32) while its state phase grows with
the row count (17.2 -> 110.7 us), taking the state share from 54% to 84%. That is the
original crossover explanation as a measurement. The fused kernel has no phase seam,
so the profile also reports it against the chain state phase -- same rows, same
64-lane occupancy: the excess is flat at +13.3 / +13.5 / +12.4 us at batches 1/8/16
(its transform plus one launch) and +96.9 us at batch 32. Through batch 16 the fused
kernel is the chain's state phase plus a constant; the batch-32 gap is the state
phase's schedule, not the transform the all-lane pass rewrote. Vendor EU counters are
not reachable from a plain Vulkan process here, so the report carries GPU
microseconds per phase plus the exact active-lane count per phase rather than a
utilisation percentage.

### One route per geometry, across every record

`decision_records.rs` is schema 2: an entry names one `Route` -- `fused`, `staged`
for the two-way producers, and `wide-fused`, `narrow-fused`, `staged` for the
key/state stage -- with every candidate's measured median in `candidates`. The loader
rejects a route that is not its own fastest candidate, and the policy scope now
covers the route (arm included) rather than only a boolean. That removed the
`compose` + `fused_arm` pair, which could disagree, and gave the key/state record the
same one-field shape as the others. `ab_harness.record_decisions` writes the new
shape, canonicalizes `fp16-storage-*` / `fp16-packed-mirror` into one storage scope,
and records the default route winning as a measured confirmation rather than
discarding it as a tie.

### Verification for this round

* `cargo check --lib --bins` exit 0; `cargo fmt --check` clean.
* `cargo test --release --lib` green (790 passed, 9 ignored), including the
  route-vocabulary, candidates, policy-scope and three-way key/state tests in
  `decision_records` (16) and the `bench_ab` suite.
* `verify_training_shaders.py` recompiles the whole manifest and reproduces the new
  `rwkv_key_state_forward_fused_wide.spv` byte-for-byte after the state-operand
  staging; the provenance, push-constant and binding-access checkers pass too.
* Receipt suites green: key/state, key/state profile, time-mix, channel-mix,
  low-rank + FP16, plus the composition, topology, training state and training
  stack suites (197 tests, 9 GPU-gated skips).
* `bench_rwkv_key_state_forward.py` and `profile_rwkv_key_state_forward.py` receipts
  re-recorded on the attached Gen9 part after both kernel changes, and the decision
  record written from the same run as the receipt (so the drift check compares one
  measurement, not two); `bench_rwkv_low_rank_fp16_forward.py` re-recorded with its
  1% band and a 40-iteration sweep.
* AMD: not measured. No AMD adapter is attached to this machine; the
  `cross-adapter-ab` job and the recording runbook in `AMD_REGRESSION_AUDIT.md`
  section 10 are the handoff, and no AMD entries were invented.

### Final feature set (closed)

These five are the round's final features; none is left as a follow-up:

1. **Phase profile with an explicit counter limit.** `profile_rwkv_key_state_forward.py`
   attributes the key/state stage with the device's own per-dispatch timestamps and
   `key_state_forward_profile.json` records, per batch, both fused arms, both chain
   kernels, the state share, and the per-row cost. The lane-utilization explanation is
   confirmed where it can be measured (the row-scaled state phase is 54% -> 84% of the
   chain and grows while the transform stays flat), and the vendor EU counters are
   reported as unreachable from a plain Vulkan process rather than estimated.
2. **Parameter-storage policy on every route record.** Time-mix projection, channel-mix,
   key/state and low-rank all resolve their table through `route_scoped` with their own
   policy; policy-less entries stay the fallback, and `fp16-storage-*` /
   `fp16-packed-mirror` canonicalize to one verdict. Reading is wired everywhere; a
   scoped entry is written only by a driver whose graph runs that storage (the FP16
   low-rank driver today), so the FP32 benches cannot launder their numbers into an
   FP16 verdict.
3. **Three candidates in one decision.** The key/state record names `wide-fused`,
   `narrow-fused` or `staged` in a single `route` beside every candidate's median; the
   `compose` flag and `fused_arm` field are gone and the loader rejects a route that is
   not its own fastest candidate.
4. **All-lane transform phase, re-measured.** The wider group's key transform keeps the
   standalone kernel's serial FMA order, publishes the constant through shared memory,
   and then writes all 64 lanes; the graphs remain bit-identical and the batch-16
   crossover is gone.
5. **State-phase operands staged once per channel.** `exp(-exp(w))`, `kk * a` and `r`
   are loop-invariant across a head's rows, so the transform phase publishes them in
   shared memory instead of the row loop recomputing them 16 times. That removed the
   batch-32 collapse (-29% there, +96.9 us excess over the chain state phase down to
   +16.3) and made the wide route the winner at every measured batch, 1 through 32,
   with staged/wide at 1.395/1.259/1.156/1.120/1.132.

Cross-validation status: the Intel Gen9 part is measured end to end. No AMD adapter is
attached to this machine, so the AMD half is the `cross-adapter-ab` CI job plus the
recording runbook in `AMD_REGRESSION_AUDIT.md` section 10 -- not claimed here.
