

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
the workgroup (four at this width). Lanes 0..3 each transform one head with the
same serial order, then `lane / head_size` selects a head slot and
`lane % head_size` its row, so all 64 lanes advance rows and the grid shrinks by
the same factor (batch 1: 8 groups to 2; batch 16: 128 to 32). At `head_size` 64 the
mapping degenerates to the original kernel's.

`bench_rwkv_key_state_forward.py` now measures four legs -- wide fused, narrow
fused, chain, default -- rotating through every sampling position in one process
(`sample_four_legs` in `src/bench_ab.rs`), with the arm label and the route's own
counter proving which kernel each leg dispatched:

| batch | wide | narrow | staged | narrow/wide | staged/narrow | staged/wide | default (arm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 55.2 us | 52.3 us | 60.8 us | 0.947 | 1.162 | 1.101 | 53.6 us (fused, narrow) |
| 4 | 58.7 us | 57.0 us | 64.9 us | 0.970 | 1.139 | 1.105 | 57.3 us (fused, narrow) |
| 8 | 77.2 us | 83.1 us | 83.0 us | 1.076 | 1.000 | 1.075 | 78.6 us (fused, wide) |
| 16 | 97.8 us | 108.6 us | 102.4 us | 1.111 | 0.943 | 1.047 | 97.0 us (fused, wide) |
| 32 | 186.7 us | 186.5 us | 186.1 us | 0.999 | 0.998 | 0.997 | 188.0 us (fused, wide) |

The narrow kernel still loses batch 16 to the chain (`0.943`, reproducing the old
`0.937-0.948`), and the wide kernel wins it (`1.047`). The crossover is removed, not
moved. Batches 1 and 4 are the other way: the wider group launches too few
workgroups to spread over the 24 EUs while the narrow kernel's finer grid still
fills them, so the record names `fused_arm: narrow` there and `wide` from batch 8
up -- a per-geometry measurement rather than a batch threshold that would match
this adapter and be wrong on another. All three graphs stay bit-identical
(`worst_max_abs` and `worst_max_abs_wide_vs_narrow` exactly 0), so the widening is
scheduling and the receipt asserts that rather than a tolerance.

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
  promotion `geometry-aware`), four fused entries with their arms, batch 32 in the
  band; re-running with the record in place dispatches the narrow arm at 1/4 and the
  wide arm from 8 up, and the batch-16 entry now names the fusion rather than the
  chain.
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

Measured staged/wide on the Intel Gen9 part: 1.183 / 1.174 / 1.153 / 1.139 / 0.808 at
batches 1/4/8/16/32 (against 1.101 / 1.105 / 1.075 / 1.047 / 0.997 before). The
batch-16 crossover is gone by 14%, and the narrow arm no longer wins any measured
geometry -- but batch 32 does not close: the wide grid's cost per row jumps from
32 ns at batch 16 to 51 ns, the chain wins by 19%, and the cause is not established
(register pressure from the transform loop is the remaining candidate). The record
runs the chain there, so the shipped default is the faster graph at every measured
geometry. All three graphs stay bit-identical, including wide against narrow.

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
* `cargo test --release --lib decision_records` green, including the new
  route-vocabulary, candidates, policy-scope and three-way key/state tests.
* `cargo test --release --lib bench_ab` green.
* `verify_training_shaders.py` recompiles the whole manifest and reproduces the new
  `rwkv_key_state_forward_fused_wide.spv` byte-for-byte.
* Receipt suites green: key/state (10), key/state profile (4), time-mix (7),
  channel-mix (9), low-rank + FP16 (15), plus the composition, topology, training
  state and training stack suites.
* `bench_rwkv_key_state_forward.py` and `profile_rwkv_key_state_forward.py` receipts
  re-recorded on the attached Gen9 part; `bench_rwkv_low_rank_fp16_forward.py`
  re-recorded with its 1% band and a 40-iteration sweep.
* AMD: not measured. No AMD adapter is attached to this machine; the
  `cross-adapter-ab` job and the recording runbook in `AMD_REGRESSION_AUDIT.md`
  section 10 are the handoff, and no AMD entries were invented.

### Final feature set (closed)

These four are the round's final features; none is left as a follow-up:

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
   crossover is gone (wide wins 1/4/8/16, the chain keeps batch 32).

Cross-validation status: the Intel Gen9 part is measured end to end. No AMD adapter is
attached to this machine, so the AMD half is the `cross-adapter-ab` CI job plus the
recording runbook in `AMD_REGRESSION_AUDIT.md` section 10 -- not claimed here.
