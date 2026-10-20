



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

`fp16-storage` (it wrote `fp16-packed-mirror` at the time), tagged with the driver

that produced them (the file-level `measured_by` stays the dense driver's): the

first sweep put the chain ahead at 8/16/32 (`0.977`/`0.971`/`0.969`) and the

fusion at 1/4. *Superseded below: those wins are not bit-identical to the frozen

FP16 trajectory, so they are recorded and withheld rather than adopted; see "A

measured route can be recorded and withheld".* The dense route is untouched -- batch

16 still has no policy-less entry, batch 4 is still its fused win -- and

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



### Level footing: the same operand staging on the chain and the narrow kernel



The batch-32 fix staged the row loop's channel-invariant operands in the wide

kernel only, which left the A/B comparing one staged row loop against two un-staged

ones. Both baselines were given the same treatment before shipping that as a fair

comparison. `rwkv_matrix_state_forward.comp` staged `kk`, `scaled_k`, `decay`,

`kk * a` and `r` once per head slot through shared memory, guarded on `head_size <=

64` with `64 % head_size == 0` (the only case where a whole head lies inside one

64-lane group, so `lane / head_size` is its slot); the narrow fused kernel staged

`decay`, `kk * a` and `r` on the row lanes -- one channel each -- rather than

lengthening the serial lane-0 transform prefix it already pays for.



Both measured **slower**, reproducibly: at batch 32 the chain went `183.0 ->

192-196 us` and the narrow kernel `183.8 -> 190-196 us` over four runs in the same

four-leg rotation, against a wide arm that stayed at `162-166 us` across all of

them. A third build removed the chain's unaligned fallback branch entirely, to ask

whether the dead second loop body was the cost rather than the staging: `192.3 us`,

the same regression. The reconciliation is what the two kernel shapes pay for the

same redundancy -- in a lane-per-row kernel the five operand vectors are four cache

lines per group broadcast to the sixteen lanes of a head, and the exponential pairs

hide behind the state row's own memory traffic, so there is nothing to recover;

what the staging adds instead is a shared-memory pass and an activation-dependent

barrier. The wide kernel's 30% was its own pressure, and the hoist does not

transfer.



The chain and the narrow kernel therefore keep their un-staged row loops -- each

arm runs the form it measured fastest, which is the equal-footing comparison the

numbers are for -- and the shipped tree is byte-identical to the commit that

recorded the receipt. The re-run reproduces it: wide/narrow/chain at batch 32 came

back `163.8/187.5/185.8 us` against the receipt's `161.7/183.8/183.0`, all three

arms bit-identical (`worst_max_abs` and `worst_max_abs_wide_vs_narrow` exactly 0) at

every batch.



### Every measured route is guarded across its whole sweep



`validation/check_measured_producer_routes.py` re-checks every measured producer's

receipt against the record it was written into, so a future kernel or driver change

cannot quietly move a route. One table covers the time-mix projection, the channel

mix, the dense and FP16-storage LN1/low-rank routes and the key/state fusion; each

spec names the receipt, the record, the batch sweep, the candidates, the fallback,

the parity contract and the dispatch accounting. It fails when the sweep stops

covering a batch, when the record carries an entry for a geometry the receipt never

measured, when an entry was written by another driver, when a recorded route is not

the fastest candidate of its own entry, when a non-default winner sits inside the

driver's noise band (2%, or the FP16 route's 1%), when the receipt now measures a

different route outside that band, when a leg's label or dispatch count contradicts

the graph it claims, or when the default stops building the recorded route or its

documented fallback. The key/state spec additionally checks the narrow arm, the

wide-versus-narrow parity and the arm the default dispatched. An adapter with no

entries at all is checked on its receipt alone, because the fallback is still

deciding there.



The two packed TBPTT chains are producers too, not a note beside the projection

records. `RwkvTbpttSequenceOp` now reads `tbptt_packed_forward_decisions.json` and

`tbptt_packed_rematerialization_decisions.json` per batch and applies a

`packed`/`legacy` route unless a benchmark arm pinned the switch; each chain has a

bit-exact receipt over 1/4/8/16/32 and a guard spec. On the Gen9 part neither

packed chain wins outside the band: the first forward is a tie through batch 8 and

the legacy transition leads by 3.7%/5.6% at 16/32, while packed rematerialization

loses from batch 4 on (6.5%/11.8%/15.7% at 8/16/32), so the records withdraw both

where the legacy graph measured faster. The multi-step TBPTT projection receipt was

re-recorded against that chain-routed default, so its batch-16 row now reports the

legacy first-forward chain. Its composed leg therefore dispatches the projection

twice per timestep -- once in the forward pass and once when rematerialization

replays it -- while the packed-forward rows still dispatch once. The receipt and the

guard derive that count from the chain route each row reports rather than assuming

`steps`, so a receipt left over from the old packed-by-default build fails instead

of quietly timing a different graph. The chain tables govern the default FP32

parameter-storage policy only: the FP16-storage trajectory golden is pinned to the

structural packed-forward route, and the legacy replay's FP16 rounding does not

reproduce it bit-for-bit, so that policy keeps the default until its own chain A/B

is recorded. The full training ladder caught this at batch 32 before the push;

scoping the tables to FP32 restored the 4,088 exact FP16 digests without changing

any FP32 decision.



The table also covers the two receipts that consume a record instead of writing

one: the multi-step TBPTT receipt is re-checked against the time-mix projection's

verdict at each batch it dispatched, and the packed-cell composition receipt must

agree that the route it qualified is the `staged` one the time-mix and channel-mix

records name. A consumer measures fewer geometries than the producer owns, so a

record entry at a batch it skipped is the producer's evidence, not a stray entry

(the `consumer` flag); the TBPTT sweep stopping at 16 no longer misreads the

record's batch-32 verdict as unsupported.



It runs in `workspace-check` on all nine committed receipt/record pairs (no GPU

needed) and in the A/B job immediately after each producer's receipt is

re-recorded, which is what makes it a check on live measurements. The workflow's

per-producer steps, receipt-contract test matrix and artifact paths are all

rendered from the same spec table by `validation/render_ab_workflow.py`, so a new

producer cannot miss its sweep, its guard step, its test module or its uploaded

receipt; `workspace-check` runs the renderer's `--check`. Every

job sweep now reaches every recorded geometry: the projection, channel-mix,

low-rank and FP16 steps gained `--batch 32` the same way the key/state sweep did,

so the geometry where the key/state collapse lived can no longer sit in the record

unmeasured. `validation/test_measured_producer_routes.py` (37 tests) proves each

rejection on every spec -- including that a measured chain win passes once the

record names the chain, that a narrow-arm win cannot hide behind a staged/wide

tie, that the FP16 receipt reads only its own policy scope, that every schema-2

route record in `validation/` has a spec, and that a consumer is not failed for a

batch it did not measure. `validation/test_ab_workflow_rendering.py` (10 tests)

pins the rendered sweep, guard step, test matrix, artifact paths and committed

workflow, and `validation/test_rwkv_tbptt_packed_chains_receipt.py` holds each

chain's bit-exact receipt to its record.



### A measured route can be recorded and withheld



The FP16-storage low-rank A/B is the case where winning the timing was not

enough to change the shipped graph. The deep FP16 producer measured the staged

chain ahead at batches 16/32 (batch 8 landed inside the route's 1% band and is a

tie), but the two graphs differ by `2.384e-07` there:

small for a forward comparison, fatal for a training trajectory that is checked

bit-exactly. Adopting the measured route diverged the FP16 trajectory at batch32

from step 2 on, while the FP32 trajectory and the FP32 route were untouched. The

record therefore gained a top-level `withheld` list beside `decisions`: the

entry carries the measurement and the staged win, the withheld entry names the

same route and the reason it may not be adopted, and `DecisionTable` skips it at

lookup for that storage policy. A withheld scoped route does not fall back to

the shared FP32 verdict either -- the reason applies to the graph change, not to

the other policy's measurement -- so the runtime keeps the structural fused

graph. The guard checks the withheld entry against the measured decision (the

route must be its winner, the reason must be non-empty) and requires the default

to follow the structural fallback at those geometries instead of the entry.



The refusal is tied to the measurement, so a rerun can lift it without an edit.

The FP16 driver withholds a staged win only when the graphs differ

(`withheld_reason` returns `None` at `worst_max_abs == 0`), and the record merge

drops every older `withheld` entry for a geometry it re-measured; a bit-identical

staged rerun therefore adopts the route and reports the adoption, while a rerun

that still shows rounding keeps the refusal. `WithheldRouteAdoptionTest` pins

both halves of that rule against `record_decisions`.



### The TBPTT FP16 refusal was lifted by its own A/B



The two packed-chain records can only choose between bit-identical graphs, so a

withheld chain is not a parity claim: it records that one parameter-storage

policy may not adopt the measured verdict because that policy's trajectory

golden is pinned to the structural graph. The FP16-storage policy was that case.

The chain A/B is now runnable under it: the bench takes

`--parameter-storage fp16-storage`, installs the cell's packed FP16 execution

mirrors on all three legs before the pins are applied, refuses to report if the

mirrors did not activate, and stamps the receipt with the policy. The driver

writes a second receipt per chain and the record entries under that policy's

scope, and `RwkvTbpttSequenceOp` reads both tables through `route_scoped` with

the active policy.



That measurement is what lifted the refusal. The FP16 forward A/B is still

bit-identical -- `worst_max_abs` is exactly zero over all five returned tensors

at every batch -- and repeats the FP32 crossover: `legacy` leads by 3.8%/5.3% at

batches 16/32 (40.855 vs 42.475 ms, 63.087 vs 66.593 ms), while 1, 4 and 8 sit

inside the 2% band. The record therefore adopts its own measured `legacy`

verdicts at 16 and 32 under `fp16-storage` and drops the `withheld` entries

that held the structural `packed` route there. Rematerialization's FP16

receipt agrees with the FP32 one: `packed` loses from batch 8 (5.8%, 10.5%,

15.7% at 8/16/32) and 1/4 are inside the band, so every measured verdict is

the structural default and the table needs no scoped entry.



The refusal mechanism remains for the next policy whose trajectory is pinned to

the structural graph, and both halves stay guarded.

`ab.refuse_shared_verdicts` regenerates the list from the measured decisions on

every chain recording, so it cannot go stale or survive a verdict that returns

to the structural route. `check_measured_producer_routes.py` rejects a chain

refusal that names a route the policy would not read, carries no reason, is

unscoped, appears without a measured geometry, refuses one move away from the

structural route while leaving another unrefused, or sits beside the policy's

own measurement of the same geometry. Since neither shipped table needs a

refusal any more, the contract suite exercises those rules against a fixture

that withdraws the FP16 measurements, so the check stays honest without a

stale refusal in the record.



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

  low-rank + FP16, both packed TBPTT chains, plus the composition, topology,

  training state and training stack suites; the full validation run is 249 tests

  with 10 GPU-gated skips. The TBPTT receipt's dispatch assertion follows the

  chain route (`2 * steps` on the legacy first-forward transition), matching the

  re-recorded batch-16 row.

* `bench_rwkv_key_state_forward.py` and `profile_rwkv_key_state_forward.py` receipts

  re-recorded on the attached Gen9 part after both kernel changes, and the decision

  record written from the same run as the receipt (so the drift check compares one

  measurement, not two); `bench_rwkv_low_rank_fp16_forward.py` re-recorded with its

  own 1% band in one 20-iteration run that wrote both the receipt and the scoped

  record entries, so the receipt's route rows and the record's `withheld` list are

  one measurement. That run's receipt is the one the audit table quotes: batch 1/4

  fused (1.038/1.066), batch 8 a 0.991 tie inside the 1% band, batches 16/32

  staged by 2.6%/2.4% and withheld.

* The withheld route is guarded from both ends. `ab_harness.record_decisions`

  takes the band it gates on separately from the band the file declares, so the

  scoped 1% recording no longer relabels the dense route's 2% record; the FP16

  receipt test fails if a route row says `withheld` while the record does not (the

  stale-artifact case this pass found and fixed), and `check_measured_producer_routes`

  fails if a withheld entry is not the measured winner or if the default follows

  it. The chain refusal is record-driven and policy-scoped; the FP16 chain A/B

  now measures that policy and its own entries lift the old refusal at 16/32,

  with the contract exercised against a fixture that puts the refusal back.

  `cargo test --release --lib` is green at 795 passed / 9 ignored, and the

  validation suite is 263 tests with 10 GPU-gated skips.

* The equal-footing round: both baseline kernels were hoisted, measured slower

  (chain 183.0 -> 192-196 us, narrow 183.8 -> 190-196 us at batch 32, four runs,

  plus a no-fallback build at 192.3 us), and reverted, so the shipped shaders are

  unchanged and the receipt still describes this tree; the re-run confirms the route

  (wide/narrow/chain 163.8/187.5/185.8 us against the receipt's

  161.7/183.8/183.0).

* `check_measured_producer_routes.py` passes on all nine committed receipt/record

  pairs (seven producers including both TBPTT chains, plus the TBPTT and

  packed-composition consumers) and rejects each mutation it claims to

  (`test_measured_producer_routes.py`, 37 tests, green); it is wired into

  `workspace-check` and after every A/B step in both jobs, and every Intel sweep

  now reaches batch 32, so a record entry there can no longer sit unmeasured. The

  workflow's steps, test matrix and artifact paths all come from the same spec

  table (`render_ab_workflow.py`; 10 rendering tests, `--check` green), so adding a

  producer cannot silently skip its sweep, its guard, its tests or its receipt.

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





### A checked ladder receipt and a measured backward-fusion route (2026-10-19)







`validation/check_training_stack_receipt.py` now guards



`validation/training_stack_receipt.json` in `workspace-check`: the ladder's twelve



steps in order, every one `PASS`, the adapter the device-aware steps name, the



probe's loss, and a floor for every count the ladder reports (144/183/79 golden



invocations, 464 matched variants out of 471 modules, 6512 dispatches, 4 topology



decisions, 2 suites / 29 runs / 32162 exact digests). A dropped or renamed step,



a non-`PASS` row, a receipt re-recorded on another adapter, a changed loss or a



shrunk count all fail; `--compare` additionally refuses a fresh run below the



committed counts, and `--record` is the one deliberate way to move the baseline.



`test_training_stack.py` carries a negative control for each failure mode.







The TBPTT reverse pass's state-backward fusion depth is now the third chain on

`bench_rwkv_tbptt_packed_chains.py`. The driver times the autotuner's schedule

against the most decomposed executable chain, holding the projection order and

low-rank fan-in fixed, and refuses to record unless the two graphs are

bit-identical. The topology labels name stages rather than kernels: `+` still

dispatches, `-` is folded in, so `rkv-add3+key+reduce` is the fully staged chain

and compiles at every batch. So does the fully folded `rkv-add3-key-reduce`:

its batch-1 shortcut writes the parameter gradients directly because each

channel owns one row, and the guarded `FUSE_KEY_PARAM_CROSS_BATCH`

specialization extends that to every batch. A workgroup owns one (batch, head)

row, so the fold cannot be formed inside a single workgroup; each workgroup

stores its partials, publishes an arrival through an atomic per-head counter,

and the workgroup that completes a head's last batch reduces that head's

partials in ascending batch order -- the order

`rwkv_key_transform_param_reduce.comp` uses -- then clears the counter. The

specialization is compiled for the A/B, which pins it, and for a process whose

backward-fusion record can select `folded` at that adapter and width, so the

kernel set follows the record instead of preceding it. The folded rung is a

routing candidate, and its acceptance gate is the fold's own output: the

receipt reads the reduction back and requires it to equal the two-dispatch

chain exactly, and only a geometry that clears it lists `folded` among its

candidates. Its returned tensors are bounded separately, because the contract

associates one FP32 product differently and moves `grad_x` by at most `2.4e-7`

(a couple of ulps), so the receipt holds them to a `1e-6` allowance while the

routing pair stays exactly bit-identical. Measured with all three candidates,

the depths are a three-way tie on this part: in the committed receipts the

shipped schedule is fastest at six of the ten measured geometries, staged at

two and folded at two, and nothing leads by more than `0.4%`. The record

therefore keeps `fused` at batches 4 and 32, withdraws 1, 8 and 16 as ties

inside the band, and names `folded` nowhere -- the route is recordable, and

the measurement says there is nothing to gain by adopting it here. A temporary

record naming `folded` at batch 4 proved the promotion end to end: the default

built the folded rung and the receipt reported `default_route folded` with the

key-parameter readback at zero again. That check also found a real defect -- the

verdict is applied at the start of every sequence, and the first implementation

only searched for a rung strictly more fused than the current one, so the

second sequence failed closed; `folded_backward_schedule` now returns the

schedule already installed when it is the folded rung.



`grad_x` never consumes `grad_k_k`/`grad_k_a`, so the receipt now reads the

reduction's own output back after the sequence and requires it to match the

two-dispatch chain exactly. It does: the folded rung's `grad_k_k`/`grad_k_a` are

bit-identical at every batch under both storage policies, and the receipt

records the staged arm's gradient magnitude beside the zero so the agreement

cannot be a vacuous zero-vs-zero. A second gate runs one identical AdamW step

on each leg and diffs the parameter snapshots (at most `7.5e-8` against `1e-6`),

but the measurement showed that gate is much weaker than it looked: AdamW's

first update is sign-normalized, so a fold that loses a batch row changes a

gradient's magnitude without changing the update, and a corrupted arrival count

moved a parameter by `1.5e-8` -- inside the identical-input noise. The direct

readback is what covers the reduction.`--negative-control` proves the guards reject a broken fold instead of assuming
it, and it does so on both publication paths. Above batch 1 it seeds one phantom
arrival per batch row through a test-only hook
(`HIERARCHOS_RWKV_CROSS_BATCH_ARRIVAL_SEED`), so no arrival can ever complete a
head and the fold never reduces. At batch 1 there is no counter at all: the rung
owns one row per channel and writes the parameter gradients directly, so the
control redirects those two bindings into a scratch buffer
(`HIERARCHOS_RWKV_KEY_PARAM_DIRECT_WRITE_SINK`) and the fold publishes nothing.
Each run builds a clean pair and a corrupted pair in the same process and
measures both. Batch 1's clean pair is identical outright -- both gates read
`0.0`, the two graphs are the same computation there -- and its corrupted pair
fails at `8.1e-3` on the key-parameter gate and `1.0e-4` on the step gate. The clean pair passes

both gates (key-parameter delta exactly zero, post-step `1.5e-8`); the

corrupted pair fails both (key-parameter delta `9.4e-3`, which is the whole

gradient, and post-step `1.0e-4`). The driver re-applies its own receipt gate

functions to those four measurements, so what is proven is the guard the sweep

uses. A one-phantom-arrival seed was rejected as the corruption: it makes the

reduce fire one row early, but whether the missing row has already stored its

partial is a race, so the control uses the deterministic extreme.



The first sweep measured only batch 1 because `STAGED_STATE_BACKWARD_PRIORITY`

was ordered most-fused-first while its comment described the opposite. The

staged arm therefore searched for something *below* the shipped schedule and

found only the batch-1 kernel; batches 4/8/16/32 were correctly reported

`unmeasured`, but for the wrong reason. Ordering the table stage-first makes the

A/B dispatch the three-dispatch chain against the autotuner's two-dispatch one

at every batch, bit-identically (`worst_max_abs` exactly zero). Fused holds at

batches 1 and 4 (1.1%/0.5% over 40 interleaved samples); 8-32 are ties inside

the 2% band, so `tbptt_backward_fusion_decisions.json` records the fused verdict

at all five without moving the autotuner's schedule. Twenty samples had drifted

across the band between passes (fused +0.2%, staged +3.4%), which is why the

backward producers sample twice as long. The FP16 pass measures fused ahead at

1 and 4 and staged ahead by at most 0.6% at 8-32; all of it is inside the band,

so it keeps the shared verdict rather than writing an unstable scoped move.

`unmeasured` remains part of the guard's sweep contract, so a geometry cannot be

dropped from both lists even though every batch is measured on this adapter.



## Why the fully folded rung does not lead (profile, 2026-10-20)

`--profile-folded` was added to the packed-chains bench and driver. It times four
legs in one process, interleaved: the fully staged chain, the autotuner's `fused`
depth, the fully folded rung, and a folded rung whose batch-order reduction cannot
fire (a full batch of phantom arrivals removes only the last workgroup's reduce
loop; every arrival, partial store and device-scope barrier stays). Every leg is
gated on the fold's `grad_k_k`/`grad_k_a` readback before timing, and each delta
is reported beside the widest leg's interquartile range so a component below the
run's own resolution is labelled noise instead of a signed cost.

Two committed profiles (FP32 and fp16-storage, width 128, 8-step windows, 40
rotated samples, 3 warmup rounds) put the folded rung at 0.9963-1.0074 of `fused`
across batches 1-32. The reduction tail measured -0.23 to +0.18 ms and the
counters, credited against the reduce dispatch they replace, -0.12 to +0.09 ms --
both inside a 0.4-2.1 ms resolution, so neither component is the limit.
`validation/check_folded_publication_profile.py` re-derives every delta, ratio and
resolution from the recorded legs, requires the gates the legs cleared, and fails
when the fold starts leading or either component becomes visible; it runs in
`workspace-check`.

The ceiling is structural. Folding the reduction removes a launch and nothing
else: the staged reduce is `width` work-items doing `batch` dependent adds over
the same partials the fold writes and reads back, in the same order that makes it
bit-identical. At width 128 / batch 32 that is one wavefront of work behind a
memory-latency chain, so the fold trades a dispatch boundary for an in-dispatch
arrival barrier and the two come out equal. The `fused` rung still wins a little
where it absorbs the elementwise key transform (real work per element), which is
why the ladder is flat below it.
