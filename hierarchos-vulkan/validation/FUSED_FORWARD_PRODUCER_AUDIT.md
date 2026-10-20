# Fused forward producers in the RWKV training graph — audit and A/B ranking

Written after the time-mix projection A/B changed the rule from "fuse and keep it"
to "measure the fusion against its composition and promote per geometry". The
question this audit answers is which of the remaining fused forward producers
still deserve that treatment, and which are already decided.

Scope: producers the packed-cell → TBPTT → full-training-graph path dispatches on
the forward sweep. Backward fusions (`linear3_time_mix_backward_fused`,
`state_backward_fused_rkv_add3*`) and the transformer graph are out of scope; the
packed-cell A/B already showed that a forward win does not imply a
forward-backward win, so backward fusions need their own measurement rather than
a forward verdict. One of them now has that measurement: the TBPTT reverse pass's
state-backward fusion depth is a third chain in
`bench_rwkv_tbptt_packed_chains.py`, with its own receipt and route record. The
`linear3` backward fusion and the transformer graph remain unmeasured here.

## Already decided

| Producer | Status | Evidence |
| --- | --- | --- |
| `rwkv_time_mix3_linear3_key_state_forward_fused*` (time-mix projection + key + state) | **Promoted per geometry**, now from a per-device decision record | `packed_cell_time_mix_bench.json`, `PACKED_CELL_COMPOSITION.md`; composition loses where the tuned `linear3-forward` module runs, wins nowhere else |
| `packed_cell_channel_mix_state_forward_fused` (channel mix + packed-state write) | **Removed** | `PACKED_CELL_COMPOSITION.md`; lost the composition at every batch and the forward-only transition by 11% |
| `group_norm_bonus_gate_linear_residual_forward_fused_{compact_one_row,two_rows}` | **Not reachable at production width** | gated on `width <= 32` in `rwkv_post_mix.rs`; the audit does not rank a kernel the width-128 graph never builds |
| `layer_norm_low_rank_producer_forward_fused*` (six FP16 arms) | **Measured among themselves and against the staged chain**; the fusion stays | `topology_decisions.json`, `verify_topology_decisions.py` for the arm choice; `packed_cell_low_rank_forward_bench.json` and `low_rank_full_forward_decisions.json` for the route (rank 2 below); `low_rank_fp16_forward_bench.json` for the deep FP16 arm |
| `rwkv_key_state_forward_fused*` (key transform + state update) | **Measured: the wider-group variant wins every measured batch after two shape fixes — an all-lane transform phase and channel-invariant operands staged out of the row loop; the same staging measured slower on both baselines, so each arm runs its fastest form** | `key_state_forward_bench.json`, `key_state_forward_profile.json` and `key_state_forward_decisions.json` (rank 3 below) |

Three of the five are settled because the measurement went against the fusion
somewhere. That is the reason this audit exists at all: the default should not be
"fused wins".

## Remaining producers, ranked

### 1. The packed path's channel-mix producer — one existing dispatch replacing two

`RwkvChannelMixOp::record_forward` dispatches the single
`layer_norm_channel_mix_full_forward_fused` kernel (layer norm + interpolation +
`key_cm` + activation + `value_cm`) when fusion is enabled. The **packed cell does
not take that path**: `record_channel_mix_and_pack` calls
`record_forward_producer` (the 14-binding producer, one dispatch) and then the
vendor `linear-residual-forward` module for `value_cm` (a second dispatch). So on
the path the TBPTT trainer actually runs, the channel mix costs two dispatches
while the plain cell spends one on the same arithmetic.

Why it ranks first: the shader already exists and is already exercised and
golden-covered outside the packed path; the composition is already the packed
path's default, so the A/B needs a wire-up and a switch rather than a new kernel;
and the saving is per layer per timestep, which is one of the two multiplicities
that dominate dispatch overhead.

Caveat: the retired `packed_cell_channel_mix_state_forward_fused` also absorbed
the state write and lost. This candidate keeps the state write separate, which is
the difference worth measuring — but the loser's result is exactly why it must be
measured rather than assumed.

### 2. The LN1/low-rank full-forward producer — one dispatch replacing the staged chain — **measured: the fusion stays**

`low_rank_full_forward_fused_fp16_packed` keeps one row's w/a/g rank activations
resident through the activation, the second low-rank matmuls, decay and the final
gate. The staged alternative runs the layer norm, the interpolation, the three
first-stage matmuls and the second-stage matmuls as separate dispatches.

Why it ranked second and not first: the merge is deeper (the staged path is the
longest chain in the cell for this shape), but the fused arms had already been
autotuned among themselves per geometry, so a loss here looked unlikely and the
payoff is bounded by the fact that the chain's matmuls are rank-sized and cheap
per dispatch. It was measured anyway, because the time-mix result was also
"surprising" at small batch.

**Result — unlike ranks 1 and 3, the measurement went with the fusion.**
`validation/bench_rwkv_packed_cell_low_rank_forward.py` measured the two complete
graphs in one process on the Intel Gen9 part (three legs, 30 interleaved rotated
samples per leg, `--warmup 5`); the ratio is staged ÷ fused, so anything above
1.0 keeps the fusion:

| batch | fused | staged | default | staged/fused | route dispatches (fused/staged/default) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.641 ms | 2.722 ms | 2.625 ms (fused) | 1.031 | 1/12/1 |
| 4 | 3.233 ms | 3.342 ms | 3.253 ms (fused) | 1.034 | 1/12/1 |
| 8 | 3.871 ms | 3.871 ms | 3.858 ms (fused) | 1.000 | 1/12/1 |
| 16 | 5.717 ms | 5.827 ms | 5.766 ms (fused) | 1.019 | 1/12/1 |
| 32 | 9.921 ms | 9.982 ms | 9.871 ms (fused) | 1.006 | 1/12/1 |

The staged chain never wins: batches 1 and 4 are recorded fused wins (1.031x,
1.034x), and 8/16/32 measure inside the 2% noise band, so they carry no entry and
the structural default — the fusion — keeps deciding them. Every returned tensor
(the output, the packed state, `grad_x`, the packed-state adjoint and the
token-feature gradient) agrees within `1.6e-5` worst absolute. These graphs are
*not* bit-identical the way rank 1's are, and that is expected: the fused arm
normalizes inside the producer while the staged arm dispatches the standalone
layer-norm kernel, so the two reductions round differently. The receipt states the
tolerance instead of claiming exactness, and the test fails if the difference
leaves it.

The dispatch column is what makes the A/B non-vacuous: one producer dispatch
against twelve. The staged count is the layer norm, the interpolation, the three
first-stage matmuls, the tanh, and the six second-stage dispatches (two
matmul+bias, decay, two sigmoids, the gate matmul).

**What this receipt measured, precisely.** On the packed cell the low-rank
parameters are FP32, so the fused arm is the portable LN1/first-stage producer
(`fused_arm: "layernorm-low-rank-producer"` in the receipt). The deeper
`low_rank_full_forward_fused_fp16_packed` arm only exists under the training
graph's FP16 parameter-storage policy, which installs packed mirrors the packed
cell does not have, and its `HIERARCHOS_VULKAN_TRAINING_PRECISION` switch is read
by the training graph, not by this bench. What is measured here is therefore the
decision that ships on the dense route: one dispatch against the chain, per
geometry, with FP32 parameter storage. The FP16 arm was measured separately — see
"The FP16 scope gap" below — and `AMD_REGRESSION_AUDIT.md` §9 no longer lists it as
an unmeasured gap.

**The FP16 scope gap — measured: the same route where the record has an entry.**
`validation/bench_rwkv_low_rank_fp16_forward.py` drives the op the policy feeds
(`RwkvLowRankOp::forward_backward` on the pre-normalized route) after
`enable_fp16_parameter_storage()` quantizes its six FP32 masters into packed FP16
execution mirrors — the op-level equivalent of
`HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-storage-fp32-compute`. Both arms then
read the same packed operands; only the dispatch structure differs. The receipt
that ships (`low_rank_fp16_forward_bench.json`, `--iterations 20 --warmup 5`) and
the record entries below come from one run:

| batch | fused (deep FP16 arm) | staged | default | staged/fused | route dispatches (fused/staged/default) | route status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.449 ms | 1.504 ms | 1.448 ms (fused) | 1.038 | 1/11/1 | agrees |
| 4 | 1.285 ms | 1.369 ms | 1.297 ms (fused) | 1.066 | 1/11/1 | agrees |
| 8 | 1.323 ms | 1.312 ms | 1.326 ms (fused) | 0.991 | 1/11/1 | tie (inside 1%) |
| 16 | 1.447 ms | 1.410 ms | 1.459 ms (fused) | 0.974 | 1/11/1 | withheld |
| 32 | 1.726 ms | 1.685 ms | 1.718 ms (fused) | 0.976 | 1/11/1 | withheld |

The fused arm is
named and counted, not inferred: `fused_arm` is
`low-rank-full-forward-fused-fp16-packed` and a dedicated
`deep_forward_arm_dispatches` counter is 1 on the fused leg and 0 on the staged
one, because the route's total dispatch count alone cannot tell the deep producer
from the first-stage producer (both are one dispatch). The staged chain is 11
dispatches here against 12 on the packed-cell receipt: there is no separate layer
norm on the pre-normalized route. Every returned tensor agrees within
`2.4e-7` — the two graphs read the same FP16 mirrors and differ only by rounding,
so the receipt states the tolerance.

**The answer to the gap's question.** The FP16 arm's crossover is *earlier* than
the FP32 arm's: the chain is ahead at 16 and 32 (2.6% and 2.4%) where the dense
route keeps the fusion, fused wins at 1 and 4, the same geometries the dense
route names, and batch 8 sits inside this route's 1% band (0.991), so it is a tie
and carries no entry rather than a sign that would flip between sweeps. The first
sweeps reported those wins as divergences because the record had no
storage-policy dimension to hold them. It has one now: the record is keyed by
`parameter_storage` as well as by adapter, width and batch, and this driver
writes its verdicts under `fp16-storage` — the storage those training policies
select, an entry only a consumer running that storage policy reads (the older
`fp16-packed-mirror` spelling canonicalizes into the same scope).

The wins do not ship, because winning the clock is not the only condition. The
two FP16 graphs differ by up to `2.384e-07` — the same rounding band every sweep
reports — while the `fp16-storage-fp32-compute` training trajectory is held
bit-exact against its golden. Adopting the measured route was tried and broke
that trajectory at batch 32 from step 2 on (the FP32 trajectory and the FP32
route were untouched). So the entries stay in the record as measured evidence and
the same geometries are listed under `withheld`: the runtime skips them at
lookup, does not fall back to the dense FP32 verdict (the reason applies to the
graph change, not to the other policy's measurement), and keeps the structural
fused graph. `route_following` therefore reports `agrees` at 1 and 4, `tie` at 8,
and `withheld` at 16 and 32 — with the default leg dispatching the fusion at
every geometry, which is the proof the withheld route was not adopted. The
policy-less entries the dense route reads are untouched: batch 16 still has no
dense entry and batch 4 is still the dense arm's fused win (1.034x on the packed
cell, 1.104x here). Nothing is discarded: an exact FP16 route can be adopted by
deleting its `withheld` entry, and the guard fails if the record withholds a
route that was not the measured winner.

**The rule.** `prefers_fused_forward_chain` keeps the fusion wherever the producer
kernels exist, withdraws it per (adapter, width, batch) where
`validation/low_rank_full_forward_decisions.json` records a staged win that is
not `withheld`, and is
pinned by `HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN` /
`HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN` for the two A/B legs. The
decision also decides whether the cell fuses LN1 at all, because a staged route
that kept the layer norm inside would not be the chain that was measured. Unlike
the vendor-gated producers there is no gate in front of it: the producer is plain
Vulkan code, so "the kernels exist" is the whole availability question.

**Scope.** The record governs the dense cell forward. The TBPTT packed
forward-only and packed-rematerialization paths keep their own fused chains (they
need the LN fusion structurally) and are measured by the packed-cell composition
A/B, not here — the record's notes say so, so a staged entry can never be read as
a claim about a graph it did not measure.

### 3. The fused key/state producer — one dispatch replacing two — **measured: the wider group wins every measured batch, 1 through 32**

`record_key_state_forward` uses a fused key-transform/state-update kernel when
`head_size <= 64` and falls back to `key_transform_forward` + `state_forward`
otherwise. It had no switch and no record; both now exist
(`HIERARCHOS_RWKV_FORCE_FUSED_KEY_STATE_FORWARD` /
`HIERARCHOS_RWKV_FORCE_STAGED_KEY_STATE_FORWARD`, and
`validation/key_state_forward_decisions.json`), and the fused route now dispatches
the *wider-group* kernel (`rwkv_key_state_forward_fused_wide`) rather than the
original one-workgroup-per-head kernel it was first measured with.

Why it ranked last: the smallest merge, and the fallback exists structurally, so
the A/B was a switch plus a sweep. The expectation was that a win would be too
small to matter. That expectation was wrong at one geometry — and the reason it
was wrong turned out to be the kernel's own shape rather than the merge.

**Why the original fused kernel lost at batch 16.** The kernel gave one 64-lane
workgroup one (batch, head): lane 0 ran the serial key normalization/scale while
the other 63 lanes waited at the barrier, and then only `head_size` (16 at this
width) of the 64 lanes advanced matrix-state rows. Its cost per row is therefore
fixed per head. The chain's two kernels do not have that shape:
`key_transform_forward` gives each lane a whole head (64 heads in flight per
workgroup, no shared memory, no barrier), and `state_forward` gives each lane one
row of the `batch * width` rows, fully occupied. At batch 1 the fusion still won,
because two dispatches cost more than the wasted lanes; by batch 16 the row count
was large enough for the chain's dispatch overhead to amortize, and the fused
kernel's 1/4 lane utilization in the state phase was the whole difference. Five
sweeps measured 0.937–0.951 (chain ÷ narrow fused) at that geometry, reproducing
the recorded loss.

**The wider-group variant.** `rwkv_key_state_forward_fused_wide.comp` packs
`64 / head_size` heads (four at this width) into one 64-lane workgroup:
`lane / head_size` selects a head slot and `lane % head_size` a row, so a full
workgroup advances `slot_count * head_size` rows instead of 16. Workgroup count
drops by that factor too (batch 1: 8 groups become 2; batch 16: 128 become 32),
which is the trade the record measures. The per-row arithmetic is untouched -- the
same products in the same order -- and the receipt asserts it:
`worst_max_abs_wide_vs_narrow` is exactly zero at every geometry.

**The transform phase now uses all 64 lanes too.** The first wide variant left the
key-transform phase on `slot_count` lanes (one per head), which is why it still paid
a fixed serial cost per head. The shipped kernel computes each head's normalization
constant on that head's row-0 lane with exactly the standalone kernel's FMA order,
publishes it through shared memory, and then lets all 64 lanes write one column each
of `kk` and `scaled_k` -- one barrier more, one 16-step serial loop fewer. A variant
that recomputed the norm in every row lane instead (no extra barrier) was also
measured: it was faster at small batches but cost 23% at batch 32 (227 µs against
the chain's 184 µs), because the redundant transform is four times the loads and
FMAs once the launch is throughput-bound. The published-constant form is what
shipped.

**Why batch 32 collapsed, and the fix.** After the all-lane transform the wide
kernel still lost batch 32 by 19%, and the profile put the whole difference in its
state phase: 50.4 ns per row against the chain state kernel's 27.0 ns, with the
same rows, the same lanes and the same access pattern. The state phase was
recomputing its channel-invariant operands once per row -- `decay = exp(-exp(w))`,
`b = kk * a`, and the receptance read `r` -- although all three depend only on the
channel. That is `head_size` (16) times the transcendentals and three extra global
loads per row. At small batches the launch dominates and the redundancy hides; at
batch 32 the kernel is throughput-bound and the row loop's live operands and
re-issued transcendentals were what this kernel had no headroom for -- the same
redundancy costs the chain's leaner state kernel nothing (see the level-footing
measurement below), so what cost the 30% was the interaction with this kernel's
own pressure rather than the algebra on its own. The transform phase already visits every channel,
so it now stages `decay`, `kk * a` and `r` in shared memory and the row loop reads
only its state row, its value and shared memory: 132 bytes of global traffic per
row instead of 320, and one exponential pair per channel instead of sixteen. The
expressions, their order and their inputs are unchanged, so the receipt's exact
zero (`worst_max_abs_wide_vs_narrow`, and `worst_max_abs` against the chain) still
holds; this was redundant work, not a numerics trade. Batch 32 went from 232.1 µs
to 161.7 µs (-30%), and the profile's per-row cost from 50.4 ns to 30.3 ns.

**Level footing: the same staging on the chain and the narrow kernel -- measured and
rejected.** The fix above left the A/B comparing a staged row loop with two
un-staged ones, which is an asymmetry in the fused route's favour however the
numbers are read. Both baselines were therefore given the same treatment. The
chain's `state_forward` staged `kk`, `scaled_k`, `decay`, `kk * a` and `r` once per
head slot through shared memory -- guarded on `head_size <= 64` with
`64 % head_size == 0`, the only case where a whole head lies inside one 64-lane
group -- and the narrow fused kernel staged `decay`, `kk * a` and `r` on the row
lanes, one channel each, rather than lengthening the serial lane-0 transform prefix
it already pays for. Both measured **slower**, reproducibly, under the standard
`--iterations 20 --repetitions 50` rotation with all four legs in one process:

| batch 32, same process rotation | wide | narrow | chain |
| --- | --- | --- | --- |
| shipped kernels (receipt) | 161.7 µs | 183.8 µs | 183.0 µs |
| with the operands hoisted | 162-166 µs | 190-196 µs (4 runs) | 192-196 µs (4 runs) |

A third build removed the chain's unaligned fallback branch entirely, to ask whether
the dead second loop body was the cost rather than the staging: 192.3 µs, the same
regression, so the hoist itself is what a lane-per-row kernel does not want. The
reconciliation is in what the two shapes pay: the wide kernel's row loop could not
keep its per-row operands live, while in the chain the same five vectors are
broadcast to the sixteen lanes of a head from four cache lines per group, the
exponential pairs are hidden behind the state row's memory traffic, and the staging
instead adds a shared-memory pass and an activation-dependent barrier. So the 30%
was that kernel's own pressure, and the hoist does not transfer. The chain and the
narrow kernel keep their un-staged row loops, which is the equal-footing result:
**each arm runs the form it measured fastest, and the wide arm's win at every
measured geometry does not rest on work the other two were never offered.**

**Every measured producer's sweep is guarded.**
`validation/check_measured_producer_routes.py` is one table-driven checker for all
five route records -- the time-mix projection, the channel mix, the dense and
FP16-storage LN1/low-rank routes, and this key/state stage -- plus the two
receipts that consume those records. Each spec names its
receipt, its record, the sweep it must dispatch, its candidates, its fallback
(the vendor gate or the structural default), its parity contract and its dispatch
accounting. The checker fails when a required batch is missing, when the record
carries a decision for a geometry the receipt never measured,
when a measured geometry outside the band carries no decision, when an entry was
written by another driver, when a recorded route is not the fastest candidate of
its own entry, when a non-default winner sits inside the driver's noise band
(2%, or the FP16 route's 1%), when the receipt now measures a different route
outside that band, or when the default leg stops building the recorded route (or
the documented fallback on an adapter with no entries). For this stage it also
checks the narrow arm, the wide-versus-narrow parity and the arm the default
actually dispatched; `worst_max_abs` must stay exactly zero because a route may
only choose between graphs that compute the same values.

Two receipts consume a producer's record instead of writing one, and the same
table checks them: the multi-step TBPTT receipt must follow the time-mix
projection's verdict at every batch it dispatched, and the packed-cell
composition receipt must agree that the route it qualified is the `staged` one
both the time-mix and channel-mix records name. A consumer measures fewer
geometries than the producer owns -- the TBPTT sweep stops at 16 -- so a record
entry at a batch it did not dispatch is the producer's evidence rather than a
stray entry. That is the `consumer` flag, not an exemption from the record.

The gap it closes is concrete: the A/B job dispatched batches 1/4/8/16 while the
key/state record carried a batch-32 entry, so the geometry where this stage's last
collapse lived was the one geometry the job never measured. Every job sweep now
reaches every recorded geometry (the projection, channel-mix, low-rank and FP16
steps gained `--batch 32` too), and each A/B step is followed immediately by the
guard for that producer on the receipt it just wrote. The per-producer steps in
`.github/workflows/time-mix-ab.yml` are rendered from the same spec table by
`validation/render_ab_workflow.py`, so a producer added to the guard cannot miss
its sweep or its guard step in either job; `workspace-check` fails when the
committed workflow and the table disagree. The workspace check runs the same
guard on the committed pairs, consumers included, without a GPU.
`validation/test_measured_producer_routes.py` proves each rejection on every
spec, including the one that keeps the guard honest -- a geometry where the chain
really is faster passes as soon as the record names the chain -- and
`validation/test_ab_workflow_rendering.py` pins the sweep, guard step and
committed-workflow agreement.

**Result.** Four legs (wide, narrow, chain, default) rotate through every sampling
position in one process; the table is the shipped receipt, `--iterations 20
--repetitions 50`, Intel Gen9 part, milliseconds of one forward:

| batch | wide | narrow | staged | narrow/wide | staged/narrow | staged/wide | default |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 36.9 µs | 42.9 µs | 51.5 µs | 1.161 | 1.202 | 1.395 | fused, wide |
| 4 | 53.1 µs | 58.4 µs | 66.8 µs | 1.100 | 1.144 | 1.259 | fused, wide |
| 8 | 71.3 µs | 81.2 µs | 82.5 µs | 1.138 | 1.016 | 1.156 | fused, wide |
| 16 | 100.0 µs | 119.8 µs | 112.0 µs | 1.199 | 0.935 | 1.120 | fused, wide |
| 32 | 161.7 µs | 183.8 µs | 183.0 µs | 1.136 | 0.996 | 1.132 | fused, wide |

The crossover question is answered on both ends: the wider group wins **every**
measured geometry, by 12-28% over the chain, so neither the batch-16 crossover nor
the batch-32 collapse survives, and the narrow kernel no longer wins a single
measured geometry. Both were kernel-shape problems in the fused route -- first the
lane mapping, then the state phase's redundant per-row work -- not a limit of
fusing the stage.

**Phase attribution.** `validation/profile_rwkv_key_state_forward.py` reads the
device's per-dispatch GPU timestamps (`HIERARCHOS_VULKAN_PROFILE_KERNELS`) and
attributes the stage to its phases; `key_state_forward_profile.json` is the receipt.
The chain's key transform is nearly flat in batch (14.4 / 16.3 / 16.8 / 21.7 µs at
1/8/16/32) while its state phase grows with the row count (17.2 / 31.0 / 52.1 /
107.8 µs), so the state share rises 54% -> 66% -> 76% -> 83%: the phase that scales
with rows is the phase that decides the crossover, exactly as the original
explanation predicted, and it is why the chain's per-row cost falls as rows grow.
The same profile attributes the fused kernel itself, which has no phase seam of its
own: measured against the chain's state phase -- the same rows at the same 64-lane
occupancy -- the wide dispatch now costs an almost constant **+9.5 / +11.2 / +11.9 /
+16.3 µs** at batches 1/8/16/32, which is its transform phase plus one launch. The
batch-32 figure was **+96.9 µs** before the state phase's operands were staged; that
column is what identified the collapse as redundant per-row work rather than the
transform or the group width.

The vendor EU counters (Intel Metrics Discovery / GPA) are not reachable from a
plain Vulkan process here, so the report carries GPU microseconds per phase and the
exact active-lane count per phase (wide 64/64, narrow 1/64 transform and 16/64
state, chain 64/64) rather than an invented utilisation percentage.

**The record carries one route, not a flag plus an arm.** Each entry names the graph
that won -- `wide-fused`, `narrow-fused` or `staged` -- and carries the measured
median of every candidate in `candidates`, so the loader can reject an entry whose
route is not its own fastest candidate. That replaced the earlier `compose` boolean
plus `fused_arm` string, which could disagree with each other and could not express
"the chain" and "which fusion" in one field. The runtime dispatches exactly the
route the table names: `prefers_fused_key_state_forward` decides the route and
`key_state_forward_kernel` picks the arm from the same entry. The A/B switches pin
either fused arm (`HIERARCHOS_RWKV_FORCE_NARROW_KEY_STATE_FORWARD` /
`HIERARCHOS_RWKV_FORCE_WIDE_KEY_STATE_FORWARD`) or the chain
(`HIERARCHOS_RWKV_FORCE_STAGED_KEY_STATE_FORWARD`), and
`HIERARCHOS_RWKV_KEY_STATE_COUNT_FORWARD_DISPATCHES` turns on the route's own
counter, which is what proves 1 against 2 -- and that the default leg dispatched the
route the record names -- in the receipt.

**Policy scoping.** The entry may carry `parameter_storage`, and the whole route --
arm included -- is scoped by it: the op resolves the table with its own policy
(`fp16-storage` once the training graph installed the packed mirrors, `fp32`
otherwise) and a policy-less entry remains the fallback. Recording one policy's
crossover therefore cannot move another's, which is the dimension rank 2 needed and
this record now shares. Reading is wired everywhere; writing a scoped entry is
reserved for a driver whose graph actually runs that storage, which today is the
FP16 low-rank driver. This bench times the FP32 graph, so it writes policy-less
entries and does not stamp a policy it did not measure.

**Method.** `validation/bench_rwkv_key_state_forward.py` measures the stage in
isolation rather than inside a whole time-mix step, because one dispatch against two
is a sub-percent difference of the full core: the benchmark records 50 dispatches
per timed sample into one submission, rotates all four legs, and reports
milliseconds of a single forward. The receipt records the record's route per
geometry (`recorded_route`) and the arm the default actually built
(`default_arm`), so "the default follows the measurement" is checked per leg rather
than asserted.

These graphs are **bit-identical**, including wide against narrow: `worst_max_abs`
and `worst_max_abs_wide_vs_narrow` are exactly zero over `new_state`, `tmix`,
`scaled_k`, `kk` and `saved_sa`, because all three evaluate the same products in
the same order. The narrowed transform phase does not change that: the constant is
the same serial FMA chain, the division and scale are elementwise, and only which
lane performs them moved. The receipt asserts the equality instead of hiding behind
a tolerance, so a future re-association -- or a transform that changed the order --
fails the test rather than silently changing the comparison into a numerics
question.

## Not ranked

- **Time-mix weight-reuse arms (`rows2/4/8`, `two-row`)**: these are variants of
  the promoted fusion, not alternatives to it, and their selection is already
  measured per geometry (`topology_decisions.json`).
- **Post-mix fused producers**: width-gated to `<= 32`; the production graph does
  not build them.
- **`layer_norm_linear_forward_fused`**: transformer-side (the RWKV path uses the
  channel-mix producer), so it is outside this graph.

## Method each A/B must follow

The time-mix harness is the template, and each of these should be measured with
the same five properties rather than a fresh ad-hoc benchmark:

1. **Both arms in one process**, differing only in construction-time switches.
2. **Dispatch accounting** proving each arm ran the graph it claims (the fused
   arm dispatches zero vendor modules where the composition dispatches one, or a
   per-arm dispatch count for portable-vs-portable comparisons).
3. **Bit-exactness or a stated tolerance** over every returned tensor, so a
   timing cannot compare different work.
4. **Interleaved, rotated sampling** so warm caches and clock drift do not favor
   one arm, with medians rather than means in the receipt.
5. **A geometry sweep** ending in a decision record, and a fail-closed default
   check that fails when the shipped graph stops following that record. A single
   batch size is how a geometry-dependent rule gets promoted as if it were
   universal.

## Statement of what is not known

All three ranks are measured, and the FP16 scope gap of rank 2 is closed — both
halves of it. The verdicts went different ways, which is the point of measuring:
rank 1 promoted the composition per geometry, rank 2 kept its fusion at every
dense geometry while its deep FP16 arm measured the chain ahead at 16 and 32
without being allowed to adopt it (not bit-identical to the frozen trajectory), and rank
3's original fusion lost batch 16 to the chain until a wider workgroup removed
that loss. The ranking itself was structural, from the graph shape rather than
from measurement, which is exactly why none of the verdicts was assumed.

Four structural gaps this audit opened are now closed rather than noted: the
storage-policy dimension in *every* route record (the FP16 arm's earlier crossover
is recorded under `fp16-storage`, withheld where it is not bit-identical, and no
longer reported as a divergence), the
fused-arm dimension for the key/state producer (its three candidate graphs are now
three routes in one measured field instead of a compose flag plus an arm string),
the per-phase attribution of the key/state stage (GPU timestamps attribute it to
transform and state, so the crossover explanation is a measurement rather than an
inference from ratios), and the TBPTT chains themselves (two more records that the
scheduler reads per batch, replacing "keep their own receipt" with a measured
route).

The TBPTT packed forward-only and rematerialization chains are now recorded too.
Each has its own schema-2 record (`packed`/`legacy`) that
`RwkvTbpttSequenceOp` reads per batch, and its own receipt/guard pair. On the
Intel Gen9 part neither packed chain wins: the first forward is inside the 2% band
at batches 1-8 and the legacy transition leads by 3.7%/5.6% at 16/32, while packed
rematerialization loses from batch 4 onward (6.5% at 8, 11.8% at 16, 15.7% at 32),
so the record withdraws both where it measured the legacy graph faster. A second
adapter keeps the structural defaults until its own entries exist. The multi-step
time-mix receipt is recorded against this chain-routed default; its composed leg
dispatches the projection once per timestep on the packed first-forward route and
twice on the legacy one, so the receipt test and the guard take that multiplier
from the chain each row reports. Both chain tables are read with the active
parameter-storage policy, and the `fp16-storage` policy now has its own
receipts (`tbptt_packed_forward_fp16_bench.json`,
`tbptt_packed_remat_fp16_bench.json`) recorded with the cell's packed FP16
execution mirrors installed. That is the measurement the earlier refusal was
waiting for. Under the mirrors the first-forward A/B is still bit-identical
(`worst_max_abs` exactly zero over `outputs`, the packed state, `grad_x`, the
packed-state adjoint and the token-feature gradient at every batch) and repeats
the FP32 crossover -- legacy ahead by 3.8% at 16 and 5.3% at 32, with 1 and 4
inside the 2% band and 8 measured as a tie -- so the policy now adopts its own
`legacy` verdicts at 16 and 32 instead of keeping the structural packed route,
and records `packed` at batch 1, where its own sweep measured that verdict inside
the band (the table records a structural-default win to document the geometry
without moving it). The `fp16-storage-fp32-compute` trajectory golden was recorded on the
structural packed-forward route, so this is safe precisely because the receipt
shows the legacy replay reproduces that route bit-for-bit under FP16 storage;
the full training trajectory is re-checked by the stack ladder's FP16 golden
after the change. Neither table carries a `withheld` entry now.

The reverse pass's fusion depth is measured on the same driver (`--chain
backward`), against the most decomposed state-backward topology the device
compiles, with the projection order and low-rank fan-in held fixed so the delta
is the fusion depth alone. The labels name stages, not kernels: `+` is a stage
that still dispatches and `-` is one folded into the state-backward kernel, so
`rkv-add3+key+reduce` is the fully staged chain (kernel, key transform, reduce),
`rkv-add3-key+reduce` folds the key transform in, and `rkv-add3-key-reduce` also
folds the reduction in. The fully folded chain used to be batch-1-only, because
its direct `grad_k_k`/`grad_k_a` writes need one row per channel; a guarded
`FUSE_KEY_PARAM_CROSS_BATCH` specialization now makes it compile at every batch
too. A workgroup owns one (batch, head) row, so the fold cannot be formed inside
one workgroup. Instead each workgroup stores its partials, publishes an arrival
through an atomic per-head counter, and the workgroup that completes a head's
last batch reduces that head's partials in ascending batch order -- the order
`rwkv_key_transform_param_reduce.comp` uses -- and clears the counter. The
variant is compiled for the A/B, which pins it, and for a process whose
backward-fusion record can select `folded` at that adapter and width -- so the
kernel set follows the record instead of preceding it. The folded rung is a
routing candidate, and its acceptance gate is the fold's own output: the
receipt reads the reduction back after the sequence --
`grad_x` never consumes `grad_k_k`/`grad_k_a`, so nothing else in the returned
tensors can see the fold -- and requires it to equal the two-dispatch chain
exactly: the delta is zero at every batch, recorded beside the staged arm's
gradient magnitude so the equality cannot be zero-vs-zero, and only a geometry
that clears it lists `folded` among its candidates. Its returned tensors are
bounded separately, because the contract associates one FP32 product
differently and moves `grad_x` by at most `2.4e-7` (a couple of ulps): the
receipt holds them to a `1e-6` allowance while the routing pair must still agree
exactly. An identical AdamW step on each leg then diffs the parameter snapshots
as a second gate (at most `7.5e-8` against `1e-6`); that gate is weaker, because
AdamW's first update is sign-normalized and a fold that loses a batch row
leaves the update almost unchanged. `--negative-control` breaks whichever publication path the geometry uses:
above batch 1 it seeds one phantom arrival per batch row so no arrival can
complete a head, and at batch 1 -- where the rung writes the gradients directly
and has no counter -- it redirects those writes into a scratch buffer. It then
requires the clean pair to pass both gates and the corrupted pair, whose
key-parameter gradients stay at zero, to fail both; the driver applies its own receipt gates to those rows, so the guard
proven is the guard the sweep uses. The topology table had been ordered most-fused-first,
which made the staged arm look for a topology *below* the shipped schedule and
resolve to the most folded kernel; while that kernel was batch-1-only the sweep
reported batches 4-32 `unmeasured`.
Ordering the table stage-first makes the A/B measure the three-dispatch chain
against the autotuner's two-dispatch one at every batch, and the graphs stay
bit-identical (`worst_max_abs` exactly zero) for the two-dispatch pair. With
the folded rung in the candidate set the three depths sit within half a percent
of each other in the committed receipts -- fused fastest at six of ten, staged
at two, folded at two -- so `tbptt_backward_fusion_decisions.json` records the
fused verdict at batches 4 and 32 and withdraws 1, 8 and 16, all inside the 2%
band. The FP16-scoped table keeps one entry, batch 32. A non-default win outside
the band would replace the schedule at that geometry; one inside the band
withdraws the entry, so the record's silence is a measured tie rather than an
omission. No entry names `folded`, so the fold's kernels are not compiled into
the shipped graph by default: the route is recordable, and this adapter's
measurements do not justify adopting it.

The refusal mechanism still exists for the next policy whose trajectory is
pinned to the structural graph: a `withheld` entry is scoped to one policy, has
to name the measured route that policy would otherwise read, carries a reason,
and once a policy refuses anything it has to refuse every measured move it did
not measure itself. The guard exercises that contract against a fixture that
withdraws the FP16 measurements, so the shipped record does not have to keep a
stale refusal alive to keep the check honest. Rematerialization's measured
verdicts are all its structural default under both policies, so its record
carries no scoped entries; the driver regenerates the list from the measured
decisions on every recording, so it cannot go stale.

What is still not covered:
* **Every adapter other than the Intel Gen9 part.** Each record carries that
  adapter's entries per geometry; a device with no entry keeps the structural
  default. Re-recording on a second GPU is a driver run
  (`--record-decisions`), not a code change.
* **Backward fusions.** Out of scope from the start, and still the largest
  unmeasured surface: a forward win did not imply a forward-backward win in the
  packed-cell A/B.

## Why the fully folded rung never leads on Intel Gen9

No entry in `tbptt_backward_fusion_decisions.json` names `folded`. In the
committed receipts the rung is within 0.35% of `fused` at every batch: ahead at
batches 1 and 8 of the FP32 pass (by 0.16% and 0.14%), behind at the other
three, inside the 2% band everywhere. "Never leads" therefore means it never
leads by enough to record, not that it always loses -- and the question that
leaves is whether the fold's arrival counters or the batch-order reduction it
performs is what keeps it that close. `bench_rwkv_tbptt_packed_chains_bench` answers that directly with
`--profile-folded` (driven by
`bench_rwkv_tbptt_packed_chains.py --profile-folded --chain backward`), which
times four legs in one process and interleaves them: the fully staged chain, the
autotuner's `fused` depth, the fully folded rung, and -- above batch 1 -- a
folded leg whose reduction is prevented from firing.

The fourth leg removes the reduction without removing anything else. Only the
workgroup that observes a head's last arrival runs that head's batch-order
reduction, so seeding a full batch of phantom arrivals removes that loop while
every workgroup still stores its partials, publishes its arrival and passes the
same device-scope barriers. `folded - without_reduction` is therefore the
reduction tail, and `without_reduction - fused` is the counters credited against
the separate reduce dispatch they replace. Both legs are gated before they are
timed, on the same two guards the sweep uses: the clean fold must still publish
the two-dispatch chain's `grad_k_k`/`grad_k_a` exactly, and the phantom leg must
fail that gate. A profile whose legs disagree with those gates is refused rather
than reported.

Each leg also reports its own spread, and the profile prints a `resolution_ms`:
the widest leg's interquartile range, i.e. the smallest delta this run can call a
measurement. A component is flagged `*_beyond_resolution` only if its delta
exceeds that number; the flags exist so a signed difference of a few tenths of a
millisecond is not read as a cost when it is a sample of noise. The legs carry
their own range and interquartile spread, so the guard re-derives the resolution
from them rather than trusting the field: a resolution edited to swallow a delta
fails `check_folded_publication_profile.py`, which runs in `workspace-check`
beside the other committed-artifact guards.

Both profiles are committed artifacts (`validation/tbptt_folded_publication_profile.json`
for FP32 and `..._fp16.json` under the packed mirrors), recorded on the Intel Gen9
part at width 128 with 8-step windows, 40 interleaved samples per leg and 3
warmup rounds. `validation/check_folded_publication_profile.py` re-derives every
delta, ratio and resolution from the timed legs, requires the gates above, and
fails if the fold starts leading or either component becomes visible.

FP32:

| batch | staged | fused | folded | folded/fused | resolution | tail | counters vs dispatch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 21.454 | 21.402 | 21.426 | 1.0011 | +/-0.485 | n/a | n/a |
| 4 | 25.844 | 25.855 | 25.801 | 0.9979 | +/-0.453 | -0.011 | -0.043 |
| 8 | 29.459 | 29.420 | 29.531 | 1.0038 | +/-0.381 | +0.112 | -0.001 |
| 16 | 40.884 | 41.009 | 41.121 | 1.0027 | +/-0.595 | +0.172 | -0.060 |
| 32 | 65.073 | 64.776 | 64.684 | 0.9986 | +/-1.232 | -0.183 | +0.091 |

FP16-storage, the policy whose route receipt also never names `folded`:

| batch | staged | fused | folded | folded/fused | resolution | tail | counters vs dispatch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 23.570 | 23.350 | 23.294 | 0.9976 | +/-0.402 | n/a | n/a |
| 4 | 29.018 | 28.867 | 28.812 | 0.9981 | +/-1.705 | +0.069 | -0.124 |
| 8 | 31.248 | 31.387 | 31.619 | 1.0074 | +/-2.125 | +0.177 | +0.055 |
| 16 | 42.235 | 42.357 | 42.358 | 1.0000 | +/-0.627 | -0.077 | +0.078 |
| 32 | 66.406 | 66.164 | 65.916 | 0.9963 | +/-1.738 | -0.232 | -0.016 |

Milliseconds, medians of the rotated samples. Every reduction tail and every
counter delta is inside its run's resolution, so `*_beyond_resolution` is false at
all twenty component rows and `folded_beats_fused_beyond_resolution` at all ten
rows of the two profiles. The fold's ratio spans `0.9963` to `1.0074` of `fused`
across them -- wider than any single batch's delta, and its sign moves with the
pass -- which is what an unresolved tie looks like: the rung is not losing, it is
invisible at this scale. The split does not change with the storage policy, so
neither does the reason it cannot lead.

**The answer is neither: both components are below the resolution, and their
sum is the reason the rung cannot lead.** The reduction tail is between -0.23
and +0.18 ms (it is negative as often as positive, because removing the loop
also removes the last workgroup's dependency on every other arrival), and the
counters, credited against the reduce dispatch they replace, land between -0.12
and +0.09 ms -- the same order. Against a 21-66 ms step that is under one
percent either way, which is exactly the band the record's 2% margin withdraws.

The structural reason the ceiling is that low is that folding the reduction
cannot remove work, only a dispatch boundary. The staged chain's reduce is
`width` work-items, each summing `batch` partials: at width 128 and batch 32 that
is 128 threads, 32 dependent adds each, 32 KB of coalesced partial reads and
512 B of stores -- a single wavefront of work behind a memory-latency chain. The
fold still writes those same partials and still reads them back (in the same
ascending batch order, which is why the output is bit-identical), so its only
saving is the launch itself plus the redundant pass over the partials that a
separate dispatch makes. Against that it pays one atomic and one device-scope
barrier per workgroup and a tail in which the last-arriving workgroup reduces a
head after every other workgroup of the dispatch has finished. Trade one
dispatch boundary for an in-dispatch arrival barrier on this part and the two
costs come out equal, which is precisely what the profile measures.

The `fused` rung's advantage over the staged chain is a different kind of fold.
It absorbs the key transform, which is elementwise over
`batch x width x head_size` values and therefore grows with every geometry
parameter, so there is real work to save: in the FP32 receipt it is 0.7% and
0.25% ahead at batches 16 and 32, and staged is only 0.3% ahead at batch 4. By
the time the chain reaches `folded`, the only stage left to absorb is the one
whose work is `width x batch` adds -- and a sub-millisecond dispatch folded into
a multi-millisecond kernel cannot produce a two-percent win at any batch this
driver sweeps.

That is the answer the record already encodes: `folded` stays a recordable
candidate with a proven acceptance gate, and no entry names it, because the
rung's maximum prize on this adapter is smaller than the measurement's noise.
It would take a geometry where the reduce dispatch is a larger share of the step
-- a much wider model, or a batch small enough that the state kernel is cheap --
for the fold to have anything measurable to win, and that geometry has to be
measured rather than assumed.

Limits of this profile: it is one adapter (Intel Gen9, one driver), batch 1 has
neither counters nor a reduction so its row is a control rather than a split,
and the resolution is 0.4-2.1 ms here -- a component smaller than that cannot be
resolved by this driver on this part, which is why the answer is "both are below
resolution" rather than a ratio between two unmeasurable numbers. Re-running the
command on a second adapter either reproduces the tie or produces the first
geometry where one of the two components is visible.
