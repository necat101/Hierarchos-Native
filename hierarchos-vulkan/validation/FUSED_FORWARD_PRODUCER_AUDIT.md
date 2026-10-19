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
a forward verdict.

## Already decided

| Producer | Status | Evidence |
| --- | --- | --- |
| `rwkv_time_mix3_linear3_key_state_forward_fused*` (time-mix projection + key + state) | **Promoted per geometry**, now from a per-device decision record | `packed_cell_time_mix_bench.json`, `PACKED_CELL_COMPOSITION.md`; composition loses where the tuned `linear3-forward` module runs, wins nowhere else |
| `packed_cell_channel_mix_state_forward_fused` (channel mix + packed-state write) | **Removed** | `PACKED_CELL_COMPOSITION.md`; lost the composition at every batch and the forward-only transition by 11% |
| `group_norm_bonus_gate_linear_residual_forward_fused_{compact_one_row,two_rows}` | **Not reachable at production width** | gated on `width <= 32` in `rwkv_post_mix.rs`; the audit does not rank a kernel the width-128 graph never builds |
| `layer_norm_low_rank_producer_forward_fused*` (six FP16 arms) | **Measured among themselves and against the staged chain**; the fusion stays | `topology_decisions.json`, `verify_topology_decisions.py` for the arm choice; `packed_cell_low_rank_forward_bench.json` and `low_rank_full_forward_decisions.json` for the route (rank 2 below); `low_rank_fp16_forward_bench.json` for the deep FP16 arm |
| `rwkv_key_state_forward_fused*` (key transform + state update) | **Measured: the wider-group variant wins every measured batch after two shape fixes — an all-lane transform phase and channel-invariant operands staged out of the row loop** | `key_state_forward_bench.json`, `key_state_forward_profile.json` and `key_state_forward_decisions.json` (rank 3 below) |

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
read the same packed operands; only the dispatch structure differs. Fresh receipt
(`low_rank_fp16_forward_bench.json`, `--iterations 30 --warmup 5`):

| batch | fused (deep FP16 arm) | staged | default | staged/fused | route dispatches (fused/staged/default) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.261 ms | 1.327 ms | 1.267 ms (fused) | 1.053 | 1/11/1 |
| 4 | 1.307 ms | 1.388 ms | 1.305 ms (fused) | 1.062 | 1/11/1 |
| 8 | 1.332 ms | 1.302 ms | 1.296 ms (staged) | 0.977 | 1/11/11 |
| 16 | 1.455 ms | 1.413 ms | 1.433 ms (staged) | 0.971 | 1/11/11 |
| 32 | 1.697 ms | 1.644 ms | 1.639 ms (staged) | 0.969 | 1/11/11 |

Three independent sweeps agreed on the verdict at every batch (the largest spread
was ~2.5 points at batch 1, where every run still put the chain behind). The fused arm is
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
the FP32 arm's: staged wins at batches 8, 16 and 32 (0.977, 0.971, 0.969) where the
dense route keeps the fusion, and fused wins at 1 and 4, the same geometries the
dense route names. The first three sweeps reported the first three of those rows
as divergences because the record had no storage-policy dimension to hold them.
It has one now: the record is keyed by `parameter_storage` as well as by adapter,
width and batch, and this driver writes its verdicts under `fp16-storage` — the
storage those training policies select, an entry only a consumer running that
storage policy reads (the older `fp16-packed-mirror` spelling canonicalizes into
the same scope).
`route_following` reports `agrees` at every geometry, the default leg dispatches
the chain exactly where this receipt measured it faster, and the policy-less
entries the dense route reads are untouched: batch 16 still has no dense entry and
batch 4 is still the dense arm's fused win (1.034x on the packed cell, 1.104x
here). That is what the dimension is for — an FP16-only win no longer has to
either ship as an FP32 claim or be thrown away.

**The rule.** `prefers_fused_forward_chain` keeps the fusion wherever the producer
kernels exist, withdraws it per (adapter, width, batch) where
`validation/low_rank_full_forward_decisions.json` records a staged win, and is
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
batch 32 the row work is what the 64 workgroups have to get through, so the
redundancy set the throughput. The transform phase already visits every channel,
so it now stages `decay`, `kk * a` and `r` in shared memory and the row loop reads
only its state row, its value and shared memory: 132 bytes of global traffic per
row instead of 320, and one exponential pair per channel instead of sixteen. The
expressions, their order and their inputs are unchanged, so the receipt's exact
zero (`worst_max_abs_wide_vs_narrow`, and `worst_max_abs` against the chain) still
holds; this was redundant work, not a numerics trade. Batch 32 went from 232.1 µs
to 161.7 µs (-30%), and the profile's per-row cost from 50.4 ns to 30.3 ns.

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
dense geometry while its deep FP16 arm takes the chain from batch 8 up, and rank
3's original fusion lost batch 16 to the chain until a wider workgroup removed
that loss. The ranking itself was structural, from the graph shape rather than
from measurement, which is exactly why none of the verdicts was assumed.

Three structural gaps this audit opened are now closed rather than noted: the
storage-policy dimension in *every* route record (the FP16 arm's earlier crossover
is recorded under `fp16-storage` and no longer reported as a divergence), the
fused-arm dimension for the key/state producer (its three candidate graphs are now
three routes in one measured field instead of a compose flag plus an arm string),
and the per-phase attribution of the key/state stage (GPU timestamps attribute it
to transform and state, so the crossover explanation is a measurement rather than
an inference from ratios).

What is still not covered:

* **The TBPTT packed forward-only and rematerialization chains**, which are
  separate fusions with their own receipt (`packed_cell_composition_bench.json`).
* **Every adapter other than the Intel Gen9 part.** Each record carries that
  adapter's entries per geometry; a device with no entry keeps the structural
  default. Re-recording on a second GPU is a driver run
  (`--record-decisions`), not a code change.
* **Backward fusions.** Out of scope from the start, and still the largest
  unmeasured surface: a forward win did not imply a forward-backward win in the
  packed-cell A/B.
