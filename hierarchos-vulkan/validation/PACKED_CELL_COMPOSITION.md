# Packed RWKV cell: vendor-linear composition

`RwkvPackedCellOp` composes its forward graph from the same modules the rest of
the training graph uses:

* the channel-mix producer runs its layer-norm/mix kernel, then `key_cm` and
  `value_cm` dispatch the vendor `linear-forward` and
  `linear-residual-forward` modules (or the portable modules when no tuned
  variant qualifies);
* when the time-mix takes its non-fused projection, that projection dispatches
  the vendor `linear3-forward` module (the default training pass uses the fused
  projection/key/state kernel, which calls no vendor module);
* the packed state is written by separate pack dispatches.

There used to be a `packed_cell_channel_mix_state_forward_fused` alternative
that performed the channel mix and the state write in one portable dispatch.
It was measured against the composition and removed (see **Throughput**): it
never won outside noise at one row, lost steadily as the batch grew, and also
lost the TBPTT forward-only transition, which was its last plausible niche.

## Parity

`verify_rwkv_packed_cell_composition.py` generates one tiny coherent-v9 package
(width 128, head size 16, batch 8, live DeepEmbed adapter), runs the packed cell
once with the tuned vendor modules and once with
`HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS`, steps both three times with each leg
carrying its own packed state, and compares `output`, `packed_new_state`,
`grad_x`, `grad_packed_state` and `token_feature_grad` every step.

Two fusion switches are forced on for both legs so the vendor modules are
actually reached:

* `HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION` makes the time-mix take
  its `linear3-forward` composition instead of the fused
  projection/key/state kernel, which calls no vendor module.
* `HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION` makes the channel mix
  take its staged composition instead of its own fused forward, which would
  otherwise absorb `key_cm`/`value_cm`.

| Check | Result |
| --- | --- |
| 15 comparisons across 3 steps | worst absolute difference `0.0` on every tensor |
| vendor dispatch | `linear-forward=1 linear-residual-forward=1 linear3-forward=1` per step |
| portable dispatch | empty (the A/B arm really ran the portable modules) |
| resolved plan | `intel-gen9-lds` for all three families on the HD 520 target |
| negative control (`x × 1.01`) | rejected at `1.5862e-4` against a `1.0389e-4` bound |

The agreement is bit-exact because the vendor modules keep the portable
modules' lane-serial accumulation order. That is the strongest form of parity,
but it also means this check cannot distinguish correctness from two identical
transcriptions of the same bug; the CPU bridge in
`verify_rwkv_cpu_bridge.py` remains the independent oracle.

Batch 8 is deliberate: the Gen9 gate needs `output_dim >= 192` below eight rows,
and `value_cm` produces 128, so a four-row fixture would leave the residual
linear on the portable module and make the check vacuous. The RDNA3 gate needs
eight rows outright.

## Vendor qualification

`--expected-vendor {auto,intel,amd}` requires the resolved plan to name that
vendor's tuned modules for all three linear families. `auto` still refuses a
portable resolution. `HIERARCHOS_VULKAN_FORCE_VENDOR` is rejected, because it
changes the plan without changing the hardware that executes it.

* Intel HD Graphics 520: `--expected-vendor intel` passes and records
  `linear-forward=intel-gen9-lds`,
  `linear-residual-forward=intel-gen9-lds` and
  `linear3-forward=intel-gen9-lds`.
* AMD RDNA3: run the same command with `--expected-vendor amd` on an RDNA3
  adapter. On this Intel host that leg fails closed with
  `linear-forward resolved 'intel-gen9-lds', not an amd module; this host cannot
  qualify the amd path` and exits 1, which is the intended guard rather than a
  host-agnostic pass.

## Throughput

The final A/B ran in one process, warmed both graphs, and interleaved their
calls, alternating which graph ran first. Release build, Intel HD Graphics 520,
width 128, head size 16, batch 8, 30 samples per graph, 5 warm-up steps. Medians
in milliseconds; `ratio` is composed ÷ fused. `forward_backward` is the public
training pass; `forward-only` is the TBPTT transition
(`record_forward_transition_into`) including its packed-state unpack and
next-slot pack.

| Graph | fused | composed | ratio |
| --- | --- | --- | --- |
| `forward_backward` | 4.039 | 3.871 | 0.958 |
| forward-only transition | 1.870 | 1.664 | 0.890 |

The two graphs were bit-identical in every run (`max_abs_output` and
`max_abs_packed_state` are `0.0`), so this compares the same work. The earlier
full sweep told the same story: composed was `0.993x` at batch 4, `0.946x` at 8,
`0.882x` at 16 and `0.872x` at 32.

**Verdict: the fused packed kernel was removed.** It lost the forward-only
transition — the one path where its single dispatch and fused pack looked
plausible — by 11%, and it lost `forward_backward` as well. The benchmark binary
and its JSON receipt were retired with the kernel; the numbers above are the
final recorded evidence from the run that made the decision. The remaining
graph stays covered by the vendor/portable A/B above and by the independent CPU
bridge.

**What this does not cover.** Only one width (128) and one adapter were timed,
the AMD qualification path is implemented but unexercised on this host, and the
forward-only timing includes the host upload/readback edges rather than a full
multi-step TBPTT sequence. The time-mix projection has since been A/B'd in its
own right; the next section records it.

## Time-mix projection: fused kernel vs `linear3-forward` composition

The channel-mix kernel above was not the only fused forward in the packed cell.
The time-mix projection also has two graphs: the fused projection/key/state
kernel the packed cell builds by default, and the composition that runs the
`linear3-forward` module (tuned where the vendor gate admits it, portable
otherwise) followed by the fused key/state kernels. They are numerically
the same graph: every returned tensor was bit-identical (`0.0` worst absolute)
in every run below, which is why the choice is scheduling and not precision.

The key/state half of that second graph now has its own route record
(`validation/key_state_forward_decisions.json`, rank 3 of
`FUSED_FORWARD_PRODUCER_AUDIT.md`), and that record is three-way: `wide-fused`,
`narrow-fused` and `staged` are each a route, so an entry names the fused kernel as
well as whether the stage fused at all. On the Intel Gen9 part every measured
geometry now names the wider-group fused kernel, batch 16 and batch 32 included,
after the all-lane transform and the staged state operands (rank 3 of the audit);
a device with no entry keeps that structural default, and `staged` remains a route
the record can carry where a measurement says so. A composed leg of this A/B and
the key/state route can therefore disagree at the same batch, and that is not a
contradiction: they measure different stages. The projection decision above is
unchanged by that; the two are separate records, and both read the same
parameter-storage policy scope.

`validation/bench_rwkv_packed_cell_time_mix.py` and its
`hierarchos-vulkan-rwkv-packed-cell-time-mix-bench` binary build three legs in
one process over one fixture — the fused kernel forced on, the composition
forced on, and the production default — then run them interleaved with a rotated
order. Each leg reports the graph it will dispatch and its vendor
`linear3-forward` dispatch count, so the receipt proves what ran instead of
asserting it. 120 samples per graph, 10 warm-up steps, Intel HD Graphics 520,
width 128, head size 16. `ratio` is composed ÷ fused.

| batch | fused | composed | ratio | vendor `linear3-forward` | default graph |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.592 | 2.617 | 1.009 | not admitted (portable) | fused |
| 2 | 2.795 | 2.819 | 1.009 | not admitted (portable) | fused |
| 4 | 3.232 | 3.253 | 1.006 | not admitted (portable) | fused |
| 8 | 4.067 | 3.979 | 0.978 | 1 dispatch | composed |
| 16 | 6.382 | 6.159 | 0.965 | 1 dispatch | composed |
| 32 | 11.095 | 10.647 | 0.960 | 1 dispatch | composed |

Two 400-sample confirmation runs at batches 4 and 8 reproduced both signs
(`1.007`/`1.010` portable, `0.985`/`0.987` tuned), and the earlier 40-sample
sweep showed the tuned win widening to `0.963`/`0.962`/`0.959` at 8/16/32.

**Verdict: promote per geometry, not per kernel.** The composition wins exactly
where the tuned module is admitted for the dispatch geometry; the fused kernel
wins where the composition would fall back to the portable module. The `rows`
threshold is not new: it is the vendor layer's own admission gate
(`INTEL_MATMUL_MIN_ROWS`/`INTEL_MATMUL_MIN_OUTPUT_DIM`, and the RDNA3 module's
`rows >= 8`), the same predicate the module's dispatch uses.

**The crossover itself is now recorded per device.** This table is one adapter's
measurement, so `prefer_composed_projection` first asks the vendor gate whether
the composition is even about a tuned module, then reads
`validation/time_mix_projection_decisions.json` (compiled in by
`src/decision_records.rs`, which serves both producers' records by `kind`) for that
adapter's own winning graph. A device
with a recorded entry follows its measurement; a device without one keeps the
conservative default — compose where its own tuned module is admitted — because
the measured win belongs to the module, not to the three-way split. Recording is
the offline measurement step (`--record-decisions`), not a runtime autotune: a
training graph must not time itself on the hot path. The record carries the ratio
that produced each decision, and the compiled table rejects an entry whose
`compose` flag contradicts its own ratio, so a hand-edited record fails at build
time instead of changing the graph. `verify_time_mix_projection_decisions.py`
replays every entry for the attached adapter and also checks that the shipped
default builds the graph the record names, which catches a stale binary.

Two switches make the measurement auditable:
`HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION` drops the fused kernel at
construction, and `HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION` restores it at
every geometry (the A/B's fused arm).

The default is what the third leg records: `fused` with zero vendor dispatches
at batches 1/2/4, `composed` with one at 8/16/32.

The rule lives in the shared full-forward time-mix projection
(`record_time_mix_projection_key_state_forward_topology`), not in the packed
cell, so the TBPTT and causal-LM graphs inherit it; the packed cell is simply
where it was measured. The causal-LM vendor parity fixture is the second
observation: its histogram moved from `linear3-forward=0` to `20` at batch 16,
where the module is admitted, and its bit-exact vendor/portable comparison still
passes.

**What this does not cover.** Only one width and one adapter were timed; the
RDNA3 gate is the same predicate but its crossover is unmeasured on this host.
The win is small (1-4%) next to the channel-mix measurement above, and it is a
forward-projection measurement — the backward path is untouched by either graph,
so a full training step sees a fraction of it.

## End to end: a whole TBPTT window

A single `forward_backward` is not the unit a training run pays for. The next
question was whether the promotion survives a whole sequence — the forward sweep
over eight timesteps, the reverse pass, the state pack/unpack — or whether the
reverse pass dilutes it into noise.

`validation/bench_rwkv_tbptt_time_mix.py` and its
`hierarchos-vulkan-rwkv-tbptt-time-mix-bench` binary build one
`RwkvTbpttSequenceOp` per leg over the same fixture and time whole sequences with
the same rotated interleaving. The binary refuses to print a timing unless the
fused leg dispatched no vendor `linear3-forward` module, the composed leg
dispatched one per timestep, the default sequence followed the same decision the
single-step bench measured, and every returned tensor over the whole sequence
agrees (`0.0` worst absolute everywhere). 8 samples per leg, 2 warm-up
sequences, 8-step window, Intel HD Graphics 520, width 128, head size 16.

| batch | fused window | composed window | ratio | per step, fused → composed | default graph |
| --- | --- | --- | --- | --- | --- |
| 1 | 21.685 | 21.551 | 0.994 | 2.711 → 2.694 | fused |
| 4 | 25.974 | 25.988 | 1.001 | 3.247 → 3.248 | fused |
| 8 | 31.820 | 31.217 | 0.981 | 3.977 → 3.902 | composed |
| 16 | 49.548 | 47.348 | 0.956 | 6.194 → 5.919 | composed |

A second run reproduced every sign in the same bands (`0.994→1.017` and
`1.001→1.009` at batches 1/4, `0.981→0.975` and `0.956→0.965` at 8/16), so the
window keeps the split rather than flattening it. At batch 16 the promoted graph
saves `2.2 ms` of a `49.5 ms` window — 4.4% of the sequence, or `0.28 ms` per
step — which is the number a training run would actually see on this part.

The end-to-end result also sharpened the small-batch conclusion: over a whole
window the composition is a tie there (`0.994` and `1.001`, against a
single-call `1.006-1.009`), not a fused win. That is why the record keeps the
fused kernel at those geometries — it is never slower — and why the receipt
test asserts "the composition does not win" rather than inventing a win the
window does not show.

## Reproduce

```bash
python hierarchos-vulkan/validation/verify_rwkv_packed_cell_composition.py \
  --expected-vendor intel --record hierarchos-vulkan/validation/packed_cell_composition.json
python hierarchos-vulkan/validation/bench_rwkv_packed_cell_time_mix.py \
  --batch 1 --batch 2 --batch 4 --batch 8 --batch 16 --batch 32 \
  --iterations 120 --warmup 10 \
  --record hierarchos-vulkan/validation/packed_cell_time_mix_bench.json \
  --record-decisions hierarchos-vulkan/validation/time_mix_projection_decisions.json
python hierarchos-vulkan/validation/bench_rwkv_tbptt_time_mix.py \
  --batch 1 --batch 4 --batch 8 --batch 16 --steps 8 --iterations 8 --warmup 2 \
  --record hierarchos-vulkan/validation/packed_cell_tbptt_time_mix_bench.json
python hierarchos-vulkan/validation/verify_time_mix_projection_decisions.py
python -m unittest discover -s hierarchos-vulkan/validation -p "test_rwkv_composition_receipt.py"
python -m unittest discover -s hierarchos-vulkan/validation -p "test_rwkv_time_mix_receipt.py"
python -m unittest discover -s hierarchos-vulkan/validation -p "test_rwkv_tbptt_time_mix_receipt.py"
python hierarchos-vulkan/validation/verify_rwkv_cpu_bridge.py
```

The receipt tests run by default; set `HIERARCHOS_RUN_GPU_VALIDATION=1` (and
optionally `HIERARCHOS_EXPECTED_VENDOR=amd`) to re-run the harnesses themselves.
Both time-mix benches fail, rather than record, when the default stops building
the graph the decision record names or when the two graphs disagree, and
`.github/workflows/time-mix-ab.yml` runs this whole sequence on a self-hosted
Vulkan runner, so the rule is measured on hardware rather than only against a
checked-in receipt.
