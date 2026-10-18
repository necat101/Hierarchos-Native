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
| `layer_norm_low_rank_producer_forward_fused*` (six FP16 arms) | **Measured among themselves**, per device and shape | `topology_decisions.json`, `verify_topology_decisions.py`; the choice between arms is autotuned, but the fused-vs-staged question below is still open |

Two of the four are settled because the measurement went against the fusion. That
is the reason this audit exists at all: the default should not be "fused wins".

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

### 2. The LN1/low-rank full-forward producer — one dispatch replacing the staged chain

`low_rank_full_forward_fused_fp16_packed` keeps one row's w/a/g rank activations
resident through the activation, the second low-rank matmuls, decay and the final
gate. The staged alternative (`HIERARCHOS_RWKV_LOW_RANK_DISABLE_FULL_FORWARD_FUSION`
forces it) runs the layer norm, the interpolation, the three first-stage matmuls
and the second-stage matmuls as separate dispatches.

Why it ranks second and not first: the merge is deeper (the staged path is the
longest chain in the cell for this shape), but the fused arms have already been
autotuned per geometry, so a loss here would be surprising and the payoff is
bounded by the fact that the chain's matmuls are rank-sized and cheap per
dispatch. The A/B should still run because the time-mix result was also
"surprising" at small batch.

The disable switch makes both arms reachable today; what is missing is a receipt
that proves the arms are bit-identical, counts their dispatches, and records the
per-geometry winner.

### 3. `rwkv_key_state_forward_fused` — one dispatch replacing two

`record_key_state_forward` uses the fused key-transform/state-update kernel when
`head_size <= 64` and falls back to `key_transform_forward` + `state_forward`
otherwise. One dispatch saved per layer-step, no switch, no record.

Why it ranks last: the smallest merge, and the fallback exists structurally, so
the A/B is a one-line switch plus a sweep. Worth doing for completeness and
because it is the cheapest measurement in this list, not because a large win is
expected.

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

No A/B in this audit has been run yet. Ranks 1–3 are ordered by expected payoff
from the graph structure (dispatches merged, per-layer-step multiplicity,
existence of a reachable composition), not by measurement, and the audit should
be revised after rank 1 reports.
