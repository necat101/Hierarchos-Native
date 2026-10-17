# Packed RWKV cell: fused shader vs vendor-linear composition

`RwkvPackedCellOp` has two forward graphs for the same cell:

* **fused** — one `packed_cell_channel_mix_state_forward_fused` dispatch per row
  performs layer-norm/mix, `key_cm`, `relu2 × deepembed`, `value_cm`, the
  residual and the packed-state write.
* **composed** — the channel-mix producer runs its layer-norm/mix kernel and then
  dispatches `key_cm` and `value_cm` through the vendor `linear-forward` and
  `linear-residual-forward` modules that the rest of the training graph uses,
  with the state pack as separate dispatches.

`HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION` selects the composed
graph for both the channel-mix op and the packed cell. The packed runner prints
`packed-cell-forward-kernel: fused|composed` on stderr, and
`RwkvPackedCellOp::packed_channel_mix_forward_fused()` exposes the same fact to
callers, so a run cannot silently be read as the graph it did not build.

## Parity

`verify_rwkv_packed_cell_composition.py` generates one tiny coherent-v9 package
(width 128, head size 16, batch 8, live DeepEmbed adapter), builds both graphs,
steps them three times with each leg carrying its own packed state, and compares
`output`, `packed_new_state`, `grad_x`, `grad_packed_state` and
`token_feature_grad` every step.

| Check | Result |
| --- | --- |
| 15 comparisons across 3 steps | worst absolute difference `0.0` on every tensor |
| kernel markers | `fused` leg built the fused shader, `composed` leg did not |
| vendor dispatch | fused leg dispatched nothing; composed dispatched `linear-forward` and `linear-residual-forward` once per step |
| resolved plan | `intel-gen9-lds` for both families on the HD 520 target |
| negative control (`x × 1.01`) | rejected at `1.5862e-4` against a `1.0389e-4` bound |

The agreement is bit-exact because the fused shader keeps the linear modules'
lane-serial accumulation order. That is the strongest form of parity, but it
also means this check cannot distinguish correctness from two identical
transcriptions of the same bug; the CPU bridge in
`verify_rwkv_cpu_bridge.py` remains the independent oracle.

Batch 8 is deliberate: the Gen9 gate needs `output_dim >= 192` below eight rows,
and `value_cm` produces 128, so a four-row fixture would leave the residual
linear on the portable module and make the check vacuous. The RDNA3 gate needs
eight rows outright.

## Vendor qualification

`--expected-vendor {auto,intel,amd}` requires the resolved plan to name that
vendor's tuned modules for both linear families. `auto` still refuses a portable
resolution. `HIERARCHOS_VULKAN_FORCE_VENDOR` is rejected, because it changes the
plan without changing the hardware that executes it.

* Intel HD Graphics 520: `--expected-vendor intel` passes and records
  `linear-forward=intel-gen9-lds`, `linear-residual-forward=intel-gen9-lds`.
* AMD RDNA3: run the same command with `--expected-vendor amd` on an RDNA3
  adapter. On this Intel host that leg fails closed with
  `linear-forward resolved 'intel-gen9-lds', not an amd module; this host cannot
  qualify the amd path` and exits 1, which is the intended guard rather than a
  host-agnostic pass.

## Throughput

`bench_rwkv_packed_cell_composition.py` builds both graphs in one process,
warms both, then interleaves their `forward_backward` calls, alternating which
graph runs first. Release build, Intel HD Graphics 520, width 128, head size 16,
30 samples per graph, 5 warm-up steps. Medians in milliseconds; `ratio` is
composed ÷ fused.

| batch | fused | composed | ratio |
| --- | --- | --- | --- |
| 1 | 2.684 | 2.687 | 1.001 |
| 4 | 3.435 | 3.409 | 0.993 |
| 8 | 4.202 | 3.974 | 0.946 |
| 16 | 6.219 | 5.485 | 0.882 |
| 32 | 11.115 | 9.696 | 0.872 |

The two graphs are bit-identical in every run (`max_abs_output` and
`max_abs_packed_state` are `0.0`), so this compares the same work.

**Verdict: the fused packed kernel can be deleted without losing throughput.**
It never wins outside noise at one row and loses steadily as the batch grows,
reaching about 12% at 16–32 rows. The fused shader is a single-dispatch
convenience, not a performance feature on this part.

**What this does not settle before deleting it.** Only the public
`forward_backward` API is timed; the TBPTT forward-only transition
(`record_forward_transition_into`) has its own packed path whose pack cost is
not in this table. The sweep covers one width (128) on one adapter, and the
`value_cm` projection only reaches the tuned module from eight rows up, so the
sub-eight-row rows above compare portable modules on both sides. Measure the
forward-only transition and one wider package, then remove the fused kernel, its
switch and this A/B; the CPU bridge would still cover the remaining graph.

## Reproduce

```bash
python hierarchos-vulkan/validation/verify_rwkv_packed_cell_composition.py \
  --expected-vendor intel --record hierarchos-vulkan/validation/packed_cell_composition.json
python hierarchos-vulkan/validation/bench_rwkv_packed_cell_composition.py \
  --batch 8 --iterations 30 --record hierarchos-vulkan/validation/packed_cell_composition_bench.json
python -m unittest discover -s hierarchos-vulkan/validation -p "test_rwkv_composition_receipt.py"
```

The last command runs the recorded-evidence contract tests by default; set
`HIERARCHOS_RUN_GPU_VALIDATION=1` (and optionally
`HIERARCHOS_EXPECTED_VENDOR=amd`) to re-run the harness itself.
