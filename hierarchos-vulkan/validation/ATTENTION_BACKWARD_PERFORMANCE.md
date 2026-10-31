# Attention backward regression closure — 2026-10-15

The measured mode-0 attention-backward slowdown at larger batches is closed on
the local AMD Radeon Graphics device. Numerical outputs remain bit-identical.

## Cause and change

`transformer_attention_backward` launches one invocation per workgroup. Its cache
optimization introduced ten workgroup variables, reserving 14,400 bytes even for
attention modes that do not use the strict-adjoint caches. Those arrays have only
one reader/writer in each workgroup, so they can use invocation-private storage.

Changing the ten storage declarations removes all Workgroup variables from the
compiled module. No arithmetic, loop order, descriptor binding, push constant,
dispatch geometry, cache bound or fallback changes. Driver register/scratch
allocation replaces the fixed shared-memory reservation.

## Measured results

GPU timestamp medians of five samples, one dispatch per sample. All rows use
mode 0, 64 queries, 64 keys, four query heads, two KV heads and head dimension 32.
The harness compares the complete output vectors bit for bit before reporting a
benchmark result.

| Batch | Original artifact | Regressed shared caches | Fixed private caches |
| ---: | ---: | ---: | ---: |
| 2 | 52.9 ms | 44.1 ms | 44.4 ms |
| 8 | 54.3 ms | 45.2 ms | 45.5 ms |
| 16 | 54.8 ms | 45.4 ms | 45.9 ms |
| 32 | 55.8 ms | 88.8 ms | 48.4 ms |
| 128 | 78.6 ms | 263.8 ms | 71.1 ms |

Original-versus-fixed and regressed-versus-fixed are separate paired runs; the
table shows the fixed result from the original pair. Both paired measurements
are retained in the JSON receipt. The batch-128 regressed-versus-fixed pair is
263.776 / 70.874 ms, a 3.72× improvement. The fixed kernel is 1.10–1.20× faster
than the original across the tested batches, while small-batch differences from
the earlier cached kernel are about one percent.

Strict mode 1 was also compared at a safe eight-query/eight-key shape: batch 32
is 46.1 ms original versus 1.4 ms fixed, with exact gradients. This smaller shape
does not establish that long strict-mode jobs avoid the driver's watchdog.

## Regression protection and verification

- All 79 existing attention invocations pass direct original-artifact comparison
  and unchanged frozen output digests, including strict modes and cache fallbacks.
- Both numerical verification profiles retain their original full-state goldens.
  See [SHIP_PARITY.md](SHIP_PARITY.md) and the refreshed release receipts.
- The shader manifest requires local size `[1,1,1]` and zero workgroup variables.
  The former artifact has ten and fails that storage policy.
- Baseline comparison in the attention harness now fails on any differing FP32
  bit, including one-ULP differences. Benchmark mode also checks complete output
  vectors instead of discarding them.
- `--bench-only` cannot bypass golden verification, and `--bench-mode 0` avoids
  automatically running the long strict-mode benchmark that resets this driver.

From `hierarchos-vulkan`, with the original shader artifacts in `BASELINE_DIR`:

```powershell
cargo build --bin attention_kernel_cache_check
./target/debug/attention_kernel_cache_check.exe BASELINE_DIR --golden validation/kernel_golden/attention.json --golden-strict
./target/debug/attention_kernel_cache_check.exe BASELINE_DIR --bench --bench-only --bench-mode 0 --bench-repeats 1 --bench-batch 128
python validation/verify_training_stack.py --release
```

Detailed timings and comparison logs are under `release_parity/attention_backward/`.
These are kernel timings on one AMD device, not whole-training throughput or
cross-vendor performance claims.

## Gen9 (Intel HD Graphics 520) re-measurement — 2026-10-15

The closure above was measured on the local AMD device. Re-running the same
harness on Intel Gen9 (`attention_kernel_cache_check` with
`.attention-optimization-baseline` as `BASELINE_DIR`) shows the private-cache
change is a win on Gen9 as well, and that the device loss Gen9 still reports is
not a regression from it.

Mode 0, backward, 64 queries, 64 keys, 4 query heads, 2 KV heads, head dim 32:

| Batch | Pre-optimization artifact | Current private caches | Speedup |
| ---: | ---: | ---: | ---: |
| 2 | 167.1 ms | 100.9 ms | 1.66x |
| 8 | 169.6 ms | 105.3 ms | 1.61x |
| 16 | 174.3 ms | 109.6 ms | 1.59x |
| 32 | 180.4 ms | 112.4 ms | 1.60x |
| 64 | 187.5 ms | 119.0 ms | 1.58x |
| 128 | 378.8 ms | 241.7 ms | 1.57x |

Forward at the same batch points is 1.27-1.75x faster than the
pre-optimization artifact, so the fix is at least as good on Gen9 as it is on
AMD at every batch size measured. Every benchmarked dispatch is compared bit for
bit against the pre-optimization artifact before its timing is reported
(`different=0 max_abs=0`), and the golden parity pass is bit-identical for every
case up to the one that faults.

### Which artifact loses the device on Gen9

Mode 1 (strict adjoint) backward is the mode that resets a driver, on AMD as well
as here, and the parity harness runs the candidate and the baseline back to back
in one session, so a device loss leaves the case's attribution ambiguous rather
than identifying which artifact faulted. Separating them means running each
artifact against itself
(`--candidate-dir shaders` for the current module, `--candidate-dir
.attention-optimization-baseline` for the pre-optimization one), and removing the
baseline argument entirely to run the candidate alone:

| Mode 1 backward, one workgroup | Pre-optimization artifact | Current private caches |
| --- | ---: | ---: |
| 4x4 | 117.6 ms | 0.969 ms |
| 8x8 | 1101.6 ms | 4.053 ms |
| 16x16 | `ERROR_DEVICE_LOST` | 14.9 ms |
| 32x32 | `ERROR_DEVICE_LOST` | 52.5 ms |
| 64x64, 4 heads | `ERROR_DEVICE_LOST` | 379.4 ms |
| 64x64, batch 2 (4 workgroups) | `ERROR_DEVICE_LOST` | 144.4 ms |
| candidate alone, full parity case list | — | `ERROR_DEVICE_LOST` at `strict_head_fallback backward` (before the fix below; that run now completes) |

The pre-optimization artifact's mode 1 timings grow by roughly a factor of ten per
doubling (117.6 -> 1101.6 -> about 11,000 ms), which crosses the driver's watchdog
interval. Every device loss seen in the mode 1 *benchmark* therefore belongs to
that artifact: each shape above completes when the same module is on both sides of
the harness and dies when the pre-optimization module is the one being measured.
The private-cache change removed the timeout exposure rather than creating it, by
121-272x at these strict shapes. There is no occupancy regression to fix here.

The one shape the kernel could not run on Gen9 before the fix below was the parity
case `strict_head_fallback` (3 queries, 4 keys, head dimension 257): it failed with
no baseline module loaded at all, so the fault was the current module's, not the
harness's. Overriding the head dimension through the harness (`--bench-qk-dim` / `--bench-value-dim`, added
for this measurement; mode 1, 8 queries, 8 keys, one batch, the same module on
both sides) gives the cliff directly:

| Head dimension | Mode 1 backward dispatch |
| ---: | ---: |
| 64 | 5.1 ms |
| 128 | 9.3 ms |
| 256 | 18.4 ms |
| 257 | `ERROR_DEVICE_LOST` |
| 512 | `ERROR_DEVICE_LOST` |

Cost is linear in the head dimension up to the 256-wide cache bound and the
dispatch resets the device immediately above it, which points at the strict
fallback rather than at the silicon: `HEAD_CACHE_SIZE` is 256, so `cache_stats &&
max_head_dim <= HEAD_CACHE_SIZE` (line 563) sends any larger head dimension down
that path. The fallback is nested the wrong way round -- its
`for (d < max_head_dim)` loop sits *outside* the per-head, per-query softmax, so
`max_score`, `exp_sum`/`inv_sum`, the softmax dot and the `dp` dot over the value
dimension are all recomputed once per head-dimension element even though none of
them depends on `d`. Every term inside that loop except the final
`q_values[q_base + d]` / `grad_output[grad_out_base + d]` multiply is
d-independent, and the code carries no head-dimension-sized array, so the failure
is a watchdog timeout rather than an out-of-bounds access: the path costs about
`head_dim` times the per-element work, and therefore about `head_dim^2` overall.
At 257 that is two orders of magnitude above the 18.4 ms the bounded path spends
at 256. The earlier note here that this shape was "far too small for a watchdog
expiry" was wrong for exactly this reason -- it compared the sequence length and
missed the head-dimension factor.

### Fix: window the head dimension in the strict cached path

`cache_stats` is derived from the row count alone -- `(last_batch - first_batch) *
queries_per_kv * pc.query_seq_len <= STATS_CACHE_SIZE` -- and the primary pass
records `row_max`, `row_inverse` and `row_softmax_dot` whenever it holds, so the
statistics the replay recomputed were already in memory.
`max(qk_head_dim, value_head_dim) <= HEAD_CACHE_SIZE` was in the guard only
because `head_k_cache`, `head_v_cache`, `shared_k_cache` and `shared_v_cache` are
256 wide, and every output element accumulates independently of the others. The
strict cached path therefore now walks the head dimension in windows of
`HEAD_CACHE_SIZE` (`d_base`, `window`, `k_window`, `v_window`) and the guard is
simply `if (cache_stats)`: any head dimension uses the cached statistics, with the
`dp` dot and the `p`/`dropout`/`ds` scalars recomputed once per window instead of
once per element. The replay is no longer reachable for head-dimension reasons,
and the per-element reduction order is untouched.

| Head dimension | Before | After |
| ---: | ---: | ---: |
| 64 | 5.1 ms | 4.8 ms |
| 128 | 9.3 ms | 9.4 ms |
| 256 | 18.4 ms | 18.4 ms |
| 257 | `ERROR_DEVICE_LOST` | 22.1 ms |
| 512 | `ERROR_DEVICE_LOST` | 43.9 ms |
| 1024 | not attempted | 117.9 ms |

### Qualification

The frozen digests could not qualify this when the change landed: every entry in
`attention.json` was recorded under `AMD Radeon Graphics`, so a Gen9 run verified 0
digests and reported 79 unverified on another device. This device's digests are
recorded now (see below), so the qualification below used two oracles instead. Below the bound,
the full parity case list run against the pre-window artifact gives 74
bit-identical comparisons (`different=0 max_abs=0.000000000`) before the run stops
-- and it stops only because that baseline replays at head dimension 257 and
resets the device, while the fixed module runs the same case to completion. The
candidate-only case list now finishes all 83 invocations with exit 0. Above the
bound, the benchmark path compares complete output vectors bit for bit, and the
fixed module against the pre-window artifact passes that comparison at head
dimensions 257 and 512 while running 1.84x and 1.94x faster. The library suite is
unchanged at 706 passed, 0 failed, 9 ignored, mode 0 backward still measures
1.59-1.64x over the pre-optimization artifact across batches 2-128, and
`validation/check_shader_provenance.py` confirms the artifact still rebuilds
byte-for-byte from the source under the pinned glslang 16.5.0.

`validation/check_attention_backward_shapes.py` guards the property that every
bit-exactness check is blind to -- that the dispatch finishes at all. It runs the
head-dimension and batch ladders against a single module, passes on the fixed
artifacts, and fails on the pre-window ones at head dimensions 257, 512 and 1024.

The other route into the replay remains: a row count above `STATS_CACHE_SIZE`
still selects it, and there a large head dimension costs the same `head_dim`
factor inside one dispatch. At rows = 1024 it measures 2365 ms at head dim 257,
which is close enough to the watchdog to be worth catching, and the dispatch guard
below does that. Windowing that path itself needs per-row statistics, which is
exactly what the row-count bound denies.

Windowing the head dimension, unlike splitting the work across workgroups, does not
change the reduction order: each output element keeps the ascending (head, query)
accumulation it has today, so no golden was re-recorded for this change. The
harness still offers `--bench-mode 0` to skip the long strict-mode benchmark that
resets a driver carrying the unpatched kernel.

### Full-device sweep: where the watchdog actually bites

With the head-dimension bound closed, the harness was swept across sequence length,
head dimension and batch to find what else resets Gen9. The same module is on both
sides of the harness throughout, one repeat per shape.

| Mode 0, batch 1, head dim 32 | Forward | Backward |
| ---: | ---: | ---: |
| seq 64 | 6.7 ms | 99.6 ms |
| seq 128 | 25.1 ms | 400 ms |
| seq 256 | 97.9 ms | 2580 ms |
| seq 512 | 386.8 ms | `ERROR_DEVICE_LOST` |
| seq 1024 | 1558 ms | `ERROR_DEVICE_LOST` |

Strict mode 1 backward is worse: 143 ms at seq 64, 576 ms at seq 128, and
`ERROR_DEVICE_LOST` from seq 256. The batch ladder is flat per dispatch (99.6,
106.1, 114.6 ms at batch 1, 8, 32; 229 ms at batch 128, where the workgroups no
longer all fit), and forward survives 1558 ms at seq 1024, so the wall is the
backward pass and sits between 2.58 s and 10 s of per-invocation work. A
512-token attention backward is simply beyond this part -- a 2015 15 W integrated
GPU doing the same FLOPs -- so no bit-exact restructure recovers it.

Sequence length is not the only route back into the replay, which still carries the
head-dimension factor. `cache_stats` is false exactly when `batch * (num_heads /
num_key_value_heads) * query_seq_len > 512`, and at rows = 1024 the replay measures
152 ms at head dim 32, 551 ms at 128 and **2365 ms at 257**, against 22.3 ms for the
cached path at the same head dimension. It completes -- one step short of the
watchdog -- and it is the one corner where the head-dimension factor is still
reachable. Windowing it needs per-row statistics, which is exactly what the
row-count bound denies; processing the row range in blocks small enough to cache
would remove the factor, but that means restructuring the primary pass that fills
the statistics, not the replay's inner loop.

### Recorded digests

`attention.json` now carries this device's digests beside the AMD entries.
`--write-golden` recorded 79 invocations and merging them added 79 entries with
**no conflict on any key**: this part reproduces the AMD machine's digests bit for
bit across all 79 invocations, which is cross-vendor evidence for the determinism
the parity work depends on. The compare run now reports

```
PASS: golden digests verified for 79 invocation(s) on Intel(R) HD Graphics 520
(0 unverified on another device)
```

so attention changes are qualified here by the frozen digests rather than by
comparison against a second module.

### The watchdog budget, bracketed

The boundary was bracketed in both modes, one repeat per shape, one invocation
per dispatch. Strict mode passes seq 240 at 2973 ms and loses the device at
seq 248; mode 0 passes seq 256 at 2576 ms and loses seq 320. The budget is
therefore about **3.0 s of per-invocation work**.

Individual points scatter with this part's power state -- seq 192 measured
2256 ms where the ladder predicts about 1.2 s -- so a guard expressed as a
predicted millisecond figure would be unreliable. Expressed as work it is
consistent. With

```
work = (num_heads / num_key_value_heads) * head_dim * query_seq_len * key_value_seq_len
```

and a 1.44x factor for strict mode, every shape that completes lands at or below
5.3e6 and every loss above 5.7e6: mode 0 seq 256 is 4.19e6 (2576 ms) and seq 320
is 6.55e6 (lost); strict seq 240 is 5.31e6 (2973 ms, passing) and seq 248 is
5.67e6 (lost). Batching does not enter it, which matches the flat batch ladder: the limit is the
submission. (This sentence used to read "per invocation". A later measurement corrected
it -- eight key-range blocks dividing one invocation's work by eight still lost the
device, which cannot happen if the limit were per invocation -- and the correction
section below records the ladder that came out of it.)

`check_attention_backward_shapes.py --cases sequence` asserts the last known-good
shape of both modes, so a narrowed boundary fails the check rather than silently
shrinking the supported envelope.

### Dispatch guard

`vendor::attention_backward_work_budget(device)` returns the per-dispatch work
budget for the class the boundary was measured on (Intel, at most 32 KiB of shared
memory) and `None` for every other adapter, so a device whose boundary has not been
measured is never refused. `TransformerKernels::new` captures it and
`record_backward` refuses a geometry above it with a message naming the shape, the
work it needs, the budget and the
`HIERARCHOS_VULKAN_IGNORE_ATTENTION_WORK_BUDGET` override, instead of dispatching
something that resets the device.

The budget is 5.0e6 units: it admits everything measured to complete and sits about
a tenth below the last passing strict shape. Mode 0 seq 256 (4.19e6) and strict
seq 224 (4.62e6) dispatch unsplit; mode 0 seq 320 (6.55e6) and strict seq 248 (5.67e6)
keep their refusals, the first because the fused path cannot be split and the second
because splitting it would move it onto the streamed adjoint.
`attention_backward_key_blocks_split_only_the_strict_adjoint` pins those points,
including that the widest block of every split shape fits, and the library suite
passes unchanged (711 tests) with the split active on the bounded device.

Those 5.0e6 units and the 1.44x strict factor are the *cached* adjoint's calibration:
every point of the ladder that produced them ran a geometry whose row statistics are
cacheable. The streamed adjoint is a different order of cost, so it has its own model --
`attention_backward_streamed_work` carries the full key span and a second head-dimension
factor, because a key-range block bounds the outer key loop while the softmax replay
inside the head-dimension loop still spans every key. Model and bound are both selected
from the geometry (`attention_backward_statistics_are_cached`), and the streamed bound
scales the budget the caller derived from its device instead of replacing it, so a
device with no measured boundary is still never refused.

### What the split does and does not rescue (measured)

The split is bit-exact, but the measurements say it does not yet rescue the shapes
that lose the device, and why. Driving the strict streaming shapes directly, with the
harness's own split drive, so both arms run the same adjoint:

| Run | Result |
| --- | --- |
| strict 248 (statistics cached), 4 key-range blocks | `ERROR_DEVICE_LOST` |
| strict 320, 8 key-range blocks | `ERROR_DEVICE_LOST` |
| strict 320, single dispatch | `ERROR_DEVICE_LOST` |
| strict 512, 8 key-range blocks | `ERROR_DEVICE_LOST` |
| strict 512, single dispatch | `ERROR_DEVICE_LOST` |

Splitting keys bounds the adjoint's outer key loop and nothing else. The streamed
adjoint recomputes the whole softmax for every outer key, and it does that inside the
head-dimension loop, so its cost is still quadratic in the key count and multiplied by
the head dimension: 320 keys at head dim 32 needs far more than 8 blocks to come
inside the watchdog. The measured consequence is that the split's practical envelope
here is smaller than the bit-exactness result suggests, and the cap of 8 is not the
binding limit -- the adjoint's own cost is.

That redirects the next step. What actually blocks a long sequence is that the row
statistics live in an invocation-private array of 512 rows, which forces any geometry
above `(batch rows) * queries per KV head * query length > 512` onto the streamed
replay, and that replay is the expensive quadratic path. Moving those three arrays
(`row_max`, `row_inverse`, `row_softmax_dot`) into a bound buffer removes the ceiling,
which would make the cheap cached adjoint work at any sequence length *and* make it
splittable, because a bound buffer is visible to every block. That is the change the
measurements point at, and it is not implemented.

The recalibration those runs forced is anchored on the losing side only, and that is its
main limitation. Every streamed geometry driven here lost the device -- 257 queries x
257 keys at head dim 8, the same shape at head dim 16, and 320 x 320 and 512 x 512 at
head dim 32 -- so no streamed geometry was measured to complete on this part, and
`ATTENTION_BACKWARD_STREAMED_WORK_BUDGET` sits below the smallest measured loss rather
than between a pass and a loss. What that bound refuses is measured; what it admits is
not. Its consequence for the split is a prediction rather than a result, and it is
pinned in the test above: the smallest measured loss needs 2 key-range blocks to come
inside the bound, while none of the head-dim-32 shapes can be rescued even at the cap
of 8, so those keep their diagnosable refusal.

### Correction: a grid partition cannot rescue the watchdog (measured)

Everything above about shapes being rescued by key-range blocks is superseded. Those
blocks are workgroups of one dispatch, so they shorten each workgroup's outer key loop
but not the submission's duration, and the watchdog bounds the submission. Measured at
257 queries, head dim 8 and 2 queries per KV head: one dispatch and 8 key-range blocks
of the same span both lost the device, and so did 4 and 2 blocks.

The streamed adjoint's own ladder at that geometry, one block: key spans 1, 4 and 8
completed in 61 ms, 395 ms and 1394 ms, and span 16 lost the device, so its boundary
lies between 2.11e6 and 8.42e6 of the streamed model's units. The bound is now 1.9e6, a
tenth under the last span measured to complete; a first version anchored on the losing
side alone came out about a thousand times too generous and admitted shapes that reset
the device. So the split credited no block count for a streamed geometry while its
blocks were grid workgroups. The per-submission interface that makes the credit sound
shipped afterwards; the section at the end of this document supersedes this paragraph,
and there each block is its own submission, carrying its identity in the push constants
rather than deriving it from the grid.
`attention_backward_budget_matches_the_measured_boundary` pins those points, and
the library suite passes unchanged (711 tests) with the guard active on the bounded
device, so it refuses nothing that currently works.

The refusal is a pure function, `vendor::attention_backward_budget_error`, so
`attention_backward_budget_error_names_the_geometry_and_the_override` drives it with
geometries this device cannot run -- mode 0 seq 512 and strict seq 248 -- and checks
both the decision and the message, including that the failure names the override,
without dispatching anything. The call site is then a call to that function and a
`bail!`, so the only part not covered is the call itself.

### Key-range split: what shipped, and what the earlier audit got wrong

The backward kernel partitions the key range across sequential submissions. As first
shipped it rode the grid's third dimension, which the model's backward dispatch already
left at 1 and which every non-strict mode ignores entirely, so that first version
changed no push constant, `AttentionPush` field, harness push array or binding -- and
the correction above measures why it could not shorten a submission. The block now
arrives in the push constants:

```
key_blocks = strict(unfused_qk 1|2|9) ? pc.key_block_count : 1
key_block  = min(pc.key_block_index, key_blocks - 1)
split_begin = key_block * key_len / key_blocks
split_end = (key_block + 1) * key_len / key_blocks
```

Every split submission is launched with a z of one, so the grid partitions nothing; a z
above one would repeat a block rather than add one.

Three properties make it bit-exact:

* Only the strict adjoint may be split. Its per-key K and V outputs are
  self-contained assignments, so a block owning keys [b, e) writes `grad_k` and
  `grad_v` for exactly those keys and no reduction changes order.
* The shared primary pass -- row statistics, `grad_q`, `grad_sink`,
  `grad_relative_bias` -- is owner-gated to block 0, or to the sole block when the
  grid is unsplit, because those accumulate over the whole key range. One primary
  pass, N per-key passes.
* Per-key zeroing is bounded to each block's own range, so it cannot race another
  block's assignments: key ownership is disjoint and no block writes an element
  another block owns.

A split run cannot use the cached-statistics adjoint, because the row statistics are
invocation-private; it takes the streaming adjoint instead, which recomputes what it
needs from the keys it owns. Every unsplit caller is bit-identical: with a grid of one
block every bound is today's, and the module compiled from this source reproduces all
79 recorded digests, exactly as the module it replaced does.

`attention_kernel_cache_check --split-key-blocks N` drives the split. It applies it
only to backward strict invocations, mirroring the shader's own gate, and prints
`split <case> backward=true blocks=N`, so a pass cannot be vacuous:

| Run | Result |
| --- | --- |
| shipped module vs split-capable module, grid 1 | every invocation `different=0`; **79 digests PASS** |
| split across 4 key-range dispatches | 20 backward invocations split 4 ways; **79 digests PASS** |

The audit this section used to carry named `grad_relative_bias` as the blocker and
concluded the split was not worth starting. That was wrong in both directions. Here
the primary pass runs once, so relative-bias needs no special case at all:
`mode1_feature3 backward=true` (relative_bias_mode 1) was split and matched its
digest. The real constraint is the fused path -- with `unfused_qk` 0 or 3 the strict
adjoint never runs, so `grad_q`, `grad_k` and `grad_v` all come from the primary pass,
whose `softmax_dot` is a running prefix feeding `ds` and whose `grad_q` buffer is
declared `writeonly`. Splitting that needs an exact cross-dispatch prefix carry plus
an in-place read-modify-write of `grad_q`, a different redesign, so the fused path
stays single-block and keeps the dispatch guard's refusal.

Wired now: the guarded dispatch site derives the block count from the same device
budget -- `vendor::attention_backward_key_blocks` returns 1 for a shape that fits,
for the fused path, for an adapter with no measured boundary, and for a geometry
whose row statistics are cached, and otherwise the number of sequential blocks needed
so that even the *widest* slice lands under the budget, capped at
`ATTENTION_BACKWARD_MAX_KEY_BLOCKS`. The count is the grid's third dimension at that
one site, and the guard then bounds the widest block rather than the whole sequence.

The cached-geometry exclusion is the one the measurements forced. A split cannot see
the owner's invocation-private row statistics, so it takes the streamed adjoint, and
that is far more expensive per key. `--split-key-blocks` on the strict seq 248 shape
-- whose statistics *are* cached -- took the device out with `ERROR_DEVICE_LOST` at 4
blocks, where a single dispatch had been refused cleanly. Splitting therefore applies
only where the unsplit dispatch would have streamed too, so the blocks divide the same
work; everything else keeps the diagnosable refusal.

One finding worth recording on the way: the committed
`transformer_attention_backward.spv` is not reproducible from its committed `.comp`
(25658 rebuilt words against 25662, 18469 differing). A fresh glslang 16.6.0 compile
still reproduced all 79 digests, so this module is qualified by the digests rather
than by byte provenance, unlike the vendor modules `vendor.rs` decodes.

### Per-submission key blocks: the split now rescues the shape that lost the device (measured)

The correction above ends by naming what a split would need: each block submitted on its
own, with the block's identity in the push constants, because a per-block submission
launched with a z of one cannot tell itself apart from the others. That is what shipped.

| Piece | What changed |
| --- | --- |
| Push block | `pc.key_block_index` and `pc.key_block_count` appended to the backward module's block. `AttentionBackwardPush` extends the shared `AttentionPush` on the host, so only the backward kernel's push range grew (120 -> 128 bytes): the forward module and its literals are untouched and every forward dispatch still passes 120. |
| Split derivation | `key_blocks = strict ? pc.key_block_count : 1`, `key_block = min(pc.key_block_index, key_blocks - 1)`, and the block's range follows from that block instead of `gl_GlobalInvocationID.z`. Split submissions launch a z of one. |
| Dispatch | The training site records one dispatch per block with a `submit_and_restart` between blocks -- the primitive the crate already uses for watchdog-safe job slicing -- so each block is its own submission, and the wait between them preserves the order they accumulate in. A count of one is still a single dispatch and no extra submission. |
| Policy | `attention_backward_key_blocks` credits a real count again, but only where each *submission* fits: the streamed model carries the full key span, so the keys one block may own is the streamed bound divided by the per-key cost. |
| Harness | `attention_kernel_cache_check --split-key-blocks N` drives N submissions, one per block, each with the block words set, and prints `split_block b/N ms=...`. |

Measured on Intel HD Graphics 520, one build, one module, strict (streamed) adjoint at
257 queries, head dim 8, 2 queries per KV head and key span 16 -- the shape the earlier
ladder recorded losing the device:

| Run | Result |
| --- | --- |
| one submission (span 16 unsplit) | `ERROR_DEVICE_LOST`, exit 1 |
| six submissions, blocks of 2/3/3/2/3/3 keys | **completes**, exit 0, 6.24 s total |
| per-submission times | 490 / 2322 / 866 / 872 / 857 / 853 ms |

The rescue is real, and the block times are not uniform: block 2 is 2.7x its neighbours
and it repeated at 2303 ms in a second pass, so it is a property of that block's key
range rather than power-state noise. Nothing in the shape of block 2 explains it yet.

Bit-exactness is held through the new interface rather than assumed.
`attention_backward_key_range_split_is_bit_exact` drives one dispatch and six
submissions of the same geometry through the library kernel and asserts bit-identical
`grad_q`, `grad_k` and `grad_v`; it passes on device.

What the split still cannot do matters as much as what it can, because the block count
is not the binding limit -- the streamed adjoint's own cost is. A block costs
`queries per KV head * head_dim^2 * queries * block keys * keys`, so the keys one block
may own is `bound / per-key cost`, and a geometry whose *single* key exceeds the bound
cannot be split at all. At 257 queries and head dim 8 one key costs
`2 * 8 * 8 * 257 * 257 = 8.45e6` against a 1.9e6 bound, so the whole 257-key sequence
keeps its refusal, as does every head-dim-32 shape measured. The rescue window at head
dim 8 reaches key span 16 with six blocks and is capped by
`ATTENTION_BACKWARD_MAX_KEY_BLOCKS = 8`, so a geometry needing more than eight
submissions is refused instead of dispatched; the guard's message for those names the
widest block's work rather than the whole span, and a shape no split can help gets a
message saying a single key already exceeds the bound.

The invariant that replaces the old "never credit a streamed geometry" regression is
pinned in `attention_backward_key_blocks_split_only_the_strict_adjoint`: a credited
count is sound only when the widest block it leaves behind fits the streamed bound, and
a count that cannot reach that must be the cap with the guard refusing the geometry.
Both arms are asserted, so a future policy crediting blocks for a shape no submission
can run fails the test instead of reaching the driver.
