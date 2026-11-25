# Phase VIII — RDNA3 GQA key-pass key-block parallelism investigation

**Date:** 2026-11-24
**Base commit:** `560809eead186f832344c15d1a9b70795da3b75d`
**Verdict:** **NO-GO for promotion.** The production key pass is retained unchanged.
Splitting the destination key sequence across more workgroups is bit-exact but
**1.5× to 4.6× slower** at the primary 8q/2kv geometry, and the workgroup count
turns out not to be the binding constraint: changing it by 8× moves the kernel by
less than 2%.

The experiment answers the Phase VII §8 hypothesis with measurement rather than
inference, and it changes no production source, no production shader, no
eligibility predicate, no selector and no default route.

---

## 1. Preservation evidence for Phases III–VII

Recorded before the first Phase VIII write (`raw/git-identity-preserved.txt`):

| Check | Value |
| --- | --- |
| Branch | `main` |
| `git rev-parse HEAD` | `560809eead186f832344c15d1a9b70795da3b75d` (expected) |
| `git status --short` | 17 entries: 13 modified tracked files + 4 untracked receipt directories |
| `git diff \| sha256sum` | `383296d7b58b0664563eda3b3b7ac1fa862beb9d7e804fb2715603a558243d77` |
| `git diff --binary \| sha256sum` | `8290f780928a8c7fa4c12c4979f309f5929d5bd2bafd9e70ae18342865d6995f` |
| `git diff --cached --binary \| sha256sum` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty index) |
| `git diff --stat` | 13 files changed, 2813 insertions(+), 202 deletions(-) |
| `git stash list` | `stash@{0}: On main: pre-v0.2.6 untracked validation evidence (preserved, not deleted)` |
| `git rev-parse stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` |

All four prior receipts verified present before any edit:

| Receipt | Files |
| --- | ---: |
| `validation/mha_attention_backward_closeout_20261124/` | 36 |
| `validation/gqa_attention_backward_experiment_20261124/` | 13 |
| `validation/gqa_production_promotion_20261124/` | 14 |
| `validation/gqa_end_to_end_training_20261124/` | 178 |

### Production shader identities (unchanged, before and after this phase)

| File | sha256 | Matches Phase VII |
| --- | --- | --- |
| `shaders/transformer_attention_backward.comp` | `3c8e6267e320c1f187175209e8a42e6a2658494c6bb3762b1e009cb8b7810a34` | yes |
| `shaders/transformer_attention_backward.spv` | `e78cb2920e3253d7187c479c23bee40d4b4ab677fedeaa97f3109bf6a00dc5a8` | yes |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` | yes |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` | yes |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `e417935459e49dee1bc56db400f321496a2c04ad05392e8b14aef40d6c92ff78` | yes |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` | yes |

### Production GQA eligibility predicate (unread, unchanged)

```
(num_heads, num_key_value_heads) in {(8,2), (4,2), (8,4), (16,4)}
head_dim == 128; q_seq == kv_seq == 64; causal == 1; unfused_qk in {0, 9}
plus every pre-existing MHA restriction and key_blocks == 1
```

`two_pass_attention_gate` still returns `Scalar{reason:"key_range_split"}` for any
`key_blocks != 1`; the bool that decides the default route and the allowlist above
were not edited.

### What this phase changed

`raw/git-identity-end-of-phase.txt`:

| Check | Value |
| --- | --- |
| `git diff --binary \| sha256sum` | `46cbb50fb76ce56c7b56f4cd93b6c916607ab11fa4b07417934a6f5baa6f2c3b` |
| `git diff --stat` | 13 files changed, **2817** insertions(+), 202 deletions(-) |
| `src/transformer.rs` | +4 lines, all `#[cfg(test)]` module declaration |
| Other tracked files | none changed (verified by `git diff --name-only`: still 13) |
| `stash@{0}` | untouched, `831f8927…` |

The tracked diff grew by exactly four lines:

```rust
#[cfg(test)]
#[path = "phase_viii_key_block_tests.rs"]
mod phase_viii_key_block_tests;
```

Everything else this phase adds is untracked: the experiment module
(`src/phase_viii_key_block_tests.rs`), the candidate shader and its four SPIR-V
builds, and this receipt. No reset, clean, rebase, checkout, discard, commit,
push, merge or release was performed.

---

## 2. The existing key-pass execution geometry

Read from source, then confirmed against runtime dispatch counters and the
profiler's per-shader dispatch counts.

| Property | Query pass | Key pass |
| --- | --- | --- |
| Artifact | `transformer_attention_backward_query_rdna3.comp/.spv` | `transformer_attention_backward_key_rdna3.comp/.spv` |
| `local_size_x` | 64 | 64 |
| Row domain | `batch * num_heads * query_seq` | `batch * num_key_value_heads * key_seq` |
| Rows at 8q/2kv | 2 × 8 × 64 = 1024 | 2 × 2 × 64 = **256** |
| Grid | `[(1024+63)/64, 1, 1]` = **16 workgroups** | `[(256+63)/64, 1, 1]` = **4 workgroups** |
| Lanes | 1024 | 256 |
| One lane owns | one `(batch, head, query)` row | one `(batch, KV head, key)` element |

**Row → coordinate mapping** (key shader, unchanged):

```
key      = row % key_value_seq_len
rest     = row / key_value_seq_len
kv_head  = rest % num_key_value_heads
batch    = rest / num_key_value_heads
```

So workgroup `w` owns rows `[64w, 64w+64)`. At the primary geometry that is
exactly one `(batch, KV head)` pair per workgroup, and its 64 lanes are the 64 key
positions `0..63`. **Key positions are not distributed inside a workgroup and are
not sub-ranged at all: every lane already owns exactly one destination key.**

**Where the full query-position accumulation occurs.** Inside each lane, over the
`queries_per_kv` query heads mapped to its KV head, and inside each head over all
`query_seq_len` ascending query positions:

* mode 0 — one continuous accumulator per element for the whole group
  (`((0+t_h0q0)+t_h0q1)+…+t_h1q0+…`), with `precise` product then `precise` add;
* mode 9 — a per-head accumulator combined into a shared accumulator in head order
  (the two-stage tree).

At 8q/2kv that is 4 heads × 64 queries = **256 accumulation steps per element per
lane** (128 at 4q/2kv). The query pass, by contrast, has 64 accumulating steps per
lane (one per key) because its row is one query position.

**`key_blocks` already has a defined meaning — for the other shader.** The generic
scalar backward (`shaders/transformer_attention_backward.comp`) carries
`key_block_index` / `key_block_count` and partitions its key range as
`[b*len/blocks, (b+1)*len/blocks)` for the strict adjoint only, and
`vendor::attention_backward_key_blocks` sizes that count from a device work
budget, purely as a watchdog bound on one submission. Its own comment states the
performance finding this phase re-measures: *"The watchdog bounds a submission
rather than an invocation, so dividing the work inside a single submission buys
nothing."* The two-pass key shader accepts a **30-word (120-byte)** push block
with no key-block words, and the gate refuses any `key_blocks != 1`. The two-pass
key pass is therefore structurally unsplittable today, which is why an experiment
needed its own artifact.

**Can additional blocks own disjoint key regions without a cross-workgroup
reduction?** Yes — and this is the one part of the hypothesis that measurement
confirms. `queue 1` below shows 0 missing and 0 overlapping destination elements
for every configuration, because a key's dK/dV is a self-contained assignment, not
an accumulation into shared state. The numerical premise was correct. The
performance premise was not.

---

## 3. Proposed partition scheme (and why its block count cannot add parallelism)

The candidate keeps the production row domain and partitions it per
`(batch, KV head)`:

```
blocks      = max(key_block_count, 1)
block       = min(key_block_index, blocks - 1)
block_begin = block * key_value_seq_len / blocks
block_end   = (block + 1) * key_value_seq_len / blocks
block_span  = block_end - block_begin

block_rows  = batch * num_key_value_heads * block_span
key         = block_begin + local_row % block_span
rest        = local_row / block_span
kv_head     = rest % num_key_value_heads
batch       = rest / num_key_value_heads
```

This is the generic shader's own partition, so a key is owned by exactly one block
for any count, and no block writes an element another block owns. One lane still
owns exactly one key; the per-key arithmetic is byte-for-byte the production
sequence (mode 0 continuous `precise` accumulator, mode 9 per-head then head-order
combine, key-mask zeroing, `pc.scale * ds * q`, `p * grad_output`).

The structural consequence that decides the phase:

> **`total_lanes = batch * kv_heads * key_seq = 256`, independent of the block
> count.** Workgroups = `ceil(256 / lanes)`. Partitioning keys cannot add a lane,
> because there is already exactly one lane per destination key. At a fixed width
> it only *shrinks* each workgroup's lane span and adds a dispatch per block; at a
> smaller width it keeps the same 256 lanes and raises the workgroup count.

That is why the width is a compile-time variant of the same source
(`KEY_WORKGROUP_LANES`, default 64 = production): it is the only way to change the
workgroup count at this geometry without changing the lane total or the arithmetic.

---

## 4. Candidate configurations

Ten, all full-utilization at the primary geometry (`blocks * span * 4` rows divides
evenly into `lanes`-wide workgroups, so no lane is created that owns nothing):

| blocks | lanes | rows per block | workgroups per block | **total workgroups** | lane slots | useful lanes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 256 | 4 | **4** | 256 | 256 |
| 2 | 64 | 128 | 2 | **4** | 256 | 256 |
| 4 | 64 | 64 | 1 | **4** | 256 | 256 |
| 8 | 64 | 32 | 1 | **8** | 512 | 256 |
| 1 | 32 | 256 | 8 | **8** | 256 | 256 |
| 1 | 16 | 256 | 16 | **16** | 256 | 256 |
| 1 | 8 | 256 | 32 | **32** | 256 | 256 |
| 2 | 32 | 128 | 4 | **8** | 256 | 256 |
| 4 | 16 | 64 | 4 | **16** | 256 | 256 |
| 8 | 8 | 32 | 4 | **32** | 256 | 256 |

Note the 8-block/64-lane entry: it is the only configuration that both raises the
workgroup count and wastes lanes (512 slots for 256 useful), which is the "generating
workgroups that do no useful work" case §4 asked to be avoided unless measurement
justifies it. It is retained only because it is the 8-block member of the requested
series, and it is the worst performer.

No tail exists at `key_value_seq_len = 64` for any of these counts; the shader
still handles the general case explicitly (`block_span == 0` returns immediately, so
a count above the sequence dispatches nothing and writes nothing).

---

## 5. Numerical qualification — first, and before any timing

`raw/parity.log`. Every candidate was compared against **both** the scalar oracle
(the production generic backward) and the production two-pass key pass, in both
arithmetic modes, on every production-qualified grouping plus the MHA control.

```
[phase-viii-parity-summary] status=PASS cases=100 tensors=300
  elements=27525120 failures=0 ownership_passes=50 missing=0 overlapping=0
```

| Geometry | heads/kv_heads | modes | configs | dQ | dK | dV |
| --- | --- | --- | ---: | --- | --- | --- |
| `8q-2kv` (primary) | 8 / 2 | 0, 9 | 10 | exact | exact | exact |
| `4q-2kv` | 4 / 2 | 0, 9 | 10 | exact | exact | exact |
| `8q-4kv` | 8 / 4 | 0, 9 | 10 | exact | exact | exact |
| `16q-4kv` | 16 / 4 | 0, 9 | 10 | exact | exact | exact |
| `8q-8kv` (MHA control) | 8 / 8 | 0, 9 | 10 | exact | exact | exact |

All 300 comparisons: `bitwise == elements`, `max_abs = 0.0`, `max_ulp = 0`,
`unwritten = 0`, `first_divergence = none`. Every line is in the log; the
production pair was itself checked against the oracle on all ten geometry/mode
combinations first, so a PASS is against a verified baseline and not against
another candidate.

**dQ is unchanged.** Each candidate arm re-ran the production query pass into its
own sentinel-filled buffer and compared it bitwise: `dQ` exact in all 100 cases.
The candidate binds only `grad_k`/`grad_v` (8 bindings, same descriptor set as
production), so it cannot write `dQ`; the measurement confirms the binding set.

**Ownership, measured rather than argued** (`raw/parity.log`,
`[phase-viii-ownership]`): each block was dispatched *alone* into a fresh
sentinel-filled dK/dV pair, and the written set was read back. 50 passes
(5 geometries × 10 configs), `missing=0`, `overlapping=0` — every destination
element written exactly once, by exactly one block, for every block count.

**Stale scratch.** Each candidate arm received a freshly zeroed
probability/dS pair, and the query pass was re-run into it, so a stale read from a
previous arm could not be masked by the production pass's output.

---

## 6. MHA regression

Two independent statements:

1. The parity matrix above includes `8q-8kv` (MHA) in both modes on all ten
   configurations — exact against both baselines.
2. The production path itself still passes its own qualification with the
   production artifacts and the production gate (`raw/production-two-pass-regression.log`):

```
two-pass gradient qualification: status=PASS cases=8/8 comparisons=24/24
elements=6316128 bitwise_equal=6316128 max_abs=0.000e0 max_ulp=0 unwritten=0
scalar_dispatches=8 query_dispatches=8 key_dispatches=8
```

The fail-closed contract is also unchanged
(`raw/failclosed.log`): `status=PASS cases=8 default_query=1 default_key=1
default_scalar=0 fallback_scalar=1 explicit_zero_loads=0 gqa_loads=0
long_context_loads=0 scratch_fallbacks=0`.

---

## 7. GPU timestamp comparisons

Device: `AMD Radeon Graphics` (integrated RDNA3, `deviceID 0x15bf`), Vulkan
1.4.359, driver `2.0.416` / `26.20.15.14 (LLPC)`, `timestamp_period_ns = 10`
(`raw/environment.txt`). Numbers are the shipped profiler's own brackets
(`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, `…SAMPLE_STRIDE=1`, 25 samples per arm per
run, 3 warmups, one dispatch set per submission), parsed by `analyze_timing.py`;
`raw/timing-gpu.json` and `raw/timing-gpu-reverse.json` are its output.

Two runs of the whole arm set were made in **opposite arm order**, because the
first arm of a process carries a startup penalty. That control changes the
interpretation of one number, so it is reported first.

### Arm-order control

| Arm (8q/2kv, mode 9) | order 1 (first) | order 2 (last) | drift |
| --- | ---: | ---: | ---: |
| `key_production` | 8.0931 | 7.5965 | **−6.1 %** |
| `key_blocks1_lanes64` | 7.6136 | 7.6036 | −0.1 % |
| `key_blocks1_lanes32` | 7.5062 | 7.4637 | −0.6 % |
| `key_blocks2_lanes64` | 11.5568 | 11.5575 | +0.0 % |
| `key_blocks4_lanes64` | 19.4190 | 19.3798 | −0.2 % |
| `key_blocks8_lanes64` | 34.7058 | 34.6848 | −0.1 % |
| `query_reference` | 3.6066 | 3.6101 | +0.1 % |
| `combined_production` | 11.2372 | 11.2691 | +0.3 % |

Every arm reproduces within ±0.6 % except the production arm when it is measured
first, whose distribution carried a 14.5 ms outlier. **The steady-state production
key pass is therefore 7.60 ms, not 8.09 ms**, and it is then indistinguishable from
the exact-equivalent candidate (`blocks=1, lanes=64`). That equivalence is the
experiment's control: candidate and production differ by ≤0.1 % in time while
being bit-identical, so every other difference in this table is attributable to the
partition or the width, and not to the candidate being a different kernel.

### 8q/2kv — GPU ms per key dispatch (mode 9, steady state)

| arm | blocks | lanes | workgroups | order 1 | order 2 | steady | vs production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `key_production` | n/a | 64 | 4 | 8.0931 | 7.5965 | **7.5965** | 1.000× |
| `key_blocks1_lanes64` | 1 | 64 | 4 | 7.6136 | 7.6036 | 7.6036 | 0.999× |
| `key_blocks1_lanes32` | 1 | 32 | 8 | 7.5062 | 7.4637 | 7.4637 | **1.018×** |
| `key_blocks1_lanes16` | 1 | 16 | 16 | 7.4991 | 7.4826 | 7.4826 | 1.015× |
| `key_blocks1_lanes8` | 1 | 8 | 32 | 7.7909 | 7.7759 | 7.7759 | 0.977× |
| `key_blocks2_lanes64` | 2 | 64 | 4 | 11.5568 | 11.5575 | 11.5575 | **0.657×** |
| `key_blocks2_lanes32` | 2 | 32 | 8 | 11.5164 | 11.5306 | 11.5306 | 0.659× |
| `key_blocks4_lanes64` | 4 | 64 | 4 | 19.4190 | 19.3798 | 19.3798 | **0.392×** |
| `key_blocks4_lanes16` | 4 | 16 | 16 | 19.5410 | 19.4908 | 19.4908 | 0.390× |
| `key_blocks8_lanes64` | 8 | 64 | 8 | 34.7058 | 34.6848 | 34.6848 | **0.219×** |
| `key_blocks8_lanes8` | 8 | 8 | 32 | 35.4092 | 35.3637 | 35.3637 | 0.215× |
| `query_reference` | n/a | n/a | 16 | 3.6066 | 3.6101 | 3.6101 | 2.104× |
| `combined_production` | n/a | 64 | 4 | 11.2372 | 11.2691 | 11.2691 | — |
| `combined_blocks1_lanes64` | 1 | 64 | 4 | 11.2912 | 11.2620 | 11.2620 | — |
| `combined_blocks4_lanes64` | 4 | 64 | 4 | 22.9016 | 22.9206 | 22.9206 | — |

Sample spread within an arm is small (e.g. `key_blocks1_lanes64` min 7.528 /
max 7.662 over 25 samples); the medians above are stable to ±0.6 % across the two
processes, so the ordering between arms is resolvable at about that scale.

### 4q/2kv — GPU ms per key dispatch (mode 9, steady state)

| arm | blocks | lanes | workgroups | steady | vs production |
| --- | ---: | ---: | ---: | ---: | ---: |
| `key_production` | n/a | 64 | 4 | 3.8079 | 1.000× |
| `key_blocks1_lanes64` | 1 | 64 | 4 | 3.8363 | 0.993× |
| `key_blocks1_lanes32` | 1 | 32 | 8 | 3.7680 | 1.011× |
| `key_blocks1_lanes16` | 1 | 16 | 16 | 3.7772 | 1.008× |
| `key_blocks1_lanes8` | 1 | 8 | 32 | 3.9622 | 0.961× |
| `key_blocks2_lanes64` | 2 | 64 | 4 | 5.8442 | 0.652× |
| `key_blocks4_lanes64` | 4 | 64 | 4 | 9.7471 | 0.391× |
| `key_blocks8_lanes64` | 8 | 64 | 8 | 17.3456 | 0.220× |
| `query_reference` | n/a | n/a | 8 | 3.2951 | 1.156× |
| `combined_production` | n/a | 64 | 4 | 7.0894 | — |

---

## 8. What the numbers say

### 8.1 Partitioning costs time, in proportion to block count

Each additional block dispatch inside one submission adds roughly **3.9–4.9 ms** of
GPU time at 8q/2kv, so a quarter of the work costs 64 % of the whole pass:

| blocks | dispatches | measured | ideal if free | per extra dispatch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 7.60 | 7.60 | — |
| 2 | 2 | 11.56 | 7.60 | +3.96 |
| 4 | 4 | 19.38 | 7.60 | +3.93 |
| 8 | 8 | 34.68 | 7.60 | +3.87 |

The same shape appears at 4q/2kv (+2.0 ms, +2.0 ms, +1.9 ms per extra dispatch).
This is the dispatch-boundary cost the generic shader's own comment already
recorded as the reason its key-range split exists for the watchdog and not for
speed. The experiment independently reproduces it in a different shader.

### 8.2 The workgroup count is not the binding constraint

Holding the lane total at 256 and changing only the width, i.e. 4 → 8 → 16 → 32
workgroups:

| workgroups | lanes | steady ms | vs 4 workgroups |
| ---: | ---: | ---: | ---: |
| 4 | 64 | 7.6036 | 1.000× |
| 8 | 32 | 7.4637 | 1.018× |
| 16 | 16 | 7.4826 | 1.015× |
| 32 | 8 | 7.7759 | 0.977× |

An **8× increase in workgroups buys 1.8 %**, and the next doubling gives it back.
The hypothesis that the key pass is workgroup-starved — the Phase VII §8 reading of
"4 workgroups versus 16" — is therefore not supported: four workgroups of 64 lanes
already saturate whatever this kernel is limited by, and spreading the same 256
lanes over more workgroups neither helps nor hurts materially.

### 8.3 What it *is* limited by: per-lane sequential work

The same configuration (`blocks=1, lanes=64`, 4 workgroups, 256 lanes) at two
geometries whose only difference is the number of query heads mapped to each KV
head:

| geometry | heads per KV head | accumulation steps per element per lane | workgroups | steady ms |
| --- | ---: | ---: | ---: | ---: |
| `4q-2kv` | 2 | 128 | 4 | 3.8363 |
| `8q-2kv` | 4 | 256 | 4 | 7.6036 |

Time scales **1.982× for 2.00× the per-lane iteration count**, at an identical
workgroup count and identical lane count. The kernel's cost is its per-lane
sequential accumulation, not its parallel slack. That is also why the query pass —
one accumulating step per lane at 64 steps, versus the key pass's 64 steps at
256 — is 2.1× faster while doing comparable total element-work with 4× the lanes.

### 8.4 Where the per-lane cost comes from (evidence, not a hardware counter)

The key shader declares **seven** function-scope arrays of 128 floats (dK, dV and
the mode-9 per-head pair, in both mode branches): 3,584 bytes of per-lane state,
against 256 32-bit VGPRs = 1,024 bytes per lane on RDNA3
(`raw/spirv-analysis.txt`). At least part of that state cannot be register-resident,
which is consistent with the measured linear-in-iterations, insensitive-to-occupancy
behaviour. This is arithmetic on the SPIR-V. No VGPR, occupancy, ISA or
hardware-counter measurement was available here and none is claimed.

---

## 9. Complete-step ABBA benchmark

**Not performed, and that is the honest outcome rather than an omission.** §8 says
only the selected winning variant needs a full-step benchmark, and §9.7 requires a
candidate to improve the complete step. There is no winning variant:

* Every partitioning candidate is 1.52×–4.57× slower than production, so none can
  improve the step.
* The only arm anywhere near a win is `blocks=1, lanes=32` at **+1.8 %** on the key
  pass. That is not the partitioning hypothesis — it is the same single-dispatch
  kernel with 8 workgroups of 32 lanes instead of 4 of 64 — and its projected
  complete-step effect is arithmetic, not a measurement: the key pass is 2 × 7.60 =
  15.20 ms of the Phase VII 53.072 ms step, so 1.8 % of it is **0.27 ms, or 0.5 %
  of the step**. That is below this phase's own bar, and measuring it end to end
  would require putting the candidate into the release trainer, which §4/§10 forbid
  in the initial experimental phase (the candidate is test-build only by design).
* The Phase VII 8q/2kv ABBA result (19.7615×, CI [19.6379, 19.8850]) therefore
  stands unmodified, and no fresh matched baseline/candidate pair exists because
  there was no candidate to pair.

### Updated attention attribution (measured, this session, mode 9)

Using the steady-state per-dispatch GPU numbers above and the Phase VII
per-step structure (2 layers, 2 dispatches per layer):

| Bucket | Phase VII (ms/step) | This session (ms/step) | Note |
| --- | ---: | ---: | --- |
| Key pass | 15.956 | **15.20** | 2 × 7.60, re-measured |
| Query pass | 7.629 | **7.22** | 2 × 3.61, re-measured |
| Attention backward, both passes | 23.585 | **22.42** | −4.9 % vs Phase VII |
| Other attention (forward, RoPE, elementwise) | 15.747 | 15.75 (not re-measured) | unchanged |
| Dense projections | 13.87 | 13.87 (not re-measured) | unchanged |

The key pass is still the single largest kernel and still 2.1× the query pass, so
the *attribution* in Phase VII stands. What changes is the **diagnosis**: the gap is
per-lane accumulation cost, not workgroup starvation. Any future attempt at it must
reduce per-lane state or per-lane iterations, not add workgroups.

---

## 10. Attention-forward opportunity assessment (short; no implementation)

**Defect.** The generic forward (`shaders/transformer_attention_forward.comp`) is
`layout(local_size_x = 1)` and dispatches `[query_seq, num_heads, batch]` =
64 × 8 × 2 = **1024 workgroups of one lane each** at the Phase VII geometry. Every
wave64 issue carries one useful lane and 63 idle ones. This is not an inference:
the short-attention specialization's header states it directly — *"The generic
transformer_attention_forward kernel launches one invocation per (batch, head,
query) row with local_size_x = 1. On RDNA3 that leaves 63 of the 64 lanes in every
wave idle."*

**It is not a workgroup-count problem either.** Forward costs 15.599 ms/step over 2
dispatches = 7.80 ms per dispatch — essentially the same as the key pass's 7.60 ms,
with **4× the workgroups and 1/4 the lanes**. This phase measured that workgroup
count moves the key pass by ≤1.8 % across an 8× range; forward already has 1024
workgroups and is not fast. "Add more workgroups" is therefore the wrong lever for
both kernels, and the forward opportunity should not be framed as one.

**The structural precedent exists and is already shipped.** 
`shaders/transformer_attention_short_amd_rdna3.comp` is exactly the row-parallel
packing: `local_size_x = 64`, 64 consecutive rows per wave64 workgroup, bit-exact by
construction because no row reads another row's scores. It is gated to
`query_seq_len <= 16 && key_value_seq_len <= 16` (plus no sink/ALiBi/relative/
sparse/compressed/paged/window/softcap/dropout, `value_scale == 1`, `unfused_qk == 0`),
so the Phase VII workload at seq 64 is *outside* it.

**Why it does not simply scale to 64.** The generic forward keeps one row's scores
in `shared float row_cache[1024]` per workgroup. At 64 lanes per workgroup that
would be 256 KB of LDS against a 64 KB budget, so a row-parallel forward at seq 64
needs either a streaming two-pass row softmax (recompute scores) or a per-lane
private score cache (64 floats ≈ 64 VGPRs) plus the 128-float output accumulator —
the same per-lane state pressure that binds the key pass.

**Comparison of the next experiment against the measured key-pass opportunity.**

| | Key pass (measured here) | Attention forward (assessed) |
| --- | --- | --- |
| Per-step GPU | 15.20 ms (largest kernel) | 15.60 ms (tied) |
| Workgroups at 8q/2kv | 4 | 1024 |
| Lane utilisation per wave | full (64/64) | 1/64 |
| Measured effect of more workgroups | ≤1.8 % over an 8× range | n/a (already 1024) |
| Plausible lever | per-lane state / iterations | lane packing + row-softmax streaming |
| Precedent | none in tree | `…_attention_short_amd_rdna3.comp` (seq ≤ 16) |
| Risk | arithmetic order (already proven exact) | softmax path is the save-file oracle for mode 9 |
| Ceiling if it reached the query pass's per-lane rate | ~7.6 → n/a (already there) | ~7.8 → ~3.6 ms/dispatch, ~7 ms/step |

The forward experiment is the better next target **if** it is scoped as a lane-packing
and per-row-state problem, not as a parallelism problem — its defect is 63 idle lanes
per issue, which is a bigger structural waste than anything measured on the key pass.
It is explicitly out of scope here.

---

## 11. GO / NO-GO

**NO-GO for production promotion.** The production key shader, the selector, the
eligibility predicate, the experimental gate, the default route and both production
SPIR-V artifacts are unchanged, and the candidate is not linked into a non-test
build.

Reasons, each measured:

1. Partitioning the destination key sequence makes the key pass **1.52×–4.57×
   slower** (2, 4 and 8 blocks), because every added block dispatch costs ~3.9 ms at
   this geometry and the work itself does not shrink.
2. The key pass **already owns one lane per destination key**, so partitioning
   cannot add lanes; workgroup count is a function of the chosen width, not of the
   partition.
3. Changing the workgroup count by 8× at a constant lane total changes time by
   **less than 2 %**, so the "parallelism starvation" reading of the 4-versus-16
   workgroup observation is not supported.
4. Cost tracks **per-lane accumulation steps** (1.982× time for 2.00× steps at
   identical geometry and workgroup count), which points at per-lane state and
   per-lane work, not occupancy.
5. No candidate can satisfy §9.6–9.7, so no complete-step benchmark could have
   produced a positive result.

The numerical half of the hypothesis was correct and is now proven rather than
assumed: key blocks *can* own disjoint destination keys with zero missing, zero
overlapping and bit-identical dK/dV in both modes, including MHA. The optimization
fails on performance, not on correctness.

---

## 12. Exact remaining limitations

* **No complete-step measurement.** No candidate improved the kernel, so no ABBA
  step run exists for this phase; the Phase VII 19.7615× result is untouched but
  was not re-measured.
* **Timing covers `unfused_qk == 9` only.** Mode 0 is numerically qualified on all
  ten configurations but not timed; the training route at this fixture is mode 9.
* **One machine, one driver, one session, integrated GPU.** `AMD Radeon Graphics`
  (`0x15bf`), driver `2.0.416`/LLPC. Absolute times are not transferable; the
  ±0.6 % arm reproducibility is this machine's, not a general claim.
* **Profiler overhead is present in every arm.** All arms were measured with
  `STRIDE=1`, which the Phase VII receipt prices at ~9.5 ms/step of wall time and
  which brackets every dispatch. It is uniform across arms, but the per-dispatch
  cost in §8.1 is measured *with* instrumentation and may differ unprofiled;
  the ordering between arms is what this receipt claims, plus the fact that
  partitioning adds dispatches.
* **Debug test binary.** The harness ran from the unoptimized test build (the same
  one the Phase VII fail-closed check used). GPU artifacts and timestamps are the
  shipping ones; host submission overhead is not representative.
* **No VGPR / occupancy / ISA / hardware-counter data.** None of that tooling is
  exposed here, so §8.4 is arithmetic on SPIR-V and an inference, not a counter
  reading.
* **Geometry coverage is the qualified matrix only**: `head_dim = 128`,
  `q_seq = kv_seq = 64`, `causal = 1`, all-ones masks, `value_scale = 1`,
  `batch = 2`, and the four qualified groupings plus `8q/8kv`. Nothing here covers
  `kv_seq < q_seq`, cross-attention, `causal = 0`, masked GQA, paged/compressed KV,
  bf16/fp16 or other head dimensions.
* **Block counts are 1/2/4/8 only**, and the candidate is test-build only; it has
  no production loader, no manifest entry and no selector.
* **The candidate module is labelled `unknown` by the profiler** (it is not in the
  shader-name registry). Its time is still counted — the analyzer preserves the
  label so the substitution is visible.

---

## 13. Receipt contents

```
validation/phase_viii_key_blocks/
  README.md                    this document
  summary.json                 every measured number, machine-readable
  assemble.py                  builds summary.json and the tables from the raw logs
  analyze_timing.py            attributes per-submission GPU time from profiler output
  reproduce.sh                 rebuilds the shaders and reruns every check
  shaders/
    transformer_attention_backward_key_blockexp_rdna3.comp      candidate source
    transformer_attention_backward_key_blockexp_l64_rdna3.spv   KEY_WORKGROUP_LANES=64
    transformer_attention_backward_key_blockexp_l32_rdna3.spv   =32
    transformer_attention_backward_key_blockexp_l16_rdna3.spv   =16
    transformer_attention_backward_key_blockexp_l8_rdna3.spv    =8
  raw/
    git-identity-preserved.txt     pre-edit branch/HEAD/diff/stash, receipt counts
    git-identity-end-of-phase.txt  end-of-phase diff, production shader hashes
    git-status.txt  git-diffstat.txt
    source-hashes.txt              production shaders, sources, harness
    shader-provenance.txt          exact glslc command lines and output hashes
    spirv-analysis.txt             per-lane array counts, local sizes, arithmetic
    environment.txt                GPU, driver, Vulkan, toolchain
    parity.log                     100 cases, 300 comparisons, 50 ownership passes
    production-two-pass-regression.log   production MHA/two-pass qualification
    failclosed.log                 production fail-closed contract
    timing-8q2kv-4q2kv.log         order 1, every arm, 25 samples each, profiled
    timing-reverse.log             order 2 (reversed), same arms
    timing-gpu.json                analyzer output, order 1
    timing-gpu-reverse.json        analyzer output, order 2
```

The candidate source and all four SPIR-V builds are in this directory and nowhere
else; `shaders/` is byte-identical to Phase VII. The compiler is
`glslc` from VulkanSDK 1.4.321.1 (`shaderc v2023.8 v2025.3`), invoked exactly as the
manifest does:

```
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
  -DKEY_WORKGROUP_LANES=<N>  transformer_attention_backward_key_blockexp_rdna3.comp \
  -o transformer_attention_backward_key_blockexp_l<N>_rdna3.spv
spirv-val --target-env vulkan1.2 <output>.spv     # all four: OK
```

## Reproduction

```bash
cd hierarchos-vulkan

# 1. Candidate artifacts (from the validation directory, so shaders/ is untouched).
OUT=validation/phase_viii_key_blocks/shaders
for L in 64 32 16 8; do
  glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
    -DKEY_WORKGROUP_LANES=$L $OUT/transformer_attention_backward_key_blockexp_rdna3.comp \
    -o $OUT/transformer_attention_backward_key_blockexp_l${L}_rdna3.spv
  spirv-val --target-env=vulkan1.2 $OUT/transformer_attention_backward_key_blockexp_l${L}_rdna3.spv
done

# 2. Test build (the candidate is #[cfg(test)] only).
cargo test --offline --lib phase_viii --no-run

# 3. Numerical parity + key ownership across both modes and all five geometries.
./target/debug/deps/hierarchos_vulkan-*.exe phase_viii_key_block_parity \
  --nocapture --test-threads=1 > validation/phase_viii_key_blocks/raw/parity.log 2>&1

# 4. GPU timestamps, both arm orders.
for order in "" "HIERARCHOS_PHASE_VIII_REVERSE=1"; do
  env $order HIERARCHOS_PHASE_VIII_KERNEL_TIMING=1 \
      HIERARCHOS_PHASE_VIII_KERNEL_ITERATIONS=25 \
      HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
      HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=64 \
      ./target/debug/deps/hierarchos_vulkan-*.exe phase_viii_key_block_timing \
      --nocapture --test-threads=1 > validation/phase_viii_key_blocks/raw/timing-$([ -n "$order" ] && echo reverse || echo 8q2kv-4q2kv).log 2>&1
done
python validation/phase_viii_key_blocks/analyze_timing.py \
  validation/phase_viii_key_blocks/raw/timing-8q2kv-4q2kv.log \
  > validation/phase_viii_key_blocks/raw/timing-gpu.json
python validation/phase_viii_key_blocks/analyze_timing.py \
  validation/phase_viii_key_blocks/raw/timing-reverse.log \
  > validation/phase_viii_key_blocks/raw/timing-gpu-reverse.json

# 5. Production regression: the pair still matches the oracle, and still fails closed.
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 ./target/debug/deps/hierarchos_vulkan-*.exe \
  two_pass_attention_backward_matches_scalar_gradients --nocapture --test-threads=1
./target/debug/deps/hierarchos_vulkan-*.exe \
  optional_two_pass_load_failure_keeps_scalar_execution_and_gradients --nocapture --test-threads=1

# 6. Rebuild summary.json and the tables.
python validation/phase_viii_key_blocks/assemble.py
```

Or simply `bash validation/phase_viii_key_blocks/reproduce.sh`.
