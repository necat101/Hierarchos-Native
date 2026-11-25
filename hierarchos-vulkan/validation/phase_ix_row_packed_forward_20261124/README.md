# Phase IX — RDNA3 wave64 row-packed Transformer attention forward

**Date:** 2026-11-24
**Base commit:** `560809eead186f832344c15d1a9b70795da3b75d` (matches the expected base)
**Verdict: GO.** One lane per row, packed 64 rows to a wave, is bit-exact end to
end and **3.98× faster** at the attention-forward kernel and **1.22× faster** per
complete training step (release harness, five-round ABBA, CI [1.2222, 1.2251]).
All ten promotion criteria are met. The candidate was **not promoted** — the
brief reserves that for a separate change, and installing it needs a new
production eligibility predicate, because the per-lane score array bounds the
candidate at 64 keys.

No production dispatch, selector, eligibility predicate, shader or default route
was changed. Nothing was committed, pushed, merged or released.

---

## 1. Preservation evidence for Phases III–VIII

Captured before the first Phase IX write (`raw/00_preservation.txt`) and again at
the end (`raw/05_identity-end-of-phase.txt`).

| Check | Before | After |
| --- | --- | --- |
| `git rev-parse HEAD` | `560809eead186f832344c15d1a9b70795da3b75d` | same |
| Branch | `main` | `main` |
| `git status --short` | 13 modified tracked + `phase_viii_key_block_tests.rs` + four receipt dirs | same 13 + one new receipt dir + one new harness file |
| `git diff \| sha256sum` | `5cf7dfbad85ff5f5051456a0a30306a40377f2dd8fa6441443d6a3a28b01357b` | `2aa3f3c01effff09e3eb8dd73e5688b39e2f3faa8168bad7964e4c31e2d479c1` |
| `git diff \| wc -c` | 148431 | 148767 |
| `git diff --stat` | 13 files, 2817 insertions(+), 202 deletions(-) | 13 files, **2821** insertions(+), 202 deletions(-) |
| `git diff --cached \| sha256sum` | `e3b0c442…b855` (empty index) | `e3b0c442…b855` (still empty) |
| `git stash list` | `stash@{0}: On main: pre-v0.2.6 untracked validation evidence` | unchanged |
| `git rev-parse stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` | `831f8927…` unchanged |

The tracked diff grew by exactly **four lines**, all of them the test-module
declaration at the end of `src/transformer.rs`:

```rust
#[cfg(test)]
#[path = "phase_ix_row_packed_forward_tests.rs"]
mod phase_ix_row_packed_forward_tests;
```

`git diff --name-only` is still the same 13 files. No reset, clean, rebase,
checkout, discard, commit, push, merge or release was performed. The four prior
receipts plus `phase_viii_key_blocks/` were read-only throughout; the Phase VIII
key-block result stays a NO-GO and its candidate was not reopened.

### Production shader identities — byte-identical before and after

| File | sha256 |
| --- | --- |
| `shaders/transformer_attention_forward.comp` | `79b3fcd5ab16c11af8b9c6d52992672bce1c7517e1813c132446f7ab558d4cc3` |
| `shaders/transformer_attention_forward.spv` | `02cbbd03d82b9097dc25fdb066b69001b0abf3295624e1a966c9a61c870d15ec` |
| `shaders/transformer_attention_short_amd_rdna3.comp` | `3e79717824913102d002430acedb5de33a9fc7baa2d4cf1cd9a9e385852fd4f4` |
| `shaders/transformer_attention_short_amd_rdna3.spv` | `34f3ecf8c11f398832db7c4ecb80ff42bd5baf282d5512e1e63d0af204fb8fd4` |
| `shaders/transformer_attention_backward.comp` | `3c8e6267e320c1f187175209e8a42e6a2658494c6bb3762b1e009cb8b7810a34` |
| `shaders/transformer_attention_backward.spv` | `e78cb2920e3253d7187c479c23bee40d4b4ab677fedeaa97f3109bf6a00dc5a8` |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `e417935459e49dee1bc56db400f321496a2c04ad05392e8b14aef40d6c92ff78` |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |

The last six are the Phase VII/VIII identities, unchanged.

---

## 2. The generic attention-forward execution, as it actually runs

`hierarchos-vulkan/shaders/transformer_attention_forward.comp`, `local_size_x = 1`.

**Workgroup → coordinate map.** The host dispatches `[query_seq_len, num_heads,
batch_size]` (`record_text_attention_dispatch` passes the geometry it was given;
`src/transformer.rs` builds it as `[seq, heads, batch]`). The shader reads

```
query = gl_GlobalInvocationID.x   head = .y   batch = .z
```

so workgroup `(q, h, b)` owns exactly row `(b, h, q)`. At 8q/2kv b=2 seq=64 this
is **1024 workgroups of one lane**: 1024 lanes doing 1024 rows, 63 of every
wave's 64 lanes masked off.

**Per-row arithmetic, in order** (`raw/10_generic-mapping.txt` records the
source lines):

1. `query_mask[batch*query_seq + query] <= 0` → write `value_head_dim` zeros and
   return.
2. `first_key = 0`; `last_key = kv_seq-1`, or `min(query_position_offset+query,
   kv_seq-1)` when `causal != 0`. `window_size`/`compressed`/`paged` alter this
   range in features the candidate excludes.
3. **Pass 1 — max.** Ascending keys. `key_is_visible` reduces, on the plain path,
   to `key_mask[batch*kv_seq + key] > 0`. `raw_score` = one FP32 accumulator over
   the head dimension with `fma`, then `* pc.scale`; `max_score = found ?
   max(max_score, s) : s`. On the plain path with `last_key - first_key < 1024`
   the score is stored in `shared float row_cache[1024]` and a per-row visibility
   bitmap. At seq 64 that cache is always used.
4. `exp_sum` starts at 0 (no sink on the plain path). **Pass 2** walks keys
   ascending, `e = attention_exp(score - max_score)`, `exp_sum += e`, and
   overwrites the cached score with `e`.
5. `inv_sum = 1.0 / exp_sum` (plain) or `fp32_div(1.0, exp_sum)` (modes 1/9).
6. `p = e * inv_sum`, `p *= dropout_scale(...)` (identity with `dropout_p == 0`),
   written back into the cache.
7. **Pass 3** — for each output channel `d` in `0..value_head_dim`, a fresh
   accumulator over ascending keys: `value = fma(p, v * value_scale, value)`
   (plain) or materialized `precise` products with `precise` adds (modes 1/9).
8. `output_values[row*output_hidden_size + head*value_head_dim + d] = value`,
   `row = batch*query_seq + query`.

**Nothing is retained for backward.** The forward has no intermediate outputs;
the two-pass backward recomputes probabilities and dS from Q/K/masks inside its
own query pass. The candidate therefore has no intermediate-state parity
obligation — and §6 below measures that claim rather than asserting it.

**Arithmetic modes.** `unfused_qk` selects four switches, and the production
Llama path sets **9** (`src/transformer.rs`, layer attention push):

| mode | QK | P@V | exp | reciprocal |
| --- | --- | --- | --- | --- |
| 0 | `fma` reduction | `fma` | hardware `exp` | `1.0 / sum` |
| 1 | separately rounded | separately rounded | SLEEF u10 fused | `fp32_div` |
| 9 | separately rounded | separately rounded | SLEEF u10 fused | `fp32_div` |

This is the single most important fact in the phase: **a mode-0 candidate is one
ulp away from the Phase VII workload**, and the first version of this experiment
was exactly that. §6 shows what that looked like before it was fixed.

**GQA head mapping.** `queries_per_kv = num_heads / num_key_value_heads`,
`kv_head = head / queries_per_kv`; Q is read at `head * qk_head_dim`, K/V at
`kv_head * qk_head_dim`. It is a quotient, not a table, so 8q/2kv maps heads
0–3→KV 0 and 4–7→KV 1.

---

## 3. What the shipped short kernel contributes

`transformer_attention_short_amd_rdna3.comp` (`local_size_x = 64`) already does
the thing this phase wants:

```
row_index = gl_GlobalInvocationID.x
query = row_index % query_seq
head  = (row_index / query_seq) % num_heads
batch = row_index / (query_seq * num_heads)
```

which is the generic `[query, head, batch]` order flattened, and it holds a
per-lane `float scores[MAX_SHORT_SEQ]` + `bool visible[...]`. It is bit-exact
against the generic kernel for its own gate (verified by
`rdna3_short_attention_is_word_exact_and_measured`).

**Reusable as-is:** the flat row index and its inverse, the per-lane score array,
the per-lane visibility array, the "no lane reads another lane's scores" property,
and the dispatch shape `[ceil(rows/64), 1, 1]`.

**Dependent on the short-sequence assumption:** `MAX_SHORT_SEQ = 16` and every
consequence of it (a 16-entry register array is free; a 64-entry one is not), the
gate `query_seq_len <= 16`, and the absence of any cached-vs-streaming decision —
it always caches, because at 16 keys caching is trivially cheap.

Increasing `local_size_x` alone does nothing: the generic kernel's row index is
`gl_GlobalInvocationID.x`, so a 64-wide workgroup would still have exactly one
lane inside `if` and 63 doing the bounds check. The packing has to change the
*row derivation*, which is what the candidate does.

---

## 4. The proposed design

**One lane owns one row; a workgroup owns 64 consecutive rows.**

```
groups    = ceil(batch * num_heads * query_seq / 64)
workgroup = workGroupID.x + numWorkGroups.x * (workGroupID.y + numWorkGroups.y * workGroupID.z)
if (workgroup >= groups) return;                       // production grid only
row_index = workgroup * 64 + localInvocationID.x
if (row_index >= total_rows) return;                   // partial final group
query = row_index % query_seq; head = ...; batch = ...
```

The row derivation is deliberately taken from the **linearised workgroup index**
rather than `gl_GlobalInvocationID.x`, so the module is correct under two grids:

* `[ceil(rows/64), 1, 1]` — the grid a promotion would use (16 workgroups here);
* `[query_seq, num_heads, batch]` — the grid the production dispatch site
  supplies today (1024 workgroups, of which 16 do the work and 1008 return
  immediately).

That second grid matters: it means the candidate can be measured *inside the real
training step* without changing production dispatch, and §7 shows the 1008
returning workgroups are free within noise.

Inside the lane, the arithmetic is the generic kernel's cached path, operation
for operation, in the generic kernel's own order, including all four
`unfused_qk` switches through the same `fp32_sleef_exp.glsl`,
`fp32_sqrt_recip.glsl` and `fp32_fma_exact.glsl` helpers. Nothing is
reassociated, no online softmax is introduced, no row is split.

**Register/shared-memory decision.** The candidate keeps `float scores[64]` and
`bool visible[64]` per lane — the short kernel's structure with the bound raised.
That is 64 + 16 registers of private state, which the compiler places in
*Function* storage (`spirv-dis` confirms: two `OpVariable … Function` arrays),
i.e. scratch memory, exactly as the shipped short kernel's arrays are. It costs
less than the alternative: §5 shows what happened when the array was removed
entirely and the scores were recomputed instead.

---

## 5. The candidate that failed, and why it is kept

Two shaders were built from one design. The first is a **negative result that
changed the second's shape**, and both are in the receipt.

### Candidate A — streaming, no per-row state (`rowpacked64-stream`)

The generic kernel has a streaming path (`use_cache == false`) that
re-traverses the keys and recomputes each score. Extending the packing to that
path removes the per-lane array entirely: three passes, no state. It was built
first, as instructed.

It is **bit-exact** (mode 0 only; it predates the mode-9 work) and it is
**7.0× slower than the generic kernel** (43.7 ms vs 6.23 ms GPU at the mode-0
geometry).

Two causes, both structural rather than tuning:

1. Recomputing the score in the P@V pass puts the QK dot product **inside the
   output-channel loop**: `value_head_dim` (128) × keys (64) × head_dim (128) =
   1.05M multiply-adds per row, against the generic path's 16.4k — a 64× FLOP
   increase that swamps any lane-utilisation gain.
2. With 64 rows per wave the three global loads (`q`, `k`, `v`) become 64-way
   gathers where the one-lane kernel issued a single address.

The measured ratio between the two candidates — 43.6 ms vs 1.65 ms at the same
packing, same lane count, same workgroup count — is the phase's cleanest
statement that **row packing alone is not the win; row packing plus keeping the
cached score reuse is.**

### Candidate B — cached, per-lane scores (`rowpacked64-cached`)

Same packing, the generic cached contract restored, all four arithmetic modes
reproduced. This is the candidate the rest of the receipt is about.

### Source and SPIR-V identities

| Artifact | sha256 |
| --- | --- |
| `..._rowpacked64_stream_rdna3.comp` | `03cc3c0e5681d4479b7fdc423ce7d6e570c7c28968c847bc219020bed38ddd67` |
| `..._rowpacked64_stream_rdna3.spv` | `24bc8a6fc69f4a8b60b09c297d2912e5206b72c12655402d03a4c7035eeab55e` |
| `..._rowpacked64_cached_rdna3.comp` | `8f7bf06f65291793b12c81a9b0bd733a5ce656ffbd4a3689fea5fabf3439e6ed` |
| `..._rowpacked64_cached_rdna3.spv` | `c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd` |

Both were compiled with the repository's recorded glslc recipe (`shaderc 2023.8`,
the release the manifest pins for the shipped RDNA3 query pass):

```
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders SRC -o OUT
spirv-val --target-env vulkan1.2 OUT   →   both pass
```

The pinned `glslangValidator` 16.5.0/16.6.0 toolchain (`validation/install_shader_toolchain.sh`)
is **not installed in this workspace** (`.tools/` is absent), so glslc is the
available pinned compiler here; `raw/03_shader-compiler.txt` records its version.
The candidate sources and their SPIR-V live under `validation/`, never in
`shaders/`.

---

## 6. Numerical qualification

`raw/parity-forward.log`, `raw/parity-gradients.log`.

```
[phase-ix-parity-summary] status=PASS cases=21 elements=1976768 bitwise_equal=1976768 failures=0
[phase-ix-gradient-summary] status=PASS cases=9 elements=2832196 bitwise_equal=2832196 failures=0
```

Every comparison is on the raw FP32 bit pattern; `max_abs = 0.0`, `max_ulp = 0`,
`first_divergence = none` on every line, and `unwritten = 0` for every output
(checked against a `-98765.4321` sentinel written into the buffer first).

| Geometry | rows | mode | generic grid | candidate grid | output |
| --- | ---: | ---: | --- | --- | --- |
| `gqa-8q-2kv-primary-mode9` (Phase VII) | 1024 | 9 | `[64,8,2]` = 1024 wg | `[16,1,1]` = 16 wg | exact |
| `gqa-8q-2kv-primary-mode0` | 1024 | 0 | same | same | exact |
| `gqa-8q-2kv-primary-mode1` | 1024 | 1 | same | same | exact |
| `mha-8q-8kv-control` | 1024 | 9 | same | same | exact |
| `gqa-8q-2kv-noncausal` | 1024 | 9 | same | same | exact |
| `gqa-8q-2kv-masked` (keys and queries masked) | 1024 | 9 | same | same | exact |
| `mha-8q-8kv-masked` (keys masked) | 1024 | 9 | same | same | exact |
| `boundary-42rows` (b2 s7 h3 d16 causal) | 42 | 9 | `[7,3,2]` = 42 wg | `[1,1,1]`, 22 lanes idle | exact |
| `boundary-130rows` (b2 s13 h5 d16) | 130 | 9 | `[13,5,2]` = 130 wg | `[3,1,1]`, 62 idle | exact |
| `boundary-162rows-gqa` (b3 s9 h6 kv2, masked queries) | 162 | 9 | `[9,6,3]` = 162 wg | `[3,1,1]`, 30 idle | exact |

The candidate was additionally run under the **production** `[query, head,
batch]` grid on all ten geometries (`implementation=rowpacked-64-cached/production-grid`):
exact everywhere, so the 1008 early-returning workgroups change nothing.

**Dispatch proof** (printed by the harness, `raw/parity-forward.log`):

```
[phase-ix-dispatch] … implementation=generic      grid=[64, 8, 2] workgroups=1024 lanes_per_workgroup=1  lane_slots=1024 useful_rows=1024 wasted_lanes=0
[phase-ix-dispatch] … implementation=row-packed-64 grid=[16, 1, 1] workgroups=16   lanes_per_workgroup=64 lane_slots=1024 useful_rows=1024 wasted_lanes=0
```

**Boundary handling** is measured, not argued: 42 rows leaves 22 lanes idle in
the only workgroup, 130 rows leaves 62 in the third, 162 leaves 30 — and in each
case the sentinel count in the output buffer is **0**, so no row is missed and no
row is written twice. The 162-row case also masks a whole query row inside the
partial group, and its zeroed output matches.

**Downstream gradients.** The production two-pass backward (query pass then key
pass, unchanged artifacts) was run from each forward result and compared:
`dq` 131072/131072, `dk`/`dv` 32768/32768 (1024/1024 at MHA), and the pass-1
`probability` and `dS` scratch 65536/65536 and 1458/1458 — all bitwise, at both
modes 9 and 0, at MHA, and at the 162-row boundary. The production scalar
backward is the independent oracle for the same tensors and matches the two-pass
pair, so the PASS is against a qualified baseline and not against another
candidate.

The scratch comparison needed one correction worth recording: `probability`/`dS`
are allocated for the whole rectangular row × key extent, so their causal upper
triangle (32256 elements of 65536 at the primary geometry) is *structurally*
never written. The gate therefore requires both arms to leave the **same set**
unwritten and to agree bitwise on everything else, which is stronger than
comparing only the written subset.

### Complete-step digest parity

`raw/step-digest-*.log`, `raw/step-digest-comparison.json`. Six complete steps,
separate processes per arm, off the Phase VII fixture
(`target/gqa-e2e/fixture-8q2kv`, 32 sequences, batch 2, seq 64, LoRA r8 α16,
AdamW lr 1e-2 β(0.9, 0.999) ε 1e-8, seed 7):

```
status=PASS  steps_compared=6
loss_bits_mismatches=0  gradient_digest_mismatches=0  parameter_digest_mismatches=0
gradient_elements_per_step=32768  gradient_bits_compared=6291456
parameter_end_digest (both arms) = d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655
```

The first three columns of history are worth keeping: before the mode-9
arithmetic was reproduced, every step differed by one ulp of loss
(`40e44309` vs `40e4430a`) and every gradient digest differed completely — while
the *kernel-level* parity test passed, because it was feeding mode 0. That is the
phase's most transferable lesson: qualify a candidate against the push the
production path actually builds, not the push the shader's documentation
suggests.

---

## 7. GPU timestamp comparison

`raw/timing-run*-order*.log` → `raw/timing-gpu.json` → `raw/timing-stats.json`.
Five processes, alternating arm order, 25 single-dispatch submissions per arm per
process after 3 warmups, profiler brackets
(`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, `…SAMPLE_STRIDE=1`).

Device: `AMD Radeon Graphics` (integrated RDNA3), Vulkan 1.4.359, driver
2.0.416 / 26.20.15.14 (LLPC), `timestamp_period_ns = 10`.

### Primary geometry, mode 9 — GPU ms per attention-forward dispatch

| run | order | generic (1024 wg × 1) | cached packed (16 wg × 64) | cached production grid (1024 wg × 64) |
| --- | --- | ---: | ---: | ---: |
| run0 | generic first | 7.073 | **1.735** | 1.736 |
| run1 | cached first | 6.614 | 2.663 † | 3.989 † |
| run2 | generic first | 6.896 | **1.733** | 1.756 |
| run3 | cached first | 6.634 | 2.744 † | 4.065 † |
| run4 | generic first | 7.131 | **1.732** | 1.731 |

† the arm measured first in its process. The first arm carries a startup penalty
of ≈ 0.9–1.0 ms for the candidate and ≈ 0.0–0.5 ms for the generic, so the
steady-state numbers are the ones in bold. Sample spread within an arm is tiny
(cached: sd 0.004–0.013 ms over 25 samples; generic: sd 0.02–2.16, the large
values coming from runs whose sample set carried a 15 ms outlier).

### Paired speedups (per-process ratio of medians, n = 5)

| comparison | mean | 95% CI | median | worst run |
| --- | ---: | --- | ---: | ---: |
| primary mode 9, packed vs generic | **3.415** | [2.320, 4.509] | **3.979** | 2.418 |
| primary mode 9, production grid vs generic | 3.083 | [1.451, 4.714] | 3.927 | 1.632 |
| primary mode 0, packed vs generic | 3.711 | [3.552, 3.870] | 3.814 | 3.555 |
| MHA 8q/8kv mode 9, packed vs generic | 3.367 | [3.223, 3.512] | 3.443 | 3.210 |
| primary mode 0, streaming vs generic | 0.143 | [0.143, 0.143] | 0.143 | 0.143 |

The wide interval on the mode-9 rows is the arm-order effect, not measurement
noise: the two runs that put the candidate first produce 2.418/2.484, the three
that put it last produce 3.979/4.076/4.117. **The conservative statement is the
worst paired run: 2.42× faster even when the candidate pays the startup penalty
and the generic does not.** At MHA the effect is 3.21–3.46× with a tight interval,
so there is no GQA dependence.

### Where the speedup comes from

Measured, not inferred:

* **Lane utilisation, not workgroup count.** The production-grid arm launches
  1024 workgroups (1008 of which return immediately) and is indistinguishable
  from the 16-workgroup packed arm: 1.736 vs 1.735 ms steady, 3.989 vs 2.663 in
  the penalised position. Changing the workgroup count by 64× moves the kernel by
  **0.06%** in steady state. This reproduces Phase VIII's finding in a different
  kernel: workgroup count is not the lever.
* **Keeping the cached score reuse.** Same packing, same 16 workgroups, same 64
  lanes, 1.65 ms vs 43.65 ms — a 26× spread that is entirely the arithmetic
  shape. Packing is necessary and nowhere near sufficient.
* **Not occupancy.** No hardware occupancy counter was used and none is claimed.
  The lanes-per-wave change is the mechanism the mapping predicts; the ratio is
  consistent with it being worth ~4× rather than the naive 64×, which is what a
  memory-gather-bound kernel with 64-way strided K/V loads should give.

---

## 8. Complete training step (ABBA)

`raw/abba-summary.json`, `raw/abba-samples.jsonl`, `raw/abba-*.log`.

Phase III/VII protocol: **3 warmup + 26 measured steps per process**, **5 paired
ABBA rounds** (A B B A), **one arm per process**, strictly serial. Arm A =
production generic attention forward; arm B = row-packed candidate. Both arms
use the identical production two-pass GQA backward. Fixture, dataset, seed,
LoRA config and optimizer as in §6; `HIERARCHOS_WMMA_MODE=off`;
`HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1` for both arms.

Run on the **release harness** (`target/release/deps/hierarchos_vulkan-70a92090d59c92cf.exe`),
which finished linking partway through this phase; the debug-harness run that
preceded it is kept in `raw/abba-summary-debug.json` and is reported below as
corroboration.

| round | A median (ms) | B median (ms) | A / B |
| ---: | ---: | ---: | ---: |
| 0 | 53.1882 | 43.4703 | 1.22355 |
| 1 | 53.2700 | 43.4933 | 1.22479 |
| 2 | 53.3012 | 43.5503 | 1.22390 |
| 3 | 53.3201 | 43.5543 | 1.22422 |
| 4 | 53.2109 | 43.5558 | 1.22167 |

* mean paired ratio **1.2236**, 95% CI **[1.2222, 1.2251]** (df 4, t 2.776), geomean 1.2236
* ratio of medians over all 260 samples per arm: **1.2236** (A 53.2592, B 43.5271)
* A: 260 samples, mean 53.3955, sd 0.4791, p25 53.0698, p95 54.3214, min 52.6498, max 55.7997
* B: 260 samples, mean 43.6288, sd 0.4138, p25 43.3893, p95 44.3810, min 42.9578, max 46.7561
* dispatch counts: identical per step (98 dispatches, 882 over 9 profiled steps), so the arms differ only in which attention-forward artifact one dispatch resolves to
* every one of the five rounds lands between 1.2217 and 1.2248 — the tightest spread of any timing block in this phase, which is what a 9.7 ms/step saving on a 53 ms step looks like when the surrounding workload is stable
* debug harness, same protocol, for comparison: mean paired ratio 1.2132, CI [1.2087, 1.2177], A 55.4977 / B 45.7888 medians. The release build's step is only 2.2 ms shorter than the debug one here, so the two agree closely; the release figure is the authoritative one because it matches the binary Phase VII's ABBA used

**Exact training parity under the same build.** The ABBA itself runs with
`digest=false` (as Phase VII's did, so the hashing cannot pollute the timing).
Parity for the release build was established separately, with the identical
binary and protocol, at 6 complete steps per arm:

```
raw/step-digest-comparison-release.json
status=PASS  steps_compared=6
loss_bits_mismatches=0  gradient_digest_mismatches=0  parameter_digest_mismatches=0
parameter_end_digest (both arms) = d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655
```

That digest is the same value the debug-harness run ends on, so the release and
debug harnesses also agree with each other bit for bit. Loss bits at step 0 are
`40e44309` on both arms in both builds.

**Per-component GPU attribution** (`raw/step-profile-*.log`, 9 profiled steps per
arm, profiler brackets, ms per step):

| component | generic | row-packed | Δ |
| --- | ---: | ---: | ---: |
| `transformer_attention_forward` | **16.491** | **4.426** | −12.065 |
| `transformer_attention_backward_key_rdna3` | 16.286 | 16.976 | +0.690 |
| `transformer_attention_backward_query_rdna3` | 7.875 | 8.240 | +0.365 |
| `linear_forward` (+`_lane4`) | 8.057 | 8.081 | +0.024 |
| `linear_input_grad` (+group4) | 6.748 | 7.072 | +0.324 |
| `transformer_rms_norm_forward` | 1.816 | 1.883 | +0.067 |
| `transformer_rms_norm_input_grad` | 1.545 | 1.601 | +0.056 |
| `falcon_h1_cross_entropy_lanes16` | 1.278 | 1.438 | +0.160 |
| **total GPU per step** | **60.737** | **50.289** | **−10.448** |

The attention-forward dispatch drops 16.49 → 4.43 ms per step (3.73× in-model,
2 dispatches per step). The +1.7 ms of drift on the untouched components is
process-to-process, not a regression: those kernels are the same artifacts on the
same inputs, and the ABBA's paired ratios (which alternate arm order) show no
such bias.

Host-observed steps (release): 53.26 → 43.53 ms, a 9.73 ms saving that matches
the attribution run's −10.45 ms GPU saving to within the process-to-process drift
of the untouched components. The residual is not attributed and is not claimed.

---

## 9. GO / NO-GO

Against the ten promotion criteria:

| # | criterion | result |
| ---: | --- | --- |
| 1 | exact forward output parity | **PASS** — 1,976,768 / 1,976,768 words, 3 arithmetic modes, 10 geometries |
| 2 | exact relevant intermediate-state parity | **PASS** — no forward intermediates exist; backward scratch `probability`/`dS` identical including the unwritten causal triangle |
| 3 | exact downstream gradient/training parity | **PASS** — dQ/dK/dV bitwise from each forward; 6 complete steps with identical loss bits, gradient digests and parameter digest |
| 4 | no MHA regression | **PASS** — 8q/8kv exact, 3.37× faster |
| 5 | safe handling of partial row groups | **PASS** — 42 / 130 / 162 rows exact, 0 unwritten, 0 duplicated |
| 6 | statistically supported forward-kernel improvement | **PASS** — 3.98× median, 2.42× worst-case, CI [2.32, 4.51] over 5 processes |
| 7 | statistically supported complete-step improvement | **PASS** — 1.2236× CI [1.2222, 1.2251], release harness, 5 rounds × 26 steps |
| 8 | preserve short-attention specialization and scalar fallbacks | **PASS** — untouched; both byte-identical |
| 9 | no production dispatch/selector change | **PASS** — the whole candidate is `#[cfg(test)]` + `validation/` |
| 10 | release-build complete-step protocol | **PASS** — `abba-summary-release.json`, release harness, same five-round protocol as Phase VII |

**Verdict: GO.** All ten criteria are satisfied, including the two that Phase VIII
failed. The mechanism is proven: one lane per row packed into wave64, with the row
arithmetic untouched, is bit-exact end to end and cuts the attention-forward
dispatch by 3.7× in-model and the complete step by 1.2236×.

**The candidate was not promoted, and no production file was touched.** Two things
still stand between this receipt and a production change, and both are
installation work rather than evidence:

1. **Coverage.** The candidate's per-lane `float scores[64]` bounds it at 64 keys.
   Installing it needs an eligibility predicate (`query_seq_len <= 64`,
   `key_value_seq_len <= 64`, plus the plain-path field list the short module
   already uses) and a documented fallback for longer sequences. That is the
   production change the brief told this experiment not to make.
2. **Placement.** The candidate is correct under both the packed grid and the
   production `[query, head, batch]` grid (§7 shows the 1008 early-returning
   workgroups cost 0.06%), so promotion does not *require* a dispatch edit — which
   makes it a one-line selector change plus a gate, not a restructuring.

Nothing in the experiment argues against promotion. The arithmetic contract is as
strong as any prior phase's, and unlike Phase VIII the performance hypothesis held.

---

## 10. Updated attention attribution, and the next bottleneck

Per step at 8q/2kv b=2 seq=64 hd=128 with the row-packed forward (GPU ms):

| operation | ms/step | share of attention | share of step GPU |
| --- | ---: | ---: | ---: |
| attention backward — key pass | **16.976** | **56.6%** | **33.8%** |
| attention backward — query pass | 8.240 | 27.5% | 16.4% |
| attention forward | 4.426 | 14.8% | 8.8% |
| attention total | 29.642 | — | 58.9% |
| dense projections (`linear_forward` + `linear_input_grad` + grads) | 15.15 | — | 30.1% |

Before this phase the three attention pieces were 16.49 / 16.29 / 7.88; now
the **key pass is the largest single attention component by 2.1×** over the
query pass and 3.8× over the forward.

**Next dominant bottleneck: the GQA backward key pass.** Phase VIII already
falsified workgroup-count and key-block partitioning as levers there (8× more
workgroups bought 1.8%; extra blocks cost 3.9–4.9 ms each), and its own
measurement — 3.84 ms at 4q/2kv versus 7.60 ms at 8q/2kv, i.e. 1.982× for 2.00×
the per-lane iteration count — points at **per-lane sequential accumulation**,
not parallel slack. The untested hypothesis that follows from this phase is the
mirror image of the one that just succeeded: the key pass's inner 256-step
accumulation is a *reduction*, and this phase has now shown that this GPU's
lanes are the scarce resource. Splitting each destination key's accumulation
across a wave and reducing deterministically would change the accumulation
order, so it needs the same kind of bitwise gate this phase used — and it is not
implemented here.

Second: the dense projections (~15.1 ms/step, 30.1%), which is the WMMA
question, explicitly out of scope for this phase.

---

## 11. Files

| Path | What it is |
| --- | --- |
| `shaders/transformer_attention_forward_rowpacked64_stream_rdna3.{comp,spv}` | candidate A, mode 0 only, kept as the negative result |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.{comp,spv}` | candidate B, the phase's candidate |
| `../../src/phase_ix_row_packed_forward_tests.rs` | the harness (untracked, `#[cfg(test)]`) |
| `reproduce.sh` | rebuilds both candidates, reruns every test and both timings |
| `analyze_timing.py` | profiler log → per-arm GPU samples |
| `assemble.py` | per-run paired ratios and confidence intervals |
| `compare_steps.py` | complete-step loss/gradient/parameter digest comparison |
| `run_abba.py` | the complete-step ABBA protocol |
| `summarize.py` | `summary.json` |
| `raw/` | every raw output, log, sample and hash this phase produced |
| `summary.json` | the machine-readable receipt |

## 12. Reproduction

```bash
cd hierarchos-vulkan
bash validation/phase_ix_row_packed_forward_20261124/reproduce.sh
```

Individual steps:

```bash
EXE=$(ls -t target/debug/deps/hierarchos_vulkan-*.exe | head -1)
$EXE phase_ix_row_packed_forward_parity          --nocapture --test-threads=1
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 \
  $EXE phase_ix_row_packed_forward_gradient_parity --nocapture --test-threads=1
HIERARCHOS_PHASE_IX_STEP_ARM=generic|rowpacked-cached \
  HIERARCHOS_PHASE_IX_STEP_STEPS=6 HIERARCHOS_PHASE_IX_STEP_DIGEST=1 \
  $EXE phase_ix_row_packed_forward_full_step     --nocapture --test-threads=1
HIERARCHOS_PHASE_IX_KERNEL_TIMING=1 HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  $EXE phase_ix_row_packed_forward_timing        --nocapture --test-threads=1
python validation/phase_ix_row_packed_forward_20261124/run_abba.py
```

The parity and gradient tests are ordinary `cargo test` targets and need no
environment variable; the two timing tests are opt-in so a qualification run can
never be confused with a measurement run.

Nothing in this receipt is committed.
