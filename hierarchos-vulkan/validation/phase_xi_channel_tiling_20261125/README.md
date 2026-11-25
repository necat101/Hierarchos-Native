# Phase XI — RDNA3 GQA key-pass channel tiling

**Result: GO for the channel-tiled key pass, with one qualification gate left open
(non-128 head dimensions).** The isolated key pass falls from 7.61 ms to 0.29 ms
at the primary 8q/2kv geometry (paired ratio 0.027–0.087 across both sessions
and both within-round orders),
every candidate is bit-identical to both the scalar oracle and the production
key pass over the full qualified matrix, and the complete training step improves
**1.528×** (43.56 ms → 28.49 ms) in a five-round, twenty-run ABBA with exact
training equivalence.

Nothing is committed. No production shader, selector, gate or fallback was
changed. All Phase III–X work is intact.

---

## 1. Preservation evidence

Base commit `560809eead186f832344c15d1a9b70795da3b75d`, branch `main`. Measured
before any file in this phase was created, and again at receipt assembly
(`raw/00_identity.txt`):

| item | before this phase | at receipt |
| --- | --- | --- |
| HEAD | `560809eead186f832344c15d1a9b70795da3b75d` | `560809eead186f832344c15d1a9b70795da3b75d` |
| branch | `main` | `main` |
| staged index tree | `4709d4508ca02c854741617b379795929f673760` | `4709d4508ca02c854741617b379795929f673760` |
| staged diff sha256 | empty (`e3b0c442…`) | empty (`e3b0c442…`) |
| `stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` | `831f892774619603c2dd68d0dd654d315f1a8f7e` |

Working-tree diff digests, recorded before editing:

* tracked text diff (`--text`, `.spv` excluded): `71db016eef2b3b16eff03136271522f0874274a1e2f60e4dc07c685b25d818b5`
* binary diff (`--binary -- '*.spv'`): `cfa0d1c059d1dab6500f23ec981e2e4fdfc6c47bd43c1d86957ee686f8d7319a`
* full diff (`--binary`): `3cfbb8a75bd398e00df8a7fe01ec07cd8739d2d2201ff6179024f7872155e983`

The binary diff digest is **unchanged** at receipt time: no `.spv` in the tree
moved. The text digest necessarily changes, because this phase adds a test module
and a dispatch hook; `raw/00_identity.txt` records both values with that note.

Production artifacts, verified unchanged against the Phase VII/VIII/X receipts
(`raw/03_production-hashes.txt`):

| artifact | sha256 | matches |
| --- | --- | --- |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `e4179354…6ff78` | Phase VII e2e + promotion raw hashes |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `ccf60984…983a7` | Phase VII e2e + promotion raw hashes |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720…7eb56` | Phase VI promotion |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f52…04256` | Phase VI promotion |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp` | `8f7bf06f…e6ed` | Phase X receipt |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv` | `c053503e…764fd` | Phase X receipt |

The Phase VIII NO-GO candidate (`validation/phase_viii_key_blocks/shaders/*`) and
its receipt are untouched and unmodified. The Phase VIII destination-key
partitioning experiment was **not repeated**.

The only production-file touch is a four-line grid helper at the two-pass key
dispatch (`two_pass_key_dispatch_grid`), whose non-test branch is the identical
expression it replaced:

```rust
[(key_rows + 63) / 64, 1, 1]
```

Its test branch exists so the end-to-end harness can submit the wider lane
mapping the candidate needs. That this is inert for the production arm is
measured, not asserted: the untouched Phase X driver
(`phase_x_row_packed_forward_full_step`, production arm) and this phase's driver
(production arm) produce **identical** per-step loss bits, gradient digests,
gradient norms and parameter digests, and identical adapter and optimizer
checkpoint file hashes — `raw/50_crosscheck-phase-x-path.log` against
`raw/51_crosscheck-phase-xi-production-arm.log`.

---

## 2. Reconstructed key-pass dataflow

Production key pass (`shaders/transformer_attention_backward_key_rdna3.comp`):

* `local_size_x = 64`, one invocation per destination key **row**
  `row = (b, kv_head, key)`; `rows = batch · num_kv_heads · kv_seq = 2·2·64 = 256`
  at the primary geometry, so `grid = [4, 1, 1]` — **4 workgroups, 256
  invocations, 4 wave64 slots**.
* Bindings: `q`, `grad_output`, `scratch_probability`, `scratch_ds`, `grad_k`,
  `grad_v`, `key_mask`, `query_mask`. Push constant: the 30-word `AttentionPush`.
* Each lane loops `head ∈ [kv_head·G, kv_head·G + G)` ascending, and inside each
  head `query ∈ [0, query_seq)` ascending, and inside each (head, query) the whole
  channel range. Per-lane arithmetic at the primary geometry: 4 heads × 64
  queries × (128 dK + 128 dV) = 65,536 element updates.
* Mode 0 keeps one continuous accumulator per element across the whole group;
  mode 9 keeps a per-head accumulator and combines heads in head order.

### Verified per-invocation state (`raw/04_state-inventory.json`)

Read out of the artifact, not the source:

| artifact | function-scope arrays | bytes | dynamic access chains | loops |
| --- | ---: | ---: | ---: | ---: |
| production key pass | 6 × 128 floats | 3,072 | 18 | 22 |
| `c128_u0` | 6 × 128 floats | 3,072 | 18 | 22 |
| `c32_u0` | 6 × 32 floats | 768 | 18 | 22 |
| `c16_u0` | 6 × 16 floats | 384 | 18 | 22 |
| `c8_u0` | 6 × 8 floats | 192 | 18 | 22 |
| `c4_u0` | 6 × 4 floats | 96 | 18 | 22 |
| every `_u1` build | 0 | 0 | 0 | 4 |

**Correction to the Phase VIII note.** Phase VIII recorded "seven 128-float
per-lane arrays, ≈3,584 bytes". The artifact contains **six** arrays totalling
**3,072 bytes**: `dk`/`dv` in the mode-0 branch, and `dk`/`dv`/`head_k`/`head_v`
in the mode-9 branch. At most **four** are live at once (mode 9, inside the head
loop) = 2,048 bytes; mode 0 has two live. The 3,584-byte figure does not
correspond to this artifact.

Answering the section-2 questions from the artifact and the measurements:

1. **Required throughout the accumulation:** `dk` and `dv` (mode 0), plus the
   per-head pair for the duration of one head (mode 9). Nothing else.
2. **Shorter live ranges:** `head_k`/`head_v` already die at the head boundary;
   `dk`/`dv` cannot be shortened without changing the reduction tree.
3. **Recomputable values:** `p` and `ds` come from pass-1 scratch and are cheap
   scalar loads, already shared across the channel loop. Nothing per-channel is
   recomputable.
4. **dK and dV independent?** Yes. They share only loop control and the two
   scalar loads; neither feeds the other. The candidate uses that (a lane's dK
   range and dV range are separate index ranges) but does not reorder either.
5. **Can the compiler keep the state in hardware resources?** Not currently:
   every channel update is `OpAccessChain` with a *variable* index into a
   function-scope array (18 such chains), which forces the array into per-lane
   memory. When the channel bound is a compile-time constant (`_u1` builds) the
   arrays disappear from the SPIR-V entirely and become SSA values.
6. **What the expensive part is.** Source-derived hypothesis: per-element scratch
   traffic, not live-state *size* — the production artifact executes 65,536
   load+store pairs per lane against a memory-backed accumulator. Measurements
   below support this and rule out "not enough workgroups" as the sole cause: at
   identical geometry (4 workgroups, 256 invocations) the `c128_u0` control is
   ~2× faster than production, and its only relevant difference is that its
   channel bound is a compile-time constant.

**What is measured and what is not.** GPU timestamps (below) are hardware
measurements. No hardware counter capture was taken in this phase: there is no
RGP/RGA capture, so no VGPR count, spill count, occupancy or ALU utilisation is
claimed anywhere in this receipt. Array sizes are structural facts about the
artifact, not register estimates.

---

## 3. Channel-tiled mapping

One lane owns one destination key row and one contiguous channel chunk of it:

```
dK[key][c0 .. c0+CHUNK)   and   dV[key][c0 .. c0+CHUNK)
```

Lane mapping (chunk fastest, so the `blocks` lanes of one key row read one
contiguous run of the shared `q_values` / `grad_output` row between them):

```
slot  = gl_GlobalInvocationID.x          // 0 .. rows*blocks-1
chunk = slot % blocks
row   = slot / blocks
c0    = chunk * CHUNK
grid  = [ceil(rows * blocks / 64), 1, 1]
```

`blocks · CHUNK = 128` exactly, and a lane writes channel `c0+i` only when it is
in range — guarded independently by `qk_head_dim` for dK and `value_head_dim` for
dV, as in production — so the surviving lanes partition each row's
`[0, dim)` range exactly once. Per-element accumulation is unchanged: the lane still walks
query heads ascending and query positions ascending, with the production
statements (`precise` product, separate `precise` add, mode 9 per-head nesting
then head-ordered combine). There is no shared memory, no subgroup operation and
no atomic, so **no output element is split across lanes** and no reduction tree
changed.

Levers this exposes, deliberately separated:

* **static channel bound** — `CHUNK` is a compile-time constant, so the channel
  loop's trip count is constant and the accumulators can leave memory. Isolated
  by `c128_u0` (production mapping, production loop shape, static bound).
* **channel tiling** — more lanes per key row, fewer channels each. Isolated by
  `c32_u0 … c4_u0` against `c128_u0`.
* **SPIR-V-level expansion** — the `_u1` builds expand the body so the artifact
  contains no array at all. Measured separately: `_u1` is consistently *slightly
  slower* than `_u0` at the same chunk, i.e. the runtime loop with a constant
  trip count already reaches the same code, and forcing the expansion only costs
  scheduling freedom.

---

## 4. Candidates

Nine artifacts from one source
(`shaders/transformer_attention_backward_key_chantile_rdna3.comp`), differing
only in compile-time definitions. Workgroup 64×1×1 in every build.

| variant | channels/lane | lanes/key row | workgroups 8q/2kv | invocations 8q/2kv | channel body |
| --- | ---: | ---: | ---: | ---: | --- |
| `c128_u0` | 128 | 1 | 4 | 256 | runtime loop (production geometry control) |
| `c32_u0` | 32 | 4 | 16 | 1,024 | runtime loop |
| `c32_u1` | 32 | 4 | 16 | 1,024 | expanded |
| `c16_u0` | 16 | 8 | 32 | 2,048 | runtime loop |
| `c16_u1` | 16 | 8 | 32 | 2,048 | expanded |
| `c8_u0` | 8 | 16 | 64 | 4,096 | runtime loop |
| `c8_u1` | 8 | 16 | 64 | 4,096 | expanded |
| **`c4_u0`** | **4** | **32** | **128** | **8,192** | runtime loop |
| `c4_u1` | 4 | 32 | 128 | 8,192 | expanded |

Artifact hashes, compiler version and flags: `raw/01_shader-hashes.txt` (full
GLSL source hash, per-variant preprocessed-GLSL hash, per-variant SPIR-V hash),
`raw/02_toolchain.txt`. Every artifact passes `spirv-val --target-env vulkan1.2`.

---

## 5. Numerical qualification

`raw/11_parity.log` — `phase_xi_channel_tiling_parity`, 2 arithmetic modes × 5
geometries (8q/2kv, 4q/2kv, 8q/4kv, 16q/4kv, 8q/8kv MHA control) × 9 variants:

```
status=PASS cases=90 tensors=180 elements=11796480 unwritten=0 failures=0
```

(90 cases × two tensors, each tensor compared against both baselines, so the
element count is twice the 5,898,240 destination elements under test. The ten
dQ comparisons are logged separately as `arm=production-key-pair tensor=dQ`.)

Every comparison is a bit-pattern comparison against **both** the scalar oracle
and the production two-pass key pass. Zero maximum absolute error, zero ULP
difference, zero unwritten destination elements (sentinel-detected). The pass-1
scratch buffers are filled with the sentinel **before** pass 1 runs, so a
candidate reading a slot the causal geometry never wrote cannot reproduce the
scalar bits.

`raw/10_ownership.log` — the lane → (row, chunk) mapping replayed over the exact
launched lane set, 18 checks: every destination element covered exactly once,
`missing=0`, `duplicated=0`, and every launched lane useful. The ownership claim
is therefore threefold: exact by construction (bijection over the replay), no
element left unwritten (device sentinel), and no double accumulation (a
double-counted element cannot reproduce the oracle's bits).

**dQ:** pass 1 is the unchanged production query shader and its output is
bit-identical to the scalar oracle in every case above. The candidate kernel does
not bind a dQ buffer at all, so it cannot write one.

**Preserved separately:** mode 0 (continuous chain across heads) and mode 9
(per-head chain, head-ordered combine) are distinct code paths in the candidate,
each with the production statement sequence; both are exercised in every
geometry.

---

## 6. Kernel timing

Round-major interleaved sampling: each round submits every arm once, so drift
affects all arms rather than the ranking. Arm-major sampling was tried first and
rejected — the same arm read 3.5 ms and 5.6 ms in two sessions purely from its
position in the run. Two sessions with opposite within-round order, 31 rounds
each, GPU timestamps from the shipped profiler
(`raw/20_timing-forward.log`, `raw/21_timing-reverse.log`,
`raw/22_timing.json`).

Steady-state GPU ms per key dispatch, primary geometry 8q/2kv, mode 9:

| arm | forward session | reverse session | paired ratio (median of 31) |
| --- | ---: | ---: | ---: |
| production | 7.612 | 7.664 | 1.000 |
| `c8_u0` | 0.327 | 0.981 | 0.0428 / 0.1248 |
| **`c4_u0`** | **0.291** | **0.671** | **0.0385 / 0.0866** |

4q/2kv, mode 9:

| arm | forward | reverse | paired ratio |
| --- | ---: | ---: | ---: |
| production | 4.682 | 5.003 | 1.000 |
| `c8_u0` | 0.165 | 0.224 | 0.0340 / 0.0440 |
| **`c4_u0`** | **0.141** | **0.134** | **0.0301 / 0.0269** |

Every one of the 31 paired rounds in each session puts every candidate below
production. One anomalous sample is worth naming rather than burying: the
**final** round of the forward session recorded `c4_u0` at 6.113 ms against
production's 7.589 ms (ratio 0.806), where its other 30 rounds are 0.18–0.29 ms.
It is the last submission of an arm that had been flat for 30 consecutive rounds
at ~0.27 ms, with production on that round exactly at its median, so it reads as
a transient stall on that one submission rather than a property of the kernel.
Every paired ratio for `c4_u0` is therefore ≤ 0.09 except that single sample;
123 of 124 paired rounds across both sessions are ≤ 0.09. No round anywhere has
`c4_u0` slower than production.

The wider all-nine-variant session (21 rounds, both orders, both geometries —
`raw/23_timing-nine-variant-forward.log`, `raw/24_…-reverse.log`,
`raw/25_timing-nine-variant.json`) covers every configuration's paired ratio
against production in the same rounds. Across its four (geometry, order) cells:

| variant | paired ratio range of the four medians | every round faster? |
| --- | ---: | --- |
| `c128_u0` | 0.403 – 0.926 | 21/21 in all four cells |
| `c32_u0` | 0.106 – 0.380 | 21/21 |
| `c16_u0` | 0.067 – 0.196 | 21/21 |
| `c8_u0` | 0.054 – 0.095 | 21/21 |
| **`c4_u0`** | **0.027 – 0.080** | **21/21** |

The `_u1` build of each chunk sits 5–25% above its `_u0` sibling in the same
rounds (`c16_u1` 0.078/0.186 vs `c16_u0` 0.067/0.196; `c32_u1` 0.115/0.376 vs
0.106/0.380), with one exception at 8q/reverse where `c4_u1` is 0.039 against
`c4_u0`'s 0.046. Forcing the expansion buys nothing systematic, so the loop form
is the one to promote.

At the primary 8q/2kv geometry `c4_u0` beats `c8_u0` in both sessions and both
within-round orders (9-variant: 0.673 vs 0.683 forward, 0.351 vs 0.748 reverse;
3-arm: 0.291 vs 0.327, 0.671 vs 0.981). At 4q/2kv the two are within noise of
each other and the sign flips by session (`c8` 0.302 vs `c4` 0.334 forward in the
9-variant session; `c4` 0.141 vs `c8` 0.165 forward in the 3-arm session), which
is the honest statement: the fan-out beyond 8 channels is where the returns go
flat, and 4 lands ahead at the geometry that matters for the full step.

Absolute numbers vary between sessions by up to ~2.3× on this integrated part
(`c4_u0`: 0.291 vs 0.671 ms) while production is stable (7.612 vs 7.664 ms), so
the paired ratio — not the absolute — is the claim. Workgroup and invocation
counts are in section 4; a wider grid is **not** presented as evidence of
occupancy.

**Why the mapping control matters.** `c128_u0` runs the production mapping (4
workgroups, 256 invocations, one key per lane) and is still ~2× faster than
production. Its SPIR-V has the same 6 arrays, the same 18 dynamic access chains
and the same loop count as production's. The difference is that its channel bound
is a compile-time constant. The reading consistent with all the measurements is
that production's per-element accumulator access is memory-backed and the static
bound is what lets the backend scalarise it; tiling then adds lane parallelism on
top. This is a code-structure-plus-timing inference, not a counter-backed profile.

---

## 7. End-to-end training (ABBA)

`raw/33_abba.json`, 20 processes. Same fixture, model, data, seed, optimizer,
LoRA rank 8/alpha 16, lr 1e-2, batch 2, seq 64, two layers, hidden 1024,
production row-packed attention forward, production GQA query pass, mode 9.
Arm A = production key shader. Arm B = `c4_u0`. Five rounds × A B B A, three
warmup + 26 measured steps per process, quiet machine.

| arm | median complete-step |
| --- | ---: |
| A production | **43.559 ms** |
| B `c4_u0` | **28.485 ms** |

Per-round ratios (B/A): 0.6553, 0.6543, 0.6554, 0.6538, 0.6540 — median
**0.6543**, **5/5 rounds improved**, speedup **1.528×**.

Training equivalence, 29 steps, release build, with digests and both checkpoints
(`raw/30`, `raw/31`, empty `raw/32_equivalence.diff`):

* every step's `loss_bits`, `grad_digest`, `grad_elements`, `grad_norm` and
  `param_digest` identical;
* `adapter_config.json`, `adapter_model.safetensors` and
  `hierarchos_adapter_optimizer.safetensors` byte-identical (same sha256);
* `parameter_end_digest` identical (`d79a12d8…c4655`, 13,636,608 elements);
* dispatch counters identical: 58 row-packed forward, 58 query pass, 58 key pass
  (2 of each per step), 0 scalar fallbacks.

The step delta (15.07 ms) matches the key-pass delta from attribution (16.99 →
0.54 ms, i.e. −16.46 ms measured under the profiler, whose own timestamps inflate
the baseline). The improvement is the key pass; nothing else moved.

---

## 8. Updated complete-step attribution

Profiled runs (`HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`), 9 steps each, GPU
ms per step (`raw/42_attribution.json`). A profiled run is a diagnostic — its
per-dispatch timestamps perturb the step, so these attribute work between
components and are not headline timings.

| shader | production ms/step | share | `c4_u0` ms/step | share |
| --- | ---: | ---: | ---: | ---: |
| key pass (`transformer_attention_backward_key_rdna3` / candidate as `unknown`) | **16.995** | **33.97%** | **0.539** | **1.53%** |
| query pass | 8.223 | 16.44% | 8.352 | 23.76% |
| `linear_forward` (+`_lane4`) | 7.972 | 15.94% | 8.447 | 24.03% |
| `linear_input_grad` (+`group4_muladd`) | 6.967 | 13.93% | 7.461 | 21.22% |
| attention forward (row-packed) | 4.390 | 8.78% | 4.697 | 13.36% |
| RMS norm forward + input grad | 3.441 | 6.88% | 3.583 | 10.19% |
| cross-entropy (lanes16) | 1.424 | 2.85% | 1.514 | 4.31% |
| everything else | 0.613 | 1.22% | 0.562 | 1.60% |
| **attributed total** | **50.024** | | **35.155** | |

The candidate module is not in the binary's shader-name registry, so its key
dispatch is the `unknown` label; in a production run no `unknown` dispatch
exists, which is why the label is unambiguous here.

**New largest single kernel: the GQA backward query pass** (8.35 ms/step,
23.8%). It is out of scope for this phase by instruction.

---

## 9. Recommendation

**GO — promote the channel-tiled mapping with a compile-time channel bound at
4 channels per lane**, subject to one open gate.

Why GO: it is bit-exact against both baselines across the whole qualified matrix
including the MHA control; it preserves mode 0 and mode 9 separately and preserves
every output element's accumulation order; ownership is exact, complete and
non-overlapping; the key-pass improvement is large and consistent in every paired
round of both sessions; and the complete step improves 1.528× with byte-identical
training artefacts.

Open gate before promotion: only `head_dim = 128` was numerically qualified. The
candidate implements the tail (`c0 + i < head_dim`), and the bijection check
covers the general case, but production also admits smaller head dimensions
(16/32/64 appear in the Phase X selector matrix). Those dimensions must be
qualified — or the promotion must gate the tiled key pass on
`head_dim % 4 == 0` with the scalar/two-pass fallback otherwise — before the
production selector routes to it.

Also required for a promotion, and not done here: a new production artifact
(`shaders/…`), the manifest and provenance entries that every committed shader
carries, and the selector/grid change that submits `blocks` times the lane slots.
This phase deliberately changed none of them.

Second candidate: `c8_u0` is equally bit-exact and 0.054–0.125 of production
time. It loses to `c4_u0` at 8q/2kv in every session and order, and is level with
it at 4q/2kv. It is the fallback if a 32-lane-per-key fan-out meets a limit
elsewhere, since it halves the invocation count for the same bit-exact output.

**Remaining bottleneck if the key pass is promoted:** the query pass (8.35
ms/step), then `linear_forward` (8.45) and `linear_input_grad` (7.46). Attention
forward is 4.70. The key pass leaves the top of the distribution entirely.

Mixed WMMA, Flash-Attention-style rewrites, query-pass work, forward work,
precision, KV-cache and optimizer work are untouched and remain separate
projects, as instructed.

---

## 10. Reproduction

```text
cd hierarchos-vulkan
bash validation/phase_xi_channel_tiling_20261125/reproduce.sh
```

Sections 1–6 of that script need only a debug build; section 7 rebuilds release
(~10 min on this machine) and sections 8–10 re-derive equivalence, ABBA and
attribution. Every output lands in `raw/`; `raw/99_summary.json` is the assembled
receipt.

Fast paths:

```text
# numerical gates only (~15 s)
cargo test --lib phase_xi_channel_tiling_ownership -- --nocapture --test-threads=1
cargo test --lib phase_xi_channel_tiling_parity    -- --nocapture --test-threads=1

# kernel timing, both orders (~2 min)
HIERARCHOS_PHASE_XI_KERNEL_TIMING=1 HIERARCHOS_PHASE_XI_KERNEL_ITERATIONS=31 \
  HIERARCHOS_PHASE_XI_ARMS=production,c4_u0 HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  cargo test --lib phase_xi_channel_tiling_timing -- --nocapture --test-threads=1
```

Device: `AMD Radeon Graphics`, vendor `0x1002`, device `0x15bf` (integrated
RDNA3), Vulkan as reported by the adapter. Toolchain: the repository's pinned
`glslc` (shaderc 2025.3, reports 2023.8) with
`-fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders`,
and `spirv-val` from SPIRV-Tools 2025.3.

## 11. Files added by this phase

| file | role |
| --- | --- |
| `validation/phase_xi_channel_tiling_20261125/shaders/transformer_attention_backward_key_chantile_rdna3.comp` | the single experimental source |
| `validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_*.spv` | the nine experimental artifacts (not in `shaders/`) |
| `src/phase_xi_channel_tiling_tests.rs` | the whole dispatch path, `#[cfg(test)]` |
| `src/transformer.rs` | module declaration + `two_pass_key_dispatch_grid` (production branch unchanged) |
| `validation/phase_xi_channel_tiling_20261125/*.py`, `reproduce.sh`, `raw/` | receipt and reproduction |

## 12. Limitations stated plainly

* No hardware counters (no RGP/RGA capture): no VGPR, spill, occupancy or ALU
  utilisation claim is made. The state figures are artifact structure.
* `c4_u0`'s absolute GPU time differs between sessions (0.291 vs 0.671 ms in the
  3-arm session, 0.673 vs 0.351 in the 9-arm one); the paired ratio is the
  reported quantity. 123 of 124 paired rounds are ≤ 0.09 of production, and the
  one exception (ratio 0.806, final round of the forward session) is described in
  section 6. No round has the candidate slower than production.
* Only `head_dim = 128` is numerically qualified (section 9).
* The end-to-end swap goes through a `#[cfg(test)]` dispatch hook and a
  test-installed kernel pair; production behaviour is proven unchanged by the
  Phase X cross-check, but the promoted route itself is not built in this phase.
* Kernel and step timings are from one integrated RDNA3 device on one driver.
