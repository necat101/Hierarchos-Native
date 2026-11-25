# Phase VI — production promotion and qualification of grouped-query attention backward

**Date:** 2026-11-24
**Base commit:** `560809eead186f832344c15d1a9b70795da3b75d`
(Phase III/IV uncommitted, Phase V experimental evidence untracked — both preserved)
**Verdict:** **GO** for narrowly scoped production GQA dispatch, on the four measured
groupings only. MHA is unchanged bit-for-bit and did not regress.

This receipt promotes the Phase V experimental key pass into the production shader
location, re-qualifies MHA against it, converts the Phase V experiment into tests
that drive the shipped artifact, adds a second KV-head count, and widens the
selector only for the shapes that were actually measured.

---

## 1. Preserved Phase III–V working tree

Recorded before any edit of this phase, on the actual tree (not on `HEAD`):

| Check | Value |
| --- | --- |
| `git rev-parse HEAD` | `560809eead186f832344c15d1a9b70795da3b75d` |
| `git status --short` | 11 modified tracked files; `?? validation/gqa_attention_backward_experiment_20261124/`; `?? validation/mha_attention_backward_closeout_20261124/` |
| `git diff --stat` | 11 files changed, 2437 insertions(+), 162 deletions(-) |
| `git diff \| sha256sum` (start of phase) | `7f92646916053489285c27cb73bbb2a5ebf4e88a3a1e26dc7a64dbf84537e214` |
| `git diff --cached \| sha256sum` | `e3b0c442…b855` (empty index — nothing was staged) |
| untracked Phase III+V receipts, sorted `sha256sum \| sha256sum` | `522ade2f500b3a5649cc536d59457103ab64e52ceacfd59f223e1482a22970f0` |
| `git stash list` | `stash@{0}: On main: pre-v0.2.6 untracked validation evidence (preserved, not deleted)` |

Production artifact hashes at the start of this phase, all matching their Phase III
identities:

| File | sha256 |
| --- | --- |
| `shaders/transformer_attention_backward.comp` | `3c8e6267e320c1f187175209e8a42e6a2658494c6bb3762b1e009cb8b7810a34` |
| `shaders/transformer_attention_backward.spv` | `e78cb2920e3253d7187c479c23bee40d4b4ab677fedeaa97f3109bf6a00dc5a8` |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `af89d64c7cb8241f790eb0a494209fb9c9c8f5f239ae281a6812c6b4ce97b4a0` |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `5f69d500fa703549bda2175e0dc566481bc646125038ff952860a3bb7da983a7` |

Phase V candidate, verified against the receipt before promotion:

| Artifact | sha256 |
| --- | --- |
| `validation/gqa_attention_backward_experiment_20261124/shaders/transformer_attention_backward_key_gqa_rdna3.comp` | `40ed5c60a4d95ae9c5a6cec623f05fcf2950ebcd896401d71b4dceff97381c0f` |
| `…_key_gqa_rdna3.spv` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |

Both match the Phase V receipt exactly (`40ed5c60…`, `ccf60984…`), and the
superseded MHA-only key artifact was copied to `target/phase6-baseline/` before
being replaced, so the "before" timing arm in §7 is the old production bytes.

No reset, clean, rebase, checkout or commit was performed. `stash@{0}` is
untouched. The Phase V receipt directory and the Phase III closeout directory
were read only.

---

## 2. Promotion

The candidate source was copied to `shaders/transformer_attention_backward_key_rdna3.comp`
and only its status header was edited (the "EXPERIMENTAL / not reachable from
production dispatch" lines, and the "candidate derived from" wording). No
declaration, layout, push constant, loop, qualifier or expression was touched.

Rebuilt with the manifest's own flags:

```
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
      shaders/transformer_attention_backward_key_rdna3.comp \
  -o  shaders/transformer_attention_backward_key_rdna3.spv
spirv-val --target-env vulkan1.2 shaders/transformer_attention_backward_key_rdna3.spv   # OK
```

| Artifact | Before | After |
| --- | --- | --- |
| `…_key_rdna3.comp` | `af89d64c…7b4a0` | `e417935459e49dee1bc56db400f321496a2c04ad05392e8b14aef40d6c92ff78` |
| `…_key_rdna3.spv` | `5f69d500…983a7` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |

**The promoted SPIR-V is byte-identical to the Phase V tested candidate**
(`cmp` against the receipt copy succeeds). Promotion therefore introduced no new
code generation: the artifact that is dispatched in production is the artifact
whose 655,360 elements were bit-exact in Phase V, and the header edit is proven
semantically inert by that byte equality.

Unchanged, as required: scalar (`3c8e6267…` / `e78cb292…`), query pass
(`2e3f5720…` / `f9652f52…`), and the class of MHA eligibilities.

### Provenance

`validation/training_shader_manifest.json` already records this artifact with
`compiler=glslc`, `compiler_version=2023.8`, `target_env=vulkan1.2` and the exact
flags above, so no manifest edit was needed and none was made: the entry describes
the source→artifact relation, which is unchanged. Reproducibility was verified
directly for all three attention shaders (scratch rebuild, byte comparison):

| Artifact | Rebuild vs committed |
| --- | --- |
| `…_query_rdna3.spv` | identical (`f9652f52…`) |
| `…_key_rdna3.spv` | identical (`ccf60984…`) |
| `transformer_attention_backward.spv` (scalar) | identical (`e78cb292…`) |

`validation/check_shader_provenance.py` reports `wrong_version` for these entries
in this environment, **not** `drifted`: the installed `glslc` identifies itself as
`shaderc v2023.8 v2025.3`, and the checker's version heuristic reads the bundled
dependency release (`2025.3`) instead of the shaderc release (`2023.8`) the
manifest pins. It therefore declines to compile. The byte-for-byte rebuild above
settles the question the check exists to answer. Historical receipts
(`mha_attention_backward_closeout_20261124/qualification.json`,
`release_parity/…`, `phase19t-policy-overlay.json`) still record `5f69d500…`;
they are frozen records of what was true then and were deliberately not rewritten.

---

## 3. MHA requalification against the promoted artifact

Full Phase III matrix, `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1`, production pair
loaded through `resolve_two_pass_attention_backward`:

| Case | Geometries | Modes |
| --- | --- | --- |
| fixture-causal | batch 8, 8 heads / 8 KV, seq 64, head_dim 128 | 0, 9 |
| fixture-full | same, `causal=0` | 0, 9 |
| padded-causal | batch 2, 2 heads / 2 KV, seq 16, head_dim 64, masked rows | 0, 9 |
| small-causal | batch 1, 1 head / 1 KV, seq 4, head_dim 4 | 0, 9 |

```
two-pass gradient qualification: status=PASS cases=8/8 comparisons=24/24
elements=6316128 bitwise_equal=6316128 max_abs=0.000e0 max_ulp=0 unwritten=0
scalar_dispatches=8 query_dispatches=8 key_dispatches=8
```

Identical to the Phase III result: all eight cases, both modes, dQ/dK/dV
bit-identical, zero absolute error, zero ULP difference, zero unwritten elements,
eight query and eight key dispatches recorded, no scalar fallback.

### Adapter and loss parity (established MHA fixture)

`validation/two_pass_backward_equivalence.sh` against the release trainer, the
1024-hidden / 2-layer fixture produced by `validation/make_wmma_fixture.py`:

```
=== negative controls ===
negative controls: 11/11 detected
scalar arm : scalar=2 query=0 key=0
twopass arm: scalar=0 query=2 key=2
two-pass backward: route=two-pass eligible=true reason=eligible kernels=true
  scratch=true key_blocks=1 batch=8 heads=8 kv_heads=8 q_seq=64 kv_seq=64
  head_dim=128 causal=1 unfused_qk=9 scratch_elements=262144 scratch_bytes=2097152
=== adapter digests ===
scalar  adapter_config.json     1fa6572209fb7a4c5fce8f4efcfaef8a90b11af831952a76dcb76b2a5da6d696
twopass adapter_config.json     1fa6572209fb7a4c5fce8f4efcfaef8a90b11af831952a76dcb76b2a5da6d696
scalar  adapter_model.safetensors b3da10f2ca2b889e595f868131dfb99ddf5d1a5740d67cc489f7fedb9550a2d4
twopass adapter_model.safetensors b3da10f2ca2b889e595f868131dfb99ddf5d1a5740d67cc489f7fedb9550a2d4
scalar  training_report.json    287a807bb5398f13a4ea11c5a28f5cd02ca3e4025a9cd281802ac9dc0f45db37
twopass training_report.json    287a807bb5398f13a4ea11c5a28f5cd02ca3e4025a9cd281802ac9dc0f45db37
=== PASS: both passes dispatched and every adapter file is bit-identical
```

Adapter weights and the whole training report (which carries the loss trajectory)
are bit-identical across arms, the candidate arm never dispatched the scalar
backward, and eleven negative controls confirm the harness can detect a mismatch.

---

## 4. GQA tests now drive the production artifact

`transformer::tests::grouped_query_attention_two_pass_matches_scalar_gradients`
replaced the Phase V experiment. It obtains **both** passes from
`resolve_two_pass_attention_backward` — the loader production dispatch calls — so
there is no test-only SPIR-V constant left in the tree, and it asserts per case
that the selector chooses exactly that geometry.

| Grouping | Batch | Mode 0 (dQ/dK/dV) | Mode 9 (dQ/dK/dV) |
| --- | ---: | --- | --- |
| 8q / 2kv (group 4) | 2 | exact 196,608/196,608 | exact 196,608/196,608 |
| 4q / 2kv (group 2) | 2 | exact 98,304/98,304 | exact 98,304/98,304 |
| 8q / 4kv (group 2) | 2 | exact 196,608/196,608 | exact 196,608/196,608 |
| 16q / 4kv (group 4) | 2 | exact 393,216/393,216 | exact 393,216/393,216 |
| 8q / 2kv (batch variation) | 1 | exact 196,608/196,608 | exact 196,608/196,608 |
| 8q / 2kv (batch variation) | 4 | exact 196,608/196,608 | exact 196,608/196,608 |

```
gqa two-pass qualification: status=PASS cases=12/12 comparisons=36/36
elements=2949120 bitwise_equal=2949120 max_abs=0.000e0 max_ulp=0 unwritten=0
scratch_bytes=7340032 scalar_dispatches=12 query_dispatches=12 key_dispatches=12
```

Twelve cases (six shapes × two modes), thirty-six tensor comparisons,
2,949,120 FP32 elements all bit-identical, zero absolute error, zero ULP, zero
unwritten. The dispatch counts are recorded from `ComputeBatch::dispatch_count()`:
one scalar dispatch in the oracle arm, one query and one key dispatch in the pair
arm, twelve of each across the matrix — the scalar route never handled a case that
claims to qualify GQA.

### Negative controls preserved

* `gradient_report_rejects_a_one_ulp_difference` still passes, so a PASS is not
  vacuous.
* The measured tree probe is retained and is now an assertion rather than a
  diagnostic. Replaying each candidate reduction on the host from pass 1's own
  operands against the scalar buffers:

| Tree | 8q-2kv dK/dV | 4q-2kv | 8q-4kv | 16q-4kv |
| --- | --- | --- | --- | --- |
| `flat_plain` (product rounded, then added) | **512/512** | **512/512** | **512/512** | **512/512** |
| `flat_fma` | 115 / 108 | 133 / 149 | 127 / 136 | 105 / 117 |
| `two_stage_plain` (the superseded MHA-only tree) | 46 / 54 | 75 / 80 | 93 / 92 | 45 / 44 |
| `two_stage_fma` | 45 / 43 | 60 / 83 | 69 / 77 | 48 / 48 |

The test asserts `flat_plain == dims` and `two_stage_plain < dims` for mode 0 in
every case: the oracle is the flat tree the promoted shader implements, and the
reduction the superseded artifact used provably does not reproduce it. That is
what makes the mode-0 grouped result meaningful rather than a coincidence.
(The Phase V `=mha` arm depended on a test-only copy of the superseded artifact;
this control measures the same thing without carrying a second SPIR-V blob.)

### Production route, end to end

`transformer::tests::grouped_query_attention_production_route_dispatches_both_passes`
goes through `VulkanTransformerLayer::record_backward` for 8q/2kv × 128 × 64 ×
batch 2 and reads the layer's recorded counters:

```
gqa production-route qualification: status=PASS default_query=1 default_key=1
default_scalar=0 scratch_allocations=1 unmeasured_loads=0
```

Gradients match the `=0` baseline bit-for-bit. The same grouping at head dimension
8 / sequence 8 is divisible but unmeasured: `query=0 key=0 scratch_allocations=0
optional_kernel_loads=0 pair_reason=None`, with the scalar kernel dispatching —
i.e. nothing optional is even built or allocated for a rejected grouped geometry.

---

## 5. Additional KV-head coverage

Phase V covered two groupings that both used **two** KV heads, so it could not
distinguish grouped reduction from an assumption about a fixed KV-head count.
This phase adds:

* **8 query heads / 4 KV heads** (group 2, four KV heads),
* **16 query heads / 4 KV heads** (group 4, four KV heads).

Both are bit-exact in both modes (§4), and both are admitted by the selector.
Batch was varied (1 and 4) for 8q/2kv to show the allowlist does not silently
depend on the fixture's batch size.

### What this evidence does **not** cover

The grouped path was measured at exactly one head dimension (128), one symmetric
sequence shape (`q_seq = kv_seq = 64`), `causal = 1`, all-ones masks, and
`value_head_dim == qk_head_dim`. Nothing here supports:

* other head dimensions (64, 96, 256) — the loop is dimension-agnostic, but the
  `MAX_HEAD_DIM 128` register array and the accumulation order were only observed
  at 128;
* `kv_seq < q_seq`, cross-attention, or any `key_value_seq_len != query_seq_len`;
* `causal = 0` under GQA (the MHA matrix covers full attention, GQA does not);
* masked/padded rows under GQA (the MHA matrix covers them, GQA does not);
* non-unit `value_scale`, `head_dim != value_head_dim` (MLA-style value widths);
* windows, softcap, ALiBi, relative bias, dropout, sinks, sparse masks,
  compressed or paged KV — all rejected by pre-existing gate clauses.

These are the tests that would be needed before the selector could reuse the
broader MHA geometry limits: `head_dim ∈ {32, 64, 96, 128}`, `q_seq = 64` with
`kv_seq ∈ {16, 32, 64}`, `causal = 0`, masked rows, and at least one
`head_dim != value_head_dim` shape — each at two groupings and both modes.

---

## 6. The production eligibility predicate

Unchanged in substance for MHA: `num_heads == num_key_value_heads` keeps exactly
the Phase III clauses. Grouped geometries are admitted only by an explicit
allowlist:

```
num_heads % num_key_value_heads == 0                    # else invalid_geometry (pre-existing)
(num_heads, num_key_value_heads) ∈ {(8,2), (4,2), (8,4), (16,4)}
head_dim == 128
query_seq_len == 64 && key_value_seq_len == 64
causal == 1
unfused_qk ∈ {0, 9}
```

together with every pre-existing restriction: `head_dim == value_head_dim`,
`head_dim <= 128`, `kv_seq <= head_dim`, no window / softcap / ALiBi / relative
bias / dropout / sink / position offset / sparse mask / compressed keys / paged KV,
`value_scale == 1`, no scratch overflow, `key_blocks == 1`, and the environment
setting. Batch is not restricted by the predicate and was measured at 1, 2 and 4.

Rejection reasons, more specific than the old blanket `grouped_query_attention`:

| Reason | Meaning |
| --- | --- |
| `grouped_query_attention_unqualified_grouping` | divisible, but this head/KV-head pair was never measured |
| `grouped_query_attention_unqualified_head_dim` | measured grouping, head dimension is not 128 |
| `grouped_query_attention_unqualified_sequence` | measured grouping, `q_seq`/`kv_seq` is not 64/64 |
| `grouped_query_attention_unqualified_causal` | measured grouping, not causal |
| `grouped_query_attention_*` (all of the above) | scalar route, no scratch allocated |

`invalid_geometry` still covers a non-divisible head count, which is the one case
where the grouping itself is meaningless.

---

## 7. Phase IV fallback behaviour and scratch, after promotion

`two_pass_attention_setting` / lazy load / fallback are untouched. Re-run:

```
two-pass fallback qualification: status=PASS cases=8 default_query=1 default_key=1
default_scalar=0 fallback_scalar=1 explicit_zero_loads=0 gqa_loads=0
long_context_loads=0 scratch_fallbacks=0
```

which covers: `=0` dispatches scalar, allocates nothing and builds no optional
kernel; the eligible default dispatches both passes exactly once with
`optional_kernel_loads=2` and `pair_reason=ready`; query-only, key-only and
both-shader incompatibilities each fall back to scalar with zero scratch while
keeping gradients bit-identical to the `=0` baseline; explicit `=1` still fails
closed with `BLOCKED … query_shader_unavailable`; the failed load is cached across
three steps (`scalar=3`, `optional_kernel_loads=2`); and an ineligible geometry
builds nothing even with the pair deliberately broken.

Scratch unchanged: `batch * num_heads * query_seq * kv_seq` floats per buffer,
slot base keyed by **query head**, K/V base by **KV head**. The GQA matrix asserts
the derived capacity per case (`2·8·64·64`, `2·4·64·64`, `2·16·64·64`, …), the
sentinel read-back after pass 1 alone shows zero stale slots in the causal
reachable set, and the total stale count is exactly `2 * (capacity − reachable)`
in every case. Allocation stays lazy and gate-first, so a rejected grouped
geometry allocates none (§4).

---

## 8. Timing

**Device:** `AMD Radeon Graphics` (`vulkaninfo --summary`: API 1.4.359,
`driverVersion 2.0.416`, `driverInfo 26.20.15.14 (LLPC)`); the Phase III/V receipts
recorded the same GPU with the vendor-form string `32.0.32015.14001`.
**Toolchain:** rustc 1.93.0, shaderc 2023.8, SPIRV-Tools 2025.3.
**Method:** one dispatch per submission; `warmup=5`; wall-clock medians of 40
samples plus GPU-time medians of 10 samples from the shipped timestamp profiler
(`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, stride 1). Geometry 8q/8kv is the MHA
case, 8q/2kv and 4q/2kv the grouped ones, all batch 2, seq 64, head_dim 128.

### MHA, previous artifact vs promoted artifact (GPU ms, median of 10)

| Arm | Before | After | Delta |
| --- | ---: | ---: | ---: |
| scalar | 131.066 | 130.660 | −0.3 % |
| query | 3.286 | 3.267 | −0.6 % |
| **key** | **2.070** | **1.965** | **−5.0 %** |
| **combined** | **5.343** | **5.226** | **−2.2 %** |

Wall-clock medians of 40 agree where they matter: combined 6.195 → 6.194 ms
(−0.02 %). The wall-clock key-pass median moved 2.403 → 2.697 ms (+12 %), but its
p25 moved 2.313 → 2.297 ms (−0.7 %) and the GPU timestamps — which exclude
submission-side noise — moved the other way. **No MHA regression is detected**, so
separate MHA and GQA key modules were not needed; the single artifact keeps the
Phase III result.

### Grouped geometries, production artifact

GPU ms, median of 10:

| Geometry | Scalar | Query | Key | Combined | Combined vs scalar |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8q-2kv | 508.260 | 3.153 | 7.447 | 10.618 | 47.9× |
| 4q-2kv | 253.954 | 2.960 | 3.735 | 6.652 | 38.2× |

Wall ms, median of 40: 8q-2kv 509.834 / 3.960 / 8.325 / 11.482;
4q-2kv 255.335 / 3.688 / 4.674 / 7.671.

These are close to the Phase V experimental numbers (506.8 / 10.68 and
253.5 / 6.66), which is the expected result given the SPIR-V is byte-identical:
that agreement is a cross-run consistency check, not new evidence. The key pass
remains parallelism-starved at realistic KV-head counts (4 workgroups at 8q-2kv,
8 at 8q-4kv, against 16 query workgroups), which is why 8q-2kv key costs 2.4× its
query pass.

**No complete-step training claim is made for GQA.** These are kernel-level
numbers at two geometries on one machine; nothing was measured for a full
optimizer step, and the Phase III 3.2492× complete-step result (95 % CI
[3.1146, 3.3838]) is an MHA-only measurement that this phase did not repeat —
the promoted key shader is marginally faster than the artifact that produced it,
so that result is not undermined, but it is not re-measured here either.

---

## 9. GO / NO-GO

**GO** for narrowly scoped production GQA dispatch:

* the promoted artifact is byte-identical to the candidate that was bit-exact in
  Phase V, so no new code generation is being trusted;
* 36/36 tensor comparisons, 2,949,120 elements, zero ULP, zero unwritten, across
  four groupings, two KV-head counts and three batch sizes, in both modes;
* MHA is unchanged bit-for-bit against the same artifact (24/24 comparisons) and
  the key pass did not regress;
* a measured negative control shows the superseded reduction does *not* reproduce
  the oracle, so the result is not vacuous;
* the production route — not a test harness — dispatches both passes for a
  qualified grouped geometry and nothing at all for an unqualified one;
* fallback, lazy loading, `=0`, fail-closed `=1` and scratch allocation behaviour
  are all preserved.

**NO-GO** for claiming general GQA support:

* four groupings, one head dimension, one symmetric sequence shape, causal only,
  unmasked, unit value scale;
* no `kv_seq < q_seq`, no `causal = 0`, no masked rows, no bf16/fp16 path, no
  paged/compressed KV;
* no complete-step GQA measurement.

---

## 10. Remaining unsupported configurations

Still scalar, with an actionable reason and no scratch:

* any grouping not in `{(8,2), (4,2), (8,4), (16,4)}` — including 2q/1kv, 4q/1kv,
  16q/8kv, 32q/8kv, and every non-divisible head count;
* `head_dim != 128` under GQA (32, 64, 96, 256…), including
  `head_dim != value_head_dim` (MLA-style value widths, which remain
  `mismatched_head_dim` for MHA too);
* `query_seq_len != 64` or `key_value_seq_len != 64` under GQA, including
  cross-attention and any `kv_seq < q_seq`;
* `causal = 0` under GQA;
* masked/padded rows under GQA;
* `unfused_qk` outside 0/9; sliding window; softcap; ALiBi; relative bias;
  dropout; attention sink; position offset; sparse mask; compressed keys; paged
  KV; `value_scale != 1`; `kv_seq > head_dim`;
* `key_blocks != 1`; any explicit `=0`; and any geometry whose scratch would
  overflow.

---

## Reproduction

```bash
cd hierarchos-vulkan

# 1. MHA matrix against the promoted artifact
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 \
  cargo test --offline --lib transformer::tests::two_pass_attention_backward_matches_scalar_gradients \
  -- --nocapture --test-threads=1

# 2. GQA matrix (production artifact, both modes, six shapes)
cargo test --offline --lib transformer::tests::grouped_query_attention_two_pass_matches_scalar_gradients \
  -- --nocapture --test-threads=1

# 3. Production route + selector + scratch + comparator + fallback
cargo test --offline --lib grouped_query_attention -- --nocapture --test-threads=1
cargo test --offline --lib two_pass -- --nocapture --test-threads=1
cargo test --offline --lib optional_kernel -- --nocapture --test-threads=1
cargo test --offline --lib gradient_report -- --nocapture --test-threads=1

# 4. Adapter/loss parity on the established MHA fixture
cargo build --release --offline --bin hierarchos-vulkan-transformer-train \
  --bin hierarchos-vulkan-training-qualification
python validation/make_wmma_fixture.py --output target/wmma-fixture --hidden 1024 \
  --layers 2 --vocab 1024 --seq-len 64 --batch-size 8 --sequences 32
bash validation/two_pass_backward_equivalence.sh

# 5. Kernel timing, wall then GPU timestamps
HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING=1 \
  HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING_ITERATIONS=40 \
  cargo test --offline --lib transformer::tests::two_pass_attention_kernel_timing \
  -- --nocapture --test-threads=1
HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING=1 \
  HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING_ITERATIONS=10 \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  cargo test --offline --lib transformer::tests::two_pass_attention_kernel_timing \
  -- --nocapture --test-threads=1 > raw/timing-after-profiled.log 2>&1
python validation/gqa_production_promotion_20261124/analyze_gpu.py raw/timing-after-profiled.log
```

`raw/` holds the logs every table above was built from:
`parity-mha.log`, `parity-gqa.log`, `gqa-production-route.log`,
`gate-scratch-fallback.log`, `equivalence.log`, `timing-before.log`,
`timing-before-profiled.log`, `timing-after.log`, `timing-after-profiled.log`,
`environment.txt`. `analyze_gpu.py` refuses to invent a sample: a log with no
profiler block yields no number.

## Limitations

* One machine, one driver, one thermal state; `timing-before*` and `timing-after*`
  were taken in the same session so their ratio is meaningful, but the absolute
  numbers are not comparable to the Phase V receipt taken hours earlier.
* `check_shader_provenance.py` declines to compile these entries because its
  version heuristic reads `glslc`'s bundled glslang release; reproducibility was
  established by a direct rebuild instead (§2).
* The MHA complete-step benchmark was deliberately not repeated: the promoted
  artifact measured marginally faster on the key pass, so the Phase III
  complete-step result remains the applicable one and this phase adds no new
  step-level measurement in either direction.
