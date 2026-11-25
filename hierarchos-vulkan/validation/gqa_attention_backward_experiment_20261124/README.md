# Phase V — RDNA3 GQA attention-backward numerical investigation

**Date:** 2026-11-24
**Base commit:** `560809eead186f832344c15d1a9b70795da3b75d` (Phase III/IV uncommitted, preserved)
**Verdict:** **GO for eventual production GQA support**, scoped to the two measured
groupings and arithmetic modes, with one production artifact to promote and a
wider matrix to measure first.
**Not done on purpose:** the production eligibility gate still rejects GQA. Nothing
in production dispatch can select the experimental path.

This directory is an isolated experiment receipt. It adds one experimental shader
artifact, one opt-in test harness and these logs. It does **not** modify the
MHA-qualified shader pair, the production gate, the scalar kernel, or any
Transformer/RWKV/attention-forward code path.

---

## 1. Starting state and the preserved Phase III/IV identity

Recorded before any edit, on the actual working tree (not on `HEAD`):

| Check | Value |
| --- | --- |
| `git rev-parse HEAD` | `560809eead186f832344c15d1a9b70795da3b75d` |
| `git status --short` | 11 modified tracked files + `?? validation/mha_attention_backward_closeout_20261124/` |
| `git diff --stat` | 11 files changed, 1701 insertions(+), 162 deletions(-) |
| Full diff identity (`git diff \| git hash-object --stdin`) | `6192e57b814218425a75e2d825df16e0af880228` |
| `git stash list` | `stash@{0}: On main: pre-v0.2.6 untracked validation evidence (preserved, not deleted)` |
| MHA receipt directory | present, 36 files |
| Phase III recorded diff bytes | `raw/git-diff.patch`, sha256 `cf1b7a7c7cc80937a0366fbba392997dd3bcf25a6278d296057b4c5bfdeb803f` |

Working-tree source and shader hashes at start:

| File | sha256 |
| --- | --- |
| `src/transformer.rs` | `aa7b33b3ee131f58a675ae08602ce02f39525ca94cd1a025d87602b78b65f895` |
| `src/vulkan.rs` | `e1ec6dc0e91c7cf5b98d4b1354dd43d483a2716f8a14b8c5c9631fd8282356aa` |
| `src/lib.rs` | `34caa58349a8d6565f30ede09573a0135d06206c835319fe8faac4dca32ee70a` |
| `shaders/transformer_attention_backward.comp` | `3c8e6267e320c1f187175209e8a42e6a2658494c6bb3762b1e009cb8b7810a34` |
| `shaders/transformer_attention_backward.spv` | `e78cb2920e3253d7187c479c23bee40d4b4ab677fedeaa97f3109bf6a00dc5a8` |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `af89d64c7cb8241f790eb0a494209fb9c9c8f5f239ae281a6812c6b4ce97b4a0` |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `5f69d500fa703549bda2175e0dc566481bc646125038ff952860a3bb7da983a7` |

No reset, rebase, clean, checkout or commit was performed. `stash@{0}` is
untouched and no file the Phase III/IV diff touches was reverted.

### Preserved-diff verification, and what could not be re-applied

The Phase III closeout recorded its diff with plain `git diff`, so its binary
`.spv` hunk has no full index line and `git apply` refuses that one file
(`cannot apply binary patch ... without full index line`). The patch is therefore
**not** re-appliable as a patch. The tree state itself is proven instead by:

* the tracked status is the same 11 files as at the start of this phase, plus one
  new untracked directory (this one) — no Phase III/IV file was dropped;
* every file that phase recorded in `raw/source-hashes.txt` still hashes
  identically here, except `src/transformer.rs` and `src/vulkan.rs`. `vulkan.rs`
  is untouched by Phase V (Phase IV touched it after that hash list was written);
  `transformer.rs` is the one file Phase V edits, and every Phase V hunk lies
  inside `mod tests`, which begins at line 95995 — the first Phase V hunk is at
  line 136636. Production code in that file is byte-identical;
* the Phase III MHA qualification still passes bit-exactly (§8), which is the
  functional statement that its behaviour is intact.

---

## 2. Scalar reduction order (the oracle)

From `shaders/transformer_attention_backward.comp`, confirmed against the shipped
SPIR-V by measurement (§6):

1. **Query head to KV head.** One invocation owns `(batch, kv_head)`;
   `kv_head = head / queries_per_kv` with `queries_per_kv = num_heads / num_key_value_heads`.
   That invocation owns query heads `kv_head*group .. kv_head*group + group - 1`,
   ascending, and the whole group's K/V slice.
2. **dQ.** Row-owned and single-head: `grad_q` is zeroed over the head's rows and
   accumulated over keys ascending inside one `(head, query)`. No cross-head
   reduction exists, which is why dQ was already exact under GQA before any change.
3. **dK and dV, `unfused_qk == 0`.** Accumulated in the *primary* pass directly
   into the output buffers, which were zeroed once for the whole key range:

   ```
   grad_k[k_base + d] += pc.scale * softcap_derivative * ds * q_values[q_base + d];
   grad_v[v_base + d] += p * dropout * grad_output[grad_out_base + d] * pc.value_scale;
   ```

   For one output element the operand sequence is therefore **one continuous
   accumulator over the mapped query heads (ascending) and, inside each head, the
   query positions (ascending)**, starting from `+0.0`.
4. **dK and dV, `unfused_qk == 9`.** The strict adjoint (cached when
   `rows*group*seq <= 512`) reduces each repeated head **from zero** and only then
   combines the completed heads: `shared_k += head_k` in head order. A two-stage
   tree, not a flat one.
5. **Query-position accumulation.** Ascending within a head, in both paths.
6. **Intermediate precision.** Mode 0 is plain FP32: a product is rounded and then
   added. Mode 9 materialises every product and sum under `precise`, uses
   `sleef_expf_u10_fused`, and the correctly rounded `fp32_div`.
7. **`unfused_qk == 0` vs `== 9`.** Different QK and P·V dot products (fused FMA vs
   materialised products), different exp and reciprocal, and — decisively for GQA —
   two different dK/dV reduction trees. Mode 9 also carries a *second*
   softmax-dot: the primary loop's plain `softmax_dot` feeds dQ, the per-key
   adjoint's `precise kv_softmax_dot` feeds dK/dV.

Two mathematically equivalent reduction trees do not produce equal FP32 output.
That is the whole finding below.

---

## 3. The experiment's isolation

| Component | What it is |
| --- | --- |
| Candidate pass 2 | `shaders/transformer_attention_backward_key_gqa_rdna3.comp` / `.spv` in this directory — a **separate artifact**. `shaders/` is untouched. |
| Pass 1 | The production `transformer_attention_backward_query_rdna3` artifact, unchanged. Only pass 2 reduces across grouped heads. |
| Harness | `transformer::tests::gqa_two_pass_experimental_matches_scalar_gradients`, compiled into test builds only. The candidate SPIR-V is a `#[cfg(test)] const`, so no production binary carries it. |
| Activation | `HIERARCHOS_ATTENTION_BACKWARD_GQA_EXPERIMENT=1` (candidate) or `=mha` (MHA-qualified pass 2 as a negative control). Nothing runs otherwise. |
| Gate | Unchanged. `two_pass_attention_rejection` still returns `grouped_query_attention`, which the harness asserts for every geometry it drives, before and after the run. |

Ordinary Transformer execution cannot reach the candidate: the gate rejects the
geometry, and the artifact is not even linked into a non-test build.

---

## 4. First grouping: 8 query heads / 2 KV heads

`batch=2, heads=8, kv_heads=2, queries_per_kv=4, head_dim=128, q_seq=64, kv_seq=64, causal=1, blocks=1`

### Candidate (`arm=candidate`)

| Mode | Tensor | Elements | Bitwise equal | max_abs | max_ulp | Unwritten | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | dQ | 131,072 | 131,072 | 0.0 | 0 | 0 | PASS |
| 0 | dK | 32,768 | 32,768 | 0.0 | 0 | 0 | PASS |
| 0 | dV | 32,768 | 32,768 | 0.0 | 0 | 0 | PASS |
| 9 | dQ | 131,072 | 131,072 | 0.0 | 0 | 0 | PASS |
| 9 | dK | 32,768 | 32,768 | 0.0 | 0 | 0 | PASS |
| 9 | dV | 32,768 | 32,768 | 0.0 | 0 | 0 | PASS |

### Negative control: the MHA-qualified pass 2 on the same geometry (`arm=mha`)

| Mode | Tensor | Elements | Bitwise equal | max_abs | max_ulp | First divergence | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | dQ | 131,072 | 131,072 | 0.0 | 0 | none | PASS |
| 0 | dK | 32,768 | 5,858 (17.9 %) | 6.557e-7 | 53,824 | `batch=0 kv_head=0 key=0 dim=0`, 6 ULP | FAIL |
| 0 | dV | 32,768 | 5,767 (17.6 %) | 2.384e-6 | 68,480 | `batch=0 kv_head=0 key=0 dim=0`, 2 ULP | FAIL |
| 9 | dQ/dK/dV | — | all | 0.0 | 0 | none | PASS |

## 5. Second grouping: 4 query heads / 2 KV heads

`batch=2, heads=4, kv_heads=2, queries_per_kv=2, head_dim=128, q_seq=64, kv_seq=64, causal=1`

| Arm | Mode | dQ | dK | dV | max_abs | max_ulp | Unwritten |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| candidate | 0 | exact 65,536/65,536 | exact 32,768/32,768 | exact 32,768/32,768 | 0.0 | 0 | 0 |
| candidate | 9 | exact 65,536/65,536 | exact 32,768/32,768 | exact 32,768/32,768 | 0.0 | 0 | 0 |
| mha control | 0 | exact | 8,847 (27.0 %) | 8,759 (26.7 %) | 2.682e-7 / 1.431e-6 | 9,216 / 589,824 | 0 |
| mha control | 9 | exact | exact | exact | 0.0 | 0 | 0 |

### Totals

| Arm | Cases | Comparisons | Elements | Bitwise equal | max_abs | max_ulp | Unwritten | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| candidate | 4/4 | 12/12 | 655,360 | 655,360 | 0.0 | 0 | 0 | **PASS** |
| mha control | 2/4 | 12/12 | 655,360 | 524,288 | 2.384e-6 | 589,824 | 0 | FAIL |

Dispatch proof: the scalar arm recorded exactly 1 dispatch per case and the
two-pass arm exactly 1 query + 1 key dispatch per case, asserted from
`ComputeBatch::dispatch_count()`, not self-reported. The one-ULP comparator
control (`gradient_report_rejects_a_one_ulp_difference`) passes, so a PASS is not
vacuous.

**Scope of the claim.** Two groupings (4× and 2×), one head dimension (128), one
sequence shape (64/64), `causal=1`, all-ones masks, one KV-head count. This is
evidence about those geometries, not about every possible GQA layout.

---

## 6. The arithmetic-order defect and its correction

Two independent defects, both invisible when `queries_per_kv == 1` — which is
exactly why the Phase III MHA qualification passed and said nothing about GQA.

### D1 — accumulator scope (mode 0)

The MHA-qualified pass 2 resets `head_k`/`head_v` per query head and then adds the
completed heads. That is the **mode-9** tree. Mode 0's scalar path has one
continuous accumulator across the group. At one head per KV head the only
cross-head step is `0 + s_h0`, which is exact, so the two trees coincide; under
GQA they differ at every head boundary. **Correction:** mode 0 uses a single
accumulator spanning the whole group, in head-major/query-minor order.

### D2 — FMA contraction (mode 0)

With the scope fixed, parity still failed (8q dK 8,913/32,768). The cause is not
order but codegen: the driver contracts `acc + a * b` into `fma(a, b, acc)`, while
the oracle rounds the product and then adds. This was determined, not assumed. A
host replay of every candidate tree against the scalar buffers — using pass 1's
scratch, which dQ proves bit-exact — matches only one:

| Tree | 8q-2kv dK | 8q-2kv dV | 4q-2kv dK | 4q-2kv dV |
| --- | ---: | ---: | ---: | ---: |
| `flat_plain` (product rounded, then added) | **512/512** | **512/512** | **512/512** | **512/512** |
| `flat_fma` | 115/512 | 108/512 | 133/512 | 149/512 |
| `two_stage_plain` | 46/512 | 54/512 | 75/512 | 80/512 |
| `two_stage_fma` | 45/512 | 43/512 | 60/512 | 83/512 |

(512 = 4 keys × 128 dimensions.) **Correction:** the mode-0 product and the
accumulating add are both `precise`, so no contraction is possible.

At one head per KV head D2 is also invisible, because the accumulator is zero for
its only contribution: `fma(a, b, 0) == round(a*b)`.

---

## 7. Scratch indexing and the memory proof

Both passes index scratch as

```
slot = (batch * num_heads + head) * query_seq_len * key_value_seq_len
     + query * key_value_seq_len + key
```

The slot base uses the **query head**; the K/V base uses the **KV head**. Capacity
is unchanged at `batch * num_heads * query_seq_len * key_value_seq_len` elements
per buffer, and that remains exactly right for GQA because grouping changes which
*K/V* row a query head contributes to, never which scratch slot it owns.

| Geometry | Elements/buffer | Bytes/buffer | Bytes, both buffers | Reachable (causal) slots | Unreachable slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8q-2kv | 65,536 | 262,144 | 524,288 | 33,280 | 32,256 |
| 4q-2kv | 32,768 | 131,072 | 262,144 | 16,640 | 16,128 |

Measured allocation across all four cases (both modes): **1,572,864 bytes**,
asserted equal to `capacity * 2 * size_of::<f32>()` per case and summed by the
harness, matching the derived requirement exactly.

Coverage assertions, on scratch pre-filled with a sentinel and read back **after
pass 1 alone**, before pass 2 can hide anything:

* every reachable slot (`key <= query`, `causal=1`) is written — 0 stale sentinels
  in the reachable set, both buffers, both groupings, both modes;
* the stale count over both buffers is exactly `2 * (capacity - reachable)`, i.e.
  pass 1 wrote nothing outside the causal reachable set and left nothing reachable
  unwritten;
* pass 2 reads only slots pass 1 wrote — any stale read would have corrupted dK/dV
  and broken the parity result, which it does not;
* query heads mapped to the same KV head never share a slot (the slot base is the
  query head), and a group never reads outside `kv_head*group .. +group-1`.

Allocation stays lazy and gate-first: `select_two_pass_attention_backward` evaluates
`two_pass_attention_gate` first and returns the scalar route before
`two_pass_attention_scratch_elements` or `TwoPassAttentionScratch` are reached, so
a production GQA geometry allocates no two-pass scratch. The gate test and the
harness's per-geometry gate assertion both cover this.

---

## 8. MHA regression

Run with the production pair and `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1` after
all Phase V edits:

```
two-pass gradient qualification: status=PASS cases=8/8 comparisons=24/24
elements=6316128 bitwise_equal=6316128 max_abs=0.000e0 max_ulp=0 unwritten=0
scalar_dispatches=8 query_dispatches=8 key_dispatches=8
```

Identical to the Phase III result, in both modes, over all four MHA geometries. The
production artifacts are byte-identical to the Phase III record (§1). The gate,
the default, the lazy load, the scalar fallback and the fail-closed `=1` behaviour
are all untouched by this phase.

---

## 9. Experimental GQA kernel performance (parity first, and only then)

Measured only after both groupings passed in both modes.

**Device:** `AMD Radeon Graphics`, driver `32.0.32015.14001` (2026-10-27),
`timestamp_period_ns=10.0`. **Toolchain:** rustc 1.93.0, shaderc 2023.8.
**Method:** one dispatch per submission; `warmup=5`; the timestamp profiler
(`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, stride 1) reports each submission's GPU
time, and host-observed submission latency was recorded independently.

### GPU time per dispatch (ms, median of 10 profiled samples)

| Geometry | Scalar | Query | Key | Combined | Combined vs scalar |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8q-2kv | 506.785 | 3.149 | 7.441 | 10.680 | 47.4× |
| 4q-2kv | 253.528 | 2.959 | 3.745 | 6.659 | 38.1× |

### Host-observed submission latency (ms, median of 40 samples; reproduced in a second 40-sample run and a 10-sample profiled run)

| Geometry | Scalar | Query | Key | Combined |
| --- | ---: | ---: | ---: | ---: |
| 8q-2kv | 509.119 | 4.021 | 8.347 | 11.578 |
| 4q-2kv | 254.997 | 3.817 | 4.669 | 7.661 |

The two runs agree to under 1 % on every arm, and GPU time is within ~0.5 % of
wall time for the scalar arm, so the instrumentation floor is not distorting it.

**Key-pass overhead, honestly.** The key pass costs 2.4× the query pass at 8q-2kv
(7.44 vs 3.15 ms) and 1.27× at 4q-2kv. That is **not** grouped accumulation cost:
the key pass has only `batch * kv_heads * seq / 64 = 4` workgroups at 8q-2kv (2 at
4q-2kv), so it is parallelism-starved, while the query pass has 16 (8). The
grouped loop is a straight sequential sum over `group × seq` terms per element;
splitting keys across more workgroups is the obvious lever, and it is out of scope
here.

**No end-to-end training claim is made from these isolated kernel timings.** They
are single-kernel numbers at one geometry on one machine, not step-time evidence.

---

## 10. Artifact identities

| Artifact | sha256 |
| --- | --- |
| candidate `…_key_gqa_rdna3.comp` | `40ed5c60a4d95ae9c5a6cec623f05fcf2950ebcd896401d71b4dceff97381c0f` |
| candidate `…_key_gqa_rdna3.spv` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |
| production query `.comp` / `.spv` | `2e3f5720…7eb56` / `f9652f52…04256` (unchanged) |
| production key `.comp` / `.spv` | `af89d64c…7b4a0` / `5f69d500…983a7` (unchanged) |
| scalar `.comp` / `.spv` | `3c8e6267…10a34` / `e78cb292…dc5a8` (unchanged) |
| `src/transformer.rs` after Phase V | `fcca55959682b9f69e2efe73e3ffb7f755ee7fe3542d1090370b843dea22f77b` |

The candidate was built with the manifest's own flags
(`glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders`)
and validated with `spirv-val --target-env vulkan1.2`. The same `glslc` reproduces
the committed MHA key artifact byte-for-byte (`5f69d500…`), so a candidate digest
here is a statement about source, not about an unknown compiler.

---

## 11. GO / NO-GO

**GO** for eventually widening production eligibility to grouped-query attention,
on this evidence:

* both groupings bit-exact in both arithmetic modes, across dQ, dK and dV
  (655,360/655,360 elements, max_abs 0.0, max_ulp 0, zero unwritten);
* the defect is understood and corrected at the level it occurs — accumulator
  scope for mode 0, and contraction — not papered over with a tolerance;
* the MHA path is unchanged and still bit-exact;
* a negative control fails on purpose, so the PASS is not vacuous.

**NO-GO** for shipping it today, because:

* the evidence covers two groupings at one head dimension, one sequence shape,
  `causal=1`, unmasked inputs and one KV-head count;
* the production pass 2 would have to be replaced by this candidate. At
  `queries_per_kv == 1` the two are provably identical (the mode-0 accumulator is
  zero for its only contribution, so contraction cannot matter), but that has not
  been *measured* on the MHA matrix yet, and it must be;
* the key pass is parallelism-starved at realistic `kv_heads`, which is a design
  issue to settle before GQA becomes a default rather than an option.

---

## 12. Smallest next step

1. Promote the candidate to a production artifact (new `shaders/` entry plus a
   manifest record) and keep the current key pass alongside it until the MHA
   matrix has been re-measured against the promoted bytes. That re-measurement is
   the gate for the swap.
2. Then widen **one** condition: replace
   `push.num_heads != push.num_key_value_heads` in `two_pass_attention_rejection`
   with a divisibility-based GQA admission, leaving every other rejection
   unchanged, and add the 8q-2kv and 4q-2kv cases to the production parity test
   rather than to this experiment.
3. Only after that, extend the matrix: `causal=0`, masked rows, `kv_seq < q_seq`,
   other head dimensions and groupings — each with its own receipt. Do not widen
   on two groupings alone.

---

## Reproduction

```bash
cd hierarchos-vulkan

# candidate: both groupings, both modes
HIERARCHOS_ATTENTION_BACKWARD_GQA_EXPERIMENT=1 \
  cargo test --lib gqa_two_pass_experimental_matches_scalar_gradients -- --nocapture

# negative control: MHA-qualified pass 2 on GQA (expected to FAIL mode 0)
HIERARCHOS_ATTENTION_BACKWARD_GQA_EXPERIMENT=mha \
  cargo test --lib gqa_two_pass_experimental_matches_scalar_gradients -- --nocapture

# MHA regression
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 \
  cargo test --lib two_pass_attention_backward_matches_scalar_gradients -- --nocapture

# kernel timing (wall), then GPU timestamps
HIERARCHOS_ATTENTION_BACKWARD_GQA_BENCH=1 \
  HIERARCHOS_ATTENTION_BACKWARD_GQA_BENCH_ITERATIONS=40 \
  cargo test --lib gqa_two_pass_experimental_kernel_timing -- --nocapture
HIERARCHOS_ATTENTION_BACKWARD_GQA_BENCH=1 \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  cargo test --lib gqa_two_pass_experimental_kernel_timing -- --nocapture \
  > raw/kernel-timing-profiled.log 2>&1
python validation/gqa_attention_backward_experiment_20261124/analyze_timing.py \
  raw/kernel-timing-profiled.log
```

The whole sequence is `reproduce.sh` in this directory.

`raw/` holds the logs these tables were built from:
`parity-gqa-candidate.log`, `parity-gqa-mha-control.log`,
`parity-mha-regression.log`, `gate-and-comparator.log`, `kernel-timing-wall.log`,
`kernel-timing-profiled.log`, `kernel-timing-gpu.json`, `source-hashes.txt`.
`analyze_timing.py` refuses to invent a sample: a log without profiler output
yields no number.

## Limitations

* Two groupings, head dimension 128, `q_seq = kv_seq = 64`, `causal=1`,
  all-ones masks. No window, softcap, ALiBi, relative bias, dropout, sink,
  offset, sparse mask, compressed keys, paged KV, or `value_scale != 1`.
* The production gate deliberately still rejects GQA, and the candidate is
  test-build-only.
* Timing is one machine, one driver, one thermal state. The ratios are
  kernel-level at this geometry and say nothing about complete training steps.
* The Phase III diff was recorded without `--binary`, so it cannot be re-applied
  as a patch; the tree state is verified by status, hashes and behaviour instead.
