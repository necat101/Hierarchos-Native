# Phase III — MHA attention-backward closeout and default readiness

**Date:** 2026-11-24
**Base commit:** `560809eead186f832344c15d1a9b70795da3b75d`
**Verdict:** **GO for the scoped MHA default.**
**Not in scope / still NO-GO:** grouped-query attention, `kv_seq_len > head_dim`,
and every other geometry the eligibility gate rejects.

This directory is the committed-form receipt for the fresh Phase III reproduction.
`qualification.json` is generated from the raw files in `raw/` by `assemble.py`.
The producer refuses to emit a receipt if a raw artifact is missing or a check
fails; it never invents a measurement.

```
python validation/mha_attention_backward_closeout_20261124/assemble.py
```

Raw evidence, protocol and hashes are listed in `raw/` and bound by the
`source_sha256` / `dataset_sha256` / `executable_sha256` fields of
`qualification.json`.

---

## 1. Repository state and the "1 file changed" discrepancy

Measured at the start of this phase, before any edit:

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `560809eead186f832344c15d1a9b70795da3b75d` (expected) |
| `git status --short` | empty |
| `git diff` / `git diff --stat` | empty |
| `git stash list` | `stash@{0}: On main: pre-v0.2.6 untracked validation evidence (preserved, not deleted)` |
| `stash@{0}` content | 48 untracked files, 32 insertions; the tracked diff of the stash is empty |
| Reflog | last operations were `checkout`, `reset: moving to HEAD`, and the fast-forward `pull` that produced `560809e`; no post-baseline source commit |
| `git fsck --unreachable` | one unreachable tag, no unreachable commit |

**Resolution.** The Phase II report's final "1 file changed" is not reproducible
as a tracked source change. At handoff the tracked tree was byte-identical to
`560809e`, and every Phase II artifact found on disk (`bench.py`, the GQA and
`seq > head_dim` fixtures, fallback reports, the sequence-16 reports) lives under
the git-ignored `target/phase2-attention-backward/` directory, which
`git status --short` does not list. The line therefore described an ignored or
already-removed report artifact, not a changed source baseline. No reset,
rebase, discard or overwrite was performed; `stash@{0}` and all untracked
evidence were left untouched.

The working tree now intentionally contains the Phase III changes (the tracked
files listed in `qualification.json` plus this untracked receipt directory) and
**no commit**.

---

## 2. Mode-0 and mode-9 numerical parity

`transformer::tests::two_pass_attention_backward_matches_scalar_gradients` now
exercises **both admitted arithmetic modes over the same four geometries**:
fixture-causal, fixture-full, padded-causal and small-causal. Each mode compares
dQ, dK and dV against the scalar kernel through the production descriptors,
push constants and grid shapes, with outputs pre-filled with a sentinel so an
unwritten element cannot pass as zero.

| Metric | Result |
| --- | --- |
| Cases executed | 8 (4 geometries × 2 modes) |
| Tensor comparisons | 24 (8 cases × dQ/dK/dV) |
| Elements compared | 6,316,128 |
| Bitwise-equal elements | 6,316,128 (100%) |
| Maximum absolute difference | 0.0 |
| Maximum ULP difference | 0 |
| Unwritten output elements | 0 |
| Verdict | `PASS` |

Dispatch proof: the scalar arm recorded exactly 1 dispatch per case, the
two-pass arm exactly 2 (one per pass), asserted from
`ComputeBatch::dispatch_count()` — not self-reported. A one-ULP negative control
(`gradient_report_rejects_a_one_ulp_difference`) proves the comparator rejects a
single-bit change, so the PASS is not vacuous.

### The mode-0 divergence that was found and fixed

Mode 9 passed before this phase; mode 0 did not. The ordering difference was in
pass 1's `scratch_ds`:

* The scalar mode-9 adjoint caches a **`precise` per-key softmax dot** in
  `row_softmax_dot` and reduces dK/dV against it.
* The scalar mode-0 path keeps no cached statistics (`cache_stats` is false for
  mode 0) and reduces dK/dV against the **plain** `softmax_dot`.
* Pass 1 wrote `p * (dp - kv_softmax_dot)` for both modes, so mode 0's dK/dV used
  the wrong accumulator and could differ by a few ULP.

The fix is mode-aware, in `shaders/transformer_attention_backward_query_rdna3.comp`:
`kv_dot = (unfused_qk == 9) ? float(kv_softmax_dot) : softmax_dot`. The eligibility
gate was **not** narrowed. dQ always used the plain `softmax_dot` and was already
exact; dK/dV now select the same accumulator the owning scalar path uses.

Shader identities changed as a result:

| Artifact | Before | After |
| --- | --- | --- |
| query `.comp` | — | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` |
| query `.spv` | `615c27dc3787cb7f0bb53aec2da16d906e1405e37a27b6ef0d8780d57bc0b108` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |

The SPIR-V was rebuilt with the manifest's own flags
(`glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders`)
and validated with `spirv-val --target-env vulkan1.2`. Historical Phase XVI
receipts bind the old `615c27dc…` bytes and remain historical; they are not
revalidated against this tree.

---

## 3. Non-vacuous hardware qualification

The gradient test now distinguishes four outcomes explicitly:

| Status | Meaning | Behavior |
| --- | --- | --- |
| `PASS` | A device was found, both kernels loaded, both passes recorded, all 24 comparisons executed and bit-exact | test succeeds |
| `FAIL` | Executed and violated a requirement (missing kernel, non-exact gradient, unwritten element, wrong dispatch count) | test fails |
| `BLOCKED` | Required device unavailable while the run explicitly requested hardware qualification (`HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1`) | test panics with `BLOCKED`; cannot be counted as numerical qualification |
| `SKIP` | Intentionally not executed: no device and no explicit hardware request, or the path is explicitly disabled | visible `SKIP` line, no numerical claim |

Ordinary CPU-only CI may therefore run the test without a Vulkan device and see
a visible SKIP; it is not a PASS. A hardware qualification run that asks for the
path (`=1`) cannot silently succeed without a device or a shader. The test also
requires 8 executed cases and 24 comparisons before it can print `PASS`.

Executed in this phase: `PASS` (explicit `=1`, 8/8 cases, 24/24 comparisons),
`SKIP` (explicit `=0`), and `BLOCKED` (explicit `=1` with the Vulkan loader
pointed at missing ICDs: `status=BLOCKED reason=no_vulkan_device
detail=creating Vulkan instance: ERROR_INCOMPATIBLE_DRIVER`, followed by the
failing `BLOCKED` panic, exit 101). The missing-kernel `FAIL` branch was not
separately forced; the one-ULP comparator control covers the numerical-failure
path.

---

## 4. Scratch allocation

### Derived size

Both shaders index scratch as

```
slot = (batch * num_heads + head) * query_seq_len * key_value_seq_len
     + query * key_value_seq_len + key
```

so the highest slot is `batch * num_heads * query_seq_len * key_value_seq_len - 1`.
Each of the two buffers is exactly that many floats. The previous allocation
used `rows * query_hidden = (batch * seq) * (num_heads * head_dim)` per buffer —
an upper bound that over-allocated by `head_dim / key_value_seq_len` (2× for the
fixture, 4 MiB per layer instead of 2 MiB) and, more importantly, allocated it at
construction for every geometry, including ones the pair would reject.

### Allocation policy

Scratch is now allocated lazily by `select_two_pass_attention_backward` on the
first **eligible** dispatch, not at layer construction. The rejection test runs
before the allocator; every rejected geometry returns with no allocation. Each
layer owns its own `RefCell<Option<TwoPassAttentionScratch>>`, so layers retain
independent storage and the scalar path never pays for the fast path.

Measured evidence (`raw/scratch-*.log`):

| Geometry | Decision | Scratch allocations | Scratch bytes |
| --- | --- | --- | --- |
| Fixture MHA, batch 8, heads 8, seq 64, head_dim 128 | `route=two-pass reason=eligible` | 2 (one per layer) | 4,194,304 total (2,097,152/layer) |
| Same geometry, `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0` | `route=scalar reason=disabled_by_environment` | 0 | 0 |
| GQA 8 query / 2 KV heads, seq 8, head_dim 32 | `route=scalar reason=grouped_query_attention` | 0 | 0 |
| MHA 8/8, seq 64, head_dim 32 | `route=scalar reason=kv_sequence_above_head_dim` | 0 | 0 |

A pure unit test (`two_pass_scratch_sizing_and_eligibility_are_geometry_exact`)
locks the exact formula and the rejection order for GQA, long context and
unsupported arithmetic modes.

---

## 5. Per-geometry diagnostics

The old diagnostic was a single `AtomicBool` one-shot: it described the first
attention-backward geometry in the process and hid every later layer's decision.
It is replaced by a cache keyed on the full decision (route, eligibility, reason,
kernel availability, key-block count, batch, heads, KV heads, query/KV sequence,
head dimension, causal mode, `unfused_qk`, scratch elements). Each distinct
decision prints once; repeated training steps do not spam.

Example lines from `raw/scratch-eligible.log` and `raw/scratch-gqa.log`:

```
two-pass backward scratch: layer=1 batch=8 heads=8 kv_heads=8 q_seq=64 kv_seq=64 elements=262144 bytes=2097152
two-pass backward: route=two-pass eligible=true reason=eligible kernels=true scratch=true key_blocks=1 batch=8 heads=8 kv_heads=8 q_seq=64 kv_seq=64 head_dim=128 causal=1 unfused_qk=9 scratch_elements=262144 scratch_bytes=2097152
two-pass backward counters: effective=true scalar=0 query=4 key=4 scratch_allocations=2 scratch_bytes=4194304
```

```
two-pass backward: route=scalar eligible=false reason=grouped_query_attention kernels=true scratch=false key_blocks=1 batch=2 heads=8 kv_heads=2 q_seq=8 kv_seq=8 head_dim=32 causal=1 unfused_qk=9 scratch_elements=0 scratch_bytes=0
two-pass backward counters: effective=true scalar=2 query=0 key=0 scratch_allocations=0 scratch_bytes=0
```

The end-of-run counters come from atomics incremented only when a dispatch is
actually recorded. A later layer with a different geometry produces its own
line, so one layer's decision can no longer describe another's.

---

## 6. Fresh performance reproduction

Protocol is the Phase II script preserved at `raw/phase2-bench.py` and run
unchanged in substance: same fixture, same dimensions, same dataset, same
trainer configuration, 5 ABBA rounds, 26 steps per run, 3 warmup steps, 12
profiled kernel steps. ABBA order per round: scalar, two-pass, two-pass, scalar.
The per-round statistic is `median(scalar step_ms) / median(two-pass step_ms)`;
the interval is a Student-t interval over the five paired ratios (df = 4).

### Kernel GPU time (timestamps, 12 steps per arm)

| Kernel | ms/step |
| --- | --- |
| `transformer_attention_backward` (scalar) | 435.2787 |
| `transformer_attention_backward_query_rdna3` | 27.7139 |
| `transformer_attention_backward_key_rdna3` | 16.7182 |
| Two-pass total | 44.4321 |
| **Backward speedup** | **9.7965×** |

Phase II reported 431.93 / 44.59 ms (9.69×). The fresh measurement reproduces
the backward result essentially exactly.

### Complete optimizer step (paired ABBA)

| Metric | Scalar | Two-pass |
| --- | --- | --- |
| Mean step | 562.40 ms | 173.90 ms |
| Median step | 560.78 ms | 171.10 ms |
| Samples per arm | 230 | 230 |

Paired round ratios: `3.3296, 3.3009, 3.2732, 3.2830, 3.0591`.

* **Mean paired speedup: 3.2492×**
* **95% CI: [3.1146, 3.3838]** (df = 4, t = 2.7764)
* Ratio of means: 3.2340× (not used as the primary statistic)

The Phase II narrative reported 3.304× with CI [3.2530, 3.3549]. That estimate
lies inside the fresh interval; the fresh point estimate is 3.2492× and its lower
bound is 3.1146×. One round (3.0591×) carried visibly more variance than the
others, so the interval is wider than the earlier run, but the complete-step
improvement remains statistically supported and above 3×. The receipt records
only the freshly measured 3.2492×.

### Numerical parity of the measured arms

| Check | Result |
| --- | --- |
| Loss bits, both steps | `[1088659837, 1088672584]` in both arms |
| LoRA `adapter_model.safetensors` SHA-256 | `b3da10f2ca2b889e595f868131dfb99ddf5d1a5740d67cc489f7fedb9550a2d4` in both arms |
| LoRA `adapter_config.json` SHA-256 | `1fa6572209fb7a4c5fce8f4efcfaef8a90b11af831952a76dcb76b2a5da6d696` in both arms |
| Parameter-end digest | `daaa3852a49494874b6f6d9e4ffd6db50690cc9afed422a6b1c4d4f062aa5a4d` in both arms |
| Equivalence harness | `PASS: both passes dispatched and every adapter file is bit-identical`; 11/11 negative controls detected |

### Dispatch counters

| Arm (12 profiled steps, 2 layers) | scalar | query | key |
| --- | --- | --- | --- |
| Scalar | 24 | 0 | 0 |
| Two-pass | 0 | 24 | 24 |

Expected per shader is `steps × layers = 24`. The scalar arm never dispatched a
two-pass kernel; the two-pass arm never fell back to the scalar kernel.

### Sequence-16 cross-check

Three LoRA steps at batch 8, seq 16, the same fixture and seed:

| Check | Result |
| --- | --- |
| Loss bits | `[1088507454, 1088697126, 1089011736]` in both arms |
| Adapter SHA-256 | `d990fa646b7f463445e80ad05d5a0054a50346823581e2f62fb27842f11d35b4` in both arms |
| Parameter-end digest | `daaa3852a49494874b6f6d9e4ffd6db50690cc9afed422a6b1c4d4f062aa5a4d` in both arms |
| Mean step | 116.33 ms scalar, 71.53 ms two-pass (1.63×) |

The 16-token case is a correctness cross-check, not the performance claim.

### Historical 2.88× narrative

No committed benchmark receipt for the 2.88× number was found. The Phase XVI
receipts in `validation/release_parity/` document the opt-in, mode-9-only
qualification of the old shader and are not a benchmark receipt. The 2.88×
figure is therefore **not preserved as a verified historical measurement**; this
receipt makes no claim about it.

---

## 7. Scoped default selection

Selection semantics, implemented in `two_pass_attention_enabled` /
`two_pass_attention_setting`:

| `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS` | Behavior |
| --- | --- |
| unset | two-pass for eligible geometry (new default) |
| `0` / `false` / `off` | scalar fallback everywhere; pair kernels are not even loaded; no scratch allocated |
| `1` / `true` / `on` | two-pass for eligible geometry, and hardware qualification is mandatory |
| anything else | warning; eligible default |

The eligibility gate is unchanged in substance (`num_heads == num_key_value_heads`,
`unfused_qk` 0/9, `head_dim <= 128`, `kv_seq_len <= head_dim`, no window,
softcap, ALiBi, relative bias, dropout, sink, sparse mask, compressed keys,
paged KV or non-unit value scale). The only gate-related change is that the old
`elements <= capacity` width check became an exact allocation, so a geometry is
rejected before allocation rather than after.

**GO** because all of the following hold:

* 24/24 bit-exact gradient comparisons across both admitted modes and all four
  geometries, with zero ULP and zero unwritten elements.
* A hardware qualification cannot pass without a device, both kernels, both
  dispatches and executed comparisons.
* Ineligible geometries allocate no scratch; eligible MHA allocates exactly the
  shader-derived size per layer.
* The complete-step improvement is statistically supported: CI lower bound
  3.1146× over five paired rounds.

**Remaining NO-GO:** GQA and long-context geometries. Do not relax
`num_heads == num_key_value_heads` on the strength of this result.

---

## 8. Receipt contents

```
validation/mha_attention_backward_closeout_20261124/
  README.md                 this document
  qualification.json        machine-readable receipt
  assemble.py              derives qualification.json from raw/, fails closed
  reproduce.sh             reruns the protocol into a separate output directory
  raw/
    parity-mode0-mode9.log  24 comparisons and the PASS summary
    parity-skip.log         visible SKIP for the explicit opt-out
    blocked.log             visible BLOCKED and failing panic with no Vulkan ICD
    comparator-negative-control.log
    equivalence.log         dispatch proof, negative controls, adapter digests
    kernel-*.err.txt        profiled 12-step arms
    numeric-*.err.txt       two-step numerical arms
    bench-summary.json      full Phase II-protocol summary (samples, rounds, CI)
    scratch-eligible.log
    scratch-optout.log
    scratch-gqa.log
    scratch-seqgt.log
    seq16-tp*.json          sequence-16 reports
    seq16-tp*-adapter*.safetensors / -config.json
    git-head-stash.txt, git-status.txt, git-diff.patch
    environment.txt         device, driver, rustc/cargo, glslc, spirv-val
    source-hashes.txt       source, shader, fixture and executable SHA-256
    phase2-bench.py         preserved copy of the Phase II protocol script (run
                            target/phase2-attention-backward/bench.py; the copy
                            lives too deep for the script's own root detection)
```

---

## 9. Next isolated GQA experiment

**Production gate stays closed.** The experiment runs outside the production
eligibility gate; until it passes for both admitted arithmetic modes, GQA keeps
the scalar kernel.

Fixture: 8 query heads, 2 KV heads, head dimension 128, sequence length 64,
batch 2, causal. Use both `unfused_qk = 0` and `unfused_qk = 9`.

1. **Bypass-only dispatch.** Add a `#[cfg(test)]` (or experiment-binary) path
   that constructs the `AttentionPush` with `num_heads = 8`,
   `num_key_value_heads = 2` and dispatches the existing pair directly, with
   scratch sized by the same `batch * num_heads * query_seq * kv_seq` formula.
   Do not change `two_pass_attention_rejection`.
2. **Direct dQ/dK/dV parity.** Compare the pair against the scalar kernel for
   both modes, reporting elements, bitwise-equality count, max abs, max ULP,
   unwritten elements and first divergence, as in the MHA test.
3. **Head grouping and accumulation order.** Check explicitly that pass 2 maps
   `kv_head` to query heads `kv_head*group .. kv_head*group + group` in ascending
   order, and that the per-key dK/dV accumulation matches the scalar order.
4. **Scratch indexing.** Verify every slot written by pass 1 is read exactly
   once by pass 2, with no cross-head aliasing: the slot base must use the query
   head, while the K/V base uses the KV head.
5. **Expected failure mode to test first.** Mode 0's scalar path accumulates dK
   and dV directly into the global gradient buffer inside the query-head loop
   (`grad_k[...] += …` with one continuous accumulator across the grouped
   heads). Pass 2 resets `head_k`/`head_v` per head and then adds the completed
   heads. For MHA those trees coincide; for GQA they differ at every head
   boundary, so mode 0 GQA is expected to show small ULP differences even when
   mode 9 matches. The experiment must decide between reproducing mode 0's
   continuous cross-head accumulation in pass 2 or keeping mode-0 GQA excluded;
   it must not treat a mode-9-only pass as GQA readiness.
6. **Promotion criterion.** Only if both modes are bit-exact across at least the
   8q/2kv fixture and a second grouping (for example 8q/4kv) may the production
   gate be widened in a separate change with its own receipt.

---

## 10. Limitations and unsupported geometries

* MHA only: `num_heads == num_key_value_heads`.
* `unfused_qk` 0 and 9 only.
* `head_dim <= 128` and `kv_seq_len <= head_dim`.
* Sliding window, softcap, ALiBi, relative bias, dropout, attention sink,
  query-position offset, sparse masks, compressed keys, paged KV and
  `value_scale != 1` keep the scalar kernel.
* `BLOCKED` was executed by pointing the Vulkan loader at missing ICDs; the
  missing-kernel `FAIL` path was not separately forced. The one-ULP comparator
  control covers the numerical-failure path.
* Complete-step timings are specific to this machine, driver and thermal state;
  the interval covers run-to-run variation on this machine, not other hardware.
* The query shader bytes changed; historical receipts that bind
  `615c27dc…` describe the old artifact and are not current-source evidence for
  the new one.
