# Phase X — production promotion of the RDNA3 row-packed attention forward

**Date:** 2026-11-25
**Base commit:** `560809eead186f832344c15d1a9b70795da3b75d` (matches the expected base)
**Verdict: GO.** The exact Phase IX cached candidate is installed in the
production RDNA3 Transformer attention-forward path behind a narrow eligibility
predicate, with the generic kernel and the shipped short-sequence module intact.
The promoted bytes are **identical** to the Phase IX artifact, so the
`1.2236×` complete-step result is attributable to them without a new benchmark
campaign — and a release-mode ABBA re-run of the shipped selector reproduces
`1.2239×` `[1.2226, 1.2252]` independently.

Nothing was committed, pushed, merged or released. No prior receipt was
rewritten, no stash was touched and the index stayed empty.

---

## 1. Exact starting repository state

Captured before the first Phase X write (`raw/00_preservation.txt`).

| Fact | Value |
| --- | --- |
| branch | `main` |
| HEAD | `560809eead186f832344c15d1a9b70795da3b75d` |
| tracked modified | 13 files (+ 2 untracked test modules, 6 untracked receipt dirs) |
| `git diff \| sha256sum` | `2aa3f3c01effff09e3eb8dd73e5688b39e2f3faa8168bad7964e4c31e2d479c1` |
| `git diff \| wc -c` | 148538 |
| `git diff --binary \| sha256sum` | `c330d0f7a899c2063e5b5de67712f2f2f36a6ade1040c854f8fed7651c57f7b1` |
| `git diff --cached \| sha256sum` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) |
| `git diff --cached \| wc -c` | 0 |
| `git ls-files -s \| sha256sum` | `9220461f849b4e4b066f45127d7bf34df182ab4acebac7a1708376c9f3e445aa` |
| `git stash list` | `stash@{0}: On main: pre-v0.2.6 untracked validation evidence (preserved, not deleted)` |
| `git rev-parse stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` |

**The Phase IX claim holds in the tree.** `git diff --name-only` was the same 13
files as the Phase IX receipt recorded, and the only tracked source Phase IX
added was the four-line `#[cfg(test)]` module declaration at the end of
`src/transformer.rs`. No production dispatch, selector or shader had been
changed by Phases III–IX.

### The six production attention shader identities (before)

`raw/01_production-shader-hashes.txt`. These are byte-identical to the table in
the Phase IX receipt.

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

All ten are **still** these values at the end of the phase
(`raw/05_identity-end-of-phase.txt`, compared with `diff`). The generic forward,
the short specialization and the scalar backward were never opened for writing.

---

## 2. Preserved Phase III–IX evidence

`raw/02_prior-receipt-hashes.txt`, re-derived at the end of the phase and
compared line for line: **identical**. Each row is
`sha256(sorted per-file sha256 list)  file count  directory`.

| Receipt | Ledger sha256 | Files |
| --- | --- | ---: |
| `mha_attention_backward_closeout_20261124/` | `3e0dfb292fb741c28bacda508d93811da243334c5e810ee8d45a35f3b2864367` | 36 |
| `gqa_attention_backward_experiment_20261124/` | `b7755f5f1d22f4f6a2caf8897d8c37c0fc019004b7e2d8378bc277f43ceb6eaa` | 13 |
| `gqa_production_promotion_20261124/` | `0acb54d538ec22b3f9f99a5967b18df3287eb39ef8456779dd1c7c16cafb2387` | 14 |
| `gqa_end_to_end_training_20261124/` | `1fb0efe83c50180b3714e133cb882324f22f80e514dd385a5fad5859f0ddef95` | 178 |
| `phase_viii_key_blocks/` | `9293ee596cdec4f60127eecb225d1ccddd562f00b1f012848291dedb7f69b4b5` | 26 |
| `phase_ix_row_packed_forward_20261124/` | `542af36848d3b83677880f798422d026d9909796d156e32b9f6c9d021af9aef6` | 61 |

The Phase VIII key-block result stays a NO-GO and its candidate was not
reopened. Nothing was reset, cleaned, rebased, checked out, discarded or
committed.

---

## 3. The promoted artifact

`raw/03_promoted-artifact-hashes.txt`, `raw/21_promoted-artifact-rebuild.json`.

| Artifact | sha256 |
| --- | --- |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp` | `8f7bf06f65291793b12c81a9b0bd733a5ce656ffbd4a3689fea5fabf3439e6ed` |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv` | `c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd` |

Both are **byte-identical** to the Phase IX receipt's candidate
(`cmp` reports no difference for either file), and the SPIR-V was regenerated
from the promoted source to prove the identity is a compiler output rather than
a copied blob:

```
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O \
    -Ishaders shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp \
    -o shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv
spirv-val --target-env vulkan1.2 <artifact>        → exit 0
```

Compiler: `C:\VulkanSDK\1.4.321.1\Bin\glslc.EXE`, banner `shaderc v2023.8
v2025.3` — the release the repository's manifest already pins for the shipped
RDNA3 query/key passes, and the recipe Phase IX recorded.

**The negative control was not promoted.** The streaming candidate
(`03cc3c0e…` GLSL, `24bc8a6f…` SPIR-V) exists only under
`validation/phase_ix_row_packed_forward_20261124/shaders/` and no file in the
crate references it. It is 7.0× slower than the generic kernel; keeping it out
of `shaders/` is the point of recording it.

### Provenance

`validation/training_shader_manifest.json` gains exactly one entry
(`git diff --numstat` for that file is `14 0`):

```json
{
  "artifact": "transformer_attention_forward_rowpacked64_cached_rdna3.spv",
  "source": "transformer_attention_forward_rowpacked64_cached_rdna3.comp",
  "compiler": "glslc",
  "compiler_version": "2023.8",
  "flags": ["-fshader-stage=compute", "--target-env=vulkan1.2",
            "--target-spv=spv1.4", "-O", "-Ishaders"],
  "target_env": "vulkan1.2"
}
```

The manifest is inserted in its own alphabetical position, uses the file's
existing LF endings and 2-space formatting, and `read_manifest`'s structural
check passes.

**Repository-wide provenance gate — limitation, not a pass.** Running
`python validation/check_shader_provenance.py` reports
`{"identical": 53, "wrong_version": 676}` and fails. That is a pre-existing
condition of this machine, not a drift introduced here: the check refuses to
compare bytes when the installed compiler does not report the release the entry
pins, and on this host `PATH`'s `glslc` prints `shaderc v2023.8 v2025.3`, from
which the checker takes the *last* release (`2025.3`), while the only installed
`glslangValidator` is 15.3.0 against pins of 16.5.0/16.6.0. Untouched,
pre-existing artifacts fail the same way — for example
`transformer_attention_forward.spv` (`expected 16.5.0, observed 15.3.0`) and
`transformer_attention_backward_query_rdna3.spv` (`expected 2023.8, observed
2025.3`). `raw/provenance.json` records all 729 rows.

Because that gate cannot speak to this phase's artifact, `check_promoted_artifact.py`
runs the same rebuild for the one promoted entry and reports byte-equality on its
own, without the version gate:

```json
{
  "artifact": "transformer_attention_forward_rowpacked64_cached_rdna3.spv",
  "committed_sha256": "c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd",
  "rebuilt_sha256":   "c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd",
  "phase_ix_sha256":  "c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd",
  "source_sha256":    "8f7bf06f65291793b12c81a9b0bd733a5ce656ffbd4a3689fea5fabf3439e6ed",
  "compiler": "glslc", "recorded_version": "2023.8",
  "compiler_banner": "shaderc v2023.8 v2025.3\nspirv-tools v2025.3 v2022.4-833-g33e02568\nglslang 11.1.0-1253-gefd24d75\n\nTarget: SPIR-V 1.0",
  "reproduces_byte_for_byte": true, "identical_to_phase_ix": true,
  "spirv_val_exit": 0, "status": "PASS"
}
```

The artifact reproduces byte-for-byte. In the repository-wide run the same
artifact is still counted as `wrong_version`
(`expected 2023.8, observed 2025.3`) — because the checker takes the last
release token of `shaderc v2023.8 v2025.3` as the observed version, exactly as it
does for every other `glslc` entry in the manifest. The promoted entry therefore
adds one row to the same pre-existing 676.

---

## 4. The production eligibility predicate

`rdna3_row_packed_attention_rejection` in `src/transformer.rs`. It returns the
first field that disqualifies a push, so a diagnostic names the reason rather
than only reporting a fallback.

**Device (checked at install, not per dispatch).** `vendor::rdna3_module_enabled`
— the repository's existing RDNA3 PCI table, the vendor A/B switch and
`HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS` — plus
`supports_compute_work_group_size([64, 1, 1])`. If any of that fails, or the
driver refuses the pipeline, the kernel field is `None`.

**Geometry**

| Condition | Why |
| --- | --- |
| `batch_size > 0` | an empty dispatch has no rows to own |
| `0 < query_seq_len <= 64` | the per-lane `float scores[64]` / `bool visible[64]`, indexed by absolute key |
| `0 < key_value_seq_len <= 64` | same array; and at `kv <= 64` the generic kernel's `use_cache` test is unconditionally true, which is the only path the module implements |
| `query_seq_len == key_value_seq_len` | the square layout every Phase IX geometry uses; non-square cross-attention is untested and keeps its route |
| `num_heads > 0`, `num_key_value_heads > 0`, `num_heads % num_key_value_heads == 0` | the GQA quotient `kv_head = head / queries_per_kv` |
| `query_hidden_size % num_heads == 0`, `value_hidden_size % num_key_value_heads == 0` | integer head dimensions |
| `qk_head_dim == value_head_dim` | the equal-width layout of the qualified matrix and of the shipped short module |
| `rows <= u32::MAX` | the flattened row index is a `uint` |

**Plain path** — `window_size == 0` (keeps `first_key` at 0), `softcap == 0`,
`alibi_mode == 0`, `relative_bias_mode == 0`, `dropout_p == 0`,
`has_sink == 0`, `causal <= 1`, `query_position_offset == 0`,
`has_sparse_mask == 0`, `compressed_key_count == 0`, `paged_kv == 0`,
`value_scale == 1`.

**Arithmetic** — `unfused_qk` in `{0, 1, 9}`. The module reproduces all four
switches through the same `fp32_sleef_exp.glsl`, `fp32_sqrt_recip.glsl` and
`fp32_fma_exact.glsl` headers, so it is arithmetically a drop-in for the whole
`unfused_qk` space; only these three modes carry Phase IX bitwise evidence, so
the diagnostic modes keep the generic kernel.

`raw/10_selector-matrix.log` is a host-only test: 24 cases, one per clause, each
asserting the exact reason string the predicate returns (two clauses are checked
by two cases, such as `65` and `0` both hitting `query_sequence_above_64`), plus
the accepted 64/64 training geometry and the inclusive bound asserted
*eligible*. No GPU is involved, so a predicate regression is caught without a
device.

```
[phase-x-eligibility] expected=non_square_sequence observed=Some("non_square_sequence") verdict=PASS
[phase-x-eligibility] expected=empty_batch observed=Some("empty_batch") verdict=PASS
[phase-x-eligibility] expected=arithmetic_mode_unqualified observed=Some("arithmetic_mode_unqualified") verdict=PASS
[phase-x-eligibility-summary] status=PASS cases=24 failures=0
```

---

## 5. Selector order and fallback

`record_text_attention_dispatch` is the single function the real training step
calls (one call site, in the layer forward). Its order, first match wins:

1. the shared image-text AVX512 GPT-2 family;
2. the shared image-text AVX512 eager family;
3. the shared image-text AVX512 SDPA module;
4. **the RDNA3 wave64 short module** (`query_seq_len <= 16`, mode 0);
5. **the RDNA3 wave64 row-packed module** (this phase);
6. **the committed generic kernel**, with its original `[query, head, batch]`
   geometry.

**The short module keeps precedence, deliberately.** It was qualified against
its own narrower gate and has never been compared with the 64-key module, so
promoting the wider route must not displace it. That is measured, not argued
(`raw/13_short-precedence.log`):

```
[phase-x-precedence] label=short-eligible-mode0 expected=short-wave64 observed=short-wave64 mode=0 elements=1024 bitwise=1024 short_delta=1 row_packed_delta=0 generic_delta=0 verdict=PASS
[phase-x-precedence] label=short-refused-mode9 expected=row-packed-64-cached observed=row-packed-64-cached mode=9 elements=1024 bitwise=1024 short_delta=0 row_packed_delta=1 generic_delta=0 verdict=PASS
[phase-x-precedence-summary] status=PASS failures=0
```

Both cases are the same shape (`b2 s8 h4 kv2 d16`) and differ only in arithmetic
mode: mode 0 satisfies both gates and the short module takes it; mode 9 falls
outside the short gate (`unfused_qk == 0`) and the promoted route takes it. The
short module is installed for the test because the shared image-text composition
is what installs it in production; the test reproduces that selection state
directly rather than building a package.

The generic route stays reachable for every unsupported case, and the fallback
is measured through the selector rather than inferred from the predicate
(`raw/12_fallback.log`):

```
[phase-x-fallback] label=seq-128-above-64 reason=query_sequence_above_64 implementation=generic grid=[128, 2, 1] generic_delta=1 row_packed_delta=0 elements=4096 bitwise=4096 verdict=PASS
[phase-x-fallback] label=dropout reason=dropout implementation=generic grid=[16, 2, 1] generic_delta=1 row_packed_delta=0 elements=512 bitwise=512 verdict=PASS
[phase-x-fallback] label=sliding-window reason=sliding_window implementation=generic grid=[16, 2, 1] generic_delta=1 row_packed_delta=0 elements=512 bitwise=512 verdict=PASS
[phase-x-fallback] label=mode-4 reason=arithmetic_mode_unqualified implementation=generic grid=[8, 2, 1] generic_delta=1 row_packed_delta=0 elements=256 bitwise=256 verdict=PASS
[phase-x-fallback] label=causal-2 reason=causal_mode implementation=generic grid=[8, 2, 1] generic_delta=1 row_packed_delta=0 elements=256 bitwise=256 verdict=PASS
[phase-x-fallback-summary] status=PASS failures=0
```

Each fallback case is also compared bitwise against the generic kernel fed the
*same* push, so "it fell back" and "the fallback is correct" are separate,
measured facts.

**The optimized module is not a prerequisite.** It is built with
`vulkan::ComputeKernel::new(...).ok()` into an `Option`; a device that does not
qualify, a vendor family that is disabled, or a pipeline the driver refuses all
leave the field `None` and the generic kernel runs. Nothing else in the graph
depends on it.

### Diagnostics

`raw/15_diagnostics.log`, captured with
`HIERARCHOS_ATTENTION_FORWARD_DIAGNOSTICS=1`. The trace is deduplicated on
`arm|reason|batch x queries x keys x heads x kv_heads x head_dim x causal x mode`,
so a training step that dispatches one geometry twice per layer prints one line:

```
[attention-forward] implementation=row-packed-64-cached device=AMD Radeon Graphics vendor_id=0x1002 device_id=0x15bf batch=2 query_seq=64 kv_seq=64 heads=8 kv_heads=2 head_dim=128 causal=1 mode=9 grid=[16, 1, 1] workgroups=16 lanes_per_workgroup=64 useful_rows=1024 eligibility=eligible fallback_reason=row_packed_eligible dispatches=1
[attention-forward] implementation=row-packed-64-cached device=AMD Radeon Graphics vendor_id=0x1002 device_id=0x15bf batch=2 query_seq=7 kv_seq=7 heads=3 kv_heads=3 head_dim=16 causal=1 mode=9 grid=[1, 1, 1] workgroups=1 lanes_per_workgroup=64 useful_rows=42 eligibility=eligible fallback_reason=row_packed_eligible dispatches=8
[attention-forward] implementation=generic device=AMD Radeon Graphics vendor_id=0x1002 device_id=0x15bf batch=1 query_seq=128 kv_seq=128 heads=2 kv_heads=2 head_dim=16 causal=1 mode=9 grid=[128, 2, 1] workgroups=256 lanes_per_workgroup=1 useful_rows=256 eligibility=fallback fallback_reason=query_sequence_above_64 dispatches=1
[attention-forward] implementation=generic device=AMD Radeon Graphics vendor_id=0x1002 device_id=0x15bf batch=1 query_seq=8 kv_seq=8 heads=2 kv_heads=2 head_dim=16 causal=2 mode=9 grid=[8, 2, 1] workgroups=16 lanes_per_workgroup=1 useful_rows=16 eligibility=fallback fallback_reason=causal_mode dispatches=5
```

Each line carries all five required facts: the selected implementation, the
device and geometry, the eligibility verdict, the fallback reason (a concrete
reason string even on the eligible path, so the field is never empty) and the
actual dispatch count for that process. The counters are always recorded and are
published as
`attention_forward_dispatch_counts() -> AttentionForwardDispatchCounts { generic,
row_packed, short }` (exported from the crate root, alongside the existing
backward counters).

---

## 6. Forward numerical matrix on the production artifact

`raw/11_parity-forward.log`. Every comparison is the raw FP32 bit pattern
against the committed generic kernel; `max_abs = 0.0`, `max_ulp = 0`,
`first_divergence = none`, `unwritten = 0` on every line.

Ten geometries, unchanged from Phase IX, each run twice — once through the
selector (which chooses the packed grid) and once with the promoted kernel
launched under the production `[query, head, batch]` grid, preserving Phase IX's
second dispatch grid as a production claim:

| Geometry | rows | mode | selector grid | production-grid arm | output |
| --- | ---: | ---: | --- | --- | --- |
| `gqa-8q-2kv-primary-mode9` (Phase VII) | 1024 | 9 | `[16,1,1]` | `[64,8,2]` | exact |
| `gqa-8q-2kv-primary-mode0` | 1024 | 0 | `[16,1,1]` | `[64,8,2]` | exact |
| `gqa-8q-2kv-primary-mode1` | 1024 | 1 | `[16,1,1]` | `[64,8,2]` | exact |
| `mha-8q-8kv-control` | 1024 | 9 | `[16,1,1]` | `[64,8,2]` | exact |
| `gqa-8q-2kv-noncausal` | 1024 | 9 | `[16,1,1]` | `[64,8,2]` | exact |
| `gqa-8q-2kv-masked` | 1024 | 9 | `[16,1,1]` | `[64,8,2]` | exact |
| `mha-8q-8kv-masked` | 1024 | 9 | `[16,1,1]` | `[64,8,2]` | exact |
| `boundary-42rows` (b2 s7 h3 d16) | 42 | 9 | `[1,1,1]`, 22 lanes idle | `[7,3,2]` | exact |
| `boundary-130rows` (b2 s13 h5 d16) | 130 | 9 | `[3,1,1]`, 62 idle | `[13,5,2]` | exact |
| `boundary-162rows-gqa` (b3 s9 h6 kv2, masked query) | 162 | 9 | `[3,1,1]`, 30 idle | `[9,6,3]` | exact |

```
[phase-x-parity-summary] status=PASS cases=20 elements=1845696 bitwise_equal=1845696 failures=0
```

The element count reconciles exactly with Phase IX: `1845696 + 131072 = 1976768`,
and 131072 is the single primary-geometry comparison Phase IX added for the
**streaming** candidate, which this phase deliberately does not promote. Ten
geometries, three arithmetic modes and both grids are all preserved.

**Dispatch proof, per geometry.** Every eligible case asserts that the selector
incremented the row-packed counter by exactly one and moved neither the generic
nor the short counter, and prints the grid it chose:

```
[phase-x-dispatch] label=gqa-8q-2kv-primary-mode9 implementation=row-packed-64-cached grid=[16, 1, 1] rows=1024 workgroups=16 selected=1 generic_delta=0 short_delta=0 verdict=PASS
[phase-x-dispatch] label=boundary-42rows implementation=row-packed-64-cached grid=[1, 1, 1] rows=42 workgroups=1 selected=1 generic_delta=0 short_delta=0 verdict=PASS
```

Partial groups are measured, not argued: 42 rows leaves 22 lanes idle in the
only workgroup, 130 leaves 62 in the third, 162 leaves 30 — and the sentinel
count is 0 in every output, so no row is missed and none is written twice.

---

## 7. Backward and intermediate-state parity

`raw/14_parity-gradients.log`. The production two-pass GQA backward (query pass
then key pass, unchanged artifacts) is run from each forward result and compared
with the same backward run from the generic forward. Four geometries: the
primary mode 9, mode 0, the MHA control and the 162-row boundary.

```
[phase-x-gradient-summary] status=PASS cases=4 elements=2373444 bitwise_equal=2373444 failures=0
```

* `dq` 131072/131072, `dk`/`dv` 32768/32768 (1024/1024 at MHA) — assignments,
  `unwritten = 0`.
* pass-1 `probability` and `dS` scratch 65536/65536 and 1458/1458, with the
  **same** structurally unwritten set in both arms (32256 elements of the causal
  upper triangle at the primary geometry, 648 at the 162-row boundary) — a
  stronger requirement than comparing only the written subset.
* The production scalar backward is the independent oracle for the same tensors
  and matches the two-pass pair, so the PASS is against a qualified baseline.

No forward intermediate exists to compare: the forward kernel has no
intermediate outputs and the two-pass backward recomputes probabilities and dS
from Q/K/masks inside its own query pass. That claim is measured here rather
than asserted — the two-pass pair is fed identical bytes in both arms and the
digests agree.

---

## 8. Real production dispatch proof and release-mode training equivalence

`raw/step-production.log`, `raw/step-generic.log`, `raw/23_step-digest-release.json`.
Fixture: `target/gqa-e2e/fixture-8q2kv` (32 sequences, 2 layers, hidden 1024,
batch 2, seq 64, 8q/2kv, head_dim 128, LoRA r8 α16, AdamW lr 1e-2 β(0.9,0.999)
ε 1e-8, seed 7), release harness, six complete steps per arm.

Both arms are the **same binary**. The production arm installs nothing — it is
the shipped selector. The generic arm is
`HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1`, which the vendor family already
gates on, so it runs this build's own committed generic kernel. The harness
refuses to run if the arm's route is not the one it names.

| | production | generic |
| --- | --- | --- |
| `row_packed_installed` | `true` | `false` |
| forward dispatches (6 steps × 2 layers) | generic 0, **row-packed 12**, short 0 | generic **12**, row-packed 0, short 0 |
| backward dispatches | scalar 0, **query 12, key 12** | scalar 0, **query 12, key 12** |

```
status=PASS  steps_compared=6
loss_bits_mismatches=0  gradient_digest_mismatches=0  parameter_digest_mismatches=0
gradient_elements_per_step=32768  gradient_bits_compared=6291456
parameter_end_digest (both arms) = d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655
checkpoint_files_compared=3  checkpoint_file_mismatches=[]
routes_distinct=true
```

Loss bit 0 is `40e44309` on both arms, the Phase IX value. The default selector
chooses the row-packed forward **and** the production two-pass GQA backward
(`attention_backward_two_pass_enabled() == true`, `query`/`key` counters 12/12,
`scalar` 0).

The adapter and optimizer state are compared as bytes, not argued from equal
gradients (`save_lora_training_checkpoint` at the end of the run):

| File | Bytes | sha256 (both arms) |
| --- | ---: | --- |
| `adapter_config.json` | 531 | `1fa6572209fb7a4c5fce8f4efcfaef8a90b11af831952a76dcb76b2a5da6d696` |
| `adapter_model.safetensors` | 131688 | `90d9899ec7b3c04a781741d7d7620b89351ca29937ba04c6b9170f64ef67bfb0` |
| `hierarchos_adapter_optimizer.safetensors` | 264096 | `90a9c85c2cefdb3c8515104d5db04861df40a45bdbb07d83d3229073af7a8391` |

The base parameter map is unchanged by design (`parameter_values()` is the frozen
HF tensors, and LoRA training only moves the adapter), which is why its digest is
constant across steps in both Phase IX and this phase; the adapter and the AdamW
moments are the tensors that actually evolve, and they are bitwise identical.

Backward shaders, optimizer operations, precision policy and training
configuration were not touched.

---

## 9. Performance confirmation

`raw/abba-summary.json`, `raw/timing-gpu.json`, `raw/attribution.json`.

### Kernel, GPU timestamps

`raw/timing-*.log` → `raw/timing-gpu.json`. Two processes, opposite arm order,
25 single-dispatch submissions per arm after 3 warmups, profiler brackets
(`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`, `…SAMPLE_STRIDE=1`). Device: `AMD Radeon
Graphics` (integrated RDNA3), vendor `0x1002`, device `0x15bf`. The promoted
module is now labelled by the profiler
(`transformer_attention_forward_rowpacked64_cached_rdna3`).

| geometry | arm | measured first | measured second | median GPU ms |
| --- | --- | ---: | ---: | ---: |
| primary mode 9 | generic | — | 6.6872 / 7.1058 | 6.6872–7.1058 |
| primary mode 9 | row-packed | 3.4291 | **1.8612** | 1.8612–3.4291 |
| primary mode 0 | generic | 6.2637 | 6.2865 | 6.2637–6.2865 |
| primary mode 0 | row-packed | 2.7247→host | **1.6350** | 1.6350–1.6491 |

Steady-state (arm measured second) mode 9: 7.1058 / 1.8612 = **3.82×**; mode 0:
6.2865 / 1.6350 = **3.85×**. The first arm in a process carries a startup penalty
of ≈1.5 ms for the promoted module, exactly the order effect Phase IX recorded;
the worst paired observation is therefore ~1.95×, still a large win.

These are a confirmation, not a new headline. Phase IX's authority numbers are
`3.98×` median with CI `[2.32, 4.51]` over five processes, and they apply here
because the artifact bytes are identical.

### Complete training step, release harness

`raw/abba-r*-*.log` → `raw/abba-summary.json`. Four rounds of `A B B A`, one arm
per process, 29 steps per process (3 warmup + 26 measured).

| round | A generic median (ms) | B production median (ms) | A / B | forward dispatches per process (A / B) |
| ---: | ---: | ---: | ---: | --- |
| 0 | 53.1242 | 43.3854 | 1.224472 | 58 generic / 58 row-packed |
| 1 | 53.1384 | 43.4511 | 1.222947 | 58 generic / 58 row-packed |
| 2 | 53.2153 | 43.4528 | 1.224669 | 58 generic / 58 row-packed |
| 3 | 53.1730 | 43.4566 | 1.223589 | 58 generic / 58 row-packed |

* mean paired ratio **1.223919**, 95% CI **[1.222646, 1.225193]** (df 3, t 3.182)
* ratio of pooled medians over 208 samples per arm: **1.22395** (A 53.1742, B 43.4448)
* every round lands between 1.2229 and 1.2247; each process records 58 forward
  dispatches for 29 steps × 2 layers, all of them on its own arm's route
  (`forward_generic=58, forward_row_packed=0` against `forward_generic=0,
  forward_row_packed=58`), so the arms differ only in which artifact one dispatch
  resolves to

Phase IX reported **1.2236** `[1.2222, 1.2251]`. This phase's independent
reproduction is **1.2239** `[1.2226, 1.2252]` — the same effect, on the shipped
selector.

**Attribution scope.** Phase IX's ABBA swapped the kernel into
`graph.kernels.attention_forward`, so it dispatched under the production
`[query, head, batch]` grid. The promoted route dispatches the packed
`[16, 1, 1]` grid. Phase IX measured those two grids at 1.736 ms and 1.735 ms —
0.06% apart — and the shader bytes, arithmetic and every surrounding kernel are
unchanged, so the `1.2236×` result is attributed to this candidate and this
execution path. The independent re-run above removes the need to rely on that
argument alone.

---

## 10. Updated complete-step attention attribution

`raw/step-profile-*.log` → `raw/attribution.json`. Nine profiled steps per arm,
release harness; profiler brackets perturb wall time, so this table is for
attribution only.

| component | generic (ms/step) | production (ms/step) | Δ |
| --- | ---: | ---: | ---: |
| `transformer_attention_forward` | **16.4167** | — | — |
| `transformer_attention_forward_rowpacked64_cached_rdna3` | — | **4.3784** | −12.038 |
| `transformer_attention_backward_key_rdna3` | 16.1666 | 16.9257 | +0.759 |
| `transformer_attention_backward_query_rdna3` | 7.7089 | 8.1424 | +0.434 |
| `linear_forward` (+`_lane4`) | 7.8912 | 8.0343 | +0.143 |
| `linear_input_grad` (+group4) | 6.5479 | 6.9139 | +0.366 |
| `transformer_rms_norm_forward` | 1.8036 | 1.8449 | +0.041 |
| `transformer_rms_norm_input_grad` | 1.5363 | 1.5850 | +0.049 |
| `falcon_h1_cross_entropy_lanes16` | 1.2627 | 1.4226 | +0.160 |
| **total GPU per step** | **59.9725** | **49.7852** | **−10.187** |

The attention-forward dispatch drops **16.42 → 4.38 ms/step**, 3.75× in-model
with two dispatches per step. The +1.9 ms of drift on untouched components is
process-to-process: those kernels are the same artifacts on the same inputs, and
the ABBA's alternating order shows no such bias.

With the promoted forward, attention is:

| operation | ms/step | share of attention | share of step GPU |
| --- | ---: | ---: | ---: |
| attention backward — key pass | **16.9257** | **57.5%** | **34.0%** |
| attention backward — query pass | 8.1424 | 27.7% | 16.4% |
| attention forward | 4.3784 | 14.9% | 8.8% |
| attention total | 29.4465 | — | 59.1% |
| dense projections | 15.15 | — | 30.4% |

Attention's share of step GPU falls from 67.2% to 59.1%. The **backward key
pass** is now the largest single attention component by 2.1× over the query pass
and 3.9× over the forward — Phase IX's stated next bottleneck, unchanged and
still unimplemented here.

---

## 11. What changed, and what did not

`git diff --numstat` for the Phase III–X tree:

```
33   4   RDNA3_WMMA.md                                     (Phases III–VIII, untouched here)
133 37   shaders/transformer_attention_backward_key_rdna3.comp   (Phase VIII, untouched here)
6    1   shaders/transformer_attention_backward_query_rdna3.comp (Phase VII, untouched here)
1    3   src/bin/{held_out_eval, hierarchos-vulkan-training-qualification,
                 training-stability, wmma_drift}.rs              (Phases III–VIII, untouched here)
14   0   src/bin/transformer_train.rs                     (Phase IX backward counters, untouched here)
5    0   src/lib.rs                                        (+1 export line for the forward counters)
2957 149 src/transformer.rs                                (Phases III–IX + this phase)
15   0   src/vulkan.rs                                     (Phase IX, untouched here)
14   0   validation/training_shader_manifest.json          (this phase: one entry)
```

**Phase X specifically**

* new: `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.{comp,spv}`;
* `src/transformer.rs`: the `ROW_PACKED_ATTENTION_FORWARD_SPV` constant, the
  optional kernel field and its loader, the eligibility predicate, the dispatch
  counters + diagnostics, the one new selector branch, and a four-line
  `#[cfg(test)]` declaration for the new harness;
* `src/lib.rs`: one export line;
* `validation/training_shader_manifest.json`: one entry;
* new: `src/phase_x_row_packed_promotion_tests.rs` and this receipt.

Checked and unchanged:

* **generic streaming shader** — `transformer_attention_forward.comp/.spv` byte-identical;
* **short-sequence specialization** — byte-identical, and still selected first
  wherever its own gate holds (measured, §5);
* **MHA/GQA backward** — `transformer_attention_backward.*`,
  `…_query_rdna3.*`, `…_key_rdna3.*` byte-identical to the pre-phase state;
* **numerical contracts** — the promoted kernel's arithmetic is byte-identical
  to Phase IX, and every parity gate above is bitwise;
* **WMMA / precision policy** — not touched (`RDNA3_WMMA.md` and `wmma.rs` are as
  Phases III–VIII left them);
* **RWKV, unrelated Transformer ops** — not touched;
* **Phase VIII key-block experiment** — not reopened; its NO-GO candidate was not
  promoted;
* **prior receipts** — all six ledgers identical (§2).

No commit, PR or release was created. The index is still empty
(`e3b0c442…`), HEAD is still `560809ee…`, and `stash@{0}` is still
`831f8927…`.

---

## 12. Verdict

| # | promotion criterion | result |
| ---: | --- | --- |
| 1 | promoted bytes are the qualified candidate | **PASS** — GLSL `8f7bf06f…`, SPIR-V `c053503e…`, `cmp`-identical to Phase IX, rebuilt byte-for-byte |
| 2 | exact forward parity on the production artifact | **PASS** — 1,845,696 / 1,845,696 words, 10 geometries, 3 modes, 2 grids |
| 3 | backward and intermediate-state parity | **PASS** — dq/dk/dv plus probability/dS bitwise, including the structurally unwritten set |
| 4 | narrow, derived eligibility predicate | **PASS** — 24-case host proof, one reason per clause |
| 5 | no displacement of the short specialization | **PASS** — short still wins where its gate holds; measured |
| 6 | generic fallback preserved | **PASS** — 5 measured fallbacks, each bitwise equal to the generic kernel |
| 7 | optimized module not a prerequisite | **PASS** — `Option`, `.ok()`, `None` keeps the generic route |
| 8 | production selector actually dispatches it | **PASS** — counters 12/12 vs 0/0, `row_packed_installed` asserted per arm |
| 9 | release-mode training equivalence | **PASS** — 6 steps, loss + gradients + adapter + optimizer bitwise |
| 10 | performance attributable to this artifact | **PASS** — identical bytes, plus an independent 1.2239× `[1.2226, 1.2252]` |
| 11 | no collateral change | **PASS** — §11, with the ten shader hashes and six ledgers byte-identical |

**GO.** The Phase IX candidate is the production RDNA3 attention-forward path for
the geometries it was qualified on, every other geometry keeps the route it had,
and the complete-step improvement reproduces on the shipped selector.

**Known limits, stated rather than hidden.** Non-square (cross) attention and
`query_seq_len != key_value_seq_len` stay on the generic kernel because Phase IX
never measured them; sequences above 64 keys stay generic because the per-lane
score array is the module's structure; diagnostic arithmetic modes other than
0/1/9 stay generic for the same evidential reason. The repository-wide shader
provenance gate cannot run cleanly on this machine because of its toolchain
version pins, which is a pre-existing condition documented in §3 and worked
around with a scoped, byte-for-byte rebuild check.

The next bottleneck is unchanged from Phase IX: the **GQA backward key pass**
(16.93 ms/step, 34% of step GPU), which needs a bitwise gate of its own before
anything can move.

---

## 13. Files

| Path | What it is |
| --- | --- |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.{comp,spv}` | the promoted artifact (in `shaders/`, embedded by `include_bytes!`) |
| `src/phase_x_row_packed_promotion_tests.rs` | the production harness (`#[cfg(test)]`) |
| `raw/00_preservation.txt`, `raw/05_identity-end-of-phase.txt` | before/after repository identity |
| `raw/01_production-shader-hashes.txt`, `raw/02_prior-receipt-hashes.txt` | the untouched shaders and ledgers |
| `raw/03_promoted-artifact-hashes.txt`, `raw/21_promoted-artifact-rebuild.json` | artifact identity and reproducible rebuild |
| `raw/10_selector-matrix.log` | eligibility predicate, 24 cases |
| `raw/11_parity-forward.log` | forward matrix + dispatch proof |
| `raw/12_fallback.log`, `raw/13_short-precedence.log` | fallback and selector-order proofs |
| `raw/15_diagnostics.log` | the runtime dispatch trace, eligible and fallback |
| `raw/14_parity-gradients.log` | backward and scratch parity |
| `raw/step-{production,generic}.log`, `raw/23_step-digest-release.json` | release complete-step equivalence |
| `raw/debug/`, `raw/22_step-digest-debug.json` | the same equivalence in a debug build |
| `raw/abba-r*-*.log`, `raw/abba-summary.json` | release complete-step timing |
| `raw/timing-*.log`, `raw/timing-gpu.json` | GPU-timestamp kernel comparison |
| `raw/step-profile-*.log`, `raw/attribution.json` | per-component attribution |
| `raw/provenance.json`, `raw/20_provenance.log` | the repository-wide provenance gate and its limitation |
| `check_promoted_artifact.py`, `compare_steps.py`, `analyze_timing.py`, `attribute.py`, `assemble.py`, `summarize.py` | the phase's analysis scripts |
| `reproduce.sh` | rebuilds the artifact and reruns every check |
| `summary.json` | machine-readable receipt |

## 14. Reproduction

```bash
cd hierarchos-vulkan
bash validation/phase_x_row_packed_forward_promotion_20261125/reproduce.sh
```

Individual gates:

```bash
EXE=$(ls -t target/debug/deps/hierarchos_vulkan-*.exe | head -1)
$EXE phase_x_row_packed_selector_matrix          --nocapture --test-threads=1
$EXE phase_x_row_packed_forward_parity           --nocapture --test-threads=1
$EXE phase_x_row_packed_fallback_dispatch        --nocapture --test-threads=1
$EXE phase_x_row_packed_short_precedence         --nocapture --test-threads=1
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 \
  $EXE phase_x_row_packed_forward_gradient_parity --nocapture --test-threads=1
```

The first four are ordinary `cargo test` targets and need no environment
variable; the timing and full-step tests are opt-in so a qualification run can
never be confused with a measurement run.
