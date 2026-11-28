# Phase XX — production promotion of the Phase XIX query scratch-reuse variant

**Status: PROMOTED LOCALLY — READY FOR INDEPENDENT CLOSEOUT.**

The independently approved Phase XIX candidate is now a production artifact with
its own identity, loader, selector, dispatch counter and profiler name. The
promotion changes no shader arithmetic: the promoted SPIR-V is **byte-identical**
to the Phase XIX measured binary (`a6b3d6c6…`), rebuilt from the sealed source
with the manifest's exact flags, so the Phase XIX numerical and complete-step
performance evidence applies to the promoted module unchanged.

The Phase XVIII dQ-unroll module stays installed as the intermediate fallback,
and the original query shader remains the final fallback. Nothing was committed,
pushed, merged, tagged or released.

---

## 1. Starting state

| item | value |
| --- | --- |
| branch | `research/v0.2.8-optimization-discovery` |
| HEAD (start and end) | `b3cfc70675734b789b5e54061064b3dc8229b1ab` (`Promote RDNA3 dQ-unroll attention backward path`) |
| `main` | `9b0091e02e976441e343d6c445101f9c29395bd6` — unchanged |
| `v0.2.7^{commit}` | `9b0091e02e976441e343d6c445101f9c29395bd6` — unchanged |
| `release/v0.2.7` | `a32d09eba62f74c9876eef81312986eba0b42492` — unchanged |
| `stash@{0}` | preserved, untouched |
| Phase XVIII production artifact | `f0214b60…` — unchanged |
| Phase XIX receipt | byte-for-byte preserved (digest below) |
| commits / pushes / merges / tags / releases | none |

The working tree at the start matched the Phase XIX report exactly:

```text
 M src/transformer.rs
?? src/phase_xix_query_scratch_reuse_tests.rs
?? validation/phase_xix_query_scratch_reuse_20261127/
```

There are two different `raw/00_identity.txt` files and neither one is the other:

* the **starting** state is recorded in
  `validation/phase_xix_query_scratch_reuse_20261127/raw/00_identity.txt`,
  captured before any Phase XIX or Phase XX edit;
* the **ending** state — after Phase XX validation, with all nine promoted paths
  present — is recorded in
  `validation/phase_xx_query_scratchreuse_promotion_20261127/raw/00_identity.txt`
  by `preservation.sh`.

The Phase XX snapshot is therefore an end-of-phase record, not a copy of the
starting state: its `M`/`??` list is the promoted inventory, not the Phase XIX
one. Use the Phase XIX file for "what did Phase XIX start from" and this
receipt's file for "what did Phase XX end with".

### 1.1 Phase XIX receipt digest

Method (Phase XII–XIX): hash of the `sha256sum` lines of every file in sorted
path order. `raw/02_phase_xix_digest.txt`:

```text
078ef22eee85c82f34cbc0de62ee594d458464353fc7a2fb1cc84723af571392  files=195  bytes=2091594  validation/phase_xix_query_scratch_reuse_20261127
```

The receipt was not edited. Its 195 files, including all 160 per-process
benchmark logs, remain the performance and numerical evidence for this module.

---

## 2. Candidate identity and build reproduction

The review did not independently rebuild the candidate SPIR-V, so the promotion
rebuilt it from the sealed source with the manifest's exact invocation before
touching production code.

```text
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
  validation/phase_xix_query_scratch_reuse_20261127/shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp \
  -o <scratch>/q_scratchreuse.spv
```

Toolchain: `shaderc v2023.8 v2025.3`, `glslang 11.1.0-1253-gefd24d75`,
`spirv-tools v2025.3` (`raw/10_compile.txt`).

| artifact | sha256 | bytes |
| --- | --- | ---: |
| Phase XIX candidate source (sealed) | `219167927214d96a16f72d0ba755ee879a55eaf4b1e054233ce924d7b0ccef2a` | 16,903 |
| Phase XIX candidate SPIR-V (sealed) | `a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee` | 89,340 |
| rebuild from the sealed source | `a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee` | 89,340 |
| promoted source `shaders/…_scratchreuse.comp` | `219167927214d96a16f72d0ba755ee879a55eaf4b1e054233ce924d7b0ccef2a` | 16,903 |
| promoted SPIR-V `shaders/…_scratchreuse.spv` | `a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee` | 89,340 |
| Phase XVIII promoted SPIR-V (untouched) | `f0214b60fac0b5fe420b846f89f43fd3cb3229f4a12d30fc2e5ad80e25c9d237` | 109,300 |
| Phase XVIII promoted source (untouched) | `430492fd8ab9a1955fc89c5c37c4cc4a145bf03c009b00fedb3cd05e4a88dffc` | |
| original query fallback SPIR-V (untouched) | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` | |

**Rebuild result: byte-identical to the measured Phase XIX binary.** `cmp` also
proves the promoted files are byte-identical to the sealed receipt artifacts
(`raw/01_production_hashes.txt`, `raw/11_artifact_hashes.txt`).

`spirv-val --target-env vulkan1.2` passes on the rebuild, the promoted artifact,
the Phase XVIII module and the shipped fallback (`raw/12_spirv_val.txt`).

SPIR-V shape is unchanged from the reviewed candidate (`raw/15_spirv_analysis.txt`):

| quantity | value |
| --- | ---: |
| instructions | 3,704 |
| Function-storage `float[128]` arrays | 1 |
| constant-index chains into it | 128 distinct (0…127) |
| runtime-index chains | 4 (the untouched generic fallback body) |

No shader text, indexing, caching, masking, softmax-dot split, precision
qualifier, workgroup size, descriptor set or push block was modified. The
promoted source differs from the Phase XVIII source by exactly the Phase XIX
delta (`raw/13_source_delta.diff`, 212 lines including comments).

---

## 3. Production integration

| path | change |
| --- | --- |
| `shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp` | new production source (sealed Phase XIX bytes) |
| `shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.spv` | new production artifact (sealed Phase XIX bytes) |
| `src/transformer.rs` | embed constant, env switch, loader + cache, dispatch counter, selector wiring, test hooks, module registration |
| `src/phase_xx_query_scratchreuse_promotion_tests.rs` | new production-route tests and A/B driver |
| `src/phase_xix_query_scratch_reuse_tests.rs` | label/identity updates only; parity matrix unchanged |
| `src/phase_xviii_dq_accumulator_promotion_tests.rs` | Phase XX variant pinned off so the fallback keeps its own evidence |
| `validation/training_shader_manifest.json` | one new recipe, flags identical to Phase XVIII |
| `validation/phase_xx_query_scratchreuse_promotion_20261127/` | this receipt |

### 3.1 Naming, embedding and profiler identity

Production identity, following the repository convention:

* artifact stem: `transformer_attention_backward_query_rdna3_dqunroll_scratchreuse`
* profiler name: `transformer_attention_backward_query_rdna3_dqunroll_scratchreuse`
* route label: `query_dqunroll_scratchreuse`
* counter: `AttentionBackwardDispatch::QueryDqUnrollScratchReuse` →
  `AttentionBackwardDispatchCounts::query_dqunroll_scratch_reuse`

`build.rs` registers every `shaders/*.spv` by FNV-1a signature and derives the
debug name from the file stem, so adding the artifact is what gives the module a
profiler identity. Measured in `raw/21_route_tests.log`:

```text
[phase-xx-identity] artifact=scratchreuse signature=0x4d636e560c2d18a7 name=Some("transformer_attention_backward_query_rdna3_dqunroll_scratchreuse")
[phase-xx-identity] artifact=dqunroll     signature=0x0d0f9e255b0bc914 name=Some("transformer_attention_backward_query_rdna3_dqunroll")
[phase-xx-identity] artifact=original     signature=0xd02953248179151d name=Some("transformer_attention_backward_query_rdna3")
```

The artifact is embedded with `include_bytes!` exactly like the Phase XVIII
module. Manifest verification via `validation/audit_shader_artifacts.py`:

```text
source_sha256   = 219167927214d96a16f72d0ba755ee879a55eaf4b1e054233ce924d7b0ccef2a
embedded_sha256 = a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee
rebuilt_sha256  = a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee
matches = true
```

### 3.2 Eligibility — Phase XVIII rules, unchanged

The promoted variant inherits the Phase XVIII admission exactly. No wider
geometry allowlist was added and no helper-level parity result was promoted into
production eligibility.

**Query selector** (`dqunroll_query_attention_rejection`, unchanged):

* `qk_head_dim == 128` — reason `dqunroll_query_head_dim_not_128`
* `value_head_dim == 128` — reason `dqunroll_query_value_head_dim_not_128`
* degenerate `num_heads == 0 || num_key_value_heads == 0` — `invalid_geometry`

**Device qualification** (unchanged, checked only after the geometry passes):

* `vendor::rdna3_module_enabled(device)`
* `supports_compute_work_group_size([64, 1, 1])`
* 9 storage-buffer bindings, `size_of::<AttentionPush>()` = 120-byte push block

**Enclosing two-pass production gate** (unchanged): `two_pass_attention_enabled()`,
`key_blocks == 1`, `two_pass_attention_rejection` (modes 0/9, causal ≤ 1, no
window/softcap/alibi/bias/dropout/sink/offset/sparse/compressed/paged/value-scale,
`head_dim <= 128`, `kv_seq <= head_dim`, and for grouped geometries the
`QUALIFIED_GROUPED_QUERY_GROUPINGS = [(8,2),(4,2),(8,4),(16,4)]` allowlist at
`head_dim == 128`, `q_seq == kv_seq == 64`, `causal == 1`; MHA keeps its own
bypass), plus scratch-size overflow.

**No new geometry rule lives in the Phase XX switch.** `scratch_reuse_query_enabled()`
is `dqunroll_query_enabled() && setting != Some(false)` only.

### 3.3 Selector order and fallback

`record_two_pass_attention_backward_dispatch`, pass 1:

| order | condition | route | counter |
| ---: | --- | --- | --- |
| 0 | two-pass disabled, `key_blocks != 1`, outer gate rejects, or a harness pair owns the dispatch | `query_original` | `query` |
| 1 | `…_DQUNROLL=0` | `query_original` (`disabled_by_environment`) | `query` |
| 2 | geometry fails the 128/128 selector | `query_original` | `query` |
| 3 | scratch reuse enabled **and** its pipeline loaded | `query_dqunroll_scratchreuse` | `query_dqunroll_scratch_reuse` |
| 4 | scratch reuse disabled by environment | `query_dqunroll` (`scratch_reuse_disabled_by_environment`) | `query_dqunroll` |
| 5 | scratch-reuse load refused | `query_dqunroll` (`scratchreuse_shader_unavailable` / `scratchreuse_device_not_qualified`) | `query_dqunroll` |
| 6 | Phase XVIII load refused | `query_original` (`dqunroll_shader_unavailable` / `dqunroll_device_not_qualified`) | `query` |

The failure shape is the Phase XVIII one: a module-specific refusal
(`decoding embedded SPIR-V`, shader-module creation, pipeline creation) is an
ordinary fallback; device-lost, out-of-memory, poisoned layout lock and
unclassified failures stay fatal. A fatal condition is never converted into an
unexplained scalar dispatch.

Loading is lazy and cached with the kernel set: an ineligible model builds no
extra pipeline, and a refusal is not retried per step.

### 3.4 Environment precedence

| environment | effective route |
| --- | --- |
| unset (default) | scratch-reuse dQ-unroll |
| `HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0` | Phase XVIII dQ-unroll |
| `…_SCRATCH_REUSE=1` (explicit) | scratch-reuse dQ-unroll |
| `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0` | original query shader — wins over either scratch-reuse value |
| `…_DQUNROLL=0` **and** `…_SCRATCH_REUSE=1` | original query shader |
| `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0` | scalar kernel (inherited, unchanged) |

Unrecognized values print one diagnostic and keep the eligible default, matching
the Phase XVIII convention.

`raw/27_env_precedence.log` drives the real process environment through a
production layer (no test hook sets the switch) and records:

```text
SCRATCH_REUSE=0                    route=query_dqunroll          reason=scratch_reuse_disabled_by_environment
SCRATCH_REUSE=1                    route=query_dqunroll_scratchreuse reason=eligible
SCRATCH_REUSE=1 + DQUNROLL=0       route=query_original          reason=disabled_by_environment
```

---

## 4. Tests

All runs use the release test binary built from the working tree; raw logs are
in `raw/`. `--test-threads=1`, `--nocapture`.

| log | tests | result |
| --- | --- | ---: |
| `21_route_tests.log` | 8 Phase XX selector/identity/fallback tests | 8 passed |
| `22_route_ab.log` | production-route LoRA A/B | 1 passed |
| `23_phase_xix_parity.log` | Phase XIX historical parity matrix | 1 passed |
| `24_phase_xviii_fallback.log` | Phase XVIII fallback suite (pinned Phase XX off) | 4 passed |
| `25_two_pass_attention.log` | two-pass kernel/route tests | 2 passed |
| `26_phase_xii_channel4.log` | channel-4 key suite | 7 passed |

Required checks and where each is evidenced:

1. **Eligible 128/128 selects scratch reuse by default** —
   `phase_xx_scratch_reuse_default_route`: `route=query_dqunroll_scratchreuse`,
   `query_dqunroll_scratch_reuse=1`, all other query counters 0; resolved kernel
   name/signature equal the promoted artifact, and different from Phase XVIII.
2. **Opt-out selects Phase XVIII** — `phase_xx_scratch_reuse_opt_out_selects_dqunroll`:
   `query_dqunroll=1`, scratch counter 0, and the scratch pipeline is never built.
3. **Overall disable selects the original shader** —
   `phase_xx_dqunroll_disable_precedence`: `query=1` with `…_SCRATCH_REUSE=1`
   also set; neither promoted pipeline is built.
4. **Missing/refused candidate falls back to Phase XVIII** —
   `phase_xx_scratch_reuse_refused_falls_back_to_dqunroll`:
   `route=query_dqunroll reason=scratchreuse_shader_unavailable`,
   `query=0`, `scalar=0`; Phase XVIII reports `ready`.
5. **Ineligible geometry unchanged** — `phase_xx_ineligible_geometry_unchanged`:
   production-admitted MHA at head dim 64 and 32 (outer gate admits, query
   selector refuses) keep `query=1` and build neither query pipeline.
6. **Gates not broadened** — `phase_xx_production_gates_not_broadened`:
   exact reason strings for head-dim 64/value-head-dim 64, grouped
   sequence-128, grouped non-causal, unqualified grouping; the grouped
   allowlist and MHA bypass are unchanged.
7. **Distinct profiler identities** — `phase_xx_profiler_identities_distinct`:
   three names, three distinct signatures (§3.1).
8. **No cross-attribution** — every route test asserts the other query counters
   are exactly zero, and the A/B counters below differ only in the pass-1 field.

### 4.1 Actual production-route end-to-end A/B

Same fixture and methodology as the Phase XIX complete-step evidence: 8q/2kv,
sequence 64, batch 2, LoRA rank 8 / alpha 16, lr `1e-2`, seed 7,
`target/gqa-e2e/fixture-8q2kv` (32 sequences), 29 complete optimizer steps per
arm, no profiler.

Both arms exercise the real production dispatch function
(`record_two_pass_attention_backward_dispatch`), the production loader, device
qualification, the production selector, the production buffers, the production
dispatch path and the production counters. The production pair is admitted with
`reason == "ready"` in both arms, and neither arm uses
`experimental_pair_override`.

Arm selection is deterministic and is **not** read from the process environment.
The paired driver injects the scratch-reuse switch value through the existing
`#[cfg(test)]` `scope_scratch_reuse` hook:

* **Arm A** — `scope_scratch_reuse(Some(Some(false)))` gives the production
  selector a pinned-off switch, so it takes its Phase XVIII dQ-unroll fallback
  arm (`scratch_reuse_disabled_by_environment`).
* **Arm B** — `scope_scratch_reuse(Some(None))` leaves the switch at its
  production default, so the selector takes the scratch-reuse arm.

The paired comparison therefore proves the production selector's two arms agree
numerically; it does not test environment-variable parsing. Actual environment
parsing, default behaviour, opt-out and precedence are verified separately by the
real-process runs recorded in `raw/27_env_precedence.log` (§3.4).

`raw/22_route_ab.log`:

```text
[phase-xx-route-ab] fixture=target/gqa-e2e/fixture-8q2kv sequences=32 batch=2 seq=64 steps=29 lora=true lr=0.01 arm_a=dqunroll arm_b=scratchreuse
[phase-xx-route-ab-end] steps=29 comparisons=87 mismatches=0 loss_bits_equal=true \
  arm_a_query_original=0 arm_a_query_dqunroll=58 arm_a_query_scratchreuse=0 arm_a_scalar=0 \
  arm_a_key_original=0 arm_a_key_channel4=58 \
  arm_b_query_original=0 arm_b_query_dqunroll=0 arm_b_query_scratchreuse=58 arm_b_scalar=0 \
  arm_b_key_original=0 arm_b_key_channel4=58 \
  param_end_a=d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655 \
  param_end_b=d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655 verdict=PASS
```

| check | result |
| --- | --- |
| per-step loss bits equal, 29/29 | yes |
| per-step LoRA gradient digests equal, 29/29 | yes |
| per-step parameter digests equal, 29/29 | yes |
| comparisons / mismatches | 87 / 0 |
| parameter end digest | identical (`d79a12d8…9c4655`) |
| arm A pass-1 counter | `query_dqunroll = 58` (2 layers × 29 steps), scratch counter 0 |
| arm B pass-1 counter | `query_dqunroll_scratch_reuse = 58`, dq-unroll and original counters 0 |
| scalar dispatches (both arms) | 0 |
| pass 2 | `key_channel4 = 58`, `key_original = 0` in both arms |

The end digest and loss trajectory are the Phase XIX in-graph LoRA values
(`raw/30_in_graph_scratch.log` in the sealed receipt), so the promoted route is
continuous with the reviewed evidence, and the two production routes are
bit-identical on this fixture.

### 4.2 Historical helper-level parity remains valid

`23_phase_xix_parity.log` — the Phase XIX matrix is preserved and now reads the
sealed receipt bytes, asserting at run time that they are the promoted artifact:

```text
[phase-xix-parity] summary cases=28 specialized=26 generic=2 tensor_comparisons=140 failed=0
[phase-xix-identity] promoted_name=Some("transformer_attention_backward_query_rdna3_dqunroll") \
  candidate_name=Some("transformer_attention_backward_query_rdna3_dqunroll_scratchreuse")
```

The historical per-case label was renamed
`promoted_selector_admits` → `dqunroll_head_dim_selector_admits`, with a comment
stating that it is only the head-dimension half of the gate. **These 28 cases
bypass the production selectors entirely** (harness pair), and of the 26
specialized cases only 15 are reachable from a real production step.

`24_phase_xviii_fallback.log` re-runs the Phase XVIII matrix against its own
module, pinned through `scope_scratch_reuse(Some(Some(false)))`:
`cases=30 specialized=28 fallback=2 tensor_comparisons=150 failed=0`. This is the
fallback requalification *by reuse*: no Phase XVIII campaign was re-run and no
tolerance changed.

---

## 5. Performance evidence

**Not re-measured.** The 160-process, five-round ABBA campaign was not repeated,
and no new timing was taken. That is deliberate: the promoted SPIR-V is
byte-identical to the measured candidate, and the production integration
preserves the dispatch geometry (grid `[ceil(batch·heads·seq / 64), 1, 1]`),
64-wide workgroup, 9 bindings, 120-byte push block and both scratch buffers. No
new timing could add information about a module whose bytes did not change.

Reused Phase XIX complete-step evidence, for the exact promoted binary:

| mode | fixture | A (ms) | B (ms) | B/A | speedup | rounds improved | 95 % CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 8q/2kv seq64 (primary) | 26.799 | 24.643 | 0.9191 | **1.088×** | 5/5 | 0.9176–0.9210 |
| Mixed | 8q/2kv seq64 (primary) | 21.386 | 19.111 | 0.8932 | **1.120×** | 5/5 | 0.8874–0.8977 |

All three additional regression series improved (8q/8kv seq64 and seq128, in
both modes; 1.071×–1.148×). Isolated pass-1 kernel timing: 1.53× (mode 0), 1.76×
(mode 9). The isolated round-0 promoted outlier (3.607 ms) remains disclosed.

Carried forward, unchanged:

* One Phase XVIII baseline comparison sits at **+0.346 %** against the ±0.35 %
  tolerance.
* Complete-step ABBA results are the headline evidence; isolated kernel timings
  are not a quantitative predictor of every complete-step improvement.
* No AMD ISA dump was obtained on this host; **no ISA-level register or occupancy
  claim is made**.

This section is a reference to `validation/phase_xix_query_scratch_reuse_20261127/`,
not a copy of its raw collection.

---

## 6. What the promotion does not change

* No shader source or binary other than the two new files.
* Channel-4 key, row-packed forward, WMMA and Mixed WMMA defaults untouched
  (Mixed WMMA remains default-off).
* Precision thresholds, tolerances and comparison methodology untouched.
* No wider geometry allowlist, no change to sequence/causal/MHA/GQA handling.
* No other optimization started; no historical campaign re-run.

---

## 7. Limitations

1. **15 of 26 specialized parity cases are production-reachable; 11 are
   helper-level only** (the two non-causal 8q/2kv seq-64 cases, the four 8q/2kv
   seq-128 cases, the non-causal 4q/2kv seq-64 case, the two 4q/2kv seq-128 cases,
   the two 4q/2kv seq-37 cases).
2. **All 28 Phase XIX parity cases bypass the production selectors.** Production
   evidence is the §4.1 A/B run and its counters, not those rows.
3. **Output write coverage is not independently measured.** Both arms pre-fill
   outputs and scratch with the same sentinel, so the comparison proves agreement
   everywhere it compares; it cannot distinguish a slot both leave at the sentinel
   from one both write identically.
4. **Exactness of the store/load round trip is not mathematically established for
   every possible subnormal input.** Source-level arithmetic is unchanged and the
   compared tensors are bitwise identical; that is an empirical result and a
   property of the unchanged expressions, not a proof over all inputs.
5. **Measurement qualifications** from Phase XIX remain in force: the arm-A
   boundary result (+0.346 % vs ±0.35 %), the profiled-attribution caveat (stride-1
   sampling; profiled totals are not step timings), and the isolated mode-0
   round-0 outlier.
6. **No AMD ISA evidence exists on this host.** The claim made is the SPIR-V one
   (§2), not a VGPR or occupancy statement.
7. **Pre-existing failing legacy assertion, not caused by this phase.**
   `grouped_query_attention_production_route_dispatches_both_passes` still asserts
   `counts.query == 1` on an 8q/2kv head-dim-128 geometry. That assertion fails
   whenever the Phase XVIII dQ-unroll route is selected: with the Phase XX variant
   explicitly disabled (`…_SCRATCH_REUSE=0`) the same test routes
   `query_dqunroll` and fails identically at `left: 0, right: 1`
   (`raw/27_env_precedence.log`). The test file was left untouched; updating it
   would be Phase XVIII requalification, which this phase is told not to do.
8. **The A/B run is a dispatch-integration check, not a performance
   measurement.** It reads parameters back every step (`HIERARCHOS_PHASE_XX_ROUTE_AB`
   runs with digests enabled), which is why no timing is derived from it.
9. **Two unused test-only allocations remain** in `phase_xix_median_ms`
   (`grad_k`, `grad_v`). They are outside the timed region and were left alone.

---

## 8. Changed files

All uncommitted, none staged. HEAD and branch unchanged.

```text
 M src/phase_xviii_dq_accumulator_promotion_tests.rs
 M src/transformer.rs
 M validation/training_shader_manifest.json
?? shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp
?? shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.spv
?? src/phase_xix_query_scratch_reuse_tests.rs
?? src/phase_xx_query_scratchreuse_promotion_tests.rs
?? validation/phase_xix_query_scratch_reuse_20261127/
?? validation/phase_xx_query_scratchreuse_promotion_20261127/
```

Phase XIX files are preserved, not rewritten: `src/phase_xix_query_scratch_reuse_tests.rs`
received the label rename, the identity assertions and a Phase XX route pin in its
historical driver (§4.2), plus `pub(super)` on three helpers the Phase XX driver
reuses. No parity case, comparison or tolerance changed, and the sealed receipt
directory is byte-identical (digest `078ef22e…`).

---

## 9. Reproduction

```text
cd hierarchos-vulkan

# 1. identity, production hashes, sealed Phase XIX digest
bash validation/phase_xx_query_scratchreuse_promotion_20261127/preservation.sh

# 2. rebuild the promoted artifact and prove byte identity with Phase XIX
bash validation/phase_xx_query_scratchreuse_promotion_20261127/reproduce_shaders.sh

# 3. release test binary
cargo test --release --lib --no-run
BIN=$(ls -t target/release/deps/hierarchos_vulkan-*.exe | head -1)

# 4. production-route selector/identity/fallback suite
$BIN phase_xx_ --nocapture --test-threads=1

# 5. one actual production-route A/B (arm A = Phase XVIII, arm B = scratch reuse)
HIERARCHOS_PHASE_XX_ROUTE_AB=1 \
  $BIN phase_xx_production_route_ab --nocapture --test-threads=1

# 6. historical parity, preserved
$BIN phase_xix_scratch_reuse_parity --nocapture --test-threads=1
$BIN phase_xviii_ --nocapture --test-threads=1
$BIN two_pass_attention --test-threads=1
$BIN phase_xii_channel4 --test-threads=1

# 7. real environment-variable precedence, through a production layer
HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0 \
  $BIN grouped_query_attention_production_route_dispatches_both_passes --nocapture
HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=1 \
  $BIN grouped_query_attention_production_route_dispatches_both_passes --nocapture
HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=1 \
HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0 \
  $BIN grouped_query_attention_production_route_dispatches_both_passes --nocapture

# 8. manifest reproduction for the new recipe
python validation/audit_shader_artifacts.py \
  transformer_attention_backward_query_rdna3_dqunroll_scratchreuse \
  --compiler-flag=-fshader-stage=compute --compiler-flag=--target-env=vulkan1.2 \
  --compiler-flag=--target-spv=spv1.4 --compiler-flag=-O --compiler-flag=-Ishaders
```

---

## 10. Raw artefacts

| file | contents |
| --- | --- |
| `raw/00_identity.txt` | **ending**-state snapshot (after Phase XX validation): HEAD, branch, refs, stash, `git status`, diff digests. The **starting** snapshot is the Phase XIX receipt's `raw/00_identity.txt`; these are different files |
| `raw/01_production_hashes.txt` | production + sealed identities, byte comparisons, source hashes |
| `raw/02_phase_xix_digest.txt` | sealed Phase XIX receipt digest and file count |
| `raw/10_compile.txt`, `raw/11_artifact_hashes.txt`, `raw/12_spirv_val.txt` | rebuild, byte-identity proofs, SPIR-V validation |
| `raw/13_source_delta.diff`, `raw/14_scratchreuse.dis`, `raw/15_spirv_analysis.txt` | source delta and SPIR-V shape |
| `raw/21_route_tests.log` | eight Phase XX selector/identity/fallback tests |
| `raw/22_route_ab.log` | production-route LoRA A/B, 29 steps × 2 arms |
| `raw/23_phase_xix_parity.log` | historical 28-case / 140-comparison parity matrix |
| `raw/24_phase_xviii_fallback.log` | Phase XVIII fallback suite, pinned Phase XX off |
| `raw/25_two_pass_attention.log`, `raw/26_phase_xii_channel4.log` | neighbouring attention-backward suites |
| `raw/27_env_precedence.log` | real environment-variable routing and the pre-existing legacy assertion |
