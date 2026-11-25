# Phase XII — production promotion of the RDNA3 four-channel key pass

**Result: GO.** The exact Phase XI `c4_u0` artifact is now the production key
pass for head-dimension-128 geometries that already passed the attention
gate. The complete training step improves **1.5297×** (43.530 ms → 28.460 ms,
five paired ABBA rounds, 20 separate release-mode processes) with byte-identical
training artifacts, and the original qualified two-pass key shader remains
installed for every other geometry.

Nothing is committed. No query shader, scalar shader, attention forward, WMMA,
precision policy, RWKV or unrelated Transformer path was changed. All Phase
III–XI work is intact and re-verified.

---

## 1. Preservation evidence

Base `560809eead186f832344c15d1a9b70795da3b75d`, branch `main`. Captured before
any edit (`raw/00_identity.txt`) and re-captured at receipt time
(`raw/09_final_identity.txt`):

| item | value |
| --- | --- |
| HEAD | `560809eead186f832344c15d1a9b70795da3b75d` (unchanged) |
| branch | `main` |
| staged index tree | `4709d4508ca02c854741617b379795929f673760` (unchanged) |
| staged diff | empty (`e3b0c442…`) |
| `stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` (unchanged) |
| binary-only diff (`*.spv`) sha256, pre-edit and receipt time | `cfa0d1c059d1dab6500f23ec981e2e4fdfc6c47bd43c1d86957ee686f8d7319a` (identical) |
| full diff (`--binary`) sha256, pre-edit and receipt time | `3a671c752ccf8bb7ad714781fae3f11e6264cd6085001da9caf9896c7244853f` (identical) |
| tracked text diff sha256, pre-edit → receipt time | `d83c0868…` → `1842686a…` (expected: this phase edits tracked sources) |

Pre-edit values are in `raw/00_identity.txt`; the receipt-time state is
`raw/09_final_identity.txt`. The binary-only diff digest is **identical to its
pre-edit value**: no pre-existing `.spv` in the working tree moved. The tracked
text diff digest necessarily changed, because this phase edits
`src/transformer.rs`, `src/lib.rs` and the shader manifest.

Production artifacts, verified against the Phase XI `raw/03` list with
`sha256sum -c` (`raw/04_phase-xi-production-hashes.txt`, all six `OK`):

| artifact | sha256 |
| --- | --- |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `e4179354…6ff78` |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `ccf60984…983a7` |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720…7eb56` |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f52…04256` |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp` | `8f7bf06f…e6ed` |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv` | `c053503e…764fd` |

Prior receipt directories were measured at the start of this phase
(`raw/05_prior_receipts.txt`). The digest is a recursive sha256 over every
file, sorted by path:

| directory | files | bytes | identity |
| --- | ---: | ---: | --- |
| `mha_attention_backward_closeout_20261124` | 36 | 787,175 | `bf2bde0f…a15e2` |
| `gqa_attention_backward_experiment_20261124` | 13 | 249,967 | `ebde677a…8852d` |
| `gqa_production_promotion_20261124` | 14 | 553,326 | `25e2a128…09107` |
| `gqa_end_to_end_training_20261124` | 178 | 4,559,007 | `4c1b4d96…df21a` |
| `phase_viii_key_blocks` | 26 | 2,242,018 | `b15767c3…fcd5f` |
| `phase_ix_row_packed_forward_20261124` | 61 | 1,771,186 | `5d24674f…594f8b` |
| `phase_x_row_packed_forward_promotion_20261125` | 66 | 2,538,532 | `b6efb85e…9ef49f` |
| `phase_xi_channel_tiling_20261125` | 69 | 2,068,107 | `048ee479…3b6948` |

The Phase XI numerical gate was re-run unchanged: `status=PASS cases=90
tensors=180 elements=11796480 unwritten=0 failures=0`
(`raw/11_phase-xi-unchanged.log`), and the Phase IV two-pass fallback suite
still reports its original eight cases (`raw/12_two-pass-fallback.log`).

No reset, clean, rebase, discard, overwrite or commit was performed.

---

## 2. The promoted artifact is the measured `c4_u0`

Source and artifact:

| item | value |
| --- | --- |
| source | `shaders/transformer_attention_backward_key_channel4_rdna3.comp` |
| source sha256 | `e1373db7ea1e7e0a1ac700ec7a9d464571f8b24f6065dbcc23c7d6df286e78f9` |
| artifact | `shaders/transformer_attention_backward_key_channel4_rdna3.spv` |
| artifact sha256 | `f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8` |
| Phase XI `c4_u0` SPIR-V | `f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8` |

`cmp` is byte-equal in both directions: the promoted source is the Phase XI
source and the promoted SPIR-V is `key_chantile_c4_u0.spv`. `glslc -E` output is
identical to the experimental build after removing `#line` filename markers
(`raw/preprocessed-promoted.glsl` against `raw/preprocessed-experimental.glsl`),
so the only textual difference is the path in the line markers.

Compiler recipe, reproduced in `raw/02_shader-rebuild.txt`:

```text
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O \
      -Ishaders -DXI_CHANNEL_CHUNK=4 -DXI_CHANNEL_BLOCKS=32 \
      -DXI_UNROLL_CHANNEL=0 \
      shaders/transformer_attention_backward_key_channel4_rdna3.comp \
      -o shaders/transformer_attention_backward_key_channel4_rdna3.spv
spirv-val --target-env vulkan1.2 <artifact>        # OK
```

Toolchain: shaderc 2025.3 (reports 2023.8), SPIRV-Tools/`spirv-val` 2025.3,
workgroup 64×1×1, chunk 4, blocks 32. The new artifact is registered in
`validation/training_shader_manifest.json` with those flags, and the binary's
shader-name registry (generated from `shaders/*.spv` by `build.rs`) now names it
`transformer_attention_backward_key_channel4_rdna3`, so profiler output no longer
carries an `unknown` label.

The previous production key shader is **retained**: `transformer_attention_backward_key_rdna3`
keeps its exact identity `e4179354…` / `ccf60984…` and is still the selected key
pass for every geometry the tiled path was not qualified on.

---

## 3. Production selector

The two-pass backward now resolves one optional tiled artifact in addition to
the Phase IV pair:

| type | role |
| --- | --- |
| `TwoPassAttentionBackward` | query pass + original key pass, unchanged |
| `Channel4KeyBackward` | optional four-channel key pass (`kernel: Option<…>`) |

Selection order in `record_two_pass_attention_backward_dispatch` (the function
the training step calls):

1. pass 1 — the qualified query shader, always;
2. key pass — `channel4_key_attention_rejection(push) == None` **and**
   `Channel4KeyBackward::load` returned a kernel **and** the pair is the
   production pair (`reason == "ready"`), then the four-channel artifact;
3. otherwise the original qualified two-pass key shader;
4. if the required pair itself is unavailable, the pre-existing scalar route is
   unchanged.

The tiled artifact is never a prerequisite: `TransformerKernels::new` builds
nothing, the loader is reached only from an eligible dispatch, and a failed load
is cached so it is not retried every step. `=0` on the two-pass switch also
disables the tiled path, so a scalar run builds neither.

The only new operator control is the narrow opt-out
`HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0`, which restores the original key
shader and builds no tiled pipeline. It exists because the release A/B needs the
two production key shaders in separate processes without changing anything else;
it is also the documented escape hatch if a future driver rejects the artifact.

Diagnostics name the executed route per geometry:

```text
two-pass backward: route=two-pass        ...   # pair resolved
two-pass backward: route=key_channel4   ...   # promoted key pass
two-pass backward: route=key_original   ...   # original key pass + reason
two-pass backward: route=scalar         ...   # existing fallback
```

and the counters distinguish all three outcomes:
`AttentionBackwardDispatchCounts { scalar, query, key, key_channel4,
scratch_allocations, scratch_bytes }`, where `key` is `key_original`. A route is
only reported as channel-tiled when `key_channel4` incremented.

### Head-dimension gate

`channel4_key_attention_rejection` is deliberately narrow:

```text
head_dim       == 128      (the only dimension Phase XI qualified)
value_head_dim == 128
blocks * chunk == 128       (32 * 4, a compile-time assertion)
```

`head_dim % 4 == 0` is **not** accepted. 16/32/64 keep the original key shader
until they are separately qualified. Everything else — arithmetic mode, causal
mode, window, softcap, ALiBi, bias, dropout, sink, sparse mask, compressed keys,
paged KV, `key_blocks == 1`, MHA-or-qualified-grouping — is still decided by the
existing attention gate, which was not broadened.

---

## 4. Dispatch geometry and ownership

The promoted source keeps the Phase XI mapping exactly:

```text
slot  = gl_GlobalInvocationID.x
chunk = slot % 32
row   = slot / 32
c0    = chunk * 4
grid  = [ceil(rows * 32 / 64), 1, 1]
```

At the primary 8q/2kv geometry (`rows = 2 · 2 · 64 = 256`):

| quantity | value |
| --- | --- |
| lanes per key row | 32 |
| lane slots | 8,192 |
| workgroups | 128 |
| invocations | 8,192 |
| channels per row | 128 |
| output elements (dK + dV) | 65,536 |

This is the same lane set the Phase XI candidate was measured with (its
ownership report lists `workgroups=128` for `c4_u0` at 8q/2kv). The host replay
in `phase_xii_channel4_key_ownership` shows `covered=65536 missing=0
duplicated=0 useful_lanes=8192`, so every destination element is owned exactly
once and every launched lane is useful. The MHA control (8q/8kv) gives 512
workgroups, 32,768 lanes, 262,144 elements, same verdict.

No new scratch exists: the tiled kernel binds the same eight buffers as the
original key pass, and reuses the pair's `probability`/`ds` scratch. The Phase IV
capacity formula (`batch · num_heads · q_seq · kv_seq`) is asserted unchanged.

---

## 5. Numerical qualification against the production artifact

`raw/10_phase-xii-gates.log`, test `phase_xii_channel4_key_parity`. Every case is
dispatched through `record_two_pass_attention_backward_dispatch` with the
embedded production artifact; the same run also dispatches the original key pass
through the production opt-out. Both arms are compared to the scalar oracle.

```text
[phase-xii-parity-summary] status=PASS cases=20 tensors=60 elements=5505024 \
  unwritten=0 failures=0
```

Matrix: 5 geometries (8q/2kv, 4q/2kv, 8q/4kv, 16q/4kv, 8q/8kv MHA control) × 2
arithmetic modes (0, 9) × 2 arms × 3 tensors (dQ, dK, dV). Every comparison is a
bit-pattern comparison; maximum absolute error 0, maximum ULP difference 0,
unwritten elements 0 (sentinel-detected, scratch sentinel-filled *before* pass
1 so a stale read cannot reproduce the oracle). The route is asserted per case:
`key_channel4=1 key_original=0 scalar=0` for the eligible default and
`key_original=1 key_channel4=0` for the opt-out.

The Phase XI matrix had 90 cases / 180 tensors / 11,796,480 elements because it
compared nine candidates against two baselines. This phase has one promoted
artifact, two routes and one baseline, so its expected count is
5 · 2 · 2 · 3 = 60 tensors / 5,505,024 elements. It is not a reduced matrix.

Bit-exactness of dQ also re-proves the query pass: the tiled kernel binds no dQ
buffer and pass 1 is the unchanged `f9652f52…` artifact.

---

## 6. Original-key fallback and unavailable-artifact fallback

**Head dimension 64, 1q/1kv MHA** (`phase_xii_channel4_key_original_fallback`):
the geometry is admitted by the existing two-pass gate, the tiled gate returns
`channel4_head_dim_not_128`, the original key shader dispatches, the scalar
shader does not, the tiled artifact is never resolved (`channel4_reason=None`),
and dQ/dK/dV are bit-identical to the scalar oracle.

**Head dimension 128, tiled artifact unavailable**
(`phase_xii_channel4_key_unavailable_falls_back_to_original`): the loader records
`channel4_shader_unavailable`, the production dispatch runs
`query=1 key_original=1 key_channel4=0 scalar=0`, two dispatches, and dK/dV are
bit-identical to the oracle. The fault is injected through the real loader (a
SPIR-V blob the driver cannot decode), not by skipping the load.

---

## 7. Optional loading and failure semantics

`phase_xii_channel4_key_optional_loading`, all through the production loaders:

| case | result |
| --- | --- |
| `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0` | pair and tiled artifact both `disabled_by_environment`; 0 optional kernels built |
| eligible default | `pair=ready`, `channel4=ready`; 3 optional kernels built |
| tiled artifact undecodable | `pair=ready`, `channel4_shader_unavailable`; 3 attempts, original key pass selected |
| `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` | `disabled_by_environment`; 0 tiled pipelines |
| required pair unavailable | `both_shaders_unavailable`, 2 attempts, scalar route (existing test) |
| explicit `=1` with a broken pair | `BLOCKED: … query_shader_unavailable`, no silent fallback |

The pre-existing `optional_two_pass_load_failure_keeps_scalar_execution_and_gradients`
still passes unchanged (`raw/12_two-pass-fallback.log`): default `query=1 key=1
scalar=0`, `=0` loads 0, every pair failure keeps scalar execution, GQA and
long-context geometries never build the optional kernel. Fatal device-loss and
out-of-memory classification is untouched.

---

## 8. Release-mode training equivalence

Two 29-step release runs, same fixture/model/data/seed/optimizer, LoRA rank 8 /
alpha 16, lr 1e-2, batch 2, seq 64, two layers, hidden 1024, 8q/2kv, head
dimension 128, production row-packed forward, production query pass.
`raw/30_equivalence-original.log` and `raw/31_equivalence-channel4.log`; the
normalized diff `raw/32_equivalence.diff` is **empty** (0 bytes) after removing
wall-clock `step_ms` and the two route counters, which are asserted separately.

Identical in both arms:

* every step's `loss_bits` (29 of 29; the concatenated loss-bit stream hashes to
  `fe96417e…` in both runs);
* every step's `grad_digest`, `grad_elements`, `grad_norm` (gradient stream hash
  `88bd5ee4…` in both);
* every step's `param_digest`;
* `parameter_end_digest` `d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655`
  (13,636,608 elements);
* both training checkpoints, file by file:
  `adapter_config.json` `1fa65722…`, `adapter_model.safetensors` `8a64128b…`,
  `hierarchos_adapter_optimizer.safetensors` `fa38c404…`.

Dispatch counters over 29 steps: 58 row-packed forward, 58 query, 58 key
(2 of each per step), 0 scalar. Arm A records all 58 as `key_original`; arm B
records all 58 as `key_channel4`.

Measurement ordering is recorded in `raw/08_measurement-ordering.txt`: the release
binary used for the ABBA was built at 02:49, the first ABBA log is 02:53, and the
only source file changed afterwards is the test module (02:55).
`src/transformer.rs` (02:38) and both shaders (02:27) predate the binary, and the
final release binary (03:06) contains the same production files — its 7/7
Phase XII tests were re-run from that binary. A later reproduction attempt
rewrote the `.spv` with identical bytes (mtime 03:08, sha256 still
`f5480cce…`), so the measured production route is the final production route.

---

## 9. Five-round ABBA (fresh Phase XII production measurement)

Protocol: three warmup steps, 26 measured steps, five rounds of A B B A, 20
separate release-mode processes, unprofiled, same device and driver. Arm A =
original production key shader (`…KEY_CHANNEL4=0`), arm B = promoted production
key shader (default). Raw per-process logs: `raw/abba-r{0..4}-p{0..3}-{A,B}.log`;
reduction: `raw/33_abba.json`.

| quantity | arm A (original) | arm B (channel4) |
| --- | ---: | ---: |
| median complete-step | **43.530 ms** | **28.460 ms** |
| mean complete-step | 43.533 ms | 28.467 ms |
| 95% CI of the mean | 43.511–43.554 ms | 28.433–28.501 ms |
| parameter-end digest | `d79a12d8…` | `d79a12d8…` |

Per-round B/A ratios (mean of the two pairings in each round):

| round | A (ms) | B (ms) | paired ratios | round ratio |
| --- | --- | --- | --- | ---: |
| 0 | 43.510 / 43.503 | 28.425 / 28.515 | 0.653292 / 0.655476 | 0.654384 |
| 1 | 43.559 / 43.592 | 28.482 / 28.437 | 0.653856 / 0.652337 | 0.653096 |
| 2 | 43.532 / 43.506 | 28.386 / 28.419 | 0.652076 / 0.653211 | 0.652644 |
| 3 | 43.528 / 43.573 | 28.439 / 28.500 | 0.653338 / 0.654072 | 0.653705 |
| 4 | 43.540 / 43.484 | 28.517 / 28.551 | 0.654967 / 0.656578 | 0.655773 |

Median round ratio **0.653705**, mean 0.653920, 95% CI 0.652400–0.655441,
**5/5 rounds improved**, speedup **1.5297×**.

This reproduces the Phase XI experimental result (43.559 → 28.485 ms, 1.528×)
through the real production selector and dispatch. It is a fresh Phase XII
measurement; the Phase XI numbers are historical and are not substituted here.

---

## 10. Updated attribution (fresh production profile)

Profiled release runs, 9 steps each, `HIERARCHOS_VULKAN_PROFILE_KERNELS=1`
`HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`; GPU ms per training step
(`raw/42_attribution.json`). A profiled run perturbs the step it measures, so
these numbers attribute work between components; the headline timings are the
unprofiled ABBA above.

| shader | arm A ms/step | share | arm B ms/step | share |
| --- | ---: | ---: | ---: | ---: |
| key pass (`…_key_rdna3` / `…_key_channel4_rdna3`) | **16.983** | **34.05%** | **0.541** | **1.54%** |
| query pass | 8.214 | 16.47% | **8.439** | **23.96%** |
| `linear_forward` + `_lane4` | 7.916 | 15.87% | 8.256 | 23.45% |
| `linear_input_grad` + `group4_muladd` | 6.947 | 13.93% | 7.486 | 21.26% |
| attention forward (row-packed) | 4.402 | 8.83% | 4.756 | 13.51% |
| RMS norm forward + input grad | 3.436 | 6.89% | 3.609 | 10.25% |
| cross-entropy (lanes16) | 1.420 | 2.85% | 1.523 | 4.32% |
| everything else | 0.555 | 1.11% | 0.705 | 2.00% |
| **attributed total** | **49.874** | | **35.214** | |

**The new largest single kernel is the GQA backward query pass** (8.439 ms/step,
23.96%), followed by `linear_forward` (8.256, 23.45%) and `linear_input_grad`
(7.486, 21.26%). Attention forward is 4.756 (13.51%). The key pass leaves the
top of the distribution: 16.98 → 0.54 ms/step in the production binary, a
−16.44 ms/step change that matches the −15.07 ms complete-step delta within the
profiler's own perturbation.

Query-pass work remains out of scope for this phase.

---

## 11. Reproduction

```text
cd hierarchos-vulkan
bash validation/phase_xii_channel4_key_production_20261125/reproduce.sh
```

Sections 1–4 need only a debug build. Section 5 rebuilds release (~10 min on
this machine); 6–8 re-derive equivalence, the ABBA and the attribution. Every
output lands in `raw/`; `raw/99_summary.txt` and `raw/99_summary.json` are the
assembled summaries. Section 2 deliberately rebuilds into `raw/` and compares,
so the shipped artifact's mtime (which `build.rs` watches) is never bumped.

One script attempt is kept as `raw/97_reproduce-attempt-link-failure.log`: the
link step returned Windows `0xc0000142` (`STATUS_DLL_INIT_FAILED`) after the
parent shell had been cancelled, which is a transient linker-process launch
failure on this machine, not a source or toolchain problem. The release binary
from the same tree (03:06) linked normally and passed all seven Phase XII tests
when executed directly; `raw/20_release-build.log` records the successful
build.

Fast paths:

```text
cargo test --lib phase_xii_channel4 -- --nocapture --test-threads=1
cargo test --lib phase_xi_channel_tiling -- --nocapture --test-threads=1
```

Device: `AMD Radeon Graphics`, vendor `0x1002`, device `0x15bf` (integrated
RDNA3), single driver, quiet measurement environment.

---

## 12. Scope limitation and files added

**The channel-tiled key pass is promoted only for head dimension 128.** MHA and
the four previously qualified grouped geometries use it where the existing
eligibility gate admits them; every other supported geometry (head dimension
16/32/64) keeps the original two-pass key shader, and the scalar kernel remains
the fallback when the two-pass architecture itself is unavailable. Generalizing
to `head_dim % 4 == 0` requires a separate numerical qualification of those
dimensions and is explicitly not done here.

| file | role |
| --- | --- |
| `shaders/transformer_attention_backward_key_channel4_rdna3.comp` | promoted source, byte-identical to the Phase XI source |
| `shaders/transformer_attention_backward_key_channel4_rdna3.spv` | promoted artifact, byte-identical to `c4_u0` |
| `src/transformer.rs` | optional artifact, narrow gate, production dispatch, counters, route diagnostics |
| `src/phase_xii_channel4_key_production_tests.rs` | parity, ownership, gate, fallbacks, loading, ABBA driver |
| `validation/training_shader_manifest.json` | provenance entry for the new artifact |
| `validation/phase_xii_channel4_key_production_20261125/` | this receipt, scripts and `raw/` |
