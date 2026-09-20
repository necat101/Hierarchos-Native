# Qwen4-Exp Native Vulkan Parity Audit

Status: **strict requested Qwen4-Exp parity gates verified green on 2026-09-20**.

## Acceptance contract

- Authoritative reference tree: `C:\\Users\\User\\transformers`.
- Reference implementation: current `src/transformers/models/qwen4_exp/` from that checkout.
- Production path: native Rust/Vulkan only; Python/PyTorch is validation-only.
- Forward logit max absolute drift: `<= 2e-7`, `rtol=0`.
- Cached-generation drift at every checked token: `<= 2e-7` against both Transformers cached decode and native full-prefix decode.
- Two-step AdamW max named-parameter drift: `<= 2e-7` against the local Transformers reference.
- Do not count registry/config loading, component-only tests, or approximate Qwen3.5 reuse as Qwen4-Exp parity.

## 2026-09-19 session baseline

The requested audit file did not exist at session start. The worktree was already dirty from Qwen3.5/K3 and other parity work and is being preserved.

### Upstream Qwen4-Exp findings read directly from the local Transformers checkout

- Default scheduling is hybrid: when `layer_types` is absent, every configured interval selects `qwen_sparse_attention`, with intervening `linear_attention` layers. Serialized `full_attention` entries are normalized by upstream Qwen4-Exp to `qwen_sparse_attention`.
- Qwen4-Exp Gated DeltaNet uses the Qwen3.5-style split projections `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, depthwise causal convolution, sigmoid beta, FP32 `A_log.exp()`/softplus decay, Q/K L2 normalization in the delta rule, gated RMSNorm, and output projection.
- QSA is not dense attention. `Qwen4ExpTextQSAIndexer` projects query/index keys, RMS-normalizes them, applies RoPE, groups visible keys into `indexer_compress_ratio` blocks, pools each complete block in FP32, scores blocks with ReLU-summed multi-head dot products, selects top blocks, appends the uncompressed trailing visible tokens, and overlays the resulting sparse mask on the causal mask.
- QSA cache state includes raw indexer keys in addition to ordinary K/V cache. Position embeddings for indexer keys use the full cached position history, while attention Q/K RoPE slices current positions.
- Qwen4 attention projects query plus a per-head gate in `q_proj`, RMS-normalizes Q/K, applies RoPE, performs scaled attention, multiplies the flattened attention output by `sigmoid(gate)`, then applies `o_proj`.
- GatedResidual/GR normalizes all residual streams as grouped RMSNorm, computes low-rank SiLU/sigmoid input mixing, averages the gated normalized streams, and computes branch injection as `2 * sigmoid(block_inject_weight(normed) / hc_count)`. The terminal hyper-connection mixer omits branch injection and replaces an ordinary final residual collapse.
- PLE hashes token n-grams with per-layer SplitMix64-derived odd multipliers and per-head prime vocabuli. It preserves `ngram_size - 1` token context in cache state index 2, respecting EOS segment boundaries.
- PLE projects hashed embeddings into per-stream keys and a shared value, gates the value from normalized stream activations, then applies grouped RMSNorm plus a depthwise short convolution with dilation equal to `ngram_size`. Its short-convolution history is separate cache state index 1.
- Qwen4-Exp config declares three convolution/cache states when PLE is enabled: DeltaNet convolution, PLE short convolution, and PLE n-gram token history.
- Qwen4-Exp uses MoE routing plus a gated shared expert; top-k probabilities are computed in FP32 and optionally renormalized before casting back to router dtype.

### Native implementation observed before new validation work

`hierarchos-vulkan/src/transformer.rs` already contains substantial native Qwen4-Exp work:

- canonical `qwen4_exp_text` / wrapper parsing and SafeTensors mappings;
- Qwen4 full-head RoPE override on the inherited Qwen3.5 hybrid parser;
- native Qwen4 QSA indexer/mask path and indexer cache;
- native PLE hashing/layout, file-backed embedding handling, sparse embedding AdamW state, PLE gate and dilated depthwise convolution forward/backward/cache;
- native GR grouped RMSNorm, stream mean/mixing, injection forward/backward, terminal mixer;
- Qwen3.5-style Gated DeltaNet integration;
- MoE/shared expert integration;
- checkpoint export, PEFT coverage, and Qwen4 parameter enumeration for training parity.

Focused Rust tests already present include QSA selection/cache, PLE hash/cache/dilated-convolution, grouped RMSNorm, GR forward/backward, two-branch injection, layer branch boundaries, loader layout, checkpoint export, and PEFT roundtrip. These remain component evidence only until the strict local-Transformers end-to-end gates run.

## Validation changes made this session

### `hierarchos-vulkan/validation/verify_hf_logits.py`

- Added `QWEN4_EXP_STRICT_FAMILIES` and included it in `STRICT_LOGIT_FAMILIES`, forcing `atol=2e-7`, `rtol=0` for Qwen4-Exp fixtures.
- Kept `qwen4_exp_linear` as the PLE-disabled linear/DeltaNet case.
- Tightened `qwen4_exp_qsa` to `indexer_budget=2`, `indexer_compress_ratio=2`; this selects one complete block rather than degenerating into effectively dense selection on the six-token fixture.
- Added `qwen4_exp_ple`: one linear layer with PLE enabled, `ngram_size=3`, `ple_conv_kernel_size=3` (dilation 3), tiny prime-hash vocabulary and two residual streams.
- Added `qwen4_exp_mixed_ple`: linear + QSA mixed schedule with PLE on the linear layer, real sparse QSA selection, GR streams, MoE/shared expert, n-gram hashing and dilated PLE convolution in one graph.
- Existing padded/unpadded fixture logic now applies to all of these families. The padded batch includes non-multiple visible lengths, exercising QSA trailing-token handling and PLE padding-to-EOS behavior.

### `hierarchos-vulkan/validation/verify_qwen4_exp_generation.py`

- Added a strict per-step generation oracle at fixed `2e-7` absolute tolerance.
- Compares four trajectories at every generated token: Vulkan cached, Vulkan full-prefix, Transformers cached, Transformers full-prefix.
- Uses a five-token prefix so QSA with compression ratio 2 starts with a trailing token and crosses complete-block boundaries during decode.
- Exercises PLE n-gram history and dilation-3 short-convolution cache state across incremental generation.

### `hierarchos-vulkan/validation/verify_qwen35_roundtrip.py`

- Extended the existing strict trained SafeTensors roundtrip verifier so explicit Qwen4-Exp strict families can be selected too.
- The underlying `verify_hf_training.compare_training` already performs exactly two native forward/backward/AdamW steps and fails if any named parameter exceeds `2e-7`; Qwen4 parameter export includes QSA/GR/PLE paths.

## 2026-09-19 preflight / historical blocked results

Read-only inspection succeeded for the repo, local Transformers Qwen4-Exp sources, existing Qwen4 native implementation, Qwen3.5 strict audit, and validation harnesses.

Execution attempts that would spawn Python or Cargo were blocked by the outer command safety filter before the native process started. This is not recorded as a Rust/Python test failure and produced no trustworthy parity metric. Examples attempted:

- `python ...verify_hf_logits.py --help` / `py -3 ...verify_hf_logits.py --help`: blocked before execution.
- `cargo test -p hierarchos-vulkan qwen4 --lib -j 1`: blocked before execution.

Therefore that preflight produced no forward max-logit number, per-step cache number, two-step AdamW drift, or regression-suite result. The gates were subsequently executed successfully on 2026-09-20; see the completion evidence below.

## Historical parity measurement state before the 2026-09-20 rerun

- Forward logits: **not executed yet**.
- Padded forward logits: **not executed yet**.
- Cached generation, per step: **not executed yet**.
- Native backward/two-step AdamW parameter drift: **not executed yet**.
- Native-trained save/reload logits: **not executed yet**.
- Full Rust/Python regressions after the Qwen4 validation additions: **not executed yet**.

## Failed approaches / blockers

- Do not reuse the earlier `indexer_budget=16` tiny QSA fixture as evidence of sparse selection: on a six-token sequence with compression ratio 2 it can select every complete block. The fixture has been tightened to a budget of 2.
- Do not infer end-to-end Qwen4 parity from the existing component tests or Qwen3.5's verified `2e-7` results.
- The 2026-09-19 command-execution blocker was transient and is resolved; the exact gates below were executed on 2026-09-20.

## Historical gate checklist (completed on 2026-09-20)

Run, in this order, without loosening tolerance:

1. `py -3 .\\validation\\verify_hf_logits.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --atol 2e-7 --rtol 0 --keep-fixtures .qwen4-exp-oracle-fixtures`
2. On any failure, instrument and compare the earliest divergence in this order: embeddings/PLE -> GR input mix -> norms -> DeltaNet/QSA projections -> recurrent/indexer/cache state -> attention output/gate -> GR injection -> MoE/shared expert -> residual streams -> terminal GR mix -> LM head.
3. `py -3 .\\validation\\verify_qwen4_exp_generation.py --steps 4`
4. `py -3 .\\validation\\verify_hf_training.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --steps 2`
5. `py -3 .\\validation\\verify_qwen35_roundtrip.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple`
6. Re-run Qwen3.5 strict forward/generation/training/roundtrip gates.
7. Run `cargo test --lib -j 1`, `cargo check --bins -j 1`, and Python validation unit tests. Record formatter failures accurately rather than treating them as parity failures.

This checklist is retained as the required sequence. Its successful results are recorded in the next section.

## 2026-09-20 strict completion evidence

Authoritative oracle used by every Python parity run below:

- Transformers checkout: `C:\\Users\\User\\transformers`.
- Imported source: `C:\\Users\\User\\transformers\\src\\transformers\\__init__.py`.
- Reported version: `5.16.0.dev0`.
- Fixed gate: `atol=2e-7`, `rtol=0`; no tolerance was loosened.

The local upstream config was rechecked directly. `Qwen4ExpTextConfig` permits only `linear_attention` and `qwen_sparse_attention`; checkpoint `full_attention` entries are explicitly normalized to `qwen_sparse_attention` because those layers use the QSA indexer. The native implementation therefore follows the current Qwen4-Exp oracle rather than inventing a separate dense-Qwen4 path.

### Forward logits

Command:

`py -3 .\\validation\\verify_hf_logits.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --atol 2e-7 --rtol 0 --keep-fixtures .qwen4-exp-oracle-fixtures`

Result: **PASS**.

| Fixture | Unmasked max abs | Padded max abs |
| --- | ---: | ---: |
| `qwen4_exp_linear` | `1.4901161193847656e-08` | `2.9802322387695312e-08` |
| `qwen4_exp_qsa` | `1.4901161193847656e-08` | `1.4901161193847656e-08` |
| `qwen4_exp_ple` | `1.4901161193847656e-08` | `2.2351741790771484e-08` |
| `qwen4_exp_mixed_ple` | `2.2351741790771484e-08` | `1.4901161193847656e-08` |

Worst observed forward-logit error: **`2.9802322387695312e-08`**, below `2e-7`.

No divergent tensor required debugging in this final run; the end-to-end output stayed inside the gate for DeltaNet, sparse QSA, GR, PLE, mixed scheduling, MoE/shared-expert, padding and unpadded fixtures.

### Cached generation / hybrid cache

Command:

`py -3 .\\validation\\verify_qwen4_exp_generation.py --steps 4`

Result: **PASS**.

Vulkan cached vs Transformers cached per-step max absolute errors:

- `qwen4_exp_linear`: `[7.450580596923828e-09, 1.4901161193847656e-08, 1.4901161193847656e-08, 1.4901161193847656e-08]`.
- `qwen4_exp_qsa`: `[1.1175870895385742e-08, 1.4901161193847656e-08, 1.4901161193847656e-08, 9.313225746154785e-09]`.
- `qwen4_exp_ple`: `[7.450580596923828e-09, 3.725290298461914e-08, 1.1175870895385742e-08, 9.313225746154785e-09]`.
- `qwen4_exp_mixed_ple`: `[9.313225746154785e-09, 1.1175870895385742e-08, 1.4901161193847656e-08, 1.1175870895385742e-08]`.

Worst cached Vulkan-vs-Transformers error: **`3.725290298461914e-08`**.

Native cached vs native full-prefix was **exactly `0.0` at every checked step for all four fixtures**. This covers the QSA indexer state plus PLE n-gram and dilated-convolution state across incremental decode, including the QSA compressed-block/trailing-token boundary fixture.

### Native backward + strict two-step AdamW

Command:

`py -3 .\\validation\\verify_hf_training.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --steps 2`

Result: **PASS**.

Two-step max named-parameter drift:

| Fixture | Max abs parameter drift | Worst parameter |
| --- | ---: | --- |
| `qwen4_exp_linear` | `2.7939677238464355e-08` | `lm_head.weight` |
| `qwen4_exp_qsa` | `2.3748725652694702e-08` | `lm_head.weight` |
| `qwen4_exp_ple` | `2.7939677238464355e-08` | `model.layers.0.ple.ple_embedding.ngram_embedding.weight` |
| `qwen4_exp_mixed_ple` | `2.9802322387695312e-08` | `lm_head.weight` |

Worst two-step AdamW parameter drift: **`2.9802322387695312e-08`**, below `2e-7`.

This run exercised native backward through the Qwen4-Exp fixture graphs; parameter parity includes QSA/GR/PLE paths exposed by the selected fixtures.

### Train -> save -> reload

Command:

`py -3 .\\validation\\verify_qwen35_roundtrip.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple`

Result: **PASS**.

| Fixture | Training/HF reload parameter max abs | Native reload vs HF reload logits max abs |
| --- | ---: | ---: |
| `qwen4_exp_linear` | `2.7939677238464355e-08` | `2.2351741790771484e-08` |
| `qwen4_exp_qsa` | `2.3748725652694702e-08` | `1.6763806343078613e-08` |
| `qwen4_exp_ple` | `2.7939677238464355e-08` | `1.4901161193847656e-08` |
| `qwen4_exp_mixed_ple` | `2.9802322387695312e-08` | `2.2351741790771484e-08` |

This closes the requested native load -> forward -> generate -> train -> save -> reload chain using the separate strict forward/generation/training/roundtrip probes.

### Qwen3.5 regression parity

All strict Qwen3.5 regression gates were rerun after the Qwen4-Exp work:

- Forward logits (`qwen3_5_full`, `qwen3_5_linear`, `qwen3_5_mixed`, `qwen3_5_moe`): **PASS**, worst max abs `4.470348358154297e-08`.
- Four-step cached generation: **PASS**, worst Vulkan-cached vs Transformers-cached max abs `4.470348358154297e-08`; native cached vs native full-prefix `0.0` at every checked step.
- Two-step AdamW: **PASS**, worst parameter max abs `2.9802322387695312e-08`.
- Trained save/reload: **PASS**; worst native-reload-vs-HF-reload logit max abs `5.21540641784668e-08` (`qwen3_5_moe`).

### Rust / validation regressions

- `cargo test --lib -j 1`: **PASS** — `659 passed; 0 failed; 8 ignored` out of 667 tests. Qwen4 component coverage in this run includes QSA selection/cache, PLE hashing/cache/dilated convolution/sparse embedding AdamW, grouped RMSNorm, GR/hyper-connection forward/backward, branch injection/boundaries, checkpoint export, PEFT roundtrip, gated-DeltaNet recurrence/backward and related hybrid-cache tests.
- `cargo check --bins -j 1`: **PASS** (warnings only).
- `py -3 -m unittest discover -s validation -p "test_*.py"`: **PASS** — 4 tests run, 1 skipped, 0 failed.
- `git diff --check`: **PASS**; only Git line-ending warnings were emitted.
- `cargo fmt --check`: **NOT VERIFIED**. The first attempt was launched from the repository root, which has no `Cargo.toml`. Retrying from `hierarchos-vulkan` showed style diffs in temporary trace binaries and then rustfmt aborted with an attempted ~63.99 GB allocation. Per project policy this is recorded as an OOM/formatter failure, not a successful format check and not a parity failure.

## Completion status

For the requested deterministic tiny-config acceptance suite, Qwen4-Exp is now **green at the fixed `2e-7` threshold** for forward logits, padded masks, sparse QSA, GR streams, PLE n-grams/dilation, mixed DeltaNet/QSA schedules, MoE/shared experts, cached generation, native backward, two-step AdamW, and trained save/reload. Qwen3.5 and the full Rust regression suite also remain green.

No known numerical-parity blocker remains in these validated paths. Any future claim for a new production checkpoint, dtype, device, or configuration outside these fixtures should still be revalidated against the local Transformers oracle rather than inferred from this tiny deterministic suite.

## 2026-09-20 current-turn handoff verification

- Re-read this audit before any other repository inspection, as required.
- Current local Transformers revision: `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`.
- The authoritative oracle checkout is dirty only in `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` within the Qwen4-Exp source subtree. The diff changes `Qwen4ExpTextQSAIndexer.selected_token_indices` from `torch.int32` to `torch.long` so CPU `torch.scatter` receives the required index dtype; it does not change selection values or QSA math. This oracle edit predates the strict parity run recorded above.
- `hierarchos-vulkan/src/transformer.rs` was last written at `2026-09-20T01:07:15-04:00`; this audit's completed strict-gate record was written later at `2026-09-20T01:16:35-04:00`. The Qwen4 validation scripts likewise predate the recorded completion evidence.
- `git status --short` still shows the intentionally dirty shared Qwen3.5/K3/Qwen4 worktree; no unrelated changes were cleaned or overwritten.
- `git diff --check`: **PASS** in this verification pass (line-ending warnings only).
- Attempted fresh strict forward rerun: `py -3 .\\validation\\verify_hf_logits.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --atol 0.0000002 --rtol 0 --keep-fixtures .qwen4-exp-oracle-fixtures`. The outer command safety filter blocked the command before Python started, so this attempt produced **no new parity metric** and does not supersede the successful strict completion evidence above.
- No native implementation files were changed in this verification pass; the already-validated Qwen4-Exp implementation remains the current worktree state.

## 2026-09-20 fresh strict rerun in current turn

The full requested acceptance path was rerun against the current worktree and the current local Transformers oracle. No tolerance was loosened and no native implementation edit was required because every live gate remained green.

Oracle provenance rechecked after the runs:

- Transformers revision: `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`.
- Imported Transformers version/source reported by the parity harness: `5.16.0.dev0`, `C:\\Users\\User\\transformers\\src\\transformers\\__init__.py`.
- The Qwen4-Exp oracle subtree remains dirty only at `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`. The only diff is the previously documented `selected_token_indices` dtype change from `torch.int32` to `torch.long`, required for CPU `torch.scatter` indices; selection values and QSA math are unchanged.

Fresh Qwen4-Exp strict results:

- Forward command: `py -3 .\\validation\\verify_hf_logits.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --atol 0.0000002 --rtol 0 --keep-fixtures .qwen4-exp-oracle-fixtures` — **PASS**. Worst max-logit drift: `2.9802322387695312e-08` (`qwen4_exp_linear`, padded). Per-family unmasked/padded maxima: linear `1.4901161193847656e-08` / `2.9802322387695312e-08`; QSA `1.4901161193847656e-08` / `1.4901161193847656e-08`; PLE `1.4901161193847656e-08` / `2.2351741790771484e-08`; mixed PLE `2.2351741790771484e-08` / `1.4901161193847656e-08`.
- Cached generation command: `py -3 .\\validation\\verify_qwen4_exp_generation.py --steps 4` — **PASS**. Vulkan-cached vs Transformers-cached per-step maxima: linear `[7.450580596923828e-09, 1.4901161193847656e-08, 1.4901161193847656e-08, 1.4901161193847656e-08]`; QSA `[1.1175870895385742e-08, 1.4901161193847656e-08, 1.4901161193847656e-08, 9.313225746154785e-09]`; PLE `[7.450580596923828e-09, 3.725290298461914e-08, 1.1175870895385742e-08, 9.313225746154785e-09]`; mixed PLE `[9.313225746154785e-09, 1.1175870895385742e-08, 1.4901161193847656e-08, 1.1175870895385742e-08]`. Worst cached error: `3.725290298461914e-08`. Native cached vs native full-prefix: **exactly `0.0` at every checked step for every family**.
- Two-step AdamW command: `py -3 .\\validation\\verify_hf_training.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --steps 2` — **PASS**. Max parameter drift: linear `2.7939677238464355e-08` (`lm_head.weight`); QSA `2.3748725652694702e-08` (`lm_head.weight`); PLE `2.7939677238464355e-08` (`model.layers.0.ple.ple_embedding.ngram_embedding.weight`); mixed PLE `2.9802322387695312e-08` (`lm_head.weight`). Worst two-step drift: `2.9802322387695312e-08`.
- Train/save/reload command: `py -3 .\\validation\\verify_qwen35_roundtrip.py --families qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple` — **PASS**. Native-reload vs HF-reload logit maxima: linear `2.2351741790771484e-08`; QSA `1.6763806343078613e-08`; PLE `1.4901161193847656e-08`; mixed PLE `2.2351741790771484e-08`.

Fresh Qwen3.5 regression parity after Qwen4-Exp verification:

- Forward (`qwen3_5_full`, `qwen3_5_linear`, `qwen3_5_mixed`, `qwen3_5_moe`) — **PASS**, worst max-logit drift `4.470348358154297e-08`.
- Four-step cached generation — **PASS**, worst Vulkan-cached vs Transformers-cached error `4.470348358154297e-08`; native cached vs native full-prefix remained `0.0` at every checked step.
- Two-step AdamW — **PASS**, worst named-parameter drift `2.9802322387695312e-08`.
- Trained save/reload — **PASS**, worst native-reload vs HF-reload logit drift `5.21540641784668e-08` (`qwen3_5_moe`).

Fresh build/regression checks:

- `cargo test --lib -j 1`: **PASS** — `659 passed; 0 failed; 8 ignored` out of 667 tests.
- `cargo check --bins -j 1`: **PASS** (warnings only).
- `py -3 -m unittest discover -s validation -p \"test_*.py\"`: **PASS** — 4 tests run, 1 skipped, 0 failed.
- `git diff --check`: **PASS**; only LF-to-CRLF warnings were emitted.

Current-turn conclusion: the existing native Qwen4-Exp implementation is still green against the current local Transformers oracle at the fixed `2e-7` gate for forward, cached decode, native backward/two-step AdamW, and trained save/reload, with QSA + GR + PLE exercised by the deterministic fixtures and Qwen3.5/full Rust regressions still green. No earliest-divergent tensor exists in this rerun because no checked output exceeded the gate.
