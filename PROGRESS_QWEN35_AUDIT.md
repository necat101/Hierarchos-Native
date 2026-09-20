# Qwen3.5 Native Vulkan Parity Audit

Status: **verified for the native Qwen3.5 text causal-LM paths covered here** (dense full-attention, pure Gated DeltaNet, default hybrid scheduling, and Qwen3.5-MoE). The required load -> forward -> generate -> backward -> two-step AdamW -> save/reload chain is green at the `2e-7` absolute ceiling on the local Transformers oracle and AMD Radeon Graphics.

## Acceptance contract

- Authoritative reference tree: `C:\Users\User\transformers`
- Production path: native Rust/Vulkan only; Python/PyTorch is validation-only.
- Forward logit max absolute drift: `<= 2e-7`
- Cached-generation max absolute logit drift at every checked step: `<= 2e-7`
- Two-step AdamW max parameter drift: `<= 2e-7`
- Required end-to-end coverage: load -> forward -> generate -> backward -> two-step AdamW -> save/reload, plus padded/unpadded fixtures and regression suites.

## 2026-09-19 handoff baseline

The audit file did not exist when this Qwen3.5 continuation began. The worktree already contained substantial Qwen3.5 implementation and diagnostic changes; unrelated dirty state is being preserved.

### Verified local Transformers findings

- `Qwen3_5TextConfig.__post_init__` defaults `partial_rotary_factor` to `0.25`.
- Default hybrid scheduling is three `linear_attention` layers followed by one `full_attention` layer (`full_attention_interval=4`) when `layer_types` is omitted.
- `Qwen3_5Attention` splits `q_proj` into query and gate per head, RMS-normalizes Q/K over `head_dim`, applies partial RoPE to Q/K, performs scaled causal attention, multiplies the flattened attention output by `sigmoid(gate)`, then applies `o_proj`.
- Qwen3.5 Gated DeltaNet uses the split projection ABI `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a`.

### Existing native implementation observed

- Canonical `qwen3_5_text` / `qwen3_5_moe_text` parsing and multimodal wrapper aliases exist.
- Hybrid layer scheduling, split Qwen3.5 Gated DeltaNet projections, recurrent Vulkan kernels, attention gates, dense/MoE loader paths, backward hooks, optimizer stepping, and HF tensor export plumbing already exist in `hierarchos-vulkan/src/transformer.rs`.
- Temporary strict-localization tools currently present in the dirty worktree: `src/bin/qwen35_trace_tmp.rs` and `validation/qwen35_trace_oracle_tmp.py`.

### First full-attention trace before the RoPE capability fix

Command actually run for native trace:

`hierarchos-vulkan\target\debug\qwen35_trace_tmp.exe hierarchos-vulkan\.qwen35-oracle-fixtures\qwen3_5_full\unmasked hierarchos-vulkan\.qwen35-oracle-fixtures\qwen3_5_full\unmasked\vulkan_logits_fixture.json hierarchos-vulkan\.qwen35-oracle-fixtures\qwen3_5_full\unmasked\vulkan_trace.json`

Command actually run for HF intermediate comparison:

`py -3 hierarchos-vulkan\validation\qwen35_trace_oracle_tmp.py hierarchos-vulkan\.qwen35-oracle-fixtures\qwen3_5_full\unmasked hierarchos-vulkan\.qwen35-oracle-fixtures\qwen3_5_full\unmasked\vulkan_logits_fixture.json hierarchos-vulkan\.qwen35-oracle-fixtures\qwen3_5_full\unmasked\vulkan_trace.json hierarchos-vulkan\.qwen35-oracle-fixtures\qwen3_5_full\unmasked\hf_trace.json`

Observed max absolute intermediate drift before the fix:

- input RMSNorm: `2.384185791015625e-07`
- Q projection: `2.2351741790771484e-08`
- K projection: `2.9802322387695312e-08`
- V projection: `2.9802322387695312e-08`
- Q head RMSNorm: `2.384185791015625e-07`
- K head RMSNorm: `3.5762786865234375e-07`
- rotated Q: `2.06738543510437` (native trace contained zeroes where HF had rotated values)
- rotated K: `2.1119234561920166` (same zero-RoPE symptom)
- attention gate pre-activation: `2.9802322387695312e-08`
- attention: `0.12723658978939056`
- gated attention: `0.06362637877464294`
- attention projection: `0.006986288353800774`
- first residual: `0.006986288819462061`
- second RMSNorm: `0.5196625590324402`
- MLP output: `0.0002341870276723057`
- layer output: `0.006934128701686859`
- final RMSNorm: `0.513921856880188`

Root cause found for the first large divergence: the Qwen3.5 parser correctly produced partial-RoPE configuration, but `VulkanTransformerArchitecture::uses_rotary_attention()` omitted `Qwen3Next`, `Qwen35`, `Qwen35Moe`, and `Qwen4Exp`. Consequently `transformer_rotary_layers_for_package()` disabled the rotary path and the native rotated-Q/K trace buffers remained zero. The architecture predicate is being corrected to include those hybrid families.

Note: even before the large RoPE break, RMSNorm drift (`2.384e-7` to `3.576e-7`) already exceeds the requested `2e-7` absolute-only threshold and remains a required follow-up after the RoPE fix.

## 2026-09-19 completion

### Implemented/fixed behavior

- Rotary capability dispatch now includes `Qwen3Next`, `Qwen35`, `Qwen35Moe`, and `Qwen4Exp`, so Qwen3.5 partial RoPE is actually executed instead of leaving rotated Q/K buffers zeroed.
- Qwen hybrid RMSNorm paths can request the same unfused FP32 product/reduction order used by the current local Transformers implementation. The Vulkan forward path also performs one explicit reciprocal-square-root refinement for this strict-parity mode; the matching input-gradient shader preserves the same non-contracted product order.
- Qwen3.5 Gated DeltaNet applies the 2D padding mask to the post-norm hidden states before its projections/convolution, matching `apply_mask_to_padding_states`; backward masks the corresponding input gradient as well.
- The packed recurrent DeltaNet backward shader no longer reconstructs prior state by dividing by `exp(decay)`. Qwen3.5 can drive that value near `1e-7`, making inversion numerically unstable. The correctness-first kernel now replays the stable forward prefix for each reverse token and differentiates from the reconstructed pre-update state.
- Canonical Qwen3.5 dense/MoE config parsing, SafeTensors loader/export mappings, full-attention query gating/QK norm/RoPE, split `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a` DeltaNet projections, recurrent cache, hybrid scheduling, dense MLP/MoE, backward, and AdamW all execute through the native Rust/Vulkan path.
- `validation/verify_qwen35_generation.py` now performs an explicit local-Transformers cache loop. For the synthetic all-linear fixture only, it supplies the effective unpadded masks directly because current upstream `Qwen3_5TextModel` constructs a full-attention mask even when no full-attention layer exists, while `DynamicCache.get_seq_length()` intentionally rejects an all-linear cache.
- Added `validation/verify_qwen35_roundtrip.py` to train for two native AdamW steps, reload the native export with local Transformers and native Vulkan, and compare the exact same checkpoint at the strict threshold.
- Updated two KDA unit-test call sites for the new optional padding-mask argument so the full Rust test target compiles again.

### Strict forward parity

Command actually run:

`py -3 .\\validation\\verify_hf_logits.py --families qwen3_5_full qwen3_5_linear qwen3_5_mixed qwen3_5_moe --atol 2e-7 --rtol 0`

Reference: local Transformers `5.16.0.dev0` from `C:\\Users\\User\\transformers\\src\\transformers\\__init__.py`.

- `qwen3_5_full`: unmasked `4.470348358154297e-08`; padded `4.470348358154297e-08`.
- `qwen3_5_linear`: unmasked `4.470348358154297e-08`; padded `4.470348358154297e-08`.
- `qwen3_5_mixed`: unmasked `4.470348358154297e-08`; padded `4.470348358154297e-08`.
- `qwen3_5_moe`: unmasked `2.9802322387695312e-08`; padded `2.9802322387695312e-08`.
- All eight cases had `failing_values = 0` under `atol=2e-7, rtol=0`.

### Cached generation parity

Command actually run:

`py -3 .\\validation\\verify_qwen35_generation.py --steps 3`

- `qwen3_5_full`: Vulkan cached vs Transformers cached `2.9802322387695312e-08`; Vulkan cached vs Vulkan full-prefix `0.0`.
- `qwen3_5_linear`: Vulkan cached vs Transformers cached `3.725290298461914e-08`; Vulkan cached vs Vulkan full-prefix `0.0`.
- `qwen3_5_mixed`: Vulkan cached vs Transformers cached `4.470348358154297e-08`; Vulkan cached vs Vulkan full-prefix `0.0`.
- `qwen3_5_moe`: Vulkan cached vs Transformers cached `2.9802322387695312e-08`; Vulkan cached vs Vulkan full-prefix `0.0`.
- Generated token sequences matched between cached/full-prefix Vulkan and cached/full-prefix Transformers for all four fixtures.

### Native backward and two-step AdamW parity

Command actually run:

`py -3 .\\validation\\verify_hf_training.py --families qwen3_5_full qwen3_5_linear qwen3_5_mixed qwen3_5_moe --steps 2`

This executes native Vulkan forward, loss, backward, and AdamW, then compares every named trainable parameter with local PyTorch/Transformers after two optimizer steps. No separate raw-gradient dump was used, so the verified backward evidence is the end-to-end loss/parameter trajectory rather than an invented gradient metric.

- `qwen3_5_full`: max parameter drift `2.7939677238464355e-08` (`lm_head.weight`).
- `qwen3_5_linear`: max parameter drift `2.8870999813079834e-08` (`lm_head.weight`).
- `qwen3_5_mixed`: max parameter drift `2.9802322387695312e-08` (`lm_head.weight`).
- `qwen3_5_moe`: max parameter drift `2.8870999813079834e-08` (`model.layers.1.self_attn.k_proj.weight`).
- All four are below the required `2e-7` ceiling.

### Canonical trained save/reload parity

Command actually run:

`py -3 .\\validation\\verify_qwen35_roundtrip.py`

The verifier uses each two-step native-trained `config.json` + `model.safetensors`, reloads that exact checkpoint through local Transformers and the Vulkan loader, and compares logits on the same weights.

- `qwen3_5_full`: native reload vs HF reload logits `4.470348358154297e-08`.
- `qwen3_5_linear`: `4.470348358154297e-08`.
- `qwen3_5_mixed`: `2.9802322387695312e-08`.
- `qwen3_5_moe`: `5.21540641784668e-08`.
- HF reload of the native-trained export preserved the two-step parameter result; worst observed parameter difference versus the independently HF-trained trajectory remained `2.9802322387695312e-08`.

An earlier version of this verifier compared logits from two *different* trained weight sets (HF AdamW output versus native AdamW output) and the MoE case measured `2.08616257e-07`. That is not a serialization/reload metric: the separately validated optimizers are allowed a nonzero parameter drift. The final roundtrip gate therefore compares Vulkan and Transformers on the exact same native-exported checkpoint while `verify_hf_training.py` independently enforces the `2e-7` optimizer parameter ceiling. No tolerance was loosened.

### Regression and build verification

- `cargo test --lib -j 1`: **pass**, `657 passed; 0 failed; 8 ignored`.
- `cargo check --bins -j 1`: **pass**.
- `py -3 -m unittest discover -s validation -p "test_*.py"`: **pass**, `4 tests`, `1 skipped`.
- `cargo test --lib qwen35 -j 1`: test target compiled after repairing the two stale KDA call sites; the literal `qwen35` filter matched zero test names (`665 filtered out`), so this was not counted as test coverage and the complete library suite above was run instead.
- `cargo fmt --check`: **did not pass/complete**. It first reported formatting in the pre-existing temporary `src/bin/qwen35_trace_tmp.rs`, then `rustfmt` aborted with an attempted `49,738,809,780` byte allocation. This is recorded as an outstanding tooling/temporary-file formatting issue, not reported as a green check.

### Files materially involved in the Qwen3.5 parity work

- `hierarchos-vulkan/src/transformer.rs`
- `hierarchos-vulkan/shaders/transformer_rms_norm_forward.comp` + compiled `.spv`
- `hierarchos-vulkan/shaders/transformer_rms_norm_input_grad.comp` + compiled `.spv`
- `hierarchos-vulkan/shaders/transformer_gated_delta_recurrent_packed_backward.comp` + compiled `.spv`
- `hierarchos-vulkan/validation/verify_hf_logits.py`
- `hierarchos-vulkan/validation/verify_qwen35_generation.py`
- `hierarchos-vulkan/validation/verify_qwen35_roundtrip.py`
- `hierarchos-vulkan/src/bin/transformer_generation_parity.rs`
- `hierarchos-vulkan/COMPATIBILITY.md`

### Remaining blockers / next task

No numerical blocker remains for the requested Qwen3.5 text causal-LM acceptance contract: strict forward, padded/unpadded, cached generation, native backward/two-step AdamW, canonical trained save/reload, full library regressions, all-bin compilation, and Python validation tests are green. The only observed non-green check is the repository-wide formatter invocation described above, caused by a temporary trace file plus a rustfmt allocation failure; it does not change the verified runtime/parity results.
