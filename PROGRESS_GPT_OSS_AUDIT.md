# GPT-OSS Native Rust/Vulkan Parity Audit

Last updated: 2026-09-19

## Completion gate

GPT-OSS is **unfinished** until all of the following are measured against the current local Hugging Face Transformers implementation under `C:\Users\User\transformers`:

- forward logits max absolute drift <= `2e-7`
- cached generation max absolute drift <= `2e-7` at every checked step
- native backward covers every trainable GPT-OSS operation
- two-step AdamW max parameter drift <= `2e-7`
- canonical HF config/SafeTensors load, train, generate, save, and reload all work
- existing architecture/regression suites remain green

## Current implementation found in-tree

- Architecture registry/config parsing for `model_type = "gpt_oss"` exists in `hierarchos-vulkan/src/transformer.rs`.
- GPT-OSS config handling includes alternating full/sliding attention, RMSNorm epsilon, YaRN defaults, attention bias, sparse MoE, top-k normalization, and router auxiliary-loss coefficient.
- Native GPT-OSS expert support exists, including HF interleaved packed `gate_up_proj`/bias conversion and dedicated clipped SwiGLU forward/backward Vulkan shaders.
- Native attention-sink loading/training/save paths exist.
- Native KV-cache/full-prefix generation regression coverage exists.
- Native sparse-bias training and save/reload regression coverage exists.

## GPT-OSS validation/handoff files

- `hierarchos-vulkan/validation/verify_hf_logits.py`
- `hierarchos-vulkan/src/bin/transformer_generation_parity.rs`
- `hierarchos-vulkan/validation/verify_gpt_oss_generation.py`
- `hierarchos-vulkan/validation/verify_gpt_oss_roundtrip.py`
- `hierarchos-vulkan/COMPATIBILITY.md`

## Current HF reference behavior confirmed from local Transformers

Source: `src/transformers/models/gpt_oss/modeling_gpt_oss.py` and `configuration_gpt_oss.py` in the user's local Transformers checkout.

- RMSNorm computes variance in float32, multiplies by `rsqrt(variance + eps)`, applies the learned weight, then casts back to the input dtype.
- Experts use HF tensors `gate_up_proj[num_experts, hidden, 2*intermediate]`, `gate_up_proj_bias[num_experts, 2*intermediate]`, `down_proj[num_experts, intermediate, hidden]`, and `down_proj_bias[num_experts, hidden]`.
- Packed gate/up channels are interleaved (`gate = [..., ::2]`, `up = [..., 1::2]`).
- Expert activation is `gate = min(gate, 7)`, `up = clamp(up, -7, 7)`, `glu = gate * sigmoid(1.702 * gate)`, output `(up + 1) * glu`.
- Router uses biased linear logits, `torch.topk` on raw logits, then softmax only across selected top-k logits.
- Attention applies RoPE to Q/K, updates cache before attention, scales QK by `head_dim**-0.5`, adds the attention mask, appends one learned sink logit per head, subtracts the row maximum, softmaxes across attention logits plus sink, then drops the sink probability before multiplying by V.
- Default layer pattern is sliding attention on even zero-based layers and full attention on odd zero-based layers; default sliding window is 128.
- Current config defaults include `head_dim=64`, `num_attention_heads=64`, `num_key_value_heads=8`, `rms_norm_eps=1e-5`, `num_local_experts=128`, `num_experts_per_tok=4`, and YaRN rope parameters with factor 32 / original max positions 4096.

## Commands/results this session

### Existing native GPT-OSS unit tests

Command:

`cargo test gpt_oss --lib`

Result: PASS, 6 passed / 0 failed.

Passing tests:

- `gpt_oss_contract_matches_current_hf_attention_moe_and_yarn_defaults`
- `gpt_oss_packed_expert_gate_up_layout_round_trips_hf_interleaving`
- `gpt_oss_moe_loader_reads_current_hf_interleaved_experts_router_and_biases`
- `gpt_oss_generation_kv_cache_matches_full_prefix_logits`
- `bias_all_trains_gpt_oss_sparse_biases_without_base_weights`
- `bias_all_round_trips_gpt_oss_attention_router_and_packed_expert_biases`

Important limitation: these tests do **not** yet establish Transformers numerical parity or two-step AdamW parity.

### Strict forward logits vs current local Transformers

Validation fixture was added to `hierarchos-vulkan/validation/verify_hf_logits.py` using the current local `GptOssConfig` / `GptOssForCausalLM`. GPT-OSS is forced to absolute-only comparison (`atol=2e-7`, `rtol=0`) rather than the broad compatibility suite's looser defaults.

Command:

`python .\validation\verify_hf_logits.py --families gpt_oss`

Reference: Transformers `5.16.0.dev0` from `C:\Users\User\transformers\src\transformers`.

Device: `AMD Radeon Graphics`.

Local Transformers checkout revision for this verification: `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`.

Results:

- unmasked: 192 compared logits, max absolute drift `4.470348358154297e-08`, 0 failures
- mixed-padding batch: 288 visible logits, max absolute drift `5.960464477539063e-08`, 0 failures

Both are below the required `2e-7` maximum.

### Native backward + two-step AdamW vs current local Transformers

Command:

`python .\validation\verify_hf_training.py --families gpt_oss --steps 2`

Result: PASS.

- Transformers losses: step 1 `3.4458606243133545`; step 2 `3.3827271461486816`
- 5,338 trainable parameter values compared after two optimizer steps
- max absolute parameter drift: `1.1920928955078125e-07`
- worst parameter: `model.norm.weight`

This is below the required `2e-7` maximum and exercises native GPT-OSS backward plus AdamW through the full tiny model.

### Cached generation vs full-prefix vs Transformers

Added `hierarchos-vulkan/src/bin/transformer_generation_parity.rs` plus `hierarchos-vulkan/validation/verify_gpt_oss_generation.py`. The native probe uses the public production generation API with raw per-step logits enabled and runs the same model once with KV caching and once with full-prefix recomputation.

Command:

`python .\validation\verify_gpt_oss_generation.py --steps 3`

Result: PASS on `AMD Radeon Graphics` for all three checked decode steps.

- Vulkan cached vs Transformers cached: max drift `2.9802322387695312e-08`; per-step `1.6763806343078613e-08`, `1.4901161193847656e-08`, `2.9802322387695312e-08`
- Vulkan full-prefix vs Transformers full-prefix: max drift `4.470348358154297e-08`; per-step `1.6763806343078613e-08`, `1.862645149230957e-08`, `4.470348358154297e-08`
- Vulkan cached vs Vulkan full-prefix: max drift `0.0` at every step
- Transformers cached vs Transformers full-prefix: max drift `1.4901161193847656e-08`
- all four paths produced the same generated sequence

All generation comparisons are below the required `2e-7` maximum.

### Validation environment

The user's global `huggingface_hub==0.36.0` is older than the local Transformers checkout's declared `huggingface-hub>=1.5.0,<2.0` requirement. The verifier now prefers `HIERARCHOS_VALIDATION_PYDEPS` when explicitly set and otherwise reuses the existing local `.k3-oracle-pydeps` validation cache (`huggingface_hub==1.32.0`) when present. This path injection is validation-only and does not add Python/PyTorch to production execution.

## Measured parity

- Forward logits vs Transformers: **PASS** — max drift `5.960464477539063e-08` across checked unmasked/padded cases
- Cached generation vs Transformers: **PASS** — max checked cross-backend drift `4.470348358154297e-08`; cached/full-prefix native drift `0.0`
- Two-step AdamW max parameter drift vs Transformers: **PASS** — `1.1920928955078125e-07`
- Canonical native export -> Transformers reload -> native reload: **PASS** — HF parameter drift `0.0`, HF reload logit drift `0.0`, native reload vs source `0.0`, native reload vs HF `4.470348358154297e-08`
- Full native library regressions: **PASS** — `657 passed; 0 failed; 8 ignored`
- Strict headline forward suite with GPT-OSS included: **PASS** — GPT-OSS unmasked `4.470348358154297e-08`, padded `5.960464477539063e-08`

## Known failures / root causes

- No GPT-OSS numerical parity failure has been observed yet.
- Initial oracle startup failed before model execution because the global `huggingface_hub==0.36.0` lacked current Transformers APIs; resolved by validation-only reuse of the existing local oracle dependency cache.

## Fresh completion-gate rerun — 2026-09-19

All completion gates below were rerun successfully in the current continuation against local Transformers revision `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c` on `AMD Radeon Graphics`:

- `python .\\validation\\verify_hf_logits.py --families gpt_oss`: PASS — unmasked max logit drift `4.470348358154297e-08`; mixed-padding max logit drift `5.960464477539063e-08`; `rtol=0`, `atol=2e-7`.
- `python .\\validation\\verify_hf_training.py --families gpt_oss --steps 2`: PASS — losses `3.4458606243133545`, `3.3827271461486816`; 5,338 parameter values compared; max parameter drift `1.1920928955078125e-07`; worst parameter `model.norm.weight`.
- `python .\\validation\\verify_gpt_oss_generation.py --steps 3`: PASS — cached Vulkan vs cached Transformers max `2.9802322387695312e-08`; full-prefix Vulkan vs full-prefix Transformers max `4.470348358154297e-08`; cached Vulkan vs full-prefix Vulkan exactly `0.0` at every checked step.
- `python .\\validation\\verify_gpt_oss_roundtrip.py`: PASS — HF exported/reloaded logits `0.0`; HF parameter drift `0.0`; native source vs HF `4.470348358154297e-08`; native reload vs HF `4.470348358154297e-08`; native reload vs native source `0.0`.
- `cargo test gpt_oss --lib -j 1`: PASS — 6 passed / 0 failed.
- `cargo test --lib -j 1`: PASS — 657 passed / 0 failed / 8 ignored.
- `rustfmt --edition 2021 --check src\\bin\\transformer_generation_parity.rs`: PASS after applying rustfmt's reported formatting diff to the new generation probe.
- `python -m py_compile .\\validation\\verify_hf_logits.py .\\validation\\verify_gpt_oss_generation.py .\\validation\\verify_gpt_oss_roundtrip.py`: PASS.
- `git diff --check`: PASS for tracked changes (only the existing LF/CRLF conversion warnings were emitted).

The earlier command-filter issue that prevented a training rerun is no longer a blocker: the two-step AdamW verifier completed successfully in this continuation.
Crate-wide `cargo fmt --all -- --check` is not a usable verification signal in this checkout on this machine: after reporting formatting for the new generation probe it aborted with an attempted `49,509,362,648`-byte allocation while traversing the very large crate. The affected new probe was formatted manually from rustfmt's exact diff and then passed a direct file-level rustfmt check; this OOM is not a GPT-OSS numerical/runtime failure.

## Next task

GPT-OSS now meets the requested deterministic tiny-fixture completion gates. Keep the strict fixtures in the regression set and investigate only if a future Transformers change or larger real checkpoint exposes a new mismatch; do not relax the `2e-7` acceptance thresholds.
