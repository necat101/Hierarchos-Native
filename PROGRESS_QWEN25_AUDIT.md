# Qwen2.5 Text Native Parity Audit

## Scope and acceptance

- Production path: native Rust/Vulkan only. `C:\Users\User\transformers` is validation/oracle code only.
- Qwen2.5 text checkpoints using `model_type: qwen2` reuse the native Qwen2 graph when the math is identical.
- Strict acceptance: forward logits, cached generation at every checked step, and two-step AdamW parameter drift must each be `<= 2e-7` with no relaxed relative tolerance.
- Qwen2.5-VL and Qwen2.5-Omni vision/audio stacks are outside this text-parity claim.

## Local Transformers reference findings

- Oracle inspected: `src/transformers/models/qwen2/configuration_qwen2.py`, `modeling_qwen2.py`, and the local Qwen2 test tree.
- Current `Qwen2Config` defaults to `rms_norm_eps=1e-6`, untied embeddings, biased Q/K/V, bias-free O/MLP, and optional layer-specific sliding attention.
- `use_sliding_window=true` derives `layer_types` so layers at/after `max_window_layers` are `sliding_attention`; earlier layers are full attention.
- Qwen2 attention uses GQA KV repetition, scale `head_dim**-0.5`, fp32 softmax, ordinary half-rotation RoPE, and cache positions derived from the seen-token count.
- RMSNorm computes variance/rsqrt in fp32 and casts normalized activations back before multiplying by the learned weight.
- SwiGLU order is `down_proj(silu(gate_proj(x)) * up_proj(x))`.

## Baseline before Qwen2.5-specific fixture expansion

- `python validation/verify_hf_logits.py --families qwen2 --atol 2e-7 --rtol 0`
  - PASS on AMD Radeon Graphics.
  - unmasked max-logit drift: `2.9802322387695312e-08`.
  - padded max-logit drift: `5.960464477539063e-08`.
- `python validation/verify_hf_training.py --families qwen2 --steps 2`
  - PASS on AMD Radeon Graphics.
  - two-step AdamW max parameter drift: `5.960464477539063e-08` at `model.norm.weight`.
- Baseline limitation: the old fixture had `num_attention_heads=2` and `num_key_value_heads=2`, so it did not exercise true GQA; it also disabled sliding attention.

## Files changed this session

- `hierarchos-vulkan/validation/verify_hf_logits.py`
  - Added strict Qwen2.5-text fixtures that deliberately retain HF `model_type=qwen2`.
  - Added an untied 4-query/2-KV-head GQA fixture with Qwen2.5-style `rope_theta=1_000_000`.
  - Added a tied GQA fixture with mixed full/sliding layers (`sliding_window=4`, `max_window_layers=2`).
- `hierarchos-vulkan/validation/verify_qwen25_generation.py`
  - Added per-token Transformers cached/full-prefix vs Vulkan cached/full-prefix parity checks at fixed `2e-7`.
- `hierarchos-vulkan/validation/verify_qwen25_roundtrip.py`
  - Added two-step AdamW, native export, HF reload, and native reload-logit parity checks at fixed `2e-7`.
- `PROGRESS_QWEN25_AUDIT.md`
  - Created this audit.
- `README.md` and `hierarchos-vulkan/COMPATIBILITY.md`
  - Documented the measured Qwen2.5 text validation scope and kept VL/Omni outside the text claim.

## Strict Qwen2.5 text results

- Strict forward logits: PASS. Worst observed drift was `5.960464477539063e-08` across GQA, padding, mixed full/sliding attention, and tied/untied fixtures.
- Four-step cached generation: PASS. Worst Vulkan-vs-Transformers cached-step drift was `4.470348358154297e-08`; native cached vs native full-prefix was exactly `0.0` at every checked step.
- Native backward + two-step AdamW + save/reload: PASS. Worst parameter drift was `1.1920928955078125e-07`; worst native-reload vs HF-reload logit drift was `5.960464477539063e-08`.
- Exported configs remain canonical `model_type: qwen2`; no redundant Qwen2.5 production graph was introduced.

## Regression results

- Python validation scripts compile successfully.
- Focused Qwen2 library tests: 7 passed, 0 failed.
- Focused Qwen3 library tests: 7 passed, 0 failed.
- Qwen3.5 and Qwen4-Experimental strict forward, cached-generation, two-step AdamW, and save/reload regressions all passed under the same `2e-7` ceiling.
- A broad Cargo test attempt hit the existing Windows paging-file/resource limit while building unrelated test binaries; the focused one-job library tests then passed.

## Commands run

- `python validation/verify_hf_logits.py --families qwen2_5_gqa qwen2_5_sliding_tied --atol 2e-7 --rtol 0`
- `python validation/verify_qwen25_generation.py --steps 4`
- `python validation/verify_qwen25_roundtrip.py`
- `python -m py_compile validation/verify_hf_logits.py validation/verify_qwen25_generation.py validation/verify_qwen25_roundtrip.py`
- `$env:CARGO_BUILD_JOBS='1'; cargo test --lib qwen2`
- `$env:CARGO_BUILD_JOBS='1'; cargo test --lib qwen3`
- Qwen3.5/Qwen4 strict forward, generation, and roundtrip validation scripts were rerun after the Qwen2.5 fixture additions.

## Failures, root causes, blockers, next task

- The only failed verification command was the broad `cargo test qwen2` invocation. Its failure was an unrelated Windows paging-file/resource exhaustion while Cargo built binary tests, not a failed Qwen assertion. The one-job `--lib` rerun passed.
- No Qwen2.5 text parity blocker remains in the deterministic FP32 qualification scope.
- Next task, if broader certification is desired: test selected production-size Qwen2.5 text checkpoints and additional precision modes separately; do not infer those results from the tiny deterministic fixtures.

## Resume verification (2026-09-20)

- Re-read this audit before touching the Qwen2.5 scope and preserved unrelated dirty/generated work in the checkout.
- Re-checked the local Transformers oracle at `C:\\Users\\User\\transformers`:
  - `Qwen2Config` still owns Qwen2.5 text behavior, including GQA, `layer_types`, `use_sliding_window`, `max_window_layers`, RoPE parameters, attention bias, and tied/untied embeddings.
  - `Qwen2Attention` still uses KV repetition for GQA, `head_dim**-0.5` scaling, fp32 softmax, ordinary RoPE application, and `DynamicCache`; the local Qwen2 tests explicitly include `Qwen/Qwen2.5-3B` sliding-window generation coverage.
- Fresh strict forward rerun: PASS at `atol=2e-7, rtol=0`. Worst Qwen2.5-text logit drift remained `5.960464477539063e-08`.
- Fresh four-step generation rerun: PASS. Worst Vulkan cached-vs-Transformers cached drift remained `4.470348358154297e-08`; native cached-vs-native full-prefix remained exactly `0.0` at every checked step.
- Fresh two-step AdamW + export/reload rerun: PASS. Worst parameter drift remained `1.1920928955078125e-07`; worst native-reload-vs-HF-reload logit drift remained `5.960464477539063e-08`.
- Fresh focused Rust regressions: `cargo test --lib qwen2` = 7 passed / 0 failed; `cargo test --lib qwen3` = 7 passed / 0 failed.
- Fresh strict Qwen3.5/Qwen4-Experimental forward regression sweep: PASS for all eight current strict fixtures under the same `2e-7` absolute-only ceiling; worst observed drift was `4.470348358154297e-08`.
- Fresh Python compile check for the Qwen2.5 validation scripts: PASS.

Fresh commands run:

- `python -m py_compile validation/verify_hf_logits.py validation/verify_qwen25_generation.py validation/verify_qwen25_roundtrip.py`
- `python validation/verify_hf_logits.py --families qwen2_5_gqa qwen2_5_sliding_tied --atol 2e-7 --rtol 0`
- `python validation/verify_qwen25_generation.py --steps 4`
- `python validation/verify_qwen25_roundtrip.py`
- `$env:CARGO_BUILD_JOBS='1'; cargo test --lib qwen2`
- `$env:CARGO_BUILD_JOBS='1'; cargo test --lib qwen3`
- `python validation/verify_hf_logits.py --families qwen3_5_full qwen3_5_linear qwen3_5_mixed qwen3_5_moe qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple qwen4_exp_mixed_ple --atol 2e-7 --rtol 0`

## Completion status

- Qwen2.5 text checkpoints that retain the canonical Qwen2 config/tensor graph are qualified, including Coder/Math/Instruction variants that share that graph.
- Existing native Qwen2 math required no production change; this session filled the missing strict Qwen2.5 validation and documentation coverage.
- Qwen2.5-VL and Qwen2.5-Omni non-text towers remain outside this claim.
