# SmolLM Native Support Audit

Updated: 2026-09-20

Status: strict qualification complete for the distinct SmolLM3 text family in the current local Transformers checkout, including default RoPE and YaRN fixtures.

## Acceptance

- Oracle: `C:\Users\User\transformers`.
- Forward and cached logits must have maximum absolute error <= `2e-7`.
- Two-step AdamW maximum named-parameter drift must be <= `2e-7`.
- Python/PyTorch/Transformers are validation only; production remains native Rust/Vulkan.
- Do not claim support until forward, cache, backward, AdamW, save/reload, and regressions pass.

## Reference findings

- Oracle is local Transformers `5.16.0.dev0` from `C:\Users\User\transformers\src\transformers`.
- The local checkout's newest and only distinct SmolLM model implementation/model type is `smollm3`; no separate `smollm` or `smollm2` model module/model type is present there.
- SmolLM3 uses SiLU/SwiGLU, RMSNorm epsilon `1e-6`, GQA, tied embeddings, and default RoPE theta `2_000_000`.
- `no_rope_layers` uses `1` for RoPE and `0` for NoPE; when omitted, every fourth layer is NoPE by default.
- When sliding attention is enabled, upstream defaults NoPE layers to sliding attention and RoPE layers to full attention.
- Native parsing recognizes canonical `smollm3`, reconstructs the NoPE schedule, parses shared RoPE/YaRN scaling, builds per-layer rotary specs from `config.json`, and reuses the native Llama-layout RMSNorm/GQA/SwiGLU/cache/backward path where the math is identical.
- Production execution remains Rust/Vulkan. Transformers/PyTorch are used only by the validation scripts.

## Files changed

- `hierarchos-vulkan/validation/verify_hf_logits.py`: added deterministic SmolLM3 and SmolLM3+YaRN fixtures, strict `2e-7` logit qualification, and SmolLM3 to the two-step AdamW headline set.
- `hierarchos-vulkan/validation/verify_smollm3_generation.py`: added strict per-token Transformers cached/full-prefix versus native cached/full-prefix validation.
- `hierarchos-vulkan/validation/verify_smollm3_roundtrip.py`: added base/YaRN two-step training, native export, Transformers reload, native reload, cached generation, and exact config-preservation checks for RoPE/NoPE scheduling, sliding attention, and tying.
- `README.md` and `hierarchos-vulkan/COMPATIBILITY.md`: promoted SmolLM3 only after all requested gates passed.
- `PROGRESS_SMOLLM_AUDIT.md`: records the source findings, commands, measured errors, regressions, and remaining scope.
- No new production Python dependency was introduced. No additional `transformer.rs` change was required in this completion pass because the current native baseline already contained the canonical SmolLM3 loader/runtime/backward wiring; the missing work was strict end-to-end qualification and regression coverage.

## Verification

- Strict forward/padding, default + YaRN:
  `python hierarchos-vulkan\validation\verify_hf_logits.py --families smollm3 smollm3_yarn --atol 2e-7 --rtol 0 --keep-fixtures hierarchos-vulkan\.smollm3-oracle-fixtures`
  - SmolLM3 unmasked/padded max abs: `5.960464477539063e-08` / `5.960464477539063e-08`.
  - SmolLM3 YaRN unmasked/padded max abs: `5.960464477539063e-08` / `2.9802322387695312e-08`.
- Default cached decoding:
  `python hierarchos-vulkan\validation\verify_smollm3_generation.py --family smollm3 --steps 4 --keep-fixtures hierarchos-vulkan\.smollm3-generation-fixtures`
  - Vulkan cached vs Transformers cached: `2.9802322387695312e-08` max.
  - Vulkan full-prefix vs Transformers full-prefix: `4.470348358154297e-08` max.
  - Vulkan cached vs Vulkan full-prefix: `0.0` at every checked token.
- YaRN cached decoding:
  `python hierarchos-vulkan\validation\verify_smollm3_generation.py --family smollm3_yarn --steps 4 --keep-fixtures hierarchos-vulkan\.smollm3-yarn-generation-fixtures`
  - Vulkan cached vs Transformers cached: `5.960464477539063e-08` max.
  - Vulkan full-prefix vs Transformers full-prefix: `5.960464477539063e-08` max.
  - Vulkan cached vs Vulkan full-prefix: `0.0` at every checked token.
- Base train/save/reload:
  `python hierarchos-vulkan\validation\verify_smollm3_roundtrip.py --family smollm3 --steps 3 --keep-fixtures hierarchos-vulkan\.smollm3-roundtrip-fixtures`
  - Two-step AdamW named-parameter drift: `1.1920928955078125e-07` max (`model.layers.1.input_layernorm.weight`).
  - Native reload vs Transformers reload logits: `3.725290298461914e-08` max.
  - Trained cached vs Transformers cached: `5.960464477539063e-08` max; trained native cached vs full-prefix: `0.0`.
- YaRN train/save/reload:
  `python hierarchos-vulkan\validation\verify_smollm3_roundtrip.py --family smollm3_yarn --steps 3 --keep-fixtures hierarchos-vulkan\.smollm3-yarn-roundtrip-fixtures`
  - Two-step AdamW named-parameter drift: `1.1920928955078125e-07` max (`model.layers.3.input_layernorm.weight`).
  - Native reload vs Transformers reload logits: `5.960464477539063e-08` max.
  - Trained cached vs Transformers cached: `5.960464477539063e-08` max; trained native cached vs full-prefix: `0.0`.
- Targeted native contract: `cargo test --manifest-path hierarchos-vulkan\Cargo.toml smollm3 --lib -j 2` -> `1 passed, 0 failed`.
- Full Vulkan library regression: `cargo test --manifest-path hierarchos-vulkan\Cargo.toml --lib -j 2` -> `659 passed, 0 failed, 8 ignored`.
- Headline forward regression: `python hierarchos-vulkan\validation\verify_hf_logits.py --headline-strict` -> pass, including SmolLM3 at `5.960464477539063e-08` max under its absolute-only gate.
- Headline two-step AdamW regression: `python hierarchos-vulkan\validation\verify_hf_training.py --headline-strict` -> pass; SmolLM3 `1.1920928955078125e-07` max over `9,872` parameter values.
- Validation Python unit regression: `python -m unittest discover -s hierarchos-vulkan\validation -p "test_*.py"` -> `4 tests`, `OK`, `1 skipped`.

## Coverage and conclusions

- The deterministic fixtures cover GQA, RoPE layers, NoPE layers, mixed RoPE/NoPE scheduling, padding/masks, tied embeddings, default RoPE, YaRN, cached decoding, native backward via two-step training, AdamW, SafeTensors export/reload, and post-training generation.
- No final SmolLM3 parity result exceeded `2e-7`; there is no unresolved numerical blocker for the tested native text causal-LM path.
- The only coverage gap found during this session was that YaRN initially had forward/cache coverage without a separate train/save/reload run. The roundtrip validator was made family-selectable, and that path then passed the same strict gate.
- Transformers emitted unrelated/deprecation warnings while constructing the broader fixture inventory; they did not cause validation failures.

## Remaining scope

- The current local Transformers checkout does not expose separate `smollm` or `smollm2` model types to implement as independent native architectures. Older checkpoints whose configs identify as Llama remain covered by the existing Llama-compatible native path rather than a separate SmolLM ABI.
- This qualification is deterministic FP32 tiny-model correctness evidence. Production-size checkpoints, other precision modes, and arbitrary hyperparameter combinations remain separate qualification work rather than implied by these fixtures.

## Final recheck

- Re-read the current local Transformers SmolLM3 config/model/tests and the native `from_smollm3_value` / `smollm3_rotary_layers` wiring after the implementation pass. Upstream still treats `no_rope_layers[layer_idx] == 1` as RoPE-enabled and derives sliding attention for configured NoPE layers; the native schedule matches that behavior.
- Re-ran the strict default + YaRN forward fixtures at `atol=2e-7, rtol=0`: maximum observed logit drift remained `5.960464477539063e-08`.
- Re-ran both four-step generation fixtures: all commands passed the `2e-7` per-token gate, including cached-vs-Transformers and cached-vs-native-full-prefix comparisons.
- Re-ran both two-step AdamW + native export/reload fixtures: maximum named-parameter drift remained `1.1920928955078125e-07`; YaRN native-reload logits were `5.960464477539063e-08` max and trained native cache-vs-full-prefix remained `0.0`.
- Re-ran `cargo test --manifest-path hierarchos-vulkan\\Cargo.toml smollm3 --lib -j 2` and the validation Python unit suite; both passed.
- Re-ran the full Vulkan library tests, `verify_hf_logits.py --headline-strict`, `verify_hf_training.py --headline-strict`, and `git diff --check`; the combined command exited `0`. The headline SmolLM3 two-step AdamW result remained `1.1920928955078125e-07` over `9,872` parameter values.
