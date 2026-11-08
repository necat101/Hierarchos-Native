# Chinese Model Families & Sparse Attention Audit

**Date:** 2026-11-08

**Source:** `60a14d9b0bfc382722570e6632873b0a9b2234d8`

**Verdict:** **PASS** — no reproducible production defect found; no production code changed.

## Scope

This audit targeted the already-supported Chinese-origin families and their
specialized attention, routing and state paths. It did not rerun the previous
optimization campaign or claim coverage for unqualified registry entries.

- **20 qualified profiles / 37 frozen shared-image cases** passed from the
  canonical 77-case fixture. The subset covers Qwen2/2.5, Qwen3-Next, Qwen3.5,
  Qwen4-Exp, DeepSeek V4, Kimi K2.5/K3, and MiniMax M2/M3.
- **218/218 active DeepSeek tests** passed, with the exact **26-test opt-in
  ignore inventory** preserved.
- **30/30 native V4.1 pipeline tests** passed; logits were bit-exact
  (`max_abs_logit = 0.0`).
- **1,139 passed / 0 failed / 40 ignored** in the complete current-source Rust
  suite.
- **149 targeted filter executions** passed across Qwen, Kimi, MiniMax,
  DeepSeek sparse/indexer/compressor/mixed, and parser families. Filters overlap;
  they are not additive unique-test totals.

## Numerical results

The shared-image canonical gate is `max_abs < 2e-7`, `rtol=0`,
`strict_less_than`. Loss remains on its existing canonical `2e-5` gate; loss and
parameter/gradient values are not relabeled as logits.

| Family/profile | Cases | Sparse / state path | Worst logit | Verdict |
|---|---:|---|---:|---|
| Qwen2 / Qwen2.5 | 6 | Dense/GQA control; sliding-window + tied embedding | `5.960464477539063e-08` | PASS |
| Qwen3-Next | 2 | Linear/full hybrid | `5.960464477539063e-08` | PASS |
| Qwen3.5 full / linear / mixed / MoE | 8 | Full, Gated DeltaNet, hybrid, MoE | `5.960464477539063e-08` | PASS |
| Qwen4-Exp linear / QSA / PLE / mixed | 8 | QSA top-k blocks, GR/PLE caches, MoE | `4.470348358154297e-08` | PASS |
| DeepSeek V4 | 2 | V4 text path | `5.960464477539063e-08` | PASS |
| DeepSeek V4.1 Flash | 218 + 30 native | Compressed top-k, ring cache, compressor/backward | `0.0` logits; `1.7881393432617188e-07` compressor | PASS |
| Kimi K2.5 / K3 KDA / MLA / text | 5 | KDA, MLA, text | `4.470348358154297e-08` | PASS |
| MiniMax M2 / M3 dense / M3 sparse | 6 | Lightning Indexer blocks + local blocks | `5.960464477539063e-08` | PASS |

The worst shared-image logit was **`5.960464477539063e-08`**. The worst shared
loss was `7.152557373046875e-07` against the existing `2e-5` loss limit; the
worst parameter/gradient metric was `1.1920928955078125e-07`.

DeepSeek’s sparse attention test uses `window_topk=3`, `compressed_topk=2`,
explicit `-1` invalid slots and a combined window/compressed gather. Its
compressor/backward tests reached `1.7881393432617188e-07`; index selection was
bit-exact. Native V4.1 had 155,179 bit-exact values out of 155,444 joint
comparisons, with a maximum non-logit drift of `1.4901161193847656e-08`.

## Sparse-attention coverage

- **Qwen4 QSA:** `token_budget=2`, `compress_ratio=2`; complete compressed-block
  top-k, trailing visible tokens, left padding, causal visible-list handling.
- **MiniMax M3:** `block_size=2`, `topk_blocks=1`, `local_blocks=1`; native
  token-index scoring and block-sparse cache path, alongside the dense control.
- **DeepSeek V4.1:** sliding-window ring rules, compressed top-k positions,
  attention-sink normalization, prefill/decode, invalid slots, compressor
  prefill/decode/backward, indexer selection and MoE separately.
- **Hybrid state:** Qwen3.5/Qwen4 Gated DeltaNet recurrence, Qwen4 PLE n-gram and
  dilated short-convolution caches, Kimi KDA/MLA, and Qwen4 GR streams.
- Sparse attention and sparse MoE remain distinct checks. MoE routing was not
  substituted for the sparse-attention gate.

## Repairs

**None.** Every targeted check passed on current source. No tolerance,
assertion, fixture, expected value, ignore list, or production kernel was
changed.

The first shared-image invocation was a harness mistake: it passed the libtest
executable where `image_text_pipeline_probe.exe` was required. All 34 cases
exited 101 with `Unrecognized option: 'model'` before numerical comparison. That
run is preserved under `attempt1-wrong-binary/`; the corrected canonical probe
run passed **37/37**. It is not counted as a model failure.

## Performance

`PERFORMANCE NOT COMPARABLE` for this audit. No production source or shader
changed, so no regression can have been introduced by a repair. There is no
trustworthy same-profile historical binary for Qwen, Kimi or MiniMax. The
established five-workload ViT/Llama/GPT-2/T5 receipt remains separate and is not
restated as a Chinese-model gain.

## Blocked / not fully qualified

- Live Transformers/PyTorch generation, two-step AdamW, and roundtrip oracles are
  **BLOCKED**: torch cannot load (`WinError 126`, `shm.dll`). Frozen authenticated
  references were used instead.
- GLM4/GLM4-MoE, Hunyuan V1, ERNIE 4.5, LongCat Flash, Qwen2-MoE, Qwen3 and
  Qwen3-MoE have native registry/parser or component coverage but no complete
  frozen-logit end-to-end suite in this pass. They are marked
  **REGISTERED / NOT FULLY QUALIFIED** in `inventory.json`.
- Shared cases are batch 2. Focused sparse tests use batch 1/2; no batch-4
  sparse-specific fixture was run.
- This is tiny deterministic fixture and scaled-checkpoint qualification, not a
  claim for arbitrary production checkpoints, dtypes, or full production-size
  throughput.

## Verification

```powershell
python validation/release_parity/multimodal_training/20261108-chinese-sparse-attention-audit/verify.py
```

Raw shared, DeepSeek, native V4.1 and Rust logs are included. `SHA256SUMS` seals
the receipt.
