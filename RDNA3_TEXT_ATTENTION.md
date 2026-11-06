# RDNA3 text attention port (shared image-text path)

Date: 2026-11-06
Scope: the text half of the shared CLIP/Gemma3 image-text training composition.
This is an additive RDNA3 family. The Gen9 vision work and the committed AVX2
text specializations are untouched.

## Host and target

- CPU: AMD Ryzen Z1 Extreme, `avx512f = true`, so the ATen-compatible width is
  16.
- GPU: AMD Radeon Graphics, PCI `1002:15bf`, RDNA3.
- The AVX2 text block remains width-8-gated and is inert on this host. The new
  AVX-512 text and vision modules are selected by separate RDNA3 gates.

## Paths

- `transformer_attention_short_amd_rdna3.comp` / `.spv` packs 64 consecutive
  `(batch, head, query)` rows into one wave64 workgroup for the plain short
  path. It preserves the generic arithmetic: hardware FMA, per-row key order,
  plain reciprocal and no cross-row reduction.
- `image_text_eager_avx512_amd_rdna3.comp` / `.spv` is the measured Gemma3
  eager-attention path. It admits sequence lengths `8`, `10` and `20`, two
  query heads, two KV heads with an 8-wide head, and a sliding window of 64
  when that window is a no-op for the sequence. The dispatch predicate is
  `rdna3_avx512_eager_supported`; the shader does not read `unfused_qk`.
- The composed dispatch was previously excluded by a `window_size <=
  query_seq_len` test. Gemma3's window is 64 while the measured sequences are
  20 and 10, so the no-op window is now admitted. This is the change that
  closed the full-grid final logit error.

## Result

All four Gemma3 SigLIP cases pass the strict composed gate
`max_abs < 2e-7`, `rtol = 0`:

| case | final logits | gate |
| --- | ---: | --- |
| `siglip_default` | 1.378e-7 | PASS |
| `siglip_batch_padded` | 8.941e-8 | PASS |
| `siglip_full_grid` | 1.304e-7 | PASS |
| `siglip_wider` | 1.527e-7 | PASS |

The full-grid step-1 logits are bit-exact and the full-grid final value moved
from `2.235e-7` to `1.304e-7` after the Gemini eager admission and the other
AVX-512 fixes. The full multimodal receipt is
[`RDNA3_MULTIMODAL_PORT.md`](RDNA3_MULTIMODAL_PORT.md).

## Shared image-text regression

The 74-case `cpu-shared-v2` replay has one numerical red case:
`gpt2_wide/unmasked` at `2.384185791015625e-07` for step-2 and final logits.
The upstream Gen9 multimodal matrix records the same worst value. The padded
GPT-2-wide case passes. Seven required shared-suite profiles are absent from
the retained fixture set, so the inventory all-green flag remains false; that
is not a new numerical regression.

## Exactness tests

```
cargo test --locked --lib rdna3_short_attention_is_word_exact_and_measured \
  -- --nocapture --test-threads=1
cargo test --locked --lib rdna3_avx512_vision_attention_projector_cpu_words \
  -- --nocapture --test-threads=1
```

The first compares the wave64 text module with the generic kernel on five
geometries. The second checks the actual full-grid and batch-padded SigLIP
attention words and the 16-row / 8-row projector RMSNorm words, using raw
`to_bits()` equality.

## Provenance and limits

- `validation/training_shader_manifest.json` records the new
  `transformer_attention_short_*`, `image_text_*avx512*` and
  `vision_attention_avx512_amd_rdna3` artifacts with
  `glslangValidator 15.3.0`, `-V -S comp`, target `vulkan1.0`.
- Gen9 preservation is static: 1,341 upstream shader/fixture files and 672
  pre-existing recipes are unchanged against baseline
  `528e3f722bc609cb2d0bdc5dcb032141c662dc5b`.
- No physical Gen9 execution was possible on this host. No commit was created
  and no end-to-end speedup is claimed.
