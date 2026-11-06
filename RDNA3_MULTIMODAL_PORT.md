# RDNA3 AVX-512 multimodal port receipt

Date: 2026-11-06
Target: `AMD Radeon Graphics`, PCI `1002:15bf` (RDNA3 / gfx11)
Host CPU: AMD Ryzen Z1 Extreme, AVX-512, `aten_vector_width() == 16`
Driver: `26.10.07.06 (LLPC)`, Windows `32.0.31007.6002`

## Result

The shared image-text training path is green for all four Gemma3 SigLIP cases
under the unchanged acceptance policy `max_abs < 2e-7`, `rtol = 0`:

| case | step-1 logits | step-2 logits | final logits | gate |
| --- | ---: | ---: | ---: | --- |
| `siglip_default` | 8.196e-8 | 6.706e-8 | 1.378e-7 | PASS |
| `siglip_batch_padded` | 5.600e-8 | 9.686e-8 | 8.941e-8 | PASS |
| `siglip_full_grid` | 0.0 | 1.267e-7 | 1.304e-7 | PASS |
| `siglip_wider` | 1.416e-7 | 1.714e-7 | 1.527e-7 | PASS |

The qualification command is:

```
python validation/qualify_gemma3_pipeline.py run \
  --fixtures target/rdna3-port/cpu-gemma3 \
  --output target/rdna3-port/cpu-gemma3/siglip-eager2 \
  --binary target/debug/gemma3_vision_probe.exe \
  --target rdna3 --device-id 0x15bf
```

The non-required `vision_features` diagnostics after the two updates remain
between 0.6e-6 and 1.7e-6. They are retained as diagnostics in the receipt, but
they are not the logit gate requested for this port.

## What changed

The work is additive and selected only on an RDNA3 adapter with the AVX-512
host width. The Gen9 shaders, routes, fixtures and tests were not edited.

- SigLIP attention has a dedicated AVX-512/RDNA3 module for sequence lengths
  `4`, `5`, `16` and `17`. It uses the measured query-key order: materialized
  products below 16 tokens and serial hardware FMA from 16 tokens upward.
- The Gemma3 projector RMSNorm has a dedicated module for the two measured
  pooled-row topologies: 16 rows use four interleaved reduction streams, 8 rows
  use sixteen. The weight offset is `(1 + weight)` and the reciprocal square
  root follows the measured CPU path.
- The RDNA3 RoPE VML trig path no longer falls through to SLEEF above eight
  radians. The existing VML polynomial matches the CPU words over the tested
  0-32 radian range, including position 17 / frequency 0.
- Gemma3's eager text-attention admission now covers two KV heads with an
  8-wide head, sequence lengths `8`, `10` and `20`, and the model's 64-token
  sliding window when that window is a no-op for the sequence. This was the
  final full-grid fix: the final logit error moved from `2.235e-7` to
  `1.304e-7` without changing the gate.

The new modules are gated by `vendor::rdna3_avx512_variants`, separate from
`rdna3_vision_variants` and the Gen9 routes. Other adapters, including Gen9,
keep the committed kernels.

## Shared-backbone regression

The 74-case `cpu-shared-v2` replay completed. 73 cases pass; the only numerical
red case is `gpt2_wide/unmasked`, whose step-2 and final logits are
`2.384185791015625e-07`. That is the same worst value recorded for Gen9 in
`validation/release_parity/multimodal_training/20261102-gen9-gpt2-promotion/all-three-matrix.log`
(the Gen9 fixture reports 3840 values, this fixture 5120). The padded GPT-2-wide
case passes on RDNA3. No tolerance or CPU expectation was changed.

The inventory-level all-green flag is still false because seven required
shared-suite profiles have no generated cases in this fixture set (`bart`,
`deepseek_v41_scalable`, the three `kimi_k3` profiles, `switch_transformers`
and `t5`), in addition to the known GPT-2-wide residual. This is a coverage
limitation of the retained fixtures, not a newly introduced numerical failure.

## Isolation

`validation/verify_rdna3_multimodal_isolation.py` was run against baseline
`528e3f722bc609cb2d0bdc5dcb032141c662dc5b`:

- 1,341 upstream Gen9 shader and fixture files checked;
- 672 pre-existing shader recipes checked;
- zero changed files or recipes.

This is a static preservation check. No physical Intel Gen9 adapter is present
on this host, so Gen9 execution was not repeated.

## Tests and provenance

The new raw-word test is:

```
cargo test --locked --lib rdna3_avx512_vision_attention_projector_cpu_words \
  -- --nocapture --test-threads=1
```

It passed on the physical adapter. The fixture
`validation/fixtures/rdna3_vision_avx512.json` contains the actual Windows
Torch 2.6.0+cpu AVX-512 stage words for the full-grid and batch-padded SigLIP
attention, plus the 16-row and 8-row projector RMSNorm. The test checks raw
`to_bits()` equality, not a tolerance.

`validation/training_shader_manifest.json` now records 27 new artifacts for
the AVX-512/RDNA3 family. All 27 reproduce from their recorded sources with
`glslangValidator 15.3.0` and target `vulkan1.0`; the residual RMS artifact
needs both `-DRDNA3_RMS_BACKWARD` and `-DRDNA3_RMS_RESIDUAL`. The new
attention and projector artifacts are:

```
vision_attention_avx512_amd_rdna3.spv
vision_projector_rms_avx512_amd_rdna3.spv
```

The existing text-attention test remains in
`src/transformer/rdna3_text_attention_tests.rs`; the new vision test is in the
same module.

## Limits

- No commit was created. The worktree contains the additive source, SPIR-V,
  fixtures and receipts.
- No Gen9 execution was performed; preservation is established statically.
- The shared suite is not advertised as all-green: the GPT-2-wide residual is
  preserved Gen9 behavior, and seven inventory profiles are absent from the
  retained fixture set.
- No end-to-end speedup claim is made for the new path.
