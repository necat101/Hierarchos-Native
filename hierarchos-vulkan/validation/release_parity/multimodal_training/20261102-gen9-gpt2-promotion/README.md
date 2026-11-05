# GPT-2 CPU-kernel promotion on Gen9 (2026-11-02)

The three CPU-matching GPT-2 kernels were briefly gated to the RDNA3 adapter
(`rdna3_gpt2_cpu_math`). This receipt answers whether they can become the
architecture default. Two configurations were built and run against **both**
full gates: the 166-run text forward regression and the 77-case multimodal
training matrix. The `2e-7`, `rtol=0`, `strict_less_than` policy was not
touched, and no fixture was regenerated.

## Configuration A — all three kernels (rejected)

`gpt2_cpu_math(architecture)` promoted the fused-Welford LayerNorm, the
correctly rounded reciprocal attention and the MKL-style tanh GELU for every
GPT-2 package on every adapter; only the RDNA3 k-block GEMM stayed adapter
gated. Patch: `all-three-kernels.patch`.

| Gate | Shipped | All three |
| --- | --- | --- |
| Text regression (166 runs) | 166/166 | **164/166** |
| `gpt2_wide/padded` text, both modes | `1.490e-7` | **`2.086e-7` FAIL** |
| `gpt2_wide/unmasked` text | `1.788e-7` | `1.937e-7` |
| Frozen Rust `gpt2_wide` regression | pass | **FAIL, padded `2.086e-7`** |
| Multimodal matrix | 63/77 cases, 30 rows | 63/77 cases, 29 rows |

The tanh GELU is what breaks `gpt2_wide/padded`. Because the frozen
`gpt2_cpu_math_tests::gpt2_wide_frozen_cpu_logits_resident_and_streamed`
regression is a shipped gate, this configuration was not kept.

## Configuration B — Welford + reciprocal (kept)

The tanh GELU and the k-block GEMM keep their RDNA3 gate; the norm and the
attention reciprocal follow the GPT-2 architecture. Patch: `kept-selection.patch`.

| Gate | Shipped | Kept |
| --- | --- | --- |
| Text regression (166 runs) | 166/166 | **166/166** |
| `gpt2_wide/unmasked` text | `1.788e-7` | `1.788e-7` |
| `gpt2_wide/padded` text | `1.490e-7` | `1.788e-7` |
| `gpt2` (small) text | `2.980e-8` | `2.980e-8` |
| Frozen Rust `gpt2_wide` regression | pass | **pass** (both modes `1.788e-7`) |
| LLaVA image pipeline | 4/4 | **4/4** |
| Multimodal matrix | 63/77 cases, 30 rows | 63/77 cases, 29 rows |

### Multimodal row-level delta

| `gpt2_wide` metric | Shipped | Kept |
| --- | ---: | ---: |
| unmasked `step-2/logits` | `2.086e-7` FAIL | **`1.788e-7`** |
| padded `step-1/logits` | `2.086e-7` FAIL | **`1.639e-7`** |
| padded `step-2/logits` | `3.017e-7` FAIL | `2.161e-7` FAIL |
| padded `final_logits` | `2.682e-7` FAIL | `2.384e-7` FAIL |
| padded `step-2/modality_embedding_grad` | `1.974e-7` | `2.198e-7` **FAIL** |

Two logit rows clear and one adjoint row crosses the line, so the failing-row
count drops by one while the failing-case count is unchanged. The new adjoint
row sits in a case that already fails on three logit rows; it is recorded here
rather than presented as a clean win.

## Reproduce

```
cargo build --bin hierarchos-vulkan-transformer-logits --bin multimodal_training_probe
python hierarchos-vulkan/validation/qualify_gen9_architectures.py run \
  --fixtures target/gen9-phase3/cpu-fixtures \
  --binary hierarchos-vulkan/target/debug/hierarchos-vulkan-transformer-logits.exe \
  --output <new-dir> --modes resident streaming --target gen9
python hierarchos-vulkan/validation/qualify_multimodal_training.py run \
  --fixtures target/multimodal-gen9/matrix-cpu \
  --binary hierarchos-vulkan/target/debug/multimodal_training_probe.exe \
  --output <new-dir> --target gen9
```

Binaries: text `13103366271f4c2df6bde846334edf974be13f4bb5130f3497262149c43ab795`,
multimodal `ee9a21c4ab936932b35a6205bae15733573fbda7666d8b068d147251895aa803`.
`cargo test --lib` reports 33-35 failures when all 1079 GPU tests run in
parallel on one iGPU; every affected module passes with `--test-threads=1`
(`vulkan::tests`, `transformer::tests::parameter_streaming`, `transformer::tests::lora`,
`rwkv_channel_mix::tests`), so the parallel failures are contention, not this change.
