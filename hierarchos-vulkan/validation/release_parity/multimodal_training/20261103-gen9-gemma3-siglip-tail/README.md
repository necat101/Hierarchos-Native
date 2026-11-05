# Native SigLIP -> Gemma3 image pipeline (Gen9): 4 of 4

This receipt closes the one remaining SigLIP/Gemma3 case. The scope, frozen
fixtures and policy are unchanged from
`../20261102-gen9-gemma3-siglip/`: raw 8x8 RGB8, two SigLIP layers at width
16/32, the Gemma3 average-pool soft-embedding projector, the real two-layer
width-16 Gemma3 text backbone, two complete composed AdamW steps, and native
export/reload. Gate: `max_abs < 2e-7`, `rtol=0`, `strict_less_than`, frozen
fixtures hash-bound.

## Result

| Case | Geometry | Image tokens | Worst visible logit | Result |
| --- | --- | ---: | ---: | --- |
| `siglip_default` | width 16, patch 2, 2 layers | 4 (2x2 pool) | `step-2` `7.450580596923828e-8` | **pass** |
| `siglip_batch_padded` | two images, left/right padding | 4 | `step-2` `1.6391277313232422e-7` | **pass** |
| `siglip_full_grid` | 16 tokens, kernel 1 | 16 | `final` `1.9650906324386597e-7` | **pass** |
| `siglip_wider` | width 32, patch 4, 2x2 grid | 4 (kernel 1) | `alternate` `1.7881393432617188e-7` | **pass** |

All 684 required checks pass: 2,048-5,120 visible logits across the
alternate-image, two training-step and final forwards, every parameter tensor
after both updates, group-update assertions, the loss, the pixel gradient, and
byte-exact export/reload.

Binary `0fde558cc5ec02070b8f26dc4c4f31bd09339371c8fa9b87291faea48a879551`;
frozen fixtures `62c8b0d16992da35397d83914113f2edb9caeee08b73876d55df1f2f3129a5f8`.

## What closed `siglip_batch_padded`

The failing seed was the `[32,16] x [16,16]^T` biased q/k/v projection at
`M=32, K=16, N=16`: `ln1` was exact while `q`, `k` and `v` differed from ATen
by `2.384e-7`, reaching `step-2/logits` at `2.757e-7`.

`shaders/vision_linear_cpu_avx2.comp` now reproduces ATen/MKL AVX2 SGEMM's
six-row panel and short final panel. For the one/two-row tail, each product is
materialized in two interleaved accumulators; a three-row tail uses two exact-FMA
lanes. Full panels and four/five-row tails retain ascending-K IEEE FMA. The
host admits the kernel only for the measured `rows 2..=36`, `K in {8,16,32}`,
`N in {16,32,64}` AVX2 shapes. `shaders/vision_patch_projection.comp` separately
selects ATen's bias-last convolution for the single-image 8x8/patch-2 case and
oneDNN's bias-initial order for the `M=32` and `K=48` geometries.

With the M=32 projection on the tail kernel, the step-2 visible logits move from
`2.757e-7` to `1.639e-7`. The other three cases remain green.

The two remaining strict intermediate diagnostics per case are the post-update
`vision_features` rows (`step-2` and `final`, up to `1.31e-6`). They are
diagnostics, not required checks: the projected features and every visible logit
stay below `2e-7`. The acceptance policy and fixtures were not altered.

## Reproduce

```
python hierarchos-vulkan/validation/qualify_gemma3_pipeline.py run \
  --fixtures <extracted cpu-fixtures>/gemma3-cpu \
  --output <new-dir> \
  --binary hierarchos-vulkan/target/release/gemma3_vision_probe.exe \
  --target gen9
```

The earlier `../20261102-gen9-gemma3-siglip/` receipt remains the historical
3-of-4 record; this directory is the current gate.
