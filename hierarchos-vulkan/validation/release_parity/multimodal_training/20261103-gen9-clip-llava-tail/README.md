# CLIP/LLaVA regression after the ATen tail panel (Gen9)

This receipt re-runs the existing frozen CLIP/LLaVA pipeline after
`vision_linear_cpu_avx2.comp` began reproducing ATen AVX2 SGEMM's six-row panel
and one-to-three-row short-panel tail. The scope, fixtures and policy are
unchanged: raw 8x8 RGB8, two CLIP layers, the LLaVA projector, the real
two-layer width-16 Llama backbone, two composed AdamW steps, and native
export/reload at `max_abs < 2e-7`, `rtol=0`, `strict_less_than`.

## Result: 4 of 4

| Case | Images | CLIP width | Patch / tokens | Worst visible logit | Result |
| --- | ---: | ---: | --- | ---: | --- |
| `clip_default_layer` | 1 | 16 | 2 / 16 | `step-2` `7.450580596923828e-8` | **pass** |
| `clip_last_layer_padded` | 2 | 16 | 2 / 16 | `step-2` `7.450580596923828e-8` | **pass** |
| `clip_full_tokens` | 1 | 16 | 2 / 17 | `step-2` `6.705522537231445e-8` | **pass** |
| `clip_wider_grid` | 1 | 32 | 4 / 4 | `final` `6.705522537231445e-8` | **pass** |

Each case carries 141 required checks (564 total): visible logits at the
alternate image, both training steps and the final forward, every parameter
tensor after both updates, group-update assertions, loss, pixel gradient and
byte-exact export/reload. The padded case exercises the changed `M=32` path;
the other three cover the full-panel and wider geometries.

Binary `a0cfb206e8dd...`; frozen fixtures `11f899fa8ff9...`. The full digests
and per-check results are in `native-qualification.json`.

Reproduce:

```
python hierarchos-vulkan/validation/qualify_llava_pipeline.py run \
  --fixtures <extracted cpu-fixtures> \
  --output <new-dir> \
  --binary hierarchos-vulkan/target/release/llava_pipeline_probe.exe \
  --target gen9
```

The earlier `../20261102-gen9-image-pipeline/` receipt remains the historical
pipeline record; this directory only re-qualifies it on the current shaders.
