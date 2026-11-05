# Native SigLIP -> Gemma3 image pipeline (Gen9)

The second native modality tower, after CLIP/LLaVA. It reuses the qualified
vision encoder layers, adds the SigLIP differences (biased patch projection, no
class token, terminal LayerNorm) and the Gemma3 average-pool soft-embedding
projector, then trains the real Gemma3 text backbone through the shared
embedding bridge.

Scope: fixed-resolution 8x8 RGB, two SigLIP layers at width 16/32, two-layer
width-16 Gemma3 text, two complete composed AdamW steps, native export/reload.
Gate: `max_abs < 2e-7`, `rtol=0`, `strict_less_than`, unchanged fixtures after
freeze.

## Result: 3 of 4 cases pass

Latest run (`native-qualification.json`, binary `ba6d07e5...`, fixtures
`62c8b0d1...`) after the two kernel repairs below:

| Case | Geometry | Image tokens | Worst visible logit | Result |
| --- | --- | ---: | ---: | --- |
| `siglip_default` | width 16, patch 2, 2 layers | 4 (2x2 pool) | `alternate` `1.639e-7` | **pass** |
| `siglip_batch_padded` | two images, left/right padding | 4 | `step-2` `2.757e-7` | fail |
| `siglip_full_grid` | 16 tokens, kernel 1 | 16 | `final` `1.527e-7` | **pass** |
| `siglip_wider` | width 32, patch 4, 2x2 grid | 4 (kernel 1) | `final` `1.714e-7` | **pass** |

Each passing case covers 171 required checks: 2,048-5,120 visible logits across
the alternate-image, two training-step and final forwards, every parameter
tensor after both updates, group-update assertions for tower/projector/backbone,
the loss, the pixel gradient, and byte-exact export/reload.

## Why the upstream zero-initialized projector must not be the gate default

`Gemma3MultiModalProjector` initializes `mm_input_projection_weight` to exact
zeros. With that package the image path is identically insensitive at step 1:
`projected_features` is 0 on both sides, the pixel gradient is 0, the image does
not change any logit, and the first AdamW step is an epsilon-limited sign update
whose relative error is dominated by the text adjoint. That produced 40-70
apparent failures per case with no defect in the tower.

The frozen fixtures therefore seed the projection at `N(0, 0.02)`, the scale a
trained checkpoint carries. With that single change three cases move from tens
of failures to zero, and the two property checks (image reaches the pixels,
image changes visible logits) become meaningful.

## Two kernel defects closed, and the one case that remains

The stage diff (`diagnose_gemma3_vision.py`, which now forces the fixture's
pinned eager attention after `from_pretrained`; the package default is SDPA and
had been comparing the native graph against a different arithmetic path) found
two real defects rather than a tower-structure problem.

**1. A stale precompiled attention module.** The committed
`transformer_attention_forward_cpu_reciprocal.spv` did not match the current
`transformer_attention_forward.comp`. Recompiling it with
`-DHIERARCHOS_CPU_ATTENTION_RECIPROCAL` moved the SigLIP attention context and
`out_proj` from `2.384e-7` to **`0.0`**. A standalone replay of the CPU eager
path (unfused QK dot, SLEEF `expf_u10`, correctly rounded reciprocal, unfused
PV in ascending key order) reproduces `torch.softmax` and the oracle context
bit-exactly, confirming the shader now matches the reference.

**2. Gen9 was running an RDNA3-only GELU module.** Both
`transformer_gelu_tanh_ha_{forward,backward}.comp` began with
`#define fp32_fma_exact fma` and a comment claiming the module is never selected
on Gen9. Gen9 has no IEEE FMA - its MAD truncates the product - so the
compensated SLEEF tanh staging drifted by one ulp at `gte`. Both modules now
include `fp32_fma_exact.glsl` and use the software exact FMA. After this,
`siglip_wider` and `siglip_default` are bit-exact through every tower boundary
(`tokens`, `ln1`, `q/k/v`, `attention`, `out_proj`, `ln2`, `fc1`, `gte`,
`layer_0`, `layer_1`, `post_norm`, `pooled_norm`, `projected` all `0.0`), and
`siglip_wider` clears the gate.

The two rows still reported as `DRIFT` by that diagnostic are dump labels, not
arithmetic: the `layer_0/attention` hook on `SiglipAttention` records the
returned post-`out_proj` tensor while the native stage is the pre-`out_proj`
context, and `pooled` is captured before the projector reshapes it. The
downstream stages that consume them (`out_proj`, `pooled_norm`, `projected`) are
exact.

### `siglip_batch_padded`: ATen GEMM blocking at M=32, K=16

`siglip_batch_padded` is the only case whose first projection runs with 32 rows:
two images x 16 tokens. There `layer_0/q`, `k` and `v` differ by `2.384e-7` from
the oracle while `ln1` is exact, so the seed is the `[32,16] x [16,16]^T` biased
projection, not attention.

The native scalar GEMM is an ascending-K IEEE FMA chain with `sum + bias` last,
and it reproduces that chain **exactly**. `torch.nn.functional.linear` /
`addmm` / `x @ w.T + b` all agree with each other and differ from the chain on
19 of 512 outputs by one ulp. Candidate orders were replayed against the frozen
input: pairwise trees, 2/4/8-blocked accumulators, strided and unrolled lane
variants, and a double-precision dot. None matches ATen; the double-precision
result is also one ulp away, so the reference kernel's blocking cannot be
recovered from the shape alone. The 1-ulp projection seed reaches the second
update's visible logits at `2.757e-7`.

This case stays red. The acceptance policy, fixtures and oracle were not
changed to accommodate it.

## Files

* `native-qualification.json`: the full gate, 684 required checks.
* `cpu-fixtures.zip`: frozen packages, RGB inputs, CPU logits/gradients/
  parameters and the imported oracle source.
* `siglip_*-diff.json`, `siglip_*-native.json`: stage dumps and diffs.
* Reproduce:
  `python hierarchos-vulkan/validation/qualify_gemma3_pipeline.py prepare --output <dir>`
  (needs Torch + the pinned Transformers checkout), then
  `... run --fixtures <dir> --binary hierarchos-vulkan/target/debug/gemma3_vision_probe.exe --target gen9`.
  Native replay needs only the standard library.

LLaVA was re-run with the shared vision-kernel changes: **4/4 still pass**.
