These frozen primitive corpora are independent of the model qualification
fixtures. Their producers use unmodified Torch CPU operators, one thread, and
record the Torch revision, AVX2 dispatch and producer hash. Producers refuse to
overwrite an existing output. Existing model references were not regenerated.

* `shared_text_sdpa.json`: eight random CPU flash-SDPA forward/adjoint cases,
  including padding, grouped queries and a local attention window.
* `shared_text_norm.json`: random width-64 LayerNorm and RMSNorm, including
  forward, input adjoint and parameter adjoints.
* `shared_training.json`: random CLIP attention, TensorIterator bias reductions,
  activation adjoints, residual-aware RMSNorm, five projection-gradient shapes,
  split-half RoPE and a bounded cosine corpus. Its producer additionally records
  the active `torch_cpu.dll` SHA256.
* `shared_patch.json`: two random RGB CLIP patch projections through active
  oneDNN convolution, including the channel-major patch layout.

The native tests require exact FP32 words for these primitives except the
wide-range NewGELU diagnostic, which permits absolute error below 1e-8 at rare
tanh polynomial boundaries. This diagnostic bound does not change the composed
pipeline's strict logit acceptance requirement: absolute error below 2e-7,
relative tolerance zero, before and after both AdamW updates and after reload.

The pinned contiguous Torch cosine dispatch is MKL VML HA, not the SLEEF
Vec256 implementation. Read-only inspection of Torch revision
`5c4886908584029761b579af026dcfb627c84070` resolved the AVX2 VML dispatch to
kernel RVA `0x636da0`: FP32 quadrant selection, FP64 split-pi reduction and
polynomial evaluation, then FP32 rounding. `fp32_vml_cos.glsl` implements that
bounded arithmetic for absolute arguments at most eight, with the existing
general cosine outside that range. Coefficients come from the active library;
they were not fitted to model inputs or logits. Ordinary text shader variants
leave this specialization disabled.

The recipes in `training_shader_manifest.json` pin glslang 16.6.0 for the new
variants. Separate multiply/add rounding is expressed with GLSL `precise`;
these artifacts require no manual SPIR-V instruction patching. The provenance
checker rebuilds into scratch and compares embedded bytes without replacing
shipping modules.
