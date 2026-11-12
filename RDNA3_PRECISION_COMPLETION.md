# Precision engineering results and remaining release gates

These engineering results require the final source and release replay before
the unified release receipt can be marked complete. Historical receipts remain
unchanged; fresh current receipts use `20261112-unified-release/`.

FP32 remains the default. Mixed uses FP16 GEMM operands with FP32 cooperative
accumulation. Full FP16 uses a distinct FP16 cooperative accumulator and widens
its stored result to FP32 on the GPU. Both retain FP32 masters, gradient storage,
normalization, softmax, loss and AdamW state. Ineligible matrix geometries retain
their explicit FP32 route. Full FP16 is not universally FP16 graph arithmetic.

The backend now records separate accumulator-type counts. The scoped Llama128
20-step full-training run executed 420 Full FP16 dispatches, including 140 each
for forward, dW and dX, and zero Mixed dispatches. Ordinary LoRA executed 280
Full FP16 forward/dX dispatches; frozen base weights correctly had no dW update.
All three precision modes decreased loss. Split LoRA training reproduced weights,
moments and scaler state exactly. Shared CLIP128 image/text training completed
two updates with bit-identical export/reload and precision-specific comparisons.
These are small-model results, not production-scale convergence claims.

Standalone GPT2/Llama full training and ordinary LoRA have GPU loss scaling,
FP32 gradient unscaling, global nonfinite detection, atomic optimizer skipping,
growth/backoff and checkpoint persistence. A 32-byte controller observation shares
the existing optimizer submission; there is no tensor readback for the decision.
Tests cover global overflow skipping, restored scaler progress, growth, backoff
and an FP16-underflowing gradient preserved by scaling. Composed image/encoder
transactions still need a shared overflow boundary across their optimizers.
Full-model exact optimizer resume and independent microbatch accumulation are
not qualified by these results.

Native V4.1 Mixed accelerates its native vision projections, preserving its
released ViT/aligner layout and language trainer. The bounded 128-wide independent
CPU fixture passes strict FP32 comparisons below `2e-7`, real Mixed forward/dW/dX,
two AdamW updates and exact reload. A separate larger-magnitude fixture retains
its failing strict FP32 diagnostics and is not promoted by the bounded result.
Native language LoRA remains FP32; native V4.1 Full FP16 remains rejected.

The original Full FP16 primitive definition remains reproducible with the default
`wmma_qualify --wmma-mode full-fp16`: nine of its 37 gates fail. Its historical
oracle assumes one rounding per 16 reduction terms. Independent cadence sweeps
instead closely match two-product accumulation. AMD documents that RDNA3 WMMA
internally uses DOT instructions and supports round-to-nearest-even; internal
operation order remains implementation dependent. See the
[AMD RDNA3 ISA, section 7.9](https://docs.amd.com/api/khub/documents/UkT_UPQL21KfKAMUBFnZTw/content)
and [KHR cooperative matrix specification](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html).

The explicit `--full-oracle rdna3-dot2` comparison preserves the original
two-binary16-ULP matched limit and FP32 max/RMS limits. Every matched-oracle gate
passes. Three `128x128x96` transposed cases still exceed the original FP32
maximum of `0.05` (approximately `0.063–0.073`). Their failing evidence remains
separate. Neither precision substitution nor a wider tolerance is used. General
Full FP16 GUI admission therefore remains incomplete despite working scoped
backend training.

`validation/qualify_precision_training.py` records convergence, actual accumulator
types, FP32 comparisons and exact LoRA resume. `qualify_v41_wmma.py` records the
native V4.1 bounded fixture and separate stress diagnostics. The existing
727-artifact shader manifest is preserved; `wmma_shader_manifest.json` separately
reproduces the four native FP16 conversion/control/cooperative artifacts.
