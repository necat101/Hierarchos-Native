# Precision engineering results and remaining release gates

These engineering results require the final source and release replay before
the unified release receipt can be marked complete. Historical receipts remain
unchanged; fresh current receipts use `20261112-unified-release/`.

FP32 remains the default. Mixed uses FP16 GEMM operands with FP32 cooperative
accumulation. Full FP16 uses a distinct FP16 cooperative accumulator and widens
its stored result to FP32 on the GPU. Both retain FP32 masters, gradient storage,
normalization, softmax, loss and AdamW state. Ineligible matrix geometries retain
their explicit FP32 route. Full FP16 is not universally FP16 graph arithmetic.

The backend records separate accumulator-type counts. The scoped Llama64
20-step full-training run executed 480 Full FP16 dispatches, including 160 each
for forward, dW and dX, and zero Mixed dispatches. Ordinary LoRA executed 320
Full FP16 forward/dX dispatches; frozen base weights correctly had no dW update.
All three precision modes decreased loss. Split LoRA training reproduced weights,
moments and scaler state exactly. Shared CLIP64 image/text training completed
two updates with bit-identical export/reload and precision-specific comparisons.
These are small-model results, not production-scale convergence claims.

Standalone GPT2/Llama full training and ordinary LoRA have GPU loss scaling,
FP32 gradient unscaling, global nonfinite detection, atomic optimizer skipping,
growth/backoff and checkpoint persistence. A 32-byte controller observation shares
the existing optimizer submission; there is no tensor readback for the decision.
Tests cover global overflow skipping, restored scaler progress, growth, backoff
and an FP16-underflowing gradient preserved by scaling. Shared CLIP/LLaVA
GPT2/Llama image transactions now queue language and image parameters under one
overflow decision. Forced overflow leaves both sets of master weights unchanged;
restoring the scaler preserves its backoff and skipped-step history. This is
scaler-state restoration, not full image optimizer resume. Other encoder/vision
families retain their original FP32 gradient policy.

Deterministic standalone scaled training supports complete equal-weight
microbatch accumulation windows. Two two-example microbatches are compared with
an independent CPU four-example batch. The 20-update Full FP16 run executes 960
cooperative dispatches over 40 microbatches; LoRA executes 640. Both accumulation
and ordinary LoRA resume reproduce masters, moments and scaler state exactly.
Dropout, partial-window checkpoint/resume, full-model optimizer resume and image
gradient accumulation remain outside this capability contract. The additional
image fixture retains its strict FP32 CPU diagnostic failure (about `2.24e-7`
after training); it does not replace or weaken the original 93-image strict gate.

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
separate. Independent CPU dot2 accumulation reproduces all three failing maxima
(`0.0731048584`, `0.0637855530`, `0.0632476807`) against FP32. These are FP16
rounding drift, not a missing widening operation or an accumulation-type alias.
Neither precision substitution nor a wider tolerance is used.

Full FP16 graph admission conservatively limits the actual reduction dimension
to 64, rejecting the entire longer-reduction class rather than selecting passing
geometries from it. Longer reductions take the caller's explicit FP32 kernel;
they never select Mixed. The raw primitive remains unrestricted so the original
37-gate failure stays reproducible. All 28 applicable short-reduction gates pass
with the matched dot2 oracle and unchanged drift limits. This is a scoped graph
policy, not a claim that the original 37/37 primitive qualification passed.

GUI/CLI admission additionally requires a local one-layer Llama64 package
(hidden/intermediate/vocabulary 64, eight heads, no dropout, 33–64 batch×sequence
rows), or its shared CLIP64 composition (batch 2, sequence 24, image 16×16,
patch 4×4), and the qualified AMD `1002:15bf` adapter with driver key `00800184`.
`precision-info` queries cooperative support and the device/driver qualification;
the desktop's **Check backend** binds that result to its CLI path and device
selection. Changing either invalidates admission. Other profiles stay disabled
with a specific reason. The backend repeats these checks independently of GUI
state. Full FP16 retains FP32-sensitive operations, masters and AdamW.

`validation/qualify_precision_training.py` records convergence, actual accumulator
types, FP32 comparisons and exact LoRA resume. `qualify_v41_wmma.py` records the
native V4.1 bounded fixture and separate stress diagnostics. The existing
727-artifact shader manifest is preserved; `wmma_shader_manifest.json` separately
reproduces the four native FP16 conversion/control/cooperative artifacts.
