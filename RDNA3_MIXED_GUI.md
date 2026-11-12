# Mixed WMMA + FP32 desktop integration

The desktop **Training precision** selector exposes **FP32 (reference)** and
**Mixed WMMA + FP32 (AMP-style)** for Transformer full training, full-parameter
fine-tuning, LoRA fine-tuning, and supported image/text training backends.
The default remains FP32. Both choices produce an explicit `--wmma-mode` value,
including `off`, so an inherited environment cannot override the GUI choice.

For Transformer and vision commands, `--amp` selects `mixed`; `--no-amp` selects
`off`. These aliases do not change the separate legacy Hierarchos trainer's
existing storage-precision flags.

Mixed mode automatically selects cooperative FP16-operand / FP32-accumulation
matrix kernels for eligible operations. Other operations retain their FP32
Vulkan kernels. Master parameters, gradient buffers, AdamW state, normalization,
softmax, and loss remain FP32. The selected GPU must expose the required
cooperative-matrix features; an unsupported device request is diagnosed.

This is an AMP-style operation policy, **without automatic loss scaling**.
It is not equivalent to the combination of PyTorch autocast and GradScaler.
See the [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html).
FP16 conversion can still lose very small gradient operands or overflow very
large operands. Production-size convergence is not established by the small
integration fixture.

Shared CLIP compositions, LLaVA, Gemma3, and Gemma4 use the common Transformer
projection infrastructure. Their mixed graphs can contain FP32 vision or
language operations; selecting mixed does not imply every operation is WMMA.
Native DeepSeek V4.1 has a separate trainer and currently has no WMMA vision
integration. Its mixed choice is disabled in the GUI and refused at runtime.
Full FP16 accumulation remains unavailable for these training graphs.

Vision training writes the requested policy and actual forward, weight-gradient,
input-gradient, fallback, and cooperative-dispatch counts into
`training_report.json`. A mixed request with zero cooperative dispatches is not
evidence of acceleration.

## Image master-weight freshness

The image tower and projector share projection executors with the language
graph, but submit their AdamW updates separately. Cache versions must therefore
advance after each submitted image update as well as each language update.
Versions now advance monotonically, independently of language optimizer-step
numbers. CLIP/LLaVA/shared, Gemma3, and Gemma4 image optimizers invalidate the
cached FP16 mirrors after submitting their updates. FP32 computation is
unchanged by these cache-version changes.

## Reproduction

Build the Rust GUI, CLI, Vulkan library tests, image probes, and Transformer
trainer with their committed Cargo lockfiles. Run the GUI and CLI unit suites.
`validation/qualify_mixed_gui_integration.py` then exercises an eligible shared
CLIP + Llama fixture with two AdamW updates in each mode, observes all three
WMMA phases, checks image/text updates and export/reload freshness, compares
precision-specific outputs and gradients, and launches the native vision CLI
with an opposing inherited precision policy. Generated weights belong under
`target/`. The script records a scoped result, source hashes, binary hashes,
device identity, commands, and independent CPU-reference diagnostics.

The wider CPU diagnostics preserve the strict `2e-7` comparison outcome in
their report. They do not replace the original frozen-oracle qualification.
The complete maintained FP32 matrix must continue to use
`validation/qualify_rdna3_full_multimodal.py`, its immutable frozen manifest,
`rtol=0`, strict comparisons, and the original parameter/loss contracts.

The larger unified release assignment also requires the packed-cell matrix,
shader provenance, Phase XIX-Z requalification, baseline comparison, and CI.
The desktop precision changes and their scoped receipt alone do not establish
those release gates.
