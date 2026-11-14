# Current numerical qualification scope

The original maintained FP32 contract requires 93 image cases and 166 text
comparisons, strict `max_abs < 2e-7`, `rtol=0`, and the original loss, parameter
update and export/reload checks. Those expectations and frozen inputs remain
unchanged. Internal feature and gradient diagnostics are recorded separately;
passing visible logits does not qualify every internal tensor as CPU-identical.

The historical 1,116 intermediate observations contain 194 strict diagnostic
failures: 186 vision-feature comparisons and eight projected-gradient
comparisons. These remain failures of that additional strict diagnostic, rather
than being relabeled PASS or assigned a larger tolerance. Representative
Gemma4 capture has identical patch input, patch projection, initial tokens,
normalization and Q/K/V projections; the live pinned CPU comparison first
differs at attention. Its own replay differs from the frozen CPU output, so it
is an investigation with disclosed reference uncertainty, not new qualification.
The audit records the full raw diagnostic inventory alongside the maintained
gate. No universal strict-equivalence claim is made for all internal tensors.

The exploratory geometry labels in the earlier handoff need care. The retained
`2.2351741790771484e-7` post-training maximum belongs to the 64-hidden/64-vocabulary
fixture. The separate 128-hidden/32-vocabulary fixture has retained maxima
`2.9802322387695312e-7` and `3.5762786865234375e-7`. Neither belongs to the
maintained 93-case gate. Their actual configuration and measurements, rather
than the handoff's “128-wide, 2.24e-7” label, define the additional observations.

Mixed uses FP16 operands and FP32 accumulation for eligible projections,
retaining FP32 master weights, gradient storage and AdamW moments. Gemma4 vision
keeps V, O, patch and MLP projections in FP32 under its measured numerical
policy; Q/K remain cooperative. Policy fallbacks are counted separately.
Standalone GPT2/Llama full training and ordinary LoRA, and shared CLIP/LLaVA
GPT2/Llama joint transactions, implement GPU loss scaling and global optimizer
skipping. Other architecture-specific trainers retain their tested unscaled
policy. FP32 gradient **storage** does not prevent quantization when gradients
become FP16 GEMM operands; automatic scaling is not claimed for those paths.
Their supported numerical scope is the executed bounded matrix, not arbitrary
gradient ranges or complete PyTorch autocast/GradScaler equivalence.

Native V4.1 retains its released ViT, two-dimensional rotary attention and
spatial aligner. The bounded independent 128-wide fixture passes all 65 strict
rows (maximum `1.564621925354004e-7`). In the stress fixture the patch projection
is identical, and first-norm output differs by `2.384185791015625e-7`.
Operand-matched CPU and GPU norms are each within two FP32 ULP of an independent
float64 calculation. Subsequent drift is retained and the stress fixture is
not qualified as strict CPU-equivalent. The qualifier explicitly pins the
CPU oracle's recorded vector width. Native language LoRA remains FP32; native
V4.1 Full FP16 and automatic scaling remain outside this capability contract.

Scoped Full FP16 uses a genuine FP16 cooperative accumulator for eligible
reductions up to 64, widening the result on the GPU. Longer reductions execute
the original FP32 kernels. A real K=128 GPU test now executes forward, dW and
dX fallback, AdamW, and parameter/moment checkpoint restoration, matching the
FP32 control bit for bit while both cooperative counters remain unchanged.
This component checkpoint test does not advertise full-model optimizer resume.
An additional internal 128-wide image/text run executes two joint updates,
records 144 oversized FP32 reduction fallbacks, and exports real HF packages.
Every retained final logit, vision feature and projected feature reloads
bit-identically. This specifically tests the internal fallback/export route;
the 128-wide package remains rejected by Full FP16 frontend admission.
The Llama64/shared CLIP64 package and adapter/driver admission rules remain
unchanged. Broad original primitive failures remain separate evidence.

Phase XIX-Z release admission requires every one of the thirteen current
packed benchmark/guard pairs to pass. A partial packed result cannot qualify a
release. The checker reexecutes the CPU route guards over the fresh measurement
receipts. Timed route producers use 1,200 observations to balance complete
three-leg and four-leg rotations. Existing batch sweeps, warmups, chains,
candidate graphs, numerical gates and decision bands remain unchanged; older
20/40/120-sample failed sweeps are preserved. Local physical GPU qualification and self-hosted GitHub runner
availability are separate facts; a queued workflow is never counted as PASS.

This scope document is not a release receipt. Current-source producers,
source/executable/shader bindings, negative controls and CI must still pass
before the candidate is declared qualified.
