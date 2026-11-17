# Current numerical qualification scope

WMMA is opt-in and admitted only on recognised AMD RDNA3-or-newer PCI devices
with the required cooperative-matrix, subgroup, memory-model and FP16 features.
Vendor-name overrides cannot bypass the physical-device gate. Default `off`
does not query or enable WMMA feature chains and leaves the existing FP32 vendor
selector in use. RDNA1/2 keep their existing AMD FP32 route, and Intel Gen9 keeps
its pre-WMMA route. The RDNA3 key-state arithmetic fix has a separate shader;
the shared source and SPIR-V are restored to pre-WMMA main. CPU CI verifies all
1,301 pre-existing shader files and the FP32 selector against `ead6cd3`.
Live Gen9 qualification will be performed later on that host, as assigned by
the user; these local controls do not claim that hardware execution.

This release target is **AVX512 + RDNA3**. AVX2 fixture reproduction belongs to
the AVX2 qualification host. The target-scoped provenance audit keeps those
assignments explicit as `PENDING_HOST_VERIFICATION`, bound to the original
fixture SHA-256 and recorded CPU. That status certifies neither an AVX2
reproduction nor an AVX512 replacement. The unscoped and AVX2-target audits
still require verified pins; successful exact reproduction on the assigned
CPU removes the pending assignment. The existing AVX512 oracle-archive and
RDNA3 numerical/dispatch gates remain required for this target.

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
spatial aligner. It is qualified **numerically, on the execution route it
actually measured**, not as an independent qualification of cooperative Mixed
WMMA dispatch. The bounded independent 128-wide fixture passes all 65 strict
rows (maximum `1.564621925354004e-7`), and the 65 stress rows are retained with
their larger-magnitude result. In the stress fixture the patch projection is
identical, and first-norm output differs by `2.384185791015625e-7`.
Operand-matched CPU and GPU norms are each within two FP32 ULP of an independent
float64 calculation. Subsequent drift is retained and the stress fixture is
not qualified as strict CPU-equivalent. The qualifier explicitly pins the
CPU oracle's recorded vector width. Native language LoRA remains FP32; native
V4.1 Full FP16 and automatic scaling remain outside this capability contract.

The measured route matters and is recorded in the receipt. At the released
fixture geometry (`8x8` image, 16 vision patches, `vision_dim` 128,
`vision_inter_dim` 256) the widest native vision projection is `wqkv` at
`16 * 384 * 128 = 786,432` MACs. `wmma_shape_profitable` requires
`WMMA_MIN_PROFITABLE_MACS = 16,000,000`, so the crossover policy declines every
native vision projection before any cooperative executor exists. The `mixed`
request therefore executes the FP32 kernels: `accumulation_dispatches [0, 0]`
and phases `[0, 0, 0]`, with the 65 mixed-vs-FP32 comparisons bit-identical to
the FP32 control. That is a numerical qualification of the FP32 route. It is
**not** current-source evidence that cooperative Mixed WMMA dispatch runs.

The current-source RDNA3 WMMA implementation is qualified separately, by the
precision-training, wide-multimodal and primitive evidence families (and by the
`wmma_mixed`, `wmma_full_original` and `wmma_full_dot2` primitive components of
this release). Those components are independent of the native V4.1 fixture and
are not weakened by this scope statement.

Historical disclosure: the pre-`08179c5` releases
(`20261112-unified-release`, `20261115-phase25-current-source`) recorded bounded
`accumulation_dispatches [32, 0]` with phases `[16, 8, 8]` at the same
`1.564621925354004e-7` maximum, because the profitability gate introduced in
`08179c5` did not exist yet. Those receipts remain valid historical records of
the earlier implementation and are preserved unchanged. They are not
current-source proof of V4.1 cooperative dispatch. No tolerance, threshold,
fixture row, stress case or negative control was changed to draw this
distinction; only the claim and its machine-checked scope metadata were
corrected.

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
three-leg and four-leg rotations. Forward/rematerialization also counterbalance
all three physical allocation positions in fresh native processes, pooling
400 raw observations per construction order into 1,200 per leg. Each rotation
uses the original warmup count. Backward fusion balances four construction
positions with 300 observations each, retaining all folded-gradient and
optimizer-update gates even for repeated schedules. Every rotation retains exact parity gates;
the production-policy leg leaves only the target chain unpinned, isolating the
other chains as in the forced arms. No order or run is selected by its winner.
Existing batch sweeps, per-process warmups, chains,
candidate graphs, numerical gates and decision bands remain unchanged; older
20/40/120-sample failed sweeps are preserved. Local physical GPU qualification and self-hosted GitHub runner
availability are separate facts; a queued workflow is never counted as PASS.

This scope document is not a release receipt. Current-source producers,
source/executable/shader bindings, negative controls and CI must still pass
before the candidate is declared qualified.
