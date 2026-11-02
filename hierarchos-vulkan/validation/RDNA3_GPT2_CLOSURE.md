# RDNA3 GPT-2 parity closure: 2026-11-02

The padded GPT-2-wide gap from the [post-Gen9 audit](RDNA3_POST_GEN9_RECEIPT.md)
is closed on the Ryzen Z1 Extreme / AMD Radeon Graphics (`1002:15bf`, driver
`32.0.31007.6002`). All **166 forward runs** and **72 generation requests** pass
against the original CPU reference words with **`max_abs < 2e-7`, `rtol=0`**.
Resident and streamed execution match every output word. The original
expectations, checkpoints, tolerance and padding policy are unchanged.

The [machine-readable receipt](release_parity/rdna3_gpt2_closure/20261102/qualification.json)
binds source and binary hashes to the [complete evidence archive](release_parity/rdna3_gpt2_closure/20261102/evidence.zip).
The earlier failing receipt and all original Gen9 archives remain intact.

## Arithmetic repair

Two CPU contracts were required. The pinned FP32 PyTorch unary implementation
uses MKL VML tanh, whereas the existing shader reconstructs SLEEF u10. A new
FP32 two-float exponential reconstructs high-accuracy tanh for GPT-2 GELU;
the degree-14 exponential coefficients are the nearest two-float expansions
of `1/n!`. This uses no FP64 shader capability and no fitted constants.

For dropout-free attention rows shorter than eight keys, CPU SDPA evaluates
scalar exponential tails, contracts unnormalized exponentials with V, then
applies the reciprocal sum to the result. New attention variants preserve that
order and use correctly rounded reciprocals. Backward reconstructs matching
exponentials and reciprocals. Longer rows keep the existing contraction order.

These contracts were inspected in the pinned PyTorch revision
`5c4886908584029761b579af026dcfb627c84070`:
[unary dispatch](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp),
[MKL VML specialization](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vml.h),
and [CPU FlashAttention](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp).
The retained CPU references use Transformers
`469230357aab0f2b303b0d638c1f8d06edb14184`, Torch `2.14.1+cpu`, AVX2 and one thread.

Selection is automatic inside the existing **RDNA3 GPT-2** repair route.
Intel Gen9 and other architecture routes retain their original pipelines.
The existing K-block-128 projection remains enabled. Shared attention source
changes are preprocessor-guarded, so rebuilding all old shader variants
preserves their bytes.

## Measured gates

| Gate | Result |
| --- | --- |
| Forward, resident and streaming | **166/166**, 43 profiles / 26 identifiers; worst `1.7881393432617188e-7` |
| GPT-2 wide, unmasked and padded | Both modes pass; worst `1.7881393432617188e-7` |
| Generation, cache on/off | **72/72**; worst `8.940696716308594e-8`; generated tokens match |
| Selected native gate | **269 passed**, including 190 default V4.1 tests, streaming, offload and four explicit release/training gates |
| V4.1 default logit comparisons | **201,714**, worst `1.4901161193847656e-7` |
| V4.1 explicit package/resident/streamed/PEFT gates | **11,184 logits, bit-exact** |
| GPT-2 tests | **50 passed**, including the original frozen wide fixture in both storage modes and the Intel selection guard |
| GPT-2 CPU LoRA, resident and streamed | Gradients `2.7939677238464355e-9`; two-step AdamW factors `1.6763806343078613e-8`; resume/uninterrupted **0** |
| GPT-2 adapter lifecycle | Save/reload, merge/unmerge and exported inference pass the strict CPU gates in both modes |
| Shader provenance and ABI | **573/573** active artifacts reproduce; **431** push blocks / **108** host structs agree |
| Gen9 preservation | All **585** pre-existing shader binaries byte-identical; Intel routing guard passes |
| Workspace / CLI | All targets across **4 crates / 65 binaries** check; **41 CLI tests pass** |
| Harness negative controls | **16** architecture, **14** fixture, **7** runner and **5** handoff tests pass |

The permanent [GPT-2 regression fixture](fixtures/gpt2_rdna3_parity/fixture.json)
copies the original checkpoint and CPU words from the immutable final Gen9
archive. Its model is 571,048 bytes; its SHA-256 is
`29f3a1856af8127137152bbc9530e8c66ea33a392ba3a9da2d35446b6aac89cc`.
All 68 original V4.1 fixture files retain their recorded hashes.
Against the completed post-Gen9 audit, all non-GPT-2 outputs remain word-exact:
61,260 forward words across 158 reports and 8,704 generation words across 68 reports.

No physical Intel adapter is available on this host. The Gen9 preservation
claim covers unchanged binaries, unchanged default preprocessor paths and
guard-tested selection; the existing [physical Gen9 receipt](GEN9_FINAL_RECEIPT.md)
retains its device/revision scope.

This qualifies the retained scalable checkpoints and measured inference/PEFT
scopes. The remaining 120 registry identifiers, all-profile CPU PEFT training,
production weights, CUDA and production throughput are outside this pass.
Forward CPU comparisons retain the existing observable-query policy; storage
identity includes padded rows. SmolLM3 retains its existing full-prefix fallback.
Native test filters overlap and their counts must not be added as unique tests.
Historical fixture toolchain gaps remain in [the provenance audit](V41_FIXTURE_PROVENANCE.md).
Parameter streaming remains opt-in through `--stream-parameters` or the GUI toggle.

To reproduce the matrix from frozen references, use the [RDNA3 runbook](../../RDNA3_REGRESSION.md).
The archive also retains the full regression commands, native logs, CPU PEFT
logs/reports, shader provenance, workspace checks and its packaging script.
