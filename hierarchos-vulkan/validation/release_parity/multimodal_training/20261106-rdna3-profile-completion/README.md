# RDNA3 multimodal profile completion, 2026-11-06

This receipt closes the six shared-image profile gaps in the retained RDNA3
suite, adds the final native DeepSeek V4.1 image profile, and closes the
GPT-2-wide residual. The acceptance policy is unchanged: `max_abs < 2e-7`,
`rtol = 0`, strict comparison.

The six previously missing shared profiles now pass 9/9 cases on the physical
AMD Radeon Graphics adapter (`1002:15bf`):

| profile | cases | worst visible logit | gate |
| --- | ---: | ---: | --- |
| `bart` | 2 | 6.705522537231445e-08 | PASS |
| `t5` | 2 | 4.470348358154297e-08 | PASS |
| `switch_transformers` | 2 | 5.960464477539063e-08 | PASS |
| `kimi_k3_kda` | 1 | 2.9802322387695312e-08 | PASS |
| `kimi_k3_mla` | 1 | 2.2351741790771484e-08 | PASS |
| `kimi_k3_text` | 1 | 3.166496753692627e-08 | PASS |

The seventh inventory profile, `deepseek_v41_scalable`, is not a CLIP text
backbone. It uses the native `DeepseekV41MultimodalTrainer`; the shared-image
bridge binds that native fixture to the inventory rather than aliasing it to
the generic CLIP path. It passes 2/2 cases with 4 optimizer updates, 1,360
logits, and 1,866 comparison rows over 155,444 values (155,179 bit-exact).
Worst non-exact value: `1.4901161193847656e-08`.

## Inventory status

- Expected profiles: 44.
- Fully green profiles: **44/44**.
- Image cases executed: 85/85.
- Passing image cases: 85/85.
- GPT-2-wide/unmasked final logits: `1.7881393432617188e-07`.
- GPT-2-wide/padded final logits: `1.7881393432617188e-07`.
- `rdna3_multimodal_qualified`: **true**. The strict gate and CPU expectations
  are unchanged.

## GPT-2-wide closure

The forward stage tape is bit-exact against eager ATen for every operation in
both layers (`divergent_operations: 0`). The last mismatch was the backward
attention seed: the short forward kernel had moved to SLEEF-u10 exp, an exact
FP32 reciprocal and separately rounded QK products, while GPT-2 backward still
reconstructed probabilities with hardware `exp`, `1.0/value` and fused QK.

`transformer_attention_backward_short_gpt2_amd_rdna3.spv` applies the same QK
muladd, SLEEF-u10 exp, exact reciprocal and serial exp sum in backward; P@V
remains fused FMA. The full RDNA3 suite then passes both GPT-2-wide cases with
no tolerance change: unmasked worst `1.7881393432617188e-07`, padded worst
`1.9371509552001953e-07`.

Earlier candidates that did **not** close the gate remain recorded in
`gpt2-wide-residual.json`: K=128 block gating, small-Welford off, AVX2 row GEMM
at width 16, and lane-folded QK/P@V. Each was measured and reverted.

## Receipt guard

`validation/check_rdna3_multimodal_receipts.py` binds every retained receipt by
SHA-256 (`receipt-hashes.json`), re-derives the 44/44 and 85/85 totals from the
per-profile rows, verifies the unchanged `max_abs < 2e-7` / `rtol = 0` policy,
and requires the GPT-2-wide closure evidence. Regenerate the manifest after
changing a bound receipt:

```text
python validation/check_rdna3_multimodal_receipts.py --write-manifest
python validation/check_rdna3_multimodal_receipts.py
```

## Reproduce

The shared suite and GPT-2-wide gate were generated with:

```text
python validation/qualify_shared_image_pipeline.py run \
  --fixtures target/rdna3-port/cpu-shared-v2 \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/rdna3-port/full-suite-final \
  --target rdna3 --device-id 0x15bf
```

Inventory aggregation:

```text
python validation/summarize_rdna3_multimodal_inventory.py --root target/rdna3-port \
  --shared target/rdna3-port/full-suite-final/qualification.json \
  --output validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/inventory-status.json \
  --six-output validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/six-profile-replay.json
```

The GPT-2-specific backward artifact is compiled from
`transformer_attention_backward.comp` with `-DRDNA3_SHORT_GPT2`; the short
forward artifact uses `-DRDNA3_CPU_RECIPROCAL -DRDNA3_SLEEF_EXP
-DRDNA3_QK_MULADD`. The qualifier accepts a caller-supplied target and device ID
for receipt provenance, but the native test uses the machine's default Vulkan
adapter. The RDNA3 run was made on `0x1002:0x15bf`; no Intel Gen9 execution is
claimed.

## Files

- `inventory-status.json`: joined 44-profile status and case counts.
- `gpt2-wide-residual.json`: forward-exact and backward-alignment closure
  evidence, plus the rejected candidates.
- `receipt-hashes.json`: SHA-256 bindings enforced by the receipt guard.
- `six-profile-replay.json`: the six newly green shared profiles and cases.
- `seq2seq-six-qualification.json`: full BART/T5/Switch receipt.
- `kimi-three-qualification.json`: full KIMI KDA/MLA/text receipt.
- `v41-shared-image-profile.json`: generated final profile manifest bound to
  `validation/fixtures/deepseek_v41_native_pipeline_avx2.json`.
- `v41-shared-bridge-qualification.json`: inventory bridge receipt.
- `v41-native-joint-qualification.json`: native joint fixture receipt.
- `v41-native-full-qualification.json`: 30-execution native V4.1 receipt.

The Gen9 reproduction pack is
[`validation/intel-handoff/MULTIMODAL_GEN9.md`](../../../intel-handoff/MULTIMODAL_GEN9.md).
