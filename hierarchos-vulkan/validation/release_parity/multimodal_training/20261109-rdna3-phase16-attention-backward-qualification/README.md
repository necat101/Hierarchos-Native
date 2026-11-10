# RDNA3 Phase XVI two-pass attention backward qualification: 2026-11-09

**Status: RDNA3 current-source qualification PASS.** This receipt is the current
source gate. Phase XV remains sealed history at
`../20261108-rdna3-phase15-qualification/` and is never presented as current
evidence; its 43 shader identities are read, its artifacts are re-validated from
the committed bytes, and nothing in it is rewritten.

Source under test: `918ee0d79f0e163670b46e293825a4cffd1e49b5`. The receipt is
committed in its child, because a commit cannot contain a receipt bound to its
own hash. `current-qualification.json` binds **145** source files: the 134
historical bindings plus the two-pass shaders, both SPIR-V artifacts, the scalar
reference shader, the shader manifest, the Phase XVI producer, checker, evidence
parser and negative controls, and `src/transformer.rs` / `src/vulkan.rs`.

All comparisons remain `max_abs < 2e-7`, `rtol = 0`, `strict_less_than`.

## Numerical gates, re-run on this exact source

| Gate | Result | Worst logit |
| --- | --- | ---: |
| GPT-2 frozen shapes | **16/16** | `5.960464477539063e-08` |
| Canonical shared image (37 profiles) | **74/74** | `1.4901161193847656e-07` |
| Kimi K3 | **3/3** | `2.9802322387695312e-08` |
| Seq2seq | **6/6** | `8.940696716308594e-08` |
| Native Gemma3 | **4/4** | `1.7881393432617188e-07` |
| Native Gemma4 (two pixel domains) | **2/2** | `1.4156103134155273e-07` |
| Native LLaVA | **4/4** | `8.940696716308594e-08` |
| GPT-2 primitives | 9 passed, 0 failed | — |
| Native DeepSeek V4.1 | **30/30** | `0.0` |
| V4.1 shared-image bridge | **2/2** | — |
| Upstream preservation | 1,341 files, 672 recipes unchanged | — |

The two Gemma4 cases are two genuine per-domain runs (`gemma4-legacy`,
`gemma4-rgb`) bound as two receipts, not a synthesized merged result.

## Two-pass attention backward

| Measure | Result |
| --- | --- |
| Cases | 4 (`fixture-causal`, `fixture-full`, `padded-causal`, `small-causal`) |
| Tensor comparisons | **12/12** (dQ, dK, dV per case) |
| Elements compared | 3,158,064 |
| Bitwise equal | **3,158,064** (100%) |
| Maximum absolute error | `0.0` |
| Maximum ULP error | `0` |
| Unwritten sentinel elements | `0` in both arms |
| Scalar control arm dispatches | scalar=2, query=0, key=0 |
| Candidate arm dispatches | scalar=0, query=2, key=2 |
| Adapter outputs | bit-identical between arms |
| Harness negative controls | 11/11 detected |

The candidate arm never falls back: `scalar=0` in the profile lines. The
one-shot diagnostic records `kernels=true scratch=true supported=true
key_blocks=1 causal=1 unfused_qk=9 head_dim=128 kv_seq=64`.

## Shader inventory: 45 artifacts

| Scope | Artifacts | Treatment |
| --- | ---: | --- |
| Sealed Phase XV | 43 | committed bytes match the sealed digests; fresh `spirv-val` run; **not** recompiled |
| Phase XVI | 2 | rebuilt byte-identically from the manifest's own flags, then validated |

`transformer_attention_backward_query_rdna3.spv` (`615c27dc…`) and
`transformer_attention_backward_key_rdna3.spv` (`5f69d500…`) were compiled by
`glslc` reporting `2023.8`, `--target-env vulkan1.2`. Validator:
`SPIRV-Tools v2025.3 v2025.3.rc1-0-g33e02568`.

The 43 historical artifacts are byte-verified, not recompiled, and the receipt
says so: `historical.recompiled = 0`. Their provenance is unchanged because
their bytes are unchanged; claiming a fresh rebuild of them would require
`glslangValidator` 15.3.0/16.5.0/16.6.0, which this host does not have.

## Commands

From `hierarchos-vulkan`:

```sh
cargo build --offline --bin image_text_pipeline_probe --bin gemma3_vision_probe \
  --bin gemma4_vision_probe --bin llava_pipeline_probe
cargo build --offline --release --bin hierarchos-vulkan-transformer-train

.venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_shapes.py run \
  --fixtures target/rdna3-port/gpt2-shapes-final-cpu \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/phase16/run/shapes

.venv-torch/Scripts/python.exe validation/qualify_shared_image_pipeline.py run \
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9/shared-image-cpu-v3 \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/phase16/run/shared --profiles <37 canonical profiles> \
  --target rdna3 --device-id 0x15bf

.venv-torch/Scripts/python.exe validation/qualify_rdna3_full_multimodal.py \
  --frozen target/rdna3-port/gen9-frozen \
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output target/phase16/multimodal \
  --stages seq2seq gemma3 gemma4-legacy gemma4-rgb llava

.venv-torch/Scripts/python.exe validation/qualify_deepseek_v41_native_pipeline.py \
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output target/phase16/run/v41native --target rdna3 --device-id 0x15bf

.venv-torch/Scripts/python.exe validation/qualify_deepseek_v41_shared_image.py run \
  --fixtures target/rdna3-port/cpu-deepseek-v41-v1 \
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output target/phase16/run/v41bridge --target rdna3 --device-id 0x15bf

cargo test --offline --lib rdna3_gpt2_ -- --test-threads=1
python validation/verify_rdna3_multimodal_isolation.py \
  --output target/phase16/run/isolation.json
python validation/emit_phase16_spirv_gate.py \
  --output target/phase16/run/phase16-spirv.json

HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 cargo test --offline --lib \
  two_pass_attention_backward_matches_scalar_gradients -- --nocapture
bash validation/two_pass_backward_equivalence.sh

python validation/qualify_phase16_attention_backward.py \
  --run target/phase16/run \
  --cpu-fixtures target/rdna3-port/gpt2-shapes-final-cpu \
  --libtest-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe
python validation/check_phase16_receipts.py
python validation/check_rdna3_gpt2_receipts.py
python -m unittest discover -s validation -p "test_phase16_receipts.py"
```

## Evidence

`evidence/` holds the exact files the receipt binds: the seven family
`qualification.json` files, the primitive log, both V4.1 receipts, the isolation
audit, the 45-artifact shader gate, and the raw two-pass test and harness logs.
The checker re-derives the gradient summary from those logs and re-validates
every family receipt; a `pass: true` field alone is not accepted.

## Limits

- Eight measured GPT-2 geometries, not a Cartesian shape certificate.
- The two-pass backward remains opt-in (`HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1`);
  the scalar path is the default.
- The 43 sealed shader artifacts are byte-verified, not recompiled.
- No physical Gen9 execution; the shader gates are static preservation checks.
- The training qualification in this receipt is the two-arm LoRA rank-8 adapter
  identity plus the retained families; mixed-WMMA full/LoRA and checkpoint
  resume are not re-run here.
