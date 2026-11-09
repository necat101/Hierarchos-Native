# Phase XV canonical commands

Run from `hierarchos-vulkan`. Binaries are the debug probes bound in
`evidence/binaries.json`; the Phase XIV release `final.exe` is performance
evidence only.

```sh
cargo build --offline --bin image_text_pipeline_probe --bin gemma3_vision_probe \
  --bin gemma4_vision_probe --bin llava_pipeline_probe

.venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_shapes.py run \
  --fixtures target/rdna3-port/gpt2-shapes-final-cpu \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/phase15/gpt2/shapes

.venv-torch/Scripts/python.exe validation/qualify_rdna3_full_multimodal.py \
  --frozen target/rdna3-port/gen9-frozen \
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output target/phase15/multimodal/full

.venv-torch/Scripts/python.exe validation/qualify_shared_image_pipeline.py run \
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9/shared-image-cpu-v3 \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/phase15/current/shared --profiles <74 non-Kimi profiles> \
  --target rdna3 --device-id 0x15bf

.venv-torch/Scripts/python.exe validation/qualify_shared_image_pipeline.py run \
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9/shared-image-cpu-v3 \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/phase15/current/kimi \
  --profiles kimi_k3_kda kimi_k3_mla kimi_k3_text \
  --target rdna3 --device-id 0x15bf

.venv-torch/Scripts/python.exe validation/qualify_deepseek_v41_native_pipeline.py \
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output target/phase15/current/v41native --target rdna3 --device-id 0x15bf

.venv-torch/Scripts/python.exe validation/qualify_deepseek_v41_shared_image.py run \
  --fixtures target/rdna3-port/cpu-deepseek-v41-v1 \
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output target/phase15/current/v41bridge --target rdna3 --device-id 0x15bf

cargo test --offline --lib rdna3_gpt2_ -- --test-threads=1

.venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_current.py \
  --run target/phase15/current \
  --cpu-fixtures target/rdna3-port/gpt2-shapes-final-cpu \
  --libtest-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output validation/release_parity/multimodal_training/\
20261108-rdna3-phase15-qualification/current-qualification.json

python validation/check_rdna3_gpt2_receipts.py
python -m unittest discover -s validation -p "test_rdna3_gpt2_receipts.py"
```

## Additional gates (2026-11-09 rebind)

The canonical 74-case shared selection is the 77-case inventory minus the three
single-case `kimi_k3_*` profiles; `kimi_k25_text` is a different, retained
backbone and stays in the canonical set (37 profiles, 74 cases):

```sh
.venv-torch/Scripts/python.exe validation/qualify_shared_image_pipeline.py run \
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9/shared-image-cpu-v3 \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/phase15/current/shared \
  --profiles bert dbrx deepseek_v4 falcon_h1_gated_after falcon_h1_gated_before \
    falcon_h1_silu gemma3 gemma3_softcap_tied gemma4 gpt2 gpt2_wide gpt_oss \
    kimi_k25_text llama llama_wide minimax_m2 minimax_m3 minimax_m3_dense mistral \
    mistral4 mixtral phi3 phi4_multimodal_text qwen2 qwen2_5_gqa \
    qwen2_5_sliding_tied qwen3_5_full qwen3_5_linear qwen3_5_mixed qwen3_5_moe \
    qwen3_next qwen4_exp_linear qwen4_exp_mixed_ple qwen4_exp_ple qwen4_exp_qsa \
    smollm3 smollm3_yarn \
  --target rdna3 --device-id 0x15bf
```

`emit_spirv_gate.py` is the 43-artifact SPIR-V producer. It takes the artifact
inventory and target environments from the sealed Phase XV gate, requires each
committed `.spv` to match its sealed digest, executes the installed `spirv-val`
for every artifact, and marks an artifact `passed` only on a zero exit:

```sh
.venv-torch/Scripts/python.exe validation/emit_spirv_gate.py \
  --output target/phase15/current/spirv.json
```

The native Gemma4 stage emits one case per pixel domain (`gemma4-rgb`,
`gemma4-legacy`); the canonical `gemma4` family is those two genuine results
with their shared binary binding.
