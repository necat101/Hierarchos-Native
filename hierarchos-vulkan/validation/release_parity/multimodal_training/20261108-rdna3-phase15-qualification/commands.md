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
