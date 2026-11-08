# Commands used

All commands were run from `hierarchos-vulkan` on the AMD Radeon host. Heavy
builds and GPU jobs were serialized.

## Builds

```bash
CARGO_INCREMENTAL=0 cargo build --locked -j1 --bin image_text_pipeline_probe
CARGO_INCREMENTAL=0 cargo build --locked -j1 --bin gemma3_vision_probe
CARGO_INCREMENTAL=0 cargo build --locked -j1 --bin gemma4_vision_probe
CARGO_INCREMENTAL=0 cargo build --locked -j1 --bin llava_pipeline_probe
CARGO_INCREMENTAL=0 cargo build --locked -j1 --bin hierarchos-vulkan-transformer-logits
CARGO_INCREMENTAL=0 cargo test --locked -j1 --lib --no-run
CARGO_INCREMENTAL=0 cargo test --release --locked -j1 --lib --no-run
```

## Nine-stage producer

```bash
python validation/qualify_rdna3_full_multimodal.py \
  --frozen target/rdna3-port/gen9-frozen \
  --output target/final-verification-20261108/numerical/full-replay \
  --bin-dir target/debug \
  --test-binary target/debug/deps/hierarchos_vulkan-8b2f33e2a2428e23.exe \
  --device-id 0x15bf
```

## Supplementary numerical checks

```bash
HIERARCHOS_ATEN_VECTOR_WIDTH=8 python validation/qualify_rdna3_gpt2_shapes.py run \
  --fixtures target/rdna3-port/gpt2-shapes-final-cpu \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/final-verification-20261108/numerical/gpt2-shapes

HIERARCHOS_ATEN_VECTOR_WIDTH=8 python validation/qualify_shared_image_pipeline.py run \
  --fixtures target/rdna3-port/cpu-kimi \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/final-verification-20261108/numerical/kimi \
  --target rdna3 --device-id 0x15bf

HIERARCHOS_ATEN_VECTOR_WIDTH=8 python validation/qualify_deepseek_v41_shared_image.py run \
  --fixtures target/rdna3-port/cpu-deepseek-v41-v1 \
  --binary target/debug/deps/hierarchos_vulkan-8b2f33e2a2428e23.exe \
  --output target/final-verification-20261108/numerical/v41-bridge \
  --target rdna3 --device-id 0x15bf
```

## Rust

```bash
target/debug/deps/hierarchos_vulkan-8b2f33e2a2428e23.exe rdna3_gpt2_ --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-8b2f33e2a2428e23.exe
```

## Performance

```bash
python validation/benchmark_rdna3_v41_paired.py \
  --baseline-binary target/phase14/binaries/372407b.exe \
  --candidate-binary target/release/deps/hierarchos_vulkan-0861896ac6f810c4.exe \
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 \
  --output target/final-verification-20261108/benchmark/b1 \
  --rounds 6 --steps 7 --batch 1 \
  --allow-legacy-batch1 --allow-cross-arm-signature-difference \
  --require-native-improvement --forbid-control-regression

for b in 1 2 4; do
  python validation/benchmark_rdna3_v41_paired.py \
    --baseline-binary target/release/deps/hierarchos_vulkan-0861896ac6f810c4.exe \
    --candidate-binary target/release/deps/hierarchos_vulkan-0861896ac6f810c4.exe \
    --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 \
    --output target/final-verification-20261108/benchmark/session-b$b \
    --rounds 6 --steps 7 --batch $b \
    --baseline-env HIERARCHOS_VULKAN_LAYER_SESSION=0
done
```
