#!/usr/bin/env bash
# Reproduce the Phase VI promotion checks and measurements.
#
#   bash validation/gqa_production_promotion_20261124/reproduce.sh
#
# Writes fresh logs into raw-reproduced/ so the bound receipt in raw/ is never
# overwritten. Every step reports its own exit status; the script does not stop at
# the first failure.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=validation/gqa_production_promotion_20261124/raw-reproduced
mkdir -p "$OUT"

status() { # label exit
  printf '%-26s exit=%s\n' "$1" "$2"
}

cargo test --offline --lib transformer::tests::two_pass_attention_backward_matches_scalar_gradients \
  -- --nocapture --test-threads=1 > "$OUT/parity-mha.log" 2>&1
status mha-matrix "$?"

cargo test --offline --lib transformer::tests::grouped_query_attention_two_pass_matches_scalar_gradients \
  -- --nocapture --test-threads=1 > "$OUT/parity-gqa.log" 2>&1
status gqa-matrix "$?"

cargo test --offline --lib grouped_query_attention -- --nocapture --test-threads=1 \
  > "$OUT/gqa-production-route.log" 2>&1
status production-route "$?"

{
  cargo test --offline --lib two_pass -- --nocapture --test-threads=1 2>&1
  cargo test --offline --lib optional_kernel -- --nocapture --test-threads=1 2>&1
  cargo test --offline --lib gradient_report -- --nocapture --test-threads=1 2>&1
} > "$OUT/gate-scratch-fallback.log" 2>&1
status selector-scratch-fallback "$?"

cargo build --release --offline --bin hierarchos-vulkan-transformer-train \
  --bin hierarchos-vulkan-training-qualification > "$OUT/release-build.log" 2>&1
status release-build "$?"

if [ ! -d target/wmma-fixture ]; then
  python validation/make_wmma_fixture.py --output target/wmma-fixture --hidden 1024 \
    --layers 2 --vocab 1024 --seq-len 64 --batch-size 8 --sequences 32 \
    > "$OUT/fixture.log" 2>&1
  status fixture "$?"
fi

bash validation/two_pass_backward_equivalence.sh > "$OUT/equivalence.log" 2>&1
status adapter-parity "$?"

HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING=1 \
  HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING_ITERATIONS=40 \
  cargo test --offline --lib transformer::tests::two_pass_attention_kernel_timing \
  -- --nocapture --test-threads=1 > "$OUT/timing.log" 2>&1
status timing-wall "$?"

HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING=1 \
  HIERARCHOS_ATTENTION_BACKWARD_KERNEL_TIMING_ITERATIONS=10 \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  cargo test --offline --lib transformer::tests::two_pass_attention_kernel_timing \
  -- --nocapture --test-threads=1 > "$OUT/timing-profiled.log" 2>&1
status timing-profiled "$?"

python validation/gqa_production_promotion_20261124/analyze_gpu.py \
  "$OUT/timing-profiled.log" > "$OUT/timing-gpu.txt" 2>&1 \
  || python3 validation/gqa_production_promotion_20261124/analyze_gpu.py \
       "$OUT/timing-profiled.log" > "$OUT/timing-gpu.txt" 2>&1
status analyze-timing "$?"

{
  echo "# Phase VI artifact identities"
  sha256sum src/transformer.rs \
    shaders/transformer_attention_backward.comp \
    shaders/transformer_attention_backward.spv \
    shaders/transformer_attention_backward_query_rdna3.comp \
    shaders/transformer_attention_backward_query_rdna3.spv \
    shaders/transformer_attention_backward_key_rdna3.comp \
    shaders/transformer_attention_backward_key_rdna3.spv
} > "$OUT/source-hashes.txt"

grep -hE "qualification: status|fallback qualification|production-route qualification" \
  "$OUT"/parity-*.log "$OUT"/gate-scratch-fallback.log "$OUT"/gqa-production-route.log
echo "DONE: $OUT"
