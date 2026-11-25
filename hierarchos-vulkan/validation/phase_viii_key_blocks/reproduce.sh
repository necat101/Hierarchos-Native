#!/usr/bin/env bash
# Phase VIII — RDNA3 GQA key-pass key-block experiment.
#
# Rebuilds the candidate shaders, builds the test harness, reruns numerical parity
# (including the measured key-ownership check), reruns the GPU-timestamp timing in
# both arm orders, reruns the production regression tests, and regenerates
# summary.json.
#
# Nothing here touches shaders/, src/ outside the #[cfg(test)] harness module, the
# eligibility predicate, the selector or the default route. Nothing is committed.
set -euo pipefail

cd "$(dirname "$0")/../.."
RECEIPT=validation/phase_viii_key_blocks
OUT=$RECEIPT/shaders

echo "== 1. candidate shaders =="
for L in 64 32 16 8; do
  glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
    -DKEY_WORKGROUP_LANES="$L" \
    "$OUT/transformer_attention_backward_key_blockexp_rdna3.comp" \
    -o "$OUT/transformer_attention_backward_key_blockexp_l${L}_rdna3.spv"
  spirv-val --target-env vulkan1.2 \
    "$OUT/transformer_attention_backward_key_blockexp_l${L}_rdna3.spv"
  echo "  lanes=$L ok"
done
sha256sum "$OUT"/*.spv "$OUT"/*.comp

echo "== 2. test build =="
cargo test --offline --lib phase_viii --no-run
EXE=$(ls -t target/debug/deps/hierarchos_vulkan-*.exe | head -1)
echo "  harness: $EXE"

mkdir -p "$RECEIPT/raw"

echo "== 3. numerical parity and key ownership (both modes, five geometries) =="
HIERARCHOS_WMMA_MODE=off "$EXE" phase_viii_key_block_parity \
  --nocapture --test-threads=1 > "$RECEIPT/raw/parity.log" 2>&1
grep 'phase-viii-parity-summary' "$RECEIPT/raw/parity.log"

echo "== 4. GPU timestamps, order 1 then reversed =="
for MODE in forward reverse; do
  if [ "$MODE" = reverse ]; then
    ORDER=1
    LOG=$RECEIPT/raw/timing-reverse.log
  else
    ORDER=0
    LOG=$RECEIPT/raw/timing-8q2kv-4q2kv.log
  fi
  HIERARCHOS_WMMA_MODE=off \
  HIERARCHOS_PHASE_VIII_REVERSE=$ORDER \
  HIERARCHOS_PHASE_VIII_KERNEL_TIMING=1 \
  HIERARCHOS_PHASE_VIII_KERNEL_ITERATIONS=25 \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=64 \
    "$EXE" phase_viii_key_block_timing --nocapture --test-threads=1 > "$LOG" 2>&1
  echo "  $MODE -> $LOG"
done
python "$RECEIPT/analyze_timing.py" "$RECEIPT/raw/timing-8q2kv-4q2kv.log" \
  > "$RECEIPT/raw/timing-gpu.json"
python "$RECEIPT/analyze_timing.py" "$RECEIPT/raw/timing-reverse.log" \
  > "$RECEIPT/raw/timing-gpu-reverse.json"

echo "== 5. production regression (unchanged artifacts) =="
HIERARCHOS_WMMA_MODE=off HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 "$EXE" \
  two_pass_attention_backward_matches_scalar_gradients --nocapture --test-threads=1 \
  > "$RECEIPT/raw/production-two-pass-regression.log" 2>&1
grep 'two-pass gradient qualification' "$RECEIPT/raw/production-two-pass-regression.log"
HIERARCHOS_WMMA_MODE=off "$EXE" \
  optional_two_pass_load_failure_keeps_scalar_execution_and_gradients \
  --nocapture --test-threads=1 > "$RECEIPT/raw/failclosed.log" 2>&1
grep 'two-pass fallback qualification' "$RECEIPT/raw/failclosed.log"

echo "== 6. summary =="
python "$RECEIPT/assemble.py" > "$RECEIPT/raw/tables.txt"
cat "$RECEIPT/raw/tables.txt"
echo "== done (nothing committed) =="
