#!/usr/bin/env bash
# Phase IX — RDNA3 wave64 row-packed attention-forward experiment.
#
# Rebuilds the candidate shaders, rebuilds the test harness, reruns the forward
# and gradient parity sweeps, reruns the GPU-timestamp kernel timing in both arm
# orders, reruns the complete-step digest parity and the complete-step ABBA, and
# regenerates summary.json.
#
# Nothing here touches shaders/ (the production shader directory), the
# production selector, the production eligibility predicates or any default
# route. The candidate sources and their SPIR-V live under validation/, the
# harness module is #[cfg(test)], and nothing is committed.
set -euo pipefail

cd "$(dirname "$0")/../.."
RECEIPT=validation/phase_ix_row_packed_forward_20261124
SHADERS=$RECEIPT/shaders
RAW=$RECEIPT/raw
mkdir -p "$RAW"

echo "== 0. preservation evidence =="
{
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "head: $(git rev-parse HEAD)"
  echo "stash-commit: $(git rev-parse stash@{0})"
  echo "diff-sha256: $(git diff | sha256sum | cut -d' ' -f1)"
  echo "cached-diff-sha256: $(git diff --cached | sha256sum | cut -d' ' -f1)"
  echo
  echo "git status --short:"
  git status --short
  echo
  echo "git diff --stat:"
  git diff --stat
} | tee "$RAW/00_preservation.txt"
sha256sum \
  shaders/transformer_attention_forward.comp \
  shaders/transformer_attention_forward.spv \
  shaders/transformer_attention_short_amd_rdna3.comp \
  shaders/transformer_attention_short_amd_rdna3.spv \
  shaders/transformer_attention_backward.comp \
  shaders/transformer_attention_backward.spv \
  shaders/transformer_attention_backward_query_rdna3.comp \
  shaders/transformer_attention_backward_query_rdna3.spv \
  shaders/transformer_attention_backward_key_rdna3.comp \
  shaders/transformer_attention_backward_key_rdna3.spv \
  | tee "$RAW/01_production-shader-hashes.txt"

echo "== 1. candidate shaders =="
# glslc from the Vulkan SDK; shaderc 2023.8 is the release the manifest records
# for the shipped RDNA3 query pass, and the flags below are that recipe.
for SRC in transformer_attention_forward_rowpacked64_stream_rdna3 \
           transformer_attention_forward_rowpacked64_cached_rdna3; do
  glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O \
    -Ishaders "$SHADERS/$SRC.comp" -o "$SHADERS/$SRC.spv"
  spirv-val --target-env vulkan1.2 "$SHADERS/$SRC.spv"
  echo "  $SRC ok"
done
sha256sum "$SHADERS"/*.comp "$SHADERS"/*.spv | tee "$RAW/02_candidate-shader-hashes.txt"
glslc --version | head -3 | tee "$RAW/03_shader-compiler.txt"

echo "== 2. test build =="
cargo test --offline --lib phase_ix --no-run
EXE=$(ls -t target/debug/deps/hierarchos_vulkan-*.exe | head -1)
echo "  harness: $EXE" | tee "$RAW/04_harness.txt"

echo "== 3. forward parity (GQA, MHA, modes, masks, boundary) =="
HIERARCHOS_WMMA_MODE=off "$EXE" phase_ix_row_packed_forward_parity \
  --nocapture --test-threads=1 > "$RAW/parity-forward.log" 2>&1
grep 'phase-ix-parity-summary' "$RAW/parity-forward.log"

echo "== 4. gradient parity through the production two-pass backward =="
HIERARCHOS_WMMA_MODE=off HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 "$EXE" \
  phase_ix_row_packed_forward_gradient_parity \
  --nocapture --test-threads=1 > "$RAW/parity-gradients.log" 2>&1
grep 'phase-ix-gradient-summary' "$RAW/parity-gradients.log"

echo "== 5. complete-step digest parity (6 steps, gradients + parameters) =="
for ARM in generic rowpacked-cached; do
  HIERARCHOS_WMMA_MODE=off HIERARCHOS_PHASE_IX_STEP_ARM=$ARM \
  HIERARCHOS_PHASE_IX_STEP_STEPS=6 HIERARCHOS_PHASE_IX_STEP_DIGEST=1 "$EXE" \
    phase_ix_row_packed_forward_full_step \
    --nocapture --test-threads=1 > "$RAW/step-digest-$ARM.log" 2>&1
done
python "$RECEIPT/compare_steps.py" "$RAW" | tee "$RAW/step-digest-comparison.json"

echo "== 6. GPU timestamps, five processes, alternating arm order =="
for R in 0 1 2 3 4; do
  ORDER=$((R % 2))
  HIERARCHOS_WMMA_MODE=off HIERARCHOS_PHASE_IX_REVERSE=$ORDER \
  HIERARCHOS_PHASE_IX_KERNEL_TIMING=1 HIERARCHOS_PHASE_IX_KERNEL_ITERATIONS=25 \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=64 "$EXE" \
    phase_ix_row_packed_forward_timing --nocapture --test-threads=1 \
    > "$RAW/timing-run$R-order$ORDER.log" 2>&1
  echo "  run $R (reverse=$ORDER) -> $RAW/timing-run$R-order$ORDER.log"
done
python "$RECEIPT/analyze_timing.py" "$RAW"/timing-run*.log > "$RAW/timing-gpu.json"
python "$RECEIPT/assemble.py" "$RAW/timing-gpu.json" > "$RAW/timing-stats.json"

echo "== 7. complete-step ABBA (debug harness; set HIERARCHOS_PHASE_IX_ABBA_RELEASE=1 for release) =="
python "$RECEIPT/run_abba.py" | tee "$RAW/abba-stdout.txt"
cp "$RAW/abba-summary.json" "$RAW/abba-summary-${HIERARCHOS_PHASE_IX_ABBA_RELEASE:+release}.json" 2>/dev/null || true

echo "== 8. summary =="
python "$RECEIPT/summarize.py" > "$RECEIPT/summary.json"
echo "== done (nothing committed) =="
