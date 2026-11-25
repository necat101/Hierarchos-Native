#!/usr/bin/env bash
# Phase X — production promotion of the RDNA3 row-packed attention forward.
#
# Rebuilds the promoted artifact from its source, reruns every qualification
# gate, then re-derives the release measurements and the receipt summary.
# Every step writes into raw/ next to this script; nothing is committed.
#
# The release build dominates the wall clock (~15 min cold). The gates in
# section 3 need only a debug build (~1 min).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RAW=validation/phase_x_row_packed_forward_promotion_20261125/raw
PROMOTED=transformer_attention_forward_rowpacked64_cached_rdna3
GLSLC=${GLSLC:-/c/VulkanSDK/1.4.321.1/Bin/glslc.EXE}
SPIRV_VAL=${SPIRV_VAL:-/c/VulkanSDK/1.4.321.1/Bin/spirv-val.EXE}
FIXTURE=target/gqa-e2e/fixture-8q2kv
mkdir -p "$RAW"

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

step "1. repository identity (must still be the Phase IX base)"
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "diff_sha256=$(git diff | sha256sum | cut -d' ' -f1)"
  echo "staged_index_sha256=$(git diff --cached | sha256sum | cut -d' ' -f1)"
  echo "stash0=$(git rev-parse stash@{0} 2>/dev/null)"
  git status --short
} | tee "$RAW/05_identity-end-of-phase.txt"
[ "$(git rev-parse HEAD)" = "560809eead186f832344c15d1a9b70795da3b75d" ] \
  || fail "HEAD moved off the Phase X base commit"

step "2. rebuild the promoted artifact and validate it"
sha256sum "shaders/$PROMOTED.comp" "shaders/$PROMOTED.spv" | tee "$RAW/03_promoted-artifact-hashes.txt"
"$GLSLC" -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
  "shaders/$PROMOTED.comp" -o "shaders/$PROMOTED.spv" 2>"$RAW/rebuild-stderr.txt" \
  || fail "glslc refused the promoted source"
"$SPIRV_VAL" --target-env vulkan1.2 "shaders/$PROMOTED.spv" || fail "spirv-val rejected the artifact"
python validation/phase_x_row_packed_forward_promotion_20261125/check_promoted_artifact.py \
  > "$RAW/21_promoted-artifact-rebuild.json" || fail "the artifact no longer reproduces byte-for-byte"

step "3. qualification gates (debug build)"
cargo test --lib --no-run >/dev/null 2>&1 || fail "the test harness does not build"
EXE=$(ls -t target/debug/deps/hierarchos_vulkan-*.exe | head -1)
[ -n "$EXE" ] || fail "no debug test binary"
"$EXE" phase_x_row_packed_selector_matrix   --nocapture --test-threads=1 > "$RAW/10_selector-matrix.log" 2>&1 \
  || fail "selector matrix"
"$EXE" phase_x_row_packed_forward_parity    --nocapture --test-threads=1 > "$RAW/11_parity-forward.log" 2>&1 \
  || fail "forward parity matrix"
"$EXE" phase_x_row_packed_fallback_dispatch --nocapture --test-threads=1 > "$RAW/12_fallback.log" 2>&1 \
  || fail "fallback dispatch"
"$EXE" phase_x_row_packed_short_precedence  --nocapture --test-threads=1 > "$RAW/13_short-precedence.log" 2>&1 \
  || fail "selector precedence"
"$EXE" phase_x_row_packed_forward_gradient_parity --nocapture --test-threads=1 \
  > "$RAW/14_parity-gradients.log" 2>&1 || fail "backward and scratch parity"
HIERARCHOS_ATTENTION_FORWARD_DIAGNOSTICS=1 "$EXE" phase_x_row_packed_forward_parity \
  --nocapture --test-threads=1 > "$RAW/15_diagnostics.log" 2>&1 || fail "diagnostics trace"

step "4. release build"
cargo test --release --lib --no-run > target/phase-x-release-build.log 2>&1 \
  || { tail -20 target/phase-x-release-build.log; fail "release build"; }
RE=$(ls -t target/release/deps/hierarchos_vulkan-*.exe | head -1)
[ -n "$RE" ] || fail "no release test binary"

step "5. release complete-step equivalence (arm A generic, arm B production)"
rm -rf "$RAW/checkpoint-production" "$RAW/checkpoint-generic"
HIERARCHOS_PHASE_X_STEP_ARM=production HIERARCHOS_PHASE_X_STEP_STEPS=6 \
  HIERARCHOS_PHASE_X_STEP_DIGEST=1 HIERARCHOS_PHASE_X_STEP_CHECKPOINT="$RAW/checkpoint-production" \
  "$RE" phase_x_row_packed_forward_full_step --nocapture --test-threads=1 \
  > "$RAW/step-production.log" 2>&1 || fail "production arm"
HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1 HIERARCHOS_PHASE_X_STEP_ARM=generic \
  HIERARCHOS_PHASE_X_STEP_STEPS=6 HIERARCHOS_PHASE_X_STEP_DIGEST=1 \
  HIERARCHOS_PHASE_X_STEP_CHECKPOINT="$RAW/checkpoint-generic" \
  "$RE" phase_x_row_packed_forward_full_step --nocapture --test-threads=1 \
  > "$RAW/step-generic.log" 2>&1 || fail "generic arm"
python validation/phase_x_row_packed_forward_promotion_20261125/compare_steps.py "$RAW" \
  > "$RAW/23_step-digest-release.json" || fail "the two arms are not bitwise identical"

step "6. GPU-timestamp kernel comparison (both arm orders)"
HIERARCHOS_PHASE_X_KERNEL_TIMING=1 HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  "$RE" phase_x_row_packed_forward_timing --nocapture --test-threads=1 \
  > "$RAW/timing-production-first.log" 2>&1 || fail "timing, production first"
HIERARCHOS_PHASE_X_KERNEL_TIMING=1 HIERARCHOS_PHASE_X_REVERSE=1 HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  "$RE" phase_x_row_packed_forward_timing --nocapture --test-threads=1 \
  > "$RAW/timing-generic-first.log" 2>&1 || fail "timing, generic first"
python validation/phase_x_row_packed_forward_promotion_20261125/analyze_timing.py \
  "$RAW/timing-production-first.log" "$RAW/timing-generic-first.log" > "$RAW/timing-gpu.json" \
  || fail "timing analysis"

step "7. complete-step ABBA (4 rounds x A B B A)"
for round in 0 1 2 3; do
  for pos in 0 1 2 3; do
    case $pos in
      1|2) arm=production; tag=B; extra="" ;;
      *)   arm=generic;    tag=A; extra="HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1" ;;
    esac
    env HIERARCHOS_PHASE_X_STEP_ARM=$arm HIERARCHOS_PHASE_X_STEP_STEPS=29 $extra \
      "$RE" phase_x_row_packed_forward_full_step --nocapture --test-threads=1 \
      > "$RAW/abba-r$round-p$pos-$tag.log" 2>&1 || fail "ABBA round $round position $pos"
  done
done
python validation/phase_x_row_packed_forward_promotion_20261125/assemble.py "$RAW" 4 \
  > "$RAW/abba-summary.json" || fail "ABBA analysis"

step "8. per-component attribution (profiled arms, 9 steps each)"
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_PHASE_X_STEP_ARM=production HIERARCHOS_PHASE_X_STEP_STEPS=9 \
  "$RE" phase_x_row_packed_forward_full_step --nocapture --test-threads=1 \
  > "$RAW/step-profile-production.log" 2>&1 || fail "profiled production arm"
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1 HIERARCHOS_PHASE_X_STEP_ARM=generic \
  HIERARCHOS_PHASE_X_STEP_STEPS=9 \
  "$RE" phase_x_row_packed_forward_full_step --nocapture --test-threads=1 \
  > "$RAW/step-profile-generic.log" 2>&1 || fail "profiled generic arm"
python validation/phase_x_row_packed_forward_promotion_20261125/attribute.py \
  "$RAW/step-profile-generic.log" "$RAW/step-profile-production.log" > "$RAW/attribution.json" \
  || fail "attribution"

step "9. receipt summary"
python validation/phase_x_row_packed_forward_promotion_20261125/summarize.py >/dev/null \
  || fail "summary"

printf '\nPhase X receipt regenerated under %s\n' "$RAW"
