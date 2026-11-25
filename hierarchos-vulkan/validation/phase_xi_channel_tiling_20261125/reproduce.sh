#!/usr/bin/env bash
# Phase XI — RDNA3 GQA key-pass channel tiling.
#
# Rebuilds every experimental artifact from its single source, re-runs the
# numerical gates, then re-derives the kernel timing, the end-to-end ABBA and the
# attribution. Every step writes into raw/ next to this script; nothing is
# committed, no production shader is rebuilt and no production selector is
# changed.
#
# The release build dominates the wall clock (~15 min cold). The gates in
# sections 4-5 need only a debug build (~1 min) and no release binary.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RAW=validation/phase_xi_channel_tiling_20261125/raw
SRC=validation/phase_xi_channel_tiling_20261125/shaders/transformer_attention_backward_key_chantile_rdna3.comp
GLSLC=${GLSLC:-/c/VulkanSDK/1.4.321.1/Bin/glslc.EXE}
SPIRV_VAL=${SPIRV_VAL:-/c/VulkanSDK/1.4.321.1/Bin/spirv-val.EXE}
SPIRV_DIS=${SPIRV_DIS:-/c/VulkanSDK/1.4.321.1/Bin/spirv-dis.EXE}
FIXTURE=target/gqa-e2e/fixture-8q2kv
PRIMARY=c4_u0
CANDIDATES="c128_u0 c32_u0 c32_u1 c16_u0 c16_u1 c8_u0 c8_u1 c4_u0 c4_u1"
mkdir -p "$RAW"

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

step "1. repository identity (must still be the Phase III-X base)"
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "tracked_diff_sha256=$(git diff --text -- . ':(exclude)*.spv' | sha256sum | cut -d' ' -f1)"
  echo "binary_diff_sha256=$(git diff --binary -- '*.spv' | sha256sum | cut -d' ' -f1)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "staged_index_tree=$(git write-tree)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo "stash0=$(git rev-parse stash@{0} 2>/dev/null)"
  git status --short
} | tee "$RAW/00_identity.txt"
[ "$(git rev-parse HEAD)" = "560809eead186f832344c15d1a9b70795da3b75d" ] \
  || fail "HEAD moved off the Phase III-X base commit"

step "2. rebuild every experimental artifact and validate it"
: > "$RAW/01_shader-hashes.txt"
echo "# source" >> "$RAW/01_shader-hashes.txt"
sha256sum "$SRC" >> "$RAW/01_shader-hashes.txt"
for config in "128 1 0" "32 4 0" "32 4 1" "16 8 0" "16 8 1" "8 16 0" "8 16 1" "4 32 0" "4 32 1"; do
  set -- $config
  chunk=$1; blocks=$2; unroll=$3
  name="key_chantile_c${chunk}_u${unroll}"
  out="validation/phase_xi_channel_tiling_20261125/shaders/$name.spv"
  "$GLSLC" -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
    -DXI_CHANNEL_CHUNK="$chunk" -DXI_CHANNEL_BLOCKS="$blocks" -DXI_UNROLL_CHANNEL="$unroll" \
    "$SRC" -o "$out" 2> "$RAW/rebuild-$name.stderr.txt" \
    || fail "glslc refused $name"
  "$SPIRV_VAL" --target-env vulkan1.2 "$out" || fail "spirv-val rejected $name"
  "$GLSLC" -E -fshader-stage=compute -Ishaders \
    -DXI_CHANNEL_CHUNK="$chunk" -DXI_CHANNEL_BLOCKS="$blocks" -DXI_UNROLL_CHANNEL="$unroll" \
    "$SRC" > "$RAW/preprocessed-$name.glsl" 2>/dev/null || fail "glslc -E for $name"
  printf '%s  chunk=%s blocks=%s unroll=%s  spv=%s  preprocessed_glsl=%s\n' \
    "$name" "$chunk" "$blocks" "$unroll" \
    "$(sha256sum "$out" | cut -d' ' -f1)" \
    "$(sha256sum "$RAW/preprocessed-$name.glsl" | cut -d' ' -f1)" \
    >> "$RAW/01_shader-hashes.txt"
done
{
  echo "compiler=$(basename "$GLSLC")"
  "$GLSLC" --version | head -3
  echo "validator=$($SPIRV_VAL --version | head -1)"
  echo "flags=-fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders"
  echo "workgroup=64x1x1"
  echo "source=$SRC"
} | tee "$RAW/02_toolchain.txt"

step "3. production artifacts must be untouched"
sha256sum shaders/transformer_attention_backward_key_rdna3.comp \
          shaders/transformer_attention_backward_key_rdna3.spv \
          shaders/transformer_attention_backward_query_rdna3.comp \
          shaders/transformer_attention_backward_query_rdna3.spv \
          shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp \
          shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv \
  | tee "$RAW/03_production-hashes.txt"

step "4. per-artifact state inventory"
python validation/phase_xi_channel_tiling_20261125/state_inventory.py "$SPIRV_DIS" \
  shaders/transformer_attention_backward_key_rdna3.spv \
  validation/phase_xi_channel_tiling_20261125/shaders/*.spv \
  > "$RAW/04_state-inventory.json" || fail "state inventory"

step "5. numerical gates (debug build)"
cargo test --lib --no-run >/dev/null 2>&1 || fail "the test harness does not build"
EXE=$(ls -t target/debug/deps/hierarchos_vulkan-*.exe | head -1)
[ -n "$EXE" ] || fail "no debug test binary"
"$EXE" phase_xi_channel_tiling_ownership --nocapture --test-threads=1 \
  > "$RAW/10_ownership.log" 2>&1 || fail "ownership bijection"
"$EXE" phase_xi_channel_tiling_parity --nocapture --test-threads=1 \
  > "$RAW/11_parity.log" 2>&1 || fail "parity matrix"
grep -c 'verdict=PASS' "$RAW/11_parity.log" > "$RAW/11_parity-passes.txt"

step "6. kernel timing, round-major, both within-round orders"
HIERARCHOS_PHASE_XI_KERNEL_TIMING=1 HIERARCHOS_PHASE_XI_KERNEL_ITERATIONS=31 \
  HIERARCHOS_PHASE_XI_ARMS=production,"$PRIMARY" HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  "$EXE" phase_xi_channel_tiling_timing --nocapture --test-threads=1 \
  > "$RAW/20_timing-forward.log" 2>&1 || fail "kernel timing, forward order"
HIERARCHOS_PHASE_XI_KERNEL_TIMING=1 HIERARCHOS_PHASE_XI_KERNEL_ITERATIONS=31 \
  HIERARCHOS_PHASE_XI_ARMS=production,"$PRIMARY" HIERARCHOS_PHASE_XI_REVERSE=1 \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  "$EXE" phase_xi_channel_tiling_timing --nocapture --test-threads=1 \
  > "$RAW/21_timing-reverse.log" 2>&1 || fail "kernel timing, reverse order"
python validation/phase_xi_channel_tiling_20261125/analyze_timing.py \
  "$RAW/20_timing-forward.log" "$RAW/21_timing-reverse.log" > "$RAW/22_timing.json" \
  || fail "timing analysis"

step "7. release build"
cargo test --release --lib --no-run > target/phase-xi-release-build.log 2>&1 \
  || { tail -20 target/phase-xi-release-build.log; fail "release build"; }
RE=$(ls -t target/release/deps/hierarchos_vulkan-*.exe | head -1)
[ -n "$RE" ] || fail "no release test binary"

step "8. training equivalence with digests and both checkpoints"
rm -rf "$RAW/checkpoint-production" "$RAW/checkpoint-$PRIMARY"
HIERARCHOS_PHASE_XI_STEP_ARM=production HIERARCHOS_PHASE_XI_STEP_STEPS=29 \
  HIERARCHOS_PHASE_XI_STEP_DIGEST=1 HIERARCHOS_PHASE_XI_STEP_CHECKPOINT="$RAW/checkpoint-production" \
  "$RE" phase_xi_channel_tiling_full_step --nocapture --test-threads=1 \
  > "$RAW/30_equivalence-production.log" 2>&1 || fail "production equivalence arm"
HIERARCHOS_PHASE_XI_STEP_ARM="$PRIMARY" HIERARCHOS_PHASE_XI_STEP_STEPS=29 \
  HIERARCHOS_PHASE_XI_STEP_DIGEST=1 HIERARCHOS_PHASE_XI_STEP_CHECKPOINT="$RAW/checkpoint-$PRIMARY" \
  "$RE" phase_xi_channel_tiling_full_step --nocapture --test-threads=1 \
  > "$RAW/31_equivalence-$PRIMARY.log" 2>&1 || fail "candidate equivalence arm"
diff <(grep -E 'phase-xi-step\]|phase-xi-checkpoint' "$RAW/30_equivalence-production.log" \
        | sed 's/arm=production/arm=X/; s/step_ms=[0-9.]*//') \
     <(grep -E 'phase-xi-step\]|phase-xi-checkpoint' "$RAW/31_equivalence-$PRIMARY.log" \
        | sed "s/arm=$PRIMARY/arm=X/; s/step_ms=[0-9.]*//") \
  > "$RAW/32_equivalence.diff" 2>&1 || fail "the two arms are not bitwise identical (see 32_equivalence.diff)"
[ -s "$RAW/32_equivalence.diff" ] && fail "equivalence diff is not empty"

step "9. complete-step ABBA (5 rounds x A B B A, 29 steps)"
for round in 0 1 2 3 4; do
  for pos in 0 1 2 3; do
    case $pos in
      1|2) arm=$PRIMARY; tag=B ;;
      *)   arm=production; tag=A ;;
    esac
    HIERARCHOS_PHASE_XI_STEP_ARM=$arm HIERARCHOS_PHASE_XI_STEP_STEPS=29 \
      "$RE" phase_xi_channel_tiling_full_step --nocapture --test-threads=1 \
      > "$RAW/abba-r$round-p$pos-$tag.log" 2>&1 || fail "ABBA round $round position $pos"
  done
done
python validation/phase_xi_channel_tiling_20261125/abba.py "$RAW" 5 production "$PRIMARY" \
  > "$RAW/33_abba.json" || fail "ABBA analysis"

step "10. per-component attribution (profiled, 9 steps each)"
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_PHASE_XI_STEP_ARM=production HIERARCHOS_PHASE_XI_STEP_STEPS=9 \
  "$RE" phase_xi_channel_tiling_full_step --nocapture --test-threads=1 \
  > "$RAW/40_profile-production.log" 2>&1 || fail "profiled production arm"
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_PHASE_XI_STEP_ARM="$PRIMARY" HIERARCHOS_PHASE_XI_STEP_STEPS=9 \
  "$RE" phase_xi_channel_tiling_full_step --nocapture --test-threads=1 \
  > "$RAW/41_profile-$PRIMARY.log" 2>&1 || fail "profiled candidate arm"
python validation/phase_xi_channel_tiling_20261125/attribute.py \
  "$RAW/40_profile-production.log" "$RAW/41_profile-$PRIMARY.log" \
  > "$RAW/42_attribution.json" || fail "attribution"

step "11. receipt summary"
python validation/phase_xi_channel_tiling_20261125/summarize.py "$RAW" > "$RAW/99_summary.json" \
  || fail "summary"

printf '\nPhase XI receipt regenerated under %s\n' "$RAW"
