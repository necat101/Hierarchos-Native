#!/usr/bin/env bash
# Phase XVII v0.2.8 optimization discovery — fresh profiled baseline.
#
# Measurement only. Nothing is rebuilt that belongs to the product and no
# source, shader, threshold, default or precision policy is edited.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RAW=validation/phase_xvii_v028_optimization_discovery_20261126/raw
FIXTURE=target/gqa-e2e/fixture-8q2kv
TEST_BIN=target/release/deps/hierarchos_vulkan-70a92090d59c92cf.exe
STEP_BIN=target/release/training-stability.exe
mkdir -p "$RAW"

step() { printf '\n=== %s ===\n' "$*"; }

step "1. repository identity"
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "tag_v0.2.7_commit=$(git rev-parse 'v0.2.7^{commit}')"
  echo "main=$(git rev-parse main)"
  echo "staged_index_tree=$(git write-tree)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "stash_count=$(git stash list | wc -l)"
  git stash list
  echo
  echo "# status"
  git status --porcelain --untracked-files=normal
} | tee "$RAW/00_identity.txt"

step "2. production attention shader identities (v0.2.7 promoted artifacts)"
sha256sum shaders/transformer_attention_backward_query_rdna3.comp \
          shaders/transformer_attention_backward_query_rdna3.spv \
          shaders/transformer_attention_backward_key_rdna3.comp \
          shaders/transformer_attention_backward_key_rdna3.spv \
          shaders/transformer_attention_backward_key_channel4_rdna3.comp \
          shaders/transformer_attention_backward_key_channel4_rdna3.spv \
          shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp \
          shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv \
          shaders/wmma_gemm_f16_f32acc.comp \
          shaders/wmma_gemm_f16_f32acc.spv \
          shaders/fp32_to_fp16_native.comp \
          shaders/fp32_to_fp16_native.spv \
          src/wmma.rs src/wmma_linear.rs src/precision_policy.rs \
  | tee "$RAW/01_production_hashes.txt"

step "3. prior receipt integrity"
{
  for d in validation/mha_attention_backward_closeout_20261124 \
           validation/gqa_attention_backward_experiment_20261124 \
           validation/gqa_production_promotion_20261124 \
           validation/gqa_end_to_end_training_20261124 \
           validation/phase_viii_key_blocks \
           validation/phase_ix_row_packed_forward_20261124 \
           validation/phase_x_row_packed_forward_promotion_20261125 \
           validation/phase_xi_channel_tiling_20261125 \
           validation/phase_xii_channel4_key_production_20261125 \
           validation/phase_xiii_mixed_wmma_requalification_20261125 \
           validation/phase_xiv_v027_closeout_20261125 \
           validation/phase_xv_v027_release_20261125; do
    [ -d "$d" ] || { echo "MISSING $d"; continue; }
    n=$(find "$d" -type f | wc -l)
    h=$(find "$d" -type f | sort | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
    b=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    echo "$h  files=$n  bytes=$b  $d"
  done
} | tee "$RAW/02_preservation.txt"

step "4. executable identities"
sha256sum "$TEST_BIN" "$STEP_BIN" target/release/wmma_probe.exe \
          target/release/wmma_qualify.exe target/release/wmma_bench.exe \
          | tee "$RAW/03_executables.txt"

step "5. fixture identities"
sha256sum "$FIXTURE/model/config.json" "$FIXTURE/model/model.safetensors" \
          "$FIXTURE/dataset.jsonl" | tee "$RAW/04_fixture.txt"

profiled() { # mode log
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_WMMA_MODE="$1" HIERARCHOS_PHASE_XII_STEP_ARM=channel4 \
  HIERARCHOS_PHASE_XII_STEP_STEPS=9 \
    "$TEST_BIN" phase_xii_channel4_key_full_step --nocapture --test-threads=1 > "$2" 2>&1
  echo "exit=$? log=$2 lines=$(wc -l < "$2")"
}

step "6. profiled attribution, FP32 arm"
profiled off "$RAW/50_profile_off.log"

step "7. profiled attribution, Mixed arm"
profiled mixed "$RAW/51_profile_mixed.log"

step "8. attribution reduction"
python validation/phase_xvii_v028_optimization_discovery_20261126/attribute.py \
  "$RAW/50_profile_off.log" "$RAW/51_profile_mixed.log" 9 > "$RAW/52_attribution.json" 2> "$RAW/52_attribution.err.txt"
echo "exit=$?"
echo "PHASE_XVII_PROFILE_DONE"
