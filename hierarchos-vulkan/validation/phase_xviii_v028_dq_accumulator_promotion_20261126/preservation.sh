#!/usr/bin/env bash
# Phase XVIII identity and preservation capture.
#
# The receipt identity method is the Phase XVII one (and Phase XII–XV before it):
# for every directory, hash the `sha256sum` lines of its files in sorted path
# order, so a single file edit is visible in one digest.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xviii_v028_dq_accumulator_promotion_20261126
RAW=$PH/raw
mkdir -p "$RAW"

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "tag_v0.2.7_object=$(git rev-parse v0.2.7)"
  echo "tag_v0.2.7_commit=$(git rev-parse 'v0.2.7^{commit}')"
  echo "main=$(git rev-parse main)"
  echo "stash_count=$(git stash list | wc -l)"
  git stash list
  echo "staged_index_tree=$(git write-tree)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo
  echo "# status"
  git status --porcelain --untracked-files=normal
} | tee "$RAW/00_identity.txt"

{
  echo "# production attention shader identities"
  sha256sum shaders/transformer_attention_backward_query_rdna3.comp \
            shaders/transformer_attention_backward_query_rdna3.spv \
            shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
            shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
            shaders/transformer_attention_backward_key_rdna3.comp \
            shaders/transformer_attention_backward_key_rdna3.spv \
            shaders/transformer_attention_backward_key_channel4_rdna3.comp \
            shaders/transformer_attention_backward_key_channel4_rdna3.spv \
            shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp \
            shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv \
            shaders/wmma_gemm_f16_f32acc.comp \
            shaders/wmma_gemm_f16_f32acc.spv
  echo
  echo "# source identities"
  sha256sum src/transformer.rs \
            src/phase_xvii_v028_optimization_discovery_tests.rs \
            src/phase_xviii_dq_accumulator_promotion_tests.rs \
            examples/amd_shader_resource_probe.rs \
            validation/training_shader_manifest.json
} | tee "$RAW/01_production_hashes.txt"

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
           validation/phase_xv_v027_release_20261125 \
           validation/phase_xvii_v028_optimization_discovery_20261126; do
    [ -d "$d" ] || { echo "MISSING $d"; continue; }
    n=$(find "$d" -type f | wc -l)
    h=$(find "$d" -type f | sort | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
    b=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    echo "$h  files=$n  bytes=$b  $d"
  done
} | tee "$RAW/02_preservation.txt"
