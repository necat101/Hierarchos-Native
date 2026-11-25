#!/usr/bin/env bash
# Phase XIV — release closeout. Captures identity, preservation and the
# production artifact hashes. Read-only: it writes nothing outside raw/.
set -uo pipefail
# The tracked-text digest Phase XIII recorded is the repository-root pathspec
# variant, so capture from the root too (the subdirectory variant differs only
# because it drops the root-level RDNA3_WMMA.md from the pathspec).
cd "$(dirname "$0")/../.." || exit 1
cd ..

RAW=hierarchos-vulkan/validation/phase_xiv_v027_closeout_20261125/raw
mkdir -p "$RAW"

step() { printf '\n=== %s ===\n' "$*"; }

step "1. repository identity"
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "staged_index_tree=$(git write-tree)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "binary_only_diff_sha256=$(git diff --binary -- '*.spv' | sha256sum | cut -d' ' -f1)"
  echo "tracked_text_diff_sha256=$(git diff --text -- . ':(exclude)*.spv' | sha256sum | cut -d' ' -f1)"
  echo "tracked_text_diff_sha256_subdir=$(git -C hierarchos-vulkan diff --text -- . ':(exclude)*.spv' | sha256sum | cut -d' ' -f1)"
  echo "stash0=$(git rev-parse stash@{0})"
  echo "stash_count=$(git stash list | wc -l)"
  echo
  echo "# staged diff (expected: empty)"
  git diff --cached --stat
  echo
  echo "# tracked changes"
  git diff --stat
  echo
  echo "# untracked"
  git status --porcelain=v1 | grep '^??'
} | tee "$RAW/00_identity.txt"

step "2. prior receipts still present"
for d in mha_attention_backward_closeout_20261124 \
         gqa_attention_backward_experiment_20261124 \
         gqa_production_promotion_20261124 \
         gqa_end_to_end_training_20261124 \
         phase_viii_key_blocks \
         phase_ix_row_packed_forward_20261124 \
         phase_x_row_packed_forward_promotion_20261125 \
         phase_xi_channel_tiling_20261125 \
         phase_xii_channel4_key_production_20261125 \
         phase_xiii_mixed_wmma_requalification_20261125 \
         phase_xiv_v027_closeout_20261125; do
  # Hash over paths relative to hierarchos-vulkan, which is how Phase XIII
  # recorded these digests (the path string is part of each sha256sum line).
  n=$(find "hierarchos-vulkan/validation/$d" -type f | wc -l)
  h=$(cd hierarchos-vulkan && find "validation/$d" -type f | sort \
        | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
  printf '%s  files=%s  %s\n' "$h" "$n" "validation/$d"
done | tee "$RAW/01_prior_receipt_hashes.txt"

step "3. production artifacts"
sha256sum \
  hierarchos-vulkan/shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp \
  hierarchos-vulkan/shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv \
  hierarchos-vulkan/shaders/transformer_attention_backward_query_rdna3.comp \
  hierarchos-vulkan/shaders/transformer_attention_backward_query_rdna3.spv \
  hierarchos-vulkan/shaders/transformer_attention_backward_key_rdna3.comp \
  hierarchos-vulkan/shaders/transformer_attention_backward_key_rdna3.spv \
  hierarchos-vulkan/shaders/transformer_attention_backward_key_channel4_rdna3.comp \
  hierarchos-vulkan/shaders/transformer_attention_backward_key_channel4_rdna3.spv \
  hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.comp \
  hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.spv \
  hierarchos-vulkan/src/wmma.rs hierarchos-vulkan/src/wmma_linear.rs \
  hierarchos-vulkan/src/precision_policy.rs \
  hierarchos-vulkan/src/transformer.rs hierarchos-vulkan/src/vulkan.rs \
  hierarchos-vulkan/src/lib.rs \
  hierarchos-vulkan/src/bin/transformer_train.rs \
  hierarchos-vulkan/validation/training_shader_manifest.json \
  | tee "$RAW/02_production_artifacts.txt"

step "4. gate thresholds as they stand in source"
grep -n -E 'WMMA_MIN_PROFITABLE_MACS|MIXED_VS_MATCHED_ORACLE_MAX_ABS|MIXED_VS_FP32|FULL_VS_FP32_MAX_ABS|FULL_VS_FP32_RMS|FULL_FP16_MAX_REDUCTION' \
  hierarchos-vulkan/src/wmma.rs hierarchos-vulkan/src/precision_policy.rs \
  | tee "$RAW/03_gate_thresholds.txt"
echo | tee -a "$RAW/03_gate_thresholds.txt"
echo "# are the threshold-bearing files modified in this worktree?" | tee -a "$RAW/03_gate_thresholds.txt"
git status --porcelain=v1 -- hierarchos-vulkan/src/wmma.rs \
  hierarchos-vulkan/src/wmma_linear.rs hierarchos-vulkan/src/precision_policy.rs \
  | tee -a "$RAW/03_gate_thresholds.txt"
echo "(empty above means both files are at HEAD and no threshold moved)" \
  | tee -a "$RAW/03_gate_thresholds.txt"

step "done"
