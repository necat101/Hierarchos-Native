#!/usr/bin/env bash
# Phase XIX preservation and identity capture.
#
# The receipt identity method is the Phase XVIII one (and Phase XII–XVII before
# it): for every directory, hash the `sha256sum` lines of its files in sorted
# path order, so a single file edit is visible in one digest.
#
# This phase does not move HEAD, does not commit, does not touch the index and
# does not create objects: the only writes are the files in this receipt.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xix_query_scratch_reuse_20261127
RAW=$PH/raw
mkdir -p "$RAW"

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "tag_v0.2.7_object=$(git rev-parse v0.2.7)"
  echo "tag_v0.2.7_commit=$(git rev-parse 'v0.2.7^{commit}')"
  echo "main=$(git rev-parse main)"
  echo "release_v0.2.7=$(git rev-parse release/v0.2.7)"
  echo "stash_count=$(git stash list | wc -l)"
  git stash list
  echo "stash0=$(git rev-parse 'stash@{0}')"
  echo "tracked_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo
  echo "# status (porcelain, untracked collapsed)"
  git status --porcelain --untracked-files=normal
  echo
  echo "# status (short)"
  git status --short
} | tee "$RAW/00_identity.txt"

{
  echo "# production attention shader identities (unchanged by this phase)"
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
  echo "# experiment artifact identities"
  sha256sum "$PH/shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp" \
            "$PH/shaders/q_scratchreuse.spv" \
            "$PH/shaders/q_control_rebuild.spv"
  echo
  echo "# source identities (transformer.rs is modified by this phase: test seam + module registration)"
  sha256sum src/transformer.rs \
            src/phase_xix_query_scratch_reuse_tests.rs \
            validation/training_shader_manifest.json
} | tee "$RAW/01_production_hashes.txt"

{
  echo "# prior receipt digests, recomputed with the Phase XVII/XIX method"
  echo "# (hash of the sha256sum lines of every file in sorted path order)"
  for d in validation/phase_viii_key_blocks \
           validation/phase_ix_row_packed_forward_20261124 \
           validation/phase_x_row_packed_forward_promotion_20261125 \
           validation/phase_xi_channel_tiling_20261125 \
           validation/phase_xii_channel4_key_production_20261125 \
           validation/phase_xiii_mixed_wmma_requalification_20261125 \
           validation/phase_xiv_v027_closeout_20261125 \
           validation/phase_xv_v027_release_20261125 \
           validation/phase_xvii_v028_optimization_discovery_20261126 \
           validation/phase_xviii_v028_dq_accumulator_promotion_20261126; do
    [ -d "$d" ] || { echo "MISSING $d"; continue; }
    n=$(find "$d" -type f | wc -l)
    h=$(find "$d" -type f | sort | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
    b=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    echo "$h  files=$n  bytes=$b  $d"
  done
} | tee "$RAW/02_preservation.txt"

echo "PHASE_XIX_PRESERVATION_DONE"
