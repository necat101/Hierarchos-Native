#!/usr/bin/env bash
# Phase XX preservation and identity capture.
#
# Same receipt method as Phase XII-XIX: for every directory, hash the
# `sha256sum` lines of its files in sorted path order, so one file edit is
# visible in one digest. Nothing here writes outside this receipt.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xx_query_scratchreuse_promotion_20261127
RAW=$PH/raw
mkdir -p "$RAW"

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "head_subject=$(git log -1 --format=%s)"
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
  echo "# status (short)"
  git status --short
} | tee "$RAW/00_identity.txt"

{
  echo "# production attention shader identities"
  sha256sum shaders/transformer_attention_backward_query_rdna3.comp \
            shaders/transformer_attention_backward_query_rdna3.spv \
            shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
            shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
            shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp \
            shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.spv
  echo
  echo "# sealed Phase XIX artifact identities (must not change)"
  sha256sum validation/phase_xix_query_scratch_reuse_20261127/shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp \
            validation/phase_xix_query_scratch_reuse_20261127/shaders/q_scratchreuse.spv \
            validation/phase_xix_query_scratch_reuse_20261127/shaders/q_control_rebuild.spv
  echo
  echo "# promoted artifact vs sealed candidate (byte comparison)"
  cmp shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp \
      validation/phase_xix_query_scratch_reuse_20261127/shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp \
    && echo "source_byte_identical=true"
  cmp shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.spv \
      validation/phase_xix_query_scratch_reuse_20261127/shaders/q_scratchreuse.spv \
    && echo "spirv_byte_identical=true"
  echo
  echo "# source identities"
  sha256sum src/transformer.rs \
            src/phase_xix_query_scratch_reuse_tests.rs \
            src/phase_xx_query_scratchreuse_promotion_tests.rs \
            src/phase_xviii_dq_accumulator_promotion_tests.rs \
            validation/training_shader_manifest.json
} | tee "$RAW/01_production_hashes.txt"

{
  echo "# sealed Phase XIX receipt digest"
  echo "# method: hash of the sha256sum lines of every file in sorted path order"
  d=validation/phase_xix_query_scratch_reuse_20261127
  n=$(find "$d" -type f | wc -l)
  h=$(find "$d" -type f | sort | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
  b=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  echo "$h  files=$n  bytes=$b  $d"
  echo "expected=078ef22eee85c82f34cbc0de62ee594d458464353fc7a2fb1cc84723af571392"
} | tee "$RAW/02_phase_xix_digest.txt"

echo "PHASE_XX_PRESERVATION_DONE"
