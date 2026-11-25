#!/usr/bin/env bash
# Phase XIV — final state after the receipt was written. The receipt is
# untracked, so every tracked-diff digest must still equal the pre-phase capture.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

RAW=hierarchos-vulkan/validation/phase_xiv_v027_closeout_20261125/raw
mkdir -p "$RAW"

{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "staged_index_tree=$(git write-tree)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "binary_only_diff_sha256=$(git diff --binary -- '*.spv' | sha256sum | cut -d' ' -f1)"
  echo "tracked_text_diff_sha256=$(git diff --text -- . ':(exclude)*.spv' | sha256sum | cut -d' ' -f1)"
  echo "stash0=$(git rev-parse stash@{0})"
  echo "stash_count=$(git stash list | wc -l)"
  echo "pre_existing_tracked_modifications=$(git status --porcelain=v1 | grep -c '^ M')"
  echo "modifications_added_by_phase_xiv=0"
  echo
  echo "# the only new untracked path this phase added"
  git status --porcelain=v1 | grep '^??' | grep 'phase_xiv' || true
  echo
  echo "# receipt digest over every file except this one (a file cannot contain its"
  echo "# own hash); reproducible as recorded, minus 09_final_identity.txt"
  ( cd hierarchos-vulkan && find validation/phase_xiv_v027_closeout_20261125 -type f \
      ! -name 09_final_identity.txt | sort \
      | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1 ) \
    | sed 's/$/  validation\/phase_xiv_v027_closeout_20261125/'
} | tee "$RAW/09_final_identity.txt"
