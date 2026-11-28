#!/usr/bin/env bash
# Phase XXI identity capture: repository state, device/driver/Vulkan, compiler and
# build identity, and the exact environment the production profile ran under.
#
# Writes into this receipt only; never touches the Phase XIX or Phase XX receipts.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xxi_postpromotion_profile_20261128
RAW=$PH/raw
mkdir -p "$RAW"

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "head_subject=$(git log -1 --format=%s)"
  echo "head_parent=$(git rev-parse HEAD^)"
  echo "main=$(git rev-parse main)"
  echo "origin_main=$(git rev-parse origin/main)"
  echo "tag_v0.2.7_commit=$(git rev-parse 'v0.2.7^{commit}')"
  echo "stash_count=$(git stash list | wc -l)"
  git stash list
  echo
  echo "# status (short)"
  git status --short
} | tee "$RAW/00_identity.txt"

{
  echo "# vulkaninfo --summary"
  vulkaninfo --summary 2>&1 | grep -E "apiVersion|driverVersion|driverInfo|deviceName|deviceType|apiVersion|conformanceVersion|GPU[0-9]|^\s+vendorID|deviceID" || true
} | tee "$RAW/01_device.txt"

{
  echo "# compiler and build identity"
  rustc -Vv
  cargo -Vv
  echo
  echo "# release test binary"
  : "${PHASE_XXI_TEST_BIN:?set PHASE_XXI_TEST_BIN}"
  sha256sum "$PHASE_XXI_TEST_BIN"
  ls -l "$PHASE_XXI_TEST_BIN"
  echo
  echo "# production shader identities"
  sha256sum shaders/transformer_attention_backward_query_rdna3.comp \
            shaders/transformer_attention_backward_query_rdna3.spv \
            shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
            shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
            shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp \
            shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.spv
  echo
  echo "# sealed Phase XIX receipt digest (must be unchanged)"
  d=validation/phase_xix_query_scratch_reuse_20261127
  n=$(find "$d" -type f | wc -l)
  h=$(find "$d" -type f | sort | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
  b=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  echo "$h  files=$n  bytes=$b  $d"
  echo
  echo "# environment at capture"
  for v in HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL \
           HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE \
           HIERARCHOS_WMMA_MODE \
           HIERARCHOS_VULKAN_PROFILE_KERNELS \
           HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE; do
    if [ -n "${!v+x}" ]; then echo "$v=${!v}"; else echo "$v=<unset>"; fi
  done
} | tee "$RAW/02_environment.txt"

echo "PHASE_XXI_IDENTITY_DONE"
