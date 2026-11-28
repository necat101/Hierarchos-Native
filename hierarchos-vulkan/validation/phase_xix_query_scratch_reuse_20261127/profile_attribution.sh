#!/usr/bin/env bash
# Phase XIX fresh post-promotion component attribution, FP32 and Mixed.
#
# Both arms here are the *production promoted* path: HIERARCHOS_ATTENTION_
# BACKWARD_QUERY_DQUNROLL is left unset so the promoted dQ-unroll pass is what
# runs, exactly as the brief requires for the phase's profiling.
#
# Profiling perturbs execution at stride 1, so nothing in raw/3x_profile_*.log is
# a performance claim; the unprofiled ABBA in run_abba_step.sh carries those.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xix_query_scratch_reuse_20261127
: "${PHASE_XIX_TEST_BIN:?set PHASE_XIX_TEST_BIN to the release deps test binary}"
BIN=$PHASE_XIX_TEST_BIN
STEPS=${PHASE_XIX_PROFILE_STEPS:-9}

unset HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL
for wmma in off mixed; do
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=4096 \
  HIERARCHOS_WMMA_MODE="$wmma" \
  HIERARCHOS_PHASE_XIX_STEP_ARM=promoted \
  HIERARCHOS_PHASE_XIX_STEP_STEPS="$STEPS" \
  HIERARCHOS_PHASE_XIX_STEP_DIGEST=0 \
    "$BIN" phase_xix_scratch_reuse_full_step --nocapture --test-threads=1 \
    > "$PH/raw/32_profile_$wmma.log" 2>&1
  echo "wmma=$wmma route=promoted exit=$?"
done

python "$PH/attribute.py" \
  "$PH/raw/32_profile_off.log" fp32 \
  "$PH/raw/32_profile_mixed.log" mixed \
  > "$PH/raw/32_attribution.json"
echo "PHASE_XIX_PROFILE_DONE"
