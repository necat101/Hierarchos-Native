#!/usr/bin/env bash
# Phase XXI post-promotion component profile, FP32 and Mixed.
#
# Production baseline: neither HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL nor
# HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE is set, so the Phase XX
# default must select the scratch-reuse module. The driver is the established
# complete-step harness (`phase_xii_channel4_key_full_step`), which pins no query
# switch at all: warmup 3, 29 steps, digests off.
#
# Mixed WMMA is still opt-in: only the `mixed` processes set
# HIERARCHOS_WMMA_MODE=mixed. The production default stays off.
#
# Profiling perturbs execution at stride 1, so nothing in the `*_profile_*` logs
# is a performance claim. The `*_plain_*` processes are the short unprofiled
# sanity measurement and are kept separate.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xxi_postpromotion_profile_20261128
: "${PHASE_XXI_TEST_BIN:?set PHASE_XXI_TEST_BIN to the release deps test binary}"
BIN=$PHASE_XXI_TEST_BIN
STEPS=${PHASE_XXI_STEPS:-29}
ROUNDS=${PHASE_XXI_ROUNDS:-3}

unset HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL
unset HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE

run_mode() {
  local mode=$1
  local wmma=off
  [ "$mode" = mixed ] && wmma=mixed
  for round in $(seq 0 $((ROUNDS - 1))); do
    HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
    HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
    HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=4096 \
    HIERARCHOS_WMMA_MODE="$wmma" \
    HIERARCHOS_PHASE_XII_STEP_ARM=channel4 \
    HIERARCHOS_PHASE_XII_STEP_STEPS="$STEPS" \
    HIERARCHOS_PHASE_XII_STEP_DIGEST=0 \
      "$BIN" phase_xii_channel4_key_full_step --nocapture --test-threads=1 \
      > "$PH/raw/profile_${mode}_r${round}.log" 2>&1
    echo "mode=$mode round=$round profiled exit=$?"
  done
  HIERARCHOS_WMMA_MODE="$wmma" \
  HIERARCHOS_PHASE_XII_STEP_ARM=channel4 \
  HIERARCHOS_PHASE_XII_STEP_STEPS="$STEPS" \
  HIERARCHOS_PHASE_XII_STEP_DIGEST=0 \
    "$BIN" phase_xii_channel4_key_full_step --nocapture --test-threads=1 \
    > "$PH/raw/plain_${mode}.log" 2>&1
  echo "mode=$mode unprofiled exit=$?"
}

run_mode fp32
run_mode mixed

python "$PH/reduce_profile.py" \
  fp32 "$PH"/raw/profile_fp32_r*.log \
  mixed "$PH"/raw/profile_mixed_r*.log \
  > "$PH/raw/attribution.json"
echo "PHASE_XXI_PROFILE_DONE"
