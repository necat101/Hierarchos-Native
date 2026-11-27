#!/usr/bin/env bash
# Phase XVIII profiler attribution.
#
# Confirms the promoted dispatch is attributed to its registered identity
# (`transformer_attention_backward_query_rdna3_dqunroll`) instead of `unknown`.
# Profiling perturbs execution, so nothing here is a performance claim: the
# unprofiled ABBA in run_abba_step.sh carries those.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xviii_v028_dq_accumulator_promotion_20261126
: "${PHASE_XVIII_TEST_BIN:?set PHASE_XVIII_TEST_BIN to the release deps test binary}"
BIN=$PHASE_XVIII_TEST_BIN

for arm in A B; do
  if [ "$arm" = B ]; then
    unset HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL
    route=dqunroll
  else
    export HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0
    route=original
  fi
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=4096 \
  HIERARCHOS_WMMA_MODE=off \
  HIERARCHOS_PHASE_XVIII_STEP_ARM="$([ "$arm" = B ] && echo dqunroll || echo original)" \
  HIERARCHOS_PHASE_XVIII_STEP_STEPS=9 \
  HIERARCHOS_PHASE_XVIII_STEP_DIGEST=0 \
  HIERARCHOS_PHASE_XVIII_EXPECT_ROUTE="$route" \
    "$BIN" phase_xviii_dq_accumulator_full_step --nocapture --test-threads=1 \
    > "$PH/raw/32_profile_$arm.log" 2>&1
  echo "arm=$arm route=$route exit=$?"
done
unset HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL

{
  grep -h "kernel_profile_shader" "$PH/raw/32_profile_A.log" | sort -u
  echo "---"
  grep -h "kernel_profile_shader" "$PH/raw/32_profile_B.log" | sort -u
  echo "---"
  echo "unknown_shader_lines_A=$(grep -c 'shader=unknown' "$PH/raw/32_profile_A.log")"
  echo "unknown_shader_lines_B=$(grep -c 'shader=unknown' "$PH/raw/32_profile_B.log")"
  echo "dqunroll_lines_A=$(grep -c 'shader=transformer_attention_backward_query_rdna3_dqunroll' "$PH/raw/32_profile_A.log")"
  echo "dqunroll_lines_B=$(grep -c 'shader=transformer_attention_backward_query_rdna3_dqunroll' "$PH/raw/32_profile_B.log")"
} | tee "$PH/raw/32_attribution.txt"
