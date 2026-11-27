#!/usr/bin/env bash
# Phase XVII complete-step ABBA. Arm A = the shipped v0.2.7 pass-1 query
# artifact; arm B = the experiment's artifact (test-build substitution only).
# Same protocol as the established Phase XII harness: 3 warmup + 26 measured
# steps per process, five rounds of A B B A, separate processes.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
RAW=validation/phase_xvii_v028_optimization_discovery_20261126/raw
TEST_BIN=${PHASE_XVII_TEST_BIN:-target/release/deps/hierarchos_vulkan-70a92090d59c92cf.exe}
ROUNDS=${PHASE_XVII_ROUNDS:-5}
mkdir -p "$RAW"

for wmma in off mixed; do
  for round in $(seq 0 $((ROUNDS - 1))); do
    for position in 0 1 2 3; do
      case "$position" in
        0|3) tag=A ;;
        *) tag=B ;;
      esac
      if [ "$tag" = B ]; then
        export HIERARCHOS_PHASE_XVII_QUERY_SPV=dqunroll
      else
        unset HIERARCHOS_PHASE_XVII_QUERY_SPV
      fi
      HIERARCHOS_WMMA_MODE="$wmma" HIERARCHOS_PHASE_XII_STEP_ARM=channel4 \
      HIERARCHOS_PHASE_XII_STEP_STEPS=29 HIERARCHOS_PHASE_XII_STEP_DIGEST=0 \
        "$TEST_BIN" phase_xii_channel4_key_full_step --nocapture --test-threads=1 \
        > "$RAW/s7-$wmma-r$round-p$position-$tag.log" 2>&1
      echo "wmma=$wmma round=$round pos=$position tag=$tag exit=$?"
    done
  done
done
unset HIERARCHOS_PHASE_XVII_QUERY_SPV
echo "PHASE_XVII_ABBA_DONE"
