#!/usr/bin/env bash
# Phase XIX Candidate A complete-step ABBA: promoted pass 1 (arm A) against the
# same promoted pass 1 with scratch-backed intermediate reuse (arm B).
#
# Protocol (unchanged from Phase XII/XIII/XVII/XVIII): three warmup steps, 26
# measured steps per process, five rounds of A B B A, one process per arm,
# unprofiled, no parameter readback, each geometry and mode reported on its own.
#
# Both arms run the *production* route: the promoted selector at its default, its
# device qualification, one pass-1 dispatch per layer reported as
# `backward_query_dqunroll`, and the promoted pass-2 key artifact. The only
# difference is the SPIR-V the promoted loader builds; arm B substitutes the
# experiment artifact through the `#[cfg(test)]` seam. The driver asserts the
# counters, so an arm that silently ran something else fails instead of timing it.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xix_query_scratch_reuse_20261127
RAW=$PH/raw
: "${PHASE_XIX_TEST_BIN:?set PHASE_XIX_TEST_BIN to the release deps test binary}"
TEST_BIN=$PHASE_XIX_TEST_BIN
ROUNDS=${PHASE_XIX_ROUNDS:-5}
STEPS=${PHASE_XIX_STEPS:-29}
FILTER=${PHASE_XIX_SERIES_FILTER:-}
mkdir -p "$RAW"

# name|fixture|seq|role
# `primary` is the representative 8q/2kv sequence-64 fixture the brief names;
# the other three are the regression series required by the predeclared gate.
# 8q/2kv at sequence 128 and 4q/2kv at sequence 128 are deliberately absent: the
# pre-existing grouped qualification admits sequence 64 only, so the production
# step never reaches either pass-1 shader there (Phase XVIII section 7.3).
SERIES=(
  "8q2kv-s64|target/gqa-e2e/fixture-8q2kv|64|primary"
  "8q8kv-s64|target/phase-xviii/fixture-8q8kv-s64|64|regression"
  "8q8kv-s128|target/phase-xviii/fixture-8q8kv-s128|128|regression"
  "4q2kv-s64|target/gqa-e2e/fixture-4q2kv|64|regression"
)

for wmma in off mixed; do
  for spec in "${SERIES[@]}"; do
    IFS='|' read -r name fixture seq role <<<"$spec"
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
      continue
    fi
    for round in $(seq 0 $((ROUNDS - 1))); do
      for position in 0 1 2 3; do
        case "$position" in
          0|3) tag=A; arm=promoted ;;
          *) tag=B; arm=scratch ;;
        esac
        HIERARCHOS_WMMA_MODE="$wmma" \
        HIERARCHOS_PHASE_XIX_STEP_ARM="$arm" \
        HIERARCHOS_PHASE_XIX_FIXTURE="$fixture" \
        HIERARCHOS_PHASE_XIX_STEP_SEQ="$seq" \
        HIERARCHOS_PHASE_XIX_STEP_BATCH=2 \
        HIERARCHOS_PHASE_XIX_STEP_STEPS="$STEPS" \
        HIERARCHOS_PHASE_XIX_STEP_DIGEST=0 \
          "$TEST_BIN" phase_xix_scratch_reuse_full_step --nocapture --test-threads=1 \
          > "$RAW/s-$wmma-$name-r$round-p$position-$tag.log" 2>&1
        echo "wmma=$wmma series=$name role=$role round=$round pos=$position tag=$tag arm=$arm exit=$?"
      done
    done
  done
done
echo "PHASE_XIX_ABBA_DONE"
