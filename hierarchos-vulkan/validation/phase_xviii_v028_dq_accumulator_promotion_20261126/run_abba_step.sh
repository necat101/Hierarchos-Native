#!/usr/bin/env bash
# Phase XVIII complete-step ABBA, per geometry and per precision mode.
#
# Arm A = the shipped qualified query shader
#          HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0
# Arm B = the promoted literal-index query pass (the production default)
#
# Protocol (Phase XII/XIII/XVII continuity): three warmup steps, 26 measured
# steps per process, five rounds of A B B A, one process per arm, unprofiled.
# Each geometry and mode is reported on its own; nothing is pooled.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xviii_v028_dq_accumulator_promotion_20261126
RAW=$PH/raw
: "${PHASE_XVIII_TEST_BIN:?set PHASE_XVIII_TEST_BIN to the release deps test binary}"
TEST_BIN=$PHASE_XVIII_TEST_BIN
ROUNDS=${PHASE_XVIII_ROUNDS:-5}
STEPS=${PHASE_XVIII_STEPS:-29}
# Optional single-series selector, so the slow excluded geometry can be measured
# in its own process batch without re-running the fast series.
FILTER=${PHASE_XVIII_SERIES_FILTER:-}
# Sham control: all four positions run the shipped query shader, so the true
# effect is exactly zero. It measures what the A B B A schedule itself produces
# and is reported beside the real series, never as a result of its own.
SHAM=${PHASE_XVIII_SHAM:-0}
mkdir -p "$RAW"

# name|fixture|seq|geometry admitted by the enclosing two-pass gate
# `admitted` means arm A must report query_original and arm B query_dqunroll;
# `excluded` means both arms must report the scalar kernel and neither query
# shader, because the pre-existing grouped-sequence qualification rejects the
# geometry before either pass-1 artifact is considered.
SERIES=(
  # Continuity control: the fixture Phase XIII and Phase XVII measured.
  "8q2kv-s64|target/gqa-e2e/fixture-8q2kv|64|admitted"
  # 8q/2kv at sequence 128 is NOT here on purpose: the pre-existing grouped
  # qualification admits sequence 64 only, so the production route is the
  # quadratic scalar kernel (~8.7 s/step) and the geometry can never reach
  # either pass-1 shader. It is not one of the brief's required geometries.
  "8q8kv-s64|target/phase-xviii/fixture-8q8kv-s64|64|admitted"
  "8q8kv-s128|target/phase-xviii/fixture-8q8kv-s128|128|admitted"
  "4q2kv-s64|target/gqa-e2e/fixture-4q2kv|64|admitted"
  # 4q/2kv at sequence 128 is outside the enclosing two-pass qualification (the
  # grouped gate admits sequence 64 only), so the production step never reaches
  # either query shader there. It is measured anyway, as the brief requires, and
  # is expected to be a null result with the scalar route.
  "4q2kv-s128|target/phase-xviii/fixture-4q2kv-s128|128|excluded"
)

for wmma in off mixed; do
  for spec in "${SERIES[@]}"; do
    IFS='|' read -r name fixture seq admission <<<"$spec"
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
      continue
    fi
    for round in $(seq 0 $((ROUNDS - 1))); do
      for position in 0 1 2 3; do
        case "$position" in
          0|3) tag=A ;;
          *) tag=B ;;
        esac
        if [ "$SHAM" = 1 ]; then
          export HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0
          arm=original
          route=original
          out_mode="sham-$wmma"
        elif [ "$tag" = B ]; then
          unset HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL
          arm=dqunroll
          route=dqunroll
          out_mode=$wmma
        else
          export HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0
          arm=original
          route=original
          out_mode=$wmma
        fi
        if [ "$admission" = excluded ]; then
          route=scalar
        fi
        HIERARCHOS_WMMA_MODE="$wmma" \
        HIERARCHOS_PHASE_XVIII_STEP_ARM="$arm" \
        HIERARCHOS_PHASE_XVIII_FIXTURE="$fixture" \
        HIERARCHOS_PHASE_XVIII_STEP_SEQ="$seq" \
        HIERARCHOS_PHASE_XVIII_STEP_BATCH=2 \
        HIERARCHOS_PHASE_XVIII_STEP_STEPS="$STEPS" \
        HIERARCHOS_PHASE_XVIII_STEP_DIGEST=0 \
        HIERARCHOS_PHASE_XVIII_EXPECT_ROUTE="$route" \
          "$TEST_BIN" phase_xviii_dq_accumulator_full_step --nocapture --test-threads=1 \
          > "$RAW/s-$out_mode-$name-r$round-p$position-$tag.log" 2>&1
        echo "wmma=$wmma series=$name round=$round pos=$position tag=$tag arm=$arm expect=$route exit=$?"
      done
    done
  done
done
unset HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL
echo "PHASE_XVIII_ABBA_DONE"
