#!/usr/bin/env bash
# Phase XXII complete-step ABBA: the promoted 32x32 output-tiled kernel (B,
# default) against the original linear_input_grad kernel (A).
#
# HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0 is the production opt-out and the
# control arm; unset is the promoted default.
#
# Five rounds of A B B A, 20 independent release-mode processes, 3 warmup + 26
# measured steps, no profiler, no per-step digest. The reduction is the Phase XII
# gate (validation/phase_xii_channel4_key_production_20261125/abba.py).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xxii_linear_input_grad_tile_20261128
: "${PHASE_XXII_TEST_BIN:?set PHASE_XXII_TEST_BIN to the release deps test binary}"
BIN=$PHASE_XXII_TEST_BIN
ROUNDS=${PHASE_XXII_ROUNDS:-5}
STEPS=${PHASE_XXII_STEPS:-29}
mkdir -p "$PH/raw"

for round in $(seq 0 $((ROUNDS - 1))); do
  for position in 0 1 2 3; do
    if [ "$position" -eq 0 ] || [ "$position" -eq 3 ]; then
      arm=A
      tile_env=(HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0)
    else
      arm=B
      tile_env=()
    fi
    env "${tile_env[@]}" \
    HIERARCHOS_PHASE_XII_STEP_ARM=channel4 \
    HIERARCHOS_PHASE_XII_STEP_STEPS="$STEPS" \
    HIERARCHOS_PHASE_XII_STEP_DIGEST=0 \
    HIERARCHOS_WMMA_MODE=off \
      "$BIN" phase_xii_channel4_key_full_step --nocapture --test-threads=1 \
      > "$PH/raw/abba-r${round}-p${position}-${arm}.log" 2>&1
    status=$?
    echo "round=$round position=$position arm=$arm exit=$status"
    [ $status -eq 0 ] || exit $status
  done
done

python validation/phase_xii_channel4_key_production_20261125/abba.py \
  "$PH/raw" "$ROUNDS" linear_input_grad_tile_opt_out linear_input_grad_tile32_default \
  > "$PH/raw/abba.json"
python "$PH/check_abba.py" "$PH/raw" "$ROUNDS"
