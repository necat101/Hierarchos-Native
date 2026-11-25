#!/usr/bin/env bash
# Phase XIII — release-mode ABBA on the exact Phase XII step harness.
#
# Arm A: HIERARCHOS_WMMA_MODE=off     (FP32 projections, default)
# Arm B: HIERARCHOS_WMMA_MODE=mixed   (FP16 operands, FP32 accumulation)
#
# Everything else is held at the Phase XII production configuration: same
# fixture, seed, batch, sequence length, LoRA rank/alpha, learning rate, step
# count, production row-packed attention forward, production GQA two-pass query
# backward, production channel4 key backward (HIERARCHOS_PHASE_XII_STEP_ARM=
# channel4, HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4 unset in both arms).
#
# The test binary is the unmodified Phase XII harness, so the measured step is
# the same production graph Phase XII measured at 28.460 ms. Separate process
# per position; three warmup steps and 26 measured steps per process.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RAW=validation/phase_xiii_mixed_wmma_requalification_20261125/raw
BIN=${PHASE_XIII_TEST_BIN:-target/release/deps/hierarchos_vulkan-70a92090d59c92cf.exe}
STEP_BIN=target/release/training-stability.exe
FIXTURE=target/gqa-e2e/fixture-8q2kv
ROUNDS=${PHASE_XIII_ROUNDS:-5}
STEPS=${PHASE_XIII_STEPS:-29}
MEASURED=${PHASE_XIII_MEASURED:-26}
mkdir -p "$RAW"

run_phase_xii_arm() { # arm log
  local arm=$1 log=$2
  case "$arm" in
    A) local mode=off ;;
    B) local mode=mixed ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  HIERARCHOS_WMMA_MODE="$mode" \
  HIERARCHOS_PHASE_XII_STEP_ARM=channel4 \
  HIERARCHOS_PHASE_XII_STEP_STEPS="$STEPS" \
    "$BIN" phase_xii_channel4_key_full_step --nocapture --test-threads=1 \
    > "$log" 2>&1
}

run_stability_arm() { # arm log
  local arm=$1 log=$2
  case "$arm" in
    A) local mode=off ;;
    B) local mode=mixed ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  # --digest-every 0 keeps the per-step host readback out of the measured step;
  # every step still prints its own WMMA dispatch deltas.
  HIERARCHOS_WMMA_MODE="$mode" \
    "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
    --output "$RAW/tmp/abba-$arm" --batch 2 --seq 64 --steps "$STEPS" --warmup 3 \
    --lora-rank 8 --lora-alpha 16 --lr 1e-2 --seed 7 --digest-every 0 --log-every 1 \
    > "$log" 2>&1
}

status=0
for round in $(seq 0 $((ROUNDS - 1))); do
  for position in 0 1 2 3; do
    case "$position" in 0|3) tag=A ;; *) tag=B ;; esac
    echo "phase-xii-harness round=$round position=$position arm=$tag"
    run_phase_xii_arm "$tag" "$RAW/abba-r${round}-p${position}-${tag}.log" \
      || { echo "FAILED r$round p$position $tag" >&2; status=1; }
  done
done

echo "stability-harness (counter-carrying) ABBA"
for round in $(seq 0 $((ROUNDS - 1))); do
  for position in 0 1 2 3; do
    case "$position" in 0|3) tag=A ;; *) tag=B ;; esac
    echo "stability round=$round position=$position arm=$tag"
    run_stability_arm "$tag" "$RAW/stab-r${round}-p${position}-${tag}.log" \
      || { echo "FAILED stability r$round p$position $tag" >&2; status=1; }
  done
done
exit "$status"
