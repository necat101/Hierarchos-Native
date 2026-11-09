#!/usr/bin/env bash
# Paired benchmark: scalar attention backward versus the two-pass pair.
#
# Two measurements, kept separate on purpose:
#   * per-kernel GPU time from the timestamp profiler, one 12-step run per arm;
#   * whole optimizer-step wall time from paired ABBA rounds, each step derived
#     as (t_long - t_short) / (long - short) so fixed startup cancels.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=target/release/hierarchos-vulkan-transformer-train
[ -x "$BIN" ] || BIN="$BIN.exe"
RUN="${TEMP:-/tmp}/hb/two-pass-bench-$(date +%s)"
mkdir -p "$RUN"
echo "RUN=$RUN"

one() { # label two_pass steps profile
  local label="$1" tp="$2" steps="$3" profile="$4"
  local out="$RUN/out-$label"
  HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS="$tp" \
  HIERARCHOS_VULKAN_PROFILE_KERNELS="$profile" \
  HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=16 \
    "$BIN" --model target/wmma-fixture/model \
      --dataset target/wmma-fixture/dataset.jsonl --output "$out" \
      --batch-size 8 --seq-len 64 --epochs 12 --max-steps "$steps" \
      --learning-rate 1e-2 --seed 7 --log-steps 1000 \
      --lora-rank 8 --lora-alpha 16 \
      > "$RUN/log-$label.txt" 2>&1
  rm -rf "$out"
}

# Kernel attribution, 12 profiled steps per arm.
one kernel-scalar 0 12 1
one kernel-twopass 1 12 1

# Whole-step wall time, paired ABBA.
for r in 1 2 3; do
  for arm in "scalar 0" "twopass 1" "twopass 1" "scalar 0"; do
    set -- $arm
    local_label="$r-$1-$2"
    t0=$(date +%s%N)
    one "$local_label" "$2" 26 0
    t1=$(date +%s%N)
    echo "$r $1 $(( (t1 - t0) / 1000000 ))" >> "$RUN/wall.txt"
  done
done
echo "=== DONE_BENCH RUN=$RUN"
