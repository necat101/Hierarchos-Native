#!/usr/bin/env bash
# Numerical equivalence for the two-pass attention backward.
#
# Runs the same LoRA steps twice: once on the qualified scalar backward and once
# with HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1. The adapter export is compared
# byte-for-byte, so any difference in dQ, dK or dV shows up as different weights
# after the first optimizer step rather than being averaged away.
set -u
CRATE='C:/Users/User/Downloads/Multimodal/Hierarchos-Native/hierarchos-vulkan'
cd "$CRATE" || exit 1
BIN=target/release/hierarchos-vulkan-transformer-train.exe
RUN="${TEMP:-/tmp}/hb/two-pass-$(date +%s)"
mkdir -p "$RUN"
echo "RUN=$RUN"

run() { # label steps
  local label="$1" steps="$2"
  local out="$RUN/adapter-$label"
  HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS="${TP:-0}" \
  HIERARCHOS_VULKAN_PROFILE_KERNELS="${PROFILE:-0}" \
    "$BIN" --model target/wmma-fixture/model \
      --dataset target/wmma-fixture/dataset.jsonl \
      --output "$out" \
      --batch-size 8 --seq-len 64 --epochs 4 --max-steps "$steps" \
      --learning-rate 1e-2 --seed 7 --log-steps 1000 \
      --lora-rank 8 --lora-alpha 16 \
      > "$RUN/train-$label.txt" 2>&1
  echo "$label $steps $?" | tee -a "$RUN/results.txt"
}

TP=0 PROFILE=0 run scalar 2
TP=1 PROFILE=1 run twopass 2

echo "=== adapter digests ==="
( cd "$RUN" && find adapter-scalar adapter-twopass -type f | sort | while read -r f; do
    printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "${f#adapter-}"
  done ) | tee "$RUN/digests.txt"

echo "=== dispatch evidence ==="
grep -o "shader=transformer_attention_backward_[a-z_]*rdna3" "$RUN/train-twopass.txt" | sort | uniq -c
grep -o "shader=transformer_attention_backward " "$RUN/train-twopass.txt" | sort | uniq -c
echo "=== DONE_TWO_PASS RUN=$RUN"
