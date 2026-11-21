#!/usr/bin/env bash
# Phase LXXXI production-route smoke.
#
# Runs the release `wmma_qualify` binary with NO variant override and NO
# experimental flag, so the Full FP16 arm is whatever the production selector
# chooses. The release build's default is the committed P3 module.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BIN="$ROOT/../Hierarchos-Native-lxxxi-release/hierarchos-vulkan/target/release/wmma_qualify.exe"
mkdir -p "$HERE/dumps" "$HERE/receipts"
# `--dump-operands` refuses to overwrite, by design; clear only the dumps this
# script owns so a rerun is honest rather than silently reusing old tensors.
rm -f "$HERE"/dumps/dump_prod_p3_*.json

SEEDS=(42 1337 1593426484 20261119 271828 314159 777 8675309)

if [ ! -x "$BIN" ]; then echo "missing binary: $BIN" >&2; exit 1; fi
echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
echo "p3_spv_sha256=$(sha256sum "$ROOT/../Hierarchos-Native-lxxxi-release/hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.spv" | cut -d' ' -f1)"
echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for seed in "${SEEDS[@]}"; do
  echo "--- seed $seed"
  "$BIN" --wmma-mode full-fp16 --seed "$seed" --json \
      --dump-operands "$HERE/dumps/dump_prod_p3_$seed.json" \
      > "$HERE/receipts/prod_p3_$seed.json" 2> "$HERE/receipts/prod_p3_$seed.err"
  echo "prod_p3 exit=$?"
done

echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
