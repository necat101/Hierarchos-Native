#!/usr/bin/env bash
# Re-run the Phase III MHA closeout measurements into a fresh directory.
#
# The bound receipt in raw/ is not overwritten; pass an output directory.
#   bash validation/mha_attention_backward_closeout_20261124/reproduce.sh [OUT]
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
OUT="${1:-target/mha-closeout-reproduce}"
mkdir -p "$OUT"
BIN=target/release/hierarchos-vulkan-transformer-train.exe
[ -x "$BIN" ] || BIN=target/release/hierarchos-vulkan-transformer-train
Q=target/release/hierarchos-vulkan-training-qualification.exe
[ -x "$Q" ] || Q=target/release/hierarchos-vulkan-training-qualification
MODEL=target/wmma-fixture/model
DATASET=target/wmma-fixture/dataset.jsonl
P=target/phase2-attention-backward

echo "== build =="
cargo build --release --offline \
  --bin hierarchos-vulkan-transformer-train \
  --bin hierarchos-vulkan-training-qualification

echo "== mode-0/mode-9 gradient qualification =="
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 cargo test --offline --lib \
  transformer::tests::two_pass_attention_backward_matches_scalar_gradients \
  -- --nocapture --test-threads=1 > "$OUT/parity-mode0-mode9.log" 2>&1
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 cargo test --offline --lib \
  transformer::tests::gradient_report_rejects_a_one_ulp_difference \
  -- --nocapture --test-threads=1 > "$OUT/comparator-negative-control.log" 2>&1
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0 cargo test --offline --lib \
  transformer::tests::two_pass_attention_backward_matches_scalar_gradients \
  -- --nocapture --test-threads=1 > "$OUT/parity-skip.log" 2>&1

echo "== dispatch-gated equivalence =="
bash validation/two_pass_backward_equivalence.sh > "$OUT/equivalence.log" 2>&1

echo "== paired ABBA benchmark (Phase II protocol) =="
BENCH=target/phase2-attention-backward/bench.py
if [ ! -f "$BENCH" ]; then
  echo "missing $BENCH; the Phase II protocol script is preserved at" >&2
  echo "validation/mha_attention_backward_closeout_20261124/raw/phase2-bench.py" >&2
  exit 1
fi
python "$BENCH" --rounds 5 --steps 26 --warmup 3 \
  --kernel-steps 12 --phase all --work "$OUT/bench" > "$OUT/bench.log" 2>&1

echo "== scratch routing =="
$BIN --model "$MODEL" --dataset "$DATASET" --output "$OUT/scratch-eligible" \
  --batch-size 8 --seq-len 64 --epochs 1 --max-steps 2 --learning-rate 1e-2 \
  --seed 7 --log-steps 1000 --lora-rank 8 --lora-alpha 16 \
  > "$OUT/scratch-eligible.stdout" 2> "$OUT/scratch-eligible.log"
HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0 $BIN --model "$MODEL" --dataset "$DATASET" \
  --output "$OUT/scratch-optout" --batch-size 8 --seq-len 64 --epochs 1 --max-steps 2 \
  --learning-rate 1e-2 --seed 7 --log-steps 1000 --lora-rank 8 --lora-alpha 16 \
  > "$OUT/scratch-optout.stdout" 2> "$OUT/scratch-optout.log"
$BIN --model "$P/fixture-gqa/model" --dataset "$P/fixture-gqa/dataset.jsonl" \
  --output "$OUT/scratch-gqa" --batch-size 2 --seq-len 8 --epochs 1 --max-steps 1 \
  --learning-rate 1e-2 --seed 7 --log-steps 1000 --lora-rank 8 --lora-alpha 16 \
  > "$OUT/scratch-gqa.stdout" 2> "$OUT/scratch-gqa.log"
$BIN --model "$P/fixture-seq-gt-headdim/model" \
  --dataset "$P/fixture-seq-gt-headdim/dataset.jsonl" --output "$OUT/scratch-seqgt" \
  --batch-size 2 --seq-len 64 --epochs 1 --max-steps 1 --learning-rate 1e-2 \
  --seed 7 --log-steps 1000 --lora-rank 8 --lora-alpha 16 \
  > "$OUT/scratch-seqgt.stdout" 2> "$OUT/scratch-seqgt.log"

echo "== sequence-16 cross-check =="
for arm in "0 scalar" "1 twopass"; do
  set -- $arm
  HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=$1 $Q --model "$MODEL" --dataset "$DATASET" --output "$OUT/seq16-$2" \
    --batch 8 --seq 16 --steps 3 --warmup 1 --lr 1e-2 --seed 7 \
    --log-every 1000 --digest-every 1 --lora-rank 8 --lora-alpha 16 \
    --adapter-out "$OUT/seq16-$2-adapter" --report "$OUT/seq16-$2.json" \
    --no-export > "$OUT/seq16-$2.stdout" 2> "$OUT/seq16-$2.stderr"
done

echo "DONE: $OUT"
echo "Compare the new logs with raw/; assemble.py intentionally reads only the bound raw/ set."
