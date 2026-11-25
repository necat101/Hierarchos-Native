#!/usr/bin/env bash
# Phase VII reproduction: end-to-end GQA Transformer training benchmark.
#
# Reruns the whole receipt into $OUT (default: a new directory beside this
# script's raw/), leaving the committed receipt files untouched.  The timing
# phase takes about fifteen minutes and must run on a quiet machine: do not run
# anything else while it is in progress.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

OUT=${OUT:-validation/gqa_end_to_end_training_20261124/rerun}
mkdir -p "$OUT/raw" "$OUT/samples"

echo "== preservation evidence =="
{
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "head: $(git rev-parse HEAD)"
  echo "stash-commit: $(git rev-parse stash@{0})"
  echo "diff-sha256: $(git diff | sha256sum | cut -d' ' -f1)"
  echo "cached-diff-sha256: $(git diff --cached | sha256sum | cut -d' ' -f1)"
} | tee "$OUT/raw/git-identity.txt"
git status --short > "$OUT/raw/git-status.txt"
git diff --stat > "$OUT/raw/git-diffstat.txt"
git diff > "$OUT/raw/git-diff.patch"
sha256sum \
  shaders/transformer_attention_backward.comp \
  shaders/transformer_attention_backward.spv \
  shaders/transformer_attention_backward_query_rdna3.comp \
  shaders/transformer_attention_backward_query_rdna3.spv \
  shaders/transformer_attention_backward_key_rdna3.comp \
  shaders/transformer_attention_backward_key_rdna3.spv \
  > "$OUT/raw/shader-hashes.txt"

echo "== build =="
cargo build --release --offline \
  --bin hierarchos-vulkan-training-qualification \
  --bin hierarchos-vulkan-transformer-train

echo "== fixtures =="
python validation/make_wmma_fixture.py --output target/gqa-e2e/fixture-8q2kv \
  --hidden 1024 --layers 2 --vocab 1024 --heads 8 --kv-heads 2 --head-dim 128 \
  --seq-len 64 --batch-size 2 --sequences 32
python validation/make_wmma_fixture.py --output target/gqa-e2e/fixture-4q2kv \
  --hidden 512 --layers 2 --vocab 1024 --heads 4 --kv-heads 2 --head-dim 128 \
  --seq-len 64 --batch-size 2 --sequences 32
sha256sum \
  target/gqa-e2e/fixture-8q2kv/model/config.json \
  target/gqa-e2e/fixture-8q2kv/model/model.safetensors \
  target/gqa-e2e/fixture-8q2kv/dataset.jsonl \
  target/gqa-e2e/fixture-4q2kv/model/config.json \
  target/gqa-e2e/fixture-4q2kv/model/model.safetensors \
  target/gqa-e2e/fixture-4q2kv/dataset.jsonl \
  > "$OUT/raw/fixture-hashes.txt"

echo "== numerics, attribution, default selector, fail-closed =="
python validation/gqa_end_to_end_training_20261124/run_benchmark.py \
  --phase numerics --receipt "$OUT"
python validation/gqa_end_to_end_training_20261124/run_benchmark.py \
  --phase profile --receipt "$OUT"
python validation/gqa_end_to_end_training_20261124/run_benchmark.py \
  --phase default --receipt "$OUT"
python validation/gqa_end_to_end_training_20261124/run_benchmark.py \
  --phase failclosed --receipt "$OUT"

echo "== paired ABBA (requires a quiet machine) =="
python validation/gqa_end_to_end_training_20261124/run_benchmark.py \
  --phase abba --receipt "$OUT"

echo "== kernel-initialization fail-closed contract =="
HIERARCHOS_WMMA_MODE=off HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 \
  target/debug/deps/hierarchos_vulkan-*.exe \
  optional_two_pass_load_failure_keeps_scalar_execution_and_gradients \
  --nocapture --test-threads=1 > "$OUT/raw/failclosed-kernel.log" 2>&1

echo "== end-of-phase preservation =="
{
  echo "diff-sha256: $(git diff | sha256sum | cut -d' ' -f1)"
  echo "cached-diff-sha256: $(git diff --cached | sha256sum | cut -d' ' -f1)"
} | tee -a "$OUT/raw/git-identity.txt"

echo "wrote $OUT"
