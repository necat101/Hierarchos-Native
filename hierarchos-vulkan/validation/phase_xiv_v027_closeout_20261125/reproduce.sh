#!/usr/bin/env bash
# Phase XIV — reproduce the closeout evidence.
#
# Read-only with respect to production: it rebuilds nothing, edits no source,
# shader, threshold, default or precision policy, and writes only into raw/.
# Requires the Phase XIII release binaries, which are the same executables that
# produced the Phase XIII receipt (training-stability 374e6235…, lib test binary
# 125a6956…).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RAW=validation/phase_xiv_v027_closeout_20261125/raw
FIX=target/gqa-e2e/fixture-8q2kv
STEP_BIN=target/release/training-stability.exe
TEST_BIN=target/release/deps/hierarchos_vulkan-70a92090d59c92cf.exe
mkdir -p "$RAW/driftsteps"

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

step "1. identity and preservation (also re-verifies the prior receipts)"
bash validation/phase_xiv_v027_closeout_20261125/capture_identity.sh > /dev/null || fail "capture_identity"

step "2. default selection requests no cooperative feature chain"
./target/release/wmma_probe.exe --json > "$RAW/04_wmma_probe_default.json" || fail "probe"
grep -q '"mode": "off"' "$RAW/04_wmma_probe_default.json" || fail "default is not off"

step "3. the unmodified 37-gate Mixed qualification"
./target/release/wmma_qualify.exe --wmma-mode mixed --json > "$RAW/05_wmma_qualify_mixed.json" \
  2> "$RAW/05_wmma_qualify_mixed.stderr.txt" || fail "a Mixed gate failed"
grep -q '^status: PASS' "$RAW/05_wmma_qualify_mixed.json" || fail "Mixed qualify did not PASS"

step "4. graph-path three-phase WMMA tests"
"$TEST_BIN" wmma_linear --nocapture --test-threads=1 > "$RAW/06_wmma_linear_graph_tests.log" 2>&1 \
  || fail "the graph-path WMMA tests failed"

step "5. adapter drift vs step count (6, 12, 29) and dispatch counters"
for n in 6 12 29; do
  for mode in off mixed; do
    HIERARCHOS_WMMA_MODE="$mode" "$STEP_BIN" --model "$FIX/model" --dataset "$FIX/dataset.jsonl" \
      --batch 2 --seq 64 --lora-rank 8 --lora-alpha 16 --lr 1e-2 --seed 7 --steps "$n" --warmup 3 \
      --digest-every 0 --log-every 1 --output "$RAW/driftsteps/run-$mode-$n" \
      --adapter-out "$RAW/driftsteps/adapter-$mode-$n" --report "$RAW/driftsteps/report-$mode-$n.json" \
      > "$RAW/driftsteps/run-$mode-$n.log" 2>&1 || fail "drift $mode $n"
  done
done

step "6. reductions"
python validation/phase_xiv_v027_closeout_20261125/adapter_drift.py \
  > "$RAW/10_adapter_drift.json" || fail "adapter_drift.py failed"
python validation/phase_xiv_v027_closeout_20261125/summarize_drift.py \
  > "$RAW/11_adapter_drift_summary.txt" 2>&1 || fail "summarize_drift.py failed"
bash validation/phase_xiv_v027_closeout_20261125/final_identity.sh > /dev/null || fail "final identity"

step "7. summary"
{
  echo "phase_xiv_status=CONDITIONAL GO for v0.2.7"
  echo "adapter_disposition=CONDITIONAL (0.02 elementwise is not an adapter master-weight gate)"
  echo "mixed_policy=default off, explicit opt-in"
  echo "head=$(git rev-parse HEAD)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "stash0=$(git rev-parse stash@{0})"
} | tee "$RAW/99_summary.txt"
