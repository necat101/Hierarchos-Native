#!/usr/bin/env bash
# Serial paired ABBA runner for two `wmma_bench` builds.
#
# Both executables must be built from the same Rust sources; the intended
# difference is the compiled cooperative-matrix SPIR-V (or any other single
# compiled artifact). Runs are strictly serial - one process at a time, no other
# GPU work in flight - so no run can steal memory bandwidth from another. Even
# pairs run the baseline first, odd pairs run the candidate first.
#
# Usage:
#   WMMA_EVIDENCE_DIR=target/wmma-phase-lx-evidence \
#     BASE=/path/wmma_bench_baseline.exe \
#     CAND=/path/wmma_bench_candidate.exe \
#     OUT_DIR=target/wmma-phase-lx-evidence/abba_mixed \
#     bash validation/wmma_abba_run.sh
#
# Then:
#   python validation/wmma_abba_analyze.py <OUT_DIR>
set -u
cd "$(dirname "$0")/.." || exit 1

EV=${WMMA_EVIDENCE_DIR:-target/wmma-phase-lx-evidence}
BASE=${BASE:-$EV/baseline/wmma_bench_baseline.exe}
CAND=${CAND:-$EV/baseline/wmma_bench_candidate.exe}
MODE=${MODE:-mixed}
PAIRS=${PAIRS:-8}
ROUNDS=${ROUNDS:-4}
ITERS=${ITERS:-20}
OUT=${OUT_DIR:-$EV/abba_$MODE}

for exe in "$BASE" "$CAND"; do
  if [ ! -x "$exe" ]; then echo "missing executable: $exe" >&2; exit 1; fi
done

mkdir -p "$OUT"
printf 'pair\tarm\trc\texe_sha256\tfile\n' > "$OUT/index.tsv"

for p in $(seq 1 "$PAIRS"); do
  if [ $((p % 2)) -eq 1 ]; then order="base cand"; else order="cand base"; fi
  for arm in $order; do
    if [ "$arm" = base ]; then exe=$BASE; else exe=$CAND; fi
    name="$OUT/p${p}_${arm}"
    "$exe" --wmma-mode "$MODE" --rounds "$ROUNDS" --iterations "$ITERS" \
      --json --dump-samples "${name}_samples.jsonl" \
      > "${name}_report.json" 2> "${name}_err.txt"
    rc=$?
    sha=$(sha256sum "$exe" | cut -d' ' -f1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$p" "$arm" "$rc" "$sha" "$name" "$exe" >> "$OUT/index.tsv"
    if [ "$rc" -ne 0 ]; then echo "RUN FAILED pair=$p arm=$arm rc=$rc" >&2; fi
  done
  echo "pair $p done"
done
echo "ABBA COMPLETE: $OUT"
