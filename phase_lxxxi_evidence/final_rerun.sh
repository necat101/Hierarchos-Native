#!/usr/bin/env bash
# Phase LXXXI final evidence rerun: production smoke + four 24-pair campaigns
# with the shipped release binary (comment-only source edits changed its hash).
set -uo pipefail
cd "$(dirname "$0")"
MAIN="../phase_lxxiii_evidence/bin/wmma_bench_main.exe"
TW="../phase_lxxiii_evidence/bin/wmma_bench_twowave.exe"
P3="../../Hierarchos-Native-lxxxi-release/hierarchos-vulkan/target/release/wmma_bench.exe"

rm -rf abba_main_vs_p3_24p abba_twowave_vs_p3_24p \
       abba_main_vs_p3_balanced_24p abba_twowave_vs_p3_balanced_24p

echo "=== SMOKE skipped (passed with final binary at 11:11Z)"

for spec in "abba_main_vs_p3_24p:$MAIN::0" \
            "abba_twowave_vs_p3_24p:$TW:two-wave:0" \
            "abba_main_vs_p3_balanced_24p:$MAIN::1" \
            "abba_twowave_vs_p3_balanced_24p:$TW:two-wave:1"; do
  IFS=':' read -r name base variant balanced <<<"$spec"
  if [ "$balanced" = "1" ]; then
    echo "=== $name (balanced ABBA/BAAB) $(date -u +%FT%TZ)"
    python run_lxxxi_balanced.py "$name" "$base" "$variant" "$P3" "" 24 16
  else
    echo "=== $name (alternating ABBA) $(date -u +%FT%TZ)"
    python run_lxxxi_abba.py "$name" "$base" "$variant" "$P3" "" 24 16
  fi
  echo "${name}_exit=$?"
done

echo "=== ANALYSIS $(date -u +%FT%TZ)"
python analyze_lxxxi.py ../../Hierarchos-Native-lxxxi-release/hierarchos-vulkan/validation . \
  abba_main_vs_p3_24p abba_twowave_vs_p3_24p \
  abba_main_vs_p3_balanced_24p abba_twowave_vs_p3_balanced_24p \
  > analysis.log 2>&1
echo "analysis_exit=$?"
echo "=== DONE $(date -u +%FT%TZ)"
