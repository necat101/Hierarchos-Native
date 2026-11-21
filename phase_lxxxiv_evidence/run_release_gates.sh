#!/usr/bin/env bash
# Phase LXXXIV release-gate runner: the exact commands the checked-in
# workflows run, executed in order with an explicit exit status per command.
# Any non-zero status is recorded and preserved; nothing is skipped silently.
set -uo pipefail
cd "$(dirname "$0")/../hierarchos-vulkan/validation" || exit 1

LOG="../../phase_lxxxiv_evidence/release_gates.txt"
: > "$LOG"
fails=0

run() {
  local label="$1"; shift
  printf '################################################################################\n' >> "$LOG"
  printf '## %s\n' "$label" >> "$LOG"
  printf '## $ %s\n' "$*" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  printf '## EXIT %d\n' "$rc" >> "$LOG"
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); printf 'FAIL[%d] %s\n' "$rc" "$label"; else printf 'PASS      %s\n' "$label"; fi
  return 0
}

run "Phase LXXXI P3 release gate"            python check_p3_release.py
run "Phase LXV hardware/accuracy/performance" python check_phase_lxv.py
run "Phase LXXXI 612-entry manifest"          python check_phase_lxxxi_manifest.py --expect-manifest-sha256 a7979f4f448ab3afa1598155d0eb708e8d36e4bed23457fa5dd515fcd22bbff4
run "Strict latest-release evidence integrity" python check_evidence_integrity.py --latest-release --strict
run "Pre-WMMA FP32 and device admission"      python check_main_compatibility.py
run "RDNA3 multimodal receipts"               python check_rdna3_multimodal_receipts.py
run "RDNA3 GPT2 receipts (historical)"        python check_rdna3_gpt2_receipts.py --historical-only
run "Phase XIX-X instrumented receipt"        python check_phase19x_current.py
run "Phase XIX-X attention receipt"           python check_attention_phase19x.py
run "Phase XIX-Y full-attention receipt"      python check_phase19y_full_attention.py
run "Phase XIX-Z current-source receipt"      python check_phase19z_current.py
run "Unified FP32 Mixed / scoped Full FP16"   python check_unified_release.py
run "Phase XIX-U historical audit"            python check_phase19u_audit.py
run "A/B workflow matches route spec"         python render_ab_workflow.py --check
run "Workflow path smoke"                     python -m unittest discover -s . -p "test_rdna3_workflow_paths.py"
run "Phase XIX-X current controls"            python -m unittest discover -s . -p "test_phase19x_current.py"
run "Mirror-epoch drift controls"             python -m unittest discover -s . -p "test_wmma_drift_epoch_reuse.py"
run "Phase XIX-X attention controls (CI)"     python -m unittest discover -s . -p "test_attention_phase19x_ci.py"
run "Phase XIX-Y controls"                    python -m unittest discover -s . -p "test_phase19y_full_attention.py"
run "Phase XIX-Z controls"                    python -m unittest discover -s . -p "test_phase19z_current.py"
run "Unified release fault injection"         python -m unittest discover -s . -p "test_unified_release.py"
run "Phase XIX-U controls"                    python -m unittest discover -s . -p "test_phase19u_audit.py"
run "RDNA3 GPT2 grade controls"               python -m unittest discover -s . -p "test_rdna3_gpt2_receipts.py"
run "Phase XVI qualification controls"        python -m unittest discover -s . -p "test_phase16_receipts.py"
run "Phase LXXI manifest negative controls"   python -m pytest test_phase_lxxxi_manifest.py -q
run "Evidence integrity negative controls"    python -m pytest test_evidence_integrity.py -q

printf '################################################################################\n' >> "$LOG"
printf '## TOTAL FAILING COMMANDS: %d\n' "$fails" >> "$LOG"
printf 'TOTAL FAILING COMMANDS: %d\n' "$fails"
