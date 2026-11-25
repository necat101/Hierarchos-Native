#!/usr/bin/env bash
# Phase XIII — Mixed WMMA requalification on the optimized production graph.
#
# Measurement only. Rebuilds nothing that belongs to the product, edits no
# source, shader, threshold, default or precision policy, and writes every
# artifact into raw/ next to this script. Two ABBA series plus a third for
# full-parameter training, all with separate release-mode processes.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RAW=validation/phase_xiii_mixed_wmma_requalification_20261125/raw
FIXTURE=target/gqa-e2e/fixture-8q2kv
TEST_BIN=target/release/deps/hierarchos_vulkan-70a92090d59c92cf.exe
STEP_BIN=target/release/training-stability.exe
ROUNDS=${PHASE_XIII_ROUNDS:-5}
mkdir -p "$RAW"

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

step "1. repository identity (must still be the Phase III-XII base)"
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "staged_index_tree=$(git write-tree)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "binary_only_diff_sha256=$(git diff --binary -- '*.spv' | sha256sum | cut -d' ' -f1)"
  echo "tracked_text_diff_sha256=$(git diff --text -- . ':(exclude)*.spv' | sha256sum | cut -d' ' -f1)"
  echo "stash0=$(git rev-parse stash@{0})"
  echo
  echo "# tracked files changed by this phase (expected: none)"
  git diff --stat
} | tee "$RAW/00_identity.txt"
[ "$(git rev-parse HEAD)" = "560809eead186f832344c15d1a9b70795da3b75d" ] || fail "HEAD moved"
[ "$(git write-tree)" = "4709d4508ca02c854741617b379795929f673760" ] || fail "staged index moved"
[ "$(git rev-parse stash@{0})" = "831f892774619603c2dd68d0dd654d315f1a8f7e" ] || fail "stash@{0} moved"
[ "$(git diff --binary | sha256sum | cut -d' ' -f1)" \
  = "3a671c752ccf8bb7ad714781fae3f11e6264cd6085001da9caf9896c7244853f" ] || fail "working tree changed"

step "2. WMMA source and shader identities"
sha256sum shaders/wmma_gemm_f16_f32acc.comp shaders/wmma_gemm_f16_f32acc.spv \
  shaders/wmma_gemm_f16_f16acc.spv shaders/gemm_f16_f32_plain.spv \
  shaders/fp32_to_fp16_native.spv shaders/linear_forward.spv \
  src/wmma.rs src/wmma_linear.rs src/projection.rs src/transformer.rs \
  src/precision_policy.rs | tee "$RAW/01_wmma_source_and_shader_hashes.txt"

step "3. prior receipts and production attention artifacts"
{
  for d in validation/mha_attention_backward_closeout_20261124 \
           validation/gqa_attention_backward_experiment_20261124 \
           validation/gqa_production_promotion_20261124 \
           validation/gqa_end_to_end_training_20261124 \
           validation/phase_viii_key_blocks \
           validation/phase_ix_row_packed_forward_20261124 \
           validation/phase_x_row_packed_forward_promotion_20261125 \
           validation/phase_xi_channel_tiling_20261125 \
           validation/phase_xii_channel4_key_production_20261125; do
    n=$(find "$d" -type f | wc -l)
    h=$(find "$d" -type f | sort | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
    b=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    echo "$h  files=$n  bytes=$b  $d"
  done
  echo
  sha256sum shaders/transformer_attention_backward_key_rdna3.comp \
            shaders/transformer_attention_backward_key_rdna3.spv \
            shaders/transformer_attention_backward_query_rdna3.comp \
            shaders/transformer_attention_backward_query_rdna3.spv \
            shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp \
            shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv \
            shaders/transformer_attention_backward_key_channel4_rdna3.comp \
            shaders/transformer_attention_backward_key_channel4_rdna3.spv
} | tee "$RAW/02_preservation.txt"

step "4. release build of the harness binaries"
cargo build --release --lib --tests 2>&1 | tail -3
cargo build --release --bin training-stability --bin wmma_probe --bin wmma_qualify --bin wmma_bench \
  2>&1 | tail -3 || fail "harness binaries did not build"
sha256sum "$TEST_BIN" "$STEP_BIN" target/release/wmma_probe.exe \
  target/release/wmma_qualify.exe target/release/wmma_bench.exe \
  | tee "$RAW/08_executable_hashes.txt"

step "5. device capability probes"
./target/release/wmma_probe.exe --wmma-mode off --json > "$RAW/03_wmma_probe_off.json"
./target/release/wmma_probe.exe --wmma-mode mixed --json > "$RAW/04_wmma_probe_mixed.json"
grep -q '"mixed_supported": true' "$RAW/04_wmma_probe_mixed.json" || fail "Mixed is not supported by this device"

step "6. the unmodified 37-gate Mixed qualification"
./target/release/wmma_qualify.exe --wmma-mode mixed --json > "$RAW/05_wmma_qualify_mixed.json" \
  2> "$RAW/05_wmma_qualify_mixed.stderr.txt" || fail "a Mixed qualification gate failed"
grep -q '^status: PASS' "$RAW/05_wmma_qualify_mixed.json" || fail "Mixed qualification did not report PASS"

step "7. graph-path three-phase WMMA test"
"$TEST_BIN" wmma_linear --nocapture --test-threads=1 > "$RAW/06_wmma_linear_graph_tests.log" 2>&1 \
  || fail "the graph-path WMMA test failed"

step "8. dispatch proof and per-projection route table (LoRA + full parameter)"
HIERARCHOS_WMMA_MODE=off "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
  --output "$RAW/tmp/off-lora" --batch 2 --seq 64 --steps 3 --warmup 1 --lora-rank 8 --lora-alpha 16 \
  --lr 1e-2 --seed 7 --report "$RAW/10_stability_lora_off.json" > "$RAW/10_stability_lora_off.log" 2>&1 \
  || fail "the off dispatch proof failed"
HIERARCHOS_WMMA_MODE=mixed "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
  --output "$RAW/tmp/mixed-lora" --batch 2 --seq 64 --steps 3 --warmup 1 --lora-rank 8 --lora-alpha 16 \
  --lr 1e-2 --seed 7 --report "$RAW/11_stability_lora_mixed.json" > "$RAW/11_stability_lora_mixed.log" 2>&1 \
  || fail "the mixed dispatch proof failed"
HIERARCHOS_WMMA_MODE=off "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
  --output "$RAW/tmp/full-off" --batch 2 --seq 64 --steps 9 --warmup 3 --lr 1e-2 --seed 7 \
  --digest-every 1 --log-every 1 --no-export --report "$RAW/40_full_off_9.json" \
  > "$RAW/40_full_off_9.log" 2>&1 || fail "the off full-parameter run failed"
HIERARCHOS_WMMA_MODE=mixed "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
  --output "$RAW/tmp/full-mixed" --batch 2 --seq 64 --steps 9 --warmup 3 --lr 1e-2 --seed 7 \
  --digest-every 1 --log-every 1 --no-export --report "$RAW/41_full_mixed_9.json" \
  > "$RAW/41_full_mixed_9.log" 2>&1 || fail "the mixed full-parameter run failed"
python validation/phase_xiii_mixed_wmma_requalification_20261125/projection_table.py \
  > "$RAW/07_projection_table.json" || fail "predicted routes disagree with the recorded counters"

step "9. LoRA numerics, determinism and exact resume"
lora() { # tag mode steps warmup extra...
  local tag=$1 mode=$2 steps=$3 warmup=$4; shift 4
  HIERARCHOS_WMMA_MODE="$mode" "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
    --batch 2 --seq 64 --lora-rank 8 --lora-alpha 16 --lr 1e-2 --seed 7 --digest-every 1 --log-every 1 \
    --steps "$steps" --warmup "$warmup" --output "$RAW/tmp/$tag" --report "$RAW/$tag.json" \
    > "$RAW/$tag.log" 2>&1 || fail "$tag failed"
}
rm -rf "$RAW/tmp/state-mixed-a"
lora 30_lora_off_29 off 29 3
lora 31_lora_mixed_29a mixed 29 3
lora 32_lora_mixed_29b mixed 29 3
lora 33_lora_mixed_part_a mixed 6 0 --save-state "$RAW/tmp/state-mixed-a"
lora 34_lora_mixed_resume mixed 6 0 --start-step 6 --resume-state "$RAW/tmp/state-mixed-a"
lora 35_lora_mixed_cont12 mixed 12 0
lora 36_lora_off_export off 29 3 --adapter-out "$RAW/tmp/36_lora_off_export-adapter" \
  --save-state "$RAW/tmp/36_lora_off_export-state"
lora 37_lora_mixed_export mixed 29 3 --adapter-out "$RAW/tmp/37_lora_mixed_export-adapter" \
  --save-state "$RAW/tmp/37_lora_mixed_export-state"

step "10. full-parameter exports and the stable-learning-rate control"
full() { # tag mode lr extra...
  local tag=$1 mode=$2 lr=$3; shift 3
  HIERARCHOS_WMMA_MODE="$mode" "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
    --batch 2 --seq 64 --lr "$lr" --seed 7 --digest-every 1 --log-every 1 --steps 9 --warmup 3 \
    --output "$RAW/tmp/$tag" --report "$RAW/$tag.json" > "$RAW/$tag.log" 2>&1 || fail "$tag failed"
}
full 42_full_off_export off 1e-2
full 43_full_mixed_export mixed 1e-2
full 44_full_off_lr1e4_export off 1e-4
full 45_full_mixed_lr1e4_export mixed 1e-4
python validation/phase_xiii_mixed_wmma_requalification_20261125/numerics.py \
  > "$RAW/60_numerics.json" || fail "numerics.py failed"

step "11. profiled attribution (9 steps per arm)"
profiled() { # mode log
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_WMMA_MODE="$1" HIERARCHOS_PHASE_XII_STEP_ARM=channel4 \
  HIERARCHOS_PHASE_XII_STEP_STEPS=9 \
    "$TEST_BIN" phase_xii_channel4_key_full_step --nocapture --test-threads=1 > "$2" 2>&1 \
    || fail "profiled $1 run failed"
}
profiled off "$RAW/50_profile_off.log"
profiled mixed "$RAW/51_profile_mixed.log"
python validation/phase_xiii_mixed_wmma_requalification_20261125/attribute.py \
  "$RAW/50_profile_off.log" "$RAW/51_profile_mixed.log" 9 > "$RAW/52_attribution.json" \
  || fail "attribution failed"

step "12. isolated GEMM ratios at the production shapes"
SHAPES=128x1024x1024,128x256x1024,128x1024x256,1024x1024x128
./target/release/wmma_bench.exe --wmma-mode off --shapes "$SHAPES" --rounds 3 --iterations 10 \
  --json > "$RAW/09_gemm_bench_off.json" 2> "$RAW/09_gemm_bench_off.err.txt" || fail "fp32 GEMM bench failed"
./target/release/wmma_bench.exe --wmma-mode mixed --shapes "$SHAPES" --rounds 3 --iterations 10 \
  --json > "$RAW/09_gemm_bench_mixed.json" 2> "$RAW/09_gemm_bench_mixed.err.txt" || fail "mixed GEMM bench failed"

step "13. ABBA series (LoRA Phase XII harness, LoRA counter harness, full parameter)"
PHASE_XIII_ROUNDS="$ROUNDS" PHASE_XIII_TEST_BIN="$TEST_BIN" \
  bash validation/phase_xiii_mixed_wmma_requalification_20261125/run_abba_phase_xii_harness.sh \
  || fail "an ABBA series failed"
for round in $(seq 0 $((ROUNDS - 1))); do
  for position in 0 1 2 3; do
    case "$position" in 0|3) tag=A; mode=off ;; *) tag=B; mode=mixed ;; esac
    HIERARCHOS_WMMA_MODE="$mode" "$STEP_BIN" --model "$FIXTURE/model" --dataset "$FIXTURE/dataset.jsonl" \
      --batch 2 --seq 64 --lr 1e-2 --seed 7 --digest-every 0 --log-every 1 --steps 29 --warmup 3 \
      --no-export --output "$RAW/tmp/full-$tag" --report "$RAW/tmp/full-r$round-p$position-$tag.json" \
      > "$RAW/full-r$round-p$position-$tag.log" 2>&1 || fail "full ABBA r$round p$position"
  done
done

step "14. reductions and summary"
python validation/phase_xiii_mixed_wmma_requalification_20261125/reduce.py "$RAW" "$ROUNDS" \
  > "$RAW/20_abba.json" || fail "reduce.py failed"
sha256sum target/gqa-e2e/fixture-8q2kv/model/config.json \
          target/gqa-e2e/fixture-8q2kv/model/model.safetensors \
          target/gqa-e2e/fixture-8q2kv/dataset.jsonl \
  > "$RAW/10_fixture_and_checkpoint_hashes.txt"
{
  echo "phase_xiii_status=GO (LoRA production path; full-parameter numerics unqualified at lr 1e-2)"
  echo "head=$(git rev-parse HEAD)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  python - "$RAW" <<'PY'
import json,sys
raw=sys.argv[1]
d=json.load(open(f"{raw}/20_abba.json"))
for key in ("phase_xii_harness","stability_harness","full_parameter_harness"):
    s=d[key]
    print(f"{key}: median_ratio={s['median_round_ratio']} ci95={s['ratio_ci95']} "
          f"speedup={s['speedup']} rounds_improved={s['rounds_improved']}/{s['rounds_total']} verdict={s['verdict']}")
PY
} | tee "$RAW/99_summary.txt"
