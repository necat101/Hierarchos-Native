#!/usr/bin/env bash
# Phase XII — production promotion of the RDNA3 four-channel key pass.
#
# Rebuilds the promoted artifact from its exact source and proves it
# byte-identical to the Phase XI `c4_u0` SPIR-V, re-runs the production gates,
# then re-derives the release-mode equivalence, the five-round ABBA and the
# attribution. Every output lands in raw/ next to this script; nothing is
# committed, no other shader is rebuilt and no other selector is changed.
#
# The release build dominates the wall clock (~10-15 min cold). Sections 1-4
# need only a debug build and no release binary.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RAW=validation/phase_xii_channel4_key_production_20261125/raw
SRC=shaders/transformer_attention_backward_key_channel4_rdna3.comp
OUT=shaders/transformer_attention_backward_key_channel4_rdna3.spv
XI_SRC=validation/phase_xi_channel_tiling_20261125/shaders/transformer_attention_backward_key_chantile_rdna3.comp
XI_C4=validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c4_u0.spv
GLSLC=${GLSLC:-/c/VulkanSDK/1.4.321.1/Bin/glslc.EXE}
SPIRV_VAL=${SPIRV_VAL:-/c/VulkanSDK/1.4.321.1/Bin/spirv-val.EXE}
FIXTURE=target/gqa-e2e/fixture-8q2kv
# The Phase XI artifact this phase is authorised to promote.
C4_U0_SPV=f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8
mkdir -p "$RAW"

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

step "1. repository identity (must still be the Phase III-XI base)"
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "tracked_diff_sha256=$(git diff --text -- . ':(exclude)*.spv' | sha256sum | cut -d' ' -f1)"
  echo "binary_diff_sha256=$(git diff --binary -- '*.spv' | sha256sum | cut -d' ' -f1)"
  echo "full_diff_sha256=$(git diff --binary | sha256sum | cut -d' ' -f1)"
  echo "staged_index_tree=$(git write-tree)"
  echo "staged_diff_sha256=$(git diff --cached --binary | sha256sum | cut -d' ' -f1)"
  echo "stash0=$(git rev-parse stash@{0} 2>/dev/null)"
  git status --short
} | tee "$RAW/00_identity.txt"
[ "$(git rev-parse HEAD)" = "560809eead186f832344c15d1a9b70795da3b75d" ] \
  || fail "HEAD moved off the Phase III-XI base commit"
[ "$(git write-tree)" = "4709d4508ca02c854741617b379795929f673760" ] \
  || fail "the staged index moved"
[ "$(git rev-parse stash@{0})" = "831f892774619603c2dd68d0dd654d315f1a8f7e" ] \
  || fail "stash@{0} moved"

step "2. rebuild the promoted artifact and prove it is the Phase XI c4_u0"
# Rebuild into raw/ and leave the shipped artifact byte-for-byte in place: a
# rewrite would bump its mtime and make build.rs rebuild the whole crate.
REBUILT=$RAW/promoted.spv.rebuilt
"$GLSLC" -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
  -DXI_CHANNEL_CHUNK=4 -DXI_CHANNEL_BLOCKS=32 -DXI_UNROLL_CHANNEL=0 \
  "$SRC" -o "$REBUILT" 2> "$RAW/rebuild.spv.stderr.txt" || fail "glslc refused the promoted source"
"$SPIRV_VAL" --target-env vulkan1.2 "$REBUILT" || fail "spirv-val rejected the promoted artifact"
[ "$(sha256sum "$REBUILT" | cut -d' ' -f1)" = "$C4_U0_SPV" ] \
  || fail "the rebuilt SPIR-V is not the experimentally qualified c4_u0 artifact"
cmp -s "$REBUILT" "$XI_C4" || fail "byte comparison against key_chantile_c4_u0.spv failed"
cmp -s "$REBUILT" "$OUT" || fail "the shipped artifact is not the rebuilt artifact"
cmp -s "$SRC" "$XI_SRC" || fail "the promoted source is not the Phase XI source"
"$GLSLC" -E -fshader-stage=compute -Ishaders \
  -DXI_CHANNEL_CHUNK=4 -DXI_CHANNEL_BLOCKS=32 -DXI_UNROLL_CHANNEL=0 "$SRC" \
  | grep -v '^#line' > "$RAW/preprocessed-promoted.glsl"
"$GLSLC" -E -fshader-stage=compute -Ishaders \
  -DXI_CHANNEL_CHUNK=4 -DXI_CHANNEL_BLOCKS=32 -DXI_UNROLL_CHANNEL=0 "$XI_SRC" \
  | grep -v '^#line' > "$RAW/preprocessed-experimental.glsl"
cmp -s "$RAW/preprocessed-promoted.glsl" "$RAW/preprocessed-experimental.glsl" \
  || fail "the promoted and experimental preprocessed token streams differ"
{
  echo "# source"
  sha256sum "$SRC" "$OUT" "$XI_SRC" "$XI_C4"
  echo "spv_match=c4_u0"
  echo "preprocessed_sha256=$(sha256sum "$RAW/preprocessed-promoted.glsl" | cut -d' ' -f1)"
  echo "compiler=glslc"
  "$GLSLC" --version
  echo "validator=SPIRV-Tools"
  "$SPIRV_VAL" --version | head -1
  echo "flags=-fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders -DXI_CHANNEL_CHUNK=4 -DXI_CHANNEL_BLOCKS=32 -DXI_UNROLL_CHANNEL=0"
  echo "workgroup=64x1x1"
  echo "chunk=4 blocks=32"
} | tee "$RAW/02_shader-rebuild.txt"

step "3. prior receipts and production artifacts are unchanged"
{
  echo "# Prior receipt directory identities"
  for d in validation/mha_attention_backward_closeout_20261124 \
           validation/gqa_attention_backward_experiment_20261124 \
           validation/gqa_production_promotion_20261124 \
           validation/gqa_end_to_end_training_20261124 \
           validation/phase_viii_key_blocks \
           validation/phase_ix_row_packed_forward_20261124 \
           validation/phase_x_row_packed_forward_promotion_20261125 \
           validation/phase_xi_channel_tiling_20261125; do
    n=$(find "$d" -type f | wc -l)
    h=$(find "$d" -type f | sort | while IFS= read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1)
    b=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    echo "$h  files=$n  bytes=$b  $d"
  done
  echo
  echo "# Production attention shader identities"
  sha256sum shaders/transformer_attention_backward_key_rdna3.comp \
            shaders/transformer_attention_backward_key_rdna3.spv \
            shaders/transformer_attention_backward_query_rdna3.comp \
            shaders/transformer_attention_backward_query_rdna3.spv \
            shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp \
            shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv \
            "$SRC" "$OUT"
} | tee "$RAW/03_preservation.txt"
sha256sum -c validation/phase_xi_channel_tiling_20261125/raw/03_production-hashes.txt \
  | tee "$RAW/04_phase-xi-production-hashes.txt" \
  || fail "a Phase XI production shader identity moved"

step "4. production numerical gates and the fallback tests"
cargo test --lib phase_xii_channel4 -- --nocapture --test-threads=1 2>&1 \
  | tee "$RAW/10_phase-xii-gates.log" \
  || fail "a Phase XII gate failed"
cargo test --lib phase_xi_channel_tiling -- --nocapture --test-threads=1 2>&1 \
  | tee "$RAW/11_phase-xi-unchanged.log" \
  || fail "Phase XI evidence no longer reproduces"
cargo test --lib optional_two_pass -- --nocapture --test-threads=1 2>&1 \
  | tee "$RAW/12_two-pass-fallback.log" \
  || fail "the existing two-pass fallback changed"

step "5. release build"
cargo test --release --lib --no-run 2>&1 | tee "$RAW/20_release-build.log" \
  || fail "the release test binary did not build"

run_step() { # arm log extra-env...
  local arm=$1 log=$2; shift 2
  env "$@" HIERARCHOS_PHASE_XII_STEP_ARM="$arm" \
    cargo test --release --lib phase_xii_channel4_key_full_step -- --nocapture \
    > "$log" 2>&1 || fail "step run $arm failed ($log)"
}

step "6. release-mode training equivalence (29 steps, both arms)"
rm -rf "$RAW/checkpoint-original" "$RAW/checkpoint-channel4"
run_step original "$RAW/30_equivalence-original.log" \
  HIERARCHOS_PHASE_XII_STEP_DIGEST=1 \
  HIERARCHOS_PHASE_XII_STEP_CHECKPOINT=validation/phase_xii_channel4_key_production_20261125/raw/checkpoint-original \
  HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0
run_step channel4 "$RAW/31_equivalence-channel4.log" \
  HIERARCHOS_PHASE_XII_STEP_DIGEST=1 \
  HIERARCHOS_PHASE_XII_STEP_CHECKPOINT=validation/phase_xii_channel4_key_production_20261125/raw/checkpoint-channel4
grep -E '^\[phase-xii-(step|step-run-end|checkpoint)\]' "$RAW/30_equivalence-original.log" \
  | sed 's/arm=original/arm=ARM/' > "$RAW/30_equivalence-original.normalized"
grep -E '^\[phase-xii-(step|step-run-end|checkpoint)\]' "$RAW/31_equivalence-channel4.log" \
  | sed 's/arm=channel4/arm=ARM/' > "$RAW/31_equivalence-channel4.normalized"
diff -u "$RAW/30_equivalence-original.normalized" \
        "$RAW/31_equivalence-channel4.normalized" \
  > "$RAW/32_equivalence.diff" || fail "the two arms are not identical"

step "7. five-round ABBA (26 measured steps per process, separate processes)"
for round in 0 1 2 3 4; do
  for position in 0 1 2 3; do
    case "$position" in 0|3) tag=A; arm=original; extra=HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0 ;; *) tag=B; arm=channel4; extra=HIERARCHOS_PHASE_XII_NOOP=1 ;; esac
    log="$RAW/abba-r${round}-p${position}-${tag}.log"
    if [ "$tag" = A ]; then
      run_step "$arm" "$log" "$extra"
    else
      run_step "$arm" "$log"
    fi
  done
done
python validation/phase_xii_channel4_key_production_20261125/abba.py "$RAW" 5 original channel4 \
  | tee "$RAW/33_abba.json"

step "8. attribution (profiled, 9 steps per arm)"
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_PHASE_XII_STEP_STEPS=9 \
  run_step original "$RAW/40_profile-original.log" \
  HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0
HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  HIERARCHOS_PHASE_XII_STEP_STEPS=9 \
  run_step channel4 "$RAW/41_profile-channel4.log"
python validation/phase_xii_channel4_key_production_20261125/attribute.py \
  "$RAW/40_profile-original.log" "$RAW/41_profile-channel4.log" \
  | tee "$RAW/42_attribution.json"

step "9. summary"
{
  echo "phase_xii_status=GO"
  echo "promoted_spv=$C4_U0_SPV"
  echo "gates=$(grep -c 'verdict=PASS' "$RAW/10_phase-xii-gates.log") pass lines"
  echo "abba=$(python -c 'import json,sys;d=json.load(open(sys.argv[1]));print("median_ratio=%s speedup=%s verdict=%s"%(d["median_round_ratio"],d["speedup"],d["verdict"]))' "$RAW/33_abba.json")"
} | tee "$RAW/99_summary.txt"
