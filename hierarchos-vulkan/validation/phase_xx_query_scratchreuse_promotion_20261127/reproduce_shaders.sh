#!/usr/bin/env bash
# Phase XX promoted-shader reproduction.
#
# Rebuilds the promoted artifact from its committed source with the manifest's
# exact flags into a temporary directory (never overwriting the committed
# artifact), validates it, then proves it is byte-identical to the sealed Phase
# XIX candidate binary that carries the numerical and performance evidence.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xx_query_scratchreuse_promotion_20261127
RAW=$PH/raw
STEM=transformer_attention_backward_query_rdna3_dqunroll_scratchreuse
FLAGS=(-fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders)
mkdir -p "$RAW"

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

{
  echo "$ glslc ${FLAGS[*]} shaders/$STEM.comp -o <scratch>/$STEM.spv"
  glslc "${FLAGS[@]}" "shaders/$STEM.comp" -o "$SCRATCH/$STEM.spv"
  echo "compile_exit=$?"
  glslc --version
} | tee "$RAW/10_compile.txt"

{
  sha256sum "shaders/$STEM.comp" "$SCRATCH/$STEM.spv" "shaders/$STEM.spv" \
            validation/phase_xix_query_scratch_reuse_20261127/shaders/q_scratchreuse.spv
  cmp "$SCRATCH/$STEM.spv" "shaders/$STEM.spv" \
    && echo "rebuilt_equals_committed=true"
  cmp "shaders/$STEM.spv" \
      validation/phase_xix_query_scratch_reuse_20261127/shaders/q_scratchreuse.spv \
    && echo "promoted_equals_phase_xix_candidate=true"
} | tee "$RAW/11_artifact_hashes.txt"

{
  spirv-val --target-env vulkan1.2 "$SCRATCH/$STEM.spv" && echo "rebuilt_valid=true"
  spirv-val --target-env vulkan1.2 "shaders/$STEM.spv" && echo "committed_valid=true"
  spirv-val --target-env vulkan1.2 shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
    && echo "phase_xviii_valid=true"
  spirv-val --target-env vulkan1.2 shaders/transformer_attention_backward_query_rdna3.spv \
    && echo "shipped_fallback_valid=true"
} | tee "$RAW/12_spirv_val.txt"

spirv-dis "$SCRATCH/$STEM.spv" -o "$RAW/14_scratchreuse.dis"
python validation/phase_xviii_v028_dq_accumulator_promotion_20261126/analyze_spirv.py \
  "$RAW/14_scratchreuse.dis" | tee "$RAW/15_spirv_analysis.txt"

diff -u shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
  "shaders/$STEM.comp" > "$RAW/13_source_delta.diff"
echo "source_delta_lines=$(wc -l < "$RAW/13_source_delta.diff")"

echo "PHASE_XX_SHADER_REPRODUCTION_DONE"
