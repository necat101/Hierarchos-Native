#!/usr/bin/env bash
# Phase XIX shader reproduction.
#
# Rebuilds the experiment artifact from its source with the manifest's exact
# flags, and rebuilds the Phase XVIII promoted source with the same invocation so
# the control can be shown byte-identical to the committed artifact. No
# production shader is written: every output lands in this receipt.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xix_query_scratch_reuse_20261127
FLAGS=(-fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders)

glslc "${FLAGS[@]}" \
  "$PH/shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp" \
  -o "$PH/shaders/q_scratchreuse.spv"
glslc "${FLAGS[@]}" \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
  -o "$PH/shaders/q_control_rebuild.spv"

spirv-val --target-env vulkan1.2 "$PH/shaders/q_scratchreuse.spv"
spirv-val --target-env vulkan1.2 "$PH/shaders/q_control_rebuild.spv"
spirv-val --target-env vulkan1.2 \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.spv

sha256sum \
  "$PH/shaders/q_scratchreuse.spv" \
  "$PH/shaders/q_control_rebuild.spv" \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
  shaders/transformer_attention_backward_query_rdna3.spv \
  shaders/transformer_attention_backward_query_rdna3.comp

for module in q_control_rebuild q_scratchreuse; do
  spirv-dis "$PH/shaders/$module.spv" -o "$PH/raw/14_$module.dis"
  python validation/phase_xviii_v028_dq_accumulator_promotion_20261126/analyze_spirv.py \
    "$PH/raw/14_$module.dis"
done

diff -u shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
  "$PH/shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp" \
  > "$PH/raw/13_source_delta.diff"
echo "source_delta_lines=$(wc -l < "$PH/raw/13_source_delta.diff")"
