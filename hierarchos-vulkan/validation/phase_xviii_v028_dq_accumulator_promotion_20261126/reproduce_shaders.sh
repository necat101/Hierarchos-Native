#!/usr/bin/env bash
# Phase XVIII shader reproduction: rebuild the promoted artifact from its source,
# rebuild the shipped control with the same invocation, and prove both hashes.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
PH=validation/phase_xviii_v028_dq_accumulator_promotion_20261126
mkdir -p "$PH/shaders" "$PH/raw"

echo "== toolchain =="
glslc --version
spirv-val --version

echo "== compile promoted artifact =="
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
  -o shaders/transformer_attention_backward_query_rdna3_dqunroll.spv

echo "== rebuild the shipped production source with the same invocation =="
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
  shaders/transformer_attention_backward_query_rdna3.comp \
  -o "$PH/shaders/q_rebuild_phase_xviii.spv"

echo "== hashes =="
sha256sum \
  shaders/transformer_attention_backward_query_rdna3.comp \
  shaders/transformer_attention_backward_query_rdna3.spv \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
  "$PH/shaders/q_rebuild_phase_xviii.spv"

echo "== control identity (unmodified source must reproduce the shipped module) =="
cmp shaders/transformer_attention_backward_query_rdna3.spv "$PH/shaders/q_rebuild_phase_xviii.spv"
echo "control: byte-identical"

echo "== promoted identity (Phase XVII candidate bytes) =="
cmp shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
  validation/phase_xvii_v028_optimization_discovery_20261126/shaders/q_dqunroll.spv
echo "promoted: byte-identical to the Phase XVII candidate"

echo "== validation =="
spirv-val --target-env vulkan1.2 shaders/transformer_attention_backward_query_rdna3_dqunroll.spv
spirv-val --target-env vulkan1.2 shaders/transformer_attention_backward_query_rdna3.spv
spirv-val --target-env vulkan1.2 "$PH/shaders/q_rebuild_phase_xviii.spv"
echo "spirv-val: all three PASS"

echo "== disassembly and accumulator analysis =="
spirv-dis --no-color shaders/transformer_attention_backward_query_rdna3.spv \
  -o "$PH/raw/13_shipped.dis.txt"
spirv-dis --no-color shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
  -o "$PH/raw/14_dqunroll.dis.txt"
{
  echo "=== SHIPPED (control) ==="
  python "$PH/analyze_spirv.py" "$PH/raw/13_shipped.dis.txt" \
    shipped_transformer_attention_backward_query_rdna3
  echo
  echo "=== CANDIDATE (promoted) ==="
  python "$PH/analyze_spirv.py" "$PH/raw/14_dqunroll.dis.txt" \
    transformer_attention_backward_query_rdna3_dqunroll
} | tee "$PH/raw/15_spirv_analysis.txt"
