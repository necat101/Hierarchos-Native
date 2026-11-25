#!/usr/bin/env bash
# Reproduce the Phase V GQA attention-backward numerical experiment.
#
# Run from the crate root, or let the script find it:
#
#   bash validation/gqa_attention_backward_experiment_20261124/reproduce.sh
#
# Writes the same files the receipt's tables were built from into raw/. The
# negative control is expected to fail its assertion; the script keeps going and
# reports the exit status of each step instead of stopping at the first one.
set -u
cd "$(dirname "$0")/../.." || exit 1
RAW=validation/gqa_attention_backward_experiment_20261124/raw
ANALYZE=validation/gqa_attention_backward_experiment_20261124/analyze_timing.py
mkdir -p "$RAW"

status() { # label exit
  printf '%-24s exit=%s\n' "$1" "$2"
}

HIERARCHOS_ATTENTION_BACKWARD_GQA_EXPERIMENT=1 \
  cargo test --lib gqa_two_pass_experimental_matches_scalar_gradients -- --nocapture \
  > "$RAW/parity-gqa-candidate.log" 2>&1
status candidate "$?"

HIERARCHOS_ATTENTION_BACKWARD_GQA_EXPERIMENT=mha \
  cargo test --lib gqa_two_pass_experimental_matches_scalar_gradients -- --nocapture \
  > "$RAW/parity-gqa-mha-control.log" 2>&1
status mha-control "$?"

HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 \
  cargo test --lib two_pass_attention_backward_matches_scalar_gradients -- --nocapture \
  > "$RAW/parity-mha-regression.log" 2>&1
status mha-regression "$?"

{
  echo "### gate"
  cargo test --lib two_pass_gate_keeps_every_ineligible_geometry_on_scalar -- --nocapture 2>&1 \
    | grep -E "^test |test result"
  echo "### scratch"
  cargo test --lib two_pass_scratch_sizing_and_eligibility_are_geometry_exact -- --nocapture 2>&1 \
    | grep -E "^test |test result"
  echo "### comparator"
  cargo test --lib gradient_report_rejects_a_one_ulp_difference -- --nocapture 2>&1 \
    | grep -E "^test |test result"
} > "$RAW/gate-and-comparator.log" 2>&1
status gate-and-comparator "$?"

HIERARCHOS_ATTENTION_BACKWARD_GQA_BENCH=1 \
  HIERARCHOS_ATTENTION_BACKWARD_GQA_BENCH_ITERATIONS="${ITERATIONS:-40}" \
  cargo test --lib gqa_two_pass_experimental_kernel_timing -- --nocapture \
  > "$RAW/kernel-timing-wall.log" 2>&1
status timing-wall "$?"

HIERARCHOS_ATTENTION_BACKWARD_GQA_BENCH=1 \
  HIERARCHOS_ATTENTION_BACKWARD_GQA_BENCH_ITERATIONS="${PROFILED_ITERATIONS:-10}" \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
  cargo test --lib gqa_two_pass_experimental_kernel_timing -- --nocapture \
  > "$RAW/kernel-timing-profiled.log" 2>&1
status timing-profiled "$?"

python "$ANALYZE" "$RAW/kernel-timing-profiled.log" > "$RAW/kernel-timing-gpu.json" 2>&1 \
  || python3 "$ANALYZE" "$RAW/kernel-timing-profiled.log" > "$RAW/kernel-timing-gpu.json" 2>&1
status analyze-timing "$?"

{
  echo "# Phase V artifact and source identities"
  sha256sum src/transformer.rs \
    shaders/transformer_attention_backward.comp \
    shaders/transformer_attention_backward.spv \
    shaders/transformer_attention_backward_query_rdna3.comp \
    shaders/transformer_attention_backward_query_rdna3.spv \
    shaders/transformer_attention_backward_key_rdna3.comp \
    shaders/transformer_attention_backward_key_rdna3.spv \
    validation/gqa_attention_backward_experiment_20261124/shaders/transformer_attention_backward_key_gqa_rdna3.comp \
    validation/gqa_attention_backward_experiment_20261124/shaders/transformer_attention_backward_key_gqa_rdna3.spv
} > "$RAW/source-hashes.txt"

grep -hE "^gqa two-pass experiment|qualification: status" "$RAW"/parity-*.log
