#!/usr/bin/env bash
# Phase LXXXVI ablation campaign.
#
# Each ablation module replaces the P3 SPIR-V in a *separate* Cargo target
# directory, is measured against the archived v0.2.5 harness, and is then
# removed. The committed P3 module is restored and its digest re-checked after
# every build, and the built executable must report the ablation module's own
# SPIR-V signature before it is measured -- a build that silently embedded the
# pristine P3 module instead is rejected rather than measured as "no change".
#
# The ablation executables are profiling instruments: their outputs are not
# numerically valid and nothing here is a release candidate.
set -u
cd "$(dirname "$0")/../.." || exit 1

E=validation/phase_lxxxvi_wmma_optimization
P3=shaders/wmma_gemm_f16_f16acc_twowave_p3.spv
P3_SHA=2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2
BASE="$E/bin/wmma_bench_v025.exe"
ABL=.lxxxvi-ablation-debug

restore() { cp "$E/bin/p3_preserved.spv" "$P3"; }
check_restored() {
  got=$(sha256sum "$P3" | cut -d' ' -f1)
  if [ "$got" != "$P3_SHA" ]; then
    echo "FATAL: P3 module not restored ($got)" >&2
    exit 1
  fi
}

# name:expected-fnv:expected-len
CASES=(
  "abl_staging_only:0x9991bba025ff7ee1:19696"
  "abl_no_global:0xd6699bbe4489185f:18048"
  "abl_no_epilogue:0xf4fd5b2aadfec2a9:20692"
)

for spec in "${CASES[@]}"; do
  name=${spec%%:*}; rest=${spec#*:}; want_fnv=${rest%%:*}; want_len=${rest##*:}
  restore
  cp "$ABL/$name.spv" "$P3"
  echo "=== building $name"
  CARGO_TARGET_DIR=target-lxxxvi-cand cargo build --release --bin wmma_bench \
    > "$E/ablation_build_$name.log" 2>&1
  rc=$?
  restore
  check_restored
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: build failed for $name (rc=$rc); see ablation_build_$name.log" >&2
    exit 1
  fi
  exe="$E/bin/wmma_bench_$name.exe"
  cp target-lxxxvi-cand/release/wmma_bench.exe "$exe"
  identity=$("$exe" --wmma-mode full-fp16 --rounds 1 --iterations 1 2>/dev/null \
    | grep -o 'spirv-fnv1a64:0x[0-9a-f]*/len:[0-9]*')
  echo "=== $name identity: $identity (want $want_fnv/len:$want_len)"
  if [ "$identity" != "spirv-fnv1a64:$want_fnv/len:$want_len" ]; then
    echo "FATAL: $name executable embedded the wrong module" >&2
    exit 1
  fi
  echo "=== built $name $(sha256sum "$exe" | cut -d' ' -f1)"
done

for spec in "${CASES[@]}"; do
  name=${spec%%:*}
  echo "=== measuring $name"
  python "$E/run_balanced.py" "$E/ablation/$name" "$BASE" p3 \
    "$E/bin/wmma_bench_$name.exe" p3 4 16 1
  echo "=== measured $name rc=$?"
done

echo "ABLATIONS COMPLETE"
