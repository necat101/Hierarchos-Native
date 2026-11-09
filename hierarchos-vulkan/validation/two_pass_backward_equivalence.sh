#!/usr/bin/env bash
# Numerical equivalence for the two-pass attention backward, with an execution
# proof first.
#
# A previous version of this harness compared adapter digests from two runs that
# had both silently executed the scalar kernel, and reported a match. So the
# dispatch evidence is checked before any digest is looked at, and the harness
# exits non-zero unless
#   * the scalar control arm dispatched the scalar backward at least once,
#   * the candidate arm dispatched both new passes,
#   * the candidate arm dispatched the scalar backward zero times,
#   * every corresponding adapter file exists in both arms and matches bit for
#     bit.
#
# The gate and the digest comparison are themselves exercised by negative
# controls before the training arms run, so a broken comparison cannot report a
# pass.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
BIN="${HIERARCHOS_TRAIN_BIN:-target/release/hierarchos-vulkan-transformer-train}"
[ -x "$BIN" ] || BIN="$BIN.exe"
if [ ! -x "$BIN" ]; then
  echo "FAIL: trainer binary missing: $BIN (cargo build --release --bin hierarchos-vulkan-transformer-train)"
  exit 1
fi
RUN="${TEMP:-/tmp}/hb/two-pass-$(date +%s)"
mkdir -p "$RUN"
echo "RUN=$RUN"

status=0

# gate <label> <scalar> <query> <key> <control|candidate>
gate() {
  local label="$1" scalar="$2" query="$3" key="$4" role="$5"
  local failed=0
  if [ "$role" = control ]; then
    [ "$scalar" -gt 0 ] || { echo "FAIL[$label]: scalar control never dispatched the scalar backward"; failed=1; }
    [ "$query" -eq 0 ] || { echo "FAIL[$label]: scalar control dispatched the query pass"; failed=1; }
    [ "$key" -eq 0 ] || { echo "FAIL[$label]: scalar control dispatched the key pass"; failed=1; }
  else
    [ "$scalar" -eq 0 ] || { echo "FAIL[$label]: candidate fell back to the scalar kernel"; failed=1; }
    [ "$query" -gt 0 ] || { echo "FAIL[$label]: query pass never dispatched"; failed=1; }
    [ "$key" -gt 0 ] || { echo "FAIL[$label]: key pass never dispatched"; failed=1; }
  fi
  return "$failed"
}

# compare_digests <expected-file> <actual-file>; both hold "<relpath> <sha256>".
compare_digests() {
  local expected="$1" actual="$2"
  if [ ! -s "$expected" ] || [ ! -s "$actual" ]; then
    echo "FAIL: empty digest set ($expected / $actual)"
    return 1
  fi
  if ! diff <(cut -d' ' -f1 "$expected" | sort) \
            <(cut -d' ' -f1 "$actual" | sort) >/dev/null; then
    echo "FAIL: the two arms exported different files"
    diff <(cut -d' ' -f1 "$expected" | sort) <(cut -d' ' -f1 "$actual" | sort) | head -10
    return 1
  fi
  local mismatched=0
  while read -r path hash; do
    local other
    other=$(awk -v path="$path" '$1 == path { print $2 }' "$actual")
    if [ -z "$other" ]; then
      echo "FAIL: $path missing from the candidate arm"
      return 1
    fi
    if [ "$hash" != "$other" ]; then
      echo "DIFF: $path expected $hash actual $other"
      mismatched=1
    fi
  done < "$expected"
  [ "$mismatched" -eq 0 ]
}

negative_controls() {
  local controls=0
  # The gate must reject every way a run can fake a pass.
  gate "control-no-scalar" 0 0 0 control >/dev/null && { echo "FAIL[control]: accepted a control arm that never dispatched"; controls=1; }
  gate "control-parallel" 2 1 0 control >/dev/null && { echo "FAIL[control]: accepted a control arm that ran the query pass"; controls=1; }
  gate "candidate-no-query" 0 0 2 candidate >/dev/null && { echo "FAIL[control]: accepted a candidate without the query pass"; controls=1; }
  gate "candidate-no-key" 0 2 0 candidate >/dev/null && { echo "FAIL[control]: accepted a candidate without the key pass"; controls=1; }
  gate "candidate-scalar" 2 2 2 candidate >/dev/null && { echo "FAIL[control]: accepted a candidate that fell back"; controls=1; }
  gate "good-control" 2 0 0 control >/dev/null || { echo "FAIL[control]: rejected a valid control arm"; controls=1; }
  gate "good-candidate" 0 2 2 candidate >/dev/null || { echo "FAIL[control]: rejected a valid candidate arm"; controls=1; }

  # The digest comparison must reject a changed byte, a missing file and an
  # empty set.
  local scratch="$RUN/negative-controls"
  mkdir -p "$scratch"
  printf 'adapter_model.safetensors aaaa\nadapter_config.json bbbb\n' > "$scratch/a.txt"
  printf 'adapter_model.safetensors aaaa\nadapter_config.json bbbb\n' > "$scratch/b.txt"
  compare_digests "$scratch/a.txt" "$scratch/b.txt" >/dev/null || { echo "FAIL[control]: rejected identical digests"; controls=1; }
  printf 'adapter_model.safetensors aaaa\nadapter_config.json bbbc\n' > "$scratch/changed.txt"
  compare_digests "$scratch/a.txt" "$scratch/changed.txt" >/dev/null && { echo "FAIL[control]: accepted a changed digest"; controls=1; }
  printf 'adapter_model.safetensors aaaa\n' > "$scratch/missing.txt"
  compare_digests "$scratch/a.txt" "$scratch/missing.txt" >/dev/null && { echo "FAIL[control]: accepted a missing file"; controls=1; }
  : > "$scratch/empty.txt"
  compare_digests "$scratch/empty.txt" "$scratch/a.txt" >/dev/null && { echo "FAIL[control]: accepted an empty digest set"; controls=1; }

  if [ "$controls" -ne 0 ]; then
    echo "=== NOT EQUIVALENT: negative controls were not detected"
    return 1
  fi
  echo "negative controls: 11/11 detected"
  return 0
}

echo "=== negative controls ==="
negative_controls || status=1
# Negative controls must pass in production; a failing one means the comparison
# itself is broken, so no training run can qualify anything.
if [ "$status" -ne 0 ]; then exit "$status"; fi

run() { # label two-pass steps
  local label="$1" two_pass="$2" steps="$3"
  local out="$RUN/adapter-$label"
  HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS="$two_pass" \
  HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
  HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=16 \
    "$BIN" --model target/wmma-fixture/model \
      --dataset target/wmma-fixture/dataset.jsonl \
      --output "$out" \
      --batch-size 8 --seq-len 64 --epochs 4 --max-steps "$steps" \
      --learning-rate 1e-2 --seed 7 --log-steps 1000 \
      --lora-rank 8 --lora-alpha 16 \
      > "$RUN/train-$label.txt" 2>&1
  local code=$?
  echo "$label $steps $code" | tee -a "$RUN/results.txt"
  return "$code"
}

echo "=== training arms ==="
run scalar 0 2 || { echo "FAIL: the scalar control arm exited non-zero (see $RUN/train-scalar.txt)"; status=1; }
run twopass 1 2 || { echo "FAIL: the two-pass candidate arm exited non-zero (see $RUN/train-twopass.txt)"; status=1; }

count() { grep -c "$1" "$RUN/train-$2.txt" 2>/dev/null || true; }
scalar_scalar=$(count "shader=transformer_attention_backward " scalar)
scalar_query=$(count "shader=transformer_attention_backward_query_rdna3" scalar)
scalar_key=$(count "shader=transformer_attention_backward_key_rdna3" scalar)
tp_scalar=$(count "shader=transformer_attention_backward " twopass)
tp_query=$(count "shader=transformer_attention_backward_query_rdna3" twopass)
tp_key=$(count "shader=transformer_attention_backward_key_rdna3" twopass)

echo "=== dispatch counts (profile lines) ==="
printf 'scalar arm : scalar=%s query=%s key=%s\n' "$scalar_scalar" "$scalar_query" "$scalar_key"
printf 'twopass arm: scalar=%s query=%s key=%s\n' "$tp_scalar" "$tp_query" "$tp_key"
grep -h "two-pass backward:" "$RUN/train-twopass.txt" | head -1

gate "scalar-arm" "$scalar_scalar" "$scalar_query" "$scalar_key" control || status=1
gate "twopass-arm" "$tp_scalar" "$tp_query" "$tp_key" candidate || status=1

echo "=== fingerprints ==="
{
  printf 'binary %s\n' "$(sha256sum "$BIN" | cut -d' ' -f1)"
  for shader in shaders/transformer_attention_backward_query_rdna3.spv \
                shaders/transformer_attention_backward_key_rdna3.spv \
                shaders/transformer_attention_backward_query_rdna3.comp \
                shaders/transformer_attention_backward_key_rdna3.comp \
                shaders/transformer_attention_backward.comp; do
    printf '%s %s\n' "$shader" "$(sha256sum "$shader" | cut -d' ' -f1)"
  done
} | tee "$RUN/fingerprints.txt"

if [ "$status" -ne 0 ]; then echo "=== NOT EQUIVALENT: execution proof failed"; exit "$status"; fi

echo "=== adapter digests ==="
for arm in scalar twopass; do
  if [ ! -d "$RUN/adapter-$arm" ]; then
    echo "FAIL: the $arm arm produced no adapter directory"
    exit 1
  fi
  ( cd "$RUN/adapter-$arm" && find . -type f | sed 's#^\./##' | sort | while read -r file; do
      printf '%s %s\n' "$file" "$(sha256sum "$file" | cut -d' ' -f1)"
    done ) > "$RUN/digest-$arm.txt"
done
cat "$RUN/digest-scalar.txt" | sed 's/^/scalar  /'
cat "$RUN/digest-twopass.txt" | sed 's/^/twopass /'

if compare_digests "$RUN/digest-scalar.txt" "$RUN/digest-twopass.txt"; then
  echo "=== PASS: both passes dispatched and every adapter file is bit-identical"
else
  echo "=== FAIL: adapters differ"
  exit 1
fi
