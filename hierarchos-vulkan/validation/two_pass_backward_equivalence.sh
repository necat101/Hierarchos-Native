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
#   * the candidate arm dispatched the scalar backward zero times.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=target/release/hierarchos-vulkan-transformer-train
[ -x "$BIN" ] || BIN="$BIN.exe"
RUN="${TEMP:-/tmp}/hb/two-pass-$(date +%s)"
mkdir -p "$RUN"
echo "RUN=$RUN"

run() { # label two_pass steps
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
  echo "$label $steps $?" | tee -a "$RUN/results.txt"
}

run scalar 0 2
run twopass 1 2

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

status=0
if [ "$scalar_scalar" -eq 0 ]; then echo "FAIL: scalar control never dispatched"; status=1; fi
if [ "$tp_query" -eq 0 ]; then echo "FAIL: query pass never dispatched"; status=1; fi
if [ "$tp_key" -eq 0 ]; then echo "FAIL: key pass never dispatched"; status=1; fi
if [ "$tp_scalar" -ne 0 ]; then echo "FAIL: candidate fell back to the scalar kernel"; status=1; fi
if [ "$scalar_query" -ne 0 ] || [ "$scalar_key" -ne 0 ]; then
  echo "FAIL: scalar control dispatched a parallel pass"; status=1
fi
if [ "$status" -ne 0 ]; then echo "=== NOT EQUIVALENT: execution proof failed"; exit "$status"; fi

echo "=== adapter digests ==="
( cd "$RUN" && find adapter-scalar adapter-twopass -type f | sort | while read -r f; do
    printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "${f#adapter-}"
  done ) | tee "$RUN/digests.txt"

if diff <(sed 's/^[^ ]*  //' "$RUN/digests.txt" | sort) \
        <(sed 's/^[^ ]*  //' "$RUN/digests.txt" | sort) >/dev/null; then :; fi
scalar_sum=$(grep '  scalar/' "$RUN/digests.txt" | awk '{print $1}' | sort)
twopass_sum=$(grep '  twopass/' "$RUN/digests.txt" | awk '{print $1}' | sort)
if [ "$scalar_sum" = "$twopass_sum" ]; then
  echo "=== PASS: both passes dispatched and the adapters are bit-identical"
else
  echo "=== FAIL: adapters differ"
  exit 1
fi
