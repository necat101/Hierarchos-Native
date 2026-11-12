#!/usr/bin/env bash
# Local replay of the cross-adapter-ab generated steps. Writes only under
# target/phase19z/packed-cell so historical validation receipts are untouched.
set +e
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH="$PWD/target/phase19z/torch-cpu"
out=target/phase19z/packed-cell
mkdir -p "$out"

run() {
  name="$1"; shift
  echo "=== $name ==="
  "$@" >"$out/$name.log" 2>&1
  status=$?
  tail -4 "$out/$name.log"
  echo "$name EXIT=$status"
  echo "$name EXIT=$status" >>"$out/status.txt"
}

guard() {
  name="$1"; producer="$2"; receipt="$3"
  echo "=== $name guard ==="
  python validation/check_measured_producer_routes.py --producer "$producer" \
    --receipt "$out/$receipt" >"$out/$name.guard.log" 2>&1
  status=$?
  cat "$out/$name.guard.log"
  echo "$name GUARD_EXIT=$status"
  echo "$name GUARD_EXIT=$status" >>"$out/status.txt"
}

run tbptt_forward python validation/bench_rwkv_tbptt_packed_chains.py --batch 1 --batch 4 --batch 8 --batch 16 --batch 32 --chain forward --steps 8 --iterations 20 --warmup 2 --record "$out/tbptt_forward.json"
guard tbptt_forward tbptt-packed-forward tbptt_forward.json

run tbptt_remat python validation/bench_rwkv_tbptt_packed_chains.py --batch 1 --batch 4 --batch 8 --batch 16 --batch 32 --chain remat --steps 8 --iterations 20 --warmup 2 --record "$out/tbptt_remat.json"
guard tbptt_remat tbptt-packed-rematerialization tbptt_remat.json

run tbptt_forward_fp16 python validation/bench_rwkv_tbptt_packed_chains.py --batch 1 --batch 4 --batch 8 --batch 16 --batch 32 --chain forward --parameter-storage fp16-storage --steps 8 --iterations 20 --warmup 2 --record "$out/tbptt_forward_fp16.json"
guard tbptt_forward_fp16 tbptt-packed-forward-fp16 tbptt_forward_fp16.json

run tbptt_remat_fp16 python validation/bench_rwkv_tbptt_packed_chains.py --batch 1 --batch 4 --batch 8 --batch 16 --batch 32 --chain remat --parameter-storage fp16-storage --steps 8 --iterations 20 --warmup 2 --record "$out/tbptt_remat_fp16.json"
guard tbptt_remat_fp16 tbptt-packed-rematerialization-fp16 tbptt_remat_fp16.json

run tbptt_backward python validation/bench_rwkv_tbptt_packed_chains.py --batch 1 --batch 4 --batch 8 --batch 16 --batch 32 --chain backward --steps 8 --iterations 40 --warmup 3 --record "$out/tbptt_backward.json"
guard tbptt_backward tbptt-backward-fusion tbptt_backward.json

run tbptt_backward_fp16 python validation/bench_rwkv_tbptt_packed_chains.py --batch 1 --batch 4 --batch 8 --batch 16 --batch 32 --chain backward --parameter-storage fp16-storage --steps 8 --iterations 40 --warmup 3 --record "$out/tbptt_backward_fp16.json"
guard tbptt_backward_fp16 tbptt-backward-fusion-fp16 tbptt_backward_fp16.json

run tbptt_time_mix python validation/bench_rwkv_tbptt_time_mix.py --batch 1 --batch 4 --batch 8 --batch 16 --steps 8 --iterations 8 --warmup 2 --record "$out/tbptt_time_mix.json"
guard tbptt_time_mix tbptt-time-mix tbptt_time_mix.json

run composition python validation/verify_rwkv_packed_cell_composition.py --expected-vendor auto --record "$out/composition.json"
guard composition packed-cell-composition composition.json

run key_state_profile python validation/profile_rwkv_key_state_forward.py --batch 1 --batch 8 --batch 16 --batch 32 --runs 5 --record "$out/key_state_profile.json"

echo DONE
