"""Freeze the pinned CPU's sqrt instruction seed and independent torch outputs.

The seed table covers every positive FP32 mantissa in [1,4), independently of
any model input. Every one of the 2**24 instruction results must equal its
2048-value bucket. The native runtime scales these seeds by exponent and uses
separately rounded FP32 refinement; it never executes CPU activation math.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "shaders/deepseek_v41_rsqrt_seed_amd_zen4.bin"
FIXTURE = ROOT / "validation/fixtures/deepseek_v41_cpu_math.json"


def generate(library):
    torch.set_num_threads(1)
    instruction = ctypes.CDLL(str(library.resolve())).rsqrt_seeds
    instruction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    inputs = np.arange(0x3f800000, 0x40800000, dtype=np.uint32)
    observed = np.empty_like(inputs)
    instruction(inputs.ctypes.data, observed.ctypes.data, inputs.size)
    buckets = observed[::2048].copy()
    assert buckets.size == 8192
    assert np.array_equal(observed.reshape(-1,2048),np.broadcast_to(buckets[:,None],(8192,2048)))
    payload = buckets.astype('<u4').tobytes()
    rng = np.random.default_rng(4141107)
    x = np.exp(rng.uniform(-20,20,4096)).astype(np.float32)
    x = np.concatenate((x,np.array([0.125,0.25,0.5,1,2,4,8,16],np.float32)))
    expected = torch.from_numpy(x).sqrt().numpy().view(np.uint32).tolist()
    fixture = dict(policy="windows_mkl_sse_rsqrt_refine_v1",
                   torch_revision=torch.version.git_version,
                   seed_table_sha256=hashlib.sha256(payload).hexdigest(),
                   seed_count=8192, bucket_width=2048,
                   input_bits=x.view(np.uint32).tolist(),expected_bits=expected)
    return payload,fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-library",type=Path,required=True)
    p.add_argument("--write-fixture",action="store_true")
    args=p.parse_args()
    if args.write_fixture and (TABLE.exists() or FIXTURE.exists()):
        raise FileExistsError("CPU math data is frozen; do not retune expectations")
    payload,fixture=generate(args.seed_library)
    if args.write_fixture:
        TABLE.write_bytes(payload)
        FIXTURE.write_text(json.dumps(fixture,separators=(",",":"),allow_nan=False))
    elif TABLE.read_bytes()!=payload or json.loads(FIXTURE.read_text())!=fixture:
        raise AssertionError("CPU sqrt policy differs from the pinned instruction/runtime")
    print("PASS CPU math oracle: 16777216 instruction outputs, 8192 seeds,",len(fixture["expected_bits"]),"sqrt values")


if __name__ == "__main__":
    main()
