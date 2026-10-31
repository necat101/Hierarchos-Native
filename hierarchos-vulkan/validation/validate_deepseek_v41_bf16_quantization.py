"""CPU-reference FP4 midpoint regression for every positive finite E4M3 scale.

BF16 compressed latents can normalize to exact FP4 midpoints. Ordinary Vulkan
division may return the adjacent FP32 value and change the quantized code.
This uses the existing independent CPU quantizer without changing its policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import validate_deepseek_v41_quantization as quant

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_bf16_quantization.json"


def make_fixture(kernel_path):
    quant.load_kernel_source(kernel_path)
    scales = torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    midpoints = np.array([.25,.75,1.25,1.75,2.5,3.5,5.0,-.25,-.75,-1.25,-1.75,-2.5,-3.5,-5.0,6.0,-6.0], dtype=np.float32)
    x = scales[:, None] * midpoints
    assert torch.equal(torch.from_numpy(x), torch.from_numpy(x).bfloat16().float())
    y, actual_scales = quant.quantize_case(x, 16, "fp4", "e4m3")
    assert np.array_equal(scales, actual_scales[:, 0])
    return dict(revision=quant.REVISION, kernel_sha256=quant.KERNEL_SHA256,
                rows=len(scales), cols=16, input=x.flatten().tolist(),
                expected_bits=y.view(np.uint32).flatten().tolist(),
                expected_scale_bits=actual_scales.view(np.uint32).flatten().tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel-reference-path", type=Path, required=True)
    p.add_argument("--write-fixture", action="store_true")
    args = p.parse_args()
    f = make_fixture(args.kernel_reference_path)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(f, separators=(",", ":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f:
        raise AssertionError("FP4 midpoint fixture differs from the independent CPU reference")
    print(f"PASS {f['rows']*f['cols']} FP4 midpoint outputs across all 126 positive finite E4M3 scales")


if __name__ == "__main__":
    main()
