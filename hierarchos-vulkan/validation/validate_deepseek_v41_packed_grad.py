"""Pinned CPU PyTorch gradients through packed and BF16 V4.1 projections.

The forward is the released block-scaled GEMM contract already qualified by
`validate_deepseek_v41_mixed.py`: FP8 activation quantization per K block, FP32
per-block dots, activation scale then weight scale application, FP32 block
accumulation, one BF16 output rounding, and - with an adapter - the PEFT merge
`base + scale * delta` rounded once more. The input is the released stream
boundary, so a BF16 stream cast precedes the quantizer.

This fixture differentiates that function with the quantizer's rounding treated
as a frozen boundary: the straight-through edge keeps the forward value (the
quantized code) but carries `1 / scale` to the stream input. Because UF8M0
scales are powers of two, that division is exact, so the pinned contract is
`grad_x = grad_out @ (decoded_weight * weight_scale)` with the activation scale
cancelling exactly. Adapter gradients ride the dense branch and are ordinary
FP32 VJPs of the merged projection.

Values are stored as FP32 bit patterns so the JSON round trip is exact.
Regenerate only together with a deliberate contract change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import validate_deepseek_v41_packed_weights as packed
import validate_deepseek_v41_quantization as quant

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_packed_grad.json"
FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
SEED = 4141205


def bits(tensor: torch.Tensor) -> list[int]:
    return tensor.detach().contiguous().numpy().view(np.uint32).reshape(-1).tolist()


def ste_round(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Forward equals the rounded value; the gradient is the identity."""
    return value + (value.to(dtype).to(torch.float32) - value).detach()


def block_size(fmt: str) -> int:
    return 128 if fmt == "fp8_block128" else 32


def make_weight(rng: np.random.Generator, rows: int, cols: int, fmt: str) -> dict:
    block = block_size(fmt)
    if fmt == "fp4_row32":
        data = rng.integers(0, 256, rows * cols // 2, dtype=np.uint8).tolist()
    else:
        codes = torch.from_numpy(rng.normal(0, 1, (rows, cols)).astype(np.float32))
        data = codes.to(torch.float8_e4m3fn).view(torch.uint8).flatten().tolist()
    # FP8 formats scale K blocks of rows; FP4 row32 stores one scale per row.
    scale_rows = rows if fmt == "fp4_row32" else (rows + block - 1) // block
    scales = rng.integers(120, 125, scale_rows * (cols // block), dtype=np.uint8).tolist()
    return dict(rows=rows, cols=cols, dtype=fmt, data=data, scales=scales)


def packed_forward(
    x: torch.Tensor,
    spec: dict,
    stream_bf16: bool,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    scale: float = 0.0,
) -> torch.Tensor:
    rows, inner = x.shape
    n, k = spec["rows"], spec["cols"]
    block = block_size(spec["dtype"])
    xb = ste_round(x, torch.bfloat16) if stream_bf16 else x
    codes, scales = quant.quantize_case(xb.detach().numpy().astype(np.float32), block, "fp8", "ue8m0")
    sa = torch.from_numpy(scales).repeat_interleave(block, dim=1)
    quantized = torch.from_numpy(codes) / sa
    # The straight-through edge: the rounded code carries the forward value,
    # and `xb / sa` (exact, power-of-two) carries the gradient.
    q = quantized + (xb / sa - (xb / sa).detach())
    w = packed.unpack(spec["data"], n, k, spec["dtype"], FP4_TABLE)
    sw = packed.scale_matrix(spec["scales"], n, k, spec["dtype"])
    total = torch.zeros(rows, n)
    for start in range(0, k, block):
        dot = F.linear(q[:, start : start + block], w[:, start : start + block])
        scaled = dot * sa[:, start : start + 1]
        scaled = scaled * sw[:, start]
        total = total + scaled
    out = ste_round(total, torch.bfloat16)
    if a is not None:
        hidden = F.linear(x, a)
        delta = F.linear(hidden, b)
        out = ste_round(out + scale * delta, torch.bfloat16)
    return out


def make_bf16_weight(rng, rows, cols):
    """BF16 dense storage: the payload is BF16-exact and the GEMM is ordinary
    FP32 arithmetic over those values, with one BF16 rounding at the output."""
    values = torch.from_numpy(rng.normal(0, 0.1, (rows, cols)).astype(np.float32)).bfloat16().float()
    return dict(rows=rows, cols=cols, dtype="bf16_dense", values_bits=bits(values))


def make_bf16_case(rng, name, rows, inner, out, rank, alpha, rslora):
    spec = make_bf16_weight(rng, out, inner)
    values = torch.tensor(spec["values_bits"], dtype=torch.uint32).view(torch.float32).reshape(out, inner)
    x = torch.from_numpy(rng.normal(0, 0.2, (rows, inner)).astype(np.float32)).requires_grad_()
    upstream = torch.from_numpy(rng.normal(0, 0.15, (rows, out)).astype(np.float32))
    scale = alpha / (np.sqrt(rank) if rslora else rank) if rank else 0.0
    a = b = None
    if rank:
        a = torch.from_numpy(rng.normal(0, 0.05, (rank, inner)).astype(np.float32)).requires_grad_()
        b = torch.from_numpy(rng.normal(0, 0.05, (out, rank)).astype(np.float32)).requires_grad_()
    xb = ste_round(x, torch.bfloat16)
    y = ste_round(F.linear(xb, values), torch.bfloat16)
    if a is not None:
        y = ste_round(y + scale * F.linear(F.linear(x, a), b), torch.bfloat16)
    (y * upstream).sum().backward()
    return {
        "name": name,
        "format": "bf16_dense",
        "rows": rows,
        "inner": inner,
        "out": out,
        "stream": "bf16",
        "rank": rank,
        "alpha": alpha,
        "rslora": rslora,
        "scale": scale,
        "weight": spec,
        "x_bits": bits(x),
        "upstream_bits": bits(upstream),
        "output_bits": bits(y),
        "grad_x_bits": bits(x.grad),
        "a_bits": bits(a) if rank else [],
        "b_bits": bits(b) if rank else [],
        "grad_a_bits": bits(a.grad) if rank else [],
        "grad_b_bits": bits(b.grad) if rank else [],
    }


def make_case(rng, name, fmt, rows, inner, out, stream_bf16, rank=0, alpha=0.0, rslora=False):
    spec = make_weight(rng, out, inner, fmt)
    # Magnitudes follow the same regime as the other V4.1 fixtures. The shared
    # input/weight-gradient kernels reproduce ATen's reduction order for the
    # recorded shapes, but a different ATen micro-kernel path can still differ
    # by one or two ulps, and the gate is absolute (`rtol=0`), so the operands
    # stay small enough that a single ulp is far below 2e-7. The same shapes at
    # four times these magnitudes measure up to 4.8e-7 (see `sensitivity`).
    x = torch.from_numpy(rng.normal(0, 0.2, (rows, inner)).astype(np.float32)).requires_grad_()
    upstream = torch.from_numpy(rng.normal(0, 0.15, (rows, out)).astype(np.float32))
    scale = alpha / (np.sqrt(rank) if rslora else rank) if rank else 0.0
    a = b = None
    if rank:
        a = torch.from_numpy(rng.normal(0, 0.05, (rank, inner)).astype(np.float32)).requires_grad_()
        b = torch.from_numpy(rng.normal(0, 0.05, (out, rank)).astype(np.float32)).requires_grad_()
    y = packed_forward(x, spec, stream_bf16, a, b, scale)
    (y * upstream).sum().backward()
    return {
        "name": name,
        "format": fmt,
        "rows": rows,
        "inner": inner,
        "out": out,
        "stream": "bf16" if stream_bf16 else "fp32",
        "rank": rank,
        "alpha": alpha,
        "rslora": rslora,
        "scale": scale,
        "weight": spec,
        "x_bits": bits(x),
        "upstream_bits": bits(upstream),
        "output_bits": bits(y),
        "grad_x_bits": bits(x.grad),
        "a_bits": bits(a) if rank else [],
        "b_bits": bits(b) if rank else [],
        "grad_a_bits": bits(a.grad) if rank else [],
        "grad_b_bits": bits(b.grad) if rank else [],
    }


# name, format, rows, inner, out, bf16 stream, rank, alpha, rslora
#
# Shapes cover the BF16 stream cast and the FP32 stream, all three packed
# weight formats, and the narrow-row/short-K and wide-row cases whose ATen
# reduction differs. Adapter cases gate the factor VJPs and the merged
# BF16 boundary.
CASES = [
    ("fp8-block32-bf16-stream", "fp8_block32", 4, 64, 32, True, 0, 0.0, False),
    ("fp8-block32-fp32-stream", "fp8_block32", 3, 64, 48, False, 0, 0.0, False),
    ("fp8-block32-wide-rows", "fp8_block32", 8, 64, 48, True, 0, 0.0, False),
    ("fp8-block128-bf16-stream", "fp8_block128", 2, 256, 64, True, 0, 0.0, False),
    ("fp8-block128-fp32-stream", "fp8_block128", 5, 128, 96, False, 0, 0.0, False),
    ("fp4-row32-bf16-stream", "fp4_row32", 5, 96, 48, True, 0, 0.0, False),
    ("fp4-row32-lora-bf16", "fp4_row32", 6, 64, 32, True, 4, 2.0, False),
    ("fp8-block32-rslora-bf16", "fp8_block32", 5, 128, 64, True, 4, 2.0, True),
    ("fp8-block32-lora-wide", "fp8_block32", 8, 64, 48, True, 8, 4.0, False),
]


def make_fixture():
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    return {
        "model": quant.MODEL,
        "revision": quant.REVISION,
        "source_sha256": packed.MODEL_SHA256,
        "kernel_sha256": quant.KERNEL_SHA256,
        "precision": "packed_mixed",
        "tolerance": 2e-7,
        "comparison": "strict_less_than",
        "rtol": 0.0,
        "torch_version": torch.__version__,
        "quantizer_note": (
            "The activation quantizer's rounding is a frozen boundary: the "
            "straight-through edge carries 1/scale (exact for UE8M0) to the "
            "stream input, so grad_x is the input gradient against the decoded, "
            "weight-scaled matrix."
        ),
        # Measured while calibrating, with input 0.6 / upstream 0.4 / adapter
        # 0.1 and merge scales 2.0: the shared input-gradient and
        # weight-gradient kernels occasionally select a different ATen
        # micro-kernel order at those magnitudes, and the differences are one
        # to three ulps of the (larger) gradients. The recorded operands keep
        # one ulp far below the absolute gate; the wider magnitudes are not
        # claimed.
        "sensitivity": {
            "fp8-block32-fp32-stream dX": 2.08616256713867188e-7,
            "fp4-row32-lora-bf16 dB": 2.384185791015625e-7,
            "fp8-block32-rslora-bf16 dB": 4.76837158203125e-7,
            "fp8-block32-lora-wide dB": 4.76837158203125e-7,
        },
        "cases": [make_case(rng, *case) for case in CASES]
        + [
            make_bf16_case(rng, "bf16-dense-bf16-stream", 5, 64, 32, 0, 0.0, False),
            make_bf16_case(rng, "bf16-dense-lora-bf16", 4, 32, 16, 4, 2.0, False),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    payload = make_fixture()
    if FIXTURE.exists() and json.loads(FIXTURE.read_text(encoding="utf-8")) != payload:
        raise SystemExit("packed-gradient fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    values = sum(
        len(case["grad_x_bits"]) + len(case["grad_a_bits"]) + len(case["grad_b_bits"])
        for case in payload["cases"]
    )
    print(f"PASS: {len(payload['cases'])} packed/BF16 projection VJPs, {values} frozen gradients; {quant.REVISION}")


if __name__ == "__main__":
    main()
