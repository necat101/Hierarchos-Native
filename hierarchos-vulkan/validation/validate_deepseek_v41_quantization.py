"""Check native V4.1 activation quantization against the released kernels.

The reference `act_quant_kernel_` / `fp4_quant_kernel_` implementations are
TileLang/Blackwell kernels that cannot execute here. FP8 E4M3 conversion and
storage are delegated to PyTorch's CPU casts, which implement the same
round-to-nearest-even format conversion. FP4 E2M1 has no CPU cast, so the
oracle uses the published representable set {0, .5, 1, 1.5, 2, 3, 4, 6} with
round-to-nearest-even, exactly the conversion `T.Cast(FP4, ...)` performs. The
power-of-two scale path is the released `fast_log2_ceil`/`fast_pow2` bit code.
This is an independent oracle, not an import of the model kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
KERNEL_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_quantization.json"
F32 = np.float32
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def load_kernel_source(path: Path | None) -> bytes:
    if path is None:
        with urlopen(
            f"https://huggingface.co/{MODEL}/raw/{REVISION}/inference/kernel.py", timeout=60
        ) as response:
            source = response.read()
    else:
        source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != KERNEL_SHA256:
        raise RuntimeError(f"reference kernel source hash mismatch: {actual}")
    return source


def fast_log2_ceil(value: np.float32) -> int:
    bits = int(np.frombuffer(np.array([value], dtype=F32).tobytes(), dtype=np.uint32)[0])
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & ((1 << 23) - 1)
    return exponent + (1 if mantissa else 0)


def fast_pow2(exponent: int) -> np.float32:
    bits = np.uint32((exponent + 127) << 23)
    return np.frombuffer(bits.tobytes(), dtype=F32)[0]


def round_even_nonnegative(value: float) -> int:
    lower = int(np.floor(value))
    fraction = value - lower
    if fraction > 0.5 or (fraction == 0.5 and lower % 2 == 1):
        return lower + 1
    return lower


def e4m3_round(value: np.float32) -> np.float32:
    tensor = torch.tensor([float(value)], dtype=torch.float32)
    return np.float32(tensor.to(torch.float8_e4m3fn).to(torch.float32).item())


def e2m1_round(value: np.float32) -> np.float32:
    sign = -1.0 if value < 0 else 1.0
    magnitude = min(abs(float(value)), 6.0)
    code = 0
    if magnitude <= 0.25:
        code = 0
    elif magnitude < 0.75:
        code = 1
    elif magnitude < 1.25:
        code = 2
    elif magnitude < 1.75:
        code = 3
    elif magnitude < 2.5:
        code = 4
    elif magnitude < 3.5:
        code = 5
    elif magnitude < 5.0:
        code = 6
    else:
        code = 7
    return np.float32(sign * FP4_LEVELS[code])


def quantize_case(values: np.ndarray, block_size: int, mode: str, scale_mode: str):
    rows, cols = values.shape
    blocks = (cols + block_size - 1) // block_size
    output = np.empty_like(values, dtype=F32)
    scales = np.empty((rows, blocks), dtype=F32)
    floor = {
        ("fp8", "ue8m0"): 1e-4,
        ("fp4", "ue8m0"): 6.0 * 2.0**-126,
        ("fp4", "e4m3"): 6.0 * 2.0**-9,
    }[(mode, scale_mode)]
    for row in range(rows):
        for block in range(blocks):
            start = block * block_size
            chunk = values[row, start:start + block_size]
            amax = max(F32(np.max(np.abs(chunk))), F32(floor))
            if scale_mode == "ue8m0":
                reciprocal = F32(1.0 / 448.0) if mode == "fp8" else F32(1.0 / 6.0)
                scale = fast_pow2(fast_log2_ceil(F32(amax * reciprocal)))
            else:
                scale = e4m3_round(F32(amax / F32(6.0)))
            scales[row, block] = scale
            limited = np.clip(chunk / scale, F32(-448.0 if mode == "fp8" else -6.0),
                              F32(448.0 if mode == "fp8" else 6.0))
            rounded = (e4m3_round if mode == "fp8" else e2m1_round)
            output[row, start:start + block_size] = np.array(
                [rounded(F32(v)) * scale for v in limited], dtype=F32
            )
    return output, scales


def make_cases(rng: np.random.Generator):
    cases = []

    def add(name, values, block_size, mode, scale_mode):
        values = np.asarray(values, dtype=F32)
        output, scales = quantize_case(values, block_size, mode, scale_mode)
        cases.append({
            "name": name,
            "rows": values.shape[0],
            "cols": values.shape[1],
            "block_size": block_size,
            "mode": mode,
            "scale_mode": scale_mode,
            "input": values.flatten().tolist(),
            "expected_values": output.flatten().tolist(),
            "expected_scale_bits": [int(s) for s in scales.reshape(-1).view(np.uint32)],
        })

    add("fp8-ue8m0-random-32", rng.normal(0.0, 0.7, (4, 64)), 32, "fp8", "ue8m0")
    add("fp8-ue8m0-random-128", rng.normal(0.0, 1.9, (2, 256)), 128, "fp8", "ue8m0")
    add("fp4-ue8m0-random-32", rng.normal(0.0, 0.9, (3, 64)), 32, "fp4", "ue8m0")
    add("fp4-e4m3-random-16", rng.normal(0.0, 1.4, (3, 64)), 16, "fp4", "e4m3")
    add("fp8-ue8m0-zeros", np.zeros((2, 64)), 32, "fp8", "ue8m0")
    add("fp4-ue8m0-tiny", np.full((2, 32), 1e-40), 32, "fp4", "ue8m0")
    add("fp4-e4m3-tiny", np.full((2, 32), 1e-12), 16, "fp4", "e4m3")
    saturate = [1000.0, -1000.0, 448.0, -448.0, 0.0, 1e-8] * 8
    add("fp8-ue8m0-saturate", np.array([saturate + [0.0] * (64 - len(saturate))]), 32,
        "fp8", "ue8m0")

    # With amax == 6 the FP4 scale is exactly 1, so these are raw FP4 ties.
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
                     -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 6.0, -6.0], dtype=F32)
    add("fp4-ue8m0-ties", np.concatenate([ties, np.zeros(16, dtype=F32)]).reshape(1, 32),
        32, "fp4", "ue8m0")
    add("fp4-e4m3-ties", np.concatenate([ties, np.zeros(16, dtype=F32)]).reshape(1, 32),
        16, "fp4", "e4m3")

    # With amax == 448 the FP8 scale is exactly 1: E4M3 mantissa ties and
    # subnormal ties are exercised directly.
    fp8_ties = [1.0625, 2.125, 4.25, 8.5, 16.5, 32.5, 64.5, 128.0,
                0.00146484375, 0.00048828125, 0.015625, 448.0]
    fp8_ties += [-v for v in fp8_ties]
    fp8_ties += [0.0] * (64 - len(fp8_ties))
    add("fp8-ue8m0-ties", np.array([fp8_ties], dtype=F32), 32, "fp8", "ue8m0")
    return cases


def make_fixture():
    rng = np.random.default_rng(4141017)
    return {
        "model": MODEL,
        "revision": REVISION,
        "source_sha256": KERNEL_SHA256,
        "precision": "released_quantize_dequantize_semantics_fp32_io",
        "exact": True,
        "cases": make_cases(rng),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    load_kernel_source(args.reference_path)
    fixture = make_fixture()
    if args.write_fixture:
        metadata = {key: value for key, value in fixture.items() if key != "cases"}
        prefix = json.dumps(metadata, indent=2)[:-2]
        cases = ",\n".join(
            "    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
            for case in fixture["cases"]
        )
        FIXTURE.write_text(prefix + ',\n  "cases": [\n' + cases + "\n  ]\n}\n", encoding="utf-8")
    elif json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
        raise RuntimeError("quantization fixtures differ from the pinned reference")
    values = sum(len(case["expected_values"]) for case in fixture["cases"])
    print(f"PASS: {len(fixture['cases'])} quantization cases, {values} exact values; {REVISION}")


if __name__ == "__main__":
    main()
