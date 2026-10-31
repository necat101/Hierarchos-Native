"""Qualify V4.1 rotary against the pinned released implementation.

Extracts the unchanged `precompute_freqs_cis` and `apply_rotary_emb` functions
from inference/model.py. The angle table is reproduced with the same FP32
operation order and checked to produce the reference's complex table exactly;
its bit patterns are part of the fixture. Cosines and sines are compared at the
fixed 2e-7 absolute ceiling, not claimed bit-exact, because PyTorch's trig
kernels and the native host libm can differ by one ulp.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import torch
from torch.nn import functional as F

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_rotary.json"
LIMIT = 2e-7


def load_source(path: Path | None) -> bytes:
    if path is None:
        with urlopen(f"https://huggingface.co/{MODEL}/raw/{REVISION}/inference/model.py", timeout=60) as response:
            source = response.read()
    else:
        source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"reference source hash mismatch: {actual}")
    return source


def load_reference(path: Path | None):
    tree = ast.parse(load_source(path))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"precompute_freqs_cis", "apply_rotary_emb"}
    ]
    if len(functions) != 2:
        raise RuntimeError("missing reference precompute_freqs_cis/apply_rotary_emb")
    namespace = {"torch": torch, "F": F, "math": math, "lru_cache": lru_cache}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "deepseek-v41-pinned-rotary", "exec"), namespace)
    return namespace["precompute_freqs_cis"], namespace["apply_rotary_emb"]


def angle_table(dim, max_positions, original_seq_len, base, factor, beta_fast, beta_slow, freqs_cis):
    """The same FP32 operations as precompute_freqs_cis, without trig."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(max_positions), freqs)
    rebuilt = torch.polar(torch.ones_like(angles), angles)
    if not torch.equal(rebuilt, freqs_cis):
        raise RuntimeError("angle table does not reproduce the reference complex table")
    return angles.to(torch.float32)


def make_fixture(reference):
    precompute, apply_rotary = reference
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(4141017)
    specs = [
        ("released-compressed", 64, 128, 65536, 160000.0, 16.0, 32, 1),
        ("released-sliding", 64, 128, 0, 10000.0, 1.0, 32, 1),
        ("small-yarn", 16, 64, 128, 10000.0, 8.0, 32, 1),
        ("no-yarn", 8, 32, 0, 500000.0, 1.0, 32, 1),
        ("wide-yarn", 32, 96, 4096, 40000.0, 40.0, 32, 1),
    ]
    cases = []
    for name, dim, max_positions, original, base, factor, beta_fast, beta_slow in specs:
        freqs = precompute(dim, max_positions, original, base, factor, beta_fast, beta_slow)
        angles = angle_table(dim, max_positions, original, base, factor, beta_fast, beta_slow, freqs)
        # The reference's view_as_complex needs even leading strides.
        rows, width = 5, dim + 8
        position_base, position_stride = 3, 2
        if position_base + (rows - 1) * position_stride >= max_positions:
            raise RuntimeError("fixture positions exceed the rotary table")
        values = torch.randn(1, rows, width, generator=generator, dtype=torch.float32) * 0.8
        selected = freqs[position_base + torch.arange(rows) * position_stride].view(rows, dim // 2)
        applications = []
        for inverse in (False, True):
            rotated = values.clone()
            apply_rotary(rotated[:, :, -dim:], selected, inverse=inverse)
            applications.append({"inverse": inverse, "expected": rotated.flatten().tolist()})
        cases.append({
            "name": name,
            "dim": dim,
            "max_positions": max_positions,
            "original_seq_len": original,
            "base": base,
            "factor": factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "angles": angles.flatten().tolist(),
            "cosine": freqs.real.contiguous().flatten().tolist(),
            "sine": freqs.imag.contiguous().flatten().tolist(),
            "rows": rows,
            "width": width,
            "position_base": position_base,
            "position_stride": position_stride,
            "input": values.flatten().tolist(),
            "applications": applications,
        })
    return {"model": MODEL, "revision": REVISION, "source_sha256": SOURCE_SHA256,
            "atol_exclusive": LIMIT, "rtol": 0.0, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    fixture = make_fixture(load_reference(args.reference_path))
    if args.write_fixture:
        metadata = {key: value for key, value in fixture.items() if key != "cases"}
        prefix = json.dumps(metadata, indent=2)[:-2]
        cases = ",\n".join(
            "    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
            for case in fixture["cases"]
        )
        FIXTURE.write_text(prefix + ',\n  "cases": [\n' + cases + "\n  ]\n}\n", encoding="utf-8")
    elif json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
        raise RuntimeError("rotary fixtures differ from the pinned reference")
    values = sum(len(case["angles"]) + len(case["cosine"]) * 2 for case in fixture["cases"])
    print(f"PASS: {len(fixture['cases'])} rotary configs, {values} table values; {REVISION}")


if __name__ == "__main__":
    main()
