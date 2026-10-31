"""Qualify V4.1 index-key production against the pinned reference classes.

Extracts the unchanged `RMSNorm` class and the `precompute_freqs_cis` /
`apply_rotary_emb` functions. The key projection uses a bias-free
`torch.nn.Linear` stand-in, all parameters stay in FP32, and the final
quantization is the released FP4-32/UE8M0 path exercised bit-exactly by the
quantization validator. This is the FP32 component contract, not the released
BF16/FP8 GEMM policy.
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
from torch import nn
from torch.nn import functional as F

from validate_deepseek_v41_quantization import quantize_case

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_index_keys.json"


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
    body = [
        node for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "RMSNorm")
        or (isinstance(node, ast.FunctionDef)
            and node.name in {"precompute_freqs_cis", "apply_rotary_emb"})
    ]
    if len(body) != 3:
        raise RuntimeError("missing reference RMSNorm/precompute_freqs_cis/apply_rotary_emb")
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math, "lru_cache": lru_cache}
    exec(compile(ast.Module(body=body, type_ignores=[]), "deepseek-v41-pinned-index-keys", "exec"), namespace)
    return namespace["RMSNorm"], namespace["precompute_freqs_cis"], namespace["apply_rotary_emb"]


def make_cases(reference):
    rms_norm, precompute, apply_rotary = reference
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(4141017)
    specs = [
        # name, head_dim, index_head_dim, rope, batch, groups, ratio, first_group, start_pos
        ("prefill", 16, 32, 8, 1, 2, 2, 0, 0),
        ("batch2", 32, 64, 16, 2, 3, 2, 0, 0),
        ("decode", 16, 32, 8, 1, 1, 2, 3, 7),
        ("ratio1", 16, 32, 8, 2, 2, 1, 0, 0),
    ]
    cases = []
    for name, head_dim, index_head_dim, rope, batch, groups, ratio, first_group, start_pos in specs:
        rows = batch * groups
        latent = torch.randn(batch, groups, head_dim, generator=generator, dtype=torch.float32) * 0.6
        wk = torch.randn(index_head_dim, head_dim, generator=generator, dtype=torch.float32) * 0.2
        k_norm = torch.rand(index_head_dim, generator=generator, dtype=torch.float32) * 0.5 + 0.5
        norm = rms_norm(index_head_dim, 1e-20)
        norm.weight.data.copy_(k_norm)
        with torch.no_grad():
            projected = norm(F.linear(latent, wk))
            freqs = precompute(rope, 64, 0, 10000.0, 1.0, 32, 1)
            positions = torch.tensor(
                [(first_group + group) * ratio for group in range(groups)], dtype=torch.long
            )
            selected = freqs[positions].view(groups, rope // 2)
            apply_rotary(projected[..., -rope:], selected)
            values, scales = quantize_case(
                projected.reshape(rows, index_head_dim).detach().numpy(), 32, "fp4", "ue8m0"
            )
        cases.append({
            "name": name,
            "head_dim": head_dim,
            "index_head_dim": index_head_dim,
            "rope_dim": rope,
            "batch": batch,
            "groups": groups,
            "ratio": ratio,
            "first_group": first_group,
            "start_pos": start_pos,
            "norm_eps": 1e-20,
            "wk": wk.flatten().tolist(),
            "k_norm": k_norm.tolist(),
            "latent": latent.flatten().tolist(),
            "expected": values.flatten().tolist(),
            "expected_scale_bits": [int(value) for value in scales.reshape(-1).view("uint32")],
        })
    return cases


def make_fixture(reference):
    return {"model": MODEL, "revision": REVISION, "source_sha256": SOURCE_SHA256,
            "atol_exclusive": 2e-7, "rtol": 0.0, "cases": make_cases(reference)}


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
        raise RuntimeError("index-key fixtures differ from the pinned reference")
    values = sum(len(case["expected"]) for case in fixture["cases"])
    print(f"PASS: {len(fixture['cases'])} index-key cases, {values} values; {REVISION}")


if __name__ == "__main__":
    main()
