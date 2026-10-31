"""Qualify FP32 V4.1 compressor fixtures against the pinned released forward.

Extracts DeepSeek's unchanged RMSNorm and Compressor classes. Dense Linear is
torch.nn.Linear without bias; all parameters are explicitly promoted to FP32
before identical fixture weights are installed. This isolates the unquantized
math and does not claim BF16/FP8 production execution parity.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from validate_deepseek_v41_candidates import MODEL, REVISION, SOURCE_SHA256, load_source


FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_compressor.json"


class DenseLinear(nn.Linear):
    def __init__(self, in_features, out_features, dtype=None):
        super().__init__(in_features, out_features, bias=False, dtype=dtype or torch.float32)


def load_reference(path):
    tree = ast.parse(load_source(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"RMSNorm", "Compressor"}]
    if len(classes) != 2:
        raise RuntimeError("missing reference RMSNorm/Compressor")
    namespace = {"torch": torch, "nn": nn, "Linear": DenseLinear}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "deepseek-v41-pinned-compressor", "exec"), namespace)
    return namespace["Compressor"]


@torch.no_grad()
def make_fixture(reference):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(4141017)
    cases = []
    specs = [(1, 8, 16, 2, 3, 8), (2, 8, 16, 2, 1, 8), (2, 8, 16, 2, 3, 9),
             (2, 8, 16, 2, 4, 10), (2, 16, 32, 1, 7, 12), (1, 16, 64, 2, 2, 7),
             (2, 16, 64, 2, 3, 9), (2, 7, 8, 1, 3, 8), (2, 32, 128, 2, 3, 8),
             (2, 32, 512, 1, 3, 7)]
    for ratio, hidden, head, batch, prefill, total in specs:
        args = SimpleNamespace(compress_ratios=[ratio], dim=hidden, head_dim=head,
                               norm_eps=1e-20, max_batch_size=batch)
        model = reference(args, 0).float()
        kv = torch.randn(head, hidden, generator=generator) * 0.15
        gate = torch.randn(head, hidden, generator=generator) * 0.7 if ratio == 2 else None
        norm = torch.rand(head, generator=generator) * 0.5 + 0.5
        model.wkv.weight.copy_(kv)
        if ratio == 2:
            model.wgate.weight.copy_(gate)
        model.norm.weight.copy_(norm)
        norm_inputs = []
        model.norm.register_forward_pre_hook(lambda module, args: norm_inputs.append(args[0].detach().clone()))
        tokens = torch.randn(batch, total, hidden, generator=generator)
        steps = []
        for start, length in [(0, prefill), *[(pos, 1) for pos in range(prefill, total)]]:
            x = tokens[:, start:start + length].contiguous()
            latent = model(x, start)
            pooled = None if latent is None else norm_inputs[-1]
            steps.append({"start_pos": start, "seq_len": length, "input": x.flatten().tolist(),
                          "groups": 0 if latent is None else latent.size(1),
                          "pre_norm": [] if pooled is None else pooled.flatten().tolist(),
                          "variance": [] if pooled is None else (pooled.square().mean(-1) + 1e-20).flatten().tolist(),
                          "rstd": [] if pooled is None else torch.rsqrt(pooled.square().mean(-1) + 1e-20).flatten().tolist(),
                          "projected_kv": model.wkv(x).flatten().tolist(),
                          "projected_gate": [] if ratio == 1 else model.wgate(x).flatten().tolist(),
                          "expected": [] if latent is None else latent.flatten().tolist()})
        full = model(tokens, 0)
        cases.append({"name": f"ratio-{ratio}-hidden-{hidden}-head-{head}-batch-{batch}-prefill-{prefill}",
                      "ratio": ratio, "hidden": hidden, "head": head, "batch": batch, "total": total,
                      "kv_weight": kv.flatten().tolist(), "gate_weight": None if gate is None else gate.flatten().tolist(),
                      "norm_weight": norm.tolist(), "input": tokens.flatten().tolist(), "steps": steps,
                      "full_expected": full.flatten().tolist()})
    return {"model": MODEL, "revision": REVISION, "source_sha256": SOURCE_SHA256,
            "precision": "unquantized_fp32", "atol_exclusive": 2e-7, "rtol": 0, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    fixture = make_fixture(load_reference(args.reference_path))
    if args.write_fixture:
        metadata = {key: value for key, value in fixture.items() if key != "cases"}
        prefix = json.dumps(metadata, indent=2)[:-2]
        cases = ",\n".join("    " + json.dumps(case, allow_nan=False, separators=(",", ":")) for case in fixture["cases"])
        FIXTURE.write_text(prefix + ',\n  "cases": [\n' + cases + "\n  ]\n}\n", encoding="utf-8")
    elif json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
        raise RuntimeError("compressor fixtures differ from the pinned reference")
    elements = sum(len(case["full_expected"]) + sum(len(step["expected"]) for step in case["steps"]) for case in fixture["cases"])
    print(f"PASS: {len(fixture['cases'])} FP32 reference compressor streams, {elements} latent values; {REVISION}")


if __name__ == "__main__":
    main()
