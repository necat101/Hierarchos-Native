"""Freeze/recheck the first production-width Engram gate against its original body.

The fixed projection comes from the independent CPU trace. The unchanged
published Engram.forward is replayed with the original checkpoint q/k weights.
This isolates the reduction bug without running the full four-layer graph.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
from torch import nn

import validate_deepseek_v41_mixed as mixed

FIXTURE = mixed.FIXTURE.parent / "deepseek_v41_engram_width.json"
PARENT = FIXTURE.with_name("deepseek_v41_schedule_width_model.json")


def generate(source_path, trace_path):
    source = source_path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == mixed.packed.MODEL_SHA256
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == "70d2218429c2d4b8623a8ebbdd6c659eec22f7f64f9eeecd3d95c5d7877fdcf0"
    parent = json.loads(PARENT.read_text())
    step = json.loads(trace_path.read_text())["steps"][0]
    assert step["name"] == parent["steps"][0]["name"]
    stages = {s["stage"]: s["values"] for s in step["stages"] if s["layer"] == 0}
    root = PARENT.parent / parent["checkpoint"]["directory"]
    index = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    def tensor(name):
        shard = root/index[name]
        assert hashlib.sha256(shard.read_bytes()).hexdigest() == parent["checkpoint"]["files"][shard.name]
        with safe_open(str(shard), framework="pt") as reader:
            return reader.get_tensor(name)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Engram")
    ns = dict(torch=torch, nn=nn, ModelArgs=object, EngramLayout=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), ns)
    gate = ns["Engram"].__new__(ns["Engram"])
    nn.Module.__init__(gate)
    gate.dim, gate.hc_mult, gate.eps, gate.clamp_value = 5120, 4, 1e-20, 1e-6
    gate.q_weight = nn.Parameter(tensor("layers.0.engram.q_weight"))
    gate.k_weight = nn.Parameter(tensor("layers.0.engram.k_weight"))
    h = torch.tensor(stages["input"]).bfloat16().view(1, 3, 4, 5120)
    kv = torch.tensor(stages["engram_kv"]).bfloat16().view(1, 3, 25600)
    class Projection(nn.Module):
        def forward(self, _): return kv
    gate.embed, gate.wkv = nn.Identity(), Projection()
    mask = torch.tensor(parent["steps"][0]["mask"], dtype=torch.bool).view(1, 3)
    with torch.inference_mode():
        expected = gate(h, torch.zeros(1, 3, 1, 1), mask)
        assert expected.float().flatten().tolist() == stages["engram"]
        weight = gate.q_weight.float()*gate.k_weight.float()
    bits = lambda t: t.float().contiguous().view(torch.uint32).flatten().tolist()
    return dict(parent_fixture_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                source_sha256=mixed.packed.MODEL_SHA256, rows=3, dim=5120, hc=4,
                stream_bits=bits(h), kv_bits=bits(kv), weight_bits=bits(weight),
                mask=mask.int().flatten().tolist(), expected_bits=bits(expected))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--write-fixture", action="store_true")
    args = p.parse_args()
    if args.write_fixture and FIXTURE.exists():
        raise FileExistsError("Engram-width expectations are frozen; repair native execution")
    torch.set_num_threads(1)
    result = generate(args.reference_path, args.trace)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    else:
        assert json.loads(FIXTURE.read_text()) == result
    print("PASS original Engram.forward at width 5120: 61,440 BF16 residual values")


if __name__ == "__main__":
    main()
