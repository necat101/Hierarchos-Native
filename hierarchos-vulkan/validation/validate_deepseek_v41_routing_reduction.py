"""Independent CPU gate regression from the frozen structure checkpoints.

Input stages come from the unchanged CPU model trace. Raw linear scores,
sqrt(softplus), top-6 selection and normalization are evaluated by CPU PyTorch.
No native result is used to generate expectations or overwrite model fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
import torch.nn.functional as F

import validate_deepseek_v41_structure_model as structure

FIXTURE = structure.FIXTURE.parent / "deepseek_v41_routing_reduction.json"
PROBES = [(0,i,0) for i in range(6)] + [(0,1,34),(1,0,0),(1,1,0)]


def generate(trace_path):
    torch.set_num_threads(1)
    parent = json.loads(structure.FIXTURE.read_text())
    traces = json.loads(trace_path.read_text())["cases"]
    assert parent["torch_revision"] == torch.version.git_version
    bits = lambda x: x.float().contiguous().view(torch.uint32).flatten().tolist()
    cases = []
    for ci,si,layer in PROBES:
        case,trace = parent["cases"][ci],traces[ci]
        assert case["seed"] == trace["seed"]
        spec,step = case["steps"][si],trace["steps"][si]
        assert spec["name"] == step["name"]
        root = structure.FIXTURE.parent / case["checkpoint"]["directory"]
        for name,digest in case["checkpoint"]["files"].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
        index = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
        def tensor(name):
            with safe_open(str(root/index[name]),framework="pt") as reader:
                return reader.get_tensor(name).float()
        stage = next(s for s in step["stages"] if s["layer"]==layer and s["stage"]=="ffn_norm")
        x = torch.tensor(stage["values"]).reshape(-1,64)
        weight = tensor(f"layers.{layer}.ffn.gate.weight")
        bias = tensor(f"layers.{layer}.ffn.gate.bias")
        dots = F.linear(x,weight)
        scores = F.softplus(dots).sqrt()
        indices = (scores+bias).topk(6,dim=-1).indices
        raw = scores.gather(1,indices)
        normalized = raw/(raw.sum(-1,keepdim=True)+1e-20)*1.5
        assert bits(normalized) == spec["routing"][str(layer)]["weight_bits"]
        cases.append(dict(name=f"{case['seed']}-{spec['name']}-layer-{layer}",
            rows=x.shape[0],dim=64,experts=weight.shape[0],topk=6,
            input_bits=bits(x),weight_bits=bits(weight),bias_bits=bits(bias),
            dot_bits=bits(dots),indices=indices.flatten().tolist(),
            raw_bits=bits(raw),normalized_bits=bits(normalized)))
    return dict(parent_fixture_sha256=hashlib.sha256(structure.FIXTURE.read_bytes()).hexdigest(),
                torch_revision=torch.version.git_version,cases=cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--write-fixture",action="store_true")
    args=parser.parse_args()
    if args.write_fixture and FIXTURE.exists():
        raise FileExistsError("routing fixture is frozen; repair native execution")
    result=generate(args.trace)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != result:
        raise AssertionError("routing fixture differs from the original CPU gate operations")
    print("PASS independent CPU routing oracle:",len(result["cases"]),"cases")


if __name__ == "__main__":
    main()
