"""Freeze a compact CPU regression for production BF16 grouped GEMV.

Retains the failing channels and their neighbors, at the original K and output
rank. Other output rows are zero. Expected values come from CPU torch.einsum;
no native reduction or result is used. Complete-model acceptance still checks
the original dense checkpoint and all 258,560 unchanged logits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

import validate_deepseek_v41_production_model as production
import trace_deepseek_v41_production_attention as observation

FIXTURE = production.FIXTURE.parent / "deepseek_v41_grouped_reduction.json"
POLICY = "bf16_short_grouped_fp32_k512_v1"
REGRESSION_ROWS = [(3,[708,709,710]),(5,[323,324,325]),(7,[1005,1006,1007])]


def make_fixture(observed):
    torch.set_num_threads(1)
    assert observed["fixture_sha256"] == hashlib.sha256(production.FIXTURE.read_bytes()).hexdigest()
    assert observed["torch_revision"] == torch.version.git_version
    parent = json.loads(production.FIXTURE.read_text())
    root = production.FIXTURE.parent / parent["checkpoint"]["directory"]
    index = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    name = "layers.0.attn.wo_a.weight"
    weights = torch.zeros(3,1024,4096,dtype=torch.bfloat16)
    sparse = []
    bits = lambda x: x.float().contiguous().view(torch.uint32).flatten().tolist()
    with safe_open(str(root/index[name]),framework="pt") as source:
        for group,(original,rows) in enumerate(REGRESSION_ROWS):
            for row in rows:
                values = source.get_slice(name)[original*1024+row:original*1024+row+1,:]
                weights[group,row] = values
                sparse.append(dict(group=group,row=row,bits=bits(values)))
    steps = []
    for step in observed["grouped"]["steps"]:
        x = torch.from_numpy(np.array(step["input_bits"],dtype=np.uint32).view(np.float32)).reshape(8,4096)
        x = x[[g for g,_ in REGRESSION_ROWS]].bfloat16().reshape(1,1,3,4096)
        expected = torch.einsum("bsgd,grd->bsgr",x,weights)
        steps.append(dict(name=step["name"],input_bits=bits(x),expected_bits=bits(expected)))
    return dict(policy=POLICY,torch_revision=torch.version.git_version,
        parent_fixture_sha256=observed["fixture_sha256"],
        original_weight_sha256=observed["grouped"]["weight_sha256"],
        groups=3,group_width=4096,output_rank=1024,weight_rows=sparse,steps=steps)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path)
    p.add_argument("--kernel-reference-path",type=Path)
    p.add_argument("--observation",type=Path)
    p.add_argument("--write-fixture",action="store_true")
    args=p.parse_args()
    observed = json.loads(args.observation.read_text()) if args.observation else observation.trace(
        args.reference_path,args.kernel_reference_path)
    f=make_fixture(observed)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(f,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f:
        raise AssertionError("grouped regression differs from the unchanged CPU einsum")
    print("PASS independent BF16 grouped CPU oracle:",sum(len(s["expected_bits"]) for s in f["steps"]),"outputs")


if __name__ == "__main__":
    main()
