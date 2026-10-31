"""Released packed Engram table + FP8 projection + BF16 stream CPU mirror.

Uses the previous fixed wider input cases without modifying that fixture.
Only the newly specified projection storage and BF16 boundaries differ.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import validate_deepseek_v41_engram as source
import validate_deepseek_v41_packed_weights as packed
import validate_deepseek_v41_mixed as mixed
from validate_deepseek_v41_mixed_model import quantize_weight

FIXTURE = packed.FIXTURE.parent / "deepseek_v41_mixed_engram.json"


def make_fixture(path,kernel_path):
    mixed.quant.load_kernel_source(kernel_path)
    ns = source.load_reference(path)
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="ParallelEngramEmbedding")
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),ns)
    torch.set_num_threads(1)
    cases = []
    for base in json.loads(packed.FIXTURE.read_text())["engrams"]:
        rows,hc,dim = base["rows"],base["hc"],base["hidden"]
        embed = ns["ParallelEngramEmbedding"].__new__(ns["ParallelEngramEmbedding"])
        nn.Module.__init__(embed)
        embed.weight = nn.Parameter(torch.tensor(base["data"],dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(base["table_rows"],base["head_dim"]),requires_grad=False)
        embed.scale = nn.Parameter(torch.tensor(base["scales"],dtype=torch.uint8).view(torch.float8_e8m0fnu).reshape(base["table_rows"],base["head_dim"]//32),requires_grad=False)
        embed.block_size,embed.vocab_start_idx,embed.vocab_end_idx = 32,0,base["table_rows"]
        module = ns["Engram"].__new__(ns["Engram"])
        nn.Module.__init__(module)
        module.dim,module.hc_mult,module.clamp_value,module.eps = dim,hc,1e-6,1e-20
        module.embed = embed
        projection = quantize_weight(torch.tensor(base["wkv"]).reshape((hc+1)*dim,-1),"fp8_block32")
        module.wkv = mixed.Linear(projection)
        module.q_weight = torch.tensor(base["q_weight"]).reshape(hc,dim)
        module.k_weight = torch.tensor(base["k_weight"]).reshape(hc,dim)
        x = torch.tensor(base["stream"]).reshape(rows,hc,dim).bfloat16()
        hashes = torch.tensor(base["hashes"]).reshape(rows,base["cols"])
        mask = torch.tensor(base["mask"],dtype=torch.bool)
        kv = module.wkv(embed(hashes).flatten(-2)).float()
        output = module(x,hashes,mask).float()
        cases.append(dict(base=base,projection=projection,expected_kv_bits=packed.bits(kv),expected_bits=packed.bits(output)))
    return dict(revision=packed.quant.REVISION,source_sha256=packed.MODEL_SHA256,
                base_fixture_sha256=hashlib.sha256(packed.FIXTURE.read_bytes()).hexdigest(),cases=cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,required=True)
    p.add_argument("--kernel-reference-path",type=Path,required=True)
    p.add_argument("--write-fixture",action="store_true")
    args = p.parse_args()
    f = make_fixture(args.reference_path,args.kernel_reference_path)
    if args.write_fixture: FIXTURE.write_text(json.dumps(f,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f: raise AssertionError("mixed Engram fixture differs from released forward")
    print(f"PASS mixed Engram oracle: {len(f['cases'])} cases from fixed packed-table inputs")


if __name__ == "__main__": main()
