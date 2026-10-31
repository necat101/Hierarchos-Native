"""Pinned CPU-mirror qualification for typed linear and packed V4.1 experts.

Released Expert/Gate/MoE forward bodies are unchanged. Packed GEMMs are explicit
CPU block mirrors, with BF16 output and the released cast,
clamp, route-weight and expert accumulation order. No full-production claim.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import validate_deepseek_v41_packed_weights as packed
import validate_deepseek_v41_quantization as quant

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_mixed.json"


def linear_reference(x, weight):
    if weight["dtype"] in ("fp32", "bf16"):
        dtype = torch.float32 if weight["dtype"] == "fp32" else torch.bfloat16
        w = torch.tensor(weight["values"], dtype=dtype).reshape(weight["rows"], weight["cols"])
        return F.linear(x.to(dtype), w)
    n, k = weight["rows"], weight["cols"]
    shape = x.shape
    x = x.float().reshape(-1, k)
    a, sa = quant.quantize_case(x.numpy(), 32, "fp8", "ue8m0")
    sa = torch.from_numpy(sa)
    aq = torch.from_numpy(a) / sa.repeat_interleave(32, dim=1)
    w = packed.unpack(weight["data"], n, k, weight["dtype"], [0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.])
    sw = packed.scale_matrix(weight["scales"], n, k, weight["dtype"])
    y = torch.zeros(x.shape[0], n)
    for start in range(0, k, 32):
        dot = F.linear(aq[:, start:start+32], w[:, start:start+32])
        y += (dot * sa[:, start//32:start//32+1]) * sw[:, start].unsqueeze(0)
    return y.bfloat16().reshape(*shape[:-1], n)


class Linear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.spec = weight
    def forward(self, x):
        return linear_reference(x, self.spec)


def source_namespace(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != packed.MODEL_SHA256:
        raise AssertionError("model source hash mismatch")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in {"Expert", "Gate", "MoE"}]
    ns = dict(torch=torch, nn=nn, F=F, world_size=1, rank=0, linear=F.linear)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def weight(rng, rows, cols, dtype):
    w = dict(rows=rows, cols=cols, dtype=dtype)
    if dtype in ("fp32", "bf16"):
        values = torch.from_numpy(rng.normal(0, .15, (rows, cols)).astype(np.float32))
        if dtype == "bf16": values = values.bfloat16().float()
        w["values"] = values.flatten().tolist()
    else:
        if dtype == "fp4_row32": data = rng.integers(0, 256, rows*cols//2, dtype=np.uint8).tolist()
        else: data = torch.from_numpy(rng.normal(0, 1, (rows,cols)).astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).flatten().tolist()
        sr = rows if dtype == "fp4_row32" else (rows+31)//32
        w.update(data=data, scales=rng.integers(120,125,sr*(cols//32),dtype=np.uint8).tolist())
    return w


def expert(ns, spec, limit):
    result = ns["Expert"].__new__(ns["Expert"])
    nn.Module.__init__(result)
    result.w1, result.w2, result.w3 = (Linear(spec[name]) for name in ("w1", "w2", "w3"))
    result.swiglu_limit = limit
    return result


def moe(ns, spec):
    config = spec["config"]
    result = ns["MoE"].__new__(ns["MoE"])
    nn.Module.__init__(result)
    result.dim, result.n_routed_experts = config["hidden_size"], config["experts"]
    result.experts_start_idx, result.experts_end_idx = 0, config["experts"]
    gate = ns["Gate"].__new__(ns["Gate"])
    nn.Module.__init__(gate)
    gate.dim, gate.topk, gate.score_func = result.dim, config["activated"], config["score_func"]
    gate.gate_temp, gate.norm_topk_prob, gate.route_scale = config["gate_temp"], config["norm_topk_prob"], config["route_scale"]
    gate.weight = nn.Parameter(torch.tensor(spec["gate_weight"]).reshape(config["experts"],result.dim), requires_grad=False)
    gate.bias = nn.Parameter(torch.tensor(spec["gate_bias"]), requires_grad=False)
    gate.bias_vl = None
    result.gate = gate
    result.experts = nn.ModuleList([expert(ns,w,config["swiglu_limit"]) for w in spec["experts"]])
    result.shared_experts = expert(ns,spec["shared"],config["swiglu_limit"])
    return result


def make_fixture(path, kernel_path):
    ns = source_namespace(path)
    quant.load_kernel_source(kernel_path)
    torch.set_num_threads(1)
    rng = np.random.default_rng(4141040)
    cast_input_bits = [0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f807fff,0x3f808000,0x3f808001,
                       0x3f818000,0xbf808000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001]
    cast_tensor = torch.tensor(cast_input_bits,dtype=torch.uint32).view(torch.float32)
    cast = dict(input_bits=cast_input_bits,expected_bits=packed.bits(cast_tensor.bfloat16().float()))
    linears = []
    for dtype, rows, inner, outputs in [("fp32",3,32,8),("bf16",1,64,32),("bf16",17,256,96),("fp8_block32",5,64,35),("fp4_row32",7,96,32)]:
        w = weight(rng, outputs, inner, dtype)
        precision = "fp32" if dtype == "fp32" else "bf16"
        x = torch.from_numpy(rng.normal(0,.7,(rows,inner)).astype(np.float32))
        rounded = x if precision == "fp32" else x.bfloat16()
        y = linear_reference(rounded,w).float()
        linears.append(dict(name=f"{dtype}-m{rows}-k{inner}-n{outputs}", rows=rows, precision=precision, weight=w,
                            input=x.flatten().tolist(), expected_bits=packed.bits(y)))
    cases = []
    specs = [(1,32,32,4,2,"sqrtsoftplus","bf16"),(7,32,64,5,3,"softmax","bf16"),
             (17,64,32,5,3,"sigmoid","bf16"),(5,32,64,3,2,"sqrtsoftplus","fp32")]
    for index,(rows,hidden,inter,experts,topk,score,precision) in enumerate(specs):
        config = dict(hidden_size=hidden,intermediate_size=inter,experts=experts,activated=topk,score_func=score,
                      gate_temp=.75,norm_topk_prob=index!=2,route_scale=-1.25 if index==2 else 1.5,swiglu_limit=0.0 if index==2 else 1.5)
        def make_expert(dtype):
            return dict(w1=weight(rng,inter,hidden,dtype),w2=weight(rng,hidden,inter,dtype),w3=weight(rng,inter,hidden,dtype))
        gate = torch.from_numpy(rng.normal(0,.15,(experts,hidden)).astype(np.float32)).bfloat16().float()
        bias = [3.0-3.0*i for i in range(experts)] if index==0 else rng.normal(0,.07,experts).astype(np.float32).tolist()
        c = dict(name=f"packed-moe-{index}-{precision}",rows=rows,precision=precision,config=config,gate_weight=gate.flatten().tolist(),gate_bias=bias,
                 experts=[make_expert("fp4_row32" if index!=2 else "fp8_block32") for _ in range(experts)],shared=make_expert("fp8_block32"))
        x = torch.from_numpy(rng.normal(0,.7,(rows,hidden)).astype(np.float32))
        rounded = x if precision == "fp32" else x.bfloat16()
        module = moe(ns,c)
        weights, indices = module.gate(rounded)
        y = module(rounded).float()
        c.update(input=x.flatten().tolist(),expected_bits=packed.bits(y),expected_indices=indices.flatten().tolist(),
                 expected_weights=weights.flatten().tolist(),counts=torch.bincount(indices.flatten(),minlength=experts).tolist())
        cases.append(c)
    return dict(revision=quant.REVISION,model_sha256=packed.MODEL_SHA256,kernel_sha256=quant.KERNEL_SHA256,
                torch_revision=torch.version.git_version,seed=4141040,linears=linears,moes=cases,cast=cast)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,required=True)
    p.add_argument("--kernel-reference-path",type=Path,required=True)
    p.add_argument("--write-fixture",action="store_true")
    args = p.parse_args()
    f = make_fixture(args.reference_path,args.kernel_reference_path)
    if args.write_fixture: FIXTURE.write_text(json.dumps(f,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f: raise AssertionError("mixed fixture differs from independent reference")
    print(f"PASS mixed oracle: {len(f['linears'])} typed linear, {len(f['moes'])} packed MoE cases; counts={[c['counts'] for c in f['moes']]}")


if __name__ == "__main__": main()
