"""CPU autograd gate for retained mixed MoE / packed expert adapter adjoints.

The pinned released Expert/Gate/MoE bodies are executed unchanged. Packed
linear modules supply the already documented FP32 STE for frozen activation
quantization and BF16 casts. This is a component training contract, not a claim
that the released TileLang kernels implement backward or full-model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import validate_deepseek_v41_mixed as mixed
from validate_deepseek_v41_packed_grad import bits, make_weight, packed_forward, ste_round

FIXTURE = Path(__file__).parent / "fixtures/deepseek_v41_mixed_moe_grad.json"
SEED = 4141210


def tensor(rng, shape, scale, grad=False):
    return torch.from_numpy(rng.normal(0, scale, shape).astype(np.float32)).requires_grad_(grad)


class LinearSTE(nn.Module):
    def __init__(self, spec, bf16):
        super().__init__()
        self.spec, self.bf16 = spec, bf16
        self.a = torch.tensor(spec["a_bits"], dtype=torch.uint32).view(torch.float32).reshape(spec["rank"], spec["cols"]).requires_grad_() if spec["rank"] else None
        self.b = torch.tensor(spec["b_bits"], dtype=torch.uint32).view(torch.float32).reshape(spec["rows"], spec["rank"]).requires_grad_() if spec["rank"] else None

    def forward(self, x):
        # Expert.forward casts the weighted activation to the stream dtype
        # before w2. The mirror carries BF16 values in FP32 tensors, so lift
        # that cast explicitly before both the packed and LoRA branches.
        if self.bf16:
            x = ste_round(x, torch.bfloat16)
        self.last_input = x
        x.retain_grad()
        return packed_forward(x, self.spec, self.bf16, self.a, self.b, self.spec["scale"])


def make_case(ns, rng, name, rows, hidden, inter, experts, topk, score, bf16, norm, route_scale, limit, rank, force_empty=False, block128=False):
    config = dict(hidden_size=hidden, intermediate_size=inter, experts=experts,
        activated=topk, score_func=score, gate_temp=0.75, norm_topk_prob=norm,
        route_scale=route_scale, swiglu_limit=limit)

    def projection(out, inner, fmt):
        spec = make_weight(rng, out, inner, fmt)
        spec.update(rank=rank, scale=0.5 if rank else 0.0,
            a_bits=bits(tensor(rng, (rank, inner), 0.05)) if rank else [],
            b_bits=bits(tensor(rng, (out, rank), 0.05)) if rank else [])
        return spec

    def expert(index):
        fmt = "fp8_block128" if block128 else ("fp4_row32" if index % 2 else "fp8_block32")
        return dict(w1=projection(inter, hidden, fmt), w2=projection(hidden, inter, fmt), w3=projection(inter, hidden, fmt))

    gate = tensor(rng, (experts, hidden), 0.15).bfloat16().float()
    bias = ([-9.0, -6.0, 9.0, -12.0] if force_empty else tensor(rng, (experts,), 0.07).tolist())
    spec = dict(name=name, rows=rows, config=config, precision="bf16" if bf16 else "fp32",
        gate_weight=gate.flatten().tolist(), gate_bias=bias,
        experts=[expert(i) for i in range(experts)], shared=expert(0))
    module = mixed.moe(ns, spec)
    linears = {}
    for index, e in enumerate(module.experts):
        for key in ("w1", "w2", "w3"):
            linear = LinearSTE(spec["experts"][index][key], bf16)
            setattr(e, key, linear)
            linears[f"experts.{index}.{key}"] = linear
    for key in ("w1", "w2", "w3"):
        linear = LinearSTE(spec["shared"][key], bf16)
        setattr(module.shared_experts, key, linear)
        linears[f"shared_experts.{key}"] = linear
    observed = {}

    def route_hook(_module, _input, output):
        observed["route"], observed["indices"] = output
        output[0].retain_grad()

    module.gate.register_forward_hook(route_hook)
    x = tensor(rng, (rows, hidden), 0.7, True)
    upstream = tensor(rng, (rows, hidden), 0.2)
    xb = ste_round(x, torch.bfloat16) if bf16 else x
    y = module(xb)
    if bf16:
        y = ste_round(y, torch.bfloat16)
    (y * upstream).sum().backward()
    expected = {}
    for target, linear in linears.items():
        if linear.a is not None:
            expected[target] = None if linear.a.grad is None else dict(a_bits=bits(linear.a.grad), b_bits=bits(linear.b.grad))
    indices = observed["indices"]
    spec.update(x_bits=bits(x), upstream_bits=bits(upstream), output_bits=bits(y),
        indices=indices.flatten().tolist(), route_bits=bits(observed["route"]),
        counts=torch.bincount(indices.flatten(), minlength=experts).tolist(),
        grad_x_bits=bits(x.grad), grad_route_bits=bits(observed["route"].grad), gradients=expected,
        trace={target: dict(input_bits=bits(linear.last_input), grad_input_bits=bits(linear.last_input.grad))
            for target,linear in linears.items() if hasattr(linear,"last_input") and target.endswith("w2")})
    return spec


def swiglu_case(rng, name, rows, inter, visible, shared=False):
    limit = 0.5
    gate = tensor(rng, (rows, inter), 0.7, True)
    up = tensor(rng, (rows, inter), 0.7, True)
    # Exact clamp equality and immediately outside both bounds.
    with torch.no_grad():
        gate[0, :4] = torch.tensor([limit, -limit, limit + 0.25, -limit - 0.25])
        up[0, :4] = torch.tensor([limit, -limit, limit + 0.25, -limit - 0.25])
    route = tensor(rng, (rows,), 1.0, True)
    upstream = tensor(rng, (rows, inter), 0.2)
    dg, du, dr = torch.zeros_like(gate), torch.zeros_like(up), torch.zeros_like(route)
    if visible:
        activation = F.silu(gate[:visible].clamp(max=limit)) * up[:visible].clamp(-limit, limit)
        y = activation if shared else activation * route[:visible, None]
        (y * upstream[:visible]).sum().backward()
        dg, du = gate.grad, up.grad
        if not shared:
            dr = route.grad
    return dict(name=name, rows=rows, inter=inter, visible=visible, shared=shared, limit=limit,
        gate_bits=bits(gate), up_bits=bits(up), route_bits=bits(route), upstream_bits=bits(upstream),
        grad_gate_bits=bits(dg), grad_up_bits=bits(du), grad_route_bits=bits(dr))


def fp32_clamp_case(ns):
    config=dict(hidden_size=4,intermediate_size=4,experts=1,activated=1,
        score_func="softmax",gate_temp=1.0,norm_topk_prob=True,route_scale=1.0,swiglu_limit=0.5)
    diagonal=torch.diag(torch.tensor([0.5,0.75,-0.5,-0.75]))
    def weight(values):
        return dict(rows=4,cols=4,dtype="fp32",values=values.flatten().tolist())
    expert=dict(w1=weight(diagonal),w2=weight(torch.eye(4)),w3=weight(diagonal))
    shared={key:weight(torch.zeros(4,4)) for key in ("w1","w2","w3")}
    spec=dict(config=config,gate_weight=[0.0]*4,gate_bias=[0.0],experts=[expert],shared=shared)
    x=torch.eye(4).requires_grad_()
    upstream=torch.tensor([[0.2,-0.1,0.3,-0.4]]).repeat(4,1)
    y=mixed.moe(ns,spec)(x)
    (y*upstream).sum().backward()
    return dict(x_bits=bits(x),upstream_bits=bits(upstream),w1_bits=bits(diagonal),w2_bits=bits(torch.eye(4)),
        w3_bits=bits(diagonal),output_bits=bits(y),grad_x_bits=bits(x.grad))


def make_fixture(path):
    ns = mixed.source_namespace(path)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    cases = [
        make_case(ns,rng,"single-token-inactive-experts",1,32,32,4,1,"sqrtsoftplus",True,True,1.5,0.125,2,force_empty=True),
        make_case(ns,rng,"softmax-mixed-clamp",4,64,32,5,3,"softmax",True,True,1.5,0.125,4),
        make_case(ns,rng,"sigmoid-fp32-negative-route",5,32,64,5,2,"sigmoid",False,False,-1.25,0.0,3),
        make_case(ns,rng,"sqrtsoftplus-block128",3,128,128,3,2,"sqrtsoftplus",True,True,1.5,1.5,4,block128=True),
        make_case(ns,rng,"zero-route",2,32,32,4,2,"sigmoid",True,True,0.0,0.5,2),
        make_case(ns,rng,"production-route-ratio-48-to-1",2,32,32,48,1,"softmax",True,True,2.0,0.5,2),
        make_case(ns,rng,"frozen-matrix-router",4,32,64,12,4,"sqrtsoftplus",True,False,1.5,0.125,0),
    ]
    return dict(seed=SEED, revision=mixed.quant.REVISION, model_sha256=mixed.packed.MODEL_SHA256,
        torch_version=torch.__version__, torch_revision=torch.version.git_version,
        tolerance=2e-7, rtol=0.0, comparison="strict_less_than",
        training_contract="FP32 STE for frozen activation quantization and BF16 casts; frozen base matrices; zero dropout; fixed discrete indices with differentiable routing scores",
        cases=cases, fp32_clamp=fp32_clamp_case(ns), swiglu=[
            swiglu_case(rng,"clamp-equality-and-padding",4,32,3),
            swiglu_case(rng,"route-sum-cascade",2,544,2),
            swiglu_case(rng,"empty-prefix",3,32,0),
            swiglu_case(rng,"shared-no-route",1,64,1,True),
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path",type=Path,default=Path(__file__).resolve().parents[2]/".tmp-v41-ref/model.py")
    args = parser.parse_args()
    fixture = make_fixture(args.reference_path)
    if FIXTURE.exists() and json.loads(FIXTURE.read_text()) != fixture:
        raise SystemExit("mixed-MoE gradient fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(fixture,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")
    values = sum(len(c["grad_x_bits"])+len(c["grad_route_bits"])+sum(len(g["a_bits"])+len(g["b_bits"]) for g in c["gradients"].values() if g is not None) for c in fixture["cases"])
    print(f"PASS: {len(fixture['cases'])} mixed MoE autograd cases, {values} chain gradients; {len(fixture['swiglu'])} isolated SwiGLU cases")


if __name__ == "__main__":
    main()
