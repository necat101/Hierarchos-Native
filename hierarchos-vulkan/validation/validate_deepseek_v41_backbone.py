"""Qualify V4.1 Hyper-Connections and MoE against the pinned reference.

Extracts the unchanged `Gate`, `Expert`, `MoE`, and `Block` class bodies from
the released `inference/model.py` and executes their forward code with FP32
stand-in Linear modules. `hc_split_sinkhorn` itself is a TileLang kernel, so
this script mirrors its scalar body (sigmoid split, row softmax, column
normalization, then alternating Sinkhorn normalizations) in torch FP32.

The recorded cases cover the stream RMS/split coefficients, collapse and
expand, all three gate score functions with correction-bias-only selection,
normalization and route scaling, and complete MoE outputs with the shared
expert, SwiGLU clamps, and balanced routing whose per-expert visible row count
selects the same dot-operation path as the native kernel.

    python validation/validate_deepseek_v41_backbone.py
    python validation/validate_deepseek_v41_backbone.py --write-fixture
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import types
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from torch import nn

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_backbone.json"

ATOL = 2e-6


def load_source(path: Path | None) -> bytes:
    if path is None:
        with urlopen(
            f"https://huggingface.co/{MODEL}/raw/{REVISION}/inference/model.py",
            timeout=60,
        ) as response:
            source = response.read()
    else:
        source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"reference source hash mismatch: {actual}")
    return source


def split_sinkhorn(mixes, scale, base, hc: int, iterations: int, eps: float):
    """Mirror of the released `hc_split_sinkhorn_kernel` scalar body."""
    mix_hc = (2 + hc) * hc
    assert mixes.shape[-1] == mix_hc
    batch, seq, _ = mixes.shape
    flat = mixes.reshape(-1, mix_hc)
    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(flat[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (flat[:, 2 * hc :].reshape(-1, hc, hc) * scale[2]
            + base[2 * hc :].reshape(hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return (pre.view(batch, seq, hc), post.view(batch, seq, hc),
            comb.view(batch, seq, hc, hc))


def load_reference(path: Path | None):
    """Compile the released Gate/Expert/MoE/Block bodies with FP32 stubs."""
    tree = ast.parse(load_source(path))
    wanted = {"Gate", "Expert", "MoE", "Block"}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("the released source no longer defines the backbone classes once")

    class LinearStub(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.weight)

    def linear(x, weight, bias=None):
        return F.linear(x, weight, bias)

    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "linear": linear,
        "math": __import__("math"),
        "world_size": 1,
        "rank": 0,
        "EngramLayout": object,
        "NgramHashState": object,
        "Aligner": object,
        "ViT": object,
        "Attention": object,
        "IMAGE": 0,
        "IMAGE_END": 1,
        "IMAGE_NEW_LINE": 2,
        "IMAGE_START": 3,
        "hc_split_sinkhorn": lambda mixes, scale, base, hc, iterations, eps: split_sinkhorn(
            mixes, scale, base, hc, iterations, eps
        ),
        "LinearStub": LinearStub,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<reference>", "exec"), namespace)
    return namespace, LinearStub


def hc_case(namespace, name, hc, dim, rows, iterations, eps, norm_eps, seed):
    rng = torch.Generator().manual_seed(seed)
    block = types.SimpleNamespace(
        norm_eps=norm_eps,
        hc_mult=hc,
        hc_sinkhorn_iters=iterations,
        hc_eps=eps,
    )
    hc_mixes = types.MethodType(namespace["Block"].hc_mixes, block)
    hc_pre = types.MethodType(namespace["Block"].hc_pre, block)
    hc_post = types.MethodType(namespace["Block"].hc_post, block)

    mix_hc = (2 + hc) * hc
    # The released methods take [batch, seq, hc, dim].
    stream = torch.randn(rows, 1, hc, dim, generator=rng, dtype=torch.float32) * 0.35
    attn_fn = torch.randn(mix_hc, hc * dim, generator=rng, dtype=torch.float32) * 0.02
    attn_base = torch.randn(mix_hc, generator=rng, dtype=torch.float32) * 0.1
    attn_scale = torch.tensor([0.8, 1.1, 0.9], dtype=torch.float32)
    pre, post, comb = hc_mixes(stream, attn_fn, attn_scale, attn_base)
    collapsed = hc_pre(stream, pre)
    sublayer = torch.randn(rows, 1, dim, generator=rng, dtype=torch.float32) * 0.25
    expanded = hc_post(sublayer, stream, post, comb)
    return {
        "name": name,
        "hc": hc,
        "dim": dim,
        "rows": rows,
        "sinkhorn_iters": iterations,
        "eps": eps,
        "norm_eps": norm_eps,
        "atol": ATOL,
        "stream": stream.flatten().tolist(),
        "attn_fn": attn_fn.flatten().tolist(),
        "attn_base": attn_base.tolist(),
        "attn_scale": attn_scale.tolist(),
        "sublayer": sublayer.flatten().tolist(),
        "expected_pre": pre.flatten().tolist(),
        "expected_post": post.flatten().tolist(),
        "expected_comb": comb.flatten().tolist(),
        "expected_collapsed": collapsed.flatten().tolist(),
        "expected_expanded": expanded.flatten().tolist(),
    }


ROUTE_SPECS = [
    # name, rows, hidden, experts, topk, score, gate_temp, norm, route_scale, bias_scale, seed
    ("sqrtsoftplus-bias", 6, 8, 6, 3, "sqrtsoftplus", 0.9, True, 2.5, 0.6, 4141021),
    ("softmax-raw", 5, 8, 5, 2, "softmax", 1.3, False, 1.0, 0.4, 4141022),
    ("sigmoid-norm", 4, 8, 5, 2, "sigmoid", 1.0, True, 1.4, 0.5, 4141023),
]


def gate_modules(namespace, hidden, experts, weights):
    gate = namespace["Gate"].__new__(namespace["Gate"])
    nn.Module.__init__(gate)
    gate.dim = hidden
    gate.topk = weights["topk"]
    gate.score_func = weights["score"]
    gate.gate_temp = weights["gate_temp"]
    gate.norm_topk_prob = weights["norm_topk"]
    gate.route_scale = weights["route_scale"]
    gate.weight = weights["gate_weight"]
    gate.bias = weights["gate_bias"]
    gate.bias_vl = None
    return gate


def route_case(namespace, name, rows, hidden, experts, topk, score, gate_temp, norm, route_scale, bias_scale, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, hidden, generator=rng, dtype=torch.float32) * 0.4
    values = {
        "topk": topk,
        "score": score,
        "gate_temp": gate_temp,
        "norm_topk": norm,
        "route_scale": route_scale,
        "gate_weight": torch.randn(experts, hidden, generator=rng, dtype=torch.float32) * 0.15,
        "gate_bias": torch.randn(experts, generator=rng, dtype=torch.float32) * bias_scale,
    }
    gate = gate_modules(namespace, hidden, experts, values)
    weights, indices = gate.forward(x)
    return {
        "name": name,
        "rows": rows,
        "hidden": hidden,
        "experts": experts,
        "topk": topk,
        "score": score,
        "gate_temp": gate_temp,
        "norm_topk": norm,
        "route_scale": route_scale,
        "atol": ATOL,
        "x": x.flatten().tolist(),
        "gate_weight": values["gate_weight"].flatten().tolist(),
        "gate_bias": values["gate_bias"].tolist(),
        "expected_indices": indices.flatten().to(torch.int32).tolist(),
        "expected_weights": weights.flatten().tolist(),
    }


def balanced_x(rows: int, hidden: int, seed: int) -> torch.Tensor:
    """Four tokens per expert for the two-expert complete-output cases."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, hidden, generator=rng, dtype=torch.float32) * 0.3
    x[:, 0] = torch.tensor([1.0, -1.0] * (rows // 2), dtype=torch.float32)
    return x


def moe_case(namespace, LinearStub, name, rows, hidden, inter, experts, topk, score,
             gate_temp, norm, route_scale, swiglu_limit, seed):
    rng = torch.Generator().manual_seed(seed)
    x = balanced_x(rows, hidden, seed)
    gate_weight = torch.zeros(experts, hidden, dtype=torch.float32)
    gate_weight[0, 0] = 2.0
    gate_weight[1, 0] = -2.0
    gate_bias = torch.zeros(experts, dtype=torch.float32)
    weights = {
        "topk": topk,
        "score": score,
        "gate_temp": gate_temp,
        "norm_topk": norm,
        "route_scale": route_scale,
        "gate_weight": gate_weight,
        "gate_bias": gate_bias,
    }
    moe = namespace["MoE"].__new__(namespace["MoE"])
    nn.Module.__init__(moe)
    moe.dim = hidden
    moe.n_routed_experts = experts
    moe.n_local_experts = experts
    moe.n_activated_experts = topk
    moe.experts_start_idx = 0
    moe.experts_end_idx = experts
    moe.gate = gate_modules(namespace, hidden, experts, weights)
    expert_w1 = [torch.randn(inter, hidden, generator=rng, dtype=torch.float32) * 0.12 for _ in range(experts)]
    expert_w2 = [torch.randn(hidden, inter, generator=rng, dtype=torch.float32) * 0.12 for _ in range(experts)]
    expert_w3 = [torch.randn(inter, hidden, generator=rng, dtype=torch.float32) * 0.30 for _ in range(experts)]
    shared_w1 = torch.randn(inter, hidden, generator=rng, dtype=torch.float32) * 0.12
    shared_w2 = torch.randn(hidden, inter, generator=rng, dtype=torch.float32) * 0.12
    shared_w3 = torch.randn(inter, hidden, generator=rng, dtype=torch.float32) * 0.30
    experts_list = []
    for index in range(experts):
        expert = namespace["Expert"].__new__(namespace["Expert"])
        nn.Module.__init__(expert)
        expert.w1 = LinearStub(expert_w1[index])
        expert.w2 = LinearStub(expert_w2[index])
        expert.w3 = LinearStub(expert_w3[index])
        expert.swiglu_limit = swiglu_limit
        experts_list.append(expert)
    shared = namespace["Expert"].__new__(namespace["Expert"])
    nn.Module.__init__(shared)
    shared.w1 = LinearStub(shared_w1)
    shared.w2 = LinearStub(shared_w2)
    shared.w3 = LinearStub(shared_w3)
    shared.swiglu_limit = swiglu_limit
    moe.experts = experts_list
    moe.shared_experts = shared
    output = moe.forward(x)
    route_weights, route_indices = moe.gate.forward(x)
    counts = torch.bincount(route_indices.flatten(), minlength=experts).tolist()
    return {
        "name": name,
        "rows": rows,
        "hidden": hidden,
        "inter": inter,
        "experts": experts,
        "topk": topk,
        "score": score,
        "gate_temp": gate_temp,
        "norm_topk": norm,
        "route_scale": route_scale,
        "swiglu_limit": swiglu_limit,
        "atol": ATOL,
        "x": x.flatten().tolist(),
        "gate_weight": gate_weight.flatten().tolist(),
        "gate_bias": gate_bias.tolist(),
        "expert_w1": [value.flatten().tolist() for value in expert_w1],
        "expert_w2": [value.flatten().tolist() for value in expert_w2],
        "expert_w3": [value.flatten().tolist() for value in expert_w3],
        "shared_w1": shared_w1.flatten().tolist(),
        "shared_w2": shared_w2.flatten().tolist(),
        "shared_w3": shared_w3.flatten().tolist(),
        "expected_indices": route_indices.flatten().to(torch.int32).tolist(),
        "expected_weights": route_weights.flatten().tolist(),
        "expected_output": output.flatten().tolist(),
        "routed_counts": counts,
    }


def make_fixture(path: Path | None):
    torch.set_num_threads(1)
    namespace, LinearStub = load_reference(path)
    return {
        "model": MODEL,
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "hc_cases": [
            hc_case(namespace, "hc-matrix", 4, 8, 6, 20, 1e-6, 1e-20, 4141019),
            hc_case(namespace, "hc-vector", 2, 8, 5, 7, 1e-6, 1e-20, 4141020),
        ],
        "route_cases": [
            route_case(namespace, name, rows, hidden, experts, topk, score, gate_temp, norm,
                       route_scale, bias_scale, seed)
            for name, rows, hidden, experts, topk, score, gate_temp, norm, route_scale,
                bias_scale, seed in ROUTE_SPECS
        ],
        "moe_cases": [
            moe_case(namespace, LinearStub, "moe-sqrtsoftplus", 8, 16, 12, 2, 1,
                     "sqrtsoftplus", 1.0, True, 1.0, 0.0, 4141024),
            moe_case(namespace, LinearStub, "moe-softmax-clamped", 8, 16, 12, 2, 1,
                     "softmax", 1.2, True, 1.7, 1.5, 4141025),
            *precision_moe_cases(namespace, LinearStub),
        ],
    }


def precision_moe_cases(namespace, LinearStub):
    cases=[]
    c=moe_case(namespace,LinearStub,"moe-unbalanced-1-3-4",8,16,12,3,1,
               "sqrtsoftplus",1.0,True,1.3,1.5,4141028)
    x=torch.zeros(8,16)
    x[0,0]=1
    x[1:4,1]=1
    x[4:,2]=1
    gate=torch.zeros(3,16)
    gate[0,0]=gate[1,1]=gate[2,2]=4
    c.update(x=x.flatten().tolist(),gate_weight=gate.flatten().tolist())
    cases.append(c)
    c=moe_case(namespace,LinearStub,"moe-top3-expert-order-cancellation",8,8,8,3,3,
               "softmax",1.0,True,3.0,0.0,4141029)
    c.update(rows=1,x=[1.0]+[0.0]*7,gate_weight=[0.0]*24,gate_bias=[1.0,0.0,2.0])
    c["expert_w1"]=[]
    c["expert_w2"]=[]
    c["expert_w3"]=[]
    for target in [1e8,-1e8,1.0]:
        w1=torch.zeros(8,8);w1[0,0]=1
        w3=torch.zeros(8,8);w3[0,0]=1
        w2=torch.zeros(8,8);w2[:,0]=target/F.silu(torch.tensor(1.0))
        for key,w in [("expert_w1",w1),("expert_w2",w2),("expert_w3",w3)]:c[key].append(w.flatten().tolist())
    for key in ["shared_w1","shared_w2","shared_w3"]:c[key]=[0.0]*64
    cases.append(c)
    for c in cases:
        moe=namespace["MoE"].__new__(namespace["MoE"]);nn.Module.__init__(moe)
        moe.dim=c["hidden"];moe.experts_start_idx=0;moe.experts_end_idx=c["experts"]
        moe.n_routed_experts=c["experts"];moe.n_activated_experts=c["topk"]
        cargs=dict(topk=c["topk"],score=c["score"],gate_temp=c["gate_temp"],norm_topk=c["norm_topk"],route_scale=c["route_scale"],
                   gate_weight=torch.tensor(c["gate_weight"]).view(c["experts"],c["hidden"]),gate_bias=torch.tensor(c["gate_bias"]))
        moe.gate=gate_modules(namespace,c["hidden"],c["experts"],cargs)
        def expert(w1,w2,w3):
            e=namespace["Expert"].__new__(namespace["Expert"]);nn.Module.__init__(e)
            e.w1=LinearStub(torch.tensor(w1).view(c["inter"],c["hidden"]))
            e.w2=LinearStub(torch.tensor(w2).view(c["hidden"],c["inter"]))
            e.w3=LinearStub(torch.tensor(w3).view(c["inter"],c["hidden"]))
            e.swiglu_limit=c["swiglu_limit"]
            return e
        moe.experts=[expert(*w) for w in zip(c["expert_w1"],c["expert_w2"],c["expert_w3"])]
        moe.shared_experts=expert(c["shared_w1"],c["shared_w2"],c["shared_w3"])
        x=torch.tensor(c["x"]).view(c["rows"],c["hidden"])
        weights,indices=moe.gate(x)
        c.update(expected_indices=indices.flatten().tolist(),expected_weights=weights.flatten().tolist(),
                 expected_output=moe(x).detach().flatten().tolist(),routed_counts=torch.bincount(indices.flatten(),minlength=c["experts"]).tolist())
    assert cases[0]["routed_counts"]==[1,3,4]
    assert cases[1]["expected_indices"]==[2,0,1]
    assert cases[1]["expected_output"]==[1.0]*8
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    fixture = make_fixture(args.reference_path)
    if args.write_fixture:
        metadata = {key: value for key, value in fixture.items()
                    if key not in {"hc_cases", "route_cases", "moe_cases"}}
        blocks = []
        for key in ("hc_cases", "route_cases", "moe_cases"):
            cases = ",\n".join(
                "    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
                for case in fixture[key]
            )
            blocks.append(f'  "{key}": [\n{cases}\n  ]')
        FIXTURE.write_text(
            json.dumps(metadata, indent=2)[:-2] + ",\n" + ",\n".join(blocks) + "\n}\n",
            encoding="utf-8",
        )
    elif json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
        raise RuntimeError("backbone fixtures differ from the pinned reference")
    hc_values = sum(len(case["expected_comb"]) + len(case["expected_pre"]) for case in fixture["hc_cases"])
    route_values = sum(len(case["expected_weights"]) for case in fixture["route_cases"])
    moe_values = sum(len(case["expected_output"]) for case in fixture["moe_cases"])
    print(
        f"PASS: {len(fixture['hc_cases'])} hyper-connection cases ({hc_values} coefficients), "
        f"{len(fixture['route_cases'])} routing cases ({route_values} weights), "
        f"{len(fixture['moe_cases'])} MoE cases ({moe_values} values); {REVISION}"
    )


if __name__ == "__main__":
    main()
