"""Pinned CPU PyTorch fixture for the V4.1 Hyper-Connection and MoE adjoints.

The V4.1 block's backward pass, after the terminal stages, runs

    grad_out --hc_post--> grad_x, grad_residual, grad_post, grad_comb
    grad_out --moe_combine--> grad_routed, grad_shared
    grad_routed --expert--> grad_x(slots), grad_route_weight
    grad_shared --shared--> grad_x(tokens)
    grad_route_weight --gate--> grad_x(gate)
    grad_slots + grad_tokens + grad_gate --> grad_x

This fixture pins each stage independently with PyTorch autograd, plus the
chained MoE input gradient. Values are stored as FP32 bit patterns so the JSON
round trip is exact. Regenerate only together with a deliberate kernel change.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 4141107
TOLERANCE = 2e-7
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v41_block_grad.json"


def bits(value: torch.Tensor) -> list[int]:
    return value.contiguous().flatten().view(torch.uint32).tolist()


def deterministic(shape: tuple[int, ...], salt: int, scale: float) -> torch.Tensor:
    """Explicit LCG so the fixture does not depend on torch's RNG stream."""
    count = 1
    for size in shape:
        count *= size
    values = []
    state = (SEED * 1103515245 + salt) & 0x7FFFFFFF
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        values.append((state / 0x7FFFFFFF) * (2.0 * scale) - scale)
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def gate_forward(x, weight, bias, cfg):
    scores = F.linear(x, weight) / cfg["gate_temp"]
    if cfg["score"] == "softmax":
        scores = torch.softmax(scores, dim=-1)
    elif cfg["score"] == "sigmoid":
        scores = torch.sigmoid(scores)
    else:
        scores = F.softplus(scores).sqrt()
    indices = (scores + bias).topk(cfg["topk"], dim=-1)[1]
    weights = scores.gather(1, indices)
    if cfg["norm_topk"] and cfg["topk"] > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * cfg["route_scale"]
    return weights, indices


def expert_forward(x, w1, w3, w2, route, limit):
    gate = F.linear(x, w1)
    up = F.linear(x, w3)
    if limit > 0:
        up = torch.clamp(up, -limit, limit)
        gate = torch.clamp(gate, max=limit)
    activated = F.silu(gate) * up
    if route is not None:
        activated = route * activated
    return F.linear(activated, w2)


def split_sinkhorn(mixes, scale, base, hc: int, iterations: int, eps: float):
    """Mirror of the released `hc_split_sinkhorn_kernel` scalar body."""
    mix_hc = (2 + hc) * hc
    batch, seq, _ = mixes.shape
    flat = mixes.reshape(-1, mix_hc)
    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(flat[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (flat[:, 2 * hc:].reshape(-1, hc, hc) * scale[2]
            + base[2 * hc:].reshape(hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.view(batch, seq, hc), post.view(batch, seq, hc), comb.view(batch, seq, hc, hc)


def ffn_norm_case() -> dict:
    rows, hidden, eps = 4, 8, 1e-6
    x = deterministic((rows, hidden), 20, 1.0).requires_grad_(True)
    weight = deterministic((hidden,), 21, 0.5)
    upstream = deterministic((rows, hidden), 22, 1.0)
    variance = x.pow(2).mean(-1, keepdim=True) + eps
    rstd = torch.rsqrt(variance)
    normalized = (x * rstd) * weight
    grad_x = torch.autograd.grad((normalized * upstream).sum(), x)[0]
    return {
        "rows": rows,
        "hidden": hidden,
        "eps": eps,
        "x_bits": bits(x.detach()),
        "weight_bits": bits(weight),
        "upstream_bits": bits(upstream),
        "grad_x_bits": bits(grad_x),
    }


def hc_pre_case() -> dict:
    rows, hc, dim = 4, 3, 8
    stream = deterministic((rows, 1, hc, dim), 30, 0.6).requires_grad_(True)
    pre = deterministic((rows, 1, hc), 31, 0.4).requires_grad_(True)
    upstream = deterministic((rows, 1, dim), 32, 1.0)
    collapsed = torch.sum(pre.unsqueeze(-1) * stream.float(), dim=2)
    grad_stream, grad_pre = torch.autograd.grad((collapsed * upstream).sum(), (stream, pre))
    return {
        "rows": rows,
        "hc": hc,
        "dim": dim,
        "stream_bits": bits(stream.detach()),
        "pre_bits": bits(pre.detach()),
        "upstream_bits": bits(upstream),
        "grad_stream_bits": bits(grad_stream),
        "grad_pre_bits": bits(grad_pre),
    }


def hc_stage_case(name: str, salt: int, hc: int, dim: int, rows: int, iterations: int,
                  eps: float) -> dict:
    mix_hc = (2 + hc) * hc
    norm_eps = 1e-6
    stream = deterministic((rows, 1, hc, dim), salt + 2, 0.5).requires_grad_(True)
    scale = torch.tensor([0.8, 1.1, 0.9], dtype=torch.float32)
    base = deterministic((mix_hc,), salt + 3, 0.1)
    weight = deterministic((mix_hc, hc * dim), salt + 4, 0.3)
    up_pre = deterministic((rows, 1, hc), salt + 5, 1.0)
    up_post = deterministic((rows, 1, hc), salt + 6, 1.0)
    up_comb = deterministic((rows, 1, hc, hc), salt + 7, 1.0)

    # `raw` is the actual `hc_fn` projection of the stream, so the split and
    # projection adjoints see one consistent forward graph.
    rstd = torch.rsqrt(stream.flatten(2).square().mean(-1, keepdim=True) + norm_eps)
    raw = F.linear(stream.flatten(2), weight)
    mix = raw * rstd
    pre, post, comb = split_sinkhorn(mix, scale, base, hc, iterations, eps)
    loss = ((pre * up_pre).sum() + (post * up_post).sum() + (comb * up_comb).sum())
    grad_mix = torch.autograd.grad(loss, mix)[0]
    grad_stream = torch.autograd.grad(
        (mix * grad_mix.detach()).sum(), stream
    )[0]

    return {
        "name": name,
        "rows": rows,
        "hc": hc,
        "dim": dim,
        "mix_hc": mix_hc,
        "iterations": iterations,
        "eps": eps,
        "norm_eps": norm_eps,
        "raw_bits": bits(raw),
        "stream_bits": bits(stream.detach()),
        "scale": scale.tolist(),
        "base_bits": bits(base),
        "weight_bits": bits(weight),
        "grad_pre_bits": bits(up_pre),
        "grad_post_bits": bits(up_post),
        "grad_comb_bits": bits(up_comb),
        "grad_mix_bits": bits(grad_mix),
        "grad_stream_bits": bits(grad_stream),
    }


def hc_post_case() -> dict:
    rows, hc, dim = 3, 3, 8
    # The released Block.hc_post runs on [batch, seq, hc, dim] tensors, so its
    # `sum(dim=2)` contracts comb's first copy axis (k), not its second (j).
    # Ranks matter to that expression: with a 3-D comb the same line would sum
    # over j instead, which is a different function.
    x = deterministic((rows, 1, dim), 1, 1.0).requires_grad_(True)
    residual = deterministic((rows, 1, hc, dim), 2, 1.0).requires_grad_(True)
    post = deterministic((rows, 1, hc), 3, 0.5).requires_grad_(True)
    comb = deterministic((rows, 1, hc, hc), 4, 0.5).requires_grad_(True)
    upstream = deterministic((rows, 1, hc, dim), 5, 1.0)
    expanded = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    loss = (expanded * upstream).sum()
    grad_x, grad_residual, grad_post, grad_comb = torch.autograd.grad(
        loss, (x, residual, post, comb)
    )
    return {
        "rows": rows,
        "hc": hc,
        "dim": dim,
        "x_bits": bits(x.detach()),
        "residual_bits": bits(residual.detach()),
        "post_bits": bits(post.detach()),
        "comb_bits": bits(comb.detach()),
        "grad_output_bits": bits(upstream),
        "grad_x_bits": bits(grad_x),
        "grad_residual_bits": bits(grad_residual),
        "grad_post_bits": bits(grad_post),
        "grad_comb_bits": bits(grad_comb),
    }


def moe_case(name: str, salt: int, geometry: dict, cfg: dict) -> dict:
    rows = geometry["rows"]
    hidden = geometry["hidden"]
    inter = geometry["inter"]
    experts = geometry["experts"]
    x = deterministic((rows, hidden), salt + 1, 1.0)
    gate_weight = deterministic((experts, hidden), salt + 2, 0.5)
    gate_bias = deterministic((experts,), salt + 3, 0.05)
    expert_w1 = [deterministic((inter, hidden), salt + 10 + 3 * e, 0.5) for e in range(experts)]
    expert_w2 = [deterministic((hidden, inter), salt + 11 + 3 * e, 0.5) for e in range(experts)]
    expert_w3 = [deterministic((inter, hidden), salt + 12 + 3 * e, 0.5) for e in range(experts)]
    shared_w1 = deterministic((inter, hidden), salt + 90, 0.5)
    shared_w2 = deterministic((hidden, inter), salt + 91, 0.5)
    shared_w3 = deterministic((inter, hidden), salt + 92, 0.5)
    upstream = deterministic((rows, hidden), salt + 4, 1.0)

    route_cfg = {"gate_temp": 1.0, "score": cfg["score"], "topk": cfg["topk"],
                 "norm_topk": cfg["norm_topk"], "route_scale": cfg["route_scale"]}

    weights, indices = gate_forward(x, gate_weight, gate_bias, route_cfg)
    limit = cfg["swiglu_limit"]

    # Combine: the routed slots summed in ascending expert order, then shared.
    routed = torch.zeros(rows, cfg["topk"], hidden, requires_grad=True)
    combined = torch.zeros(rows, hidden)
    for expert in range(experts):
        idx, top = torch.where(indices == expert)
        if idx.numel() == 0:
            continue
        combined = combined.index_add(0, idx, routed[idx, top, :])
    shared_out = expert_forward(x, shared_w1, shared_w3, shared_w2, None, limit)
    shared_out = shared_out.clone().requires_grad_(True)
    combined = combined + shared_out
    grad_routed, grad_shared = torch.autograd.grad(
        (combined * upstream).sum(), (routed, shared_out)
    )

    # One routed expert's SwiGLU adjoint, in the slot layout the kernel emits.
    # Pick the most-used expert so the stage is never vacuous.
    counts = torch.bincount(indices.flatten(), minlength=experts)
    stage_expert = int(counts.argmax())
    idx, top = torch.where(indices == stage_expert)
    assert idx.numel() > 0, f"{name}: stage expert has no routed rows"
    gathered = x[idx].detach().requires_grad_(True)
    route_gathered = weights[idx, top].detach().requires_grad_(True)
    out = expert_forward(
        gathered, expert_w1[stage_expert], expert_w3[stage_expert],
        expert_w2[stage_expert], route_gathered.unsqueeze(-1), limit,
    )
    grad_gathered, grad_route_gathered = torch.autograd.grad(
        (out * upstream[idx]).sum(), (gathered, route_gathered)
    )
    expert_grad_input = torch.zeros(rows, cfg["topk"], hidden)
    expert_grad_weight = torch.zeros(rows, cfg["topk"])
    expert_grad_input[idx, top, :] = grad_gathered
    expert_grad_weight[idx, top] = grad_route_gathered

    # Shared expert adjoint.
    shared_x = x.detach().requires_grad_(True)
    shared_only = expert_forward(shared_x, shared_w1, shared_w3, shared_w2, None, limit)
    grad_shared_x = torch.autograd.grad((shared_only * upstream).sum(), shared_x)[0]

    # Chained MoE input gradient, matching the reference MoE.forward order.
    total_x = x.detach().requires_grad_(True)
    total_weights, total_indices = gate_forward(total_x, gate_weight, gate_bias, route_cfg)
    total = torch.zeros(rows, hidden)
    for expert in range(experts):
        idx, top = torch.where(total_indices == expert)
        if idx.numel() == 0:
            continue
        contribution = expert_forward(
            total_x[idx], expert_w1[expert], expert_w3[expert],
            expert_w2[expert], total_weights[idx, top, None], limit,
        )
        total = total.index_add(0, idx, contribution)
    total = total + expert_forward(total_x, shared_w1, shared_w3, shared_w2, None, limit)
    grad_total_x, grad_gate_weights = torch.autograd.grad(
        (total * upstream).sum(), (total_x, total_weights)
    )

    # Gate adjoint, driven by the complete routing-weight gradient the chain
    # produces (one expert's partial gradient would leave most slots zero).
    gate_x = x.detach().requires_grad_(True)
    gate_weights, _ = gate_forward(gate_x, gate_weight, gate_bias, route_cfg)
    grad_gate_x = torch.autograd.grad(
        gate_weights, gate_x, grad_outputs=grad_gate_weights.detach()
    )[0]

    return {
        "name": name,
        "rows": rows,
        "hidden": hidden,
        "inter": inter,
        "experts": experts,
        "topk": cfg["topk"],
        "score": cfg["score"],
        "gate_temp": 1.0,
        "norm_topk": cfg["norm_topk"],
        "route_scale": cfg["route_scale"],
        "swiglu_limit": limit,
        "x_bits": bits(x),
        "gate_weight_bits": bits(gate_weight),
        "gate_bias_bits": bits(gate_bias),
        "expert_w1_bits": [bits(w) for w in expert_w1],
        "expert_w2_bits": [bits(w) for w in expert_w2],
        "expert_w3_bits": [bits(w) for w in expert_w3],
        "shared_w1_bits": bits(shared_w1),
        "shared_w2_bits": bits(shared_w2),
        "shared_w3_bits": bits(shared_w3),
        "grad_output_bits": bits(upstream),
        "indices": indices.flatten().tolist(),
        "weights_bits": bits(weights),
        "combine_grad_routed_bits": bits(grad_routed),
        "combine_grad_shared_bits": bits(grad_shared),
        "stage_expert": stage_expert,
        "expert_grad_input_bits": bits(expert_grad_input),
        "expert_grad_weight_bits": bits(expert_grad_weight),
        "shared_grad_x_bits": bits(grad_shared_x),
        "gate_grad_weights_bits": bits(grad_gate_weights),
        "gate_grad_x_bits": bits(grad_gate_x),
        "total_grad_x_bits": bits(grad_total_x),
    }


def main() -> None:
    torch.set_num_threads(1)
    small = {"rows": 4, "hidden": 8, "inter": 8, "experts": 6}
    matrix = {"rows": 8, "hidden": 16, "inter": 12, "experts": 2}
    wide = {"rows": 4, "hidden": 8, "inter": 8, "experts": 12}
    payload = {
        "seed": SEED,
        "tolerance": TOLERANCE,
        "comparison": "strict_less_than",
        "rtol": 0,
        "ffn_norm": ffn_norm_case(),
        "hc_pre": hc_pre_case(),
        "hc_stages": [
            hc_stage_case("sinkhorn-3x8", 600, 3, 8, 4, 3, 1e-6),
            hc_stage_case("sinkhorn-single-iter", 700, 2, 4, 5, 1, 1e-6),
            hc_stage_case("sinkhorn-wide-eps", 800, 4, 6, 3, 4, 1e-3),
        ],
        "hc_post": hc_post_case(),
        "moe_cases": [
            moe_case("softmax-clamped", 100, small,
                     {"topk": 3, "score": "softmax", "norm_topk": False,
                      "route_scale": 1.0, "swiglu_limit": 0.5}),
            moe_case("sigmoid-normalized", 200, small,
                     {"topk": 3, "score": "sigmoid", "norm_topk": True,
                      "route_scale": 1.5, "swiglu_limit": 0.0}),
            moe_case("sqrtsoftplus-released", 300, small,
                     {"topk": 3, "score": "sqrtsoftplus", "norm_topk": True,
                      "route_scale": 1.5, "swiglu_limit": 10.0}),
            moe_case("matrix-body", 400, matrix,
                     {"topk": 2, "score": "softmax", "norm_topk": False,
                      "route_scale": 1.0, "swiglu_limit": 0.0}),
            moe_case("wide-router", 500, wide,
                     {"topk": 2, "score": "sqrtsoftplus", "norm_topk": False,
                      "route_scale": 1.0, "swiglu_limit": 0.0}),
        ],
    }
    previous = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None
    if previous is not None and previous != payload:
        raise SystemExit(
            "block-gradient fixture would change; refusing to overwrite pinned evidence"
        )
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    counts = {case["name"]: len(case["total_grad_x_bits"]) for case in payload["moe_cases"]}
    print(f"PASS V4.1 block-gradient fixture: hc_post {payload['hc_post']['rows']}x"
          f"{payload['hc_post']['hc']}x{payload['hc_post']['dim']}; moe {counts}")


if __name__ == "__main__":
    main()
