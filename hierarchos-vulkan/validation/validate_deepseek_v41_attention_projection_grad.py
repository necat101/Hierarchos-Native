"""Freeze the connected attention projection/cache graph against CPU autograd.

Native kernels never supply expected values. Linear storage/PEFT and rounding
follow the existing packed projection contract; the functional cache graph
keeps cross-step gradients live. RMSNorm, complex-table RoPE, the sparse sink
expression and grouped output follow the pinned released attention forward.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from validate_deepseek_v41_block_grad import bits, deterministic
from validate_deepseek_v41_model import sparse_attn
from validate_deepseek_v41_packed_grad import make_weight, packed_forward
from validate_deepseek_v41_quantization import quantize_case
from validate_deepseek_v41_rotary import load_reference, SOURCE_SHA256, REVISION

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures/deepseek_v41_attention_projection_grad.json"
SOURCE = HERE.parents[1] / ".tmp-v41-ref/model.py"


class Boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, frozen_value):
        return frozen_value

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def bf16(x):
    return Boundary.apply(x, x.detach().bfloat16().float())


def quant(x):
    shape = x.shape
    result, _ = quantize_case(x.detach().reshape(-1, shape[-1]).numpy(), 32, "fp8", "ue8m0")
    return Boundary.apply(x, torch.from_numpy(result).reshape(shape))


def case(spec, ordinal):
    name, storage, batch, hidden, qr, heads, dim, groups, rank_out, window, prefill, decode, ratio, rslora = spec
    serial = 8100 + ordinal * 1000
    rng = np.random.default_rng(serial)

    def tensor(shape, scale):
        nonlocal serial
        serial += 1
        return deterministic(shape, serial, scale)

    projections = {}
    factors = {}
    matrices = {}
    for module, n, k in [("wq_a", qr, hidden), ("wq_b", heads*dim, qr),
                         ("wkv", dim, hidden), ("wo_b", hidden, groups*rank_out)]:
        if storage == "packed":
            weight = make_weight(rng, n, k, "fp8_block32")
        else:
            w = tensor((n, k), 0.1)
            if storage == "bf16":
                w = w.bfloat16().float()
            weight = dict(rows=n, cols=k, dtype=storage, values_bits=bits(w))
            matrices[module] = w
        rank = 4
        alpha = 2.0
        scale = float(np.float32(alpha / (np.sqrt(rank) if rslora else rank)))
        a = tensor((rank, k), 0.05).requires_grad_()
        b = tensor((n, rank), 0.05).requires_grad_()
        factors[module] = (a, b)
        projections[module] = dict(weight=weight, rank=rank, scale=scale, alpha=alpha,
                                   a_bits=bits(a), b_bits=bits(b))

    def linear(x, module):
        a, b = factors[module]
        s = projections[module]
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if storage == "packed":
            result = packed_forward(x, s["weight"], True, a, b, s["scale"])
        else:
            result = F.linear(bf16(x) if storage == "bf16" else x, matrices[module])
            if storage == "bf16":
                result = bf16(result)
            result = result + s["scale"] * F.linear(F.linear(x, a), b)
            if storage == "bf16":
                result = bf16(result)
        return result.reshape(*shape[:-1], -1)

    mixed = storage != "fp32"
    cast = bf16 if mixed else lambda x: x
    q_norm = tensor((qr,), 0.2) + 1
    kv_norm = tensor((dim,), 0.2) + 1
    wo_a = tensor((groups, rank_out, heads*dim//groups), 0.1)
    # wo_a is converted dense BF16 in the released package; the FP32 cases
    # remain diagnostic counterparts of the established native FP32 path.
    if mixed:
        wo_a = wo_a.bfloat16().float()
    sink = tensor((heads,), 0.5)
    head = tensor((11, hidden), 0.1)
    precompute, _ = load_reference(SOURCE)
    freqs = precompute(16, 64, 0, 10000.0, 1.0, 32, 1)

    def rope(x, position, inverse=False):
        selected = freqs[position:position+x.shape[1]]
        c, s = selected.real, selected.imag
        if x.ndim == 4:
            c, s = c[None, :, None, :], s[None, :, None, :]
        else:
            c, s = c[None, :, :], s[None, :, :]
        if inverse:
            s = -s
        re, im = x[..., -16::2], x[..., -15::2]
        tail = torch.stack((re*c-im*s, re*s+im*c), -1).flatten(-2)
        return torch.cat((x[..., :-16], tail), -1)

    def norm(x, weight):
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)) * weight

    initial = torch.zeros(batch, window, dim, requires_grad=True)
    cache = initial
    capacity = (prefill+decode)//max(ratio,1) + 1 if ratio else 0
    compressed = tensor((batch, capacity, dim), 0.5).requires_grad_()
    steps = []
    for i, (position, seq) in enumerate([(0, prefill)] + [(p, 1) for p in range(prefill, prefill+decode)]):
        x = tensor((batch, seq, hidden), 0.5).requires_grad_()
        latent = cast(norm(linear(x, "wq_a"), q_norm))
        query = cast(rope(linear(latent, "wq_b").reshape(batch, seq, heads, dim), position))
        window_kv = cast(quant(cast(rope(cast(norm(linear(x, "wkv"), kv_norm)), position))))
        cache = cache.clone()
        for row in range(seq):
            if position > 0 or row >= seq-window:
                cache[:, (position+row) % window] = window_kv[:, row]
        w = window_kv if position == 0 else cache
        width = w.shape[1]
        topk = min(seq, window) if position == 0 else window
        if position == 0:
            wi = [p if p <= row else -1 for row in range(seq)
                  for p in range(max(0,row-window+1), max(0,row-window+1)+topk)]
        else:
            wi = [p if p <= position else -1 for p in [((position % window)+1+s)%window for s in range(window)]]
        wi *= batch
        ready = (position+seq)//ratio if ratio else 0
        ct = min(ready, 2)
        ci = [width + min(slot, (position+row+1)//max(ratio,1)-1)
              if (position+row+1)//max(ratio,1)>0 else -1
              for _ in range(batch) for row in range(seq) for slot in range(ct)]
        idx = torch.cat((torch.tensor(wi).reshape(batch, seq, topk), torch.tensor(ci, dtype=torch.long).reshape(batch, seq, ct)), -1)
        sparse = sparse_attn(query, torch.cat((w, compressed), 1), sink, idx, dim**-0.5)
        rotated = cast(rope(cast(sparse), position, True))
        grouped_input = cast(quant(rotated.reshape(batch, seq, groups, -1)))
        grouped = cast(torch.einsum("bsgd,grd->bsgr", grouped_input, wo_a))
        output = linear(grouped.flatten(2), "wo_b")
        upstream = tensor(output.shape, 0.15)
        # An output may be absent while its KV still receives future gradients.
        seeded = i != 1
        extra = tensor(latent.shape, 0.01) if i == len(range(decode+1))-1 else None
        for value in (latent, query, window_kv, sparse):
            value.retain_grad()
        steps.append(dict(position=position, seq=seq, x_bits=bits(x), query_bits=bits(query),
            latent_bits=bits(latent), window_kv_bits=bits(window_kv), sparse_bits=bits(sparse),
            output_bits=bits(output), logits_bits=bits(F.linear(output, head)), upstream_bits=bits(upstream),
            wi=wi, ci=ci, topk=topk, compressed_topk=ct, compressed_width=ready, seed=seeded,
            extra_latent_bits=bits(extra) if extra is not None else None,
            _x=x, _latent=latent, _q=query, _kv=window_kv, _sparse=sparse, _out=output, _g=upstream, _extra=extra))
    outputs, gradients = [], []
    for s in steps:
        if s["seed"]:
            outputs.append(s["_out"]); gradients.append(s["_g"])
        if s["_extra"] is not None:
            outputs.append(s["_latent"]); gradients.append(s["_extra"])
    torch.autograd.backward(outputs, gradients)
    for s in steps:
        for key, tensor_name in [("grad_x_bits", "_x"), ("grad_latent_bits", "_latent"),
                                  ("grad_query_bits", "_q"), ("grad_window_kv_bits", "_kv"), ("grad_sparse_bits", "_sparse")]:
            value = s[tensor_name].grad
            s[key] = bits(value) if value is not None else None
        for key in list(s):
            if key.startswith("_"):
                del s[key]
    grads = {name: dict(a_bits=bits(a.grad), b_bits=bits(b.grad)) for name,(a,b) in factors.items()}
    return dict(name=name, storage=storage, batch=batch, hidden=hidden, qr=qr, heads=heads, dim=dim,
                groups=groups, output_rank=rank_out, window=window, ratio=ratio, capacity=capacity,
                rslora=rslora, q_norm_bits=bits(q_norm), kv_norm_bits=bits(kv_norm), wo_a_bits=bits(wo_a),
                sink_bits=bits(sink), head_bits=bits(head), compressed_bits=bits(compressed),
                grad_compressed_bits=bits(compressed.grad) if compressed.grad is not None else None,
                grad_initial_window_bits=bits(initial.grad) if initial.grad is not None else None,
                projections=projections, steps=steps, gradients=grads)


def main():
    torch.set_num_threads(1)
    specs = [
        ("fp32-sliding-lora", "fp32", 1, 32, 16, 2, 32, 2, 8, 4, 3, 4, 0, False),
        ("fp32-compressed-batch-rslora", "fp32", 2, 64, 16, 4, 32, 2, 8, 3, 2, 3, 2, True),
        ("bf16-sliding-lora", "bf16", 1, 64, 32, 2, 32, 2, 16, 3, 4, 2, 0, False),
        ("fp8-compressed-rslora", "packed", 1, 64, 32, 2, 32, 1, 32, 4, 3, 3, 2, True),
    ]
    payload = dict(tolerance=2e-7, rtol=0, comparison="strict_less_than", revision=REVISION,
        source_sha256=SOURCE_SHA256, torch_version=torch.__version__, torch_git_version=torch.version.git_version,
        contract="Existing FP32 STE rounding contract, fixed sparse indices, functional cross-step caches",
        dependencies={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in [
            "validate_deepseek_v41_model.py", "validate_deepseek_v41_packed_grad.py", "validate_deepseek_v41_quantization.py"]},
        cases=[case(s,i) for i,s in enumerate(specs)])
    if FIXTURE.exists() and json.loads(FIXTURE.read_text()) != payload:
        raise SystemExit("attention-projection fixture changed; refusing to overwrite frozen CPU evidence")
    FIXTURE.write_text(json.dumps(payload,indent=2)+"\n")
    print("PASS CPU attention projection fixture:",len(payload["cases"]),"graphs,",
          sum(len(c["steps"]) for c in payload["cases"]),"steps;",hashlib.sha256(FIXTURE.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
