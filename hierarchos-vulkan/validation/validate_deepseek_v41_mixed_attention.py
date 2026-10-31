"""CPU PyTorch qualification of packed attention and BF16 index-score arithmetic.

Attention forward and cache methods come from the pinned released source.
The grouped projection, BF16 GEMM/normalization/rotary/output boundaries and
score multiply/sum are compared by their bits, independently of full logits.
"""
import argparse
import json
from pathlib import Path

import torch

import validate_deepseek_v41_attention_model as hybrid
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_model as model
import validate_deepseek_v41_packed_weights as packed
import validate_deepseek_v41_quantization as quant

FIXTURE = model.FIXTURE.parent / "deepseek_v41_mixed_attention.json"
SEED = 4141050


@torch.inference_mode()
def make_fixture(path, kernel_path):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    ns = model.load_reference(path, kernel_path)
    ns["sparse_attn"] = hybrid.sparse_attn
    rng = torch.Generator().manual_seed(SEED)
    attentions = []
    # batch, hidden, q-rank, heads, head-dim, groups, output-rank, input dtype
    for batch, hidden, rank, heads, dim, groups, output_rank, precision in [
        (2, 64, 64, 4, 32, 2, 32, "fp32"),
        (1, 32, 32, 2, 64, 2, 32, "bf16"),
    ]:
        args = ns["ModelArgs"](max_batch_size=batch, max_seq_len=16, dim=hidden, q_lora_rank=rank,
            head_dim=dim, rope_head_dim=16, n_heads=heads, o_groups=groups, o_lora_rank=output_rank,
            n_layers=1, compress_ratios=(0,), kv_source_layers=(), index_source_layers=(), window_size=4)
        attention = ns["Attention"](0, args)
        for name, p in attention.named_parameters():
            p.copy_(torch.randn(p.shape, generator=rng) * .15 + (1.0 if "norm.weight" in name else 0.0))
        weights = {}
        for name in ("wq_a", "wq_b", "wkv", "wo_b"):
            weights[name] = hybrid.packed_model.quantize_weight(getattr(attention, name).weight, "fp8_block32")
            setattr(attention, name, hybrid.StreamOutputLinear(weights[name]) if name == "wo_b" else mixed.Linear(weights[name]))
        attention.wo_a.weight = torch.nn.Parameter(attention.wo_a.weight.detach().bfloat16(), requires_grad=False)
        wo_a = attention.wo_a.weight
        weights["wo_a"] = dict(rows=groups * output_rank, cols=heads // groups * dim,
                               dtype="bf16", values=wo_a.float().flatten().tolist())
        case = dict(batch=batch, hidden=hidden, q_rank=rank, heads=heads, dim=dim, groups=groups,
                    output_rank=output_rank, precision=precision, weights=weights,
                    q_norm=attention.q_norm.weight.tolist(), kv_norm=attention.kv_norm.weight.tolist(),
                    sink=attention.attn_sink.tolist(), steps=[])
        for pos, seq in [(0, 5), (5, 1), (6, 1), (7, 1), (0, 1), (1, 1)]:
            x = torch.randn(batch, seq, hidden, generator=rng) * .7
            if precision == "bf16": x = x.bfloat16()
            qr = attention.q_norm(attention.wq_a(x))
            y = attention(x, pos)
            case["steps"].append(dict(pos=pos, seq=seq, input=x.float().flatten().tolist(),
                query_bits=packed.bits(qr.float()), expected_bits=packed.bits(y.float()),
                cache_bits=packed.bits(attention.window_kv_cache.float())))
        attentions.append(case)

    scores = []
    for batch, seq, pos, heads, dim, ratio in [(2, 5, 0, 4, 32, 2), (2, 1, 7, 32, 128, 2),
                                              (1, 7, 0, 2, 64, 1), (2, 1, 5, 8, 64, 1)]:
        width = (pos + seq) // ratio
        q = torch.randn(batch, seq, heads, dim, generator=rng)
        k = torch.randn(batch, width, dim, generator=rng)
        model.fp4_act_quant(q, 32, True)
        model.fp4_act_quant(k, 32, True)
        q, k = q.bfloat16(), k.bfloat16()
        weights = torch.randn(batch, seq, heads, generator=rng).bfloat16()
        scale = dim ** -.5 * heads ** -.5
        # Exact released Indexer.forward score expressions, including the
        # dtype-preserving scale/product/sum, causal and candidate masks.
        gates = weights * scale
        score = torch.einsum("bshd,btd->bsht", q, k)
        score = (score.relu_() * gates.unsqueeze(-1)).sum(dim=2)
        visible = (torch.arange(1, seq + 1) // ratio).view(1, seq, 1) if pos == 0 else width
        score.masked_fill_(torch.arange(width) >= visible, -torch.inf)
        candidates = torch.rand(batch, seq, width, generator=rng) > .25
        score.masked_fill_(~candidates, -torch.inf)
        scores.append(dict(batch=batch, seq=seq, pos=pos, heads=heads, dim=dim, ratio=ratio,
                           width=width, scale=scale, query=q.float().flatten().tolist(),
                           keys=k.float().flatten().tolist(), weights=weights.float().flatten().tolist(),
                           candidates=candidates.int().flatten().tolist(), expected_bits=packed.bits(score.float())))
    return dict(seed=SEED, revision=quant.REVISION, model_sha256=packed.MODEL_SHA256,
                kernel_sha256=quant.KERNEL_SHA256, attentions=attentions, scores=scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path", type=Path, required=True)
    p.add_argument("--kernel-reference-path", type=Path, required=True)
    p.add_argument("--write-fixture", action="store_true")
    args = p.parse_args()
    f = make_fixture(args.reference_path, args.kernel_reference_path)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(f, separators=(",", ":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f:
        raise AssertionError("mixed-attention fixture differs from the pinned CPU PyTorch reference")
    print(f"PASS mixed attention CPU oracle: {len(f['attentions'])} stateful attention cases, {len(f['scores'])} BF16 score cases")


if __name__ == "__main__":
    main()
