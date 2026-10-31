"""Complete CPU PyTorch oracle with packed attention, indexer and MoE.

Released Attention/Indexer/Block/Expert/MoE forward bodies are unchanged.
Packed GEMMs use the existing block-scaled CPU mirror and BF16 outputs. The
sparse kernel mirror accumulates in FP32 and returns the query dtype. Dense
BF16 linears explicitly round their input; the residual/compressor/Engram
backbone remains FP32. The wo_b wrapper promotes its BF16 GEMM result to FP32
at that hybrid boundary. Seeds and geometry are fixed before native comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import validate_deepseek_v41_model as model
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_mixed_model as packed_model
from trace_deepseek_v41_model import Observer

FIXTURE = model.FIXTURE.parent / "deepseek_v41_attention_model.json"
SEEDS = [4141051, 4141052]


class Bf16Linear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().bfloat16(), requires_grad=False)

    def forward(self, x):
        return F.linear(x.bfloat16(), self.weight)


class StreamOutputLinear(mixed.Linear):
    def forward(self, x):
        return super().forward(x).float()


def sparse_attn(q, kv, sink, indices, scale):
    return model.sparse_attn(q.float(), kv.float(), sink, indices, scale).to(q.dtype)


def make_case(path, kernel_path, seed, topk, *, observer=None):
    ns = mixed.source_namespace(path)
    packed_tensors = {}
    bf16_tensors = set()

    def transform(net):
        namespace = type(net.layers[0].attn).forward.__globals__
        namespace["sparse_attn"] = sparse_attn
        for layer in net.layers:
            prefix = f"layers.{layer.layer_id}"
            attention = layer.attn
            for name in ("wq_a", "wq_b", "wkv", "wo_b"):
                spec = packed_model.quantize_weight(getattr(attention, name).weight, "fp8_block32")
                packed_tensors[f"{prefix}.attn.{name}.weight"] = spec
                setattr(attention, name, StreamOutputLinear(spec) if name == "wo_b" else mixed.Linear(spec))
            attention.wo_a.weight = nn.Parameter(attention.wo_a.weight.detach().bfloat16(), requires_grad=False)
            bf16_tensors.add(f"{prefix}.attn.wo_a.weight")
            if attention.indexer is not None:
                indexer = attention.indexer
                spec = packed_model.quantize_weight(indexer.wq_b.weight, "fp8_block32")
                indexer.wq_b = mixed.Linear(spec)
                packed_tensors[f"{prefix}.attn.indexer.wq_b.weight"] = spec
                for name in (("weights_proj", "wk") if indexer.owns_k else ("weights_proj",)):
                    setattr(indexer, name, Bf16Linear(getattr(indexer, name).weight))
                    bf16_tensors.add(f"{prefix}.attn.indexer.{name}.weight")
                if indexer.owns_k:
                    indexer.k_cache = indexer.k_cache.bfloat16()
            moe = layer.ffn
            for index, expert in enumerate(moe.experts):
                spec = {name: packed_model.quantize_weight(getattr(expert, name).weight, "fp4_row32")
                        for name in ("w1", "w2", "w3")}
                moe.experts[index] = mixed.expert(ns, spec, expert.swiglu_limit)
                for name, weight in spec.items():
                    packed_tensors[f"{prefix}.ffn.experts.{index}.{name}.weight"] = weight
            expert = moe.shared_experts
            spec = {name: packed_model.quantize_weight(getattr(expert, name).weight, "fp8_block32")
                    for name in ("w1", "w2", "w3")}
            moe.shared_experts = mixed.expert(ns, spec, expert.swiglu_limit)
            for name, weight in spec.items():
                packed_tensors[f"{prefix}.ffn.shared_experts.{name}.weight"] = weight

    f = model.make_fixture(path, kernel_path, observer, seed=seed, topk=topk,
                           args_overrides={"q_lora_rank": 32, "o_lora_rank": 32, "moe_inter_dim": 32},
                           model_transform=transform)
    f["config"]["text_config"].update(q_lora_rank=32, o_lora_rank=32, moe_intermediate_size=32)
    f["precision"] = "cpu_pytorch_packed_attention_indexer_moe_bf16_boundaries_fp32_residual_backbone"
    f["packed_tensors"] = packed_tensors
    for name in bf16_tensors:
        f["tensors"][name]["dtype"] = "bf16"
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path", type=Path, required=True)
    p.add_argument("--kernel-reference-path", type=Path, required=True)
    p.add_argument("--write-fixture", action="store_true")
    p.add_argument("--native-output", type=Path)
    p.add_argument("--trace-output", type=Path)
    args = p.parse_args()
    observers = [Observer() if args.trace_output else None for _ in SEEDS]
    f = dict(cases=[make_case(args.reference_path, args.kernel_reference_path, seed, topk, observer=observer)
                    for seed, topk, observer in zip(SEEDS, [2, 3], observers)])
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(f, separators=(",", ":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f:
        raise AssertionError("packed-attention fixture differs from the independent CPU reference")
    if args.trace_output:
        args.trace_output.write_text(json.dumps(dict(cases=[dict(seed=seed, steps=observer.steps)
                                           for seed, observer in zip(SEEDS, observers)]), allow_nan=False))
    print(f"PASS packed attention/indexer model CPU oracle: seeds {SEEDS}, 12 full-logit steps")
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        assert len(native["cases"]) == len(f["cases"])
        binding = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        for expected, actual, observer in zip(f["cases"], native["cases"], observers):
            assert expected["seed"] == actual["seed"]
            print(f"PACKED ATTENTION seed {expected['seed']}")
            model.compare_native(expected, actual, fixture_sha256=binding)
            if observer is not None:
                compared = 0
                for reference_step, native_step in zip(observer.steps, actual["steps"]):
                    reference = {(s["layer"], s["stage"]): s["values"] for s in reference_step["stages"]}
                    for stage in native_step["stages"]:
                        if stage["stage"] not in {"attention_projection_input", "attention_grouped", "attention"}:
                            continue
                        a = torch.tensor(stage["values"], dtype=torch.float32)
                        b = torch.tensor(reference[(stage["layer"], stage["stage"])], dtype=torch.float32)
                        if not a.numel() or a.shape != b.shape or not torch.equal(a.view(torch.uint32), b.view(torch.uint32)):
                            raise AssertionError(f"packed attention stage differs: {native_step['name']}, {stage['layer']}, {stage['stage']}")
                        compared += a.numel()
                assert compared == 15360, "must compare every attention/grouped/input stage"
                print(f"PASS {compared} attention projection-input/grouped/output values bit-exact")


if __name__ == "__main__":
    main()
