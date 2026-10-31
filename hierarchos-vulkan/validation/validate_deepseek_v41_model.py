"""Independent tiny V4.1 text-model oracle and strict full-logit comparator.

Runs unchanged released model classes/forward bodies with FP32 Linear storage.
Unavailable TileLang activation-quantization, Sinkhorn and sparse-attention
kernels have explicit CPU mirrors. Engram conversion artifacts are supplied,
and its table contains post-dequantization BF16 values. This is an independent
FP32 diagnostic reference; the complete mixed-precision CPU graph is separate.
Native output is accepted only by the unchanged <2e-7, rtol=0 comparator.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import json
import hashlib
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

import validate_deepseek_v41_backbone as backbone
import validate_deepseek_v41_engram as engram
import validate_deepseek_v41_quantization as quantization
from deepseek_v41_parity import require_logit_parity

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_model.json"
SEED = 4141017


def config(topk=2):
    value = json.loads((FIXTURE.parent / "deepseek_v41_flash_1017_config.json").read_text())
    value["text_config"].update({
        "vocab_size":16, "hidden_size":32, "moe_intermediate_size":12,
        "num_hidden_layers":6, "num_nextn_predict_layers":0,
        "num_attention_heads":2, "head_dim":32, "qk_rope_head_dim":16,
        "q_lora_rank":8, "o_groups":1, "o_lora_rank":16,
        "sliding_window":4, "max_position_embeddings":16,
        "compress_ratios":[0,2,2,1,1,1], "kv_source_layer_ids":[1,3],
        "index_source_layer_ids":[1,3,5], "index_n_heads":2,
        "index_head_dim":32, "index_topk":2, "candidate_source_layer_id":3,
        "candidate_topk_blocks":2, "candidate_block_size":2,
        "n_routed_experts":3, "n_shared_experts":1, "num_experts_per_tok":topk,
        "routed_scaling_factor":1.5, "swiglu_limit":1.5,
        "hc_mult":2, "hc_sinkhorn_iters":20, "rope_scaling":None,
        "engram_layer_ids":[0,4], "engram_num_embeddings":[128,128],
        "engram_max_ngram_size":3, "engram_vocab_size":5, "engram_n_heads":2,
        "engram_head_dim":4, "engram_pad_token_id":2, "engram_compressed_vocab_size":8,
        "dspark_block_size":0, "dspark_target_layer_ids":[],
    })
    return value


TOKENS = {"token_map":[0,1,2,3,4,5,6,7,3,2,1,0,5,4,7,6],
          "multipliers":[[3,5,7],[11,13,17]], "pad_token_id":2}
STEPS = [
    {"name":"odd-prefill", "pos":0, "seq":5, "ids":[0,1,2,3,4,5,6,7,8,9], "mask":[1,1,0,1,1,1,1,1,1,1]},
    {"name":"group-completing-decode", "pos":5, "seq":1, "ids":[10,11], "mask":[1,1]},
    {"name":"incomplete-group-decode", "pos":6, "seq":1, "ids":[12,13], "mask":[0,1]},
    {"name":"ring-wrap-decode", "pos":7, "seq":1, "ids":[14,15], "mask":[1,1]},
    {"name":"shorter-reset", "pos":0, "seq":1, "ids":[9,8], "mask":[1,0]},
    {"name":"short-prefill-group-completion", "pos":1, "seq":1, "ids":[7,6], "mask":[1,1]},
]


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.register_parameter("bias", nn.Parameter(torch.empty(out_features)) if bias else None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class EngramEmbedding(nn.Module):
    def __init__(self, rows, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, dim))

    def forward(self, ids):
        live = (ids >= 0) & (ids < self.weight.shape[0])
        return F.embedding(ids.clamp(0, self.weight.shape[0]-1), self.weight) * live.unsqueeze(-1)


def activation(x, block_size, mode, scale_mode):
    shape = x.shape
    output, _ = quantization.quantize_case(x.detach().float().reshape(-1, shape[-1]).numpy(), block_size, mode, scale_mode)
    x.copy_(torch.from_numpy(output).reshape(shape))
    return x


def act_quant(x, block_size, scale_fmt, scale_dtype, use_dequant):
    assert use_dequant and scale_fmt == "ue8m0"
    return activation(x, block_size, "fp8", "ue8m0")


def fp4_act_quant(x, block_size, use_dequant, scale_dtype=None):
    assert use_dequant
    return activation(x, block_size, "fp4", "e4m3" if scale_dtype == torch.float8_e4m3fn else "ue8m0")


def sparse_attn(q, kv, sink, indices, scale):
    # Independent FP32 gather/softmax mirror of the released kernel. The sink
    # participates only in the denominator and has an implicit zero value.
    batch = torch.arange(q.shape[0])[:, None, None]
    gathered = kv[batch, indices.clamp_min(0).long()]
    scores = torch.einsum("bshd,bskd->bshk", q, gathered) * scale
    scores.masked_fill_(indices.unsqueeze(2) < 0, -torch.inf)
    scores = torch.cat((scores, sink[None,None,:,None].expand(*scores.shape[:-1],1)), -1)
    weights = scores.softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", weights, gathered)


def load_reference(path, kernel_path, *, hash_config=None, tokens=None):
    source = backbone.load_source(path)
    quantization.load_kernel_source(kernel_path)
    hash_namespace = engram.load_reference(path)
    classes = {"ModelArgs", "ParallelEmbedding", "RMSNorm", "Engram", "Compressor", "Indexer",
               "Attention", "Gate", "Expert", "MoE", "Block", "ParallelHead", "SharedAttentionRuntime", "Transformer"}
    functions = {"set_dtype", "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs",
                 "select_candidate_blocks", "make_identity_pre_mix", "sample"}
    nodes = [n for n in ast.parse(source).body if
             (isinstance(n,ast.ClassDef) and n.name in classes) or
             (isinstance(n,ast.FunctionDef) and n.name in functions)]
    assert {n.name for n in nodes} == classes | functions
    namespace = dict(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass, Literal=Literal,
        contextmanager=contextmanager, lru_cache=lru_cache, world_size=1, rank=0, default_dtype=torch.float32,
        Linear=Linear, ColumnParallelLinear=Linear, RowParallelLinear=Linear,
        ParallelEngramEmbedding=EngramEmbedding, EngramLayout=hash_namespace["EngramLayout"],
        fp8_block_size=32, fp4_block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu,
        act_quant=act_quant, fp4_act_quant=fp4_act_quant, sparse_attn=sparse_attn,
        hc_split_sinkhorn=backbone.split_sinkhorn, dist=torch.distributed,
        IMAGE=0, IMAGE_START=1, IMAGE_END=2, IMAGE_NEW_LINE=3)
    hc = dict(layer_ids=[0,4], num_embeddings=[128,128], max_ngram_size=3, vocab_size=5,
              n_heads=2, head_dim=4, max_batch_size=2, max_seq_len=16)
    hc.update(hash_config or {})
    token_artifacts = tokens or TOKENS
    namespace["NgramHashState"] = lambda *args: engram.hash_state(hash_namespace,hc,
        token_artifacts["token_map"],token_artifacts["multipliers"],token_artifacts["pad_token_id"])[0]
    exec(compile(ast.Module(body=nodes,type_ignores=[]), "<pinned-v41-model>", "exec"),namespace)
    namespace["linear"] = F.linear
    namespace["shared_attn"] = namespace["SharedAttentionRuntime"]()
    return namespace


@torch.inference_mode()
def make_fixture(path, kernel_path, observer=None, *, seed=SEED, topk=2, args_overrides=None,
                 model_transform=None, steps=None, tokens=None, max_prefill=8, config_overrides=None):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    overrides = args_overrides or {}
    hash_names = {"engram_layer_ids":"layer_ids", "engram_num_embeddings":"num_embeddings",
                  "engram_max_ngram_size":"max_ngram_size", "engram_vocab_size":"vocab_size",
                  "engram_n_heads":"n_heads", "engram_head_dim":"head_dim",
                  "max_batch_size":"max_batch_size", "max_seq_len":"max_seq_len"}
    namespace = load_reference(path,kernel_path, hash_config={hash_names[k]:v for k,v in overrides.items()
                                                           if k in hash_names}, tokens=tokens)
    args = namespace["ModelArgs"](max_batch_size=2,max_seq_len=16,temperature=0,dtype="bf16",expert_dtype=None,
        vocab_size=16,dim=32,moe_inter_dim=12,n_layers=6,n_mtp_layers=0,n_heads=2,
        n_routed_experts=3,n_activated_experts=topk,route_scale=1.5,swiglu_limit=1.5,
        q_lora_rank=8,head_dim=32,rope_head_dim=16,o_groups=1,o_lora_rank=16,window_size=4,
        compress_ratios=(0,2,2,1,1,1),kv_source_layers=(1,3),index_source_layers=(1,3,5),
        compress_rope_theta=160000,original_seq_len=0,index_n_heads=2,index_head_dim=32,index_topk=2,
        candidate_source_layer=3,candidate_topk_blocks=2,candidate_block_size=2,hc_mult=2,
        engram_layer_ids=(0,4),engram_num_embeddings=(128,128),engram_max_ngram_size=3,
        engram_vocab_size=5,engram_n_heads=2,engram_head_dim=4,engram_compressed_vocab_size=8)
    for key,value in (args_overrides or {}).items():
        setattr(args,key,value)
    model = namespace["Transformer"](args)
    rng = torch.Generator().manual_seed(seed)
    for name, parameter in model.named_parameters():
        # Fixed before measuring native drift: nonzero weights and all experts,
        # both compressed ratios, shared selections, Engram and full vocab.
        value = torch.randn(parameter.shape,generator=rng) * 0.05
        if "norm.weight" in name or name.endswith(("q_weight","k_weight")):
            value = 1 + torch.randn(parameter.shape,generator=rng) * 0.02
        elif name.endswith(("hc_attn_scale","hc_ffn_scale")):
            value = torch.tensor([0.8,1.1,0.9])
        elif "hc_" in name:
            value *= 0.4
        elif name == "embed.weight":
            value *= 6
        elif name == "head.weight":
            value *= 1.6
        elif name.endswith("ffn.gate.bias"):
            # Preserve every existing three-expert fixture exactly. Larger
            # routing probes declare the same bias range before native runs.
            value = (torch.tensor([-0.15,0.0,0.15]) if parameter.numel() == 3
                     else torch.linspace(-0.15,0.15,parameter.numel()).reshape(parameter.shape))
        if name in {"embed.weight","norm.weight","head.weight"} or name.endswith("engram.embed.weight"):
            value = value.to(torch.bfloat16).float()
        parameter.copy_(value)
    if model_transform is not None:
        model_transform(model)
    tensors = {name:{"shape":list(value.shape), "values":value.flatten().tolist()} for name,value in model.state_dict().items()}
    if observer is not None:
        observer.attach(model)
    head_inputs = []
    collapsed_inputs = []
    hook = model.head.register_forward_pre_hook(lambda module,inputs:head_inputs.append(inputs[0].clone()))
    norm_hook = model.norm.register_forward_pre_hook(lambda module,inputs:collapsed_inputs.append(inputs[0].clone()))
    outputs = []
    for spec in (STEPS if steps is None else steps):
        if observer is not None:
            observer.begin(spec)
        ids = torch.tensor(spec["ids"]).view(args.max_batch_size,spec["seq"])
        # The released wrapper derives the Engram mask from token_types. There
        # is no VL gate bias in this text-only model, so only the hash/gate uses it.
        types = torch.tensor([(-1 if v else 0) for v in spec["mask"]]).view_as(ids)
        model(ids,spec["pos"],token_types=types)
        captured = head_inputs[-1]
        logits = model.head(captured,full_logits=True)
        if observer is not None:
            observer.end(spec)
        outputs.append(spec | {"logits":logits.flatten().tolist(), "shape":list(logits.shape),
                             "collapsed":collapsed_inputs[-1].flatten().tolist(),"normalized":captured.flatten().tolist()})
    hook.remove()
    norm_hook.remove()
    native_config = config(topk)
    native_config["text_config"].update(config_overrides or {})
    return dict(model=backbone.MODEL,revision=backbone.REVISION,source_sha256=backbone.SOURCE_SHA256,
        engram_sha256=engram.ENGRAM_SHA256,precision="independent_fp32_mirror_with_released_quantize_dequantize_semantics",
        seed=seed, config=native_config, batch_size=args.max_batch_size,max_prefill=max_prefill,
        max_positions=args.max_seq_len,tokens=tokens or TOKENS,tensors=tensors,steps=outputs)


def compare_native(fixture, path, *, fixture_sha256=None):
    native = json.loads(path.read_text()) if isinstance(path,Path) else path
    if native.get("fixture_sha256") != (fixture_sha256 or hashlib.sha256(FIXTURE.read_bytes()).hexdigest()):
        raise AssertionError("native output does not use the identical pinned fixture/weights/inputs")
    if native["revision"] != fixture["revision"] or len(native["steps"]) != len(fixture["steps"]):
        raise AssertionError("native output provenance/step count mismatch")
    worst = 0.0
    failures = []
    for actual,expected in zip(native["steps"],fixture["steps"]):
        if actual["name"] != expected["name"] or actual["shape"] != expected["shape"]:
            raise AssertionError("native step identity/shape mismatch")
        reference = torch.tensor(expected["logits"],dtype=torch.float32).reshape(expected["shape"])
        values = torch.tensor(actual["logits"],dtype=torch.float32).reshape(actual["shape"])
        # Feed native final activations through the independent norm/head to
        # separate accumulated backbone drift from the new vocabulary path.
        if "collapsed" in actual and "normalized" in actual:
            hidden_size = fixture["config"]["text_config"]["hidden_size"]
            head = torch.tensor(fixture["tensors"]["head.weight"]["values"]).reshape(expected["shape"][-1],hidden_size)
            norm = torch.tensor(fixture["tensors"]["norm.weight"]["values"])
            collapsed = torch.tensor(actual["collapsed"]).reshape(*expected["shape"][:-1],hidden_size)
            normalized = torch.tensor(actual["normalized"]).reshape_as(collapsed)
            oracle_norm = collapsed * torch.rsqrt(collapsed.square().mean(-1,keepdim=True)+1e-20) * norm
            if fixture.get("stream_precision") == "bf16":
                oracle_norm = oracle_norm.bfloat16().float()
            expected_hidden = torch.tensor(expected["collapsed"]).reshape_as(collapsed)
            hidden_drift = (collapsed.double()-expected_hidden.double()).abs().max().item()
            norm_drift = (normalized.double()-oracle_norm.double()).abs().max().item()
            head_drift = (values.double()-F.linear(normalized,head).double()).abs().max().item()
            print(f"TRACE {expected['name']}: collapsed={hidden_drift:.17g}, norm-on-identical-hidden={norm_drift:.17g}, head-on-identical-normalized={head_drift:.17g}")
        try:
            drift = require_logit_parity(values,reference,label=expected["name"])
            worst = max(worst,drift)
            print(f"PASS {expected['name']}: {drift:.17g}")
        except AssertionError as error:
            failures.append(str(error))
            print(f"FAIL {error}")
    if failures:
        raise AssertionError(f"{len(failures)} full-logit steps failed the unchanged strict gate")
    print(f"PASS all complete logits: worst {worst:.17g}; production dimensions remain unqualified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path",type=Path)
    parser.add_argument("--kernel-reference-path",type=Path)
    parser.add_argument("--write-fixture",action="store_true")
    parser.add_argument("--native-output",type=Path)
    args = parser.parse_args()
    fixture = make_fixture(args.reference_path,args.kernel_reference_path)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(fixture,allow_nan=False,separators=(",",":")),encoding="utf-8")
    elif json.loads(FIXTURE.read_text()) != fixture:
        raise AssertionError("full-model fixture differs from the pinned independent reference")
    print(f"PASS independent FP32 oracle: {len(fixture['tensors'])} tensors, {len(fixture['steps'])} complete logit steps")
    if args.native_output:
        compare_native(fixture,args.native_output)


if __name__ == "__main__":
    main()
