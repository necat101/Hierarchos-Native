"""Observe released production-width attention from the frozen checkpoint.

Loads only attention parameters and the used vocabulary rows. Forward methods,
CPU sparse mirror and frozen complete-model logits are unchanged. Captures each
step separately, avoiding stale or differently shaped diagnostic comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
from torch import nn

import validate_deepseek_v41_model as base
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_production_model as production


@torch.inference_mode()
def trace(reference_path, kernel_path):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    fixture = json.loads(production.FIXTURE.read_text())
    root = production.FIXTURE.parent / fixture["checkpoint"]["directory"]
    for name, expected in fixture["checkpoint"]["files"].items():
        with (root/name).open("rb") as payload:
            digest = hashlib.file_digest(payload, "sha256").hexdigest()
        assert digest == expected, name
    assert json.loads((root/"config.json").read_text()) == fixture["config"]
    index = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(str(root/index[name]), framework="pt") as source:
            return source.get_tensor(name)

    namespace = base.load_reference(reference_path,kernel_path)
    args = namespace["ModelArgs"](**production.ARGS)
    attention = namespace["Attention"](0,args)
    for name in ("wq_a","wq_b","wkv","wo_b"):
        prefix = f"layers.0.attn.{name}"
        weight, scales = tensor(prefix+".weight"), tensor(prefix+".scale")
        spec = dict(rows=weight.shape[0],cols=weight.shape[1],dtype="fp8_block32",
                    data=weight.view(torch.uint8).flatten().tolist(),
                    scales=scales.view(torch.uint8).flatten().tolist())
        setattr(attention,name,mixed.Linear(spec))
    for name in ("q_norm","kv_norm","wo_a"):
        getattr(attention,name).weight = nn.Parameter(tensor(f"layers.0.attn.{name}.weight"),requires_grad=False)
    attention.attn_sink = nn.Parameter(tensor("layers.0.attn.attn_sink"),requires_grad=False)
    attention.window_kv_cache = attention.window_kv_cache.bfloat16()
    wo_a_sha256 = hashlib.sha256(attention.wo_a.weight.detach().view(torch.uint8).numpy().tobytes()).hexdigest()
    norm = namespace["RMSNorm"](args.dim,args.norm_eps)
    norm.weight = nn.Parameter(tensor("layers.0.attn_norm.weight"),requires_grad=False)
    records = []
    grouped = []
    grouped_inputs = []
    original_quant = namespace["act_quant"]
    bits = lambda x: x.float().contiguous().view(torch.uint32).flatten().tolist()
    def quant(x,*args,**kwargs):
        result = original_quant(x,*args,**kwargs)
        if x.ndim == 4:
            grouped_inputs.append(bits(x))
        return result
    namespace["act_quant"] = quant
    def capture_grouped(module,inputs):
        grouped.append(dict(name=fixture["steps"][len(grouped)]["name"],
            input_bits=grouped_inputs[-1],expected_bits=bits(inputs[0])))
    attention.wo_b.register_forward_pre_hook(capture_grouped)

    def sparse(q,kv,sink,indices,scale):
        qf,kf = q.float(),kv.float()
        gathered = kf[torch.arange(q.shape[0])[:,None,None],indices.clamp_min(0).long()]
        scores = torch.einsum("bshd,bskd->bshk",qf,gathered)*scale
        scores.masked_fill_(indices.unsqueeze(2)<0,-torch.inf)
        full_scores = torch.cat((scores,sink[None,None,:,None].expand(*scores.shape[:-1],1)),-1)
        weights = full_scores.softmax(-1)[...,:-1]
        result = base.sparse_attn(qf,kf,sink,indices,scale)
        observed = torch.einsum("bshk,bskd->bshd",weights,gathered)
        assert torch.equal(result.view(torch.uint32),observed.view(torch.uint32))
        bits = lambda x: x.float().contiguous().view(torch.uint32).flatten().tolist()
        records.append(dict(name=fixture["steps"][len(records)]["name"],
            q_shape=list(q.shape),kv_shape=list(kv.shape),indices_shape=list(indices.shape),
            q_bits=bits(qf),kv_bits=bits(kf),sink_bits=bits(sink),scale=scale,
            indices=indices.flatten().tolist(),score_bits=bits(scores),weight_bits=bits(weights),
            raw_bits=bits(result),rounded_bits=bits(result.to(q.dtype))))
        return result.to(q.dtype)

    namespace["sparse_attn"] = sparse
    inputs = []
    for spec in fixture["steps"]:
        with safe_open(str(root/index["embed.weight"]),framework="pt") as source:
            rows = [source.get_slice("embed.weight")[token:token+1,:] for token in spec["ids"]]
        stream = torch.cat(rows).view(1,spec["seq"],1,args.dim).repeat(1,1,args.hc_mult,1)
        pre = namespace["make_identity_pre_mix"](stream,args.hc_mult)
        collapsed = namespace["Block"].hc_pre(None,stream,pre)
        normalized = norm(collapsed)
        inputs.append(dict(name=spec["name"],values=normalized.float().flatten().tolist()))
        attention(normalized,spec["pos"])
    return dict(fixture_sha256=hashlib.sha256(production.FIXTURE.read_bytes()).hexdigest(),
        checkpoint_files=fixture["checkpoint"]["files"],torch_revision=torch.version.git_version,
        steps=records,attention_inputs=inputs,
        grouped=dict(checkpoint_directory=fixture["checkpoint"]["directory"],
                     weight_sha256=wo_a_sha256,groups=8,group_width=4096,output_rank=1024,
                     steps=grouped))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,required=True)
    p.add_argument("--kernel-reference-path",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--write-fixture",action="store_true")
    args=p.parse_args()
    result=trace(args.reference_path,args.kernel_reference_path)
    frozen = production.FIXTURE.parent / "deepseek_v41_production_attention.json"
    sparse_fixture = {k:result[k] for k in ("fixture_sha256","torch_revision","steps")}
    if args.write_fixture:
        frozen.write_text(json.dumps(sparse_fixture,separators=(",",":"),allow_nan=False))
    elif json.loads(frozen.read_text()) != sparse_fixture:
        raise AssertionError("pinned sparse CPU fixture differs from the unchanged checkpoint replay")
    args.output.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    print("PASS frozen attention observation:",[(s["name"],s["q_shape"],s["indices_shape"]) for s in result["steps"]])


if __name__ == "__main__":
    main()
