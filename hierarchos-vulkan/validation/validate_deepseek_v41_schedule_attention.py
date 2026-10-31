"""Isolate production-width scheduled attention without changing frozen logits."""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
from torch import nn

import validate_deepseek_v41_model as base
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_schedule_width_model as schedule

FIXTURE = base.FIXTURE.parent / "deepseek_v41_schedule_attention.json"


def generate(source, kernel, trace):
    parent = json.loads(schedule.FIXTURE.read_text())
    binding = hashlib.sha256(schedule.FIXTURE.read_bytes()).hexdigest()
    assert binding == "70d2218429c2d4b8623a8ebbdd6c659eec22f7f64f9eeecd3d95c5d7877fdcf0"
    root = schedule.FIXTURE.parent / parent["checkpoint"]["directory"]
    index = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    def tensor(name):
        with safe_open(str(root/index[name]), framework="pt") as f:
            return f.get_tensor(name)
    bits = lambda t: t.float().contiguous().view(torch.uint32).flatten().tolist()
    stages = {(s["layer"],s["stage"]):torch.tensor(s["values"]) for s in
              json.loads(trace.read_text())["steps"][0]["stages"]}
    ns = base.load_reference(source, kernel)
    args = ns["ModelArgs"](**schedule.ARGS)
    attention = ns["Attention"](2, args)
    for name in ("wq_a", "wq_b", "wkv", "wo_b", "indexer.wq_b"):
        module, leaf = attention, name
        if "." in name:
            prefix, leaf = name.split(".")
            module = getattr(attention, prefix)
        key = f"layers.2.attn.{name}"
        w, s = tensor(key+".weight"), tensor(key+".scale")
        setattr(module,leaf,mixed.Linear(dict(rows=w.shape[0],cols=w.shape[1],dtype="fp8_block32",
                    data=w.view(torch.uint8).flatten(), scales=s.view(torch.uint8).flatten())))
    for name,_ in list(attention.named_parameters()):
        module = attention
        parts = name.split(".")
        for part in parts[:-1]: module = getattr(module,part)
        setattr(module,parts[-1],nn.Parameter(tensor(f"layers.2.attn.{name}"),requires_grad=False))
    attention.window_kv_cache = attention.window_kv_cache.bfloat16()
    attention.compress_kv_cache = attention.compress_kv_cache.bfloat16()
    attention.indexer.k_cache = attention.indexer.k_cache.bfloat16()
    records = []
    def sparse(q,kv,sink,indices,scale):
        qf,kf=q.float(),kv.float()
        gathered=kf[torch.arange(q.shape[0])[:,None,None],indices.clamp_min(0).long()]
        score=torch.einsum("bshd,bskd->bshk",qf,gathered)*scale
        score.masked_fill_(indices.unsqueeze(2)<0,-torch.inf)
        full=torch.cat((score,sink[None,None,:,None].expand(*score.shape[:-1],1)),-1)
        weight=full.softmax(-1)[...,:-1]
        raw=base.sparse_attn(qf,kf,sink,indices,scale)
        assert torch.equal(raw,torch.einsum("bshk,bskd->bshd",weight,gathered))
        records.append(dict(q_shape=list(q.shape),kv_shape=list(kv.shape),indices_shape=list(indices.shape),
            q_bits=bits(qf),kv_bits=bits(kf),sink_bits=bits(sink),indices=indices.flatten().tolist(),
            weight_bits=bits(weight),raw_bits=bits(raw),rounded_bits=bits(raw.bfloat16())))
        return raw.to(q.dtype)
    ns["sparse_attn"]=sparse
    with torch.inference_mode():
        out=attention(stages[2,"attn_norm"].bfloat16().view(1,3,5120),0)
        assert out.float().flatten().tolist()==stages[2,"attention"].tolist()
    # Compact grouped regression: retain real rows and the original N=1024,
    # K=4096 matrix shape, so dispatch/reduction decisions remain unchanged.
    full_weight=tensor("layers.0.attn.wo_a.weight").view(8,1024,4096)
    selected=[(1,407),(4,908),(6,421)]
    weights=torch.zeros_like(full_weight)
    rows=[]
    for group,row in selected:
        for r in range(max(row-1,0),min(row+2,1024)):
            weights[group,r]=full_weight[group,r]
            rows.append(dict(group=group,row=r,bits=bits(full_weight[group,r])))
    x=stages[0,"attention_projection_input"].bfloat16().view(1,3,8,4096)
    grouped=[]
    for n in (1,2,3,4):
        inp=x[:,:n] if n<=3 else torch.cat((x,x[:,:1]),1)
        y=torch.einsum("bsgd,grd->bsgr",inp,weights)
        if n==3:
            original=stages[0,"attention_grouped"].view(1,3,8,1024)
            for row in rows:
                assert y[:,:,row["group"],row["row"]].float().flatten().tolist()==\
                    original[:,:,row["group"],row["row"]].flatten().tolist()
        grouped.append(dict(rows=n,input_bits=bits(inp),expected_bits=bits(y)))
    return dict(parent_fixture_sha256=binding,sparse=records[0],
                grouped=dict(groups=8,width=4096,rank=1024,weight_rows=rows,steps=grouped))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,required=True)
    p.add_argument("--kernel-reference-path",type=Path,required=True)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--write-fixture",action="store_true")
    a=p.parse_args()
    if a.write_fixture and FIXTURE.exists(): raise FileExistsError("Frozen attention expectations")
    torch.set_num_threads(1)
    result=generate(a.reference_path,a.kernel_reference_path,a.trace)
    if a.write_fixture: FIXTURE.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    else: assert json.loads(FIXTURE.read_text())==result
    print("PASS frozen layer-2 sparse and 1/2/3/4-row grouped CPU contractions")


if __name__=="__main__": main()
