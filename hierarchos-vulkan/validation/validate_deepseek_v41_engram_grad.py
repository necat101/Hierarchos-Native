"""Frozen CPU autograd gate for Engram's normalized gate and projection tape.

Execute the pinned Engram.forward unchanged. Mixed cases lift BF16 casts and
frozen activation quantization with the previously documented FP32 STE. Tables,
hashes and norm weights are frozen; selected embedding values are independent
leaves solely to check the projection-input VJP, not trainable table entries.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import validate_deepseek_v41_engram as source
from validate_deepseek_v41_packed_grad import bits, packed_forward, make_weight, ste_round

FIXTURE=Path(__file__).parent/"fixtures/deepseek_v41_engram_grad.json"
SEED=4141215

def tensor(rng,shape,scale,grad=False):
    return torch.from_numpy(rng.normal(0,scale,shape).astype(np.float32)).requires_grad_(grad)

def from_bits(values,shape):
    return torch.tensor(values,dtype=torch.uint32).view(torch.float32).reshape(shape)

def module(ns,hc,hidden,eps,weight,projection,embed):
    result=ns["Engram"].__new__(ns["Engram"])
    nn.Module.__init__(result)
    result.dim,result.hc_mult,result.eps,result.clamp_value=hidden,hc,eps,1e-6
    result.q_weight=weight
    result.k_weight=torch.ones_like(weight)
    result.wkv,result.embed=projection,embed
    return result

class Constant(nn.Module):
    def __init__(self,kv):
        super().__init__();self.kv=kv
    def forward(self,_):return self.kv

def gate_case(ns,rng,name,rows,hc,hidden,eps=1e-20,precision="fp32",mask=None,kind=None):
    x=tensor(rng,(1,rows,hc,hidden),0.7,True)
    kv=tensor(rng,(1,rows,(hc+1)*hidden),0.5,True)
    weight=tensor(rng,(hc,hidden),0.1)
    if kind=="floor":
        weight*=1e-8
    if kind=="zero":
        with torch.no_grad():x.zero_();kv[:,:,:hc*hidden].zero_()
    if kind=="boundary":
        x= torch.tensor([1.0,-1.0,0.0,-0.0]).reshape(1,4,1,1).requires_grad_()
        kv= torch.tensor([1.0,0.5]*4).reshape(1,4,2).requires_grad_()
        weight=torch.tensor([[1e-6]])
    if kind=="released":
        parent=json.loads(FIXTURE.with_name("deepseek_v41_engram_width.json").read_text())
        assert (rows,hc,hidden)==(parent["rows"],parent["hc"],parent["dim"])
        x=from_bits(parent["stream_bits"],(1,rows,hc,hidden)).requires_grad_()
        kv=from_bits(parent["kv_bits"],(1,rows,(hc+1)*hidden)).requires_grad_()
        weight=from_bits(parent["weight_bits"],(hc,hidden))
        mask=parent["mask"]
    upstream=tensor(rng,(1,rows,hc,hidden),0.2)
    xb=ste_round(x,torch.bfloat16) if precision=="bf16" else x
    m=module(ns,hc,hidden,eps,weight,Constant(kv),nn.Identity())
    mask_tensor=None if mask is None else torch.tensor(mask,dtype=torch.bool).reshape(1,rows)
    y=m(xb,torch.zeros(1,rows,1,1),mask_tensor)
    if precision=="bf16":y=ste_round(y,torch.bfloat16)
    (y*upstream).sum().backward()
    if kind in ("floor","zero","boundary"):
        assert torch.equal(x.grad,upstream)
        assert torch.count_nonzero(kv.grad[:,:,:hc*hidden])==0
    if kind=="released":
        assert bits(y)==parent["expected_bits"]
    return dict(name=name,rows=rows,hc=hc,hidden=hidden,eps=eps,precision=precision,mask=mask,
        stream_bits=bits(x),kv_bits=bits(kv),weight_bits=bits(weight),upstream_bits=bits(upstream),
        output_bits=bits(y),grad_stream_bits=bits(x.grad),grad_kv_bits=bits(kv.grad))

class Lookup(nn.Module):
    def __init__(self,table):super().__init__();self.table=table
    def forward(self,hashes):
        valid=(hashes>=0)&(hashes<len(self.table))
        embedded=F.embedding(hashes.clamp(0,len(self.table)-1),self.table)
        self.selected=embedded.masked_fill(~valid.unsqueeze(-1),0).detach().requires_grad_()
        return self.selected

class Projection(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec
        self.a=from_bits(spec["a_bits"],(spec["rank"],spec["cols"])).requires_grad_() if spec["rank"] else None
        self.b=from_bits(spec["b_bits"],(spec["rows"],spec["rank"])).requires_grad_() if spec["rank"] else None
    def forward(self,x):
        spec=self.spec
        shape=x.shape
        x=x.reshape(-1,spec["cols"])
        if spec["dtype"].startswith("fp8"):
            out=packed_forward(x,spec,True,self.a,self.b,spec["scale"])
        else:
            w=from_bits(spec["values_bits"],(spec["rows"],spec["cols"]))
            bf16=spec["dtype"]=="bf16"
            xb=ste_round(x,torch.bfloat16) if bf16 else x
            out=F.linear(xb,w)
            if bf16:out=ste_round(out,torch.bfloat16)
            if self.a is not None:
                out=out+spec["scale"]*F.linear(F.linear(x,self.a),self.b)
                if bf16:out=ste_round(out,torch.bfloat16)
        self.output=out.reshape(*shape[:-1],spec["rows"])
        self.output.retain_grad()
        return self.output

def chain_case(ns,rng,name,rows,hc,hidden,cols,head_dim,dtype,precision,rank=0):
    table_rows=7
    if dtype=="fp32":
        table=tensor(rng,(table_rows,head_dim),0.5).bfloat16().float()
        table_spec=dict(dtype="dense",values_bits=bits(table),data=[],scales=[])
    else:
        data=tensor(rng,(table_rows,head_dim),1.0).to(torch.float8_e4m3fn).view(torch.uint8)
        scales=rng.integers(123,126,(table_rows,head_dim//32),dtype=np.uint8)
        scale=torch.from_numpy(np.exp2(scales.astype(np.int32)-127).astype(np.float32)).repeat_interleave(32,dim=1)
        table=(data.view(torch.float8_e4m3fn).float()*scale).bfloat16().float()
        table_spec=dict(dtype="fp8_row32",values_bits=bits(table),data=data.flatten().tolist(),scales=scales.flatten().tolist())
    inp,out=cols*head_dim,(hc+1)*hidden
    if dtype.startswith("fp8"):
        spec=make_weight(rng,out,inp,dtype)
    else:
        w=tensor(rng,(out,inp),0.15)
        if dtype=="bf16":w=w.bfloat16().float()
        spec=dict(rows=out,cols=inp,dtype=dtype,values_bits=bits(w),data=[],scales=[])
    spec.update(rank=rank,scale=0.5 if rank else 0.0,
        a_bits=bits(tensor(rng,(rank,inp),0.05)) if rank else [],
        b_bits=bits(tensor(rng,(out,rank),0.05)) if rank else [])
    x=tensor(rng,(1,rows,hc,hidden),0.7,True)
    upstream=tensor(rng,(1,rows,hc,hidden),0.2)
    hashes=torch.tensor(rng.integers(0,table_rows,(1,rows,cols)),dtype=torch.int64)
    # Repeated ids and out-of-range lookup sentinels are frozen/discrete.
    hashes[0,0,:]=2
    if dtype=="fp32":hashes[0,-1,:]=torch.tensor([table_rows,-1])
    mask=[int(i%3!=1) for i in range(rows)]
    lookup=Lookup(table)
    projection=Projection(spec)
    weight=tensor(rng,(hc,hidden),0.1)
    m=module(ns,hc,hidden,1e-20,weight,projection,lookup)
    xb=ste_round(x,torch.bfloat16) if precision=="bf16" else x
    y=m(xb,hashes,torch.tensor(mask,dtype=torch.bool).reshape(1,rows))
    if precision=="bf16":y=ste_round(y,torch.bfloat16)
    (y*upstream).sum().backward()
    return dict(name=name,rows=rows,hc=hc,hidden=hidden,cols=cols,head_dim=head_dim,table_rows=table_rows,
        eps=1e-20,precision=precision,table=table_spec,projection=spec,
        hashes=hashes.to(torch.int32).view(torch.uint32).flatten().tolist(),mask=mask,
        stream_bits=bits(x),weight_bits=bits(weight),upstream_bits=bits(upstream),output_bits=bits(y),
        grad_stream_bits=bits(x.grad),grad_kv_bits=bits(projection.output.grad),
        grad_embedded_bits=bits(lookup.selected.grad),
        grad_a_bits=bits(projection.a.grad) if rank else [],grad_b_bits=bits(projection.b.grad) if rank else [])

def make_fixture(path):
    ns=source.load_reference(path)
    torch.set_num_threads(1)
    rng=np.random.default_rng(SEED)
    return dict(seed=SEED,revision=source.REVISION,source_sha256=source.SOURCE_SHA256,
        engram_sha256=source.ENGRAM_SHA256,torch_version=torch.__version__,torch_revision=torch.version.git_version,
        tolerance=2e-7,rtol=0.0,comparison="strict_less_than",
        mixed_contract="FP32 STE for frozen BF16/activation-quantizer rounding; frozen hashes/table/norms; zero dropout",
        gates=[
            gate_case(ns,rng,"masked-hc2",3,2,8,mask=[1,0,1]),
            gate_case(ns,rng,"odd-width-and-epsilon",5,1,17,eps=1e-3),
            gate_case(ns,rng,"all-masked",2,4,64,mask=[0,0]),
            gate_case(ns,rng,"below-sqrt-floor",3,2,32,kind="floor"),
            gate_case(ns,rng,"zero-norms",2,4,16,eps=1e-3,kind="zero"),
            gate_case(ns,rng,"clamp-boundary-signed-zero",4,1,1,kind="boundary"),
            gate_case(ns,rng,"hc16-vector-sigmoid",3,16,16),
            gate_case(ns,rng,"cascade-width544",2,3,544),
            gate_case(ns,rng,"released-width5120",3,4,5120,precision="bf16",kind="released"),
        ],chains=[
            chain_case(ns,rng,"dense-frozen-invalid-hashes",3,2,16,2,4,"fp32","fp32"),
            chain_case(ns,rng,"bf16-adapter-packed-table",4,4,32,2,32,"bf16","bf16",2),
            chain_case(ns,rng,"fp8-adapter-packed-table",3,2,32,2,32,"fp8_block32","bf16",4),
            chain_case(ns,rng,"fp8-fp32-stream",2,2,32,2,32,"fp8_block32","fp32",2),
        ])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,default=Path(__file__).resolve().parents[2]/".tmp-v41-ref/model.py")
    args=p.parse_args()
    result=make_fixture(args.reference_path)
    if FIXTURE.exists() and json.loads(FIXTURE.read_text())!=result:
        raise SystemExit("Engram-gradient fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")
    gradients=sum(len(c["grad_stream_bits"])+len(c["grad_kv_bits"]) for c in result["gates"])
    gradients+=sum(sum(len(c[k]) for k in ["grad_stream_bits","grad_kv_bits","grad_embedded_bits","grad_a_bits","grad_b_bits"]) for c in result["chains"])
    print(f"PASS: {len(result['gates'])} Engram gate and {len(result['chains'])} lookup/projection chains; {gradients} frozen gradients")

if __name__=="__main__":main()
