"""Frozen CPU AdamW trajectories with None gradients and independent A/B clocks.

Also runs three real mixed-MoE autograd steps through the pinned released
Expert/Gate/MoE bodies and the already qualified packed FP32 STE mirror.
No native outputs are used to create expectations; existing fixtures stay fixed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import validate_deepseek_v41_mixed_moe_grad as moe_grad
from validate_deepseek_v41_packed_grad import ste_round

FIXTURE=Path(__file__).parent/"fixtures/deepseek_v41_conditional_adamw.json"
SEED=4141231
MODULES=[
    ("layers.0.attn.wq_a",2,16,8,3.0),
    ("layers.0.ffn.experts.0.w1",3,8,12,2.0),
    ("layers.0.ffn.experts.1.w1",2,8,12,4.0),
    ("layers.0.ffn.shared_experts.w2",4,12,8,2.25),
    ("layers.0.attn.indexer.weights_proj",2,8,4,1.0),
]

def bits(value):return value.detach().contiguous().view(torch.uint32).flatten().tolist()
def tensor(values,shape):return torch.tensor(values,dtype=torch.uint32).view(torch.float32).reshape(shape)
def deterministic(shape,salt,scale):
    state=(SEED*1103515245+salt)&0x7fffffff
    count=1
    for width in shape:count*=width
    values=[]
    for _ in range(count):
        state=(1103515245*state+12345)&0x7fffffff
        values.append((state/0x7fffffff*2.0-1.0)*scale)
    return torch.tensor(values,dtype=torch.float32).reshape(shape)

def state(optimizer,pairs):
    result={}
    for target,(a,b) in pairs.items():
        sa,sb=optimizer.state.get(a,{}),optimizer.state.get(b,{})
        result[target]=dict(a_bits=bits(a),b_bits=bits(b),
            exp_avg_a_bits=bits(sa.get("exp_avg",torch.zeros_like(a))),
            exp_avg_sq_a_bits=bits(sa.get("exp_avg_sq",torch.zeros_like(a))),
            exp_avg_b_bits=bits(sb.get("exp_avg",torch.zeros_like(b))),
            exp_avg_sq_b_bits=bits(sb.get("exp_avg_sq",torch.zeros_like(b))),
            step_a=int(sa["step"].item()) if sa else 0,step_b=int(sb["step"].item()) if sb else 0)
    return result

def optimizer(pairs,hyper):
    return torch.optim.AdamW([p for pair in pairs.values() for p in pair],
        lr=hyper["learning_rate"],betas=(hyper["beta1"],hyper["beta2"]),eps=hyper["eps"],
        weight_decay=hyper["weight_decay"],foreach=False)

def unit_case(rslora):
    pairs={};modules=[]
    for i,(target,rank,width,out,alpha) in enumerate(MODULES):
        a=deterministic((rank,width),100+i*2,0.2).requires_grad_()
        b=deterministic((out,rank),101+i*2,0.2).requires_grad_()
        pairs[target]=(a,b)
        modules.append(dict(target=target,rank=rank,width=width,out=out,alpha=alpha,a_bits=bits(a),b_bits=bits(b)))
    hyper=dict(learning_rate=0.003,beta1=0.9,beta2=0.999,eps=1e-8,weight_decay=0.1)
    opt=optimizer(pairs,hyper)
    # Missing, explicitly None, present-zero, late-first-use, independently
    # absent A/B and a permanently unused parameter all have distinct paths.
    schedule=[
        {0:("g","g"),3:("g","g")},
        {1:("g",None),3:("z","z")},
        {},
        {0:(None,"g"),1:(None,"g"),3:("g","g")},
        {1:("z","z"),2:("g","g"),3:("z",None)},
        {0:("g",None),1:("g",None),2:(None,"z"),3:(None,"g")},
        {},
        {0:("z","z"),2:("g",None),3:("g","g")},
        {0:(None,"g"),1:("g","g"),2:("g","g")},
    ]
    x=deterministic((2,16),901,0.7);base=deterministic((8,16),902,0.2);head=deterministic((11,8),903,0.1)
    scale=MODULES[0][4]/(MODULES[0][1]**0.5 if rslora else MODULES[0][1])
    def logits():
        a,b=pairs[MODULES[0][0]]
        return F.linear(F.linear(x,base)+scale*F.linear(F.linear(x,a),b),head)
    steps=[]
    for index,active in enumerate(schedule):
        opt.zero_grad(set_to_none=True)
        gradients={}
        for mi,(ka,kb) in active.items():
            target=MODULES[mi][0];a,b=pairs[target]
            ga=None if ka is None else torch.zeros_like(a) if ka=="z" else deterministic(a.shape,1000+index*17+mi*2,0.3)
            gb=None if kb is None else torch.zeros_like(b) if kb=="z" else deterministic(b.shape,1001+index*17+mi*2,0.3)
            a.grad,b.grad=ga,gb
            gradients[target]=dict(a_bits=None if ga is None else bits(ga),b_bits=None if gb is None else bits(gb))
        opt.step()
        steps.append(dict(gradients=gradients,states=state(opt,pairs),logits_bits=bits(logits())))
    return dict(name="rslora" if rslora else "lora",hyper=hyper,modules=modules,
        adapter_config=dict(peft_type="LORA",task_type="CAUSAL_LM",r=MODULES[0][1],lora_alpha=3.0,
            target_modules=[m[0] for m in MODULES],lora_dropout=0.0,bias="none",inference_mode=False,
            use_rslora=rslora,rank_pattern={m[0]:m[1] for m in MODULES},alpha_pattern={m[0]:m[4] for m in MODULES}),
        probe=dict(rows=2,width=16,out=8,vocab=11,x_bits=bits(x),base_bits=bits(base),head_bits=bits(head)),steps=steps)

def mixed_moe_case(path):
    parent=moe_grad.FIXTURE
    base=json.loads(parent.read_text())["cases"][0]
    ns=moe_grad.mixed.source_namespace(path)
    model=moe_grad.mixed.moe(ns,base)
    pairs={}
    for name,expert,spec in [(f"experts.{i}",e,s) for i,(e,s) in enumerate(zip(model.experts,base["experts"]))]+[("shared_experts",model.shared_experts,base["shared"])]:
        for leaf in ["w1","w2","w3"]:
            linear=moe_grad.LinearSTE(spec[leaf],True)
            setattr(expert,leaf,linear)
            pairs[f"layers.0.ffn.{name}.{leaf}"]=(linear.a,linear.b)
    hyper=dict(learning_rate=0.002,beta1=0.9,beta2=0.999,eps=1e-3,weight_decay=0.01)
    opt=optimizer(pairs,hyper)
    def forward():
        x=tensor(base["x_bits"],(base["rows"],base["config"]["hidden_size"])).requires_grad_()
        return ste_round(model(ste_round(x,torch.bfloat16)),torch.bfloat16)
    upstream=tensor(base["upstream_bits"],(base["rows"],base["config"]["hidden_size"]))
    steps=[]
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        y=forward();(y*upstream).sum().backward()
        gradients={target:dict(a_bits=bits(a.grad),b_bits=bits(b.grad)) for target,(a,b) in pairs.items() if a.grad is not None}
        opt.step()
        steps.append(dict(output_bits=bits(y),gradients=gradients,states=state(opt,pairs),updated_output_bits=bits(forward())))
    assert len(steps[0]["gradients"])<len(pairs)
    return dict(parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),case_index=0,
        name=base["name"],hyper=hyper,steps=steps)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,default=Path(__file__).resolve().parents[2]/".tmp-v41-ref/model.py")
    args=p.parse_args();torch.set_num_threads(1)
    result=dict(seed=SEED,torch_version=torch.__version__,torch_revision=torch.version.git_version,
        tolerance=2e-7,rtol=0.0,comparison="strict_less_than",units=[unit_case(False),unit_case(True)],mixed_moe=mixed_moe_case(args.reference_path))
    if FIXTURE.exists() and json.loads(FIXTURE.read_text())!=result:
        raise SystemExit("conditional AdamW fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")
    print("PASS: LoRA/rsLoRA, nine conditional AdamW calls each; three computed-gradient mixed-MoE updates")

if __name__=="__main__":main()
