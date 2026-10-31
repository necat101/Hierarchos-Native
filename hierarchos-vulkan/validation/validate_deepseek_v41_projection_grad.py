"""CPU PyTorch projection/LoRA VJPs and a real-gradient AdamW trajectory.

The integration graph is the FP32 V4.1 query-projection subgraph
wq_a -> RMSNorm -> wq_b, followed by a frozen classification head. It qualifies
loss-to-factor plumbing without claiming a complete V4.1 attention/model graph.
Expectations are frozen before native execution and never silently replaced.
"""
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from validate_deepseek_v41_block_grad import bits, deterministic

FIXTURE=Path(__file__).parent/"fixtures/deepseek_v41_projection_grad.json"

def projection(x,w,a,b,scale):
    result=F.linear(x,w)
    if a is not None: result=result+F.linear(F.linear(x,a),b)*scale
    return result

def standalone(name,salt,rows,width,out,rank,alpha,rslora,zero_b=False):
    x=deterministic((rows,width),salt,0.5).requires_grad_()
    w=deterministic((out,width),salt+1,0.1)
    g=deterministic((rows,out),salt+2,0.2)
    a=deterministic((rank,width),salt+3,0.1).requires_grad_() if rank else None
    b=(torch.zeros(out,rank) if zero_b else deterministic((out,rank),salt+4,0.1)).requires_grad_() if rank else None
    scale=alpha/(math.sqrt(rank) if rslora else rank) if rank else 1.0
    y=projection(x,w,a,b,scale)
    y.backward(g)
    return dict(name=name,rows=rows,width=width,out=out,rank=rank,alpha=alpha,rslora=rslora,scale=scale,
        x_bits=bits(x),weight_bits=bits(w),a_bits=bits(a) if rank else [],b_bits=bits(b) if rank else [],
        upstream_bits=bits(g),output_bits=bits(y),grad_x_bits=bits(x.grad),
        grad_a_bits=bits(a.grad) if rank else [],grad_b_bits=bits(b.grad) if rank else [])

def training_case(rslora,salt):
    rows,width,latent,out,vocab=4,16,8,16,23
    x=deterministic((rows,width),salt,0.5).requires_grad_()
    w1=deterministic((latent,width),salt+1,0.2)
    w2=deterministic((out,latent),salt+2,0.2)
    norm=1.0+deterministic((latent,),salt+3,0.1)
    head=deterministic((vocab,out),salt+4,0.1)
    targets=torch.tensor([2,3,5,7])
    modules=[]
    for i,(target,rank,inp,nout,alpha) in enumerate([
        ("layers.0.attn.wq_a",2,width,latent,3.0),
        ("layers.0.attn.wq_b",4,latent,out,2.25),
    ]):
        a=deterministic((rank,inp),salt+10+i*2,0.05).requires_grad_()
        b=deterministic((nout,rank),salt+11+i*2,0.04).requires_grad_()
        modules.append(dict(target=target,rank=rank,width=inp,out=nout,alpha=alpha,
            scale=alpha/(math.sqrt(rank) if rslora else rank),a=a,b=b))
    initial=[{k:v for k,v in m.items() if k not in ("a","b")}|
             dict(a_bits=bits(m["a"]),b_bits=bits(m["b"])) for m in modules]
    hyper=dict(learning_rate=0.002,beta1=0.9,beta2=0.999,eps=1e-3,weight_decay=0.01)
    optimizer=torch.optim.AdamW([m[k] for m in modules for k in ("a","b")],
        lr=hyper["learning_rate"],betas=(hyper["beta1"],hyper["beta2"]),eps=hyper["eps"],
        weight_decay=hyper["weight_decay"],foreach=False)
    def forward():
        m1,m2=modules
        h=projection(x,w1,m1["a"],m1["b"],m1["scale"])
        h=(h*torch.rsqrt(h.square().mean(-1,keepdim=True)+1e-20))*norm
        q=projection(h,w2,m2["a"],m2["b"],m2["scale"])
        return F.linear(q,head)
    steps=[]
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); x.grad=None
        logits=forward(); loss=F.cross_entropy(logits,targets)
        loss.backward()
        gradients={m["target"]:dict(a_bits=bits(m["a"].grad),b_bits=bits(m["b"].grad)) for m in modules}
        grad_x=bits(x.grad)
        optimizer.step()
        states={}
        for m in modules:
            states[m["target"]]=dict(a_bits=bits(m["a"]),b_bits=bits(m["b"]),
                exp_avg_a_bits=bits(optimizer.state[m["a"]]["exp_avg"]),
                exp_avg_sq_a_bits=bits(optimizer.state[m["a"]]["exp_avg_sq"]),
                exp_avg_b_bits=bits(optimizer.state[m["b"]]["exp_avg"]),
                exp_avg_sq_b_bits=bits(optimizer.state[m["b"]]["exp_avg_sq"]))
        with torch.no_grad(): after=forward()
        steps.append(dict(logits_bits=bits(logits),loss=loss.item(),gradients=gradients,
            grad_x_bits=grad_x,states=states,logits_after_bits=bits(after)))
    assert max(abs(torch.tensor(steps[0]["logits_bits"],dtype=torch.uint32).view(torch.float32)-
                   torch.tensor(steps[-1]["logits_after_bits"],dtype=torch.uint32).view(torch.float32)))>1e-5
    return dict(name="rslora" if rslora else "lora",rows=rows,width=width,latent=latent,out=out,vocab=vocab,
        x_bits=bits(x),w1_bits=bits(w1),w2_bits=bits(w2),norm_bits=bits(norm),head_bits=bits(head),
        targets=targets.tolist(),modules=initial,hyper=hyper,steps=steps,
        adapter_config=dict(peft_type="LORA",task_type="CAUSAL_LM",r=2,lora_alpha=3.0,
            lora_dropout=0.0,target_modules=["wq_a","wq_b"],bias="none",inference_mode=False,
            use_rslora=rslora,rank_pattern={"attn.wq_b":4},alpha_pattern={"attn.wq_b":2.25}))

def main():
    torch.set_num_threads(1)
    payload=dict(tolerance=2e-7,comparison="strict_less_than",rtol=0,torch_version=torch.__version__,
        projection=[standalone(*s) for s in [
            ("single-row",100,1,16,12,2,4.0,False),
            ("two-row-rslora",200,2,32,16,4,3.0,True),
            ("odd-rank-and-width",300,3,15,17,3,2.25,False),
            ("odd-batch-narrow-rank-tail",400,5,16,12,6,3.0,True),
            ("matrix-rank16",500,4,64,96,16,16.0,False),
            ("zero-B",600,4,32,16,4,8.0,False,True),
            ("frozen-no-adapter",700,4,16,32,0,1.0,False),
        ]],training=[training_case(False,1000),training_case(True,2000)])
    if FIXTURE.exists() and json.loads(FIXTURE.read_text())!=payload:
        raise SystemExit("projection-gradient fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(payload,indent=2)+"\n")
    # Preserve the older terminal fixture's explicit row_loss.mean() objective
    # while freezing a separate default-reduction CE expectation on the same
    # immutable logits. Their one-ULP differences are part of the operator API.
    old_path=FIXTURE.with_name("deepseek_v41_terminal_grad.json")
    old=json.loads(old_path.read_text())
    loss_cases=[]
    for name,rows,vocab,targets,logit_bits in [
        ("original-terminal",old["rows"],old["vocab"],old["targets"],old["logits_bits"]),
        ("query-step",payload["training"][0]["rows"],payload["training"][0]["vocab"],payload["training"][0]["targets"],payload["training"][0]["steps"][1]["logits_bits"]),
    ]:
        logits=torch.tensor(logit_bits,dtype=torch.uint32).view(torch.float32).reshape(rows,vocab)
        labels=torch.tensor(targets)
        per_row=F.cross_entropy(logits,labels,reduction="none")
        loss_cases.append(dict(name=name,rows=rows,vocab=vocab,targets=targets,logits_bits=logit_bits,
            row_loss_bits=bits(per_row),row_mean_bits=bits(per_row.mean().reshape(1)),
            reduced_mean_bits=bits(F.cross_entropy(logits,labels).reshape(1))))
    loss_payload=dict(tolerance=2e-7,rtol=0,comparison="strict_less_than",cases=loss_cases)
    loss_path=FIXTURE.with_name("deepseek_v41_loss_reductions.json")
    if loss_path.exists() and json.loads(loss_path.read_text())!=loss_payload:
        raise SystemExit("loss-reduction fixture changed; refusing to overwrite pinned evidence")
    loss_path.write_text(json.dumps(loss_payload,indent=2)+"\n")
    print("PASS CPU projection fixture: 7 projection VJPs; LoRA and rsLoRA, 3 computed-gradient AdamW steps each")

if __name__=="__main__":main()
