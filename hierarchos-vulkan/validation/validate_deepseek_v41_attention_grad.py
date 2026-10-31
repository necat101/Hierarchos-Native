"""Freeze independent CPU PyTorch attention VJPs, never rewriting prior evidence.

The gather/softmax/value expression is the existing V4.1 kernel mirror. Only
FP32 operands and fixed discrete indices are differentiated here; this does not
assign a surrogate derivative to a packed activation quantizer.
"""
import hashlib
import json
from pathlib import Path
import torch
from validate_deepseek_v41_block_grad import bits, deterministic
from validate_deepseek_v41_model import sparse_attn as source_sparse_attn

FIXTURE = Path(__file__).parent / "fixtures/deepseek_v41_attention_grad.json"

def sparse_case(name, salt, batch, seq, heads, dim, window, topk, compressed, capacity):
    q = deterministic((batch,seq,heads,dim),salt,0.5).requires_grad_()
    w = deterministic((batch,window,dim),salt+1,0.5).requires_grad_()
    c = deterministic((batch,capacity,dim),salt+2,0.5).requires_grad_()
    sink = deterministic((heads,),salt+3,0.5)
    upstream = deterministic(q.shape,salt+4,0.5)
    wi, ci = [], []
    for row in range(batch*seq):
        wi.append([(-1 if (slot+row)%5==0 else (slot+row)%window) for slot in range(topk)])
        ci.append([(-1 if (slot+row)%3==0 else window+(slot+row)%capacity) for slot in range(compressed)])
    # Explicit duplicate gathers and a fully masked query (the sink owns it).
    if topk >= 2 and batch*seq > 1: wi[1][:2] = [0,0]
    wi[0] = [-1]*topk
    ci[0] = [-1]*compressed
    indices = torch.tensor([a+b for a,b in zip(wi,ci)],dtype=torch.long).reshape(batch,seq,topk+compressed)
    joined = torch.cat((w,c),dim=1)
    gathered = joined[torch.arange(batch)[:,None,None],indices.clamp_min(0)]
    dots = torch.einsum("bshd,bskd->bshk",q,gathered)
    dots.retain_grad()
    logits = (dots * (dim**-0.5)).masked_fill(indices.unsqueeze(2)<0,-torch.inf)
    probability = torch.cat((logits,sink[None,None,:,None].expand(*logits.shape[:-1],1)),-1).softmax(-1)[...,:-1]
    result = torch.einsum("bshk,bskd->bshd",probability,gathered)
    reference = source_sparse_attn(q,joined,sink,indices,dim**-0.5)
    reference_grads = torch.autograd.grad(reference,(q,w,c),upstream,retain_graph=True)
    assert torch.equal(reference,result), "gradient fixture changed the existing forward mirror"
    result.backward(upstream)
    for actual,expected in zip((q.grad,w.grad,c.grad),reference_grads):
        assert torch.equal(actual,expected), "gradient fixture changed the existing mirror's VJP"
    return dict(name=name,batch=batch,seq=seq,heads=heads,dim=dim,window=window,topk=topk,
                compressed=compressed,capacity=capacity,wi=sum(wi,[]),ci=sum(ci,[]),
                q_bits=bits(q),w_bits=bits(w),c_bits=bits(c),sink_bits=bits(sink),
                upstream_bits=bits(upstream),output_bits=bits(result),grad_q_bits=bits(q.grad),
                grad_w_bits=bits(w.grad),grad_c_bits=bits(c.grad),grad_dot_bits=bits(dots.grad))

def grouped_case(rows,groups,width,rank,salt):
    x=deterministic((rows,groups,width),salt,0.5).requires_grad_()
    w=deterministic((groups,rank,width),salt+1,0.5)
    upstream=deterministic((rows,groups,rank),salt+2,0.5)
    out=torch.einsum("bgd,grd->bgr",x,w)
    out.backward(upstream)
    return dict(rows=rows,groups=groups,width=width,rank=rank,weight_bits=bits(w),
                upstream_bits=bits(upstream),grad_x_bits=bits(x.grad))

def rotary_case(inverse, salt):
    batch,seq,heads,width,rd=2,3,2,12,8
    x=deterministic((batch,seq,heads,width),salt,0.5).requires_grad_()
    upstream=deterministic(x.shape,salt+1,0.5)
    angles=torch.arange(32,dtype=torch.float32)[:,None]*torch.arange(1,rd//2+1,dtype=torch.float32)[None,:]/13
    c,s=angles.cos(),angles.sin()
    positions=1+torch.arange(seq)*2
    cc,ss=c[positions][None,:,None,:],s[positions][None,:,None,:]
    if inverse: ss=-ss
    re,im=x[...,width-rd::2],x[...,width-rd+1::2]
    tail=torch.stack((re*cc-im*ss,re*ss+im*cc),-1).flatten(-2)
    y=torch.cat((x[...,:width-rd],tail),-1)
    y.backward(upstream)
    return dict(batch=batch,seq=seq,heads=heads,width=width,rd=rd,inverse=inverse,
                cosine_bits=bits(c),sine_bits=bits(s),upstream_bits=bits(upstream),grad_x_bits=bits(x.grad))

def cache_case(slots,batch,capacity,dim,salt):
    source=deterministic((batch,len(slots),dim),salt,0.5).requires_grad_()
    previous=deterministic((batch,capacity,dim),salt+1,0.5).requires_grad_()
    upstream=deterministic(previous.shape,salt+2,0.5)
    cache=previous.clone()
    for row,slot in enumerate(slots):
        if slot>=0: cache[:,slot,:]=source[:,row,:]
    cache.backward(upstream)
    return dict(slots=slots,batch=batch,capacity=capacity,dim=dim,upstream_bits=bits(upstream),
                grad_source_bits=bits(source.grad),grad_previous_bits=bits(previous.grad))

def main():
    torch.set_num_threads(1)
    source=Path(__file__).with_name("validate_deepseek_v41_model.py")
    payload=dict(tolerance=2e-7,rtol=0,comparison="strict_less_than",torch_version=torch.__version__,
        attention_forward_mirror_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sparse=[sparse_case(*spec) for spec in [
            ("prefill-compressed",100,1,3,2,8,3,3,2,2),
            ("sliding-only",200,1,4,2,8,4,4,0,0),
            ("batch-shared-cache",300,2,2,3,16,4,3,2,3),
            ("narrow-blas",400,1,2,8,32,4,4,2,2),
            ("long-softmax",500,1,2,4,32,20,20,3,4),
            ("released-heads-and-width",600,1,2,64,512,128,128,0,0),
        ]],
        grouped=[grouped_case(*spec) for spec in [(1,2,16,4,700),(4,2,32,16,800),(2,8,64,32,900)]],
        rotary=[rotary_case(False,1000),rotary_case(True,1100)],
        cache=[cache_case(*spec) for spec in [([0,1,2],2,5,8,1200),([0,1,2,0,1,2,0],2,3,8,1300),([2],2,4,8,1400),([-1,1,1],1,4,8,1500)]])
    if FIXTURE.exists() and json.loads(FIXTURE.read_text()) != payload:
        raise SystemExit("attention-gradient fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(payload,indent=2)+"\n")
    print("PASS CPU PyTorch attention-gradient fixture: 6 sparse, 3 grouped, 2 RoPE, 4 cache cases")

if __name__ == "__main__": main()
