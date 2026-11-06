"""Independent CPU-only raw-word corpus for the RDNA3 width-16 norm family."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def words(tensor):
    return tensor.detach().contiguous().view(torch.int32).flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.backends.cpu.get_cpu_capability() != 'AVX512':
        raise RuntimeError('requires the native AVX512 CPU oracle')
    torch.set_num_threads(1)
    torch.manual_seed(0x512AD3)
    cases = []
    for rows, dim in [(1,16), (3,32), (5,48), (10,64), (16,128), (7,96)]:
        x = torch.randn(rows, dim, requires_grad=True)
        gamma = torch.randn(dim, requires_grad=True)
        beta = torch.randn(dim, requires_grad=True)
        dy = torch.randn_like(x)
        y, mean, rstd = torch.native_layer_norm(x, [dim], gamma, beta, 1e-5)
        y.backward(dy)
        cases.append(dict(rows=rows, dim=dim, eps=1e-5, x=words(x), gamma=words(gamma),
            beta=words(beta), dy=words(dy), y=words(y), mean=words(mean), rstd=words(rstd),
            dx=words(x.grad), dg=words(gamma.grad), db=words(beta.grad)))
    x = (torch.randn(512) * .15).requires_grad_()
    dy = torch.randn_like(x)
    y = torch.nn.functional.gelu(x)
    y.backward(dy)
    gelu = dict(x=words(x),dy=words(dy),y=words(y),dx=words(x.grad))
    rms = []
    for rows in [3,16]:
        x = torch.randn(rows,64,requires_grad=True)
        w = torch.randn(64)
        dy = torch.randn_like(x)
        ms = x.pow(2).mean(-1,keepdim=True) + 1e-5
        rs = torch.rsqrt(ms)
        y = (x*rs)*w
        y.backward(dy)
        rms.append(dict(rows=rows,x=words(x),w=words(w),dy=words(dy),y=words(y),
            ms=words(ms),rs=words(rs),dx=words(x.grad)))
    gemm = []
    for reduction in [256,320]:
        dy = torch.randn(16,reduction)
        w = torch.randn(reduction,64)
        gemm.append(dict(reduction=reduction,dy=words(dy),w=words(w),dx=words(dy@w)))
    pow_rms = []
    for rows,dim in [(16,16),(32,8)]:
        x=(torch.randn(rows,dim)*.1).requires_grad_()
        w=torch.randn(dim)*.1
        dy=torch.randn_like(x)
        ms=x.pow(2).mean(-1,keepdim=True)+1e-6
        rs=torch.pow(ms,-.5)
        y=(x*rs)*(w+1.)
        y.backward(dy)
        pow_rms.append(dict(rows=rows,dim=dim,x=words(x),w=words(w),dy=words(dy),
            rs=words(rs),ms=words(ms),power=words(torch.pow(ms,-1.5)),dx=words(x.grad)))
    attention=[]
    for kind,batch,seq,heads,kvheads,hd,window,causal in [
        ('eager',1,16,4,4,4,0,0),('eager',2,16,4,4,4,0,0),
        ('eager',2,8,2,1,8,4,1),('eager',1,20,2,1,8,0,1),
        ('eager',2,20,2,1,8,4,1),('flash',2,8,2,1,8,0,1),
        ('flash',2,8,2,1,8,4,1)]:
        q=torch.randn(batch,seq,heads*hd,requires_grad=True)
        k=torch.randn(batch,seq,kvheads*hd,requires_grad=True)
        v=torch.randn(batch,seq,kvheads*hd,requires_grad=True)
        qh=q.view(batch,seq,heads,hd).transpose(1,2)
        kh=k.view(batch,seq,kvheads,hd).transpose(1,2)
        vh=v.view(batch,seq,kvheads,hd).transpose(1,2)
        if heads!=kvheads:
            kh=kh[:,:,None].expand(batch,kvheads,heads//kvheads,seq,hd).reshape(batch,heads,seq,hd)
            vh=vh[:,:,None].expand(batch,kvheads,heads//kvheads,seq,hd).reshape(batch,heads,seq,hd)
        mask=torch.ones(batch,seq);mask[0,-1]=0 if causal else 1
        rows=torch.arange(seq)
        allowed=mask[:,None,None,:].bool().expand(batch,1,seq,seq).clone()
        if causal:allowed&=(rows[:,None]>=rows[None,:])[None,None]
        if window:allowed&=(rows[None,:]+window>rows[:,None])[None,None]
        scale=1. if kind=='flash' else hd**(-.5)
        if kind=='flash':
            oh=torch.nn.functional.scaled_dot_product_attention(qh,kh,vh,attn_mask=allowed,scale=scale)
        else:
            scores=(qh@kh.transpose(-1,-2))*scale
            oh=torch.softmax(scores.masked_fill(~allowed,-float('inf')),dim=-1)@vh
        out=oh.transpose(1,2).contiguous().view(batch,seq,heads*hd)*mask[:,:,None]
        dy=torch.randn_like(out)*.25
        out.backward(dy)
        attention.append(dict(kind=kind,batch=batch,seq=seq,heads=heads,kvheads=kvheads,hd=hd,
            window=window,causal=causal,scale=scale,q=words(q),k=words(k),v=words(v),
            qm=words(mask),km=words(mask),dy=words(dy*mask[:,:,None]),o=words(out),
            dq=words(q.grad),dk=words(k.grad),dv=words(v.grad)))
    args.output.write_text(json.dumps(dict(schema=1, cpu='AVX512', threads=1,
        torch=torch.__version__, torch_revision=torch.version.git_version,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_config=torch.__config__.show(),cases=cases,gelu=gelu,rms=rms,gemm=gemm,
        pow_rms=pow_rms,attention=attention),
        separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
