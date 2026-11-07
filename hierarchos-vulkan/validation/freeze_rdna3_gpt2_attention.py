"""Independent CPU words for the GPT-2 strided QKV attention contract."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if torch.backends.cpu.get_cpu_capability()!='AVX512':
        raise ValueError('native AVX512 CPU required')
    torch.set_num_threads(1);torch.manual_seed(177)
    cases=[]
    def words(t):return t.detach().contiguous().view(torch.int32).flatten().tolist()
    def layout(t):return words(t.transpose(1,2).contiguous())
    for batch,seq,heads in [(1,8,4),(2,8,4),(3,8,4),(2,6,4),(2,10,4),(2,16,4),(2,8,2),(2,8,8)]:
        for padded in [False,True]:
            # Preserve the actual GPT-2 QKV view strides through the CPU graph.
            x=torch.randn(batch,seq,192,requires_grad=True)
            hd=64//heads
            q,k,v=[t.view(batch,seq,heads,hd).transpose(1,2) for t in x.chunk(3,dim=-1)]
            for t in [q,k,v]:t.retain_grad()
            mask=torch.ones(batch,seq)
            if padded:
                mask[0,-1]=0
                if batch>1:mask[-1,0]=0
            allowed=torch.ones(batch,1,seq,seq).tril().bool() & mask[:,None,None,:].bool()
            oh=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=allowed)
            out=oh.transpose(1,2).contiguous()*mask[:,:,None,None]
            dy=torch.randn_like(out)*.25
            out.backward(dy)
            cases.append(dict(batch=batch,seq=seq,heads=heads,hd=hd,scale=hd**-.5,padded=padded,q=layout(q),k=layout(k),v=layout(v),
                o=words(out),dy=words(dy*mask[:,:,None,None]),mask=words(mask),
                dq=layout(q.grad),dk=layout(k.grad),dv=layout(v.grad)))
    clip=[]
    for batch in [1,2,3]:
        q,k,v=[torch.randn(batch,5,16,requires_grad=True) for _ in range(3)]
        qh,kh,vh=[t.view(batch,5,2,8).transpose(1,2) for t in [q,k,v]]
        p=((qh@kh.transpose(-1,-2))*(8**-.5)).softmax(-1)
        out=(p@vh).transpose(1,2).contiguous()
        dy=torch.randn_like(out);out.backward(dy)
        clip.append(dict(batch=batch,q=words(q),k=words(k),v=words(v),dy=words(dy),
            dq=words(q.grad),dk=words(k.grad),dv=words(v.grad)))
    x=torch.randn(8192).requires_grad_()
    dy=torch.randn_like(x)
    y=.5*x*(1.+torch.tanh((2./torch.pi)**.5*(x+.044715*x.pow(3))))
    y.backward(dy)
    new_gelu=dict(x=words(x),dy=words(dy),y=words(y),dx=words(x.grad))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:
        json.dump(dict(cpu='AVX512',torch=torch.__version__,torch_revision=torch.version.git_version,
            torch_config=torch.__config__.show(),threads=1,
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,clip=clip,new_gelu=new_gelu),file,separators=(',',':'))
        file.write('\n')


if __name__=='__main__':main()
