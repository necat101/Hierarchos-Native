"""Freeze independent strided AVX2 eager-attention VJPs for the RDNA3 SigLIP module."""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if torch.__version__!='2.14.1+cpu' or torch.backends.cpu.get_cpu_capability()!='AVX2':
        raise ValueError('requires pinned Torch 2.14.1 CPU and AVX2')
    torch.set_num_threads(1);torch.manual_seed(0x516A82)
    words=lambda t:t.detach().contiguous().view(torch.int32).flatten().tolist()
    cases=[]
    for scale in [.125,.5,1.,2.]*2:
        inputs=[(torch.randn(1,16,16)*scale).requires_grad_() for _ in range(3)]
        q,k,v=[t.view(1,16,4,4).transpose(1,2) for t in inputs]
        probabilities=(torch.matmul(q,k.transpose(-1,-2))*.5).softmax(-1)
        y=torch.matmul(probabilities,v).transpose(1,2).contiguous().view(1,16,16)
        dy=torch.randn_like(y)*.3
        gradients=torch.autograd.grad(y,inputs,dy)
        cases.append(dict(q=words(inputs[0]),k=words(inputs[1]),v=words(inputs[2]),dy=words(dy),
            dq=words(gradients[0]),dk=words(gradients[1]),dv=words(gradients[2])))
    with a.output.open('x') as out:
        json.dump(dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,
            torch_revision=torch.version.git_version,torch_config=torch.__config__.show(),
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            batch=1,seq=16,heads=4,hidden=16,cases=cases),out,separators=(',',':'));out.write('\n')

if __name__=='__main__':main()
