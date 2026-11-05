"""Independent FP32 matmul and routed SwiGLU adjoints for native AVX2 training."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    mm=[];swiglu=[]
    values=lambda t:t.detach().flatten().tolist()
    bits=lambda t:t.detach().flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
    for rows in (1,2,3,4,5,6,7,8,9,12):
        for reduction,columns in ((12,32),(32,12),(16,32),(32,8),(64,8),(96,16),(32,2),(12,2)):
            for seed in (1901,7183):
                torch.manual_seed(seed+rows+reduction+columns);a=torch.randn(rows,reduction)*.1;b=torch.randn(reduction,columns)*.2;y=a@b
                mm.append(dict(rows=rows,reduction=reduction,columns=columns,seed=seed,dy=values(a),weight=values(b),expected_bits=bits(y)))
    for rows in (1,2,3,5,8,9):
        for inter in (12,32):
            for shared in (False,True):
                for seed in (1901,7183):
                    torch.manual_seed(seed+rows+inter);g=(torch.randn(rows,inter)*1.3).requires_grad_();u=(torch.randn_like(g)*1.3).requires_grad_();r=torch.rand(rows,1).requires_grad_();dy=torch.randn_like(g)*.05
                    with torch.no_grad():g.flatten()[0]=1.5;u.flatten()[1]=-1.5
                    value=F.silu(torch.clamp(g,max=1.5))*torch.clamp(u,min=-1.5,max=1.5)
                    if not shared:value=r*value
                    dg,du=torch.autograd.grad(value,(g,u),dy,retain_graph=True)
                    dr=torch.zeros(rows) if shared else torch.autograd.grad(value,r,dy)[0].flatten()
                    swiglu.append(dict(rows=rows,inter=inter,shared=shared,seed=seed,gate=values(g),up=values(u),route=values(r),dy=values(dy),dg_bits=bits(dg),du_bits=bits(du),dr_bits=bits(dr)))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),contract='Fixed random native Torch matmul and autograd through SiLU, scalar clamps and optional routing weights; includes exact clamp boundaries',matmul=mm,swiglu=swiglu)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(mm),'matmul and',len(swiglu),'SwiGLU controls',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
