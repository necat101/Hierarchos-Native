"""Freeze native LoRA transposed GEMM, odd reductions and narrow row tails."""
from pathlib import Path
import argparse,hashlib,json
import torch
import torch.nn.functional as F
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    values=lambda t:t.detach().flatten().tolist()
    bits=lambda t:t.detach().flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
    weights=[];odd=[];narrow=[]
    for rows in (1,2,3,4,5,6,7,8,9,12):
        for inp,out in ((2,8),(2,12),(2,32),(2,64),(8,2),(12,2),(32,2),(64,2)):
            for seed in (1901,7183):
                torch.manual_seed(seed+rows+inp+out);x=torch.randn(rows,inp)*.2;g=torch.randn(rows,out)*.03;y=g.T@x
                weights.append(dict(rows=rows,input_dim=inp,output_dim=out,seed=seed,input=values(x),upstream=values(g),expected_bits=bits(y)))
    for rows in (1,2,3,8,9):
        for reduction in (3,5,7,9):
            for columns in (12,32):
                for seed in (1901,7183):
                    torch.manual_seed(seed+rows+reduction+columns);x=torch.randn(rows,reduction)*.03;w=torch.randn(reduction,columns)*.1;y=x@w
                    odd.append(dict(rows=rows,reduction=reduction,columns=columns,seed=seed,input=values(x),weight=values(w),expected_bits=bits(y)))
    for rows in (1,3,9):
        for inp in (12,16,20,24,28,32,48,64,96,128,192):
            for out in (2,3):
                for seed in (1901,7183):
                    torch.manual_seed(seed+rows+inp+out);x=torch.randn(rows,inp)*.3;w=torch.randn(out,inp)*.1;y=F.linear(x,w)
                    narrow.append(dict(rows=rows,input_dim=inp,output_dim=out,seed=seed,input=values(x),weight=values(w),expected_bits=bits(y)))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),contract='Independent random native Torch grad.T@input, contiguous matmul with odd reductions and bias-free F.linear row tails; expected values independent of native implementation',weight_gradients=weights,odd_inputs=odd,narrow=narrow)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(weights),'weight VJPs,',len(odd),'odd input VJPs,',len(narrow),'narrow tails',hashlib.sha256(args.output.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':main()
