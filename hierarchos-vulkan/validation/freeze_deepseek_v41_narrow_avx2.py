"""Freeze narrow AVX2 Linear dots for native routing and LoRA-A projections."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    cases=[]
    for rows in (1,2,3,4,5,6,7,8,9,12):
        for inp in (8,12,16,32,64):
            for out in (2,3):
                for seed in (1901,7183):
                    torch.manual_seed(seed+rows*101+inp+out);x=torch.randn(rows,inp)*.3;w=torch.randn(out,inp)*.1;y=F.linear(x,w)
                    cases.append(dict(rows=rows,input_dim=inp,output_dim=out,seed=seed,input=x.flatten().tolist(),weight=w.flatten().tolist(),expected_bits=y.flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),contract='Independent fixed random bias-free contiguous FP32 F.linear, output width two or three, including incomplete vector tails',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'narrow AVX2 projections',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
