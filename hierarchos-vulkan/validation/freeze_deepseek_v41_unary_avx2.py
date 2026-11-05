"""Freeze sigmoid and SiLU across the AVX2 vector/scalar dispatch boundary."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    cases=[]
    for length in (1,6,8,16,18,24,32,36,48,64,96):
        for seed in (1901,7183):
            torch.manual_seed(seed+length);x=torch.randn(length)*2
            for op,fn in [(1,torch.sigmoid),(2,F.silu)]:
                y=fn(x)
                cases.append(dict(length=length,seed=seed,op=op,input=x.tolist(),expected_bits=y.view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contract='Independent fixed random native Torch sigmoid/SiLU; no values from Vulkan or pipeline traces',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'AVX2 unary controls',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
