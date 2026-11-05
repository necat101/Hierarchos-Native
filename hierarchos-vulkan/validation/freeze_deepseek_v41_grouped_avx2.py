"""Independent AVX2 grouped einsum projection controls for short panels."""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    cases=[]
    for rows in (2,3,4,5,6,7,8,9,10,11,12):
        for groups in (1,2):
            for seed in (1901,7183):
                torch.manual_seed(seed+rows*101+groups)
                x=torch.randn(1,rows,groups,64)*.3;w=torch.randn(groups,16,64)*.1
                y=torch.einsum('bsgd,grd->bsgr',x,w)
                cases.append(dict(rows=rows,groups=groups,group_width=64,output_rank=16,seed=seed,
                    input=x.flatten().tolist(),weight=w.flatten().tolist(),expected_bits=y.flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contract='Independent fixed random FP32 grouped torch.einsum, same operand order as released attention; no trace-fitted values',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'independent grouped projections',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
