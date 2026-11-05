"""Freeze independent AVX2 projection shapes, including short SGEMM panels."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1)
    assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    cases=[]
    for rows in (2,3,4,5,6,7,8,9,10,11,12):
        for inp,out in ((16,96),(64,8),(32,8),(64,16),(32,12),(12,32),(32,16)):
            for seed in (1901,7183):
                torch.manual_seed(seed+rows*101+inp+out)
                x=torch.randn(rows,inp)*.3;w=torch.randn(out,inp)*.1;y=F.linear(x,w)
                cases.append(dict(rows=rows,input_dim=inp,output_dim=out,seed=seed,
                    input=x.flatten().tolist(),weight=w.flatten().tolist(),expected_bits=y.flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contract='Independent fixed random contiguous FP32 F.linear without bias; no pipeline intermediates or fitted values',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'independent language projections',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
