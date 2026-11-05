"""Independent pinned Intel AVX2 Torch sqrt corpus for the native VL router."""
import argparse,hashlib,json
from pathlib import Path
import torch
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    torch.manual_seed(1901);x=torch.pow(2.,torch.rand(4096)*20.-10.);y=x.sqrt()
    result=dict(schema=1,cpu='Intel AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),contract='Native Torch sqrt on fixed random positive values, spanning 2^-10 through 2^10',input=x.tolist(),expected_bits=y.view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen 4096 Intel AVX2 sqrt words',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
