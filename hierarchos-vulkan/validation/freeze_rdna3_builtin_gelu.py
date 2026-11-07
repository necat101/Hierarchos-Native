"""Freeze an independent ATen builtin tanh-GELU derivative corpus.

Use the isolated pinned Torch CPU runtime, with ATEN_CPU_CAPABILITY=avx2.
The corpus covers the FP32 3*kappa rounding point; it is not a model oracle.
"""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if torch.__version__!='2.14.1+cpu' or torch.backends.cpu.get_cpu_capability()!='AVX2':
        raise ValueError('requires pinned Torch 2.14.1 CPU and AVX2')
    torch.set_num_threads(1);torch.manual_seed(0x214512)
    x=torch.randn(8192,requires_grad=True);dy=torch.randn_like(x)
    y=torch.nn.functional.gelu(x,approximate='tanh')
    dx=torch.autograd.grad(y,x,dy)[0]
    words=lambda t:t.detach().contiguous().view(torch.int32).flatten().tolist()
    with args.output.open('x') as out:
        json.dump(dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,
            torch_revision=torch.version.git_version,torch_config=torch.__config__.show(),
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            x=words(x),dy=words(dy),dx=words(dx)),out,separators=(',',':'));out.write('\n')

if __name__=='__main__':main()
