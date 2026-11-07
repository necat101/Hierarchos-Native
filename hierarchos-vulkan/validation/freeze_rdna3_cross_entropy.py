"""Freeze independent CPU log-softmax adjoints for RDNA3 image training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if torch.backends.cpu.get_cpu_capability()!='AVX512':raise ValueError('native AVX512 required')
    torch.set_num_threads(1);torch.manual_seed(0xCE512)
    cases=[]
    def words(t):return t.detach().contiguous().view(torch.int32).flatten().tolist()
    for rows in [8,12,16,20,24,32]:
        x=(torch.randn(rows,320)*.2).requires_grad_()
        targets=torch.randint(0,320,(rows,));targets[::8]=-100
        weight=torch.ones(rows);weight[::8]=0
        loss=(torch.nn.functional.cross_entropy(x,targets,reduction='none')*weight).sum()/weight.sum()
        loss.backward()
        cases.append(dict(rows=rows,vocab=320,x=words(x),targets=targets.tolist(),weights=words(weight),dx=words(x.grad)))
    gemm=[]
    for rows in [8,12,16,20,24,32]:
        for reduction in [256,320]:
            dy=torch.randn(rows,reduction);w=torch.randn(reduction,64)
            gemm.append(dict(rows=rows,reduction=reduction,dy=words(dy),w=words(w),dx=words(dy@w)))
    with args.output.open('x') as f:
        json.dump(dict(cpu='AVX512',torch=torch.__version__,threads=1,torch_revision=torch.version.git_version,
            torch_config=torch.__config__.show(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,gemm=gemm),f,separators=(',',':'))
        f.write('\n')


if __name__=='__main__':main()
