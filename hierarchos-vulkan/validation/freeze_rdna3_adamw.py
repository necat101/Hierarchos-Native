"""CPU-only three-step AdamW states for the qualified Windows RDNA3 profile."""
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
    torch.set_num_threads(1);torch.manual_seed(0xADA512)
    def words(x):return x.detach().contiguous().view(torch.int32).flatten().tolist()
    cases=[]
    for length in [16,31,32,64,256,1024]:
        for eps,decay in [(.001,0.),(1e-8,.01)]:
            hyper=dict(lr=.001,beta1=.9,beta2=.999,eps=eps,weight_decay=decay)
            p=torch.nn.Parameter(torch.randn(length)*.02)
            initial=words(p)
            optimizer=torch.optim.AdamW([p],lr=hyper['lr'],betas=(.9,.999),eps=eps,
                weight_decay=decay,foreach=False,fused=False)
            steps=[]
            for _ in range(3):
                g=torch.randn(length)*.03;p.grad=g;optimizer.step()
                state=optimizer.state[p]
                steps.append(dict(g=words(g),p=words(p),m=words(state['exp_avg']),v=words(state['exp_avg_sq'])))
            cases.append(dict(length=length,initial=initial,hyper=hyper,steps=steps))
    with args.output.open('x') as file:
        json.dump(dict(cpu='AVX512',torch=torch.__version__,torch_revision=torch.version.git_version,threads=1,
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases),file,separators=(',',':'))
        file.write('\n')


if __name__=='__main__':main()
