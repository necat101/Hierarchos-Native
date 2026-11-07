"""Independent CPU outer reductions, including both vector blocks and tails."""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if torch.backends.cpu.get_cpu_capability()!='AVX512':raise ValueError('AVX512 required')
    torch.set_num_threads(1);torch.manual_seed(0xB1A5512)
    def words(x):return x.detach().contiguous().view(torch.int32).flatten().tolist()
    cases=[]
    for rows in [5,8,10,12,15,16,20,24,32,65]:
        for dim in [16,32,48,64,80,96,192,256]:
            dy=torch.randn(rows,dim)*.03
            cases.append(dict(rows=rows,dim=dim,dy=words(dy),db=words(dy.sum(0))))
    with a.output.open('x') as f:
        json.dump(dict(cpu='AVX512',torch=torch.__version__,torch_revision=torch.version.git_version,
            threads=1,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases),f,separators=(',',':'))
        f.write('\n')
if __name__=='__main__':main()
