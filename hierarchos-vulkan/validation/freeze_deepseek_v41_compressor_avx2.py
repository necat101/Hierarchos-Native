"""Freeze the native ratio-one compressor and unmodified CPU input autograd."""
from pathlib import Path
import argparse,hashlib,json
import torch
import freeze_deepseek_v41_native_pipeline as ref

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    base=json.loads(ref.training.BASE.read_text())
    source=ref.vision_oracle.REFERENCE/'model.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'
    ns=ref.training.model.load_reference(source,ref.training.KERNEL,tokens=base['tokens'])
    ref.training.install_functional(ns);config=ref.args_for(ns);cases=[]
    values=lambda t:t.detach().flatten().tolist()
    bits=lambda t:t.detach().flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
    for rows in (1,2,3,8,9,10):
        for seed in (1901,7183):
            torch.manual_seed(seed+rows)
            compressor=ns['Compressor'](config,3)
            assert compressor.compress_ratio==1
            with torch.no_grad():
                compressor.wkv.weight.copy_(torch.randn_like(compressor.wkv.weight)*.1)
                compressor.norm.weight.copy_(1.+torch.randn_like(compressor.norm.weight)*.03)
            compressor.requires_grad_(False)
            x=(torch.randn(1,rows,32)*.3).requires_grad_()
            y=compressor(x,0);dy=torch.randn_like(y)*.03
            dx,=torch.autograd.grad(y,x,dy)
            cases.append(dict(rows=rows,seed=seed,dim=32,hidden=32,ratio=1,eps=compressor.norm.eps,input=values(x),weight=values(compressor.wkv.weight),norm_weight=values(compressor.norm.weight),upstream=values(dy),output_bits=bits(y),input_bits=bits(dx)))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fallback_sha256=hashlib.sha256(Path(ref.training.__file__).read_bytes()).hexdigest(),contract='Same released Compressor and FP32 functional CPU fallback as immutable native pipeline; ratio1 normalized Linear and unchanged Torch autograd; independent random inputs/upstream',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'native compressor adjoints',hashlib.sha256(args.output.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':main()
