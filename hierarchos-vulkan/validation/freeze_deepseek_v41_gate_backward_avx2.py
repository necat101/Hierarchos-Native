"""Freeze the unchanged released native VL Gate forward and input autograd."""
import argparse,hashlib,json
from pathlib import Path
import torch
import freeze_deepseek_v41_native_pipeline as ref
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    base=json.loads(ref.training.BASE.read_text());ns=ref.training.model.load_reference(ref.vision_oracle.REFERENCE/'model.py',ref.training.KERNEL,tokens=base['tokens']);config=ref.args_for(ns);cases=[]
    values=lambda t:t.detach().flatten().tolist()
    bits=lambda t:t.detach().flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
    for rows in (1,2,3,8,9):
        for seed in (1901,7183,20983,34129):
            torch.manual_seed(seed+rows);gate=ns['Gate'](0,config)
            with torch.no_grad():
                gate.weight.copy_(torch.randn_like(gate.weight)*.1);gate.bias.copy_(torch.tensor([-.15,0.,.15]));gate.bias_vl.copy_(torch.tensor([.15,-.15,0.]))
            gate.requires_grad_(False);x=(torch.randn(rows,32)*.3).requires_grad_();mask=torch.arange(rows)%3==0
            weight,indices=gate(x,mask);upstream=torch.randn_like(weight)*.1;dx,=torch.autograd.grad(weight,x,upstream)
            cases.append(dict(rows=rows,seed=seed,input=values(x),weight=values(gate.weight),bias=values(gate.bias),bias_vl=values(gate.bias_vl),text_mask=(~mask).int().tolist(),route=values(weight),indices=indices.flatten().tolist(),upstream=values(upstream),expected_bits=bits(dx)))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ref.vision_oracle.REFERENCE/'model.py').read_bytes()).hexdigest(),contract='Unchanged released Gate class, with distinct native image correction biases and native Torch autograd',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'released Gate adjoints',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
