"""Freeze CPU projection operands for the independent compressor VJP corpus."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    source=Path(__file__).parent/'fixtures/vision_avx2_row_tail/deepseek_v41_compressor_avx2.json'
    payload=source.read_bytes();assert hashlib.sha256(payload).hexdigest()=='6968719fd8bc8e7daab0ef55178ca60c9a04aa2b4817ca26588e551cd2db769a'
    cpu=json.loads(payload);cases=[]
    bits=lambda t:t.detach().flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
    for case in cpu['cases']:
        rows=case['rows'];x=torch.tensor(case['input']).reshape(1,rows,32).requires_grad_();weight=torch.tensor(case['weight']).reshape(32,32)
        projected=F.linear(x,weight)
        normalized=(projected*(projected.square().mean(-1,keepdim=True)+case['eps']).rsqrt())*torch.tensor(case['norm_weight'])
        dx,=torch.autograd.grad(normalized,x,torch.tensor(case['upstream']).reshape_as(normalized))
        assert bits(normalized)==case['output_bits'] and bits(dx)==case['input_bits'],'CPU operand observer changed frozen compressor scenario'
        cases.append(dict(rows=rows,seed=case['seed'],projected_bits=bits(projected)))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_fixture_sha256=hashlib.sha256(payload).hexdigest(),contract='CPU F.linear operands; independently asserts every existing compressor output and input gradient word before writing; expected gradients unchanged',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'verified CPU compressor operands',hashlib.sha256(args.output.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':main()
