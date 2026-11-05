"""Independent native VL loss, logit VJP and frozen vocabulary-head adjoint."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    source=ROOT/'hierarchos-vulkan/validation/fixtures/deepseek_v41_native_pipeline_avx2.json';sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha=='9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b'
    pipeline=json.loads(source.read_text());base=json.loads((source.parent/'deepseek_v41_model.json').read_text());w=torch.tensor(base['tensors']['head.weight']['values']).reshape(16,32);cases=[]
    for c in pipeline['cases']:
        target=torch.tensor(c['targets']);live=target!=-100
        for step,record in enumerate(c['steps']):
            logits=torch.tensor(record['logits']['values']).reshape(-1,16).requires_grad_();loss=F.cross_entropy(logits,target,ignore_index=-100);gradient,=torch.autograd.grad(loss,logits)
            rows=F.cross_entropy(logits.detach(),target,ignore_index=-100,reduction='none');head=gradient@w
            cases.append(dict(name=c['name'],step=step,rows=len(target),vocab=16,hidden=32,targets=target.tolist(),logits=logits.detach().flatten().tolist(),loss=loss.item(),row_loss=rows.tolist(),gradient=gradient.flatten().tolist(),head_gradient=head.flatten().tolist(),head_weight=w.flatten().tolist()))
    result=dict(schema=1,cpu='AVX2',threads=1,torch=torch.__version__,torch_revision=torch.version.git_version,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_fixture_sha256=sha,contract='Native Torch cross_entropy and autograd on unchanged frozen CPU pipeline logits; vocabulary head input VJP is Torch matmul with the original frozen head',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen',len(cases),'native terminal controls',hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=='__main__':main()
