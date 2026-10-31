"""Complete CPU autograd/AdamW gate with actual packed FP4/FP8 experts.

The existing full hybrid-model checkpoint is the base; no operand magnitudes,
optimizer defaults, logits, or reference tolerances are adjusted for a pass.
"""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import validate_deepseek_v41_full_training as full
from validate_deepseek_v41_packed_grad import packed_forward

HERE=Path(__file__).resolve().parent
BASE=HERE/'fixtures/deepseek_v41_mixed_model.json'
FIXTURE=HERE/'fixtures/deepseek_v41_full_mixed_training.json'
original_build=full.build

class PackedLinear(nn.Module):
    def __init__(self,spec,a,b,scale):
        super().__init__();self.spec=spec;self.a=a;self.b=b;self.scale=scale
    def forward(self,x):
        shape=x.shape
        return packed_forward(x.reshape(-1,shape[-1]),self.spec,False,self.a,self.b,self.scale).reshape(*shape[:-1],self.spec['rows'])

def build(base,params,specs):
    ns,net=original_build(base,params,specs,args_overrides={'moe_inter_dim':32})
    for name,spec in base['packed_tensors'].items():
        target=name[:-7]
        path,leaf=target.rsplit('.',1)
        a,b=params[target]
        setattr(net.get_submodule(path),leaf,PackedLinear(spec,a,b,specs[target]['scale']))
    return ns,net

def main():
    torch.set_num_threads(1)
    original=json.loads(BASE.read_text())['cases'][0]
    base=copy.deepcopy(original)
    # Constructor slots only: every packed matrix is replaced before forward.
    # These zero tensors never contribute to an expected output or gradient.
    for name,spec in base['packed_tensors'].items():
        base['tensors'][name]=dict(shape=[spec['rows'],spec['cols']],values=[0.0]*(spec['rows']*spec['cols']))
    full.build=build
    cases=[]
    for rslora in [False,True]:
        c=full.case(base,rslora)
        c['base_fixture']='deepseek_v41_mixed_model.json'
        c['base_seed']=original['seed']
        cases.append(c)
    payload=dict(tolerance=2e-7,rtol=0,comparison='strict_less_than',base_fixture_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(full.SOURCE.read_bytes()).hexdigest(),torch_version=torch.__version__,torch_git_version=torch.version.git_version,
        contract='Full released model bodies with FP4 routed / FP8 shared projections and established FP32 STE boundaries',cases=cases)
    if FIXTURE.exists() and json.loads(FIXTURE.read_text())!=payload:raise SystemExit('mixed training CPU fixture changed; refusing overwrite')
    FIXTURE.write_text(json.dumps(payload,indent=2)+'\n')
    print('PASS full mixed CPU training fixture',hashlib.sha256(FIXTURE.read_bytes()).hexdigest())

if __name__=='__main__':main()
