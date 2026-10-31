"""Small-width full training qualification with the released 384/6 MoE and HC4.

No production checkpoint is downloaded. The seed and geometry are fixed before
native comparison; all released forward modules, cache sharing and the existing
BF16/FP4/FP8 STE adjoint contract are inherited from the independent CPU oracle.
"""
import hashlib
import json
from pathlib import Path
import torch
import validate_deepseek_v41_bf16_model as forward
import validate_deepseek_v41_full_bf16_training as training

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BASE=HERE/'fixtures/deepseek_v41_production_ratio_model.json'
FIXTURE=HERE/'fixtures/deepseek_v41_production_ratio_training_v2.json'
SEED=4141384

def frozen_write(path,payload):
    if path.exists() and json.loads(path.read_text())!=payload:
        raise SystemExit(f'{path.name}: frozen CPU evidence changed; refusing overwrite')
    path.write_text(json.dumps(payload,separators=(',',':'))+'\n')

def main():
    torch.set_num_threads(1)
    if BASE.exists():
        original=json.loads(BASE.read_text())['cases'][0]
    else:
        original=forward.make_case(ROOT/'.tmp-v41-ref/model.py',
            ROOT/'target/deepseek-v41-reference/inference_kernel.py',SEED,6,4,
            args_overrides={'n_routed_experts':384},
            config_overrides={'n_routed_experts':384})
        frozen_write(BASE,dict(cases=[original]))
    text=original['config']['text_config']
    assert (text['n_routed_experts'],text['num_experts_per_tok'],text['hc_mult'])==(384,6,4)
    base=training.constructor_base(original)
    specs,params=training.full.make_parameters(base,False)
    ns,net=training.build(base,params,specs)
    captured=[]
    net.head.register_forward_pre_hook(lambda m,a:captured.append(a[0]))
    for step in original['steps'][:4]:
        ids=torch.tensor(step['ids']).reshape(2,step['seq'])
        types=torch.tensor([-1 if v else 0 for v in step['mask']]).reshape_as(ids)
        ns['Transformer'].forward.__wrapped__(net,ids,step['pos'],token_types=types)
        logits=torch.nn.functional.linear(captured[-1],net.head.weight).flatten()
        drift=(logits-torch.tensor(step['logits'])).abs().max().item()
        print('Production-ratio BF16 initial forward',step['name'],drift,flush=True)
        assert drift<2e-7
    del net,params,specs,captured
    training.full.build=training.build
    cases=[]
    for rslora in [False,True]:
        c=training.full.case(base,rslora)
        c['base_fixture']=BASE.name;c['base_seed']=SEED;cases.append(c)
    payload=dict(tolerance=2e-7,rtol=0,comparison='strict_less_than',
        base_fixture_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(training.full.SOURCE.read_bytes()).hexdigest(),
        torch_version=torch.__version__,torch_git_version=torch.version.git_version,
        contract='Released 384 routed experts / top6 / HC4, small width, full BF16 FP4 FP8 cache-chain training with explicit FP32 STE boundaries and the released indexer bf16 score boundaries',cases=cases)
    frozen_write(FIXTURE,payload)
    print('PASS production-ratio CPU training fixture',hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),flush=True)

if __name__=='__main__':main()
