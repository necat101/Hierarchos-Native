"""Independent released native V4.1 vision fixture with cooperative geometries.

The maintained small FP32 fixtures are immutable. This separate fixture widens
the same released module bodies; it is not a shared CLIP or text surrogate.
"""
import argparse,json
from pathlib import Path
import torch
from types import SimpleNamespace
import freeze_deepseek_v41_vision as producer

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight-std',type=float,default=.08)
    p.add_argument('--upstream-std',type=float,default=.03)
    a=p.parse_args()
    torch.set_num_threads(1)
    configuration={**producer.arguments(),'vision_dim':128,'vision_n_heads':4,'vision_inter_dim':256,
        'vision_n_layers':1,'dim':128}
    vision,processor=producer.load_references()
    torch.manual_seed(0x563431+8*100+8)
    options=SimpleNamespace(**configuration)
    net=torch.nn.ModuleDict({'vision':vision.ViT(options),'aligner':vision.Aligner(options)})
    with torch.no_grad():
        for name,parameter in net.named_parameters():
            parameter.copy_(1+torch.randn_like(parameter)*.02 if '.norm' in name else
                torch.randn_like(parameter)*(.02 if name.endswith('bias') else a.weight_std))
    patches,image=producer.image_input(processor,options,8,8);patches.requires_grad_()
    initial={name:producer.tensor(parameter) for name,parameter in net.named_parameters()}
    upstream=torch.randn(image['n_llm_h']*image['n_llm_w'],options.dim)*a.upstream_std
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,betas=(.9,.999),eps=.001,weight_decay=0.,foreach=False,fused=False)
    steps=[]
    for step in range(3):
        features=net['vision'](patches,image['n_vit_h'],image['n_vit_w'])
        projected=net['aligner'](features,image['n_vit_h'],image['n_vit_w'])
        row=dict(features=producer.tensor(features),projected=producer.tensor(projected))
        if step<2:
            optimizer.zero_grad(set_to_none=True);patches.grad=None;projected.backward(upstream)
            row['patch_gradient']=producer.tensor(patches.grad)
            row['gradients']={name:producer.tensor(parameter.grad) for name,parameter in net.named_parameters()}
            optimizer.step()
            row['updated_parameters']={name:producer.tensor(parameter) for name,parameter in net.named_parameters()}
        steps.append(row)
    case=dict(name='native_vit_8x8',config=configuration,image=image,parameters=initial,
        patches=producer.tensor(patches),upstream=producer.tensor(upstream),steps=steps,
        parameter_std=a.weight_std,upstream_std=a.upstream_std)
    case['oracle']={'torch':torch.__version__,'revision':producer.REVISION,'source_sha256':producer.HASHES,
        'cpu':torch.backends.cpu.get_cpu_capability(),'threads':1,'scope':'native ViT/aligner, not joint text training'}
    with a.output.open('x') as out:json.dump(case,out,separators=(',',':'))
    print('Prepared native V4.1 cooperative fixture',configuration,flush=True)

if __name__=='__main__':main()
