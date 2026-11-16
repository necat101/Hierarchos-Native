"""Independent released native V4.1 vision fixtures with cooperative geometries.

The maintained small FP32 fixtures are immutable. These separate fixtures widen
the same released module bodies; they are not a shared CLIP or text surrogate.

Geometry is part of the contract, not a convenience. `ProjectionWmma::admits`
now calls `wmma_shape_profitable`, so a vision projection is only admitted when
`rows * columns * reduction` reaches `WMMA_MIN_PROFITABLE_MACS` (16,000,000).
At the historical 8x8 image the vision tower has 16 patches, so its widest
projection (`wqkv`: 384 columns, 128 reduction) computes
`16 * 384 * 128 = 786_432` and the crossover policy declines every projection;
the tower then exercises no cooperative route at all.

`--inter-dim` is the lever used for the cooperative fixture: widening the MLP
(`w1`: `inter` columns, 128 reduction) reaches
`16 * 16_384 * 128 = 33_554_432 >= 16_000_000`
while the image, the patch count and therefore the *population* of compared
output elements are unchanged. Enlarging the image instead (40x40 = 400
patches) also clears the floor, but it multiplies `projected` from 512 to 12,800
elements and the native-vs-torch FP32 maximum then grows from 1.6e-7 to 3.6e-7,
breaching the bounded 2e-7 strict limit for a reason that has nothing to do with
cooperative dispatch. `--width`/`--height`/`--inter-dim` default to 8/8/256 so
the historical bytes stay reproducible.

The oracle must be generated with `ATEN_CPU_CAPABILITY=avx2` (plus
`MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`): on an AVX512 host
the same torch build emits different kernels and the fixture is no longer
byte-identical to the recorded AVX2 oracle.

`--stress` emits the larger-magnitude diagnostic through the released
`freeze_deepseek_v41_vision.case` path (which also records the intermediate
trace); it is a separate artefact and never promotes the bounded result.
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
    p.add_argument('--width',type=int,default=8);p.add_argument('--height',type=int,default=8)
    p.add_argument('--max-token',type=int,default=32,
        help='recorded vision_max_n_token; raise it when the image exceeds the release token cap')
    p.add_argument('--inter-dim',type=int,default=256,
        help='MLP width; 16384 admits w1/w2 at the historical patch count')
    p.add_argument('--stress',action='store_true',
        help='emit the larger-magnitude diagnostic with the released trace path')
    a=p.parse_args()
    torch.set_num_threads(1)
    configuration={**producer.arguments(),'vision_dim':128,'vision_n_heads':4,'vision_inter_dim':256,
        'vision_n_layers':1,'dim':128,'vision_max_n_token':a.max_token,'vision_inter_dim':a.inter_dim}
    vision,processor=producer.load_references()
    if a.stress:
        # `case` builds its own net from `arguments()`; widen the same released
        # module bodies rather than copying the generator here.
        original=producer.arguments
        producer.arguments=lambda: configuration
        try:case=producer.case(vision,processor,a.width,a.height)
        finally:producer.arguments=original
        case['oracle']={'torch':torch.__version__,'revision':producer.REVISION,'source_sha256':producer.HASHES,
            'cpu':torch.backends.cpu.get_cpu_capability(),'threads':1,'scope':'native ViT/aligner, not joint text training'}
    else:
        torch.manual_seed(0x563431+a.width*100+a.height)
        options=SimpleNamespace(**configuration)
        net=torch.nn.ModuleDict({'vision':vision.ViT(options),'aligner':vision.Aligner(options)})
        with torch.no_grad():
            for name,parameter in net.named_parameters():
                parameter.copy_(1+torch.randn_like(parameter)*.02 if '.norm' in name else
                    torch.randn_like(parameter)*(.02 if name.endswith('bias') else a.weight_std))
        patches,image=producer.image_input(processor,options,a.width,a.height);patches.requires_grad_()
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
        case=dict(name=f'native_vit_{a.width}x{a.height}',config=configuration,image=image,parameters=initial,
            patches=producer.tensor(patches),upstream=producer.tensor(upstream),steps=steps,
            parameter_std=a.weight_std,upstream_std=a.upstream_std)
        case['oracle']={'torch':torch.__version__,'revision':producer.REVISION,'source_sha256':producer.HASHES,
            'cpu':torch.backends.cpu.get_cpu_capability(),'threads':1,'scope':'native ViT/aligner, not joint text training'}
    with a.output.open('x') as out:json.dump(case,out,separators=(',',':'))
    print('Prepared native V4.1', 'stress' if a.stress else 'bounded', a.width, 'x', a.height,
          'patches', len(case['patches']['values'])//(3*configuration['vision_patch_size']**2), flush=True)


if __name__=='__main__':main()
