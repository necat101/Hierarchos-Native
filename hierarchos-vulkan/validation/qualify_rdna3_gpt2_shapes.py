"""Freeze and replay additional GPT-2 image-training shapes, independently on CPU.

Uses a previously frozen model as input weights, never as expected outputs.
The maintained shared-image qualifier enforces the original logits, parameters,
two AdamW updates, image sensitivity, pixel gradients and export/reload gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from qualify_multimodal_training import digest, write_json

SHAPES = [(2, 8, 4), (1, 8, 4), (3, 8, 4), (2, 6, 4),
          (2, 10, 4), (2, 16, 4), (2, 8, 2), (2, 8, 8)]


def prepare(args):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file

    checkout = Path(os.environ['HIERARCHOS_TRANSFORMERS_CHECKOUT'])
    sys.path.insert(0, str(checkout / 'src'))
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, GPT2LMHeadModel

    if torch.backends.cpu.get_cpu_capability() != 'AVX512':
        raise ValueError('native AVX512 CPU oracle required')
    torch.set_num_threads(1)
    source = args.model.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profiles = [f'gpt2_b{b}_s{s}_h{h}' for b, s, h in SHAPES]
    manifest = dict(schema=1, complete=False, scope='rdna3_gpt2_additional_shapes',
        absolute_limit=2e-7, rtol=0, comparison='strict_less_than',
        required_green_profiles=profiles, requested_profiles=profiles, cases=[],
        oracle=dict(cpu='AVX512', torch=torch.__version__, torch_revision=torch.version.git_version,
            threads=1, text_attention='sdpa', vision_attention='eager',
            transformers_revision=subprocess.check_output(
                ['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),
            torch_config=torch.__config__.show(), source_sha256={}),
        input_model_sha256={p.relative_to(source).as_posix():digest(p)
            for p in source.rglob('*') if p.is_file()})
    for batch, seq, heads in SHAPES:
        profile = f'gpt2_b{batch}_s{seq}_h{heads}'
        for padded in [False, True]:
            name = profile + ('/padded' if padded else '/unmasked')
            folder = output / name
            package = folder / 'model'
            shutil.copytree(source, package)
            config_path = package / 'text/config.json'
            config = json.loads(config_path.read_text())
            config['n_head'] = heads
            write_json(config_path, config)
            text = GPT2LMHeadModel.from_pretrained(package / 'text').float().train()
            text.set_attn_implementation('sdpa')
            width = text.config.n_embd
            image_config = json.loads((package / 'image/config.json').read_text())
            state = load_file(str(package / 'image/model.safetensors'))
            tower = CLIPVisionModel(CLIPVisionConfig(**image_config['vision_config'])).float().train()
            tower.set_attn_implementation('eager')
            prefix = 'vision_tower.' if 'vision_model.embeddings.class_embedding' in tower.state_dict() else 'vision_tower.vision_model.'
            tower.load_state_dict({k.removeprefix(prefix):v for k,v in state.items()
                if k.startswith('vision_tower.')})
            projector = torch.nn.Sequential(torch.nn.Linear(16,width),torch.nn.GELU(),torch.nn.Linear(width,width))
            for index, label in [(0, 'linear_1'), (2, 'linear_2')]:
                projector[index].load_state_dict({k:state[f'multi_modal_projector.{label}.{k}']
                    for k in ['weight','bias']})
            image_parameters = {prefix+k:v for k,v in tower.named_parameters()}
            image_parameters.update({'multi_modal_projector.linear_1.'+k:v for k,v in projector[0].named_parameters()})
            image_parameters.update({'multi_modal_projector.linear_2.'+k:v for k,v in projector[2].named_parameters()})
            processor = CLIPImageProcessor.from_pretrained(package / 'image')
            token = image_config['image_token_index']
            # Deterministic inputs, including the original two images at batch=2.
            rgb = np.random.default_rng(0x494D47+sum(b'gpt2_wide')).integers(0,256,(batch,8,8,3),dtype=np.uint8)
            pixels = processor(images=[Image.fromarray(x) for x in rgb],return_tensors='pt')['pixel_values'].requires_grad_()
            alternate = processor(images=[Image.fromarray(255-x) for x in rgb],return_tensors='pt')['pixel_values']
            tails = ([7,3,2,9,4,5,6,10,11,12,13], [3,7,1,9,4,5,6,10,11,12,13])
            sequences = [[1+i]+[token]*4+(tails[i%2]*2)[:seq-5] for i in range(batch)]
            ids = torch.tensor(sequences)
            mask = torch.ones_like(ids)
            if padded:
                mask[0,-1] = 0
                if batch > 1: mask[-1,0] = 0
            positions = (ids==token).flatten().nonzero().flatten()
            targets = ids.roll(-1,dims=1)
            weights = mask.float();weights[:,-1]=0;weights[:,:-1]*=mask[:,1:];weights[:,:4]=0
            targets[weights==0]=-100
            fixture = dict(batch_size=batch,seq_len=seq,image_count=batch,input_ids=ids.flatten().tolist(),
                attention_mask=mask.float().flatten().tolist(),targets=[x if x>=0 else 0xffffffff for x in targets.flatten().tolist()],
                loss_weights=weights.flatten().tolist(),rgb=rgb.flatten().tolist(),alternate_rgb=(255-rgb).flatten().tolist(),
                steps=2,lr=.001,beta1=.9,beta2=.999,eps=.001,weight_decay=0.)
            write_json(folder/'input.json',fixture)
            holder = {}
            def replace_rows(module, inputs, values):
                return values.reshape(-1,width).index_copy(0,positions,holder['projected']).reshape_as(values)
            hook = text.get_input_embeddings().register_forward_hook(replace_rows)
            def forward(values):
                features=tower(values,output_hidden_states=True).hidden_states[-1][:,1:,:].reshape(-1,16)
                holder['projected']=projector(features)
                return dict(logits=text(input_ids=ids,attention_mask=mask,use_cache=False).logits,
                    vision_features=features,projected_features=holder['projected'])
            def record(values):return {k:v.detach().flatten().tolist() for k,v in values.items()}
            with torch.no_grad(): alternate_output=record(forward(alternate))
            optimizer=torch.optim.AdamW([*text.parameters(),*tower.parameters(),*projector.parameters()],
                lr=.001,betas=(.9,.999),eps=.001,weight_decay=0.,foreach=False,fused=False)
            steps=[]
            for step in [1,2]:
                optimizer.zero_grad(set_to_none=True);pixels.grad=None
                result=forward(pixels);holder['projected'].retain_grad()
                loss=(torch.nn.functional.cross_entropy(result['logits'].reshape(-1,text.config.vocab_size),
                    targets.flatten(),reduction='none').reshape_as(weights)*weights).sum()/weights.sum()
                loss.backward()
                steps.append(dict(forward=record(result),loss=loss.item(),pixel_gradient=pixels.grad.flatten().tolist(),
                    projected_gradient=holder['projected'].grad.flatten().tolist()))
                optimizer.step()
                parameters={'text.'+k:v for k,v in text.named_parameters()}
                parameters.update({'image.'+k:v for k,v in image_parameters.items()})
                save_file({k:v.detach().contiguous().clone() for k,v in parameters.items()},str(folder/f'cpu-step-{step}.safetensors'))
            with torch.no_grad(): final=record(forward(pixels))
            hook.remove()
            write_json(folder/'cpu.json',dict(pixels=pixels.detach().flatten().tolist(),alternate=alternate_output,
                steps=steps,final=final,visible=mask.bool().reshape(-1,1).expand(-1,text.config.vocab_size).flatten().tolist(),
                text_visible=((ids!=token)&mask.bool()).reshape(-1,1).expand(-1,text.config.vocab_size).flatten().tolist(),aliases={}))
            if (batch,seq,heads)==(2,8,4):
                control=source.parent.parent/('padded' if padded else 'unmasked')/'cpu.json'
                expected=json.loads(control.read_text())
                actual=json.loads((folder/'cpu.json').read_text())
                def words(values):return b''.join(struct.pack('<f',x) for x in values)
                labels=[('alternate',actual['alternate'],expected['alternate']),
                    ('final',actual['final'],expected['final'])]
                labels.extend((f'step-{i+1}',a['forward'],b['forward'])
                    for i,(a,b) in enumerate(zip(actual['steps'],expected['steps'],strict=True)))
                for label,a,b in labels:
                    if words(a['logits'])!=words(b['logits']):
                        raise ValueError(f'original GPT2 control changed: {name}/{label}')
                manifest.setdefault('original_control',[]).append(dict(case=name,
                    cpu_sha256=digest(control),logit_word_differences=0))
            manifest['cases'].append(dict(profile=profile,name=name,geometry=dict(batch=batch,seq=seq,heads=heads),
                files={p.relative_to(output).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}))
            write_json(output/'manifest.json',manifest)
            print('CPU frozen',name,flush=True)
    snapshots=output/'oracle-source';snapshots.mkdir()
    for name in [Path(__file__).name,'qualify_shared_image_pipeline.py','qualify_multimodal_training.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.is_relative_to(checkout/'src'):
                dest=snapshots/'transformers'/path.relative_to(checkout/'src');dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,dest)
    manifest['oracle']['source_sha256']={p.relative_to(output).as_posix():digest(p) for p in snapshots.rglob('*') if p.is_file()}
    manifest['complete']=True
    write_json(output/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    else:
        from qualify_shared_image_pipeline import run
        args.profiles=None;args.target='rdna3';args.device_id=0x15bf
        run(args)


if __name__=='__main__':main()
