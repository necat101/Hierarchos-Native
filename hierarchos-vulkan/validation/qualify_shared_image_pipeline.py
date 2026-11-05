"""Freeze and replay full CLIP image training for the green text-backbone inventory.

These explicit shared-adapter compositions are distinct from upstream native
multimodal families. CPU preparation does not call native code. Frozen replay
uses Python's standard library and retains missing inventory coverage as pending.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from qualify_multimodal_training import compare, digest, oracle_env, read_f32_tensors, validate_device, write_json

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 2e-7
INVENTORY = Path(__file__).with_name('multimodal_training_inventory.json')


def prepare(args):
    import torch
    import numpy as np
    import PIL
    from PIL import Image
    from safetensors.torch import save_file
    import verify_hf_logits as reference
    from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
    from verify_falcon_h1 import tiny
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(0x53484D4D)
    required = [p['profile'] for p in json.loads(INVENTORY.read_text())['profiles']]
    if args.profiles is not None and (not args.profiles or set(args.profiles)-set(required)):
        raise ValueError('unknown or empty requested green inventory')
    models = reference.tiny_models(training_reference=True)+reference.production_width_models()
    models.extend((name,tiny(norm=norm,before=before),False) for name,norm,before in [
        ('falcon_h1_gated_after',True,False),('falcon_h1_gated_before',True,True),('falcon_h1_silu',False,False)])
    models.append(('qwen3_next',Qwen3NextForCausalLM(Qwen3NextConfig(
        vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,max_position_embeddings=16,attention_dropout=0.,use_cache=False,
        tie_word_embeddings=True,attn_implementation='eager',head_dim=8,linear_key_head_dim=8,
        linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=2,linear_conv_kernel_dim=3,
        num_experts=2,num_experts_per_tok=2,moe_intermediate_size=16,shared_expert_intermediate_size=16,
        mlp_only_layers=[0,1],layer_types=['linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.})),False))
    selected = set(args.profiles or required)
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest = {'schema':1,'complete':False,'scope':'shared_clip_full_parameter_image_text_training',
        'absolute_limit':LIMIT,'rtol':0,'comparison':'strict_less_than','cases':[],
        'required_green_profiles':required,'requested_profiles':sorted(selected),'pending':[],
        'oracle':{'cpu':torch.backends.cpu.get_cpu_capability(),'torch':torch.__version__,
            'torch_revision':torch.version.git_version,'numpy':np.__version__,'pillow':PIL.__version__,
            'threads':1,'transformers_revision':subprocess.check_output(['git','-C',str(reference.TRANSFORMERS_SOURCE.parent),
                'rev-parse','HEAD'],text=True).strip()}}
    available={name for name,_,_ in models}
    manifest['pending'].extend({'profile':name,'reason':'dedicated runtime image bridge pending'} for name in sorted(selected-available))
    for name,base_model,_ in models:
        if name not in selected:continue
        if base_model.config.is_encoder_decoder:
            manifest['pending'].append({'profile':name,'reason':'joint encoder-decoder image training pending'})
            continue
        for padded in [False,True]:
            if padded and name in reference.KIMI_K3_ORACLE_FAMILIES:continue
            case = name+'/'+('padded' if padded else 'unmasked')
            folder=out/case;package=folder/'model';package.mkdir(parents=True)
            text=copy.deepcopy(base_model).float().train()
            for module in text.modules():
                for field in ['jitter_noise','moe_jitter_eps']:
                    if hasattr(module,field):setattr(module,field,0.)
                if isinstance(module,torch.nn.Dropout) and module.p:raise ValueError('dropout requires a shared RNG oracle')
            if hasattr(text.config,'router_jitter_noise'):text.config.router_jitter_noise=0.
            if name=='dbrx':text.config.ffn_config.moe_jitter_eps=0.
            reference.save_reference_checkpoint(name,text,package/'text')
            embedding=text.get_input_embeddings();width=embedding.weight.shape[1]
            vision_config=CLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                num_attention_heads=2,image_size=8,patch_size=4,num_channels=3,hidden_act='quick_gelu',attention_dropout=0.)
            torch.manual_seed(0x434C4950+sum(name.encode()))
            tower=CLIPVisionModel(vision_config).float().train()
            tower.set_attn_implementation('eager')
            projector=torch.nn.Sequential(torch.nn.Linear(16,width),torch.nn.GELU(),torch.nn.Linear(width,width))
            with torch.no_grad():
                for p in projector.parameters():p.mul_(.15)
            image_parameters={'vision_tower.vision_model.'+k:v for k,v in tower.named_parameters()}
            image_parameters.update({'multi_modal_projector.linear_1.'+k:v for k,v in projector[0].named_parameters()})
            image_parameters.update({'multi_modal_projector.linear_2.'+k:v for k,v in projector[2].named_parameters()})
            image=package/'image';image.mkdir()
            save_file({k:v.detach().contiguous().clone() for k,v in image_parameters.items()},str(image/'model.safetensors'))
            vocab=embedding.weight.shape[0];token=vocab-1
            write_json(package/'config.json',{'model_type':'hierarchos_image_text','schema':1,'text_profile':name})
            write_json(image/'config.json',{'model_type':'hierarchos_clip_adapter','vision_config':vision_config.to_dict(),
                'vision_feature_layer':-1,'vision_feature_select_strategy':'default','projector_hidden_act':'gelu',
                'multimodal_projector_bias':True,'image_token_index':token})
            processor=CLIPImageProcessor(do_resize=False,do_center_crop=False)
            processor.save_pretrained(image)
            rgb=np.random.default_rng(0x494D47+sum(name.encode())).integers(0,256,(2,8,8,3),dtype=np.uint8)
            alt=255-rgb
            pixels=processor(images=[Image.fromarray(x) for x in rgb],return_tensors='pt')['pixel_values'].requires_grad_()
            alternate=processor(images=[Image.fromarray(x) for x in alt],return_tensors='pt')['pixel_values']
            ids=torch.tensor([[1,token,token,token,token,7,3,2],[2,token,token,token,token,3,7,1]])
            mask=torch.ones_like(ids)
            if padded:mask[0,-1]=0;mask[1,0]=0
            positions=(ids==token).flatten().nonzero().flatten()
            targets=ids.roll(-1,dims=1);weights=mask.float();weights[:,-1]=0;weights[:,:-1]*=mask[:,1:]
            weights[:,:4]=0;targets[weights==0]=-100
            fixture={'batch_size':2,'seq_len':8,'image_count':2,'input_ids':ids.flatten().tolist(),
                'attention_mask':mask.float().flatten().tolist(),'targets':[x if x>=0 else 0xffffffff for x in targets.flatten().tolist()],
                'loss_weights':weights.flatten().tolist(),'rgb':rgb.flatten().tolist(),'alternate_rgb':alt.flatten().tolist(),
                'steps':2,'lr':.001,'beta1':.9,'beta2':.999,'eps':.001,'weight_decay':0.}
            write_json(folder/'input.json',fixture)
            holder={}
            def replace_rows(module,inputs,output):
                replacement=holder['projected']*getattr(module,'embed_scale',1.)
                return output.reshape(-1,width).index_copy(0,positions,replacement).reshape_as(output)
            handle=embedding.register_forward_hook(replace_rows)
            def forward(values):
                features=tower(values,output_hidden_states=True).hidden_states[-1][:,1:,:].reshape(-1,16)
                projected=projector(features);holder['projected']=projected
                logits=text(input_ids=ids,attention_mask=mask,use_cache=False).logits
                return {'logits':logits,'vision_features':features,'projected_features':projected}
            def record(result):return {k:v.detach().flatten().tolist() for k,v in result.items()}
            with torch.no_grad():alternate_result=record(forward(alternate))
            optimizer=torch.optim.AdamW([*text.parameters(),*tower.parameters(),*projector.parameters()],
                lr=.001,betas=(.9,.999),eps=.001,weight_decay=0.,foreach=False,fused=False)
            steps=[]
            for step in [1,2]:
                optimizer.zero_grad(set_to_none=True);pixels.grad=None
                result=forward(pixels);holder['projected'].retain_grad()
                row_loss=torch.nn.functional.cross_entropy(result['logits'].reshape(-1,vocab),targets.flatten(),reduction='none').reshape_as(weights)
                loss=(row_loss*weights).sum()/weights.sum();loss.backward()
                steps.append({'forward':record(result),'loss':loss.item(),'pixel_gradient':pixels.grad.flatten().tolist(),
                    'projected_gradient':holder['projected'].grad.flatten().tolist()})
                optimizer.step()
                parameters={'text.'+k:v for k,v in text.named_parameters()}
                parameters.update({'image.'+k:v for k,v in image_parameters.items()})
                save_file({k:v.detach().contiguous().clone() for k,v in parameters.items()},str(folder/f'cpu-step-{step}.safetensors'))
            with torch.no_grad():final=record(forward(pixels))
            handle.remove()
            write_json(folder/'cpu.json',{'pixels':pixels.detach().flatten().tolist(),'alternate':alternate_result,'steps':steps,'final':final,
                'visible':mask.bool().reshape(-1,1).expand(-1,vocab).flatten().tolist(),
                'text_visible':((ids!=token)&mask.bool()).reshape(-1,1).expand(-1,vocab).flatten().tolist(),
                'aliases':{'text.'+k:'text.'+v for k,v in getattr(text,'all_tied_weights_keys',{}).items()}})
            manifest['cases'].append({'profile':name,'name':case,'files':{p.relative_to(out).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}})
            write_json(out/'manifest.json',manifest);print('CPU',case,'full image training frozen',flush=True)
    source=out/'oracle-source';source.mkdir()
    for name in [Path(__file__).name,'qualify_multimodal_training.py','verify_hf_logits.py','verify_falcon_h1.py','kimi_k3_reference.py']:
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.suffix=='.py' and p.is_relative_to(reference.TRANSFORMERS_SOURCE):
                dest=source/'transformers'/p.relative_to(reference.TRANSFORMERS_SOURCE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    manifest['oracle']['source_sha256']={p.relative_to(out).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    manifest['complete']=True;write_json(out/'manifest.json',manifest)


def validate_manifest(manifest,profiles):
    if (manifest.get('complete'),manifest.get('absolute_limit'),manifest.get('rtol'),manifest.get('comparison'))!=(True,LIMIT,0,'strict_less_than'):
        raise ValueError('altered or incomplete frozen image acceptance policy')
    cases=manifest['cases'];names=[c['name'] for c in cases]
    if not names or len(names)!=len(set(names)):raise ValueError('empty or duplicate frozen image cases')
    if profiles is not None:
        if not profiles or set(profiles)-{c['profile'] for c in cases}:raise ValueError('empty or unknown requested profiles')
        cases=[c for c in cases if c['profile'] in profiles]
    return cases


def inventory_coverage(required,results):
    """A profile needs every originally covered mask case, not any passing row."""
    if not required or len(required)!=len(set(required)):
        raise ValueError('empty or duplicate green inventory')
    by_name={r['case']:r for r in results}
    if len(by_name)!=len(results):raise ValueError('duplicate replay cases')
    unpadded_only={'kimi_k3_text','kimi_k3_kda','kimi_k3_mla'}
    missing=[]
    for profile in required:
        expected=[profile+'/unmasked']
        if profile not in unpadded_only:expected.append(profile+'/padded')
        if any(name not in by_name or by_name[name]['profile']!=profile or not by_name[name]['pass'] for name in expected):
            missing.append(profile)
    return sorted(missing)


def run(args):
    root=args.fixtures.resolve();manifest=json.loads((root/'manifest.json').read_text());cases=validate_manifest(manifest,args.profiles)
    for name,h in manifest['oracle']['source_sha256'].items():
        if digest(root/name)!=h:raise ValueError('frozen CPU source changed: '+name)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=digest(args.binary);results=[]
    def persist():
        missing=inventory_coverage(manifest['required_green_profiles'],results)
        write_json(out/'qualification.json',{'schema':1,'scope':manifest['scope'],'absolute_limit':LIMIT,'rtol':0,
            'comparison':'strict_less_than','target':args.target,'binary_sha256':binary,'runner_sha256':digest(__file__),
            'fixtures_sha256':digest(root/'manifest.json'),'expected_cases':len(cases),'completed_cases':len(results),
            'complete':len(cases)==len(results),'results':results,'missing_green_profiles':missing,
            'all_green_backbones_qualified':not missing and len(cases)==len(results) and all(r['pass'] for r in results),
            'pass':len(cases)==len(results) and all(r['pass'] for r in results),'aten_vector_width':8 if manifest['oracle']['cpu']=='AVX2' else 16})
    persist()
    for case in cases:
        for name,h in case['files'].items():
            if digest(root/name)!=h:raise ValueError('frozen CPU fixture changed: '+name)
        folder=root/case['name'];dest=out/case['name'];dest.parent.mkdir(parents=True,exist_ok=True)
        process=subprocess.run([str(args.binary.resolve()),'--model',str(folder/'model'),'--fixture',str(folder/'input.json'),'--output',str(dest)],
            text=True,capture_output=True,env=oracle_env(manifest['oracle']['cpu']))
        (dest.parent/(dest.name+'.log')).write_text(process.stdout+process.stderr)
        checks=[];diagnostics=[]
        if process.returncode:
            results.append({'case':case['name'],'profile':case['profile'],'pass':False,'exit_code':process.returncode});persist();print('FAIL',case['name'],'exit',process.returncode,flush=True);continue
        actual=json.loads((dest/'report.json').read_text());cpu=json.loads((folder/'cpu.json').read_text());validate_device(actual,args.target,args.device_id)
        def check(label,a,b,required=True,visible=None):
            if visible is not None:
                if len(a)!=len(visible) or len(b)!=len(visible) or not all(math.isfinite(x) for x in a+b):raise ValueError('invalid logit coverage')
                a=[x for x,keep in zip(a,visible) if keep];b=[x for x,keep in zip(b,visible) if keep]
            (checks if required else diagnostics).append({'metric':label,**compare(a,b)})
        check('RGB preprocessing',actual['pixels'],cpu['pixels'])
        def forward(label,a,b):
            check(label+'/logits',a['logits'],b['logits'],visible=cpu['visible'])
            for key in ['vision_features','projected_features']:check(label+'/'+key,a[key],b[key],False)
        forward('alternate',actual['alternate'],cpu['alternate'])
        if len(actual['steps'])!=2:raise ValueError('two full training updates required')
        for index,(a,b) in enumerate(zip(actual['steps'],cpu['steps'],strict=True),1):
            forward(f'step-{index}',a['forward'],b['forward'])
            delta=abs(a['training']['text']['loss']-b['loss']);checks.append({'metric':f'step-{index}/loss','max_abs':delta,'pass':math.isfinite(delta) and delta<2e-5})
            for key in ['pixel_gradient','projected_gradient']:check(f'step-{index}/'+key,a['training'][key],b[key],False)
            native={prefix+'.'+k:v for prefix in ['text','image'] for k,v in read_f32_tensors(dest/f'step-{index}'/prefix/'model.safetensors').items()}
            for name,(shape,values) in read_f32_tensors(folder/f'cpu-step-{index}.safetensors').items():
                key=name if name in native else cpu['aliases'].get(name,name)
                if key not in native or native[key][0]!=shape:raise ValueError('missing or reshaped '+name)
                check(f'step-{index}/parameter/'+name,native[key][1],values)
        forward('final',actual['final'],cpu['final'])
        for prefix,group in [('text',''),('image','vision_tower.'),('image','multi_modal_projector.')]:
            initial=read_f32_tensors(folder/'model'/prefix/'model.safetensors');trained=read_f32_tensors(dest/'step-2'/prefix/'model.safetensors')
            if prefix=='text' and 'final_logits_bias' in initial:
                expected=initial['final_logits_bias']
                for step in [1,2]:
                    buffer=read_f32_tensors(dest/f'step-{step}'/prefix/'model.safetensors').get('final_logits_bias')
                    if buffer is None or buffer[0]!=expected[0]:raise ValueError('missing fixed final_logits_bias buffer')
                    differences=sum(struct.pack('<f',x)!=struct.pack('<f',y) for x,y in zip(buffer[1],expected[1],strict=True))
                    checks.append({'metric':f'step-{step}/fixed final_logits_bias','word_differences':differences,'pass':differences==0})
            change=max(abs(x-y) for name,(_,values) in initial.items() if name.startswith(group) for x,y in zip(values,trained[name][1],strict=True))
            checks.append({'metric':prefix+'/'+group+' parameters updated','max_abs_change':change,'pass':change>0.})
        sensitivity=max(abs(x-y) for x,y,keep in zip(actual['alternate']['logits'],actual['steps'][0]['forward']['logits'],cpu['text_visible'],strict=True) if keep)
        checks.append({'metric':'image affects visible text logits','max_abs_change':sensitivity,'pass':sensitivity>0.})
        gradient=actual['steps'][0]['training']['pixel_gradient']
        checks.append({'metric':'loss reaches RGB pixels','pass':all(math.isfinite(x) for x in gradient) and max(abs(x) for x in gradient)>0.})
        differences=sum(struct.pack('<f',x)!=struct.pack('<f',y) for x,y in zip(actual['final']['logits'],actual['reload']['logits'],strict=True))
        checks.append({'metric':'native export/reload','word_differences':differences,'pass':differences==0})
        failures=[c for c in checks if not c['pass']]
        results.append({'case':case['name'],'profile':case['profile'],'device':actual['device'],'pass':not failures,
            'checks':checks,'strict_intermediate_diagnostics':diagnostics,'elapsed_seconds':actual['elapsed_seconds']})
        persist();print('FAIL' if failures else 'PASS',case['name'],len(failures),'required failures',flush=True)
        for failure in failures[:6]:print(failure,flush=True)
    if digest(args.binary)!=binary:raise ValueError('native binary changed during qualification')
    if not all(r['pass'] for r in results):raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True);p.add_argument('--profiles',nargs='+')
    p=sub.add_parser('run');p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profiles',nargs='+');p.add_argument('--target',choices=['gen9','rdna3'],required=True)
    p.add_argument('--device-id',type=lambda x:int(x,0))
    a=parser.parse_args();(prepare if a.command=='prepare' else run)(a)


if __name__=='__main__':main()
