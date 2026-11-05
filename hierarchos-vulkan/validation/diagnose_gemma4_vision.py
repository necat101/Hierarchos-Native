"""Compare ordinary native image boundaries with the unmodified frozen CPU graph.

This diagnostic cannot qualify a pipeline. Its CPU first logits must reproduce
the frozen oracle word for word before any boundary comparison is accepted.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import faulthandler
import struct
from qualify_gemma4_pipeline import digest, oracle_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--case',default='gemma4_default')
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    folder=a.fixtures.resolve()/a.case
    f=json.loads((folder/'input.json').read_text())
    oracle=json.loads((folder/'cpu.json').read_text())
    manifest=json.loads((a.fixtures/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['name']==a.case)
    for name,expected in {**manifest['oracle']['source_sha256'],**case['files']}.items():
        if digest(a.fixtures/name)!=expected:raise ValueError('frozen CPU file changed: '+name)
    pixel_domain=case.get('pixel_input_domain','raw_patch_values')
    binary_hash=digest(a.binary)
    native_path=a.output/'native.json'
    subprocess.run([str(a.binary.resolve()),'--model',str(folder/'model'),
        '--fixture',str(folder/'input.json'),'--output',str(a.output/'unused'),
        '--stages',str(native_path),'--pixel-domain',pixel_domain],check=True,
        env=oracle_env(manifest['oracle']['cpu']))
    native=json.loads(native_path.read_text())
    checkout=Path(os.environ.get('HIERARCHOS_TRANSFORMERS_CHECKOUT','C:/Users/User/transformers'))
    sys.path.insert(0,str(checkout/'src'))
    print('Native stages saved; loading pinned CPU modules',flush=True)
    faulthandler.dump_traceback_later(60,repeat=True)
    import torch
    from transformers import Gemma4ForConditionalGeneration
    import transformers.models.gemma4.modeling_gemma4 as gm
    torch.set_num_threads(1)
    print('Loading frozen CPU package',flush=True)
    model=Gemma4ForConditionalGeneration.from_pretrained(folder/'model').float().train()
    model.set_attn_implementation('eager')
    seen={};handles=[]
    def register(module,name,pre=False):
        def keep(m,inputs,output=None):
            t=inputs[0] if pre else output
            if isinstance(t,tuple):t=t[0]
            seen[name]=t.detach().flatten().tolist()
        handles.append((module.register_forward_pre_hook if pre else module.register_forward_hook)(keep))
    tower=model.model.vision_tower
    text=model.model.language_model
    register(text.layers[0],'text/input',True)
    for i,l in enumerate(text.layers):
        prefix=f'text/layer_{i}/'
        for module,name in [(l.input_layernorm,'n1'),(l.self_attn.q_proj,'q'),
            (l.self_attn.q_norm,'qn'),(l.self_attn.k_proj,'k'),(l.self_attn.k_norm,'kn'),
            (l.self_attn.v_proj,'v'),(l.self_attn.o_proj,'o'),
            (l.post_attention_layernorm,'n2'),(l.pre_feedforward_layernorm,'n3'),
            (l.mlp.gate_proj,'g'),(l.mlp.up_proj,'u'),(l.mlp.down_proj,'d'),
            (l.post_feedforward_layernorm,'n4'),(l,'out')]:register(module,prefix+name)
        register(l.self_attn.o_proj,prefix+'a',True)
        register(l.pre_feedforward_layernorm,prefix+'r',True)
        register(l.mlp.down_proj,prefix+'p',True)
    register(text.norm,'text/final_norm')
    eager=gm.eager_attention_forward
    text_attention=[]
    def keep_eager(module,query,key,value,*args,**kwargs):
        if module.config.model_type=='gemma4_text':
            prefix=f'text/layer_{module.layer_idx}/'
            for name,t in [('qr',query),('kr',key),('vn',value)]:
                seen[prefix+name]=t.transpose(1,2).detach().flatten().tolist()
            text_attention.append({'layer':module.layer_idx,'q':query.detach().flatten().tolist(),
                'k':key.detach().flatten().tolist(),'v':value.detach().flatten().tolist(),
                'shape':list(query.shape),'scale':kwargs.get('scaling',module.scaling)})
        return eager(module,query,key,value,*args,**kwargs)
    gm.eager_attention_forward=keep_eager
    register(tower.patch_embedder.input_proj,'patch_input',True)
    register(tower.patch_embedder.input_proj,'patch_output')
    register(tower.patch_embedder,'tokens')
    rope=gm.apply_multidimensional_rope;rope_index=[0]
    def keep_rope(*args,**kwargs):
        result=rope(*args,**kwargs);i=rope_index[0];rope_index[0]+=1
        seen[f'layer_{i//2}/'+('qr' if i%2==0 else 'kr')]=result.detach().flatten().tolist()
        return result
    gm.apply_multidimensional_rope=keep_rope
    for i,l in enumerate(tower.encoder.layers):
        prefix=f'layer_{i}/'
        for module,name in [(l.input_layernorm,'n1'),(l.self_attn.q_proj,'q'),
            (l.self_attn.q_norm,'qn'),(l.self_attn.k_proj,'k'),(l.self_attn.k_norm,'kn'),
            (l.self_attn.v_proj,'v'),(l.self_attn.v_norm,'vn'),(l.self_attn.o_proj,'o'),
            (l.post_attention_layernorm,'n2'),(l.pre_feedforward_layernorm,'n3'),
            (l.mlp.gate_proj,'g'),(l.mlp.act_fn,'ga'),(l.mlp.up_proj,'u'),
            (l.mlp.down_proj,'d'),(l.post_feedforward_layernorm,'n4'),(l,'out')]:
            register(module,prefix+name)
        register(l.self_attn.o_proj,prefix+'a',True)
        register(l.pre_feedforward_layernorm,prefix+'r',True)
        register(l.mlp.down_proj,prefix+'p',True)
    register(model.model.embed_vision.embedding_projection,'pooled_norm',True)
    register(model.model.embed_vision,'projected')
    def pool_hook(m,inputs,output):seen['vision_features']=output[0].detach().flatten().tolist()
    handles.append(tower.pooler.register_forward_hook(pool_hook))
    batch,seq=f['batch_size'],f['seq_len']
    pixels=torch.tensor(oracle['pixels']).view(1,-1,3*model.config.vision_config.patch_size**2)
    kwargs=dict(input_ids=torch.tensor(f['input_ids']).view(batch,seq),
        attention_mask=torch.tensor(f['attention_mask'],dtype=torch.long).view(batch,seq),
        pixel_values=pixels,image_position_ids=torch.tensor(f['positions']).view(1,-1,2),use_cache=False)
    with torch.no_grad():logits=model(**kwargs).logits.flatten().tolist()
    for h in handles:h.remove()
    gm.apply_multidimensional_rope=rope
    gm.eager_attention_forward=eager
    word=lambda x:struct.pack('<f',x)
    if [word(x) for x in logits]!=[word(x) for x in oracle['steps'][0]['forward']['logits']]:
        raise ValueError('CPU diagnostic does not reproduce the frozen oracle')
    faulthandler.cancel_dump_traceback_later()
    rows=[]
    for name,expected in seen.items():
        actual=native['text']['stages'].get(name[5:]) if name.startswith('text/') else native.get(name)
        if actual is None or len(actual)!=len(expected):raise ValueError('missing/reshaped native stage '+name)
        row={'stage':name,'values':len(expected),'different_words':sum(word(x)!=word(y) for x,y in zip(actual,expected)),
            'max_abs':max(abs(x-y) for x,y in zip(actual,expected))}
        rows.append(row);print(name,row['different_words'],row['max_abs'],flush=True)
    (a.output/'cpu.json').write_text(json.dumps({'stages':seen,'logits':logits,'text_attention':text_attention},allow_nan=False))
    if digest(a.binary)!=binary_hash:raise ValueError('native binary changed during diagnostic')
    (a.output/'diff.json').write_text(json.dumps({'cpu_replay_max_abs':0.,'cpu_replay_word_differences':0,
        'binary_sha256':binary_hash,'runner_sha256':digest(__file__),
        'fixtures_sha256':digest(a.fixtures/'manifest.json'),'pixel_input_domain':pixel_domain,
        'stages':rows},indent=1,allow_nan=False))


if __name__=='__main__':main()
