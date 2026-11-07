"""Observe a frozen shared-image CPU text graph; require exact frozen replay first."""
from pathlib import Path
import argparse,json,os,struct,sys
from qualify_multimodal_training import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backward',action='store_true')
    p.add_argument('--step',type=int,choices=[1,2],default=1)
    a=p.parse_args();folder=a.fixtures/a.case;manifest=json.loads((a.fixtures/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['name']==a.case)
    for name,h in {**case['files'],**manifest['oracle']['source_sha256']}.items():
        if digest(a.fixtures/name)!=h:raise ValueError('changed frozen CPU file '+name)
    sys.path.insert(0,str(Path(os.environ.get('HIERARCHOS_TRANSFORMERS_CHECKOUT','C:/Users/User/transformers'))/'src'))
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(1)
    model=AutoModelForCausalLM.from_pretrained(folder/'model/text').float().train()
    if a.step==2:
        from safetensors.torch import load_file
        state=load_file(str(folder/'cpu-step-1.safetensors'))
        with torch.no_grad():
            for name,parameter in model.named_parameters():parameter.copy_(state['text.'+name])
    f=json.loads((folder/'input.json').read_text());cpu=json.loads((folder/'cpu.json').read_text())
    shape=(f['batch_size'],f['seq_len']);ids=torch.tensor(f['input_ids']).view(shape)
    mask=torch.tensor(f['attention_mask'],dtype=torch.long).view(shape)
    image_config=json.loads((folder/'model/image/config.json').read_text())
    positions=(ids==image_config['image_token_index']).flatten().nonzero().flatten()
    embedding=model.get_input_embeddings();width=embedding.weight.shape[1]
    projected=torch.tensor(cpu['steps'][a.step-1]['forward']['projected_features']).view(-1,width)
    handles=[];stages={};attention_calls=[];retained={};attention_retained=[]
    original_sdpa=torch.nn.functional.scaled_dot_product_attention
    def observed_sdpa(query,key,value,*args,**kwargs):
        output=original_sdpa(query,key,value,*args,**kwargs)
        if a.backward:
            for tensor in [query,key,value,output]:tensor.retain_grad()
            attention_retained.append([query,key,value,output])
        def tensor_record(t):
            if t is None:return None
            return {'shape':list(t.shape),'values':t.detach().flatten().tolist()}
        attention_calls.append({'q':tensor_record(query),'k':tensor_record(key),
            'v':tensor_record(value),'output':tensor_record(output),
            'mask':tensor_record(kwargs.get('attn_mask',args[0] if args else None)),
            'is_causal':kwargs.get('is_causal',False),'scale':kwargs.get('scale'),
            'enable_gqa':kwargs.get('enable_gqa',False)})
        return output
    torch.nn.functional.scaled_dot_product_attention=observed_sdpa
    def replace_rows(module,inputs,output):
        replacement=projected*getattr(module,'embed_scale',1.)
        return output.reshape(-1,width).index_copy(0,positions,replacement).reshape_as(output)
    handles.append(embedding.register_forward_hook(replace_rows))
    def observe(module,name,before=False):
        def hook(m,inputs,output=None):
            value=inputs[0] if before else output
            if isinstance(value,tuple):value=value[0]
            stages[name]=value.detach().flatten().tolist()
            if a.backward and value.requires_grad:
                value.retain_grad();retained[name]=value
        handles.append((module.register_forward_pre_hook if before else module.register_forward_hook)(hook))
    def observe_norm(module,name):
        def hook(m,inputs):
            value=inputs[0].detach().float()
            if isinstance(m,torch.nn.LayerNorm):
                _,mean,rstd=torch.native_layer_norm(value,m.normalized_shape,m.weight,m.bias,m.eps)
            else:
                mean=value.pow(2).mean(-1,keepdim=True)+getattr(m,'variance_epsilon',getattr(m,'eps',1e-6))
                rstd=torch.rsqrt(mean)
            stages[name+'/mean']=mean.flatten().tolist()
            stages[name+'/rstd']=rstd.flatten().tolist()
        handles.append(module.register_forward_pre_hook(hook))
    observe(embedding,'lookup')
    architecture=model.config.model_type
    if architecture=='gpt2':
        for i,layer in enumerate(model.transformer.h):
            prefix=f'layer_{i}/';observe(layer,prefix+'input',True)
            observe_norm(layer.ln_1,prefix+'n1');observe_norm(layer.ln_2,prefix+'n3')
            for module,name in [(layer.ln_1,'n1'),(layer.attn.c_attn,'qkv'),(layer.attn.c_proj,'o'),
                (layer.ln_2,'n3'),(layer.mlp.c_fc,'u'),(layer.mlp.act,'activation'),(layer.mlp.c_proj,'d'),(layer,'out')]:observe(module,prefix+name)
            observe(layer.attn.c_proj,prefix+'a',True)
            observe(layer.ln_2,prefix+'r',True)
        observe(model.transformer.ln_f,'final_norm')
    else:
        for i,layer in enumerate(model.model.layers):
            prefix=f'layer_{i}/';observe(layer,prefix+'input',True)
            observe_norm(layer.input_layernorm,prefix+'n1')
            for module,name in [(layer.input_layernorm,'n1'),(layer.self_attn.q_proj,'q'),(layer.self_attn.k_proj,'k'),
                (layer.self_attn.v_proj,'v'),(layer.self_attn.o_proj,'o'),(layer.mlp.gate_proj,'g'),
                (layer.mlp.up_proj,'u'),(layer.mlp.down_proj,'d'),(layer,'out')]:observe(module,prefix+name)
            observe(layer.self_attn.o_proj,prefix+'a',True);observe(layer.mlp.down_proj,prefix+'p',True)
            if architecture=='gemma4_text':
                for field,name in [('post_attention_layernorm','n2'),('pre_feedforward_layernorm','n3'),('post_feedforward_layernorm','n4')]:observe(getattr(layer,field),prefix+name)
                observe(layer.pre_feedforward_layernorm,prefix+'r',True)
                for field,name in [('q_norm','qn'),('k_norm','kn')]:observe(getattr(layer.self_attn,field),prefix+name)
            else:
                observe(layer.post_attention_layernorm,prefix+'n3');observe(layer.post_attention_layernorm,prefix+'r',True)
        observe(model.model.norm,'final_norm')
    with torch.set_grad_enabled(a.backward):
        result=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        logits=result.detach().flatten().tolist()
        if a.backward:
            result.retain_grad();retained['logits']=result
            targets=torch.tensor(f['targets'],dtype=torch.long).view(shape);targets[targets==0xffffffff]=-100
            loss_weights=torch.tensor(f['loss_weights']).view(shape)
            losses=torch.nn.functional.cross_entropy(result.reshape(-1,result.shape[-1]),targets.flatten(),reduction='none').view(shape)
            loss=(losses*loss_weights).sum()/loss_weights.sum();loss.backward()
    gradients={name:t.grad.flatten().tolist() for name,t in retained.items() if t.grad is not None}
    for call,tensors in zip(attention_calls,attention_retained):
        for key,tensor in zip(['dq','dk','dv','dy'],tensors):call[key]={'shape':list(tensor.shape),'values':tensor.grad.flatten().tolist()}
    for handle in handles:handle.remove()
    torch.nn.functional.scaled_dot_product_attention=original_sdpa
    expected=cpu['steps'][a.step-1]['forward']['logits']
    words=lambda values:[struct.pack('<f',x) for x in values]
    different=sum(x!=y for x,y in zip(words(logits),words(expected),strict=True))
    print('CPU replay differences',different,'attention implementation',model.config._attn_implementation,flush=True)
    if different:raise ValueError('CPU diagnostic does not reproduce frozen logits')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'cpu.json',{'stages':stages,'logits':logits,'cpu_replay_word_differences':different,
        'attention_implementation':model.config._attn_implementation,'step':a.step,'runner_sha256':digest(__file__),
        'fixtures_sha256':digest(a.fixtures/'manifest.json'),'attention_calls':attention_calls,'gradients':gradients,
        'parameter_gradients':{name:p.grad.detach().flatten().tolist() for name,p in model.named_parameters() if p.grad is not None}})


if __name__=='__main__':main()
