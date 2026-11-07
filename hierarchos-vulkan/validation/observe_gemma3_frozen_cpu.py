"""Read-only CPU stage/VJP observer for the original frozen SigLIP contract."""
import argparse,hashlib,json,os,struct,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--step',type=int,choices=[1,2],default=1);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.fixtures/a.case;manifest=json.loads((a.fixtures/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['name']==a.case)
    for name,expected in {**case['files'],**manifest['oracle']['source_sha256']}.items():
        if hashlib.sha256((a.fixtures/name).read_bytes()).hexdigest()!=expected:raise ValueError('frozen file changed: '+name)
    sys.path.insert(0,str(Path(os.environ['HIERARCHOS_TRANSFORMERS_CHECKOUT'])/'src'))
    # This observer executes vision/text only. Isolate the incompatible optional
    # system torchaudio 2.6 extension from the separate Torch 2.14 CPU runtime.
    sys.modules['torchaudio']=None
    import torch
    from safetensors.torch import load_file
    from transformers import Gemma3ForConditionalGeneration
    torch.set_num_threads(1)
    if torch.__version__!=manifest['oracle']['torch'] or torch.backends.cpu.get_cpu_capability()!=manifest['oracle']['cpu']:
        raise ValueError('observer requires the frozen Torch version and CPU vector contract')
    model=Gemma3ForConditionalGeneration.from_pretrained(folder/'model').float().train()
    model.set_attn_implementation('eager')
    if a.step==2:
        state=load_file(str(folder/'cpu-step-1.safetensors'))
        with torch.no_grad():
            for name,parameter in model.named_parameters():parameter.copy_(state[name])
    f=json.loads((folder/'input.json').read_text());cpu=json.loads((folder/'cpu.json').read_text())
    shape=(f['batch_size'],f['seq_len']);ids=torch.tensor(f['input_ids']).view(shape)
    mask=torch.tensor(f['attention_mask'],dtype=torch.long).view(shape)
    side=model.config.vision_config.image_size
    pixels=torch.tensor(cpu['pixels']).view(f['image_count'],3,side,side).requires_grad_()
    stages={};retained={};handles=[]
    def observe(module,name,before=False):
        def hook(module,inputs,output=None):
            t=inputs[0] if before else output
            if isinstance(t,tuple):t=t[0]
            stages[name]=t.detach().flatten().tolist()
            if t.requires_grad:t.retain_grad();retained[name]=t
        handles.append((module.register_forward_pre_hook if before else module.register_forward_hook)(hook))
    tower=model.model.vision_tower
    observe(tower.embeddings,'tokens')
    for i,layer in enumerate(tower.encoder.layers):
        prefix=f'layer_{i}/'
        for field,name in [('layer_norm1','ln1'),('layer_norm2','ln2')]:observe(getattr(layer,field),prefix+name)
        for field,name in [('q_proj','q'),('k_proj','k'),('v_proj','v'),('out_proj','out_proj')]:observe(getattr(layer.self_attn,field),prefix+name)
        observe(layer.self_attn.out_proj,prefix+'attention',True)
        for field,name in [('fc1','fc1'),('activation_fn','activation'),('fc2','fc2')]:observe(getattr(layer.mlp,field),prefix+name)
        observe(layer,prefix+'out')
    observe(tower.post_layernorm,'post_norm')
    observe(model.model.multi_modal_projector,'projected')
    logits=model(input_ids=ids,attention_mask=mask,pixel_values=pixels,use_cache=False).logits
    targets=torch.tensor(f['targets']).view(shape);targets[targets==0xffffffff]=-100
    weights=torch.tensor(f['loss_weights']).view(shape)
    loss=(torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view(shape)*weights).sum()/weights.sum()
    loss.backward()
    expected=cpu['steps'][a.step-1]['forward']['logits'];actual=logits.detach().flatten().tolist()
    bits=lambda x:struct.pack('<f',x)
    differences=sum(bits(x)!=bits(y) for x,y in zip(actual,expected,strict=True))
    maximum=max(abs(x-y) for x,y in zip(actual,expected,strict=True))
    print('frozen CPU logit replay',differences,'word differences; max',maximum,flush=True)
    for handle in handles:handle.remove()
    result=dict(torch=torch.__version__,cpu=torch.backends.cpu.get_cpu_capability(),step=a.step,
        frozen_logit_word_differences=differences,frozen_logit_max_abs=maximum,stages=stages,
        gradients={k:v.grad.flatten().tolist() for k,v in retained.items() if v.grad is not None},
        parameter_gradients={k:v.grad.flatten().tolist() for k,v in model.named_parameters() if v.grad is not None},
        pixel_gradient=pixels.grad.flatten().tolist(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as file:json.dump(result,file,separators=(',',':'))

if __name__=='__main__':main()
