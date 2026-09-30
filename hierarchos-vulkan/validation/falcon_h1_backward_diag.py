"""Ordered forward/backward comparisons with retained native training tape."""
import json
import torch
import verify_falcon_h1 as v

torch.set_num_threads(1)
model=v.tiny().train()
folder=v.ROOT/'.falcon-h1-fixtures/dual/gated_after'
native=json.loads((folder/'native-backward-trace.json').read_text())
captured={}
hooks=[]
def hook(name):
    def save(module,args,out):
        tensor=out[0] if isinstance(out,tuple) else out
        captured['forward.'+name]=tensor.detach()
        tensor.register_hook(lambda grad: captured.__setitem__('backward.'+name,grad.detach()))
    return save
for i,l in enumerate(model.model.layers):
    for name,module in [('ln1',l.input_layernorm),('ln2',l.pre_ff_layernorm),('output',l),('mamba.gated',l.mamba.norm),('mamba.output',l.mamba.out_proj)]:
        hooks.append(module.register_forward_hook(hook(f'layers.{i}.{name}')))
hooks.append(model.model.final_layernorm.register_forward_hook(hook('final_norm')))
ids=torch.tensor([[1,5,9,3,7,2,6]]);mask=torch.tensor([[1,1,0,1,1,1,1]])
logits=model(ids,attention_mask=mask,use_cache=False).logits
logits.retain_grad()
loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.tensor([5,9,3,7,2,6,-100]))
loss.backward()
captured['forward.logits']=logits.detach()
captured['backward.logits']=logits.grad
for h in hooks:h.remove()
for name in ['forward.final_norm','forward.logits','backward.logits','backward.final_norm','backward.layers.1.output','backward.layers.1.ln2','backward.layers.1.ln1','backward.layers.0.output','backward.layers.0.ln2','backward.layers.0.ln1','backward.layers.0.mamba.output']:
    key={'backward.logits':'grad_logits','backward.final_norm':'grad_final_norm_output','backward.layers.1.output':'grad_final_norm_input','backward.layers.0.output':'layers.1.grad_input','backward.layers.0.mamba.output':'layers.0.mamba.grad_output'}.get(name,name.replace('backward.','').replace('.ln1','.grad_ln1').replace('.ln2','.grad_ln2'))
    ref=captured[name];a=torch.tensor(native[key]).reshape_as(ref)
    print(name,'max',float((a-ref).abs().max()),'different',int((a!=ref).sum()))
    if name=='backward.logits':
        nl=torch.tensor(native['forward.logits']).reshape_as(logits).requires_grad_()
        torch.nn.functional.cross_entropy(nl.flatten(0,1),torch.tensor([5,9,3,7,2,6,-100])).backward()
        print('CE kernel isolated',float((a-nl.grad).abs().max()))
