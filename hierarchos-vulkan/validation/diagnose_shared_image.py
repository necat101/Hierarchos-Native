"""Observe the frozen image VJP, seeded with its original text gradient."""
from pathlib import Path
import argparse,json,struct,sys,os
from qualify_multimodal_training import digest,write_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.fixtures/'manifest.json').read_text());case=next(c for c in manifest['cases'] if c['name']==a.case)
 for name,h in {**case['files'],**manifest['oracle']['source_sha256']}.items():
  if digest(a.fixtures/name)!=h:raise ValueError('changed frozen CPU file '+name)
 sys.path.insert(0,str(Path(os.environ.get('HIERARCHOS_TRANSFORMERS_CHECKOUT','C:/Users/User/transformers'))/'src'))
 import torch
 from safetensors.torch import load_file
 from transformers import CLIPVisionConfig,CLIPVisionModel
 torch.set_num_threads(1);folder=a.fixtures/a.case;cpu=json.loads((folder/'cpu.json').read_text());config=json.loads((folder/'model/image/config.json').read_text())
 state=load_file(str(folder/'model/image/model.safetensors'))
 tower=CLIPVisionModel(CLIPVisionConfig(**config['vision_config'])).float().train();tower.set_attn_implementation('eager')
 tower.load_state_dict({k.removeprefix('vision_tower.vision_model.'):v for k,v in state.items() if k.startswith('vision_tower.')})
 width=state['multi_modal_projector.linear_1.weight'].shape[0]
 projector=torch.nn.Sequential(torch.nn.Linear(16,width),torch.nn.GELU(),torch.nn.Linear(width,width))
 for index,name in [(0,'linear_1'),(2,'linear_2')]:projector[index].load_state_dict({k:state[f'multi_modal_projector.{name}.{k}'] for k in ['weight','bias']})
 stages={};retained={};handles=[]
 def observe(module,name,before=False):
  def hook(m,inputs,output=None):
   t=inputs[0] if before else output
   if isinstance(t,tuple):t=t[0]
   if hasattr(t,'last_hidden_state'):t=t.last_hidden_state
   stages[name]=t.detach().flatten().tolist()
   if t.requires_grad:t.retain_grad();retained[name]=t
  handles.append((module.register_forward_pre_hook if before else module.register_forward_hook)(hook))
 observe(tower.embeddings,'tokens');observe(tower.pre_layrnorm,'pre_output')
 for i,layer in enumerate(tower.encoder.layers):
  prefix=f'layer_{i}/'
  for module,name in [(layer.layer_norm1,'n1'),(layer.self_attn.q_proj,'q'),(layer.self_attn.k_proj,'k'),(layer.self_attn.v_proj,'v'),(layer.self_attn.out_proj,'o'),(layer.layer_norm2,'n2'),(layer.mlp.fc1,'f'),(layer.mlp.activation_fn,'act'),(layer.mlp.fc2,'out_linear'),(layer,'out')]:observe(module,prefix+name)
  observe(layer.self_attn.out_proj,prefix+'a',True);observe(layer.layer_norm2,prefix+'r',True)
 observe(projector[0],'projector_pre');observe(projector[1],'projector_act');observe(projector[2],'projected')
 pixels=torch.tensor(cpu['pixels']).view(2,3,8,8).requires_grad_()
 hidden=tower(pixels,output_hidden_states=True).hidden_states[-1]
 features=hidden[:,1:,:].reshape(8,16);features.retain_grad();retained['selected']=features;stages['selected']=features.detach().flatten().tolist()
 projected=projector(features)
 words=lambda values:[struct.pack('<f',x) for x in values]
 for key,actual in [('vision_features',features),('projected_features',projected)]:
  if words(actual.detach().flatten().tolist())!=words(cpu['steps'][0]['forward'][key]):raise ValueError('CPU image replay differs '+key)
 projected.backward(torch.tensor(cpu['steps'][0]['projected_gradient']).view_as(projected))
 if words(pixels.grad.flatten().tolist())!=words(cpu['steps'][0]['pixel_gradient']):raise ValueError('CPU image VJP does not reproduce frozen pixels')
 gradients={name:t.grad.flatten().tolist() for name,t in retained.items() if t.grad is not None}
 params={'vision_tower.'+k:p.grad.flatten().tolist() for k,p in tower.named_parameters() if p.grad is not None}
 for index,name in [(0,'linear_1'),(2,'linear_2')]:params.update({'multi_modal_projector.'+name+'.'+k:p.grad.flatten().tolist() for k,p in projector[index].named_parameters()})
 a.output.mkdir(parents=True,exist_ok=False);write_json(a.output/'cpu.json',{'stages':stages,'gradients':gradients,'parameter_gradients':params,'pixel_gradient':pixels.grad.flatten().tolist(),
  'runner_sha256':digest(__file__),'fixtures_sha256':digest(a.fixtures/'manifest.json'),'frozen_forward_word_differences':0,'frozen_pixel_gradient_word_differences':0})
 print('CPU image forward and VJP reproduce frozen words exactly',flush=True)

if __name__=='__main__':main()
