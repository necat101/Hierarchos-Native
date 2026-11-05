"""Independent released native VL + frozen-base LoRA CPU training draft."""
from pathlib import Path
import argparse,hashlib,json,sys,types
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'hierarchos-vulkan/validation'))
import torch
import torch.nn.functional as F
from torch import nn
import freeze_deepseek_v41_vision as vision_oracle
import validate_deepseek_v41_full_training as training

def args_for(ns):
    args=ns['ModelArgs'](max_batch_size=2,max_seq_len=16,temperature=0,dtype='bf16',expert_dtype=None,
        vocab_size=16,dim=32,moe_inter_dim=12,n_layers=6,n_mtp_layers=0,n_heads=2,
        n_routed_experts=3,n_activated_experts=2,route_scale=1.5,swiglu_limit=1.5,
        q_lora_rank=8,head_dim=32,rope_head_dim=16,o_groups=1,o_lora_rank=16,window_size=4,
        compress_ratios=(0,2,2,1,1,1),kv_source_layers=(1,3),index_source_layers=(1,3,5),
        compress_rope_theta=160000,original_seq_len=0,index_n_heads=2,index_head_dim=32,index_topk=2,
        candidate_source_layer=3,candidate_topk_blocks=2,candidate_block_size=2,hc_mult=2,
        engram_layer_ids=(0,4),engram_num_embeddings=(128,128),engram_max_ngram_size=3,
        engram_vocab_size=5,engram_n_heads=2,engram_head_dim=4,engram_compressed_vocab_size=8)
    for key,value in vision_oracle.arguments().items():setattr(args,key,value)
    args.image_token_id=15
    return args

def build(base,vision,processor,params,specs,image_params):
    ns=training.model.load_reference(vision_oracle.REFERENCE/'model.py',training.KERNEL,tokens=base['tokens'])
    training.install_functional(ns)
    ns.update(ViT=vision.ViT,Aligner=vision.Aligner,
              **{key:getattr(processor,key) for key in ['IMAGE','IMAGE_START','IMAGE_END','IMAGE_NEW_LINE']})
    args=args_for(ns)
    net=ns['Transformer'](args)
    state={n:torch.tensor(t['values']).reshape(t['shape']) for n,t in base['tensors'].items()}
    with torch.no_grad():
        for i,layer in enumerate(net.layers):
            # Distinct VL correction bias is essential; the released gate selects
            # routes with it and derives route weights from unbiased scores.
            layer.ffn.gate.bias_vl.copy_(torch.tensor([0.15,-0.15,0.0]))
    result=net.load_state_dict(state,strict=False)
    allowed=set(image_params)|{f'layers.{i}.ffn.gate.bias_vl' for i in range(6)}
    assert set(result.missing_keys)==allowed and not result.unexpected_keys
    net.requires_grad_(False)
    for name,p in image_params.items():
        path,leaf=name.rsplit('.',1) if '.' in name else ('',name)
        net.get_submodule(path)._parameters[leaf]=p
    for target,(a,b) in params.items():
        path,leaf=target.rsplit('.',1);owner=net.get_submodule(path)
        setattr(owner,leaf,training.Lora(getattr(owner,leaf),a,b,specs[target]['scale']))
    net.encode_image=types.MethodType(ns['Transformer'].encode_image.__wrapped__,net)
    return ns,net,args

def case(base,vision,processor,component):
    specs,params=training.make_parameters(base)
    torch.manual_seed(0x4D41524B)
    image_params={n:nn.Parameter(torch.tensor(t['values']).reshape(t['shape'])) for n,t in component['parameters'].items()}
    image_params.update({n:nn.Parameter(torch.randn(32)*.08) for n in ['image_start','image_end','image_newline']})
    initial={n:vision_oracle.tensor(p) for n,p in image_params.items()}
    ns,net,args=build(base,vision,processor,params,specs,image_params)
    patches,image=vision_oracle.image_input(processor,args,component['image']['width'],component['image']['height'])
    patches.requires_grad_()
    types_=processor.image_token_types(image['n_llm_h'],image['n_llm_w'])
    ids=torch.tensor([0,1]+[args.image_token_id]*types_.numel()+[3,4]).unsqueeze(0)
    token_types=torch.cat([torch.tensor([-1,-1]),types_,torch.tensor([-1,-1])]).unsqueeze(0)
    images=[[processor.ImageInput(2,patches,image['n_vit_h'],image['n_vit_w'],types_)]]
    targets=(ids+1)%16;targets[0,:2]=-100
    hyper=dict(learning_rate=.001,beta1=.9,beta2=.999,eps=.001,weight_decay=0.)
    optimizer=torch.optim.AdamW(list(image_params.values())+[p for pair in params.values() for p in pair],
        lr=.001,betas=(.9,.999),eps=.001,weight_decay=0.,foreach=False,fused=False)
    steps=[]
    for step in range(3):
        ns,net,args=build(base,vision,processor,params,specs,image_params)
        head=[];streams=[];features=[];routes={}
        net.head.register_forward_pre_hook(lambda _,ins:head.append(ins[0]))
        def stream_hook(_,ins):
            ins[0].retain_grad();streams.append(ins[0])
        net.layers[0].engram.register_forward_pre_hook(stream_hook)
        def feature_hook(_,ins,out):out.retain_grad();features.append(out)
        net.aligner.register_forward_hook(feature_hook)
        for i,layer in enumerate(net.layers):
            layer.ffn.gate.register_forward_hook(lambda _,ins,out,i=i:routes.update({str(i):dict(weights=vision_oracle.tensor(out[0]),indices=out[1].flatten().tolist())}))
        ns['Transformer'].forward.__wrapped__(net,ids,0,images=images,token_types=token_types)
        logits=F.linear(head[-1],net.head.weight)
        record=dict(logits=vision_oracle.tensor(logits),features=vision_oracle.tensor(features[-1]),
                    injected_stream=vision_oracle.tensor(streams[0]),routes=routes)
        if step<2:
            loss=F.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=-100)
            optimizer.zero_grad(set_to_none=True);patches.grad=None;loss.backward()
            record.update(loss=loss.item(),stream_gradient=vision_oracle.tensor(streams[0].grad),
                feature_gradient=vision_oracle.tensor(features[-1].grad),patch_gradient=vision_oracle.tensor(patches.grad),
                image_gradients={n:vision_oracle.tensor(p.grad) for n,p in image_params.items()},
                adapter_gradients={n:None if a.grad is None else dict(a=vision_oracle.tensor(a.grad),b=vision_oracle.tensor(b.grad)) for n,(a,b) in params.items()})
            assert all(p.grad is not None and torch.count_nonzero(p.grad) for p in image_params.values())
            optimizer.step()
            record['updated_image_parameters']={n:vision_oracle.tensor(p) for n,p in image_params.items()}
            record['updated_adapters']={n:dict(a=vision_oracle.tensor(a),b=vision_oracle.tensor(b)) for n,(a,b) in params.items()}
        steps.append(record)
        print(component['name'],'joint CPU',step,'complete',flush=True)
    assert all(initial[n]['values']!=vision_oracle.tensor(p)['values'] for n,p in image_params.items())
    config=json.loads(json.dumps(base['config']))
    config['image_token_id']=15
    config['vision_config'].update(num_hidden_layers=2,hidden_size=16,num_attention_heads=2,intermediate_size=32,patch_size=2,rope_theta=10000,downsample_ratio=2,max_image_tokens=32,min_pixels=0,max_wh_ratio=None)
    return dict(name=component['name'],base_fixture_sha256=hashlib.sha256(training.BASE.read_bytes()).hexdigest(),
        config=config,vision_config=component['config'],image=image,
        image_parameters=initial,gate_bias_vl=[.15,-.15,0.],patches=vision_oracle.tensor(patches),
        token_ids=ids.flatten().tolist(),token_types=token_types.flatten().tolist(),targets=targets.flatten().tolist(),
        image_start=2,batch_size=1,seq_len=ids.shape[1],adapters=specs,seed=1793,hyper=hyper,steps=steps)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise ValueError('AVX2 CPU required')
    vision,processor=vision_oracle.load_references()
    base=json.loads(training.BASE.read_text())
    source=ROOT/'hierarchos-vulkan/validation/fixtures/deepseek_v41_native_vision_avx2.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='f6bb7f095bb829f1124fd528ed79d36d12e8fc2fe041a89734c9a2ab4cdfc439'
    components=json.loads(source.read_text())['cases']
    result=dict(schema=1,tolerance=2e-7,rtol=0,comparison='strict_less_than',revision=vision_oracle.REVISION,
        source_sha256={**vision_oracle.HASHES,'engram.py':hashlib.sha256((vision_oracle.REFERENCE/'engram.py').read_bytes()).hexdigest()},kernel_source_sha256=hashlib.sha256(training.KERNEL.read_bytes()).hexdigest(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu='AVX2',threads=1,component_fixture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__,torch_revision=torch.version.git_version,
        contract='Unchanged released native vision, aligner, merge, HC repeat and VL routing. FP32 training with frozen language base and trainable LoRA, native vision and image delimiter embeddings.',
        cases=[case(base,vision,processor,c) for c in components])
    dest=args.output
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as out:json.dump(result,out,indent=1);out.write('\n')
    print('Frozen independent joint native VL CPU reference',hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':main()
