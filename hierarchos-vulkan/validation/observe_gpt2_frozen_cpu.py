"""Observe standalone GPT2 CPU forward stages without writing any expectations."""
import argparse,json,os,struct,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(os.environ['HIERARCHOS_TRANSFORMERS_CHECKOUT'])/'src'))
    sys.modules['torchaudio']=None
    import torch
    from transformers import GPT2LMHeadModel
    torch.set_num_threads(1)
    model=GPT2LMHeadModel.from_pretrained(a.case/'model').float().eval();model.set_attn_implementation('sdpa')
    f=json.loads((a.case/'input.json').read_text());cpu=json.loads((a.case/'cpu.json').read_text())
    shape=(f['batch_size'],f['seq_len']);ids=torch.tensor(f['input_ids']).view(shape)
    mask=torch.tensor(f['attention_mask'],dtype=torch.long).view(shape)
    stages={};handles=[]
    def observe(module,name,before=False):
        def hook(module,inputs,output=None):
            t=inputs[0] if before else output
            if isinstance(t,tuple):t=t[0]
            stages[name]=t.detach().flatten().tolist()
        handles.append((module.register_forward_pre_hook if before else module.register_forward_hook)(hook))
    for i,layer in enumerate(model.transformer.h):
        for module,name in [(layer.ln_1,'ln1'),(layer.attn.c_attn,'qkv'),(layer.attn.c_proj,'o'),
            (layer.ln_2,'ln2'),(layer.mlp.c_fc,'u'),(layer.mlp.act,'activation'),(layer.mlp.c_proj,'d'),(layer,'out')]:
            observe(module,f'layer_{i}/'+name)
        observe(layer,f'layer_{i}/input',True);observe(layer.attn.c_proj,f'layer_{i}/attention',True)
    observe(model.transformer.ln_f,'final_norm');observe(model.transformer.ln_f,'final_prefix',True)
    with torch.no_grad():actual=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.flatten().tolist()
    bits=lambda v:struct.unpack('<I',struct.pack('<f',v))[0]
    different=sum(bits(a)!=b for a,b in zip(actual,cpu['bits'],strict=True))
    expected=[struct.unpack('<f',struct.pack('<I',v))[0] for v in cpu['bits']]
    maximum=max(abs(a-b) for a,b in zip(actual,expected,strict=True))
    print('CPU frozen replay',different,'words; max',maximum,flush=True)
    with a.output.open('x') as file:json.dump(dict(torch=torch.__version__,cpu=torch.backends.cpu.get_cpu_capability(),
        word_differences=different,max_abs=maximum,stages=stages,logits=actual),file,separators=(',',':'))

if __name__=='__main__':main()
