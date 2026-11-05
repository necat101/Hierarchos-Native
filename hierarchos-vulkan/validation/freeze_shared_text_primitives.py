"""Create new independent AVX2 primitive fixtures; never overwrite a corpus."""
from pathlib import Path
import argparse,json,struct,hashlib
import torch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(0x53445041)
 if torch.backends.cpu.get_cpu_capability()!='AVX2':raise ValueError('AVX2 CPU oracle required')
 bits=lambda values:[struct.unpack('<I',struct.pack('<f',x))[0] for x in values]
 def record(t):return t.detach().transpose(1,2).contiguous().flatten().tolist()
 cases=[]
 for D,H,KH,window in [(16,4,4,0),(16,4,2,0),(8,2,2,4),(8,2,1,0)]:
  for padded in [False,True]:
   q,k,v=[(torch.randn(2,8,h,D)*.3).transpose(1,2).requires_grad_() for h in [H,KH,KH]]
   query_mask=torch.ones(2,8,dtype=torch.bool);key_mask=query_mask.clone()
   if padded:query_mask[0,-1]=False;key_mask[0,-1]=False;query_mask[1,0]=False;key_mask[1,0]=False
   allowed=torch.ones(8,8,dtype=torch.bool).tril()
   if window:allowed &= torch.arange(8)[None,:]+window>torch.arange(8)[:,None]
   allowed=allowed[None,None,:,:]&query_mask[:,None,:,None]&key_mask[:,None,None,:]
   scale=D**-.5 if D==16 else 1.
   out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=allowed,scale=scale,enable_gqa=H!=KH)
   dy=torch.randn_like(out)*.05;dy*=query_mask[:,None,:,None]
   gradients=torch.autograd.grad(out,[q,k,v],dy)
   with torch.no_grad():
    flash,lse=torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(q,k,v,attn_mask=torch.where(allowed,0.,-torch.inf),scale=scale)
   if bits(record(out))!=bits(record(flash)):raise ValueError('public SDPA differs from active flash kernel')
   cases.append({'name':f'd{D}-h{H}-kv{KH}-w{window}-'+('padded' if padded else 'unmasked'),
    'batch':2,'seq':8,'heads':H,'kv_heads':KH,'dim':D,'window':window,'scale':scale,
    'q':record(q),'k':record(k),'v':record(v),'dy':record(dy),'query_mask':query_mask.float().flatten().tolist(),
    'key_mask':key_mask.float().flatten().tolist(),'expected_forward_bits':bits(record(out)),
    'expected_dq_bits':bits(record(gradients[0])),'expected_dk_bits':bits(record(gradients[1])),
    'expected_dv_bits':bits(record(gradients[2])),'expected_lse_bits':bits(lse.flatten().tolist())})
 data={'schema':1,'torch':torch.__version__,'torch_revision':torch.version.git_version,'cpu':torch.backends.cpu.get_cpu_capability(),
  'threads':1,'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf-8') as file:json.dump(data,file,indent=1,allow_nan=False);file.write('\n')
 print('frozen',len(cases),'independent SDPA forward/backward cases',flush=True)

if __name__=='__main__':main()
