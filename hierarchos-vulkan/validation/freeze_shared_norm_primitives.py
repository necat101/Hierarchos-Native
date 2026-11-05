"""Freeze new width-64 CPU normalization primitives; refuse corpus overwrite."""
from pathlib import Path
import argparse,hashlib,json,struct,torch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(0x4E4F524D)
 if torch.backends.cpu.get_cpu_capability()!='AVX2':raise ValueError('AVX2 oracle required')
 bits=lambda t:[struct.unpack('<I',struct.pack('<f',x))[0] for x in t.detach().flatten().tolist()]
 values=lambda t:t.detach().flatten().tolist()
 cases=[]
 for kind in ['layer_norm','rms_norm']:
  x=(torch.randn(16,64)*.02).requires_grad_();weight=(1.+torch.randn(64)*.1).requires_grad_()
  bias=(torch.randn(64)*.01).requires_grad_();dy=torch.randn_like(x)*.01
  eps=1e-5 if kind=='layer_norm' else 1e-6
  if kind=='layer_norm':
   out,mean,rstd=torch.native_layer_norm(x,(64,),weight,bias,eps)
   dx,dw,db=torch.autograd.grad(out,[x,weight,bias],dy)
  else:
   mean=x.pow(2).mean(-1,keepdim=True)+eps;rstd=torch.rsqrt(mean);out=(x*rstd)*weight
   dx,dw=torch.autograd.grad(out,[x,weight],dy);db=torch.zeros_like(bias)
  cases.append({'kind':kind,'rows':16,'dim':64,'eps':eps,'x':values(x),'weight':values(weight),'bias':values(bias),'dy':values(dy),
   'mean':values(mean),'rstd':values(rstd),'expected_forward_bits':bits(out),'expected_dx_bits':bits(dx),'expected_dw_bits':bits(dw),'expected_db_bits':bits(db)})
 data={'schema':1,'torch':torch.__version__,'torch_revision':torch.version.git_version,'cpu':torch.backends.cpu.get_cpu_capability(),'threads':1,
  'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases}
 with a.output.open('x',encoding='utf-8') as file:json.dump(data,file,indent=1);file.write('\n')
 print('frozen',len(cases),'normalization primitives',flush=True)

if __name__=='__main__':main()
