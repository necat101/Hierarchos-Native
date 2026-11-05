"""Freeze the released HC-repeat adjoint through Torch autograd; never overwrite."""
from pathlib import Path
import argparse,hashlib,json,struct
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0x4843564A)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise ValueError('AVX2 oracle required')
    reference=Path(__file__).parent/'reference/deepseek_v41_vl/1bae0e1088993b9b82de08f59de5be5a34e42170/model.py'
    payload=reference.read_bytes()
    if hashlib.sha256(payload).hexdigest()!='e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d':raise ValueError('changed released model source')
    if b'h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)' not in payload:raise ValueError('released repeat contract changed')
    cases=[]
    for batch,seq,hc,dim in [(1,8,2,16),(2,8,2,32),(1,8,4,16),(2,8,4,32),(1,4,4,64),(2,4,8,16)]:
        merged=(torch.randn(batch,seq,dim)*.3).requires_grad_()
        stream=merged.unsqueeze(2).repeat(1,1,hc,1)
        gradient=torch.randn_like(stream)*.03
        dx,=torch.autograd.grad(stream,merged,gradient)
        cases.append({'rows':batch*seq,'hc':hc,'dim':dim,'gradient':gradient.flatten().tolist(),
                      'expected_bits':[struct.unpack('<I',struct.pack('<f',x))[0] for x in dx.flatten().tolist()]})
    document={'schema':1,'torch':torch.__version__,'torch_revision':torch.version.git_version,
              'cpu':torch.backends.cpu.get_cpu_capability(),'threads':1,
              'reference_revision':'1bae0e1088993b9b82de08f59de5be5a34e42170',
              'reference_sha256':hashlib.sha256(payload).hexdigest(),
              'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'contract':'autograd of released h.unsqueeze(2).repeat(1,1,hc_mult,1)','cases':cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:json.dump(document,file,indent=1);file.write('\n')
    print('Frozen',len(cases),'released HC-repeat adjoints',flush=True)

if __name__=='__main__':main()
