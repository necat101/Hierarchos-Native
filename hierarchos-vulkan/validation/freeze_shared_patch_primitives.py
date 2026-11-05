"""Freeze independent oneDNN patch projections without replacing references."""
from pathlib import Path
import argparse,hashlib,json,struct,torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0x50415443)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise ValueError('AVX2 oracle required')
    values=lambda t:t.detach().contiguous().flatten().tolist()
    cases=[]
    for amplitude in [.3,1.3]:
        pixels=torch.randn(2,3,8,8)*amplitude
        weight=torch.randn(16,3,4,4)*.02
        patches=torch.nn.functional.unfold(pixels,4,stride=4).transpose(1,2).contiguous()
        output=torch.nn.functional.conv2d(pixels,weight,stride=4).flatten(2).transpose(1,2).contiguous()
        cases.append({'rows':8,'input_dim':48,'output_dim':16,'channels':3,'input':values(patches),'weight':values(weight),
                      'expected_bits':[struct.unpack('<I',struct.pack('<f',x))[0] for x in values(output)]})
    data={'schema':1,'torch':torch.__version__,'torch_revision':torch.version.git_version,'cpu':torch.backends.cpu.get_cpu_capability(),
          'threads':1,'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases}
    with args.output.open('x',encoding='utf-8') as file:json.dump(data,file,indent=1);file.write('\n')
    print('Frozen independent CLIP patch projections',flush=True)

if __name__=='__main__':main()
