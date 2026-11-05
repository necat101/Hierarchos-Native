"""Freeze independent CPU adjoints and RoPE; never replace existing evidence."""
from pathlib import Path
import argparse, hashlib, json, math, struct
import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0x53484152)
    if torch.backends.cpu.get_cpu_capability() != 'AVX2':
        raise ValueError('AVX2 CPU oracle required')
    values = lambda t: t.detach().contiguous().flatten().tolist()
    bits = lambda t: [struct.unpack('<I', struct.pack('<f', x))[0] for x in values(t)]
    record = lambda t: t.detach().transpose(1, 2).contiguous()
    attention = []
    for amplitude in [.3, 1.3]:
        q, k, v = [(torch.randn(2, 5, 2, 8) * amplitude).transpose(1, 2).requires_grad_() for _ in range(3)]
        probability = ((q @ k.transpose(-1, -2)) * (8 ** -.5)).softmax(-1, dtype=torch.float32)
        output = probability @ v
        dy = torch.randn_like(output) * .05
        gradients = torch.autograd.grad(output, [q, k, v], dy)
        attention.append({**{name: values(record(t)) for name, t in [('q', q), ('k', k), ('v', v), ('dy', dy)]},
                          'scale': 8 ** -.5, 'expected_forward_bits': bits(record(output)),
                          **{'expected_' + name + '_bits': bits(record(t)) for name, t in zip(['dq', 'dk', 'dv'], gradients)}})
    bias = []
    for rows, dim in [(8,16),(10,16),(10,32),(16,16),(16,32),(17,19),(17,48),(36,16),(16,64)]:
        dy = torch.randn(rows, dim) * .01
        bias.append({'rows': rows, 'dim': dim, 'dy': values(dy), 'expected_bits': bits(dy.sum(0))})
    unary = []
    for name in ['new_gelu', 'silu']:
        x = torch.cat([torch.linspace(-8, 8, 4096), torch.randn(4096) * .3]).requires_grad_()
        dy = torch.randn_like(x) * .01
        y = .5 * x * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + .044715 * torch.pow(x, 3.)))) if name == 'new_gelu' else torch.nn.functional.silu(x)
        dx, = torch.autograd.grad(y, x, dy)
        unary.append({'name': name, 'x': values(x), 'dy': values(dy), 'expected_backward_bits': bits(dx)})
    residual_norm = []
    for eps in [1e-5, 1e-6]:
        x = (torch.randn(16,64) * .02).requires_grad_()
        weight = (1. + torch.randn(64) * .1).requires_grad_()
        dy = torch.randn_like(x) * .01
        residual = torch.randn_like(x) * .02
        mean = x.pow(2).mean(-1,keepdim=True) + eps
        rstd = torch.rsqrt(mean)
        out = (x * rstd) * weight
        dx, = torch.autograd.grad((out, x), (x,), (dy, residual))
        residual_norm.append({**{name: values(t) for name, t in [('x',x),('weight',weight),('dy',dy),('residual',residual),('mean',mean),('rstd',rstd)]},
                              'rows':16,'dim':64,'eps':eps,'expected_dx_bits':bits(dx)})
    weight_gradient = []
    for output in [32, 64, 192, 256, 320]:
        x = (torch.randn(16,64) * .3).requires_grad_()
        weight = (torch.randn(output,64) * .02).requires_grad_()
        dy = torch.randn(16,output) * .01
        out = torch.nn.functional.linear(x,weight)
        dw, = torch.autograd.grad(out,(weight,),dy)
        weight_gradient.append({'rows':16,'input_dim':64,'output_dim':output,'x':values(x),'dy':values(dy),'expected_bits':bits(dw)})
    rope = []
    frequency = 1. / (10000. ** (torch.arange(0,16,2).float() / 16))
    phase = torch.arange(8).float()[:,None] * frequency[None,:]
    phase = torch.cat((phase,phase),dim=-1)[None,:,None,:]
    cos, sin = phase.cos(), phase.sin()
    for heads in [2,4]:
        x = (torch.randn(2,8,heads,16) * .3).requires_grad_()
        dy = torch.randn_like(x) * .01
        rotated = torch.cat((-x[...,8:],x[...,:8]),dim=-1)
        y = (x*cos) + (rotated*sin)
        dx, = torch.autograd.grad(y,(x,),dy)
        rope.append({'heads':heads,'x':values(x),'dy':values(dy),'frequency':values(frequency),
                     'expected_forward_bits':bits(y),'expected_backward_bits':bits(dx)})
    cosine_input = torch.cat([torch.linspace(-8,8,8192), phase.flatten()])
    library = Path(torch.__file__).parent/'lib/torch_cpu.dll'
    document = {'schema':1,'torch':torch.__version__,'torch_revision':torch.version.git_version,
                'cpu':torch.backends.cpu.get_cpu_capability(),'threads':1,
                'torch_cpu_sha256':hashlib.file_digest(library.open('rb'),'sha256').hexdigest(),
                'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'attention':attention,'bias':bias,'unary':unary,'residual_norm':residual_norm,
                'weight_gradient':weight_gradient,'rope':rope,
                'cosine':{'x':values(cosine_input),'expected_bits':bits(cosine_input.cos())}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(document,file,indent=1,allow_nan=False)
        file.write('\n')
    print('Frozen independent shared-training primitives',flush=True)

if __name__ == '__main__':
    main()
