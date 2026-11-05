"""Freeze unchanged released HC mixes and their native Torch input adjoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'hierarchos-vulkan/validation'))
import torch
import torch.nn.functional as F
import freeze_deepseek_v41_native_pipeline as ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert torch.backends.cpu.get_cpu_capability() == 'AVX2'
    source = ref.vision_oracle.REFERENCE / 'model.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'
    base = json.loads(ref.training.BASE.read_text())
    ns = ref.training.model.load_reference(source, ref.training.KERNEL, tokens=base['tokens'])
    values = lambda t: t.detach().flatten().tolist()
    bits = lambda t: t.detach().flatten().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
    cases = []
    original = ns['hc_split_sinkhorn']
    captured = []
    def observe(mix, *args, **kwargs):
        captured.append(mix)
        return original(mix, *args, **kwargs)
    ns['hc_split_sinkhorn'] = observe
    for hc in (2, 4):
        for rows in (1, 2, 3, 8, 9):
            for seed in (1901, 7183):
                torch.manual_seed(seed + rows + hc)
                config = types.SimpleNamespace(norm_eps=1e-20, hc_mult=hc, hc_sinkhorn_iters=3, hc_eps=1e-6)
                x = (torch.randn(1, rows, hc, 32) * .3).requires_grad_()
                weight = torch.randn(hc * (hc + 2), hc * 32) * .04
                scale = torch.tensor([.6, -.4, .3])
                bias = torch.randn(hc * (hc + 2)) * .07
                captured.clear()
                outputs = ns['Block'].hc_mixes(config, x, weight, scale, bias)
                assert len(captured) == 1
                upstream = tuple(torch.randn_like(t) * .03 for t in outputs)
                dx, dmix = torch.autograd.grad(outputs, (x, captured[0]), upstream)
                with torch.no_grad():
                    raw = F.linear(x.flatten(2), weight)
                cases.append(dict(rows=rows, hc=hc, dim=32, seed=seed, iterations=3,
                                  norm_eps=1e-20, hc_eps=1e-6, input=values(x), weight=values(weight),
                                  scale=values(scale), bias=values(bias), raw=values(raw),
                                  grad_pre=values(upstream[0]), grad_post=values(upstream[1]), grad_comb=values(upstream[2]),
                                  mix_gradient=values(dmix), mix_bits=bits(dmix), input_bits=bits(dx)))
    result = dict(schema=1, cpu='AVX2', threads=1, torch=torch.__version__, torch_revision=torch.version.git_version,
                  producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  contract='Unchanged released Block.hc_mixes with CPU hc_split_sinkhorn fallback and unmodified Torch autograd', cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=1)
        out.write('\n')
    print('Frozen', len(cases), 'released HC adjoints', hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
