"""Freeze native small sparse attention via the unmodified pinned CPU fallback."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import torch
import validate_deepseek_v41_model as ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert torch.backends.cpu.get_cpu_capability() == 'AVX2'
    cases = []
    values = lambda t: t.detach().flatten().tolist()
    for rows in (8, 9):
        for compressed in (0, 2):
            for seed in (1901, 7183):
                torch.manual_seed(seed + rows + compressed)
                capacity = 4 if compressed else 0
                q = (torch.randn(1, rows, 2, 32) * .3).requires_grad_()
                kv = (torch.randn(1, rows + capacity, 32) * .3).requires_grad_()
                sink = torch.randn(2) * .1
                indices = torch.tensor([[list(range(i - 3, i + 1)) + ([rows + i % 4, rows + (i + 1) % 4] if compressed else []) for i in range(rows)]])
                observed = {}
                def profile(frame, event, result):
                    if frame.f_code is ref.sparse_attn.__code__ and event == 'return':
                        for name in ('gathered', 'scores', 'weights'):
                            t = frame.f_locals[name]
                            t.retain_grad()
                            observed[name] = t
                # Observational access to the original function's tensors. No
                # reference operation, reduction or backward formula is changed.
                sys.setprofile(profile)
                try:
                    y = ref.sparse_attn(q, kv, sink, indices, 32 ** -.5)
                finally:
                    sys.setprofile(None)
                dy = torch.randn_like(y) * .03
                y.backward(dy)
                total = 4 + compressed
                cases.append(dict(rows=rows, seed=seed, compressed_topk=compressed, compressed_capacity=capacity,
                                  query=values(q), window=values(kv[:, :rows]), compressed=values(kv[:, rows:]),
                                  sink=values(sink), indices=indices.flatten().tolist(), upstream=values(dy),
                                  weights=values(observed['weights']), weight_gradient=values(observed['weights'].grad),
                                  dot_gradient=values(observed['scores'].grad[..., :total] * (32 ** -.5)),
                                  gather_gradient=values(observed['gathered'].grad), query_gradient=values(q.grad),
                                  window_gradient=values(kv.grad[:, :rows]), compressed_gradient=values(kv.grad[:, rows:]), output=values(y)))
    result = dict(schema=1, cpu='AVX2', threads=1, torch=torch.__version__, torch_revision=torch.version.git_version,
                  producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
                  contract='Unchanged CPU sparse_attn gather/einsum/softmax fallback from the original model oracle; random independent upstream; probability gradients at masked slots need not be evaluated by native kernels', cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=1)
        out.write('\n')
    print('Frozen', len(cases), 'unmodified sparse CPU autograd cases', hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
