"""Independent random CPU flash-SDPA controls for the released V4.1 ViT."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def tensor(x):
    return x.detach().transpose(0, 1).contiguous().flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise ValueError("Pinned AVX2 CPU oracle required")
    cases = []
    for rows in (4, 6):
        for seed in (1901, 7183, 20983, 34129):
            torch.manual_seed(seed + rows)
            # Use the actual native ViT [patch,head,head_dim] strides, then
            # transpose as the unchanged released Attention.forward does.
            qs = [(torch.randn(rows, 2, 8) * .4).requires_grad_() for _ in range(3)]
            q, k, v = [x.transpose(0, 1) for x in qs]
            output = F.scaled_dot_product_attention(q, k, v)
            dy = torch.randn_like(output) * .1
            gradients = torch.autograd.grad(output, qs, dy)
            cases.append(dict(rows=rows, dim=16, heads=2, seed=seed + rows,
                              q=tensor(q), k=tensor(k), v=tensor(v), output=tensor(output),
                              dy=tensor(dy), dq=gradients[0].flatten().tolist(),
                              dk=gradients[1].flatten().tolist(), dv=gradients[2].flatten().tolist()))
    payload = dict(schema=1, torch=torch.__version__, torch_revision=torch.version.git_version,
                   cpu="AVX2", threads=1, tolerance=2e-7, rtol=0, comparison="strict_less_than",
                   producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   contract="Torch autograd of unmodified F.scaled_dot_product_attention on released native ViT layouts; independent fixed random seeds; no GPU-derived expected values",
                   cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(payload, out, indent=1)
        out.write("\n")
    print("Frozen", len(cases), "independent native vision SDPA controls", flush=True)


if __name__ == "__main__":
    main()
