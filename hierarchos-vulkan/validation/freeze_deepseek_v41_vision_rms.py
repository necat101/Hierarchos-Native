"""Freeze native RMSNorm with the aligner's column-major upstream gradient."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import freeze_deepseek_v41_vision as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise ValueError("AVX2 CPU required")
    vision, _ = reference.load_references()
    cases = []
    for rows in (4, 6):
        for seed in (1901, 7183, 20983, 34129):
            torch.manual_seed(seed + rows)
            module = vision.RMSNorm(16)
            with torch.no_grad():
                module.weight.copy_(1. + torch.randn(16) * .02)
            x = (torch.randn(rows, 16) * .3).requires_grad_()
            y = module(x)
            dy = (torch.randn(rows, 16) * .1).T.contiguous().T
            dx, dw = torch.autograd.grad(y, (x, module.weight), dy)
            rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + module.eps)
            cases.append(dict(rows=rows, dim=16, seed=seed + rows, upstream_strides=list(dy.stride()),
                              **{name: value.detach().flatten().tolist() for name, value in
                                 [("x", x), ("weight", module.weight), ("dy", dy), ("rstd", rstd), ("dx", dx), ("dw", dw)]}))
    payload = dict(schema=1, torch=torch.__version__, torch_revision=torch.version.git_version,
                   cpu="AVX2", threads=1, source_sha256=reference.HASHES,
                   producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   contract="Unchanged released RMSNorm; Torch autograd with the column-major upstream gradient emitted by released spatial aligner", cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(payload, out, indent=1)
        out.write("\n")
    print("Frozen", len(cases), "native RMSNorm layout controls", flush=True)


if __name__ == "__main__":
    main()
