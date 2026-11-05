"""Freeze Linear GEMV and column-major unfold adjoints independently in Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise ValueError("AVX2 CPU required")
    cases = []
    for rows in (1, 2):
        for width in (32, 64):
            for seed in (1901, 7183, 20983, 34129):
                torch.manual_seed(seed + width + rows)
                x = (torch.randn(rows, width) * .3).T.contiguous().T.requires_grad_()
                weight = (torch.randn(32, width) * .1).requires_grad_()
                bias = (torch.randn(32) * .02).requires_grad_()
                output = F.linear(x, weight, bias)
                dy = torch.randn_like(output) * .1
                dx, dw, db = torch.autograd.grad(output, (x, weight, bias), dy)
                cases.append(dict(rows=rows, input_dim=width, output_dim=32, seed=seed + width + rows,
                                  input_strides=list(x.stride()),
                                  **{name: value.detach().flatten().tolist() for name, value in
                                     [("x", x), ("weight", weight), ("bias", bias), ("output", output),
                                      ("dy", dy), ("dx", dx), ("dw", dw), ("db", db)]}))
    payload = dict(schema=1, torch=torch.__version__, torch_revision=torch.version.git_version,
                   cpu="AVX2", threads=1, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   contract="Independent fixed random F.linear and Torch autograd; column-major input matching released aligner F.unfold(...).transpose(0,1)", cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(payload, out, indent=1)
        out.write("\n")
    print("Frozen", len(cases), "native aligner Linear controls", flush=True)


if __name__ == "__main__":
    main()
