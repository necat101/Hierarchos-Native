"""Independent CPU regressions for wide SGEMM panels and tied index selection."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

FIXTURE = Path(__file__).parent / "fixtures/deepseek_v41_wide_kernels.json"


def values(seed, count):
    # Declarative test inputs with a portable integer-only generator, shared
    # with Rust only to avoid embedding large duplicate input/weight arrays.
    result = []
    state = seed
    for _ in range(count):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        result.append(0x3d000000 | (state & 0x01ffffff) | ((state & 0x02000000) << 6))
    return torch.tensor(result, dtype=torch.uint32).view(torch.float32)


def digest(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def make_fixture():
    torch.set_num_threads(1)
    linears = []
    widths = [191,192,193,224,256,288,320,352,383,384,512,1024]
    for rows in [2,6]:
        for width in widths:
            seed = 4141080 + rows*1024 + width
            x, w = values(seed, rows*width), values(seed+1, 24*width)
            y = torch.nn.functional.linear(x.reshape(rows,width), w.reshape(24,width))
            by = torch.nn.functional.linear(x.reshape(rows,width).bfloat16(),w.reshape(24,width).bfloat16()).float()
            linears.append(dict(rows=rows, inner=width, outputs=24, seed=seed,
                input_sha256=digest(x), weight_sha256=digest(w), expected_bits=y.view(torch.uint32).flatten().tolist(),
                expected_bf16_bits=by.view(torch.uint32).flatten().tolist()))
    topks = []
    rng = torch.Generator().manual_seed(4141081)
    for width in [31,32,33,40,41,65,69,127,128,129,511,512,513,1024]:
        for k in sorted({1,2,min(8,width),min(32,width),min(512,width)}):
            rows, seq, ratio, start, offset = 4, 2, 2, width, 16
            scores = torch.randint(-3,4,(rows,width),generator=rng).float()
            visible = (start+seq)//ratio
            scores[:, visible:] = -torch.inf
            scores[2, ::3] = -torch.inf  # candidate-mask ties inside visibility
            scores[3] = 0.0  # finite all-equal partition/heap behavior
            chosen = scores.topk(k,dim=-1,sorted=False).indices.sort(-1).values
            expected = torch.where(chosen < visible, chosen+offset, -1)
            topks.append(dict(rows=rows,seq=seq,ratio=ratio,start_pos=start,offset=offset,
                width=width,topk=k,score_bits=scores.view(torch.uint32).flatten().tolist(),
                expected_indices=expected.flatten().tolist()))
    return dict(torch_version=torch.__version__, torch_revision=torch.version.git_version,
                policy="single_threaded_windows_cpu_pytorch", linears=linears, topks=topks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-fixture", action="store_true")
    args = p.parse_args()
    f = make_fixture()
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(f,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f:
        raise AssertionError("wide kernel fixture differs from pinned CPU PyTorch")
    print(f"PASS {len(f['linears'])} SGEMM panel cases, {len(f['topks'])} tied top-k cases")


if __name__ == "__main__":
    main()
