"""Offline reduction replay on captured Gemma3 operands; validation only."""
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
probe = Path(sys.argv[1])
trace = json.loads((probe / "trace.json").read_text())
x = torch.tensor(trace["final_prefix"]).reshape(-1, 16)
sq = x.square()
expected = sq.mean(-1) + 1e-6
native = torch.tensor(trace["final_norm_mean"])
print("torch", expected.tolist(), "native", native.tolist())
for width in [1, 4, 8, 16]:
    lanes = torch.zeros(x.shape[0], width)
    for i in range(x.shape[1]):
        lanes[:, i % width] += sq[:, i]
    total = torch.zeros(x.shape[0])
    for i in range(width):
        total += lanes[:, i]
    print("width", width, (total / 16 + 1e-6).tolist())
    while lanes.shape[-1] > 1:
        half = lanes.shape[-1] // 2
        lanes = lanes[:, :half] + lanes[:, half:]
    print("tree", width, (lanes[:, 0] / 16 + 1e-6).tolist())
