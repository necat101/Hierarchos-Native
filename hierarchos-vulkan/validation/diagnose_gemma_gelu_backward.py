"""Replay GELU backward from native trace operands without qualifying a model."""
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
trace = json.loads(Path(sys.argv[1]).read_text())
for layer in (0, 1):
    prefix = f"layers.{layer}."
    x = torch.tensor(trace[prefix + "mlp_gate_output"], requires_grad=True)
    incoming = torch.tensor(trace[prefix + "grad_mlp_gate_activation"])
    native = torch.tensor(trace[prefix + "grad_mlp_gate_output"])
    torch.nn.functional.gelu(x, approximate="tanh").backward(incoming)
    x = x.detach()
    x2 = x * x
    x3 = x2 * x
    t = torch.tanh(0.7978845608028654 * (x + 0.044715 * x3))
    left = 0.5 * (1.0 + t)
    right = ((0.5 * x) * (1.0 - t * t)) * (
        0.7978845608028654 * (1.0 + (3.0 * 0.044715) * x2)
    )
    replay = incoming * (left + right)
    # Obtain the autograd reference again after detaching the replay inputs.
    ref_x = x.clone().requires_grad_()
    torch.nn.functional.gelu(ref_x, approximate="tanh").backward(incoming)
    print(json.dumps({"layer": layer,
        "native_vs_torch": float((native-ref_x.grad).abs().max()),
        "staged_vs_torch": float((replay-ref_x.grad).abs().max()),
        "staged_vs_native": float((replay-native).abs().max())}))
