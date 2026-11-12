"""Independent CPU einsum oracle for packed LoRA layout/mapping regressions."""
from pathlib import Path
import json
import torch

torch.set_num_threads(1)
cases = []
for index, (name, layout, width) in enumerate([
    ("input_output", "expert_in_out", 4), ("output_input", "expert_out_in", 4),
    ("square_input_output", "expert_in_out", 2), ("square_output_input", "expert_out_in", 2),
]):
    experts, rank, inputs = 3, 2, 2
    a = (torch.arange(experts*rank*inputs, dtype=torch.float32).reshape(experts,rank,inputs)-3+index)/4
    b = (torch.arange(width*rank*experts, dtype=torch.float32).reshape(width,rank,experts).remainder(11)-4-index)/8
    delta = torch.einsum("ore,eri->eio" if layout == "expert_in_out" else "ore,eri->eoi", b, a)*1.5
    base = (torch.arange(delta.numel(), dtype=torch.float32).reshape_as(delta)-7)/8
    cases.append(dict(name=name, layout=layout, experts=experts, rank=rank, input=inputs,
        output=width, scale=1.5, shape=list(base.shape), a=a.flatten().tolist(), b=b.flatten().tolist(),
        base=base.flatten().tolist(), merged=(base+delta).flatten().tolist()))
output = Path(__file__).with_name("fixtures") / "packed_lora_layouts.json"
output.write_bytes((json.dumps(dict(oracle="independent torch.einsum on CPU", torch=torch.__version__, cases=cases), indent=2)+"\n").encode())
