"""Diagnostic only: compare CPU GEMM reduction order at PEFT projections."""
import sys
import json
import os
import tempfile
from pathlib import Path
import torch
import numpy as np

family = sys.argv[1] if len(sys.argv) > 1 else "gemma4"
saved_module = os.environ.get("HIERARCHOS_PEFT_SAVED_MODULE", "model.embed_tokens")
sys.argv = [sys.argv[0], family]
import verify_peft_lora_strict as strict
import verify_peft_multi_adapter as multi
from peft import PeftModel

torch.set_num_threads(1)
probe_root = Path(tempfile.mkdtemp(prefix="linear-order-", dir=str(strict.FIXTURE_ROOT)))
adapter = probe_root / "adapter"
multi.SAVED = saved_module
multi._write_adapter(adapter, 101)
model = PeftModel.from_pretrained(
    multi.load_base(),
    adapter,
).eval()
captured = []
def capture(name):
    def hook(module, args, output):
        captured.append(
            (
                name,
                args[0].detach().reshape(-1, args[0].shape[-1]),
                module,
                output.detach().reshape(-1, output.shape[-1]),
            )
        )
    return hook
for name, module in model.named_modules():
    if any(name.endswith("layers.0.self_attn." + p) for p in ["q_proj", "k_proj", "v_proj"]):
        module.register_forward_hook(capture(name.rsplit(".", 1)[-1]))
    elif name.endswith("layers.0.mlp.down_proj"):
        module.register_forward_hook(capture("down_proj"))
fixture = json.loads(multi.FIXTURE_JSON.read_text())
ids = torch.tensor(fixture["input_ids"]).reshape(fixture["batch_size"], fixture["seq_len"])
with torch.no_grad():
    model(input_ids=ids, attention_mask=torch.tensor(fixture["attention_mask"]).reshape_as(ids))

for projection, x, module, hf_output in captured:
    print("projection", projection, flush=True)
    a_out = module.lora_A.default(x).detach()
    base_out = module.base_layer(x).detach().numpy()
    delta_out = module.lora_B.default(a_out).detach().numpy()
    scale = np.float32(module.scaling["default"])
    hf_total = hf_output.numpy()
    separate_total = np.float32(base_out + np.float32(delta_out * scale))
    fused_total = np.float32(
        base_out.astype(np.float64)
        + delta_out.astype(np.float64) * np.float64(scale)
    )
    print(
        "combine",
        "scale", float(scale),
        "separate", float(np.abs(separate_total - hf_total).max()), int(np.count_nonzero(separate_total != hf_total)),
        "fused", float(np.abs(fused_total - hf_total).max()), int(np.count_nonzero(fused_total != hf_total)),
        flush=True,
    )
    for label, xx_t, weight in [
        ("base", x, module.base_layer.weight),
        ("A", x, module.lora_A.default.weight),
        ("B", a_out, module.lora_B.default.weight),
    ]:
        w = weight.detach()
        target = (xx_t @ w.T).numpy()
        xx, ww = xx_t.numpy().astype(np.float64), w.numpy().astype(np.float64)
        results = []
        for lanes in [1, 2, 4, 8, 16]:
            for block in [1, 2, 4, 8, 16]:
                for reverse, fused in [(False, True), (True, True), (False, False), (True, False)]:
                    acc = np.zeros((*target.shape, lanes), dtype=np.float32)
                    order = list(range(xx_t.shape[1]))
                    if reverse: order.reverse()
                    for k in order:
                        lane = (k // block) % lanes
                        product = xx[:, k, None] * ww[None, :, k]
                        if not fused: product = product.astype(np.float32).astype(np.float64)
                        acc[:, :, lane] = (product + acc[:, :, lane].astype(np.float64)).astype(np.float32)
                    for reduction in ["sequential", "balanced", "halves"]:
                        values = [acc[:, :, i].copy() for i in range(lanes)]
                        if reduction == "sequential":
                            out = np.zeros_like(target)
                            for value in values: out = out + value
                        else:
                            while len(values) > 1:
                                if reduction == "balanced": values = [values[i] + values[i+1] for i in range(0, len(values), 2)]
                                else:
                                    half = len(values)//2
                                    values = [values[i] + values[i+half] for i in range(half)]
                            out = values[0]
                        results.append((float(np.abs(out-target).max()), int(np.count_nonzero(out != target)), lanes, block, reverse, fused, reduction))
        print(label, list(w.shape), sorted(results)[:8], flush=True)
