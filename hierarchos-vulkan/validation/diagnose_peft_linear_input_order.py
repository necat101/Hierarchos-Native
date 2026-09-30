"""Diagnostic only: compare CPU linear-input-gradient reduction order."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

family = sys.argv[1] if len(sys.argv) > 1 else "minimax_m2"
saved_module = os.environ.get("HIERARCHOS_PEFT_SAVED_MODULE", "model.embed_tokens")
sys.argv = [sys.argv[0], family]

import verify_peft_lora_strict as strict
import verify_peft_multi_adapter as multi
from peft import PeftModel


def serial_fma_input_grad(grad_output: torch.Tensor, weight: torch.Tensor) -> np.ndarray:
    grad = grad_output.detach().cpu().numpy().astype(np.float64)
    w = weight.detach().cpu().numpy().astype(np.float64)
    out = np.zeros((grad.shape[0], w.shape[1]), dtype=np.float32)
    for k in range(w.shape[0]):
        out = (grad[:, k, None] * w[k, None, :] + out.astype(np.float64)).astype(np.float32)
    return out


def best_orders(grad_output: torch.Tensor, weight: torch.Tensor) -> list[tuple]:
    target = (grad_output @ weight).detach().cpu().numpy()
    grad = grad_output.detach().cpu().numpy().astype(np.float64)
    w = weight.detach().cpu().numpy().astype(np.float64)
    results = []
    for lanes in [1, 2, 4, 8, 16, 32]:
        if lanes > w.shape[0]:
            continue
        for fused in [True, False]:
            acc = np.zeros((*target.shape, lanes), dtype=np.float32)
            for k in range(w.shape[0]):
                product = grad[:, k, None] * w[k, None, :]
                if not fused:
                    product = product.astype(np.float32).astype(np.float64)
                lane = k % lanes
                acc[:, :, lane] = (product + acc[:, :, lane].astype(np.float64)).astype(np.float32)
            for reduction in ["sequential", "halves"]:
                values = [acc[:, :, i].copy() for i in range(lanes)]
                if reduction == "sequential":
                    out = np.zeros_like(target)
                    for value in values:
                        out = np.float32(out + value)
                else:
                    while len(values) > 1:
                        if len(values) % 2:
                            break
                        half = len(values) // 2
                        values = [np.float32(values[i] + values[i + half]) for i in range(half)]
                    if len(values) != 1:
                        continue
                    out = values[0]
                results.append(
                    (
                        float(np.abs(out - target).max()),
                        int(np.count_nonzero(out != target)),
                        lanes,
                        fused,
                        reduction,
                    )
                )
    return sorted(results)[:8]


torch.set_num_threads(1)
model = PeftModel.from_pretrained(
    multi.load_base(),
    strict.FIXTURE_ROOT / ("multi-saved-" + saved_module) / "adapter-a",
    is_trainable=True,
).train()
fixture = json.loads(multi.FIXTURE_JSON.read_text())
ids = torch.tensor(fixture["input_ids"]).reshape(fixture["batch_size"], fixture["seq_len"])
mask = torch.tensor(fixture["attention_mask"]).reshape_as(ids)

records = []
handles = []


def capture(name):
    def hook(module, args, output):
        record = {
            "name": name,
            "module": module,
            "input": args[0].detach().reshape(-1, args[0].shape[-1]),
        }

        def save_grad(grad):
            record["grad_output"] = grad.detach().reshape(-1, grad.shape[-1])

        output.register_hook(save_grad)
        records.append(record)

    return hook


for name, module in model.named_modules():
    if any(name.endswith("layers.0.self_attn." + p) for p in ["q_proj", "k_proj", "v_proj"]):
        handles.append(module.register_forward_hook(capture(name.rsplit(".", 1)[-1])))

model(input_ids=ids, attention_mask=mask, labels=ids).loss.backward()
for handle in handles:
    handle.remove()

for record in records:
    module = record["module"]
    grad_output = record["grad_output"]
    scale = torch.tensor(module.scaling["default"], dtype=grad_output.dtype)
    scaled = grad_output * scale
    grad_a_output = scaled @ module.lora_B.default.weight
    cases = [
        ("base", grad_output, module.base_layer.weight),
        ("B_input", scaled, module.lora_B.default.weight),
        ("A_input", grad_a_output, module.lora_A.default.weight),
    ]
    print("projection", record["name"], "scale", float(scale), flush=True)
    for label, grad, weight in cases:
        target = (grad @ weight).detach().cpu().numpy()
        serial = serial_fma_input_grad(grad, weight)
        print(
            label,
            "shape",
            list(weight.shape),
            "serial_max",
            float(np.abs(serial - target).max()),
            "serial_count",
            int(np.count_nonzero(serial != target)),
            "best",
            best_orders(grad, weight),
            flush=True,
        )
