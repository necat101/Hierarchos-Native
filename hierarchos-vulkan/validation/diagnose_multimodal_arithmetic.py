"""Attribute multimodal logit drift to a specific operation.

The frozen CPU oracle is replayed through the unmodified Transformers model with
the fixture's modality rows injected by the same embedding hook the fixture
generator used. A zero difference against `cpu.json` proves the harness runs the
reference op sequence, so any remaining difference is native drift rather than a
harness artifact.

`--mode norm` then replaces every normalization module with the arithmetic of the
corresponding native shader (`layer_norm_forward.comp` or the vector-CPU-rsqrt
mode of `transformer_rms_norm_forward.comp`) and reports the distance to the
native probe report. That separates forward-shape problems from reduction-order
problems instead of blaming the whole graph.

This is a diagnostic, never a qualification: it reads frozen expectations and
never writes or relaxes them.

Requires the pinned Torch/Transformers environment used to prepare the fixtures
(see `qualify_multimodal_training.py`). Execution still reads only frozen files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def f32(value):
    return np.float32(value)


def native_rms(x, weight, eps):
    """`transformer_rms_norm_forward.comp` in vector CPU-rsqrt mode."""
    rows = []
    weights = weight.tolist()
    for row in x.reshape(-1, x.shape[-1]).tolist():
        lanes = [np.float32(0.0)] * 8
        for index, value in enumerate(row):
            lanes[index & 7] = f32(lanes[index & 7] + f32(value * value))
        total = np.float32(0.0)
        for lane in lanes:
            total = f32(total + lane)
        mean_squared = f32(f32(total / f32(len(row))) + f32(eps))
        rstd = f32(np.float32(1.0) / np.sqrt(mean_squared))
        rows.append([f32(f32(f32(value * rstd) * weights[i])) for i, value in enumerate(row)])
    return torch.tensor(np.asarray(rows, dtype=np.float32)).view_as(x)


def native_layer_norm(x, weight, bias, eps):
    """`layer_norm_forward.comp` arithmetic."""
    rows = []
    weights, biases = weight.tolist(), bias.tolist()
    for row in x.reshape(-1, x.shape[-1]).tolist():
        mean = np.float32(0.0)
        for value in row:
            mean = f32(mean + value)
        mean = f32(mean / f32(len(row)))
        variance = np.float32(0.0)
        for value in row:
            centered = f32(value - mean)
            variance = f32(f32(centered * centered) + variance)
        variance = f32(variance / f32(len(row)))
        rstd = f32(np.float32(1.0) / np.sqrt(f32(variance + f32(eps))))
        rows.append([f32(f32(f32(f32(value - mean) * rstd) * weights[i]) + biases[i])
                     for i, value in enumerate(row)])
    return torch.tensor(np.asarray(rows, dtype=np.float32)).view_as(x)


class PatchedNorm(torch.nn.Module):
    def __init__(self, wrapped, kind):
        super().__init__()
        self.wrapped = wrapped
        self.kind = kind

    def forward(self, hidden_states):
        wrapped = self.wrapped
        eps = getattr(wrapped, "variance_epsilon", getattr(wrapped, "eps", 1e-6))
        if self.kind == "rms":
            return native_rms(hidden_states, wrapped.weight, eps)
        return native_layer_norm(hidden_states, wrapped.weight, wrapped.bias, eps)


def patch_norms(container):
    count = 0
    for name, module in list(container.named_children()):
        kind = ("rms" if "RMSNorm" in type(module).__name__
                else "ln" if "LayerNorm" in type(module).__name__ else None)
        if kind:
            setattr(container, name, PatchedNorm(module, kind))
            count += 1
        else:
            count += patch_norms(module)
    return count


def worst(actual, expected, visible):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    keep = np.asarray(visible, dtype=bool)
    return float(np.max(np.abs(actual[keep] - expected[keep])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="frozen CPU fixture root containing <case>/cpu.json")
    parser.add_argument("--case", required=True, help="case path, for example gemma4/unmasked")
    parser.add_argument("--native", type=Path,
                        help="native probe output root containing <case>/report.json")
    parser.add_argument("--mode", choices=["none", "norm"], default="norm",
                        help="operation to replace after the oracle check")
    args = parser.parse_args()

    torch.set_num_threads(1)
    case = args.fixtures / args.case
    fixture = json.loads((case / "input.json").read_text())
    cpu = json.loads((case / "cpu.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(case / "model", torch_dtype=torch.float32).eval()
    ids = torch.tensor(fixture["input_ids"], dtype=torch.long).view(
        fixture["batch_size"], fixture["seq_len"])
    mask = torch.tensor(fixture["attention_mask"], dtype=torch.float32).view_as(ids)
    embedding = model.get_input_embeddings()
    projected = torch.tensor(cpu["steps"][0]["projected"], dtype=torch.float32)
    rows = torch.tensor(fixture["modality_rows"])
    dim = projected.numel() // len(fixture["modality_rows"])

    def hook():
        def replace_rows(module, inputs, output):
            replacement = projected.view(len(fixture["modality_rows"]), dim)
            replacement = replacement * getattr(module, "embed_scale", 1.0)
            return output.reshape(-1, dim).index_copy(0, rows, replacement).reshape_as(output)
        return embedding.register_forward_hook(replace_rows)

    def logits():
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask, use_cache=False).logits.flatten().numpy()

    visible = cpu["visible"]
    frozen = np.asarray(cpu["steps"][0]["logits"], dtype=np.float32)
    handle = hook()
    baseline = logits()
    handle.remove()
    print(f"unpatched: vs frozen cpu = {worst(baseline, frozen, visible):.6g}")
    if worst(baseline, frozen, visible) != 0.0:
        raise SystemExit("harness does not reproduce the frozen oracle; do not interpret the patch")

    if args.native is None or args.mode == "none":
        return
    report = json.loads((args.native / args.case / "report.json").read_text())
    native = np.asarray(report["steps"][0]["logits"], dtype=np.float32)
    print(f"unpatched: vs native report = {worst(baseline, native, visible):.6g}")
    handle = hook()
    patched_count = patch_norms(model)
    patched = logits()
    handle.remove()
    print(f"patched {patched_count} norms: vs frozen cpu = {worst(patched, frozen, visible):.6g}; "
          f"vs native report = {worst(patched, native, visible):.6g}")


if __name__ == "__main__":
    main()
