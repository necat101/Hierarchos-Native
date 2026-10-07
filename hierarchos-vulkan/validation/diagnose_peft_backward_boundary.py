"""Localize the first native-vs-HF divergent backward boundary for a PEFT
saved-module probe.

Runs the HF eager oracle for one fixture with module-level forward hooks plus
tensor hooks on each boundary (forward value, grad wrt module output, grad wrt
module input), monkeypatches the family's rotary helper, then matches every
captured boundary against the native ``--debug-first-layer-output`` trace from
``diagnose_peft_saved_gradient.py``.  A bit-exact match (max_abs == 0.0) proves
the key pairing; the first boundary whose best match is not bit-exact is where
native and HF execution topology diverge.

This never mutates fixtures or tolerances; it only reads the native trace.
"""
from __future__ import annotations

import argparse
import json
import tempfile

import torch
import verify_peft_multi_adapter as multi
import verify_peft_lora_strict as strict
from peft import PeftModel


def _tensor(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _tensor(item)
            if tensor is not None:
                return tensor
        return None
    if torch.is_tensor(value):
        return value
    return None


def _capture(model, args, fixture_ids, fixture_mask):
    capture = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, output):
            value = _tensor(output)
            if value is not None:
                capture[name + ".value"] = value.detach().float().cpu()
                if value.requires_grad:
                    value.register_hook(
                        lambda grad, key=name: capture.__setitem__(
                            key + ".grad_output", grad.detach().float().cpu()
                        )
                    )
            input_tensor = _tensor(inputs)
            if input_tensor is not None:
                capture[name + ".input"] = input_tensor.detach().float().cpu()
                if input_tensor.requires_grad:
                    input_tensor.register_hook(
                        lambda grad, key=name: capture.__setitem__(
                            key + ".grad_input", grad.detach().float().cpu()
                        )
                    )
        return hook

    for name, module in model.named_modules():
        if not name or name.startswith(("base_model.model.model.modules_to_save",)):
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    # Rotary boundaries are functions, not modules: patch the family helper.
    # Signatures differ across families: llama applies rotary to (q, k) in one
    # call while gemma-style helpers take a single tensor.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    modeling = type(base).__module__
    module_obj = __import__(modeling, fromlist=["apply_rotary_pos_emb"])
    original = module_obj.apply_rotary_pos_emb
    rotary_calls = []

    def make_rotary_entry(tensor):
        entry = {"value": tensor.detach().float().cpu()}
        if tensor.requires_grad:
            tensor.register_hook(
                lambda grad, captured=entry: captured.__setitem__(
                    "grad_output", grad.detach().float().cpu()
                )
            )
        return entry

    def capture_rotary(*f_args, **f_kwargs):
        result = original(*f_args, **f_kwargs)
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        rotary_calls.append(
            {
                "inputs": [
                    argument.detach().float().cpu()
                    for argument in f_args
                    if torch.is_tensor(argument)
                ],
                "outputs": [make_rotary_entry(item) for item in outputs if torch.is_tensor(item)],
            }
        )
        return result

    module_obj.apply_rotary_pos_emb = capture_rotary

    base_model = model.base_model.model
    logits_grad = {}

    def logits_hook(_module, _inputs, output):
        value = _tensor(output)
        if value is not None and value.requires_grad:
            value.register_hook(
                lambda grad: logits_grad.__setitem__(
                    "grad_logits", grad.detach().float().cpu()
                )
            )

    handles.append(base_model.register_forward_hook(logits_hook))
    try:
        output = model(input_ids=fixture_ids, attention_mask=fixture_mask, labels=fixture_ids)
        output.loss.backward()
    finally:
        module_obj.apply_rotary_pos_emb = original
        for handle in handles:
            handle.remove()
    capture["model.logits.grad_logits"] = logits_grad.get("grad_logits")
    return capture, rotary_calls, float(output.loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family")
    parser.add_argument("--module", default="model.embed_tokens")
    parser.add_argument("--adapter", choices=["a", "b"], default="a")
    parser.add_argument("--trace", required=True, help="native trace.json to compare against")
    parser.add_argument("--top", type=int, default=8, help="candidate native keys per boundary")
    parser.add_argument("--output", default=None, help="write the full comparison JSON here")
    parser.add_argument("--tensors", default=None, help="torch.save captured boundaries + parameter grads here")
    args = parser.parse_args()

    torch.set_num_threads(1)
    from pathlib import Path

    output = Path(tempfile.mkdtemp(prefix="boundary-probe-", dir=strict.FIXTURE_ROOT))
    multi.SAVED = args.module
    adapter = output / "adapter"
    multi._write_adapter(adapter, 101 if args.adapter == "a" else 202)
    fixture = json.loads(multi.FIXTURE_JSON.read_text(encoding="utf-8"))
    ids = torch.tensor(fixture["input_ids"]).reshape(fixture["batch_size"], fixture["seq_len"])
    mask = torch.tensor(fixture["attention_mask"]).reshape_as(ids)
    model = PeftModel.from_pretrained(multi.load_base(), adapter, is_trainable=True).train()
    capture, rotary_calls, loss = _capture(model, args, ids, mask)
    if args.tensors:
        gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        weights = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        torch.save(
            {
                "capture": {key: value for key, value in capture.items() if value is not None},
                "parameter_gradients": gradients,
                "parameters": weights,
                "input_ids": ids.clone(),
            },
            args.tensors,
        )

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    native = {key: torch.tensor(value, dtype=torch.float32) for key, value in trace.items()}

    # Match every captured boundary against native keys of the same element
    # count; bit-exact candidates are the trustworthy pairings.
    matches = {}
    for name, value in sorted(capture.items()):
        if value is None or not torch.is_tensor(value):
            continue
        candidates = []
        for key, other in native.items():
            if other.numel() != value.numel():
                continue
            delta = float((value.reshape(-1) - other.reshape(-1)).abs().max())
            candidates.append((delta, key))
        candidates.sort()
        matches[name] = {
            "numel": int(value.numel()),
            "max_abs": float(value.abs().max()),
            "candidates": [
                {"native_key": key, "max_abs": delta} for delta, key in candidates[: args.top]
            ],
            "bit_exact": [key for delta, key in candidates if delta == 0.0][: args.top],
        }
    rotary = [
        {
            "call": index,
            "outputs": [
                {
                    "numel": int(entry["value"].numel()),
                    "value_max_abs": float(entry["value"].abs().max()),
                    "grad_output_max_abs": float(entry["grad_output"].abs().max())
                    if "grad_output" in entry
                    else None,
                    "grad_candidates": sorted(
                        (
                            float(
                                (entry["grad_output"].reshape(-1) - other.reshape(-1)).abs().max()
                            ),
                            key,
                        )
                        for key, other in native.items()
                        if "grad_output" in entry and other.numel() == entry["value"].numel()
                    )[: args.top],
                    "value_candidates": sorted(
                        (
                            float((entry["value"].reshape(-1) - other.reshape(-1)).abs().max()),
                            key,
                        )
                        for key, other in native.items()
                        if other.numel() == entry["value"].numel()
                    )[: args.top],
                }
                for entry in call["outputs"]
            ],
        }
        for index, call in enumerate(rotary_calls)
    ]
    report = {
        "family": args.family,
        "fixture": "training.json",
        "module": args.module,
        "adapter": args.adapter,
        "loss": loss,
        "matches": matches,
        "rotary_calls": rotary,
        "trace": str(args.trace),
    }
    for name in sorted(matches):
        row = matches[name]
        bit = "BIT-EXACT" if row["bit_exact"] else "DIFF"
        if not row["candidates"]:
            # The native trace carries no tensor with this element count, so
            # there is nothing to pair this boundary against.  Report it
            # instead of indexing an empty candidate list.
            print(
                f"{name:52s} n={row['numel']:6d} max_abs={row['max_abs']:.3e} "
                f"best=<no native tensor of matching size> [{bit}]"
            )
            continue
        head = row["candidates"][0]
        print(
            f"{name:52s} n={row['numel']:6d} max_abs={row['max_abs']:.3e} "
            f"best={head['native_key']} d={head['max_abs']:.3e} [{bit}]"
        )
        for alternative in row["candidates"][1:4]:
            print(f"{'':52s}   alt {alternative['native_key']} d={alternative['max_abs']:.3e}")
    for entry in rotary:
        for output in entry["outputs"]:
            print(
                f"rotary call {entry['call']} n={output['numel']} "
                f"value={output['value_max_abs']:.3e} "
                f"value_best={output['value_candidates'][:2]} "
                f"grad_best={output['grad_candidates'][:2]}"
            )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
