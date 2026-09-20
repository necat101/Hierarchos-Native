#!/usr/bin/env python3
"""Verify canonical GPT-OSS SafeTensors export/reload through HF and Vulkan."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from verify_hf_logits import REPO, build_binary, save_reference_checkpoint, tiny_models


ATOL = 2.0e-7


def tiny_gpt_oss():
    for name, model, _ in tiny_models():
        if name == "gpt_oss":
            return model
    raise RuntimeError("tiny GPT-OSS fixture is missing")


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor, label: str) -> float:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{label}: shape mismatch {lhs.shape} != {rhs.shape}")
    delta = float((lhs - rhs).abs().max().item())
    if delta > ATOL:
        raise AssertionError(f"{label}: max_abs={delta:.9g} exceeds {ATOL:.9g}")
    return delta


def run_native(binary: Path, model_dir: Path, fixture: Path, report: Path, export_dir=None, device_index=None):
    command = [str(binary), "--model", str(model_dir), "--fixture", str(fixture), "--output", str(report)]
    if export_dir is not None:
        command.extend(["--export-model", str(export_dir)])
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    subprocess.run(command, cwd=REPO, check=True)
    return json.loads(report.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int)
    args = parser.parse_args()
    torch.manual_seed(0x48494552)
    model = tiny_gpt_oss().eval()
    binary = build_binary()

    with tempfile.TemporaryDirectory(prefix="vulkan-gpt-oss-roundtrip-") as directory:
        root = Path(directory)
        source = root / "source"
        exported = root / "exported"
        save_reference_checkpoint("gpt_oss", model, source)
        input_ids = torch.tensor([[1, 5, 9, 3, 7, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        fixture = source / "fixture.json"
        fixture.write_text(json.dumps({
            "batch_size": 1,
            "seq_len": input_ids.shape[1],
            "input_ids": input_ids.reshape(-1).tolist(),
            "attention_mask": attention_mask.to(torch.float32).reshape(-1).tolist(),
        }), encoding="utf-8")

        with torch.no_grad():
            reference = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu().float().reshape(-1)
        first = run_native(binary, source, fixture, root / "native-source.json", exported, args.device_index)
        if not (exported / "config.json").is_file() or not (exported / "model.safetensors").is_file():
            raise AssertionError("native export did not create canonical config.json and model.safetensors")
        config = json.loads((exported / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "gpt_oss":
            raise AssertionError(f"exported model_type is {config.get('model_type')!r}, expected 'gpt_oss'")

        reloaded = type(model).from_pretrained(exported).eval()
        worst_parameter = (0.0, "")
        reloaded_parameters = dict(reloaded.named_parameters())
        for name, parameter in model.named_parameters():
            if name not in reloaded_parameters:
                raise AssertionError(f"HF reload is missing parameter {name}")
            delta = float((parameter.detach().cpu() - reloaded_parameters[name].detach().cpu()).abs().max().item())
            worst_parameter = max(worst_parameter, (delta, name))
        if worst_parameter[0] > ATOL:
            raise AssertionError(f"HF reload parameter drift {worst_parameter[0]:.9g} at {worst_parameter[1]}")

        with torch.no_grad():
            hf_reloaded = reloaded(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu().float().reshape(-1)
        second = run_native(binary, exported, fixture, root / "native-reload.json", None, args.device_index)
        native_source = torch.tensor(first["logits"], dtype=torch.float32)
        native_reload = torch.tensor(second["logits"], dtype=torch.float32)
        results = {
            "hf_export_reload_logits": max_abs(hf_reloaded, reference, "HF export/reload logits"),
            "native_source_vs_hf": max_abs(native_source, reference, "native source vs HF"),
            "native_reload_vs_hf": max_abs(native_reload, reference, "native reload vs HF"),
            "native_reload_vs_native_source": max_abs(native_reload, native_source, "native reload vs native source"),
            "hf_parameter_max_abs": worst_parameter[0],
            "hf_parameter_worst": worst_parameter[1],
        }
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
