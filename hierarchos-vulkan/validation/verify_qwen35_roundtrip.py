#!/usr/bin/env python3
"""Verify trained strict Qwen3.5/Qwen4-Exp native save/reload against local Transformers."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from verify_hf_logits import QWEN35_STRICT_FAMILIES, QWEN4_EXP_STRICT_FAMILIES, REPO, build_binary, tiny_models
from verify_hf_training import compare_training


ATOL = 2.0e-7
STRICT_FAMILIES = (*QWEN35_STRICT_FAMILIES, *QWEN4_EXP_STRICT_FAMILIES)


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor, label: str) -> float:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{label}: shape mismatch {lhs.shape} != {rhs.shape}")
    delta = float((lhs - rhs).abs().max().item())
    if delta > ATOL:
        raise AssertionError(f"{label}: max_abs={delta:.9g} exceeds {ATOL:.9g}")
    return delta


def native_logits(binary: Path, model_dir: Path, fixture: Path, report: Path, device_index: int | None):
    command = [str(binary), "--model", str(model_dir), "--fixture", str(fixture), "--output", str(report)]
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    subprocess.run(command, cwd=REPO, check=True)
    return torch.tensor(json.loads(report.read_text(encoding="utf-8"))["logits"], dtype=torch.float32)


def verify_family(train_binary: Path, logits_binary: Path, root: Path, name: str, model, device_index: int | None):
    training = compare_training(train_binary, root, name, model, device_index, jitter=False, steps=2)
    trained = root / name / "trained"
    config_path = trained / "config.json"
    weights_path = trained / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise AssertionError(f"{name}: native training export did not create config.json and model.safetensors")

    exported_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_model_type = model.config.model_type
    if exported_config.get("model_type") != expected_model_type:
        raise AssertionError(
            f"{name}: exported model_type={exported_config.get('model_type')!r}, expected {expected_model_type!r}"
        )

    model.eval()
    reloaded = type(model).from_pretrained(trained).eval()
    reloaded_parameters = dict(reloaded.named_parameters())
    worst_parameter = (0.0, "")
    for parameter_name, parameter in model.named_parameters():
        if parameter_name not in reloaded_parameters:
            raise AssertionError(f"{name}: HF reload is missing parameter {parameter_name}")
        delta = float(
            (parameter.detach().cpu() - reloaded_parameters[parameter_name].detach().cpu()).abs().max().item()
        )
        worst_parameter = max(worst_parameter, (delta, parameter_name))
    if worst_parameter[0] > ATOL:
        raise AssertionError(
            f"{name}: HF reload parameter drift {worst_parameter[0]:.9g} at {worst_parameter[1]}"
        )

    input_ids = torch.tensor([[1, 5, 9, 3, 7, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        hf_reloaded = (
            reloaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            .logits.cpu()
            .float()
            .reshape(-1)
        )

    fixture = root / name / "reload_fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "seq_len": input_ids.shape[1],
                "input_ids": input_ids.reshape(-1).tolist(),
                "attention_mask": attention_mask.float().reshape(-1).tolist(),
            }
        ),
        encoding="utf-8",
    )
    native_reloaded = native_logits(logits_binary, trained, fixture, root / name / "native_reload.json", device_index)

    return {
        "family": name,
        "training_parameter_max_abs": training["max_abs"],
        "hf_reload_parameter_max_abs": worst_parameter[0],
        "hf_reload_parameter_worst": worst_parameter[1],
        "native_reload_vs_hf_reload_max_abs": max_abs(
            native_reloaded, hf_reloaded, f"{name} native reload vs HF reload"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--families", nargs="+", choices=STRICT_FAMILIES)
    args = parser.parse_args()

    requested = set(args.families or QWEN35_STRICT_FAMILIES)
    torch.manual_seed(0x48494552)
    models = [(name, model) for name, model, _ in tiny_models(training_reference=True) if name in requested]
    missing = requested - {name for name, _ in models}
    if missing:
        raise RuntimeError(f"missing Qwen hybrid fixtures: {sorted(missing)}")

    train_binary = build_binary("transformer_parity")
    logits_binary = build_binary()
    with tempfile.TemporaryDirectory(prefix="vulkan-qwen35-roundtrip-") as directory:
        root = Path(directory)
        results = [
            verify_family(train_binary, logits_binary, root, name, model, args.device_index)
            for name, model in models
        ]
    print(json.dumps({"result": "pass", "atol": ATOL, "families": results}, indent=2))


if __name__ == "__main__":
    main()
