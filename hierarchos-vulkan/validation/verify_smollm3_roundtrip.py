#!/usr/bin/env python3
"""Strict SmolLM3 train/export/reload/forward/generation parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from verify_hf_logits import REPO, build_binary, tiny_models
from verify_hf_training import compare_training
from verify_smollm3_generation import ATOL, compare_steps, hf_generation, native_generation


FAMILIES = ("smollm3", "smollm3_yarn")


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor, label: str) -> float:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{label}: shape mismatch {lhs.shape} != {rhs.shape}")
    delta = float((lhs - rhs).abs().max().item())
    if delta > ATOL:
        raise AssertionError(f"{label}: max_abs={delta:.9g} exceeds {ATOL:.9g}")
    return delta


def native_logits(binary, model_dir, fixture, report, device_index):
    command = [str(binary), "--model", str(model_dir), "--fixture", str(fixture), "--output", str(report)]
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    subprocess.run(command, cwd=REPO, check=True)
    return torch.tensor(json.loads(report.read_text(encoding="utf-8"))["logits"], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--family", choices=FAMILIES, default="smollm3")
    parser.add_argument("--keep-fixtures", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    torch.manual_seed(0x48494552)
    matches = [(name, model) for name, model, _ in tiny_models(training_reference=True) if name == args.family]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique {args.family} training fixture")
    name, model = matches[0]

    temporary = None
    if args.keep_fixtures is None:
        temporary = tempfile.TemporaryDirectory(prefix="vulkan-smollm3-roundtrip-")
        root = Path(temporary.name)
    else:
        root = args.keep_fixtures.resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        train_binary = build_binary("transformer_parity")
        logits_binary = build_binary()
        generation_binary = build_binary("transformer_generation_parity")
        training = compare_training(train_binary, root, name, model, args.device_index, jitter=False, steps=2)
        trained = root / name / "trained"
        config_path = trained / "config.json"
        weights_path = trained / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise AssertionError("native training export did not create config.json and model.safetensors")

        exported_config = json.loads(config_path.read_text(encoding="utf-8"))
        if exported_config.get("model_type") != "smollm3":
            raise AssertionError(f"exported model_type={exported_config.get('model_type')!r}, expected 'smollm3'")
        expected_nope = list(model.config.no_rope_layers)
        expected_types = list(model.config.layer_types)
        if exported_config.get("no_rope_layers") != expected_nope:
            raise AssertionError("exported no_rope_layers changed across native training/save")
        if exported_config.get("layer_types") != expected_types:
            raise AssertionError("exported layer_types changed across native training/save")
        expected_rope = dict(model.config.rope_parameters)
        if exported_config.get("rope_parameters") != expected_rope:
            raise AssertionError("exported rope_parameters changed across native training/save")
        if exported_config.get("use_sliding_window") != model.config.use_sliding_window:
            raise AssertionError("exported use_sliding_window changed across native training/save")
        if exported_config.get("sliding_window") != model.config.sliding_window:
            raise AssertionError("exported sliding_window changed across native training/save")
        if exported_config.get("tie_word_embeddings") != model.config.tie_word_embeddings:
            raise AssertionError("exported tie_word_embeddings changed across native training/save")

        reloaded = type(model).from_pretrained(trained).eval()
        reloaded_parameters = dict(reloaded.named_parameters())
        worst_parameter = (0.0, "")
        for parameter_name, parameter in model.named_parameters():
            if parameter_name not in reloaded_parameters:
                raise AssertionError(f"HF reload is missing parameter {parameter_name}")
            delta = float(
                (parameter.detach().cpu() - reloaded_parameters[parameter_name].detach().cpu()).abs().max().item()
            )
            worst_parameter = max(worst_parameter, (delta, parameter_name))
        if worst_parameter[0] > ATOL:
            raise AssertionError(
                f"HF reload parameter drift {worst_parameter[0]:.9g} at {worst_parameter[1]} exceeds {ATOL:.9g}"
            )

        input_ids = torch.tensor([[1, 5, 9, 3, 7, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            hf_reloaded = (
                reloaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                .logits.cpu().float().reshape(-1)
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
        native_reloaded = native_logits(
            logits_binary, trained, fixture, root / name / "native_reload.json", args.device_index
        )
        reload_forward_max_abs = max_abs(native_reloaded, hf_reloaded, "native reload vs HF reload")

        generation_input = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
        generation_fixture = root / name / "reload_generation.json"
        generation_fixture.write_text(
            json.dumps({"input_ids": generation_input.reshape(-1).tolist(), "max_new_tokens": args.steps}),
            encoding="utf-8",
        )
        hf_cached_sequence, hf_cached_logits = hf_generation(reloaded, generation_input, True, args.steps)
        hf_full_sequence, hf_full_logits = hf_generation(reloaded, generation_input, False, args.steps)
        native_cached = native_generation(
            generation_binary, trained, generation_fixture, True, args.device_index
        )
        native_full = native_generation(
            generation_binary, trained, generation_fixture, False, args.device_index
        )
        native_cached_logits = [torch.tensor(step, dtype=torch.float32) for step in native_cached["logits"]]
        native_full_logits = [torch.tensor(step, dtype=torch.float32) for step in native_full["logits"]]
        sequences = [hf_cached_sequence, hf_full_sequence, native_cached["sequence"], native_full["sequence"]]
        if any(sequence != sequences[0] for sequence in sequences[1:]):
            raise AssertionError(f"trained/reloaded cached/full-prefix generated sequences differ: {sequences}")

        print(
            json.dumps(
                {
                    "result": "pass",
                    "atol": ATOL,
                    "family": name,
                    "training_parameter_max_abs": training["max_abs"],
                    "hf_reload_parameter_max_abs": worst_parameter[0],
                    "hf_reload_parameter_worst": worst_parameter[1],
                    "native_reload_vs_hf_reload_max_abs": reload_forward_max_abs,
                    "trained_vulkan_cached_vs_transformers_cached": compare_steps(
                        native_cached_logits, hf_cached_logits, "trained Vulkan cached vs Transformers cached"
                    ),
                    "trained_vulkan_full_vs_transformers_full": compare_steps(
                        native_full_logits, hf_full_logits, "trained Vulkan full vs Transformers full"
                    ),
                    "trained_vulkan_cached_vs_vulkan_full": compare_steps(
                        native_cached_logits, native_full_logits, "trained Vulkan cached vs Vulkan full"
                    ),
                    "sequence": sequences[0],
                },
                indent=2,
            )
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
