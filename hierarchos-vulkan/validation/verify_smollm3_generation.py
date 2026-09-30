#!/usr/bin/env python3
"""Strict per-step SmolLM3 cached/full-prefix generation parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from verify_hf_logits import REPO, build_binary, save_reference_checkpoint, tiny_models
from transformers import DynamicCache


ATOL = 2.0e-7
FAMILIES = ("smollm3", "smollm3_yarn")


def hf_generation(model, input_ids, use_cache, steps):
    sequence = input_ids.clone()
    logits = []
    cache = DynamicCache(config=model.config) if use_cache else None
    current_input = input_ids

    with torch.no_grad():
        for _ in range(steps):
            if use_cache:
                position_start = sequence.shape[1] - current_input.shape[1]
                position_ids = torch.arange(
                    position_start, sequence.shape[1], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
                model_input = current_input
            else:
                position_ids = torch.arange(sequence.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
                model_input = sequence

            output = model(
                input_ids=model_input,
                attention_mask=torch.ones_like(sequence),
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=use_cache,
            )
            step_logits = output.logits[:, -1, :]
            logits.append(step_logits.detach().cpu().to(torch.float32).reshape(-1))
            next_token = step_logits.argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_token), dim=1)
            current_input = next_token

    return sequence.detach().cpu().reshape(-1).tolist(), logits


def native_generation(binary, model_dir, fixture_path, use_cache, device_index):
    suffix = "cache" if use_cache else "full"
    report_path = model_dir / f"native-{suffix}.json"
    command = [str(binary), "--model", str(model_dir), "--fixture", str(fixture_path)]
    command.extend(["--output", str(report_path), "--use-cache", str(use_cache).lower()])
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    subprocess.run(command, cwd=REPO, check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def compare_steps(lhs, rhs, label):
    if len(lhs) != len(rhs):
        raise AssertionError(f"{label}: step count mismatch")
    per_step = []
    for index, (left, right) in enumerate(zip(lhs, rhs, strict=True)):
        if left.shape != right.shape:
            raise AssertionError(f"{label}: step {index} shape mismatch")
        delta = float((left - right).abs().max().item())
        per_step.append(delta)
        if delta > ATOL:
            raise AssertionError(f"{label}: step {index} max_abs={delta:.9g} exceeds {ATOL:.9g}")
    return {"max_abs": max(per_step, default=0.0), "per_step_max_abs": per_step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--family", choices=FAMILIES, default="smollm3")
    parser.add_argument("--keep-fixtures", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    torch.manual_seed(0x48494552)
    matches = [(name, model) for name, model, _ in tiny_models() if name == args.family]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique {args.family} fixture")
    name, model = matches[0]
    model = model.eval()

    temporary = None
    if args.keep_fixtures is None:
        temporary = tempfile.TemporaryDirectory(prefix="vulkan-smollm3-generation-")
        root = Path(temporary.name)
    else:
        root = args.keep_fixtures.resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        model_dir = root / name
        save_reference_checkpoint(name, model, model_dir)
        input_ids = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
        fixture_path = model_dir / "generation.json"
        fixture_path.write_text(
            json.dumps({"input_ids": input_ids.reshape(-1).tolist(), "max_new_tokens": args.steps}),
            encoding="utf-8",
        )

        hf_cached_sequence, hf_cached_logits = hf_generation(model, input_ids, True, args.steps)
        hf_full_sequence, hf_full_logits = hf_generation(model, input_ids, False, args.steps)
        binary = build_binary("transformer_generation_parity")
        native_cached = native_generation(binary, model_dir, fixture_path, True, args.device_index)
        native_full = native_generation(binary, model_dir, fixture_path, False, args.device_index)
        native_cached_logits = [torch.tensor(step, dtype=torch.float32) for step in native_cached["logits"]]
        native_full_logits = [torch.tensor(step, dtype=torch.float32) for step in native_full["logits"]]
        sequences = [hf_cached_sequence, hf_full_sequence, native_cached["sequence"], native_full["sequence"]]
        if any(sequence != sequences[0] for sequence in sequences[1:]):
            raise AssertionError(f"{name}: cached/full-prefix generated sequences differ: {sequences}")

        report = {
            "result": "pass",
            "atol": ATOL,
            "family": name,
            "sequence": sequences[0],
            "vulkan_cached_vs_transformers_cached": compare_steps(
                native_cached_logits, hf_cached_logits, "Vulkan cached vs Transformers cached"
            ),
            "vulkan_full_vs_transformers_full": compare_steps(
                native_full_logits, hf_full_logits, "Vulkan full-prefix vs Transformers full-prefix"
            ),
            "vulkan_cached_vs_vulkan_full": compare_steps(
                native_cached_logits, native_full_logits, "Vulkan cached vs Vulkan full-prefix"
            ),
            "transformers_cached_vs_transformers_full": compare_steps(
                hf_cached_logits, hf_full_logits, "Transformers cached vs Transformers full-prefix"
            ),
        }
        print(json.dumps(report, indent=2))
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
