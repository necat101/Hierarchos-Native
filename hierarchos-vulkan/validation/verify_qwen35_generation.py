#!/usr/bin/env python3
"""Strict per-step Qwen3.5 generation parity against the local Transformers checkout."""

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
FAMILIES = ("qwen3_5_full", "qwen3_5_linear", "qwen3_5_mixed", "qwen3_5_moe")


def hf_generation(model, input_ids, use_cache, steps):
    sequence = input_ids.clone()
    logits = []
    cache = DynamicCache(config=model.config) if use_cache else None
    current_input = input_ids

    with torch.no_grad():
        for step in range(steps):
            if use_cache:
                position_start = sequence.shape[1] - current_input.shape[1]
                position_ids = torch.arange(
                    position_start,
                    sequence.shape[1],
                    dtype=torch.long,
                    device=input_ids.device,
                ).unsqueeze(0)
                model_input = current_input
            else:
                position_ids = torch.arange(
                    sequence.shape[1], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
                model_input = sequence

            attention_mask = torch.ones_like(sequence)
            if use_cache and all(layer_type == "linear_attention" for layer_type in model.config.layer_types):
                # Qwen3.5 constructs both mask families before dispatching layers.  An
                # all-linear synthetic config has no sequence-bearing attention cache,
                # so upstream create_causal_mask cannot query a cache length even though
                # no full-attention layer will consume that mask.  For this unpadded
                # generation fixture both effective masks are exactly None.
                attention_mask = {"full_attention": None, "linear_attention": None}

            output = model(
                input_ids=model_input,
                attention_mask=attention_mask,
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


def run_family(binary, root, name, model, steps, device_index):
    model = model.eval()
    model_dir = root / name
    save_reference_checkpoint(name, model, model_dir)
    input_ids = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
    fixture_path = model_dir / "generation.json"
    fixture_path.write_text(
        json.dumps({"input_ids": input_ids.reshape(-1).tolist(), "max_new_tokens": steps}),
        encoding="utf-8",
    )
    hf_cached_sequence, hf_cached_logits = hf_generation(model, input_ids, True, steps)
    hf_full_sequence, hf_full_logits = hf_generation(model, input_ids, False, steps)
    native_cached = native_generation(binary, model_dir, fixture_path, True, device_index)
    native_full = native_generation(binary, model_dir, fixture_path, False, device_index)
    native_cached_logits = [torch.tensor(step, dtype=torch.float32) for step in native_cached["logits"]]
    native_full_logits = [torch.tensor(step, dtype=torch.float32) for step in native_full["logits"]]
    sequences = [hf_cached_sequence, hf_full_sequence, native_cached["sequence"], native_full["sequence"]]
    if any(sequence != sequences[0] for sequence in sequences[1:]):
        raise AssertionError(f"{name}: cached/full-prefix generated sequences differ: {sequences}")
    return {
        "family": name,
        "sequence": sequences[0],
        "vulkan_cached_vs_transformers_cached": compare_steps(
            native_cached_logits, hf_cached_logits, f"{name} Vulkan cached vs Transformers cached"
        ),
        "vulkan_full_vs_transformers_full": compare_steps(
            native_full_logits, hf_full_logits, f"{name} Vulkan full-prefix vs Transformers full-prefix"
        ),
        "vulkan_cached_vs_vulkan_full": compare_steps(
            native_cached_logits, native_full_logits, f"{name} Vulkan cached vs Vulkan full-prefix"
        ),
        "transformers_cached_vs_transformers_full": compare_steps(
            hf_cached_logits, hf_full_logits, f"{name} Transformers cached vs Transformers full-prefix"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    requested = set(args.families or FAMILIES)
    torch.manual_seed(0x48494552)
    models = [(name, model) for name, model, _ in tiny_models() if name in requested]
    missing = requested - {name for name, _ in models}
    if missing:
        raise RuntimeError(f"missing Qwen3.5 fixtures: {sorted(missing)}")
    binary = build_binary("transformer_generation_parity")
    with tempfile.TemporaryDirectory(prefix="vulkan-qwen35-generation-") as directory:
        results = [run_family(binary, Path(directory), name, model, args.steps, args.device_index) for name, model in models]
    print(json.dumps({"result": "pass", "atol": ATOL, "families": results}, indent=2))


if __name__ == "__main__":
    main()
