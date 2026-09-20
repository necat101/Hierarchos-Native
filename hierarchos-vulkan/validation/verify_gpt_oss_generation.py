#!/usr/bin/env python3
"""Strict per-step GPT-OSS generation parity against local Transformers."""

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


def hf_generation(model, input_ids, use_cache, steps):
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=steps,
            do_sample=False,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_logits=True,
            eos_token_id=[],
            pad_token_id=0,
        )
    logits = [step.detach().cpu().to(torch.float32).reshape(-1) for step in output.logits]
    if len(logits) != steps:
        raise AssertionError(f"Transformers returned {len(logits)} generation steps, expected {steps}")
    return output.sequences.detach().cpu().reshape(-1).tolist(), logits


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


def run_parity(binary, root, model, steps, device_index):
    model_dir = root / "gpt_oss"
    save_reference_checkpoint("gpt_oss", model, model_dir)
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
        raise AssertionError("cached/full-prefix generated sequences differ")

    comparisons = {}
    comparisons["vulkan_cached_vs_transformers_cached"] = compare_steps(
        native_cached_logits, hf_cached_logits, "Vulkan cached vs Transformers cached"
    )
    comparisons["vulkan_full_vs_transformers_full"] = compare_steps(
        native_full_logits, hf_full_logits, "Vulkan full-prefix vs Transformers full-prefix"
    )
    comparisons["vulkan_cached_vs_vulkan_full"] = compare_steps(
        native_cached_logits, native_full_logits, "Vulkan cached vs Vulkan full-prefix"
    )
    comparisons["transformers_cached_vs_transformers_full"] = compare_steps(
        hf_cached_logits, hf_full_logits, "Transformers cached vs Transformers full-prefix"
    )
    print(json.dumps(comparisons, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    torch.manual_seed(0x48494552)
    model = tiny_gpt_oss().eval()
    binary = build_binary("transformer_generation_parity")
    with tempfile.TemporaryDirectory(prefix="vulkan-gpt-oss-generation-") as directory:
        run_parity(binary, Path(directory), model, args.steps, args.device_index)


if __name__ == "__main__":
    main()
