#!/usr/bin/env python3
"""Check Vulkan loss, backward, AdamW and checkpoint export against this checkout.

Uses tiny dense/MoE causal and masked language models, two optimizer steps, and
compares every named trainable parameter. Dropout is disabled; --jitter supplies
shared Philox draws to HF's input jitter while retaining its forward/autograd.
This checks stochastic math, not equivalence to PyTorch's native RNG stream.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from verify_hf_logits import (
    HEADLINE_ADAMW_FAMILIES,
    REPO,
    build_binary,
    model_vocab_size,
    save_reference_checkpoint,
    tiny_models,
    transformers,
)


def philox_word(seed, word_index):
    mask = 0xFFFFFFFF
    block = word_index // 4
    counter = [block & mask, block >> 32, 0, 0]
    key = [seed & mask, seed >> 32]
    for _ in range(10):
        p0, p1 = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
        counter = [(p1 >> 32) ^ counter[1] ^ key[0], p1 & mask, (p0 >> 32) ^ counter[3] ^ key[1], p0 & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter[word_index % 4]


def compare_training(binary, root, name, model, device_index, jitter=False, steps=2):
    # The generic train probe takes same-position targets. Explicitly shift
    # causal labels here; masked LM uses labels at their original positions.
    model.train()
    if not jitter:
        for module in model.modules():
            for field in ("jitter_noise", "moe_jitter_eps"):
                if hasattr(module, field):
                    setattr(module, field, 0.0)
        if hasattr(model.config, "router_jitter_noise"):
            model.config.router_jitter_noise = 0.0
        if name == "dbrx":
            model.config.ffn_config.moe_jitter_eps = 0.0
    source = root / name / "source"
    destination = root / name / "trained"
    # Validate the backend against Transformers' current normalized parameter
    # graph, not an optional reverse conversion back into a vendor/legacy Hub
    # checkpoint layout.  DeepSeek-V4 in particular has a registered reverse
    # mapping (`self_attn.*` -> `attn.w*`, packed experts -> per-expert w1/w2/w3)
    # that changes serialization names/shapes without changing the model math.
    # Keeping that conversion disabled makes this an AdamW/gradient parity
    # check rather than a checkpoint-conversion test.
    save_reference_checkpoint(name, model, source)
    input_ids = torch.tensor([[1, 5, 9, 3, 7, 2]], dtype=torch.long)
    targets = torch.tensor([[5, 9, 3, 7, 2, -100]], dtype=torch.long)
    if name == "bert":
        targets = torch.tensor([[-100, 5, -100, 3, 7, -100]], dtype=torch.long)
    hyper = {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 0.001, "weight_decay": 0.0}
    fixture = {
        "batch_size": 1,
        "seq_len": 6,
        "input_ids": input_ids.reshape(-1).tolist(),
        "targets": [int(t) if t >= 0 else 0xFFFFFFFF for t in targets.reshape(-1)],
        "attention_mask": [1.0] * 6,
        "loss_weights": targets.ne(-100).float().reshape(-1).tolist(),
        **hyper,
    }
    fixture_path = root / name / "training.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    command = [
        str(binary),
        "--model",
        str(source),
        "--fixture",
        str(fixture_path),
        "--output",
        str(destination),
        "--steps",
        str(steps),
    ]
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    try:
        native = subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise RuntimeError(f"{name}: native training probe failed\n{details}") from exc
    report = json.loads(native.stdout)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyper["learning_rate"],
        betas=(hyper["beta1"], hyper["beta2"]),
        eps=hyper["eps"],
        weight_decay=hyper["weight_decay"],
        foreach=False,
    )
    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        draw_index = 0

        def replay_uniform(tensor, low=0.0, high=1.0, *, generator=None):
            nonlocal draw_index
            # Dense MoE fixtures have exactly one input-jitter draw per layer.
            # Replace only the reference RNG; keep HF's forward and autograd.
            site = (0x50000000 + draw_index) ^ 0x90000000
            seed = (site << 32) | 0x48494552
            words = [philox_word(seed, (step << 32) + i) for i in range(tensor.numel())]
            uniform = torch.tensor([word >> 8 for word in words], dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(low + (high - low) * (uniform / 16777216.0))
            draw_index += 1
            return tensor

        with patch.object(torch.Tensor, "uniform_", replay_uniform) if jitter else nullcontext():
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits
        if jitter and draw_index != model.config.num_hidden_layers:
            raise AssertionError(f"{name}: expected one jitter draw per layer, got {draw_index}")
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model_vocab_size(model)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss_mismatches = []
    for index, (native_loss, reference_loss) in enumerate(zip(report["losses"], losses, strict=True)):
        if not math.isclose(native_loss, reference_loss, rel_tol=2e-5, abs_tol=2e-5):
            loss_mismatches.append(
                f"step {index + 1}: Vulkan loss {native_loss} != HF {reference_loss}"
            )
    saved = load_file(destination / "model.safetensors")
    worst = (0.0, "")
    parameter_deltas = []
    count = 0
    # named_parameters deduplicates tied tables, so each optimizer parameter
    # is checked once. Resolve HF's tied aliases in the exported package.
    aliases = getattr(model, "all_tied_weights_keys", {})
    for key, parameter in model.named_parameters():
        saved_key = key
        if saved_key not in saved:
            saved_key = aliases.get(key, key)
        if saved_key not in saved:
            raise AssertionError(f"{name}: exported checkpoint is missing parameter {key}")
        actual = saved[saved_key]
        expected = parameter.detach().cpu()
        delta = float((actual - expected).abs().max())
        parameter_deltas.append((delta, key))
        worst = max(worst, (delta, key))
        count += parameter.numel()
    parameter_deltas.sort(reverse=True)
    if loss_mismatches or worst[0] > 2.0e-7:
        top = ", ".join(f"{key}={delta:.9g}" for delta, key in parameter_deltas[:12])
        details = "; ".join(loss_mismatches)
        if details:
            details += "; "
        raise AssertionError(
            f"{name}: {details}worst parameter error {worst[0]:.9g} at {worst[1]}; top deltas: {top}"
        )
    return {
        "family": name,
        "router_jitter": jitter,
        "device": report["device"],
        "steps": steps,
        "losses": losses,
        "parameter_values": count,
        "max_abs": worst[0],
        "worst_parameter": worst[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--families", nargs="+")
    parser.add_argument(
        "--headline-eight",
        "--headline-strict",
        dest="headline_eight",
        action="store_true",
        help="run the source-backed families used by the strict two-step AdamW compatibility claim",
    )
    parser.add_argument("--steps", type=int, default=2, help="optimizer steps to compare (default: 2)")
    parser.add_argument("--jitter", action="store_true", help="verify MoE jitter with shared Philox draws")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.headline_eight and args.families:
        parser.error("--headline-strict cannot be combined with --families")
    if args.headline_eight and args.jitter:
        parser.error("--headline-strict cannot be combined with --jitter")
    if args.headline_eight and args.steps != 2:
        parser.error("--headline-strict is defined as exactly two AdamW steps")
    torch.manual_seed(0x48494552)
    models = [
        entry
        for entry in tiny_models(training_reference=True)
        if not entry[1].config.is_encoder_decoder
    ]
    if args.jitter:
        models = [entry for entry in models if entry[0] in {"mixtral", "minimax_m2", "dbrx"}]
    if args.headline_eight:
        model_by_name = {entry[0]: entry for entry in models}
        missing = [name for name in HEADLINE_ADAMW_FAMILIES if name not in model_by_name]
        if missing:
            parser.error(f"headline families are missing training fixtures: {missing}")
        models = [model_by_name[name] for name in HEADLINE_ADAMW_FAMILIES]
    elif args.families:
        unknown = set(args.families) - {name for name, _, _ in models}
        if unknown:
            raise ValueError(f"unknown training families: {sorted(unknown)}")
        models = [entry for entry in models if entry[0] in args.families]
    binary = build_binary("transformer_parity")
    with tempfile.TemporaryDirectory(prefix="vulkan-hf-training-") as directory:
        results = [
            compare_training(
                binary,
                Path(directory),
                name,
                model,
                args.device_index,
                args.jitter,
                args.steps,
            )
            for name, model, _ in models
        ]
    print(
        json.dumps(
            {
                "result": "pass",
                "transformers_version": transformers.__version__,
                "transformers_source": transformers.__file__,
                "families": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
