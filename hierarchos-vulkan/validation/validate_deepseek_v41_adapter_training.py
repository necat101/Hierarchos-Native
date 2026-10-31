"""Pinned CPU PyTorch fixture for the V4.1 adapter optimizer.

The native trainer owns host FP32 LoRA factor masters plus AdamW moments. This
script builds a two-module adapter in the standard PEFT layout, runs three
`torch.optim.AdamW` steps on pinned gradients with `foreach=False` (the
single-tensor CPU path), and records every factor and moment after each step as
FP32 bit patterns. `train_tests.rs` replays the same steps through the native
state and gates every value at the release tolerance.

The gradients are pinned rather than produced by the graph: the block adjoints
that will produce them are not implemented yet, and this fixture qualifies the
update rule, the moment layout and the resume contract on their own.

    .\\.venv-vulkan\\Scripts\\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_adapter_training.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

SEED = 4141201
STEPS = 3
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
TOLERANCE = 2e-7
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v41_adapter_training.json"
ADAPTER = (
    Path(__file__).resolve().parent / "fixtures" / "deepseek_v41_adapter_training" / "adapter"
)
TARGETS = [
    ("layers.1.attn.wq_b", 3, 17, 11, 6.0),
    ("layers.2.ffn.shared_experts.gate_proj", 2, 19, 13, 4.0),
]


def bits(value: torch.Tensor) -> list[int]:
    return value.contiguous().flatten().view(torch.uint32).tolist()


def deterministic(count: int, salt: int, scale: float) -> torch.Tensor:
    """Explicit LCG so the fixture does not depend on torch's RNG stream."""
    values = []
    state = (SEED * 1103515245 + salt) & 0x7FFFFFFF
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        values.append(((state / 0x7FFFFFFF) * 2.0 - 1.0) * scale)
    return torch.tensor(values, dtype=torch.float32)


def write_adapter(initial: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    if ADAPTER.exists():
        shutil.rmtree(ADAPTER)
    ADAPTER.mkdir(parents=True)
    tensors = {}
    for target, (a, b) in initial.items():
        tensors[f"base_model.model.{target}.lora_A.weight"] = a
        tensors[f"base_model.model.{target}.lora_B.weight"] = b
    save_file(tensors, str(ADAPTER / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": TARGETS[0][1],
        "lora_alpha": 8.0,
        "lora_dropout": 0.0,
        "target_modules": [target for target, *_ in TARGETS],
        "bias": "none",
        "inference_mode": False,
        "base_model_name_or_path": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "rank_pattern": {target: rank for target, rank, *_ in TARGETS},
        "alpha_pattern": {target: alpha for target, *_, alpha in TARGETS},
    }
    (ADAPTER / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    torch.set_num_threads(1)
    initial: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, (target, rank, width, out, _) in enumerate(TARGETS):
        a = deterministic(rank * width, 31 + index, 0.5).reshape(rank, width)
        b = deterministic(out * rank, 61 + index, 0.5).reshape(out, rank)
        initial[target] = (a, b)
    write_adapter(initial)

    trajectories = {}
    for index, (target, rank, width, out, _) in enumerate(TARGETS):
        a = initial[target][0].clone().requires_grad_(True)
        b = initial[target][1].clone().requires_grad_(True)
        trajectories[target] = {"a": a, "b": b}

    optimizer = torch.optim.AdamW(
        [t for pair in trajectories.values() for t in (pair["a"], pair["b"])],
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )

    steps = []
    for step in range(STEPS):
        gradients = {}
        for index, (target, rank, width, out, _) in enumerate(TARGETS):
            grad_a = deterministic(rank * width, 101 + step * 17 + index, 1.0).reshape(
                rank, width
            )
            grad_b = deterministic(out * rank, 211 + step * 19 + index, 1.0).reshape(
                out, rank
            )
            trajectories[target]["a"].grad = grad_a
            trajectories[target]["b"].grad = grad_b
            gradients[target] = {"a": grad_a, "b": grad_b}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps.append(
            {
                "gradients": {
                    target: {
                        "a_bits": bits(grad["a"]),
                        "b_bits": bits(grad["b"]),
                    }
                    for target, grad in gradients.items()
                },
                "parameters": {
                    target: {
                        "a_bits": bits(pair["a"].detach()),
                        "b_bits": bits(pair["b"].detach()),
                    }
                    for target, pair in trajectories.items()
                },
                "moments": {
                    target: {
                        "exp_avg_a_bits": bits(optimizer.state[pair["a"]]["exp_avg"]),
                        "exp_avg_sq_a_bits": bits(optimizer.state[pair["a"]]["exp_avg_sq"]),
                        "exp_avg_b_bits": bits(optimizer.state[pair["b"]]["exp_avg"]),
                        "exp_avg_sq_b_bits": bits(optimizer.state[pair["b"]]["exp_avg_sq"]),
                    }
                    for target, pair in trajectories.items()
                },
            }
        )

    payload = {
        "seed": SEED,
        "adapter": "deepseek_v41_adapter_training/adapter",
        "tolerance": TOLERANCE,
        "comparison": "strict_less_than",
        "rtol": 0,
        "config": {
            "learning_rate": LR,
            "beta1": BETAS[0],
            "beta2": BETAS[1],
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
        },
        "modules": [
            {
                "target": target,
                "rank": rank,
                "width": width,
                "out": out,
                "alpha": alpha,
                "a_bits": bits(initial[target][0]),
                "b_bits": bits(initial[target][1]),
            }
            for target, rank, width, out, alpha in TARGETS
        ],
        "steps": steps,
    }
    previous = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None
    if previous is not None and previous != payload:
        raise SystemExit(
            "adapter-training fixture would change; refusing to overwrite pinned evidence"
        )
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"PASS V4.1 adapter-training fixture: {len(TARGETS)} modules, {STEPS} AdamW steps, "
        f"{len(payload['steps'][-1]['parameters'])} factor pairs"
    )


if __name__ == "__main__":
    main()
