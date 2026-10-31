"""Pinned CPU PyTorch fixture for the V4.1 terminal training adjoints.

The V4.1 block's backward pass begins at the loss. This fixture pins the stages
the native graph can already reach, in forward order:

    x --rms_norm--> hidden --head--> logits --cross_entropy--> loss

and, in reverse, `d loss / d logits`, `d loss / d hidden` (the frozen head's
input gradient) and `d loss / d x` (the final RMSNorm's input gradient). The
block adjoints below the norm are separate passes and are not covered here.

Values are stored as FP32 bit patterns, not decimal, so the fixture is exact
across JSON round trips. Regenerate only together with a deliberate kernel
change:

    .\\.venv-vulkan\\Scripts\\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_terminal_grad.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 4141097
ROWS = 4
HIDDEN = 32
VOCAB = 64
TARGETS = [3, 17, 41, 0]
NORM_EPS = 1e-6
TOLERANCE = 2e-7
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v41_terminal_grad.json"


def bits(value: torch.Tensor) -> list[int]:
    return value.contiguous().flatten().view(torch.uint32).tolist()


def deterministic(shape: tuple[int, ...], salt: int) -> torch.Tensor:
    """Explicit LCG so the fixture does not depend on torch's RNG stream."""
    count = 1
    for size in shape:
        count *= size
    values = []
    state = (SEED * 1103515245 + salt) & 0x7FFFFFFF
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        # [-2, 2) with enough low bits to exercise every rounding path.
        values.append((state / 0x7FFFFFFF) * 4.0 - 2.0)
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def main() -> None:
    torch.set_num_threads(1)
    x = deterministic((ROWS, HIDDEN), 3).requires_grad_(True)
    norm_weight = deterministic((HIDDEN,), 5)
    head = deterministic((VOCAB, HIDDEN), 11)

    variance = x.pow(2).mean(-1, keepdim=True) + NORM_EPS
    rstd = torch.rsqrt(variance)
    hidden = (x * rstd) * norm_weight
    logits = F.linear(hidden, head)
    targets = torch.tensor(TARGETS, dtype=torch.long)

    row_loss = F.cross_entropy(logits, targets, reduction="none")
    loss = row_loss.mean()
    grad_logits, grad_hidden, grad_x = torch.autograd.grad(loss, (logits, hidden, x))

    payload = {
        "seed": SEED,
        "rows": ROWS,
        "hidden": HIDDEN,
        "vocab": VOCAB,
        "targets": TARGETS,
        "norm_eps": NORM_EPS,
        "tolerance": TOLERANCE,
        "comparison": "strict_less_than",
        "rtol": 0,
        "x_bits": bits(x.detach()),
        "norm_weight_bits": bits(norm_weight),
        "head_bits": bits(head),
        "hidden_bits": bits(hidden.detach()),
        "logits_bits": bits(logits.detach()),
        "row_loss_bits": bits(row_loss),
        "loss_bits": bits(loss.detach().reshape(1)),
        "grad_logits_bits": bits(grad_logits),
        "grad_hidden_bits": bits(grad_hidden),
        "grad_x_bits": bits(grad_x),
    }
    previous = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None
    if previous is not None and previous != payload:
        raise SystemExit(
            "terminal-gradient fixture would change; refusing to overwrite pinned evidence"
        )
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"PASS V4.1 terminal-gradient fixture: rows={ROWS} hidden={HIDDEN} vocab={VOCAB} "
        f"loss={payload['loss_bits']} "
        f"grad_logits={len(payload['grad_logits_bits'])} "
        f"grad_hidden={len(payload['grad_hidden_bits'])} "
        f"grad_x={len(payload['grad_x_bits'])}"
    )


if __name__ == "__main__":
    main()
