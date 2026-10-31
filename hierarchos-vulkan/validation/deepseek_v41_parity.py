"""Fixed absolute-only acceptance gate for future V4.1 model validators.

Passing component tests or this comparator on synthetic tensors does not qualify
the model. Full prefill and cached-decode logits must come from the independent
pinned DeepSeek reference and the native graph with identical inputs/weights.
"""

from __future__ import annotations

import torch


LOGIT_DRIFT_LIMIT = 2.0e-7
LOGIT_RTOL = 0.0


def require_logit_parity(native: torch.Tensor, reference: torch.Tensor, *, label: str) -> float:
    """Compare every logit; the strict ceiling has no caller-controlled override."""
    if native.shape != reference.shape or native.numel() == 0:
        raise AssertionError(f"{label}: logits must have identical, nonempty shapes")
    if not native.is_floating_point() or not reference.is_floating_point():
        raise AssertionError(f"{label}: logits must be floating point")
    if not torch.isfinite(native).all().item() or not torch.isfinite(reference).all().item():
        raise AssertionError(f"{label}: non-finite logits cannot pass parity")
    # Widen before subtraction so FP32 rounding cannot hide a boundary failure.
    difference = (native.detach().to(device="cpu", dtype=torch.float64)
                  - reference.detach().to(device="cpu", dtype=torch.float64)).abs()
    worst = difference.max().item()
    if not worst < LOGIT_DRIFT_LIMIT:
        raise AssertionError(f"{label}: max absolute logit drift {worst:.17g} must be < {LOGIT_DRIFT_LIMIT:g} (rtol=0)")
    return worst
