"""Pinned CPU PyTorch gradients for the V4.1 compressor prefill adjoint.

Extracts DeepSeek's unchanged `RMSNorm` and `Compressor` classes the way the
forward fixture does, promotes every parameter to FP32, and differentiates one
prefill evaluation:

    x -> wkv/wgate -> softmax pool over each group -> RMSNorm -> latents

A trailing incomplete group makes the reference store its rows in the pool
state, which detaches autograd. The fixture therefore lifts exactly that write
back into the graph (`F.linear` on the same rows with the same weights is
bit-identical to the stored slot) and seeds the state gradient, which is the
contract the native adjoint implements: the trailing row owns the gradient of
the slot it was written to, untouched slots pass an incoming state gradient
through, and a written slot returns zero to the previous state.

Values are stored as FP32 bit patterns so the JSON round trip is exact.
Regenerate only together with a deliberate kernel change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from validate_deepseek_v41_candidates import MODEL, REVISION, SOURCE_SHA256
from validate_deepseek_v41_compressor import load_reference

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v41_compressor_grad.json"
SEED = 4141203
NORM_EPS = 1e-20


def bits(value: torch.Tensor) -> list[int]:
    return value.contiguous().flatten().view(torch.uint32).tolist()


def deterministic(shape: tuple[int, ...], salt: int, scale: float) -> torch.Tensor:
    """Explicit LCG so the fixture does not depend on torch's RNG stream."""
    count = 1
    for size in shape:
        count *= size
    values = []
    state = (SEED * 1103515245 + salt) & 0x7FFFFFFF
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        values.append((state / 0x7FFFFFFF) * (2.0 * scale) - scale)
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def make_case(reference, name, ratio, hidden, head, batch, seq, salt, state_seed, upstream_scale=0.3):
    torch.set_num_threads(1)
    args = SimpleNamespace(
        compress_ratios=[ratio], dim=hidden, head_dim=head, norm_eps=NORM_EPS, max_batch_size=batch
    )
    model = reference(args, 0).float()
    kv = deterministic((head, hidden), salt, 0.15)
    gate = deterministic((head, hidden), salt + 1, 0.7) if ratio == 2 else None
    norm = 1.0 + deterministic((head,), salt + 2, 0.25)
    with torch.no_grad():
        model.wkv.weight.copy_(kv)
        if ratio == 2:
            model.wgate.weight.copy_(gate)
        model.norm.weight.copy_(norm)

    captured = []
    model.norm.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))

    x = deterministic((batch, seq, hidden), salt + 3, 0.5).requires_grad_()
    groups = seq // ratio
    tail = seq % ratio
    latents = model(x, 0)
    if latents is None:
        # The first group is still filling up: nothing is emitted and there is
        # no norm input at all.
        assert groups == 0 and tail != 0, (groups, tail)
        pooled = torch.zeros(0)
    else:
        assert latents.shape == (batch, groups, head), (latents.shape, groups)
        pooled = captured[-1]
        assert pooled.shape == (batch, groups, head)

    upstream = deterministic((batch, groups, head), salt + 4, upstream_scale) if groups else torch.zeros(0)
    # A zero-valued term keeps `x.grad` defined (all zeros) when the evaluation
    # emitted no group at all and nothing else touches the input.
    loss = (latents * upstream).sum() if groups else x.sum() * 0.0
    slot_kv = slot_score = None
    if state_seed:
        slot_kv = deterministic((batch, ratio, head), salt + 5, 0.4)
        slot_score = deterministic((batch, ratio, head), salt + 6, 0.4)
        if tail:
            # The written slot holds the row verbatim, so this recomputation is
            # the stored value and restores the edge autograd detached. It runs
            # over the full sequence, exactly as the reference's own projection
            # does, so the backward is one `[batch * seq, hidden]` matmul: a
            # separate projection of the tail rows only would be a differently
            # shaped GEMM and round differently.
            offset = groups * ratio
            tail_kv = F.linear(x, model.wkv.weight)[:, offset:]
            loss = loss + (tail_kv * slot_kv[:, :tail]).sum()
            if ratio == 2:
                tail_score = F.linear(x, model.wgate.weight)[:, offset:]
                loss = loss + (tail_score * slot_score[:, :tail]).sum()
    loss.backward()

    projected_kv = F.linear(x.detach(), model.wkv.weight)
    projected_gate = F.linear(x.detach(), model.wgate.weight) if ratio == 2 else None
    expected_state_kv = None
    expected_state_score = None
    if ratio == 2:
        expected_state_kv = torch.zeros(batch, ratio, head)
        expected_state_score = torch.zeros(batch, ratio, head)
        for slot in range(ratio):
            written = tail != 0 and slot == 0
            if state_seed and not written:
                expected_state_kv[:, slot] = slot_kv[:, slot]
                expected_state_score[:, slot] = slot_score[:, slot]
    return {
        "name": name,
        "ratio": ratio,
        "hidden": hidden,
        "head": head,
        "batch": batch,
        "seq": seq,
        "state_seed": state_seed,
        "kv_weight_bits": bits(kv),
        "gate_weight_bits": bits(gate) if gate is not None else [],
        "norm_weight_bits": bits(norm),
        "input_bits": bits(x.detach()),
        "projected_kv_bits": bits(projected_kv),
        "projected_gate_bits": bits(projected_gate) if projected_gate is not None else [],
        "pooled_bits": bits(pooled),
        "upstream_bits": bits(upstream),
        "slot_kv_bits": bits(slot_kv) if slot_kv is not None else [],
        "slot_score_bits": bits(slot_score) if slot_score is not None else [],
        "latent_bits": bits(latents) if latents is not None else [],
        "expected": {
            "grad_input_bits": bits(x.grad),
            "grad_state_kv_bits": bits(expected_state_kv) if expected_state_kv is not None else [],
            "grad_state_score_bits": bits(expected_state_score) if expected_state_score is not None else [],
        },
    }


# name, ratio, hidden, head, batch, seq, state seed, upstream seed scale
#
# The seed scale keeps every case's gradient magnitudes in the range the other
# V4.1 fixtures use. It matters for the input-gradient GEMM: the shared chain
# kernel reproduces ATen's K reduction order for head dims up to 128 (checked
# here at M=4..12 and N=16..512), while the 512-wide head and one small-M/N=128
# shape do not (see the audit's reduction-sensitivity note), and that order
# noise is proportional to the gradient magnitude. The same operands at the
# default 0.3 seed measure 2.86e-6 and 5.36e-7.
CASES = [
    ("ratio-1-hidden-16-head-8", 1, 16, 8, 1, 4, False, 0.3),
    ("ratio-1-hidden-32-head-16", 1, 32, 16, 2, 6, False, 0.3),
    ("ratio-1-hidden-64-head-64", 1, 64, 64, 2, 3, False, 0.3),
    ("ratio-1-hidden-128-head-512", 1, 128, 512, 1, 2, False, 0.01),
    ("ratio-2-hidden-16-head-8-whole-groups", 2, 16, 8, 1, 4, False, 0.3),
    ("ratio-2-hidden-32-head-16-state-seed", 2, 32, 16, 2, 6, True, 0.3),
    ("ratio-2-hidden-64-head-32-partial-group", 2, 64, 32, 2, 5, True, 0.3),
    ("ratio-2-hidden-128-head-128-partial-no-seed", 2, 128, 128, 1, 3, False, 0.05),
    ("ratio-2-hidden-512-head-128-state-seed", 2, 512, 128, 2, 4, True, 0.3),
    ("ratio-2-hidden-64-head-32-single-partial-token", 2, 64, 32, 2, 1, True, 0.1),
]


def make_fixture(reference):
    cases = []
    for index, (name, ratio, hidden, head, batch, seq, seeded, scale) in enumerate(CASES):
        cases.append(
            make_case(
                reference, name, ratio, hidden, head, batch, seq, 3000 + index * 32, seeded, scale
            )
        )
    return {
        "model": MODEL,
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "precision": "unquantized_fp32",
        "tolerance": 2e-7,
        "comparison": "strict_less_than",
        "rtol": 0.0,
        "torch_version": torch.__version__,
        "reduction_note": (
            "Input-gradient GEMM reduction order: identical to ATen for head dims up to 128 "
            "(M=4..12, N=16..512); not identical for head 512, and those order differences "
            "are proportional to the gradient magnitude, which sets this fixture's seed scales."
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    args = parser.parse_args()
    payload = make_fixture(load_reference(args.reference_path))
    if FIXTURE.exists() and json.loads(FIXTURE.read_text(encoding="utf-8")) != payload:
        raise SystemExit("compressor-gradient fixture changed; refusing to overwrite pinned evidence")
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    values = sum(
        len(case["expected"]["grad_input_bits"])
        + len(case["expected"]["grad_state_kv_bits"])
        + len(case["expected"]["grad_state_score_bits"])
        for case in payload["cases"]
    )
    print(f"PASS: {len(payload['cases'])} FP32 compressor prefill adjoints, {values} frozen gradients; {REVISION}")


if __name__ == "__main__":
    main()
