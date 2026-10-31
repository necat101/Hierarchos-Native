"""Pinned CPU PyTorch gradients for the V4.1 compressor decode-state chain.

The released `Compressor` carries an incomplete group in `kv_state`/`score_state`
across decode steps. A decode step writes its projected row into slot
`start_pos % ratio` and only pools -- and only emits a latent -- when the group
just completed, so the state a step leaves behind is both the write's
destination and the next pool's source.

The reference keeps that state in `register_buffer` tensors, which detaches
autograd. The fixture therefore gates the step contract directly, and chains it
the way a trainer does. For each step, in reverse:

* the state before the step is a leaf, cloned into the module's buffer (the
  released forward writes into the clone, and a leaf cannot be mutated);
* the released forward runs unchanged, so the emitted latent and the written
  state come from the pinned code;
* the loss is that step's latent dot upstream, plus a seed dot the state the
  step left behind -- the seed is the gradient the *next* step returned, or the
  final seed for the last step.

That seed is applied to the post-write tensor, so autograd hands the leaf
exactly the adjoint's return value: the written slot is zeroed (it was
overwritten) and the untouched slot carries the seed plus its pool term. The
leaf's gradient therefore gates `grad_state_*`, and the step's input gradient
gates `grad_input`. Ratio 1 has no state and no gate, so its steps are gated on
their own latents alone.

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

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "deepseek_v41_compressor_decode_grad.json"
)
SEED = 4141227
NORM_EPS = 1e-20


def bits(value: torch.Tensor | None) -> list[int]:
    if value is None:
        return []
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


def make_case(
    reference,
    name,
    ratio,
    hidden,
    head,
    batch,
    prefill_len,
    steps,
    seed_state,
    salt,
    upstream_scale=0.3,
    seed_scale=0.4,
):
    torch.set_num_threads(1)
    args = SimpleNamespace(
        compress_ratios=[ratio],
        dim=hidden,
        head_dim=head,
        norm_eps=NORM_EPS,
        max_batch_size=batch,
    )
    model = reference(args, 0).float()
    kv_weight = deterministic((head, hidden), salt, 0.15)
    gate_weight = deterministic((head, hidden), salt + 1, 0.7) if ratio == 2 else None
    norm_weight = 1.0 + deterministic((head,), salt + 2, 0.25)
    with torch.no_grad():
        model.wkv.weight.copy_(kv_weight)
        if ratio == 2:
            model.wgate.weight.copy_(gate_weight)
        model.norm.weight.copy_(norm_weight)

    captured = []
    model.norm.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )

    prefill_input = deterministic((batch, prefill_len, hidden), salt + 3, 0.5)
    with torch.no_grad():
        model(prefill_input, 0)
    if ratio == 2:
        state_kv = model.kv_state[:batch].clone()
        state_score = model.score_state[:batch].clone()
    else:
        state_kv = state_score = None
    initial_state = (state_kv, state_score)

    # Value pass: replay the released sequence and keep every operand a caller
    # would retain for backward, including the state before each step.
    recorded = []
    for index, start_pos in enumerate(range(prefill_len, prefill_len + steps)):
        x = deterministic((batch, 1, hidden), salt + 10 + index * 7, 0.5)
        with torch.no_grad():
            if ratio == 2:
                model.kv_state = state_kv.clone()
                model.score_state = state_score.clone()
            latent = model(x, start_pos)
            pooled = None if latent is None else captured[-1]
            if ratio == 2:
                next_kv = model.kv_state
                next_score = model.score_state
        groups = 0 if latent is None else latent.size(1)
        upstream = (
            deterministic((batch, groups, head), salt + 4, upstream_scale)
            if groups
            else None
        )
        if ratio == 2 and groups:
            # The pool the step emitted must be the pool its state now holds:
            # this is the operand layout the native adjoint assumes.
            recomputed = (
                next_kv[:batch] * next_score[:batch].softmax(dim=1)
            ).sum(dim=1, keepdim=True)
            assert torch.equal(recomputed, pooled), (name, start_pos)
        recorded.append(
            {
                "start_pos": start_pos,
                "groups": groups,
                "input": x,
                "upstream": upstream,
                "before_kv": state_kv,
                "before_score": state_score,
                "state_kv": next_kv if ratio == 2 else None,
                "state_score": next_score if ratio == 2 else None,
                "pooled": pooled,
                "latent": latent,
            }
        )
        if ratio == 2:
            state_kv, state_score = next_kv, next_score

    final_seed_kv = final_seed_score = None
    if ratio == 2 and seed_state:
        final_seed_kv = deterministic((batch, ratio, head), salt + 5, seed_scale)
        final_seed_score = deterministic((batch, ratio, head), salt + 6, seed_scale)
    incoming_kv = final_seed_kv
    incoming_score = final_seed_score

    # Backward pass: one independent graph per step, chained through the seed,
    # which is exactly the contract a cached decode replay uses.
    for record in reversed(recorded):
        x = record["input"].clone().requires_grad_()
        if ratio == 2:
            leaf_kv = record["before_kv"].clone().requires_grad_()
            leaf_score = record["before_score"].clone().requires_grad_()
            model.kv_state = leaf_kv.clone()
            model.score_state = leaf_score.clone()
        else:
            leaf_kv = leaf_score = None
        latent = model(x, record["start_pos"])
        assert (latent is None) == (record["groups"] == 0)
        if record["groups"]:
            assert torch.equal(latent, record["latent"])
            loss = (latent * record["upstream"]).sum()
        else:
            loss = latent.sum() * 0.0 if latent is not None else x.sum() * 0.0
        if ratio == 2:
            state_kv_after = model.kv_state
            state_score_after = model.score_state
            if incoming_kv is not None:
                loss = loss + (state_kv_after * incoming_kv).sum()
                loss = loss + (state_score_after * incoming_score).sum()
        if not record["groups"]:
            # Keep the input gradient defined for a step that emits nothing.
            loss = loss + x.sum() * 0.0
        loss.backward()
        # A step nothing consumes has no path to the loss; its honest gradient
        # is zero, which is what the native adjoint returns for a zero seed.
        record["grad_input"] = x.grad if x.grad is not None else torch.zeros_like(x)
        if ratio == 2:
            grad_state_kv = (
                leaf_kv.grad if leaf_kv.grad is not None else torch.zeros_like(leaf_kv)
            )
            grad_state_score = (
                leaf_score.grad
                if leaf_score.grad is not None
                else torch.zeros_like(leaf_score)
            )
            record["grad_state_kv"] = grad_state_kv
            record["grad_state_score"] = grad_state_score
            incoming_kv = grad_state_kv
            incoming_score = grad_state_score

    steps_payload = []
    for record in recorded:
        steps_payload.append(
            {
                "start_pos": record["start_pos"],
                "groups": record["groups"],
                "input_bits": bits(record["input"]),
                "projected_kv_bits": bits(F.linear(record["input"], model.wkv.weight.detach())),
                "projected_gate_bits": (
                    bits(F.linear(record["input"], model.wgate.weight.detach()))
                    if ratio == 2
                    else []
                ),
                "pooled_bits": bits(record["pooled"]),
                "latent_bits": bits(record["latent"]),
                "upstream_bits": bits(record["upstream"]),
                "state_kv_bits": bits(record["state_kv"]),
                "state_score_bits": bits(record["state_score"]),
                "expected": {
                    "grad_input_bits": bits(record["grad_input"]),
                    "grad_state_kv_bits": bits(record.get("grad_state_kv")),
                    "grad_state_score_bits": bits(record.get("grad_state_score")),
                },
            }
        )

    return {
        "name": name,
        "ratio": ratio,
        "hidden": hidden,
        "head": head,
        "batch": batch,
        "prefill": prefill_len,
        "kv_weight_bits": bits(kv_weight),
        "gate_weight_bits": bits(gate_weight),
        "norm_weight_bits": bits(norm_weight),
        "prefill_input_bits": bits(prefill_input),
        "state0_kv_bits": bits(initial_state[0]),
        "state0_score_bits": bits(initial_state[1]),
        "final_seed_kv_bits": bits(final_seed_kv),
        "final_seed_score_bits": bits(final_seed_score),
        "steps": steps_payload,
    }


# name, ratio, hidden, head, batch, prefill, decode steps, final seed, upstream scale
#
# `prefill` is the number of tokens before the first decode step; decode steps
# follow one per position, so `(prefill + t + 1) % ratio == 0` marks the steps
# that complete a group and emit a latent. Head dims stay at or below the
# 128-wide regime where the shared input-gradient reduction order is pinned
# (the prefill fixture records the 512-wide limit).
CASES = [
    ("ratio-2-single-completing-step", 2, 8, 4, 1, 1, 1, True, 0.3),
    ("ratio-2-complete-then-pending", 2, 16, 8, 1, 1, 3, True, 0.3),
    ("ratio-2-full-state-pending", 2, 32, 16, 2, 2, 4, True, 0.3),
    ("ratio-2-two-cycles", 2, 64, 32, 2, 1, 6, True, 0.3),
    ("ratio-2-single-pending-step", 2, 32, 8, 1, 2, 1, True, 0.3),
    ("ratio-2-unseeded-tail", 2, 128, 64, 1, 1, 2, False, 0.3),
    ("ratio-2-wide-hidden", 2, 512, 128, 1, 1, 2, True, 0.1),
    ("ratio-2-long-chain", 2, 16, 8, 1, 1, 8, True, 0.3),
    ("ratio-1-three-steps", 1, 16, 8, 1, 1, 3, False, 0.3),
    ("ratio-1-batch-two", 1, 64, 64, 2, 2, 2, False, 0.3),
]


def make_fixture(reference):
    cases = []
    for index, (
        name,
        ratio,
        hidden,
        head,
        batch,
        prefill,
        steps,
        seeded,
        scale,
    ) in enumerate(CASES):
        cases.append(
            make_case(
                reference,
                name,
                ratio,
                hidden,
                head,
                batch,
                prefill,
                steps,
                seeded,
                5000 + index * 32,
                scale,
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
        "state_note": (
            "Each step is a separate graph: the state before it is a leaf, cloned into the "
            "module's buffer, and the seed (the next step's returned gradient) is applied to "
            "the post-write state. The leaf gradient is the adjoint return value."
        ),
        "reduction_note": (
            "Input-gradient GEMM reduction order: identical to ATen for head dims up to 128, "
            "which is why these cases stay at or below 128; the 512-wide head remains the "
            "documented prefill-fixture limit."
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    payload = make_fixture(load_reference(args.reference_path))
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    elif not FIXTURE.exists() or json.loads(FIXTURE.read_text(encoding="utf-8")) != payload:
        raise SystemExit("decode-gradient fixture changed; refusing to overwrite pinned evidence")
    gradients = sum(
        sum(
            len(step["expected"]["grad_input_bits"])
            + len(step["expected"]["grad_state_kv_bits"])
            + len(step["expected"]["grad_state_score_bits"])
            for step in case["steps"]
        )
        for case in payload["cases"]
    )
    steps = sum(len(case["steps"]) for case in payload["cases"])
    print(
        f"PASS: {len(payload['cases'])} FP32 compressor decode chains, "
        f"{steps} steps, {gradients} frozen gradients; {REVISION}"
    )


if __name__ == "__main__":
    main()
