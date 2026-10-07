"""Pin minimax_m2's per-expert MoE accumulation topology from a native trace.

The reference `MiniMaxM2SparseMoeBlock` calls each expert with only the tokens
routed to it, and PyTorch's CPU GEMM chooses its accumulation topology from that
call's row count `M`.  The confirmed rule for these shapes (exhaustive FP32
round-to-nearest survey against `torch.nn.functional.linear`, `M` 1..4, K=16/32,
N=16/32) is:

  M == 1 : eight strided FMA lanes, reduced by cross-halves.
  M == 2 : two strided lanes whose products are materialized in FP32.
  M == 3 : two strided FMA lanes, joined by one add.
  M >= 4 : one serial FMA chain.

`diagnose_minimax_m2_moe_row_count.py` replays the whole block at the reference
row counts; this script goes one level finer and identifies, *per expert*, which
topology the native trace actually used.  It caches each candidate topology's
projection for each expert's routed rows, then sweeps every per-expert
assignment and reports the ones that reproduce the trace bit-for-bit, next to
the assignment the rule predicts.  A module that silently contracts or splits
its FMA ext-instructions (the Intel Gen9 hazard recorded in
`transformer_moe_expert_forward.comp` and `linear_forward_lane2.comp`) shows up
here as a bit-exact assignment that is not the predicted one.

It only reads fixtures and a native trace; it never mutates either, and it is a
diagnostic rather than a gate, so it always exits 0.

Usage (a trace is auto-discovered when omitted; each `lora` run rotates the
fixture directory into a `.prior-*` sibling, so older traces are searched too):

  .\\.venv-vulkan\\Scripts\\python.exe validation/diagnose_minimax_m2_moe_expert_topology.py
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / ".peft-oracle-fixtures" / "minimax_m2-strict"

# lane count and accumulate kind: 'f' fuses each product into its lane, 'm'
# materializes the product in FP32 first.
SCHEMES = {}
for _lanes in (1, 2, 4, 8):
    for _kind in ("f", "m"):
        SCHEMES[f"l{_lanes}{_kind}"] = (_lanes, _kind)


def rn32(value: Fraction) -> float:
    """Round an exact fraction to the nearest FP32 (ties to even)."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    while Fraction(2) ** exponent > value:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= value:
        exponent += 1
    scaled = value / Fraction(2) ** (exponent - 23)
    mantissa = scaled.numerator // scaled.denominator
    remainder = scaled - mantissa
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and mantissa % 2):
        mantissa += 1
    if mantissa == 1 << 24:
        mantissa >>= 1
        exponent += 1
    return sign * float(Fraction(mantissa) * Fraction(2) ** (exponent - 23))


def _fs(value) -> Fraction:
    return Fraction(float(np.float32(value)))


def mul(a, b):
    return rn32(_fs(a) * _fs(b))


def add(a, b):
    return rn32(_fs(a) + _fs(b))


def fma(a, b, c):
    return rn32(_fs(a) * _fs(b) + _fs(c))


def fold_half(values):
    values = list(values)
    span = len(values) // 2
    while span > 0:
        for index in range(span):
            values[index] = add(values[index], values[index + span])
        span //= 2
    return values[0]


def fold_seq(values):
    total = values[0]
    for value in values[1:]:
        total = add(total, value)
    return total


def dot(x, weight, lanes, kind):
    accumulators = [0.0] * lanes
    for index in range(len(x)):
        lane = index & (lanes - 1) if lanes in (2, 4, 8) else 0
        if kind == "f":
            accumulators[lane] = fma(x[index], weight[index], accumulators[lane])
        else:
            accumulators[lane] = add(accumulators[lane], mul(x[index], weight[index]))
    return fold_half(accumulators) if lanes == 8 else fold_seq(accumulators)


def linear(x, weight, scheme):
    lanes, kind = SCHEMES[scheme]
    rows, _ = weight.shape
    return np.array(
        [
            [dot(x[row].tolist(), weight[col].tolist(), lanes, kind) for col in range(rows)]
            for row in range(len(x))
        ],
        np.float32,
    )


def reference_topology(routed: int) -> str:
    if routed <= 1:
        return "l8f"
    if routed == 2:
        return "l2m"
    if routed == 3:
        return "l2f"
    return "l1f"


def newest_trace() -> Path:
    """Newest `gradient-probe-*/trace.json` in the fixture or its rotations."""
    candidates = []
    for root in sorted(ROOT.joinpath(".peft-oracle-fixtures").glob("minimax_m2-strict*")):
        candidates.extend(root.glob("gradient-probe-*/trace.json"))
    if not candidates:
        raise SystemExit(
            "no gradient-probe trace found under .peft-oracle-fixtures/minimax_m2-strict*; "
            "run validation/diagnose_peft_saved_gradient.py first or pass --trace"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", nargs="?", type=int, default=0)
    parser.add_argument("--trace", default=None, help="native trace.json to compare against")
    parser.add_argument("--adapter", default="base", help="checkpoint subdirectory of the fixture")
    parser.add_argument("--output", default=None, help="write the comparison JSON here")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.trace is None:
        args.trace = newest_trace()
        print(f"using newest trace {args.trace}")

    prefix = f"layers.{args.layer}.mlp"
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    if f"{prefix}_norm_output" not in trace or f"{prefix}_output" not in trace:
        raise SystemExit(f"{args.trace} has no {prefix}_norm_output/{prefix}_output boundary")

    config = json.loads((FIXTURE / "base" / "config.json").read_text(encoding="utf-8"))
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    experts = config["num_local_experts"]
    top_k = config["num_experts_per_tok"]

    x = torch.tensor(
        [float(value) for value in trace[f"{prefix}_norm_output"]], dtype=torch.float32
    ).reshape(-1, hidden)
    native = torch.tensor(
        [float(value) for value in trace[f"{prefix}_output"]], dtype=torch.float32
    ).reshape(-1, hidden)
    checkpoint = load_file(FIXTURE / args.adapter / "model.safetensors")
    weights = {
        "gate": checkpoint[f"model.layers.{args.layer}.mlp.gate.weight"].float(),
        "up": checkpoint[f"model.layers.{args.layer}.mlp.experts.gate_up_proj"].float(),
        "down": checkpoint[f"model.layers.{args.layer}.mlp.experts.down_proj"].float(),
        "bias": checkpoint[f"model.layers.{args.layer}.mlp.e_score_correction_bias"].float(),
    }

    # Reference router: sigmoid scores, top-k on the biased scores, renormalized
    # selected weights.  Unselected experts keep exact zeros.
    logits = F.linear(x, weights["gate"])
    routing_weights = torch.sigmoid(logits.float())
    _, selected = torch.topk(routing_weights + weights["bias"], top_k, dim=-1, sorted=False)
    selected_weights = routing_weights.gather(1, selected)
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)

    subsets = []
    routed_counts = []
    for index in range(experts):
        rows, _ = torch.where(selected == index)
        rows = rows.numpy()
        routed_counts.append(int(rows.size))
        cache = {}
        if rows.size:
            states = x.numpy()[rows]
            for scheme in SCHEMES:
                gate = linear(states, weights["up"][index][:intermediate].numpy(), scheme)
                up = linear(states, weights["up"][index][intermediate:].numpy(), scheme)
                activation = (F.silu(torch.tensor(gate)) * torch.tensor(up)).numpy()
                cache[scheme] = linear(activation, weights["down"][index].numpy(), scheme)
        subsets.append((rows, cache))

    # An expert the router sent no tokens to is skipped by the reference and its
    # zero routing weight removes it from the block output, so no topology is
    # observable for it; the sweep runs over the experts that were routed to.
    active = [index for index in range(experts) if routed_counts[index] > 0]
    skipped = [index for index in range(experts) if routed_counts[index] == 0]
    predicted = tuple(reference_topology(routed_counts[index]) for index in active)

    def replay(assignment) -> np.ndarray:
        replay_weights = selected_weights.numpy()
        output = np.zeros((x.shape[0], hidden), np.float32)
        for position, index in enumerate(active):
            rows, cache = subsets[index]
            _, slot = torch.where(selected == index)
            contribution = (
                cache[assignment[position]] * replay_weights[rows, slot.numpy()][:, None]
            )
            output[rows] += contribution.astype(np.float32)
        return output

    predicted_value = replay(predicted)
    predicted_bit_exact = int((predicted_value != native.numpy()).sum())

    matches = []
    for assignment in product(SCHEMES, repeat=len(active)):
        if int((replay(assignment) != native.numpy()).sum()) == 0:
            matches.append(list(assignment))

    report = {
        "family": "minimax_m2",
        "layer": args.layer,
        "trace": str(args.trace),
        "rows": int(x.shape[0]),
        "hidden": hidden,
        "experts": experts,
        "top_k": top_k,
        "routed_tokens_per_expert": routed_counts,
        "skipped_experts": skipped,
        "swept_experts": active,
        "reference_topology": dict(zip(active, predicted)),
        "reference_expert_gemm_rows": {index: routed_counts[index] for index in active},
        "reference_bit_exact": predicted_bit_exact == 0,
        "reference_differing_entries": predicted_bit_exact,
        "reference_max_abs": float(np.abs(predicted_value - native.numpy()).max()),
        "bit_exact_topologies": [dict(zip(active, match)) for match in matches],
        "topology_matches_reference": [list(predicted)] == matches,
    }

    print(
        f"layers.{args.layer}.mlp: rows={x.shape[0]} hidden={hidden} experts={experts} top_k={top_k}"
    )
    print(f"  routed tokens per expert: {routed_counts}")
    if skipped:
        print(f"  experts the router skipped (no observable topology): {skipped}")
    print(
        f"  reference topology (experts {active}) {list(predicted)} on M="
        f"{[routed_counts[index] for index in active]}: "
        f"differing={predicted_bit_exact}/{native.numel()} "
        f"max_abs={report['reference_max_abs']:.3e} bit_exact={report['reference_bit_exact']}"
    )
    if matches:
        for match in matches:
            print(f"  bit-exact per-expert topology: {match}")
    else:
        print("  bit-exact per-expert topology: NONE")
    print(
        "  conclusion: the trace "
        + ("agrees with" if report["topology_matches_reference"] else "DISAGREES with")
        + " the reference's per-expert topology"
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
