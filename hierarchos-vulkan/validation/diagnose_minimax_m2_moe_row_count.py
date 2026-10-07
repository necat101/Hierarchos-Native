"""Quantify minimax_m2's layer-0 MoE residual as a per-expert row-count effect.

The reference `MiniMaxM2SparseMoeBlock` calls each expert with only the tokens
routed to it, so its `F.linear` GEMM runs at `M` = the number of tokens that
picked that expert (2, 3, 3 for the strict fixture's four tokens and top-2 of
three experts).  PyTorch's CPU GEMM is `M`-dependent, so that call shape picks a
different accumulation topology than a dense one.  The native Vulkan graph
still dispatches every expert over all `rows` (M = 4) and zero-weights the rows
the expert was not routed to, but its expert projection counts the routed rows
on the device (shaders/transformer_moe_expert_count.comp) and selects the
reference topology for that count, so the dense dispatch is bit-identical to
the routed call.

This diagnostic replays the fixture's own weights through three call shapes and
reports which of them the native trace agrees with bit-for-bit:

  subset - HF's own shape: one expert GEMM per expert on its routed rows.
  dense  - the shape the graph used before the expert projection learned the
           per-expert row count: every expert over all rows, zero-weighted.
  mixed  - the dense GEMM shape with HF's own per-expert scatter/accumulate,
           which separates the two candidate causes.

It only reads fixtures and a native trace; it never mutates either.  The point
is documentation and regression *visibility*: the graph reproduces the routed
subset shape bit-for-bit, and a regression to the dense shape shows up here as
a non-zero `subset` delta long before it could reach the 2e-7 PEFT ceiling.

Usage (a trace is auto-discovered when omitted; each `lora` run rotates the
fixture directory into a `.prior-*` sibling, so older traces are searched too):

  .\\.venv-vulkan\\Scripts\\python.exe validation/diagnose_minimax_m2_moe_row_count.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / ".peft-oracle-fixtures" / "minimax_m2-strict"
FAMILY = "minimax_m2"


def _reference_routing(weights, x):
    """Reference router: sigmoid scores, top-k choice, renormalized weights."""
    top_k = 2
    logits = F.linear(x, weights["gate"])
    routing_weights = torch.sigmoid(logits.float())
    scores = routing_weights + weights["bias"]
    _, selected = torch.topk(scores, top_k, dim=-1, sorted=False)
    selected_weights = routing_weights.gather(1, selected)
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
    return routing_weights, selected, selected_weights


def _expert(weights, index, states):
    gate, up = F.linear(states, weights["up"][index]).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, weights["down"][index])


def replay(weights, x, routing, shape):
    """Expert call `M` per expert, plus the replayed block output."""
    _, selected, selected_weights = routing
    experts = weights["up"].shape[0]
    output = torch.zeros_like(x)
    counts = []
    call_rows = []
    for index in range(experts):
        rows, slots = torch.where(selected == index)
        counts.append(int(rows.numel()))
        call_rows.append(int(rows.numel()) if shape == "subset" else int(x.shape[0]))
        if shape == "subset":
            if rows.numel() == 0:
                continue
            contribution = _expert(weights, index, x[rows])
            output.index_add_(0, rows, contribution * selected_weights[rows, slots, None])
            continue
        full = _expert(weights, index, x)
        if shape == "dense":
            weight = torch.where(
                selected == index, selected_weights, torch.zeros_like(selected_weights)
            ).sum(dim=1, keepdim=True)
            output = output + full * weight
        else:
            output.index_add_(
                0, rows, full[rows] * selected_weights[rows, slots, None]
            )
    return output, counts, call_rows


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

    hidden = 16
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

    routing = _reference_routing(weights, x)
    report = {
        "family": FAMILY,
        "layer": args.layer,
        "rows": int(x.shape[0]),
        "routed_experts_per_row": int(routing[1].shape[1]),
        "trace": str(args.trace),
        "replays": {},
    }
    print(f"{prefix}: rows={x.shape[0]} hidden={x.shape[1]}")
    for shape in ("subset", "dense", "mixed"):
        value, counts, call_rows = replay(weights, x, routing, shape)
        delta = float((value - native).abs().max())
        differing = int((value != native).sum())
        report["replays"][shape] = {
            "routed_tokens_per_expert": counts,
            "expert_gemm_rows": call_rows,
            "native_max_abs": delta,
            "native_differing_entries": differing,
            "native_bit_exact": differing == 0,
        }
        print(
            f"  {shape:6s} expert GEMM M={call_rows} (routed {counts}) "
            f"native_max_abs={delta:.3e} differing={differing}/{native.numel()} "
            f"bit_exact={differing == 0}"
        )
    if report["replays"]["subset"]["native_bit_exact"]:
        print(
            "  conclusion: the native graph reproduces the reference's routed-row "
            "expert GEMM bit-for-bit"
        )
    else:
        print(
            "  conclusion: the native graph does NOT reproduce the routed-row "
            "expert GEMM; the expert projection is falling back to a dense shape"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
