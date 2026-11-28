#!/usr/bin/env python3
"""Phase XIX in-graph comparison: promoted arm against scratch-reuse arm.

Both arms run the full production path (`graph.train_step`), so every step's loss
is the loss of the model the previous step produced. The only difference is the
SPIR-V behind the promoted pass-1 identity.

Checks, per step:
  * `loss_bits` identical,
  * `grad_digest` identical (LoRA),
  * `param_digest` identical,
and reports the end-of-run parameter digest, the production dispatch counters
each arm reported, and the arm/module provenance each process printed.

Usage: compare_in_graph.py <arm-a-log> <arm-b-log>
"""
from __future__ import annotations

import sys
from pathlib import Path


def steps(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xix-step]" not in line:
            continue
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        out.append(fields)
    return out


def end_line(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xix-step-run-end]" in line:
            return dict(token.split("=", 1) for token in line.split() if "=" in token)
    raise SystemExit(f"{path.name}: no [phase-xix-step-run-end] line")


def seam(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xix-step-seam]" in line:
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            return f"{fields.get('arm')} module={fields.get('module')} sha256_prefix={fields.get('sha256_prefix')}"
    return "unknown"


def main() -> None:
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = steps(a_path), steps(b_path)
    if len(a) != len(b):
        raise SystemExit(f"step count differs: {len(a)} vs {len(b)}")
    mismatched = []
    compared = 0
    for index, (left, right) in enumerate(zip(a, b)):
        for key in ("loss_bits", "grad_digest", "param_digest"):
            if left.get(key) is None and right.get(key) is None:
                continue
            compared += 1
            if left.get(key) != right.get(key):
                mismatched.append(
                    (index, key, left.get(key), right.get(key))
                )
    end_a, end_b = end_line(a_path), end_line(b_path)
    same_end = end_a.get("parameter_end_digest") == end_b.get("parameter_end_digest")
    losses_a = [row.get("loss_bits") for row in a]
    losses_b = [row.get("loss_bits") for row in b]
    summary = {
        "series": f"{seam(a_path)}  vs  {seam(b_path)}",
        "steps": len(a),
        "comparisons": compared,
        "mismatches": len(mismatched),
        "first_mismatches": mismatched[:8],
        "loss_bits_equal": losses_a == losses_b,
        "loss_trajectory": [row.get("loss") for row in a],
        "parameter_end_digest": {
            "A": end_a.get("parameter_end_digest"),
            "B": end_b.get("parameter_end_digest"),
            "equal": same_end,
        },
        "counters": {
            "A": {k: v for k, v in end_a.items() if k.startswith("backward_") or k.startswith("forward_")},
            "B": {k: v for k, v in end_b.items() if k.startswith("backward_") or k.startswith("forward_")},
        },
        "verdict": "PASS" if not mismatched and same_end and losses_a == losses_b else "FAIL",
    }
    print(f"[phase-xix-in-graph] series={summary['series']} steps={summary['steps']} "
          f"comparisons={summary['comparisons']} mismatches={summary['mismatches']} "
          f"loss_bits_equal={summary['loss_bits_equal']} "
          f"param_end_a={summary['parameter_end_digest']['A']} "
          f"param_end_b={summary['parameter_end_digest']['B']} "
          f"verdict={summary['verdict']}")
    print(f"[phase-xix-in-graph] loss_trajectory={summary['loss_trajectory']}")
    print(f"[phase-xix-in-graph] counters_A={summary['counters']['A']}")
    print(f"[phase-xix-in-graph] counters_B={summary['counters']['B']}")
    for row in mismatched[:8]:
        print(f"[phase-xix-in-graph] mismatch step={row[0]} key={row[1]} A={row[2]} B={row[3]}")
    if summary["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
