#!/usr/bin/env python3
"""Compare two Phase XVIII step logs step by step.

The shader promotion is a pure implementation change, so the two arms must
produce the same `loss_bits` on every step, the same parameter digest and (under
LoRA) the same gradient digest. This script reports that equality rather than
recomputing any loss: the harness prints the bits the training step returned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def steps(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xviii-step]" not in line:
            continue
        rows.append(dict(token.split("=", 1) for token in line.split() if "=" in token))
    return rows


def end(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xviii-step-run-end]" in line:
            return dict(token.split("=", 1) for token in line.split() if "=" in token)
    return {}


def compare(label: str, a_path: Path, b_path: Path) -> dict:
    a, b = steps(a_path), steps(b_path)
    if len(a) != len(b):
        raise SystemExit(f"{label}: {len(a)} vs {len(b)} steps")
    mismatches = []
    graded = 0
    for left, right in zip(a, b):
        if "grad_digest" in left and "grad_digest" in right:
            graded += 1
            if left["grad_digest"] != right["grad_digest"]:
                mismatches.append(("grad_digest", left["step"]))
        if left["loss_bits"] != right["loss_bits"]:
            mismatches.append(("loss_bits", left["step"]))
        if left["param_digest"] != right["param_digest"]:
            mismatches.append(("param_digest", left["step"]))
    identical = not mismatches
    result = {
        "series": label,
        "arm_a_log": a_path.name,
        "arm_b_log": b_path.name,
        "steps": len(a),
        "gradient_comparisons": graded,
        "loss_bits": [row["loss_bits"] for row in a],
        "mismatches": mismatches,
        "arm_a_end": end(a_path).get("parameter_end_digest"),
        "arm_b_end": end(b_path).get("parameter_end_digest"),
        "verdict": "PASS" if identical else "FAIL",
    }
    print(
        f"[phase-xviii-in-graph] series={label} steps={len(a)} grad_compared={graded} "
        f"loss_bits_equal={all(x['loss_bits'] == y['loss_bits'] for x, y in zip(a, b))} "
        f"param_end_a={result['arm_a_end']} param_end_b={result['arm_b_end']} "
        f"mismatches={len(mismatches)} verdict={result['verdict']}"
    )
    return result


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: compare_in_graph.py <out.json> <label> <a.log> <b.log> [...]")
    out = Path(sys.argv[1])
    triples = sys.argv[2:]
    results = []
    for index in range(0, len(triples), 3):
        label, a, b = triples[index : index + 3]
        results.append(compare(label, Path(a), Path(b)))
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if any(result["verdict"] != "PASS" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
