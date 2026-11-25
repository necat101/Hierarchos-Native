#!/usr/bin/env python3
"""Compare the two complete-step digest arms of the Phase IX experiment.

Reads `step-digest-generic.log` and `step-digest-rowpacked-cached.log` from the
directory given on the command line and compares, per step:

  * the loss bit pattern,
  * the host-side SHA-256 digest of the LoRA gradient map,
  * the digest of the parameter map that the same step produced.

The digests are computed by the harness over `lora_gradient_values()` and
`parameter_values()` with a name-, length- and bit-pattern-exact hash, so an
equality here is a bitwise equality of every gradient tensor in the run, not an
approximate agreement.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STEP = re.compile(
    r"\[phase-ix-step\] arm=(\S+) step=(\d+) epoch=(\d+) batch=(\d+) "
    r"measured=(\w+) step_ms=([0-9.]+) loss_bits=([0-9a-f]{8}) loss=(\S+) "
    r"grad_digest=(\w+) grad_elements=(\d+) grad_norm=(\S+) "
    r"param_digest=(\w+) param_elements=(\d+)"
)
RUN = re.compile(r"\[phase-ix-step-run\] arm=(\S+) fixture=(\S+) sequences=(\d+)")
END = re.compile(r"\[phase-ix-step-run-end\] arm=(\S+) parameter_end_digest=(\w+)")


def read(path: Path) -> tuple[dict, list[dict], dict]:
    header: dict = {}
    end: dict = {}
    steps: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RUN.search(line)
        if match:
            header = {
                "arm": match.group(1),
                "fixture": match.group(2),
                "sequences": int(match.group(3)),
            }
        match = END.search(line)
        if match:
            end = {"arm": match.group(1), "parameter_end_digest": match.group(2)}
        match = STEP.search(line)
        if match:
            steps.append({
                "step": int(match.group(2)),
                "epoch": int(match.group(3)),
                "batch": int(match.group(4)),
                "loss_bits": match.group(7),
                "grad_digest": match.group(9),
                "grad_elements": int(match.group(10)),
                "grad_norm": match.group(11),
                "param_digest": match.group(12),
                "param_elements": int(match.group(13)),
            })
    return header, steps, end


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: compare_steps.py RAW_DIR", file=sys.stderr)
        return 2
    raw = Path(argv[1])
    header_a, steps_a, end_a = read(raw / "step-digest-generic.log")
    header_b, steps_b, end_b = read(raw / "step-digest-rowpacked-cached.log")
    if len(steps_a) != len(steps_b) or not steps_a:
        print(json.dumps({"status": "FAIL", "reason": "step count mismatch"}, indent=2))
        return 1
    rows = []
    loss_mismatches = 0
    grad_mismatches = 0
    param_mismatches = 0
    for a, b in zip(steps_a, steps_b):
        loss_equal = a["loss_bits"] == b["loss_bits"]
        grad_equal = a["grad_digest"] == b["grad_digest"]
        param_equal = a["param_digest"] == b["param_digest"]
        loss_mismatches += not loss_equal
        grad_mismatches += not grad_equal
        param_mismatches += not param_equal
        rows.append({
            "step": a["step"],
            "loss_bits_generic": a["loss_bits"],
            "loss_bits_rowpacked": b["loss_bits"],
            "loss_bits_equal": loss_equal,
            "grad_digest_generic": a["grad_digest"],
            "grad_digest_rowpacked": b["grad_digest"],
            "grad_elements": a["grad_elements"],
            "grad_digest_equal": grad_equal,
            "param_digest_equal": param_equal,
        })
    report = {
        "status": "PASS"
        if loss_mismatches == 0 and grad_mismatches == 0 and param_mismatches == 0
        else "FAIL",
        "fixture": header_a.get("fixture"),
        "dataset_sequences": header_a.get("sequences"),
        "steps_compared": len(rows),
        "loss_bits_mismatches": loss_mismatches,
        "gradient_digest_mismatches": grad_mismatches,
        "parameter_digest_mismatches": param_mismatches,
        "gradient_elements_per_step": steps_a[0]["grad_elements"],
        "gradient_bits_compared": steps_a[0]["grad_elements"] * 32 * len(rows),
        "parameter_end_digest_generic": end_a.get("parameter_end_digest"),
        "parameter_end_digest_rowpacked": end_b.get("parameter_end_digest"),
        "per_step": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
