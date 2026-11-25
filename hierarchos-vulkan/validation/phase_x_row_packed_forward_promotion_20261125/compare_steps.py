#!/usr/bin/env python3
"""Compare the two complete-step digest arms of the Phase X promotion.

Reads `step-production.log` and `step-generic.log` from the directory given on
the command line. The production arm installs nothing -- it is the shipped
selector -- and the generic arm is the same release binary with
`HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1`, so both arms are this build's own
committed generic kernel and the comparison cannot be satisfied by two runs of
the same execution.

Per step it compares:

  * the loss bit pattern,
  * the host-side SHA-256 digest of the LoRA gradient map,
  * the digest of the parameter map that the same step produced.

The digests are name-, length- and bit-pattern-exact over
`lora_gradient_values()` and `parameter_values()`, so equality here is a bitwise
equality of every gradient tensor in the run.

It also reports the dispatch counters each arm recorded, because a digest match
between two runs that both took the generic route would be a tautology rather
than an equivalence.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STEP = re.compile(
    r"\[phase-x-step\] arm=(\S+) step=(\d+) epoch=(\d+) batch=(\d+) "
    r"measured=(\w+) step_ms=([0-9.]+) loss_bits=([0-9a-f]{8}) loss=(\S+) "
    r"grad_digest=(\w+) grad_elements=(\d+) grad_norm=(\S+) "
    r"param_digest=(\w+) param_elements=(\d+)"
)
RUN = re.compile(
    r"\[phase-x-step-run\] arm=(\S+) fixture=(\S+) sequences=(\d+) .*"
    r"row_packed_installed=(\w+) parameter_start_digest=(\w+)"
)
CHECKPOINT = re.compile(
    r"\[phase-x-checkpoint\] arm=(\S+) file=(\S+) bytes=(\d+) sha256=(\w+)"
)
END = re.compile(
    r"\[phase-x-step-run-end\] arm=(\S+) parameter_end_digest=(\w+) "
    r"parameter_elements=(\d+) forward_generic=(\d+) forward_row_packed=(\d+) "
    r"forward_short=(\d+) forward_total=(\d+) backward_scalar=(\d+) "
    r"backward_query=(\d+) backward_key=(\d+)"
)


def read(path: Path) -> tuple[dict, list[dict], dict]:
    header: dict = {}
    end: dict = {}
    steps: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CHECKPOINT.search(line)
        if match:
            header.setdefault("checkpoint", {})[match.group(2)] = match.group(4)
        match = RUN.search(line)
        if match:
            header = {
                "arm": match.group(1),
                "fixture": match.group(2),
                "sequences": int(match.group(3)),
                "row_packed_installed": match.group(4) == "true",
                "parameter_start_digest": match.group(5),
            }
        match = END.search(line)
        if match:
            end = {
                "arm": match.group(1),
                "parameter_end_digest": match.group(2),
                "parameter_elements": int(match.group(3)),
                "forward_generic": int(match.group(4)),
                "forward_row_packed": int(match.group(5)),
                "forward_short": int(match.group(6)),
                "forward_total": int(match.group(7)),
                "backward_scalar": int(match.group(8)),
                "backward_query": int(match.group(9)),
                "backward_key": int(match.group(10)),
            }
        match = STEP.search(line)
        if match:
            steps.append(
                {
                    "arm": match.group(1),
                    "step": int(match.group(2)),
                    "loss_bits": match.group(7),
                    "grad_digest": match.group(9),
                    "grad_elements": int(match.group(10)),
                    "param_digest": match.group(12),
                    "param_elements": int(match.group(13)),
                }
            )
    return header, steps, end


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: compare_steps.py RAW_DIR", file=sys.stderr)
        return 2
    raw = Path(argv[1])
    generic_header, generic_steps, generic_end = read(raw / "step-generic.log")
    production_header, production_steps, production_end = read(raw / "step-production.log")
    if len(generic_steps) != len(production_steps) or not generic_steps:
        print(
            json.dumps(
                {"status": "FAIL", "reason": "step count mismatch"},
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    rows = []
    loss_mismatches = 0
    grad_mismatches = 0
    param_mismatches = 0
    for generic, production in zip(generic_steps, production_steps):
        loss_equal = generic["loss_bits"] == production["loss_bits"]
        grad_equal = generic["grad_digest"] == production["grad_digest"]
        param_equal = generic["param_digest"] == production["param_digest"]
        loss_mismatches += not loss_equal
        grad_mismatches += not grad_equal
        param_mismatches += not param_equal
        rows.append(
            {
                "step": generic["step"],
                "loss_bits_generic": generic["loss_bits"],
                "loss_bits_production": production["loss_bits"],
                "loss_bits_equal": loss_equal,
                "grad_digest_generic": generic["grad_digest"],
                "grad_digest_production": production["grad_digest"],
                "grad_elements": generic["grad_elements"],
                "grad_digest_equal": grad_equal,
                "param_digest_equal": param_equal,
            }
        )
    # A digest match is only evidence of equivalence when the two arms actually
    # executed different attention-forward routes.
    routes_distinct = (
        production_header.get("row_packed_installed") is True
        and generic_header.get("row_packed_installed") is False
        and production_end.get("forward_row_packed", 0) > 0
        and production_end.get("forward_generic", 0) == 0
        and generic_end.get("forward_row_packed", 0) == 0
        and generic_end.get("forward_generic", 0) > 0
    )
    generic_checkpoint = generic_header.get("checkpoint", {})
    production_checkpoint = production_header.get("checkpoint", {})
    checkpoint_names = sorted(set(generic_checkpoint) | set(production_checkpoint))
    checkpoint_mismatches = [
        name
        for name in checkpoint_names
        if generic_checkpoint.get(name) != production_checkpoint.get(name)
    ]
    exact = (
        loss_mismatches == 0
        and grad_mismatches == 0
        and param_mismatches == 0
        and not checkpoint_mismatches
        and bool(checkpoint_names)
    )
    report = {
        "status": "PASS" if exact and routes_distinct else "FAIL",
        "checkpoint_files_compared": len(checkpoint_names),
        "checkpoint_file_mismatches": checkpoint_mismatches,
        "checkpoint_digests": {
            name: {
                "generic": generic_checkpoint.get(name),
                "production": production_checkpoint.get(name),
            }
            for name in checkpoint_names
        },
        "fixture": generic_header.get("fixture"),
        "dataset_sequences": generic_header.get("sequences"),
        "steps_compared": len(rows),
        "loss_bits_mismatches": loss_mismatches,
        "gradient_digest_mismatches": grad_mismatches,
        "parameter_digest_mismatches": param_mismatches,
        "gradient_elements_per_step": generic_steps[0]["grad_elements"],
        "gradient_bits_compared": generic_steps[0]["grad_elements"] * 32 * len(rows),
        "parameter_start_digest_generic": generic_header.get("parameter_start_digest"),
        "parameter_start_digest_production": production_header.get("parameter_start_digest"),
        "parameter_end_digest_generic": generic_end.get("parameter_end_digest"),
        "parameter_end_digest_production": production_end.get("parameter_end_digest"),
        "parameter_end_digest_equal": generic_end.get("parameter_end_digest")
        == production_end.get("parameter_end_digest"),
        "routes_distinct": routes_distinct,
        "dispatch_counters_generic": {
            key: value
            for key, value in generic_end.items()
            if key.startswith(("forward_", "backward_"))
        },
        "dispatch_counters_production": {
            key: value
            for key, value in production_end.items()
            if key.startswith(("forward_", "backward_"))
        },
        "per_step": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
