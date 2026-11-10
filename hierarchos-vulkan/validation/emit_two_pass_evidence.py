#!/usr/bin/env python3
"""Structure the two-pass attention backward evidence from its raw logs.

The two logs are the only source of truth: the Rust comparison's ``--nocapture``
output for the twelve gradient tensors, and the dispatch-gated equivalence
harness output for the execution proof and adapter digests. Nothing here reads a
previously written summary, so a receipt can always be re-derived from the logs
it ships.

    python validation/emit_two_pass_evidence.py \\
        --test-log target/phase16/run/two-pass/test.log \\
        --harness-log target/phase16/run/two-pass/harness.log \\
        --output target/phase16/run/two-pass/gradient.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

CASES = ("fixture-causal", "fixture-full", "padded-causal", "small-causal")
TENSORS = ("q", "k", "v")

GRAD = re.compile(
    r"^\[(?P<case>[a-z0-9-]+)\] batch=(?P<batch>\d+) heads=(?P<heads>\d+) seq=(?P<seq>\d+) "
    r"head_dim=(?P<head_dim>\d+) causal=(?P<causal>[01]) masked=(?P<masked>true|false) "
    r"unwritten=(?P<unwritten>\d+) grad_(?P<tensor>[qkv]): elements=(?P<elements>\d+) "
    r"bitwise_equal=(?P<equal>\d+) max_abs=(?P<max_abs>\S+) max_rel=(?P<max_rel>\S+) "
    r"rms=(?P<rms>\S+) max_ulp=(?P<max_ulp>\d+) .* (?P<verdict>PASS|FAIL)$",
    re.M,
)
DISPATCH = re.compile(
    r"^(?P<arm>scalar arm|twopass arm)\s*:\s*scalar=(?P<scalar>\d+) "
    r"query=(?P<query>\d+) key=(?P<key>\d+)$",
    re.M,
)
ADAPTER = re.compile(r"^(?P<arm>scalar|twopass)\s+(?P<path>\S+)\s+(?P<sha>[0-9a-f]{64})$", re.M)
CONTROLS = re.compile(r"negative controls: (?P<detected>\d+)/(?P<total>\d+) detected")
DIAGNOSTIC = "two-pass backward:"
HARNESS_PASS = "PASS: both passes dispatched and every adapter file is bit-identical"
ENVIRONMENT = "HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _failures(rows, dispatches, diagnostic, harness, controls, adapters):
    failures = []
    if len(rows) != len(CASES) * len(TENSORS):
        failures.append(f"expected 12 gradient comparisons, found {len(rows)}")
    keys = {(row["case"], row["tensor"]) for row in rows}
    if keys != {(case, tensor) for case in CASES for tensor in TENSORS}:
        failures.append("gradient cases are missing, duplicated or unexpected")
    for row in rows:
        if row["verdict"] != "PASS":
            failures.append(f"{row['case']}/{row['tensor']} verdict is {row['verdict']}")
        if row["unwritten"] != 0:
            failures.append(f"{row['case']}/{row['tensor']} left {row['unwritten']} unwritten elements")
        if row["elements"] <= 0 or row["bitwise_equal"] != row["elements"]:
            failures.append(
                f"{row['case']}/{row['tensor']} is not bit-exact "
                f"({row['bitwise_equal']}/{row['elements']})"
            )
        if row["max_abs"] != 0.0 or row["max_ulp"] != 0:
            failures.append(f"{row['case']}/{row['tensor']} reports a nonzero error")
    for case in CASES:
        geometry = {
            (row["batch"], row["heads"], row["seq"], row["head_dim"], row["causal"], row["masked"])
            for row in rows
            if row["case"] == case
        }
        if len(geometry) != 1:
            failures.append(f"{case} does not describe one geometry across dQ/dK/dV")
    if set(dispatches) != {"scalar_arm", "twopass_arm"}:
        failures.append("dispatch counts are missing an arm")
    else:
        control, candidate = dispatches["scalar_arm"], dispatches["twopass_arm"]
        if control["scalar"] <= 0 or control["query"] != 0 or control["key"] != 0:
            failures.append("the scalar control arm did not run the scalar backward alone")
        if candidate["scalar"] != 0 or candidate["query"] <= 0 or candidate["key"] <= 0:
            failures.append("the candidate arm did not run both new passes without falling back")
    if not diagnostic:
        failures.append("the one-shot mode-9 diagnostic is missing")
    if not harness:
        failures.append("the equivalence harness did not report its PASS line")
    if not controls or controls["total"] <= 0 or controls["detected"] != controls["total"]:
        failures.append("the harness negative controls were not all detected")
    if not adapters["identical"]:
        failures.append("the two arms exported different adapter bytes")
    if len(adapters["files"]) < 2:
        failures.append("the adapter digest set is empty or too small")
    return failures


def parse(test_log: Path, harness_log: Path) -> dict:
    text = Path(test_log).read_text(errors="replace")
    harness_text = Path(harness_log).read_text(errors="replace")
    rows = []
    for match in GRAD.finditer(text):
        groups = match.groupdict()
        rows.append(dict(
            case=groups["case"], tensor=groups["tensor"],
            batch=int(groups["batch"]), heads=int(groups["heads"]),
            seq=int(groups["seq"]), head_dim=int(groups["head_dim"]),
            causal=int(groups["causal"]), masked=groups["masked"] == "true",
            elements=int(groups["elements"]), bitwise_equal=int(groups["equal"]),
            max_abs=float(groups["max_abs"]), max_ulp=int(groups["max_ulp"]),
            unwritten=int(groups["unwritten"]), verdict=groups["verdict"],
        ))
    dispatches = {}
    for match in DISPATCH.finditer(harness_text):
        arm = match.group("arm").split()[0] + "_arm"
        dispatches[arm] = dict(
            scalar=int(match.group("scalar")),
            query=int(match.group("query")),
            key=int(match.group("key")),
        )
    diagnostic = next(
        (line.strip() for line in harness_text.splitlines()
         if DIAGNOSTIC in line and "kernels=true" in line),
        "",
    )
    digests = {}
    for match in ADAPTER.finditer(harness_text):
        digests.setdefault(match.group("arm"), {})[match.group("path")] = match.group("sha")
    adapters = dict(
        identical=bool(digests.get("scalar")) and digests.get("scalar") == digests.get("twopass"),
        files=sorted((digests.get("scalar") or {}).items()),
    )
    controls_match = CONTROLS.search(harness_text)
    controls = (dict(detected=int(controls_match.group("detected")),
                     total=int(controls_match.group("total")))
                if controls_match else None)
    harness = HARNESS_PASS in harness_text
    failures = _failures(rows, dispatches, diagnostic, harness, controls, adapters)
    return dict(
        schema=1,
        scope="phase16_two_pass_attention_backward_gradients",
        **{"pass": not failures},
        environment=ENVIRONMENT,
        cases=rows,
        comparisons=len(rows),
        elements=sum(row["elements"] for row in rows),
        bitwise_equal=sum(row["bitwise_equal"] for row in rows),
        max_abs=max((row["max_abs"] for row in rows), default=0.0),
        max_ulp=max((row["max_ulp"] for row in rows), default=0),
        unwritten=sum(row["unwritten"] for row in rows),
        dispatches=dispatches,
        diagnostic=diagnostic,
        adapters=adapters,
        negative_controls=controls,
        harness_pass=harness,
        failures=failures,
        evidence=dict(test_log=digest(test_log), harness_log=digest(harness_log)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--harness-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = parse(args.test_log, args.harness_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if report['pass'] else 'FAIL'}: {report['comparisons']} comparisons, "
          f"{report['bitwise_equal']}/{report['elements']} bit-exact elements, "
          f"dispatches {report['dispatches']}")
    for failure in report["failures"]:
        print("  " + failure)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
