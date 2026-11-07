#!/usr/bin/env python3
"""Fail RDNA3 CI only on a statistically clear performance regression.

The paired benchmark deliberately reports three outcomes for each case:
candidate faster, candidate slower, or inconclusive.  Ordinary CI must not turn
an inconclusive small win into a failure, and it must not turn an inconclusive
small loss into a pass.  This gate therefore fails when the *entire* 95%
confidence interval for any native or control case is below 1.0.

A release promotion can additionally pass ``--require-native-improvement`` to
require every native case's interval to be wholly above 1.0.  That stricter
policy is not the default for the scheduled RDNA3 job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = ("native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5")
NATIVE = {"native_vit_4x4", "native_vit_6x4"}
REQUIRED_CASES = set(CASES)


def _ci(result: dict) -> tuple[float, float]:
    ci = result.get("speedup_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{result.get('case')}: missing speedup_ci95")
    low, high = float(ci[0]), float(ci[1])
    if not (0.0 < low <= high):
        raise ValueError(f"{result.get('case')}: invalid confidence interval {ci}")
    return low, high


def evaluate(report: dict, require_native_improvement: bool = False) -> dict:
    if report.get("schema") != 1:
        raise ValueError("unsupported benchmark receipt schema")
    if not report.get("cross_arm_word_exact"):
        raise ValueError("cross-arm final words differ; performance is not comparable")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark receipt has no results")
    by_case = {result.get("case"): result for result in results}
    missing = sorted(REQUIRED_CASES - set(by_case))
    if missing:
        raise ValueError(f"benchmark receipt is missing cases: {', '.join(missing)}")
    if len(by_case) != len(results):
        raise ValueError("benchmark receipt contains duplicate cases")

    rows = []
    regressions = []
    improvements = []
    native_improvements = []
    inconclusive = []
    for case in CASES:
        result = by_case[case]
        low, high = _ci(result)
        if result.get("role") != ("native" if case in NATIVE else "control"):
            raise ValueError(f"{case}: unexpected role {result.get('role')!r}")
        if high < 1.0:
            conclusion = "candidate_slower"
            regressions.append((case, low, high))
        elif low > 1.0:
            conclusion = "candidate_faster"
            improvements.append((case, low, high))
            if case in NATIVE:
                native_improvements.append(case)
        else:
            conclusion = "inconclusive"
            inconclusive.append((case, low, high))
        rows.append(
            {
                "case": case,
                "role": result["role"],
                "speedup": result.get("speedup"),
                "paired_median_speedup": result.get("paired_median_speedup"),
                "speedup_ci95": [low, high],
                "conclusion": conclusion,
            }
        )

    allowed_harness_failure = "native speedup CI does not exclude 1.0"
    unexpected_failures = [
        failure
        for failure in report.get("gate_failures", [])
        if allowed_harness_failure not in failure
    ]
    if unexpected_failures:
        raise ValueError("benchmark harness reported a non-performance failure")

    failures = []
    if regressions:
        failures.extend(
            f"{case}: 95% CI upper bound {high:.4f} is below 1.0"
            for case, _low, high in regressions
        )
    if require_native_improvement:
        failures.extend(
            f"{case}: native improvement is not statistically established ({low:.4f}, {high:.4f})"
            for case, low, high in inconclusive
            if case in NATIVE
        )
    return {
        "schema": 1,
        "policy": "clear-regression-only"
        if not require_native_improvement
        else "require-native-improvement",
        "pass": not failures,
        "cross_arm_word_exact": True,
        "cases": rows,
        "significant_regressions": [case for case, _low, _high in regressions],
        "significant_improvements": [case for case, _low, _high in improvements],
        "native_improvements": native_improvements,
        "inconclusive": [case for case, _low, _high in inconclusive],
        "gate_failures": failures,
    }


def markdown(gate: dict) -> str:
    lines = [
        "# RDNA3 V4.1 performance gate",
        "",
        f"Policy: `{gate['policy']}`; result: **{'PASS' if gate['pass'] else 'FAIL'}**.",
        "",
        "| Case | Role | Paired median | 95% CI | Conclusion |",
        "|---|---:|---:|---:|---|",
    ]
    for row in gate["cases"]:
        lines.append(
            "| {case} | {role} | {median:.4f}x | [{low:.4f}, {high:.4f}] | {conclusion} |".format(
                case=row["case"],
                role=row["role"],
                median=row["paired_median_speedup"],
                low=row["speedup_ci95"][0],
                high=row["speedup_ci95"][1],
                conclusion=row["conclusion"],
            )
        )
    lines.extend(
        [
            "",
            "Significant regressions: "
            + (", ".join(gate["significant_regressions"]) or "none"),
            "",
            "Inconclusive cases (not treated as regressions): "
            + (", ".join(gate["inconclusive"]) or "none"),
            "",
            "A candidate is faster only when its full interval is above 1.0. "
            "A regression fails only when its full interval is below 1.0.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="benchmark.json from the paired harness")
    parser.add_argument("--require-native-improvement", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    gate = evaluate(report, args.require_native_improvement)
    if args.json:
        args.json.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    text = markdown(gate)
    if args.markdown:
        args.markdown.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
