#!/usr/bin/env python3
"""Stateful RDNA3 performance qualification gate.

The paired benchmark reports a 95% interval for the candidate/baseline
training-step latency ratio.  A release decision must not confuse "we could not
prove a regression" with "we proved performance is acceptable".  This gate
therefore emits one of four explicit states:

* PASS          sufficient evidence satisfies the selected policy;
* REGRESSION    the interval proves a slowdown beyond the allowed margin;
* INCONCLUSIVE  precision is inadequate to accept or reject the case;
* INVALID       the receipt, artifacts, matrix, or configuration are unusable.

Development mode returns exit 0 for INCONCLUSIVE so unrelated work is not
blocked, but the state and report still say INCONCLUSIVE, never PASS.  Release
mode returns success only for PASS and requires the complete batch matrix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

CASES = ("native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5")
NATIVE = {"native_vit_4x4", "native_vit_6x4"}
REQUIRED_CASES = set(CASES)
REQUIRED_BATCHES = (1, 2, 4)
STATES = ("PASS", "REGRESSION", "INCONCLUSIVE", "INVALID")
# A control case may not be accepted as equivalent outside this product bound.
# The per-case margin below is tightened from measured dispersion and can never
# exceed this value.
MAX_EQUIVALENCE_MARGIN = 0.02
MIN_EQUIVALENCE_MARGIN = 0.005


class InvalidReceipt(ValueError):
    """The evidence cannot be attributed to a valid benchmark run."""


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidReceipt(f"{name}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidReceipt(f"{name}: expected a finite number")
    return number


def _ci(result: dict) -> tuple[float, float]:
    ci = result.get("speedup_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise InvalidReceipt(f"{result.get('case')}: missing speedup_ci95")
    low, high = _finite(ci[0], "CI lower"), _finite(ci[1], "CI upper")
    if not (0.0 < low <= high):
        raise InvalidReceipt(f"{result.get('case')}: invalid confidence interval {ci}")
    return low, high


def _sha(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise InvalidReceipt(f"{name}: expected a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise InvalidReceipt(f"{name}: expected a SHA-256 hex digest") from error
    return value.lower()


def _case_margin(result: dict, cap: float) -> float:
    baseline = result.get("baseline")
    candidate = result.get("candidate")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise InvalidReceipt(f"{result.get('case')}: missing arm summaries")
    noise = max(
        _finite(baseline.get("relative_mad"), "baseline relative_mad"),
        _finite(candidate.get("relative_mad"), "candidate relative_mad"),
        _finite(baseline.get("sample_cv"), "baseline sample_cv"),
        _finite(candidate.get("sample_cv"), "candidate sample_cv"),
    )
    if noise < 0:
        raise InvalidReceipt(f"{result.get('case')}: negative dispersion")
    # Three robust deviations are a conservative per-case noise allowance.  The
    # lower bound prevents dividing by zero on a very quiet host; the cap is the
    # product's maximum accepted control drift and prevents noisy runs from
    # widening themselves into equivalence.
    return min(cap, max(MIN_EQUIVALENCE_MARGIN, 3.0 * noise))


def _validate_report(report: dict, source: str) -> dict:
    if not isinstance(report, dict):
        raise InvalidReceipt(f"{source}: receipt root must be an object")
    if report.get("schema") != 1:
        raise InvalidReceipt(f"{source}: unsupported benchmark receipt schema")
    if report.get("cross_arm_word_exact") is not True:
        raise InvalidReceipt(f"{source}: cross-arm final words differ")
    rounds = report.get("rounds")
    steps = report.get("measured_steps")
    batch = report.get("batch")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2 or rounds % 2:
        raise InvalidReceipt(f"{source}: invalid paired round count")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 3 <= steps <= 100:
        raise InvalidReceipt(f"{source}: invalid measured step count")
    if report.get("warmup_steps") != 2:
        raise InvalidReceipt(f"{source}: invalid warmup step count")
    if isinstance(batch, bool) or batch not in (1, 2, 4):
        raise InvalidReceipt(f"{source}: invalid or missing batch size")
    fixtures = report.get("fixtures")
    if not isinstance(fixtures, str) or not fixtures:
        raise InvalidReceipt(f"{source}: missing fixture identity")
    for arm in ("baseline", "candidate"):
        side = report.get(arm)
        if not isinstance(side, dict):
            raise InvalidReceipt(f"{source}: missing {arm} artifact identity")
        _sha(side.get("sha256"), f"{source}: {arm} sha256")
        if not isinstance(side.get("env"), dict):
            raise InvalidReceipt(f"{source}: {arm} environment is not an object")
        if not isinstance(side.get("binary"), str) or not side["binary"]:
            raise InvalidReceipt(f"{source}: {arm} binary path is missing")

    results = report.get("results")
    if not isinstance(results, list):
        raise InvalidReceipt(f"{source}: benchmark receipt has no results")
    if len(results) != len(CASES):
        raise InvalidReceipt(f"{source}: benchmark matrix is incomplete or duplicated")
    by_case = {}
    for result in results:
        if not isinstance(result, dict):
            raise InvalidReceipt(f"{source}: result entry is not an object")
        case = result.get("case")
        if case in by_case:
            raise InvalidReceipt(f"{source}: duplicate case {case!r}")
        by_case[case] = result
    missing = sorted(REQUIRED_CASES - set(by_case))
    if missing:
        raise InvalidReceipt(f"{source}: missing cases: {', '.join(missing)}")
    extra = sorted(set(by_case) - REQUIRED_CASES)
    if extra:
        raise InvalidReceipt(f"{source}: unexpected cases: {', '.join(extra)}")

    for case in CASES:
        result = by_case[case]
        role = "native" if case in NATIVE else "control"
        if result.get("role") != role:
            raise InvalidReceipt(f"{source}: {case}: unexpected role {result.get('role')!r}")
        low, high = _ci(result)
        if low <= 0.0 or high > 10.0:
            raise InvalidReceipt(f"{source}: {case}: implausible confidence interval")
        ratios = result.get("paired_round_speedups")
        if not isinstance(ratios, list) or len(ratios) != rounds:
            raise InvalidReceipt(f"{source}: {case}: missing paired round ratios")
        for ratio in ratios:
            value = _finite(ratio, f"{source}: {case} paired ratio")
            if value <= 0.0:
                raise InvalidReceipt(f"{source}: {case}: non-positive paired ratio")
        for arm in ("baseline", "candidate"):
            summary = result.get(arm)
            if not isinstance(summary, dict):
                raise InvalidReceipt(f"{source}: {case}: missing {arm} summary")
            medians = summary.get("round_medians_ms")
            if not isinstance(medians, list) or len(medians) != rounds:
                raise InvalidReceipt(f"{source}: {case}: missing {arm} round medians")
            if any(_finite(value, "round median") <= 0.0 for value in medians):
                raise InvalidReceipt(f"{source}: {case}: non-positive round median")
            samples = summary.get("samples")
            if isinstance(samples, bool) or samples != rounds * steps:
                raise InvalidReceipt(f"{source}: {case}: missing {arm} samples")
            _sha(summary.get("signature_sha256"), f"{source}: {case}: signature")
        _case_margin(result, MAX_EQUIVALENCE_MARGIN)
    return {
        "source": source,
        "batch": batch,
        "rounds": rounds,
        "steps": steps,
        "fixtures": fixtures,
        "baseline_sha": report["baseline"]["sha256"].lower(),
        "candidate_sha": report["candidate"]["sha256"].lower(),
        "baseline_env": report["baseline"]["env"],
        "candidate_env": report["candidate"]["env"],
        "round_medians_schema": True,
        "by_case": by_case,
    }


def _classify(case: str, result: dict, margin: float, mode: str) -> tuple[str, str]:
    low, high = _ci(result)
    regression_bound = 1.0 - margin
    improvement_bound = 1.0 + margin
    if high < regression_bound:
        return "REGRESSION", f"95% upper bound {high:.4f} is below {regression_bound:.4f}"
    if case in NATIVE:
        if low > 1.0:
            return "PASS", f"native improvement excludes 1.0 ({low:.4f}, {high:.4f})"
        if mode == "release":
            return "INCONCLUSIVE", f"release requires a proved native improvement ({low:.4f}, {high:.4f})"
        return "INCONCLUSIVE", f"native improvement is not established ({low:.4f}, {high:.4f})"
    if low > improvement_bound:
        return "PASS", f"control improvement exceeds margin ({low:.4f}, {high:.4f})"
    if low >= regression_bound and high <= improvement_bound:
        return "PASS", (
            f"control is practically equivalent within +/-{margin:.4f} "
            f"({low:.4f}, {high:.4f})"
        )
    return "INCONCLUSIVE", (
        f"control interval exceeds the +/-{margin:.4f} equivalence bounds "
        f"({low:.4f}, {high:.4f})"
    )


def evaluate_reports(
    reports: list[tuple[str, dict]],
    mode: str = "development",
    required_batches: Iterable[int] | None = None,
    expected_candidate_sha: str | None = None,
    expected_baseline_sha: str | None = None,
    expected_fixtures: str | None = None,
    equivalence_cap: float = MAX_EQUIVALENCE_MARGIN,
) -> dict:
    if mode not in ("development", "release"):
        raise InvalidReceipt(f"unknown gate mode {mode!r}")
    if not reports:
        raise InvalidReceipt("no benchmark reports supplied")
    if not (0.0 < equivalence_cap <= MAX_EQUIVALENCE_MARGIN):
        raise InvalidReceipt("equivalence cap must be in (0, 0.02]")
    validated = [_validate_report(report, source) for source, report in reports]
    if len({entry["batch"] for entry in validated}) != len(validated):
        raise InvalidReceipt("duplicate batch sizes in the benchmark matrix")

    baseline_sha = validated[0]["baseline_sha"]
    candidate_sha = validated[0]["candidate_sha"]
    fixtures = validated[0]["fixtures"]
    rounds = validated[0]["rounds"]
    steps = validated[0]["steps"]
    baseline_env = validated[0]["baseline_env"]
    candidate_env = validated[0]["candidate_env"]
    for entry in validated[1:]:
        if entry["baseline_sha"] != baseline_sha:
            raise InvalidReceipt("baseline binary differs across the benchmark matrix")
        if entry["candidate_sha"] != candidate_sha:
            raise InvalidReceipt("candidate binary differs across the benchmark matrix")
        if entry["fixtures"] != fixtures:
            raise InvalidReceipt("fixtures differ across the benchmark matrix")
        if entry["rounds"] != rounds or entry["steps"] != steps:
            raise InvalidReceipt("benchmark configuration differs across the matrix")
        if entry["baseline_env"] != baseline_env or entry["candidate_env"] != candidate_env:
            raise InvalidReceipt("benchmark environment differs across the matrix")
    if expected_baseline_sha and baseline_sha != _sha(expected_baseline_sha, "expected baseline"):
        raise InvalidReceipt("baseline binary hash does not match the expected artifact")
    if expected_candidate_sha and candidate_sha != _sha(expected_candidate_sha, "expected candidate"):
        raise InvalidReceipt("candidate binary hash does not match the expected artifact")
    if expected_fixtures and fixtures != expected_fixtures:
        raise InvalidReceipt("fixture identity does not match the expected fixture root")

    present = {entry["batch"] for entry in validated}
    if required_batches is None:
        required = set(REQUIRED_BATCHES) if mode == "release" else set(present)
    else:
        required = set(required_batches)
    missing = sorted(required - present)
    if missing:
        raise InvalidReceipt(f"benchmark matrix is missing batches: {', '.join(map(str, missing))}")

    report_rows = []
    all_rows = []
    for entry in sorted(validated, key=lambda item: item["batch"]):
        rows = []
        for case in CASES:
            result = entry["by_case"][case]
            margin = _case_margin(result, equivalence_cap)
            state, reason = _classify(case, result, margin, mode)
            low, high = _ci(result)
            row = {
                "case": case,
                "role": result["role"],
                "batch": entry["batch"],
                "speedup": result.get("speedup"),
                "paired_median_speedup": result.get("paired_median_speedup"),
                "speedup_ci95": [low, high],
                "equivalence_margin": margin,
                "state": state,
                "reason": reason,
            }
            rows.append(row)
            all_rows.append(row)
        report_state = _overall_state(row["state"] for row in rows)
        report_rows.append({"batch": entry["batch"], "state": report_state, "cases": rows})

    state = _overall_state(row["state"] for row in all_rows)
    if mode == "release" and state == "PASS":
        absent = sorted(required - present)
        if absent:
            state = "INVALID"
            reason = f"release matrix lacks batches {absent}"
        else:
            reason = "all required workloads and batch sizes have conclusive evidence"
    elif state == "PASS":
        reason = "all cases satisfy the development policy"
    else:
        reason = f"{sum(row['state'] != 'PASS' for row in all_rows)} case(s) are not release-acceptable"

    return {
        "schema": 2,
        "mode": mode,
        "policy": "release-qualification" if mode == "release" else "development",
        "state": state,
        "pass": state == "PASS",
        "release_qualified": mode == "release" and state == "PASS",
        "reason": reason,
        "required_batches": sorted(required),
        "batch_present": sorted(present),
        "cross_arm_word_exact": True,
        "fixtures": fixtures,
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "rounds": rounds,
        "measured_steps": steps,
        "reports": report_rows,
        "cases": all_rows,
        "regressions": [row["case"] for row in all_rows if row["state"] == "REGRESSION"],
        "inconclusive": [row["case"] for row in all_rows if row["state"] == "INCONCLUSIVE"],
    }


def _overall_state(states: Iterable[str]) -> str:
    states = list(states)
    if "INVALID" in states:
        return "INVALID"
    if "REGRESSION" in states:
        return "REGRESSION"
    if "INCONCLUSIVE" in states:
        return "INCONCLUSIVE"
    return "PASS"


def evaluate(report: dict, require_native_improvement: bool = False) -> dict:
    """Compatibility wrapper for callers that evaluate one batch."""
    return evaluate_reports(
        [("<memory>", report)],
        mode="release" if require_native_improvement else "development",
        required_batches=None,
    )


def markdown(gate: dict) -> str:
    lines = [
        "# RDNA3 V4.1 performance gate",
        "",
        f"Mode: `{gate['mode']}`; state: **{gate['state']}**.",
        "",
        f"Reason: {gate['reason']}.",
        "",
        f"Batches: `{', '.join(map(str, gate['batch_present']))}` "
        f"(required: `{', '.join(map(str, gate['required_batches']))}`).",
        "",
        "| Batch | Case | Role | Paired median | 95% CI | Margin | State |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in gate["cases"]:
        lines.append(
            "| {batch} | {case} | {role} | {median:.4f}x | [{low:.4f}, {high:.4f}] | "
            "+/-{margin:.4f} | {state} |".format(
                batch=row["batch"],
                case=row["case"],
                role=row["role"],
                median=row["paired_median_speedup"],
                low=row["speedup_ci95"][0],
                high=row["speedup_ci95"][1],
                margin=row["equivalence_margin"],
                state=row["state"],
            )
        )
    lines.extend(
        [
            "",
            "PASS requires conclusive evidence. REGRESSION and INVALID always fail. "
            "Development mode does not block on INCONCLUSIVE, but it does not label "
            "the performance qualification successful. Release mode requires PASS "
            "for every case at batches 1, 2, and 4.",
            "",
        ]
    )
    return "\n".join(lines)


def _invalid(source: str, message: str) -> dict:
    return {
        "schema": 2,
        "mode": "unknown",
        "policy": "invalid",
        "state": "INVALID",
        "pass": False,
        "release_qualified": False,
        "reason": f"{source}: {message}",
        "required_batches": [],
        "batch_present": [],
        "cross_arm_word_exact": False,
        "cases": [],
        "reports": [],
        "regressions": [],
        "inconclusive": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="benchmark.json receipts")
    parser.add_argument("--mode", choices=("development", "release"), default="development")
    parser.add_argument("--require-batches", default=None, help="e.g. 1,2,4")
    parser.add_argument("--expected-candidate-sha", default=None)
    parser.add_argument("--expected-baseline-sha", default=None)
    parser.add_argument("--expected-fixtures", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--require-native-improvement",
        action="store_true",
        help="deprecated alias for --mode release",
    )
    args = parser.parse_args()

    mode = "release" if args.require_native_improvement else args.mode
    try:
        batches = None
        if args.require_batches:
            batches = tuple(int(value) for value in args.require_batches.split(","))
            if not batches or any(batch not in (1, 2, 4) for batch in batches):
                raise InvalidReceipt("--require-batches values must be drawn from 1,2,4")
        reports = []
        for path in args.reports:
            try:
                reports.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError) as error:
                raise InvalidReceipt(f"{path}: unreadable or corrupted JSON ({error})") from error
        gate = evaluate_reports(
            reports,
            mode=mode,
            required_batches=batches,
            expected_candidate_sha=args.expected_candidate_sha,
            expected_baseline_sha=args.expected_baseline_sha,
            expected_fixtures=args.expected_fixtures,
        )
        exit_code = 0 if gate["state"] == "PASS" else 1
        if gate["state"] == "INCONCLUSIVE" and mode == "development":
            exit_code = 0
    except Exception as error:  # The gate must emit INVALID, never a traceback-only failure.
        gate = _invalid("gate", str(error))
        gate["mode"] = mode
        exit_code = 2

    if args.json:
        args.json.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    text = markdown(gate)
    if args.markdown:
        args.markdown.write_text(text + "\n", encoding="utf-8")
    print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
