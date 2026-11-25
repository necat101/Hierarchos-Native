#!/usr/bin/env python3
"""Paired A/B analysis of the Phase X complete-step timing rounds.

One arm per process, four processes per round in A B B A order, alternating
which arm runs first between rounds. `A` is the committed generic forward and
`B` is the promoted row-packed route as the shipped selector chooses it.

Per process the measured steps (warmups excluded by the harness) yield a median;
per round the two A medians and the two B medians pool to one median each and the
round's ratio is A/B. The report gives the mean paired ratio with a 95%
confidence interval over rounds, plus the pooled medians over every sample.

Nothing is invented: a missing process log is an error, and a round with no
measured samples is skipped rather than counted.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

STEP = re.compile(
    r"\[phase-x-step\] arm=(\S+) step=(\d+) .* measured=(\w+) step_ms=([0-9.]+)"
)
END = re.compile(
    r"\[phase-x-step-run-end\] arm=(\S+) .*forward_generic=(\d+) "
    r"forward_row_packed=(\d+)"
)

# Two-tailed 95% Student t for small samples.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def samples(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP.search(line)
        if match and match.group(3) == "true":
            values.append(float(match.group(4)))
    return values


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: assemble.py RAW_DIR [ROUNDS]", file=sys.stderr)
        return 2
    raw = Path(argv[1])
    rounds = int(argv[2]) if len(argv) > 2 else 4
    report: dict = {"status": "PASS", "rounds": [], "missing": []}
    pooled = {"generic": [], "production": []}
    ratios: list[float] = []
    for index in range(rounds):
        a_values: list[float] = []
        b_values: list[float] = []
        counters: dict[str, dict] = {}
        for position in range(4):
            arm = "production" if position in (1, 2) else "generic"
            path = raw / f"abba-r{index}-p{position}-{'B' if arm == 'production' else 'A'}.log"
            if not path.is_file():
                report["missing"].append(path.name)
                continue
            values = samples(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            match = END.search(text)
            if match:
                counters[arm] = {
                    "forward_generic": int(match.group(2)),
                    "forward_row_packed": int(match.group(3)),
                }
            if arm == "production":
                b_values.extend(values)
            else:
                a_values.extend(values)
        if not a_values or not b_values:
            report["missing"].append(f"round {index}: no measured samples")
            continue
        a_median = statistics.median(a_values)
        b_median = statistics.median(b_values)
        ratio = a_median / b_median
        ratios.append(ratio)
        pooled["generic"].extend(a_values)
        pooled["production"].extend(b_values)
        report["rounds"].append(
            {
                "round": index,
                "generic_median_ms": round(a_median, 6),
                "production_median_ms": round(b_median, 6),
                "ratio": round(ratio, 6),
                "generic_samples": len(a_values),
                "production_samples": len(b_values),
                "counters": counters,
            }
        )
    if report["missing"]:
        report["status"] = "FAIL"
    if not ratios:
        report.update({"status": "FAIL", "reason": "no complete rounds"})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    mean = statistics.fmean(ratios)
    if len(ratios) > 1:
        stdev = statistics.stdev(ratios)
        half = T95.get(len(ratios) - 1, 1.96) * stdev / math.sqrt(len(ratios))
    else:
        stdev, half = 0.0, 0.0
    all_a = pooled["generic"]
    all_b = pooled["production"]

    def describe(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "samples": len(values),
            "median_ms": round(statistics.median(values), 6),
            "mean_ms": round(statistics.fmean(values), 6),
            "stdev_ms": round(statistics.stdev(values), 6),
            "p25_ms": round(ordered[len(ordered) // 4], 6),
            "p95_ms": round(ordered[int(len(ordered) * 0.95)], 6),
            "min_ms": round(ordered[0], 6),
            "max_ms": round(ordered[-1], 6),
        }

    report["summary"] = {
        "paired_ratio_mean": round(mean, 6),
        "paired_ratio_ci95": [round(mean - half, 6), round(mean + half, 6)],
        "paired_ratio_stdev": round(stdev, 6),
        "paired_ratio_median": round(statistics.median(ratios), 6),
        "paired_ratio_min": round(min(ratios), 6),
        "paired_ratio_max": round(max(ratios), 6),
        "ratio_of_pooled_medians": round(
            statistics.median(all_a) / statistics.median(all_b), 6
        ),
        "generic": describe(all_a),
        "production": describe(all_b),
        "student_t_df": len(ratios) - 1,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
