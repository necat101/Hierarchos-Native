#!/usr/bin/env python3
"""Phase XVII complete-step reduction: per-round paired A/B ratios and a CI.

Reads the `[phase-xii-step]` lines the established step harness prints, keeps
only the measured steps (`measured=true`, i.e. after the three warmups), and
reports the median per-round ratio, the mean, and a 95% confidence interval
over the five round ratios. Each round is an A B B A sequence, so the two A
processes and the two B processes are averaged into that round's A and B
medians, which is the Phase XII aggregation.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
T_CRIT = {5: 2.776, 4: 3.182}  # two-sided 95%, df = rounds - 1


def step_ms(path: Path) -> list[float]:
    out: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xii-step]" not in line:
            continue
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if fields.get("measured") != "true":
            continue
        out.append(float(fields["step_ms"]))
    return out


def summarize(arm_ms: list[float]) -> dict:
    return {
        "samples": len(arm_ms),
        "median_ms": statistics.median(arm_ms),
        "mean_ms": statistics.fmean(arm_ms),
        "stdev_ms": statistics.stdev(arm_ms) if len(arm_ms) > 1 else 0.0,
    }


def series(wmma: str) -> dict:
    rounds = []
    for round_index in range(ROUNDS):
        a_runs, b_runs = [], []
        for position in range(4):
            tag = "A" if position in (0, 3) else "B"
            path = RAW / f"s7-{wmma}-r{round_index}-p{position}-{tag}.log"
            samples = step_ms(path)
            if len(samples) != 26:
                raise SystemExit(f"{path.name}: {len(samples)} measured steps, expected 26")
            (a_runs if tag == "A" else b_runs).append(statistics.median(samples))
        a = statistics.median(a_runs)
        b = statistics.median(b_runs)
        round_dict = {
            "round": round_index,
            "a_runs_ms": a_runs,
            "b_runs_ms": b_runs,
            "a_ms": a,
            "b_ms": b,
            "ratio": b / a,
        }
        rounds.append(round_dict)

    ratios = [r["ratio"] for r in rounds]
    median = statistics.median(ratios)
    mean = statistics.fmean(ratios)
    stdev = statistics.stdev(ratios)
    half = T_CRIT.get(ROUNDS, 2.776) * stdev / (len(ratios) ** 0.5)
    return {
        "wmma_mode": wmma,
        "rounds": rounds,
        "round_ratios": ratios,
        "median_round_ratio": median,
        "mean_round_ratio": mean,
        "ratio_stdev": stdev,
        "ratio_ci95": [mean - half, mean + half],
        "speedup": 1.0 / median,
        "rounds_improved": sum(1 for r in ratios if r < 1.0),
        "rounds_total": len(ratios),
        "median_step_ms": {
            "A": statistics.median([r["a_ms"] for r in rounds]),
            "B": statistics.median([r["b_ms"] for r in rounds]),
        },
        "arm": {
            "A": summarize([statistics.median(step_ms(RAW / f"s7-{wmma}-r{r}-p{p}-A.log"))
                            for r in range(ROUNDS) for p in (0, 3)]),
            "B": summarize([statistics.median(step_ms(RAW / f"s7-{wmma}-r{r}-p{p}-B.log"))
                            for r in range(ROUNDS) for p in (1, 2)]),
        },
        "verdict": "PASS" if all(r < 1.0 for r in ratios) else "FAIL",
    }


def main() -> None:
    result = {
        "protocol": {
            "warmup_steps": 3,
            "measured_steps_per_process": 26,
            "rounds": ROUNDS,
            "processes_per_round": 4,
            "aggregation": "median of two A process medians and two B process medians per round",
        },
        "series": [series("off"), series("mixed")],
    }
    for entry in result["series"]:
        entry["complete_step_speedup_estimate"] = None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
