#!/usr/bin/env python3
"""Phase XIX complete-step reduction: per-round paired ratios and a CI.

Same protocol and estimator as Phase XVIII, with the Phase XIX log tags. Reads
the `[phase-xix-step]` lines the step harness prints, keeps only the measured
steps (`measured=true`, i.e. after the three warmups), and reports, for every
geometry and precision mode independently:

* each round's A and B medians and the B/A ratio,
* the median round ratio and its 95 % CI over the five rounds,
* the complete-step milliseconds of both arms,
* how many rounds improved, and
* the production dispatch counters each process reported.

Nothing is pooled across geometries or modes.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

T_CRIT = {5: 2.776, 4: 3.182}
COUNTER_RE = re.compile(r"(\w+)=(-?[0-9.]+)")


def step_ms(path: Path) -> list[float]:
    out: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xix-step]" not in line:
            continue
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if fields.get("measured") != "true":
            continue
        out.append(float(fields["step_ms"]))
    return out


def end_counters(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xix-step-run-end]" in line:
            return {
                key: value
                for key, value in COUNTER_RE.findall(line)
                if key.startswith("backward_") or key.startswith("forward_") or key == "arm"
            }
    raise SystemExit(f"{path.name}: no [phase-xix-step-run-end] line")


def seam_arm(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xix-step-seam]" in line:
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            return f"{fields.get('arm')}/{fields.get('sha256_prefix')}"
    return "unknown"


def summarize(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def series(raw: Path, mode: str, name: str, rounds: int) -> dict:
    round_rows = []
    for round_index in range(rounds):
        a_runs, b_runs = [], []
        for position in range(4):
            tag = "A" if position in (0, 3) else "B"
            path = raw / f"s-{mode}-{name}-r{round_index}-p{position}-{tag}.log"
            samples = step_ms(path)
            if len(samples) != 26:
                raise SystemExit(f"{path.name}: {len(samples)} measured steps, expected 26")
            (a_runs if tag == "A" else b_runs).append(statistics.median(samples))
        a = statistics.median(a_runs)
        b = statistics.median(b_runs)
        round_rows.append(
            {
                "round": round_index,
                "a_runs_ms": a_runs,
                "b_runs_ms": b_runs,
                "a_ms": a,
                "b_ms": b,
                "ratio": b / a,
            }
        )

    ratios = [row["ratio"] for row in round_rows]
    median = statistics.median(ratios)
    mean = statistics.fmean(ratios)
    stdev = statistics.stdev(ratios)
    half = T_CRIT.get(rounds, 2.776) * stdev / (len(ratios) ** 0.5)
    arm_a = [end_counters(raw / f"s-{mode}-{name}-r{r}-p{p}-A.log") for r in range(rounds) for p in (0, 3)]
    arm_b = [end_counters(raw / f"s-{mode}-{name}-r{r}-p{p}-B.log") for r in range(rounds) for p in (1, 2)]
    seam = {
        "A": sorted({seam_arm(raw / f"s-{mode}-{name}-r{r}-p{p}-A.log") for r in range(rounds) for p in (0, 3)}),
        "B": sorted({seam_arm(raw / f"s-{mode}-{name}-r{r}-p{p}-B.log") for r in range(rounds) for p in (1, 2)}),
    }
    run_medians = {
        "A": [round(r["a_runs_ms"][0], 6) for r in round_rows],
        "B": [round(r["b_runs_ms"][0], 6) for r in round_rows],
        "A_second": [round(r["a_runs_ms"][1], 6) for r in round_rows],
        "B_second": [round(r["b_runs_ms"][1], 6) for r in round_rows],
    }
    position_medians = {
        "pos0": round(statistics.median([r["a_runs_ms"][0] for r in round_rows]), 6),
        "pos1": round(statistics.median([r["b_runs_ms"][0] for r in round_rows]), 6),
        "pos2": round(statistics.median([r["b_runs_ms"][1] for r in round_rows]), 6),
        "pos3": round(statistics.median([r["a_runs_ms"][1] for r in round_rows]), 6),
    }
    all_runs = run_medians["A"] + run_medians["B"] + run_medians["A_second"] + run_medians["B_second"]
    ci_lo, ci_hi = mean - half, mean + half
    return {
        "mode": mode,
        "series": name,
        "rounds": round_rows,
        "round_ratios": ratios,
        "median_round_ratio": median,
        "mean_round_ratio": mean,
        "ratio_stdev": stdev,
        "ratio_ci95": [ci_lo, ci_hi],
        "speedup": 1.0 / median,
        "rounds_improved": sum(1 for r in ratios if r < 1.0),
        "rounds_total": len(ratios),
        "median_step_ms": {
            "A": statistics.median([row["a_ms"] for row in round_rows]),
            "B": statistics.median([row["b_ms"] for row in round_rows]),
        },
        "arm": {
            "A": summarize([statistics.median(step_ms(raw / f"s-{mode}-{name}-r{r}-p{p}-A.log"))
                            for r in range(rounds) for p in (0, 3)]),
            "B": summarize([statistics.median(step_ms(raw / f"s-{mode}-{name}-r{r}-p{p}-B.log"))
                            for r in range(rounds) for p in (1, 2)]),
        },
        "counters": {
            "A": sorted({tuple(sorted(entry.items())) for entry in arm_a}),
            "B": sorted({tuple(sorted(entry.items())) for entry in arm_b}),
        },
        "seam": seam,
        "run_medians_ms": run_medians,
        "position_medians_ms": position_medians,
        "run_range_ms": [min(all_runs), max(all_runs)],
        "run_spread_ratio": max(all_runs) / min(all_runs),
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: reduce_step.py <raw-dir> <rounds> [mode:series ...]")
    raw = Path(sys.argv[1]).resolve()
    rounds = int(sys.argv[2])
    keys = sys.argv[3:]
    result = {
        "protocol": {
            "warmup_steps": 3,
            "measured_steps_per_process": 26,
            "rounds": rounds,
            "processes_per_round": 4,
            "schedule": "A B B A",
            "arm_a": "Phase XVIII promoted dq-unroll pass 1 (f0214b60)",
            "arm_b": "same promoted path, scratch-reuse module (a6b3d6c6)",
            "aggregation": "median of two A process medians and two B process medians per round",
        },
        "series": [],
    }
    for key in keys:
        mode, name = key.split(":", 1)
        result["series"].append(series(raw, mode, name, rounds))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
