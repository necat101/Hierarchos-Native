#!/usr/bin/env python3
"""Reduce the Phase XII release-mode ABBA logs.

Every process prints one `[phase-xii-step]` line per step and one
`[phase-xii-step-run-end]` line. The measured window is the steps with
`measured=true` (three warmup steps are excluded), and a run's value is the
median step time over that window.

ABBA is four positions per round: A B B A. Each round contributes two arm-A
medians and two arm-B medians; the round ratio is the mean of the two B/A
pairings. Per-round ratios are kept because a slow machine or a cool GPU shows
up as a round-level shift that affects both arms, not as a difference between
them.

Arm A is the original production key shader
(`HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0`). Arm B is the promoted
four-channel key pass (variable unset). Both are separate processes and both
run the production selector; the dispatch counters in each `run-end` line prove
which key pass actually executed.

Usage: abba.py RAW_DIR ROUNDS ARM_A ARM_B > abba.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

STEP = "[phase-xii-step]"
RUN_END = "[phase-xii-step-run-end]"

# Two-sided 95% t quantile for the small round counts this protocol uses.
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def parse_run(path: Path) -> dict:
    steps: list[float] = []
    arm = None
    end: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if STEP in line:
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            arm = fields.get("arm", arm)
            if fields.get("measured") == "true":
                steps.append(float(fields["step_ms"]))
        elif RUN_END in line:
            end = dict(token.split("=", 1) for token in line.split() if "=" in token)
    return {"arm": arm, "measured_steps": steps, "end": end}


def median(values: list[float]) -> float:
    return statistics.median(values)


def ci95(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]] if values else [0.0, 0.0]
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    half = T975.get(len(values), 2.0) * stdev / (len(values) ** 0.5)
    return [round(mean - half, 6), round(mean + half, 6)]


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print("usage: abba.py RAW_DIR ROUNDS ARM_A ARM_B", file=sys.stderr)
        return 2
    raw = Path(argv[1])
    rounds = int(argv[2])
    arm_a = argv[3]
    arm_b = argv[4]

    per_round = []
    for index in range(rounds):
        runs = {}
        for position in range(4):
            tag = "A" if position in (0, 3) else "B"
            path = raw / f"abba-r{index}-p{position}-{tag}.log"
            if not path.exists():
                print(f"missing {path}", file=sys.stderr)
                return 1
            runs[position] = parse_run(path)
            if len(runs[position]["measured_steps"]) != 26:
                print(
                    f"r{index} p{position}: {len(runs[position]['measured_steps'])} "
                    "measured steps, expected 26",
                    file=sys.stderr,
                )
                return 1
        a = [median(runs[p]["measured_steps"]) for p in (0, 3)]
        b = [median(runs[p]["measured_steps"]) for p in (1, 2)]
        pairs = [b[0] / a[0], b[1] / a[1]]
        per_round.append(
            {
                "round": index,
                "arm_a_ms": [round(value, 6) for value in a],
                "arm_b_ms": [round(value, 6) for value in b],
                "arm_b_over_a": [round(value, 6) for value in pairs],
                "arm_b_over_a_mean": round(statistics.fmean(pairs), 6),
                "arm_a_steps": len(runs[0]["measured_steps"]),
                "arm_b_steps": len(runs[1]["measured_steps"]),
                "arm_a_end_digest": runs[0]["end"].get("parameter_end_digest"),
                "arm_b_end_digest": runs[1]["end"].get("parameter_end_digest"),
                "arm_a_dispatch": {
                    key: runs[0]["end"].get(key)
                    for key in (
                        "backward_scalar",
                        "backward_query",
                        "backward_key_original",
                        "backward_key_channel4",
                    )
                },
                "arm_b_dispatch": {
                    key: runs[1]["end"].get(key)
                    for key in (
                        "backward_scalar",
                        "backward_query",
                        "backward_key_original",
                        "backward_key_channel4",
                    )
                },
            }
        )

    ratios = [entry["arm_b_over_a_mean"] for entry in per_round]
    a_all = [value for entry in per_round for value in entry["arm_a_ms"]]
    b_all = [value for entry in per_round for value in entry["arm_b_ms"]]
    improved = sum(1 for value in ratios if value < 1.0)
    report = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "rounds": rounds,
        "positions_per_round": 4,
        "per_round": per_round,
        "arm_a_median_ms": round(median(a_all), 6),
        "arm_b_median_ms": round(median(b_all), 6),
        "arm_a_mean_ms": round(statistics.fmean(a_all), 6),
        "arm_b_mean_ms": round(statistics.fmean(b_all), 6),
        "arm_a_ci95_ms": ci95(a_all),
        "arm_b_ci95_ms": ci95(b_all),
        "median_round_ratio": round(median(ratios), 6),
        "mean_round_ratio": round(statistics.fmean(ratios), 6),
        "ratio_ci95": ci95(ratios),
        "min_round_ratio": round(min(ratios), 6),
        "max_round_ratio": round(max(ratios), 6),
        "rounds_improved": improved,
        "rounds_total": len(ratios),
        "speedup": round(1.0 / median(ratios), 4),
        "verdict": "PASS" if improved == len(ratios) and median(ratios) < 1.0 else "FAIL",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
