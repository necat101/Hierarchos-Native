#!/usr/bin/env python3
"""Reduce the Phase XI end-to-end ABBA logs.

Every process prints one `[phase-xi-step]` line per step and a
`[phase-xi-step-run-end]` line. The measured window is the steps with
`measured=true` (three warmup steps are excluded), and a run's value is the
median step time over that window.

ABBA is four positions per round: A B B A. Each round therefore contributes two
arm-A medians and two arm-B medians; the round's ratio is the mean of the two
B/A pairings. The report keeps the per-round ratios, because the point of the
design is that a slow machine or a cool GPU shows up as a round-level shift that
affects both arms, not as a difference between them.

Usage: abba.py RAW_DIR ROUNDS ARM_A ARM_B > abba.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

STEP = "[phase-xi-step]"
RUN_END = "[phase-xi-step-run-end]"


def parse_run(path: Path) -> dict:
    steps: list[float] = []
    arm = None
    end = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if STEP in line:
            fields = dict(
                token.split("=", 1) for token in line.split() if "=" in token
            )
            arm = fields.get("arm", arm)
            if fields.get("measured") == "true":
                steps.append(float(fields["step_ms"]))
        elif RUN_END in line:
            end = dict(
                token.split("=", 1) for token in line.split() if "=" in token
            )
    return {"arm": arm, "measured_steps": steps, "end": end}


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
        a = [statistics.median(runs[p]["measured_steps"]) for p in (0, 3)]
        b = [statistics.median(runs[p]["measured_steps"]) for p in (1, 2)]
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
                    for key in ("backward_scalar", "backward_query", "backward_key")
                },
                "arm_b_dispatch": {
                    key: runs[1]["end"].get(key)
                    for key in ("backward_scalar", "backward_query", "backward_key")
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
        "arm_a_median_ms": round(statistics.median(a_all), 6),
        "arm_b_median_ms": round(statistics.median(b_all), 6),
        "median_round_ratio": round(statistics.median(ratios), 6),
        "min_round_ratio": round(min(ratios), 6),
        "max_round_ratio": round(max(ratios), 6),
        "rounds_improved": improved,
        "rounds_total": len(ratios),
        "speedup": round(1.0 / statistics.median(ratios), 4),
        "verdict": "PASS" if improved == len(ratios) and statistics.median(ratios) < 1.0 else "FAIL",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
