#!/usr/bin/env python3
"""Phase XIII reduction: WMMA off vs Mixed on the production training graph.

Two independent ABBA series are reduced here, both five rounds of A B B A with
26 measured steps (three warmup steps excluded) per process:

* `abba-r*-p*-{A,B}.log`  the unmodified Phase XII release step harness.
                          Timing only; its own counters prove the attention
                          route, not the WMMA route.
* `stab-r*-p*-{A,B}.log`  `training-stability --digest-every 0`, same fixture
                          and hyperparameters. Every step prints its own
                          forward / weight-gradient / input-gradient WMMA
                          dispatch deltas, so this series carries the per-arm
                          dispatch proof inside each timed process.

Arm A is `HIERARCHOS_WMMA_MODE=off`; arm B is `HIERARCHOS_WMMA_MODE=mixed`.
Nothing else differs, and both arms run the production channel4 key pass
(`HIERARCHOS_PHASE_XII_STEP_ARM=channel4` for the step harness).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def ci95(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [round(values[0], 6), round(values[0], 6)] if values else [0.0, 0.0]
    mean = statistics.fmean(values)
    half = T975.get(len(values), 2.0) * statistics.stdev(values) / (len(values) ** 0.5)
    return [round(mean - half, 6), round(mean + half, 6)]


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return round(ordered[index], 6)


def parse_step_harness(path: Path) -> dict:
    steps: list[float] = []
    end: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xii-step]" in line:
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            if fields.get("measured") == "true":
                steps.append(float(fields["step_ms"]))
        elif "[phase-xii-step-run-end]" in line:
            end = dict(token.split("=", 1) for token in line.split() if "=" in token)
    return {"measured_steps": steps, "end": end}


STAB_STEP = re.compile(r"^\{\"event\":\"qual_step\"")


def parse_stability(path: Path) -> dict:
    measured: list[float] = []
    per_step = []
    counters = {}
    mode_line = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "WMMA: mixed selected" in line:
            mode_line = line.strip()
        if STAB_STEP.match(line):
            record = json.loads(line)
            if record["global_step"] >= 3:
                measured.append(record["step_ms"])
            per_step.append(
                {
                    "step": record["global_step"],
                    "step_ms": round(record["step_ms"], 6),
                    "loss_bits": record["loss_bits"],
                    "wmma_forward": record["wmma_forward"],
                    "wmma_weight_grad": record["wmma_weight_grad"],
                    "wmma_input_grad": record["wmma_input_grad"],
                    "wmma_cooperative": record["wmma_cooperative"],
                    "wmma_shape_fallbacks": record["wmma_shape_fallbacks"],
                    "wmma_weight_conversions": record["wmma_weight_conversions"],
                    "wmma_mirror_reuses": record["wmma_mirror_reuses"],
                }
            )
        if line.startswith("{") and '"wmma_forward"' in line and "qual_step" not in line:
            pass
    return {"measured_steps": measured, "per_step": per_step, "mode_line": mode_line}


def series(raw: Path, prefix: str, rounds: int, parser) -> dict:
    per_round = []
    for index in range(rounds):
        runs = {}
        for position in range(4):
            tag = "A" if position in (0, 3) else "B"
            path = raw / f"{prefix}-r{index}-p{position}-{tag}.log"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            runs[position] = parser(path)
            if len(runs[position]["measured_steps"]) != 26:
                raise SystemExit(
                    f"{path.name}: {len(runs[position]['measured_steps'])} measured steps, expected 26"
                )
        a = [statistics.median(runs[p]["measured_steps"]) for p in (0, 3)]
        b = [statistics.median(runs[p]["measured_steps"]) for p in (1, 2)]
        pairs = [b[0] / a[0], b[1] / a[1]]
        entry = {
            "round": index,
            "arm_a_ms": [round(v, 6) for v in a],
            "arm_b_ms": [round(v, 6) for v in b],
            "arm_b_over_a": [round(v, 6) for v in pairs],
            "arm_b_over_a_mean": round(statistics.fmean(pairs), 6),
        }
        if prefix == "abba":
            entry["arm_a_dispatch"] = {
                key: runs[0]["end"].get(key)
                for key in (
                    "forward_row_packed",
                    "backward_query",
                    "backward_key_original",
                    "backward_key_channel4",
                    "backward_scalar",
                )
            }
            entry["arm_b_dispatch"] = {
                key: runs[1]["end"].get(key)
                for key in (
                    "forward_row_packed",
                    "backward_query",
                    "backward_key_original",
                    "backward_key_channel4",
                    "backward_scalar",
                )
            }
            entry["arm_a_end_digest"] = runs[0]["end"].get("parameter_end_digest")
            entry["arm_b_end_digest"] = runs[1]["end"].get("parameter_end_digest")
        else:
            steps = {tag: runs[p]["per_step"] for tag, p in (("A", 0), ("B", 1))}
            entry["arm_a_dispatch_per_step"] = [
                {k: s[k] for k in ("wmma_forward", "wmma_weight_grad", "wmma_input_grad", "wmma_cooperative", "wmma_shape_fallbacks")}
                for s in steps["A"]
            ]
            entry["arm_b_dispatch_per_step"] = [
                {k: s[k] for k in ("wmma_forward", "wmma_weight_grad", "wmma_input_grad", "wmma_cooperative", "wmma_shape_fallbacks")}
                for s in steps["B"]
            ]
            entry["arm_a_loss_bits"] = [s["loss_bits"] for s in steps["A"]]
            entry["arm_b_loss_bits"] = [s["loss_bits"] for s in steps["B"]]
        per_round.append(entry)

    ratios = [e["arm_b_over_a_mean"] for e in per_round]
    a_all = [v for e in per_round for v in e["arm_a_ms"]]
    b_all = [v for e in per_round for v in e["arm_b_ms"]]
    a_steps = [v for index in range(rounds) for p in (0, 3) for v in series_steps(raw, prefix, index, p, parser)]
    b_steps = [v for index in range(rounds) for p in (1, 2) for v in series_steps(raw, prefix, index, p, parser)]
    improved = sum(1 for v in ratios if v < 1.0)
    ratio_ci = ci95(ratios)
    return {
        "series": prefix,
        "rounds": rounds,
        "per_round": per_round,
        "arm_a_median_ms": round(statistics.median(a_all), 6),
        "arm_b_median_ms": round(statistics.median(b_all), 6),
        "arm_a_mean_ms": round(statistics.fmean(a_all), 6),
        "arm_b_mean_ms": round(statistics.fmean(b_all), 6),
        "arm_a_ci95_ms": ci95(a_all),
        "arm_b_ci95_ms": ci95(b_all),
        "arm_a_p95_ms": p95(a_steps),
        "arm_b_p95_ms": p95(b_steps),
        "arm_a_stdev_ms": round(statistics.stdev(a_steps), 6),
        "arm_b_stdev_ms": round(statistics.stdev(b_steps), 6),
        "median_round_ratio": round(statistics.median(ratios), 6),
        "mean_round_ratio": round(statistics.fmean(ratios), 6),
        "ratio_ci95": ratio_ci,
        "min_round_ratio": round(min(ratios), 6),
        "max_round_ratio": round(max(ratios), 6),
        "rounds_improved": improved,
        "rounds_total": len(ratios),
        "speedup": round(1.0 / statistics.median(ratios), 4),
        "ci_includes_one": ratio_ci[0] <= 1.0 <= ratio_ci[1],
        "verdict": (
            "PASS"
            if improved == len(ratios) and statistics.median(ratios) < 1.0 and ratio_ci[1] < 1.0
            else "INCONCLUSIVE"
            if ratio_ci[0] <= 1.0 <= ratio_ci[1]
            else "FAIL"
        ),
    }


_CACHE: dict[tuple, list[float]] = {}


def series_steps(raw: Path, prefix: str, index: int, position: int, parser) -> list[float]:
    key = (prefix, index, position)
    if key not in _CACHE:
        tag = "A" if position in (0, 3) else "B"
        _CACHE[key] = parser(raw / f"{prefix}-r{index}-p{position}-{tag}.log")["measured_steps"]
    return _CACHE[key]


def main(argv: list[str]) -> int:
    raw = Path(argv[1])
    rounds = int(argv[2]) if len(argv) > 2 else 5
    report = {
        "phase_xii_harness": series(raw, "abba", rounds, parse_step_harness),
        "stability_harness": series(raw, "stab", rounds, parse_stability),
        "full_parameter_harness": series(raw, "full", rounds, parse_stability),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
