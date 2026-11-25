#!/usr/bin/env python3
"""Phase IX statistics: paired per-run kernel ratios and their confidence interval.

Each timing run is one process. Inside a process every arm is measured as
`warmup + iterations` single-dispatch submissions, so a run yields one median
per arm. The paired ratio for a geometry is `generic / candidate` from the two
medians of the *same* process, which removes the process-to-process drift that
the arm-order control measures. The confidence interval is the Student-t
interval over the per-run paired ratios.

This script reads only the JSON that `analyze_timing.py` produced. A run without
profiler output contributes no ratio.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def t_critical(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        24: 2.064, 25: 2.060, 30: 2.042,
    }
    if df in table:
        return table[df]
    nearest = min(table, key=lambda key: abs(key - df))
    return table[nearest]


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = statistics.stdev(values)
    half = t_critical(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: assemble.py TIMING_GPU_JSON", file=sys.stderr)
        return 2
    raw = json.loads(Path(argv[1]).read_text(encoding="utf-8"))

    # run name -> geometry -> arm -> entry
    runs: dict[str, dict[str, dict[str, dict]]] = {}
    for key, entry in raw.items():
        run = key.split("|", 1)[0]
        runs.setdefault(run, {}).setdefault(entry["geometry"], {})[entry["arm"]] = entry

    comparisons = {
        "primary_mode9_packed_vs_generic": ("gqa-8q-2kv-primary-mode9", "forward_rowpacked64cached", "forward_generic"),
        "primary_mode9_productiongrid_vs_generic": ("gqa-8q-2kv-primary-mode9", "forward_rowpacked64cached_productiongrid", "forward_generic"),
        "primary_mode0_packed_vs_generic": ("gqa-8q-2kv-primary-mode0", "forward_rowpacked64cached", "forward_generic"),
        "mha_mode9_packed_vs_generic": ("mha-8q-8kv-control", "forward_rowpacked64cached", "forward_generic"),
        "primary_mode0_stream_vs_generic": ("gqa-8q-2kv-primary-mode0", "forward_rowpacked64stream", "forward_generic"),
    }

    report: dict = {"runs": sorted(runs), "comparisons": {}, "arms": {}}
    for name, (geometry, candidate, baseline) in comparisons.items():
        ratios = []
        per_run = []
        for run in sorted(runs):
            arms = runs[run].get(geometry, {})
            if baseline not in arms or candidate not in arms:
                continue
            base = arms[baseline]
            cand = arms[candidate]
            ratio = base["median_gpu_ms"] / cand["median_gpu_ms"]
            ratios.append(ratio)
            per_run.append({
                "run": run,
                "baseline_median_gpu_ms": base["median_gpu_ms"],
                "candidate_median_gpu_ms": cand["median_gpu_ms"],
                "baseline_samples": base["samples"],
                "candidate_samples": cand["samples"],
                "ratio": round(ratio, 4),
            })
        if not ratios:
            continue
        mean, low, high = mean_ci(ratios)
        report["comparisons"][name] = {
            "geometry": geometry,
            "baseline_arm": baseline,
            "candidate_arm": candidate,
            "runs": len(ratios),
            "samples_per_arm_per_run": 25,
            "paired_ratios": [round(value, 4) for value in ratios],
            "mean_paired_ratio": round(mean, 4),
            "geomean_paired_ratio": round(statistics.geometric_mean(ratios), 4),
            "ci95": {
                "low": round(low, 4),
                "high": round(high, 4),
                "df": len(ratios) - 1,
                "t_critical": t_critical(len(ratios) - 1),
            },
            "per_run": per_run,
        }

    for run in sorted(runs):
        for geometry, arms in sorted(runs[run].items()):
            for arm, entry in sorted(arms.items()):
                report["arms"][f"{run}|{geometry}|{arm}"] = {
                    "workgroups": entry["workgroups"],
                    "lanes": entry["lanes"],
                    "samples": entry["samples"],
                    "median_gpu_ms": entry["median_gpu_ms"],
                    "mean_gpu_ms": entry["mean_gpu_ms"],
                    "stdev_gpu_ms": entry["stdev_gpu_ms"],
                    "p25_gpu_ms": entry["p25_gpu_ms"],
                    "min_gpu_ms": entry["min_gpu_ms"],
                    "max_gpu_ms": entry["max_gpu_ms"],
                    "median_host_ms": entry.get("host", {}).get("median_host_ms"),
                    "shader_labels": entry["shader_labels"],
                }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
