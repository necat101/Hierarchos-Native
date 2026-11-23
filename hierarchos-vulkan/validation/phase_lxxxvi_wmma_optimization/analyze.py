#!/usr/bin/env python3
"""Phase LXXXVI paired-ABBA analysis.

Identical rule to every previous phase: `hierarchos-vulkan/validation/abba_stats.py`
is the only interval implementation (two-sided 95 % Student-t over per-pair
ratios, df = pairs - 1, from the regularized incomplete beta function, no
normal-quantile fallback). Ratio convention unchanged:
reference_ms / candidate_ms, so > 1.0 means the candidate was faster.

Usage: analyze.py <abba-root> <sweep> [<sweep> ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
sys.path.insert(0, str(VALIDATION))

import abba_stats  # noqa: E402

ARM = "wmma-full-f16acc"
PROTECTED = ("2048x768x2048", "512x2048x768", "512x768x768", "128x8192x768")
CONTROLS = ("64x64x64", "16x128x32")


def verdict(low: float, high: float) -> str:
    if low >= 1.0:
        return "FASTER (CI lower bound >= 1.0)"
    if high < 1.0:
        return "SLOWER (CI excludes 1.0)"
    return "INCONCLUSIVE (CI includes 1.0)"


def main() -> int:
    root = sys.argv[1]
    names = sys.argv[2:]

    report = {}
    for name in names:
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            print(f"{name}: MISSING")
            continue
        summary = abba_stats.summarize(directory, ARM)
        rows = {}
        print(f"\n== {name}")
        print(
            f"   {'shape':<16}{'pairs':>6}{'base ms':>10}{'cand ms':>10}"
            f"{'speedup':>9}{'95% t CI':>22}  verdict"
        )
        for shape, row in summary.items():
            low, high = row["ci95_low"], row["ci95_high"]
            row["verdict"] = verdict(low, high)
            row["protected"] = shape in PROTECTED
            row["control"] = shape in CONTROLS
            rows[shape] = row
            print(
                f"   {shape:<16}{row['pairs']:>6}{row['median_reference_ms']:>10.4f}"
                f"{row['median_candidate_ms']:>10.4f}{row['mean_ratio']:>9.4f}"
                f"   [{low:.4f}, {high:.4f}]  {row['verdict']}"
            )
        missing = [shape for shape in PROTECTED if shape not in rows]
        if missing:
            print(f"   MISSING PROTECTED SHAPES: {missing}")
            return 1
        report[name] = rows

    report["statistics_implementation"] = str(VALIDATION / "abba_stats.py")
    report["interval_rule"] = (
        "two-sided 95 % Student-t over per-pair ratios, df = pairs - 1, from the "
        "regularized incomplete beta function; no normal-quantile fallback"
    )
    report["ratio_convention"] = "reference_ms / candidate_ms; > 1.0 means candidate faster"
    out = os.path.join(root, "phase_lxxxvi_abba_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
