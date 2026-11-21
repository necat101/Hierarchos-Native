#!/usr/bin/env python3
"""Phase LXXXI paired-ABBA analysis.

Same implementation as `phase_lxxi_evidence/analyze_lxxi.py`: the shared
`hierarchos-vulkan/validation/abba_stats.py` is the only interval rule
(two-sided 95 % Student-t over per-pair ratios, df = pairs - 1, computed from
the regularized incomplete beta function, no normal-quantile fallback).

Ratio convention, unchanged: reference_ms / candidate_ms, > 1.0 means the
candidate was faster.

Usage: analyze_lxxii.py <validation-dir> <abba-root> <sweep> [<sweep> ...]
"""
from __future__ import annotations

import json
import os
import sys

ARM = "wmma-full-f16acc"
PROTECTED = ("2048x768x2048", "512x2048x768", "128x8192x768", "512x768x768")


def verdict(low: float, high: float) -> str:
    if low >= 1.0:
        return "PARITY-OR-BETTER (CI lower bound >= 1.0)"
    if high < 1.0:
        return "SLOWER (CI excludes 1.0)"
    return "INCONCLUSIVE (CI includes 1.0)"


def main() -> int:
    validation_dir = sys.argv[1]
    root = sys.argv[2]
    names = sys.argv[3:]
    sys.path.insert(0, validation_dir)
    import abba_stats  # noqa: E402

    report = {}
    for name in names:
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            print(f"{name}: MISSING")
            continue
        summary = abba_stats.summarize(directory, ARM)
        rows = {}
        print(f"\n== {name}")
        print(f"   {'shape':<16}{'pairs':>6}{'ref ms':>11}{'cand ms':>11}"
              f"{'ratio':>9}{'95% t CI':>24}  verdict")
        for shape, row in summary.items():
            low, high = row["ci95_low"], row["ci95_high"]
            row["verdict"] = verdict(low, high)
            row["protected"] = shape in PROTECTED
            rows[shape] = row
            print(f"   {shape:<16}{row['pairs']:>6}{row['median_reference_ms']:>11.4f}"
                  f"{row['median_candidate_ms']:>11.4f}{row['mean_ratio']:>9.4f}"
                  f"   [{low:.4f}, {high:.4f}]  {row['verdict']}")
        missing = [shape for shape in PROTECTED if shape not in rows]
        if missing:
            print(f"   MISSING PROTECTED SHAPES: {missing}")
            return 1
        report[name] = rows

    report["statistics_implementation"] = os.path.join(validation_dir, "abba_stats.py")
    report["interval_rule"] = (
        "two-sided 95 % Student-t over per-pair ratios, df = pairs - 1, from the "
        "regularized incomplete beta function; no normal-quantile fallback"
    )
    report["ratio_convention"] = "reference_ms / candidate_ms; > 1.0 means candidate faster"
    out = os.path.join(root, "phase_lxxxi_abba_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
