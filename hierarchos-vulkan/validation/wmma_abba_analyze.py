#!/usr/bin/env python3
"""Paired ABBA analysis for `wmma_abba_run.sh` output.

For every (shape, arm) this reports the per-pair pooled median, the paired
ratio candidate/baseline, the mean ratio with its 95% paired-t interval, and the
geometric mean.

A ratio above 1.0 means the candidate executable produced the *faster* GEMM, so
the printed ratio is `baseline_ms / candidate_ms`. The `fp32-baseline` arm is a
negative control: its kernel is byte-identical in both executables, so its ratio
should straddle 1.0 and bounds the run-to-run noise floor.

The statistics live in `abba_stats.py`, which is also what `phase_lxv_stats.py`
and every newer qualification tool use. This module used to carry its own
Student-t table that stopped at 12 degrees of freedom and fell back to
`1.96` above it, which made every interval with 13 or more pairs too narrow. It
now shares the single implementation: computed t quantiles, no normal
approximation, and a hard error instead of a silent fallback.

Usage: python wmma_abba_analyze.py <OUT_DIR>
"""
import math
import statistics
import sys

import abba_stats

ARMS = ("fp32-baseline", "plain-f16-storage-f32acc", "wmma-mixed-f32acc")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: wmma_abba_analyze.py <OUT_DIR>")
    runs = abba_stats.load_dir(sys.argv[1])
    pairs = sorted(runs)
    print("pairs: %d" % len(pairs))

    keys = set()
    for pair in pairs:
        for arm_map in runs[pair].values():
            keys.update(arm_map)
    shapes = sorted({shape for shape, _ in keys})

    print()
    print("%-16s %-24s %9s %9s %8s %18s %8s" % (
        "shape", "arm", "base_ms", "cand_ms", "ratio", "95% CI", "geomean"))
    print("-" * 98)
    verdicts = {}
    for shape in shapes:
        for arm in ARMS:
            base, cand, ratios = [], [], []
            for pair in pairs:
                b = runs[pair].get("base", {}).get((shape, arm))
                c = runs[pair].get("cand", {}).get((shape, arm))
                if not b or not c:
                    continue
                mb, mc = statistics.median(b), statistics.median(c)
                base.append(mb)
                cand.append(mc)
                ratios.append(mb / mc)
            if not ratios:
                continue
            mean, lo, hi = abba_stats.mean_ci(ratios)
            geo = math.exp(statistics.fmean([math.log(r) for r in ratios]))
            print("%-16s %-24s %9.4f %9.4f %8.4f  [%6.3f, %6.3f] %8.4f" % (
                shape, arm, statistics.median(base) * 1e3,
                statistics.median(cand) * 1e3, mean, lo, hi, geo))
            if arm == "wmma-mixed-f32acc":
                verdicts[shape] = (geo, lo, hi)
        print()

    print("verdicts (candidate vs baseline, wmma-mixed arm):")
    for shape, (geo, lo, hi) in verdicts.items():
        if lo > 1.0:
            label = "SPEEDUP (CI excludes 1.0)"
        elif hi < 1.0:
            label = "REGRESSION (CI excludes 1.0)"
        else:
            label = "INCONCLUSIVE (CI includes 1.0)"
        print("  %-16s geomean %.4fx  CI [%.3f, %.3f]  %s" % (shape, geo, lo, hi, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
