#!/usr/bin/env python3
"""Paired ABBA analysis for `wmma_abba_run.sh` output.

For every (shape, arm) this reports the per-pair pooled median, the paired
ratio candidate/baseline, the mean ratio with its 95% paired-t interval, and the
geometric mean.

A ratio above 1.0 means the candidate executable produced the *faster* GEMM, so
the printed ratio is `baseline_ms / candidate_ms`. The `fp32-baseline` arm is a
negative control: its kernel is byte-identical in both executables, so its ratio
should straddle 1.0 and bounds the run-to-run noise floor.

Usage: python wmma_abba_analyze.py <OUT_DIR>
"""
import json
import math
import os
import statistics
import sys

# Two-tailed 95% t quantiles, df = 1..12.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
}
ARMS = ("fp32-baseline", "plain-f16-storage-f32acc", "wmma-mixed-f32acc")


def load_runs(out_dir):
    runs = {}
    with open(os.path.join(out_dir, "index.tsv"), encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            pair, arm, rc, _sha, base_name = parts[:5]
            if rc != "0":
                raise SystemExit("run failed: pair %s arm %s rc %s" % (pair, arm, rc))
            grouped = {}
            with open(base_name + "_samples.jsonl", encoding="utf-8") as samples:
                for row in samples:
                    row = row.strip()
                    if not row:
                        continue
                    record = json.loads(row)
                    key = (record["shape"], record["arm"])
                    grouped.setdefault(key, []).extend(record["samples_s"])
            runs.setdefault(int(pair), {})[arm] = grouped
    return runs


def mean_ci(values):
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, float("nan"), float("nan")
    half = T95.get(n - 1, 1.96) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: wmma_abba_analyze.py <OUT_DIR>")
    runs = load_runs(sys.argv[1])
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
            mean, lo, hi = mean_ci(ratios)
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
