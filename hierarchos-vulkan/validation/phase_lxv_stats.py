#!/usr/bin/env python3
"""Shared statistics for the Phase LXV performance evidence.

The numbers in `phase_lxv_receipt.json` are not hand-written: both the receipt
generator and the gate recompute them from the per-iteration samples that the
paired ABBA runner wrote. This module is the single implementation, so a receipt
that disagrees with its raw samples cannot pass.

The ratio convention is `reference_ms / candidate_ms`, so above 1.0 means the
candidate (`CAND`) was faster. Only rounds share a pair, so pairing is by pair
index, exactly as `wmma_abba_analyze.py` does it.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
from pathlib import Path

T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    31: 2.040, 32: 2.037,
}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def load_dir(directory: Path):
    """Return {pair: {"base": {(shape, arm): [samples]}, "cand": {...}}}."""
    runs = {}
    with open_text(directory / "index.tsv") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            pair, arm, rc, _sha, base_name = parts[:5]
            if rc != "0":
                raise ValueError("run failed: pair %s arm %s rc %s" % (pair, arm, rc))
            grouped = {}
            samples_path = directory / (Path(base_name).name + "_samples.jsonl")
            if not samples_path.exists():
                samples_path = directory / (Path(base_name).name + "_samples.jsonl.gz")
            with open_text(samples_path) as samples:
                for row in samples:
                    row = row.strip()
                    if not row:
                        continue
                    record = json.loads(row)
                    grouped.setdefault((record["shape"], record["arm"]), []).extend(
                        record["samples_s"]
                    )
            runs.setdefault(int(pair), {})[arm] = grouped
    return runs


def mean_ci(values):
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, float("nan"), float("nan")
    half = T95.get(n - 1, 1.96) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def summarize(directory: Path, arm: str):
    """Per shape: {median_base_ms, median_cand_ms, mean_ratio, lower, upper, geomean}."""
    runs = load_dir(directory)
    pairs = sorted(runs)
    keys = set()
    for pair in pairs:
        for arm_map in runs[pair].values():
            keys.update(arm_map)
    shapes = sorted({shape for shape, _ in keys})
    out = {}
    for shape in shapes:
        base, cand, ratios = [], [], []
        for pair in pairs:
            b = runs[pair].get("base", {}).get((shape, arm))
            c = runs[pair].get("cand", {}).get((shape, arm))
            if not b or not c:
                continue
            base.append(statistics.median(b))
            cand.append(statistics.median(c))
            ratios.append(statistics.median(b) / statistics.median(c))
        if not ratios:
            continue
        mean, low, high = mean_ci(ratios)
        out[shape] = {
            "pairs": len(ratios),
            "median_reference_ms": statistics.median(base) * 1.0e3,
            "median_candidate_ms": statistics.median(cand) * 1.0e3,
            "mean_ratio": mean,
            "ci95_low": low,
            "ci95_high": high,
            "geomean_ratio": math.exp(statistics.fmean([math.log(r) for r in ratios])),
        }
    return out
