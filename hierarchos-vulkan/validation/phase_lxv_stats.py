#!/usr/bin/env python3
"""Shared statistics for the Phase LXV performance evidence.

The numbers in `phase_lxv_receipt.json` are not hand-written: both the receipt
generator and the gate recompute them from the per-iteration samples that the
paired ABBA runner wrote. This module is the single implementation, so a receipt
that disagrees with its raw samples cannot pass.

The implementation itself now lives in `abba_stats.py`, which is shared with
`wmma_abba_analyze.py` and any future qualification tool. This module keeps the
names the Phase LXV receipt generator and gate already import, so there is
exactly one calculation and no second copy of the t table. The old local table
(which extended to 32 degrees of freedom and then fell back to `1.96`) is gone:
`t_critical` is computed and refuses out-of-range sample counts.

The ratio convention is `reference_ms / candidate_ms`, so above 1.0 means the
candidate (`CAND`) was faster. Only rounds share a pair, so pairing is by pair
index, exactly as `wmma_abba_analyze.py` does it.
"""
from __future__ import annotations

from pathlib import Path

import abba_stats

StatisticsError = abba_stats.StatisticsError
t_critical = abba_stats.t_critical
mean_ci_normal = abba_stats.mean_ci_normal


def open_text(path: Path):
    return abba_stats.open_text(path)


def load_dir(directory: Path):
    """Return {pair: {"base": {(shape, arm): [samples]}, "cand": {...}}}."""
    return abba_stats.load_dir(directory)


def mean_ci(values):
    return abba_stats.mean_ci(values)


def summarize(directory: Path, arm: str):
    """Per shape: {median_base_ms, median_cand_ms, mean_ratio, lower, upper, geomean}."""
    return abba_stats.summarize(directory, arm)
