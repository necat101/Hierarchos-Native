#!/usr/bin/env python3
"""Canonical paired-ABBA statistics for every performance qualification gate.

This is the single implementation of the paired ratio analysis. Both
`wmma_abba_analyze.py` (the historical command-line analyzer) and
`phase_lxv_stats.py` (the receipt/gate module) import it, so a future tool
cannot silently grow a second, different interval rule.

What the historical analyzers got wrong
---------------------------------------
`wmma_abba_analyze.py` carried a two-tailed 95 % Student-t table that stopped at
12 degrees of freedom and then did `T95.get(n - 1, 1.96)`. Above 12 pairs that
silently substitutes the *normal* quantile for a finite-sample t quantile, which
makes every interval too narrow. `phase_lxv_stats.py` extended the table to 32
degrees of freedom but kept the same `1.96` fallback, so it was correct for the
24-pair Phase LXV sweep and wrong again for 34 or more pairs.

This module has no table and no fallback. `t_critical` computes the Student-t
quantile from the regularized incomplete beta function and refuses (raises)
outside an explicitly supported degrees-of-freedom range.

Conventions preserved from the existing tools
---------------------------------------------
* A ratio is `reference_median_ms / candidate_median_ms`, so > 1.0 means the
  candidate was faster.
* Each arm's statistic for a pair is the median of that run's per-iteration
  samples; pairing is by pair index.
* The interval is the two-sided Student-t interval over the per-pair ratios
  (arithmetic mean, sample standard deviation, `df = pairs - 1`).
* The reported "geomean" is the geometric mean of the same per-pair ratios.
  Both statistics are documented here because switching between them changes
  what a 1 % difference means.
* Both plain `.jsonl` and gzip-compressed `.jsonl.gz` sample files load.

Fail-closed rules
-----------------
Every condition that would previously have produced a plausible-looking number
raises `StatisticsError` instead: a failed run, a duplicate pair identifier, a
pair missing an arm, an empty sample list, a non-positive or non-finite timing,
malformed JSON, or a sample count outside the supported range.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
from pathlib import Path

# Degrees of freedom the interval implementation is validated for. Beyond this
# the quantile still converges to the normal value, but the caller has to say so
# explicitly rather than receive a silent 1.96.
MIN_DF = 1
MAX_DF = 100_000

# Quadrature-free exact evaluation: the incomplete beta continued fraction
# converges quickly for the parameters used here, but keep a hard iteration cap
# so a pathological input fails instead of hanging.
_BETA_EPS = 3.0e-16
_BETA_MAX_ITER = 500
_BISECTION_ITER = 200


class StatisticsError(ValueError):
    """Raised when an input cannot support a defensible interval."""


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, _BETA_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETA_EPS:
            return h
    raise StatisticsError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function."""
    if not (0.0 <= x <= 1.0):
        raise StatisticsError(f"x outside [0, 1]: {x!r}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """CDF of Student's t for `df` degrees of freedom."""
    if df <= 0:
        raise StatisticsError(f"degrees of freedom must be positive, got {df!r}")
    if math.isinf(t):
        return 0.0 if t < 0 else 1.0
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_critical(df: int, confidence: float = 0.95) -> float:
    """Two-sided Student-t critical value for `df` and `confidence`.

    Computed, not tabulated, and never approximated by a normal quantile for a
    finite sample count. Raises outside `[MIN_DF, MAX_DF]` or for a confidence
    that is not strictly inside (0, 1).
    """
    if not isinstance(df, int):
        raise StatisticsError(f"degrees of freedom must be an integer, got {df!r}")
    if df < MIN_DF or df > MAX_DF:
        raise StatisticsError(
            f"degrees of freedom {df} outside the supported range "
            f"[{MIN_DF}, {MAX_DF}]; refusing to substitute a normal quantile"
        )
    if not (0.0 < confidence < 1.0):
        raise StatisticsError(f"confidence must be in (0, 1), got {confidence!r}")
    target = 1.0 - (1.0 - confidence) / 2.0
    lo, hi = 0.0, 1.0
    # Grow the bracket until the CDF clears the target.
    while student_t_cdf(hi, df) < target:
        hi *= 2.0
        if hi > 1.0e12:
            raise StatisticsError("failed to bracket the t quantile")
    for _ in range(_BISECTION_ITER):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1.0e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def mean_ci(values, confidence: float = 0.95):
    """Two-sided t interval for the mean of `values`.

    Returns `(mean, low, high)`. Fewer than two observations, or a non-finite
    observation, raises: an interval over one sample is not an interval.
    """
    values = list(values)
    n = len(values)
    if n < 2:
        raise StatisticsError(
            f"a t interval needs at least 2 paired observations, got {n}"
        )
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise StatisticsError(f"non-finite or non-numeric observation: {value!r}")
    mean = statistics.fmean(values)
    half = t_critical(n - 1, confidence) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def mean_ci_normal(values):
    """The old, incorrect rule, retained only to show what changed.

    This is what `T95.get(n - 1, 1.96)` produced above 12 pairs. It is not used
    by any gate; the historical-correction record calls it to print the
    superseded intervals beside the corrected ones.
    """
    values = list(values)
    n = len(values)
    if n < 2:
        raise StatisticsError("need at least two observations")
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _check_samples(path: Path, samples, require_positive: bool) -> list:
    if not isinstance(samples, list) or not samples:
        raise StatisticsError(f"{path.name}: empty or non-list samples_s")
    out = []
    for value in samples:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise StatisticsError(f"{path.name}: non-numeric sample {value!r}")
        if not math.isfinite(value):
            raise StatisticsError(f"{path.name}: non-finite sample {value!r}")
        if require_positive and value <= 0.0:
            raise StatisticsError(f"{path.name}: non-positive timing {value!r}")
        out.append(float(value))
    return out


def load_dir(directory, require_positive: bool = True):
    """Load an ABBA output directory into `{pair: {arm: {(shape, arm): samples}}}`.

    `directory/index.tsv` is authoritative: one row per (pair, arm) with the
    process exit code and the executable hash. Anything that would make a
    silently wrong interval possible is an error, not a skipped row.
    """
    directory = Path(directory)
    index = directory / "index.tsv"
    if not index.exists():
        raise StatisticsError(f"no index.tsv in {directory}")
    runs: dict[int, dict[str, dict]] = {}
    seen: set[tuple[int, str]] = set()
    with open(index, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            pair_text, arm, rc, _sha, base_name = parts[:5]
            try:
                pair = int(pair_text)
            except ValueError as error:
                raise StatisticsError(f"bad pair identifier {pair_text!r}") from error
            if arm not in ("base", "cand"):
                raise StatisticsError(f"pair {pair}: unknown arm {arm!r}")
            if (pair, arm) in seen:
                raise StatisticsError(f"pair {pair} arm {arm} appears twice")
            seen.add((pair, arm))
            if rc != "0":
                raise StatisticsError(f"run failed: pair {pair} arm {arm} rc {rc}")
            samples_path = directory / (Path(base_name).name + "_samples.jsonl")
            if not samples_path.exists():
                samples_path = directory / (
                    Path(base_name).name + "_samples.jsonl.gz"
                )
            if not samples_path.exists():
                raise StatisticsError(f"pair {pair} arm {arm}: no sample file")
            grouped: dict[tuple[str, str], list] = {}
            with open_text(samples_path) as samples:
                for row in samples:
                    row = row.strip()
                    if not row:
                        continue
                    try:
                        record = json.loads(row)
                    except json.JSONDecodeError as error:
                        raise StatisticsError(
                            f"{samples_path.name}: malformed JSON"
                        ) from error
                    if "shape" not in record or "arm" not in record or "samples_s" not in record:
                        raise StatisticsError(
                            f"{samples_path.name}: record missing shape/arm/samples_s"
                        )
                    key = (record["shape"], record["arm"])
                    value = _check_samples(samples_path, record["samples_s"], require_positive)
                    grouped.setdefault(key, []).extend(value)
            runs.setdefault(pair, {})[arm] = grouped

    for pair in sorted(runs):
        for arm in ("base", "cand"):
            if arm not in runs[pair]:
                raise StatisticsError(f"pair {pair} is missing its {arm} arm")
    return runs


def paired_ratios(runs, arm: str):
    """Per-shape paired ratios, plus the medians of each arm.

    Returns `{shape: {"ratios": [...], "reference_ms": [...], "candidate_ms": [...]}}`.
    Pairs that do not contain the requested shape/arm in both arms raise: a
    missing pair would silently change the sample size of the interval.
    """
    pairs = sorted(runs)
    keys = set()
    for pair in pairs:
        for arm_map in runs[pair].values():
            keys.update(arm_map)
    shapes = sorted({shape for shape, _ in keys})
    out = {}
    for shape in shapes:
        ratios, reference, candidate = [], [], []
        for pair in pairs:
            b = runs[pair]["base"].get((shape, arm))
            c = runs[pair]["cand"].get((shape, arm))
            if not b or not c:
                raise StatisticsError(
                    f"pair {pair} is missing shape {shape!r} arm {arm!r} in an arm"
                )
            mb, mc = statistics.median(b), statistics.median(c)
            reference.append(mb)
            candidate.append(mc)
            ratios.append(mb / mc)
        if ratios:
            out[shape] = {
                "ratios": ratios,
                "reference_ms": reference,
                "candidate_ms": candidate,
            }
    return out


def summarize(directory, arm: str, require_positive: bool = True):
    """Per shape: medians, paired mean ratio, its t interval, and the geomean."""
    runs = load_dir(directory, require_positive=require_positive)
    out = {}
    for shape, data in paired_ratios(runs, arm).items():
        ratios = data["ratios"]
        mean, low, high = mean_ci(ratios)
        out[shape] = {
            "pairs": len(ratios),
            "median_reference_ms": statistics.median(data["reference_ms"]) * 1.0e3,
            "median_candidate_ms": statistics.median(data["candidate_ms"]) * 1.0e3,
            "mean_ratio": mean,
            "ci95_low": low,
            "ci95_high": high,
            "geomean_ratio": math.exp(statistics.fmean([math.log(r) for r in ratios])),
        }
    return out
