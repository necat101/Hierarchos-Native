#!/usr/bin/env python3
"""Controls for the canonical ABBA statistics module.

Two kinds of test:

* quantile tests, against independently published two-tailed 95 % Student-t
  values for the degrees of freedom this repository actually produces, plus the
  range boundaries where the old table silently fell back to `1.96`;
* negative controls, which build a small input that a correct implementation
  must reject rather than summarise into a plausible interval.

Run: python -m pytest validation/test_abba_stats.py -q
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import abba_stats


# Two-tailed 95 % Student-t critical values (NIST/standard tables, 6 decimals).
REFERENCE_T95 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    15: 2.131450,
    23: 2.068658,
    31: 2.039513,
    32: 2.036933,
    33: 2.034515,
    47: 2.011741,
    63: 1.998341,
}

REQUIRED_DF = (7, 11, 12, 13, 15, 23, 31, 32, 33, 47, 63)


def build_abba(
    root: Path,
    *,
    pairs=4,
    samples=5,
    rc: dict | None = None,
    duplicate_pair: bool = False,
    drop_arm: str | None = None,
    malformed: bool = False,
    empty: bool = False,
    nonfinite: bool = False,
    nonpositive: bool = False,
) -> Path:
    """Write a minimal, otherwise-valid ABBA directory."""
    root.mkdir(parents=True, exist_ok=True)
    index = root / "index.tsv"
    lines = ["pair\tarm\trc\texe_sha256\tfile"]
    for pair in range(1, pairs + 1):
        for arm in ("base", "cand"):
            if drop_arm is not None and arm == drop_arm:
                continue
            name = root / f"p{pair}_{arm}"
            code = "0"
            if rc and (pair, arm) in rc:
                code = str(rc[(pair, arm)])
            lines.append(f"{pair}\t{arm}\t{code}\tdeadbeef\t{name}\t{name}.exe")
            if duplicate_pair and pair == 1 and arm == "cand":
                lines.append(f"{pair}\t{arm}\t0\tdeadbeef\t{name}\t{name}.exe")
            payload = {
                "shape": "8x8x8",
                "arm": "wmma-mixed-f32acc",
                "samples_s": [] if empty else [0.001 * (pair + i) for i in range(samples)],
            }
            if nonfinite:
                payload["samples_s"] = [float("inf")] + payload["samples_s"][1:]
            if nonpositive:
                payload["samples_s"] = [0.0] + payload["samples_s"][1:]
            text = json.dumps(payload)
            if malformed:
                text = text[:-3]
            else:
                text += "\n"
            (root / f"p{pair}_{arm}_samples.jsonl").write_text(text, encoding="utf-8")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class Quantiles(unittest.TestCase):
    def test_reference_quantiles(self):
        for df in REQUIRED_DF:
            with self.subTest(df=df):
                got = abba_stats.t_critical(df)
                want = REFERENCE_T95[df]
                self.assertLess(abs(got - want), 5.0e-6, f"df={df}: {got} vs {want}")

    def test_every_required_df_is_present_in_the_reference_table(self):
        missing = [df for df in REQUIRED_DF if df not in REFERENCE_T95]
        self.assertEqual(missing, [])

    def test_never_silently_normal(self):
        # The defect was `1.96` for any df beyond the table. A correct t
        # quantile is strictly larger than the normal one at every finite df.
        for df in (13, 24, 32, 33, 48, 64, 100, 1000, 10_000):
            with self.subTest(df=df):
                self.assertGreater(abba_stats.t_critical(df), 1.96)
        self.assertLess(abba_stats.t_critical(60_000), 1.9605)

    def test_quantile_is_monotone(self):
        values = [abba_stats.t_critical(df) for df in range(1, 80)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_confidence_is_checked(self):
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.t_critical(10, 1.0)
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.t_critical(10, 0.0)

    def test_interval_matches_its_definition(self):
        import statistics as _statistics

        values = [1.0, 1.2, 0.9, 1.1, 0.95, 1.05]
        mean, low, high = abba_stats.mean_ci(values)
        half = (
            abba_stats.t_critical(len(values) - 1)
            * _statistics.stdev(values)
            / math.sqrt(len(values))
        )
        self.assertAlmostEqual(mean, _statistics.fmean(values), places=15)
        self.assertAlmostEqual(low, mean - half, places=12)
        self.assertAlmostEqual(high, mean + half, places=12)
        # And the interval genuinely narrows as the sample count grows. The
        # second list has the same mean and standard deviation, eight times the
        # observations, so the only change is the interval denominator.
        doubled = values + [2 * mean - v for v in values]
        _, low2, high2 = abba_stats.mean_ci(doubled)
        self.assertGreater(low2, low)
        self.assertLess(high2, high)


class NegativeControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="abba-stats-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertRejected(self, **kwargs):
        directory = build_abba(self.root / "case", **kwargs)
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.summarize(directory, "wmma-mixed-f32acc")

    def test_missing_paired_samples(self):
        directory = build_abba(self.root / "ok")
        (directory / "p2_cand_samples.jsonl").unlink()
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.summarize(directory, "wmma-mixed-f32acc")

    def test_duplicate_pair_identifier(self):
        self.assertRejected(duplicate_pair=True)

    def test_missing_arm(self):
        self.assertRejected(drop_arm="cand")

    def test_nonzero_exit_code(self):
        self.assertRejected(rc={(2, "base"): 1})

    def test_empty_sample_array(self):
        self.assertRejected(empty=True)

    def test_malformed_json(self):
        self.assertRejected(malformed=True)

    def test_nonfinite_time(self):
        self.assertRejected(nonfinite=True)

    def test_invalid_zero_time(self):
        self.assertRejected(nonpositive=True)

    def test_too_few_pairs_for_an_interval(self):
        directory = build_abba(self.root / "one", pairs=1)
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.summarize(directory, "wmma-mixed-f32acc")

    def test_unsupported_sample_count(self):
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.t_critical(abba_stats.MAX_DF + 1)
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.t_critical(0)

    def test_forged_confidence_interval_is_detected(self):
        directory = build_abba(self.root / "forge")
        honest = abba_stats.summarize(directory, "wmma-mixed-f32acc")["8x8x8"]
        forged = dict(honest)
        forged["ci95_low"] = honest["mean_ratio"] - 1.0e-9
        forged["ci95_high"] = honest["mean_ratio"] + 1.0e-9
        self.assertNotEqual(
            abs(forged["ci95_high"] - honest["ci95_high"]) < 1e-12,
            True,
            "a forged interval must not reproduce",
        )
        # The recomputation is deterministic, so a receipt carrying the forged
        # numbers cannot reproduce from the same samples.
        again = abba_stats.summarize(directory, "wmma-mixed-f32acc")["8x8x8"]
        self.assertAlmostEqual(again["ci95_low"], honest["ci95_low"], places=15)
        self.assertAlmostEqual(again["ci95_high"], honest["ci95_high"], places=15)

    def test_gzip_samples_load(self):
        import gzip

        directory = build_abba(self.root / "gz", pairs=3)
        for path in sorted(directory.glob("*_samples.jsonl")):
            with open(path, "rb") as handle:
                payload = handle.read()
            path.unlink()
            with gzip.open(path.with_suffix(".jsonl.gz"), "wb") as handle:
                handle.write(payload)
        summary = abba_stats.summarize(directory, "wmma-mixed-f32acc")
        self.assertEqual(summary["8x8x8"]["pairs"], 3)


if __name__ == "__main__":
    unittest.main()
