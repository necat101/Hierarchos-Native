#!/usr/bin/env python3
"""Recompute every preserved ABBA interval with the canonical statistics.

The old analyzers produced their intervals from a Student-t table that stopped at
12 degrees of freedom (`wmma_abba_analyze.py`) or 32 (`phase_lxv_stats.py`), and
then used `1.96` beyond it. This script recomputes every interval that the
preserved raw samples can support, prints the superseded value beside the
corrected one, and states whether any recorded conclusion moves.

Nothing is written back into a historical receipt. The output is a new, dated
record: `phase_lxvi_statistics_correction.json` and its Markdown rendering.

Usage:
    python phase_lxvi_historical_ci.py            # write the record
    python phase_lxvi_historical_ci.py --check    # verify the record reproduces
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import sys
from pathlib import Path

import abba_stats

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "phase_lxv_evidence"
RECORD = HERE / "phase_lxvi_statistics_correction.json"
MARKDOWN = HERE / "PHASE_LXVI_STATISTICS_CORRECTION.md"

# The exact lookup the superseded analyzer used: a 12-entry table, then 1.96.
LEGACY_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
}

# The Phase LXV module's own extension of that table (to 32 df).
LXV_T95 = dict(LEGACY_T95)
LXV_T95.update({
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    31: 2.040, 32: 2.037,
})

NONINFERIORITY_LOWER = 0.99


def legacy_interval(values, table):
    """Reproduce a superseded interval exactly. Archival only, never a decision."""
    n = len(values)
    mean = statistics.fmean(values)
    quantile = table.get(n - 1, 1.96)
    half = quantile * statistics.stdev(values) / (n ** 0.5)
    return mean, mean - half, mean + half, quantile


def conclusion(mean, low, high):
    if low > 1.0:
        return "SPEEDUP (CI excludes 1.0)"
    if high < 1.0:
        return "REGRESSION (CI excludes 1.0)"
    return "INCONCLUSIVE (CI includes 1.0)"


def non_inferior(mean, low):
    return mean >= 1.0 and low >= NONINFERIORITY_LOWER


def discover():
    """Every preserved ABBA directory, with the arms it actually contains."""
    found = []
    for index in sorted(EVIDENCE.glob("abba_*/index.tsv")):
        directory = index.parent
        runs = abba_stats.load_dir(directory)
        arms = sorted({arm for pair in runs.values() for key in pair["cand"] for _, arm in [key]})
        digest = hashlib.sha256(index.read_bytes()).hexdigest()
        found.append({"name": directory.name, "directory": directory, "arms": arms,
                      "index_sha256": digest, "pairs": len(runs)})
    return found


def build_record():
    entries = []
    for source in discover():
        for arm in source["arms"]:
            stats = abba_stats.summarize(source["directory"], arm)
            runs = abba_stats.load_dir(source["directory"])
            paired = abba_stats.paired_ratios(runs, arm)
            for shape in sorted(stats):
                values = paired[shape]["ratios"]
                mean, low, high = abba_stats.mean_ci(values)
                l_mean, l_low, l_high, l_q = legacy_interval(values, LEGACY_T95)
                x_mean, x_low, x_high, x_q = legacy_interval(values, LXV_T95)
                z_mean, z_low, z_high, _ = legacy_interval(values, {})
                entries.append({
                    "sweep": source["name"],
                    "arm": arm,
                    "shape": shape,
                    "pairs": len(values),
                    "df": len(values) - 1,
                    "index_sha256": source["index_sha256"],
                    "mean_ratio": mean,
                    "corrected_ci95": [low, high],
                    "corrected_quantile": abba_stats.t_critical(len(values) - 1),
                    "legacy_z_fallback_quantile": l_q,
                    "legacy_z_fallback_ci95": [l_low, l_high],
                    "z_1p96_ci95": [z_low, z_high],
                    "z_1p96_quantile": 1.96,
                    "lxv_table_quantile": x_q,
                    "lxv_table_ci95": [x_low, x_high],
                    "interval_widened_by": max(abs(l_low - low), abs(l_high - high)),
                    "conclusion_before": conclusion(l_mean, l_low, l_high),
                    "conclusion_after": conclusion(mean, low, high),
                    "non_inferior_before": non_inferior(l_mean, l_low),
                    "non_inferior_after": non_inferior(mean, low),
                })
    record = {
        "record": "phase_lxvi_statistics_correction",
        "generated": datetime.date.today().isoformat(),
        "implementation": "hierarchos-vulkan/validation/abba_stats.py",
        "defect": (
            "wmma_abba_analyze.py used a 12-entry two-tailed 95% t table and "
            "T95.get(df, 1.96); phase_lxv_stats.py extended the table to 32 "
            "degrees of freedom and kept the same 1.96 fallback. Both silently "
            "replace a finite-sample t quantile with the normal quantile, which "
            "makes intervals too narrow."
        ),
        "method": (
            "Each pair's statistic is the median of that run's per-iteration "
            "samples (unchanged). The interval is the two-sided 95% Student-t "
            "interval over the per-pair ratios with df = pairs - 1, using a "
            "computed quantile. The point estimate and the geomean are unchanged."
        ),
        "recomputable_sweeps": sorted({entry["sweep"] for entry in entries}),
        "unrecomputable_sweeps": [
            {
                "sweep": "phase_lxi / phase_lxii / phase_lxiv",
                "reason": (
                    "no raw per-iteration ABBA samples are present in this tree; "
                    "only reported summary numbers survive. Those intervals are "
                    "therefore not recomputed here and are not used for any new "
                    "decision. Re-running the sweeps would be required."
                ),
            }
        ],
        "entries": entries,
        "conclusion_changes": [
            {
                "sweep": e["sweep"], "arm": e["arm"], "shape": e["shape"],
                "before": e["conclusion_before"], "after": e["conclusion_after"],
            }
            for e in entries
            if e["conclusion_before"] != e["conclusion_after"]
        ],
        "non_inferiority_changes": [
            {
                "sweep": e["sweep"], "arm": e["arm"], "shape": e["shape"],
                "before": e["non_inferior_before"], "after": e["non_inferior_after"],
            }
            for e in entries
            if e["non_inferior_before"] != e["non_inferior_after"]
        ],
    }
    return record


def render(record):
    lines = [
        "# Phase LXVI — dated statistical correction",
        "",
        f"Generated {record['generated']} from the preserved raw samples by",
        "`phase_lxvi_historical_ci.py`. No historical receipt, sample or hash was",
        "modified.",
        "",
        "## What was wrong",
        "",
        record["defect"],
        "",
        "## What was done",
        "",
        record["method"],
        "",
        f"The single implementation is `{record['implementation']}`. It computes the",
        "quantile from the regularized incomplete beta function, refuses degrees of",
        "freedom outside its supported range, and contains no `1.96` fallback on any",
        "live path.",
        "",
        "## Recomputable sweeps",
        "",
        ", ".join(record["recomputable_sweeps"]),
        "",
        "## Not recomputable here",
        "",
    ]
    for missing in record["unrecomputable_sweeps"]:
        lines.append(f"* **{missing['sweep']}** — {missing['reason']}")
    lines += [
        "",
        "## Old and corrected intervals",
        "",
        "`legacy` is the interval the superseded `T95.get(df, 1.96)` rule printed.",
        "`lxv` is the interval the Phase LXV in-tree table printed (it mattered only",
        "where the table was missing). `corrected` is the canonical t interval.",
        "",
        "| Sweep | Shape | n | legacy CI | lxv CI | corrected CI | widen | conclusion |",
        "|---|---|---:|---|---|---|---:|---|",
    ]
    for e in record["entries"]:
        lines.append(
            "| {sweep} | {shape} | {pairs} | [{l0:.4f}, {l1:.4f}] | [{x0:.4f}, {x1:.4f}] | "
            "[{c0:.4f}, {c1:.4f}] | {widen:.2e} | {conclusion} |".format(
                sweep=e["sweep"], shape=e["shape"], pairs=e["pairs"],
                l0=e["legacy_z_fallback_ci95"][0], l1=e["legacy_z_fallback_ci95"][1],
                x0=e["lxv_table_ci95"][0], x1=e["lxv_table_ci95"][1],
                c0=e["corrected_ci95"][0], c1=e["corrected_ci95"][1],
                widen=e["interval_widened_by"], conclusion=e["conclusion_after"],
            )
        )
    lines += ["", "## Did any conclusion change?", ""]
    if not record["conclusion_changes"]:
        lines.append("No. Every qualitative verdict (speedup / regression / inconclusive) is unchanged.")
    else:
        for change in record["conclusion_changes"]:
            lines.append(
                f"* {change['sweep']} / {change['shape']}: {change['before']} → {change['after']}"
            )
    lines += ["", "## Did the non-inferiority gate change?", ""]
    if not record["non_inferiority_changes"]:
        lines.append(
            "No. The FINAL-versus-Candidate-A comparison remains `INCONCLUSIVE` under the "
            "declared 0.99 lower-bound rule, with the corrected (wider) intervals."
        )
    else:
        for change in record["non_inferiority_changes"]:
            lines.append(
                f"* {change['sweep']} / {change['shape']}: "
                f"{change['before']} → {change['after']}"
            )
    lines += [
        "",
        "## Where the corrected numbers are used",
        "",
        "* every new performance decision, including the Phase LXVI gate;",
        "* `check_phase_lxv.py` verifies the archived receipt against this record,",
        "  so the historical receipt stays byte-identical while the live comparison",
        "  uses the corrected interval.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the committed record reproduces; do not write")
    args = parser.parse_args()
    record = build_record()
    if args.check:
        committed = json.loads(RECORD.read_text(encoding="utf-8"))
        if committed["entries"] != record["entries"]:
            print("record does not reproduce from the raw samples", file=sys.stderr)
            return 1
        if committed["unrecomputable_sweeps"] != record["unrecomputable_sweeps"]:
            print("unrecomputable-sweep block does not reproduce", file=sys.stderr)
            return 1
        print(f"correction record reproduces: {len(record['entries'])} entries")
        return 0
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(render(record), encoding="utf-8")
    print(f"wrote {RECORD.name} ({len(record['entries'])} entries) and {MARKDOWN.name}")
    print(f"conclusion changes: {len(record['conclusion_changes'])}")
    print(f"non-inferiority changes: {len(record['non_inferiority_changes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
