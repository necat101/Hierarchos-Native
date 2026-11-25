#!/usr/bin/env python3
"""Attribute Phase X production row-packed attention-forward kernel timings.

The harness submits exactly one measured arm per submission and prints

    [phase-x-bench-sample] geometry=... arm=... grid=... workgroups=... lanes=...

immediately before it; `[phase-x-bench-warmup]` closes the previous block. Every
`hierarchos_vulkan_kernel_profile_shader` line the shipped timestamp profiler
prints after a sample marker belongs to that submission, so the submission's GPU
time is the sum of its per-shader `gpu_ms` lines with a non-zero dispatch count.

Two quantities are reported, because they are not the same thing:

  * `gpu_ms`    -- the submission's GPU time, from the shipped timestamp queries.
  * `host_ms`   -- the harness's own fence-to-fence wall time for the submission,
                   parsed from the `[phase-x-bench-result]` line.

The promoted module is not in the binary's shader-name registry, so the
profiler labels them `unknown`. They are counted (dropping them would understate
the arm) and the label is preserved in the output so the substitution is visible.

Nothing is invented: a run without profiler output yields no GPU number, and an
arm whose block has no dispatch produces no sample.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

SAMPLE = re.compile(
    r"^\[phase-x-bench-sample\] geometry=(\S+) arm=(\S+) grid=\[[^\]]*\] "
    r"workgroups=(\d+) lanes=(\d+)"
)
WARMUP = re.compile(r"^\[phase-x-bench-warmup\]")
SHADER = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=\S+ "
    r"dispatches=(\d+) gpu_ms=([0-9.]+)"
)
HOST = re.compile(
    r"^\[phase-x-bench-result\] geometry=(\S+) arm=(\S+) workgroups=(\d+) "
    r"lanes=(\d+) dispatches_per_submission=(\d+) iterations=(\d+) "
    r"median_host_ms=([0-9.]+) p25_host_ms=([0-9.]+) min_host_ms=([0-9.]+) "
    r"max_host_ms=([0-9.]+)"
)


def parse(path: Path) -> dict:
    samples: dict[tuple[str, str, int, int], list[tuple[float, list[str]]]] = {}
    host: dict[tuple[str, str], dict] = {}
    current = None
    total = 0.0
    names: list[str] = []

    def flush() -> None:
        nonlocal total, names
        if current is not None and total > 0.0:
            samples.setdefault(current, []).append((total, list(names)))
        total = 0.0
        names = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if WARMUP.match(stripped):
            flush()
            current = None
            continue
        marker = SAMPLE.match(stripped)
        if marker:
            flush()
            geometry, arm, workgroups, lanes = marker.groups()
            current = (geometry, arm, int(workgroups), int(lanes))
            continue
        shader = SHADER.search(line)
        if shader and current is not None and int(shader.group(2)) > 0:
            total += float(shader.group(3))
            names.append(shader.group(1))
        parsed_host = HOST.match(stripped)
        if parsed_host:
            geometry, arm, workgroups, lanes, dispatches, iterations, median, p25, lo, hi = (
                parsed_host.groups()
            )
            host[(geometry, arm)] = {
                "workgroups": int(workgroups),
                "lanes": int(lanes),
                "dispatches_per_submission": int(dispatches),
                "iterations": int(iterations),
                "median_host_ms": float(median),
                "p25_host_ms": float(p25),
                "min_host_ms": float(lo),
                "max_host_ms": float(hi),
            }
    flush()

    report = {}
    for (geometry, arm, workgroups, lanes), entries in sorted(samples.items()):
        totals = [value for value, _ in entries]
        entry = {
            "geometry": geometry,
            "arm": arm,
            "workgroups": workgroups,
            "lanes": lanes,
            "shader_labels": sorted({name for _, group in entries for name in group}),
            "samples": len(totals),
            "median_gpu_ms": round(statistics.median(totals), 6),
            "mean_gpu_ms": round(statistics.fmean(totals), 6),
            "stdev_gpu_ms": round(statistics.stdev(totals), 6) if len(totals) > 1 else 0.0,
            "p25_gpu_ms": round(sorted(totals)[len(totals) // 4], 6),
            "min_gpu_ms": round(min(totals), 6),
            "max_gpu_ms": round(max(totals), 6),
            "all_gpu_ms": [round(value, 6) for value in totals],
        }
        if (geometry, arm) in host:
            entry["host"] = host[(geometry, arm)]
        report[f"{path.name}|{geometry}|{arm}"] = entry
    return report


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: analyze_timing.py LOG [LOG...]", file=sys.stderr)
        return 2
    merged: dict[str, dict] = {}
    for name in argv[1:]:
        merged.update(parse(Path(name)))
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
