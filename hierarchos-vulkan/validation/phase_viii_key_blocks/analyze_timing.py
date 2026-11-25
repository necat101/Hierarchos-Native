#!/usr/bin/env python3
"""Attribute the Phase VIII key-block kernel timings from a profiled harness run.

The harness submits one measured arm per submission and prints

    [phase-viii-bench-sample] geometry=... mode=... arm=... blocks=... lanes=...

immediately before it; `[phase-viii-bench-warmup]` closes the previous block. Every
`hierarchos_vulkan_kernel_profile_shader` line the shipped timestamp profiler
prints after a sample marker belongs to that submission, so the submission's GPU
time is the sum of its per-shader `gpu_ms` lines with a non-zero dispatch count.

Two properties are reported, because the key point of this phase is that they
are not the same thing:

  * `gpu_ms` -- the submission's GPU time, which is what a training step pays.
  * `workgroups` -- parsed from the harness marker, so a reader can see the
    configuration that produced the number without trusting this script's
    arithmetic.

The candidate module is not in the binary's shader-name registry, so the profiler
labels it `unknown`. It is counted (dropping it would understate the arm) and the
label is preserved in the output so the substitution is visible.

Nothing is invented: a run without profiler output yields no number, and an arm
whose block has no dispatch produces no sample.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

SAMPLE = re.compile(
    r"^\[phase-viii-bench-sample\] geometry=(\S+) mode=(\S+) arm=(\S+) "
    r"blocks=(\d+) lanes=(\d+) workgroups=(\d+)"
)
WARMUP = re.compile(r"^\[phase-viii-bench-warmup\]")
SHADER = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=\S+ "
    r"dispatches=(\d+) gpu_ms=([0-9.]+)"
)


def parse(path: Path) -> dict:
    samples: dict[tuple[str, str, str, str, str, str], list[tuple[float, list[str]]]] = {}
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
            current = marker.groups()
            continue
        shader = SHADER.search(line)
        if shader and current is not None and int(shader.group(2)) > 0:
            total += float(shader.group(3))
            names.append(shader.group(1))
    flush()

    report = {}
    for (geometry, mode, arm, blocks, lanes, workgroups), entries in sorted(samples.items()):
        totals = [value for value, _ in entries]
        report[f"{geometry}|mode{mode}|{arm}"] = {
            "geometry": geometry,
            "mode": int(mode),
            "arm": arm,
            "blocks": int(blocks),
            "lanes": int(lanes),
            "workgroups": int(workgroups),
            "shader_labels": sorted({name for _, group in entries for name in group}),
            "samples": len(totals),
            "median_gpu_ms": round(statistics.median(totals), 6),
            "mean_gpu_ms": round(statistics.fmean(totals), 6),
            "min_gpu_ms": round(min(totals), 6),
            "max_gpu_ms": round(max(totals), 6),
            "all_gpu_ms": [round(value, 6) for value in totals],
        }
    return report


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: analyze_timing.py LOG [LOG...]", file=sys.stderr)
        return 2
    merged: dict[str, dict] = {}
    for name in argv[1:]:
        for key, value in parse(Path(name)).items():
            merged[f"{Path(name).name}:{key}"] = value
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
