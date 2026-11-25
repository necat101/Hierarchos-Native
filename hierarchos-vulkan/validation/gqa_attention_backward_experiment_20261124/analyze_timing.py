#!/usr/bin/env python3
"""Attribute Phase V GQA kernel timings from a profiled harness run.

The harness submits a known set of dispatches per submission and prints
`[gqa-bench-sample] geometry=... arm=...` immediately before it. Everything the
shipped timestamp profiler prints after that marker, up to the next marker, is
that submission's block; the block's GPU time is the sum of its per-shader
`gpu_ms` lines with a non-zero dispatch count.

The experimental pass-2 artifact is not in the binary's shader-name registry, so
the profiler labels it `unknown`. It is counted, and labelled as the experiment's
own module in the output, because dropping it would silently understate the
two-pass total.

A warmup marker closes the previous block. Nothing is invented: a run without
profiler output yields no number.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

MARKER = re.compile(r"^\[gqa-bench-sample\] geometry=(\S+) arm=(\S+)")
WARMUP = re.compile(r"^\[gqa-bench-warmup\] geometry=(\S+) arm=(\S+)")
SHADER = re.compile(
    r"kernel_profile_shader shader=(\S+) category=\S+ dispatches=(\d+) gpu_ms=([0-9.]+)"
)


def parse(path: Path) -> dict:
    samples: dict[tuple[str, str], list[tuple[float, list[str]]]] = {}
    current: tuple[str, str] | None = None
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
        marker = MARKER.match(stripped)
        if marker:
            flush()
            current = (marker.group(1), marker.group(2))
            continue
        shader = SHADER.search(line)
        if shader and current is not None:
            if int(shader.group(2)) > 0:
                total += float(shader.group(3))
                names.append(shader.group(1))
    flush()

    report = {}
    for (geometry, arm), entries in sorted(samples.items()):
        totals = [value for value, _ in entries]
        labels = sorted({name for _, group in entries for name in group})
        report[f"{geometry}/{arm}"] = {
            "shader_labels": labels,
            "samples": len(totals),
            "median_gpu_ms": round(statistics.median(totals), 6),
            "min_gpu_ms": round(min(totals), 6),
            "max_gpu_ms": round(max(totals), 6),
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
