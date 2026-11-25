#!/usr/bin/env python3
"""Attribute the Phase XI channel-tiling kernel timings from a profiled run.

The harness submits one measured arm per submission and prints

    [phase-xi-bench-sample] geometry=... mode=... arm=... chunk=... blocks=... unroll=... workgroups=...

immediately before it; `[phase-xi-bench-warmup]` marks a warmup submission. Every
`hierarchos_vulkan_kernel_profile_shader` line the shipped timestamp profiler
prints after a sample marker belongs to that submission, so the submission's GPU
time is the sum of its per-shader `gpu_ms` lines with a non-zero dispatch count.

The candidate module is not in the binary's shader-name registry, so the profiler
labels it `unknown`. Every measured arm here is key-only, so the `unknown` bucket
of a submission is the candidate dispatch and nothing else; the label is carried
through to the output so the substitution stays visible.

The harness samples round-major (every arm once per round). Because each round
contributes exactly one sample per arm, the k-th sample of one arm pairs with the
k-th sample of the production arm, and this script reports the median of those
per-round ratios alongside the unpaired medians. The ratio is the quantity a
promotion argument needs: it is insensitive to a slow first round.

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
    r"^\[phase-xi-bench-sample\] geometry=(\S+) mode=(\S+) arm=(\S+) "
    r"chunk=(\d+) blocks=(\d+) unroll=(\d+) workgroups=(\d+)"
)
WARMUP = re.compile(r"^\[phase-xi-bench-warmup\]")
RESULT = re.compile(
    r"^\[phase-xi-bench-result\] geometry=(\S+) mode=(\S+) arm=(\S+) "
    r"workgroups=(\d+) iterations=(\d+) median_ms=([0-9.]+) min_ms=([0-9.]+) max_ms=([0-9.]+)"
)
SHADER = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=\S+ "
    r"dispatches=(\d+) gpu_ms=([0-9.]+)"
)


def parse(path: Path) -> dict:
    # key -> list of (gpu_ms, shader labels, host median from the result line)
    samples: dict[tuple, list[tuple[float, list[str]]]] = {}
    host_medians: dict[tuple, float] = {}
    current = None
    total = 0.0
    names: list[str] = []
    saw_marker = False

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
            saw_marker = True
            continue
        marker = SAMPLE.match(stripped)
        if marker:
            flush()
            geometry, mode, arm, chunk, blocks, unroll, workgroups = marker.groups()
            current = (geometry, int(mode), arm, int(chunk), int(blocks), int(unroll), int(workgroups))
            saw_marker = True
            continue
        result = RESULT.match(stripped)
        if result:
            geometry, mode, arm, _workgroups, _iterations, median, _mn, _mx = result.groups()
            host_medians[(geometry, int(mode), arm)] = float(median)
            continue
        shader = SHADER.search(line)
        if shader and current is not None and int(shader.group(2)) > 0:
            total += float(shader.group(3))
            names.append(shader.group(1))
    flush()

    report: dict[str, dict] = {}
    for key, entries in sorted(samples.items()):
        geometry, mode, arm, chunk, blocks, unroll, workgroups = key
        totals = [value for value, _ in entries]
        report[f"{geometry}|mode{mode}|{arm}"] = {
            "geometry": geometry,
            "mode": mode,
            "arm": arm,
            "chunk": chunk,
            "blocks": blocks,
            "unroll": unroll,
            "workgroups": workgroups,
            "shader_labels": sorted({name for _, group in entries for name in group}),
            "samples": len(totals),
            "median_gpu_ms": round(statistics.median(totals), 6),
            "mean_gpu_ms": round(statistics.fmean(totals), 6),
            "min_gpu_ms": round(min(totals), 6),
            "max_gpu_ms": round(max(totals), 6),
            "all_gpu_ms": [round(value, 6) for value in totals],
            "host_median_ms": host_medians.get((geometry, mode, arm)),
        }

    # Paired ratios against the production arm of the same geometry and mode.
    paired: dict[str, dict] = {}
    for key, value in report.items():
        if value["arm"] == "key_production":
            continue
        production = report.get(f"{value['geometry']}|mode{value['mode']}|key_production")
        if not production:
            continue
        left = value["all_gpu_ms"]
        right = production["all_gpu_ms"]
        count = min(len(left), len(right))
        ratios = [left[index] / right[index] for index in range(count) if right[index] > 0.0]
        if not ratios:
            continue
        paired[key] = {
            "geometry": value["geometry"],
            "mode": value["mode"],
            "arm": value["arm"],
            "paired_rounds": len(ratios),
            "median_paired_ratio": round(statistics.median(ratios), 6),
            "min_paired_ratio": round(min(ratios), 6),
            "max_paired_ratio": round(max(ratios), 6),
            "all_paired_ratios": [round(ratio, 6) for ratio in ratios],
        }
    return {"markers_seen": saw_marker, "arms": report, "paired": paired}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: analyze_timing.py LOG [LOG...]", file=sys.stderr)
        return 2
    merged: dict[str, dict] = {"sessions": {}}
    for name in argv[1:]:
        parsed = parse(Path(name))
        merged["sessions"][Path(name).name] = parsed
        for key, value in parsed["arms"].items():
            merged.setdefault("arms_by_session", {})[f"{Path(name).name}:{key}"] = value
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
