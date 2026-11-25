#!/usr/bin/env python3
"""Per-component GPU attribution of one complete Phase X training step.

The harness prints one `[phase-x-step]` line per step; the shipped timestamp
profiler prints one `hierarchos_vulkan_kernel_profile_shader` line per shader per
profiled submission. Summing a shader's `gpu_ms` over a log and dividing by the number of steps the
run *submitted* gives that shader's GPU time per training step. The profiler
covers every submission, warmups included, so the denominator is the total step
count -- dividing by the measured-only count would overstate each kernel by
`steps / measured_steps`.

Use it on the two `step-profile-*.log` arms to see which kernel the promotion
moved and which it left alone. Profiling perturbs wall time, so these logs are
for attribution only; the timing claim comes from `assemble.py`.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

SHADER = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=(\S+) "
    r"dispatches=(\d+) gpu_ms=([0-9.]+)"
)
STEP = re.compile(r"\[phase-x-step\] arm=(\S+) step=(\d+) .* measured=(\w+)")
END = re.compile(
    r"\[phase-x-step-run-end\] arm=(\S+) .*forward_generic=(\d+) "
    r"forward_row_packed=(\d+) forward_short=(\d+)"
)


def parse(path: Path) -> dict:
    totals: dict[str, float] = collections.defaultdict(float)
    dispatches: dict[str, int] = collections.defaultdict(int)
    categories: dict[str, str] = {}
    measured = 0
    steps = 0
    arm = None
    counters: dict = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        step = STEP.search(line)
        if step:
            arm = step.group(1)
            steps += 1
            if step.group(3) == "true":
                measured += 1
        end = END.search(line)
        if end:
            counters = {
                "forward_generic": int(end.group(2)),
                "forward_row_packed": int(end.group(3)),
                "forward_short": int(end.group(4)),
            }
        shader = SHADER.search(line)
        if shader and int(shader.group(3)) > 0:
            name = shader.group(1)
            totals[name] += float(shader.group(4))
            dispatches[name] += int(shader.group(3))
            categories[name] = shader.group(2)
    if steps == 0:
        raise SystemExit(f"{path}: no steps")
    per_step = {
        name: {
            "gpu_ms_per_step": round(value / steps, 4),
            "dispatches_per_step": round(dispatches[name] / steps, 3),
            "category": categories[name],
        }
        for name, value in totals.items()
    }
    return {
        "log": path.name,
        "arm": arm,
        "profiled_steps": steps,
        "measured_steps": measured,
        "counters": counters,
        "total_gpu_ms_per_step": round(sum(totals.values()) / steps, 4),
        "per_step": dict(sorted(per_step.items())),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: attribute.py LOG [LOG...]", file=sys.stderr)
        return 2
    report = {"logs": [parse(Path(name)) for name in argv[1:]]}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
