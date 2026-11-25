#!/usr/bin/env python3
"""Per-shader GPU attribution for a profiled Phase XI step run.

Usage: attribute.py LOG [LOG...] > attribution.json

Each log is one process that ran `phase_xi_channel_tiling_full_step` with
`HIERARCHOS_VULKAN_PROFILE_KERNELS=1 HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`.
The shipped profiler prints a `hierarchos_vulkan_kernel_profile_shader` line per
shader at the end of the run; this sums those and divides by the number of
*measured* steps the harness reported, so the numbers are per training step.

The candidate module is not in the binary's shader-name registry, so its key
dispatch appears as `unknown`. In a `c4_u0` run that label is the channel-tiled
key pass; in a production run no `unknown` dispatch exists. The report keeps the
label and states which arm produced it rather than renaming it.

A profiled run is a diagnostic: per-dispatch timestamp queries perturb the step
they measure, so these numbers attribute work between components and are not the
headline timings. The headline numbers come from the unprofiled ABBA runs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SHADER = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=(\S+) "
    r"dispatches=(\d+) gpu_ms=([0-9.]+) pct=([0-9.]+)"
)
STEP = re.compile(r"^\[phase-xi-step\] .*?arm=(\S+) .*?measured=(true|false)")
RUN = re.compile(r"^\[phase-xi-step-run\] arm=(\S+)")


def parse(path: Path) -> dict:
    shaders: dict[str, dict] = {}
    arm = None
    measured = 0
    total_steps = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        run = RUN.match(line)
        if run:
            arm = run.group(1)
        step = STEP.match(line)
        if step:
            arm = arm or step.group(1)
            total_steps += 1
            if step.group(2) == "true":
                measured += 1
            continue
        match = SHADER.search(line)
        if match:
            name, category, dispatches, gpu_ms, pct = match.groups()
            entry = shaders.setdefault(
                name, {"category": category, "dispatches": 0, "gpu_ms": 0.0, "pct": 0.0}
            )
            entry["dispatches"] += int(dispatches)
            entry["gpu_ms"] += float(gpu_ms)
            entry["pct"] += float(pct)
    if total_steps == 0:
        return {"arm": arm, "measured_steps": 0, "shaders": {}}
    total = sum(entry["gpu_ms"] for entry in shaders.values())
    per_step = {
        name: {
            "category": entry["category"],
            "dispatches_per_step": round(entry["dispatches"] / total_steps, 3),
            "gpu_ms_per_step": round(entry["gpu_ms"] / total_steps, 6),
            "share_pct": round(100.0 * entry["gpu_ms"] / total, 3) if total else 0.0,
        }
        for name, entry in shaders.items()
    }
    return {
        "arm": arm,
        "steps_profiled": total_steps,
        "measured_steps": measured,
        "attributed_gpu_ms_per_step": round(total / total_steps, 6),
        "shaders": dict(sorted(per_step.items(), key=lambda item: -item[1]["gpu_ms_per_step"])),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: attribute.py LOG [LOG...]", file=sys.stderr)
        return 2
    print(json.dumps({Path(name).name: parse(Path(name)) for name in argv[1:]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
