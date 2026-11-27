#!/usr/bin/env python3
"""Phase XIII kernel attribution from the two profiled release runs.

The profiler prints one block per submitted compute batch: a
`hierarchos_vulkan_kernel_profile device=... gpu_ms=...` header followed by
per-category, per-shader and placement lines for that batch. A training step is
one batch, so a batch is selected when it contains the two-layer GQA backward
query dispatch (`transformer_attention_backward_query_rdna3 dispatches=2`).
Selection is by the query pass rather than by step count so a startup or
checkpoint batch can never be averaged into the profile.

Usage: attribute.py OFF_LOG MIXED_LOG STEPS
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

QUERY = "transformer_attention_backward_query_rdna3"


def batches(path: Path) -> list[dict]:
    out: list[dict] = []
    current: dict | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "hierarchos_vulkan_kernel_profile device=" in line:
            if current is not None:
                out.append(current)
            fields = dict(t.split("=", 1) for t in line.split() if "=" in t)
            current = {"gpu_ms": float(fields["gpu_ms"]), "shaders": {}, "raw": [line]}
        elif current is not None and "hierarchos_vulkan_kernel_profile_shader" in line:
            fields = dict(t.split("=", 1) for t in line.split() if "=" in t)
            current["shaders"][fields["shader"]] = {
                "dispatches": int(fields["dispatches"]),
                "gpu_ms": float(fields["gpu_ms"]),
            }
            current["raw"].append(line)
        elif current is not None and "hierarchos_vulkan_kernel_profile" in line:
            current["raw"].append(line)
    if current is not None:
        out.append(current)
    return out


def steps(path: Path) -> list[dict]:
    return [b for b in batches(path) if b["shaders"].get(QUERY, {}).get("dispatches", 0) > 0]


def summarize(path: Path, expected: int) -> dict:
    selected = steps(path)
    if len(selected) != expected:
        raise SystemExit(f"{path.name}: {len(selected)} step batches, expected {expected}")
    per_shader: dict[str, dict[str, float]] = {}
    totals = []
    for batch in selected:
        totals.append(batch["gpu_ms"])
        for name, stats in batch["shaders"].items():
            entry = per_shader.setdefault(name, {"gpu_ms": 0.0, "dispatches": 0, "steps": 0})
            entry["gpu_ms"] += stats["gpu_ms"]
            entry["dispatches"] += stats["dispatches"]
            entry["steps"] += 1
    for entry in per_shader.values():
        entry["ms_per_step"] = round(entry["gpu_ms"] / len(selected), 6)
        entry["dispatches_per_step"] = round(entry["dispatches"] / len(selected), 4)
    return {
        "batches": len(selected),
        "gpu_ms_per_step_total": round(statistics.fmean(totals), 6),
        "gpu_ms_per_step_median": round(statistics.median(totals), 6),
        "shaders": {
            name: {
                "ms_per_step": entry["ms_per_step"],
                "dispatches_per_step": entry["dispatches_per_step"],
            }
            for name, entry in sorted(per_shader.items(), key=lambda item: -item[1]["ms_per_step"])
        },
    }


def main(argv: list[str]) -> int:
    off = summarize(Path(argv[1]), int(argv[3]))
    mixed = summarize(Path(argv[2]), int(argv[3]))
    names = sorted(set(off["shaders"]) | set(mixed["shaders"]), key=lambda n: -(off["shaders"].get(n, {}).get("ms_per_step", 0.0)))
    table = []
    for name in names:
        a = off["shaders"].get(name, {"ms_per_step": 0.0, "dispatches_per_step": 0.0})
        b = mixed["shaders"].get(name, {"ms_per_step": 0.0, "dispatches_per_step": 0.0})
        table.append(
            {
                "shader": name,
                "off_ms": round(a["ms_per_step"], 6),
                "mixed_ms": round(b["ms_per_step"], 6),
                "delta_ms": round(b["ms_per_step"] - a["ms_per_step"], 6),
                "off_dispatch": a["dispatches_per_step"],
                "mixed_dispatch": b["dispatches_per_step"],
            }
        )
    print(
        json.dumps(
            {
                "off": {"batches": off["batches"], "gpu_ms_per_step": off["gpu_ms_per_step_total"]},
                "mixed": {"batches": mixed["batches"], "gpu_ms_per_step": mixed["gpu_ms_per_step_total"]},
                "delta_gpu_ms_per_step": round(
                    mixed["gpu_ms_per_step_total"] - off["gpu_ms_per_step_total"], 6
                ),
                "table": table,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
