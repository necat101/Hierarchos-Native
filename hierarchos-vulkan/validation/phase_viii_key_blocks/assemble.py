#!/usr/bin/env python3
"""Assemble the Phase VIII receipt's summary.json and its markdown tables.

Every number here is derived from a raw artifact written by the harness:
`raw/parity.log` for the parity and ownership results, `raw/timing-gpu.json`
and `raw/timing-gpu-reverse.json` for the GPU timestamps, and `raw/*-hashes.txt`
for the artifact identities. Nothing is typed in from memory; a missing artifact
is an error rather than a blank cell.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

SUMMARY = re.compile(r"^\[phase-viii-parity-summary\] (.*)$")
OWNERSHIP = re.compile(r"^\[phase-viii-ownership\] (.*)$")
PARITY = re.compile(r"^\[phase-viii-parity\] (.*)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_parity() -> dict:
    text = (RAW / "parity.log").read_text(encoding="utf-8", errors="replace")
    summary = None
    ownership = []
    verdicts = {"PASS": 0, "FAIL": 0}
    for line in text.splitlines():
        line = line.strip()
        match = SUMMARY.match(line)
        if match:
            summary = dict(part.split("=", 1) for part in match.group(1).split())
        own = OWNERSHIP.match(line)
        if own:
            ownership.append(dict(part.split("=", 1) for part in own.group(1).split()))
        par = PARITY.match(line)
        if par:
            fields = dict(
                (token.split("=", 1) + ["", ""])[:2] for token in par.group(1).split()
            )
            if "verdict" in fields:
                verdicts[fields["verdict"]] += 1
    if summary is None:
        raise SystemExit("parity.log has no summary line")
    return {
        "summary": {key: int(value) if value.isdigit() else value for key, value in summary.items()},
        "comparison_verdicts": verdicts,
        "ownership_passes": [
            {
                "geometry": row["geometry"],
                "lanes": int(row["lanes"]),
                "blocks": int(row["blocks"]),
                "destination_elements": int(row["destination_elements"]),
                "missing": int(row["missing"]),
                "overlapping": int(row["overlapping"]),
                "workgroups": int(row["workgroups"]),
                "verdict": row["verdict"],
            }
            for row in ownership
        ],
    }


def load_timing(name: str) -> dict:
    path = RAW / name
    if not path.exists():
        raise SystemExit(f"missing timing artifact {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = {}
    for key, value in data.items():
        merged[f"{value['geometry']}|{value['arm']}"] = value
    return merged


def arm(order1: dict, order2: dict, geometry: str, arm: str) -> dict | None:
    left = order1.get(f"{geometry}|{arm}")
    right = order2.get(f"{geometry}|{arm}")
    if left is None:
        return None
    rows = []
    for run in (left, right):
        if run is not None:
            rows.append(run["median_gpu_ms"])
    return {
        "geometry": geometry,
        "arm": arm,
        "blocks": left["blocks"],
        "lanes": left["lanes"],
        "workgroups": left["workgroups"],
        "samples_per_order": [left["samples"], right["samples"] if right else None],
        "order1_median_gpu_ms": left["median_gpu_ms"],
        "order2_median_gpu_ms": right["median_gpu_ms"] if right else None,
        # The second order puts the arm last in its process, so it carries no
        # first-arm penalty. That value is the steady state when the two orders
        # disagree; both are reported either way.
        "steady_median_gpu_ms": right["median_gpu_ms"] if right else left["median_gpu_ms"],
        "order1_min_max_gpu_ms": [left["min_gpu_ms"], left["max_gpu_ms"]],
        "order2_min_max_gpu_ms": [right["min_gpu_ms"], right["max_gpu_ms"]] if right else None,
        "order_delta_percent": (
            round(100.0 * (right["median_gpu_ms"] - left["median_gpu_ms"]) / left["median_gpu_ms"], 3)
            if right
            else None
        ),
    }


def main() -> int:
    parity = parse_parity()
    order1 = load_timing("timing-gpu.json")
    order2 = load_timing("timing-gpu-reverse.json")

    arms = [
        "key_production",
        "key_blocks1_lanes64",
        "key_blocks1_lanes32",
        "key_blocks1_lanes16",
        "key_blocks1_lanes8",
        "key_blocks2_lanes64",
        "key_blocks2_lanes32",
        "key_blocks4_lanes64",
        "key_blocks4_lanes16",
        "key_blocks8_lanes64",
        "key_blocks8_lanes8",
        "query_reference",
        "combined_production",
        "combined_blocks1_lanes64",
        "combined_blocks4_lanes64",
    ]
    timing = {}
    for geometry in ("8q-2kv", "4q-2kv"):
        production = arm(order1, order2, geometry, "key_production")
        base = production["steady_median_gpu_ms"]
        rows = []
        for name in arms:
            row = arm(order1, order2, geometry, name)
            if row is None:
                continue
            row["vs_production_key_speedup"] = round(base / row["steady_median_gpu_ms"], 4)
            row["vs_production_key_percent"] = round(
                100.0 * (base - row["steady_median_gpu_ms"]) / base, 3
            )
            rows.append(row)
        timing[geometry] = {"production_key_steady_median_gpu_ms": base, "arms": rows}

    # The one quantity that separates "starved of workgroups" from "bound by
    # per-lane sequential work": the same configuration at two geometries whose
    # workgroup count is identical and whose per-lane iteration count differs 2x.
    per_lane = {}
    for geometry, heads in (("8q-2kv", 8), ("4q-2kv", 4)):
        candidate = arm(order1, order2, geometry, "key_blocks1_lanes64")
        per_lane[geometry] = {
            "heads_per_kv_head": heads // 2,
            "queries_per_lane": (heads // 2) * 64,
            "workgroups": candidate["workgroups"],
            "steady_median_gpu_ms": candidate["steady_median_gpu_ms"],
        }
    per_lane["ratio_8q_over_4q"] = round(
        per_lane["8q-2kv"]["steady_median_gpu_ms"] / per_lane["4q-2kv"]["steady_median_gpu_ms"], 4
    )
    per_lane["ratio_queries_per_lane"] = round(
        per_lane["8q-2kv"]["queries_per_lane"] / per_lane["4q-2kv"]["queries_per_lane"], 4
    )

    hashes = {}
    for name in (
        "source-hashes.txt",
        "shader-provenance.txt",
        "environment.txt",
    ):
        path = RAW / name
        if path.exists():
            hashes[name] = path.read_text(encoding="utf-8")

    summary = {
        "phase": "VIII",
        "subject": "RDNA3 GQA attention-backward key-pass key-block parallelism",
        "base_commit": "560809eead186f832344c15d1a9b70795da3b75d",
        "verdict": "NO-GO for promotion; production key pass retained",
        "parity": parity["summary"],
        "parity_comparison_verdicts": parity["comparison_verdicts"],
        "ownership": parity["ownership_passes"],
        "timing_gpu_ms_per_key_dispatch_mode9": timing,
        "per_lane_work_scaling": per_lane,
        "receipt_hashes": {
            "candidate_source": sha256(
                HERE / "shaders" / "transformer_attention_backward_key_blockexp_rdna3.comp"
            ),
            "candidate_spv": {
                lanes: sha256(
                    HERE / "shaders" / f"transformer_attention_backward_key_blockexp_l{lanes}_rdna3.spv"
                )
                for lanes in (64, 32, 16, 8)
            },
            "parity_log": sha256(RAW / "parity.log"),
            "timing_log_order1": sha256(RAW / "timing-8q2kv-4q2kv.log"),
            "timing_log_order2": sha256(RAW / "timing-reverse.log"),
        },
        "artifacts": hashes,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Markdown tables for the README, printed to stdout so a reader can diff the
    # document against the artifacts.
    for geometry, table in timing.items():
        print(f"\n### {geometry} — GPU ms per key dispatch (mode 9)\n")
        print("| arm | blocks | lanes | workgroups | order 1 | order 2 | steady | vs production |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in table["arms"]:
            order2_cell = (
                f"{row['order2_median_gpu_ms']:.4f}" if row["order2_median_gpu_ms"] else "—"
            )
            print(
                f"| `{row['arm']}` | {row['blocks'] or '—'} | {row['lanes'] or '—'} | "
                f"{row['workgroups']} | {row['order1_median_gpu_ms']:.4f} | {order2_cell} | "
                f"{row['steady_median_gpu_ms']:.4f} | {row['vs_production_key_speedup']:.3f}× |"
            )
    print("\n### per-lane work scaling\n")
    print(json.dumps(per_lane, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
