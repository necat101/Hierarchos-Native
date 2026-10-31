#!/usr/bin/env python3
"""Phase attribution for the key/state forward: where the time actually goes.

`bench_rwkv_key_state_forward.py` answers *which* graph wins. This driver answers
*why*, using the device's own GPU timestamps (the `HIERARCHOS_VULKAN_PROFILE_KERNELS`
timestamp queries the scheduler already owns) instead of a wall-clock split: one
dispatch per leg, one submission per leg, and the reported `gpu_ms` is time the GPU
spent executing that kernel, not time the host spent waiting.

The stage has three kernels and four legs, so one run measures every phase twice:

* the fused legs each dispatch exactly one kernel, so their `gpu_ms` is the whole
  fused stage (key transform *and* state update, including the barrier between
  them);
* the staged leg dispatches `key_transform_forward` and `matrix_state_forward`
  separately, so its two kernels give the transform phase and the state phase in
  isolation, on the same fixture and at the same geometry.

That is what the lane-utilization explanation needs. It predicts that the state
phase -- the part that scales with `batch * width` rows -- dominates the stage, and
that the narrow kernel's state phase costs more per row than the wide one's because
only `head_size` of its 64 lanes advance a row, while the chain's `state_forward`
kernel gives every lane one row and its `key_transform_forward` kernel gives every
lane one head. The report prints those per-row costs next to the active-lane count
each phase actually has, so the comparison is a measurement and an accounting, not
a story.

The profile also answers what dominates the *fused* kernel, which has no phase
seam of its own: the chain's state kernel advances the same rows with the same
64-lane occupancy, so `wide_excess_over_chain_state_us` -- the fused dispatch minus
that phase at the same row count -- is the fusion's own cost, its transform phase
plus one launch. Reading that column across batches separates a transform that is
constant from a state phase whose schedule got worse.

The vendor EU counters (Intel Metrics Discovery / GPA) are not reachable from a
plain Vulkan process on this host, so `eu_active_pct` is *not* reported. What is
reported is the quantity those counters would explain: GPU microseconds per phase
and per row, with the exact number of active lanes per phase derived from the
kernel's own dispatch geometry.

Usage:
    python hierarchos-vulkan/validation/profile_rwkv_key_state_forward.py
    python hierarchos-vulkan/validation/profile_rwkv_key_state_forward.py \\
        --batch 1 --batch 16 --record hierarchos-vulkan/validation/key_state_forward_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_harness as ab  # noqa: E402
import verify_rwkv_cpu_bridge as bridge  # noqa: E402
import verify_rwkv_packed_cell_composition as composition  # noqa: E402

BENCH_BINARY = "hierarchos-vulkan-rwkv-key-state-forward-bench"
PROFILE_ENV = "HIERARCHOS_VULKAN_PROFILE_KERNELS"

WIDE_SHADER = "rwkv_key_state_forward_fused_wide"
NARROW_SHADER = "rwkv_key_state_forward_fused"
TRANSFORM_SHADER = "rwkv_key_transform_forward"
STATE_SHADER = "rwkv_matrix_state_forward"

SHADER_LINE = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(?P<shader>\S+) "
    r"category=(?P<category>\S+) dispatches=(?P<dispatches>\d+) "
    r"gpu_ms=(?P<gpu_ms>[0-9.]+)"
)


def run_profiled(binary: Path, model_dir: Path, work: Path, batch: int) -> dict[str, list[float]]:
    """One binary run per geometry, with kernel timestamps on.

    `--iterations 1 --repetitions 1` keeps one submission per leg, so every
    reported `gpu_ms` belongs to exactly one dispatch and the four legs cannot be
    confused with each other's samples. Repeated runs are medianed by the caller.
    """
    case_path = work / f"profile-case-batch-{batch}.json"
    case_path.write_text(json.dumps(ab.key_state_case(batch)), encoding="utf-8")
    completed = subprocess.run(
        [
            str(binary),
            "--case",
            str(case_path),
            "--model-dir",
            str(model_dir),
            "--prefix",
            bridge.CELL_PREFIX,
            "--iterations",
            "1",
            "--repetitions",
            "1",
        ],
        cwd=bridge.REPO,
        env={**os.environ, PROFILE_ENV: "1"},
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise bridge.Mismatch(
            f"profiled key/state run failed at batch {batch}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    samples: dict[str, list[float]] = {}
    phases = {WIDE_SHADER, NARROW_SHADER, TRANSFORM_SHADER, STATE_SHADER}
    for line in completed.stderr.splitlines():
        match = SHADER_LINE.search(line)
        if match is None:
            continue
        shader = match.group("shader")
        # The op's fixture upload and parity checks dispatch other kernels in the
        # same process; they are not phases of this stage and are not attributed.
        if shader not in phases:
            continue
        dispatches = int(match.group("dispatches"))
        if dispatches != 1:
            raise bridge.Mismatch(
                f"batch {batch}: the profile aggregated {dispatches} {shader} dispatches, so a "
                "phase cannot be attributed to one kernel; run with one iteration and one "
                "repetition"
            )
        samples.setdefault(shader, []).append(float(match.group("gpu_ms")))
    if not samples:
        raise bridge.Mismatch(
            f"batch {batch}: the device produced no timestamp samples (set {PROFILE_ENV}=1); "
            "a wall-clock split is not a phase measurement"
        )
    return samples


def lane_accounting(head_size: int) -> dict[str, dict[str, int]]:
    """Active lanes per phase, from each kernel's own dispatch geometry.

    `local_size_x` is 64 everywhere here. The narrow fused kernel runs one head per
    workgroup, so a single lane performs the serial transform and `head_size` lanes
    advance rows. The wide kernel packs `64 / head_size` heads per workgroup and
    maps lane -> (head slot, column), so both phases use all 64. The chain's two
    kernels are one lane per head (`key_transform_forward`) and one lane per
    `(batch, width)` element (`matrix_state_forward`).
    """
    heads_per_group = max(1, 64 // head_size)
    return {
        "wide": {"workgroup": 64, "transform_lanes": 64, "state_lanes": 64},
        "narrow": {"workgroup": 64, "transform_lanes": 1, "state_lanes": head_size},
        "chain": {"workgroup": 64, "transform_lanes": 64, "state_lanes": 64},
        "_heads_per_group": {"workgroup": heads_per_group},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--batch",
        type=int,
        action="append",
        dest="batches",
        help="batch size to profile (repeatable; default 1, 8, 16)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="profiled runs per geometry; the median of each shader's samples is reported",
    )
    parser.add_argument("--record", type=Path, help="write the JSON report here")
    args = parser.parse_args()
    batches = args.batches or [1, 8, 16]
    if any(batch <= 0 for batch in batches):
        parser.error("--batch values must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")

    binary = ab.build_release_binary(BENCH_BINARY)
    cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-key-state-profile-")
    try:
        work = Path(cleanup.name)
        model_dir = work / "model"
        composition.write_package(model_dir)
        head_size = bridge.HEAD_SIZE
        lanes = lane_accounting(head_size)
        rows = []
        for batch in batches:
            collected: dict[str, list[float]] = {}
            for _ in range(args.runs):
                for shader, samples in run_profiled(binary, model_dir, work, batch).items():
                    collected.setdefault(shader, []).extend(samples)
            missing = {
                WIDE_SHADER,
                NARROW_SHADER,
                TRANSFORM_SHADER,
                STATE_SHADER,
            } - set(collected)
            if missing:
                raise bridge.Mismatch(
                    f"batch {batch}: the profile is missing {sorted(missing)}; the four legs did "
                    "not all run their own kernel"
                )
            phase_us = {
                name: statistics.median(collected[shader]) * 1e3
                for name, shader in (
                    ("wide_fused", WIDE_SHADER),
                    ("narrow_fused", NARROW_SHADER),
                    ("chain_transform", TRANSFORM_SHADER),
                    ("chain_state", STATE_SHADER),
                )
            }
            row_count = batch * bridge.WIDTH
            chain_us = phase_us["chain_transform"] + phase_us["chain_state"]
            row = {
                "batch": batch,
                "rows": row_count,
                "head_size": head_size,
                "wide_fused_us": phase_us["wide_fused"],
                "narrow_fused_us": phase_us["narrow_fused"],
                "chain_transform_us": phase_us["chain_transform"],
                "chain_state_us": phase_us["chain_state"],
                "chain_total_us": chain_us,
                "chain_state_share": phase_us["chain_state"] / chain_us,
                "wide_us_per_row": phase_us["wide_fused"] / row_count,
                "narrow_us_per_row": phase_us["narrow_fused"] / row_count,
                "chain_state_us_per_row": phase_us["chain_state"] / row_count,
                "chain_transform_us_per_row": phase_us["chain_transform"] / row_count,
                "wide_over_narrow": phase_us["wide_fused"] / phase_us["narrow_fused"],
                # What the fused dispatch pays over the chain's state phase at the same
                # row count. The chain's state kernel runs the same per-row arithmetic
                # with the same 64-lane occupancy, so this difference is the fused
                # stage's own overhead: its transform phase plus one launch while the
                # two costs a dispatch each to separate. Reading it as a column is how
                # the profile says what dominates the fused kernel's time instead of
                # leaving it as the residual of a total.
                "wide_over_chain_state_us": phase_us["wide_fused"] / phase_us["chain_state"],
                "wide_excess_over_chain_state_us": phase_us["wide_fused"]
                - phase_us["chain_state"],
                "wide_excess_us_per_row": (phase_us["wide_fused"] - phase_us["chain_state"])
                / row_count,
                "wide_fused_lanes": lanes["wide"]["state_lanes"],
                "narrow_state_lanes": lanes["narrow"]["state_lanes"],
                "narrow_transform_lanes": lanes["narrow"]["transform_lanes"],
                "chain_state_lanes": lanes["chain"]["state_lanes"],
                "samples_per_shader": len(collected[WIDE_SHADER]),
            }
            rows.append(row)
            print(
                f"batch {batch:>2}: wide {row['wide_fused_us']:.3f} us "
                f"({row['wide_us_per_row'] * 1e3:.1f} ns/row, {row['wide_fused_lanes']}/64 lanes), "
                f"narrow {row['narrow_fused_us']:.3f} us "
                f"({row['narrow_us_per_row'] * 1e3:.1f} ns/row, "
                f"{row['narrow_state_lanes']}/64 state lanes), chain transform "
                f"{row['chain_transform_us']:.3f} us + state {row['chain_state_us']:.3f} us = "
                f"{chain_us:.3f} us (state share {row['chain_state_share']:.1%}), "
                f"wide over chain state {row['wide_over_chain_state_us']:.3f}x "
                f"(+{row['wide_excess_over_chain_state_us']:.3f} us), "
                f"wide/narrow {row['wide_over_narrow']:.4f}"
            )
        early = [
            row["wide_excess_over_chain_state_us"] for row in rows if row["batch"] <= 16
        ]
        worst = max(rows, key=lambda row: row["wide_excess_over_chain_state_us"])
        report = {
            "schema": 1,
            "kind": "rwkv-key-state-forward-profile",
            "device": run_device_name(binary, model_dir, work, batches[0]),
            "head_size": head_size,
            "width": bridge.WIDTH,
            "runs": args.runs,
            "timestamp_source": (
                "Vulkan timestamp queries per dispatch, read by the scheduler's kernel profile "
                "(`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`); one dispatch per leg, so each number is "
                "the GPU time of that kernel, not host wait time"
            ),
            "lane_accounting": lanes,
            "attribution": {
                "fused_excess_through_batch_16_us": (
                    statistics.median(early) if early else None
                ),
                "largest_excess_batch": worst["batch"],
                "largest_excess_us": worst["wide_excess_over_chain_state_us"],
                "note": (
                    "The chain's state kernel and the fused kernel advance the same rows with the "
                    "same 64-lane occupancy, so the fused stage minus the chain state phase is "
                    "what the fusion costs on top of the row work: its transform phase and its "
                    "single launch. Flat through batch 16 and a jump at a larger batch would put "
                    "the remaining time in the state phase's schedule, not in the transform."
                ),
            },
            "batches": rows,
        }
        if args.record is not None:
            args.record.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        cleanup.cleanup()


def run_device_name(binary: Path, model_dir: Path, work: Path, batch: int) -> str:
    """The adapter name from the same binary, through its normal receipt path."""
    case_path = work / f"name-case-batch-{batch}.json"
    case_path.write_text(json.dumps(ab.key_state_case(batch)), encoding="utf-8")
    completed = subprocess.run(
        [
            str(binary),
            "--case",
            str(case_path),
            "--model-dir",
            str(model_dir),
            "--prefix",
            bridge.CELL_PREFIX,
            "--iterations",
            "1",
            "--repetitions",
            "1",
        ],
        cwd=bridge.REPO,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise bridge.Mismatch(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)["device"]


if __name__ == "__main__":
    sys.exit(main())
