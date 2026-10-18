#!/usr/bin/env python3
"""Time a multi-step TBPTT sequence under each time-mix projection graph.

The packed-cell A/B in `bench_rwkv_packed_cell_time_mix.py` measures one
`forward_backward` call. A training run pays that call once per timestep inside
a TBPTT window, plus the reverse pass and the state pack/unpack edges, so this
driver re-runs the same comparison end to end: one `RwkvTbpttSequenceOp` per
leg, a whole forward sweep plus reverse pass per sample, and the promotion
checked on the sequence the default sequence would really dispatch.

The three legs differ only in the switches resolved at construction:

* forced fused projection (`HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION`),
* forced composition (`HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION`),
* the shipped default.

The binary refuses to report a timing unless the fused leg dispatched no vendor
`linear3-forward` module, the composed leg dispatched at least one, the default
sequence followed the tuned module's admission, and all returned tensors agree
within tolerance. This driver adds the cross-batch verdict: the composed
projection may only be promoted where the tuned module ran, and the fused
kernel must stay everywhere else.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_tbptt_time_mix.py
    python hierarchos-vulkan/validation/bench_rwkv_tbptt_time_mix.py \\
        --batch 1 --batch 4 --steps 8 \\
        --record hierarchos-vulkan/validation/packed_cell_tbptt_time_mix_bench.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rwkv_packed_cell_time_mix as single_step  # noqa: E402  (release builder)
import verify_rwkv_cpu_bridge as bridge  # noqa: E402  (fixture constants)
import verify_rwkv_packed_cell_composition as composition  # noqa: E402  (live fixture)

BENCH_BINARY = "hierarchos-vulkan-rwkv-tbptt-time-mix-bench"


def run_batch(
    binary: Path,
    model_dir: Path,
    batch: int,
    *,
    steps: int,
    iterations: int,
    warmup: int,
    atol: float,
    rtol: float,
) -> dict:
    completed = subprocess.run(
        [
            str(binary),
            "--model-dir",
            str(model_dir),
            "--cell-prefix",
            bridge.CELL_PREFIX,
            "--adapter-prefix",
            bridge.ADAPTER_PREFIX,
            "--head-size",
            str(bridge.HEAD_SIZE),
            "--input-dim",
            str(bridge.INPUT_DIM),
            "--batch",
            str(batch),
            "--steps",
            str(steps),
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--atol",
            repr(atol),
            "--rtol",
            repr(rtol),
        ],
        cwd=bridge.REPO,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise bridge.Mismatch(
            f"TBPTT benchmark failed at batch {batch}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout)


def summarise(runs: list[dict]) -> dict:
    ratios = sorted(run["composed_over_fused"] for run in runs)
    middle = len(ratios) // 2
    median_ratio = (
        ratios[middle] if len(ratios) % 2 else (ratios[middle - 1] + ratios[middle]) * 0.5
    )
    composed_wins = all(ratio < 1.0 for ratio in ratios)
    fused_wins = all(ratio > 1.0 for ratio in ratios)
    # Same contract the single-step receipt enforces, now over a whole sequence:
    # compose where the tuned linear3 module ran, keep the fused kernel
    # everywhere else.
    promoted_where_vendor = all(
        run["default"]["label"] == "composed"
        for run in runs
        if run["composed"]["linear3_dispatches"] > 0
    ) and all(
        run["default"]["label"].startswith("fused")
        for run in runs
        if run["composed"]["linear3_dispatches"] == 0
    )
    return {
        "result": "pass",
        "device": runs[0]["device"],
        "width": runs[0]["width"],
        "head_size": runs[0]["head_size"],
        "state_size": runs[0]["state_size"],
        "steps": runs[0]["steps"],
        "linear3_variant": runs[0]["linear3_variant"],
        "promotion": "geometry-aware" if promoted_where_vendor else "UNEXPECTED",
        "batches": [
            {
                "batch": run["batch"],
                "fused_ms": run["fused"]["timing"]["median_ms"],
                "composed_ms": run["composed"]["timing"]["median_ms"],
                "default_ms": run["default"]["timing"]["median_ms"],
                "fused_ms_per_step": run["fused"]["ms_per_step"],
                "composed_ms_per_step": run["composed"]["ms_per_step"],
                "default_ms_per_step": run["default"]["ms_per_step"],
                "composed_over_fused": run["composed_over_fused"],
                "default_over_fused": run["default_over_fused"],
                "worst_max_abs": run["worst_max_abs"],
                "fused_label": run["fused"]["label"],
                "composed_label": run["composed"]["label"],
                "default_label": run["default"]["label"],
                "fused_linear3_dispatches": run["fused"]["linear3_dispatches"],
                "composed_linear3_dispatches": run["composed"]["linear3_dispatches"],
                "default_linear3_dispatches": run["default"]["linear3_dispatches"],
                "composed_ran_vendor_linear3": run["composed"]["linear3_dispatches"] > 0,
                "tensors": run["tensors"],
            }
            for run in runs
        ],
        "median_composed_over_fused": median_ratio,
        "winner": "composed" if composed_wins else ("fused" if fused_wins else "mixed"),
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
        help="batch size to time (repeatable; default 1, 4)",
    )
    parser.add_argument("--steps", type=int, default=8, help="TBPTT window length")
    parser.add_argument("--iterations", type=int, default=10, help="timed sequences per graph")
    parser.add_argument("--warmup", type=int, default=2, help="untimed sequences per graph")
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--record", type=Path, help="write the JSON receipt here")
    args = parser.parse_args()
    batches = args.batches or [1, 4]
    if any(batch <= 0 for batch in batches):
        parser.error("--batch values must be positive")
    if args.steps <= 0 or args.iterations <= 0:
        parser.error("--steps and --iterations must be positive")

    binary = single_step.build_release_binary(BENCH_BINARY)
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-tbptt-")
        fixture_root = Path(cleanup.name)
    try:
        model_dir = fixture_root / "model"
        composition.write_package(model_dir)
        runs = []
        for batch in batches:
            run = run_batch(
                binary,
                model_dir,
                batch,
                steps=args.steps,
                iterations=args.iterations,
                warmup=args.warmup,
                atol=args.atol,
                rtol=args.rtol,
            )
            runs.append(run)
            print(
                f"batch {batch:>2} x {run['steps']} steps: fused "
                f"{run['fused']['timing']['median_ms']:.3f} ms "
                f"({run['fused']['ms_per_step']:.3f}/step), composed "
                f"{run['composed']['timing']['median_ms']:.3f} ms "
                f"({run['composed']['ms_per_step']:.3f}/step), default "
                f"{run['default']['timing']['median_ms']:.3f} ms, ratio "
                f"{run['composed_over_fused']:.3f}, worst |diff| "
                f"{run['worst_max_abs']:.3e}, vendor linear3 dispatches "
                f"{run['fused']['linear3_dispatches']}/"
                f"{run['composed']['linear3_dispatches']}/"
                f"{run['default']['linear3_dispatches']}, default "
                f"{run['default']['label']}"
            )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    summary = summarise(runs)
    if summary["promotion"] != "geometry-aware":
        drifted = [
            f"batch {row['batch']}: composed leg ran the tuned module "
            f"{row['composed_ran_vendor_linear3']}, default built {row['default_label']}"
            for row in summary["batches"]
            if (row["default_label"] == "composed") != row["composed_ran_vendor_linear3"]
        ]
        raise bridge.Mismatch(
            "the default sequence stopped following the tuned linear3 module's gate: "
            + "; ".join(drifted)
        )
    summary["samples"] = args.iterations
    summary["warmup"] = args.warmup
    summary["atol"] = args.atol
    summary["rtol"] = args.rtol
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("device", "linear3_variant", "winner", "median_composed_over_fused")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except bridge.Mismatch as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
