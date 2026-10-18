#!/usr/bin/env python3
"""Measure the packed cell's fused time-mix projection against its composition.

`RwkvPackedCellOp` has two forward graphs for the time-mix projection:

* the fused projection/key/state kernel, which is what the packed cell builds by
  default, and
* the `linear3-forward` composition, selected by
  `HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION`.

This driver builds the cell both ways over the same fixture, checks the two legs
really took different graphs (the fused one must dispatch no vendor `linear3`
module, the composed one must dispatch at least one), interleaves their
`forward_backward` calls in one process, and compares the returned tensors under
the same tolerance the parity harness uses. A timing that compared different
work would be rejected rather than reported.

The binary records per-tensor agreement and both dispatch counts, so the receipt
is self-checking: the numbers in the JSON prove the A/B was real.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_packed_cell_time_mix.py
    python hierarchos-vulkan/validation/bench_rwkv_packed_cell_time_mix.py \\
        --batch 1 --batch 8 --batch 32 --record hierarchos-vulkan/validation/packed_cell_time_mix_bench.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_rwkv_cpu_bridge as bridge  # noqa: E402  (shared fixture builder)
import verify_rwkv_packed_cell_composition as composition  # noqa: E402  (live adapter fixture)

BENCH_BINARY = "hierarchos-vulkan-rwkv-packed-cell-time-mix-bench"
STATE_SIZE = 4 + bridge.HEAD_SIZE


def build_release_binary(name: str) -> Path:
    """A throughput A/B is only meaningful on the optimized shader host build."""
    completed = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(bridge.VULKAN_MANIFEST),
            "--bin",
            name,
            "--message-format=json-render-diagnostics",
        ],
        cwd=bridge.REPO,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    for line in completed.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and message.get("executable"):
            if message["target"]["name"] == name:
                return Path(message["executable"])
    raise RuntimeError(f"Cargo did not report a release executable for {name}")


def make_case(batch: int) -> dict:
    """One fixed-shape payload, reseeded per batch so every batch is distinct."""
    generator = torch.Generator().manual_seed(0x54494D45 + batch)
    state = bridge.tensor((batch, bridge.WIDTH, STATE_SIZE), generator=generator)
    return {
        "batch": batch,
        "width": bridge.WIDTH,
        "head_size": bridge.HEAD_SIZE,
        "input_dim": bridge.INPUT_DIM,
        "state_mode": "explicit-output",
        "state_clamp": 0.0,
        "x": bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
        "token_features": [0.0] * (batch * bridge.INPUT_DIM),
        "packed_state": state.reshape(-1).tolist(),
        "grad_output": bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
        "grad_packed_new_state": bridge.tensor(
            (batch, bridge.WIDTH, STATE_SIZE), generator=generator
        )
        .reshape(-1)
        .tolist(),
    }


def run_batch(
    binary: Path,
    model_dir: Path,
    work: Path,
    batch: int,
    *,
    iterations: int,
    warmup: int,
    atol: float,
    rtol: float,
) -> dict:
    case_path = work / f"case-batch-{batch}.json"
    case_path.write_text(json.dumps(make_case(batch)), encoding="utf-8")
    completed = subprocess.run(
        [
            str(binary),
            "--case",
            str(case_path),
            "--model-dir",
            str(model_dir),
            "--cell-prefix",
            bridge.CELL_PREFIX,
            "--adapter-prefix",
            bridge.ADAPTER_PREFIX,
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
            f"benchmark failed at batch {batch}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout)


DECISIONS_NOTES = (
    "One entry per (adapter, width, batch) the projection crossover was measured at. "
    "`compose` is the winning graph for that geometry; `composed_over_fused` is the recorded "
    "median ratio that produced it (< 1 means the composition won). The runtime reads this "
    "table before falling back to the tuned linear3-forward module's admission, so a device "
    "with no entry here still composes exactly where its own vendor module is admitted."
)


def _device_key(name: str) -> str:
    return name.strip().lower()


def record_decisions(path: Path, runs: list[dict]) -> dict:
    """Merge this run's per-geometry winners into a decision record.

    Entries for every other adapter, and for this adapter's other widths, are
    preserved verbatim, so recording on a second GPU never loses the first
    one's crossover. The record carries the ratio that produced each decision:
    a hand-edited entry then has to lie about its own measurement, and the
    embedded-table check in `rwkv_time_mix_decisions` rejects it at build time.
    """
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    device = runs[0]["device"]
    width = runs[0]["width"]
    measured = {run["batch"] for run in runs}
    kept = [
        entry
        for entry in existing.get("decisions", [])
        if not (
            _device_key(entry["device"]) == _device_key(device)
            and entry["width"] == width
            and entry["batch"] in measured
        )
    ]
    kept.extend(
        {
            "device": device,
            "width": width,
            "batch": run["batch"],
            "compose": run["composed_over_fused"] < 1.0,
            "composed_over_fused": run["composed_over_fused"],
            "fused_ms": run["fused"]["timing"]["median_ms"],
            "composed_ms": run["composed"]["timing"]["median_ms"],
            "source": "measured",
        }
        for run in runs
    )
    kept.sort(key=lambda entry: (_device_key(entry["device"]), entry["width"], entry["batch"]))
    record = {
        "schema": 1,
        "kind": "rwkv-time-mix-projection",
        "measured_by": "validation/bench_rwkv_packed_cell_time_mix.py",
        "notes": existing.get("notes", DECISIONS_NOTES),
        "decisions": kept,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def summarise(runs: list[dict]) -> dict:
    ratios = [run["composed_over_fused"] for run in runs]
    ratios.sort()
    middle = len(ratios) // 2
    median_ratio = (
        ratios[middle] if len(ratios) % 2 else (ratios[middle - 1] + ratios[middle]) * 0.5
    )
    composed_wins = all(ratio < 1.0 for ratio in ratios)
    fused_wins = all(ratio > 1.0 for ratio in ratios)
    # The promotion rule this receipt records: the composition is promoted where
    # the tuned linear3 module runs, the fused kernel stays everywhere else.
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
        "head_size": bridge.HEAD_SIZE,
        "state_size": runs[0]["state_size"],
        "linear3_variant": runs[0]["linear3_variant"],
        "promotion": "geometry-aware" if promoted_where_vendor else "UNEXPECTED",
        "batches": [
            {
                "batch": run["batch"],
                "fused_ms": run["fused"]["timing"]["median_ms"],
                "composed_ms": run["composed"]["timing"]["median_ms"],
                "default_ms": run["default"]["timing"]["median_ms"],
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
        help="batch size to time (repeatable; default 1, 4, 8, 16, 32)",
    )
    parser.add_argument("--iterations", type=int, default=30, help="timed interleaved samples per graph")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warm-up passes per graph")
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--record", type=Path, help="write the JSON receipt here")
    parser.add_argument(
        "--record-decisions",
        type=Path,
        help="merge this adapter's measured winners into a decision record",
    )
    args = parser.parse_args()
    batches = args.batches or [1, 4, 8, 16, 32]
    if any(batch <= 0 for batch in batches):
        parser.error("--batch values must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    binary = build_release_binary(BENCH_BINARY)
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-time-mix-")
        fixture_root = Path(cleanup.name)
    try:
        model_dir = fixture_root / "model"
        composition.write_package(model_dir)
        runs = []
        for batch in batches:
            run = run_batch(
                binary,
                model_dir,
                fixture_root,
                batch,
                iterations=args.iterations,
                warmup=args.warmup,
                atol=args.atol,
                rtol=args.rtol,
            )
            runs.append(run)
            print(
                f"batch {batch:>2}: fused {run['fused']['timing']['median_ms']:.3f} ms "
                f"({run['fused']['label']}), composed "
                f"{run['composed']['timing']['median_ms']:.3f} ms "
                f"({run['composed']['label']}), default "
                f"{run['default']['timing']['median_ms']:.3f} ms "
                f"({run['default']['label']}), ratio "
                f"{run['composed_over_fused']:.3f}, worst |diff| "
                f"{run['worst_max_abs']:.3e}, vendor linear3 dispatches "
                f"{run['fused']['linear3_dispatches']}/"
                f"{run['composed']['linear3_dispatches']}/"
                f"{run['default']['linear3_dispatches']}"
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
            "the default graph stopped following the tuned linear3 module's gate: "
            + "; ".join(drifted)
        )
    summary["samples"] = args.iterations
    summary["warmup"] = args.warmup
    summary["atol"] = args.atol
    summary["rtol"] = args.rtol
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.record_decisions is not None:
        record = record_decisions(args.record_decisions, runs)
        for entry in record["decisions"]:
            print(
                f"decision {entry['device']} width {entry['width']} batch {entry['batch']:>2}: "
                f"{'composed' if entry['compose'] else 'fused'} "
                f"(ratio {entry['composed_over_fused']:.4f})"
            )
    print(json.dumps({key: summary[key] for key in ("device", "linear3_variant", "winner", "median_composed_over_fused")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except bridge.Mismatch as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
