#!/usr/bin/env python3
"""Measure the packed cell's fused time-mix projection against its composition.

`RwkvPackedCellOp` has two forward graphs for the time-mix projection:

* the fused projection/key/state kernel, which is what the packed cell builds by
  default, and
* the `linear3-forward` composition, selected by
  `HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION`.

This driver builds the cell both ways over the same fixture and hands the result
to `ab_harness`, which owns the shared shape: prove each leg ran the graph it
claims, require the returned tensors to agree, rotate the sampling order, fail
closed when the default stops following the recorded decision, and merge this
adapter's measured winners into `time_mix_projection_decisions.json`.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_packed_cell_time_mix.py
    python hierarchos-vulkan/validation/bench_rwkv_packed_cell_time_mix.py \\
        --batch 1 --batch 8 --batch 32 \\
        --record hierarchos-vulkan/validation/packed_cell_time_mix_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_harness as ab  # noqa: E402  (shared three-leg A/B driver harness)
import verify_rwkv_cpu_bridge as bridge  # noqa: E402  (fixture constants)
import verify_rwkv_packed_cell_composition as composition  # noqa: E402  (live fixture)

BENCH_BINARY = "hierarchos-vulkan-rwkv-packed-cell-time-mix-bench"
PROBE_KEY = "linear3"
VARIANT_KEY = "linear3_variant"
RECORD_KIND = "rwkv-time-mix-projection"
TENSORS = {"output", "packed_new_state", "grad_x", "grad_packed_state", "token_feature_grad"}


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
    case_path.write_text(json.dumps(ab.packed_cell_case(batch)), encoding="utf-8")
    return ab.run_json(
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
        batch=batch,
    )


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
        help="merge this adapter's measured winners into the decision record",
    )
    parser.add_argument(
        "--decision-margin",
        type=float,
        default=0.02,
        help="a geometry measured inside this band around 1.0 is omitted and left to the gate",
    )
    args = parser.parse_args()
    batches = args.batches or [1, 4, 8, 16, 32]
    if any(batch <= 0 for batch in batches):
        parser.error("--batch values must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    binary = ab.build_release_binary(BENCH_BINARY)
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
            ab.require_tensors(run, TENSORS)
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

    summary = ab.summarise(runs, probe_key=PROBE_KEY, variant_key=VARIANT_KEY)
    ab.require_geometry_aware(summary)
    summary["samples"] = args.iterations
    summary["warmup"] = args.warmup
    summary["atol"] = args.atol
    summary["rtol"] = args.rtol
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.record_decisions is not None:
        record, omitted = ab.record_decisions(
            args.record_decisions,
            runs,
            kind=RECORD_KIND,
            probe_key=PROBE_KEY,
            measured_by="validation/bench_rwkv_packed_cell_time_mix.py",
            margin=args.decision_margin,
        )
        for entry in record["decisions"]:
            candidates = entry["candidates"]
            print(
                f"decision {entry['device']} width {entry['width']} batch {entry['batch']:>2}: "
                f"{entry['route']} "
                f"(staged/fused {candidates['staged'] / candidates['fused']:.4f})"
            )
        if omitted:
            print(
                f"omitted inside the {args.decision_margin:.1%} noise band (the vendor gate "
                f"decides): batches {', '.join(str(batch) for batch in omitted)}"
            )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("device", VARIANT_KEY, "winner", "median_composed_over_fused")
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
