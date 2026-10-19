#!/usr/bin/env python3
"""Measure the packed cell's fused LN1/low-rank producer against the staged chain.

Rank 2 of `FUSED_FORWARD_PRODUCER_AUDIT.md`. The dense cell forward has the
low-rank a/w/g block in two complete Vulkan graphs:

* the fused LN1/low-rank producer, which keeps one row's w/a/g rank activations
  resident through the layer norm, the interpolation, the first-stage matmuls, the
  activation, the second-stage matmuls, decay and the final gate. This is the
  shipped structural default, and
  `HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN` holds a leg on it at every
  geometry; and
* the staged chain, where the layer norm, the interpolation, the three first-stage
  matmuls, the activations and the second-stage matmuls each dispatch on their own.
  `HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN` forces it, and it is what
  the production default falls back to where a recorded measurement says it wins.

`ab_harness` owns the shared shape: build one cell per leg over the same fixture,
prove each leg ran the graph it claims (label plus the route's own dispatch
count), require every returned tensor to agree, rotate the sampling order, fail
closed when the default stops following the recorded decision, and merge this
adapter's measured winners into `low_rank_full_forward_decisions.json`.

Unlike the channel-mix A/B the two graphs are not bit-identical: the fused arm
normalizes inside the producer while the staged arm runs the standalone
layer-norm kernel, so the two reductions round differently. The receipt therefore
records a tolerance and the tests state it, rather than claiming an exact match
that would quietly turn a numerics change into a schedule change.

Both legs are complete GPU shader sequences; the decision only chooses which one
the default dispatches, and no arithmetic moves to the host.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_packed_cell_low_rank_forward.py
    python hierarchos-vulkan/validation/bench_rwkv_packed_cell_low_rank_forward.py \\
        --batch 1 --batch 8 --batch 32 \\
        --record hierarchos-vulkan/validation/packed_cell_low_rank_forward_bench.json \\
        --record-decisions hierarchos-vulkan/validation/low_rank_full_forward_decisions.json
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

BENCH_BINARY = "hierarchos-vulkan-rwkv-packed-cell-low-rank-forward-bench"
PROBE_KEY = "low_rank_forward"
VARIANT_KEY = "fused_arm"
RECORD_KIND = "rwkv-low-rank-full-forward"
TENSORS = {"output", "packed_new_state", "grad_x", "grad_packed_state", "token_feature_grad"}

RECORD_NOTES = (
    "One entry per (adapter, width, batch) the LN1/low-rank forward route was measured at. "
    "`compose` is the winning route for that geometry: true means the staged chain (a separate "
    "layer norm, then the interpolation, the three first-stage matmuls and the second-stage "
    "matmuls on their own dispatches) won, false means the single LN1/low-rank producer that "
    "keeps one row's w/a/g rank activations resident through the layer norm, the activation, "
    "the second-stage matmuls, decay and the final gate won. `composed_over_fused` is the "
    "recorded median ratio that produced the decision (< 1 means the staged chain won). The "
    "runtime reads this table in the dense cell forward, so a device with no entry here keeps "
    "the structural default (the fused producer wherever the kernels exist). An empty list "
    "means no adapter has been measured yet. A geometry measured inside the noise band is a tie "
    "and carries no entry at all: the driver withdraws it, and the structural default decides "
    "that geometry until a quiet run measures a sign outside the band. The record covers the "
    "dense forward route; the TBPTT packed forward-only and packed-rematerialization paths keep "
    "their own fused chains and are measured by the packed-cell composition A/B, not here. "
    "Entries written here carry no `parameter_storage`, which makes them the default every "
    "storage policy falls back to; a policy that measured a different crossover records its "
    "own policy-scoped entries beside them (the FP16 packed-mirror route does, through "
    "`bench_rwkv_low_rank_fp16_forward.py`) and never rewrites these. Re-record its own "
    "entries with `--record-decisions`."
)


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
        label="low-rank forward benchmark",
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
        help="a geometry measured inside this band around 1.0 is omitted and left to the default",
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
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-low-rank-forward-")
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
                f"{run['worst_max_abs']:.3e}, route dispatches "
                f"{run['fused']['low_rank_forward_dispatches']}/"
                f"{run['composed']['low_rank_forward_dispatches']}/"
                f"{run['default']['low_rank_forward_dispatches']}"
            )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    summary = ab.summarise(
        runs, probe_key=PROBE_KEY, variant_key=VARIANT_KEY, structural_default=True
    )
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
            notes=RECORD_NOTES,
            measured_by="validation/bench_rwkv_packed_cell_low_rank_forward.py",
            margin=args.decision_margin,
            probe_label="low-rank-forward",
        )
        for entry in record["decisions"]:
            print(
                f"decision {entry['device']} width {entry['width']} batch {entry['batch']:>2}: "
                f"{'composed' if entry['route'] == 'staged' else 'fused'} "
                f"(ratio {entry['composed_over_fused']:.4f})"
            )
        if omitted:
            print(
                f"omitted inside the {args.decision_margin:.1%} noise band (the structural "
                f"default decides): batches {', '.join(str(batch) for batch in omitted)}"
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
