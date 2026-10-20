#!/usr/bin/env python3
"""Close the rank-2 scope gap: measure the FP16 packed-mirror producer.

The audited rank-2 receipt (`bench_rwkv_packed_cell_low_rank_forward.py`) measures
the packed cell, whose low-rank parameters are FP32, so its fused arm is the
portable first-stage producer. The deep
`low_rank_full_forward_fused_fp16_packed` arm only exists under the training
graph's `fp16-storage-fp32-compute` policy, which installs packed FP16 mirrors the
packed cell does not have.

This driver measures that arm where the policy actually feeds it: on the
pre-normalized `RwkvLowRankOp` route, with the op's own FP32 masters quantized into
packed FP16 mirrors first. The two graphs are the deep producer (one dispatch)
against the staged chain (the interpolation, the three first-stage matmuls with
packed FP16 operands, the activation and the six second-stage dispatches), and the
default leg is the shipped route from `low_rank_full_forward_decisions.json`.

The gap asked whether the FP16 arm *follows the same per-geometry route*. That is
two questions and the receipt answers both:

* which graph the fused leg ran — the arm name plus the deep producer's own
  dispatch counter (the route counter alone cannot tell it from the first-stage
  producer, since both are one dispatch);
* whether the measured sign for each geometry agrees with the record entry the
  shipped route consulted. Disagreeing is a finding, not a failure of the
  measurement: the driver reports it, and the tests require the receipt to carry
  an explicit answer for every recorded geometry rather than assert agreement.

The record is keyed by parameter-storage policy, so this driver writes its
verdicts under `fp16-storage` and never touches the policy-less entries the
dense route reads. That is what lets the FP16 route take a staged win at a
geometry where the FP32 arm keeps the fusion: both verdicts exist side by side,
each measured on the graph it describes.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_low_rank_fp16_forward.py
    python hierarchos-vulkan/validation/bench_rwkv_low_rank_fp16_forward.py \\
        --batch 1 --batch 8 --batch 32 \\
        --record hierarchos-vulkan/validation/low_rank_fp16_forward_bench.json \\
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

BENCH_BINARY = "hierarchos-vulkan-rwkv-low-rank-fp16-forward-bench"
PROBE_KEY = "low_rank_forward"
VARIANT_KEY = "fused_arm"
PRECISION_POLICY = "fp16-storage-fp32-compute"
PARAMETER_STORAGE = "fp16-storage"
# This route's staged wins are 1.4-3.5% and have never flipped sign, so it records
# at 1% rather than the dense route's 2%; the record keeps declaring the dense
# band and the guard carries this one per producer.
DECISION_BAND = 0.01
DEEP_ARM = "low-rank-full-forward-fused-fp16-packed"
FIRST_STAGE_ARM = "low-rank-producer-forward-fused-fp16-packed"
DECISIONS = Path(__file__).resolve().parent / "low_rank_full_forward_decisions.json"
RECORD_NOTES = (
    "One entry per (adapter, width, batch) the LN1/low-rank forward route was measured at, "
    "keyed by parameter-storage policy where the policy has its own measurement. `route` is "
    "the winning graph for that geometry: `staged` means the staged chain (a separate layer "
    "norm, then the interpolation, the three first-stage matmuls and the second-stage matmuls "
    "on their own dispatches) won, `fused` means the single LN1/low-rank producer that keeps "
    "one row's w/a/g rank activations resident won. `candidates` holds the measured median "
    "milliseconds of each candidate, so the recorded route has to be its fastest one. The "
    "entries this driver writes carry `parameter_storage: fp16-storage` and are the deep FP16 "
    "producer's own verdicts: the runtime reads them only for a consumer running that storage "
    "policy, and falls back to the policy-less entry everywhere else, so recording the FP16 "
    "crossover cannot move the FP32 route. The several `fp16-storage-*` training policies "
    "differ in compute, not storage, so they share this one verdict. This route records at "
    "a 1% band rather than the dense route's 2%: its staged wins are 1.4-3.5% and have "
    "never flipped sign, so the narrower band is what keeps the batch-8 crossover on the "
    "record. A geometry measured inside that band is a tie and carries no entry for this "
    "policy at all; the record's own `margin` stays the dense band, so this scoped "
    "recording does not relabel entries it does not own. A geometry whose winner is not "
    "bit-identical "
    "between the two graphs is recorded and also listed in `withheld`: the training "
    "trajectory is bit-exact and cannot adopt a route that changes the values, so the "
    "runtime keeps the structural fused graph there while the measurement stays auditable. "
    "A rerun that measures the two graphs bit-identical drops that entry and adopts the "
    "route, so closing the rounding gap needs no hand edit. Re-record its own entries with "
    "`--record-decisions`."
)

WITHHELD_REASON = (
    "The fused and staged graphs differ under fp16-storage parameter values "
    "(worst |diff| above zero in this driver's parity comparison), while the training "
    "trajectory is held bit-exact. The measured win stays in the record as evidence, but "
    "the runtime does not adopt it and keeps the structural fused graph."
)

TENSORS = {
    "output",
    "w",
    "g",
    "grad_x_norm",
    "grad_previous",
    "grad_w1",
    "grad_w2",
    "grad_a1",
    "grad_a2",
    "grad_g1",
    "grad_g2",
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
    case_path = work / f"low-rank-fp16-case-batch-{batch}.json"
    case_path.write_text(json.dumps(ab.low_rank_case(batch)), encoding="utf-8")
    run = ab.run_json(
        [
            str(binary),
            "--case",
            str(case_path),
            "--model-dir",
            str(model_dir),
            "--prefix",
            bridge.CELL_PREFIX,
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
        label="FP16 low-rank forward benchmark",
    )
    if run["precision_policy"] != PRECISION_POLICY:
        raise bridge.Mismatch(
            f"batch {batch}: receipt reports policy {run['precision_policy']!r}, "
            f"expected {PRECISION_POLICY!r}"
        )
    if run["parameter_storage"] != PARAMETER_STORAGE:
        raise bridge.Mismatch(
            f"batch {batch}: receipt reports storage {run['parameter_storage']!r}, "
            f"expected {PARAMETER_STORAGE!r}; the FP16 arm was not measured"
        )
    if run["fused_arm"] not in (DEEP_ARM, FIRST_STAGE_ARM):
        raise bridge.Mismatch(
            f"batch {batch}: fused leg named {run['fused_arm']!r}, which is not an FP16 arm"
        )
    if run["fused_arm"] == DEEP_ARM and run["fused"]["deep_forward_arm_dispatches"] != 1:
        raise bridge.Mismatch(
            f"batch {batch}: the deep arm was named but dispatched "
            f"{run['fused']['deep_forward_arm_dispatches']} time(s) on the fused leg"
        )
    if run["composed"]["deep_forward_arm_dispatches"] != 0:
        raise bridge.Mismatch(
            f"batch {batch}: the staged leg dispatched the deep producer, so it did not run the chain"
        )
    ab.require_tensors(run, TENSORS)
    return run


def withheld_route(
    record: dict, device: str, width: int, batch: int, parameter_storage: str
) -> dict | None:
    """The measured-but-not-adopted route for this geometry, if the record has one."""
    policy = ab.canonical_policy(parameter_storage)
    for entry in record.get("withheld", []):
        if (
            entry["device"].strip().lower() != device.strip().lower()
            or entry["width"] != width
            or entry["batch"] != batch
        ):
            continue
        entry_policy = ab.canonical_policy(entry.get("parameter_storage"))
        if entry_policy is None or entry_policy == policy:
            return entry
    return None


def withheld_reason(run: dict) -> str | None:
    """Why this staged win may not ship, or `None` when it may.

    A staged win is adopted only when the two graphs are bit-identical: the
    training trajectory is held bit-exact, so a rerun that closes the rounding
    gap (`worst_max_abs` exactly zero) lifts the refusal on its own. The next
    `--record-decisions` drops the older `withheld` entry for that geometry and
    records the route, which is what makes an improved FP16 route a measurement
    rather than a hand edit.
    """
    if run["composed"]["timing"]["median_ms"] >= run["fused"]["timing"]["median_ms"]:
        return None
    return None if run["worst_max_abs"] == 0.0 else WITHHELD_REASON


def effective_route(
    record: dict, device: str, width: int, batch: int, parameter_storage: str
) -> dict | None:
    """The entry the runtime actually adopts: `route_scoped` minus withheld routes.

    A withheld scoped route does not fall back to the shared verdict, because the
    reason it was withheld applies to the graph change itself; the structural
    default decides that geometry instead.
    """
    policy = ab.canonical_policy(parameter_storage)
    scoped = None
    shared = None
    for entry in record.get("decisions", []):
        if (
            entry["device"].strip().lower() != device.strip().lower()
            or entry["width"] != width
            or entry["batch"] != batch
        ):
            continue
        entry_policy = ab.canonical_policy(entry.get("parameter_storage"))
        if entry_policy is None:
            shared = entry
        elif entry_policy == policy:
            scoped = entry
    held = withheld_route(record, device, width, batch, parameter_storage)
    if scoped is not None:
        return None if held is not None and held["route"] == scoped["route"] else scoped
    if shared is None or (held is not None and held["route"] == shared["route"]):
        return None
    return shared


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
        help="merge this policy's measured winners into the decision record",
    )
    parser.add_argument(
        "--decision-margin",
        type=float,
        default=DECISION_BAND,
        help="the band inside which this run reports a tie against the recorded route",
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
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-low-rank-fp16-")
        fixture_root = Path(cleanup.name)
    record = json.loads(DECISIONS.read_text(encoding="utf-8"))
    declared_margin = record.get("margin", args.decision_margin)
    if args.record_decisions is not None and args.record_decisions.exists():
        declared_margin = json.loads(
            args.record_decisions.read_text(encoding="utf-8")
        ).get("margin", args.decision_margin)
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
                f"({run['fused_arm']}), composed "
                f"{run['composed']['timing']['median_ms']:.3f} ms, default "
                f"{run['default']['timing']['median_ms']:.3f} ms "
                f"({run['default']['label']}), ratio {run['composed_over_fused']:.3f}, "
                f"worst |diff| {run['worst_max_abs']:.3e}, route dispatches "
                f"{run['fused']['low_rank_forward_dispatches']}/"
                f"{run['composed']['low_rank_forward_dispatches']}/"
                f"{run['default']['low_rank_forward_dispatches']}, deep arm "
                f"{run['fused']['deep_forward_arm_dispatches']}"
            )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    summary = ab.summarise(
        runs, probe_key=PROBE_KEY, variant_key=VARIANT_KEY, structural_default=True
    )
    summary["samples"] = args.iterations
    summary["warmup"] = args.warmup
    summary["atol"] = args.atol
    summary["rtol"] = args.rtol
    summary["precision_policy"] = PRECISION_POLICY
    summary["parameter_storage"] = PARAMETER_STORAGE
    summary["deep_arm"] = DEEP_ARM
    summary["decision_margin"] = args.decision_margin
    summary["record"] = str(DECISIONS.relative_to(Path(__file__).resolve().parents[2]))

    routes = []
    for run in runs:
        entry = effective_route(
            record, summary["device"], summary["width"], run["batch"], PARAMETER_STORAGE
        )
        held = withheld_route(
            record, summary["device"], summary["width"], run["batch"], PARAMETER_STORAGE
        )
        recorded = entry["route"] if entry is not None else None
        shipped_compose = recorded == "staged"
        measured_compose = run["composed_over_fused"] < 1.0
        measured_route = "staged" if measured_compose else "fused"
        tie = abs(run["composed_over_fused"] - 1.0) <= args.decision_margin
        if held is not None and held["route"] == measured_route and not shipped_compose:
            status = "withheld"
        elif tie:
            status = "tie"
        elif measured_compose == shipped_compose:
            status = "agrees"
        else:
            status = "diverges"
        routes.append(
            {
                "batch": run["batch"],
                "recorded_route": recorded,
                "recorded_staged_over_fused": (
                    entry["candidates"]["staged"] / entry["candidates"]["fused"]
                    if entry is not None
                    else None
                ),
                "shipped_route": "staged" if shipped_compose else "fused",
                "measured_route": "staged" if measured_compose else "fused",
                "measured_composed_over_fused": run["composed_over_fused"],
                "measured_inside_band": tie,
                "route_status": status,
                "withheld_route": held["route"] if held is not None else None,
                "withheld_reason": held["reason"] if held is not None else None,
                "fused_arm": run["fused_arm"],
            }
        )
    summary["route_following"] = routes
    summary["follows_recorded_route_everywhere"] = all(
        route["route_status"] != "diverges" for route in routes
    )
    summary["route_following_note"] = (
        "`shipped_route` is the graph the shipped default dispatches for this geometry: the "
        "`fp16-storage` entry where one exists, otherwise the policy-less entry the dense "
        "route also reads, minus any route the record withholds. `measured_route` is what "
        "this FP16 run measured, and `route_status` is agrees / tie / withheld / diverges. "
        "A withheld row is a measured win the runtime may not adopt because the graphs are "
        "not bit-identical while the training trajectory is; it is recorded as evidence and "
        "the structural fused graph stays. A rerun that measures the same win bit-identical "
        "drops the withheld entry and adopts it. A diverging row means the FP16 arm does not yet "
        "follow its own measured crossover; recording it with `--record-decisions` writes "
        "that verdict under this storage policy only, so the FP32 route is untouched."
    )
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.record_decisions is not None:
        previously_held = (
            {
                entry["batch"]
                for entry in json.loads(
                    args.record_decisions.read_text(encoding="utf-8")
                ).get("withheld", [])
                if ab.canonical_policy(entry.get("parameter_storage"))
                == PARAMETER_STORAGE
            }
            if args.record_decisions.exists()
            else set()
        )
        merged, omitted = ab.record_decisions(
            args.record_decisions,
            runs,
            kind="rwkv-low-rank-full-forward",
            probe_key=PROBE_KEY,
            notes=RECORD_NOTES,
            measured_by="validation/bench_rwkv_low_rank_fp16_forward.py",
            margin=args.decision_margin,
            # The file's own band is the dense route's; this scoped recording is
            # stricter and must not relabel the entries it does not own.
            record_margin=declared_margin,
            probe_label="low-rank-forward",
            parameter_storage=PARAMETER_STORAGE,
            withhold_when=withheld_reason,
        )
        for entry in merged["decisions"]:
            if ab.canonical_policy(entry.get("parameter_storage")) == PARAMETER_STORAGE:
                timings = ", ".join(
                    f"{label} {ms:.4f} ms" for label, ms in entry["candidates"].items()
                )
                print(
                    f"decision [{PARAMETER_STORAGE}] batch {entry['batch']:>2}: "
                    f"{entry['route']} ({timings})"
                )
        if omitted:
            print(
                f"omitted inside the {args.decision_margin:.1%} noise band for "
                f"{PARAMETER_STORAGE}: batches {', '.join(str(batch) for batch in omitted)}"
            )
        held = [
            entry["batch"]
            for entry in merged.get("withheld", [])
            if ab.canonical_policy(entry.get("parameter_storage")) == PARAMETER_STORAGE
        ]
        if held:
            print(
                f"withheld from the runtime because the graphs are not bit-identical: "
                f"batches {', '.join(str(batch) for batch in held)}"
            )
        adopted = sorted(
            batch
            for batch in previously_held - set(held)
            # Only a geometry whose merged verdict is still the chain counts as
            # adopted; a rerun that put the fused arm back has nothing to adopt.
            if any(
                entry["batch"] == batch
                and ab.canonical_policy(entry.get("parameter_storage"))
                == PARAMETER_STORAGE
                and entry["route"] == "staged"
                for entry in merged["decisions"]
            )
        )
        if adopted:
            print(
                "adopted a bit-identical staged win (its withheld entry was dropped): "
                f"batches {', '.join(str(batch) for batch in adopted)}"
            )
    for route in routes:
        print(
            f"route batch {route['batch']:>2}: shipped {route['shipped_route']}, measured "
            f"{route['measured_route']} ({route['measured_composed_over_fused']:.4f}), "
            f"recorded {route['recorded_route']}, {route['route_status']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
