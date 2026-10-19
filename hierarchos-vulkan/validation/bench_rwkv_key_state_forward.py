#!/usr/bin/env python3
"""Rank 3 of `FUSED_FORWARD_PRODUCER_AUDIT.md`: key+state forward fusion A/B.

`record_key_state_forward` dispatches the wider-group fused
`rwkv_key_state_forward_fused_wide` kernel when `head_size <= 64` and otherwise the
`key_transform_forward` + `state_forward` chain. The stage is tiny next to a whole
time-mix step, so this driver measures it in isolation: all arms run the same
shader sequence the shipped route dispatches, repeated dispatches share one
submission, and the receipt reports milliseconds per forward.

There are four legs, built in one process from the same fixture and differing only
in construction-time switches (`HIERARCHOS_RWKV_FORCE_FUSED_KEY_STATE_FORWARD`,
`HIERARCHOS_RWKV_FORCE_WIDE_KEY_STATE_FORWARD`,
`HIERARCHOS_RWKV_FORCE_NARROW_KEY_STATE_FORWARD`,
`HIERARCHOS_RWKV_FORCE_STAGED_KEY_STATE_FORWARD`): the shipped wider-group fused
kernel, the narrow one-workgroup-per-head kernel it replaced, the two-dispatch
chain, and the default. Each proves what it ran with the op's own arm label plus a
measured dispatch count (1 for either fused arm against 2 for the chain), and all
four rotate through every sampling position. The narrow arm is the control that
explains the original crossover: it is the kernel that lost to the chain at batch
16, and the receipt carries its ratio alongside the wide arm's so "the wider group
removed the crossover" is a comparison, not a claim. The wide arm is pinned on
every geometry the sweep covers, so the comparison does not depend on the record
having already been re-recorded.

The default leg is checked against `validation/key_state_forward_decisions.json`
where this adapter has an entry, and against the structural default (the fusion)
where it has none.

`ab_harness` owns the shared shape, so this producer's receipt carries the same
schema as ranks 1 and 2, and `--record-decisions` merges this adapter's measured
winners into the record without touching any other adapter's entries.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_key_state_forward.py
    python hierarchos-vulkan/validation/bench_rwkv_key_state_forward.py \\
        --batch 1 --batch 8 --batch 32 \\
        --record hierarchos-vulkan/validation/key_state_forward_bench.json \\
        --record-decisions hierarchos-vulkan/validation/key_state_forward_decisions.json
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

BENCH_BINARY = "hierarchos-vulkan-rwkv-key-state-forward-bench"
PROBE_KEY = "key_state_forward"
VARIANT_KEY = "fused_arm"
RECORD_KIND = "rwkv-key-state-forward"
TENSORS = {"new_state", "tmix", "scaled_k", "kk", "saved_sa"}
WIDE_ARM = "key-state-forward-fused-wide"
NARROW_ARM = "key-state-forward-fused"

RECORD_NOTES = (
    "One entry per (adapter, width, batch) the key-transform/state-update forward route was "
    "measured at. `route` is the winning graph, from the three candidates the stage builds: "
    "`wide-fused` is the wider-group fused kernel (64 / head_size heads per workgroup, so both "
    "the key transform and the state update keep every lane busy), `narrow-fused` is the "
    "one-workgroup-per-head kernel it replaced, and `staged` is the two-dispatch chain "
    "(`key_transform_forward` then `state_forward`). `candidates` holds each candidate's "
    "measured median milliseconds, so the recorded route has to agree with the measurement "
    "that produced it; the loader rejects an entry whose route is not its fastest candidate. "
    "Naming each fused arm as its own route is what replaced the earlier compose-plus-arm "
    "pair. Both arms are measured in the same sweep, so the arm is a measurement rather than "
    "a batch threshold; the runtime reads this table wherever a fused kernel exists "
    "(head_size <= 64), so a device with no entry here keeps the structural default, which is "
    "the fusion with the wider-group arm. An entry may carry `parameter_storage` to scope the "
    "whole route, arm included, to one storage policy; the policy-less entry is what every "
    "other policy reads. An empty list means no adapter has been measured yet. A geometry "
    "measured inside the 2% noise band is a tie and carries no entry at all: the driver "
    "withdraws it and the structural default decides it until a quiet run measures a sign "
    "outside the band. Re-record a new adapter's own entries with `--record-decisions`."
)


def run_batch(
    binary: Path,
    model_dir: Path,
    work: Path,
    batch: int,
    *,
    iterations: int,
    repetitions: int,
    atol: float,
    rtol: float,
) -> dict:
    case_path = work / f"key-state-case-batch-{batch}.json"
    case_path.write_text(json.dumps(ab.key_state_case(batch)), encoding="utf-8")
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
            "--repetitions",
            str(repetitions),
            "--atol",
            repr(atol),
            "--rtol",
            repr(rtol),
        ],
        batch=batch,
        label="key+state forward benchmark",
    )
    if run["fused_arm"] != WIDE_ARM or run["narrow_arm"] != NARROW_ARM:
        raise bridge.Mismatch(
            f"batch {batch}: fused/narrow legs named {run['fused_arm']!r}/"
            f"{run['narrow_arm']!r}, expected {WIDE_ARM!r}/{NARROW_ARM!r}"
        )
    if run["fused"]["key_state_forward_dispatches"] != 1:
        raise bridge.Mismatch(
            f"batch {batch}: fused leg dispatched the route "
            f"{run['fused']['key_state_forward_dispatches']} time(s), expected one producer"
        )
    if run["narrow"]["key_state_forward_dispatches"] != 1:
        raise bridge.Mismatch(
            f"batch {batch}: narrow leg dispatched the route "
            f"{run['narrow']['key_state_forward_dispatches']} time(s), expected one producer"
        )
    # The record names one route, so the arm the default must build follows from
    # that route rather than from a second field that could disagree with it.
    expected_default_arm = {
        "narrow-fused": NARROW_ARM,
        "wide-fused": WIDE_ARM,
        "staged": None,
        None: WIDE_ARM if run["default"]["label"] == "fused" else None,
    }[run["recorded_route"]]
    if run["default_arm"] != expected_default_arm:
        raise bridge.Mismatch(
            f"batch {batch}: the default leg built {run['default_arm']!r} but the record names "
            f"the {run['recorded_route']!r} route; expected {expected_default_arm!r}"
        )
    if run["composed"]["key_state_forward_dispatches"] != 2:
        raise bridge.Mismatch(
            f"batch {batch}: composed leg dispatched the route "
            f"{run['composed']['key_state_forward_dispatches']} time(s), expected the two-kernel chain"
        )
    ab.require_tensors(run, TENSORS)
    return run


def per_forward(run: dict) -> dict:
    """A view of one run whose timings are milliseconds of a single forward.

    The binary times `repetitions` dispatches in one submission, because a single
    key/state dispatch is too small for a per-submission timer to resolve. Both
    arms are divided by the same repetition count, so the ratio is unchanged; the
    record then stores physical per-forward milliseconds rather than batch totals.
    """
    view = dict(run)
    for leg in ("fused", "narrow", "composed", "default"):
        view[leg] = {**run[leg], "timing": dict(run[leg]["timing"])}
        view[leg]["timing"]["median_ms"] = run[leg]["ms_per_forward"]
    return view


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
    parser.add_argument(
        "--iterations", type=int, default=30, help="timed interleaved samples per graph"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=50,
        help="dispatches per timed sample (a key/state dispatch is too small to time alone)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="untimed warm-up samples per graph")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
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
    if args.iterations <= 0 or args.repetitions <= 0:
        parser.error("--iterations and --repetitions must be positive")

    binary = ab.build_release_binary(BENCH_BINARY)
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-key-state-")
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
                repetitions=args.repetitions,
                atol=args.atol,
                rtol=args.rtol,
            )
            runs.append(run)
            print(
                f"batch {batch:>2}: wide {run['fused']['ms_per_forward'] * 1e3:.3f} us "
                f"({run['fused_arm']}), narrow "
                f"{run['narrow']['ms_per_forward'] * 1e3:.3f} us, composed "
                f"{run['composed']['ms_per_forward'] * 1e3:.3f} us, default "
                f"{run['default']['ms_per_forward'] * 1e3:.3f} us ({run['default']['label']}), "
                f"composed/wide {run['composed_over_fused']:.4f}, "
                f"narrow/wide {run['narrow_over_fused']:.4f}, "
                f"composed/narrow {run['composed_over_narrow']:.4f}, arm "
                f"{run['default_arm'] or 'chain'}, worst |diff| "
                f"{run['worst_max_abs']:.3e} (wide vs narrow "
                f"{run['worst_max_abs_wide_vs_narrow']:.3e}), route dispatches "
                f"{run['fused']['key_state_forward_dispatches']}/"
                f"{run['narrow']['key_state_forward_dispatches']}/"
                f"{run['composed']['key_state_forward_dispatches']}/"
                f"{run['default']['key_state_forward_dispatches']}"
            )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    views = [per_forward(run) for run in runs]
    summary = ab.summarise(
        views, probe_key=PROBE_KEY, variant_key=VARIANT_KEY, structural_default=True
    )
    summary["samples"] = args.iterations
    summary["repetitions"] = args.repetitions
    summary["warmup"] = args.warmup
    summary["atol"] = args.atol
    summary["rtol"] = args.rtol
    summary["unit"] = (
        "milliseconds of one key/state forward; each timed sample is `repetitions` dispatches "
        "in one submission, divided by the repetition count"
    )
    summary["fused_arm"] = runs[0]["fused_arm"]
    summary["narrow_arm"] = runs[0]["narrow_arm"]
    summary["composed_arm"] = runs[0]["composed_arm"]
    summary["arms_note"] = (
        "`fused` is the shipped wider-group kernel (64 / head_size heads per workgroup), "
        "`narrow` is the one-workgroup-per-head kernel it replaced, `composed` is the "
        "two-dispatch chain and `default` is the route the decision record selects. "
        "`narrow_over_fused` and `composed_over_narrow` are what show whether widening the "
        "group moved the crossover instead of assuming it did, and the record is written from "
        "all three candidate timings."
    )
    # The shared summary carries the route legs; the arm comparison is this
    # producer's own dimension, so it is attached per row rather than widening the
    # schema every A/B shares.
    for row, run in zip(summary["batches"], views):
        row["narrow_ms"] = run["narrow"]["timing"]["median_ms"]
        row["narrow_over_fused"] = run["narrow_over_fused"]
        row["composed_over_narrow"] = run["composed_over_narrow"]
        row["worst_max_abs_wide_vs_narrow"] = run["worst_max_abs_wide_vs_narrow"]
        row["fused_arm"] = run["fused_arm"]
        row["narrow_arm"] = run["narrow_arm"]
        row["recorded_route"] = run["recorded_route"]
        row["default_arm"] = run["default_arm"]
        row["narrow_key_state_forward_dispatches"] = run["narrow"][
            "key_state_forward_dispatches"
        ]
    summary["arm_rule"] = (
        "`fused_arm` names the fused kernel each leg is: `wide` is the wider-group variant "
        "(64 / head_size heads per workgroup), `narrow` the one-workgroup-per-head kernel. "
        "`recorded_route` is the single field the record stores per geometry -- `wide-fused`, "
        "`narrow-fused` or `staged` -- so the arm is the route rather than a second field. "
        "`default_arm` is what the shipped default built, or null where it dispatched the chain."
    )
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.record_decisions is not None:
        record, omitted = ab.record_decisions(
            args.record_decisions,
            views,
            kind=RECORD_KIND,
            probe_key=PROBE_KEY,
            notes=RECORD_NOTES,
            measured_by="validation/bench_rwkv_key_state_forward.py",
            margin=args.decision_margin,
            probe_label="key-state-forward",
            candidates=lambda run: {
                "wide-fused": run["fused"]["timing"]["median_ms"],
                "narrow-fused": run["narrow"]["timing"]["median_ms"],
                "staged": run["composed"]["timing"]["median_ms"],
            },
        )
        for entry in record["decisions"]:
            timings = ", ".join(
                f"{label} {ms * 1e3:.3f} us" for label, ms in entry["candidates"].items()
            )
            print(
                f"decision {entry['device']} width {entry['width']} batch {entry['batch']:>2}: "
                f"{entry['route']} ({timings})"
            )
        if omitted:
            print(f"omitted inside the noise band: batches {sorted(omitted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
