#!/usr/bin/env python3
"""Replay the recorded time-mix projection decisions on this adapter.

`validation/time_mix_projection_decisions.json` is the measured per-device
crossover the runtime consumes (`src/decision_records.rs`, compiled into the
binary). This script is its replay: for every geometry recorded for the
attached adapter it re-runs the A/B, and fails when

* the recorded winner no longer wins (the measurement moved), or
* the shipped default does not build the graph the record names (the compiled
  table and the checked-in record disagree), or
* the adapter has no recorded entry at all, because then the replay would be
  vacuous and the device would silently be running on the fallback gate.

A run that skips every geometry is reported as a failure, not a pass: the whole
point of the record is that its decisions are re-measured on the hardware that
reads them.

Usage:
    python hierarchos-vulkan/validation/verify_time_mix_projection_decisions.py
    python hierarchos-vulkan/validation/verify_time_mix_projection_decisions.py \\
        --decisions hierarchos-vulkan/validation/time_mix_projection_decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_harness as ab  # noqa: E402  (shared release builder, for both producers)
import bench_rwkv_packed_cell_time_mix as bench  # noqa: E402  (this producer's runner)
import verify_rwkv_cpu_bridge as bridge  # noqa: E402  (fixture constants)
import verify_rwkv_packed_cell_composition as composition  # noqa: E402  (live fixture)


def device_key(name: str) -> str:
    return name.strip().lower()

# The recorded ratios at the geometries where one graph wins are 0.6-4%; the
# smallest of them are inside this adapter's run-to-run spread (the same batch-1
# pair has measured 1.017 and 0.994 on re-runs). A replay is therefore a
# regression check, not a re-derivation: a decision is stale only when the
# measured ratio crosses to the other side by more than this margin. Reporting a
# sign flip inside the spread would make the verifier fail on scheduling noise.
DECISION_MARGIN = 0.015


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=bridge.ROOT / "validation" / "time_mix_projection_decisions.json",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--keep-fixtures", type=Path)
    args = parser.parse_args()

    record = json.loads(args.decisions.read_text(encoding="utf-8"))
    if record.get("schema") != 2:
        raise bridge.Mismatch(f"unsupported decision record schema {record.get('schema')!r}")
    recorded = record["decisions"]
    if not recorded:
        raise bridge.Mismatch("the decision record carries no entries")

    binary = ab.build_release_binary(bench.BENCH_BINARY)
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-time-mix-decisions-")
        fixture_root = Path(cleanup.name)
    try:
        model_dir = fixture_root / "model"
        composition.write_package(model_dir)

        first = recorded[0]
        probe = bench.run_batch(
            binary,
            model_dir,
            fixture_root,
            first["batch"],
            iterations=args.iterations,
            warmup=args.warmup,
            atol=1.0e-4,
            rtol=1.0e-5,
        )
        device = probe["device"]
        entries = [entry for entry in recorded if device_key(entry["device"]) == device_key(device)]
        if not entries:
            raise bridge.Mismatch(
                f"no recorded time-mix projection decisions for {device!r}; run "
                "validation/bench_rwkv_packed_cell_time_mix.py --batch ... --record-decisions "
                f"{args.decisions.relative_to(bridge.REPO)} on this adapter, then rebuild so the "
                "table is compiled in"
            )

        replayed = 0
        for entry in entries:
            batch = entry["batch"]
            run = (
                probe
                if batch == first["batch"]
                else bench.run_batch(
                    binary,
                    model_dir,
                    fixture_root,
                    batch,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    atol=1.0e-4,
                    rtol=1.0e-5,
                )
            )
            if run["width"] != entry["width"]:
                raise bridge.Mismatch(
                    f"recorded width {entry['width']} for batch {batch} does not match the "
                    f"fixture's {run['width']}"
                )
            ratio = run["composed_over_fused"]
            measured_compose = ratio < 1.0
            recorded_compose = entry["route"] == "staged"
            recorded_ratio = entry["candidates"]["staged"] / entry["candidates"]["fused"]
            expected_label = "composed" if recorded_compose else run["fused"]["label"]
            print(
                f"width {entry['width']} batch {batch:>2}: recorded {entry['route']} "
                f"(ratio {recorded_ratio:.4f}), measured "
                f"{'composed' if measured_compose else 'fused'} (ratio {ratio:.4f}), "
                f"default {run['default']['label']}"
            )
            lost = (
                ratio > 1.0 + DECISION_MARGIN
                if recorded_compose
                else ratio < 1.0 - DECISION_MARGIN
            )
            if lost:
                raise bridge.Mismatch(
                    f"batch {batch}: the recorded route is {entry['route']} at ratio "
                    f"{recorded_ratio:.4f}, but this run measured the composition at "
                    f"{ratio:.4f}, past the {DECISION_MARGIN:.1%} noise band; re-record the decision "
                    "(and rebuild) or restore the kernel that regressed"
                )
            if run["default"]["label"] != expected_label:
                raise bridge.Mismatch(
                    f"batch {batch}: the decision record says the default builds {expected_label}, "
                    f"but this binary built {run['default']['label']}; the compiled table and the "
                    "checked-in record disagree"
                )
            replayed += 1
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    print(
        json.dumps(
            {
                "result": "pass",
                "device": device,
                "replayed": replayed,
                "decisions": str(args.decisions.relative_to(bridge.REPO)),
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
