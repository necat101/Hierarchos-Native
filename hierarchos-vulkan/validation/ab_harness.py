#!/usr/bin/env python3
"""Shared driver-side harness for the three-leg kernel A/B receipts.

Each producer A/B follows one shape, and this module is that shape:

* build the release harness binary (one process per measurement leg group);
* run one geometry at a time and collect the binary's JSON, which already proves
  which graph each leg built and how many tuned dispatches it performed;
* summarise the sweep with the cross-batch verdict: the composition may only be
  promoted where the tuned module actually ran, and the fused graph must stay
  everywhere else;
* fail closed when the default stops following that contract, so a driver run is
  a check and not only a recorder;
* merge the resulting per-geometry winners into a per-device decision record,
  preserving every other adapter's entries.

Producer-specific knowledge is passed in rather than duplicated: the probe
family's receipt key (`linear3` or `linear_residual`), the record `kind`, and
the notes that explain the record. The shared `summarise` emits the same schema
for every producer, so the receipt tests read one shape.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_rwkv_cpu_bridge as bridge  # noqa: E402  (shared Mismatch + repo root)

STATE_SIZE = 4 + bridge.HEAD_SIZE

DEFAULT_RECORD_NOTES = (
    "One entry per (adapter, width, batch) the crossover was measured at. `route` is the "
    "winning graph for that geometry (`staged` or `fused`, and `wide-fused`/`narrow-fused` for "
    "the key/state producer's two fused kernels), and `candidates` holds each candidate's "
    "measured median milliseconds so a recorded route has to agree with the measurement that "
    "produced it. A geometry measured inside the 2% noise band is a tie and carries no entry "
    "at all: the driver withdraws it, and the structural default decides that geometry until a "
    "quiet run measures a sign outside the band. An entry may carry `parameter_storage` to "
    "scope it to one storage policy; the policy-less entry is the fallback every other policy "
    "reads. The runtime reads this table before falling back to its structural default, so a "
    "device with no entry here keeps the conservative default; re-record its own entries with "
    "`--record-decisions`."
)

# Candidate route labels, in the order the record's vocabulary lists them. The
# order only breaks exact ties, which cannot be a measurement.
ROUTE_ORDER = ("fused", "wide-fused", "narrow-fused", "staged")

# The graph the structural default dispatches with no record entry. The key/state
# producer's default is the wider-group fusion; the others default to their fused
# producer (the vendor-gated producers compose where their module is admitted,
# which the receipt's dispatch count proves).
STRUCTURAL_DEFAULT_ROUTE = {"key-state-forward": "wide-fused"}


def canonical_policy(policy: str | None) -> str | None:
    """The storage policy a record scope selects, as `decision_records` reads it.

    The training's `fp16-storage-*` policies share the packed mirrors and differ
    only in compute, and the low-rank op reports the older `fp16-packed-mirror`
    label, so all of them are one storage verdict.
    """
    if policy is None:
        return None
    policy = policy.strip().lower()
    if policy.startswith("fp16"):
        return "fp16-storage"
    if policy.startswith("bf16"):
        return "bf16-storage"
    return policy


def recorded_route(run: dict) -> str | None:
    """The route the compiled-in record names for one measured run."""
    route = run.get("recorded_route")
    if route is not None:
        return route
    # Receipts from a two-way producer that predates the route vocabulary carry
    # the boolean instead; reading it here keeps one code path for all drivers.
    compose = run.get("recorded_compose")
    if compose is None:
        return None
    return "staged" if compose else "fused"


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


def device_key(name: str) -> str:
    return name.strip().lower()


def packed_cell_case(batch: int) -> dict:
    """One fixed-shape packed-cell payload, reseeded per batch.

    The packed-cell A/Bs all use this fixture, so their receipts are comparable
    and a harness added later does not quietly measure a different shape.
    """
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


def low_rank_case(batch: int) -> dict:
    """One fixed-shape pre-normalized low-rank payload, reseeded per batch.

    `RwkvLowRankOp` consumes `x_norm`/`previous` and the three incoming adjoints,
    so the rank-2 FP16 arm A/B stops at the same boundary the op does: no layer
    norm and no cell around it, which is exactly the graph whose producer the
    training graph's FP16-storage policy switches on.
    """
    generator = torch.Generator().manual_seed(0x4C4F5752 + batch)
    return {
        "batch": batch,
        "x_norm": bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
        "previous": bridge.tensor((batch, bridge.WIDTH), generator=generator)
        .reshape(-1)
        .tolist(),
        "grad_a": bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
        "grad_w": bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
        "grad_g": bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
    }


def key_state_case(batch: int) -> dict:
    """One fixed-shape key/state forward payload, reseeded per batch.

    The rank-3 A/B measures the key-transform/state-update stage in isolation, so
    the fixture is the stage's own inputs: the recurrent state plus r/k/v/a/w.
    """
    generator = torch.Generator().manual_seed(0x4B535452 + batch)
    vector = (batch, bridge.WIDTH)
    state = bridge.tensor((batch, bridge.WIDTH, bridge.HEAD_SIZE), generator=generator)
    return {
        "batch": batch,
        "head_size": bridge.HEAD_SIZE,
        "state": state.reshape(-1).tolist(),
        "raw_k": bridge.tensor(vector, generator=generator).reshape(-1).tolist(),
        "a": bridge.tensor(vector, generator=generator).reshape(-1).tolist(),
        "w": bridge.tensor(vector, generator=generator).reshape(-1).tolist(),
        "r": bridge.tensor(vector, generator=generator).reshape(-1).tolist(),
        "v": bridge.tensor(vector, generator=generator).reshape(-1).tolist(),
    }


def run_json(argv: list[str], *, batch: int, label: str = "benchmark") -> dict:
    """Run one harness invocation and parse its receipt, failing on non-zero exit."""
    completed = subprocess.run(argv, cwd=bridge.REPO, capture_output=True, text=True)
    if completed.returncode != 0:
        raise bridge.Mismatch(
            f"{label} failed at batch {batch}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout)


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) * 0.5


def summarise(
    runs: list[dict],
    *,
    probe_key: str,
    variant_key: str,
    steps: int | None = None,
    structural_default: bool = False,
) -> dict:
    """Cross-batch verdict for a three-leg sweep.

    `probe_key` names the probe family's dispatch fields (`linear3_dispatches`
    for the time-mix projection, `linear_residual_dispatches` for the channel
    mix, `low_rank_forward_dispatches` for the LN1/low-rank route), so every
    producer's receipt carries the same schema with the family named in the key.

    `structural_default` says the unrecorded expectation is the shipped default
    graph itself rather than a vendor gate: the rank-2 low-rank route fuses unless
    a measurement says the staged chain won, so its `fused` leg is what an adapter
    with no entry builds. The row then omits the vendor-dispatch flag, which names
    a gate that producer does not have.
    """
    if not runs:
        raise bridge.Mismatch("a sweep needs at least one batch")
    ratios = sorted(run["composed_over_fused"] for run in runs)
    composed_wins = all(ratio < 1.0 for ratio in ratios)
    fused_wins = all(ratio > 1.0 for ratio in ratios)
    # The promotion contract, exactly as the binaries enforce it per geometry:
    # the default follows the recorded decision where this adapter has one, and
    # otherwise composes where the tuned module ran and keeps the fused graph
    # everywhere else.
    promoted = all(
        run["default"]["label"] == expected_default_pair(run, probe_key, structural_default)
        for run in runs
    )
    rows = []
    for run in runs:
        row = {
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
            "recorded_route": recorded_route(run),
            "recorded_compose": None
            if recorded_route(run) is None
            else recorded_route(run) == "staged",
            "expected_default_label": expected_default_pair(run, probe_key, structural_default),
            f"fused_{probe_key}_dispatches": run["fused"][f"{probe_key}_dispatches"],
            f"composed_{probe_key}_dispatches": run["composed"][f"{probe_key}_dispatches"],
            f"default_{probe_key}_dispatches": run["default"][f"{probe_key}_dispatches"],
            "tensors": run["tensors"],
        }
        if not structural_default:
            row[f"composed_ran_vendor_{probe_key}"] = (
                run["composed"][f"{probe_key}_dispatches"] > 0
            )
        if steps is not None:
            for leg in ("fused", "composed", "default"):
                row[f"{leg}_ms_per_step"] = run[leg]["ms_per_step"]
        rows.append(row)
    summary = {
        "result": "pass",
        "device": runs[0]["device"],
        "width": runs[0]["width"],
        variant_key: runs[0].get(variant_key, ""),
        "promotion": "geometry-aware" if promoted else "UNEXPECTED",
        "batches": rows,
        "median_composed_over_fused": median(ratios),
        "winner": "composed" if composed_wins else ("fused" if fused_wins else "mixed"),
    }
    if steps is not None:
        summary["steps"] = steps
        summary["head_size"] = runs[0]["head_size"]
        summary["state_size"] = runs[0]["state_size"]
    return summary


def expected_default_pair(run: dict, probe_key: str, structural_default: bool = False) -> str:
    """The graph the default must build for one measured geometry.

    A recorded route wins where the adapter has one (the record *is* the
    measurement), and the route decides the leg: `staged` is the composed leg,
    either fused route is the fused leg with that arm's label. Otherwise
    `structural_default` producers keep their shipped graph, which is the fused
    leg; the vendor-gated producers compose where the tuned module is admitted, and
    the composed leg's dispatch count is the proof of what the gate did.
    """
    recorded = recorded_route(run)
    if recorded == "staged":
        return "composed"
    if recorded == "narrow-fused" and "narrow" in run:
        return run["narrow"]["label"]
    if recorded is not None:
        return run["fused"]["label"]
    if structural_default:
        return run["fused"]["label"]
    if run["composed"][f"{probe_key}_dispatches"] > 0:
        return "composed"
    return run["fused"]["label"]


def require_geometry_aware(summary: dict) -> None:
    """Fail closed when the default stopped following the decision it should follow."""
    if summary["promotion"] == "geometry-aware":
        return
    drifted = [
        f"batch {row['batch']}: default built {row['default_label']}, expected "
        f"{row.get('expected_default_label', '?')}"
        f" (recorded {row.get('recorded_compose')})"
        for row in summary["batches"]
        if row["default_label"] != row.get("expected_default_label")
    ]
    raise bridge.Mismatch(
        "the default graph stopped following the decision record (and the shipped default where "
        "the record has no entry): " + "; ".join(drifted)
    )


def require_tensors(row: dict, expected: set[str]) -> None:
    labels = {entry["label"] for entry in row["tensors"]}
    if labels != expected:
        raise bridge.Mismatch(
            f"batch {row['batch']}: compared tensors {sorted(labels)} do not match {sorted(expected)}"
        )


def record_decisions(
    path: Path,
    runs: list[dict],
    *,
    kind: str,
    probe_key: str,
    notes: str = DEFAULT_RECORD_NOTES,
    measured_by: str = "validation/bench_rwkv_packed_cell_*_bench.py",
    margin: float = 0.0,
    probe_label: str | None = None,
    parameter_storage: str | None = None,
    candidates=None,
) -> tuple[dict, list[int]]:
    """Merge this run's per-geometry winners into a decision record.

    Entries for every other adapter, and for this adapter's other widths, are
    preserved verbatim, so recording on a second GPU never loses the first one's
    crossover. The record carries every candidate's measured milliseconds next to
    the route it chose: a hand-edited entry then has to contradict its own
    measurement, and the embedded-table validation rejects it at build time.

    A geometry whose winner sits inside `margin` of the structural default is
    **omitted**, and any entry recorded for it earlier is dropped, because a sign
    that flips between runs is not a decision: that geometry falls back to the
    default. Returns the record and the omitted batches so the driver can report
    them.

    `candidates` maps one run to `{route label: median milliseconds}`; the default
    is the two-way `{"fused": ..., "staged": ...}` set. A three-way producer (the
    key/state stage) passes all three labels, and the winner is the fastest of
    them, so the record names the route and the arm in one field instead of a
    compose flag plus an arm.

    `parameter_storage` writes the entries under one storage policy. Entries are
    replaced and dropped per policy, so recording the FP16-storage arm never
    deletes or rewrites the policy-less (FP32) verdict for the same geometry, and
    a policy-less recording never touches a scoped entry. That separation is what
    lets the FP16 route take a staged win at a geometry where the FP32 arm keeps
    the fusion. The policy name is canonicalized to the storage it selects, so
    `fp16-packed-mirror` and `fp16-storage-fp32-compute` write the same scope.
    """
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("kind") not in (None, kind):
            raise bridge.Mismatch(
                f"{path} holds a {existing.get('kind')!r} record, not a {kind!r} one"
            )
        if existing.get("schema", 2) != 2:
            raise bridge.Mismatch(
                f"{path} is schema {existing.get('schema')}; the route record is schema 2"
            )
    device = runs[0]["device"]
    width = runs[0]["width"]
    measured = {run["batch"] for run in runs}
    policy = canonical_policy(parameter_storage)
    baseline = STRUCTURAL_DEFAULT_ROUTE.get(probe_label or "", "fused")
    kept = [
        entry
        for entry in existing.get("decisions", [])
        if not (
            device_key(entry["device"]) == device_key(device)
            and entry["width"] == width
            and entry["batch"] in measured
            and canonical_policy(entry.get("parameter_storage")) == policy
        )
    ]
    decided, omitted = [], []
    for run in runs:
        if candidates is None:
            timings = {
                "fused": run["fused"]["timing"]["median_ms"],
                "staged": run["composed"]["timing"]["median_ms"],
            }
        else:
            timings = candidates(run)
        unknown = set(timings) - set(ROUTE_ORDER)
        if unknown:
            raise bridge.Mismatch(
                f"batch {run['batch']}: candidate routes {sorted(unknown)} are not in the "
                f"record's vocabulary {ROUTE_ORDER}"
            )
        if baseline not in timings:
            raise bridge.Mismatch(
                f"batch {run['batch']}: no timing for the structural default route {baseline!r}"
            )
        best = min(timings, key=lambda label: (timings[label], ROUTE_ORDER.index(label)))
        # The default route winning is not a tie: recording it is how the table
        # documents that the geometry was measured and the shipped graph held. A
        # different winner is a decision only outside the noise band; inside it the
        # sign is not stable, so the geometry keeps the structural default.
        if best != baseline and abs(timings[best] / timings[baseline] - 1.0) <= margin:
            omitted.append(run["batch"])
            continue
        decided.append((run, best, timings))
    for run, best, timings in decided:
        entry = {
            "device": device,
            "width": width,
            "batch": run["batch"],
            "route": best,
            "candidates": {label: round(ms, 6) for label, ms in timings.items()},
            "probe": probe_label or f"{probe_key}-forward",
            "source": "measured",
        }
        if policy is not None:
            entry["parameter_storage"] = policy
        # A record can now hold entries from more than one measurement driver
        # (the dense route and the FP16-storage route), so each entry names the
        # driver that produced it rather than inheriting the file-level claim.
        entry["measured_by"] = measured_by
        kept.append(entry)
    kept.sort(
        key=lambda entry: (
            device_key(entry["device"]),
            entry["width"],
            entry["batch"],
            entry.get("parameter_storage") or "",
        )
    )
    record = {
        "schema": 2,
        "kind": kind,
        "measured_by": existing.get("measured_by", measured_by),
        "margin": margin,
        "notes": existing.get("notes", notes),
        "decisions": kept,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record, omitted
