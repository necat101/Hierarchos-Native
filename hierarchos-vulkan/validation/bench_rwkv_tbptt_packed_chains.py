#!/usr/bin/env python3
"""Time and record the two packed TBPTT chains against their legacy graphs.

The TBPTT scheduler has two opt-in packed chains, and each one is a route:
the first forward transition can keep recurrent matrix state packed and
suppress backward-only tape stores, and reverse rematerialization can rebuild
only the vector caches and consume packed matrix history directly. Both are
bit-identical to their legacy graphs, so which one wins is a per-geometry
performance question.

This driver measures one chain per invocation, one process per geometry, and
writes the receipt the guard reads plus (with `--record-decisions`) the route
record `RwkvTbpttSequenceOp` consults before recording a sequence. The binary
pins the two forced arms, so the record cannot override the A/B, and it measures
the shipped default as a third leg to prove the default followed the recorded
route (or the structural default where the adapter has no entry).

`--parameter-storage fp16-storage` runs the same A/B with the cell's packed FP16
execution mirrors installed -- the op-level analogue of the training graph's
`fp16-storage-fp32-compute` policy -- and writes that policy's own receipt. Its
verdicts are recorded under the policy scope, so the FP16 route can be adopted
from its own measurement without moving the FP32 verdicts.

`--chain backward` measures the reverse pass's state-backward fusion depth on
the legacy rematerialization graph: the `fused` arm keeps the autotuner's shipped
state-backward schedule, while the `staged` arm swaps only that depth for the most
decomposed executable chain -- the recurrence/add3 kernel, the key transform and
the key-parameter reduction as separate dispatches. The schedule's projection
order and low-rank fan-in are held fixed, so the delta is the fusion depth alone,
and both arms keep the structural first-forward (packed) and
reverse-rematerialization (legacy) chains. Which topology each arm actually
dispatched is written into the receipt, so the comparison names the graphs it
measured instead of assuming them. The receipt rejects the run
unless the two topologies are bit-identical, which is the premise the route
record rests on.

The same sweep times the fully folded rung as a third leg: both the key
transform and the key-parameter reduction folded into the state-backward
kernel, so all three fusion depths are measurable at every batch. It is a
routing candidate, and the gate it has to clear is the fold's own output. On
this adapter its contract associates one FP32 product differently from the
two-dispatch chain and moves `grad_x` by a couple of ulps, so the returned
tensors are held to a ulp allowance rather than exact equality; `grad_k_k` and
`grad_k_a` -- the two tensors the cross-batch reduction actually publishes, and
the ones that could silently lose a batch row -- are not.

That readback is the acceptance gate for the route. `grad_x` never consumes
`grad_k_k`/`grad_k_a`, so the binary reads those gradients back after the
sequence and requires the folded reduction to equal the two-dispatch chain
exactly, recording the staged arm's magnitude beside the zero so the agreement
cannot be zero-vs-zero. Only a geometry that clears it records `folded` as a
candidate; a fold that moves them fails the sweep instead of losing a few
percent. An AdamW step on each leg follows as a second gate; it is the weaker
one, because AdamW's first update is sign-normalized and can hide a gradient
that lost a batch row -- which is why it is not the gate the route rests on.

`--profile-folded` answers what the fold costs instead of what it moves. It
times four legs in one process -- the staged chain, the autotuner's `fused`
depth, the folded rung, and a folded rung whose reduction is suppressed by the
same phantom-arrival seed the control uses -- so the split between the arrival
counters and the batch-order reduction is a measurement instead of an
inference. It writes no receipt and no decision: every leg is gated on the fold's
`grad_k_k`/`grad_k_a` readback first, and each delta is reported beside the
widest leg's interquartile range, so a component smaller than the run's own
resolution is labelled noise rather than signed cost.

`--negative-control` proves those gates reject a broken fold on both of its
publication paths. Above batch 1 it seeds one phantom arrival per batch row, so a
head's counter never completes and the fold never reduces; at batch 1 the rung
writes the gradients directly, so it redirects those writes into a scratch buffer.
Each run builds a clean pair and a corrupted pair in one process and requires the
clean pair to pass both gates and the corrupted pair to fail both. This side
applies its own guard functions to those measurements, so the artifact under test
is the gate `measure` uses on every receipt row.

The records start empty: an adapter with no entry keeps the structural default
(the first forward packed, rematerialization legacy, the autotuner's backward
schedule). A geometry inside the 2% band carries no entry, so the default
decides it until a quiet run measures a sign outside the band.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py \\
        --chain forward --record hierarchos-vulkan/validation/tbptt_packed_forward_bench.json
    python hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py \\
        --chain remat --record hierarchos-vulkan/validation/tbptt_packed_remat_bench.json \\
        --record-decisions
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_harness as ab  # noqa: E402  (shared measurement + record merge)
import verify_rwkv_cpu_bridge as bridge  # noqa: E402  (fixture constants)
import verify_rwkv_packed_cell_composition as composition  # noqa: E402  (live adapter fixture)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
BINARY = "hierarchos-vulkan-rwkv-tbptt-packed-chains-bench"
MEASURED_BY = "validation/bench_rwkv_tbptt_packed_chains.py"
MARGIN = 0.02
# The folded rung's returned tensors. On the Intel Gen9 target its machine code
# contracts one state-kernel product differently from the two-dispatch chain,
# which moves `grad_x` by one or two FP32 ulps; the routing pair itself must
# still agree exactly, and this allowance bounds only the folded leg.
FOLDED_ATOL = 1.0e-6
# One AdamW step on each leg reaches the parameter gradients, which the returned
# tensors cannot: `grad_x` never consumes `grad_k_k`/`grad_k_a`. The first step's
# update magnitude is `lr = 1e-4`, so this absolute window admits a contracted
# FP32 product and rejects a fold that never publishes a parameter gradient.
FOLDED_PARAMETER_ATOL = 1.0e-6
# The fold's own output gets no allowance at all: the fused kernel adds the batch
# rows in the shipped reduction's ascending order, so `grad_k_k`/`grad_k_a` are
# the two-dispatch chain's bit for bit. This is the only tensor-level check of the
# fold itself; the post-step parameters see an AdamW-normalized update, which can
# hide a gradient that lost a batch row.
FOLDED_KEY_PARAMETER_ATOL = 0.0
# The binary's `--negative-control` breaks one of the fold's two publication
# paths: at batch 1 the rung writes the parameter gradients directly (no arrival
# count exists), and above it the reduce fires only when a head's arrival count
# completes. The default sweep runs both regimes.
NEGATIVE_CONTROL_BATCHES = (1, 4)
CORRUPTIONS = {
    1: "batch-1-direct-write-sink",
    4: "cross-batch-arrival-count",
}


def expected_corruption(batch: int) -> str:
    """Which publication path a batch's folded rung uses, and so which one is broken."""
    return "batch-1-direct-write-sink" if batch == 1 else "cross-batch-arrival-count"
TENSORS = {
    "outputs",
    "final_packed_state",
    "grad_x",
    "token_feature_grad",
    "grad_initial_packed_state",
}

CHAINS = {
    "forward": {
        "kind": "rwkv-tbptt-packed-forward",
        "probe": "tbptt-packed-forward",
        "structural_default": "packed",
        "receipts": {
            "fp32": VALIDATION / "tbptt_packed_forward_bench.json",
            "fp16-storage": VALIDATION / "tbptt_packed_forward_fp16_bench.json",
        },
        "record": VALIDATION / "tbptt_packed_forward_decisions.json",
        "arms": ("legacy", "packed"),
        "ratio": "packed_over_legacy",
        "baseline": "legacy",
        # The FP16 refusal this table used to carry was lifted by measuring: the
        # fp16-storage A/B is bit-identical and its own receipt records the same
        # crossover, so the policy adopts a verdict at every measured geometry.
        # A policy whose trajectory genuinely pins the structural graph keeps
        # the generic path: list it here with its reason and the driver will
        # withhold every measured move it has not measured itself.
        "policy_refusals": {},
    },
    "remat": {
        "kind": "rwkv-tbptt-packed-rematerialization",
        "probe": "tbptt-packed-rematerialization",
        "structural_default": "legacy",
        "receipts": {
            "fp32": VALIDATION / "tbptt_packed_remat_bench.json",
            "fp16-storage": VALIDATION / "tbptt_packed_remat_fp16_bench.json",
        },
        "record": VALIDATION / "tbptt_packed_rematerialization_decisions.json",
        "arms": ("legacy", "packed"),
        "ratio": "packed_over_legacy",
        "baseline": "legacy",
        # Every measured rematerialization verdict is the structural default, and
        # the fp16-storage A/B agrees, so no policy refuses anything today. The
        # table still goes through the same regeneration, so a measured `packed`
        # win under a policy whose frozen trajectory uses the structural graph
        # would be withheld automatically.
        "policy_refusals": {},
    },
    "backward": {
        "kind": "rwkv-tbptt-backward-fusion",
        "probe": "tbptt-backward-fusion",
        "structural_default": "fused",
        "receipts": {
            "fp32": VALIDATION / "tbptt_backward_fusion_bench.json",
            "fp16-storage": VALIDATION / "tbptt_backward_fusion_fp16_bench.json",
        },
        "record": VALIDATION / "tbptt_backward_fusion_decisions.json",
        "arms": ("staged", "fused"),
        "ratio": "fused_over_staged",
        "baseline": "staged",
        # The shipped default is the autotuner's backward schedule, so with no
        # recorded verdict the default leg can be either measured arm; the
        # receipt records which one it built and the guard accepts that choice.
        "default_from_measurement": True,
        "policy_refusals": {},
    },
}

RECORD_NOTES = {
    "forward": (
        "One entry per (adapter, width, batch, parameter-storage policy) the TBPTT "
        "first-forward crossover was measured at. `route` is `packed` (the packed "
        "matrix-state transition with backward-only tape stores suppressed) or `legacy` "
        "(the unpacked transition followed by a pack copy). Every candidate's measured "
        "median milliseconds are recorded beside the route. The structural default "
        "winning keeps an entry that documents the measurement without moving the route; "
        "a non-default winner inside the 2% noise band carries no entry, because the run "
        "withdraws it and `packed` decides that geometry until a quiet run measures a "
        "sign outside the band. `RwkvTbpttSequenceOp` reads this table per batch under the active "
        "parameter-storage policy, unless a benchmark arm pinned the switch, so a "
        "recorded `legacy` verdict withdraws the packed transition at that geometry. A "
        "policy-less entry is the default (FP32) measurement; an entry with "
        "`parameter_storage` is that policy's own chain A/B, so the FP16 policy can "
        "follow its own receipt without moving the FP32 route. A policy that has not "
        "measured the route keeps the structural packed default; that is expressed as "
        "a `withheld` entry naming the shared verdict it may not adopt, never as a "
        "rewrite of the measurement."
    ),
    "remat": (
        "One entry per (adapter, width, batch, parameter-storage policy) the TBPTT "
        "reverse-rematerialization crossover was measured at. `route` is `packed` (the "
        "reverse pass restores only vector caches and feeds packed matrix history "
        "directly into the fused state-backward kernel) or `legacy` (the reverse pass "
        "unpacks the full matrix tape before replaying the cell backward). Every "
        "candidate's measured median milliseconds are recorded beside the route. The "
        "structural default winning keeps an entry that documents the measurement "
        "without moving the route; a non-default winner inside the 2% noise band carries "
        "no entry, because the run withdraws it and `legacy` decides that geometry until "
        "a quiet run measures a sign outside the band. `RwkvTbpttSequenceOp` reads this table per "
        "batch under the active parameter-storage policy, unless a benchmark arm pinned "
        "the switch, so a recorded `packed` verdict promotes rematerialization only "
        "where it was measured. Packed loses under FP16 storage too, and every measured "
        "verdict is the structural default, so that policy needs no scoped entry. A "
        "policy that cannot adopt a measured move (its trajectory golden is pinned to "
        "the structural graph) records a `withheld` entry naming that measured route."
    ),
    "backward": (
        "One entry per (adapter, width, batch, parameter-storage policy) the TBPTT "
        "reverse-pass state-backward fusion depth was measured at. `route` is `fused` "
        "(the autotuner's shipped schedule: recurrence backward, vector add3 and the key "
        "transform in one kernel, with the key-parameter reduction separate on this "
        "adapter), `staged` (the most decomposed executable chain: the recurrence/add3 "
        "kernel, the key transform and the reduction each dispatched separately), or "
        "`folded` (both stages folded into the state-backward kernel, whose cross-batch "
        "key-parameter reduction is dispatched once per head), with the schedule's "
        "projection order and low-rank fan-in held fixed so the candidates differ only "
        "in fusion depth. The two-dispatch pair is bit-identical by construction and the "
        "receipt refuses to record unless it is. The folded rung is a candidate only "
        "where the receipt measured its `grad_k_k`/`grad_k_a` bit-identical to the "
        "two-dispatch reduction with a non-zero staged gradient -- the fold's own output, "
        "which no returned tensor consumes -- and its returned tensors then stay within "
        "a few contracted FP32 ulps on `grad_x`; `folded` is absent from an entry's "
        "candidates where that gate did not hold. Every candidate's measured median "
        "milliseconds are recorded beside the route. The structural default (`fused`) "
        "winning keeps an entry that documents the measurement without moving the route; "
        "a non-default winner inside the 2% noise band carries no entry, because the run "
        "withdraws it and the autotuner's schedule decides that geometry until a quiet "
        "run measures a sign outside the band. `RwkvTbpttSequenceOp` reads this table per "
        "batch under the active parameter-storage policy, unless a benchmark arm pinned "
        "the fusion depth, so a recorded `staged` verdict withdraws the fused backward "
        "and a recorded `folded` verdict pins the fully folded rung, failing closed where "
        "the device does not compile it. A policy-less entry is the default (FP32) "
        "measurement; an entry with `parameter_storage` is that policy's own A/B."
    ),
}


def refusals_by_policy(
    record: dict, device: str, width: int
) -> list[tuple[str, list[int]]]:
    """The refusals this run's adapter and width carry, per policy, for the report."""
    by_policy: dict[str, list[int]] = {}
    for entry in record.get("withheld", []):
        if entry["device"].strip().lower() != device.strip().lower():
            continue
        if entry["width"] != width:
            continue
        scope = entry.get("parameter_storage")
        if scope is None:
            continue
        by_policy.setdefault(scope, []).append(entry["batch"])
    return [(policy, sorted(batches)) for policy, batches in sorted(by_policy.items())]


def run_batch(
    binary: Path,
    model_dir: Path,
    chain: str,
    batch: int,
    *,
    steps: int,
    iterations: int,
    warmup: int,
    parameter_storage: str,
) -> dict:
    return ab.run_json(
        [
            str(binary),
            "--chain",
            chain,
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
            "--parameter-storage",
            parameter_storage,
        ],
        batch=batch,
        label=f"TBPTT packed {chain} benchmark",
    )


def require_folded_tensor_parity(batch: int, run: dict) -> None:
    """The folded rung's returned tensors, against the staged chain."""
    if "folded_max_abs" not in run:
        return
    if run["folded_max_abs"] > FOLDED_ATOL:
        raise bridge.Mismatch(
            f"batch {batch}: the folded fusion rung diverges from the staged chain by "
            f"{run['folded_max_abs']:.3e} on {run.get('folded_schedule')}, beyond the "
            f"{FOLDED_ATOL:.1e} ulp allowance; the ladder cannot be recorded"
        )


def require_folded_key_parameter_parity(batch: int, run: dict) -> None:
    """The fold's own reduction output, against the two-dispatch reduce."""
    if "staged_key_parameter_max_abs" in run and run["staged_key_parameter_max_abs"] <= 0.0:
        raise bridge.Mismatch(
            f"batch {batch}: the staged arm's grad_k_k/grad_k_a are all zero, so the folded "
            "rung's agreement with them would prove nothing"
        )
    if "folded_key_parameter_max_abs" not in run:
        return
    delta = run["folded_key_parameter_max_abs"]
    if delta > FOLDED_KEY_PARAMETER_ATOL:
        raise bridge.Mismatch(
            f"batch {batch}: the folded rung moved grad_k_k/grad_k_a by {delta:.3e} away "
            "from the two-dispatch chain; it must publish the gradients that reduction "
            "produces, so no allowance applies"
        )


def require_folded_parameter_parity(batch: int, run: dict) -> None:
    """One identical AdamW step on both legs, compared through the parameters."""
    if "folded_parameter_delta" not in run:
        return
    delta = run["folded_parameter_delta"]
    if delta > FOLDED_PARAMETER_ATOL:
        raise bridge.Mismatch(
            f"batch {batch}: one AdamW step on the folded rung moved a parameter by "
            f"{delta:.3e}, beyond the {FOLDED_PARAMETER_ATOL:.1e} allowance; the "
            "cross-batch fold does not reproduce the shipped reduction"
        )


def folded_candidate(batch: int, run: dict) -> bool:
    """Whether this geometry's folded rung may be a routing candidate.

    The acceptance gate is the fold's own output: `grad_k_k`/`grad_k_a` exactly
    equal to the two-dispatch reduction, with the staged arm's gradient non-zero so
    the agreement is not zero-vs-zero. That is what a fold that loses a batch row
    or never publishes fails; it is also the only thing the folded kernel computes
    that no returned tensor consumes, so no other comparison can stand in for it.

    The remaining conditions are the bounds the returned tensors and one AdamW step
    are held to. They are recorded per batch, and a row that fails any of them is
    refused here rather than mined for a route: `folded` is a candidate where the
    fold was measured exact, and nowhere else.
    """
    # A binary receipt names the leg `folded`; a merged receipt row names its
    # timing `folded_ms`. Either shape means the rung was measured here.
    if "folded" not in run and "folded_ms" not in run:
        # The device compiles no fully folded rung at this geometry, so there is no
        # candidate to record and the two-dispatch pair decides.
        return False
    delta = run.get("folded_key_parameter_max_abs")
    magnitude = run.get("staged_key_parameter_max_abs")
    if delta is None or magnitude is None:
        raise bridge.Mismatch(
            f"batch {batch}: the folded rung was timed but its key-parameter readback "
            "was not recorded, so the route has no acceptance gate"
        )
    if magnitude <= 0.0:
        raise bridge.Mismatch(
            f"batch {batch}: the staged arm's grad_k_k/grad_k_a are all zero, so the "
            "folded rung's agreement with them would prove nothing"
        )
    require_folded_key_parameter_parity(batch, run)
    require_folded_tensor_parity(batch, run)
    require_folded_parameter_parity(batch, run)
    return True


def no_decomposed_graph(stderr: str) -> bool:
    """Whether the binary reported that this geometry has one backward graph.

    A device that compiles no more decomposed state-backward topology than the
    one it ships has nothing to A/B at that batch. That is a fact about the
    kernel set, not a measurement failure, so the driver records it and moves
    on; every other error still fails the sweep.
    """
    return "has no staged graph to be measured against" in stderr or (
        "compiles only one backward topology" in stderr
    )


def measure(
    binary: Path,
    model_dir: Path,
    chain: str,
    batches: list[int],
    *,
    steps: int,
    iterations: int,
    warmup: int,
    parameter_storage: str,
) -> tuple[list[dict], list[int]]:
    runs = []
    unmeasured: list[int] = []
    for batch in batches:
        try:
            run = run_batch(
                binary,
                model_dir,
                chain,
                batch,
                steps=steps,
                iterations=iterations,
                warmup=warmup,
                parameter_storage=parameter_storage,
            )
        except bridge.Mismatch as exc:
            if no_decomposed_graph(str(exc)):
                unmeasured.append(batch)
                print(f"batch {batch:>2}: no decomposed backward graph to measure; skipped")
                continue
            raise
        labels = {tensor["label"] for tensor in run["tensors"]}
        if labels != TENSORS:
            raise bridge.Mismatch(
                f"batch {batch}: compared tensors {sorted(labels)} do not match "
                f"{sorted(TENSORS)}"
            )
        if run["worst_max_abs"] != 0.0:
            raise bridge.Mismatch(
                f"batch {batch}: the {chain} arms diverge by {run['worst_max_abs']:.3e}; "
                "a route may only choose between bit-identical graphs"
            )
        spec = CHAINS[chain]
        staged_arm, packed_arm = spec["arms"]
        staged = run[staged_arm]
        packed = run[packed_arm]
        # The folded rung is a candidate only where this returns true, and it
        # raises instead of returning false when the fold was measured but failed
        # its gate: a broken fold is a sweep failure, not a slower candidate.
        recordable_folded = folded_candidate(batch, run)
        row = {
            "batch": batch,
            f"{staged_arm}_ms": staged["median_ms"],
            f"{packed_arm}_ms": packed["median_ms"],
            "default_ms": run["default"]["median_ms"],
            spec["ratio"]: run[spec["ratio"]],
            f"default_over_{spec['baseline']}": run["default"]["median_ms"]
            / run[spec["baseline"]]["median_ms"],
            f"{staged_arm}_label": staged["label"],
            f"{packed_arm}_label": packed["label"],
            "default_label": run["default_route"],
            "recorded_route": run["recorded_route"],
            "expected_default_label": run["recorded_route"]
            or (
                run["default_route"]
                if CHAINS[chain].get("default_from_measurement")
                else CHAINS[chain]["structural_default"]
            ),
            "worst_max_abs": run["worst_max_abs"],
            "tensors": run["tensors"],
            "parameter_storage": run["parameter_storage"],
            "precision_policy": run["precision_policy"],
        }
        for field in (
            "staged_schedule",
            "fused_schedule",
            "folded_schedule",
            "default_schedule",
            "available_schedules",
        ):
            if field in run:
                row[field] = run[field]
        if "folded" in run:
            row["folded_label"] = run["folded_label"]
            row["folded_ms"] = run["folded"]["median_ms"]
            row["folded_over_staged"] = run["folded_over_staged"]
            row["folded_max_abs"] = run["folded_max_abs"]
            row["folded_key_parameter_max_abs"] = run["folded_key_parameter_max_abs"]
            row["staged_key_parameter_max_abs"] = run["staged_key_parameter_max_abs"]
            row["folded_parameter_delta"] = run["folded_parameter_delta"]
            row["folded_candidate"] = recordable_folded
        # A geometry where both arms dispatch the same topology is one graph
        # measured twice. The A/B refuses to call that a comparison: it is
        # neither a win nor a tie, and a receipt row for it would be evidence of
        # nothing.
        if row.get("staged_schedule") is not None and row["staged_schedule"] == row.get(
            "fused_schedule"
        ):
            raise bridge.Mismatch(
                f"batch {batch}: the {chain} arms dispatched the same schedule "
                f"({row['staged_schedule']}); there is no second graph to measure"
            )
        if row["default_label"] != row["expected_default_label"]:
            raise bridge.Mismatch(
                f"batch {batch}: the default built {row['default_label']}, expected "
                f"{row['expected_default_label']}"
            )
        run.update(row)
        runs.append(run)
        folded = (
            f"  folded {row['folded_ms']:8.3f} ms (delta {row['folded_max_abs']:.1e}, "
            f"k-param {row['folded_key_parameter_max_abs']:.1e}, "
            f"step {row['folded_parameter_delta']:.1e})"
            if "folded_ms" in row
            else ""
        )
        print(
            f"batch {batch:>2}: {staged_arm} {row[f'{staged_arm}_ms']:8.3f} ms  "
            f"{packed_arm} {row[f'{packed_arm}_ms']:8.3f} ms  "
            f"ratio {row[spec['ratio']]:.4f}  default {row['default_label']}{folded}"
        )
    if not runs:
        raise bridge.Mismatch(
            f"the {chain} sweep measured no geometry: this adapter compiles one backward "
            "topology at every batch, so there is no fusion A/B to record"
        )
    return runs, unmeasured


def run_negative_control(
    binary: Path,
    model_dir: Path,
    batch: int,
    *,
    steps: int,
    iterations: int,
    warmup: int,
    parameter_storage: str,
) -> dict:
    return ab.run_json(
        [
            str(binary),
            "--chain",
            "backward",
            "--negative-control",
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
            "--parameter-storage",
            parameter_storage,
        ],
        batch=batch,
        label="TBPTT folded arrival-count negative control",
    )


def require_guard_rejects(gate, batch: int, row: dict, what: str) -> None:
    """A guard that accepts the corrupted row is the failure this control exists for."""
    try:
        gate(batch, row)
    except bridge.Mismatch:
        return
    raise bridge.Mismatch(
        f"the receipt's {what} guard accepted the corrupted arrival count at batch "
        f"{batch}: it would have recorded the run ({row})"
    )


def measure_negative_control(
    binary: Path,
    model_dir: Path,
    batches: list[int],
    *,
    steps: int,
    iterations: int,
    warmup: int,
    parameter_storage: str,
) -> list[dict]:
    """Run the binary's corrupted-arrival control and apply the receipt's own gates.

    The binary measures a clean pair and a corrupted pair in one process. This
    side does not take its verdict on trust: it feeds both pairs through the two
    gates `measure` applies to a receipt row, requires the clean one to pass, and
    requires the corrupted one to be rejected by both.
    """
    runs = []
    for batch in batches:
        run = run_negative_control(
            binary,
            model_dir,
            batch,
            steps=steps,
            iterations=iterations,
            warmup=warmup,
            parameter_storage=parameter_storage,
        )
        if run.get("result") != "pass":
            raise bridge.Mismatch(
                f"batch {batch}: the negative control reported {run.get('result')!r}"
            )
        if run["corruption"] != expected_corruption(batch):
            raise bridge.Mismatch(
                f"batch {batch}: the control broke {run['corruption']!r}, but this "
                f"geometry publishes through {expected_corruption(batch)!r}"
            )
        if batch == 1:
            if run["arrival_seed"] is not None:
                raise bridge.Mismatch(
                    "batch 1: the folded rung has no arrival count, so the control "
                    f"cannot have seeded {run['arrival_seed']} phantom arrivals"
                )
        elif run["arrival_seed"] != batch:
            raise bridge.Mismatch(
                f"batch {batch}: the control seeded {run['arrival_seed']} phantom "
                f"arrivals, not {batch}, so a real arrival could still complete the head"
            )
        clean = {
            "folded_key_parameter_max_abs": run["clean_key_parameter_max_abs"],
            "folded_parameter_delta": run["clean_parameter_delta"],
        }
        corrupted = {
            "folded_key_parameter_max_abs": run["corrupted_key_parameter_max_abs"],
            "folded_parameter_delta": run["corrupted_parameter_delta"],
        }
        # The clean pair is the control for the control: if it failed a gate the
        # rejection could not be attributed to the seeded arrivals.
        require_folded_key_parameter_parity(batch, clean)
        require_folded_parameter_parity(batch, clean)
        require_guard_rejects(require_folded_key_parameter_parity, batch, corrupted, "key-parameter")
        require_guard_rejects(require_folded_parameter_parity, batch, corrupted, "parameter-step")
        print(
            f"batch {batch:>2}: clean key-param {run['clean_key_parameter_max_abs']:.1e} "
            f"step {run['clean_parameter_delta']:.1e}; corrupted key-param "
            f"{run['corrupted_key_parameter_max_abs']:.1e} step "
            f"{run['corrupted_parameter_delta']:.1e}; both guards reject"
        )
        runs.append(run)
    return runs


def run_profile_folded_mode(args: argparse.Namespace) -> int:
    """Time the folded rung with and without the reduction it publishes through.

    This is a measurement, not a recording: it writes no receipt row and no
    decision, and the binary gates both legs before timing them (the clean fold
    must publish the two-dispatch chain's gradients exactly, the phantom-arrival
    leg must not publish them at all). The split it reports is what answers
    whether the fold's cost is its counters or its batch-order reduction.
    """
    batches = args.batches or list(NEGATIVE_CONTROL_BATCHES)
    if any(batch <= 0 for batch in batches):
        raise bridge.Mismatch("--profile-folded needs positive batches")
    binary = ab.build_release_binary(BINARY)
    profiles = []
    with tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-tbptt-profile-") as root:
        model_dir = Path(root) / "model"
        composition.write_package(model_dir)
        for batch in batches:
            profiles.append(
                ab.run_json(
                    [
                        str(binary),
                        "--chain",
                        "backward",
                        "--profile-folded",
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
                        str(args.steps),
                        "--iterations",
                        str(args.iterations),
                        "--warmup",
                        str(args.warmup),
                        "--parameter-storage",
                        args.parameter_storage,
                    ],
                    batch=batch,
                    label="TBPTT folded publication profile",
                )
            )
            profile = profiles[-1]
            legs = {leg["label"]: leg["median_ms"] for leg in profile["legs"]}
            split = ""
            if "folded-without-reduction" in legs:
                tail = profile["reduction_tail_beyond_resolution"]
                counters = profile["counters_beyond_resolution"]
                split = (
                    f"  tail {profile['folded_minus_without_reduction_ms']:+.3f} ms"
                    f" ({'signal' if tail else 'noise'})"
                    f"  counters-vs-dispatch {profile['without_reduction_minus_fused_ms']:+.3f} ms"
                    f" ({'signal' if counters else 'noise'})"
                )
            print(
                f"batch {batch:>2}: staged {legs['staged']:8.3f} ms  "
                f"fused {legs['fused']:8.3f} ms  folded {legs['folded']:8.3f} ms  "
                f"folded/fused {profile['folded_over_fused']:.4f}  "
                f"resolution +/-{profile['resolution_ms']:.3f} ms{split}"
            )
    summary = {
        "result": "pass",
        "profile": "folded-publication",
        "parameter_storage": (
            None if args.parameter_storage == "fp32" else args.parameter_storage
        ),
        "batches": profiles,
    }
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def run_negative_control_mode(args: argparse.Namespace) -> int:
    batches = args.batches or list(NEGATIVE_CONTROL_BATCHES)
    if any(batch <= 0 for batch in batches):
        raise bridge.Mismatch("--negative-control needs positive batches")
    binary = ab.build_release_binary(BINARY)
    with tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-tbptt-control-") as root:
        model_dir = Path(root) / "model"
        composition.write_package(model_dir)
        runs = measure_negative_control(
            binary,
            model_dir,
            batches,
            steps=args.steps,
            iterations=args.iterations,
            warmup=args.warmup,
            parameter_storage=args.parameter_storage,
        )
    summary = {
        "result": "pass",
        "negative_control": "packed-fold-publication",
        "corruptions": [expected_corruption(batch) for batch in batches],
        "guards": ["folded_key_parameter_max_abs", "folded_parameter_delta"],
        "batches": runs,
    }
    # The build's own output shares stdout with the driver, so a programmatic
    # reader (the receipt tests) takes the record file instead of parsing it.
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chain", choices=sorted(CHAINS))
    parser.add_argument(
        "--batch",
        type=int,
        action="append",
        dest="batches",
        help="batch size to time (repeatable; default 1, 4, 8, 16, 32)",
    )
    parser.add_argument("--steps", type=int, default=8, help="TBPTT window length")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument(
        "--parameter-storage",
        choices=("fp32", "fp16-storage"),
        default="fp32",
        help="the parameter-storage policy to time under (default fp32)",
    )
    parser.add_argument(
        "--negative-control",
        action="store_true",
        help="inject a broken cross-batch arrival count and prove both folded guards reject it",
    )
    parser.add_argument(
        "--profile-folded",
        action="store_true",
        help="time the folded rung with and without its reduction, to split the counters from "
        "the batch-order reduction",
    )
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--record-decisions", action="store_true")
    args = parser.parse_args()
    if args.profile_folded and args.negative_control:
        parser.error("--profile-folded and --negative-control are separate measurements")
    if args.profile_folded:
        if args.chain not in (None, "backward"):
            parser.error("--profile-folded measures the folded backward rung only")
        return run_profile_folded_mode(args)
    if args.negative_control:
        if args.chain not in (None, "backward"):
            parser.error("--negative-control measures the folded backward rung only")
        return run_negative_control_mode(args)
    if args.chain is None:
        parser.error("--chain is required")
    batches = args.batches or [1, 4, 8, 16, 32]
    if any(batch <= 0 for batch in batches):
        parser.error("--batch values must be positive")
    if args.steps <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("--steps/--iterations must be positive and --warmup non-negative")
    if args.record is not None and args.record_decisions:
        parser.error(
            "--record and --record-decisions are separate passes: record the decisions, "
            "rebuild so the binary embeds them, then record the receipt"
        )

    spec = CHAINS[args.chain]
    receipt_path = args.record or spec["receipts"][args.parameter_storage]
    record_path = spec["record"]
    binary = ab.build_release_binary(BINARY)
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-tbptt-chains-")
        fixture_root = Path(cleanup.name)
    try:
        model_dir = fixture_root / "model"
        composition.write_package(model_dir)
        runs, unmeasured = measure(
            binary,
            model_dir,
            args.chain,
            batches,
            steps=args.steps,
            iterations=args.iterations,
            warmup=args.warmup,
            parameter_storage=args.parameter_storage,
        )
        if unmeasured:
            print(
                "unmeasured (the shipped backward is already the most decomposed topology "
                "this device compiles): batches "
                + ", ".join(str(batch) for batch in unmeasured)
            )
        if args.record_decisions:
            ab.record_decisions(
                record_path,
                runs,
                kind=spec["kind"],
                probe_key="tbptt",
                notes=RECORD_NOTES[args.chain],
                measured_by=MEASURED_BY,
                margin=args.margin,
                probe_label=spec["probe"],
                parameter_storage=(
                    None if args.parameter_storage == "fp32" else args.parameter_storage
                ),
                candidates=lambda run: {
                    **{
                        label: run[f"{label}_ms"]
                        for label in CHAINS[args.chain]["arms"]
                    },
                    # The deepest rung joins the routing pair only where this
                    # receipt proved its key-parameter reduction exact.
                    **(
                        {"folded": run["folded_ms"]}
                        if run.get("folded_candidate")
                        else {}
                    ),
                },
            )
            merged = ab.refuse_shared_verdicts(
                record_path,
                device=runs[0]["device"],
                width=runs[0]["width"],
                default_route=spec["structural_default"],
                refusals=spec["policy_refusals"],
            )
            for policy, batches in refusals_by_policy(
                merged, runs[0]["device"], runs[0]["width"]
            ):
                print(
                    f"withheld for {policy} (its trajectory pins the structural "
                    f"{spec['structural_default']} route): batches "
                    + ", ".join(str(batch) for batch in batches)
                )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    if args.record is not None:
        summary = {
            "result": "pass",
            "device": runs[0]["device"],
            "width": runs[0]["width"],
            "head_size": runs[0]["head_size"],
            "steps": args.steps,
            "chain": args.chain,
            # The default policy is unscoped; only a scoped run names a policy.
            "parameter_storage": (
                None if args.parameter_storage == "fp32" else args.parameter_storage
            ),
            "precision_policy": runs[0]["precision_policy"],
            "samples": args.iterations,
            "warmup": args.warmup,
            "margin": args.margin,
            "unmeasured": unmeasured,
            "batches": runs,
        }
        receipt_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    ratio_field = CHAINS[args.chain]["ratio"]
    medians = sorted(run[ratio_field] for run in runs)
    middle = len(medians) // 2
    median_ratio = (
        medians[middle]
        if len(medians) % 2
        else (medians[middle - 1] + medians[middle]) * 0.5
    )
    print(
        json.dumps(
            {
                "device": runs[0]["device"],
                "chain": args.chain,
                f"median_{ratio_field}": median_ratio,
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
