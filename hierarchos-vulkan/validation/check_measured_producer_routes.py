#!/usr/bin/env python3
"""Check every measured producer's receipt against the record it was written into.

The producer A/Bs in `FUSED_FORWARD_PRODUCER_AUDIT.md` all end in the same
contract: one driver times a geometry sweep, the default is proven to build the
graph the record names, and the record stores the winning route beside the
candidates that justified it. That contract is what makes the shipped graph
auditable, and it is only worth reading while the receipt and the record still
agree. This guard is the check that they do:

* every geometry of the producer's sweep was actually dispatched -- a verdict
  that quietly stops being measured is not a stale verdict, it is an unsupported
  one;
* every record entry this receipt reads names a geometry this receipt measured,
  so a geometry the sweep dropped cannot keep a decision, and a policy-scoped
  entry names the driver whose measurement it carries (the shared policy-less
  fallback may be the dense driver's by design);
* a recorded route is the fastest candidate in its own entry, and the receipt
  that produced it still measures that route ahead (a divergence inside the
  driver's noise band is a tie and passes; one outside it means the record is
  stale and has to be re-recorded deliberately);
* every measured geometry outside the band carries a decision, because a
  geometry that moved is the one case the table exists for;
* each leg reports the graph it claims, with the dispatch count that proves it,
  and the default leg builds the recorded route -- or the documented fallback
  (the vendor gate, or the structural default) where the receipt's adapter has
  no entry;
* the arms agree exactly where the producer promises bit-identity, and within
  the receipt's stated tolerance where the graphs re-associate.

One spec per producer keeps that contract in one place: the time-mix projection,
the channel mix, the LN1/low-rank dense and FP16-storage routes, and the
key/state fusion. The key/state entry is the three-route case (wide-fused,
narrow-fused, staged), so its checks also cover the narrow arm, the
wide-versus-narrow parity and the arm the default actually dispatched.

The two packed TBPTT chains are their own producers. The first-forward transition
and reverse rematerialization each have a record with `packed`/`legacy` routes
that `RwkvTbpttSequenceOp` reads per batch, so the guard checks them like any
other producer: the full sweep, the bit-identical parity, the route that is the
fastest candidate in its own entry, and the default that followed the record or
the structural default on an unmeasured adapter.

Two receipts consume a producer's record instead of writing one, and they are
checked the same way: the multi-step TBPTT receipt must still follow the time-mix
projection's verdict at every batch it dispatched, and the packed-cell
composition receipt must agree that the route it qualified is the staged one the
time-mix and channel-mix records name. A consumer narrows what it measures (the
TBPTT sweep stops at 16), so a record entry at a batch it did not dispatch is the
producer's evidence, not a stray entry; that distinction is the `consumer` flag,
not an exemption from the record.

The A/B workflow's per-producer steps, receipt-contract test matrix and artifact
paths are rendered from this same table by `validation/render_ab_workflow.py`, so
a producer cannot be added to the guard without being added to the sweep, the
guard step, the test list and the uploaded receipts.

This is the CPU-side guard, so it runs on the committed receipts in the ordinary
workspace check. The A/B jobs run the same command immediately after each
receipt is re-recorded, which is what makes it a check on live measurements
rather than only on the checked-in snapshot.

Usage:
    python hierarchos-vulkan/validation/check_measured_producer_routes.py
    python hierarchos-vulkan/validation/check_measured_producer_routes.py \\
        --producer time-mix-projection --producer key-state-forward
    python hierarchos-vulkan/validation/check_measured_producer_routes.py \\
        --producer low-rank-forward-fp16
    python hierarchos-vulkan/validation/check_measured_producer_routes.py \\
        --producer channel-mix --receipt validation/custom.json \\
        --record validation/custom-decisions.json --sweep 1 --sweep 32
    python hierarchos-vulkan/validation/check_measured_producer_routes.py \\
        --producer tbptt-time-mix --producer packed-cell-composition
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"

PACKED_TENSORS = frozenset(
    {"output", "packed_new_state", "grad_x", "grad_packed_state", "token_feature_grad"}
)
FP16_TENSORS = frozenset(
    {
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
)
KEY_STATE_TENSORS = frozenset({"new_state", "tmix", "scaled_k", "kk", "saved_sa"})
TBPTT_TENSORS = frozenset(
    {
        "outputs",
        "final_packed_state",
        "grad_x",
        "token_feature_grad",
        "grad_initial_packed_state",
    }
)


class Problem(Exception):
    """A check failed; the message is the report line."""


@dataclass(frozen=True)
class Producer:
    """One measured producer: its receipt, its record, and what they must say."""

    name: str
    receipt: Path
    record: Path
    kind: str
    probe: str
    # Batch geometries the sweep is required to dispatch. A recorded geometry is
    # in every spec's sweep: a record entry for a geometry the job never measured
    # is how the key/state route's last collapse went unnoticed.
    sweep: tuple[int, ...]
    candidates: tuple[tuple[str, str], ...]
    route_order: tuple[str, ...]
    default_route: str
    # "vendor": with no entry the vendor gate decides (compose where the tuned
    # module ran); "structural": the shipped graph is the structural default.
    fallback: str
    dispatch_key: str
    fused_dispatches: int
    composed_min_dispatches: int
    tensors: frozenset[str]
    exact_parity: bool
    measured_by: str
    # The band the record was written with. The FP16-storage route records with a
    # 1% band, narrower than the dense route's 2%, because its staged wins are
    # smaller and have never flipped sign.
    band: float | None = None
    # The policy this receipt reads. None means the policy-less entries.
    policy: str | None = None
    # The FP16 route reads its own policy first and falls back to the shared
    # (policy-less) entry, which is what `DecisionTable::route_scoped` does.
    allow_shared: bool = False
    fused_arm: str | None = None
    parameter_storage: str | None = None
    precision_policy: str | None = None
    narrow_arm: str | None = None
    composed_arm: str | None = None
    narrow_dispatches: int | None = None
    composed_must_exceed_fused: bool = False
    composed_exact: int | None = None
    wide_vs_narrow_parity: bool = False
    default_arm: bool = False
    unit_scale: float = 1.0
    unit: str = "ms"
    # A consumer receipt measures a graph that reads a producer's record rather
    # than writing one. It may cover fewer geometries than the record (the record
    # is that producer's full sweep), so an entry at a batch the consumer did not
    # dispatch is not a stray entry, and the consumer cannot demand a new entry
    # for a geometry the producer measured as a tie.
    consumer: bool = False
    # The TBPTT sequence dispatches the composed projection once per timestep, so
    # its vendor count is the receipt's `steps`, not one.
    vendor_dispatches_per_step: bool = False
    # A consumer that runs the shipped default also runs whatever chain routes the
    # default dispatches. Each tuple is (record, kind, structural default, receipt
    # field) so the guard can prove the receipt was measured against the shipped
    # chain route rather than an older build.
    chains: tuple[tuple[Path, str, str, str], ...] = ()
    # Workflow generation: the step label, the driver that records the receipt,
    # the flags after the batch list, and which jobs run it. Keeping these beside
    # the guard contract is what lets the A/B workflow be rendered from one table.
    label: str = ""
    driver: str = ""
    bench_args: tuple[str, ...] = ()
    jobs: tuple[str, ...] = ("producer-ab", "cross-adapter-ab")
    guard_label: str = ""
    step_kind: str = "ab"
    # Workflow rendering also owns the receipt-contract test list and the
    # artifact paths, so a new producer cannot leave either behind.
    test_module: str = ""
    extra_artifacts: tuple[Path, ...] = ()
    extra_test_modules: tuple[str, ...] = ()
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(route for route, _ in self.candidates)

    @property
    def fields(self) -> dict[str, str]:
        return dict(self.candidates)

    @property
    def vendor_flag(self) -> str | None:
        return f"composed_ran_vendor_{self.dispatch_key}" if self.fallback == "vendor" else None


PRODUCERS: tuple[Producer, ...] = (
    Producer(
        name="time-mix-projection",
        label="Time-mix projection",
        driver="hierarchos-vulkan/validation/bench_rwkv_packed_cell_time_mix.py",
        bench_args=("--iterations", "20", "--warmup", "5"),
        guard_label="Time-mix projection route across the sweep",
        receipt=VALIDATION / "packed_cell_time_mix_bench.json",
        record=VALIDATION / "time_mix_projection_decisions.json",
        kind="rwkv-time-mix-projection",
        probe="linear3-forward",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("fused", "fused_ms"), ("staged", "composed_ms")),
        route_order=("fused", "staged"),
        default_route="fused",
        fallback="vendor",
        dispatch_key="linear3",
        fused_dispatches=0,
        composed_min_dispatches=0,
        tensors=PACKED_TENSORS,
        exact_parity=True,
        measured_by="validation/bench_rwkv_packed_cell_time_mix.py",
        test_module="test_rwkv_time_mix_receipt.py",
    ),
    Producer(
        name="channel-mix",
        label="Channel-mix",
        driver="hierarchos-vulkan/validation/bench_rwkv_packed_cell_channel_mix.py",
        bench_args=("--iterations", "20", "--warmup", "5"),
        guard_label="Channel-mix route across the sweep",
        receipt=VALIDATION / "packed_cell_channel_mix_bench.json",
        record=VALIDATION / "channel_mix_forward_decisions.json",
        kind="rwkv-channel-mix-forward",
        probe="linear_residual-forward",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("fused", "fused_ms"), ("staged", "composed_ms")),
        route_order=("fused", "staged"),
        default_route="fused",
        fallback="vendor",
        dispatch_key="linear_residual",
        fused_dispatches=0,
        composed_min_dispatches=0,
        tensors=PACKED_TENSORS,
        exact_parity=True,
        measured_by="validation/bench_rwkv_packed_cell_channel_mix.py",
        test_module="test_rwkv_channel_mix_receipt.py",
    ),
    Producer(
        name="low-rank-forward",
        label="LN1/low-rank forward",
        driver="hierarchos-vulkan/validation/bench_rwkv_packed_cell_low_rank_forward.py",
        bench_args=("--iterations", "20", "--warmup", "5"),
        guard_label="LN1/low-rank route across the sweep",
        receipt=VALIDATION / "packed_cell_low_rank_forward_bench.json",
        record=VALIDATION / "low_rank_full_forward_decisions.json",
        kind="rwkv-low-rank-full-forward",
        probe="low-rank-forward",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("fused", "fused_ms"), ("staged", "composed_ms")),
        route_order=("fused", "staged"),
        default_route="fused",
        fallback="structural",
        dispatch_key="low_rank_forward",
        fused_dispatches=1,
        composed_min_dispatches=6,
        composed_must_exceed_fused=True,
        tensors=PACKED_TENSORS,
        exact_parity=False,
        measured_by="validation/bench_rwkv_packed_cell_low_rank_forward.py",
        fused_arm="layernorm-low-rank-producer",
        test_module="test_rwkv_low_rank_forward_receipt.py",
    ),
    Producer(
        name="low-rank-forward-fp16",
        label="FP16-storage low-rank forward",
        driver="hierarchos-vulkan/validation/bench_rwkv_low_rank_fp16_forward.py",
        bench_args=("--iterations", "20", "--warmup", "5"),
        guard_label="FP16-storage low-rank route across the sweep",
        receipt=VALIDATION / "low_rank_fp16_forward_bench.json",
        record=VALIDATION / "low_rank_full_forward_decisions.json",
        kind="rwkv-low-rank-full-forward",
        probe="low-rank-forward",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("fused", "fused_ms"), ("staged", "composed_ms")),
        route_order=("fused", "staged"),
        default_route="fused",
        fallback="structural",
        dispatch_key="low_rank_forward",
        fused_dispatches=1,
        composed_min_dispatches=6,
        composed_must_exceed_fused=True,
        tensors=FP16_TENSORS,
        exact_parity=False,
        band=0.01,
        policy="fp16-storage",
        allow_shared=True,
        measured_by="validation/bench_rwkv_low_rank_fp16_forward.py",
        fused_arm="low-rank-full-forward-fused-fp16-packed",
        parameter_storage="fp16-storage",
        precision_policy="fp16-storage-fp32-compute",
        test_module="test_rwkv_low_rank_forward_receipt.py",
    ),
    Producer(
        name="key-state-forward",
        label="Key/state forward",
        driver="hierarchos-vulkan/validation/bench_rwkv_key_state_forward.py",
        bench_args=("--iterations", "20", "--repetitions", "50"),
        guard_label="Key/state route across the sweep",
        receipt=VALIDATION / "key_state_forward_bench.json",
        record=VALIDATION / "key_state_forward_decisions.json",
        kind="rwkv-key-state-forward",
        probe="key-state-forward",
        sweep=(1, 4, 8, 16, 32),
        candidates=(
            ("wide-fused", "fused_ms"),
            ("narrow-fused", "narrow_ms"),
            ("staged", "composed_ms"),
        ),
        route_order=("wide-fused", "narrow-fused", "staged"),
        default_route="wide-fused",
        fallback="structural",
        dispatch_key="key_state_forward",
        fused_dispatches=1,
        composed_min_dispatches=2,
        composed_exact=2,
        tensors=KEY_STATE_TENSORS,
        exact_parity=True,
        measured_by="validation/bench_rwkv_key_state_forward.py",
        fused_arm="key-state-forward-fused-wide",
        narrow_arm="key-state-forward-fused",
        composed_arm="key-transform-forward+state-forward",
        narrow_dispatches=1,
        wide_vs_narrow_parity=True,
        default_arm=True,
        unit_scale=1.0e3,
        unit="us",
        test_module="test_rwkv_key_state_forward_receipt.py",
        extra_artifacts=(VALIDATION / "key_state_forward_profile.json",),
        extra_test_modules=("test_rwkv_key_state_profile.py",),
    ),
)

PRODUCERS_BY_NAME = {producer.name: producer for producer in PRODUCERS}

# The TBPTT sequence reads the time-mix projection record rather than writing a
# record of its own. It is a consumer: the record holds the producer's full
# 1/4/8/16/32 sweep, while this receipt dispatches 1/4/8/16, so the recorded
# batch-32 verdict is not an unmeasured stray entry here.
TBPTT_TIME_MIX = Producer(
    name="tbptt-time-mix",
    label="Multi-step TBPTT",
    driver="hierarchos-vulkan/validation/bench_rwkv_tbptt_time_mix.py",
    bench_args=("--steps", "8", "--iterations", "8", "--warmup", "2"),
    guard_label="TBPTT time-mix route across the sweep",
    receipt=VALIDATION / "packed_cell_tbptt_time_mix_bench.json",
    record=VALIDATION / "time_mix_projection_decisions.json",
    kind="rwkv-time-mix-projection",
    probe="linear3-forward",
    sweep=(1, 4, 8, 16),
    candidates=(("fused", "fused_ms"), ("staged", "composed_ms")),
    route_order=("fused", "staged"),
    default_route="fused",
    fallback="vendor",
    dispatch_key="linear3",
    fused_dispatches=0,
    composed_min_dispatches=0,
    tensors=TBPTT_TENSORS,
    exact_parity=False,
    measured_by="validation/bench_rwkv_packed_cell_time_mix.py",
    consumer=True,
    vendor_dispatches_per_step=True,
    chains=(
        (
            VALIDATION / "tbptt_packed_forward_decisions.json",
            "rwkv-tbptt-packed-forward",
            "packed",
            "default_packed_forward",
        ),
        (
            VALIDATION / "tbptt_packed_rematerialization_decisions.json",
            "rwkv-tbptt-packed-rematerialization",
            "legacy",
            "default_packed_rematerialization",
        ),
    ),
    test_module="test_rwkv_tbptt_time_mix_receipt.py",
)


TBPTT_CHAIN_TENSORS = frozenset(
    {
        "outputs",
        "final_packed_state",
        "grad_x",
        "token_feature_grad",
        "grad_initial_packed_state",
    }
)

# The bounds the folded state-backward rung's receipt is held to. Its returned
# tensors may differ from the two-dispatch chain by a few contracted FP32 ulps on
# `grad_x`; `grad_k_k`/`grad_k_a` -- the reduction the cross-batch fold actually
# publishes, and the one a lost batch row moves -- may not differ at all. Those two
# numbers are the acceptance gate for the route, so this guard repeats them here
# instead of trusting the driver's verdict.
FOLDED_TENSOR_ATOL = 1.0e-6
FOLDED_PARAMETER_ATOL = 1.0e-6


@dataclass(frozen=True)
class ChainProducer:
    """A TBPTT chain A/B that writes its own schema-2 route record.

    The two packed chains are not projection producers: they replace a whole
    transition with a bit-identical graph, and their candidates are `packed` and
    `legacy`. The record is read by `RwkvTbpttSequenceOp` per batch, so the guard
    re-checks the same three things the other producers promise: the full sweep
    was measured, each entry is its own fastest candidate, and the shipped
    default followed the recorded route (or the structural default where the
    adapter has no entry).

    The backward-fusion chain has a third candidate, `folded`, which is the one
    route whose parity is not assumed: its key-parameter reduction has to be
    bit-identical to the two-dispatch chain's, and the receipt carries that
    readback. `folded_rung` switches on those checks and lets the candidate set
    vary per geometry, because the folded depth compiles only where the adapter
    offers it and only where the sweep proved the fold exact.
    """

    name: str
    label: str
    driver: str
    bench_args: tuple[str, ...]
    guard_label: str
    receipt: Path
    record: Path
    kind: str
    probe: str
    sweep: tuple[int, ...]
    candidates: tuple[tuple[str, str], ...]
    route_order: tuple[str, ...]
    default_route: str
    tensors: frozenset[str]
    measured_by: str
    jobs: tuple[str, ...] = ("producer-ab", "cross-adapter-ab")
    step_kind: str = "ab"
    band: float | None = None
    test_module: str = ""
    extra_artifacts: tuple[Path, ...] = ()
    extra_test_modules: tuple[str, ...] = ()
    # The ratios the receipt must carry, as (numerator, denominator, field). The
    # two packed chains report packed/legacy; the backward-fusion chain reports
    # fused/staged. `default_over_<baseline>` is checked against the baseline arm.
    ratios: tuple[tuple[str, str, str], ...] = (
        ("packed", "legacy", "packed_over_legacy"),
    )
    baseline: str = "legacy"
    # The shipped default is the device's own autotuned choice rather than a
    # structural route, so with no recorded entry the default may build either
    # measured arm and the receipt names which one it built. A recorded entry
    # still has to be followed exactly.
    default_from_measurement: bool = False
    # The chain's record also admits the fully folded rung as a third route, where
    # the receipt proved its key-parameter reduction exact. The flag turns on the
    # evidence checks below; the candidate list then carries the folded timing, and
    # a geometry whose device compiles no folded rung simply omits it.
    folded_rung: bool = False

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(route for route, _ in self.candidates)


CHAIN_PRODUCERS: tuple[ChainProducer, ...] = (
    ChainProducer(
        name="tbptt-packed-forward",
        label="TBPTT packed forward",
        driver="hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py",
        bench_args=(
            "--chain",
            "forward",
            "--steps",
            "8",
            "--iterations",
            "20",
            "--warmup",
            "2",
        ),
        guard_label="TBPTT packed-forward route across the sweep",
        receipt=VALIDATION / "tbptt_packed_forward_bench.json",
        record=VALIDATION / "tbptt_packed_forward_decisions.json",
        kind="rwkv-tbptt-packed-forward",
        probe="tbptt-packed-forward",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("legacy", "legacy_ms"), ("packed", "packed_ms")),
        route_order=("packed", "legacy"),
        default_route="packed",
        tensors=TBPTT_CHAIN_TENSORS,
        measured_by="validation/bench_rwkv_tbptt_packed_chains.py",
        test_module="test_rwkv_tbptt_packed_chains_receipt.py",
    ),
    ChainProducer(
        name="tbptt-packed-rematerialization",
        label="TBPTT packed rematerialization",
        driver="hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py",
        bench_args=(
            "--chain",
            "remat",
            "--steps",
            "8",
            "--iterations",
            "20",
            "--warmup",
            "2",
        ),
        guard_label="TBPTT packed-rematerialization route across the sweep",
        receipt=VALIDATION / "tbptt_packed_remat_bench.json",
        record=VALIDATION / "tbptt_packed_rematerialization_decisions.json",
        kind="rwkv-tbptt-packed-rematerialization",
        probe="tbptt-packed-rematerialization",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("legacy", "legacy_ms"), ("packed", "packed_ms")),
        route_order=("packed", "legacy"),
        default_route="legacy",
        tensors=TBPTT_CHAIN_TENSORS,
        measured_by="validation/bench_rwkv_tbptt_packed_chains.py",
        test_module="test_rwkv_tbptt_packed_chains_receipt.py",
    ),
    ChainProducer(
        name="tbptt-packed-forward-fp16",
        label="TBPTT packed forward (fp16-storage)",
        driver="hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py",
        bench_args=(
            "--chain",
            "forward",
            "--parameter-storage",
            "fp16-storage",
            "--steps",
            "8",
            "--iterations",
            "20",
            "--warmup",
            "2",
        ),
        guard_label="TBPTT packed-forward fp16-storage route across the sweep",
        receipt=VALIDATION / "tbptt_packed_forward_fp16_bench.json",
        record=VALIDATION / "tbptt_packed_forward_decisions.json",
        kind="rwkv-tbptt-packed-forward",
        probe="tbptt-packed-forward",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("legacy", "legacy_ms"), ("packed", "packed_ms")),
        route_order=("packed", "legacy"),
        default_route="packed",
        tensors=TBPTT_CHAIN_TENSORS,
        measured_by="validation/bench_rwkv_tbptt_packed_chains.py",
        test_module="test_rwkv_tbptt_packed_chains_receipt.py",
    ),
    ChainProducer(
        name="tbptt-packed-rematerialization-fp16",
        label="TBPTT packed rematerialization (fp16-storage)",
        driver="hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py",
        bench_args=(
            "--chain",
            "remat",
            "--parameter-storage",
            "fp16-storage",
            "--steps",
            "8",
            "--iterations",
            "20",
            "--warmup",
            "2",
        ),
        guard_label="TBPTT packed-rematerialization fp16-storage route across the sweep",
        receipt=VALIDATION / "tbptt_packed_remat_fp16_bench.json",
        record=VALIDATION / "tbptt_packed_rematerialization_decisions.json",
        kind="rwkv-tbptt-packed-rematerialization",
        probe="tbptt-packed-rematerialization",
        sweep=(1, 4, 8, 16, 32),
        candidates=(("legacy", "legacy_ms"), ("packed", "packed_ms")),
        route_order=("packed", "legacy"),
        default_route="legacy",
        tensors=TBPTT_CHAIN_TENSORS,
        measured_by="validation/bench_rwkv_tbptt_packed_chains.py",
        test_module="test_rwkv_tbptt_packed_chains_receipt.py",
    ),
    ChainProducer(
        name="tbptt-backward-fusion",
        label="TBPTT backward fusion",
        driver="hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py",
        # The two backward arms differ by well under a percent on this adapter, so
        # the sweep samples twice as long as the forward/remat chains: a 20-sample
        # run has twice flipped the sign outside the 2% band between passes, which
        # is exactly the instability the tie rule is meant to absorb.
        bench_args=(
            "--chain",
            "backward",
            "--steps",
            "8",
            "--iterations",
            "40",
            "--warmup",
            "3",
        ),
        guard_label="TBPTT backward-fusion route across the sweep",
        receipt=VALIDATION / "tbptt_backward_fusion_bench.json",
        record=VALIDATION / "tbptt_backward_fusion_decisions.json",
        kind="rwkv-tbptt-backward-fusion",
        probe="tbptt-backward-fusion",
        sweep=(1, 4, 8, 16, 32),
        candidates=(
            ("staged", "staged_ms"),
            ("fused", "fused_ms"),
            ("folded", "folded_ms"),
        ),
        route_order=("fused", "staged", "folded"),
        default_route="fused",
        tensors=TBPTT_CHAIN_TENSORS,
        measured_by="validation/bench_rwkv_tbptt_packed_chains.py",
        ratios=(("fused", "staged", "fused_over_staged"),),
        baseline="staged",
        default_from_measurement=True,
        folded_rung=True,
        test_module="test_rwkv_tbptt_packed_chains_receipt.py",
    ),
    ChainProducer(
        name="tbptt-backward-fusion-fp16",
        label="TBPTT backward fusion (fp16-storage)",
        driver="hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py",
        bench_args=(
            "--chain",
            "backward",
            "--parameter-storage",
            "fp16-storage",
            "--steps",
            "8",
            "--iterations",
            "40",
            "--warmup",
            "3",
        ),
        guard_label="TBPTT backward-fusion fp16-storage route across the sweep",
        receipt=VALIDATION / "tbptt_backward_fusion_fp16_bench.json",
        record=VALIDATION / "tbptt_backward_fusion_decisions.json",
        kind="rwkv-tbptt-backward-fusion",
        probe="tbptt-backward-fusion",
        sweep=(1, 4, 8, 16, 32),
        candidates=(
            ("staged", "staged_ms"),
            ("fused", "fused_ms"),
            ("folded", "folded_ms"),
        ),
        route_order=("fused", "staged", "folded"),
        default_route="fused",
        tensors=TBPTT_CHAIN_TENSORS,
        measured_by="validation/bench_rwkv_tbptt_packed_chains.py",
        ratios=(("fused", "staged", "fused_over_staged"),),
        baseline="staged",
        default_from_measurement=True,
        folded_rung=True,
        test_module="test_rwkv_tbptt_packed_chains_receipt.py",
    ),
)


@dataclass(frozen=True)
class CompositionConsumer:
    """A parity receipt that qualifies the staged route two records describe.

    The packed-cell composition receipt has no per-geometry route rows: it runs
    one fixed geometry with both forward fusions disabled, so the route it
    qualifies is the `staged` candidate. The records it consumes are the time-mix
    projection and channel-mix tables; at the receipt's width and batch, an entry
    for the qualified vendor must be absent (the gate decides) or name `staged`.
    Entries at other batches are that producer's sweep, not this receipt's, and
    are not stray.
    """

    name: str
    receipt: Path
    records: tuple[tuple[Path, str], ...]
    label: str
    driver: str
    bench_args: tuple[str, ...]
    jobs: tuple[str, ...]
    guard_label: str
    step_kind: str = "composition"
    test_module: str = ""
    extra_artifacts: tuple[Path, ...] = ()
    extra_test_modules: tuple[str, ...] = ()


PACKED_COMPOSITION = CompositionConsumer(
    name="packed-cell-composition",
    receipt=VALIDATION / "packed_cell_composition.json",
    records=(
        (VALIDATION / "time_mix_projection_decisions.json", "rwkv-time-mix-projection"),
        (VALIDATION / "channel_mix_forward_decisions.json", "rwkv-channel-mix-forward"),
    ),
    label="Packed-cell vendor composition",
    driver="hierarchos-vulkan/validation/verify_rwkv_packed_cell_composition.py",
    bench_args=("--expected-vendor", "auto"),
    jobs=("producer-ab", "cross-adapter-ab"),
    guard_label="Packed-cell composition route",
    test_module="test_rwkv_composition_receipt.py",
)

CONSUMERS = (TBPTT_TIME_MIX, PACKED_COMPOSITION)
ALL_SPECS = PRODUCERS + CHAIN_PRODUCERS + CONSUMERS
CONSUMERS_BY_NAME = {consumer.name: consumer for consumer in CONSUMERS}
ALL_SPECS_BY_NAME = {spec.name: spec for spec in ALL_SPECS}


def canonical_policy(policy: str | None) -> str | None:
    """The storage policy a record scope selects, as `decision_records` reads it."""
    if policy is None:
        return None
    policy = policy.strip().lower()
    if policy.startswith("fp16"):
        return "fp16-storage"
    if policy.startswith("bf16"):
        return "bf16-storage"
    return policy


def winner(timings: dict[str, float], order: tuple[str, ...]) -> str:
    return min(timings, key=lambda label: (timings[label], order.index(label)))


def band(candidate: float, baseline: float) -> float:
    return abs(candidate / baseline - 1.0)


def close(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-9 * max(1.0, abs(right))


def normalized(device: str) -> str:
    return device.strip().lower()


def entries_for(
    producer: Producer, record: dict, receipt: dict
) -> tuple[dict[int, dict], list[dict]]:
    """The entries this receipt reads, and every entry that applies at all.

    The first map is the lookup the shipped op performs: the receipt's policy
    first, then the policy-less fallback where that producer allows one. The
    second list keeps the scoped and shared entries separately so a stray entry
    is rejected even when a scoped one shadowed it.
    """
    device = normalized(receipt["device"])
    width = receipt["width"]
    applicable: list[dict] = []
    for entry in record["decisions"]:
        if normalized(entry["device"]) != device or entry["width"] != width:
            continue
        scope = canonical_policy(entry.get("parameter_storage"))
        if producer.policy is None:
            if scope is not None:
                continue
        elif scope is not None and scope != producer.policy:
            continue
        applicable.append(entry)
    lookup: dict[int, dict] = {}
    scopes: dict[int, str | None] = {}
    for entry in applicable:
        batch = entry["batch"]
        scope = canonical_policy(entry.get("parameter_storage"))
        previous = lookup.get(batch)
        if previous is None:
            lookup[batch] = entry
            scopes[batch] = scope
            continue
        if scopes[batch] == scope:
            raise Problem(
                f"batch {batch}: the record carries two {scope or 'policy-less'} entries "
                "for one geometry"
            )
        # The scoped verdict is what the op reads; the shared one is only the
        # fallback, so a scoped entry shadows it.
        if scope == producer.policy and scopes[batch] is None:
            lookup[batch] = entry
            scopes[batch] = scope
    held = withheld_for(producer, record, receipt)
    for batch in held:
        lookup.pop(batch, None)
    return lookup, applicable


def withheld_for(producer: Producer, record: dict, receipt: dict) -> dict[int, dict]:
    """The measured-but-not-adopted routes this receipt's device and policy can read.

    A withheld route is one the A/B measured ahead, but whose two graphs are not
    bit-identical while the training trajectory is held bit-exact. The runtime
    skips it and keeps the structural default, so the guard must not require the
    default to follow the entry -- and must still fail when the withheld route is
    not the one the entry measured, or carries no reason.
    """
    device = normalized(receipt["device"])
    width = receipt["width"]
    held: dict[int, dict] = {}
    for entry in record.get("withheld", []):
        if normalized(entry["device"]) != device or entry["width"] != width:
            continue
        scope = canonical_policy(entry.get("parameter_storage"))
        if producer.policy is None:
            if scope is not None:
                continue
        elif scope is not None and scope != producer.policy:
            continue
        held[entry["batch"]] = entry
    return held


def check_receipt(producer: Producer, receipt: dict) -> list[dict]:
    if receipt.get("result") != "pass":
        raise Problem(
            f"the receipt's result is {receipt.get('result')!r}, so it measured nothing usable"
        )
    if not receipt.get("batches"):
        raise Problem("the receipt carries no measured batch")
    if int(receipt.get("samples", 0)) < 1:
        raise Problem(f"the receipt records samples={receipt.get('samples')!r}")
    if "repetitions" in receipt and int(receipt["repetitions"]) < 1:
        raise Problem(f"the receipt records repetitions={receipt['repetitions']!r}")
    if producer.vendor_dispatches_per_step and int(receipt.get("steps", 0)) < 1:
        raise Problem(f"the receipt records steps={receipt.get('steps')!r}")
    if receipt.get("promotion") != "geometry-aware":
        raise Problem(
            f"the receipt's promotion is {receipt.get('promotion')!r}: the default stopped "
            "following the decision it should follow"
        )
    if not isinstance(receipt.get("device"), str) or not receipt["device"].strip():
        raise Problem("the receipt does not name the adapter it measured")
    if not isinstance(receipt.get("width"), int):
        raise Problem(f"the receipt records width={receipt.get('width')!r}")
    if producer.fused_arm is not None and receipt.get("fused_arm") != producer.fused_arm:
        raise Problem(
            f"the receipt names the fused arm {receipt.get('fused_arm')!r}, expected "
            f"{producer.fused_arm!r}"
        )
    if producer.parameter_storage is not None and (
        receipt.get("parameter_storage") != producer.parameter_storage
    ):
        raise Problem(
            f"the receipt reports parameter storage {receipt.get('parameter_storage')!r}, "
            f"expected {producer.parameter_storage!r}"
        )
    if producer.precision_policy is not None and (
        receipt.get("precision_policy") != producer.precision_policy
    ):
        raise Problem(
            f"the receipt reports policy {receipt.get('precision_policy')!r}, expected "
            f"{producer.precision_policy!r}"
        )
    if producer.narrow_arm is not None and receipt.get("narrow_arm") != producer.narrow_arm:
        raise Problem(
            f"the receipt names the narrow arm {receipt.get('narrow_arm')!r}, expected "
            f"{producer.narrow_arm!r}"
        )
    if producer.composed_arm is not None and receipt.get("composed_arm") != producer.composed_arm:
        raise Problem(
            f"the receipt names the chain {receipt.get('composed_arm')!r}, expected "
            f"{producer.composed_arm!r}"
        )
    if not producer.exact_parity and not float(receipt.get("atol", 0.0)) > 0.0:
        raise Problem("a tolerance-based producer must state a positive atol")
    batches = [row["batch"] for row in receipt["batches"]]
    if len(set(batches)) != len(batches):
        raise Problem(f"the receipt measures a batch twice: {batches}")
    return receipt["batches"]


def check_record(producer: Producer, record: dict) -> float:
    if record.get("kind") != producer.kind:
        raise Problem(
            f"the record is a {record.get('kind')!r} table, not {producer.kind!r}"
        )
    if record.get("schema") != 2:
        raise Problem(f"the record is schema {record.get('schema')!r}; the route table is schema 2")
    margin = record.get("margin")
    if not isinstance(margin, (int, float)) or margin <= 0.0:
        raise Problem(f"the record's noise band is {margin!r}")
    if producer.band is not None and producer.band != margin:
        margin = producer.band
    return float(margin)


def check_sweep(
    rows: list[dict], sweep: tuple[int, ...], unmeasured: list[int] | None = None
) -> None:
    """Every sweep geometry was dispatched, or the receipt says why it could not be.

    A receipt may carry an `unmeasured` list, which is a claim about the device's
    kernel set rather than a skip: the backward-fusion A/B reports a geometry the
    device compiles only one topology for, because one graph cannot be compared
    with itself. Those geometries are still required to be reported -- dropping
    one from both the sweep and the list would hide it -- and a geometry the
    receipt did measure may not appear in the list.
    """
    measured = {row["batch"] for row in rows}
    reported = set(unmeasured or [])
    overlap = sorted(measured & reported)
    if overlap:
        raise Problem(
            "batch " + ", ".join(str(batch) for batch in overlap) + " is both measured and "
            "reported unmeasured"
        )
    stray = sorted(reported - set(sweep))
    if stray:
        raise Problem(
            "batch "
            + ", ".join(str(batch) for batch in stray)
            + " is reported unmeasured but is not part of the sweep"
        )
    missing = [batch for batch in sweep if batch not in measured and batch not in reported]
    if missing:
        raise Problem(
            "the receipt does not dispatch batch "
            + ", ".join(str(batch) for batch in missing)
            + " -- the sweep that decides this route is incomplete, so a crossover there "
            "would go unmeasured"
        )


def check_entries(
    producer: Producer,
    record: dict,
    rows: list[dict],
    lookup: dict[int, dict],
    applicable: list[dict],
    withheld: dict[int, dict],
    margin: float,
) -> None:
    measured = {row["batch"] for row in rows}
    # A consumer reads a producer's record, which covers that producer's full
    # sweep. A verdict at a batch this consumer did not dispatch is the
    # producer's evidence, not a stray entry here.
    stray = [] if producer.consumer else sorted(
        {entry["batch"] for entry in applicable} - measured
    )
    if stray:
        raise Problem(
            "the record carries a decision for batch "
            + ", ".join(str(batch) for batch in stray)
            + " that this receipt never measured, so the verdict is unsupported"
        )
    for entry in applicable:
        batch = entry["batch"]
        if entry.get("probe") != producer.probe:
            raise Problem(
                f"batch {batch}: the record's probe is {entry.get('probe')!r}, expected "
                f"{producer.probe!r}"
            )
        if entry.get("source") != "measured":
            raise Problem(
                f"batch {batch}: the record's source is {entry.get('source')!r}, so the "
                "route was never measured"
            )
        scope = canonical_policy(entry.get("parameter_storage"))
        if producer.policy is None or scope == producer.policy:
            driver = entry.get("measured_by", record.get("measured_by"))
            if driver != producer.measured_by:
                raise Problem(
                    f"batch {batch}: the entry names the driver {driver!r}, expected "
                    f"{producer.measured_by!r}"
                )
        if set(entry["candidates"]) != set(producer.routes):
            raise Problem(
                f"batch {batch}: the record's candidates are "
                f"{sorted(entry['candidates'])}, expected {sorted(producer.routes)}"
            )
        timings = entry["candidates"]
        for label, value in timings.items():
            if not isinstance(value, (int, float)) or value <= 0.0:
                raise Problem(f"batch {batch}: the {label} candidate has timing {value!r}")
        best = winner(timings, producer.route_order)
        if entry["route"] != best:
            raise Problem(
                f"batch {batch}: the record names {entry['route']!r} but {best!r} is the "
                "fastest candidate it lists "
                f"({', '.join(f'{label} {value * 1e3:.3f} us' for label, value in timings.items())})"
            )
        if entry["route"] != producer.default_route:
            drift = band(timings[entry["route"]], timings[producer.default_route])
            if drift <= margin:
                raise Problem(
                    f"batch {batch}: {entry['route']!r} wins by {drift:.1%}, inside the "
                    f"{margin:.0%} noise band, so the decision is not stable"
                )
    for batch, entry in withheld.items():
        route = next(
            (candidate for candidate in applicable if candidate["batch"] == batch), None
        )
        if route is None:
            raise Problem(
                f"batch {batch}: the record withholds a route but carries no measured decision "
                "for that geometry"
            )
        if not str(entry.get("reason", "")).strip():
            raise Problem(f"batch {batch}: the withheld route carries no reason")
        timings = route["candidates"]
        if entry.get("route") != winner(timings, producer.route_order):
            raise Problem(
                f"batch {batch}: the record withholds {entry.get('route')!r}, but the measured "
                f"decision names {route['route']!r}"
            )
    # The lookup is a subset of the applicable entries; every entry it points at
    # has already been checked above.
    resolved = {entry["batch"] for entry in applicable if entry["batch"] not in withheld}
    if lookup and len(lookup) != len(resolved):
        raise Problem("the record's scoped and shared entries do not resolve to one route per geometry")


def composed_vendor_dispatches(
    producer: Producer, receipt: dict, row: dict, batch: int
) -> int:
    """How often the composed leg must dispatch its vendor module.

    A sequence dispatches the projection once per timestep. The first-forward
    chain decides whether it runs a second time: the packed transition shares
    the forward pass with rematerialization, while the legacy transition replays
    the whole forward pass in reverse, so the composed projection is dispatched
    once in each pass. This is why the receipt's count is `steps`, or two times
    `steps` after batch 16 moved to the legacy first-forward route; a count that
    does not follow the chain the default actually built means the receipt was
    recorded against a different default.
    """
    if not producer.vendor_dispatches_per_step:
        return 1
    steps = int(receipt["steps"])
    if not producer.chains:
        return steps
    forward = row.get("default_packed_forward")
    if forward is None:
        raise Problem(
            f"batch {batch}: the receipt does not record default_packed_forward, so the "
            "composed dispatch count cannot be tied to the shipped chain route; re-record it"
        )
    return (1 if forward else 2) * steps


def check_row(
    producer: Producer,
    receipt: dict,
    row: dict,
    entry: dict | None,
    margin: float,
    default_checked: bool,
    withheld: bool = False,
) -> None:
    batch = row["batch"]
    timings = {label: row[field] for label, field in producer.candidates}
    for label, value in timings.items():
        if not isinstance(value, (int, float)) or value <= 0.0:
            raise Problem(f"batch {batch}: the {label} candidate has timing {value!r}")

    if row.get("fused_label") != "fused":
        raise Problem(
            f"batch {batch}: the fused leg reports {row.get('fused_label')!r}, so it is not "
            "the graph this check names"
        )
    if row.get("composed_label") != "composed":
        raise Problem(
            f"batch {batch}: the composed leg reports {row.get('composed_label')!r}"
        )

    worst = row.get("worst_max_abs")
    if not isinstance(worst, (int, float)):
        raise Problem(f"batch {batch}: the receipt has no worst_max_abs")
    if producer.exact_parity:
        if worst != 0.0:
            raise Problem(
                f"batch {batch}: the arms diverge (worst |diff| {worst:.3e}); a route may "
                "only choose between graphs that compute the same values"
            )
        if producer.wide_vs_narrow_parity and row.get("worst_max_abs_wide_vs_narrow") != 0.0:
            raise Problem(
                f"batch {batch}: wide and narrow diverge "
                f"(worst |diff| {row['worst_max_abs_wide_vs_narrow']:.3e})"
            )
    else:
        atol = float(receipt["atol"])
        if worst > atol:
            raise Problem(
                f"batch {batch}: the arms diverge by {worst:.3e}, outside the stated atol "
                f"{atol:.3e}; a route may not promote across a numerics change"
            )

    labels = {tensor["label"] for tensor in row.get("tensors", [])}
    if labels != producer.tensors:
        raise Problem(
            f"batch {batch}: compared tensors {sorted(labels)} do not match "
            f"{sorted(producer.tensors)}"
        )
    for tensor in row["tensors"]:
        excess = tensor.get("worst_excess")
        if isinstance(excess, (int, float)) and excess > 0.0:
            raise Problem(
                f"batch {batch}: {tensor['label']} left the atol+rtol band "
                f"(worst excess {excess:.3e})"
            )

    fused = row[f"fused_{producer.dispatch_key}_dispatches"]
    composed = row[f"composed_{producer.dispatch_key}_dispatches"]
    default = row[f"default_{producer.dispatch_key}_dispatches"]
    if fused != producer.fused_dispatches:
        raise Problem(
            f"batch {batch}: the fused leg dispatched {fused} kernel(s), expected "
            f"{producer.fused_dispatches}"
        )
    if producer.narrow_dispatches is not None:
        narrow = row[f"narrow_{producer.dispatch_key}_dispatches"]
        if narrow != producer.narrow_dispatches:
            raise Problem(
                f"batch {batch}: the narrow leg dispatched {narrow} kernel(s), expected "
                f"{producer.narrow_dispatches}"
            )
    if producer.fallback == "vendor":
        vendor = bool(row.get(producer.vendor_flag or ""))
        expected_composed = 0
        if vendor:
            expected_composed = composed_vendor_dispatches(producer, receipt, row, batch)
        if composed != expected_composed:
            raise Problem(
                f"batch {batch}: the composed leg dispatched {composed} vendor kernel(s), "
                f"but its vendor flag is {vendor}"
            )
    else:
        if producer.composed_exact is not None:
            if composed != producer.composed_exact:
                raise Problem(
                    f"batch {batch}: the composed leg dispatched {composed} kernel(s), "
                    f"expected {producer.composed_exact}"
                )
        elif composed < producer.composed_min_dispatches:
            raise Problem(
                f"batch {batch}: the composed leg dispatched only {composed} kernel(s), "
                f"expected at least {producer.composed_min_dispatches}"
            )
        if producer.composed_must_exceed_fused and composed <= fused:
            raise Problem(
                f"batch {batch}: the fusion did not replace a dispatch ({composed} against "
                f"{fused}), so the A/B is vacuous"
            )

    composed_ratio = row["composed_over_fused"]
    default_ratio = row["default_over_fused"]
    baseline = timings[producer.default_route]
    if not close(composed_ratio, timings["staged"] / baseline):
        raise Problem(
            f"batch {batch}: composed_over_fused {composed_ratio!r} is not the measured ratio "
            f"{timings['staged'] / baseline!r}"
        )
    if not close(default_ratio, row["default_ms"] / baseline):
        raise Problem(
            f"batch {batch}: default_over_fused {default_ratio!r} is not the measured ratio "
            f"{row['default_ms'] / baseline!r}"
        )
    if producer.wide_vs_narrow_parity:
        residual = row["narrow_over_fused"] * row["composed_over_narrow"] - composed_ratio
        if abs(residual) > 1.0e-9:
            raise Problem(
                f"batch {batch}: the wide/narrow/chain ratios do not multiply out "
                f"({residual:.3e} residual), so the three legs are not one measurement"
            )
        if not close(row["narrow_over_fused"], timings["narrow-fused"] / timings["wide-fused"]):
            raise Problem(f"batch {batch}: narrow_over_fused is not the measured narrow/wide ratio")
        if not close(row["composed_over_narrow"], timings["staged"] / timings["narrow-fused"]):
            raise Problem(f"batch {batch}: composed_over_narrow is not the measured chain/narrow ratio")

    if entry is not None:
        recorded_route = row.get("recorded_route")
        if recorded_route is None and row.get("recorded_compose") is not None:
            recorded_route = "staged" if row["recorded_compose"] else "fused"
        if recorded_route is not None and recorded_route != entry["route"]:
            raise Problem(
                f"batch {batch}: the receipt was measured against route {recorded_route!r} but "
                f"the record now names {entry['route']!r}; rebuild and re-record the receipt"
            )
        measured_winner = winner(timings, producer.route_order)
        if measured_winner != entry["route"]:
            drift = band(timings[measured_winner], timings[entry["route"]])
            if drift > margin:
                raise Problem(
                    f"batch {batch}: the record names {entry['route']!r}, but this receipt "
                    f"measures {measured_winner!r} ahead by {drift:.1%}, outside the "
                    f"{margin:.0%} noise band; the record is stale"
                )
    elif default_checked and not producer.consumer and not withheld:
        # This adapter has decisions elsewhere, so a geometry outside the band
        # with no entry is a measured crossover the record has not absorbed. The
        # first check covers a winning arm that is not the default route at all
        # (the key/state narrow arm); the second covers the staged/wide sign.
        measured = winner(timings, producer.route_order)
        if measured != producer.default_route and band(
            timings[measured], timings[producer.default_route]
        ) > margin:
            raise Problem(
                f"batch {batch}: {measured!r} is ahead of the default route by "
                f"{band(timings[measured], timings[producer.default_route]):.1%}, outside "
                f"the {margin:.0%} noise band, but the record carries no decision for it"
            )
        if band(composed_ratio, 1.0) > margin:
            raise Problem(
                f"batch {batch}: the measured route is {composed_ratio:.3f} against the "
                f"default, outside the {margin:.0%} noise band, but the record carries no "
                "decision for it"
            )

    if not default_checked:
        if row.get("default_label") != row.get("expected_default_label"):
            raise Problem(
                f"batch {batch}: the default built {row.get('default_label')!r}, while the "
                f"receipt expected {row.get('expected_default_label')!r}"
            )
        return

    if entry is not None:
        expected = "composed" if entry["route"] == "staged" else row["fused_label"]
    elif producer.fallback == "vendor":
        expected = "composed" if row.get(producer.vendor_flag or "") else row["fused_label"]
    else:
        expected = row["fused_label"]
    if row.get("default_label") != expected:
        raise Problem(
            f"batch {batch}: the default built {row.get('default_label')!r}, expected "
            f"{expected!r}"
        )
    if row.get("expected_default_label") != expected:
        raise Problem(
            f"batch {batch}: the receipt expected {row.get('expected_default_label')!r}, "
            f"expected {expected!r}; the receipt and the record were not measured together"
        )
    expected_dispatches = composed if expected == "composed" else fused
    if default != expected_dispatches:
        raise Problem(
            f"batch {batch}: the default leg dispatched {default} kernel(s), but the "
            f"{'composed' if expected == 'composed' else 'fused'} leg it built dispatched "
            f"{expected_dispatches}"
        )
    if producer.default_arm:
        if entry is not None and entry["route"] == "staged":
            expected_arm = None
        elif entry is not None and entry["route"] == "narrow-fused":
            expected_arm = receipt.get("narrow_arm")
        else:
            expected_arm = receipt.get("fused_arm")
        if row.get("default_arm") != expected_arm:
            raise Problem(
                f"batch {batch}: the default dispatched arm {row.get('default_arm')!r}, "
                f"expected {expected_arm!r}"
            )


def check_consumer_chain_routes(producer: Producer, receipt: dict) -> None:
    """The consumer's default leg must have built the shipped chain routes.

    The TBPTT receipt is measured on the shipped default, and that default now
    dispatches the first-forward and rematerialization routes the chain records
    name. A receipt recorded before a route moved still looks internally
    consistent, so this compares the chains the default reported against the
    records instead of trusting the timing alone.
    """
    device = normalized(receipt["device"])
    width = receipt["width"]
    for record_path, kind, default, field in producer.chains:
        if not record_path.exists():
            raise Problem(f"missing {record_path}; the chain route record is not optional")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("kind") != kind or record.get("schema") != 2:
            raise Problem(
                f"{record_path.name} is {record.get('kind')!r} schema "
                f"{record.get('schema')!r}, expected {kind!r} schema 2"
            )
        entries = {
            entry["batch"]: entry
            for entry in record["decisions"]
            if normalized(entry["device"]) == device and entry["width"] == width
        }
        for row in receipt["batches"]:
            batch = row["batch"]
            if field not in row:
                raise Problem(
                    f"batch {batch}: the receipt does not record the {field} route it ran; "
                    "re-record it against the chain-routed default"
                )
            entry = entries.get(batch)
            expected = entry["route"] if entry else default
            actual = "packed" if row[field] else "legacy"
            if actual != expected:
                raise Problem(
                    f"batch {batch}: the TBPTT default built the {actual} {kind} chain, but "
                    f"the record names {expected}; re-record the receipt after the route moved"
                )


def check_producer(
    producer: Producer,
    *,
    receipt_path: Path,
    record_path: Path,
    sweep: tuple[int, ...],
) -> list[str]:
    if not receipt_path.exists():
        raise Problem(f"missing {receipt_path}; record it with the producer's A/B harness")
    if not record_path.exists():
        raise Problem(f"missing {record_path}; the route record is not optional")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    rows = check_receipt(producer, receipt)
    margin = check_record(producer, record)
    lookup, applicable = entries_for(producer, record, receipt)
    withheld = withheld_for(producer, record, receipt)
    check_sweep(rows, sweep)
    check_entries(producer, record, rows, lookup, applicable, withheld, margin)
    if producer.consumer and producer.chains:
        check_consumer_chain_routes(producer, receipt)
    default_checked = bool(applicable)
    report = []
    for row in rows:
        entry = lookup.get(row["batch"])
        check_row(
            producer,
            receipt,
            row,
            entry,
            margin,
            default_checked,
            withheld=row["batch"] in withheld,
        )
        shown = "  ".join(
            f"{label} {row[field] * producer.unit_scale:7.3f} {producer.unit}"
            for label, field in producer.candidates
        )
        if row["batch"] in withheld:
            decision = "withheld"
        elif entry is not None and default_checked:
            decision = entry["route"]
        else:
            decision = "fallback"
        report.append(f"batch {row['batch']:>2}: {decision:<12} {shown}")
    return report


def receipt_policy(receipt: dict) -> str | None:
    """The parameter-storage policy a chain receipt measured under.

    `fp32` and an absent field are the same unscoped default; only a scoped
    receipt names a storage policy.
    """
    policy = receipt.get("parameter_storage")
    if policy in (None, "", "fp32"):
        return None
    return canonical_policy(policy)


def chain_entries(record: dict, receipt: dict) -> list[dict]:
    """The chain entries this receipt's adapter, width and policy can read.

    A scoped entry from another policy is that policy's measurement, so this
    receipt neither validates nor applies it. The shared entries are the default
    policy's evidence; a scoped receipt reads them only as the fallback the
    runtime uses where its own policy has no entry.
    """
    device = normalized(receipt["device"])
    width = receipt["width"]
    policy = receipt_policy(receipt)
    entries = []
    for entry in record["decisions"]:
        if normalized(entry["device"]) != device or entry["width"] != width:
            continue
        scope = canonical_policy(entry.get("parameter_storage"))
        if policy is None and scope is not None:
            continue
        if policy is not None and scope is not None and scope != policy:
            continue
        entries.append(entry)
    return entries


def chain_lookup(record: dict, receipt: dict, applicable: list[dict]) -> dict[int, dict]:
    """The entry the runtime adopts per batch for this receipt's policy.

    A scoped verdict wins over the shared fallback, and a refusal for this
    policy withdraws the batch entirely (the structural default decides it).
    """
    policy = receipt_policy(receipt)
    shared = {
        entry["batch"]: entry
        for entry in applicable
        if canonical_policy(entry.get("parameter_storage")) is None
    }
    lookup = dict(shared)
    if policy is not None:
        for entry in applicable:
            if canonical_policy(entry.get("parameter_storage")) == policy:
                lookup[entry["batch"]] = entry
    for batch in chain_withheld_batches(record, receipt):
        lookup.pop(batch, None)
    return lookup


def chain_withheld_batches(record: dict, receipt: dict) -> set[int]:
    """Batches this receipt's policy refuses, whether scoped or shared."""
    device = normalized(receipt["device"])
    width = receipt["width"]
    policy = receipt_policy(receipt)
    held = set()
    for entry in record.get("withheld", []):
        if normalized(entry["device"]) != device or entry["width"] != width:
            continue
        scope = canonical_policy(entry.get("parameter_storage"))
        if policy is None:
            if scope is None:
                held.add(entry["batch"])
        elif scope == policy:
            held.add(entry["batch"])
    return held


def chain_route_for(
    decisions: list[dict], device: str, width: int, batch: int, policy: str
) -> dict | None:
    """The entry one policy would read at a geometry: scoped first, then shared."""
    scoped = None
    shared = None
    for entry in decisions:
        if normalized(entry["device"]) != device or entry["width"] != width:
            continue
        if entry["batch"] != batch:
            continue
        scope = canonical_policy(entry.get("parameter_storage"))
        if scope == policy:
            scoped = entry
        elif scope is None:
            shared = entry
    return scoped or shared


# The state-backward fusion rungs, most decomposed first. A schedule label starts
# with one of these and continues with `+`-separated projection and fan-in factors,
# whose own separators make splitting on `+` unsafe.
BACKWARD_FUSION_RUNGS = ("rkv-add3+key+reduce", "rkv-add3-key+reduce", "rkv-add3-key-reduce")
FOLDED_RUNG = BACKWARD_FUSION_RUNGS[2]


def backward_fusion_rung(label: str) -> str | None:
    for rung in BACKWARD_FUSION_RUNGS:
        if label == rung or label.startswith(f"{rung}+"):
            return rung
    return None


def check_folded_evidence(chain: ChainProducer, row: dict) -> set[str]:
    """Re-derive the folded rung's acceptance gate from one receipt row.

    Returns the candidate labels this geometry may record: the two-dispatch pair
    always, plus `folded` when the receipt carries the evidence the route rests on.

    The gate is the fold's own output. `grad_k_k`/`grad_k_a` are what the
    cross-batch reduction publishes and no returned tensor consumes, so a fold
    that loses a batch row or never publishes is invisible everywhere else; the
    receipt reads them back and they must equal the two-dispatch chain's bit for
    bit, with the staged arm's gradient non-zero so the agreement is not two
    zeroes. The returned tensors and one AdamW step are held to the recorded ulp
    bounds around it, not instead of it.
    """
    batch = row["batch"]
    candidates = {label for label, field in chain.candidates if field in row}
    stated = row.get("folded_candidate")
    if "folded" in candidates and stated is not True:
        raise Problem(
            f"batch {batch}: the receipt timed the folded rung but does not mark it a "
            "routing candidate, so its timing cannot be recorded as one"
        )
    if "folded" not in candidates:
        if stated:
            raise Problem(
                f"batch {batch}: the receipt marks a folded candidate it never timed"
            )
        return candidates
    if row.get("folded_label") != "folded":
        raise Problem(
            f"batch {batch}: the folded leg reports {row.get('folded_label')!r}, so it is "
            "not the graph this check names"
        )
    schedule = row.get("folded_schedule")
    staged_schedule = row.get("staged_schedule")
    if not isinstance(schedule, str) or not isinstance(staged_schedule, str):
        raise Problem(f"batch {batch}: the folded evidence does not name the schedules it built")
    rung = backward_fusion_rung(schedule)
    staged_rung = backward_fusion_rung(staged_schedule)
    if rung != FOLDED_RUNG:
        raise Problem(
            f"batch {batch}: the folded leg pinned {schedule!r}, which is not the fully "
            f"folded rung {FOLDED_RUNG!r}"
        )
    if staged_rung != BACKWARD_FUSION_RUNGS[0]:
        raise Problem(
            f"batch {batch}: the staged leg pinned {staged_schedule!r}, which is not the "
            "fully staged rung, so the two candidates are not the ends of the ladder"
        )
    if schedule[len(rung) :] != staged_schedule[len(staged_rung) :]:
        raise Problem(
            f"batch {batch}: the folded and staged arms differ in projection order or "
            f"fan-in as well as depth ({schedule!r} vs {staged_schedule!r})"
        )
    magnitude = row.get("staged_key_parameter_max_abs")
    if not isinstance(magnitude, (int, float)) or magnitude <= 0.0:
        raise Problem(
            f"batch {batch}: the staged arm's grad_k_k/grad_k_a are {magnitude!r}, so the "
            "folded rung's agreement with them would prove nothing"
        )
    delta = row.get("folded_key_parameter_max_abs")
    if not isinstance(delta, (int, float)):
        raise Problem(f"batch {batch}: the receipt carries no folded key-parameter readback")
    if delta != 0.0:
        raise Problem(
            f"batch {batch}: the folded rung moved grad_k_k/grad_k_a by {delta:.3e} away "
            "from the two-dispatch chain; the route's acceptance gate admits no allowance"
        )
    tensors = row.get("folded_max_abs")
    if not isinstance(tensors, (int, float)) or tensors > FOLDED_TENSOR_ATOL:
        raise Problem(
            f"batch {batch}: the folded rung's returned tensors diverge by {tensors!r}, "
            f"beyond the {FOLDED_TENSOR_ATOL:.1e} contracted-ulp bound"
        )
    step = row.get("folded_parameter_delta")
    if not isinstance(step, (int, float)) or step > FOLDED_PARAMETER_ATOL:
        raise Problem(
            f"batch {batch}: one AdamW step on the folded rung moved a parameter by "
            f"{step!r}, beyond the {FOLDED_PARAMETER_ATOL:.1e} bound"
        )
    ratio = row.get("folded_over_staged")
    if not close(ratio, row["folded_ms"] / row["staged_ms"]):
        raise Problem(
            f"batch {batch}: folded_over_staged is not the measured folded/staged ratio"
        )
    return candidates


def check_chain_withheld(
    chain: ChainProducer, record: dict, receipt: dict
) -> dict[str, list[int]]:
    """Check the measured verdicts the chain table refuses, per policy.

    A chain is bit-identical to its alternative by construction, so a withheld
    chain route is not a numerical-parity claim: it records that one policy may
    not adopt the measured verdict because that policy's frozen trajectory is
    pinned to the structural graph. The route therefore has to be the verdict the
    policy would otherwise read, the refusal has to be scoped (withholding the
    shared verdict would contradict the receipt that measured the default
    following it), and every verdict that differs from the structural default has
    to be refused once a policy refuses anything.
    """
    device = normalized(receipt["device"])
    width = receipt["width"]
    policy = receipt_policy(receipt)
    measured = {row["batch"] for row in receipt["batches"]}

    def adopted_routes(scope: str) -> set[int]:
        """Batches this policy measured itself, so it has a verdict to adopt."""
        return {
            entry["batch"]
            for entry in record["decisions"]
            if normalized(entry["device"]) == device
            and entry["width"] == width
            and canonical_policy(entry.get("parameter_storage")) == scope
        }

    refusals: dict[str, list[int]] = {}
    seen: set[tuple[str, int]] = set()
    for entry in record.get("withheld", []):
        if normalized(entry["device"]) != device or entry["width"] != width:
            continue
        batch = entry["batch"]
        scope = canonical_policy(entry.get("parameter_storage"))
        if scope is None:
            raise Problem(
                f"batch {batch}: a chain refusal has to name the policy it applies to; "
                "withholding the shared verdict would contradict the receipt that measured "
                "the default following it"
            )
        # A receipt measured under one policy validates that policy's refusals.
        # The default-policy receipt provides the shared measurement every
        # scoped refusal names, so it validates all of them; a scoped receipt's
        # own timings are not evidence for another policy's fallback.
        if policy is not None and scope != policy:
            continue
        if not str(entry.get("reason", "")).strip():
            raise Problem(f"batch {batch}: the withheld {scope} route carries no reason")
        if batch not in measured:
            raise Problem(
                f"batch {batch}: the {scope} refusal has no measured geometry in this receipt"
            )
        if batch in adopted_routes(scope):
            raise Problem(
                f"batch {batch}: the {scope} policy measured its own verdict here, so a "
                "refusal cannot also cover it; drop the refusal when the policy adopts a route"
            )
        target = chain_route_for(record["decisions"], device, width, batch, scope)
        if target is None:
            raise Problem(
                f"batch {batch}: the {scope} refusal has no decision to refuse, so it was "
                "not measured"
            )
        if entry["route"] != target["route"]:
            raise Problem(
                f"batch {batch}: the {scope} refusal names {entry['route']!r} but that "
                f"policy would read {target['route']!r}; a refusal has to name the measured "
                "verdict it refuses"
            )
        if entry["route"] == chain.default_route:
            raise Problem(
                f"batch {batch}: the {scope} refusal names the structural route "
                f"{chain.default_route!r}, so there is nothing to refuse"
            )
        key = (scope, batch)
        if key in seen:
            raise Problem(f"batch {batch}: the {scope} policy refuses one geometry twice")
        seen.add(key)
        refusals.setdefault(scope, []).append(batch)
    for scope, batches in refusals.items():
        expected = sorted(
            entry["batch"]
            for entry in record["decisions"]
            if normalized(entry["device"]) == device
            and entry["width"] == width
            and entry.get("parameter_storage") is None
            and entry["route"] != chain.default_route
            and entry["batch"] in measured
            and entry["batch"] not in adopted_routes(scope)
        )
        if sorted(batches) != expected:
            raise Problem(
                f"the {scope} policy refuses batches "
                f"{', '.join(str(batch) for batch in sorted(batches))} but this table "
                f"records non-structural verdicts at "
                f"{', '.join(str(batch) for batch in expected) or 'none'}; a policy that "
                "refuses anything has to refuse every measured move away from the "
                "structural route that it has not measured itself"
            )
    return refusals


def check_chain_producer(
    chain: ChainProducer,
    *,
    receipt_path: Path,
    record_path: Path,
    sweep: tuple[int, ...],
) -> list[str]:
    """Re-check a packed TBPTT chain receipt against its own route record."""
    if not receipt_path.exists():
        raise Problem(f"missing {receipt_path}; record it with {chain.driver}")
    if not record_path.exists():
        raise Problem(f"missing {record_path}; the route record is not optional")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if receipt.get("result") != "pass":
        raise Problem(
            f"the receipt's result is {receipt.get('result')!r}, so it measured nothing usable"
        )
    if not isinstance(receipt.get("device"), str) or not receipt["device"].strip():
        raise Problem("the receipt does not name the adapter it measured")
    if not isinstance(receipt.get("width"), int):
        raise Problem(f"the receipt records width={receipt.get('width')!r}")
    if int(receipt.get("steps", 0)) < 1:
        raise Problem(f"the receipt records steps={receipt.get('steps')!r}")
    if int(receipt.get("samples", 0)) < 1:
        raise Problem(f"the receipt records samples={receipt.get('samples')!r}")
    batches = [row["batch"] for row in receipt.get("batches", [])]
    if not batches:
        raise Problem("the receipt carries no measured batch")
    if len(set(batches)) != len(batches):
        raise Problem(f"the receipt measures a batch twice: {batches}")
    if record.get("kind") != chain.kind:
        raise Problem(f"the record is a {record.get('kind')!r} table, not {chain.kind!r}")
    if record.get("schema") != 2:
        raise Problem(f"the record is schema {record.get('schema')!r}; the route table is schema 2")
    margin = record.get("margin")
    if not isinstance(margin, (int, float)) or margin <= 0.0:
        raise Problem(f"the record's noise band is {margin!r}")
    if chain.band is not None and chain.band != margin:
        margin = chain.band
    margin = float(margin)

    check_sweep(receipt["batches"], tuple(sweep), receipt.get("unmeasured"))

    applicable = chain_entries(record, receipt)
    policy = receipt_policy(receipt)
    measured = set(batches)
    # This receipt's own measurements: its scoped entries, or every shared entry
    # for the default policy. A scoped receipt reads a shared verdict it did not
    # measure only as its fallback, so that verdict's timings are not re-checked
    # against this run.
    measured_entries = [
        entry
        for entry in applicable
        if canonical_policy(entry.get("parameter_storage")) == policy
    ]
    stray = sorted({entry["batch"] for entry in measured_entries} - measured)
    if stray:
        raise Problem(
            "the record carries a decision for batch "
            + ", ".join(str(batch) for batch in stray)
            + " that this receipt never measured, so the verdict is unsupported"
        )
    lookup = chain_lookup(record, receipt, applicable)
    withheld_batches = chain_withheld_batches(record, receipt)
    # What each measured geometry could record: the two-dispatch pair always, the
    # folded rung only where this receipt's fold cleared its gate. The record's
    # candidate list has to equal this set exactly, so an entry cannot claim a
    # measurement that was not taken or silently drop one that was.
    supported: dict[int, set[str]] = {}
    for row in receipt["batches"]:
        if chain.folded_rung:
            supported[row["batch"]] = check_folded_evidence(chain, row)
        else:
            supported[row["batch"]] = {
                label for label, field in chain.candidates if field in row
            }
    for entry in measured_entries:
        batch = entry["batch"]
        if entry.get("probe") != chain.probe:
            raise Problem(
                f"batch {batch}: the record's probe is {entry.get('probe')!r}, expected "
                f"{chain.probe!r}"
            )
        if entry.get("source") != "measured":
            raise Problem(
                f"batch {batch}: the record's source is {entry.get('source')!r}, so the "
                "route was never measured"
            )
        driver = entry.get("measured_by", record.get("measured_by"))
        if driver != chain.measured_by:
            raise Problem(
                f"batch {batch}: the entry names the driver {driver!r}, expected "
                f"{chain.measured_by!r}"
            )
        expected_candidates = supported.get(batch, set(chain.routes))
        if set(entry["candidates"]) != expected_candidates:
            raise Problem(
                f"batch {batch}: the record's candidates are "
                f"{sorted(entry['candidates'])}, expected {sorted(expected_candidates)}; "
                "the folded rung is a candidate only where the receipt proved its "
                "key-parameter reduction exact"
            )
        timings = entry["candidates"]
        for label, value in timings.items():
            if not isinstance(value, (int, float)) or value <= 0.0:
                raise Problem(f"batch {batch}: the {label} candidate has timing {value!r}")
        best = winner(timings, chain.route_order)
        if entry["route"] != best:
            raise Problem(
                f"batch {batch}: the record names {entry['route']!r} but {best!r} is the "
                "fastest candidate it lists "
                f"({', '.join(f'{label} {value:.3f} ms' for label, value in timings.items())})"
            )
        if entry["route"] != chain.default_route:
            drift = band(timings[entry["route"]], timings[chain.default_route])
            if drift <= margin:
                raise Problem(
                    f"batch {batch}: {entry['route']!r} wins by {drift:.1%}, inside the "
                    f"{margin:.0%} noise band, so the decision is not stable"
                )

    refusals = check_chain_withheld(chain, record, receipt)
    default_checked = bool(applicable)
    report = [
        f"withheld: {policy} may not adopt the measured route at batch "
        + ", ".join(str(batch) for batch in sorted(batches))
        for policy, batches in sorted(refusals.items())
    ]
    staged, fused = chain.candidates[0][0], chain.candidates[1][0]
    for row in receipt["batches"]:
        batch = row["batch"]
        if staged in row and fused in row:
            stage_a, stage_b = row.get(f"{staged}_schedule"), row.get(f"{fused}_schedule")
            if stage_a is not None and stage_a == stage_b:
                raise Problem(
                    f"batch {batch}: both arms dispatched {stage_b!r}, so the receipt records "
                    "one graph measured twice rather than a fusion A/B"
                )
        # A backward-fusion row may carry the folded leg and may not; the
        # two-dispatch pair is always there. `check_folded_evidence` has already
        # refused a row whose folded timing is not recordable.
        timings = {
            label: row[field] for label, field in chain.candidates if field in row
        }
        for label, value in timings.items():
            if not isinstance(value, (int, float)) or value <= 0.0:
                raise Problem(f"batch {batch}: the {label} candidate has timing {value!r}")
        for label in timings:
            if row.get(f"{label}_label") != label:
                raise Problem(
                    f"batch {batch}: the {label} leg reports {row.get(f'{label}_label')!r}"
                )
        worst = row.get("worst_max_abs")
        if worst != 0.0:
            raise Problem(
                f"batch {batch}: the chains diverge (worst |diff| {worst!r}); a route may "
                "only choose between bit-identical graphs"
            )
        labels = {tensor["label"] for tensor in row.get("tensors", [])}
        if labels != chain.tensors:
            raise Problem(
                f"batch {batch}: compared tensors {sorted(labels)} do not match "
                f"{sorted(chain.tensors)}"
            )
        for tensor in row["tensors"]:
            excess = tensor.get("worst_excess")
            if isinstance(excess, (int, float)) and excess > 0.0:
                raise Problem(
                    f"batch {batch}: {tensor['label']} left the exact-parity bound "
                    f"(worst excess {excess:.3e})"
                )
        for numerator, denominator, field in chain.ratios:
            if field not in row:
                raise Problem(f"batch {batch}: the receipt carries no {field}")
            if not close(row[field], timings[numerator] / timings[denominator]):
                raise Problem(
                    f"batch {batch}: {field} is not the measured {numerator}/{denominator} "
                    "ratio"
                )
        default_field = f"default_over_{chain.baseline}"
        if default_field not in row:
            raise Problem(f"batch {batch}: the receipt carries no {default_field}")
        if not close(row[default_field], row["default_ms"] / timings[chain.baseline]):
            raise Problem(
                f"batch {batch}: {default_field} is not the measured default/"
                f"{chain.baseline} ratio"
            )

        entry = lookup.get(batch)
        own = entry is not None and entry in measured_entries
        recorded = row.get("recorded_route")
        if entry is not None:
            if recorded is not None and recorded != entry["route"]:
                raise Problem(
                    f"batch {batch}: the receipt was measured against route {recorded!r} but "
                    f"the record now names {entry['route']!r}; rebuild and re-record the receipt"
                )
            if own:
                measured_winner = winner(timings, chain.route_order)
                if measured_winner != entry["route"]:
                    drift = band(timings[measured_winner], timings[entry["route"]])
                    if drift > margin:
                        raise Problem(
                            f"batch {batch}: the record names {entry['route']!r}, but this "
                            f"receipt measures {measured_winner!r} ahead by {drift:.1%}, "
                            f"outside the {margin:.0%} noise band; the record is stale"
                        )
            expected = entry["route"]
        else:
            if chain.default_from_measurement:
                expected = row.get("default_label")
                if expected not in chain.routes:
                    raise Problem(
                        f"batch {batch}: the default built {expected!r}, which is not one of "
                        f"the measured routes {sorted(chain.routes)}"
                    )
            else:
                expected = chain.default_route
                if default_checked and batch not in withheld_batches:
                    measured_winner = winner(timings, chain.route_order)
                    if measured_winner != chain.default_route and band(
                        timings[measured_winner], timings[chain.default_route]
                    ) > margin:
                        raise Problem(
                            f"batch {batch}: {measured_winner!r} is ahead of the structural "
                            f"default by "
                            f"{band(timings[measured_winner], timings[chain.default_route]):.1%}, "
                            f"outside the {margin:.0%} noise band, but the record carries no "
                            "decision for it"
                        )
        if recorded is not None and recorded != expected:
            raise Problem(
                f"batch {batch}: the default was recorded as {recorded!r}, expected {expected!r}; "
                "the receipt and the record were not measured together"
            )
        if row.get("default_label") != expected:
            raise Problem(
                f"batch {batch}: the default built {row.get('default_label')!r}, expected "
                f"{expected!r}"
            )
        if row.get("expected_default_label") != expected:
            raise Problem(
                f"batch {batch}: the receipt expected {row.get('expected_default_label')!r}, "
                f"expected {expected!r}"
            )
        shown = "  ".join(
            f"{label} {timings[label]:7.3f} ms"
            for label in chain.route_order
            if label in timings
        )
        report.append(f"batch {batch:>2}: {expected:<7} {shown}")
    return report


VENDOR_FAMILIES = ("linear-forward", "linear-residual-forward", "linear3-forward")
VENDOR_PREFIXES = {"intel": "intel-", "amd": "amd-"}
COMPOSITION_TENSORS = {
    "output",
    "packed_new_state",
    "grad_x",
    "grad_packed_state",
    "token_feature_grad",
}


def parse_pairs(line: str) -> dict[str, str]:
    pairs = {}
    for entry in line.split():
        key, _, value = entry.partition("=")
        pairs[key] = value
    return pairs


def device_matches_vendor(device: str, vendor: str) -> bool:
    """The composition receipt names a vendor, not an adapter.

    The route records are keyed by adapter, so the closest scope the receipt can
    support is every entry for the vendor it qualified. That is deliberately
    conservative: if two adapters of one vendor disagree at the receipt's
    geometry, this receipt cannot be evidence for both.
    """
    text = device.strip().lower()
    if vendor == "intel":
        return "intel" in text
    if vendor == "amd":
        return "amd" in text or "radeon" in text
    return False


def check_composition_consumer(
    consumer: CompositionConsumer,
    *,
    receipt_path: Path,
    records_dir: Path | None,
) -> list[str]:
    """Re-check the packed composition receipt against the routes it qualifies.

    The receipt runs one fixed geometry with both forward fusions disabled, so it
    exercises the `staged` candidate of the time-mix and channel-mix tables. At
    that width and batch the table may omit the geometry (the vendor gate decides)
    but it may not name the fusion: this receipt is not evidence for a graph the
    record does not ship there. Verdicts at other batches belong to the producer's
    own sweep and are not this receipt's stray entries.
    """
    if not receipt_path.exists():
        raise Problem(
            f"missing {receipt_path}; record it with verify_rwkv_packed_cell_composition.py"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("result") != "pass":
        raise Problem(
            f"the receipt's result is {receipt.get('result')!r}, so it measured nothing usable"
        )
    if receipt.get("negative_control") != "rejected":
        raise Problem(
            "the receipt's negative control did not reject, so its parity comparison does not "
            "discriminate"
        )
    steps = int(receipt.get("steps", 0))
    if steps < 1:
        raise Problem(f"the receipt records steps={receipt.get('steps')!r}")
    if receipt.get("comparisons") != steps * 5:
        raise Problem(
            f"the receipt compares {receipt.get('comparisons')!r} tensors, expected {steps * 5}"
        )
    atol = float(receipt.get("atol", 0.0))
    if atol <= 0.0 or float(receipt.get("worst_max_abs", float("inf"))) > atol:
        raise Problem(
            f"the receipt's worst |diff| is {receipt.get('worst_max_abs')!r} against atol "
            f"{receipt.get('atol')!r}"
        )
    for field in ("time_mix_forward_fusion", "channel_mix_forward_fusion"):
        if receipt.get(field) != "disabled":
            raise Problem(
                f"the receipt's {field} is {receipt.get(field)!r}; it did not exercise the "
                "staged route this check qualifies"
            )
    vendor = receipt.get("qualified_vendor")
    if vendor not in VENDOR_PREFIXES:
        raise Problem(f"the receipt qualifies vendor {vendor!r}, not intel or amd")
    variants = parse_pairs(receipt.get("vendor_plan", ""))
    for family in VENDOR_FAMILIES:
        if family not in variants:
            raise Problem(f"the vendor plan has no {family} entry")
        if variants[family] == "portable":
            raise Problem(
                f"{family} resolved the portable module; the composition check is vacuous"
            )
        if not variants[family].startswith(VENDOR_PREFIXES[vendor]):
            raise Problem(
                f"{family} resolved {variants[family]!r}, not an {vendor} module"
            )
    counts = parse_pairs(receipt.get("vendor_dispatch", ""))
    missing = [family for family in VENDOR_FAMILIES if int(counts.get(family, 0)) < 1]
    if missing:
        raise Problem(f"the vendor leg did not dispatch {missing} (counts {counts})")
    if parse_pairs(receipt.get("portable_dispatch", "")):
        raise Problem("the portable leg dispatched vendor modules, so this is not an A/B")
    width = receipt.get("width")
    batch = receipt.get("batch")
    if not isinstance(width, int) or not isinstance(batch, int):
        raise Problem(f"the receipt records width={width!r} batch={batch!r}")

    report = []
    for default_path, kind in consumer.records:
        record_path = (records_dir / default_path.name) if records_dir else default_path
        if not record_path.exists():
            raise Problem(f"missing {record_path}; the route record is not optional")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("kind") != kind or record.get("schema") != 2:
            raise Problem(
                f"{record_path.name} is {record.get('kind')!r} schema {record.get('schema')!r}, "
                f"expected {kind!r} schema 2"
            )
        matching = [
            entry
            for entry in record["decisions"]
            if entry.get("width") == width
            and entry.get("batch") == batch
            and entry.get("parameter_storage") is None
            and device_matches_vendor(entry.get("device", ""), vendor)
        ]
        for entry in matching:
            if entry.get("route") != "staged":
                raise Problem(
                    f"{kind} names {entry.get('route')!r} at batch {batch} for "
                    f"{entry.get('device')!r}, but this receipt qualifies the staged route"
                )
        if matching:
            devices = ", ".join(sorted({entry["device"] for entry in matching}))
            report.append(
                f"{kind}: batch {batch:>2} staged for {devices}"
            )
        else:
            report.append(
                f"{kind}: batch {batch:>2} has no {vendor} entry; the vendor gate decides"
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--producer",
        action="append",
        choices=sorted(ALL_SPECS_BY_NAME),
        help=(
            "receipt spec to check, producer or consumer (repeatable; default every "
            "measured producer and every consumer)"
        ),
    )
    parser.add_argument("--receipt", type=Path, help="override the receipt path (one spec only)")
    parser.add_argument("--record", type=Path, help="override the record path (one producer only)")
    parser.add_argument(
        "--records-dir",
        type=Path,
        help="directory holding the consumed records (one composition consumer only)",
    )
    parser.add_argument(
        "--sweep",
        type=int,
        action="append",
        dest="sweep",
        help="a batch the sweep must cover (repeatable; defaults to each producer's sweep)",
    )
    args = parser.parse_args(argv)

    names = args.producer or [spec.name for spec in ALL_SPECS]
    specs = [ALL_SPECS_BY_NAME[name] for name in names]
    overridden = (args.receipt, args.record, args.records_dir)
    if any(path is not None for path in overridden) and len(names) != 1:
        parser.error("--receipt/--record/--records-dir need exactly one --producer")
    if args.record is not None and any(
        isinstance(spec, CompositionConsumer) for spec in specs
    ):
        parser.error("--record does not apply to a composition consumer; use --records-dir")
    if args.records_dir is not None and any(
        not isinstance(spec, CompositionConsumer) for spec in specs
    ):
        parser.error("--records-dir applies only to a composition consumer")
    if args.sweep is not None and any(
        isinstance(spec, CompositionConsumer) for spec in specs
    ):
        parser.error("--sweep applies only to a receipt with per-geometry rows")
    if args.sweep is not None and any(batch <= 0 for batch in args.sweep):
        parser.error("--sweep values must be positive")

    failures = []
    checked = []
    for spec in specs:
        name = spec.name
        receipt_path = args.receipt or spec.receipt
        try:
            if isinstance(spec, ChainProducer):
                record_path = args.record or spec.record
                sweep = tuple(args.sweep or spec.sweep)
                report = check_chain_producer(
                    spec, receipt_path=receipt_path, record_path=record_path, sweep=sweep
                )
                receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
                device = receipt_doc["device"]
                for line in report:
                    print(f"{name}: {line}")
                coverage = f"batch {', '.join(str(batch) for batch in sweep)}"
                reported = set(receipt_doc.get("unmeasured", []))
                if reported:
                    coverage += (
                        " (batch "
                        + ", ".join(str(batch) for batch in sorted(reported))
                        + " reported unmeasured: this device compiles one topology there)"
                    )
                record_doc = json.loads(record_path.read_text(encoding="utf-8"))
                applicable = chain_entries(record_doc, receipt_doc)
                held = [
                    entry
                    for entry in record_doc.get("withheld", [])
                    if normalized(entry["device"]) == normalized(device)
                    and entry["width"] == receipt_doc["width"]
                ]
                if applicable:
                    message = (
                        f"PASS: {name} is measured across {coverage} on {device}, and its "
                        "record follows it"
                    )
                    if held:
                        policies = sorted(
                            {canonical_policy(entry["parameter_storage"]) for entry in held}
                        )
                        message += (
                            f"; {len(held)} geometry(ies) are withheld from "
                            + "/".join(policies)
                            + " because that policy may not adopt the measured route"
                        )
                    print(message)
                else:
                    print(
                        f"PASS: {name} covers {coverage} on {device}; this adapter has no "
                        "recorded route, so the structural default decides it"
                    )
                checked.append(name)
                continue
            if isinstance(spec, CompositionConsumer):
                report = check_composition_consumer(
                    spec, receipt_path=receipt_path, records_dir=args.records_dir
                )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                for line in report:
                    print(f"{name}: {line}")
                print(
                    f"PASS: {name} qualifies the staged route at width "
                    f"{receipt['width']} batch {receipt['batch']} for "
                    f"{receipt['qualified_vendor']}, and its records agree"
                )
                checked.append(name)
                continue

            record_path = args.record or spec.record
            sweep = tuple(args.sweep or spec.sweep)
            report = check_producer(
                spec, receipt_path=receipt_path, record_path=record_path, sweep=sweep
            )
            device = json.loads(receipt_path.read_text(encoding="utf-8"))["device"]
            for line in report:
                print(f"{name}: {line}")
            coverage = f"batch {', '.join(str(batch) for batch in sweep)}"
            record_doc = json.loads(record_path.read_text(encoding="utf-8"))
            receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
            lookup, applicable = entries_for(spec, record_doc, receipt_doc)
            held = withheld_for(spec, record_doc, receipt_doc)
            if applicable:
                message = (
                    f"PASS: {name} is measured across {coverage} on {device}, and its "
                    "record follows it"
                )
                if held:
                    message += (
                        f"; {len(held)} geometry(ies) are measured but withheld "
                        "from the runtime because their graphs are not bit-identical"
                    )
                print(message)
            else:
                print(
                    f"PASS: {name} covers {coverage} on {device}; this adapter has no "
                    "recorded route, so the fallback decides it"
                )
            checked.append(name)
        except Problem as problem:
            failures.append((name, str(problem)))
            print(f"FAIL: {name}: {problem}", file=sys.stderr)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            failures.append((name, f"malformed receipt or record: {error!r}"))
            print(f"FAIL: {name}: malformed receipt or record: {error!r}", file=sys.stderr)

    if failures:
        return 1
    print(
        f"PASS: {len(checked)} receipt(s) re-checked against the record(s) they consume: "
        + ", ".join(checked)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
