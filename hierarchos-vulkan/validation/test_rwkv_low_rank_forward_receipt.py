#!/usr/bin/env python3
"""Contract tests for the packed-cell LN1/low-rank forward A/B receipt.

`bench_rwkv_packed_cell_low_rank_forward.py` produces
`validation/packed_cell_low_rank_forward_bench.json`, the rank-2 measurement from
`FUSED_FORWARD_PRODUCER_AUDIT.md`: the single fused LN1/low-rank producer against
the staged chain it replaces (a standalone layer norm, then the interpolation, the
three first-stage matmuls and the second-stage matmuls on their own dispatches).

The receipt is what makes the shipped default auditable, so these tests check the
decision rather than only the numbers:

* both graphs return the same tensors within the stated tolerance — unlike the
  channel-mix A/B these two are not bit-identical, because the fused arm
  normalizes inside the producer while the staged arm runs the standalone
  layer-norm kernel, so this test states the tolerance instead of claiming
  exactness;
* a staged win whose graphs differ is recorded as evidence and withheld from
  adoption, because the exact training trajectory cannot follow it and the TBPTT
  packed forward-only leg keeps its own fused chain;
* each leg really ran the graph it claims (label plus the route's own dispatch
  count: one producer dispatch against the chain);
* the default follows the recorded decision where this adapter has an entry, and
  the structural default (the fusion) where it has none;
* the timings justify the split, and the geometries left inside the noise band
  are the ones the structural default keeps deciding.

The receipt is evidence, not a live run. Set `HIERARCHOS_RUN_GPU_VALIDATION=1`
to re-run the A/B on the attached adapter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RECEIPT = ROOT / "validation" / "packed_cell_low_rank_forward_bench.json"
DECISIONS = ROOT / "validation" / "low_rank_full_forward_decisions.json"
HARNESS = ROOT / "validation" / "bench_rwkv_packed_cell_low_rank_forward.py"

sys.path.insert(0, str(ROOT / "validation"))
import ab_harness as ab  # noqa: E402
from bench_rwkv_packed_cell_low_rank_forward import format_decision

TENSORS = {"output", "packed_new_state", "grad_x", "grad_packed_state", "token_feature_grad"}
# The band inside which a measured sign is not a decision; the driver omits those
# geometries and the structural default keeps deciding them.
NOISE_BAND = 0.02
# The staged chain cannot be fewer than the layer norm, the interpolation and the
# first stage; the fused arm is exactly one dispatch.
MIN_CHAIN_DISPATCHES = 6


class LowRankForwardReceiptTest(unittest.TestCase):
    def test_mixed_storage_decisions_print_from_canonical_schema_without_legacy_ratio(self):
        entries = json.loads(DECISIONS.read_text())["decisions"]
        for entry in entries:
            with self.subTest(device=entry['device'], policy=entry.get('parameter_storage')):
                text = format_decision(entry)
                self.assertIn(entry['device'], text)
                self.assertIn('ratio ', text)
        normalized = dict(device='AMD test adapter', width=128, batch=8, route='staged',
                          parameter_storage='fp16-storage', candidates={'fused':4.0,'staged':3.0})
        self.assertNotIn('composed_over_fused', normalized)
        self.assertIn('ratio 0.7500, storage fp16-storage', format_decision(normalized))

    @classmethod
    def setUpClass(cls) -> None:
        if not RECEIPT.exists():
            raise AssertionError(
                f"missing {RECEIPT.relative_to(REPO)}; run "
                "validation/bench_rwkv_packed_cell_low_rank_forward.py --record to record it"
            )
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        cls.batches = cls.receipt["batches"]
        cls.record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        # This receipt measures the FP32 producer, so it reads the policy-less
        # entries: the policy-scoped ones (the FP16 packed-mirror route) are a
        # different graph's measurement and must not decide this one.
        cls.entries = {
            (entry["device"].strip().lower(), entry["width"], entry["batch"]): entry
            for entry in cls.record["decisions"]
            if "parameter_storage" not in entry
        }

    def recorded(self, row: dict) -> dict | None:
        return self.entries.get(
            (self.receipt["device"].strip().lower(), self.receipt["width"], row["batch"])
        )

    def test_passed_within_gate(self) -> None:
        self.assertEqual(self.receipt["result"], "pass")
        self.assertGreaterEqual(len(self.batches), 2)
        self.assertGreater(self.receipt["atol"], 0.0)
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertLessEqual(row["worst_max_abs"], self.receipt["atol"])

    def test_the_two_graphs_agree_within_the_stated_tolerance(self) -> None:
        """A route choice may not change the numbers beyond rounding.

        The fused arm computes the layer norm inside the producer and the staged
        arm dispatches the standalone layer-norm kernel, so the two reductions
        round differently and the tensors are not bit-identical. The contract is
        therefore the stated tolerance, and a step out of it is a numerics change
        that this record must not be allowed to promote across.
        """
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertLessEqual(row["worst_max_abs"], self.receipt["atol"])
                for tensor in row["tensors"]:
                    self.assertLessEqual(
                        tensor["worst_excess"],
                        0.0,
                        f"{tensor['label']} left the atol+rtol band",
                    )

    def test_legs_ran_different_graphs(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual(row["fused_label"], "fused")
                self.assertEqual(row["composed_label"], "composed")
                self.assertEqual(
                    row["fused_low_rank_forward_dispatches"],
                    1,
                    "the fused arm must be the single producer dispatch",
                )
                self.assertGreaterEqual(
                    row["composed_low_rank_forward_dispatches"], MIN_CHAIN_DISPATCHES
                )
                self.assertGreater(
                    row["composed_low_rank_forward_dispatches"],
                    row["fused_low_rank_forward_dispatches"],
                    "the fusion did not replace any dispatch, so the A/B is vacuous",
                )

    def test_default_follows_the_decision_it_should(self) -> None:
        self.assertEqual(
            self.receipt["promotion"],
            "geometry-aware",
            "the receipt no longer shows the default following the decision record",
        )
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                entry = self.recorded(row)
                if entry is None:
                    self.assertEqual(
                        row["default_label"],
                        row["fused_label"],
                        "an unrecorded geometry must keep the structural default: the fusion",
                    )
                else:
                    self.assertEqual(
                        row["default_label"],
                        "composed" if entry["route"] == "staged" else row["fused_label"],
                    )
                self.assertEqual(row["default_label"], row["expected_default_label"])
                self.assertEqual(
                    row["default_low_rank_forward_dispatches"],
                    row["composed_low_rank_forward_dispatches"]
                    if row["default_label"] == "composed"
                    else row["fused_low_rank_forward_dispatches"],
                    "the default leg's dispatch count contradicts the graph it reports",
                )

    def test_record_agrees_with_the_receipt(self) -> None:
        """Every recorded entry must match the receipt's own measurement."""
        self.assertEqual(self.record["schema"], 2)
        self.assertEqual(self.record["kind"], "rwkv-low-rank-full-forward")
        self.assertEqual(self.record["measured_by"], "validation/bench_rwkv_packed_cell_low_rank_forward.py")
        device = self.receipt["device"].strip().lower()
        width = self.receipt["width"]
        rows = {row["batch"]: row for row in self.batches}
        for entry in self.record["decisions"]:
            if "parameter_storage" in entry:
                # A policy-scoped entry is another storage policy's measurement:
                # this receipt measures the FP32 producer, and comparing an FP16
                # verdict against it would report a disagreement between two
                # graphs neither run timed.
                continue
            self.assertEqual(entry["source"], "measured")
            self.assertEqual(entry["device"].strip().lower(), device)
            self.assertEqual(entry["width"], width)
            self.assertEqual(entry["probe"], "low-rank-forward")
            self.assertIn(
                entry["batch"],
                rows,
                "the record carries an entry for a geometry this receipt never measured",
            )
            self.assertEqual(
                entry["route"] == "staged",
                (entry["candidates"]["staged"] / entry["candidates"]["fused"]) < 1.0,
                "the recorded decision contradicts its own ratio",
            )
            self.assertGreater(
                abs((entry["candidates"]["staged"] / entry["candidates"]["fused"]) - 1.0),
                NOISE_BAND,
                "a decision recorded inside the noise band is not a decision",
            )
            row = rows[entry["batch"]]
            self.assertEqual(
                entry["route"] == "staged",
                row["composed_over_fused"] < 1.0,
                "the record and the receipt disagree about which graph won",
            )

    def test_geometries_inside_the_band_are_left_to_the_default(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                band = abs(row["composed_over_fused"] - 1.0) <= NOISE_BAND
                if band:
                    self.assertIsNone(
                        self.recorded(row),
                        "a geometry measured inside the noise band kept a recorded decision",
                    )
                    self.assertEqual(
                        row["default_label"],
                        row["fused_label"],
                        "a tie keeps the structural default, and the default must follow it",
                    )

    def test_timings_justify_the_shipped_route(self) -> None:
        """The record's verdict, checked against the receipt's own numbers.

        Rank 2 came out the other way round from rank 1: the fusion holds at every
        geometry this adapter measured, and the staged chain never wins by more
        than the noise band. If a later run measures a staged win, this test keeps
        passing (a staged entry would appear in the record); what it refuses is a
        record that promotes the chain where the timing says otherwise.
        """
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                entry = self.recorded(row)
                if entry is None:
                    self.assertLessEqual(
                        abs(row["composed_over_fused"] - 1.0),
                        NOISE_BAND,
                        "a geometry outside the band must carry a decision",
                    )
                else:
                    self.assertEqual(
                        entry["route"] == "staged", row["composed_over_fused"] < 1.0
                    )
                    if not entry["route"] == "staged":
                        self.assertGreaterEqual(
                            row["composed_over_fused"],
                            1.0,
                            "the record keeps the fusion where the chain measured faster",
                        )

    def test_comparisons_cover_state_and_gradients(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual({entry["label"] for entry in row["tensors"]}, TENSORS)


FP16_RECEIPT = ROOT / "validation" / "low_rank_fp16_forward_bench.json"
FP16_HARNESS = ROOT / "validation" / "bench_rwkv_low_rank_fp16_forward.py"
FP16_TENSORS = {
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
DEEP_ARM = "low-rank-full-forward-fused-fp16-packed"
FP16_POLICY = "fp16-storage-fp32-compute"


class LowRankFp16ForwardReceiptTest(unittest.TestCase):
    """The rank-2 scope gap: the deep FP16 arm under the FP16-storage policy.

    The packed-cell receipt above measures the FP32 arm, because the packed cell
    has no FP16 parameter mirrors. This receipt measures the arm the training
    graph's `fp16-storage-fp32-compute` policy actually installs, on the op that
    policy feeds, and answers whether it follows the same per-geometry route.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not FP16_RECEIPT.exists():
            raise AssertionError(
                f"missing {FP16_RECEIPT.relative_to(REPO)}; run "
                "validation/bench_rwkv_low_rank_fp16_forward.py --record to record it"
            )
        cls.receipt = json.loads(FP16_RECEIPT.read_text(encoding="utf-8"))
        cls.batches = cls.receipt["batches"]
        cls.record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        cls.scoped = {
            (entry["device"].strip().lower(), entry["width"], entry["batch"]): entry
            for entry in cls.record["decisions"]
            if entry.get("parameter_storage") == "fp16-storage"
        }
        cls.shared = {
            (entry["device"].strip().lower(), entry["width"], entry["batch"]): entry
            for entry in cls.record["decisions"]
            if "parameter_storage" not in entry
        }
        cls.withheld = {
            (entry["device"].strip().lower(), entry["width"], entry["batch"]): entry
            for entry in cls.record.get("withheld", [])
            if ab.canonical_policy(entry.get("parameter_storage")) == "fp16-storage"
        }
        cls.routes = {route["batch"]: route for route in cls.receipt["route_following"]}

    def skey(self, row: dict) -> tuple:
        return (self.receipt["device"].strip().lower(), self.receipt["width"], row["batch"])

    def recorded(self, row: dict) -> dict | None:
        """The entry the FP16 route adopts: its own policy, else the shared one.

        A withheld route is measured evidence, not an adopted decision: the
        runtime skips it and keeps the structural fused graph.
        """
        if self.skey(row) in self.withheld:
            return None
        return self.scoped.get(self.skey(row)) or self.shared.get(self.skey(row))

    def test_measured_the_deep_fp16_arm_not_a_proxy(self) -> None:
        """The policy, the storage and the arm name are the receipt's subject."""
        self.assertEqual(self.receipt["result"], "pass")
        self.assertEqual(self.receipt["precision_policy"], FP16_POLICY)
        self.assertEqual(self.receipt["parameter_storage"], "fp16-storage")
        self.assertEqual(self.receipt["deep_arm"], DEEP_ARM)
        self.assertEqual(
            self.receipt["fused_arm"],
            DEEP_ARM,
            "the fused leg did not run the deep FP16 producer this gap is about",
        )
        self.assertGreaterEqual(len(self.batches), 2)
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual(row["fused_label"], "fused")
                self.assertEqual(row["composed_label"], "composed")
                self.assertEqual(
                    row["fused_low_rank_forward_dispatches"],
                    1,
                    "the fused arm must be the single producer dispatch",
                )
                self.assertGreaterEqual(
                    row["composed_low_rank_forward_dispatches"], MIN_CHAIN_DISPATCHES
                )
                self.assertLessEqual(row["worst_max_abs"], self.receipt["atol"])
                for tensor in row["tensors"]:
                    self.assertLessEqual(tensor["worst_excess"], 0.0)

    def test_default_follows_the_same_record(self) -> None:
        self.assertEqual(self.receipt["promotion"], "geometry-aware")
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                entry = self.recorded(row)
                expected = "composed" if entry and entry["route"] == "staged" else row["fused_label"]
                self.assertEqual(row["default_label"], expected)
                self.assertEqual(row["default_label"], row["expected_default_label"])
                self.assertEqual(
                    row["default_low_rank_forward_dispatches"],
                    row["composed_low_rank_forward_dispatches"]
                    if row["default_label"] == "composed"
                    else row["fused_low_rank_forward_dispatches"],
                )

    def test_route_answer_is_explicit_for_every_geometry(self) -> None:
        """The gap's question answered per geometry, not inferred from a switch.

        Where the dense route record carries an entry, the FP16 measurement must
        agree with it: that is the claim "the same route decision gates the deep
        arm". Where it has none, the row still has to say which graph the shipped
        default builds and which one this run measured, and whether they diverge —
        a diverging row is the finding that the two storage policies crossover
        differently, and it must be visible in the receipt rather than averaged
        away.
        """
        self.assertEqual(set(self.routes), {row["batch"] for row in self.batches})
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                route = self.routes[row["batch"]]
                self.assertEqual(route["fused_arm"], DEEP_ARM)
                entry = self.recorded(row)
                self.assertEqual(
                    route["recorded_route"],
                    None if entry is None else entry["route"],
                )
                shipped = "staged" if entry and entry["route"] == "staged" else "fused"
                self.assertEqual(route["shipped_route"], shipped)
                measured = "staged" if row["composed_over_fused"] < 1.0 else "fused"
                self.assertEqual(route["measured_route"], measured)
                if entry is not None:
                    # A rerun can land inside the 2% band the record's entries are
                    # gated by (batch 16 sits right at it); a tie is consistent with
                    # the recorded route, a divergence is not. A withheld row is
                    # neither: its recorded win is admitted to be non-adoptable.
                    self.assertIn(
                        route["route_status"],
                        {"agrees", "tie", "withheld"},
                        "the FP16 arm diverges from the recorded route at a geometry the "
                        "record covers; the audit's scope-gap claim needs re-deriving, not a "
                        "quiet edit to this test",
                    )
                if route["route_status"] == "withheld":
                    self.assertIn(
                        self.skey(row),
                        self.withheld,
                        "the receipt says this win was withheld but the record no longer "
                        "withholds it: the two artifacts were written by different runs",
                    )
                if self.skey(row) in self.withheld:
                    self.assertEqual(route["route_status"], "withheld")
                    self.assertEqual(
                        route["withheld_route"], self.withheld[self.skey(row)]["route"]
                    )
                    self.assertTrue(route["withheld_reason"])
        self.assertEqual(
            self.receipt["follows_recorded_route_everywhere"],
            all(route["route_status"] != "diverges" for route in self.routes.values()),
        )
        self.assertTrue(
            self.receipt["follows_recorded_route_everywhere"],
            "the FP16 route stopped following its own policy-scoped record: the decision "
            "records are keyed by parameter-storage policy, so a staged win measured here "
            "must be dispatched here",
        )

    def test_fp16_verdicts_are_policy_scoped_and_leave_the_fp32_route_alone(self) -> None:
        """The dimension that lets the FP16 route take its own staged wins.

        This receipt measures the deep FP16 producer, whose crossover is earlier
        than the FP32 arm's. Its verdicts therefore live in entries scoped to
        `fp16-storage`, which only a consumer running that storage policy
        reads; the policy-less entries stay the dense camera's measurements, each
        tagged with the driver that produced it. Recording here must not rewrite a
        shared entry, or an FP16 measurement would move the FP32 route.
        """
        self.assertEqual(
            self.record["measured_by"],
            "validation/bench_rwkv_packed_cell_low_rank_forward.py",
            "the policy-less default entries were not written by the dense driver",
        )
        for entry in self.record["decisions"]:
            self.assertEqual(entry["probe"], "low-rank-forward")
            policy = entry.get("parameter_storage")
            expected_driver = (
                "validation/bench_rwkv_low_rank_fp16_forward.py"
                if policy == "fp16-storage"
                else "validation/bench_rwkv_packed_cell_low_rank_forward.py"
            )
            self.assertEqual(
                entry.get("measured_by", self.record["measured_by"]),
                expected_driver,
                "an entry does not name the driver whose measurement it carries",
            )
        self.assertTrue(
            self.scoped,
            "the record carries no FP16-scoped entry, so this receipt's verdicts ship nowhere",
        )
        # The policies really do crossover differently: a geometry the FP16 route
        # takes the chain at, where the dense route keeps the fusion.
        fps = [
            key
            for key, entry in self.scoped.items()
            if entry["route"] == "staged"
            and (key not in self.shared or self.shared[key]["route"] != "staged")
        ]
        self.assertTrue(
            fps,
            "no geometry has the FP16 route composing while the dense route keeps the fusion "
            "(by its own entry or by the structural default), which is the difference this "
            "policy dimension exists for",
        )
        self.assertTrue(
            self.withheld,
            "the non-bit-identical FP16 staged wins are not recorded as withheld: the "
            "training trajectory cannot adopt them, and dropping the evidence would hide why",
        )
        for key, entry in self.withheld.items():
            self.assertIn(key, self.scoped, "a withheld route without a measured entry")
            self.assertEqual(entry["route"], self.scoped[key]["route"])
            self.assertGreater(
                self.scoped[key]["candidates"]["staged"]
                / self.scoped[key]["candidates"]["fused"],
                0.0,
                "a withheld staged route must still be the measured one",
            )
            self.assertTrue(entry["reason"].strip())
        for key in fps:
            self.assertGreater(
                abs(
                    self.scoped[key]["candidates"]["staged"]
                    / self.scoped[key]["candidates"]["fused"]
                    - 1.0
                ),
                NOISE_BAND,
                "a policy-scoped decision recorded inside the noise band is not a decision",
            )

    def test_comparisons_cover_the_shared_parameter_gradients(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual({entry["label"] for entry in row["tensors"]}, FP16_TENSORS)


class WithheldRouteAdoptionTest(unittest.TestCase):
    """A rerun that closes the rounding gap adopts its own measured win.

    The refusal is a fact about the measurement, not a permanent verdict: the
    driver only withholds a staged win whose graphs differ, so a bit-identical
    rerun records the route and the merge drops the older `withheld` entry.
    """

    RUN = {
        "device": "GPU",
        "width": 128,
        "batch": 8,
        "worst_max_abs": 0.0,
        "fused": {"timing": {"median_ms": 1.0}},
        "composed": {"timing": {"median_ms": 0.9}},
    }

    def merge(self, worst: float) -> dict:
        sys.path.insert(0, str(ROOT / "validation"))
        import bench_rwkv_low_rank_fp16_forward as driver  # noqa: E402  (torch fixture)

        run = json.loads(json.dumps(self.RUN))
        run["worst_max_abs"] = worst
        with tempfile.TemporaryDirectory(prefix="fp16-adopt-") as scratch:
            path = Path(scratch) / "record.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "kind": "rwkv-low-rank-full-forward",
                        "measured_by": "validation/bench_rwkv_low_rank_fp16_forward.py",
                        "margin": 0.02,
                        "decisions": [
                            {
                                "device": "GPU",
                                "width": 128,
                                "batch": 8,
                                "route": "staged",
                                "candidates": {"fused": 1.0, "staged": 0.9},
                                "probe": "low-rank-forward",
                                "source": "measured",
                                "parameter_storage": "fp16-storage",
                            }
                        ],
                        "withheld": [
                            {
                                "device": "GPU",
                                "width": 128,
                                "batch": 8,
                                "route": "staged",
                                "reason": "the graphs differ by 2.384e-07",
                                "parameter_storage": "fp16-storage",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            merged, _ = ab.record_decisions(
                path,
                [run],
                kind="rwkv-low-rank-full-forward",
                probe_key="low-rank-forward",
                margin=0.01,
                probe_label="low-rank-forward",
                parameter_storage="fp16-storage",
                withhold_when=driver.withheld_reason,
            )
            return merged

    def test_a_bit_identical_rerun_drops_the_withheld_entry(self) -> None:
        merged = self.merge(worst=0.0)
        self.assertEqual(
            merged["withheld"],
            [],
            "a bit-identical staged win has nothing to refuse, so its refusal is dropped",
        )
        scoped = [
            entry
            for entry in merged["decisions"]
            if entry.get("parameter_storage") == "fp16-storage"
        ]
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0]["route"], "staged")

    def test_a_rerun_with_a_rounding_gap_keeps_the_refusal(self) -> None:
        merged = self.merge(worst=2.384185791015625e-07)
        self.assertEqual(len(merged["withheld"]), 1)
        self.assertEqual(merged["withheld"][0]["batch"], 8)
        self.assertEqual(merged["withheld"][0]["route"], "staged")
        self.assertTrue(merged["withheld"][0]["reason"].strip())


class DenseRouteWithholdTest(unittest.TestCase):
    """A dense staged win that changes bits is measured, not adopted.

    The FP16 arm has its own policy-scoped refusal; this one covers the
    policy-less dense route. The fused producer normalizes inside the kernel and
    the staged chain runs a separate layer norm, so their reductions can round
    differently. Adopting such a win moves the dense forward of a trajectory the
    exact gates hold bit-exact, and stops the TBPTT packed forward-only leg
    (which keeps its own fused chain) from matching the dense leg.
    """

    RUN = {
        "device": "GPU",
        "width": 128,
        "batch": 8,
        "worst_max_abs": 0.0,
        "composed_over_fused": 0.9,
        "fused": {"timing": {"median_ms": 1.0}},
        "composed": {"timing": {"median_ms": 0.9}},
    }

    def merge(self, worst: float, ratio: float = 0.9) -> dict:
        sys.path.insert(0, str(ROOT / "validation"))
        import bench_rwkv_packed_cell_low_rank_forward as driver  # noqa: E402

        run = json.loads(json.dumps(self.RUN))
        run["worst_max_abs"] = worst
        run["composed_over_fused"] = ratio
        with tempfile.TemporaryDirectory(prefix="dense-adopt-") as scratch:
            path = Path(scratch) / "record.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "kind": "rwkv-low-rank-full-forward",
                        "measured_by": "validation/bench_rwkv_packed_cell_low_rank_forward.py",
                        "margin": 0.02,
                        "decisions": [
                            {
                                "device": "GPU",
                                "width": 128,
                                "batch": 8,
                                "route": "staged",
                                "candidates": {"fused": 1.0, "staged": 0.9},
                                "probe": "low-rank-forward",
                                "source": "measured",
                            }
                        ],
                        "withheld": [
                            {
                                "device": "GPU",
                                "width": 128,
                                "batch": 8,
                                "route": "staged",
                                "reason": "the graphs differ by 2.384e-07",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            merged, _ = ab.record_decisions(
                path,
                [run],
                kind="rwkv-low-rank-full-forward",
                probe_key="low-rank-forward",
                margin=0.01,
                probe_label="low-rank-forward",
                withhold_when=driver.withheld_reason,
            )
            return merged

    def test_a_bit_identical_win_is_adopted(self) -> None:
        merged = self.merge(worst=0.0)
        self.assertEqual(
            merged["withheld"],
            [],
            "a bit-identical staged win has nothing to refuse, so its refusal is dropped",
        )
        self.assertEqual([entry["route"] for entry in merged["decisions"]], ["staged"])

    def test_a_rounding_gap_keeps_the_refusal(self) -> None:
        merged = self.merge(worst=2.086162567138672e-07)
        self.assertEqual(len(merged["withheld"]), 1)
        self.assertEqual(merged["withheld"][0]["route"], "staged")
        self.assertTrue(merged["withheld"][0]["reason"].strip())
        self.assertEqual(
            merged["decisions"][0]["route"],
            "staged",
            "the measured win stays in the record as evidence",
        )

    def test_a_slower_staged_arm_is_not_withheld(self) -> None:
        merged = self.merge(worst=2.086162567138672e-07, ratio=1.1)
        self.assertEqual(merged["withheld"], [])


class LiveLowRankFp16ForwardBenchTest(unittest.TestCase):
    """Re-run the FP16-storage A/B on the attached adapter."""

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU FP16 low-rank forward A/B",
    )
    def test_fp16_arm_still_follows_the_record(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(FP16_HARNESS),
                "--batch",
                "1",
                "--batch",
                "4",
                "--iterations",
                "8",
                "--warmup",
                "3",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


class LiveLowRankForwardBenchTest(unittest.TestCase):
    """Re-run the A/B, which fails on a promotion or parity regression."""

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU low-rank forward A/B",
    )
    def test_shipped_route_holds_on_this_adapter(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--batch",
                "4",
                "--batch",
                "8",
                "--iterations",
                "12",
                "--warmup",
                "4",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
