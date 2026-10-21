#!/usr/bin/env python3
"""Contract tests for the key+state forward A/B receipt (rank 3).

`bench_rwkv_key_state_forward.py` produces
`validation/key_state_forward_bench.json`: the wider-group fused
`rwkv_key_state_forward_fused_wide` dispatch, the narrow
`rwkv_key_state_forward_fused` it replaced, and the `key_transform_forward` +
`state_forward` chain, all measured in isolation so one dispatch against two is
visible rather than averaged into a whole time-mix step.

The tests check the decision rather than only the numbers:

* every graph returns the same tensors; on this adapter they are bit-identical,
  because the fused kernels and the two chain kernels evaluate the same products
  in the same order — the receipt's `worst_max_abs` and
  `worst_max_abs_wide_vs_narrow` are zero and this contract says so instead of
  hiding behind a tolerance;
* each leg really ran the graph it claims (arm label plus the measured dispatch
  count: one per fused arm against two for the chain);
* the default follows the recorded route *and* the recorded fused arm where this
  adapter has an entry, and the structural default (the fusion, wider group)
  where it has none;
* the wider group removed the batch-16 crossover: at the geometry where the
  narrow kernel loses to the chain, the wide kernel wins;
* the timings justify the record, including the geometry where the chain wins;
* the record and the receipt agree about which graph and which fused arm won at
  every recorded geometry.

The receipt is evidence, not a live run. Set `HIERARCHOS_RUN_GPU_VALIDATION=1` to
re-run the A/B on the attached adapter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RECEIPT = ROOT / "validation" / "key_state_forward_bench.json"
DECISIONS = ROOT / "validation" / "key_state_forward_decisions.json"
HARNESS = ROOT / "validation" / "bench_rwkv_key_state_forward.py"

TENSORS = {"new_state", "tmix", "scaled_k", "kk", "saved_sa"}
WIDE_ARM = "key-state-forward-fused-wide"
NARROW_ARM = "key-state-forward-fused"
# The order a tie breaks in, matching the record's vocabulary.
ROUTE_ORDER = ("fused", "wide-fused", "narrow-fused", "staged")
# The route an unrecorded geometry dispatches: the wider-group fusion.
DEFAULT_ROUTE = "wide-fused"
# The band inside which a measured sign is not a decision; the driver omits those
# geometries and the structural default keeps deciding them.
NOISE_BAND = 0.02
FUSED_DISPATCHES = 1
COMPOSED_DISPATCHES = 2


class KeyStateForwardReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RECEIPT.exists():
            raise AssertionError(
                f"missing {RECEIPT.relative_to(REPO)}; run "
                "validation/bench_rwkv_key_state_forward.py --record to record it"
            )
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        cls.batches = cls.receipt["batches"]
        cls.record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        cls.entries = {
            (entry["device"].strip().lower(), entry["width"], entry["batch"]): entry
            for entry in cls.record["decisions"]
        }

    def recorded(self, row: dict) -> dict | None:
        return self.entries.get(
            (self.receipt["device"].strip().lower(), self.receipt["width"], row["batch"])
        )

    def test_passed_with_a_real_sweep(self) -> None:
        self.assertEqual(self.receipt["result"], "pass")
        self.assertGreaterEqual(len(self.batches), 2)
        self.assertGreaterEqual(self.receipt["samples"], 1)
        self.assertGreaterEqual(self.receipt["repetitions"], 1)
        self.assertEqual(self.receipt["fused_arm"], WIDE_ARM)
        self.assertEqual(self.receipt["narrow_arm"], NARROW_ARM)
        self.assertEqual(
            self.receipt["composed_arm"], "key-transform-forward+state-forward"
        )
        self.assertIn("one key/state forward", self.receipt["unit"])
        self.assertIn("narrow_over_fused", self.receipt["arms_note"])
        self.assertIn("`narrow`", self.receipt["arm_rule"])
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertGreater(row["fused_ms"], 0.0)
                self.assertGreater(row["narrow_ms"], 0.0)
                self.assertGreater(row["composed_ms"], 0.0)
                self.assertEqual(row["fused_arm"], WIDE_ARM)
                self.assertEqual(row["narrow_arm"], NARROW_ARM)

    def test_the_two_graphs_are_bit_identical(self) -> None:
        """The fusion may not change a single value.

        Unlike the LN1/low-rank A/B, these two graphs do not re-associate any
        reduction: the fused kernel and `key_transform_forward` + `state_forward`
        compute the same products in the same order. Zero here is the claim; a
        non-zero value means one graph changed its arithmetic and this record
        cannot be allowed to choose between them as if it only changed timing.
        """
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual(
                    row["worst_max_abs"],
                    0.0,
                    "the fused and composed key/state graphs are no longer bit-identical",
                )
                self.assertEqual(
                    row["worst_max_abs_wide_vs_narrow"],
                    0.0,
                    "widening the workgroup changed a value, so it is no longer only a "
                    "scheduling change",
                )
                for tensor in row["tensors"]:
                    self.assertLessEqual(tensor["worst_excess"], 0.0)

    def test_legs_ran_different_graphs(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual(row["fused_label"], "fused")
                self.assertEqual(row["composed_label"], "composed")
                self.assertEqual(
                    row["fused_key_state_forward_dispatches"],
                    FUSED_DISPATCHES,
                    "the fused arm must be the single key/state dispatch",
                )
                self.assertEqual(
                    row["narrow_key_state_forward_dispatches"],
                    FUSED_DISPATCHES,
                    "the narrow fused arm must be one dispatch too",
                )
                self.assertEqual(
                    row["composed_key_state_forward_dispatches"],
                    COMPOSED_DISPATCHES,
                    "the composed arm is `key_transform_forward` + `state_forward`",
                )
                self.assertEqual(
                    row["default_key_state_forward_dispatches"],
                    COMPOSED_DISPATCHES
                    if row["default_label"] == "composed"
                    else FUSED_DISPATCHES,
                    "the default leg's dispatch count contradicts the graph it reports",
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
                route = entry["route"] if entry is not None else None
                if entry is None:
                    self.assertEqual(
                        row["default_label"],
                        row["fused_label"],
                        "an unrecorded geometry must keep the structural default: the fusion",
                    )
                else:
                    # Both fused arms report the fused label; the route says which
                    # kernel, and `default_arm` below checks that the default built
                    # exactly the one the route names.
                    self.assertEqual(
                        row["default_label"],
                        "composed" if route == "staged" else row["fused_label"],
                    )
                self.assertEqual(row["default_label"], row["expected_default_label"])
                # The record names one route, so the arm is that route rather than
                # a second field that could disagree with it.
                expected_arm = {
                    "staged": None,
                    "narrow-fused": NARROW_ARM,
                    "wide-fused": WIDE_ARM,
                    None: WIDE_ARM if row["default_label"] == "fused" else None,
                }[route]
                self.assertEqual(row["recorded_route"], route)
                self.assertEqual(row["default_arm"], expected_arm)

    def test_record_agrees_with_the_receipt(self) -> None:
        self.assertEqual(self.record["schema"], 2)
        self.assertEqual(self.record["kind"], "rwkv-key-state-forward")
        self.assertEqual(
            self.record["measured_by"],
            "validation/bench_rwkv_key_state_forward.py",
        )
        self.assertAlmostEqual(self.record["margin"], NOISE_BAND)
        device = self.receipt["device"].strip().lower()
        width = self.receipt["width"]
        rows = {row["batch"]: row for row in self.batches}
        for entry in self.record["decisions"]:
            self.assertEqual(entry["source"], "measured")
            self.assertTrue(entry["device"].strip())
            self.assertEqual(entry["candidates"][entry["route"]], min(entry["candidates"].values()))
            # This historical receipt measures one adapter. Other adapters keep
            # their own measurements and must not be compared to Intel timings.
            if entry["device"].strip().lower() != device:
                continue
            self.assertEqual(entry["width"], width)
            self.assertEqual(entry["probe"], "key-state-forward")
            self.assertIn(
                entry["route"],
                {"wide-fused", "narrow-fused", "staged"},
                "the key/state record may only name one of the three graphs it builds",
            )
            self.assertIn(
                entry["batch"],
                rows,
                "the record carries an entry for a geometry this receipt never measured",
            )
            row = rows[entry["batch"]]
            candidates = entry["candidates"]
            self.assertEqual(
                entry["route"],
                min(candidates, key=lambda label: (candidates[label], ROUTE_ORDER.index(label))),
                "the recorded route is not the fastest candidate it measured",
            )
            # The record is a snapshot of an earlier sweep and the receipt is a
            # rerun of the same legs, so the two agree on the measurement to within
            # run-to-run drift rather than to the last digit: the route and the
            # ordering are the claims, not a frozen microsecond.
            for label, value in (
                ("wide-fused", row["fused_ms"]),
                ("narrow-fused", row["narrow_ms"]),
                ("staged", row["composed_ms"]),
            ):
                self.assertLess(
                    abs(candidates[label] - value) / value,
                    0.15,
                    f"the recorded {label} timing and this run's disagree by more than drift",
                )
            if entry["route"] != "wide-fused":
                self.assertGreater(
                    abs(candidates["wide-fused"] / candidates[entry["route"]] - 1.0),
                    NOISE_BAND,
                    "a decision recorded inside the noise band is not a decision",
                )
            self.assertNotIn(
                "fused_arm",
                entry,
                "the arm is the route now; a separate fused_arm field is the flag pair the "
                "route replaced",
            )

    def test_geometries_inside_the_band_are_left_to_the_default(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                timings = {
                    "wide-fused": row["fused_ms"],
                    "narrow-fused": row["narrow_ms"],
                    "staged": row["composed_ms"],
                }
                best = min(timings, key=lambda label: (timings[label], ROUTE_ORDER.index(label)))
                # The default route winning is a measured confirmation, not a tie:
                # the record keeps it. A *different* winner inside the band is a
                # sign that is not stable, and that geometry must carry no entry.
                band = abs(timings[best] / timings["wide-fused"] - 1.0) <= NOISE_BAND
                if band:
                    entry = self.recorded(row)
                    if entry is not None:
                        self.assertEqual(
                            entry["route"],
                            "wide-fused",
                            "a non-default winner inside the noise band kept a recorded decision",
                        )
                    self.assertEqual(row["default_label"], row["fused_label"])

    def test_the_wider_group_removed_the_batch16_crossover(self) -> None:
        """The reason the wide kernel exists, stated as a checked claim.

        The narrow one-workgroup-per-head kernel lost to the two-dispatch chain at
        batch 16 on this adapter because only 16 of its 64 lanes advanced state
        rows, so its cost per element stayed fixed while the chain's scaled with
        rows. The wider group packs `64 / head_size` heads into the workgroup and
        keeps every lane busy, so this test requires the crossover to be gone: at
        every geometry where the narrow arm loses to the chain outside the band,
        the wide arm must win.
        """
        lost_with_narrow = []
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                narrow_loses = row["composed_over_narrow"] < 1.0 - NOISE_BAND
                wide_wins = row["composed_over_fused"] > 1.0
                if narrow_loses:
                    lost_with_narrow.append(row["batch"])
                    self.assertTrue(
                        wide_wins,
                        f"batch {row['batch']}: the narrow arm loses to the chain "
                        f"({row['composed_over_narrow']:.4f}) but the wider group does not win "
                        f"({row['composed_over_fused']:.4f}), so the crossover was not removed",
                    )
                self.assertAlmostEqual(
                    row["narrow_over_fused"] * row["composed_over_narrow"],
                    row["composed_over_fused"],
                    places=9,
                    msg="the wide/narrow/chain ratios are not one consistent measurement",
                )
        self.assertIn(
            16,
            lost_with_narrow,
            "the receipt no longer reproduces the narrow arm's batch-16 loss, so the "
            "crossover this kernel was built to remove is unmeasured",
        )

    def test_the_record_covers_every_decided_geometry(self) -> None:
        """A receipt that stopped covering geometries would hide a verdict.

        The writer records a geometry when the structural default wins (that is a
        measurement, not a tie) and when any other candidate wins outside the band,
        so every measured row must carry an entry whose route is its fastest
        candidate. A non-default winner inside the band is the one case that carries
        no entry, and that geometry is allowed to fall back to the default.
        """
        verdicts = set()
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                entry = self.recorded(row)
                timings = {
                    "wide-fused": row["fused_ms"],
                    "narrow-fused": row["narrow_ms"],
                    "staged": row["composed_ms"],
                }
                best = min(timings, key=lambda label: (timings[label], ROUTE_ORDER.index(label)))
                outside = abs(timings[best] / timings[DEFAULT_ROUTE] - 1.0) > NOISE_BAND
                if best != DEFAULT_ROUTE and not outside:
                    continue
                self.assertIsNotNone(
                    entry,
                    "a decided geometry must carry a recorded decision: the default winning "
                    "is recorded, and so is a winner outside the band",
                )
                self.assertEqual(entry["route"], best)
                verdicts.add(entry["route"])
        self.assertTrue(
            verdicts,
            "the record carries no decision at all, so the default is unmeasured here",
        )

    def test_comparisons_cover_state_and_derived_key(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual({entry["label"] for entry in row["tensors"]}, TENSORS)


class LiveKeyStateForwardBenchTest(unittest.TestCase):
    """Re-run the A/B, which fails on a promotion or parity regression."""

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU key+state forward A/B",
    )
    def test_shipped_route_holds_on_this_adapter(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--batch",
                "1",
                "--batch",
                "16",
                "--iterations",
                "10",
                "--repetitions",
                "20",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
