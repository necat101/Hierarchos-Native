#!/usr/bin/env python3
"""Contract tests for the packed-cell channel-mix A/B receipt.

`bench_rwkv_packed_cell_channel_mix.py` produces
`validation/packed_cell_channel_mix_bench.json`, the rank-1 measurement from
`FUSED_FORWARD_PRODUCER_AUDIT.md`: the single fused channel-mix producer against
the staged composition that dispatches the tuned `linear-residual-forward` module
for `value_cm`.

The receipt is what makes the shipped default auditable, so these tests check the
decision rather than only the numbers:

* both graphs return bit-identical tensors (the promotion is scheduling, not
  precision — a tolerance-based result would need a numerics policy instead of a
  graph choice);
* each leg really ran the graph it claims (label plus vendor dispatch count);
* the default follows the recorded decision where this adapter has an entry, and
  the vendor gate otherwise;
* the timings justify the split, and the geometries left inside the noise band
  are the ones the gate keeps.

The receipt is evidence, not a live run. Set `HIERARCHOS_RUN_GPU_VALIDATION=1`
to re-run the A/B on the attached adapter.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RECEIPT = ROOT / "validation" / "packed_cell_channel_mix_bench.json"
DECISIONS = ROOT / "validation" / "channel_mix_forward_decisions.json"
HARNESS = ROOT / "validation" / "bench_rwkv_packed_cell_channel_mix.py"

TENSORS = {"output", "packed_new_state", "grad_x", "grad_packed_state", "token_feature_grad"}
# The band inside which a measured sign is not a decision; the driver omits those
# geometries and the vendor gate keeps deciding them.
NOISE_BAND = 0.02


class ChannelMixReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RECEIPT.exists():
            raise AssertionError(
                f"missing {RECEIPT.relative_to(REPO)}; run "
                "validation/bench_rwkv_packed_cell_channel_mix.py --record to record it"
            )
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        cls.batches = cls.receipt["batches"]
        cls.record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        cls.entries = {
            (entry["device"].strip().lower(), entry["width"], entry["batch"]): entry
            for entry in cls.record["decisions"]
        }

    def vendor_batches(self) -> list[dict]:
        return [row for row in self.batches if row["composed_ran_vendor_linear_residual"]]

    def portable_batches(self) -> list[dict]:
        return [row for row in self.batches if not row["composed_ran_vendor_linear_residual"]]

    def recorded(self, row: dict) -> dict | None:
        return self.entries.get(
            (self.receipt["device"].strip().lower(), self.receipt["width"], row["batch"])
        )

    def expected_default(self, row: dict) -> str:
        entry = self.recorded(row)
        if entry is not None:
            return "composed" if entry["route"] == "staged" else row["fused_label"]
        if row["composed_ran_vendor_linear_residual"]:
            return "composed"
        return row["fused_label"]

    def test_passed_within_gate(self) -> None:
        self.assertEqual(self.receipt["result"], "pass")
        self.assertGreaterEqual(len(self.batches), 2)
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertLessEqual(row["worst_max_abs"], self.receipt["atol"])

    def test_the_two_graphs_are_bit_identical(self) -> None:
        """A graph choice may not change the numbers.

        Every returned tensor, including both gradients, must be exactly equal:
        the fused producer and the composition keep the same arithmetic order, so
        switching graphs is scheduling. If this ever stops holding, the decision
        needs a numerics policy, not a record entry.
        """
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual(
                    row["worst_max_abs"],
                    0.0,
                    "the fused and composed graphs no longer agree exactly; do not promote "
                    "a schedule change across a numerics change",
                )

    def test_legs_ran_different_graphs(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual(row["fused_label"], "fused")
                self.assertEqual(row["composed_label"], "composed")
                self.assertEqual(
                    row["fused_linear_residual_dispatches"],
                    0,
                    "the forced-fusion leg reached the vendor linear-residual module",
                )
                if row["composed_ran_vendor_linear_residual"]:
                    self.assertEqual(
                        row["composed_linear_residual_dispatches"], 1
                    )

    def test_default_follows_the_decision_it_should(self) -> None:
        self.assertEqual(
            self.receipt["promotion"],
            "geometry-aware",
            "the receipt no longer shows the default following the decision record",
        )
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual(row["default_label"], self.expected_default(row))

    def test_record_agrees_with_the_receipt(self) -> None:
        """Every recorded entry must match the receipt's own measurement."""
        self.assertEqual(self.record["schema"], 2)
        self.assertEqual(self.record["kind"], "rwkv-channel-mix-forward")
        device = self.receipt["device"].strip().lower()
        width = self.receipt["width"]
        for entry in self.record["decisions"]:
            self.assertEqual(entry["source"], "measured")
            self.assertTrue(entry["device"].strip())
            self.assertEqual(entry["candidates"][entry["route"]], min(entry["candidates"].values()))
            # Compare a receipt only with the adapter it actually measured.
            if entry["device"].strip().lower() != device:
                continue
            self.assertEqual(entry["width"], width)
            self.assertEqual(
                entry["route"] == "staged",
                (entry["candidates"]["staged"] / entry["candidates"]["fused"]) < 1.0,
                "the recorded decision contradicts its own ratio",
            )
            row = next(row for row in self.batches if row["batch"] == entry["batch"])
            self.assertGreater(
                abs((entry["candidates"]["staged"] / entry["candidates"]["fused"]) - 1.0),
                NOISE_BAND,
                "a decision recorded inside the noise band is not a decision",
            )
            self.assertEqual(
                entry["route"] == "staged",
                row["composed_over_fused"] < 1.0,
                "the record and the receipt disagree about which graph won",
            )

    def test_geometries_inside_the_band_are_left_to_the_gate(self) -> None:
        for row in self.portable_batches():
            with self.subTest(batch=row["batch"]):
                band = abs(row["composed_over_fused"] - 1.0) <= NOISE_BAND
                if band:
                    self.assertIsNone(
                        self.recorded(row),
                        "a geometry measured inside the noise band kept a recorded decision",
                    )
                    self.assertTrue(
                        row["default_label"].startswith("fused"),
                        "the gate keeps the fused graph where the composition has no tuned "
                        "module, and the default must follow it",
                    )

    def test_timings_justify_the_promotion(self) -> None:
        vendor = self.vendor_batches()
        self.assertTrue(vendor, "no batch ran the tuned module; the A/B is vacuous")
        decided = [
            row
            for row in vendor
            if abs(row["composed_over_fused"] - 1.0) > NOISE_BAND
        ]
        self.assertTrue(decided, "no tuned geometry was measured outside the noise band")
        for row in decided:
            with self.subTest(batch=row["batch"]):
                self.assertLess(
                    row["composed_over_fused"],
                    1.0,
                    "the composition is not faster where the tuned module ran",
                )
        self.assertLess(
            statistics.median(row["composed_over_fused"] for row in vendor),
            1.0,
        )

    def test_comparisons_cover_state_and_gradients(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual({entry["label"] for entry in row["tensors"]}, TENSORS)


class LiveChannelMixBenchTest(unittest.TestCase):
    """Re-run the A/B, which fails on a promotion or parity regression."""

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU channel-mix A/B",
    )
    def test_promotion_holds_on_this_adapter(self) -> None:
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
