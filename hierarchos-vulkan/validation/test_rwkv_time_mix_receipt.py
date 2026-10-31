#!/usr/bin/env python3
"""Contract tests for the packed-cell time-mix promotion evidence.

`bench_rwkv_packed_cell_time_mix.py` produces
`validation/packed_cell_time_mix_bench.json`: the A/B between the fused
projection/key/state kernel and the composed `linear3-forward` projection, plus
what the production default graph does at each batch size.

The receipt is what makes the promotion auditable, so these tests check the
promotion itself rather than only the numbers: the default must compose exactly
where the tuned module ran, keep the fused kernel where it did not, and the
timings that justify the split must show the composition faster in the first
group and the fused kernel faster in the second.

The receipt is evidence, not a live run: these tests catch a stale or hand-edited
record. Set `HIERARCHOS_RUN_GPU_VALIDATION=1` to re-run the A/B on the attached
adapter, which is the check that fails when the code regresses.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RECEIPT = ROOT / "validation" / "packed_cell_time_mix_bench.json"
DECISIONS = ROOT / "validation" / "time_mix_projection_decisions.json"
HARNESS = ROOT / "validation" / "bench_rwkv_packed_cell_time_mix.py"

TENSORS = {"output", "packed_new_state", "grad_x", "grad_packed_state", "token_feature_grad"}
# The band inside which a measured sign is not a decision. The driver withdraws an
# entry for a geometry it measures in here and leaves the geometry to the vendor
# gate, so the record can never claim a crossover the measurement no longer shows.
NOISE_BAND = 0.02


class RecurrentCompositionAdmissionTest(unittest.TestCase):
    def test_single_cell_agreement_does_not_promote_the_known_recurrent_rounding_change(self):
        import ab_harness as ab
        import bench_rwkv_packed_cell_time_mix as driver
        run = dict(device=driver.WITHHELD_DEVICE, width=128, batch=8,
                   composed_over_fused=0.9, worst_max_abs=0.0,
                   fused={"timing": {"median_ms": 1.0}},
                   composed={"timing": {"median_ms": 0.9}})
        with tempfile.TemporaryDirectory() as directory:
            record, _ = ab.record_decisions(
                Path(directory) / "routes.json", [run], kind=driver.RECORD_KIND,
                probe_key=driver.PROBE_KEY, margin=0.02,
                withhold_when=driver.withheld_reason)
        self.assertEqual(record["decisions"][0]["route"], "staged")
        self.assertEqual(record["withheld"][0]["parameter_storage"], "fp32")
        self.assertIn("eight-timestep", record["withheld"][0]["reason"])
        self.assertIsNone(driver.withheld_reason(dict(run, device="Intel(R) HD Graphics 520")))
        self.assertIsNone(driver.withheld_reason(dict(run, composed_over_fused=1.1)))


class TimeMixBenchReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RECEIPT.exists():
            raise AssertionError(
                f"missing {RECEIPT.relative_to(REPO)}; run "
                "validation/bench_rwkv_packed_cell_time_mix.py --record to record it"
            )
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        cls.batches = cls.receipt["batches"]

    def vendor_batches(self) -> list[dict]:
        return [row for row in self.batches if row["composed_ran_vendor_linear3"]]

    def portable_batches(self) -> list[dict]:
        return [row for row in self.batches if not row["composed_ran_vendor_linear3"]]

    def test_passed_within_gate(self) -> None:
        self.assertEqual(self.receipt["result"], "pass")
        self.assertGreaterEqual(len(self.batches), 2)
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertLessEqual(row["worst_max_abs"], self.receipt["atol"])

    def test_legs_ran_different_graphs(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertTrue(row["fused_label"].startswith("fused"))
                self.assertEqual(row["composed_label"], "composed")
                self.assertEqual(
                    row["fused_linear3_dispatches"],
                    0,
                    "the forced-fusion leg reached the vendor linear3 module",
                )

    def test_promotion_is_geometry_aware(self) -> None:
        """The default must build the decision it should build.

        That decision is the compiled-in per-device record where this adapter has
        an entry, and the tuned module's own admission gate otherwise. The gate
        proxy is the fallback, not the rule, so a record entry may legitimately
        keep the fused kernel at a geometry where the gate would have composed.
        """
        self.assertEqual(
            self.receipt["promotion"],
            "geometry-aware",
            "the receipt no longer shows the default following its decision record",
        )
        self.assertTrue(self.vendor_batches(), "no batch ran the tuned module; the A/B is vacuous")
        self.assertTrue(self.portable_batches(), "no batch fell back to the portable module")
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                entry = self.decision_entries().get(
                    (self.receipt["device"].strip().lower(), self.receipt["width"], row["batch"])
                )
                if entry is not None:
                    expected = "composed" if entry["route"] == "staged" else row["fused_label"]
                elif row["composed_ran_vendor_linear3"]:
                    expected = "composed"
                else:
                    expected = row["fused_label"]
                self.assertEqual(row["default_label"], expected)
                if expected == "composed":
                    self.assertEqual(
                        row["default_linear3_dispatches"],
                        row["composed_linear3_dispatches"],
                    )
                else:
                    self.assertEqual(row["default_linear3_dispatches"], 0)

    def decision_entries(self) -> dict:
        record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        return {
            (entry["device"].strip().lower(), entry["width"], entry["batch"]): entry
            for entry in record["decisions"]
        }

    def test_timings_justify_the_split(self) -> None:
        vendor = self.vendor_batches()
        portable = self.portable_batches()
        composed_ratio = statistics.median(row["composed_over_fused"] for row in vendor)
        fused_ratio = statistics.median(row["composed_over_fused"] for row in portable)
        self.assertLess(
            composed_ratio,
            1.0 - NOISE_BAND,
            f"the composition is not faster where the tuned module ran (median ratio {composed_ratio})",
        )
        # The mirror of the promotion is not a win for the fused kernel: an individual
        # geometry inside the band is a tie, and the record's job is to keep the
        # composition from being promoted there, which it does by carrying no entry.
        self.assertGreater(
            fused_ratio,
            1.0 - NOISE_BAND,
            f"the composition wins where only the portable module exists (median {fused_ratio})",
        )

    def test_default_follows_the_recorded_decision_table(self) -> None:
        """The receipt and the compiled-in decision table must name one graph.

        `src/decision_records.rs` compiles this file into the binary, so a table
        that disagrees with the receipt would make the shipped default follow a
        rule nobody measured. A geometry the receipt measured *outside* the noise
        band therefore has to have an entry agreeing with its own ratio, and the
        default has to build what that entry names. A geometry measured *inside*
        the band is a tie, and a tie is not a decision: the driver withdraws the
        entry, the geometry falls back to the vendor gate, and an entry left there
        would be evidence a measurement no longer supports -- which is exactly what
        batch 4 did when a re-run measured it at `0.997x` against a recorded
        `1.006x`.
        """
        self.assertTrue(DECISIONS.exists(), f"missing {DECISIONS.relative_to(REPO)}")
        record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        self.assertEqual(record["schema"], 2)
        entries: dict[tuple[str, int, int], dict] = {}
        for entry in record["decisions"]:
            key = (entry["device"].strip().lower(), entry["width"], entry["batch"])
            self.assertNotIn(key, entries, "duplicate decision for one device/width/batch")
            self.assertEqual(entry["source"], "measured")
            entries[key] = entry

        device = self.receipt["device"].strip().lower()
        width = self.receipt["width"]
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                entry = entries.get((device, width, row["batch"]))
                if abs(row["composed_over_fused"] - 1.0) <= NOISE_BAND:
                    self.assertIsNone(
                        entry,
                        f"batch {row['batch']} measures inside the {NOISE_BAND:.0%} noise "
                        "band, so it is a tie and must not carry a recorded decision; "
                        "re-record with --record-decisions after timing it",
                    )
                    continue
                self.assertIsNotNone(
                    entry,
                    f"no recorded decision for batch {row['batch']}; re-record with "
                    "--record-decisions after timing it",
                )
                self.assertEqual(
                    entry["route"] == "staged",
                    row["composed_over_fused"] < 1.0,
                    "the recorded decision contradicts the receipt's own measurement",
                )
                if not row["composed_ran_vendor_linear3"]:
                    self.assertFalse(
                        entry["route"] == "staged",
                        "a decision composed at a geometry where the composition would run the "
                        "portable module",
                    )
                if entry["route"] == "staged":
                    self.assertEqual(row["default_label"], "composed")
                else:
                    self.assertTrue(row["default_label"].startswith("fused"))

    def test_comparisons_cover_state_and_gradients(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual({entry["label"] for entry in row["tensors"]}, TENSORS)


class LiveTimeMixBenchTest(unittest.TestCase):
    """Re-run the A/B, which fails on a promotion or parity regression."""

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU time-mix A/B",
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
                "20",
                "--warmup",
                "5",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
