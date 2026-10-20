#!/usr/bin/env python3
"""Contract tests for the multi-step TBPTT time-mix receipt.

`bench_rwkv_tbptt_time_mix.py` produces
`validation/packed_cell_tbptt_time_mix_bench.json`: the same fused-versus-composed
projection A/B, but read end to end through a full TBPTT sequence (forward sweep
plus reverse pass), which is what a training run actually pays per window.

These tests enforce the promotion on the sequence rather than on a single call:
the shipped default must compose exactly where the tuned `linear3-forward` module
ran, keep the fused kernel where it did not, and the whole-sequence timings must
show the same sign change the single-step receipt recorded. Without that last
check the end-to-end receipt could agree with the rule while the gain had quietly
evaporated.

The receipt is evidence, not a live run. Set `HIERARCHOS_RUN_GPU_VALIDATION=1`
to re-run the sequence A/B on the attached adapter.
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
RECEIPT = ROOT / "validation" / "packed_cell_tbptt_time_mix_bench.json"
HARNESS = ROOT / "validation" / "bench_rwkv_tbptt_time_mix.py"

TENSORS = {
    "outputs",
    "final_packed_state",
    "grad_x",
    "token_feature_grad",
    "grad_initial_packed_state",
}


class TbpttTimeMixReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RECEIPT.exists():
            raise AssertionError(
                f"missing {RECEIPT.relative_to(REPO)}; run "
                "validation/bench_rwkv_tbptt_time_mix.py --record to record it"
            )
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        cls.batches = cls.receipt["batches"]

    def vendor_batches(self) -> list[dict]:
        return [row for row in self.batches if row["composed_ran_vendor_linear3"]]

    def portable_batches(self) -> list[dict]:
        return [row for row in self.batches if not row["composed_ran_vendor_linear3"]]

    def test_passed_within_gate(self) -> None:
        self.assertEqual(self.receipt["result"], "pass")
        self.assertGreaterEqual(self.receipt["steps"], 2, "a TBPTT window needs at least two steps")
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
                    "the forced-fusion sequence reached the vendor linear3 module",
                )
                if row["composed_ran_vendor_linear3"]:
                    # One dispatch per timestep in the window, not one per
                    # sequence: the receipt proves the whole sweep composed.
                    # The legacy first-forward transition replays the forward
                    # pass for rematerialization, so the projection dispatches
                    # in both passes and the count doubles.
                    per_step = 1 if row["default_packed_forward"] else 2
                    self.assertEqual(
                        row["composed_linear3_dispatches"],
                        per_step * self.receipt["steps"],
                        "the composed sequence did not dispatch once per timestep "
                        "on the chain the default actually built",
                    )

    def test_promotion_is_geometry_aware(self) -> None:
        self.assertEqual(
            self.receipt["promotion"],
            "geometry-aware",
            "the receipt no longer shows the default sequence following the tuned module's gate",
        )
        self.assertTrue(self.vendor_batches(), "no batch ran the tuned module; the A/B is vacuous")
        self.assertTrue(self.portable_batches(), "no batch fell back to the portable module")
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                if row["composed_ran_vendor_linear3"]:
                    self.assertEqual(row["default_label"], "composed")
                    self.assertEqual(
                        row["default_linear3_dispatches"], row["composed_linear3_dispatches"]
                    )
                else:
                    self.assertTrue(
                        row["default_label"].startswith("fused"),
                        "the default composed where the composed sequence would have run the "
                        "portable module",
                    )
                    self.assertEqual(row["default_linear3_dispatches"], 0)

    def test_sequence_timings_keep_the_single_step_sign_change(self) -> None:
        """The window must keep the split, not flatten it.

        Where the tuned module ran, the composition has to win the whole window.
        Where it did not, the single-step A/B measured the fused kernel ahead by
        under 1.5%, and one TBPTT window dilutes that into a tie -- so the
        end-to-end claim at that geometry is "the composition does not win",
        which is exactly why the fused kernel stays, not "the fused kernel wins".
        A real composed win there would mean the rule should be revisited.
        """
        composed_ratio = statistics.median(
            row["composed_over_fused"] for row in self.vendor_batches()
        )
        fused_ratio = statistics.median(
            row["composed_over_fused"] for row in self.portable_batches()
        )
        self.assertLess(
            composed_ratio,
            1.0,
            f"the composition is not faster over a whole TBPTT window where the tuned module "
            f"ran (median ratio {composed_ratio})",
        )
        self.assertGreater(
            fused_ratio,
            0.99,
            f"the composition now wins a whole window where only the portable module exists "
            f"(median ratio {fused_ratio}); the fused kernel is no longer the safe choice there",
        )
        self.assertLess(
            composed_ratio + 0.01,
            fused_ratio,
            "the window timing no longer separates the tuned-module geometry from the "
            f"portable one (composed {composed_ratio}, portable {fused_ratio})",
        )

    def test_per_step_timing_is_consistent(self) -> None:
        """The per-step column must be the window divided by its steps."""
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertAlmostEqual(
                    row["default_ms"] / self.receipt["steps"],
                    row["default_ms_per_step"],
                    places=6,
                )

    def test_comparisons_cover_state_and_gradients(self) -> None:
        for row in self.batches:
            with self.subTest(batch=row["batch"]):
                self.assertEqual({entry["label"] for entry in row["tensors"]}, TENSORS)


class LiveTbpttTimeMixBenchTest(unittest.TestCase):
    """Re-run the sequence A/B, which fails on a promotion or parity regression."""

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU TBPTT A/B",
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
                "--steps",
                "8",
                "--iterations",
                "4",
                "--warmup",
                "1",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
