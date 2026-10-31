#!/usr/bin/env python3
"""Contract tests for the RWKV packed-cell vendor-composition evidence.

`verify_rwkv_packed_cell_composition.py` produces
`validation/packed_cell_composition.json`. This module turns that evidence into
a test the ordinary `unittest discover` run executes, so a receipt that loses a
vendor dispatch count, the time-mix fusion switch or the negative control fails
the suite instead of sitting in the tree as a stale artifact.

The receipt is evidence, not a live run: these tests catch a regressed or
hand-edited record, not a code change that would produce a different one. Set
`HIERARCHOS_RUN_GPU_VALIDATION=1` to also re-run the harness on the attached
adapter, which is the check that fails when the code regresses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "validation" / "packed_cell_composition.json"
HARNESS = ROOT / "validation" / "verify_rwkv_packed_cell_composition.py"

VENDOR_FAMILIES = ("linear-forward", "linear-residual-forward", "linear3-forward")
VENDOR_PREFIXES = {"intel": "intel-", "amd": "amd-"}


def parse_pairs(line: str) -> dict[str, str]:
    pairs = {}
    for entry in line.split():
        key, _, value = entry.partition("=")
        pairs[key] = value
    return pairs


class CompositionReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RECEIPT.exists():
            raise AssertionError(
                f"missing {RECEIPT.relative_to(ROOT)}; run "
                "validation/verify_rwkv_packed_cell_composition.py to record it"
            )
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))

    def test_passed_within_gate(self) -> None:
        self.assertEqual(self.receipt["result"], "pass")
        self.assertEqual(self.receipt["negative_control"], "rejected")
        self.assertEqual(self.receipt["comparisons"], self.receipt["steps"] * 5)
        self.assertLessEqual(self.receipt["worst_max_abs"], self.receipt["atol"])

    def test_forward_fusion_switches_were_disabled(self) -> None:
        self.assertEqual(
            self.receipt["time_mix_forward_fusion"],
            "disabled",
            "the receipt no longer proves the time-mix took the linear3-forward composition",
        )
        self.assertEqual(
            self.receipt["channel_mix_forward_fusion"],
            "disabled",
            "the receipt no longer proves key_cm/value_cm reached the vendor linears",
        )

    def test_portable_leg_dispatches_no_vendor_modules(self) -> None:
        self.assertEqual(
            parse_pairs(self.receipt["portable_dispatch"]),
            {},
            "the portable leg must not dispatch vendor modules, or the run is not an A/B",
        )

    def test_vendor_leg_dispatches_all_three_linears(self) -> None:
        counts = parse_pairs(self.receipt["vendor_dispatch"])
        for family in VENDOR_FAMILIES:
            with self.subTest(family=family):
                self.assertIn(family, counts)
                self.assertGreater(int(counts[family]), 0)

    def test_plan_resolves_vendor_specific_variants(self) -> None:
        variants = parse_pairs(self.receipt["vendor_plan"])
        for family in VENDOR_FAMILIES:
            with self.subTest(family=family):
                self.assertIn(family, variants)
                self.assertNotEqual(
                    variants[family],
                    "portable",
                    "the vendor leg resolved the portable module; the check is vacuous",
                )

    def test_expected_vendor_matches_the_recorded_plan(self) -> None:
        expected = self.receipt.get("expected_vendor", "auto")
        if expected == "auto":
            self.skipTest("receipt was recorded without an expected-vendor qualification")
        variants = parse_pairs(self.receipt["vendor_plan"])
        prefix = VENDOR_PREFIXES[expected]
        for family in VENDOR_FAMILIES:
            with self.subTest(family=family):
                self.assertTrue(
                    variants[family].startswith(prefix),
                    f"{family} resolved {variants[family]!r}, not an {expected} module",
                )

    def test_comparisons_cover_state_and_gradients(self) -> None:
        labels = {row["label"].rsplit(" ", 1)[-1] for row in self.receipt["rows"]}
        self.assertEqual(
            labels,
            {"output", "packed_new_state", "grad_x", "grad_packed_state", "token_feature_grad"},
        )


class LiveCompositionTest(unittest.TestCase):
    """Re-run the harness, which fails on a dispatch or parity regression."""

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU composition check",
    )
    def test_harness_passes_on_this_adapter(self) -> None:
        expected_vendor = os.environ.get("HIERARCHOS_EXPECTED_VENDOR", "auto")
        completed = subprocess.run(
            [sys.executable, str(HARNESS), "--expected-vendor", expected_vendor],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
