#!/usr/bin/env python3
"""Negative controls for the Phase XIX-Y full-attention receipt."""
from __future__ import annotations

import copy
import unittest

from check_phase19y_full_attention import (
    RECEIPT, load, run_negative_controls, validate,
)


@unittest.skipUnless(RECEIPT.is_file(), "Phase XIX-Y receipt is not present")
class FullAttentionNegativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load(RECEIPT)
        cls.binary_root = RECEIPT.parents[2] / "target/debug"

    def test_positive_receipt_validates(self):
        validate(self.document, RECEIPT.parent, binary_root=self.binary_root)

    def test_all_controls_are_rejected(self):
        result = run_negative_controls(self.document, RECEIPT.parent,
                                       binary_root=self.binary_root)
        self.assertGreaterEqual(result["total"], 12)
        self.assertEqual(result["detected"], result["total"])
        self.assertTrue(all(row["rejected"] for row in result["controls"]))

    def test_missing_evidence_is_rejected(self):
        document = copy.deepcopy(self.document)
        document["evidence"]["missing.json"] = "0" * 64
        with self.assertRaises(ValueError):
            validate(document, RECEIPT.parent, binary_root=self.binary_root)

    def test_threshold_is_not_weakened(self):
        document = copy.deepcopy(self.document)
        document["shapes"]["results"][0]["checks"][0]["max_abs"] = 2e-7
        with self.assertRaises(ValueError):
            validate(document, RECEIPT.parent, binary_root=self.binary_root)


if __name__ == "__main__":
    unittest.main()