#!/usr/bin/env python3
"""Negative controls for the Phase XIX-U historical audit checker."""

from __future__ import annotations

import copy
import unittest

import check_phase19u_audit as audit


class HistoricalAuditControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = audit.load_document(audit.ROOT)

    def reject(self, document: dict) -> None:
        with self.assertRaises(audit.AuditError):
            audit.verify_document(document, audit.ROOT)

    def test_baseline_audit_passes(self) -> None:
        summary = audit.verify_document(copy.deepcopy(self.document), audit.ROOT)
        self.assertEqual(summary["runs"], 5)
        self.assertEqual(summary["recoveries"], 2)

    def test_current_source_cannot_change(self) -> None:
        document = copy.deepcopy(self.document)
        document["current_qualification"]["numerical_source_revision"] = "0" * 40
        self.reject(document)

    def test_historical_receipt_cannot_change(self) -> None:
        document = copy.deepcopy(self.document)
        document["runs"][0]["receipt_revision"] = "0" * 40
        self.reject(document)

    def test_test_source_digest_cannot_change(self) -> None:
        document = copy.deepcopy(self.document)
        document["runs"][0]["test_sha256"] = "0" * 64
        self.reject(document)

    def test_log_digest_cannot_change(self) -> None:
        document = copy.deepcopy(self.document)
        document["runs"][0]["log"]["sha256"] = "0" * 64
        self.reject(document)

    def test_count_cannot_be_inflated(self) -> None:
        document = copy.deepcopy(self.document)
        document["runs"][0]["counts"]["passed"] = 25
        self.reject(document)

    def test_recovered_bytes_cannot_be_substituted(self) -> None:
        document = copy.deepcopy(self.document)
        document["runs"][0]["recovery"]["recovered_artifacts"][0]["sha256"] = "0" * 64
        self.reject(document)

    def test_recovery_record_cannot_be_removed(self) -> None:
        document = copy.deepcopy(self.document)
        del document["runs"][0]["recovery"]
        self.reject(document)

    def test_missing_current_control_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["runs"][0]["current_source_controls"].append(
            "hierarchos-vulkan/validation/not-a-real-control.py"
        )
        self.reject(document)


if __name__ == "__main__":
    unittest.main()
