#!/usr/bin/env python3
"""Fail-closed controls for the Phase XIX-V historical CI runner."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import run_phase19u_historical_ci as ci

LOGS = ci.ROOT / "hierarchos-vulkan/validation/historical_audit/logs"


class HistoricalRunnerControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = ci.audit.load_document(ci.ROOT)
        cls.logs = {
            "phase16-training": (LOGS / "training.log").read_text(encoding="utf-8"),
            "phase16-stability": (LOGS / "stability.log").read_text(encoding="utf-8"),
            "phase17-heldout": (LOGS / "heldout.log").read_text(encoding="utf-8"),
            "phase18-wmma-drift": (LOGS / "drift.log").read_text(encoding="utf-8"),
            "phase19r-activation": (LOGS / "activation.log").read_text(encoding="utf-8"),
        }
        cls.recovery_logs = {
            "phase16-training": (LOGS / "training-recovered.log").read_text(encoding="utf-8"),
            "phase16-stability": (LOGS / "stability-recovered.log").read_text(encoding="utf-8"),
        }

    def reject_original(self, run_id: str, text: str, exit_code: int) -> None:
        with self.assertRaises(ci.ContractError):
            ci.classify_original(run_id, text, exit_code)

    def test_all_original_logs_match_contract(self) -> None:
        for run_id, text in self.logs.items():
            result = ci.classify_original(
                run_id, text, ci.audit.EXPECTED_RUNS[run_id]["exit"])
            self.assertTrue(result["counts"]["total"] > 0)

    def test_recovery_logs_match_contract(self) -> None:
        for run_id, text in self.recovery_logs.items():
            ci.classify_recovery(run_id, text, 0)

    def test_wrong_total_is_rejected(self) -> None:
        text = self.logs["phase16-training"].replace("Ran 25 tests", "Ran 24 tests")
        self.reject_original("phase16-training", text, 1)

    def test_truncated_output_is_rejected(self) -> None:
        self.reject_original("phase16-training", "Ran 25 tests in 1.0s\n", 1)

    def test_wrong_exit_is_rejected(self) -> None:
        self.reject_original("phase16-training", self.logs["phase16-training"], 0)

    def test_import_error_is_rejected(self) -> None:
        text = "ERROR: setUpClass (test_training_phase16.TrainingQualificationControls)\nModuleNotFoundError: no module named x\n\nRan 1 test in 0.1s\n\nFAILED (errors=1)\n"
        self.reject_original("phase16-training", text, 1)

    def test_wrong_error_test_is_rejected(self) -> None:
        text = self.logs["phase16-training"].replace(
            "test_missing_checkpoint_state", "test_unrelated_failure")
        self.reject_original("phase16-training", text, 1)

    def test_wrong_missing_artifact_is_rejected(self) -> None:
        text = self.logs["phase16-training"].replace(
            "hierarchos_adapter_optimizer.safetensors", "wrong-artifact.safetensors")
        self.reject_original("phase16-training", text, 1)

    def test_unexpected_skip_is_rejected(self) -> None:
        text = self.logs["phase19r-activation"].replace("OK (skipped=1)", "OK")
        self.reject_original("phase19r-activation", text, 0)

    def test_original_cannot_use_recovery_bytes(self) -> None:
        self.reject_original("phase16-training", self.recovery_logs["phase16-training"], 0)

    def test_recovery_cannot_be_original(self) -> None:
        with self.assertRaises(ci.ContractError):
            ci.classify_recovery("phase16-training", self.logs["phase16-training"], 1)

    def test_wrong_identity_is_rejected(self) -> None:
        record = copy.deepcopy(self.document["runs"][0])
        record["receipt_revision"] = "0" * 40
        with self.assertRaises(ci.ContractError):
            ci.validate_identity(record, ci.ROOT)

    def test_wrong_test_hash_is_rejected(self) -> None:
        record = copy.deepcopy(self.document["runs"][0])
        record["test_sha256"] = "0" * 64
        with self.assertRaises(ci.ContractError):
            ci.validate_identity(record, ci.ROOT)

    def test_missing_source_revision_is_rejected(self) -> None:
        record = copy.deepcopy(self.document["runs"][0])
        record["source_revision"] = "0" * 40
        with self.assertRaises(ci.ContractError):
            ci.validate_identity(record, ci.ROOT)

    def test_unknown_suite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ci.ContractError):
                ci.build_report(ci.ROOT, Path(tmp), "not-a-suite")

    def test_wrong_recovered_bytes_are_rejected(self) -> None:
        expected_sha = "075e6e1f0a3c2f35a09352e32633d0ef3570041574e0c9cd2109db8f1f91188e"
        expected_blob = "2fe0bd4cc7bc14ed87b10e42f94ab3d8cb6385f3"
        with self.assertRaises(ci.ContractError):
            ci.validate_recovered_bytes(expected_sha, expected_blob, b"substitute")

    def test_ci_cannot_claim_current_qualification(self) -> None:
        report = {
            "schema": ci.CI_SCHEMA,
            "scope": "current",
            "not_current_qualification": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ci.ContractError):
                ci.verify_report(report, path)


if __name__ == "__main__":
    unittest.main()
