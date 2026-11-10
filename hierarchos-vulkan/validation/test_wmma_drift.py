#!/usr/bin/env python3
"""Negative controls for the Phase XVIII drift checker.

Each control forges a mutation *and* updates the evidence hash that guards it,
so the rejection must come from the semantic rule the control targets rather
than from the hash guard.
"""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import check_wmma_drift as checker
import qualify_wmma_drift as q


class DriftControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = q.RECEIPT_ROOT
        if not (cls.source / "current-qualification.json").is_file():
            raise unittest.SkipTest("Phase XVIII receipt is not installed")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        shutil.copytree(cls.source / "evidence", cls.root / "evidence")
        cls.receipt = json.loads(
            (cls.source / "current-qualification.json").read_text(encoding="utf-8")
        )
        cls.backups: dict[Path, bytes] = {}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.receipt = json.loads(
            (self.source / "current-qualification.json").read_text(encoding="utf-8")
        )
        for path, data in list(self.backups.items()):
            path.write_bytes(data)
        self.backups.clear()
        self.write()

    def write(self) -> None:
        checker.q  # keep the import reference explicit
        (self.root / "current-qualification.json").write_text(
            json.dumps(self.receipt, indent=2) + "\n", encoding="utf-8"
        )

    def expect(self, fragment: str) -> None:
        self.write()
        with self.assertRaises(checker.QualificationError) as context:
            checker.verify(self.root / "current-qualification.json")
        self.assertIn(fragment, str(context.exception))

    def edit_evidence(self, name: str, mutate) -> None:
        path = self.root / "evidence" / name
        self.backups.setdefault(path, path.read_bytes())
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.receipt["evidence"][name] = q.sha256_file(path)
        self.write()

    def run_record(self, index: int) -> dict:
        return self.receipt["runs"][index]

    def test_mismatched_initial_weights(self) -> None:
        self.run_record(3)["parameter_start_digest"] = "0" * 64
        self.expect("initial parameter digests differ")

    def test_mismatched_training_batch(self) -> None:
        self.run_record(1)["partition_stream_sha256"]["train"] = "0" * 64
        self.expect("train stream digest changed")

    def test_incorrect_wmma_mode(self) -> None:
        run = self.run_record(1)
        run["mode"] = "off"
        self.expect("FP32 arm dispatched cooperative kernels")

    def test_missing_cooperative_dispatch(self) -> None:
        self.run_record(1)["records"][7]["wmma_weight_grad"] = 0
        self.expect("did not dispatch every weight gradient")

    def test_scalar_fallback_mislabeled_mixed(self) -> None:
        self.run_record(1)["records"][9]["wmma_shape_fallbacks"] = 1
        self.expect("fell back on a shape")

    def test_no_weight_mirror_conversion(self) -> None:
        self.run_record(1)["records"][4]["weight_conversions"] = 0
        self.expect("no weight-mirror conversion")

    def test_tampered_step_loss(self) -> None:
        record = self.run_record(0)["records"][5]
        record["loss_bits"] ^= 0x1
        self.expect("loss bits do not encode its loss")

    def test_tampered_validation_loss(self) -> None:
        evaluation = next(
            e for e in self.run_record(0)["evaluations"] if e["partition"] == "validation"
        )
        evaluation["cross_entropy"] += 0.01
        self.expect("loss was tampered")

    def test_tampered_test_loss(self) -> None:
        evaluation = next(
            e for e in self.run_record(2)["evaluations"] if e["partition"] == "test"
        )
        evaluation["cross_entropy"] -= 0.01
        self.expect("loss was tampered")

    def test_forged_numerical_pass(self) -> None:
        self.receipt["pass"] = True
        evaluation = next(
            e for e in self.run_record(2)["evaluations"] if e["partition"] == "test"
        )
        evaluation["cross_entropy"] = 0.0
        self.expect("loss was tampered")

    def test_wrong_optimizer_configuration(self) -> None:
        self.receipt["protocol"]["beta2"] = 0.99
        self.expect("optimizer configuration changed")

    def test_evaluation_advanced_the_optimizer(self) -> None:
        evaluation = self.run_record(1)["evaluations"][1]
        evaluation["parameter_digest_after"] = "1" * 64
        self.expect("evaluation modified parameters")

    def test_evaluation_dispatched_a_gradient(self) -> None:
        evaluation = self.run_record(1)["evaluations"][1]
        evaluation["wmma_weight_grad_delta"] = 15
        self.expect("evaluation dispatched a gradient kernel")

    def test_forward_comparison_did_not_dispatch(self) -> None:
        self.receipt["observations"]["forward_comparisons"]["step-0"]["mixed_dispatches"] = 0
        self.expect("did not dispatch every projection")

    def test_forward_comparison_weight_mismatch(self) -> None:
        comparison = self.receipt["observations"]["forward_comparisons"]["step-0"]
        comparison["mixed_parameter_digest"] = "2" * 64
        self.expect("did not hold identical weights")

    def test_forward_comparison_does_not_match_training(self) -> None:
        comparison = self.receipt["observations"]["forward_comparisons"]["step-100"]
        comparison["cross_entropy_off"] += 0.01
        self.expect("does not match the trained weights")

    def test_altered_update_statistic(self) -> None:
        divergence = self.receipt["observations"]["step_1_update"]["divergence"]
        group = next(iter(divergence))
        divergence[group]["cosine"] = 1.5
        self.expect("invalid cosine")

    def test_missing_parameter_coverage(self) -> None:
        divergence = self.receipt["observations"]["step_1_update"]["divergence"]
        group = next(iter(divergence))
        divergence[group]["count"] -= 1
        self.expect("does not cover every parameter")

    def test_wrong_order_statistic(self) -> None:
        self.receipt["order_analysis"]["validation_gap_mean"] += 0.01
        self.expect("does not follow the pairs")

    def test_claimed_confidence_interval(self) -> None:
        self.receipt["order_analysis"]["confidence_interval"] = [0.0, 1.0]
        self.expect("confidence interval is claimed")

    def test_duplicate_run(self) -> None:
        self.receipt["runs"][1]["label"] = self.receipt["runs"][0]["label"]
        self.expect("run set changed")

    def test_missing_evaluation(self) -> None:
        run = self.run_record(0)
        run["evaluations"] = [
            evaluation
            for evaluation in run["evaluations"]
            if not (evaluation["partition"] == "validation" and evaluation["after_step"] == 20)
        ]
        self.expect("validation schedule changed")

    def test_invalid_source_ancestry(self) -> None:
        self.receipt["source_revision"] = "1" * 40
        self.expect("not an ancestor")

    def test_missing_evidence_artifact(self) -> None:
        path = self.root / "evidence" / "analysis.json"
        self.backups.setdefault(path, path.read_bytes())
        path.unlink()
        self.expect("missing evidence artifact")

    def test_changed_model_identity(self) -> None:
        self.receipt["fixture"]["model_config_sha256"] = "0" * 64
        self.expect("model config changed")

    def test_changed_partition_file(self) -> None:
        fixture_dir = q.FIXTURE
        path = fixture_dir / "validation.jsonl"
        backup = path.read_bytes()
        self.backups.setdefault(path, backup)
        rows = [json.loads(line) for line in backup.decode("utf-8").splitlines() if line.strip()]
        rows[0]["input_ids"][0] = (rows[0]["input_ids"][0] + 1) % 1024
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )
        try:
            self.expect("partition file changed")
        finally:
            path.write_bytes(backup)


if __name__ == "__main__":
    unittest.main()
