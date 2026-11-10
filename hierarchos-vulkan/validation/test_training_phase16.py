#!/usr/bin/env python3
"""Negative controls for the Phase XVI training qualification contract.

Each test mutates a copy of a genuinely passing receipt. Where the mutation is
inside a per-step report or profiler log, the evidence hash map is refreshed so
the rejection has to come from the checker re-deriving the claim rather than
from a stale file digest.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import check_training_phase16 as check
import qualify_training_phase16 as q


class TrainingQualificationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt_dir = Path(os.environ.get("TRAIN_QUAL_RECEIPT", str(q.RECEIPT_DIR)))
        if not (cls.receipt_dir / "current-qualification.json").is_file():
            raise unittest.SkipTest("training qualification receipt is not installed")

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="train-qual-controls-"))
        self.dir = self.temp / "receipt"
        shutil.copytree(self.receipt_dir, self.dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    @property
    def evidence(self) -> Path:
        return self.dir / "evidence"

    def load(self) -> dict:
        return json.loads((self.dir / "current-qualification.json").read_text(encoding="utf-8"))

    def save(self, receipt: dict) -> None:
        (self.dir / "current-qualification.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    def refresh_evidence(self, receipt: dict) -> None:
        receipt["evidence"] = {
            str(path.relative_to(self.evidence)).replace("\\", "/"): q.sha256_file(path)
            for path in sorted(self.evidence.rglob("*"))
            if path.is_file()
        }
        self.save(receipt)

    def reject(self, receipt: dict | None = None, refresh: bool = False) -> None:
        if receipt is not None:
            if refresh:
                self.refresh_evidence(receipt)
            else:
                self.save(receipt)
        with self.assertRaises(
            (check.QualificationError, FileNotFoundError, KeyError, ValueError, AssertionError)
        ):
            check.verify(self.dir)

    # --- source and provenance ------------------------------------------------
    def test_missing_receipt(self) -> None:
        (self.dir / "current-qualification.json").unlink()
        self.reject()

    def test_unrelated_source_revision(self) -> None:
        receipt = self.load()
        receipt["source_revision"] = "0" * 40
        self.reject(receipt)

    def test_changed_transformer_binding(self) -> None:
        receipt = self.load()
        receipt["source_files"]["src/transformer.rs"] = "0" * 64
        self.reject(receipt)

    def test_changed_manifest_binding(self) -> None:
        receipt = self.load()
        receipt["source_files"]["validation/training_shader_manifest.json"] = "1" * 64
        self.reject(receipt)

    def test_changed_query_shader_binding(self) -> None:
        receipt = self.load()
        receipt["source_files"]["shaders/transformer_attention_backward_query_rdna3.spv"] = "2" * 64
        self.reject(receipt)

    def test_changed_phase16_receipt(self) -> None:
        receipt = self.load()
        receipt["phase16_receipt"]["sha256"] = "3" * 64
        self.reject(receipt)

    def test_changed_harness_binding(self) -> None:
        receipt = self.load()
        receipt["source_files"]["src/bin/hierarchos-vulkan-training-qualification.rs"] = "4" * 64
        self.reject(receipt)

    # --- matrix and dispatch --------------------------------------------------
    def test_missing_arm(self) -> None:
        (self.evidence / "arm-C.json").unlink()
        self.reject()

    def test_forged_step_count(self) -> None:
        receipt = self.load()
        receipt["arms"]["A"]["summary"]["steps"] = 1
        self.reject(receipt)

    def test_zero_query_dispatch_claim(self) -> None:
        receipt = self.load()
        receipt["arms"]["F"]["dispatch"]["query"] = 0
        self.reject(receipt)

    def test_scalar_fallback_in_two_pass_log(self) -> None:
        receipt = self.load()
        path = self.evidence / "arm-F.err.txt"
        lines = [
            line
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if q.QUERY_SHADER not in line
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_tampered_profile_log(self) -> None:
        receipt = self.load()
        path = self.evidence / "arm-A.err.txt"
        lines = [
            line
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "kernel_profile_shader" not in line
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_fp32_arm_claims_wmma(self) -> None:
        receipt = self.load()
        receipt["arms"]["A"]["wmma"]["cooperative"] = 1
        self.reject(receipt)

    def test_lora_claims_weight_gradient(self) -> None:
        receipt = self.load()
        receipt["arms"]["H"]["wmma"]["weight_grad"] = 5
        self.reject(receipt)

    # --- comparisons ----------------------------------------------------------
    def test_changed_loss_bit(self) -> None:
        receipt = self.load()
        path = self.evidence / "arm-B.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][0]["loss_bits"] += 1
        path.write_text(json.dumps(report), encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_nonfinite_loss(self) -> None:
        receipt = self.load()
        path = self.evidence / "arm-B.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][0]["loss"] = float("nan")
        report["records"][0]["loss_finite"] = False
        path.write_text(json.dumps(report), encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_changed_final_parameter_digest(self) -> None:
        receipt = self.load()
        path = self.evidence / "arm-A.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["parameter_end_digest"] = "5" * 64
        path.write_text(json.dumps(report), encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_changed_adapter_bytes(self) -> None:
        receipt = self.load()
        path = self.evidence / "states/F/r/adapter_model.safetensors"
        with path.open("ab") as handle:
            handle.write(b"tampered")
        self.reject(receipt, refresh=True)

    def test_changed_optimizer_configuration(self) -> None:
        receipt = self.load()
        receipt["arms"]["A"]["config"]["two_pass"] = "1"
        self.reject(receipt)

    # --- resume ---------------------------------------------------------------
    def test_missing_checkpoint_state(self) -> None:
        (self.evidence / "states/F/u/hierarchos_adapter_optimizer.safetensors").unlink()
        self.reject()

    def test_wrong_resumed_step(self) -> None:
        receipt = self.load()
        path = self.evidence / "resume-F-r.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][0]["optimizer_step"] = 99
        path.write_text(json.dumps(report), encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_changed_resumed_loss(self) -> None:
        receipt = self.load()
        path = self.evidence / "resume-F-r.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][0]["loss_bits"] += 1
        path.write_text(json.dumps(report), encoding="utf-8")
        self.reject(receipt, refresh=True)

    # --- performance and reversal --------------------------------------------
    def test_performance_sample_tamper(self) -> None:
        receipt = self.load()
        path = sorted(self.evidence.glob("perf-F-*.json"))[0]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][-1]["step_ms"] += 1.0
        path.write_text(json.dumps(report), encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_reversal_window_tamper(self) -> None:
        receipt = self.load()
        path = sorted(self.evidence.glob("reversal-fp32-*.json"))[0]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][0]["step_ms"] += 5.0
        path.write_text(json.dumps(report), encoding="utf-8")
        self.reject(receipt, refresh=True)

    def test_changed_source_binding(self) -> None:
        receipt = self.load()
        receipt["source_files"]["src/lib.rs"] = "6" * 64
        self.reject(receipt)


if __name__ == "__main__":
    unittest.main()
