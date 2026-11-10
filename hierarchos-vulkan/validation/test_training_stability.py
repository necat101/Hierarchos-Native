#!/usr/bin/env python3
"""Negative controls for the training-stability qualification contract."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import check_training_stability as check
import qualify_training_stability as p
import qualify_training_phase16 as q


class StabilityControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt_dir = Path(
            os.environ.get("TRAIN_STAB_RECEIPT", str(p.RECEIPT_DIR))
        )
        if not (cls.receipt_dir / "current-qualification.json").is_file():
            raise unittest.SkipTest("stability receipt is not installed")

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="train-stab-controls-"))
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

    def arm(self, arm_id: str) -> dict:
        return json.loads((self.evidence / f"arm-{arm_id}.json").read_text(encoding="utf-8"))

    def save_arm(self, arm_id: str, report: dict) -> None:
        (self.evidence / f"arm-{arm_id}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    # --- provenance -----------------------------------------------------------
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

    def test_changed_harness_binding(self) -> None:
        receipt = self.load()
        receipt["source_files"]["src/bin/training-stability.rs"] = "1" * 64
        self.reject(receipt)

    def test_changed_fixture_config(self) -> None:
        receipt = self.load()
        fake = self.temp / "fixture"
        model = fake / "model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"fake")
        (fake / "data.jsonl").write_text("{}\n", encoding="utf-8")
        receipt["fixture"]["model"] = str(model)
        receipt["fixture"]["dataset"] = str(fake / "data.jsonl")
        receipt["fixture"]["model_config_sha256"] = "0" * 64
        receipt["fixture"]["model_weights_sha256"] = "0" * 64
        receipt["fixture"]["dataset_sha256"] = "0" * 64
        self.reject(receipt)

    def test_changed_dataset_hash(self) -> None:
        receipt = self.load()
        receipt["fixture"]["dataset_sha256"] = "2" * 64
        self.reject(receipt)

    # --- optimizer and per-step evidence -------------------------------------
    def test_changed_per_step_learning_rate(self) -> None:
        report = self.arm("A")
        report["records"][3]["learning_rate"] = 0.02
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_beta1(self) -> None:
        report = self.arm("A")
        report["records"][3]["beta1"] = 0.8
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_beta2(self) -> None:
        report = self.arm("A")
        report["records"][3]["beta2"] = 0.99
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_epsilon(self) -> None:
        report = self.arm("A")
        report["records"][3]["eps"] = 1e-7
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_weight_decay(self) -> None:
        report = self.arm("A")
        report["records"][3]["weight_decay"] = 0.01
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_optimizer_status(self) -> None:
        report = self.arm("A")
        report["records"][3]["optimizer_status"] = "skipped"
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_optimizer_step(self) -> None:
        report = self.arm("A")
        report["records"][3]["optimizer_step"] = 99
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    # --- matrix and dispatch --------------------------------------------------
    def test_missing_arm(self) -> None:
        receipt = self.load()
        del receipt["arms"]["A"]
        self.reject(receipt)

    def test_forged_step_count(self) -> None:
        receipt = self.load()
        receipt["arms"]["A"]["summary"]["steps"] += 1
        self.reject(receipt)

    def test_zero_query_dispatch_claim(self) -> None:
        receipt = self.load()
        receipt["arms"]["B"]["summary"]["dispatch"]["query"] = 0
        self.reject(receipt)

    def test_scalar_fallback_in_two_pass_log(self) -> None:
        path = self.evidence / "arm-B.err.txt"
        text = path.read_text(encoding="utf-8", errors="replace")
        path.write_text(text.replace("transformer_attention_backward_query_rdna3",
                                     "transformer_attention_backward", 1), encoding="utf-8")
        self.reject(self.load(), refresh=True)

    def test_tampered_profile_log(self) -> None:
        path = self.evidence / "arm-B.err.txt"
        path.write_bytes(path.read_bytes() + b"tampered\n")
        self.reject(self.load(), refresh=True)

    def test_full_finetune_labeled_lora(self) -> None:
        report = self.arm("A")
        report["mode"] = "lora"
        report["lora"] = True
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_lora_labeled_full_finetune(self) -> None:
        report = self.arm("E")
        report["mode"] = "full-finetune"
        report["lora"] = False
        self.save_arm("E", report)
        self.reject(self.load(), refresh=True)

    def test_failed_subprocess_returncode(self) -> None:
        receipt = self.load()
        receipt["arms"]["A"]["summary"]["returncode"] = 1
        self.reject(receipt)

    # --- losses, parameters and resume ---------------------------------------
    def test_changed_loss_bit(self) -> None:
        report = self.arm("A")
        report["records"][3]["loss_bits"] += 1
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_nonfinite_loss(self) -> None:
        report = self.arm("A")
        report["records"][3]["loss_finite"] = False
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_final_parameter_digest(self) -> None:
        report = self.arm("A")
        report["parameter_end_digest"] = "3" * 64
        self.save_arm("A", report)
        self.reject(self.load(), refresh=True)

    def test_changed_adapter_bytes(self) -> None:
        path = self.evidence / "adapter-E-u.safetensors"
        path.write_bytes(path.read_bytes() + b"x")
        self.reject(self.load(), refresh=True)

    def test_missing_checkpoint_state(self) -> None:
        (self.evidence / "states/F/u/hierarchos_adapter_optimizer.safetensors").unlink()
        self.reject(self.load(), refresh=True)

    def test_wrong_resumed_step(self) -> None:
        path = self.evidence / "resume-E-r.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][0]["optimizer_step"] = 99
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        self.reject(self.load(), refresh=True)

    def test_changed_resumed_loss(self) -> None:
        path = self.evidence / "resume-E-r.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["records"][0]["loss_bits"] += 1
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        self.reject(self.load(), refresh=True)

    # --- selection and performance ------------------------------------------
    def test_changed_selection(self) -> None:
        receipt = self.load()
        receipt["sweep"]["selected_lr"] = 0.003
        self.reject(receipt)

    def test_changed_selection_criterion(self) -> None:
        receipt = self.load()
        receipt["sweep"]["criterion"] = "pick the lowest final loss"
        self.reject(receipt)

    def test_performance_sample_tamper(self) -> None:
        receipt = self.load()
        receipt["performance"]["A"]["samples"]["scalar"][0] += 1.0
        self.reject(receipt)


if __name__ == "__main__":
    unittest.main()
