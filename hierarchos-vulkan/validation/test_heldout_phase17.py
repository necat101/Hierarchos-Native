#!/usr/bin/env python3
"""Negative controls for the Phase XVII held-out receipt.

Every case starts from a receipt that genuinely passes, applies one mutation,
and requires the checker to reject it for the intended reason. A control that
fails for an unrelated reason does not count.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import check_heldout_phase17 as checker
import qualify_heldout_phase17 as p


class HeldOutControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (p.RECEIPT_DIR / "current-qualification.json").is_file():
            raise unittest.SkipTest("Phase XVII receipt is not installed")
        cls.temp = Path(tempfile.mkdtemp(prefix="phase17-controls-"))
        cls.root = cls.temp / "receipt"
        shutil.copytree(p.RECEIPT_DIR, cls.root)
        # Establish that the untouched copy passes before mutating anything.
        checker.verify(cls.root)
        cls.receipt_path = cls.root / "current-qualification.json"
        cls.original = cls.receipt_path.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp, ignore_errors=True)

    def setUp(self) -> None:
        self.receipt = json.loads(self.original)
        self.evidence_backups: dict[Path, bytes] = {}

    def tearDown(self) -> None:
        self.receipt_path.write_bytes(self.original)
        for path, data in self.evidence_backups.items():
            path.write_bytes(data)
        self.evidence_backups.clear()

    # --- helpers -----------------------------------------------------------
    def write_receipt(self) -> None:
        self.receipt_path.write_text(json.dumps(self.receipt, indent=2) + "\n", encoding="utf-8")

    def edit_evidence(self, name: str, mutate) -> None:
        """Forge a hash-consistent mutation so the semantic rule is what rejects it."""
        path = self.root / "evidence" / name
        self.evidence_backups.setdefault(path, path.read_bytes())
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.receipt["evidence"][name] = checker.q.sha256_file(path)
        self.write_receipt()

    def expect(self, fragment: str) -> None:
        with self.assertRaises(checker.QualificationError) as context:
            checker.verify(self.root)
        self.assertIn(fragment, str(context.exception))

    # --- dataset / split ---------------------------------------------------
    def test_duplicate_training_example_in_validation(self) -> None:
        duplicate = self.receipt["fixture"]["manifest"]["partitions"]["train"]["documents_list"][0]
        self.receipt["fixture"]["manifest"]["partitions"]["validation"]["documents_list"].append(duplicate)
        self.write_receipt()
        self.expect("duplicate documents across partitions: train/validation")

    def test_duplicate_training_example_in_test(self) -> None:
        duplicate = self.receipt["fixture"]["manifest"]["partitions"]["train"]["documents_list"][1]
        self.receipt["fixture"]["manifest"]["partitions"]["test"]["documents_list"].append(duplicate)
        self.write_receipt()
        self.expect("duplicate documents across partitions: train/test")

    def test_overlapping_source_windows_across_partitions(self) -> None:
        fixture_dir = p.CRATE / self.receipt["fixture"]["directory"]
        train_row = next(
            line for line in (fixture_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        path = fixture_dir / "validation.jsonl"
        backup = path.read_bytes()
        rows = [json.loads(line) for line in backup.decode("utf-8").splitlines() if line.strip()]
        # Keep the held-out document's identity and only reuse a training
        # document's tokens, so the overlap rule is what rejects the receipt.
        rows[0]["input_ids"] = json.loads(train_row)["input_ids"]
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
            + "\n",
            encoding="utf-8",
        )
        try:
            self.receipt["fixture"]["files"]["validation.jsonl"] = checker.q.sha256_file(path)
            self.write_receipt()
            self.expect("8-gram overlap between train and validation")
        finally:
            path.write_bytes(backup)

    def test_incorrect_split_identifier(self) -> None:
        self.receipt["fixture"]["manifest"]["fixture_version"] = "phase17-heldout-v2"
        self.write_receipt()
        self.expect("unexpected split manifest identifier")

    def test_altered_dataset_hash(self) -> None:
        self.receipt["fixture"]["phase16_dataset_sha256"] = "0" * 64
        self.write_receipt()
        self.expect("Phase XVI dataset copy changed")

    def test_changed_tokenizer_identity(self) -> None:
        self.receipt["fixture"]["manifest"]["tokenizer"]["revision"] = "2"
        self.write_receipt()
        self.expect("tokenizer identity changed")

    def test_forged_valid_token_count(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: data["evaluations"][0].__setitem__("valid_target_tokens", 999999),
        )
        self.expect("valid target token count")

    def test_padding_included_in_loss(self) -> None:
        manifest = self.receipt["fixture"]["manifest"]["partitions"]["validation"]
        padded = manifest["windows"] * self.receipt["fixture"]["manifest"]["seq_len"]
        # Claim the padded geometry as supervised work while leaving the
        # committed manifest untouched, so the padding rule is what rejects it.
        self.edit_evidence(
            "arm-A.json",
            lambda data: data["evaluations"][0].__setitem__("valid_target_tokens", padded),
        )
        self.expect("padding excluded")

    def test_wrong_causal_target_alignment(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: data["partition_stream_sha256"].__setitem__("validation", "0" * 64),
        )
        self.expect("causal target alignment")

    def test_changed_fixture_bytes(self) -> None:
        fixture_dir = p.CRATE / self.receipt["fixture"]["directory"]
        path = fixture_dir / "corpus.jsonl"
        backup = path.read_bytes()
        # corpus.jsonl feeds neither the leakage nor the window checks, so this
        # control isolates generator reproducibility from those rules.
        path.write_bytes(backup + b"{\"tampered\":true}\n")
        try:
            self.receipt["fixture"]["files"]["corpus.jsonl"] = checker.q.sha256_file(path)
            self.write_receipt()
            self.expect("not reproducible")
        finally:
            path.write_bytes(backup)

    # --- losses and evaluation output -------------------------------------
    def test_tampered_validation_loss(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: next(
                e for e in data["evaluations"] if e["tag"] == "validation@100"
            ).__setitem__("cross_entropy", 1.0),
        )
        self.expect("token-weighted cross-entropy does not recompute")

    def test_tampered_test_loss(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: next(
                e for e in data["evaluations"] if e["tag"] == "test@final"
            ).__setitem__("cross_entropy", 0.5),
        )
        self.expect("token-weighted cross-entropy does not recompute")

    def test_missing_evaluation_output(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: data.__setitem__(
                "evaluations",
                [e for e in data["evaluations"] if e["tag"] != "test@final"],
            ),
        )
        self.expect("missing evaluation output")

    def test_evaluation_modifying_parameters(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: data["evaluations"][0].__setitem__("parameter_digest_after", "f" * 64),
        )
        self.expect("evaluation mutated parameters")

    def test_evaluation_advancing_optimizer_steps(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: data["evaluations"][0].__setitem__(
                "optimizer_step_after", data["evaluations"][0]["optimizer_step_before"] + 1
            ),
        )
        self.expect("evaluation advanced the optimizer step")

    # --- arms and dispatch -------------------------------------------------
    def test_missing_execution_arm(self) -> None:
        del self.receipt["arms"]["D"]
        self.write_receipt()
        self.expect("missing execution arm D")

    def test_scalar_fallback_mislabeled_as_two_pass(self) -> None:
        self.receipt["arms"]["B"]["config"]["backward"] = "scalar"
        self.write_receipt()
        self.expect("execution mode changed")

    def test_tampered_dispatch_profile(self) -> None:
        path = self.root / "evidence" / "arm-B.err.txt"
        self.evidence_backups.setdefault(path, path.read_bytes())
        text = path.read_text(encoding="utf-8", errors="replace")
        path.write_text(
            "\n".join(
                line for line in text.splitlines()
                if "transformer_attention_backward_query_rdna3" not in line
            ) + "\n",
            encoding="utf-8",
        )
        self.receipt["evidence"]["arm-B.err.txt"] = checker.q.sha256_file(path)
        self.write_receipt()
        self.expect("dispatch does not match the declared two-pass route")

    def test_invalid_binary_hash(self) -> None:
        # A truncated digest is malformed, so the rule holds even where no
        # harness binary is built (CI) and the byte comparison is skipped.
        self.receipt["binaries"]["held_out_eval"] = "0" * 63
        self.write_receipt()
        self.expect("not a SHA-256 digest")

    def test_unrelated_source_revision(self) -> None:
        self.receipt["source_revision"] = "1" * 40
        self.write_receipt()
        self.expect("source revision is not an ancestor")

    def test_missing_evidence_artifact(self) -> None:
        path = self.root / "evidence" / "arm-C.json"
        self.evidence_backups.setdefault(path, path.read_bytes())
        path.unlink()
        self.expect("missing evidence artifact")

    def test_validation_only_result_presented_as_test(self) -> None:
        self.edit_evidence(
            "arm-A.json",
            lambda data: next(
                e for e in data["evaluations"] if e["tag"] == "test@final"
            ).__setitem__("partition", "validation"),
        )
        self.expect("validation-only result")


if __name__ == "__main__":
    unittest.main()
