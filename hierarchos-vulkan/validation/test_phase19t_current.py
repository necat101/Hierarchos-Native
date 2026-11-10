#!/usr/bin/env python3
"""Non-destructive negative controls for the Phase XIX-T policy overlay."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import check_phase19t_current as checker
from phase19t_policy import (
    DEFAULT_OVERLAY,
    RECEIPT_DIR,
    PolicyError,
    check_evidence,
    digest,
    load,
    manifest_sha256,
    safe_relative,
    validate_manifest,
)

RECEIPT = RECEIPT_DIR / "current-qualification.json"


@unittest.skipUnless(RECEIPT.is_file() and DEFAULT_OVERLAY.is_file(),
                     "Phase XIX-T policy overlay is not present")
class PhaseXIXTControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_overlay = load(DEFAULT_OVERLAY)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "receipt"
        shutil.copytree(RECEIPT_DIR, self.root)
        self.receipt_path = self.root / "current-qualification.json"
        self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.overlay = copy.deepcopy(self.base_overlay)
        self.overlay_path = Path(self.temporary.name) / "policy.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write_receipt(self):
        self.receipt_path.write_text(json.dumps(self.receipt, indent=2) + "\n",
                                     encoding="utf-8")
        self.overlay["numerical_receipt"]["sha256"] = digest(self.receipt_path)

    def write_overlay(self):
        self.overlay["source_manifest_sha256"] = manifest_sha256(
            self.overlay["required_sources"])
        self.overlay["evidence_manifest_sha256"] = manifest_sha256(
            self.overlay["required_evidence"])
        self.overlay_path.write_text(json.dumps(self.overlay, indent=2) + "\n",
                                     encoding="utf-8")

    def run_checker(self, strict=True, binary_root=None):
        self.write_overlay()
        return checker.verify(self.overlay_path, self.receipt_path,
                              binary_root=binary_root, strict_policy=strict)

    def expect(self, fragment, strict=True, binary_root=None):
        with self.assertRaises(PolicyError) as context:
            self.run_checker(strict=strict, binary_root=binary_root)
        self.assertIn(fragment, str(context.exception).lower())
        return context.exception

    def rebind_subreceipt(self, name, mutate):
        path = self.root / name / "current-qualification.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        subhash = digest(path)
        self.receipt["subreceipts"][name]["sha256"] = subhash
        self.receipt["evidence"][f"{name}/current-qualification.json"] = subhash
        self.overlay["required_evidence"][f"{name}/current-qualification.json"] = subhash

    def test_baseline_passes(self):
        summary = self.run_checker()
        self.assertEqual(summary["sources"], 178)
        self.assertEqual(summary["evidence"], 338)
        self.assertEqual(summary["safetensors"], 36)
        self.assertEqual(summary["subreceipts"], 6)

    def test_missing_required_source(self):
        self.receipt["source_files"].pop("hierarchos-vulkan/src/transformer.rs")
        self.write_receipt()
        self.expect("receipt source set differs")

    def test_malformed_receipt_source_digest(self):
        self.receipt["source_files"]["hierarchos-vulkan/src/transformer.rs"] = "0" * 63
        self.write_receipt()
        self.expect("malformed receipt source digest")

    def test_required_source_manifest_cannot_shrink(self):
        self.receipt["source_files"].pop("hierarchos-vulkan/src/transformer.rs")
        self.overlay["required_sources"].pop("hierarchos-vulkan/src/transformer.rs")
        self.write_receipt()
        self.expect("required source count changed", strict=False)

    def test_missing_evidence(self):
        (self.root / "training/evidence/arm-A.json").unlink()
        self.expect("missing evidence")

    def test_empty_mandatory_evidence(self):
        (self.root / "training/evidence/arm-A.json").write_bytes(b"")
        self.expect("empty mandatory evidence")

    def test_changed_safetensors(self):
        path = self.root / "training/evidence/adapter-E.safetensors"
        path.write_bytes(path.read_bytes() + b"changed")
        self.expect("evidence changed")

    def test_required_safetensors_manifest_cannot_shrink(self):
        self.overlay["required_safetensors"].pop()
        self.expect("required safetensors manifest changed", strict=False)

    def test_missing_binary_binding(self):
        self.receipt["binaries"].pop("wmma_drift")
        self.write_receipt()
        self.expect("binary binding set changed")

    def test_malformed_binary_digest(self):
        self.receipt["binaries"]["wmma_drift"] = "0" * 63
        self.write_receipt()
        self.expect("malformed binary digest")

    def test_changed_supplied_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "activation_probe.exe").write_bytes(b"not the qualified probe")
            self.expect("binary hash changed", binary_root=root)

    def test_subreceipt_device_conflict(self):
        self.rebind_subreceipt("training", lambda document: document.__setitem__(
            "device", "Intel(R) HD Graphics 520"))
        self.write_receipt()
        self.expect("device identities differ", strict=False)

    def test_subreceipt_revision_conflict(self):
        self.rebind_subreceipt("training", lambda document: document.__setitem__(
            "source_revision", "99a87f579e9fb0f303b5f3a0edeca523709ec7d1"))
        self.write_receipt()
        self.expect("training source revision changed", strict=False)

    def test_policy_file_changed(self):
        self.overlay["policy_files"]["hierarchos-vulkan/validation/phase19t_policy.py"] = "0" * 64
        self.expect("policy file changed", strict=False)

    def test_path_traversal(self):
        with self.assertRaises(PolicyError) as context:
            safe_relative("../outside.json", self.root, "evidence")
        self.assertIn("unsafe", str(context.exception).lower())

    def test_duplicate_normalized_identity(self):
        with self.assertRaises(PolicyError) as context:
            validate_manifest({"evidence/a.json": "0" * 64,
                               "Evidence/a.json": "1" * 64}, "evidence")
        self.assertIn("duplicate", str(context.exception).lower())

    def test_untracked_evidence_is_rejected(self):
        evidence = {"activation/evidence/anomaly.json":
                    self.receipt["evidence"]["activation/evidence/anomaly.json"]}
        with self.assertRaises(PolicyError) as context:
            check_evidence(self.root, evidence,
                           "hierarchos-vulkan/" + self.overlay["numerical_receipt"]["path"],
                           set())
        self.assertIn("not tracked", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()
