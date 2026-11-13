#!/usr/bin/env python3
"""Negative controls for the Phase XIX-Y full-attention receipt."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from check_phase19y_full_attention import (
    CRATE, RECEIPT, load, run_negative_controls, validate, verify_fixture_tree,
)


@unittest.skipUnless(RECEIPT.is_file(), "Phase XIX-Y receipt is not present")
class FullAttentionNegativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load(RECEIPT)
        cls.binary_root = CRATE / Path(cls.document["binaries"]["image_text_pipeline_probe"]["path"]).parent

    def test_positive_receipt_validates(self):
        validate(self.document, RECEIPT.parent, binary_root=self.binary_root)

    def test_executable_path_cannot_escape_crate(self):
        document = copy.deepcopy(self.document)
        document["binaries"]["image_text_pipeline_probe"]["path"] = "../target/release/image_text_pipeline_probe.exe"
        with self.assertRaises(ValueError):
            validate(document, RECEIPT.parent, binary_root=self.binary_root)

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
        for check in document["shapes"]["results"][0]["checks"]:
            if check.get("metric") == "final/logits":
                check["max_abs"] = 2e-7
                break
        else:
            self.skipTest("final/logits check is absent")
        with self.assertRaises(ValueError):
            validate(document, RECEIPT.parent, binary_root=self.binary_root)

    def test_fixture_bytes_are_verified(self):
        shape_root = CRATE / "target/rdna3-port/gpt2-shapes-final-cpu"
        shared_root = (CRATE / "target/rdna3-port/gen9-frozen/target"
                       / "multimodal-gen9/shared-image-cpu-v3")
        if not shape_root.is_dir() or not shared_root.is_dir():
            self.skipTest("recovered fixture bytes are not present")
        shape_manifest = load(RECEIPT.parent / "fixtures/shape-manifest.json")
        shared_manifest = load(RECEIPT.parent / "fixtures/shared-manifest.json")
        self.assertEqual(verify_fixture_tree(shape_root, shape_manifest, "shape"),
                         self.document["fixture_bytes"]["shape_files"])
        self.assertEqual(verify_fixture_tree(shared_root, shared_manifest, "shared"),
                         self.document["fixture_bytes"]["shared_files"])

    def test_altered_fixture_byte_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "case").mkdir()
            payload = root / "case/value.bin"
            payload.write_bytes(b"altered")
            manifest = {"cases": [{"files": {
                "case/value.bin": hashlib.sha256(b"expected").hexdigest()}}]}
            with self.assertRaises(ValueError):
                verify_fixture_tree(root, manifest, "test")
            payload.unlink()
            with self.assertRaises(ValueError):
                verify_fixture_tree(root, manifest, "test")


if __name__ == "__main__":
    unittest.main()
