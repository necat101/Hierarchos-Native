#!/usr/bin/env python3
"""Isolated negative controls for the Phase XIX-X current-source verifier."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
RECEIPT = (
    CRATE
    / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19x"
    / "current-qualification.json"
)
CHECKER = CRATE / "validation/check_phase19x_current.py"


@unittest.skipUnless(RECEIPT.is_file(), "Phase XIX-X receipt is not present")
class PhaseXIXXControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.receipt = self.root / "receipt"
        shutil.copytree(RECEIPT.parent, self.receipt)
        self.document = json.loads((self.receipt / "current-qualification.json").read_text())

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.receipt / "current-qualification.json").write_text(
            json.dumps(self.document, indent=2) + "\n", encoding="utf-8"
        )

    def expect(self, fragment):
        self.write()
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--receipt", str(self.receipt / "current-qualification.json")],
            cwd=CRATE,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0, "checker accepted a forged receipt")
        self.assertIn(fragment, (result.stdout + result.stderr).lower())

    def test_missing_receipt(self):
        path = self.root / "missing.json"
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--receipt", str(path)],
            cwd=CRATE,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", (result.stdout + result.stderr).lower())

    def test_forged_status(self):
        self.document["pass"] = False
        self.expect("not a pass")

    def test_changed_transformer(self):
        self.document["source_files"]["hierarchos-vulkan/src/transformer.rs"] = "0" * 64
        self.expect("bound source changed")

    def test_missing_subreceipt(self):
        del self.document["subreceipts"]["activation"]
        self.expect("sub-receipt set changed")

    def test_changed_shader_inventory(self):
        self.document["shaders"]["inventory"] = 44
        self.expect("shader inventory")

    def test_stale_mirror(self):
        self.document["mirror_epoch"]["post_fix_in_process_step1"] = 7.0
        self.expect("mirror evidence disagrees")

    def test_tampered_evidence(self):
        self.document["evidence"]["activation/current-qualification.json"] = "0" * 64
        self.expect("evidence changed")

    def test_wrong_kind(self):
        self.document["kind"] = "phase19s-instrumented-current-source"
        self.expect("unexpected phase xix-x kind")


if __name__ == "__main__":
    unittest.main()
