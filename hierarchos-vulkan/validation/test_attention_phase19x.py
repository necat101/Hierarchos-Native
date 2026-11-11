#!/usr/bin/env python3
"""Isolated negative controls for the Phase XIX-X attention verifier."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261111-phase19x-current-source" / "attention" / "current-qualification.json"
)
CHECKER = CRATE / "validation/check_attention_phase19x.py"


@unittest.skipUnless(RECEIPT.is_file(), "attention receipt is not present")
class AttentionPhaseXIXXControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / "attention"
        shutil.copytree(RECEIPT.parent, self.directory)
        self.path = self.directory / "current-qualification.json"
        self.document = json.loads(self.path.read_text())

    def tearDown(self):
        self.temp.cleanup()

    def expect(self, fragment):
        self.path.write_text(json.dumps(self.document, indent=2) + "\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--receipt", str(self.path)],
            cwd=CRATE, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0, "checker accepted a forged receipt")
        self.assertIn(fragment, (result.stdout + result.stderr).lower())

    def test_missing_receipt(self):
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--receipt", str(self.root / "missing.json")],
            cwd=CRATE, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", (result.stdout + result.stderr).lower())

    def test_forged_status(self):
        self.document["pass"] = False
        self.expect("not a pass")

    def test_changed_transformer(self):
        self.document["source_files"]["src/transformer.rs"] = "0" * 64
        self.expect("source changed")

    def test_changed_trainer(self):
        self.document["binaries"]["trainer"] = "0" * 64
        self.expect("trainer changed")

    def test_changed_coverage(self):
        self.document["attention_backward"]["comparisons"] = 11
        self.expect("coverage changed")

    def test_changed_evidence(self):
        self.document["evidence"]["two-pass/test.log"] = "0" * 64
        self.expect("evidence changed")

    def test_changed_historical_coverage(self):
        self.document["historical_phase16"]["retained"]["shared"] = 73
        self.expect("retained coverage changed")

    def test_wrong_kind(self):
        self.document["kind"] = "phase16-two-pass"
        self.expect("unexpected attention kind")


if __name__ == "__main__":
    unittest.main()
