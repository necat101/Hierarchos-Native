#!/usr/bin/env python3
"""Negative controls for the Phase XIX-S current-source verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
RECEIPT = (CRATE / "validation/release_parity/multimodal_training"
           / "20261110-phase19s-current-source" / "current-qualification.json")
CHECKER = CRATE / "validation/check_phase19s_current.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(RECEIPT.is_file(), "Phase XIX-S receipt is not present")
class PhaseXIXSControls(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(RECEIPT.read_text(encoding="utf-8"))

    def run_checker(self, receipt=RECEIPT, binary_root=None):
        command = [sys.executable, str(CHECKER), "--receipt", str(receipt)]
        if binary_root:
            command += ["--binary-root", str(binary_root)]
        return subprocess.run(command, cwd=CRATE, capture_output=True, text=True)

    def expect(self, fragment, receipt=RECEIPT, binary_root=None):
        result = self.run_checker(receipt, binary_root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(fragment, (result.stdout + result.stderr).lower())
        return result

    def test_missing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            self.expect("missing", Path(directory) / "current-qualification.json")

    def test_forged_pass(self):
        document = copy.deepcopy(self.document)
        document["pass"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current-qualification.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.expect("not a pass", path)

    def test_changed_transformer(self):
        original = RECEIPT.read_bytes()
        try:
            document = copy.deepcopy(self.document)
            document["source_files"]["hierarchos-vulkan/src/transformer.rs"] = "0" * 64
            RECEIPT.write_text(json.dumps(document), encoding="utf-8")
            self.expect("bound source changed")
        finally:
            RECEIPT.write_bytes(original)

    def test_changed_binary_hash(self):
        document = copy.deepcopy(self.document)
        document["binaries"]["activation_probe"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "activation_probe.exe").write_bytes(b"not the built probe")
            path = root / "current-qualification.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            shutil.copytree(RECEIPT.parent, root / "receipt", dirs_exist_ok=True)
            # The copied receipt resolves evidence beside itself.
            self.expect("binary hash changed", path, binary_root=root)

    def test_missing_subreceipt(self):
        original = RECEIPT.read_bytes()
        try:
            document = copy.deepcopy(self.document)
            del document["subreceipts"]["heldout"]
            RECEIPT.write_text(json.dumps(document), encoding="utf-8")
            self.expect("sub-receipt set changed")
        finally:
            RECEIPT.write_bytes(original)

    def test_tampered_evidence(self):
        original = RECEIPT.read_bytes()
        try:
            document = copy.deepcopy(self.document)
            document["evidence"]["activation/current-qualification.json"] = "0" * 64
            RECEIPT.write_text(json.dumps(document), encoding="utf-8")
            self.expect("evidence changed")
        finally:
            RECEIPT.write_bytes(original)

    def test_stale_mirror(self):
        original = RECEIPT.read_bytes()
        try:
            document = copy.deepcopy(self.document)
            document["mirror_epoch"]["post_fix_in_process_step1"] = 7.060197
            RECEIPT.write_text(json.dumps(document), encoding="utf-8")
            self.expect("post-fix")
        finally:
            RECEIPT.write_bytes(original)

    def test_changed_shader_inventory(self):
        original = RECEIPT.read_bytes()
        try:
            document = copy.deepcopy(self.document)
            document["shaders"]["inventory"] = 44
            RECEIPT.write_text(json.dumps(document), encoding="utf-8")
            self.expect("shader inventory")
        finally:
            RECEIPT.write_bytes(original)

    def test_missing_training_arm(self):
        original_top = RECEIPT.read_bytes()
        training = RECEIPT.parent / "training" / "current-qualification.json"
        original_training = training.read_bytes()
        try:
            document = json.loads(training.read_text(encoding="utf-8"))
            document["arms"].pop("A")
            training.write_text(json.dumps(document), encoding="utf-8")
            top = copy.deepcopy(self.document)
            key = "training/current-qualification.json"
            top["subreceipts"]["training"]["sha256"] = digest(training)
            top["evidence"][key] = digest(training)
            RECEIPT.write_text(json.dumps(top), encoding="utf-8")
            self.expect("training arm")
        finally:
            training.write_bytes(original_training)
            RECEIPT.write_bytes(original_top)

    def test_zero_query_dispatch(self):
        original_top = RECEIPT.read_bytes()
        attention = RECEIPT.parent / "attention" / "current-qualification.json"
        original_attention = attention.read_bytes()
        try:
            document = json.loads(attention.read_text(encoding="utf-8"))
            document["attention_backward"]["dispatches"]["twopass_arm"]["query"] = 0
            attention.write_text(json.dumps(document), encoding="utf-8")
            top = copy.deepcopy(self.document)
            key = "attention/current-qualification.json"
            top["subreceipts"]["attention"]["sha256"] = digest(attention)
            top["evidence"][key] = digest(attention)
            RECEIPT.write_text(json.dumps(top), encoding="utf-8")
            self.expect("dispatch")
        finally:
            attention.write_bytes(original_attention)
            RECEIPT.write_bytes(original_top)

    def test_wrong_optimizer_step(self):
        original_top = RECEIPT.read_bytes()
        training = RECEIPT.parent / "training" / "current-qualification.json"
        original_training = training.read_bytes()
        try:
            document = json.loads(training.read_text(encoding="utf-8"))
            document["arms"]["A"]["summary"]["steps"] = 19
            training.write_text(json.dumps(document), encoding="utf-8")
            top = copy.deepcopy(self.document)
            key = "training/current-qualification.json"
            top["subreceipts"]["training"]["sha256"] = digest(training)
            top["evidence"][key] = digest(training)
            RECEIPT.write_text(json.dumps(top), encoding="utf-8")
            self.expect("record count")
        finally:
            training.write_bytes(original_training)
            RECEIPT.write_bytes(original_top)


if __name__ == "__main__":
    unittest.main()
