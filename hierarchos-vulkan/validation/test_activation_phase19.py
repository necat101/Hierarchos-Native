#!/usr/bin/env python3
"""Negative controls for the Phase XIX activation receipt checker."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
RECEIPT = CRATE / "validation/release_parity/multimodal_training/20261110-phase19-activation-trace"
CHECKER = CRATE / "validation/check_activation_phase19.py"


class ActivationControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name) / "receipt"
        shutil.copytree(RECEIPT, self.dir)
        self.receipt = json.loads((self.dir / "current-qualification.json").read_text())

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.dir / "current-qualification.json").write_text(
            json.dumps(self.receipt, indent=2), encoding="utf-8")

    def expect(self, fragment):
        self.write()
        result = subprocess.run([sys.executable, str(CHECKER), "--receipt", str(self.dir)],
                                cwd=CRATE, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "checker accepted a forged receipt")
        self.assertIn(fragment, result.stdout + result.stderr)

    def test_changed_weights_identity(self):
        self.receipt["checkpoints"]["step1-fp32"]["weights_sha256"] = \
            self.receipt["checkpoints"]["initial"]["weights_sha256"]
        self.expect("step-1 FP32 arm did not update")

    def test_mislabeled_mode(self):
        self.receipt["trace_runs"]["initial|off"]["precision"] = "mixed"
        self.expect("trace mode is mislabeled")

    def test_missing_dispatch(self):
        self.receipt["trace_runs"]["initial|mixed"]["dispatches"] = [0, 0, 0]
        self.expect("missing cooperative dispatches")

    def test_fp32_dispatched(self):
        self.receipt["trace_runs"]["initial|off"]["dispatches"] = [1, 0, 0]
        self.expect("FP32 arm dispatched")

    def test_tensor_order(self):
        tensors = self.receipt["comparisons"]["initial"]["tensors"]
        tensors[0], tensors[1] = tensors[1], tensors[0]
        self.expect("trace tensor order changed")

    def test_missing_tensor(self):
        self.receipt["comparisons"]["initial"]["tensors"].pop()
        self.expect("comparison tensor count changed")

    def test_changed_shape(self):
        tensor = self.receipt["comparisons"]["initial"]["tensors"][0]
        tensor["shape"] = [8, 63, tensor["shape"][2]]
        self.expect("trace tensor order changed")

    def test_tampered_tensor_digest(self):
        evidence = next(item for item in self.receipt["evidence"]
                        if item["name"] == "trace-initial-off")
        path = self.dir / "evidence" / f"{evidence['name']}.json"
        value = json.loads(path.read_text())
        value["tensors"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        self.expect("evidence digest changed")

    def test_forged_rms(self):
        tensor = self.receipt["comparisons"]["initial"]["tensors"][2]
        tensor["err_sumsq"] *= 4.0
        self.expect("forged RMS")

    def test_equal_exceeds_elements(self):
        tensor = self.receipt["comparisons"]["initial"]["tensors"][0]
        tensor["equal"] = tensor["elements"] + 1
        self.expect("equal count exceeds elements")

    def test_incomplete_matrix(self):
        self.receipt["six_cell"]["cells"].pop("step1-fp32|off")
        self.expect("incomplete cross-evaluation matrix")

    def test_decomposition_does_not_close(self):
        self.receipt["six_cell"]["residual"] += 0.1
        self.expect("decomposition residual is nonzero")

    def test_step1_arms_swapped(self):
        self.receipt["checkpoints"]["step1-mixed"]["weights_sha256"] = \
            self.receipt["checkpoints"]["step1-fp32"]["weights_sha256"]
        self.expect("step-1 arms swapped")

    def test_missing_evidence(self):
        (self.dir / "evidence" / "trace-initial-off.json").unlink()
        self.expect("missing evidence file")

    def test_bad_ancestry(self):
        self.receipt["source_revision"] = "0" * 40
        self.expect("not an ancestor")

    def test_dirty_source(self):
        self.receipt["source_clean"] = False
        self.expect("source tree was not clean")

    def test_forged_status(self):
        self.receipt["comparisons"].pop("initial")
        self.receipt["status"] = "PASS"
        self.expect("comparison set does not match")


if __name__ == "__main__":
    unittest.main()
