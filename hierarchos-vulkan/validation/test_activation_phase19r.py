#!/usr/bin/env python3
"""Negative controls for the Phase XIX-R instrumented activation receipt."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
RECEIPT = CRATE / "validation/release_parity/multimodal_training/20261110-phase19r-activation"
CHECKER = CRATE / "validation/check_activation_phase19r.py"


class ActivationXIXRControls(unittest.TestCase):
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

    def test_missing_receipt(self):
        (self.dir / "current-qualification.json").unlink()
        result = subprocess.run([sys.executable, str(CHECKER), "--receipt", str(self.dir)],
                                cwd=CRATE, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing receipt", result.stdout + result.stderr)

    def test_unrelated_revision(self):
        self.receipt["source_revision"] = "0" * 40
        self.expect("not an ancestor")

    def test_changed_transformer(self):
        self.receipt["source_files"]["src/transformer.rs"] = "0" * 64
        self.expect("bound source changed: src/transformer.rs")

    def test_changed_tracer(self):
        self.receipt["source_files"]["src/activation_trace.rs"] = "0" * 64
        self.expect("bound source changed: src/activation_trace.rs")

    def test_changed_binary(self):
        self.receipt["binaries"]["activation_probe"] = "0" * 64
        self.expect("binary hash changed: activation_probe")

    def test_mislabeled_precision(self):
        self.receipt["trace_runs"]["initial|off"]["precision"] = "mixed"
        self.expect("trace mode is mislabeled")

    def test_missing_dispatch(self):
        self.receipt["trace_runs"]["initial|mixed"]["dispatches"] = [0, 0, 0]
        self.expect("missing cooperative forward dispatches")

    def test_fp32_dispatched(self):
        self.receipt["trace_runs"]["initial|off"]["dispatches"] = [15, 0, 0]
        self.expect("FP32 arm dispatched cooperative GEMMs")

    def test_changed_partition(self):
        self.receipt["protocol"]["partition_sha256"] = "0" * 64
        self.expect("validation partition changed")

    def test_missing_tensor(self):
        self.receipt["comparisons"]["initial"]["tensors"].pop()
        self.expect("expected 27 tensors")

    def test_tensor_order(self):
        tensors = self.receipt["comparisons"]["initial"]["tensors"]
        tensors[1], tensors[2] = tensors[2], tensors[1]
        self.expect("tensor order changed")

    def test_duplicate_tensor(self):
        tensors = self.receipt["comparisons"]["initial"]["tensors"]
        tensors[2] = dict(tensors[1])
        self.expect("duplicate tensor")

    def test_wrong_shape(self):
        tensor = self.receipt["comparisons"]["initial"]["tensors"][1]
        tensor["shape"] = [8, 63, tensor["shape"][2]]
        self.expect("element count does not match shape")

    def test_tampered_trace_digest(self):
        path = self.dir / "evidence" / "trace-initial-mixed.json"
        report = json.loads(path.read_text())
        report["tensors"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(report), encoding="utf-8")
        self.expect("evidence digest changed: trace-initial-mixed")

    def test_forged_statistics(self):
        tensor = self.receipt["comparisons"]["initial"]["tensors"][2]
        tensor["err_sumsq"] *= 4.0
        self.expect("forged RMS")

    def test_checkpoint_identity(self):
        self.receipt["checkpoints"]["step1-mixed"]["weights_sha256"] = \
            self.receipt["checkpoints"]["step1-fp32"]["weights_sha256"]
        self.expect("checkpoints step1-fp32 and step1-mixed are identical")

    def test_incomplete_matrix(self):
        self.receipt["six_cell"]["cells"].pop("step1-mixed|mixed")
        self.expect("incomplete cross-evaluation matrix")

    def test_decomposition(self):
        self.receipt["six_cell"]["residual"] = 1e-3
        self.expect("decomposition residual is nonzero")

    def test_raw_provenance_missing(self):
        self.receipt["limitations"] = ["nothing to see"]
        self.expect("raw-capture provenance")

    def test_historical_receipt_presented_as_current(self):
        self.receipt["kind"] = "phase18-wmma-drift-investigation"
        self.expect("wrong receipt kind")

    def test_post_fix_still_wrong(self):
        self.receipt["anomaly"]["post_fix_in_process_step1"] = 7.0
        self.expect("post-fix in-process evaluation still disagrees")

    def test_step100_gap_missing(self):
        self.receipt["anomaly"]["step100_fresh_mixed"] = \
            self.receipt["anomaly"]["step100_fresh_fp32"]
        self.expect("step-100 drift is not reproduced")


if __name__ == "__main__":
    unittest.main()
