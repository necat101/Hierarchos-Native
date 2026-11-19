#!/usr/bin/env python3
"""Positive and negative controls for the Phase LXV gate.

Every mutation starts from the genuinely passing evidence and must be rejected.
Mutated copies live in a temporary directory; no canonical receipt, capture,
listing or shader in the tree is written by this suite.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import check_phase_lxv as c


class PhaseLxvControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="phase-lxv-")
        cls.tree = Path(cls.tmp.name) / "tree"
        # Only the files the gate reads need to exist.
        for relative in (
            c.SHADERS["wmma_gemm_f16_f32acc"]["source"],
            c.SHADERS["wmma_gemm_f16_f32acc"]["binary"],
            c.SHADERS["wmma_gemm_f16_f16acc"]["source"],
            c.SHADERS["wmma_gemm_f16_f16acc"]["binary"],
        ):
            target = cls.tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c.ROOT / relative, target)
        cls.evidence_root = Path(cls.tmp.name) / "evidence-src"
        shutil.copytree(c.EVIDENCE, cls.evidence_root)
        cls.evidence = Path(cls.tmp.name) / "evidence"
        cls.receipt = json.loads(c.RECEIPT.read_text(encoding="utf-8"))
        c.EVIDENCE = cls.evidence

    @classmethod
    def tearDownClass(cls):
        c.EVIDENCE = c.HERE / "phase_lxv_evidence"
        cls.tmp.cleanup()

    def setUp(self):
        # Every control starts from an untouched copy of the evidence.
        if self.evidence.exists():
            shutil.rmtree(self.evidence)
        shutil.copytree(self.evidence_root, self.evidence)

    # -- helpers ---------------------------------------------------------
    def forged(self) -> dict:
        return json.loads(json.dumps(self.receipt))

    def assertRejected(self, *, tree=None, receipt=None, message="the gate accepted forged evidence"):
        with self.assertRaises(ValueError, msg=message):
            c.validate(self.tree if tree is None else tree, receipt=receipt)

    def edit_json(self, path: Path, mutate):
        # Accuracy receipts end with a human-readable "status: PASS" line, so the
        # object is read with raw_decode rather than json.loads.
        data = json.JSONDecoder().raw_decode(path.read_text(encoding="utf-8"))[0]
        mutate(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    # -- positive --------------------------------------------------------
    def test_genuine_evidence_passes(self):
        c.validate(self.tree)

    # -- hardware / RGP --------------------------------------------------
    def test_forged_static_wmma_count(self):
        def mutate(data):
            for code_object in data["final-full"]["code_objects"]:
                if code_object["wmma_total"] == 32:
                    code_object["wmma_total"] = 4
        self.edit_json(self.evidence / "hardware.json", mutate)
        self.assertRejected(message="a forged static WMMA count was accepted")

    def test_missing_capture_digest(self):
        def mutate(data):
            data["final-full"]["profile_sha256"] = ""
        self.edit_json(self.evidence / "hardware.json", mutate)
        self.assertRejected(message="a capture without a digest was accepted")

    def test_edited_isa_listing(self):
        listing = next(
            self.evidence.glob("isa/final-full_*obj04*.sas")
        )
        text = listing.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if "v_wmma_f16_16x16x16_f16" not in line]
        listing.write_text("\n".join(lines), encoding="utf-8")
        self.assertRejected(message="an edited ISA listing was accepted")

    def test_swapped_code_object(self):
        source = next(self.evidence.glob("isa/main-full_*obj04*.elf"))
        target = next(self.evidence.glob("isa/final-full_*obj04*.elf"))
        shutil.copyfile(source, target)
        self.assertRejected(message="a swapped code object was accepted")

    def test_declared_16_bank_implementation(self):
        receipt = self.forged()
        receipt["sixteen_bank"]["implemented"] = True
        self.assertRejected(receipt=receipt,
                            message="an undeclared 16-bank build was accepted")

    # -- accuracy --------------------------------------------------------
    def test_relaxed_limit(self):
        path = self.evidence / "accuracy" / "seed1593426484_full-fp16_tile16.json"
        self.edit_json(path, lambda data: data["results"].__getitem__(0).__setitem__(
            "max_abs_limit", 0.5))
        self.assertRejected(message="a widened numerical limit was accepted")

    def test_forged_pass_with_failing_value(self):
        path = self.evidence / "accuracy" / "seed1593426484_full-fp16_tile16.json"

        def mutate(data):
            row = data["results"][0]
            row["passed"] = True
            row["max_abs"] = row["max_abs_limit"] * 2.0
        self.edit_json(path, mutate)
        self.assertRejected(message="a gate marked passed while over its limit was accepted")

    def test_forged_failure_count(self):
        path = self.evidence / "accuracy" / "seed777_full-fp16_tile16.json"
        self.edit_json(path, lambda data: data.__setitem__("failed", 0))
        self.assertRejected(message="a receipt that hides its own failure was accepted")

    def test_dropped_seed(self):
        (self.evidence / "accuracy" / "seed314159_mixed_tile16.json").unlink()
        self.assertRejected(message="a missing seed was accepted")

    def test_margin_target_claim(self):
        receipt = self.forged()
        receipt["accuracy_margin"]["target_met"] = True
        self.assertRejected(receipt=receipt,
                            message="a false margin-target claim was accepted")

    def test_margin_threshold_moved(self):
        receipt = self.forged()
        receipt["accuracy_margin"]["target_headroom"] = 0.05
        self.assertRejected(receipt=receipt,
                            message="a weakened margin threshold was accepted")

    def test_dropped_original_gate(self):
        receipt = self.forged()
        receipt["accuracy_margin"]["gates"] = receipt["accuracy_margin"]["gates"][1:]
        self.assertRejected(receipt=receipt,
                            message="a dropped margin gate was accepted")

    # -- performance -----------------------------------------------------
    def test_altered_statistic(self):
        receipt = self.forged()
        name = next(iter(receipt["performance"]))
        shape = next(iter(receipt["performance"][name]["shapes"]))
        receipt["performance"][name]["shapes"][shape]["mean_ratio"] += 0.25
        self.assertRejected(receipt=receipt,
                            message="an edited ABBA statistic was accepted")

    def test_missing_raw_samples(self):
        directory = next(self.evidence.glob("abba_*"))
        samples = next(directory.glob("*_samples.jsonl*"))
        samples.unlink()
        self.assertRejected(message="missing raw benchmark data was accepted")

    def test_relabelled_verdict(self):
        receipt = self.forged()
        for entry in receipt["performance"].values():
            if entry.get("production"):
                entry["verdict"] = "NON-INFERIOR"
                entry["gate_satisfied"] = True
        self.assertRejected(receipt=receipt,
                            message="an unreproducible non-inferiority verdict was accepted")

    def test_changed_dispatch_accounting(self):
        receipt = self.forged()
        receipt["dispatch"]["cooperative_dispatches"] += 1
        self.assertRejected(receipt=receipt,
                            message="a forged dispatch count was accepted")

    # -- scope -----------------------------------------------------------
    def test_unapproved_source_file(self):
        with self.assertRaises(ValueError):
            c.validate(self.tree, changed=["hierarchos-vulkan/src/vulkan.rs"])

    def test_approved_scope_passes(self):
        c.validate(self.tree, changed=["hierarchos-vulkan/validation/check_phase_lxv.py",
                                       "PHASE_LXV_ACCURACY.md"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
