#!/usr/bin/env python3
"""Positive and negative controls for the Phase LXV gate.

Every mutation starts from the genuinely passing evidence and must be rejected.
Mutated copies live in a temporary directory; no canonical receipt, capture,
listing or shader in the tree is written by this suite.
"""
from __future__ import annotations

import json
import shutil
import subprocess
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


class PhaseLxvHistoricalScope(unittest.TestCase):
    """The scope control is about Phase LXV's own commits, not its descendants.

    The gate used to diff `base..HEAD`. On a branch that inherits later phases
    that range grows with every release commit, so legitimate work this phase
    never touched (the release `.gitattributes` rule, `phase_lxxxi_evidence/**`,
    the P3 shaders) was reported as an unapproved Phase LXV change. These
    controls pin the corrected contract: the historical range is read from the
    two recorded endpoints and still rejected when it contains an unapproved
    path, while a later release commit is not part of that range at all.
    """

    @staticmethod
    def git(*args) -> str:
        result = subprocess.run(["git", "-C", str(c.ROOT), *args],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()[:200]}")
        return result.stdout

    def later_scope_paths(self):
        return [line for line in self.git("diff", "--name-only",
                                          c.PHASE_LXV_ENDPOINT, "HEAD").splitlines()
                if line.strip()]

    def test_endpoint_is_a_committed_revision(self):
        for revision in (c.PHASE_LXV_SCOPE_BASE, c.PHASE_LXV_ENDPOINT):
            self.git("cat-file", "-e", revision + "^{commit}")
        ancestry = subprocess.run(
            ["git", "-C", str(c.ROOT), "merge-base", "--is-ancestor",
             c.PHASE_LXV_SCOPE_BASE, c.PHASE_LXV_ENDPOINT], capture_output=True)
        self.assertEqual(ancestry.returncode, 0,
                         "the Phase LXV base is not an ancestor of its endpoint")

    def test_historical_range_is_only_phase_lxv_paths(self):
        paths = c.historical_scope_paths(c.ROOT)
        self.assertTrue(paths, "the recorded Phase LXV range is empty")
        approvals = tuple(prefix for prefix in c.APPROVED_CHANGES)
        for path in paths:
            if path.startswith("hierarchos-vulkan/validation/fixtures/"):
                continue
            self.assertTrue(
                any(path.startswith(prefix) or path == prefix.rstrip("/")
                    for prefix in approvals),
                f"the recorded Phase LXV range contains an unapproved path: {path}")

    def test_historical_range_passes_the_scope_check(self):
        c.check_scope(c.historical_scope_paths(c.ROOT))

    def test_later_release_commit_is_not_a_scope_violation(self):
        historical = c.historical_scope_paths(c.ROOT)
        later = self.later_scope_paths()
        # The release commit adds the byte-preserving checkout rule and the
        # Phase LXXXI evidence tree; neither belongs to Phase LXV.
        self.assertNotIn(".gitattributes", historical)
        self.assertIn(".gitattributes", later)
        self.assertTrue(any(path.startswith("phase_lxxxi_evidence/")
                            for path in later),
                        "the later release scope does not contain the Phase LXXXI evidence")
        # The corrected gate accepts the historical range...
        c.check_scope(historical)
        # ...while that same descendant diff is exactly what used to be misread.
        with self.assertRaises(ValueError):
            c.check_scope(later)

    def test_unapproved_path_inside_the_historical_range_still_fails(self):
        # The original negative control is retained: scope is not waived for a
        # source-file modification, and folding a source path into the
        # historical list is still rejected.
        with self.assertRaises(ValueError):
            c.check_scope(c.historical_scope_paths(c.ROOT)
                          + ["hierarchos-vulkan/src/vulkan.rs"])

    def test_unavailable_range_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="phase-lxv-nogit-") as scratch:
            with self.assertRaises(ValueError):
                c.historical_scope_paths(Path(scratch))


if __name__ == "__main__":
    unittest.main(verbosity=2)
