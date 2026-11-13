#!/usr/bin/env python3
"""Negative controls for the Phase XIX-Z current-source checker.

Each mutation starts from the real receipt and must be rejected before any
expensive sub-qualification rerun. The positive case exercises the full checker
when the receipt and evidence tree are present.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import check_phase19z_current as c


class PackedReleaseControls(unittest.TestCase):
    def test_partial_failed_missing_duplicate_and_forged_routes_are_rejected(self):
        from check_measured_producer_routes import ALL_SPECS
        names=sorted(spec.name for spec in ALL_SPECS)
        valid={"kind":"phase19z-packed-cell-amd-local","status":"PASS","passes":names,"failures":[],
            "results":{name:{"bench":0,"guard":0} for name in names}}
        c.verify_packed_status(valid)
        mutations=[lambda d:d.update(status="PARTIAL"),
            lambda d:d.update(failures=[names[0]]),lambda d:d["passes"].pop(),
            lambda d:d["passes"].__setitem__(0,names[1]),
            lambda d:d["results"].pop(names[0]),
            lambda d:d["results"][names[0]].update(bench=1),
            lambda d:d["results"][names[0]].update(guard=1)]
        for mutate in mutations:
            doc=copy.deepcopy(valid);mutate(doc)
            with self.assertRaises(c.QualificationError):c.verify_packed_status(doc)


class Phase19ZCurrentControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not c.RECEIPT.is_file():
            raise unittest.SkipTest("Phase XIX-Z receipt has not been assembled")
        cls.original = json.loads(c.RECEIPT.read_text(encoding="utf-8"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def reject(self, name, mutate):
        document = copy.deepcopy(self.original)
        mutate(document)
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises((c.QualificationError, KeyError, FileNotFoundError), msg=name):
            c.verify(path, verify_subs=False, require_evidence=False,
                     require_binaries=False)

    def test_positive_receipt_and_evidence(self):
        summary = c.verify(require_binaries=False)
        self.assertEqual(summary["status"], "PASS")
        self.assertGreaterEqual(summary["source_files"], 1500)
        self.assertEqual(summary["counts"]["selected_lr"], 0.001)

    def test_stale_source_binding(self):
        self.reject("stale-source", lambda d: d["source_files"].__setitem__(
            "hierarchos-vulkan/src/transformer.rs", "0" * 64))

    def test_wrong_source_revision(self):
        self.reject("wrong-revision", lambda d: d.__setitem__("source_revision", "f" * 40))

    def test_wrong_status(self):
        self.reject("wrong-status", lambda d: d.__setitem__("status", "FAIL"))

    def test_wrong_shader(self):
        self.reject("wrong-shader", lambda d: d["shader"].__setitem__("sha256", "0" * 64))

    def test_wrong_shader_inventory(self):
        self.reject("wrong-inventory", lambda d: d["shader"].__setitem__("inventory", 44))

    def test_wrong_oracle_archive(self):
        self.reject("wrong-oracle", lambda d: d["oracle"]["shared"].__setitem__(
            "archive_sha256", "0" * 64))

    def test_wrong_attention_accounting(self):
        self.reject("wrong-accounting", lambda d: d["attention"].__setitem__(
            "total_accounted", 92))

    def test_wrong_training_count(self):
        self.reject("wrong-training-count", lambda d: d["counts"].__setitem__(
            "training_arms", 7))

    def test_wrong_selected_lr(self):
        self.reject("wrong-lr", lambda d: d["counts"].__setitem__("selected_lr", 0.003))

    def test_wrong_subreceipt_hash(self):
        self.reject("wrong-subreceipt", lambda d: d["subreceipts"]["drift"].__setitem__(
            "sha256", "0" * 64))

    def test_wrong_binary_hash_when_present(self):
        binary = c.CRATE / "target/release/held_out_eval.exe"
        if not binary.is_file():
            self.skipTest("local executable not present")
        self.reject("wrong-binary", lambda d: d["binaries"].__setitem__(
            "held_out_eval", "0" * 64))


if __name__ == "__main__":
    unittest.main()
