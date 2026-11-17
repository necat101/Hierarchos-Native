"""Fault injection for the evidence digest guard.

Every fixture here is a temporary directory. No test mutates a canonical evidence
root: the regression case reads the Phase XV root, asserts what the guard finds there,
and never writes to it.
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import check_evidence_integrity as g

REPO = Path(__file__).resolve().parents[2]
PHASE15 = (REPO / "hierarchos-vulkan" / "validation" / "release_parity" /
           "multimodal_training" / "20261108-rdna3-phase15-qualification")
# The newest sealed root, discovered by the same rule `--latest-release` uses.
# Pinning a phase directory here made this guard keep auditing a superseded root
# after the Phase XLI migration, so it failed on source drift that belonged to
# the older seal rather than to the current one.
SEALED = sorted((REPO / "hierarchos-vulkan" / "validation" /
                 "release_parity").rglob("release-qualification.json"))
RELEASE = SEALED[-1] if SEALED else (REPO / "hierarchos-vulkan" / "validation" /
                                     "release_parity" / "multimodal_training" /
                                     "no-sealed-release")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(tmp: Path, *, evidence=b'{"pass": true}\n',
          family=b'{"pass": true, "worst_logit": 1.6e-07}\n') -> tuple[Path, Path]:
    """A minimal but structurally real root: evidence map + retained family hash."""
    root = tmp / "root"
    (root / "evidence").mkdir(parents=True)
    (root / "retained" / "fam").mkdir(parents=True)
    ev = root / "evidence" / "a.json"
    ev.write_bytes(evidence)
    fam = root / "retained" / "fam" / "qualification.json"
    fam.write_bytes(family)
    receipt = {
        "schema": 1,
        "pass": True,
        "evidence": {"a.json": sha(ev)},
        "retained": {"fam": {"qualification_sha256": sha(fam), "worst_logit": 1.6e-07}},
        "sources": {},
    }
    path = root / "current-qualification.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return root, path


def rewrite(path: Path, **changes):
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.update(changes)
    path.write_text(json.dumps(doc), encoding="utf-8")


class DigestGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="digest-guard-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def audit(self, root, receipt):
        return g.audit_receipt(receipt, root, REPO)

    # 1. A valid evidence root passes.
    def test_valid_root_has_no_failures(self):
        root, receipt = build(self.tmp)
        report = self.audit(root, receipt)
        self.assertEqual(report.failures, [])
        self.assertEqual(report.count(g.VALID), 2)
        self.assertNotIn("integrity_mismatch", report.flags)

    # 2. A referenced file modified by one byte fails.
    def test_one_byte_change_is_a_mismatch(self):
        root, receipt = build(self.tmp)
        (root / "evidence" / "a.json").write_bytes(b'{"pass": false}\n')
        report = self.audit(root, receipt)
        self.assertEqual(report.count(g.MISMATCH), 1)
        self.assertIn("integrity_mismatch", report.flags)

    # 3. A declared digest changed while the file stays unchanged fails.
    def test_declared_digest_change_is_a_mismatch(self):
        root, receipt = build(self.tmp)
        rewrite(receipt, evidence={"a.json": "0" * 64})
        self.assertEqual(self.audit(root, receipt).count(g.MISMATCH), 1)

    # 4. A referenced file removed fails.
    def test_removed_file_is_missing(self):
        root, receipt = build(self.tmp)
        (root / "evidence" / "a.json").unlink()
        report = self.audit(root, receipt)
        self.assertEqual(report.count(g.MISSING), 1)
        self.assertIn("evidence_missing", report.flags)

    # 5. An aggregate digest that disagrees with its family receipt fails.
    def test_family_receipt_disagreement_is_a_mismatch(self):
        root, receipt = build(self.tmp)
        (root / "retained" / "fam" / "qualification.json").write_bytes(
            b'{"pass": true, "worst_logit": 1.4e-07}\n')
        report = self.audit(root, receipt)
        retained = [f for f in report.findings if f.kind == "retained"]
        self.assertEqual([f.status for f in retained], [g.MISMATCH])

    # 6. A retained qualification digest that disagrees with the actual receipt fails.
    def test_retained_digest_disagreement_is_a_mismatch(self):
        root, receipt = build(self.tmp)
        doc = json.loads(receipt.read_text(encoding="utf-8"))
        doc["retained"]["fam"]["qualification_sha256"] = "f" * 64
        receipt.write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(self.audit(root, receipt).count(g.MISMATCH), 1)

    # 7. An invalid digest format fails.
    def test_invalid_digest_format(self):
        root, receipt = build(self.tmp)
        rewrite(receipt, evidence={"a.json": "not-a-digest"})
        report = self.audit(root, receipt)
        self.assertEqual(report.count(g.INVALID), 1)

    # 8. A path traversal attempt fails.
    def test_path_traversal_is_unsafe(self):
        root, receipt = build(self.tmp)
        (self.tmp / "outside.json").write_text("{}", encoding="utf-8")
        rewrite(receipt, evidence={"../outside.json": sha(self.tmp / "outside.json")})
        report = self.audit(root, receipt)
        self.assertEqual(report.count(g.UNSAFE), 1)

    # 9. A symlink escaping the root fails.
    def test_symlink_escape_is_unsafe(self):
        root, receipt = build(self.tmp)
        outside = self.tmp / "outside.json"
        outside.write_text('{"pass": true}\n', encoding="utf-8")
        link = root / "evidence" / "link.json"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this platform")
        rewrite(receipt, evidence={"link.json": sha(outside)})
        report = self.audit(root, receipt)
        statuses = {f.status for f in report.findings if "link.json" in f.label}
        self.assertTrue(statuses & {g.UNSAFE, g.MISSING},
                        f"escaping symlink was accepted: {statuses}")

    # 10. An unsupported schema is explicitly reported.
    def test_unsupported_schema_is_reported(self):
        root = self.tmp / "root"
        root.mkdir()
        receipt = root / "current-qualification.json"
        receipt.write_text(json.dumps({"schema": 9, "pass": True}), encoding="utf-8")
        report = self.audit(root, receipt)
        self.assertEqual(report.count(g.UNSUPPORTED), 1)

    # 11. Content absent from the path's history is distinguished from a mismatch.
    def test_history_absence_is_distinct_from_mismatch(self):
        tracked = REPO / "RDNA3_RELEASE_CONTRACT.md"
        if not tracked.exists():
            self.skipTest("contract file missing")
        current = sha(tracked)
        self.assertTrue(g.history_has("RDNA3_RELEASE_CONTRACT.md", current, REPO))
        self.assertFalse(g.history_has("RDNA3_RELEASE_CONTRACT.md", "b" * 64, REPO))

    # 12. A source SHA mismatch is not mistaken for a digest pass.
    def test_source_drift_is_not_a_pass(self):
        repo = self.tmp / "repo"
        repo.mkdir()
        (repo / "src.txt").write_text("actual\n", encoding="utf-8")
        root, receipt = build(self.tmp)
        rewrite(receipt, sources={"src.txt": "c" * 64})
        report = g.audit_receipt(receipt, root, repo)
        self.assertEqual(report.count(g.SOURCE_DRIFT), 1)
        self.assertIn(g.SOURCE_DRIFT, {f.status for f in report.failures})

    # 13. Digest-consistent evidence without preserved provenance is not "verified".
    def test_provenance_is_reported_separately(self):
        root, receipt = build(self.tmp)
        rewrite(receipt, binaries={"probe": "d" * 64})
        report = self.audit(root, receipt)
        self.assertEqual(report.failures, [])
        self.assertEqual(report.count(g.PROVENANCE), 1)
        self.assertIn("provenance_unsupported", report.flags)

    # 14. The current release root passes strict validation.
    def test_current_release_passes_strict(self):
        if not RELEASE.exists():
            self.skipTest("unified release not sealed")
        self.assertEqual(g.main(["--release", str(RELEASE), "--strict"]), 0)

    # CLI strictness: a mismatch must produce a failing exit status.
    def test_strict_cli_fails_on_mismatch(self):
        root, receipt = build(self.tmp)
        (root / "evidence" / "a.json").write_bytes(b"tampered\n")
        self.assertEqual(g.main(["--root", str(root), "--strict"]), 1)

    def test_strict_cli_passes_on_valid_root(self):
        root, _ = build(self.tmp)
        self.assertEqual(g.main(["--root", str(root), "--strict"]), 0)


class PhaseFifteenRegression(unittest.TestCase):
    """The 9e636fb rebinding must stay visible, and the guard must detect it."""

    def test_rebound_root_is_flagged(self):
        if not PHASE15.exists():
            self.skipTest("Phase XV root not present")
        report = g.audit_receipt(PHASE15 / "current-qualification.json", PHASE15, REPO)
        evidence = [f for f in report.findings if f.kind == "evidence"]
        retained = [f for f in report.findings if f.kind == "retained"]
        self.assertGreaterEqual(sum(f.status == g.MISMATCH for f in evidence), 12,
                                "the rebinding's evidence map no longer trips the guard")
        self.assertGreaterEqual(sum(f.status == g.MISMATCH for f in retained), 6,
                                "the rebinding's family hashes no longer trip the guard")
        self.assertIn("integrity_mismatch", report.flags)
        self.assertIn("provenance_unsupported", report.flags)

    def test_guard_depends_on_evidence_map_validation(self):
        """Negative control: disabling the evidence-map check hides the defect."""
        if not PHASE15.exists():
            self.skipTest("Phase XV root not present")
        original = g.check_map
        try:
            g.check_map = lambda *a, **k: None
            report = g.audit_receipt(PHASE15 / "current-qualification.json", PHASE15, REPO)
        finally:
            g.check_map = original
        evidence = [f for f in report.findings if f.kind == "evidence"]
        self.assertEqual([f for f in evidence if f.status == g.MISMATCH], [],
                         "the negative control did not disable the evidence-map check")

    def test_guard_depends_on_digest_resolution(self):
        """Negative control: accepting every binding hides every mismatch."""
        if not PHASE15.exists():
            self.skipTest("Phase XV root not present")
        original = g.resolve_binding
        try:
            g.resolve_binding = lambda *a, **k: (g.VALID, "", "")
            report = g.audit_receipt(PHASE15 / "current-qualification.json", PHASE15, REPO)
        finally:
            g.resolve_binding = original
        self.assertEqual(report.count(g.MISMATCH), 0,
                         "the negative control did not disable digest resolution")


if __name__ == "__main__":
    unittest.main()
