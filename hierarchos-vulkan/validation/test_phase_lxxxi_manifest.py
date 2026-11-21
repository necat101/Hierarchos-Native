"""Fault injection for the Phase LXXXI manifest gate.

Every fixture is a throwaway Git repository under a temporary directory: real
commits, real blobs, real bytes on disk. Nothing here mocks SHA-256 and nothing
replaces Git plumbing with a fabricated answer, so a control only passes when
the gate genuinely fails for the reason it claims.

The last class runs the gate against this checkout, because that is the thing
the release actually depends on.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import check_phase_lxxxi_manifest as gate  # noqa: E402

REPO = HERE.parents[1]
CHECKER = HERE / "check_phase_lxxxi_manifest.py"
EVIDENCE = gate.EVIDENCE_DIR
MANIFEST = gate.MANIFEST_NAME

# The frozen Phase LXXXI manifest. It is never regenerated; the gate pins it so a
# "fix" that replaces the manifest with one derived from the repaired bytes is
# rejected instead of silently accepted.
FROZEN_MANIFEST_SHA256 = "a7979f4f448ab3afa1598155d0eb708e8d36e4bed23457fa5dd515fcd22bbff4"

DEFAULT_FILES: dict[str, bytes] = {
    "PHASE_LXXXI_RELEASE_REPORT.md": b"# Phase LXXXI\r\n\r\nrelease report\r\n",
    "analysis.log": b"== abba_main_vs_p3_24p\r\nshape ratio verdict\r\n",
    "final_rerun.log": b"=== DONE\nanalysis_exit=0\n",
    "abba_main_vs_p3_24p/index.tsv": b"shape\tpairs\r\n128x8192x768\t24\r\n",
    "abba_main_vs_p3_24p/p10_base_meta.json": b'{"shape": "128x8192x768"}\r\n',
    "receipts/summary.json": b'{"verdict": "PARITY-OR-BETTER"}\n',
}


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({completed.returncode}): {completed.stderr}"
        )
    return completed


def init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(root)], check=True, capture_output=True
    )
    git(root, "config", "user.email", "phase-lxxxi-gate@example.invalid")
    git(root, "config", "user.name", "Phase LXXXI gate")
    # The fixtures must not depend on the host's line-ending policy; the real
    # defect was a checkout policy, so pin the fixture to store raw bytes.
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "commit.gpgsign", "false")


def write_manifest(evidence: Path, files: dict[str, bytes]) -> None:
    lines = [
        f"{hashlib.sha256(files[rel]).hexdigest()}  {rel}" for rel in sorted(files)
    ]
    (evidence / MANIFEST).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def build_repo(
    tmp: str | Path,
    files: dict[str, bytes] | None = None,
    *,
    attributes: bool = True,
    commit: bool = True,
) -> Path:
    files = dict(DEFAULT_FILES if files is None else files)
    root = Path(tmp) / "repo"
    init_repo(root)
    (root / ".gitignore").write_bytes(b"*.log\n")
    if attributes:
        (root / ".gitattributes").write_bytes(b"phase_lxxxi_evidence/** -text\n")
    evidence = root / EVIDENCE
    evidence.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        target = evidence / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_manifest(evidence, files)
    if commit:
        git(root, "add", "-A", "-f")
        git(root, "commit", "-q", "-m", "phase lxxxi evidence")
    return root


def run_checker(root: Path, *extra: str) -> tuple[int, dict, str]:
    completed = subprocess.run(
        [sys.executable, str(CHECKER), "--repo-root", str(root), "--json", *extra],
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    if not stdout.startswith("{"):
        raise AssertionError(
            f"checker produced no JSON report (exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.returncode, json.loads(stdout), completed.stderr


def rewrite_manifest(root: Path, transform) -> None:
    path = root / EVIDENCE / MANIFEST
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the file's own trailing newline, not an entry
    path.write_text("\n".join(transform(lines)) + "\n", encoding="utf-8")


def make_directory_link(link: Path, target: Path) -> str:
    """Create a directory symlink, or a Windows junction where symlinks need a privilege."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return "junction"
    raise unittest.SkipTest("directory links are unavailable on this platform")


class GateCase(unittest.TestCase):
    def assert_passes(self, root: Path, *extra: str) -> dict:
        code, payload, stderr = run_checker(root, *extra)
        self.assertEqual(
            code,
            0,
            msg=f"expected PASS, failures={payload.get('failures')} stderr={stderr}",
        )
        self.assertTrue(payload["ok"])
        return payload

    def assert_fails(self, root: Path, needle: str, *extra: str) -> dict:
        code, payload, stderr = run_checker(root, *extra)
        self.assertEqual(
            code,
            1,
            msg=f"expected FAIL, payload={payload} stderr={stderr}",
        )
        self.assertFalse(payload["ok"])
        joined = "\n".join(payload["failures"])
        self.assertIn(needle, joined, msg=f"missing reason {needle!r} in:\n{joined}")
        return payload


class PositiveControl(GateCase):
    def test_valid_inventory_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            payload = self.assert_passes(root)
            total = len(DEFAULT_FILES)
            self.assertEqual(payload["entries"], total)
            self.assertEqual(payload["tracked_evidence_paths"], total)
            self.assertEqual(payload["working_tree"]["verified"], total)
            self.assertEqual(payload["working_tree"]["failed"], 0)
            self.assertEqual(payload["git_blob"]["verified"], total)
            self.assertEqual(payload["git_blob"]["failed"], 0)
            self.assertEqual(payload["attributes"]["verified"], total)
            self.assertEqual(payload["attributes"]["failed"], 0)
            self.assertEqual(payload["inventory"]["missing_from_tree"], [])
            self.assertEqual(payload["inventory"]["unlisted_in_tree"], [])
            self.assertEqual(payload["inventory"]["extra_on_disk"], [])

    def test_manifest_pin_accepts_the_real_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            digest = hashlib.sha256((root / EVIDENCE / MANIFEST).read_bytes()).hexdigest()
            payload = self.assert_passes(root, "--expect-manifest-sha256", digest)
            self.assertEqual(payload["manifest_sha256"], digest)


class WorkingTreeControls(GateCase):
    def test_modified_evidence_bytes_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            target = root / EVIDENCE / "abba_main_vs_p3_24p" / "index.tsv"
            target.write_bytes(b"shape\tpairs\r\n512x768x768\t24\r\n")
            payload = self.assert_fails(
                root, "working tree: abba_main_vs_p3_24p/index.tsv hashes"
            )
            # The committed blob is untouched, so only the working-tree side fails.
            self.assertEqual(payload["working_tree"]["failed"], 1)
            self.assertEqual(payload["git_blob"]["failed"], 0)

    def test_missing_tracked_evidence_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            (root / EVIDENCE / "receipts" / "summary.json").unlink()
            payload = self.assert_fails(root, "working tree: missing receipts/summary.json")
            self.assertEqual(payload["working_tree"]["failed"], 1)

    def test_unreadable_file_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            # A directory where the manifest expects a file is unreadable as bytes.
            target = root / EVIDENCE / "receipts" / "summary.json"
            target.unlink()
            target.mkdir()
            self.assert_fails(root, "receipts/summary.json")


class ManifestControls(GateCase):
    def test_modified_manifest_digest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)

            def swap(lines):
                return [
                    f"{'0' * 64}  receipts/summary.json"
                    if line.endswith("  receipts/summary.json")
                    else line
                    for line in lines
                ]

            rewrite_manifest(root, swap)
            self.assert_fails(root, "manifest says " + "0" * 64)

    def test_malformed_digest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)

            def truncate(lines):
                return [
                    f"{'a' * 63}  receipts/summary.json"
                    if line.endswith("  receipts/summary.json")
                    else line
                    for line in lines
                ]

            rewrite_manifest(root, truncate)
            self.assert_fails(root, "malformed SHA-256 digest")

    def test_uppercase_digest_is_not_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)

            def upper(lines):
                return [
                    f"{'A' * 64}  receipts/summary.json"
                    if line.endswith("  receipts/summary.json")
                    else line
                    for line in lines
                ]

            rewrite_manifest(root, upper)
            self.assert_fails(root, "malformed SHA-256 digest")

    def test_duplicate_manifest_entry_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)

            def duplicate(lines):
                entry = next(l for l in lines if l.endswith("  analysis.log"))
                return lines + [entry]

            rewrite_manifest(root, duplicate)
            self.assert_fails(root, "duplicate path 'analysis.log'")

    def test_missing_manifest_entry_for_committed_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)

            def drop(lines):
                return [l for l in lines if not l.endswith("  receipts/summary.json")]

            rewrite_manifest(root, drop)
            payload = self.assert_fails(
                root, "committed evidence file is absent from the manifest: receipts/summary.json"
            )
            self.assertEqual(
                payload["inventory"]["unlisted_in_tree"], ["receipts/summary.json"]
            )

    def test_manifest_lists_a_path_absent_from_the_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)

            def ghost(lines):
                return lines + [f"{'b' * 64}  receipts/ghost.json"]

            rewrite_manifest(root, ghost)
            payload = self.assert_fails(
                root, "manifest lists a path absent from the committed tree: receipts/ghost.json"
            )
            self.assertEqual(
                payload["inventory"]["missing_from_tree"], ["receipts/ghost.json"]
            )

    def test_unlisted_tracked_evidence_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            extra = root / EVIDENCE / "receipts" / "late.json"
            extra.write_bytes(b'{"late": true}\n')
            git(root, "add", "-f", "--", f"{EVIDENCE}/receipts/late.json")
            git(root, "commit", "-q", "-m", "add unlisted evidence")
            payload = self.assert_fails(
                root, "committed evidence file is absent from the manifest: receipts/late.json"
            )
            self.assertEqual(payload["inventory"]["unlisted_in_tree"], ["receipts/late.json"])

    def test_extra_untracked_file_on_disk_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            (root / EVIDENCE / "scratch.log").write_bytes(b"not evidence\n")
            self.assert_fails(root, "unexpected file in phase_lxxxi_evidence/: scratch.log")


class PathSafetyControls(GateCase):
    def test_path_traversal_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            rewrite_manifest(root, lambda lines: lines + [f"{'c' * 64}  ../escape.txt"])
            self.assert_fails(root, "unsafe path '../escape.txt'")

    def test_absolute_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            rewrite_manifest(root, lambda lines: lines + [f"{'d' * 64}  /etc/passwd"])
            self.assert_fails(root, "unsafe path '/etc/passwd'")

    def test_nested_traversal_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            rewrite_manifest(
                root, lambda lines: lines + [f"{'e' * 64}  receipts/../../escape.txt"]
            )
            self.assert_fails(root, "unsafe path 'receipts/../../escape.txt'")

    def test_directory_escape_is_rejected_even_when_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            outside = Path(tmp) / "outside"
            (outside / "abba_main_vs_p3_24p").mkdir(parents=True)
            for rel in ("index.tsv", "p10_base_meta.json"):
                source = root / EVIDENCE / "abba_main_vs_p3_24p" / rel
                (outside / "abba_main_vs_p3_24p" / rel).write_bytes(source.read_bytes())
            evidence_subdir = root / EVIDENCE / "abba_main_vs_p3_24p"
            for child in sorted(evidence_subdir.iterdir()):
                child.unlink()
            evidence_subdir.rmdir()
            kind = make_directory_link(evidence_subdir, outside / "abba_main_vs_p3_24p")
            payload = self.assert_fails(
                root, "unsafe evidence path abba_main_vs_p3_24p/index.tsv"
            )
            # The bytes still match; only the escape makes it fail.
            self.assertEqual(payload["working_tree"]["failed"], 0)
            self.assertIn(kind, "\n".join(payload["failures"]))


class AttributeControls(GateCase):
    def test_missing_text_rule_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp, attributes=False)
            payload = self.assert_fails(root, "expected unset (-text)")
            self.assertEqual(payload["attributes"]["failed"], len(DEFAULT_FILES))
            self.assertEqual(payload["attributes"]["verified"], 0)

    def test_explicit_text_attribute_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            (root / ".gitattributes").write_bytes(b"phase_lxxxi_evidence/** text\n")
            self.assert_fails(root, "has text=set, expected unset (-text)")


class BlobControls(GateCase):
    def test_working_tree_and_committed_blob_disagreement_fails(self) -> None:
        """The Phase LXXXI signature: disk was "fixed", the commit was not."""
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            target = root / EVIDENCE / "receipts" / "summary.json"
            target.write_bytes(b'{"verdict": "REWRITTEN"}\n')
            files = {
                rel: (root / EVIDENCE / rel).read_bytes() for rel in DEFAULT_FILES
            }
            write_manifest(root / EVIDENCE, files)
            git(root, "add", "-f", "--", f"{EVIDENCE}/{MANIFEST}")
            git(root, "commit", "-q", "-m", "regenerate manifest from disk")
            payload = self.assert_fails(
                root, "committed blob: receipts/summary.json hashes"
            )
            self.assertEqual(payload["working_tree"]["failed"], 0)
            self.assertEqual(payload["git_blob"]["failed"], 1)

    def test_missing_git_object_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            (root / EVIDENCE / "analysis.log").unlink()
            git(root, "rm", "-q", "--cached", "--", f"{EVIDENCE}/analysis.log")
            git(root, "commit", "-q", "-m", "drop analysis.log")
            payload = self.assert_fails(root, "analysis.log")
            self.assertIn("analysis.log", payload["inventory"]["missing_from_tree"])

    def test_missing_analysis_log_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            (root / EVIDENCE / "analysis.log").unlink()
            git(root, "rm", "-q", "--cached", "--", f"{EVIDENCE}/analysis.log")
            git(root, "commit", "-q", "-m", "the *.log ignore rule won")
            payload = self.assert_fails(
                root, "manifest lists a path absent from the committed tree: analysis.log"
            )
            self.assertEqual(payload["inventory"]["missing_from_tree"], ["analysis.log"])

    def test_missing_final_rerun_log_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            (root / EVIDENCE / "final_rerun.log").unlink()
            git(root, "rm", "-q", "--cached", "--", f"{EVIDENCE}/final_rerun.log")
            git(root, "commit", "-q", "-m", "the *.log ignore rule won again")
            payload = self.assert_fails(
                root, "manifest lists a path absent from the committed tree: final_rerun.log"
            )
            self.assertEqual(payload["inventory"]["missing_from_tree"], ["final_rerun.log"])


class ManifestPinControl(GateCase):
    def test_replaced_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_repo(tmp)
            self.assert_fails(
                root, "manifest was replaced", "--expect-manifest-sha256", "0" * 64
            )


class RepositoryGate(unittest.TestCase):
    """The gate applied to this checkout, which is what the release depends on."""

    def test_frozen_manifest_digest_is_pinned(self) -> None:
        manifest = REPO / EVIDENCE / MANIFEST
        self.assertTrue(manifest.is_file(), f"missing {manifest}")
        self.assertEqual(
            hashlib.sha256(manifest.read_bytes()).hexdigest(), FROZEN_MANIFEST_SHA256
        )

    def test_committed_evidence_passes_the_gate(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--repo-root",
                str(REPO),
                "--expect-manifest-sha256",
                FROZEN_MANIFEST_SHA256,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"{completed.stdout}\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
