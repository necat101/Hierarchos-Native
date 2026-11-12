#!/usr/bin/env python3
"""Negative controls for the immutable oracle archive verifier."""
from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest
import zipfile

from verify_phase19y_oracles import verify_archive, verify_manifest


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_zip(path: pathlib.Path, entries: dict[str, bytes]) -> tuple[str, int]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in entries.items():
            bundle.writestr(name, data)
    data = path.read_bytes()
    return sha(data), len(data)


class OracleArchiveControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.payload = b"fixture-bytes"
        self.oracle = b"oracle-source"
        self.manifest = {
            "complete": True,
            "cases": [{"files": {"case/value.bin": sha(self.payload)}}],
            "oracle": {"source_sha256": {"oracle-source/source.py": sha(self.oracle)}},
        }
        self.manifest_bytes = (json.dumps(self.manifest) + "\n").encode()

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, extra=None, missing=False):
        entries = {
            "manifest.json": self.manifest_bytes,
            "case/value.bin": self.payload,
            "oracle-source/source.py": self.oracle,
        }
        if missing:
            entries.pop("case/value.bin")
        entries.update(extra or {})
        path = self.root / "oracle.zip"
        expected = make_zip(path, entries)
        return path, expected

    def test_positive_archive_and_manifest(self):
        path, (expected_sha, expected_size) = self.archive()
        destination = self.root / "extracted"
        verify_archive(path, expected_sha, expected_size, "test", destination)
        result = verify_manifest(destination, sha(self.manifest_bytes), "test")
        self.assertEqual(result["case_files"], 1)
        self.assertEqual(result["total_files"], 3)

    def test_wrong_archive_hash(self):
        path, (_, expected_size) = self.archive()
        with self.assertRaises(ValueError):
            verify_archive(path, "0" * 64, expected_size, "test", self.root / "out")

    def test_truncated_archive(self):
        path, (_, expected_size) = self.archive()
        path.write_bytes(path.read_bytes()[:-20])
        with self.assertRaises(ValueError):
            verify_archive(path, sha(path.read_bytes()), expected_size, "test", self.root / "out")

    def test_missing_fixture(self):
        path, (expected_sha, expected_size) = self.archive(missing=True)
        destination = self.root / "out"
        verify_archive(path, expected_sha, expected_size, "test", destination)
        with self.assertRaises(ValueError):
            verify_manifest(destination, sha(self.manifest_bytes), "test")

    def test_altered_fixture_byte(self):
        path, (expected_sha, expected_size) = self.archive()
        destination = self.root / "out"
        verify_archive(path, expected_sha, expected_size, "test", destination)
        (destination / "case/value.bin").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            verify_manifest(destination, sha(self.manifest_bytes), "test")

    def test_extra_fixture(self):
        path, (expected_sha, expected_size) = self.archive(extra={"case/extra.bin": b"x"})
        destination = self.root / "out"
        verify_archive(path, expected_sha, expected_size, "test", destination)
        with self.assertRaises(ValueError):
            verify_manifest(destination, sha(self.manifest_bytes), "test")

    def test_wrong_manifest(self):
        path, (expected_sha, expected_size) = self.archive()
        destination = self.root / "out"
        verify_archive(path, expected_sha, expected_size, "test", destination)
        with self.assertRaises(ValueError):
            verify_manifest(destination, "0" * 64, "test")

    def test_path_traversal_is_rejected(self):
        path, (_, _) = self.archive(extra={"../escape.bin": b"x"})
        data = path.read_bytes()
        with self.assertRaises(ValueError):
            verify_archive(path, sha(data), len(data), "test", self.root / "out")
        self.assertFalse((self.root / "escape.bin").exists())


if __name__ == "__main__":
    unittest.main()
