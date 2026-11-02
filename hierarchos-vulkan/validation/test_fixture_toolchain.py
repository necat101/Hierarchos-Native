from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import fixture_toolchain as ft

PAIR = {"torch": "2.14.1+cpu", "numpy": "2.5.3", "cpu": "AVX2"}


class ToolchainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "deepseek_v41_example.json"
        self.current = patch.object(ft, "current", return_value=dict(PAIR))
        self.current.start()
        self.addCleanup(self.current.stop)

    def write(self, value=None):
        ft.write_fixture(self.path, json.dumps({"values": [1, 2]} if value is None else value))

    def test_verified_write_records_libraries_cpu_and_hash(self):
        self.write()
        self.assertEqual(ft.pin(self.path), PAIR)
        self.assertEqual(ft._read(self.root / ft.SIDECAR)[self.path.name]["sha256"],
                         hashlib.sha256(self.path.read_bytes()).hexdigest())
        ft.check_file(self.path)

    def test_reproduction_preserves_frozen_bytes(self):
        self.path.write_bytes(b'{ "values" : [1,2] }\n')
        before = self.path.read_bytes()
        with self.assertWarns(UserWarning):
            self.write()
        self.assertEqual(before, self.path.read_bytes())

    def test_changed_values_are_never_re_pinned(self):
        self.write()
        before = (self.root / ft.SIDECAR).read_bytes()
        with self.assertRaisesRegex(ValueError, "frozen CPU fixture"):
            self.write({"values": [9]})
        self.assertEqual(before, (self.root / ft.SIDECAR).read_bytes())

    def test_stale_hash_fails(self):
        self.write()
        self.path.write_text('{"values":[3]}')
        with self.assertRaisesRegex(ValueError, "stale toolchain pin"):
            ft.check_file(self.path)

    def test_each_toolchain_dimension_is_enforced(self):
        self.write()
        for key in PAIR:
            with self.subTest(key=key), self.assertRaises(SystemExit):
                ft.check_file(self.path, actual={**PAIR, key: "different"})

    def test_override_warns_and_does_not_replace_pin(self):
        self.write()
        with self.assertWarns(UserWarning):
            ft.check_file(self.path, actual={**PAIR, "cpu": "AVX512"}, allow=True)
        self.assertEqual(ft.pin(self.path), PAIR)

    def test_legacy_warns(self):
        self.path.write_text('{}')
        with self.assertWarnsRegex(UserWarning, "predates"):
            ft.check_file(self.path)

    def test_variants_are_explicit_siblings(self):
        with patch.dict(os.environ, {"HIERARCHOS_FIXTURE_VARIANT": "avx2"}):
            self.assertEqual(ft.variant_path(self.path), self.path.with_name("deepseek_v41_example.avx2.json"))
        with patch.dict(os.environ, {"HIERARCHOS_FIXTURE_VARIANT": "../escape"}):
            with self.assertRaises(ValueError): ft.variant_path(self.path)

    def test_empty_and_unpinned_directories_fail(self):
        self.assertTrue(ft.audit(self.root, PAIR)["problems"])
        self.path.write_text('{}')
        self.assertTrue(ft.audit(self.root, PAIR)["problems"])

    def test_legacy_baseline_is_hash_bound_and_shrinks_after_reproduction(self):
        self.path.write_text('{"values":[1,2]}')
        baseline = self.root / ft.LEGACY
        baseline.write_text(json.dumps({self.path.name: {"sha256": ft._sha(self.path)}}))
        self.assertFalse(ft.audit(self.root, PAIR)["problems"])
        self.assertTrue(ft.audit(self.root, PAIR, strict=True)["problems"])
        with self.assertWarns(UserWarning): self.write()
        self.assertEqual(ft._read(baseline), {})
        self.assertFalse(ft.audit(self.root, PAIR, strict=True)["problems"])

    def test_audit_rejects_mismatch_missing_cpu_and_stale_baseline(self):
        self.write()
        self.assertTrue(ft.audit(self.root, {**PAIR, "torch": "other"})["problems"])
        (self.root / ft.LEGACY).write_text(json.dumps({"deepseek_v41_missing.json": {}}))
        self.assertTrue(ft.audit(self.root, PAIR)["problems"])
        pins = ft._read(self.root / ft.SIDECAR)
        del pins[self.path.name]["toolchain"]["cpu"]
        (self.root / ft.SIDECAR).write_text(json.dumps(pins))
        self.assertTrue(ft.audit(self.root, PAIR)["problems"])

    def test_inline_sidecar_conflict_fails(self):
        self.path.write_text(json.dumps({"toolchain": {**PAIR, "cpu": "other"}}))
        (self.root / ft.SIDECAR).write_text(json.dumps({self.path.name: {
            "toolchain": PAIR, "sha256": ft._sha(self.path)}}))
        with self.assertRaisesRegex(ValueError, "disagree"):
            ft.pin(self.path)

    def test_array_fixtures_keep_their_bytes_and_audit(self):
        self.path.write_bytes(b'[1,2,3]\n')
        with self.assertWarns(UserWarning):
            ft.write_fixture(self.path, '[1, 2, 3]')
        self.assertEqual(self.path.read_bytes(), b'[1,2,3]\n')
        self.assertEqual(ft.pin(self.path), PAIR)
        self.assertFalse(ft.audit(self.root, PAIR, strict=True)["problems"])

    def test_legacy_baseline_cannot_grow_beyond_the_original_handoff(self):
        self.path.write_text('{}')
        entry = {"sha256": ft._sha(self.path)}
        (self.root / ft.LEGACY).write_text(json.dumps({self.path.name: entry}))
        key = "hierarchos-vulkan/validation/fixtures/" + self.path.name
        self.assertFalse(ft.audit(self.root, PAIR, historical={key: entry})["problems"])
        self.assertTrue(ft.audit(self.root, PAIR, historical={})["problems"])
        self.assertTrue(ft.audit(self.root, PAIR, historical={key: {"sha256": "changed"}})["problems"])


if __name__ == "__main__":
    unittest.main()
