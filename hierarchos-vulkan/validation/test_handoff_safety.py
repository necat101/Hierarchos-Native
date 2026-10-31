"""Rejected tuning must not replace the known-good adapter routing tables."""
import os
from pathlib import Path
import tempfile
import unittest

from qualify_handoff import protect_routes, qualification_commands
from check_stable_probe_parity import write_source


class HandoffSafetyTests(unittest.TestCase):
    def test_complete_requalification_includes_both_profiles_network_and_frontends(self):
        def python(script, *args):
            return [script, *map(str, args)]
        before = dict(qualification_commands(python, Path("oracle"), Path("before"), "amd"))
        after = dict(qualification_commands(python, Path("oracle"), Path("after"), "amd"))
        self.assertEqual(set(before), {
            "python-contracts", "all-binaries", "library-and-trainer", "two-rank-tcp",
            "hierarchos-inference", "hierarchos-native-cli", "hierarchos-gui",
            "stable-recurrent", "stable-logits", "rwkv-vendor-composition", "rwkv-cpu-bridge",
            "training-stack-release", "training-stack-debug",
        })
        self.assertEqual(set(before), set(after))
        self.assertIn("--release", after["training-stack-release"])
        self.assertNotIn("--release", after["training-stack-debug"])
        self.assertIn("both", after["stable-recurrent"])
        self.assertIn("--with-inference-logits", after["stable-recurrent"])
        self.assertIn("--extended", after["stable-recurrent"])
        self.assertIn("amd", after["rwkv-vendor-composition"])
        self.assertIn(str(Path("before") / "stable-recurrent.json"), before["stable-recurrent"])
        self.assertIn(str(Path("after") / "stable-recurrent.json"), after["stable-recurrent"])

    def test_failed_tuning_restores_original_bytes_before_rebuilding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            original = b'{"intel": "qualified"}\r\n'
            path.write_bytes(original)
            rebuilt = []
            def measure():
                path.write_text('{"amd": "unqualified"}')
                return False
            self.assertFalse(protect_routes([path], measure, lambda: rebuilt.append(path.read_bytes())))
            self.assertEqual(rebuilt, [original])

    def test_exception_restores_routes_and_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_bytes(b"qualified")
            def measure():
                path.write_bytes(b"broken")
                raise RuntimeError("parity failed")
            with self.assertRaisesRegex(RuntimeError, "parity failed"):
                protect_routes([path], measure, lambda: None)
            self.assertEqual(path.read_bytes(), b"qualified")

    def test_success_keeps_measured_routes_and_does_not_rebuild_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_bytes(b"qualified")
            def measure():
                path.write_bytes(b"new qualified measurement")
                return True
            self.assertTrue(protect_routes([path], measure, lambda: self.fail("unexpected rollback")))
            self.assertEqual(path.read_bytes(), b"new qualified measurement")

    def test_identical_generated_observer_keeps_its_cargo_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.rs"
            write_source(path, "fn main() {}\n")
            os.utime(path, ns=(1_000_000_000_000, 1_000_000_000_000))
            previous = path.stat().st_mtime_ns
            write_source(path, "fn main() {}\n")
            self.assertEqual(path.stat().st_mtime_ns, previous)
            write_source(path, "fn main() { panic!(); }\n")
            self.assertNotEqual(path.stat().st_mtime_ns, previous)


if __name__ == "__main__":
    unittest.main()
