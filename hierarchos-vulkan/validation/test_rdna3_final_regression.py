"""A collected failure must remain red without hiding later regression gates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_rdna3_final_regression as runner
from qualify_gen9_architectures import digest, write_json


class FailureCollection(unittest.TestCase):
    def exercise(self, *, keep_going, wrong_device=False, empty_test=False, failed_test=False,
                 forward_pass=False, forward_exit=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        write_json(root / "HANDOFF-MANIFEST.json", {})
        for label in ("forward", "generation", "packages"):
            directory = root / label
            directory.mkdir()
            write_json(directory / "manifest.json", {"cases": []})
        write_json(root / "generation/manifest.json", {
            "cases": [], "source_manifest_sha256": digest(root / "packages/manifest.json")})
        binaries = {}
        for name in [*runner.BINARIES, "units"]:
            binaries[name] = root / (name + ".exe")
            binaries[name].write_bytes(b"test executable")
        unit_calls = []

        def command(args, **kwargs):
            text = ""
            code = 0
            if args[0] == "cargo":
                names = ["units"] if args[1] == "test" else runner.BINARIES
                text = "\n".join(json.dumps({
                    "reason": "compiler-artifact", "target": {"name": name},
                    "executable": str(binaries[name]), "profile": {"test": name == "units"},
                }) for name in names)
            elif "qualify_gen9_architectures.py" in args[1]:
                out = Path(args[args.index("--output") + 1])
                out.mkdir()
                write_json(out / "qualification.json", {
                    "pass": forward_pass, "rdna_profile_matrix_qualified": True,
                    "records": [{"vendor_id": 0x1002,
                                 "device_id": 0xffff if wrong_device else 0x15bf}],
                })
                code = int(not forward_pass) if forward_exit is None else forward_exit
            elif "qualify_gen9_generation.py" in args[1]:
                out = Path(args[args.index("--output") + 1])
                out.mkdir()
                write_json(out / "qualification.json", {"pass": True})
            elif args[0] == str(binaries["units"]):
                unit_calls.append(args)
                text = "test result: ok. 0 passed; 0 failed;" if empty_test else (
                    "test result: ok. 1 passed; 0 failed;")
                if failed_test and len(unit_calls) == 1:
                    code = 1
            return subprocess.CompletedProcess(args, code, text, "")

        arguments = ["run_rdna3_final_regression.py", "--forward-fixtures", str(root / "forward"),
                     "--generation-fixtures", str(root / "generation"),
                     "--generation-packages", str(root / "packages"),
                     "--expected-device-id", "0x15bf", "--output", str(root / "out")]
        if keep_going:
            arguments.append("--keep-going")
        with patch.object(runner, "ROOT", root), patch.object(sys, "argv", arguments), \
                patch.object(runner.subprocess, "run", side_effect=command):
            try:
                status = runner.main()
            except RuntimeError as error:
                status = error
        receipt = json.loads((root / "out/qualification.json").read_text())
        return status, receipt, unit_calls

    def test_default_still_stops_on_failure(self):
        status, receipt, calls = self.exercise(keep_going=False)
        self.assertIsInstance(status, RuntimeError)
        self.assertFalse(receipt["pass"])
        self.assertEqual(calls, [])

    def test_complete_pass_is_successful(self):
        status, receipt, calls = self.exercise(keep_going=True, forward_pass=True)
        self.assertEqual(status, 0)
        self.assertTrue(receipt["pass"])
        self.assertEqual(len(calls), 10)

    def test_zero_exit_cannot_override_a_failing_child_receipt(self):
        status, receipt, calls = self.exercise(keep_going=True, forward_exit=0)
        self.assertEqual(status, 1)
        self.assertFalse(receipt["pass"])
        self.assertEqual(len(calls), 10)

    def test_collected_numerical_failure_stays_red_and_runs_all_native_gates(self):
        status, receipt, calls = self.exercise(keep_going=True)
        self.assertEqual(status, 1)
        self.assertFalse(receipt["pass"])
        self.assertEqual(len(calls), 10)
        self.assertEqual(receipt["scopes"], {
            "expanded_forward_matrix": False, "previous_rdna_profile_matrix": True,
            "generation": True, "native_gates": True})

    def test_collection_never_bypasses_physical_adapter_check(self):
        status, receipt, calls = self.exercise(keep_going=True, wrong_device=True)
        self.assertIsInstance(status, RuntimeError)
        self.assertIn("physical AMD adapter", str(status))
        self.assertFalse(receipt["pass"])
        self.assertEqual(calls, [])

    def test_empty_test_filters_cannot_certify_native_scope(self):
        status, receipt, calls = self.exercise(keep_going=True, empty_test=True)
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 10)
        self.assertFalse(receipt["scopes"]["native_gates"])
        self.assertFalse(receipt["pass"])

    def test_later_tests_are_collected_after_native_failure(self):
        status, receipt, calls = self.exercise(keep_going=True, failed_test=True)
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 10)
        self.assertFalse(receipt["scopes"]["native_gates"])
        self.assertFalse(receipt["pass"])


if __name__ == "__main__":
    unittest.main()
