"""Negative controls for strict, device-specific CPU logit qualification."""
import math
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import qualify_gen9_architectures as q


class StrictLogits(unittest.TestCase):
    def test_exact_words_pass(self):
        bits = q.words([0.125, -0.5, 2.0])
        result = q.compare_logits(bits, bits, [True] * 3)
        self.assertTrue(result["pass"])
        self.assertEqual(result["word_differences"], 0)

    def test_threshold_is_strict(self):
        with patch.object(q, "LIMIT", 0.25):
            result = q.compare_logits(q.words([0.25]), q.words([0.0]), [True])
        self.assertFalse(result["pass"])
        self.assertEqual(result["failing_values"], 1)

    def test_no_relative_tolerance(self):
        result = q.compare_logits(q.words([100000008.0]), q.words([100000000.0]), [True])
        self.assertFalse(result["pass"])

    def test_inconsistent_and_empty_shapes_fail(self):
        for a, b, mask in [([], [], []), ([0], [0, 0], [True]), ([0], [0], [])]:
            with self.subTest(a=a, b=b, mask=mask), self.assertRaises(ValueError):
                q.compare_logits(a, b, mask)

    def test_no_visible_values_is_not_a_pass(self):
        with self.assertRaises(ValueError):
            q.compare_logits([0], [0], [False])

    def test_nonfinite_values_fail_even_in_excluded_rows(self):
        for value in [math.nan, math.inf, -math.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                q.compare_logits(q.words([0, value]), [0, 0], [True, False])

    def test_padding_exclusion_is_counted(self):
        result = q.compare_logits(q.words([0.125, 1.0]), q.words([0.125, 0.0]), [True, False])
        self.assertTrue(result["pass"])
        self.assertEqual((result["values"], result["excluded_values"]), (1, 1))

    def test_invalid_words_fail(self):
        for bits in [[-1], [1 << 32], [True], [0.0]]:
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                q.values(bits)


class ResidencyCoverage(unittest.TestCase):
    def report(self, **updates):
        return {"device": "Intel(R) HD Graphics 520", "streamed_linears": 0,
                "source_first_parameters": False, "loading_linear_package_io": [], **updates}

    def test_resident_does_not_accept_streaming(self):
        for updates in [{"streamed_linears": 1}, {"source_first_parameters": True},
                        {"loading_linear_package_io": [["model", {}]]}]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                q.validate_residency(self.report(**updates), "resident")

    def test_ignored_streaming_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            q.validate_residency(self.report(), "streaming")

    def test_wrong_gpu_is_rejected(self):
        with self.assertRaises(ValueError):
            q.validate_residency(self.report(device="AMD Radeon Graphics"), "resident")

    def test_real_modes_pass(self):
        q.validate_residency(self.report(), "resident")
        q.validate_residency(self.report(streamed_linears=12), "streaming")

    def test_rdna_handoff_requires_amd_and_keeps_storage_guards(self):
        amd = self.report(device="AMD Radeon Graphics", vendor_id=0x1002, device_id=0x15bf)
        q.validate_residency(amd, "resident", "rdna3", 0x15bf)
        for invalid in [self.report(), {**amd, "vendor_id": 0x8086}, {**amd, "streamed_linears": 1}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                q.validate_residency(invalid, "resident", "rdna3")
        with self.assertRaises(ValueError):
            q.validate_residency(amd, "resident", "rdna3", 0xffff)

    def test_gen9_rejects_wrong_pci_even_with_correct_name(self):
        with self.assertRaises(ValueError):
            q.validate_residency(self.report(vendor_id=0x1002, device_id=0x15bf), "resident")


class QualificationFailureControls(unittest.TestCase):
    def test_reversed_modes_still_compare_all_storage_words(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            q.write_json(fixtures / "cpu.json", {"bits": q.words([0.0]), "visible": [True], "vocab_size": 1})
            q.write_json(fixtures / "input.json", {})
            files = {name: q.digest(fixtures / name) for name in ("cpu.json", "input.json")}
            q.write_json(fixtures / "manifest.json", {"absolute_limit": q.LIMIT, "rtol": 0,
                "comparison": "strict_less_than", "oracle": {}, "rdna_verified_profiles": ["control"],
                "registry": {"canonical": ["control"]}, "cases": [{"family": "control",
                    "case": "unmasked", "files": files, "cpu": "cpu.json", "input": "input.json", "package": "."}]})
            binary = root / "probe.exe"
            binary.write_bytes(b"mock executable")
            args = argparse.Namespace(fixtures=fixtures, binary=binary, output=root / "run",
                families=None, modes=["streaming", "resident"], timeout=1)
            def probe(command, **kwargs):
                streaming = "--stream-parameters" in command
                q.write_json(Path(command[command.index("--output") + 1]), {
                    "device": "Intel(R) HD Graphics 520", "architecture": "control", "vocab_size": 1,
                    "streamed_linears": int(streaming), "source_first_parameters": False,
                    # Each mode is within the CPU bound, but storage must be bit-exact.
                    "logits_bits": q.words([1e-8 if streaming else 0.0])})
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(q.subprocess, "run", side_effect=probe):
                self.assertEqual(q.run(args), 1)
            result = json.loads((args.output / "qualification.json").read_text())
            self.assertFalse(result["rdna_profile_matrix_qualified"])
            self.assertEqual(result["records"][1]["resident_word_differences"], 1)

    def test_empty_residency_cannot_qualify(self):
        with self.assertRaises(ValueError):
            q.run(argparse.Namespace(modes=[]))


if __name__ == "__main__":
    unittest.main()
