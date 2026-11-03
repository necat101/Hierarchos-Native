"""Negative controls for the projected-feature qualification gate (stdlib)."""
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qualify_multimodal_training as q
from diagnose_multimodal_updates import write_f32_tensors


class MultimodalGateTests(unittest.TestCase):
    def test_frozen_cpu_vector_policy_overrides_replay_host(self):
        with patch.dict("os.environ", {"HIERARCHOS_ATEN_VECTOR_WIDTH": "16"}):
            self.assertEqual(q.oracle_env("AVX2")["HIERARCHOS_ATEN_VECTOR_WIDTH"], "8")
        self.assertEqual(q.oracle_env("AVX512")["HIERARCHOS_ATEN_VECTOR_WIDTH"], "16")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            q.oracle_env("DEFAULT")

    def test_incomplete_cpu_preparation_cannot_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps({"complete": False}))
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                q.run(SimpleNamespace(fixtures=root, output=output))
            self.assertFalse(output.exists())

    def test_strict_threshold_and_no_relative_tolerance(self):
        self.assertTrue(q.compare([0.0], [0.0])["pass"])
        self.assertFalse(q.compare([q.LIMIT], [0.0])["pass"])
        self.assertFalse(q.compare([100.000001], [100.0])["pass"])

    def test_no_empty_or_partial_comparisons(self):
        for actual, expected in [([], []), ([1.], []), ([1.], [1., 2.])]:
            with self.assertRaises(ValueError):
                q.compare(actual, expected)

    def test_nonfinite_is_never_a_pass(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.assertRaises(ValueError):
                q.compare([value], [0.])
            with self.assertRaises(ValueError):
                q.compare([0.], [value])

    def test_physical_device_required(self):
        intel = {"vendor_id": 0x8086, "device_id": 0x1916}
        amd = {"vendor_id": 0x1002, "device_id": 0x15bf}
        q.validate_device(intel, "gen9")
        q.validate_device(amd, "rdna3", 0x15bf)
        for report, target, device in [(amd, "gen9", None), (intel, "rdna3", 0x1916),
                (amd, "rdna3", None), (amd, "rdna3", 0x1234)]:
            with self.assertRaises(ValueError):
                q.validate_device(report, target, device)

    def test_safetensors_shape_and_payload_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.safetensors"
            def write(shape, end, payload):
                header = json.dumps({"weight": {"dtype": "F32", "shape": shape,
                    "data_offsets": [0, end]}}).encode()
                path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            write([2], 8, struct.pack("<2f", 1., 2.))
            self.assertEqual(q.read_f32_tensors(path)["weight"], ([2], [1., 2.]))
            write([3], 8, struct.pack("<2f", 1., 2.))
            with self.assertRaises(ValueError):
                q.read_f32_tensors(path)
            write([2], 8, struct.pack("<f", 1.))
            with self.assertRaises(ValueError):
                q.read_f32_tensors(path)

    def test_diagnostic_state_writer_preserves_tensor_names_shapes_and_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.safetensors"
            tensors = {"embed.weight": ([2, 2], [1., -2., 3., 4.]), "norm.weight": ([2], [1., 1.])}
            write_f32_tensors(path, tensors)
            self.assertEqual(q.read_f32_tensors(path), tensors)


if __name__ == "__main__":
    unittest.main()
