import unittest
import tempfile
from pathlib import Path

from ab_harness import expected_default_pair, record_decisions


class VendorCompositionPolicyTests(unittest.TestCase):
    def test_new_amd_fp32_measurement_cannot_be_inherited_by_fp16(self):
        for device, policy in [("AMD Radeon Graphics [pci=1002:15bf;driver=00800184]", "fp32"),
                               ("Intel(R) HD Graphics 520", None)]:
            run = {"device": device, "width": 128, "batch": 8,
                   "fused": {"timing": {"median_ms": 1.0}},
                   "composed": {"timing": {"median_ms": 1.1}}}
            with tempfile.TemporaryDirectory() as directory:
                record, _ = record_decisions(Path(directory) / "record.json", [run],
                    kind="rwkv-low-rank-full-forward", probe_key="low-rank-forward",
                    margin=0.02, probe_label="low-rank-forward")
            self.assertEqual(record["decisions"][0].get("parameter_storage"), policy)

    def test_vendor_dispatch_does_not_imply_a_qualified_fusion_replacement(self):
        run = {"composed": {"linear3_dispatches": 1}, "fused": {"label": "fused"},
               "unmeasured_compose": False}
        self.assertEqual(expected_default_pair(run, "linear3"), "fused")
        self.assertEqual(expected_default_pair({**run, "unmeasured_compose": True}, "linear3"), "composed")
        self.assertEqual(expected_default_pair({**run, "recorded_compose": True}, "linear3"), "composed")
        self.assertEqual(expected_default_pair({**run, "unmeasured_compose": True,
                                               "recorded_compose": False}, "linear3"), "fused")
