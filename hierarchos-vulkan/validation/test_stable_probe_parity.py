#!/usr/bin/env python3
"""Contract tests for `validation/check_stable_probe_parity.py`.

The parity gate is the one check that says the optimized tree still computes what
the pre-optimization tree computed, so its comparison has to be exactly as strict
as it claims. These tests drive the comparison itself with fabricated snapshots --
one differing bit, one missing tensor, one moved optimizer clock -- and the CLI
with a checkout that is not a checkout, so the failure paths are exercised on a
machine with no adapter.

The GPU run is opt-in with `HIERARCHOS_RUN_STABLE_PARITY=1` and needs the stable
tree on disk; it is the check that fails when a kernel changes a value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "check_stable_probe_parity.py"

sys.path.insert(0, str(ROOT / "validation"))
import check_stable_probe_parity as parity  # noqa: E402  (the comparison under test)


def snapshot(*, value: str = "aa", extra: str = "bb") -> list[dict]:
    """One two-step probe snapshot with two tensors per step."""
    return [
        {
            "step": index,
            "optimizer_step": index,
            "tensors": {
                "parameter/w": value,
                "exp_avg/w": extra,
            },
            "optimizer_slot_steps": {"w": index},
        }
        for index in range(2)
    ]


class SnapshotComparisonTest(unittest.TestCase):
    def test_reference_association_is_adapter_driver_scoped_and_preserves_goldens(self):
        from safetensors.numpy import save_file
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            save_file({f"h_rnn.{name}": np.zeros((128, rank), dtype=np.float32)
                       for name, rank in (("w1", 32), ("a1", 32), ("g1", 64))},
                      str(package / "model.safetensors"))
            device = "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]"
            for batch, expected in ((2, "fused-base"), (32, "split"), (64, "fused-base")):
                environment = parity.reference_fan_in_environment(device, package, ["--batch", str(batch)])
                self.assertEqual(environment["HIERARCHOS_RWKV_LOW_RANK_FAN_IN_SCHEDULE"], expected)
                self.assertEqual(environment["HIERARCHOS_RWKV_BACKWARD_SEGMENT_CACHE_PATH"], "")
            self.assertEqual(parity.reference_fan_in_environment(
                device.replace("00800184", "00800185"), package, ["--batch", "32"]), {})
            self.assertEqual(parity.reference_fan_in_environment(None, package, []), {})

    def test_extended_cases_include_long_context_and_masked_multi_sequence_accumulation(self):
        long, masked = parity.EXTENDED_CASES[:2]
        self.assertIn("128", long["args"])
        self.assertEqual(masked["env"], {parity.PROBE_MASK_ENV: "1"})
        self.assertIn("--sequences", masked["args"])
        self.assertEqual(masked["args"][masked["args"].index("--sequences") + 1], "3")

    def test_extended_catalog_keeps_raw_gradients_resume_and_both_policy_geometries(self):
        cases = {case["name"]: case for case in parity.extended_catalog()}
        self.assertTrue({"batch2", "batch32", "odd_multi_sequence", "small_hidden32"} <= cases.keys())
        self.assertEqual(cases["raw_gradient_accumulation"]["env"][parity.PROBE_GRADIENT_ENV], "1")
        self.assertEqual(cases["checkpoint_resume"]["env"][parity.PROBE_RESUME_ENV], "1")
        for batch in (1, 4, 8, 9, 16, 17, 32, 64):
            self.assertIn(f"batch{batch}_tokens16", cases)
        self.assertEqual(cases["odd_multi_sequence"]["env_by_policy"]["fp16-storage-fp32-compute"]
                         ["HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY"], "one-row-rank128")

    def test_identical_snapshots_pass(self) -> None:
        self.assertEqual(parity.diff_snapshots(snapshot(), snapshot()), [])
        self.assertEqual(parity.digests(snapshot()), 4)

    def test_one_differing_digest_fails(self) -> None:
        observed = snapshot()
        observed[1]["tensors"]["exp_avg/w"] = "bc"
        differences = parity.diff_snapshots(snapshot(), observed)
        self.assertEqual(len(differences), 1)
        self.assertIn("step 1: exp_avg/w differs", differences[0])

    def test_a_missing_tensor_is_a_difference(self) -> None:
        observed = snapshot()
        del observed[0]["tensors"]["parameter/w"]
        differences = parity.diff_snapshots(snapshot(), observed)
        self.assertIn("step 0: parameter/w differs", differences)

    def test_a_moved_optimizer_clock_is_a_difference(self) -> None:
        observed = snapshot()
        observed[1]["optimizer_slot_steps"]["w"] = 7
        differences = parity.diff_snapshots(snapshot(), observed)
        self.assertEqual(differences, ["step 1: optimizer_slot_steps differs"])

    def test_a_short_run_is_a_difference(self) -> None:
        differences = parity.diff_snapshots(snapshot(), snapshot()[:1])
        self.assertIn("step count 1 != 2", differences)

    def test_the_golden_geometries_are_the_three_trajectories(self) -> None:
        self.assertEqual(
            [case["name"] for case in parity.CASES],
            ["batch2", "batch32", "odd_multi_sequence"],
        )
        self.assertEqual(
            sum(
                sum(len(step["tensors"]) for step in case["devices"]["Intel(R) HD Graphics 520"])
                for case in parity.CASES
            ),
            3817,
            "the parity gate compares this many digests per device",
        )


class CommandLineTest(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
        )

    def test_the_stable_checkout_is_required(self) -> None:
        completed = self.run_script()
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("--stable", completed.stderr)

    def test_a_path_that_is_not_a_checkout_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="not-a-checkout-") as empty:
            completed = self.run_script("--stable", empty)
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("is not a Hierarchos checkout", completed.stderr)
        self.assertIn("Pass the repository root", completed.stderr)

    def test_a_case_the_golden_does_not_record_is_refused(self) -> None:
        completed = self.run_script(
            "--stable",
            str(ROOT.parent / "does-not-exist"),
            "--case",
            "batch7",
        )
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("unknown case(s) ['batch7']", completed.stderr)


class LiveStableParityTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_STABLE_PARITY") == "1",
        "set HIERARCHOS_RUN_STABLE_PARITY=1 (and HIERARCHOS_STABLE_CHECKOUT) to compare "
        "against a stable tree",
    )
    def test_every_golden_geometry_matches_the_stable_tree(self) -> None:
        checkout = os.environ.get("HIERARCHOS_STABLE_CHECKOUT")
        if not checkout:
            self.skipTest("HIERARCHOS_STABLE_CHECKOUT names the checkout to compare against")
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--stable", checkout],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PASS:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
