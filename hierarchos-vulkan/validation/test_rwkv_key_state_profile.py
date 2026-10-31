#!/usr/bin/env python3
"""Contracts for the key/state phase profile (`validation/key_state_forward_profile.json`).

The profile is the *why* beside the A/B's *which*: GPU microseconds per phase from
the device's own timestamp queries. These tests keep it an explanation of the
measured ranking rather than a second, contradictory benchmark.

What they check:

* the report covers the phases the explanation needs -- both fused arms and both
  chain kernels -- at every geometry it claims, and says the numbers are per-dispatch
  GPU time rather than host wait time;
* the chain's cost is dominated by the state phase and that share rises with batch,
  while the transform phase stays roughly flat: that is the shape the lane
  utilization story predicts, and it is why the chain's per-row cost falls as rows
  grow;
* the wider group's state phase costs less per row than the narrow kernel's at the
  geometries where the bench measured it faster, and the per-row numbers are derived
  from the same row count the kernels dispatch.

Set `HIERARCHOS_RUN_GPU_VALIDATION=1` to re-profile the attached adapter.
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
REPO = ROOT.parent
PROFILE = ROOT / "validation" / "key_state_forward_profile.json"
DECISIONS = ROOT / "validation" / "key_state_forward_decisions.json"
DRIVER = ROOT / "validation" / "profile_rwkv_key_state_forward.py"


class KeyStateProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PROFILE.exists():
            raise AssertionError(
                f"missing {PROFILE.relative_to(REPO)}; run "
                "validation/profile_rwkv_key_state_forward.py --record to produce it"
            )
        cls.report = json.loads(PROFILE.read_text(encoding="utf-8"))
        cls.rows = cls.report["batches"]

    def test_the_profile_is_per_dispatch_gpu_time(self) -> None:
        self.assertEqual(self.report["kind"], "rwkv-key-state-forward-profile")
        self.assertGreaterEqual(len(self.rows), 2)
        self.assertIn("timestamp queries", self.report["timestamp_source"])
        self.assertGreater(self.report["runs"], 0)
        self.assertEqual(self.report["lane_accounting"]["wide"]["state_lanes"], 64)
        self.assertEqual(
            self.report["lane_accounting"]["narrow"]["state_lanes"], self.report["head_size"]
        )
        self.assertEqual(self.report["lane_accounting"]["narrow"]["transform_lanes"], 1)
        for row in self.rows:
            with self.subTest(batch=row["batch"]):
                for key in (
                    "wide_fused_us",
                    "narrow_fused_us",
                    "chain_transform_us",
                    "chain_state_us",
                ):
                    self.assertGreater(row[key], 0.0, f"{key} was not measured")
                self.assertEqual(row["rows"], row["batch"] * self.report["width"])
                self.assertAlmostEqual(
                    row["chain_total_us"], row["chain_transform_us"] + row["chain_state_us"]
                )
                self.assertAlmostEqual(
                    row["chain_state_share"],
                    row["chain_state_us"] / row["chain_total_us"],
                )
                self.assertAlmostEqual(
                    row["wide_us_per_row"], row["wide_fused_us"] / row["rows"]
                )
                self.assertAlmostEqual(
                    row["narrow_us_per_row"], row["narrow_fused_us"] / row["rows"]
                )
                self.assertAlmostEqual(
                    row["wide_over_chain_state_us"],
                    row["wide_fused_us"] / row["chain_state_us"],
                )
                self.assertAlmostEqual(
                    row["wide_excess_over_chain_state_us"],
                    row["wide_fused_us"] - row["chain_state_us"],
                )
                self.assertAlmostEqual(
                    row["wide_excess_us_per_row"],
                    row["wide_excess_over_chain_state_us"] / row["rows"],
                )

    def test_the_state_phase_dominates_and_grows_with_rows(self) -> None:
        """The claim the crossover explanation rests on, measured per phase.

        The transform phase costs what one head's serial normalization costs, so it
        is nearly flat in batch; the state phase advances `batch * width` rows and
        grows with them. A profile where transform time grew with the batch would
        mean the explanation is about something else.
        """
        ordered = sorted(self.rows, key=lambda row: row["batch"])
        shares = [row["chain_state_share"] for row in ordered]
        # Monotone within the record's own 2% band: a single noisy pair must not
        # turn a measured trend into a red test, and a real inversion (the state
        # phase shrinking as rows grow) is far larger than that band.
        band = json.loads(DECISIONS.read_text(encoding="utf-8")).get("margin", 0.02)
        for earlier, later in zip(shares, shares[1:]):
            self.assertGreaterEqual(
                later,
                earlier - band,
                f"the state share fell from {earlier:.3f} to {later:.3f}: the row-scaled phase "
                "is not the one that grows",
            )
        self.assertGreater(shares[-1], 0.6, "the state phase is not the dominant phase")
        transform = [row["chain_transform_us"] for row in ordered]
        state = [row["chain_state_us"] for row in ordered]
        self.assertLess(
            transform[-1] / transform[0],
            state[-1] / state[0],
            "the transform phase grew faster than the row-scaled state phase",
        )

    def test_the_fused_excess_is_attributed_to_the_state_phase(self) -> None:
        """What dominates the fused kernel, from the phase that runs its rows.

        The fused dispatch has no phase seam, so its own cost is measured against
        the one reference phase that advances the same rows at the same occupancy:
        chain state. The report's summary has to be that column, and where the
        record names the chain at the largest profiled batch, the largest excess
        has to be at that batch -- otherwise the fused kernel lost its singleton
        launch cost, not its per-row work, and the attribution would be wrong.
        """
        ordered = sorted(self.rows, key=lambda row: row["batch"])
        excess = [row["wide_excess_over_chain_state_us"] for row in ordered]
        attribution = self.report["attribution"]
        worst = max(ordered, key=lambda row: row["wide_excess_over_chain_state_us"])
        self.assertEqual(attribution["largest_excess_batch"], worst["batch"])
        self.assertAlmostEqual(attribution["largest_excess_us"], max(excess))
        self.assertIn("state phase", attribution["note"])

        record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        device = self.report["device"]
        largest = ordered[-1]["batch"]
        routes = {
            entry["batch"]: entry["route"]
            for entry in record["decisions"]
            if entry["device"] == device and entry["width"] == self.report["width"]
        }
        if routes.get(largest) == "staged":
            self.assertEqual(
                worst["batch"],
                largest,
                "the chain won this batch, so the fused kernel's cost has to have grown "
                "with the row work rather than stayed at its launch overhead",
            )

    def test_the_wider_group_costs_less_per_row_where_the_record_prefers_it(self) -> None:
        """Cross-artifact contract: the ranking and the per-row accounting agree.

        The geometries are not chosen here. They are the ones the A/B *recorded* as
        `wide-fused` wins outside its own noise band, so this cannot pass by
        selecting a friendly geometry: at each of those batches the wide kernel's
        measured cost per row has to be below the narrow kernel's, which is what
        "all 64 lanes advance rows" has to mean in time. On an adapter that
        recorded no wide win there is nothing to contract, and the test skips
        instead of inventing a ranking.
        """
        record = json.loads(DECISIONS.read_text(encoding="utf-8"))
        device = self.report["device"]
        wide_batches = {
            entry["batch"]
            for entry in record["decisions"]
            if entry["device"] == device
            and entry["width"] == self.report["width"]
            and entry["route"] == "wide-fused"
        }
        rows = {row["batch"]: row for row in self.rows}
        checked = sorted(wide_batches & set(rows))
        if not checked:
            self.skipTest("this adapter's record names no wide-fused geometry the profile covers")
        for batch in checked:
            row = rows[batch]
            with self.subTest(batch=batch):
                self.assertLess(
                    row["wide_us_per_row"],
                    row["narrow_us_per_row"],
                    "the record names the wide route here but its own phase profile puts it "
                    "behind the narrow kernel per row",
                )


class LiveProfileTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-profile on the GPU",
    )
    def test_profile_reruns_on_this_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            record = Path(scratch) / "profile.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DRIVER),
                    "--batch",
                    "1",
                    "--batch",
                    "16",
                    "--runs",
                    "2",
                    "--record",
                    str(record),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            self.assertTrue(record.exists())


if __name__ == "__main__":
    unittest.main()
