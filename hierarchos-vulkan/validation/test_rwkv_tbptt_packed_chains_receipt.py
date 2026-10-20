#!/usr/bin/env python3
"""Contract tests for the packed TBPTT chain and backward-fusion receipts.

`bench_rwkv_tbptt_packed_chains.py` records one receipt per chain and merges the
winners into the schema-2 record `RwkvTbpttSequenceOp` reads. These tests keep
both halves honest: the receipt must show bit-identical graphs, a complete
sweep and the route the record names at every geometry, and the record must name
a candidate its own timings put first. The two chain A/Bs name their arms
`packed`/`legacy`; the backward-fusion A/B names them `fused`/`staged` and also
times the fully folded rung, which joins the routing pair as a third candidate
wherever the receipt proved its key-parameter reduction exact. That optional third
route is the only difference the table below encodes.

Set `HIERARCHOS_RUN_GPU_VALIDATION=1` to re-run the driver on the attached
adapter; that is the check that fails when the kernel or the default regresses.
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
DRIVER = ROOT / "validation" / "bench_rwkv_tbptt_packed_chains.py"

sys.path.insert(0, str(ROOT / "validation"))
import bench_rwkv_tbptt_packed_chains as driver  # noqa: E402  (the gates under test)
EXPECTED_TENSORS = {
    "outputs",
    "final_packed_state",
    "grad_x",
    "token_feature_grad",
    "grad_initial_packed_state",
}
CHAINS = {
    "forward": {
        "receipt": ROOT / "validation" / "tbptt_packed_forward_bench.json",
        "record": ROOT / "validation" / "tbptt_packed_forward_decisions.json",
        "kind": "rwkv-tbptt-packed-forward",
        "probe": "tbptt-packed-forward",
        "driver_chain": "forward",
        "arms": ("legacy", "packed"),
        "ratio": "packed_over_legacy",
        "default": "packed",
        "policy": None,
    },
    "remat": {
        "receipt": ROOT / "validation" / "tbptt_packed_remat_bench.json",
        "record": ROOT / "validation" / "tbptt_packed_rematerialization_decisions.json",
        "kind": "rwkv-tbptt-packed-rematerialization",
        "probe": "tbptt-packed-rematerialization",
        "driver_chain": "remat",
        "arms": ("legacy", "packed"),
        "ratio": "packed_over_legacy",
        "default": "legacy",
        "policy": None,
    },
    "backward": {
        "receipt": ROOT / "validation" / "tbptt_backward_fusion_bench.json",
        "record": ROOT / "validation" / "tbptt_backward_fusion_decisions.json",
        "kind": "rwkv-tbptt-backward-fusion",
        "probe": "tbptt-backward-fusion",
        "driver_chain": "backward",
        "arms": ("staged", "fused"),
        "optional": ("folded",),
        "ratio": "fused_over_staged",
        "default": "fused",
        "policy": None,
    },
    "forward-fp16": {
        "receipt": ROOT / "validation" / "tbptt_packed_forward_fp16_bench.json",
        "record": ROOT / "validation" / "tbptt_packed_forward_decisions.json",
        "kind": "rwkv-tbptt-packed-forward",
        "probe": "tbptt-packed-forward",
        "driver_chain": "forward",
        "arms": ("legacy", "packed"),
        "ratio": "packed_over_legacy",
        "default": "packed",
        "policy": "fp16-storage",
    },
    "remat-fp16": {
        "receipt": ROOT / "validation" / "tbptt_packed_remat_fp16_bench.json",
        "record": ROOT / "validation" / "tbptt_packed_rematerialization_decisions.json",
        "kind": "rwkv-tbptt-packed-rematerialization",
        "probe": "tbptt-packed-rematerialization",
        "driver_chain": "remat",
        "arms": ("legacy", "packed"),
        "ratio": "packed_over_legacy",
        "default": "legacy",
        "policy": "fp16-storage",
    },
    "backward-fp16": {
        "receipt": ROOT / "validation" / "tbptt_backward_fusion_fp16_bench.json",
        "record": ROOT / "validation" / "tbptt_backward_fusion_decisions.json",
        "kind": "rwkv-tbptt-backward-fusion",
        "probe": "tbptt-backward-fusion",
        "driver_chain": "backward",
        "arms": ("staged", "fused"),
        "optional": ("folded",),
        "ratio": "fused_over_staged",
        "default": "fused",
        "policy": "fp16-storage",
    },
}


def entry_for(record: dict, receipt: dict, batch: int) -> dict | None:
    """The verdict the runtime adopts for this receipt's policy at one batch.

    A scoped entry wins over the shared fallback, and a scoped refusal withdraws
    the batch entirely. The default policy reads only the shared entries.
    """
    device = receipt["device"].strip().lower()
    width = receipt["width"]
    policy = receipt.get("parameter_storage")
    scoped = shared = None
    for entry in record["decisions"]:
        if (
            entry["device"].strip().lower() != device
            or entry["width"] != width
            or entry["batch"] != batch
        ):
            continue
        scope = entry.get("parameter_storage")
        if policy is not None and scope == policy:
            scoped = entry
        elif scope is None:
            shared = entry
    if policy is not None and any(
        withheld.get("parameter_storage") == policy
        and withheld["device"].strip().lower() == device
        and withheld["width"] == width
        and withheld["batch"] == batch
        for withheld in record.get("withheld", [])
    ):
        return None
    return scoped or shared


class PackedChainReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs = {}
        for chain, spec in CHAINS.items():
            if not spec["receipt"].exists():
                raise AssertionError(
                    f"missing {spec['receipt']}; run "
                    "validation/bench_rwkv_tbptt_packed_chains.py --chain "
                    f"{spec['driver_chain']}"
                    + (
                        " --parameter-storage fp16-storage"
                        if spec["policy"]
                        else ""
                    )
                )
            cls.pairs[chain] = (
                json.loads(spec["receipt"].read_text(encoding="utf-8")),
                json.loads(spec["record"].read_text(encoding="utf-8")),
            )

    def test_receipts_measured_a_bit_identical_complete_sweep(self) -> None:
        for chain, (receipt, _) in self.pairs.items():
            spec = CHAINS[chain]
            with self.subTest(chain=chain):
                self.assertEqual(receipt["result"], "pass")
                self.assertEqual(receipt["chain"], spec["driver_chain"])
                self.assertEqual(receipt.get("parameter_storage"), spec["policy"])
                self.assertGreaterEqual(receipt["steps"], 1)
                self.assertGreaterEqual(receipt["samples"], 1)
                self.assertEqual(receipt["margin"], 0.02)
                measured = {row["batch"] for row in receipt["batches"]}
                unmeasured = set(receipt.get("unmeasured", []))
                self.assertEqual(
                    measured | unmeasured,
                    {1, 4, 8, 16, 32},
                    "the sweep must dispatch every geometry or report why it could not",
                )
                self.assertEqual(
                    measured & unmeasured,
                    set(),
                    "a geometry cannot be both measured and reported unmeasured",
                )
                self.assertEqual(
                    unmeasured,
                    set(),
                    "every chain compiles two graphs at every geometry on this adapter",
                )
                if spec["arms"] == ("staged", "fused"):
                    # The staged arm must be the fully staged chain at every
                    # geometry, not the one-dispatch folded kernel the batch-1
                    # shortcut offers. `+` names a stage that still dispatches;
                    # `-` names one folded into the state-backward kernel. The
                    # folded rung is timed alongside both routing candidates so
                    # all three fusion depths stay measurable at every batch.
                    for row in receipt["batches"]:
                        self.assertNotEqual(
                            row["staged_schedule"],
                            row["fused_schedule"],
                            "the two arms must dispatch different graphs",
                        )
                        self.assertTrue(
                            row["staged_schedule"].startswith("rkv-add3+key+reduce"),
                            "the staged arm has to be the fully staged chain, got "
                            f"{row['staged_schedule']!r}",
                        )
                        self.assertTrue(
                            row["folded_schedule"].startswith("rkv-add3-key-reduce"),
                            "the evidence rung has to fold both stages, got "
                            f"{row['folded_schedule']!r}",
                        )
                        self.assertEqual(
                            len(
                                {
                                    row["staged_schedule"],
                                    row["folded_schedule"],
                                    row["fused_schedule"],
                                }
                            ),
                            3,
                            "the ladder needs three distinct fusion depths per batch",
                        )
                        self.assertLessEqual(
                            row["folded_max_abs"],
                            1.0e-6,
                            "the folded rung may only differ from the staged chain by "
                            "contracted FP32 ulps; a larger delta means the kernel "
                            "changed, not fused",
                        )
                        self.assertAlmostEqual(
                            row["folded_over_staged"],
                            row["folded_ms"] / row["staged_ms"],
                        )
                        self.assertLessEqual(
                            row["folded_parameter_delta"],
                            1.0e-6,
                            "an AdamW step on the folded rung has to leave the "
                            "parameters where the staged rung leaves them; a larger "
                            "delta means the cross-batch reduction did not reproduce "
                            "the shipped fold",
                        )
                        # The fold's own output, not an optimizer-normalized view of
                        # it: the fused kernel sums the batch rows in the shipped
                        # reduction's order, so this has to be exact.
                        self.assertGreater(
                            row["staged_key_parameter_max_abs"],
                            0.0,
                            "a zero staged gradient would make the parity below vacuous",
                        )
                        self.assertEqual(
                            row["folded_key_parameter_max_abs"],
                            0.0,
                            "the folded reduction must reproduce the two-dispatch "
                            "chain's grad_k_k/grad_k_a exactly at every batch",
                        )
                for row in receipt["batches"]:
                    self.assertEqual(row["worst_max_abs"], 0.0)
                    self.assertEqual(
                        {tensor["label"] for tensor in row["tensors"]},
                        EXPECTED_TENSORS,
                    )
                    for tensor in row["tensors"]:
                        self.assertLessEqual(tensor["worst_excess"], 0.0)
                    numerator, denominator = spec["arms"][1], spec["arms"][0]
                    self.assertAlmostEqual(
                        row[spec["ratio"]],
                        row[f"{numerator}_ms"] / row[f"{denominator}_ms"],
                    )

    def test_records_are_fastest_candidate_evidence(self) -> None:
        for chain, (receipt, record) in self.pairs.items():
            spec = CHAINS[chain]
            with self.subTest(chain=chain):
                self.assertEqual(record["schema"], 2)
                self.assertEqual(record["kind"], spec["kind"])
                self.assertEqual(record["margin"], 0.02)
                self.assertTrue(
                    record["decisions"],
                    "an empty record is legal only before the first measurement",
                )
                # The default winning is not a tie: the receipt documents that the
                # geometry was measured and the shipped graph held, so every such
                # row needs an entry this policy can read.
                for row in receipt["batches"]:
                    with self.subTest(chain=chain, batch=row["batch"]):
                        candidates = spec["arms"] + tuple(
                            route
                            for route in spec.get("optional", ())
                            if row.get(f"{route}_candidate")
                        )
                        winner = min(
                            candidates, key=lambda route: row[f"{route}_ms"]
                        )
                        if winner != spec["default"]:
                            continue
                        # A second candidate inside the noise band makes this a tie,
                        # and a tie is withdrawn: the run that recorded may have
                        # measured the other graph a fraction of a percent ahead and
                        # left the geometry unrecorded, which this receipt cannot
                        # tell apart from a missing verdict. Only a clear win by the
                        # shipped graph has to be documented.
                        runner_up = min(
                            row[f"{route}_ms"]
                            for route in candidates
                            if route != winner
                        )
                        if runner_up / row[f"{winner}_ms"] - 1.0 <= receipt["margin"]:
                            continue
                        entry = entry_for(record, receipt, row["batch"])
                        self.assertIsNotNone(
                            entry,
                            "a clear win by the shipped graph still needs an entry "
                            "documenting the measurement",
                        )
                        self.assertEqual(entry["route"], winner)
                for entry in record["decisions"]:
                    with self.subTest(chain=chain, batch=entry["batch"]):
                        self.assertEqual(entry["device"], "Intel(R) HD Graphics 520")
                        self.assertEqual(entry["width"], 128)
                        self.assertEqual(entry["source"], "measured")
                        self.assertEqual(entry["probe"], spec["probe"])
                        self.assertEqual(
                            entry["measured_by"],
                            "validation/bench_rwkv_tbptt_packed_chains.py",
                        )
                        # The two-dispatch pair is always a candidate; the folded
                        # rung joins it only where this record's own measurement
                        # cleared its gate, so the set is checked per entry below.
                        self.assertLessEqual(
                            set(spec["arms"]),
                            set(entry["candidates"]),
                            "every entry lists the routing pair it was measured against",
                        )
                        self.assertLessEqual(
                            set(entry["candidates"]),
                            set(spec["arms"]) | set(spec.get("optional", ())),
                            "an entry may not name a candidate this chain does not measure",
                        )
                        fastest = min(entry["candidates"].values())
                        self.assertEqual(
                            entry["candidates"][entry["route"]],
                            fastest,
                            "the recorded route has to be the fastest candidate it lists",
                        )

    def test_the_folded_rung_is_a_candidate_exactly_where_it_is_recordable(self) -> None:
        """The record's candidate list has to mirror the receipt's gate.

        The folded rung is recordable where the receipt measured its
        `grad_k_k`/`grad_k_a` bit-identical to the two-dispatch reduction with a
        non-zero staged gradient, and nowhere else. A record entry at such a
        geometry has to carry that candidate; one at a geometry that failed the
        gate must not, because a recorded `folded` route would then rest on a
        measurement that never cleared its acceptance gate.
        """
        for chain in ("backward", "backward-fp16"):
            receipt, record = self.pairs[chain]
            spec = CHAINS[chain]
            for row in receipt["batches"]:
                with self.subTest(chain=chain, batch=row["batch"]):
                    self.assertEqual(
                        row["folded_key_parameter_max_abs"],
                        0.0,
                        "the route's acceptance gate has to hold at every measured batch",
                    )
                    self.assertTrue(row["folded_candidate"])
                    entry = entry_for(record, receipt, row["batch"])
                    if entry is None:
                        continue
                    self.assertIn(
                        "folded",
                        entry["candidates"],
                        "a record entry measured against the folded rung has to list it",
                    )

    def test_default_follows_the_record_or_the_structural_default(self) -> None:
        for chain, (receipt, record) in self.pairs.items():
            for row in receipt["batches"]:
                with self.subTest(chain=chain, batch=row["batch"]):
                    entry = entry_for(record, receipt, row["batch"])
                    expected = entry["route"] if entry else CHAINS[chain]["default"]
                    self.assertEqual(row["default_label"], expected)
                    self.assertEqual(row["expected_default_label"], expected)
                    self.assertEqual(row["recorded_route"], entry["route"] if entry else None)

    def test_a_policy_that_measured_the_chain_adopts_its_own_verdict(self) -> None:
        """The FP16 refusal is lifted by a measurement, not by a code change.

        The packed forward chain's legacy transition was withheld from FP16
        storage because no chain A/B had been run under that policy. The FP16
        receipt now measures the same transition, its graph is bit-identical to
        the packed one, and the policy records the verdict under its own scope,
        so the refusal is gone and the shared FP32 entries are unchanged.
        """
        _, forward = self.pairs["forward"]
        structural = CHAINS["forward"]["default"]
        moves = {
            entry["batch"]
            for entry in forward["decisions"]
            if entry.get("parameter_storage") is None and entry["route"] != structural
        }
        scoped = {
            entry["batch"]: entry
            for entry in forward["decisions"]
            if entry.get("parameter_storage") == "fp16-storage"
        }
        self.assertTrue(moves, "the packed-forward table records a measured move away from packed")
        self.assertEqual(
            forward["withheld"],
            [],
            "the FP16 policy measured every move itself, so nothing is refused any more",
        )
        for batch in moves:
            self.assertIn(
                batch,
                scoped,
                "a measured move without an FP16 verdict would leave the refusal path",
            )
            self.assertEqual(
                scoped[batch]["route"],
                {entry["batch"]: entry for entry in forward["decisions"]}[batch]["route"],
                "the FP16 verdict has to answer the same geometry as the shared one",
            )
        _, remat = self.pairs["remat"]
        self.assertEqual(
            remat["withheld"],
            [],
            "rematerialization's measured verdicts are its structural default, so no "
            "policy refuses anything there",
        )

    def test_a_measured_win_outside_the_band_has_an_entry(self) -> None:
        for chain, (receipt, record) in self.pairs.items():
            for row in receipt["batches"]:
                with self.subTest(chain=chain, batch=row["batch"]):
                    spec = CHAINS[chain]
                    winner = min(
                        spec["arms"]
                        + tuple(
                            route
                            for route in spec.get("optional", ())
                            if row.get(f"{route}_candidate")
                        ),
                        key=lambda route: row[f"{route}_ms"],
                    )
                    if winner == CHAINS[chain]["default"]:
                        continue
                    drift = abs(
                        row[f"{winner}_ms"] / row[f"{CHAINS[chain]['default']}_ms"] - 1.0
                    )
                    if drift <= receipt["margin"]:
                        continue
                    entry = entry_for(record, receipt, row["batch"])
                    self.assertIsNotNone(
                        entry,
                        "a measured win outside the band needs a verdict this policy can read",
                    )
                    self.assertEqual(entry["route"], winner)
                    self.assertEqual(row["default_label"], winner)
                    self.assertEqual(
                        "folded" in entry["candidates"],
                        bool(row.get("folded_candidate")),
                        "the recorded candidate list has to match what the receipt proved",
                    )


class FoldedGuardTest(unittest.TestCase):
    """The two gates a backward receipt row has to clear, without an adapter.

    `measure` applies these to every measured geometry; the negative control
    proves they reject a real broken fold, and these keep the rejection itself
    from being weakened to a no-op.
    """

    def test_the_parameter_step_gate_rejects_a_corrupted_fold(self) -> None:
        with self.assertRaises(driver.bridge.Mismatch):
            driver.require_folded_parameter_parity(4, {"folded_parameter_delta": 1.0e-4})
        driver.require_folded_parameter_parity(4, {"folded_parameter_delta": 1.5e-8})

    def test_the_key_parameter_gate_admits_no_allowance(self) -> None:
        with self.assertRaises(driver.bridge.Mismatch):
            driver.require_folded_key_parameter_parity(
                4, {"folded_key_parameter_max_abs": 1.0e-8}
            )
        driver.require_folded_key_parameter_parity(4, {"folded_key_parameter_max_abs": 0.0})

    def test_a_zero_staged_gradient_is_not_parity(self) -> None:
        with self.assertRaises(driver.bridge.Mismatch):
            driver.require_folded_key_parameter_parity(
                4,
                {
                    "staged_key_parameter_max_abs": 0.0,
                    "folded_key_parameter_max_abs": 0.0,
                },
            )

    def test_the_gates_ignore_rows_without_a_folded_rung(self) -> None:
        driver.require_folded_tensor_parity(4, {})
        driver.require_folded_key_parameter_parity(4, {})
        driver.require_folded_parameter_parity(4, {})
        self.assertFalse(
            driver.folded_candidate(4, {}),
            "a geometry that compiles no folded rung has no folded candidate",
        )

    def test_the_candidate_gate_is_the_folds_own_readback(self) -> None:
        exact = {
            "folded_ms": 1.0,
            "folded_max_abs": 1.1920928955078125e-07,
            "folded_key_parameter_max_abs": 0.0,
            "staged_key_parameter_max_abs": 0.0094,
            "folded_parameter_delta": 1.4901161193847656e-08,
        }
        self.assertTrue(driver.folded_candidate(4, exact))
        for field, value in (
            ("folded_key_parameter_max_abs", 1.0e-8),
            ("staged_key_parameter_max_abs", 0.0),
            ("folded_max_abs", 1.0e-3),
            ("folded_parameter_delta", 1.0e-3),
        ):
            with self.subTest(field=field):
                broken = dict(exact, **{field: value})
                with self.assertRaises(driver.bridge.Mismatch):
                    driver.folded_candidate(4, broken)
        missing = dict(exact)
        del missing["folded_key_parameter_max_abs"]
        with self.assertRaises(driver.bridge.Mismatch):
            driver.folded_candidate(4, missing)


class LivePackedChainTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the GPU chain A/Bs",
    )
    def test_driver_passes_on_this_adapter(self) -> None:
        for chain, spec in CHAINS.items():
            with self.subTest(chain=chain):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(DRIVER),
                        "--chain",
                        spec["driver_chain"],
                        *(
                            ("--parameter-storage", spec["policy"])
                            if spec["policy"]
                            else ()
                        ),
                        "--iterations",
                        "2",
                        "--warmup",
                        "1",
                    ],
                    cwd=ROOT.parent,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)

    @unittest.skipUnless(
        os.environ.get("HIERARCHOS_RUN_GPU_VALIDATION") == "1",
        "set HIERARCHOS_RUN_GPU_VALIDATION=1 to re-run the folded-rung negative control",
    )
    def test_the_guards_reject_a_broken_publication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hierarchos-tbptt-control-") as root:
            record = Path(root) / "negative_control.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DRIVER),
                    "--negative-control",
                    "--batch",
                    "1",
                    "--batch",
                    "4",
                    "--record",
                    str(record),
                ],
                cwd=ROOT.parent,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            control = json.loads(record.read_text(encoding="utf-8"))
        self.assertEqual(control["result"], "pass")
        self.assertEqual(control["negative_control"], "packed-fold-publication")
        self.assertEqual(
            {run["batch"] for run in control["batches"]},
            {1, 4},
            "batch 1 and batch >1 publish the fold differently, and both must be proven",
        )
        for run in control["batches"]:
            with self.subTest(batch=run["batch"]):
                expected = (
                    "batch-1-direct-write-sink" if run["batch"] == 1 else "cross-batch-arrival-count"
                )
                self.assertEqual(run["corruption"], expected)
                if run["batch"] == 1:
                    self.assertIsNone(
                        run["arrival_seed"],
                        "batch 1 writes the gradients directly, so it has no arrival count",
                    )
                else:
                    self.assertEqual(run["arrival_seed"], run["batch"])
                self.assertGreater(run["staged_key_parameter_max_abs"], 0.0)
                self.assertEqual(run["clean_key_parameter_max_abs"], 0.0)
                self.assertLessEqual(run["clean_parameter_delta"], 1.0e-6)
                self.assertGreater(run["corrupted_key_parameter_max_abs"], 0.0)
                self.assertGreater(run["corrupted_parameter_delta"], 1.0e-6)
                self.assertTrue(run["post_step_guard_saw_the_corruption"])
                self.assertTrue(run["rejection"].startswith(f"batch {run['batch']}"))


if __name__ == "__main__":
    unittest.main()
