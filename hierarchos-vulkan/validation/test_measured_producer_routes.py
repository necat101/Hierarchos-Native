#!/usr/bin/env python3
"""Contract tests for `validation/check_measured_producer_routes.py`.

The guard is only useful if it rejects the things it claims to reject, so each
test hands it a receipt and a record with one property broken and the committed
pair untouched. The tests are table-driven over every producer spec, so adding a
producer to the guard without adding it to the sweep coverage, the parity
contract and the record agreement checks fails here rather than silently getting
a weaker guard.

The last test is the one that keeps the guard honest: a geometry where the chain
really is faster must *pass* once the record names the chain, so this is a check
on the record following the measurement rather than a check that the fused arm
always wins.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "validation" / "check_measured_producer_routes.py"
sys.path.insert(0, str(ROOT / "validation"))
import check_measured_producer_routes as guard  # noqa: E402

PRODUCERS = guard.PRODUCERS
CHAIN_PRODUCERS = guard.CHAIN_PRODUCERS
ROUTE_SPECS = PRODUCERS + CHAIN_PRODUCERS


def load(producer: guard.Producer) -> tuple[dict, dict]:
    return (
        json.loads(producer.receipt.read_text(encoding="utf-8")),
        json.loads(producer.record.read_text(encoding="utf-8")),
    )


def recompute_chain_ratios(chain: guard.ChainProducer, row: dict) -> None:
    """Rewrite every derived ratio from the arm timings after editing one arm."""
    for numerator, denominator, field in chain.ratios:
        row[field] = row[f"{numerator}_ms"] / row[f"{denominator}_ms"]
    row[f"default_over_{chain.baseline}"] = (
        row["default_ms"] / row[f"{chain.baseline}_ms"]
    )
    # The folded rung is derived from the staged arm, which a test may have just
    # moved; leaving the ratio behind would fail the consistency check instead of
    # the property under test.
    if "folded_ms" in row:
        row["folded_over_staged"] = row["folded_ms"] / row["staged_ms"]


def rows_by_batch(receipt: dict) -> dict[int, dict]:
    return {row["batch"]: row for row in receipt["batches"]}


def entries_by_batch(
    producer: guard.Producer, receipt: dict, record: dict
) -> tuple[dict[int, dict], list[dict]]:
    return guard.entries_for(producer, record, receipt)


def band_of(producer: guard.Producer, record: dict) -> float:
    return producer.band if producer.band is not None else float(record["margin"])


def measured_chain_entries(
    receipt: dict, record: dict
) -> list[dict]:
    """The chain entries the receipt's own policy measured.

    These are the entries the guard re-checks against the receipt's timings; a
    scoped receipt reads the shared entries only as its fallback, so mutating one
    of them is not evidence about that receipt.
    """
    policy = guard.receipt_policy(receipt)
    return [
        entry
        for entry in guard.chain_entries(record, receipt)
        if guard.canonical_policy(entry.get("parameter_storage")) == policy
    ]


def recompute_ratios(row: dict) -> None:
    row["composed_over_fused"] = row["composed_ms"] / row["fused_ms"]
    row["default_over_fused"] = row["default_ms"] / row["fused_ms"]
    if "narrow_ms" in row:
        row["narrow_over_fused"] = row["narrow_ms"] / row["fused_ms"]
        row["composed_over_narrow"] = row["composed_ms"] / row["narrow_ms"]


class MeasuredProducerRouteGuardTest(unittest.TestCase):
    def run_guard(
        self,
        producer: guard.Producer,
        receipt: dict,
        record: dict,
        *extra: str,
    ) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix="producer-route-") as scratch:
            receipt_path = Path(scratch) / "receipt.json"
            record_path = Path(scratch) / "record.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            record_path.write_text(json.dumps(record), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--producer",
                    producer.name,
                    "--receipt",
                    str(receipt_path),
                    "--record",
                    str(record_path),
                    *extra,
                ],
                capture_output=True,
                text=True,
            )

    def assert_rejected(
        self,
        producer: guard.Producer,
        receipt: dict,
        record: dict,
        fragment: str,
    ) -> None:
        completed = self.run_guard(producer, receipt, record)
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn(fragment, completed.stderr)

    def test_every_schema_two_route_record_is_covered(self) -> None:
        """A new measured producer must not ship an unguarded record."""
        records = set()
        kinds = set()
        for path in sorted((ROOT / "validation").glob("*_decisions.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema") != 2:
                continue
            records.add(path.name)
            kinds.add(record["kind"])
        self.assertEqual(records, {spec.record.name for spec in ROUTE_SPECS})
        self.assertEqual(kinds, {spec.kind for spec in ROUTE_SPECS})

    def test_every_committed_pair_passes(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                completed = self.run_guard(producer, receipt, record)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(f"PASS: {producer.name}", completed.stdout)

    def test_a_sweep_that_stops_before_a_recorded_geometry_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                last = max(producer.sweep)
                receipt["batches"] = [
                    row for row in receipt["batches"] if row["batch"] != last
                ]
                self.assert_rejected(
                    producer, receipt, record, f"does not dispatch batch {last}"
                )

    def test_an_entry_the_receipt_never_measured_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                lookup, _ = entries_by_batch(producer, receipt, record)
                self.assertTrue(lookup, "the committed record carries no entry to copy")
                source = next(iter(lookup.values()))
                stray = copy.deepcopy(source)
                stray["batch"] = max(producer.sweep) + 32
                record["decisions"].append(stray)
                self.assert_rejected(producer, receipt, record, "never measured")

    def test_a_recorded_geometry_without_its_decision_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                lookup, _ = entries_by_batch(producer, receipt, record)
                band = band_of(producer, record)
                rows = rows_by_batch(receipt)
                batch = next(
                    batch
                    for batch in lookup
                    if abs(rows[batch]["composed_over_fused"] - 1.0) > band
                )
                record["decisions"] = [
                    entry for entry in record["decisions"] if entry["batch"] != batch
                ]
                self.assert_rejected(producer, receipt, record, "no decision for it")

    def test_a_narrow_win_without_an_entry_is_rejected(self) -> None:
        """The key/state record must absorb a narrow-arm win, not only staged/wide."""
        producer = guard.PRODUCERS_BY_NAME["key-state-forward"]
        receipt, record = load(producer)
        row = rows_by_batch(receipt)[8]
        row["narrow_ms"] = row["fused_ms"] * 0.90
        row["composed_ms"] = row["fused_ms"]
        recompute_ratios(row)
        record["decisions"] = [
            entry for entry in record["decisions"] if entry["batch"] != 8
        ]
        self.assert_rejected(
            producer, receipt, record, "'narrow-fused' is ahead of the default route"
        )

    def test_a_route_that_is_not_the_fastest_candidate_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                lookup, _ = entries_by_batch(producer, receipt, record)
                entry = next(iter(lookup.values()))
                entry["route"] = max(entry["candidates"], key=entry["candidates"].get)
                self.assert_rejected(producer, receipt, record, "fastest candidate it lists")

    def test_a_receipt_that_moved_ahead_of_its_record_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                lookup, _ = entries_by_batch(producer, receipt, record)
                batch = next(iter(lookup))
                row = rows_by_batch(receipt)[batch]
                entry = lookup[batch]
                route = entry["route"]
                if route == "staged":
                    row["fused_ms"] = row["composed_ms"] * 0.90
                else:
                    other = next(label for label in producer.routes if label != route)
                    row[producer.fields[other]] = row[producer.fields[route]] * 0.90
                recompute_ratios(row)
                self.assert_rejected(producer, receipt, record, "the record names")

    def test_a_non_default_winner_inside_the_noise_band_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                lookup, _ = entries_by_batch(producer, receipt, record)
                entry = next(iter(lookup.values()))
                baseline = producer.default_route
                target = (
                    "narrow-fused"
                    if "narrow-fused" in producer.routes
                    else "staged"
                )
                entry["route"] = target
                entry["candidates"][target] = entry["candidates"][baseline] * 0.995
                self.assert_rejected(producer, receipt, record, "noise band")

    def test_arms_that_stop_agreeing_are_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                row = receipt["batches"][0]
                if producer.exact_parity:
                    row["worst_max_abs"] = 1.0e-9
                    self.assert_rejected(producer, receipt, record, "the arms diverge")
                else:
                    row["worst_max_abs"] = receipt["atol"] * 2.0
                    self.assert_rejected(
                        producer, receipt, record, "outside the stated atol"
                    )

    def test_narrow_and_wide_that_stop_agreeing_are_rejected(self) -> None:
        producer = guard.PRODUCERS_BY_NAME["key-state-forward"]
        receipt, record = load(producer)
        receipt["batches"][0]["worst_max_abs_wide_vs_narrow"] = 1.0e-9
        self.assert_rejected(producer, receipt, record, "wide and narrow diverge")

    def test_a_leg_that_did_not_dispatch_its_graph_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                row = receipt["batches"][0]
                row[f"fused_{producer.dispatch_key}_dispatches"] += 1
                self.assert_rejected(producer, receipt, record, "the fused leg dispatched")

    def test_a_composed_leg_that_did_not_dispatch_its_graph_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                row = receipt["batches"][0]
                key = f"composed_{producer.dispatch_key}_dispatches"
                if producer.fallback == "vendor":
                    row[key] += 1
                    fragment = "the composed leg dispatched"
                elif producer.composed_exact is not None:
                    row[key] += 1
                    fragment = f"expected {producer.composed_exact}"
                else:
                    row[key] = 1
                    fragment = "expected at least"
                self.assert_rejected(producer, receipt, record, fragment)

    def test_a_leg_label_that_names_another_graph_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                receipt["batches"][0]["composed_label"] = "fused"
                self.assert_rejected(producer, receipt, record, "the composed leg reports")

    def test_a_default_that_stops_following_the_record_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                lookup, _ = entries_by_batch(producer, receipt, record)
                batch = next(iter(lookup))
                row = rows_by_batch(receipt)[batch]
                row["default_label"] = (
                    "composed" if row["default_label"] != "composed" else row["fused_label"]
                )
                self.assert_rejected(producer, receipt, record, "the default built")

    def test_an_entry_written_by_another_driver_is_rejected(self) -> None:
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                _, applicable = entries_by_batch(producer, receipt, record)
                scoped = [
                    entry for entry in applicable if entry.get("parameter_storage")
                ]
                with_driver = [
                    entry for entry in applicable if "measured_by" in entry
                ]
                if scoped:
                    scoped[0]["measured_by"] = "validation/bench_wrong_driver.py"
                elif with_driver:
                    with_driver[0]["measured_by"] = "validation/bench_wrong_driver.py"
                else:
                    record["measured_by"] = "validation/bench_wrong_driver.py"
                self.assert_rejected(producer, receipt, record, "names the driver")

    def test_an_adapter_with_no_recorded_route_is_checked_on_its_receipt(self) -> None:
        """A fresh adapter has no verdicts yet; that is not a failure."""
        for producer in PRODUCERS:
            with self.subTest(producer=producer.name):
                receipt, record = load(producer)
                receipt["device"] = "AMD Radeon 780M"
                completed = self.run_guard(producer, receipt, record)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("no recorded route", completed.stdout)
                receipt["batches"] = [
                    row for row in receipt["batches"] if row["batch"] != max(producer.sweep)
                ]
                self.assert_rejected(
                    producer,
                    receipt,
                    record,
                    f"does not dispatch batch {max(producer.sweep)}",
                )

    def test_the_sweep_can_be_narrowed_explicitly(self) -> None:
        producer = guard.PRODUCERS_BY_NAME["time-mix-projection"]
        receipt, record = load(producer)
        receipt["batches"] = [row for row in receipt["batches"] if row["batch"] != 32]
        record["decisions"] = [entry for entry in record["decisions"] if entry["batch"] != 32]
        completed = self.run_guard(
            producer,
            receipt,
            record,
            "--sweep",
            "1",
            "--sweep",
            "4",
            "--sweep",
            "8",
            "--sweep",
            "16",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_the_fp16_receipt_must_measure_the_fp16_policy(self) -> None:
        producer = guard.PRODUCERS_BY_NAME["low-rank-forward-fp16"]
        receipt, record = load(producer)
        receipt["parameter_storage"] = "fp32"
        self.assert_rejected(producer, receipt, record, "parameter storage")
        receipt, record = load(producer)
        receipt["precision_policy"] = "fp32"
        self.assert_rejected(producer, receipt, record, "reports policy")

    def test_each_receipt_reads_only_its_own_policy_scope(self) -> None:
        """The FP16 guard must not fall back to the dense verdict under mutation.

        The dense guard ignores policy-scoped entries by construction; this pins
        that the FP16 guard really reads them by breaking one and checking that
        only the FP16 run notices.
        """
        fp16 = guard.PRODUCERS_BY_NAME["low-rank-forward-fp16"]
        dense = guard.PRODUCERS_BY_NAME["low-rank-forward"]
        receipt, record = load(fp16)
        lookup, _ = entries_by_batch(fp16, receipt, record)
        # A scoped entry the receipt actually adopted: the staged wins are now
        # withheld, so the mutation has to move a fused verdict and catch the
        # receipt naming a different route than the record.
        batch = next(
            batch
            for batch, entry in lookup.items()
            if entry.get("parameter_storage") == "fp16-storage"
            and rows_by_batch(receipt)[batch].get("recorded_route") == entry["route"]
            and entry["route"] == "fused"
        )
        entry = lookup[batch]
        entry["route"] = "staged"
        entry["candidates"]["staged"] = entry["candidates"]["fused"] * 0.90
        completed = self.run_guard(fp16, receipt, record)
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("the receipt was measured against route", completed.stderr)
        dense_receipt, _ = load(dense)
        completed = self.run_guard(dense, dense_receipt, record)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def run_composition(self, receipt: dict, records: dict[str, dict] | None) -> subprocess.CompletedProcess:
        """Run the packed-composition consumer against copies of its records."""
        with tempfile.TemporaryDirectory(prefix="composition-route-") as scratch:
            scratch_path = Path(scratch)
            receipt_path = scratch_path / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            records_dir = scratch_path / "validation"
            records_dir.mkdir()
            sources = {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path, _ in guard.PACKED_COMPOSITION.records
            }
            merged = dict(sources)
            merged.update(records or {})
            for name, record in merged.items():
                (records_dir / name).write_text(json.dumps(record), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--producer",
                    guard.PACKED_COMPOSITION.name,
                    "--receipt",
                    str(receipt_path),
                    "--records-dir",
                    str(records_dir),
                ],
                capture_output=True,
                text=True,
            )

    def test_the_consumer_receipts_pass_against_their_records(self) -> None:
        for spec in guard.CONSUMERS:
            with self.subTest(spec=spec.name):
                if isinstance(spec, guard.CompositionConsumer):
                    receipt = json.loads(spec.receipt.read_text(encoding="utf-8"))
                    completed = self.run_composition(receipt, records=None)
                else:
                    receipt, record = load(spec)
                    completed = self.run_guard(spec, receipt, record)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(f"PASS: {spec.name}", completed.stdout)

    def test_a_consumer_does_not_call_an_unmeasured_geometry_stray(self) -> None:
        """The TBPTT sweep stops at 16; the time-mix record's batch 32 is its own."""
        producer = guard.TBPTT_TIME_MIX
        receipt, record = load(producer)
        self.assertNotIn(32, producer.sweep)
        self.assertIn(32, {entry["batch"] for entry in record["decisions"]})
        completed = self.run_guard(producer, receipt, record)
        self.assertEqual(completed.returncode, 0, completed.stderr)

        # The same receipt read as a record owner is the failure the consumer
        # flag exists to avoid.
        owner = dataclasses.replace(producer, consumer=False)
        with self.assertRaisesRegex(guard.Problem, "never measured"):
            guard.check_producer(
                owner,
                receipt_path=producer.receipt,
                record_path=producer.record,
                sweep=producer.sweep,
            )

    def test_a_consumer_receipt_that_contradicts_its_record_is_rejected(self) -> None:
        producer = guard.TBPTT_TIME_MIX
        receipt, record = load(producer)
        entry = next(entry for entry in record["decisions"] if entry["batch"] == 16)
        entry["route"] = "fused"
        entry["candidates"]["fused"] = entry["candidates"]["staged"] * 0.90
        self.assert_rejected(producer, receipt, record, "the record names")

    def test_composition_ignores_verdicts_at_batches_it_did_not_run(self) -> None:
        receipt = json.loads(guard.PACKED_COMPOSITION.receipt.read_text(encoding="utf-8"))
        _, time_mix = load(guard.PRODUCERS_BY_NAME["time-mix-projection"])
        time_mix["decisions"].append(
            {
                "device": "Intel(R) HD Graphics 520",
                "width": receipt["width"],
                "batch": 1,
                "route": "fused",
                "candidates": {"fused": 1.0, "staged": 2.0},
                "probe": "linear3-forward",
                "source": "measured",
            }
        )
        completed = self.run_composition(
            receipt, records={"time_mix_projection_decisions.json": time_mix}
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_composition_rejects_a_fused_verdict_at_its_geometry(self) -> None:
        receipt = json.loads(guard.PACKED_COMPOSITION.receipt.read_text(encoding="utf-8"))
        _, time_mix = load(guard.PRODUCERS_BY_NAME["time-mix-projection"])
        time_mix["decisions"].append(
            {
                "device": "Intel(R) HD Graphics 520",
                "width": receipt["width"],
                "batch": receipt["batch"],
                "route": "fused",
                "candidates": {"fused": 1.0, "staged": 2.0},
                "probe": "linear3-forward",
                "source": "measured",
            }
        )
        completed = self.run_composition(
            receipt, records={"time_mix_projection_decisions.json": time_mix}
        )
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("qualifies the staged route", completed.stderr)

    def test_composition_rejects_a_receipt_that_lost_its_negative_control(self) -> None:
        receipt = json.loads(guard.PACKED_COMPOSITION.receipt.read_text(encoding="utf-8"))
        receipt["negative_control"] = "accepted"
        completed = self.run_composition(receipt, records=None)
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("negative control", completed.stderr)

    def test_composition_rejects_a_receipt_for_another_vendor(self) -> None:
        receipt = json.loads(guard.PACKED_COMPOSITION.receipt.read_text(encoding="utf-8"))
        receipt["qualified_vendor"] = "amd"
        completed = self.run_composition(receipt, records=None)
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("not an amd module", completed.stderr)

    def test_the_guard_cli_rejects_a_record_override_for_a_consumer(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--producer",
                guard.PACKED_COMPOSITION.name,
                "--record",
                "record.json",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("--records-dir", completed.stderr)

    def test_the_packed_chain_receipts_pass_against_their_records(self) -> None:
        for chain in CHAIN_PRODUCERS:
            with self.subTest(chain=chain.name):
                receipt, record = load(chain)
                completed = self.run_guard(chain, receipt, record)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(f"PASS: {chain.name}", completed.stdout)

    def assert_no_entry_is_a_withdrawn_tie(
        self, chain: guard.ChainProducer, receipt: dict, record: dict
    ) -> None:
        """A record with no verdict measured only defaults or inside-band wins.

        The FP16 rematerialization receipt measures structural defaults, and the
        FP16 backward-fusion receipt measures a staged win inside the band; both
        are correctly unrecorded, so a control that needs an entry to contradict
        has to say why there is none instead of assuming an arm name.
        """
        for row in receipt["batches"]:
            winner = min(chain.routes, key=lambda route: row[f"{route}_ms"])
            if winner == chain.default_route:
                continue
            drift = abs(
                row[f"{winner}_ms"] / row[f"{chain.default_route}_ms"] - 1.0
            )
            self.assertLessEqual(
                drift,
                record["margin"],
                f"{chain.name} batch {row['batch']} measures a {drift:.1%} win with no entry",
            )

    def test_a_chain_route_that_is_not_the_fastest_candidate_is_rejected(self) -> None:
        for chain in CHAIN_PRODUCERS:
            with self.subTest(chain=chain.name):
                receipt, record = load(chain)
                entries = measured_chain_entries(receipt, record)
                if not entries:
                    self.assert_no_entry_is_a_withdrawn_tie(chain, receipt, record)
                    continue
                entry = entries[0]
                entry["route"] = next(
                    route for route in chain.routes if route != entry["route"]
                )
                self.assert_rejected(chain, receipt, record, "fastest candidate it lists")

    def test_a_chain_receipt_that_moved_ahead_of_its_record_is_rejected(self) -> None:
        for chain in CHAIN_PRODUCERS:
            with self.subTest(chain=chain.name):
                receipt, record = load(chain)
                entries = measured_chain_entries(receipt, record)
                if not entries:
                    self.assert_no_entry_is_a_withdrawn_tie(chain, receipt, record)
                    continue
                entry = entries[0]
                row = rows_by_batch(receipt)[entry["batch"]]
                other = next(route for route in chain.routes if route != entry["route"])
                row[f"{other}_ms"] = row[f"{entry['route']}_ms"] * 0.90
                recompute_chain_ratios(chain, row)
                self.assert_rejected(chain, receipt, record, "the record names")

    def test_a_chain_win_outside_the_band_without_an_entry_is_rejected(self) -> None:
        chain = guard.ALL_SPECS_BY_NAME["tbptt-packed-forward"]
        receipt, record = load(chain)
        measured = {entry["batch"] for entry in record["decisions"]}
        batch = next(batch for batch in chain.sweep if batch not in measured)
        row = rows_by_batch(receipt)[batch]
        row["legacy_ms"] = row["packed_ms"] * 0.92
        row["packed_over_legacy"] = row["packed_ms"] / row["legacy_ms"]
        row["default_over_legacy"] = row["default_ms"] / row["legacy_ms"]
        self.assert_rejected(chain, receipt, record, "no decision for it")

    def test_a_chain_default_that_stops_following_the_record_is_rejected(self) -> None:
        for chain in CHAIN_PRODUCERS:
            with self.subTest(chain=chain.name):
                receipt, record = load(chain)
                entries = measured_chain_entries(receipt, record)
                if not entries:
                    # A table that records no verdict at this receipt's geometries
                    # has no default it could fail to follow; the tie rule below
                    # is what keeps that state honest.
                    self.assert_no_entry_is_a_withdrawn_tie(chain, receipt, record)
                    continue
                entry = entries[0]
                row = rows_by_batch(receipt)[entry["batch"]]
                row["default_label"] = next(
                    route for route in chain.routes if route != entry["route"]
                )
                self.assert_rejected(chain, receipt, record, "the default built")

    def test_packed_chain_parity_must_stay_exact(self) -> None:
        for chain in CHAIN_PRODUCERS:
            with self.subTest(chain=chain.name):
                receipt, record = load(chain)
                receipt["batches"][0]["worst_max_abs"] = 1.0e-9
                self.assert_rejected(chain, receipt, record, "the chains diverge")

    def backward_pair(self) -> tuple[guard.ChainProducer, dict, dict]:
        chain = guard.ALL_SPECS_BY_NAME["tbptt-backward-fusion"]
        receipt, record = load(chain)
        return chain, receipt, record

    def test_the_folded_rung_is_recorded_only_where_its_reduction_is_exact(self) -> None:
        """The route's acceptance gate is the fold's own readback, with no allowance."""
        chain, receipt, record = self.backward_pair()
        row = receipt["batches"][0]
        self.assertEqual(row["folded_key_parameter_max_abs"], 0.0)
        self.assertGreater(row["staged_key_parameter_max_abs"], 0.0)
        self.assertTrue(row["folded_candidate"])
        row["folded_key_parameter_max_abs"] = 1.0e-8
        self.assert_rejected(chain, receipt, record, "acceptance gate admits no allowance")

    def test_a_zero_staged_key_parameter_gradient_is_not_evidence(self) -> None:
        chain, receipt, record = self.backward_pair()
        receipt["batches"][0]["staged_key_parameter_max_abs"] = 0.0
        self.assert_rejected(chain, receipt, record, "would prove nothing")

    def test_a_folded_win_without_its_candidate_is_rejected(self) -> None:
        """The record cannot drop a candidate the receipt proved, nor keep one it did not."""
        chain, receipt, record = self.backward_pair()
        entries = [
            entry
            for entry in measured_chain_entries(receipt, record)
            if "folded" in entry["candidates"]
        ]
        if entries:
            entries[0]["candidates"].pop("folded")
            self.assert_rejected(
                chain, receipt, record, "the folded rung is a candidate only where"
            )
            return
        # No entry at this receipt's geometries: the same contract is proved from
        # the receipt side, where a folded candidate has to be a measured one.
        receipt["batches"][0].pop("folded_ms")
        self.assert_rejected(chain, receipt, record, "a folded candidate it never timed")

    def test_a_folded_rung_at_a_shallower_depth_is_rejected(self) -> None:
        chain, receipt, record = self.backward_pair()
        row = receipt["batches"][0]
        row["folded_schedule"] = row["fused_schedule"]
        self.assert_rejected(chain, receipt, record, "not the fully folded rung")

    def refused_forward_pair(self) -> tuple[guard.ChainProducer, dict, dict]:
        """The committed forward pair with the FP16 policy's measurements withdrawn.

        The committed record lifts the FP16 refusal by recording the same
        crossover under `fp16-storage`; removing those entries reconstructs the
        state the refusal was recorded in, so the refusal machinery stays
        exercised by a fixture instead of by the shipped table.
        """
        chain = guard.CHAIN_PRODUCERS[0]
        receipt, record = load(chain)
        record["decisions"] = [
            entry
            for entry in record["decisions"]
            if entry.get("parameter_storage") is None
        ]
        receipt["parameter_storage"] = "fp16-storage"
        for row in receipt["batches"]:
            if row["batch"] in (16, 32):
                row["recorded_route"] = None
                row["expected_default_label"] = "packed"
                row["default_label"] = "packed"
        record["withheld"] = [
            {
                "device": receipt["device"],
                "width": receipt["width"],
                "batch": batch,
                "route": "legacy",
                "reason": (
                    "the FP16 trajectory golden was recorded on the structural "
                    "packed route"
                ),
                "parameter_storage": "fp16-storage",
            }
            for batch in (16, 32)
        ]
        return chain, receipt, record

    def test_a_scoped_chain_refusal_is_read_by_its_own_policy(self) -> None:
        chain, receipt, record = self.refused_forward_pair()
        completed = self.run_guard(chain, receipt, record)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("withheld: fp16-storage may not adopt", completed.stdout)
        self.assertIn("batch 16: packed", completed.stdout)

    def test_the_fp16_receipt_lifts_the_refusal_it_measured(self) -> None:
        """The shipped FP16 receipt and record agree that the refusal is gone."""
        chain = guard.ALL_SPECS_BY_NAME["tbptt-packed-forward-fp16"]
        receipt, record = load(chain)
        self.assertEqual(guard.receipt_policy(receipt), "fp16-storage")
        self.assertFalse(
            [
                entry
                for entry in record["withheld"]
                if entry.get("parameter_storage") == "fp16-storage"
            ],
            "a recorded FP16 measurement must lift the refusal at its geometry",
        )
        measured = {entry["batch"] for entry in measured_chain_entries(receipt, record)}
        self.assertEqual(measured, {1, 16, 32})
        completed = self.run_guard(chain, receipt, record)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("batch 16: legacy", completed.stdout)
        self.assertIn("batch 32: legacy", completed.stdout)

    def test_a_scoped_chain_verdict_leaves_the_default_receipt_alone(self) -> None:
        chain = guard.ALL_SPECS_BY_NAME["tbptt-packed-forward"]
        receipt, record = load(chain)
        scoped = next(
            entry for entry in record["decisions"] if entry.get("parameter_storage")
        )
        scoped["route"] = "packed" if scoped["route"] == "legacy" else "legacy"
        default_receipt = json.loads(chain.receipt.read_text(encoding="utf-8"))
        completed = self.run_guard(chain, default_receipt, record)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("batch 16: legacy", completed.stdout)

    def test_a_scoped_chain_verdict_that_is_not_its_fastest_is_rejected(self) -> None:
        chain = guard.ALL_SPECS_BY_NAME["tbptt-packed-forward-fp16"]
        receipt, record = load(chain)
        scoped = next(
            entry
            for entry in measured_chain_entries(receipt, record)
            if entry.get("parameter_storage")
        )
        scoped["route"] = "packed" if scoped["route"] == "legacy" else "legacy"
        self.assert_rejected(chain, receipt, record, "fastest candidate it lists")

    def test_a_refusal_at_a_batch_the_policy_measured_is_rejected(self) -> None:
        chain = guard.ALL_SPECS_BY_NAME["tbptt-packed-forward-fp16"]
        receipt, record = load(chain)
        record["withheld"].append(
            {
                "device": receipt["device"],
                "width": receipt["width"],
                "batch": 16,
                "route": "legacy",
                "reason": "stale refusal beside a measurement",
                "parameter_storage": "fp16-storage",
            }
        )
        self.assert_rejected(chain, receipt, record, "measured its own verdict")

    def test_a_chain_refusal_must_name_the_measured_verdict(self) -> None:
        chain, receipt, record = self.refused_forward_pair()
        held = record["withheld"][0]
        held["route"] = "packed" if held["route"] == "legacy" else "legacy"
        self.assert_rejected(chain, receipt, record, "would read")

    def test_a_chain_refusal_without_a_reason_is_rejected(self) -> None:
        chain, receipt, record = self.refused_forward_pair()
        record["withheld"][0]["reason"] = " "
        self.assert_rejected(chain, receipt, record, "carries no reason")

    def test_a_chain_refusal_has_to_be_scoped_to_a_policy(self) -> None:
        chain, receipt, record = self.refused_forward_pair()
        del record["withheld"][0]["parameter_storage"]
        self.assert_rejected(chain, receipt, record, "has to name the policy")

    def test_a_chain_policy_that_refuses_one_verdict_must_refuse_them_all(self) -> None:
        chain, receipt, record = self.refused_forward_pair()
        record["withheld"] = [record["withheld"][0]]
        self.assert_rejected(chain, receipt, record, "has to refuse every measured move")

    def test_a_chain_sweep_that_stops_before_a_recorded_geometry_is_rejected(self) -> None:
        for chain in CHAIN_PRODUCERS:
            with self.subTest(chain=chain.name):
                receipt, record = load(chain)
                # Removing the geometry from both the measured rows and the
                # `unmeasured` report is the only way a sweep can hide it, so both
                # lists have to be edited here; a receipt that measures batch 32
                # has it only in the first, and one that cannot has it only in
                # the second.
                receipt["batches"] = [
                    row for row in receipt["batches"] if row["batch"] != 32
                ]
                receipt["unmeasured"] = [
                    batch for batch in receipt.get("unmeasured", []) if batch != 32
                ]
                self.assert_rejected(chain, receipt, record, "does not dispatch batch 32")

    def test_an_adapter_with_no_chain_entries_is_checked_on_its_receipt(self) -> None:
        for chain in CHAIN_PRODUCERS:
            with self.subTest(chain=chain.name):
                receipt, record = load(chain)
                receipt["device"] = "AMD Radeon 780M"
                for row in receipt["batches"]:
                    row["recorded_route"] = None
                    row["expected_default_label"] = chain.default_route
                    row["default_label"] = chain.default_route
                completed = self.run_guard(chain, receipt, record)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("structural default", completed.stdout)

    def test_a_measured_chain_win_passes_once_the_record_names_it(self) -> None:
        """The guard follows the measurement; it does not defend the fused arm."""
        producer = guard.PRODUCERS_BY_NAME["key-state-forward"]
        receipt, record = load(producer)
        row = rows_by_batch(receipt)[32]
        row["composed_ms"] = row["fused_ms"] * 0.90
        row["default_ms"] = row["composed_ms"]
        row["default_label"] = "composed"
        row["expected_default_label"] = "composed"
        row["default_arm"] = None
        row["default_key_state_forward_dispatches"] = 2
        row["recorded_route"] = "staged"
        row["recorded_compose"] = True
        recompute_ratios(row)
        entry = entries_by_batch(producer, receipt, record)[0][32]
        entry["route"] = "staged"
        entry["candidates"]["staged"] = entry["candidates"][producer.default_route] * 0.90
        completed = self.run_guard(producer, receipt, record)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("batch 32: staged", completed.stdout)


if __name__ == "__main__":
    unittest.main()
