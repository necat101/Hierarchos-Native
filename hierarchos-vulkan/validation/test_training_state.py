"""Negative controls for the shipping trajectory comparison (no GPU needed).

The trajectory suite runs once per LN1/low-rank topology arm, and an arm is only
worth running if it really ran the variant it asked for: a variant this device
cannot offer is rejected by the producer, which then runs the default kernel and
passes. The tests below pin both halves of that -- the request the producer
acknowledged is what credits the arm, and an arm that was neither exercised nor
rejected is a failure rather than a pass.

They also pin what this driver owns of the checked-in coverage record: the entry it
builds from the rows it measured, and which cases a forced arm is allowed to skip.
The record's own rules -- audit, merge, shrink refusal, bootstrap -- are the shared
implementation in `src/coverage_record.rs`, covered by `cargo test --lib
coverage_record`, so there is no second copy of them to pin here.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from verify_training_state import (
    PRECISION_ENV,
    SUITES,
    arm_evidence,
    arm_plan,
    canary_case,
    compare_snapshots,
    coverage_entry,
    forced_arm_report,
    load_coverage,
)

FORCED = (
    "RWKV LN1/low-rank FP32 forward forced topology=ln-low-rank-f32-column-slices "
    "requested=column-slices device=Intel(R) HD Graphics 520 batch=2\n"
)
UNAVAILABLE = (
    "RWKV LN1/low-rank FP32 forward topology=ln-low-rank-f32-column-slices "
    "requested=column-slices is unavailable on device=Intel(R) HD Graphics 520; "
    "using ln-low-rank-f32-one-row\n"
)


class TrainingStateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.expected = [{"step": 0, "optimizer_step": 1,
                          "optimizer_slot_steps": {"weight": 1},
                          "tensors": {"parameter/weight": "a", "exp_avg/weight": "b"}}]

    def test_exact_state_passes(self):
        self.assertEqual(compare_snapshots(self.expected, copy.deepcopy(self.expected)), [])

    def test_same_loss_cannot_hide_parameter_or_moment_drift(self):
        for tensor in self.expected[0]["tensors"]:
            observed = copy.deepcopy(self.expected)
            observed[0]["tensors"][tensor] = "changed"
            self.assertTrue(compare_snapshots(self.expected, observed))

    def test_missing_tensor_step_and_optimizer_clock_fail(self):
        observed = copy.deepcopy(self.expected)
        del observed[0]["tensors"]["parameter/weight"]
        self.assertTrue(compare_snapshots(self.expected, observed))
        self.assertTrue(compare_snapshots(self.expected, []))
        observed = copy.deepcopy(self.expected)
        observed[0]["optimizer_slot_steps"]["weight"] = 2
        self.assertTrue(compare_snapshots(self.expected, observed))



class ArmEvidenceTests(unittest.TestCase):
    def test_the_acknowledged_request_is_what_credits_the_arm(self):
        self.assertEqual(arm_evidence(FORCED, "column-slices"), (True, False))

    def test_another_arms_request_does_not_credit_this_one(self):
        # The producer logs the value it was asked for, so a run that forced a
        # different variant is never evidence that this one ran.
        self.assertEqual(arm_evidence(FORCED, "one-row"), (False, False))

    def test_a_rejected_request_is_reported_as_unavailable(self):
        self.assertEqual(arm_evidence(UNAVAILABLE, "column-slices"), (False, True))

    def test_an_unrelated_probe_stream_is_no_evidence(self):
        self.assertEqual(
            arm_evidence("training step: 6368 dispatches\n", "one-row"), (False, False)
        )


def arm_rows(arm, exercised=(), unavailable=(), cases=("batch2", "batch32", "odd")):
    return [
        {"arm": arm, "case": case, "forced": case in exercised,
         "unavailable": case in unavailable,
         "dispatched": case in exercised or case in unavailable}
        for case in cases
    ]


class ForcedArmReportTests(unittest.TestCase):
    def test_an_exercised_arm_is_verified_where_it_ran(self):
        notes, problems = forced_arm_report(
            arm_rows("one-row", exercised=("batch2", "batch32")), ("one-row",)
        )
        self.assertEqual(problems, [])
        self.assertTrue(any("2/3 case(s)" in note for note in notes), notes)

    def test_partly_rejected_arms_report_both_halves(self):
        notes, problems = forced_arm_report(
            arm_rows("column-slices", exercised=("batch2",), unavailable=("batch32", "odd")),
            ("column-slices",),
        )
        self.assertEqual(problems, [])
        self.assertTrue(any("ran in 1/3" in note for note in notes), notes)
        self.assertTrue(any("unavailable in 2 case(s)" in note for note in notes), notes)

    def test_a_device_that_rejects_the_variant_everywhere_is_a_note(self):
        notes, problems = forced_arm_report(
            arm_rows("two-rows", unavailable=("batch2", "batch32", "odd")), ("two-rows",)
        )
        self.assertEqual(problems, [])
        self.assertTrue(any("verifies nothing here" in note for note in notes), notes)

    def test_an_arm_the_producer_never_dispatched_for_is_a_note(self):
        # No variant line and no rejection: the fused producer never saw the request at
        # those shapes. That is a fact about the shape rather than a verification, and it
        # is recorded as `not_dispatched`; the coverage audit is what fails if the record
        # has this arm exercised on this device and it stopped running.
        notes, problems = forced_arm_report(arm_rows("one-row"), ("one-row",))
        self.assertEqual(problems, [])
        self.assertTrue(any("did not dispatch the request in 3 case(s)" in note
                            for note in notes), notes)

    def test_an_arm_with_no_verdict_at_all_fails(self):
        notes, problems = forced_arm_report([], ("one-row",))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("nothing was verified", problems[0])


def coverage(device="Intel(R) HD Graphics 520", suite="fp32"):
    return {
        "suite": suite,
        "device": device,
        "cases": {"batch2": {"steps": 6, "tensor_comparisons": 1752}},
        "arms": {"one-row": {"exercised": ["batch2"], "unavailable": []}},
    }


def coverage_rows(exercised=("batch2",), unavailable=()):
    rows = [{"suite": "fp32", "case": "batch2", "arm": "measured", "device": "D",
             "steps": 6, "tensor_comparisons": 1752, "forced": False,
             "unavailable": False, "dispatched": True}]
    rows.append({"suite": "fp32", "case": "batch2", "arm": "one-row", "device": "D",
                 "steps": 6, "tensor_comparisons": 1752,
                 "forced": "batch2" in exercised, "unavailable": "batch2" in unavailable,
                 "dispatched": "batch2" in exercised or "batch2" in unavailable})
    return rows


class CoverageEntryTests(unittest.TestCase):
    def test_the_entry_records_the_cases_and_where_each_arm_ran(self):
        suite = next(suite for suite in SUITES if suite.name == "fp32")
        entry = coverage_entry(suite, "D", coverage_rows(exercised=("batch2",)), ["batch2"])
        self.assertEqual(entry["cases"]["batch2"], {"steps": 6, "tensor_comparisons": 1752})
        self.assertEqual(entry["arms"]["one-row"]["exercised"], ["batch2"])
        self.assertEqual(
            entry["arms"]["two-rows"],
            {"exercised": [], "unavailable": [], "not_dispatched": [], "asserted": []},
        )

    def test_a_case_the_producer_never_dispatched_in_is_recorded_as_its_own_verdict(self):
        # `not_dispatched` is not a rejection: there is no fallback kernel to blame, the
        # fused producer simply is not in that shape. Keeping them apart is what lets the
        # next run carry the verdict without claiming the device rejected anything.
        suite = next(suite for suite in SUITES if suite.name == "fp32")
        row = {"suite": "fp32", "case": "batch2", "arm": "one-row", "device": "D",
               "steps": 6, "tensor_comparisons": 1752, "forced": False,
               "unavailable": False, "dispatched": False}
        measured = dict(row, arm="measured", dispatched=True)
        entry = coverage_entry(suite, "D", [measured, row], ["batch2"])
        self.assertEqual(entry["arms"]["one-row"]["exercised"], [])
        self.assertEqual(entry["arms"]["one-row"]["unavailable"], [])
        self.assertEqual(entry["arms"]["one-row"]["not_dispatched"], ["batch2"])

    def test_a_carried_claim_is_written_apart_from_a_measured_rejection(self):
        # The record has to say which verdicts this run reached and which it re-checked
        # through a canary, or a carried claim would look like fresh evidence forever.
        suite = next(suite for suite in SUITES if suite.name == "fp32")
        entry = coverage_entry(
            suite,
            "D",
            coverage_rows(exercised=(), unavailable=("batch2",)),
            ["batch2"],
            {"one-row": ["odd"]},
        )
        self.assertEqual(entry["arms"]["one-row"]["unavailable"], ["batch2"])
        self.assertEqual(entry["arms"]["one-row"]["asserted"], ["odd"])

    def test_an_unavailable_case_is_recorded_beside_the_exercised_one(self):
        suite = next(suite for suite in SUITES if suite.name == "fp32")
        entry = coverage_entry(
            suite, "D", coverage_rows(exercised=(), unavailable=("batch2",)), ["batch2"]
        )
        self.assertEqual(entry["arms"]["one-row"]["exercised"], [])
        self.assertEqual(entry["arms"]["one-row"]["unavailable"], ["batch2"])


CASE_ARGS = {
    "batch2": ["--batch", "2", "--warmup", "0", "--steps", "6"],
    "batch32": ["--batch", "32", "--warmup", "0", "--steps", "4"],
    "hidden32": ["--batch", "2", "--hidden", "32", "--warmup", "0", "--steps", "4"],
}


def golden_cases(*names):
    return [{"name": name, "args": CASE_ARGS[name]} for name in names]


def arm_entry(exercised=(), unavailable=(), asserted=(), not_dispatched=()):
    return {"exercised": list(exercised), "unavailable": list(unavailable),
            "not_dispatched": list(not_dispatched), "asserted": list(asserted)}


def coverage_entry_for(arms, device="Intel(R) HD Graphics 520", suite="fp16"):
    covered = coverage(device=device, suite=suite)
    covered["arms"] = arms
    return covered


class ArmGateTests(unittest.TestCase):
    """Which cases a forced arm pays for, given what the record already says.

    The gating is the runtime cut, so it is also the thing most able to hide a real
    loss of coverage: the tests below pin that it may only skip what the record
    claims unavailable, that a canary re-checks that claim every run, and that
    anything the record does not claim -- including a case it has never seen -- is
    still measured.
    """

    def test_a_device_with_no_entry_still_runs_every_case(self):
        cases = golden_cases("batch2", "batch32", "hidden32")
        must, canary, skipped = arm_plan(cases, "two-rows", None)
        self.assertEqual(must, ["batch2", "batch32", "hidden32"])
        self.assertIsNone(canary)
        self.assertEqual(skipped, [])

    def test_a_verified_case_is_always_run(self):
        cases = golden_cases("batch2", "batch32", "hidden32")
        entry = coverage_entry_for({
            "two-rows": arm_entry(exercised=("hidden32",), unavailable=("batch2", "batch32")),
        })
        must, canary, skipped = arm_plan(cases, "two-rows", entry)
        self.assertEqual(must, ["hidden32"])
        # The canary is the widest claimed case, which here is the deepest one too.
        self.assertEqual(canary, "batch2")
        self.assertEqual(skipped, ["batch2", "batch32"])

    def test_a_case_the_record_does_not_mention_is_run(self):
        # A golden case the device's entry never saw has no verdict to lean on, so it
        # is measured rather than assumed to be unavailable.
        cases = golden_cases("batch2", "batch32", "hidden32")
        entry = coverage_entry_for({"two-rows": arm_entry(unavailable=("batch32",))})
        must, canary, skipped = arm_plan(cases, "two-rows", entry)
        self.assertEqual(must, ["batch2", "hidden32"])
        self.assertEqual(canary, "batch32")
        self.assertEqual(skipped, ["batch32"])

    def test_a_carried_claim_is_still_gated_on(self):
        # `asserted` is the record's own claim being carried, not a hole in it: the
        # next run has to treat it exactly like a measured rejection.
        cases = golden_cases("batch2", "batch32")
        entry = coverage_entry_for({"two-rows": arm_entry(asserted=("batch2", "batch32"))})
        must, canary, skipped = arm_plan(cases, "two-rows", entry)
        self.assertEqual(must, [])
        self.assertEqual(canary, "batch2")
        self.assertEqual(skipped, ["batch2", "batch32"])

    def test_a_shape_the_producer_never_dispatched_in_is_gated_too(self):
        # The width-256 case is the FP32 suite's version of this: the fused producer is
        # not there, so forcing a variant replays the fallback for 26 seconds and can
        # only ever re-confirm that the arm is not in that shape.
        cases = golden_cases("batch2", "batch32", "hidden32")
        entry = coverage_entry_for({
            "two-rows": arm_entry(exercised=("batch2", "batch32"),
                                  not_dispatched=("hidden32",)),
        })
        must, canary, skipped = arm_plan(cases, "two-rows", entry)
        self.assertEqual(must, ["batch2", "batch32"])
        self.assertEqual(canary, "hidden32")
        self.assertEqual(skipped, ["hidden32"])

    def test_the_canary_is_the_widest_available_shape_then_the_deepest(self):
        cases = golden_cases("batch2", "batch32", "hidden32")
        # Width first: a variant's availability is a property of the shape the producer
        # dispatches, so the widest claimed-unavailable case is nearest the boundary.
        self.assertEqual(canary_case(cases, {"batch32", "hidden32"}), "batch32")
        # Then depth, then golden order, so the choice does not depend on set iteration.
        self.assertEqual(canary_case(cases, {"batch2", "batch32", "hidden32"}), "batch2")
        self.assertEqual(canary_case(cases, {"batch2", "hidden32"}), "batch2")
        self.assertIsNone(canary_case(cases, set()))


class SuiteTests(unittest.TestCase):
    def test_each_suite_golden_names_the_policy_it_was_recorded_under(self):
        # The suite name is the policy value, so a golden whose `env` block disagrees
        # with its suite is a suite running under the wrong precision.
        for suite in SUITES:
            document = json.loads(Path(suite.golden).read_text(encoding="utf-8"))
            self.assertEqual(document.get("env", {}).get(PRECISION_ENV, "fp32"), suite.name)

    def test_every_suite_has_a_non_empty_arm_list_without_duplicates(self):
        for suite in SUITES:
            self.assertTrue(suite.arms, suite)
            self.assertEqual(len(set(suite.arms)), len(suite.arms), suite)

    def test_the_fp32_suite_still_forces_the_shipped_variants(self):
        fp32 = next(suite for suite in SUITES if suite.name == "fp32")
        self.assertEqual(set(fp32.arms), {"one-row", "two-rows", "column-slices"})


if __name__ == "__main__":
    unittest.main()
