"""Negative controls for the folded-publication profile guard (no GPU needed).

The committed profiles are what the audit quotes when it says the fully folded
state-backward rung cannot lead on Intel Gen9 because neither its arrival
counters nor its batch-order reduction is resolvable. That conclusion rests on
the artifact, so the artifact has to be checked rather than trusted:

* the numbers are re-derived from the legs, so a hand-edited delta, ratio or
  resolution fails;
* the gates the legs cleared are recorded and checked, so a profile that timed
  two graphs is not read as two components;
* the claim is checked in the direction that matters -- a fold that starts
  leading, or a component that becomes visible, fails here instead of quietly
  contradicting the document that cites it.

Each test mutates one field of the committed profile and asserts the guard
rejects it. They are the reason the committed file is worth citing: an edit that
makes the rung look better is a failing check, not a better-looking number.
"""
import copy
import json
import unittest
from pathlib import Path

from check_folded_publication_profile import (
    MIN_SAMPLES,
    PHANTOM,
    Problem,
    RECORDED_DEVICE,
    STEPS,
    SWEEP,
    check_compare,
    check_profile,
    load,
    spec_named,
)

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
FP32 = VALIDATION / "tbptt_folded_publication_profile.json"
FP16 = VALIDATION / "tbptt_folded_publication_profile_fp16.json"


def committed(path=FP32):
    return copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))


def row_for(profile, batch):
    for row in profile["batches"]:
        if row["batch"] == batch:
            return row
    raise AssertionError(f"the committed profile has no batch {batch}")


def legs_of(row):
    return {leg["label"]: leg for leg in row["legs"]}


def retime(profile, batch, **timings):
    """Rewrite one row's legs and every summary the guard re-derives from them.

    Only the legs are inputs; the medians, the ratio, the two deltas and the
    flags are recomputed here the way the binary computes them, so a fixture can
    describe a different measurement without ever promising inconsistent numbers.
    The guard independently re-derives the same values, so a mistake in this
    helper shows up as the arithmetic check rather than as a passing test.
    """
    row = row_for(profile, batch)
    legs = legs_of(row)
    for label, median in timings.items():
        leg = legs[label]
        leg["median_ms"] = median
        leg["min_ms"] = min(leg["min_ms"], median)
        leg["max_ms"] = max(leg["max_ms"], median)
        if leg["iqr_ms"] > leg["max_ms"] - leg["min_ms"]:
            leg["iqr_ms"] = leg["max_ms"] - leg["min_ms"]
    for leg in row["legs"]:
        leg["ms_per_step"] = leg["median_ms"] / row["steps"]
    staged, fused, folded = (legs[depth]["median_ms"] for depth in ("staged", "fused", "folded"))
    resolution = max(leg["iqr_ms"] for leg in row["legs"])
    row["resolution_ms"] = resolution
    row["range_resolution_ms"] = max((leg["max_ms"] - leg["min_ms"]) / 2.0 for leg in row["legs"])
    row["folded_over_fused"] = folded / fused
    row["folded_minus_fused_ms"] = folded - fused
    row["folded_beats_fused_beyond_resolution"] = fused - folded > resolution
    if PHANTOM in legs:
        without = legs[PHANTOM]["median_ms"]
        row["folded_minus_without_reduction_ms"] = folded - without
        row["without_reduction_minus_fused_ms"] = without - fused
        row["reduction_tail_beyond_resolution"] = abs(folded - without) > resolution
        row["counters_beyond_resolution"] = abs(without - fused) > resolution
    return row


class CommittedProfileTests(unittest.TestCase):
    def test_the_committed_profiles_hold_the_claim(self):
        for name in ("fp32", "fp16-storage"):
            spec = spec_named(name)
            rows = check_profile(load(spec.path), spec)
            self.assertEqual([row["batch"] for row in rows], list(SWEEP))
            for row in rows:
                self.assertFalse(row["folded_beats_fused_beyond_resolution"], row["batch"])
                if PHANTOM in legs_of(row):
                    self.assertFalse(row["reduction_tail_beyond_resolution"], row["batch"])
                    self.assertFalse(row["counters_beyond_resolution"], row["batch"])

    def test_both_profiles_cover_the_whole_sweep(self):
        for path, policy in ((FP32, None), (FP16, "fp16-storage")):
            profile = committed(path)
            self.assertEqual(profile["parameter_storage"], policy)
            self.assertEqual([row["batch"] for row in profile["batches"]], list(SWEEP))


class HeaderTests(unittest.TestCase):
    def test_a_profile_that_did_not_pass_fails(self):
        profile = committed()
        profile["result"] = "fail"
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_profile_of_another_kind_fails(self):
        profile = committed()
        profile["profile"] = "backward-fusion"
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_profile_recorded_under_the_wrong_policy_fails(self):
        profile = committed()
        profile["parameter_storage"] = "fp16-storage"
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_profile_timed_with_too_few_samples_fails(self):
        profile = committed()
        for row in profile["batches"]:
            row["samples"] = MIN_SAMPLES - 1
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_profile_with_shorter_windows_fails(self):
        profile = committed()
        for row in profile["batches"]:
            row["steps"] = STEPS - 1
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_another_adapters_profile_fails(self):
        profile = committed()
        for row in profile["batches"]:
            row["device"] = "AMD Radeon Graphics"
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

        profile = committed()
        row_for(profile, 16)["device"] = "AMD Radeon Graphics"
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))


class SweepTests(unittest.TestCase):
    def test_a_dropped_geometry_fails(self):
        profile = committed()
        profile["batches"] = [row for row in profile["batches"] if row["batch"] != 32]
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_duplicated_geometry_fails(self):
        profile = committed()
        profile["batches"][-1] = copy.deepcopy(profile["batches"][3])
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_missing_phantom_leg_fails_above_batch_one(self):
        profile = committed()
        row = row_for(profile, 4)
        row["legs"] = [leg for leg in row["legs"] if leg["label"] != PHANTOM]
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_phantom_leg_at_batch_one_fails(self):
        profile = committed()
        row = row_for(profile, 1)
        row["legs"].append(copy.deepcopy(legs_of(row_for(committed(), 4))[PHANTOM]))
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_row_that_disagrees_about_its_geometry_fails(self):
        profile = committed()
        row_for(profile, 8)["width"] = 256
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))


class DepthTests(unittest.TestCase):
    def test_a_leg_that_pinned_another_depth_fails(self):
        profile = committed()
        row_for(profile, 4)["folded_schedule"] = row_for(profile, 4)["fused_schedule"]
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_missing_depth_fails(self):
        profile = committed()
        row = row_for(profile, 4)
        row["legs"] = [leg for leg in row["legs"] if leg["label"] != "staged"]
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_an_unexpected_leg_fails(self):
        profile = committed()
        row = row_for(profile, 4)
        extra = copy.deepcopy(legs_of(row)["folded"])
        extra["label"] = "folded-with-reduction-disabled"
        row["legs"].append(extra)
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))


class ArithmeticTests(unittest.TestCase):
    def test_a_ratio_that_disagrees_with_the_legs_fails(self):
        profile = committed()
        row_for(profile, 4)["folded_over_fused"] *= 1.01
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_hand_edited_delta_fails(self):
        profile = committed()
        row_for(profile, 8)["folded_minus_fused_ms"] = -0.9
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_hand_edited_resolution_fails(self):
        profile = committed()
        row_for(profile, 32)["resolution_ms"] *= 4
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_hand_edited_range_resolution_fails(self):
        profile = committed()
        row_for(profile, 4)["range_resolution_ms"] *= 0.5
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_tail_that_disagrees_with_the_legs_fails(self):
        profile = committed()
        row_for(profile, 16)["folded_minus_without_reduction_ms"] = 0.4
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_median_outside_its_own_range_fails(self):
        profile = committed()
        leg = legs_of(row_for(profile, 8))["fused"]
        leg["median_ms"] = leg["max_ms"] + 1.0
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))


class FlagTests(unittest.TestCase):
    def test_a_hand_flipped_lead_flag_fails(self):
        profile = committed()
        row_for(profile, 16)["folded_beats_fused_beyond_resolution"] = True
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_hand_flipped_component_flag_fails(self):
        profile = committed()
        row_for(profile, 4)["counters_beyond_resolution"] = True
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))


class ClaimTests(unittest.TestCase):
    def test_a_fold_that_starts_leading_fails(self):
        profile = committed()
        row = retime(profile, 32, folded=legs_of(row_for(profile, 32))["folded"]["median_ms"] - 4.0)
        self.assertTrue(row["folded_beats_fused_beyond_resolution"])
        with self.assertRaises(Problem) as caught:
            check_profile(profile, spec_named("fp32"))
        self.assertIn("now leads", str(caught.exception))

    def test_a_reduction_tail_that_becomes_visible_fails(self):
        profile = committed()
        row = retime(
            profile,
            4,
            **{PHANTOM: legs_of(row_for(profile, 4))[PHANTOM]["median_ms"] - 3.0},
        )
        self.assertTrue(row["reduction_tail_beyond_resolution"])
        with self.assertRaises(Problem) as caught:
            check_profile(profile, spec_named("fp32"))
        self.assertIn("batch-order reduction", str(caught.exception))

    def test_counters_that_become_visible_are_named_too(self):
        # Moving the phantom leg away from the fold surfaces the counter delta as
        # well, because `counters = without - fused` and `tail = folded - without`
        # are the two halves of the same gap. The guard must name both rather than
        # report only the first one it tests.
        profile = committed()
        retime(profile, 32, **{PHANTOM: legs_of(row_for(profile, 32))[PHANTOM]["median_ms"] + 5.0})
        with self.assertRaises(Problem) as caught:
            check_profile(profile, spec_named("fp32"))
        self.assertIn("arrival counters", str(caught.exception))
        self.assertIn("batch-order reduction", str(caught.exception))


class GateTests(unittest.TestCase):
    def test_a_nonzero_allowance_fails(self):
        profile = committed()
        row_for(profile, 4)["key_parameter_atol"] = 1.0e-6
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_zero_staged_magnitude_fails(self):
        profile = committed()
        row_for(profile, 4)["staged_key_parameter_max_abs"] = 0.0
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_clean_fold_that_moved_the_gradients_fails(self):
        profile = committed()
        row_for(profile, 32)["folded_key_parameter_max_abs"] = 1.0e-3
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_phantom_leg_that_published_anyway_fails(self):
        profile = committed()
        row_for(profile, 8)["without_reduction_key_parameter_max_abs"] = 0.0
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_missing_gate_record_fails(self):
        profile = committed()
        del row_for(profile, 16)["folded_key_parameter_max_abs"]
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))

    def test_a_phantom_delta_recorded_at_batch_one_fails(self):
        profile = committed()
        row_for(profile, 1)["without_reduction_key_parameter_max_abs"] = 0.01
        with self.assertRaises(Problem):
            check_profile(profile, spec_named("fp32"))


class CompareTests(unittest.TestCase):
    def test_the_committed_profile_compares_clean_against_itself(self):
        profile = committed()
        notes = check_compare(profile, copy.deepcopy(profile), spec_named("fp32"))
        self.assertEqual(notes, [])

    def test_a_fresh_run_that_keeps_the_tie_passes_with_a_note_when_it_drifts(self):
        committed_profile = committed()
        fresh = committed()
        fused = legs_of(row_for(fresh, 32))["fused"]["median_ms"]
        resolution = row_for(fresh, 32)["resolution_ms"]
        # One millisecond behind `fused` is still inside the run's resolution, so
        # the claim holds -- but the ratio moved enough to be worth reporting.
        row = retime(fresh, 32, folded=fused + min(1.0, resolution * 0.8))
        self.assertFalse(row["folded_beats_fused_beyond_resolution"])
        notes = check_compare(fresh, committed_profile, spec_named("fp32"))
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("folded/fused moved", notes[0])

    def test_a_fresh_run_whose_fold_leads_fails(self):
        committed_profile = committed()
        fresh = committed()
        retime(fresh, 8, folded=legs_of(row_for(fresh, 8))["folded"]["median_ms"] - 4.0)
        with self.assertRaises(Problem):
            check_compare(fresh, committed_profile, spec_named("fp32"))

    def test_a_fresh_run_that_lost_its_gates_fails(self):
        committed_profile = committed()
        fresh = committed()
        row_for(fresh, 32)["folded_key_parameter_max_abs"] = 2.0e-3
        with self.assertRaises(Problem):
            check_compare(fresh, committed_profile, spec_named("fp32"))


if __name__ == "__main__":
    unittest.main()
