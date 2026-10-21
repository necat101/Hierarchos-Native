"""Negative controls for the checked-in topology decision record (no GPU needed).

The record is only useful if a decision that moves fails and a timing that moves
does not: a shared GPU's medians drift between runs while the variant it selected
usually does not, so the tests below pin both halves -- a changed `selected` is a
failure, drift is a note, an unrecorded measurement is a failure, and another
device's entries are neither.

The unanimous-round rule added a second thing to get wrong: a variant may only displace
the structural default by beating it in every round, so the evidence an entry carries has
to agree with the choice it records. A variant "selected" after winning two of three
rounds, a kept default that claims a round it won, and a candidate that stops being timed
are each failures here.
"""
import json
import tempfile
import unittest
from pathlib import Path

from verify_topology_decisions import (
    compare,
    decision_key,
    describe,
    geometry_report,
    load_record,
    merge,
    read_jsonl,
    recordable,
    stability_problems,
)

GEN9 = "Intel(R) HD Graphics 520"
AMD = "AMD Radeon Graphics"
ONE_ROW = "ln-low-rank-f32-one-row"
TWO_ROWS = "ln-low-rank-f32-two-rows"


def decision(device=GEN9, batch_pairs=1, selected=ONE_ROW,
             one=0.22848, two=0.27713, family="ln-low-rank-f32",
             round_wins=None, rounds=3, decisive_margin_fraction=0.0,
             margin_fraction=0.0, contention_spread=1.02, contended=False):
    return {
        "family": family,
        "device": device,
        "subgroup_size": 32,
        "width": 128,
        "w_rank": 32,
        "a_rank": 32,
        "g_rank": 64,
        "batch_pairs": batch_pairs,
        "has_unpaired_tail": False,
        "timings_ms": {
            ONE_ROW: one,
            TWO_ROWS: two,
        },
        "rounds": rounds,
        "round_wins": {ONE_ROW: 0, TWO_ROWS: 0} if round_wins is None else round_wins,
        "decisive_margin_fraction": decisive_margin_fraction,
        "selected": selected,
        "structural_default": ONE_ROW,
        "margin_fraction": margin_fraction,
        "contention_spread": contention_spread,
        "contended": contended,
    }


class PerformanceMarginTests(unittest.TestCase):
    def test_a_unanimous_win_must_clear_its_recorded_admission_margin(self):
        entry = decision(selected=TWO_ROWS, round_wins={ONE_ROW: 0, TWO_ROWS: 3},
                         decisive_margin_fraction=0.05)
        entry['required_margin_fraction'] = 0.15
        self.assertTrue(any('below' in p for p in stability_problems(entry, 'AMD profile')))
        entry['decisive_margin_fraction'] = 0.2
        self.assertEqual(stability_problems(entry, 'AMD profile'), [])
        entry['required_margin_fraction'] = float('nan')
        self.assertTrue(any('invalid' in p for p in stability_problems(entry, 'AMD profile')))


class ReplayParsingTests(unittest.TestCase):
    def write(self, lines):
        scratch = tempfile.TemporaryDirectory(prefix="topology-record-")
        self.addCleanup(scratch.cleanup)
        path = Path(scratch.name) / "decisions.jsonl"
        path.write_text(lines, encoding="utf-8")
        return path

    def test_the_last_line_for_a_geometry_wins(self):
        path = self.write(
            "\n".join(
                json.dumps(entry)
                for entry in (decision(), decision(selected="ln-low-rank-f32-two-rows"))
            )
            + "\n"
        )
        measured, problems = read_jsonl(path)
        self.assertEqual(problems, [])
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["selected"], "ln-low-rank-f32-two-rows")

    def test_malformed_lines_are_reported_not_skipped(self):
        path = self.write(
            json.dumps(decision())
            + "\nnot json\n"
            + json.dumps([1, 2, 3])
            + "\n"
            + json.dumps({"family": "ln-low-rank-f32"})
            + "\n"
        )
        measured, problems = read_jsonl(path)
        self.assertEqual(len(measured), 1)
        self.assertEqual(len(problems), 3, problems)
        self.assertTrue(any("not JSON" in problem for problem in problems))
        self.assertTrue(any("not a JSON object" in problem for problem in problems))
        self.assertTrue(any("missing" in problem for problem in problems))

    def test_a_missing_key_field_is_a_problem(self):
        with self.assertRaises(KeyError):
            decision_key({"family": "ln-low-rank-f32"})


class RecordFileTests(unittest.TestCase):
    def write(self, document):
        scratch = tempfile.TemporaryDirectory(prefix="topology-record-")
        self.addCleanup(scratch.cleanup)
        path = Path(scratch.name) / "topology_decisions.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_a_missing_record_is_a_problem(self):
        entries, problems = load_record(Path("does-not-exist.json"))
        self.assertEqual(entries, [])
        self.assertTrue(any("missing" in problem for problem in problems))

    def test_an_unsupported_schema_is_rejected(self):
        path = self.write({"schema": 2, "decisions": []})
        entries, problems = load_record(path)
        self.assertEqual(entries, [])
        self.assertTrue(any("unsupported schema" in problem for problem in problems))

    def test_duplicates_and_unsorted_entries_mean_hand_editing(self):
        first, second = decision(batch_pairs=1), decision(batch_pairs=4)
        path = self.write({"schema": 1, "decisions": [second, first, first]})
        entries, problems = load_record(path)
        self.assertEqual(len(entries), 3)
        self.assertTrue(any("duplicate" in problem for problem in problems))
        self.assertTrue(any("not sorted" in problem for problem in problems))

    def test_a_sorted_record_loads_clean(self):
        path = self.write(
            {"schema": 1, "decisions": [decision(batch_pairs=1), decision(batch_pairs=4)]}
        )
        entries, problems = load_record(path)
        self.assertEqual(problems, [])
        self.assertEqual([entry["batch_pairs"] for entry in entries], [1, 4])

    def test_a_record_claiming_a_variant_that_did_not_win_every_round_is_rejected(self):
        # The rule is the entry's own evidence: a variant recorded as selected after two of
        # three rounds contradicts the only way it could have been selected.
        contradictory = decision(
            selected=TWO_ROWS, round_wins={ONE_ROW: 0, TWO_ROWS: 2}, decisive_margin_fraction=0.05
        )
        path = self.write({"schema": 1, "decisions": [contradictory]})
        _, problems = load_record(path)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("winning only 2/3 round(s)", problems[0])

    def test_a_record_whose_default_claims_a_round_is_rejected(self):
        path = self.write(
            {"schema": 1, "decisions": [decision(round_wins={ONE_ROW: 1, TWO_ROWS: 2})]}
        )
        _, problems = load_record(path)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("kept ln-low-rank-f32-one-row as the default", problems[0])

    def test_a_record_without_stability_evidence_is_audited_for_its_choice_alone(self):
        legacy = decision()
        for field in ("structural_default", "rounds", "round_wins", "decisive_margin_fraction",
                      "contention_spread", "contended"):
            legacy.pop(field)
        path = self.write({"schema": 1, "decisions": [legacy]})
        entries, problems = load_record(path)
        self.assertEqual(problems, [])
        self.assertEqual(len(entries), 1)


class CompareTests(unittest.TestCase):
    def test_matching_selection_does_not_hide_a_changed_performance_admission_policy(self):
        replay = decision()
        replay['required_margin_fraction'] = 0.15
        problems, _ = compare([replay], [decision()], GEN9)
        self.assertTrue(any('admission margin changed' in p for p in problems))

    def test_the_recorded_choice_reproducing_is_a_pass_with_a_note(self):
        problems, notes = compare([decision()], [decision()], GEN9)
        self.assertEqual(problems, [])
        self.assertTrue(any("reproduced" in note for note in notes))

    def test_a_selection_that_moves_is_a_failure(self):
        unanimous = dict(round_wins={ONE_ROW: 0, TWO_ROWS: 3}, decisive_margin_fraction=0.12)
        problems, _ = compare(
            [decision(selected=TWO_ROWS, **unanimous)],
            [decision(selected=TWO_ROWS, **unanimous)],
            GEN9,
        )
        # The same unanimous choice replayed is a pass, so the failure below is the
        # movement itself rather than the new fields disagreeing.
        self.assertEqual(problems, [])
        problems, _ = compare([decision(selected=TWO_ROWS, **unanimous)], [decision()], GEN9)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn(f"recorded {ONE_ROW}", problems[0])
        self.assertIn(f"now selects {TWO_ROWS}", problems[0])

    def test_a_measurement_that_was_never_recorded_is_a_failure(self):
        problems, _ = compare([decision(batch_pairs=4)], [decision()], GEN9)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("not recorded", problems[0])

    def test_timing_drift_is_a_note_and_not_a_failure(self):
        problems, notes = compare(
            [decision(one=0.9, two=0.95)], [decision(one=0.22848, two=0.27713)], GEN9
        )
        self.assertEqual(problems, [])
        self.assertTrue(any("vs recorded" in note for note in notes), notes)

    def test_small_timing_drift_is_not_even_a_note(self):
        problems, notes = compare(
            [decision(one=0.23, two=0.28)], [decision(one=0.22848, two=0.27713)], GEN9
        )
        self.assertEqual(problems, [])
        self.assertFalse(any("vs recorded" in note for note in notes), notes)

    def test_another_devices_entries_are_never_compared(self):
        problems, notes = compare([decision()], [decision(), decision(device=AMD)], GEN9)
        self.assertEqual(problems, [])
        self.assertTrue(any("other device" in note for note in notes), notes)
        self.assertTrue(any(AMD in note for note in notes), notes)

    def test_keeping_the_default_on_a_sub_margin_lead_is_recorded_as_evidence(self):
        contested = decision(
            round_wins={ONE_ROW: 0, TWO_ROWS: 2},
            margin_fraction=0.019,
        )
        problems, notes = compare([contested], [contested], GEN9)
        self.assertEqual(problems, [])
        self.assertTrue(any("1.9% median lead" in note for note in notes), notes)
        self.assertTrue(any("two-rows 2" in note for note in notes), notes)

    def test_a_contended_record_entry_may_not_select_a_variant(self):
        problems = stability_problems(
            decision(selected=TWO_ROWS, round_wins={ONE_ROW: 0, TWO_ROWS: 3},
                     decisive_margin_fraction=0.12, contended=True, contention_spread=1.4),
            "geometry",
        )
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("contended sweep", problems[0])
        self.assertIn("is not evidence", problems[0])

    def test_a_spread_that_contradicts_its_own_flag_is_rejected(self):
        # An entry is hand-edited evidence if its numbers do not agree with its flag.
        quiet_but_wide = decision(contended=False, contention_spread=1.9)
        problems = stability_problems(quiet_but_wide, "geometry")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("not marked contended", problems[0])
        contended_but_tight = decision(contended=True, contention_spread=1.05)
        problems = stability_problems(contended_but_tight, "geometry")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("under the 1.15x threshold", problems[0])

    def test_recordable_drops_the_contended_decisions(self):
        quiet, contended = recordable(
            [decision(), decision(batch_pairs=4, contended=True, contention_spread=1.3)]
        )
        self.assertEqual([entry["batch_pairs"] for entry in quiet], [1])
        self.assertEqual([entry["batch_pairs"] for entry in contended], [4])

    def test_a_variant_may_only_be_selected_unanimously(self):
        problems = stability_problems(
            decision(selected=TWO_ROWS, round_wins={ONE_ROW: 0, TWO_ROWS: 3},
                     decisive_margin_fraction=0.12),
            "geometry",
        )
        self.assertEqual(problems, [])
        problems = stability_problems(
            decision(selected=TWO_ROWS, round_wins={ONE_ROW: 1, TWO_ROWS: 2}),
            "geometry",
        )
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("only by beating it in every round", problems[0])

    def test_a_contended_replay_does_not_move_the_record(self):
        # The case that made this rule necessary: with a second process on the GPU the
        # recorded column-slice win failed the margin in every round and the replay
        # reported the default. That is the load, not the kernels, so it is a note.
        contended = decision(
            selected=ONE_ROW, round_wins={ONE_ROW: 0, TWO_ROWS: 2},
            contention_spread=1.42, contended=True,
        )
        problems, notes = compare([contended], [decision(selected=TWO_ROWS)], GEN9)
        self.assertEqual(problems, [], problems)
        self.assertTrue(any("contended sweep" in note for note in notes), notes)
        self.assertTrue(any("1.42x" in note for note in notes), notes)
        self.assertTrue(any("stands" in note for note in notes), notes)

    def test_a_contended_replay_at_an_unrecorded_geometry_is_a_note(self):
        problems, notes = compare(
            [decision(batch_pairs=4, contended=True, contention_spread=1.30)], [decision()], GEN9
        )
        self.assertEqual(problems, [], problems)
        self.assertTrue(any("cannot record a decision" in note for note in notes), notes)

    def test_a_contended_replay_that_claims_a_variant_is_still_a_failure(self):
        # Skipping the comparison must not skip the entry's own contradictions.
        problems, _ = compare(
            [decision(selected=TWO_ROWS, round_wins={ONE_ROW: 0, TWO_ROWS: 3},
                      decisive_margin_fraction=0.1, contended=True, contention_spread=1.4)],
            [decision()],
            GEN9,
        )
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("contended sweep", problems[0])

    def test_a_candidate_that_stops_being_timed_is_a_failure(self):
        # A matching `selected` must not be able to hide a variant this device no longer
        # exercises at that geometry.
        narrowed = decision()
        narrowed["timings_ms"].pop(TWO_ROWS)
        problems, _ = compare([narrowed], [decision()], GEN9)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("this run did not time it", problems[0])

    def test_a_newly_timed_candidate_is_a_note(self):
        widened = decision()
        widened["timings_ms"]["ln-low-rank-f32-column-slices"] = 0.2
        problems, notes = compare([widened], [decision()], GEN9)
        self.assertEqual(problems, [])
        self.assertTrue(any("newly measured" in note for note in notes), notes)

    def test_a_recorded_geometry_this_fixture_does_not_exercise_is_a_note(self):
        problems, notes = compare([decision()], [decision(), decision(batch_pairs=32)], GEN9)
        self.assertEqual(problems, [])
        self.assertTrue(any("batch_pairs=32" in note for note in notes), notes)

    def test_describe_names_the_geometry(self):
        self.assertEqual(
            describe(decision(batch_pairs=4)),
            f"ln-low-rank-f32 device={GEN9!r} subgroup=32 width=128 ranks=32/32/64 batch_pairs=4",
        )


GEOMETRIES = (("width64-batch2", ["--batch", "2"]), ("width128-batch2", ["--batch", "2"]))


class GeometryCoverageTests(unittest.TestCase):
    def test_every_geometry_that_measured_something_passes_with_a_note(self):
        notes, problems = geometry_report(
            {"width64-batch2": [decision()], "width128-batch2": [decision(batch_pairs=4)]},
            GEOMETRIES,
        )
        self.assertEqual(problems, [])
        self.assertEqual(len(notes), 2, notes)
        self.assertTrue(any("batch_pairs=4" in note for note in notes), notes)

    def test_the_note_names_the_variant_each_geometry_selected(self):
        notes, _ = geometry_report({"width64-batch2": [decision()],
                                    "width128-batch2": [decision()]}, GEOMETRIES)
        self.assertTrue(all("ln-low-rank-f32-one-row" in note for note in notes), notes)

    def test_a_geometry_that_stopped_measuring_fails(self):
        # The record is only as wide as the geometries that measured it, so a shape
        # that goes quiet must fail rather than shrink the audited coverage.
        notes, problems = geometry_report({"width64-batch2": [decision()]}, GEOMETRIES)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("width128-batch2", problems[0])
        self.assertIn("measured no decision", problems[0])


class MergeTests(unittest.TestCase):
    def test_a_replay_replaces_its_own_key_and_keeps_other_devices(self):
        recorded = [decision(), decision(device=AMD)]
        merged = merge(recorded, [decision(selected="ln-low-rank-f32-two-rows", one=0.1)])
        self.assertEqual(len(merged), 2)
        by_device = {entry["device"]: entry for entry in merged}
        self.assertEqual(by_device[GEN9]["selected"], "ln-low-rank-f32-two-rows")
        self.assertEqual(by_device[AMD]["selected"], "ln-low-rank-f32-one-row")

    def test_the_merged_record_is_sorted(self):
        merged = merge([], [decision(batch_pairs=4), decision(batch_pairs=1), decision(device=AMD)])
        self.assertEqual([decision_key(entry) for entry in merged], sorted(decision_key(entry) for entry in merged))


if __name__ == "__main__":
    unittest.main()
