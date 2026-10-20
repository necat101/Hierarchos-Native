"""Negative controls for the stack ladder's probe expectation (no GPU needed).

The expectation is device-keyed: the vendor layer selects different kernels on
different parts, so the same fixture does not dispatch the same number of times
everywhere. These tests pin that down -- a run is compared against its own
device's entry, another device's numbers are never substituted for it, and a
device with no entry fails instead of passing by default.

The committed receipt has its own guard, and these tests are its negative
controls: a dropped or renamed step, a step that is no longer PASS, and a
coverage count below its floor all have to fail, because each of those is a
receipt that would otherwise be cited as a green ladder it did not record.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from check_training_stack_receipt import (
    COVERAGE,
    LADDER_STEPS,
    RECORDED_ADAPTER,
    Problem,
    check_adapter,
    check_compare,
    check_coverage,
    check_results,
    check_rows,
    check_structure,
    committed_floors,
)
from verify_training_stack import (
    PROBE_ARGS,
    SHADER_TOOLS,
    ensure_shader_toolchain,
    expectation_problems,
)

RECEIPT = Path(__file__).resolve().parents[1] / "validation" / "training_stack_receipt.json"

AMD = "AMD Radeon Graphics"
GEN9 = "Intel(R) HD Graphics 520"


def expectation():
    return {
        "probe_args": list(PROBE_ARGS),
        "shader_barrier_count_max": 3076,
        "devices": {
            AMD: {"mean_loss": 5.543217, "dispatch_count": 6464},
            GEN9: {"mean_loss": 5.5432177, "dispatch_count": 6368},
        },
    }


def observed(device=GEN9, dispatches=6368, barriers=2983, loss=5.5432177):
    return {"device": device, "dispatch_count": dispatches,
            "shader_barrier_count": barriers, "mean_loss": loss}


class ProbeExpectationTests(unittest.TestCase):
    def test_each_device_is_compared_against_its_own_entry(self):
        self.assertEqual(expectation_problems(observed(GEN9, 6368, loss=5.5432177), expectation(), GEN9), [])
        self.assertEqual(expectation_problems(observed(AMD, 6464, 3076, 5.543217), expectation(), AMD), [])

    def test_another_devices_dispatch_count_is_not_substituted(self):
        # The Gen9 count is what this part records; comparing it against AMD's
        # 6464 is the mistake the per-device entry exists to prevent.
        problems = expectation_problems(observed(GEN9, 6368), expectation(), GEN9)
        self.assertEqual(problems, [])
        self.assertTrue(expectation_problems(observed(AMD, 6368), expectation(), AMD))

    def test_an_unrecorded_device_fails_rather_than_defaults(self):
        problems = expectation_problems(observed("Some Other GPU"), expectation(), "Some Other GPU")
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any("no dispatch count recorded" in problem for problem in problems))
        self.assertTrue(any("no expectation recorded" in problem for problem in problems))

    def test_loss_must_match_exactly_for_this_device(self):
        problems = expectation_problems(observed(GEN9, loss=5.543217), expectation(), GEN9)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("loss", problems[0])

    def test_barrier_count_is_an_upper_bound(self):
        self.assertEqual(expectation_problems(observed(GEN9, barriers=2000), expectation(), GEN9), [])
        self.assertEqual(expectation_problems(observed(GEN9, barriers=3076), expectation(), GEN9), [])
        self.assertTrue(expectation_problems(observed(GEN9, barriers=3077), expectation(), GEN9))

    def test_probe_argument_drift_fails(self):
        drifted = copy.deepcopy(expectation())
        drifted["probe_args"] = PROBE_ARGS[:-2]
        problems = expectation_problems(observed(), drifted, GEN9)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("probe arguments", problems[0])


class ShaderToolchainTests(unittest.TestCase):
    """The ladder's toolchain step, with every dependency injected.

    The live paths are exercised by the ladder itself; these pin the decisions it
    makes -- what goes on `PATH`, when the installer is allowed to run, and what the
    blocker says -- without a compiler, a download, or a GPU.
    """

    def test_a_resolved_toolchain_leaves_path_alone(self):
        environment = {"PATH": "/usr/bin"}
        missing, detail = ensure_shader_toolchain(
            lookup=lambda name: f"/usr/bin/{name}",
            directories=lambda: [],
            environment=environment,
        )
        self.assertEqual(missing, [])
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertIn("resolve on PATH", detail)

    def test_a_fresh_clone_reports_what_to_install(self):
        environment = {"PATH": "/usr/bin"}
        missing, detail = ensure_shader_toolchain(
            lookup=lambda name: None, directories=lambda: [], environment=environment
        )
        self.assertEqual(missing, list(SHADER_TOOLS))
        self.assertIn("install_shader_toolchain.sh", detail)
        self.assertIn("16.5.0", detail)
        self.assertEqual(environment["PATH"], "/usr/bin")

    def test_the_repository_install_is_prepended_to_path(self):
        with tempfile.TemporaryDirectory(prefix="toolchain-") as scratch:
            environment = {"PATH": "/usr/bin"}

            def lookup(name):
                # The pinned directory does not exist in this fake machine; a tool
                # resolves only once that directory is ahead of the system PATH.
                return f"{scratch}/{name}" if environment["PATH"].startswith(scratch) else None

            missing, detail = ensure_shader_toolchain(
                lookup=lookup, directories=lambda: [Path(scratch)], environment=environment
            )
            self.assertEqual(missing, [])
            self.assertTrue(environment["PATH"].startswith(scratch))
            self.assertIn("prepended", detail)

    def test_the_installer_runs_once_and_its_install_then_resolves(self):
        with tempfile.TemporaryDirectory(prefix="toolchain-") as scratch:
            environment = {"PATH": "/usr/bin"}
            calls = []

            def directories():
                return [Path(scratch)] if calls else []

            def install():
                calls.append("install")
                return 0, ""

            def lookup(name):
                return f"{scratch}/{name}" if environment["PATH"].startswith(scratch) else None

            missing, detail = ensure_shader_toolchain(
                lookup=lookup, directories=directories, install=install, environment=environment
            )
        self.assertEqual(missing, [])
        self.assertEqual(calls, ["install"])
        self.assertIn("installed", detail)

    def test_a_failing_installer_is_reported_rather_than_retried(self):
        environment = {"PATH": "/usr/bin"}
        calls = []

        def install():
            calls.append("install")
            return 1, "error: this script installs the Windows x86-64 builds"

        missing, detail = ensure_shader_toolchain(
            lookup=lambda name: None, directories=lambda: [], install=install, environment=environment
        )
        self.assertEqual(missing, list(SHADER_TOOLS))
        self.assertEqual(calls, ["install"])
        self.assertIn("Windows x86-64", detail)
        self.assertIn("install_shader_toolchain.sh", detail)



DETAILS = {
    "cargo build (ladder binaries)": "finished (debug)",
    "shader toolchain": "glslangValidator 16.5.0; glslc 2023.8; spirv-val 2025.5",
    "binding-access audit": "0 provably narrower binding(s) over 0 shader(s) out of 471 modules",
    "shader provenance": "464 variant(s) match their sources and flags",
    "push-constant layouts": "PASS: push-constant blocks agree",
    "oracle parity": "PASS: every shared shader matches the stable oracle",
    "golden dense (matmul)": "PASS: 144 invocation(s) exercised on this device match coverage",
    "golden reduction": "PASS: 183 invocation(s) exercised on this device match coverage",
    "golden attention": "PASS: 79 invocation(s) exercised on this device match coverage",
    "training-step probe": (
        "6512 dispatches, 3021 barriers (upper bound 3076), loss 5.5432177 on "
        + RECORDED_ADAPTER
    ),
    "topology decisions": (
        "PASS: 4 measured decision(s) match the record on '" + RECORDED_ADAPTER + "'"
    ),
    "training-state parity": (
        "PASS: 2 suite(s), 29 run(s), 32162 exact tensor digests compared"
    ),
}


def receipt():
    return [
        {"step": step, "status": "PASS", "detail": DETAILS[step]} for step in LADDER_STEPS
    ]


class CommittedReceiptTests(unittest.TestCase):
    """The committed receipt is a checked artifact, not a log."""

    def test_the_committed_receipt_describes_the_ladder(self):
        committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
        check_structure(committed)
        check_rows(committed)
        check_adapter(committed)
        check_results(committed)
        check_coverage(committed)

    def test_a_dropped_step_fails(self):
        with self.assertRaises(Problem):
            check_structure(receipt()[:-1])

    def test_a_renamed_or_extra_step_fails(self):
        renamed = receipt()
        renamed[3]["step"] = "shader provenance (skipped)"
        with self.assertRaises(Problem):
            check_structure(renamed)
        with self.assertRaises(Problem):
            check_structure(receipt() + [{"step": "extra", "status": "PASS", "detail": "x"}])

    def test_a_reordered_step_fails(self):
        reordered = receipt()
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaises(Problem):
            check_structure(reordered)

    def test_a_non_pass_row_fails(self):
        for status in ("FAIL", "BLOCKED"):
            rows = receipt()
            rows[9]["status"] = status
            with self.assertRaises(Problem):
                check_rows(rows)

    def test_shrinking_coverage_fails(self):
        rows = receipt()
        rows[7]["detail"] = rows[7]["detail"].replace("183", "182")
        with self.assertRaises(Problem):
            check_coverage(rows)

    def test_a_changed_result_fails_even_when_every_count_matches(self):
        rows = receipt()
        rows[9]["detail"] = rows[9]["detail"].replace("5.5432177", "5.5432178")
        with self.assertRaises(Problem):
            check_results(rows)

    def test_a_receipt_recorded_on_another_adapter_fails(self):
        rows = receipt()
        for row in rows:
            row["detail"] = row["detail"].replace(RECORDED_ADAPTER, "AMD Radeon Graphics")
        with self.assertRaises(Problem):
            check_adapter(rows)

    def test_device_aware_steps_that_disagree_fail(self):
        rows = receipt()
        rows[10]["detail"] = rows[10]["detail"].replace(RECORDED_ADAPTER, "AMD Radeon Graphics")
        with self.assertRaises(Problem):
            check_adapter(rows)

    def test_a_detail_that_no_longer_names_the_adapter_fails(self):
        rows = receipt()
        rows[9]["detail"] = "6512 dispatches, 3021 barriers, loss 5.5432177"
        with self.assertRaises(Problem):
            check_adapter(rows)

    def test_a_missing_count_fails_rather_than_passing_by_default(self):
        rows = receipt()
        rows[11]["detail"] = "PASS: trajectories match"
        with self.assertRaises(Problem):
            check_coverage(rows)

    def test_growing_coverage_passes_and_is_noted_on_compare(self):
        fresh = receipt()
        fresh[6]["detail"] = fresh[6]["detail"].replace("144", "150")
        check_coverage(fresh)
        committed = receipt()
        notes = check_compare(fresh, committed)
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("re-record", notes[0])

    def test_a_fresh_run_below_the_committed_counts_fails(self):
        committed = receipt()
        fresh = receipt()
        fresh[11]["detail"] = fresh[11]["detail"].replace("32162", "32161")
        with self.assertRaises(Problem):
            check_compare(fresh, committed)

    def test_floors_cover_every_step_that_reports_a_count(self):
        # A new reporting step added to the ladder without a floor here would
        # pass this guard while covering nothing, so the spec is checked too.
        for step in COVERAGE:
            self.assertIn(step, LADDER_STEPS, step)
        committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
        floors = committed_floors(committed)
        self.assertEqual(set(floors), set(COVERAGE))
        for step, patterns in floors.items():
            self.assertEqual(len(patterns), len(COVERAGE[step]), step)
            for (_, floor), (_, expected) in zip(patterns, COVERAGE[step]):
                self.assertGreaterEqual(floor, expected, step)


if __name__ == "__main__":
    unittest.main()
