"""Negative controls for the stack ladder's probe expectation (no GPU needed).

The expectation is device-keyed: the vendor layer selects different kernels on
different parts, so the same fixture does not dispatch the same number of times
everywhere. These tests pin that down -- a run is compared against its own
device's entry, another device's numbers are never substituted for it, and a
device with no entry fails instead of passing by default.
"""
import copy
import tempfile
import unittest
from pathlib import Path

from verify_training_stack import (
    PROBE_ARGS,
    SHADER_TOOLS,
    ensure_shader_toolchain,
    expectation_problems,
)

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


if __name__ == "__main__":
    unittest.main()
