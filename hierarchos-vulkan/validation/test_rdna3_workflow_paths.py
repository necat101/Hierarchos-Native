"""Regression checks for the RDNA3 workflow's isolated Windows paths.

The static checks run on every platform.  The runtime check only runs on
Windows, where Git Bash supplies ``cygpath``; it creates no Cargo target and
does not require a GPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "rdna3-performance.yml"


def git_bash() -> str | None:
    """Prefer the bundled Git Bash, not a WSL or unrelated bash wrapper."""
    candidates: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "Git" / "bin" / "bash.exe")
    discovered = shutil.which("bash")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file() and "system32" not in str(candidate).lower():
            return str(candidate)
    return None


class WorkflowPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_uses_cygpath_and_checks_the_converted_directory(self) -> None:
        self.assertNotIn("PureWindowsPath", self.text)
        self.assertIn('command -v cygpath', self.text)
        self.assertIn(
            'RUNNER_TEMP_POSIX="$(cygpath -u "$RUNNER_TEMP")"', self.text
        )
        self.assertIn('test -d "$RUNNER_TEMP_POSIX"', self.text)

    def test_run_root_and_build_log_root_are_initialized_before_use(self) -> None:
        init = self.text.index('RUN_ROOT="$RUNNER_TEMP_POSIX/rdna3-v41-${GITHUB_RUN_ID}"')
        export = self.text.index('"$RUN_ROOT/build-logs"', init)
        candidate = self.text.index(
            '"$BUILD_LOG_ROOT/candidate-build.json"', export
        )
        baseline = self.text.index(
            '"$BUILD_LOG_ROOT/baseline-build.json"', export
        )
        self.assertLess(init, export)
        self.assertLess(export, candidate)
        self.assertLess(export, baseline)

    def test_candidate_and_baseline_logs_have_distinct_names(self) -> None:
        self.assertIn(
            '> "$BUILD_LOG_ROOT/candidate-build.json"', self.text
        )
        self.assertIn(
            '> "$BUILD_LOG_ROOT/baseline-build.json"', self.text
        )
        self.assertNotIn("$RUNNER_TEMP/candidate-build.json", self.text)
        self.assertNotIn("$RUNNER_TEMP/baseline-build.json", self.text)

    def test_full_history_and_nonpersistent_checkout_are_preserved(self) -> None:
        self.assertIn("fetch-depth: 0", self.text)
        self.assertIn("persist-credentials: false", self.text)

    @unittest.skipUnless(
        sys.platform == "win32", "cygpath and MSYS path interpretation are Windows-only"
    )
    def test_cygpath_build_log_paths_are_usable(self) -> None:
        cygpath = shutil.which("cygpath")
        bash = git_bash()
        if cygpath is None or bash is None:
            self.skipTest("Git Bash and cygpath are required")

        # Use a Windows path under the workspace, like RUNNER_TEMP.  Python's
        # default temp directory is exported by Git Bash as /tmp, which is not
        # the C:\... path the workflow starts with.
        smoke_parent = ROOT / "target" / "rdna3-path-smoke"
        smoke_parent.mkdir(parents=True, exist_ok=True)
        raw = tempfile.mkdtemp(prefix="hierarchos-path-smoke-", dir=smoke_parent)
        try:
            converted = subprocess.run(
                [cygpath, "-u", raw],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertTrue(converted.startswith("/"), converted)
            self.assertNotEqual(converted, raw)
            windows = subprocess.run(
                [cygpath, "-w", converted],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(Path(windows).resolve(), Path(raw).resolve())
            self.assertTrue(os.path.isdir(windows), windows)

            run_root = Path(windows) / f"rdna3-v41-{os.getpid()}"
            build_logs = run_root / "build-logs"
            build_logs.mkdir(parents=True)
            self.assertTrue(build_logs.is_dir())
            candidate_log = build_logs / "candidate-build.json"
            baseline_log = build_logs / "baseline-build.json"
            self.assertNotEqual(candidate_log, baseline_log)
            self.assertEqual(candidate_log.parent, baseline_log.parent)

            executable = run_root / "candidate-test.exe"
            executable.write_bytes(b"")
            candidate_log.write_text(
                json.dumps(
                    {
                        "reason": "compiler-artifact",
                        "target": {"name": "hierarchos_vulkan"},
                        "executable": str(executable),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            # Git Bash converts an MSYS path argument before launching native
            # Python, which is how the workflow's selector receives the log.
            candidate_posix = (
                f"{converted}/rdna3-v41-{os.getpid()}/build-logs/{candidate_log.name}"
            )
            script = (
                "import pathlib,sys; from cargo_test_binary import select; "
                "print(select(pathlib.Path(sys.argv[1]), 'hierarchos_vulkan').resolve())"
            )
            python_posix = subprocess.run(
                [cygpath, "-u", sys.executable],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            probe = subprocess.run(
                [
                    bash,
                    "--noprofile",
                    "--norc",
                    "-c",
                    f'{shlex.quote(python_posix)} -c {shlex.quote(script)} "$1"',
                    "bash",
                    candidate_posix,
                ],
                cwd=ROOT / "hierarchos-vulkan" / "validation",
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                probe.returncode,
                0,
                f"bash={bash!r} python={python_posix!r} "
                f"stdout={probe.stdout!r} stderr={probe.stderr!r}",
            )
            selected = probe.stdout.strip()
            self.assertEqual(Path(selected).resolve(), executable.resolve())
            baseline_log.write_text("{}\n", encoding="utf-8")
            self.assertTrue(baseline_log.is_file())
        finally:
            shutil.rmtree(raw, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
