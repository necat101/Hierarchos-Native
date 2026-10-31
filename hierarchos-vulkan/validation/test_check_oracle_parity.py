import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from check_oracle_parity import (
    EXIT_DRIFT,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    compare,
    evaluate,
    main,
    read_divergences,
    shader_dir,
)


def seed(root, files):
    """Write `files` into `<root>/hierarchos-vulkan/shaders`."""
    shaders = root / "hierarchos-vulkan" / "shaders"
    shaders.mkdir(parents=True)
    for name, data in files.items():
        path = shaders / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Bytes on purpose: text mode would translate `\n` on Windows and make
        # the line-ending test write `\r\r\n` instead of `\r\n`.
        path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return root


class TreeComparisonTests(unittest.TestCase):
    def compare_trees(self, oracle_files, local_files, divergences=None):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            oracle = seed(scratch / "oracle", oracle_files)
            local = seed(scratch / "local", local_files)
            divergences_path = scratch / "divergences.json"
            if divergences is not None:
                divergences_path.write_text(
                    json.dumps({"divergences": divergences}), encoding="utf-8"
                )
            report = compare(shader_dir(local), shader_dir(oracle))
            return report, evaluate(report, read_divergences(divergences_path))

    def test_identical_trees_have_no_problems(self):
        files = {"linear_forward.comp": "void main() {}\n", "linear_forward.spv": b"\x03\x02\x23\x07"}
        _, problems = self.compare_trees(files, files)
        self.assertEqual(problems, [])

    def test_a_changed_artifact_is_drift(self):
        # The regression that shipped once: a regenerated portable module.
        _, problems = self.compare_trees(
            {"parameter_matmul_weight_grad.spv": b"oracle-bytes"},
            {"parameter_matmul_weight_grad.spv": b"regenerated-bytes"},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("parameter_matmul_weight_grad.spv", problems[0])

    def test_line_endings_are_not_drift(self):
        _, problems = self.compare_trees(
            {"linear_forward.comp": "#version 450\nvoid main() {}\n"},
            {"linear_forward.comp": "#version 450\r\nvoid main() {}\r\n"},
        )
        self.assertEqual(problems, [])

    def test_a_documented_divergence_is_accepted(self):
        _, problems = self.compare_trees(
            {"demo.spv": b"oracle"},
            {"demo.spv": b"deliberate"},
            divergences=[{"path": "demo.spv", "reason": "tuned"}],
        )
        self.assertEqual(problems, [])

    def test_a_stale_divergence_fails(self):
        files = {"demo.spv": b"same"}
        _, problems = self.compare_trees(files, files, divergences=[{"path": "demo.spv", "reason": "tuned"}])
        self.assertEqual(len(problems), 1)
        self.assertIn("stale", problems[0])

    def test_an_oracle_file_missing_here_is_drift(self):
        _, problems = self.compare_trees({"gone.spv": b"x"}, {})
        self.assertEqual(len(problems), 1)
        self.assertIn("gone.spv", problems[0])

    def test_vendor_modules_are_excluded(self):
        _, problems = self.compare_trees(
            {
                "linear_forward_intel_gen9.comp": "old tuning\n",
                "linear_forward_intel_gen9.spv": b"old",
                "vendor_experiments/silu.comp": "old\n",
            },
            {
                "linear_forward_intel_gen9.comp": "new tuning\n",
                "linear_forward_intel_gen9.spv": b"new",
                "vendor_experiments/silu.comp": "new\n",
            },
        )
        self.assertEqual(problems, [])

    def test_only_here_is_not_drift(self):
        report, problems = self.compare_trees({}, {"added.spv": b"x"})
        self.assertEqual(problems, [])
        self.assertEqual(report["added"], ["added.spv"])


class ShaderDirTests(unittest.TestCase):
    def test_accepts_crate_root_and_repository_root(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            crate = root / "hierarchos-vulkan"
            (crate / "shaders").mkdir(parents=True)
            self.assertEqual(shader_dir(crate), crate / "shaders")
            self.assertEqual(shader_dir(root), crate / "shaders")

    def test_missing_oracle_is_a_skip(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "shaders").mkdir()
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main(["--local", str(root), "--oracle", str(root / "absent")])
            self.assertEqual(code, EXIT_UNAVAILABLE)
            self.assertIn("SKIP", err.getvalue())

    def test_exit_codes_for_drift_and_parity(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            oracle = seed(scratch / "oracle", {"a.spv": b"1"})
            local = seed(scratch / "local", {"a.spv": b"2"})
            divergences = scratch / "none.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([
                        "--local", str(local),
                        "--oracle", str(oracle),
                        "--divergences", str(divergences),
                    ]),
                    EXIT_DRIFT,
                )
            (local / "hierarchos-vulkan" / "shaders" / "a.spv").write_bytes(b"1")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([
                        "--local", str(local),
                        "--oracle", str(oracle),
                        "--divergences", str(divergences),
                    ]),
                    EXIT_OK,
                )


if __name__ == "__main__":
    unittest.main()
