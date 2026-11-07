import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rdna3_performance_gate import (
    CASES,
    InvalidReceipt,
    evaluate_reports,
    MIN_EQUIVALENCE_MARGIN,
)

ROUNDS = 6
STEPS = 4
SHA_A = "a" * 64
SHA_B = "b" * 64


def report(
    cis=None,
    *,
    batch=1,
    rounds=ROUNDS,
    steps=STEPS,
    exact=True,
    candidate_sha=SHA_A,
    fixtures="/frozen/gen9",
    noise=0.002,
    signature="c" * 64,
):
    cis = cis or [
        (1.02, 1.05),
        (1.02, 1.05),
        (0.999, 1.001),
        (0.999, 1.001),
        (0.999, 1.001),
    ]
    results = []
    for case, (low, high) in zip(CASES, cis):
        median = (low + high) / 2.0
        results.append(
            {
                "case": case,
                "role": "native" if case.startswith("native") else "control",
                "speedup": median,
                "paired_median_speedup": median,
                "speedup_ci95": [low, high],
                "paired_round_speedups": [median] * rounds,
                "baseline": {
                    "round_medians_ms": [100.0] * rounds,
                    "samples": rounds * steps,
                    "relative_mad": noise,
                    "sample_cv": noise,
                    "signature_sha256": signature,
                },
                "candidate": {
                    "round_medians_ms": [100.0 / median] * rounds,
                    "samples": rounds * steps,
                    "relative_mad": noise,
                    "sample_cv": noise,
                    "signature_sha256": signature,
                },
            }
        )
    return {
        "schema": 1,
        "cross_arm_word_exact": exact,
        "batch": batch,
        "rounds": rounds,
        "measured_steps": steps,
        "warmup_steps": 2,
        "fixtures": fixtures,
        "baseline": {"binary": "baseline.exe", "sha256": SHA_B, "env": {}},
        "candidate": {"binary": "candidate.exe", "sha256": candidate_sha, "env": {}},
        "results": results,
        "gate_failures": [],
    }


def gate(report_obj, mode="development", **kwargs):
    return evaluate_reports([("test", report_obj)], mode=mode, **kwargs)


class GateTests(unittest.TestCase):
    def test_clear_improvement_passes(self):
        result = gate(report())
        self.assertEqual(result["state"], "PASS")
        self.assertTrue(result["pass"])

    def test_practical_equivalence_passes_controls_in_release(self):
        cis = [
            (1.02, 1.05),
            (1.02, 1.05),
            (0.999, 1.001),
            (0.999, 1.001),
            (0.999, 1.001),
        ]
        result = gate(report(cis), mode="release", required_batches=(1,))
        self.assertEqual(result["state"], "PASS")
        self.assertTrue(result["release_qualified"])

    def test_clear_regression_fails(self):
        cis = [
            (0.94, 0.98),
            (1.02, 1.05),
            (0.999, 1.001),
            (0.999, 1.001),
            (0.999, 1.001),
        ]
        result = gate(report(cis))
        self.assertEqual(result["state"], "REGRESSION")
        self.assertFalse(result["pass"])

    def test_noisy_slowdown_is_inconclusive(self):
        cis = [
            (0.97, 1.03),
            (0.97, 1.03),
            (0.999, 1.001),
            (0.999, 1.001),
            (0.999, 1.001),
        ]
        result = gate(report(cis))
        self.assertEqual(result["state"], "INCONCLUSIVE")
        self.assertFalse(result["pass"])

    def test_missing_samples_is_invalid(self):
        bad = report()
        bad["results"][2]["baseline"]["samples"] = 0
        with self.assertRaises(InvalidReceipt):
            gate(bad)

    def test_mismatched_binary_is_invalid(self):
        with self.assertRaises(InvalidReceipt):
            evaluate_reports(
                [("b1", report(batch=1)), ("b2", report(batch=2, candidate_sha=SHA_B))],
                mode="development",
            )

    def test_mismatched_configuration_is_invalid(self):
        with self.assertRaises(InvalidReceipt):
            evaluate_reports(
                [("b1", report(batch=1)), ("b2", report(batch=2, steps=7))],
                mode="development",
            )

    def test_corrupted_json_file_is_invalid(self):
        gate_script = Path(__file__).with_name("rdna3_performance_gate.py")
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "benchmark.json"
            bad.write_text("{not json", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(gate_script), str(bad)],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("state: **INVALID**", completed.stdout)

    def test_equality_at_equivalence_boundary_is_accepted(self):
        margin = MIN_EQUIVALENCE_MARGIN
        cis = [
            (1.02, 1.05),
            (1.02, 1.05),
            (1.0 - margin, 1.0 + margin),
            (1.0 - margin, 1.0 + margin),
            (1.0 - margin, 1.0 + margin),
        ]
        result = gate(report(cis, noise=0.0), mode="release", required_batches=(1,))
        self.assertEqual(result["state"], "PASS")
        # Native acceptance is strict at 1.0: equality is not improvement.
        cis[0] = (1.0, 1.02)
        result = gate(report(cis, noise=0.0), mode="release", required_batches=(1,))
        self.assertEqual(result["state"], "INCONCLUSIVE")

    def test_incomplete_release_matrix_is_invalid(self):
        with self.assertRaises(InvalidReceipt):
            gate(report(batch=1), mode="release")
        with self.assertRaises(InvalidReceipt):
            evaluate_reports(
                [("b1", report(batch=1)), ("b2", report(batch=2))],
                mode="release",
                required_batches=(1, 2, 4),
            )


if __name__ == "__main__":
    unittest.main()
