import unittest

from rdna3_performance_gate import evaluate


def report(*cis, exact=True):
    cases = ["native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5"]
    results = [
        {
            "case": case,
            "role": "native" if case.startswith("native") else "control",
            "speedup": (low + high) / 2.0,
            "paired_median_speedup": (low + high) / 2.0,
            "speedup_ci95": [low, high],
        }
        for case, (low, high) in zip(cases, cis)
    ]
    return {
        "schema": 1,
        "cross_arm_word_exact": exact,
        "results": results,
        "gate_failures": [],
    }


GOOD = [(0.977, 1.038), (1.016, 1.045), (0.99, 1.02), (0.98, 1.01), (0.99, 1.03)]


class GateTests(unittest.TestCase):
    def test_inconclusive_is_not_a_regression(self):
        gate = evaluate(report(*GOOD))
        self.assertTrue(gate["pass"])
        self.assertIn("native_vit_4x4", gate["inconclusive"])
        self.assertNotIn("native_vit_6x4", gate["inconclusive"])

    def test_clear_improvement_is_reported(self):
        gate = evaluate(report(*GOOD))
        self.assertIn("native_vit_6x4", gate["native_improvements"])

    def test_clear_regression_fails(self):
        bad = list(GOOD)
        bad[3] = (0.90, 0.98)
        gate = evaluate(report(*bad))
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["significant_regressions"], ["gpt2"])

    def test_release_policy_requires_native_improvement(self):
        gate = evaluate(report(*GOOD), require_native_improvement=True)
        self.assertFalse(gate["pass"])
        self.assertTrue(any("native_vit_4x4" in failure for failure in gate["gate_failures"]))

    def test_cross_arm_word_difference_fails(self):
        with self.assertRaises(ValueError):
            evaluate(report(*GOOD, exact=False))


if __name__ == "__main__":
    unittest.main()
