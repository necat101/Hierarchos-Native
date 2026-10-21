"""Negative controls for direct, all-position stable logit comparison."""
import copy
import unittest

from check_stable_logits import compare, comparisons_pass


def report():
    return {"device": "test GPU", "architecture": "test", "batch_size": 1,
            "seq_len": 1, "vocab_size": 2, "logits": [0.0, 0.25]}


class StableLogitTests(unittest.TestCase):
    def test_hf_failure_cannot_be_hidden_by_stable_bit_identity(self):
        self.assertFalse(comparisons_pass([{"pass": True, "hf_failing_values": 1}]))
        self.assertFalse(comparisons_pass([{"pass": False, "hf_failing_values": 0}]))
        self.assertFalse(comparisons_pass([]))
        self.assertTrue(comparisons_pass([{"pass": True, "hf_failing_values": 0}]))

    def test_identical(self):
        self.assertTrue(compare(report(), report())["pass"])

    def test_reported_error_uses_the_fp32_values_under_comparison(self):
        candidate = report()
        candidate["logits"][1] = 0.2500000001
        result = compare(report(), candidate)
        self.assertTrue(result["pass"])
        self.assertEqual(result["max_abs"], 0.0)

    def test_single_ulp_is_rejected(self):
        candidate = report()
        candidate["logits"][1] = 0.2500000298023224
        self.assertEqual(compare(report(), candidate)["different_bits"], 1)

    def test_signed_zero_is_rejected(self):
        candidate = report()
        candidate["logits"][0] = -0.0
        self.assertFalse(compare(report(), candidate)["pass"])

    def test_nonfinite_is_rejected_on_either_side(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            broken = report()
            broken["logits"][0] = value
            for left, right in ((broken, report()), (report(), broken)):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    compare(left, right)

    def test_missing_empty_and_short_are_rejected(self):
        for value in (None, [], [0.0]):
            broken = report()
            broken["logits"] = value
            with self.assertRaises(ValueError):
                compare(report(), broken)

    def test_missing_metadata_or_different_device_fails(self):
        for key in ("device", "architecture", "batch_size", "seq_len", "vocab_size"):
            broken = report()
            del broken[key]
            with self.assertRaises(ValueError):
                compare(broken, report())
        broken = report()
        broken["device"] = "another GPU"
        with self.assertRaises(ValueError):
            compare(report(), broken)

    def test_truncated_reports_cannot_agree(self):
        broken = copy.deepcopy(report())
        broken["logits"].pop()
        with self.assertRaisesRegex(ValueError, "shape product"):
            compare(broken, broken)


if __name__ == "__main__":
    unittest.main()
