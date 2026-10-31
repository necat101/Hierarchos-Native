import math
import unittest

import torch

from deepseek_v41_parity import LOGIT_DRIFT_LIMIT, require_logit_parity


class DeepseekV41StrictLogitGateTests(unittest.TestCase):
    def test_strict_boundary_and_no_relative_tolerance(self):
        zero = torch.tensor([0.0], dtype=torch.float64)
        below = math.nextafter(LOGIT_DRIFT_LIMIT, 0.0)
        self.assertEqual(require_logit_parity(torch.tensor([below], dtype=torch.float64), zero, label="below"), below)
        for value in [LOGIT_DRIFT_LIMIT, math.nextafter(LOGIT_DRIFT_LIMIT, math.inf), 1e-6]:
            with self.assertRaises(AssertionError):
                require_logit_parity(torch.tensor([value], dtype=torch.float64), zero, label="boundary")
        with self.assertRaises(AssertionError):
            require_logit_parity(torch.tensor([1e6 + 1]), torch.tensor([1e6]), label="large logits")

    def test_all_elements_and_invalid_data(self):
        values = torch.zeros(2, 3, 8)
        self.assertEqual(require_logit_parity(values, values, label="identical"), 0.0)
        changed = values.clone()
        changed[-1, -1, -1] = 1e-6
        with self.assertRaises(AssertionError):
            require_logit_parity(changed, values, label="last element")
        for invalid in [torch.tensor([math.nan]), torch.tensor([math.inf]), torch.tensor([-math.inf])]:
            with self.assertRaises(AssertionError):
                require_logit_parity(invalid, invalid, label="nonfinite")
        for native, reference in [(torch.empty(0), torch.empty(0)), (torch.zeros(1), torch.zeros(1, 1)),
                                  (torch.tensor([1]), torch.tensor([1]))]:
            with self.assertRaises(AssertionError):
                require_logit_parity(native, reference, label="shape/dtype")


if __name__ == "__main__":
    unittest.main()
