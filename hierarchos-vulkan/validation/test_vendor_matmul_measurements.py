import unittest

from vendor_matmul_measurements import measurements, validates_fallback


class AmdMatmulEvidenceTests(unittest.TestCase):
    def evidence(self, route):
        row = next(row for row in measurements() if row["route"] == route)
        evidence = {key: row[key] for key in
                    ("device", "family", "rows", "input_dim", "output_dim", "parameter_storage")}
        return dict(evidence, reason="measured-portable")

    def test_measured_loser_is_an_explicit_fallback(self):
        evidence = self.evidence("portable")
        self.assertTrue(validates_fallback(evidence, evidence["family"]))

    def test_winner_and_unreported_dispatch_cannot_be_waived(self):
        evidence = self.evidence("amd")
        self.assertFalse(validates_fallback(evidence, evidence["family"]))
        self.assertFalse(validates_fallback({}, "linear-forward"))
        evidence = self.evidence("portable")
        self.assertFalse(validates_fallback(evidence, "different-family"))

    def test_measured_claim_cannot_be_inherited_by_another_driver(self):
        evidence = self.evidence("portable")
        evidence["device"] = evidence["device"].replace("00800184", "00800185")
        self.assertFalse(validates_fallback(evidence, evidence["family"]))
        evidence["reason"] = "unmeasured-portable"
        self.assertTrue(validates_fallback(evidence, evidence["family"]))

    def test_known_winner_cannot_masquerade_as_unmeasured(self):
        evidence = self.evidence("amd")
        evidence["reason"] = "unmeasured-portable"
        self.assertFalse(validates_fallback(evidence, evidence["family"]))
