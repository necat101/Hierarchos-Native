"""Negative controls for the Phase XIX-W architecture compatibility matrix."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from phase19w_matrix import DEFAULT_OUTPUT, validate
from phase19w_registry import parse_registry


class CompatibilityNegativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = parse_registry()
        cls.matrix = json.loads(Path(DEFAULT_OUTPUT).read_text(encoding="utf-8"))

    def rejected(self, change):
        matrix = copy.deepcopy(self.matrix)
        change(matrix)
        with self.assertRaises(ValueError):
            validate(matrix, self.registry)

    def row(self, matrix, model_type):
        return next(row for row in matrix["records"] if row["model_type"] == model_type)

    def test_positive_matrix_validates(self):
        validate(self.matrix, self.registry)

    def test_missing_canonical_is_rejected(self):
        def change(matrix):
            matrix["records"] = [row for row in matrix["records"] if row["model_type"] != "llama"]
        self.rejected(change)

    def test_missing_alias_is_rejected(self):
        def change(matrix):
            matrix["records"] = [row for row in matrix["records"] if row["model_type"] != "llava"]
        self.rejected(change)

    def test_duplicate_row_is_rejected(self):
        def change(matrix):
            matrix["records"][-1] = copy.deepcopy(matrix["records"][0])
        self.rejected(change)

    def test_missing_historical_architecture_is_rejected(self):
        def change(matrix):
            self.row(matrix, matrix["historical_native_architectures"][0])["overall_status"] = "not_tested"
        self.rejected(change)

    def test_forged_historical_pass_is_rejected(self):
        def change(matrix):
            matrix["forward_records"][0]["max_abs"] = 2e-7
        self.rejected(change)

    def test_wrong_device_is_rejected(self):
        def change(matrix):
            matrix["forward_records"][0]["device_id"] = 0x1916
        self.rejected(change)

    def test_wrong_policy_is_rejected(self):
        def change(matrix):
            matrix["policy"]["absolute_limit"] = 2.1e-7
        self.rejected(change)

    def test_reduced_generation_matrix_is_rejected(self):
        def change(matrix):
            matrix["summary"]["generation_requests"] = 71
        self.rejected(change)

    def test_alias_to_wrong_graph_is_rejected(self):
        def change(matrix):
            matrix["alias_targets"]["llava"] = "gpt2"
        self.rejected(change)

    def test_forged_alias_numerical_pass_is_rejected(self):
        def change(matrix):
            self.row(matrix, "llava")["forward_result"] = {"pass": True, "runs": 2}
        self.rejected(change)

    def test_missing_evidence_is_rejected(self):
        def change(matrix):
            matrix["evidence"][0]["sha256"] = "0" * 64
        self.rejected(change)

    def test_missing_mode_is_rejected(self):
        def change(matrix):
            matrix["forward_records"] = [
                row for row in matrix["forward_records"] if row["mode"] != "streaming"
            ]
        self.rejected(change)

    def test_false_untested_forward_claim_is_rejected(self):
        def change(matrix):
            self.row(matrix, "llama")["forward_result"] = {"pass": True, "runs": 2}
        self.rejected(change)

    def test_unknown_status_is_rejected(self):
        def change(matrix):
            self.row(matrix, "llama")["overall_status"] = "green"
        self.rejected(change)


if __name__ == "__main__":
    unittest.main()
