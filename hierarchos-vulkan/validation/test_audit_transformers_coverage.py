import os
import unittest
from pathlib import Path

from audit_transformers_coverage import _mapping_model_types, build_report


class RegistryAuditTests(unittest.TestCase):
    def test_comments_multiline_keys_and_mapping_expansion(self):
        source = """
A = OrderedDict([
    # An entry may span lines and use either quote style.
    (
        'text', SomeModel,
    ),
])
B = OrderedDict([
    *list(A.items()),
    ("vision", (ModelOne, ModelTwo)),
])
"""
        self.assertEqual(_mapping_model_types(source, "B"), {"text", "vision"})

    def test_empty_mapping(self):
        self.assertEqual(_mapping_model_types("A = OrderedDict()", "A"), set())

    def test_cyclic_mapping_fails_without_evaluating_python(self):
        with self.assertRaisesRegex(RuntimeError, "cyclic"):
            _mapping_model_types("A = OrderedDict([*list(A.items())])", "A")

    def test_adjacent_checkout_includes_non_language_tasks_without_claiming_support(self):
        repo_root = Path(__file__).resolve().parents[2]
        configured = os.environ.get("TRANSFORMERS_CHECKOUT")
        transformers_root = Path(configured) if configured else repo_root
        modeling_auto = transformers_root / "src/transformers/models/auto/modeling_auto.py"
        if not modeling_auto.is_file():
            self.skipTest(
                "Transformers source checkout unavailable; set TRANSFORMERS_CHECKOUT to run registry integration coverage"
            )
        report = build_report(repo_root, transformers_root)
        additional = report["additional_task_registry_overlap"]
        for task in ("sequence_classification", "image_classification", "speech_seq_2_seq", "multimodal_lm"):
            self.assertIn(task, additional)
            self.assertFalse(additional[task]["task_implementation_verified"])
            self.assertGreater(additional[task]["total"], 0)


if __name__ == "__main__":
    unittest.main()
