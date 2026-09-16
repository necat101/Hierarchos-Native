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
        report = build_report(Path(__file__).resolve().parents[2])
        additional = report["additional_task_registry_overlap"]
        for task in ("sequence_classification", "image_classification", "speech_seq_2_seq", "multimodal_lm"):
            self.assertIn(task, additional)
            self.assertFalse(additional[task]["task_implementation_verified"])
            self.assertGreater(additional[task]["total"], 0)


if __name__ == "__main__":
    unittest.main()
