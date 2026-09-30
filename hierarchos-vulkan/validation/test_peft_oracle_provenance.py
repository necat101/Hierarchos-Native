"""Oracle identity regressions; no model creation or Vulkan dispatch required."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The legacy CLI harness selects its family at import time from argv.
with patch.object(sys, "argv", ["verify_peft_lora_strict.py", "gpt2"]):
    import verify_peft_lora_strict as strict


class LocalTransformersProvenanceTests(unittest.TestCase):
    def test_accepts_requested_checkout_and_records_import(self):
        source = strict.TRANSFORMERS_ROOT / "src/transformers/__init__.py"
        oracle = SimpleNamespace(__file__=str(source), __version__="test-local")
        with patch.dict(sys.modules, transformers=oracle):
            evidence = strict.local_transformers_provenance()
        self.assertEqual(evidence["transformers_source"], str(source.resolve()))
        self.assertEqual(evidence["transformers_version"], "test-local")

    def test_rejects_site_packages_and_similarly_named_checkout(self):
        for source in [Path("site-packages/transformers/__init__.py"),
                       strict.TRANSFORMERS_ROOT / "src/transformers-copy/__init__.py"]:
            with self.subTest(source=source):
                oracle = SimpleNamespace(__file__=str(source), __version__="test-wheel")
                with patch.dict(sys.modules, transformers=oracle):
                    with self.assertRaisesRegex(RuntimeError, "must load from"):
                        strict.local_transformers_provenance()


if __name__ == "__main__":
    unittest.main()
