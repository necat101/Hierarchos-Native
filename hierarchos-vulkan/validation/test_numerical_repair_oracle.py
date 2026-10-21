import unittest
import tempfile
from pathlib import Path

from numerical_repair_oracle import REPAIRED_FAMILIES, replace_once, repair_source, source_digest, verify_prepared


class NumericalRepairOracleTests(unittest.TestCase):
    def test_exception_is_limited_to_gpt2_registry_cases(self):
        self.assertEqual(REPAIRED_FAMILIES, {"gpt2", "gpt2_wide"})

    def test_missing_or_duplicated_reference_seam_is_rejected(self):
        for source in ("absent", "seam seam"):
            with self.assertRaisesRegex(ValueError, "seam changed"):
                replace_once(source, "seam", "repair")
        self.assertEqual(replace_once("one seam only", "seam", "repair"), "one repair only")
        with self.assertRaisesRegex(ValueError, "seam changed"):
            repair_source("a different reference implementation")

    def test_reference_content_and_unexpected_files_are_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crate = root / "hierarchos-vulkan"
            crate.mkdir()
            path = crate / "source.rs"
            path.write_bytes(b"approved source")
            metadata = {"copied_source_sha256": source_digest({"hierarchos-vulkan/source.rs": path.read_bytes()})}
            verify_prepared(crate, metadata)
            path.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "sources changed"):
                verify_prepared(crate, metadata)
            path.write_bytes(b"approved source")
            (root / "unexpected.toml").write_bytes(b"unapproved build settings")
            with self.assertRaisesRegex(ValueError, "sources changed"):
                verify_prepared(crate, metadata)
