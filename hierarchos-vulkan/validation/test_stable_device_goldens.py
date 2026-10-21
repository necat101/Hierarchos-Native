import unittest

from establish_stable_device_goldens import (
    missing_device, require_environment, require_reference, ORACLE_COMMIT, ORACLE_SOURCE_SHA256,
)


class StableDeviceGoldenTests(unittest.TestCase):
    def test_execution_overrides_cannot_change_the_reference_policy(self):
        require_environment({"HIERARCHOS_STABLE_ORACLE": "oracle", "CARGO_BUILD_JOBS": "1"})
        for key in ("HIERARCHOS_VULKAN_FORCE_VENDOR", "HIERARCHOS_VULKAN_TRAINING_PRECISION",
                    "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unset execution overrides"):
                require_environment({key: "override"})

    def test_dirty_wrong_commit_or_modified_sources_are_rejected(self):
        reference = {"commit": ORACLE_COMMIT, "status": "", "source_sha256": ORACLE_SOURCE_SHA256}
        require_reference(reference)
        for key, value in (("commit", "candidate"), ("status", " M shader.spv"),
                           ("source_sha256", "changed")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact clean Windows oracle"):
                require_reference({**reference, key: value})

    def test_only_wholly_missing_device_entries_can_be_added(self):
        document = {"cases": [{"devices": {"Intel": [1]}}, {"devices": {"Intel": [2]}}]}
        self.assertTrue(missing_device(document, "AMD"))
        self.assertFalse(missing_device(document, "Intel"))
        document["cases"][0]["devices"]["AMD"] = [3]
        with self.assertRaisesRegex(ValueError, "partial device golden"):
            missing_device(document, "AMD")
