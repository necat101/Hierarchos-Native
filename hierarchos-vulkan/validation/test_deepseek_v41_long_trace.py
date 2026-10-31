"""Diagnostic regressions; synthetic traces do not qualify model logits."""
from copy import deepcopy
import unittest

from validate_deepseek_v41_long_context_model import compare_traced


class DeepseekV41SharedLongTraceTests(unittest.TestCase):
    def inputs(self):
        config = dict(text_config=dict(index_source_layer_ids=[0], compress_ratios=[1, 1, 1]))
        spec = dict(pos=1, seq=1)
        stages = [dict(layer=0, stage="moe", values=[1.0])]
        cache = {i: dict(kv=[0x3f800000], keys=[0x3f000000]) for i in range(3)}
        reference = dict(stages=stages, index_offset=2, cache_diagnostics=cache,
                         index_diagnostics={0: dict(indices=[2, 3], score_bits=[0x3f800000, 0x40000000])})
        native = dict(stages=deepcopy(stages), index_offset=128,
                      cache_diagnostics={str(k): deepcopy(v) for k, v in cache.items()},
                      index_diagnostics={str(i): dict(indices=[128, 129],
                          score_bits=[0x3f800000, 0x40000000]) for i in range(3)})
        return config, spec, native, reference

    def test_counts_include_every_shared_consumer_and_keep_owner_totals(self):
        report = compare_traced(*self.inputs())
        self.assertEqual(report["bit_exact_index_scores_including_shared_consumers"], 6)
        self.assertEqual(report["bit_exact_indices_including_shared_consumers"], 6)
        self.assertEqual(report["bit_exact_owner_index_scores"], 2)
        self.assertEqual(report["bit_exact_owner_selected_indices"], 2)
        self.assertEqual(report["bit_exact_cache_values"], 6)
        self.assertEqual(report["trace_values"], report["bit_exact_values"])

    def test_corrupted_shared_scores_cannot_hide_behind_an_exact_owner(self):
        config, spec, native, reference = self.inputs()
        native["index_diagnostics"]["2"]["score_bits"][0] ^= 1
        with self.assertRaisesRegex(AssertionError, "index scores differ"):
            compare_traced(config, spec, native, reference)

    def test_corrupted_shared_positions_are_rejected_after_offset_normalization(self):
        config, spec, native, reference = self.inputs()
        native["index_diagnostics"]["2"]["indices"][1] += 1
        with self.assertRaisesRegex(AssertionError, "selected compressed positions differ"):
            compare_traced(config, spec, native, reference)

    def test_missing_shared_consumer_is_rejected(self):
        config, spec, native, reference = self.inputs()
        del native["index_diagnostics"]["1"]
        with self.assertRaisesRegex(AssertionError, "complete long-context index diagnostics required"):
            compare_traced(config, spec, native, reference)


if __name__ == "__main__":
    unittest.main()
