"""Allocation balance must be complete, numeric and traceable to raw readings."""
import copy
import unittest
from packed_allocation_timing import merge_allocation_runs, TENSORS
import check_measured_producer_routes as guard


class AllocationTimingTest(unittest.TestCase):
    def parts(self):
        parts = []
        for rotation in range(3):
            run = dict(device='test adapter', width=128, head_size=64, batch=1, steps=8,
                       chain='remat', parameter_storage='fp32', precision_policy='fp32',
                       default_route='packed', recorded_route='packed', allocation_rotation=rotation,
                       samples=400, warmup=2, worst_max_abs=0.0,
                       tensors=[dict(label=label, values=16, max_abs=0.0, worst_excess=0.0)
                                for label in sorted(TENSORS)])
            traces = []
            for role, arm in enumerate(('legacy', 'packed', 'default')):
                slot = (role - rotation) % 3
                ms = (10.0 if arm == 'legacy' else 9.0) * (1, 2, 4)[slot]
                traces.append([ms] * 400)
                run[arm] = dict(label='packed' if arm == 'default' else arm,
                                median_ms=ms, ms_per_step=ms / 8)
            run['timing_samples'] = traces
            parts.append(run)
        return parts

    def merged(self):
        return merge_allocation_runs(self.parts(), samples=1200, warmup=2)

    def test_balancing_removes_position_bias_without_selecting_a_run(self):
        row = self.merged()
        self.assertEqual([row[a]['median_ms'] for a in ('legacy', 'packed', 'default')], [20, 18, 18])
        self.assertEqual([len(t) for t in row['timing_samples']], [1200] * 3)
        self.assertEqual(len(row['allocation_runs']), 3)

    def reject(self, mutate):
        parts = self.parts()
        mutate(parts)
        with self.assertRaises(ValueError):
            merge_allocation_runs(parts, samples=1200, warmup=2)

    def test_missing_rotation(self):
        self.reject(lambda parts: parts.pop())

    def test_duplicate_rotation(self):
        self.reject(lambda parts: parts[1].update(allocation_rotation=0))

    def test_metadata_change(self):
        self.reject(lambda parts: parts[1].update(parameter_storage='fp16-storage'))

    def test_nonfinite_observation(self):
        self.reject(lambda parts: parts[1]['timing_samples'][0].__setitem__(0, float('nan')))

    def test_forged_native_median(self):
        self.reject(lambda parts: parts[1]['packed'].update(median_ms=1.0))

    def test_shortened_trace(self):
        self.reject(lambda parts: parts[1]['timing_samples'][0].pop())

    def test_bad_parity_in_a_nonselected_rotation(self):
        self.reject(lambda parts: parts[1]['tensors'][0].update(max_abs=1e-8))

    def test_warmup_change(self):
        self.reject(lambda parts: parts[1].update(warmup=1))

    def test_guard_rederives_pooled_readings_and_rejects_unbalanced_fallback(self):
        chain = next(
            s for s in guard.ALL_SPECS if s.name == 'tbptt-packed-rematerialization')
        row = self.merged()
        row.update(default_label='packed', default_ms=18, legacy_ms=20, packed_ms=18,
                   decision_timing_sources=dict(legacy='forced-arm', packed='shipped-default'))
        guard.check_shipped_route_timings(chain, row, require_balanced=True)
        bad = copy.deepcopy(row)
        bad['legacy']['median_ms'] = bad['legacy_ms'] = 21
        with self.assertRaises(guard.Problem):
            guard.check_shipped_route_timings(chain, bad, require_balanced=True)
        bad = copy.deepcopy(row)
        bad['timing_protocol'] = 'shipped-route-v1'
        with self.assertRaises(guard.Problem):
            guard.check_shipped_route_timings(chain, bad, require_balanced=True)


if __name__ == '__main__':
    unittest.main()
