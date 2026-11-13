"""Prevent shortened or unbalanced timing receipts from qualifying the release."""
import unittest
import check_measured_producer_routes as guard


class SamplingProtocolControls(unittest.TestCase):
    def test_all_route_producers_cover_balanced_rotation_cycles(self):
        producers=guard.PRODUCERS+guard.CHAIN_PRODUCERS
        self.assertEqual(len(producers),11)
        for spec in producers:
            with self.subTest(producer=spec.name):
                samples=int(spec.bench_args[spec.bench_args.index('--iterations')+1])
                self.assertGreaterEqual(samples,120)
                self.assertEqual(samples%3,0)
                self.assertEqual(samples%4,0)
                self.assertEqual(spec.sweep,(1,4,8,16,32))

    def test_valid_counts_and_counterfeit_counts(self):
        for spec in guard.ALL_SPECS:
            if '--iterations' not in spec.bench_args:continue
            expected=int(spec.bench_args[spec.bench_args.index('--iterations')+1])
            guard.check_sample_count(spec,{'samples':expected})
            for bad in (None,True,0,expected-1,expected+1,str(expected)):
                with self.subTest(producer=spec.name,samples=bad):
                    with self.assertRaises(guard.Problem):
                        guard.check_sample_count(spec,{'samples':bad})

    def test_old_short_measurements_cannot_qualify_current_producers(self):
        for spec in guard.PRODUCERS+guard.CHAIN_PRODUCERS:
            for old in (20,40):
                with self.subTest(producer=spec.name,samples=old):
                    with self.assertRaises(guard.Problem):
                        guard.check_sample_count(spec,{'samples':old})


if __name__=='__main__':unittest.main()
