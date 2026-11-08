"""Negative controls for the expanded qualification's acceptance guard."""
import copy
import json
from pathlib import Path
import unittest
from check_rdna3_gpt2_receipts import validate_qualification

class ReceiptGuardTests(unittest.TestCase):
    def setUp(self):
        self.receipt=dict(target='rdna3',complete=True,rtol=0,comparison='strict_less_than',
            absolute_limit=2e-7,expected_cases=1,completed_cases=1,**{'pass':True},
            results=[dict(case='control',**{'pass':True},checks=[dict(metric='final/logits',
                max_abs=1e-7,**{'pass':True})])])

    def test_rederives_strict_limit_even_when_pass_flags_are_true(self):
        validate_qualification(self.receipt,1)
        for value in [2e-7,float('inf'),float('nan')]:
            changed=copy.deepcopy(self.receipt);changed['results'][0]['checks'][0]['max_abs']=value
            with self.assertRaises(ValueError):validate_qualification(changed,1)

    def test_rejects_missing_cases_and_false_subchecks(self):
        changed=copy.deepcopy(self.receipt);changed['results']=[]
        with self.assertRaises(ValueError):validate_qualification(changed,1)
        changed=copy.deepcopy(self.receipt);changed['results'][0]['checks'][0]['pass']=False
        with self.assertRaises(ValueError):validate_qualification(changed,1)

    def test_rejects_relaxed_policy(self):
        changed=copy.deepcopy(self.receipt);changed['comparison']='less_than_or_equal'
        with self.assertRaises(ValueError):validate_qualification(changed,1)

if __name__=='__main__':unittest.main()


from check_rdna3_gpt2_receipts import verify_sources, verify_current


class RevisionAwareSourceTests(unittest.TestCase):
    def setUp(self):
        self.sources = {'hierarchos-vulkan/src/transformer.rs': 'a' * 64}

    def test_current_bytes_verify_without_history(self):
        def no_history(name, expected):
            raise AssertionError('a matching checkout must not need history')

        current, historical = verify_sources(
            self.sources, current_sha256=lambda name: 'a' * 64, history_lookup=no_history
        )
        self.assertEqual(list(current), list(self.sources))
        self.assertEqual(historical, {})

    def test_changed_bytes_verify_against_immutable_history(self):
        current, historical = verify_sources(
            self.sources,
            current_sha256=lambda name: 'b' * 64,
            history_lookup=lambda name, expected: 'c7e096d0948317dd6d4d9a1f9de5674703539d5e',
        )
        self.assertEqual(current, {})
        self.assertEqual(historical[self.sources and next(iter(self.sources))]['sha256'], 'a' * 64)

    def test_bytes_absent_from_history_fail_closed(self):
        with self.assertRaises(ValueError):
            verify_sources(
                self.sources,
                current_sha256=lambda name: 'b' * 64,
                history_lookup=lambda name, expected: None,
            )

    def test_missing_file_still_consults_history(self):
        def missing(name):
            raise FileNotFoundError(name)

        current, historical = verify_sources(
            self.sources,
            current_sha256=missing,
            history_lookup=lambda name, expected: '61591bc6b5187560e08f65b0158db2e711459815',
        )
        self.assertEqual(len(historical), 1)


class CurrentQualificationTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'current-qualification.json'
        self.sources = {'hierarchos-vulkan/src/transformer.rs': 'a' * 64}
        self.head = '1' * 40

    def tearDown(self):
        self.dir.cleanup()

    def write(self, payload):
        self.path.write_text(json.dumps(payload))

    def test_missing_current_receipt_is_unqualified(self):
        status = verify_current(
            self.sources, receipt_path=self.path, source_head=self.head,
            current_sha256=lambda name: 'a' * 64,
        )
        self.assertEqual(status, 3)

    def test_current_receipt_bound_to_this_checkout_passes(self):
        self.write({'source_head': self.head, 'pass': True, 'sources': self.sources})
        status = verify_current(
            self.sources, receipt_path=self.path, source_head=self.head,
            current_sha256=lambda name: 'a' * 64,
        )
        self.assertEqual(status, 0)

    def test_historical_receipt_cannot_qualify_the_current_checkout(self):
        self.write({'source_head': 'c7e096d0948317dd6d4d9a1f9de5674703539d5e',
                    'pass': True, 'sources': self.sources})
        with self.assertRaises(ValueError):
            verify_current(
                self.sources, receipt_path=self.path, source_head=self.head,
                current_sha256=lambda name: 'a' * 64,
            )

    def test_current_receipt_must_bind_every_source(self):
        self.write({'source_head': self.head, 'pass': True, 'sources': {}})
        with self.assertRaises(ValueError):
            verify_current(
                self.sources, receipt_path=self.path, source_head=self.head,
                current_sha256=lambda name: 'a' * 64,
            )

    def test_failed_current_qualification_is_not_promoted(self):
        self.write({'source_head': self.head, 'pass': False, 'sources': self.sources})
        with self.assertRaises(ValueError):
            verify_current(
                self.sources, receipt_path=self.path, source_head=self.head,
                current_sha256=lambda name: 'a' * 64,
            )


class ShallowHistoryTests(unittest.TestCase):
    """A shallow checkout must report missing history, not source corruption."""

    def setUp(self):
        self.sources = {'hierarchos-vulkan/src/transformer.rs': 'a' * 64}

    def test_shallow_checkout_reports_missing_history(self):
        with self.assertRaises(ValueError) as caught:
            verify_sources(
                self.sources,
                current_sha256=lambda name: 'b' * 64,
                history_lookup=lambda name, expected: None,
                history_available=lambda: False,
            )
        self.assertIn('shallow', str(caught.exception))
        self.assertIn('fetch-depth: 0', str(caught.exception))

    def test_full_checkout_without_the_blob_is_a_real_failure(self):
        with self.assertRaises(ValueError) as caught:
            verify_sources(
                self.sources,
                current_sha256=lambda name: 'b' * 64,
                history_lookup=lambda name, expected: None,
                history_available=lambda: True,
            )
        self.assertIn('no verifiable revision', str(caught.exception))
        self.assertNotIn('shallow', str(caught.exception))

    def test_shallow_checkout_passes_when_every_bound_byte_still_matches(self):
        current, historical = verify_sources(
            self.sources,
            current_sha256=lambda name: 'a' * 64,
            history_lookup=lambda name, expected: None,
            history_available=lambda: False,
        )
        self.assertEqual(len(current), 1)
        self.assertEqual(historical, {})
