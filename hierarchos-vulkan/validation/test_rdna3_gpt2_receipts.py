"""Negative controls for the expanded qualification's acceptance guard."""
import copy
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
