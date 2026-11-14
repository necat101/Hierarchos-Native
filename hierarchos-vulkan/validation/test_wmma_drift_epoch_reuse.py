"""Current mirror-epoch controls retain every historical drift mutation."""
import unittest
from pathlib import Path
from unittest.mock import patch
import test_wmma_drift as base
import check_wmma_drift as checker

CURRENT=Path(__file__).resolve().parent/'release_parity/multimodal_training/20261112-unified-release/phase19x/drift'


@unittest.skipUnless((CURRENT/'current-qualification.json').is_file(),'current drift receipt is not present')
class EpochReuseDriftControls(base.DriftControls):
    @classmethod
    def setUpClass(cls):
        with patch.object(base.q,'RECEIPT_ROOT',CURRENT):
            super().setUpClass()
        checker.verify(CURRENT/'current-qualification.json',epoch_reuse=True)

    def expect(self,fragment):
        self.write()
        with self.assertRaises(checker.QualificationError) as context:
            checker.verify(self.root/'current-qualification.json',epoch_reuse=True)
        self.assertIn(fragment,str(context.exception))

    def test_no_weight_mirror_conversion(self):
        # Step 3 follows no evaluation. Retain full mirror-use accounting so
        # rejection specifically proves an evaluation must precede epoch reuse.
        row=self.run_record(1)['records'][3]
        row.update(weight_conversions=0,mirror_reuses=2*checker.PER_STEP_PHASES)
        self.expect('without a preceding evaluation')

    def test_pretraining_evaluation_must_use_initial_weights(self):
        row=self.run_record(1)['evaluations'][0]
        row.update(parameter_digest_before='0'*64,parameter_digest_after='0'*64)
        self.expect('different parameter epoch')

    def test_later_evaluation_must_use_preceding_update(self):
        row=next(e for e in self.run_record(1)['evaluations'] if e['after_step']==1)
        row.update(parameter_digest_before='0'*64,parameter_digest_after='0'*64)
        self.expect('different parameter epoch')

    def test_all_mirror_uses_are_still_required(self):
        self.run_record(1)['records'][0]['mirror_reuses']-=1
        self.expect('account for every mirror use')

    def test_legacy_conversion_policy_remains_separate(self):
        with self.assertRaisesRegex(checker.QualificationError,'no weight-mirror conversion'):
            checker.verify(CURRENT/'current-qualification.json')


if __name__=='__main__':unittest.main()
