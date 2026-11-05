import unittest
import qualify_shared_image_pipeline as q


class CoverageTests(unittest.TestCase):
    def row(self,case,passed=True,profile='bert'):
        return {'case':case,'profile':profile,'pass':passed}

    def test_partial_profile_does_not_qualify(self):
        self.assertEqual(q.inventory_coverage(['bert'],[self.row('bert/unmasked')]),['bert'])

    def test_one_failed_mask_case_keeps_profile_pending(self):
        self.assertEqual(q.inventory_coverage(['bert'],[
            self.row('bert/unmasked'),self.row('bert/padded',False)]),['bert'])

    def test_all_required_cases_qualify(self):
        self.assertEqual(q.inventory_coverage(['bert'],[
            self.row('bert/unmasked'),self.row('bert/padded')]),[])

    def test_alias_cannot_claim_another_profile(self):
        self.assertEqual(q.inventory_coverage(['bert'],[
            self.row('bert/unmasked',profile='alias'),self.row('bert/padded')]),['bert'])

    def test_empty_or_duplicate_inventory_and_cases_rejected(self):
        for required,rows in [([],[]),(['bert','bert'],[]),
                (['bert'],[self.row('bert/unmasked'),self.row('bert/unmasked')])]:
            with self.assertRaises(ValueError):q.inventory_coverage(required,rows)

    def test_kimi_scope_matches_original_unpadded_cpu_qualification(self):
        self.assertEqual(q.inventory_coverage(['kimi_k3_kda'],[
            self.row('kimi_k3_kda/unmasked',profile='kimi_k3_kda')]),[])

    def test_missing_runtime_keeps_full_inventory_pending(self):
        self.assertEqual(q.inventory_coverage(['bert','deepseek_v41_scalable'],[
            self.row('bert/unmasked'),self.row('bert/padded')]),['deepseek_v41_scalable'])

    def test_policy_and_requested_coverage_fail_closed(self):
        manifest={'complete':True,'absolute_limit':2e-7,'rtol':0,'comparison':'strict_less_than',
            'cases':[{'name':'bert/unmasked','profile':'bert'}]}
        for requested in [[],['unknown']]:
            with self.assertRaises(ValueError):q.validate_manifest(manifest,requested)
        manifest['rtol']=1e-5
        with self.assertRaises(ValueError):q.validate_manifest(manifest,None)


if __name__=='__main__':unittest.main()
