"""Fault injection starts from the genuinely passing unified release receipt."""
import copy, json, os, tempfile, unittest
from pathlib import Path
import check_unified_release as c
from check_phase19z_current import QualificationError


class UnifiedReleaseControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not c.DEFAULT.exists():
            raise AssertionError('unified release receipt missing: '+str(c.DEFAULT))
        cls.original,cls.components=c.verify(check_subreceipts=False)

    def reject(self,mutate):
        doc=copy.deepcopy(self.original);parts=copy.deepcopy(self.components);mutate(doc,parts)
        with self.assertRaises((ValueError,QualificationError)):c.validate_payload(doc,parts)

    def reject_binding(self,mutate):
        doc=copy.deepcopy(self.original);mutate(doc)
        fd,name=tempfile.mkstemp(prefix='.negative-unified-',suffix='.json',dir=c.DEFAULT.parent);os.close(fd)
        path=Path(name)
        try:
            path.write_text(json.dumps(doc))
            with self.assertRaises((ValueError,QualificationError)):c.verify(path,check_subreceipts=False)
        finally:path.unlink()

    def test_positive_including_all_current_source_subcheckers(self):c.verify()
    def test_stale_source(self):
        self.reject_binding(lambda d:d['source_files'].__setitem__('hierarchos-vulkan/src/transformer.rs','0'*64))
    def test_host_assignment_policy_binding_cannot_be_removed(self):
        self.reject_binding(lambda d:d['source_files'].pop(c.HOST_QUALIFICATIONS))
    def test_stale_host_assignment_policy(self):
        self.reject_binding(lambda d:d['source_files'].__setitem__(c.HOST_QUALIFICATIONS,'0'*64))
    def test_pending_other_host_verification_cannot_be_hidden(self):
        self.reject_binding(lambda d:d['fixture_provenance'].update(deferred_host_verification=[]))
    def test_release_cannot_claim_the_other_cpu_target(self):
        self.reject(lambda d,p:d['qualification_target'].update(cpu='AVX2'))
    def test_wrong_shader(self):
        self.reject_binding(lambda d:d['source_files'].__setitem__('hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.spv','0'*64))
    def test_shared_shader_cannot_replace_isolated_rdna3_coverage(self):
        def mutate(d,p):
            rows=p['shader727']['shaders']
            i=next(i for i,r in enumerate(rows) if r['artifact']=='rwkv_key_state_forward_fused_wide_amd_rdna3.spv')
            rows[i]=next(r for r in rows if r['artifact']=='rwkv_key_state_forward_fused_wide.spv')
        self.reject(mutate)
    def test_wrong_executable(self):
        self.reject(lambda d,p:d['binaries']['image_text_pipeline_probe'].update(sha256='0'*64))
    def test_component_hash_mismatch(self):
        self.reject_binding(lambda d:d['components']['precision'].update(sha256='0'*64))
    def test_missing_fp32_comparisons_with_forged_pass(self):
        self.reject(lambda d,p:p['text166'].update(records=[],**{'pass':True}))
    def test_missing_image_case_with_forged_pass(self):
        self.reject(lambda d,p:p['shared77']['results'].pop())
    def test_missing_mixed_dispatches(self):
        self.reject(lambda d,p:p['precision']['reports']['full-mixed']['precision'].update(accumulation_dispatches=[0,0]))
    def test_full_fp16_silently_using_mixed(self):
        self.reject(lambda d,p:p['precision']['reports']['full-full-fp16']['precision'].update(accumulation_dispatches=[480,0]))
    def test_unqualified_full_reduction_range(self):
        self.reject(lambda d,p:d['precision_policy'].update(full_fp16_max_reduction=128))
    def test_incomplete_optimizer_trajectory(self):
        self.reject(lambda d,p:p['precision']['reports']['full-mixed'].update(optimizer_steps=19))
    def test_incomplete_accumulation_window(self):
        self.reject(lambda d,p:p['precision']['reports']['accumulation-full-full-fp16']['loss_scaler'].update(pending_microbatches=1))
    def test_nonfinite_loss(self):
        self.reject(lambda d,p:p['precision']['losses']['full-mixed'].__setitem__(1,float('nan')))
    def test_failed_wide_family_with_forged_outer_pass(self):
        self.reject(lambda d,p:p['wide']['results'][0].update(status='FAIL'))
    def test_failed_wide_numeric_observation(self):
        def mutate(d,p):
            metrics=p['wide']['results'][0]['checks']
            key=next(k for k,v in metrics.items() if k.startswith('mixed/step-') and 'max_abs' in v)
            metrics[key].update(max_abs=.5)
        self.reject(mutate)
    def test_incomplete_architecture_coverage(self):
        self.reject(lambda d,p:p['wide']['results'].pop())
    def test_unsupported_precision_accepted(self):
        self.reject(lambda d,p:p['gui_cli']['unsupported_controls'][0].update(exit_code=0))
    def test_missing_rejection_control(self):
        self.reject(lambda d,p:p['gui_cli']['unsupported_controls'].pop())
    def test_duplicated_gui_workflow_does_not_replace_coverage(self):
        self.reject(lambda d,p:p['gui_cli']['workflows'].__setitem__(0,p['gui_cli']['workflows'][1]))
    def test_primitive_from_another_executable(self):
        self.reject(lambda d,p:p['wmma_mixed']['producer'].update(binary_sha256='0'*64))
    def test_oversized_model_fallback_must_execute(self):
        self.reject(lambda d,p:p['tests']['oversized_fallback']['model_export']['precision'].update(full_fp16_reduction_fallbacks=0))
    def test_restored_scaler_history_is_required(self):
        def mutate(d,p):
            key=next(k for k in p['precision']['checks'] if k.startswith('vision-overflow-') and '/restored-state' in k)
            p['precision']['checks'][key].update(skipped_steps=0)
        self.reject(mutate)
    def test_hidden_diagnostic_failures(self):
        self.reject(lambda d,p:p['intermediates'].update(strict_diagnostic_failures=0))
    def test_forged_diagnostic_flags_cannot_hide_observed_failures(self):
        def mutate(d,p):
            for row in p['intermediates']['raw_diagnostics']:row['pass']=True
            p['intermediates']['strict_diagnostic_failures']=0
        self.reject(mutate)
    def test_forged_native_bounded_qualification(self):
        def mutate(d,p):
            checks=p['native_v41']['checks']['bounded']
            key=next(k for k in checks if k.endswith('independent-FP32'))
            checks[key]['max_abs']=5e-7
        self.reject(mutate)
    def forge_cooperative(self,p,counts=(32,0),phase_counts=(16,8,8)):
        """A fully self-consistent receipt that relabels the route as cooperative."""
        for label in ('bounded','stress'):
            route=p['native_v41']['reports'][label]['route']
            route.update(scope='mixed_cooperative_numerical',measured_route='cooperative_mixed',
                cooperative_mixed_executed=True,accumulation_dispatches=list(counts),phases=list(phase_counts),
                declined_by=None)
            p['native_v41']['reports'][label]['precision'].update(accumulation_dispatches=list(counts),phases=list(phase_counts))
            for key,row in p['native_v41']['checks'][label].items():
                if key.endswith('mixed-vs-FP32'):row['route']='cooperative_parity'
        p['native_v41']['qualification_scope'].update(kind='mixed_cooperative_numerical',
            measured_route='cooperative_mixed',cooperative_mixed_executed=True,not_claimed=[])
    def test_native_fp32_qualification_is_accepted(self):
        # The positive path is the sealed receipt itself: both fixtures must be
        # a numerical qualification on the route they actually measured.
        self.assertTrue(self.original['native_v41_scope']['cooperative_mixed_executed'] is False)
        self.assertEqual(self.original['native_v41_scope']['kind'],'numerical_fp32_route')
        c.verify_native_route(self.components['native_v41'])
    def test_native_requires_every_bounded_comparison(self):
        def mutate(d,p):
            key=next(k for k in p['native_v41']['checks']['bounded'] if k.endswith('independent-FP32'))
            p['native_v41']['checks']['bounded'].pop(key)
        self.reject(mutate)
    def test_native_requires_every_stress_comparison(self):
        def mutate(d,p):
            key=next(k for k in p['native_v41']['checks']['stress'] if k.endswith('independent-FP32'))
            p['native_v41']['checks']['stress'].pop(key)
        self.reject(mutate)
    def test_native_stress_strict_limit_still_enforced(self):
        def mutate(d,p):
            key=next(k for k in p['native_v41']['checks']['stress'] if k.endswith('independent-FP32'))
            p['native_v41']['checks']['stress'][key].update(max_abs=2e-7,within_strict_limit=False)
        self.reject(mutate)
    def test_native_reload_must_stay_exact(self):
        def mutate(d,p):
            p['native_v41']['checks']['bounded']['mixed/reload']['max_abs']=1e-9
        self.reject(mutate)
    def test_native_route_counters_must_match_the_executed_precision(self):
        def mutate(d,p):
            p['native_v41']['reports']['bounded']['route'].update(accumulation_dispatches=[32,0],phases=[16,8,8])
        self.reject(mutate)
    def test_native_fallback_cannot_be_relabeled_as_cooperative(self):
        def mutate(d,p):
            route=p['native_v41']['reports']['bounded']['route']
            route.update(scope='mixed_cooperative_numerical',measured_route='cooperative_mixed',
                cooperative_mixed_executed=True,declined_by=None)
        self.reject(mutate)
    def test_native_cooperative_claim_below_the_profitability_floor(self):
        def mutate(d,p):
            self.forge_cooperative(p);d['native_v41_scope']=c.native_scope_summary(p['native_v41'])
        self.reject(mutate)
    def test_native_historical_dispatch_disclosure_cannot_be_removed(self):
        def mutate(d,p):
            p['native_v41']['qualification_scope']['historical_pre_gate_receipts']=[]
        self.reject(mutate)
    def test_native_historical_receipt_cannot_be_presented_as_current(self):
        def mutate(d,p):
            history=p['native_v41']['qualification_scope']['historical_pre_gate_receipts']
            history[0]['release_source_revision']=p['native_v41']['source_revision']
        self.reject(mutate)
    def test_native_scope_cannot_claim_cooperative_qualification(self):
        def mutate(d,p):
            p['native_v41']['qualification_scope'].update(kind='mixed_cooperative_numerical')
            d['native_v41_scope']=c.native_scope_summary(p['native_v41'])
        self.reject(mutate)
    def test_native_manifest_scope_must_match_the_component(self):
        def mutate(d,p):
            d['native_v41_scope']['cooperative_mixed_executed']=True
        self.reject(mutate)
    def test_native_parity_label_must_match_the_measured_route(self):
        def mutate(d,p):
            key=next(k for k in p['native_v41']['checks']['bounded'] if k.endswith('mixed-vs-FP32'))
            p['native_v41']['checks']['bounded'][key]['route']='cooperative_parity'
        self.reject(mutate)
    def test_separate_current_source_wmma_evidence_is_required(self):
        def mutate(d,p):p.pop('wmma_mixed')
        self.reject(mutate)
    def test_missing_replay_stage_with_forged_aggregate_pass(self):
        self.reject(lambda d,p:p['fp32_replay']['results'].pop())
    def test_forged_primitive_pass_flag(self):
        def mutate(d,p):
            row=next(r for r in p['wmma_full_original']['results'] if not r['passed']);row['passed']=True
        self.reject(mutate)
    def test_partial_packed_cell_cannot_qualify(self):
        self.reject(lambda d,p:p['packed']['results'][p['packed']['passes'][0]].update(guard=1))


if __name__=='__main__':unittest.main()
