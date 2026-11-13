"""Fault injection starts from the genuinely passing unified release receipt."""
import copy, json, os, tempfile, unittest
from pathlib import Path
import check_unified_release as c
from check_phase19z_current import QualificationError


class UnifiedReleaseControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not c.DEFAULT.exists():raise unittest.SkipTest('unified release not sealed yet')
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
    def test_wrong_shader(self):
        self.reject_binding(lambda d:d['source_files'].__setitem__('hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.spv','0'*64))
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
    def test_hidden_diagnostic_failures(self):
        self.reject(lambda d,p:p['intermediates'].update(strict_diagnostic_failures=0))
    def test_forged_primitive_pass_flag(self):
        def mutate(d,p):
            row=next(r for r in p['wmma_full_original']['results'] if not r['passed']);row['passed']=True
        self.reject(mutate)
    def test_partial_packed_cell_cannot_qualify(self):
        self.reject(lambda d,p:p['packed']['results'][p['packed']['passes'][0]].update(guard=1))


if __name__=='__main__':unittest.main()
