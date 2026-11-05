"""Controls for frozen Gemma4 coverage and fail-closed native inputs."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import qualify_gemma4_pipeline as q


class Gemma4GateTests(unittest.TestCase):
    def rejected_manifest(self,cases,requested=None):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'native'
            (root/'manifest.json').write_text(json.dumps({'complete':True,
                'logit_absolute_limit':2e-7,'parameter_absolute_limit':2e-7,'rtol':0,
                'comparison':'strict_less_than','cases':cases,
                'oracle':{'cpu':'AVX2','source_sha256':{}}}))
            with self.assertRaises(ValueError):q.run(SimpleNamespace(fixtures=root,output=output,cases=requested))
            self.assertFalse(output.exists())

    def test_empty_inventory_cannot_pass(self):self.rejected_manifest([])

    def test_duplicate_cases_cannot_inflate_coverage(self):
        self.rejected_manifest([{'name':'image'},{'name':'image'}])

    def test_empty_and_unknown_requests_cannot_pass(self):
        for requested in [[],['missing']]:self.rejected_manifest([{'name':'image'}],requested)

    def test_unknown_pixel_domain_cannot_run(self):
        self.rejected_manifest([{'name':'image','pixel_input_domain':'unknown'}])

    def test_modified_cpu_source_cannot_run(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'oracle.py';source.write_text('original')
            expected=q.digest(source);source.write_text('changed');output=root/'native'
            (root/'manifest.json').write_text(json.dumps({'complete':True,
                'logit_absolute_limit':2e-7,'parameter_absolute_limit':2e-7,'rtol':0,
                'comparison':'strict_less_than','cases':[{'name':'image'}],
                'oracle':{'cpu':'AVX2','source_sha256':{'oracle.py':expected}}}))
            with self.assertRaisesRegex(ValueError,'frozen CPU source changed'):
                q.run(SimpleNamespace(fixtures=root,output=output,cases=None))
            self.assertFalse(output.exists())


@unittest.skipUnless(os.environ.get('HIERARCHOS_GEMMA4_TEST_BINARY') and
    os.environ.get('HIERARCHOS_GEMMA4_TEST_FIXTURES'),'native Gemma4 fixture paths not configured')
class NativeGemma4Guards(unittest.TestCase):
    def test_bad_inputs_and_unsupported_features_cannot_export_or_modify_source(self):
        root=Path(os.environ['HIERARCHOS_GEMMA4_TEST_FIXTURES'])/'gemma4_default'
        original={str(p):q.digest(p) for p in root.rglob('*') if p.is_file()}
        fixture=json.loads((root/'input.json').read_text())
        for label in ['short_rgb','short_alternate_rgb','hidden_image','nonbinary_mask',
                      'image_count','standardize','clipped_linears','processor_geometry',
                      'invalid_rescale']:
            with self.subTest(case=label),TemporaryDirectory() as temporary:
                folder=Path(temporary);f=copy.deepcopy(fixture);model=root/'model'
                if label=='short_rgb':f['rgb'].pop()
                elif label=='short_alternate_rgb':f['alternate_rgb'].pop()
                elif label=='hidden_image':f['attention_mask'][f['input_ids'].index(63)]=0.
                elif label=='nonbinary_mask':f['attention_mask'][0]=.5
                elif label=='image_count':f['image_count']=0
                else:
                    model=folder/'model';shutil.copytree(root/'model',model)
                    if label in ['processor_geometry','invalid_rescale']:
                        path=model/'preprocessor_config.json';config=json.loads(path.read_text())
                        config['patch_size' if label=='processor_geometry' else 'rescale_factor']=3 if label=='processor_geometry' else 0.
                    else:
                        path=model/'config.json';config=json.loads(path.read_text())
                        config['vision_config']['standardize' if label=='standardize' else 'use_clipped_linears']=True
                    path.write_text(json.dumps(config))
                path=folder/'input.json';path.write_text(json.dumps(f));output=folder/'native'
                result=subprocess.run([os.environ['HIERARCHOS_GEMMA4_TEST_BINARY'],
                    '--model',str(model),'--fixture',str(path),'--output',str(output)],
                    capture_output=True,text=True,timeout=60)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertFalse(list(output.rglob('*.safetensors')))
        self.assertEqual(original,{str(p):q.digest(p) for p in root.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()
