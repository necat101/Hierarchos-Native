import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import qualify_llava_pipeline as q

class PipelineAcceptance(unittest.TestCase):
    def test_frozen_cpu_vector_width_is_pinned(self):
        self.assertEqual(q.oracle_env("AVX2")["HIERARCHOS_ATEN_VECTOR_WIDTH"],"8")
        self.assertEqual(q.oracle_env("AVX512")["HIERARCHOS_ATEN_VECTOR_WIDTH"],"16")
        with self.assertRaises(ValueError):q.oracle_env("unknown")

    def test_limit_is_strict(self):
        self.assertFalse(q.compare([2e-7],[0.])["pass"])
        self.assertTrue(q.compare([1e-7],[0.])["pass"])

    def test_empty_matrix_and_duplicate_cases_cannot_qualify(self):
        for cases in [[],[{"name":"x"},{"name":"x"}]]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                (root/"manifest.json").write_text(json.dumps({"complete":True,
                    "logit_absolute_limit":2e-7,"parameter_absolute_limit":2e-7,
                    "rtol":0,"comparison":"strict_less_than","oracle":{"source_sha256":{}},"cases":cases}))
                with self.assertRaisesRegex(ValueError,"coverage"):
                    q.run(SimpleNamespace(fixtures=root,output=root/"out"))
                self.assertFalse((root/"out").exists())

    def test_relaxed_or_incomplete_policy_cannot_run(self):
        for complete,limit,rtol in [(False,2e-7,0),(True,3e-7,0),(True,2e-7,1e-5)]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                (root/"manifest.json").write_text(json.dumps({"complete":complete,
                    "logit_absolute_limit":limit,"parameter_absolute_limit":2e-7,"rtol":rtol,"comparison":"strict_less_than"}))
                with self.assertRaises(ValueError):q.run(SimpleNamespace(fixtures=root))

    def test_nonfinite_intermediates_are_errors(self):
        with self.assertRaises(ValueError):q.compare([float("nan")],[0.])


@unittest.skipUnless(os.environ.get("HIERARCHOS_LLAVA_TEST_BINARY") and os.environ.get("HIERARCHOS_LLAVA_TEST_FIXTURES"),"native image fixtures/binary are opt-in")
class NativeImageGuards(unittest.TestCase):
    def test_invalid_inputs_fail_before_export(self):
        root=Path(os.environ["HIERARCHOS_LLAVA_TEST_FIXTURES"])/"clip_default_layer"
        fixture=json.loads((root/"input.json").read_text())
        before=q.digest(root/"model"/"model.safetensors")
        cases={}
        x=copy.deepcopy(fixture);x["rgb"]=x["rgb"][:-1];cases["truncated RGB"]=x
        x=copy.deepcopy(fixture);x["input_ids"][1]=5;cases["missing placeholder"]=x
        x=copy.deepcopy(fixture);x["attention_mask"][1]=0.;cases["masked image"]=x
        x=copy.deepcopy(fixture);x["attention_mask"][0]=.5;cases["nonbinary mask"]=x
        x=copy.deepcopy(fixture);x["image_count"]=0;cases["zero images"]=x
        x=copy.deepcopy(fixture);x["targets"][0]=63;x["loss_weights"][0]=1.;cases["image target"]=x
        for label,value in cases.items():
            with self.subTest(label=label),tempfile.TemporaryDirectory() as folder:
                p=Path(folder);(p/"input.json").write_text(json.dumps(value))
                result=subprocess.run([os.environ["HIERARCHOS_LLAVA_TEST_BINARY"],"--model",str(root/"model"),
                    "--fixture",str(p/"input.json"),"--output",str(p/"native")],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertFalse((p/"native"/"step-1").exists())
        self.assertEqual(q.digest(root/"model"/"model.safetensors"),before)

if __name__=="__main__":unittest.main()
