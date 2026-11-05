"""Controls preventing a partial/empty SigLIP replay from becoming a pass."""
import json
import copy
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import qualify_gemma3_pipeline as q


class Gemma3GateTests(unittest.TestCase):
    def rejected_manifest(self, cases, requested=None, sources=None):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest={"complete":True,"logit_absolute_limit":2e-7,
                "parameter_absolute_limit":2e-7,"rtol":0,"comparison":"strict_less_than",
                "cases":cases,"oracle":{"cpu":"AVX2","source_sha256":sources or {}}}
            (root/"manifest.json").write_text(json.dumps(manifest))
            output=root/"native"
            args=SimpleNamespace(fixtures=root,output=output,cases=requested)
            with self.assertRaises(ValueError):
                q.run(args)
            self.assertFalse(output.exists())

    def test_empty_frozen_inventory_cannot_pass(self):
        self.rejected_manifest([])

    def test_duplicate_frozen_case_cannot_inflate_coverage(self):
        self.rejected_manifest([{"name":"image"},{"name":"image"}])

    def test_unknown_and_empty_requests_cannot_pass(self):
        for request in ([],["missing"]):
            self.rejected_manifest([{"name":"image"}],request)

    def test_modified_oracle_source_cannot_run(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/"oracle.py"
            source.write_text("original")
            source_hash=q.digest(source)
            source.write_text("changed")
            (root/"manifest.json").write_text(json.dumps({"complete":True,
                "logit_absolute_limit":2e-7,"parameter_absolute_limit":2e-7,"rtol":0,
                "comparison":"strict_less_than","cases":[{"name":"image"}],
                "oracle":{"cpu":"AVX2","source_sha256":{"oracle.py":source_hash}}}))
            output=root/"native"
            with self.assertRaisesRegex(ValueError,"frozen CPU source changed"):
                q.run(SimpleNamespace(fixtures=root,output=output,cases=None))
            self.assertFalse(output.exists())


@unittest.skipUnless(os.environ.get("HIERARCHOS_GEMMA3_TEST_BINARY") and
    os.environ.get("HIERARCHOS_GEMMA3_TEST_FIXTURES"), "native Gemma3 fixture paths not configured")
class NativeGemma3Guards(unittest.TestCase):
    def test_malformed_inputs_cannot_export_or_change_source(self):
        root=Path(os.environ["HIERARCHOS_GEMMA3_TEST_FIXTURES"])/"siglip_default"
        model=root/"model"
        fixture=json.loads((root/"input.json").read_text())
        original={str(p):q.digest(p) for p in root.rglob("*") if p.is_file()}
        def short_rgb(f):
            f["rgb"].pop()
        def hidden_image(f):
            f["attention_mask"][f["input_ids"].index(63)]=0.0
        def nonbinary_mask(f):
            f["attention_mask"][0]=0.5
        def wrong_image_count(f):
            f["image_count"]=0
        for label,change in [("short_rgb",short_rgb),("hidden_image",hidden_image),
            ("nonbinary_mask",nonbinary_mask),("wrong_image_count",wrong_image_count)]:
            with self.subTest(label=label), TemporaryDirectory() as temporary:
                folder=Path(temporary)
                altered=copy.deepcopy(fixture)
                change(altered)
                path=folder/"input.json"
                path.write_text(json.dumps(altered))
                output=folder/"native"
                result=subprocess.run([os.environ["HIERARCHOS_GEMMA3_TEST_BINARY"],
                    "--model",str(model),"--fixture",str(path),"--output",str(output)],
                    capture_output=True,text=True,timeout=60)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertFalse(list(output.rglob("*.safetensors")))
        self.assertEqual(original,{str(p):q.digest(p) for p in root.rglob("*") if p.is_file()})


if __name__=="__main__":
    unittest.main()
