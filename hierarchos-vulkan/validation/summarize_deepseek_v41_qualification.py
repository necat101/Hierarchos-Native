"""Recheck every frozen model logit and summarize the current native exports.

This reads existing CPU fixtures/native exports without rewriting expectations.
Optional structure traces additionally verify every stage, shared cache, index
score/selection and routing field. CPU PyTorch is the reference; CUDA is not a
qualification requirement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from deepseek_v41_parity import require_logit_parity
import validate_deepseek_v41_long_context_model as long_context
from validate_deepseek_v41_wide_model import compare_diagnostics

REPO=Path(__file__).resolve().parents[2]
FIXTURES=REPO/"hierarchos-vulkan/validation/fixtures"
MODEL_EXPORTS=[
    ("deepseek_v41_model.json","model-native-logits.json"),
    ("deepseek_v41_model_additional.json","model-additional-native-logits.json"),
    ("deepseek_v41_mixed_model.json","mixed-model-native-logits.json"),
    ("deepseek_v41_attention_model.json","attention-model-native-logits.json"),
    ("deepseek_v41_bf16_model.json","bf16-model-native-logits.json"),
    ("deepseek_v41_wide_model.json","wide-model-native-logits.json"),
    ("deepseek_v41_production_model.json","production-native-logits.json"),
    ("deepseek_v41_schedule_width_model.json","schedule-width-native-logits.json"),
    ("deepseek_v41_structure_model.json","structure-model-native-logits.json"),
    ("deepseek_v41_joint_model.json","scalable-release-native-logits.json"),
    ("deepseek_v41_long_context_model.json","long-context-model-native-logits.json"),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root",type=Path,default=REPO/"target/deepseek-v41-reference")
    p.add_argument("--structure-reference-trace",type=Path)
    p.add_argument("--joint-reference-trace",type=Path)
    p.add_argument("--long-reference-trace",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    models=[]
    routed_values=0
    for fixture_name,export_name in MODEL_EXPORTS:
        fixture_path=FIXTURES/fixture_name
        expected=json.loads(fixture_path.read_text())
        native=json.loads((args.native_root/export_name).read_text())
        binding=hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        ee=expected["cases"] if "cases" in expected else [expected]
        aa=native["cases"] if "cases" in native else [native]
        assert ee and len(ee)==len(aa)
        for e,a in zip(ee,aa):
            assert a["fixture_sha256"]==binding
            if fixture_name in {"deepseek_v41_production_model.json", "deepseek_v41_schedule_width_model.json", "deepseek_v41_joint_model.json"}:
                assert a.get("loader")=="from_hf_package", \
                    "production release gate must be the shipping default loader"
            assert a["revision"]==e["revision"]
            if "seed" in a:
                assert a["seed"]==e["seed"]
            if "checkpoint" in e:
                assert a["checkpoint_files"]==e["checkpoint"]["files"]
            assert e["steps"] and len(e["steps"])==len(a["steps"])
            worst=0.0
            count=0
            for es,actual in zip(e["steps"],a["steps"]):
                assert es["name"]==actual["name"] and es["shape"]==actual["shape"]
                x=torch.tensor(actual["logits"],dtype=torch.float32).reshape(actual["shape"])
                y=torch.tensor(es["logits"],dtype=torch.float32).reshape(es["shape"])
                worst=max(worst,require_logit_parity(x,y,label=f"{e['seed']} {es['name']}"))
                count+=x.numel()
                if "routing" in es:
                    # PyTorch leaves top-k tie order unspecified; compare by expert id.
                    assert actual["routing"].keys()==es["routing"].keys()
                    for layer,route in es["routing"].items():
                        assert long_context.canonical_routing(actual["routing"][layer])==\
                            long_context.canonical_routing(route),"routing bits differ"
                    routed_values+=sum(len(r["indices"]) for r in es["routing"].values())
            models.append(dict(seed=e["seed"],fixture=fixture_name,fixture_sha256=binding,
                steps=len(e["steps"]),logits=count,max_abs_drift=worst,
                bf16=e.get("stream_precision")=="bf16"))
    traces=[]
    joint_traces=[]
    for trace_path,fixture_name,export_name,reports in [
        (args.structure_reference_trace,"deepseek_v41_structure_model.json","structure-model-native-logits.json",traces),
        (args.joint_reference_trace,"deepseek_v41_joint_model.json","joint-model-native-logits.json",joint_traces),
    ]:
        if trace_path is None:
            continue
        fixture=json.loads((FIXTURES/fixture_name).read_text())
        actual=json.loads((args.native_root/export_name).read_text())
        reference=json.loads(trace_path.read_text())
        assert len(fixture["cases"])==len(actual["cases"])==len(reference["cases"])
        for e,a,r in zip(fixture["cases"],actual["cases"],reference["cases"]):
            assert e["seed"]==a["seed"]==r["seed"]
            report=compare_diagnostics(e,a,r)
            assert report["trace_values"]==report["bit_exact_values"]
            reports.append(report)
    long_traces=[]
    if args.long_reference_trace:
        fixture=json.loads((FIXTURES/"deepseek_v41_long_context_model.json").read_text())
        actual=json.loads((args.native_root/"long-context-model-native-logits.json").read_text())
        reference=json.loads(args.long_reference_trace.read_text())
        assert len(fixture["cases"])==len(actual["cases"])==len(reference["cases"])==1
        e,a,r=fixture["cases"][0],actual["cases"][0],reference["cases"][0]
        assert e["seed"]==a["seed"]==r["seed"]
        assert len(r["steps"])==1
        step=r["steps"][0]["name"]
        fixture_step=next(s for s in e["steps"] if s["name"]==step)
        native_step=next(s for s in a["steps"] if s["name"]==step)
        report=long_context.compare_traced(e["config"],fixture_step,native_step,r["steps"][0])
        assert report["trace_values"]==report["bit_exact_values"]
        long_traces.append(report)
    bf16=[m for m in models if m["bf16"]]
    assert all(m["max_abs_drift"]==0 for m in bf16)
    result=dict(reference="single-threaded CPU PyTorch",absolute_logit_limit_strict=2e-7,
        rtol=0,cuda_required=False,model_count=len(models),step_count=sum(m["steps"] for m in models),
        logit_count=sum(m["logits"] for m in models),max_abs_drift=max(m["max_abs_drift"] for m in models),
        bf16_models=len(bf16),bf16_bit_exact_logits=sum(m["logits"] for m in bf16),
        bit_exact_routing_slots=routed_values,structure_traces=traces,joint_traces=joint_traces,
        long_traces=long_traces,models=models,
        scope="joint 40-layer/384-expert/top-6 schedule qualified at reduced widths, including a single 2,048-token prefill at the released 32 index heads; production width/vocabulary qualified in twenty-four layers, plus a separate four-layer compression/shared-KV/index/candidate/Engram/top-6 case, both through the shipping default loader",
        remaining="joint full-size geometry, intermediate grouped-projection bit parity at released widths, released Engram table sizes, prompt chunks after a start-zero prefill, million-token contexts, training, DSpark, vision")
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {"models","structure_traces","joint_traces","long_traces"}},indent=2))


if __name__=="__main__":
    main()
