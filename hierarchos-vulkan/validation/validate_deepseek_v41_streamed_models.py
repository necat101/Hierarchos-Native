"""Check checkpoint-streamed execution against frozen CPU and resident outputs.

This validator never regenerates or writes any expectations. These are new
execution modes of existing models; they do not increase distinct-model counts.
Memory figures are live Vulkan tensor bytes, not driver arena reservations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from deepseek_v41_parity import require_logit_parity
import validate_deepseek_v41_model as model
from validate_deepseek_v41_long_context_model import canonical_routing, compare_traced, TRACE_STEP
from validate_deepseek_v41_wide_model import compare_diagnostics

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [
    ("deepseek_v41_mixed_model.json", "mixed-model-native-logits.json", "streamed-mixed-model-native-logits.json", ""),
    ("deepseek_v41_bf16_model.json", "bf16-model-native-logits.json", "streamed-bf16-model-native-logits.json", "bf16"),
    ("deepseek_v41_structure_model.json", "structure-model-native-logits.json", "streamed-structure-model-native-logits.json", "structure"),
]
JOINT = ("deepseek_v41_joint_model.json", "joint-model-native-logits.json", "streamed-joint-model-native-logits.json", "joint")
PRODUCTION = ("deepseek_v41_production_model.json", "production-native-logits.json", "streamed-production-native-logits.json", "production")
LONG_CONTEXT = ("deepseek_v41_long_context_model.json", "long-context-model-native-logits.json", "streamed-long-context-model-native-logits.json", "long_context")


def cases(payload):
    return payload["cases"] if "cases" in payload else [payload]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root", type=Path, default=ROOT / "target/deepseek-v41-reference")
    p.add_argument("--include-joint", action="store_true")
    p.add_argument("--include-production", action="store_true")
    p.add_argument("--include-long-context", action="store_true")
    p.add_argument("--all-experts", action="store_true",
                   help="validate StreamedAllExperts, including shared-expert staging")
    p.add_argument("--streamed-tables", action="store_true",
                   help="validate all-expert streaming with checkpoint-staged Engram tables")
    p.add_argument("--streamed-projections", action="store_true",
                   help="validate all-expert, Engram-table and attention/indexer/compressor "
                        "projection streaming")
    p.add_argument("--streamed-vocabulary", action="store_true",
                   help="also stage embedding and head for their individual operations")
    for name in ("bf16", "structure", "joint", "production", "long-context"):
        p.add_argument(f"--{name}-reference-trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.include_long_context and args.long_context_reference_trace is None:
        p.error("--include-long-context requires --long-context-reference-trace")
    if args.streamed_vocabulary:
        if args.streamed_projections or args.streamed_tables or args.all_experts:
            p.error("--streamed-vocabulary already selects projection, table and all-expert streaming")
        args.streamed_projections = True
    if args.streamed_tables and args.all_experts:
        p.error("--streamed-tables already selects all-expert streaming")
    if args.streamed_projections and (args.streamed_tables or args.all_experts):
        p.error("--streamed-projections already selects all-expert and table streaming")
    if args.streamed_projections:
        args.streamed_tables = args.all_experts = True
    torch.set_num_threads(1)
    reports = []
    pairs = PAIRS + ([JOINT] if args.include_joint else []) + ([PRODUCTION] if args.include_production else [])
    if args.include_long_context:
        pairs.append(LONG_CONTEXT)
    for fixture_name, resident_name, streamed_name, trace_name in pairs:
        path = model.FIXTURE.parent / fixture_name
        expected = cases(json.loads(path.read_text()))
        resident = cases(json.loads((args.native_root / resident_name).read_text()))
        actual_name = (streamed_name.replace("streamed-", "streamed-vocabulary-", 1) if args.streamed_vocabulary
                       else streamed_name.replace("streamed-", "streamed-block-", 1) if args.streamed_projections
                       else streamed_name.replace("streamed-", "streamed-tables-", 1) if args.streamed_tables
                       else streamed_name.replace("streamed-", "streamed-all-", 1) if args.all_experts
                       else streamed_name)
        streamed = cases(json.loads((args.native_root / actual_name).read_text()))
        routed_only = cases(json.loads((args.native_root / streamed_name).read_text())) if args.all_experts else [None] * len(expected)
        binding = hashlib.sha256(path.read_bytes()).hexdigest()
        block_bytes = None
        if args.streamed_vocabulary:
            block = cases(json.loads((args.native_root / streamed_name.replace("streamed-", "streamed-block-", 1)).read_text()))
            assert len(block) == len(expected)
            block_bytes = []
            for e, old in zip(expected, block):
                assert old["seed"] == e["seed"] and old["revision"] == e["revision"]
                assert old["fixture_sha256"] == binding
                assert old["projection_residency"] == "streamed" and old["resident_projection_layers"] == 0
                assert old.get("resident_vocabulary_tensors", 2) == 2
                block_bytes.append(old["initial_live_buffer_bytes"])
            del block
        trace_path = getattr(args, f"{trace_name}_reference_trace", None) if trace_name else None
        traces = cases(json.loads(trace_path.read_text())) if trace_path else [None] * len(expected)
        assert len(expected) == len(resident) == len(streamed) == len(traces) == len(routed_only)
        for case_index, (e, r, s, trace, prior) in enumerate(zip(expected, resident, streamed, traces, routed_only)):
            assert e["seed"] == r["seed"] == s["seed"]
            streamed_experts = args.all_experts or args.streamed_tables
            assert s["expert_residency"] == ("streamed_all_experts" if streamed_experts else "streamed")
            assert s["resident_routed_experts"] == 0
            if streamed_experts:
                assert s["resident_shared_experts"] == 0
            if args.all_experts:
                assert prior["expert_residency"] == "streamed" and prior["resident_routed_experts"] == 0
                assert prior["seed"] == e["seed"] and prior["fixture_sha256"] == binding
                assert prior["revision"] == e["revision"]
                assert s["initial_live_buffer_bytes"] < prior["initial_live_buffer_bytes"]
            text = e["config"].get("text_config", e["config"])
            if args.streamed_vocabulary:
                assert s["vocabulary_residency"] == "streamed" and s["resident_vocabulary_tensors"] == 0
                elements = text["vocab_size"] * text["hidden_size"]
                vocabulary_bytes = 2 * ((elements + 1) // 2) * 4 if e.get("stream_precision") == "bf16" else 2 * elements * 4
                assert block_bytes[case_index] - s["initial_live_buffer_bytes"] == vocabulary_bytes, "vocabulary residency byte delta differs"
            engram_layers = len(text.get("engram_layer_ids") or [])
            if args.streamed_tables:
                if engram_layers:
                    assert s["engram_residency"] == "streamed_tables"
                    assert s["resident_engram_tables"] == 0
                else:
                    # Graphs without a scheduled table carry no Engram fields.
                    assert s.get("engram_residency") in (None, "streamed_tables")
                    assert s.get("resident_engram_tables", 0) == 0
                if "resident_engram_tables" in r:
                    assert r["resident_engram_tables"] == engram_layers
            if args.streamed_projections:
                assert s["projection_residency"] == "streamed"
                assert s["resident_projection_layers"] == 0
                assert s.get("resident_engram_tables", 0) == 0
                assert s["resident_shared_experts"] == 0
                assert s["resident_routed_experts"] == 0
                if "resident_projection_layers" in r:
                    assert r["resident_projection_layers"] == len(text["compress_ratios"])
            assert r["expert_residency"] == "resident" and r["resident_routed_experts"] > 0
            assert s["initial_live_buffer_bytes"] < r["initial_live_buffer_bytes"]
            if "checkpoint" in e:
                assert s["checkpoint_files"] == r["checkpoint_files"] == e["checkpoint"]["files"]
            model.compare_native(e, s, fixture_sha256=binding)
            assert r["fixture_sha256"] == binding and r["revision"] == s["revision"] == e["revision"]
            assert len(e["steps"]) == len(r["steps"]) == len(s["steps"])
            if prior is not None:
                assert len(prior["steps"]) == len(e["steps"])
            count, worst, routes = 0, 0.0, 0
            for step_index, (es, rs, ss) in enumerate(zip(e["steps"], r["steps"], s["steps"])):
                assert es["name"] == rs["name"] == ss["name"]
                assert es["shape"] == rs["shape"] == ss["shape"]
                x = torch.tensor(ss["logits"], dtype=torch.float32).reshape(es["shape"])
                y = torch.tensor(es["logits"], dtype=torch.float32).reshape(es["shape"])
                old = torch.tensor(rs["logits"], dtype=torch.float32).reshape(es["shape"])
                worst = max(worst, require_logit_parity(x, y, label=f"streamed {e['seed']} {es['name']}"))
                assert torch.equal(x.view(torch.uint32), old.view(torch.uint32)), "resident/streamed logits differ"
                if prior is not None:
                    ps = prior["steps"][step_index]
                    assert ps["name"] == es["name"] and ps["shape"] == es["shape"]
                    previous = torch.tensor(ps["logits"], dtype=torch.float32).reshape(es["shape"])
                    assert torch.equal(x.view(torch.uint32), previous.view(torch.uint32)), "shared staging changed logits"
                count += x.numel()
                if "routing" in es:
                    assert ss["routing"].keys() == rs["routing"].keys() == es["routing"].keys()
                    for layer, route in es["routing"].items():
                        assert canonical_routing(ss["routing"][layer]) == canonical_routing(route)
                        assert ss["routing"][layer] == rs["routing"][layer]
                        routes += len(route["indices"])
            if e.get("stream_precision") == "bf16":
                assert worst == 0
            report = dict(seed=e["seed"], fixture=fixture_name, steps=len(e["steps"]), logits=count,
                          max_abs_drift=worst, bit_exact_with_resident=True, routing_slots=routes,
                          resident_routed_experts=r["resident_routed_experts"], streamed_resident_experts=0,
                          resident_shared_experts=r.get("resident_shared_experts"),
                          streamed_resident_shared_experts=s.get("resident_shared_experts"),
                          resident_engram_tables=r.get("resident_engram_tables"),
                          streamed_resident_engram_tables=s.get("resident_engram_tables"),
                          resident_projection_layers=r.get("resident_projection_layers"),
                          streamed_resident_projection_layers=s.get("resident_projection_layers"),
                          resident_live_tensor_bytes=r["initial_live_buffer_bytes"],
                          streamed_live_tensor_bytes=s["initial_live_buffer_bytes"])
            if args.streamed_vocabulary:
                report["streamed_resident_vocabulary_tensors"] = s["resident_vocabulary_tensors"]
                report["block_streamed_live_tensor_bytes"] = block_bytes[case_index]
                report["vocabulary_live_tensor_bytes_released"] = vocabulary_bytes
            if prior is not None:
                report["streamed_resident_shared_experts"] = s["resident_shared_experts"]
                report["routed_only_live_tensor_bytes"] = prior["initial_live_buffer_bytes"]
                report["bit_exact_with_routed_only"] = True
            if trace is not None:
                assert trace.get("seed", e["seed"]) == e["seed"]
                if trace_name == "long_context":
                    assert len(trace["steps"]) == 1 and trace["steps"][0]["name"] == TRACE_STEP
                    es = next(step for step in e["steps"] if step["name"] == TRACE_STEP)
                    rs = next(step for step in r["steps"] if step["name"] == TRACE_STEP)
                    ss = next(step for step in s["steps"] if step["name"] == TRACE_STEP)
                    diagnostics = compare_traced(e["config"], es, ss, trace["steps"][0])
                    baseline = compare_traced(e["config"], es, rs, trace["steps"][0])
                    assert diagnostics["trace_values"] == diagnostics["bit_exact_values"]
                    paired_steps = [(rs, ss)]
                else:
                    diagnostics = compare_diagnostics(e, s, trace)
                    baseline = compare_diagnostics(e, r, trace)
                    paired_steps = zip(r["steps"], s["steps"])
                assert diagnostics == baseline, "streaming changed CPU trace agreement"
                for rs, ss in paired_steps:
                    old = {(stage["layer"], stage["stage"]): stage["values"] for stage in rs["stages"]}
                    new = {(stage["layer"], stage["stage"]): stage["values"] for stage in ss["stages"]}
                    assert old.keys() == new.keys()
                    for key, values in new.items():
                        x = torch.tensor(values, dtype=torch.float32).view(torch.uint32)
                        y = torch.tensor(old[key], dtype=torch.float32).view(torch.uint32)
                        assert torch.equal(x, y), f"streamed/resident stage differs: {key}"
                report["diagnostics"] = diagnostics
                report["bit_exact_stages_with_resident"] = True
            reports.append(report)
    result = dict(reference="CPU PyTorch", absolute_logit_limit_strict=2e-7, rtol=0,
                  execution_mode=("streamed experts, Engram tables, block projections, embedding and head" if args.streamed_vocabulary else
                                  "streamed routed/shared experts, Engram tables and attention/indexer/compressor projections" if args.streamed_projections else
                                  "streamed routed, shared and Engram-table weights" if args.streamed_tables else
                                  "streamed routed and shared experts" if args.all_experts else "streamed routed experts"),
                  model_executions=len(reports),
                  steps=sum(r["steps"] for r in reports), logits=sum(r["logits"] for r in reports),
                  max_abs_drift=max(r["max_abs_drift"] for r in reports),
                  bit_exact_routing_slots=sum(r["routing_slots"] for r in reports), models=reports,
                  remaining=("Hyper-Connections, norms, gates, caches and Engram projections remain resident; full released geometry unqualified" if args.streamed_vocabulary else
                             "vocabulary and Engram projections remain resident; full released geometry unqualified" if args.streamed_projections else
                             "attention, vocabulary and Engram projections remain resident; full released geometry unqualified" if args.streamed_tables else
                             "attention, vocabulary and Engram tables remain resident; full released geometry unqualified" if args.all_experts else
                             "shared experts, attention, vocabulary and Engram tables remain resident; full released geometry unqualified"))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
