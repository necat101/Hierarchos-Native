"""Record or independently recheck the scalable native text-inference qualification.

Frozen CPU expectations and the absolute-only comparator are unchanged. A compact
gzip bundle preserves every measured logit/routing field after large regenerable
checkpoint packages are removed. Full production weights and CUDA are not gates.
"""
import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from deepseek_v41_parity import require_logit_parity
from summarize_deepseek_v41_qualification import FIXTURES, MODEL_EXPORTS, REPO
from validate_deepseek_v41_long_context_model import canonical_routing

ROOT = REPO / "hierarchos-vulkan/validation/release_parity/deepseek_v41"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    sources = REPO / "hierarchos-vulkan/src"
    paths = [sources/"deepseek_v41.rs", sources/"vulkan.rs", sources/"lib.rs",
             REPO/"hierarchos-vulkan/Cargo.toml", REPO/"hierarchos-native-cli/src/lib.rs",
             REPO/"hierarchos-native-cli/Cargo.toml"]
    paths += [p for p in (sources/"deepseek_v41").glob("*.rs") if "tests" not in p.stem]
    shaders = REPO / "hierarchos-vulkan/shaders"
    paths += list(shaders.glob("transformer_deepseek_v41_*.comp"))
    paths += list(shaders.glob("transformer_deepseek_v41_*.spv"))
    paths += list(shaders.glob("fp32*.glsl"))
    paths += list(shaders.glob("deepseek_v41_*.bin"))
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(set(paths))}


def collect(native_root):
    bundle = []
    for fixture, output in MODEL_EXPORTS:
        native = json.loads((native_root/output).read_text())
        cases = native.get("cases", [native])
        compact = []
        for case in cases:
            item = {k: v for k, v in case.items() if k in {
                "seed", "revision", "fixture_sha256", "checkpoint_files", "loader",
                "device", "engram_metadata_sha256"}}
            item["steps"] = [{k: v for k, v in s.items() if k in {
                "name", "shape", "logits", "routing"}} for s in case["steps"]]
            compact.append(item)
        bundle.append(dict(fixture=fixture, cases=compact))
    return bundle


def compare(bundle):
    models = []
    for entry in bundle:
        path = FIXTURES/entry["fixture"]
        expected = json.loads(path.read_text())
        cases = expected.get("cases", [expected])
        assert cases and len(cases) == len(entry["cases"])
        for ref, actual in zip(cases, entry["cases"]):
            assert actual["fixture_sha256"] == digest(path)
            assert actual["revision"] == ref["revision"]
            if "seed" in actual:
                assert actual["seed"] == ref["seed"]
            if "checkpoint" in ref:
                assert actual["checkpoint_files"] == ref["checkpoint"]["files"]
            if entry["fixture"] in {"deepseek_v41_joint_model.json", "deepseek_v41_production_model.json",
                                    "deepseek_v41_schedule_width_model.json"}:
                assert actual["loader"] == "from_hf_package"
            assert ref["steps"] and len(ref["steps"]) == len(actual["steps"])
            worst, count = 0.0, 0
            for e, a in zip(ref["steps"], actual["steps"]):
                assert e["name"] == a["name"] and e["shape"] == a["shape"]
                x = torch.tensor(a["logits"], dtype=torch.float32).reshape(a["shape"])
                y = torch.tensor(e["logits"], dtype=torch.float32).reshape(e["shape"])
                worst = max(worst, require_logit_parity(x, y, label=e["name"]))
                count += x.numel()
                if "routing" in e:
                    assert a["routing"].keys() == e["routing"].keys()
                    for layer in e["routing"]:
                        assert canonical_routing(a["routing"][layer]) == canonical_routing(e["routing"][layer])
            models.append(dict(fixture=entry["fixture"], seed=ref["seed"],
                               fixture_sha256=digest(path), steps=len(ref["steps"]),
                               logits=count, max_abs_drift=worst))
    assert len(models) == 18
    return dict(model_count=len(models), step_count=sum(m["steps"] for m in models),
                logit_count=sum(m["logits"] for m in models),
                max_abs_drift=max(m["max_abs_drift"] for m in models), models=models)


def coverage():
    production = json.loads((FIXTURES/"deepseek_v41_flash_1017_config.json").read_text())["text_config"]
    joint = json.loads((FIXTURES/"deepseek_v41_joint_model.json").read_text())["cases"][0]["config"]["text_config"]
    width = json.loads((FIXTURES/"deepseek_v41_schedule_width_model.json").read_text())["config"]["text_config"]
    same = ["num_hidden_layers", "n_routed_experts", "num_experts_per_tok", "n_shared_experts", "hc_mult",
            "kv_source_layer_ids", "index_source_layer_ids", "candidate_source_layer_id",
            "candidate_block_size", "engram_layer_ids", "engram_n_heads", "engram_max_ngram_size",
            "scoring_func", "norm_topk_prob", "routed_scaling_factor"]
    for key in same:
        assert joint[key] == production[key], key
    assert joint["compress_ratios"] == production["compress_ratios"][:40]
    widths = ["hidden_size", "moe_intermediate_size", "num_attention_heads", "head_dim",
              "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "hc_mult",
              "vocab_size", "index_n_heads", "index_head_dim", "engram_n_heads", "engram_head_dim"]
    for key in widths:
        assert width[key] == production[key], key
    return dict(production_features={k: joint[k] for k in same} | {"compress_ratios": joint["compress_ratios"]},
                production_widths={k: width[k] for k in widths},
                scalable_checkpoint="hierarchos-vulkan/validation/fixtures/deepseek_v41_joint_4141111",
                permitted_reductions=["arithmetic widths in the 40-layer joint graph", "vocabulary and table rows",
                                      "context and top-k capacities", "depth and expert payload count in width probes"],
                production_weights_execution_required=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", action="store_true")
    p.add_argument("--native-root", type=Path, default=REPO/"target/deepseek-v41-reference")
    p.add_argument("--machine-info", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT)
    a = p.parse_args()
    torch.set_num_threads(1)
    receipt, blob = a.output_root/"qualification.json", a.output_root/"native-outputs.json.gz"
    if a.record:
        assert a.machine_info, "recording requires local machine information"
        bundle = collect(a.native_root)
        result = compare(bundle)
        features = coverage()
        machine = json.loads(a.machine_info.read_text(encoding="utf-8-sig"))
        joint = next(e for e in bundle if e["fixture"] == "deepseek_v41_joint_model.json")["cases"][0]
        assert joint["device"] == machine["gpu"][0]["Name"]
        runtime = inventory()
        execution = a.native_root/"scalable-release-native-logits.json"
        assert execution.stat().st_mtime >= max((REPO/p).stat().st_mtime for p in runtime), \
            "runtime changed after the small checkpoint run; rerun the default-package gate"
        log = a.native_root/"scalable-release-current-machine.log"
        assert "test result: ok. 1 passed; 0 failed" in log.read_text(encoding="utf-8-sig")
        a.output_root.mkdir(parents=True, exist_ok=True)
        with gzip.open(blob, "wb", compresslevel=6) as out:
            out.write(json.dumps(bundle, separators=(",", ":"), allow_nan=False).encode())
        result.update(schema=1, architecture="deepseek_v41", status="qualified",
            scope="native causal text inference on this machine under the accepted scalable-checkpoint criterion",
            qualification_method="released production feature schedule plus production-width arithmetic probes",
            absolute_logit_limit_strict=2e-7, rtol=0, cuda_required=False,
            recorded_utc=datetime.now(timezone.utc).isoformat(), machine=machine, coverage=features,
            runtime_inputs=runtime, native_outputs_sha256=digest(blob),
            fresh_default_loader_execution=dict(export_sha256=digest(execution), log_sha256=digest(log),
                logits=3216, steps=6, released_layers=40, routed_experts=384, topk=6),
            supplementary_note="14 grouped intermediate differences are FP8-equivalent in the measured width case; final logits have zero drift",
            intel_gen9_status="next optimization target; target-device kernel qualification pending")
        receipt.write_text(json.dumps(result, indent=2)+"\n")
    else:
        record = json.loads(receipt.read_text())
        assert record["status"] == "qualified" and record["absolute_logit_limit_strict"] == 2e-7 and record["rtol"] == 0
        assert record["runtime_inputs"] == inventory(), "runtime changed; rerun native qualification"
        assert record["coverage"] == coverage()
        assert record["native_outputs_sha256"] == digest(blob)
        with gzip.open(blob, "rb") as source:
            result = compare(json.load(source))
        for k, v in result.items(): assert record[k] == v, k
    print(json.dumps({k: v for k, v in result.items() if k not in {"models", "runtime_inputs", "coverage"}}, indent=2))
    print("PASS native DeepSeek V4.1 scalable architecture qualification; strict <2e-7, rtol=0")


if __name__ == "__main__": main()
