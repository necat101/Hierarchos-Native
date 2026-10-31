"""Frozen wider BF16 CPU PyTorch qualification with compact checkpoint payloads.

Seeds, dimensions, inputs and initialization are declared before native drift is
measured. The unchanged pinned model runs with the existing CPU kernel mirrors.
Every output position and vocabulary logit must pass <2e-7, rtol=0. These cases
increase context and width; they do not qualify the complete production model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import safetensors
import torch

import validate_deepseek_v41_bf16_model as bf16
import validate_deepseek_v41_model as model

FIXTURE = model.FIXTURE.parent / "deepseek_v41_wide_model.json"
# Fixed before the first native run. Keep failing cases: repair the native path.
SPECS = [
    dict(seed=4141071, topk=3, hidden=128, head_dim=128, qrank=128, orank=64,
         inter=128, index_heads=4, vocab=97, original_context=16, sharded=False),
    dict(seed=4141072, topk=2, hidden=256, head_dim=512, qrank=256, orank=128,
         inter=256, index_heads=32, vocab=129, original_context=32, sharded=True),
    # Long-context continuation: 512-token YaRN prefill with a 64-slot window
    # (eight ring wraps), 128 candidate blocks at ratio 1 and 64 at ratio 2,
    # four decodes across compression groups, then an odd 129-token reset.
    dict(seed=4141073, topk=3, hidden=128, head_dim=128, qrank=128, orank=64,
         inter=128, index_heads=4, vocab=97, original_context=32, sharded=True,
         window=64, max_seq=768, max_prefill=512, block_size=4,
         schedule=[("long-yarn-prefill", 0, 512), ("long-group-decode", 512, 1),
                   ("long-incomplete-decode", 513, 1), ("long-group-decode-2", 514, 1),
                   ("long-incomplete-decode-2", 515, 1), ("odd-prefill-reset", 0, 129),
                   ("odd-prefill-decode", 129, 1), ("odd-prefill-decode-2", 130, 1)]),
]
DEFAULT_SCHEDULE = [("yarn-odd-prefill", 0, 65), ("group-completing-decode", 65, 1),
                    ("incomplete-group-decode", 66, 1), ("group-completing-decode-2", 67, 1),
                    ("incomplete-group-decode-2", 68, 1), ("shorter-reset", 0, 3),
                    ("short-prefill-group-completion", 3, 1)]


def inputs(vocab, schedule=None):
    schedule = schedule or DEFAULT_SCHEDULE
    steps = []
    for i, (name, pos, seq) in enumerate(schedule):
        ids = [(b*7 + (pos+s)*3 + i*11) % vocab for b in range(2) for s in range(seq)]
        mask = [int((b*seq+s+i) % 11 != 0) for b in range(2) for s in range(seq)]
        steps.append(dict(name=name, pos=pos, seq=seq, ids=ids, mask=mask))
    tokens = dict(token_map=[i % 8 for i in range(vocab)],
                  multipliers=[[3,5,7],[11,13,17]], pad_token_id=2)
    return steps, tokens


def make_case(path, kernel_path, spec, observer=None):
    steps, tokens = inputs(spec["vocab"], spec.get("schedule"))
    window, max_seq = spec.get("window", 16), spec.get("max_seq", 128)
    block_size = spec.get("block_size", 4)
    args = dict(dim=spec["hidden"], head_dim=spec["head_dim"], rope_head_dim=64,
                n_heads=4, o_groups=2, q_lora_rank=spec["qrank"], o_lora_rank=spec["orank"],
                moe_inter_dim=spec["inter"], index_n_heads=spec["index_heads"], index_head_dim=128,
                vocab_size=spec["vocab"], window_size=window, max_seq_len=max_seq,
                original_seq_len=spec["original_context"], rope_factor=4,
                index_topk=8, candidate_topk_blocks=2, candidate_block_size=block_size)
    cfg = dict(hidden_size=spec["hidden"], head_dim=spec["head_dim"], qk_rope_head_dim=64,
               num_attention_heads=4, o_groups=2, q_lora_rank=spec["qrank"], o_lora_rank=spec["orank"],
               moe_intermediate_size=spec["inter"], index_n_heads=spec["index_heads"], index_head_dim=128,
               vocab_size=spec["vocab"], sliding_window=window, max_position_embeddings=max_seq,
               index_topk=8, candidate_topk_blocks=2, candidate_block_size=block_size,
               rope_scaling=dict(rope_type="yarn", factor=4, beta_fast=32, beta_slow=1,
                                 original_max_position_embeddings=spec["original_context"]))
    return bf16.make_case(path, kernel_path, spec["seed"], spec["topk"], 4, observer=observer,
                          args_overrides=args, config_overrides=cfg, steps=steps, tokens=tokens,
                          max_prefill=spec.get("max_prefill", 65))


def checkpoint_bytes(fixture, sharded):
    """Serialize canonical logical shapes; retain storage until serialization ends."""
    buffers, specs = {}, {}
    def add(name, dtype, shape, values):
        buffers[name] = values.contiguous()
        specs[name] = safetensors.TensorSpec(dtype=dtype, shape=shape,
            data_ptr=buffers[name].data_ptr(), data_len=buffers[name].numel()*buffers[name].element_size())
    for name, tensor in fixture["tensors"].items():
        dtype = torch.bfloat16 if tensor.get("dtype") == "bf16" else torch.float32
        add(name, str(dtype).removeprefix("torch."), tensor["shape"], torch.tensor(tensor["values"], dtype=dtype))
    for name, tensor in fixture["packed_tensors"].items():
        fp4 = tensor["dtype"] == "fp4_row32"
        rows, cols = tensor["rows"], tensor["cols"]
        add(name, "float4_e2m1fn_x2" if fp4 else "float8_e4m3fn", [rows, cols//2 if fp4 else cols],
            torch.tensor(tensor["data"], dtype=torch.uint8))
        scale_rows = rows if fp4 or tensor["dtype"] == "engram_fp8_row32" else (rows+31)//32
        add(name.removesuffix("weight")+"scale", "float8_e8m0fnu", [scale_rows, cols//32],
            torch.tensor(tensor["scales"], dtype=torch.uint8))
    files = {"config.json": json.dumps(fixture["config"], separators=(",", ":")).encode()}
    weight_map = {}
    for shard in range(2 if sharded else 1):
        name = f"weights-{shard}.safetensors" if sharded else "model.safetensors"
        selected = {k:specs[k] for i,k in enumerate(sorted(specs)) if not sharded or i % 2 == shard}
        files[name] = safetensors.serialize(selected)
        weight_map.update({k:name for k in selected})
    if sharded:
        files["model.safetensors.index.json"] = json.dumps(dict(weight_map=weight_map), separators=(",", ":")).encode()
    return files


def compact(fixture, spec, write):
    files = checkpoint_bytes(fixture, spec["sharded"])
    folder = f"deepseek_v41_wide_{spec['seed']}"
    root = FIXTURE.parent / folder
    if write:
        root.mkdir(exist_ok=True)
    for name, data in files.items():
        if write:
            (root / name).write_bytes(data)
        elif (root / name).read_bytes() != data:
            raise AssertionError(f"frozen checkpoint differs from pinned CPU weights: {root/name}")
    fixture["checkpoint"] = dict(directory=folder,
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        tensor_count=len(fixture["tensors"]), packed_count=len(fixture["packed_tensors"]))
    # Retain only the small independent norm/head replay inputs in JSON.
    fixture["tensors"] = {k:v for k,v in fixture["tensors"].items() if k in {"head.weight", "norm.weight"}}
    del fixture["packed_tensors"]
    return fixture


def compare_diagnostics(fixture, native, reference):
    """Observe intermediate differences separately from strict logit acceptance."""
    assert len(native["steps"]) == len(reference["steps"]) == len(fixture["steps"])
    total = exact = cache_count = score_count = index_count = 0
    differences = {}
    for actual, expected, spec in zip(native["steps"], reference["steps"], fixture["steps"]):
        assert actual["name"] == expected["name"] == spec["name"]
        ref = {(s["layer"],s["stage"]):s["values"] for s in expected["stages"]}
        assert {(s["layer"],s["stage"]) for s in actual["stages"]} == ref.keys()
        for stage in actual["stages"]:
            x = torch.tensor(stage["values"],dtype=torch.float32)
            y = torch.tensor(ref[(stage["layer"],stage["stage"])],dtype=torch.float32)
            assert x.numel() and x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
            equal = x.view(torch.uint32) == y.view(torch.uint32)
            total += x.numel()
            exact += int(equal.sum())
            if not equal.all():
                entry = differences.setdefault(stage["stage"],dict(differing_values=0,max_abs_drift=0.0))
                entry["differing_values"] += int((~equal).sum())
                entry["max_abs_drift"] = max(entry["max_abs_drift"],float((x.double()-y.double()).abs().max()))
        caches = {str(k):v for k,v in expected["cache_diagnostics"].items()}
        assert actual["cache_diagnostics"] == caches, "shared cache bits differ"
        cache_count += sum(len(v["kv"])+len(v["keys"]) for v in caches.values())
        owners = fixture["config"]["text_config"]["index_source_layer_ids"]
        expected_indices = {str(k):v for k,v in expected["index_diagnostics"].items() if v["indices"]}
        ratios = fixture["config"]["text_config"]["compress_ratios"]
        layers = {str(i) for i,ratio in enumerate(ratios) if ratio and (spec["pos"]+spec["seq"])//ratio}
        assert actual["index_diagnostics"].keys() == layers, "complete index diagnostics required"
        for layer, selection in actual["index_diagnostics"].items():
            owner = max(i for i in owners if i <= int(layer))
            assert selection == expected_indices[str(owner)], "index scores or selected positions differ"
            score_count += len(selection["score_bits"])
            index_count += len(selection["indices"])
        assert set(expected_indices) <= actual["index_diagnostics"].keys()
    report = dict(seed=fixture["seed"],trace_values=total,bit_exact_values=exact,
                  nonexact_stages=differences,bit_exact_cache_values=cache_count,
                  bit_exact_index_scores_including_shared_consumers=score_count,
                  bit_exact_indices_including_shared_consumers=index_count)
    print("TRACE " + json.dumps(report,sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    cases, traces = [], []
    for spec in SPECS:
        print(f"Generating pinned CPU case {spec}", flush=True)
        observer = bf16.Bf16Observer() if args.trace_output else None
        f = make_case(args.reference_path, args.kernel_reference_path, spec, observer)
        cases.append(compact(f, spec, args.write_fixture))
        if observer:
            traces.append(dict(seed=spec["seed"], steps=observer.steps))
        print(f"CPU seed {spec['seed']}: {sum(len(s['logits']) for s in f['steps'])} complete logits", flush=True)
    fixture = dict(torch_version=torch.__version__, torch_revision=torch.version.git_version, cases=cases)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    elif json.loads(FIXTURE.read_text()) != fixture:
        raise AssertionError("wider fixture differs from pinned CPU PyTorch")
    if args.trace_output:
        args.trace_output.write_text(json.dumps(dict(cases=traces), separators=(",", ":"), allow_nan=False))
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        assert len(native["cases"]) == len(cases)
        binding = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        for i, (expected, actual) in enumerate(zip(cases, native["cases"])):
            assert expected["seed"] == actual["seed"]
            assert expected["checkpoint"]["files"] == actual["checkpoint_files"]
            model.compare_native(expected, actual, fixture_sha256=binding)
            if traces:
                compare_diagnostics(expected, actual, traces[i])
    print("PASS wider CPU oracle and checkpoint provenance; production geometry remains unqualified")


if __name__ == "__main__":
    main()
