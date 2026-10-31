"""Qualify released text depth/schedules and routing cardinality independently.

Fixed before any native run: one 40-layer, eight-expert graph with the released
compression/KV/index/Engram layer schedule, and one 384-expert, top-6 graph.
Widths, vocabulary, table sizes and context are reduced. These two probes do
not qualify the combined full-size model. The pinned forward bodies and CPU
kernel mirrors are unchanged; acceptance compares every logit at <2e-7, rtol=0.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import validate_deepseek_v41_bf16_model as bf16
import validate_deepseek_v41_model as model
import validate_deepseek_v41_wide_model as wide

FIXTURE = model.FIXTURE.parent / "deepseek_v41_structure_model.json"
SPECS = [
    dict(seed=4141101, layers=40, experts=8, engram=True),
    dict(seed=4141102, layers=1, experts=384, engram=False),
]
SCHEDULE = [
    ("schedule-odd-prefill", 0, 17),
    ("schedule-group-decode", 17, 1),
    ("schedule-incomplete-decode", 18, 1),
    ("schedule-group-decode-2", 19, 1),
    ("schedule-reset", 0, 3),
    ("schedule-reset-decode", 3, 1),
]


class StructureObserver(bf16.Bf16Observer):
    """Capture routing without changing scores, indices or expert execution."""
    def attach(self, net):
        super().attach(net)
        for layer in net.layers:
            def capture(module, args, output, *, layer_id=layer.layer_id):
                weights, indices = output
                self.routes[str(layer_id)] = dict(
                    indices=indices.flatten().tolist(),
                    weight_bits=weights.contiguous().view(torch.uint32).flatten().tolist(),
                    counts=torch.bincount(indices.flatten(), minlength=module.weight.shape[0]).tolist(),
                )
            layer.ffn.gate.register_forward_hook(capture)

    def begin(self, spec):
        super().begin(spec)
        self.routes = {}

    def end(self, spec):
        super().end(spec)
        self.steps[-1]["routing"] = self.routes


def make_case(path, kernel_path, spec, observer):
    released = json.loads((model.FIXTURE.parent / "deepseek_v41_flash_1017_config.json").read_text())["text_config"]
    depth = spec["layers"] == released["num_hidden_layers"]
    ratios = released["compress_ratios"][:40] if depth else [0]
    kv_sources = released["kv_source_layer_ids"] if depth else []
    index_sources = released["index_source_layer_ids"] if depth else []
    engram_layers = released["engram_layer_ids"] if spec["engram"] else []
    tables = [8192] * len(engram_layers)
    steps, _ = wide.inputs(67, SCHEDULE)
    tokens = dict(token_map=[i % 16 for i in range(67)],
                  multipliers=[[3,5,7,11],[13,17,19,23]] if engram_layers else [],
                  pad_token_id=2)
    hidden, inter = spec.get("hidden", 64), spec.get("inter", 64)
    args = dict(dim=hidden, n_layers=spec["layers"], n_routed_experts=spec["experts"],
                n_heads=4, head_dim=64, rope_head_dim=64, q_lora_rank=32,
                o_groups=2, o_lora_rank=32, moe_inter_dim=inter, vocab_size=67,
                window_size=8, max_seq_len=64, original_seq_len=8, rope_factor=16,
                compress_ratios=tuple(ratios), kv_source_layers=tuple(kv_sources),
                index_source_layers=tuple(index_sources), index_n_heads=4,
                index_head_dim=128, index_topk=4, candidate_source_layer=20 if depth else -1,
                candidate_topk_blocks=2 if depth else 0, candidate_block_size=8 if depth else 0,
                swiglu_limit=10.0, engram_layer_ids=tuple(engram_layers),
                engram_num_embeddings=tuple(tables), engram_max_ngram_size=4,
                engram_vocab_size=17, engram_n_heads=8, engram_head_dim=32,
                engram_compressed_vocab_size=16)
    config = dict(hidden_size=hidden, num_hidden_layers=spec["layers"],
                  n_routed_experts=spec["experts"], num_attention_heads=4,
                  head_dim=64, qk_rope_head_dim=64, q_lora_rank=32, o_groups=2,
                  o_lora_rank=32, moe_intermediate_size=inter, vocab_size=67,
                  sliding_window=8, max_position_embeddings=64, compress_ratios=ratios,
                  kv_source_layer_ids=kv_sources, index_source_layer_ids=index_sources,
                  index_n_heads=4, index_head_dim=128, index_topk=4,
                  candidate_source_layer_id=20 if depth else -1,
                  candidate_topk_blocks=2 if depth else 0, candidate_block_size=8 if depth else 0,
                  swiglu_limit=10.0, engram_layer_ids=engram_layers,
                  engram_num_embeddings=tables, engram_max_ngram_size=4,
                  engram_vocab_size=17, engram_n_heads=8, engram_head_dim=32,
                  engram_compressed_vocab_size=16,
                  rope_scaling=dict(rope_type="yarn", factor=16, beta_fast=32,
                                    beta_slow=1, original_max_position_embeddings=8))
    return bf16.make_case(path, kernel_path, spec["seed"], 6, 4,
                          args_overrides=args, config_overrides=config,
                          observer=observer, steps=steps, tokens=tokens, max_prefill=17)


def compact(fixture, write, expected_files=None, *, prefix="deepseek_v41_structure"):
    files = wide.checkpoint_bytes(fixture, True)
    digests = {name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    if expected_files is not None and digests != expected_files:
        raise AssertionError("CPU checkpoint bytes differ from the frozen package hashes")
    folder = f"{prefix}_{fixture['seed']}"
    root = FIXTURE.parent / folder
    if write:
        root.mkdir(exist_ok=True)
    for name, data in files.items():
        if write:
            (root / name).write_bytes(data)
        elif (root / name).read_bytes() != data:
            raise AssertionError(f"frozen checkpoint differs from CPU weights: {root/name}")
    fixture["checkpoint"] = dict(directory=folder,
        files=digests,
        tensor_count=len(fixture["tensors"]), packed_count=len(fixture["packed_tensors"]))
    fixture["tensors"] = {k:v for k,v in fixture["tensors"].items() if k in {"norm.weight","head.weight"}}
    del fixture["packed_tensors"]
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--write-checkpoints", action="store_true",
                        help="rebuild local shard payloads only after matching frozen hashes; never writes expectations")
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--observation-output", type=Path,
                        help="independent CPU replay diagnostic; never replaces the frozen fixture")
    args = parser.parse_args()
    if args.write_fixture and FIXTURE.exists():
        raise FileExistsError("structure fixture is already frozen; repair native execution")
    frozen = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None
    if args.write_checkpoints and frozen is None:
        raise FileNotFoundError("checkpoint reconstruction requires an existing frozen CPU fixture")
    cases, traces = [], []
    for case_index,spec in enumerate(SPECS):
        print(f"Generating fixed CPU structure case {spec}", flush=True)
        observer = StructureObserver()
        expected_files = frozen["cases"][case_index]["checkpoint"]["files"] if frozen else None
        fixture = compact(make_case(args.reference_path, args.kernel_reference_path, spec, observer),
                          args.write_fixture or args.write_checkpoints, expected_files)
        # Routing is frozen alongside all logits, including unused-expert counts.
        for step, trace in zip(fixture["steps"], observer.steps):
            step["routing"] = trace["routing"]
        cases.append(fixture)
        traces.append(dict(seed=spec["seed"], steps=observer.steps))
        print(f"CPU seed {spec['seed']}: {sum(len(s['logits']) for s in fixture['steps'])} complete logits", flush=True)
    result = dict(torch_version=torch.__version__, torch_revision=torch.version.git_version, cases=cases)
    if args.observation_output:
        args.observation_output.write_text(json.dumps(result, separators=(",",":"), allow_nan=False))
    if args.trace_output:
        args.trace_output.write_text(json.dumps(dict(cases=traces), separators=(",",":"), allow_nan=False))
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(result, separators=(",",":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != result:
        raise AssertionError("structure fixture differs from the pinned CPU reference")
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        assert len(native["cases"]) == len(cases)
        binding = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        for expected, actual in zip(cases, native["cases"]):
            assert expected["seed"] == actual["seed"]
            assert expected["checkpoint"]["files"] == actual["checkpoint_files"]
            model.compare_native(expected, actual, fixture_sha256=binding)
            for step, trace in zip(actual["steps"], expected["steps"]):
                assert step["routing"] == trace["routing"], "routing weights, indices or counts differ"
    print("PASS structure CPU oracle, checkpoint provenance and all logits; full-size joint graph remains unqualified")


if __name__ == "__main__":
    main()
