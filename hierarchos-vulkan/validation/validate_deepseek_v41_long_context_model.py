"""Frozen 40-layer / 384-expert CPU PyTorch 2,048-token qualification.

The released compression, KV/index sharing and Engram schedules run together
with 384 routed experts and top-6 selection, as in the joint probe. The prompt
is one **single** 2,048-token start-zero prefill at the released 32 index heads,
followed by decodes that read the resulting 2,048-position state and a short
reset sequence. Hidden/intermediate widths, vocabulary, ranks and Engram tables
remain reduced; the combined released-width graph is out of local memory scope
and is documented in the audit.

A single pass is what the pinned reference defines: it prefill-seeds the sliding
window ring only at `start_pos == 0` and its decode branch consumes one token,
so a prompt chunk after the first has no reference semantics to compare
against. The prompt is therefore kept whole, and the released `2,048 x 32 =
65,536` per-head query rows are dispatched by tiling per-head-row kernels
(rotary application and activation quantization) on the grid's z axis instead
of capping the prompt at 2,047 tokens.

The deep trace is the first decode after the prompt, so its stage values, shared
caches and index scores are the completed 2,048-position state and are compared
without slicing. Selected compressed positions carry the window-buffer offset,
which differs between a 2,048-row prefill window and the decode ring, so both
sides are normalized by their declared offset before comparison.

Every complete logit must still satisfy <2e-7, rtol=0.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import validate_deepseek_v41_bf16_model as bf16
import validate_deepseek_v41_model as model
import validate_deepseek_v41_structure_model as structure
import validate_deepseek_v41_wide_model as wide

FIXTURE = model.FIXTURE.parent / "deepseek_v41_long_context_model.json"
SPEC = dict(seed=4141112, layers=40, experts=384, engram=True, hidden=32, inter=32)
WINDOW = 128
PROMPT = 2048
TRACE_STEP = "long-decode-2049"
# Fixture steps: one start-zero prefill, decodes that read its caches, a reset.
SCHEDULE = [
    ("long-prefill-2048", 0, PROMPT),
    (TRACE_STEP, PROMPT, 1),
    ("long-decode-2050", PROMPT + 1, 1),
    ("long-decode-2051", PROMPT + 2, 1),
    ("long-reset", 0, 8),
    ("long-reset-decode", 8, 1),
    ("long-reset-decode-2", 9, 1),
]


def tokens_at(vocab, length):
    """Position-indexed token stream; every step slices the same prompt prefix."""
    return [((position + 1) * 7 + 3) % vocab for position in range(length)]


def steps_for(vocab, length):
    mask = [int((position + 1) % 11 != 0) for position in range(length)]
    ids = tokens_at(vocab, length)
    return [dict(name=name, pos=pos, seq=seq,
                 ids=ids[pos:pos + seq], mask=mask[pos:pos + seq])
            for name, pos, seq in SCHEDULE]


def make_case(path, kernel_path, spec, observer):
    released = json.loads(
        (model.FIXTURE.parent / "deepseek_v41_flash_1017_config.json").read_text()
    )["text_config"]
    ratios = released["compress_ratios"][:40]
    kv_sources = released["kv_source_layer_ids"]
    index_sources = released["index_source_layer_ids"]
    engram_layers = released["engram_layer_ids"] if spec["engram"] else []
    tables = [8192] * len(engram_layers)
    steps = steps_for(67, PROMPT + 8)
    tokens = dict(token_map=[i % 16 for i in range(67)],
                  multipliers=[[3, 5, 7, 11], [13, 17, 19, 23]], pad_token_id=2)
    hidden, inter = spec["hidden"], spec["inter"]
    args = dict(dim=hidden, n_layers=spec["layers"],
                n_routed_experts=spec["experts"], n_heads=4, head_dim=64,
                rope_head_dim=64, q_lora_rank=32, o_groups=2, o_lora_rank=32,
                moe_inter_dim=inter, vocab_size=67, window_size=WINDOW,
                max_seq_len=2064, original_seq_len=128, rope_factor=16,
                compress_ratios=tuple(ratios),
                kv_source_layers=tuple(kv_sources),
                index_source_layers=tuple(index_sources), index_n_heads=32,
                index_head_dim=128, index_topk=16,
                candidate_source_layer=20, candidate_topk_blocks=8,
                candidate_block_size=8, swiglu_limit=10.0,
                engram_layer_ids=tuple(engram_layers),
                engram_num_embeddings=tuple(tables), engram_max_ngram_size=4,
                engram_vocab_size=17, engram_n_heads=8, engram_head_dim=32,
                engram_compressed_vocab_size=16, max_batch_size=1)
    config = dict(hidden_size=hidden, num_hidden_layers=spec["layers"],
                  n_routed_experts=spec["experts"], num_attention_heads=4,
                  head_dim=64, qk_rope_head_dim=64, q_lora_rank=32,
                  o_groups=2, o_lora_rank=32, moe_intermediate_size=inter,
                  vocab_size=67, sliding_window=WINDOW,
                  max_position_embeddings=2064, compress_ratios=ratios,
                  kv_source_layer_ids=kv_sources,
                  index_source_layer_ids=index_sources, index_n_heads=32,
                  index_head_dim=128, index_topk=16,
                  candidate_source_layer_id=20, candidate_topk_blocks=8,
                  candidate_block_size=8, swiglu_limit=10.0,
                  engram_layer_ids=engram_layers,
                  engram_num_embeddings=tables, engram_max_ngram_size=4,
                  engram_vocab_size=17, engram_n_heads=8, engram_head_dim=32,
                  engram_compressed_vocab_size=16,
                  rope_scaling=dict(rope_type="yarn", factor=16, beta_fast=32,
                                    beta_slow=1,
                                    original_max_position_embeddings=128))
    return bf16.make_case(path, kernel_path, spec["seed"], 6, 4,
                          args_overrides=args, config_overrides=config,
                          observer=observer, steps=steps, tokens=tokens,
                          max_prefill=PROMPT)


def canonical_routing(route, topk=6):
    """PyTorch leaves top-k tie order unspecified; compare by expert id."""
    indices, bits = route["indices"], route["weight_bits"]
    assert len(indices) == len(bits) and len(indices) % topk == 0
    rows = []
    for start in range(0, len(indices), topk):
        pairs = sorted(zip(indices[start:start + topk], bits[start:start + topk]))
        assert len({pair[0] for pair in pairs}) == topk
        rows.append(pairs)
    return route["counts"], rows


def shift(indices, offset):
    return [index - offset if index >= 0 else index for index in indices]


def compare_traced(config, fixture_step, native_step, reference_step):
    """Bit-exact comparison of the bounded deep-context stage evidence."""
    expected = {(s["layer"], s["stage"]): s["values"]
                for s in reference_step["stages"]}
    actual = {(s["layer"], s["stage"]): s["values"]
              for s in native_step.get("stages", [])}
    assert actual.keys() == expected.keys(), "complete deep-context trace required"
    total = exact = 0
    differences = {}
    for key, values in actual.items():
        x = torch.tensor(values, dtype=torch.float32)
        y = torch.tensor(expected[key], dtype=torch.float32)
        assert x.numel() and x.shape == y.shape
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        equal = x.view(torch.uint32) == y.view(torch.uint32)
        total += x.numel()
        exact += int(equal.sum())
        if not equal.all():
            entry = differences.setdefault(key[1], dict(differing_values=0, max_abs_drift=0.0))
            entry["differing_values"] += int((~equal).sum())
            entry["max_abs_drift"] = max(
                entry["max_abs_drift"],
                float((x.double() - y.double()).abs().max()),
            )
    assert native_step["cache_diagnostics"] == {
        str(k): v for k, v in reference_step["cache_diagnostics"].items()
    }, "long-context shared cache bits differ"
    text = config["text_config"]
    owners = text["index_source_layer_ids"]
    ratios = text["compress_ratios"]
    positions = fixture_step["pos"] + fixture_step["seq"]
    expected_indices = {str(k): v for k, v in reference_step["index_diagnostics"].items()
                        if v["indices"]}
    layers = {str(i) for i, ratio in enumerate(ratios)
              if ratio and positions // ratio}
    assert native_step["index_diagnostics"].keys() == layers, \
        "complete long-context index diagnostics required"
    native_offset = native_step["index_offset"]
    reference_offset = reference_step["index_offset"]
    for layer, selection in native_step["index_diagnostics"].items():
        owner = max(i for i in owners if i <= int(layer))
        expected = expected_indices[str(owner)]
        assert selection["score_bits"] == expected["score_bits"], \
            "long-context index scores differ"
        assert shift(selection["indices"], native_offset) == shift(
            expected["indices"], reference_offset), \
            "long-context selected compressed positions differ"
    # The CPU observer stores owning indexers only. Every native shared
    # consumer was checked above; count those fields as well as owner fields.
    owner_score_values = sum(len(v["score_bits"])
                             for v in reference_step["index_diagnostics"].values())
    owner_index_values = sum(len(v["indices"])
                             for v in reference_step["index_diagnostics"].values())
    score_values = sum(len(v["score_bits"])
                       for v in native_step["index_diagnostics"].values())
    index_values = sum(len(v["indices"])
                       for v in native_step["index_diagnostics"].values())
    report = dict(step=TRACE_STEP, trace_values=total, bit_exact_values=exact,
                  nonexact_stages=differences,
                  bit_exact_cache_values=sum(len(v["kv"]) + len(v["keys"])
                                             for v in reference_step["cache_diagnostics"].values()),
                  bit_exact_index_scores_including_shared_consumers=score_values,
                  bit_exact_indices_including_shared_consumers=index_values,
                  bit_exact_owner_index_scores=owner_score_values,
                  bit_exact_owner_selected_indices=owner_index_values)
    print("TRACE " + json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--write-checkpoints", action="store_true",
                        help="rebuild local shards only after matching frozen hashes")
    parser.add_argument("--refresh-fixture", action="store_true",
                        help="rewrite expectations for an extended schedule/context "
                             "only after matching every frozen checkpoint hash")
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    if sum([args.write_fixture, args.write_checkpoints, args.refresh_fixture]) > 1:
        raise ValueError("choose one fixture/checkpoint write mode")
    if args.write_fixture and FIXTURE.exists():
        raise FileExistsError("long-context fixture already frozen; repair native execution")
    if args.refresh_fixture and not FIXTURE.exists():
        raise FileNotFoundError("fixture refresh requires the previous frozen CPU fixture")
    frozen = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None
    if frozen is None and not args.write_fixture:
        raise FileNotFoundError("initial CPU freeze required before native qualification")
    if args.refresh_fixture:
        print("Refreshing expectations after verifying the frozen checkpoint shards", flush=True)
    print(f"Generating fixed long-context CPU graph {SPEC}", flush=True)
    observer = structure.StructureObserver()
    expected_files = frozen["cases"][0]["checkpoint"]["files"] if frozen else None
    fixture = make_case(args.reference_path, args.kernel_reference_path, SPEC, observer)
    if args.refresh_fixture:
        # A longer single prefill must not move a byte of weight data.
        previous = frozen["cases"][0]["checkpoint"]["files"]
        fresh = {name: hashlib.sha256(data).hexdigest()
                 for name, data in wide.checkpoint_bytes(fixture, True).items()}
        assert fresh.keys() == previous.keys()
        changed = sorted(name for name in previous if previous[name] != fresh[name])
        assert changed in ([], ["config.json"]), \
            f"only declared context metadata may change, got {changed}"
        expected_files = dict(previous, **{name: fresh[name] for name in changed})
        print(f"Preserved {len(previous) - len(changed)} weight/index hashes; "
              f"changed {changed or 'nothing'}", flush=True)
    for step, trace in zip(fixture["steps"], observer.steps):
        step["routing"] = trace["routing"]
    case = structure.compact(
        fixture, args.write_fixture or args.write_checkpoints or args.refresh_fixture,
        expected_files, prefix="deepseek_v41_long_context")
    result = dict(torch_version=torch.__version__,
                  torch_revision=torch.version.git_version, cases=[case])
    if args.trace_output:
        traced = [s for s in observer.steps if s["name"] == TRACE_STEP]
        args.trace_output.write_text(json.dumps(
            dict(cases=[dict(seed=SPEC["seed"], steps=traced)]),
            separators=(",", ":"), allow_nan=False))
    if args.write_fixture or args.refresh_fixture:
        FIXTURE.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    elif result != frozen:
        raise AssertionError("long-context fixture differs from the pinned CPU reference")
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        assert len(native["cases"]) == 1
        actual = native["cases"][0]
        assert actual["seed"] == SPEC["seed"]
        assert actual["checkpoint_files"] == case["checkpoint"]["files"]
        model.compare_native(case, actual, fixture_sha256=hashlib.sha256(
            FIXTURE.read_bytes()).hexdigest())
        assert len(actual["steps"]) == len(case["steps"])
        for step, expected in zip(actual["steps"], case["steps"]):
            assert step["routing"].keys() == expected["routing"].keys()
            for layer, route in step["routing"].items():
                assert canonical_routing(route) == canonical_routing(
                    expected["routing"][layer]), \
                    "long-context routing experts, weights or counts differ"
        assert args.trace_output, "deep-context trace is required for stage evidence"
        trace = json.loads(args.trace_output.read_text())["cases"][0]["steps"]
        assert len(trace) == 1 and trace[0]["name"] == TRACE_STEP
        native_step = next(s for s in actual["steps"] if s["name"] == TRACE_STEP)
        fixture_step = next(s for s in case["steps"] if s["name"] == TRACE_STEP)
        compare_traced(case["config"], fixture_step, native_step, trace[0])
    print(f"PASS long-context CPU oracle: {sum(len(s['logits']) for s in case['steps'])} "
          "complete logits from one 2,048-token prefill; combined released widths remain unqualified",
          flush=True)


if __name__ == "__main__":
    main()
