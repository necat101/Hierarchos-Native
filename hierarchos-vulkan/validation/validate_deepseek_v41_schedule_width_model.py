"""Separate production-width schedule/top-6 CPU oracle; the 24-layer gate is immutable.

Four blocks exercise ratio-2 pooling, shared KV/index selections, ratio-1
compression, candidate publication/consumption, two Engram injections, masked
history, and six selected routed experts at the released arithmetic widths.
Depth, router cardinality, table row count and context remain reduced explicitly.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

import validate_deepseek_v41_production_model as production
import validate_deepseek_v41_bf16_model as bf16
import validate_deepseek_v41_model as model
import validate_deepseek_v41_mixed as mixed

SEED = 4141211
FIXTURE = model.FIXTURE.parent / "deepseek_v41_schedule_width_model.json"
ARGS = production.ARGS | dict(
    n_layers=4, n_routed_experts=6, n_activated_experts=6,
    rope_factor=40,
    max_seq_len=16, compress_ratios=(2, 2, 1, 1),
    kv_source_layers=(0, 2), index_source_layers=(0, 2, 3),
    index_topk=2, candidate_source_layer=2, candidate_topk_blocks=2,
    candidate_block_size=2, engram_layer_ids=(0, 3),
    engram_num_embeddings=(4096, 8192), engram_max_ngram_size=4,
    engram_vocab_size=50, engram_n_heads=8, engram_head_dim=256,
    engram_compressed_vocab_size=99092,
)
STEPS = [
    dict(name="schedule-odd-prefill", pos=0, seq=3, ids=[129279, 13, 701], mask=[1, 0, 1]),
    dict(name="schedule-group-completion", pos=3, seq=1, ids=[1101], mask=[1]),
    dict(name="schedule-incomplete-group", pos=4, seq=1, ids=[411], mask=[1]),
    dict(name="schedule-shorter-reset", pos=0, seq=1, ids=[999], mask=[1]),
    dict(name="schedule-reset-completion", pos=1, seq=1, ids=[23], mask=[1]),
]


def config(tokens):
    value = production.native_config()
    text = value["text_config"]
    text.update(num_hidden_layers=4, max_position_embeddings=16,
                n_routed_experts=6, num_experts_per_tok=6,
                compress_ratios=[2, 2, 1, 1], kv_source_layer_ids=[0, 2],
                index_source_layer_ids=[0, 2, 3], index_topk=2,
                candidate_source_layer_id=2, candidate_topk_blocks=2,
                candidate_block_size=2, engram_layer_ids=[0, 3],
                engram_num_embeddings=[4096, 8192], engram_max_ngram_size=4,
                engram_vocab_size=50, engram_n_heads=8, engram_head_dim=256,
                engram_compressed_vocab_size=99092)
    text["rope_scaling"]["factor"] = ARGS["rope_factor"]
    return value


ORIGINAL_QUANTIZE = production.attention_model.packed_model.quantize_weight


def fp4_tile(weight):
    """Vector form of the existing scalar FP4 weight converter, byte-checked below."""
    values = weight.detach().float().numpy().reshape(weight.shape[0], -1, 32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite reference weight")
    maximum = np.maximum(np.abs(values).max(-1), np.float32(6*2.0**-126))
    bits = (maximum*np.float32(1/6)).view(np.uint32)
    exponent = (bits >> 23) + ((bits & 0x7fffff) != 0).astype(np.uint32)
    scales = (exponent << 23).view(np.float32)
    limited = np.clip(values/scales[..., None], np.float32(-6), np.float32(6))
    magnitude = np.abs(limited)
    codes = np.searchsorted(np.array([.25,.75,1.25,1.75,2.5,3.5,5.0], dtype=np.float32),
                            magnitude, side="right").astype(np.uint8)
    codes[magnitude == np.float32(.25)] = 0
    # The scalar weight converter discards a rounded negative zero.
    codes |= ((limited < 0) & (codes != 0)).astype(np.uint8) << 3
    codes = codes.reshape(-1)
    return (codes[::2] | (codes[1::2] << 4)), exponent.astype(np.uint8).reshape(-1)


def bounded_quantize(weight, dtype):
    """Quantize independent aligned row tiles with bounded temporary storage.

    FP8 uses the unchanged converter; FP4 uses the byte-checked vector form.
    This bounds temporary allocations for the 25600x6144 Engram projection.
    """
    rows, cols = weight.shape
    fp4 = dtype == "fp4_row32"
    data = production.page_backed((rows * cols // (2 if fp4 else 1),), torch.uint8,
                                 production.packed_folder())
    scale_rows = rows if fp4 else (rows+31)//32
    scales = production.page_backed((scale_rows * (cols//32),), torch.uint8,
                                   production.packed_folder())
    for start in range(0, rows, 128):
        end = min(start+128, rows)
        if fp4:
            raw, scale = fp4_tile(weight[start:end])
            part = {"data": raw, "scales": scale}
        else:
            part = ORIGINAL_QUANTIZE(weight[start:end], dtype)
        offset = start*cols//(2 if fp4 else 1)
        data[offset:offset+len(part["data"])] = torch.as_tensor(part["data"], dtype=torch.uint8)
        offset = (start if fp4 else start//32)*(cols//32)
        scales[offset:offset+len(part["scales"])] = torch.as_tensor(part["scales"], dtype=torch.uint8)
    return dict(rows=rows, cols=cols, dtype=dtype, data=data, scales=scales)


def verify_tiling():
    rng = torch.Generator().manual_seed(SEED)
    for dtype in ("fp4_row32", "fp8_block32"):
        weight = torch.randn((137, 64), generator=rng)*.05
        expected = ORIGINAL_QUANTIZE(weight, dtype)
        actual = bounded_quantize(weight, dtype)
        assert actual["data"].tolist() == expected["data"]
        assert actual["scales"].tolist() == expected["scales"]
    frozen = json.loads((model.FIXTURE.parent / "deepseek_v41_quantization.json").read_text())
    for case in frozen["cases"]:
        if case["mode"] == "fp4" and case["scale_mode"] == "ue8m0" and case["block_size"] == 32:
            weight = torch.tensor(case["input"]).reshape(case["rows"], case["cols"])
            expected = ORIGINAL_QUANTIZE(weight, "fp4_row32")
            raw, scale = fp4_tile(weight)
            assert raw.tolist() == expected["data"], case["name"]
            assert scale.tolist() == expected["scales"], case["name"]
    print("PASS aligned oracle tiling: every packed byte/scale identical", flush=True)


def tokenizer_metadata(path, tokenizer_path):
    # This module has no TileLang dependency. Its normalization and NumPy seed
    # code execute verbatim from the already hash-checked published source.
    source = path.parent / "engram.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == model.engram.ENGRAM_SHA256
    spec = importlib.util.spec_from_file_location("schedule_width_pinned_engram", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    class Wrapper:
        backend_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        def __len__(self):
            return self.backend_tokenizer.get_vocab_size(True)
    token_map, count = module.build_compressed_token_map(Wrapper())
    assert len(token_map) == 129280 and count == 99092
    return dict(token_map=token_map,
                multipliers=module.compute_hash_multipliers((0, 3), 4, count).tolist(),
                pad_token_id=2)


def make_case(path, kernel_path, tokens):
    hash_config = dict(layer_ids=[0, 3], num_embeddings=[4096, 8192], max_ngram_size=4,
                       vocab_size=50, n_heads=8, head_dim=256, max_batch_size=1, max_seq_len=16)
    namespace = model.load_reference(path, kernel_path, hash_config=hash_config, tokens=tokens)
    expert_namespace = mixed.source_namespace(path)
    # Retain the released FP8 row lookup, not the diagnostic dense replacement.
    node = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.ClassDef)
                and n.name == "ParallelEngramEmbedding")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-engram-table>", "exec"), namespace)
    production.defer_large_allocations(namespace)
    args = namespace["ModelArgs"](**ARGS)
    torch.set_default_dtype(torch.bfloat16)
    args.n_layers = 0
    net = namespace["Transformer"](args)
    args.n_layers = 4
    torch.set_default_dtype(torch.float32)
    namespace["sparse_attn"] = production.attention_model.sparse_attn
    packed = {}
    rng = torch.Generator().manual_seed(SEED)
    net.embed.materialize()
    production.fill_parameters([(f"embed.{n}", p) for n,p in net.embed.named_parameters()], rng)
    for layer_id in range(4):
        print("BUILD schedule block", layer_id, flush=True)
        # Table initialization is dense; conversion below restores the exact
        # released row-scaled lookup class with its fixed nonzero payload.
        original_table = namespace["ParallelEngramEmbedding"]
        namespace["ParallelEngramEmbedding"] = model.EngramEmbedding
        layer = namespace["Block"](layer_id, args, net.engram_layout)
        namespace["ParallelEngramEmbedding"] = original_table
        net.layers.append(layer)
        for name, param in layer.named_parameters():
            full = f"layers.{layer_id}.{name}"
            if name.endswith("ffn.gate.bias"):
                torch.randn(param.shape, generator=rng)  # original initialization draw
                with torch.no_grad(): param.copy_(torch.linspace(-.15, .15, 6))
            else:
                production.fill_parameters([(full, param)], rng)
        production.transform_layer(layer, packed, expert_namespace)
        attention = layer.attn
        if attention.is_kv_source:
            attention.compress_kv_cache = attention.compress_kv_cache.bfloat16()
        if attention.indexer is not None:
            indexer = attention.indexer
            weight = bounded_quantize(indexer.wq_b.weight, "fp8_block32")
            packed[f"layers.{layer_id}.attn.indexer.wq_b.weight"] = weight
            indexer.wq_b = mixed.Linear(weight)
            if indexer.owns_k: indexer.k_cache = indexer.k_cache.bfloat16()
        if layer.engram is not None:
            engram = layer.engram
            weight = bounded_quantize(engram.wkv.weight, "fp8_block32")
            packed[f"layers.{layer_id}.engram.wkv.weight"] = weight
            engram.wkv = mixed.Linear(weight)
            engram.embed, weight = bf16.engram_table(engram.embed.weight.detach(), original_table)
            production.compact_spec(weight)
            packed[f"layers.{layer_id}.engram.embed.weight"] = weight
        for name, parameter in layer.named_parameters():
            pooling = attention.compress_ratio > 1 and name in ("attn.compressor.wkv.weight", "attn.compressor.wgate.weight")
            if "hc_" not in name and not pooling:
                parameter.data = parameter.data.detach().bfloat16()
        gc.collect()
        print("PACKED schedule block", layer_id, flush=True)
    production.fill_parameters([(f"norm.{n}", p) for n,p in net.norm.named_parameters()], rng)
    net.norm.weight.data = net.norm.weight.detach().bfloat16()
    net.head.materialize()
    production.fill_parameters([(f"head.{n}", p) for n,p in net.head.named_parameters()], rng)
    normalized, collapsed = [], []
    net.norm.register_forward_pre_hook(lambda m,i: collapsed.append(i[0].clone()))
    net.norm.register_forward_hook(lambda m,i,o: normalized.append(o.clone()))
    observer = bf16.Bf16Observer()
    observer.attach(net)
    steps = []
    with torch.inference_mode():
        for step in STEPS:
            print("FORWARD", step["name"], flush=True)
            ids = torch.tensor(step["ids"]).view(1, step["seq"])
            types = torch.tensor([[-1 if live else 1 for live in step["mask"]]])
            observer.begin(step)
            net(ids, step["pos"], token_types=types)
            observer.end(step)
            logits = net.head(normalized[-1], full_logits=True)
            assert logits.numel() and torch.isfinite(logits).all()
            steps.append(step | dict(logits=logits.flatten().tolist(), shape=list(logits.shape),
                                    collapsed=collapsed[-1].flatten().tolist(), normalized=normalized[-1].flatten().tolist()))
    trace = Path(__file__).resolve().parents[2] / "target/deepseek-v41-reference/schedule-width-reference-trace.json"
    trace.write_text(json.dumps({"steps":observer.steps}, separators=(",", ":"), allow_nan=False))
    return net, packed, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    if args.write_fixture and FIXTURE.exists():
        raise RuntimeError("Refusing to rebuild an already frozen schedule-width fixture")
    torch.set_num_threads(1)
    tokens = tokenizer_metadata(args.reference_path, args.tokenizer)
    verify_tiling()
    production.attention_model.packed_model.quantize_weight = bounded_quantize
    try:
        net, packed, steps = make_case(args.reference_path, args.kernel_reference_path, tokens)
        counts = len(net.state_dict()), len(packed)
        folder = f"deepseek_v41_schedule_width_{SEED}"
        native = config(tokens)
        production.native_config = lambda: native
        files = production.write_checkpoint(net, packed, FIXTURE.parent / folder)
        (FIXTURE.parent / folder / "engram_tokens.json").write_text(json.dumps(tokens, separators=(",", ":")))
        for name in ("config.json", "model.safetensors.index.json", "engram_tokens.json"):
            files[name] = hashlib.sha256((FIXTURE.parent / folder / name).read_bytes()).hexdigest()
        result = dict(seed=SEED, revision=model.backbone.REVISION, source_sha256=model.backbone.SOURCE_SHA256,
                      engram_sha256=model.engram.ENGRAM_SHA256, precision="production_width_schedule_top6_bf16_cpu_pytorch",
                      stream_precision="bf16", config=native, batch_size=1, max_prefill=3, max_positions=16,
                      checkpoint=dict(directory=folder, files=files, tensor_count=counts[0], packed_count=counts[1]),
                      steps=steps)
        if args.write_fixture:
            if FIXTURE.exists(): raise RuntimeError("Refusing to overwrite the frozen schedule-width oracle")
            FIXTURE.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
        elif json.loads(FIXTURE.read_text()) != result:
            raise AssertionError("Schedule-width fixture differs from the frozen CPU oracle")
        print("PASS schedule-width CPU oracle", sum(len(s["logits"]) for s in steps), "full logits", flush=True)
    finally:
        production.attention_model.packed_model.quantize_weight = ORIGINAL_QUANTIZE
        production.release_scratch()


if __name__ == "__main__":
    main()
