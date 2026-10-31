"""Complete BF16 residual/HC V4.1 CPU PyTorch oracle.

The pinned model forward bodies are unchanged. Attention/indexer/Engram/shared
experts use FP8 weights, routed experts use FP4, and the block-scaled GEMM CPU
mirror returns BF16. All declared stream casts, dense BF16 weights, FP32 HC and
ratio-2 pooling, and FP32 logit output follow the released graph. CUDA parity
is not required. The existing strict complete-logit comparator is unchanged.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

import validate_deepseek_v41_attention_model as attention_model
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_model as model
from trace_deepseek_v41_model import Observer
from deepseek_v41_parity import require_logit_parity

FIXTURE = model.FIXTURE.parent / "deepseek_v41_bf16_model.json"
SPECS = [(4141061, 2, 2), (4141062, 3, 4)]


class Bf16Observer(Observer):
    """Observation-only index diagnostics; no reference tensor is modified."""
    def attach(self, net):
        super().attach(net)
        namespace = type(net.layers[0].attn).forward.__globals__
        original_sparse = namespace["sparse_attn"]
        def sparse(q, kv, sink, indices, scale):
            self.save(self.active_layer, "attention_queries", q)
            result = original_sparse(q, kv, sink, indices, scale)
            self.save(self.active_layer, "attention_sparse", result)
            return result
        namespace["sparse_attn"] = sparse
        for layer in net.layers:
            if layer.attn.compress_ratio:
                def capture_cache(module, args, output, *, layer_id=layer.layer_id):
                    x, pos = args
                    batch, seq = x.shape[:2]
                    width = (pos + seq) // module.compress_ratio
                    runtime = namespace["shared_attn"]
                    bits = lambda x: x.float().contiguous().view(torch.uint32).flatten().tolist()
                    self.cache_diagnostics[layer_id] = dict(kv=bits(runtime.compress_kv[:batch, :width]),
                                                          keys=bits(runtime.index_k[:batch, :width]))
                layer.attn.register_forward_hook(capture_cache)
            indexer = layer.attn.indexer
            if indexer is None:
                continue
            def capture(module, args, output, *, layer_id=layer.layer_id):
                x, qr, latent, pos, offset = args
                batch, seq = x.shape[:2]
                width = (pos + seq) // module.compress_ratio
                q = module.wq_b(qr).unflatten(-1, (module.n_local_heads, module.index_head_dim))
                namespace["apply_rotary_emb"](q[..., -module.rope_head_dim:], module.freqs_cis[pos:pos+seq])
                namespace["fp4_act_quant"](q, 32, True)
                k = namespace["shared_attn"].index_k[:batch, :width]
                weights = module.weights_proj(x) * (module.softmax_scale * module.n_heads ** -.5)
                scores = (torch.einsum("bshd,btd->bsht", q, k).relu_() * weights.unsqueeze(-1)).sum(2)
                visible = (torch.arange(1, seq+1) // module.compress_ratio).view(1,seq,1) if pos == 0 else width
                scores.masked_fill_(torch.arange(width) >= visible, -torch.inf)
                if module.uses_candidates:
                    scores.masked_fill_(~namespace["shared_attn"].candidates, -torch.inf)
                self.index_diagnostics[layer_id] = dict(indices=output.flatten().tolist(),
                    score_bits=scores.float().contiguous().view(torch.uint32).flatten().tolist())
            indexer.register_forward_hook(capture)

    def begin(self, spec):
        super().begin(spec)
        self.index_diagnostics = {}
        self.cache_diagnostics = {}

    def end(self, spec):
        super().end(spec)
        self.steps[-1]["index_diagnostics"] = self.index_diagnostics
        self.steps[-1]["cache_diagnostics"] = self.cache_diagnostics


def engram_table(values, cls):
    rows, cols = values.shape
    dequant, scales = mixed.quant.quantize_case(values.float().numpy(), 32, "fp8", "ue8m0")
    scales = torch.from_numpy(scales)
    raw = (torch.from_numpy(dequant) / scales.repeat_interleave(32, 1)).to(torch.float8_e4m3fn)
    scale_bytes = scales.to(torch.float8_e8m0fnu).view(torch.uint8)
    # These tensors deliberately are not registered Parameters: the payload is
    # exported as packed bytes, while the released lookup method runs verbatim.
    embed = cls.__new__(cls)
    nn.Module.__init__(embed)
    embed.weight, embed.scale = raw, scale_bytes.view(torch.float8_e8m0fnu)
    embed.block_size, embed.vocab_start_idx, embed.vocab_end_idx = 32, 0, rows
    spec = dict(rows=rows, cols=cols, dtype="engram_fp8_row32",
                data=raw.view(torch.uint8).flatten().tolist(), scales=scale_bytes.flatten().tolist())
    return embed, spec


def make_case(path, kernel_path, seed, topk, hc, *, observer=None, args_overrides=None,
              config_overrides=None, steps=None, tokens=None, max_prefill=8):
    expert_ns = mixed.source_namespace(path)
    packed_tensors = {}
    bf16_tensors = set()

    def transform(net):
        namespace = type(net.layers[0].attn).forward.__globals__
        namespace["sparse_attn"] = attention_model.sparse_attn
        # The model source has already been hash-checked by load_reference.
        node = next(n for n in ast.parse(path.read_bytes()).body
                    if isinstance(n, ast.ClassDef) and n.name == "ParallelEngramEmbedding")
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        for layer in net.layers:
            prefix = f"layers.{layer.layer_id}"
            attention = layer.attn
            for name in ("wq_a", "wq_b", "wkv", "wo_b"):
                spec = attention_model.packed_model.quantize_weight(getattr(attention, name).weight, "fp8_block32")
                packed_tensors[f"{prefix}.attn.{name}.weight"] = spec
                setattr(attention, name, mixed.Linear(spec))
            if attention.indexer is not None:
                indexer = attention.indexer
                spec = attention_model.packed_model.quantize_weight(indexer.wq_b.weight, "fp8_block32")
                indexer.wq_b = mixed.Linear(spec)
                packed_tensors[f"{prefix}.attn.indexer.wq_b.weight"] = spec
                if indexer.owns_k:
                    indexer.k_cache = indexer.k_cache.bfloat16()
            attention.window_kv_cache = attention.window_kv_cache.bfloat16()
            if attention.is_kv_source:
                attention.compress_kv_cache = attention.compress_kv_cache.bfloat16()
            moe = layer.ffn
            for index, expert in enumerate(moe.experts):
                spec = {name: attention_model.packed_model.quantize_weight(getattr(expert, name).weight, "fp4_row32")
                        for name in ("w1", "w2", "w3")}
                moe.experts[index] = mixed.expert(expert_ns, spec, expert.swiglu_limit)
                for name, weight in spec.items():
                    packed_tensors[f"{prefix}.ffn.experts.{index}.{name}.weight"] = weight
            expert = moe.shared_experts
            spec = {name: attention_model.packed_model.quantize_weight(getattr(expert, name).weight, "fp8_block32")
                    for name in ("w1", "w2", "w3")}
            moe.shared_experts = mixed.expert(expert_ns, spec, expert.swiglu_limit)
            for name, weight in spec.items():
                packed_tensors[f"{prefix}.ffn.shared_experts.{name}.weight"] = weight
            if layer.engram is not None:
                engram = layer.engram
                spec = attention_model.packed_model.quantize_weight(engram.wkv.weight, "fp8_block32")
                engram.wkv = mixed.Linear(spec)
                packed_tensors[f"{prefix}.engram.wkv.weight"] = spec
                engram.embed, spec = engram_table(engram.embed.weight.detach(), namespace["ParallelEngramEmbedding"])
                packed_tensors[f"{prefix}.engram.embed.weight"] = spec
        for name, parameter in net.named_parameters():
            # HC coefficients and ratio>1 projections are explicitly FP32.
            is_pool = any(name.startswith(f"layers.{i}.attn.compressor.")
                          and name.endswith(("wkv.weight", "wgate.weight"))
                          for i, layer in enumerate(net.layers) if layer.attn.compress_ratio > 1)
            if "hc_" in name or is_pool:
                continue
            if name == "head.weight":
                parameter.data = parameter.detach().bfloat16().float()
            else:
                parameter.data = parameter.detach().bfloat16()
            bf16_tensors.add(name)

    overrides = {"q_lora_rank":32, "o_lora_rank":32, "moe_inter_dim":32,
                 "hc_mult":hc, "engram_head_dim":32} | (args_overrides or {})
    native_config = dict(q_lora_rank=32, o_lora_rank=32, moe_intermediate_size=32,
                         hc_mult=hc, engram_head_dim=32) | (config_overrides or {})
    f = model.make_fixture(path, kernel_path, observer, seed=seed, topk=topk,
                          args_overrides=overrides, model_transform=transform,
                          config_overrides=native_config, steps=steps, tokens=tokens, max_prefill=max_prefill)
    f["precision"] = "bf16_residual_hc_norm_compressor_engram_packed_attention_indexer_experts_cpu_pytorch"
    f["stream_precision"] = "bf16"
    f["packed_tensors"] = packed_tensors
    for name in bf16_tensors:
        f["tensors"][name]["dtype"] = "bf16"
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path", type=Path, required=True)
    p.add_argument("--kernel-reference-path", type=Path, required=True)
    p.add_argument("--write-fixture", action="store_true")
    p.add_argument("--native-output", type=Path)
    p.add_argument("--trace-output", type=Path)
    args = p.parse_args()
    observers = [Bf16Observer() if args.trace_output else None for _ in SPECS]
    f = dict(cases=[make_case(args.reference_path, args.kernel_reference_path, *spec, observer=observer)
                    for spec, observer in zip(SPECS, observers)])
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(f, separators=(",", ":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f:
        raise AssertionError("BF16-model fixture differs from the pinned CPU PyTorch reference")
    if args.trace_output:
        args.trace_output.write_text(json.dumps(dict(cases=[dict(seed=spec[0], steps=observer.steps)
            for spec, observer in zip(SPECS, observers)]), allow_nan=False))
    print(f"PASS complete BF16 CPU oracle: fixed seeds/topk/HC {SPECS}")
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        assert len(native["cases"]) == len(f["cases"])
        binding = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        for expected, actual, observer in zip(f["cases"], native["cases"], observers):
            assert expected["seed"] == actual["seed"]
            print(f"BF16 seed {expected['seed']}")
            model.compare_native(expected, actual, fixture_sha256=binding)
            if observer is not None:
                exact_count = cache_count = score_count = 0
                sparse_drift = 0.0
                for a, b in zip(actual["steps"], observer.steps):
                    stages = {(s["layer"], s["stage"]): s["values"] for s in a["stages"]}
                    reference = {(s["layer"], s["stage"]): s["values"] for s in b["stages"]}
                    assert stages.keys() == reference.keys(), "complete stage trace required"
                    for key, values in stages.items():
                        x = torch.tensor(values, dtype=torch.float32)
                        y = torch.tensor(reference[key], dtype=torch.float32)
                        assert x.numel() and x.shape == y.shape
                        if key[1] == "attention_sparse":
                            sparse_drift = max(sparse_drift, require_logit_parity(x, y, label=f"sparse {key}"))
                        else:
                            assert torch.equal(x.view(torch.uint32), y.view(torch.uint32)), (a["name"], key)
                            exact_count += x.numel()
                    assert a["cache_diagnostics"].keys() == {str(layer) for layer in b["cache_diagnostics"]}
                    for layer, cache in b["cache_diagnostics"].items():
                        assert a["cache_diagnostics"][str(layer)] == cache
                    cache_count += sum(len(c["kv"]) + len(c["keys"]) for c in b["cache_diagnostics"].values())
                    for layer, scores in b["index_diagnostics"].items():
                        assert a["index_diagnostics"][str(layer)] == scores
                        score_count += len(scores["score_bits"])
                print(f"PASS {exact_count} HC/stream/norm/projection values, {cache_count} cache values, "
                      f"{score_count} index scores and every selected index bit-exact; sparse max drift {sparse_drift:.17g}")


if __name__ == "__main__":
    main()
