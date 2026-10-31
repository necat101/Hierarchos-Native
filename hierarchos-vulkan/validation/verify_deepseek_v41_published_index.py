#!/usr/bin/env python3
"""Verify the native V4.1 loader's tensor-name contract against a published index.

No weight payload is read. Point --index at the published
model.safetensors.index.json; the released revision names 96,085 tensors.
Optional tensors are allowed to be absent: a ratio-1 compressor has no gate and
shared-index consumers own no key projection.
"""
import argparse
import json
import pathlib
import sys

DEFAULT_CONFIG = pathlib.Path(__file__).with_name("fixtures") / "deepseek_v41_flash_1017_config.json"


def required_names(text):
    required = {"embed.weight", "norm.weight", "head.weight"}
    optional = set()
    layers = text["num_hidden_layers"]
    experts = text["n_routed_experts"]
    for layer in range(layers):
        prefix = f"layers.{layer}."
        required.update(
            prefix + name
            for name in (
                "hc_attn_fn",
                "hc_ffn_fn",
                "hc_attn_base",
                "hc_ffn_base",
                "hc_attn_scale",
                "hc_ffn_scale",
                "attn.attn_sink",
                "attn.q_norm.weight",
                "attn.kv_norm.weight",
                "attn.wq_a.weight",
                "attn.wq_a.scale",
                "attn.wq_b.weight",
                "attn.wq_b.scale",
                "attn.wkv.weight",
                "attn.wkv.scale",
                "attn.wo_a.weight",
                "attn.wo_a.scale",
                "attn.wo_b.weight",
                "attn.wo_b.scale",
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn.gate.weight",
                "ffn.gate.bias",
                "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w1.scale",
                "ffn.shared_experts.w2.weight",
                "ffn.shared_experts.w2.scale",
                "ffn.shared_experts.w3.weight",
                "ffn.shared_experts.w3.scale",
            )
        )
        for expert in range(experts):
            for module in ("w1", "w2", "w3"):
                name = f"{prefix}ffn.experts.{expert}.{module}"
                required.add(f"{name}.weight")
                required.add(f"{name}.scale")
    ratios = text["compress_ratios"]
    for layer in text["kv_source_layer_ids"]:
        prefix = f"layers.{layer}.attn.compressor."
        required.update(prefix + name for name in ("wkv.weight", "norm.weight"))
        gate = prefix + "wgate.weight"
        (required if ratios[layer] > 1 else optional).add(gate)
    for layer in text["index_source_layer_ids"]:
        prefix = f"layers.{layer}.attn.indexer."
        required.update(
            prefix + name
            for name in ("wq_b.weight", "wq_b.scale", "weights_proj.weight")
        )
        # Only the KV owner produces shared index keys.
        key_names = {prefix + "wk.weight", prefix + "k_norm.weight"}
        (required if layer in text["kv_source_layer_ids"] else optional).update(key_names)
    for layer in text["engram_layer_ids"]:
        prefix = f"layers.{layer}.engram."
        required.update(
            prefix + name
            for name in (
                "embed.weight",
                "embed.scale",
                "wkv.weight",
                "wkv.scale",
                "q_weight",
                "k_weight",
            )
        )
    return required, optional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    weight_map = json.loads(args.index.read_text(encoding="utf-8"))["weight_map"]
    text = json.loads(args.config.read_text(encoding="utf-8"))["text_config"]
    required, optional = required_names(text)
    missing = sorted(required - weight_map.keys())
    if missing:
        print(f"FAIL: {len(missing)} required tensors are absent", file=sys.stderr)
        for name in missing[:20]:
            print(f"  {name}", file=sys.stderr)
        return 1
    absent_optional = sorted(optional - weight_map.keys())
    print(
        f"PASS: {len(required)} required names present; "
        f"{len(absent_optional)} of {len(optional)} optional names absent as expected; "
        f"index contains {len(weight_map)} tensors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
