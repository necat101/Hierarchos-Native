"""Deterministic eligible-geometry fixture for the RDNA3 WMMA training path.

The frozen GPT-2 shape matrix and the local Llama package are far below the
cooperative-matrix tile contract (``hidden_size = 16``), so every projection in
them falls back to its FP32 kernel and a WMMA run on them proves nothing. This
generator builds a model whose dense projections are actually eligible —
``hidden_size = 1024``, so ``q/k/v/o`` and ``gate/up/down`` each produce 1024
columns — and a pretokenized dataset and logits fixture to go with it.

Everything is derived from a fixed seed, so the package is reproducible byte for
byte on the same numpy version. Nothing here is a quality claim about a trained
model: it exists to exercise cooperative dispatches and to compare precision
policies on identical weights.

Usage (from ``hierarchos-vulkan``):

    .venv-torch/Scripts/python.exe validation/make_wmma_fixture.py \
        --output target/wmma-fixture --hidden 1024 --layers 2 --vocab 1024 \
        --seq-len 64 --batch-size 8 --sequences 32

Writes ``model/config.json``, ``model/model.safetensors``, ``dataset.jsonl`` and
``logits.json``. Generated weights and datasets are scratch artifacts and are not
committed.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

SEED = 20261109


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict[str, object]] = {"__metadata__": {"format": "pt"}}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = np.ascontiguousarray(tensors[name], dtype="<f4")
        blob = tensor.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def build_model(hidden: int, intermediate: int, layers: int, vocab: int,
                heads: int, kv_heads: int, head_dim: int,
                max_positions: int) -> tuple[dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    scale = 0.02
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.normal(0.0, scale, (vocab, hidden)).astype(np.float32),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        "lm_head.weight": rng.normal(0.0, scale, (vocab, hidden)).astype(np.float32),
    }
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = rng.normal(0.0, scale, (q_width, hidden)).astype(np.float32)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = rng.normal(0.0, scale, (kv_width, hidden)).astype(np.float32)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = rng.normal(0.0, scale, (kv_width, hidden)).astype(np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = rng.normal(0.0, scale, (hidden, q_width)).astype(np.float32)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = rng.normal(0.0, scale, (intermediate, hidden)).astype(np.float32)
        tensors[f"{prefix}.mlp.up_proj.weight"] = rng.normal(0.0, scale, (intermediate, hidden)).astype(np.float32)
        tensors[f"{prefix}.mlp.down_proj.weight"] = rng.normal(0.0, scale, (hidden, intermediate)).astype(np.float32)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "initializer_range": scale,
        "intermediate_size": intermediate,
        "max_position_embeddings": max_positions,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": heads,
        "num_hidden_layers": layers,
        "num_key_value_heads": kv_heads,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "dtype": "float32",
        "use_cache": False,
        "vocab_size": vocab,
    }
    return config, tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequences", type=int, default=32)
    args = parser.parse_args()

    intermediate = args.intermediate or args.hidden
    if args.heads * args.head_dim != args.hidden:
        raise SystemExit("heads * head_dim must equal the hidden size")

    config, tensors = build_model(
        args.hidden, intermediate, args.layers, args.vocab,
        args.heads, args.kv_heads, args.head_dim, args.max_positions,
    )
    model_dir = args.output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_safetensors(model_dir / "model.safetensors", tensors)

    # Pretokenized causal rows: a learnable repeating pattern inside the regular
    # token range, so loss falls quickly and reproducibly.
    rng = np.random.default_rng(SEED + 1)
    low, high = 4, max(5, args.vocab - 4)
    with (args.output / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(args.sequences):
            base = int(rng.integers(low, high))
            ids = [int((base + step * 7 + index) % (high - low) + low)
                   for step in range(args.seq_len)]
            handle.write(json.dumps({"input_ids": ids}) + "\n")

    input_ids = [int(value) for value in rng.integers(low, high, args.seq_len * args.batch_size)]
    fixture = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "input_ids": input_ids,
        "attention_mask": [1.0] * len(input_ids),
    }
    (args.output / "logits.json").write_text(json.dumps(fixture) + "\n")
    parameter_bytes = sum(t.size * t.itemsize for t in tensors.values())
    print(json.dumps({
        "output": str(args.output),
        "model": str(model_dir),
        "parameters": int(sum(t.size for t in tensors.values())),
        "parameter_bytes": int(parameter_bytes),
        "projections": args.layers * 7,
        "columns_per_projection": args.hidden,
        "rows": args.batch_size * args.seq_len,
    }, indent=2))


if __name__ == "__main__":
    main()
