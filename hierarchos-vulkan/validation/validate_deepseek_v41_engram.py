"""Qualify V4.1 Engram against the pinned released sources.

Extracts the unchanged `find_next_prime`/`EngramLayout` from `inference/engram.py`
and `NgramHashState`/`Engram` from `inference/model.py`, then executes their
forward code. The compressed token map and the NumPy-seeded multipliers are
conversion artifacts in the native design, so this oracle also records the
primes/offsets and runs hash cases with explicitly supplied multipliers. The
embedding table is exercised at its post-dequantization BF16-rounded values; the
released lookup dequantizes FP8 rows with UE8M0 scales.

    python validation/validate_deepseek_v41_engram.py
    python validation/validate_deepseek_v41_engram.py --write-fixture
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import types
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
from sympy import isprime
from torch import nn

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
ENGRAM_SHA256 = "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_engram.json"

ATOL = 2e-6


def load_source(name: str, expected: str, path: Path | None) -> bytes:
    if path is None:
        with urlopen(
            f"https://huggingface.co/{MODEL}/raw/{REVISION}/inference/{name}",
            timeout=60,
        ) as response:
            source = response.read()
    else:
        source = (path.parent / name).read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{name} hash mismatch: {actual}")
    return source


def load_reference(path: Path | None):
    engram_tree = ast.parse(load_source("engram.py", ENGRAM_SHA256, path))
    model_tree = ast.parse(load_source("model.py", SOURCE_SHA256, path))
    engram_nodes = [
        node
        for node in engram_tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "find_next_prime")
        or (isinstance(node, ast.ClassDef) and node.name in {"EngramLayout", "NgramHashState"})
    ]
    if {getattr(node, "name", "") for node in engram_nodes} != {
        "find_next_prime", "EngramLayout", "NgramHashState"
    }:
        raise RuntimeError("the released engram.py no longer defines the expected hash code")
    hash_nodes = [
        node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Engram"
    ]
    if {node.name for node in hash_nodes} != {"Engram"}:
        raise RuntimeError("the released model.py no longer defines Engram")
    namespace: dict[str, object] = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "dataclass": __import__("dataclasses").dataclass,
        "isprime": isprime,
        "EngramLayout": object,
        "ModelArgs": object,
        "ParallelEngramEmbedding": object,
        "Linear": object,
        "world_size": 1,
        "rank": 0,
    }
    exec(compile(ast.Module(body=engram_nodes, type_ignores=[]), "<engram>", "exec"), namespace)
    exec(compile(ast.Module(body=hash_nodes, type_ignores=[]), "<model>", "exec"), namespace)
    return namespace


def layout_from_config(namespace, config) -> types.SimpleNamespace:
    args = types.SimpleNamespace(
        engram_layer_ids=config["layer_ids"],
        engram_num_embeddings=config["num_embeddings"],
        engram_max_ngram_size=config["max_ngram_size"],
        engram_vocab_size=config["vocab_size"],
        engram_n_heads=config["n_heads"],
        engram_head_dim=config["head_dim"],
    )
    layout = namespace["EngramLayout"].from_args(args)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
    return types.SimpleNamespace(layout=layout, offsets=offsets)


def layout_case(namespace, name, config):
    derived = layout_from_config(namespace, config)
    layout = derived.layout
    return {
        "name": name,
        "config": config,
        "expected_primes": [[list(heads) for heads in layer] for layer in layout.primes],
        "expected_offsets": [[int(value) for value in row] for row in derived.offsets],
    }


def hash_state(namespace, config, token_map, multipliers, pad_token_id):
    derived = layout_from_config(namespace, config)
    layout = derived.layout
    state = namespace["NgramHashState"].__new__(namespace["NgramHashState"])
    nn.Module.__init__(state)
    state.layout = layout
    state.pad_id = int(token_map[pad_token_id])
    state.register_buffer("primes", torch.tensor(layout.primes, dtype=torch.int64), persistent=False)
    state.register_buffer("offsets", torch.tensor(np.array(derived.offsets), dtype=torch.int64), persistent=False)
    state.register_buffer("multipliers", torch.tensor(multipliers, dtype=torch.int64), persistent=False)
    state.register_buffer("token_map", torch.tensor(token_map, dtype=torch.int64), persistent=False)
    state.register_buffer(
        "cache", torch.empty(config["max_batch_size"], config["max_seq_len"], dtype=torch.int64), persistent=False
    )
    return state, derived


HASH_CONFIG = {
    # Lists so the in-memory fixture equals its JSON round trip.
    "layer_ids": [1, 3],
    "num_embeddings": [400, 400],
    "max_ngram_size": 4,
    "vocab_size": 50,
    "n_heads": 2,
    "head_dim": 4,
    "pad_token_id": 2,
    "compressed_vocab_size": 8,
    "max_batch_size": 2,
    "max_seq_len": 12,
}

TRIGRAM_CONFIG = HASH_CONFIG | {
    "layer_ids": [2, 5],
    "num_embeddings": [300, 900],
    "max_ngram_size": 3,
    "max_seq_len": 8,
}
TRIGRAM_TOKEN_MAP = [7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7]
TRIGRAM_MULTIPLIERS = [[3, 5, 9], [7, 11, 13]]

# Three-token lookback with a dead span, a fully masked step, and a lone decode
# token whose n-gram reaches back across the previous step's cache.
TRIGRAM_STEPS = [
    {
        "start_pos": 0,
        "seq": 3,
        "token_ids": [[0, 1, 2], [3, 4, 5]],
        "token_mask": [[1, 0, 1], [1, 1, 1]],
    },
    {
        "start_pos": 3,
        "seq": 2,
        "token_ids": [[6, 7], [8, 9]],
        "token_mask": [[0, 0], [1, 1]],
    },
    {
        "start_pos": 5,
        "seq": 1,
        "token_ids": [[2], [10]],
        "token_mask": [[1], [1]],
    },
]
HASH_TOKEN_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 3, 2, 1, 0]
HASH_MULTIPLIERS = [[3, 5, 7, 9], [11, 13, 15, 17]]

HASH_STEPS = [
    {
        "start_pos": 0,
        "seq": 4,
        "token_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "token_mask": [[1, 1, 1, 1], [1, 1, 0, 0]],
    },
    {
        "start_pos": 4,
        "seq": 3,
        "token_ids": [[8, 9, 10], [1, 2, 3]],
        "token_mask": [[1, 1, 1], [1, 1, 1]],
    },
]


def hash_case(namespace, name, config, token_map, multipliers, steps):
    state, derived = hash_state(namespace, config, token_map, multipliers, config["pad_token_id"])
    expected = [[None for _ in steps] for _ in config["layer_ids"]]
    for step_index, step in enumerate(steps):
        ids = torch.tensor(step["token_ids"], dtype=torch.int64)
        mask = torch.tensor(step["token_mask"], dtype=torch.bool)
        hashes = state(ids, step["start_pos"], mask)
        for layer_index in range(len(config["layer_ids"])):
            expected[layer_index][step_index] = (
                hashes[:, :, layer_index, :].flatten().to(torch.int64).tolist()
            )
    return {
        "name": name,
        "config": config,
        "token_map": token_map,
        "pad_token_id": config["pad_token_id"],
        "multipliers": multipliers,
        "primes": [[list(heads) for heads in layer] for layer in derived.layout.primes],
        "offsets": [[int(value) for value in row] for row in derived.offsets],
        "steps": steps,
        "expected": expected,
    }


class EmbedStub(nn.Module):
    """Released `ParallelEngramEmbedding` lookup with its rows already dequantized.

    The mask is the released forward's own arithmetic ("indices outside this
    shard fetch row 0 and are then zeroed"); with one shard, any id outside
    `[0, num_embeddings)` is dead. Keeping the mask here is what lets a case use
    negative and past-the-end ids instead of crashing `F.embedding`.
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = nn.Parameter(table)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        mask = (indices < 0) | (indices >= self.table.shape[0])
        local = indices.masked_fill(mask, 0)
        return F.embedding(local, self.table).masked_fill(mask.unsqueeze(-1), 0)


class LinearStub(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


GATE_CONFIG = {
    "hc_mult": 2,
    "dim": 8,
    "head_dim": 4,
    "max_ngram_size": 2,
    "n_heads": 2,
    "num_embeddings": 5,
}


def gate_case(namespace, name, rows, mask, seed, hashes=None):
    rng = torch.Generator().manual_seed(seed)
    hc = GATE_CONFIG["hc_mult"]
    dim = GATE_CONFIG["dim"]
    head_dim = GATE_CONFIG["head_dim"]
    cols = (GATE_CONFIG["max_ngram_size"] - 1) * GATE_CONFIG["n_heads"]
    stream = torch.randn(rows, hc, dim, generator=rng, dtype=torch.float32) * 0.4
    # The native table contract is the released lookup's BF16-rounded output.
    table = (
        torch.randn(GATE_CONFIG["num_embeddings"], head_dim, generator=rng, dtype=torch.float32) * 0.5
    ).to(torch.bfloat16).float()
    if hashes is None:
        hashes = torch.randint(0, GATE_CONFIG["num_embeddings"], (rows, cols), generator=rng)
    else:
        hashes = torch.tensor(hashes, dtype=torch.int64)
        assert hashes.shape == (rows, cols)
    wkv = torch.randn(dim * (hc + 1), cols * head_dim, generator=rng, dtype=torch.float32) * 0.2
    q_weight = torch.randn(hc, dim, generator=rng, dtype=torch.float32) * 0.3
    k_weight = torch.randn(hc, dim, generator=rng, dtype=torch.float32) * 0.3

    engram = namespace["Engram"].__new__(namespace["Engram"])
    nn.Module.__init__(engram)
    engram.dim = dim
    engram.hc_mult = hc
    engram.clamp_value = 1e-6
    engram.eps = 1e-20
    engram.embed = EmbedStub(table)
    engram.wkv = LinearStub(wkv)
    engram.q_weight = q_weight
    engram.k_weight = k_weight
    token_mask = None if mask is None else torch.tensor(mask, dtype=torch.bool)
    expected = engram.forward(stream, hashes, token_mask)
    assert expected.numel() == rows * hc * dim
    return {
        "name": name,
        "rows": rows,
        "hc": hc,
        "hidden": dim,
        "head_dim": head_dim,
        "cols": cols,
        "num_embeddings": GATE_CONFIG["num_embeddings"],
        "eps": 1e-20,
        "atol": ATOL,
        "stream": stream.flatten().tolist(),
        "table": table.flatten().tolist(),
        "hashes": hashes.flatten().to(torch.int32).tolist(),
        "wkv": wkv.flatten().tolist(),
        "q_weight": q_weight.flatten().tolist(),
        "k_weight": k_weight.flatten().tolist(),
        "token_mask": None if mask is None else [int(value) for value in mask],
        "expected": expected.flatten().tolist(),
    }


def released_config():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "deepseek_v41_flash_1017_config.json").read_text()
    )["text_config"]
    return {
        # Lists, not tuples: these values are embedded in the JSON fixture, and a
        # tuple would never compare equal to its own round-tripped form.
        "layer_ids": list(fixture["engram_layer_ids"]),
        "num_embeddings": list(fixture["engram_num_embeddings"]),
        "max_ngram_size": fixture["engram_max_ngram_size"],
        "vocab_size": fixture["engram_vocab_size"],
        "n_heads": fixture["engram_n_heads"],
        "head_dim": fixture["engram_head_dim"],
        "pad_token_id": fixture["engram_pad_token_id"],
        "compressed_vocab_size": fixture["engram_compressed_vocab_size"],
    }


def make_fixture(path: Path | None):
    namespace = load_reference(path)
    return {
        "model": MODEL,
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "engram_source_sha256": ENGRAM_SHA256,
        "layout_cases": [
            layout_case(namespace, "tiny-hash-layout", HASH_CONFIG),
            layout_case(namespace, "released-layout", released_config() | {
                "max_batch_size": 1,
                "max_seq_len": 1,
            }),
        ],
        "hash_cases": [
            hash_case(
                namespace,
                "steps-mask",
                HASH_CONFIG,
                HASH_TOKEN_MAP,
                HASH_MULTIPLIERS,
                HASH_STEPS,
            ),
            hash_case(
                namespace,
                "trigram-dead-span",
                TRIGRAM_CONFIG,
                TRIGRAM_TOKEN_MAP,
                TRIGRAM_MULTIPLIERS,
                TRIGRAM_STEPS,
            ),
        ],
        "gate_cases": [
            gate_case(namespace, "gate-unmasked", 3, None, 4141031),
            gate_case(namespace, "gate-masked", 2, [1, 0], 4141032),
            # More rows than one 64-wide dispatch row: the gate reduction must not
            # depend on how many rows share a workgroup.
            gate_case(
                namespace,
                "gate-wide-masked",
                17,
                [index % 3 != 1 for index in range(17)],
                4141033,
            ),
            # Negative ids and ids past the table are dead in the released lookup
            # (`masked_fill(..., 0)`), so their rows contribute zeros while the
            # gate still runs on the stream.
            gate_case(
                namespace,
                "gate-dead-ids",
                6,
                [1, 1, 0, 1, 1, 1],
                4141034,
                hashes=[
                    [-1, 0],
                    [3, 4],
                    [0, 0],
                    [99, 100],
                    [1, -7],
                    [4, 3],
                ],
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    fixture = make_fixture(args.reference_path)
    if args.write_fixture:
        metadata = {key: value for key, value in fixture.items()
                    if key not in {"layout_cases", "hash_cases", "gate_cases"}}
        blocks = []
        for key in ("layout_cases", "hash_cases", "gate_cases"):
            cases = ",\n".join(
                "    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
                for case in fixture[key]
            )
            blocks.append(f'  "{key}": [\n{cases}\n  ]')
        FIXTURE.write_text(
            json.dumps(metadata, indent=2)[:-2] + ",\n" + ",\n".join(blocks) + "\n}\n",
            encoding="utf-8",
        )
    elif json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
        raise RuntimeError("Engram fixtures differ from the pinned reference")
    released = fixture["layout_cases"][1]
    released_primes = sum(
        len(heads) for layer in released["expected_primes"] for heads in layer
    )
    hash_values = sum(
        len(expected) for case in fixture["hash_cases"] for expected in case["expected"]
    )
    gate_values = sum(len(case["expected"]) for case in fixture["gate_cases"])
    print(
        f"PASS: {len(fixture['layout_cases'])} layout cases "
        f"({released_primes} released primes), "
        f"{len(fixture['hash_cases'])} hash cases ({hash_values} id vectors), "
        f"{len(fixture['gate_cases'])} gate cases ({gate_values} values); {REVISION}"
    )


if __name__ == "__main__":
    main()
