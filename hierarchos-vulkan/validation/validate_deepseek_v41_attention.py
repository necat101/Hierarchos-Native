"""Qualify V4.1 sparse attention against the pinned reference semantics.

Extracts the unchanged `get_window_topk_idxs` from the released
`inference/model.py` and records its ring-buffer answers for prefill and decode
shapes. The gather itself cannot be run here -- the released `sparse_attn` is a
TileLang kernel -- so this script mirrors its body: gather the sliding-window
rows followed by the selected compressed rows, skip every `-1` slot, include the
attention sink as a virtual zero-value slot in the denominator only, and
normalize. It computes that in float64, which is the contract the FP32 kernels
are checked against, not a PyTorch/MKL reduction-order claim.

    python validation/validate_deepseek_v41_attention.py
    python validation/validate_deepseek_v41_attention.py --write-fixture
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen

import torch

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_attention.json"

HEADS = 2
DIM = 32


def load_source(path: Path | None) -> bytes:
    if path is None:
        with urlopen(
            f"https://huggingface.co/{MODEL}/raw/{REVISION}/inference/model.py",
            timeout=60,
        ) as response:
            source = response.read()
    else:
        source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"reference source hash mismatch: {actual}")
    return source


def load_window_indices(path: Path | None):
    """Compile the released `get_window_topk_idxs` verbatim."""
    tree = ast.parse(load_source(path))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_window_topk_idxs"
    ]
    if len(nodes) != 1:
        raise RuntimeError("the released source no longer defines get_window_topk_idxs once")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace: dict[str, object] = {}
    exec("from functools import lru_cache\nimport torch\n", namespace)
    exec(compile(module, "<reference>", "exec"), namespace)
    return namespace["get_window_topk_idxs"]


WINDOW_SPECS = [
    # name, batch, seq, start_pos, window
    ("prefill-fit", 1, 3, 0, 4),
    ("prefill-exact", 1, 4, 0, 4),
    ("prefill-wrap", 2, 6, 0, 4),
    ("prefill-window-one", 1, 5, 0, 1),
    ("decode-fill", 1, 1, 2, 4),
    ("decode-ring", 2, 1, 6, 4),
    ("decode-max", 1, 1, 11, 3),
]

ATTENTION_SPECS = [
    # name, batch, seq, start_pos, window, ratio, topk, capacity
    ("prefill", 2, 4, 0, 3, 2, 2, 3),
    ("prefill-wrap", 1, 6, 0, 4, 2, 3, 4),
    ("decode", 2, 1, 6, 4, 2, 3, 4),
    ("decode-fill", 1, 1, 2, 4, 2, 3, 2),
    ("sliding-only", 1, 5, 0, 4, 2, 0, 0),
]


def window_cases(get_window_indices):
    cases = []
    for name, batch, seq, start_pos, window in WINDOW_SPECS:
        idxs = get_window_indices(window, batch, seq, start_pos)
        cases.append(
            {
                "name": name,
                "batch": batch,
                "seq": seq,
                "start_pos": start_pos,
                "window": window,
                "topk": idxs.size(-1),
                "expected_indices": idxs.flatten().tolist(),
            }
        )
    return cases


def compress_indices(rng, rows, seq, start_pos, ratio, topk, offset):
    """Sorted compressed slots with the reference's additive offset and -1s."""
    width = (start_pos + seq) // ratio
    if width == 0 or topk == 0:
        return [], 0, width
    topk = min(topk, width)
    indices = []
    for row in range(rows):
        visible = (row % seq + 1) // ratio if start_pos == 0 else width
        row_indices = []
        if row != 0:
            available = torch.randperm(visible, generator=rng).tolist()[:topk]
            row_indices = [position + offset for position in available]
        row_indices += [-1] * (topk - len(row_indices))
        indices.append(sorted(row_indices))
    return indices, topk, width


def gather_case(name, batch, seq, start_pos, window, ratio, topk_setting, capacity, rng, get_window_indices):
    rows = batch * seq
    window_idxs = get_window_indices(window, batch, seq, start_pos)
    window_topk = window_idxs.size(-1)
    window_rows = seq if start_pos == 0 else window
    offset = window_rows

    compressed_idxs, compressed_topk, _ = compress_indices(
        rng, rows, seq, start_pos, ratio, topk_setting, offset
    )
    capacity = max(capacity, compressed_topk)

    q = torch.randn(rows, HEADS, DIM, generator=rng) * 0.5
    window_kv = torch.randn(batch, window_rows, DIM, generator=rng) * 0.5
    compressed_kv = (
        torch.randn(batch, capacity, DIM, generator=rng) * 0.5
        if compressed_topk
        else torch.zeros(batch, 0, DIM)
    )
    sink = torch.randn(HEADS, generator=rng) * 0.5

    combined = torch.cat([window_kv, compressed_kv], dim=1).double()
    scale = DIM**-0.5
    expected = torch.zeros(rows, HEADS, DIM, dtype=torch.float64)
    for row in range(rows):
        item = row // seq
        for head in range(HEADS):
            slots = []
            for slot in range(window_topk):
                index = window_idxs[item, row % seq, slot].item()
                if index >= 0:
                    slots.append((index, window_kv[item, index].double()))

            def dot(rows_tensor) -> float:
                return float(torch.dot(q[row, head].double(), rows_tensor)) * scale

            scores = [dot(kv) for _, kv in slots]
            for slot in range(compressed_topk):
                index = compressed_idxs[row][slot]
                if index < 0:
                    continue
                slots.append((index, combined[item, index]))
                scores.append(dot(combined[item, index]))
            if not scores:
                continue
            maximum = max(scores)
            # The sink is a virtual slot: it contributes to the denominator
            # only, with a zero value vector.
            denominator = math.exp(sink[head].item() - maximum)
            denominator += sum(math.exp(score - maximum) for score in scores)
            for slot_index, (_, kv_row) in enumerate(slots):
                weight = math.exp(scores[slot_index] - maximum) / denominator
                expected[row, head] += weight * kv_row

    return {
        "name": name,
        "batch": batch,
        "seq": seq,
        "start_pos": start_pos,
        "rows": rows,
        "heads": HEADS,
        "head_dim": DIM,
        "window_rows": window_rows,
        "window_topk": window_topk,
        "compressed_topk": compressed_topk,
        "compressed_capacity": capacity,
        "atol": 2e-6,
        "window_indices": window_idxs.flatten().tolist(),
        "compressed_indices": [value for row in compressed_idxs for value in row],
        "q": q.flatten().tolist(),
        "window_kv": window_kv.flatten().tolist(),
        "compressed_kv": compressed_kv.flatten().tolist(),
        "sink": sink.tolist(),
        "expected": expected.flatten().tolist(),
    }


def make_fixture(path: Path | None):
    rng = torch.Generator().manual_seed(4141018)
    get_window_indices = load_window_indices(path)
    return {
        "model": MODEL,
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "window_cases": window_cases(get_window_indices),
        "attention_cases": [
            gather_case(name, batch, seq, start_pos, window, ratio, topk, capacity, rng, get_window_indices)
            for name, batch, seq, start_pos, window, ratio, topk, capacity in ATTENTION_SPECS
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    fixture = make_fixture(args.reference_path)
    if args.write_fixture:
        metadata = {key: value for key, value in fixture.items() if key not in {"window_cases", "attention_cases"}}
        window_cases_json = ",\n".join(
            "    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
            for case in fixture["window_cases"]
        )
        attention_cases_json = ",\n".join(
            "    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
            for case in fixture["attention_cases"]
        )
        FIXTURE.write_text(
            json.dumps(metadata, indent=2)[:-2]
            + ',\n  "window_cases": [\n'
            + window_cases_json
            + '\n  ],\n  "attention_cases": [\n'
            + attention_cases_json
            + "\n  ]\n}\n",
            encoding="utf-8",
        )
    elif json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
        raise RuntimeError("attention fixtures differ from the pinned reference")
    window_values = sum(len(case["expected_indices"]) for case in fixture["window_cases"])
    attention_values = sum(len(case["expected"]) for case in fixture["attention_cases"])
    print(
        f"PASS: {len(fixture['window_cases'])} window cases ({window_values} indices) and "
        f"{len(fixture['attention_cases'])} gather cases ({attention_values} values); {REVISION}"
    )


if __name__ == "__main__":
    main()
