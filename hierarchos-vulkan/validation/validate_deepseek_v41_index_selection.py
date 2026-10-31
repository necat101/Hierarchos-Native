"""Qualify V4.1 fine-index selection against the pinned reference semantics.

The selection is the inline tail of the released `Indexer.forward`:

    topk = min(index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()

This script copies only those lines, masks unreachable positions and optional
candidate blocks exactly as the model does, and writes the resulting indices.
It deliberately does not generate the indexer projections/scores, whose
reduction-order parity is a separate, currently unresolved issue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_index_selection.json"


def load_source(path: Path | None) -> bytes:
    if path is None:
        with urlopen(f"https://huggingface.co/{MODEL}/raw/{REVISION}/inference/model.py", timeout=60) as response:
            source = response.read()
    else:
        source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"reference source hash mismatch: {actual}")
    return source


def make_cases(rng: torch.Generator):
    specs = [
        # name, batch, seq, start_pos, ratio, topk, offset
        ("prefill-4", 1, 4, 0, 2, 3, 5),
        ("prefill-8", 2, 8, 0, 2, 4, 7),
        ("decode-7", 2, 1, 7, 2, 3, 11),
        ("chunk-4-plus-3", 1, 3, 4, 2, 2, 1),
        ("ratio-1", 1, 5, 0, 1, 3, 2),
        ("wide", 1, 6, 0, 2, 6, 0),
    ]
    cases = []
    for name, batch, seq, start_pos, ratio, topk_setting, offset in specs:
        rows = batch * seq
        width = (start_pos + seq) // ratio
        if width == 0:
            continue
        topk = min(topk_setting, width)
        scores = torch.randn(rows, width, generator=rng, dtype=torch.float32)
        # Unique values avoid depending on unspecified equal-score top-k order.
        scores += torch.randperm(rows * width, generator=rng).float().view(rows, width) * 1e-4
        # start_pos == 0 masks each prefill query causally; decode/chunked calls
        # use the single end length, exactly like Indexer.forward.
        if start_pos == 0:
            visible_values = [(row % seq + 1) // ratio for row in range(rows)]
        else:
            visible_values = [(start_pos + seq) // ratio for _ in range(rows)]
        visible = torch.tensor(visible_values, dtype=torch.int64).unsqueeze(-1)
        scores.masked_fill_(torch.arange(width).unsqueeze(0) >= visible, -torch.inf)
        candidates = torch.ones(rows, width, dtype=torch.bool)
        if name in {"prefill-8", "wide"}:
            # Keep a strict subset of reachable positions for half the rows.
            for row in range(0, rows, 2):
                keep = (torch.arange(width) + row) % 3 != 1
                candidates[row] &= keep
                scores[row].masked_fill_(~candidates[row], -torch.inf)
        indices = scores.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        expected = torch.where(indices < visible, indices + offset, -1).to(torch.int32)
        cases.append({
            "name": name,
            "batch": batch,
            "seq": seq,
            "rows": rows,
            "width": width,
            "start_pos": start_pos,
            "ratio": ratio,
            "topk": topk,
            "offset": offset,
            "scores": [None if not torch.isfinite(v) else v.item() for v in scores.flatten()],
            "expected_indices": expected.flatten().tolist(),
        })
    return cases


def make_fixture():
    rng = torch.Generator().manual_seed(4141017)
    return {"model": MODEL, "revision": REVISION, "source_sha256": SOURCE_SHA256,
            "cases": make_cases(rng)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    load_source(args.reference_path)
    fixture = make_fixture()
    if args.write_fixture:
        metadata = {key: value for key, value in fixture.items() if key != "cases"}
        prefix = json.dumps(metadata, indent=2)[:-2]
        cases = ",\n".join(
            "    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
            for case in fixture["cases"]
        )
        FIXTURE.write_text(prefix + ',\n  "cases": [\n' + cases + "\n  ]\n}\n", encoding="utf-8")
    elif json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
        raise RuntimeError("index selection fixtures differ from the pinned reference")
    values = sum(len(case["expected_indices"]) for case in fixture["cases"])
    print(f"PASS: {len(fixture['cases'])} index selection cases, {values} exact indices; {REVISION}")


if __name__ == "__main__":
    main()
