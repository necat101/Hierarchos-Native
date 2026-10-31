"""Check native candidate fixtures against DeepSeek's pinned released function.

Only select_candidate_blocks is extracted from the reference AST. The full
model, custom kernels, quantizers, and distributed/CUDA setup are never imported.
PyTorch is a development oracle only; the native runtime uses Rust/Vulkan.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch
from torch.nn import functional as F


MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_candidates.json"


def load_source(path: Path | None):
    if path is None:
        with urlopen(f"https://huggingface.co/{MODEL}/raw/{REVISION}/inference/model.py", timeout=60) as response:
            source = response.read()
    else:
        source = path.read_bytes()
    actual_hash = hashlib.sha256(source).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise RuntimeError(f"reference source hash mismatch: {actual_hash}")
    return source


def load_reference(path: Path | None):
    tree = ast.parse(load_source(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_candidate_blocks"]
    if len(functions) != 1:
        raise RuntimeError("reference must define exactly one select_candidate_blocks")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "deepseek-v41-pinned-candidates", "exec"), namespace)
    return namespace["select_candidate_blocks"]


def make_fixture(reference):
    generator = torch.Generator().manual_seed(1017)
    cases = []
    for width, block_size in [(0, 8), (1, 8), (7, 3), (17, 4), (65, 8), (129, 1), (529, 8)]:
        lengths = sorted({0, min(1, width), width // 2, max(0, width - 1), width})
        logits = torch.rand(len(lengths), width, generator=generator, dtype=torch.float32)
        # Unique maxima avoid depending on PyTorch's unspecified equal-score ordering.
        if width:
            logits += torch.randperm(width, generator=generator).float().unsqueeze(0) * 2.0
        visible = torch.tensor(lengths, dtype=torch.int64).unsqueeze(-1)
        logits.masked_fill_(torch.arange(width).unsqueeze(0) >= visible, -torch.inf)
        for budget in [0, 1, 2, width + 1]:
            expected = reference(logits.clone(), visible, budget, block_size)
            cases.append({
                "name": f"width-{width}-block-{block_size}-budget-{budget}",
                "width": width,
                "block_size": block_size,
                "topk_blocks": budget,
                "visible_lengths": lengths,
                "logits": [None if not torch.isfinite(x) else x.item() for x in logits.flatten()],
                "expected_mask": expected.to(torch.int32).flatten().tolist(),
            })
    # Equal block maxima past 17 blocks. PyTorch's CPU topk picks different tie
    # sets for k=2 (heap staging) and k>=3 (median partition), so this freezes the
    # actual pinned behavior instead of a lower-block-id convention.
    blocks, block_size = 128, 4
    width = blocks * block_size
    tied = torch.full((1, width), -1.0)
    for block in range(blocks):
        tied[0, block * block_size] = 3.0
    tied[0, -1] = 4.0  # pinned newest block keeps the released +inf override
    tied_visible = torch.tensor([[width]], dtype=torch.int64)
    for budget in [1, 2, 3, 4, 5, 17, 18, 64]:
        expected = reference(tied.clone(), tied_visible, budget, block_size)
        cases.append({
            "name": f"tied-blocks-128-budget-{budget}",
            "width": width,
            "block_size": block_size,
            "topk_blocks": budget,
            "visible_lengths": [width],
            "logits": tied.flatten().tolist(),
            "expected_mask": expected.to(torch.int32).flatten().tolist(),
        })
    # Ties at the selection boundary, with the newest block pinned: only two of
    # the three equal maxima fit, and the k=2/k=3 split must stay observable.
    boundary = torch.full((1, width), 1.0)
    for block in (10, 20, 30):
        boundary[0, block * block_size] = 5.0
    for budget in [2, 3, 4]:
        expected = reference(boundary.clone(), tied_visible, budget, block_size)
        cases.append({
            "name": f"tied-boundary-blocks-128-budget-{budget}",
            "width": width,
            "block_size": block_size,
            "topk_blocks": budget,
            "visible_lengths": [width],
            "logits": boundary.flatten().tolist(),
            "expected_mask": expected.to(torch.int32).flatten().tolist(),
        })
    # The newest partial block must beat higher-scoring older blocks. This also
    # covers negative logits, a padded last block, and a non-monotonic ranking.
    logits = torch.tensor([[20., 19., 8., 7., 10., 9., -3., -4., -100., -torch.inf, -torch.inf]])
    visible = torch.tensor([[9]])
    for budget in [1, 2, 3, 8]:
        expected = reference(logits.clone(), visible, budget, 2)
        cases.append({
            "name": f"newest-partial-block-pinned-budget-{budget}",
            "width": 11,
            "block_size": 2,
            "topk_blocks": budget,
            "visible_lengths": [9],
            "logits": [None if not torch.isfinite(x) else x.item() for x in logits.flatten()],
            "expected_mask": expected.to(torch.int32).flatten().tolist(),
        })
    return {"model": MODEL, "revision": REVISION, "source_sha256": SOURCE_SHA256, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, help="optional local copy of the pinned inference/model.py")
    parser.add_argument("--write-fixture", action="store_true", help="regenerate the checked-in oracle fixture")
    args = parser.parse_args()
    fixture = make_fixture(load_reference(args.reference_path))
    if args.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        # Generated numeric arrays stay compact instead of adding tens of
        # thousands of one-number lines to the source tree.
        metadata = {key: value for key, value in fixture.items() if key != "cases"}
        prefix = json.dumps(metadata, indent=2)[:-2]
        cases = ",\n".join("    " + json.dumps(case, allow_nan=False, separators=(",", ":")) for case in fixture["cases"])
        FIXTURE.write_text(prefix + ',\n  "cases": [\n' + cases + "\n  ]\n}\n", encoding="utf-8")
    else:
        if json.loads(FIXTURE.read_text(encoding="utf-8")) != fixture:
            raise RuntimeError("checked-in candidate fixtures differ from the pinned reference")
    elements = sum(len(case["expected_mask"]) for case in fixture["cases"])
    print(f"PASS: {len(fixture['cases'])} reference cases, {elements} exact candidate mask values; {REVISION}")


if __name__ == "__main__":
    main()
