#!/usr/bin/env python3
"""Verify the sealed Chinese-model / sparse-attention audit receipt."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "60a14d9b0bfc382722570e6632873b0a9b2234d8"
LIMIT = 2e-7


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def need(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    need(path.is_file(), f"missing {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def logit_max(obj) -> float:
    values = []
    def walk(node) -> None:
        if isinstance(node, dict):
            if "metric" in node and "max_abs" in node and "logit" in str(node["metric"]).lower():
                values.append(float(node["max_abs"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(obj)
    return max(values, default=0.0)


def check_manifest() -> None:
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    need(len(lines) >= 100, "manifest unexpectedly small")
    names = []
    for line in lines:
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        need(path.is_file(), f"manifest path missing: {relative}")
        need(digest(path) == expected, f"manifest digest mismatch: {relative}")
        names.append(relative)
    for required in ("README.md", "source.json", "receipt.json", "summary.json", "inventory.json", "verify.py"):
        need(required in names, f"unsealed file: {required}")


def main() -> None:
    check_manifest()
    source = load(ROOT / "source.json")
    receipt = load(ROOT / "receipt.json")
    summary = load(ROOT / "summary.json")
    need(source["source_commit"] == SOURCE, "source commit changed")
    need(source["source_tree_clean_at_capture"] is True, "source was not clean")
    need(source["production_numerical_sources_changed"] is False, "production source change claimed")
    need(source["binaries"]["libtest"]["sha256"] == "cc558cc72012bc5bb0362093af6f9a3b5e8bdf07fdda06be3883d24d200dad62", "libtest changed")
    need(source["binaries"]["image_text_pipeline_probe"]["sha256"] == "33fa39728c9084d6c1fc6eb988e2d5fdefc98b8b17c03a8047025e4158e53dc2", "image probe changed")
    need(receipt["verdict"] == "PASS" and summary["verdict"] == "PASS", "receipt not green")
    need(receipt["production_changes"] == [], "production change claimed")
    need(receipt["shared_image"]["cases"] == 37, "shared case count changed")
    need(receipt["shared_image"]["pass"] is True and receipt["shared_image"]["complete"] is True, "shared image not green")
    need(receipt["shared_image"]["absolute_limit"] == LIMIT, "logit limit changed")
    need(receipt["shared_image"]["rtol"] == 0 and receipt["shared_image"]["comparison"] == "strict_less_than", "comparison changed")
    need(receipt["shared_image"]["worst_logit"] == 5.960464477539063e-08, "shared worst logit changed")
    need(receipt["shared_image"]["worst_loss"] < receipt["shared_image"]["loss_absolute_limit"], "loss exceeded canonical gate")
    need(receipt["deepseek"]["active_tests"] == 218 and receipt["deepseek"]["ignored"] == 26, "DeepSeek counts changed")
    need(receipt["deepseek"]["shards"] == 28, "DeepSeek shard count changed")
    need(receipt["deepseek"]["worst_sparse_attention"] == 7.708e-8, "DeepSeek sparse drift changed")
    need(receipt["deepseek"]["worst_compressor"] == 1.7881393432617188e-07, "DeepSeek compressor drift changed")
    need(receipt["native_v41"]["pass"] is True and receipt["native_v41"]["tests"] == 30, "native V4.1 count changed")
    need(receipt["native_v41"]["max_abs_logit"] == 0.0, "native V4.1 logits changed")
    need(receipt["rust"]["full_suite"] == {"passed": 1139, "failed": 0, "ignored": 40}, "full Rust suite changed")
    need(receipt["performance"]["status"] == "NOT_COMPARABLE", "performance claim changed")
    need("no production code changed" in (ROOT / "README.md").read_text(encoding="utf-8").lower(), "README does not state source-only audit")
    need("attempt1-wrong-binary" in receipt["attempts"][0]["preserved_at"], "failed attempt lost")
    need("Unrecognized option: 'model'" in (ROOT / "attempt1-wrong-binary" / "qwen2" / "unmasked.log").read_text(encoding="utf-8", errors="replace"), "failed attempt evidence changed")
    shared = load(ROOT / "shared" / "qualification.json")
    need(shared["pass"] is True and len(shared["results"]) == 37, "shared qualification changed")
    need(logit_max(shared) < LIMIT, "shared logit exceeds strict gate")
    deepseek = load(ROOT / "deepseek218" / "qualification.json")
    need(deepseek["pass"] is True and deepseek["verified_passed"] == 218, "DeepSeek qualification changed")
    native = load(ROOT / "native-v41" / "qualification.json")
    need(native["pass"] is True and native["completed_tests"] == 30, "native qualification changed")
    full_log = (ROOT / "rust" / "full-suite.log").read_text(encoding="utf-8", errors="replace")
    need(re.search(r"test result: ok\. 1139 passed; 0 failed; 40 ignored;", full_log) is not None, "full-suite log changed")
    print("PASS: Chinese-model / sparse-attention audit receipt is source-bound, complete and hash-sealed")


if __name__ == "__main__":
    main()
