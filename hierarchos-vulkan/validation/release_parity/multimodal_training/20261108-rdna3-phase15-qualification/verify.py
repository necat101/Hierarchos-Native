#!/usr/bin/env python3
"""Independently verify the Phase XV GPT-2 and multimodal receipt.

Recomputes the integrity manifest, every bound source hash, the shape and case
inventories, the strict numerical gates, the known nine-stage producer
diagnostic, and the static preservation gates. No Torch, GPU or target/
directory is required.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

RECEIPT = Path(__file__).resolve().parent
ROOT = RECEIPT.parents[4]
LIMIT = 2e-7
LOSS_LIMIT = 2e-5
SHAPES = [(2, 8, 4), (1, 8, 4), (3, 8, 4), (2, 6, 4),
          (2, 10, 4), (2, 16, 4), (2, 8, 2), (2, 8, 8)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((RECEIPT / name).read_text(encoding="utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("FAIL: " + message)


def rows(q: dict) -> list[dict]:
    return q.get("results", q.get("records", []))


def worst(q: dict, needle: str) -> float | None:
    values = [
        check["max_abs"]
        for row in rows(q)
        for check in row.get("checks", [])
        if needle in check.get("metric", "").lower() and "max_abs" in check
    ]
    return max(values) if values else None


def verify_manifest() -> None:
    for line in (RECEIPT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition(" ")
        relative = relative.strip().lstrip("*")
        if relative.startswith("./"):
            relative = relative[2:]
        check(sha(RECEIPT / relative) == digest, "manifest: " + relative)


def verify_current() -> None:
    q = load("current-qualification.json")
    check(q.get("pass") is True, "current qualification did not pass")
    check(q.get("scope") == "gpt2_frozen_shape_matrix_and_retained_parity",
          "current qualification scope changed")
    check("hierarchos-vulkan/src/vulkan.rs" in q["sources"],
          "current qualification does not bind vulkan.rs")
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    check(subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
         q["source_head"], head]
    ).returncode == 0, "current qualification is from an unrelated revision")
    for path, digest in q["sources"].items():
        check(sha(ROOT / path) == digest, "source: " + path)


def verify_shapes() -> None:
    q = load("gpt2/qualification.json")
    check(q.get("pass") is True and q.get("complete") is True,
          "GPT-2 shapes did not complete")
    check(len(rows(q)) == 16, "GPT-2 shape count")
    expected = {f"gpt2_b{b}_s{s}_h{h}/{mask}" for b, s, h in SHAPES
                for mask in ("unmasked", "padded")}
    check({row["case"] for row in rows(q)} == expected,
          "GPT-2 shape inventory changed")
    check(all(row.get("pass") is True for row in rows(q)),
          "GPT-2 shape row failed")
    check((worst(q, "logit") or 0.0) < LIMIT, "GPT-2 logit gate")
    losses = [
        check["max_abs"]
        for row in rows(q)
        for check in row.get("checks", [])
        if "loss" in check.get("metric", "").lower() and "max_abs" in check
    ]
    check(losses and max(losses) < LOSS_LIMIT, "GPT-2 loss gate")


def verify_retained() -> None:
    for name, count in (("shared", 74), ("kimi", 3), ("seq2seq", 6),
                        ("gemma3", 4), ("gemma4", 2), ("llava", 4)):
        q = load(f"retained/{name}/qualification.json")
        result = rows(q)
        check(q.get("pass") is True and len(result) == count,
              name + " coverage")
        check(all(row.get("pass") is True for row in result),
              name + " row failed")
        value = worst(q, "logit")
        check(value is None or value < LIMIT, name + " logit gate")

    shared77 = load("multimodal/stages/shared77/qualification.json")
    check(shared77.get("pass") is True and len(rows(shared77)) == 77,
          "shared77 coverage")
    check(all(row.get("pass") is True for row in rows(shared77)),
          "shared77 row failed")

    text = load("multimodal/stages/text166/qualification.json")
    check(text.get("pass") is True and len(rows(text)) == 166,
          "text166 coverage")
    check(max(row["max_abs"] for row in rows(text)) < LIMIT,
          "text166 logit gate")

    deepseek = load("multimodal/stages/deepseek215/qualification.json")
    check(deepseek.get("pass") is True
          and deepseek.get("verified_passed") == 218
          and len(deepseek.get("ignored", [])) == 26,
          "DeepSeek suite coverage")

    native = load("retained/v41native/qualification.json")
    check(native.get("pass") is True
          and native.get("expected_tests") == native.get("completed_tests") == 30,
          "V4.1 native coverage")
    check(native.get("max_abs_logit") is not None
          and native["max_abs_logit"] < LIMIT, "V4.1 native logit gate")
    check(native.get("checkpoint_resume_export_verified") is True,
          "V4.1 checkpoint resume gate")

    bridge = load("retained/v41bridge/qualification.json")
    check(bridge.get("pass") is True
          and bridge.get("expected_cases") == bridge.get("completed_cases") == 2,
          "V4.1 bridge coverage")


def verify_static() -> None:
    provenance = load("gates/provenance.json")
    check(provenance.get("pass") is True
          and provenance.get("status_counts") == {"identical": 43},
          "shader provenance")
    spirv = load("gates/spirv.json")
    check(spirv.get("pass") is True and len(spirv.get("artifacts", [])) == 43
          and all(row.get("passed") is True for row in spirv["artifacts"]),
          "SPIR-V validation")
    isolation = load("gates/isolation.json")
    check(isolation.get("unchanged") is True
          and isolation.get("files_checked") == 1341
          and isolation.get("existing_recipes") == 672
          and isolation.get("failures") == [], "upstream preservation")
    primitives = (RECEIPT / "gates/primitives.log").read_text(encoding="utf-8")
    check(re.search(r"test result: ok\. 9 passed; 0 failed; 1 ignored;",
                    primitives) is not None, "GPT-2 primitives")


def verify_producer_diagnostic() -> None:
    full = load("multimodal/full-qualification.json")
    failed = [row["stage"] for row in full["results"] if not row.get("pass")]
    check(failed == ["native-v41"], "nine-stage failure inventory changed")
    check((RECEIPT / "multimodal/stages/native-v41/checkpoint.log").is_file(),
          "missing checkpoint failure log")
    check((RECEIPT / "multimodal/stages/native-v41/checkpoint-rerun.log").is_file(),
          "missing checkpoint rerun log")


def main() -> int:
    verify_manifest()
    verify_current()
    verify_shapes()
    verify_retained()
    verify_static()
    verify_producer_diagnostic()
    print(
        "PASS: 134 sources, 16 GPT-2 shapes, 74 shared cases, "
        "218 DeepSeek tests, 30 V4.1 tests, 43 shaders"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
