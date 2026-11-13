#!/usr/bin/env python3
"""Fail-closed verifier for the current-source two-pass attention receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from emit_two_pass_evidence import parse

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19x" / "attention" / "current-qualification.json"
)
SHA256 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
SOURCE_FILES = (
    "src/transformer.rs",
    "src/lora.rs",
    "src/transformer/peft_state.rs",
    "src/bin/transformer_train.rs",
    "validation/two_pass_backward_equivalence.sh",
    "validation/qualify_attention_phase19x.py",
    "validation/check_attention_phase19x.py",
    "validation/test_attention_phase19x.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path: Path = RECEIPT, crate: Path = CRATE) -> dict:
    require(path.is_file(), f"missing {path}")
    document = load(path)
    require(document.get("schema") == 1, "unexpected attention schema")
    require(document.get("kind") == "phase19x-two-pass-attention-current-source",
            "unexpected attention kind")
    require(document.get("status") == "PASS" and document.get("pass") is True,
            "attention receipt is not a pass")
    require(document.get("target") == "rdna3", "attention target changed")
    revision = str(document.get("source_revision", ""))
    require(HEX40.fullmatch(revision) is not None, "invalid attention revision")
    ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                               revision, "HEAD"], capture_output=True)
    require(ancestor.returncode == 0, "attention revision is not an ancestor of HEAD")
    require(document.get("source_clean") is True, "attention receipt came from a dirty tree")

    sources = document.get("source_files") or {}
    require(set(sources) == set(SOURCE_FILES), "attention source set changed")
    for name, expected in sources.items():
        require(SHA256.fullmatch(str(expected)) is not None, "invalid attention source digest")
        actual = crate / name
        require(actual.is_file(), "missing attention source " + name)
        require(digest(actual) == expected, "attention source changed: " + name)

    binaries = document.get("binaries") or {}
    require(SHA256.fullmatch(str(binaries.get("trainer"))) is not None,
            "missing attention trainer binding")
    trainer = crate / "target/release/hierarchos-vulkan-transformer-train.exe"
    if trainer.is_file():
        require(digest(trainer) == binaries["trainer"], "attention trainer changed")

    backward = document.get("attention_backward") or {}
    require(backward.get("enabled_by") == "HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS",
            "two-pass opt-in contract changed")
    require(backward.get("cases") == 4 and backward.get("comparisons") == 12,
            "attention coverage changed")
    require(backward.get("elements", 0) > 0
            and backward.get("bitwise_equal") == backward.get("elements"),
            "attention gradients are not bit-exact")
    require(backward.get("max_abs") == 0.0 and backward.get("max_ulp") == 0
            and backward.get("unwritten") == 0,
            "attention gradients report an error")
    require(backward.get("adapters_identical") is True,
            "two-pass adapter outputs differ")
    control = (backward.get("dispatches") or {}).get("scalar_arm") or {}
    candidate = (backward.get("dispatches") or {}).get("twopass_arm") or {}
    require(control.get("scalar", 0) > 0 and control.get("query") == 0 and control.get("key") == 0,
            "scalar control did not run alone")
    require(candidate.get("scalar") == 0 and candidate.get("query", 0) > 0
            and candidate.get("key", 0) > 0, "two-pass candidate fell back")
    require("kernels=true" in str(backward.get("diagnostic", ""))
            and "unfused_qk=9" in str(backward.get("diagnostic", "")),
            "two-pass dispatch diagnostic changed")

    evidence = document.get("evidence") or {}
    for name in ("two-pass/test.log", "two-pass/harness.log", "phase16-spirv.json"):
        require(name in evidence, "missing attention evidence " + name)
    for name, expected in evidence.items():
        candidate_path = path.parent / "evidence" / name
        require(candidate_path.is_file(), "missing attention evidence file " + name)
        require(digest(candidate_path) == expected, "attention evidence changed: " + name)
    derived = parse(path.parent / "evidence" / "two-pass/test.log",
                    path.parent / "evidence" / "two-pass/harness.log")
    require(derived["pass"], "attention logs no longer derive a pass")
    derived_cases = len({row["case"] for row in derived["cases"]})
    for key, value in (("cases", derived_cases),
                       ("comparisons", derived["comparisons"]),
                       ("elements", derived["elements"]),
                       ("bitwise_equal", derived["bitwise_equal"]),
                       ("max_abs", derived["max_abs"]), ("max_ulp", derived["max_ulp"]),
                       ("unwritten", derived["unwritten"]),
                       ("adapters_identical", derived["adapters"]["identical"]),
                       ("dispatches", derived["dispatches"]),
                       ("diagnostic", derived["diagnostic"]),
                       ("negative_controls", derived["negative_controls"])):
        require(backward.get(key) == value,
                "attention receipt disagrees with logs: " + key)
    require(derived["negative_controls"] == {"detected": 11, "total": 11},
            "attention negative controls changed")

    shaders = document.get("shaders") or {}
    require(shaders.get("inventory") == 45
            and shaders.get("status_counts") == {"identical": 45},
            "45-artifact shader inventory changed")
    require(shaders.get("historical_artifacts") == 43
            and shaders.get("phase16_artifacts") == 2, "shader scope changed")

    historical = document.get("historical_phase16") or {}
    require(historical.get("shapes") == 16, "historical shape count changed")
    require(sum((historical.get("retained") or {}).values()) == 93,
            "historical retained coverage changed")
    require(HEX40.fullmatch(str(historical.get("source_head"))) is not None,
            "historical Phase XVI head missing")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    try:
        document = verify(args.receipt)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"ATTENTION-XIXX: FAIL ({error})")
        return 1
    print(f"ATTENTION-XIXX: PASS ({document['source_revision'][:12]}, "
          f"{document['attention_backward']['comparisons']} tensors, "
          f"{document['attention_backward']['elements']} elements)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
