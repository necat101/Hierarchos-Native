#!/usr/bin/env python3
"""Independently verify the sealed V4.1 in-session readback receipt."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CRATE = ROOT.parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> None:
    receipt = json.loads((ROOT / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != 1:
        fail("unsupported receipt schema")
    hashes = json.loads((ROOT / "hashes.json").read_text(encoding="utf-8"))
    if digest(ROOT / "hashes.json") != receipt["hashes_sha256"]:
        fail("hashes.json does not match the sealed receipt")
    for name, expected in hashes.items():
        path = ROOT / name
        if not path.is_file():
            fail(f"missing receipt input {name}")
        if digest(path) != expected:
            fail(f"hash mismatch for {name}")

    commit = receipt["source_commit"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=CRATE,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    benchmark = json.loads((ROOT / "benchmark.json").read_text(encoding="utf-8"))
    gate = json.loads((ROOT / "gate.json").read_text(encoding="utf-8"))
    if benchmark["candidate"]["sha256"] != receipt["candidate_binary_sha256"]:
        fail("candidate binary hash does not match benchmark.json")
    if benchmark["baseline"]["sha256"] != receipt["baseline_binary_sha256"]:
        fail("baseline binary hash does not match benchmark.json")
    if not benchmark["cross_arm_word_exact"]:
        fail("benchmark is not cross-arm word exact")
    if not gate["pass"]:
        fail("clear-regression gate did not pass")
    if "native_vit_4x4" not in gate["inconclusive"]:
        fail("4x4 must remain recorded as inconclusive, not a significant win")
    if not (ROOT / "source.patch").stat().st_size:
        fail("source patch is empty")
    print(
        "PASS: receipt hashes, source commit, binary bindings, exact words and "
        "clear-regression gate verified"
    )


if __name__ == "__main__":
    main()
