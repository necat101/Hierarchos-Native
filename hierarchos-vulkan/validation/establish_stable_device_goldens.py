#!/usr/bin/env python3
"""Add missing device trajectories using only the exact, clean Windows oracle.

Existing device expectations are never replaced. All measurements are staged in
scratch files before publication, and the receipt pins the reference sources and
executable. This is separate from candidate qualification and route tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from check_stable_logits import identity
from check_stable_probe_parity import build_probe
from record_training_state import record
from verify_training_state import SUITES

ORACLE_COMMIT = "2e7893d4e3db17370304a8d67adacab963b22bbc"
ORACLE_SOURCE_SHA256 = "6a9e9dc0c24cac31b56220b9df489be5ec38dcd016a10f606aee138b6077794f"


def require_reference(reference):
    if (reference["commit"] != ORACLE_COMMIT or reference["status"]
            or reference["source_sha256"] != ORACLE_SOURCE_SHA256):
        raise ValueError("goldens require the exact clean Windows oracle and its archived source fingerprint")


def require_environment(environment):
    paths = {"HIERARCHOS_STABLE_ORACLE", "HIERARCHOS_TRANSFORMERS_CHECKOUT",
             "HIERARCHOS_TRANSFORMERS_ROOT", "HIERARCHOS_KIMI_K3_REMOTE_DIR",
             "HIERARCHOS_VALIDATION_PYDEPS"}
    overrides = sorted(key for key in environment if key.startswith("HIERARCHOS_") and key not in paths)
    if overrides:
        raise ValueError(f"unset execution overrides before establishing reference goldens: {', '.join(overrides)}")


def missing_device(document, device):
    present = [device in case["devices"] for case in document["cases"]]
    if any(present) and not all(present):
        raise ValueError("partial device golden: investigate instead of overwriting existing expectations")
    return not any(present)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_environment(os.environ)
    stable = args.stable.resolve()
    reference = identity(stable)
    require_reference(reference)
    binary = build_probe(stable / "hierarchos-vulkan", "debug")
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    pending = []
    rows = []
    with tempfile.TemporaryDirectory(prefix="stable-device-goldens-") as scratch:
        for suite in SUITES:
            original = suite.golden.read_bytes()
            if not missing_device(json.loads(original), args.device):
                continue
            staged = Path(scratch) / suite.golden.name
            staged.write_bytes(original)
            device, cases = record(binary, staged, args.device)
            contents = staged.read_bytes()
            pending.append((suite.golden, original, contents))
            rows.append({"suite": suite.name, "golden": str(suite.golden), "device": device,
                         "policy": json.loads(original).get("env", {}), "cases": cases,
                         "before_sha256": hashlib.sha256(original).hexdigest(),
                         "after_sha256": hashlib.sha256(contents).hexdigest()})
        if identity(stable) != reference or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash:
            raise ValueError("oracle source or executable changed during measurement")
        if any(path.read_bytes() != original for path, original, _ in pending):
            raise ValueError("golden file changed during measurement")
        for path, _, contents in pending:
            path.write_bytes(contents)
    receipt = {"result": "pass", "reference": reference,
               "binary": {"path": str(binary), "sha256": binary_hash}, "added": rows,
               "note": "Only missing device entries were added, from the clean stable executable; existing entries were preserved."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(rows)} missing suite(s) established independently from stable")


if __name__ == "__main__":
    main()
