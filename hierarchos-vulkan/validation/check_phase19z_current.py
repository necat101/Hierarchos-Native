#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XIX-Z current-source receipt.

The checker re-hashes every bound source and binary byte, re-runs every
sub-qualification checker (including the preserved Phase XIX-Y full-attention
receipt), verifies the restored shader and immutable oracle report, enforces
the declared counts and threshold, and requires the committed evidence tree to
match the receipt exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19z/current-qualification.json"
)
SHADER = "hierarchos-vulkan/shaders/transformer_attention_forward_cpu_reciprocal.spv"
SHADER_SHA256 = "f8439dc2a5e94dd39d6e9bf3d537d87b9c25c734e0b216627ba7bafe57dd3226"
DRIFTED_SHA256 = "518a55b2fb84b16df8cbdd0a73068740f4d71b8d3bd16843b105dd04aee76a40"
SHA256 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
SCHEMA = 1
KIND = "phase19z-instrumented-current-source"
EXPECTED_SUBRECEIPTS = ("attention", "training", "stability", "heldout", "drift", "activation")


class QualificationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(
    receipt_path: Path = RECEIPT,
    crate: Path = CRATE,
    *,
    require_binaries: bool = False,
    verify_subs: bool = True,
    require_evidence: bool = True,
) -> dict:
    root = crate.parent
    receipt_path = Path(receipt_path)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing {receipt_path}")
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == SCHEMA, "unexpected receipt schema")
    require(receipt.get("kind") == KIND, "unexpected receipt kind")
    require(receipt.get("status") == "PASS" and receipt.get("pass") is True,
            "receipt is not PASS")

    revision = str(receipt.get("source_revision", ""))
    require(HEX40.fullmatch(revision) is not None, "invalid source revision")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=root, capture_output=True, text=True,
    )
    require(ancestor.returncode == 0, "source revision is not an ancestor of HEAD")
    require(receipt.get("source_clean") is True, "receipt was produced from a dirty worktree")

    source_files = receipt.get("source_files") or {}
    require(len(source_files) >= 1500, "current source binding is incomplete")
    for relative, expected in source_files.items():
        require(isinstance(relative, str) and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts,
                f"unsafe source path {relative!r}")
        require(SHA256.fullmatch(str(expected)) is not None,
                f"invalid source digest for {relative}")
        path = root / relative
        require(path.is_file(), f"missing bound source {relative}")
        require(sha256_file(path) == expected, f"bound source changed: {relative}")
    for required in (
        "hierarchos-vulkan/src/transformer.rs",
        "hierarchos-vulkan/validation/qualify_phase19z_current.py",
        "hierarchos-vulkan/validation/check_phase19z_current.py",
        "hierarchos-vulkan/validation/verify_phase19y_oracles.py",
        SHADER,
    ):
        require(required in source_files, f"required source missing from binding: {required}")
    require(source_files[SHADER] == SHADER_SHA256, "restored shader binding is not f843...")

    binaries = receipt.get("binaries") or {}
    require(binaries, "no executable bindings")
    existing_binaries = 0
    for name, expected in binaries.items():
        require(SHA256.fullmatch(str(expected)) is not None,
                f"invalid binary digest for {name}")
        path = crate / "target/release" / {
            "training_harness": "hierarchos-vulkan-training-qualification.exe",
            "trainer": "hierarchos-vulkan-transformer-train.exe",
            "stability_harness": "training-stability.exe",
            "held_out_eval": "held_out_eval.exe",
            "activation_probe": "activation_probe.exe",
            "wmma_drift": "wmma_drift.exe",
        }[name]
        if path.is_file():
            require(sha256_file(path) == expected, f"binary changed: {name}")
            existing_binaries += 1
        else:
            require(not require_binaries, f"missing required binary {name}")
    require(not require_binaries or existing_binaries == len(binaries),
            "not every required binary was verified")

    shader = receipt.get("shader") or {}
    require(shader.get("path") == SHADER, "wrong shader path")
    require(shader.get("sha256") == SHADER_SHA256, "wrong restored shader hash")
    require(shader.get("sealed_historical_sha256") == SHADER_SHA256,
            "sealed shader hash is not restored")
    require(shader.get("drifted_sha256") == DRIFTED_SHA256,
            "drifted shader provenance changed")
    require(shader.get("inventory") == 45 and shader.get("status_counts") == {"identical": 45},
            "shader inventory is not 45/45 identical")

    oracle = receipt.get("oracle") or {}
    require(oracle.get("schema") == 1 and oracle.get("status") == "PASS",
            "oracle report is not PASS")
    for label, expected in {
        "shape": {
            "archive_sha256": "a45396b752d62b01f75aa0f0cc8a83ee71640bf44a9f83b841e7476056b6ef14",
            "archive_bytes": 32948229,
            "entries": 363,
            "cases": 16,
            "case_files": 176,
            "oracle_source_files": 186,
            "total_files": 363,
        },
        "shared": {
            "archive_sha256": "8a85120ef2a27f9567788c39f4668c6dc52c808e8cae2c179ad8093119d0a95c",
            "archive_bytes": 26077578,
            "entries": 1119,
            "cases": 77,
            "case_files": 842,
            "oracle_source_files": 276,
            "total_files": 1119,
        },
    }.items():
        actual = oracle.get(label) or {}
        for key, value in expected.items():
            require(actual.get(key) == value,
                    f"oracle {label}.{key} changed")
    require("PHASE XIX-Y FULL ATTENTION: PASS" in str(oracle.get("checker", "")),
            "oracle report did not run the Phase XIX-Y checker")

    attention = receipt.get("attention") or {}
    require(HEX40.fullmatch(str(attention.get("source_revision", ""))) is not None,
            "invalid attention source revision")
    require(attention.get("shapes") == 16, "attention shape count is not 16")
    require(attention.get("shared_current") == 74, "attention shared count is not 74")
    require(attention.get("architecture_specific_historical") == 19,
            "attention historical accounting is not 19")
    require(attention.get("total_accounted") == 93, "attention accounting is not 93")
    require(float(attention.get("strict_max_logit", 1.0)) <= 2e-7,
            "strict shape logit threshold violated")
    require(float(attention.get("strict_max_shared", 1.0)) <= 2e-7,
            "strict shared logit threshold violated")
    controls = attention.get("negative_controls") or {}
    require(controls.get("total") == 20 and controls.get("detected") == 20,
            "Phase XIX-Y negative controls are incomplete")
    require(all(row.get("rejected") is True for row in controls.get("controls", [])),
            "a Phase XIX-Y negative control was not rejected")

    counts = receipt.get("counts") or {}
    expected_counts = {
        "training_arms": 8,
        "training_resume": 3,
        "training_performance": 8,
        "stability_arms": 4,
        "stability_resume": 3,
        "stability_performance": 2,
        "heldout_arms": 4,
        "drift_runs": 6,
        "activation_cells": 6,
        "oracle_shape_files": 363,
        "oracle_shared_files": 1119,
    }
    for key, value in expected_counts.items():
        require(counts.get(key) == value, f"count {key} changed: {counts.get(key)!r}")
    require(float(counts.get("selected_lr", 0.0)) == 0.001,
            "selected learning rate is not 0.001")
    packed = receipt.get("packed_cell") or {}
    require(packed.get("path") == "validation/release_parity/multimodal_training/20261112-unified-release/phase19z/packed-cell/status.json",
            "packed-cell status path changed")
    packed_path = crate / packed["path"]
    require(packed_path.is_file(), "missing packed-cell status")
    require(sha256_file(packed_path) == packed.get("sha256"),
            "packed-cell status changed")
    require(packed.get("device", "").startswith("AMD Radeon Graphics"),
            "packed-cell status is not from the AMD adapter")
    require(packed.get("status") in ("PASS", "PARTIAL"),
            "packed-cell status is invalid")
    if packed.get("status") == "PARTIAL":
        require(bool(packed.get("failures")), "packed-cell PARTIAL status has no failures")
    require(counts.get("packed_routes_passed") == len(packed.get("passes") or []),
            "packed-cell pass count changed")
    require(counts.get("packed_routes_failed") == len(packed.get("failures") or []),
            "packed-cell failure count changed")
    require(counts.get("packed_routes_passed") + counts.get("packed_routes_failed") >= 13,
            "packed-cell producer accounting is incomplete")

    subreceipts = receipt.get("subreceipts") or {}
    require(set(subreceipts) == set(EXPECTED_SUBRECEIPTS),
            "subreceipt set changed")
    for name in EXPECTED_SUBRECEIPTS:
        entry = subreceipts[name]
        path = crate / entry["path"]
        require(path.is_file(), f"missing subreceipt {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"subreceipt changed: {name}")

    if verify_subs:
        from check_training_phase16 import verify as verify_training
        from check_training_stability import verify as verify_stability
        from check_heldout_phase17 import verify as verify_heldout
        from check_wmma_drift import verify as verify_drift

        verify_training(crate / "validation/release_parity/multimodal_training/20261112-unified-release/phase19z/training", crate)
        verify_stability(crate / "validation/release_parity/multimodal_training/20261112-unified-release/phase19z/stability", crate)
        verify_heldout(crate / "validation/release_parity/multimodal_training/20261112-unified-release/phase19z/heldout", crate)
        verify_drift(
            crate / "validation/release_parity/multimodal_training/20261112-unified-release/phase19z/drift/current-qualification.json",
            crate, epoch_reuse=True,
        )
        activation = subprocess.run(
            [sys.executable, str(crate / "validation/check_activation_phase19x.py")],
            cwd=crate,
            env={
                **os.environ,
                "HIERARCHOS_ACTIVATION_RECEIPT": str(
                    crate / "validation/release_parity/multimodal_training/20261112-unified-release/phase19z/activation"
                ),
            },
            capture_output=True, text=True,
        )
        require(activation.returncode == 0,
                "activation receipt did not verify: "
                + (activation.stdout + activation.stderr).strip()[-300:])
        attention_check = subprocess.run(
            [sys.executable, str(crate / "validation/check_phase19y_full_attention.py"),
             "--receipt", str(crate / subreceipts["attention"]["path"])],
            cwd=crate, capture_output=True, text=True,
        )
        require(attention_check.returncode == 0,
                "Phase XIX-Y receipt did not verify: "
                + (attention_check.stdout + attention_check.stderr).strip()[-300:])

    evidence = receipt.get("evidence") or {}
    if require_evidence:
        actual = {
            path.relative_to(receipt_path.parent).as_posix(): sha256_file(path)
            for path in sorted(receipt_path.parent.rglob("*"))
            if path.is_file() and path != receipt_path
        }
        require(set(actual) == set(evidence), "evidence tree does not match the receipt")
        for name, expected in evidence.items():
            require(SHA256.fullmatch(str(expected)) is not None,
                    f"invalid evidence digest for {name}")
            require(actual[name] == expected, f"evidence changed: {name}")
        require(len(evidence) > 300, "too few evidence artifacts")

    return {
        "status": "PASS",
        "revision": revision,
        "source_files": len(source_files),
        "binaries": len(binaries),
        "binaries_present": existing_binaries,
        "evidence": len(evidence),
        "counts": counts,
        "shader": SHADER_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--require-binaries", action="store_true")
    args = parser.parse_args()
    try:
        summary = verify(args.receipt, require_binaries=args.require_binaries)
    except (QualificationError, ValueError, KeyError, FileNotFoundError) as error:
        print(f"PHASE XIX-Z CURRENT SOURCE: FAIL ({error})")
        return 1
    print("PHASE XIX-Z CURRENT SOURCE: PASS "
          f"({summary['revision'][:12]}, {summary['source_files']} sources, "
          f"{summary['counts']['training_arms']} training arms, "
          f"{summary['counts']['heldout_arms']} heldout arms, "
          f"{summary['counts']['drift_runs']} drift runs, "
          f"{summary['counts']['activation_cells']} activation cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
