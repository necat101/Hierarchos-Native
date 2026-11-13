#!/usr/bin/env python3
"""Assemble the Phase XIX-Z current-source receipt from real qualifications.

The producer refuses to invent results: it re-verifies the current-source
training, stability, held-out, drift and activation receipts, re-runs the
Phase XIX-Y full-attention checker, binds the immutable oracle verification
report, the restored shader, every exercised executable and every evidence
byte into one versioned receipt. A missing or failing sub-qualification aborts
assembly. Historical Phase XIX-X/Y receipts are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19z"
)
SUBDIRS = {
    "training": RECEIPT / "training",
    "stability": RECEIPT / "stability",
    "heldout": RECEIPT / "heldout",
    "drift": RECEIPT / "drift",
    "activation": RECEIPT / "activation",
}
ATTENTION = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19y/current-qualification.json"
)
ORACLE_REPORT = RECEIPT / "oracle-verification.json"
SHADER = "hierarchos-vulkan/shaders/transformer_attention_forward_cpu_reciprocal.spv"
SHADER_SHA256 = "f8439dc2a5e94dd39d6e9bf3d537d87b9c25c734e0b216627ba7bafe57dd3226"
ORACLE_SHAPE = {
    "archive_sha256": "a45396b752d62b01f75aa0f0cc8a83ee71640bf44a9f83b841e7476056b6ef14",
    "archive_bytes": 32948229,
    "entries": 363,
    "cases": 16,
    "case_files": 176,
    "oracle_source_files": 186,
}
ORACLE_SHARED = {
    "archive_sha256": "8a85120ef2a27f9567788c39f4668c6dc52c808e8cae2c179ad8093119d0a95c",
    "archive_bytes": 26077578,
    "entries": 1119,
    "cases": 77,
    "case_files": 842,
    "oracle_source_files": 276,
}
EXTRA_SOURCES = (
    "hierarchos-vulkan/validation/qualify_phase19z_current.py",
    "hierarchos-vulkan/validation/check_phase19z_current.py",
    "hierarchos-vulkan/validation/test_phase19z_current.py",
    "hierarchos-vulkan/validation/verify_phase19y_oracles.py",
    "hierarchos-vulkan/validation/test_verify_phase19y_oracles.py",
    "hierarchos-vulkan/validation/check_phase19y_full_attention.py",
    "hierarchos-vulkan/validation/qualify_activation_phase19x.py",
    "hierarchos-vulkan/validation/check_activation_phase19x.py",
    "hierarchos-vulkan/validation/test_activation_phase19x.py",
    "hierarchos-vulkan/validation/qualify_wmma_drift.py",
    "hierarchos-vulkan/validation/check_wmma_drift.py",
    "hierarchos-vulkan/validation/qualify_training_phase16.py",
    "hierarchos-vulkan/validation/qualify_training_stability.py",
    "hierarchos-vulkan/validation/qualify_heldout_phase17.py",
    "hierarchos-vulkan/validation/check_training_phase16.py",
    "hierarchos-vulkan/validation/check_training_stability.py",
    "hierarchos-vulkan/validation/check_heldout_phase17.py",
    "hierarchos-vulkan/validation/verify_rdna3_multimodal_isolation.py",
    ".github/workflows/phase19y-oracles.yml",
)


def require(condition, message):
    if not condition:
        raise SystemExit(f"PHASE XIX-Z PRODUCER: FAIL: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.relative_to(CRATE).as_posix()


def main() -> int:
    global ATTENTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-receipt", type=Path, default=ATTENTION)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    ATTENTION = args.attention_receipt.resolve()
    require(not (RECEIPT / "current-qualification.json").exists(), "refusing to overwrite a receipt")

    revision = git("rev-parse", "HEAD")
    require(len(revision) == 40, "cannot resolve the tested revision")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    require(not dirty or args.allow_dirty,
            "tracked worktree must be clean before assembly:\n" + dirty)

    from check_training_phase16 import verify as verify_training
    from check_training_stability import verify as verify_stability
    from check_heldout_phase17 import verify as verify_heldout
    from check_wmma_drift import verify as verify_drift

    training = verify_training(SUBDIRS["training"], CRATE)
    stability = verify_stability(SUBDIRS["stability"], CRATE)
    heldout = verify_heldout(SUBDIRS["heldout"], CRATE)
    drift = verify_drift(
        SUBDIRS["drift"] / "current-qualification.json", CRATE, epoch_reuse=True
    )

    activation_check = subprocess.run(
        [sys.executable, str(CRATE / "validation/check_activation_phase19x.py")],
        cwd=CRATE,
        env={**os.environ, "HIERARCHOS_ACTIVATION_RECEIPT": str(SUBDIRS["activation"])},
        capture_output=True,
        text=True,
    )
    require(activation_check.returncode == 0,
            "activation receipt did not verify: " + (activation_check.stdout
            + activation_check.stderr).strip()[-400:])
    activation = load(SUBDIRS["activation"] / "current-qualification.json")

    attention_check = subprocess.run(
        [sys.executable, str(CRATE / "validation/check_phase19y_full_attention.py"),
         "--receipt", str(ATTENTION)],
        cwd=CRATE, capture_output=True, text=True,
    )
    require(attention_check.returncode == 0,
            "Phase XIX-Y full-attention receipt did not verify: "
            + (attention_check.stdout + attention_check.stderr).strip()[-400:])
    attention = load(ATTENTION)
    require(attention.get("status") == "PASS" and attention.get("pass") is True,
            "Phase XIX-Y receipt is not passing")
    require(len(attention["source_revision"]) == 40,
            "Phase XIX-Y source revision is invalid")

    require(ORACLE_REPORT.is_file(), f"missing {relative(ORACLE_REPORT)}")
    oracle = load(ORACLE_REPORT)
    require(oracle.get("schema") == 1 and oracle.get("status") == "PASS",
            "oracle verification report is not PASS")
    for label, expected in (("shape", ORACLE_SHAPE), ("shared", ORACLE_SHARED)):
        actual = oracle.get(label) or {}
        for key, value in expected.items():
            require(actual.get(key) == value,
                    f"oracle {label}.{key} changed: {actual.get(key)!r} != {value!r}")
    require("PHASE XIX-Y FULL ATTENTION: PASS" in oracle.get("checker", ""),
            "oracle report did not run the Phase XIX-Y checker")

    source_files = {}
    # The Phase XIX-Y receipt binds the full 1,551-file source/shaders/validation
    # inventory. Recompute every byte against this checkout; historical paths
    # contribute only their names.
    for name, _ in attention["source_files"].items():
        path = ROOT / name
        require(path.is_file(), "missing Phase XIX-Y source " + name)
        source_files[name] = digest(path)
    for sub in (training, stability, heldout, drift, activation):
        for name in (sub.get("source_files") or {}):
            repository_name = name if (ROOT / name).is_file() else "hierarchos-vulkan/" + name
            path = ROOT / repository_name
            require(path.is_file(), "missing sub-receipt source " + name)
            source_files[repository_name] = digest(path)
    for name in EXTRA_SOURCES:
        path = ROOT / name
        require(path.is_file(), "missing Phase XIX-Z source " + name)
        source_files[name] = digest(path)
    require(len(source_files) >= 1500, "current source binding is incomplete")
    require("hierarchos-vulkan/src/transformer.rs" in source_files,
            "instrumented transformer is not bound")
    require(SHADER in source_files, "restored shader is not bound")
    require(source_files[SHADER] == SHADER_SHA256,
            "restored shader hash changed: " + source_files[SHADER])

    binaries = {}
    for name, filename in {
        "training_harness": "hierarchos-vulkan-training-qualification.exe",
        "trainer": "hierarchos-vulkan-transformer-train.exe",
        "stability_harness": "training-stability.exe",
        "held_out_eval": "held_out_eval.exe",
        "activation_probe": "activation_probe.exe",
        "wmma_drift": "wmma_drift.exe",
    }.items():
        path = CRATE / "target/release" / filename
        require(path.is_file(), "missing current binary " + filename)
        binaries[name] = digest(path)

    subreceipts = {"attention": {"path": relative(ATTENTION), "sha256": digest(ATTENTION)}}
    for name, directory in SUBDIRS.items():
        path = directory / "current-qualification.json"
        require(path.is_file(), f"missing sub-receipt {relative(path)}")
        subreceipts[name] = {"path": relative(path), "sha256": digest(path)}

    # The independently executed verifier report was required above. Never
    # replace it with a report from an unrelated scratch run during assembly.

    evidence = {}
    for path in sorted(RECEIPT.rglob("*")):
        if path.is_file() and path != RECEIPT / "current-qualification.json":
            evidence[path.relative_to(RECEIPT).as_posix()] = digest(path)
    require(len(evidence) > 300, "too few Phase XIX-Z evidence artifacts")

    packed_status_path = RECEIPT / "packed-cell/status.json"
    require(packed_status_path.is_file(), "missing packed-cell status evidence")
    packed_status = load(packed_status_path)
    require(packed_status.get("kind") == "phase19z-packed-cell-amd-local",
            "packed-cell status has the wrong kind")
    from check_phase19z_current import verify_packed_status
    verify_packed_status(packed_status,packed_status_path.parent,CRATE,run_guards=True)

    six_cell = (activation.get("six_cell") or {}).get("cells") or {}
    training_doc = load(SUBDIRS["training"] / "current-qualification.json")
    stability_doc = load(SUBDIRS["stability"] / "current-qualification.json")
    heldout_doc = load(SUBDIRS["heldout"] / "current-qualification.json")
    drift_doc = load(SUBDIRS["drift"] / "current-qualification.json")
    receipt = {
        "schema": 1,
        "kind": "phase19z-instrumented-current-source",
        "status": "PASS",
        "pass": True,
        "target": "rdna3",
        "scope": "phase19z_restored_shader_requalification",
        "source_revision": revision,
        "source_clean": not dirty,
        "source_files": dict(sorted(source_files.items())),
        "binaries": binaries,
        "subreceipts": subreceipts,
        "shader": {
            "path": SHADER,
            "sha256": SHADER_SHA256,
            "sealed_historical_sha256": SHADER_SHA256,
            "drifted_sha256": "518a55b2fb84b16df8cbdd0a73068740f4d71b8d3bd16843b105dd04aee76a40",
            "inventory": attention["shader"]["inventory"],
            "status_counts": attention["shader"]["status_counts"],
        },
        "attention": {
            "path": relative(ATTENTION),
            "sha256": digest(ATTENTION),
            "source_revision": attention["source_revision"],
            "shapes": attention["historical_phase16"]["shapes"],
            "shared_current": attention["retained"]["coverage"]["shared_current"],
            "architecture_specific_historical": attention["retained"]["coverage"]["architecture_specific_historical"],
            "total_accounted": attention["retained"]["coverage"]["total_accounted"],
            "strict_max_logit": attention["shapes"]["max_abs"],
            "strict_max_shared": attention["retained"]["shared_current"]["max_abs"],
            "negative_controls": attention["negative_controls"],
        },
        "oracle": oracle,
        "packed_cell": {
            "path": relative(packed_status_path),
            "sha256": digest(packed_status_path),
            "status": packed_status["status"],
            "device": packed_status["device"],
            "adapter": packed_status["adapter"],
            "ci": packed_status["ci"],
            "passes": packed_status["passes"],
            "failures": packed_status["failures"],
        },
        "counts": {
            "training_arms": len(training_doc["arms"]),
            "training_resume": len(training_doc["resume"]),
            "training_performance": len(training_doc["performance"]),
            "stability_arms": len(stability_doc["arms"]),
            "stability_resume": len(stability_doc["resume"]),
            "stability_performance": len(stability_doc["performance"]),
            "selected_lr": stability_doc["sweep"]["selected_lr"],
            "heldout_arms": len(heldout_doc["arms"]),
            "drift_runs": len(drift_doc["runs"]),
            "activation_cells": len(six_cell),
            "packed_routes_passed": len(packed_status.get("passes") or []),
            "packed_routes_failed": len(packed_status.get("failures") or []),
            "oracle_shape_files": oracle["shape"]["total_files"],
            "oracle_shared_files": oracle["shared"]["total_files"],
        },
        "evidence": evidence,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Raw RDNA3 captures and parameter dumps stay under target/ and are bound by hash, not committed.",
            "CPU CI re-verifies committed evidence and immutable oracle archives; it does not rerun the GPU matrix.",
            "Historical Phase XIX-X receipts use shader 518a... and remain preserved, non-current evidence.",
            "The Phase XIX-Y full-attention receipt is preserved unchanged; Phase XIX-Z binds it as a prerequisite.",
            "Full-parameter exact resume remains unsupported; only LoRA resume is qualified.",
            "A queued packed-cell GitHub workflow is not a passing hardware result and is not counted as one.",
        ],
    }
    RECEIPT.mkdir(parents=True, exist_ok=True)
    out = RECEIPT / "current-qualification.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE XIX-Z PRODUCER: PASS ({len(source_files)} sources, "
          f"{len(evidence)} evidence files, {revision[:12]})")
    print(f"receipt={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
