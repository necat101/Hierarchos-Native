#!/usr/bin/env python3
"""Assemble the Phase XIX-S current-source receipt from real sub-qualifications.

This producer never invents a numerical result. It requires the attention,
training, stability, held-out, drift and activation receipts to exist, verifies
each one with its own checker, then binds the current source, executables,
shader inventory, mirror-epoch evidence and every evidence byte into one
versioned receipt. A missing or failing sub-qualification aborts assembly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261110-phase19s-current-source"
)
SUBDIRS = {
    "attention": RECEIPT / "attention",
    "training": RECEIPT / "training",
    "stability": RECEIPT / "stability",
    "heldout": RECEIPT / "heldout",
    "drift": RECEIPT / "drift",
    "activation": RECEIPT / "activation",
}
EXTRA_SOURCES = (
    "hierarchos-vulkan/validation/qualify_phase19s_current.py",
    "hierarchos-vulkan/validation/check_phase19s_current.py",
    "hierarchos-vulkan/validation/test_phase19s_current.py",
    "hierarchos-vulkan/validation/qualify_activation_phase19r.py",
    "hierarchos-vulkan/validation/check_activation_phase19r.py",
    "hierarchos-vulkan/validation/test_activation_phase19r.py",
    "hierarchos-vulkan/validation/qualify_wmma_drift.py",
    "hierarchos-vulkan/validation/verify_rdna3_multimodal_isolation.py",
    "hierarchos-vulkan/validation/check_training_phase16.py",
    "hierarchos-vulkan/validation/check_training_stability.py",
    "hierarchos-vulkan/validation/check_heldout_phase17.py",
    "hierarchos-vulkan/validation/check_wmma_drift.py",
)


def require(condition, message):
    if not condition:
        raise SystemExit(f"PHASE XIX-S PRODUCER: FAIL: {message}")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    revision = git("rev-parse", "HEAD")
    require(len(revision) == 40, "cannot resolve the tested revision")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    require(not dirty or args.allow_dirty,
            "tracked worktree must be clean before assembly:\n" + dirty)

    from check_phase16_receipts import verify_receipt
    from check_training_phase16 import verify as verify_training
    from check_training_stability import verify as verify_stability
    from check_heldout_phase17 import verify as verify_heldout
    from check_wmma_drift import verify as verify_drift

    attention_path = SUBDIRS["attention"] / "current-qualification.json"
    require(attention_path.is_file(), f"missing {relative(attention_path)}")
    attention = load(attention_path)
    verify_receipt(attention, SUBDIRS["attention"], source_head=revision)

    training = verify_training(SUBDIRS["training"], CRATE)
    stability = verify_stability(SUBDIRS["stability"], CRATE)
    heldout = verify_heldout(SUBDIRS["heldout"], CRATE)
    drift = verify_drift(
        SUBDIRS["drift"] / "current-qualification.json", CRATE, epoch_reuse=True
    )

    activation_path = SUBDIRS["activation"] / "current-qualification.json"
    require(activation_path.is_file(), f"missing {relative(activation_path)}")
    activation_check = subprocess.run(
        [sys.executable, str(CRATE / "validation/check_activation_phase19r.py")],
        cwd=CRATE,
        env={**os.environ, "HIERARCHOS_ACTIVATION_RECEIPT": str(SUBDIRS["activation"])},
        capture_output=True,
        text=True,
    )
    require(activation_check.returncode == 0,
            "activation receipt did not verify: " + (activation_check.stdout
            + activation_check.stderr).strip()[-400:])
    activation = load(activation_path)

    anomaly = activation["anomaly"]
    require(abs(anomaly["post_fix_in_process_step1"] - anomaly["fresh_step1_mixed"]) <= 1e-3,
            "mirror epoch regression does not agree with fresh evaluation")
    require(anomaly["pre_fix_in_process_step1"] - anomaly["fresh_step1_mixed"] > 0.2,
            "mirror epoch regression no longer reproduces the stale-mirror lag")
    require(anomaly["step100_fresh_mixed"] - anomaly["step100_fresh_fp32arm"] > 0.05,
            "step-100 mixed/FP32 drift was not reproduced")
    require(abs(anomaly["step100_in_process_mixed"] - anomaly["step100_fresh_mixed"]) <= 5e-3,
            "step-100 in-process evaluation differs from a fresh process")

    source_files = dict(attention["sources"])
    # Sub-receipts bind crate-relative source paths; fold them into the
    # repository-relative Phase XIX-S manifest so the tracer, mirror-epoch
    # logic, trainers and harnesses are all covered by one verifier.
    for sub in (training, stability, heldout, drift, activation):
        for name in (sub.get("source_files") or {}):
            repository_name = name if (ROOT / name).is_file() else "hierarchos-vulkan/" + name
            require((ROOT / repository_name).is_file(), "missing sub-receipt source " + name)
            source_files[repository_name] = digest(ROOT / repository_name)
    for name in EXTRA_SOURCES:
        path = ROOT / name
        require(path.is_file(), "missing Phase XIX-S source " + name)
        source_files[name] = digest(path)
    for name in (
        "hierarchos-vulkan/validation/qualify_training_phase16.py",
        "hierarchos-vulkan/validation/qualify_training_stability.py",
        "hierarchos-vulkan/validation/qualify_heldout_phase17.py",
    ):
        source_files[name] = digest(ROOT / name)
    require("hierarchos-vulkan/src/transformer.rs" in source_files,
            "instrumented transformer is not bound")

    binaries = dict(attention["binaries"])
    binaries.update({
        "activation_probe": digest(CRATE / "target/release/activation_probe.exe"),
        "wmma_drift": digest(CRATE / "target/release/wmma_drift.exe"),
        "training_harness": digest(
            CRATE / "target/release/hierarchos-vulkan-training-qualification.exe"),
        "trainer": digest(CRATE / "target/release/hierarchos-vulkan-transformer-train.exe"),
        "stability_harness": digest(CRATE / "target/release/training-stability.exe"),
        "held_out_eval": digest(CRATE / "target/release/held_out_eval.exe"),
    })

    subreceipts = {}
    for name, directory in SUBDIRS.items():
        path = directory / "current-qualification.json"
        require(path.is_file(), f"missing sub-receipt {relative(path)}")
        subreceipts[name] = {"path": relative(path), "sha256": digest(path)}

    evidence = {}
    for path in sorted(RECEIPT.rglob("*")):
        if path.is_file() and path != RECEIPT / "current-qualification.json":
            evidence[path.relative_to(RECEIPT).as_posix()] = digest(path)
    require(len(evidence) > 100, "too few Phase XIX-S evidence artifacts")

    receipt = {
        "schema": 1,
        "kind": "phase19s-instrumented-current-source",
        "status": "PASS",
        "pass": True,
        "target": "rdna3",
        "scope": "phase19s_instrumented_transformer_requalification",
        "source_revision": revision,
        "source_clean": not dirty,
        "source_files": dict(sorted(source_files.items())),
        "binaries": binaries,
        "subreceipts": subreceipts,
        "shaders": {
            "inventory": attention["shaders"]["inventory"],
            "status_counts": attention["shaders"]["status_counts"],
            "historical_artifacts": attention["shaders"]["historical"]["artifacts"],
            "phase16_artifacts": attention["shaders"]["phase16"]["artifacts"],
            "sealed_gate_sha256": attention["shaders"]["historical"]["sealed_gate_sha256"],
        },
        "mirror_epoch": anomaly,
        "counts": {
            "shapes": attention["shapes"]["cases"],
            "retained": attention["retained"],
            "attention_backward": {
                "cases": attention["attention_backward"]["cases"],
                "comparisons": attention["attention_backward"]["comparisons"],
                "elements": attention["attention_backward"]["elements"],
                "bitwise_equal": attention["attention_backward"]["bitwise_equal"],
                "max_abs": attention["attention_backward"]["max_abs"],
                "max_ulp": attention["attention_backward"]["max_ulp"],
            },
            "training_arms": training["arms"],
            "resume_arms": len(training["resume_arms"]),
            "stability_arms": stability["arms"],
            "heldout_arms": heldout["arms"],
            "drift_runs": drift["runs"],
            "activation_tensors": activation["protocol"]["trace_ops"],
        },
        "evidence": evidence,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Raw RDNA3 captures and parameter dumps stay under target/ and are bound by hash, not committed.",
            "CPU CI re-verifies committed evidence; it does not rerun the GPU matrix.",
            "Historical Phase XV-XIX receipts are preserved unchanged and are not current-source evidence.",
            "Full-parameter exact resume remains unsupported; only LoRA resume is qualified.",
        ],
    }
    RECEIPT.mkdir(parents=True, exist_ok=True)
    out = RECEIPT / "current-qualification.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE XIX-S PRODUCER: PASS ({len(source_files)} sources, "
          f"{len(evidence)} evidence files, {revision[:12]})")
    print(f"receipt={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
