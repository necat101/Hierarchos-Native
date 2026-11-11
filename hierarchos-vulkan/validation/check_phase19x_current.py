#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XIX-X instrumented current-source receipt."""

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
    / "20261111-phase19x-current-source" / "current-qualification.json"
)
SHA256 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
SUBDIRS = ("attention", "training", "stability", "heldout", "drift", "activation")
BINARY_FILES = {
    "image_text_pipeline_probe": "image_text_pipeline_probe.exe",
    "gemma3_vision_probe": "gemma3_vision_probe.exe",
    "gemma4_vision_probe": "gemma4_vision_probe.exe",
    "llava_pipeline_probe": "llava_pipeline_probe.exe",
    "activation_probe": "activation_probe.exe",
    "wmma_drift": "wmma_drift.exe",
    "training_harness": "hierarchos-vulkan-training-qualification.exe",
    "trainer": "hierarchos-vulkan-transformer-train.exe",
    "stability_harness": "training-stability.exe",
    "held_out_eval": "held_out_eval.exe",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def verify(path: Path = RECEIPT, crate: Path = CRATE, binary_root: Path | None = None) -> dict:
    require(path.is_file(), f"missing {path}")
    document = load(path)
    require(document.get("schema") == 1, "unexpected Phase XIX-S schema")
    require(document.get("kind") == "phase19x-instrumented-current-source",
            "unexpected Phase XIX-X kind")
    require(document.get("status") == "PASS" and document.get("pass") is True,
            "Phase XIX-S receipt is not a pass")
    require(document.get("target") == "rdna3", "target changed")

    revision = str(document.get("source_revision", ""))
    require(HEX40.fullmatch(revision) is not None, "invalid source revision")
    ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                               revision, "HEAD"], capture_output=True)
    require(ancestor.returncode == 0, "source revision is not an ancestor of HEAD")
    require(document.get("source_clean") is True, "receipt came from a dirty worktree")

    sources = document.get("source_files") or {}
    require(len(sources) >= 140, "Phase XIX-S binds too few sources")
    require("hierarchos-vulkan/src/transformer.rs" in sources,
            "instrumented transformer is not bound")
    for name, expected in sources.items():
        require(SHA256.fullmatch(str(expected)) is not None, "invalid source digest")
        actual = crate.parent / name
        require(actual.is_file(), "missing bound source " + name)
        require(digest(actual) == expected, "bound source changed: " + name)

    binaries = document.get("binaries") or {}
    for name in ("image_text_pipeline_probe", "gemma3_vision_probe", "gemma4_vision_probe",
                 "llava_pipeline_probe", "activation_probe", "wmma_drift", "training_harness",
                 "trainer", "stability_harness", "held_out_eval"):
        require(SHA256.fullmatch(str(binaries.get(name))) is not None,
                "missing or malformed binary binding: " + name)
    root = Path(binary_root) if binary_root else crate / "target/release"
    for name, filename in BINARY_FILES.items():
        candidate = root / filename
        if candidate.is_file():
            require(digest(candidate) == binaries[name], "binary hash changed: " + name)

    subreceipts = document.get("subreceipts") or {}
    require(set(subreceipts) == set(SUBDIRS), "sub-receipt set changed")
    resolved = {}
    for name in SUBDIRS:
        entry = subreceipts[name]
        candidate = crate / entry["path"]
        require(candidate.is_file(), f"missing sub-receipt {name}")
        require(digest(candidate) == entry["sha256"], f"sub-receipt changed: {name}")
        resolved[name] = candidate

    from check_attention_phase19x import verify as verify_attention
    from check_training_phase16 import verify as verify_training
    from check_training_stability import verify as verify_stability
    from check_heldout_phase17 import verify as verify_heldout
    from check_wmma_drift import verify as verify_drift

    head = git("rev-parse", "HEAD")
    verify_attention(resolved["attention"], crate)
    training = verify_training(resolved["training"].parent, crate)
    stability = verify_stability(resolved["stability"].parent, crate)
    heldout = verify_heldout(resolved["heldout"].parent, crate)
    drift = verify_drift(resolved["drift"], crate, epoch_reuse=True)

    activation = subprocess.run(
        [sys.executable, str(crate / "validation/check_activation_phase19x.py")],
        cwd=crate,
        env={**os.environ, "HIERARCHOS_ACTIVATION_RECEIPT":
             str(resolved["activation"].parent)},
        capture_output=True, text=True,
    )
    require(activation.returncode == 0, "activation qualification failed: "
            + (activation.stdout + activation.stderr).strip()[-400:])
    activation_doc = load(resolved["activation"])

    counts = document.get("counts") or {}
    historical = counts.get("historical_attention") or {}
    require(historical.get("shapes") == 16, "historical GPT-2 shape coverage changed")
    require(sum((historical.get("retained") or {}).values()) == 93,
            "historical retained-family coverage changed")
    backward = counts["attention_backward"]
    require(backward["cases"] == 4 and backward["comparisons"] == 12,
            "attention-backward coverage changed")
    require(backward["elements"] > 0 and backward["bitwise_equal"] == backward["elements"],
            "attention-backward results are not bit-exact")
    require(backward["max_abs"] == 0.0 and backward["max_ulp"] == 0,
            "attention-backward reports an error")
    require(counts["training_arms"] == 8 and counts["resume_arms"] == 3,
            "training/resume arm coverage changed")
    require(counts["stability_arms"] == 4, "stability arm coverage changed")
    require(counts["heldout_arms"] == 4, "held-out arm coverage changed")
    require(counts["drift_runs"] == 6, "drift run coverage changed")
    require(len(counts["activation_tensors"]) == 27, "activation tensor inventory changed")

    shaders = document.get("shaders") or {}
    require(shaders.get("inventory") == 45
            and shaders.get("status_counts") == {"identical": 45},
            "45-artifact shader inventory changed")
    require(shaders.get("historical_artifacts") == 43
            and shaders.get("phase16_artifacts") == 2,
            "shader scope changed")

    mirror = document.get("mirror_epoch") or {}
    source_mirror = activation_doc["mirror_epoch"]
    for key, value in mirror.items():
        if key in source_mirror:
            require(source_mirror[key] == value,
                    "mirror evidence disagrees with activation receipt: " + key)
    require(abs(mirror["post_fix_in_process_step1"] - mirror["fresh_step1_mixed"]) <= 1e-3,
            "post-step in-process evaluation disagrees with fresh weights")
    require(abs(mirror["fresh_step1_mixed"] - mirror["fresh_step1_fp32arm"]) <= 1e-3,
            "fresh step-one mixed and FP32 arms disagree")
    require(mirror["step100_fresh_mixed"] - mirror["step100_fresh_fp32arm"] > 0.05,
            "step-100 mixed/FP32 drift is not reproduced")
    require(abs(mirror["step100_in_process_mixed"] - mirror["step100_fresh_mixed"]) <= 5e-3,
            "step-100 in-process evaluation differs from fresh weights")

    evidence = document.get("evidence") or {}
    require(len(evidence) > 100, "too few Phase XIX-S evidence files")
    receipt_dir = path.parent
    for name, expected in evidence.items():
        candidate = receipt_dir / name
        require(candidate.is_file(), "missing evidence " + name)
        require(digest(candidate) == expected, "evidence changed: " + name)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--binary-root", type=Path)
    args = parser.parse_args()
    try:
        document = verify(args.receipt, binary_root=args.binary_root)
    except KeyError as error:
        print(f"PHASE XIX-X CURRENT SOURCE: FAIL (missing training arm or required sub-receipt evidence: {error})")
        return 1
    except (ValueError, FileNotFoundError) as error:
        print(f"PHASE XIX-X CURRENT SOURCE: FAIL ({error})")
        return 1
    print("PHASE XIX-X CURRENT SOURCE: PASS "
          f"({document['source_revision'][:12]}, "
          f"{len(document['source_files'])} sources, "
          f"{document['counts']['attention_backward']['comparisons']} gradient tensors, "
          f"{document['shaders']['inventory']} shader artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
