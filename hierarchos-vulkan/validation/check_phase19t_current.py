#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XIX-T policy overlay.

The numerical Phase XIX-S receipt is frozen. This checker verifies that receipt
through the six stage checkers, then applies the stronger Phase XIX-T policy:
exact source/evidence inventories, tracked-file presence, path confinement,
sub-receipt lineage, binary identities and cross-stage fixture/config binds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from phase19t_policy import (
    CRATE,
    DEFAULT_OVERLAY,
    DEFAULT_RECEIPT,
    EVIDENCE_KIND,
    EVIDENCE_SCHEMA,
    EXPECTED_NUMERICAL_SOURCE_MANIFEST_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_SUBREVISIONS,
    HEX40,
    NUMERICAL_KIND,
    NUMERICAL_SOURCE_REVISION,
    POLICY_FILES,
    PolicyError,
    check_binaries,
    check_evidence,
    check_manifest_constants,
    check_policy_files,
    check_receipt_manifests,
    check_sources,
    cross_bind,
    digest,
    enabled_tracked_files,
    git,
    load,
    manifest_sha256,
    require,
    safe_relative,
)


def verify(
    overlay_path: Path = DEFAULT_OVERLAY,
    receipt_path: Path | None = None,
    crate: Path = CRATE,
    binary_root: Path | None = None,
    strict_policy: bool = True,
) -> dict:
    require(overlay_path.is_file(), f"missing policy overlay {overlay_path}")
    overlay = load(overlay_path)
    require(overlay.get("schema") == EVIDENCE_SCHEMA, "unexpected Phase XIX-T schema")
    require(overlay.get("kind") == EVIDENCE_KIND, "unexpected Phase XIX-T kind")
    require(overlay.get("status") == "PASS" and overlay.get("pass") is True,
            "Phase XIX-T policy overlay is not a pass")
    require(overlay.get("target") == "rdna3", "policy target changed")

    head = git("rev-parse", "HEAD")
    policy_revision = str(overlay.get("policy_revision", ""))
    require(HEX40.fullmatch(policy_revision) is not None, "invalid policy revision")
    require(policy_revision == head or _ancestor(policy_revision, head),
            "policy revision is not an ancestor of HEAD")
    require(overlay.get("numerical_source_revision") == NUMERICAL_SOURCE_REVISION,
            "numerical source revision changed")
    check_policy_files(overlay)

    numerical_sources, sources, evidence = check_manifest_constants(
        overlay, strict=strict_policy)
    if strict_policy:
        require(overlay.get("numerical_source_manifest_sha256")
                == EXPECTED_NUMERICAL_SOURCE_MANIFEST_SHA256,
                "overlay numerical source manifest digest changed")
        require(overlay.get("source_manifest_sha256") == EXPECTED_SOURCE_MANIFEST_SHA256,
                "overlay source manifest digest changed")
        require(overlay.get("evidence_manifest_sha256") == manifest_sha256(evidence),
                "overlay evidence manifest digest changed")

    receipt_entry = overlay.get("numerical_receipt")
    require(isinstance(receipt_entry, dict), "missing numerical receipt binding")
    production_receipt = safe_relative(receipt_entry.get("path"), crate,
                                       "numerical receipt")
    actual_receipt = Path(receipt_path) if receipt_path is not None else production_receipt
    require(actual_receipt.is_file(), f"missing Phase XIX-S receipt {actual_receipt}")
    require(receipt_entry.get("sha256") == digest(actual_receipt),
            "numerical receipt changed")
    require(receipt_entry.get("source_revision") == NUMERICAL_SOURCE_REVISION,
            "numerical receipt source revision changed")
    document = load(actual_receipt)
    require(document.get("kind") == NUMERICAL_KIND, "unexpected numerical receipt kind")
    require(document.get("status") == "PASS" and document.get("pass") is True,
            "numerical receipt is not a pass")
    require(document.get("source_revision") == NUMERICAL_SOURCE_REVISION,
            "numerical receipt source revision changed")
    check_receipt_manifests(document, numerical_sources, evidence)

    tracked = enabled_tracked_files()
    check_sources(sources, tracked)
    receipt_dir = actual_receipt.resolve().parent
    check_evidence(receipt_dir, evidence,
                   "hierarchos-vulkan/" + receipt_entry["path"], tracked)
    check_binaries(document, binary_root)

    subentries = document.get("subreceipts") or {}
    require(isinstance(subentries, dict), "missing sub-receipt manifest")
    require(set(subentries) == set(EXPECTED_SUBREVISIONS), "sub-receipt set changed")
    subdocs: dict[str, dict] = {}
    subdirs: dict[str, Path] = {}
    for name in EXPECTED_SUBREVISIONS:
        entry = subentries[name]
        require(isinstance(entry, dict), f"malformed sub-receipt entry {name}")
        production = safe_relative(entry.get("path"), crate, f"{name} sub-receipt")
        require(production.name == "current-qualification.json",
                f"{name} sub-receipt filename changed")
        local = receipt_dir / name / "current-qualification.json"
        require(local.is_file(), f"missing sub-receipt {name}")
        require(digest(local) == entry.get("sha256"), f"sub-receipt changed: {name}")
        subdocs[name] = load(local)
        subdirs[name] = local.parent

    from check_phase16_receipts import verify_receipt
    from check_training_phase16 import verify as verify_training
    from check_training_stability import verify as verify_stability
    from check_heldout_phase17 import verify as verify_heldout
    from check_wmma_drift import verify as verify_drift

    def stage(label, function):
        try:
            return function()
        except Exception as error:  # stage checker exception types are independent
            raise PolicyError(f"{label} qualification failed: {error}") from error

    stage("attention", lambda: verify_receipt(
        subdocs["attention"], subdirs["attention"], source_head=head))
    stage("training", lambda: verify_training(subdirs["training"], crate))
    stage("stability", lambda: verify_stability(subdirs["stability"], crate))
    stage("held-out", lambda: verify_heldout(subdirs["heldout"], crate))
    stage("drift", lambda: verify_drift(
        subdirs["drift"] / "current-qualification.json", crate, epoch_reuse=True))

    activation = subprocess.run(
        [sys.executable, str(crate / "validation/check_activation_phase19r.py")],
        cwd=crate,
        env={**os.environ, "HIERARCHOS_ACTIVATION_RECEIPT": str(subdirs["activation"])},
        capture_output=True,
        text=True,
    )
    require(activation.returncode == 0, "activation qualification failed: "
            + (activation.stdout + activation.stderr).strip()[-400:])

    cross_bind(document, subdocs, policy_revision, sources)

    counts = document.get("counts") or {}
    require(counts.get("shapes") == 16, "GPT-2 shape coverage changed")
    retained = counts.get("retained") or {}
    for name, expected in (("shared", 74), ("kimi", 3), ("seq2seq", 6),
                           ("gemma3", 4), ("llava", 4)):
        require(retained.get(name, {}).get("cases") == expected,
                f"retained family {name} coverage changed")
    backward = counts.get("attention_backward") or {}
    require(backward.get("cases") == 4 and backward.get("comparisons") == 12,
            "attention-backward coverage changed")
    require(backward.get("elements", 0) > 0
            and backward.get("bitwise_equal") == backward.get("elements"),
            "attention-backward results are not bit-exact")
    require(backward.get("max_abs") == 0.0 and backward.get("max_ulp") == 0,
            "attention-backward reports an error")
    require(counts.get("training_arms") == 8 and counts.get("resume_arms") == 3,
            "training/resume arm coverage changed")
    require(counts.get("stability_arms") == 4, "stability arm coverage changed")
    require(counts.get("heldout_arms") == 4, "held-out arm coverage changed")
    require(counts.get("drift_runs") == 6, "drift run coverage changed")
    require(len(counts.get("activation_tensors") or []) == 27,
            "activation tensor inventory changed")

    shaders = document.get("shaders") or {}
    require(shaders.get("inventory") == 45
            and shaders.get("status_counts") == {"identical": 45},
            "45-artifact shader inventory changed")
    require(shaders.get("historical_artifacts") == 43
            and shaders.get("phase16_artifacts") == 2,
            "shader scope changed")
    require(shaders == {
        "inventory": subdocs["attention"]["shaders"]["inventory"],
        "status_counts": subdocs["attention"]["shaders"]["status_counts"],
        "historical_artifacts": subdocs["attention"]["shaders"]["historical"]["artifacts"],
        "phase16_artifacts": subdocs["attention"]["shaders"]["phase16"]["artifacts"],
        "sealed_gate_sha256": subdocs["attention"]["shaders"]["historical"]["sealed_gate_sha256"],
    }, "top-level shader inventory disagrees with attention receipt")

    mirror = document.get("mirror_epoch") or {}
    anomaly = subdocs["activation"]["anomaly"]
    for key, value in mirror.items():
        require(key in anomaly and anomaly[key] == value,
                "mirror evidence disagrees with activation receipt: " + key)
    require(abs(mirror["post_fix_in_process_step1"] - mirror["fresh_step1_mixed"]) <= 1e-3,
            "post-fix in-process evaluation disagrees with fresh weights")
    require(mirror["pre_fix_in_process_step1"] - mirror["fresh_step1_mixed"] > 0.2,
            "stale-mirror step-one lag is not reproduced")
    require(abs(mirror["fresh_step1_mixed"] - mirror["fresh_step1_fp32arm"]) <= 1e-3,
            "fresh step-one mixed and FP32 arms disagree")
    require(mirror["step100_fresh_mixed"] - mirror["step100_fresh_fp32arm"] > 0.05,
            "step-100 mixed/FP32 drift is not reproduced")
    require(abs(mirror["step100_in_process_mixed"] - mirror["step100_fresh_mixed"]) <= 5e-3,
            "step-100 in-process evaluation differs from fresh weights")

    return {
        "policy_revision": policy_revision,
        "numerical_source_revision": document["source_revision"],
        "sources": len(sources),
        "evidence": len(evidence),
        "safetensors": len(overlay["required_safetensors"]),
        "subreceipts": len(subdocs),
        "binaries": len(document["binaries"]),
    }


def _ancestor(revision: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(CRATE.parent), "merge-base", "--is-ancestor",
         revision, descendant],
        capture_output=True,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--binary-root", type=Path)
    args = parser.parse_args()
    try:
        summary = verify(args.overlay, args.receipt, binary_root=args.binary_root)
    except (PolicyError, ValueError, KeyError, FileNotFoundError) as error:
        print(f"PHASE XIX-T CURRENT SOURCE: FAIL ({error})")
        return 1
    print("PHASE XIX-T CURRENT SOURCE: PASS "
          f"(policy {summary['policy_revision'][:12]}, "
          f"numerical {summary['numerical_source_revision'][:12]}, "
          f"{summary['sources']} sources, {summary['evidence']} evidence files, "
          f"{summary['safetensors']} safetensors, {summary['subreceipts']} subreceipts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
