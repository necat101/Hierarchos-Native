#!/usr/bin/env python3
"""Freeze and verify the Phase XIX-T policy overlay.

This producer deliberately does not regenerate any Phase XIX-S receipt or
evidence. It first passes the frozen Phase XIX-S checker, then snapshots the
exact source and evidence inventories into a separately versioned policy
overlay. A changed numerical receipt or evidence byte invalidates the overlay.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from phase19t_policy import (
    CRATE,
    DEFAULT_OVERLAY,
    DEFAULT_RECEIPT,
    EXPECTED_EVIDENCE_COUNT,
    EXPECTED_EVIDENCE_MANIFEST_SHA256,
    EXPECTED_SAFETENSOR_COUNT,
    EXPECTED_SOURCE_COUNT,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_NUMERICAL_SOURCE_COUNT,
    EXPECTED_NUMERICAL_SOURCE_MANIFEST_SHA256,
    POLICY_FILES,
    PolicyError,
    digest,
    git,
    load,
    manifest_sha256,
    require,
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(CRATE.resolve()).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    revision = git("rev-parse", "HEAD")
    require(len(revision) == 40, "cannot resolve policy revision")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    require(not dirty or args.allow_dirty,
            "tracked worktree must be clean before freezing the policy:\n" + dirty)

    # The frozen checker must approve the frozen numerical receipt before the
    # stronger policy can snapshot it.
    from check_phase19s_current import verify as verify_phase19s

    verify_phase19s(args.receipt, CRATE)
    document = load(args.receipt)
    require(document.get("kind") == "phase19s-instrumented-current-source",
            "unexpected numerical receipt kind")
    sources = document.get("source_files")
    evidence = document.get("evidence")
    require(isinstance(sources, dict) and isinstance(evidence, dict),
            "numerical receipt manifests are missing")
    require(len(sources) == EXPECTED_NUMERICAL_SOURCE_COUNT,
            f"unexpected numerical source count {len(sources)}")
    require(len(evidence) == EXPECTED_EVIDENCE_COUNT,
            f"unexpected evidence count {len(evidence)}")
    require(manifest_sha256(sources) == EXPECTED_NUMERICAL_SOURCE_MANIFEST_SHA256,
            "numerical source manifest does not match the frozen policy")
    complete_sources = dict(sources)
    additional_sources = {}
    for name in document["subreceipts"]:
        sub = load(CRATE / document["subreceipts"][name]["path"])
        sub_sources = sub.get("sources") if name == "attention" else sub.get("source_files")
        for relative_name, expected in (sub_sources or {}).items():
            repository_name = (relative_name if (CRATE.parent / relative_name).is_file()
                               else "hierarchos-vulkan/" + relative_name)
            path = CRATE.parent / repository_name
            require(path.is_file(), "missing sub-receipt source " + relative_name)
            require(digest(path) == expected,
                    "sub-receipt source changed: " + relative_name)
            if repository_name not in complete_sources:
                complete_sources[repository_name] = expected
                additional_sources[repository_name] = expected
    require(len(complete_sources) == EXPECTED_SOURCE_COUNT,
            f"unexpected complete source count {len(complete_sources)}")
    require(manifest_sha256(complete_sources) == EXPECTED_SOURCE_MANIFEST_SHA256,
            "complete source manifest does not match the frozen policy")
    require(manifest_sha256(evidence) == EXPECTED_EVIDENCE_MANIFEST_SHA256,
            "evidence manifest does not match the frozen policy")

    safetensors = sorted(name for name in evidence if name.endswith(".safetensors"))
    require(len(safetensors) == EXPECTED_SAFETENSOR_COUNT,
            f"unexpected safetensors count {len(safetensors)}")

    policy_files = {}
    for name in POLICY_FILES:
        path = CRATE.parent / name
        require(path.is_file(), "missing policy file: " + name)
        policy_files[name] = digest(path)

    overlay = {
        "schema": 1,
        "kind": "phase19t-policy-overlay",
        "status": "PASS",
        "pass": True,
        "target": "rdna3",
        "policy_revision": revision,
        "numerical_source_revision": document["source_revision"],
        "numerical_receipt": {
            "path": relative(args.receipt),
            "sha256": digest(args.receipt),
            "source_revision": document["source_revision"],
            "kind": document["kind"],
        },
        "numerical_source_manifest_sha256": manifest_sha256(sources),
        "source_manifest_sha256": manifest_sha256(complete_sources),
        "numerical_sources": dict(sorted(sources.items())),
        "additional_sources": dict(sorted(additional_sources.items())),
        "required_sources": dict(sorted(complete_sources.items())),
        "evidence_manifest_sha256": manifest_sha256(evidence),
        "required_evidence": dict(sorted(evidence.items())),
        "required_safetensors": safetensors,
        "policy_files": policy_files,
        "limitations": [
            "The numerical evidence was produced on RDNA3; CPU CI re-verifies committed bytes and derivations, not a fresh GPU execution.",
            "This overlay hardens verification policy without changing the frozen Phase XIX-S receipt or its numerical source revision.",
            "The activation receipt binds five external weight dumps by SHA-256; those raw dumps are not committed.",
            "Adjacent historical receipts remain historical evidence and are not made current by this overlay.",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")

    from check_phase19t_current import verify as verify_phase19t

    summary = verify_phase19t(args.overlay, args.receipt)
    print("PHASE XIX-T POLICY PRODUCER: PASS "
          f"(policy {summary['policy_revision'][:12]}, "
          f"numerical {summary['numerical_source_revision'][:12]}, "
          f"{summary['sources']} sources, {summary['evidence']} evidence files)")
    print(f"overlay={args.overlay}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PolicyError as error:
        print(f"PHASE XIX-T POLICY PRODUCER: FAIL: {error}")
        sys.exit(1)
