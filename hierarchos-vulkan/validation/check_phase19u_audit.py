#!/usr/bin/env python3
"""Independently verify the Phase XIX-U historical-test audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "hierarchos-vulkan"
AUDIT_REL = "hierarchos-vulkan/validation/historical_audit/phase19u-historical-audit.json"
OVERLAY_REL = "hierarchos-vulkan/validation/phase19t-policy-overlay.json"
HISTORICAL_REL = "hierarchos-vulkan/validation/historical_audit"
CURRENT_SOURCE = "60063fd3b0a3a963b05e7d5c472cf9b27a4a1cab"
POLICY_REVISION = "dac3534c47f6330c65274f3a9495ab2aab303e1f"
EXPECTED_IDS = (
    "phase16-training",
    "phase16-stability",
    "phase17-heldout",
    "phase18-wmma-drift",
    "phase19r-activation",
)
EXPECTED_RUNS = {
    "phase16-training": {
        "receipt": "b34e51d062f5b9667770af57fdca408eab560b47",
        "source": "8a12af2d907cec715675c70399d8e3afe70b404c",
        "test": "hierarchos-vulkan/validation/test_training_phase16.py",
        "counts": {"total": 25, "passed": 24, "failed": 0, "errors": 1,
                   "skipped": 0, "blocked": 0},
        "exit": 1,
        "missing": ["states/F/u/hierarchos_adapter_optimizer.safetensors"],
    },
    "phase16-stability": {
        "receipt": "3fb0639ff47117a27949616f0dbfeee2f29080e3",
        "source": "21027bf9699757859e7f2a9c07c437402d5aa2ae",
        "test": "hierarchos-vulkan/validation/test_training_stability.py",
        "counts": {"total": 31, "passed": 29, "failed": 0, "errors": 2,
                   "skipped": 0, "blocked": 0},
        "exit": 1,
        "missing": ["adapter-E-u.safetensors",
                    "states/F/u/hierarchos_adapter_optimizer.safetensors"],
    },
    "phase17-heldout": {
        "receipt": "27f853fb477d6904183a91f77f5dabc7e7d4d3ed",
        "source": "5cd0c0c6eb7f9a8d0c9ef0bdc214186a8df76fb6",
        "test": "hierarchos-vulkan/validation/test_heldout_phase17.py",
        "counts": {"total": 22, "passed": 22, "failed": 0, "errors": 0,
                   "skipped": 0, "blocked": 0},
        "exit": 0,
        "missing": [],
    },
    "phase18-wmma-drift": {
        "receipt": "1e77b401489c7c59a95661461fbaaf5a91f3b77c",
        "source": "87a0095b83012f3b5fc3efa1085447516c45b4a7",
        "test": "hierarchos-vulkan/validation/test_wmma_drift.py",
        "counts": {"total": 26, "passed": 26, "failed": 0, "errors": 0,
                   "skipped": 0, "blocked": 0},
        "exit": 0,
        "missing": [],
    },
    "phase19r-activation": {
        "receipt": "e69f7f978101f29e8ad6e09b0a88ce5fb1b767ad",
        "source": "3ac63b4d6315f1940688765c17c3e0751567c0ce",
        "test": "hierarchos-vulkan/validation/test_activation_phase19r.py",
        "counts": {"total": 22, "passed": 21, "failed": 0, "errors": 0,
                   "skipped": 1, "blocked": 0},
        "exit": 0,
        "missing": [],
    },
}
EXPECTED_RECOVERY = {
    "phase16-training": {
        "counts": {"total": 25, "passed": 25, "failed": 0, "errors": 0,
                   "skipped": 0, "blocked": 0},
        "artifacts": {
            "states/F/u/hierarchos_adapter_optimizer.safetensors": (
                "075e6e1f0a3c2f35a09352e32633d0ef3570041574e0c9cd2109db8f1f91188e",
                "2fe0bd4cc7bc14ed87b10e42f94ab3d8cb6385f3",
            ),
        },
    },
    "phase16-stability": {
        "counts": {"total": 31, "passed": 31, "failed": 0, "errors": 0,
                   "skipped": 0, "blocked": 0},
        "artifacts": {
            "adapter-E-u.safetensors": (
                "cff27588f3e15e32435992e36e649787ee0bdaefe135161c2182928880ebd2ce",
                "0d770f4896a956ee0e4309ed56d9840d18e6ba86",
            ),
            "states/F/u/hierarchos_adapter_optimizer.safetensors": (
                "575810a7aa43b3165451867671786896fbda45840fb650846e3850fbc4abf45d",
                "02105125bda6f91ec42255152a40208f2120d1f7",
            ),
        },
    },
}


class AuditError(Exception):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def git_blob(root: Path, rev: str, path: str) -> bytes:
    try:
        out = subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "ls-tree", "-r", rev, "--", path],
            cwd=root,
            stderr=subprocess.PIPE,
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError as exc:
        raise AuditError(f"Git lookup failed: {rev}:{path}") from exc
    require(out, f"historical path missing: {rev}:{path}")
    fields = out.split()
    require(len(fields) >= 3, f"malformed ls-tree result for {rev}:{path}")
    oid = fields[2]
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "cat-file", "blob", oid],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise AuditError(f"Git blob lookup failed: {rev}:{path}") from exc


def git_commit_exists(root: Path, revision: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", revision + "^{commit}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def safe_relative(name: str) -> str:
    path = Path(name)
    require(name and not path.is_absolute(), f"unsafe absolute path: {name}")
    require("\\" not in name, f"unsafe backslash path: {name}")
    require(path.as_posix() == name, f"non-canonical path: {name}")
    require(all(part not in ("", ".", "..") for part in path.parts),
            f"unsafe relative path: {name}")
    return name


def resolve_under(root: Path, name: str, base: Path | None = None) -> Path:
    safe_relative(name)
    candidate = (root / name).resolve()
    allowed = (base or root).resolve()
    require(candidate == allowed or allowed in candidate.parents,
            f"path escapes its allowed directory: {name}")
    return candidate


def parse_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    total_match = re.search(r"^Ran (\d+) tests? in ", text, re.M)
    status_match = re.search(r"^(OK|FAILED)(?: \(([^)]*)\))?$", text, re.M)
    require(total_match is not None, f"missing test count in {path}")
    require(status_match is not None, f"missing unittest status in {path}")
    total = int(total_match.group(1))
    skipped = failed = errors = 0
    if status_match.group(2):
        for part in status_match.group(2).split(", "):
            require("=" in part, f"malformed unittest status in {path}")
            key, value = part.split("=", 1)
            if key == "skipped":
                skipped = int(value)
            elif key == "failures":
                failed = int(value)
            elif key == "errors":
                errors = int(value)
    return {
        "total": total,
        "passed": total - skipped - failed - errors,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "blocked": 0,
    }


def check_counts(actual: dict, expected: dict[str, int]) -> None:
    require(actual == expected, f"count mismatch: {actual} != {expected}")


def check_log(root: Path, entry: dict, expected: dict[str, int],
              exit_code: int) -> None:
    rel = safe_relative(entry["path"])
    require(rel.startswith(HISTORICAL_REL + "/logs/"), f"log outside audit logs: {rel}")
    path = resolve_under(root, rel)
    require(path.is_file(), f"log missing: {rel}")
    require(entry["sha256"] == sha256_path(path), f"log digest changed: {rel}")
    check_counts(parse_counts(path), expected)
    require(exit_code == (0 if expected["failed"] == 0 and expected["errors"] == 0 else 1),
            f"exit code does not match unittest result: {rel}")


def load_document(root: Path) -> dict:
    path = root / AUDIT_REL
    require(path.is_file(), "historical audit JSON is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_document(document: dict, root: Path) -> dict:
    require(document.get("schema") == "hierarchos.phase19u.historical-audit.v1",
            "unexpected audit schema")
    require(document.get("qualification_scope") == "historical_only",
            "audit scope must be historical_only")
    require(document.get("not_current_qualification") is True,
            "audit must explicitly disclaim current qualification")
    env = document.get("environment", {})
    for key in ("platform", "machine", "python"):
        require(isinstance(env.get(key), str) and env[key], f"missing environment {key}")

    current = document.get("current_qualification", {})
    require(current.get("numerical_source_revision") == CURRENT_SOURCE,
            "current numerical source identity changed")
    require(current.get("phase19t_policy_revision") == POLICY_REVISION,
            "Phase XIX-T policy revision changed")
    overlay_path = resolve_under(root, OVERLAY_REL)
    require(overlay_path.is_file(), "Phase XIX-T overlay is missing")
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    require(overlay.get("numerical_source_revision") == CURRENT_SOURCE,
            "overlay numerical source disagrees with the audit")
    require(overlay.get("policy_revision") == POLICY_REVISION,
            "overlay policy revision disagrees with the audit")
    require(current.get("phase19t_overlay_path") == OVERLAY_REL,
            "unexpected overlay path")
    require(current.get("phase19t_overlay_sha256") == sha256_path(overlay_path),
            "Phase XIX-T overlay digest changed")

    runs = document.get("runs")
    require(isinstance(runs, list), "runs must be a list")
    require([r.get("id") for r in runs] == list(EXPECTED_IDS),
            "historical run set or order changed")
    recoveries = 0
    for record in runs:
        run = EXPECTED_RUNS[record["id"]]
        require(record.get("scope") == "historical_only",
                f"{record['id']} must be historical_only")
        require(record.get("receipt_revision") == run["receipt"],
                f"{record['id']} receipt revision changed")
        require(record.get("source_revision") == run["source"],
                f"{record['id']} source revision changed")
        require(record.get("test_source") == run["test"],
                f"{record['id']} test source changed")
        require(git_commit_exists(root, run["receipt"]),
                f"{record['id']} receipt commit unavailable")
        require(git_commit_exists(root, run["source"]),
                f"{record['id']} source commit unavailable")
        require(git_is_ancestor(root, run["source"], run["receipt"]),
                f"{record['id']} source is not an ancestor of its receipt")
        test_bytes = git_blob(root, run["receipt"], run["test"])
        require(record.get("test_sha256") == sha256_bytes(test_bytes),
                f"{record['id']} test digest changed")
        receipt_bytes = git_blob(root, run["receipt"], record["receipt_path"])
        require(record.get("receipt_sha256") == sha256_bytes(receipt_bytes),
                f"{record['id']} receipt digest changed")
        receipt = json.loads(receipt_bytes)
        evidence_map = receipt.get("evidence", {})
        if run["missing"] or record["id"] in EXPECTED_RECOVERY:
            require(isinstance(evidence_map, dict),
                    f"{record['id']} receipt has no evidence map")
        require(record.get("missing_artifacts") == run["missing"],
                f"{record['id']} missing-artifact list changed")
        for artifact in run["missing"]:
            safe_relative(artifact)
            require(artifact in evidence_map,
                    f"{record['id']} missing artifact is not receipt-bound: {artifact}")
        require(record.get("exit_code") == run["exit"],
                f"{record['id']} exit code changed")
        check_counts(record.get("counts"), run["counts"])
        check_log(root, record["log"], run["counts"], run["exit"])
        controls = record.get("current_source_controls")
        require(isinstance(controls, list) and controls,
                f"{record['id']} has no current-source controls")
        for control in controls:
            safe_relative(control)
            require((root / control).is_file(),
                    f"{record['id']} current control is missing: {control}")

        expected_recovery = EXPECTED_RECOVERY.get(record["id"])
        if expected_recovery is None:
            require("recovery" not in record or record["recovery"] is None,
                    f"{record['id']} has an unexpected recovery record")
            continue
        recoveries += 1
        recovery = record.get("recovery")
        require(isinstance(recovery, dict), f"{record['id']} recovery is missing")
        require(recovery.get("status") == "passed",
                f"{record['id']} recovery did not pass")
        require(recovery.get("exit_code") == 0,
                f"{record['id']} recovery exit code changed")
        check_counts(recovery.get("counts"), expected_recovery["counts"])
        check_log(root, recovery["log"], expected_recovery["counts"], 0)
        artifacts = recovery.get("recovered_artifacts")
        require(isinstance(artifacts, list), f"{record['id']} recovery artifacts missing")
        actual_targets = [a.get("target") for a in artifacts]
        require(actual_targets == list(expected_recovery["artifacts"]),
                f"{record['id']} recovered artifact set changed")
        for artifact in artifacts:
            target = safe_relative(artifact["target"])
            expected_sha, expected_blob = expected_recovery["artifacts"][target]
            stored = resolve_under(root, artifact["stored_path"],
                                   root / HISTORICAL_REL / "recovered")
            require(stored.is_file(), f"recovered artifact missing: {stored}")
            data = stored.read_bytes()
            require(artifact.get("sha256") == sha256_bytes(data),
                    f"{record['id']} recovered artifact digest changed")
            require(artifact.get("receipt_expected_sha256") == expected_sha,
                    f"{record['id']} receipt expectation changed")
            require(expected_sha == sha256_bytes(data),
                    f"{record['id']} recovered bytes do not match receipt")
            require(evidence_map.get(target) == expected_sha,
                    f"{record['id']} receipt does not bind recovered artifact")
            require(artifact.get("git_blob") == expected_blob,
                    f"{record['id']} recovered Git blob identity changed")
            require(git_blob_sha1(data) == expected_blob,
                    f"{record['id']} recovered bytes do not match Git blob")
            require(artifact.get("size") == len(data),
                    f"{record['id']} recovered artifact size changed")
            require(re.fullmatch(r"[0-9a-f]{40}", artifact.get("source_commit", "")),
                    f"{record['id']} recovered source commit is malformed")
            safe_relative(artifact["source_path"])
            # The orphaned provenance commits are not required to be present in CI.
            if git_commit_exists(root, artifact["source_commit"]):
                require(git_blob(root, artifact["source_commit"], artifact["source_path"]) == data,
                        f"{record['id']} recovered provenance blob changed")
    require(recoveries == len(EXPECTED_RECOVERY),
            "recovery record count changed")
    return {
        "runs": len(runs),
        "recoveries": recoveries,
        "audit_sha256": sha256_bytes((root / AUDIT_REL).read_bytes()),
    }


def verify(root: Path = ROOT) -> dict:
    return verify_document(load_document(root), root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        summary = verify(args.root)
    except (AuditError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PHASE XIX-U HISTORICAL AUDIT: FAIL: {exc}")
        return 1
    print(
        "PHASE XIX-U HISTORICAL AUDIT: PASS "
        f"({summary['runs']} runs, {summary['recoveries']} recoveries, "
        f"audit {summary['audit_sha256'][:12]})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
