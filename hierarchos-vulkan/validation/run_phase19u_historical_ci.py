#!/usr/bin/env python3
"""Reproduce the five Phase XIX-U historical suites in CI.

This runner intentionally writes a new report and new logs. It never edits the
retained Phase XIX-U audit or Windows execution logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import check_phase19u_audit as audit

ROOT = Path(__file__).resolve().parents[2]
CI_SCHEMA = "hierarchos.phase19u.historical-ci.v1"
ERROR_CONTRACT = {
    "phase16-training": {
        "test_training_phase16.TrainingQualificationControls.test_missing_checkpoint_state":
            "states/F/u/hierarchos_adapter_optimizer.safetensors",
    },
    "phase16-stability": {
        "test_training_stability.StabilityControls.test_changed_adapter_bytes":
            "adapter-E-u.safetensors",
        "test_training_stability.StabilityControls.test_missing_checkpoint_state":
            "states/F/u/hierarchos_adapter_optimizer.safetensors",
    },
}
SKIP_CONTRACT = {
    "phase19r-activation": (
        "test_changed_binary",
        "probe binary is not checked out",
    ),
}


class ContractError(Exception):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parse_output(text: str) -> tuple[dict[str, int], dict[str, str], list[str]]:
    """Parse unittest verbose output and return counts, error blocks and IDs."""
    total_match = re.search(r"^Ran (\d+) tests? in ", text, re.M)
    status_match = re.search(r"^(OK|FAILED)(?: \(([^)]*)\))?$", text, re.M)
    require(total_match is not None, "missing unittest test count")
    require(status_match is not None, "missing unittest status")
    total = int(total_match.group(1))
    skipped = failed = errors = 0
    if status_match.group(2):
        for part in status_match.group(2).split(", "):
            require("=" in part, "malformed unittest status")
            key, value = part.split("=", 1)
            if key == "skipped":
                skipped = int(value)
            elif key == "failures":
                failed = int(value)
            elif key == "errors":
                errors = int(value)
        require(status_match.group(1) == "FAILED" or failed == errors == 0,
                "unittest status and error totals disagree")
    counts = {
        "total": total,
        "passed": total - skipped - failed - errors,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "blocked": 0,
    }
    matches = list(re.finditer(r"^ERROR: (\S+) \(([^)]+)\)\s*$", text, re.M))
    blocks = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(2)] = text[match.start():end]
    failure_ids = re.findall(r"^FAIL: (\S+) \(([^)]+)\)\s*$", text, re.M)
    failures = [item[1] for item in failure_ids]
    return counts, blocks, failures


def classify_original(run_id: str, text: str, exit_code: int) -> dict:
    expected = audit.EXPECTED_RUNS[run_id]
    counts, error_blocks, failures = parse_output(text)
    require(counts == expected["counts"],
            f"{run_id} original counts changed: {counts}")
    require(exit_code == expected["exit"], f"{run_id} original exit changed")
    require(not failures, f"{run_id} original has assertion failures")
    expected_errors = ERROR_CONTRACT.get(run_id, {})
    require(set(error_blocks) == set(expected_errors),
            f"{run_id} original error identities changed: {sorted(error_blocks)}")
    for test_id, artifact in expected_errors.items():
        require(artifact in error_blocks[test_id],
                f"{run_id} error {test_id} does not identify {artifact}")
    if run_id in SKIP_CONTRACT:
        test_name, message = SKIP_CONTRACT[run_id]
        require(counts["skipped"] == 1, f"{run_id} expected one documented skip")
        require(test_name in text and message in text,
                f"{run_id} skip is not the documented executable-dependent test")
    return {
        "counts": counts,
        "exit_code": exit_code,
        "verified_error_ids": sorted(error_blocks),
        "verified_missing_artifacts": sorted(expected_errors.values()),
        "verified_skip": run_id in SKIP_CONTRACT,
    }


def classify_recovery(run_id: str, text: str, exit_code: int) -> dict:
    expected = audit.EXPECTED_RECOVERY[run_id]
    counts, error_blocks, failures = parse_output(text)
    require(counts == expected["counts"],
            f"{run_id} recovery counts changed: {counts}")
    require(exit_code == 0, f"{run_id} recovery exited nonzero")
    require(not error_blocks, f"{run_id} recovery still has errors")
    require(not failures, f"{run_id} recovery still has failures")
    require(counts["skipped"] == 0, f"{run_id} recovery unexpectedly skipped")
    return {
        "counts": counts,
        "exit_code": exit_code,
        "verified_error_ids": [],
        "verified_missing_artifacts": [],
        "verified_skip": False,
    }


def validate_recovered_bytes(expected_sha: str, expected_blob: str, data: bytes) -> None:
    require(sha256_bytes(data) == expected_sha, "recovered artifact SHA-256 changed")
    require(audit.git_blob_sha1(data) == expected_blob,
            "recovered artifact Git blob changed")


def validate_identity(record: dict, root: Path) -> None:
    expected = audit.EXPECTED_RUNS.get(record.get("id"))
    require(expected is not None, f"unknown historical suite: {record.get('id')}")
    require(record.get("receipt_revision") == expected["receipt"],
            f"{record['id']} receipt revision changed")
    require(record.get("source_revision") == expected["source"],
            f"{record['id']} source revision changed")
    require(record.get("test_source") == expected["test"],
            f"{record['id']} test source changed")
    require(audit.git_commit_exists(root, record["receipt_revision"]),
            f"{record['id']} receipt commit is missing")
    require(audit.git_commit_exists(root, record["source_revision"]),
            f"{record['id']} source commit is missing")
    test_bytes = audit.git_blob(root, record["receipt_revision"], expected["test"])
    require(record.get("test_sha256") == audit.sha256_bytes(test_bytes),
            f"{record['id']} test source digest changed")
    receipt_bytes = audit.git_blob(root, record["receipt_revision"],
                                   record["receipt_path"])
    require(record.get("receipt_sha256") == audit.sha256_bytes(receipt_bytes),
            f"{record['id']} receipt digest changed")


def git_checkout(root: Path, revision: str) -> None:
    if audit.git_commit_exists(root, revision):
        return
    try:
        subprocess.run(
            ["git", "fetch", "--no-tags", "origin", revision],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"required Git revision could not be fetched: {revision}") from exc
    require(audit.git_commit_exists(root, revision),
            f"required Git revision is absent after fetch: {revision}")


def run_command(root: Path, crate: Path, test_source: str, timeout: int = 300) -> tuple[int, str]:
    command = [sys.executable, "-m", "unittest", "discover",
               "-s", "validation", "-p", Path(test_source).name, "-v"]
    try:
        completed = subprocess.run(
            command,
            cwd=crate,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "") + f"\nTIMEOUT after {timeout}s\n"
    return completed.returncode, completed.stdout


def worktree_remove(root: Path, path: Path) -> None:
    try:
        subprocess.run(
            ["git", "-c", "core.longpaths=true", "worktree", "remove", "--force", str(path)],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        subprocess.run(["git", "worktree", "prune"], cwd=root,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_one(root: Path, record: dict, mode: str, output_dir: Path) -> dict:
    run_id = record["id"]
    validate_identity(record, root)
    if mode == "original":
        expected = audit.EXPECTED_RUNS[run_id]
    else:
        require(run_id in audit.EXPECTED_RECOVERY, f"no recovery contract for {run_id}")
        expected = audit.EXPECTED_RECOVERY[run_id]
    git_checkout(root, record["receipt_revision"])
    require(audit.git_is_ancestor(root, record["source_revision"],
                                  record["receipt_revision"]),
            f"{run_id} source is not an ancestor of its receipt")
    parent = Path(tempfile.mkdtemp(prefix=f"phase19u-{run_id}-{mode}-"))
    worktree = parent / "worktree"
    require(not worktree.exists(), "temporary worktree path unexpectedly exists")
    try:
        subprocess.run(
            ["git", "-c", "core.longpaths=true", "worktree", "add", "--detach",
             str(worktree), record["receipt_revision"]],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        test_path = worktree / record["test_source"]
        receipt_path = worktree / record["receipt_path"]
        require(test_path.is_file(), f"{run_id} test source missing in worktree")
        require(receipt_path.is_file(), f"{run_id} receipt missing in worktree")
        require(sha256_path(test_path) == record["test_sha256"],
                f"{run_id} historical test source hash changed")
        require(sha256_path(receipt_path) == record["receipt_sha256"],
                f"{run_id} historical receipt hash changed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verified_artifacts = []
        if mode == "exact_byte_recovery":
            recovery_entries = {
                entry["target"]: entry
                for entry in record.get("recovery", {}).get("recovered_artifacts", [])
            }
            require(set(recovery_entries) == set(expected["artifacts"]),
                    f"{run_id} recovery artifact set changed")
            for target, expected_pair in expected["artifacts"].items():
                target = audit.safe_relative(target)
                expected_sha, expected_blob = expected_pair
                entry = recovery_entries[target]
                stored = audit.resolve_under(
                    root, entry["stored_path"],
                    root / audit.HISTORICAL_REL / "recovered")
                require(stored.is_file(), f"{run_id} retained recovery artifact missing")
                require(sha256_path(stored) == expected_sha,
                        f"{run_id} retained recovery artifact hash changed")
                destination = receipt_path.parent / "evidence" / target
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stored, destination)
                validate_recovered_bytes(expected_sha, expected_blob,
                                         destination.read_bytes())
                require(receipt.get("evidence", {}).get(target) == expected_sha,
                        f"{run_id} recovered artifact is not receipt-bound")
                verified_artifacts.append({
                    "target": target,
                    "sha256": expected_sha,
                    "git_blob": expected_blob,
                    "size": destination.stat().st_size,
                })
        exit_code, output = run_command(root, worktree / "hierarchos-vulkan",
                                         record["test_source"])
        log_name = f"{run_id}-{mode}.log"
        log_path = output_dir / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        try:
            if mode == "original":
                assessment = classify_original(run_id, output, exit_code)
            else:
                assessment = classify_recovery(run_id, output, exit_code)
            passed = True
        except ContractError:
            passed = False
            raise
        result = {
            "id": run_id,
            "mode": mode,
            "scope": "historical_only",
            "receipt_revision": record["receipt_revision"],
            "source_revision": record["source_revision"],
            "test_source": record["test_source"],
            "test_sha256": record["test_sha256"],
            "receipt_path": record["receipt_path"],
            "receipt_sha256": record["receipt_sha256"],
            "expected_counts": expected["counts"],
            "counts": assessment["counts"],
            "expected_exit_code": expected.get("exit", 0),
            "exit_code": exit_code,
            "expected_error_ids": sorted(ERROR_CONTRACT.get(run_id, {})),
            "expected_missing_artifacts": sorted(ERROR_CONTRACT.get(run_id, {}).values()),
            "verified_error_ids": assessment["verified_error_ids"],
            "verified_missing_artifacts": assessment["verified_missing_artifacts"],
            "verified_skip": assessment["verified_skip"],
            "recovered_artifacts": verified_artifacts,
            "contract_passed": passed,
            "log": {
                "path": str(log_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": sha256_path(log_path),
            },
        }
        return result
    finally:
        removed = not worktree.exists()
        if worktree.exists():
            try:
                worktree_remove(root, worktree)
                removed = True
            except subprocess.CalledProcessError:
                removed = False
        if removed:
            shutil.rmtree(parent, ignore_errors=True)


def build_report(root: Path, output_dir: Path, only: str | None = None) -> dict:
    audit_summary = audit.verify(root)
    document = audit.load_document(root)
    selected = [r for r in document["runs"] if only is None or r["id"] == only]
    require(selected, f"unknown suite: {only}")
    runs = []
    for record in selected:
        runs.append(run_one(root, record, "original", output_dir))
        if record["id"] in audit.EXPECTED_RECOVERY:
            runs.append(run_one(root, record, "exact_byte_recovery", output_dir))
    require(len(runs) == (7 if only is None else (1 + (only in audit.EXPECTED_RECOVERY))),
            "unexpected historical run count")
    report = {
        "schema": CI_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "historical_only",
        "not_current_qualification": True,
        "audit": {
            "path": audit.AUDIT_REL,
            "sha256": audit_summary["audit_sha256"],
        },
        "workflow_sha": os.environ.get("GITHUB_SHA", ""),
        "runner": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "runs": runs,
    }
    return report


def verify_report(report: dict, report_path: Path) -> None:
    require(report.get("schema") == CI_SCHEMA, "unexpected CI report schema")
    require(report.get("scope") == "historical_only", "CI report scope must be historical_only")
    require(report.get("not_current_qualification") is True,
            "CI report must disclaim current qualification")
    audit_info = report.get("audit", {})
    require(audit_info.get("path") == audit.AUDIT_REL, "CI report audit path changed")
    require(audit_info.get("sha256") == audit.sha256_path(
        audit.ROOT / audit.AUDIT_REL),
            "CI report audit digest changed")
    runs = report.get("runs")
    require(isinstance(runs, list) and len(runs) == 7, "CI report must contain seven runs")
    original = [r for r in runs if r.get("mode") == "original"]
    recovery = [r for r in runs if r.get("mode") == "exact_byte_recovery"]
    require([r["id"] for r in original] == list(audit.EXPECTED_IDS),
            "original CI run set changed")
    require([r["id"] for r in recovery] == list(audit.EXPECTED_RECOVERY),
            "recovery CI run set changed")
    for record in runs:
        require(record.get("contract_passed") is True, f"{record.get('id')} contract failed")
        counts, errors, failures = parse_output(
            (report_path.parent / record["log"]["path"]).read_text(encoding="utf-8"))
        require(counts == record["counts"], f"{record['id']} report counts disagree with log")
        require(not failures, f"{record['id']} report log has failures")
        if record["mode"] == "original":
            require(record["counts"] == audit.EXPECTED_RUNS[record["id"]]["counts"],
                    f"{record['id']} original report count changed")
        else:
            require(record["counts"] == audit.EXPECTED_RECOVERY[record["id"]]["counts"],
                    f"{record['id']} recovery report count changed")
            require(not errors, f"{record['id']} recovery report has errors")
        require(sha256_path(report_path.parent / record["log"]["path"])
                == record["log"]["sha256"], f"{record['id']} CI log digest changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite", choices=audit.EXPECTED_IDS)
    parser.add_argument("--check-report", type=Path)
    args = parser.parse_args()
    try:
        if args.check_report:
            report_path = args.check_report.resolve()
            verify_report(json.loads(report_path.read_text(encoding="utf-8")), report_path)
            print(f"PHASE XIX-V CI REPORT: PASS ({report_path})")
            return 0
        output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="phase19u-ci-"))
        output_dir.mkdir(parents=True, exist_ok=True)
        report = build_report(args.root, output_dir, args.suite)
        report_path = output_dir / "phase19u-historical-ci-results.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        verify_report(report, report_path)
        print(f"PHASE XIX-V HISTORICAL CI: PASS ({len(report['runs'])} runs)")
        print(report_path)
        return 0
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"PHASE XIX-V HISTORICAL CI: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
