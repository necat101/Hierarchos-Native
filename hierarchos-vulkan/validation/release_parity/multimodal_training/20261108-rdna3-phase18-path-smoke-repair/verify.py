#!/usr/bin/env python3
"""Verify the narrow Phase XVIII Windows path-smoke repair receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE = "ce1757d3c2e362d811f32f9467ff065fb7dcf0b8"
BINARY = "cc558cc72012bc5bb0362093af6f9a3b5e8bdf07fdda06be3883d24d200dad62"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    require(path.is_file(), f"missing {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest() -> None:
    manifest = ROOT / "SHA256SUMS"
    require(manifest.is_file(), "missing SHA256SUMS")
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        path = ROOT / relative.lstrip("*")
        require(path.is_file(), f"manifest path missing: {relative}")
        require(digest(path) == expected, f"digest mismatch: {relative}")
        count += 1
    require(count >= 8, "repair manifest is unexpectedly small")


def check_source() -> None:
    source = load(ROOT / "source.json")
    require(source["source_commit"] == SOURCE, "source revision changed")
    require(source["parent_commit"] == "58ab4fbc7374bc44c64bc65e183155f277db9ea6",
            "parent revision changed")
    require(source["numerical_source_changed"] is False,
            "repair claims numerical source changes")
    require(source["test_binary_sha256"] == BINARY, "test binary changed")
    require(digest(ROOT / "source.diff") == source["source_diff_sha256"],
            "source diff digest mismatch")
    for relative, expected in source["changed_files"].items():
        path = REPO / relative
        require(path.is_file(), f"changed source missing: {relative}")
        require(digest(path) == expected, f"changed source digest mismatch: {relative}")
    main = REPO / "hierarchos-vulkan" / source["main_numerical_receipt"]["path"]
    require((main / "source.json").is_file(), "main numerical receipt missing")
    main_source = json.loads((main / "source.json").read_text(encoding="utf-8"))
    require(main_source["source_commit"] == "d799890861af26fa8b33f03a5c17f7a34059c6fc",
            "main numerical receipt source changed")
    require(source["path_smoke"]["exit_code"] == 0, "path smoke exit changed")
    require(source["path_smoke"]["tests"] == 5, "path smoke count changed")


def check_local_path_smoke() -> None:
    log = (ROOT / "ci" / "path-smoke.log").read_text(encoding="utf-8", errors="replace")
    require("Ran 5 tests" in log and "\nOK\n" in log,
            "local path smoke did not pass 5/5")


def check_remote() -> None:
    source = load(ROOT / "source.json")
    initial = (ROOT / "ci" / "initial-remote-failure.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require("FAIL: test_cygpath_build_log_paths_are_usable" in initial,
            "initial remote failure is not recorded")
    require("FAILED (failures=1)" in initial, "initial remote failure result missing")
    followup = load(ROOT / "ci" / "followup-remote-status.json")
    require(followup["status"] in {"in_progress", "completed"},
            "replacement remote run has no usable status")
    workflow = next(
        (job for job in followup["jobs"] if job["name"] == "workflow-validation"),
        None,
    )
    require(workflow is not None and workflow["conclusion"] == "success",
            "remote workflow validation did not succeed")
    workspace = next(
        (job for job in followup["jobs"] if job["name"] == "workspace-check"),
        None,
    )
    require(workspace is not None, "remote workspace job missing")
    path_step = next(
        (step for step in workspace["steps"]
         if step["name"] == "RDNA3 Windows path smoke test"),
        None,
    )
    require(path_step is not None and path_step["conclusion"] == "success",
            "replacement remote path smoke step did not succeed")
    require(followup["conclusion"] in {"", "failure", "success"},
            "unexpected remote conclusion")
    require(source["remote"]["repair_run"]["workspace_check_pass_claimed"] is False,
            "receipt incorrectly claims a remote workspace pass")


def check_current_source() -> None:
    current = load(ROOT / "gpt2" / "current-qualification.json")
    require(current["pass"] is True, "current GPT-2 qualification is not green")
    require(current["source_head"] == SOURCE, "current GPT-2 head changed")
    require(current["shapes"]["cases"] == 16, "current GPT-2 shape count changed")
    require(current["shapes"]["worst_logit"] <= 2e-7,
            "current GPT-2 logit exceeds limit")
    log = (ROOT / "gpt2" / "current-source.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require("CURRENT QUALIFICATION: PASS" in log,
            "current GPT-2 producer did not pass")
    check = (ROOT / "gpt2" / "check-current.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require("CURRENT SOURCE QUALIFICATION: PASS" in check,
            "current GPT-2 guard did not pass")


def main() -> None:
    check_manifest()
    check_source()
    check_local_path_smoke()
    check_remote()
    check_current_source()
    print(
        "PASS: Phase XVIII path-smoke repair verified; "
        f"source={SOURCE[:12]}, binary={BINARY[:12]}, "
        "remote-path-step=success, workspace-overall=not-claimed"
    )


if __name__ == "__main__":
    main()
