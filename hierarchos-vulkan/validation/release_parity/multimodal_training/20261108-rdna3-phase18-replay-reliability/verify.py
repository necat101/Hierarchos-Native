#!/usr/bin/env python3
"""Verify the sealed Phase XVIII checkpoint-replay receipt."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE_COMMIT = "d799890861af26fa8b33f03a5c17f7a34059c6fc"
BINARY_SHA = "cc558cc72012bc5bb0362093af6f9a3b5e8bdf07fdda06be3883d24d200dad62"
LIMIT = 2e-7


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
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        require(len(parts) == 2, f"bad manifest line: {line!r}")
        expected, relative = parts[0], parts[1].lstrip("*")
        path = ROOT / relative
        require(path.is_file(), f"manifest path missing: {relative}")
        require(digest(path) == expected, f"manifest digest mismatch: {relative}")
        entries.append(relative)
    require(len(entries) >= 30, "receipt manifest is unexpectedly small")
    require("README.md" in entries, "README.md is not sealed")
    require("source.json" in entries, "source.json is not sealed")
    require("verify.py" in entries, "verify.py is not sealed")


def check_source() -> dict:
    source = load(ROOT / "source.json")
    require(source["source_commit"] == SOURCE_COMMIT, "source commit changed")
    require(
        source["parent_commit"] == "259ccd63da89f30212dc34ba795d6b8e950bb999",
        "parent commit changed",
    )
    require(source["production_numerical_sources_changed"] is False,
            "receipt claims production numerical changes")
    require(source["test_only_source_changes"] is True,
            "receipt does not identify test-only changes")
    require(source["historical_receipts_modified"] is False,
            "historical receipts were modified")
    require(source["test_binary_sha256"] == BINARY_SHA, "test binary hash changed")
    require(digest(ROOT / "source.diff") == source["source_diff_sha256"],
            "source diff digest mismatch")
    require(digest(REPO / "hierarchos-vulkan/Cargo.toml") == source["cargo_toml_sha256"],
            "Cargo.toml digest mismatch")
    require(digest(REPO / "hierarchos-vulkan/Cargo.lock") == source["cargo_lock_sha256"],
            "Cargo.lock digest mismatch")
    for relative, expected in source["changed_files"].items():
        path = REPO / relative
        require(path.is_file(), f"changed source missing: {relative}")
        require(digest(path) == expected, f"changed source digest mismatch: {relative}")
    return source


def check_aggregate(run: str, expected_pass: bool) -> dict:
    root = ROOT / "aggregate" / run
    data = load(root / "qualification.json")
    require(data["complete"] is True, f"{run}: incomplete aggregate")
    require(data["pass"] is expected_pass, f"{run}: pass flag changed")
    require(len(data["results"]) == 9, f"{run}: stage count changed")
    require(data["absolute_limit"] == LIMIT, f"{run}: absolute limit changed")
    require(data["rtol"] == 0, f"{run}: rtol changed")
    require(data["comparison"] == "strict_less_than", f"{run}: comparison changed")
    for row in data["results"]:
        require(row["pass"] is True if expected_pass else row["pass"] is expected_pass
                or row["stage"] == "deepseek215",
                f"{run}: unexpected stage status for {row['stage']}")
    exit_code = (root / "exit-code.txt").read_text(encoding="utf-8").strip()
    require(exit_code == ("0" if expected_pass else "1"),
            f"{run}: exit-code evidence mismatch")
    return data


def check_counts(run: str) -> None:
    root = ROOT / "aggregate" / run
    expected = {
        "shared77": 77,
        "seq2seq": 6,
        "gemma4-legacy": 1,
        "gemma4-rgb": 1,
        "gemma3": 4,
        "llava": 4,
    }
    for stage, count in expected.items():
        data = load(root / stage / "qualification.json")
        require(data["pass"] is True, f"{run}/{stage}: stage not green")
        require(len(data["results"]) == count, f"{run}/{stage}: count changed")
    text = load(root / "text166" / "qualification.json")
    require(len(text["records"]) == 166, f"{run}/text166: count changed")
    ds = load(root / "deepseek215" / "qualification.json")
    require(ds["pass"] is True, f"{run}/deepseek: stage not green")
    require(ds["verified_passed"] == 218, f"{run}/deepseek: active count changed")
    require(ds["expected_passed"] == 218, f"{run}/deepseek: expected count changed")
    require(ds["expected_ignored"] == 26, f"{run}/deepseek: ignored count changed")
    require(len(ds["shards"]) == 28, f"{run}/deepseek: shard count changed")
    native = load(root / "native-v41" / "qualification.json")
    require(native["pass"] is True, f"{run}/native: stage not green")
    require(native["completed_tests"] == 30, f"{run}/native: completed count changed")
    require(native["expected_tests"] == 30, f"{run}/native: expected count changed")
    require(native["max_abs_logit"] == 0.0, f"{run}/native: logit error changed")
    require(native["binary_sha256"] == BINARY_SHA, f"{run}/native: binary changed")


def check_failed_run() -> None:
    root = ROOT / "aggregate" / "full-4-failed"
    data = load(root / "qualification.json")
    require(data["complete"] is True, "full-4: incomplete record")
    require(data["pass"] is False, "full-4: failure was relabeled as a pass")
    require(len(data["results"]) == 9, "full-4: stage count changed")
    deepseek = next(row for row in data["results"] if row["stage"] == "deepseek215")
    require(deepseek["pass"] is False, "full-4: failed stage was relabeled")
    require((root / "exit-code.txt").read_text(encoding="utf-8").strip() == "1",
            "full-4: failure exit code missing")
    log = (root / "deepseek215.log").read_text(encoding="utf-8", errors="replace")
    require("full suite enumeration changed" in log,
            "full-4: failure diagnostic missing")


def check_targeted() -> None:
    for name in (
        "checkpoint-serial-1",
        "checkpoint-serial-2",
        "checkpoint-serial-3",
        "checkpoint-parallel-1",
        "checkpoint-parallel-2",
    ):
        log = (ROOT / "ci" / "targeted" / f"{name}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        require("test result: ok. 1 passed" in log, f"{name}: checkpoint result missing")
        require("FAILED" not in log, f"{name}: checkpoint failure present")
    new_tests = {
        "temporary_test_roots_are_unique_under_concurrent_reservation":
            "checkpoint-concurrent.log",
        "temporary_test_roots_support_repeated_sequential_reservation":
            "checkpoint-sequential.log",
        "temporary_test_roots_do_not_reuse_stale_directories":
            "checkpoint-stale.log",
        "native_checkpoint_cleanup_is_scoped_to_test_owned_paths":
            "checkpoint-cleanup.log",
        "deepseek_v41_checkpoint_reads_sharded_f32_bf16_and_actual_compressor_keys":
            "checkpoint-compressor-keys.log",
    }
    for name, filename in new_tests.items():
        path = ROOT / "ci" / "targeted" / filename
        require(path.is_file(), f"missing targeted log for {name}")
        text = path.read_text(encoding="utf-8", errors="replace")
        require("test result: ok. 1 passed" in text, f"{name}: not green")
    path_log = (ROOT / "ci" / "path-smoke.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require("Ran 5 tests" in path_log and "\nOK\n" in path_log,
            "Windows path smoke test did not pass 5/5")
    require((ROOT / "ci" / "path-smoke.exit").read_text().strip() == "0",
            "Windows path smoke exit is not zero")


def check_ci() -> None:
    require((ROOT / "ci" / "actionlint.log").stat().st_size == 0,
            "actionlint produced diagnostics")
    require((ROOT / "ci" / "actionlint.exit").read_text().strip() == "0",
            "actionlint did not exit zero")
    for name in ("check-multimodal-receipts", "phase15-verify",
                 "check-gpt2-historical"):
        log = (ROOT / "ci" / f"{name}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        require("PASS" in log, f"{name}: PASS not recorded")
        require((ROOT / "ci" / f"{name}.exit").read_text().strip() == "0",
                f"{name}: exit is not zero")
    require((ROOT / "ci" / "rustfmt-changed.exit").read_text().strip() == "0",
            "allocator rustfmt check failed")
    require((ROOT / "ci" / "rustfmt-changed.log").stat().st_size == 0,
            "allocator rustfmt check emitted diagnostics")
    require((ROOT / "ci" / "rustfmt-native.exit").read_text().strip() == "1",
            "native-file formatting limitation record missing")
    native_fmt = (ROOT / "ci" / "rustfmt-native.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require("Diff in " in native_fmt,
            "native-file formatting limitation log missing diagnostics")
    require((ROOT / "ci" / "diff-check.exit").read_text().strip() == "0",
            "git diff --check failed")
    require((ROOT / "ci" / "cargo-fmt-check.exit").read_text().strip() == "1",
            "known cargo-fmt limitation record missing")
    fmt = (ROOT / "ci" / "cargo-fmt-check.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require("Diff in " in fmt, "cargo-fmt limitation log missing diagnostics")
    enumeration = (ROOT / "ci" / "deepseek-enumeration.txt").read_text(
        encoding="utf-8"
    )
    for line in ("discovered=248", "non_numerical=4", "numerical=244",
                 "active=218", "ignored=26"):
        require(line in enumeration, f"enumeration evidence missing {line}")


def check_gpt2() -> None:
    current = load(ROOT / "gpt2" / "current-qualification.json")
    require(current["pass"] is True, "current GPT-2 qualification is not green")
    require(current["source_head"] == SOURCE_COMMIT, "current GPT-2 source changed")
    require(current["shapes"]["cases"] == 16, "current GPT-2 shape count changed")
    require(current["shapes"]["worst_logit"] <= LIMIT,
            "current GPT-2 logit exceeds limit")
    bridge = load(ROOT / "gpt2" / "v41bridge-qualification.json")
    require(bridge["pass"] is True, "V4.1 bridge is not green")
    check = (ROOT / "gpt2" / "check-current.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require("CURRENT SOURCE QUALIFICATION: PASS" in check,
            "current GPT-2 receipt guard did not pass")
    require((ROOT / "gpt2" / "current-source.exit").read_text().strip() == "0",
            "current GPT-2 producer did not exit zero")
    require((ROOT / "gpt2" / "bridge.exit").read_text().strip() == "0",
            "V4.1 bridge did not exit zero")
    require((ROOT / "gpt2" / "check-current.exit").read_text().strip() == "0",
            "current GPT-2 guard did not exit zero")


def main() -> None:
    check_manifest()
    source = check_source()
    finished = check_aggregate("full-5", True)
    repeated = check_aggregate("full-6", True)
    check_failed_run()
    check_counts("full-5")
    check_counts("full-6")
    check_targeted()
    check_ci()
    check_gpt2()
    print(
        "PASS: Phase XVIII receipt verified; "
        f"source={source['source_commit'][:12]}, "
        f"binary={BINARY_SHA[:12]}, "
        f"full-5={finished['pass']}, full-6={repeated['pass']}, "
        "full-4=failed-diagnostic"
    )


if __name__ == "__main__":
    main()
