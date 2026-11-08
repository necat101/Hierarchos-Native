#!/usr/bin/env python3
"""Verify the Phase XVI receipt without trusting its prose summary."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEBUG_BINARY_SHA = "8c23c033badea7cadff1540f2062ce6bac66fece7fb631948bcd8092e2666c42"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def exit_code(relative: str) -> int:
    return int((ROOT / relative).read_text(encoding="utf-8").strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def verify_manifest() -> None:
    hashes = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        hashes[relative] = digest
    require(hashes, "empty SHA256SUMS")
    for relative, expected in hashes.items():
        path = ROOT / relative
        require(path.is_file(), f"missing hashed file: {relative}")
        require(sha256(path) == expected, f"hash mismatch: {relative}")


def verify_source() -> None:
    source = load("source.json")
    require(source["source_commit"] == "4c973dd0e2b13d28b439f644f9cb74d65dfde329", "source commit changed")
    require(source["parent_commit"] == "da71b8894378f3096a8e82595cc2007f63dfedca", "parent changed")
    require(source["qualified_numerical_commit"] == "a2d68f96802f278fbf79f019d58454270989ade5", "numerical binding changed")
    require(source["production_numerical_sources_changed"] is False, "production numerical source changed")
    repository = ROOT.parents[4]
    for relative, expected in source["changed_files"].items():
        require(sha256(repository / relative) == expected, f"changed source hash differs: {relative}")


def verify_aggregate(run: str) -> None:
    aggregate = load(f"aggregate/{run}/qualification.json")
    require(aggregate.get("pass") is True, f"{run}: aggregate not pass")
    require(aggregate.get("complete") is True, f"{run}: aggregate not complete")
    require(len(aggregate.get("results", [])) == 9, f"{run}: stage count changed")
    require(aggregate.get("absolute_limit") == 2e-7, f"{run}: absolute limit changed")
    require(aggregate.get("rtol") == 0, f"{run}: rtol changed")
    require(aggregate.get("comparison") == "strict_less_than", f"{run}: comparison changed")
    for row in aggregate["results"]:
        require(row.get("pass") is True, f"{run}: stage failed: {row.get('stage')}")

    require((ROOT / f"aggregate/{run}/exit-code.txt").read_text().strip() == "0", f"{run}: nonzero exit")
    images = {
        "shared77": 77,
        "seq2seq": 6,
        "gemma4-legacy": 1,
        "gemma4-rgb": 1,
        "gemma3": 4,
        "llava": 4,
    }
    for stage, count in images.items():
        image = load(f"aggregate/{run}/{stage}/qualification.json")
        require(image.get("pass") is True, f"{run}/{stage}: image gate failed")
        require(len(image.get("results", [])) == count, f"{run}/{stage}: case count changed")
        require(image.get("rtol") == 0, f"{run}/{stage}: rtol changed")
        require(image.get("absolute_limit", image.get("absolute_logit_limit")) == 2e-7, f"{run}/{stage}: limit changed")

    text = load(f"aggregate/{run}/text166/qualification.json")
    require(text.get("pass") is True and len(text.get("records", [])) == 166, f"{run}: text inventory changed")
    require(all(record.get("pass") and 0 <= record.get("max_abs", 1) < 2e-7 for record in text["records"]), f"{run}: text tolerance failed")

    deepseek = load(f"aggregate/{run}/deepseek215/qualification.json")
    require(deepseek.get("pass") is True and deepseek.get("complete") is True, f"{run}: DeepSeek incomplete")
    require(deepseek.get("verified_passed") == deepseek.get("expected_passed") == 218, f"{run}: DeepSeek count changed")
    require(deepseek.get("expected_ignored") == 26, f"{run}: ignored count changed")
    require(len(deepseek.get("shards", [])) == 28, f"{run}: shard count changed")

    native = load(f"aggregate/{run}/native-v41/qualification.json")
    require(native.get("pass") is True and native.get("completed_tests") == native.get("expected_tests") == 30, f"{run}: native V4.1 count changed")

    hashes = aggregate.get("binary_sha256", {})
    require(DEBUG_BINARY_SHA in hashes.values(), f"{run}: qualified test binary hash absent")


def verify_ci() -> None:
    for relative in (
        "ci/actionlint.exit",
        "ci/actionlint-performance.exit",
        "ci/check-gpt2-current.exit",
        "ci/check-gpt2-historical.exit",
        "ci/check-multimodal-receipts.exit",
        "ci/checkpoint-resume.exit",
        "ci/diff-check.exit",
        "ci/phase15-verify.exit",
        "ci/release-build.exit",
        "ci/render-ab.exit",
        "ci/rustfmt-model_tests.exit",
        "ci/scratch-regression.exit",
        "ci/selector-real.exit",
        "ci/selector-synthetic.exit",
        "ci/test-gpt2-receipts.exit",
        "ci/test-multimodal-training.exit",
        "ci/worktree-smoke.exit",
        "ci/worktree-smoke-file.exit",
        "ci/worktree-smoke-remove.exit",
    ):
        require(exit_code(relative) == 0, f"nonzero recorded check: {relative}")
    require(exit_code("ci/actionlint-negative.exit") == 1, "actionlint negative control did not fail")
    require(exit_code("ci/selector-multiple.exit") == 2, "ambiguous selector control did not fail")

    require("context \"runner\" is not allowed here" in (ROOT / "ci/actionlint-negative.log").read_text(), "negative actionlint did not detect runner context")
    require("RUN_ROOT=D:/a/_temp/rdna3-v41-123" in (ROOT / "ci/path-normalization.log").read_text(), "Windows path was not normalized")
    require("expected one hierarchos_vulkan test executable" in (ROOT / "ci/selector-multiple.log").read_text(), "ambiguous selector reason missing")
    require("test result: ok. 1 passed" in (ROOT / "ci/scratch-regression.log").read_text(), "scratch regression did not run one test")
    require("test result: ok. 1 passed" in (ROOT / "ci/checkpoint-resume.log").read_text(), "checkpoint test did not run one test")
    require("cargo_toml_diff_exit=0" in (ROOT / "ci/profile-parity.log").read_text(), "candidate/baseline manifests differ")

    release = load("ci/release-build.json")
    require(release.get("exit") == 0, "release build did not succeed")
    require(release.get("profile", {}).get("opt_level") == "3", "candidate was not optimized")
    require(release.get("profile", {}).get("debug_assertions") is False, "candidate kept debug assertions")
    require(release.get("binary_sha256") == "ee4ae6f7fbdb711ebe4a7001147225d3f7be94de6c1ec8c5866e7dd864a65277", "release binary hash changed")

    enumeration = dict(line.split("=", 1) for line in (ROOT / "ci/deepseek-enumeration.txt").read_text().splitlines())
    require(enumeration == {"all": "245", "numerical": "244", "excluded": "1"}, "DeepSeek enumeration changed")
    require("PASS:" in (ROOT / "ci/phase15-verify.log").read_text(), "Phase XV verifier did not report PASS")


def main() -> int:
    verify_manifest()
    verify_source()
    verify_aggregate("full-2")
    verify_aggregate("full-3")
    verify_ci()
    print("PASS: Phase XVI receipt integrity, aggregate replays, CI evidence, and source bindings verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
