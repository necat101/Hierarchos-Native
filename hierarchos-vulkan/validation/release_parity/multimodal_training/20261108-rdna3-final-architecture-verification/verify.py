#!/usr/bin/env python3
"""Verify the sealed final-architecture-verification receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "3f0aaf541a5ed26b826515ca723bf25057dcea16"
LIMIT = 2e-7
BASELINE = "981e1576415c6fc0a4e54979a8a021a462a9d97a70fb3514ca2e9e7bc88bfd51"
DEBUG = "17aa6e998ea5763e083429bed49757ac9fa86167918908d4563bdbdd119a2355"
RELEASE = "11a83692282bf3e3d4122d4b7490c85fc3c29708837ce7578a12133460b9cf7d"


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def need(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    need(path.is_file(), f"missing {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_max(obj) -> float:
    values = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if "metric" in node and "max_abs" in node:
                if "logit" in str(node["metric"]).lower():
                    values.append(float(node["max_abs"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    return max(values) if values else 0.0


def check_manifest() -> None:
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    need(len(lines) >= 180, "manifest unexpectedly small")
    names = []
    for line in lines:
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        need(path.is_file(), f"manifest path missing: {relative}")
        need(digest(path) == expected, f"manifest digest mismatch: {relative}")
        names.append(relative)
    for required in ("README.md", "source.json", "receipt.json", "verify.py", "commands.md"):
        need(required in names, f"unsealed file: {required}")


def main() -> None:
    check_manifest()
    source = load(ROOT / "source.json")
    receipt = load(ROOT / "receipt.json")
    need(source["source_commit"] == SOURCE, "source commit changed")
    need(source["source_tree_clean_at_capture"] is True, "source was not clean")
    need(source["production_numerical_sources_changed"] is False, "production source change claimed")
    need(source["frozen_manifest"]["sha256"] ==
         "a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636",
         "frozen manifest changed")
    need(source["binaries"]["historical_baseline"]["sha256"] == BASELINE, "baseline hash changed")
    need(source["binaries"]["debug_harness"]["sha256"] == DEBUG, "debug harness changed")
    need(source["binaries"]["release_candidate"]["sha256"] == RELEASE, "release harness changed")
    capture = (ROOT / "source-status.txt").read_text(encoding="utf-8")
    need(f"source_head={SOURCE}" in capture, "source status head mismatch")
    need("source_status_begin\nsource_status_end" in capture, "capture was not clean")
    need((ROOT / "build/release-test.exit").read_text().strip() == "0", "release build failed")
    need((ROOT / "build/debug-test.exit").read_text().strip() == "0", "debug build failed")
    environment = (ROOT / "environment.txt").read_text(encoding="utf-8")
    need("AMD Radeon Graphics" in environment and "1002" in environment, "GPU evidence missing")

    full = load(ROOT / "aggregate/qualification.json")
    need(full["pass"] is True and full["complete"] is True, "aggregate not green")
    need(len(full["results"]) == 9, "aggregate stage count changed")
    need(full["absolute_limit"] == LIMIT, "absolute limit changed")
    need(full["rtol"] == 0 and full["comparison"] == "strict_less_than", "comparison changed")

    expected = {
        "shared77": 77,
        "seq2seq": 6,
        "gemma4-legacy": 1,
        "gemma4-rgb": 1,
        "gemma3": 4,
        "llava": 4,
        "text166": 166,
        "deepseek215": 218,
        "native-v41": 30,
    }
    worst = 0.0
    for stage, count in expected.items():
        data = load(ROOT / "aggregate" / stage / "qualification.json")
        need(data["pass"] is True, f"{stage} not green")
        if stage == "text166":
            got = len(data["records"])
            maximum = max(float(row["max_abs"]) for row in data["records"])
        elif stage == "deepseek215":
            got = data["verified_passed"]
            maximum = 0.0
            need(data["expected_ignored"] == 26, "DeepSeek ignored count changed")
            need(len(data["shards"]) == 28, "DeepSeek shard count changed")
        elif stage == "native-v41":
            got = data["completed_tests"]
            maximum = float(data["max_abs_logit"])
            need(maximum == 0.0, "native logit error changed")
        else:
            got = len(data["results"])
            maximum = metric_max(data)
        need(got == count, f"{stage} count changed: {got}")
        need(maximum <= LIMIT, f"{stage} exceeds logit limit")
        worst = max(worst, maximum)
    need(worst < LIMIT, "aggregate worst logit exceeds limit")

    gpt2 = load(ROOT / "numerical/gpt2-shapes-qualification.json")
    need(gpt2["pass"] is True and len(gpt2["results"]) == 16, "GPT-2 shapes changed")
    kimi = load(ROOT / "numerical/kimi-qualification.json")
    need(kimi["pass"] is True and len(kimi["results"]) == 3, "Kimi changed")
    bridge = load(ROOT / "numerical/v41-bridge-qualification.json")
    need(bridge["pass"] is True and len(bridge["results"]) == 2, "V4.1 bridge changed")
    need(bridge["max_abs_logit"] == 0.0, "V4.1 bridge logit changed")

    batch1 = load(ROOT / "benchmarks/b1/benchmark.json")
    need(batch1["pass"] is True, "batch-1 benchmark failed")
    need(batch1["rounds"] == 6 and batch1["measured_steps"] == 7, "batch-1 protocol changed")
    need(batch1["batch"] == 1 and not batch1["gate_failures"], "batch-1 gate changed")
    need(len(batch1["results"]) == 5, "batch-1 case count changed")
    need(all(row["conclusion"] == "candidate_faster" and row["speedup"] > 1
             for row in batch1["results"]), "batch-1 verdict changed")

    for batch in (1, 2, 4):
        session = load(ROOT / f"benchmarks/session-b{batch}/benchmark.json")
        need(session["pass"] is True, f"session batch {batch} failed")
        need(session["rounds"] == 6 and session["measured_steps"] == 7, "session protocol changed")
        need(session["batch"] == batch and not session["gate_failures"], "session gate changed")
        by_case = {row["case"]: row for row in session["results"]}
        for case in ("native_vit_4x4", "native_vit_6x4"):
            need(by_case[case]["conclusion"] == "candidate_faster",
                 f"session ViT verdict changed at batch {batch}")
        for case in ("llama", "gpt2", "t5"):
            need(by_case[case]["conclusion"] == "inconclusive",
                 f"session text verdict changed at batch {batch}")

    rust = (ROOT / "rust/full-suite.log").read_text(encoding="utf-8", errors="replace")
    need("running 1179 tests" in rust, "full suite discovery changed")
    need("1139 passed; 0 failed; 40 ignored" in rust, "full suite count changed")
    need((ROOT / "rust/full-suite.exit").read_text().strip() == "0", "full suite exit changed")
    primitives = (ROOT / "rust/gpt2-primitives.log").read_text(encoding="utf-8", errors="replace")
    need("9 passed; 0 failed; 1 ignored" in primitives, "GPT-2 primitive count changed")
    need((ROOT / "rust/gpt2-primitives.exit").read_text().strip() == "0", "GPT-2 primitives failed")
    need((ROOT / "attempt1/gpt2-shapes.precreate-failure.exit").read_text().strip() == "1",
         "setup-failure exit missing")
    failed = (ROOT / "attempt1/gpt2-shapes.precreate-failure.log").read_text(
        encoding="utf-8", errors="replace"
    )
    need("FileExistsError" in failed, "setup-failure diagnostic missing")
    need(receipt["status"] == "pass", "receipt status changed")
    print(
        "PASS: sealed final receipt verified; "
        f"source={SOURCE[:12]}, stages=9, cases=507, worst={worst:.17g}, "
        "full-suite=1139/40, batch1=5/5 positive, session text=inconclusive"
    )


if __name__ == "__main__":
    main()
