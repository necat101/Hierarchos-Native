#!/usr/bin/env python3
"""Fail closed on the adapter-span optimization receipt.

The check re-derives every performance median and speedup from the retained raw
JSON samples, verifies the ABBA round count and binary identities, enforces the
two native confidence-interval improvements, allows only the documented GPT-2
historical word difference, and checks the correctness and source manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "validation/release_parity/multimodal_training/20261107-rdna3-v41-adapter-span-batch"
CASES = ["native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5"]
NATIVE = {"native_vit_4x4", "native_vit_6x4"}
CONTROLS = set(CASES) - NATIVE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def performance(receipt, name, baseline_key, candidate_key, exact, mismatch):
    directory = receipt / name
    result = load(directory / "benchmark.json")
    require(result.get("pass") is True, f"{name}: benchmark did not pass")
    require(result.get("gate_failures") == [], f"{name}: gate failures retained")
    require(result.get("rounds", 0) >= 6, f"{name}: fewer than six ABBA rounds")
    require(result.get("measured_steps") == 7, f"{name}: wrong measured step count")
    require(result.get("warmup_steps") == 2, f"{name}: wrong warmup count")
    require(
        result.get("cross_arm_word_exact") is exact,
        f"{name}: cross-arm word-exact flag changed",
    )
    require(
        result.get("cross_arm_mismatched_cases") == mismatch,
        f"{name}: cross-arm mismatch set changed",
    )
    binaries = load(receipt / "binaries.json")
    require(
        result["baseline"]["sha256"] == binaries[baseline_key]["sha256"],
        f"{name}: baseline binary hash changed",
    )
    require(
        result["candidate"]["sha256"] == binaries[candidate_key]["sha256"],
        f"{name}: candidate binary hash changed",
    )
    require(
        [row["case"] for row in result["results"]] == CASES,
        f"{name}: case order/coverage changed",
    )
    by_case = {row["case"]: row for row in result["results"]}
    for case in CASES:
        row = by_case[case]
        baseline = []
        candidate = []
        for round_number in range(1, result["rounds"] + 1):
            for arm, target in (("baseline", baseline), ("candidate", candidate)):
                raw = load(directory / f"round-{round_number}-{arm}.json")
                require(raw.get("warmup") == 2, f"{name}: raw warmup changed")
                require(raw.get("iterations") == 7, f"{name}: raw iterations changed")
                records = raw.get("records", [])
                require(len(records) == 5, f"{name}: raw case count changed")
                record = next(item for item in records if item["case"] == case)
                samples = record.get("samples", [])
                require(len(samples) == 7, f"{name}: raw sample count changed")
                require(
                    all(
                        math.isfinite(sample["total_ms"]) and sample["total_ms"] > 0
                        for sample in samples
                    ),
                    f"{name}: invalid raw timing",
                )
                target.append(statistics.median(sample["total_ms"] for sample in samples))
        require(
            all(
                abs(left - right) < 1e-9
                for left, right in zip(
                    row["baseline"]["round_medians_ms"], baseline
                )
            ),
            f"{name}: baseline round medians do not re-derive",
        )
        require(
            all(
                abs(left - right) < 1e-9
                for left, right in zip(
                    row["candidate"]["round_medians_ms"], candidate
                )
            ),
            f"{name}: candidate round medians do not re-derive",
        )
        baseline_median = statistics.median(baseline)
        candidate_median = statistics.median(candidate)
        require(
            abs(row["baseline"]["median_ms"] - baseline_median) < 1e-9,
            f"{name}: baseline median does not re-derive",
        )
        require(
            abs(row["candidate"]["median_ms"] - candidate_median) < 1e-9,
            f"{name}: candidate median does not re-derive",
        )
        require(
            abs(row["speedup"] - baseline_median / candidate_median) < 1e-9,
            f"{name}: speedup does not re-derive",
        )
        low, high = row["speedup_ci95"]
        require(0 < low <= high, f"{name}: invalid confidence interval")
        if case in NATIVE:
            require(low > 1.0, f"{name}: native interval does not exclude 1.0")
        else:
            require(high >= 1.0, f"{name}: control regression is significant")
    hashes = load(directory / "hashes.json")
    for filename, expected in hashes.items():
        if filename == "runner.py":
            continue
        require(
            (directory / filename).is_file()
            and digest(directory / filename) == expected,
            f"{name}: raw manifest hash mismatch for {filename}",
        )
    print(f"PASS {name}: {result['rounds']} ABBA rounds, five cases, CIs re-derived")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    try:
        performance(receipt, "before-after", "before_adapter_batch", "optimized", True, [])
        performance(receipt, "session", "optimized", "optimized", True, [])
        performance(
            receipt,
            "historical",
            "historical_372407b",
            "optimized",
            False,
            ["gpt2"],
        )

        full = load(receipt / "correctness/full-replay-qualification.json")
        require(full.get("pass") is True and full.get("complete") is True, "full replay incomplete")
        require(len(full.get("results", [])) == 9, "full replay stage count changed")
        require(all(row.get("pass") is True for row in full["results"]), "full replay has a failed stage")
        require(full.get("unchanged_frozen_files") == 2748, "frozen file count changed")
        deepseek = load(receipt / "correctness/deepseek-suite-qualification.json")
        require(deepseek.get("pass") is True, "DeepSeek suite did not pass")
        require(deepseek.get("expected_tests") == 244, "DeepSeek total inventory changed")
        require(deepseek.get("verified_passed") == 218, "DeepSeek active count changed")
        require(len(deepseek.get("ignored", [])) == 26, "DeepSeek ignored count changed")
        native = load(receipt / "correctness/native-v41-qualification.json")
        require(native.get("pass") is True, "native V4.1 did not pass")
        require(native.get("completed_tests") == 30 == native.get("expected_tests"), "native V4.1 count changed")
        require(native.get("max_abs_logit") == 0.0, "native V4.1 logits are not exact")
        native_mm = load(receipt / "correctness/native-multimodal-qualification.json")
        require(native_mm.get("pass") is True, "native multimodal track did not pass")
        compressor = (receipt / "correctness/compressor-regression-tests.log").read_text(
            encoding="utf-8"
        )
        require("test result: ok. 2 passed; 0 failed;" in compressor, "compressor regressions did not pass")
        adapter = (receipt / "correctness/adapter-span-test.log").read_text(encoding="utf-8")
        require("1 passed; 0 failed;" in adapter, "adapter-span regression did not pass")
        python_log = (receipt / "correctness/python-unittest.log").read_text(encoding="utf-8")
        require("FAILED" not in python_log, "Python regression suite failed")
        require("OK" in python_log, "Python regression suite has no OK result")
        require((receipt / "correctness/prefill-decode.json").is_file(), "missing prefill/decode receipt")
        prefill = load(receipt / "correctness/prefill-decode.json")
        timings = prefill.get("timings")
        require(
            isinstance(timings, dict) and set(timings) == {"resident", "default"},
            "prefill/decode residency set changed",
        )
        require(
            isinstance(prefill.get("limits"), list)
            and len(prefill["limits"]) == 3
            and any("digest guards" in item for item in prefill["limits"]),
            "prefill/decode limits changed",
        )
        probe_binaries = load(receipt / "binaries.json")
        require(
            prefill["binaries"]["baseline"]["sha256"]
            == probe_binaries["forward_probe_baseline_372407b"]["sha256"],
            "prefill/decode baseline probe hash changed",
        )
        require(
            prefill["binaries"]["candidate"]["sha256"]
            == probe_binaries["forward_probe_optimized"]["sha256"],
            "prefill/decode candidate probe hash changed",
        )
        runs = prefill.get("runs")
        require(isinstance(runs, list) and len(runs) >= 6, "prefill/decode runs missing")
        for run in runs:
            require(bool(re.fullmatch(r"[0-9a-f]{16}", run.get("digest", ""))), "invalid prefill digest")
            phases = run.get("phases")
            require(isinstance(phases, list) and phases, "prefill/decode phases missing")
            require(
                all(
                    phase.get("phase") in {"prefill", "decode"}
                    and math.isfinite(phase.get("ms", 0.0))
                    and phase["ms"] > 0
                    for phase in phases
                ),
                "invalid prefill/decode phase sample",
            )
        for mode, arms in timings.items():
            require(
                set(arms) >= {"baseline", "candidate", "latency_reduction_percent"},
                f"{mode}: incomplete prefill/decode arms",
            )
            for arm in ("baseline", "candidate"):
                for phase in ("prefill_ms", "decode_ms"):
                    require(
                        math.isfinite(arms[arm].get(phase, 0.0))
                        and arms[arm][phase] > 0,
                        f"{mode}: invalid {arm} {phase}",
                    )
            for phase in ("prefill_ms", "decode_ms"):
                baseline = arms["baseline"][phase]
                candidate = arms["candidate"][phase]
                expected = (baseline - candidate) / baseline * 100.0
                require(
                    abs(arms["latency_reduction_percent"][phase] - expected) < 1e-9,
                    f"{mode}: {phase} reduction does not re-derive",
                )

        commit = load(receipt / "commit.json")
        require(bool(re.fullmatch(r"[0-9a-f]{40}", commit.get("source_commit", ""))), "invalid source commit")
        sources = load(receipt / "source-sha256.json")
        require(len(sources) > 100, "source manifest is unexpectedly small")
        for name, expected in sources.items():
            train = ROOT.parent / name
            require(train.is_file(), f"missing bound source {name}")
            require(digest(train) == expected, f"source hash mismatch: {name}")

        manifest = load(receipt / "receipt-hashes.json")
        bound = manifest.get("files", {})
        for name in [
            "README.md",
            "binaries.json",
            "commit.json",
            "source-sha256.json",
            "benchmark-before-after.json",
            "benchmark-session.json",
            "benchmark-historical.json",
        ]:
            require(name in bound, f"receipt manifest omits {name}")
            require((receipt / name).is_file(), f"receipt omits {name}")
            require(
                digest(receipt / name) == bound[name],
                f"receipt hash mismatch: {name}",
            )
        for name, expected in bound.items():
            require((receipt / name).is_file(), f"listed receipt file missing: {name}")
            require(digest(receipt / name) == expected, f"listed receipt hash mismatch: {name}")
    except (FileNotFoundError, KeyError, ValueError, StopIteration) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: adapter-span optimization receipt is internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
