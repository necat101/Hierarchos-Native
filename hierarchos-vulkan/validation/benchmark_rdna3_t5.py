"""Dedicated paired T5 gate for the native multimodal training benchmark.

This runner deliberately measures only the shared T5 image/encoder/decoder case.
It uses the same ABBA ordering, warmup, step count and bootstrap confidence
interval as the five-case V4.1 harness, but does not let a strong control-case
result hide a T5 regression.
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path

CASE = "t5"
CLEAR = [
    "HIERARCHOS_V41_INDIVIDUAL_UPLOADS",
    "HIERARCHOS_VULKAN_PROFILE_KERNELS",
    "HIERARCHOS_VULKAN_SUBMISSION_REPORT",
    "HIERARCHOS_V41_NATIVE_DIAGNOSTICS",
    "HIERARCHOS_V41_HC_REPLAY",
    "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS",
    "HIERARCHOS_VULKAN_FORCE_VENDOR",
    "HIERARCHOS_MULTIMODAL_BENCH_RESIDENT_EXPERTS",
    "HIERARCHOS_MULTIMODAL_BENCH_TELEMETRY",
    "HIERARCHOS_MULTIMODAL_BENCH_ALLOW_PROFILE",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, percent):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percent / 100.0 * len(ordered)) - 1))
    return ordered[index]


def bootstrap_ci(values, draws, seed):
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]
    return percentile(medians, 2.5), percentile(medians, 97.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-env", action="append", default=[])
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--allow-signature-difference", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or not 3 <= args.steps <= 100:
        parser.error("use an even number of paired rounds >=2 and 3..100 measured steps")

    def parse_env(items):
        parsed = {}
        for item in items:
            key, value = item.split("=", 1)
            if not key:
                raise ValueError(f"invalid environment assignment {item!r}")
            parsed[key] = value
        return parsed

    baseline = args.baseline_binary.resolve(strict=True)
    candidate = args.candidate_binary.resolve(strict=True)
    fixtures = args.fixtures.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binaries = {"baseline": baseline, "candidate": candidate}
    hashes = {arm: digest(path) for arm, path in binaries.items()}
    arm_env = {
        "baseline": parse_env(args.baseline_env),
        "candidate": parse_env(args.candidate_env),
    }
    rounds = []
    signatures = {"baseline": [], "candidate": []}

    for number in range(1, args.rounds + 1):
        order = ("baseline", "candidate") if number % 2 else ("candidate", "baseline")
        for arm in order:
            name = f"round-{number}-{arm}"
            destination = (output / f"{name}.json").resolve()
            env = {
                **os.environ,
                "HIERARCHOS_ATEN_VECTOR_WIDTH": "16",
                "HIERARCHOS_VULKAN_RDNA3_V41": "0",
                "HIERARCHOS_MULTIMODAL_BENCH_CASE": CASE,
                "HIERARCHOS_MULTIMODAL_BENCH_OUTPUT": str(destination),
                "HIERARCHOS_MULTIMODAL_BENCH_FIXTURES": str(fixtures),
                "HIERARCHOS_MULTIMODAL_BENCH_STEPS": str(args.steps),
            }
            for key in CLEAR:
                env.pop(key, None)
            env.update(arm_env[arm])
            print("Starting", name, flush=True)
            result = subprocess.run(
                [
                    str(binaries[arm]),
                    "multimodal_training_throughput_benchmark",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout_seconds,
            )
            (output / f"{name}.log").write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            if result.returncode or "test result: ok. 1 passed; 0 failed;" not in (
                result.stdout + result.stderr
            ):
                raise ValueError(f"{name}: benchmark failed")
            if digest(binaries[arm]) != hashes[arm]:
                raise ValueError(f"{name}: binary changed")
            raw = json.loads(destination.read_text(encoding="utf-8"))
            if raw.get("warmup") != 2 or raw.get("iterations") != args.steps:
                raise ValueError(f"{name}: warmup/step count changed")
            records = raw.get("records", [])
            if len(records) != 1 or records[0].get("case") != CASE:
                raise ValueError(f"{name}: expected exactly one T5 record")
            samples = records[0].get("samples", [])
            if len(samples) != args.steps or any(
                not math.isfinite(sample["total_ms"]) or sample["total_ms"] <= 0
                for sample in samples
            ):
                raise ValueError(f"{name}: invalid T5 samples")
            signatures[arm].append(
                (
                    records[0]["final_logits_bits"],
                    records[0].get("image_digest"),
                    records[0].get("adapter_digest"),
                )
            )
            rounds.append(
                {
                    "round": number,
                    "arm": arm,
                    "median_ms": statistics.median(
                        sample["total_ms"] for sample in samples
                    ),
                    "submissions": sorted({sample["submissions"] for sample in samples}),
                }
            )

    if not args.allow_signature_difference and set(signatures["baseline"] + signatures["candidate"]) != {
        signatures["baseline"][0]
    }:
        raise ValueError("T5 final logits/image/adapter signatures differ between arms")

    by_round = {}
    for row in rounds:
        by_round.setdefault(row["round"], {})[row["arm"]] = row
    ratios = [
        data["baseline"]["median_ms"] / data["candidate"]["median_ms"]
        for data in by_round.values()
    ]
    ci = bootstrap_ci(ratios, args.bootstrap_draws, args.seed)
    if ci[0] > 1.0:
        conclusion = "pass_t5_faster"
    elif ci[1] < 1.0:
        conclusion = "fail_t5_regression"
    else:
        conclusion = "inconclusive"
    summary = {
        "schema": 1,
        "case": CASE,
        "rounds": args.rounds,
        "steps": args.steps,
        "baseline_binary_sha256": hashes["baseline"],
        "candidate_binary_sha256": hashes["candidate"],
        "paired_round_speedups": ratios,
        "speedup_median": statistics.median(ratios),
        "speedup_ci95": list(ci),
        "conclusion": conclusion,
        "pass": conclusion == "pass_t5_faster",
        "rounds_data": rounds,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
