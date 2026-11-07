"""Paired ABBA benchmark for two V4.1 benchmark binaries or two modes.

The benchmark test is the same executable test used by the native qualifier:
five frozen cases, two warmups and a requested number of measured optimizer
updates.  Each round runs baseline and candidate in opposite order, so slow
shader-cache or thermal drift is not tied to one arm.

Performance gates use a paired bootstrap confidence interval over the
round-level median ratio.  There is deliberately no fixed percentage cutoff:
a control regression is reported only when the 95% interval lies below 1.0,
and a native improvement only when it lies above 1.0.  Dispersion, p95 and all
raw samples remain in the receipt so an unstable host can be rejected by the
reviewer or by a dedicated CI job.
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

CASES = ["native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5"]
NATIVE = {"native_vit_4x4", "native_vit_6x4"}
CONTROLS = set(CASES) - NATIVE
CLEAR = [
    "HIERARCHOS_V41_INDIVIDUAL_UPLOADS",
    "HIERARCHOS_VULKAN_PROFILE_KERNELS",
    "HIERARCHOS_VULKAN_SUBMISSION_REPORT",
    "HIERARCHOS_V41_NATIVE_DIAGNOSTICS",
    "HIERARCHOS_V41_HC_REPLAY",
    "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS",
    "HIERARCHOS_VULKAN_FORCE_VENDOR",
    "HIERARCHOS_MULTIMODAL_BENCH_RESIDENT_EXPERTS",
]
RESERVED = {
    "HIERARCHOS_MULTIMODAL_BENCH_OUTPUT",
    "HIERARCHOS_MULTIMODAL_BENCH_FIXTURES",
    "HIERARCHOS_MULTIMODAL_BENCH_STEPS",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(record):
    return (
        record["final_logits_bits"],
        record.get("image_digest"),
        record.get("adapter_digest"),
    )


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    index = max(0, min(len(ordered) - 1, math.ceil(percent / 100.0 * len(ordered)) - 1))
    return ordered[index]


def median_absolute_deviation(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def bootstrap_median_ci(values, draws, seed):
    if len(values) < 2:
        raise ValueError("at least two paired rounds are required")
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)
    ]
    return percentile(medians, 2.5), percentile(medians, 97.5)


def parse_env(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key or key in RESERVED:
            raise ValueError(f"invalid or reserved environment variable {key!r}")
        parsed[key] = value
    return parsed


def arm_summary(records):
    round_medians = [
        statistics.median(sample["total_ms"] for sample in record["samples"])
        for record in records
    ]
    sample_rows = [sample for record in records for sample in record["samples"]]
    samples = [sample["total_ms"] for sample in sample_rows]
    median_ms = statistics.median(round_medians)
    return {
        "round_medians_ms": round_medians,
        "median_ms": median_ms,
        "p95_ms": percentile(samples, 95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mad_ms": median_absolute_deviation(round_medians),
        "relative_mad": median_absolute_deviation(round_medians) / median_ms,
        "sample_cv": statistics.stdev(samples) / statistics.fmean(samples),
        "submissions": sorted({sample["submissions"] for sample in sample_rows}),
        "samples": len(samples),
        "signature_sha256": hashlib.sha256(
            repr(signature(records[0])).encode("utf-8")
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--baseline-env", action="append", default=[])
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--scope", default="paired baseline/candidate V4.1 training benchmark")
    parser.add_argument("--allow-cross-arm-signature-difference", action="store_true")
    parser.add_argument("--require-native-improvement", action="store_true")
    parser.add_argument("--forbid-control-regression", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or not 3 <= args.steps <= 100:
        parser.error("use an even number of paired rounds >=2 and 3..100 measured steps")
    if args.bootstrap_draws < 1000:
        parser.error("use at least 1000 bootstrap draws")

    baseline_binary = args.baseline_binary.resolve(strict=True)
    candidate_binary = args.candidate_binary.resolve(strict=True)
    fixtures = args.fixtures.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_env = parse_env(args.baseline_env)
    candidate_env = parse_env(args.candidate_env)
    binaries = {
        "baseline": baseline_binary,
        "candidate": candidate_binary,
    }
    binary_sha = {arm: digest(path) for arm, path in binaries.items()}
    arm_env = {"baseline": baseline_env, "candidate": candidate_env}
    runs = []
    arm_signatures = {}
    devices = set()

    for round_number in range(1, args.rounds + 1):
        order = ("baseline", "candidate") if round_number % 2 else ("candidate", "baseline")
        for arm in order:
            name = f"round-{round_number}-{arm}"
            destination = (output / f"{name}.json").resolve()
            env = {
                **os.environ,
                "HIERARCHOS_ATEN_VECTOR_WIDTH": "16",
                "HIERARCHOS_VULKAN_RDNA3_V41": "0",
                "HIERARCHOS_MULTIMODAL_BENCH_OUTPUT": str(destination),
                "HIERARCHOS_MULTIMODAL_BENCH_FIXTURES": str(fixtures),
                "HIERARCHOS_MULTIMODAL_BENCH_STEPS": str(args.steps),
            }
            for key in CLEAR:
                env.pop(key, None)
            env.update(arm_env[arm])
            print("Starting", name, flush=True)
            try:
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
            except subprocess.TimeoutExpired as error:
                log = (error.stdout or "") + (error.stderr or "")
                (output / f"{name}.log").write_text(log, encoding="utf-8")
                raise ValueError(f"{name}: benchmark timed out") from error
            log = result.stdout + result.stderr
            (output / f"{name}.log").write_text(log, encoding="utf-8")
            if result.returncode or (
                "test result: ok. 1 passed; 0 failed; 0 ignored;" not in log
            ):
                raise ValueError(f"{name}: benchmark test failed (exit {result.returncode})")
            if digest(binaries[arm]) != binary_sha[arm]:
                raise ValueError(f"{name}: benchmark binary changed")
            raw = json.loads(destination.read_text(encoding="utf-8"))
            records = raw["records"]
            if (
                raw["warmup"] != 2
                or raw["iterations"] != args.steps
                or len(records) != len(CASES)
                or {record["case"] for record in records} != set(CASES)
            ):
                raise ValueError(f"{name}: benchmark coverage changed")
            for record in records:
                samples = record["samples"]
                if len(samples) != args.steps or any(
                    not math.isfinite(sample["total_ms"]) or sample["total_ms"] <= 0
                    for sample in samples
                ):
                    raise ValueError(f"{name}: invalid timing samples")
            signatures = {record["case"]: signature(record) for record in records}
            if arm in arm_signatures and signatures != arm_signatures[arm]:
                raise ValueError(f"{name}: final words changed within the {arm} arm")
            arm_signatures.setdefault(arm, signatures)
            devices.add(raw.get("device"))
            runs.append(
                {
                    "round": round_number,
                    "arm": arm,
                    "binary_sha256": binary_sha[arm],
                    "records": records,
                }
            )
            print(
                name,
                "word-consistent; submissions",
                {
                    record["case"]: sorted(
                        {sample["submissions"] for sample in record["samples"]}
                    )
                    for record in records
                },
                flush=True,
            )

    mismatched = sorted(
        case
        for case in CASES
        if arm_signatures["baseline"][case] != arm_signatures["candidate"][case]
    )
    if mismatched and not args.allow_cross_arm_signature_difference:
        raise ValueError(f"cross-arm final words differ for {', '.join(mismatched)}")

    results = []
    for case in CASES:
        arms = {}
        for arm in ("baseline", "candidate"):
            records = [
                next(record for record in run["records"] if record["case"] == case)
                for run in runs
                if run["arm"] == arm
            ]
            arms[arm] = arm_summary(records)
        ratios = [
            baseline / candidate
            for baseline, candidate in zip(
                arms["baseline"]["round_medians_ms"],
                arms["candidate"]["round_medians_ms"],
            )
        ]
        ci_low, ci_high = bootstrap_median_ci(ratios, args.bootstrap_draws, args.seed)
        median_speedup = arms["baseline"]["median_ms"] / arms["candidate"]["median_ms"]
        paired_median = statistics.median(ratios)
        if ci_low > 1.0:
            conclusion = "candidate_faster"
        elif ci_high < 1.0:
            conclusion = "candidate_slower"
        else:
            conclusion = "inconclusive"
        results.append(
            {
                "case": case,
                "role": "native" if case in NATIVE else "control",
                **arms,
                "speedup": median_speedup,
                "paired_median_speedup": paired_median,
                "paired_round_speedups": ratios,
                "speedup_ci95": [ci_low, ci_high],
                "conclusion": conclusion,
            }
        )

    failures = []
    if mismatched and not args.allow_cross_arm_signature_difference:
        failures.append(f"final words differ: {', '.join(mismatched)}")
    if args.require_native_improvement:
        for result in results:
            if result["role"] == "native" and result["speedup_ci95"][0] <= 1.0:
                failures.append(
                    f"{result['case']}: native speedup CI does not exclude 1.0 "
                    f"({result['speedup_ci95']})"
                )
    if args.forbid_control_regression:
        for result in results:
            if result["role"] == "control" and result["speedup_ci95"][1] < 1.0:
                failures.append(
                    f"{result['case']}: control regression is significant "
                    f"({result['speedup_ci95']})"
                )

    receipt = {
        "schema": 1,
        "pass": not failures,
        "scope": args.scope,
        "order": "ABBA",
        "rounds": args.rounds,
        "measured_steps": args.steps,
        "warmup_steps": 2,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "cross_arm_word_exact": not mismatched,
        "cross_arm_mismatched_cases": mismatched,
        "allowed_cross_arm_signature_difference": args.allow_cross_arm_signature_difference,
        "devices": sorted(str(device) for device in devices),
        "fixtures": str(fixtures),
        "baseline": {
            "binary": str(baseline_binary),
            "sha256": binary_sha["baseline"],
            "env": baseline_env,
        },
        "candidate": {
            "binary": str(candidate_binary),
            "sha256": binary_sha["candidate"],
            "env": candidate_env,
        },
        "results": results,
        "gate_failures": failures,
    }
    (output / "benchmark.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "hashes.json":
            hashes[path.name] = digest(path)
    hashes["runner.py"] = digest(Path(__file__))
    (output / "hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)
    if failures:
        for failure in failures:
            print("FAIL:", failure, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
