"""Same-binary ABBA ablation of native V4.1 reverse-graph compute sessions.

Both arms run the same binary with software FMA so the only difference is
whether the retained reverse graph is recorded into one compute session.
Every final logit word, image digest and adapter digest must match.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

CASES = ["native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5"]
NATIVE = {"native_vit_4x4", "native_vit_6x4"}
CLEAR = [
    "HIERARCHOS_V41_INDIVIDUAL_UPLOADS",
    "HIERARCHOS_VULKAN_PROFILE_KERNELS",
    "HIERARCHOS_VULKAN_SUBMISSION_REPORT",
    "HIERARCHOS_V41_NATIVE_DIAGNOSTICS",
    "HIERARCHOS_V41_HC_REPLAY",
    "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS",
    "HIERARCHOS_VULKAN_FORCE_VENDOR",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(record: dict) -> tuple:
    return (
        record["final_logits_bits"],
        record.get("image_digest"),
        record.get("adapter_digest"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=7)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or not 3 <= args.steps <= 100:
        raise ValueError("use an even number of paired rounds >=2 and 3..100 measured updates")

    binary = args.binary.resolve(strict=True)
    fixtures = args.fixtures.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    binary_sha = digest(binary)
    reference = None
    runs = []

    for round_ in range(args.rounds):
        order = ["session", "baseline"] if round_ % 2 == 0 else ["baseline", "session"]
        for mode in order:
            name = f"{round_ + 1}-{mode}"
            destination = (args.output / f"{name}.json").resolve()
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
            if mode == "baseline":
                env["HIERARCHOS_VULKAN_LAYER_SESSION"] = "0"
            else:
                env.pop("HIERARCHOS_VULKAN_LAYER_SESSION", None)

            print("Starting", name, flush=True)
            result = subprocess.run(
                [
                    str(binary),
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
            )
            log = result.stdout + result.stderr
            (args.output / f"{name}.log").write_text(log, encoding="utf-8")
            if result.returncode or "test result: ok. 1 passed; 0 failed; 0 ignored;" not in log:
                raise ValueError(f"{name}: benchmark failed")
            if digest(binary) != binary_sha:
                raise ValueError("benchmark binary changed")
            raw = json.loads(destination.read_text())
            if (
                raw["warmup"] != 2
                or raw["iterations"] != args.steps
                or len(raw["records"]) != len(CASES)
                or {record["case"] for record in raw["records"]} != set(CASES)
            ):
                raise ValueError("benchmark coverage changed")
            words = {record["case"]: signature(record) for record in raw["records"]}
            if reference is None:
                reference = words
            elif words != reference:
                raise ValueError(f"{name}: final logit or parameter words changed")
            for record in raw["records"]:
                samples = record["samples"]
                if len(samples) != args.steps or any(
                    not math.isfinite(sample["total_ms"]) or sample["total_ms"] <= 0
                    for sample in samples
                ):
                    raise ValueError("invalid timing samples")
            runs.append(dict(round=round_ + 1, mode=mode, records=raw["records"]))
            print(
                name,
                "word exact; submissions",
                {record["case"]: sorted({s["submissions"] for s in record["samples"]}) for record in raw["records"]},
                flush=True,
            )

    results = []
    for case in CASES:
        arms = {}
        for mode in ["baseline", "session"]:
            rows = [
                next(record for record in run["records"] if record["case"] == case)
                for run in runs
                if run["mode"] == mode
            ]
            medians = [statistics.median(sample["total_ms"] for sample in row["samples"]) for row in rows]
            arms[mode] = dict(
                round_medians_ms=medians,
                median_ms=statistics.median(medians),
                submissions=sorted({sample["submissions"] for row in rows for sample in row["samples"]}),
            )
        results.append(
            dict(
                case=case,
                **arms,
                speedup=arms["baseline"]["median_ms"] / arms["session"]["median_ms"],
            )
        )

    receipt = dict(
        schema=1,
        **{"pass": True},
        word_exact=True,
        order="ABBA",
        rounds=args.rounds,
        measured_steps=args.steps,
        warmup_steps=2,
        binary_sha256=binary_sha,
        results=results,
        scope="same binary; software FMA in both arms; session changes submission granularity only",
    )
    (args.output / "benchmark.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
