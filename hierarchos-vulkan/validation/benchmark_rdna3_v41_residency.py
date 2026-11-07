"""Same-binary ABBA comparison of streamed and resident native V4.1 experts.

The default training configuration streams expert payloads per layer, so each
layer's routing counts gate which payloads are uploaded. Resident experts upload
once at model construction, which removes the per-layer counts readback and the
per-step expert copies at the cost of holding every expert in device memory.

Both arms run the same binary with software FMA. Final logits must agree within
the frozen acceptance gate (`max_abs < 2e-7`, `rtol = 0`); identical digests are
reported but not required, because residency can select different kernels.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
from pathlib import Path

CASES = ["native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5"]
NATIVE = {"native_vit_4x4", "native_vit_6x4"}
GATE = 2e-7
CLEAR = [
    "HIERARCHOS_V41_INDIVIDUAL_UPLOADS",
    "HIERARCHOS_VULKAN_PROFILE_KERNELS",
    "HIERARCHOS_VULKAN_SUBMISSION_REPORT",
    "HIERARCHOS_V41_NATIVE_DIAGNOSTICS",
    "HIERARCHOS_V41_HC_REPLAY",
    "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS",
    "HIERARCHOS_VULKAN_FORCE_VENDOR",
    "HIERARCHOS_VULKAN_LAYER_SESSION",
    "HIERARCHOS_MULTIMODAL_BENCH_RESIDENT_EXPERTS",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def max_abs(left: list, right: list) -> float:
    if len(left) != len(right):
        raise ValueError("logit length changed")
    return max((abs(f32(a) - f32(b)) for a, b in zip(left, right)), default=0.0)


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
    reference = {}
    worst = 0.0
    digests_identical = True
    runs = []

    for round_ in range(args.rounds):
        order = ["resident", "streamed"] if round_ % 2 == 0 else ["streamed", "resident"]
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
            if mode == "resident":
                env["HIERARCHOS_MULTIMODAL_BENCH_RESIDENT_EXPERTS"] = "1"

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
            for record in raw["records"]:
                samples = record["samples"]
                if len(samples) != args.steps or any(
                    not math.isfinite(sample["total_ms"]) or sample["total_ms"] <= 0
                    for sample in samples
                ):
                    raise ValueError("invalid timing samples")
                case = record["case"]
                if case not in reference:
                    reference[case] = record
                else:
                    worst = max(worst, max_abs(record["final_logits_bits"], reference[case]["final_logits_bits"]))
                    for key in ("image_digest", "adapter_digest"):
                        if record.get(key) and reference[case].get(key) and record[key] != reference[case][key]:
                            digests_identical = False
            runs.append(dict(round=round_ + 1, mode=mode, records=raw["records"]))

    if worst >= GATE:
        raise ValueError(f"logits left the frozen gate: max_abs={worst}")

    results = []
    for case in CASES:
        arms = {}
        for mode in ["streamed", "resident"]:
            rows = [
                next(record for record in run["records"] if record["case"] == case)
                for run in runs
                if run["mode"] == mode
            ]
            medians = [statistics.median(sample["total_ms"] for sample in row["samples"]) for row in rows]
            memory = sorted(
                row["memory"]["peak_live_buffer_bytes"]
                for row in rows
                if row.get("memory") and row["memory"].get("peak_live_buffer_bytes")
            )
            arms[mode] = dict(
                round_medians_ms=medians,
                median_ms=statistics.median(medians),
                submissions=sorted({sample["submissions"] for row in rows for sample in row["samples"]}),
                peak_live_buffer_bytes=memory,
            )
        results.append(
            dict(
                case=case,
                **arms,
                speedup=arms["streamed"]["median_ms"] / arms["resident"]["median_ms"],
            )
        )

    receipt = dict(
        schema=1,
        **{"pass": True},
        gate=dict(max_abs=GATE, rtol=0, observed_max_abs=worst),
        digests_identical=digests_identical,
        order="ABBA",
        rounds=args.rounds,
        measured_steps=args.steps,
        warmup_steps=2,
        binary_sha256=binary_sha,
        results=results,
        scope="same binary; software FMA in both arms; residency moves expert payloads to device memory and removes the per-step expert uploads",
    )
    (args.output / "benchmark.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
