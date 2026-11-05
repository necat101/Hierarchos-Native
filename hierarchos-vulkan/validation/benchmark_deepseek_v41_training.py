"""Compare grouped and individual V4.1 parameter uploads in one libtest binary."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(record):
    return {key: record[key] for key in ("case", "kind", "final_logits_bits", "image_digest", "adapter_digest") if key in record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path,
                        help="Directory containing shared-image-cpu-v3 and seq2seq-image-cpu-v1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=7)
    args = parser.parse_args()
    if args.rounds < 2 or not 3 <= args.steps <= 100:
        parser.error("use at least two paired rounds and 3..100 measured steps")
    binary = args.binary.resolve(strict=True)
    fixtures = args.fixtures.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_sha = digest(binary)
    runs = []
    expected_cases = {"native_vit_4x4", "native_vit_6x4", "llama", "gpt2", "t5"}
    reference = None
    for round_ in range(args.rounds):
        # ABBA order compensates for shader-cache and thermal drift.
        order = ("individual", "grouped") if round_ % 2 == 0 else ("grouped", "individual")
        for mode in order:
            name = f"round-{round_ + 1}-{mode}"
            samples = output / (name + ".json")
            env = dict(os.environ, HIERARCHOS_MULTIMODAL_BENCH_OUTPUT=str(samples),
                       HIERARCHOS_MULTIMODAL_BENCH_FIXTURES=str(fixtures),
                       HIERARCHOS_MULTIMODAL_BENCH_STEPS=str(args.steps))
            for key in ("HIERARCHOS_VULKAN_PROFILE_KERNELS", "HIERARCHOS_VULKAN_SUBMISSION_REPORT",
                        "HIERARCHOS_V41_NATIVE_DIAGNOSTICS", "HIERARCHOS_V41_HC_REPLAY",
                        "HIERARCHOS_V41_INDIVIDUAL_UPLOADS"):
                env.pop(key, None)
            if mode == "individual":
                env["HIERARCHOS_V41_INDIVIDUAL_UPLOADS"] = "1"
            print("Starting", name, flush=True)
            result = subprocess.run([str(binary), "multimodal_training_throughput_benchmark",
                                     "--ignored", "--nocapture", "--test-threads=1"],
                                    env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            log = result.stdout + result.stderr
            (output / (name + ".log")).write_text(log, encoding="utf-8")
            if result.returncode or not re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", log):
                raise ValueError(f"{name}: benchmark did not pass exactly one test")
            if digest(binary) != binary_sha:
                raise ValueError("Executable changed during the comparison")
            raw = json.loads(samples.read_text(encoding="utf-8"))
            records = raw["records"]
            if len(records) != 5 or {r["case"] for r in records} != expected_cases or raw["iterations"] != args.steps:
                raise ValueError("Incomplete benchmark cases or iterations")
            signatures = {r["case"]: signature(r) for r in records}
            if reference is None:
                reference = signatures
            if signatures != reference:
                raise ValueError(f"{name}: trained logits or image/adapter words changed")
            for record in records:
                values = record["samples"]
                if len(values) != args.steps or any(not math.isfinite(s["total_ms"]) or s["total_ms"] <= 0 for s in values):
                    raise ValueError("Invalid timing samples")
            runs.append(dict(round=round_ + 1, mode=mode, records=records))
            (output / "run-status.json").write_text(json.dumps([
                dict(round=r["round"], mode=r["mode"], word_exact=True) for r in runs], indent=2) + "\n")
            print(name, "word-exact after", raw["warmup"] + args.steps, "updates", flush=True)
    summary = []
    for case in sorted(expected_cases):
        arms = {}
        for mode in ("individual", "grouped"):
            records = [next(r for r in run["records"] if r["case"] == case) for run in runs if run["mode"] == mode]
            per_run = [statistics.median(s["total_ms"] for s in r["samples"]) for r in records]
            counts = sorted({s["submissions"] for r in records for s in r["samples"]})
            arms[mode] = dict(median_ms=statistics.median(per_run), round_medians_ms=per_run, submissions=counts)
        summary.append(dict(case=case, **arms, speedup=arms["individual"]["median_ms"] / arms["grouped"]["median_ms"]))
    receipt = dict(schema=1, **{"pass": True}, word_exact=True, binary_sha256=binary_sha,
                   runner_sha256=digest(Path(__file__)), rounds=args.rounds, measured_steps=args.steps,
                   warmup_steps=2, updates_before_digest=2 + args.steps, results=summary,
                   scope="same-binary comparison on frozen tiny training geometry; host/GPU otherwise idle")
    (output / "benchmark.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
