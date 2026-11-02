"""Observe warm Gen9 latency while retaining CPU and storage parity guards.

Run after correctness qualification and without another build or GPU workload.
These retained small checkpoints are operator controls, not production throughput.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

from qualify_gen9_architectures import compare_logits, digest, words, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--families", nargs="+", default=["gpt2_wide", "llama_wide", "qwen4_exp_mixed_ple", "minimax_m3"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=11)
    p.add_argument("--baseline-resident", action="store_true",
                   help="baseline supports the explicit residency argument; compare both modes")
    args = p.parse_args()
    if args.rounds < 2 or args.warmup < 1 or args.iterations < 2:
        p.error("at least two rounds, one warmup and two measured iterations required")
    root = args.fixtures.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    cases = {case["family"]: case for case in manifest["cases"] if case["case"] == "unmasked"}
    if set(args.families) - cases.keys() or len(set(args.families)) != len(args.families):
        p.error("missing or duplicate benchmark profiles")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {name: path.resolve() for name, path in [("baseline", args.baseline), ("candidate", args.candidate)]}
    result = {"scope": "warm forward on retained scaled checkpoints; observation only",
        "manifest_sha256": digest(root / "manifest.json"),
        "binaries": {name: {"path": str(path), "sha256": digest(path)} for name, path in binaries.items()},
        "pass": False, "records": [], "summary": {}}
    for family in args.families:
        case = cases[family]
        for name, expected_hash in case["files"].items():
            if digest(root / name) != expected_hash:
                raise ValueError(f"frozen benchmark input changed: {name}")
        expected = json.loads((root / case["cpu"]).read_text())
        storage_words = None
        for round_index in range(args.rounds):
            order = [("baseline", "streaming"), ("candidate", "streaming"), ("candidate", "resident")]
            if args.baseline_resident:
                order.insert(0, ("baseline", "resident"))
            if round_index % 2:
                order.reverse()
            for version, mode in order:
                command = [str(binaries[version]), str(root / case["package"]), str(root / case["input"]),
                           str(args.warmup), str(args.iterations), "inference"]
                if version == "candidate" or args.baseline_resident:
                    command.append(mode)
                completed = subprocess.run(command, capture_output=True, text=True,
                    timeout=300, env={key: value for key, value in os.environ.items()
                        if key not in {"HIERARCHOS_VULKAN_PROFILE_KERNELS", "HIERARCHOS_VULKAN_VARIANT_TRACE"}})
                label = f"{family}-{round_index}-{version}-{mode}"
                (out / (label + ".log")).write_text(completed.stderr)
                if completed.returncode:
                    raise RuntimeError(f"benchmark {label} exited {completed.returncode}")
                report = json.loads(completed.stdout)
                write_json(out / (label + ".json"), report)
                actual = report.get("logits_bits")
                if actual is None:
                    actual = words(report["logits"])
                check = compare_logits(actual, expected["bits"], expected["visible"])
                if not check["pass"] or "HD Graphics 520" not in report["device"]:
                    raise ValueError(f"benchmark {label} failed CPU/device parity")
                if version == "candidate" or args.baseline_resident:
                    if report["residency"] != mode or (report["streamed_linears"] > 0) != (mode == "streaming"):
                        raise ValueError("benchmark residency differs")
                if storage_words is not None and actual != storage_words:
                    raise ValueError("baseline/candidate or storage logit words differ")
                storage_words = actual
                record = {"family": family, "round": round_index, "version": version,
                    "mode": mode, "command": command, "median_ms": report["median_ms"],
                    "cpu_max_abs": check["max_abs"], "submissions": report["submissions"],
                    "iterations": report["iterations"], "memory_before": report["memory_before"],
                    "io_before": report["io_before"], "io_after": report["io_after"]}
                result["records"].append(record)
                print(f"{label}: {record['median_ms']:.3f} ms", flush=True)
                write_json(out / "benchmark.json", result)
        medians = {f"{version}_{mode}": statistics.median(r["median_ms"] for r in result["records"]
            if r["family"] == family and r["version"] == version and r["mode"] == mode)
            for version, mode in order}
        result["summary"][family] = {**medians,
            "candidate_streaming_change_percent": 100 * (medians["candidate_streaming"] / medians["baseline_streaming"] - 1),
            "resident_speedup_vs_streaming": medians["candidate_streaming"] / medians["candidate_resident"]}
        if args.baseline_resident:
            result["summary"][family]["candidate_resident_change_percent"] = 100 * (
                medians["candidate_resident"] / medians["baseline_resident"] - 1)
    for name, path in binaries.items():
        if digest(path) != result["binaries"][name]["sha256"]:
            raise ValueError("benchmark executable changed")
    result["pass"] = True
    write_json(out / "benchmark.json", result)


if __name__ == "__main__":
    main()
