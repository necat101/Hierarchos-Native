"""Consume frozen Gen9 CPU references on the known RDNA3 adapter before push.

Uses Python's standard library; never prepares or rewrites CPU expectations.
Builds and GPU tests run sequentially. Supply the known RDNA3 PCI device ID;
the forward and generation gates verify it in every native report.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

from qualify_gen9_architectures import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = Path(__file__).resolve().parent
BINARIES = ["hierarchos-vulkan-transformer-logits", "transformer_generation_parity",
            "parameter_streaming_bench", "hierarchos-vulkan-deepseek-v41-forward-probe"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-fixtures", type=Path, required=True)
    parser.add_argument("--generation-fixtures", type=Path, required=True)
    parser.add_argument("--generation-packages", type=Path, required=True)
    parser.add_argument("--expected-device-id", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": 1, "pass": False, "target": "rdna3",
        "expected_vendor_id": 0x1002, "expected_device_id": args.expected_device_id,
        "absolute_limit": 2e-7, "rtol": 0, "comparison": "strict_less_than",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "runs": [],
        "limits": ["Retained scaled CPU checkpoints; not production throughput",
                   "SmolLM3 cache requests exercise the existing full-prefix fallback",
                   "120 registry identifiers remain outside this measured matrix"]}

    def run(label, command, tests=False):
        print("RUN", label, flush=True)
        command = [str(item) for item in command]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log = out / (label + ".log")
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        record = {"name": label, "command": command, "exit": completed.returncode,
                  "log": log.name, "sha256": digest(log)}
        receipt["runs"].append(record)
        write_json(out / "qualification.json", receipt)
        if completed.returncode:
            raise RuntimeError(f"{label} failed; see {log}")
        if tests:
            match = re.search(r"test result: ok\. (\d+) passed; 0 failed;", completed.stdout)
            if not match or int(match[1]) == 0:
                raise RuntimeError(f"{label} executed no passing tests")
            record["passed_tests"] = int(match[1])
        print("PASS", label, flush=True)
        return completed.stdout

    frozen = json.loads((ROOT / "HANDOFF-MANIFEST.json").read_text())
    for name, expected in frozen.items():
        if name.startswith("hierarchos-vulkan/validation/fixtures/"):
            path = ROOT / name
            if digest(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"original frozen fixture changed: {name}")
    # Hash every payload before compiling, including the original generation
    # package manifest, whose bytes differ from the reproduced forward manifest.
    for root in [args.forward_fixtures, args.generation_fixtures, args.generation_packages]:
        manifest = json.loads((root / "manifest.json").read_text())
        for case in manifest["cases"]:
            for name, expected in case["files"].items():
                if digest(root / name) != expected:
                    raise ValueError(f"frozen reference changed: {root / name}")
    generation = json.loads((args.generation_fixtures / "manifest.json").read_text())
    if digest(args.generation_packages / "manifest.json") != generation["source_manifest_sha256"]:
        raise ValueError("generation packages do not match the frozen source manifest")
    receipt["reference_manifests"] = {
        label: digest(root / "manifest.json") for label, root in [
            ("forward", args.forward_fixtures), ("generation", args.generation_fixtures),
            ("generation_packages", args.generation_packages)]}
    run("fixture-provenance", [sys.executable, VALIDATION / "fixture_toolchain.py", "check-all",
        "--expect-from", VALIDATION / "fixtures/deepseek_v41_mixed_block128.json"])
    run("harness-negative-controls", [sys.executable, "-m", "unittest", "discover",
        "-s", VALIDATION, "-p", "test_gen9_architectures.py"])
    common = ["--locked", "--manifest-path", ROOT / "hierarchos-vulkan/Cargo.toml", "-j1"]
    build = run("release-build", ["cargo", "build", *common, "--release", "--message-format=json",
        *[arg for name in BINARIES for arg in ["--bin", name]]])

    def artifacts(content):
        return [entry for line in content.splitlines() if line.startswith("{")
                for entry in [json.loads(line)] if entry.get("reason") == "compiler-artifact"
                and entry.get("executable")]

    binaries = {entry["target"]["name"]: Path(entry["executable"]) for entry in artifacts(build)}
    if set(BINARIES) - binaries.keys():
        raise ValueError("release build did not report every requested executable")
    receipt["binaries"] = {name: {"path": str(binaries[name]), "sha256": digest(binaries[name])}
                           for name in BINARIES}
    hardware = ["--target", "rdna3", "--expected-device-id", hex(args.expected_device_id)]
    # These gates establish the actual adapter before the longer unit suite.
    run("forward", [sys.executable, VALIDATION / "qualify_gen9_architectures.py", "run",
        "--fixtures", args.forward_fixtures.resolve(), "--binary", binaries[BINARIES[0]],
        "--output", out / "forward", *hardware])
    run("generation", [sys.executable, VALIDATION / "qualify_gen9_generation.py", "run",
        "--fixtures", args.generation_fixtures.resolve(), "--packages", args.generation_packages.resolve(),
        "--binary", binaries[BINARIES[1]], "--output", out / "generation", *hardware])
    build = run("unit-build", ["cargo", "test", *common, "--lib", "--no-run", "--message-format=json"])
    units = [Path(entry["executable"]) for entry in artifacts(build) if entry["profile"].get("test")]
    if len(units) != 1:
        raise ValueError(f"expected exactly one library test executable: {units}")
    for label, name in [("v41", "deepseek_v41::"), ("vulkan", "vulkan::tests"),
                        ("streaming", "parameter_streaming"), ("offload", "weight_offload::tests"),
                        ("kv-cache", "generation_kv_cache_matches_full_prefix"),
                        ("qwen-next-backward", "qwen3_next_gated_delta_end_to_end_backward_matches_finite_difference")]:
        run(label, [units[0], name, "--nocapture", "--test-threads=1"], tests=True)
    for label, name in [
        ("package-default", "deepseek_v41_default_package_scalable_release_strict_all_logits"),
        ("joint-resident", "deepseek_v41_joint_depth_and_384_expert_strict_all_logits"),
        ("joint-streamed", "deepseek_v41_streamed_vocabulary_joint_strict_all_logits"),
        ("peft-trajectory", "deepseek_v41_production_uninterrupted_training_logit_trajectory")]:
        run(label, [units[0], name, "--ignored", "--nocapture", "--test-threads=1"], tests=True)
    for name in BINARIES:
        if digest(binaries[name]) != receipt["binaries"][name]["sha256"]:
            raise ValueError(f"executable changed during qualification: {name}")
    receipt["pass"] = True
    write_json(out / "qualification.json", receipt)
    print(f"PASS RDNA3 final regression: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
