#!/usr/bin/env python3
"""Sequential Intel/ROG Ally qualification with durable logs and strict exits.

Use a separate stable checkout. For b519361 on Windows, apply ONLY the supplied
oracle build adapter in an isolated checkout (see ROG_ALLY_HANDOFF.md).
No GPU benchmarks run concurrently and no numerical golden is re-recorded.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CRATE = Path(__file__).resolve().parents[1]


def protect_routes(paths, measure_and_verify, rebuild):
    """A failed tuning pass must leave the previously qualified routes intact."""
    originals = {path: path.read_bytes() for path in paths}
    accepted = False
    try:
        accepted = bool(measure_and_verify())
        return accepted
    finally:
        if not accepted:
            for path, contents in originals.items():
                path.write_bytes(contents)
            if sys.exc_info()[0] is not KeyboardInterrupt:
                rebuild()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", required=True, type=Path)
    parser.add_argument("--expected-vendor", choices=("intel", "amd"), required=True)
    parser.add_argument("--output", type=Path, default=CRATE / "target" / "handoff-validation")
    parser.add_argument("--tune", action="store_true", help="measure and merge ONLY this adapter's route rows, then requalify")
    args = parser.parse_args()
    for key in ("HIERARCHOS_VULKAN_FORCE_VENDOR", "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS",
                "HIERARCHOS_RWKV_CROSS_BATCH_ARRIVAL_SEED", "HIERARCHOS_RWKV_KEY_PARAM_DIRECT_WRITE_SINK"):
        if key in os.environ:
            parser.error(f"unset {key} before qualification")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def run(name: str, command: list[str], *, environment: dict | None = None) -> bool:
        print(f"START {name}", flush=True)
        started = time.monotonic()
        path = out / f"{name}.log"
        with path.open("w", encoding="utf-8") as log:
            if environment is None:
                environment = dict(os.environ)
            environment["HIERARCHOS_STABLE_ORACLE"] = str(args.stable.resolve())
            completed = subprocess.run([str(part) for part in command], cwd=CRATE,
                                       stdout=log, stderr=subprocess.STDOUT, env=environment)
        rows.append({"step": name, "command": [str(part) for part in command],
                     "exit_code": completed.returncode, "seconds": time.monotonic() - started,
                     "log": str(path)})
        (out / "qualification.json").write_text(json.dumps({"result": "running", "steps": rows}, indent=2) + "\n")
        print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} {name} ({rows[-1]['seconds']:.1f}s)", flush=True)
        if completed.returncode:
            print(path.read_text(encoding="utf-8", errors="replace")[-5000:], flush=True)
        return completed.returncode == 0

    def python(script: str, *arguments: object) -> list[str]:
        return [sys.executable, str(CRATE / "validation" / script), *map(str, arguments)]

    run("python-contracts", [sys.executable, "-m", "unittest", "discover", "-s", "validation", "-p", "test_*.py"])
    run("all-binaries", ["cargo", "check", "--bins", "-j", "1"])
    run("library-and-trainer", ["cargo", "test", "--lib", "--bin", "hierarchos-vulkan-train", "-j", "1", "--", "--test-threads=1"])
    run("two-rank-tcp", python("verify_network_training.py", "--output", out / "two-rank-tcp"))
    for crate in ("hierarchos-inference", "hierarchos-native-cli", "hierarchos-gui"):
        run(crate, ["cargo", "test", "--manifest-path", str(CRATE.parent / crate / "Cargo.toml"), "-j", "1"])
    stable_args = ["--stable", args.stable.resolve(), "--precision", "both", "--with-inference-logits",
                   "--keep-results", out / "recurrent-snapshots", "--record", out / "stable-recurrent.json"]
    run("stable-recurrent", python("check_stable_probe_parity.py", *stable_args))
    run("stable-logits", python("check_stable_logits.py", "--stable", args.stable.resolve(), "--output", out / "stable-logits"))
    run("rwkv-vendor-composition", python("verify_rwkv_packed_cell_composition.py", "--expected-vendor", args.expected_vendor,
                                         "--record", out / "rwkv-vendor-composition.json"))
    run("rwkv-cpu-bridge", python("verify_rwkv_cpu_bridge.py"))
    for profile in ("release", "debug"):
        options = ["--release"] if profile == "release" else []
        run(f"training-stack-{profile}", python("verify_training_stack.py", *options, "--receipt", out / f"training-stack-{profile}.json"))
    if args.tune and all(row["exit_code"] == 0 for row in rows):
        tuning_start = len(rows)
        route_paths = list((CRATE / "validation").glob("*_decisions.json"))
        backup = out / "routes-before-tuning"
        backup.mkdir(exist_ok=True)
        for path in route_paths:
            (backup / path.name).write_bytes(path.read_bytes())
        def measure_and_verify():
        # Every driver verifies its own A/B before recording a measured winner.
        # Records are device/shape/policy keyed; Intel rows survive the AMD merge.
            for script, record in (
                ("bench_rwkv_packed_cell_time_mix.py", "time_mix_projection_decisions.json"),
                ("bench_rwkv_packed_cell_channel_mix.py", "channel_mix_forward_decisions.json"),
                ("bench_rwkv_packed_cell_low_rank_forward.py", "low_rank_full_forward_decisions.json"),
                ("bench_rwkv_low_rank_fp16_forward.py", "low_rank_full_forward_decisions.json"),
                ("bench_rwkv_key_state_forward.py", "key_state_forward_decisions.json"),
            ):
                run(script.removesuffix(".py"), python(script, "--record-decisions", CRATE / "validation" / record,
                    "--record", out / (script.removesuffix(".py") + ".json")))
            for chain in ("forward", "remat", "backward"):
                for policy in ("fp32", "fp16-storage"):
                    run(f"tune-{chain}-{policy}", python("bench_rwkv_tbptt_packed_chains.py", "--chain", chain,
                        "--parameter-storage", policy, "--iterations", "40", "--record-decisions"))
                    run(f"receipt-{chain}-{policy}", python("bench_rwkv_tbptt_packed_chains.py", "--chain", chain,
                        "--parameter-storage", policy, "--iterations", "40", "--record", out / f"{chain}-{policy}.json"))
            run("stable-recurrent-after-tuning", python("check_stable_probe_parity.py", *stable_args))
            run("stable-logits-after-tuning", python("check_stable_logits.py", "--stable", args.stable.resolve(), "--output", out / "stable-logits-tuned"))
            run("training-stack-after-tuning", python("verify_training_stack.py", "--release", "--receipt", out / "training-stack-tuned.json"))
            return all(row["exit_code"] == 0 for row in rows[tuning_start:])
        def rebuild_original_routes():
            print("Tuning failed: restored the qualified route records; rebuilding their binaries.", flush=True)
            run("rebuild-restored-routes", ["cargo", "build", "--release", "-j", "1",
                "--bin", "hierarchos-vulkan-train", "--bin", "training_step_probe",
                "--bin", "hierarchos-vulkan-transformer-logits"])
        protect_routes(route_paths, measure_and_verify, rebuild_original_routes)
    result = "pass" if rows and all(row["exit_code"] == 0 for row in rows) else "fail"
    (out / "qualification.json").write_text(json.dumps({"result": result, "vendor": args.expected_vendor,
        "stable": str(args.stable.resolve()), "steps": rows}, indent=2) + "\n")
    print(f"{result.upper()}: {out / 'qualification.json'}")
    return int(result != "pass")


if __name__ == "__main__":
    sys.exit(main())
