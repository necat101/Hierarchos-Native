#!/usr/bin/env python3
"""Scoped Phase II reproduction: RDNA3 two-pass MHA attention backward.

Reuses the shipped qualification harness protocol (the same binary, flags,
warmup, ABBA ordering and report fields as
validation/qualify_training_phase16.py) with 5 ABBA rounds and 26 steps, which
is the protocol the historical narrative claimed.

Outputs one JSON document with:
  * per-kernel GPU timestamps from HIERARCHOS_VULKAN_PROFILE_KERNELS,
  * per-round paired whole-step samples and ratios with a Student-t interval,
  * dispatch counts from the profiler and the one-shot diagnostic line,
  * bitwise comparison of per-step losses and exported LoRA adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

CRATE = Path(__file__).resolve().parents[2]
BIN = CRATE / "target/release/hierarchos-vulkan-training-qualification.exe"
MODEL = CRATE / "target/wmma-fixture/model"
DATASET = CRATE / "target/wmma-fixture/dataset.jsonl"
sys.path.insert(0, str(CRATE / "validation"))
from abba_stats import mean_ci, t_critical  # noqa: E402

DEVICE_RE = re.compile(
    r"hierarchos_vulkan_kernel_profile device=.* dispatches=(\d+) gpu_ms=([0-9.]+)"
)
SHADER_RE = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=\S+ "
    r"dispatches=(\d+) gpu_ms=([0-9.]+)"
)
SCALAR = "transformer_attention_backward"
QUERY = "transformer_attention_backward_query_rdna3"
KEY = "transformer_attention_backward_key_rdna3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_profile(text: str) -> dict:
    blocks = []
    current = None
    for line in text.splitlines():
        match = DEVICE_RE.search(line)
        if match:
            if current is not None and current["dispatches"] > 0:
                blocks.append(current)
            current = {
                "dispatches": int(match.group(1)),
                "gpu_ms": float(match.group(2)),
                "shaders": {},
            }
            continue
        if current is None:
            continue
        shader = SHADER_RE.search(line)
        if shader:
            current["shaders"][shader.group(1)] = {
                "dispatches": int(shader.group(2)),
                "gpu_ms": float(shader.group(3)),
            }
    if current is not None and current["dispatches"] > 0:
        blocks.append(current)
    totals: dict[str, dict] = {}
    for block in blocks:
        for name, values in block["shaders"].items():
            entry = totals.setdefault(name, {"dispatches": 0, "gpu_ms": 0.0})
            entry["dispatches"] += values["dispatches"]
            entry["gpu_ms"] += values["gpu_ms"]
    return {"blocks": blocks, "totals": totals}


def run(tag: str, two_pass: bool, steps: int, warmup: int, profile: bool,
        lora_rank: int | None, adapter_out: Path | None, work: Path,
        digest_every: int = 0) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    report = work / f"{tag}.report.json"
    out = work / f"{tag}.scratch"
    stdout_path = work / f"{tag}.out.txt"
    stderr_path = work / f"{tag}.err.txt"
    cmd = [
        str(BIN), "--model", str(MODEL), "--dataset", str(DATASET),
        "--output", str(out), "--no-export", "--steps", str(steps),
        "--warmup", str(warmup), "--digest-every", str(digest_every),
        "--seed", "7", "--seq", "64", "--report", str(report),
    ]
    if lora_rank:
        cmd += ["--lora-rank", str(lora_rank), "--lora-alpha", "16"]
        if adapter_out is not None:
            cmd += ["--adapter-out", str(adapter_out)]
    env = os.environ.copy()
    env["HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS"] = "1" if two_pass else "0"
    env["HIERARCHOS_WMMA_MODE"] = "off"
    if profile:
        env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
        env["HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS"] = "32"
    else:
        env.pop("HIERARCHOS_VULKAN_PROFILE_KERNELS", None)
        env.pop("HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS", None)
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open(
        "w", encoding="utf-8"
    ) as se:
        completed = subprocess.run(cmd, cwd=CRATE, env=env, stdout=so, stderr=se,
                                   timeout=3600)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(f"{tag} exited {completed.returncode}; {stderr_path}")
    data = json.loads(report.read_text(encoding="utf-8"))
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    execution_ms = sum(record["step_ms"] for record in data["records"])
    return {
        "tag": tag, "report": data, "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path), "elapsed_ms": elapsed_ms,
        "setup_ms": max(0.0, elapsed_ms - execution_ms),
        "profile": parse_profile(stderr_text) if profile else None,
        "diagnostic": next((line for line in stderr_text.splitlines()
                            if line.startswith("two-pass backward:")), None),
        "adapter_out": str(adapter_out) if adapter_out else None,
    }


def dispatch_counts(profile: dict, steps: int) -> dict:
    totals = profile["totals"]
    layers = 2
    return {
        "scalar": totals.get(SCALAR, {}).get("dispatches", 0),
        "query": totals.get(QUERY, {}).get("dispatches", 0),
        "key": totals.get(KEY, {}).get("dispatches", 0),
        "expected_per_shader": steps * layers,
        "blocks": len(profile["blocks"]),
    }


def kernel_ms_per_step(profile: dict, steps: int) -> dict:
    out = {}
    for name, values in profile["totals"].items():
        out[name] = {
            "dispatches": values["dispatches"],
            "gpu_ms_total": round(values["gpu_ms"], 6),
            "ms_per_step": round(values["gpu_ms"] / steps, 4),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--kernel-steps", type=int, default=12)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "kernel", "abba", "numerics"],
                        default="all")
    args = parser.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"work": str(work), "binary": str(BIN),
                    "binary_sha256": sha256_file(BIN),
                    "model": str(MODEL), "dataset": str(DATASET),
                    "dataset_sha256": sha256_file(DATASET),
                    "config_sha256": sha256_file(MODEL / "config.json")}

    if args.phase in ("all", "kernel"):
        print("== kernel attribution ==", flush=True)
        kernel = {}
        for label, two_pass in (("scalar", False), ("twopass", True)):
            run_result = run(f"kernel-{label}", two_pass, args.kernel_steps, 0,
                             True, 8, None, work / "kernel")
            kernel[label] = {
                "dispatch": dispatch_counts(run_result["profile"], args.kernel_steps),
                "kernels": kernel_ms_per_step(run_result["profile"], args.kernel_steps),
                "diagnostic": run_result["diagnostic"],
                "step_ms_mean": run_result["report"]["measured_step_ms_mean"],
            }
            print(f"  {label}: scalar={kernel[label]['dispatch']['scalar']} "
                  f"query={kernel[label]['dispatch']['query']} "
                  f"key={kernel[label]['dispatch']['key']}", flush=True)
        result["kernel"] = kernel

    if args.phase in ("all", "numerics"):
        print("== numerical parity (2 LoRA steps) ==", flush=True)
        arms = {}
        for label, two_pass in (("scalar", False), ("twopass", True)):
            adapter = work / f"adapter-{label}"
            run_result = run(f"numeric-{label}", two_pass, 2, 0, True, 8, adapter,
                             work / "numeric", digest_every=1)
            files = {}
            if adapter.is_dir():
                for path in sorted(adapter.rglob("*")):
                    if path.is_file():
                        files[str(path.relative_to(adapter)).replace("\\", "/")] = \
                            sha256_file(path)
            arms[label] = {
                "dispatch": dispatch_counts(run_result["profile"], 2),
                "diagnostic": run_result["diagnostic"],
                "loss_bits": [record["loss_bits"] for record in
                              run_result["report"]["records"]],
                "losses": [record["loss"] for record in
                           run_result["report"]["records"]],
                "adapter_files": files,
                "parameter_end_digest": run_result["report"]["parameter_end_digest"],
                "parameter_frozen": run_result["report"]["parameter_frozen"],
            }
            print(f"  {label}: adapter={files.get('adapter_model.safetensors')} "
                  f"loss_bits={arms[label]['loss_bits']}", flush=True)
        result["numerics"] = {
            "scalar": arms["scalar"], "twopass": arms["twopass"],
            "loss_bits_equal": arms["scalar"]["loss_bits"] == arms["twopass"]["loss_bits"],
            "adapters_equal": arms["scalar"]["adapter_files"] == arms["twopass"]["adapter_files"],
            "parameter_end_equal": (arms["scalar"]["parameter_end_digest"]
                                    == arms["twopass"]["parameter_end_digest"]),
        }

    if args.phase in ("all", "abba"):
        print("== paired ABBA whole step ==", flush=True)
        samples = {"scalar": [], "twopass": []}
        rounds = []
        setup = []
        for round_index in range(args.rounds):
            per_arm = {"scalar": [], "twopass": []}
            for position, backward in enumerate(
                ["scalar", "twopass", "twopass", "scalar"]
            ):
                label = "twopass" if backward == "twopass" else "scalar"
                run_result = run(
                    f"abba-r{round_index}-{position}-{label}",
                    label == "twopass", args.steps, args.warmup, False, 8, None,
                    work / "abba",
                )
                measured = [record["step_ms"] for record in
                            run_result["report"]["records"][args.warmup:]]
                per_arm[label].extend(measured)
                samples[label].extend(measured)
                setup.append(run_result["setup_ms"])
                dispatch = None
            med = {arm: statistics.median(values) for arm, values in per_arm.items()}
            ratio = med["scalar"] / med["twopass"]
            rounds.append({"round": round_index, **med, "ratio": ratio})
            print(f"  round {round_index}: scalar={med['scalar']:.2f} "
                  f"twopass={med['twopass']:.2f} ratio={ratio:.4f}",
                  flush=True)
        ratios = [entry["ratio"] for entry in rounds]
        ci_mean, ci_low, ci_high = mean_ci(ratios)
        result["abba"] = {
            "rounds": args.rounds, "steps": args.steps, "warmup": args.warmup,
            "per_round": rounds,
            "scalar_samples": samples["scalar"],
            "twopass_samples": samples["twopass"],
            "scalar_median_ms": statistics.median(samples["scalar"]),
            "twopass_median_ms": statistics.median(samples["twopass"]),
            "scalar_mean_ms": statistics.mean(samples["scalar"]),
            "twopass_mean_ms": statistics.mean(samples["twopass"]),
            "scalar_std_ms": statistics.stdev(samples["scalar"]),
            "twopass_std_ms": statistics.stdev(samples["twopass"]),
            "scalar_p95_ms": sorted(samples["scalar"])[
                int(0.95 * (len(samples["scalar"]) - 1))],
            "twopass_p95_ms": sorted(samples["twopass"])[
                int(0.95 * (len(samples["twopass"]) - 1))],
            "ratio_mean": ci_mean,
            "ratio_ci": {"mean": ci_mean, "low": ci_low, "high": ci_high,
                         "df": len(ratios) - 1, "n": len(ratios)},
            "ratio_geomean": statistics.geometric_mean(ratios),
            "t_critical": t_critical(len(ratios) - 1),
            "setup_ms": setup,
        }
        print(f"  ratio CI95 {ci_low:.4f}..{ci_high:.4f} "
              f"(mean {ci_mean:.4f}, n={len(ratios)})", flush=True)

    (work / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {work / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
