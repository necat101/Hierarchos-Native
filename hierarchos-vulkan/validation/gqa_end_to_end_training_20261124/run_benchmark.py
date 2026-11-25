#!/usr/bin/env python3
"""Phase VII: end-to-end GQA Transformer training benchmark.

Runs the native trainer harness (``hierarchos-vulkan-training-qualification``)
as separate processes per arm, using the Phase III protocol: 3 warmup steps,
26 measured steps, 5 paired ABBA rounds (scalar, two-pass, two-pass, scalar).

Everything measured here is derived from a run's own report or its profiler
stderr.  Nothing is synthesised: a missing sample or a failed process raises.

Phases
------
numerics   two-step LoRA arms with per-step digests, exported adapter and the
           exact training checkpoint (adapter + AdamW moments + RNG) compared
           bit-for-bit between the scalar and accelerated arms.
profile    twelve profiled steps per arm; per-shader GPU milliseconds are
           aggregated into the attribution table.
abba       paired ABBA whole-step timing on the primary and secondary geometry.
default    the environment variable unset, to show the production selector
           chooses the GQA two-pass route without being told to.
failclosed explicit ``=1`` with the Vulkan loader pointed at no ICD.
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
sys.path.insert(0, str(CRATE / "validation"))
from abba_stats import mean_ci, t_critical  # noqa: E402

BIN = CRATE / "target/release/hierarchos-vulkan-training-qualification.exe"
WORK = CRATE / "target/gqa-e2e"
RECEIPT = CRATE / "validation/gqa_end_to_end_training_20261124"

DEVICE_RE = re.compile(
    r"hierarchos_vulkan_kernel_profile device=.* dispatches=(\d+) gpu_ms=([0-9.]+)"
)
SHADER_RE = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=(\S+) "
    r"dispatches=(\d+) gpu_ms=([0-9.]+)"
)
ROUTE_RE = re.compile(r"^two-pass backward: route=(\S+) ")

ATTENTION_QUERY = "transformer_attention_backward_query_rdna3"
ATTENTION_KEY = "transformer_attention_backward_key_rdna3"
ATTENTION_SCALAR = "transformer_attention_backward"

# Attribution buckets, in the order the final table reports them.  Each shader
# name is mapped explicitly: a new shader falls into `other` and is visible
# there rather than being silently folded into a category it does not belong to.
BUCKETS = {
    "attention_backward_query": {ATTENTION_QUERY},
    "attention_backward_key": {ATTENTION_KEY},
    "attention_backward_scalar": {ATTENTION_SCALAR},
    "attention_other": {
        "transformer_attention_forward",
        "transformer_rope_forward",
        "transformer_rope_backward",
        "transformer_elementwise_mul_forward",
        "transformer_elementwise_mul_backward",
    },
    "linear_forward": {"linear_forward", "linear_forward_lane4"},
    "linear_weight_grad": {"linear_weight_grad"},
    "linear_input_grad": {"linear_input_grad", "linear_input_grad_group4_muladd"},
    "normalization": {
        "transformer_rms_norm_forward",
        "transformer_rms_norm_input_grad",
    },
    "mlp": {"silu_forward", "silu_backward"},
    "optimizer": {"adamw", "adamw_fp16_mirror", "gradient_accumulate",
                  "gradient_accumulate4", "gradient_accumulate8",
                  "gradient_scale"},
    "other": {
        "falcon_h1_cross_entropy_lanes16",
        "embedding_forward",
        "ordered_f32_sum",
        "transformer_lora_scaled_add",
        "transformer_scale_copy",
        "vector_add",
        "bias_add",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return out


def parse_profile(text: str) -> dict:
    """Per-shader totals over every non-empty profiler submission block."""
    totals: dict[str, dict] = {}
    blocks = 0
    for line in text.splitlines():
        device = DEVICE_RE.search(line)
        if device:
            if int(device.group(1)) > 0:
                blocks += 1
            continue
        shader = SHADER_RE.search(line)
        if shader:
            entry = totals.setdefault(
                shader.group(1), {"category": shader.group(2),
                                  "dispatches": 0, "gpu_ms": 0.0}
            )
            entry["dispatches"] += int(shader.group(3))
            entry["gpu_ms"] += float(shader.group(4))
    return {"blocks": blocks, "totals": totals}


def route_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("two-pass backward:")]


def run(tag: str, geometry: str, two_pass: str | None, steps: int, warmup: int,
        profile: bool, lora_rank: int | None, adapter_out: Path | None,
        save_state: Path | None, digest_every: int, extra_env: dict | None,
        out_dir: Path) -> dict:
    """One trainer process.  `two_pass` is the literal environment value or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{tag}.report.json"
    stdout_path = out_dir / f"{tag}.stdout.txt"
    stderr_path = out_dir / f"{tag}.stderr.txt"
    scratch = out_dir / f"{tag}.scratch"
    fixture = WORK / f"fixture-{geometry}"
    cmd = [
        str(BIN), "--model", str(fixture / "model"),
        "--dataset", str(fixture / "dataset.jsonl"),
        "--output", str(scratch), "--batch", "2", "--seq", "64",
        "--steps", str(steps), "--warmup", str(warmup),
        "--lr", "1e-2", "--seed", "7", "--log-every", str(max(steps, 1)),
        "--digest-every", str(digest_every), "--report", str(report),
    ]
    if lora_rank:
        cmd += ["--lora-rank", str(lora_rank), "--lora-alpha", "16"]
    if adapter_out is not None:
        cmd += ["--adapter-out", str(adapter_out)]
    if save_state is not None:
        cmd += ["--save-state", str(save_state)]
    if adapter_out is None and save_state is None:
        cmd.append("--no-export")

    env = os.environ.copy()
    for key in ("HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS",
                "HIERARCHOS_VULKAN_PROFILE_KERNELS",
                "HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS",
                "HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE",
                "VK_ICD_FILENAMES", "VK_DRIVER_FILES"):
        env.pop(key, None)
    env["HIERARCHOS_WMMA_MODE"] = "off"
    if two_pass is not None:
        env["HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS"] = two_pass
    if profile:
        env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
        env["HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS"] = "4096"
        env["HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE"] = "1"
    if extra_env:
        env.update(extra_env)

    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open(
        "w", encoding="utf-8"
    ) as se:
        completed = subprocess.run(cmd, cwd=CRATE, env=env, stdout=so, stderr=se,
                                   timeout=7200)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    result = {
        "tag": tag, "geometry": geometry, "two_pass_env": two_pass,
        "steps": steps, "warmup": warmup, "profile": profile,
        "exit_code": completed.returncode, "elapsed_ms": elapsed_ms,
        "report_path": str(report), "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "routes": route_lines(stderr_text),
    }
    if profile:
        result["kernel_profile"] = parse_profile(stderr_text)
    if completed.returncode != 0:
        result["failure_tail"] = "\n".join(stderr_text.splitlines()[-12:])
        return result
    data = json.loads(report.read_text(encoding="utf-8"))
    result["report"] = data
    result["step_ms"] = [record["step_ms"] for record in data["records"]]
    result["measured_step_ms"] = result["step_ms"][warmup:]
    return result


def measure_pair(geometry: str, rounds: int, steps: int, warmup: int,
                 samples_dir: Path, raw_dir: Path) -> dict:
    """Phase III ABBA order: scalar, two-pass, two-pass, scalar per round.

    ``steps`` is the total optimizer steps per process; ``warmup`` of them are
    discarded, so each run contributes ``steps - warmup`` measured samples.
    """
    arm_samples = {"scalar": [], "twopass": []}
    per_round = []
    samples_path = samples_dir / f"abba-{geometry}.jsonl"
    with samples_path.open("w", encoding="utf-8") as samples:
        for round_index in range(rounds):
            per_arm = {"scalar": [], "twopass": []}
            for position, arm in enumerate(
                ["scalar", "twopass", "twopass", "scalar"]
            ):
                two_pass = "0" if arm == "scalar" else "1"
                result = run(
                    f"abba-{geometry}-r{round_index}-p{position}-{arm}",
                    geometry, two_pass, steps, warmup, False, 8, None, None, 0,
                    None, raw_dir / f"abba-{geometry}",
                )
                if result["exit_code"] != 0:
                    raise RuntimeError(
                        f"{result['tag']} failed rc={result['exit_code']}: "
                        f"{result.get('failure_tail', '')}"
                    )
                measured = result["measured_step_ms"]
                if len(measured) != steps - warmup:
                    raise RuntimeError(
                        f"{result['tag']}: expected {steps - warmup} measured "
                        f"samples, got {len(measured)}"
                    )
                per_arm[arm].extend(measured)
                arm_samples[arm].extend(measured)
                samples.write(json.dumps({
                    "geometry": geometry, "round": round_index,
                    "position": position, "arm": arm,
                    "steps": steps, "warmup": warmup,
                    "step_ms": result["step_ms"],
                    "measured_step_ms": measured,
                    "route": result["routes"],
                }) + "\n")
                samples.flush()
            medians = {arm: statistics.median(values)
                       for arm, values in per_arm.items()}
            ratio = medians["scalar"] / medians["twopass"]
            per_round.append({
                "round": round_index,
                "scalar_median_ms": medians["scalar"],
                "twopass_median_ms": medians["twopass"],
                "ratio": ratio,
            })
            print(f"  {geometry} round {round_index}: "
                  f"scalar={medians['scalar']:.2f} "
                  f"twopass={medians['twopass']:.2f} ratio={ratio:.4f}",
                  flush=True)
    ratios = [entry["ratio"] for entry in per_round]
    mean, low, high = mean_ci(ratios)
    scalar = arm_samples["scalar"]
    twopass = arm_samples["twopass"]

    def p95(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[int(0.95 * (len(ordered) - 1))]

    return {
        "geometry": geometry,
        "protocol": {
            "rounds": rounds, "steps_per_run": steps, "warmup_per_run": warmup,
            "measured_per_run": steps - warmup,
            "order": ["scalar", "twopass", "twopass", "scalar"],
            "processes": "one per run, strictly serial",
            "wmma_mode": "off",
            "samples_file": str(samples_path),
        },
        "per_round": per_round,
        "paired_ratios": ratios,
        "mean_paired_ratio": mean,
        "ci95": {"low": low, "high": high, "df": len(ratios) - 1,
                 "t_critical": t_critical(len(ratios) - 1)},
        "geomean_ratio": statistics.geometric_mean(ratios),
        "ratio_of_medians": statistics.median(scalar) / statistics.median(twopass),
        "scalar": {
            "samples": len(scalar),
            "median_ms": statistics.median(scalar),
            "mean_ms": statistics.mean(scalar),
            "std_ms": statistics.stdev(scalar),
            "p95_ms": p95(scalar),
            "min_ms": min(scalar), "max_ms": max(scalar),
        },
        "twopass": {
            "samples": len(twopass),
            "median_ms": statistics.median(twopass),
            "mean_ms": statistics.mean(twopass),
            "std_ms": statistics.stdev(twopass),
            "p95_ms": p95(twopass),
            "min_ms": min(twopass), "max_ms": max(twopass),
        },
    }


def attribute(profile: dict, steps: int) -> dict:
    totals = profile["totals"]
    assigned: set[str] = set()
    buckets: dict[str, dict] = {}
    for bucket, names in BUCKETS.items():
        dispatches = 0
        gpu_ms = 0.0
        matched = []
        for name in sorted(names):
            entry = totals.get(name)
            if not entry:
                continue
            assigned.add(name)
            dispatches += entry["dispatches"]
            gpu_ms += entry["gpu_ms"]
            matched.append({"shader": name, **entry})
        buckets[bucket] = {
            "dispatches_per_step": dispatches / steps,
            "gpu_ms_per_step": gpu_ms / steps,
            "shaders": matched,
        }
    unassigned = {
        name: entry for name, entry in totals.items() if name not in assigned
    }
    buckets["other"]["shaders"].extend(
        {"shader": name, **entry} for name, entry in sorted(unassigned.items())
    )
    buckets["other"]["dispatches_per_step"] += sum(
        entry["dispatches"] for entry in unassigned.values()
    ) / steps
    buckets["other"]["gpu_ms_per_step"] += sum(
        entry["gpu_ms"] for entry in unassigned.values()
    ) / steps
    total_ms = sum(entry["gpu_ms"] for entry in totals.values()) / steps
    for bucket in buckets.values():
        bucket["share_pct"] = (
            100.0 * bucket["gpu_ms_per_step"] / total_ms if total_ms else 0.0
        )
    return {
        "steps": steps,
        "blocks": profile["blocks"],
        "gpu_ms_per_step_total": total_ms,
        "buckets": buckets,
        "unassigned_shaders": sorted(unassigned),
    }


def main() -> int:
    global WORK, RECEIPT
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "numerics", "profile", "abba", "default",
                                 "failclosed", "primary", "secondary"])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()

    WORK = args.work
    RECEIPT = args.receipt
    raw = RECEIPT / "raw"
    samples = RECEIPT / "samples"
    raw.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)

    summary_path = RECEIPT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if (
        summary_path.exists()) else {}
    summary.setdefault("binary", str(BIN))
    summary.setdefault("binary_sha256", sha256_file(BIN))
    summary.setdefault("geometries", {})
    summary.setdefault("environment", {
        "device": "AMD Radeon Graphics",
        "driver": "2.0.416 / 26.20.15.14 (LLPC)",
        "rustc": "1.93.0",
        "wmma_mode": "off",
    })

    do_numerics = args.phase in ("all", "numerics", "primary", "secondary")
    do_profile = args.phase in ("all", "profile", "primary", "secondary")
    do_abba = args.phase in ("all", "abba", "primary", "secondary")
    geometries = []
    if args.phase in ("all", "numerics", "profile", "abba", "primary"):
        geometries.append(("8q2kv", 5))
    if args.phase in ("all", "numerics", "profile", "abba", "secondary"):
        geometries.append(("4q2kv", 3))

    for geometry, rounds in geometries:
        entry = summary["geometries"].setdefault(geometry, {})
        fixture = WORK / f"fixture-{geometry}"
        entry["fixture"] = {
            "path": str(fixture),
            "config_sha256": sha256_file(fixture / "model" / "config.json"),
            "dataset_sha256": sha256_file(fixture / "dataset.jsonl"),
            "weights_sha256": sha256_file(fixture / "model" / "model.safetensors"),
            "config": json.loads((fixture / "model" / "config.json").read_text()),
        }

        if do_numerics:
            print(f"== {geometry} numerics ==", flush=True)
            arms = {}
            for arm, two_pass in (("scalar", "0"), ("twopass", "1")):
                adapter = raw / f"numerics-{geometry}-{arm}-adapter"
                state = raw / f"numerics-{geometry}-{arm}-state"
                result = run(
                    f"numerics-{geometry}-{arm}", geometry, two_pass, 2, 0,
                    True, 8, adapter, state, 1, None,
                    raw / f"numerics-{geometry}",
                )
                if result["exit_code"] != 0:
                    raise RuntimeError(
                        f"{result['tag']} failed: {result.get('failure_tail')}"
                    )
                report = result["report"]
                arms[arm] = {
                    "routes": result["routes"],
                    "losses": [r["loss"] for r in report["records"]],
                    "loss_bits": [r["loss_bits"] for r in report["records"]],
                    "grad_digests": [r["grad_digest"] for r in report["records"]],
                    "grad_norms": [r["grad_norm"] for r in report["records"]],
                    "param_digests": [r["param_digest"] for r in report["records"]],
                    "parameter_end_digest": report["parameter_end_digest"],
                    "parameter_frozen": report["parameter_frozen"],
                    "adapter_files": tree_hashes(adapter),
                    "optimizer_state_files": tree_hashes(state),
                    "kernel_profile": result.get("kernel_profile"),
                }
            same = {
                "loss_bits": arms["scalar"]["loss_bits"] == arms["twopass"]["loss_bits"],
                "grad_digests": arms["scalar"]["grad_digests"] == arms["twopass"]["grad_digests"],
                "param_digests": arms["scalar"]["param_digests"] == arms["twopass"]["param_digests"],
                "parameter_end_digest": (arms["scalar"]["parameter_end_digest"]
                                         == arms["twopass"]["parameter_end_digest"]),
                "adapter_files": arms["scalar"]["adapter_files"] == arms["twopass"]["adapter_files"],
                "optimizer_state_files": (arms["scalar"]["optimizer_state_files"]
                                          == arms["twopass"]["optimizer_state_files"]),
            }
            entry["numerics"] = {"scalar": arms["scalar"], "twopass": arms["twopass"],
                                 "bit_identical": same,
                                 "all_identical": all(same.values())}
            print(f"  identical: {same}", flush=True)

        if do_profile:
            print(f"== {geometry} profile (12 steps) ==", flush=True)
            prof = {}
            for arm, two_pass in (("scalar", "0"), ("twopass", "1")):
                result = run(
                    f"profile-{geometry}-{arm}", geometry, two_pass, 12, 0,
                    True, 8, None, None, 0, None, raw / f"profile-{geometry}",
                )
                if result["exit_code"] != 0:
                    raise RuntimeError(
                        f"{result['tag']} failed: {result.get('failure_tail')}"
                    )
                prof[arm] = {
                    "routes": result["routes"],
                    "wall_ms_per_step": statistics.mean(result["step_ms"]),
                    "attribution": attribute(result["kernel_profile"], 12),
                }
            entry["profile"] = prof
            print(f"  scalar gpu/step="
                  f"{prof['scalar']['attribution']['gpu_ms_per_step_total']:.3f} ms, "
                  f"twopass gpu/step="
                  f"{prof['twopass']['attribution']['gpu_ms_per_step_total']:.3f} ms",
                  flush=True)

        if do_abba:
            print(f"== {geometry} ABBA ({rounds} rounds, 3 warmup + 26 measured) ==", flush=True)
            entry["abba"] = measure_pair(geometry, rounds, 29, 3, samples, raw)
            print(f"  mean={entry['abba']['mean_paired_ratio']:.4f} "
                  f"CI95=[{entry['abba']['ci95']['low']:.4f}, "
                  f"{entry['abba']['ci95']['high']:.4f}]", flush=True)

        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if args.phase in ("all", "default"):
        print("== default-environment sanity ==", flush=True)
        result = run("default-8q2kv", "8q2kv", None, 3, 0, True, 8, None, None,
                     0, None, raw / "default")
        if result["exit_code"] != 0:
            raise RuntimeError(f"default run failed: {result.get('failure_tail')}")
        totals = result["kernel_profile"]["totals"]
        summary["default_environment"] = {
            "two_pass_env": None,
            "routes": result["routes"],
            "query_dispatches": totals.get(ATTENTION_QUERY, {}).get("dispatches", 0),
            "key_dispatches": totals.get(ATTENTION_KEY, {}).get("dispatches", 0),
            "scalar_dispatches": totals.get(ATTENTION_SCALAR, {}).get("dispatches", 0),
            "steps": 3, "layers": 2,
            "report_two_pass_requested": result["report"]["two_pass_requested"],
        }
        print(f"  {summary['default_environment']}", flush=True)

    if args.phase in ("all", "failclosed"):
        print("== fail-closed with no Vulkan ICD ==", flush=True)
        result = run("failclosed-8q2kv", "8q2kv", "1", 1, 0, False, 8, None,
                     None, 0,
                     {"VK_ICD_FILENAMES": str(WORK / "no-such-icd.json"),
                      "VK_DRIVER_FILES": str(WORK / "no-such-icd.json")},
                     raw / "failclosed")
        summary["fail_closed"] = {
            "exit_code": result["exit_code"],
            "nonzero": result["exit_code"] != 0,
            "routes": result["routes"],
            "failure_tail": result.get("failure_tail"),
        }
        print(f"  exit={result['exit_code']} tail={result.get('failure_tail')}",
              flush=True)

    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
