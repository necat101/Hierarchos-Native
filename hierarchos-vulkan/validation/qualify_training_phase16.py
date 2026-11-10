#!/usr/bin/env python3
"""Phase XVI training qualification producer.

Runs the native RDNA3 Transformer trainer through the four precision/backward
arms in both full-fine-tuning and LoRA modes, verifies exact LoRA resume, exercises
adverse geometries, measures paired whole-step latency, and records the 48-step
WMMA reversal. Every derived number is recomputed by the checker from the shipped
per-step reports and profile logs; nothing is trusted from a summary field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
REPO = CRATE.parent
BIN = CRATE / "target/release/hierarchos-vulkan-training-qualification.exe"
TRAINER = CRATE / "target/release/hierarchos-vulkan-transformer-train.exe"
MODEL = CRATE / "target/wmma-fixture/model"
DATASET = CRATE / "target/wmma-fixture/dataset.jsonl"
RECEIPT_DIR = Path(os.environ.get(
    "HIERARCHOS_RECEIPT_DIR",
    CRATE / "validation/release_parity/multimodal_training" / "20261110-phase16-training-qualification",
))
PHASE16_RECEIPT = (
    CRATE
    / "validation/release_parity/multimodal_training"
    / "20261109-rdna3-phase16-attention-backward-qualification"
    / "current-qualification.json"
)
SCHEMA = 1
EXPECTED_PARAMETERS = 16_782_336

ARMS = {
    "A": dict(precision="fp32", wmma="off", backward="scalar", two_pass="0", mode="full"),
    "B": dict(precision="fp32", wmma="off", backward="twopass", two_pass="1", mode="full"),
    "C": dict(precision="mixed", wmma="mixed", backward="scalar", two_pass="0", mode="full"),
    "D": dict(precision="mixed", wmma="mixed", backward="twopass", two_pass="1", mode="full"),
    "E": dict(precision="fp32", wmma="off", backward="scalar", two_pass="0", mode="lora"),
    "F": dict(precision="fp32", wmma="off", backward="twopass", two_pass="1", mode="lora"),
    "G": dict(precision="mixed", wmma="mixed", backward="scalar", two_pass="0", mode="lora"),
    "H": dict(precision="mixed", wmma="mixed", backward="twopass", two_pass="1", mode="lora"),
}

SCALAR_SHADER = "transformer_attention_backward"
QUERY_SHADER = "transformer_attention_backward_query_rdna3"
KEY_SHADER = "transformer_attention_backward_key_rdna3"

DEVICE_RE = re.compile(
    r"hierarchos_vulkan_kernel_profile device=.* dispatches=(\d+) gpu_ms=([0-9.]+)"
)
SHADER_RE = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=\S+ "
    r"dispatches=(\d+) gpu_ms=([0-9.]+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def parse_profile(text: str) -> list[dict]:
    """One entry per submission that actually dispatched work.

    A training step is one submission, so the nonzero blocks are the steps. The
    caller asserts the count equals the requested step count before using them.
    """
    blocks: list[dict] = []
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
    return blocks


def profile_totals(blocks: list[dict]) -> dict:
    totals: dict[str, dict] = {}
    for block in blocks:
        for name, values in block["shaders"].items():
            entry = totals.setdefault(name, {"dispatches": 0, "gpu_ms": 0.0})
            entry["dispatches"] += values["dispatches"]
            entry["gpu_ms"] += values["gpu_ms"]
    return totals


def arm_env(arm: dict, profile: bool) -> dict:
    env = os.environ.copy()
    env["HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS"] = arm["two_pass"]
    env["HIERARCHOS_WMMA_MODE"] = arm["wmma"]
    if profile:
        env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
        env["HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS"] = "32"
    else:
        env.pop("HIERARCHOS_VULKAN_PROFILE_KERNELS", None)
        env.pop("HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS", None)
    return env


def run_arm(
    work: Path,
    tag: str,
    arm: dict,
    *,
    steps: int,
    digest_every: int,
    profile: bool,
    seed: int = 7,
    seq: int = 64,
    warmup: int = 0,
    start_step: int = 0,
    resume_state: Path | None = None,
    save_state: Path | None = None,
    timeout: int = 1800,
) -> dict:
    reports = work / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"{tag}.json"
    out = work / "scratch" / tag
    adapter = work / "adapters" / tag
    if adapter.exists():
        shutil.rmtree(adapter)
    adapter.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = work / "logs" / f"{tag}.out.txt"
    stderr_path = work / "logs" / f"{tag}.err.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN),
        "--model", str(MODEL),
        "--dataset", str(DATASET),
        "--output", str(out),
        "--no-export",
        "--steps", str(steps),
        "--start-step", str(start_step),
        "--warmup", str(warmup),
        "--digest-every", str(digest_every),
        "--seed", str(seed),
        "--seq", str(seq),
        "--report", str(report_path),
    ]
    if arm["mode"] == "lora" or resume_state is not None:
        cmd += ["--lora-rank", "8", "--lora-alpha", "16"]
        cmd += ["--adapter-out", str(adapter)]
    if resume_state is not None:
        cmd += ["--resume-state", str(resume_state)]
    if save_state is not None:
        if save_state.exists():
            shutil.rmtree(save_state)
        cmd += ["--save-state", str(save_state)]
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            cmd,
            cwd=CRATE,
            env=arm_env(arm, profile),
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
        )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"{tag}: harness exited {completed.returncode}; see {stderr_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    blocks = parse_profile(stderr_text) if profile else []
    execution_ms = sum(record["step_ms"] for record in report["records"])
    return {
        "tag": tag,
        "arm": arm,
        "report": report,
        "report_path": report_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "adapter_dir": adapter if adapter.exists() else None,
        "save_state": save_state,
        "blocks": blocks,
        "process_ms": elapsed_ms,
        "setup_ms": max(0.0, elapsed_ms - execution_ms),
        "expected_steps": steps,
        "start_step": start_step,
    }


def check_run(run: dict) -> dict:
    report = run["report"]
    arm = run["arm"]
    steps = run["expected_steps"]
    if len(report["records"]) != steps:
        raise AssertionError(f"{run['tag']}: {len(report['records'])} records != {steps}")
    start_step = run["start_step"]
    expected = list(range(start_step + 1, start_step + steps + 1))
    actual = [record["optimizer_step"] for record in report["records"]]
    if actual != expected:
        raise AssertionError(f"{run['tag']}: optimizer steps {actual} != {expected}")
    for record in report["records"]:
        if not record["loss_finite"]:
            raise AssertionError(f"{run['tag']}: non-finite loss at {record['global_step']}")
        if record.get("grad_nonfinite") not in (None, 0):
            raise AssertionError(f"{run['tag']}: non-finite gradients at {record['global_step']}")
        if record.get("param_nonfinite") not in (None, 0):
            raise AssertionError(f"{run['tag']}: non-finite parameters at {record['global_step']}")
    if report["two_pass_requested"] != (arm["two_pass"] == "1"):
        raise AssertionError(f"{run['tag']}: two-pass mode does not match the arm label")
    if report["precision_requested"] != ("mixed" if arm["precision"] == "mixed" else "off"):
        raise AssertionError(f"{run['tag']}: precision does not match the arm label")
    if report["lora"] != (arm["mode"] == "lora"):
        raise AssertionError(f"{run['tag']}: LoRA label does not match the run")
    if report["mode"] == "full-finetune":
        if report["parameter_elements"] != EXPECTED_PARAMETERS:
            raise AssertionError(
                f"{run['tag']}: {report['parameter_elements']} parameters, expected "
                f"{EXPECTED_PARAMETERS}"
            )
        if not report["parameter_end_digest"]:
            raise AssertionError(f"{run['tag']}: missing final parameter digest")
    else:
        if report["parameter_frozen"] is not True:
            raise AssertionError(f"{run['tag']}: LoRA base parameters moved")
        if not run["adapter_dir"]:
            raise AssertionError(f"{run['tag']}: LoRA run produced no adapter")
        adapter = run["adapter_dir"] / "adapter_model.safetensors"
        if not adapter.is_file():
            raise AssertionError(f"{run['tag']}: missing {adapter}")

    dispatch = None
    if run["blocks"]:
        if len(run["blocks"]) != steps:
            raise AssertionError(
                f"{run['tag']}: {len(run['blocks'])} profiled submissions for {steps} steps"
            )
        totals = profile_totals(run["blocks"])
        scalar = totals.get(SCALAR_SHADER, {}).get("dispatches", 0)
        query = totals.get(QUERY_SHADER, {}).get("dispatches", 0)
        key = totals.get(KEY_SHADER, {}).get("dispatches", 0)
        layers = 2
        if arm["backward"] == "scalar":
            if scalar != steps * layers or query or key:
                raise AssertionError(
                    f"{run['tag']}: scalar dispatch proof failed "
                    f"scalar={scalar} query={query} key={key}"
                )
        else:
            if scalar or query != steps * layers or key != steps * layers:
                raise AssertionError(
                    f"{run['tag']}: two-pass dispatch proof failed "
                    f"scalar={scalar} query={query} key={key}"
                )
        dispatch = {
            "scalar": scalar,
            "query": query,
            "key": key,
            "gpu_ms": {
                name: round(values["gpu_ms"], 6) for name, values in totals.items()
            },
        }

    wmma = {
        "forward": report["wmma_forward"],
        "weight_grad": report["wmma_weight_grad"],
        "input_grad": report["wmma_input_grad"],
        "cooperative": report["wmma_cooperative"],
        "shape_fallbacks": report["wmma_shape_fallbacks"],
        "weight_conversions": report["wmma_weight_conversions"],
        "mirror_reuses": report["wmma_mirror_reuses"],
    }
    if arm["precision"] == "fp32":
        if any(wmma[key] for key in ("forward", "weight_grad", "input_grad", "cooperative")):
            raise AssertionError(f"{run['tag']}: FP32 arm dispatched WMMA work")
    else:
        if not wmma["forward"] or not wmma["input_grad"] or not wmma["cooperative"]:
            raise AssertionError(f"{run['tag']}: mixed arm did not dispatch WMMA GEMMs")
        if arm["mode"] == "full" and wmma["weight_grad"] <= 0:
            raise AssertionError(f"{run['tag']}: full fine-tune missed weight gradients")
        if arm["mode"] == "lora" and wmma["weight_grad"] != 0:
            raise AssertionError(f"{run['tag']}: LoRA run computed base weight gradients")
    return {"dispatch": dispatch, "wmma": wmma}


def record_field(records: list[dict], field: str) -> list:
    return [record[field] for record in records]


def compare(a: dict, b: dict, fields: list[str]) -> dict:
    ra, rb = a["report"]["records"], b["report"]["records"]
    if len(ra) != len(rb):
        raise AssertionError(f"{a['tag']} and {b['tag']} have different step counts")
    result = {"left": a["tag"], "right": b["tag"], "fields": {}, "bitwise_equal": True}
    for field in fields:
        left, right = record_field(ra, field), record_field(rb, field)
        equal = left == right
        result["fields"][field] = {"equal": equal}
        if not equal:
            result["bitwise_equal"] = False
            if all(isinstance(value, (int, float)) for value in left + right):
                diffs = [float(x) - float(y) for x, y in zip(left, right)]
                result["fields"][field]["max_abs_diff"] = max(abs(value) for value in diffs)
            else:
                first = next(i for i, (x, y) in enumerate(zip(left, right)) if x != y)
                result["fields"][field]["first_difference"] = {
                    "index": first,
                    "left": left[first],
                    "right": right[first],
                }
    if a["report"]["mode"] == "full-finetune":
        result["final_parameter_digest_equal"] = (
            a["report"]["parameter_end_digest"] == b["report"]["parameter_end_digest"]
        )
    else:
        left_adapter = a["adapter_dir"] / "adapter_model.safetensors"
        right_adapter = b["adapter_dir"] / "adapter_model.safetensors"
        result["adapter_sha256"] = {
            "left": sha256_file(left_adapter),
            "right": sha256_file(right_adapter),
            "equal": sha256_file(left_adapter) == sha256_file(right_adapter),
        }
    return result


def loss_drift(left: dict, right: dict) -> dict:
    a = [record["loss"] for record in left["report"]["records"]]
    b = [record["loss"] for record in right["report"]["records"]]
    diffs = [x - y for x, y in zip(a, b)]
    first_abs = abs(diffs[0])
    trajectory_max = max(abs(value) for value in diffs)
    trajectory_rms = math.sqrt(sum(value * value for value in diffs) / len(diffs))
    return {
        "left": left["tag"],
        "right": right["tag"],
        # The mixed-precision policy is a matched-initial-step bound: after the
        # first AdamW update the two trajectories sit on different loss surfaces
        # and a trajectory-wide bound would test chaos, not precision. The
        # trajectory drift is reported so compounding is visible.
        "first_step_abs": first_abs,
        "first_step_within_policy": first_abs <= 2.0e-2,
        "policy_max_abs": 2.0e-2,
        "policy_scope": "matched initial step (wmma::MIXED_VS_FP32_MAX_ABS)",
        "max_abs": trajectory_max,
        "rms": trajectory_rms,
        "final_abs": abs(diffs[-1]),
        "steps": len(diffs),
    }


def summarize_run(run: dict) -> dict:
    report = run["report"]
    return {
        "tag": run["tag"],
        "mode": report["mode"],
        "precision": run["arm"]["precision"],
        "backward": run["arm"]["backward"],
        "steps": report["steps"],
        "start_step": report["start_step"],
        "optimizer_steps": report["optimizer_steps"],
        "first_loss": report["first_loss"],
        "final_loss": report["final_loss"],
        "loss_min": report["loss_min"],
        "loss_max": report["loss_max"],
        "all_losses_finite": report["all_losses_finite"],
        "grad_norm_min": report["grad_norm_min"],
        "grad_norm_max": report["grad_norm_max"],
        "parameter_start_digest": report["parameter_start_digest"],
        "parameter_end_digest": report["parameter_end_digest"],
        "parameter_elements": report["parameter_elements"],
        "parameter_frozen": report["parameter_frozen"],
        "measured_step_ms_mean": report["measured_step_ms_mean"],
        "process_ms": round(run["process_ms"], 3),
        "setup_ms": round(run["setup_ms"], 3),
        "adapter_sha256": (
            sha256_file(run["adapter_dir"] / "adapter_model.safetensors")
            if run["adapter_dir"]
            else None
        ),
        "profile_blocks": len(run["blocks"]),
    }


def files_under(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return result


def command_version(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        ).stdout.strip().splitlines()[0]
    except Exception as error:  # pragma: no cover - environment dependent
        return f"unavailable: {error}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--quick", action="store_true", help="short smoke matrix")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    args = parser.parse_args()

    if not BIN.is_file():
        print(f"missing qualification binary {BIN}; build it first", file=sys.stderr)
        return 2
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        print(f"worktree has tracked changes:\n{dirty}", file=sys.stderr)
        return 2
    revision = git("rev-parse", "HEAD")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    work = args.work or (CRATE / "target/train-phase16" / stamp)
    work.mkdir(parents=True, exist_ok=True)
    steps = 3 if args.quick else 20
    perf_steps = 8 if args.quick else 20
    perf_warmup = 1 if args.quick else 3
    perf_rounds = 1 if args.quick else 2
    reversal_steps = 8 if args.quick else 48
    reversal_repeats = 1 if args.quick else 2

    print(f"work={work}")
    print(f"revision={revision}")

    arms: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for arm_id, arm in ARMS.items():
        print(f"arm {arm_id} {arm['precision']} {arm['backward']} {arm['mode']}")
        run = run_arm(
            work,
            f"arm-{arm_id}",
            arm,
            steps=steps,
            digest_every=1,
            profile=True,
        )
        checks = check_run(run)
        run["checks"] = checks
        runs[arm_id] = run
        arms[arm_id] = {
            "config": arm,
            "summary": summarize_run(run),
            "dispatch": checks["dispatch"],
            "wmma": checks["wmma"],
            "report_sha256": sha256_file(run["report_path"]),
            "stderr_sha256": sha256_file(run["stderr_path"]),
        }

    fields_full = ["loss_bits", "param_digest", "update_norm", "update_max_abs"]
    fields_lora = ["loss_bits", "grad_digest", "grad_norm"]
    comparisons = {
        "A_vs_B": compare(runs["A"], runs["B"], fields_full),
        "C_vs_D": compare(runs["C"], runs["D"], fields_full),
        "E_vs_F": compare(runs["E"], runs["F"], fields_lora),
        "G_vs_H": compare(runs["G"], runs["H"], fields_lora),
        "B_vs_D": loss_drift(runs["B"], runs["D"]),
        "F_vs_H": loss_drift(runs["F"], runs["H"]),
        "A_vs_C": loss_drift(runs["A"], runs["C"]),
        "E_vs_G": loss_drift(runs["E"], runs["G"]),
    }
    for key in ("A_vs_B", "C_vs_D"):
        if not comparisons[key]["bitwise_equal"]:
            raise AssertionError(f"{key} is not bit-identical")
    for key in ("E_vs_F", "G_vs_H"):
        if not comparisons[key]["bitwise_equal"]:
            raise AssertionError(f"{key} is not bit-identical")
        if not comparisons[key]["adapter_sha256"]["equal"]:
            raise AssertionError(f"{key} adapter bytes differ")
    for key in ("B_vs_D", "F_vs_H"):
        if not comparisons[key]["first_step_within_policy"]:
            raise AssertionError(f"{key} exceeds the matched-step mixed policy")

    # Exact LoRA resume: uninterrupted vs 8-step checkpoint reloaded for 4.
    resume = {}
    resume_state_dirs = {}
    for arm_id in ("E", "F", "H"):
        arm = ARMS[arm_id]
        base = f"resume-{arm_id}"
        total = 4 if args.quick else 12
        split = 2 if args.quick else 8
        uninterrupted = run_arm(
            work, f"{base}-u", arm, steps=total, digest_every=2, profile=False,
            save_state=work / "states" / f"{base}-u",
        )
        first = run_arm(
            work, f"{base}-i", arm, steps=split, digest_every=2, profile=False,
            save_state=work / "states" / f"{base}-i",
        )
        resumed = run_arm(
            work, f"{base}-r", arm, steps=total - split, digest_every=2, profile=True,
            start_step=split, resume_state=work / "states" / f"{base}-i",
            save_state=work / "states" / f"{base}-r",
        )
        check_run(uninterrupted)
        check_run(first)
        resumed_checks = check_run(resumed)
        u_dir = Path(run_arm_state(uninterrupted, "save_state"))
        r_dir = Path(run_arm_state(resumed, "save_state"))
        names = ["adapter_model.safetensors", "adapter_config.json",
                 "hierarchos_adapter_optimizer.safetensors"]
        hashes = {
            name: {"uninterrupted": sha256_file(u_dir / name), "resumed": sha256_file(r_dir / name)}
            for name in names
        }
        equal = all(value["uninterrupted"] == value["resumed"] for value in hashes.values())
        u_bits = [record["loss_bits"] for record in uninterrupted["report"]["records"][split:]]
        r_bits = [record["loss_bits"] for record in resumed["report"]["records"]]
        if not equal:
            raise AssertionError(f"{base}: resumed checkpoint bytes differ")
        if u_bits != r_bits:
            raise AssertionError(f"{base}: resumed losses differ")
        resumed_dispatch = resumed_checks["dispatch"]
        expected_backward_steps = (total - split) * 2
        if arm["backward"] == "scalar":
            if (resumed_dispatch["scalar"] != expected_backward_steps
                    or resumed_dispatch["query"] or resumed_dispatch["key"]):
                raise AssertionError(f"{base}: resumed scalar dispatch proof failed")
        else:
            if (resumed_dispatch["query"] != expected_backward_steps
                    or resumed_dispatch["key"] != expected_backward_steps
                    or resumed_dispatch["scalar"]):
                raise AssertionError(f"{base}: resumed two-pass dispatch proof failed")
        resume_state_dirs[arm_id] = {"uninterrupted": u_dir, "resumed": r_dir}
        resume[arm_id] = {
            "total_steps": total,
            "split_step": split,
            "checkpoint_sha256": hashes,
            "checkpoint_bytes_equal": equal,
            "loss_bits_equal": u_bits == r_bits,
            "loss_bits": {"uninterrupted_tail": u_bits, "resumed": r_bits},
            "resumed_optimizer_steps": [r["optimizer_step"] for r in resumed["report"]["records"]],
            "dispatch": resumed_checks["dispatch"],
            "parameter_frozen": resumed["report"]["parameter_frozen"],
            "adapter_sha256": {
                "uninterrupted": sha256_file(uninterrupted["adapter_dir"] / "adapter_model.safetensors"),
                "resumed": sha256_file(resumed["adapter_dir"] / "adapter_model.safetensors"),
            },
        }

    # Adverse conditions: padded longer sequence, alternate seed, determinism.
    adverse = {}
    padded = run_arm(
        work, "adverse-seq96", ARMS["F"], steps=steps, digest_every=2, profile=True, seq=96
    )
    adverse["seq96"] = {"checks": check_run(padded), "summary": summarize_run(padded)}
    seed_a = run_arm(work, "adverse-seed7", ARMS["F"], steps=steps, digest_every=2, profile=False, seed=7)
    seed_b = run_arm(work, "adverse-seed13", ARMS["F"], steps=steps, digest_every=2, profile=False, seed=13)
    check_run(seed_a)
    check_run(seed_b)
    adverse["seed"] = {
        "seed7_adapter": sha256_file(seed_a["adapter_dir"] / "adapter_model.safetensors"),
        "seed13_adapter": sha256_file(seed_b["adapter_dir"] / "adapter_model.safetensors"),
        "adapters_differ": sha256_file(seed_a["adapter_dir"] / "adapter_model.safetensors")
        != sha256_file(seed_b["adapter_dir"] / "adapter_model.safetensors"),
        "seed13_losses_finite": seed_b["report"]["all_losses_finite"],
        "seed13_final_loss": seed_b["report"]["final_loss"],
    }
    repeat = run_arm(work, "adverse-repeat", ARMS["F"], steps=steps, digest_every=1, profile=False)
    check_run(repeat)
    deterministic = (
        record_field(runs["F"]["report"]["records"], "loss_bits")
        == record_field(repeat["report"]["records"], "loss_bits")
        and record_field(runs["F"]["report"]["records"], "grad_digest")
        == record_field(repeat["report"]["records"], "grad_digest")
    )
    if not deterministic:
        raise AssertionError("repeated identical run diverged")
    adverse["determinism"] = {
        "run_a": runs["F"]["tag"],
        "run_b": repeat["tag"],
        "loss_and_gradient_bits_equal": deterministic,
        "steps": steps,
    }

    # Full-parameter exact checkpoint is explicitly unsupported; preserve refusal.
    refusal_dir = work / "states" / "full-ft-refusal"
    if refusal_dir.exists():
        shutil.rmtree(refusal_dir)
    refusal = subprocess.run(
        [
            str(TRAINER), "--model", str(MODEL), "--dataset", str(DATASET),
            "--output", str(work / "scratch" / "refusal-out"),
            "--batch-size", "8", "--seq-len", "64", "--max-steps", "1",
            "--save-training-state", str(refusal_dir),
        ],
        cwd=CRATE,
        capture_output=True,
        text=True,
        timeout=300,
    )
    refusal_text = (refusal.stdout + refusal.stderr).strip()
    refusal_ok = refusal.returncode != 0 and "full-parameter exact state is not implemented" in refusal_text
    if not refusal_ok:
        raise AssertionError("full-parameter checkpoint refusal no longer holds")
    (work / "logs").mkdir(exist_ok=True)
    (work / "logs" / "full-ft-refusal.txt").write_text(
        f"exit={refusal.returncode}\n{refusal_text}\n", encoding="utf-8"
    )

    performance = {}
    if not args.skip_perf:
        for arm_id, arm in ARMS.items():
            samples = {"scalar": [], "twopass": []}
            setup = []
            for round_index in range(perf_rounds):
                for position, backward in enumerate(
                    ["scalar", "twopass", "twopass", "scalar"]
                ):
                    variant = dict(arm)
                    variant["backward"] = backward
                    variant["two_pass"] = "0" if backward == "scalar" else "1"
                    tag = f"perf-{arm_id}-r{round_index}-{position}-{backward}"
                    run = run_arm(
                        work, tag, variant, steps=perf_steps, digest_every=0,
                        profile=False, warmup=perf_warmup,
                    )
                    check_run(run)
                    measured = [
                        record["step_ms"] for record in run["report"]["records"][perf_warmup:]
                    ]
                    samples[backward].extend(measured)
                    setup.append(run["setup_ms"])
            performance[arm_id] = {
                "config": arm,
                "steps": perf_steps,
                "warmup": perf_warmup,
                "rounds": perf_rounds,
                "scalar_samples": samples["scalar"],
                "twopass_samples": samples["twopass"],
                "setup_ms": setup,
                **latency_stats(samples["scalar"], samples["twopass"]),
            }

    wmma_reversal = {}
    if not args.skip_perf:
        for precision, wmma in (("fp32", "off"), ("mixed", "mixed")):
            base = dict(ARMS["F"])
            base["precision"] = precision
            base["wmma"] = wmma
            durations = []
            conversions = []
            reuses = []
            for repeat_index in range(reversal_repeats):
                run = run_arm(
                    work, f"reversal-{precision}-{repeat_index}", base,
                    steps=reversal_steps, digest_every=0, profile=False,
                )
                check_run(run)
                records = run["report"]["records"]
                durations.append([record["step_ms"] for record in records])
                conversions.append([record["wmma_weight_conversions"] for record in records])
                reuses.append([record["wmma_mirror_reuses"] for record in records])
            wmma_reversal[precision] = {
                "durations_ms": durations,
                "weight_conversions": conversions,
                "mirror_reuses": reuses,
                "windows": window_stats(durations),
            }
        wmma_reversal["mixed_vs_fp32"] = {
            "early": ratio(
                wmma_reversal["fp32"]["windows"]["early_median_ms"],
                wmma_reversal["mixed"]["windows"]["early_median_ms"],
            ),
            "middle": ratio(
                wmma_reversal["fp32"]["windows"]["middle_median_ms"],
                wmma_reversal["mixed"]["windows"]["middle_median_ms"],
            ),
            "late": ratio(
                wmma_reversal["fp32"]["windows"]["late_median_ms"],
                wmma_reversal["mixed"]["windows"]["late_median_ms"],
            ),
        }

    # Source-bound fingerprints.
    source_files = [
        "src/transformer.rs",
        "src/vulkan.rs",
        "src/wmma_linear.rs",
        "src/wmma.rs",
        "src/transformer/peft_state.rs",
        "src/lib.rs",
        "src/bin/hierarchos-vulkan-training-qualification.rs",
        "src/bin/transformer_train.rs",
        "Cargo.toml",
        "shaders/transformer_attention_backward.spv",
        "shaders/transformer_attention_backward_query_rdna3.spv",
        "shaders/transformer_attention_backward_key_rdna3.spv",
        "shaders/transformer_attention_backward_query_rdna3.comp",
        "shaders/transformer_attention_backward_key_rdna3.comp",
        "validation/training_shader_manifest.json",
        "validation/qualify_training_phase16.py",
        "validation/check_training_phase16.py",
        "validation/test_training_phase16.py",
    ]
    bindings = {}
    for relative in source_files:
        path = CRATE / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing bound source {path}")
        bindings[relative] = sha256_file(path)

    driver_path = CRATE / "target/release/hierarchos-vulkan-transformer-train.exe"
    evidence_dir = work / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True)
    for run in list(runs.values()) + [padded, seed_a, seed_b, repeat]:
        copy_evidence(work, evidence_dir, run)
    for arm_id, run in runs.items():
        if run["adapter_dir"] is not None:
            shutil.copy2(
                run["adapter_dir"] / "adapter_model.safetensors",
                evidence_dir / f"adapter-{arm_id}.safetensors",
            )
    (evidence_dir / "adverse").mkdir(exist_ok=True)
    for label, run in (("seed7", seed_a), ("seed13", seed_b)):
        shutil.copy2(
            run["adapter_dir"] / "adapter_model.safetensors",
            evidence_dir / "adverse" / f"{label}.safetensors",
        )
    for arm_id in ("E", "F", "H"):
        for suffix in ("-u", "-i", "-r"):
            pass
    # Resume runs are re-run here only for evidence copying; they are already on disk.
    for path in (work / "reports").glob("resume-*.json"):
        shutil.copy2(path, evidence_dir / path.name)
        stderr = work / "logs" / (path.stem + ".err.txt")
        if stderr.is_file():
            shutil.copy2(stderr, evidence_dir / stderr.name)
    for arm_id, dirs in resume_state_dirs.items():
        for phase in ("uninterrupted", "resumed"):
            target = evidence_dir / "states" / arm_id / ("u" if phase == "uninterrupted" else "r")
            target.mkdir(parents=True, exist_ok=True)
            for name in ("adapter_model.safetensors", "adapter_config.json",
                         "hierarchos_adapter_optimizer.safetensors"):
                shutil.copy2(dirs[phase] / name, target / name)
    for path in (work / "reports").glob("perf-*.json"):
        shutil.copy2(path, evidence_dir / path.name)
    for path in (work / "reports").glob("reversal-*.json"):
        shutil.copy2(path, evidence_dir / path.name)
    shutil.copy2(work / "logs" / "full-ft-refusal.txt", evidence_dir / "full-ft-refusal.txt")

    toolchains = {
        "rustc": command_version(["rustc", "--version"]),
        "cargo": command_version(["cargo", "--version"]),
        "glslc": command_version(["C:/VulkanSDK/1.4.321.1/Bin/glslc.exe", "--version"]),
        "spirv_val": command_version(["C:/VulkanSDK/1.4.321.1/Bin/spirv-val.exe", "--version"]),
        "python": sys.version.split()[0],
    }
    device = runs["A"]["report"]["device"]
    receipt = {
        "schema": SCHEMA,
        "kind": "phase16-training-qualification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_clean": not dirty,
        "phase16_receipt": {
            "path": str(PHASE16_RECEIPT.relative_to(CRATE)).replace("\\", "/"),
            "sha256": sha256_file(PHASE16_RECEIPT),
        },
        "toolchains": toolchains,
        "device": device,
        "fixture": {
            "model": str(MODEL.relative_to(CRATE)).replace("\\", "/"),
            "dataset": str(DATASET.relative_to(CRATE)).replace("\\", "/"),
            "model_config_sha256": sha256_file(MODEL / "config.json"),
            "model_weights_sha256": sha256_file(MODEL / "model.safetensors"),
            "dataset_sha256": sha256_file(DATASET),
            "parameters": EXPECTED_PARAMETERS,
        },
        "binary_sha256": sha256_file(BIN),
        "trainer_binary_sha256": sha256_file(driver_path),
        "source_files": bindings,
        "arms": arms,
        "comparisons": comparisons,
        "resume": resume,
        "adverse": adverse,
        "performance": performance,
        "wmma_reversal": wmma_reversal,
        "refusal": {
            "exit": refusal.returncode,
            "message_present": "full-parameter exact state is not implemented" in refusal_text,
            "sha256": sha256_file(work / "logs" / "full-ft-refusal.txt"),
        },
        "limitations": [
            "Full-parameter gradient tensors are not exposed by the native trainer; "
            "full-FT runs record exact per-step parameter trajectories and update "
            "norms, while gradient norms/digests are recorded for the LoRA arms.",
            "Gradient accumulation, forced non-finite updates, bidirectional "
            "attention, grouped-query fallback and alternate head dimensions are "
            "not supported by this fixture and are not exercised here; the "
            "two-pass support gate is qualified in the Phase XVI receipt.",
            "Mixed-precision loss drift is gated at the matched initial step "
            "against wmma::MIXED_VS_FP32_MAX_ABS; the full trajectory drift is "
            "reported because full-fine-tuning trajectories separate after the "
            "first optimizer update.",
            "Performance is a single-device measurement on this RDNA3 part, not a "
            "cross-vendor throughput claim.",
        ],
    }
    evidence_files = files_under(evidence_dir)
    receipt["evidence"] = evidence_files
    (work / "current-qualification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(f"receipt={work / 'current-qualification.json'}")
    print(f"evidence_files={len(evidence_files)}")

    if args.install:
        if RECEIPT_DIR.exists():
            shutil.rmtree(RECEIPT_DIR)
        RECEIPT_DIR.mkdir(parents=True)
        shutil.copytree(evidence_dir, RECEIPT_DIR / "evidence")
        shutil.copy2(work / "current-qualification.json", RECEIPT_DIR / "current-qualification.json")
        print(f"installed={RECEIPT_DIR}")
    return 0


def run_arm_state(run: dict, key: str) -> str:
    value = run.get(key)
    return str(value) if value is not None else ""


def copy_evidence(work: Path, evidence: Path, run: dict) -> None:
    shutil.copy2(run["report_path"], evidence / run["report_path"].name)
    shutil.copy2(run["stderr_path"], evidence / run["stderr_path"].name)


def latency_stats(scalar: list[float], twopass: list[float]) -> dict:
    if not scalar or not twopass:
        return {}
    def med(values):
        return statistics.median(values)
    def p95(values):
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    def mean(values):
        return statistics.fmean(values)
    def std(values):
        return statistics.pstdev(values) if len(values) > 1 else 0.0
    rng = random.Random(1618)
    ratios = []
    for _ in range(10_000):
        a = [rng.choice(scalar) for _ in range(len(scalar))]
        b = [rng.choice(twopass) for _ in range(len(twopass))]
        ratios.append(med(a) / med(b))
    ratios.sort()
    return {
        "scalar_median_ms": med(scalar),
        "twopass_median_ms": med(twopass),
        "scalar_p95_ms": p95(scalar),
        "twopass_p95_ms": p95(twopass),
        "scalar_mean_ms": mean(scalar),
        "twopass_mean_ms": mean(twopass),
        "scalar_std_ms": std(scalar),
        "twopass_std_ms": std(twopass),
        "speedup": med(scalar) / med(twopass),
        "speedup_ci95": [ratios[249], ratios[9749]],
        "sample_count": {"scalar": len(scalar), "twopass": len(twopass)},
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def window_stats(durations: list[list[float]]) -> dict:
    windows = {
        "early": (0, 8),
        "middle": (16, 32),
        "late": (40, 48),
    }
    result = {}
    for name, (start, end) in windows.items():
        values = [
            value
            for run in durations
            for value in run[start:min(end, len(run))]
        ]
        result[f"{name}_median_ms"] = statistics.median(values) if values else float("nan")
        result[f"{name}_mean_ms"] = statistics.fmean(values) if values else float("nan")
        result[f"{name}_samples"] = len(values)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
