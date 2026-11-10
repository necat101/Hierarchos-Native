#!/usr/bin/env python3
"""Phase XVI full-fine-tuning stability qualification producer.

This is deliberately a separate producer/receipt from the sealed
20261110-phase16-training-qualification receipt. It runs a controlled AdamW
learning-rate sweep on FP32 scalar/two-pass, selects a stable LR by a declared
rule, then runs 100-step convergence for all four full-FT arms, exact LoRA
resume for E/F/H, seed/geometry reuse checks, and a fresh ABBA benchmark.

The checker recomputes every claim from the shipped per-step reports; no
summary field is trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import qualify_training_phase16 as q

CRATE = q.CRATE
REPO = q.REPO
BIN = CRATE / "target/release/training-stability.exe"
TRAINER = q.TRAINER
MODEL = q.MODEL
DATASET = q.DATASET
RECEIPT_DIR = (
    CRATE
    / "validation/release_parity/multimodal_training"
    / "20261110-phase16-training-stability"
)
PHASE16_RECEIPT = q.PHASE16_RECEIPT
SCHEMA = 2
KIND = "phase16-training-stability"
EXPECTED_PARAMETERS = q.EXPECTED_PARAMETERS

SWEEP_LRS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2)
CRITERION = (
    "stable iff all losses finite, final < 0.5*first, last-20% mean < "
    "0.5*first-20% mean, max <= 1.5*first, and least-squares slope < 0; "
    "select the stable LR with the lowest late-window mean (ties: lower LR)"
)

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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_harness(
    work: Path,
    tag: str,
    arm: dict,
    *,
    lr: float,
    steps: int,
    profile: bool,
    seq: int = 64,
    seed: int = 7,
    start_step: int = 0,
    warmup: int = 0,
    digest_every: int = 1,
    save_state: Path | None = None,
    resume_state: Path | None = None,
    adapter_out: Path | None = None,
) -> dict:
    reports = work / "reports"
    logs = work / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"{tag}.json"
    out = work / "scratch" / tag
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stderr_path = logs / f"{tag}.err.txt"
    stdout_path = logs / f"{tag}.out.txt"
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
        "--lr", f"{lr:.10g}",
        "--beta1", "0.9",
        "--beta2", "0.999",
        "--eps", "1e-8",
        "--weight-decay", "0",
        "--seed", str(seed),
        "--seq", str(seq),
        "--log-every", "10",
        "--report", str(report_path),
    ]
    if arm["mode"] == "lora" or resume_state is not None:
        cmd += ["--lora-rank", "8", "--lora-alpha", "16"]
    if adapter_out is not None:
        if adapter_out.exists():
            shutil.rmtree(adapter_out)
        adapter_out.mkdir(parents=True, exist_ok=True)
        cmd += ["--adapter-out", str(adapter_out)]
    if save_state is not None:
        if save_state.exists():
            shutil.rmtree(save_state)
        cmd += ["--save-state", str(save_state)]
    if resume_state is not None:
        cmd += ["--resume-state", str(resume_state)]
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            cmd,
            cwd=CRATE,
            env=q.arm_env(arm, profile),
            stdout=stdout,
            stderr=stderr,
            timeout=1800,
        )
    process_ms = (time.monotonic() - started) * 1000.0
    if completed.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{tag}: harness exited {completed.returncode}\n{tail}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    blocks = q.parse_profile(stderr_text) if profile else []
    execution_ms = sum(float(record["step_ms"]) for record in report["records"])
    return {
        "tag": tag,
        "arm": arm,
        "lr": lr,
        "report": report,
        "report_path": report_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "adapter_dir": adapter_out,
        "save_state": save_state,
        "resume_state": resume_state,
        "returns": completed.returncode,
        "process_ms": process_ms,
        "setup_ms": max(0.0, process_ms - execution_ms),
        "expected_steps": steps,
        "start_step": start_step,
        "blocks": blocks,
    }


def check_harness(run: dict) -> dict:
    report = run["report"]
    arm = run["arm"]
    steps = run["expected_steps"]
    lr = run["lr"]
    records = report["records"]
    assert len(records) == steps, f"{run['tag']}: record count"
    assert report["steps"] == steps, f"{run['tag']}: report step count"
    assert report["start_step"] == run["start_step"], f"{run['tag']}: start step"
    assert [r["optimizer_step"] for r in records] == list(
        range(run["start_step"] + 1, run["start_step"] + steps + 1)
    ), f"{run['tag']}: optimizer sequence"
    assert report["mode"] == ("full-finetune" if arm["mode"] == "full" else "lora"), (
        f"{run['tag']}: mode"
    )
    assert report["lora"] == (arm["mode"] == "lora"), f"{run['tag']}: LoRA flag"
    assert report["two_pass_requested"] == (arm["two_pass"] == "1"), f"{run['tag']}: two-pass"
    assert report["precision_requested"] == ("mixed" if arm["precision"] == "mixed" else "off"), (
        f"{run['tag']}: precision"
    )
    assert abs(report["learning_rate"] - lr) <= 1e-12, f"{run['tag']}: report LR"
    assert report["lr_schedule"] == "constant", f"{run['tag']}: LR schedule"
    assert report["gradient_norm_source"] == (
        "native-lora-gradients" if arm["mode"] == "lora" else "unavailable-full-parameter"
    ), f"{run['tag']}: gradient norm source"
    for key, expected in (
        ("beta1", 0.9),
        ("beta2", 0.999),
        ("eps", 1e-8),
        ("weight_decay", 0.0),
    ):
        assert abs(report[key] - expected) <= 1e-12, f"{run['tag']}: {key}"
        for record in records:
            assert abs(record[key] - expected) <= 1e-12, f"{run['tag']}: per-step {key}"
    for index, record in enumerate(records):
        assert record["event"] == "qual_step", f"{run['tag']}: event"
        assert abs(record["learning_rate"] - lr) <= 1e-12, f"{run['tag']}: per-step LR"
        assert record["optimizer_status"] == "applied", f"{run['tag']}: optimizer status"
        assert record["loss_finite"] is True, f"{run['tag']}: loss finite"
        assert record.get("grad_nonfinite") in (None, 0), f"{run['tag']}: gradients"
        assert record.get("param_nonfinite") in (None, 0), f"{run['tag']}: parameters"
        assert record["global_step"] == run["start_step"] + index, f"{run['tag']}: global step"
    assert report["all_losses_finite"] is True, f"{run['tag']}: loss flag"
    if arm["mode"] == "full":
        assert report["parameter_elements"] == EXPECTED_PARAMETERS, f"{run['tag']}: params"
        assert report["parameter_end_digest"], f"{run['tag']}: final digest"
    else:
        assert report["parameter_frozen"] is True, f"{run['tag']}: base moved"
        assert report["grad_nonfinite_total"] == 0, f"{run['tag']}: nonfinite LoRA grads"
        assert run["adapter_dir"] is not None, f"{run['tag']}: missing adapter"
        assert (run["adapter_dir"] / "adapter_model.safetensors").is_file(), (
            f"{run['tag']}: adapter file"
        )
    checks = q.check_run(run)
    return checks


def loss_metrics(losses: list[float]) -> dict:
    first, final = float(losses[0]), float(losses[-1])
    n = len(losses)
    width = max(1, n // 5)
    early = losses[:width]
    late = losses[-width:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    mean = sum(losses) / n
    variance = sum((value - mean) ** 2 for value in losses) / n
    xs = list(range(n))
    x_mean = sum(xs) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (value - mean) for x, value in zip(xs, losses)) / denom
    return {
        "steps": n,
        "first": first,
        "final": final,
        "min": min(losses),
        "max": max(losses),
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "reduction": (first - final) / first if first != 0 else 0.0,
        "late_reduction": (early_mean - late_mean) / early_mean if early_mean != 0 else 0.0,
        "slope": slope,
        "max_spike_ratio": max(losses) / first if first != 0 else float("inf"),
        "stable": (
            all(math.isfinite(value) for value in losses)
            and final < 0.5 * first
            and late_mean < 0.5 * early_mean
            and max(losses) <= 1.5 * first
            and slope < 0.0
        ),
    }


def compare_runs(left: dict, right: dict, fields: list[str]) -> dict:
    a, b = left["report"]["records"], right["report"]["records"]
    assert len(a) == len(b), "comparison step mismatch"
    result = {"left": left["tag"], "right": right["tag"], "fields": {}, "bitwise_equal": True}
    for field in fields:
        av = [record[field] for record in a]
        bv = [record[field] for record in b]
        result["fields"][field] = {"equal": av == bv}
        if av != bv:
            result["bitwise_equal"] = False
    if left["report"]["mode"] == "full-finetune":
        result["final_parameter_digest_equal"] = (
            left["report"]["parameter_end_digest"] == right["report"]["parameter_end_digest"]
        )
    else:
        ah = q.sha256_file(left["adapter_dir"] / "adapter_model.safetensors")
        bh = q.sha256_file(right["adapter_dir"] / "adapter_model.safetensors")
        result["adapter_sha256"] = {"left": ah, "right": bh, "equal": ah == bh}
    return result


def drift(left: dict, right: dict) -> dict:
    a = [record["loss"] for record in left["report"]["records"]]
    b = [record["loss"] for record in right["report"]["records"]]
    diffs = [x - y for x, y in zip(a, b)]
    return {
        "left": left["tag"],
        "right": right["tag"],
        "first_step_abs": abs(diffs[0]),
        "first_step_within_policy": abs(diffs[0]) <= 2.0e-2,
        "policy_max_abs": 2.0e-2,
        "max_abs": max(abs(value) for value in diffs),
        "rms": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
        "final_abs": abs(diffs[-1]),
        "steps": len(diffs),
    }


def summary(run: dict, checks: dict) -> dict:
    report = run["report"]
    return {
        "tag": run["tag"],
        "mode": report["mode"],
        "precision": run["arm"]["precision"],
        "backward": run["arm"]["backward"],
        "lr": run["lr"],
        "steps": report["steps"],
        "start_step": report["start_step"],
        "optimizer_steps": report["optimizer_steps"],
        "first_loss": report["first_loss"],
        "final_loss": report["final_loss"],
        "loss_min": report["loss_min"],
        "loss_max": report["loss_max"],
        "all_losses_finite": report["all_losses_finite"],
        "learning_rate": report["learning_rate"],
        "beta1": report["beta1"],
        "beta2": report["beta2"],
        "eps": report["eps"],
        "weight_decay": report["weight_decay"],
        "lr_schedule": report["lr_schedule"],
        "gradient_norm_source": report["gradient_norm_source"],
        "grad_norm_min": report["grad_norm_min"],
        "grad_norm_max": report["grad_norm_max"],
        "parameter_start_digest": report["parameter_start_digest"],
        "parameter_end_digest": report["parameter_end_digest"],
        "parameter_elements": report["parameter_elements"],
        "parameter_frozen": report["parameter_frozen"],
        "measured_step_ms_mean": report["measured_step_ms_mean"],
        "process_ms": round(run["process_ms"], 3),
        "setup_ms": round(run["setup_ms"], 3),
        "returncode": run["returns"],
        "adapter_sha256": (
            q.sha256_file(run["adapter_dir"] / "adapter_model.safetensors")
            if run["adapter_dir"]
            else None
        ),
        "profile_blocks": len(run["blocks"]),
        "dispatch": checks["dispatch"],
        "wmma": checks["wmma"],
        "report_sha256": q.sha256_file(run["report_path"]),
        "stderr_sha256": q.sha256_file(run["stderr_path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not BIN.is_file():
        print(f"missing stability binary {BIN}; build it first", file=sys.stderr)
        return 2
    dirty = q.git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        print(f"worktree has tracked changes:\n{dirty}", file=sys.stderr)
        return 2
    revision = q.git("rev-parse", "HEAD")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    work = args.work or (CRATE / "target/stability-phase16" / stamp)
    work.mkdir(parents=True, exist_ok=True)
    sweep_steps = 20 if args.quick else 100
    conv_steps = 20 if args.quick else 100
    perf_steps = 8 if args.quick else 20
    perf_rounds = 1 if args.quick else 2
    lrs = (1.0e-3,) if args.quick else SWEEP_LRS
    print(f"work={work}")
    print(f"revision={revision}")

    # --- controlled LR sweep on FP32 scalar/two-pass -------------------------
    sweep: dict[str, dict] = {}
    for lr in lrs:
        sweep[str(lr)] = {}
        for arm_id in ("A", "B"):
            tag = f"sweep-{lr:g}-{arm_id}"
            print(f"sweep lr={lr:g} {arm_id}")
            stderr = work / "logs" / f"{tag}.err.txt"
            try:
                run = run_harness(
                    work, tag, ARMS[arm_id], lr=lr, steps=sweep_steps, profile=False
                )
                check_harness(run)
                losses = [record["loss"] for record in run["report"]["records"]]
                sweep[str(lr)][arm_id] = {
                    "report": run,
                    "metrics": loss_metrics(losses),
                }
            except Exception as error:
                print(f"sweep lr={lr:g} {arm_id} FAILED: {error}")
                sweep[str(lr)][arm_id] = {
                    "error": str(error),
                    "stderr": stderr.name,
                    "stderr_sha256": q.sha256_file(stderr) if stderr.is_file() else None,
                }
        a = sweep[str(lr)]["A"].get("report")
        b = sweep[str(lr)]["B"].get("report")
        if a is not None and b is not None:
            a_records = a["report"]["records"]
            b_records = b["report"]["records"]
            assert [r["loss_bits"] for r in a_records] == [r["loss_bits"] for r in b_records], (
                f"sweep {lr:g}: scalar/two-pass loss divergence"
            )
            assert [r["param_digest"] for r in a_records] == [r["param_digest"] for r in b_records], (
                f"sweep {lr:g}: scalar/two-pass parameter divergence"
            )
    stable_lrs = [
        float(lr) for lr in lrs
        if "report" in sweep[str(lr)]["A"]
        and "report" in sweep[str(lr)]["B"]
        and sweep[str(lr)]["A"]["metrics"]["stable"]
        and sweep[str(lr)]["B"]["metrics"]["stable"]
    ]
    if not stable_lrs:
        raise RuntimeError("no learning rate satisfied the declared stability criterion")
    selected_lr = min(
        stable_lrs,
        key=lambda lr: (sweep[str(lr)]["A"]["metrics"]["late_mean"], lr),
    )
    print(f"selected_lr={selected_lr:g} stable={stable_lrs}")

    # --- additional-seed validation at the selected LR -----------------------
    seeds: dict[str, dict] = {}
    for seed in ((7,) if args.quick else (13, 21)):
        tag = f"seed-{seed}"
        print(f"seed validation {seed}")
        run = run_harness(
            work, tag, ARMS["A"], lr=selected_lr, steps=conv_steps, profile=False, seed=seed
        )
        check_harness(run)
        metrics = loss_metrics([record["loss"] for record in run["report"]["records"]])
        assert metrics["stable"], f"seed {seed} did not satisfy the stability criterion"
        seeds[str(seed)] = {"report": run, "metrics": metrics}

    # --- 100-step convergence, all four full-FT arms -------------------------
    arms: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for arm_id in ("A", "B", "C", "D"):
        print(f"convergence arm {arm_id}")
        run = run_harness(
            work, f"arm-{arm_id}", ARMS[arm_id], lr=selected_lr,
            steps=conv_steps, profile=True
        )
        checks = check_harness(run)
        metrics = loss_metrics([record["loss"] for record in run["report"]["records"]])
        if not metrics["stable"]:
            raise RuntimeError(f"arm {arm_id} did not satisfy the convergence criterion: {metrics}")
        runs[arm_id] = run
        arms[arm_id] = {"config": ARMS[arm_id], "summary": summary(run, checks),
                        "metrics": metrics}
    comparisons = {
        "A_vs_B": compare_runs(runs["A"], runs["B"],
                               ["loss_bits", "param_digest", "update_norm", "update_max_abs"]),
        "C_vs_D": compare_runs(runs["C"], runs["D"],
                               ["loss_bits", "param_digest", "update_norm", "update_max_abs"]),
        "B_vs_D": drift(runs["B"], runs["D"]),
        "A_vs_C": drift(runs["A"], runs["C"]),
    }
    for key in ("A_vs_B", "C_vs_D"):
        assert comparisons[key]["bitwise_equal"], f"{key} is not bitwise identical"
        assert comparisons[key]["final_parameter_digest_equal"], f"{key} final parameters differ"
    for key in ("B_vs_D", "A_vs_C"):
        assert comparisons[key]["first_step_within_policy"], f"{key} first-step mixed drift"

    # --- exact LoRA resume at the selected LR -------------------------------
    resume: dict[str, dict] = {}
    resume_runs: dict[str, dict] = {}
    for arm_id in ("E", "F", "H"):
        arm = ARMS[arm_id]
        total, split = 12, 8
        u_state = work / "states" / arm_id / "u"
        u = run_harness(
            work, f"resume-{arm_id}-u", arm, lr=selected_lr, steps=total,
            profile=False, save_state=u_state,
            adapter_out=work / "adapters" / f"resume-{arm_id}-u",
        )
        check_harness(u)
        state = work / "states" / arm_id / "i"
        i = run_harness(
            work, f"resume-{arm_id}-i", arm, lr=selected_lr, steps=split,
            profile=False, save_state=state, adapter_out=work / "adapters" / f"resume-{arm_id}-i",
        )
        check_harness(i)
        r = run_harness(
            work, f"resume-{arm_id}-r", arm, lr=selected_lr, steps=total - split,
            start_step=split, profile=True, resume_state=state,
            adapter_out=work / "adapters" / f"resume-{arm_id}-r",
        )
        check_harness(r)
        # The resumed process writes its post-4-step optimizer state only if it
        # is asked to save again; the uninterrupted 12-step run saved its own
        # state above, so both sides are full 12-step artifacts.
        restate = work / "states" / arm_id / "r"
        # Re-run the tail with a state save so the resumed optimizer bytes exist.
        r2 = run_harness(
            work, f"resume-{arm_id}-r-save", arm, lr=selected_lr, steps=total - split,
            start_step=split, profile=False, resume_state=state, save_state=restate,
            adapter_out=work / "adapters" / f"resume-{arm_id}-r-save",
        )
        check_harness(r2)
        u_opt_paths = {
            "adapter_model.safetensors": u["adapter_dir"] / "adapter_model.safetensors",
            "adapter_config.json": u["adapter_dir"] / "adapter_config.json",
            "hierarchos_adapter_optimizer.safetensors": u_state / "hierarchos_adapter_optimizer.safetensors",
        }
        r_opt_paths = {
            "adapter_model.safetensors": r2["adapter_dir"] / "adapter_model.safetensors",
            "adapter_config.json": r2["adapter_dir"] / "adapter_config.json",
            "hierarchos_adapter_optimizer.safetensors": restate / "hierarchos_adapter_optimizer.safetensors",
        }
        # The uninterrupted 12-step adapter must be exported from a 12-step run;
        # u has it. The resumed tail's 12-step adapter is r2.
        u_hashes = {name: q.sha256_file(path) for name, path in u_opt_paths.items()}
        r_hashes = {name: q.sha256_file(path) for name, path in r_opt_paths.items()}
        u_bits = [record["loss_bits"] for record in u["report"]["records"][split:]]
        r_bits = [record["loss_bits"] for record in r["report"]["records"]]
        assert u_bits == r_bits, f"resume {arm_id}: loss bits differ"
        assert u_hashes == r_hashes, f"resume {arm_id}: checkpoint bytes differ"
        assert r["report"]["parameter_frozen"] is True, f"resume {arm_id}: base moved"
        resume[arm_id] = {
            "total_steps": total,
            "split_step": split,
            "checkpoint_sha256": {
                name: {"uninterrupted": u_hashes[name], "resumed": r_hashes[name]}
                for name in u_hashes
            },
            "checkpoint_bytes_equal": True,
            "loss_bits_equal": True,
            "loss_bits": {"uninterrupted_tail": u_bits, "resumed": r_bits},
            "resumed_optimizer_steps": [record["optimizer_step"] for record in r["report"]["records"]],
            "dispatch": (q.check_run(r)["dispatch"] if r["blocks"] else None),
            "parameter_frozen": True,
            "adapter_sha256": {"uninterrupted": u_hashes["adapter_model.safetensors"],
                               "resumed": r_hashes["adapter_model.safetensors"]},
        }
        resume_runs[arm_id] = {
            "u": u, "i": i, "r": r, "r2": r2,
            "u_state": u_state, "restate": restate,
        }

    # --- geometry reuse and deterministic repeat ----------------------------
    adverse: dict[str, dict] = {}
    seq96 = {}
    for arm_id in ("A", "B"):
        run = run_harness(work, f"adverse-seq96-{arm_id}", ARMS[arm_id],
                          lr=selected_lr, steps=8 if args.quick else 20, seq=96, profile=True)
        checks = check_harness(run)
        seq96[arm_id] = {
            "summary": summary(run, checks),
            "metrics": loss_metrics([record["loss"] for record in run["report"]["records"]]),
            "report": run,
        }
    adverse["seq96"] = seq96
    repeat = {}
    for index in (1, 2):
        run = run_harness(work, f"repeat-{index}", ARMS["A"],
                          lr=selected_lr, steps=20, profile=False)
        check_harness(run)
        repeat[str(index)] = run
    assert [record["loss_bits"] for record in repeat["1"]["report"]["records"]] == [
        record["loss_bits"] for record in repeat["2"]["report"]["records"]
    ], "repeat loss divergence"
    assert repeat["1"]["report"]["parameter_end_digest"] == repeat["2"]["report"]["parameter_end_digest"], (
        "repeat final parameter divergence"
    )
    adverse["repeat"] = {
        "loss_bits_equal": True,
        "parameter_digest_equal": True,
        "steps": 20,
        "sentinel_scratch": "unavailable: two-pass scratch is private and has no public poison hook",
    }
    adverse["seeds"] = seeds

    # --- fresh paired benchmark ---------------------------------------------
    performance = {}
    if not args.skip_perf:
        for arm_id, backward in (("A", "scalar"), ("B", "twopass")):
            samples = {"scalar": [], "twopass": []}
            for round_index in range(perf_rounds):
                order = (("A", "scalar"), ("B", "twopass"),
                         ("B", "twopass"), ("A", "scalar"))
                for run_arm_id, mode in order:
                    run = run_harness(
                        work, f"perf-{arm_id}-r{round_index}-{len(samples[mode])}-{mode}",
                        ARMS[run_arm_id], lr=selected_lr, steps=perf_steps,
                        warmup=3, profile=False,
                    )
                    check_harness(run)
                    measured = [
                        record["step_ms"] for record in run["report"]["records"][3:]
                    ]
                    samples[mode].extend(measured)
            performance[arm_id] = {
                "steps": perf_steps,
                "warmup": 3,
                "rounds": perf_rounds,
                "samples": samples,
                **q.latency_stats(samples["scalar"], samples["twopass"]),
            }

    # --- evidence and receipt ------------------------------------------------
    evidence = work / "evidence"
    if evidence.exists():
        shutil.rmtree(evidence)
    evidence.mkdir(parents=True)
    for path in sorted((work / "reports").glob("*.json")):
        shutil.copy2(path, evidence / path.name)
    for path in sorted((work / "logs").glob("*.err.txt")):
        shutil.copy2(path, evidence / path.name)
    for arm_id in ("E", "F", "H"):
        bundle = resume_runs[arm_id]
        for label, run in (("u", bundle["u"]), ("r", bundle["r2"])):
            if run["adapter_dir"]:
                shutil.copy2(run["adapter_dir"] / "adapter_model.safetensors",
                             evidence / f"adapter-{arm_id}-{label}.safetensors")
        for phase, state in (("u", bundle["u_state"]), ("r", bundle["restate"])):
            target = evidence / "states" / arm_id / phase
            target.mkdir(parents=True, exist_ok=True)
            for name in ("adapter_model.safetensors", "adapter_config.json",
                         "hierarchos_adapter_optimizer.safetensors"):
                shutil.copy2(state / name, target / name)
    for path in sorted((work / "reports").glob("*.json")):
        pass

    source_files = [
        "src/transformer.rs",
        "src/vulkan.rs",
        "src/wmma_linear.rs",
        "src/wmma.rs",
        "src/transformer/peft_state.rs",
        "src/lib.rs",
        "src/bin/training-stability.rs",
        "src/bin/transformer_train.rs",
        "Cargo.toml",
        "shaders/transformer_attention_backward.spv",
        "shaders/transformer_attention_backward_query_rdna3.spv",
        "shaders/transformer_attention_backward_key_rdna3.spv",
        "shaders/transformer_attention_backward_query_rdna3.comp",
        "shaders/transformer_attention_backward_key_rdna3.comp",
        "validation/training_shader_manifest.json",
        "validation/qualify_training_stability.py",
        "validation/check_training_stability.py",
        "validation/test_training_stability.py",
    ]
    bindings = {}
    for relative in source_files:
        path = CRATE / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing bound source {path}")
        bindings[relative] = q.sha256_file(path)

    receipt = {
        "schema": SCHEMA,
        "kind": KIND,
        "generated_at": now(),
        "source_revision": revision,
        "source_clean": not dirty,
        "phase16_receipt": {
            "path": str(PHASE16_RECEIPT.relative_to(CRATE)).replace("\\", "/"),
            "sha256": q.sha256_file(PHASE16_RECEIPT),
        },
        "toolchains": {
            "rustc": q.command_version(["rustc", "--version"]),
            "cargo": q.command_version(["cargo", "--version"]),
            "python": sys.version.split()[0],
        },
        "device": runs["A"]["report"]["device"],
        "fixture": {
            "model": str(MODEL.relative_to(CRATE)).replace("\\", "/"),
            "dataset": str(DATASET.relative_to(CRATE)).replace("\\", "/"),
            "model_config_sha256": q.sha256_file(MODEL / "config.json"),
            "model_weights_sha256": q.sha256_file(MODEL / "model.safetensors"),
            "dataset_sha256": q.sha256_file(DATASET),
            "parameters": EXPECTED_PARAMETERS,
        },
        "binaries": {
            "harness": q.sha256_file(BIN),
            "trainer": q.sha256_file(TRAINER),
        },
        "sweep": {
            "steps": sweep_steps,
            "lrs": [float(lr) for lr in lrs],
            "criterion": CRITERION,
            "selected_lr": selected_lr,
            "stable_lrs": stable_lrs,
            "runs": {
                lr: {
                    arm_id: (
                        {"metrics": entry["metrics"],
                         "summary": summary(entry["report"],
                                            q.check_run(entry["report"]))}
                        if "report" in entry else
                        {"error": entry["error"],
                         "stderr": entry["stderr"],
                         "stderr_sha256": entry["stderr_sha256"]}
                    )
                    for arm_id, entry in arms_for_lr.items()
                }
                for lr, arms_for_lr in sweep.items()
            },
        },
        "seeds": {
            seed: {"metrics": entry["metrics"],
                   "summary": summary(entry["report"], q.check_run(entry["report"]))}
            for seed, entry in seeds.items()
        },
        "arms": arms,
        "comparisons": comparisons,
        "resume": resume,
        "adverse": adverse,
        "performance": performance,
        "limitations": [
            "full-parameter raw gradient tensors are not exposed; per-step gradient norms are recorded only for LoRA",
            "full-parameter exact checkpoint resume is not implemented and is refused",
            "two-pass scratch has no public sentinel/poison hook; sentinel contamination is not claimed",
            "mixed WMMA remains opt-in",
            "memory, scratch-byte overhead and checkpoint-export timing are not measured",
            "the sweep and convergence measure training loss, not held-out generalization",
        ],
        "source_files": bindings,
        "evidence": {},
    }
    evidence_files = {}
    for path in sorted(evidence.rglob("*")):
        if path.is_file():
            evidence_files[str(path.relative_to(evidence)).replace("\\", "/")] = q.sha256_file(path)
    receipt["evidence"] = evidence_files
    work_receipt = work / "current-qualification.json"
    work_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt={work_receipt}")
    print(f"evidence_files={len(evidence_files)}")
    if args.install:
        if RECEIPT_DIR.exists():
            shutil.rmtree(RECEIPT_DIR)
        RECEIPT_DIR.mkdir(parents=True)
        shutil.copytree(evidence, RECEIPT_DIR / "evidence")
        shutil.copy2(work_receipt, RECEIPT_DIR / "current-qualification.json")
        print(f"installed={RECEIPT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"stability qualification failed: {error}", file=sys.stderr)
        raise
