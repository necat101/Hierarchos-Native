#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XVI training-stability receipt.

Re-derives the LR sweep, selection rule, convergence runs, per-step AdamW
telemetry, dispatch counts, comparisons, exact LoRA resume, seed/reuse checks
and ABBA statistics from the shipped evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import qualify_training_stability as p
import qualify_training_phase16 as q

SCHEMA = 2
KIND = "phase16-training-stability"


class QualificationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    left, right = float(left), float(right)
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def load_run(evidence: Path, name: str, arm: dict, steps: int, start: int,
             adapter: Path | None = None, profile: bool = False) -> dict:
    report = read_json(evidence / f"{name}.json")
    stderr = evidence / f"{name}.err.txt"
    blocks = q.parse_profile(stderr.read_text(encoding="utf-8", errors="replace")) if profile else []
    return {
        "tag": name,
        "arm": arm,
        "report": report,
        "expected_steps": steps,
        "start_step": start,
        "adapter_dir": adapter,
        "blocks": blocks,
    }


def verify(receipt_dir: Path, crate: Path | None = None) -> dict:
    crate = crate or q.CRATE
    receipt_path = receipt_dir / "current-qualification.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing {receipt_path}")
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == SCHEMA, "unexpected receipt schema")
    require(receipt.get("kind") == KIND, "unexpected receipt kind")
    evidence = receipt_dir / "evidence"
    require(evidence.is_dir(), "missing evidence directory")

    revision = receipt.get("source_revision", "")
    require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
            "invalid source revision")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=crate.parent, capture_output=True, text=True,
    )
    require(ancestor.returncode == 0, "source revision is not an ancestor of HEAD")
    require(receipt.get("source_clean") is True, "receipt was produced from a dirty worktree")

    for relative, expected in receipt["source_files"].items():
        path = crate / relative
        require(path.is_file(), f"missing bound source {relative}")
        require(q.sha256_file(path) == expected, f"bound source changed: {relative}")
    require("Cargo.toml" in receipt["source_files"], "Cargo.toml binding missing")

    binary = crate / "target/release/training-stability.exe"
    trainer = crate / "target/release/hierarchos-vulkan-transformer-train.exe"
    binaries_present = binary.is_file() and trainer.is_file()
    if binaries_present:
        require(q.sha256_file(binary) == receipt["binaries"]["harness"],
                "stability harness binary changed")
        require(q.sha256_file(trainer) == receipt["binaries"]["trainer"],
                "trainer binary changed")

    model = crate / receipt["fixture"]["model"]
    dataset = crate / receipt["fixture"]["dataset"]
    fixture_present = (
        (model / "config.json").is_file()
        and (model / "model.safetensors").is_file()
        and dataset.is_file()
    )
    if fixture_present:
        require(q.sha256_file(model / "config.json") == receipt["fixture"]["model_config_sha256"],
                "model config changed")
        require(q.sha256_file(model / "model.safetensors") == receipt["fixture"]["model_weights_sha256"],
                "model weights changed")
        require(q.sha256_file(dataset) == receipt["fixture"]["dataset_sha256"],
                "dataset changed")

    phase16 = crate / receipt["phase16_receipt"]["path"]
    require(phase16.is_file(), "Phase XVI receipt missing")
    require(q.sha256_file(phase16) == receipt["phase16_receipt"]["sha256"],
            "Phase XVI receipt changed")

    recorded_evidence = receipt.get("evidence", {})
    require(recorded_evidence, "missing evidence map")
    for relative, expected in recorded_evidence.items():
        path = evidence / relative
        require(path.is_file(), f"missing evidence {relative}")
        require(q.sha256_file(path) == expected, f"evidence changed: {relative}")

    def check_optim(run: dict, lr: float, steps: int, start: int) -> None:
        report = run["report"]
        records = report["records"]
        require(len(records) == steps, f"{run['tag']} record count")
        require(abs(report["learning_rate"] - lr) <= 1e-12, f"{run['tag']} run LR")
        require(report["lr_schedule"] == "constant", f"{run['tag']} LR schedule")
        require([r["optimizer_step"] for r in records]
                == list(range(start + 1, start + steps + 1)), f"{run['tag']} optimizer steps")
        for record in records:
            require(abs(record["learning_rate"] - lr) <= 1e-12, f"{run['tag']} per-step LR")
            require(record["optimizer_status"] == "applied", f"{run['tag']} optimizer status")
            for key, expected in (("beta1", 0.9), ("beta2", 0.999),
                                  ("eps", 1e-8), ("weight_decay", 0.0)):
                require(abs(record[key] - expected) <= 1e-12, f"{run['tag']} per-step {key}")
            require(record["loss_finite"] is True, f"{run['tag']} non-finite loss")
            require(record.get("grad_nonfinite") in (None, 0), f"{run['tag']} gradients")
            require(record.get("param_nonfinite") in (None, 0), f"{run['tag']} parameters")
        require(report["all_losses_finite"] is True, f"{run['tag']} loss flag")

    # --- sweep and declared selection rule ---------------------------------
    sweep = receipt["sweep"]
    require(sweep["criterion"] == p.CRITERION, "selection criterion changed")
    lrs = [float(value) for value in sweep["lrs"]]
    require(lrs == list(p.SWEEP_LRS), "sweep LR list changed")
    recomputed_sweep = {}
    stable = []
    for lr in lrs:
        recomputed_sweep[str(lr)] = {}
        for arm_id in ("A", "B"):
            claimed = sweep["runs"][str(lr)][arm_id]
            if "error" in claimed:
                require("report" not in claimed, f"sweep {lr:g} {arm_id} failure claim")
                require(claimed["stderr"], f"sweep {lr:g} {arm_id} stderr name")
                stderr = evidence / claimed["stderr"]
                require(stderr.is_file(), f"sweep {lr:g} {arm_id} missing stderr")
                require(q.sha256_file(stderr) == claimed["stderr_sha256"],
                        f"sweep {lr:g} {arm_id} stderr changed")
                recomputed_sweep[str(lr)][arm_id] = {"failed": True}
                continue
            run = load_run(evidence, f"sweep-{lr:g}-{arm_id}", p.ARMS[arm_id],
                           sweep["steps"], 0)
            check_optim(run, lr, sweep["steps"], 0)
            checks = q.check_run(run)
            metrics = p.loss_metrics([r["loss"] for r in run["report"]["records"]])
            recomputed_sweep[str(lr)][arm_id] = {"metrics": metrics, "checks": checks}
            for key in ("first", "final", "min", "max", "early_mean", "late_mean",
                        "reduction", "late_reduction", "slope", "max_spike_ratio", "stable"):
                require(close(claimed["metrics"][key], metrics[key]),
                        f"sweep {lr:g} {arm_id} {key}")
            require(claimed["summary"]["returncode"] == run.get("returncode", 0),
                    f"sweep {lr:g} {arm_id} return code")
            for key in ("learning_rate", "steps", "loss_min", "loss_max",
                        "all_losses_finite", "gradient_norm_source"):
                require(claimed["summary"][key] == run["report"][key],
                        f"sweep {lr:g} {arm_id} summary {key}")
        a_path = evidence / f"sweep-{lr:g}-A.json"
        b_path = evidence / f"sweep-{lr:g}-B.json"
        if a_path.is_file() and b_path.is_file():
            a = [r["loss_bits"] for r in read_json(a_path)["records"]]
            b = [r["loss_bits"] for r in read_json(b_path)["records"]]
            require(a == b, f"sweep {lr:g} scalar/two-pass loss mismatch")
        if ("metrics" in recomputed_sweep[str(lr)]["A"]
                and "metrics" in recomputed_sweep[str(lr)]["B"]
                and recomputed_sweep[str(lr)]["A"]["metrics"]["stable"]
                and recomputed_sweep[str(lr)]["B"]["metrics"]["stable"]):
            stable.append(lr)
    require(stable == [float(x) for x in sweep["stable_lrs"]], "stable LR set changed")
    selected = (
        min(stable, key=lambda lr: (recomputed_sweep[str(lr)]["A"]["metrics"]["late_mean"], lr))
        if stable else None
    )
    require(selected is not None, "no stable LR")
    require(close(sweep["selected_lr"], selected), "selected LR changed")
    require(all(recomputed_sweep[str(lr)]["A"]["metrics"]["stable"]
                and recomputed_sweep[str(lr)]["B"]["metrics"]["stable"] for lr in stable),
            "stable LR does not satisfy criterion")

    # --- additional seeds ---------------------------------------------------
    for seed, entry in receipt["seeds"].items():
        run = load_run(evidence, f"seed-{seed}", p.ARMS["A"],
                       sweep["steps"], 0)
        check_optim(run, selected, sweep["steps"], 0)
        q.check_run(run)
        metrics = p.loss_metrics([r["loss"] for r in run["report"]["records"]])
        require(metrics["stable"], f"seed {seed} is not stable")
        for key in ("first", "final", "early_mean", "late_mean", "stable"):
            require(close(entry["metrics"][key], metrics[key]), f"seed {seed} {key}")

    # --- convergence arms ---------------------------------------------------
    arms = {}
    runs = {}
    for arm_id in ("A", "B", "C", "D"):
        entry = receipt["arms"][arm_id]
        require(entry["config"] == p.ARMS[arm_id], f"arm {arm_id} config changed")
        run = load_run(evidence, f"arm-{arm_id}", p.ARMS[arm_id],
                       entry["summary"]["steps"], 0, profile=True)
        check_optim(run, selected, entry["summary"]["steps"], 0)
        checks = q.check_run(run)
        require(entry["summary"]["returncode"] == 0, f"arm {arm_id} return code")
        require(entry["summary"]["dispatch"] == checks["dispatch"],
                f"arm {arm_id} dispatch claim")
        require(entry["summary"]["wmma"] == checks["wmma"],
                f"arm {arm_id} WMMA claim")
        require(entry["summary"]["report_sha256"] == q.sha256_file(evidence / f"arm-{arm_id}.json"),
                f"arm {arm_id} report hash")
        require(entry["summary"]["stderr_sha256"] == q.sha256_file(evidence / f"arm-{arm_id}.err.txt"),
                f"arm {arm_id} stderr hash")
        metrics = p.loss_metrics([r["loss"] for r in run["report"]["records"]])
        require(metrics["stable"], f"arm {arm_id} did not converge")
        require(close(entry["metrics"]["final"], metrics["final"]), f"arm {arm_id} final metric")
        require(close(entry["metrics"]["early_mean"], metrics["early_mean"]),
                f"arm {arm_id} early metric")
        require(close(entry["metrics"]["late_mean"], metrics["late_mean"]),
                f"arm {arm_id} late metric")
        arms[arm_id] = entry
        runs[arm_id] = run

    def compare(left_id: str, right_id: str, fields: list[str]) -> dict:
        left, right = runs[left_id], runs[right_id]
        result = {"fields": {}, "bitwise_equal": True}
        for field in fields:
            a = [r[field] for r in left["report"]["records"]]
            b = [r[field] for r in right["report"]["records"]]
            result["fields"][field] = {"equal": a == b}
            if a != b:
                result["bitwise_equal"] = False
        if left["report"]["mode"] == "full-finetune":
            result["final_parameter_digest_equal"] = (
                left["report"]["parameter_end_digest"] == right["report"]["parameter_end_digest"]
            )
        return result

    for key, (left, right) in {"A_vs_B": ("A", "B"), "C_vs_D": ("C", "D")}.items():
        result = compare(left, right,
                         ["loss_bits", "param_digest", "update_norm", "update_max_abs"])
        require(result["bitwise_equal"], f"{key} not bitwise equal")
        require(result["final_parameter_digest_equal"], f"{key} final parameters differ")
        require(receipt["comparisons"][key]["bitwise_equal"] is True, f"{key} claim")

    def recompute_drift(left_id: str, right_id: str) -> dict:
        a = [r["loss"] for r in runs[left_id]["report"]["records"]]
        b = [r["loss"] for r in runs[right_id]["report"]["records"]]
        diffs = [x - y for x, y in zip(a, b)]
        return {
            "first_step_abs": abs(diffs[0]),
            "first_step_within_policy": abs(diffs[0]) <= 2e-2,
            "max_abs": max(abs(x) for x in diffs),
            "rms": math.sqrt(sum(x * x for x in diffs) / len(diffs)),
            "final_abs": abs(diffs[-1]),
        }

    for key, (left, right) in {"B_vs_D": ("B", "D"), "A_vs_C": ("A", "C")}.items():
        drift = recompute_drift(left, right)
        require(drift["first_step_within_policy"], f"{key} matched-step drift")
        for field in ("first_step_abs", "max_abs", "rms", "final_abs"):
            require(close(receipt["comparisons"][key][field], drift[field]), f"{key} {field}")

    # --- exact LoRA resume --------------------------------------------------
    for arm_id, entry in receipt["resume"].items():
        total, split = entry["total_steps"], entry["split_step"]
        u = load_run(evidence, f"resume-{arm_id}-u", p.ARMS[arm_id], total, 0,
                     adapter=evidence / "states" / arm_id / "u")
        r = load_run(evidence, f"resume-{arm_id}-r", p.ARMS[arm_id], total - split, split,
                     adapter=evidence / "states" / arm_id / "r", profile=True)
        check_optim(u, selected, total, 0)
        check_optim(r, selected, total - split, split)
        require(r["report"]["parameter_frozen"] is True, f"resume {arm_id} base moved")
        require([x["optimizer_step"] for x in r["report"]["records"]]
                == list(range(split + 1, total + 1)), f"resume {arm_id} steps")
        u_bits = [x["loss_bits"] for x in u["report"]["records"][split:]]
        r_bits = [x["loss_bits"] for x in r["report"]["records"]]
        require(u_bits == r_bits, f"resume {arm_id} losses")
        require(entry["loss_bits_equal"] is True, f"resume {arm_id} loss claim")
        for name in ("adapter_model.safetensors", "adapter_config.json",
                     "hierarchos_adapter_optimizer.safetensors"):
            up = evidence / "states" / arm_id / "u" / name
            rp = evidence / "states" / arm_id / "r" / name
            require(up.is_file() and rp.is_file(), f"resume {arm_id} missing {name}")
            uh, rh = q.sha256_file(up), q.sha256_file(rp)
            require(uh == rh, f"resume {arm_id} {name} differs")
            require(entry["checkpoint_sha256"][name]["uninterrupted"] == uh,
                    f"resume {arm_id} uninterrupted {name} claim")
            require(entry["checkpoint_sha256"][name]["resumed"] == rh,
                    f"resume {arm_id} resumed {name} claim")
        require(entry["checkpoint_bytes_equal"] is True, f"resume {arm_id} bytes claim")
        resumed_dispatch = q.check_run(r)["dispatch"]
        require(entry["dispatch"] == resumed_dispatch, f"resume {arm_id} dispatch")
        require(resumed_dispatch and resumed_dispatch["scalar"] == 0
                and resumed_dispatch["query"] == (total - split) * 2
                and resumed_dispatch["key"] == (total - split) * 2,
                f"resume {arm_id} two-pass dispatch")

    # --- reuse/geometry checks ---------------------------------------------
    for arm_id in ("A", "B"):
        entry = receipt["adverse"]["seq96"][arm_id]
        run = load_run(evidence, f"adverse-seq96-{arm_id}", p.ARMS[arm_id],
                       entry["summary"]["steps"], 0, profile=True)
        check_optim(run, selected, entry["summary"]["steps"], 0)
        q.check_run(run)
        require(run["report"]["seq"] == 96, "seq96 report geometry")
        require(run["report"]["all_losses_finite"] is True, "seq96 non-finite loss")
    repeat = receipt["adverse"]["repeat"]
    r1 = load_run(evidence, "repeat-1", p.ARMS["A"], repeat["steps"], 0)
    r2 = load_run(evidence, "repeat-2", p.ARMS["A"], repeat["steps"], 0)
    check_optim(r1, selected, repeat["steps"], 0)
    check_optim(r2, selected, repeat["steps"], 0)
    require([x["loss_bits"] for x in r1["report"]["records"]]
            == [x["loss_bits"] for x in r2["report"]["records"]], "repeat loss divergence")
    require(r1["report"]["parameter_end_digest"] == r2["report"]["parameter_end_digest"],
            "repeat parameter divergence")

    # --- performance --------------------------------------------------------
    for arm_id, entry in receipt["performance"].items():
        scalar_samples, twopass_samples = [], []
        for path in sorted(evidence.glob(f"perf-{arm_id}-*.json")):
            run = read_json(path)
            measured = [x["step_ms"] for x in run["records"][entry["warmup"]:]]
            if "twopass" in path.stem:
                twopass_samples.extend(measured)
            else:
                scalar_samples.extend(measured)
        require(scalar_samples == entry["samples"]["scalar"], f"perf {arm_id} scalar samples")
        require(twopass_samples == entry["samples"]["twopass"], f"perf {arm_id} twopass samples")
        stats = q.latency_stats(scalar_samples, twopass_samples)
        for key in ("scalar_median_ms", "twopass_median_ms", "scalar_p95_ms",
                    "twopass_p95_ms", "scalar_mean_ms", "twopass_mean_ms",
                    "scalar_std_ms", "twopass_std_ms", "speedup"):
            require(close(entry[key], stats[key]), f"perf {arm_id} {key}")
        require(close(entry["speedup_ci95"][0], stats["speedup_ci95"][0])
                and close(entry["speedup_ci95"][1], stats["speedup_ci95"][1]),
                f"perf {arm_id} confidence interval")

    require(bool(receipt.get("limitations")), "missing limitations")
    return {
        "status": "PASS",
        "revision": revision,
        "selected_lr": selected,
        "stable_lrs": stable,
        "arms": len(receipt["arms"]),
        "resume_arms": sorted(receipt["resume"]),
        "performance_arms": sorted(receipt["performance"]),
        "binaries_verified": binaries_present,
        "fixture_verified": fixture_present,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, default=p.RECEIPT_DIR)
    args = parser.parse_args()
    try:
        summary = verify(args.receipt_dir)
    except FileNotFoundError as error:
        print(f"STABILITY QUALIFICATION: UNQUALIFIED ({error})", file=sys.stderr)
        return 3
    except (QualificationError, KeyError, ValueError, AssertionError) as error:
        print(f"STABILITY QUALIFICATION: FAIL ({error})", file=sys.stderr)
        return 1
    print(
        "STABILITY QUALIFICATION: PASS "
        f"({summary['revision'][:12]}, lr={summary['selected_lr']:g}, "
        f"{summary['arms']} arms, {len(summary['resume_arms'])} resume arms, "
        f"{len(summary['performance_arms'])} performance arms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
