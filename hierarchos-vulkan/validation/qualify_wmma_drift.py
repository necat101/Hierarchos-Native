#!/usr/bin/env python3
"""Phase XVIII mixed-WMMA drift investigation producer.

Runs matched FP32 and mixed-WMMA full-parameter fine-tuning on the committed
Phase XVII corpus under three genuinely different training-window orders, logs
every step and every scheduled evaluation, and separates:

* same-weight forward error (two processes over the same weights dump logits),
* one-step update divergence (exact parameter snapshots at steps 0 and 1),
* accumulated trajectory divergence (100-step loss and validation curves).

It writes a source-bound receipt and copies the evidence that a checker
re-derives. Raw 67 MB parameter dumps and 75 MB logit tensors stay in
`target/`; the receipt binds their SHA-256 and the compact statistics computed
from them, and states that limitation explicitly.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
MODEL = CRATE / "target/wmma-fixture/model"
FIXTURE = CRATE / "validation/fixtures/phase17"
BINARY = CRATE / "target/release/wmma_drift.exe"
RECEIPT_ROOT = (
    CRATE
    / "validation/release_parity/multimodal_training"
    / "20261110-phase18-wmma-drift"
)

SCHEMA = 1
KIND = "phase18-wmma-drift-investigation"
STEPS = 100
ORDERS = (0, 37, 512)
MODES = ("off", "mixed")
EVAL_AT = "0,1,2,5,10,20,40,60,80,100"
DUMP_AT = (0, 1, 10, 100)
EXPORT_AT = (10, 100)
BATCH = 8
SEQ = 64
LR = 1.0e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1.0e-8
WEIGHT_DECAY = 0.0
SEED = 7
TWO_PASS = "1"

SOURCE_FILES = (
    "src/wmma.rs",
    "src/wmma_linear.rs",
    "src/transformer.rs",
    "src/bin/wmma_drift.rs",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=CRATE.parent, check=True, capture_output=True, text=True
    ).stdout.strip()


def invoke(command: list[str], env: dict[str, str], log: Path, label: str) -> None:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=CRATE, env=env, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    print(f"{label}: exit={result.returncode} {elapsed:.1f}s", flush=True)
    if result.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise SystemExit(f"{label} failed:\n" + "\n".join(tail))


def drift_env(mode: str) -> dict[str, str]:
    env = dict(os.environ)
    env["HIERARCHOS_WMMA_MODE"] = mode
    env["HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS"] = TWO_PASS
    return env


def group_of(name: str) -> str:
    if name.startswith("model.embed_tokens"):
        return "embed"
    if name.startswith("lm_head"):
        return "lm_head"
    if ".self_attn." in name:
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if projection in name:
                return f"attn.{projection}"
        return "attn.other"
    if ".mlp." in name:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            if projection in name:
                return f"mlp.{projection}"
        return "mlp.other"
    if "norm" in name:
        return "norm"
    return "other"


def load_dump(index_path: Path, binary_path: Path) -> dict[str, array.array]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    values = array.array("f")
    values.frombytes(binary_path.read_bytes())
    tensors = {}
    for entry in index:
        start = entry["offset"]
        stop = start + entry["len"]
        tensors[entry["name"]] = values[start:stop]
    return tensors


def group_deltas(before: dict, after: dict, reference: dict | None) -> dict:
    """Per-group statistics for `after - before`, optionally against a reference delta."""
    stats: dict[str, dict] = {}
    for name, values in after.items():
        group = group_of(name)
        entry = stats.setdefault(
            group,
            {"count": 0, "sumsq": 0.0, "max_abs": 0.0, "sign_flips": 0, "sign_compared": 0,
             "dot": 0.0, "reference_sumsq": 0.0, "delta_sumsq": 0.0, "near_zero": 0},
        )
        previous = before[name]
        reference_values = reference[name] if reference is not None else None
        for index, value in enumerate(values):
            delta = value - previous[index]
            entry["count"] += 1
            entry["sumsq"] += delta * delta
            entry["max_abs"] = max(entry["max_abs"], abs(delta))
            if abs(delta) < 1.0e-9:
                entry["near_zero"] += 1
            if reference_values is not None:
                other = reference_values[index] - previous[index]
                entry["reference_sumsq"] += other * other
                entry["delta_sumsq"] += delta * delta
                entry["dot"] += delta * other
                if abs(delta) > 0.0 and abs(other) > 0.0:
                    entry["sign_compared"] += 1
                    if (delta > 0.0) != (other > 0.0):
                        entry["sign_flips"] += 1
    for entry in stats.values():
        count = entry["count"]
        entry["rms"] = math.sqrt(entry["sumsq"] / count) if count else 0.0
        if reference is not None:
            denominator = math.sqrt(entry["delta_sumsq"] * entry["reference_sumsq"])
            entry["cosine"] = entry["dot"] / denominator if denominator > 0.0 else None
            entry["sign_flip_fraction"] = (
                entry["sign_flips"] / entry["sign_compared"] if entry["sign_compared"] else None
            )
            entry["rms_ratio"] = (
                math.sqrt(entry["sumsq"] / entry["reference_sumsq"])
                if entry["reference_sumsq"] > 0.0
                else None
            )
        for key in ("dot", "reference_sumsq", "delta_sumsq"):
            entry.pop(key, None)
    return stats


def weight_divergence(left: dict, right: dict) -> dict:
    stats: dict[str, dict] = {}
    for name, values in left.items():
        group = group_of(name)
        entry = stats.setdefault(group, {"count": 0, "sumsq": 0.0, "max_abs": 0.0, "left_sumsq": 0.0})
        other = right[name]
        for index, value in enumerate(values):
            delta = value - other[index]
            entry["count"] += 1
            entry["sumsq"] += delta * delta
            entry["left_sumsq"] += value * value
            entry["max_abs"] = max(entry["max_abs"], abs(delta))
    for entry in stats.values():
        count = entry["count"]
        entry["rms"] = math.sqrt(entry["sumsq"] / count)
        entry["relative_rms"] = (
            math.sqrt(entry["sumsq"] / entry["left_sumsq"]) if entry["left_sumsq"] > 0.0 else None
        )
        entry.pop("left_sumsq", None)
        entry.pop("sumsq", None)
    return stats


def compare_logits(left_path: Path, right_path: Path, left_meta: dict, right_meta: dict) -> dict:
    left = array.array("f")
    left.frombytes(left_path.read_bytes())
    right = array.array("f")
    right.frombytes(right_path.read_bytes())
    if len(left) != len(right):
        raise SystemExit(f"logit length mismatch: {len(left)} vs {len(right)}")
    maximum = 0.0
    sum_sq = 0.0
    sum_abs = 0.0
    over_1e_3 = 0
    over_1e_2 = 0
    nonfinite = 0
    for a, b in zip(left, right):
        if not (math.isfinite(a) and math.isfinite(b)):
            nonfinite += 1
            continue
        delta = a - b
        absolute = abs(delta)
        maximum = max(maximum, absolute)
        sum_sq += delta * delta
        sum_abs += absolute
        if absolute > 1.0e-3:
            over_1e_3 += 1
        if absolute > 1.0e-2:
            over_1e_2 += 1
    elements = len(left)
    return {
        "elements": elements,
        "max_abs": maximum,
        "rms": math.sqrt(sum_sq / elements),
        "mean_abs": sum_abs / elements,
        "elements_over_1e_3": over_1e_3,
        "elements_over_1e_2": over_1e_2,
        "nonfinite": nonfinite,
        "cross_entropy_off": left_meta["cross_entropy"],
        "cross_entropy_mixed": right_meta["cross_entropy"],
        "cross_entropy_delta": right_meta["cross_entropy"] - left_meta["cross_entropy"],
        "off_dispatches": left_meta["wmma_forward_dispatches"],
        "mixed_dispatches": right_meta["wmma_forward_dispatches"],
        "mixed_fallbacks": right_meta["wmma_shape_fallbacks"],
    }


def first_divergence(records_off: list, records_mixed: list) -> dict:
    for left, right in zip(records_off, records_mixed):
        if left["loss_bits"] != right["loss_bits"]:
            return {
                "step": left["optimizer_step"],
                "loss_off": left["loss"],
                "loss_mixed": right["loss"],
                "abs_delta": abs(left["loss"] - right["loss"]),
            }
    return {"step": None}


def validate_fixture() -> dict:
    manifest = json.loads((FIXTURE / "split_manifest.json").read_text(encoding="utf-8"))
    phases = {}
    for name in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (FIXTURE / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tokens = sum(len(row["input_ids"]) for row in rows)
        windows = 0
        valid = 0
        for row in rows:
            predict = len(row["input_ids"]) - 1
            windows += math.ceil(predict / SEQ)
            valid += predict
        phases[name] = {
            "documents": len(rows),
            "tokens": tokens,
            "windows": windows,
            "valid_targets": valid,
            "sha256": sha256_file(FIXTURE / f"{name}.jsonl"),
        }
    return {"manifest_sha256": sha256_file(FIXTURE / "split_manifest.json"), "partitions": phases}


def analyze(work: Path, runs: list[dict]) -> dict:
    mode_runs = {
        mode: next(run for run in runs if run["label"] == f"order-0-{mode}")
        for mode in MODES
    }
    dumps = {}
    for mode in MODES:
        dumps[mode] = {
            step: load_dump(
                work / f"dump-{mode}/params-{step:04}.json",
                work / f"dump-{mode}/params-{step:04}.bin",
            )
            for step in DUMP_AT
        }
    one_step = {}
    for mode in MODES:
        one_step[mode] = group_deltas(dumps[mode][0], dumps[mode][1], None)
    update_divergence = group_deltas(
        dumps["off"][0], dumps["off"][1], dumps["mixed"][1]
    )
    # The mixed update's own magnitude is measured against its own step-0 state.
    mixed_update = group_deltas(dumps["mixed"][0], dumps["mixed"][1], None)
    for group, entry in update_divergence.items():
        entry["off_rms"] = one_step["off"][group]["rms"]
        entry["mixed_rms"] = mixed_update[group]["rms"]
    trajectory = {}
    for step in (10, 100):
        trajectory[str(step)] = weight_divergence(dumps["off"][step], dumps["mixed"][step])
    step_one = weight_divergence(dumps["off"][1], dumps["mixed"][1])
    forward = {}
    for checkpoint in runs[0]["forward"]:
        forward[checkpoint["checkpoint"]] = checkpoint
    return {
        "dumps": {
            mode: {
                str(step): {
                    "parameter_digest": next(
                        entry["parameter_digest"]
                        for entry in mode_runs[mode]["dumps"]
                        if entry["step"] == step
                    ),
                    "binary_sha256": sha256_file(work / f"dump-{mode}/params-{step:04}.bin"),
                }
                for step in DUMP_AT
            }
            for mode in MODES
        },
        "step_1_update": {
            "off": one_step["off"],
            "mixed": one_step["mixed"],
            "divergence": update_divergence,
        },
        "step_1_weight_divergence": step_one,
        "weight_divergence": trajectory,
        "forward_comparisons": forward,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--reuse-work", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    if not BINARY.is_file():
        raise SystemExit(f"missing harness binary {BINARY}; run cargo build --release --bin wmma_drift")
    if not MODEL.is_dir():
        raise SystemExit(f"missing model fixture {MODEL}")

    revision = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    if dirty and not args.reuse_work:
        raise SystemExit(
            "the producer requires a clean tracked tree so the receipt binds committed source:\n" + dirty
        )

    if args.work is None:
        work = CRATE / "target" / "wmma-drift-phase18" / time.strftime("%Y%m%d-%H%M%S")
    else:
        work = args.work.resolve()
    if work.exists() and not args.reuse_work:
        raise SystemExit(f"work directory already exists: {work}")
    work.mkdir(parents=True, exist_ok=True)
    print(f"work={work}", flush=True)

    runs: list[dict] = []
    if args.skip_gpu:
        for order in ORDERS:
            for mode in MODES:
                label = f"order-{order}-{mode}"
                report = work / f"{label}.json"
                log = work / f"{label}.log"
                if not report.is_file():
                    raise SystemExit(f"--skip-gpu requires an existing report: {report}")
                record = json.loads(report.read_text(encoding="utf-8"))
                runs.append(
                    {
                        "label": label,
                        "order_offset": order,
                        "mode": mode,
                        "report": report,
                        "log": log,
                        "data": record,
                        "sha256": sha256_file(report),
                        "dumps": record["dumps"],
                        "exports": record["exports"],
                        "forward": [],
                    }
                )
    else:
        for order in ORDERS:
            for mode in MODES:
                label = f"order-{order}-{mode}"
                report = work / f"{label}.json"
                log = work / f"{label}.log"
                command = [
                    str(BINARY),
                    "--model", str(MODEL),
                    "--train", str(FIXTURE / "train.jsonl"),
                    "--validation", str(FIXTURE / "validation.jsonl"),
                    "--test", str(FIXTURE / "test.jsonl"),
                    "--report", str(report),
                    "--steps", str(STEPS),
                    "--eval-at", EVAL_AT,
                    "--order-offset", str(order),
                    "--lr", repr(LR),
                    "--beta1", repr(BETA1),
                    "--beta2", repr(BETA2),
                    "--eps", repr(EPS),
                    "--weight-decay", repr(WEIGHT_DECAY),
                    "--seed", str(SEED),
                ]
                if order == 0:
                    command += [
                        "--dump-at", ",".join(str(step) for step in DUMP_AT),
                        "--dump-dir", str(work / f"dump-{mode}"),
                        "--export-at", ",".join(str(step) for step in EXPORT_AT),
                        "--export-root", str(work / "exports"),
                    ]
                invoke(command, drift_env(mode), log, label)
                record = json.loads(report.read_text(encoding="utf-8"))
                runs.append(
                    {
                        "label": label,
                        "order_offset": order,
                        "mode": mode,
                        "report": report,
                        "log": log,
                        "data": record,
                        "sha256": sha256_file(report),
                        "dumps": record["dumps"],
                        "exports": record["exports"],
                        "forward": [],
                    }
                )

        # Same-weight forward comparisons: dump exact logits from the initial
        # checkpoint and from each weights-only export, once per precision mode.
        checkpoint_dirs = {
            "step-0": {
                "directory": MODEL,
                "run": "order-0-off",
                "digest": runs[0]["data"]["parameter_start_digest"],
            }
        }
        for run_record in runs:
            if run_record["order_offset"] != 0:
                continue
            for export in run_record["exports"]:
                key = f"step-{export['step']}"
                if key in checkpoint_dirs:
                    continue
                checkpoint_dirs[key] = {
                    "directory": Path(export["directory"]),
                    "run": run_record["label"],
                    "digest": export["parameter_digest"],
                }
        for key, checkpoint in sorted(checkpoint_dirs.items()):
            directory = checkpoint["directory"]
            summaries = {}
            for mode in MODES:
                binary_path = work / "forward" / f"{key}-{mode}.bin"
                sidecar = binary_path.with_suffix(".json")
                if binary_path.is_file() and sidecar.is_file():
                    summaries[mode] = (binary_path, json.loads(sidecar.read_text(encoding="utf-8")))
                    continue
                command = [
                    str(BINARY),
                    "--model", str(directory),
                    "--validation", str(FIXTURE / "validation.jsonl"),
                    "--dump-logits", str(binary_path),
                    "--report", str(work / "forward" / "unused.json"),
                ]
                invoke(command, drift_env(mode), work / "forward" / f"{key}-{mode}.log", f"forward-{key}-{mode}")
                summaries[mode] = (binary_path, json.loads(sidecar.read_text(encoding="utf-8")))
            comparison = compare_logits(
                summaries["off"][0], summaries["mixed"][0], summaries["off"][1], summaries["mixed"][1]
            )
            comparison["checkpoint"] = key
            comparison["directory"] = str(directory)
            comparison["source_run"] = checkpoint["run"]
            comparison["source_parameter_digest"] = checkpoint["digest"]
            comparison["off_parameter_digest"] = summaries["off"][1]["parameter_digest"]
            comparison["mixed_parameter_digest"] = summaries["mixed"][1]["parameter_digest"]
            comparison["off_logits_sha256"] = sha256_file(summaries["off"][0])
            comparison["mixed_logits_sha256"] = sha256_file(summaries["mixed"][0])
            for run_record in runs:
                if run_record["order_offset"] == 0:
                    run_record["forward"].append(comparison)

    analysis = analyze(work, runs)
    observations = {
        "one_step_parameter_divergence": analysis["step_1_weight_divergence"],
        "step_1_update_divergence": analysis["step_1_update"]["divergence"],
        "first_divergence": first_divergence(
            mode_runs["off"]["data"]["records"], mode_runs["mixed"]["data"]["records"]
        ),
        "validation_trajectories": {
            run["label"]: {
                str(evaluation["after_step"]): evaluation["cross_entropy"]
                for evaluation in run["data"]["evaluations"]
                if evaluation["partition"] == "validation"
            }
            for run in runs
        },
        "test_final": {
            run["label"]: next(
                evaluation["cross_entropy"]
                for evaluation in run["data"]["evaluations"]
                if evaluation["partition"] == "test"
            )
            for run in runs
        },
        "forward_comparisons": analysis["forward_comparisons"],
    }
    pairs = {}
    for order in ORDERS:
        off = next(run for run in runs if run["order_offset"] == order and run["mode"] == "off")
        mixed = next(run for run in runs if run["order_offset"] == order and run["mode"] == "mixed")
        off_validation = [
            evaluation["cross_entropy"]
            for evaluation in off["data"]["evaluations"]
            if evaluation["partition"] == "validation" and evaluation["after_step"] == STEPS
        ][0]
        mixed_validation = [
            evaluation["cross_entropy"]
            for evaluation in mixed["data"]["evaluations"]
            if evaluation["partition"] == "validation" and evaluation["after_step"] == STEPS
        ][0]
        off_test = next(
            evaluation["cross_entropy"]
            for evaluation in off["data"]["evaluations"]
            if evaluation["partition"] == "test"
        )
        mixed_test = next(
            evaluation["cross_entropy"]
            for evaluation in mixed["data"]["evaluations"]
            if evaluation["partition"] == "test"
        )
        pairs[str(order)] = {
            "validation_off": off_validation,
            "validation_mixed": mixed_validation,
            "validation_gap": mixed_validation - off_validation,
            "test_off": off_test,
            "test_mixed": mixed_test,
            "test_gap": mixed_test - off_test,
        }
    gaps = [pair["validation_gap"] for pair in pairs.values()]
    test_gaps = [pair["test_gap"] for pair in pairs.values()]
    order_summary = {
        "pairs": pairs,
        "validation_gap_mean": sum(gaps) / len(gaps),
        "validation_gap_min": min(gaps),
        "validation_gap_max": max(gaps),
        "validation_gap_positive": sum(1 for gap in gaps if gap > 0),
        "test_gap_mean": sum(test_gaps) / len(test_gaps),
        "test_gap_min": min(test_gaps),
        "test_gap_max": max(test_gaps),
        "n_orders": len(ORDERS),
        "confidence_interval": None,
        "confidence_note": "three paired orders: report the full range, no interval is claimed",
    }

    receipt = {
        "schema": SCHEMA,
        "kind": KIND,
        "source_revision": revision,
        "source_clean": not dirty,
        "source_files": {relative: sha256_file(CRATE / relative) for relative in SOURCE_FILES},
        "binaries": {"wmma_drift": sha256_file(BINARY)},
        "device": runs[0]["data"]["device"] if runs else None,
        "fixture": {
            "model": str(MODEL.relative_to(CRATE)),
            "model_config_sha256": sha256_file(MODEL / "config.json"),
            "model_weights_sha256": sha256_file(MODEL / "model.safetensors"),
            "phase17_receipt_sha256": sha256_file(
                CRATE
                / "validation/release_parity/multimodal_training/20261110-phase17-heldout-generalization/current-qualification.json"
            ),
            **validate_fixture(),
        },
        "protocol": {
            "steps": STEPS,
            "batch": BATCH,
            "seq": SEQ,
            "lr": LR,
            "beta1": BETA1,
            "beta2": BETA2,
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "two_pass_attention_backward": True,
            "orders": list(ORDERS),
            "eval_at": [int(step) for step in EVAL_AT.split(",")],
            "dump_steps": list(DUMP_AT),
            "export_steps": list(EXPORT_AT),
            "parameter_snapshot_is_resume_checkpoint": False,
        },
        "runs": [
            {
                "label": run["label"],
                "order_offset": run["order_offset"],
                "mode": run["mode"],
                "report_sha256": run["sha256"],
                "log_sha256": sha256_file(run["log"]),
                "parameter_start_digest": run["data"]["parameter_start_digest"],
                "parameter_end_digest": run["data"]["parameter_end_digest"],
                "parameter_elements": run["data"]["parameter_elements"],
                "optimizer_step": run["data"]["optimizer_step"],
                "dropout_seed": run["data"]["dropout_seed"],
                "partition_stream_sha256": run["data"]["partition_stream_sha256"],
                "wmma_dispatch_total": sum(
                    record["wmma_forward"] for record in run["data"]["records"]
                ),
                "wmma_weight_grad_total": sum(
                    record["wmma_weight_grad"] for record in run["data"]["records"]
                ),
                "wmma_input_grad_total": sum(
                    record["wmma_input_grad"] for record in run["data"]["records"]
                ),
                "shape_fallbacks": sum(
                    record["wmma_shape_fallbacks"] for record in run["data"]["records"]
                ),
                "records": run["data"]["records"],
                "evaluations": run["data"]["evaluations"],
                "exports": run["data"]["exports"],
                "dumps": run["data"]["dumps"],
                "forward": run["forward"],
            }
            for run in runs
        ],
        "observations": observations,
        "order_analysis": order_summary,
        "limitations": [
            "Parameter snapshots are weights-only; they are not optimizer-resume checkpoints.",
            "AdamW first/second moments are private, so the one-step analysis compares the applied update, not the raw gradient tensors.",
            "Layer-wise activation tracing is not instrumented; group statistics are per-parameter projections, not per-layer activations.",
            "Raw 67 MB parameter dumps and 75 MB logit tensors stay in target/; their SHA-256 is bound, and the compact statistics in this receipt are computed from them.",
            "CPU CI re-verifies committed evidence; it does not re-execute RDNA3 workloads.",
        ],
    }

    evidence_dir = work / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for record_run in runs:
        shutil.copy2(record_run["report"], evidence_dir / f"{record_run['label']}.json")
        shutil.copy2(record_run["log"], evidence_dir / f"{record_run['label']}.log")
    for path in (work / "forward").glob("*.json"):
        shutil.copy2(path, evidence_dir / f"forward-{path.name}")
    write_json(work / "analysis.json", analysis)
    shutil.copy2(work / "analysis.json", evidence_dir / "analysis.json")
    receipt["evidence"] = {
        path.name: sha256_file(path)
        for path in sorted(evidence_dir.iterdir())
        if path.is_file()
    }
    write_json(work / "receipt.json", receipt)
    print(f"receipt={work / 'receipt.json'}", flush=True)
    print(
        "order_gaps=" + json.dumps({order: round(pair['validation_gap'], 6) for order, pair in pairs.items()}),
        flush=True,
    )
    print(
        "forward="
        + json.dumps(
            {
                key: {
                    "ce_delta": round(value["cross_entropy_delta"], 9),
                    "max_abs": round(value["max_abs"], 6),
                    "rms": round(value["rms"], 8),
                }
                for key, value in analysis["forward_comparisons"].items()
            }
        ),
        flush=True,
    )

    if args.install:
        if RECEIPT_ROOT.exists():
            raise SystemExit(f"refusing to overwrite existing receipt {RECEIPT_ROOT}")
        RECEIPT_ROOT.mkdir(parents=True)
        shutil.copy2(work / "receipt.json", RECEIPT_ROOT / "current-qualification.json")
        shutil.copytree(evidence_dir, RECEIPT_ROOT / "evidence")
        print(f"installed={RECEIPT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
