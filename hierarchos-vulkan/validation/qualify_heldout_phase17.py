#!/usr/bin/env python3
"""Phase XVII held-out generalization qualification producer.

Runs the four declared execution arms for 100 optimizer steps on the *training*
partition only, evaluates the validation partition at the declared schedule, and
touches the untouched test partition exactly twice per arm (before training and
at the end). Hyperparameters are fixed in advance; nothing is selected from test.

It also runs the read-only controls (interleaved evaluation versus no
evaluation), the seed investigation, and controlled order-variation replication.

Everything the checker needs is written to a new receipt directory. Phase XV and
Phase XVI receipts are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import qualify_training_phase16 as q

CRATE = q.CRATE
REPO = q.REPO
BIN = CRATE / "target/release/held_out_eval.exe"
MODEL = q.MODEL
FIXTURE = CRATE / "validation/fixtures/phase17"
TRAIN = FIXTURE / "train.jsonl"
VALIDATION = FIXTURE / "validation.jsonl"
TEST = FIXTURE / "test.jsonl"
MANIFEST = FIXTURE / "split_manifest.json"
AUDIT = FIXTURE / "phase16_dataset_audit.json"
PHASE16_DATASET_COPY = FIXTURE / "phase16_dataset.jsonl"
PHASE16_RECEIPT = q.PHASE16_RECEIPT
STABILITY_RECEIPT = (
    CRATE
    / "validation/release_parity/multimodal_training"
    / "20261110-phase16-training-stability/current-qualification.json"
)
RECEIPT_DIR = Path(os.environ.get(
    "HIERARCHOS_RECEIPT_DIR",
    CRATE / "validation/release_parity/multimodal_training" / "20261110-phase17-heldout-generalization",
))
SCHEMA = 1
KIND = "phase17-heldout-generalization"
EXPECTED_PARAMETERS = q.EXPECTED_PARAMETERS
STEPS = 100
BATCH = 8
SEQ = 64
LR = 1.0e-3
SCHEDULE = (0, 10, 20, 40, 60, 80, 100)
ARMS = {
    "A": dict(precision="fp32", wmma="off", backward="scalar", two_pass="0"),
    "B": dict(precision="fp32", wmma="off", backward="twopass", two_pass="1"),
    "C": dict(precision="mixed", wmma="mixed", backward="scalar", two_pass="0"),
    "D": dict(precision="mixed", wmma="mixed", backward="twopass", two_pass="1"),
}
_REUSE_WORK = False

CONTROL_ARMS = ("A", "D")
CONTROL_STEPS = 20
CONTROL_EVAL_AT = 10
SEED_PROBES = (7, 13, 21)
ORDER_OFFSETS = (37, 512)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_windows(path: Path, seq_len: int) -> list[dict]:
    """Mirror of the harness windowing rule, used to re-derive the stream."""
    windows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row["input_ids"]
        if len(tokens) < 2:
            continue
        predict = len(tokens) - 1
        for start in range(0, predict, seq_len):
            count = min(predict - start, seq_len)
            windows.append(
                {
                    "document_id": row["document_id"],
                    "input_ids": tokens[start:start + count],
                    "targets": tokens[start + 1:start + 1 + count],
                    "valid_tokens": count,
                }
            )
    return windows


def stream_digest(windows: list[dict]) -> str:
    digest = hashlib.sha256()
    for window in windows:
        document_id = window["document_id"].encode("utf-8")
        digest.update(len(document_id).to_bytes(8, "little"))
        digest.update(document_id)
        digest.update(window["valid_tokens"].to_bytes(8, "little"))
        for value in window["input_ids"]:
            digest.update(int(value).to_bytes(4, "little"))
        for value in window["targets"]:
            digest.update(int(value).to_bytes(4, "little"))
    return digest.hexdigest()


def run_arm(
    work: Path,
    tag: str,
    arm: dict,
    *,
    steps: int,
    schedule: tuple[int, ...],
    profile: bool,
    seed: int = 7,
    order_offset: int = 0,
    eval_at: int | None = None,
    no_eval: bool = False,
    final_test: bool = True,
) -> dict:
    reports = work / "reports"
    logs = work / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"{tag}.json"
    stdout_path = logs / f"{tag}.out.txt"
    stderr_path = logs / f"{tag}.err.txt"
    if _REUSE_WORK and report_path.is_file() and stderr_path.is_file():
        # Reassembly mode: the GPU evidence already exists in this work
        # directory, so re-read it instead of rerunning the workloads.
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        return {
            "tag": tag,
            "arm": arm,
            "report": report,
            "report_path": report_path,
            "stderr_path": stderr_path,
            "returncode": 0,
            "process_ms": 0.0,
            "blocks": q.parse_profile(stderr_text) if profile else [],
            "seed": seed,
            "order_offset": order_offset,
            "expected_steps": steps,
            "expected_schedule": schedule,
            "eval_at": eval_at,
            "final_test": final_test,
        }
    cmd = [
        str(BIN),
        "--model", str(MODEL),
        "--train", str(TRAIN),
        "--validation", str(VALIDATION),
        "--test", str(TEST),
        "--report", str(report_path),
        "--batch", str(BATCH),
        "--seq", str(SEQ),
        "--steps", str(steps),
        "--lr", "1e-3",
        "--beta1", "0.9",
        "--beta2", "0.999",
        "--eps", "1e-8",
        "--weight-decay", "0",
        "--seed", str(seed),
        "--schedule", ",".join(str(value) for value in schedule),
        "--order-offset", str(order_offset),
    ]
    if eval_at is not None:
        cmd += ["--eval-at", str(eval_at)]
    if no_eval:
        cmd += ["--no-eval"]
    if not final_test:
        cmd += ["--no-final-test"]
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            cmd,
            cwd=CRATE,
            env=q.arm_env(
                {"precision": arm["precision"], "wmma": arm["wmma"],
                 "two_pass": arm["two_pass"]},
                profile,
            ),
            stdout=stdout,
            stderr=stderr,
            timeout=3600,
        )
    process_ms = (time.monotonic() - started) * 1000.0
    if completed.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"{tag}: harness exited {completed.returncode}\n{tail}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    blocks = q.parse_profile(stderr_text) if profile else []
    return {
        "tag": tag,
        "arm": arm,
        "report": report,
        "report_path": report_path,
        "stderr_path": stderr_path,
        "returncode": completed.returncode,
        "process_ms": process_ms,
        "blocks": blocks,
        "seed": seed,
        "order_offset": order_offset,
        "expected_steps": steps,
        "expected_schedule": schedule,
        "eval_at": eval_at,
        "final_test": final_test,
    }


def check_arm(run: dict, manifest: dict) -> dict:
    report = run["report"]
    arm = run["arm"]
    steps = run["expected_steps"]
    records = report["records"]
    tag = run["tag"]
    assert report["mode"] == "training", f"{tag}: mode"
    assert len(records) == steps, f"{tag}: {len(records)} records for {steps} steps"
    assert [record["optimizer_step"] for record in records] == list(range(1, steps + 1)), (
        f"{tag}: optimizer step sequence"
    )
    assert report["parameter_elements"] == EXPECTED_PARAMETERS, f"{tag}: parameters"
    assert report["all_losses_finite"] is True, f"{tag}: non-finite training loss"
    assert report["parameter_nonfinite"] == 0, f"{tag}: non-finite parameters"
    assert report["two_pass_requested"] == (arm["two_pass"] == "1"), f"{tag}: two-pass label"
    assert report["precision_requested"] == ("mixed" if arm["precision"] == "mixed" else "off"), (
        f"{tag}: precision label"
    )
    for record in records:
        assert abs(record["learning_rate"] - LR) <= 1e-12, f"{tag}: LR"
        assert record["loss_finite"] is True, f"{tag}: loss"
        assert record["param_digest"], f"{tag}: digest"

    expected_validation = {f"validation@{step}" for step in run["expected_schedule"]}
    if run.get("eval_at") is not None:
        expected_validation.add(f"validation@{run['eval_at']}")
    tags = [evaluation["tag"] for evaluation in report["evaluations"]]
    actual_validation = {name for name in tags if name.startswith("validation@")}
    assert actual_validation == expected_validation, (
        f"{tag}: validation schedule {sorted(actual_validation)} != {sorted(expected_validation)}"
    )
    if run["final_test"]:
        assert tags.count("test@0") == 1, f"{tag}: missing initial test evaluation"
        assert tags.count("test@final") == 1, f"{tag}: missing final test evaluation"
    else:
        assert not any(name.startswith("test@") for name in tags), f"{tag}: unexpected test"

    for evaluation in report["evaluations"]:
        partition = evaluation["partition"]
        assert evaluation["examples"] > 0, f"{tag}: no examples in {evaluation['tag']}"
        assert evaluation["valid_target_tokens"] == manifest["partitions"][partition][
            "valid_target_tokens"
        ], f"{tag}: {evaluation['tag']} token count"
        assert evaluation["examples"] == manifest["partitions"][partition]["windows"], (
            f"{tag}: {evaluation['tag']} window count"
        )
        assert evaluation["nonfinite_logits"] == 0, f"{tag}: non-finite logits"
        assert evaluation["nonfinite_examples"] == 0, f"{tag}: non-finite examples"
        assert evaluation["parameter_digest_before"] == evaluation["parameter_digest_after"], (
            f"{tag}: evaluation changed parameters"
        )
        assert evaluation["optimizer_state_digest_before"] == evaluation[
            "optimizer_state_digest_after"
        ], f"{tag}: evaluation changed optimizer state"
        assert evaluation["optimizer_step_before"] == evaluation["optimizer_step_after"], (
            f"{tag}: evaluation advanced the optimizer step"
        )
        assert evaluation["dropout_seed_before"] == evaluation["dropout_seed_after"], (
            f"{tag}: evaluation advanced the dropout seed"
        )
        assert evaluation["wmma_weight_grad_delta"] == 0, f"{tag}: evaluation dispatched weight grads"
        assert evaluation["wmma_input_grad_delta"] == 0, f"{tag}: evaluation dispatched input grads"
        if arm["precision"] == "mixed":
            assert evaluation["wmma_forward_delta"] > 0, (
                f"{tag}: mixed evaluation did not use the cooperative forward"
            )
        else:
            assert evaluation["wmma_forward_delta"] == 0, f"{tag}: FP32 evaluation used WMMA"
        # Per-example sums must reproduce the token-weighted total.
        recomputed = sum(
            example["loss_sum"] for example in evaluation["per_example"]
        ) / evaluation["valid_target_tokens"]
        assert math.isclose(recomputed, evaluation["cross_entropy"], rel_tol=1e-12, abs_tol=1e-12), (
            f"{tag}: {evaluation['tag']} token-weighted loss does not recompute"
        )
        assert evaluation["perplexity"] == math.exp(evaluation["cross_entropy"]), (
            f"{tag}: {evaluation['tag']} perplexity"
        )

    dispatch = None
    if run["blocks"]:
        totals = q.profile_totals(run["blocks"])
        scalar = totals.get(q.SCALAR_SHADER, {}).get("dispatches", 0)
        query = totals.get(q.QUERY_SHADER, {}).get("dispatches", 0)
        key = totals.get(q.KEY_SHADER, {}).get("dispatches", 0)
        layers = 2
        if arm["backward"] == "scalar":
            assert scalar == steps * layers and query == 0 and key == 0, (
                f"{tag}: scalar dispatch proof failed scalar={scalar} query={query} key={key}"
            )
        else:
            assert scalar == 0 and query == steps * layers and key == steps * layers, (
                f"{tag}: two-pass dispatch proof failed scalar={scalar} query={query} key={key}"
            )
        dispatch = {"scalar": scalar, "query": query, "key": key}
    return {"dispatch": dispatch}


def compare_runs(left: dict, right: dict) -> dict:
    a, b = left["report"]["records"], right["report"]["records"]
    assert len(a) == len(b), "comparison step mismatch"
    result = {
        "left": left["tag"],
        "right": right["tag"],
        "loss_bits_equal": [record["loss_bits"] for record in a]
        == [record["loss_bits"] for record in b],
        "param_digest_equal": [record["param_digest"] for record in a]
        == [record["param_digest"] for record in b],
        "final_parameter_digest_equal": left["report"]["parameter_end_digest"]
        == right["report"]["parameter_end_digest"],
        "wmma_forward_equal": [record["wmma_forward"] for record in a]
        == [record["wmma_forward"] for record in b],
        "wmma_weight_grad_equal": [record["wmma_weight_grad"] for record in a]
        == [record["wmma_weight_grad"] for record in b],
        "wmma_input_grad_equal": [record["wmma_input_grad"] for record in a]
        == [record["wmma_input_grad"] for record in b],
    }
    result["bitwise_equal"] = (
        result["loss_bits_equal"]
        and result["param_digest_equal"]
        and result["final_parameter_digest_equal"]
    )
    return result


def drift(left: dict, right: dict) -> dict:
    a = [record["loss"] for record in left["report"]["records"]]
    b = [record["loss"] for record in right["report"]["records"]]
    diffs = [x - y for x, y in zip(a, b)]
    va = [
        evaluation["cross_entropy"]
        for evaluation in left["report"]["evaluations"]
        if evaluation["tag"] == "validation@100"
    ]
    vb = [
        evaluation["cross_entropy"]
        for evaluation in right["report"]["evaluations"]
        if evaluation["tag"] == "validation@100"
    ]
    return {
        "left": left["tag"],
        "right": right["tag"],
        "first_step_abs": abs(diffs[0]),
        "first_step_within_policy": abs(diffs[0]) <= 2.0e-2,
        "policy_max_abs": 2.0e-2,
        "max_abs": max(abs(value) for value in diffs),
        "rms": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
        "final_abs": abs(diffs[-1]),
        "validation_final_abs": abs(va[0] - vb[0]) if va and vb else None,
        "steps": len(diffs),
    }


def evaluation_summary(evaluation: dict) -> dict:
    return {
        "tag": evaluation["tag"],
        "partition": evaluation["partition"],
        "after_step": evaluation["after_step"],
        "examples": evaluation["examples"],
        "valid_target_tokens": evaluation["valid_target_tokens"],
        "cross_entropy": evaluation["cross_entropy"],
        "perplexity": evaluation["perplexity"],
        "eval_ms": evaluation["eval_ms"],
        "nonfinite_logits": evaluation["nonfinite_logits"],
        "parameter_digest_before": evaluation["parameter_digest_before"],
        "parameter_digest_after": evaluation["parameter_digest_after"],
        "optimizer_state_digest_before": evaluation["optimizer_state_digest_before"],
        "optimizer_state_digest_after": evaluation["optimizer_state_digest_after"],
        "optimizer_step_before": evaluation["optimizer_step_before"],
        "optimizer_step_after": evaluation["optimizer_step_after"],
        "dropout_seed_before": evaluation["dropout_seed_before"],
        "dropout_seed_after": evaluation["dropout_seed_after"],
        "wmma_forward_delta": evaluation["wmma_forward_delta"],
        "wmma_weight_grad_delta": evaluation["wmma_weight_grad_delta"],
        "wmma_input_grad_delta": evaluation["wmma_input_grad_delta"],
        "per_example_sha256": hashlib.sha256(
            json.dumps(evaluation["per_example"], sort_keys=True).encode()
        ).hexdigest(),
    }


def family_breakdown(evaluation: dict) -> dict:
    totals: dict[str, dict] = {}
    for example in evaluation["per_example"]:
        entry = totals.setdefault(
            example["family"], {"tokens": 0, "loss": 0.0, "examples": 0}
        )
        entry["tokens"] += example["valid_tokens"]
        entry["loss"] += example["loss_sum"]
        entry["examples"] += 1
    return {
        family: {
            "examples": entry["examples"],
            "tokens": entry["tokens"],
            "cross_entropy": entry["loss"] / entry["tokens"],
            "perplexity": math.exp(entry["loss"] / entry["tokens"]),
        }
        for family, entry in totals.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--skip-replication", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--reuse-work", action="store_true")
    args = parser.parse_args()
    global _REUSE_WORK
    _REUSE_WORK = args.reuse_work

    if not BIN.is_file():
        print(f"missing {BIN}; run cargo build --release --bin held_out_eval", file=sys.stderr)
        return 2
    dirty = q.git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        print(f"worktree has tracked changes:\n{dirty}", file=sys.stderr)
        return 2
    revision = q.git("rev-parse", "HEAD")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    work = args.work or (CRATE / "target/heldout-phase17" / stamp)
    work.mkdir(parents=True, exist_ok=True)
    print(f"work={work}")
    print(f"revision={revision}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Re-derive the window streams from the committed partitions and require the
    # harness to report exactly those digests.
    streams = {
        name: stream_digest(derive_windows(FIXTURE / f"{name}.jsonl", SEQ))
        for name in ("train", "validation", "test")
    }

    arms: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for arm_id in ("A", "B", "C", "D"):
        print(f"arm {arm_id}")
        run = run_arm(
            work, f"arm-{arm_id}", ARMS[arm_id],
            steps=STEPS, schedule=SCHEDULE, profile=True,
        )
        checks = check_arm(run, manifest)
        report = run["report"]
        assert report["partition_stream_sha256"] == streams, f"arm {arm_id}: stream digest"
        assert report["validation_windows"] == manifest["partitions"]["validation"]["windows"]
        assert report["test_windows"] == manifest["partitions"]["test"]["windows"]
        arms[arm_id] = {
            "config": ARMS[arm_id],
            "summary": {
                "steps": STEPS,
                "first_training_loss": report["records"][0]["loss"],
                "final_training_loss": report["records"][-1]["loss"],
                "parameter_start_digest": report["parameter_start_digest"],
                "parameter_end_digest": report["parameter_end_digest"],
                "parameter_elements": report["parameter_elements"],
                "total_wall_ms": report["total_wall_ms"],
                "report_sha256": q.sha256_file(run["report_path"]),
                "stderr_sha256": q.sha256_file(run["stderr_path"]),
                "dispatch": checks["dispatch"],
            },
            "evaluations": [evaluation_summary(e) for e in report["evaluations"]],
            "test_family_breakdown": family_breakdown(
                next(e for e in report["evaluations"] if e["tag"] == "test@final")
            ),
        }
        runs[arm_id] = run

    comparisons = {
        "A_vs_B": compare_runs(runs["A"], runs["B"]),
        "C_vs_D": compare_runs(runs["C"], runs["D"]),
        "A_vs_C": drift(runs["A"], runs["C"]),
        "B_vs_D": drift(runs["B"], runs["D"]),
    }
    for key in ("A_vs_B", "C_vs_D"):
        assert comparisons[key]["bitwise_equal"], f"{key} is not bitwise identical"
        assert comparisons[key]["wmma_forward_equal"], f"{key} forward dispatch differs"
        assert comparisons[key]["wmma_weight_grad_equal"], f"{key} weight-grad dispatch differs"
        assert comparisons[key]["wmma_input_grad_equal"], f"{key} input-grad dispatch differs"
    for key in ("A_vs_C", "B_vs_D"):
        assert comparisons[key]["first_step_within_policy"], f"{key} first-step mixed drift"

    # --- seed investigation -------------------------------------------------
    seed_runs = {}
    for seed in SEED_PROBES:
        run = run_arm(
            work, f"seed-{seed}", ARMS["A"], steps=CONTROL_STEPS,
            schedule=(0, CONTROL_STEPS), profile=False, seed=seed, final_test=False,
        )
        check_arm(run, manifest)
        seed_runs[str(seed)] = run
    seed_bits = {
        seed: [record["loss_bits"] for record in run["report"]["records"]]
        for seed, run in seed_runs.items()
    }
    seed_identical = len({tuple(bits) for bits in seed_bits.values()}) == 1
    assert seed_identical, "full-FT trajectories differ across dropout seeds"

    # --- read-only control: interleaved evaluation must not change training --
    controls = {}
    for arm_id in CONTROL_ARMS:
        with_eval = run_arm(
            work, f"control-{arm_id}-eval", ARMS[arm_id], steps=CONTROL_STEPS,
            schedule=(), profile=False, eval_at=CONTROL_EVAL_AT, final_test=False,
        )
        check_arm(with_eval, manifest)
        without_eval = run_arm(
            work, f"control-{arm_id}-noeval", ARMS[arm_id], steps=CONTROL_STEPS,
            schedule=(), profile=False, no_eval=True, final_test=False,
        )
        check_arm(without_eval, manifest)
        comparison = compare_runs(with_eval, without_eval)
        assert comparison["bitwise_equal"], (
            f"arm {arm_id}: interleaved evaluation changed subsequent training"
        )
        controls[arm_id] = {
            "eval_at_step": CONTROL_EVAL_AT,
            "steps": CONTROL_STEPS,
            "interleaved_evaluation_tags": [
                evaluation["tag"] for evaluation in with_eval["report"]["evaluations"]
            ],
            "comparison": comparison,
            "with_eval_parameter_end_digest": with_eval["report"]["parameter_end_digest"],
            "without_eval_parameter_end_digest": without_eval["report"]["parameter_end_digest"],
        }

    # --- controlled order replication --------------------------------------
    replication = {}
    if not args.skip_replication:
        for offset in ORDER_OFFSETS:
            print(f"order replication offset={offset}")
            run = run_arm(
                work, f"order-{offset}", ARMS["A"], steps=STEPS,
                schedule=SCHEDULE, profile=False, order_offset=offset,
            )
            check_arm(run, manifest)
            evaluations = {e["tag"]: e for e in run["report"]["evaluations"]}
            replication[str(offset)] = {
                "order_offset": offset,
                "final_validation_cross_entropy": evaluations["validation@100"]["cross_entropy"],
                "final_validation_perplexity": evaluations["validation@100"]["perplexity"],
                "final_test_cross_entropy": evaluations["test@final"]["cross_entropy"],
                "final_test_perplexity": evaluations["test@final"]["perplexity"],
                "parameter_end_digest": run["report"]["parameter_end_digest"],
            }

    # --- evidence ----------------------------------------------------------
    evidence = work / "evidence"
    if evidence.exists():
        shutil.rmtree(evidence)
    evidence.mkdir(parents=True)
    for path in sorted((work / "reports").glob("*.json")):
        shutil.copy2(path, evidence / path.name)
    for path in sorted((work / "logs").glob("*.err.txt")):
        shutil.copy2(path, evidence / path.name)

    source_files = [
        "src/transformer.rs",
        "src/vulkan.rs",
        "src/wmma_linear.rs",
        "src/wmma.rs",
        "src/lib.rs",
        "src/bin/held_out_eval.rs",
        "Cargo.toml",
        "shaders/transformer_attention_backward.spv",
        "shaders/transformer_attention_backward_query_rdna3.spv",
        "shaders/transformer_attention_backward_key_rdna3.spv",
        "validation/training_shader_manifest.json",
        "validation/make_phase17_fixture.py",
        "validation/audit_phase16_dataset.py",
        "validation/qualify_heldout_phase17.py",
        "validation/check_heldout_phase17.py",
        "validation/test_heldout_phase17.py",
    ]
    bindings = {}
    for relative in source_files:
        path = CRATE / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing bound source {path}")
        bindings[relative] = q.sha256_file(path)

    fixture_bindings = {}
    for name in (
        "corpus.jsonl", "vocab.json", "train.jsonl", "validation.jsonl",
        "test.jsonl", "split_manifest.json", "phase16_dataset.jsonl",
        "phase16_dataset_audit.json",
    ):
        fixture_bindings[name] = q.sha256_file(FIXTURE / name)

    test_metrics = {}
    for arm_id in ("A", "B", "C", "D"):
        evaluations = {e["tag"]: e for e in arms[arm_id]["evaluations"]}
        initial, final = evaluations["test@0"], evaluations["test@final"]
        test_metrics[arm_id] = {
            "initial_cross_entropy": initial["cross_entropy"],
            "final_cross_entropy": final["cross_entropy"],
            "initial_perplexity": initial["perplexity"],
            "final_perplexity": final["perplexity"],
            "relative_improvement": (initial["cross_entropy"] - final["cross_entropy"])
            / initial["cross_entropy"],
            "examples": final["examples"],
            "valid_target_tokens": final["valid_target_tokens"],
            "families": arms[arm_id]["test_family_breakdown"],
        }

    validation_trajectories = {
        arm_id: {
            evaluation["tag"]: evaluation["cross_entropy"]
            for evaluation in arms[arm_id]["evaluations"]
            if evaluation["partition"] == "validation"
        }
        for arm_id in ("A", "B", "C", "D")
    }

    receipt = {
        "schema": SCHEMA,
        "kind": KIND,
        "generated_at": now(),
        "source_revision": revision,
        "source_clean": not dirty,
        "device": runs["A"]["report"]["device"],
        "protocol": {
            "training_partition": "validation/fixtures/phase17/train.jsonl",
            "validation_partition": "validation/fixtures/phase17/validation.jsonl",
            "test_partition": "validation/fixtures/phase17/test.jsonl",
            "steps": STEPS,
            "batch": BATCH,
            "seq": SEQ,
            "learning_rate": LR,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "schedule": list(SCHEDULE),
            "hyperparameter_source": "fixed in advance; no selection on test data",
            "test_access": "two evaluations per arm: before training and at the final step",
        },
        "fixture": {
            "directory": str(FIXTURE.relative_to(CRATE)).replace("\\", "/"),
            "files": fixture_bindings,
            "manifest": json.loads(MANIFEST.read_text(encoding="utf-8")),
            "phase16_audit": json.loads(AUDIT.read_text(encoding="utf-8")),
            "phase16_dataset_sha256": q.sha256_file(PHASE16_DATASET_COPY),
            "phase16_receipt_dataset_sha256": json.loads(
                STABILITY_RECEIPT.read_text(encoding="utf-8")
            )["fixture"]["dataset_sha256"],
            "model": {
                "path": str(MODEL.relative_to(CRATE)).replace("\\", "/"),
                "config_sha256": q.sha256_file(MODEL / "config.json"),
                "weights_sha256": q.sha256_file(MODEL / "model.safetensors"),
                "parameters": EXPECTED_PARAMETERS,
                "config": json.loads((MODEL / "config.json").read_text(encoding="utf-8")),
            },
            "derived_window_streams": streams,
        },
        "binaries": {
            "held_out_eval": q.sha256_file(BIN),
            "trainer": q.sha256_file(q.TRAINER),
        },
        "toolchains": {
            "rustc": q.command_version(["rustc", "--version"]),
            "cargo": q.command_version(["cargo", "--version"]),
            "python": sys.version.split()[0],
        },
        "arms": arms,
        "comparisons": comparisons,
        "test_metrics": test_metrics,
        "validation_trajectories": validation_trajectories,
        "seed_investigation": {
            "seeds": list(SEED_PROBES),
            "steps": CONTROL_STEPS,
            "loss_bits_identical": seed_identical,
            "loss_bits": seed_bits,
            "finding": (
                "dropout is 0.0 for this fixture, dataset order is a deterministic "
                "function of the step index, and full-parameter initialization comes "
                "from the frozen checkpoint, so the dropout seed changes no input to "
                "this experiment; identical trajectories across seeds are therefore "
                "not independent replication"
            ),
            "active_randomness_source": (
                "training-window order offset, used for the order replication below"
            ),
        },
        "order_replication": replication,
        "read_only_controls": controls,
        "limitations": [
            "the fixture is a synthetic three-family grammar corpus; the claim is scoped to it",
            "full-parameter AdamW moment tensors are private; the observable optimizer-state "
            "digest covers step, learning rate, coefficients and dropout seed, while the "
            "interleaved-evaluation control covers the moments behaviourally",
            "the evaluation loss is computed on the host from forward_logits output, not by "
            "the training cross-entropy shader",
            "mixed WMMA remains opt-in and the device scope is this AMD RDNA3 machine",
            "no tokenizer beyond the fixture's committed word-level vocabulary was exercised",
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
    work_receipt.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"receipt={work_receipt}")
    print(f"evidence_files={len(evidence_files)}")
    if args.install:
        if RECEIPT_DIR.exists():
            raise SystemExit(f"refusing to overwrite existing receipt {RECEIPT_DIR}")
        RECEIPT_DIR.mkdir(parents=True)
        shutil.copytree(evidence, RECEIPT_DIR / "evidence")
        shutil.copy2(work_receipt, RECEIPT_DIR / "current-qualification.json")
        print(f"installed={RECEIPT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"held-out qualification failed: {error}", file=sys.stderr)
        raise
