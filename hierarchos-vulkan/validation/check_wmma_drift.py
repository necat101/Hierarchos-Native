#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XVIII mixed-WMMA drift receipt.

The checker trusts no aggregate. It re-derives the committed partitions and
window streams, re-adds every per-window evaluation loss, recomputes
perplexity and order statistics, and cross-binds each same-weight forward
comparison to the training run that produced those weights. Structural claims
(parameter counts, dispatch phases, read-only evaluation, mirror conversions)
are checked against the raw per-step records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

import qualify_wmma_drift as q

CRATE = q.CRATE
DEFAULT_RECEIPT = q.RECEIPT_ROOT / "current-qualification.json"
SCHEMA = 1
KIND = "phase18-wmma-drift-investigation"
SHA256 = re.compile(r"[0-9a-f]{64}")
EXPECTED_VALIDATION_STEPS = [0, 1, 2, 5, 10, 20, 40, 60, 80, 100]
PER_STEP_PHASES = 15  # 2 layers x (q,k,v,o,gate,up,down) + lm_head


class QualificationError(Exception):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_windows(rows: list[list[int]], seq: int) -> list[tuple[list[int], list[int], int]]:
    windows = []
    for ids in rows:
        if len(ids) < 2:
            continue
        predict = len(ids) - 1
        for start in range(0, predict, seq):
            count = min(predict - start, seq)
            windows.append((ids[start : start + count], ids[start + 1 : start + 1 + count], count))
    return windows


def stream_digest(windows: list[tuple[list[int], list[int], int]]) -> str:
    digest = hashlib.sha256()
    for ids, targets, valid in windows:
        digest.update(valid.to_bytes(8, "little"))
        for value in ids:
            digest.update(value.to_bytes(4, "little"))
        for value in targets:
            digest.update(value.to_bytes(4, "little"))
    return digest.hexdigest()


def partition_rows(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line)["input_ids"])
    return rows


def loss_bits_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def verify(receipt_path: Path = DEFAULT_RECEIPT, crate: Path = CRATE) -> dict:
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == SCHEMA, "unexpected receipt schema")
    require(receipt.get("kind") == KIND, "unexpected receipt kind")

    revision = str(receipt.get("source_revision", ""))
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "invalid source revision")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=crate.parent,
        capture_output=True,
        text=True,
    )
    require(ancestor.returncode == 0, "source revision is not an ancestor of HEAD")
    require(receipt.get("source_clean") is True, "receipt was produced from a dirty worktree")

    for relative, expected in receipt["source_files"].items():
        path = crate / relative
        require(path.is_file(), f"missing bound source {relative}")
        require(q.sha256_file(path) == expected, f"bound source changed: {relative}")
    require("src/bin/wmma_drift.rs" in receipt["source_files"], "harness binding missing")

    binary_digest = str(receipt["binaries"]["wmma_drift"])
    require(SHA256.fullmatch(binary_digest) is not None, "harness digest is not a SHA-256")
    binary = crate / "target/release/wmma_drift.exe"
    if binary.is_file():
        require(q.sha256_file(binary) == binary_digest, "harness binary hash changed")

    protocol = receipt["protocol"]
    require(protocol["steps"] == q.STEPS, "step count changed")
    require(protocol["batch"] == q.BATCH and protocol["seq"] == q.SEQ, "geometry changed")
    require(protocol["lr"] == q.LR, "optimizer configuration changed: lr")
    require(protocol["beta1"] == q.BETA1, "optimizer configuration changed: beta1")
    require(protocol["beta2"] == q.BETA2, "optimizer configuration changed: beta2")
    require(protocol["eps"] == q.EPS, "optimizer configuration changed: eps")
    require(protocol["weight_decay"] == q.WEIGHT_DECAY, "optimizer configuration changed: weight decay")
    require(protocol["seed"] == q.SEED, "dropout seed changed")
    require(protocol["two_pass_attention_backward"] is True, "two-pass backward was not requested")
    require(list(protocol["orders"]) == list(q.ORDERS), "training-order set changed")
    require(list(protocol["eval_at"]) == EXPECTED_VALIDATION_STEPS, "evaluation schedule changed")
    require(protocol["parameter_snapshot_is_resume_checkpoint"] is False,
            "a weights-only snapshot is presented as a resume checkpoint")

    fixture = receipt["fixture"]
    require(fixture["manifest_sha256"] == q.sha256_file(q.FIXTURE / "split_manifest.json"),
            "split manifest changed")
    streams = {}
    counts = {}
    windows_by_partition = {}
    for name in ("train", "validation", "test"):
        path = q.FIXTURE / f"{name}.jsonl"
        require(fixture["partitions"][name]["sha256"] == q.sha256_file(path),
                f"{name} partition file changed")
        windows = build_windows(partition_rows(path), q.SEQ)
        windows_by_partition[name] = windows
        streams[name] = stream_digest(windows)
        counts[name] = {
            "documents": len(partition_rows(path)),
            "windows": len(windows),
            "valid_targets": sum(window[2] for window in windows),
            "tokens": sum(len(ids) for ids in partition_rows(path)),
        }
        for key, value in counts[name].items():
            require(fixture["partitions"][name][key] == value,
                    f"{name} {key} does not match the committed partition")
    require(fixture["model_config_sha256"] == q.sha256_file(q.MODEL / "config.json"),
            "model config changed")
    require(fixture["model_weights_sha256"] == q.sha256_file(q.MODEL / "model.safetensors"),
            "model weights changed")
    phase17 = crate / "validation/release_parity/multimodal_training/20261110-phase17-heldout-generalization/current-qualification.json"
    require(fixture["phase17_receipt_sha256"] == q.sha256_file(phase17),
            "Phase XVII receipt changed")

    runs = receipt["runs"]
    expected_labels = [f"order-{order}-{mode}" for order in q.ORDERS for mode in q.MODES]
    require([run["label"] for run in runs] == expected_labels, "run set changed or reordered")
    require(len({run["label"] for run in runs}) == len(runs), "duplicate run labels")

    start_digests = {run["parameter_start_digest"] for run in runs}
    require(len(start_digests) == 1, "initial parameter digests differ across runs")
    elements = runs[0]["parameter_elements"]
    require(elements > 0, "parameter count is zero")
    for run in runs:
        require(run["parameter_elements"] == elements, "parameter count differs across runs")
        require(run["optimizer_step"] == q.STEPS, f"{run['label']}: optimizer step is not final")
        require(run["dropout_seed"] == q.SEED, f"{run['label']}: dropout seed changed")
        # The harness rotates the deterministic training order by the declared
        # offset, and binds the rotated stream. Re-derive that rotation so the
        # digest proves the order is the one the run claims.
        training_windows = windows_by_partition["train"]
        shift = run["order_offset"] % len(training_windows)
        rotated = training_windows[shift:] + training_windows[:shift]
        require(run["partition_stream_sha256"]["train"] == stream_digest(rotated),
                f"{run['label']}: train stream digest changed")
        for partition in ("validation", "test"):
            require(run["partition_stream_sha256"][partition] == streams[partition],
                    f"{run['label']}: {partition} stream digest changed")
        records = run["records"]
        require(len(records) == q.STEPS, f"{run['label']}: step count changed")
        mixed = run["mode"] == "mixed"
        for index, record in enumerate(records):
            require(record["global_step"] == index, f"{run['label']}: step index changed")
            require(record["optimizer_step"] == index + 1, f"{run['label']}: optimizer step changed")
            require(record["loss_bits"] == struct.unpack("<I", struct.pack("<f", record["loss"]))[0],
                    f"{run['label']}: step {index} loss bits do not encode its loss")
            require(math.isfinite(record["loss"]), f"{run['label']}: nonfinite training loss")
            require(SHA256.fullmatch(record["param_digest"]) is not None,
                    f"{run['label']}: invalid parameter digest")
            if mixed:
                require(record["wmma_weight_grad"] == PER_STEP_PHASES,
                        f"{run['label']}: mixed arm did not dispatch every weight gradient")
                require(record["wmma_input_grad"] == PER_STEP_PHASES,
                        f"{run['label']}: mixed arm did not dispatch every input gradient")
                require(record["wmma_forward"] == PER_STEP_PHASES,
                        f"{run['label']}: mixed arm did not dispatch every forward projection")
                require(record["wmma_shape_fallbacks"] == 0,
                        f"{run['label']}: mixed arm fell back on a shape")
                require(record["weight_conversions"] > 0,
                        f"{run['label']}: no weight-mirror conversion occurred")
            else:
                require(
                    record["wmma_forward"] == 0
                    and record["wmma_weight_grad"] == 0
                    and record["wmma_input_grad"] == 0,
                    f"{run['label']}: FP32 arm dispatched cooperative kernels",
                )
        require(sum(record["wmma_shape_fallbacks"] for record in records) == 0,
                f"{run['label']}: shape fallbacks recorded")

        evaluations = run["evaluations"]
        validation_steps = [e["after_step"] for e in evaluations if e["partition"] == "validation"]
        require(validation_steps == EXPECTED_VALIDATION_STEPS,
                f"{run['label']}: validation schedule changed")
        test_steps = [e["after_step"] for e in evaluations if e["partition"] == "test"]
        require(test_steps == [0, q.STEPS], f"{run['label']}: test partition usage changed")
        for evaluation in evaluations:
            partition = evaluation["partition"]
            expected = counts[partition]
            require(evaluation["examples"] == expected["windows"],
                    f"{run['label']}: {partition} window count changed")
            require(evaluation["valid_target_tokens"] == expected["valid_targets"],
                    f"{run['label']}: {partition} valid target count changed")
            require(evaluation["stream_sha256"] == streams[partition],
                    f"{run['label']}: {partition} evaluation stream changed")
            require(len(evaluation["loss_sums"]) == expected["windows"],
                    f"{run['label']}: missing per-window losses")
            require(sum(evaluation["window_valid_tokens"]) == expected["valid_targets"],
                    f"{run['label']}: per-window token counts do not cover the partition")
            recomputed = sum(evaluation["loss_sums"]) / sum(evaluation["window_valid_tokens"])
            require(abs(recomputed - evaluation["cross_entropy"]) <= 1e-9,
                    f"{run['label']}: {partition}@{evaluation['after_step']} loss was tampered")
            require(math.isclose(evaluation["perplexity"], math.exp(evaluation["cross_entropy"]),
                                 rel_tol=1e-9),
                    f"{run['label']}: perplexity does not follow its cross-entropy")
            require(evaluation["parameter_digest_before"] == evaluation["parameter_digest_after"],
                    f"{run['label']}: evaluation modified parameters")
            require(evaluation["wmma_weight_grad_delta"] == 0
                    and evaluation["wmma_input_grad_delta"] == 0,
                    f"{run['label']}: evaluation dispatched a gradient kernel")
            if mixed:
                chunks = math.ceil(expected["windows"] / q.BATCH)
                require(evaluation["wmma_forward_delta"] == chunks * PER_STEP_PHASES,
                        f"{run['label']}: evaluation forward dispatch count changed")
            else:
                require(evaluation["wmma_forward_delta"] == 0,
                        f"{run['label']}: FP32 evaluation dispatched cooperative kernels")

    observations = receipt["observations"]
    analysis = receipt["analysis"]
    step_one = analysis["step_1_update"]
    for side in ("off", "mixed"):
        require(sum(entry["count"] for entry in step_one[side].values()) == elements,
                f"step-1 {side} update does not cover every parameter")
        require(all(entry["rms"] >= 0.0 for entry in step_one[side].values()),
                f"step-1 {side} update has a negative RMS")
    divergence = step_one["divergence"]
    require(sum(entry["count"] for entry in divergence.values()) == elements,
            "step-1 update divergence does not cover every parameter")
    for group, entry in divergence.items():
        if entry["cosine"] is not None:
            require(-1.000001 <= entry["cosine"] <= 1.000001,
                    f"step-1 update has an invalid cosine in {group}")
        if entry["sign_flip_fraction"] is not None:
            require(0.0 <= entry["sign_flip_fraction"] <= 1.0,
                    f"step-1 update has an invalid sign-flip fraction in {group}")

    for step in ("10", "100"):
        table = analysis["weight_divergence"][step]
        require(sum(entry["count"] for entry in table.values()) == elements,
                f"step-{step} weight divergence does not cover every parameter")
        require(any(entry["rms"] > 0.0 for entry in table.values()),
                f"step-{step} reports no weight divergence")
    require(observations["first_divergence"]["step"] == 1,
            "the first bitwise loss divergence is not step 1")

    comparisons = observations["forward_comparisons"]
    require(set(comparisons) == {"step-0", "step-10", "step-100"},
            "same-weight forward checkpoints changed")
    windows = counts["validation"]["windows"]
    vocab = int(load_json(q.MODEL / "config.json")["vocab_size"])
    expected_elements = math.ceil(windows / q.BATCH) * q.BATCH * q.SEQ * vocab
    for key, comparison in comparisons.items():
        step = int(key.split("-")[1])
        require(comparison["elements"] == expected_elements,
                f"{key}: logit element count changed")
        require(comparison["nonfinite"] == 0, f"{key}: nonfinite logits in the comparison")
        require(comparison["off_dispatches"] == 0,
                f"{key}: FP32 forward comparison dispatched cooperative kernels")
        chunks = math.ceil(windows / q.BATCH)
        require(comparison["mixed_dispatches"] == chunks * PER_STEP_PHASES,
                f"{key}: mixed forward comparison did not dispatch every projection")
        require(comparison["mixed_fallbacks"] == 0, f"{key}: mixed comparison fell back")
        require(comparison["rms"] <= comparison["max_abs"] + 1e-12,
                f"{key}: RMS exceeds the maximum absolute error")
        require(comparison["elements_over_1e_3"] <= comparison["elements"],
                f"{key}: impossible over-threshold count")
        require(abs(comparison["cross_entropy_delta"]
                    - (comparison["cross_entropy_mixed"] - comparison["cross_entropy_off"])) <= 1e-12,
                f"{key}: cross-entropy delta is inconsistent")
        require(comparison["source_run"] == "order-0-off",
                f"{key}: forward comparison did not use the FP32 checkpoint")
        require(comparison["off_parameter_digest"] == comparison["source_parameter_digest"],
                f"{key}: FP32 comparison weights do not match the exported checkpoint")
        require(comparison["mixed_parameter_digest"] == comparison["source_parameter_digest"],
                f"{key}: the two comparison graphs did not hold identical weights")
        source = next(run for run in runs if run["label"] == comparison["source_run"])
        if step > 0:
            exported = next(
                export for export in source["exports"] if export["step"] == step
            )
            require(exported["parameter_digest"] == comparison["source_parameter_digest"],
                    f"{key}: export digest does not match the comparison")
        evaluation = next(
            evaluation
            for evaluation in source["evaluations"]
            if evaluation["partition"] == "validation" and evaluation["after_step"] == step
        )
        require(abs(comparison["cross_entropy_off"] - evaluation["cross_entropy"]) <= 1e-12,
                f"{key}: forward comparison does not match the trained weights")
        if step == 0:
            mixed_run = next(run for run in runs if run["label"] == "order-0-mixed")
            mixed_evaluation = next(
                evaluation
                for evaluation in mixed_run["evaluations"]
                if evaluation["partition"] == "validation" and evaluation["after_step"] == 0
            )
            require(abs(comparison["cross_entropy_mixed"] - mixed_evaluation["cross_entropy"]) <= 1e-12,
                    f"{key}: mixed initial-weights comparison does not match its run")

    order = receipt["order_analysis"]
    require(order["n_orders"] == len(q.ORDERS), "paired-order count changed")
    recomputed_pairs = {}
    for offset in q.ORDERS:
        off = next(run for run in runs if run["order_offset"] == offset and run["mode"] == "off")
        mixed = next(run for run in runs if run["order_offset"] == offset and run["mode"] == "mixed")
        off_validation = next(
            e["cross_entropy"] for e in off["evaluations"]
            if e["partition"] == "validation" and e["after_step"] == q.STEPS
        )
        mixed_validation = next(
            e["cross_entropy"] for e in mixed["evaluations"]
            if e["partition"] == "validation" and e["after_step"] == q.STEPS
        )
        off_test = next(
            e["cross_entropy"] for e in off["evaluations"]
            if e["partition"] == "test" and e["after_step"] == q.STEPS
        )
        mixed_test = next(
            e["cross_entropy"] for e in mixed["evaluations"]
            if e["partition"] == "test" and e["after_step"] == q.STEPS
        )
        recomputed_pairs[str(offset)] = {
            "validation_off": off_validation,
            "validation_mixed": mixed_validation,
            "validation_gap": mixed_validation - off_validation,
            "test_off": off_test,
            "test_mixed": mixed_test,
            "test_gap": mixed_test - off_test,
        }
    for offset, pair in recomputed_pairs.items():
        recorded = order["pairs"][offset]
        for key, value in pair.items():
            require(abs(recorded[key] - value) <= 1e-12,
                    f"order {offset}: {key} does not follow the run evaluations")
    gaps = [pair["validation_gap"] for pair in recomputed_pairs.values()]
    require(abs(order["validation_gap_mean"] - sum(gaps) / len(gaps)) <= 1e-12,
            "validation gap mean does not follow the pairs")
    require(abs(order["validation_gap_min"] - min(gaps)) <= 1e-12, "validation gap minimum changed")
    require(abs(order["validation_gap_max"] - max(gaps)) <= 1e-12, "validation gap maximum changed")
    require(order["validation_gap_positive"] == sum(1 for gap in gaps if gap > 0.0),
            "positive-gap count does not follow the pairs")
    require(order["confidence_interval"] is None, "a confidence interval is claimed for n=3")

    evidence_dir = receipt_path.parent / "evidence"
    require(evidence_dir.is_dir(), "missing evidence directory")
    for name, expected in receipt["evidence"].items():
        path = evidence_dir / name
        require(path.is_file(), f"missing evidence artifact {name}")
        require(q.sha256_file(path) == expected, f"evidence changed: {name}")
    require(len(receipt["evidence"]) >= 14, "too few evidence artifacts")

    return {
        "runs": len(runs),
        "parameters": elements,
        "forward": {
            key: {
                "max_abs": comparison["max_abs"],
                "rms": comparison["rms"],
                "ce_delta": comparison["cross_entropy_delta"],
            }
            for key, comparison in comparisons.items()
        },
        "validation_gaps": {offset: pair["validation_gap"] for offset, pair in recomputed_pairs.items()},
        "test_gaps": {offset: pair["test_gap"] for offset, pair in recomputed_pairs.items()},
        "step_1_cosine": {
            group: entry["cosine"] for group, entry in divergence.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    try:
        summary = verify(args.receipt)
    except QualificationError as error:
        print(f"WMMA DRIFT QUALIFICATION: FAIL: {error}")
        return 1
    print(
        "WMMA DRIFT QUALIFICATION: PASS "
        f"({summary['runs']} matched runs, {summary['parameters']} parameters, "
        f"max same-weight logit error "
        f"{max(v['max_abs'] for v in summary['forward'].values()):.6g})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
