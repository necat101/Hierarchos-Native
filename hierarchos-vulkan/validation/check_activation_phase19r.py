#!/usr/bin/env python3
"""Fail-closed checker for the Phase XIX-R instrumented activation receipt.

Recomputes every statistic the receipt derives, re-binds the instrumented
source files, verifies the six-cell decomposition, and checks the step-one
evaluation-lag investigation against the evidence it cites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
FIXTURE = CRATE / "validation/fixtures/phase17"
DEFAULT = CRATE / "validation/release_parity/multimodal_training/20261110-phase19r-activation"
LAYERS = 2
PER_LAYER = [
    "pre_attention_norm", "q_proj", "k_proj", "v_proj", "attention_context",
    "attention_output_projection", "attention_residual", "pre_mlp_norm",
    "mlp_gate_up_projection", "mlp_activation", "mlp_output", "layer_output",
]


def require(condition, message):
    if not condition:
        raise SystemExit(f"ACTIVATION-XIXR QUALIFICATION: FAIL: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_ops():
    ops = [(0, "token_embedding")]
    for layer in range(LAYERS):
        ops.extend((layer, op) for op in PER_LAYER)
    ops.extend([(0, "final_norm"), (0, "logits")])
    return ops


def check_comparison(name, comparison, run_mixed):
    tensors = comparison["tensors"]
    require(len(tensors) == len(expected_ops()) == 27, f"{name}: expected 27 tensors")
    seen = []
    first = None
    peak = None
    for index, (tensor, (layer, op)) in enumerate(zip(tensors, expected_ops())):
        key = (tensor["layer"], tensor["operation"], tuple(tensor["shape"]))
        require(key not in seen, f"{name}: duplicate tensor {key}")
        seen.append(key)
        require((tensor["layer"], tensor["operation"]) == (layer, op),
                f"{name}: tensor order changed at {index}")
        elements = tensor["elements"]
        batch, seq, width = tensor["shape"]
        require(batch == 8 and seq == 64 and width > 0, f"{name}: invalid shape at {index}")
        require(elements == batch * seq * width, f"{name}: element count does not match shape")
        require(0 <= tensor["equal"] <= elements, f"{name}: equal count exceeds elements")
        require(0 <= tensor["nonfinite_mismatch"] <= elements, f"{name}: invalid nonfinite count")
        count = elements - tensor["nonfinite_mismatch"]
        require(count > 0, f"{name}: no finite elements")
        err_rms = math.sqrt(tensor["err_sumsq"] / count)
        ref_rms = math.sqrt(tensor["ref_sumsq"] / count)
        require(abs(err_rms - tensor["err_rms"]) <= 1e-12 + 1e-9 * abs(err_rms),
                f"{name}: forged RMS at {key}")
        require(abs(ref_rms - tensor["ref_rms"]) <= 1e-12 + 1e-9 * abs(ref_rms),
                f"{name}: forged reference RMS at {key}")
        normalized = err_rms / ref_rms if ref_rms > 0 else 0.0
        require(abs(normalized - tensor["normalized_rms"]) <= 1e-12 + 1e-9 * abs(normalized),
                f"{name}: normalized RMS does not match at {key}")
        require(tensor["max_abs"] >= err_rms - 1e-12, f"{name}: max below RMS at {key}")
        if tensor["equal"] != elements:
            require(tensor["nonfinite_mismatch"] == 0,
                    f"{name}: bitwise-equal tensor has nonfinite mismatch at {key}")
        require(len(tensor["mixed_sha256"]) == 64 and len(tensor["off_sha256"]) == 64,
                f"{name}: missing tensor digest at {key}")
        if first is None and tensor["equal"] != elements:
            first = index
        if first is not None and (peak is None or normalized > peak["normalized_rms"]):
            peak = {"layer": tensor["layer"], "operation": tensor["operation"],
                    "normalized_rms": normalized, "err_rms": err_rms}
    require(comparison["first_divergence"] is not None, f"{name}: no divergence recorded")
    expected_first = {
        "index": first,
        "layer": tensors[first]["layer"],
        "operation": tensors[first]["operation"],
        "equal": tensors[first]["equal"],
        "elements": tensors[first]["elements"],
        "max_abs": tensors[first]["max_abs"],
    }
    require(comparison["first_divergence"] == expected_first,
            f"{name}: first divergence does not match the tensor table")
    require(comparison["max_normalized"] == peak,
            f"{name}: largest normalized error does not match the tensor table")
    require(comparison["precision_off"] == "fp32", f"{name}: FP32 arm is mislabeled")
    require(comparison["precision_mixed"] == "mixed", f"{name}: mixed arm is mislabeled")
    require(comparison["bundle_off"] != comparison["bundle_mixed"],
            f"{name}: both arms report the same bundle")
    require(len(run_mixed["tensors"]) == len(tensors), f"{name}: trace tensor count changed")
    for left, right in zip(tensors, run_mixed["tensors"]):
        require((left["layer"], left["operation"], left["shape"]) ==
                (right["layer"], right["operation"], right["shape"]),
                f"{name}: trace tensor order changed")
        require(left["elements"] == right["elements"], f"{name}: tensor shape changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=DEFAULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    receipt = args.receipt
    require((receipt / "current-qualification.json").is_file(), "missing receipt")
    document = load(receipt / "current-qualification.json")
    require(document.get("phase") == "XIX-R", "wrong phase")
    require(document.get("kind") == "phase19r-instrumented-activation", "wrong receipt kind")
    require(document.get("status") == "PASS", "status is not PASS")
    require(document.get("source_clean") is True, "source tree was not clean")

    revision = document["source_revision"]
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"],
                              cwd=ROOT, capture_output=True)
    require(ancestry.returncode == 0, "source revision is not an ancestor of HEAD")

    for relative, expected in document["source_files"].items():
        path = CRATE / relative
        require(path.is_file(), f"missing bound source {relative}")
        require(sha256(path) == expected, f"bound source changed: {relative}")
    require("src/transformer.rs" in document["source_files"], "transformer binding missing")
    require("src/activation_trace.rs" in document["source_files"], "tracer binding missing")
    require("src/bin/activation_probe.rs" in document["source_files"], "probe binding missing")

    for name, expected in document["binaries"].items():
        path = CRATE / "target/release" / f"{name}.exe"
        if path.is_file():
            require(sha256(path) == expected, f"binary hash changed: {name}")

    protocol = document["protocol"]
    require(protocol["batch"] == 8 and protocol["seq"] == 64, "trace geometry changed")
    require(protocol["steps"] == 1, "step protocol changed")
    require(protocol["partition_sha256"] == sha256(FIXTURE / "validation.jsonl"),
            "validation partition changed")
    require(protocol["train_sha256"] == sha256(FIXTURE / "train.jsonl"),
            "training partition changed")
    require(protocol["model_config_sha256"] == sha256(CRATE / "target/wmma-fixture/model/config.json")
            if (CRATE / "target/wmma-fixture/model/config.json").is_file() else True,
            "model config changed")

    runs = document["trace_runs"]
    require(len(runs) == 10, "expected ten trace runs")
    checkpoints = document["checkpoints"]
    require(set(checkpoints) == {"initial", "step1-fp32", "step1-mixed", "step10", "step100"},
            "checkpoint set changed")
    for key, run in runs.items():
        name, mode = key.split("|")
        require(name in checkpoints, f"{key}: unknown checkpoint")
        require(run["precision"] == ("fp32" if mode == "off" else "mixed"),
                f"{key}: trace mode is mislabeled")
        dispatches = run["dispatches"]
        require(len(dispatches) == 3 and all(isinstance(value, int) and value >= 0
                                             for value in dispatches),
                f"{key}: invalid dispatch counts")
        if mode == "off":
            require(dispatches == [0, 0, 0], f"{key}: FP32 arm dispatched cooperative GEMMs")
        else:
            require(dispatches[0] > 0, f"{key}: missing cooperative forward dispatches")
            require(dispatches[1] == 0 and dispatches[2] == 0,
                    f"{key}: evaluation dispatched a gradient kernel")
            require(run["shape_fallbacks"] == 0, f"{key}: shape fallback in traced batch")
        require(len(run["tensors"]) == 27, f"{key}: missing activation captures")
        require(all(tensor["nonfinite"] == 0 for tensor in run["tensors"]),
                f"{key}: nonfinite activation")

    comparisons = document["comparisons"]
    require(set(comparisons) == set(checkpoints), "comparison set does not match checkpoints")
    for name, comparison in comparisons.items():
        check_comparison(name, comparison, runs[f"{name}|mixed"])
        analysis = document["analysis"]["first_divergence"][name]
        require(analysis == comparison["first_divergence"],
                f"{name}: analysis first divergence disagrees")
        require(document["analysis"]["max_normalized"][name] == comparison["max_normalized"],
                f"{name}: analysis peak disagrees")

    weights = {name: value["weights_sha256"] for name, value in checkpoints.items()}
    for left, right in (("initial", "step1-fp32"), ("initial", "step1-mixed"),
                        ("step1-fp32", "step1-mixed"), ("step1-fp32", "step100")):
        require(weights[left] != weights[right], f"checkpoints {left} and {right} are identical")
    require(document["comparisons"]["step1-fp32"]["bundle_mixed"] !=
            document["comparisons"]["step1-mixed"]["bundle_mixed"],
            "step-1 arms produced the same mixed bundle")

    matrix = document["six_cell"]
    expected_cells = {f"{name}|{mode}" for name in ("initial", "step1-fp32", "step1-mixed")
                      for mode in ("off", "mixed")}
    require(set(matrix["cells"]) == expected_cells, "incomplete cross-evaluation matrix")
    for key, value in matrix["cells"].items():
        require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0,
                f"{key}: invalid cross-entropy")
        evidence = load(receipt / "evidence" / f"eval-{key.replace('|', '-')}.json")
        require(abs(evidence["cross_entropy"] - value) <= 1e-12,
                f"{key}: matrix cell does not match its evaluation evidence")
        if key.endswith("|off"):
            require(evidence["dispatches"] == [0, 0, 0], f"{key}: FP32 evaluation dispatched WMMA")
        else:
            require(evidence["dispatches"][0] > 0, f"{key}: mixed evaluation dispatched no WMMA")
        require(evidence["valid_target_tokens"] > 0, f"{key}: no supervised tokens")
    cells = matrix["cells"]
    observed = cells["step1-mixed|mixed"] - cells["step1-fp32|off"]
    arithmetic = sum(cells[f"{name}|mixed"] - cells[f"{name}|off"]
                     for name in ("initial", "step1-fp32", "step1-mixed")) / 3.0
    effect = sum(cells[f"step1-mixed|{mode}"] - cells[f"step1-fp32|{mode}"]
                 for mode in ("off", "mixed")) / 2.0
    interaction = observed - arithmetic - effect
    require(abs(matrix["observed_gap"] - observed) <= 1e-9, "observed gap does not close")
    require(abs(matrix["arithmetic_effect"] - arithmetic) <= 1e-9,
            "arithmetic effect does not close")
    require(abs(matrix["weight_effect"] - effect) <= 1e-9, "weight effect does not close")
    require(abs(matrix["interaction"] - interaction) <= 1e-9,
            "interaction does not follow the decomposition")
    require(abs(matrix["residual"]) <= 1e-9, "decomposition residual is nonzero")

    anomaly = document["anomaly"]
    require(abs(anomaly["post_fix_in_process_step1"] - anomaly["fresh_step1_mixed"]) <= 1e-3,
            "post-fix in-process evaluation still disagrees with the exported weights")
    require(abs(anomaly["pre_fix_in_process_step1"] - anomaly["phase18_claim"]) <= 5e-6,
            "the pre-fix evaluation does not reproduce the Phase XVIII step-one loss")
    require(anomaly["pre_fix_in_process_step1"] - anomaly["fresh_step1_mixed"] > 0.2,
            "the recorded evaluation lag is not large enough to explain the Phase XVIII gap")
    require(abs(anomaly["fresh_step1_mixed"] - anomaly["fresh_step1_mixed_weights_off"]) <= 1e-3,
            "step-one arithmetic comparison does not agree at fixed weights")
    require(abs(anomaly["fresh_step1_mixed"] - anomaly["fresh_step1_fp32arm"]) <= 1e-3,
            "fresh step-one mixed and FP32 arms do not agree")
    require(anomaly["step100_fresh_mixed"] - anomaly["step100_fresh_fp32arm"] > 0.05,
            "the step-100 drift is not reproduced by fresh evaluation")
    require(abs(anomaly["step100_fresh_mixed"] - anomaly["step100_fresh_mixed_weights_off"]) <= 1e-3,
            "step-100 arithmetic comparison does not agree at fixed weights")
    require(abs(anomaly["step100_in_process_mixed"] - anomaly["step100_fresh_mixed"]) <= 5e-3,
            "step-100 in-process evaluation differs from the exported weights")
    require(anomaly["step100_fresh_mixed"] > anomaly["step100_fresh_fp32arm"],
            "the step-100 mixed arm is not worse than FP32")

    evidence_dir = receipt / "evidence"
    require(evidence_dir.is_dir(), "missing evidence directory")
    require(len(document["evidence"]) >= 12, "too few evidence artifacts")
    for item in document["evidence"]:
        if item["name"].startswith("weights-"):
            continue
        path = evidence_dir / f"{item['name']}.json"
        require(path.is_file(), f"missing evidence file {item['name']}")
        require(sha256(path) == item["sha256"], f"evidence digest changed: {item['name']}")
        require(load(path) is not None, f"unreadable evidence {item['name']}")
    anomaly_file = load(evidence_dir / "anomaly.json")
    require(anomaly_file == anomaly, "anomaly evidence disagrees with the receipt")

    limitations = document.get("limitations") or []
    require(any("raw" in text for text in limitations),
            "raw-capture provenance limitation is not recorded")

    if not args.quiet:
        initial = comparisons["initial"]
        print("ACTIVATION-XIXR QUALIFICATION: PASS "
              f"({len(comparisons)} checkpoints, {len(runs)} trace runs, "
              f"first={initial['first_divergence']['layer']}/"
              f"{initial['first_divergence']['operation']}, "
              f"max normalized={initial['max_normalized']['normalized_rms']:.3e}, "
              f"step1 pre={anomaly['pre_fix_in_process_step1']:.6f} "
              f"post={anomaly['post_fix_in_process_step1']:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
