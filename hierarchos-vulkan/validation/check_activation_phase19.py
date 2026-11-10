#!/usr/bin/env python3
"""Fail-closed checker for the Phase XIX activation-trace receipt."""

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
DEFAULT = CRATE / "validation/release_parity/multimodal_training/20261110-phase19-activation-trace"

ORDER = [
    (0, "token_embedding"), (0, "final_norm"), (0, "logits"),
]


def require(condition, message):
    if not condition:
        raise SystemExit(f"ACTIVATION QUALIFICATION: FAIL: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_comparison(name, comparison):
    tensors = comparison["tensors"]
    require(tensors, f"{name}: no activation tensors")
    expected = [(0, "token_embedding")]
    del expected
    seen = []
    first = None
    peak = None
    for index, tensor in enumerate(tensors):
        key = (tensor["layer"], tensor["operation"], tuple(tensor["shape"]))
        require(key not in seen, f"{name}: duplicate tensor {key}")
        seen.append(key)
        elements = tensor["elements"]
        require(isinstance(elements, int) and elements > 0, f"{name}: invalid element count")
        require(0 <= tensor["equal"] <= elements, f"{name}: equal count exceeds elements")
        require(tensor["nonfinite_mismatch"] <= elements, f"{name}: invalid nonfinite count")
        count = elements - tensor["nonfinite_mismatch"]
        require(count > 0, f"{name}: no finite elements")
        err_rms = math.sqrt(tensor["err_sumsq"] / count)
        ref_rms = math.sqrt(tensor["ref_sumsq"] / count)
        require(abs(err_rms - tensor["err_rms"]) <= 1e-12 + 1e-9 * abs(err_rms),
                f"{name}: forged RMS at {key}")
        require(abs(ref_rms - tensor["ref_rms"]) <= 1e-12 + 1e-9 * abs(ref_rms),
                f"{name}: forged reference RMS at {key}")
        expected_norm = err_rms / ref_rms if ref_rms > 0 else 0.0
        require(abs(expected_norm - tensor["normalized_rms"]) <= 1e-12 + 1e-9 * abs(expected_norm),
                f"{name}: normalized RMS does not match")
        require(tensor["max_abs"] >= err_rms - 1e-12, f"{name}: max below RMS at {key}")
        if first is None and tensor["equal"] != elements:
            first = index
        if first is not None:
            if peak is None or tensor["normalized_rms"] > peak["normalized_rms"]:
                peak = {"layer": tensor["layer"], "operation": tensor["operation"],
                        "normalized_rms": tensor["normalized_rms"],
                        "err_rms": tensor["err_rms"]}
    expected_first = None if first is None else {
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
    require(comparison["precision_off"] == "fp32",
            f"{name}: FP32 arm is mislabeled")
    require(comparison["precision_mixed"] == "mixed",
            f"{name}: mixed arm is mislabeled")
    require(comparison["bundle_off"] != comparison["bundle_mixed"],
            f"{name}: both arms report the same activation bundle")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=DEFAULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    receipt = args.receipt
    require((receipt / "current-qualification.json").is_file(), "missing receipt")
    document = load(receipt / "current-qualification.json")
    require(document.get("phase") == "XIX", "wrong phase")
    require(document.get("status") == "PASS", "status is not PASS")
    require(document.get("source_clean") is True, "source tree was not clean")

    revision = document["source_revision"]
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"],
                              cwd=ROOT, capture_output=True)
    require(ancestry.returncode == 0, "source revision is not an ancestor of HEAD")

    for item in document["evidence"]:
        path = receipt / "evidence" / f"{item['name']}.json"
        require(path.is_file(), f"missing evidence file {item['name']}")
        require(sha256(path) == item["sha256"], f"evidence digest changed: {item['name']}")
        require(load(path) is not None, f"unreadable evidence {item['name']}")

    protocol = document["protocol"]
    require(protocol["partition_sha256"] == sha256(FIXTURE / "validation.jsonl"),
            "validation partition changed")
    require(protocol["train_sha256"] == sha256(FIXTURE / "train.jsonl"),
            "training partition changed")
    require(protocol["batch"] == 8 and protocol["seq"] == 64, "unexpected trace geometry")
    require(protocol["steps"] == 1, "step-1 protocol changed")
    require(len(protocol["trace_ops"]) == len(set(protocol["trace_ops"])),
            "trace op list contains duplicates")

    runs = document["trace_runs"]
    require(len(runs) == 8, "expected eight trace runs")
    for key, run in runs.items():
        name, mode = key.split("|")
        require(run["precision"] == ("fp32" if mode == "off" else "mixed"),
                f"{key}: trace mode is mislabeled")
        dispatches = run["dispatches"]
        require(len(dispatches) == 3 and all(isinstance(value, int) and value >= 0
                                             for value in dispatches),
                f"{key}: invalid dispatch counts")
        if mode == "off":
            require(dispatches == [0, 0, 0], f"{key}: FP32 arm dispatched cooperative GEMMs")
        else:
            require(dispatches[0] > 0, f"{key}: missing cooperative dispatches")
            require(run["shape_fallbacks"] == 0, f"{key}: shape fallback in traced batch")
        require(run["tensors"], f"{key}: missing activation captures")
        require(all(tensor["nonfinite"] == 0 for tensor in run["tensors"]),
                f"{key}: nonfinite activation")

    comparisons = document["comparisons"]
    require(set(comparisons) == set(document["checkpoints"]),
            "comparison set does not match checkpoints")
    require(len(comparisons) == 4, "expected four checkpoint comparisons")
    for name, comparison in comparisons.items():
        check_comparison(name, comparison)
        run_mixed = runs[f"{name}|mixed"]
        require(len(comparison["tensors"]) == len(run_mixed["tensors"]),
                f"{name}: comparison tensor count changed")
        for left, right in zip(comparison["tensors"], run_mixed["tensors"]):
            require((left["layer"], left["operation"], left["shape"]) ==
                    (right["layer"], right["operation"], right["shape"]),
                    f"{name}: trace tensor order changed")
            require(left["elements"] == right["elements"],
                    f"{name}: tensor shape changed")

    checkpoints = document["checkpoints"]
    require(checkpoints["step1-fp32"]["weights_sha256"] !=
            checkpoints["step1-mixed"]["weights_sha256"],
            "step-1 arms swapped or identical")
    require(checkpoints["step1-fp32"]["weights_sha256"] !=
            checkpoints["initial"]["weights_sha256"],
            "step-1 FP32 arm did not update its weights")
    require(checkpoints["step1-mixed"]["weights_sha256"] !=
            checkpoints["initial"]["weights_sha256"],
            "step-1 mixed arm did not update its weights")

    matrix = document["six_cell"]
    expected_cells = {f"{weights}|{mode}" for weights in
                      ("initial", "step1-fp32", "step1-mixed") for mode in ("off", "mixed")}
    require(set(matrix["cells"]) == expected_cells, "incomplete cross-evaluation matrix")
    for key, value in matrix["cells"].items():
        require(isinstance(value, (int, float)) and math.isfinite(value),
                f"{key}: invalid cross-entropy")
        require(value > 0.0, f"{key}: nonpositive cross-entropy")
    cells = matrix["cells"]
    observed = cells["step1-mixed|mixed"] - cells["step1-fp32|off"]
    arithmetic = sum(cells[f"{name}|mixed"] - cells[f"{name}|off"]
                     for name in ("initial", "step1-fp32", "step1-mixed")) / 3.0
    weights = sum(cells[f"step1-mixed|{mode}"] - cells[f"initial|{mode}"]
                  for mode in ("off", "mixed")) / 2.0
    interaction = observed - arithmetic - weights
    require(abs(matrix["observed_gap"] - observed) <= 1e-9,
            "observed gap does not close")
    require(abs(matrix["arithmetic_effect"] - arithmetic) <= 1e-9,
            "arithmetic effect does not close")
    require(abs(matrix["weight_effect"] - weights) <= 1e-9,
            "weight effect does not close")
    require(abs(matrix["interaction"] - interaction) <= 1e-9,
            "decomposition does not close")
    require(abs(matrix["residual"]) <= 1e-9, "decomposition residual is nonzero")

    analysis = document["analysis"]
    for name, comparison in comparisons.items():
        require(analysis["first_divergence"][name] == comparison["first_divergence"],
                f"{name}: analysis first divergence disagrees")
        require(analysis["max_normalized"][name] == comparison["max_normalized"],
                f"{name}: analysis peak disagrees")

    if not args.quiet:
        print(f"ACTIVATION QUALIFICATION: PASS ({len(comparisons)} checkpoints, "
              f"{len(runs)} trace runs, first={analysis['first_divergence']['initial']['layer']}/"
              f"{analysis['first_divergence']['initial']['operation']}, "
              f"max normalized={analysis['max_normalized']['initial']['normalized_rms']:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
