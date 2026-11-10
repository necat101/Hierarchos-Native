#!/usr/bin/env python3
"""Phase XIX activation-level drift producer.

Observation-only. Generates one-step weights-only packages for both precision
arms, captures per-layer activations from the production Vulkan graph at
identical weights and inputs, and runs the six-cell first-step cross-evaluation
matrix. No kernel, shader, optimizer rule or tolerance is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
MODEL = CRATE / "target/wmma-fixture/model"
FIXTURE = CRATE / "validation/fixtures/phase17"
PROBE = CRATE / "target/release/activation_probe.exe"
DRIFT = CRATE / "target/release/wmma_drift.exe"
WORK = CRATE / "target/phase19-activation"
EXPORTS = WORK / "exports"
PHASE18_EXPORTS = CRATE / "target/wmma-drift-phase18/20261110-005149/exports"
RECEIPT = CRATE / "validation/release_parity/multimodal_training/20261110-phase19-activation-trace"

OPS = [
    "token_embedding", "pre_attention_norm", "q_proj", "k_proj", "v_proj",
    "attention_context", "attention_output_projection", "attention_residual",
    "pre_mlp_norm", "mlp_gate_up_projection", "mlp_activation", "mlp_output",
    "layer_output", "final_norm", "logits",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd, env, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(os.environ)
    merged.update(env)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run([str(part) for part in cmd], env=merged, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def read_values(path: Path) -> array:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def trace(label: str, model: Path, mode: str, reuse: bool) -> dict:
    out = WORK / "traces" / f"{label}-{mode}"
    report = out / "trace.json"
    if reuse and report.is_file():
        return load_json(report)
    if out.exists():
        shutil.rmtree(out)
    run([PROBE, "--model", model, "--partition", FIXTURE / "validation.jsonl",
         "--out", out, "--batch", "8", "--seq", "64", "--limit", "1", "--trace"],
        {"HIERARCHOS_WMMA_MODE": mode, "HIERARCHOS_ACTIVATION_TRACE": "1"},
        WORK / "logs" / f"trace-{label}-{mode}.log")
    return load_json(report)


def evaluate(label: str, model: Path, mode: str, reuse: bool) -> dict:
    out = WORK / "eval" / f"{label}-{mode}"
    report = out / "eval.json"
    if reuse and report.is_file():
        return load_json(report)
    if out.exists():
        shutil.rmtree(out)
    run([PROBE, "--model", model, "--partition", FIXTURE / "validation.jsonl",
         "--out", out, "--batch", "8", "--seq", "64", "--eval"],
        {"HIERARCHOS_WMMA_MODE": mode},
        WORK / "logs" / f"eval-{label}-{mode}.log")
    return load_json(report)


def compare(off: dict, mixed: dict, off_dir: Path, mixed_dir: Path) -> dict:
    if len(off["tensors"]) != len(mixed["tensors"]):
        raise SystemExit("trace tensor counts differ")
    tensors = []
    previous = None
    for left, right in zip(off["tensors"], mixed["tensors"]):
        if (left["layer"], left["operation"], left["shape"]) != (
                right["layer"], right["operation"], right["shape"]):
            raise SystemExit("trace tensor identity or order differs")
        a = read_values(off_dir / "tensors" / f'{left["layer"]:02}_{left["operation"]}.f32')
        b = read_values(mixed_dir / "tensors" / f'{right["layer"]:02}_{right["operation"]}.f32')
        if len(a) != len(b) or len(a) != left["elements"]:
            raise SystemExit("trace tensor length differs")
        equal = 0
        max_abs = 0.0
        err_sq = 0.0
        ref_sq = 0.0
        nonfinite = 0
        for x, y in zip(a, b):
            if x == y:
                equal += 1
            if not (math.isfinite(x) and math.isfinite(y)):
                nonfinite += 1
                continue
            delta = float(y) - float(x)
            max_abs = max(max_abs, abs(delta))
            err_sq += delta * delta
            ref_sq += float(x) * float(x)
        count = len(a) - nonfinite
        err_rms = math.sqrt(err_sq / count) if count else 0.0
        ref_rms = math.sqrt(ref_sq / count) if count else 0.0
        normalized = err_rms / ref_rms if ref_rms > 0 else 0.0
        amplification = None
        if previous and previous > 1e-12:
            amplification = normalized / previous
        tensors.append({
            "layer": left["layer"],
            "operation": left["operation"],
            "shape": left["shape"],
            "elements": len(a),
            "equal": equal,
            "max_abs": max_abs,
            "err_sumsq": err_sq,
            "ref_sumsq": ref_sq,
            "nonfinite_mismatch": nonfinite,
            "err_rms": err_rms,
            "ref_rms": ref_rms,
            "normalized_rms": normalized,
            "amplification": amplification,
        })
        previous = normalized
    first = next((index for index, tensor in enumerate(tensors)
                  if tensor["equal"] != tensor["elements"]), None)
    after = [tensor for tensor in tensors[first:]] if first is not None else tensors
    peak = max(after, key=lambda tensor: tensor["normalized_rms"]) if after else None
    return {
        "precision_off": off["precision"],
        "precision_mixed": mixed["precision"],
        "bundle_off": off["bundle_sha256"],
        "bundle_mixed": mixed["bundle_sha256"],
        "first_divergence": None if first is None else {
            "index": first,
            "layer": tensors[first]["layer"],
            "operation": tensors[first]["operation"],
            "equal": tensors[first]["equal"],
            "elements": tensors[first]["elements"],
            "max_abs": tensors[first]["max_abs"],
        },
        "max_normalized": None if peak is None else {
            "layer": peak["layer"],
            "operation": peak["operation"],
            "normalized_rms": peak["normalized_rms"],
            "err_rms": peak["err_rms"],
        },
        "tensors": tensors,
    }


def ensure_step1(mode: str, reuse: bool) -> Path:
    target = EXPORTS / f"step-0001-{mode}"
    if reuse and (target / "model.safetensors").is_file():
        return target
    EXPORTS.mkdir(parents=True, exist_ok=True)
    run([DRIFT, "--model", MODEL, "--train", FIXTURE / "train.jsonl",
         "--validation", FIXTURE / "validation.jsonl", "--report",
         WORK / f"step1-{mode}.json", "--steps", "1", "--eval-at", "0",
         "--no-final-test", "--export-at", "1", "--export-root", EXPORTS],
        {"HIERARCHOS_WMMA_MODE": mode}, WORK / "logs" / f"step1-{mode}.log")
    if not (target / "model.safetensors").is_file():
        raise SystemExit(f"missing step-1 export {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if not args.skip_gpu:
        if not PROBE.is_file() or not DRIFT.is_file():
            raise SystemExit("build activation_probe and wmma_drift in release mode first")
    WORK.mkdir(parents=True, exist_ok=True)

    checkpoints = {"initial": MODEL}
    if not args.skip_gpu:
        checkpoints["step1-fp32"] = ensure_step1("off", args.reuse)
        checkpoints["step1-mixed"] = ensure_step1("mixed", args.reuse)
    for name in ("step-0010-off", "step-0100-off"):
        candidate = PHASE18_EXPORTS / name
        if (candidate / "model.safetensors").is_file():
            checkpoints[name.replace("step-00", "step").replace("-off", "")] = candidate
    if args.skip_gpu:
        for stale in ("step1-fp32", "step1-mixed"):
            checkpoints.pop(stale, None)
        checkpoints["step1-fp32"] = EXPORTS / "step-0001-off"
        checkpoints["step1-mixed"] = EXPORTS / "step-0001-mixed"

    traces = {}
    if not args.skip_gpu:
        for name, path in checkpoints.items():
            for mode in ("off", "mixed"):
                traces[(name, mode)] = trace(name, path, mode, args.reuse)
    else:
        for name in checkpoints:
            for mode in ("off", "mixed"):
                report = WORK / "traces" / f"{name}-{mode}" / "trace.json"
                if not report.is_file():
                    raise SystemExit(f"missing retained trace {report}")
                traces[(name, mode)] = load_json(report)

    comparisons = {}
    for name in checkpoints:
        off_dir = WORK / "traces" / f"{name}-off"
        mixed_dir = WORK / "traces" / f"{name}-mixed"
        if not args.skip_gpu:
            if not (off_dir / "tensors").is_dir() or not (mixed_dir / "tensors").is_dir():
                raise SystemExit("missing raw activation captures")
        comparisons[name] = compare(traces[(name, "off")], traces[(name, "mixed")],
                                    off_dir, mixed_dir)

    cells = {}
    if not args.skip_gpu:
        for name in ("initial", "step1-fp32", "step1-mixed"):
            for mode in ("off", "mixed"):
                cells[f"{name}|{mode}"] = evaluate(name, checkpoints[name], mode, args.reuse)
    else:
        for name in ("initial", "step1-fp32", "step1-mixed"):
            for mode in ("off", "mixed"):
                report = WORK / "eval" / f"{name}-{mode}" / "eval.json"
                if not report.is_file():
                    raise SystemExit(f"missing retained evaluation {report}")
                cells[f"{name}|{mode}"] = load_json(report)

    ce = {key: value["cross_entropy"] for key, value in cells.items()}
    observed = ce["step1-mixed|mixed"] - ce["step1-fp32|off"]
    # Path decomposition: hold the FP32-trained weights, change arithmetic;
    # then hold mixed arithmetic, change to the mixed-trained weights.
    arithmetic = ce["step1-fp32|mixed"] - ce["step1-fp32|off"]
    weights = ce["step1-mixed|mixed"] - ce["step1-fp32|mixed"]
    matrix = {
        "cells": ce,
        "arithmetic_effect": arithmetic,
        "weight_effect": weights,
        "observed_gap": observed,
        "residual": observed - arithmetic - weights,
        "arithmetic_at_mixed_weights": ce["step1-mixed|mixed"] - ce["step1-mixed|off"],
        "weight_at_fp32_arithmetic": ce["step1-mixed|off"] - ce["step1-fp32|off"],
        "definition": ("observed = CE(mixed-trained,mixed) - CE(fp32-trained,off); "
                       "arithmetic = CE(fp32-trained,mixed) - CE(fp32-trained,off); "
                       "weights = CE(mixed-trained,mixed) - CE(fp32-trained,mixed); "
                       "residual is the path closure error"),
    }

    trace_runs = {}
    for (name, mode), report in traces.items():
        trace_runs[f"{name}|{mode}"] = {
            "precision": report["precision"],
            "dispatches": report["dispatches"],
            "shape_fallbacks": report["shape_fallbacks"],
            "bundle_sha256": report["bundle_sha256"],
            "tensors": report["tensors"],
        }

    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip() == ""

    evidence = []
    for name, path in checkpoints.items():
        evidence.append({"name": f"weights-{name}", "path": str(path),
                         "sha256": sha256_file(path / "model.safetensors")})
    for (name, mode), report in traces.items():
        evidence.append({"name": f"trace-{name}-{mode}", "path": str(WORK / "traces" / f"{name}-{mode}" / "trace.json"),
                         "sha256": sha256_file(WORK / "traces" / f"{name}-{mode}" / "trace.json")})
    for key, value in cells.items():
        path = WORK / "eval" / key.replace("|", "-") / "eval.json"
        evidence.append({"name": f"eval-{key.replace('|', '-')}", "path": str(path),
                         "sha256": sha256_file(path)})

    receipt = {
        "phase": "XIX",
        "status": "PASS",
        "source_revision": revision,
        "source_clean": clean,
        "device": next(iter(traces.values()))["device"] if traces else "",
        "binaries": {
            "activation_probe": sha256_file(PROBE) if PROBE.is_file() else "",
            "wmma_drift": sha256_file(DRIFT) if DRIFT.is_file() else "",
        },
        "protocol": {
            "activation_trace_env": "HIERARCHOS_ACTIVATION_TRACE=1",
            "mode_env": "HIERARCHOS_WMMA_MODE",
            "batch": 8,
            "seq": 64,
            "windows": 1,
            "partition": "validation",
            "partition_sha256": sha256_file(FIXTURE / "validation.jsonl"),
            "train_sha256": sha256_file(FIXTURE / "train.jsonl"),
            "steps": 1,
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "seed": 7,
            "order_offset": 0,
            "trace_ops": OPS,
            "note": "taps are copies of materialized production buffers; no shader output was altered",
        },
        "checkpoints": {name: {"path": str(path),
                               "weights_sha256": next(item["sha256"] for item in evidence
                                                      if item["name"] == f"weights-{name}")}
                        for name, path in checkpoints.items()},
        "trace_runs": trace_runs,
        "comparisons": comparisons,
        "six_cell": matrix,
        "analysis": {
            "first_divergence": {name: comparison["first_divergence"]
                                 for name, comparison in comparisons.items()},
            "max_normalized": {name: comparison["max_normalized"]
                               for name, comparison in comparisons.items()},
            "gradient_evidence": "not captured; only AdamW updates were compared in Phase XVIII",
        },
        "evidence": evidence,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    if args.install:
        if RECEIPT.exists():
            shutil.rmtree(RECEIPT)
        RECEIPT.mkdir(parents=True)
        shutil.copy2(WORK / "receipt.json", RECEIPT / "current-qualification.json")
        for item in evidence:
            destination = RECEIPT / "evidence" / f"{item['name']}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = Path(item["path"])
            if source.suffix == ".json":
                shutil.copy2(source, destination)
        print(f"installed={RECEIPT}")
    print(f"revision={revision} clean={clean}")
    for name, comparison in comparisons.items():
        first = comparison["first_divergence"]
        peak = comparison["max_normalized"]
        print(f"{name}: first={first['layer']}/{first['operation']} "
              f"peak={peak['layer']}/{peak['operation']} {peak['normalized_rms']:.3e}")
    print("six_cell=" + json.dumps(ce, sort_keys=True))
    print(f"observed={observed:.6f} arithmetic={arithmetic:.6f} weights={weights:.6f} "
          f"residual={matrix['residual']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
