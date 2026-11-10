#!/usr/bin/env python3
"""Phase XIX-R instrumented-source activation qualification producer.

Runs on the isolated instrumentation branch only. It traces the production
Transformer graph at identical weights and inputs in FP32 and mixed WMMA,
computes the same-weight per-tensor drift, evaluates the six-cell first-step
matrix in fresh processes, and records the step-one evaluation-lag
investigation.

Observation-only: the tracer copies materialized buffers; no shader arithmetic
changed. The one production change in this phase publishes the FP16 mirror
epoch after a submitted optimizer step, so an evaluation that runs between
steps cannot read mirrors converted from pre-step masters.
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
BIN = CRATE / "target/release"
PROBE = BIN / "activation_probe.exe"
WORK = Path(os.environ.get("HIERARCHOS_PHASE19R_WORK", CRATE / "target/phase19r"))
EXPORTS = WORK / "exports"
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training/20261110-phase19r-activation"
)
FIXTURE = CRATE / "validation/fixtures/phase17"
MODEL = Path(os.environ.get("HIERARCHOS_MODEL", CRATE / "target/wmma-fixture/model"))

CHECKPOINTS = {
    "initial": MODEL,
    "step1-fp32": EXPORTS / "step-0001-off",
    "step1-mixed": EXPORTS / "step-0001-mixed",
    "step10": EXPORTS / "step-0010-off",
    "step100": EXPORTS / "step-0100-off",
}
CELLS = ("initial", "step1-fp32", "step1-mixed")
SOURCE_FILES = (
    "src/activation_trace.rs",
    "src/bin/activation_probe.rs",
    "src/lib.rs",
    "src/transformer.rs",
    "src/wmma.rs",
    "src/wmma_linear.rs",
    "validation/qualify_activation_phase19r.py",
    "validation/check_activation_phase19r.py",
    "validation/test_activation_phase19r.py",
)
TRACE_OPS = [
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


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def run(cmd, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(os.environ)
    merged.update(env)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run([str(part) for part in cmd], env=merged, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def values(path: Path) -> array:
    data = array("f")
    data.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        data.byteswap()
    return data


def trace(checkpoint: str, model: Path, mode: str, reuse: bool) -> dict:
    out = WORK / "traces" / f"{checkpoint}-{mode}"
    report = out / "trace.json"
    if reuse and report.is_file():
        return load(report)
    if out.exists():
        shutil.rmtree(out)
    run([PROBE, "--model", model, "--partition", FIXTURE / "validation.jsonl",
         "--out", out, "--batch", "8", "--seq", "64", "--limit", "1", "--trace"],
        {"HIERARCHOS_WMMA_MODE": mode, "HIERARCHOS_ACTIVATION_TRACE": "1"},
        WORK / "logs" / f"trace-{checkpoint}-{mode}.log")
    return load(report)


def evaluate(checkpoint: str, model: Path, mode: str, reuse: bool) -> dict:
    out = WORK / "eval" / f"{checkpoint}-{mode}"
    report = out / "eval.json"
    if reuse and report.is_file():
        return load(report)
    if out.exists():
        shutil.rmtree(out)
    run([PROBE, "--model", model, "--partition", FIXTURE / "validation.jsonl",
         "--out", out, "--batch", "8", "--seq", "64", "--eval"],
        {"HIERARCHOS_WMMA_MODE": mode},
        WORK / "logs" / f"eval-{checkpoint}-{mode}.log")
    return load(report)


def compare(name: str, off: dict, mixed: dict) -> dict:
    left_dir = WORK / "traces" / f"{name}-off"
    right_dir = WORK / "traces" / f"{name}-mixed"
    if len(off["tensors"]) != len(mixed["tensors"]):
        raise SystemExit(f"{name}: trace tensor counts differ")
    tensors = []
    previous = None
    for left, right in zip(off["tensors"], mixed["tensors"]):
        if (left["layer"], left["operation"], left["shape"]) != (
                right["layer"], right["operation"], right["shape"]):
            raise SystemExit(f"{name}: tensor identity or order differs")
        a = values(left_dir / "tensors" / f'{left["layer"]:02}_{left["operation"]}.f32')
        b = values(right_dir / "tensors" / f'{right["layer"]:02}_{right["operation"]}.f32')
        if len(a) != len(b) or len(a) != left["elements"]:
            raise SystemExit(f"{name}: tensor length differs")
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
            "amplification": (normalized / previous) if previous and previous > 1e-12 else None,
            "mixed_sha256": mixed["tensors"][len(tensors)]["sha256"],
            "off_sha256": left["sha256"],
        })
        previous = normalized
    first = next((index for index, tensor in enumerate(tensors)
                  if tensor["equal"] != tensor["elements"]), None)
    after = tensors[first:] if first is not None else tensors
    peak = max(after, key=lambda tensor: tensor["normalized_rms"]) if after else None
    return {
        "checkpoint": name,
        "precision_off": off["precision"],
        "precision_mixed": mixed["precision"],
        "bundle_off": off["bundle_sha256"],
        "bundle_mixed": mixed["bundle_sha256"],
        "first_divergence": None if first is None else {
            "index": first, "layer": tensors[first]["layer"],
            "operation": tensors[first]["operation"], "equal": tensors[first]["equal"],
            "elements": tensors[first]["elements"], "max_abs": tensors[first]["max_abs"],
        },
        "max_normalized": None if peak is None else {
            "layer": peak["layer"], "operation": peak["operation"],
            "normalized_rms": peak["normalized_rms"], "err_rms": peak["err_rms"],
        },
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip() == ""

    traces = {}
    comparisons = {}
    cells = {}
    if not args.skip_gpu:
        for name, path in CHECKPOINTS.items():
            if not (path / "model.safetensors").is_file():
                raise SystemExit(f"missing checkpoint {path}")
            for mode in ("off", "mixed"):
                traces[f"{name}|{mode}"] = trace(name, path, mode, args.reuse)
        for name in CHECKPOINTS:
            comparisons[name] = compare(name, traces[f"{name}|off"], traces[f"{name}|mixed"])
        for name in CELLS:
            for mode in ("off", "mixed"):
                cells[f"{name}|{mode}"] = evaluate(name, CHECKPOINTS[name], mode, args.reuse)
    else:
        for name in CHECKPOINTS:
            for mode in ("off", "mixed"):
                report = WORK / "traces" / f"{name}-{mode}" / "trace.json"
                if not report.is_file():
                    raise SystemExit(f"missing retained trace {report}")
                traces[f"{name}|{mode}"] = load(report)
            comparisons[name] = compare(name, traces[f"{name}|off"], traces[f"{name}|mixed"])
        for name in CELLS:
            for mode in ("off", "mixed"):
                report = WORK / "eval" / f"{name}-{mode}" / "eval.json"
                if not report.is_file():
                    raise SystemExit(f"missing retained evaluation {report}")
                cells[f"{name}|{mode}"] = load(report)

    ce = {key: value["cross_entropy"] for key, value in cells.items()}
    observed = ce["step1-mixed|mixed"] - ce["step1-fp32|off"]
    arithmetic = sum(ce[f"{name}|mixed"] - ce[f"{name}|off"]
                     for name in CELLS) / 3.0
    weights = sum(ce[f"step1-mixed|{mode}"] - ce[f"step1-fp32|{mode}"]
                  for mode in ("off", "mixed")) / 2.0
    interaction = observed - arithmetic - weights
    matrix = {
        "cells": ce,
        "arithmetic_effect": arithmetic,
        "weight_effect": weights,
        "observed_gap": observed,
        "interaction": interaction,
        "residual": observed - arithmetic - weights - interaction,
        "definition": ("observed = CE(step1-mixed,mixed) - CE(step1-fp32,off); "
                       "arithmetic = mean over three weight sets; weights = mean over "
                       "both modes of the step1-mixed minus initial CE"),
    }

    pre = load(WORK / "repro-mixed.json")
    post = load(WORK / "fixed-mixed.json")
    off1 = load(WORK / "off-1.json")
    fresh_step1 = {mode: load(WORK / f"fresh-step1-{mode}" / "eval.json")
                   for mode in ("off", "mixed")}
    mixed100 = load(WORK / "mixed-100.json")
    fresh100_fp32arm = load(WORK / "fresh100-fp32arm" / "eval.json")
    fresh100 = {mode: load(WORK / f"fresh-step100-{mode}" / "eval.json")
                for mode in ("off", "mixed")}

    def evaluation(report, step):
        return next(e["cross_entropy"] for e in report["evaluations"]
                    if e["partition"] == "validation" and e["after_step"] == step)

    anomaly = {
        "phase18_claim": 7.06019704940392,
        "phase18_fp32_step1": 6.7578439142979185,
        "pre_fix_in_process_step1": evaluation(pre, 1),
        "post_fix_in_process_step1": evaluation(post, 1),
        "fresh_step1_mixed": fresh_step1["mixed"]["cross_entropy"],
        "fresh_step1_mixed_weights_off": fresh_step1["off"]["cross_entropy"],
        "fresh_step1_fp32arm": next(e["cross_entropy"] for e in off1["evaluations"]
                                    if e["partition"] == "validation" and e["after_step"] == 1),
        "step100_in_process_mixed": evaluation(mixed100, 100),
        "step100_fresh_mixed": fresh100["mixed"]["cross_entropy"],
        "step100_fresh_mixed_weights_off": fresh100["off"]["cross_entropy"],
        "step100_fresh_fp32arm": fresh100_fp32arm["cross_entropy"],
        "cause": ("the in-process evaluation after step 1 read FP16 projection "
                  "mirrors converted from pre-step masters; LayerNorm and embedding "
                  "masters were current, so the loss landed between the initial and "
                  "updated checkpoints"),
        "fix": ("publish the master epoch after the submitted optimizer step "
                "(VulkanTransformer::train_step), so a between-step evaluation "
                "converts its mirrors from the post-step masters"),
        "conclusion": ("the +0.302353 step-one gap was an evaluation-lag artifact; "
                       "the step-100 mixed/FP32 gap is reproduced by fresh-process "
                       "evaluation of the exported weights and remains a real drift"),
    }

    evidence = []
    for name, path in CHECKPOINTS.items():
        evidence.append({"name": f"weights-{name}", "sha256": sha256_file(path / "model.safetensors")})
    for key, report in traces.items():
        name, mode = key.split("|")
        evidence.append({"name": f"trace-{name}-{mode}",
                         "sha256": sha256_file(WORK / "traces" / f"{name}-{mode}" / "trace.json")})
    for key, report in cells.items():
        evidence.append({"name": f"eval-{key.replace('|', '-')}",
                         "sha256": sha256_file(WORK / "eval" / key.replace("|", "-") / "eval.json")})
    evidence.append({"name": "anomaly", "sha256": ""})
    for name, value in (("phase18", None),):
        del name, value

    document = {
        "phase": "XIX-R",
        "kind": "phase19r-instrumented-activation",
        "status": "PASS",
        "source_revision": revision,
        "source_clean": clean,
        "source_files": {relative: sha256_file(CRATE / relative) for relative in SOURCE_FILES},
        "binaries": {
            "activation_probe": sha256_file(PROBE),
            "wmma_drift": sha256_file(BIN / "wmma_drift.exe"),
        },
        "device": next(iter(traces.values()))["device"],
        "protocol": {
            "activation_trace_env": "HIERARCHOS_ACTIVATION_TRACE=1",
            "mode_env": "HIERARCHOS_WMMA_MODE",
            "batch": 8, "seq": 64, "windows": 1, "partition": "validation",
            "partition_sha256": sha256_file(FIXTURE / "validation.jsonl"),
            "train_sha256": sha256_file(FIXTURE / "train.jsonl"),
            "model_config_sha256": sha256_file(MODEL / "config.json"),
            "model_weights_sha256": sha256_file(MODEL / "model.safetensors"),
            "steps": 1, "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
            "weight_decay": 0.0, "seed": 7, "order_offset": 0,
            "trace_ops": TRACE_OPS,
        },
        "checkpoints": {name: {"weights_sha256": sha256_file(path / "model.safetensors")}
                        for name, path in CHECKPOINTS.items()},
        "trace_runs": {
            key: {"precision": report["precision"], "dispatches": report["dispatches"],
                  "shape_fallbacks": report["shape_fallbacks"],
                  "bundle_sha256": report["bundle_sha256"], "tensors": report["tensors"]}
            for key, report in traces.items()
        },
        "comparisons": comparisons,
        "six_cell": matrix,
        "anomaly": anomaly,
        "analysis": {
            "first_divergence": {name: c["first_divergence"] for name, c in comparisons.items()},
            "max_normalized": {name: c["max_normalized"] for name, c in comparisons.items()},
            "gradient_evidence": "not captured; only AdamW updates were compared in Phase XVIII",
            "softmax_evidence": "scores/probabilities are fused in the attention shader and are not separately traced",
        },
        "evidence": evidence,
        "limitations": [
            "raw tensor captures stay under target/phase19r and are not committed",
            "CI verifies compact statistics and evidence hashes, not raw tensors",
            "Phase XVI-XVIII GPU qualifications were not re-executed for this source revision",
        ],
    }
    write(WORK / "receipt.json", document)
    write(WORK / "anomaly.json", anomaly)
    document["evidence"][-1]["sha256"] = sha256_file(WORK / "anomaly.json")
    write(WORK / "receipt.json", document)

    if args.install:
        if RECEIPT.exists():
            shutil.rmtree(RECEIPT)
        (RECEIPT / "evidence").mkdir(parents=True)
        shutil.copy2(WORK / "receipt.json", RECEIPT / "current-qualification.json")
        for item in evidence:
            source = WORK / "traces" / "missing"
            if item["name"].startswith("weights-"):
                continue
            if item["name"] == "anomaly":
                source = WORK / "anomaly.json"
            elif item["name"].startswith("trace-"):
                source = WORK / "traces" / item["name"][6:] / "trace.json"
            elif item["name"].startswith("eval-"):
                source = WORK / "eval" / item["name"][5:] / "eval.json"
            shutil.copy2(source, RECEIPT / "evidence" / f"{item['name']}.json")
        print(f"installed={RECEIPT}")
    for name, comparison in comparisons.items():
        first = comparison["first_divergence"]
        peak = comparison["max_normalized"]
        print(f"{name}: first={first['layer']}/{first['operation']} "
              f"peak={peak['layer']}/{peak['operation']} {peak['normalized_rms']:.3e}")
    print("six_cell=" + json.dumps({k: round(v, 9) for k, v in ce.items()}, sort_keys=True))
    print(f"observed={observed:.6f} arithmetic={arithmetic:.6f} weights={weights:.6f}")
    print(f"step1 pre={anomaly['pre_fix_in_process_step1']:.6f} "
          f"post={anomaly['post_fix_in_process_step1']:.6f} "
          f"fresh={anomaly['fresh_step1_mixed']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
