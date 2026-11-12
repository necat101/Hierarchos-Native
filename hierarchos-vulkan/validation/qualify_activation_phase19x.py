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
DRIFT = BIN / "wmma_drift.exe"
WORK = Path(os.environ.get("HIERARCHOS_PHASE19X_ACTIVATION_WORK", CRATE / "target/phase19x/activation"))
EXPORTS = WORK / "exports"
DRIFT_WORK = Path(os.environ.get("HIERARCHOS_DRIFT_WORK", CRATE / "target/phase19x/drift"))
RECEIPT = Path(os.environ.get(
    "HIERARCHOS_RECEIPT_DIR",
    CRATE / "validation/release_parity/multimodal_training/20261112-unified-release/phase19x/activation",
))
FIXTURE = CRATE / "validation/fixtures/phase17"
MODEL = Path(os.environ.get("HIERARCHOS_MODEL", CRATE / "target/wmma-fixture/model"))

CHECKPOINTS = {
    "initial": MODEL,
    "step1-fp32": EXPORTS / "step-0001-off",
    "step1-mixed": EXPORTS / "step-0001-mixed",
    "step10": DRIFT_WORK / "exports/step-0010-off",
    "step100": DRIFT_WORK / "exports/step-0100-off",
}
STEP100_MIXED = DRIFT_WORK / "exports/step-0100-mixed"
CELLS = ("initial", "step1-fp32", "step1-mixed")
SOURCE_FILES = (
    "src/activation_trace.rs",
    "src/bin/activation_probe.rs",
    "src/lib.rs",
    "src/transformer.rs",
    "src/transformer/peft_state.rs",
    "src/lora.rs",
    "src/wmma.rs",
    "src/wmma_linear.rs",
    "validation/qualify_activation_phase19x.py",
    "validation/check_activation_phase19x.py",
    "validation/test_activation_phase19x.py",
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


def ensure_step1(mode: str, reuse: bool) -> Path:
    target = EXPORTS / f"step-0001-{mode}"
    if reuse and (target / "model.safetensors").is_file():
        return target
    EXPORTS.mkdir(parents=True, exist_ok=True)
    run([DRIFT, "--model", MODEL, "--train", FIXTURE / "train.jsonl",
         "--validation", FIXTURE / "validation.jsonl",
         "--report", WORK / f"step1-{mode}.json", "--steps", "1",
         "--eval-at", "0,1", "--no-final-test", "--export-at", "1",
         "--export-root", EXPORTS],
        {"HIERARCHOS_WMMA_MODE": mode}, WORK / "logs" / f"step1-{mode}.log")
    if not (target / "model.safetensors").is_file():
        raise SystemExit(f"missing step-1 export {target}")
    return target


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
    clean = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip() == ""

    traces = {}
    comparisons = {}
    cells = {}
    if not args.skip_gpu:
        for mode in ("off", "mixed"):
            ensure_step1(mode, args.reuse)
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

    mixed_run = load(DRIFT_WORK / "order-0-mixed.json")
    off_run = load(DRIFT_WORK / "order-0-off.json")

    def evaluation(report, step):
        return next(e["cross_entropy"] for e in report["evaluations"]
                    if e["partition"] == "validation" and e["after_step"] == step)

    fresh_step1_mixed = evaluate("fresh-step1-mixed", CHECKPOINTS["step1-mixed"], "mixed", args.reuse)
    fresh_step1_mixed_off = evaluate("fresh-step1-mixed-off", CHECKPOINTS["step1-mixed"], "off", args.reuse)
    fresh_step1_fp32arm = evaluate("fresh-step1-fp32arm", CHECKPOINTS["step1-fp32"], "off", args.reuse)
    fresh100_mixed = evaluate("fresh-step100-mixed", STEP100_MIXED, "mixed", args.reuse)
    fresh100_mixed_off = evaluate("fresh-step100-mixed-off", STEP100_MIXED, "off", args.reuse)
    fresh100_fp32arm = evaluate("fresh-step100-fp32arm", CHECKPOINTS["step100"], "off", args.reuse)

    mirror_epoch = {
        "post_fix_in_process_step1": evaluation(mixed_run, 1),
        "fresh_step1_mixed": fresh_step1_mixed["cross_entropy"],
        "fresh_step1_mixed_weights_off": fresh_step1_mixed_off["cross_entropy"],
        "fresh_step1_fp32arm": fresh_step1_fp32arm["cross_entropy"],
        "step100_in_process_mixed": evaluation(mixed_run, 100),
        "step100_fresh_mixed": fresh100_mixed["cross_entropy"],
        "step100_fresh_mixed_weights_off": fresh100_mixed_off["cross_entropy"],
        "step100_fresh_fp32arm": fresh100_fp32arm["cross_entropy"],
        "historical_pre_fix": {
            "value": 7.06019704940392,
            "source": "Phase XIX-R sealed receipt, not rerun on this source",
        },
        "conclusion": ("the current source publishes the post-step mirror epoch; "
                       "between-step evaluation agrees with fresh evaluation of the "
                       "exported weights, while the step-100 mixed/FP32 drift remains real"),
    }
    if abs(mirror_epoch["post_fix_in_process_step1"] - mirror_epoch["fresh_step1_mixed"]) > 1e-3:
        raise SystemExit("post-step in-process evaluation disagrees with fresh weights")
    if abs(mirror_epoch["fresh_step1_mixed"] - mirror_epoch["fresh_step1_fp32arm"]) > 1e-3:
        raise SystemExit("fresh step-one mixed and FP32 arms disagree")
    if mirror_epoch["step100_fresh_mixed"] - mirror_epoch["step100_fresh_fp32arm"] <= 0.05:
        raise SystemExit("step-100 mixed/FP32 drift is not reproduced")
    if abs(mirror_epoch["step100_in_process_mixed"] - mirror_epoch["step100_fresh_mixed"]) > 5e-3:
        raise SystemExit("step-100 in-process evaluation differs from fresh weights")

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
    fresh_modes = {
        "fresh-step1-mixed": "mixed",
        "fresh-step1-mixed-off": "off",
        "fresh-step1-fp32arm": "off",
        "fresh-step100-mixed": "mixed",
        "fresh-step100-mixed-off": "off",
        "fresh-step100-fp32arm": "off",
    }
    for name, mode, report in (("fresh-step1-mixed", "mixed", fresh_step1_mixed),
                               ("fresh-step1-mixed-off", "off", fresh_step1_mixed_off),
                               ("fresh-step1-fp32arm", "off", fresh_step1_fp32arm),
                               ("fresh-step100-mixed", "mixed", fresh100_mixed),
                               ("fresh-step100-mixed-off", "off", fresh100_mixed_off),
                               ("fresh-step100-fp32arm", "off", fresh100_fp32arm)):
        evidence.append({"name": name,
                         "sha256": sha256_file(WORK / "eval" / f"{name}-{mode}" / "eval.json")})
    write(WORK / "mirror-epoch.json", mirror_epoch)
    evidence.append({"name": "mirror-epoch", "sha256": sha256_file(WORK / "mirror-epoch.json")})
    document = {
        "phase": "XIX-X",
        "kind": "phase19x-instrumented-activation",
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
        "mirror_epoch": mirror_epoch,
        "analysis": {
            "first_divergence": {name: c["first_divergence"] for name, c in comparisons.items()},
            "max_normalized": {name: c["max_normalized"] for name, c in comparisons.items()},
            "gradient_evidence": "not captured; only AdamW updates were compared in Phase XVIII",
            "softmax_evidence": "scores/probabilities are fused in the attention shader and are not separately traced",
        },
        "evidence": evidence,
        "limitations": [
            "raw tensor captures stay under target/phase19x and are not committed",
            "CI verifies compact statistics and evidence hashes, not raw tensors",
            "the pre-fix mirror-lag anomaly is historical and is not claimed as rerun on this source",
        ],
    }
    write(WORK / "receipt.json", document)

    if args.install:
        if RECEIPT.exists():
            raise SystemExit(f"refusing to overwrite existing receipt {RECEIPT}")
        (RECEIPT / "evidence").mkdir(parents=True)
        shutil.copy2(WORK / "receipt.json", RECEIPT / "current-qualification.json")
        for item in evidence:
            source = WORK / "traces" / "missing"
            if item["name"].startswith("weights-"):
                continue
            if item["name"].startswith("trace-"):
                source = WORK / "traces" / item["name"][6:] / "trace.json"
            elif item["name"].startswith("eval-"):
                source = WORK / "eval" / item["name"][5:] / "eval.json"
            elif item["name"].startswith("fresh-"):
                source = WORK / "eval" / f"{item['name']}-{fresh_modes[item['name']]}" / "eval.json"
            elif item["name"] == "mirror-epoch":
                source = WORK / "mirror-epoch.json"
            shutil.copy2(source, RECEIPT / "evidence" / f"{item['name']}.json")
        print(f"installed={RECEIPT}")
    for name, comparison in comparisons.items():
        first = comparison["first_divergence"]
        peak = comparison["max_normalized"]
        print(f"{name}: first={first['layer']}/{first['operation']} "
              f"peak={peak['layer']}/{peak['operation']} {peak['normalized_rms']:.3e}")
    print("six_cell=" + json.dumps({k: round(v, 9) for k, v in ce.items()}, sort_keys=True))
    print(f"observed={observed:.6f} arithmetic={arithmetic:.6f} weights={weights:.6f}")
    print(f"step1 post={mirror_epoch['post_fix_in_process_step1']:.6f} "
          f"fresh={mirror_epoch['fresh_step1_mixed']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
