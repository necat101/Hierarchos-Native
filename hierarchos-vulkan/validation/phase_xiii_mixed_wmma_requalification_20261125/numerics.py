#!/usr/bin/env python3
"""Phase XIII numerical and training-state comparison.

Compares the FP32 arm (the oracle) with the Mixed arm on the exact Phase XII
fixture using the existing Mixed precision contract, which bounds a trajectory
or master-tensor difference by `max_abs < 0.02` and `rms < 0.003`
(`validation/qualify_precision_training.py`). Nothing is relaxed here: the same
two numbers are applied, and every value is reported even when it passes.

Sections:
  loss          per-step loss trajectory, off vs mixed
  grad          per-step native LoRA gradient norm
  determinism   mixed rerun must be bit-identical (loss bits + gradient digest)
  resume        split run + resumed run vs one continuous run, bit-identical
  adapter       trained adapter weights and optimizer moments, off vs mixed
  full_masters  full-parameter master weights after 9 steps, off vs mixed
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

RAW = Path(__file__).resolve().parent / "raw"
MAX_ABS = 0.02
RMS = 0.003


def report(name: str) -> dict:
    return json.loads((RAW / name).read_text())


def metric(a, b) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise SystemExit(f"geometry mismatch {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise SystemExit("non-finite comparison")
    d = a - b
    return {
        "values": int(a.size),
        "max_abs": float(np.abs(d).max()),
        "rms": float(np.sqrt(np.mean(d * d))),
        "within_contract": bool(np.abs(d).max() < MAX_ABS and np.sqrt(np.mean(d * d)) < RMS),
    }


def tensors(left: Path, right: Path) -> dict:
    a, b = load_file(str(left)), load_file(str(right))
    if a.keys() != b.keys():
        raise SystemExit(f"tensor inventory mismatch for {left.name}")
    return {name: metric(a[name], b[name]) for name in sorted(a)}


def losses(name: str) -> list[float]:
    return [record["loss"] for record in report(name)["records"]]


def norms(name: str) -> list[float]:
    return [record["grad_norm"] for record in report(name)["records"]]


def bits(name: str) -> list[int]:
    return [record["loss_bits"] for record in report(name)["records"]]


def digests(name: str) -> list[str]:
    return [record["grad_digest"] for record in report(name)["records"]]


REQUIRED = [
    RAW / "tmp/36_lora_off_export-adapter/adapter_model.safetensors",
    RAW / "tmp/37_lora_mixed_export-adapter/adapter_model.safetensors",
    RAW / "tmp/42_full_off_export/model.safetensors",
    RAW / "tmp/43_full_mixed_export/model.safetensors",
    RAW / "tmp/44_full_off_lr1e4_export/model.safetensors",
    RAW / "tmp/45_full_mixed_lr1e4_export/model.safetensors",
]


def main() -> int:
    missing = [str(path.relative_to(RAW)) for path in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit(
            "missing exported tensors: " + ", ".join(missing) +
            "\nrun reproduce.sh steps 9-10 first; the 54 MB master exports are not "
            "kept in the receipt, only their hashes (raw/11_exported_master_hashes.txt) "
            "and the per-tensor comparison this script produced (raw/60_numerics.json)"
        )
    out: dict = {"contract": {"max_abs": MAX_ABS, "rms": RMS}}

    off, mixed = report("30_lora_off_29.json"), report("31_lora_mixed_29a.json")
    out["loss"] = metric(losses("30_lora_off_29.json"), losses("31_lora_mixed_29a.json"))
    out["loss"]["first_off"] = off["records"][0]["loss"]
    out["loss"]["last_off"] = off["records"][-1]["loss"]
    out["loss"]["first_mixed"] = mixed["records"][0]["loss"]
    out["loss"]["last_mixed"] = mixed["records"][-1]["loss"]
    out["loss"]["per_step_delta"] = [
        round(m - o, 9)
        for m, o in zip(losses("31_lora_mixed_29a.json"), losses("30_lora_off_29.json"))
    ]

    grad = metric(norms("30_lora_off_29.json"), norms("31_lora_mixed_29a.json"))
    grad["off_range"] = [min(norms("30_lora_off_29.json")), max(norms("30_lora_off_29.json"))]
    grad["mixed_range"] = [min(norms("31_lora_mixed_29a.json")), max(norms("31_lora_mixed_29a.json"))]
    out["grad"] = grad

    out["determinism"] = {
        "mixed_loss_bits_identical": bits("31_lora_mixed_29a.json") == bits("32_lora_mixed_29b.json"),
        "mixed_grad_digests_identical": digests("31_lora_mixed_29a.json") == digests("32_lora_mixed_29b.json"),
        "mixed_parameter_end_digest": [
            report("31_lora_mixed_29a.json")["parameter_end_digest"],
            report("32_lora_mixed_29b.json")["parameter_end_digest"],
        ],
        "off_loss_bits_identical": bits("30_lora_off_29.json") == bits("36_lora_off_export.json"),
        "off_grad_digests_identical": digests("30_lora_off_29.json") == digests("36_lora_off_export.json"),
        "off_parameter_end_digest": [
            report("30_lora_off_29.json")["parameter_end_digest"],
            report("36_lora_off_export.json")["parameter_end_digest"],
        ],
    }

    split = report("33_lora_mixed_part_a.json")["records"] + report("34_lora_mixed_resume.json")["records"]
    continuous = report("35_lora_mixed_cont12.json")["records"]
    out["resume"] = {
        "steps": len(split),
        "loss_bits_identical": [r["loss_bits"] for r in split] == [r["loss_bits"] for r in continuous],
        "grad_digests_identical": [r["grad_digest"] for r in split] == [r["grad_digest"] for r in continuous],
        "parameter_end_digest_identical": report("35_lora_mixed_cont12.json")["parameter_end_digest"]
        == report("34_lora_mixed_resume.json")["parameter_end_digest"],
        "continuous_parameter_end_digest": report("35_lora_mixed_cont12.json")["parameter_end_digest"],
        "resumed_parameter_end_digest": report("34_lora_mixed_resume.json")["parameter_end_digest"],
    }

    adapter_off = RAW / "tmp/36_lora_off_export-adapter/adapter_model.safetensors"
    adapter_mixed = RAW / "tmp/37_lora_mixed_export-adapter/adapter_model.safetensors"
    state_off = RAW / "tmp/36_lora_off_export-state/hierarchos_adapter_optimizer.safetensors"
    state_mixed = RAW / "tmp/37_lora_mixed_export-state/hierarchos_adapter_optimizer.safetensors"
    out["adapter"] = {
        "weights": tensors(adapter_off, adapter_mixed),
        "moments": tensors(state_off, state_mixed),
        "base_parameters_frozen_identical": report("36_lora_off_export.json")["parameter_end_digest"]
        == report("37_lora_mixed_export.json")["parameter_end_digest"],
        "base_parameter_end_digest": report("37_lora_mixed_export.json")["parameter_end_digest"],
        "base_parameter_frozen": report("37_lora_mixed_export.json")["parameter_frozen"],
    }

    out["full"] = {
        "loss": metric(losses("40_full_off_9.json"), losses("41_full_mixed_9.json")),
        "loss_off": losses("40_full_off_9.json"),
        "loss_mixed": losses("41_full_mixed_9.json"),
        "masters": tensors(
            RAW / "tmp/42_full_off_export/model.safetensors",
            RAW / "tmp/43_full_mixed_export/model.safetensors",
        ),
        "dispatch_off": {
            key: report("40_full_off_9.json")[key]
            for key in ("wmma_forward", "wmma_weight_grad", "wmma_input_grad", "wmma_cooperative", "wmma_shape_fallbacks")
        },
        "dispatch_mixed": {
            key: report("41_full_mixed_9.json")[key]
            for key in ("wmma_forward", "wmma_weight_grad", "wmma_input_grad", "wmma_cooperative", "wmma_shape_fallbacks")
        },
        "steps": len(losses("40_full_off_9.json")),
        "grad_nonfinite_off": report("40_full_off_9.json")["grad_nonfinite_total"],
        "grad_nonfinite_mixed": report("41_full_mixed_9.json")["grad_nonfinite_total"],
    }
    worst_master = max(out["full"]["masters"].items(), key=lambda item: item[1]["max_abs"])
    out["full"]["worst_master_tensor"] = {"name": worst_master[0], **worst_master[1]}

    # Full-parameter at lr 1e-2 does not descend in either arm (the FP32 oracle
    # itself runs 7.13 -> 19.56), so its drift is an unstable-trajectory
    # artefact rather than a precision measurement. lr 1e-4 keeps both arms
    # bounded, which is what makes the master comparison interpretable.
    out["full_lr1e4"] = {
        "loss": metric(losses("44_full_off_lr1e4_export.json"), losses("45_full_mixed_lr1e4_export.json")),
        "loss_off": losses("44_full_off_lr1e4_export.json"),
        "loss_mixed": losses("45_full_mixed_lr1e4_export.json"),
        "masters": tensors(
            RAW / "tmp/44_full_off_lr1e4_export/model.safetensors",
            RAW / "tmp/45_full_mixed_lr1e4_export/model.safetensors",
        ),
        "dispatch_off": {
            key: report("44_full_off_lr1e4_export.json")[key]
            for key in ("wmma_forward", "wmma_weight_grad", "wmma_input_grad", "wmma_cooperative", "wmma_shape_fallbacks")
        },
        "dispatch_mixed": {
            key: report("45_full_mixed_lr1e4_export.json")[key]
            for key in ("wmma_forward", "wmma_weight_grad", "wmma_input_grad", "wmma_cooperative", "wmma_shape_fallbacks")
        },
    }
    worst_lr1e4 = max(out["full_lr1e4"]["masters"].items(), key=lambda item: item[1]["max_abs"])
    out["full_lr1e4"]["worst_master_tensor"] = {"name": worst_lr1e4[0], **worst_lr1e4[1]}
    out["full_lr1e4"]["all_masters_within_contract"] = all(
        row["within_contract"] for row in out["full_lr1e4"]["masters"].values()
    )

    worst_adapter = max(out["adapter"]["weights"].items(), key=lambda item: item[1]["max_abs"])
    out["adapter"]["worst_weight_tensor"] = {"name": worst_adapter[0], **worst_adapter[1]}

    out["finite"] = {
        "off_all_losses_finite": report("30_lora_off_29.json")["all_losses_finite"],
        "mixed_all_losses_finite": report("31_lora_mixed_29a.json")["all_losses_finite"],
        "off_grad_nonfinite": report("30_lora_off_29.json")["grad_nonfinite_total"],
        "mixed_grad_nonfinite": report("31_lora_mixed_29a.json")["grad_nonfinite_total"],
        "off_parameter_nonfinite": report("30_lora_off_29.json")["parameter_nonfinite"],
        "mixed_parameter_nonfinite": report("31_lora_mixed_29a.json")["parameter_nonfinite"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
