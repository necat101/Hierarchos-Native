#!/usr/bin/env python3
"""Phase XIV: is the LoRA adapter drift transient or growing, and what does the
established contract actually assert about it?

Reads the preserved Phase XIII exports (29-step arms) and the Phase XIV
drift-vs-steps arms (6, 12, 29 steps) produced by the unchanged
`training-stability` binary. Reports, per tensor and per step count:
max_abs, rms, the number of elements over 0.02, p99 of |d|, and the relative
RMS drift (|d|_rms / |w|_rms).

Also verifies that the 29-step reproduction reproduces the Phase XIII value and
that the 6/12-step runs share the 29-step loss prefix, so the three points are
one trajectory rather than three different experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

HERE = Path(__file__).resolve().parent
STEPS = HERE / "raw" / "driftsteps"
XIII = HERE.parent / "phase_xiii_mixed_wmma_requalification_20261125" / "raw"

MAX_ABS = 0.02
RMS = 0.003


def compare(left: Path, right: Path) -> dict:
    a, b = load_file(str(left)), load_file(str(right))
    if a.keys() != b.keys():
        raise SystemExit(f"inventory mismatch {left} vs {right}")
    out = {}
    for name in sorted(a):
        x = np.asarray(a[name], dtype=np.float64)
        y = np.asarray(b[name], dtype=np.float64)
        d = y - x
        ad = np.abs(d)
        flat = int(np.argmax(ad))
        index = tuple(int(i) for i in np.unravel_index(flat, ad.shape))
        out[name] = {
            "shape": list(x.shape),
            "values": int(x.size),
            "max_abs": float(ad.max()),
            "rms": float(np.sqrt(np.mean(d * d))),
            "elements_over_max_abs": int((ad > MAX_ABS).sum()),
            "p99_abs": float(np.percentile(ad, 99)),
            "weight_abs_rms": float(np.sqrt(np.mean(x * x))),
            "relative_rms": float(np.sqrt(np.mean(d * d)) / np.sqrt(np.mean(x * x))),
            "argmax_index": list(index),
            "reference_value": float(x[index]),
            "mixed_value": float(y[index]),
            "signed_difference": float(d[index]),
            "max_abs_within": bool(ad.max() < MAX_ABS),
            "rms_within": bool(np.sqrt(np.mean(d * d)) < RMS),
        }
    return out


def losses(path: Path) -> list[float]:
    return [record["loss"] for record in json.loads(path.read_text())["records"]]


def dispatch(path: Path) -> dict:
    report = json.loads(path.read_text())
    return {
        key: report[key]
        for key in (
            "wmma_forward",
            "wmma_weight_grad",
            "wmma_input_grad",
            "wmma_cooperative",
            "wmma_shape_fallbacks",
        )
    }


def main() -> int:
    out: dict = {"contract": {"max_abs": MAX_ABS, "rms": RMS}}

    # Trajectory coherence: the 6/12-step arms must be prefixes of the 29-step arm.
    prefix = {}
    for steps in (6, 12, 29):
        prefix[f"off_{steps}"] = losses(STEPS / f"report-off-{steps}.json")
        prefix[f"mixed_{steps}"] = losses(STEPS / f"report-mixed-{steps}.json")
    out["trajectory_prefix_consistent"] = {
        mode: all(
            prefix[f"{mode}_{n}"][i] == prefix[f"{mode}_29"][i]
            for n in (6, 12)
            for i in range(n)
        )
        for mode in ("off", "mixed")
    }
    out["loss_at_steps"] = {
        mode: {str(n): prefix[f"{mode}_{n}"][-1] for n in (6, 12, 29)}
        for mode in ("off", "mixed")
    }
    out["dispatch"] = {
        str(n): {
            mode: dispatch(STEPS / f"report-{mode}-{n}.json") for mode in ("off", "mixed")
        }
        for n in (6, 12, 29)
    }

    # Drift vs steps.
    out["drift_vs_steps"] = {}
    for steps in (6, 12, 29):
        out["drift_vs_steps"][str(steps)] = compare(
            STEPS / f"adapter-off-{steps}/adapter_model.safetensors",
            STEPS / f"adapter-mixed-{steps}/adapter_model.safetensors",
        )

    # Reproduction of the Phase XIII number from the preserved exports.
    out["reproduction_of_phase_xiii_29"] = {
        "phase_xiii_export_compared_to_phase_xiv_off": None,
        "phase_xiv_29_equals_phase_xiii_worst": None,
    }
    xiii_off = XIII / "tmp/36_lora_off_export-adapter/adapter_model.safetensors"
    xiii_mixed = XIII / "tmp/37_lora_mixed_export-adapter/adapter_model.safetensors"
    if xiii_off.is_file() and xiii_mixed.is_file():
        xiii = compare(xiii_off, xiii_mixed)
        worst = max(xiii.values(), key=lambda row: row["max_abs"])
        out["phase_xiii_preserved_exports"] = xiii
        out["phase_xiii_worst"] = {
            name: row for name, row in xiii.items() if row["max_abs"] == worst["max_abs"]
        }
        xiv_worst = max(
            out["drift_vs_steps"]["29"].items(), key=lambda item: item[1]["max_abs"]
        )
        out["reproduction_of_phase_xiii_29"] = {
            "phase_xiii_worst_max_abs": worst["max_abs"],
            "phase_xiv_worst_max_abs": xiv_worst[1]["max_abs"],
            "absolute_delta": abs(worst["max_abs"] - xiv_worst[1]["max_abs"]),
            "reproduced": abs(worst["max_abs"] - xiv_worst[1]["max_abs"]) < 1e-9,
        }

    # Transient vs growing, per tensor, on the worst tensor by max_abs.
    worst_name = max(
        out["drift_vs_steps"]["29"].items(), key=lambda item: item[1]["max_abs"]
    )[0]
    out["worst_tensor"] = worst_name
    out["worst_tensor_trajectory"] = {
        str(steps): out["drift_vs_steps"][str(steps)][worst_name] for steps in (6, 12, 29)
    }
    out["monotone_growth_worst_tensor"] = all(
        out["drift_vs_steps"]["6"][worst_name]["max_abs"]
        <= out["drift_vs_steps"]["12"][worst_name]["max_abs"]
        <= out["drift_vs_steps"]["29"][worst_name]["max_abs"]
        for _ in (0,)
    )

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
