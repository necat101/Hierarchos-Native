#!/usr/bin/env python3
"""Phase XIX post-promotion component attribution.

The profiler prints one block per submitted compute batch: a
`hierarchos_vulkan_kernel_profile device=... gpu_ms=...` header followed by
per-shader lines for that batch. A training step is one batch, so a batch is
selected only when it contains the promoted pass-1 dispatch
(`transformer_attention_backward_query_rdna3_dqunroll`), which keeps startup
batches out of the average.

Everything here is **profiler-instrumented attribution**, not a performance
measurement: sampling perturbs execution, so the headline numbers are the
unprofiled ABBA. This script answers "where does a step go now", which is the
question the phase asks, and groups the per-shader lines into the components the
brief names.

Usage: attribute.py <profile-log> <label> [<profile-log> <label> ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

QUERY = "transformer_attention_backward_query_rdna3_dqunroll"

# Components the brief names. A shader may appear in exactly one component.
COMPONENTS: list[tuple[str, tuple[str, ...]]] = [
    ("query_backward_dqunroll", (QUERY,)),
    (
        "attention_forward_rowpacked",
        (
            "transformer_attention_forward_rowpacked64_cached_rdna3",
            "transformer_attention_forward",
        ),
    ),
    (
        "key_backward_channel4",
        (
            "transformer_attention_backward_key_channel4_rdna3",
            "transformer_attention_backward_key_rdna3",
        ),
    ),
    (
        "dense_projection",
        (
            "linear_forward",
            "linear_forward_lane4",
            "linear_input_grad",
            "linear_input_grad_group4_muladd",
            "linear_weight_grad",
            "parameter_matmul_forward_fp16_packed",
            "parameter_matmul_input_grad_fp16_packed",
            "parameter_matmul_bias_forward_fp16_packed",
            "linear_forward_fp16_packed",
            "linear_bias_forward_fp16_packed",
            "linear_input_grad_fp16_packed",
        ),
    ),
    (
        "wmma_gemm",
        (
            "wmma_gemm_f16_f32acc",
            "wmma_gemm_f16_f16acc",
            "wmma_gemm_f16_f16acc_twobank",
            "wmma_gemm_f16_f16acc_twowave",
            "wmma_gemm_f16_f16acc_twowave_p3",
            "wmma_gemm_f16_f16acc_twowave_p3_c1",
            "gemm_f16_f32_plain",
        ),
    ),
    # `fp32_to_fp16_native` is the WMMA activation cast (src/wmma.rs);
    # `fp32_to_fp16_packed` is the parameter-storage cast that builds the FP16
    # weight mirrors (src/mixed_precision.rs). They are separate identities, so
    # the split the brief asks for is measurable rather than guessed.
    ("fp16_conversion_activations", ("fp32_to_fp16_native",)),
    ("fp16_conversion_weight_mirrors", ("fp32_to_fp16_packed",)),
    (
        "rms_norm",
        ("transformer_rms_norm_forward", "transformer_rms_norm_input_grad"),
    ),
    (
        "cross_entropy",
        ("falcon_h1_cross_entropy_lanes16", "ordered_f32_sum"),
    ),
]

CATEGORY_OF = {name: component for component, names in COMPONENTS for name in names}


def batches(path: Path) -> list[dict]:
    out: list[dict] = []
    current: dict | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "hierarchos_vulkan_kernel_profile device=" in line:
            if current is not None:
                out.append(current)
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            current = {"gpu_ms": float(fields["gpu_ms"]), "shaders": {}}
        elif current is not None and "hierarchos_vulkan_kernel_profile_shader" in line:
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            current["shaders"][fields["shader"]] = {
                "category": fields.get("category", "other"),
                "dispatches": int(fields["dispatches"]),
                "gpu_ms": float(fields["gpu_ms"]),
            }
    if current is not None:
        out.append(current)
    return out


def step_times(path: Path) -> list[float]:
    out: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[phase-xix-step]" not in line:
            continue
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if fields.get("measured") != "true":
            continue
        out.append(float(fields["step_ms"]))
    return out


def summarize(path: Path) -> dict:
    selected = [b for b in batches(path) if b["shaders"].get(QUERY, {}).get("dispatches", 0) > 0]
    if not selected:
        raise SystemExit(f"{path.name}: no batches contain the promoted pass-1 dispatch")

    per_shader: dict[str, dict] = {}
    for batch in selected:
        for name, stats in batch["shaders"].items():
            entry = per_shader.setdefault(
                name, {"dispatches": 0, "gpu_ms": 0.0, "category": stats["category"]}
            )
            entry["dispatches"] += stats["dispatches"]
            entry["gpu_ms"] += stats["gpu_ms"]

    n = len(selected)
    components: dict[str, dict] = {}
    for name, stats in per_shader.items():
        component = CATEGORY_OF.get(name, "other")
        entry = components.setdefault(component, {"ms_per_step": 0.0, "dispatches_per_step": 0.0})
        entry["ms_per_step"] += stats["gpu_ms"] / n
        entry["dispatches_per_step"] += stats["dispatches"] / n
    for entry in components.values():
        entry["ms_per_step"] = round(entry["ms_per_step"], 6)
        entry["dispatches_per_step"] = round(entry["dispatches_per_step"], 4)

    profiled_total = statistics.fmean(batch["gpu_ms"] for batch in selected)
    steps = step_times(path)
    return {
        "log": path.name,
        "step_batches": n,
        "instrumented_step_ms_median": round(statistics.median(steps), 6) if steps else None,
        "instrumented_step_ms_count": len(steps),
        "profiled_gpu_ms_per_step": round(profiled_total, 6),
        "components_ms_per_step": dict(
            sorted(components.items(), key=lambda item: -item[1]["ms_per_step"])
        ),
        "shaders_ms_per_step": {
            name: {
                "ms_per_step": round(stats["gpu_ms"] / n, 6),
                "dispatches_per_step": round(stats["dispatches"] / n, 4),
                "category": stats["category"],
            }
            for name, stats in sorted(per_shader.items(), key=lambda item: -item[1]["gpu_ms"])
        },
        "dqunroll_dispatches_per_step": round(
            per_shader.get(QUERY, {}).get("dispatches", 0) / n, 4
        ),
        "unknown_shader_lines": sum(
            1 for batch in selected if "unknown" in batch["shaders"]
        ),
    }


def main() -> None:
    args = sys.argv[1:]
    if not args or len(args) % 2 != 0:
        raise SystemExit("usage: attribute.py <profile-log> <label> ...")
    result = {}
    for index in range(0, len(args), 2):
        path, label = Path(args[index]), args[index + 1]
        result[label] = summarize(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
