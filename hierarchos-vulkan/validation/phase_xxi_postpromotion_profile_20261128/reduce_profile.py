#!/usr/bin/env python3
"""Phase XXI post-promotion component attribution reducer.

The profiler prints one block per submitted compute batch: a
`hierarchos_vulkan_kernel_profile device=... gpu_ms=...` header followed by
per-shader lines for that batch. A training step is one batch, so a batch is
selected only when it contains the *new production* pass-1 dispatch
(`transformer_attention_backward_query_rdna3_dqunroll_scratchreuse`), which keeps
startup batches out of the average and simultaneously proves which query module
ran.

Everything here is **profiler-instrumented attribution**, not a performance
measurement: stride-1 sampling perturbs execution. The unprofiled step timing in
the same log is reported separately and is not combined with the profiled total.

Usage:
    reduce_profile.py <mode> <log> [<log> ...] [<mode> <log> ...]

Each mode gets per-process summaries and a median across processes.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

QUERY = "transformer_attention_backward_query_rdna3_dqunroll_scratchreuse"
QUERY_FALLBACKS = (
    "transformer_attention_backward_query_rdna3_dqunroll",
    "transformer_attention_backward_query_rdna3",
)

# Same grouping as the Phase XIX reducer so the historical comparison is
# component-for-component. The new query identity replaces the old one and is
# deliberately in its own component, never merged with the Phase XVIII row.
COMPONENTS: list[tuple[str, tuple[str, ...]]] = [
    ("query_backward_scratchreuse", (QUERY,)),
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
    # Kept apart on purpose: `fp32_to_fp16_native` is the WMMA activation cast
    # (src/wmma.rs); `fp32_to_fp16_packed` builds FP16 parameter/weight mirrors
    # (src/mixed_precision.rs). They are different call sites with different
    # lifetimes and must never be summed into one "conversion" number.
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
            current = {
                "device": fields.get("device", "").strip('"'),
                "gpu_ms": float(fields["gpu_ms"]),
                "dispatch_total": int(fields.get("dispatches", "0")),
                "shaders": {},
            }
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
        if "[phase-xii-step]" not in line and "[phase-xix-step]" not in line:
            continue
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if fields.get("measured") != "true":
            continue
        out.append(float(fields["step_ms"]))
    return out


def summarize(path: Path) -> dict:
    selected = [
        b for b in batches(path) if b["shaders"].get(QUERY, {}).get("dispatches", 0) > 0
    ]
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
        entry = components.setdefault(
            component, {"ms_per_step": 0.0, "dispatches_per_step": 0.0}
        )
        entry["ms_per_step"] += stats["gpu_ms"] / n
        entry["dispatches_per_step"] += stats["dispatches"] / n
    for entry in components.values():
        entry["ms_per_step"] = round(entry["ms_per_step"], 6)
        entry["dispatches_per_step"] = round(entry["dispatches_per_step"], 4)

    profiled_total = statistics.fmean(batch["gpu_ms"] for batch in selected)
    steps = step_times(path)

    # Query-route identity evidence: the production module must be the only query
    # module in the selected steps.
    fallback_counts = {
        name: round(per_shader.get(name, {}).get("dispatches", 0) / n, 4)
        for name in QUERY_FALLBACKS
    }
    return {
        "log": path.name,
        "device": selected[0]["device"],
        "step_batches": n,
        "instrumented_step_ms_median": round(statistics.median(steps), 6) if steps else None,
        "instrumented_step_ms_count": len(steps),
        "profiled_gpu_ms_per_step": round(profiled_total, 6),
        "profiled_dispatch_total_per_step": round(
            statistics.fmean(batch["dispatch_total"] for batch in selected), 4
        ),
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
        "scratchreuse_dispatches_per_step": round(
            per_shader.get(QUERY, {}).get("dispatches", 0) / n, 4
        ),
        "query_fallback_dispatches_per_step": fallback_counts,
        "unknown_shader_lines": sum(
            1 for batch in selected if "unknown" in batch["shaders"]
        ),
    }


def median_of(values: list[float]) -> float:
    return round(statistics.median(values), 6)


def aggregate(summaries: list[dict]) -> dict:
    component_names = sorted(
        {name for summary in summaries for name in summary["components_ms_per_step"]}
    )
    components = {}
    for name in component_names:
        components[name] = {
            "median_ms_per_step": median_of(
                [
                    summary["components_ms_per_step"].get(name, {}).get("ms_per_step", 0.0)
                    for summary in summaries
                ]
            ),
            "median_dispatches_per_step": median_of(
                [
                    summary["components_ms_per_step"]
                    .get(name, {})
                    .get("dispatches_per_step", 0.0)
                    for summary in summaries
                ]
            ),
        }
    shader_names = sorted(
        {name for summary in summaries for name in summary["shaders_ms_per_step"]}
    )
    shaders = {
        name: {
            "median_ms_per_step": median_of(
                [
                    summary["shaders_ms_per_step"]
                    .get(name, {})
                    .get("ms_per_step", 0.0)
                    for summary in summaries
                ]
            ),
            "median_dispatches_per_step": median_of(
                [
                    summary["shaders_ms_per_step"]
                    .get(name, {})
                    .get("dispatches_per_step", 0.0)
                    for summary in summaries
                ]
            ),
        }
        for name in shader_names
    }
    return {
        "processes": len(summaries),
        "median_profiled_gpu_ms_per_step": median_of(
            [summary["profiled_gpu_ms_per_step"] for summary in summaries]
        ),
        "median_instrumented_step_ms": median_of(
            [
                summary["instrumented_step_ms_median"]
                for summary in summaries
                if summary["instrumented_step_ms_median"] is not None
            ]
        ),
        "median_profiled_dispatch_total_per_step": median_of(
            [summary["profiled_dispatch_total_per_step"] for summary in summaries]
        ),
        "components": components,
        "shaders": shaders,
    }


def main() -> None:
    args = sys.argv[1:]
    if not args or len(args) < 2:
        raise SystemExit("usage: reduce_profile.py <mode> <log> [<log> ...] ...")
    result: dict[str, dict] = {}
    index = 0
    while index < len(args):
        mode = args[index]
        index += 1
        logs = []
        while index < len(args) and Path(args[index]).suffix == ".log":
            logs.append(Path(args[index]))
            index += 1
        if not logs:
            raise SystemExit(f"mode {mode!r} has no logs")
        summaries = [summarize(path) for path in logs]
        result[mode] = {
            "processes": summaries,
            "median": aggregate(summaries),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
