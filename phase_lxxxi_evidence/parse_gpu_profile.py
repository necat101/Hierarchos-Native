#!/usr/bin/env python3
"""Parse `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` stderr from wmma_bench.

The Vulkan layer brackets every sampled compute dispatch with two timestamp
queries and, per submitted `ComputeBatch`, prints one
`hierarchos_vulkan_kernel_profile_shader` line for each shader that batch
dispatched, carrying that shader's summed GPU time in milliseconds.

This module turns that stream into per-(shape, arm) GPU timings without
changing a byte of the benchmark or the kernel. The only inference is the
shape segmentation, and it is *checked*, never assumed: the stream is split
into as many equal blocks as there are shapes, and every block must carry an
identical shader-sequence signature. A mismatch raises.

Usage as a library:
    from parse_gpu_profile import parse_stream, blocks
"""

from __future__ import annotations

import re

SHAPE_ORDER = (
    "512x768x768",
    "512x2048x768",
    "2048x768x2048",
    "128x8192x768",
    "64x64x64",
    "16x128x32",
)

# `hierarchos_vulkan_kernel_profile_shader shader=X category=Y dispatches=N
#  gpu_ms=Z pct=P samples=S`
_SHADER_RE = re.compile(
    r"^hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=(\S+) "
    r"dispatches=(\d+) gpu_ms=([0-9.]+) pct=([0-9.]+) samples=(\d+)$"
)
_BATCH_RE = re.compile(
    r"^hierarchos_vulkan_kernel_profile device=.* dispatches=(\d+) "
    r"gpu_ms=([0-9.]+) timestamp_period_ns=([0-9.]+)$"
)
_SPAN_RE = re.compile(
    r"^hierarchos_vulkan_kernel_profile_span span_ms=([0-9.]+) "
    r"dispatch_ms=([0-9.]+) unattributed_ms=([0-9.]+) unattributed_pct=([0-9.]+)"
)

COOP_SHADERS = (
    "wmma_gemm_f16_f16acc",
    "wmma_gemm_f16_f16acc_twobank",
    "wmma_gemm_f16_f16acc_twowave",
    "wmma_gemm_f16_f16acc_twowave_p3",
)


class ProfileError(RuntimeError):
    pass


def parse_stream(lines):
    """Return (batches, timestamp_period_ns).

    Each batch is `{"dispatches": N, "gpu_ms": X, "span": dict|None,
    "shaders": [(name, dispatches, gpu_ms)]}`.
    """
    batches = []
    current = None
    period = None
    for raw in lines:
        line = raw.rstrip("\n")
        m = _BATCH_RE.match(line)
        if m:
            if current is not None and current["shaders"]:
                batches.append(current)
            current = {
                "dispatches": int(m.group(1)),
                "gpu_ms": float(m.group(2)),
                "span": None,
                "shaders": [],
            }
            period = float(m.group(3))
            continue
        m = _SPAN_RE.match(line)
        if m:
            if current is None:
                raise ProfileError("span line before any batch line")
            current["span"] = {
                "span_ms": float(m.group(1)),
                "dispatch_ms": float(m.group(2)),
                "unattributed_ms": float(m.group(3)),
                "unattributed_pct": float(m.group(4)),
            }
            continue
        m = _SHADER_RE.match(line)
        if m:
            if current is None:
                raise ProfileError("shader line before any batch line")
            current["shaders"].append(
                (m.group(1), int(m.group(3)), float(m.group(4)))
            )
    if current is not None and current["shaders"]:
        batches.append(current)
    if period is None:
        raise ProfileError("no timestamp_period_ns line found")
    return batches, period


def signature(batch):
    return tuple(sorted(name for name, _, _ in batch["shaders"]))


def blocks(batches, shapes=SHAPE_ORDER):
    """Split into one block per shape and require identical signatures.

    Returns `{shape: [batch, ...]}`. Raises if the stream does not divide
    evenly or if any block's per-batch shader signature differs from the first
    block's: that equality is what makes positional segmentation legitimate.
    """
    count = len(batches)
    if count % len(shapes) != 0:
        raise ProfileError(
            f"{count} batches do not divide into {len(shapes)} shape blocks"
        )
    size = count // len(shapes)
    split = [batches[index * size:(index + 1) * size] for index in range(len(shapes))]
    reference = [signature(batch) for batch in split[0]]
    for shape, block in zip(shapes[1:], split[1:]):
        observed = [signature(batch) for batch in block]
        if observed != reference:
            first = next(
                (i for i, (a, b) in enumerate(zip(reference, observed)) if a != b),
                None,
            )
            raise ProfileError(
                f"block {shape} deviates from the first block at batch {first}: "
                f"{observed[first] if first is not None else 'length'} != "
                f"{reference[first] if first is not None else 'length'}"
            )
    return dict(zip(shapes, split))


def steady_state_gpu_ms(block, shader_name, warmup=3):
    """GPU ms of the steady-state dispatches of one cooperative shader.

    A dispatch batch that contains only that single shader is either a warm-up
    dispatch or a timed iteration; the block's correctness `matmul` batch also
    carries the two FP32->FP16 conversions, so it is excluded by requiring the
    batch to be single-dispatch. The declared number of warm-up dispatches is
    then dropped from the front.
    """
    solo = [
        batch["gpu_ms"]
        for batch in block
        if len(batch["shaders"]) == 1 and batch["shaders"][0][0] == shader_name
    ]
    if len(solo) <= warmup:
        raise ProfileError(
            f"only {len(solo)} solo dispatches of {shader_name}; expected more than {warmup}"
        )
    return solo[warmup:]
